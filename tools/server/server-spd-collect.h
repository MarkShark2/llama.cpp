#pragma once

// [fork, SPD] training-data collection: assembles tapped per-layer hidden
// states of collect-enabled requests into shard files for offline SPD
// speculation-head training.
//
// Shard file layout (all little-endian):
//   [records ...][footer JSON][u64 footer_len]["SPDSHRD1"]
// Record per sequence, in footer "seqs" order:
//   tokens  : i32  * n_tokens
//   labels  : u8   * n_tokens          (1 = position carries training loss)
//   taps    : f16  * n_tokens * n_embd, one block per tap in footer "taps" order
// Shards are written as shard-XXXXXXXX.tmp and renamed to .spdshard when
// rotated, so readers only ever see complete files.

#include "llama.h"
#include "ggml.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

struct server_spd_seq {
    std::string id;

    int32_t n_tokens = 0; // expected prompt length

    std::vector<llama_token> tokens; // provided at end_seq
    std::vector<uint8_t>     labels; // per-token loss mask

    // per tap: n_tokens * n_embd half-precision rows
    std::vector<std::vector<ggml_fp16_t>> taps;

    std::vector<uint8_t> row_filled; // per-token harvest tracking
    int32_t n_filled = 0;

    bool complete() const {
        return n_filled == n_tokens;
    }
};

class server_spd_collector {
public:
    server_spd_collector(
            std::string dir,
            std::vector<int32_t> taps,
            uint32_t n_embd,
            uint32_t n_layer,
            std::string model_desc,
            int64_t shard_tokens) :
        dir_(std::move(dir)),
        taps_(std::move(taps)),
        n_embd_(n_embd),
        n_layer_(n_layer),
        model_desc_(std::move(model_desc)),
        shard_tokens_(shard_tokens) {
        namespace fs = std::filesystem;

        fs::create_directories(dir_);

        // continue shard numbering after any completed shards; drop stale
        // partials from a previous crashed process
        for (const auto & ent : fs::directory_iterator(dir_)) {
            const auto name = ent.path().filename().string();
            if (name.size() == std::strlen("shard-00000000.spdshard") &&
                name.rfind("shard-", 0) == 0 && name.find(".spdshard") != std::string::npos) {
                next_shard_ = std::max(next_shard_, (uint64_t) std::stoull(name.substr(6, 8)) + 1);
            } else if (name.find(".tmp") != std::string::npos && name.rfind("shard-", 0) == 0) {
                std::error_code ec;
                fs::remove(ent.path(), ec);
            }
        }

        writer_ = std::thread([this]() { writer_loop(); });
    }

    ~server_spd_collector() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        if (writer_.joinable()) {
            writer_.join();
        }
        // keep a partially filled shard: rotate it out as a valid (short) shard
        if (shard_file_ != nullptr) {
            if (shard_seq_count_ > 0) {
                close_shard();
            } else {
                std::fclose(shard_file_);
                std::error_code ec;
                std::filesystem::remove(shard_tmp_path_, ec);
            }
        }
    }

    server_spd_collector(const server_spd_collector &) = delete;
    server_spd_collector & operator=(const server_spd_collector &) = delete;

    const std::vector<int32_t> & taps() const { return taps_; }

    std::shared_ptr<server_spd_seq> begin_seq(
            const std::string & id,
            int32_t n_tokens,
            const std::vector<std::pair<int32_t, int32_t>> & label_ranges) {
        auto seq = std::make_shared<server_spd_seq>();
        seq->id       = id;
        seq->n_tokens = n_tokens;

        if (label_ranges.empty()) {
            seq->labels.assign(n_tokens, 1);
        } else {
            seq->labels.assign(n_tokens, 0);
            for (const auto & [lo, hi] : label_ranges) {
                for (int32_t i = std::max<int32_t>(lo, 0); i < std::min<int32_t>(hi, n_tokens); ++i) {
                    seq->labels[i] = 1;
                }
            }
        }

        seq->taps.resize(taps_.size());
        for (auto & tap : seq->taps) {
            tap.resize((size_t) n_tokens * n_embd_);
        }
        seq->row_filled.assign(n_tokens, 0);

        return seq;
    }

    // convert n_rows fp32 rows starting at src into the sequence's tap buffer
    // at token position pos0 (idempotent on overwrite - decode retries re-run
    // the same rows)
    void harvest(server_spd_seq & seq, size_t tap_idx, const float * src, int32_t pos0, int32_t n_rows) {
        GGML_ASSERT(tap_idx < seq.taps.size());
        GGML_ASSERT(pos0 >= 0 && pos0 + n_rows <= seq.n_tokens);

        ggml_fp16_t * dst = seq.taps[tap_idx].data() + (size_t) pos0 * n_embd_;
        ggml_fp32_to_fp16_row(src, dst, (int64_t) n_rows * n_embd_);

        // count each token once (when its first tap lands)
        if (tap_idx == 0) {
            for (int32_t i = 0; i < n_rows; ++i) {
                if (!seq.row_filled[pos0 + i]) {
                    seq.row_filled[pos0 + i] = 1;
                    seq.n_filled++;
                }
            }
        }
    }

    // hand a fully harvested sequence to the background writer.
    // blocks briefly if the writer is behind. queue depth 1: each queued seq
    // holds n_tokens*n_taps*n_embd fp16 in host RAM (~830 MB at 15k-token
    // packs), which a UMA collection head (BC-250) cannot spare 4x of; the
    // writer outpaces collection anyway, so depth only bounds the worst case.
    void end_seq(std::shared_ptr<server_spd_seq> seq, std::vector<llama_token> tokens) {
        GGML_ASSERT(seq->complete());
        GGML_ASSERT((int32_t) tokens.size() == seq->n_tokens);
        seq->tokens = std::move(tokens);

        std::unique_lock<std::mutex> lock(mutex_);
        cv_room_.wait(lock, [this]() { return queue_.size() < 1 || stop_; });
        if (stop_) {
            return;
        }
        queue_.push_back(std::move(seq));
        cv_.notify_all();
    }

private:
    void writer_loop() {
        for (;;) {
            std::shared_ptr<server_spd_seq> seq;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return !queue_.empty() || stop_; });
                if (queue_.empty()) {
                    return; // stop requested and drained
                }
                seq = std::move(queue_.front());
                queue_.pop_front();
                cv_room_.notify_all();
            }
            write_seq(*seq);
        }
    }

    void open_shard() {
        char name[64];
        std::snprintf(name, sizeof(name), "shard-%08llu.tmp", (unsigned long long) next_shard_);
        shard_tmp_path_   = (std::filesystem::path(dir_) / name).string();
        shard_final_path_ = shard_tmp_path_.substr(0, shard_tmp_path_.size() - 4) + ".spdshard";
        next_shard_++;

        shard_file_ = std::fopen(shard_tmp_path_.c_str(), "wb");
        if (shard_file_ == nullptr) {
            fprintf(stderr, "spd-collect: failed to open shard file %s\n", shard_tmp_path_.c_str());
            return;
        }
        shard_seqs_       = nlohmann::json::array();
        shard_offset_     = 0;
        shard_token_sum_  = 0;
        shard_seq_count_  = 0;
    }

    void close_shard() {
        if (shard_file_ == nullptr) {
            return;
        }

        nlohmann::json footer = {
            { "version",      1              },
            { "model",        model_desc_    },
            { "n_embd",       n_embd_        },
            { "n_layer",      n_layer_       },
            { "taps",         taps_          },
            { "dtype",        "f16"          },
            { "seqs",         shard_seqs_    },
            { "total_tokens", shard_token_sum_ },
        };
        const std::string footer_str = footer.dump();
        const uint64_t    footer_len = footer_str.size();

        std::fwrite(footer_str.data(), 1, footer_str.size(), shard_file_);
        std::fwrite(&footer_len, sizeof(footer_len), 1, shard_file_);
        std::fwrite("SPDSHRD1", 1, 8, shard_file_);
        std::fclose(shard_file_);
        shard_file_ = nullptr;

        std::error_code ec;
        std::filesystem::rename(shard_tmp_path_, shard_final_path_, ec);
        if (ec) {
            fprintf(stderr, "spd-collect: failed to rename %s -> %s: %s\n",
                    shard_tmp_path_.c_str(), shard_final_path_.c_str(), ec.message().c_str());
        } else {
            fprintf(stderr, "spd-collect: wrote %s (%lld tokens, %d seqs)\n",
                    shard_final_path_.c_str(), (long long) shard_token_sum_, shard_seq_count_);
        }
    }

    void write_seq(const server_spd_seq & seq) {
        if (shard_file_ == nullptr) {
            open_shard();
            if (shard_file_ == nullptr) {
                return; // disk problem - drop the sequence, error already logged
            }
        }

        const uint64_t off = shard_offset_;

        std::fwrite(seq.tokens.data(), sizeof(llama_token), seq.tokens.size(), shard_file_);
        std::fwrite(seq.labels.data(), 1, seq.labels.size(), shard_file_);
        for (const auto & tap : seq.taps) {
            std::fwrite(tap.data(), sizeof(ggml_fp16_t), tap.size(), shard_file_);
        }

        shard_offset_ += sizeof(llama_token) * seq.tokens.size()
                       + seq.labels.size()
                       + taps_.size() * sizeof(ggml_fp16_t) * (size_t) seq.n_tokens * n_embd_;

        shard_seqs_.push_back({
            { "id",       seq.id       },
            { "off",      off          },
            { "n_tokens", seq.n_tokens },
        });
        shard_token_sum_ += seq.n_tokens;
        shard_seq_count_++;

        if (shard_token_sum_ >= shard_tokens_) {
            close_shard();
        }
    }

    // config
    std::string          dir_;
    std::vector<int32_t> taps_;
    uint32_t             n_embd_ = 0;
    uint32_t             n_layer_ = 0;
    std::string          model_desc_;
    int64_t              shard_tokens_ = 0;

    // writer thread + queue
    std::thread                                  writer_;
    std::mutex                                   mutex_;
    std::condition_variable                      cv_;
    std::condition_variable                      cv_room_;
    std::deque<std::shared_ptr<server_spd_seq>>  queue_;
    bool                                         stop_ = false;

    // current shard state (writer thread only, plus destructor after join)
    uint64_t       next_shard_      = 0;
    std::FILE *    shard_file_      = nullptr;
    std::string    shard_tmp_path_;
    std::string    shard_final_path_;
    nlohmann::json shard_seqs_;
    uint64_t       shard_offset_    = 0;
    int64_t        shard_token_sum_ = 0;
    int32_t        shard_seq_count_ = 0;
};
