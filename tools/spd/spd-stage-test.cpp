#include "llama.h"
#include "../../src/llama-ext.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t SPD_STAGE_COUNT = 8;
constexpr uint32_t SPD_MAX_STAGE_COUNT = 16;

using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;

void log_errors(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        std::fputs(text, stderr);
    }
}

void usage(const char * argv0) {
    std::fprintf(stderr,
            "usage: %s -m target.gguf [-p prompt | --prompt-file path] [-n greedy_tokens] [-ngl layers]\n"
            "       [--stage-layers 2,2,3,4,3,4,4,4,4,4,4,4,3] [-c context] [-b batch]\n"
            "       [--rpc host:port,...] [--rpc-cache] [--device RPC0,...,CUDA0]\n"
            "       [--device-head CUDA0] [--tensor-split 2,2,3,4,3,4,4,4,4,4,4,4,3,0]\n"
            "       [--flash-attn on|off|auto] [--cache-type-k f16] [--cache-type-v f16] [--no-mmap]\n"
            "       [--atol 0.001] [--rtol 0.001] [--near-full]\n"
            "Without --stage-layers, use the legacy eight ceil-sized stages.\n"
            "Check prompt/one-token logits, seq_cp rollback depths 1 through stage_count - 1,\n"
            "and full sequence checkpoints restored into fresh stage contexts. GLM also checks\n"
            "repeated in-place partial restores, wrong-branch cropping and full re-export at\n"
            "prefixes before, at and after the nearest k-pool boundary. Use a long --prompt-file\n"
            "to exercise sparse top-k selection and long-context pool-prefix caching.\n"
            "--near-full pads extra rollback and checkpoint cases to the configured context limit.\n", argv0);
}

std::vector<std::string> split(const std::string & value) {
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const size_t first = item.find_first_not_of(" \t");
        const size_t last = item.find_last_not_of(" \t");
        result.push_back(first == std::string::npos ? "" : item.substr(first, last - first + 1));
    }
    if (value.empty() || value.back() == ',') {
        result.emplace_back();
    }
    return result;
}

int32_t parse_int(const std::string & value) {
    size_t end = 0;
    const long long result = std::stoll(value, &end);
    if (end != value.size() || result < INT32_MIN || result > INT32_MAX) {
        throw std::invalid_argument("invalid integer: " + value);
    }
    return (int32_t) result;
}

float parse_nonnegative(const std::string & value) {
    size_t end = 0;
    const float result = std::stof(value, &end);
    if (end != value.size() || !std::isfinite(result) || result < 0.0f) {
        throw std::invalid_argument("invalid non-negative number: " + value);
    }
    return result;
}

void register_rpc_servers(const std::string & servers, bool cache) {
    if (servers.empty()) {
        if (cache) {
            throw std::invalid_argument("--rpc-cache requires --rpc");
        }
        return;
    }
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (rpc_reg == nullptr) {
        throw std::invalid_argument("failed to find RPC backend");
    }
    using add_server_fn = ggml_backend_reg_t (*)(const char * endpoint);
    auto add_server = (add_server_fn) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (add_server == nullptr) {
        throw std::invalid_argument("failed to find RPC add-server function");
    }
    for (const std::string & endpoint : split(servers)) {
        if (endpoint.empty()) {
            throw std::invalid_argument("empty RPC endpoint");
        }
        ggml_backend_reg_t server = add_server(endpoint.c_str());
        if (server == nullptr) {
            throw std::runtime_error("failed to register RPC server: " + endpoint);
        }
        ggml_backend_register(server);
    }
    if (cache) {
        using set_cache_fn = void (*)(bool enabled);
        auto set_cache = (set_cache_fn) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_set_client_cache");
        if (set_cache == nullptr) {
            throw std::invalid_argument("failed to find RPC client-cache function");
        }
        set_cache(true);
    }
}

std::vector<ggml_backend_dev_t> parse_devices(const std::string & value) {
    std::vector<ggml_backend_dev_t> result;
    if (!value.empty()) {
        for (const std::string & name : split(value)) {
            ggml_backend_dev_t device = name.empty() ? nullptr : ggml_backend_dev_by_name(name.c_str());
            if (device == nullptr || ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                throw std::invalid_argument("invalid device: " + name);
            }
            result.push_back(device);
        }
        result.push_back(nullptr);
    }
    return result;
}

ggml_type parse_cache_type(const std::string & value) {
    if (value == "f32")  return GGML_TYPE_F32;
    if (value == "f16")  return GGML_TYPE_F16;
    if (value == "bf16") return GGML_TYPE_BF16;
    if (value == "q8_0") return GGML_TYPE_Q8_0;
    throw std::invalid_argument("cache type must be f32, f16, bf16 or q8_0");
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool special) {
    if (text.size() > INT32_MAX) {
        throw std::invalid_argument("text is too long");
    }
    const int32_t size = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, special, true);
    if (size <= 0) {
        throw std::runtime_error("failed to size tokenized text");
    }
    std::vector<llama_token> tokens(size);
    if (llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(), size, special, true) != size) {
        throw std::runtime_error("failed to tokenize text");
    }
    return tokens;
}

context_ptr make_context(llama_model * model, const llama_context_params & params) {
    context_ptr ctx(llama_init_from_model(model, params), llama_free);
    if (!ctx) {
        throw std::runtime_error("failed to create test context");
    }
    return ctx;
}

void decode(llama_context * ctx, llama_token * tokens, float * embeddings, int32_t count, llama_pos pos) {
    std::vector<llama_pos> positions(count);
    for (int32_t i = 0; i < count; ++i) {
        positions[i] = pos + i;
    }
    llama_batch batch = {};
    batch.n_tokens = count;
    batch.token = tokens;
    batch.embd = embeddings;
    batch.pos = positions.data();
    if (llama_decode(ctx, batch) != 0) {
        throw std::runtime_error("decode failed at position " + std::to_string(pos));
    }
}

int32_t argmax(const float * logits, int32_t n_vocab) {
    return (int32_t) (std::max_element(logits, logits + n_vocab) - logits);
}

std::vector<float> get_logits(llama_context * ctx, int32_t n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        throw std::runtime_error("target produced no logits");
    }
    return std::vector<float>(logits, logits + n_vocab);
}

bool compare(const std::string & label, const float * expected, const float * actual, size_t count,
        float atol, float rtol, bool report) {
    double squared_error = 0.0;
    double max_abs_error = 0.0;
    size_t mismatches = 0;
    size_t first_mismatch = count;
    for (size_t i = 0; i < count; ++i) {
        const double error = std::abs((double) expected[i] - actual[i]);
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]) ||
            error > atol + (double) rtol * std::abs(expected[i])) {
            ++mismatches;
            first_mismatch = std::min(first_mismatch, i);
        }
        max_abs_error = std::max(max_abs_error, error);
        squared_error += error * error;
    }
    if (report || mismatches != 0) {
        std::printf("%s: max abs error=%.9g RMSE=%.9g mismatches=%zu/%zu\n",
                label.c_str(), max_abs_error, std::sqrt(squared_error / count), mismatches, count);
    }
    if (mismatches != 0) {
        std::fprintf(stderr, "%s: first mismatch at %zu: expected=%.9g actual=%.9g\n",
                label.c_str(), first_mismatch, expected[first_mismatch], actual[first_mismatch]);
    }
    return mismatches == 0;
}

struct token_result {
    std::vector<float> boundaries;
    std::vector<float> logits;
};

void run_stages(const std::vector<context_ptr> & stages, llama_context * head, uint32_t width,
        llama_token * tokens, int32_t count, llama_pos pos, token_result * result = nullptr) {
    std::vector<float> hidden((size_t) count * width);
    if (result != nullptr) {
        result->boundaries.resize(stages.size() * width);
    }
    for (size_t stage = 0; stage < stages.size(); ++stage) {
        decode(stages[stage].get(), stage == 0 ? tokens : nullptr,
                stage == 0 ? nullptr : hidden.data(), count, pos);
        const float * output = llama_get_embeddings(stages[stage].get());
        if (output == nullptr) {
            throw std::runtime_error("stage " + std::to_string(stage) + " produced no boundary");
        }
        for (int32_t row = 1; row < count; ++row) {
            if (llama_get_embeddings_ith(stages[stage].get(), row) != output + (size_t) row * width) {
                throw std::runtime_error("wrong boundary row stride in stage " + std::to_string(stage));
            }
        }
        if (llama_get_embeddings_ith(stages[stage].get(), -1) != output + (size_t) (count - 1) * width) {
            throw std::runtime_error("wrong final boundary row in stage " + std::to_string(stage));
        }
        std::copy(output, output + hidden.size(), hidden.begin());
        if (result != nullptr) {
            std::copy(hidden.end() - width, hidden.end(), result->boundaries.begin() + stage * width);
        }
    }
    if (head != nullptr) {
        decode(head, nullptr, hidden.data() + (size_t) (count - 1) * width, 1, pos + count - 1);
    }
}

void prefill_stages(const std::vector<context_ptr> & stages, llama_context * head, uint32_t width,
        std::vector<llama_token> & tokens, uint32_t batch) {
    for (size_t begin = 0; begin < tokens.size(); begin += batch) {
        const int32_t count = (int32_t) std::min<size_t>(batch, tokens.size() - begin);
        run_stages(stages, head, width, tokens.data() + begin, count, (llama_pos) begin);
    }
}

void clear_stages(const std::vector<context_ptr> & stages) {
    for (const auto & stage : stages) {
        llama_memory_clear(llama_get_memory(stage.get()), true);
    }
}

bool check_rollback(const std::vector<context_ptr> & stages, llama_context * head, uint32_t width,
        const llama_vocab * vocab, std::vector<llama_token> & prompt, uint32_t batch, float atol, float rtol) {
    const uint32_t max_depth = (uint32_t) stages.size() - 1;
    const uint32_t speculative_count = 2 * max_depth;
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::string text = " The researcher opened the notebook and carefully recorded the observations from the garden. "
                       "The morning was clear, and the birds gathered beside the old stone wall. ";
    std::vector<llama_token> speculative = tokenize(vocab, text, false);
    while (speculative.size() < speculative_count) {
        text += text;
        speculative = tokenize(vocab, text, false);
    }
    std::vector<llama_token> replacement = tokenize(vocab, " However, the evening brought a different result.", false);
    if (replacement.size() < 2) {
        throw std::runtime_error("rollback text must contain at least two replacement tokens");
    }

    auto restore_prefix = [&](llama_pos pos) {
        const llama_seq_id alias = (llama_seq_id) (pos % max_depth) + 1;
        for (size_t stage = 0; stage < stages.size(); ++stage) {
            llama_memory_t memory = llama_get_memory(stages[stage].get());
            if (llama_memory_seq_pos_max(memory, alias) != pos - 1 ||
                !llama_memory_seq_rm(memory, 0, -1, -1)) {
                throw std::runtime_error("missing rollback alias before position " + std::to_string(pos) +
                        ", stage " + std::to_string(stage));
            }
            llama_memory_seq_cp(memory, alias, 0, -1, -1);
            for (llama_seq_id stale = 1; stale <= (llama_seq_id) max_depth; ++stale) {
                if (!llama_memory_seq_rm(memory, stale, -1, -1) || llama_memory_seq_pos_max(memory, stale) != -1) {
                    throw std::runtime_error("failed to retire rollback aliases in stage " + std::to_string(stage));
                }
            }
            if (llama_memory_seq_pos_max(memory, 0) != pos - 1) {
                throw std::runtime_error("wrong restored position in stage " + std::to_string(stage));
            }
        }
    };
    std::vector<std::array<token_result, 2>> restored(max_depth);
    const llama_pos original_tail = (llama_pos) prompt.size() + speculative_count - 1;
    for (uint32_t depth = 1; depth <= max_depth; ++depth) {
        clear_stages(stages);
        prefill_stages(stages, nullptr, width, prompt, batch);
        // Two turns through the ring also exercise alias retirement before reuse.
        for (uint32_t step = 0; step < speculative_count; ++step) {
            const llama_pos pos = (llama_pos) prompt.size() + step;
            const llama_seq_id alias = (llama_seq_id) (pos % max_depth) + 1;
            for (const auto & stage : stages) {
                llama_memory_t memory = llama_get_memory(stage.get());
                if (!llama_memory_seq_rm(memory, alias, -1, -1)) {
                    throw std::runtime_error("failed to retire rollback ring alias");
                }
                llama_memory_seq_cp(memory, 0, alias, -1, -1);
            }
            run_stages(stages, nullptr, width, &speculative[step], 1, pos);
        }
        for (const auto & stage : stages) {
            if (llama_memory_seq_pos_max(llama_get_memory(stage.get()), 0) != original_tail) {
                throw std::runtime_error("wrong live tail before rollback depth " + std::to_string(depth));
            }
        }
        const llama_pos pos = original_tail + 1 - depth;
        restore_prefix(pos);
        for (uint32_t step = 0; step < 2; ++step) {
            token_result & result = restored[depth - 1][step];
            run_stages(stages, head, width, &replacement[step], 1, pos + step, &result);
            result.logits = get_logits(head, n_vocab);
        }
    }

    bool matches = true;
    for (uint32_t depth = 1; depth <= max_depth; ++depth) {
        clear_stages(stages);
        prefill_stages(stages, nullptr, width, prompt, batch);
        const uint32_t accepted = speculative_count - depth;
        // Replay the same prompt and one-token calls, without any copied state.
        for (uint32_t step = 0; step < accepted; ++step) {
            run_stages(stages, nullptr, width, &speculative[step], 1, (llama_pos) prompt.size() + step);
        }
        bool depth_matches = true;
        for (uint32_t step = 0; step < 2; ++step) {
            token_result clean;
            run_stages(stages, head, width, &replacement[step], 1,
                    (llama_pos) prompt.size() + accepted + step, &clean);
            clean.logits = get_logits(head, n_vocab);
            const token_result & actual = restored[depth - 1][step];
            const std::string label = "rollback depth " + std::to_string(depth) + " token " + std::to_string(step);
            for (size_t stage = 0; stage < stages.size(); ++stage) {
                depth_matches &= compare(label + " stage " + std::to_string(stage),
                        clean.boundaries.data() + stage * width, actual.boundaries.data() + stage * width,
                        width, atol, rtol, false);
            }
            depth_matches &= compare(label + " logits", clean.logits.data(), actual.logits.data(), n_vocab, atol, rtol, false);
            if (argmax(clean.logits.data(), n_vocab) != argmax(actual.logits.data(), n_vocab)) {
                std::fprintf(stderr, "%s: greedy token mismatch\n", label.c_str());
                depth_matches = false;
            }
        }
        std::printf("seq_cp rollback depth %u: %s (all stage streams and two one-token head outputs)\n",
                depth, depth_matches ? "PASS" : "FAIL");
        matches &= depth_matches;
    }
    return matches;
}

bool check_checkpoints(llama_model * model, const std::vector<llama_context_params> & stage_params,
        const llama_context_params & head_params, std::vector<context_ptr> & stages, context_ptr & head,
        uint32_t width, std::vector<llama_token> & tokens, const std::vector<size_t> & prefix_sizes,
        uint32_t batch, float atol, float rtol) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    bool matches = true;
    for (const size_t prefix_size : prefix_sizes) {
        const llama_pos pos = (llama_pos) prefix_size;
        std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + prefix_size);
        std::array<token_result, 2> expected;
        clear_stages(stages);
        prefill_stages(stages, nullptr, width, prefix, batch);
        for (size_t step = 0; step < expected.size(); ++step) {
            run_stages(stages, head.get(), width, &tokens[prefix_size + step], 1, pos + step, &expected[step]);
            expected[step].logits = get_logits(head.get(), n_vocab);
        }

        // flags=0 must carry attention, recurrent and indexer data without live source contexts.
        std::array<std::vector<std::vector<uint8_t>>, 2> snapshots;
        auto capture = [&](llama_seq_id source) {
            auto & blobs = snapshots[source];
            blobs.resize(stages.size());
            for (size_t stage = 0; stage < stages.size(); ++stage) {
                llama_context * ctx = stages[stage].get();
                if (llama_memory_seq_pos_max(llama_get_memory(ctx), source) != pos - 1) {
                    throw std::runtime_error("wrong checkpoint source position in stage " + std::to_string(stage));
                }
                const size_t size = llama_state_seq_get_size_ext(ctx, source, LLAMA_STATE_SEQ_FLAGS_NONE);
                if (size == 0) {
                    throw std::runtime_error("empty checkpoint in stage " + std::to_string(stage));
                }
                blobs[stage].resize(size);
                if (llama_state_seq_get_data_ext(ctx, blobs[stage].data(), size, source,
                            LLAMA_STATE_SEQ_FLAGS_NONE) != size) {
                    throw std::runtime_error("failed to capture checkpoint in stage " + std::to_string(stage));
                }
            }
        };

        clear_stages(stages);
        prefill_stages(stages, nullptr, width, prefix, batch);
        capture(0);
        for (const auto & stage : stages) {
            llama_memory_seq_cp(llama_get_memory(stage.get()), 0, 1, -1, -1);
        }
        // Decode materializes pending copies before alias serialization.
        run_stages(stages, nullptr, width, &tokens[prefix_size], 1, pos);
        capture(1);
        run_stages(stages, nullptr, width, &tokens[prefix_size + 1], 1, pos + 1);

        for (llama_seq_id source = 0; source < 2; ++source) {
            stages.clear();
            head.reset();
            for (const auto & params : stage_params) {
                stages.push_back(make_context(model, params));
            }
            head = make_context(model, head_params);
            const auto & blobs = snapshots[source];
            for (size_t stage = 0; stage < stages.size(); ++stage) {
                llama_context * ctx = stages[stage].get();
                if (llama_state_seq_set_data_ext(ctx, blobs[stage].data(), blobs[stage].size(), 0,
                            LLAMA_STATE_SEQ_FLAGS_NONE) != blobs[stage].size()) {
                    throw std::runtime_error("failed to restore checkpoint in stage " + std::to_string(stage));
                }
                llama_memory_t memory = llama_get_memory(ctx);
                if (llama_memory_seq_pos_max(memory, 0) != pos - 1) {
                    throw std::runtime_error("wrong checkpoint destination position in stage " + std::to_string(stage));
                }
                for (llama_seq_id alias = 1; alias < (llama_seq_id) stages.size(); ++alias) {
                    if (llama_memory_seq_pos_max(memory, alias) != -1) {
                        throw std::runtime_error("checkpoint restored an unrelated alias in stage " + std::to_string(stage));
                    }
                }
            }

            bool restored_matches = true;
            const std::string label = "checkpoint prefix " + std::to_string(pos) + " source " + std::to_string(source);
            for (size_t step = 0; step < expected.size(); ++step) {
                token_result actual;
                run_stages(stages, head.get(), width, &tokens[prefix_size + step], 1, pos + step, &actual);
                actual.logits = get_logits(head.get(), n_vocab);
                const std::string token_label = label + " token " + std::to_string(step);
                for (size_t stage = 0; stage < stages.size(); ++stage) {
                    restored_matches &= compare(token_label + " stage " + std::to_string(stage),
                            expected[step].boundaries.data() + stage * width, actual.boundaries.data() + stage * width,
                            width, atol, rtol, false);
                }
                restored_matches &= compare(token_label + " logits", expected[step].logits.data(), actual.logits.data(),
                        n_vocab, atol, rtol, false);
                if (argmax(expected[step].logits.data(), n_vocab) != argmax(actual.logits.data(), n_vocab)) {
                    std::fprintf(stderr, "%s: greedy token mismatch\n", token_label.c_str());
                    restored_matches = false;
                }
            }
            std::printf("%s -> fresh seq 0: %s (flags=0, all stage boundaries and two head outputs)\n",
                    label.c_str(), restored_matches ? "PASS" : "FAIL");
            matches &= restored_matches;
        }
    }
    return matches;
}

bool check_partial_checkpoints(llama_model * model, const std::vector<llama_context_params> & stage_params,
        const llama_context_params & head_params, std::vector<context_ptr> & stages, context_ptr & head,
        uint32_t width, std::vector<llama_token> & tokens, const std::vector<size_t> & prefix_sizes,
        uint32_t batch, float atol, float rtol) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const std::vector<llama_token> branch = tokenize(vocab,
            " Meanwhile, a different experiment began in the neighboring laboratory.", false);
    std::vector<size_t> partial_sizes(stages.size());
    bool matches = true;
    for (const size_t prefix_size : prefix_sizes) {
        const llama_pos pos = (llama_pos) prefix_size;
        const std::string label = "partial checkpoint prefix " + std::to_string(pos);
        std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + prefix_size);
        std::vector<std::vector<uint8_t>> partial(stages.size());
        std::vector<std::vector<uint8_t>> full(stages.size());
        std::vector<size_t> full_sizes(stages.size());
        std::array<token_result, 2> expected;
        clear_stages(stages);
        prefill_stages(stages, nullptr, width, prefix, batch);
        for (size_t stage = 0; stage < stages.size(); ++stage) {
            llama_context * ctx = stages[stage].get();
            full_sizes[stage] = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
            const size_t size = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (size == 0 || size >= full_sizes[stage]) {
                throw std::runtime_error(label + ": partial checkpoint did not omit attention/indexer state");
            }
            if (partial_sizes[stage] != 0 && size != partial_sizes[stage]) {
                throw std::runtime_error(label + ": partial checkpoint size grew with the resident prefix");
            }
            partial_sizes[stage] = size;
            partial[stage].resize(size);
            if (llama_state_seq_get_data_ext(ctx, partial[stage].data(), size, 0,
                        LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != size) {
                throw std::runtime_error(label + ": partial capture failed");
            }
        }
        for (size_t step = 0; step < expected.size(); ++step) {
            run_stages(stages, head.get(), width, &tokens[prefix_size + step], 1, pos + step, &expected[step]);
            expected[step].logits = get_logits(head.get(), n_vocab);
        }

        auto restore_partial = [&] {
            for (size_t stage = 0; stage < stages.size(); ++stage) {
                llama_context * ctx = stages[stage].get();
                llama_memory_t memory = llama_get_memory(ctx);
                for (llama_seq_id alias = 1; alias < (llama_seq_id) stages.size(); ++alias) {
                    if (!llama_memory_seq_rm(memory, alias, -1, -1) || llama_memory_seq_pos_max(memory, alias) != -1) {
                        throw std::runtime_error(label + ": alias retirement failed");
                    }
                }
                if (llama_state_seq_set_data_ext(ctx, partial[stage].data(), partial[stage].size(), 0,
                            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != partial[stage].size() ||
                    llama_memory_seq_pos_min(memory, 0) != pos - 1 ||
                    llama_memory_seq_pos_max(memory, 0) != pos - 1 || !llama_memory_seq_rm(memory, 0, pos, -1)) {
                    throw std::runtime_error(label + ": in-place restore/crop failed");
                }
                // The recurrent frontier alone can hide uncropped attention and indexer rows.
                if (llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE) != full_sizes[stage]) {
                    throw std::runtime_error(label + ": full export retained wrong-branch rows");
                }
            }
        };
        auto check_continuation = [&](const std::string & mode) {
            bool restored_matches = true;
            for (size_t step = 0; step < expected.size(); ++step) {
                token_result actual;
                run_stages(stages, head.get(), width, &tokens[prefix_size + step], 1, pos + step, &actual);
                actual.logits = get_logits(head.get(), n_vocab);
                const std::string token_label = label + " " + mode + " token " + std::to_string(step);
                for (size_t stage = 0; stage < stages.size(); ++stage) {
                    restored_matches &= compare(token_label + " stage " + std::to_string(stage),
                            expected[step].boundaries.data() + stage * width, actual.boundaries.data() + stage * width,
                            width, atol, rtol, false);
                }
                restored_matches &= compare(token_label + " logits", expected[step].logits.data(), actual.logits.data(),
                        n_vocab, atol, rtol, false);
                if (argmax(expected[step].logits.data(), n_vocab) != argmax(actual.logits.data(), n_vocab)) {
                    std::fprintf(stderr, "%s: greedy token mismatch\n", token_label.c_str());
                    restored_matches = false;
                }
            }
            std::printf("%s %s: %s (all stage boundaries and two head outputs)\n",
                    label.c_str(), mode.c_str(), restored_matches ? "PASS" : "FAIL");
            return restored_matches;
        };

        const auto wrong = std::find_if(branch.begin(), branch.end(),
                [&](llama_token token) { return token != tokens[prefix_size]; });
        if (wrong == branch.end()) {
            throw std::runtime_error("partial checkpoint needs a distinct wrong-branch token");
        }
        for (int repeat = 0; repeat < 2; ++repeat) {
            restore_partial();
            for (size_t step = 0; step < expected.size(); ++step) {
                const llama_seq_id alias = (llama_seq_id) (step % (stages.size() - 1)) + 1;
                for (const auto & stage : stages) {
                    llama_memory_t memory = llama_get_memory(stage.get());
                    if (!llama_memory_seq_rm(memory, alias, -1, -1)) {
                        throw std::runtime_error(label + ": wrong-branch alias replacement failed");
                    }
                    llama_memory_seq_cp(memory, 0, alias, -1, -1);
                }
                llama_token token = step == 0 ? *wrong : branch.back();
                run_stages(stages, nullptr, width, &token, 1, pos + step);
            }
            restore_partial();
            if (repeat == 1) {
                for (size_t stage = 0; stage < stages.size(); ++stage) {
                    full[stage].resize(full_sizes[stage]);
                    if (llama_state_seq_get_data_ext(stages[stage].get(), full[stage].data(), full[stage].size(), 0,
                                LLAMA_STATE_SEQ_FLAGS_NONE) != full[stage].size()) {
                        throw std::runtime_error(label + ": full re-export failed");
                    }
                }
            }
            matches &= check_continuation("in-place restore " + std::to_string(repeat + 1));
        }

        stages.clear();
        head.reset();
        for (const auto & params : stage_params) {
            stages.push_back(make_context(model, params));
        }
        head = make_context(model, head_params);
        for (size_t stage = 0; stage < stages.size(); ++stage) {
            if (llama_state_seq_set_data_ext(stages[stage].get(), full[stage].data(), full[stage].size(), 0,
                        LLAMA_STATE_SEQ_FLAGS_NONE) != full[stage].size() ||
                llama_memory_seq_pos_min(llama_get_memory(stages[stage].get()), 0) != pos - 1 ||
                llama_memory_seq_pos_max(llama_get_memory(stages[stage].get()), 0) != pos - 1) {
                throw std::runtime_error(label + ": fresh full re-export restore failed");
            }
        }
        matches &= check_continuation("full re-export -> fresh context");
    }
    return matches;
}

int run(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "The quick brown fox jumps over the lazy dog.";
    std::string prompt_file;
    std::string rpc_servers;
    std::string device_names;
    std::string head_device_name;
    std::string tensor_split_arg;
    std::vector<uint32_t> stage_layers;
    int32_t n_gpu_layers = 99;
    int32_t n_greedy = 8;
    int32_t requested_ctx = 0;
    int32_t requested_batch = 0;
    bool rpc_cache = false;
    bool use_mmap = true;
    bool near_full = false;
    float atol = 1e-3f;
    float rtol = 1e-3f;
    llama_context_params cparams = llama_context_default_params();

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        }
        if (arg == "--rpc-cache") {
            rpc_cache = true;
            continue;
        }
        if (arg == "--no-mmap") {
            use_mmap = false;
            continue;
        }
        if (arg == "--near-full") {
            near_full = true;
            continue;
        }
        if (++i >= argc) {
            throw std::invalid_argument("missing value for " + arg);
        }
        const std::string value = argv[i];
        if (arg == "-m" || arg == "--model") {
            model_path = value;
        } else if (arg == "-p" || arg == "--prompt") {
            prompt = value;
        } else if (arg == "--prompt-file") {
            prompt_file = value;
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            n_gpu_layers = parse_int(value);
        } else if (arg == "-n" || arg == "--n-predict") {
            n_greedy = parse_int(value);
        } else if (arg == "-c" || arg == "--ctx-size") {
            requested_ctx = parse_int(value);
            if (requested_ctx <= 0) throw std::invalid_argument("context size must be positive");
        } else if (arg == "-b" || arg == "--batch-size") {
            requested_batch = parse_int(value);
            if (requested_batch <= 0) throw std::invalid_argument("batch size must be positive");
        } else if (arg == "--stage-layers") {
            stage_layers.clear();
            for (const std::string & item : split(value)) {
                const int32_t count = parse_int(item);
                if (count <= 0) throw std::invalid_argument("stage layer counts must be positive");
                stage_layers.push_back((uint32_t) count);
            }
            if (stage_layers.size() < 2 || stage_layers.size() > SPD_MAX_STAGE_COUNT) {
                throw std::invalid_argument("stage count must be between 2 and 16");
            }
        } else if (arg == "--rpc") {
            rpc_servers = value;
        } else if (arg == "--device") {
            device_names = value;
        } else if (arg == "--device-head") {
            head_device_name = value;
        } else if (arg == "--tensor-split") {
            tensor_split_arg = value;
        } else if (arg == "--flash-attn") {
            if (value == "on") cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            else if (value == "off") cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            else if (value == "auto") cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
            else throw std::invalid_argument("flash attention must be on, off or auto");
        } else if (arg == "--cache-type-k") {
            cparams.type_k = parse_cache_type(value);
        } else if (arg == "--cache-type-v") {
            cparams.type_v = parse_cache_type(value);
        } else if (arg == "--atol") {
            atol = parse_nonnegative(value);
        } else if (arg == "--rtol") {
            rtol = parse_nonnegative(value);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (model_path.empty() || n_greedy <= 0) {
        throw std::invalid_argument("a model and positive greedy token count are required");
    }
    if (!prompt_file.empty()) {
        std::ifstream input(prompt_file, std::ios::binary);
        if (!input) throw std::invalid_argument("failed to open prompt file: " + prompt_file);
        prompt.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }

    ggml_backend_load_all();
    llama_log_set(log_errors, nullptr);
    register_rpc_servers(rpc_servers, rpc_cache);
    std::vector<ggml_backend_dev_t> devices = parse_devices(device_names);
    std::vector<ggml_backend_dev_t> head_devices = parse_devices(head_device_name);
    if (!head_devices.empty() && head_devices.size() != 2) {
        throw std::invalid_argument("--device-head requires exactly one device");
    }
    std::vector<float> tensor_split(llama_max_devices(), 0.0f);
    if (!tensor_split_arg.empty()) {
        const auto values = split(tensor_split_arg);
        if (devices.empty() || values.size() + 1 != devices.size() || values.size() > tensor_split.size()) {
            throw std::invalid_argument("--tensor-split requires one entry per explicit --device");
        }
        for (size_t i = 0; i < values.size(); ++i) {
            tensor_split[i] = parse_nonnegative(values[i]);
        }
        if (*std::max_element(tensor_split.begin(), tensor_split.end()) == 0.0f) {
            throw std::invalid_argument("tensor split must contain a positive entry");
        }
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    mparams.load_mode = use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
    mparams.devices = devices.empty() ? nullptr : devices.data();
    mparams.tensor_split = tensor_split_arg.empty() ? nullptr : tensor_split.data();
    mparams.mtp_dev = head_devices.empty() ? nullptr : head_devices.front();
    model_ptr model(llama_model_load_from_file(model_path.c_str(), mparams), llama_model_free);
    if (!model) throw std::runtime_error("failed to load target model");

    const uint32_t stage_count = stage_layers.empty() ? SPD_STAGE_COUNT : (uint32_t) stage_layers.size();
    const int32_t n_layers = llama_model_n_layer(model.get());
    if (!stage_layers.empty()) {
        uint64_t total = 0;
        for (uint32_t count : stage_layers) total += count;
        if (total != (uint64_t) n_layers) {
            throw std::invalid_argument("stage layer counts must sum to the target trunk layer count " + std::to_string(n_layers));
        }
    } else if (n_layers <= 0 || ((n_layers + stage_count - 1) / stage_count) * (stage_count - 1) >= (uint32_t) n_layers) {
        throw std::invalid_argument("legacy partition leaves an empty stage; provide --stage-layers");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    std::vector<llama_token> tokens = tokenize(vocab, prompt, true);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const uint32_t width = llama_model_n_embd_spd_boundary(model.get());
    if (width == 0 || n_vocab <= 0) throw std::runtime_error("invalid target dimensions");
    std::vector<size_t> checkpoint_sizes = { tokens.size() };
    bool glm_checkpoint = false;
    char metadata[32] = {};
    if (llama_model_meta_val_str(model.get(), "glm5-next.attention.indexer.kpool", metadata, sizeof(metadata)) > 0) {
        glm_checkpoint = true;
        const int32_t kpool = parse_int(metadata);
        if (kpool <= 1) throw std::runtime_error("invalid GLM k-pool size");
        const size_t boundary = (tokens.size() + kpool - 1) / kpool * kpool;
        checkpoint_sizes.insert(checkpoint_sizes.end(), { boundary - 1, boundary, boundary + 1 });
        std::sort(checkpoint_sizes.begin(), checkpoint_sizes.end());
        checkpoint_sizes.erase(std::unique(checkpoint_sizes.begin(), checkpoint_sizes.end()), checkpoint_sizes.end());
        std::printf("checkpoint k-pool boundary: %zu (pool=%d), prefixes:", boundary, kpool);
        for (const size_t size : checkpoint_sizes) std::printf(" %zu", size);
        std::printf("\n");
        if (llama_model_meta_val_str(model.get(), "glm5-next.attention.indexer.top_k", metadata, sizeof(metadata)) > 0 &&
            (checkpoint_sizes.back() + 2) / kpool <= (size_t) parse_int(metadata) / kpool) {
            std::fprintf(stderr, "[spd-stage-test] prompt is below sparse top-k selection; repeat with a longer --prompt-file\n");
        }
    }
    std::vector<llama_token> checkpoint_tokens = tokens;
    const std::vector<llama_token> extra = tokenize(vocab,
            " The researcher compared the new observations with the notes from the previous morning.", false);
    const int32_t n_checks = std::max(2, n_greedy);
    const uint64_t needed_ctx = std::max<uint64_t>(checkpoint_sizes.back() + 2,
            tokens.size() + std::max<uint64_t>(n_checks, 2 * (stage_count - 1) + 2)) + 8;
    const uint64_t n_ctx = requested_ctx > 0 ? (uint64_t) requested_ctx : std::max<uint64_t>(256, needed_ctx);
    const uint64_t max_ctx = glm_checkpoint ? INT32_MAX - (stage_count - 1) : INT32_MAX / stage_count;
    if (n_ctx < needed_ctx || n_ctx > max_ctx) {
        throw std::invalid_argument("context is too small for the test or too large for rollback aliases");
    }
    if (near_full) {
        checkpoint_sizes.push_back((size_t) n_ctx - 2);
    }
    while (checkpoint_tokens.size() < checkpoint_sizes.back() + 2) {
        checkpoint_tokens.insert(checkpoint_tokens.end(), extra.begin(), extra.end());
    }
    const uint32_t n_batch = requested_batch > 0 ? (uint32_t) requested_batch : std::max<uint32_t>(32, (uint32_t) tokens.size());
    cparams.n_ctx = (uint32_t) n_ctx;
    cparams.n_batch = n_batch;
    cparams.n_ubatch = n_batch;
    cparams.n_seq_max = 1;
    cparams.n_rs_seq = 0;
    cparams.kv_unified = false;
    cparams.no_perf = false;
    std::printf("stages=%u boundary_width=%u rollback_depths=1..%u unified_kv=%s atol=%.9g rtol=%.9g\n",
            stage_count, width, stage_count - 1, glm_checkpoint ? "true" : "false", atol, rtol);

    std::vector<std::vector<float>> expected_logits;
    std::vector<llama_token> expected_tokens;
    {
        std::fprintf(stderr, "[spd-stage-test] running full-target baseline\n");
        context_ptr ctx = make_context(model.get(), cparams);
        for (size_t begin = 0; begin < tokens.size(); begin += n_batch) {
            decode(ctx.get(), tokens.data() + begin, nullptr,
                    (int32_t) std::min<size_t>(n_batch, tokens.size() - begin), (llama_pos) begin);
        }
        for (int32_t step = 0; step < n_checks; ++step) {
            expected_logits.push_back(get_logits(ctx.get(), n_vocab));
            expected_tokens.push_back(argmax(expected_logits.back().data(), n_vocab));
            if (step + 1 < n_checks) {
                decode(ctx.get(), &expected_tokens.back(), nullptr, 1, (llama_pos) tokens.size() + step);
            }
        }
    }

    std::vector<context_ptr> stages;
    std::vector<llama_context_params> stage_params;
    uint32_t layer_start = 0;
    for (uint32_t stage = 0; stage < stage_count; ++stage) {
        llama_context_params sp = cparams;
        sp.ctx_type = LLAMA_CONTEXT_TYPE_SPD_STAGE;
        sp.spd_stage = stage;
        sp.spd_stage_count = stage_count;
        if (!stage_layers.empty()) {
            sp.spd_layer_start = layer_start;
            layer_start += stage_layers[stage];
            sp.spd_layer_end = layer_start;
        }
        sp.n_seq_max = stage_count;
        if (glm_checkpoint) {
            sp.kv_unified = true;
            sp.n_ctx += stage_count - 1;
        } else {
            sp.n_ctx *= stage_count;
        }
        sp.embeddings = true;
        stage_params.push_back(sp);
        std::fprintf(stderr, "[spd-stage-test] creating stage %u\n", stage);
        stages.push_back(make_context(model.get(), sp));
    }
    llama_context_params hp = cparams;
    hp.ctx_type = LLAMA_CONTEXT_TYPE_SPD_HEAD;
    hp.n_batch = hp.n_ubatch = 1;
    context_ptr head = make_context(model.get(), hp);
    prefill_stages(stages, head.get(), width, tokens, n_batch);
    bool matches = true;
    int32_t greedy_matched = 0;
    for (int32_t step = 0; step < n_checks; ++step) {
        const std::vector<float> actual_logits = get_logits(head.get(), n_vocab);
        matches &= compare(step == 0 ? "prompt logits" : "one-token logits " + std::to_string(step),
                expected_logits[step].data(), actual_logits.data(), n_vocab, atol, rtol, true);
        if (argmax(actual_logits.data(), n_vocab) != expected_tokens[step]) {
            std::fprintf(stderr, "greedy mismatch at step %d: baseline=%d staged=%d\n",
                    step, expected_tokens[step], argmax(actual_logits.data(), n_vocab));
            matches = false;
        } else if (step < n_greedy) {
            ++greedy_matched;
        }
        if (step + 1 < n_checks) {
            // Keep both arms on the same prefix after any mismatch.
            run_stages(stages, head.get(), width, &expected_tokens[step], 1, (llama_pos) tokens.size() + step);
        }
    }
    std::printf("greedy tokens: %d/%d matched\n", greedy_matched, n_greedy);
    std::vector<llama_token> rollback_prompt = tokens;
    if (near_full) {
        const size_t size = (size_t) n_ctx - 2 * (stage_count - 1) - 2;
        while (rollback_prompt.size() < size) {
            rollback_prompt.insert(rollback_prompt.end(), extra.begin(), extra.end());
        }
        rollback_prompt.resize(size);
        std::printf("near-full prefixes: rollback=%zu checkpoint=%zu user_context=%u\n",
                size, checkpoint_sizes.back(), (uint32_t) n_ctx);
    }
    matches &= check_rollback(stages, head.get(), width, vocab, rollback_prompt, n_batch, atol, rtol);
    matches &= check_checkpoints(model.get(), stage_params, hp, stages, head, width,
            checkpoint_tokens, checkpoint_sizes, n_batch, atol, rtol);
    if (glm_checkpoint) {
        matches &= check_partial_checkpoints(model.get(), stage_params, hp, stages, head, width,
                checkpoint_tokens, checkpoint_sizes, n_batch, atol, rtol);
    }
    if (!matches) {
        std::fprintf(stderr, "FAIL: SPD stage, head, rollback or checkpoint parity check failed\n");
        return 2;
    }
    std::printf("PASS: SPD stage/head parity, seq_cp rollback and fresh-context checkpoints%s (stages=%u, depths=1..%u)\n",
            glm_checkpoint ? ", compact in-place checkpoint restore/re-export" : "", stage_count, stage_count - 1);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "spd-stage-test: %s\n", error.what());
        return 1;
    }
}
