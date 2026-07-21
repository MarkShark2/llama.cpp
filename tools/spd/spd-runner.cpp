#include "llama.h"
#include "spd-pipeline.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct baseline_result {
    std::vector<llama_token> tokens;
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
};

double seconds_since(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

int32_t argmax(const float * logits, int32_t n_vocab) {
    int32_t result = 0;
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[result]) {
            result = i;
        }
    }
    return result;
}

void log_errors(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        std::fputs(text, stderr);
    }
}

void usage(const char * argv0) {
    std::fprintf(stderr,
            "usage: %s -m target.gguf -md spd.gguf [-p prompt] [-n tokens] "
            "[-c context] [-b batch] [-ngl layers] [-ngld layers] [--serial-stages]\n",
            argv0);
}

bool tokenize(const llama_vocab * vocab, const std::string & text, std::vector<llama_token> & tokens) {
    const int32_t size = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, true, true);
    if (size <= 0) {
        return false;
    }
    tokens.resize(size);
    return llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(), size, true, true) == size;
}

bool run_baseline(
        llama_model * model,
        const std::vector<llama_token> & prompt,
        int32_t n_predict,
        uint32_t n_ctx,
        uint32_t n_batch,
        baseline_result & result) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_batch;
    cp.n_ubatch = n_batch;
    cp.n_seq_max = 1;
    cp.no_perf = false;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        return false;
    }

    const auto prefill_start = clock_type::now();
    for (size_t begin = 0; begin < prompt.size(); begin += n_batch) {
        const int32_t count = (int32_t) std::min<size_t>(n_batch, prompt.size() - begin);
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(prompt.data() + begin), count);
        if (llama_decode(ctx, batch) != 0) {
            llama_free(ctx);
            return false;
        }
    }
    result.prefill_seconds = seconds_since(prefill_start);

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        llama_free(ctx);
        return false;
    }

    const auto decode_start = clock_type::now();
    for (int32_t i = 0; i < n_predict; ++i) {
        const llama_token token = argmax(logits, n_vocab);
        result.tokens.push_back(token);
        if (i + 1 == n_predict) {
            break;
        }
        llama_batch batch = llama_batch_get_one(&result.tokens.back(), 1);
        if (llama_decode(ctx, batch) != 0) {
            llama_free(ctx);
            return false;
        }
        logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            llama_free(ctx);
            return false;
        }
    }
    result.decode_seconds = seconds_since(decode_start);
    llama_free(ctx);
    return true;
}

std::string detokenize(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::string result;
    std::vector<char> buffer(256);
    for (llama_token token : tokens) {
        int32_t size = llama_token_to_piece(vocab, token, buffer.data(), (int32_t) buffer.size(), 0, true);
        if (size < 0) {
            buffer.resize(-size);
            size = llama_token_to_piece(vocab, token, buffer.data(), (int32_t) buffer.size(), 0, true);
        }
        if (size > 0) {
            result.append(buffer.data(), size);
        }
    }
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    std::string target_path;
    std::string sidecar_path;
    std::string prompt = "The quick brown fox jumps over the lazy dog. Explain why this sentence is useful.";
    int32_t n_predict = 64;
    int32_t n_gpu_layers = 99;
    int32_t n_gpu_layers_draft = 99;
    uint32_t n_ctx = 4096;
    uint32_t n_batch = 512;
    bool parallel_stages = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            target_path = argv[++i];
        } else if ((std::strcmp(argv[i], "-md") == 0 || std::strcmp(argv[i], "--model-draft") == 0) && i + 1 < argc) {
            sidecar_path = argv[++i];
        } else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            n_ctx = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            n_batch = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-ngld") == 0 && i + 1 < argc) {
            n_gpu_layers_draft = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--serial-stages") == 0) {
            parallel_stages = false;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (target_path.empty() || sidecar_path.empty() || n_predict <= 0 || n_batch == 0) {
        usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();
    llama_log_set(log_errors, nullptr);

    llama_model_params target_params = llama_model_default_params();
    target_params.n_gpu_layers = n_gpu_layers;
    llama_model * target = llama_model_load_from_file(target_path.c_str(), target_params);
    if (target == nullptr) {
        std::fprintf(stderr, "failed to load target model\n");
        return 1;
    }

    std::vector<llama_token> prompt_tokens;
    const llama_vocab * vocab = llama_model_get_vocab(target);
    if (!tokenize(vocab, prompt, prompt_tokens)) {
        std::fprintf(stderr, "failed to tokenize prompt\n");
        llama_model_free(target);
        return 1;
    }

    baseline_result baseline;
    std::fprintf(stderr, "[spd] running full-target greedy baseline\n");
    if (!run_baseline(target, prompt_tokens, n_predict, n_ctx, n_batch, baseline)) {
        std::fprintf(stderr, "baseline generation failed\n");
        llama_model_free(target);
        return 1;
    }

    llama_model_params sidecar_params = llama_model_default_params();
    sidecar_params.n_gpu_layers = n_gpu_layers_draft;
    llama_model * sidecar = llama_model_load_from_file(sidecar_path.c_str(), sidecar_params);
    if (sidecar == nullptr) {
        std::fprintf(stderr, "failed to load SPD sidecar model\n");
        llama_model_free(target);
        return 1;
    }

    common_spd_params sp;
    sp.n_ctx = n_ctx;
    sp.n_batch = n_batch;
    sp.n_ubatch = n_batch;
    sp.parallel_stages = parallel_stages;

    std::fprintf(stderr, "[spd] initializing eight-stage SPD controller\n");
    auto pipeline = std::make_unique<common_spd_pipeline>(target, sidecar, sp);
    if (!pipeline->valid()) {
        std::fprintf(stderr, "SPD initialization failed: %s\n", pipeline->error().c_str());
        pipeline.reset();
        llama_model_free(sidecar);
        llama_model_free(target);
        return 1;
    }

    common_spd_result actual;
    std::fprintf(stderr, "[spd] running verified speculative pipeline decoding\n");
    if (!pipeline->generate(prompt_tokens, n_predict, actual)) {
        std::fprintf(stderr, "SPD generation failed: %s\n", pipeline->error().c_str());
        pipeline.reset();
        llama_model_free(sidecar);
        llama_model_free(target);
        return 1;
    }

    size_t matched = 0;
    while (matched < baseline.tokens.size() && matched < actual.tokens.size() &&
           baseline.tokens[matched] == actual.tokens[matched]) {
        ++matched;
    }
    const double baseline_tps = baseline.decode_seconds > 0.0 ? n_predict / baseline.decode_seconds : 0.0;
    const double spd_tps = actual.decode_seconds > 0.0 ? n_predict / actual.decode_seconds : 0.0;
    const uint64_t verified_drafts = actual.n_accepted > 0 ? actual.n_accepted - 1 : 0;
    const uint64_t decisions = verified_drafts + actual.n_rejected;
    const double acceptance = decisions > 0 ? 100.0*verified_drafts/decisions : 100.0;

    std::printf("baseline prefill: %.6f s\n", baseline.prefill_seconds);
    std::printf("SPD prefill:      %.6f s\n", actual.prefill_seconds);
    std::printf("baseline decode:  %.3f tok/s\n", baseline_tps);
    std::printf("SPD decode:       %.3f tok/s\n", spd_tps);
    std::printf("SPD steps:        %llu\n", (unsigned long long) actual.decode_steps);
    std::printf("SPD acceptance:   %.2f%% (%llu accepted, %llu rejected)\n",
            acceptance,
            (unsigned long long) verified_drafts,
            (unsigned long long) actual.n_rejected);
    std::printf("correctness:      %zu/%d greedy tokens matched\n", matched, n_predict);
    std::printf("baseline ids:     ");
    for (llama_token token : baseline.tokens) {
        std::printf("%d ", token);
    }
    std::printf("\nSPD ids:          ");
    for (size_t i = 0; i < actual.tokens.size(); ++i) {
        std::printf("%d%c ", actual.tokens[i], actual.accepted[i] ? '+' : '!');
    }
    std::printf("\n");
    std::printf("\n%s\n", detokenize(vocab, actual.tokens).c_str());

    pipeline.reset();
    llama_model_free(sidecar);
    llama_model_free(target);

    if (matched != (size_t) n_predict) {
        std::fprintf(stderr, "FAIL: verified SPD output differs from the full-target greedy baseline\n");
        return 2;
    }
    std::printf("PASS: verified SPD output matches the full-target greedy baseline\n");
    return 0;
}
