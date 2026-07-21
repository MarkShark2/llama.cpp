#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr uint32_t SPD_STAGE_COUNT = 8;
constexpr uint32_t SPD_ROLLBACK_TOKENS = SPD_STAGE_COUNT - 1;

void log_errors(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        std::fputs(text, stderr);
    }
}

void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s -m target.gguf [-p prompt] [-n greedy_tokens] [-ngl layers]\n", argv0);
}

llama_context_params context_params(uint32_t n_ctx, uint32_t n_batch) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    params.n_ubatch = n_batch;
    params.n_rs_seq = SPD_ROLLBACK_TOKENS;
    params.no_perf = false;
    return params;
}

bool decode_tokens(llama_context * ctx, std::vector<llama_token> & tokens) {
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    return llama_decode(ctx, batch) == 0;
}

bool decode_embeddings(llama_context * ctx, float * embeddings, int32_t n_tokens) {
    llama_batch batch = {};
    batch.n_tokens = n_tokens;
    batch.embd = embeddings;
    return llama_decode(ctx, batch) == 0;
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

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "The quick brown fox jumps over the lazy dog.";
    int32_t n_gpu_layers = 99;
    int32_t n_greedy = 8;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_greedy = std::stoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();
    llama_log_set(log_errors, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load target model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        std::fprintf(stderr, "failed to size tokenized prompt\n");
        llama_model_free(model);
        return 1;
    }

    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), tokens.data(), n_prompt, true, true) != n_prompt) {
        std::fprintf(stderr, "failed to tokenize prompt\n");
        llama_model_free(model);
        return 1;
    }

    if (n_greedy <= 0) {
        std::fprintf(stderr, "greedy token count must be positive\n");
        llama_model_free(model);
        return 1;
    }

    const uint32_t n_ctx = std::max<uint32_t>(256, (uint32_t) n_prompt + n_greedy + 8);
    const uint32_t n_batch = std::max<uint32_t>(32, (uint32_t) n_prompt);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const int32_t n_embd = llama_model_n_embd(model);

    std::vector<float> expected_logits(n_vocab);
    std::vector<llama_token> expected_tokens;
    expected_tokens.reserve(n_greedy);
    {
        std::fprintf(stderr, "[spd-stage-test] running full-target baseline\n");
        llama_context_params cparams = context_params(n_ctx, n_batch);
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (ctx == nullptr || !decode_tokens(ctx, tokens)) {
            std::fprintf(stderr, "baseline target decode failed\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            std::fprintf(stderr, "baseline target produced no logits\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        std::copy(logits, logits + n_vocab, expected_logits.begin());

        for (int32_t i = 0; i < n_greedy; ++i) {
            const llama_token token = argmax(logits, n_vocab);
            expected_tokens.push_back(token);
            if (i + 1 == n_greedy) {
                break;
            }
            std::vector<llama_token> one_token = { token };
            if (!decode_tokens(ctx, one_token)) {
                std::fprintf(stderr, "baseline continuation decode failed at step %d\n", i);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
            logits = llama_get_logits_ith(ctx, -1);
            if (logits == nullptr) {
                std::fprintf(stderr, "baseline continuation produced no logits at step %d\n", i);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
        }
        llama_free(ctx);
    }

    std::vector<llama_context *> stages;
    stages.reserve(SPD_STAGE_COUNT);
    for (uint32_t stage = 0; stage < SPD_STAGE_COUNT; ++stage) {
        std::fprintf(stderr, "[spd-stage-test] creating stage %u\n", stage);
        llama_context_params cparams = context_params(n_ctx, n_batch);
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_SPD_STAGE;
        cparams.spd_stage = stage;
        cparams.spd_stage_count = SPD_STAGE_COUNT;
        cparams.embeddings = true;
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (ctx == nullptr) {
            std::fprintf(stderr, "failed to create SPD stage %u\n", stage);
            for (llama_context * old : stages) {
                llama_free(old);
            }
            llama_model_free(model);
            return 1;
        }
        stages.push_back(ctx);
    }

    std::vector<float> hidden((size_t) n_prompt * n_embd);
    for (uint32_t stage = 0; stage < SPD_STAGE_COUNT; ++stage) {
        std::fprintf(stderr, "[spd-stage-test] running stage %u\n", stage);
        const bool ok = stage == 0
                ? decode_tokens(stages[stage], tokens)
                : decode_embeddings(stages[stage], hidden.data(), n_prompt);
        if (!ok) {
            std::fprintf(stderr, "SPD stage %u decode failed\n", stage);
            for (llama_context * ctx : stages) {
                llama_free(ctx);
            }
            llama_model_free(model);
            return 1;
        }

        const float * output = llama_get_embeddings(stages[stage]);
        if (output == nullptr) {
            std::fprintf(stderr, "SPD stage %u produced no hidden-state output\n", stage);
            for (llama_context * ctx : stages) {
                llama_free(ctx);
            }
            llama_model_free(model);
            return 1;
        }
        std::copy(output, output + hidden.size(), hidden.begin());
    }

    llama_context_params hparams = context_params(n_ctx, 1);
    hparams.ctx_type = LLAMA_CONTEXT_TYPE_SPD_HEAD;
    std::fprintf(stderr, "[spd-stage-test] creating target head\n");
    llama_context * head = llama_init_from_model(model, hparams);
    std::fprintf(stderr, "[spd-stage-test] running target head\n");
    if (head == nullptr || !decode_embeddings(head, hidden.data() + (size_t) (n_prompt - 1) * n_embd, 1)) {
        std::fprintf(stderr, "SPD target head decode failed\n");
        llama_free(head);
        for (llama_context * ctx : stages) {
            llama_free(ctx);
        }
        llama_model_free(model);
        return 1;
    }

    const float * actual_logits = llama_get_logits_ith(head, -1);
    if (actual_logits == nullptr) {
        std::fprintf(stderr, "SPD target head produced no logits\n");
        llama_free(head);
        for (llama_context * ctx : stages) {
            llama_free(ctx);
        }
        llama_model_free(model);
        return 1;
    }

    double squared_error = 0.0;
    float max_abs_error = 0.0f;
    const int32_t expected_argmax = argmax(expected_logits.data(), n_vocab);
    int32_t actual_argmax = argmax(actual_logits, n_vocab);
    const int32_t actual_first_argmax = actual_argmax;
    for (int32_t i = 0; i < n_vocab; ++i) {
        const float error = std::abs(expected_logits[i] - actual_logits[i]);
        max_abs_error = std::max(max_abs_error, error);
        squared_error += (double) error * error;
    }
    const double rmse = std::sqrt(squared_error / n_vocab);

    bool tokens_match = true;
    for (int32_t step = 0; step < n_greedy; ++step) {
        actual_argmax = argmax(actual_logits, n_vocab);
        if (actual_argmax != expected_tokens[step]) {
            std::fprintf(stderr, "greedy mismatch at step %d: baseline=%d staged=%d\n",
                    step, expected_tokens[step], actual_argmax);
            tokens_match = false;
            break;
        }
        if (step + 1 == n_greedy) {
            break;
        }

        std::vector<llama_token> one_token = { actual_argmax };
        for (uint32_t stage = 0; stage < SPD_STAGE_COUNT; ++stage) {
            const bool ok = stage == 0
                    ? decode_tokens(stages[stage], one_token)
                    : decode_embeddings(stages[stage], hidden.data(), 1);
            if (!ok) {
                std::fprintf(stderr, "SPD continuation stage %u failed at step %d\n", stage, step);
                tokens_match = false;
                break;
            }
            const float * output = llama_get_embeddings(stages[stage]);
            if (output == nullptr) {
                std::fprintf(stderr, "SPD continuation stage %u produced no output at step %d\n", stage, step);
                tokens_match = false;
                break;
            }
            std::copy(output, output + n_embd, hidden.begin());
        }
        if (!tokens_match || !decode_embeddings(head, hidden.data(), 1)) {
            tokens_match = false;
            break;
        }
        actual_logits = llama_get_logits_ith(head, -1);
        if (actual_logits == nullptr) {
            tokens_match = false;
            break;
        }
    }

    std::printf("baseline argmax: %d\n", expected_argmax);
    std::printf("staged argmax:   %d\n", actual_first_argmax);
    std::printf("max abs error:   %.9g\n", max_abs_error);
    std::printf("logit RMSE:      %.9g\n", rmse);
    std::printf("greedy tokens:   %d/%d matched\n", tokens_match ? n_greedy : 0, n_greedy);

    llama_free(head);
    for (llama_context * ctx : stages) {
        llama_free(ctx);
    }
    llama_model_free(model);

    if (!tokens_match || !std::isfinite(rmse)) {
        std::fprintf(stderr, "FAIL: staged target does not match baseline\n");
        return 2;
    }

    std::printf("PASS: eight four-layer SPD target stages match the full target greedy continuation\n");
    return 0;
}
