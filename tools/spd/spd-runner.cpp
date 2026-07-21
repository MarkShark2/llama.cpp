#include "llama.h"
#include "spd-pipeline.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <memory>
#include <sstream>
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
            "[-c context] [-b batch] [-ngl layers] [-ngld layers] [--serial-stages]\n"
            "       [--rpc host:port,...] [--rpc-cache] [--device RPC0,...,CUDA0]\n"
            "       [--device-draft CUDA0] [--tensor-split 8,8,8,4,4,0]\n"
            "       [--cache-type-k q8_0] [--cache-type-v q8_0]\n"
            "       [--flash-attn on|off|auto] [--no-mmap]\n",
            argv0);
}

std::vector<std::string> split(const std::string & value, char delimiter) {
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, delimiter)) {
        const size_t first = item.find_first_not_of(" \t");
        const size_t last = item.find_last_not_of(" \t");
        result.push_back(first == std::string::npos ? "" : item.substr(first, last - first + 1));
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
    for (const std::string & endpoint : split(servers, ',')) {
        if (endpoint.empty()) {
            throw std::invalid_argument("empty RPC endpoint");
        }
        ggml_backend_register(add_server(endpoint.c_str()));
    }

    if (cache) {
        using set_cache_fn = void (*)(bool enabled);
        auto set_cache = (set_cache_fn) ggml_backend_reg_get_proc_address(
                rpc_reg, "ggml_backend_rpc_set_client_cache");
        if (set_cache == nullptr) {
            throw std::invalid_argument("failed to find RPC client-cache function");
        }
        set_cache(true);
    }
}

std::vector<ggml_backend_dev_t> parse_devices(const std::string & value) {
    std::vector<ggml_backend_dev_t> result;
    if (value.empty()) {
        return result;
    }
    for (const std::string & name : split(value, ',')) {
        ggml_backend_dev_t device = name.empty() ? nullptr : ggml_backend_dev_by_name(name.c_str());
        if (device == nullptr || ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            throw std::invalid_argument("invalid device: " + name);
        }
        result.push_back(device);
    }
    result.push_back(nullptr);
    return result;
}

std::vector<float> parse_tensor_split(const std::string & value) {
    std::vector<float> result(llama_max_devices(), 0.0f);
    if (value.empty()) {
        return result;
    }
    const std::vector<std::string> values = split(value, ',');
    if (values.size() > result.size()) {
        throw std::invalid_argument("tensor split has more entries than llama_max_devices()");
    }
    bool any = false;
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = std::stof(values[i]);
        if (result[i] < 0.0f) {
            throw std::invalid_argument("tensor split entries must be non-negative");
        }
        any = any || result[i] > 0.0f;
    }
    if (!any) {
        throw std::invalid_argument("tensor split must contain a positive entry");
    }
    return result;
}

ggml_type parse_cache_type(const std::string & value) {
    if (value == "f32")    return GGML_TYPE_F32;
    if (value == "f16")    return GGML_TYPE_F16;
    if (value == "bf16")   return GGML_TYPE_BF16;
    if (value == "q8_0")   return GGML_TYPE_Q8_0;
    if (value == "q4_0")   return GGML_TYPE_Q4_0;
    if (value == "q4_1")   return GGML_TYPE_Q4_1;
    if (value == "q5_0")   return GGML_TYPE_Q5_0;
    if (value == "q5_1")   return GGML_TYPE_Q5_1;
    if (value == "iq4_nl") return GGML_TYPE_IQ4_NL;
    throw std::invalid_argument("invalid cache type: " + value);
}

llama_flash_attn_type parse_flash_attn(const std::string & value) {
    if (value == "on")   return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (value == "off")  return LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (value == "auto") return LLAMA_FLASH_ATTN_TYPE_AUTO;
    throw std::invalid_argument("invalid Flash Attention mode: " + value);
}

void print_devices(const char * label, const std::vector<ggml_backend_dev_t> & devices) {
    std::fprintf(stderr, "[spd] %s:", label);
    if (devices.empty()) {
        std::fprintf(stderr, " auto");
    } else {
        for (ggml_backend_dev_t device : devices) {
            if (device != nullptr) {
                std::fprintf(stderr, " %s", ggml_backend_dev_name(device));
            }
        }
    }
    std::fprintf(stderr, "\n");
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
        const llama_context_params & context_template,
        baseline_result & result) {
    llama_context_params cp = context_template;
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
    bool rpc_cache = false;
    bool use_mmap = true;
    std::string rpc_servers;
    std::string target_device_names;
    std::string draft_device_names;
    std::string tensor_split_arg;
    std::string cache_type_k = "f16";
    std::string cache_type_v = "f16";
    std::string flash_attn = "auto";

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
        } else if (std::strcmp(argv[i], "--rpc") == 0 && i + 1 < argc) {
            rpc_servers = argv[++i];
        } else if (std::strcmp(argv[i], "--rpc-cache") == 0) {
            rpc_cache = true;
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            target_device_names = argv[++i];
        } else if (std::strcmp(argv[i], "--device-draft") == 0 && i + 1 < argc) {
            draft_device_names = argv[++i];
        } else if (std::strcmp(argv[i], "--tensor-split") == 0 && i + 1 < argc) {
            tensor_split_arg = argv[++i];
        } else if (std::strcmp(argv[i], "--cache-type-k") == 0 && i + 1 < argc) {
            cache_type_k = argv[++i];
        } else if (std::strcmp(argv[i], "--cache-type-v") == 0 && i + 1 < argc) {
            cache_type_v = argv[++i];
        } else if (std::strcmp(argv[i], "--flash-attn") == 0 && i + 1 < argc) {
            flash_attn = argv[++i];
        } else if (std::strcmp(argv[i], "--no-mmap") == 0) {
            use_mmap = false;
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

    std::vector<ggml_backend_dev_t> target_devices;
    std::vector<ggml_backend_dev_t> draft_devices;
    std::vector<float> tensor_split;
    llama_context_params target_context = llama_context_default_params();
    try {
        register_rpc_servers(rpc_servers, rpc_cache);
        target_devices = parse_devices(target_device_names);
        draft_devices = parse_devices(draft_device_names);
        tensor_split = parse_tensor_split(tensor_split_arg);
        target_context.type_k = parse_cache_type(cache_type_k);
        target_context.type_v = parse_cache_type(cache_type_v);
        target_context.flash_attn_type = parse_flash_attn(flash_attn);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "invalid arguments: %s\n", error.what());
        return 1;
    }
    if (!tensor_split_arg.empty() && target_devices.empty()) {
        std::fprintf(stderr, "--tensor-split requires an explicit --device list\n");
        return 1;
    }
    if (!draft_device_names.empty() && draft_devices.empty()) {
        std::fprintf(stderr, "--device-draft did not select a device\n");
        return 1;
    }
    print_devices("target devices", target_devices);
    print_devices("sidecar devices", draft_devices);

    llama_model_params target_params = llama_model_default_params();
    target_params.n_gpu_layers = n_gpu_layers;
    target_params.use_mmap = use_mmap;
    target_params.devices = target_devices.empty() ? nullptr : target_devices.data();
    target_params.tensor_split = tensor_split_arg.empty() ? nullptr : tensor_split.data();
    if (!draft_devices.empty()) {
        // Pin embedded MTP/output tensors beside the independently loaded SPD
        // sidecar. This also makes the target split count only the 32 base layers.
        target_params.mtp_dev = draft_devices.front();
    }
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
    if (!run_baseline(target, prompt_tokens, n_predict, n_ctx, n_batch, target_context, baseline)) {
        std::fprintf(stderr, "baseline generation failed\n");
        llama_model_free(target);
        return 1;
    }

    llama_model_params sidecar_params = llama_model_default_params();
    sidecar_params.n_gpu_layers = n_gpu_layers_draft;
    sidecar_params.use_mmap = use_mmap;
    sidecar_params.devices = draft_devices.empty() ? nullptr : draft_devices.data();
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
    sp.target_context = target_context;

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
