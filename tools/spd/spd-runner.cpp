#include "llama.h"
#include "spd-pipeline.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
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
            "       [--flash-attn on|off|auto] [--no-mmap]\n"
            "       [--prompt-file path] [--duration seconds] [--phase-file path]\n"
            "       [--arm both|baseline|spd]\n"
            "       [--check-prefix-reuse --checkpoint-file path]\n"
            "Prefix checks compare repeated, extended and changed prompts with cold greedy output,\n"
            "wrap the bounded prefill ring, repeat in-place rewinds, and check bundle/cancellation recovery.\n"
            "Prefix checks require -n greater than stage_count, batch size at least 2, and enough context\n"
            "for stage_count + 2 batches plus the prompt. --checkpoint-file is overwritten.\n",
            argv0);
}

std::string read_text_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::invalid_argument("failed to open prompt file: " + path);
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void phase_marker(const std::string & path, const char * phase, const char * state) {
    const int64_t unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    std::fprintf(stderr, "[spd-bench] phase=%s state=%s unix_ms=%lld\n",
            phase, state, (long long) unix_ms);
    std::fflush(stderr);
    if (!path.empty()) {
        std::ofstream output(path, std::ios::app);
        output << unix_ms << ',' << phase << ',' << state << '\n';
    }
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

bool tokenize(const llama_vocab * vocab, const std::string & text, std::vector<llama_token> & tokens, bool special = true) {
    const int32_t size = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, special, true);
    if (size <= 0) {
        return false;
    }
    tokens.resize(size);
    return llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(), size, special, true) == size;
}

bool run_baseline(
        llama_model * model,
        const std::vector<llama_token> & prompt,
        int32_t n_predict,
        uint32_t n_ctx,
        uint32_t n_batch,
        const llama_context_params & context_template,
        baseline_result & result,
        bool ignore_eos = false,
        const std::string & grammar = {}) {
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

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> grammar_sampler(nullptr, llama_sampler_free);
    std::vector<llama_token_data> grammar_candidates;
    if (!grammar.empty()) {
        grammar_sampler.reset(llama_sampler_init_grammar(vocab, grammar.c_str(), "root"));
        if (!grammar_sampler) {
            llama_free(ctx);
            return false;
        }
        grammar_candidates.resize(n_vocab);
    }
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        llama_free(ctx);
        return false;
    }

    const auto decode_start = clock_type::now();
    for (int32_t i = 0; i < n_predict; ++i) {
        llama_token token = argmax(logits, n_vocab);
        if (grammar_sampler) {
            for (llama_token candidate = 0; candidate < n_vocab; ++candidate) {
                grammar_candidates[candidate] = { candidate,
                    ignore_eos && llama_vocab_is_eog(vocab, candidate) ? -INFINITY : logits[candidate], 0.0f };
            }
            llama_token_data_array candidates = { grammar_candidates.data(), grammar_candidates.size(), -1, false };
            llama_sampler_apply(grammar_sampler.get(), &candidates);
            const auto * best = std::max_element(candidates.data, candidates.data + candidates.size,
                    [](const llama_token_data & a, const llama_token_data & b) { return a.logit < b.logit; });
            if (candidates.size == 0 || !std::isfinite(best->logit)) {
                llama_free(ctx);
                return false;
            }
            token = best->id;
            llama_sampler_accept(grammar_sampler.get(), token);
        }
        if (ignore_eos && llama_vocab_is_eog(vocab, token)) {
            token = LLAMA_TOKEN_NULL;
            for (llama_token candidate = 0; candidate < n_vocab; ++candidate) {
                if (!llama_vocab_is_eog(vocab, candidate) && (token == LLAMA_TOKEN_NULL || logits[candidate] > logits[token])) {
                    token = candidate;
                }
            }
            if (token == LLAMA_TOKEN_NULL) {
                llama_free(ctx);
                return false;
            }
        }
        result.tokens.push_back(token);
        if (i + 1 == n_predict || (grammar_sampler && !ignore_eos && llama_vocab_is_eog(vocab, token))) {
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

size_t matching_prefix(
        const std::vector<llama_token> & expected,
        const std::vector<llama_token> & actual) {
    size_t matched = 0;
    while (matched < expected.size() && matched < actual.size() &&
           expected[matched] == actual[matched]) {
        ++matched;
    }
    return matched;
}

void print_mismatch(
        const std::vector<llama_token> & expected,
        const std::vector<llama_token> & actual) {
    const size_t matched = matching_prefix(expected, actual);
    std::fprintf(stderr, "mismatch after %zu token(s): expected=", matched);
    if (matched < expected.size()) {
        std::fprintf(stderr, "%d", expected[matched]);
    } else {
        std::fprintf(stderr, "<end>");
    }
    std::fprintf(stderr, " actual=");
    if (matched < actual.size()) {
        std::fprintf(stderr, "%d", actual[matched]);
    } else {
        std::fprintf(stderr, "<end>");
    }
    std::fprintf(stderr, "\nexpected ids:");
    for (llama_token token : expected) {
        std::fprintf(stderr, " %d", token);
    }
    std::fprintf(stderr, "\nactual ids:  ");
    for (llama_token token : actual) {
        std::fprintf(stderr, " %d", token);
    }
    std::fprintf(stderr, "\n");
}

void check_prefix_reuse(llama_model * target, llama_model * sidecar, const common_spd_params & params,
        std::unique_ptr<common_spd_pipeline> & pipeline, const std::vector<llama_token> & prompt,
        int32_t n_predict, const std::string & checkpoint_file) {
    const uint32_t stage_count = pipeline->stage_count();
    if (n_predict <= (int32_t) stage_count) {
        throw std::invalid_argument("prefix checks require -n greater than stage_count to exercise committed generated tokens");
    }
    if (params.n_batch < 2) {
        throw std::invalid_argument("prefix checks require batch size at least 2 for an interior-chunk checkpoint");
    }
    common_spd_gen_params gparams;
    gparams.n_predict = n_predict;
    gparams.ignore_eos = true;
    gparams.prefix_reuse = true;
    gparams.n_ctx_checkpoints = 4;
    gparams.checkpoint_min_step = 256;
    const llama_vocab * vocab = llama_model_get_vocab(target);
    std::vector<llama_token> suffix;
    if (!tokenize(vocab, " Explain the reasoning and compare it with a different example.", suffix, false) || suffix.empty()) {
        throw std::runtime_error("failed to tokenize prefix-check suffix");
    }
    std::vector<std::vector<llama_token>> prompts(3);
    prompts[0] = prompt;
    while (prompts[0].size() < 2 * stage_count) {
        prompts[0].insert(prompts[0].end(), suffix.begin(), suffix.end());
    }
    prompts[1] = prompts[0];
    prompts[1].insert(prompts[1].end(), suffix.begin(), suffix.begin() + std::min<size_t>(2, suffix.size()));
    prompts[2] = prompts[0];
    const auto changed = std::find_if(suffix.begin(), suffix.end(),
            [&](llama_token token) { return token != prompts[2].back(); });
    if (changed == suffix.end()) {
        throw std::runtime_error("prefix-check suffix must contain a distinct replacement token");
    }
    prompts[2].back() = *changed;
    const uint64_t ring_tokens = ((uint64_t) stage_count + 1) * params.n_batch;
    const uint64_t long_size = prompts[0].size() + ((uint64_t) stage_count + 2) * params.n_batch + 1;
    if (long_size + n_predict + stage_count >= params.n_ctx) {
        throw std::invalid_argument("prefix checks need a larger context or smaller batch for ring wrap");
    }
    prompts.push_back(prompts[1]);
    while (prompts.back().size() < long_size) {
        prompts.back().insert(prompts.back().end(), suffix.begin(), suffix.end());
    }
    prompts.back().resize((size_t) long_size);
    const std::vector<llama_token> cancel_prompt = prompts.back();

    pipeline.reset();
    std::vector<baseline_result> expected(prompts.size());
    for (size_t i = 0; i < prompts.size(); ++i) {
        if (!run_baseline(target, prompts[i], n_predict, params.n_ctx, params.n_batch, params.target_context, expected[i], true)) {
            throw std::runtime_error("prefix-check cold baseline failed");
        }
    }
    auto recreate = [&] {
        pipeline.reset();
        pipeline = std::make_unique<common_spd_pipeline>(target, sidecar, params);
        if (!pipeline->valid()) {
            throw std::runtime_error("prefix-check initialization failed: " + pipeline->error());
        }
    };
    recreate();

    const size_t min_reused = prompts[0].size() - stage_count + 1;
    common_spd_result current;
    auto check = [&](const std::string & label, size_t prompt_index, bool reused, bool allow_eviction = false) {
        if (!pipeline->generate(prompts[prompt_index], gparams, current)) {
            throw std::runtime_error(label + " failed: " + pipeline->error());
        }
        if (current.tokens != expected[prompt_index].tokens) {
            print_mismatch(expected[prompt_index].tokens, current.tokens);
            throw std::runtime_error(label + " differs from cold greedy output");
        }
        if (current.n_prompt_reused < 0 || current.n_prompt_processed < 0 ||
            (size_t) current.n_prompt_reused + current.n_prompt_processed != prompts[prompt_index].size() ||
            (reused ? (size_t) current.n_prompt_reused < min_reused && !(allow_eviction && current.n_prompt_reused == 0)
                    : current.n_prompt_reused != 0)) {
            throw std::runtime_error(label + " has unexpected prompt reuse counters");
        }
        std::printf("prefix check %s: PASS (reused=%d processed=%d cached=%d)\n", label.c_str(),
                current.n_prompt_reused, current.n_prompt_processed, current.n_cached_tokens);
    };
    check("cold request", 0, false);
    check("same prompt", 0, true);
    check("extended prompt", 1, true);
    check("changed suffix", 2, true);
    check("changed suffix repeat", 2, true);
    check("return to original prompt", 0, true);
    check("changed suffix after rewind", 2, true);
    check("retained ring-wrap prefill", 3, true);
    const size_t checkpoint_limit = prompts[3].size() - stage_count + 1;
    if ((uint64_t) current.n_prompt_processed <= ring_tokens ||
        (size_t) current.n_prompt_reused != min_reused ||
        (checkpoint_limit - current.n_prompt_reused) % params.n_batch != 1) {
        throw std::runtime_error("retained request did not wrap the ring with an interior-chunk checkpoint");
    }
    check("ring-wrap partial-chunk checkpoint rewind", 3, true);
    if ((size_t) current.n_prompt_reused != checkpoint_limit || current.n_prompt_processed != (int32_t) stage_count - 1) {
        throw std::runtime_error("ring-wrap rewind missed the exact checkpoint limit");
    }
    std::printf("prefix check ring-wrap coverage: PASS (ring_slots=%u checkpoint_limit=%zu replayed=%d)\n",
            stage_count + 1, checkpoint_limit, current.n_prompt_processed);
    check("short prompt recovery after ring wrap", 2, true, true);

    std::vector<llama_token> resident = prompts[2];
    resident.insert(resident.end(), current.tokens.begin(), current.tokens.end());
    if (current.n_cached_tokens <= (int32_t) prompts[2].size() || (size_t) current.n_cached_tokens > resident.size()) {
        throw std::runtime_error("prefix-check state must retain committed generated tokens after the prompt");
    }
    resident.resize(current.n_cached_tokens);
    const size_t state_size = pipeline->state_io().get_size(0, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (state_size == 0) {
        throw std::runtime_error("prefix-check bundle is empty");
    }
    std::vector<uint8_t> state(state_size);
    if (pipeline->state_io().get_data(state.data(), state.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) != state.size()) {
        throw std::runtime_error("prefix-check bundle capture failed");
    }
    const size_t disk_size = pipeline->state_io().save_file(checkpoint_file, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (disk_size == 0) {
        throw std::runtime_error("prefix-check disk bundle save failed");
    }

    const size_t continuation_index = prompts.size();
    prompts.push_back(resident);
    while (prompts.back().size() < resident.size() + stage_count - 1) {
        prompts.back().insert(prompts.back().end(), suffix.begin(), suffix.end());
    }
    const std::string grammar = "root ::= \"A\"";
    const int32_t grammar_predict = std::max(8, n_predict);
    if ((uint64_t) prompts.back().size() + n_predict + stage_count >= params.n_ctx ||
        (uint64_t) prompts[2].size() + grammar_predict + stage_count >= params.n_ctx) {
        throw std::invalid_argument("prefix checks need a larger context for saved-prefix continuation");
    }
    pipeline.reset();
    expected.emplace_back();
    baseline_result grammar_expected;
    if (!run_baseline(target, prompts.back(), n_predict, params.n_ctx, params.n_batch, params.target_context, expected.back(), true) ||
        !run_baseline(target, prompts[2], grammar_predict, params.n_ctx, params.n_batch, params.target_context, grammar_expected, false, grammar)) {
        throw std::runtime_error("saved-prefix continuation or grammar baseline failed");
    }

    recreate();
    if (pipeline->state_io().set_data(state.data(), state.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) != state.size()) {
        throw std::runtime_error("prefix-check bundle restore failed");
    }
    pipeline->note_resident_prefix(resident);
    check("fresh pipeline RAM restore", 2, true);
    check("restored same prompt repeat", 2, true);
    {
        std::vector<llama_token> reexport_resident = prompts[2];
        reexport_resident.insert(reexport_resident.end(), current.tokens.begin(), current.tokens.end());
        if (current.n_cached_tokens <= (int32_t) prompts[2].size() ||
            (size_t) current.n_cached_tokens > reexport_resident.size()) {
            throw std::runtime_error("repeated rewind lost the committed prefix before re-export");
        }
        reexport_resident.resize(current.n_cached_tokens);
        const size_t reexport_size = pipeline->state_io().get_size(0, LLAMA_STATE_SEQ_FLAGS_NONE);
        std::vector<uint8_t> reexport(reexport_size);
        if (reexport.empty() || pipeline->state_io().get_data(reexport.data(), reexport.size(), 0,
                    LLAMA_STATE_SEQ_FLAGS_NONE) != reexport.size()) {
            throw std::runtime_error("repeated rewind bundle re-export failed");
        }
        recreate();
        if (pipeline->state_io().set_data(reexport.data(), reexport.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) != reexport.size()) {
            throw std::runtime_error("re-exported bundle restore failed");
        }
        pipeline->note_resident_prefix(reexport_resident);
        check("re-export after repeated rewind", 2, true);
    }
    if (pipeline->state_io().set_data(state.data(), state.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) != state.size()) {
        throw std::runtime_error("used pipeline bundle restore failed");
    }
    pipeline->note_resident_prefix(resident);
    check("used pipeline RAM restore", 2, true);
    recreate();
    if (pipeline->state_io().set_data(state.data(), state.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) != state.size()) {
        throw std::runtime_error("prefix-check continuation bundle restore failed");
    }
    pipeline->note_resident_prefix(resident);
    check("RAM committed-prefix continuation", continuation_index, true);
    if ((size_t) current.n_prompt_reused != resident.size()) {
        throw std::runtime_error("RAM continuation did not reuse the whole committed prefix");
    }
    recreate();
    if (pipeline->state_io().load_file(checkpoint_file, 0, LLAMA_STATE_SEQ_FLAGS_NONE) != disk_size) {
        throw std::runtime_error("prefix-check disk bundle restore failed");
    }
    pipeline->note_resident_prefix(resident);
    check("disk committed-prefix continuation", continuation_index, true);
    if ((size_t) current.n_prompt_reused != resident.size()) {
        throw std::runtime_error("disk continuation did not reuse the whole committed prefix");
    }

    recreate();
    if (pipeline->state_io().set_data(state.data(), state.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) != state.size()) {
        throw std::runtime_error("prefix-check grammar bundle restore failed");
    }
    pipeline->note_resident_prefix(resident);
    common_spd_gen_params grammar_params = gparams;
    grammar_params.grammar = grammar;
    grammar_params.ignore_eos = false;
    grammar_params.n_predict = grammar_predict;
    common_spd_result grammar_actual;
    if (!pipeline->generate(prompts[2], grammar_params, grammar_actual) ||
        grammar_actual.tokens != grammar_expected.tokens || (size_t) grammar_actual.n_prompt_reused < min_reused) {
        throw std::runtime_error("restored grammar request differs from cold constrained greedy output or lost prefix reuse");
    }
    std::printf("prefix check restored grammar: PASS (reused=%d)\n", grammar_actual.n_prompt_reused);

    // Import into a used pipeline so failure must invalidate its previous prefix record.
    const std::array<const char *, 4> damaged_labels = {
        "bad-magic recovery", "truncated-bundle recovery", "middle-byte recovery", "checksum-footer recovery",
    };
    for (size_t variant = 0; variant < damaged_labels.size(); ++variant) {
        std::vector<uint8_t> corrupt = state;
        if (variant == 0) {
            corrupt[0] ^= 0xff;
        } else if (variant == 1) {
            corrupt.pop_back();
        } else if (variant == 2) {
            corrupt[corrupt.size() / 2] ^= 1;
        } else {
            corrupt.back() ^= 1;
        }
        if (pipeline->state_io().set_data(corrupt.data(), corrupt.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) != 0) {
            throw std::runtime_error("corrupt prefix-check bundle was accepted");
        }
        check(damaged_labels[variant], 2, false);
    }

    {
        std::fstream corrupt(checkpoint_file, std::ios::binary | std::ios::in | std::ios::out);
        char magic = 0;
        corrupt.read(&magic, 1);
        magic ^= 0x7f;
        corrupt.seekp(0);
        corrupt.write(&magic, 1);
        corrupt.flush();
        if (!corrupt) {
            throw std::runtime_error("failed to write bad-magic disk bundle");
        }
    }
    if (pipeline->state_io().load_file(checkpoint_file, 0, LLAMA_STATE_SEQ_FLAGS_NONE) != 0) {
        throw std::runtime_error("bad-magic disk bundle was accepted");
    }
    check("bad-magic disk recovery", 2, false);
    if (pipeline->state_io().save_file(checkpoint_file, 0, LLAMA_STATE_SEQ_FLAGS_NONE) == 0) {
        throw std::runtime_error("failed to replace the damaged test file with a valid bundle");
    }

    common_spd_gen_params cancelled_params = gparams;
    cancelled_params.prefix_reuse = false;
    size_t polls = 0;
    const size_t cancel_poll = (size_t) stage_count + 3;
    cancelled_params.should_cancel = [&] { return ++polls == cancel_poll; };
    common_spd_result cancelled;
    if (pipeline->generate(cancel_prompt, cancelled_params, cancelled) || !cancelled.cancelled ||
        !pipeline->error().empty() || polls != cancel_poll || cancelled.n_cached_tokens != 0) {
        throw std::runtime_error("ring-wrap cancellation did not discard the unfinished request");
    }
    std::printf("prefix check ring-wrap cancellation: PASS (ring_slots=%u stage0_chunks=%zu)\n",
            stage_count + 1, polls - 1);
    check("ring-wrap cancellation recovery", 2, false);

    common_spd_result stopped;
    size_t callbacks = 0;
    if (!pipeline->generate(prompts[2], gparams, stopped, [&](llama_token token, size_t index) {
                if (index != 0 || token != expected[2].tokens.front()) {
                    throw std::runtime_error("wrong first token in stopped prefix-check request");
                }
                ++callbacks;
                return false;
            }) || stopped.cancelled || callbacks != 1 || stopped.tokens.size() != 1) {
        throw std::runtime_error("token callback stop failed");
    }
    check("early-stop recovery", 2, true);
    std::printf("PASS: prefix reuse, full-bundle restore and interrupted-request recovery match cold greedy output\n");
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
    int32_t duration_seconds = 0;
    bool parallel_stages = true;
    bool rpc_cache = false;
    bool use_mmap = true;
    bool check_reuse = false;
    std::string rpc_servers;
    std::string target_device_names;
    std::string draft_device_names;
    std::string tensor_split_arg;
    std::string cache_type_k = "f16";
    std::string cache_type_v = "f16";
    std::string flash_attn = "auto";
    std::string prompt_file;
    std::string phase_file;
    std::string arm = "both";
    std::string checkpoint_file;

    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-m") == 0 || std::strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            target_path = argv[++i];
        } else if ((std::strcmp(argv[i], "-md") == 0 || std::strcmp(argv[i], "--model-draft") == 0) && i + 1 < argc) {
            sidecar_path = argv[++i];
        } else if ((std::strcmp(argv[i], "-p") == 0 || std::strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--n-predict") == 0) && i + 1 < argc) {
            n_predict = std::stoi(argv[++i]);
        } else if ((std::strcmp(argv[i], "-c") == 0 || std::strcmp(argv[i], "--ctx-size") == 0) && i + 1 < argc) {
            n_ctx = std::stoul(argv[++i]);
        } else if ((std::strcmp(argv[i], "-b") == 0 || std::strcmp(argv[i], "--batch-size") == 0) && i + 1 < argc) {
            n_batch = std::stoul(argv[++i]);
        } else if ((std::strcmp(argv[i], "-ngl") == 0 || std::strcmp(argv[i], "--n-gpu-layers") == 0) && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        } else if ((std::strcmp(argv[i], "-ngld") == 0 || std::strcmp(argv[i], "--n-gpu-layers-draft") == 0) && i + 1 < argc) {
            n_gpu_layers_draft = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--serial-stages") == 0) {
            parallel_stages = false;
        } else if (std::strcmp(argv[i], "--check-prefix-reuse") == 0) {
            check_reuse = true;
        } else if (std::strcmp(argv[i], "--checkpoint-file") == 0 && i + 1 < argc) {
            checkpoint_file = argv[++i];
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
        } else if (std::strcmp(argv[i], "--prompt-file") == 0 && i + 1 < argc) {
            prompt_file = argv[++i];
        } else if (std::strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            duration_seconds = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--phase-file") == 0 && i + 1 < argc) {
            phase_file = argv[++i];
        } else if (std::strcmp(argv[i], "--arm") == 0 && i + 1 < argc) {
            arm = argv[++i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    const bool run_baseline_arm = arm == "both" || arm == "baseline";
    const bool run_spd_arm = arm == "both" || arm == "spd";
    if (target_path.empty() || (run_spd_arm && sidecar_path.empty()) ||
        (!run_baseline_arm && !run_spd_arm) || n_predict <= 0 || n_batch == 0 || duration_seconds < 0 ||
        (check_reuse && (!run_spd_arm || checkpoint_file.empty() || duration_seconds != 0)) ||
        (!checkpoint_file.empty() && !check_reuse)) {
        usage(argv[0]);
        return 1;
    }

    try {
        if (!prompt_file.empty()) {
            prompt = read_text_file(prompt_file);
        }
        if (!phase_file.empty()) {
            std::ofstream(phase_file, std::ios::trunc);
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "invalid arguments: %s\n", error.what());
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
    target_params.load_mode = use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
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
    double baseline_prefill_seconds = 0.0;
    double baseline_decode_seconds = 0.0;
    uint64_t baseline_requests = 0;
    uint64_t baseline_generated = 0;
    const int32_t baseline_duration_seconds = run_baseline_arm ? duration_seconds : 0;
    const char * baseline_phase = run_baseline_arm ? "baseline" : "reference";
    std::fprintf(stderr, "[spd] running full-target greedy %s%s\n",
            baseline_phase, baseline_duration_seconds > 0 ? " duration benchmark" : "");
    phase_marker(phase_file, baseline_phase, "start");
    const auto baseline_wall_start = clock_type::now();
    do {
        baseline_result current;
        if (!run_baseline(target, prompt_tokens, n_predict, n_ctx, n_batch, target_context, current)) {
            std::fprintf(stderr, "baseline generation failed\n");
            llama_model_free(target);
            return 1;
        }
        if (baseline_requests == 0) {
            baseline = current;
        } else if (current.tokens != baseline.tokens) {
            std::fprintf(stderr, "baseline greedy output changed between benchmark iterations\n");
            llama_model_free(target);
            return 1;
        }
        baseline_prefill_seconds += current.prefill_seconds;
        baseline_decode_seconds += current.decode_seconds;
        baseline_generated += current.tokens.size();
        ++baseline_requests;
    } while (baseline_duration_seconds > 0 && seconds_since(baseline_wall_start) < baseline_duration_seconds);
    const double baseline_wall_seconds = seconds_since(baseline_wall_start);
    phase_marker(phase_file, baseline_phase, "end");

    const double baseline_pp = baseline_prefill_seconds > 0.0 ?
            (double) baseline_requests*prompt_tokens.size()/baseline_prefill_seconds : 0.0;
    const double baseline_tps = baseline_decode_seconds > 0.0 ? baseline_generated/baseline_decode_seconds : 0.0;

    if (!run_spd_arm) {
        std::printf("prompt tokens:    %zu\n", prompt_tokens.size());
        std::printf("baseline window:  %.3f s (%llu requests)\n",
                baseline_wall_seconds, (unsigned long long) baseline_requests);
        std::printf("baseline PP:      %.3f tok/s\n", baseline_pp);
        std::printf("baseline TG:      %.3f tok/s\n", baseline_tps);
        std::printf("baseline ids:     ");
        for (llama_token token : baseline.tokens) {
            std::printf("%d ", token);
        }
        std::printf("\n\n%s\n", detokenize(vocab, baseline.tokens).c_str());
        llama_model_free(target);
        std::printf("PASS: full-target greedy baseline completed\n");
        return 0;
    }

    llama_model_params sidecar_params = llama_model_default_params();
    sidecar_params.n_gpu_layers = n_gpu_layers_draft;
    sidecar_params.load_mode = use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
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

    auto pipeline = std::make_unique<common_spd_pipeline>(target, sidecar, sp);
    if (!pipeline->valid()) {
        std::fprintf(stderr, "SPD initialization failed: %s\n", pipeline->error().c_str());
        pipeline.reset();
        llama_model_free(sidecar);
        llama_model_free(target);
        return 1;
    }
    std::fprintf(stderr, "[spd] initialized %u-stage SPD controller\n", pipeline->stage_count());

    if (check_reuse) {
        int status = 0;
        try {
            check_prefix_reuse(target, sidecar, sp, pipeline, prompt_tokens, n_predict, checkpoint_file);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "FAIL: %s\n", error.what());
            status = 2;
        }
        pipeline.reset();
        llama_model_free(sidecar);
        llama_model_free(target);
        return status;
    }

    common_spd_result actual;
    double spd_prefill_seconds = 0.0;
    double spd_decode_seconds = 0.0;
    uint64_t spd_requests = 0;
    uint64_t spd_generated = 0;
    uint64_t spd_steps = 0;
    uint64_t spd_accepted = 0;
    uint64_t spd_rejected = 0;
    std::fprintf(stderr, "[spd] running verified speculative pipeline decoding%s\n",
            duration_seconds > 0 ? " duration benchmark" : "");
    phase_marker(phase_file, "spd", "start");
    const auto spd_wall_start = clock_type::now();
    do {
        common_spd_result current;
        common_spd_gen_params gparams;
        gparams.n_predict = n_predict;
        // This loop re-sends the same prompt to build a duration benchmark.
        // With prefix reuse on, every iteration after the first would reuse the
        // prompt wholesale and the prefill number would measure nothing.
        gparams.prefix_reuse = false;
        if (!pipeline->generate(prompt_tokens, gparams, current)) {
            std::fprintf(stderr, "SPD generation failed: %s\n", pipeline->error().c_str());
            pipeline.reset();
            llama_model_free(sidecar);
            llama_model_free(target);
            return 1;
        }
        if (spd_requests == 0) {
            actual = current;
        }
        if (current.tokens != baseline.tokens) {
            std::fprintf(stderr, "FAIL: verified SPD iteration differs from the full-target greedy baseline\n");
            print_mismatch(baseline.tokens, current.tokens);
            pipeline.reset();
            llama_model_free(sidecar);
            llama_model_free(target);
            return 2;
        }
        spd_prefill_seconds += current.prefill_seconds;
        spd_decode_seconds += current.decode_seconds;
        spd_generated += current.tokens.size();
        spd_steps += current.decode_steps;
        spd_accepted += current.n_accepted;
        spd_rejected += current.n_rejected;
        ++spd_requests;
    } while (duration_seconds > 0 && seconds_since(spd_wall_start) < duration_seconds);
    const double spd_wall_seconds = seconds_since(spd_wall_start);
    phase_marker(phase_file, "spd", "end");

    const size_t matched = matching_prefix(baseline.tokens, actual.tokens);
    const double spd_pp = spd_prefill_seconds > 0.0 ?
            (double) spd_requests*prompt_tokens.size()/spd_prefill_seconds : 0.0;
    const double spd_tps = spd_decode_seconds > 0.0 ? spd_generated/spd_decode_seconds : 0.0;
    const uint64_t verified_drafts = spd_accepted > spd_requests ? spd_accepted - spd_requests : 0;
    const uint64_t decisions = verified_drafts + spd_rejected;
    const double acceptance = decisions > 0 ? 100.0*verified_drafts/decisions : 100.0;

    std::printf("prompt tokens:    %zu\n", prompt_tokens.size());
    std::printf("%s window: %.3f s (%llu requests)\n", baseline_phase,
            baseline_wall_seconds, (unsigned long long) baseline_requests);
    std::printf("SPD window:       %.3f s (%llu requests)\n", spd_wall_seconds, (unsigned long long) spd_requests);
    std::printf("%s PP: %.3f tok/s\n", baseline_phase, baseline_pp);
    std::printf("SPD PP:           %.3f tok/s\n", spd_pp);
    std::printf("%s TG: %.3f tok/s\n", baseline_phase, baseline_tps);
    std::printf("SPD TG:           %.3f tok/s\n", spd_tps);
    std::printf("SPD steps:        %llu\n", (unsigned long long) spd_steps);
    std::printf("SPD acceptance:   %.2f%% (%llu accepted, %llu rejected)\n",
            acceptance,
            (unsigned long long) verified_drafts,
            (unsigned long long) spd_rejected);
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
