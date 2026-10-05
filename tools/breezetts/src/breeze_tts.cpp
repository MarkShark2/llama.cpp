// Breeze-TTS 2 engine: T5Gemma2 text encoder -> Qwen3 backbone -> depth decoder over
// 16 codebooks, vocoded by the Qwen3-TTS-12Hz codec (the tokenizer Breeze ships is
// bit-identical to it, so the qwen3tts library's decoder/encoder are reused).
//
// Weights come from the audio.cpp "Breeze-TTS-2" GGUF (Q8_0 matrices, HF tensor names
// under "model/"). The architecture and sampling follow audio.cpp's BreezeTTS runtime
// (Apache-2.0, ShugoAI LLC) re-expressed against plain ggml.

#include "breeze_tts.h"

#include "audio_codec_encoder.h"
#include "audio_tokenizer_decoder.h"
#include "qwen3_tts.h"
#include "gguf_loader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <set>
#include <thread>
#include <thread>

namespace breeze_tts {

using json = nlohmann::json;
using clk = std::chrono::steady_clock;

static int64_t ms_since(clk::time_point t0) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count();
}

// ---------------------------------------------------------------------------
// tokenizer
// ---------------------------------------------------------------------------

static const char * kSpace = "\xE2\x96\x81";  // U+2581

bool bpe_tokenizer::load(const std::string & tokenizer_json, std::string & err) {
    json j;
    try {
        j = json::parse(tokenizer_json);
    } catch (const std::exception & e) {
        err = std::string("tokenizer.json: ") + e.what();
        return false;
    }
    const auto & model = j.at("model");
    if (model.value("type", "") != "BPE") {
        err = "tokenizer.json is not a BPE model";
        return false;
    }
    for (auto & [tok, id] : model.at("vocab").items()) {
        vocab_[tok] = id.get<int32_t>();
    }
    int32_t rank = 0;
    for (const auto & m : model.at("merges")) {
        std::string a, b;
        if (m.is_string()) {
            const std::string s = m.get<std::string>();
            const size_t sp = s.find(' ');
            if (sp == std::string::npos) continue;
            a = s.substr(0, sp);
            b = s.substr(sp + 1);
        } else {
            a = m[0].get<std::string>();
            b = m[1].get<std::string>();
        }
        std::string key = a;
        key.push_back('\0');
        key += b;
        ranks_.emplace(std::move(key), rank++);
    }
    for (int i = 0; i < 256; i++) {
        char buf[16];
        snprintf(buf, sizeof(buf), "<0x%02X>", i);
        auto it = vocab_.find(buf);
        byte_tokens_[i] = it == vocab_.end() ? -1 : it->second;
    }
    auto unk = vocab_.find("<unk>");
    if (unk != vocab_.end()) unk_ = unk->second;
    if (j.contains("added_tokens")) {
        for (const auto & t : j["added_tokens"]) {
            const std::string content = t.at("content").get<std::string>();
            const int32_t id = t.at("id").get<int32_t>();
            vocab_[content] = id;
            if (!content.empty()) specials_.emplace_back(content, id);
        }
    }
    std::sort(specials_.begin(), specials_.end(),
              [](const auto & a, const auto & b) { return a.first.size() > b.first.size(); });
    return true;
}

bool bpe_tokenizer::token_id(const std::string & s, int32_t & id) const {
    auto it = vocab_.find(s);
    if (it == vocab_.end()) return false;
    id = it->second;
    return true;
}

void bpe_tokenizer::encode_plain(const std::string & raw, std::vector<int32_t> & out) const {
    if (raw.empty()) return;
    std::string text;
    for (char c : raw) {
        if (c == ' ') text += kSpace; else text.push_back(c);
    }
    // symbols = utf-8 characters
    std::vector<std::string> sym;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = (unsigned char) text[i];
        size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        n = std::min(n, text.size() - i);
        sym.emplace_back(text.substr(i, n));
        i += n;
    }
    std::string key;
    while (sym.size() > 1) {
        int best = -1;
        int32_t best_rank = INT32_MAX;
        for (size_t i = 0; i + 1 < sym.size(); i++) {
            key = sym[i];
            key.push_back('\0');
            key += sym[i + 1];
            auto it = ranks_.find(key);
            if (it != ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best = (int) i;
            }
        }
        if (best < 0) break;
        sym[best] += sym[best + 1];
        sym.erase(sym.begin() + best + 1);
    }
    for (const auto & s : sym) {
        auto it = vocab_.find(s);
        if (it != vocab_.end()) {
            out.push_back(it->second);
            continue;
        }
        for (unsigned char b : s) {
            out.push_back(byte_tokens_[b] >= 0 ? byte_tokens_[b] : unk_);
        }
    }
}

std::vector<int32_t> bpe_tokenizer::encode(const std::string & text) const {
    std::vector<int32_t> out;
    size_t start = 0;
    size_t i = 0;
    while (i < text.size()) {
        bool hit = false;
        for (const auto & [content, id] : specials_) {
            if (content.size() <= text.size() - i && text.compare(i, content.size(), content) == 0) {
                encode_plain(text.substr(start, i - start), out);
                out.push_back(id);
                i += content.size();
                start = i;
                hit = true;
                break;
            }
        }
        if (!hit) i++;
    }
    encode_plain(text.substr(start), out);
    return out;
}

// ---------------------------------------------------------------------------
// weights
// ---------------------------------------------------------------------------

namespace {

constexpr int kHid = 2048;        // backbone hidden
constexpr int kHeads = 16;
constexpr int kKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kInter = 6144;
constexpr int kLayers = 28;
constexpr int kDHid = 1024;       // depth decoder
constexpr int kDHeads = 8;
constexpr int kDKvHeads = 2;
constexpr int kDInter = 8192;
constexpr int kDLayers = 12;
constexpr int kDCtx = 32;         // depth kv slots (17 used)
constexpr int kTHid = 1152;       // T5Gemma2 text encoder
constexpr int kTHeads = 4;
constexpr int kTHeadDim = 256;
constexpr int kTInter = 6912;
constexpr int kTLayers = 26;
constexpr int kLmHead = 2052;     // codebook vocab + eos
constexpr int32_t kEosToken = 2051;
constexpr float kTRmsEps = 1e-6f;

struct bb_layer {
    ggml_tensor * attn_norm, * wqkv, * q_norm, * k_norm, * wo, * ffn_norm, * w_gate_up, * w_down;
};
struct dp_layer {
    ggml_tensor * attn_norm, * wqkv, * wo, * ffn_norm, * w_gate_up, * w_down;
};
struct te_layer {
    ggml_tensor * pre_attn, * wqkv, * q_norm, * k_norm, * wo, * post_attn,
                * pre_ff, * w_gate_up, * w_down, * post_ff;
};

struct model_weights {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<bb_layer> bb;
    std::vector<dp_layer> dp;
    std::vector<te_layer> te;
    ggml_tensor * bb_norm = nullptr, * lm_head = nullptr;
    ggml_tensor * audio_emb = nullptr;        // [2048, 16*2051] f16, shared embedding table
    ggml_tensor * dp_proj = nullptr;          // [2048, 1024] f16
    ggml_tensor * dp_norm = nullptr;
    ggml_tensor * dp_heads = nullptr;         // [1024, 2051, 15] f16 (transposed at load)
    ggml_tensor * te_embed = nullptr;         // [1152, 262158] f16
    ggml_tensor * te_norm = nullptr, * te_proj = nullptr;
    ggml_tensor * dp_rope = nullptr;   // llama3 rope frequency factors (depth decoder only)
    std::vector<uint16_t> audio_emb_host;     // f16 copy for building clone prompts
};

// one source tensor in the file
struct src_tensor {
    int64_t idx = -1;
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {1, 1, 1, 1};
    size_t offset = 0;
    size_t size = 0;
};

class gguf_file {
public:
    ~gguf_file() {
        if (fp_) fclose(fp_);
        if (g_) gguf_free(g_);
        if (meta_) ggml_free(meta_);
    }
    bool open(const std::string & path, std::string & err) {
        gguf_init_params p = {true, &meta_};
        g_ = gguf_init_from_file(path.c_str(), p);
        if (!g_) {
            err = "failed to read GGUF: " + path;
            return false;
        }
        fp_ = fopen(path.c_str(), "rb");
        if (!fp_) {
            err = "failed to open " + path;
            return false;
        }
        data_off_ = gguf_get_data_offset(g_);
        return true;
    }
    gguf_context * ctx() const { return g_; }
    gguf_context * ctx() const { return g_; }
    bool find(const std::string & name, src_tensor & t) const {
        // names past ggml's 64-byte limit are stored as "_audiocpp.<index>", the index being
        // the position of the real name in the audiocpp.tensor_names array
        std::string stored = name;
        int64_t i = gguf_find_tensor(g_, stored.c_str());
        if (i < 0) {
            const int64_t kn = gguf_find_key(g_, "audiocpp.tensor_names");
            if (kn < 0) return false;
            const size_t n = gguf_get_arr_n(g_, kn);
            for (size_t k = 0; k < n; k++) {
                if (name == gguf_get_arr_str(g_, kn, k)) {
                    stored = "_audiocpp." + std::to_string(k);
                    i = gguf_find_tensor(g_, stored.c_str());
                    break;
                }
            }
            if (i < 0) return false;
        }
        ggml_tensor * m = ggml_get_tensor(meta_, stored.c_str());
        t.idx = i;
        t.type = gguf_get_tensor_type(g_, i);
        t.offset = data_off_ + gguf_get_tensor_offset(g_, i);
        t.size = gguf_get_tensor_size(g_, i);
        for (int d = 0; d < 4; d++) t.ne[d] = m->ne[d];
        return true;
    }
    bool read(size_t offset, void * dst, size_t n) const {
#ifdef _WIN32
        _fseeki64(fp_, (long long) offset, SEEK_SET);
#else
        fseeko(fp_, (off_t) offset, SEEK_SET);
#endif
        return fread(dst, 1, n, fp_) == n;
    }
    // embedded file by name ("config.json", "tokenizer.json"...)
    bool embedded(const std::string & name, std::string & out) const {
        const int64_t kn = gguf_find_key(g_, "audiocpp.embedded_files.names");
        const int64_t ko = gguf_find_key(g_, "audiocpp.embedded_files.offsets");
        const int64_t kd = gguf_find_key(g_, "audiocpp.embedded_files.data");
        if (kn < 0 || ko < 0 || kd < 0) return false;
        const size_t n = gguf_get_arr_n(g_, kn);
        const uint64_t * offs = (const uint64_t *) gguf_get_arr_data(g_, ko);
        const uint8_t * data = (const uint8_t *) gguf_get_arr_data(g_, kd);
        for (size_t i = 0; i < n; i++) {
            if (name == gguf_get_arr_str(g_, kn, i)) {
                out.assign((const char *) data + offs[i], (size_t) (offs[i + 1] - offs[i]));
                return true;
            }
        }
        return false;
    }

private:
    gguf_context * g_ = nullptr;
    ggml_context * meta_ = nullptr;
    FILE * fp_ = nullptr;
    size_t data_off_ = 0;
};

// pending uploads, executed after the weight buffer exists
struct upload {
    ggml_tensor * dst;
    std::function<bool(const gguf_file &, std::string &)> run;
};

// A LoRA adapter (convert_lora.py output) merged into the base weights while they load, so serving
// pays nothing per step.  The stored matrix is dequantized, W += scale * B A, and requantized.
struct lora_adapter {
    gguf_file file;
    float scale = 1.0f;
    size_t merged = 0;

    bool open(const std::string & path, float strength, std::string & err) {
        if (!file.open(path, err)) return false;
        const int64_t kr = gguf_find_key(file.ctx(), "breeze-tts.lora.rank");
        const int64_t ka = gguf_find_key(file.ctx(), "breeze-tts.lora.alpha");
        if (kr < 0 || ka < 0) {
            err = "not a breeze-tts LoRA GGUF: " + path;
            return false;
        }
        scale = gguf_get_val_f32(file.ctx(), ka) / (float) gguf_get_val_u32(file.ctx(), kr) * strength;
        return true;
    }
    bool has(const std::string & key) const {
        src_tensor t;
        return file.find(key + ".lora_a", t);
    }
    // `raw` holds the stored bytes of the matrix `key` ("<weight name> minus .weight")
    bool merge(const std::string & key, const src_tensor & s, uint8_t * raw, std::string & err) {
        src_tensor ta, tb;
        if (!file.find(key + ".lora_a", ta) || !file.find(key + ".lora_b", tb)) return true;
        const int64_t in = s.ne[0], out = s.ne[1], r = ta.ne[1];
        if (ta.ne[0] != in || tb.ne[1] != out || tb.ne[0] != r) {
            err = "LoRA shape mismatch for " + key;
            return false;
        }
        std::vector<float> A(in * r), B(out * r);
        if (!file.read(ta.offset, A.data(), A.size() * sizeof(float)) ||
            !file.read(tb.offset, B.data(), B.size() * sizeof(float))) {
            err = "short read (LoRA " + key + ")";
            return false;
        }
        const ggml_type_traits * tr = ggml_get_type_traits(s.type);
        if (!tr || !tr->to_float) {
            err = "cannot merge LoRA into this weight type: " + key;
            return false;
        }
        const size_t row_bytes = ggml_row_size(s.type, in);
        const int nt = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
        auto work = [&](int64_t o0, int64_t o1) {
            std::vector<float> row(in);
            std::vector<uint8_t> q(row_bytes);
            for (int64_t o = o0; o < o1; o++) {
                uint8_t * dst = raw + o * row_bytes;
                tr->to_float(dst, row.data(), in);
                for (int64_t k = 0; k < r; k++) {
                    const float c = scale * B[o * r + k];
                    const float * a = &A[k * in];
                    for (int64_t i = 0; i < in; i++) row[i] += c * a[i];
                }
                if (s.type == GGML_TYPE_F32) memcpy(dst, row.data(), row_bytes);
                else ggml_quantize_chunk(s.type, row.data(), dst, 0, 1, in, nullptr);
            }
        };
        std::vector<std::thread> th;
        for (int t = 0; t < nt; t++) th.emplace_back(work, out * t / nt, out * (t + 1) / nt);
        for (auto & t : th) t.join();
        merged++;
        return true;
    }
};

// A LoRA adapter (convert_lora.py output) merged into the base weights while they load, so serving
// pays nothing per step.  The stored matrix is dequantized, W += scale * B A, and requantized.
struct lora_adapter {
    gguf_file file;
    float scale = 1.0f;
    size_t merged = 0;

    bool open(const std::string & path, float strength, std::string & err) {
        if (!file.open(path, err)) return false;
        const int64_t kr = gguf_find_key(file.ctx(), "breeze-tts.lora.rank");
        const int64_t ka = gguf_find_key(file.ctx(), "breeze-tts.lora.alpha");
        if (kr < 0 || ka < 0) {
            err = "not a breeze-tts LoRA GGUF: " + path;
            return false;
        }
        scale = gguf_get_val_f32(file.ctx(), ka) / (float) gguf_get_val_u32(file.ctx(), kr) * strength;
        return true;
    }
    bool has(const std::string & key) const {
        src_tensor t;
        return file.find(key + ".lora_a", t);
    }
    // `raw` holds the stored bytes of the matrix `key` ("<weight name> minus .weight")
    bool merge(const std::string & key, const src_tensor & s, uint8_t * raw, std::string & err) {
        src_tensor ta, tb;
        if (!file.find(key + ".lora_a", ta) || !file.find(key + ".lora_b", tb)) return true;
        const int64_t in = s.ne[0], out = s.ne[1], r = ta.ne[1];
        if (ta.ne[0] != in || tb.ne[1] != out || tb.ne[0] != r) {
            err = "LoRA shape mismatch for " + key;
            return false;
        }
        std::vector<float> A(in * r), B(out * r);
        if (!file.read(ta.offset, A.data(), A.size() * sizeof(float)) ||
            !file.read(tb.offset, B.data(), B.size() * sizeof(float))) {
            err = "short read (LoRA " + key + ")";
            return false;
        }
        const ggml_type_traits * tr = ggml_get_type_traits(s.type);
        if (!tr || !tr->to_float) {
            err = "cannot merge LoRA into this weight type: " + key;
            return false;
        }
        const size_t row_bytes = ggml_row_size(s.type, in);
        const int nt = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
        auto work = [&](int64_t o0, int64_t o1) {
            std::vector<float> row(in);
            for (int64_t o = o0; o < o1; o++) {
                uint8_t * dst = raw + o * row_bytes;
                tr->to_float(dst, row.data(), in);
                for (int64_t k = 0; k < r; k++) {
                    const float c = scale * B[o * r + k];
                    const float * a = &A[k * in];
                    for (int64_t i = 0; i < in; i++) row[i] += c * a[i];
                }
                if (s.type == GGML_TYPE_F32) memcpy(dst, row.data(), row_bytes);
                else ggml_quantize_chunk(s.type, row.data(), dst, 0, 1, in, nullptr);
            }
        };
        std::vector<std::thread> th;
        for (int t = 0; t < nt; t++) th.emplace_back(work, out * t / nt, out * (t + 1) / nt);
        for (auto & t : th) t.join();
        merged++;
        return true;
    }
};

std::vector<float> llama3_rope_factors(int head_dim, float theta, float factor,
                                       float low_freq_factor, float high_freq_factor,
                                       int orig_ctx) {
    const double pi = 3.14159265358979323846;
    const double low_wl = (double) orig_ctx / low_freq_factor;
    const double high_wl = (double) orig_ctx / high_freq_factor;
    std::vector<float> out(head_dim / 2, 1.0f);
    for (int i = 0; i < head_dim / 2; i++) {
        const double inv = 1.0 / std::pow((double) theta, (double) (2 * i) / head_dim);
        const double wl = 2.0 * pi / inv;
        double scaled = inv;
        if (wl > low_wl) {
            scaled = inv / factor;
        } else if (wl >= high_wl) {
            const double smooth = ((double) orig_ctx / wl - low_freq_factor) /
                                  ((double) high_freq_factor - low_freq_factor);
            scaled = (1.0 - smooth) * inv / factor + smooth * inv;
        }
        out[i] = (float) (inv / scaled);
    }
    return out;
}

// ---------------------------------------------------------------------------
// compiled graph with its own allocator
// ---------------------------------------------------------------------------

struct graph {
    std::vector<uint8_t> meta;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_backend_t backend = nullptr;

    ~graph() { reset(); }
    void reset() {
        if (galloc) ggml_gallocr_free(galloc);
        if (ctx) ggml_free(ctx);
        galloc = nullptr;
        ctx = nullptr;
        gf = nullptr;
    }
    void init(ggml_backend_t be, size_t n_nodes) {
        reset();
        backend = be;
        meta.resize(ggml_tensor_overhead() * n_nodes * 2 + ggml_graph_overhead_custom(n_nodes, false) + 4096);
        ggml_init_params p = {meta.size(), meta.data(), true};
        ctx = ggml_init(p);
        gf = ggml_new_graph_custom(ctx, n_nodes, false);
    }
    bool alloc() {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        return ggml_gallocr_alloc_graph(galloc, gf);
    }
    bool run() { return ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS; }
    ggml_tensor * get(const char * name) { return ggml_graph_get_tensor(gf, name); }
};

bool g_bf16 = false;   // round activations through bf16 as the reference implementation does

ggml_tensor * rnd(ggml_context * c, ggml_tensor * x) {
    if (!g_bf16) return x;
    return ggml_cast(c, ggml_cast(c, x, GGML_TYPE_BF16), GGML_TYPE_F32);
}

ggml_tensor * rms_w(ggml_context * c, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
}

ggml_tensor * named(ggml_tensor * t, const char * name, bool out = false) {
    ggml_set_name(t, name);
    if (out) ggml_set_output(t);
    return t;
}

ggml_tensor * input(ggml_context * c, ggml_type ty, const char * name, int64_t n0, int64_t n1 = 1) {
    ggml_tensor * t = n1 == 1 ? ggml_new_tensor_1d(c, ty, n0) : ggml_new_tensor_2d(c, ty, n0, n1);
    ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}

// a column slice [rows of one packed projection] of a [total, n] matmul result
ggml_tensor * slice_cols(ggml_context * c, ggml_tensor * x, int64_t ne0, int64_t n, size_t col_off) {
    return ggml_cont(c, ggml_view_2d(c, x, ne0, n, x->nb[1], col_off * sizeof(float)));
}

// one decoder layer (backbone or depth) over n tokens, kv cache written at `pos`
struct dec_dims {
    int hid, heads, kv_heads, hd, inter;
    float theta;
    float eps;
};

ggml_tensor * decoder_layer(ggml_context * c, ggml_cgraph * gf, const dec_dims & d,
                            ggml_tensor * x, int n,
                            ggml_tensor * attn_norm, ggml_tensor * wqkv,
                            ggml_tensor * q_norm, ggml_tensor * k_norm, ggml_tensor * wo,
                            ggml_tensor * ffn_norm, ggml_tensor * w_gate_up, ggml_tensor * w_down,
                            ggml_tensor * rope_factors, ggml_tensor * pos, ggml_tensor * rows64,
                            ggml_tensor * kc, ggml_tensor * vc, int n_kv, ggml_tensor * mask) {
    const int q_dim = d.heads * d.hd;
    const int kv_dim = d.kv_heads * d.hd;
    ggml_tensor * h = rnd(c, rms_w(c, x, attn_norm, d.eps));
    ggml_tensor * qkv = rnd(c, ggml_mul_mat(c, wqkv, h));              // [q+2kv, n]
    // q, k and v are read straight out of the packed projection: a copy per slice costs a
    // kernel launch each, and a one-token step is bound by the number of dependent kernels
    const size_t fsz = sizeof(float);
    ggml_tensor * q = ggml_view_3d(c, qkv, d.hd, d.heads, n, d.hd * fsz, qkv->nb[1], 0);
    ggml_tensor * k = ggml_view_3d(c, qkv, d.hd, d.kv_heads, n, d.hd * fsz, qkv->nb[1], q_dim * fsz);
    ggml_tensor * v2 = ggml_view_2d(c, qkv, kv_dim, n, qkv->nb[1], (q_dim + kv_dim) * fsz);
    if (q_norm) {
        q = rnd(c, rms_w(c, q, q_norm, d.eps));
        k = rnd(c, rms_w(c, k, k_norm, d.eps));
    }
    q = rnd(c, ggml_rope_ext(c, q, pos, rope_factors, d.hd, GGML_ROPE_TYPE_NEOX, 0, d.theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f));
    k = rnd(c, ggml_rope_ext(c, k, pos, rope_factors, d.hd, GGML_ROPE_TYPE_NEOX, 0, d.theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f));

    // the K cache is a plain 2D leaf so rope + cache write stay three adjacent nodes the backend can fuse
    const int64_t n_ctx = kc->ne[1];
    ggml_tensor * vc2 = ggml_view_2d(c, vc, kv_dim, n_ctx, vc->nb[2], 0);
    // a VIEW (not a reshape) with I64 row indices lets the Vulkan backend fuse rope + cache write
    ggml_tensor * k2 = ggml_view_2d(c, k, kv_dim, n, (size_t) kv_dim * sizeof(float), 0);
    ggml_build_forward_expand(gf, ggml_set_rows(c, kc, k2, rows64));
    ggml_build_forward_expand(gf, ggml_set_rows(c, vc2, v2, rows64));

    ggml_tensor * K = ggml_view_3d(c, kc, d.hd, d.kv_heads, n_kv, d.hd * ggml_type_size(kc->type), kc->nb[1], 0);
    ggml_tensor * V = ggml_view_3d(c, vc, d.hd, d.kv_heads, n_kv, vc->nb[1], vc->nb[2], 0);
    ggml_tensor * Q = ggml_permute(c, q, 0, 2, 1, 3);
    K = ggml_permute(c, K, 0, 2, 1, 3);
    V = ggml_permute(c, V, 0, 2, 1, 3);
    ggml_tensor * att = ggml_flash_attn_ext(c, Q, K, V, mask, 1.0f / sqrtf((float) d.hd), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(att, GGML_PREC_F32);
    att = rnd(c, ggml_reshape_2d(c, att, q_dim, n));
    ggml_tensor * o = rnd(c, ggml_mul_mat(c, wo, att));
    x = rnd(c, ggml_add(c, x, o));

    h = rnd(c, rms_w(c, x, ffn_norm, d.eps));
    ggml_tensor * gu = rnd(c, ggml_mul_mat(c, w_gate_up, h));          // [2*inter, n] (gate; up)
    ggml_tensor * act = rnd(c, ggml_swiglu(c, gu));
    ggml_tensor * dn = rnd(c, ggml_mul_mat(c, w_down, act));
    return rnd(c, ggml_add(c, x, dn));
}

// llama.cpp-style sampling over a logits vector
int32_t sample_token(std::vector<float> & logits, const std::vector<int32_t> * history,
                     float rep_penalty, float temperature, int top_k, float top_p,
                     std::mt19937_64 & rng) {
    const int n = (int) logits.size();
    if (history && rep_penalty != 1.0f) {
        std::set<int32_t> seen(history->begin(), history->end());
        for (int32_t t : seen) {
            if (t >= 0 && t < n) logits[t] = logits[t] > 0 ? logits[t] / rep_penalty : logits[t] * rep_penalty;
        }
    }
    if (temperature <= 0.0f) {
        return (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    std::vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i;
    int keep = n;
    if (top_k > 0 && top_k < n) {
        std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        keep = top_k;
    } else {
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    }
    const float mx = logits[idx[0]];
    std::vector<double> p(keep);
    double sum = 0;
    for (int i = 0; i < keep; i++) {
        p[i] = std::exp((double) (logits[idx[i]] - mx) / temperature);
        sum += p[i];
    }
    for (auto & v : p) v /= sum;
    if (top_p < 1.0f) {
        double cum = 0;
        int cut = keep;
        for (int i = 0; i < keep; i++) {
            cum += p[i];
            if (cum >= top_p) { cut = i + 1; break; }
        }
        keep = std::max(1, cut);
        sum = 0;
        for (int i = 0; i < keep; i++) sum += p[i];
        for (int i = 0; i < keep; i++) p[i] /= sum;
    }
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng), acc = 0;
    for (int i = 0; i < keep; i++) {
        acc += p[i];
        if (r <= acc) return idx[i];
    }
    return idx[keep - 1];
}

}  // namespace

// ---------------------------------------------------------------------------
// engine
// ---------------------------------------------------------------------------

struct engine_impl {
    std::string err;
    ggml_backend_t backend = nullptr;
    model_weights w;
    bpe_tokenizer tok;
    int n_ctx = 2048;
    std::string lora_path;
    float lora_strength = 1.0f;
    std::unique_ptr<lora_adapter> lora;
    int32_t bos = 2, s0 = 262146, ins_bos = 262156, ins_eos = 262157;

    // kv caches
    ggml_context * cache_ctx = nullptr;
    ggml_backend_buffer_t cache_buf = nullptr;
    std::vector<ggml_tensor *> bb_k, bb_v, dp_k, dp_v;

    qwen3_tts::AudioTokenizerDecoder decoder;
    qwen3_tts::AudioCodecEncoder codec_enc;
    bool codec_enc_loaded = false;
    std::string codec_path;

    // static per-frame graphs
    std::vector<std::unique_ptr<graph>> bb_step;   // one per kv bucket
    std::vector<int> bb_step_kv;
    graph dp_prefill, dp_step;
    bool depth_ready = false;

    bool timing = false;

    ~engine_impl() {
        bb_step.clear();
        dp_prefill.reset();
        dp_step.reset();
        if (cache_buf) ggml_backend_buffer_free(cache_buf);
        if (cache_ctx) ggml_free(cache_ctx);
        if (w.buf) ggml_backend_buffer_free(w.buf);
        if (w.ctx) ggml_free(w.ctx);
        decoder.unload_model();
        if (backend) qwen3_tts::release_preferred_backend(backend);
    }

    bool load(const std::string & model_path, const std::string & codec, int ctx, std::string & e);
    bool build_depth_graphs();
    graph * get_bb_step(int n_past);
    bool text_encode(const std::vector<int32_t> & ids, std::vector<float> & out, std::string & e);
    bool backbone_prefill(const std::vector<float> & embd, int n, std::vector<float> & logits,
                          std::vector<float> & hidden, std::string & e);
    bool backbone_step(const int32_t * rows, int n_past, std::vector<float> & logits,
                       std::vector<float> & hidden, std::string & e);
    void frame_embedding(const int32_t * codes, float * out);
    tts_result synthesize(const std::string & text, const tts_params & p, const reference_voice * ref);
};

bool engine_impl::load(const std::string & model_path, const std::string & codec, int ctx, std::string & e) {
    n_ctx = ctx;
    timing = std::getenv("BREEZE_TTS_TIMING") != nullptr;
    g_bf16 = std::getenv("BREEZE_TTS_BF16") != nullptr;
    codec_path = codec;

    std::string berr;
    backend = qwen3_tts::init_preferred_backend("BreezeTTS", &berr);
    if (!backend) {
        e = berr;
        return false;
    }
    fprintf(stderr, "breeze-tts: backend %s\n", ggml_backend_name(backend));

    gguf_file gf;
    if (!gf.open(model_path, e)) return false;
    if (!lora_path.empty()) {
        lora = std::make_unique<lora_adapter>();
        if (!lora->open(lora_path, lora_strength, e)) return false;
        fprintf(stderr, "breeze-tts: LoRA %s (scale %.3f)\n", lora_path.c_str(), lora->scale);
    }
    lora_adapter * lo = lora.get();

    std::string tj;
    if (!gf.embedded("tokenizer.json", tj)) {
        e = "model GGUF has no embedded tokenizer.json";
        return false;
    }
    if (!tok.load(tj, e)) return false;
    tok.token_id("<bos>", bos);
    tok.token_id("[S0]", s0);
    tok.token_id("<ins_bos>", ins_bos);
    tok.token_id("<ins_eos>", ins_eos);

    // ---- declare weight tensors -------------------------------------------------
    const size_t n_decl = 4096;
    ggml_init_params ip = {ggml_tensor_overhead() * n_decl, nullptr, true};
    w.ctx = ggml_init(ip);
    ggml_context * c = w.ctx;
    std::vector<upload> ups;
    bool ok = true;

    auto find = [&](const std::string & name, src_tensor & t) {
        if (!gf.find("model/" + name, t)) {
            e = "missing tensor model/" + name;
            ok = false;
            return false;
        }
        return true;
    };
    // matrix kept in its stored type
    auto mat = [&](const std::string & name) -> ggml_tensor * {
        src_tensor s;
        if (!find(name, s)) return nullptr;
        ggml_tensor * t = ggml_new_tensor_2d(c, s.type, s.ne[0], s.ne[1]);
        ggml_set_name(t, name.c_str());
        const std::string key = name.substr(0, name.size() - 7);  // minus ".weight"
        ups.push_back({t, [s, t, lo, key](const gguf_file & f, std::string & er) {
            if (lo && lo->has(key)) {
                std::vector<uint8_t> whole(s.size);
                if (!f.read(s.offset, whole.data(), s.size)) { er = "short read"; return false; }
                if (!lo->merge(key, s, whole.data(), er)) return false;
                ggml_backend_tensor_set(t, whole.data(), 0, s.size);
                return true;
            }
            std::vector<uint8_t> buf(std::min<size_t>(s.size, 64u << 20));
            for (size_t off = 0; off < s.size; off += buf.size()) {
                const size_t n = std::min(buf.size(), s.size - off);
                if (!f.read(s.offset + off, buf.data(), n)) { er = "short read"; return false; }
                ggml_backend_tensor_set(t, buf.data(), off, n);
            }
            return true;
        }});
        return t;
    };
    // several same-type matrices concatenated along the output dimension
    auto pack = [&](const std::vector<std::string> & names, const std::string & label) -> ggml_tensor * {
        std::vector<src_tensor> ss(names.size());
        int64_t out = 0;
        for (size_t i = 0; i < names.size(); i++) {
            if (!find(names[i], ss[i])) return nullptr;
            out += ss[i].ne[1];
        }
        ggml_tensor * t = ggml_new_tensor_2d(c, ss[0].type, ss[0].ne[0], out);
        ggml_set_name(t, label.c_str());
        ups.push_back({t, [ss, t, lo, names](const gguf_file & f, std::string & er) {
            size_t dst = 0;
            std::vector<uint8_t> buf;
            for (size_t si = 0; si < ss.size(); si++) {
                const auto & s = ss[si];
                buf.resize(s.size);
                if (!f.read(s.offset, buf.data(), s.size)) { er = "short read"; return false; }
                if (lo && !lo->merge(names[si].substr(0, names[si].size() - 7), s, buf.data(), er)) return false;
                ggml_backend_tensor_set(t, buf.data(), dst, s.size);
                dst += s.size;
            }
            return true;
        }});
        return t;
    };
    // norm weight (bf16/f16/f32 in the file) as f32, optionally +1 (Gemma)
    auto norm = [&](const std::string & name, bool plus_one) -> ggml_tensor * {
        src_tensor s;
        if (!find(name, s)) return nullptr;
        ggml_tensor * t = ggml_new_tensor_1d(c, GGML_TYPE_F32, s.ne[0]);
        ggml_set_name(t, name.c_str());
        ups.push_back({t, [s, t, plus_one](const gguf_file & f, std::string & er) {
            std::vector<uint8_t> raw(s.size);
            if (!f.read(s.offset, raw.data(), s.size)) { er = "short read"; return false; }
            std::vector<float> v(s.ne[0]);
            for (int64_t i = 0; i < s.ne[0]; i++) {
                if (s.type == GGML_TYPE_BF16) v[i] = ggml_bf16_to_fp32(((const ggml_bf16_t *) raw.data())[i]);
                else if (s.type == GGML_TYPE_F16) v[i] = ggml_fp16_to_fp32(((const ggml_fp16_t *) raw.data())[i]);
                else v[i] = ((const float *) raw.data())[i];
                if (plus_one) v[i] += 1.0f;
            }
            ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
            return true;
        }});
        return t;
    };
    auto f32_vec = [&](int64_t n, const std::string & label, std::vector<float> data) -> ggml_tensor * {
        ggml_tensor * t = ggml_new_tensor_1d(c, GGML_TYPE_F32, n);
        ggml_set_name(t, label.c_str());
        ups.push_back({t, [t, data](const gguf_file &, std::string &) {
            ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
            return true;
        }});
        return t;
    };

    // backbone: llama3 rope scaling (factor 32, 0.125/0.5, orig 1024); depth: (32, 1/512, 1/128, orig 16)
    w.dp_rope = f32_vec(kHeadDim / 2, "dp_rope", llama3_rope_factors(kHeadDim, 500000.f, 32.f, 0.001953125f, 0.0078125f, 16));

    for (int il = 0; il < kLayers && ok; il++) {
        const std::string p = "backbone_model.layers." + std::to_string(il);
        bb_layer L;
        L.attn_norm = norm(p + ".input_layernorm.weight", false);
        L.wqkv = pack({p + ".self_attn.q_proj.weight", p + ".self_attn.k_proj.weight", p + ".self_attn.v_proj.weight"},
                      p + ".wqkv");
        L.q_norm = norm(p + ".self_attn.q_norm.weight", false);
        L.k_norm = norm(p + ".self_attn.k_norm.weight", false);
        L.wo = mat(p + ".self_attn.o_proj.weight");
        L.ffn_norm = norm(p + ".post_attention_layernorm.weight", false);
        L.w_gate_up = pack({p + ".mlp.gate_proj.weight", p + ".mlp.up_proj.weight"}, p + ".w_gate_up");
        L.w_down = mat(p + ".mlp.down_proj.weight");
        w.bb.push_back(L);
    }
    w.bb_norm = norm("backbone_model.norm.weight", false);
    w.lm_head = mat("lm_head.weight");
    for (int il = 0; il < kDLayers && ok; il++) {
        const std::string p = "depth_decoder.model.layers." + std::to_string(il);
        dp_layer L;
        L.attn_norm = norm(p + ".input_layernorm.weight", false);
        L.wqkv = pack({p + ".self_attn.q_proj.weight", p + ".self_attn.k_proj.weight", p + ".self_attn.v_proj.weight"},
                      p + ".wqkv");
        L.wo = mat(p + ".self_attn.o_proj.weight");
        L.ffn_norm = norm(p + ".post_attention_layernorm.weight", false);
        L.w_gate_up = pack({p + ".mlp.gate_proj.weight", p + ".mlp.up_proj.weight"}, p + ".w_gate_up");
        L.w_down = mat(p + ".mlp.down_proj.weight");
        w.dp.push_back(L);
    }
    w.dp_norm = norm("depth_decoder.model.norm.weight", false);
    w.audio_emb = mat("depth_decoder.model.embed_tokens.weight");
    w.dp_proj = mat("depth_decoder.model.inputs_embeds_projector.weight");
    {
        // codebooks_head is stored [15][1024 in][2051 out]; the matmul wants [1024 in, 2051 out] per head
        src_tensor s;
        if (find("depth_decoder.codebooks_head.weight", s)) {
            w.dp_heads = ggml_new_tensor_3d(c, GGML_TYPE_F16, kDHid, kCodebookVocab, kNumCodebooks - 1);
            ggml_set_name(w.dp_heads, "dp_heads");
            ggml_tensor * t = w.dp_heads;
            ups.push_back({t, [s, t](const gguf_file & f, std::string & er) {
                std::vector<uint16_t> src(s.size / 2), dst(s.size / 2);
                if (!f.read(s.offset, src.data(), s.size)) { er = "short read"; return false; }
                for (int cb = 0; cb < kNumCodebooks - 1; cb++)
                    for (int i = 0; i < kDHid; i++)
                        for (int v = 0; v < kCodebookVocab; v++)
                            dst[((size_t) cb * kCodebookVocab + v) * kDHid + i] =
                                src[((size_t) cb * kDHid + i) * kCodebookVocab + v];
                ggml_backend_tensor_set(t, dst.data(), 0, s.size);
                return true;
            }});
        }
    }
    for (int il = 0; il < kTLayers && ok; il++) {
        const std::string p = "text_encoder.layers." + std::to_string(il);
        te_layer L;
        L.pre_attn = norm(p + ".pre_self_attn_layernorm.weight", true);
        L.wqkv = pack({p + ".self_attn.q_proj.weight", p + ".self_attn.k_proj.weight", p + ".self_attn.v_proj.weight"},
                      p + ".wqkv");
        L.q_norm = norm(p + ".self_attn.q_norm.weight", true);
        L.k_norm = norm(p + ".self_attn.k_norm.weight", true);
        L.wo = mat(p + ".self_attn.o_proj.weight");
        L.post_attn = norm(p + ".post_self_attn_layernorm.weight", true);
        L.pre_ff = norm(p + ".pre_feedforward_layernorm.weight", true);
        L.w_gate_up = pack({p + ".mlp.gate_proj.weight", p + ".mlp.up_proj.weight"}, p + ".w_gate_up");
        L.w_down = mat(p + ".mlp.down_proj.weight");
        L.post_ff = norm(p + ".post_feedforward_layernorm.weight", true);
        w.te.push_back(L);
    }
    w.te_embed = mat("text_encoder.embed_tokens.weight");
    w.te_norm = norm("text_encoder.norm.weight", true);
    w.te_proj = mat("text_encoder_proj.weight");
    if (!ok) return false;

    w.buf = ggml_backend_alloc_ctx_tensors(c, backend);
    if (!w.buf) {
        e = "failed to allocate weights on " + std::string(ggml_backend_name(backend));
        return false;
    }
    ggml_backend_buffer_set_usage(w.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const auto t0 = clk::now();
    for (auto & u : ups) {
        if (!u.run(gf, e)) {
            if (e.empty()) e = std::string("failed to load ") + ggml_get_name(u.dst);
            return false;
        }
    }
    // host copy of the audio embedding table (clone prompts sum rows of it)
    {
        src_tensor s;
        if (!gf.find("model/depth_decoder.model.embed_tokens.weight", s)) {
            e = "missing audio embedding table";
            return false;
        }
        w.audio_emb_host.resize(s.size / 2);
        if (!gf.read(s.offset, w.audio_emb_host.data(), s.size)) {
            e = "short read (audio embedding)";
            return false;
        }
    }
    if (lora) fprintf(stderr, "breeze-tts: merged %zu LoRA matrices\n", lora->merged);
    fprintf(stderr, "breeze-tts: weights loaded in %lld ms\n", (long long) ms_since(t0));

    // ---- kv caches ---------------------------------------------------------------
    ggml_init_params cp = {ggml_tensor_overhead() * 256, nullptr, true};
    cache_ctx = ggml_init(cp);
    for (int il = 0; il < kLayers; il++) {
        bb_k.push_back(ggml_new_tensor_2d(cache_ctx, GGML_TYPE_F16, kHeadDim * kKvHeads, n_ctx));
        bb_v.push_back(ggml_new_tensor_3d(cache_ctx, GGML_TYPE_F16, kHeadDim, kKvHeads, n_ctx));
    }
    for (int il = 0; il < kDLayers; il++) {
        dp_k.push_back(ggml_new_tensor_2d(cache_ctx, GGML_TYPE_F16, kHeadDim * kDKvHeads, kDCtx));
        dp_v.push_back(ggml_new_tensor_3d(cache_ctx, GGML_TYPE_F16, kHeadDim, kDKvHeads, kDCtx));
    }
    cache_buf = ggml_backend_alloc_ctx_tensors(cache_ctx, backend);
    if (!cache_buf) {
        e = "failed to allocate kv cache";
        return false;
    }
    ggml_backend_buffer_clear(cache_buf, 0);

    // ---- vocoder -----------------------------------------------------------------
    if (!decoder.load_model(codec)) {
        e = "failed to load codec: " + decoder.get_error();
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// text encoder (T5Gemma2, bidirectional, one pass per segment)
// ---------------------------------------------------------------------------

bool engine_impl::text_encode(const std::vector<int32_t> & ids, std::vector<float> & out, std::string & e) {
    const int n = (int) ids.size();
    graph g;
    g.init(backend, 8192);
    ggml_context * c = g.ctx;
    ggml_tensor * tokens = input(c, GGML_TYPE_I32, "tokens", n);
    ggml_tensor * pos = input(c, GGML_TYPE_I32, "pos", n);
    ggml_tensor * x = ggml_get_rows(c, w.te_embed, tokens);
    x = ggml_scale(c, x, sqrtf((float) kTHid));
    for (int il = 0; il < kTLayers; il++) {
        const te_layer & L = w.te[il];
        const bool full = il % 6 == 5;
        const float theta = full ? 1000000.f : 10000.f;
        const float fscale = full ? 1.0f / 8.0f : 1.0f;
        ggml_tensor * h = rms_w(c, x, L.pre_attn, kTRmsEps);
        ggml_tensor * qkv = ggml_mul_mat(c, L.wqkv, h);                // [1536, n]
        ggml_mul_mat_set_prec(qkv, GGML_PREC_F32);
        ggml_tensor * q = slice_cols(c, qkv, kTHeads * kTHeadDim, n, 0);
        ggml_tensor * k = slice_cols(c, qkv, kTHeadDim, n, kTHeads * kTHeadDim);
        ggml_tensor * v = slice_cols(c, qkv, kTHeadDim, n, kTHeads * kTHeadDim + kTHeadDim);
        q = ggml_reshape_3d(c, q, kTHeadDim, kTHeads, n);
        k = ggml_reshape_3d(c, k, kTHeadDim, 1, n);
        v = ggml_reshape_3d(c, v, kTHeadDim, 1, n);
        q = rms_w(c, q, L.q_norm, kTRmsEps);
        k = rms_w(c, k, L.k_norm, kTRmsEps);
        q = ggml_rope_ext(c, q, pos, nullptr, kTHeadDim, GGML_ROPE_TYPE_NEOX, 0, theta, fscale, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(c, k, pos, nullptr, kTHeadDim, GGML_ROPE_TYPE_NEOX, 0, theta, fscale, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * Q = ggml_permute(c, q, 0, 2, 1, 3);              // [256, n, 4]
        ggml_tensor * K = ggml_permute(c, k, 0, 2, 1, 3);              // [256, n, 1]
        ggml_tensor * Vt = ggml_cont(c, ggml_transpose(c, ggml_permute(c, v, 0, 2, 1, 3)));  // [n, 256, 1]
        ggml_tensor * kq = ggml_mul_mat(c, K, Q);                      // [n_k, n_q, 4]
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(c, kq, nullptr, 1.0f / sqrtf((float) kTHeadDim), 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(c, Vt, kq);                   // [256, n_q, 4]
        ggml_mul_mat_set_prec(kqv, GGML_PREC_F32);
        kqv = ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), kTHeads * kTHeadDim, n);
        ggml_tensor * o = ggml_mul_mat(c, L.wo, kqv);
        ggml_mul_mat_set_prec(o, GGML_PREC_F32);
        o = rms_w(c, o, L.post_attn, kTRmsEps);
        x = ggml_add(c, x, o);
        h = rms_w(c, x, L.pre_ff, kTRmsEps);
        ggml_tensor * gu = ggml_mul_mat(c, L.w_gate_up, h);
        ggml_mul_mat_set_prec(gu, GGML_PREC_F32);
        ggml_tensor * act = ggml_geglu(c, gu);
        ggml_tensor * d = ggml_mul_mat(c, L.w_down, act);
        ggml_mul_mat_set_prec(d, GGML_PREC_F32);
        d = rms_w(c, d, L.post_ff, kTRmsEps);
        x = ggml_add(c, x, d);
    }
    x = rms_w(c, x, w.te_norm, kTRmsEps);
    ggml_tensor * outp = ggml_mul_mat(c, w.te_proj, x);                // [2048, n]
    named(outp, "out", true);
    ggml_build_forward_expand(g.gf, outp);
    if (!g.alloc()) {
        e = "text encoder graph allocation failed";
        return false;
    }
    std::vector<int32_t> p(n);
    for (int i = 0; i < n; i++) p[i] = i;
    ggml_backend_tensor_set(tokens, ids.data(), 0, n * sizeof(int32_t));
    ggml_backend_tensor_set(pos, p.data(), 0, n * sizeof(int32_t));
    if (!g.run()) {
        e = "text encoder compute failed";
        return false;
    }
    out.resize((size_t) n * kHid);
    ggml_backend_tensor_get(outp, out.data(), 0, out.size() * sizeof(float));
    return true;
}

// ---------------------------------------------------------------------------
// backbone
// ---------------------------------------------------------------------------

static const dec_dims kBbDims = {kHid, kHeads, kKvHeads, kHeadDim, kInter, 1000000.f, 1e-6f};
static const dec_dims kDpDims = {kDHid, kDHeads, kDKvHeads, kHeadDim, kDInter, 500000.f, 1e-5f};

static void causal_mask(std::vector<ggml_fp16_t> & m, int n_kv, int n, int n_past) {
    m.assign((size_t) n_kv * n, ggml_fp32_to_fp16(-INFINITY));
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    for (int q = 0; q < n; q++)
        for (int k = 0; k <= n_past + q && k < n_kv; k++) m[(size_t) q * n_kv + k] = zero;
}

bool engine_impl::backbone_prefill(const std::vector<float> & embd, int n, std::vector<float> & logits,
                                   std::vector<float> & hidden, std::string & e) {
    const int n_kv = std::min(n_ctx, (n + 255) / 256 * 256);
    graph g;
    g.init(backend, 8192);
    ggml_context * c = g.ctx;
    ggml_tensor * inp = input(c, GGML_TYPE_F32, "embd", kHid, n);
    ggml_tensor * pos = input(c, GGML_TYPE_I32, "pos", n);
    ggml_tensor * pos64 = input(c, GGML_TYPE_I64, "pos64", n);
    ggml_tensor * mask = input(c, GGML_TYPE_F16, "mask", n_kv, n);
    ggml_tensor * x = inp;
    for (int il = 0; il < kLayers; il++) {
        const bb_layer & L = w.bb[il];
        x = decoder_layer(c, g.gf, kBbDims, x, n, L.attn_norm, L.wqkv, L.q_norm, L.k_norm, L.wo, L.ffn_norm,
                          L.w_gate_up, L.w_down, nullptr, pos, pos64, bb_k[il], bb_v[il], n_kv, mask);
    }
    ggml_tensor * last = ggml_view_2d(c, x, kHid, 1, x->nb[1], (size_t) (n - 1) * x->nb[1]);
    ggml_tensor * hid = named(rms_w(c, last, w.bb_norm, kBbDims.eps), "hidden", true);
    ggml_tensor * lg = named(ggml_mul_mat(c, w.lm_head, hid), "logits", true);
    ggml_build_forward_expand(g.gf, hid);
    ggml_build_forward_expand(g.gf, lg);
    if (!g.alloc()) {
        e = "backbone prefill graph allocation failed";
        return false;
    }
    std::vector<int32_t> p(n);
    for (int i = 0; i < n; i++) p[i] = i;
    std::vector<ggml_fp16_t> m;
    causal_mask(m, n_kv, n, 0);
    ggml_backend_tensor_set(inp, embd.data(), 0, embd.size() * sizeof(float));
    ggml_backend_tensor_set(pos, p.data(), 0, n * sizeof(int32_t));
    std::vector<int64_t> p64(p.begin(), p.end());
    ggml_backend_tensor_set(pos64, p64.data(), 0, n * sizeof(int64_t));
    ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(ggml_fp16_t));
    if (!g.run()) {
        e = "backbone prefill compute failed";
        return false;
    }
    logits.resize(kLmHead);
    hidden.resize(kHid);
    ggml_backend_tensor_get(lg, logits.data(), 0, kLmHead * sizeof(float));
    ggml_backend_tensor_get(hid, hidden.data(), 0, kHid * sizeof(float));
    return true;
}

graph * engine_impl::get_bb_step(int n_past) {
    // smallest kv bucket (multiple of 256) holding n_past + 1 entries
    const int need = n_past + 1;
    const int n_kv = std::min(n_ctx, (need + 255) / 256 * 256);
    for (size_t i = 0; i < bb_step.size(); i++) {
        if (bb_step_kv[i] == n_kv) return bb_step[i].get();
    }
    auto g = std::make_unique<graph>();
    g->init(backend, 8192);
    ggml_context * c = g->ctx;
    ggml_tensor * rows = input(c, GGML_TYPE_I32, "rows", kNumCodebooks);
    ggml_tensor * pos = input(c, GGML_TYPE_I32, "pos", 1);
    ggml_tensor * pos64 = input(c, GGML_TYPE_I64, "pos64", 1);
    ggml_tensor * mask = input(c, GGML_TYPE_F16, "mask", n_kv, 1);
    // frame embedding = sum over the 16 codebook rows of the shared table
    ggml_tensor * er = ggml_get_rows(c, w.audio_emb, rows);                       // [2048, 16]
    ggml_tensor * x = ggml_reshape_2d(c, ggml_sum_rows(c, ggml_cont(c, ggml_transpose(c, er))), kHid, 1);
    for (int il = 0; il < kLayers; il++) {
        const bb_layer & L = w.bb[il];
        x = decoder_layer(c, g->gf, kBbDims, x, 1, L.attn_norm, L.wqkv, L.q_norm, L.k_norm, L.wo, L.ffn_norm,
                          L.w_gate_up, L.w_down, nullptr, pos, pos64, bb_k[il], bb_v[il], n_kv, mask);
    }
    ggml_tensor * hid = named(rms_w(c, x, w.bb_norm, kBbDims.eps), "hidden", true);
    ggml_tensor * lg = named(ggml_mul_mat(c, w.lm_head, hid), "logits", true);
    ggml_build_forward_expand(g->gf, hid);
    ggml_build_forward_expand(g->gf, lg);
    if (!g->alloc()) return nullptr;
    bb_step_kv.push_back(n_kv);
    bb_step.push_back(std::move(g));
    return bb_step.back().get();
}

bool engine_impl::backbone_step(const int32_t * rows, int n_past, std::vector<float> & logits,
                                std::vector<float> & hidden, std::string & e) {
    graph * g = get_bb_step(n_past);
    if (!g) {
        e = "backbone step graph allocation failed";
        return false;
    }
    int n_kv = 0;
    for (size_t i = 0; i < bb_step.size(); i++)
        if (bb_step[i].get() == g) n_kv = bb_step_kv[i];
    std::vector<ggml_fp16_t> m;
    causal_mask(m, n_kv, 1, n_past);
    const int32_t p = n_past;
    ggml_backend_tensor_set(g->get("rows"), rows, 0, kNumCodebooks * sizeof(int32_t));
    ggml_backend_tensor_set(g->get("pos"), &p, 0, sizeof(int32_t));
    const int64_t p64 = n_past;
    ggml_backend_tensor_set(g->get("pos64"), &p64, 0, sizeof(int64_t));
    ggml_backend_tensor_set(g->get("mask"), m.data(), 0, m.size() * sizeof(ggml_fp16_t));
    if (!g->run()) {
        e = "backbone step compute failed";
        return false;
    }
    logits.resize(kLmHead);
    hidden.resize(kHid);
    ggml_backend_tensor_get(g->get("logits"), logits.data(), 0, kLmHead * sizeof(float));
    ggml_backend_tensor_get(g->get("hidden"), hidden.data(), 0, kHid * sizeof(float));
    return true;
}

// ---------------------------------------------------------------------------
// depth decoder: prefill [backbone hidden, first-code embedding], then one step per codebook
// ---------------------------------------------------------------------------

bool engine_impl::build_depth_graphs() {
    {   // prefill: positions 0 and 1
        graph & g = dp_prefill;
        g.init(backend, 4096);
        ggml_context * c = g.ctx;
        ggml_tensor * hid = input(c, GGML_TYPE_F32, "hidden", kHid, 1);
        ggml_tensor * row = input(c, GGML_TYPE_I32, "row", 1);
        ggml_tensor * pos = input(c, GGML_TYPE_I32, "pos", 2);
        ggml_tensor * pos64 = input(c, GGML_TYPE_I64, "pos64", 2);
        ggml_tensor * mask = input(c, GGML_TYPE_F16, "mask", kDCtx, 2);
        ggml_tensor * e0 = ggml_mul_mat(c, w.dp_proj, hid);                               // [1024, 1]
        ggml_tensor * e1 = ggml_mul_mat(c, w.dp_proj, ggml_get_rows(c, w.audio_emb, row));  // [1024, 1]
        ggml_tensor * x = ggml_concat(c, e0, e1, 1);                                      // [1024, 2]
        for (int il = 0; il < kDLayers; il++) {
            const dp_layer & L = w.dp[il];
            x = decoder_layer(c, g.gf, kDpDims, x, 2, L.attn_norm, L.wqkv, nullptr, nullptr, L.wo, L.ffn_norm,
                              L.w_gate_up, L.w_down, w.dp_rope, pos, pos64, dp_k[il], dp_v[il], kDCtx, mask);
        }
        ggml_tensor * last = ggml_view_2d(c, x, kDHid, 1, x->nb[1], x->nb[1]);
        ggml_tensor * h = rms_w(c, last, w.dp_norm, kDpDims.eps);
        ggml_tensor * head0 = ggml_view_2d(c, w.dp_heads, kDHid, kCodebookVocab, w.dp_heads->nb[1], 0);
        ggml_tensor * lg = named(ggml_mul_mat(c, head0, h), "logits", true);
        ggml_build_forward_expand(g.gf, lg);
        if (!g.alloc()) return false;
    }
    {   // step: one token at positions 2..16, head chosen by an id input
        graph & g = dp_step;
        g.init(backend, 4096);
        ggml_context * c = g.ctx;
        ggml_tensor * row = input(c, GGML_TYPE_I32, "row", 1);
        ggml_tensor * pos = input(c, GGML_TYPE_I32, "pos", 1);
        ggml_tensor * pos64 = input(c, GGML_TYPE_I64, "pos64", 1);
        ggml_tensor * head = input(c, GGML_TYPE_I32, "head", 1, 1);
        ggml_tensor * mask = input(c, GGML_TYPE_F16, "mask", kDCtx, 1);
        ggml_tensor * x = ggml_mul_mat(c, w.dp_proj, ggml_get_rows(c, w.audio_emb, row));  // [1024, 1]
        for (int il = 0; il < kDLayers; il++) {
            const dp_layer & L = w.dp[il];
            x = decoder_layer(c, g.gf, kDpDims, x, 1, L.attn_norm, L.wqkv, nullptr, nullptr, L.wo, L.ffn_norm,
                              L.w_gate_up, L.w_down, w.dp_rope, pos, pos64, dp_k[il], dp_v[il], kDCtx, mask);
        }
        ggml_tensor * h = rms_w(c, x, w.dp_norm, kDpDims.eps);
        ggml_tensor * h3 = ggml_reshape_3d(c, h, kDHid, 1, 1);
        ggml_tensor * ids = ggml_reshape_2d(c, head, 1, 1);
        ggml_tensor * lg = named(ggml_reshape_1d(c, ggml_mul_mat_id(c, w.dp_heads, h3, ids), kCodebookVocab),
                                 "logits", true);
        ggml_build_forward_expand(g.gf, lg);
        if (!g.alloc()) return false;
    }
    depth_ready = true;
    return true;
}

// sum of the 16 codebook rows of the shared embedding table for one frame
void engine_impl::frame_embedding(const int32_t * codes, float * out) {
    std::fill(out, out + kHid, 0.0f);
    for (int cb = 0; cb < kNumCodebooks; cb++) {
        const uint16_t * row = w.audio_emb_host.data() + (size_t) (cb * kCodebookVocab + codes[cb]) * kHid;
        for (int i = 0; i < kHid; i++) out[i] += ggml_fp16_to_fp32(row[i]);
    }
}

// ---------------------------------------------------------------------------
// synthesis
// ---------------------------------------------------------------------------

tts_result engine_impl::synthesize(const std::string & text, const tts_params & p, const reference_voice * ref) {
    tts_result r;
    const auto t_start = clk::now();
    std::string e;
    auto fail = [&](const std::string & m) {
        r.success = false;
        r.error = m;
        return r;
    };
    if (text.empty()) return fail("empty input text");

    // ---- prompt ---------------------------------------------------------------
    const std::string instr = p.instruction.empty() ? "Speak clearly and naturally." : p.instruction;
    auto t0 = clk::now();
    std::vector<int32_t> seg_instr = tok.encode("<bos>[S0]<ins_bos>" + instr + "<ins_eos>" + text);
    std::vector<int32_t> seg_ref;
    if (ref) seg_ref = tok.encode("<bos>[S0]" + ref->text);
    r.t_tokenize_ms = ms_since(t0);
    r.n_text_tokens = (int32_t) seg_instr.size();

    t0 = clk::now();
    std::vector<float> embd;   // [n_prompt, 2048] row-major
    std::vector<float> enc;
    if (ref) {
        if (!text_encode(seg_ref, enc, e)) return fail(e);
        embd.insert(embd.end(), enc.begin(), enc.end());
        const size_t base = embd.size();
        embd.resize(base + (size_t) (ref->n_frames + 1) * kHid);
        for (int f = 0; f < ref->n_frames; f++) {
            frame_embedding(ref->codes.data() + (size_t) f * kNumCodebooks, embd.data() + base + (size_t) f * kHid);
        }
        int32_t eos_codes[kNumCodebooks] = {0};
        frame_embedding(eos_codes, embd.data() + base + (size_t) ref->n_frames * kHid);
    }
    if (!text_encode(seg_instr, enc, e)) return fail(e);
    embd.insert(embd.end(), enc.begin(), enc.end());
    r.t_text_encode_ms = ms_since(t0);

    const int n_prompt = (int) (embd.size() / kHid);
    r.n_prompt_tokens = n_prompt;
    const int max_frames = std::min<int>(p.max_frames, n_ctx - n_prompt - 1);
    if (max_frames <= 0) return fail("prompt too long for the context window");

    // ---- backbone prefill ---------------------------------------------------------
    t0 = clk::now();
    ggml_backend_buffer_clear(cache_buf, 0);
    std::vector<float> logits, hidden;
    if (!backbone_prefill(embd, n_prompt, logits, hidden, e)) return fail(e);
    r.t_prefill_ms = ms_since(t0);
    if (!depth_ready && !build_depth_graphs()) return fail("depth decoder graph allocation failed");

    // ---- autoregressive loop ------------------------------------------------------
    std::mt19937_64 rng(p.seed >= 0 ? (uint64_t) p.seed : std::random_device{}());
    std::vector<int32_t> codes;               // frame-major
    std::vector<int32_t> history;             // first-codebook tokens
    std::vector<float> dl;
    std::vector<ggml_fp16_t> dmask;
    double t_bb = 0, t_dp = 0;
    t0 = clk::now();
    int n_past = n_prompt;
    for (int step = 0; step < max_frames; step++) {
        // first codebook
        for (int t = kCodecSize; t < kCodebookVocab; t++) logits[t] = -INFINITY;
        const int32_t t0c = sample_token(logits, &history, p.repetition_penalty, p.temperature, p.top_k, p.top_p, rng);
        if (t0c == kEosToken) break;
        history.push_back(t0c);
        int32_t frame[kNumCodebooks];
        frame[0] = t0c;

        const auto td = clk::now();
        {   // depth prefill: [hidden, embed(t0)]
            const int32_t row = t0c;
            const int32_t pos2[2] = {0, 1};
            causal_mask(dmask, kDCtx, 2, 0);
            ggml_backend_tensor_set(dp_prefill.get("hidden"), hidden.data(), 0, kHid * sizeof(float));
            ggml_backend_tensor_set(dp_prefill.get("row"), &row, 0, sizeof(row));
            ggml_backend_tensor_set(dp_prefill.get("pos"), pos2, 0, sizeof(pos2));
            const int64_t pos2_64[2] = {pos2[0], pos2[1]};
            ggml_backend_tensor_set(dp_prefill.get("pos64"), pos2_64, 0, sizeof(pos2_64));
            ggml_backend_tensor_set(dp_prefill.get("mask"), dmask.data(), 0, dmask.size() * sizeof(ggml_fp16_t));
            if (!dp_prefill.run()) return fail("depth prefill compute failed");
            dl.resize(kCodebookVocab);
            ggml_backend_tensor_get(dp_prefill.get("logits"), dl.data(), 0, kCodebookVocab * sizeof(float));
        }
        for (int cb = 1; cb < kNumCodebooks; cb++) {
            for (int t = kCodecSize; t < kCodebookVocab; t++) dl[t] = -INFINITY;
            const int32_t tok_cb = sample_token(dl, nullptr, 1.0f, p.depth_temperature, p.top_k, p.top_p, rng);
            frame[cb] = tok_cb;
            if (cb + 1 < kNumCodebooks) {
                const int32_t row = cb * kCodebookVocab + tok_cb;
                const int32_t pos = 1 + cb;
                const int32_t head = cb;      // logits for codebook cb+1 come from head slice cb
                causal_mask(dmask, kDCtx, 1, pos);
                ggml_backend_tensor_set(dp_step.get("row"), &row, 0, sizeof(row));
                ggml_backend_tensor_set(dp_step.get("pos"), &pos, 0, sizeof(pos));
                const int64_t pos64 = pos;
                ggml_backend_tensor_set(dp_step.get("pos64"), &pos64, 0, sizeof(pos64));
                ggml_backend_tensor_set(dp_step.get("head"), &head, 0, sizeof(head));
                ggml_backend_tensor_set(dp_step.get("mask"), dmask.data(), 0, dmask.size() * sizeof(ggml_fp16_t));
                if (!dp_step.run()) return fail("depth step compute failed");
                ggml_backend_tensor_get(dp_step.get("logits"), dl.data(), 0, kCodebookVocab * sizeof(float));
            }
        }
        t_dp += (double) std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - td).count() / 1000.0;
        codes.insert(codes.end(), frame, frame + kNumCodebooks);

        int32_t rows[kNumCodebooks];
        for (int cb = 0; cb < kNumCodebooks; cb++) rows[cb] = cb * kCodebookVocab + frame[cb];
        const auto tb = clk::now();
        if (!backbone_step(rows, n_past, logits, hidden, e)) return fail(e);
        t_bb += (double) std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - tb).count() / 1000.0;
        n_past++;
    }
    r.t_generate_ms = ms_since(t0);
    r.n_frames = (int32_t) (codes.size() / kNumCodebooks);
    if (r.n_frames == 0) return fail("no audio generated");
    if (timing) {
        fprintf(stderr, "breeze-tts timing: prompt=%d tok=%lldms enc=%lldms prefill=%lldms | %d frames in %lldms "
                        "(backbone %.0fms depth %.0fms, %.1f ms/frame)\n",
                n_prompt, (long long) r.t_tokenize_ms, (long long) r.t_text_encode_ms, (long long) r.t_prefill_ms,
                r.n_frames, (long long) r.t_generate_ms, t_bb, t_dp, (double) r.t_generate_ms / r.n_frames);
    }

    if (const char * dbg = std::getenv("BREEZE_TTS_DEBUG_DIR")) {
        if (FILE * f = fopen((std::string(dbg) + "/last_codes.bin").c_str(), "wb")) {
            fwrite(codes.data(), sizeof(int32_t), codes.size(), f);
            fclose(f);
        }
    }

    r.codes = codes;

    // ---- vocoder ------------------------------------------------------------------
    t0 = clk::now();
    const int32_t ctx_frames = (int32_t) (p.vocoder_context.size() / kNumCodebooks);
    std::vector<int32_t> voc_codes = p.vocoder_context;
    voc_codes.insert(voc_codes.end(), codes.begin(), codes.end());
    if (!decoder.decode(voc_codes.data(), r.n_frames + ctx_frames, r.audio)) {
        return fail("vocoder failed: " + decoder.get_error());
    }
    if (ctx_frames > 0) {
        const size_t drop = (size_t) ctx_frames * (kSampleRate / 12.5);
        r.audio.erase(r.audio.begin(), r.audio.begin() + std::min(drop, r.audio.size()));
    }
    for (float & s : r.audio) s = std::max(-1.0f, std::min(1.0f, s));
    r.t_decode_ms = ms_since(t0);
    r.t_total_ms = ms_since(t_start);
    r.success = true;
    return r;
}

// ---------------------------------------------------------------------------
// public api
// ---------------------------------------------------------------------------

engine::engine() : impl_(new engine_impl()) {}
engine::~engine() = default;

void engine::set_lora(const std::string & path, float strength) {
    impl_->lora_path = path;
    impl_->lora_strength = strength;
}

bool engine::load(const std::string & model_path, const std::string & codec_path, int n_ctx, std::string & err) {
    return impl_->load(model_path, codec_path, n_ctx, err);
}

tts_result engine::synthesize(const std::string & text, const tts_params & params, const reference_voice * ref) {
    return impl_->synthesize(text, params, ref);
}

bool engine::make_reference(const std::vector<float> & samples, const std::string & text,
                            reference_voice & out, std::string & err) {
    auto & d = *impl_;
    if (!d.codec_enc_loaded) {
        if (!d.codec_enc.load_model(d.codec_path)) {
            err = "failed to load codec encoder: " + d.codec_enc.get_error();
            return false;
        }
        d.codec_enc_loaded = true;
    }
    std::vector<int32_t> codes;
    int32_t n_frames = 0;
    if (!d.codec_enc.encode(samples.data(), (int32_t) samples.size(), codes, n_frames)) {
        err = "failed to encode reference audio: " + d.codec_enc.get_error();
        return false;
    }
    if (const char * dbg = std::getenv("BREEZE_TTS_DEBUG_DIR")) {
        // debugging aid: substitute externally computed reference codes
        if (FILE * f = fopen((std::string(dbg) + "/override_codes.bin").c_str(), "rb")) {
            fseek(f, 0, SEEK_END);
            const long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            codes.resize(sz / sizeof(int32_t));
            if (fread(codes.data(), sizeof(int32_t), codes.size(), f) != codes.size()) codes.clear();
            fclose(f);
            n_frames = (int32_t) (codes.size() / kNumCodebooks);
        }
        if (FILE * f = fopen((std::string(dbg) + "/ref_codes.bin").c_str(), "wb")) {
            fwrite(codes.data(), sizeof(int32_t), codes.size(), f);
            fclose(f);
        }
        // round trip: the reference codes through the vocoder should sound like the reference
        std::vector<float> back;
        if (d.decoder.decode(codes.data(), n_frames, back)) {
            qwen3_tts::save_audio_file(std::string(dbg) + "/ref_roundtrip.wav", back, kSampleRate);
        }
    }
    out.text = text;
    out.codes = std::move(codes);
    out.n_frames = n_frames;
    return true;
}

const std::string & engine::error() const { return impl_->err; }

}  // namespace breeze_tts
