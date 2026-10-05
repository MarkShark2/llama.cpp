#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace breeze_tts {

constexpr int kNumCodebooks = 16;
constexpr int kCodebookVocab = 2051;   // 2048 codes + pad/eos/reserved
constexpr int kCodecSize = 2048;       // codes the vocoder understands
constexpr int kSampleRate = 24000;

struct tts_params {
    std::string instruction;           // voice / style description
    bool plain_prompt = false;         // no instruction segment: the prompt a LoRA was trained on
    int32_t max_frames = 1500;         // 12.5 frames per second
    float   temperature = 0.9f;        // first codebook
    float   depth_temperature = 0.9f;  // codebooks 1..15
    int32_t top_k = 50;
    float   top_p = 1.0f;
    float   repetition_penalty = 1.1f; // first codebook history only
    int64_t seed = -1;                 // < 0: nondeterministic
    // codes (frames * 16) of audio that precedes this utterance; the vocoder decodes them as left
    // context and drops their samples, so consecutive chunks join without a seam
    std::vector<int32_t> vocoder_context;
};

// A reference voice for cloning: its transcript and its 16-codebook codes.
struct reference_voice {
    std::string text;
    std::vector<int32_t> codes;        // frames * 16, frame-major
    int32_t n_frames = 0;
};

struct tts_result {
    std::vector<float> audio;          // 24 kHz mono
    std::vector<int32_t> codes;        // generated codes, frames * 16 (usable as a reference as is)
    bool success = false;
    std::string error;
    int32_t n_text_tokens = 0;
    int32_t n_prompt_tokens = 0;
    int32_t n_frames = 0;
    int64_t t_tokenize_ms = 0;
    int64_t t_text_encode_ms = 0;
    int64_t t_prefill_ms = 0;
    int64_t t_generate_ms = 0;         // autoregressive loop
    int64_t t_decode_ms = 0;           // vocoder
    int64_t t_total_ms = 0;
};

// Gemma-style BPE (T5Gemma2 vocabulary) read from the tokenizer.json embedded in the GGUF.
class bpe_tokenizer {
public:
    bool load(const std::string & tokenizer_json, std::string & err);
    std::vector<int32_t> encode(const std::string & text) const;
    bool token_id(const std::string & s, int32_t & id) const;

private:
    void encode_plain(const std::string & text, std::vector<int32_t> & out) const;

    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<std::string, int32_t> ranks_;   // "left\0right" -> rank
    std::vector<std::pair<std::string, int32_t>> specials_;  // longest first
    int32_t byte_tokens_[256];
    int32_t unk_ = 3;
};

struct engine_impl;

class engine {
public:
    engine();
    ~engine();

    // a convert_lora.py adapter merged into the weights at load; call before load().
    // strength multiplies the trained alpha/rank scale
    void set_lora(const std::string & path, float strength);

    // model: the audio.cpp Breeze-TTS-2 GGUF; codec: qwen3-tts-tokenizer GGUF
    bool load(const std::string & model_path, const std::string & codec_path,
              int n_ctx, std::string & err);

    tts_result synthesize(const std::string & text, const tts_params & params,
                          const reference_voice * ref = nullptr);

    // encode 24 kHz mono audio into a reference voice
    bool make_reference(const std::vector<float> & samples, const std::string & text,
                        reference_voice & out, std::string & err);

    const std::string & error() const;

private:
    std::unique_ptr<engine_impl> impl_;
};

}  // namespace breeze_tts
