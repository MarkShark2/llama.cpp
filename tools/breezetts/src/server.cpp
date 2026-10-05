// openai-compatible tts server for Breeze-TTS 2
//
// endpoints:
//   GET    /health
//   GET    /v1/models
//   GET    /v1/audio/voices           - default + cloned voices
//   POST   /v1/audio/voices           - clone a voice (multipart: audio_sample + ref_text [+ name])
//   DELETE /v1/audio/voices/X
//   POST   /v1/audio/speech           - synthesize (wav, pcm, mp3/opus when libav is built in)
//
// A speech request's `voice` names a cloned voice or the server default voice; with neither,
// the voice is designed from `instruction` (request field, else the server default).

#include "breeze_tts.h"
#include "qwen3_tts.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <unistd.h>

using json = nlohmann::json;

static std::string encode_wav(const std::vector<float> & samples, int sample_rate) {
    const int data_size = (int) samples.size() * 2;
    std::string buf(44 + data_size, '\0');
    char * p = buf.data();
    auto u32 = [](char * d, uint32_t v) { for (int i = 0; i < 4; i++) d[i] = (char) ((v >> (8 * i)) & 0xff); };
    auto u16 = [](char * d, uint16_t v) { d[0] = (char) (v & 0xff); d[1] = (char) ((v >> 8) & 0xff); };
    memcpy(p, "RIFF", 4);      u32(p + 4, 36 + data_size);
    memcpy(p + 8, "WAVE", 4);
    memcpy(p + 12, "fmt ", 4); u32(p + 16, 16);
    u16(p + 20, 1); u16(p + 22, 1); u32(p + 24, sample_rate); u32(p + 28, sample_rate * 2);
    u16(p + 32, 2); u16(p + 34, 16);
    memcpy(p + 36, "data", 4); u32(p + 40, data_size);
    int16_t * dst = reinterpret_cast<int16_t *>(p + 44);
    for (size_t i = 0; i < samples.size(); i++) {
        float s = std::max(-1.0f, std::min(1.0f, samples[i]));
        dst[i] = (int16_t) (s * 32767.0f);
    }
    return buf;
}

static std::string encode_pcm(const std::vector<float> & samples) {
    std::string buf(samples.size() * 2, '\0');
    int16_t * dst = reinterpret_cast<int16_t *>(buf.data());
    for (size_t i = 0; i < samples.size(); i++) {
        float s = std::max(-1.0f, std::min(1.0f, samples[i]));
        dst[i] = (int16_t) (s * 32767.0f);
    }
    return buf;
}

static const char * content_type_for(const std::string & fmt) {
    if (fmt == "mp3")  return "audio/mpeg";
    if (fmt == "opus") return "audio/ogg";
    if (fmt == "pcm")  return "audio/pcm";
    return "audio/wav";
}

static std::string encode_format(const std::string & fmt, const std::vector<float> & samples, int sr) {
    qwen3_tts::audio_codec codec;
    if (qwen3_tts::codec_from_name(fmt, codec)) return qwen3_tts::encode_compressed(codec, samples, sr);
    if (fmt == "pcm") return encode_pcm(samples);
    return encode_wav(samples, sr);
}

static std::vector<float> to_24k(const std::vector<float> & in, int sr) {
    if (sr == breeze_tts::kSampleRate || sr <= 0) return in;
    const int64_t n = (int64_t) in.size() * breeze_tts::kSampleRate / sr;
    std::vector<float> out(n);
    for (int64_t i = 0; i < n; i++) {
        const float src = (float) i * sr / breeze_tts::kSampleRate;
        const int idx = (int) src;
        const float frac = src - idx;
        out[i] = idx + 1 < (int) in.size() ? in[idx] * (1 - frac) + in[idx + 1] * frac
                                          : in[std::min(idx, (int) in.size() - 1)];
    }
    return out;
}

// Split long input into chunks of at most max_chars bytes, never inside a sentence when the sentence
// fits: sentences (ending . ! ? or a newline) are packed greedily, and a sentence longer than a chunk
// is cut at clause punctuation, else at a space. Chunks stay near the clip length the model trains on
// (a few seconds to ~15 s); longer generations drift in voice and prosody.
static bool ends_unit(const std::string & s, size_t i, size_t n) {
    // s[i] is the last byte of a candidate end; true when a sentence ends here
    static const char * kEnds[] = {".", "!", "?", "\xE3\x80\x82", "\xEF\xBC\x81", "\xEF\xBC\x9F"};
    if (s[i] == '\n') return true;
    bool hit = false;
    for (const char * e : kEnds) {
        const size_t el = strlen(e);
        if (i + 1 >= el && s.compare(i + 1 - el, el, e) == 0) hit = true;
    }
    if (!hit) return false;
    size_t k = i + 1;
    while (k < n && (s[k] == '"' || s[k] == '\'' || s[k] == ')')) k++;   // closing quote / bracket belongs here
    return k >= n || s[k] == ' ' || s[k] == '\n';
}

static std::vector<std::string> split_text(const std::string & in, size_t max_chars) {
    // 1. sentence units
    std::vector<std::string> units;
    std::string cur;
    for (size_t i = 0; i < in.size(); i++) {
        cur += in[i];
        if (ends_unit(in, i, in.size())) {
            units.push_back(cur);
            cur.clear();
        }
    }
    units.push_back(cur);
    // 2. cut oversized units at clause punctuation (, ; : and the em dash), else at a space
    std::vector<std::string> pieces;
    for (auto & u : units) {
        std::string rest = u;
        while (rest.size() > max_chars) {
            size_t cut = std::string::npos;
            for (size_t i = 0; i < rest.size() && i < max_chars; i++) {
                const char c = rest[i];
                if ((c == ',' || c == ';' || c == ':') && i + 1 < rest.size() && rest[i + 1] == ' ') cut = i + 1;
                else if (c == '\xE2' && i + 2 < rest.size() && rest[i + 1] == '\x80' && rest[i + 2] == '\x94') cut = i + 3;
            }
            if (cut == std::string::npos || cut < max_chars / 3) {
                size_t sp = rest.rfind(' ', max_chars);
                cut = (sp == std::string::npos || sp < max_chars / 3) ? max_chars : sp + 1;
                while (cut < rest.size() && ((unsigned char) rest[cut] & 0xC0) == 0x80) cut++;   // not inside a UTF-8 sequence
            }
            pieces.push_back(rest.substr(0, cut));
            rest = rest.substr(cut);
        }
        pieces.push_back(rest);
    }
    // 3. pack pieces up to the budget
    std::vector<std::string> out;
    std::string chunk;
    auto flush = [&]() {
        const size_t b = chunk.find_first_not_of(" \t\r\n");
        if (b != std::string::npos) out.push_back(chunk.substr(b));
        chunk.clear();
    };
    for (auto & pc : pieces) {
        if (!chunk.empty() && chunk.size() + pc.size() > max_chars) flush();
        chunk += pc;
    }
    flush();
    if (out.empty()) out.push_back(in);
    return out;
}

static json error_json(const std::string & msg, const char * type) {
    return json{{"error", {{"message", msg}, {"type", type}}}};
}

struct server_params {
    std::string model, codec;
    std::string lora;
    bool plain_prompt = false;
    float lora_strength = 1.0f;
    std::string instruction = "Speak clearly and naturally.";
    std::string voice_audio, voice_text;
    int64_t anchor_seed = -1;          // >= 0: synthesize the default voice from the instruction
    std::string anchor_text = "Hello there! This is a short test of the Breeze text to speech model running on the Vega GPU.";
    bool continuity = true;
    std::string host = "127.0.0.1";
    int port = 8080;
    int n_ctx = 2048;
    float temperature = 0.9f;
    float depth_temperature = 0.9f;
    int top_k = 50;
    float repetition_penalty = 1.1f;
    int64_t seed = -1;
};

static void print_usage(const char * prog) {
    fprintf(stderr,
            "usage: %s -m <breeze.gguf> -v <qwen3-tts-tokenizer.gguf> [options]\n"
            "  -m, --model <file>          Breeze-TTS-2 GGUF\n"
            "  -v, --vocoder <file>        Qwen3-TTS tokenizer GGUF (codec)\n"
            "      --lora <file>           LoRA GGUF (convert_lora.py), merged into the weights at load\n"
            "      --plain-prompt          no instruction in the prompt (what a LoRA is trained on)\n"
            "      --lora-strength <f>     multiplier of the trained alpha/rank scale (default 1)\n"
            "  -H, --host <host>           listen host (default 127.0.0.1)\n"
            "  -p, --port <port>           listen port (default 8080)\n"
            "  -c, --ctx <n>               backbone context in frames/tokens (default 2048)\n"
            "      --instruction <text>    default voice/style description\n"
            "      --voice-audio <wav>     default cloned voice: reference audio\n"
            "      --voice-text <text>     default cloned voice: transcript of the reference\n"
            "      --anchor-seed <n>       default voice = the voice this seed draws for --instruction\n"
            "      --anchor-text <text>    what the anchor voice says (default: a test sentence)\n"
            "      --no-continuity         do not condition each chunk on the previous one\n"
            "      --temperature <f>       first-codebook temperature (default 0.9)\n"
            "      --depth-temperature <f> depth temperature (default 0.9)\n"
            "      --top-k <n>             top-k (default 50)\n"
            "      --repetition-penalty <f> (default 1.1)\n"
            "      --seed <n>              default seed (default random)\n",
            prog);
}

static bool parse_args(int argc, char ** argv, server_params & sp) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (++i >= argc) { fprintf(stderr, "error: missing %s\n", what); return nullptr; }
            return argv[i];
        };
        const char * v = nullptr;
        if (a == "-h" || a == "--help") { print_usage(argv[0]); exit(0); }
        else if (a == "-m" || a == "--model")        { if (!(v = next("model"))) return false; sp.model = v; }
        else if (a == "-v" || a == "--vocoder")      { if (!(v = next("vocoder"))) return false; sp.codec = v; }
        else if (a == "--lora")                      { if (!(v = next("lora"))) return false; sp.lora = v; }
        else if (a == "--plain-prompt")              { sp.plain_prompt = true; }
        else if (a == "--lora-strength")             { if (!(v = next("lora-strength"))) return false; sp.lora_strength = std::stof(v); }
        else if (a == "-H" || a == "--host")         { if (!(v = next("host"))) return false; sp.host = v; }
        else if (a == "-p" || a == "--port")         { if (!(v = next("port"))) return false; sp.port = std::stoi(v); }
        else if (a == "-c" || a == "--ctx")          { if (!(v = next("ctx"))) return false; sp.n_ctx = std::stoi(v); }
        else if (a == "-j" || a == "--threads")      { if (!(v = next("threads"))) return false; }
        else if (a == "--instruction")               { if (!(v = next("instruction"))) return false; sp.instruction = v; }
        else if (a == "--voice-audio")               { if (!(v = next("voice-audio"))) return false; sp.voice_audio = v; }
        else if (a == "--voice-text")                { if (!(v = next("voice-text"))) return false; sp.voice_text = v; }
        else if (a == "--anchor-seed")               { if (!(v = next("anchor-seed"))) return false; sp.anchor_seed = std::stoll(v); }
        else if (a == "--anchor-text")               { if (!(v = next("anchor-text"))) return false; sp.anchor_text = v; }
        else if (a == "--no-continuity")             { sp.continuity = false; }
        else if (a == "--temperature")               { if (!(v = next("temperature"))) return false; sp.temperature = std::stof(v); }
        else if (a == "--depth-temperature")         { if (!(v = next("depth-temperature"))) return false; sp.depth_temperature = std::stof(v); }
        else if (a == "--top-k")                     { if (!(v = next("top-k"))) return false; sp.top_k = std::stoi(v); }
        else if (a == "--repetition-penalty")        { if (!(v = next("repetition-penalty"))) return false; sp.repetition_penalty = std::stof(v); }
        else if (a == "--seed")                      { if (!(v = next("seed"))) return false; sp.seed = std::stoll(v); }
        else { fprintf(stderr, "error: unknown argument: %s\n", a.c_str()); return false; }
    }
    if (sp.model.empty() || sp.codec.empty()) {
        fprintf(stderr, "error: -m and -v are required\n");
        return false;
    }
    return true;
}

static bool load_reference(breeze_tts::engine & eng, const std::string & wav_path, const std::string & text,
                           breeze_tts::reference_voice & out, std::string & err) {
    std::vector<float> samples;
    int sr = 0;
    if (!qwen3_tts::load_audio_file(wav_path, samples, sr)) {
        err = "failed to read reference audio (wav only): " + wav_path;
        return false;
    }
    return eng.make_reference(to_24k(samples, sr), text, out, err);
}

int main(int argc, char ** argv) {
    server_params sp;
    if (!parse_args(argc, argv, sp)) {
        print_usage(argv[0]);
        return 1;
    }

    breeze_tts::engine eng;
    fprintf(stderr, "loading model: %s\n", sp.model.c_str());
    std::string err;
    if (!sp.lora.empty()) eng.set_lora(sp.lora, sp.lora_strength);
    if (!eng.load(sp.model, sp.codec, sp.n_ctx, err)) {
        fprintf(stderr, "fatal: %s\n", err.c_str());
        return 1;
    }

    std::string model_id = sp.model;
    if (auto s = model_id.find_last_of("/\\"); s != std::string::npos) model_id = model_id.substr(s + 1);
    if (auto d = model_id.rfind('.'); d != std::string::npos) model_id = model_id.substr(0, d);

    std::mutex synth_mutex;
    std::map<std::string, breeze_tts::reference_voice> voices;
    std::mutex voices_mutex;
    int next_voice_id = 1;
    std::string default_voice;

    if (!sp.voice_audio.empty()) {
        breeze_tts::reference_voice rv;
        if (sp.voice_text.empty()) {
            fprintf(stderr, "fatal: --voice-audio needs --voice-text (the transcript)\n");
            return 1;
        }
        if (!load_reference(eng, sp.voice_audio, sp.voice_text, rv, err)) {
            fprintf(stderr, "fatal: %s\n", err.c_str());
            return 1;
        }
        default_voice = "default";
        voices[default_voice] = std::move(rv);
        fprintf(stderr, "default voice: cloned from %s (%d frames)\n", sp.voice_audio.c_str(),
                voices[default_voice].n_frames);
    }

    if (default_voice.empty() && sp.anchor_seed >= 0) {
        // The voice a seed draws for the instruction becomes the default speaker. Its own generated
        // codes are the reference, so nothing passes through the lossy audio encoder.
        breeze_tts::tts_params ap;
        ap.instruction = sp.instruction;
        ap.plain_prompt = sp.plain_prompt;
        ap.seed = sp.anchor_seed;
        ap.temperature = sp.temperature;
        auto ar = eng.synthesize(sp.anchor_text, ap, nullptr);
        if (!ar.success) {
            fprintf(stderr, "fatal: anchor voice failed: %s\n", ar.error.c_str());
            return 1;
        }
        breeze_tts::reference_voice rv;
        rv.text = sp.anchor_text;
        rv.codes = ar.codes;
        rv.n_frames = ar.n_frames;
        default_voice = "default";
        voices[default_voice] = std::move(rv);
        fprintf(stderr, "default voice: seed %lld for '%s' (%d frames)\n", (long long) sp.anchor_seed,
                sp.instruction.c_str(), ar.n_frames);
        if (const char * dbg = std::getenv("BREEZE_TTS_DEBUG_DIR")) {
            qwen3_tts::save_audio_file(std::string(dbg) + "/anchor.wav", ar.audio, breeze_tts::kSampleRate);
        }
    }

    // the previous chunk (text + its generated codes) conditions the next one so the speaker carries over
    breeze_tts::reference_voice prev;
    std::string prev_key;

    httplib::Server svr;
    svr.set_logger([](const httplib::Request & req, const httplib::Response & res) {
        if (req.method == "GET" && res.status < 400) return;
        fprintf(stderr, "%s %s -> %d\n", req.method.c_str(), req.path.c_str(), res.status);
    });

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        json m = {{"object", "list"}, {"data", json::array({{{"id", model_id}, {"object", "model"}, {"owned_by", "breezeblue"}}})}};
        res.set_content(m.dump(), "application/json");
    });

    svr.Get("/v1/audio/languages", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(R"({"languages":[{"code":"en"},{"code":"zh"}]})", "application/json");
    });

    svr.Get("/v1/audio/voices", [&](const httplib::Request &, httplib::Response & res) {
        json list = json::array({"default"});
        std::lock_guard<std::mutex> lock(voices_mutex);
        for (auto & [id, v] : voices) {
            if (id != "default") list.push_back(id);
        }
        res.set_content(json({{model_id, list}}).dump(), "application/json");
    });

    svr.Post("/v1/audio/voices", [&](const httplib::Request & req, httplib::Response & res) {
        if (!req.form.has_file("audio_sample")) {
            res.status = 400;
            res.set_content(error_json("'audio_sample' file is required", "invalid_request_error").dump(), "application/json");
            return;
        }
        std::string name = "custom", ref_text;
        if (req.form.has_field("name")) name = req.form.get_field("name");
        if (req.form.has_field("ref_text")) ref_text = req.form.get_field("ref_text");
        if (ref_text.empty()) {
            res.status = 400;
            res.set_content(error_json("'ref_text' (the transcript of the sample) is required to clone a voice",
                                       "invalid_request_error").dump(), "application/json");
            return;
        }
        auto file = req.form.get_file("audio_sample");
        char tmp[] = "/tmp/breezetts_voice_XXXXXX.wav";
        int fd = mkstemps(tmp, 4);
        if (fd < 0) {
            res.status = 500;
            res.set_content(error_json("failed to create temp file", "server_error").dump(), "application/json");
            return;
        }
        if (write(fd, file.content.data(), file.content.size()) < 0) { /* reported by the load below */ }
        close(fd);

        breeze_tts::reference_voice rv;
        std::string e;
        bool ok;
        {
            std::lock_guard<std::mutex> lock(synth_mutex);
            ok = load_reference(eng, tmp, ref_text, rv, e);
        }
        unlink(tmp);
        if (!ok) {
            res.status = 400;
            res.set_content(error_json(e, "invalid_request_error").dump(), "application/json");
            return;
        }
        std::string id;
        {
            std::lock_guard<std::mutex> lock(voices_mutex);
            id = "voice_" + std::to_string(next_voice_id++);
            voices[id] = std::move(rv);
        }
        fprintf(stderr, "created voice '%s' (id: %s, %d frames)\n", name.c_str(), id.c_str(), voices[id].n_frames);
        res.set_content(json({{"id", id}, {"name", name}, {"ref_frames", voices[id].n_frames}}).dump(), "application/json");
    });

    svr.Delete(R"(/v1/audio/voices/(.+))", [&](const httplib::Request & req, httplib::Response & res) {
        std::lock_guard<std::mutex> lock(voices_mutex);
        if (voices.erase(req.matches[1])) {
            res.set_content(R"({"deleted":true})", "application/json");
        } else {
            res.status = 404;
            res.set_content(error_json("voice not found", "not_found").dump(), "application/json");
        }
    });

    svr.Post("/v1/audio/speech", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception &) {
            res.status = 400;
            res.set_content(error_json("invalid JSON body", "invalid_request_error").dump(), "application/json");
            return;
        }
        const std::string input = body.value("input", "");
        if (input.empty()) {
            res.status = 400;
            res.set_content(error_json("'input' is required", "invalid_request_error").dump(), "application/json");
            return;
        }
        const std::string fmt = body.value("response_format", "wav");
        const std::string voice = body.value("voice", "");

        breeze_tts::tts_params p;
        p.instruction = body.value("instruction", sp.instruction);
        p.plain_prompt = sp.plain_prompt;
        p.temperature = body.value("temperature", sp.temperature);
        p.depth_temperature = body.value("depth_temperature", sp.depth_temperature);
        p.top_k = body.value("top_k", sp.top_k);
        p.top_p = body.value("top_p", 1.0f);
        p.repetition_penalty = body.value("repetition_penalty", sp.repetition_penalty);
        p.seed = body.value("seed", sp.seed);
        p.max_frames = body.value("max_tokens", 1500);

        breeze_tts::reference_voice ref;
        bool have_ref = false;
        std::string ref_key;
        {
            std::lock_guard<std::mutex> lock(voices_mutex);
            auto it = voices.find(voice);
            if (it == voices.end() && !default_voice.empty() && voice != "none") it = voices.find(default_voice);
            if (it != voices.end() && voice != "none") {
                ref = it->second;
                ref_key = it->first;
                have_ref = true;
            }
        }

        breeze_tts::tts_result r;
        {
            std::lock_guard<std::mutex> lock(synth_mutex);
            const auto chunks = split_text(input, 220);
            std::vector<int32_t> last_codes;
            if (!have_ref || ref_key != prev_key) {   // a request never continues from another request
                prev = breeze_tts::reference_voice();
                prev_key = ref_key;
            }
            for (size_t ci = 0; ci < chunks.size(); ci++) {
                breeze_tts::tts_params pc = p;
                if (pc.seed >= 0) pc.seed += (int64_t) ci;
                if (ci > 0 && !last_codes.empty()) {
                    const size_t n = std::min(last_codes.size(), (size_t) 25 * breeze_tts::kNumCodebooks);
                    pc.vocoder_context.assign(last_codes.end() - n, last_codes.end());
                }
                breeze_tts::reference_voice use;
                const breeze_tts::reference_voice * rp = nullptr;
                if (have_ref || (sp.continuity && prev.n_frames > 0)) {
                    if (have_ref) use = ref;
                    if (sp.continuity && prev.n_frames > 0) {
                        if (!use.text.empty()) use.text += " ";
                        use.text += prev.text;
                        use.codes.insert(use.codes.end(), prev.codes.begin(), prev.codes.end());
                        use.n_frames += prev.n_frames;
                    }
                    rp = &use;
                }
                breeze_tts::tts_result rc = eng.synthesize(chunks[ci], pc, rp);
                if (!rc.success) {
                    prev = breeze_tts::reference_voice();
                    r = std::move(rc);
                    break;
                }
                last_codes = rc.codes;
                fprintf(stderr, "chunk %zu/%zu: %.1fs + %.1fs (%d frames) \"%.50s...\"\n", ci + 1, chunks.size(),
                        ci == 0 ? 0.0 : (double) r.audio.size() / breeze_tts::kSampleRate,
                        (double) rc.audio.size() / breeze_tts::kSampleRate, rc.n_frames, chunks[ci].c_str());
                if (sp.continuity && rc.n_frames <= 300) {
                    prev.text = chunks[ci];
                    prev.codes = rc.codes;
                    prev.n_frames = rc.n_frames;
                } else {
                    prev = breeze_tts::reference_voice();   // too long to reuse: never keep an older sentence
                }
                if (ci == 0) {
                    r = std::move(rc);
                } else {
                    r.audio.insert(r.audio.end(), rc.audio.begin(), rc.audio.end());
                    r.n_prompt_tokens += rc.n_prompt_tokens;
                    r.n_frames += rc.n_frames;
                    r.t_generate_ms += rc.t_generate_ms;
                    r.t_decode_ms += rc.t_decode_ms;
                    r.t_total_ms += rc.t_total_ms;
                }
            }
        }
        if (!r.success) {
            res.status = 500;
            res.set_content(error_json(r.error, "server_error").dump(), "application/json");
            return;
        }
        const double secs = (double) r.audio.size() / breeze_tts::kSampleRate;
        fprintf(stderr, "synthesized %.1fs of audio in %.1fs (rtf %.2f; prompt %d tok, %d frames, gen %lld ms, vocoder %lld ms)\n",
                secs, r.t_total_ms / 1000.0, secs > 0 ? r.t_total_ms / 1000.0 / secs : 0.0, r.n_prompt_tokens,
                r.n_frames, (long long) r.t_generate_ms, (long long) r.t_decode_ms);
        const std::string out = encode_format(fmt, r.audio, breeze_tts::kSampleRate);
        if (out.empty()) {
            res.status = 400;
            res.set_content(error_json("unsupported response_format: " + fmt, "invalid_request_error").dump(),
                            "application/json");
            return;
        }
        res.set_content(out, content_type_for(fmt));
    });

    {   // build the per-frame graphs and compile the kernels now rather than in the first request
        breeze_tts::tts_params wp;
        wp.instruction = sp.instruction;
        wp.plain_prompt = sp.plain_prompt;
        wp.max_frames = 6;
        wp.seed = 0;
        const auto it = voices.find(default_voice);
        const auto w = eng.synthesize("Warming up.", wp, it != voices.end() ? &it->second : nullptr);
        fprintf(stderr, "warmup %s\n", w.success ? "done" : ("failed: " + w.error).c_str());
    }

    fprintf(stderr, "listening on %s:%d\n", sp.host.c_str(), sp.port);
    if (!svr.listen(sp.host.c_str(), sp.port)) {
        fprintf(stderr, "fatal: failed to listen on %s:%d\n", sp.host.c_str(), sp.port);
        return 1;
    }
    return 0;
}
