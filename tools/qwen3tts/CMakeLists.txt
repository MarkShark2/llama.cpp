# qwen3tts - Qwen3-TTS-12Hz text-to-speech (talker + code predictor + WavTokenizer codec)
#
# Ported from https://github.com/khimaros/qwen3-tts.cpp (a fork of
# predict-woo/qwen3-tts.cpp). The fork is used rather than predict-woo upstream
# because only the fork supports the 1.7B models: they add a
# code_pred.mtp_proj that projects the talker hidden size down to a smaller
# code-predictor hidden size, which upstream neither converts nor reads. It also
# ships the OpenAI-compatible server this deployment needs.
#
# Like tools/s2 the engine talks to ggml directly and never touches llama.cpp,
# so it only links the ggml target. Upstream builds ggml from a submodule of its
# own and fetches httplib/json at configure time; here we link the fork's ggml
# and the already-vendored cpp-httplib + nlohmann/json instead, so no network is
# needed at configure time.
#
# Note: upstream's CoreML code-predictor path (coreml_code_predictor.mm) is an
# Apple-only accelerator and is not carried over - the stub upstream compiles
# when QWEN3_TTS_COREML is off is used unconditionally. Upstream's C API
# bindings, tests/ and Python parity harness are also not carried over.

find_package(Threads REQUIRED)

# Optional compressed audio (mp3/opus) via libav/ffmpeg, matching upstream's
# auto-detection: builds without it succeed and simply omit those formats, in
# which case the server can only return wav.
find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
    pkg_check_modules(LIBAV IMPORTED_TARGET libavformat libavcodec libavutil libswresample)
endif()

# shared engine, used by both the CLI and the server
add_library(qwen3tts STATIC
    src/qwen3_tts.cpp
    src/text_tokenizer.cpp
    src/tokenizer_unicode.cpp
    src/tokenizer_unicode_data.cpp
    src/tts_transformer.cpp
    src/audio_tokenizer_encoder.cpp
    src/audio_codec_encoder.cpp
    src/audio_tokenizer_decoder.cpp
    src/gguf_loader.cpp
    src/coreml_code_predictor_stub.cpp
)

target_include_directories(qwen3tts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries     (qwen3tts PUBLIC ggml Threads::Threads)
target_compile_features   (qwen3tts PUBLIC cxx_std_17)

if (LIBAV_FOUND)
    message(STATUS "qwen3tts: compressed audio (mp3/opus) enabled (libavcodec ${LIBAV_libavcodec_VERSION})")
    target_link_libraries     (qwen3tts PUBLIC PkgConfig::LIBAV)
    target_compile_definitions(qwen3tts PUBLIC QWEN3_TTS_LIBAV)
else()
    message(STATUS "qwen3tts: compressed audio (mp3/opus) disabled (libav/ffmpeg not found), wav only")
endif()

if (UNIX AND NOT APPLE)
    target_link_libraries(qwen3tts PUBLIC m)
endif()

# CLI
set(TARGET qwen3-tts-cli)
add_executable         (${TARGET} src/main.cpp)
set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME qwen3-tts-cli)
target_link_libraries  (${TARGET} PRIVATE qwen3tts Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

# OpenAI-compatible HTTP server
set(TARGET_SRV qwen3-tts-server)
add_executable         (${TARGET_SRV} src/server.cpp)
set_target_properties  (${TARGET_SRV} PROPERTIES OUTPUT_NAME qwen3-tts-server)
target_link_libraries  (${TARGET_SRV} PRIVATE qwen3tts cpp-httplib Threads::Threads)
target_include_directories(${TARGET_SRV} PRIVATE
    ${CMAKE_SOURCE_DIR}/vendor
    ${CMAKE_SOURCE_DIR}/vendor/cpp-httplib
)
target_compile_features(${TARGET_SRV} PRIVATE cxx_std_17)
if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET_SRV} RUNTIME)
endif()
