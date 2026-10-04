# breezetts - Breeze-TTS 2 text-to-speech (T5Gemma2 text encoder + Qwen3 backbone + depth decoder)
#
# Re-expressed against plain ggml from audio.cpp's BreezeTTS runtime (Apache-2.0, ShugoAI LLC).
# The audio codec Breeze ships is bit-identical to Qwen3-TTS-Tokenizer-12Hz, so the vocoder and the
# reference-voice encoder come from the qwen3tts library; this tool links it.

find_package(Threads REQUIRED)

add_library(breezetts STATIC src/breeze_tts.cpp)
target_include_directories(breezetts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/vendor)
target_link_libraries     (breezetts PUBLIC qwen3tts ggml Threads::Threads)
target_compile_features   (breezetts PUBLIC cxx_std_17)

set(TARGET breeze-tts-server)
add_executable         (${TARGET} src/server.cpp)
set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME breeze-tts-server)
target_link_libraries  (${TARGET} PRIVATE breezetts cpp-httplib Threads::Threads)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/vendor
    ${CMAKE_SOURCE_DIR}/vendor/cpp-httplib
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
