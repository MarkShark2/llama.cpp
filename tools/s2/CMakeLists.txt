# s2 - Fish Audio S2 Pro text-to-speech (dual-AR + RVQ codec)
#
# Ported from https://github.com/rodrigomatta/s2.cpp . The engine talks to ggml
# directly and does not use llama.cpp, so it only links the ggml target. The
# upstream project drives backend selection with its own S2_VULKAN/S2_CUDA/
# S2_METAL options; here we follow the fork's GGML_* options instead so s2 is
# built with whatever backends the rest of the tree already enabled.
#
# Note: the upstream patches/ directory is intentionally not carried over. Both
# patches only touch ggml-cuda conv kernels (grid-size limits for long audio);
# they are not needed for the Vulkan builds this fork ships to the cluster.

find_package(Threads REQUIRED)

set(TARGET s2-cli)

add_executable(${TARGET}
    src/main.cpp
    src/s2_audio.cpp
    src/s2_tokenizer.cpp
    src/s2_log.cpp
    src/s2_sampler.cpp
    src/s2_model.cpp
    src/s2_codec.cpp
    src/s2_prompt.cpp
    src/s2_generate.cpp
    src/s2_pipeline.cpp
    src/s2_server.cpp
    src/s2_voice.cpp
)

set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME s2-cli)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries  (${TARGET} PRIVATE ggml Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (GGML_VULKAN)
    target_compile_definitions(${TARGET} PRIVATE GGML_USE_VULKAN)
endif()
if (GGML_CUDA)
    target_compile_definitions(${TARGET} PRIVATE GGML_USE_CUDA)
endif()
if (GGML_METAL)
    target_compile_definitions(${TARGET} PRIVATE GGML_USE_METAL)
endif()

if (UNIX AND NOT APPLE)
    target_link_libraries(${TARGET} PRIVATE m)
endif()

if (WIN32)
    # no NOMINMAX here: src/main.cpp uses the windows.h max() macro
    target_compile_definitions(${TARGET} PRIVATE WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    target_link_libraries     (${TARGET} PRIVATE ws2_32 wsock32)
endif()

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
