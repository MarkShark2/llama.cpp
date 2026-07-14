# voxcpm2

find_package(Threads REQUIRED)

# MiniCPMLM wrapper around the llama.cpp BaseLM
add_library(voxcpm2_llm STATIC
    voxcpm2_llm.cpp
    voxcpm2_llm.h
)
target_link_libraries     (voxcpm2_llm PUBLIC llama llama-common)
target_include_directories(voxcpm2_llm PUBLIC .)
target_compile_features   (voxcpm2_llm PRIVATE cxx_std_17)

# FSQ module
add_library(voxcpm2_fsq STATIC
    voxcpm2_fsq.cpp
    voxcpm2_fsq.h
)
target_link_libraries     (voxcpm2_fsq PUBLIC ggml llama-common)
target_include_directories(voxcpm2_fsq PUBLIC .)
target_compile_features   (voxcpm2_fsq PRIVATE cxx_std_17)

# acoustic transformer helpers (LocEnc / LocDiT / CFM / AudioVAE)
add_library(voxcpm2_acoustic STATIC
    voxcpm2_transformer.cpp
    voxcpm2_transformer.h
    voxcpm2_localenc.cpp
    voxcpm2_localenc.h
    voxcpm2_locdit.cpp
    voxcpm2_locdit.h
    voxcpm2_cfm.cpp
    voxcpm2_cfm.h
    voxcpm2_components.cpp
    voxcpm2_components.h
    voxcpm2_audiovae.cpp
    voxcpm2_audiovae.h
)
target_link_libraries     (voxcpm2_acoustic PUBLIC ggml llama-common)
target_include_directories(voxcpm2_acoustic PUBLIC .)
target_compile_features   (voxcpm2_acoustic PRIVATE cxx_std_17)

# integrated runtime
add_library(voxcpm2_runtime STATIC
    voxcpm2_runtime.cpp
    voxcpm2_runtime.h
)
target_link_libraries     (voxcpm2_runtime PUBLIC voxcpm2_llm voxcpm2_fsq voxcpm2_acoustic llama ggml llama-common)
target_include_directories(voxcpm2_runtime PUBLIC .)
target_compile_features   (voxcpm2_runtime PRIVATE cxx_std_17)

if (WIN32)
    foreach (tgt voxcpm2_llm voxcpm2_fsq voxcpm2_acoustic voxcpm2_runtime)
        target_compile_definitions(${tgt} PRIVATE NOMINMAX _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
    endforeach()
endif()

set(TARGET voxcpm2-cli)
add_executable         (${TARGET} voxcpm2_cli.cpp)
set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME voxcpm2-cli)
target_link_libraries  (${TARGET} PRIVATE voxcpm2_runtime llama-common ggml Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (WIN32)
    target_compile_definitions(${TARGET} PRIVATE NOMINMAX _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
endif()
if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
