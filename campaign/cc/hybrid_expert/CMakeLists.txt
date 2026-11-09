cmake_minimum_required(VERSION 3.18)
project(hybrid_moe_bench C CXX)
# Out-of-tree harness: builds ggml (CPU + CUDA backends) from the READ-ONLY llama.cpp tree into this build dir.
set(LLAMA_DIR "/llama" CACHE PATH "llama.cpp source tree (read-only mount)")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ON CACHE BOOL "" FORCE)
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "" FORCE)
# Native ISA of the build host (Zen 2: AVX2/FMA/F16C, no AVX-512). The engine image (campaign/llamacpp:b0) also uses AVX2 for
# ggml_vec_dot_q4_K_q8_K (verified by disassembly, see RESULTS.md), so the CPU miss kernel is the same family.
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
find_package(Threads REQUIRED)
add_subdirectory(${LLAMA_DIR}/ggml ggml)
add_executable(hybrid_moe_bench hybrid_moe_bench.cpp)
target_link_libraries(hybrid_moe_bench PRIVATE ggml ggml-base ggml-cpu ggml-cuda Threads::Threads)
target_compile_options(hybrid_moe_bench PRIVATE -O2 -Wall)
