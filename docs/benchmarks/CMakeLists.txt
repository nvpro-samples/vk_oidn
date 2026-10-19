# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

# oidn_gpu_bench, built against the official Open Image Denoise 2.5.1 SDK:
#   cmake -S docs/benchmarks -B docs/benchmarks/build -DOIDN_SDK=<oidn-2.5.1.x64.windows>
#   cmake --build docs/benchmarks/build --config Release
cmake_minimum_required(VERSION 3.20)
project(oidn_gpu_bench CXX)
find_package(CUDAToolkit REQUIRED)
set(OIDN_SDK "" CACHE PATH "Official Open Image Denoise 2.5.1 package")
if(NOT EXISTS "${OIDN_SDK}/include/OpenImageDenoise/oidn.h")
  message(FATAL_ERROR "set OIDN_SDK to the official Open Image Denoise 2.5.1 package")
endif()
add_executable(oidn_gpu_bench oidn_gpu_bench.cpp)
target_compile_features(oidn_gpu_bench PRIVATE cxx_std_17)
target_include_directories(oidn_gpu_bench PRIVATE ${OIDN_SDK}/include)
target_link_libraries(oidn_gpu_bench PRIVATE ${OIDN_SDK}/lib/OpenImageDenoise.lib CUDA::cudart_static)
set_target_properties(oidn_gpu_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
