# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)
project(cuopt_java_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(JNI REQUIRED)

set(CUOPT_PREFIX "" CACHE PATH "cuOpt installation prefix")
if(NOT CUOPT_PREFIX)
  set(CUOPT_PREFIX "$ENV{CONDA_PREFIX}")
endif()
if(NOT CUOPT_PREFIX)
  message(FATAL_ERROR "Set CUOPT_PREFIX to the cuOpt conda environment")
endif()

set(CUOPT_INCLUDE_DIR "${CUOPT_PREFIX}/include")
set(CUOPT_LIBRARY "${CUOPT_PREFIX}/lib/libcuopt.so")
set(CUOPT_CUDA_RUNTIME_DIR "${CUOPT_PREFIX}/targets/x86_64-linux/lib"
    CACHE PATH "CUDA runtime library directory")
set(CUOPT_CUDA_INCLUDE_DIR "" CACHE PATH "CUDA toolkit include directory")
if(NOT CUOPT_CUDA_INCLUDE_DIR AND DEFINED ENV{CUDA_HOME})
  set(CUOPT_CUDA_INCLUDE_DIR "$ENV{CUDA_HOME}/include")
endif()
if(NOT CUOPT_CUDA_INCLUDE_DIR AND EXISTS "/usr/local/cuda/include")
  set(CUOPT_CUDA_INCLUDE_DIR "/usr/local/cuda/include")
endif()
if(NOT EXISTS "${CUOPT_LIBRARY}")
  message(FATAL_ERROR "cuOpt shared library was not found at ${CUOPT_LIBRARY}")
endif()
set(CUOPT_CUDART_LIBRARY "${CUOPT_CUDA_RUNTIME_DIR}/libcudart.so")
if(NOT EXISTS "${CUOPT_CUDART_LIBRARY}")
  message(FATAL_ERROR "CUDA runtime library was not found at ${CUOPT_CUDART_LIBRARY}")
endif()

add_library(cuopt_jni SHARED
  src/main/native/cuopt_java_native_api.cpp
  src/main/native/cuopt_jni.cpp)

target_include_directories(cuopt_jni PRIVATE
  ${JNI_INCLUDE_DIRS}
  ${CUOPT_INCLUDE_DIR}
  ${CUOPT_PREFIX}/include/rapids
  ${CUOPT_PREFIX}/targets/x86_64-linux/include
  ${CUOPT_CUDA_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/src)

target_link_libraries(cuopt_jni PRIVATE "${CUOPT_LIBRARY}" "${CUOPT_CUDART_LIBRARY}")

# The Java module is built outside the main cuOpt build.  Keep its native
# loader self-contained while allowing the script to add the cuOpt runtime
# directory supplied by the selected conda environment.
set_target_properties(cuopt_jni PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN")

if(CUOPT_RUNTIME_LIBRARY_DIR)
  set_property(TARGET cuopt_jni APPEND PROPERTY
    BUILD_RPATH "${CUOPT_RUNTIME_LIBRARY_DIR}")
endif()

set_property(TARGET cuopt_jni APPEND PROPERTY BUILD_RPATH "${CUOPT_CUDA_RUNTIME_DIR}")
set_property(TARGET cuopt_jni APPEND PROPERTY INSTALL_RPATH "${CUOPT_CUDA_RUNTIME_DIR}")
