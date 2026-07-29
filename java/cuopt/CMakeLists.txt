# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.22)
project(cuopt_java_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CUOPT_PREFIX "" CACHE PATH "cuOpt installation prefix")
if(NOT CUOPT_PREFIX)
  set(CUOPT_PREFIX "$ENV{CONDA_PREFIX}")
endif()
if(NOT CUOPT_PREFIX)
  message(FATAL_ERROR "Set CUOPT_PREFIX to the cuOpt conda environment")
endif()

if(NOT DEFINED CUDAToolkit_ROOT AND EXISTS "${CUOPT_PREFIX}/targets")
  set(CUDAToolkit_ROOT "${CUOPT_PREFIX}" CACHE PATH "CUDA toolkit root")
endif()

find_package(JNI REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CUOPT_INCLUDE_DIR "${CUOPT_PREFIX}/include")
set(CUOPT_LIBRARY "${CUOPT_PREFIX}/lib/libcuopt.so")
if(NOT EXISTS "${CUOPT_LIBRARY}")
  message(FATAL_ERROR "cuOpt shared library was not found at ${CUOPT_LIBRARY}")
endif()

get_target_property(CUOPT_CUDART_LIBRARY CUDA::cudart IMPORTED_LOCATION)
if(NOT CUOPT_CUDART_LIBRARY)
  find_library(CUOPT_CUDART_LIBRARY NAMES cudart HINTS "${CUDAToolkit_LIBRARY_DIR}" REQUIRED)
endif()
get_filename_component(CUOPT_CUDA_RUNTIME_DIR "${CUOPT_CUDART_LIBRARY}" DIRECTORY)

add_library(cuopt_jni SHARED
  src/main/native/cuopt_java_native_api.cpp
  src/main/native/cuopt_jni.cpp)

target_include_directories(cuopt_jni PRIVATE
  ${JNI_INCLUDE_DIRS}
  ${CUOPT_INCLUDE_DIR}
  ${CUOPT_PREFIX}/include/rapids
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/src)

target_link_libraries(cuopt_jni PRIVATE "${CUOPT_LIBRARY}" CUDA::cudart)

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
