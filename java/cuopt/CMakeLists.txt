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

# Building against a cuOpt build tree rather than an install prefix: the top-level build.sh
# points these at cpp/build and its CPM-fetched dependencies. The rmm and raft headers must
# be the exact ones libcuopt was compiled with -- rmm puts its version in an inline namespace
# (rmm::_RMM_26_10), so mixing a different copy links cleanly and then fails at dlopen with an
# undefined symbol.
set(CUOPT_LIBRARY "" CACHE FILEPATH "Path to libcuopt.so (defaults to CUOPT_PREFIX/lib)")
# Embedding cuOpt into libcuopt_jni.so instead of linking the shared library, so a published
# classifier JAR carries everything it needs. See #1817.
set(CUOPT_STATIC_BUILD_DIR "" CACHE PATH
    "Build tree holding libcuopt_static.a; when set, cuOpt is linked into cuopt_jni statically")
set(CUOPT_EXTRA_INCLUDE_DIRS "" CACHE STRING
    "Extra include directories searched before CUOPT_PREFIX/include")

if(CUOPT_STATIC_BUILD_DIR)
  set(CUOPT_STATIC_ARCHIVE "${CUOPT_STATIC_BUILD_DIR}/libcuopt_static.a")
  if(NOT EXISTS "${CUOPT_STATIC_ARCHIVE}")
    message(FATAL_ERROR "libcuopt_static.a was not found in ${CUOPT_STATIC_BUILD_DIR}")
  endif()
  # Fetched and built alongside cuOpt rather than installed, so they are found by path.
  file(GLOB CUOPT_STATIC_DEP_ARCHIVES
       "${CUOPT_STATIC_BUILD_DIR}/_deps/pslp-build/libPSLP.a"
       "${CUOPT_STATIC_BUILD_DIR}/_deps/kaminpar-build/kaminpar-shm/libKaMinPar.a"
       "${CUOPT_STATIC_BUILD_DIR}/_deps/kaminpar-build/kaminpar-common/libKaMinParCommon.a")
elseif(NOT CUOPT_LIBRARY)
  set(CUOPT_LIBRARY "${CUOPT_PREFIX}/lib/libcuopt.so")
endif()
if(NOT CUOPT_STATIC_BUILD_DIR AND NOT EXISTS "${CUOPT_LIBRARY}")
  message(FATAL_ERROR "cuOpt shared library was not found at ${CUOPT_LIBRARY}")
endif()

get_target_property(CUOPT_CUDART_LIBRARY CUDA::cudart IMPORTED_LOCATION)
if(NOT CUOPT_CUDART_LIBRARY)
  find_library(CUOPT_CUDART_LIBRARY NAMES cudart HINTS "${CUDAToolkit_LIBRARY_DIR}" REQUIRED)
endif()
get_filename_component(CUOPT_CUDA_RUNTIME_DIR "${CUOPT_CUDART_LIBRARY}" DIRECTORY)

add_library(cuopt_jni SHARED
  src/main/native/cuopt_jni.cpp)

target_include_directories(cuopt_jni PRIVATE
  ${JNI_INCLUDE_DIRS}
  ${CUOPT_EXTRA_INCLUDE_DIRS}
  ${CUOPT_INCLUDE_DIR}
  ${CUOPT_PREFIX}/include/rapids
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/src)

if(CUOPT_STATIC_BUILD_DIR)
  find_package(OpenMP REQUIRED)
  # The JNI layer references only a fraction of cuOpt directly; the rest is reached through
  # registrations and virtual dispatch, so the archive has to be kept whole.
  target_link_libraries(cuopt_jni PRIVATE
    -Wl,--whole-archive "${CUOPT_STATIC_ARCHIVE}" -Wl,--no-whole-archive
    ${CUOPT_STATIC_DEP_ARCHIVES}
    CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::cudart_static
    OpenMP::OpenMP_CXX
    # rmm is mostly header-only, but its exception types are defined in librmm.so, and
    # rapids_logger likewise. Conda ships no static variant of either, so they are linked
    # shared and packaged next to cuopt_jni, which finds them through its $ORIGIN RPATH.
    "${CUOPT_PREFIX}/lib/librmm.so"
    "${CUOPT_PREFIX}/lib/librapids_logger.so"
    # KaMinPar, which the static archive pulls in, throws through TBB.
    "${CUOPT_PREFIX}/lib/libtbb.so.12"
    # PDLP's distributed path references NCCL unconditionally; cpp/CMakeLists.txt has no
    # switch to compile it out, so it has to be linked even for a single-GPU JAR.
    "${CUOPT_PREFIX}/lib/libnccl.so.2"
    "${CUOPT_PREFIX}/lib/libcudss.so.0")
else()
  target_link_libraries(cuopt_jni PRIVATE "${CUOPT_LIBRARY}" CUDA::cudart)
endif()

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
