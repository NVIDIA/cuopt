# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

include(../../../../cmake/rapids_config.cmake)

project(
  mpsparser-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX)

option(FIND_MPS_PARSER_CPP "Search for existing CUOPT C++ installations before defaulting to local files"
       OFF)
option(CUOPT_BUILD_WHEELS "Whether this build is generating a Python wheel." ON)

# If the user requested it we attempt to find MPS Parser.
if(FIND_MPS_PARSER_CPP)
  find_package(mps_parser "${RAPIDS_VERSION}")
else()
  set(mps_parser_FOUND OFF)
endif()

if(NOT mps_parser_FOUND)

  # Since cuopt only enables CUDA optionally we need to manually include the file that
  # rapids_cuda_init_architectures relies on `project` including.
  # include("${CMAKE_PROJECT_cuopt-python_INCLUDE}")

  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)

  set(_exclude_from_all "")
  if(CUOPT_BUILD_WHEELS)
    # Don't install the cuopt C++ targets into wheels
    set(_exclude_from_all EXCLUDE_FROM_ALL)
  endif()

  add_subdirectory(../../../../cpp/libmps_parser cuopt-cpp ${_exclude_from_all})
  set(cython_lib_dir data_model)
  install(TARGETS mps_parser DESTINATION ${cython_lib_dir})
  set(cython_lib_dir cuopt_mps_parser)
  install(TARGETS mps_parser DESTINATION ${cython_lib_dir})

endif()

include_directories("../../../../cpp/libmps_parser/src" "../../../../cpp/libmps_parser/include")
include(rapids-cython-core)
rapids_cython_init()

add_subdirectory(cuopt_mps_parser)
add_subdirectory(data_model)
if(DEFINED cython_lib_dir)
  rapids_cython_add_rpath_entries(TARGET mps_parser PATHS "${cython_lib_dir}")
endif()
set(rpaths
  "$ORIGIN/../lib64"
  "$ORIGIN/../../rapids_logger/lib64"
)
set_property(TARGET mps_parser PROPERTY INSTALL_RPATH ${rpaths} APPEND)
