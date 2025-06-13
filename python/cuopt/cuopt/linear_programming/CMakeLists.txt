# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

set(cuopt_version 25.08.00)

include(../../../../cmake/rapids_config.cmake)

project(
  mpsparser-python
  VERSION ${cuopt_version}
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
  find_package(mps_parser ${cuopt_version})
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
    # Statically link dependencies if building wheels
    # set(CUDA_STATIC_RUNTIME ON)
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
