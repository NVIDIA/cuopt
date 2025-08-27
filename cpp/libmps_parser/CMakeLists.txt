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
include(../../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

project(
  MPS_PARSER
  VERSION "${CUOPT_VERSION}"
  LANGUAGES CXX
)

# Write the version header
rapids_cmake_write_version_file(include/mps_parser/version_config.hpp)

# ##################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# #############################################################################
# - User Options  ------------------------------------------------------------
option(BUILD_TESTS "Configure CMake to build tests" ON)

message(VERBOSE "cuOpt: Build mps-parser unit-tests: ${BUILD_TESTS}")

# ##################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(MPS_PARSER_CXX_FLAGS "")

if(CMAKE_COMPILER_IS_GNUCXX)
  list(APPEND MPS_PARSER_CXX_FLAGS -Werror -Wno-error=deprecated-declarations)
endif(CMAKE_COMPILER_IS_GNUCXX)

if(DEFINE_ASSERT)
  add_definitions(-DASSERT_MODE)
endif(DEFINE_ASSERT)


# ##################################################################################################
# - find CPM based dependencies  ------------------------------------------------------------------
rapids_cpm_init()
if(BUILD_TESTS)
  include(cmake/thirdparty/get_gtest.cmake)
endif()

add_library(mps_parser SHARED
  src/data_model_view.cpp
  src/mps_data_model.cpp
  src/mps_parser.cpp
  src/parser.cpp
  src/utilities/cython_mps_parser.cpp
)

set_target_properties(mps_parser
  PROPERTIES BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"

  # set target compile options
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(mps_parser
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${MPS_PARSER_CXX_FLAGS}>"
)

file(WRITE "${MPS_PARSER_BINARY_DIR}/fatbin.ld"
  [=[
SECTIONS
{
  .nvFatBinSegment : { *(.nvFatBinSegment) }
  .nv_fatbin : { *(.nv_fatbin) }
}
]=])
target_link_options(mps_parser PRIVATE "${MPS_PARSER_BINARY_DIR}/fatbin.ld")

add_library(cuopt::mps_parser ALIAS mps_parser)

# ##################################################################################################
# - include paths ---------------------------------------------------------------------------------
target_include_directories(mps_parser
  PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)

# ##################################################################################################
# - generate tests --------------------------------------------------------------------------------
if(BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif(BUILD_TESTS)


# ##################################################################################################
# * mps_parser Install ----------------------------------------------------------------------------
install(TARGETS mps_parser
  DESTINATION lib
  EXPORT mps-parser-exports)

install(DIRECTORY include/mps_parser/
  DESTINATION include/mps_parser)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/mps_parser/version_config.hpp
  DESTINATION include/mps_parser)

rapids_export(INSTALL mps_parser
  EXPORT_SET mps-parser-exports
  GLOBAL_TARGETS mpsparser
  NAMESPACE cuopt::
)
