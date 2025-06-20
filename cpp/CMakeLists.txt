# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include(GNUInstallDirs)

set(RAFT_NVTX ON)
include(../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(CUOPT)

project(
  CUOPT
  VERSION 25.08.00
  LANGUAGES CXX CUDA C
)

if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS 12.0.0)
  list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES "90")
  list(APPEND CMAKE_CUDA_ARCHITECTURES "80-virtual")
endif()

set(DEPENDENT_LIB_MAJOR_VERSION "25")
set(DEPENDENT_LIB_MINOR_VERSION "08")

rapids_cmake_write_version_file(include/cuopt/version_config.hpp)
# ##################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# #############################################################################
# - User Options  ------------------------------------------------------------
option(CMAKE_CUDA_LINEINFO "Enable the -lineinfo option for nvcc useful for cuda-memcheck / profiler" ON)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(DISABLE_OPENMP "Disable OpenMP" OFF)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA toolkit runtime and libraries" OFF)

message(VERBOSE "cuOpt: Enable nvcc -lineinfo: ${CMAKE_CUDA_LINEINFO}")
message(VERBOSE "cuOpt: Build cuOpt unit-tests: ${BUILD_TESTS}")
message(VERBOSE "cuOpt: Build cuOpt multigpu tests: ${BUILD_TESTS}")
message(VERBOSE "cuOpt: Disable OpenMP: ${DISABLE_OPENMP}")

# ##################################################################################################
# - compiler options ------------------------------------------------------------------------------

# CUDA runtime
rapids_cuda_init_runtime(USE_STATIC ${CUDA_STATIC_RUNTIME})

rapids_find_package(CUDAToolkit REQUIRED
  BUILD_EXPORT_SET cuopt-exports
  INSTALL_EXPORT_SET cuopt-exports
)

set(CUOPT_CXX_FLAGS "")
set(CUOPT_CUDA_FLAGS "")

if(CMAKE_COMPILER_IS_GNUCXX)
  list(APPEND CUOPT_CXX_FLAGS -Werror -Wno-error=deprecated-declarations)
endif(CMAKE_COMPILER_IS_GNUCXX)

if(DEFINE_ASSERT)
  add_definitions(-DASSERT_MODE)
endif(DEFINE_ASSERT)

if(DEFINE_BENCHMARK)
  add_definitions(-DBENCHMARK)
endif(DEFINE_BENCHMARK)

if(DEFINE_PDLP_VERBOSE_MODE)
  add_definitions(-DPDLP_VERBOSE_MODE)
endif(DEFINE_PDLP_VERBOSE_MODE)

# Set logging level
set(LIBCUOPT_LOGGING_LEVEL
  "INFO"
  CACHE STRING "Choose the logging level."
)
set_property(
  CACHE LIBCUOPT_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL"
                                       "OFF"
)
message(VERBOSE "CUOPT: LIBCUOPT_LOGGING_LEVEL = '${LIBCUOPT_LOGGING_LEVEL}'.")

#add_compile_definitions(CUOPT_LOG_ACTIVE_LEVEL=CUOPT_LOG_LEVEL_${LIBCUOPT_LOGGING_LEVEL})

message("-- Building with logging level = ${LIBCUOPT_LOGGING_LEVEL}")

message("-- Building for GPU_ARCHS = ${CMAKE_CUDA_ARCHITECTURES}")

# make the flags global in order to propagate flags to test cmake files
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8.0)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -static-global-template-stub=false")
endif()
list(APPEND CUOPT_CUDA_FLAGS -Werror=cross-execution-space-call -Wno-deprecated-declarations -Xcompiler=-Werror)
list(APPEND CUOPT_CUDA_FLAGS -Xcompiler=-Wall -Wno-error=non-template-friend)
list(APPEND CUOPT_CUDA_FLAGS -Xfatbin=-compress-all)
list(APPEND CUOPT_CUDA_FLAGS -fopenmp)


if(NOT DISABLE_OPENMP)
  find_package(OpenMP)

  if(OPENMP_FOUND)
    message(VERBOSE "cuOpt: OpenMP found in ${OpenMP_CXX_INCLUDE_DIRS}")
  endif()
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
  message(STATUS "Building with debugging flags")
  list(APPEND CUOPT_CUDA_FLAGS -G -Xcompiler=-rdynamic -O0)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
elseif(CMAKE_CUDA_LINEINFO)
  message(STATUS "Enabling line info")
  list(APPEND CUOPT_CUDA_FLAGS -lineinfo)
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo")
endif(CMAKE_BUILD_TYPE MATCHES Debug)


# ##################################################################################################
# - find CPM based dependencies  ------------------------------------------------------------------
rapids_cpm_init()
rapids_cmake_install_lib_dir(lib_dir)

option(FETCH_RAPIDS "Fetch RAPIDS dependencies" ON)

if (FETCH_RAPIDS)
  include(cmake/thirdparty/get_cccl.cmake)
  include(cmake/thirdparty/get_rmm.cmake)
  include(cmake/thirdparty/get_raft.cmake)
else()
  find_package(CCCL REQUIRED)
  find_package(RMM REQUIRED)
  find_package(RAFT REQUIRED)
endif()
include(${rapids-cmake-dir}/cpm/rapids_logger.cmake)
# generate logging macros
rapids_cpm_rapids_logger(BUILD_EXPORT_SET cuopt-exports INSTALL_EXPORT_SET cuopt-exports)
create_logger_macros(CUOPT "cuopt::default_logger()" include/cuopt)

if(BUILD_TESTS)
  include(cmake/thirdparty/get_gtest.cmake)
endif()

set(CUOPT_SRC_FILES )
add_subdirectory(src)
add_library(cuopt SHARED
  ${CUOPT_SRC_FILES}
)

set_target_properties(cuopt
  PROPERTIES BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"

  # set target compile options
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_definitions(cuopt PUBLIC "CUOPT_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_${LIBCUOPT_LOGGING_LEVEL}")

target_compile_options(cuopt
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUOPT_CXX_FLAGS}>"
  "$<$<COMPILE_LANGUAGE:CUDA>:${CUOPT_CUDA_FLAGS}>"
)

file(WRITE "${CUOPT_BINARY_DIR}/fatbin.ld"
  [=[
SECTIONS
{
  .nvFatBinSegment : { *(.nvFatBinSegment) }
  .nv_fatbin : { *(.nv_fatbin) }
}
]=])
target_link_options(cuopt PRIVATE "${CUOPT_BINARY_DIR}/fatbin.ld")

add_library(cuopt::cuopt ALIAS cuopt)
# ##################################################################################################
# - include paths ---------------------------------------------------------------------------------
target_include_directories(cuopt
  PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libmps_parser/include>"
  "$<INSTALL_INTERFACE:include>"
)

# ##################################################################################################
# - link libraries --------------------------------------------------------------------------------

set(CUOPT_PRIVATE_CUDA_LIBS
  CUDA::curand
  CUDA::cusolver
  OpenMP::OpenMP_CXX)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.4)
  list(PREPEND CUOPT_PRIVATE_CUDA_LIBS CUDA::cublasLt)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libmps_parser)
set(CMAKE_LIBRARY_PATH ${CMAKE_CURRENT_BINARY_DIR}/libmps_parser/)


target_link_libraries(cuopt
  PUBLIC
  CUDA::cublas
  CUDA::cusparse
  rmm::rmm
  rapids_logger::rapids_logger
  CCCL::CCCL
  raft::raft
  cuopt::mps_parser
  PRIVATE
  ${CUOPT_PRIVATE_CUDA_LIBS}
  )

# ##################################################################################################
# - generate tests --------------------------------------------------------------------------------
if(BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif(BUILD_TESTS)

# ##################################################################################################
# - install targets -------------------------------------------------------------------------------
install(TARGETS cuopt mps_parser
  DESTINATION ${lib_dir}
  EXPORT cuopt-exports)

install(DIRECTORY include/cuopt/
  DESTINATION include/cuopt)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cuopt/version_config.hpp
  DESTINATION include/cuopt)

# ###############################################################################################
# - install export -------------------------------------------------------------------------------
set(doc_string
  [=[
Provide targets for cuOpt.

cuOpt library is a collection of GPU accelerated combinatorial optimization algorithms.

]=])

rapids_export(INSTALL cuopt
  EXPORT_SET cuopt-exports
  GLOBAL_TARGETS cuopt
  NAMESPACE cuopt::
  DOCUMENTATION doc_string
)

# ###############################################################################################
# - build export -------------------------------------------------------------------------------
rapids_export(BUILD cuopt
  EXPORT_SET cuopt-exports
  GLOBAL_TARGETS cuopt
  NAMESPACE cuopt::
  DOCUMENTATION doc_string
)

# ##################################################################################################
# - make documentation ----------------------------------------------------------------------------
# requires doxygen and graphviz to be installed
# from build directory, run make docs_cuopt

# doc targets for cuOpt
find_package(Doxygen)

if(Doxygen_FOUND)
  add_custom_command(OUTPUT CUOPT_DOXYGEN
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
    COMMAND doxygen Doxyfile
    VERBATIM)

  add_custom_target(docs_cuopt DEPENDS CUOPT_DOXYGEN)
endif()



list(APPEND CUOPT_CXX_FLAGS -g -O0)
add_executable(cuopt_cli cuopt_cli.cpp)
target_compile_options(cuopt_cli
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUOPT_CXX_FLAGS}>"
  "$<$<COMPILE_LANGUAGE:CUDA>:${CUOPT_CUDA_FLAGS}>"
)

target_include_directories(cuopt_cli
  PRIVATE  
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(cuopt_cli
  PUBLIC
  cuopt
  OpenMP::OpenMP_CXX
)
set_property(TARGET cuopt_cli PROPERTY INSTALL_RPATH "$ORIGIN/../${lib_dir}")

# FIXME:: Is this the right way? 
install(TARGETS cuopt_cli
  DESTINATION ${CMAKE_INSTALL_BINDIR})


option(BUILD_BENCHMARKS "Build benchmarks" ON)
if(BUILD_BENCHMARKS)
  add_executable(solve_MPS_file ../benchmarks/linear_programming/cuopt/run_mip.cpp)
  target_compile_options(solve_MPS_file
    PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUOPT_CXX_FLAGS}>"
    "$<$<COMPILE_LANGUAGE:CUDA>:${CUOPT_CUDA_FLAGS}>"
  )
  target_link_libraries(solve_MPS_file
    PUBLIC
    cuopt
    OpenMP::OpenMP_CXX
  )
endif()
