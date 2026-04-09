# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

# Add our custom Find modules to the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/thirdparty")

include(GNUInstallDirs)

set(RAFT_NVTX ON)
include(../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(CUOPT)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
message(STATUS "CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")

project(
  CUOPT
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA C
)

# Disable C++20 module scanning as the codebase doesn't use modules
set(CMAKE_CXX_SCAN_FOR_MODULES OFF CACHE BOOL "Disable C++20 module scanning")

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
option(BUILD_LP_ONLY "Build only linear programming components, exclude routing and MIP-specific files" OFF)
option(SKIP_C_PYTHON_ADAPTERS "Skip building C and Python adapter files (cython_solve.cu and cuopt_c.cpp)" OFF)
option(SKIP_ROUTING_BUILD "Skip building routing components" OFF)
option(WRITE_FATBIN "Enable fatbin writing" ON)
option(HOST_LINEINFO "Build with debug line information for host code" OFF)

message(VERBOSE "cuOpt: Enable nvcc -lineinfo: ${CMAKE_CUDA_LINEINFO}")
message(VERBOSE "cuOpt: Build cuOpt unit-tests: ${BUILD_TESTS}")
message(VERBOSE "cuOpt: Build cuOpt multigpu tests: ${BUILD_TESTS}")
message(VERBOSE "cuOpt: Disable OpenMP: ${DISABLE_OPENMP}")
message(VERBOSE "cuOpt: Build LP-only mode: ${BUILD_LP_ONLY}")
message(VERBOSE "cuOpt: Skip C/Python adapters: ${SKIP_C_PYTHON_ADAPTERS}")
message(VERBOSE "cuOpt: Skip routing build: ${SKIP_ROUTING_BUILD}")
message(VERBOSE "cuOpt: Build with debug line information for host code: ${HOST_LINEINFO}")
message(VERBOSE "cuOpt: fatbin: ${WRITE_FATBIN}")

# ##################################################################################################
# - compiler options ------------------------------------------------------------------------------

# CUDA runtime
rapids_cuda_init_runtime(USE_STATIC ON)

rapids_find_package(CUDAToolkit REQUIRED
  BUILD_EXPORT_SET cuopt-exports
  INSTALL_EXPORT_SET cuopt-exports
)

set(CUOPT_CXX_FLAGS "")
set(CUOPT_CUDA_FLAGS "")

if(CMAKE_COMPILER_IS_GNUCXX)
  list(APPEND CUOPT_CXX_FLAGS -Werror -Wno-error=deprecated-declarations)
endif(CMAKE_COMPILER_IS_GNUCXX)

# Papilo pulls in Boost.Multiprecision float128 support, which expects quadmath.h from the GCC
# toolchain internals. Conda clang ships libquadmath, but does not surface the matching GCC
# internal include directory by default. Add it late in the search order so clang still prefers its
# own builtin intrinsic headers.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} --print-file-name=libquadmath.a
    OUTPUT_VARIABLE CUOPT_QUADMATH_LIB
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(IS_ABSOLUTE "${CUOPT_QUADMATH_LIB}")
    get_filename_component(CUOPT_QUADMATH_LIBDIR "${CUOPT_QUADMATH_LIB}" DIRECTORY)
    set(CUOPT_QUADMATH_INCLUDEDIR "${CUOPT_QUADMATH_LIBDIR}/include")

    if(EXISTS "${CUOPT_QUADMATH_INCLUDEDIR}/quadmath.h")
      message(STATUS "Adding clang fallback include for quadmath: ${CUOPT_QUADMATH_INCLUDEDIR}")
      add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-idirafter${CUOPT_QUADMATH_INCLUDEDIR}>")
    endif()
  endif()
endif()

# To use sanitizer with cuda runtime, one must follow a few steps:
# 1. Run the binary with env var set: LD_PRELOAD="$(gcc -print-file-name=libasan.so)" ASAN_OPTIONS='protect_shadow_gap=0:replace_intrin=0'
# 2. (Optional) To run with a debugger (gdb or cuda-gdb) use the additional ASAN option alloc_dealloc_mismatch=0
if(BUILD_SANITIZER)
  list(APPEND CUOPT_CXX_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer -g)
  if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    list(APPEND CUOPT_CXX_FLAGS -Wno-error=maybe-uninitialized)
  endif()
  add_link_options(-fsanitize=address,undefined)
endif(BUILD_SANITIZER)

# To use ThreadSanitizer:
# 1. Build with clang and the -tsan flag
# 2. Run the binary with env var set: OMP_TOOL_LIBRARIES=/usr/lib/llvm-17/lib/libarcher.so ARCHER_OPTIONS='verbose=1' TSAN_OPTIONS='suppresions=cpp/utilities/tsan_suppressions.txt:ignore_noninstrumented_modules=1:halt_on_error=1'
#     Replace with local llvm install path. libarcher.so must be presetn
if(BUILD_TSAN)
  message(STATUS "Building with ThreadSanitizer enabled")
  list(APPEND CUOPT_CXX_FLAGS -fsanitize=thread -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=thread)
endif(BUILD_TSAN)

# To use MemorySanitizer:
# 1. Build with clang and the -msan flag (MemorySanitizer requires clang)
# 2. Run the binary with env var set: MSAN_OPTIONS='halt_on_error=1'
# Note: MemorySanitizer requires all code (including libraries) to be instrumented for accurate results
if(BUILD_MSAN)
  message(STATUS "Building with MemorySanitizer enabled")
  list(APPEND CUOPT_CXX_FLAGS -fsanitize=memory -fno-omit-frame-pointer -g -fsanitize-memory-track-origins=1)
  add_link_options(-fsanitize=memory)
endif(BUILD_MSAN)

# Note: -UNDEBUG is applied via CUOPT_CXX_FLAGS / CUOPT_CUDA_FLAGS (not add_definitions)
# to avoid leaking into dependencies that are built in-tree.
#
# For CXX: -UNDEBUG is applied per-source-file (not target-wide) to exclude gRPC
# infrastructure files. Those files include abseil headers, and abseil's shared library
# on conda-forge doesn't export Mutex::Dtor() in NDEBUG builds (abseil-cpp#1624).
# Keeping NDEBUG defined for gRPC files makes the header inline an empty Dtor(),
# avoiding the missing symbol at runtime.  Additionally, gRPC files are always
# compiled with -DNDEBUG (see below) so Debug builds also avoid the missing symbol.
if(DEFINE_ASSERT)
  add_definitions(-DASSERT_MODE)
  list(APPEND CUOPT_CUDA_FLAGS -UNDEBUG)
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
                                       "OFF")
message(VERBOSE "CUOPT: LIBCUOPT_LOGGING_LEVEL = '${LIBCUOPT_LOGGING_LEVEL}'.")

message("-- Building with logging level = ${LIBCUOPT_LOGGING_LEVEL}")

message("-- Building for GPU_ARCHS = '${CMAKE_CUDA_ARCHITECTURES}'")
message("-- Host target architecture = '${CMAKE_SYSTEM_PROCESSOR}'")

# make the flags global in order to propagate flags to test cmake files
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.9)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -static-global-template-stub=false")
endif()
list(APPEND CUOPT_CUDA_FLAGS -Werror=cross-execution-space-call -Wno-deprecated-declarations -Xcompiler=-Werror --default-stream=per-thread)
if("${CMAKE_CUDA_HOST_COMPILER}" MATCHES "clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  list(APPEND CUOPT_CUDA_FLAGS -Xcompiler=-Wall)
else()
  list(APPEND CUOPT_CUDA_FLAGS -Xcompiler=-Wall -Wno-error=non-template-friend)
endif()
list(APPEND CUOPT_CUDA_FLAGS -Xfatbin=-compress-all)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
  list(APPEND CUOPT_CUDA_FLAGS -Xfatbin=--compress-level=3)
endif()
list(APPEND CUOPT_CUDA_FLAGS -fopenmp)

# Add jobserver flags for parallel compilation if PARALLEL_LEVEL is set
if(PARALLEL_LEVEL AND NOT "${PARALLEL_LEVEL}" STREQUAL "")
  message(STATUS "Enabling nvcc parallel compilation support")
  list(APPEND CUOPT_CUDA_FLAGS --threads=0 --split-compile=0)
  if(USE_NVCC_JOBSERVER AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    message(STATUS "Enabling nvcc jobserver support (NVCC >= 13.0)")
    list(APPEND CUOPT_CUDA_FLAGS --jobserver)
  endif()
endif()

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
  # Source-built RMM can hide out-of-line utility symbols such as
  # rmm::align_up / rmm::get_current_cuda_device when built with hidden visibility on clang.
  # Force default visibility on the fetched rmm target until this is fixed upstream/figured out.
  if(TARGET rmm)
    set_target_properties(rmm PROPERTIES CXX_VISIBILITY_PRESET default)
  endif()
else()
  find_package(CCCL REQUIRED)
  find_package(RMM REQUIRED)
  find_package(RAFT REQUIRED)
endif()

FetchContent_Declare(
  papilo
  GIT_REPOSITORY "https://github.com/scipopt/papilo.git"
  # We would want to get the main branch. However, the main branch
  # does not have some of the presolvers and settings that we need
  # Mainly, probing and clique merging.
  # This is the reason we are using the development branch
  # from Oct 12, 2025. Once these changes are merged into the main branch,
  #we can switch to the main branch.
  GIT_TAG "741a2b9c8155b249d6df574d758b4d97d4417520"
  GIT_PROGRESS TRUE
  EXCLUDE_FROM_ALL
  SYSTEM
)

find_package(TBB REQUIRED)
set(BUILD_TESTING OFF CACHE BOOL "Disable test build for papilo")
set(PAPILO_NO_BINARIES ON)
option(LUSOL "Disable LUSOL" OFF)

FetchContent_MakeAvailable(papilo)

# PSLP - Lightweight C presolver for linear programs
# https://github.com/dance858/PSLP
FetchContent_Declare(
  pslp
  GIT_REPOSITORY "https://github.com/dance858/PSLP.git"
  GIT_TAG "v0.0.8"
  GIT_PROGRESS TRUE
  EXCLUDE_FROM_ALL
  SYSTEM
)

# Build PSLP as static to embed in cuopt (avoids runtime library path issues)
set(BUILD_SHARED_LIBS_SAVED ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
FetchContent_MakeAvailable(pslp)
set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_SAVED})


include(${rapids-cmake-dir}/cpm/rapids_logger.cmake)
# generate logging macros
rapids_cpm_rapids_logger(BUILD_EXPORT_SET cuopt-exports INSTALL_EXPORT_SET cuopt-exports)
create_logger_macros(CUOPT "cuopt::default_logger()" include/cuopt)

find_package(CUDSS REQUIRED)

# ##################################################################################################
# - gRPC and Protobuf setup (REQUIRED) ------------------------------------------------------------

# gRPC is required for this branch - it provides remote execution features
# gRPC can come from either:
# - an installed CMake package (gRPCConfig.cmake), or
# - an in-tree build (e.g. python/libcuopt uses FetchContent(grpc), which defines gRPC::grpc++).
if(NOT TARGET gRPC::grpc++)
  find_package(gRPC CONFIG REQUIRED)
endif()

# Find Protobuf (should come with gRPC, but verify)
if(NOT TARGET protobuf::libprotobuf)
  find_package(protobuf CONFIG REQUIRED)
endif()

set(CUOPT_ENABLE_GRPC ON)
add_compile_definitions(CUOPT_ENABLE_GRPC)
message(STATUS "gRPC enabled (target gRPC::grpc++ is available)")

# Find protoc compiler (provided by config package or target)
if(TARGET protobuf::protoc)
  get_target_property(_PROTOBUF_PROTOC protobuf::protoc IMPORTED_LOCATION_RELEASE)
  if(NOT _PROTOBUF_PROTOC)
    get_target_property(_PROTOBUF_PROTOC protobuf::protoc IMPORTED_LOCATION)
  endif()
else()
  find_package(protobuf CONFIG REQUIRED)
  get_target_property(_PROTOBUF_PROTOC protobuf::protoc IMPORTED_LOCATION_RELEASE)
  if(NOT _PROTOBUF_PROTOC)
    get_target_property(_PROTOBUF_PROTOC protobuf::protoc IMPORTED_LOCATION)
  endif()
endif()

if(NOT _PROTOBUF_PROTOC)
  message(FATAL_ERROR "protoc not found (Protobuf_PROTOC_EXECUTABLE is empty)")
endif()

# Find grpc_cpp_plugin
if(TARGET grpc_cpp_plugin)
  set(_GRPC_CPP_PLUGIN_EXECUTABLE "$<TARGET_FILE:grpc_cpp_plugin>")
else()
  find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
  if(NOT _GRPC_CPP_PLUGIN_EXECUTABLE)
    message(FATAL_ERROR "grpc_cpp_plugin not found")
  endif()
endif()

# Generate C++ code from cuopt_remote.proto (base message definitions)
set(PROTO_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/grpc/cuopt_remote.proto")
set(PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/cuopt_remote.pb.cc")
set(PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/cuopt_remote.pb.h")

add_custom_command(
  OUTPUT "${PROTO_SRCS}" "${PROTO_HDRS}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR}
       --proto_path ${CMAKE_CURRENT_SOURCE_DIR}/src/grpc
       ${PROTO_FILE}
  DEPENDS ${PROTO_FILE}
  COMMENT "Generating C++ code from cuopt_remote.proto"
  VERBATIM
)

# Generate gRPC service code from cuopt_remote_service.proto
set(GRPC_PROTO_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/grpc/cuopt_remote_service.proto")
set(GRPC_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/cuopt_remote_service.pb.cc")
set(GRPC_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/cuopt_remote_service.pb.h")
set(GRPC_SERVICE_SRCS "${CMAKE_CURRENT_BINARY_DIR}/cuopt_remote_service.grpc.pb.cc")
set(GRPC_SERVICE_HDRS "${CMAKE_CURRENT_BINARY_DIR}/cuopt_remote_service.grpc.pb.h")

add_custom_command(
  OUTPUT "${GRPC_PROTO_SRCS}" "${GRPC_PROTO_HDRS}" "${GRPC_SERVICE_SRCS}" "${GRPC_SERVICE_HDRS}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR}
       --grpc_out ${CMAKE_CURRENT_BINARY_DIR}
       --plugin=protoc-gen-grpc=${_GRPC_CPP_PLUGIN_EXECUTABLE}
       --proto_path ${CMAKE_CURRENT_SOURCE_DIR}/src/grpc
       ${GRPC_PROTO_FILE}
  DEPENDS ${GRPC_PROTO_FILE} ${PROTO_FILE}
  COMMENT "Generating gRPC C++ code from cuopt_remote_service.proto"
  VERBATIM
)

message(STATUS "gRPC protobuf code generation configured")

if(BUILD_TESTS)
  include(cmake/thirdparty/get_gtest.cmake)
endif()

set(CUOPT_SRC_FILES )
add_subdirectory(src)
if (HOST_LINEINFO)
  set_source_files_properties(${CUOPT_SRC_FILES} DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTIES COMPILE_OPTIONS "-g1")
endif()

# Apply -UNDEBUG only to solver source files (not gRPC infrastructure).
# Must happen before gRPC files are appended to CUOPT_SRC_FILES.
# Uses APPEND to preserve any existing per-file options (e.g. -g1 from HOST_LINEINFO).
if(DEFINE_ASSERT)
  set_property(SOURCE ${CUOPT_SRC_FILES} DIRECTORY ${CMAKE_SOURCE_DIR}
    APPEND PROPERTY COMPILE_OPTIONS "-UNDEBUG")
endif()

# Add gRPC mapper files and generated protobuf sources
set(GRPC_INFRA_FILES
  ${PROTO_SRCS}
  ${GRPC_PROTO_SRCS}
  ${GRPC_SERVICE_SRCS}
  src/grpc/grpc_problem_mapper.cpp
  src/grpc/grpc_solution_mapper.cpp
  src/grpc/grpc_settings_mapper.cpp
  src/grpc/grpc_service_mapper.cpp
  src/grpc/client/grpc_client.cpp
  src/grpc/client/solve_remote.cpp
)
list(APPEND CUOPT_SRC_FILES ${GRPC_INFRA_FILES})

# Always keep NDEBUG defined for gRPC infrastructure files so that abseil
# headers inline Mutex::Dtor() instead of emitting an external call.
# The conda-forge abseil shared library is built with NDEBUG and does not
# export that symbol (abseil-cpp#1624).  Without this, Debug builds fail
# at runtime with "undefined symbol: absl::…::Mutex::Dtor".
set_property(SOURCE ${GRPC_INFRA_FILES} DIRECTORY ${CMAKE_SOURCE_DIR}
  APPEND PROPERTY COMPILE_OPTIONS "-DNDEBUG")

add_library(cuopt SHARED
  ${CUOPT_SRC_FILES}
)

set_target_properties(cuopt
  PROPERTIES BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"

  # set target compile options
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  INTERFACE_POSITION_INDEPENDENT_CODE ON
  CXX_SCAN_FOR_MODULES OFF
)

target_compile_definitions(cuopt PUBLIC "CUOPT_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_${LIBCUOPT_LOGGING_LEVEL}")

target_compile_options(cuopt
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUOPT_CXX_FLAGS}>"
  "$<$<COMPILE_LANGUAGE:CUDA>:${CUOPT_CUDA_FLAGS}>"
)

if(WRITE_FATBIN)
  file(WRITE "${CUOPT_BINARY_DIR}/fatbin.ld"
    [=[
  SECTIONS
  {
    .nvFatBinSegment : { *(.nvFatBinSegment) }
    .nv_fatbin : { *(.nv_fatbin) }
  }
  ]=])
  target_link_options(cuopt PRIVATE "${CUOPT_BINARY_DIR}/fatbin.ld")
endif()

add_library(cuopt::cuopt ALIAS cuopt)
# ##################################################################################################
# - include paths ---------------------------------------------------------------------------------
message(STATUS "target include directories CUDSS_INCLUDES = ${CUDSS_INCLUDE}")

# Adding Papilo as a system include messes up clang's include resolution if papilo is already installed as a conda package
target_include_directories(cuopt PRIVATE
  "${papilo_SOURCE_DIR}/src"
  "${papilo_BINARY_DIR}"
)

target_include_directories(cuopt SYSTEM PRIVATE
  "${pslp_SOURCE_DIR}/include"
)

target_include_directories(cuopt
  PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/grpc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/grpc/client"
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${CUDSS_INCLUDE}"
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libmps_parser/include>"
  INTERFACE
  "$<INSTALL_INTERFACE:include>"
  ${CUDSS_INCLUDE}
)

# Link PSLP by file to avoid export dependency tracking
target_link_libraries(cuopt PRIVATE $<TARGET_FILE:PSLP>)
add_dependencies(cuopt PSLP)

# ##################################################################################################
# - link libraries --------------------------------------------------------------------------------

set(CUOPT_PRIVATE_CUDA_LIBS
  CUDA::curand
  CUDA::cusolver
  TBB::tbb
  OpenMP::OpenMP_CXX)

list(PREPEND CUOPT_PRIVATE_CUDA_LIBS CUDA::cublasLt)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libmps_parser)
set(CMAKE_LIBRARY_PATH ${CMAKE_CURRENT_BINARY_DIR}/libmps_parser/)


# Pass CUDSS_MT_LIB_FILE_NAME as a compile definition
get_filename_component(CUDSS_MT_LIB_FILE_NAME "${CUDSS_MT_LIB_FILE}" NAME)
target_compile_definitions(cuopt PRIVATE CUDSS_MT_LIB_FILE_NAME="${CUDSS_MT_LIB_FILE_NAME}")

execute_process(
  COMMAND git rev-parse --short HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message("-- Building with GIT_COMMIT_HASH = '${GIT_COMMIT_HASH}'")

# Generate build_info.hpp from template
# configure_file() only updates the output if content changes, avoiding unnecessary rebuilds
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utilities/build_info.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/utilities/build_info.hpp
  @ONLY
)

# Add the generated include directory
target_include_directories(cuopt PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)

list(JOIN CMAKE_CUDA_ARCHITECTURES "," JOINED_CUDA_ARCHITECTURES)
target_compile_definitions(cuopt PUBLIC
  CUOPT_CUDA_ARCHITECTURES="${JOINED_CUDA_ARCHITECTURES}"
  CUOPT_CPU_ARCHITECTURE="${CMAKE_SYSTEM_PROCESSOR}")

target_link_libraries(cuopt
  PUBLIC
  CUDA::cublas
  CUDA::cusparse
  rmm::rmm
  rapids_logger::rapids_logger
  CCCL::CCCL
  raft::raft
  cuopt::mps_parser
  ${CUDSS_LIB_FILE}
  PRIVATE
  ${CUOPT_PRIVATE_CUDA_LIBS}
  protobuf::libprotobuf
  gRPC::grpc++
  )


# ##################################################################################################
# - generate tests --------------------------------------------------------------------------------
if(BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif(BUILD_TESTS)

# ##################################################################################################
# - install targets -------------------------------------------------------------------------------

# allows for CPack component builds and install location
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_ALL runtime dev)
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local")

#If using cpack to create a deb package
if(CPACK_GENERATOR STREQUAL "DEB")
  set(_BIN_DEST "bin")
  set(_LIB_DEST "lib")
  set(_INCLUDE_DEST "lib/cuopt")

#If building locally use the Default install paths(e.g. for local development or other package types)
else()
  set(_BIN_DEST "${CMAKE_INSTALL_BINDIR}")
  set(_LIB_DEST "${lib_dir}")
  set(_INCLUDE_DEST  include/cuopt/)
endif()

# adds the .so files to the runtime deb package
install(TARGETS cuopt mps_parser
  DESTINATION ${_LIB_DEST}
  COMPONENT runtime
  EXPORT cuopt-exports
)

# adds the .so files to the development deb package
install(TARGETS cuopt mps_parser
  DESTINATION ${_LIB_DEST}
  COMPONENT dev
)

# adds the header files to the development deb package
install(DIRECTORY include/cuopt/
  DESTINATION ${_INCLUDE_DEST}
  COMPONENT dev
)

# adds the version header file to the development deb package
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cuopt/version_config.hpp
  DESTINATION ${_INCLUDE_DEST}
  COMPONENT dev
)
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


rapids_cpm_find(
  argparse 3.2.0
  GLOBAL_TARGETS argparse::argparse
  CPM_ARGS
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG v3.2
  GIT_SHALLOW TRUE
)

if(NOT BUILD_LP_ONLY)
add_executable(cuopt_cli cuopt_cli.cpp)

# PIE executable: auditwheel/patchelf expands .dynstr/RPATH when repairing wheels; non-PIE
# (ET_EXEC) binaries are prone to corrupt segment layout. PIE (ET_DYN) survives RPATH edits.
set_target_properties(cuopt_cli
  PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_SCAN_FOR_MODULES OFF
  POSITION_INDEPENDENT_CODE ON
)

target_compile_options(cuopt_cli
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUOPT_CXX_FLAGS}>"
  "$<$<COMPILE_LANGUAGE:CUDA>:${CUOPT_CUDA_FLAGS}>"
)

target_link_options(cuopt_cli PRIVATE -pie)

target_include_directories(cuopt_cli
  PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
  ${CUDSS_INCLUDE}
  "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(cuopt_cli
  PUBLIC
  cuopt
  OpenMP::OpenMP_CXX
  ${CUDSS_LIBRARIES}
  TBB::tbb
  PRIVATE
  argparse::argparse
)
  # Use RUNPATH when building locally in order to allow LD_LIBRARY_PATH to override the conda env path
if(NOT DEFINED INSTALL_TARGET OR "${INSTALL_TARGET}" STREQUAL "")
  target_link_options(cuopt_cli PRIVATE -Wl,--enable-new-dtags)
endif()
set_property(TARGET cuopt_cli PROPERTY INSTALL_RPATH "$ORIGIN/../${lib_dir}")

# adds the cuopt_cli executable to the runtime deb package
install(TARGETS cuopt_cli
  COMPONENT runtime
  RUNTIME DESTINATION ${_BIN_DEST}
)
endif()


option(BUILD_MIP_BENCHMARKS "Build MIP benchmarks" OFF)
if(BUILD_MIP_BENCHMARKS AND NOT BUILD_LP_ONLY)
  add_executable(solve_MIP ../benchmarks/linear_programming/cuopt/run_mip.cpp)
  target_include_directories(solve_MIP
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  )

  set_target_properties(solve_MIP
    PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_SCAN_FOR_MODULES OFF
  )

  target_compile_options(solve_MIP
    PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUOPT_CXX_FLAGS}>"
    "$<$<COMPILE_LANGUAGE:CUDA>:${CUOPT_CUDA_FLAGS}>"
  )
  target_link_libraries(solve_MIP
    PUBLIC
    cuopt
    OpenMP::OpenMP_CXX
    PRIVATE
  )
  if(NOT DEFINED INSTALL_TARGET OR "${INSTALL_TARGET}" STREQUAL "")
    target_link_options(solve_MIP PRIVATE -Wl,--enable-new-dtags)
  endif()

  target_include_directories(solve_MIP
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )

endif()

option(BUILD_LP_BENCHMARKS "Build LP benchmarks" OFF)
if(BUILD_LP_BENCHMARKS)
  add_executable(solve_LP ../benchmarks/linear_programming/cuopt/run_pdlp.cu)

  set_target_properties(solve_LP
    PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CXX_SCAN_FOR_MODULES OFF
  )

  target_compile_options(solve_LP
    PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUOPT_CXX_FLAGS}>"
    "$<$<COMPILE_LANGUAGE:CUDA>:${CUOPT_CUDA_FLAGS}>"
  )
  target_link_libraries(solve_LP
    PUBLIC
    cuopt
    OpenMP::OpenMP_CXX
    PRIVATE
  )
  if(NOT DEFINED INSTALL_TARGET OR "${INSTALL_TARGET}" STREQUAL "")
    target_link_options(solve_LP PRIVATE -Wl,--enable-new-dtags)
  endif()
endif()

# ##################################################################################################
# - cuopt_grpc_server - gRPC-based remote server --------------------------------------------------

add_executable(cuopt_grpc_server
  src/grpc/server/grpc_server_main.cpp
  src/grpc/server/grpc_server_logger.cpp
  src/grpc/server/grpc_worker.cpp
  src/grpc/server/grpc_worker_infra.cpp
  src/grpc/server/grpc_server_threads.cpp
  src/grpc/server/grpc_pipe_io.cpp
  src/grpc/server/grpc_job_management.cpp
  src/grpc/server/grpc_service_impl.cpp
)

set_target_properties(cuopt_grpc_server
  PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_SCAN_FOR_MODULES OFF
)

target_compile_options(cuopt_grpc_server
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUOPT_CXX_FLAGS}>"
)

target_include_directories(cuopt_grpc_server
  PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/grpc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/grpc/server"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/libmps_parser/include"
  "${CMAKE_CURRENT_BINARY_DIR}"
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
)

find_library(UUID_LIBRARY uuid REQUIRED)

target_link_libraries(cuopt_grpc_server
  PUBLIC
  cuopt
  OpenMP::OpenMP_CXX
  PRIVATE
  protobuf::libprotobuf
  gRPC::grpc++
  ${UUID_LIBRARY}
  argparse::argparse
)

# Use RUNPATH when building locally
target_link_options(cuopt_grpc_server PRIVATE -Wl,--enable-new-dtags)
set_property(TARGET cuopt_grpc_server PROPERTY INSTALL_RPATH "$ORIGIN/../${lib_dir}")

# Install the grpc server executable
install(TARGETS cuopt_grpc_server
  COMPONENT runtime
  RUNTIME DESTINATION ${_BIN_DEST}
)

message(STATUS "Building cuopt_grpc_server (gRPC-based remote solve server)")

# ##################################################################################################
# - CPack has to be the last item in the cmake file-------------------------------------------------
# Used to create an installable deb package for cuOpt

set(CPACK_GENERATOR "DEB")

# Runtime package metadata
execute_process(COMMAND dpkg --print-architecture OUTPUT_VARIABLE DEB_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)

# general package metadata
set(CPACK_DEBIAN_PACKAGE_NAME "cuOpt")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Nvidia")
set(CPACK_PACKAGE_FILE_NAME "cuOpt_${CPACK_PACKAGE_VERSION}_${DEB_ARCH}")

# runtime package metadata
set(CPACK_COMPONENT_RUNTIME_DESCRIPTION "cuOpt runtime components (binaries and shared libraries)")
set(CPACK_COMPONENT_RUNTIME_DISPLAY_NAME "cuOpt Runtime")
set(CPACK_COMPONENT_RUNTIME_GROUP "Runtime")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_MAINTAINER "NVIDIA")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "cuopt")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_FILE_NAME "cuopt_${PROJECT_VERSION}_${DEB_ARCH}")

# Dev package metadata
set(CPACK_COMPONENT_DEV_DESCRIPTION "cuOpt development files (headers, symlinks, etc.)")
set(CPACK_COMPONENT_DEV_DISPLAY_NAME "cuOpt Development")
set(CPACK_COMPONENT_DEV_GROUP "Development")
set(CPACK_DEBIAN_DEV_PACKAGE_MAINTAINER "NVIDIA")
set(CPACK_DEBIAN_DEV_PACKAGE_NAME "cuopt-dev")
set(CPACK_DEBIAN_DEV_PACKAGE_FILE_NAME "cuopt-dev_${PROJECT_VERSION}_${DEB_ARCH}")

# MUST BE THE LAST ITEM IN THE CMAKE FILE!!!
include(CPack)
