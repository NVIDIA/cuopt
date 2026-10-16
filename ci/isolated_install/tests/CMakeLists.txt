# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Sample gtest that builds against ONE installed cuOpt component only.
#   cmake -S tests -B build -DCOMPONENT=<client|mathopt|routing> -DCMAKE_PREFIX_PATH=<...>
cmake_minimum_required(VERSION 3.30)

set(COMPONENT "" CACHE STRING "cuOpt component under test: client, mathopt or routing")
if(NOT COMPONENT MATCHES "^(client|mathopt|routing)$")
  message(FATAL_ERROR "COMPONENT must be client, mathopt or routing (got '${COMPONENT}')")
endif()

if(COMPONENT STREQUAL "client")
  project(isolated_${COMPONENT} LANGUAGES CXX)
else()
  project(isolated_${COMPONENT} LANGUAGES CXX CUDA)
endif()

find_package(cuopt CONFIG REQUIRED)
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  # CI containers don't ship gtest; fetch a pinned release.
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

if(NOT TARGET cuopt::${COMPONENT})
  message(FATAL_ERROR "Target cuopt::${COMPONENT} not provided by the installed package")
endif()

if(COMPONENT STREQUAL "routing")
  set(_src ${COMPONENT}/${COMPONENT}_test.cu)
else()
  set(_src ${COMPONENT}/${COMPONENT}_test.cpp)
endif()

add_executable(isolated_${COMPONENT}_test ${_src})
target_compile_features(isolated_${COMPONENT}_test PRIVATE cxx_std_20)
# Only the component under test (plus whatever it pulls in transitively) is linked.
target_link_libraries(isolated_${COMPONENT}_test PRIVATE cuopt::${COMPONENT}
                                                         GTest::gtest GTest::gtest_main)
if(COMPONENT STREQUAL "client")
  # The client exports nothing through its installed headers, so the test dlopen()s the library.
  target_compile_definitions(isolated_client_test
                             PRIVATE CUOPT_CLIENT_LIB="$<TARGET_FILE:cuopt::client>")
  target_link_libraries(isolated_client_test PRIVATE ${CMAKE_DL_LIBS})
endif()
