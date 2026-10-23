# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.30)
project(cuopt_split_package_smoke LANGUAGES C CXX)

set(COMPONENT "" CACHE STRING "cuOpt component to link: client, mathopt or routing")
if(NOT COMPONENT MATCHES "^(client|mathopt|routing)$")
  message(FATAL_ERROR "COMPONENT must be client, mathopt or routing, got '${COMPONENT}'")
endif()

# Wheels install into lib64, which CMake skips on Debian-derived distros.
set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS ON)

find_package(cuopt REQUIRED)
if(NOT TARGET cuopt::${COMPONENT})
  message(FATAL_ERROR "cuopt::${COMPONENT} was not found in this installation")
endif()

if(COMPONENT STREQUAL "mathopt")
  set(SMOKE_SOURCE ${CMAKE_CURRENT_LIST_DIR}/../../docs/cuopt/source/cuopt-c/convex/examples/simple_lp_example.c)
else()
  set(SMOKE_SOURCE ${COMPONENT}.cpp)
endif()

add_executable(smoke_${COMPONENT} ${SMOKE_SOURCE})
target_compile_features(smoke_${COMPONENT} PRIVATE cxx_std_20)
# Keep the component in the link even if the program uses none of it, so a library that cannot
# load fails at startup.
target_link_options(smoke_${COMPONENT} PRIVATE LINKER:--no-as-needed)
target_link_libraries(smoke_${COMPONENT} PRIVATE cuopt::${COMPONENT})
