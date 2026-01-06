# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# MsgPack Serializer - demonstrates pluggable serialization
#
# This builds a separate shared library that implements the remote_serializer_t interface
# using MsgPack instead of Protocol Buffers.

find_package(msgpack-cxx QUIET)

if(msgpack-cxx_FOUND)
  message(STATUS "Building msgpack serializer plugin")

  add_library(cuopt_msgpack_serializer SHARED
    msgpack_serializer.cpp
  )

  target_include_directories(cuopt_msgpack_serializer
    PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/libmps_parser/include"
    "${CMAKE_SOURCE_DIR}/src"
    ${CUDA_INCLUDE_DIRS}
  )

  # Required for RMM headers
  target_compile_definitions(cuopt_msgpack_serializer
    PRIVATE
    LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
  )

  target_link_libraries(cuopt_msgpack_serializer
    PRIVATE
    msgpack-cxx
    rmm::rmm
    raft::raft
    cuopt
  )

  set_target_properties(cuopt_msgpack_serializer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
  )

  # Install the plugin
  install(TARGETS cuopt_msgpack_serializer
    COMPONENT runtime
    LIBRARY DESTINATION lib
  )
else()
  message(STATUS "msgpack-cxx not found, skipping msgpack serializer plugin")
endif()
