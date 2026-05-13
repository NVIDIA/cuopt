# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

if(NOT CUOPT_PYTHON_COMPONENT STREQUAL "LP")
  add_subdirectory(distance_engine)
endif()
if(NOT CUOPT_PYTHON_COMPONENT STREQUAL "ROUTING")
  add_subdirectory(linear_programming/data_model)
  add_subdirectory(linear_programming/solver)
endif()

# We don't need to have mps_parser within cuOpt
# Remove subdirectory addition in future
if(NOT CUOPT_PYTHON_COMPONENT STREQUAL "LP")
  add_subdirectory(routing)
endif()

set(cython_sources)
set(linked_libraries cuopt::cuopt)
if(NOT CUOPT_PYTHON_COMPONENT STREQUAL "ROUTING")
  list(APPEND linked_libraries cuopt::mps_parser)
endif()

rapids_cython_create_modules(SOURCE_FILES "${cython_sources}" LINKED_LIBRARIES "${linked_libraries}" ASSOCIATED_TARGETS cuopt
                                                                               CXX)
