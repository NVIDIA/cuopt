# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

add_subdirectory(distance_engine)
add_subdirectory(linear_programming/data_model)
add_subdirectory(linear_programming/mps_parser)
add_subdirectory(linear_programming/solver)
add_subdirectory(routing)

set(cython_sources)
set(linked_libraries cuopt::cuopt)

rapids_cython_create_modules(SOURCE_FILES "${cython_sources}" LINKED_LIBRARIES "${linked_libraries}" ASSOCIATED_TARGETS cuopt
                                                                               CXX)
