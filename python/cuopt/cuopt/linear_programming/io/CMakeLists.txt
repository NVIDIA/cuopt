# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(cython_sources parser_wrapper.pyx)
set(linked_libraries cuopt::client)

rapids_cython_create_modules(SOURCE_FILES "${cython_sources}" LINKED_LIBRARIES "${linked_libraries}" ASSOCIATED_TARGETS cuopt
                                                                               CXX)

# parser_wrapper.pyx cimports mathopt headers; mathopt-dev is now a separate wheel, so
# add its include dir without linking the library.
target_include_directories(parser_wrapper PRIVATE
    $<TARGET_PROPERTY:cuopt::mathopt,INTERFACE_INCLUDE_DIRECTORIES>
)
