# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(cython_sources data_model_wrapper.pyx)
set(linked_libraries cuopt::mps_parser)

rapids_cython_create_modules(SOURCE_FILES "${cython_sources}" LINKED_LIBRARIES "${linked_libraries}" ASSOCIATED_TARGETS cuopt
                                                                               CXX)

# Set rpath for Cython modules to find libmps_parser.so from the libcuopt wheel
set(libcuopt_rpaths
  "$ORIGIN/../../../libcuopt/lib64"
  "$ORIGIN/../../../libcuopt/lib"
)
foreach(_cython_target IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  set_property(TARGET ${_cython_target} PROPERTY INSTALL_RPATH ${libcuopt_rpaths} APPEND)
endforeach()
