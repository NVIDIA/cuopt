# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(cython_sources internals.pyx)
set(linked_libraries cuopt::cuopt)
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX internals_
  ASSOCIATED_TARGETS cuopt
)

target_include_directories(internals_internals PRIVATE "${CMAKE_CURRENT_LIST_DIR}")

# Set rpath for Cython modules to find libcuopt.so from the libcuopt wheel
set(libcuopt_rpaths
  "$ORIGIN/../../../libcuopt/lib64"
  "$ORIGIN/../../../libcuopt/lib"
)
foreach(_cython_target IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  set_property(TARGET ${_cython_target} PROPERTY INSTALL_RPATH ${libcuopt_rpaths} APPEND)
endforeach()
