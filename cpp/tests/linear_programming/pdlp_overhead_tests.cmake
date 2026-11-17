# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Included by linear_programming/CMakeLists.txt after ConfigureTest is defined.
# Static linkage also resolves the logger used by capture-invalidation recovery.
ConfigureTest(PDLP_OVERHEAD_TEST
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/reflected_projection_direct_test.cu
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/batched_halpern_test.cu
    LABELS numopt
    STATIC_LIB)
