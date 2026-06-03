# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

ConfigureTest(SOCP_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/second_order_cone_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/solve_barrier_socp.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/general_quadratic_test.cu
    LABELS numopt)
