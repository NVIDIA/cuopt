# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

ConfigureTest(DUAL_SIMPLEX_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/solve.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/solve_barrier.cu
)
