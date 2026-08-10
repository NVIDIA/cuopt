# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Single binary for all numopt tests that require internal symbols (cuopt_static).
# Add new internal numopt tests here instead of creating separate STATIC_LIB targets.
ConfigureTest(NUMOPT_INTERNAL_TEST
    ${CUOPT_TEST_DIR}/internal/main.cu
    # dual_simplex
    ${CUOPT_TEST_DIR}/dual_simplex/unit_tests/solve.cpp
    ${CUOPT_TEST_DIR}/dual_simplex/unit_tests/solve_barrier.cu
    ${CUOPT_TEST_DIR}/dual_simplex/unit_tests/right_looking_ldlt.cpp
    # linear_programming
    ${CUOPT_TEST_DIR}/linear_programming/pdlp_test.cu
    ${CUOPT_TEST_DIR}/linear_programming/unit_tests/optimization_problem_test.cu
    ${CUOPT_TEST_DIR}/linear_programming/unit_tests/presolve_test.cu
    ${CUOPT_TEST_DIR}/linear_programming/parser_test.cpp
    # mip
    ${CUOPT_TEST_DIR}/mip/miplib_test.cu
    ${CUOPT_TEST_DIR}/mip/problem_test.cu
    ${CUOPT_TEST_DIR}/mip/elim_var_remap_test.cu
    ${CUOPT_TEST_DIR}/mip/bounds_standardization_test.cu
    ${CUOPT_TEST_DIR}/mip/multi_probe_test.cu
    ${CUOPT_TEST_DIR}/mip/cuts_test.cu
    ${CUOPT_TEST_DIR}/mip/unit_test.cu
    ${CUOPT_TEST_DIR}/mip/integer_with_real_bounds.cu
    ${CUOPT_TEST_DIR}/mip/empty_fixed_problems_test.cu
    ${CUOPT_TEST_DIR}/mip/presolve_test.cu
    ${CUOPT_TEST_DIR}/mip/gf2_presolve_test.cpp
    ${CUOPT_TEST_DIR}/mip/termination_test.cu
    ${CUOPT_TEST_DIR}/mip/determinism_test.cu
    # socp
    ${CUOPT_TEST_DIR}/socp/second_order_cone_kernels.cu
    ${CUOPT_TEST_DIR}/socp/sparse_augmented_kkt_test.cu
    ${CUOPT_TEST_DIR}/socp/solve_barrier_socp.cu
    ${CUOPT_TEST_DIR}/socp/general_quadratic_test.cu
    LABELS numopt
    STATIC_LIB)
