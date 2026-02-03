# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# ##################################################################################################
# - MIP tests ----------------------------------------------------------------------
ConfigureTest(MIP_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/miplib_test.cu
)
ConfigureTest(PROBLEM_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/problem_test.cu
)
ConfigureTest(ELIM_VAR_REMAP_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/elim_var_remap_test.cu
)
ConfigureTest(STANDARDIZATION_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/bounds_standardization_test.cu
)
ConfigureTest(MULTI_PROBE_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_probe_test.cu
)
ConfigureTest(INCUMBENT_CALLBACK_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/incumbent_callback_test.cu
)
ConfigureTest(DOC_EXAMPLE_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/doc_example_test.cu
)
ConfigureTest(CUTS_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/cuts_test.cu
)
ConfigureTest(UNIT_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/integer_with_real_bounds.cu
)
ConfigureTest(EMPTY_FIXED_PROBLEMS_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/empty_fixed_problems_test.cu
)
ConfigureTest(PRESOLVE_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/presolve_test.cu
)
# Disable for now
# ConfigureTest(FEASIBILITY_JUMP_TEST
#    ${CMAKE_CURRENT_SOURCE_DIR}/feasibility_jump_tests.cu
# )
ConfigureTest(MIP_TERMINATION_STATUS_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/termination_test.cu
)
