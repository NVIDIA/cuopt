# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# ##################################################################################################
# - MIP tests that use only public API (no internal symbols) ---------------------------------------
# Internal MIP tests are consolidated into NUMOPT_INTERNAL_TEST (cpp/tests/internal/CMakeLists.txt).
ConfigureTest(SEMI_CONTINUOUS_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/semi_continuous_test.cu
    LABELS numopt)
ConfigureTest(INCUMBENT_CALLBACK_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/incumbent_callback_test.cu
    LABELS numopt)
ConfigureTest(DOC_EXAMPLE_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/doc_example_test.cu
    LABELS numopt)
ConfigureTest(HEURISTICS_HYPER_PARAMS_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/heuristics_hyper_params_test.cu
    LABELS numopt)
