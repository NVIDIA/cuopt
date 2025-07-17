# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
ConfigureTest(UNIT_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/integer_with_real_bounds.cu
)
ConfigureTest(EMPTY_FIXED_PROBLEMS_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/empty_fixed_problems_test.cu
)
ConfigureTest(FEASIBILITY_JUMP_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/feasibility_jump_tests.cu
)
ConfigureTest(MIP_TERMINATION_STATUS_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/termination_test.cu
)
