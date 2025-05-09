/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuopt/linear_programming/cuopt_c.h>

#ifdef __cplusplus
extern "C" {
#endif

int test_int_size();
int test_float_size();
void test_version(int* major, int* minor, int* patch);
cuopt_int_t burglar_problem();
cuopt_int_t solve_mps_file(const char* filename,
                           double time_limit,
                           double iteration_limit,
                           cuopt_int_t* termination_status);
cuopt_int_t test_missing_file();
cuopt_int_t test_infeasible_problem();
cuopt_int_t test_bad_parameter_name();
cuopt_int_t test_ranged_problem(cuopt_int_t* termination_status_ptr, cuopt_float_t* objective_ptr);

#ifdef __cplusplus
}
#endif
