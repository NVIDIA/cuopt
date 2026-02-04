/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/cuopt_c.h>

#ifdef __cplusplus
extern "C" {
#endif

int test_int_size();
int test_float_size();
cuopt_int_t burglar_problem();
cuopt_int_t solve_mps_file(const char* filename,
                           double time_limit,
                           double iteration_limit,
                           cuopt_int_t* termination_status,
#ifdef __cplusplus
                           cuopt_float_t* solve_time = 0,
                           cuopt_int_t method        = CUOPT_METHOD_DUAL_SIMPLEX);
#else
                           cuopt_float_t* solve_time,
                           cuopt_int_t method);
#endif
cuopt_int_t test_missing_file();
cuopt_int_t test_infeasible_problem();
cuopt_int_t test_bad_parameter_name();
cuopt_int_t test_mip_get_callbacks_only();
cuopt_int_t test_mip_get_set_callbacks();
cuopt_int_t test_ranged_problem(cuopt_int_t* termination_status_ptr, cuopt_float_t* objective_ptr);
cuopt_int_t test_invalid_bounds(cuopt_int_t test_mip);
cuopt_int_t test_quadratic_problem(cuopt_int_t* termination_status_ptr,
                                   cuopt_float_t* objective_ptr);
cuopt_int_t test_quadratic_ranged_problem(cuopt_int_t* termination_status_ptr,
                                          cuopt_float_t* objective_ptr);
cuopt_int_t test_write_problem(const char* input_filename, const char* output_filename);
#ifdef __cplusplus
}
#endif
