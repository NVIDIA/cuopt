/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <mip/presolve/gf2_presolve.cuh>

#include <mip/mip_constants.hpp>
#include <mip/presolve/bounds_update_helpers.cuh>

namespace cuopt::linear_programming::detail {

// Inspired from
// https://github.com/snowberryfield/printemps/blob/0dc1f9d2b78a667451c11d8ab156861ce54e9fb7/printemps/model_component/constraint.h
// (MIT License)

#define NOT_GF2(reason, ...)                                                  \
  {                                                                           \
    printf("NO : Cons %d is not gf2: " reason "\n", cstr_idx, ##__VA_ARGS__); \
    goto not_valid;                                                           \
  }

// TODO: implement fully on GPU + cuDSS?
template <typename i_t, typename f_t>
void gf2_presolve(problem_t<i_t, f_t>& problem)
{
  auto h_constraint_lower_bounds = cuopt::host_copy(problem.constraint_lower_bounds);
  auto h_constraint_upper_bounds = cuopt::host_copy(problem.constraint_upper_bounds);
  auto h_coefficients            = cuopt::host_copy(problem.coefficients);
  auto h_variables               = cuopt::host_copy(problem.variables);
  auto h_is_binary_variable      = cuopt::host_copy(problem.is_binary_variable);
  auto h_variable_bounds         = cuopt::host_copy(problem.variable_bounds);
  auto h_offsets                 = cuopt::host_copy(problem.offsets);
  auto h_variable_types          = cuopt::host_copy(problem.variable_types);

  for (i_t cstr_idx = 0; cstr_idx < problem.n_constraints; ++cstr_idx) {
    i_t key_var_idx              = -1;
    f_t key_var_coeff            = 0.0;
    f_t key_var_lb               = 0.0;
    f_t key_var_ub               = 0.0;
    f_t min_activity_without_key = 0.0;
    f_t max_activity_without_key = 0.0;

    //__asm__ __volatile__("int3");

    f_t rhs = round(h_constraint_lower_bounds[cstr_idx]);

    // needs to be an equality constraint
    if (h_constraint_lower_bounds[cstr_idx] != h_constraint_upper_bounds[cstr_idx])
      NOT_GF2("not an equality constraint");
    if (!isfinite(h_constraint_lower_bounds[cstr_idx])) NOT_GF2("not finite");
    if (!is_integer(h_constraint_lower_bounds[cstr_idx], problem.tolerances.integrality_tolerance))
      NOT_GF2("rhs not integer %f", h_constraint_lower_bounds[cstr_idx]);

    // only accept 0, 1, -1 as rhs
    if (rhs != 0.0 && rhs != 1.0 && rhs != -1.0) NOT_GF2("invalid rhs %f", rhs);

    // check if nnzs match the pattern of a gf2 constraint
    for (i_t j = h_offsets[cstr_idx]; j < h_offsets[cstr_idx + 1]; ++j) {
      if (!is_integer(h_coefficients[j], problem.tolerances.integrality_tolerance))
        NOT_GF2("coeff not integer (%d,%f)", j, h_coefficients[j]);
      i_t var_idx = h_variables[j];
      f_t coeff   = round(h_coefficients[j]);
      if (h_variable_types[var_idx] != var_t::INTEGER)
        NOT_GF2("var not integer (%d,%f)", var_idx, h_coefficients[j]);
      bool is_binary = h_is_binary_variable[var_idx];

      // only the key variable can have a coefficient of 2
      if (is_binary && (abs(coeff) != 1.0 && abs(coeff) != 2.0))
        NOT_GF2("invalid coef bin (%d,%f)", var_idx, coeff);
      if (!is_binary && (abs(coeff) != 2.0))
        NOT_GF2("invalid coef non-bin (%d,%f)", var_idx, coeff);

      // key var
      if (abs(coeff) == 2.0) {
        // can only be one
        if (key_var_idx != -1) NOT_GF2("multiple key variables");
        key_var_idx   = var_idx;
        key_var_coeff = coeff;
      } else {
        min_activity_without_key += min_act_of_var(
          coeff, get_lower(h_variable_bounds[var_idx]), get_upper(h_variable_bounds[var_idx]));
        max_activity_without_key += max_act_of_var(
          coeff, get_lower(h_variable_bounds[var_idx]), get_upper(h_variable_bounds[var_idx]));
      }
    }

    // no key var found
    if (key_var_idx == -1) NOT_GF2("no key variable");

    if (key_var_coeff > 0) {
      std::swap(min_activity_without_key, max_activity_without_key);
      min_activity_without_key *= -1;
      max_activity_without_key *= -1;
    }
    cuopt_assert(min_activity_without_key <= max_activity_without_key, "invalid activities?");

    key_var_lb = get_lower(h_variable_bounds[key_var_idx]);
    key_var_ub = get_upper(h_variable_bounds[key_var_idx]);
    if (isfinite(key_var_lb) && key_var_lb > ceil(min_activity_without_key / 2.0))
      NOT_GF2("invalid key variable bounds lower ((%f,%f), %f)",
              key_var_lb,
              key_var_ub,
              min_activity_without_key);
    if (isfinite(key_var_ub) && key_var_ub < floor(max_activity_without_key / 2.0))
      NOT_GF2("invalid key variable bounds upper ((%f,%f), %f)",
              key_var_lb,
              key_var_ub,
              max_activity_without_key);

    printf("YES: Cons %d is gf2\n", cstr_idx);
    continue;

  not_valid:
    continue;
  }

  exit(0);
}

#define INSTANTIATE(F_TYPE) \
  template void gf2_presolve<int, F_TYPE>(problem_t<int, F_TYPE> & problem);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
