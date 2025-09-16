/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include <utilities/copy_helpers.hpp>
#include <utilities/macros.cuh>

#include <cuopt/linear_programming/optimization_problem.hpp>

#include <vector>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct constraints_delta_t {
  std::vector<i_t> constraint_variables;
  std::vector<f_t> constraint_coefficients;
  std::vector<i_t> constraint_offsets = std::vector<i_t>(1, 0);
  std::vector<f_t> constraint_upper_bounds;
  std::vector<f_t> constraint_lower_bounds;

  i_t n_constraints() const { return constraint_lower_bounds.size(); }
  i_t matrix_size() const { return constraint_variables.size(); }
  void add_constraint(std::vector<i_t> constr_indices,
                      std::vector<f_t> constr_coeffs,
                      f_t lower_bound,
                      f_t upper_bound)
  {
    constraint_variables.insert(
      constraint_variables.end(), constr_indices.begin(), constr_indices.end());
    constraint_coefficients.insert(
      constraint_coefficients.end(), constr_coeffs.begin(), constr_coeffs.end());
    constraint_lower_bounds.push_back(lower_bound);
    constraint_upper_bounds.push_back(upper_bound);
    constraint_offsets.push_back(constraint_offsets.back() + constr_indices.size());
  }
};

template <typename i_t, typename f_t>
struct variables_delta_t {
  using f_t2 = typename type_2<f_t>::type;
  std::vector<f_t> objective_coefficients;
  std::vector<f_t2> variable_bounds;
  std::vector<var_t> variable_types;
  std::vector<i_t> is_binary_variable;

  i_t n_vars;

  i_t size() const { return variable_bounds.size(); }

  // returns the added variable id
  i_t add_variable(f_t lower_bound, f_t upper_bound, f_t obj_weight, var_t var_type)
  {
    cuopt_assert(lower_bound >= 0, "Variable bounds must be non-negative!");
    variable_bounds.push_back(f_t2{lower_bound, upper_bound});
    objective_coefficients.push_back(obj_weight);
    variable_types.push_back(var_type);
    is_binary_variable.push_back(0);
    return n_vars++;
  }
};

}  // namespace cuopt::linear_programming::detail
