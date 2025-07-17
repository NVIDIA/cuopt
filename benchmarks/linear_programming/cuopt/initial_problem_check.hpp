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

template <typename f_t>
double combine_finite_abs_bounds(f_t lower, f_t upper)
{
  f_t val = f_t(0);
  if (isfinite(upper)) { val = raft::max<f_t>(val, raft::abs(upper)); }
  if (isfinite(lower)) { val = raft::max<f_t>(val, raft::abs(lower)); }
  return val;
}

template <typename f_t>
struct violation {
  violation() {}
  violation(f_t* _scalar) {}
  __device__ __host__ f_t operator()(f_t value, f_t lower, f_t upper)
  {
    if (value < lower) {
      return lower - value;
    } else if (value > upper) {
      return value - upper;
    }
    return f_t(0);
  }
};

bool test_constraint_and_variable_sanity(
  const cuopt::mps_parser::mps_data_model_t<int, double>& op_problem,
  const std::vector<double>& primal_vars,
  double abs_tol,
  double rel_tol,
  double int_tol = 1e-5)
{
  const std::vector<double>& values                  = op_problem.get_constraint_matrix_values();
  const std::vector<int>& indices                    = op_problem.get_constraint_matrix_indices();
  const std::vector<int>& offsets                    = op_problem.get_constraint_matrix_offsets();
  const std::vector<double>& constraint_lower_bounds = op_problem.get_constraint_lower_bounds();
  const std::vector<double>& constraint_upper_bounds = op_problem.get_constraint_upper_bounds();
  const std::vector<double>& variable_lower_bounds   = op_problem.get_variable_lower_bounds();
  const std::vector<double>& variable_upper_bounds   = op_problem.get_variable_upper_bounds();
  const std::vector<char>& variable_types            = op_problem.get_variable_types();
  std::vector<double> residual(constraint_lower_bounds.size(), 0.0);
  std::vector<double> viol(constraint_lower_bounds.size(), 0.0);

  // CSR SpMV
  for (size_t i = 0; i < offsets.size() - 1; ++i) {
    for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
      residual[i] += values[j] * primal_vars[indices[j]];
    }
  }

  auto functor = violation<double>{};

  bool feasible = true;
  // Compute violation to lower/upper bound
  for (size_t i = 0; i < residual.size(); ++i) {
    double tolerance = abs_tol + combine_finite_abs_bounds<double>(constraint_lower_bounds[i],
                                                                   constraint_upper_bounds[i]) *
                                   rel_tol;
    double viol = functor(residual[i], constraint_lower_bounds[i], constraint_upper_bounds[i]);
    if (viol > tolerance) {
      feasible = false;
      CUOPT_LOG_ERROR(
        "feasibility violation %f at cstr %d is more than total tolerance %f lb %f ub %f \n",
        viol,
        i,
        tolerance,
        constraint_lower_bounds[i],
        constraint_upper_bounds[i]);
    }
  }
  bool feasible_variables = true;
  for (size_t i = 0; i < primal_vars.size(); ++i) {
    if (variable_types[i] == 'I' && abs(primal_vars[i] - round(primal_vars[i])) > int_tol) {
      feasible_variables = false;
    }
    // Not always stricly true because we apply variable bound clamping on the scaled problem
    // After unscaling it, the variables might not respect exactly (this adding an epsilon)
    if (!(primal_vars[i] >= variable_lower_bounds[i] - int_tol &&
          primal_vars[i] <= variable_upper_bounds[i] + int_tol)) {
      CUOPT_LOG_ERROR("error at bounds var %d lb %f ub %f val %f\n",
                      i,
                      variable_lower_bounds[i],
                      variable_upper_bounds[i],
                      primal_vars[i]);
      feasible_variables = false;
    }
  }
  if (!feasible || !feasible_variables) { CUOPT_LOG_ERROR("Initial solution is infeasible"); }
  return feasible_variables;
}