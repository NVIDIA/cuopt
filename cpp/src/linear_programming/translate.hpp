/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include <cuopt/linear_programming/optimization_problem.hpp>

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/sparse_matrix.hpp>

#include <utilities/copy_helpers.hpp>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
static dual_simplex::user_problem_t<i_t, f_t> cuopt_problem_to_simplex_problem(
  detail::problem_t<i_t, f_t>& model)
{
  dual_simplex::user_problem_t<i_t, f_t> user_problem(model.handle_ptr);

  int m                  = model.n_constraints;
  int n                  = model.n_variables;
  int nz                 = model.nnz;
  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = cuopt::host_copy(model.objective_coefficients);

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.x         = cuopt::host_copy(model.coefficients);
  csr_A.j         = cuopt::host_copy(model.variables);
  csr_A.row_start = cuopt::host_copy(model.offsets);

  csr_A.to_compressed_col(user_problem.A);

  user_problem.rhs.resize(m);
  user_problem.row_sense.resize(m);
  user_problem.range_rows.clear();
  user_problem.range_value.clear();

  auto model_constraint_lower_bounds = cuopt::host_copy(model.constraint_lower_bounds);
  auto model_constraint_upper_bounds = cuopt::host_copy(model.constraint_upper_bounds);

  // All constraints have lower and upper bounds
  // lr <= a_i^T x <= ur
  for (int i = 0; i < m; ++i) {
    const double constraint_lower_bound = model_constraint_lower_bounds[i];
    const double constraint_upper_bound = model_constraint_upper_bounds[i];
    if (constraint_lower_bound == constraint_upper_bound) {
      user_problem.row_sense[i] = 'E';
      user_problem.rhs[i]       = constraint_lower_bound;
    } else if (constraint_upper_bound == std::numeric_limits<double>::infinity()) {
      user_problem.row_sense[i] = 'G';
      user_problem.rhs[i]       = constraint_lower_bound;
    } else if (constraint_lower_bound == -std::numeric_limits<double>::infinity()) {
      user_problem.row_sense[i] = 'L';
      user_problem.rhs[i]       = constraint_upper_bound;
    } else {
      // This is range row
      user_problem.row_sense[i] = 'E';
      user_problem.rhs[i]       = constraint_lower_bound;
      user_problem.range_rows.push_back(i);
      const double bound_difference = constraint_upper_bound - constraint_lower_bound;
      user_problem.range_value.push_back(bound_difference);
    }
  }
  user_problem.num_range_rows = user_problem.range_rows.size();
  std::tie(user_problem.lower, user_problem.upper) =
    extract_host_bounds<f_t>(model.variable_bounds, model.handle_ptr);
  user_problem.problem_name = model.original_problem_ptr->get_problem_name();
  if (model.row_names.size() > 0) {
    user_problem.row_names.resize(m);
    for (int i = 0; i < m; ++i) {
      user_problem.row_names[i] = model.row_names[i];
    }
  }
  if (model.var_names.size() > 0) {
    user_problem.col_names.resize(n);
    for (int j = 0; j < n; ++j) {
      user_problem.col_names[j] = model.var_names[j];
    }
  }
  user_problem.obj_constant = model.presolve_data.objective_offset;
  user_problem.obj_scale    = model.presolve_data.objective_scaling_factor;
  user_problem.var_types.resize(n);

  auto model_variable_types = cuopt::host_copy(model.variable_types);
  for (int j = 0; j < n; ++j) {
    user_problem.var_types[j] =
      model_variable_types[j] == var_t::CONTINUOUS
        ? cuopt::linear_programming::dual_simplex::variable_type_t::CONTINUOUS
        : cuopt::linear_programming::dual_simplex::variable_type_t::INTEGER;
  }

  return user_problem;
}

template <typename i_t, typename f_t>
void translate_to_crossover_problem(const detail::problem_t<i_t, f_t>& problem,
                                    optimization_problem_solution_t<i_t, f_t>& sol,
                                    dual_simplex::lp_problem_t<i_t, f_t>& lp,
                                    dual_simplex::lp_solution_t<i_t, f_t>& initial_solution)
{
  CUOPT_LOG_DEBUG("Starting translation");

  std::vector<f_t> pdlp_objective = cuopt::host_copy(problem.objective_coefficients);

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(
    problem.n_constraints, problem.n_variables, problem.nnz);
  csr_A.x         = cuopt::host_copy(problem.coefficients);
  csr_A.j         = cuopt::host_copy(problem.variables);
  csr_A.row_start = cuopt::host_copy(problem.offsets);

  problem.handle_ptr->get_stream().synchronize();
  CUOPT_LOG_DEBUG("Converting to compressed column");
  csr_A.to_compressed_col(lp.A);
  CUOPT_LOG_DEBUG("Converted to compressed column");

  std::vector<f_t> slack(problem.n_constraints);
  std::vector<f_t> tmp_x = cuopt::host_copy(sol.get_primal_solution());
  problem.handle_ptr->get_stream().synchronize();
  dual_simplex::matrix_vector_multiply(lp.A, 1.0, tmp_x, 0.0, slack);
  CUOPT_LOG_DEBUG("Multiplied A and x");

  lp.A.col_start.resize(problem.n_variables + problem.n_constraints + 1);
  lp.A.x.resize(problem.nnz + problem.n_constraints);
  lp.A.i.resize(problem.nnz + problem.n_constraints);
  i_t nz = problem.nnz;
  for (i_t j = problem.n_variables; j < problem.n_variables + problem.n_constraints; ++j) {
    lp.A.col_start[j] = nz;
    lp.A.i[nz]        = j - problem.n_variables;
    lp.A.x[nz]        = -1.0;
    ++nz;
  }
  lp.A.col_start[problem.n_variables + problem.n_constraints] = nz;
  CUOPT_LOG_DEBUG("Finished with A");

  const i_t n = problem.n_variables + problem.n_constraints;
  const i_t m = problem.n_constraints;
  lp.num_cols = n;
  lp.num_rows = m;
  lp.A.n      = n;
  lp.rhs.resize(m, 0.0);
  lp.lower.resize(n);
  lp.upper.resize(n);
  lp.obj_constant = problem.presolve_data.objective_offset;
  lp.obj_scale    = problem.presolve_data.objective_scaling_factor;

  auto [lower, upper] = extract_host_bounds<f_t>(problem.variable_bounds, problem.handle_ptr);

  std::vector<f_t> constraint_lower = cuopt::host_copy(problem.constraint_lower_bounds);
  std::vector<f_t> constraint_upper = cuopt::host_copy(problem.constraint_upper_bounds);

  lp.objective.resize(n, 0.0);
  std::copy(
    pdlp_objective.begin(), pdlp_objective.begin() + problem.n_variables, lp.objective.begin());
  std::copy(lower.begin(), lower.begin() + problem.n_variables, lp.lower.begin());
  std::copy(upper.begin(), upper.begin() + problem.n_variables, lp.upper.begin());

  problem.handle_ptr->get_stream().synchronize();
  for (i_t i = 0; i < m; ++i) {
    lp.lower[problem.n_variables + i] = constraint_lower[i];
    lp.upper[problem.n_variables + i] = constraint_upper[i];
  }
  CUOPT_LOG_DEBUG("Finished with lp");

  initial_solution.resize(m, n);

  std::copy(tmp_x.begin(), tmp_x.begin() + problem.n_variables, initial_solution.x.begin());
  for (i_t j = problem.n_variables; j < n; ++j) {
    initial_solution.x[j] = slack[j - problem.n_variables];
    // Project slack variables inside their bounds
    if (initial_solution.x[j] < lp.lower[j]) { initial_solution.x[j] = lp.lower[j]; }
    if (initial_solution.x[j] > lp.upper[j]) { initial_solution.x[j] = lp.upper[j]; }
  }
  CUOPT_LOG_DEBUG("Finished with x");
  initial_solution.y = cuopt::host_copy(sol.get_dual_solution());

  std::vector<f_t> tmp_z = cuopt::host_copy(sol.get_reduced_cost());
  problem.handle_ptr->get_stream().synchronize();
  std::copy(tmp_z.begin(), tmp_z.begin() + problem.n_variables, initial_solution.z.begin());
  for (i_t j = problem.n_variables; j < n; ++j) {
    initial_solution.z[j] = initial_solution.y[j - problem.n_variables];
  }
  CUOPT_LOG_DEBUG("Finished with z");

  CUOPT_LOG_DEBUG("Finished translating");
}

}  // namespace cuopt::linear_programming
