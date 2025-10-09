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

#pragma once

#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/types.hpp>
#include <dual_simplex/user_problem.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
struct lp_problem_t {
  lp_problem_t(raft::handle_t const* handle_ptr_, i_t m, i_t n, i_t nz)
    : handle_ptr(handle_ptr_),
      num_rows(m),
      num_cols(n),
      objective(n),
      A(m, n, nz),
      rhs(m),
      lower(n),
      upper(n),
      obj_constant(0.0)
  {
  }
  raft::handle_t const* handle_ptr;
  i_t num_rows;
  i_t num_cols;
  std::vector<f_t> objective;
  csc_matrix_t<i_t, f_t> A;
  std::vector<f_t> rhs;
  std::vector<f_t> lower;
  std::vector<f_t> upper;
  f_t obj_constant;
  f_t obj_scale;  // 1.0 for min, -1.0 for max
};

template <typename i_t, typename f_t>
struct folding_info_t {
  folding_info_t()
    : D(0, 0, 0),
      C_s(0, 0, 0),
      D_s(0, 0, 0),
      c_tilde(0),
      A_tilde(0, 0, 0),
      num_upper_bounds(0),
      is_folded(false)
  {
  }
  csc_matrix_t<i_t, f_t> D;
  csc_matrix_t<i_t, f_t> C_s;
  csc_matrix_t<i_t, f_t> D_s;
  std::vector<f_t> c_tilde;
  csc_matrix_t<i_t, f_t> A_tilde;
  i_t num_upper_bounds;
  bool is_folded;
};

template <typename i_t, typename f_t>
struct presolve_info_t {
  // indices of variables in the original problem that remain in the presolved problem
  std::vector<i_t> remaining_variables;
  // indicies of variables in the original problem that have been removed in the presolved problem
  std::vector<i_t> removed_variables;
  // values of the removed variables
  std::vector<f_t> removed_values;
  // values of the removed reduced costs
  std::vector<f_t> removed_reduced_costs;
  // Free variable pairs
  std::vector<i_t> free_variable_pairs;
  // Removed lower bounds
  std::vector<f_t> removed_lower_bounds;
  // indices of the constraints in the original problem that remain in the presolved problem
  std::vector<i_t> remaining_constraints;
  // indices of the constraints in the original problem that have been removed in the presolved
  // problem
  std::vector<i_t> removed_constraints;

  folding_info_t<i_t, f_t> folding_info;
};

template <typename i_t, typename f_t>
struct dualize_info_t {
  dualize_info_t()
    : solving_dual(false),
      primal_problem(nullptr, 0, 0, 0),
      zl_start(0),
      zu_start(0),
      equality_rows({}),
      vars_with_upper_bounds({})
  {
  }
  bool solving_dual;
  lp_problem_t<i_t, f_t> primal_problem;
  i_t zl_start;
  i_t zu_start;
  std::vector<i_t> equality_rows;
  std::vector<i_t> vars_with_upper_bounds;
};

template <typename i_t, typename f_t>
void convert_user_problem(const user_problem_t<i_t, f_t>& user_problem,
                          const simplex_solver_settings_t<i_t, f_t>& settings,
                          lp_problem_t<i_t, f_t>& problem,
                          std::vector<i_t>& new_slacks,
                          dualize_info_t<i_t, f_t>& dualize_info);

template <typename i_t, typename f_t>
void convert_user_problem_with_guess(const user_problem_t<i_t, f_t>& user_problem,
                                     const std::vector<f_t>& guess,
                                     lp_problem_t<i_t, f_t>& problem,
                                     std::vector<f_t>& converted_guess);

template <typename i_t, typename f_t>
void convert_user_lp_with_guess(const user_problem_t<i_t, f_t>& user_problem,
                                const lp_solution_t<i_t, f_t>& initial_solution,
                                const std::vector<f_t>& initial_slack,
                                lp_problem_t<i_t, f_t>& lp,
                                lp_solution_t<i_t, f_t>& converted_solution);

template <typename i_t, typename f_t>
i_t presolve(const lp_problem_t<i_t, f_t>& original,
             const simplex_solver_settings_t<i_t, f_t>& settings,
             lp_problem_t<i_t, f_t>& presolved,
             presolve_info_t<i_t, f_t>& presolve_info);

template <typename i_t, typename f_t>
void crush_primal_solution(const user_problem_t<i_t, f_t>& user_problem,
                           const lp_problem_t<i_t, f_t>& problem,
                           const std::vector<f_t>& user_solution,
                           const std::vector<i_t>& new_slacks,
                           std::vector<f_t>& solution);

template <typename i_t, typename f_t>
void crush_primal_solution_with_slack(const user_problem_t<i_t, f_t>& user_problem,
                                      const lp_problem_t<i_t, f_t>& problem,
                                      const std::vector<f_t>& user_solution,
                                      const std::vector<f_t>& user_slack,
                                      const std::vector<i_t>& new_slacks,
                                      std::vector<f_t>& solution);

template <typename i_t, typename f_t>
void crush_dual_solution(const user_problem_t<i_t, f_t>& user_problem,
                         const lp_problem_t<i_t, f_t>& problem,
                         const std::vector<i_t>& new_slacks,
                         const std::vector<f_t>& user_y,
                         const std::vector<f_t>& user_z,
                         std::vector<f_t>& y,
                         std::vector<f_t>& z);

template <typename i_t, typename f_t>
void uncrush_primal_solution(const user_problem_t<i_t, f_t>& user_problem,
                             const lp_problem_t<i_t, f_t>& problem,
                             const std::vector<f_t>& solution,
                             std::vector<f_t>& user_solution);

template <typename i_t, typename f_t>
void uncrush_dual_solution(const user_problem_t<i_t, f_t>& user_problem,
                           const lp_problem_t<i_t, f_t>& problem,
                           const std::vector<f_t>& y,
                           const std::vector<f_t>& z,
                           std::vector<f_t>& user_y,
                           std::vector<f_t>& user_z);

template <typename i_t, typename f_t>
void uncrush_solution(const presolve_info_t<i_t, f_t>& presolve_info,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      const std::vector<f_t>& crushed_x,
                      const std::vector<f_t>& crushed_y,
                      const std::vector<f_t>& crushed_z,
                      std::vector<f_t>& uncrushed_x,
                      std::vector<f_t>& uncrushed_y,
                      std::vector<f_t>& uncrushed_z);

// For pure LP bounds strengthening, var_types should be defaulted (i.e. left empty)
template <typename i_t, typename f_t>
bool bound_strengthening(const std::vector<char>& row_sense,
                         const simplex_solver_settings_t<i_t, f_t>& settings,
                         lp_problem_t<i_t, f_t>& problem,
                         const csc_matrix_t<i_t, f_t>& Arow,
                         const std::vector<variable_type_t>& var_types = {},
                         const std::vector<bool>& bounds_changed       = {});

}  // namespace cuopt::linear_programming::dual_simplex
