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
  lp_problem_t(i_t m, i_t n, i_t nz)
    : num_rows(m),
      num_cols(n),
      objective(n),
      A(m, n, nz),
      rhs(m),
      lower(n),
      upper(n),
      obj_constant(0.0)
  {
  }
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
struct presolve_info_t {
  // indices of variables in the original problem that remain in the presolved problem
  std::vector<i_t> remaining_variables;
  // indicies of variables in the original problem that have been removed in the presolved problem
  std::vector<i_t> removed_variables;
  // values of the removed variables
  std::vector<f_t> removed_values;
  // values of the removed reduced costs
  std::vector<f_t> removed_reduced_costs;
};

template <typename i_t, typename f_t>
void convert_user_problem(const user_problem_t<i_t, f_t>& user_problem,
                          lp_problem_t<i_t, f_t>& problem,
                          std::vector<i_t>& new_slacks);

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
                      const std::vector<f_t>& crushed_x,
                      const std::vector<f_t>& crushed_z,
                      std::vector<f_t>& uncrushed_x,
                      std::vector<f_t>& uncrushed_z);

}  // namespace cuopt::linear_programming::dual_simplex
