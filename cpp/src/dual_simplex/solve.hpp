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

#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/types.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
bool is_mip(const user_problem_t<i_t, f_t>& problem);

enum class lp_status_t {
  OPTIMAL          = 0,
  INFEASIBLE       = 1,
  UNBOUNDED        = 2,
  ITERATION_LIMIT  = 3,
  TIME_LIMIT       = 4,
  NUMERICAL_ISSUES = 5,
  CUTOFF           = 6,
  CONCURRENT_LIMIT = 7,
  UNSET            = 8
};

template <typename i_t, typename f_t>
f_t compute_objective(const lp_problem_t<i_t, f_t>& problem, const std::vector<f_t>& x);

template <typename i_t, typename f_t>
f_t compute_user_objective(const lp_problem_t<i_t, f_t>& lp, const std::vector<f_t>& x);

template <typename i_t, typename f_t>
f_t compute_user_objective(const lp_problem_t<i_t, f_t>& lp, f_t obj);

template <typename i_t, typename f_t>
lp_status_t solve_linear_program_advanced(const lp_problem_t<i_t, f_t>& original_lp,
                                          const f_t start_time,
                                          const simplex_solver_settings_t<i_t, f_t>& settings,
                                          lp_solution_t<i_t, f_t>& original_solution,
                                          std::vector<variable_status_t>& vstatus,
                                          std::vector<f_t>& edge_norms);

template <typename i_t, typename f_t>
lp_status_t solve_linear_program(const user_problem_t<i_t, f_t>& user_problem,
                                 const simplex_solver_settings_t<i_t, f_t>& settings,
                                 lp_solution_t<i_t, f_t>& solution);

template <typename i_t, typename f_t>
i_t solve_mip(const user_problem_t<i_t, f_t>& user_problem, mip_solution_t<i_t, f_t>& solution);

template <typename i_t, typename f_t>
i_t solve_mip_with_guess(const user_problem_t<i_t, f_t>& problem,
                         const simplex_solver_settings_t<i_t, f_t>& settings,
                         const std::vector<f_t>& guess,
                         mip_solution_t<i_t, f_t>& solution);

template <typename i_t, typename f_t>
i_t solve(const user_problem_t<i_t, f_t>& user_problem,
          const simplex_solver_settings_t<i_t, f_t>& settings,
          std::vector<f_t>& primal_solution);

}  // namespace cuopt::linear_programming::dual_simplex
