/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <mip/presolve/bounds_presolve.cuh>
#include <mip/problem/problem.cuh>
#include <mip/solution/solution.cuh>
#include "lp_state.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> get_relaxed_lp_solution(
  problem_t<i_t, f_t>& op_problem,
  solution_t<i_t, f_t>& solution,
  f_t tolerance,
  f_t time_limit             = 20.,
  bool check_infeasibility   = true,
  bool return_first_feasible = false);

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> get_relaxed_lp_solution(
  problem_t<i_t, f_t>& op_problem,
  rmm::device_uvector<f_t>& assignment,
  lp_state_t<i_t, f_t>& lp_state,
  f_t tolerance,
  f_t time_limit             = 20.,
  bool check_infeasibility   = true,
  bool return_first_feasible = false,
  bool save_state            = true);

template <typename i_t, typename f_t>
bool run_lp_with_vars_fixed(problem_t<i_t, f_t>& op_problem,
                            solution_t<i_t, f_t>& solution,
                            const rmm::device_uvector<i_t>& variables_to_fix,
                            typename mip_solver_settings_t<i_t, f_t>::tolerances_t tols,
                            lp_state_t<i_t, f_t>& lp_state,
                            f_t time_limit                             = 20.,
                            bool return_first_feasible                 = false,
                            bound_presolve_t<i_t, f_t>* bound_presolve = nullptr);

}  // namespace cuopt::linear_programming::detail
