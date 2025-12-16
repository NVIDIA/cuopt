/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <mip/presolve/bounds_presolve.cuh>
#include <mip/problem/problem.cuh>
#include <mip/solution/solution.cuh>
#include "lp_state.cuh"

namespace cuopt::linear_programming::detail {

struct relaxed_lp_settings_t {
  double tolerance                  = 1e-4;
  double time_limit                 = 1.0;
  bool check_infeasibility          = true;
  bool return_first_feasible        = false;
  bool save_state                   = true;
  bool per_constraint_residual      = true;
  bool has_initial_primal           = true;
  std::atomic<int>* concurrent_halt = nullptr;
};

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> get_relaxed_lp_solution(
  problem_t<i_t, f_t>& op_problem,
  solution_t<i_t, f_t>& solution,
  const relaxed_lp_settings_t& settings);

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> get_relaxed_lp_solution(
  problem_t<i_t, f_t>& op_problem,
  rmm::device_uvector<f_t>& assignment,
  lp_state_t<i_t, f_t>& lp_state,
  const relaxed_lp_settings_t& settings);

template <typename i_t, typename f_t>
bool run_lp_with_vars_fixed(problem_t<i_t, f_t>& op_problem,
                            solution_t<i_t, f_t>& solution,
                            const rmm::device_uvector<i_t>& variables_to_fix,
                            relaxed_lp_settings_t& settings,
                            bound_presolve_t<i_t, f_t>* bound_presolve = nullptr,
                            bool check_fixed_assignment_feasibility    = false,
                            bool use_integer_fixed_problem             = false);

}  // namespace cuopt::linear_programming::detail
