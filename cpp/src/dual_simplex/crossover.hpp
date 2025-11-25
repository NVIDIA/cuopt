/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/types.hpp>
#include <dual_simplex/user_problem.hpp>

namespace cuopt::linear_programming::dual_simplex {

enum class crossover_status_t : int8_t {
  OPTIMAL          = 0,
  PRIMAL_FEASIBLE  = 1,
  DUAL_FEASIBLE    = 2,
  TIME_LIMIT       = 3,
  NUMERICAL_ISSUES = 4,
  CONCURRENT_LIMIT = 5,
};

template <typename i_t, typename f_t>
crossover_status_t crossover(const lp_problem_t<i_t, f_t>& problem,
                             const simplex_solver_settings_t<i_t, f_t>& settings,
                             const lp_solution_t<i_t, f_t>& initial_solution,
                             f_t start_time,
                             lp_solution_t<i_t, f_t>& solution,
                             std::vector<variable_status_t>& vstatus);

}  // namespace cuopt::linear_programming::dual_simplex
