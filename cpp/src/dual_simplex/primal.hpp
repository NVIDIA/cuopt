/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/phase1.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/types.hpp>

#include <vector>

namespace cuopt::linear_programming::dual_simplex {

namespace primal {
enum class status_t {
  OPTIMAL          = 0,
  PRIMAL_UNBOUNDED = 1,
  NUMERICAL        = 2,
  NOT_LOADED       = 3,
  TIME_LIMIT       = 4,
  ITERATION_LIMIT  = 5,
  CONCURRENT_LIMIT = 6
};
}

template <typename i_t, typename f_t>
primal::status_t primal_phase2(i_t phase,
                               f_t start_time,
                               const lp_problem_t<i_t, f_t>& lp,
                               const simplex_solver_settings_t<i_t, f_t>& settings,
                               std::vector<variable_status_t>& vstatus,
                               lp_solution_t<i_t, f_t>& sol,
                               i_t& iter);

}  // namespace cuopt::linear_programming::dual_simplex
