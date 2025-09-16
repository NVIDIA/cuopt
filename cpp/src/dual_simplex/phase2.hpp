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
#include <dual_simplex/logger.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/types.hpp>

#include <vector>

namespace cuopt::linear_programming::dual_simplex {

namespace dual {
enum class status_t {
  OPTIMAL          = 0,
  DUAL_UNBOUNDED   = 1,
  NUMERICAL        = 2,
  CUTOFF           = 3,
  TIME_LIMIT       = 4,
  ITERATION_LIMIT  = 5,
  CONCURRENT_LIMIT = 6,
  UNSET            = 7
};

static std::string status_to_string(status_t status)
{
  switch (status) {
    case status_t::OPTIMAL: return "OPTIMAL";
    case status_t::DUAL_UNBOUNDED: return "DUAL_UNBOUNDED";
    case status_t::NUMERICAL: return "NUMERICAL";
    case status_t::CUTOFF: return "CUTOFF";
    case status_t::TIME_LIMIT: return "TIME_LIMIT";
    case status_t::ITERATION_LIMIT: return "ITERATION_LIMIT";
    case status_t::CONCURRENT_LIMIT: return "CONCURRENT_LIMIT";
    case status_t::UNSET: return "UNSET";
  }
  return "UNKNOWN";
}
}  // namespace dual

template <typename i_t, typename f_t>
dual::status_t dual_phase2(i_t phase,
                           i_t slack_basis,
                           f_t start_time,
                           const lp_problem_t<i_t, f_t>& lp,
                           const simplex_solver_settings_t<i_t, f_t>& settings,
                           std::vector<variable_status_t>& vstatus,
                           lp_solution_t<i_t, f_t>& sol,
                           i_t& iter,
                           std::vector<f_t>& steepest_edge_norms);

}  // namespace cuopt::linear_programming::dual_simplex
