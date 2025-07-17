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

#include <cuopt/linear_programming/mip/solver_stats.hpp>

#include <linear_programming/initial_scaling_strategy/initial_scaling.cuh>
#include <mip/problem/problem.cuh>
#include <mip/relaxed_lp/lp_state.cuh>

#pragma once

namespace cuopt::linear_programming::detail {

// Aggregate structure containing the global context of the solving process for convenience:
// The current problem, user settings, raft handle and statistics objects
template <typename i_t, typename f_t>
struct mip_solver_context_t {
  explicit mip_solver_context_t(raft::handle_t const* handle_ptr_,
                                problem_t<i_t, f_t>* problem_ptr_,
                                mip_solver_settings_t<i_t, f_t> settings_,
                                pdlp_initial_scaling_strategy_t<i_t, f_t>& scaling)
    : handle_ptr(handle_ptr_), problem_ptr(problem_ptr_), settings(settings_), scaling(scaling)
  {
    cuopt_assert(problem_ptr != nullptr, "problem_ptr is nullptr");
    stats.solution_bound = problem_ptr->maximize ? std::numeric_limits<f_t>::infinity()
                                                 : -std::numeric_limits<f_t>::infinity();
  }

  raft::handle_t const* const handle_ptr;
  problem_t<i_t, f_t>* problem_ptr;
  const mip_solver_settings_t<i_t, f_t> settings;
  pdlp_initial_scaling_strategy_t<i_t, f_t>& scaling;
  solver_stats_t<i_t, f_t> stats;
};

}  // namespace cuopt::linear_programming::detail
