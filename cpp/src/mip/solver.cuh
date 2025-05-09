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

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <mip/problem/problem.cuh>
#include <mip/solver_context.cuh>
#include <mip/solver_stats.cuh>
#include <utilities/timer.hpp>
#pragma once

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class mip_solver_t {
 public:
  explicit mip_solver_t(const problem_t<i_t, f_t>& op_problem,
                        const mip_solver_settings_t<i_t, f_t>& solver_settings,
                        pdlp_initial_scaling_strategy_t<i_t, f_t>& scaling,
                        timer_t timer);

  solution_t<i_t, f_t> run_solver();
  solver_stats_t<i_t, f_t>& get_solver_stats() { return context.stats; }

  mip_solver_context_t<i_t, f_t> context;
  // reference to the original problem
  const problem_t<i_t, f_t>& op_problem_;
  const mip_solver_settings_t<i_t, f_t>& solver_settings_;
  timer_t timer_;
};

}  // namespace cuopt::linear_programming::detail
