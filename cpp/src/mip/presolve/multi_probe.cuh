/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <mip/problem/problem.cuh>
#include <mip/solution/solution.cuh>
#include <mip/solver.cuh>
#include <mip/utils.cuh>

#include <utilities/timer.hpp>

#include "bounds_update_data.cuh"
#include "utils.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class multi_probe_t {
 public:
  struct settings_t {
    f_t time_limit{60.0};
    i_t iteration_limit{std::numeric_limits<i_t>::max()};
  };

  multi_probe_t(mip_solver_context_t<i_t, f_t>& context_, settings_t settings = settings_t{});
  void resize(problem_t<i_t, f_t>& problem);

  termination_criterion_t solve(
    problem_t<i_t, f_t>& pb,
    const std::tuple<std::vector<i_t>, std::vector<f_t>, std::vector<f_t>>& var_probe_vals,
    bool use_host_bounds = false);

  termination_criterion_t solve_for_interval(
    problem_t<i_t, f_t>& pb,
    const std::tuple<i_t, std::pair<f_t, f_t>, std::pair<f_t, f_t>>& var_interval_vals,
    const raft::handle_t* handle_ptr);

  void calculate_activity(problem_t<i_t, f_t>& pb, const raft::handle_t* handle_ptr);
  bool calculate_bounds_update(problem_t<i_t, f_t>& pb, const raft::handle_t* handle_ptr);
  void set_updated_bounds(problem_t<i_t, f_t>& pb,
                          i_t select_update,
                          const raft::handle_t* handle_ptr);
  void set_updated_bounds(const raft::handle_t* handle_ptr,
                          raft::device_span<f_t> output_lb,
                          raft::device_span<f_t> output_ub,
                          i_t select_update);
  termination_criterion_t bound_update_loop(problem_t<i_t, f_t>& pb,
                                            const raft::handle_t* handle_ptr,
                                            timer_t timer);
  void set_interval_bounds(
    const std::tuple<i_t, std::pair<f_t, f_t>, std::pair<f_t, f_t>>& var_interval_vals,
    problem_t<i_t, f_t>& pb,
    const raft::handle_t* handle_ptr);
  void set_bounds(
    const std::tuple<std::vector<i_t>, std::vector<f_t>, std::vector<f_t>>& var_probe_vals,
    const raft::handle_t* handle_ptr);
  void constraint_stats(problem_t<i_t, f_t>& pb, const raft::handle_t* handle_ptr);
  void copy_problem_into_probing_buffers(problem_t<i_t, f_t>& pb, const raft::handle_t* handle_ptr);
  void update_host_bounds(const raft::handle_t* handle_ptr,
                          const raft::device_span<f_t> variable_lb,
                          const raft::device_span<f_t> variable_ub);
  void update_device_bounds(const raft::handle_t* handle_ptr);
  mip_solver_context_t<i_t, f_t>& context;
  bounds_update_data_t<i_t, f_t> upd_0;
  bounds_update_data_t<i_t, f_t> upd_1;
  std::vector<f_t> host_lb;
  std::vector<f_t> host_ub;
  bool skip_0;
  bool skip_1;
  settings_t settings;
  bool compute_stats             = true;
  bool init_changed_constraints  = true;
  i_t infeas_constraints_count_0 = 0;
  i_t redund_constraints_count_0 = 0;
  i_t infeas_constraints_count_1 = 0;
  i_t redund_constraints_count_1 = 0;
};

}  // namespace cuopt::linear_programming::detail
