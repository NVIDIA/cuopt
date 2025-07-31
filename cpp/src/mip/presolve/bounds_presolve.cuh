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

#include "probing_cache.cuh"

#include <mip/problem/problem.cuh>
#include <mip/solution/solution.cuh>
#include <mip/solver.cuh>
#include <mip/utils.cuh>

#include <utilities/timer.hpp>

#include "bounds_update_data.cuh"
#include "utils.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class probing_cache_t;

template <typename i_t, typename f_t>
class bound_presolve_t {
 public:
  struct settings_t {
    f_t time_limit{60.0};
    i_t iteration_limit{std::numeric_limits<i_t>::max()};
    bool parallel_bounds_update{true};
  };

  bound_presolve_t(mip_solver_context_t<i_t, f_t>& context, settings_t settings = settings_t{});
  void resize(problem_t<i_t, f_t>& problem);

  // This is a single bounds accepting solve
  // when we need to accept a vector, we can use input_lb version
  termination_criterion_t solve(problem_t<i_t, f_t>& pb, f_t lb, f_t ub, i_t var_idx);

  termination_criterion_t solve(problem_t<i_t, f_t>& pb,
                                raft::device_span<f_t> input_lb = {},
                                raft::device_span<f_t> input_ub = {});

  termination_criterion_t solve(problem_t<i_t, f_t>& pb,
                                const std::vector<thrust::pair<i_t, f_t>>& var_probe_val_pairs,
                                bool use_host_bounds = false);

  void calculate_activity(problem_t<i_t, f_t>& pb);
  void calculate_activity_on_problem_bounds(problem_t<i_t, f_t>& pb);
  bool calculate_bounds_update(problem_t<i_t, f_t>& pb);
  void set_updated_bounds(problem_t<i_t, f_t>& pb);
  void set_updated_bounds(const raft::handle_t* handle_ptr,
                          raft::device_span<f_t> output_lb,
                          raft::device_span<f_t> output_ub);
  termination_criterion_t bound_update_loop(problem_t<i_t, f_t>& pb, timer_t timer);
  void set_bounds(raft::device_span<f_t> var_lb,
                  raft::device_span<f_t> var_ub,
                  const std::vector<thrust::pair<i_t, f_t>>& var_probe_vals,
                  const raft::handle_t* handle_ptr);
  bool calculate_infeasible_redundant_constraints(problem_t<i_t, f_t>& pb);
  void update_host_bounds(const raft::handle_t* handle_ptr,
                          const raft::device_span<f_t> variable_lb,
                          const raft::device_span<f_t> variable_ub);
  void update_device_bounds(const raft::handle_t* handle_ptr);
  void copy_input_bounds(problem_t<i_t, f_t>& pb);
  void calc_and_set_updated_constraint_bounds(problem_t<i_t, f_t>& pb);

  mip_solver_context_t<i_t, f_t>& context;
  bounds_update_data_t<i_t, f_t> upd;
  std::vector<f_t> host_lb;
  std::vector<f_t> host_ub;
  settings_t settings;
  i_t infeas_constraints_count = 0;
  i_t redund_constraints_count = 0;
  probing_cache_t<i_t, f_t> probing_cache;
  i_t solve_iter;
};

}  // namespace cuopt::linear_programming::detail
