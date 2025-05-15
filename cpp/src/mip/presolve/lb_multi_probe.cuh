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

#include "lb_bounds_update_data.cuh"
#include "load_balanced_bounds_presolve.cuh"
#include "utils.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class lb_multi_probe_t {
 public:
  static constexpr i_t heavy_degree_cutoff = 16 * 1024;
  struct settings_t {
    f_t time_limit{60.0};
    i_t iteration_limit{std::numeric_limits<i_t>::max()};
  };

  struct activity_view_t {
    using f_t2 = typename type_2<f_t>::type;
    raft::device_span<const i_t> cnst_reorg_ids;
    raft::device_span<const f_t> coeff;
    raft::device_span<const i_t> vars;
    raft::device_span<const i_t> offsets;
    raft::device_span<const f_t2> cnst_bnd;  // new indexing
    i_t nnz;
    typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances;
  };

  struct bounds_update_view_t {
    using f_t2 = typename type_2<f_t>::type;
    raft::device_span<const i_t> vars_reorg_ids;
    raft::device_span<const f_t> coeff;
    raft::device_span<const i_t> cnst;
    raft::device_span<const i_t> offsets;
    raft::device_span<const var_t> vars_types;  // new indexing
    i_t nnz;
    typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances;
  };
  ~lb_multi_probe_t();
  lb_multi_probe_t(lb_multi_probe_t&&) = default;

  lb_multi_probe_t(const load_balanced_problem_t<i_t, f_t>& problem_,
                   mip_solver_context_t<i_t, f_t>& context_,
                   settings_t settings   = settings_t{},
                   i_t max_stream_count_ = 32);
  void setup(const load_balanced_problem_t<i_t, f_t>& problem);

  activity_view_t get_activity_view(const load_balanced_problem_t<i_t, f_t>& pb);
  bounds_update_view_t get_bounds_update_view(const load_balanced_problem_t<i_t, f_t>& pb);

  void calculate_cnst_slack(const raft::handle_t* handle_ptr);
  bool calculate_bounds_update(const raft::handle_t* handle_ptr);
  void calculate_bounds_update_graph(bool dry_run = false);
  void calculate_cnst_slack_graph(bool erase_inf_cnst, bool dry_run = false);
  termination_criterion_t solve(
    const std::tuple<std::vector<i_t>, std::vector<f_t>, std::vector<f_t>>& var_probe_vals);

  termination_criterion_t solve_for_interval(
    const std::tuple<i_t, std::pair<f_t, f_t>, std::pair<f_t, f_t>>& var_interval_vals,
    const raft::handle_t* handle_ptr);

  void set_updated_bounds(load_balanced_problem_t<i_t, f_t>& problem,
                          i_t select_update,
                          const raft::handle_t* handle_ptr);
  void set_updated_bounds(const raft::handle_t* handle_ptr,
                          raft::device_span<f_t> output_bounds,
                          i_t select_update);
  void set_updated_bounds(const raft::handle_t* handle_ptr,
                          raft::device_span<f_t> output_lb,
                          raft::device_span<f_t> output_ub,
                          i_t select_update);
  termination_criterion_t bound_update_loop(const raft::handle_t* handle_ptr, timer_t timer);
  void set_interval_bounds(
    const std::tuple<i_t, std::pair<f_t, f_t>, std::pair<f_t, f_t>>& var_interval_vals,
    const raft::handle_t* handle_ptr);
  void set_bounds(
    const std::tuple<std::vector<i_t>, std::vector<f_t>, std::vector<f_t>>& var_probe_vals,
    const raft::handle_t* handle_ptr);
  // void constraint_stats(problem_t<i_t, f_t>& pb, const raft::handle_t* handle_ptr);
  void copy_problem_into_probing_buffers(const raft::handle_t* handle_ptr);

  mip_solver_context_t<i_t, f_t>& context;

  const load_balanced_problem_t<i_t, f_t>* pb;

  managed_stream_pool streams;
  lb_bounds_update_data_t<i_t, f_t> upd_0;
  lb_bounds_update_data_t<i_t, f_t> upd_1;

  // Number of blocks for heavy ids
  rmm::device_uvector<i_t> heavy_cnst_block_segments;
  rmm::device_uvector<i_t> heavy_vars_block_segments;
  rmm::device_uvector<i_t> heavy_cnst_vertex_ids;
  rmm::device_uvector<i_t> heavy_vars_vertex_ids;
  rmm::device_uvector<i_t> heavy_cnst_pseudo_block_ids;
  rmm::device_uvector<i_t> heavy_vars_pseudo_block_ids;

  i_t num_blocks_heavy_cnst;
  i_t num_blocks_heavy_vars;

  // sub warp meta data
  bool is_cnst_sub_warp_single_bin;
  i_t cnst_sub_warp_count;
  rmm::device_uvector<i_t> warp_cnst_offsets;
  rmm::device_uvector<i_t> warp_cnst_id_offsets;

  bool is_vars_sub_warp_single_bin;
  i_t vars_sub_warp_count;
  rmm::device_uvector<i_t> warp_vars_offsets;
  rmm::device_uvector<i_t> warp_vars_id_offsets;

  // graphs
  bool calc_slack_erase_inf_cnst_graph_created;
  bool calc_slack_graph_created;
  bool upd_bnd_graph_created;

  cudaGraphExec_t calc_slack_erase_inf_cnst_exec;
  cudaGraph_t calc_slack_erase_inf_cnst_graph;
  cudaGraphExec_t calc_slack_exec;
  cudaGraph_t calc_slack_graph;
  cudaGraphExec_t upd_bnd_exec;
  cudaGraph_t upd_bnd_graph;

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
