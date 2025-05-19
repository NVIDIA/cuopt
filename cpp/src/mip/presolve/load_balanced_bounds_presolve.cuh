/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <mip/problem/load_balanced_problem.cuh>
#include <mip/problem/problem.cuh>
#include <mip/solution/solution.cuh>
#include <mip/solver_context.cuh>
#include <mip/utils.cuh>

#include <utilities/timer.hpp>

#include "load_balanced_partition_helpers.cuh"
#include "utils.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class load_balanced_problem_t;

template <typename i_t, typename f_t>
class lb_probing_cache_t;

class managed_stream_pool {
 public:
  static constexpr std::size_t default_size{16};  ///< Default stream pool size

  /**
   * @brief Construct a new cuda stream pool object of the given non-zero size
   *
   * @throws logic_error if `pool_size` is zero
   * @param pool_size The number of streams in the pool
   */
  explicit managed_stream_pool(std::size_t pool_size = default_size) : streams_(pool_size)
  {
    RMM_EXPECTS(pool_size > 0, "Stream pool size must be greater than zero");
  }
  ~managed_stream_pool() = default;

  managed_stream_pool(managed_stream_pool&&)                 = delete;
  managed_stream_pool(managed_stream_pool const&)            = delete;
  managed_stream_pool& operator=(managed_stream_pool&&)      = delete;
  managed_stream_pool& operator=(managed_stream_pool const&) = delete;

  /**
   * @brief Get a `cuda_stream_view` of a stream in the pool.
   *
   * This function is thread safe with respect to other calls to the same function.
   *
   * @return rmm::cuda_stream_view
   */
  rmm::cuda_stream_view get_stream() const noexcept
  {
    int stream_id = (next_stream++) % streams_.size();
    end_unsycned  = std::max(stream_id, end_unsycned);
    return streams_[stream_id].view();
  }

  /**
   * @brief Get the number of streams in the pool.
   *
   * This function is thread safe with respect to other calls to the same function.
   *
   * @return the number of streams in the pool
   */
  std::size_t get_pool_size() const noexcept { return streams_.size(); }

  void wait_issued_on_event(cudaEvent_t e)
  {
    for (int i = 0; i < end_unsycned + 1; ++i) {
      cudaStreamWaitEvent(streams_[i].view(), e, 0);
    }
  }

  int reset_issued() const noexcept
  {
    int max_streams = end_unsycned;
    end_unsycned    = -1;
    next_stream     = 0;
    return max_streams;
  }

  std::vector<cudaEvent_t> create_events_on_issued()
  {
    std::vector<cudaEvent_t> events(end_unsycned + 1);
    for (auto& e : events) {
      cudaEventCreate(&e);
    }
    for (int i = 0; i < end_unsycned + 1; ++i) {
      cudaEventRecord(events[i], streams_[i].view());
    }
    return events;
  }

  void sync_test_all_issued()
  {
    for (int i = 0; i < end_unsycned + 1; ++i) {
      streams_[i].synchronize();
      RAFT_CHECK_CUDA(streams_[i].value());
    }
    end_unsycned = -1;
    next_stream  = 0;
  }

  void sync_all() const noexcept
  {
    for (size_t i = 0; i < streams_.size(); ++i) {
      streams_[i].synchronize();
    }
    end_unsycned = -1;
    next_stream  = 0;
  }

  void sync_all_issued() const noexcept
  {
    for (int i = 0; i < end_unsycned + 1; ++i) {
      streams_[i].synchronize();
    }
    end_unsycned = -1;
    next_stream  = 0;
  }

 private:
  std::vector<rmm::cuda_stream> streams_;
  mutable int next_stream{};
  mutable int end_unsycned{-1};
};

template <typename i_t, typename f_t>
class load_balanced_bounds_presolve_t {
 public:
  using f_t2                               = typename type_2<f_t>::type;
  static constexpr i_t heavy_degree_cutoff = 16 * 1024;

  struct settings_t {
    f_t time_limit{60.0};
    i_t iteration_limit{std::numeric_limits<i_t>::max()};
  };

  load_balanced_bounds_presolve_t(const load_balanced_problem_t<i_t, f_t>& problem,
                                  mip_solver_context_t<i_t, f_t>& context,
                                  settings_t settings   = settings_t{},
                                  i_t max_stream_count_ = 32);
  ~load_balanced_bounds_presolve_t();
  void copy_input_bounds(const load_balanced_problem_t<i_t, f_t>& pb);
  void setup(const load_balanced_problem_t<i_t, f_t>& pb);

  void calculate_activity_graph(bool erase_inf_cnst, bool dry_run = false);
  void calculate_bounds_update_graph(bool dry_run = false);

  void calculate_constraint_slack(const raft::handle_t* handle_ptr);
  void calculate_constraint_slack_iter(const raft::handle_t* handle_ptr);
  bool update_bounds_from_slack(const raft::handle_t* handle_ptr);

  termination_criterion_t bound_update_loop(const raft::handle_t* handle_ptr, timer_t timer);
  bool calculate_infeasible_redundant_constraints(const raft::handle_t* handle_ptr);

  // void calculate_constraint_slack_on_problem_bounds();

  termination_criterion_t solve(f_t lb, f_t ub, i_t var_idx);
  termination_criterion_t solve(raft::device_span<f_t> input_bounds = {});
  termination_criterion_t solve(const std::vector<thrust::pair<i_t, f_t>>& var_probe_val_pairs,
                                bool use_host_bounds = false);
  void update_host_bounds(const load_balanced_problem_t<i_t, f_t>& pb);
  void update_host_bounds(const raft::handle_t* handle_ptr);
  void update_device_bounds(const raft::handle_t* handle_ptr);
  void set_bounds(const std::vector<thrust::pair<i_t, f_t>>& var_probe_vals,
                  const raft::handle_t* handle_ptr);
  void set_updated_bounds(load_balanced_problem_t<i_t, f_t>* problem);

  struct activity_view_t {
    raft::device_span<const i_t> cnst_reorg_ids;
    raft::device_span<const f_t> coeff;
    raft::device_span<const i_t> vars;
    raft::device_span<const i_t> offsets;
    raft::device_span<const f_t2> cnst_bnd;  // new indexing
    raft::device_span<const f_t2> vars_bnd;  // old indexing
    raft::device_span<f_t2> cnst_slack;      // old indexing
    i_t nnz;
    typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances;
  };

  struct bounds_update_view_t {
    raft::device_span<const i_t> vars_reorg_ids;
    raft::device_span<const f_t> coeff;
    raft::device_span<const i_t> cnst;
    raft::device_span<const i_t> offsets;
    raft::device_span<const var_t> vars_types;  // new indexing
    raft::device_span<f_t2> vars_bnd;           // old indexing
    raft::device_span<f_t2> cnst_slack;         // old indexing
    i_t* bounds_changed;
    i_t nnz;
    typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances;
  };

  activity_view_t get_activity_view(const load_balanced_problem_t<i_t, f_t>& pb);
  bounds_update_view_t get_bounds_update_view(const load_balanced_problem_t<i_t, f_t>& pb);

  rmm::cuda_stream main_stream;
  rmm::cuda_stream act_stream;
  rmm::cuda_stream bnd_stream;
  managed_stream_pool streams;

  const load_balanced_problem_t<i_t, f_t>* pb;

  rmm::device_scalar<i_t> bounds_changed;

  rmm::device_uvector<f_t> cnst_slack;
  rmm::device_uvector<f_t> vars_bnd;
  rmm::device_uvector<f_t> tmp_act;
  rmm::device_uvector<f_t> tmp_bnd;

  // Number of blocks for heavy ids
  rmm::device_uvector<i_t> heavy_cnst_block_segments;
  rmm::device_uvector<i_t> heavy_vars_block_segments;
  rmm::device_uvector<i_t> heavy_cnst_vertex_ids;
  rmm::device_uvector<i_t> heavy_vars_vertex_ids;
  rmm::device_uvector<i_t> heavy_cnst_pseudo_block_ids;
  rmm::device_uvector<i_t> heavy_vars_pseudo_block_ids;

  i_t num_blocks_heavy_cnst;
  i_t num_blocks_heavy_vars;

  settings_t settings;
  // set to false when solving new problem
  bool calc_slack_erase_inf_cnst_graph_created;
  bool calc_slack_graph_created;
  bool upd_bnd_graph_created;

  cudaGraphExec_t calc_slack_erase_inf_cnst_exec;
  cudaGraph_t calc_slack_erase_inf_cnst_graph;
  cudaGraphExec_t calc_slack_exec;
  cudaGraph_t calc_slack_graph;
  cudaGraphExec_t upd_bnd_exec;
  cudaGraph_t upd_bnd_graph;

  bool is_cnst_sub_warp_single_bin;
  i_t cnst_sub_warp_count;
  rmm::device_uvector<i_t> warp_cnst_offsets;
  rmm::device_uvector<i_t> warp_cnst_id_offsets;

  bool is_vars_sub_warp_single_bin;
  i_t vars_sub_warp_count;
  rmm::device_uvector<i_t> warp_vars_offsets;
  rmm::device_uvector<i_t> warp_vars_id_offsets;

  std::vector<f_t> host_bounds;
  i_t infeas_constraints_count      = 0;
  bool infeas_cnst_slack_set_to_nan = false;
  mip_solver_context_t<i_t, f_t>& context;
  lb_probing_cache_t<i_t, f_t> probing_cache;
  i_t solve_iter;
};

}  // namespace cuopt::linear_programming::detail
