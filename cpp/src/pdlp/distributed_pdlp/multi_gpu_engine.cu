/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pdlp/distributed_pdlp/multi_gpu_engine.hpp>
// compute_A_x() / compute_At_y() (defined inline in the engine header) call
// shard.sub_pdlp->pdhg_solver_.compute_* — pdlp_solver_t must be complete at
// the explicit instantiation point below.
#include <pdlp/pdlp.cuh>

#include <cuopt/error.hpp>

#include <raft/core/device_setter.hpp>

#include <nccl.h>

#include <numeric>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
multi_gpu_engine_t<i_t, f_t>::multi_gpu_engine_t(
  std::vector<rank_data_t<i_t, f_t>>&& rank_data,
  std::vector<f_t> const& h_global_obj,
  std::vector<f_t> const& h_global_var_lower,
  std::vector<f_t> const& h_global_var_upper,
  std::vector<f_t> const& h_global_cstr_lower,
  std::vector<f_t> const& h_global_cstr_upper,
  std::vector<f_t> const& h_global_obj_scaled,
  std::vector<f_t> const& h_global_var_lower_scaled,
  std::vector<f_t> const& h_global_var_upper_scaled,
  std::vector<f_t> const& h_global_cstr_lower_scaled,
  std::vector<f_t> const& h_global_cstr_upper_scaled,
  std::vector<f_t> const& h_global_cummulative_cstr_scaling,
  std::vector<f_t> const& h_global_cummulative_var_scaling,
  f_t h_bound_rescaling,
  f_t h_objective_rescaling,
  bool maximize,
  f_t objective_offset,
  f_t objective_scaling_factor,
  pdlp_solver_settings_t<i_t, f_t> const& sub_solver_settings)
  : stream()
{
  const int nb_parts = static_cast<int>(rank_data.size());
  cuopt_expects(
    nb_parts > 0, error_type_t::ValidationError, "multi_gpu_engine_t: rank_data must be non-empty");

  shards.reserve(nb_parts);
  std::vector<int> devices(nb_parts);
  std::iota(devices.begin(), devices.end(), 0);

  // Create NCCL Comms then let shards own them
  std::vector<ncclComm_t> raw_comms(nb_parts);
  cuopt_expects(ncclCommInitAll(raw_comms.data(), nb_parts, devices.data()) == ncclSuccess,
                error_type_t::RuntimeError,
                "ncclCommInitAll failed");

  // 3. Construct one shard per rank, pinned to its device.
  for (int r = 0; r < nb_parts; ++r) {
    raft::device_setter guard(devices[r]);  // shard ctor needs device set
    shards.emplace_back(std::make_unique<pdlp_shard_t<i_t, f_t>>(devices[r],
                                                                 std::move(rank_data[r]),
                                                                 raw_comms[r],
                                                                 h_global_obj,
                                                                 h_global_var_lower,
                                                                 h_global_var_upper,
                                                                 h_global_cstr_lower,
                                                                 h_global_cstr_upper,
                                                                 h_global_obj_scaled,
                                                                 h_global_var_lower_scaled,
                                                                 h_global_var_upper_scaled,
                                                                 h_global_cstr_lower_scaled,
                                                                 h_global_cstr_upper_scaled,
                                                                 h_global_cummulative_cstr_scaling,
                                                                 h_global_cummulative_var_scaling,
                                                                 h_bound_rescaling,
                                                                 h_objective_rescaling,
                                                                 maximize,
                                                                 objective_offset,
                                                                 objective_scaling_factor,
                                                                 sub_solver_settings));
  }

  // Two different events
  // capture_*_event_ are used inside graph capture
  // ext_*_event_ are used when sync is needed outside of graph
  graph_master_ready_event_ = std::make_unique<cuopt::event_handler_t>();
  sync_master_ready_event_  = std::make_unique<cuopt::event_handler_t>();
  graph_shard_ready_events_.reserve(nb_parts);
  sync_shard_ready_events_.reserve(nb_parts);
  for (int r = 0; r < nb_parts; ++r) {
    raft::device_setter guard(devices[r]);
    graph_shard_ready_events_.emplace_back(std::make_unique<cuopt::event_handler_t>());
    sync_shard_ready_events_.emplace_back(std::make_unique<cuopt::event_handler_t>());
  }
}

template struct multi_gpu_engine_t<int, double>;
// template struct multi_gpu_engine_t<int, float>;

}  // namespace cuopt::linear_programming::detail
