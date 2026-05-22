/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <pdlp/distributed_pdlp/rank_data.hpp>

#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <mip_heuristics/problem/problem.cuh>

#include <raft/core/device_setter.hpp>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

#include <nccl.h>

#include <memory>
#include <optional>
#include <vector>

namespace cuopt::linear_programming::detail {

// Forward-declare to break the cyclic include with pdlp.cuh
// (pdlp.cuh -> multi_gpu_engine.hpp -> shard.hpp -> pdlp.cuh).
// Definitions of out-of-line members live in shard.cu, which includes pdlp.cuh.
template <typename i_t, typename f_t>
class pdlp_solver_t;

// RAII deleter for ncclComm_t; sets the right device before destroy.
struct nccl_comm_deleter_t {
  int device_id{-1};
  void operator()(ncclComm* comm) const noexcept
  {
    if (comm == nullptr) return;
    raft::device_setter guard(device_id);
    ncclCommDestroy(comm);
  }
};
using nccl_comm_unique_ptr_t = std::unique_ptr<ncclComm, nccl_comm_deleter_t>;

template <typename i_t, typename f_t>
struct pdlp_shard_t {
  // Out-of-line (in shard.cu) because pdlp_solver_t is incomplete here.
  ~pdlp_shard_t();

  // sub worker for distributed pdlp. Owns its own view on scaled problem and unscaled problem
  // Owns necessary multi-gpu data (rank_data, device_id, nccl_comm)
  pdlp_shard_t(int device_id,
               rank_data_t<i_t, f_t>&& rd,
               ncclComm_t raw_comm,
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
               pdlp_solver_settings_t<i_t, f_t> const& settings);

  pdlp_shard_t(const pdlp_shard_t&)            = delete;
  pdlp_shard_t& operator=(const pdlp_shard_t&) = delete;
  // Move ops are implicitly deleted (user-declared dtor + deleted copy).
  // Intentional: shard owns device-affine resources and must never move.
  // Store as std::unique_ptr in any container.

  int device_id;
  rmm::cuda_stream stream;
  raft::handle_t handle;
  nccl_comm_unique_ptr_t comm;
  rank_data_t<i_t, f_t> rank_data;
  std::optional<optimization_problem_t<i_t, f_t>> opt_problem;
  std::optional<problem_t<i_t, f_t>> sub_problem;
  std::unique_ptr<pdlp_solver_t<i_t, f_t>> sub_pdlp;

  // Per-peer halo-exchange state. Inner index = peer rank.
  // Slot for self (peer == this rank) is present but unused (size 0).
  // var_send_indices_d[peer] : local indices into primal vector to gather and ncclSend
  // var_send_buf_d    [peer] : staging buffer for outgoing variable values
  // cstr_send_indices_d/cstr_send_buf_d : same, for dual vector
  std::vector<rmm::device_uvector<i_t>> var_send_indices_d;
  std::vector<rmm::device_uvector<f_t>> var_send_buf_d;
  std::vector<rmm::device_uvector<i_t>> cstr_send_indices_d;
  std::vector<rmm::device_uvector<f_t>> cstr_send_buf_d;
};

}  // namespace cuopt::linear_programming::detail
