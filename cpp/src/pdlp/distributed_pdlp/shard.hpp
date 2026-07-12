/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <pdlp/distributed_pdlp/nccl_helpers.hpp>
#include <pdlp/distributed_pdlp/rank_data.hpp>
#include <utilities/macros.cuh>

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/optimization_problem.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <mip_heuristics/problem/problem.cuh>

#include <raft/core/device_setter.hpp>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

#include <nccl.h>

#include <memory>
#include <optional>
#include <vector>

namespace cuopt::mathematical_optimization::pdlp {

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
    cuopt_assert(device_id >= 0, "nccl_comm_deleter_t: device_id not set");
    raft::device_setter guard(device_id);
    CUOPT_NCCL_TRY_NO_THROW(ncclCommDestroy(comm));
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
               nccl_comm_unique_ptr_t&& comm,
               io::mps_data_model_t<i_t, f_t> const& mps,
               pdlp_solver_settings_t<i_t, f_t> const& settings);

  pdlp_shard_t(const pdlp_shard_t&)            = delete;
  pdlp_shard_t& operator=(const pdlp_shard_t&) = delete;

  int device_id{-1};
  rmm::cuda_stream stream;
  raft::handle_t handle;
  nccl_comm_unique_ptr_t comm;
  rank_data_t<i_t, f_t> rank_data;
  optimization_problem_t<i_t, f_t> opt_problem;
  std::optional<mip::problem_t<i_t, f_t>> sub_problem;
  std::unique_ptr<pdlp_solver_t<i_t, f_t>> sub_pdlp;

  // var_send_indices_d[peer] : local indices into primal vector to gather and ncclSend
  // var_send_buf_d    [peer] : staging buffer for outgoing variable values
  // cstr_send_indices_d/cstr_send_buf_d : same, for dual vector
  std::vector<rmm::device_uvector<i_t>> var_send_indices_d;
  std::vector<rmm::device_uvector<f_t>> var_send_buf_d;
  std::vector<rmm::device_uvector<i_t>> cstr_send_indices_d;
  std::vector<rmm::device_uvector<f_t>> cstr_send_buf_d;

  // Non-owning bundle of per-axis halo-exchange metadata, indexed by peer.
  // Consumed by multi_gpu_engine_t::halo_exchange_bufs_impl
  struct halo_axis_t {
    std::vector<rmm::device_uvector<i_t>>& send_indices;  // [peer]
    std::vector<rmm::device_uvector<f_t>>& send_buf;      // [peer]
    i_t owned_size;
    std::vector<i_t> const& recv_offsets;  // [peer]
    std::vector<i_t> const& recv_counts;   // [peer]
  };
  halo_axis_t var_halo_axis()
  {
    return {var_send_indices_d,
            var_send_buf_d,
            rank_data.owned_var_size,
            rank_data.var_recv_offsets,
            rank_data.var_recv_counts};
  }
  halo_axis_t cstr_halo_axis()
  {
    return {cstr_send_indices_d,
            cstr_send_buf_d,
            rank_data.owned_cstr_size,
            rank_data.cstr_recv_offsets,
            rank_data.cstr_recv_counts};
  }
};

}  // namespace cuopt::mathematical_optimization::pdlp
