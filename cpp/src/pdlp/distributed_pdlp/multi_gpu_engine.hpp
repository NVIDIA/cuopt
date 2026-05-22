/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <pdlp/distributed_pdlp/rank_data.hpp>
#include <pdlp/distributed_pdlp/shard.hpp>
#include <pdlp/pdhg.hpp>

#include <cuopt/linear_programming/pdlp/solver_settings.hpp>

#include <raft/core/device_setter.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_transform.cuh>
#include <cuda/std/tuple>
#include <thrust/gather.h>

#include <nccl.h>

#include <memory>
#include <tuple>
#include <vector>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct multi_gpu_engine_t {
  // Constructs shards from rank_data
  multi_gpu_engine_t(std::vector<rank_data_t<i_t, f_t>>&& rank_data,
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
                     pdlp_solver_settings_t<i_t, f_t> const& sub_solver_settings);

  multi_gpu_engine_t(const multi_gpu_engine_t&)            = delete;
  multi_gpu_engine_t& operator=(const multi_gpu_engine_t&) = delete;



  template <typename Fn>
  void for_each_shard(Fn&& fn)
  {
    for (auto& s : shards) {
      raft::device_setter guard(s->device_id);   
      fn(*s);                                     
    }
  }

  template <typename... InAccess,
          typename OutAccess,
          typename SizeAccess,
          typename Op>
  void distributed_transform(std::tuple<InAccess...> in_accessors,
                            OutAccess                out,
                            SizeAccess               sz,
                            Op                       op)
  {
    for_each_shard([&](auto& shard) {
      auto& sub = *shard.sub_pdlp;
      // turns the Tuple of lambdas into a tuple of rmm::device_uvector
      auto cub_inputs = std::apply(
        [&sub](auto&... acc) { return cuda::std::make_tuple(acc(sub)...); },
        in_accessors);

      cub::DeviceTransform::Transform(cub_inputs,
                                      out(sub),
                                      sz(sub),
                                      op,
                                      shard.stream.view());
    });
  }
  // --- 2) convenience: single input accessor (delegates) ---
  template <typename InAccess,
  typename OutAccess,
  typename SizeAccess,
  typename Op>
  void distributed_transform(InAccess   in,
                  OutAccess  out,
                  SizeAccess sz,
                  Op         op)
  {
  distributed_transform(std::make_tuple(in), out, sz, op);
  }

  // -------- Halo exchange (variables / x) ---------------------------------
  // Fills the halo slice [owned_var_size, total_var_size) of the per-shard
  // reflected_primal vector (the buffer A @ x reads). Step 1: thrust::gather
  // per-peer outgoing values into staging buffers. Step 2: a single NCCL
  // group with matched ncclSend / ncclRecv across all (rank, peer) pairs.
  void halo_exchange_var()
  {
    const int nb = static_cast<int>(shards.size());

    // Step 1: gather owned values that each peer needs into per-peer staging.
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      auto& x = s.sub_pdlp->pdhg_solver_.get_reflected_primal();
      for (int peer = 0; peer < nb; ++peer) {
        if (peer == r) continue;
        if (s.var_send_indices_d[peer].size() == 0) continue;
        thrust::gather(rmm::exec_policy_nosync(s.stream.view()),
                       s.var_send_indices_d[peer].begin(),
                       s.var_send_indices_d[peer].end(),
                       x.begin(),
                       s.var_send_buf_d[peer].begin());
      }
    }

    // Step 2: matched send / recv across the whole topology in one NCCL group.
    ncclGroupStart();
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      for (int peer = 0; peer < nb; ++peer) {
        if (peer == r) continue;
        ncclSend(s.var_send_buf_d[peer].data(),
                 s.var_send_buf_d[peer].size(),
                 ncclFloat64,
                 peer,
                 s.comm.get(),
                 s.stream.view().value());
      }
    }
    for (int r = 0; r < nb; ++r) {
      auto& s   = *shards[r];
      auto& rd  = s.rank_data;
      raft::device_setter guard(s.device_id);
      auto& x   = s.sub_pdlp->pdhg_solver_.get_reflected_primal();
      for (int peer = 0; peer < nb; ++peer) {
        if (peer == r) continue;
        f_t* recv_ptr = x.data() + rd.owned_var_size + rd.var_recv_offsets[peer];
        ncclRecv(recv_ptr,
                 static_cast<size_t>(rd.var_recv_counts[peer]),
                 ncclFloat64,
                 peer,
                 s.comm.get(),
                 s.stream.view().value());
      }
    }
    ncclGroupEnd();
  }

  // -------- Halo exchange (constraints / y) -------------------------------
  // Same as halo_exchange_var but for the per-shard dual solution (the buffer
  // A_T @ y reads) and constraint halos.
  void halo_exchange_cstr()
  {
    const int nb = static_cast<int>(shards.size());

    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      auto& y = s.sub_pdlp->pdhg_solver_.get_dual_solution();
      for (int peer = 0; peer < nb; ++peer) {
        if (peer == r) continue;
        if (s.cstr_send_indices_d[peer].size() == 0) continue;
        thrust::gather(rmm::exec_policy_nosync(s.stream.view()),
                       s.cstr_send_indices_d[peer].begin(),
                       s.cstr_send_indices_d[peer].end(),
                       y.begin(),
                       s.cstr_send_buf_d[peer].begin());
      }
    }

    ncclGroupStart();
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      for (int peer = 0; peer < nb; ++peer) {
        if (peer == r) continue;
        ncclSend(s.cstr_send_buf_d[peer].data(),
                 s.cstr_send_buf_d[peer].size(),
                 ncclFloat64,
                 peer,
                 s.comm.get(),
                 s.stream.view().value());
      }
    }
    for (int r = 0; r < nb; ++r) {
      auto& s   = *shards[r];
      auto& rd  = s.rank_data;
      raft::device_setter guard(s.device_id);
      auto& y   = s.sub_pdlp->pdhg_solver_.get_dual_solution();
      for (int peer = 0; peer < nb; ++peer) {
        if (peer == r) continue;
        f_t* recv_ptr = y.data() + rd.owned_cstr_size + rd.cstr_recv_offsets[peer];
        ncclRecv(recv_ptr,
                 static_cast<size_t>(rd.cstr_recv_counts[peer]),
                 ncclFloat64,
                 peer,
                 s.comm.get(),
                 s.stream.view().value());
      }
    }
    ncclGroupEnd();
  }

  // -------- High-level: A @ x and A_T @ y ---------------------------------
  // A @ x: halo-update the reflected_primal vector, then per-shard SpMV.
  // Named distributed_* (rather than compute_*) to make call sites in pdhg.cu
  // self-documenting and to avoid name collision with pdhg_solver_t's own
  // compute_A_x / compute_At_y, which the engine dispatches into per shard.
  void distributed_compute_A_x()
  {
    halo_exchange_var();
    for_each_shard([&](auto& shard) {
      shard.sub_pdlp->pdhg_solver_.spmvop_A_x();
    });
  }

  // A_T @ y: halo-update the dual solution vector, then per-shard SpMV.
  void distributed_compute_At_y()
  {
    halo_exchange_cstr();
    for_each_shard([&](auto& shard) {
      shard.sub_pdlp->pdhg_solver_.spmvop_At_y();
    });
  }

  // Engine-level stream for fork/join orchestration (master side).
  rmm::cuda_stream stream;

  // Shards stored by unique_ptr because pdlp_shard_t is immovable
  // (owns device-affine resources: handle, NCCL comm, RMM buffers).
  std::vector<std::unique_ptr<pdlp_shard_t<i_t, f_t>>> shards;
};

}  // namespace cuopt::linear_programming::detail
