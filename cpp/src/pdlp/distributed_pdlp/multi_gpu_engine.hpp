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

#include <thrust/gather.h>
#include <cub/device/device_transform.cuh>
#include <cuda/std/tuple>

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

  template <typename... InAccess, typename OutAccess, typename SizeAccess, typename Op>
  void distributed_transform(std::tuple<InAccess...> in_accessors,
                             OutAccess out,
                             SizeAccess sz,
                             Op op)
  {
    for_each_shard([&](auto& shard) {
      auto& sub = *shard.sub_pdlp;
      // turns the Tuple of lambdas into a tuple of rmm::device_uvector
      auto cub_inputs = std::apply(
        [&sub](auto&... acc) { return cuda::std::make_tuple(acc(sub)...); }, in_accessors);

      cub::DeviceTransform::Transform(cub_inputs, out(sub), sz(sub), op, shard.stream.view());
    });
  }
  // --- 2) convenience: single input accessor (delegates) ---
  template <typename InAccess, typename OutAccess, typename SizeAccess, typename Op>
  void distributed_transform(InAccess in, OutAccess out, SizeAccess sz, Op op)
  {
    distributed_transform(std::make_tuple(in), out, sz, op);
  }

  // -------- Halo exchange (variables / x) ---------------------------------
  // Fills the halo slice [owned_var_size, total_var_size) of the per-shard
  // input buffer returned by `buf_access(pdhg)` (the buffer A @ x will read).
  // Step 1: thrust::gather per-peer outgoing values into staging buffers.
  // Step 2: a single NCCL group with matched ncclSend / ncclRecv across all
  // (rank, peer) pairs.
  template <typename BufAccess>
  void halo_exchange_var(BufAccess&& buf_access)
  {
    const int nb = static_cast<int>(shards.size());

    // Step 1: gather owned values that each peer needs into per-peer staging.
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      auto& x = buf_access(s.sub_pdlp->pdhg_solver_);
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
      auto& s  = *shards[r];
      auto& rd = s.rank_data;
      raft::device_setter guard(s.device_id);
      auto& x = buf_access(s.sub_pdlp->pdhg_solver_);
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
  // Same as halo_exchange_var but for a constraint-shaped buffer (the input
  // A_T @ y will read) and constraint halos.
  template <typename BufAccess>
  void halo_exchange_cstr(BufAccess&& buf_access)
  {
    const int nb = static_cast<int>(shards.size());

    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      auto& y = buf_access(s.sub_pdlp->pdhg_solver_);
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
      auto& s  = *shards[r];
      auto& rd = s.rank_data;
      raft::device_setter guard(s.device_id);
      auto& y = buf_access(s.sub_pdlp->pdhg_solver_);
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

  // -------- NCCL allreduce (sum, in place) --------------------------------
  // Per-shard in-place sum-allreduce. Each shard's stream issues an
  // ncclAllReduce(buf, buf, count, ncclFloat64, ncclSum, ...) inside a single
  // group. After this returns, every shard's buffer holds the global sum.
  //
  // PtrAccess: pdlp_solver_t<i_t,f_t>& -> f_t*  (e.g. into step_size_strategy_).
  template <typename PtrAccess>
  void allreduce_sum_inplace(PtrAccess&& ptr_access, size_t count = 1)
  {
    ncclGroupStart();
    for (auto& s : shards) {
      raft::device_setter guard(s->device_id);
      f_t* buf = ptr_access(*s->sub_pdlp);
      ncclAllReduce(buf, buf, count, ncclFloat64, ncclSum, s->comm.get(), s->stream.view().value());
    }
    ncclGroupEnd();
  }

  // -------- Generic distributed SpMVs -------------------------------------
  // distributed_spmv_A : halo-update the var-shaped input buffer returned by
  // `in_buf(pdhg)`, then per-shard A @ in_buf -> out_desc.
  // distributed_spmv_At: halo-update the cstr-shaped input buffer returned by
  // `in_buf(pdhg)`, then per-shard A_T @ in_buf -> out_desc.
  //
  // Accessor signatures:
  //   in_buf  (pdhg_solver_t<i_t,f_t>&) -> rmm::device_uvector<f_t>&
  //   out_desc(pdhg_solver_t<i_t,f_t>&) -> cusparseDnVecDescr_t
  template <typename InBufAccess, typename OutDescAccess>
  void distributed_spmv_A(InBufAccess&& in_buf, OutDescAccess&& out_desc)
  {
    halo_exchange_var(in_buf);
    for_each_shard([&](auto& shard) {
      auto& sub_pdhg = shard.sub_pdlp->pdhg_solver_;
      sub_pdhg.spmv_A_into(in_buf(sub_pdhg), out_desc(sub_pdhg));
    });
  }

  template <typename InBufAccess, typename OutDescAccess>
  void distributed_spmv_At(InBufAccess&& in_buf, OutDescAccess&& out_desc)
  {
    halo_exchange_cstr(in_buf);
    for_each_shard([&](auto& shard) {
      auto& sub_pdhg = shard.sub_pdlp->pdhg_solver_;
      sub_pdhg.spmv_At_into(in_buf(sub_pdhg), out_desc(sub_pdhg));
    });
  }

  // -------- High-level: A @ x and A_T @ y ---------------------------------
  // Thin wrappers used from pdhg_solver_t::compute_A_x / compute_At_y when an
  // engine is wired in. They use the canonical PDHG buffers/descriptors so the
  // result lands where single-GPU PDHG would have put it (dual_gradient for A,
  // current_AtY for A_T).
  void distributed_compute_A_x()
  {
    distributed_spmv_A(
      [](auto& pdhg) -> rmm::device_uvector<f_t>& { return pdhg.get_reflected_primal(); },
      [](auto& pdhg) -> cusparseDnVecDescr_t { return pdhg.get_cusparse_view().dual_gradient; });
  }

  void distributed_compute_At_y()
  {
    distributed_spmv_At(
      [](auto& pdhg) -> rmm::device_uvector<f_t>& { return pdhg.get_dual_solution(); },
      [](auto& pdhg) -> cusparseDnVecDescr_t { return pdhg.get_cusparse_view().current_AtY; });
  }

  // Engine-level stream for fork/join orchestration (master side).
  rmm::cuda_stream stream;

  // Shards stored by unique_ptr because pdlp_shard_t is immovable
  // (owns device-affine resources: handle, NCCL comm, RMM buffers).
  std::vector<std::unique_ptr<pdlp_shard_t<i_t, f_t>>> shards;
};

}  // namespace cuopt::linear_programming::detail
