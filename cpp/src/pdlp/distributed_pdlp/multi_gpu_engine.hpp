/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <pdlp/distributed_pdlp/rank_data.hpp>
#include <pdlp/distributed_pdlp/shard.hpp>
#include <pdlp/pdhg.hpp>
#include <utilities/cuda_helpers.cuh>
#include <utilities/event_handler.cuh>

#include <cuopt/linear_programming/io/mps_data_model.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/core/device_setter.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/reduce.cuh>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <cub/device/device_transform.cuh>
#include <cuda/std/tuple>

#include <nccl.h>

#include <cmath>
#include <memory>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

namespace cuopt::linear_programming::detail {

// Maps the solver floating-point type to the matching NCCL datatype so that
// halo exchanges / all-reduces transfer the correct element size for both
// double and float instantiations.
template <typename f_t>
constexpr ncclDataType_t nccl_data_type()
{
  static_assert(std::is_same_v<f_t, double> || std::is_same_v<f_t, float>,
                "Unsupported floating-point type for NCCL transfers");
  if constexpr (std::is_same_v<f_t, double>) {
    return ncclFloat64;
  } else {
    return ncclFloat32;
  }
}

// Element-wise sqrt functor. Defined at namespace scope (not as a local
// extended HD lambda) because nvcc disallows extended __host__ __device__
// lambdas appearing inside templates whose template arguments are
// themselves local lambda types (which happens when distributed_l2_norm is
// invoked with closure accessors).
template <typename f_t>
struct sqrt_inplace_op_t {
  __host__ __device__ f_t operator()(f_t x) const { return raft::sqrt(x); }
};

template <typename i_t, typename f_t>
struct multi_gpu_engine_t {
  // Constructs shards from rank_data. The global (unpartitioned) problem is
  // read straight from `mps`; each shard slices out the entries it owns.
  multi_gpu_engine_t(std::vector<rank_data_t<i_t, f_t>>&& rank_data,
                     io::mps_data_model_t<i_t, f_t> const& mps,
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
  // Allows to use distributed_transform on single input without having to do a std::make_tuple(in)
  template <typename InAccess, typename OutAccess, typename SizeAccess, typename Op>
  void distributed_transform(InAccess in, OutAccess out, SizeAccess sz, Op op)
  {
    distributed_transform(std::make_tuple(in), out, sz, op);
  }

  // -------- Halo exchange (variables / x) ---------------------------------
  // Fills the halo slice [owned_var_size, total_var_size) of the per-shard
  // input buffer returned by `buf_access(pdhg)` (the buffer A @ x will read).
  template <typename BufAccess>
  void halo_exchange_var(BufAccess&& buf_access)
  {
    halo_exchange_var_shard([&](pdlp_shard_t<i_t, f_t>& s) -> rmm::device_uvector<f_t>& {
      return buf_access(s.sub_pdlp->pdhg_solver_);
    });
  }

  // Core variable halo exchange. ShardBufAccess maps a shard to the var-shaped
  // device buffer to synchronize (owned slice [0, owned_var_size) followed by
  // the per-peer halo tail).
  // Step 1: thrust::gather per-peer outgoing values into staging buffers.
  // Step 2: a single NCCL group with matched ncclSend / ncclRecv across all
  // (rank, peer) pairs, receiving into each shard's halo region.
  template <typename ShardBufAccess>
  void halo_exchange_var_shard(ShardBufAccess&& buf_access)
  {
    const int nb = static_cast<int>(shards.size());

    // Step 1: gather owned values that each peer needs into per-peer staging.
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      auto& x = buf_access(s);
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
                 nccl_data_type<f_t>(),
                 peer,
                 s.comm.get(),
                 s.stream.view().value());
      }
    }
    for (int r = 0; r < nb; ++r) {
      auto& s  = *shards[r];
      auto& rd = s.rank_data;
      raft::device_setter guard(s.device_id);
      auto& x = buf_access(s);
      for (int peer = 0; peer < nb; ++peer) {
        if (peer == r) continue;
        f_t* recv_ptr = x.data() + rd.owned_var_size + rd.var_recv_offsets[peer];
        ncclRecv(recv_ptr,
                 static_cast<size_t>(rd.var_recv_counts[peer]),
                 nccl_data_type<f_t>(),
                 peer,
                 s.comm.get(),
                 s.stream.view().value());
      }
    }
    ncclGroupEnd();
  }

  // -------- Halo exchange (constraints / y) -------------------------------
  // Same as halo_exchange_var but for a constraint-shaped buffer (the input
  // A_T @ y will read) and constraint halos. buf_access maps a
  // pdhg_solver_t to the cstr-shaped buffer to exchange.
  template <typename BufAccess>
  void halo_exchange_cstr(BufAccess&& buf_access)
  {
    halo_exchange_cstr_shard([&](pdlp_shard_t<i_t, f_t>& s) -> rmm::device_uvector<f_t>& {
      return buf_access(s.sub_pdlp->pdhg_solver_);
    });
  }

  // Same as halo_exchange_var_shard for cstr
  template <typename ShardBufAccess>
  void halo_exchange_cstr_shard(ShardBufAccess&& buf_access)
  {
    const int nb = static_cast<int>(shards.size());

    // Gather each owner's owned values that peers need.
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      auto& y = buf_access(s);
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
                 nccl_data_type<f_t>(),
                 peer,
                 s.comm.get(),
                 s.stream.view().value());
      }
    }
    for (int r = 0; r < nb; ++r) {
      auto& s  = *shards[r];
      auto& rd = s.rank_data;
      raft::device_setter guard(s.device_id);
      auto& y = buf_access(s);
      for (int peer = 0; peer < nb; ++peer) {
        if (peer == r) continue;
        f_t* recv_ptr = y.data() + rd.owned_cstr_size + rd.cstr_recv_offsets[peer];
        ncclRecv(recv_ptr,
                 static_cast<size_t>(rd.cstr_recv_counts[peer]),
                 nccl_data_type<f_t>(),
                 peer,
                 s.comm.get(),
                 s.stream.view().value());
      }
    }
    ncclGroupEnd();
  }

  // -------- NCCL allreduce (sum, in place) --------------------------------
  // Per-shard in-place sum-allreduce. Each shard's stream issues an
  // ncclAllReduce(buf, buf, count, nccl_data_type<f_t>(), ncclSum, ...) inside a single
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
      ncclAllReduce(buf, buf, count, nccl_data_type<f_t>(), ncclSum, s->comm.get(), s->stream.view().value());
    }
    ncclGroupEnd();
  }

  // -------- Distributed L2 norm ------------------------------------------
  // Computes sqrt(Σ_k Σ_{i ∈ owned_k} buf_k[i]²) and writes the scalar into
  // the buffer returned by `out_access` on EVERY shard.
  //
  // Algorithm:
  //   1) per shard: out = cublasdot(buf[0:n_owned], buf[0:n_owned])  (partial Σ²)
  //   2) NCCL allreduce SUM on out (count = 1)                       (global Σ²)
  //   3) per shard: out = sqrt(out)
  //
  // The caller is responsible for clipping correctness via `size_access`
  // (which picks `rank_data.owned_var_size` or `rank_data.owned_cstr_size`
  // depending on the shape of the input buffer), and for mirroring the
  // result back to master if downstream code needs it there.
  //
  // BufAccess  : pdlp_solver_t<i_t,f_t>& -> rmm::device_uvector<f_t>&
  // OutAccess  : pdlp_solver_t<i_t,f_t>& -> f_t*   (single scalar in shard memory)
  // SizeAccess : pdlp_shard_t<i_t,f_t>&  -> i_t    (owned slice length)
  template <typename BufAccess, typename OutAccess, typename SizeAccess>
  void distributed_l2_norm(BufAccess&& buf_access, OutAccess&& out_access, SizeAccess&& size_access)
  {
    for_each_shard([&](auto& shard) {
      auto& sub   = *shard.sub_pdlp;
      auto& buf   = buf_access(sub);
      const i_t n = size_access(shard);
      f_t* out    = out_access(sub);
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(shard.handle.get_cublas_handle(),
                                                      static_cast<int>(n),
                                                      buf.data(),
                                                      1,
                                                      buf.data(),
                                                      1,
                                                      out,
                                                      shard.stream.view().value()));
    });

    allreduce_sum_inplace(out_access, /*count=*/1);

    for_each_shard([&](auto& shard) {
      f_t* out = out_access(*shard.sub_pdlp);
      cub::DeviceTransform::Transform(
        out, out, 1, sqrt_inplace_op_t<f_t>{}, shard.stream.view().value());
    });
  }

  // -------- High-level: A @ x and A_T @ y ---------------------------------
  // Distributed counterpart to pdhg_solver_t::compute_A_x() / compute_At_y().
  void distributed_compute_A_x();
  void distributed_compute_At_y();

  // Engine-level stream for fork/join orchestration (master side).
  rmm::cuda_stream stream;

  // Shards stored by unique_ptr because pdlp_shard_t is immovable
  // (owns device-affine resources: handle, NCCL comm, RMM buffers).
  std::vector<std::unique_ptr<pdlp_shard_t<i_t, f_t>>> shards;

  // ===== Cross-stream synchronization events =====
  // two different events
  // capture_*_event_ are used inside graph capture
  // ext_*_event_ are used when sync is needed outside of graph
  std::unique_ptr<cuopt::event_handler_t> graph_master_ready_event_;
  std::vector<std::unique_ptr<cuopt::event_handler_t>> graph_shard_ready_events_;
  std::unique_ptr<cuopt::event_handler_t> sync_master_ready_event_;
  std::vector<std::unique_ptr<cuopt::event_handler_t>> sync_shard_ready_events_;

  // Forks master stream to shards, so that the captured graph can see the work on the shards
  void graph_capture_fork_to_shards(rmm::cuda_stream_view master_stream);

  // Joins shards back to master stream for correct graph capture
  void graph_capture_join_from_shards(rmm::cuda_stream_view master_stream);

  // Functionnaly same as graph_capture_fork_to_shards but on a different event to avoid race
  // conditions Can be used as a way to sync shards with master stream
  void sync_await_master(rmm::cuda_stream_view master_stream);

  // Same as sync_await_master
  // Can be used as a way to sync master stream with shards
  void sync_await_shards(rmm::cuda_stream_view master_stream);
};

}  // namespace cuopt::linear_programming::detail
