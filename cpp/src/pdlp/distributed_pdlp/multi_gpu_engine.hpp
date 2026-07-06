/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <pdlp/distributed_pdlp/nccl_helpers.hpp>
#include <pdlp/distributed_pdlp/rank_data.hpp>
#include <pdlp/distributed_pdlp/shard.hpp>
#include <pdlp/pdhg.hpp>
#include <utilities/cuda_helpers.cuh>
#include <utilities/event_handler.cuh>

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_setter.hpp>
#include <raft/core/device_span.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/reduce.cuh>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <cub/device/device_transform.cuh>
#include <cuda/std/cmath>
#include <cuda/std/tuple>

#include <nccl.h>

#include <cmath>
#include <memory>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

namespace cuopt::mathematical_optimization::pdlp {

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
  __host__ __device__ f_t operator()(f_t x) const { return cuda::std::sqrt(x); }
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

  // Core: launches cub::DeviceTransform on every shard using per-shard
  // pre-resolved inputs / outputs / sizes.
  //   - in_tuples[r] is the tuple passed as cub input for shard r (any
  //     iterator-shaped types cub accepts: raw pointers, thrust iterators, ...)
  //   - outs[r]      is the output iterator for shard r
  //   - sizes[r]     is the element count for shard r
  // All three must have size == shards.size(). The heterogeneous per-shard
  // input types are captured by the caller into whatever container / element
  // type is convenient (e.g. std::vector<cuda::std::tuple<f_t*, f_t*>>).
  template <typename PerShardInTuple, typename OutIter, typename Op>
  void distributed_transform_bufs(std::vector<PerShardInTuple> const& in_tuples,
                                  std::vector<OutIter> const& outs,
                                  std::vector<i_t> const& sizes,
                                  Op op)
  {
    const int nb = static_cast<int>(shards.size());
    cuopt_expects(static_cast<int>(in_tuples.size()) == nb && static_cast<int>(outs.size()) == nb &&
                    static_cast<int>(sizes.size()) == nb,
                  error_type_t::RuntimeError,
                  "distributed_transform_bufs: in_tuples / outs / sizes must "
                  "all have size == shards.size()");
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      cub::DeviceTransform::Transform(in_tuples[r], outs[r], sizes[r], op, s.stream.view());
    }
  }

  // Wrapper: accessor form. Resolves each shard's cub input_tuple / output /
  // size via the provided accessors, then delegates to
  // distributed_transform_bufs.
  template <typename... InAccess, typename OutAccess, typename SizeAccess, typename Op>
  void distributed_transform(std::tuple<InAccess...> in_accessors,
                             OutAccess out,
                             SizeAccess sz,
                             Op op)
  {
    cuopt_expects(
      !shards.empty(), error_type_t::RuntimeError, "distributed_transform: engine has no shards");

    // Deduce per-shard tuple / output types from the accessors themselves so
    // the runtime vector element types match cub's expectations exactly.
    auto& sample_sub = *shards[0]->sub_pdlp;
    using in_tuple_t = decltype(std::apply(
      [&sample_sub](auto&... acc) { return cuda::std::make_tuple(acc(sample_sub)...); },
      in_accessors));
    using out_iter_t = decltype(out(sample_sub));

    std::vector<in_tuple_t> in_tuples;
    std::vector<out_iter_t> outs;
    std::vector<i_t> sizes;
    in_tuples.reserve(shards.size());
    outs.reserve(shards.size());
    sizes.reserve(shards.size());
    for (auto& s : shards) {
      raft::device_setter guard(s->device_id);
      auto& sub = *s->sub_pdlp;
      // Turns a tuple of accessors into a tuple of values.
      in_tuples.emplace_back(std::apply(
        [&sub](auto&... acc) { return cuda::std::make_tuple(acc(sub)...); }, in_accessors));
      outs.emplace_back(out(sub));
      sizes.emplace_back(sz(sub));
    }
    distributed_transform_bufs(in_tuples, outs, sizes, op);
  }

  // --- 2) convenience: single input accessor (delegates) ---
  // Allows to use distributed_transform on single input without having to do a std::make_tuple(in)
  template <typename InAccess, typename OutAccess, typename SizeAccess, typename Op>
  void distributed_transform(InAccess in, OutAccess out, SizeAccess sz, Op op)
  {
    distributed_transform(std::make_tuple(in), out, sz, op);
  }

  // -------- Halo exchange (variables / x) ---------------------------------
  // Step 1: thrust::gather per-peer outgoing values into staging buffers.
  // Step 2: a single NCCL group with matched ncclSend / ncclRecv across all
  // (rank, peer) pairs, receiving into each shard's halo region.
  void halo_exchange_var_bufs(std::vector<raft::device_span<f_t>> const& bufs)
  {
    const int nb = static_cast<int>(shards.size());
    cuopt_expects(static_cast<int>(bufs.size()) == nb,
                  error_type_t::RuntimeError,
                  "halo_exchange_var_bufs: bufs.size() must equal shards.size()");

    // Step 1: gather owned values that each peer needs into per-peer staging.
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      auto x = bufs[r];
      for (int peer = 0; peer < nb; ++peer) {
        if (peer == r) continue;
        if (s.var_send_indices_d[peer].size() == 0) continue;
        thrust::gather(rmm::exec_policy_nosync(s.stream.view()),
                       s.var_send_indices_d[peer].begin(),
                       s.var_send_indices_d[peer].end(),
                       x.data(),
                       s.var_send_buf_d[peer].begin());
      }
    }

    // Step 2: matched send / recv across the whole topology in one NCCL group.
    CUOPT_NCCL_TRY(ncclGroupStart());
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      for (int peer = 0; peer < nb; ++peer) {
        if (peer == r) continue;
        CUOPT_NCCL_TRY(ncclSend(s.var_send_buf_d[peer].data(),
                                s.var_send_buf_d[peer].size(),
                                nccl_data_type<f_t>(),
                                peer,
                                s.comm.get(),
                                s.stream.view().value()));
      }
    }
    for (int r = 0; r < nb; ++r) {
      auto& s  = *shards[r];
      auto& rd = s.rank_data;
      raft::device_setter guard(s.device_id);
      auto x = bufs[r];
      for (int peer = 0; peer < nb; ++peer) {
        if (peer == r) continue;
        f_t* recv_ptr = x.data() + rd.owned_var_size + rd.var_recv_offsets[peer];
        CUOPT_NCCL_TRY(ncclRecv(recv_ptr,
                                static_cast<size_t>(rd.var_recv_counts[peer]),
                                nccl_data_type<f_t>(),
                                peer,
                                s.comm.get(),
                                s.stream.view().value()));
      }
    }
    CUOPT_NCCL_TRY(ncclGroupEnd());
  }

  // Wrapper: pdhg_solver_t accessor.
  // buf_access : pdhg_solver_t<i_t,f_t>& -> rmm::device_uvector<f_t>&
  template <typename BufAccess>
  void halo_exchange_var(BufAccess&& buf_access)
  {
    halo_exchange_var_shard([&](pdlp_shard_t<i_t, f_t>& s) -> rmm::device_uvector<f_t>& {
      return buf_access(s.sub_pdlp->pdhg_solver_);
    });
  }

  // Wrapper: pdlp_shard_t accessor. Resolves one uvector per shard into a
  // vector of spans, then delegates to halo_exchange_var_bufs.
  // buf_access : pdlp_shard_t<i_t,f_t>& -> rmm::device_uvector<f_t>&
  template <typename ShardBufAccess>
  void halo_exchange_var_shard(ShardBufAccess&& buf_access)
  {
    std::vector<raft::device_span<f_t>> bufs;
    bufs.reserve(shards.size());
    for (auto& s : shards) {
      raft::device_setter guard(s->device_id);
      auto& x = buf_access(*s);
      bufs.emplace_back(x.data(), x.size());
    }
    halo_exchange_var_bufs(bufs);
  }

  // -------- Halo exchange (constraints / y) -------------------------------
  // Cstr-halo counterpart of halo_exchange_var_bufs. Same structure: contains
  // all the gather + NCCL send/recv logic; accessor overloads below are thin
  // wrappers that resolve one buffer per shard and delegate.
  // Requirements: bufs.size() == shards.size(); bufs[r] is shard r's owned +
  // halo tail (contiguous) with total_cstr_size elements.
  void halo_exchange_cstr_bufs(std::vector<raft::device_span<f_t>> const& bufs)
  {
    const int nb = static_cast<int>(shards.size());
    cuopt_expects(static_cast<int>(bufs.size()) == nb,
                  error_type_t::RuntimeError,
                  "halo_exchange_cstr_bufs: bufs.size() must equal shards.size()");

    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      auto y = bufs[r];
      for (int peer = 0; peer < nb; ++peer) {
        if (peer == r) continue;
        if (s.cstr_send_indices_d[peer].size() == 0) continue;
        thrust::gather(rmm::exec_policy_nosync(s.stream.view()),
                       s.cstr_send_indices_d[peer].begin(),
                       s.cstr_send_indices_d[peer].end(),
                       y.data(),
                       s.cstr_send_buf_d[peer].begin());
      }
    }

    CUOPT_NCCL_TRY(ncclGroupStart());
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      for (int peer = 0; peer < nb; ++peer) {
        if (peer == r) continue;
        CUOPT_NCCL_TRY(ncclSend(s.cstr_send_buf_d[peer].data(),
                                s.cstr_send_buf_d[peer].size(),
                                nccl_data_type<f_t>(),
                                peer,
                                s.comm.get(),
                                s.stream.view().value()));
      }
    }
    for (int r = 0; r < nb; ++r) {
      auto& s  = *shards[r];
      auto& rd = s.rank_data;
      raft::device_setter guard(s.device_id);
      auto y = bufs[r];
      for (int peer = 0; peer < nb; ++peer) {
        if (peer == r) continue;
        f_t* recv_ptr = y.data() + rd.owned_cstr_size + rd.cstr_recv_offsets[peer];
        CUOPT_NCCL_TRY(ncclRecv(recv_ptr,
                                static_cast<size_t>(rd.cstr_recv_counts[peer]),
                                nccl_data_type<f_t>(),
                                peer,
                                s.comm.get(),
                                s.stream.view().value()));
      }
    }
    CUOPT_NCCL_TRY(ncclGroupEnd());
  }

  // Wrapper: pdhg_solver_t accessor.
  // buf_access : pdhg_solver_t<i_t,f_t>& -> rmm::device_uvector<f_t>&
  template <typename BufAccess>
  void halo_exchange_cstr(BufAccess&& buf_access)
  {
    halo_exchange_cstr_shard([&](pdlp_shard_t<i_t, f_t>& s) -> rmm::device_uvector<f_t>& {
      return buf_access(s.sub_pdlp->pdhg_solver_);
    });
  }

  // Wrapper: pdlp_shard_t accessor. Resolves one uvector per shard into a
  // vector of spans, then delegates to halo_exchange_cstr_bufs.
  // buf_access : pdlp_shard_t<i_t,f_t>& -> rmm::device_uvector<f_t>&
  template <typename ShardBufAccess>
  void halo_exchange_cstr_shard(ShardBufAccess&& buf_access)
  {
    std::vector<raft::device_span<f_t>> bufs;
    bufs.reserve(shards.size());
    for (auto& s : shards) {
      raft::device_setter guard(s->device_id);
      auto& y = buf_access(*s);
      bufs.emplace_back(y.data(), y.size());
    }
    halo_exchange_cstr_bufs(bufs);
  }

  // -------- NCCL allreduce (sum, in place) --------------------------------
  // Core: per-shard in-place sum-allreduce on a single f_t scalar viewed by
  // scalars[r], wrapped in one NCCL group so it executes as a single
  // collective. After this returns, every shard's scalar holds the global sum.
  void allreduce_sum_inplace_bufs(std::vector<raft::device_scalar_view<f_t>> const& scalars)
  {
    const int nb = static_cast<int>(shards.size());
    cuopt_expects(static_cast<int>(scalars.size()) == nb,
                  error_type_t::RuntimeError,
                  "allreduce_sum_inplace_bufs: scalars.size() must equal shards.size()");
    if (nb == 0) return;

    CUOPT_NCCL_TRY(ncclGroupStart());
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      f_t* p = scalars[r].data_handle();
      CUOPT_NCCL_TRY(ncclAllReduce(p,
                                   p,
                                   /*count=*/1,
                                   nccl_data_type<f_t>(),
                                   ncclSum,
                                   s.comm.get(),
                                   s.stream.view().value()));
    }
    CUOPT_NCCL_TRY(ncclGroupEnd());
  }

  // Wrapper: pdlp_solver_t accessor for a single per-shard scalar.
  // ptr_access : pdlp_solver_t<i_t,f_t>& -> f_t*   (pointer to the scalar
  //              to reduce; one per shard)
  template <typename PtrAccess>
  void allreduce_sum_inplace(PtrAccess&& ptr_access)
  {
    std::vector<raft::device_scalar_view<f_t>> scalars;
    scalars.reserve(shards.size());
    for (auto& s : shards) {
      raft::device_setter guard(s->device_id);
      scalars.emplace_back(raft::make_device_scalar_view<f_t>(ptr_access(*s->sub_pdlp)));
    }
    allreduce_sum_inplace_bufs(scalars);
  }

  // -------- Distributed dot / L2 norm -------------------------------------
  // Computes the dot product of two vectors for each shard. Returns the global result in
  // out_scalars.
  void distributed_dot_bufs(std::vector<raft::device_span<f_t>> const& a_bufs,
                            std::vector<raft::device_span<f_t>> const& b_bufs,
                            std::vector<raft::device_scalar_view<f_t>> const& out_scalars)
  {
    const int nb = static_cast<int>(shards.size());
    cuopt_expects(static_cast<int>(a_bufs.size()) == nb && static_cast<int>(b_bufs.size()) == nb &&
                    static_cast<int>(out_scalars.size()) == nb,
                  error_type_t::RuntimeError,
                  "distributed_dot_bufs: a_bufs / b_bufs / out_scalars must "
                  "all have size == shards.size()");

    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      cuopt_expects(a_bufs[r].size() == b_bufs[r].size(),
                    error_type_t::RuntimeError,
                    "distributed_dot_bufs: a_bufs[r] and b_bufs[r] must have equal size");
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(s.handle.get_cublas_handle(),
                                                      static_cast<int>(a_bufs[r].size()),
                                                      a_bufs[r].data(),
                                                      1,
                                                      b_bufs[r].data(),
                                                      1,
                                                      out_scalars[r].data_handle(),
                                                      s.stream.view().value()));
    }

    allreduce_sum_inplace_bufs(out_scalars);
  }

  // Core L2 norm: writes sqrt(sum_r ||in_bufs[r]||_2^2) into every
  // *out_scalars[r].data_handle(). Delegates to distributed_dot_bufs(in, in,
  // out) then does a per-shard in-place sqrt on the resulting scalar.
  void distributed_l2_norm_bufs(std::vector<raft::device_span<f_t>> const& in_bufs,
                                std::vector<raft::device_scalar_view<f_t>> const& out_scalars)
  {
    distributed_dot_bufs(in_bufs, in_bufs, out_scalars);
    for (std::size_t r = 0; r < shards.size(); ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      cub::DeviceTransform::Transform(out_scalars[r].data_handle(),
                                      out_scalars[r].data_handle(),
                                      1,
                                      sqrt_inplace_op_t<f_t>{},
                                      s.stream.view().value());
    }
  }

  // Wrapper: accessor form. Resolves per-shard input / output / owned-length
  // then delegates to distributed_l2_norm_bufs.
  //   BufAccess  : pdlp_solver_t<i_t,f_t>& -> rmm::device_uvector<f_t>&
  //   OutAccess  : pdlp_solver_t<i_t,f_t>& -> f_t*   (single scalar)
  //   SizeAccess : pdlp_shard_t<i_t,f_t>&  -> i_t    (owned slice length)
  template <typename BufAccess, typename OutAccess, typename SizeAccess>
  void distributed_l2_norm(BufAccess&& buf_access, OutAccess&& out_access, SizeAccess&& size_access)
  {
    std::vector<raft::device_span<f_t>> in_bufs;
    std::vector<raft::device_scalar_view<f_t>> out_scalars;
    in_bufs.reserve(shards.size());
    out_scalars.reserve(shards.size());
    for (auto& s : shards) {
      raft::device_setter guard(s->device_id);
      auto& sub   = *s->sub_pdlp;
      auto& buf   = buf_access(sub);
      const i_t n = size_access(*s);
      in_bufs.emplace_back(buf.data(), static_cast<std::size_t>(n));
      out_scalars.emplace_back(raft::make_device_scalar_view<f_t>(out_access(sub)));
    }
    distributed_l2_norm_bufs(in_bufs, out_scalars);
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

}  // namespace cuopt::mathematical_optimization::pdlp
