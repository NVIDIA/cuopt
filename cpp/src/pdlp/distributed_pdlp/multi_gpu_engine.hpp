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
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <cub/device/device_transform.cuh>
#include <cuda/std/cmath>
#include <cuda/std/tuple>

#include <nccl.h>

#include <cmath>
#include <cstddef>
#include <memory>
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

/**
 * @brief Distributed PDLP terminology and ownership model.
 *
 * - Master: the top-level `pdlp_solver_t` that owns the global, developer-facing
 *   solve.
 * - Multi-GPU engine: the host-side coordinator for one distributed solve. It
 *   owns all shards and centralizes collective operations.
 * - Rank: Rank `r`, partition `r`, NCCL rank `r`, and `shards[r]` refer to the
 *   same worker on CUDA device `r`. A peer is any other rank participating in
 *   communication.
 * - Shard: the complete device-local worker for one rank. It owns that device's
 *   stream, RAFT handle, NCCL communicator, local optimization problem,
 *   subordinate `pdlp_solver_t`, rank data, and device staging buffers used by
 *   halo exchange.
 * - Rank data: the host-side partition description for one shard. It records
 *   owned variables/constraints, global/local index maps, local CSR matrices,
 *   and per-peer send/receive plans. It is produced during partitioning and
 *   moved into the corresponding shard.
 * - Owned entries: variables or constraints assigned to a rank by the
 *   partitioner. Each global entry has exactly one owner.
 * - Halo entries: local copies of entries owned by peers but needed
 *   by a shard's local SpMV.
 */
template <typename i_t, typename f_t>
struct multi_gpu_engine_t {
  // Constructs shards from rank_data. The global (unpartitioned) problem is
  // read straight from `mps`; each shard slices out the entries it owns.
  multi_gpu_engine_t(std::vector<rank_data_t<i_t, f_t>>&& rank_data,
                     io::mps_data_model_t<i_t, f_t> const& mps,
                     pdlp_solver_settings_t<i_t, f_t> const& sub_solver_settings);

  multi_gpu_engine_t(const multi_gpu_engine_t&)            = delete;
  multi_gpu_engine_t& operator=(const multi_gpu_engine_t&) = delete;

  // Invokes `fn` on every shard with the shard's device pre-set. `fn` may be
  //   (pdlp_shard_t<i_t,f_t>&)        or
  //   (pdlp_shard_t<i_t,f_t>&, int r) — the second overload also gets the shard's rank.
  template <typename Fn>
  void for_each_shard(Fn&& fn)
  {
    for (int r = 0; r < static_cast<int>(shards.size()); ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      // If the function is invocable with a pdlp_shard_t<i_t, f_t>& and an int, call it with the
      // shard and the rank.
      if constexpr (std::is_invocable_v<Fn&, pdlp_shard_t<i_t, f_t>&, int>) {
        fn(s, r);
        // If the function is invocable only with a pdlp_shard_t<i_t, f_t>&, call it with the shard.
      } else if constexpr (std::is_invocable_v<Fn&, pdlp_shard_t<i_t, f_t>&>) {
        fn(s);
      } else {
        // Otherwise, the function has an invalid signature.
        cuopt_assert(false, "for_each_shard: invalid function signature");
      }
    }
  }

  // Host-blocking barrier: waits until every shard stream has drained.
  void synchronize_shards();

  // Core: launches cub::DeviceTransform on every shard using per-shard
  // pre-resolved inputs / outputs / sizes.
  //   - in_tuples[r] is the tuple passed as cub input for shard r (any
  //     iterator-shaped types cub accepts: raw pointers, thrust iterators, ...)
  //   - outs[r]      is the output iterator for shard r
  //   - sizes[r]     is the element count for shard r
  // All three must have size == shards.size().
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
    for_each_shard([&](auto& s, int r) {
      cub::DeviceTransform::Transform(in_tuples[r], outs[r], sizes[r], op, s.stream.view().get());
    });
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
    // the runtime vector doesn't complain
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

    for_each_shard([&](auto& s) {
      auto& sub = *s.sub_pdlp;
      // apply() = Turns a tuple of accessors into a tuple of values.
      in_tuples.emplace_back(std::apply(
        [&sub](auto&... acc) { return cuda::std::make_tuple(acc(sub)...); }, in_accessors));
      outs.emplace_back(out(sub));
      sizes.emplace_back(sz(sub));
    });
    distributed_transform_bufs(in_tuples, outs, sizes, op);
  }

  // --- 2) convenience: single input accessor (delegates) ---
  // Allows to use distributed_transform on single input without having to do a std::make_tuple(in)
  template <typename InAccess, typename OutAccess, typename SizeAccess, typename Op>
  void distributed_transform(InAccess in, OutAccess out, SizeAccess sz, Op op)
  {
    distributed_transform(std::make_tuple(in), out, sz, op);
  }

  // -------- Halo exchange (owner -> halo) ---------------------------------
  // {var/cstr}-agnostic core:
  //   Step 1: thrust::gather per-peer outgoing values into staging buffers.
  //   Step 2: one NCCL group with matched ncclSend / ncclRecv across all
  //           (rank, peer) pairs, receiving into each shard's halo tail.
  void halo_exchange_bufs_impl(
    std::vector<raft::device_span<f_t>> const& bufs,
    std::vector<typename pdlp_shard_t<i_t, f_t>::halo_axis_t> const& axes);

  // -------- Halo exchange (variables / x) ---------------------------------
  void halo_exchange_var_bufs(std::vector<raft::device_span<f_t>> const& bufs);

  // Overload: accept the owning device_uvector directly (rmm doesn't provide
  // an implicit conversion to raft::device_span)
  void halo_exchange_var_bufs(std::vector<rmm::device_uvector<f_t>>& bufs);

  // Wrapper: pdlp_solver_t accessor. Resolves one uvector per shard into a
  // vector of spans, then delegates to halo_exchange_var_bufs.
  //   buf_access : pdlp_solver_t<i_t,f_t>& -> rmm::device_uvector<f_t>&
  template <typename BufAccess>
  void halo_exchange_var(BufAccess&& buf_access)
  {
    std::vector<raft::device_span<f_t>> bufs;
    bufs.reserve(shards.size());
    for_each_shard([&](pdlp_shard_t<i_t, f_t>& s) {
      auto& x = buf_access(*s.sub_pdlp);
      bufs.emplace_back(x.data(), x.size());
    });
    halo_exchange_var_bufs(bufs);
  }

  // -------- Halo exchange (constraints / y) -------------------------------
  void halo_exchange_cstr_bufs(std::vector<raft::device_span<f_t>> const& bufs);

  // Overload: same rationale as halo_exchange_var_bufs above.
  void halo_exchange_cstr_bufs(std::vector<rmm::device_uvector<f_t>>& bufs);

  // Wrapper: pdlp_solver_t accessor. Resolves one uvector per shard into a
  // vector of spans, then delegates to halo_exchange_cstr_bufs.
  //   buf_access : pdlp_solver_t<i_t,f_t>& -> rmm::device_uvector<f_t>&
  template <typename BufAccess>
  void halo_exchange_cstr(BufAccess&& buf_access)
  {
    std::vector<raft::device_span<f_t>> bufs;
    bufs.reserve(shards.size());
    for_each_shard([&](pdlp_shard_t<i_t, f_t>& s) {
      auto& y = buf_access(*s.sub_pdlp);
      bufs.emplace_back(y.data(), y.size());
    });
    halo_exchange_cstr_bufs(bufs);
  }

  // -------- Gather owned slices to master ---------------------------------
  // {var/cstr}-agnostic core: scatters each shard's owned slice into
  // master_buf using local_to_globals[r] as the destination index list.
  //  local_to_globals[r] is
  // the axis-specific rank_data.local_to_global_{var/cstr} for shard r.
  void gather_owned_to_master_bufs_impl(
    std::vector<raft::device_span<f_t const>> const& shard_owned,
    raft::device_span<f_t> master_buf,
    std::vector<std::vector<i_t>> const& local_to_globals);

  // -------- Gather (variables / x) ----------------------------------------
  void gather_owned_var_to_master_bufs(std::vector<raft::device_span<f_t const>> const& shard_owned,
                                       raft::device_span<f_t> master_buf);

  // Wrapper: pdlp_solver_t accessor. Slices each shard's owned prefix and
  // then delegates to gather_owned_var_to_master_bufs.
  //   buf_access : pdlp_solver_t<i_t,f_t>& -> rmm::device_uvector<f_t>&
  template <typename BufAccess>
  void gather_owned_var_to_master(BufAccess&& buf_access)
  {
    cuopt_assert(master_pdlp_ != nullptr, "gather_owned_var_to_master requires set_master(...)");
    std::vector<raft::device_span<f_t const>> shard_bufs;
    shard_bufs.reserve(shards.size());
    for_each_shard([&](pdlp_shard_t<i_t, f_t>& s) {
      auto& x = buf_access(*s.sub_pdlp);
      shard_bufs.emplace_back(x.data(), static_cast<std::size_t>(s.rank_data.owned_var_size));
    });
    auto& m = buf_access(*master_pdlp_);
    gather_owned_var_to_master_bufs(shard_bufs, raft::device_span<f_t>{m.data(), m.size()});
  }

  // -------- Gather (constraints / y) --------------------------------------
  void gather_owned_cstr_to_master_bufs(
    std::vector<raft::device_span<f_t const>> const& shard_owned,
    raft::device_span<f_t> master_buf);

  // Wrapper: same rationale as gather_owned_var_to_master.
  template <typename BufAccess>
  void gather_owned_cstr_to_master(BufAccess&& buf_access)
  {
    cuopt_assert(master_pdlp_ != nullptr, "gather_owned_cstr_to_master requires set_master(...)");
    std::vector<raft::device_span<f_t const>> shard_bufs;
    shard_bufs.reserve(shards.size());
    for_each_shard([&](pdlp_shard_t<i_t, f_t>& s) {
      auto& x = buf_access(*s.sub_pdlp);
      shard_bufs.emplace_back(x.data(), static_cast<std::size_t>(s.rank_data.owned_cstr_size));
    });
    auto& m = buf_access(*master_pdlp_);
    gather_owned_cstr_to_master_bufs(shard_bufs, raft::device_span<f_t>{m.data(), m.size()});
  }

  // -------- NCCL allreduce (sum, in place) --------------------------------
  // Core: per-shard in-place sum-allreduce on a single f_t scalar viewed by
  // scalars[r], wrapped in one NCCL group so it executes as a single
  // collective. After this returns, every shard's scalar holds the global sum.
  void allreduce_sum_inplace_bufs(std::vector<raft::device_scalar_view<f_t>> const& scalars);

  // Wrapper: pdlp_solver_t accessor for a single per-shard scalar.
  // ptr_access : pdlp_solver_t<i_t,f_t>& -> f_t*   (pointer to the scalar
  //              to reduce; one per shard)
  template <typename PtrAccess>
  void allreduce_sum_inplace(PtrAccess&& ptr_access)
  {
    std::vector<raft::device_scalar_view<f_t>> scalars;
    scalars.reserve(shards.size());
    for_each_shard([&](auto& s) {
      scalars.emplace_back(raft::make_device_scalar_view<f_t>(ptr_access(*s.sub_pdlp)));
    });
    allreduce_sum_inplace_bufs(scalars);
  }

  // Core: same as allreduce_sum_inplace_bufs, plus after the collective the
  // value is D2D-copied from shard 0 into master_dst. On master's stream.
  void allreduce_sum_inplace_to_master_buf(
    std::vector<raft::device_scalar_view<f_t>> const& shard_scalars,
    raft::device_scalar_view<f_t> master_dst);

  // Wrapper: applies the ptr_access lambda to each shard's sub_pdlp to build
  // the per-shard scalar views and to master_pdlp_ to obtain the master
  // destination, then delegates to allreduce_sum_inplace_to_master_buf.
  template <typename PtrAccess>
  void allreduce_sum_inplace_to_master(PtrAccess&& ptr_access)
  {
    cuopt_assert(master_pdlp_ != nullptr,
                 "allreduce_sum_inplace_to_master requires set_master(...) to have been called");
    std::vector<raft::device_scalar_view<f_t>> shard_scalars;
    shard_scalars.reserve(shards.size());
    for_each_shard([&](auto& s) {
      shard_scalars.emplace_back(raft::make_device_scalar_view<f_t>(ptr_access(*s.sub_pdlp)));
    });
    allreduce_sum_inplace_to_master_buf(
      shard_scalars, raft::make_device_scalar_view<f_t>(ptr_access(*master_pdlp_)));
  }

  // -------- Set a host scalar on master AND every shard -------------------
  // Writes a host-side value to master and every shard through `ptr_access`.
  template <typename PtrAccess>
  void set_scalar_on_master_and_shards(f_t value, PtrAccess&& ptr_access)
  {
    cuopt_assert(master_pdlp_ != nullptr,
                 "set_scalar_on_master_and_shards requires set_master(...) to have been called");
    auto master_stream = master_pdlp_->get_handle_ptr()->get_stream();
    raft::copy(ptr_access(*master_pdlp_), &value, 1, master_stream);
    for_each_shard([&](auto& shard) {
      raft::copy(ptr_access(*shard.sub_pdlp), &value, 1, shard.stream.view());
    });
    master_pdlp_->get_handle_ptr()->sync_stream(master_stream);
    synchronize_shards();
  }

  // -------- Distributed dot / L2 norm -------------------------------------
  // Computes the dot product of two vectors for each shard. Returns the global result in
  // out_scalars.
  void distributed_dot_bufs(std::vector<raft::device_span<f_t>> const& a_bufs,
                            std::vector<raft::device_span<f_t>> const& b_bufs,
                            std::vector<raft::device_scalar_view<f_t>> const& out_scalars);

  // Overload: accept owning device_scalar outputs directly. Wraps each into a
  // scalar_view and delegates to the span-based core. Convenience for the typical
  // case where per-shard outputs are rmm::device_scalar<f_t>.
  void distributed_dot_bufs(std::vector<raft::device_span<f_t>> const& a_bufs,
                            std::vector<raft::device_span<f_t>> const& b_bufs,
                            std::vector<rmm::device_scalar<f_t>>& out_scalars);

  // Core L2 norm: writes sqrt(sum_r ||in_bufs[r]||_2^2) into every
  // *out_scalars[r].data_handle(). Delegates to distributed_dot_bufs(in, in,
  // out) then does a per-shard in-place sqrt on the resulting scalar.
  void distributed_l2_norm_bufs(std::vector<raft::device_span<f_t>> const& in_bufs,
                                std::vector<raft::device_scalar_view<f_t>> const& out_scalars);

  // Overload: same rationale as distributed_dot_bufs above. Allows to use rmm::device_scalar<f_t>
  // directly.
  void distributed_l2_norm_bufs(std::vector<raft::device_span<f_t>> const& in_bufs,
                                std::vector<rmm::device_scalar<f_t>>& out_scalars);

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
    for_each_shard([&](auto& s) {
      auto& sub   = *s.sub_pdlp;
      auto& buf   = buf_access(sub);
      const i_t n = size_access(s);
      in_bufs.emplace_back(buf.data(), static_cast<std::size_t>(n));
      out_scalars.emplace_back(raft::make_device_scalar_view<f_t>(out_access(sub)));
    });
    distributed_l2_norm_bufs(in_bufs, out_scalars);
  }

  // Core: same as distributed_l2_norm_bufs, plus after the collective the
  // value is D2D-copied from shard 0 into master_dst. On master's stream.
  // Mirrors the allreduce_sum_inplace_to_master_buf pattern.
  void distributed_l2_norm_to_master_buf(
    std::vector<raft::device_span<f_t>> const& in_bufs,
    std::vector<raft::device_scalar_view<f_t>> const& shard_out,
    raft::device_scalar_view<f_t> master_dst);

  // Wrapper: applies out_access to master_pdlp_ to obtain the master
  // destination, then delegates to distributed_l2_norm_to_master_buf.
  // out_access is the one accessor for shards and master
  //   BufAccess  : pdlp_solver_t<i_t,f_t>& -> rmm::device_uvector<f_t>&
  //   OutAccess  : pdlp_solver_t<i_t,f_t>& -> f_t*   (single scalar)
  //   SizeAccess : pdlp_shard_t<i_t,f_t>&  -> i_t    (owned slice length)
  template <typename BufAccess, typename OutAccess, typename SizeAccess>
  void distributed_l2_norm_to_master(BufAccess&& buf_access,
                                     OutAccess&& out_access,
                                     SizeAccess&& size_access)
  {
    cuopt_assert(master_pdlp_ != nullptr,
                 "distributed_l2_norm_to_master requires set_master(...) to have been called");
    std::vector<raft::device_span<f_t>> in_bufs;
    std::vector<raft::device_scalar_view<f_t>> shard_out;
    in_bufs.reserve(shards.size());
    shard_out.reserve(shards.size());
    for_each_shard([&](auto& s) {
      auto& sub   = *s.sub_pdlp;
      auto& buf   = buf_access(sub);
      const i_t n = size_access(s);
      in_bufs.emplace_back(buf.data(), static_cast<std::size_t>(n));
      shard_out.emplace_back(raft::make_device_scalar_view<f_t>(out_access(sub)));
    });
    distributed_l2_norm_to_master_buf(
      in_bufs, shard_out, raft::make_device_scalar_view<f_t>(out_access(*master_pdlp_)));
  }

  // -------- High-level: A @ x and A_T @ y ---------------------------------
  // Distributed counterpart to pdhg_solver_t::compute_A_x() / compute_At_y().
  void distributed_compute_A_x();
  void distributed_compute_At_y();

  // Distributed A^T @ in on caller-owned scratch. Refreshes the halo of `in_bufs`
  // (cstr axis, since the input is cstr-shaped), then dispatches each shard's
  // local spmv_At_into that reads from in_descs[r] and writes into out_descs[r].
  void distributed_spmv_At(std::vector<rmm::device_uvector<f_t>>& in_bufs,
                           std::vector<cusparse_dn_vec_uptr>& in_descs,
                           std::vector<cusparse_dn_vec_uptr>& out_descs);

  // Distributed A @ in on caller-owned scratch. Refreshes the halo of `in_bufs`
  // (var axis, since the input is var-shaped), then dispatches each shard's
  // local spmv_A_into. Caller owns / sizes the descriptor vectors as above
  // (in_descs to var_total, out_descs to cstr_total).
  void distributed_spmv_A(std::vector<rmm::device_uvector<f_t>>& in_bufs,
                          std::vector<cusparse_dn_vec_uptr>& in_descs,
                          std::vector<cusparse_dn_vec_uptr>& out_descs);

  // -------- High-level algorithms (defined in distributed_algorithms.cu) ---
  // Refreshes the halo copies of the cumulative variable + constraint scalings on
  // every shard. Used by the matrix-scaling passes (Ruiz, Pock-Chambolle)
  void refresh_halo_cummulative_scalings();

  // Global bound/objective rescaling: allreduce the owned partial squared norms
  // of the constraint bounds and (weighted) objective, then apply the identical
  // scalar on every shard.
  void distributed_bound_objective_rescaling(f_t c_scaling_weight);

  // Distributed Ruiz inf-scaling (num_iter passes). Each shard computes both its
  // owned-row and owned-column inf-norms locally then broadcasts the cumulative scalings to all
  // shards.
  void distributed_ruiz_inf_scaling(int num_iter, i_t n_global_vars);

  // Distributed Pock-Chambolle scaling (one pass), mirroring the single-GPU
  // pock_chambolle_scaling.
  void distributed_pock_chambolle_scaling(f_t alpha, i_t n_global_vars);

  // Full distributed scaling entry point. Mirrors what scale_problem() does in
  // single-GPU by orchestrating:
  //   - Ruiz inf-scaling -> populates cumulative row/col scalings
  //   - Pock-Chambolle scaling -> same
  //   - per-shard apply_cummulative_scaling_to_problem()
  //   - global bound/objective rescaling via distributed_bound_objective_rescaling
  void distributed_scaling(pdlp_hyper_params_t const& hyper_params,
                           i_t n_global_vars,
                           bool inside_mip);

  // Distributed sigma_max(A)^2 via power iteration (used to seed the initial
  // step size). Returns the square of the largest singular value of the scaled
  // constraint matrix.
  f_t distributed_max_singular_value_squared(i_t n_global_cstrs,
                                             int max_iterations = 5000,
                                             f_t tolerance      = 1e-4);

  // Distributed counterpart of pdlp_solver_t::compute_initial_step_size.
  void distributed_compute_initial_step_size(pdlp_hyper_params_t const& hyper_params,
                                             i_t n_global_cstrs,
                                             f_t scaling_factor,
                                             int max_iterations,
                                             f_t tolerance);

  // Distributed counterpart of pdlp_solver_t::compute_initial_primal_weight.
  // Writes primal_weight = best_primal_weight = 1 onto master + every shard,
  // mirroring the Stable3-shaped short-circuit
  // (!initial_primal_weight_combined_bounds && bound_objective_rescaling).
  void distributed_compute_initial_primal_weight(pdlp_hyper_params_t const& hyper_params);

  // Gather the global potential_next primal/dual solutions and the reduced cost
  // onto the master from the owned slices distributed across shards.
  void gather_potential_next_solutions_to_master();

  // Engine-level stream for fork/join orchestration (master side).
  rmm::cuda_stream stream;

  // Shards stored by unique_ptr because pdlp_shard_t is immovable
  // (owns device-affine resources: handle, NCCL comm, RMM buffers).
  std::vector<std::unique_ptr<pdlp_shard_t<i_t, f_t>>> shards;

  // Cached per-shard partition metadata, populated once at construction.
  // Consumed by gather_owned_*_to_master_bufs; caching avoids copying the
  // sizeable local_to_global_* host vectors on every termination check.
  //   local_to_global_{vars,cstrs}_[r] == shards[r]->rank_data.local_to_global_{var,cstr}
  std::vector<std::vector<i_t>> local_to_global_vars_;
  std::vector<std::vector<i_t>> local_to_global_cstrs_;

  // Non-owning back-pointer to the master pdlp_solver_t.
  pdlp_solver_t<i_t, f_t>* master_pdlp_ = nullptr;
  void set_master(pdlp_solver_t<i_t, f_t>* m);

  // ===== Cross-stream synchronization events =====
  // two different events
  // graph_*_event_ are used inside graph capture
  // sync_*_event_ are used when sync is needed outside of graph
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
