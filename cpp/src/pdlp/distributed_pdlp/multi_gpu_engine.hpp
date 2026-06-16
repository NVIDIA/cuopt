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
#include <vector>

namespace cuopt::linear_programming::detail {

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
      auto& x = buf_access(s);
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
      auto& y = buf_access(s);
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

  // -------- Broadcast owned constraint (row) scaling into halo ------------
  // The cumulative constraint-matrix (row) scaling is computed only on owned
  // rows; push each owner's values into the peers' halo copies.
  void broadcast_constraint_scaling_to_halo()
  {
    halo_exchange_cstr_shard([](pdlp_shard_t<i_t, f_t>& s) -> rmm::device_uvector<f_t>& {
      return s.sub_pdlp->get_initial_scaling_strategy().get_cummulative_constraint_matrix_scaling();
    });
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

  // -------- Distributed bound / objective rescaling -----------------------
  void distributed_bound_objective_rescaling(f_t c_scaling_weight)
  {
    const int nb = static_cast<int>(shards.size());

    // Per-shard packed partial squared norms: [0] = bound (rhs) sq, [1] = obj sq.
    std::vector<rmm::device_uvector<f_t>> sq;
    sq.reserve(nb);

    // 1) per-shard partial squared norms over OWNED entries only 
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      sq.emplace_back(2, s.stream.view());

      const auto& scaled     = s.sub_pdlp->get_initial_scaling_strategy().get_scaled_op_problem();
      const int n_owned_cstr = static_cast<int>(s.rank_data.owned_cstr_size);
      const int n_owned_var  = static_cast<int>(s.rank_data.owned_var_size);

      // Squared-norm contribution of each constraint's [lower, upper] bound pair
      // (mirrors rhs_sum_of_squares_t). The lower bound is the reduce input; the
      // matching upper bound is fetched by index inside the op.
      const f_t* upper = scaled.constraint_upper_bounds.data();
      auto bound_op    = [upper] __device__(f_t lower, i_t i) {
        const f_t u = upper[i];
        f_t sum     = f_t(0);
        if (isfinite(lower) && (lower != u)) sum += lower * lower;
        if (isfinite(u)) sum += u * u;
        return sum;
      };
      raft::linalg::reduce<true, true, f_t, f_t, i_t>(sq[r].data() + 0,
                                                      scaled.constraint_lower_bounds.data(),
                                                      n_owned_cstr,
                                                      1,
                                                      f_t(0),
                                                      s.stream.view(),
                                                      false,
                                                      bound_op,
                                                      raft::Sum<f_t>());

      // Weighted sum of squares of the objective coefficients.
      auto obj_op = [c_scaling_weight] __device__(f_t v, i_t) { return v * v * c_scaling_weight; };
      raft::linalg::reduce<true, true, f_t, f_t, i_t>(sq[r].data() + 1,
                                                      scaled.objective_coefficients.data(),
                                                      n_owned_var,
                                                      1,
                                                      f_t(0),
                                                      s.stream.view(),
                                                      false,
                                                      obj_op,
                                                      raft::Sum<f_t>());
    }

    // 2) NCCL allreduce SUM (both scalars at once) -> every shard holds the
    //    global squared norms.
    ncclGroupStart();
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      ncclAllReduce(
        sq[r].data(), sq[r].data(), 2, ncclFloat64, ncclSum, s.comm.get(), s.stream.view().value());
    }
    ncclGroupEnd();

    // 3) derive the identical scalars and apply on every shard.
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      f_t h_sq[2] = {f_t(0), f_t(0)};
      raft::copy(h_sq, sq[r].data(), 2, s.stream.view());
      s.stream.synchronize();
      const f_t bound_rescaling     = f_t(1) / (std::sqrt(h_sq[0]) + f_t(1));
      const f_t objective_rescaling = f_t(1) / (std::sqrt(h_sq[1]) + f_t(1));
      s.sub_pdlp->get_initial_scaling_strategy().apply_distributed_bound_objective_rescaling(
        bound_rescaling, objective_rescaling);
    }
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      s.stream.synchronize();
    }
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
  // Distributed counterpart to pdhg_solver_t::compute_A_x()
  // We don't use distributed_spmv_A() because we are using SpMVOp rather than SpMV
  void distributed_compute_A_x()
  {
    halo_exchange_var(
      [](auto& pdhg) -> rmm::device_uvector<f_t>& { return pdhg.get_reflected_primal(); });
    for_each_shard([](auto& shard) { shard.sub_pdlp->pdhg_solver_.spmvop_A_x(); });
  }

  // Distributed counterpart to pdhg_solver_t::compute_At_y()
  // We don't use distributed_spmv_At() because we are using SpMVOp rather than SpMV
  void distributed_compute_At_y()
  {
    halo_exchange_cstr(
      [](auto& pdhg) -> rmm::device_uvector<f_t>& { return pdhg.get_dual_solution(); });
    for_each_shard([](auto& shard) { shard.sub_pdlp->pdhg_solver_.spmvop_At_y(); });
  }

  // -------- Distributed Ruiz inf-scaling -----------------------------------
  std::vector<rmm::device_uvector<f_t>> alloc_global_var_scratch(i_t n_global_vars)
  {
    const int nb = static_cast<int>(shards.size());
    std::vector<rmm::device_uvector<f_t>> global_var_buf;
    global_var_buf.reserve(nb);
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      global_var_buf.emplace_back(static_cast<std::size_t>(n_global_vars), s.stream.view());
    }
    return global_var_buf;
  }

  void reduce_iteration_variable_scaling_across_shards(
    ncclRedOp_t op,
    i_t n_global_vars,
    std::vector<rmm::device_uvector<f_t>>& global_var_buf)
  {
    const int nb = static_cast<int>(shards.size());

    // Zero global buffers, then scatter each shard's local values into their
    // global column indices.
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      RAFT_CUDA_TRY(cudaMemsetAsync(global_var_buf[r].data(),
                                    0,
                                    sizeof(f_t) * static_cast<std::size_t>(n_global_vars),
                                    s.stream.view().value()));
      auto& iter_var_scaling =
        s.sub_pdlp->get_initial_scaling_strategy().get_iteration_variable_scaling();
      if (s.rank_data.total_var_size > 0) {
        thrust::scatter(rmm::exec_policy_nosync(s.stream.view()),
                        iter_var_scaling.begin(),
                        iter_var_scaling.begin() + s.rank_data.total_var_size,
                        s.local_to_global_var_d.begin(),
                        global_var_buf[r].begin());
      }
    }

    ncclGroupStart();
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      ncclAllReduce(global_var_buf[r].data(),
                    global_var_buf[r].data(),
                    static_cast<size_t>(n_global_vars),
                    ncclFloat64,
                    op,
                    s.comm.get(),
                    s.stream.view().value());
    }
    ncclGroupEnd();

    // Gather the global per-column value back into each shard's local iter vector.
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      auto& iter_var_scaling =
        s.sub_pdlp->get_initial_scaling_strategy().get_iteration_variable_scaling();
      if (s.rank_data.total_var_size > 0) {
        thrust::gather(rmm::exec_policy_nosync(s.stream.view()),
                       s.local_to_global_var_d.begin(),
                       s.local_to_global_var_d.begin() + s.rank_data.total_var_size,
                       global_var_buf[r].begin(),
                       iter_var_scaling.begin());
      }
    }
  }

  void distributed_ruiz_inf_scaling(int num_iter, i_t n_global_vars)
  {
    if (num_iter <= 0 || n_global_vars <= 0) return;
    raft::common::nvtx::range scope("distributed_ruiz_inf_scaling");

    auto global_var_buf = alloc_global_var_scratch(n_global_vars);

    for (int it = 0; it < num_iter; ++it) {
      // 1) per-shard local kernel: writes iteration_variable_scaling (per-column
      //    inf-norm partial) and iteration_constraint_matrix_scaling (row, complete).
      for_each_shard([](auto& shard) {
        shard.sub_pdlp->get_initial_scaling_strategy().ruiz_iter_compute_local_iteration_vectors();
      });

      // 2) cross-shard column inf-norm reduction (MAX).
      reduce_iteration_variable_scaling_across_shards(ncclMax, n_global_vars, global_var_buf);

      // 3) per-shard fold into cumulative + reset iter vectors.
      for_each_shard([](auto& shard) {
        shard.sub_pdlp->get_initial_scaling_strategy().ruiz_iter_apply_cumulative_update();
      });
    }

    // Make sure per-shard cumulative writes are observable on subsequent
    // calls (e.g., the next distributed_max_singular_value).
    for_each_shard([](auto& shard) { shard.stream.synchronize(); });
  }

  // Distributed Pock-Chambolle: one pass, mirroring single-GPU
  // pock_chambolle_scaling but with the per-column sum-of-powers reduced across
  // shards (SUM) between the local kernels and the cumulative fold. Each shard
  // stores its owned rows complete, so the row half is computed locally (then
  // broadcast to halo copies); only the column half is split across shards and
  // needs the reduction. Runs after the distributed Ruiz pass, matching the
  // single-GPU order (Ruiz then Pock-Chambolle).
  void distributed_pock_chambolle_scaling(f_t alpha, i_t n_global_vars)
  {
    if (n_global_vars <= 0) return;
    raft::common::nvtx::range scope("distributed_pock_chambolle_scaling");

    auto global_var_buf = alloc_global_var_scratch(n_global_vars);

    // 1) per-shard local kernels: row sum (complete) + column sum (partial).
    for_each_shard([alpha](auto& shard) {
      shard.sub_pdlp->get_initial_scaling_strategy().pock_chambolle_compute_local_iteration_vectors(
        alpha);
    });

    // 2) cross-shard column sum-of-powers reduction (SUM).
    reduce_iteration_variable_scaling_across_shards(ncclSum, n_global_vars, global_var_buf);

    // 3) per-shard fold into cumulative (cumulative /= sqrt(iteration)).
    for_each_shard([](auto& shard) {
      shard.sub_pdlp->get_initial_scaling_strategy().pock_chambolle_apply_cumulative_update();
    });

    for_each_shard([](auto& shard) { shard.stream.synchronize(); });
  }

  // -------- Distributed σ_max(A) via power iteration ----------------------
  f_t distributed_max_singular_value(i_t n_global_cstrs,
                                     int max_iterations = 5000,
                                     f_t tolerance      = 1e-4)
  {
    raft::common::nvtx::range scope("distributed_max_singular_value");

    const int nb = static_cast<int>(shards.size());

    // Generate the GLOBAL z[] sequence in cstr-index order from a fresh
    // mt19937(1), once per call. It's m doubles regardless of N (cheap).
    // Each shard then keeps only z[global_idx_for_owned_local_i].
    std::vector<f_t> h_global_z(static_cast<std::size_t>(n_global_cstrs));
    {
      std::mt19937 gen(1);
      std::normal_distribution<f_t> dist(f_t(0.0), f_t(1.0));
      for (i_t i = 0; i < n_global_cstrs; ++i) {
        h_global_z[i] = dist(gen);
      }
    }

    // Per-shard scratch lives on each shard's device. We use total (owned +
    // halo) sizes for q/z/atq because they're SpMV inputs that need halo
    // space. Norms / dot are scalars.
    // We use size-1 rmm::device_uvector instead of rmm::device_scalar for the
    // per-shard scratch scalars: nvcc + libcudacxx <cuda/basic_any> fail the
    // copy_constructible concept check when device_scalar<T> appears in a
    // std::vector (the check transitively touches rmm::cuda_stream, which is
    // non-copyable). device_uvector<T> avoids that path.
    std::vector<rmm::device_uvector<f_t>> q;
    std::vector<rmm::device_uvector<f_t>> z;
    std::vector<rmm::device_uvector<f_t>> atq;
    std::vector<rmm::device_uvector<f_t>> sigma_sq;
    std::vector<rmm::device_uvector<f_t>> norm_q;
    std::vector<rmm::device_uvector<f_t>> residual_norm;
    std::vector<cusparseDnVecDescr_t> z_dn(nb, nullptr);
    std::vector<cusparseDnVecDescr_t> atq_dn(nb, nullptr);
    q.reserve(nb);
    z.reserve(nb);
    atq.reserve(nb);
    sigma_sq.reserve(nb);
    norm_q.reserve(nb);
    residual_norm.reserve(nb);

    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      const i_t cstr_total = s.rank_data.total_cstr_size;
      const i_t var_total  = s.rank_data.total_var_size;
      q.emplace_back(static_cast<std::size_t>(cstr_total), s.stream.view());
      z.emplace_back(static_cast<std::size_t>(cstr_total), s.stream.view());
      atq.emplace_back(static_cast<std::size_t>(var_total), s.stream.view());
      sigma_sq.emplace_back(std::size_t{1}, s.stream.view());
      norm_q.emplace_back(std::size_t{1}, s.stream.view());
      residual_norm.emplace_back(std::size_t{1}, s.stream.view());
      RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednvec(
        &z_dn[r], static_cast<int64_t>(cstr_total), z.back().data()));
      RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednvec(
        &atq_dn[r], static_cast<int64_t>(var_total), atq.back().data()));

      std::vector<f_t> h_owned_z(static_cast<std::size_t>(s.rank_data.owned_cstr_size));
      for (i_t i = 0; i < s.rank_data.owned_cstr_size; ++i) {
        const i_t g  = s.rank_data.local_to_global_cstr[i];
        h_owned_z[i] = h_global_z[g];
      }
      if (s.rank_data.owned_cstr_size > 0) {
        RAFT_CUDA_TRY(
          cudaMemcpyAsync(z.back().data(),
                          h_owned_z.data(),
                          sizeof(f_t) * static_cast<std::size_t>(s.rank_data.owned_cstr_size),
                          cudaMemcpyHostToDevice,
                          s.stream.view().value()));
      }
      if (cstr_total > s.rank_data.owned_cstr_size) {
        RAFT_CUDA_TRY(cudaMemsetAsync(
          z.back().data() + s.rank_data.owned_cstr_size,
          0,
          sizeof(f_t) * static_cast<std::size_t>(cstr_total - s.rank_data.owned_cstr_size),
          s.stream.view().value()));
      }
      // Sync to ensure h_owned_z stays valid through the H2D copy (it goes
      // out of scope at end of this iteration of the per-shard loop).
      s.stream.synchronize();
    }

    // Local halo-exchange helpers that work directly on per-shard external
    // buffers (the engine's halo_exchange_var/cstr expect accessors that
    // resolve through pdhg_solver_t, which doesn't see our scratch).
    auto halo_exchange_cstr_bufs = [&](std::vector<rmm::device_uvector<f_t>>& bufs) {
      for (int r = 0; r < nb; ++r) {
        auto& s = *shards[r];
        raft::device_setter guard(s.device_id);
        auto& y = bufs[r];
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
        auto& y = bufs[r];
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
    };
    auto halo_exchange_var_bufs = [&](std::vector<rmm::device_uvector<f_t>>& bufs) {
      for (int r = 0; r < nb; ++r) {
        auto& s = *shards[r];
        raft::device_setter guard(s.device_id);
        auto& x = bufs[r];
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
        auto& x = bufs[r];
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
    };

    // Per-shard partial reductions over the OWNED cstr slice + NCCL allreduce.
    // For norm: out := sqrt(Σ_r ||bufs[r][0:owned_cstr]||²).
    // For dot : out := Σ_r <a[r][0:owned_cstr], b[r][0:owned_cstr]>.
    auto distributed_norm_owned_cstr = [&](std::vector<rmm::device_uvector<f_t>>& bufs,
                                           std::vector<rmm::device_uvector<f_t>>& out) {
      for (int r = 0; r < nb; ++r) {
        auto& s = *shards[r];
        raft::device_setter guard(s.device_id);
        const i_t n_owned = s.rank_data.owned_cstr_size;
        RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(s.handle.get_cublas_handle(),
                                                        static_cast<int>(n_owned),
                                                        bufs[r].data(),
                                                        1,
                                                        bufs[r].data(),
                                                        1,
                                                        out[r].data(),
                                                        s.stream.view().value()));
      }
      ncclGroupStart();
      for (int r = 0; r < nb; ++r) {
        auto& s = *shards[r];
        raft::device_setter guard(s.device_id);
        ncclAllReduce(out[r].data(),
                      out[r].data(),
                      1,
                      ncclFloat64,
                      ncclSum,
                      s.comm.get(),
                      s.stream.view().value());
      }
      ncclGroupEnd();
      for (int r = 0; r < nb; ++r) {
        auto& s = *shards[r];
        raft::device_setter guard(s.device_id);
        cub::DeviceTransform::Transform(
          out[r].data(), out[r].data(), 1, sqrt_inplace_op_t<f_t>{}, s.stream.view().value());
      }
    };
    auto distributed_dot_owned_cstr = [&](std::vector<rmm::device_uvector<f_t>>& a,
                                          std::vector<rmm::device_uvector<f_t>>& b,
                                          std::vector<rmm::device_uvector<f_t>>& out) {
      for (int r = 0; r < nb; ++r) {
        auto& s = *shards[r];
        raft::device_setter guard(s.device_id);
        const i_t n_owned = s.rank_data.owned_cstr_size;
        RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(s.handle.get_cublas_handle(),
                                                        static_cast<int>(n_owned),
                                                        a[r].data(),
                                                        1,
                                                        b[r].data(),
                                                        1,
                                                        out[r].data(),
                                                        s.stream.view().value()));
      }
      ncclGroupStart();
      for (int r = 0; r < nb; ++r) {
        auto& s = *shards[r];
        raft::device_setter guard(s.device_id);
        ncclAllReduce(out[r].data(),
                      out[r].data(),
                      1,
                      ncclFloat64,
                      ncclSum,
                      s.comm.get(),
                      s.stream.view().value());
      }
      ncclGroupEnd();
    };

    // ===== Power iteration =====
    // Mirrors single-GPU compute_initial_step_size: z is the carried iterate
    // (A Aᵀ q each step); at the top of each iteration q := z then q is
    // normalized; the residual z − σ²q is written back into q only to drive
    // the convergence check (next iteration's q := z discards it).
    for (int it = 0; it < max_iterations; ++it) {
      // q := z on the owned slice (the carried iterate), then normalize.
      for (int r = 0; r < nb; ++r) {
        auto& s = *shards[r];
        raft::device_setter guard(s.device_id);
        const i_t n_owned = s.rank_data.owned_cstr_size;
        raft::copy(q[r].data(), z[r].data(), n_owned, s.stream.view());
      }

      // ||q||₂ over the global OWNED cstr slice (one allreduce-sum + sqrt).
      distributed_norm_owned_cstr(q, norm_q);

      // q /= ||q||₂ on owned slice (halo gets refreshed by next exchange).
      for (int r = 0; r < nb; ++r) {
        auto& s = *shards[r];
        raft::device_setter guard(s.device_id);
        const i_t n_owned = s.rank_data.owned_cstr_size;
        cub::DeviceTransform::Transform(
          q[r].data(),
          q[r].data(),
          n_owned,
          [n = norm_q[r].data()] __device__(f_t v) { return v / *n; },
          s.stream.view().value());
      }

      // atq = A^T q : halo-exchange q, then per-shard SpMV. spmv_At_into
      // rebinds the dual_solution dnvec to q[r].data() and restores the
      // canonical binding after the call (see pdhg.cu:643-644).
      halo_exchange_cstr_bufs(q);
      for (int r = 0; r < nb; ++r) {
        auto& s = *shards[r];
        raft::device_setter guard(s.device_id);
        s.sub_pdlp->pdhg_solver_.spmv_At_into(q[r], atq_dn[r]);
      }

      // z = A atq : halo-exchange atq, then per-shard SpMV.
      halo_exchange_var_bufs(atq);
      for (int r = 0; r < nb; ++r) {
        auto& s = *shards[r];
        raft::device_setter guard(s.device_id);
        s.sub_pdlp->pdhg_solver_.spmv_A_into(atq[r], z_dn[r]);
      }

      // σ² = q · z over the global OWNED cstr slice (= q^T A A^T q = σ_max²
      // when q is the dominant left-singular vector).
      distributed_dot_owned_cstr(q, z, sigma_sq);

      // q := -σ² q + z (owned slice) — residual of the eigen-equation.
      for (int r = 0; r < nb; ++r) {
        auto& s = *shards[r];
        raft::device_setter guard(s.device_id);
        const i_t n_owned = s.rank_data.owned_cstr_size;
        cub::DeviceTransform::Transform(
          cuda::std::make_tuple(q[r].data(), z[r].data()),
          q[r].data(),
          n_owned,
          [s2 = sigma_sq[r].data()] __device__(f_t qv, f_t zv) { return -(*s2) * qv + zv; },
          s.stream.view().value());
      }

      // Convergence check via global residual norm.
      distributed_norm_owned_cstr(q, residual_norm);
      auto& s0 = *shards[0];
      raft::device_setter guard0(s0.device_id);
      f_t h_res{};
      RAFT_CUDA_TRY(cudaMemcpyAsync(&h_res,
                                    residual_norm[0].data(),
                                    sizeof(f_t),
                                    cudaMemcpyDeviceToHost,
                                    s0.stream.view().value()));
      s0.stream.synchronize();
      if (h_res < tolerance) break;
    }

    // σ_max² is the same on every shard after the last allreduce.
    auto& s0 = *shards[0];
    raft::device_setter guard0(s0.device_id);
    f_t sigma_sq_h{};
    RAFT_CUDA_TRY(cudaMemcpyAsync(&sigma_sq_h,
                                  sigma_sq[0].data(),
                                  sizeof(f_t),
                                  cudaMemcpyDeviceToHost,
                                  s0.stream.view().value()));
    s0.stream.synchronize();

    for (int r = 0; r < nb; ++r) {
      raft::device_setter guard(shards[r]->device_id);
      RAFT_CUSPARSE_TRY(cusparseDestroyDnVec(z_dn[r]));
      RAFT_CUSPARSE_TRY(cusparseDestroyDnVec(atq_dn[r]));
    }

    return std::sqrt(std::max(sigma_sq_h, f_t(0)));
  }

  // -------- Solution gather (shards -> master) ----------------------------
  // Assembles the global potential_next primal/dual solutions and the
  // reduced_cost on the master from the owned slices distributed across
  // shards. Each shard's first owned_var_size (resp. owned_cstr_size) entries
  // of its potential_next_primal_solution_ / reduced_cost_ (resp.
  // potential_next_dual_solution_) are the live, up-to-date owned values; the
  // master buffers are not updated during iterations and would otherwise
  // return stale data.
  //
  // Used right before fill_return_problem_solution() at the return sites in
  // pdlp_solver_t::check_termination() and pdlp_solver_t::check_limits(): the
  // user-visible solution must contain gathered global values for primal,
  // dual, and reduced_cost.
  //
  // Mirrors the metis_tests engine::get_x_output / get_y_output pattern:
  // per shard: alloc small host tmp, copy owned slice device->host, sync,
  // host-scatter via rank_data.local_to_global_{var,cstr} into a contiguous
  // host buffer. Then one host->device copy into the master buffer per field.
  //
  // master_pdhg          : provides destinations for primal / dual.
  // master_reduced_cost  : destination for the reduced_cost (var-shaped, lives
  //                        in the master pdlp_solver_t's termination strategy
  //                        convergence_information_).
  void gather_potential_next_solutions_to_master(pdhg_solver_t<i_t, f_t>& master_pdhg,
                                                 rmm::device_uvector<f_t>& master_reduced_cost)
  {
    const std::size_t total_vars  = master_pdhg.get_potential_next_primal_solution().size();
    const std::size_t total_cstrs = master_pdhg.get_potential_next_dual_solution().size();

    std::vector<f_t> h_primal(total_vars);
    std::vector<f_t> h_dual(total_cstrs);
    std::vector<f_t> h_reduced_cost(total_vars);

    for (auto& s_uptr : shards) {
      auto& s = *s_uptr;
      raft::device_setter guard(s.device_id);
      const i_t nv = s.rank_data.owned_var_size;
      const i_t nc = s.rank_data.owned_cstr_size;

      std::vector<f_t> tmp_primal(nv);
      std::vector<f_t> tmp_dual(nc);
      std::vector<f_t> tmp_reduced_cost(nv);

      auto& sub_reduced_cost = s.sub_pdlp->get_current_termination_strategy()
                                 .get_convergence_information()
                                 .get_reduced_cost();

      if (nv > 0) {
        RAFT_CUDA_TRY(
          cudaMemcpyAsync(tmp_primal.data(),
                          s.sub_pdlp->pdhg_solver_.get_potential_next_primal_solution().data(),
                          static_cast<std::size_t>(nv) * sizeof(f_t),
                          cudaMemcpyDeviceToHost,
                          s.stream.view().value()));
        RAFT_CUDA_TRY(cudaMemcpyAsync(tmp_reduced_cost.data(),
                                      sub_reduced_cost.data(),
                                      static_cast<std::size_t>(nv) * sizeof(f_t),
                                      cudaMemcpyDeviceToHost,
                                      s.stream.view().value()));
      }
      if (nc > 0) {
        RAFT_CUDA_TRY(
          cudaMemcpyAsync(tmp_dual.data(),
                          s.sub_pdlp->pdhg_solver_.get_potential_next_dual_solution().data(),
                          static_cast<std::size_t>(nc) * sizeof(f_t),
                          cudaMemcpyDeviceToHost,
                          s.stream.view().value()));
      }
      RAFT_CUDA_TRY(cudaStreamSynchronize(s.stream.view().value()));

      if (nv > 0) {
        thrust::scatter(thrust::host,
                        tmp_primal.begin(),
                        tmp_primal.end(),
                        s.rank_data.local_to_global_var.begin(),
                        h_primal.begin());
        thrust::scatter(thrust::host,
                        tmp_reduced_cost.begin(),
                        tmp_reduced_cost.end(),
                        s.rank_data.local_to_global_var.begin(),
                        h_reduced_cost.begin());
      }
      if (nc > 0) {
        thrust::scatter(thrust::host,
                        tmp_dual.begin(),
                        tmp_dual.end(),
                        s.rank_data.local_to_global_cstr.begin(),
                        h_dual.begin());
      }
    }

    // Host -> master device. engine.stream lives on the master device
    // (created at engine construction when master device was current).
    RAFT_CUDA_TRY(cudaMemcpyAsync(master_pdhg.get_potential_next_primal_solution().data(),
                                  h_primal.data(),
                                  total_vars * sizeof(f_t),
                                  cudaMemcpyHostToDevice,
                                  stream.view().value()));
    RAFT_CUDA_TRY(cudaMemcpyAsync(master_pdhg.get_potential_next_dual_solution().data(),
                                  h_dual.data(),
                                  total_cstrs * sizeof(f_t),
                                  cudaMemcpyHostToDevice,
                                  stream.view().value()));
    RAFT_CUDA_TRY(cudaMemcpyAsync(master_reduced_cost.data(),
                                  h_reduced_cost.data(),
                                  total_vars * sizeof(f_t),
                                  cudaMemcpyHostToDevice,
                                  stream.view().value()));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream.view().value()));
  }

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
  void graph_capture_fork_to_shards(rmm::cuda_stream_view master_stream)
  {
    graph_master_ready_event_->record(master_stream);
    for (auto& s : shards) {
      raft::device_setter guard(s->device_id);
      graph_master_ready_event_->stream_wait(s->stream.view());
    }
  }

  // Joins shards back to master stream for correct graph capture
  void graph_capture_join_from_shards(rmm::cuda_stream_view master_stream)
  {
    const int nb = static_cast<int>(shards.size());
    for (int r = 0; r < nb; ++r) {
      raft::device_setter guard(shards[r]->device_id);
      graph_shard_ready_events_[r]->record(shards[r]->stream.view());
    }
    for (auto& e : graph_shard_ready_events_) {
      e->stream_wait(master_stream);
    }
  }

  // Functionnaly same as graph_capture_fork_to_shards but on a different event to avoid race
  // conditions Can be used as a way to sync shards with master stream
  void sync_await_master(rmm::cuda_stream_view master_stream)
  {
    sync_master_ready_event_->record(master_stream);
    for (auto& s : shards) {
      raft::device_setter guard(s->device_id);
      sync_master_ready_event_->stream_wait(s->stream.view());
    }
  }

  // Same as sync_await_master
  // Can be used as a way to sync master stream with shards
  void sync_await_shards(rmm::cuda_stream_view master_stream)
  {
    const int nb = static_cast<int>(shards.size());
    for (int r = 0; r < nb; ++r) {
      raft::device_setter guard(shards[r]->device_id);
      sync_shard_ready_events_[r]->record(shards[r]->stream.view());
    }
    for (auto& e : sync_shard_ready_events_) {
      e->stream_wait(master_stream);
    }
  }
};

}  // namespace cuopt::linear_programming::detail
