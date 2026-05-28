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

#include <cuopt/linear_programming/pdlp/solver_settings.hpp>

#include <raft/core/device_setter.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <cub/device/device_transform.cuh>
#include <cuda/std/tuple>

#include <nccl.h>

#include <memory>
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
  void distributed_l2_norm(BufAccess&& buf_access,
                           OutAccess&& out_access,
                           SizeAccess&& size_access)
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
  // engine is wired in. They drive the per-shard plan-based SpMV via the
  // canonical cusparse_view bindings (no rebinding) so the descriptor binding
  // is never disturbed by mGPU machinery.
  //
  // The halo-exchange MUST target the exact buffer the canonical descriptor
  // is bound to in the PDHG cusparse_view (see cusparse_view.cu lines 516-519
  // and 595-599):
  //   - cv.reflected_primal_solution -> reflected_primal_ (var-shaped)
  //   - cv.dual_solution             -> current.dual_solution_ (cstr-shaped)
  // For 1 shard the halo-exchange is a no-op, but the buffer choice is what
  // makes multi-shard correctness work, so we keep it accurate either way.
  void distributed_compute_A_x()
  {
    halo_exchange_var(
      [](auto& pdhg) -> rmm::device_uvector<f_t>& { return pdhg.get_reflected_primal(); });
    for_each_shard([](auto& shard) { shard.sub_pdlp->pdhg_solver_.spmvop_A_x(); });
  }

  void distributed_compute_At_y()
  {
    halo_exchange_cstr(
      [](auto& pdhg) -> rmm::device_uvector<f_t>& { return pdhg.get_dual_solution(); });
    for_each_shard([](auto& shard) { shard.sub_pdlp->pdhg_solver_.spmvop_At_y(); });
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
  void gather_potential_next_solutions_to_master(
    pdhg_solver_t<i_t, f_t>& master_pdhg, rmm::device_uvector<f_t>& master_reduced_cost)
  {
    const std::size_t total_vars =
      master_pdhg.get_potential_next_primal_solution().size();
    const std::size_t total_cstrs =
      master_pdhg.get_potential_next_dual_solution().size();

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

  // Functionnaly same as graph_capture_fork_to_shards but on a different event to avoid race conditions
  // Can be used as a way to sync shards with master stream
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
