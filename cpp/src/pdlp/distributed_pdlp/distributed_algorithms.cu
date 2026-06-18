/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pdlp/cusparse_view.hpp>
#include <pdlp/distributed_pdlp/distributed_algorithms.hpp>
#include <pdlp/distributed_pdlp/multi_gpu_engine.hpp>
#include <pdlp/pdlp.cuh>

#include <raft/core/nvtx.hpp>

#include <thrust/fill.h>

#include <cmath>
#include <random>
#include <vector>

namespace cuopt::linear_programming::detail {

// -------- Broadcast owned constraint (row) scaling into halo --------------
template <typename i_t, typename f_t>
void broadcast_constraint_scaling_to_halo(multi_gpu_engine_t<i_t, f_t>& engine)
{
  engine.halo_exchange_cstr_shard([](pdlp_shard_t<i_t, f_t>& s) -> rmm::device_uvector<f_t>& {
    return s.sub_pdlp->get_initial_scaling_strategy().get_cummulative_constraint_matrix_scaling();
  });
}

// -------- Broadcast owned variable (column) scaling into halo -------------
template <typename i_t, typename f_t>
void broadcast_variable_scaling_to_halo(multi_gpu_engine_t<i_t, f_t>& engine)
{
  engine.halo_exchange_var_shard([](pdlp_shard_t<i_t, f_t>& s) -> rmm::device_uvector<f_t>& {
    return s.sub_pdlp->get_initial_scaling_strategy().get_cummulative_variable_scaling();
  });
}

// -------- Solution gather (shards -> master) ------------------------------
template <typename i_t, typename f_t>
void gather_potential_next_solutions_to_master(multi_gpu_engine_t<i_t, f_t>& engine,
                                               pdhg_solver_t<i_t, f_t>& master_pdhg,
                                               rmm::device_uvector<f_t>& master_reduced_cost)
{
  const std::size_t total_vars  = master_pdhg.get_potential_next_primal_solution().size();
  const std::size_t total_cstrs = master_pdhg.get_potential_next_dual_solution().size();

  std::vector<f_t> h_primal(total_vars);
  std::vector<f_t> h_dual(total_cstrs);
  std::vector<f_t> h_reduced_cost(total_vars);

  for (auto& s_uptr : engine.shards) {
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
      raft::copy(tmp_primal.data(),
                 s.sub_pdlp->pdhg_solver_.get_potential_next_primal_solution().data(),
                 static_cast<std::size_t>(nv),
                 s.stream.view());
      raft::copy(tmp_reduced_cost.data(),
                 sub_reduced_cost.data(),
                 static_cast<std::size_t>(nv),
                 s.stream.view());
    }
    if (nc > 0) {
      raft::copy(tmp_dual.data(),
                 s.sub_pdlp->pdhg_solver_.get_potential_next_dual_solution().data(),
                 static_cast<std::size_t>(nc),
                 s.stream.view());
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
  raft::copy(master_pdhg.get_potential_next_primal_solution().data(),
             h_primal.data(),
             total_vars,
             engine.stream.view());
  raft::copy(master_pdhg.get_potential_next_dual_solution().data(),
             h_dual.data(),
             total_cstrs,
             engine.stream.view());
  raft::copy(master_reduced_cost.data(), h_reduced_cost.data(), total_vars, engine.stream.view());
  RAFT_CUDA_TRY(cudaStreamSynchronize(engine.stream.view().value()));
}

// -------- Distributed bound / objective rescaling -------------------------
template <typename i_t, typename f_t>
void distributed_bound_objective_rescaling(multi_gpu_engine_t<i_t, f_t>& engine,
                                           f_t c_scaling_weight)
{
  const int nb = static_cast<int>(engine.shards.size());

  // Per-shard packed partial squared norms: [0] = bound (rhs) sq, [1] = obj sq.
  std::vector<rmm::device_uvector<f_t>> sq;
  sq.reserve(nb);

  // 1) per-shard partial squared norms over owned entries only
  for (int r = 0; r < nb; ++r) {
    auto& s = *engine.shards[r];
    raft::device_setter guard(s.device_id);
    sq.emplace_back(2, s.stream.view());

    const auto& scaled     = s.sub_pdlp->get_initial_scaling_strategy().get_scaled_op_problem();
    const int n_owned_cstr = static_cast<int>(s.rank_data.owned_cstr_size);
    const int n_owned_var  = static_cast<int>(s.rank_data.owned_var_size);

    // Squared-norm contribution of each constraint's [lower, upper] bound pair
    // (mirrors rhs_sum_of_squares_t).
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
  CUOPT_NCCL_TRY(ncclGroupStart());
  for (int r = 0; r < nb; ++r) {
    auto& s = *engine.shards[r];
    raft::device_setter guard(s.device_id);
    CUOPT_NCCL_TRY(ncclAllReduce(sq[r].data(),
                                 sq[r].data(),
                                 2,
                                 nccl_data_type<f_t>(),
                                 ncclSum,
                                 s.comm.get(),
                                 s.stream.view().value()));
  }
  CUOPT_NCCL_TRY(ncclGroupEnd());

  // 3) derive the identical scalars and apply on every shard.
  for (int r = 0; r < nb; ++r) {
    auto& s = *engine.shards[r];
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
    auto& s = *engine.shards[r];
    raft::device_setter guard(s.device_id);
    s.stream.synchronize();
  }
}

// -------- Distributed Ruiz inf-scaling ------------------------------------
// Each shard owns its rows AND its columns and stores both complete (h_A =
// owned rows, h_A_t = owned columns)
template <typename i_t, typename f_t>
void distributed_ruiz_inf_scaling(multi_gpu_engine_t<i_t, f_t>& engine,
                                  int num_iter,
                                  i_t n_global_vars)
{
  if (num_iter <= 0 || n_global_vars <= 0) return;
  raft::common::nvtx::range scope("distributed_ruiz_inf_scaling");

  for (int it = 0; it < num_iter; ++it) {
    // Refresh halo copies of both cumulative scalings (owner -> halo) so the
    // per-shard kernels read correct opposite-axis factors on their halo.
    broadcast_variable_scaling_to_halo(engine);
    broadcast_constraint_scaling_to_halo(engine);

    // Per-shard local kernels: row inf-norm (owned rows, complete) + column
    // inf-norm from A_T (owned columns, complete; halo columns -> 0).
    engine.for_each_shard([](auto& shard) {
      shard.sub_pdlp->get_initial_scaling_strategy().ruiz_iter_compute_local_iteration_vectors();
    });

    // Fold into cumulative on owned entries (halo entries get refreshed by
    // the next iteration's broadcast).
    engine.for_each_shard([](auto& shard) {
      shard.sub_pdlp->get_initial_scaling_strategy().ruiz_iter_apply_cumulative_update();
    });
  }

  // Final refresh so downstream consumers (the scaled problem, the next
  // distributed_max_singular_value, etc.) see correct halo factors.
  broadcast_variable_scaling_to_halo(engine);
  broadcast_constraint_scaling_to_halo(engine);

  engine.for_each_shard([](auto& shard) { shard.stream.synchronize(); });
}

// -------- Distributed Pock-Chambolle scaling ------------------------------
// Distributed Pock-Chambolle: one pass, mirroring single-GPU
// pock_chambolle_scaling. Row sum-of-powers come from the row-major matrix
// (owned rows) and column sum-of-powers from A_T (owned columns).
template <typename i_t, typename f_t>
void distributed_pock_chambolle_scaling(multi_gpu_engine_t<i_t, f_t>& engine,
                                        f_t alpha,
                                        i_t n_global_vars)
{
  if (n_global_vars <= 0) return;
  raft::common::nvtx::range scope("distributed_pock_chambolle_scaling");

  // Refresh halo copies of both cumulative scalings
  broadcast_variable_scaling_to_halo(engine);
  broadcast_constraint_scaling_to_halo(engine);

  engine.for_each_shard([alpha](auto& shard) {
    shard.sub_pdlp->get_initial_scaling_strategy().pock_chambolle_compute_local_iteration_vectors(
      alpha);
  });

  engine.for_each_shard([](auto& shard) {
    shard.sub_pdlp->get_initial_scaling_strategy().pock_chambolle_apply_cumulative_update();
  });

  // Final refresh for downstream consumers.
  broadcast_variable_scaling_to_halo(engine);
  broadcast_constraint_scaling_to_halo(engine);

  engine.for_each_shard([](auto& shard) { shard.stream.synchronize(); });
}

// -------- Distributed sigma_max(A) via power iteration --------------------
// The function has to re-implement the multi_gpu_engine_t preimitives as the scratch buffers
// are not associated with shards.
template <typename i_t, typename f_t>
f_t distributed_max_singular_value(multi_gpu_engine_t<i_t, f_t>& engine,
                                   i_t n_global_cstrs,
                                   int max_iterations,
                                   f_t tolerance)
{
  raft::common::nvtx::range scope("distributed_max_singular_value");

  const int nb = static_cast<int>(engine.shards.size());

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
  // RAII descriptors: created below per shard, destroyed automatically when
  // the vectors go out of scope (no manual cusparseDestroyDnVec needed).
  std::vector<cusparse_dn_vec_descr_wrapper_t<f_t>> z_dn(nb);
  std::vector<cusparse_dn_vec_descr_wrapper_t<f_t>> atq_dn(nb);
  q.reserve(nb);
  z.reserve(nb);
  atq.reserve(nb);
  sigma_sq.reserve(nb);
  norm_q.reserve(nb);
  residual_norm.reserve(nb);

  for (int r = 0; r < nb; ++r) {
    auto& s = *engine.shards[r];
    raft::device_setter guard(s.device_id);
    const i_t cstr_total = s.rank_data.total_cstr_size;
    const i_t var_total  = s.rank_data.total_var_size;
    q.emplace_back(static_cast<std::size_t>(cstr_total), s.stream.view());
    z.emplace_back(static_cast<std::size_t>(cstr_total), s.stream.view());
    atq.emplace_back(static_cast<std::size_t>(var_total), s.stream.view());
    sigma_sq.emplace_back(std::size_t{1}, s.stream.view());
    norm_q.emplace_back(std::size_t{1}, s.stream.view());
    residual_norm.emplace_back(std::size_t{1}, s.stream.view());
    z_dn[r].create(static_cast<int64_t>(cstr_total), z.back().data());
    atq_dn[r].create(static_cast<int64_t>(var_total), atq.back().data());

    std::vector<f_t> h_owned_z(static_cast<std::size_t>(s.rank_data.owned_cstr_size));
    for (i_t i = 0; i < s.rank_data.owned_cstr_size; ++i) {
      const i_t g  = s.rank_data.local_to_global_cstr[i];
      h_owned_z[i] = h_global_z[g];
    }
    if (s.rank_data.owned_cstr_size > 0) {
      raft::copy(z.back().data(),
                 h_owned_z.data(),
                 static_cast<std::size_t>(s.rank_data.owned_cstr_size),
                 s.stream.view());
    }
    if (cstr_total > s.rank_data.owned_cstr_size) {
      thrust::fill(rmm::exec_policy_nosync(s.stream.view()),
                   z.back().data() + s.rank_data.owned_cstr_size,
                   z.back().data() + cstr_total,
                   f_t(0));
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
      auto& s = *engine.shards[r];
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
    CUOPT_NCCL_TRY(ncclGroupStart());
    for (int r = 0; r < nb; ++r) {
      auto& s = *engine.shards[r];
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
      auto& s  = *engine.shards[r];
      auto& rd = s.rank_data;
      raft::device_setter guard(s.device_id);
      auto& y = bufs[r];
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
  };
  auto halo_exchange_var_bufs = [&](std::vector<rmm::device_uvector<f_t>>& bufs) {
    for (int r = 0; r < nb; ++r) {
      auto& s = *engine.shards[r];
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
    CUOPT_NCCL_TRY(ncclGroupStart());
    for (int r = 0; r < nb; ++r) {
      auto& s = *engine.shards[r];
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
      auto& s  = *engine.shards[r];
      auto& rd = s.rank_data;
      raft::device_setter guard(s.device_id);
      auto& x = bufs[r];
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
  };

  // Per-shard partial reductions over the OWNED cstr slice + NCCL allreduce.
  // For norm: out := sqrt(Σ_r ||bufs[r][0:owned_cstr]||²).
  // For dot : out := Σ_r <a[r][0:owned_cstr], b[r][0:owned_cstr]>.
  auto distributed_norm_owned_cstr = [&](std::vector<rmm::device_uvector<f_t>>& bufs,
                                         std::vector<rmm::device_uvector<f_t>>& out) {
    for (int r = 0; r < nb; ++r) {
      auto& s = *engine.shards[r];
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
    CUOPT_NCCL_TRY(ncclGroupStart());
    for (int r = 0; r < nb; ++r) {
      auto& s = *engine.shards[r];
      raft::device_setter guard(s.device_id);
      CUOPT_NCCL_TRY(ncclAllReduce(out[r].data(),
                                   out[r].data(),
                                   1,
                                   nccl_data_type<f_t>(),
                                   ncclSum,
                                   s.comm.get(),
                                   s.stream.view().value()));
    }
    CUOPT_NCCL_TRY(ncclGroupEnd());
    for (int r = 0; r < nb; ++r) {
      auto& s = *engine.shards[r];
      raft::device_setter guard(s.device_id);
      cub::DeviceTransform::Transform(
        out[r].data(), out[r].data(), 1, sqrt_inplace_op_t<f_t>{}, s.stream.view().value());
    }
  };
  auto distributed_dot_owned_cstr = [&](std::vector<rmm::device_uvector<f_t>>& a,
                                        std::vector<rmm::device_uvector<f_t>>& b,
                                        std::vector<rmm::device_uvector<f_t>>& out) {
    for (int r = 0; r < nb; ++r) {
      auto& s = *engine.shards[r];
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
    CUOPT_NCCL_TRY(ncclGroupStart());
    for (int r = 0; r < nb; ++r) {
      auto& s = *engine.shards[r];
      raft::device_setter guard(s.device_id);
      CUOPT_NCCL_TRY(ncclAllReduce(out[r].data(),
                                   out[r].data(),
                                   1,
                                   nccl_data_type<f_t>(),
                                   ncclSum,
                                   s.comm.get(),
                                   s.stream.view().value()));
    }
    CUOPT_NCCL_TRY(ncclGroupEnd());
  };

  // ===== Power iteration =====
  // Mirrors single-GPU compute_initial_step_size
  for (int it = 0; it < max_iterations; ++it) {
    // q := z on the owned slice (the carried iterate), then normalize.
    for (int r = 0; r < nb; ++r) {
      auto& s = *engine.shards[r];
      raft::device_setter guard(s.device_id);
      const i_t n_owned = s.rank_data.owned_cstr_size;
      raft::copy(q[r].data(), z[r].data(), n_owned, s.stream.view());
    }

    // ||q||₂ over the global OWNED cstr slice (one allreduce-sum + sqrt).
    distributed_norm_owned_cstr(q, norm_q);

    // q /= ||q||₂ on owned slice (halo gets refreshed by next exchange).
    for (int r = 0; r < nb; ++r) {
      auto& s = *engine.shards[r];
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
    // canonical binding after the call
    halo_exchange_cstr_bufs(q);
    for (int r = 0; r < nb; ++r) {
      auto& s = *engine.shards[r];
      raft::device_setter guard(s.device_id);
      s.sub_pdlp->pdhg_solver_.spmv_At_into(q[r], atq_dn[r]);
    }

    // z = A atq : halo-exchange atq, then per-shard SpMV.
    halo_exchange_var_bufs(atq);
    for (int r = 0; r < nb; ++r) {
      auto& s = *engine.shards[r];
      raft::device_setter guard(s.device_id);
      s.sub_pdlp->pdhg_solver_.spmv_A_into(atq[r], z_dn[r]);
    }

    // σ² = q · z over the global OWNED cstr slice (= q^T A A^T q = σ_max²
    // when q is the dominant left-singular vector).
    distributed_dot_owned_cstr(q, z, sigma_sq);

    // q := -σ² q + z (owned slice) — residual of the eigen-equation.
    for (int r = 0; r < nb; ++r) {
      auto& s = *engine.shards[r];
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
    auto& s0 = *engine.shards[0];
    raft::device_setter guard0(s0.device_id);
    f_t h_res{};
    raft::copy(&h_res, residual_norm[0].data(), 1, s0.stream.view());
    s0.stream.synchronize();
    if (h_res < tolerance) break;
  }

  // σ_max² is the same on every shard after the last allreduce.
  auto& s0 = *engine.shards[0];
  raft::device_setter guard0(s0.device_id);
  f_t sigma_sq_h{};
  raft::copy(&sigma_sq_h, sigma_sq[0].data(), 1, s0.stream.view());
  s0.stream.synchronize();

  // z_dn / atq_dn descriptors are released by their RAII wrappers on return.
  return std::sqrt(std::max(sigma_sq_h, f_t(0)));
}

// ----- Explicit instantiations (mirror multi_gpu_engine_t<int, {double,float}>) -----
#define INSTANTIATE(F_TYPE)                                                                       \
  template void broadcast_constraint_scaling_to_halo<int, F_TYPE>(                                \
    multi_gpu_engine_t<int, F_TYPE> & engine);                                                    \
  template void broadcast_variable_scaling_to_halo<int, F_TYPE>(multi_gpu_engine_t<int, F_TYPE> & \
                                                                engine);                          \
  template void distributed_bound_objective_rescaling<int, F_TYPE>(                               \
    multi_gpu_engine_t<int, F_TYPE> & engine, F_TYPE c_scaling_weight);                           \
  template void distributed_ruiz_inf_scaling<int, F_TYPE>(                                        \
    multi_gpu_engine_t<int, F_TYPE> & engine, int num_iter, int n_global_vars);                   \
  template void distributed_pock_chambolle_scaling<int, F_TYPE>(                                  \
    multi_gpu_engine_t<int, F_TYPE> & engine, F_TYPE alpha, int n_global_vars);                   \
  template F_TYPE distributed_max_singular_value<int, F_TYPE>(                                    \
    multi_gpu_engine_t<int, F_TYPE> & engine,                                                     \
    int n_global_cstrs,                                                                           \
    int max_iterations,                                                                           \
    F_TYPE tolerance);                                                                            \
  template void gather_potential_next_solutions_to_master<int, F_TYPE>(                           \
    multi_gpu_engine_t<int, F_TYPE> & engine,                                                     \
    pdhg_solver_t<int, F_TYPE> & master_pdhg,                                                     \
    rmm::device_uvector<F_TYPE> & master_reduced_cost);

INSTANTIATE(double)
INSTANTIATE(float)

#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
