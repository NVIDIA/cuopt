/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pdlp/cusparse_view.hpp>
#include <pdlp/distributed_pdlp/distributed_algorithms.hpp>
#include <pdlp/distributed_pdlp/multi_gpu_engine.hpp>
#include <pdlp/pdlp.cuh>
#include <pdlp/utils.cuh>

#include <raft/core/nvtx.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>

#include <cmath>
#include <random>
#include <vector>

namespace cuopt::mathematical_optimization::pdlp {

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
// compute and apply_bound_objective_rescaling_to_problem, unfused because we need a
// raw squared-sum on device to hand to NCCL AllReduce and the base version comptues
// tranform->reduce->transform in one cub call for efficiency
template <typename i_t, typename f_t>
void distributed_bound_objective_rescaling(multi_gpu_engine_t<i_t, f_t>& engine,
                                           f_t c_scaling_weight)
{
  raft::common::nvtx::range scope("distributed_bound_objective_rescaling");

  // 1) + 2) Local transform-reduce on each shard, accumulate the global
  //         squared L2 norms on host as we go.
  f_t global_bound_sq = f_t(0);
  f_t global_obj_sq   = f_t(0);
  engine.for_each_shard([&](auto& s) {
    const auto& scaled     = s.sub_pdlp->get_initial_scaling_strategy().get_scaled_op_problem();
    const i_t n_owned_cstr = static_cast<i_t>(s.rank_data.owned_cstr_size);
    const i_t n_owned_var  = static_cast<i_t>(s.rank_data.owned_var_size);
    auto policy            = rmm::exec_policy(s.stream.view());

    auto bounds_begin = thrust::make_zip_iterator(scaled.constraint_lower_bounds.data(),
                                                  scaled.constraint_upper_bounds.data());
    global_bound_sq += thrust::transform_reduce(policy,
                                                bounds_begin,
                                                bounds_begin + n_owned_cstr,
                                                rhs_sum_of_squares_t<f_t>{},
                                                f_t(0),
                                                thrust::plus<f_t>{});
    global_obj_sq += thrust::transform_reduce(policy,
                                              scaled.objective_coefficients.data(),
                                              scaled.objective_coefficients.data() + n_owned_var,
                                              weighted_square_op<f_t>{c_scaling_weight},
                                              f_t(0),
                                              thrust::plus<f_t>{});
  });

  // 3) Host-side derivation of the (identical on every shard) scaling scalars.
  const f_t bound_rescaling = rescaling_from_squared_norm_op<f_t>{}(global_bound_sq);
  const f_t obj_rescaling   = rescaling_from_squared_norm_op<f_t>{}(global_obj_sq);

  // 4) Publish + apply on every shard via the shared helpers.
  engine.for_each_shard([&](auto& s) {
    auto& scaling = s.sub_pdlp->get_initial_scaling_strategy();
    scaling.set_h_bound_rescaling(bound_rescaling);
    scaling.set_h_objective_rescaling(obj_rescaling);
    scaling.apply_bound_objective_rescaling_to_problem();
  });

  engine.for_each_shard([](auto& shard) { shard.stream.synchronize(); });
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
    engine.halo_exchange_var_shard([](auto& s) -> auto& {
      return s.sub_pdlp->get_initial_scaling_strategy().get_cummulative_variable_scaling();
    });
    engine.halo_exchange_cstr_shard([](auto& s) -> auto& {
      return s.sub_pdlp->get_initial_scaling_strategy().get_cummulative_constraint_matrix_scaling();
    });

    // Shard-local Ruiz iteration
    // rows: inf norm only over OWNED (full) rows from A
    // cols: inf norm only over OWNED (full) cols from A_T
    // Then fold into cumulative on owned entries (halo entries get refreshed by
    // the next iteration's halo update)
    engine.for_each_shard(
      [](auto& shard) { shard.sub_pdlp->get_initial_scaling_strategy().ruiz_iter_local(); });
  }

  // Final refresh so downstream consumers (the scaled problem, the next
  // distributed_max_singular_value, etc.) see correct halo factors.
  engine.halo_exchange_var_shard([](auto& s) -> auto& {
    return s.sub_pdlp->get_initial_scaling_strategy().get_cummulative_variable_scaling();
  });
  engine.halo_exchange_cstr_shard([](auto& s) -> auto& {
    return s.sub_pdlp->get_initial_scaling_strategy().get_cummulative_constraint_matrix_scaling();
  });

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
  engine.halo_exchange_var_shard([](auto& s) -> auto& {
    return s.sub_pdlp->get_initial_scaling_strategy().get_cummulative_variable_scaling();
  });
  engine.halo_exchange_cstr_shard([](auto& s) -> auto& {
    return s.sub_pdlp->get_initial_scaling_strategy().get_cummulative_constraint_matrix_scaling();
  });

  engine.for_each_shard([alpha](auto& shard) {
    shard.sub_pdlp->get_initial_scaling_strategy().pock_chambolle_scaling(alpha);
  });

  // Final refresh for downstream consumers.
  engine.halo_exchange_var_shard([](auto& s) -> auto& {
    return s.sub_pdlp->get_initial_scaling_strategy().get_cummulative_variable_scaling();
  });
  engine.halo_exchange_cstr_shard([](auto& s) -> auto& {
    return s.sub_pdlp->get_initial_scaling_strategy().get_cummulative_constraint_matrix_scaling();
  });

  engine.for_each_shard([](auto& shard) { shard.stream.synchronize(); });
}

// -------- Distributed scaling orchestration ------------------------------
// Mirrors what scale_problem() does in single-GPU by composing the
// individual distributed passes. See the header for the full pipeline.
template <typename i_t, typename f_t>
void distributed_scaling(multi_gpu_engine_t<i_t, f_t>& engine,
                         pdlp_hyper_params_t const& hyper_params,
                         i_t n_global_vars,
                         bool inside_mip)
{
  raft::common::nvtx::range scope("distributed_scaling");

  // 1) Reset per-shard scaling state (cumulative row/col scalings back to 1),
  //    then sync so subsequent scaling passes start from a clean slate.
  engine.for_each_shard([](auto& shard) {
    shard.sub_pdlp->get_initial_scaling_strategy().reset_scaling_state_for_distributed();
  });
  engine.for_each_shard([](auto& shard) { shard.stream.synchronize(); });

  // 2) Matrix scaling passes populate the cumulative row/col scalings on
  //    every shard. Each pass keeps the halo copies refreshed internally.
  if (hyper_params.do_ruiz_scaling) {
    distributed_ruiz_inf_scaling(engine, hyper_params.default_l_inf_ruiz_iterations, n_global_vars);
  }
  if (hyper_params.do_pock_chambolle_scaling) {
    distributed_pock_chambolle_scaling(
      engine, static_cast<f_t>(hyper_params.default_alpha_pock_chambolle_rescaling), n_global_vars);
  }

  // 3) Per-shard apply of the accumulated scaling to A, c, variable and
  //    constraint bounds. This is scale_problem() minus its local
  //    bound/objective rescaling; the equivalent global step happens in (4).
  engine.for_each_shard([](auto& shard) {
    shard.sub_pdlp->get_initial_scaling_strategy().apply_cummulative_scaling_to_problem();
  });
  engine.for_each_shard([](auto& shard) { shard.stream.synchronize(); });

  // 4) Global bound/objective rescaling (all shards get the identical scalar).
  if (hyper_params.bound_objective_rescaling) {
    distributed_bound_objective_rescaling(
      engine, static_cast<f_t>(hyper_params.initial_primal_weight_c_scaling));
  }
}

// -------- Distributed sigma_max(A) via power iteration --------------------
// Owns per-shard scratch (q / z / atq / scalar reductions) and drives the
// iteration; every cross-shard operation goes through multi_gpu_engine_t's
// *_bufs helpers (halo_exchange_{cstr,var}_bufs, distributed_l2_norm_bufs,
// distributed_dot_bufs), so this function contains no NCCL calls directly.
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

  // Per-shard scratch lives on each shard's device.]
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

  // Scatter z according to partition
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

  // Build the per-shard views (spans) used by the engine's *_bufs helpers.
  std::vector<raft::device_span<f_t>> q_full, atq_full;
  std::vector<raft::device_span<f_t>> q_owned, z_owned;
  std::vector<raft::device_scalar_view<f_t>> norm_q_scalar, sigma_sq_scalar, residual_scalar;
  q_full.reserve(nb);
  atq_full.reserve(nb);
  q_owned.reserve(nb);
  z_owned.reserve(nb);
  norm_q_scalar.reserve(nb);
  sigma_sq_scalar.reserve(nb);
  residual_scalar.reserve(nb);
  for (int r = 0; r < nb; ++r) {
    auto& s                 = *engine.shards[r];
    const std::size_t owned = static_cast<std::size_t>(s.rank_data.owned_cstr_size);
    q_full.emplace_back(q[r].data(), q[r].size());
    atq_full.emplace_back(atq[r].data(), atq[r].size());
    q_owned.emplace_back(q[r].data(), owned);
    z_owned.emplace_back(z[r].data(), owned);
    norm_q_scalar.emplace_back(raft::make_device_scalar_view<f_t>(norm_q[r].data()));
    sigma_sq_scalar.emplace_back(raft::make_device_scalar_view<f_t>(sigma_sq[r].data()));
    residual_scalar.emplace_back(raft::make_device_scalar_view<f_t>(residual_norm[r].data()));
  }

  // ===== Power iteration =====
  // Mirrors single-GPU compute_initial_step_size. All cross-shard math and
  // NCCL comms are delegated to multi_gpu_engine_t's *_bufs helpers; the
  // only inline work is (a) the two elementwise transforms whose functor
  // captures each shard's own scalar (norm_q[r], sigma_sq[r]) and (b) the
  // per-shard SpMVs that call pdhg_solver_'s A_into / A_T_into.
  for (int it = 0; it < max_iterations; ++it) {
    // q := z on the owned slice (the carried iterate).
    for (int r = 0; r < nb; ++r) {
      auto& s = *engine.shards[r];
      raft::device_setter guard(s.device_id);
      const i_t n_owned = s.rank_data.owned_cstr_size;
      raft::copy(q[r].data(), z[r].data(), n_owned, s.stream.view());
    }

    // ||q||₂ over the global OWNED cstr slice (one allreduce-sum + sqrt).
    engine.distributed_l2_norm_bufs(q_owned, norm_q_scalar);

    // q /= ||q||₂ on owned slice (halo gets refreshed by next exchange).
    // Kept inline: the divisor differs per shard (each shard reads its own
    // norm_q[r]) so a single shared functor won't do.
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

    // atq = A^T q : refresh halo of q, then per-shard SpMV.
    engine.halo_exchange_cstr_bufs(q_full);
    for (int r = 0; r < nb; ++r) {
      auto& s = *engine.shards[r];
      raft::device_setter guard(s.device_id);
      s.sub_pdlp->pdhg_solver_.spmv_At_into(q[r], atq_dn[r]);
    }

    // z = A atq : refresh halo of atq, then per-shard SpMV.
    engine.halo_exchange_var_bufs(atq_full);
    for (int r = 0; r < nb; ++r) {
      auto& s = *engine.shards[r];
      raft::device_setter guard(s.device_id);
      s.sub_pdlp->pdhg_solver_.spmv_A_into(atq[r], z_dn[r]);
    }

    // σ² = q · z over the global OWNED cstr slice (= q^T A A^T q = σ_max²
    // when q is the dominant left-singular vector).
    engine.distributed_dot_bufs(q_owned, z_owned, sigma_sq_scalar);

    // q := -σ² q + z (owned slice) — residual of the eigen-equation.
    // Kept inline for the same per-shard-scalar reason as normalize above.
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
    engine.distributed_l2_norm_bufs(q_owned, residual_scalar);
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
  template void distributed_bound_objective_rescaling<int, F_TYPE>(                               \
    multi_gpu_engine_t<int, F_TYPE> & engine, F_TYPE c_scaling_weight);                           \
  template void distributed_ruiz_inf_scaling<int, F_TYPE>(                                        \
    multi_gpu_engine_t<int, F_TYPE> & engine, int num_iter, int n_global_vars);                   \
  template void distributed_pock_chambolle_scaling<int, F_TYPE>(                                  \
    multi_gpu_engine_t<int, F_TYPE> & engine, F_TYPE alpha, int n_global_vars);                   \
  template void distributed_scaling<int, F_TYPE>(multi_gpu_engine_t<int, F_TYPE> & engine,        \
                                                 pdlp_hyper_params_t const& hyper_params,         \
                                                 int n_global_vars,                               \
                                                 bool inside_mip);                                \
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

}  // namespace cuopt::mathematical_optimization::pdlp
