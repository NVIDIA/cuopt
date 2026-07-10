/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Out-of-line definitions of multi_gpu_engine_t's high-level algorithm methods
// used by the pdlp solver.
#include <pdlp/cusparse_view.hpp>
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
// Gather the potential next primal/dual solutions and the reduced cost from shards to master.
template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::gather_potential_next_solutions_to_master()
{
  cuopt_assert(master_pdlp_ != nullptr,
               "gather_potential_next_solutions_to_master requires set_master(...)");

  gather_owned_var_to_master([](pdlp_solver_t<i_t, f_t>& p) -> rmm::device_uvector<f_t>& {
    return p.pdhg_solver_.get_potential_next_primal_solution();
  });

  gather_owned_cstr_to_master([](pdlp_solver_t<i_t, f_t>& p) -> rmm::device_uvector<f_t>& {
    return p.pdhg_solver_.get_potential_next_dual_solution();
  });

  gather_owned_var_to_master([](pdlp_solver_t<i_t, f_t>& p) -> rmm::device_uvector<f_t>& {
    return p.get_current_termination_strategy().get_convergence_information().get_reduced_cost();
  });
}

// -------- Distributed bound / objective rescaling -------------------------
// compute and apply_bound_objective_rescaling_to_problem, unfused because we need a
// raw squared-sum on device to reduce and the base version comptues
// tranform->reduce->transform in one cub call for efficiency
template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::distributed_bound_objective_rescaling(f_t c_scaling_weight)
{
  raft::common::nvtx::range scope("distributed_bound_objective_rescaling");

  // 1) + 2) Local transform-reduce on each shard, accumulate the global
  //         squared L2 norms on host as we go.
  f_t global_bound_sq = f_t(0);
  f_t global_obj_sq   = f_t(0);
  for_each_shard([&](auto& s) {
    const auto& scaled     = s.sub_pdlp->get_initial_scaling_strategy().get_scaled_op_problem();
    auto policy            = rmm::exec_policy(s.stream.view());
    auto bounds_begin = thrust::make_zip_iterator(scaled.constraint_lower_bounds.data(),
                                                  scaled.constraint_upper_bounds.data());
    global_bound_sq += thrust::transform_reduce(policy,
                                                bounds_begin,
                                                bounds_begin + s.rank_data.owned_cstr_size,
                                                rhs_sum_of_squares_t<f_t>{},
                                                f_t(0),
                                                thrust::plus<f_t>{});
    global_obj_sq += thrust::transform_reduce(policy,
                                              scaled.objective_coefficients.data(),
                                              scaled.objective_coefficients.data() + s.rank_data.owned_var_size,
                                              weighted_square_op<f_t>{c_scaling_weight},
                                              f_t(0),
                                              thrust::plus<f_t>{});
  });

  // 3) Host-side derivation of the (identical on every shard) scaling scalars.
  const f_t bound_rescaling = rescaling_from_squared_norm_op<f_t>{}(global_bound_sq);
  const f_t obj_rescaling   = rescaling_from_squared_norm_op<f_t>{}(global_obj_sq);

  // 4) Publish + apply on every shard via the shared helpers.
  for_each_shard([&](auto& s) {
    auto& scaling = s.sub_pdlp->get_initial_scaling_strategy();
    scaling.set_h_bound_rescaling(bound_rescaling);
    scaling.set_h_objective_rescaling(obj_rescaling);
    scaling.apply_bound_objective_rescaling_to_problem();
  });

  synchronize_shards();
}

// -------- Refresh halo of cumulative scalings -----------------------------
// Refreshes the halo copies of the cumulative variable + constraint scalings on
// every shard. Called before and after each matrix-scaling pass in ruiz and pock-chambolle
template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::refresh_halo_cummulative_scalings()
{
  halo_exchange_var([](pdlp_solver_t<i_t, f_t>& p) -> auto& {
    return p.get_initial_scaling_strategy().get_cummulative_variable_scaling();
  });
  halo_exchange_cstr([](pdlp_solver_t<i_t, f_t>& p) -> auto& {
    return p.get_initial_scaling_strategy().get_cummulative_constraint_matrix_scaling();
  });
}

// -------- Distributed Ruiz inf-scaling ------------------------------------
// Each shard owns its rows AND its columns and stores both complete (h_A =
// owned rows, h_A_t = owned columns)
template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::distributed_ruiz_inf_scaling(int num_iter, i_t n_global_vars)
{
  if (num_iter <= 0 || n_global_vars <= 0) return;
  raft::common::nvtx::range scope("distributed_ruiz_inf_scaling");

  for (int it = 0; it < num_iter; ++it) {
    refresh_halo_cummulative_scalings();

    // Shard-local Ruiz iteration
    // rows: inf norm only over OWNED (full) rows from A
    // cols: inf norm only over OWNED (full) cols from A_T
    // Then fold into cumulative on owned entries (halo entries get refreshed by
    // the next iteration's halo update)
    for_each_shard(
      [](auto& shard) { shard.sub_pdlp->get_initial_scaling_strategy().ruiz_iter_local(); });
  }

  // Final refresh after last iteration
  refresh_halo_cummulative_scalings();

  synchronize_shards();
}

// -------- Distributed Pock-Chambolle scaling ------------------------------
// Distributed Pock-Chambolle: one pass, mirroring single-GPU
// pock_chambolle_scaling. Row sum-of-powers come from the row-major matrix
// (owned rows) and column sum-of-powers from A_T (owned columns).
template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::distributed_pock_chambolle_scaling(f_t alpha, i_t n_global_vars)
{
  if (n_global_vars <= 0) return;
  raft::common::nvtx::range scope("distributed_pock_chambolle_scaling");

  refresh_halo_cummulative_scalings();

  for_each_shard([alpha](auto& shard) {
    shard.sub_pdlp->get_initial_scaling_strategy().pock_chambolle_scaling(alpha);
  });

  // Final refresh for downstream consumers.
  refresh_halo_cummulative_scalings();

  synchronize_shards();
}

// -------- Distributed scaling orchestration ------------------------------
// Mirrors single GPU scaling
template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::distributed_scaling(pdlp_hyper_params_t const& hyper_params,
                                                       i_t n_global_vars,
                                                       bool inside_mip)
{
  raft::common::nvtx::range scope("distributed_scaling");

  // 1) Matrix scaling passes populate the cumulative row/col scalings on
  //    every shard. Each pass keeps the halo copies refreshed internally.
  if (hyper_params.do_ruiz_scaling) {
    distributed_ruiz_inf_scaling(hyper_params.default_l_inf_ruiz_iterations, n_global_vars);
  }
  if (hyper_params.do_pock_chambolle_scaling) {
    distributed_pock_chambolle_scaling(
      static_cast<f_t>(hyper_params.default_alpha_pock_chambolle_rescaling), n_global_vars);
  }

  // 2) Per-shard apply of the accumulated scaling to A, c, variable and
  //    constraint bounds. This is scale_problem() minus its local
  //    bound/objective rescaling; the equivalent global step happens in (3).
  for_each_shard([](auto& shard) {
    shard.sub_pdlp->get_initial_scaling_strategy().apply_cummulative_scaling_to_problem();
  });
  synchronize_shards();

  // 3) Global bound/objective rescaling (all shards get the identical scalar).
  if (hyper_params.bound_objective_rescaling) {
    distributed_bound_objective_rescaling(
      static_cast<f_t>(hyper_params.initial_primal_weight_c_scaling));
  }
}

// -------- Distributed sigma_max(A) via power iteration --------------------
// Owns per-shard scratch (q / z / atq / scalar reductions) and drives the
// iteration; every cross-shard operation goes through multi_gpu_engine_t's
// *_bufs helpers (halo_exchange_{cstr,var}_bufs, distributed_l2_norm_bufs,
// distributed_dot_bufs), so this function contains no NCCL calls directly.
template <typename i_t, typename f_t>
f_t multi_gpu_engine_t<i_t, f_t>::distributed_max_singular_value(i_t n_global_cstrs,
                                                                 int max_iterations,
                                                                 f_t tolerance)
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
    auto& s                 = *shards[r];
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
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      const i_t n_owned = s.rank_data.owned_cstr_size;
      raft::copy(q[r].data(), z[r].data(), n_owned, s.stream.view());
    }

    // ||q||₂ over the global OWNED cstr slice (one allreduce-sum + sqrt).
    distributed_l2_norm_bufs(q_owned, norm_q_scalar);

    // q /= ||q||₂ on owned slice (halo gets refreshed by next exchange).
    // Kept inline: the divisor differs per shard (each shard reads its own
    // norm_q[r]) so a single shared functor won't do.
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

    // atq = A^T q : refresh halo of q, then per-shard SpMV.
    halo_exchange_cstr_bufs(q_full);
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      s.sub_pdlp->pdhg_solver_.spmv_At_into(q[r], atq_dn[r]);
    }

    // z = A atq : refresh halo of atq, then per-shard SpMV.
    halo_exchange_var_bufs(atq_full);
    for (int r = 0; r < nb; ++r) {
      auto& s = *shards[r];
      raft::device_setter guard(s.device_id);
      s.sub_pdlp->pdhg_solver_.spmv_A_into(atq[r], z_dn[r]);
    }

    // σ² = q · z over the global OWNED cstr slice (= q^T A A^T q = σ_max²
    // when q is the dominant left-singular vector).
    distributed_dot_bufs(q_owned, z_owned, sigma_sq_scalar);

    // q := -σ² q + z (owned slice) — residual of the eigen-equation.
    // Kept inline for the same per-shard-scalar reason as normalize above.
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
    distributed_l2_norm_bufs(q_owned, residual_scalar);
    auto& s0 = *shards[0];
    raft::device_setter guard0(s0.device_id);
    f_t h_res{};
    raft::copy(&h_res, residual_norm[0].data(), 1, s0.stream.view());
    s0.stream.synchronize();
    if (h_res < tolerance) break;
  }

  // σ_max² is the same on every shard after the last allreduce.
  auto& s0 = *shards[0];
  raft::device_setter guard0(s0.device_id);
  f_t sigma_sq_h{};
  raft::copy(&sigma_sq_h, sigma_sq[0].data(), 1, s0.stream.view());
  s0.stream.synchronize();

  // z_dn / atq_dn descriptors are released by their RAII wrappers on return.
  return std::sqrt(std::max(sigma_sq_h, f_t(0)));
}

// -------- Distributed initial step size ---------------------------------
// Sigma_max(A) via the shared power-iteration primitive.
//
// This function mirrors single-GPU's compute_initial_step_size exactly
// and broadcasts the result to every shard
template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::distributed_compute_initial_step_size(
  pdlp_hyper_params_t const& hyper_params,
  i_t n_global_cstrs,
  f_t scaling_factor,
  int max_iterations,
  f_t tolerance)
{
  raft::common::nvtx::range scope("distributed_compute_initial_step_size");
  cuopt_assert(master_pdlp_ != nullptr,
               "distributed_compute_initial_step_size requires set_master(...)");
  cuopt_expects(hyper_params.initial_step_size_max_singular_value,
                error_type_t::ValidationError,
                "distributed_compute_initial_step_size requires "
                "initial_step_size_max_singular_value = true; the max-abs-value "
                "of A fallback is single-GPU only. This should have been rejected "
                "earlier in solve_lp_distributed_from_mps.");

  const f_t sigma_max = distributed_max_singular_value(n_global_cstrs, max_iterations, tolerance);

  auto& master     = *master_pdlp_;
  auto* handle_ptr = master.get_handle_ptr();
  auto stream_view = handle_ptr->get_stream();

  const f_t h_step_size = (sigma_max > f_t{0}) ? scaling_factor / sigma_max : f_t{1};

  raft::copy(master.get_step_size().data(), &h_step_size, 1, stream_view);
  for_each_shard([&](auto& shard) {
    raft::copy(shard.sub_pdlp->get_step_size().data(), &h_step_size, 1, shard.stream);
  });
  sync_await_shards(stream_view);
  handle_ptr->sync_stream(stream_view);
}

// -------- Distributed initial primal weight ------------------------------
// Distributed PDLP is currently restricted to the Stable3-shaped hyper-param
// profile (validated up front in solve_lp_distributed_from_mps, and defended
// again here). In that regime, single-GPU compute_initial_primal_weight
// short-circuits to primal_weight = 1 without touching the norms (see
// pdlp.cu:
//   !initial_primal_weight_combined_bounds && bound_objective_rescaling
//   -> uninitialized_fill(primal_weight_ / best_primal_weight_, 1); return
// This function also fills the shards and masters primal_weight / step_size buffers
template <typename i_t, typename f_t>
void multi_gpu_engine_t<i_t, f_t>::distributed_compute_initial_primal_weight(
  pdlp_hyper_params_t const& hyper_params)
{
  raft::common::nvtx::range scope("distributed_compute_initial_primal_weight");
  cuopt_assert(master_pdlp_ != nullptr,
               "distributed_compute_initial_primal_weight requires set_master(...)");
  cuopt_expects(
    !hyper_params.initial_primal_weight_combined_bounds && hyper_params.bound_objective_rescaling,
    error_type_t::ValidationError,
    "distributed_compute_initial_primal_weight: only the Stable3-shaped "
    "short-circuit is supported (initial_primal_weight_combined_bounds=false "
    "and bound_objective_rescaling=true). This should have been rejected "
    "earlier in solve_lp_distributed_from_mps.");
  const f_t h_primal_weight = f_t(1);

  auto& master     = *master_pdlp_;
  auto* handle_ptr = master.get_handle_ptr();
  auto stream_view = handle_ptr->get_stream();

  raft::copy(master.get_primal_weight().data(), &h_primal_weight, 1, stream_view);
  raft::copy(master.get_best_primal_weight().data(), &h_primal_weight, 1, stream_view);
  for_each_shard([&](auto& shard) {
    auto& sub = *shard.sub_pdlp;
    raft::copy(sub.get_primal_weight().data(), &h_primal_weight, 1, shard.stream);
    raft::copy(sub.get_best_primal_weight().data(), &h_primal_weight, 1, shard.stream);
  });
  sync_await_shards(stream_view);
  handle_ptr->sync_stream(stream_view);
}

// ----- Explicit instantiations (member-by-member) --------------------------
// The class template is instantiated in multi_gpu_engine.cu; here we only
// explicit-instantiate the out-of-line members defined in this TU.
#define INSTANTIATE(F_TYPE)                                                                       \
  template void multi_gpu_engine_t<int, F_TYPE>::gather_potential_next_solutions_to_master();     \
  template void multi_gpu_engine_t<int, F_TYPE>::refresh_halo_cummulative_scalings();             \
  template void multi_gpu_engine_t<int, F_TYPE>::distributed_bound_objective_rescaling(F_TYPE);   \
  template void multi_gpu_engine_t<int, F_TYPE>::distributed_ruiz_inf_scaling(int, int);          \
  template void multi_gpu_engine_t<int, F_TYPE>::distributed_pock_chambolle_scaling(F_TYPE, int); \
  template void multi_gpu_engine_t<int, F_TYPE>::distributed_scaling(                             \
    pdlp_hyper_params_t const&, int, bool);                                                       \
  template F_TYPE multi_gpu_engine_t<int, F_TYPE>::distributed_max_singular_value(                \
    int, int, F_TYPE);                                                                            \
  template void multi_gpu_engine_t<int, F_TYPE>::distributed_compute_initial_step_size(           \
    pdlp_hyper_params_t const&, int, F_TYPE, int, F_TYPE);                                        \
  template void multi_gpu_engine_t<int, F_TYPE>::distributed_compute_initial_primal_weight(       \
    pdlp_hyper_params_t const&);

INSTANTIATE(double)
INSTANTIATE(float)

#undef INSTANTIATE

}  // namespace cuopt::mathematical_optimization::pdlp
