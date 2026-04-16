/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "semi_continuous.cuh"

#include "bounds_presolve.cuh"

#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solver_context.cuh>
#include <utilities/logger.hpp>

#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace cuopt::linear_programming::detail {

namespace {

constexpr double sc_infinity_threshold = 1e30;

template <typename f_t>
bool is_effectively_infinite_sc_upper_bound(f_t ub)
{
  return !std::isfinite(ub) || ub >= static_cast<f_t>(sc_infinity_threshold);
}

}  // namespace

template <typename i_t, typename f_t>
bool reformulate_semi_continuous(optimization_problem_t<i_t, f_t>& op_problem,
                                 const mip_solver_settings_t<i_t, f_t>& settings,
                                 std::vector<uint8_t>* used_fallback_big_m)
{
  // 1. Identify semi-continuous variables
  auto var_types = op_problem.get_variable_types_host();
  auto var_ub    = op_problem.get_variable_upper_bounds_host();
  std::vector<i_t> sc_indices;
  bool normalized_large_sc_ub = false;
  for (i_t i = 0; i < static_cast<i_t>(var_types.size()); ++i) {
    if (var_types[i] != var_t::SEMI_CONTINUOUS) { continue; }
    sc_indices.push_back(i);
    if (is_effectively_infinite_sc_upper_bound(var_ub[i])) {
      CUOPT_LOG_WARN(
        "Semi-continuous var %d upper bound %.6g exceeds semi-continuous infinity "
        "threshold %.6g; treating it as +inf",
        i,
        static_cast<double>(var_ub[i]),
        sc_infinity_threshold);
      var_ub[i]              = std::numeric_limits<f_t>::infinity();
      normalized_large_sc_ub = true;
    }
  }
  if (sc_indices.empty()) { return false; }
  if (normalized_large_sc_ub) {
    op_problem.set_variable_upper_bounds(var_ub.data(), var_ub.size());
  }

  const i_t n_orig       = op_problem.get_n_variables();
  const i_t n_sc         = static_cast<i_t>(sc_indices.size());
  const auto* handle_ptr = op_problem.get_handle_ptr();
  const f_t big_m        = settings.sc_big_m;
  if (used_fallback_big_m != nullptr) { used_fallback_big_m->assign(n_orig, uint8_t{0}); }

  CUOPT_LOG_INFO("Reformulating %d semi-continuous variable(s) before presolve", n_sc);

  // 2. Build a relaxed copy where SC vars become continuous [0, original_ub].
  //    This lets GPU bounds propagation derive tight upper bounds from the
  //    constraint structure without the binary domain {0} ∪ [L, U].
  optimization_problem_t<i_t, f_t> op_relaxed(op_problem);
  {
    auto relaxed_types = var_types;
    auto relaxed_ub    = var_ub;
    auto relaxed_lb    = op_problem.get_variable_lower_bounds_host();
    for (i_t idx : sc_indices) {
      relaxed_types[idx] = var_t::CONTINUOUS;
      // Relax to the convex hull of {0} U [L, U] before running GPU bound propagation.
      relaxed_lb[idx] = std::min(f_t(0), relaxed_lb[idx]);
      if (std::isfinite(relaxed_ub[idx])) { relaxed_ub[idx] = std::max(f_t(0), relaxed_ub[idx]); }
    }
    op_relaxed.set_variable_types(relaxed_types.data(), n_orig);
    op_relaxed.set_variable_lower_bounds(relaxed_lb.data(), n_orig);
    op_relaxed.set_variable_upper_bounds(relaxed_ub.data(), n_orig);
  }

  // 3. Run GPU bounds propagation on the relaxed problem to tighten UBs.
  //    Skip propagation when there are no constraints (nothing to propagate).
  auto tight_ub = var_ub;  // fallback: normalized original UBs

  if (op_relaxed.get_n_constraints() > 0) {
    problem_t<i_t, f_t> temp_pb(op_relaxed, settings.get_tolerances());
    mip_solver_context_t<i_t, f_t> ctx(handle_ptr, &temp_pb, settings);

    typename bound_presolve_t<i_t, f_t>::settings_t bp_settings;
    bp_settings.time_limit = 5.0;
    bound_presolve_t<i_t, f_t> bps(ctx, bp_settings);
    bps.resize(temp_pb);
    bps.solve(temp_pb);

    // Copy tightened upper bounds from GPU to host
    raft::copy(tight_ub.data(), bps.upd.ub.data(), n_orig, handle_ptr->get_stream());
    handle_ptr->sync_stream();
  }

  // 4. Fetch all host arrays we need to extend with the new binary variables
  //    and linking constraints.
  auto var_lb = op_problem.get_variable_lower_bounds_host();
  auto obj_c  = op_problem.get_objective_coefficients_host();
  auto A_vals = op_problem.get_constraint_matrix_values_host();
  auto A_idx  = op_problem.get_constraint_matrix_indices_host();
  auto A_off  = op_problem.get_constraint_matrix_offsets_host();
  auto clb    = op_problem.get_constraint_lower_bounds_host();
  auto cub    = op_problem.get_constraint_upper_bounds_host();

  // Optional arrays — only extend if they were originally set
  auto b_rhs       = op_problem.get_constraint_bounds_host();
  auto row_types_h = op_problem.get_row_types_host();

  // Ensure objective and variable arrays are sized to n_orig
  if (obj_c.empty()) { obj_c.assign(n_orig, f_t(0)); }

  // 5. Count how many SC vars truly need the binary-variable reformulation.
  //    If 0 is already inside [L, U], then "x=0 OR L<=x<=U" simplifies to
  //    plain continuous [L, U] — no binary needed.
  std::vector<bool> needs_binary(n_sc, true);
  i_t n_binary_needed = 0;
  for (i_t s = 0; s < n_sc; ++s) {
    const i_t idx = sc_indices[s];
    needs_binary[s] =
      !(var_lb[idx] <= f_t(0) && std::isfinite(var_ub[idx]) && var_ub[idx] >= f_t(0)) &&
      !(var_lb[idx] <= f_t(0) && !std::isfinite(var_ub[idx]));
    if (needs_binary[s]) { ++n_binary_needed; }
  }

  // Extend variable arrays (one binary per SC var that actually needs it)
  var_types.resize(n_orig + n_binary_needed, var_t::INTEGER);
  var_lb.resize(n_orig + n_binary_needed, f_t(0));
  var_ub.resize(n_orig + n_binary_needed, f_t(1));
  obj_c.resize(n_orig + n_binary_needed, f_t(0));

  // 6. For each SC variable: derive U when needed, then either add binary + 2
  //    linking constraints or simply relax to continuous if 0 is already in
  //    the interval [L, U].
  i_t binary_count = 0;
  for (i_t s = 0; s < n_sc; ++s) {
    const i_t idx    = sc_indices[s];
    const f_t L      = var_lb[idx];
    const f_t orig_u = var_ub[idx];

    if (!needs_binary[s]) {
      // 0 already lies in [L, U], so the SC disjunction is just the interval itself.
      CUOPT_LOG_DEBUG(
        "SC var %d interval [%.6g, %.6g] already contains 0; treating it as continuous",
        idx,
        L,
        orig_u);
      var_types[idx] = var_t::CONTINUOUS;
      continue;
    }

    // Use GPU-propagated upper bound for positive-side SC variables when available.
    // For negative-side intervals, keep the original upper bound because the relaxed
    // convex hull includes 0 and is not useful for tightening the negative upper edge.
    f_t U = orig_u;
    if (orig_u >= f_t(0) || !std::isfinite(orig_u)) { U = tight_ub[idx]; }
    if (!std::isfinite(orig_u) && std::isfinite(U)) {
      CUOPT_LOG_DEBUG(
        "Semi-continuous var %d upper bound was tightened from %.6g to %.6g by "
        "bounds strengthening",
        idx,
        static_cast<double>(orig_u),
        static_cast<double>(U));
    }
    if (!std::isfinite(U)) { U = orig_u; }
    if (!std::isfinite(U)) {
      cuopt_assert(std::isfinite(big_m) && big_m >= L,
                   "Semi-continuous fallback sc_big_m must be finite and >= lower bound");
      U = big_m;
      CUOPT_LOG_DEBUG(
        "Semi-continuous var %d has no finite upper bound after bounds "
        "strengthening; using fallback sc_big_m %.6g",
        idx,
        static_cast<double>(big_m));
      if (used_fallback_big_m != nullptr) { (*used_fallback_big_m)[idx] = uint8_t{1}; }
    }

    CUOPT_LOG_DEBUG("SC var %d: L=%.6g, U=%.6g (after propagation)", idx, L, U);

    const i_t b_idx = n_orig + binary_count;
    ++binary_count;

    // Convert SC var to the convex hull of {0} U [L, U].
    var_types[idx] = var_t::CONTINUOUS;
    var_lb[idx]    = std::min(f_t(0), L);
    var_ub[idx]    = std::max(f_t(0), U);

    // Constraint 1: x_i - L * b_i >= 0  (clb=0, cub=+inf)
    A_vals.push_back(f_t(1));
    A_idx.push_back(idx);
    A_vals.push_back(-L);
    A_idx.push_back(b_idx);
    A_off.push_back(A_off.back() + 2);
    clb.push_back(f_t(0));
    cub.push_back(std::numeric_limits<f_t>::infinity());
    if (!b_rhs.empty()) { b_rhs.push_back(f_t(0)); }
    if (!row_types_h.empty()) { row_types_h.push_back('G'); }

    // Constraint 2: x_i - U * b_i <= 0  (clb=-inf, cub=0)
    A_vals.push_back(f_t(1));
    A_idx.push_back(idx);
    A_vals.push_back(-U);
    A_idx.push_back(b_idx);
    A_off.push_back(A_off.back() + 2);
    clb.push_back(-std::numeric_limits<f_t>::infinity());
    cub.push_back(f_t(0));
    if (!b_rhs.empty()) { b_rhs.push_back(f_t(0)); }
    if (!row_types_h.empty()) { row_types_h.push_back('L'); }
  }

  // 7. Rebuild op_problem with the extended data.
  const i_t new_n_vars        = n_orig + n_binary_needed;
  const i_t new_n_cons        = static_cast<i_t>(clb.size());
  const i_t new_nnz           = static_cast<i_t>(A_vals.size());
  const i_t added_constraints = 2 * n_binary_needed;

  CUOPT_LOG_INFO("SC reformulation added %d variable(s) and %d constraint(s)",
                 n_binary_needed,
                 added_constraints);

  op_problem.set_objective_coefficients(obj_c.data(), new_n_vars);
  op_problem.set_variable_lower_bounds(var_lb.data(), new_n_vars);
  op_problem.set_variable_upper_bounds(var_ub.data(), new_n_vars);
  op_problem.set_variable_types(var_types.data(), new_n_vars);
  op_problem.set_csr_constraint_matrix(
    A_vals.data(), new_nnz, A_idx.data(), new_nnz, A_off.data(), new_n_cons + 1);
  op_problem.set_constraint_lower_bounds(clb.data(), new_n_cons);
  op_problem.set_constraint_upper_bounds(cub.data(), new_n_cons);
  if (!b_rhs.empty()) { op_problem.set_constraint_bounds(b_rhs.data(), new_n_cons); }
  if (!row_types_h.empty()) { op_problem.set_row_types(row_types_h.data(), new_n_cons); }

  return true;
}

#if MIP_INSTANTIATE_FLOAT
template bool reformulate_semi_continuous<int, float>(optimization_problem_t<int, float>&,
                                                      const mip_solver_settings_t<int, float>&,
                                                      std::vector<uint8_t>*);
#endif

#if MIP_INSTANTIATE_DOUBLE
template bool reformulate_semi_continuous<int, double>(optimization_problem_t<int, double>&,
                                                       const mip_solver_settings_t<int, double>&,
                                                       std::vector<uint8_t>*);
#endif

}  // namespace cuopt::linear_programming::detail
