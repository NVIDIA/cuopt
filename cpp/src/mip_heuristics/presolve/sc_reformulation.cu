/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "sc_reformulation.cuh"

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

template <typename i_t, typename f_t>
bool reformulate_semi_continuous(optimization_problem_t<i_t, f_t>& op_problem,
                                 const mip_solver_settings_t<i_t, f_t>& settings)
{
  // -------------------------------------------------------------------------
  // 1. Identify semi-continuous variables
  // -------------------------------------------------------------------------
  auto var_types = op_problem.get_variable_types_host();
  std::vector<i_t> sc_indices;
  for (i_t i = 0; i < static_cast<i_t>(var_types.size()); ++i) {
    if (var_types[i] == var_t::SEMI_CONTINUOUS) { sc_indices.push_back(i); }
  }
  if (sc_indices.empty()) { return false; }

  const i_t n_orig     = op_problem.get_n_variables();
  const i_t n_sc       = static_cast<i_t>(sc_indices.size());
  const auto* handle_ptr = op_problem.get_handle_ptr();
  const f_t big_m      = settings.sc_big_m;

  CUOPT_LOG_INFO("Reformulating %d semi-continuous variable(s) before presolve", n_sc);

  // -------------------------------------------------------------------------
  // 2. Build a relaxed copy where SC vars become continuous [0, original_ub].
  //    This lets GPU bounds propagation derive tight upper bounds from the
  //    constraint structure without the binary domain {0} ∪ [L, U].
  // -------------------------------------------------------------------------
  optimization_problem_t<i_t, f_t> op_relaxed(op_problem);
  {
    auto relaxed_types = var_types;
    auto relaxed_lb    = op_problem.get_variable_lower_bounds_host();
    for (i_t idx : sc_indices) {
      relaxed_types[idx] = var_t::CONTINUOUS;
      relaxed_lb[idx]    = f_t(0);  // include 0 in the relaxed domain
    }
    op_relaxed.set_variable_types(relaxed_types.data(), n_orig);
    op_relaxed.set_variable_lower_bounds(relaxed_lb.data(), n_orig);
  }

  // -------------------------------------------------------------------------
  // 3. Run GPU bounds propagation on the relaxed problem to tighten UBs.
  //    Skip propagation when there are no constraints (nothing to propagate).
  // -------------------------------------------------------------------------
  auto tight_ub = op_problem.get_variable_upper_bounds_host();  // fallback: original UBs

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

  // -------------------------------------------------------------------------
  // 4. Fetch all host arrays we need to extend with the new binary variables
  //    and linking constraints.
  // -------------------------------------------------------------------------
  auto var_lb = op_problem.get_variable_lower_bounds_host();
  auto var_ub = op_problem.get_variable_upper_bounds_host();
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

  // -------------------------------------------------------------------------
  // 5. Count how many SC vars truly need the binary-variable reformulation.
  //    If L <= 0, then 0 is already in [L, U], so "x=0 OR L<=x<=U" simplifies
  //    to plain continuous [L, U] — no binary needed.
  // -------------------------------------------------------------------------
  i_t n_binary_needed = 0;
  for (i_t idx : sc_indices) {
    if (var_lb[idx] > f_t(0)) { ++n_binary_needed; }
  }

  // Extend variable arrays (one binary per SC var that actually needs it)
  var_types.resize(n_orig + n_binary_needed, var_t::INTEGER);
  var_lb.resize(n_orig + n_binary_needed, f_t(0));
  var_ub.resize(n_orig + n_binary_needed, f_t(1));
  obj_c.resize(n_orig + n_binary_needed, f_t(0));

  // -------------------------------------------------------------------------
  // 6. For each SC variable: derive U, then either add binary + 2 linking
  //    constraints (L > 0) or simply relax to continuous (L <= 0).
  // -------------------------------------------------------------------------
  i_t binary_count = 0;
  for (i_t s = 0; s < n_sc; ++s) {
    const i_t idx = sc_indices[s];
    const f_t L   = var_lb[idx];

    // Use GPU-propagated bound; fall back to original UB; then to big-M
    f_t U = tight_ub[idx];
    if (!std::isfinite(U)) { U = var_ub[idx]; }
    if (!std::isfinite(U)) { U = big_m; }

    if (L <= f_t(0)) {
      // 0 already in [L, U]: SC constraint is trivially the full range [L, U].
      CUOPT_LOG_WARN(
        "SC var %d has non-positive lower bound (L=%.6g); treating as continuous [%.6g, %.6g]",
        idx, L, L, U);
      var_types[idx] = var_t::CONTINUOUS;
      // lb and ub unchanged (keep [L, U])
      var_ub[idx] = U;
      continue;
    }

    CUOPT_LOG_DEBUG("SC var %d: L=%.6g, U=%.6g (after propagation)", idx, L, U);

    const i_t b_idx = n_orig + binary_count;
    ++binary_count;

    // Convert SC var to continuous [0, U]
    var_types[idx] = var_t::CONTINUOUS;
    var_lb[idx]    = f_t(0);
    var_ub[idx]    = U;

    // Constraint 1: x_i - L * b_i >= 0  (clb=0, cub=+inf)
    A_vals.push_back(f_t(1));   A_idx.push_back(idx);
    A_vals.push_back(-L);       A_idx.push_back(b_idx);
    A_off.push_back(A_off.back() + 2);
    clb.push_back(f_t(0));
    cub.push_back(std::numeric_limits<f_t>::infinity());
    if (!b_rhs.empty()) { b_rhs.push_back(f_t(0)); }
    if (!row_types_h.empty()) { row_types_h.push_back('G'); }

    // Constraint 2: x_i - U * b_i <= 0  (clb=-inf, cub=0)
    A_vals.push_back(f_t(1));   A_idx.push_back(idx);
    A_vals.push_back(-U);       A_idx.push_back(b_idx);
    A_off.push_back(A_off.back() + 2);
    clb.push_back(-std::numeric_limits<f_t>::infinity());
    cub.push_back(f_t(0));
    if (!b_rhs.empty()) { b_rhs.push_back(f_t(0)); }
    if (!row_types_h.empty()) { row_types_h.push_back('L'); }
  }

  // -------------------------------------------------------------------------
  // 7. Rebuild op_problem with the extended data.
  // -------------------------------------------------------------------------
  const i_t new_n_vars = n_orig + n_binary_needed;
  const i_t new_n_cons = static_cast<i_t>(clb.size());
  const i_t new_nnz    = static_cast<i_t>(A_vals.size());

  op_problem.set_objective_coefficients(obj_c.data(), new_n_vars);
  op_problem.set_variable_lower_bounds(var_lb.data(), new_n_vars);
  op_problem.set_variable_upper_bounds(var_ub.data(), new_n_vars);
  op_problem.set_variable_types(var_types.data(), new_n_vars);
  op_problem.set_csr_constraint_matrix(
    A_vals.data(), new_nnz, A_idx.data(), new_nnz, A_off.data(), new_n_cons + 1);
  op_problem.set_constraint_lower_bounds(clb.data(), new_n_cons);
  op_problem.set_constraint_upper_bounds(cub.data(), new_n_cons);
  if (!b_rhs.empty()) { op_problem.set_constraint_bounds(b_rhs.data(), new_n_cons); }
  if (!row_types_h.empty()) {
    op_problem.set_row_types(row_types_h.data(), new_n_cons);
  }

  return true;
}

#if MIP_INSTANTIATE_FLOAT
template bool reformulate_semi_continuous<int, float>(optimization_problem_t<int, float>&,
                                                      const mip_solver_settings_t<int, float>&);
#endif

#if MIP_INSTANTIATE_DOUBLE
template bool reformulate_semi_continuous<int, double>(optimization_problem_t<int, double>&,
                                                       const mip_solver_settings_t<int, double>&);
#endif

}  // namespace cuopt::linear_programming::detail
