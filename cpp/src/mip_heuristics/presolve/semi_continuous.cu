/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "semi_continuous.cuh"

#include "bounds_presolve.cuh"

#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solver_context.cuh>
#include <pdlp/translate.hpp>
#include <utilities/copy_helpers.hpp>
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

template <typename i_t, typename f_t>
std::vector<f_t> call_host_bounds_strengthening(const optimization_problem_t<i_t, f_t>& op_problem,
                                                const mip_solver_settings_t<i_t, f_t>& settings,
                                                const std::vector<i_t>& sc_indices)
{
  auto user_problem =
    cuopt_problem_to_simplex_problem<i_t, f_t>(op_problem.get_handle_ptr(), op_problem);

  dual_simplex::lp_problem_t<i_t, f_t> lp_problem(op_problem.get_handle_ptr(), 1, 1, 1);
  std::vector<i_t> new_slacks;
  dual_simplex::dualize_info_t<i_t, f_t> dualize_info;
  dual_simplex::simplex_solver_settings_t<i_t, f_t> simplex_settings;
  simplex_settings.primal_tol  = settings.tolerances.presolve_absolute_tolerance;
  simplex_settings.integer_tol = settings.tolerances.integrality_tolerance;
  simplex_settings.set_log(false);

  dual_simplex::convert_user_problem(
    user_problem, simplex_settings, lp_problem, new_slacks, dualize_info);

  auto var_types = user_problem.var_types;
  var_types.resize(lp_problem.num_cols, dual_simplex::variable_type_t::CONTINUOUS);

  dual_simplex::csr_matrix_t<i_t, f_t> Arow(1, 1, 1);
  lp_problem.A.to_compressed_row(Arow);

  // convert_user_problem returns an equality-form LP. Empty row_sense makes
  // bounds_strengthening_t use rhs as both lower and upper row bounds.
  std::vector<char> row_sense;
  dual_simplex::bounds_strengthening_t<i_t, f_t> strengthening(
    lp_problem, Arow, row_sense, var_types);
  std::vector<bool> bounds_changed(lp_problem.num_cols, false);
  for (i_t idx : sc_indices) {
    bounds_changed[idx] = true;
  }
  auto lower = lp_problem.lower;
  auto upper = lp_problem.upper;
  auto ok    = strengthening.bounds_strengthening(simplex_settings, bounds_changed, lower, upper);
  if (!ok) { return op_problem.get_variable_upper_bounds_host(); }

  upper.resize(user_problem.num_cols);
  return upper;
}

}  // namespace

template <typename i_t, typename f_t>
bool reformulate_semi_continuous(optimization_problem_t<i_t, f_t>& op_problem,
                                 const mip_solver_settings_t<i_t, f_t>& settings,
                                 std::vector<uint8_t>* used_fallback_big_m,
                                 std::vector<i_t>* semi_continuous_binary_to_original_indices)
{
  // 1. Identify semi-continuous variables
  auto var_types = op_problem.get_variable_types_host();
  auto var_lb    = op_problem.get_variable_lower_bounds_host();
  auto var_ub    = op_problem.get_variable_upper_bounds_host();
  std::vector<i_t> sc_indices;
  bool normalized_zero_lb_sc  = false;
  bool normalized_large_sc_ub = false;
  for (i_t i = 0; i < static_cast<i_t>(var_types.size()); ++i) {
    if (var_types[i] != var_t::SEMI_CONTINUOUS) { continue; }
    if (var_lb[i] == f_t(0)) {
      CUOPT_LOG_DEBUG("Semi-continuous variable %d has zero lower bound; treating it as continuous",
                      i);
      var_types[i]          = var_t::CONTINUOUS;
      normalized_zero_lb_sc = true;
      continue;
    }
    sc_indices.push_back(i);
    if (is_effectively_infinite_sc_upper_bound(var_ub[i])) {
      CUOPT_LOG_DEBUG(
        "Semi-continuous variable %d upper bound %.6g exceeds semi-continuous infinity "
        "threshold %.6g; treating it as +inf",
        i,
        static_cast<double>(var_ub[i]),
        sc_infinity_threshold);
      var_ub[i]              = std::numeric_limits<f_t>::infinity();
      normalized_large_sc_ub = true;
    }
  }
  if (normalized_zero_lb_sc) { op_problem.set_variable_types(var_types.data(), var_types.size()); }
  if (sc_indices.empty()) { return false; }
  if (normalized_large_sc_ub) {
    op_problem.set_variable_upper_bounds(var_ub.data(), var_ub.size());
  }

  const i_t n_orig       = op_problem.get_n_variables();
  const i_t n_sc         = static_cast<i_t>(sc_indices.size());
  const auto* handle_ptr = op_problem.get_handle_ptr();
  const f_t big_m        = settings.semi_continuous_big_m;
  if (used_fallback_big_m != nullptr) { used_fallback_big_m->assign(n_orig, uint8_t{0}); }

  CUOPT_LOG_INFO("Reformulating %d semi-continuous variables before presolve", n_sc);

  // 2. Build a relaxed copy where SC vars become continuous [0, original_ub].
  //    This lets deterministic CPU bounds strengthening derive tight upper bounds from the
  //    constraint structure without the binary domain {0} ∪ [L, U].
  optimization_problem_t<i_t, f_t> op_relaxed(op_problem);
  {
    auto relaxed_types = var_types;
    auto relaxed_ub    = var_ub;
    auto relaxed_lb    = op_problem.get_variable_lower_bounds_host();
    for (i_t idx : sc_indices) {
      relaxed_types[idx] = var_t::CONTINUOUS;
      relaxed_lb[idx]    = std::min(f_t(0), relaxed_lb[idx]);
      if (std::isfinite(relaxed_ub[idx])) { relaxed_ub[idx] = std::max(f_t(0), relaxed_ub[idx]); }
    }
    op_relaxed.set_variable_types(relaxed_types.data(), n_orig);
    op_relaxed.set_variable_lower_bounds(relaxed_lb.data(), n_orig);
    op_relaxed.set_variable_upper_bounds(relaxed_ub.data(), n_orig);
  }

  // 3. Run deterministic CPU bounds strengthening on the relaxed problem to tighten UBs.
  //    Skip strengthening when there are no constraints (nothing to propagate).
  auto tight_ub = var_ub;  // fallback: normalized original UBs

  if (op_relaxed.get_n_constraints() > 0) {
    tight_ub = call_host_bounds_strengthening(op_relaxed, settings, sc_indices);
  }

  // 4. Fetch all host arrays we need to extend with the new binary variables
  //    and linking constraints.
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
  if (semi_continuous_binary_to_original_indices != nullptr) {
    semi_continuous_binary_to_original_indices->clear();
    semi_continuous_binary_to_original_indices->reserve(n_binary_needed);
  }

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
        "Semi-continuous variable %d interval [%.6g, %.6g] already contains 0; treating it as "
        "continuous",
        idx,
        L,
        orig_u);
      var_types[idx] = var_t::CONTINUOUS;
      continue;
    }

    // Use CPU-strengthened upper bound for positive-side SC variables when available.
    // For negative-side intervals, keep the original upper bound because the relaxed
    // convex hull includes 0 and is not useful for tightening the negative upper edge.
    f_t U = orig_u;
    if (orig_u >= f_t(0) || !std::isfinite(orig_u)) { U = tight_ub[idx]; }
    if (!std::isfinite(orig_u) && std::isfinite(U)) {
      CUOPT_LOG_DEBUG(
        "Semi-continuous variable %d upper bound was tightened from %.6g to %.6g by "
        "CPU bounds strengthening",
        idx,
        static_cast<double>(orig_u),
        static_cast<double>(U));
    }
    if (!std::isfinite(U)) { U = orig_u; }
    if (!std::isfinite(U)) {
      cuopt_assert(
        std::isfinite(big_m) && big_m >= L,
        "Semi-continuous fallback mip_semi_continuous_big_m must be finite and >= lower bound");
      U = big_m;
      CUOPT_LOG_DEBUG(
        "Semi-continuous variable %d has no finite upper bound after bounds "
        "strengthening; using fallback mip_semi_continuous_big_m %.6g",
        idx,
        static_cast<double>(big_m));
      if (used_fallback_big_m != nullptr) { (*used_fallback_big_m)[idx] = uint8_t{1}; }
    }

    CUOPT_LOG_DEBUG("Semi-continuous variable %d: L=%.6g, U=%.6g (after propagation)", idx, L, U);

    const i_t b_idx = n_orig + binary_count;
    ++binary_count;
    if (semi_continuous_binary_to_original_indices != nullptr) {
      semi_continuous_binary_to_original_indices->push_back(idx);
    }

    // Convert SC var to the continuous interval [0, U].
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

  CUOPT_LOG_INFO("Semi-continuous reformulation added %d variables and %d constraints",
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

template <typename i_t, typename f_t>
void append_semi_continuous_auxiliaries_to_assignment(
  std::vector<f_t>& assignment,
  const std::vector<i_t>& semi_continuous_binary_to_original_indices,
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances)
{
  if (semi_continuous_binary_to_original_indices.empty()) { return; }

  const auto original_size = static_cast<i_t>(assignment.size());
  const f_t active_tol = std::max(tolerances.absolute_tolerance, tolerances.integrality_tolerance);
  assignment.reserve(assignment.size() + semi_continuous_binary_to_original_indices.size());
  for (i_t idx : semi_continuous_binary_to_original_indices) {
    cuopt_expects(idx >= 0 && idx < original_size,
                  error_type_t::ValidationError,
                  "Semi-continuous callback solution references an invalid parent variable index "
                  "%d.",
                  idx);
    assignment.push_back(assignment[idx] <= active_tol ? f_t(0) : f_t(1));
  }
}

template <typename i_t, typename f_t>
void strip_semi_continuous_auxiliaries_from_assignment(std::vector<f_t>& assignment,
                                                       i_t original_num_variables)
{
  if (assignment.size() <= static_cast<size_t>(original_num_variables)) { return; }
  cuopt_expects(
    original_num_variables >= 0 && original_num_variables <= static_cast<i_t>(assignment.size()),
    error_type_t::ValidationError,
    "Semi-continuous callback translation has invalid original variable count %d.",
    original_num_variables);
  assignment.resize(original_num_variables);
}

template <typename i_t, typename f_t>
void expand_initial_solutions_for_semi_continuous(
  mip_solver_settings_t<i_t, f_t>& settings,
  const std::vector<i_t>& semi_continuous_binary_to_original_indices,
  rmm::cuda_stream_view stream)
{
  if (semi_continuous_binary_to_original_indices.empty()) { return; }

  for (auto& initial_solution : settings.initial_solutions) {
    if (initial_solution == nullptr || initial_solution->is_empty()) { continue; }

    auto host_initial = cuopt::host_copy(*initial_solution, stream);
    std::vector<f_t> expanded_initial(host_initial.begin(), host_initial.end());
    append_semi_continuous_auxiliaries_to_assignment(
      expanded_initial, semi_continuous_binary_to_original_indices, settings.get_tolerances());

    initial_solution = std::make_shared<rmm::device_uvector<f_t>>(expanded_initial.size(), stream);
    raft::copy(initial_solution->data(), expanded_initial.data(), expanded_initial.size(), stream);
  }
}

#if MIP_INSTANTIATE_FLOAT
template bool reformulate_semi_continuous<int, float>(optimization_problem_t<int, float>&,
                                                      const mip_solver_settings_t<int, float>&,
                                                      std::vector<uint8_t>*,
                                                      std::vector<int>*);
template void append_semi_continuous_auxiliaries_to_assignment(
  std::vector<float>&, const std::vector<int>&, mip_solver_settings_t<int, float>::tolerances_t);
template void strip_semi_continuous_auxiliaries_from_assignment(std::vector<float>&, int);
template void expand_initial_solutions_for_semi_continuous(mip_solver_settings_t<int, float>&,
                                                           const std::vector<int>&,
                                                           rmm::cuda_stream_view);
#endif

#if MIP_INSTANTIATE_DOUBLE
template bool reformulate_semi_continuous<int, double>(optimization_problem_t<int, double>&,
                                                       const mip_solver_settings_t<int, double>&,
                                                       std::vector<uint8_t>*,
                                                       std::vector<int>*);
template void append_semi_continuous_auxiliaries_to_assignment(
  std::vector<double>&, const std::vector<int>&, mip_solver_settings_t<int, double>::tolerances_t);
template void strip_semi_continuous_auxiliaries_from_assignment(std::vector<double>&, int);
template void expand_initial_solutions_for_semi_continuous(mip_solver_settings_t<int, double>&,
                                                           const std::vector<int>&,
                                                           rmm::cuda_stream_view);
#endif

}  // namespace cuopt::linear_programming::detail
