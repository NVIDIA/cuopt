/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda/iterator>
#include <cuopt/mathematical_optimization/optimization_problem.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_solution.hpp>
#include <mip_heuristics/utils.hpp>
#include <utilities/copy_helpers.hpp>

#include <cmath>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::detail {

template <typename f_t>
struct reported_objectives_t {
  f_t primal{std::numeric_limits<f_t>::quiet_NaN()};
  f_t dual{std::numeric_limits<f_t>::quiet_NaN()};
  f_t gap{std::numeric_limits<f_t>::quiet_NaN()};
  f_t relative_gap{std::numeric_limits<f_t>::quiet_NaN()};
};

// For minimization this is -sigma_[lower,upper](-multiplier).
// For maximization, select the bound using the negated multiplier instead.
// Zero contributes zero even on a free interval. An inadmissible nonzero
// multiplier on an infinite bound does not provide a finite dual objective.
template <typename f_t>
f_t objective_bound_term(f_t multiplier, f_t lower, f_t upper, f_t sense)
{
  if (!std::isfinite(multiplier) || std::isnan(lower) || std::isnan(upper)) {
    return std::numeric_limits<f_t>::quiet_NaN();
  }
  if (multiplier == 0) { return 0; }
  f_t bound = sense * multiplier > 0 ? lower : upper;
  return std::isfinite(bound) ? multiplier * bound : std::numeric_limits<f_t>::quiet_NaN();
}

template <typename i_t, typename f_t>
reported_objectives_t<f_t> evaluate_solution_objectives(
  const optimization_problem_t<i_t, f_t>& model,
  const std::vector<f_t>& x,
  const std::vector<f_t>& y,
  const std::vector<f_t>& z)
{
  reported_objectives_t<f_t> result;
  const auto n = model.get_n_variables(), m = model.get_n_constraints();
  if (x.size() != static_cast<size_t>(n)) { return result; }
  for (auto value : x) {
    if (!std::isfinite(value)) { return result; }
  }
  const auto c = model.get_objective_coefficients_host();
  if (c.size() != x.size()) { return result; }
  // The setter stores H = Q + Q^T. The user objective is
  // P = scale * (c^T x + (1/2) x^T H x + offset).
  std::vector<f_t> half_hx(n, 0);
  const auto& offsets = model.get_quadratic_objective_offsets();
  const auto& indices = model.get_quadratic_objective_indices();
  const auto& values  = model.get_quadratic_objective_values();
  if (!values.empty()) {
    for (size_t row = 0; row + 1 < offsets.size(); ++row) {
      half_hx[row] = f_t(0.5) * mip::compensated_dot2_csr(
                                  offsets.data(), indices.data(), values.data(), x.data(), row);
    }
  }
  const f_t quadratic = mip::compensated_dot2(half_hx.data(), x.data(), x.size());
  const f_t linear    = mip::compensated_dot2(c.data(), x.data(), x.size());
  const f_t scale     = model.get_objective_scaling_factor();
  const f_t offset    = model.get_objective_offset();
  result.primal       = scale * (linear + quadratic + offset);
  if (y.size() != static_cast<size_t>(m) || z.size() != x.size() ||
      model.has_quadratic_constraints()) {
    return result;
  }
  const auto row_lower  = model.get_constraint_lower_bounds_host();
  const auto row_upper  = model.get_constraint_upper_bounds_host();
  const auto rhs        = model.get_constraint_bounds_host();
  const auto types      = model.get_row_types_host();
  const auto lower      = model.get_variable_lower_bounds_host();
  const auto upper      = model.get_variable_upper_bounds_host();
  const bool boxed_rows = row_lower.size() == y.size() && row_upper.size() == y.size();
  const f_t inf         = std::numeric_limits<f_t>::infinity();
  const f_t sense       = model.get_sense() ? -1 : 1;
  std::vector<f_t> terms(m + n);
  for (i_t row = 0; row < m; ++row) {
    const char type = types.empty() ? 'E' : types[row];
    const f_t lo    = boxed_rows ? row_lower[row] : (type == 'L' ? -inf : rhs[row]);
    const f_t hi    = boxed_rows ? row_upper[row] : (type == 'G' ? inf : rhs[row]);
    terms[row]      = objective_bound_term(y[row], lo, hi, sense);
  }
  for (i_t col = 0; col < n; ++col) {
    terms[m + col] = objective_bound_term(z[col], lower[col], upper[col], sense);
  }
  for (auto value : terms) {
    if (!std::isfinite(value)) { return result; }
  }
  // At dual equality Hx + c = A^T y + z, the minimization dual value is
  // D = scale * (-sigma_[row_lower,row_upper](-y)
  //              -sigma_[lower,upper](-z) - (1/2)x^T H x + offset).
  // With inexact dual equality this is a dual objective estimate, not a certified bound.
  const f_t bound_sum =
    mip::compensated_dot2(terms.data(), cuda::make_constant_iterator(f_t(1)), terms.size());
  result.dual = scale * (bound_sum - quadratic + offset);
  if (std::isfinite(result.primal) && std::isfinite(result.dual)) {
    result.gap          = std::abs(result.primal - result.dual);
    result.relative_gap = result.gap / (1 + std::abs(result.primal) + std::abs(result.dual));
  }
  return result;
}

// Native solvers export vectors before outer postsolve. Evaluate only after that
// restoration, using the same original model and vectors that the caller receives.
template <typename i_t, typename f_t>
void update_native_objectives(const optimization_problem_t<i_t, f_t>& model,
                              optimization_problem_solution_t<i_t, f_t>& solution,
                              bool dual_available)
{
  auto& infos = solution.get_additional_termination_informations();
  if (infos.size() != 1) { return; }
  auto& info = infos.front();
  if (info.solved_by != method_t::Barrier && info.solved_by != method_t::DualSimplex) { return; }
  const f_t nan       = std::numeric_limits<f_t>::quiet_NaN();
  info.dual_objective = info.gap = info.relative_gap = nan;
  // A native solver can allocate zero-filled vectors without exporting an iterate.
  if (!std::isfinite(info.primal_objective)) { return; }
  const auto status = solution.get_termination_status();
  if (status != pdlp_termination_status_t::Optimal &&
      status != pdlp_termination_status_t::TimeLimit &&
      status != pdlp_termination_status_t::IterationLimit &&
      status != pdlp_termination_status_t::ConcurrentLimit) {
    return;
  }
  const auto stream = model.get_handle_ptr()->get_stream();
  const auto x      = cuopt::host_copy(solution.get_primal_solution(), stream);
  const auto y =
    dual_available ? cuopt::host_copy(solution.get_dual_solution(), stream) : std::vector<f_t>{};
  const auto z =
    dual_available ? cuopt::host_copy(solution.get_reduced_cost(), stream) : std::vector<f_t>{};
  const auto values     = evaluate_solution_objectives(model, x, y, z);
  info.primal_objective = values.primal;
  if (dual_available) {
    info.dual_objective = values.dual;
    info.gap            = values.gap;
    info.relative_gap   = values.relative_gap;
  }
}
}  // namespace cuopt::mathematical_optimization::detail
