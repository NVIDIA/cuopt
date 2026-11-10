/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once
// Original-space diagnostics copied from the audited preceding public49 driver.
// This header has no cuOpt, PSLP, CUDA or RMM dependencies; CPU tests cover its math.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static std::string quote(const std::string& value)
{
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else
      out << c;
  }
  out << '"';
  return out.str();
}
struct Json {
  std::ostringstream out;
  bool first = true;
  Json() { out << '{' << std::setprecision(17); }
  void key(const char* name)
  {
    if (!first) out << ',';
    first = false;
    out << quote(name) << ':';
  }
  void str(const char* name, const std::string& value)
  {
    key(name);
    out << quote(value);
  }
  void num(const char* name, long double value)
  {
    key(name);
    if (std::isfinite(value))
      out << value;
    else
      out << "null";
  }
  void flag(const char* name, bool value)
  {
    key(name);
    out << (value ? "true" : "false");
  }
  void emit() { std::cout << "\nRESULT_JSON " << out.str() << "}\n" << std::flush; }
};

struct Model {
  int m = 0, n = 0, nnz = 0, sense = 1;
  double offset = 0;
  std::vector<int> ap, ai;
  std::vector<double> ax, c, lhs, rhs, lb, ub;
  std::vector<char> types;
  // Preserve original coordinates while sorting CSR and combining duplicate entries.
  // This is the same canonicalization used in the preceding public49 study.
  void canonicalize()
  {
    if (ap.size() != size_t(m) + 1 || ap.front() != 0 || ap.back() != nnz)
      throw std::runtime_error("Invalid CSR row offsets");
    bool sorted = true;
    for (int i = 0; i < m; ++i) {
      if (ap[i] > ap[i + 1] || ap[i] < 0 || ap[i + 1] > nnz)
        throw std::runtime_error("Invalid CSR row offsets");
      for (int k = ap[i]; k < ap[i + 1]; ++k) {
        if (ai[k] < 0 || ai[k] >= n || !std::isfinite(ax[k]))
          throw std::runtime_error("Invalid CSR entry");
        if (k > ap[i] && ai[k - 1] >= ai[k]) sorted = false;
      }
    }
    if (sorted) return;
    std::vector<int> new_ap(size_t(m) + 1), new_ai;
    std::vector<double> new_ax;
    std::vector<std::pair<int, double>> row;
    new_ai.reserve(nnz);
    new_ax.reserve(nnz);
    for (int i = 0; i < m; ++i) {
      row.clear();
      for (int k = ap[i]; k < ap[i + 1]; ++k)
        row.emplace_back(ai[k], ax[k]);
      std::stable_sort(row.begin(), row.end(), [](auto a, auto b) { return a.first < b.first; });
      for (size_t k = 0; k < row.size();) {
        const int col     = row[k].first;
        long double value = 0;
        do {
          value += row[k++].second;
        } while (k < row.size() && row[k].first == col);
        if (value != 0) {
          new_ai.push_back(col);
          new_ax.push_back(double(value));
        }
      }
      new_ap[i + 1] = int(new_ai.size());
    }
    ap.swap(new_ap);
    ai.swap(new_ai);
    ax.swap(new_ax);
    nnz = int(ai.size());
  }
};

struct Solution {
  const double* x;
  const double* y;
  const double* z;
  size_t dim_x;
  size_t dim_y;
};

// Native cuOpt solves the original objective sense. Its row multipliers and
// reduced costs use that sense. Convert only the fetched diagnostic data to the
// signed-minimum convention used by the unchanged residual formulas below.
static void signed_minimum_diagnostics(Model& a, std::vector<double>& y, std::vector<double>& z)
{
  if (a.sense != 1 && a.sense != -1) throw std::runtime_error("Invalid objective sense");
  if (y.size() < size_t(a.m) || z.size() < size_t(a.n))
    throw std::runtime_error("Insufficient original solution dimensions");
  if (a.sense == -1) {
    for (auto& value : a.c)
      value = -value;
    for (int i = 0; i < a.m; ++i)
      y[i] = -y[i];
    for (int i = 0; i < a.n; ++i)
      z[i] = -z[i];
  }
}

// Checks are evaluated against the ORIGINAL model, using the signed-minimum
// objective. We report raw and explicitly defined normalized residuals, rather
// than claiming these diagnostics reproduce the benchmark acceptance test.
static void validate(
  const Model& a, const Solution& s, Json& j, double absolute_tolerance, double relative_tolerance)
{
  if (s.dim_x != size_t(a.n) || s.dim_y != size_t(a.m))
    throw std::runtime_error("Native solution has inconsistent original dimensions");
  bool finite = true, projected = false;
  long double primal_abs = 0, primal_componentwise = 0, stationarity = 0;
  long double row_violation = 0, variable_bound_violation = 0;
  long double row_bounds_inf = 0, variable_bounds_inf = 0;
  long double sign_violation = 0, complementarity = 0, cnorm = 0;
  long double pobj = a.sense * a.offset, dobj = a.sense * a.offset;
  long double row_residual_squared = 0, finite_rhs_squared = 0;
  long double stationarity_squared = 0, objective_squared = 0;
  long double stable3_dual = a.sense * a.offset, stable3_z_dot_x = 0;
  long double stable3_row_dual = 0;
  std::vector<long double> aty(a.n, 0);
  auto interval = [&](
                    long double value, double low, double high, long double multiplier, bool row) {
    auto& bounds_inf         = row ? row_bounds_inf : variable_bounds_inf;
    auto& interval_violation = row ? row_violation : variable_bound_violation;
    if (std::isfinite(low))
      bounds_inf = std::max(bounds_inf, std::fabs(static_cast<long double>(low)));
    if (std::isfinite(high))
      bounds_inf = std::max(bounds_inf, std::fabs(static_cast<long double>(high)));
    finite = finite && std::isfinite(value) && std::isfinite(multiplier);
    if (!std::isfinite(value) || !std::isfinite(multiplier)) return;
    if (std::isfinite(low)) {
      long double violation = std::max(0.L, static_cast<long double>(low) - value);
      primal_abs            = std::max(primal_abs, violation);
      interval_violation    = std::max(interval_violation, violation);
      primal_componentwise  = std::max(primal_componentwise, violation / (1 + std::fabs(low)));
    }
    if (std::isfinite(high)) {
      long double violation = std::max(0.L, value - static_cast<long double>(high));
      primal_abs            = std::max(primal_abs, violation);
      interval_violation    = std::max(interval_violation, violation);
      primal_componentwise  = std::max(primal_componentwise, violation / (1 + std::fabs(high)));
    }
    if (multiplier == 0) return;
    const double side = multiplier > 0 ? low : high;
    if (std::isfinite(side)) {
      dobj += multiplier * side;
      complementarity = std::max(complementarity, std::fabs(multiplier * (value - side)));
    } else {
      // Project an inadmissible signed multiplier to zero when forming an
      // approximate dual objective; its magnitude remains an explicit residual.
      projected      = true;
      sign_violation = std::max(sign_violation, std::fabs(multiplier));
    }
  };
  for (int i = 0; i < a.m; ++i) {
    long double activity = 0;
    for (int k = a.ap[i]; k < a.ap[i + 1]; ++k) {
      activity += static_cast<long double>(a.ax[k]) * s.x[a.ai[k]];
      aty[a.ai[k]] += static_cast<long double>(a.ax[k]) * s.y[i];
    }
    const long double clamped      = std::max(static_cast<long double>(a.lhs[i]),
                                         std::min(activity, static_cast<long double>(a.rhs[i])));
    const long double row_residual = activity - clamped;
    row_residual_squared += row_residual * row_residual;
    if (std::isfinite(a.rhs[i]))
      finite_rhs_squared += static_cast<long double>(a.rhs[i]) * a.rhs[i];
    if (std::isfinite(a.lhs[i]) && a.lhs[i] != a.rhs[i])
      finite_rhs_squared += static_cast<long double>(a.lhs[i]) * a.lhs[i];
    stable3_row_dual +=
      std::max(static_cast<long double>(s.y[i]), 0.L) * (std::isfinite(a.lhs[i]) ? a.lhs[i] : 0.0) +
      std::min(static_cast<long double>(s.y[i]), 0.L) * (std::isfinite(a.rhs[i]) ? a.rhs[i] : 0.0);
    interval(activity, a.lhs[i], a.rhs[i], s.y[i], true);
  }
  for (int k = 0; k < a.n; ++k) {
    pobj += static_cast<long double>(a.c[k]) * s.x[k];
    cnorm                = std::max(cnorm, std::fabs(static_cast<long double>(a.c[k])));
    long double residual = a.c[k] - aty[k] - s.z[k];
    stationarity_squared += residual * residual;
    objective_squared += static_cast<long double>(a.c[k]) * a.c[k];
    stable3_z_dot_x += static_cast<long double>(s.z[k]) * s.x[k];
    finite = finite && std::isfinite(residual);
    if (std::isfinite(residual)) stationarity = std::max(stationarity, std::fabs(residual));
    interval(s.x[k], a.lb[k], a.ub[k], s.z[k], false);
  }
  finite                            = finite && std::isfinite(pobj) && std::isfinite(dobj);
  const long double gap             = std::fabs(pobj - dobj);
  const long double objective_scale = 1 + std::fabs(pobj) + std::fabs(dobj);
  // Source-matched Stable3 formulas evaluated in the ORIGINAL model using
  // long-double CPU accumulation. This is a diagnostic only: the solver stops
  // in reduced space, and its GPU floating-point reduction order differs.
  stable3_dual += stable3_row_dual + stable3_z_dot_x;
  const long double row_l2                = std::sqrt(row_residual_squared);
  const long double rhs_l2                = std::sqrt(finite_rhs_squared);
  const long double stationarity_l2       = std::sqrt(stationarity_squared);
  const long double objective_l2          = std::sqrt(objective_squared);
  const long double stable3_gap           = std::fabs(pobj - stable3_dual);
  const long double stable3_scale         = std::fabs(pobj) + std::fabs(stable3_dual);
  const long double native_row_threshold  = absolute_tolerance + relative_tolerance * rhs_l2;
  const long double native_dual_threshold = absolute_tolerance + relative_tolerance * objective_l2;
  const long double native_gap_threshold  = absolute_tolerance + relative_tolerance * stable3_scale;
  const bool native_finite = finite && std::isfinite(row_l2) && std::isfinite(rhs_l2) &&
                             std::isfinite(stationarity_l2) && std::isfinite(objective_l2) &&
                             std::isfinite(stable3_dual);
  j.num("original_primal_row_residual_l2", row_l2);
  j.num("original_finite_rhs_l2", rhs_l2);
  j.num("original_objective_coefficients_l2", objective_l2);
  j.num("original_stationarity_l2", stationarity_l2);
  j.num("original_primal_row_residual_l2_div_1_plus_rhs_l2", row_l2 / (1 + rhs_l2));
  j.num("original_stationarity_l2_div_1_plus_c_l2", stationarity_l2 / (1 + objective_l2));
  j.num("original_stable3_z_dot_x", stable3_z_dot_x);
  j.num("original_stable3_row_dual_contribution", stable3_row_dual);
  j.num("original_stable3_signed_min_dual_objective", stable3_dual);
  j.num("original_stable3_dual_objective", a.sense * stable3_dual);
  j.num("original_stable3_absolute_gap", stable3_gap);
  j.num("original_stable3_relative_gap", stable3_gap / (1 + stable3_scale));
  j.num("original_native_formula_primal_l2_threshold", native_row_threshold);
  j.num("original_native_formula_dual_l2_threshold", native_dual_threshold);
  j.num("original_native_formula_gap_threshold", native_gap_threshold);
  j.flag("original_native_formula_primal_l2_pass", native_finite && row_l2 <= native_row_threshold);
  j.flag("original_native_formula_dual_l2_pass",
         native_finite && stationarity_l2 <= native_dual_threshold);
  j.flag("original_native_formula_gap_pass", native_finite && stable3_gap <= native_gap_threshold);
  j.flag("original_native_formula_all_checks",
         native_finite && row_l2 <= native_row_threshold &&
           stationarity_l2 <= native_dual_threshold && stable3_gap <= native_gap_threshold);
  j.str("original_native_formula_diagnostic_note",
        "Original model; long-double CPU accumulation; Stable3 formulas, not native reduced-space "
        "acceptance");
  j.flag("validation_finite", finite);
  j.num("primal_objective", a.sense * pobj);
  j.num("dual_objective_from_finite_sides", a.sense * dobj);
  j.flag("dual_objective_uses_sign_projection", projected);
  j.num("primal_violation_inf", primal_abs);
  j.num("primal_row_violation_inf", row_violation);
  j.num("primal_variable_bound_violation_inf", variable_bound_violation);
  j.num("finite_row_bounds_inf", row_bounds_inf);
  j.num("finite_variable_bounds_inf", variable_bounds_inf);
  j.num("objective_coefficients_inf", cnorm);
  j.num("primal_violation_componentwise", primal_componentwise);
  j.num("stationarity_inf", stationarity);
  j.num("stationarity_div_1_plus_c_inf", stationarity / (1 + cnorm));
  j.num("dual_sign_violation_inf", sign_violation);
  j.num("dual_sign_div_1_plus_c_inf", sign_violation / (1 + cnorm));
  j.num("complementarity_inf", complementarity);
  j.num("complementarity_div_objective_scale", complementarity / objective_scale);
  j.num("absolute_gap", gap);
  j.num("relative_gap", gap / objective_scale);
  j.flag("diagnostic_checks_1e_minus_6",
         finite && primal_componentwise <= 1e-6L && stationarity / (1 + cnorm) <= 1e-6L &&
           sign_violation / (1 + cnorm) <= 1e-6L && complementarity / objective_scale <= 1e-6L &&
           gap / objective_scale <= 1e-6L);
  // Independent mixed absolute/relative infinity-norm diagnostics. Separate
  // row and variable-bound scales prevent a large box bound from masking a
  // violated row. These are not cuOpt's default L2 stopping test or the board's
  // acceptance test. Complementarity is reported above and not a separate gate.
  const long double row_threshold   = 1e-4L + 1e-6L * row_bounds_inf;
  const long double bound_threshold = 1e-4L + 1e-6L * variable_bounds_inf;
  const long double dual_threshold  = 1e-4L + 1e-6L * cnorm;
  const long double gap_threshold   = 1e-4L + 1e-6L * (std::fabs(pobj) + std::fabs(dobj));
  j.num("mixed_inf_row_threshold", row_threshold);
  j.num("mixed_inf_variable_bound_threshold", bound_threshold);
  j.num("mixed_inf_stationarity_and_sign_threshold", dual_threshold);
  j.num("mixed_inf_gap_threshold", gap_threshold);
  j.flag("mixed_inf_diagnostic_checks",
         finite && row_violation <= row_threshold && variable_bound_violation <= bound_threshold &&
           stationarity <= dual_threshold && sign_violation <= dual_threshold &&
           gap <= gap_threshold);
  j.flag(
    "configured_mixed_inf_diagnostic_checks",
    finite && row_violation <= absolute_tolerance + relative_tolerance * row_bounds_inf &&
      variable_bound_violation <= absolute_tolerance + relative_tolerance * variable_bounds_inf &&
      stationarity <= absolute_tolerance + relative_tolerance * cnorm &&
      sign_violation <= absolute_tolerance + relative_tolerance * cnorm &&
      gap <= absolute_tolerance + relative_tolerance * (std::fabs(pobj) + std::fabs(dobj)));
}
