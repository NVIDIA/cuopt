/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/bounds_strengthening.hpp>

#include <algorithm>
#include <cmath>

namespace cuopt::linear_programming::dual_simplex {

template <typename f_t>
static inline f_t update_lb(f_t curr_lb, f_t coeff, f_t delta_min_act, f_t delta_max_act)
{
  auto comp_bnd = (coeff < 0.) ? delta_min_act / coeff : delta_max_act / coeff;
  return std::max(curr_lb, comp_bnd);
}

template <typename f_t>
static inline f_t update_ub(f_t curr_ub, f_t coeff, f_t delta_min_act, f_t delta_max_act)
{
  auto comp_bnd = (coeff < 0.) ? delta_max_act / coeff : delta_min_act / coeff;
  return std::min(curr_ub, comp_bnd);
}

template <typename i_t, typename f_t>
static inline bool check_infeasibility(f_t min_a, f_t max_a, f_t cnst_lb, f_t cnst_ub, f_t eps)
{
  return (min_a > cnst_ub + eps) || (max_a < cnst_lb - eps);
}

#define DEBUG_BOUND_STRENGTHENING 0

template <typename i_t, typename f_t>
void print_bounds_stats(const std::vector<f_t>& lower,
                        const std::vector<f_t>& upper,
                        const simplex_solver_settings_t<i_t, f_t>& settings,
                        const std::string msg)
{
#if DEBUG_BOUND_STRENGTHENING
  f_t lb_norm = 0.0;
  f_t ub_norm = 0.0;

  i_t sz = lower.size();
  for (i_t i = 0; i < sz; ++i) {
    if (std::isfinite(lower[i])) { lb_norm += abs(lower[i]); }
    if (std::isfinite(upper[i])) { ub_norm += abs(upper[i]); }
  }
  settings.log.printf("%s :: lb norm %e, ub norm %e\n", msg.c_str(), lb_norm, ub_norm);
#endif
}

template <typename i_t, typename f_t>
bounds_strengthening_t<i_t, f_t>::bounds_strengthening_t(
  const lp_problem_t<i_t, f_t>& problem,
  const csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<char>& row_sense,
  const std::vector<variable_type_t>& var_types)
  : bounds_changed(problem.num_cols, false),
    A(problem.A),
    Arow(Arow),
    var_types(var_types),
    delta_min_activity(problem.num_rows),
    delta_max_activity(problem.num_rows),
    constraint_lb(problem.num_rows),
    constraint_ub(problem.num_rows)
{
  const bool is_row_sense_empty = row_sense.empty();
  if (is_row_sense_empty) {
    std::copy(problem.rhs.begin(), problem.rhs.end(), constraint_lb.begin());
    std::copy(problem.rhs.begin(), problem.rhs.end(), constraint_ub.begin());
  } else {
    //  Set the constraint bounds
    for (i_t i = 0; i < problem.num_rows; ++i) {
      if (row_sense[i] == 'E') {
        constraint_lb[i] = problem.rhs[i];
        constraint_ub[i] = problem.rhs[i];
      } else if (row_sense[i] == 'L') {
        constraint_ub[i] = problem.rhs[i];
        constraint_lb[i] = -inf;
      } else {
        constraint_lb[i] = problem.rhs[i];
        constraint_ub[i] = inf;
      }
    }
  }
}

template <typename i_t, typename f_t>
bool bounds_strengthening_t<i_t, f_t>::bounds_strengthening(
  std::vector<f_t>& lower_bounds,
  std::vector<f_t>& upper_bounds,
  const simplex_solver_settings_t<i_t, f_t>& settings)
{
  const i_t m = A.m;
  const i_t n = A.n;

  std::vector<bool> constraint_changed(m, true);
  std::vector<bool> variable_changed(n, false);
  std::vector<bool> constraint_changed_next(m, false);

  if (!bounds_changed.empty()) {
    std::fill(constraint_changed.begin(), constraint_changed.end(), false);
    for (i_t i = 0; i < n; ++i) {
      if (bounds_changed[i]) {
        const i_t row_start = A.col_start[i];
        const i_t row_end   = A.col_start[i + 1];
        for (i_t p = row_start; p < row_end; ++p) {
          const i_t j           = A.i[p];
          constraint_changed[j] = true;
        }
      }
    }
  }

  lower = lower_bounds;
  upper = upper_bounds;
  print_bounds_stats(lower, upper, settings, "Initial bounds");

  i_t iter             = 0;
  const i_t iter_limit = 10;
  while (iter < iter_limit) {
    for (i_t i = 0; i < m; ++i) {
      if (!constraint_changed[i]) { continue; }
      const i_t row_start = Arow.row_start[i];
      const i_t row_end   = Arow.row_start[i + 1];

      f_t min_a = 0.0;
      f_t max_a = 0.0;
      for (i_t p = row_start; p < row_end; ++p) {
        const i_t j    = Arow.j[p];
        const f_t a_ij = Arow.x[p];

        variable_changed[j] = true;
        if (a_ij > 0) {
          min_a += a_ij * lower[j];
          max_a += a_ij * upper[j];
        } else if (a_ij < 0) {
          min_a += a_ij * upper[j];
          max_a += a_ij * lower[j];
        }
        if (upper[j] == inf && a_ij > 0) { max_a = inf; }
        if (lower[j] == -inf && a_ij < 0) { max_a = inf; }

        if (lower[j] == -inf && a_ij > 0) { min_a = -inf; }
        if (upper[j] == inf && a_ij < 0) { min_a = -inf; }
      }

      f_t cnst_lb = constraint_lb[i];
      f_t cnst_ub = constraint_ub[i];
      bool is_infeasible =
        check_infeasibility<i_t, f_t>(min_a, max_a, cnst_lb, cnst_ub, settings.primal_tol);
      if (is_infeasible) {
        settings.log.printf(
          "Iter:: %d, Infeasible constraint %d, cnst_lb %e, cnst_ub %e, min_a %e, max_a %e\n",
          iter,
          i,
          cnst_lb,
          cnst_ub,
          min_a,
          max_a);
        return false;
      }

      delta_min_activity[i] = cnst_ub - min_a;
      delta_max_activity[i] = cnst_lb - max_a;
    }

    i_t num_bounds_changed = 0;

    for (i_t k = 0; k < n; ++k) {
      if (!variable_changed[k]) { continue; }
      f_t old_lb = lower[k];
      f_t old_ub = upper[k];

      f_t new_lb = old_lb;
      f_t new_ub = old_ub;

      const i_t row_start = A.col_start[k];
      const i_t row_end   = A.col_start[k + 1];
      for (i_t p = row_start; p < row_end; ++p) {
        const i_t i = A.i[p];

        if (!constraint_changed[i]) { continue; }
        const f_t a_ik = A.x[p];

        f_t delta_min_act = delta_min_activity[i];
        f_t delta_max_act = delta_max_activity[i];

        delta_min_act += (a_ik < 0) ? a_ik * old_ub : a_ik * old_lb;
        delta_max_act += (a_ik > 0) ? a_ik * old_ub : a_ik * old_lb;

        new_lb = std::max(new_lb, update_lb(old_lb, a_ik, delta_min_act, delta_max_act));
        new_ub = std::min(new_ub, update_ub(old_ub, a_ik, delta_min_act, delta_max_act));
      }

      // Integer rounding
      if (!var_types.empty() &&
          (var_types[k] == variable_type_t::INTEGER || var_types[k] == variable_type_t::BINARY)) {
        new_lb = std::ceil(new_lb - settings.integer_tol);
        new_ub = std::floor(new_ub + settings.integer_tol);
      }

      bool lb_updated = std::abs(new_lb - old_lb) > 1e3 * settings.primal_tol;
      bool ub_updated = std::abs(new_ub - old_ub) > 1e3 * settings.primal_tol;

      new_lb = std::max(new_lb, lower_bounds[k]);
      new_ub = std::min(new_ub, upper_bounds[k]);

      if (new_lb > new_ub + 1e-6) {
        settings.log.printf(
          "Iter:: %d, Infeasible variable after update %d, %e > %e\n", iter, k, new_lb, new_ub);
        return false;
      }
      if (new_lb != old_lb || new_ub != old_ub) {
        for (i_t p = row_start; p < row_end; ++p) {
          const i_t i                = A.i[p];
          constraint_changed_next[i] = true;
        }
      }

      lower[k] = std::min(new_lb, new_ub);
      upper[k] = std::max(new_lb, new_ub);

      bool bounds_changed = lb_updated || ub_updated;
      if (bounds_changed) { num_bounds_changed++; }
    }

    if (num_bounds_changed == 0) { break; }

    std::swap(constraint_changed, constraint_changed_next);
    std::fill(constraint_changed_next.begin(), constraint_changed_next.end(), false);
    std::fill(variable_changed.begin(), variable_changed.end(), false);

    iter++;
  }

  // settings.log.printf("Total strengthened variables %d\n", total_strengthened_variables);

#if DEBUG_BOUND_STRENGTHENING
  f_t lb_change      = 0.0;
  f_t ub_change      = 0.0;
  int num_lb_changed = 0;
  int num_ub_changed = 0;

  for (i_t i = 0; i < n; ++i) {
    if (lower[i] > problem.lower[i] + settings.primal_tol ||
        (!std::isfinite(problem.lower[i]) && std::isfinite(lower[i]))) {
      num_lb_changed++;
      lb_change +=
        std::isfinite(problem.lower[i])
          ? (lower[i] - problem.lower[i]) / (1e-6 + std::max(abs(lower[i]), abs(problem.lower[i])))
          : 1.0;
    }
    if (upper[i] < problem.upper[i] - settings.primal_tol ||
        (!std::isfinite(problem.upper[i]) && std::isfinite(upper[i]))) {
      num_ub_changed++;
      ub_change +=
        std::isfinite(problem.upper[i])
          ? (problem.upper[i] - upper[i]) / (1e-6 + std::max(abs(problem.upper[i]), abs(upper[i])))
          : 1.0;
    }
  }

  if (num_lb_changed > 0 || num_ub_changed > 0) {
    settings.log.printf(
      "lb change %e, ub change %e, num lb changed %d, num ub changed %d, iter %d\n",
      100 * lb_change / std::max(1, num_lb_changed),
      100 * ub_change / std::max(1, num_ub_changed),
      num_lb_changed,
      num_ub_changed,
      iter);
  }
  print_bounds_stats(lower, upper, settings, "Final bounds");
#endif

  lower_bounds = lower;
  upper_bounds = upper;

  return true;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class bounds_strengthening_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::dual_simplex
