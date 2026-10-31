/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/domain.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>

namespace cuopt::mathematical_optimization::simplex {

namespace {
// One step of the Ogita-Rump-Oishi Dot2 algorithm: adds coeff * value to sum and accumulates the
// rounding errors of the product and of the addition in err, so sum + err is accurate to about
// twice the working precision. The product is formed with fma(coeff, value, 0) rather than coeff *
// value, so GCC's default -ffp-contract=fast cannot fuse it into the addition and break the TwoSum.
template <typename f_t>
void dot2_add(f_t coeff, f_t value, f_t& sum, f_t& err)
{
  const f_t h = std::fma(coeff, value, 0.0);
  const f_t t = sum + h;
  const f_t z = t - sum;
  err += ((sum - (t - z)) + (h - z)) + std::fma(coeff, value, -h);
  sum = t;
}
}  // namespace

template <typename i_t, typename f_t>
void domain_t<i_t, f_t>::compute_row_activity(i_t i,
                                              const csr_matrix_t<i_t, f_t>& Arow,
                                              const std::vector<f_t>& lower,
                                              const std::vector<f_t>& upper)
{
  i_t begin = Arow.row_start[i];
  i_t end   = Arow.row_start[i + 1];
  nnz_processed += end - begin;

  row_activity_t<i_t, f_t> activity;

  for (i_t k = begin; k < end; ++k) {
    i_t j         = Arow.j[k];
    f_t aij       = Arow.x[k];
    f_t bound_max = aij > 0 ? upper[j] : lower[j];
    f_t bound_min = aij < 0 ? upper[j] : lower[j];
    f_t alpha_max = aij * bound_max;
    f_t alpha_min = aij * bound_min;

    if (std::isfinite(alpha_max) && std::abs(alpha_max) < params.huge_value) {
      dot2_add(aij, bound_max, activity.max, activity.max_err);
    } else {
      ++activity.max_inf;
    }

    if (std::isfinite(alpha_min) && std::abs(alpha_min) < params.huge_value) {
      dot2_add(aij, bound_min, activity.min, activity.min_err);
    } else {
      ++activity.min_inf;
    }
  }

  // The peaks track the largest |activity| since the last full computation and are used by
  // update_activities to detect numerical cancellation.
  activity.max_peak = std::abs(activity.max);
  activity.min_peak = std::abs(activity.min);
  row_activities[i] = activity;
}

template <typename i_t, typename f_t>
void domain_t<i_t, f_t>::compute_activities(const csr_matrix_t<i_t, f_t>& Arow,
                                            const lp_problem_t<i_t, f_t>& lp)
{
  row_activities.resize(Arow.m);
  row_queued.resize(Arow.m, false);
  row_queue.clear_resize(std::max(Arow.m, 1));

  for (i_t i = 0; i < Arow.m; ++i) {
    compute_row_activity(i, Arow, lp.lower, lp.upper);
  }
}

template <typename i_t, typename f_t>
void domain_t<i_t, f_t>::update_activities(const lp_problem_t<i_t, f_t>& lp,
                                           const bound_change_t<i_t, f_t>& bound_change)
{
  i_t begin = lp.A.col_start[bound_change.var];
  i_t end   = lp.A.col_start[bound_change.var + 1];
  nnz_processed += end - begin;

  for (i_t k = begin; k < end; ++k) {
    i_t i                              = lp.A.i[k];
    row_activity_t<i_t, f_t>& activity = row_activities[i];

    // A wider range can raise the capacity threshold of the rows, so reset it until the row is
    // propagated again.
    if (bound_change.new_upper - bound_change.new_lower >
        bound_change.old_upper - bound_change.old_lower) {
      activity.capacity_threshold = inf;
    }

    // A row waiting for recomputation is rebuilt from the current bounds anyway.
    if (activity.recompute) { continue; }

    f_t aij           = lp.A.x[k];
    f_t old_bound_max = aij > 0 ? bound_change.old_upper : bound_change.old_lower;
    f_t old_bound_min = aij < 0 ? bound_change.old_upper : bound_change.old_lower;
    f_t new_bound_max = aij > 0 ? bound_change.new_upper : bound_change.new_lower;
    f_t new_bound_min = aij < 0 ? bound_change.new_upper : bound_change.new_lower;
    f_t old_alpha_max = aij * old_bound_max;
    f_t old_alpha_min = aij * old_bound_min;
    f_t new_alpha_max = aij * new_bound_max;
    f_t new_alpha_min = aij * new_bound_min;

    // Replace the old contribution to the max activity with the new one, moving it between the
    // finite sum and the infinite count when needed.
    if (old_alpha_max != new_alpha_max) {
      if (std::isfinite(old_alpha_max) && std::abs(old_alpha_max) < params.huge_value) {
        dot2_add(-aij, old_bound_max, activity.max, activity.max_err);
      } else {
        --activity.max_inf;
      }

      if (std::isfinite(new_alpha_max) && std::abs(new_alpha_max) < params.huge_value) {
        dot2_add(aij, new_bound_max, activity.max, activity.max_err);
      } else {
        ++activity.max_inf;
      }

      activity.max_peak = std::max(activity.max_peak, std::abs(activity.max));
    }

    // Same for the min activity.
    if (old_alpha_min != new_alpha_min) {
      if (std::isfinite(old_alpha_min) && std::abs(old_alpha_min) < params.huge_value) {
        dot2_add(-aij, old_bound_min, activity.min, activity.min_err);
      } else {
        --activity.min_inf;
      }

      if (std::isfinite(new_alpha_min) && std::abs(new_alpha_min) < params.huge_value) {
        dot2_add(aij, new_bound_min, activity.min, activity.min_err);
      } else {
        ++activity.min_inf;
      }

      activity.min_peak = std::max(activity.min_peak, std::abs(activity.min));
    }

    // An activity much smaller than its peak has lost precision to numerical cancellations, so mark
    // the row to be recomputed.
    if (activity.max_peak > params.recompute_factor * std::max<f_t>(std::abs(activity.max), 1.0) ||
        activity.min_peak > params.recompute_factor * std::max<f_t>(std::abs(activity.min), 1.0)) {
      activity.recompute = true;
    }
  }
}

template <typename i_t, typename f_t>
void domain_t<i_t, f_t>::queue_row(i_t i, const lp_problem_t<i_t, f_t>& lp, f_t tol)
{
  if (row_queued[i]) { return; }

  const row_activity_t<i_t, f_t>& activity = row_activities[i];
  if (!activity.recompute) {
    const f_t max_a = activity.max + activity.max_err;
    const f_t min_a = activity.min + activity.min_err;
    const f_t rhs   = lp.rhs[i];

    const bool propagate_upper =
      (activity.max_inf != 0 || max_a > rhs + tol) &&
      (activity.min_inf == 1 ||
       (activity.min_inf == 0 && rhs - min_a <= activity.capacity_threshold));
    const bool propagate_lower =
      (activity.min_inf != 0 || min_a < rhs - tol) &&
      (activity.max_inf == 1 ||
       (activity.max_inf == 0 && max_a - rhs <= activity.capacity_threshold));
    if (!propagate_upper && !propagate_lower) { return; }
  }

  row_queue.push_back(i);
  row_queued[i] = true;
}

template <typename i_t, typename f_t>
void domain_t<i_t, f_t>::queue_variable(i_t j, const lp_problem_t<i_t, f_t>& lp, f_t tol)
{
  const i_t col_start = lp.A.col_start[j];
  const i_t col_end   = lp.A.col_start[j + 1];
  nnz_processed += col_end - col_start;
  for (i_t p = col_start; p < col_end; ++p) {
    queue_row(lp.A.i[p], lp, tol);
  }
}

template <typename i_t, typename f_t>
bool domain_t<i_t, f_t>::tighten_bounds(i_t k,
                                        f_t new_lower,
                                        f_t new_upper,
                                        bound_change_origin_t origin,
                                        const std::vector<variable_type_t>& var_types,
                                        const simplex_solver_settings_t<i_t, f_t>& settings,
                                        lp_problem_t<i_t, f_t>& lp)
{
  const f_t tol             = settings.primal_tol;
  const f_t min_improvement = params.min_improvement_factor * tol;
  const f_t lb              = lp.lower[k];
  const f_t ub              = lp.upper[k];

  // Discard derived bounds that are too large in magnitude.
  if (std::abs(new_lower) >= params.max_derived_bound) { new_lower = -inf; }
  if (std::abs(new_upper) >= params.max_derived_bound) { new_upper = inf; }

  bool tighten_lb = false;
  bool tighten_ub = false;

  if (var_types[k] == variable_type_t::INTEGER) {
    new_lower  = std::ceil(new_lower - settings.integer_tol);
    new_upper  = std::floor(new_upper + settings.integer_tol);
    tighten_lb = new_lower - lb > settings.integer_tol;
    tighten_ub = ub - new_upper > settings.integer_tol;
  } else if (lp.A.col_start[k + 1] - lp.A.col_start[k] == 1) {
    // A continuous column singleton (e.g., a slack) is only bounded, never tightened: its
    // finite bounds are implied by its only row and only make the LP harder to solve.
    tighten_lb = lb == -inf && new_lower > -inf;
    tighten_ub = ub == inf && new_upper < inf;
  } else {
    const f_t lb_range = ub < inf ? ub - lb : std::max(std::abs(lb), std::abs(new_lower));
    const f_t ub_range = lb > -inf ? ub - lb : std::max(std::abs(ub), std::abs(new_upper));
    tighten_lb         = new_lower - min_improvement > lb &&
                 (lb == -inf || new_lower - lb >= params.min_relative_improvement * lb_range);
    tighten_ub = new_upper + min_improvement < ub &&
                 (ub == inf || ub - new_upper >= params.min_relative_improvement * ub_range);
  }

  if (!tighten_lb && !tighten_ub) { return true; }
  if (!tighten_lb) { new_lower = lb; }
  if (!tighten_ub) { new_upper = ub; }

  if (new_lower > new_upper) {
    if (new_lower - new_upper > tol * std::max({1.0, std::abs(new_lower), std::abs(new_upper)})) {
      settings.log.debug(
        "Infeasible variable %d after applying a propagation, %e > %e\n", k, new_lower, new_upper);
      return false;
    }
    if (tighten_ub) {
      new_upper = new_lower;
    } else {
      new_lower = new_upper;
    }
  }

  apply(lp, {.var = k, .new_upper = new_upper, .new_lower = new_lower, .origin = origin});
  queue_variable(k, lp, tol);
  return true;
}

template <typename i_t, typename f_t>
bool domain_t<i_t, f_t>::propagate_full(const csr_matrix_t<i_t, f_t>& Arow,
                                        const std::vector<variable_type_t>& var_types,
                                        const simplex_solver_settings_t<i_t, f_t>& settings,
                                        lp_problem_t<i_t, f_t>& lp)
{
  compute_activities(Arow, lp);
  for (i_t i = 0; i < lp.A.m; ++i) {
    queue_row(i, lp, settings.primal_tol);
  }
  return run_bound_propagation(Arow, var_types, settings, lp);
}

template <typename i_t, typename f_t>
bool domain_t<i_t, f_t>::propagate_from_variables(
  const csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<variable_type_t>& var_types,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  lp_problem_t<i_t, f_t>& lp,
  const std::vector<i_t>& vars)
{
  compute_activities(Arow, lp);
  for (i_t j : vars) {
    queue_variable(j, lp, settings.primal_tol);
  }
  return run_bound_propagation(Arow, var_types, settings, lp);
}

template <typename i_t, typename f_t>
bool domain_t<i_t, f_t>::propagate_from_stack(const csr_matrix_t<i_t, f_t>& Arow,
                                              const std::vector<variable_type_t>& var_types,
                                              const simplex_solver_settings_t<i_t, f_t>& settings,
                                              lp_problem_t<i_t, f_t>& lp,
                                              i_t start)
{
  for (size_t k = start; k < bound_changes.size(); ++k) {
    queue_variable(bound_changes[k].var, lp, settings.primal_tol);
  }

  return run_bound_propagation(Arow, var_types, settings, lp);
}

template <typename i_t, typename f_t>
bool domain_t<i_t, f_t>::apply_and_propagate(const csr_matrix_t<i_t, f_t>& Arow,
                                             const std::vector<variable_type_t>& var_types,
                                             const simplex_solver_settings_t<i_t, f_t>& settings,
                                             bound_change_t<i_t, f_t> bound_change,
                                             lp_problem_t<i_t, f_t>& lp)
{
  if (!apply(lp, bound_change)) {
    last_nnz_processed = nnz_processed;
    nnz_processed      = 0;
    return true;
  }

  queue_variable(bound_change.var, lp, settings.primal_tol);
  return run_bound_propagation(Arow, var_types, settings, lp);
}

template <typename i_t, typename f_t>
void domain_t<i_t, f_t>::backtrack_to_parent(lp_problem_t<i_t, f_t>& lp)
{
  bool popped_branch = false;
  while (!bound_changes.empty() && !popped_branch) {
    bound_change_t<i_t, f_t> bound_change = bound_changes.back();
    bound_changes.pop_back();
    popped_branch = bound_change.origin == bound_change_origin_t::BRANCH;

    lp.lower[bound_change.var] = bound_change.old_lower;
    lp.upper[bound_change.var] = bound_change.old_upper;

    std::swap(bound_change.old_lower, bound_change.new_lower);
    std::swap(bound_change.old_upper, bound_change.new_upper);
    update_activities(lp, bound_change);
  }
}

template <typename i_t, typename f_t>
bool domain_t<i_t, f_t>::run_bound_propagation(const csr_matrix_t<i_t, f_t>& Arow,
                                               const std::vector<variable_type_t>& var_types,
                                               const simplex_solver_settings_t<i_t, f_t>& settings,
                                               lp_problem_t<i_t, f_t>& lp)
{
  bool feasible = true;
  i_t iter      = 0;
  while (feasible && !row_queue.empty()) {
    i_t row         = row_queue.pop_front();
    row_queued[row] = false;
    ++iter;

    if (row_activities[row].recompute) { compute_row_activity(row, Arow, lp.lower, lp.upper); }
    row_activity_t activity = row_activities[row];
    const f_t max_a         = activity.max + activity.max_err;
    const f_t min_a         = activity.min + activity.min_err;
    f_t max_tol = settings.primal_tol * std::max({1.0, std::abs(max_a), std::abs(lp.rhs[row])});
    f_t min_tol = settings.primal_tol * std::max({1.0, std::abs(min_a), std::abs(lp.rhs[row])});

    if ((activity.max_inf == 0 && lp.rhs[row] - max_a > max_tol) ||
        (activity.min_inf == 0 && min_a - lp.rhs[row] > min_tol)) {
      settings.log.debug(
        "Iter:: %d, Infeasible constraint %d, rhs %e, min_a %e (%d inf), max_a %e (%d inf)\n",
        iter,
        row,
        lp.rhs[row],
        min_a,
        activity.min_inf,
        max_a,
        activity.max_inf);
      feasible = false;
      break;
    }

    // Smallest absolute change for which a new bound is accepted.
    const f_t tol             = settings.primal_tol;
    const f_t min_improvement = params.min_improvement_factor * tol;
    const f_t rhs             = lp.rhs[row];

    // A side of the row (a x <= rhs or a x >= rhs) is propagated only if it is not redundant and
    // either a single infinite contribution can be bounded or its slack is within the capacity
    // threshold.
    const bool propagate_upper =
      (activity.max_inf != 0 || max_a > rhs + tol) &&
      (activity.min_inf == 1 ||
       (activity.min_inf == 0 && rhs - min_a <= activity.capacity_threshold));
    const bool propagate_lower =
      (activity.min_inf != 0 || min_a < rhs - tol) &&
      (activity.max_inf == 1 ||
       (activity.max_inf == 0 && max_a - rhs <= activity.capacity_threshold));
    if (!propagate_upper && !propagate_lower) { continue; }

    i_t row_start = Arow.row_start[row];
    i_t row_end   = Arow.row_start[row + 1];
    nnz_processed += row_end - row_start;

    // The capacity threshold is recomputed from the bounds seen during this scan.
    f_t threshold = -tol;

    for (i_t p = row_start; p < row_end; ++p) {
      const i_t j    = Arow.j[p];
      const f_t a_ij = Arow.x[p];

      // A fixed variable cannot be tightened.
      const f_t lb = lp.lower[j];
      const f_t ub = lp.upper[j];
      if (lb == ub) { continue; }

      // Largest slack for which x_j can still receive an accepted bound.
      if (!std::isfinite(ub) || !std::isfinite(lb)) {
        threshold = inf;
      } else {
        f_t slack = ub - lb;

        if (var_types[j] == variable_type_t::INTEGER) {
          slack -= settings.integer_tol;
        } else {
          slack -= std::max(params.min_relative_improvement * slack, min_improvement);
        }

        threshold = std::max({threshold, std::abs(a_ij) * slack, tol});
      }

      // Read the activity again for each variable, since earlier tightenings in this row might have
      // already updated it, possibly marking it for recomputation.
      if (row_activities[row].recompute) { compute_row_activity(row, Arow, lp.lower, lp.upper); }
      const row_activity_t<i_t, f_t>& current = row_activities[row];

      // Contribution of x_j to the min/max activity, flagged as infinite the same way as in
      // compute_row_activity.
      const f_t bound_min = a_ij < 0 ? ub : lb;
      const f_t bound_max = a_ij > 0 ? ub : lb;
      const f_t alpha_min = a_ij * bound_min;
      const f_t alpha_max = a_ij * bound_max;
      const bool alpha_min_inf =
        !std::isfinite(alpha_min) || std::abs(alpha_min) >= params.huge_value;
      const bool alpha_max_inf =
        !std::isfinite(alpha_max) || std::abs(alpha_max) >= params.huge_value;

      f_t new_lb = lb;
      f_t new_ub = ub;

      // a_ij x_j <= rhs - (min activity of the other variables): an upper bound if a_ij > 0, a
      // lower bound otherwise. If x_j is the only infinite contributor, the residual is the finite
      // sum. The slack rhs - residual is formed with Dot2, since rhs and the residual often nearly
      // cancel.
      if (propagate_upper && (current.min_inf == 0 || (current.min_inf == 1 && alpha_min_inf))) {
        f_t slack     = rhs;
        f_t slack_err = -current.min_err;
        dot2_add(-1.0, current.min, slack, slack_err);
        if (current.min_inf == 0) { dot2_add(a_ij, bound_min, slack, slack_err); }
        const f_t gamma = (slack + slack_err) / a_ij;
        if (a_ij > 0) {
          new_ub = std::min(new_ub, gamma);
        } else {
          new_lb = std::max(new_lb, gamma);
        }
      }

      // a_ij x_j >= rhs - (max activity of the other variables): a lower bound if a_ij > 0, an
      // upper bound otherwise.
      if (propagate_lower && (current.max_inf == 0 || (current.max_inf == 1 && alpha_max_inf))) {
        f_t slack     = rhs;
        f_t slack_err = -current.max_err;
        dot2_add(-1.0, current.max, slack, slack_err);
        if (current.max_inf == 0) { dot2_add(a_ij, bound_max, slack, slack_err); }
        const f_t gamma = (slack + slack_err) / a_ij;
        if (a_ij > 0) {
          new_lb = std::max(new_lb, gamma);
        } else {
          new_ub = std::min(new_ub, gamma);
        }
      }

      feasible = tighten_bounds(
        j, new_lb, new_ub, bound_change_origin_t::BOUND_PROPAGATION, var_types, settings, lp);
      if (!feasible) { break; }
    }

    // An infeasible scan stops early and only saw part of the row.
    if (feasible) { row_activities[row].capacity_threshold = threshold; }
  }

  // Clear the rows left in the queue when infeasibility stops the propagation early.
  while (!row_queue.empty()) {
    row_queued[row_queue.pop_front()] = false;
  }

  last_nnz_processed = nnz_processed;
  nnz_processed      = 0;
  return feasible;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class domain_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::simplex
