/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <branch_and_bound/domain_propagation.hpp>

#include <presolve/conflict_graph/clique_table.cuh>
#include <presolve/probing_implied_bound.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>

namespace cuopt::mathematical_optimization::mip {

using simplex::lp_problem_t;
using simplex::simplex_solver_settings_t;
using simplex::variable_type_t;

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
void domain_propagation_t<i_t, f_t>::add_activity_term(activity_t<i_t, f_t>& side,
                                                       f_t aij,
                                                       f_t bound,
                                                       i_t sign) const
{
  const f_t alpha = aij * bound;
  if (std::isfinite(alpha) && std::abs(alpha) < huge_value) {
    dot2_add(sign * aij, bound, side.activity_sum, side.activity_sum_err);
  } else {
    side.num_infinite_bounds += sign;
  }
}

template <typename i_t, typename f_t>
void domain_propagation_t<i_t, f_t>::compute_row_activity(i_t i,
                                                          const csr_matrix_t<i_t, f_t>& Arow,
                                                          const std::vector<f_t>& lower,
                                                          const std::vector<f_t>& upper)
{
  const i_t row_start = Arow.row_start[i];
  const i_t row_end   = Arow.row_start[i + 1];
  nnz_processed_ += row_end - row_start;

  row_activity_t<i_t, f_t> activity;

  for (i_t p = row_start; p < row_end; ++p) {
    const i_t j   = Arow.j[p];
    const f_t aij = Arow.x[p];
    add_activity_term(activity.max_activity, aij, aij > 0 ? upper[j] : lower[j], 1);
    add_activity_term(activity.min_activity, aij, aij < 0 ? upper[j] : lower[j], 1);
  }

  activity.max_activity.activity_peak = std::abs(activity.max_activity.activity_sum);
  activity.min_activity.activity_peak = std::abs(activity.min_activity.activity_sum);
  row_activities_[i]                  = activity;
}

template <typename i_t, typename f_t>
void domain_propagation_t<i_t, f_t>::compute_activities(const csr_matrix_t<i_t, f_t>& Arow,
                                                        const lp_problem_t<i_t, f_t>& lp)
{
  row_activities_.resize(Arow.m);
  row_queued_.resize(Arow.m, false);
  row_queue_.clear_resize(std::max(Arow.m, 1));
  var_queued_.resize(lp.num_cols, false);
  var_queue_.clear_resize(std::max(lp.num_cols, 1));
  last_change_.resize(lp.num_cols, 0);

  for (i_t i = 0; i < Arow.m; ++i) {
    compute_row_activity(i, Arow, lp.lower, lp.upper);
  }

  objective_vars_.clear();
  objective_vars_.reserve(lp.num_cols);
  for (i_t j = 0; j < lp.num_cols; ++j) {
    if (lp.objective[j] != 0) { objective_vars_.push_back(j); }
  }
}

template <typename i_t, typename f_t>
void domain_propagation_t<i_t, f_t>::update_activities(const lp_problem_t<i_t, f_t>& lp,
                                                       const bound_change_t<i_t, f_t>& bound_change)
{
  const i_t col_start = lp.A.col_start[bound_change.var];
  const i_t col_end   = lp.A.col_start[bound_change.var + 1];
  nnz_processed_ += col_end - col_start;

  for (i_t p = col_start; p < col_end; ++p) {
    i_t i                              = lp.A.i[p];
    row_activity_t<i_t, f_t>& activity = row_activities_[i];

    // A wider range can raise the capacity threshold of the rows, so reset it until the row is
    // propagated again.
    if (bound_change.new_upper - bound_change.new_lower >
        bound_change.old_upper - bound_change.old_lower) {
      activity.capacity_threshold = inf;
    }

    // A row waiting for recomputation is rebuilt from the current bounds anyway.
    if (activity.recompute) { continue; }

    const f_t aij           = lp.A.x[p];
    const f_t old_bound_max = aij > 0 ? bound_change.old_upper : bound_change.old_lower;
    const f_t old_bound_min = aij < 0 ? bound_change.old_upper : bound_change.old_lower;
    const f_t new_bound_max = aij > 0 ? bound_change.new_upper : bound_change.new_lower;
    const f_t new_bound_min = aij < 0 ? bound_change.new_upper : bound_change.new_lower;

    // Replace the old contribution of x_j with the new one, moving it between the finite sum and
    // the infinite count when needed.
    if (aij * old_bound_max != aij * new_bound_max) {
      add_activity_term(activity.max_activity, aij, old_bound_max, -1);
      add_activity_term(activity.max_activity, aij, new_bound_max, 1);
      activity.max_activity.activity_peak =
        std::max(activity.max_activity.activity_peak, std::abs(activity.max_activity.activity_sum));
    }

    if (aij * old_bound_min != aij * new_bound_min) {
      add_activity_term(activity.min_activity, aij, old_bound_min, -1);
      add_activity_term(activity.min_activity, aij, new_bound_min, 1);
      activity.min_activity.activity_peak =
        std::max(activity.min_activity.activity_peak, std::abs(activity.min_activity.activity_sum));
    }

    // An activity much smaller than its peak has lost precision to numerical cancellations, so mark
    // the row to be recomputed.
    if (activity.max_activity.activity_peak >
          recompute_factor * std::max<f_t>(std::abs(activity.max_activity.activity_sum), 1.0) ||
        activity.min_activity.activity_peak >
          recompute_factor * std::max<f_t>(std::abs(activity.min_activity.activity_sum), 1.0)) {
      activity.recompute = true;
    }
  }
}

template <typename i_t, typename f_t>
void domain_propagation_t<i_t, f_t>::queue_row(i_t i, const lp_problem_t<i_t, f_t>& lp, f_t tol)
{
  if (row_queued_[i]) { return; }

  const row_activity_t<i_t, f_t>& activity = row_activities_[i];
  if (!activity.recompute) {
    const f_t max_a = activity.max_activity.activity_sum + activity.max_activity.activity_sum_err;
    const f_t min_a = activity.min_activity.activity_sum + activity.min_activity.activity_sum_err;
    const f_t rhs   = lp.rhs[i];

    const bool propagate_upper =
      (activity.max_activity.num_infinite_bounds != 0 || max_a > rhs + tol) &&
      (activity.min_activity.num_infinite_bounds == 1 ||
       (activity.min_activity.num_infinite_bounds == 0 &&
        rhs - min_a <= activity.capacity_threshold));
    const bool propagate_lower =
      (activity.min_activity.num_infinite_bounds != 0 || min_a < rhs - tol) &&
      (activity.max_activity.num_infinite_bounds == 1 ||
       (activity.max_activity.num_infinite_bounds == 0 &&
        max_a - rhs <= activity.capacity_threshold));
    if (!propagate_upper && !propagate_lower) { return; }
  }

  row_queue_.push_back(i);
  row_queued_[i] = true;
}

template <typename i_t, typename f_t>
void domain_propagation_t<i_t, f_t>::queue_variable(i_t j,
                                                    const std::vector<variable_type_t>& var_types,
                                                    const lp_problem_t<i_t, f_t>& lp,
                                                    f_t tol)
{
  const i_t col_start = lp.A.col_start[j];
  const i_t col_end   = lp.A.col_start[j + 1];
  nnz_processed_ += col_end - col_start;
  for (i_t p = col_start; p < col_end; ++p) {
    queue_row(lp.A.i[p], lp, tol);
  }

  if (implied_bounds == nullptr && clique_table == nullptr) { return; }
  const f_t value = lp.lower[j];
  if (!var_queued_[j] && var_types[j] == variable_type_t::INTEGER && value == lp.upper[j] &&
      (value == 0 || value == 1)) {
    var_queue_.push_back(j);
    var_queued_[j] = true;
  }
}

template <typename i_t, typename f_t>
bool domain_propagation_t<i_t, f_t>::tighten_bounds(
  i_t j,
  f_t new_lower,
  f_t new_upper,
  bound_change_origin_t origin,
  const std::vector<variable_type_t>& var_types,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  lp_problem_t<i_t, f_t>& lp)
{
  const f_t tol             = settings.primal_tol;
  const f_t min_improvement = min_improvement_factor * tol;
  const f_t lb              = lp.lower[j];
  const f_t ub              = lp.upper[j];

  bool tighten_lb = false;
  bool tighten_ub = false;

  if (var_types[j] == variable_type_t::INTEGER) {
    new_lower = std::ceil(new_lower - settings.integer_tol);
    new_upper = std::floor(new_upper + settings.integer_tol);
    // The step must also be a fraction of the bound magnitude, so that a cycle of small steps
    // cannot tighten an integer bound forever.
    tighten_lb =
      new_lower - lb > std::max(settings.integer_tol, min_improvement * std::abs(new_lower));
    tighten_ub =
      ub - new_upper > std::max(settings.integer_tol, min_improvement * std::abs(new_upper));
  } else if (lp.A.col_start[j + 1] - lp.A.col_start[j] == 1) {
    // A continuous column singleton (e.g., a slack) is only bounded, never tightened: its
    // finite bounds are implied by its only row and only make the LP harder to solve.
    tighten_lb = lb == -inf && new_lower > -inf;
    tighten_ub = ub == inf && new_upper < inf;
  } else {
    const f_t lb_range = ub < inf ? ub - lb : std::max(std::abs(lb), std::abs(new_lower));
    const f_t ub_range = lb > -inf ? ub - lb : std::max(std::abs(ub), std::abs(new_upper));
    tighten_lb         = new_lower - min_improvement > lb &&
                 (lb == -inf || new_lower - lb >= min_relative_improvement * lb_range);
    tighten_ub = new_upper + min_improvement < ub &&
                 (ub == inf || ub - new_upper >= min_relative_improvement * ub_range);
  }

  if (!tighten_lb && !tighten_ub) { return true; }
  if (!tighten_lb) { new_lower = lb; }
  if (!tighten_ub) { new_upper = ub; }

  // Bounds crossing by less than the tolerance are snapped together.
  if (new_lower > new_upper) {
    if (new_lower - new_upper > tol * std::max({1.0, std::abs(new_lower), std::abs(new_upper)})) {
      settings.log.debug(
        "Infeasible variable %d after applying a propagation, %e > %e\n", j, new_lower, new_upper);
      return false;
    }
    if (tighten_ub) {
      new_upper = new_lower;
    } else {
      new_lower = new_upper;
    }
  }

  // Count the number of tightening the variable x_j has. It uses the stack to keep the track
  // of the count in O(1) time. Changes before the start of the bound propagation are ignored.
  const size_t last = last_change_[j];
  i_t count         = 1;
  if (last >= propagation_start_ && last < bound_changes_.size() && bound_changes_[last].var == j) {
    count = bound_changes_[last].count + 1;
  }

  // If it surpasses the maximum allowed, then do not queue this variable again.
  // This prevents the bound propagation to enter into a cycle.
  if (count > max_var_tightenings_per_call) { return true; }

  if (apply(lp,
            {.var       = j,
             .new_upper = new_upper,
             .new_lower = new_lower,
             .origin    = origin,
             .count     = count})) {
    last_change_[j] = bound_changes_.size() - 1;
  }

  return true;
}

template <typename i_t, typename f_t>
bool domain_propagation_t<i_t, f_t>::propagate_implications(
  i_t j,
  const std::vector<variable_type_t>& var_types,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  lp_problem_t<i_t, f_t>& lp)
{
  assert(lp.lower[j] == lp.upper[j] && "Implications require a fixed variable");
  const bool one = lp.lower[j] == 1;

  // Probing x_j = 0 (resp. 1) implies lower[p] <= x_k <= upper[p] for every entry p of x_j.
  if (implied_bounds != nullptr && j + 1 < implied_bounds->zero_offsets.size()) {
    const auto& offsets   = one ? implied_bounds->one_offsets : implied_bounds->zero_offsets;
    const auto& variables = one ? implied_bounds->one_variables : implied_bounds->zero_variables;
    const auto& lower = one ? implied_bounds->one_lower_bound : implied_bounds->zero_lower_bound;
    const auto& upper = one ? implied_bounds->one_upper_bound : implied_bounds->zero_upper_bound;
    nnz_processed_ += offsets[j + 1] - offsets[j];

    for (i_t p = offsets[j]; p < offsets[j + 1]; ++p) {
      const i_t k = variables[p];
      if (k == j || k >= lp.num_cols) { continue; }
      if (!tighten_bounds(k,
                          lower[p],
                          upper[p],
                          bound_change_origin_t::BOUND_IMPLICATION,
                          var_types,
                          settings,
                          lp)) {
        return false;
      }
    }
  }

  if (clique_table == nullptr || !clique_table->ready.load(std::memory_order_acquire)) {
    return true;
  }

  // Vertex v < n is the literal x_v and v >= n its complement 1 - x_{v - n}. At most one literal
  // of a clique is true, so every neighbour of the true literal of x_j is false.
  const i_t n = clique_table->n_variables;
  if (j >= n) { return true; }
  const i_t literal = one ? j : j + n;

  for (i_t v : clique_table->get_adj_set_of_var(literal)) {
    ++nnz_processed_;
    const i_t k         = v < n ? v : v - n;
    const f_t new_lower = v < n ? lp.lower[k] : 1.0;
    const f_t new_upper = v < n ? 0.0 : lp.upper[k];
    if (!tighten_bounds(
          k, new_lower, new_upper, bound_change_origin_t::CLIQUE, var_types, settings, lp)) {
      return false;
    }
  }

  return true;
}

template <typename i_t, typename f_t>
bool domain_propagation_t<i_t, f_t>::propagate_full(
  const csr_matrix_t<i_t, f_t>& Arow,
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
bool domain_propagation_t<i_t, f_t>::propagate_from_variables(
  const csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<variable_type_t>& var_types,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  lp_problem_t<i_t, f_t>& lp,
  const std::vector<i_t>& vars)
{
  compute_activities(Arow, lp);
  for (i_t j : vars) {
    queue_variable(j, var_types, lp, settings.primal_tol);
  }
  return run_bound_propagation(Arow, var_types, settings, lp);
}

template <typename i_t, typename f_t>
bool domain_propagation_t<i_t, f_t>::propagate_from_stack(
  const csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<variable_type_t>& var_types,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  lp_problem_t<i_t, f_t>& lp,
  i_t start)
{
  for (size_t k = start; k < bound_changes_.size(); ++k) {
    queue_variable(bound_changes_[k].var, var_types, lp, settings.primal_tol);
  }

  return run_bound_propagation(Arow, var_types, settings, lp);
}

template <typename i_t, typename f_t>
bool domain_propagation_t<i_t, f_t>::apply_and_propagate(
  const csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<variable_type_t>& var_types,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  bound_change_t<i_t, f_t> bound_change,
  lp_problem_t<i_t, f_t>& lp)
{
  if (!apply(lp, bound_change)) {
    last_nnz_processed = nnz_processed_;
    nnz_processed_     = 0;
    return true;
  }

  queue_variable(bound_change.var, var_types, lp, settings.primal_tol);
  return run_bound_propagation(Arow, var_types, settings, lp);
}

template <typename i_t, typename f_t>
bool domain_propagation_t<i_t, f_t>::propagate_objective(
  const csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<variable_type_t>& var_types,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  lp_problem_t<i_t, f_t>& lp,
  f_t cutoff)
{
  if (!std::isfinite(cutoff) || objective_vars_.empty()) { return true; }

  // The work is added to the one of the preceding propagation, so the callers record both.
  const size_t prior_nnz = last_nnz_processed;
  size_t work            = 0;
  const f_t rhs          = cutoff + settings.primal_tol * std::max<f_t>(1.0, std::abs(cutoff));
  bool feasible          = true;

  for (i_t round = 0; round < max_objective_rounds; ++round) {
    // Minimum objective over the bounds in lp, with the infinite contributions flagged as in
    // compute_row_activity.
    f_t min     = 0;
    f_t min_err = 0;
    i_t min_inf = 0;
    for (i_t j : objective_vars_) {
      const f_t c_j       = lp.objective[j];
      const f_t bound_min = c_j < 0 ? lp.upper[j] : lp.lower[j];
      const f_t alpha_min = c_j * bound_min;
      if (std::isfinite(alpha_min) && std::abs(alpha_min) < huge_value) {
        dot2_add(c_j, bound_min, min, min_err);
      } else {
        ++min_inf;
      }
    }
    work += objective_vars_.size();

    if (min_inf == 0 && min + min_err > rhs) {
      feasible = false;
      break;
    }
    if (min_inf > 1) { break; }

    const size_t num_changes = bound_changes_.size();
    for (i_t j : objective_vars_) {
      const f_t lb = lp.lower[j];
      const f_t ub = lp.upper[j];
      if (lb == ub) { continue; }

      const f_t c_j            = lp.objective[j];
      const f_t bound_min      = c_j < 0 ? ub : lb;
      const f_t alpha_min      = c_j * bound_min;
      const bool alpha_min_inf = !std::isfinite(alpha_min) || std::abs(alpha_min) >= huge_value;

      // With a single infinite contribution, only its variable can be bounded.
      if (min_inf == 1 && !alpha_min_inf) { continue; }

      // c_j x_j <= rhs - (minimum objective of the other variables): an upper bound if c_j > 0, a
      // lower bound otherwise. The slack is formed with Dot2, as in the row propagation.
      f_t slack     = rhs;
      f_t slack_err = -min_err;
      dot2_add(-1.0, min, slack, slack_err);
      if (min_inf == 0) { dot2_add(c_j, bound_min, slack, slack_err); }
      const f_t gamma     = (slack + slack_err) / c_j;
      const f_t new_lower = c_j > 0 ? lb : gamma;
      const f_t new_upper = c_j > 0 ? gamma : ub;
      if (!tighten_bounds(
            j, new_lower, new_upper, bound_change_origin_t::OBJECTIVE, var_types, settings, lp)) {
        feasible = false;
        break;
      }
    }
    work += objective_vars_.size();

    if (!feasible || bound_changes_.size() == num_changes) { break; }

    // Propagate the rows of the tightened variables, which may raise the minimum objective.
    feasible = run_bound_propagation(Arow, var_types, settings, lp);
    work += last_nnz_processed;
    if (!feasible) { break; }
  }

  if (!feasible) { clear_queues(); }

  work += nnz_processed_;
  nnz_processed_     = 0;
  last_nnz_processed = prior_nnz + work;
  return feasible;
}

template <typename i_t, typename f_t>
void domain_propagation_t<i_t, f_t>::backtrack_to_parent(lp_problem_t<i_t, f_t>& lp)
{
  bool popped_branch = false;
  while (!bound_changes_.empty() && !popped_branch) {
    bound_change_t<i_t, f_t> bound_change = bound_changes_.back();
    bound_changes_.pop_back();
    popped_branch = bound_change.origin == bound_change_origin_t::BRANCH;

    lp.lower[bound_change.var] = bound_change.old_lower;
    lp.upper[bound_change.var] = bound_change.old_upper;

    std::swap(bound_change.old_lower, bound_change.new_lower);
    std::swap(bound_change.old_upper, bound_change.new_upper);
    update_activities(lp, bound_change);
  }
}

template <typename i_t, typename f_t>
i_t domain_propagation_t<i_t, f_t>::count_fixed_integers(
  const std::vector<variable_type_t>& var_types,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  size_t start) const
{
  i_t num_fixed = 0;
  for (size_t k = start; k < bound_changes_.size(); ++k) {
    const bound_change_t<i_t, f_t>& change = bound_changes_[k];
    if (var_types[change.var] != variable_type_t::CONTINUOUS &&
        change.old_upper - change.old_lower > settings.fixed_tol &&
        change.new_upper - change.new_lower <= settings.fixed_tol) {
      ++num_fixed;
    }
  }
  return num_fixed;
}

template <typename i_t, typename f_t>
bool domain_propagation_t<i_t, f_t>::run_bound_propagation(
  const csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<variable_type_t>& var_types,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  lp_problem_t<i_t, f_t>& lp)
{
  bool feasible      = true;
  i_t iter           = 0;
  propagation_start_ = bound_changes_.size();
  while (feasible && (!row_queue_.empty() || !var_queue_.empty())) {
    // Apply the implications of the fixed variables before propagating the rows.
    if (!var_queue_.empty()) {
      const i_t j    = var_queue_.pop_front();
      var_queued_[j] = false;
      feasible       = propagate_implications(j, var_types, settings, lp);
      continue;
    }
    i_t i          = row_queue_.pop_front();
    row_queued_[i] = false;
    ++iter;

    if (row_activities_[i].recompute) { compute_row_activity(i, Arow, lp.lower, lp.upper); }
    row_activity_t activity = row_activities_[i];
    const f_t max_a = activity.max_activity.activity_sum + activity.max_activity.activity_sum_err;
    const f_t min_a = activity.min_activity.activity_sum + activity.min_activity.activity_sum_err;
    f_t max_tol     = settings.primal_tol * std::max({1.0, std::abs(max_a), std::abs(lp.rhs[i])});
    f_t min_tol     = settings.primal_tol * std::max({1.0, std::abs(min_a), std::abs(lp.rhs[i])});

    // a x = rhs is infeasible if rhs lies outside [min_a, max_a] beyond the tolerance.
    if ((activity.max_activity.num_infinite_bounds == 0 && lp.rhs[i] - max_a > max_tol) ||
        (activity.min_activity.num_infinite_bounds == 0 && min_a - lp.rhs[i] > min_tol)) {
      settings.log.debug(
        "Iter:: %d, Infeasible constraint %d, rhs %e, min_a %e (%d inf), max_a %e (%d inf)\n",
        iter,
        i,
        lp.rhs[i],
        min_a,
        activity.min_activity.num_infinite_bounds,
        max_a,
        activity.max_activity.num_infinite_bounds);
      feasible = false;
      break;
    }

    // Smallest absolute change for which a new bound is accepted.
    const f_t tol             = settings.primal_tol;
    const f_t min_improvement = min_improvement_factor * tol;
    const f_t rhs             = lp.rhs[i];

    // A side of the row (a x <= rhs or a x >= rhs) is propagated only if it is not redundant and
    // either a single infinite contribution can be bounded or its slack is within the capacity
    // threshold.
    const bool propagate_upper =
      (activity.max_activity.num_infinite_bounds != 0 || max_a > rhs + tol) &&
      (activity.min_activity.num_infinite_bounds == 1 ||
       (activity.min_activity.num_infinite_bounds == 0 &&
        rhs - min_a <= activity.capacity_threshold));
    const bool propagate_lower =
      (activity.min_activity.num_infinite_bounds != 0 || min_a < rhs - tol) &&
      (activity.max_activity.num_infinite_bounds == 1 ||
       (activity.max_activity.num_infinite_bounds == 0 &&
        max_a - rhs <= activity.capacity_threshold));
    if (!propagate_upper && !propagate_lower) { continue; }

    const i_t row_start = Arow.row_start[i];
    const i_t row_end   = Arow.row_start[i + 1];
    nnz_processed_ += row_end - row_start;

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
          slack -= std::max(min_relative_improvement * slack, min_improvement);
        }

        threshold = std::max({threshold, std::abs(a_ij) * slack, tol});
      }

      // Read the activity again for each variable, since earlier tightenings in this row might have
      // already updated it, possibly marking it for recomputation.
      if (row_activities_[i].recompute) { compute_row_activity(i, Arow, lp.lower, lp.upper); }
      const row_activity_t<i_t, f_t>& current = row_activities_[i];

      // Activities of the other variables of the row, without the contribution of x_j.
      activity_t<i_t, f_t> rest_min = current.min_activity;
      activity_t<i_t, f_t> rest_max = current.max_activity;
      add_activity_term(rest_min, a_ij, a_ij < 0 ? ub : lb, -1);
      add_activity_term(rest_max, a_ij, a_ij > 0 ? ub : lb, -1);

      f_t new_lb = lb;
      f_t new_ub = ub;

      // a_ij x_j <= rhs - (min activity of the other variables): an upper bound if a_ij > 0, a
      // lower bound otherwise. The slack is formed with Dot2, since rhs and the residual often
      // nearly cancel.
      if (propagate_upper && rest_min.num_infinite_bounds == 0) {
        f_t slack     = rhs;
        f_t slack_err = -rest_min.activity_sum_err;
        dot2_add(-1.0, rest_min.activity_sum, slack, slack_err);
        const f_t gamma = (slack + slack_err) / a_ij;
        if (a_ij > 0) {
          new_ub = std::min(new_ub, gamma);
        } else {
          new_lb = std::max(new_lb, gamma);
        }
      }

      // a_ij x_j >= rhs - (max activity of the other variables): a lower bound if a_ij > 0, an
      // upper bound otherwise.
      if (propagate_lower && rest_max.num_infinite_bounds == 0) {
        f_t slack     = rhs;
        f_t slack_err = -rest_max.activity_sum_err;
        dot2_add(-1.0, rest_max.activity_sum, slack, slack_err);
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
      if (lp.lower[j] != lb || lp.upper[j] != ub) { queue_variable(j, var_types, lp, tol); }
    }

    // An infeasible scan stops early and only saw part of the row.
    if (feasible) { row_activities_[i].capacity_threshold = threshold; }
  }

  // Clear pending work when infeasibility stops propagation early.
  clear_queues();

  last_nnz_processed = nnz_processed_;
  nnz_processed_     = 0;
  return feasible;
}

template <typename i_t, typename f_t>
void domain_propagation_t<i_t, f_t>::clear_queues()
{
  while (!row_queue_.empty()) {
    row_queued_[row_queue_.pop_front()] = false;
  }
  while (!var_queue_.empty()) {
    var_queued_[var_queue_.pop_front()] = false;
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class domain_propagation_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
