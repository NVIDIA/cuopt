/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <utilities/circular_deque.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace cuopt::mathematical_optimization::mip {
template <typename i_t, typename f_t>
struct probing_implied_bound_t;
template <typename i_t, typename f_t>
struct clique_table_t;
}  // namespace cuopt::mathematical_optimization::mip

namespace cuopt::mathematical_optimization::simplex {

enum class bound_change_origin_t {
  BRANCH            = 0,  // Regular variable branching
  BOUND_PROPAGATION = 1,  // Bound propagation
  CUT               = 2,  // Cut
  BOUND_IMPLICATION = 3,  // Derived from the implied bounds (via probing cache)
  CLIQUE            = 4,  // Derived from cliques
  OBJECTIVE         = 5,  // Objective cutoff
  REDUCED_COST      = 6,  // Reduced cost strengthening
  SYMMETRY          = 7
};

template <typename i_t, typename f_t>
struct bound_change_t {
  i_t var;
  f_t old_upper;
  f_t new_upper;
  f_t old_lower;
  f_t new_lower;
  bound_change_origin_t origin;

  void apply(std::vector<f_t>& lower, std::vector<f_t>& upper)
  {
    old_lower  = lower[var];
    lower[var] = new_lower;
    old_upper  = upper[var];
    upper[var] = new_upper;
  }
};

struct domain_params {
  double huge_value               = 1e15;
  double recompute_factor         = 1e6;
  double min_relative_improvement = 0.3;
  double min_improvement_factor   = 1e3;
  // Derived bounds larger than this in magnitude are discarded
  double max_derived_bound = 1e8;
};

// Cumulative statistics of the bound propagation performed by a domain. A round is one pass over
// the rows queued when it starts. propagation_time includes the activity updates of its own
// tightenings, which are also counted in update_activities_time. The probing and clique counters
// cover the implications of fixed integer variables, and their times are included in
// propagation_time.
struct domain_stats_t {
  int64_t calls               = 0;
  int64_t infeasible          = 0;
  int64_t rounds              = 0;
  int64_t max_rounds          = 0;
  int64_t rows_processed      = 0;
  int64_t rows_scanned        = 0;
  int64_t applied_changes     = 0;
  int64_t tightenings         = 0;
  int64_t fixings             = 0;
  int64_t activity_updates    = 0;
  int64_t activity_recomputes = 0;
  int64_t nnz_processed       = 0;

  int64_t implication_calls    = 0;
  int64_t probing_implications = 0;
  int64_t probing_tightenings  = 0;
  int64_t probing_fixings      = 0;
  int64_t probing_infeasible   = 0;
  int64_t clique_implications  = 0;
  int64_t clique_tightenings   = 0;
  int64_t clique_fixings       = 0;
  int64_t clique_infeasible    = 0;

  double propagation_time        = 0.0;
  double compute_activities_time = 0.0;
  double update_activities_time  = 0.0;
  double probing_time            = 0.0;
  double clique_time             = 0.0;

  domain_stats_t& operator+=(const domain_stats_t& other)
  {
    calls += other.calls;
    infeasible += other.infeasible;
    rounds += other.rounds;
    max_rounds = std::max(max_rounds, other.max_rounds);
    rows_processed += other.rows_processed;
    rows_scanned += other.rows_scanned;
    applied_changes += other.applied_changes;
    tightenings += other.tightenings;
    fixings += other.fixings;
    activity_updates += other.activity_updates;
    activity_recomputes += other.activity_recomputes;
    nnz_processed += other.nnz_processed;
    implication_calls += other.implication_calls;
    probing_implications += other.probing_implications;
    probing_tightenings += other.probing_tightenings;
    probing_fixings += other.probing_fixings;
    probing_infeasible += other.probing_infeasible;
    clique_implications += other.clique_implications;
    clique_tightenings += other.clique_tightenings;
    clique_fixings += other.clique_fixings;
    clique_infeasible += other.clique_infeasible;
    propagation_time += other.propagation_time;
    compute_activities_time += other.compute_activities_time;
    update_activities_time += other.update_activities_time;
    probing_time += other.probing_time;
    clique_time += other.clique_time;
    return *this;
  }
};

// The finite parts of the activities are accumulated with compensated (Dot2) summation: the
// activity is max + max_err (resp. min + min_err), where the error terms collect the rounding of
// every product and addition.
template <typename i_t, typename f_t>
struct row_activity_t {
  f_t max      = 0;
  f_t max_err  = 0;
  f_t max_peak = 0;
  i_t max_inf  = 0;

  f_t min      = 0;
  f_t min_err  = 0;
  f_t min_peak = 0;
  i_t min_inf  = 0;

  // Largest slack for which some variable can still receive an accepted bound.
  f_t capacity_threshold = std::numeric_limits<f_t>::infinity();

  bool recompute = false;
};

// This object stores the row activities over the LP bounds as well as the stack of bound changes
// that produced those bounds, in the order they were applied.
// This is used for bound propagation and backtracking.
template <typename i_t, typename f_t>
class domain_t {
 public:
  domain_t() = default;

  // Records a bound change on the stack.
  void push(const bound_change_t<i_t, f_t>& bound_change) { bound_changes.push_back(bound_change); }

  // Apply the `bound_change` to `lp`, store in the stack and then update the activities.
  bool apply(lp_problem_t<i_t, f_t>& lp, bound_change_t<i_t, f_t> bound_change)
  {
    if (bound_change.var < 0) { return false; }

    bound_change.apply(lp.lower, lp.upper);
    if (bound_change.old_lower == bound_change.new_lower &&
        bound_change.old_upper == bound_change.new_upper) {
      return false;
    }

    push(bound_change);
    update_activities(lp, bound_change);
    return true;
  }

  const bound_change_t<i_t, f_t>& operator[](size_t k) const { return bound_changes[k]; }
  size_t size() const { return bound_changes.size(); }
  bool is_empty() const { return bound_changes.empty(); }
  void clear() { bound_changes.clear(); }

  // Applies every recorded change to [lower, upper], in order, refreshing their old bounds.
  void apply_stack(std::vector<f_t>& lower, std::vector<f_t>& upper)
  {
    for (auto& bound_change : bound_changes) {
      bound_change.apply(lower, upper);
    }
  }

  // Pops the changes back to and including the last branching, restoring their bounds in lp and
  // reverting their activities.
  void backtrack_to_parent(lp_problem_t<i_t, f_t>& lp);

  // Computes the activities of all rows from the bounds in lp.
  void compute_activities(const csr_matrix_t<i_t, f_t>& Arow, const lp_problem_t<i_t, f_t>& lp);

  // Incrementally updates the activities of every row containing bound_change.var after its bounds
  // changed from [old_lower, old_upper] to [new_lower, new_upper].
  void update_activities(const lp_problem_t<i_t, f_t>& lp,
                         const bound_change_t<i_t, f_t>& bound_change);

  // Recomputes all activities from the bounds in lp and propagates every row, tightening the bounds
  // in place.
  bool propagate_full(const csr_matrix_t<i_t, f_t>& Arow,
                      const std::vector<variable_type_t>& var_types,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      lp_problem_t<i_t, f_t>& lp);

  // Recomputes all activities from the bounds in lp and propagates from the rows containing vars.
  bool propagate_from_variables(const csr_matrix_t<i_t, f_t>& Arow,
                                const std::vector<variable_type_t>& var_types,
                                const simplex_solver_settings_t<i_t, f_t>& settings,
                                lp_problem_t<i_t, f_t>& lp,
                                const std::vector<i_t>& vars);

  // Propagates from the rows containing the variables of the bound changes from start onwards. The
  // activities must already include these changes, as apply does.
  bool propagate_from_stack(const csr_matrix_t<i_t, f_t>& Arow,
                            const std::vector<variable_type_t>& var_types,
                            const simplex_solver_settings_t<i_t, f_t>& settings,
                            lp_problem_t<i_t, f_t>& lp,
                            i_t start = 0);

  // Applies bound_change to lp and propagates from the rows containing its variable. A change that
  // does not move any bound is neither recorded nor propagated.
  bool apply_and_propagate(const csr_matrix_t<i_t, f_t>& Arow,
                           const std::vector<variable_type_t>& var_types,
                           const simplex_solver_settings_t<i_t, f_t>& settings,
                           bound_change_t<i_t, f_t> bound_change,
                           lp_problem_t<i_t, f_t>& lp);

  size_t last_nnz_processed{0};

  // Implications applied when an integer variable is fixed to 0 or 1. Either may be null, and the
  // clique table is skipped until it is ready.
  const mip::probing_implied_bound_t<i_t, f_t>* implied_bounds = nullptr;
  const mip::clique_table_t<i_t, f_t>* clique_table            = nullptr;

  domain_stats_t stats;

 private:
  domain_params params;

  std::vector<row_activity_t<i_t, f_t>> row_activities;
  std::vector<uint8_t> row_queued;
  circular_deque_t<i_t> row_queue;

  std::vector<uint8_t> var_queued;
  circular_deque_t<i_t> var_queue;

  std::vector<bound_change_t<i_t, f_t>> bound_changes;

  size_t nnz_processed{0};

  // Queues row i unless it is already queued or neither of its sides can tighten a bound. A row
  // awaiting recomputation is always queued.
  void queue_row(i_t i, const lp_problem_t<i_t, f_t>& lp, f_t tol);

  // Queues the rows containing x_j and, if x_j is an integer fixed to 0 or 1, x_j itself so its
  // implications are applied.
  void queue_variable(i_t j,
                      const std::vector<variable_type_t>& var_types,
                      const lp_problem_t<i_t, f_t>& lp,
                      f_t tol);

  // Applies the probing and clique implications of x_j at its fixed value. Returns false if an
  // implied bound proves infeasibility.
  bool propagate_implications(i_t j,
                              const std::vector<variable_type_t>& var_types,
                              const simplex_solver_settings_t<i_t, f_t>& settings,
                              lp_problem_t<i_t, f_t>& lp);

  // Intersects the bounds of x_k with [new_lower, new_upper], applying and queueing x_k if they
  // improve. Returns false if the bounds cross.
  bool tighten_bounds(i_t k,
                      f_t new_lower,
                      f_t new_upper,
                      bound_change_origin_t origin,
                      const std::vector<variable_type_t>& var_types,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      lp_problem_t<i_t, f_t>& lp);

  // Propagates the queued rows in FIFO order, applying each accepted bound immediately
  // and queueing the rows it affects. Returns false if a row or a variable proves infeasibility.
  bool run_bound_propagation(const csr_matrix_t<i_t, f_t>& Arow,
                             const std::vector<variable_type_t>& var_types,
                             const simplex_solver_settings_t<i_t, f_t>& settings,
                             lp_problem_t<i_t, f_t>& lp);

  // Recomputes the activity of row i from the bounds in [lower, upper].
  void compute_row_activity(i_t i,
                            const csr_matrix_t<i_t, f_t>& Arow,
                            const std::vector<f_t>& lower,
                            const std::vector<f_t>& upper);
};

template <typename i_t, typename f_t>
bool full_bound_strengthening(const csr_matrix_t<i_t, f_t>& Arow,
                              const std::vector<variable_type_t>& var_types,
                              const simplex_solver_settings_t<i_t, f_t>& settings,
                              lp_problem_t<i_t, f_t>& lp,
                              domain_stats_t* stats = nullptr)
{
  domain_t<i_t, f_t> domain;
  const bool feasible = domain.propagate_full(Arow, var_types, settings, lp);
  if (stats != nullptr) { *stats += domain.stats; }
  return feasible;
}

}  // namespace cuopt::mathematical_optimization::simplex
