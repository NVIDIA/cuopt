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

// Where a bound change came from. backtrack_to_parent pops changes until it reaches a BRANCH.
enum class bound_change_origin_t {
  BRANCH            = 0,  // Regular variable branching
  BOUND_PROPAGATION = 1,  // Bound propagation
  BOUND_IMPLICATION = 2,  // Derived from the implied bounds (via probing cache)
  CLIQUE            = 3,  // Derived from cliques
  OBJECTIVE         = 4,  // Objective cutoff
  SYMMETRY          = 5   // Orbital fixing
};

// old_lower/old_upper are filled in by domain_propagation_t::apply (or apply_stack).
template <typename i_t, typename f_t>
struct bound_change_t {
  i_t var;
  f_t old_upper;
  f_t new_upper;
  f_t old_lower;
  f_t new_lower;
  bound_change_origin_t origin;

  // Tightenings of var by the propagation call that recorded this change, including this one.
  i_t count = 0;
};

// One side (min or max) of a row activity. The finite part is accumulated with compensated (Dot2)
// summation: the activity is activity_sum + activity_sum_err, where activity_sum_err collects the
// rounding of every product and addition. Infinite or huge contributions are counted in
// num_infinite_bounds instead.
template <typename i_t, typename f_t>
struct activity_t {
  f_t activity_sum     = 0;
  f_t activity_sum_err = 0;

  // Largest |activity_sum| since the last full computation, used by update_activities to detect
  // numerical cancellation.
  f_t activity_peak = 0;

  // Number of variables with an infinite contribution, kept out of activity_sum. If a row contains
  // a single such variable, the residual activity is infinite for every other variable but finite
  // for that one. With two or more, every residual activity is infinite.
  i_t num_infinite_bounds = 0;
};

template <typename i_t, typename f_t>
struct row_activity_t {
  activity_t<i_t, f_t> min_activity;
  activity_t<i_t, f_t> max_activity;

  // Largest slack for which some variable can still receive an accepted bound. A side whose slack
  // exceeds it cannot tighten any bound, so the row is not queued.
  f_t capacity_threshold = std::numeric_limits<f_t>::infinity();

  // Set when cancellation is detected; the activity is rebuilt from the bounds before the row is
  // used again.
  bool recompute = false;
};

// This object stores the row activities over the LP bounds as well as the stack of bound changes
// that produced those bounds, in the order they were applied.
// This is used for bound propagation and backtracking. Every row is treated as an equality
// a x = rhs; inequalities must carry a slack column.
template <typename i_t, typename f_t>
class domain_propagation_t {
 public:
  domain_propagation_t() = default;

  // Records a bound change on the stack without applying it or updating the activities.
  void push(const bound_change_t<i_t, f_t>& bound_change)
  {
    bound_changes_.push_back(bound_change);
  }

  // Sets the bounds of bound_change.var in lp to [new_lower, new_upper], records the previous
  // bounds in old_lower/old_upper so backtrack_to_parent can restore them, pushes the change on the
  // stack and updates the activities. A change that does not move any bound is ignored.
  bool apply(simplex::lp_problem_t<i_t, f_t>& lp, bound_change_t<i_t, f_t> bound_change)
  {
    if (bound_change.var < 0) { return false; }

    bound_change.old_lower = lp.lower[bound_change.var];
    bound_change.old_upper = lp.upper[bound_change.var];
    if (bound_change.old_lower == bound_change.new_lower &&
        bound_change.old_upper == bound_change.new_upper) {
      return false;
    }

    lp.lower[bound_change.var] = bound_change.new_lower;
    lp.upper[bound_change.var] = bound_change.new_upper;

    push(bound_change);
    update_activities(lp, bound_change);
    return true;
  }

  size_t size() const { return bound_changes_.size(); }
  bool is_empty() const { return bound_changes_.empty(); }
  void clear() { bound_changes_.clear(); }

  // Number of integer variables that the bound changes from start onwards fixed.
  i_t count_fixed_integers(const std::vector<simplex::variable_type_t>& var_types,
                           const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                           size_t start) const;

  // Applies every recorded change to [lower, upper], in order, refreshing their old bounds. The
  // activities are not updated; call compute_activities afterwards.
  void apply_stack(std::vector<f_t>& lower, std::vector<f_t>& upper)
  {
    for (auto& bound_change : bound_changes_) {
      bound_change.old_lower  = lower[bound_change.var];
      lower[bound_change.var] = bound_change.new_lower;
      bound_change.old_upper  = upper[bound_change.var];
      upper[bound_change.var] = bound_change.new_upper;
    }
  }

  // Pops the changes back to and including the last branching, restoring their bounds in lp and
  // reverting their activities.
  void backtrack_to_parent(simplex::lp_problem_t<i_t, f_t>& lp);

  // Computes the activities of all rows from the bounds in lp.
  void compute_activities(const csr_matrix_t<i_t, f_t>& Arow,
                          const simplex::lp_problem_t<i_t, f_t>& lp);

  // Incrementally updates the activities of every row containing bound_change.var after its bounds
  // changed from [old_lower, old_upper] to [new_lower, new_upper].
  void update_activities(const simplex::lp_problem_t<i_t, f_t>& lp,
                         const bound_change_t<i_t, f_t>& bound_change);

  // Recomputes all activities from the bounds in lp and propagates every row, tightening the bounds
  // in place. Returns false if the propagation proves the problem infeasible.
  bool propagate_full(const csr_matrix_t<i_t, f_t>& Arow,
                      const std::vector<simplex::variable_type_t>& var_types,
                      const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                      simplex::lp_problem_t<i_t, f_t>& lp);

  // Recomputes all activities from the bounds in lp and propagates from the rows containing vars.
  // Returns false if the propagation proves the problem infeasible.
  bool propagate_from_variables(const csr_matrix_t<i_t, f_t>& Arow,
                                const std::vector<simplex::variable_type_t>& var_types,
                                const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                                simplex::lp_problem_t<i_t, f_t>& lp,
                                const std::vector<i_t>& vars);

  // Propagates from the rows containing the variables of the bound changes from start onwards. The
  // activities must already include these changes, as apply does. Returns false if the propagation
  // proves the problem infeasible.
  bool propagate_from_stack(const csr_matrix_t<i_t, f_t>& Arow,
                            const std::vector<simplex::variable_type_t>& var_types,
                            const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                            simplex::lp_problem_t<i_t, f_t>& lp,
                            i_t start = 0);

  // Applies bound_change to lp and propagates from the rows containing its variable. A change that
  // does not move any bound is neither recorded nor propagated. Returns false if the propagation
  // proves the problem infeasible.
  bool apply_and_propagate(const csr_matrix_t<i_t, f_t>& Arow,
                           const std::vector<simplex::variable_type_t>& var_types,
                           const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                           bound_change_t<i_t, f_t> bound_change,
                           simplex::lp_problem_t<i_t, f_t>& lp);

  // Propagates the objective cutoff and any affected rows. Returns false if the cutoff is
  // infeasible.
  bool propagate_objective(const csr_matrix_t<i_t, f_t>& Arow,
                           const std::vector<simplex::variable_type_t>& var_types,
                           const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                           simplex::lp_problem_t<i_t, f_t>& lp,
                           f_t cutoff);

  // Nonzeros touched by the last propagation call, used as a work estimate.
  size_t last_nnz_processed{0};
  size_t objective_fixings{0};
  size_t objective_cutoffs{0};

  // Implications applied when an integer variable is fixed to 0 or 1. Either may be null, and the
  // clique table is skipped until it is ready.
  const probing_implied_bound_t<i_t, f_t>* implied_bounds = nullptr;
  const clique_table_t<i_t, f_t>* clique_table            = nullptr;

 private:
  // Contributions with |a_ij x_j| >= huge_value are counted as infinite.
  static constexpr f_t huge_value = 1e15;

  // The maximum number of bound propagation rounds after the objective propagation
  static constexpr i_t max_objective_rounds = 3;

  // An activity is recomputed once its peak exceeds it by this factor.
  static constexpr f_t recompute_factor = 1e6;

  // A continuous bound must shrink the domain by at least this fraction.
  static constexpr f_t min_relative_improvement = 0.3;

  // Relative to the primal_tol. The smallest absolute change accepted for a continuous bound and
  // the smallest change relative to its magnitude accepted for an integer bound.
  static constexpr f_t min_improvement_factor = 1e3;

  // The maximum number a given variable can be tightened in a single propagation call.
  static constexpr i_t max_var_tightenings_per_call = 50;

  // Stack position of the last tightening of each variable. An entry is stale unless it points
  // at a change of that variable recorded since propagation_start_.
  std::vector<size_t> last_change_;

  // Stack position at the start of the bound propagation.
  size_t propagation_start_{0};

  std::vector<row_activity_t<i_t, f_t>> row_activities_;
  std::vector<uint8_t> row_queued_;
  circular_deque_t<i_t> row_queue_;

  std::vector<uint8_t> var_queued_;
  circular_deque_t<i_t> var_queue_;

  std::vector<i_t> objective_vars_;
  std::vector<bound_change_t<i_t, f_t>> bound_changes_;

  size_t nnz_processed_{0};

  void clear_queues();

  // Queues the rows containing x_j and any implications of fixing x_j to 0 or 1.
  void queue_variable(i_t j,
                      const std::vector<simplex::variable_type_t>& var_types,
                      const simplex::lp_problem_t<i_t, f_t>& lp,
                      f_t tol);

  // Queues row i unless it is already queued or neither of its sides can tighten a bound. A row
  // awaiting recomputation is always queued.
  void queue_row(i_t i, const simplex::lp_problem_t<i_t, f_t>& lp, f_t tol);

  // Applies the probing and clique implications of x_j at its fixed value.
  bool propagate_implications(i_t j,
                              const std::vector<simplex::variable_type_t>& var_types,
                              const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                              simplex::lp_problem_t<i_t, f_t>& lp);

  // Intersects the bounds of x_j with [new_lower, new_upper], applying them if they improve.
  // Returns false if the bounds cross.
  bool tighten_bounds(i_t j,
                      f_t new_lower,
                      f_t new_upper,
                      bound_change_origin_t origin,
                      const std::vector<simplex::variable_type_t>& var_types,
                      const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                      simplex::lp_problem_t<i_t, f_t>& lp);

  // Propagates the queued rows in FIFO order, applying each accepted bound immediately
  // and queueing the rows it affects. Returns false if a row or a variable proves infeasibility.
  bool run_bound_propagation(const csr_matrix_t<i_t, f_t>& Arow,
                             const std::vector<simplex::variable_type_t>& var_types,
                             const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                             simplex::lp_problem_t<i_t, f_t>& lp);

  // Adds (sign = 1) or removes (sign = -1) the term a_ij * bound: to the Dot2 sum of side when it
  // is finite and smaller than huge_value, to its infinite count otherwise.
  void add_activity_term(activity_t<i_t, f_t>& side, f_t aij, f_t bound, i_t sign) const;

  // Recomputes the activity of row i from the bounds in [lower, upper].
  void compute_row_activity(i_t i,
                            const csr_matrix_t<i_t, f_t>& Arow,
                            const std::vector<f_t>& lower,
                            const std::vector<f_t>& upper);
};

template <typename i_t, typename f_t>
bool full_bound_strengthening(const csr_matrix_t<i_t, f_t>& Arow,
                              const std::vector<simplex::variable_type_t>& var_types,
                              const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                              simplex::lp_problem_t<i_t, f_t>& lp)
{
  domain_propagation_t<i_t, f_t> domain_propagation;
  return domain_propagation.propagate_full(Arow, var_types, settings, lp);
}

}  // namespace cuopt::mathematical_optimization::mip
