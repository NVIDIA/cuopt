/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../fj_cpu_binary.cuh"

#include "cdcl_sat_utils.hpp"

#include <utilities/macros.cuh>
#include <utilities/pcgenerator.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

namespace {

// Compact CDCL solver used only for large, objective-free Boolean feasibility models. Literals are
// encoded as (variable << 1) | satisfying_value.
struct sat_t {
  static constexpr int no_reason                       = -1;
  static constexpr int no_literal                      = -1;
  static constexpr int no_variable                     = -1;
  static constexpr int no_conflict                     = -1;
  static constexpr int propagation_stopped             = -2;
  static constexpr int8_t unassigned                   = -1;
  static constexpr double activity_decay_min           = 0.94;
  static constexpr double activity_decay_step          = 0.01;
  static constexpr int activity_decay_choices          = 5;
  static constexpr int restart_interval_unit           = 64;
  static constexpr int restart_interval_choices        = 4;
  static constexpr int restart_growth_limit            = 4096;
  static constexpr double restart_growth_ratio         = 1.5;
  static constexpr size_t propagation_stop_poll_period = 256;
  static constexpr int64_t solve_stop_poll_period      = 128;
  static constexpr int protected_clause_lbd            = 2;
  static constexpr int protected_clause_size           = 2;
  static constexpr int clause_stale_conflicts          = 2000;
  static constexpr int database_reduction_interval     = 4000;
  static constexpr size_t learned_clause_reserve       = 10000;

  struct watch_t {
    clause_ref_t id;
    int blocker;
  };

  clause_arena_t arena;
  std::vector<clause_ref_t> clause_refs;
  std::vector<std::vector<watch_t>> watches;
  std::vector<int8_t> value, phase;
  std::vector<int> assignment_depth;
  std::vector<clause_ref_t> reason;
  std::vector<uint8_t> seen;
  sat_variable_order_t variable_order;
  trail_t trail;
  double decay{activity_decay_min};
  int conflicts{0};
  int restart_initial_limit{restart_interval_unit};
  int64_t steps{0};

  int eval_literal(int lit) const
  {
    const int state = value[LIT_VAR(lit)];
    return state == unassigned ? unassigned : state == LIT_POLARITY(lit);
  }
  // enqueue an assignment to the queue. returns false if this causes a contradiction
  bool enqueue(int lit, clause_ref_t why)
  {
    if (const int state = eval_literal(lit); state != unassigned) return state;
    const int v         = LIT_VAR(lit);
    const int requested = LIT_POLARITY(lit);
    value[v] = phase[v] = requested;
    assignment_depth[v] = trail.decision_depth();
    reason[v]           = why;
    trail.enqueue(lit);
    return true;
  }
  // insert a new clause into the arena
  clause_ref_t add(std::vector<int> lits, bool learned = false, int lbd = 0)
  {
    const clause_ref_t ref = arena.add(std::span<const int>{lits}, learned, lbd, conflicts);
    clause_refs.push_back(ref);
    if (lits.size() >= 2) {
      watches[lits[0]].push_back({ref, lits[1]});
      watches[lits[1]].push_back({ref, lits[0]});
    }
    return ref;
  }
  // Two-watched-literal BCP [Moskewicz et al. 2001, Sec. 2; Een-Sorensson 2003],
  // with MiniSat 2.1 blocking literals [Een-Sorensson 2008].
  // A clause propagates only when all but one literals are false, and conflicts when all are false.
  // Two non-false watched literals therefore certify that neither case is possible, so assignments
  // to unwatched literals need no work. When a watch becomes false, a true blocker proves
  // satisfaction; otherwise move the watch to another non-false literal. If none exists, all
  // unwatched literals are false and the other watch is true (satisfied), unassigned (unit), or
  // false (conflict). Backtracking only changes false literals to non-false, so it cannot require
  // watch repair.

  // Basically, the idea is:
  // A clause only becomes relevant for propagation when it is close to failure.
  // IF at least two literals are non-false, nothing needs to happen.
  // If exactly one is non-false, the literal needs to be true for the clause to be satisfied.
  // if all literals are false, there is a conflict!
  // So, to know if we need to propagate, we only need to track two literals.
  // As long as two watched literals are non-false, the clause cannot be unit (necessarily
  // satisfied) or conflicting. returns the clause that conflicted, if any
  // conveniently, backtracking requires zero watchlist updates
  // since undoing assignents cannot turn a non-false into a false (only false/true -> unassigned)
  template <typename Stop>
  clause_ref_t propagate(Stop& stop)
  {
    // anything remaining in the propagation queue?
    while (trail.has_pending_propagation()) {
      // false_lit is the literal this propagated assignment just falsified
      const int false_lit = LIT_NEG(trail.next_to_propagate());
      // list all clauses whose watch may have been broken by this new assignment
      auto& watched        = watches[false_lit];
      const int8_t* values = value.data();

      // out is used to compact the watch list survivors at the front, in-place, as the watch list
      // is walked
      size_t out = 0;
      // the watch list sotres (clause reference, blocker) where blocker is the other watched
      // literal
      for (size_t k = 0; k < watched.size(); ++k) {
        const watch_t w         = watched[k];
        const int blocker_value = values[LIT_VAR(w.blocker)];
        // blocker is true, so the clause is already satisfied, skip it
        if (blocker_value != unassigned && blocker_value == LIT_POLARITY(w.blocker)) {
          watched[out++] = w;
          continue;
        }

        // actually load the clause
        const clause_ref_t ref = w.id;
        auto& clause           = arena[ref];
        if (clause.deleted()) continue;
        const int n_lits = clause.size;
        // watched literals are physically the first two literals of the clause
        // normalize them so that lits[0] = the OTHER watch and lits[1] = the watch that just became
        // false list[0] = surviving watch, list[1] = broken watch
        if (clause.literals[0] == false_lit) std::swap(clause.literals[0], clause.literals[1]);
        // if the surviving watch is true, then the clause is satisfied, nothing to do
        if (eval_literal(clause.literals[0]) == 1) {
          watched[out++] = {ref, clause.literals[0]};
          continue;
        }
        // let's look through the unwatched literals (hence starting index 2) looking for a
        // non-false to replace the watch with.
        bool moved = false;
        for (int j = 2; j < n_lits; ++j) {
          if (eval_literal(clause.literals[j]) == 0) continue;
          // swap the literals to move that non-false in the watcher slot
          std::swap(clause.literals[1], clause.literals[j]);
          // update the watch list for that literal
          watches[clause.literals[1]].push_back({ref, clause.literals[0]});
          moved = true;
          break;
        }
        if (moved) continue;
        watched[out++] = {ref, clause.literals[0]};
        // no replacement for one watch found. thus, the other watched has to be true for the
        // constraint to be satisfied. let's try pushing it to the assignment queue. if this fails,
        // we've got ourselves a conflict!
        if (!enqueue(clause.literals[0], ref)) {
          while (++k < watched.size())
            watched[out++] = watched[k];
          watched.resize(out);
          return ref;
        }
      }
      watched.resize(out);

      if (trail.propagated_count() % propagation_stop_poll_period == 0 && stop())
        return propagation_stopped;
    }
    return no_conflict;
  }

  // backtrack to a given decision depth
  void backtrack(int target_depth)
  {
    if (!trail.above_depth(target_depth)) return;
    const size_t retained_size = trail.backtrack_offset(target_depth);
    for (size_t k = trail.size(); k > retained_size;) {
      const int v = LIT_VAR(trail[--k]);
      value[v]    = unassigned;
      reason[v]   = no_reason;
      variable_order.insert(v);
    }
    trail.truncate_to_depth(target_depth);
  }

  // first-UIP style conflict analyzis to generate a learned nogood clause
  // let's figure out why the variables involved in the conflict clause got their values
  // based on the clauses which forced their values
  std::vector<int> analyze_first_uip(clause_ref_t conflict, int& backtrack_depth, int& lbd)
  {
    // literal[0] will be the first UIP
    std::vector<int> learned(1, no_literal);
    int paths = 0;
    int pivot = no_variable;
    int index = (int)trail.size() - 1;
    do {
      // start with the conflicting clause then thourhg the "reason" clauses of the variables
      // involved
      auto& clause = arena[conflict];
      clause.touch(conflicts);  // update conflict activity for this clause
      for (int i = 0; i < clause.size; ++i) {
        const int lit = clause.literals[i];
        const int v   = LIT_VAR(lit);
        // skip uninteresting literals for this clause (the variable being resolved, variables being
        // encountered, and root assignments)
        if ((pivot != no_variable && v == LIT_VAR(pivot)) || seen[v] || assignment_depth[v] == 0)
          continue;
        seen[v] = 1;
        variable_order.bump(v);
        // if the variable is from the current decision depth
        if (assignment_depth[v] == trail.decision_depth())
          ++paths;
        else
          learned.push_back(lit);
      }
      // find the most recentlyt assignment variable that participates in the current conflict
      // frontier
      while (!seen[LIT_VAR(trail[index])])
        --index;
      // pick the next variable to resolve
      pivot                = trail[index--];
      seen[LIT_VAR(pivot)] = 0;
      --paths;
      // replace the pivot by its reason clause
      conflict = reason[LIT_VAR(pivot)];
      cuopt_assert(paths == 0 || conflict != no_reason, "active conflict path has no reason");
    } while (paths > 0);  // until no unresolved current-level paths remain

    // first UIP (unique implication point) is found
    // create the asserting literal
    learned[0]      = LIT_NEG(pivot);
    backtrack_depth = 0;
    size_t best     = 1;
    // find the largest decision depth from all the learned literals
    for (size_t k = 1; k < learned.size(); ++k) {
      const int v = LIT_VAR(learned[k]);
      seen[v]     = 0;
      if (assignment_depth[v] > backtrack_depth) {
        backtrack_depth = assignment_depth[v];
        best            = k;
      }
    }
    // put the highest level non-UIP in position 1 so that it is one of the watched literals of the
    // clause
    if (learned.size() > 1) std::swap(learned[1], learned[best]);

    // Compute the LBD value of this clause (distinct decision levels in the clause)
    std::vector<int> depths;
    depths.reserve(learned.size());
    depths.push_back(trail.decision_depth());
    for (size_t k = 1; k < learned.size(); ++k)
      depths.push_back(assignment_depth[LIT_VAR(learned[k])]);
    std::sort(depths.begin(), depths.end());
    lbd = std::unique(depths.begin(), depths.end()) - depths.begin();
    variable_order.decay_activity(decay);
    return learned;
  }

  void reduce_database()
  {
    // pin the current assigments so their conflict analysis clauses don't get dropped
    // we mark as deleted but we don't really have proper garbage collection / compaction (yet?
    // likely unecessary given the limits)
    for (int lit : trail)
      if (reason[LIT_VAR(lit)] != no_reason) arena[reason[LIT_VAR(lit)]].set_locked(true);
    for (clause_ref_t ref : clause_refs) {
      auto& clause = arena[ref];
      if (clause.learned() && !clause.deleted() && !clause.locked() &&
          clause.lbd > protected_clause_lbd && clause.size > protected_clause_size &&
          clause.last_conflict < conflicts - clause_stale_conflicts) {
        clause.mark_deleted();
      }
    }
    for (int lit : trail)
      if (reason[LIT_VAR(lit)] != no_reason) arena[reason[LIT_VAR(lit)]].set_locked(false);
  }

  sat_t(int n,
        std::vector<std::vector<int>> input,
        const std::vector<int8_t>& initial_assignment,
        uint64_t rng_seed)
    : watches(2 * n),
      value(n, unassigned),
      phase(n, 0),
      assignment_depth(n),
      reason(n, no_reason),
      seen(n),
      variable_order(n, rng_seed)
  {
    cuopt::pcgenerator_t policy_rng(rng_seed);
    decay =
      activity_decay_min + activity_decay_step * policy_rng.uniform(0, activity_decay_choices);
    restart_initial_limit =
      restart_interval_unit * policy_rng.uniform(1, restart_interval_choices + 1);

    for (int v = 0; v < n; ++v)
      if (v < (int)initial_assignment.size()) phase[v] = initial_assignment[v];

    clause_refs.reserve(input.size() + learned_clause_reserve);
    size_t total_lits = 0;
    for (const auto& clause : input)
      total_lits += clause.size();
    arena.reserve(clause_arena_t::storage_words(input.size(), total_lits) * 3 / 2);
    for (auto& clause : input) {
      for (int lit : clause)
        variable_order.add_initial_activity(LIT_VAR(lit));
      add(std::move(clause));
    }
    // Encoding nodes have no independent model meaning; conflicts may still promote them later.
    for (int v = (int)initial_assignment.size(); v < n; ++v)
      variable_order.clear_activity(v);
    variable_order.build_heap();
  }

  template <typename Stop>
  sat_result_t solve(Stop stop)
  {
    for (clause_ref_t ref : clause_refs) {
      auto& clause     = arena[ref];
      const int n_lits = clause.size;
      // empty clauses are false since the identity of OR is false
      // if these is only one literal, then its value is forced
      if (n_lits == 0 || (n_lits == 1 && !enqueue(clause.literals[0], ref)))
        return sat_result_t::infeasible;
    }
    int restart_limit = restart_initial_limit;
    int since_restart = 0;
    while (true) {
      if (++steps % solve_stop_poll_period == 0 && stop()) return sat_result_t::stopped;
      const clause_ref_t conflict = propagate(stop);
      if (conflict == propagation_stopped) return sat_result_t::stopped;

      // we've got conflicts with the current assginment!
      if (conflict != no_conflict) {
        ++conflicts;
        ++since_restart;
        if (trail.at_root()) return sat_result_t::infeasible;
        int backtrack_depth = 0;
        int lbd             = 0;
        auto learned        = analyze_first_uip(conflict, backtrack_depth, lbd);
        const int asserting = learned[0];
        backtrack(backtrack_depth);
        const clause_ref_t id                          = add(std::move(learned), true, lbd);
        [[maybe_unused]] const bool asserting_enqueued = enqueue(asserting, id);
        cuopt_assert(asserting_enqueued, "learned clause is not asserting after backtrack");
        if (conflicts % database_reduction_interval == 0) reduce_database();
        if (since_restart >= restart_limit) {
          backtrack(0);
          since_restart = 0;
          restart_limit = restart_limit < restart_growth_limit
                            ? restart_limit * restart_growth_ratio
                            : restart_initial_limit;
        }
        // no conflcits with the current partial assignment. let's make a decision
      } else {
        int decision = no_variable;
        while (!variable_order.empty()) {
          const int candidate = variable_order.pop_best();
          if (value[candidate] == unassigned) {
            decision = candidate;
            break;
          }
        }
        // the assignment is full, and there's no conflict, so SAT!
        if (decision == no_variable) return sat_result_t::successful;
        trail.start_decision_depth();
        [[maybe_unused]] const bool decision_enqueued =
          enqueue(LIT(decision, phase[decision]), no_reason);
        cuopt_assert(decision_enqueued, "decision variable is already assigned");
      }
    }
  }
};

struct sat_bve_t {
  static constexpr int stop_poll_period    = 256;
  static constexpr size_t occurrence_limit = 32;
  static constexpr size_t resolvent_limit  = 64;
  static constexpr int no_required_value   = -1;

  struct extension_t {
    int variable;
    std::vector<std::vector<int>> clauses;
  };
  std::vector<extension_t> extension;
  std::vector<int> compact_to_original;
  int model_variables{0};

  static bool canonicalize(std::vector<int>& clause)
  {
    std::sort(clause.begin(), clause.end());
    size_t out = 0;
    for (int lit : clause) {
      // duplicate
      if (out && clause[out - 1] == lit) continue;
      // tautology, drop
      if (out && LIT_NEG(clause[out - 1]) == lit) return false;
      clause[out++] = lit;
    }
    clause.resize(out);
    return true;
  }

  // perform standard BVE elimination to remove a variable and its clause and replace them by
  // projections
  // see Eén–Biere 2005 for details
  template <typename Stop>
  bool presolve(std::vector<std::vector<int>>& input, int n, int original_variables, Stop& stop)
  {
    struct clause_t {
      std::vector<int> lits;
      bool deleted{false};
    };
    std::vector<clause_t> clauses;
    clauses.reserve(input.size() * 2);
    for (auto& clause : input) {
      if (canonicalize(clause)) clauses.push_back({std::move(clause), false});
    }
    // build a list of clauses containing a literal
    std::vector<std::vector<int>> occurrence(2 * n);
    for (size_t id = 0; id < clauses.size(); ++id)
      for (int lit : clauses[id].lits)
        occurrence[lit].push_back(id);

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    // try variables with fewer occurences first
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      const size_t degree_a = occurrence[LIT(a, 0)].size() + occurrence[LIT(a, 1)].size();
      const size_t degree_b = occurrence[LIT(b, 0)].size() + occurrence[LIT(b, 1)].size();
      return degree_a != degree_b ? degree_a < degree_b : a < b;
    });
    std::vector<int> positive, negative;
    std::vector<std::vector<int>> resolvents;
    for (int v : order) {
      if (v % stop_poll_period == 0 && stop()) return false;
      positive.clear();
      negative.clear();
      // gather live positive and negative clauses for this variable
      for (int id : occurrence[LIT(v, 1)])
        if (!clauses[id].deleted) positive.push_back(id);
      for (int id : occurrence[LIT(v, 0)])
        if (!clauses[id].deleted) negative.push_back(id);
      // gate trivial or expensive cases
      if (positive.empty() || negative.empty() || positive.size() > occurrence_limit ||
          negative.size() > occurrence_limit)
        continue;

      size_t n_res = 0;
      bool reject  = false;
      // eliminate all clauses involving v by combining their "remanants" together
      for (int p : positive) {
        for (int q : negative) {
          if (resolvents.size() <= n_res) resolvents.emplace_back();
          std::vector<int>& resolvent = resolvents[n_res];
          resolvent.clear();
          for (int lit : clauses[p].lits)
            if (LIT_VAR(lit) != v) resolvent.push_back(lit);
          for (int lit : clauses[q].lits)
            if (LIT_VAR(lit) != v) resolvent.push_back(lit);
          if (!canonicalize(resolvent)) continue;
          ++n_res;
          // avoid fill-in
          if (n_res > resolvent_limit || n_res > positive.size() + negative.size()) {
            reject = true;
            break;
          }
        }
        if (reject) break;
      }
      if (reject) continue;

      // save the old clauses to enable reconstruction in postsolve, then delete them
      extension_t saved{v, {}};
      saved.clauses.reserve(positive.size() + negative.size());
      for (int id : positive) {
        saved.clauses.push_back(std::move(clauses[id].lits));
        clauses[id].deleted = true;
      }
      for (int id : negative) {
        saved.clauses.push_back(std::move(clauses[id].lits));
        clauses[id].deleted = true;
      }
      extension.push_back(std::move(saved));
      // insert the generated resolvent clauses
      for (size_t r = 0; r < n_res; ++r) {
        const int id = (int)clauses.size();
        clauses.push_back({resolvents[r], false});
        for (int lit : clauses[id].lits)
          occurrence[lit].push_back(id);
      }
    }

    // compact surviving variables
    std::vector<uint8_t> used(n);
    for (const auto& clause : clauses)
      if (!clause.deleted)
        for (int lit : clause.lits)
          used[LIT_VAR(lit)] = 1;
    for (int v = 0; v < original_variables; ++v)
      if (used[v]) compact_to_original.push_back(v);

    // consider auxilliary generated variables
    model_variables = (int)compact_to_original.size();
    for (int v = original_variables; v < n; ++v)
      if (used[v]) compact_to_original.push_back(v);

    std::vector<int> remap(n, -1);
    for (size_t v = 0; v < compact_to_original.size(); ++v)
      remap[compact_to_original[v]] = v;
    input.clear();

    for (auto& clause : clauses)
      if (!clause.deleted) {
        for (int& lit : clause.lits)
          lit = LIT(remap[LIT_VAR(lit)], LIT_POLARITY(lit));
        input.push_back(std::move(clause.lits));
      }
    return true;
  }

  void postsolve(const std::vector<int8_t>& compact,
                 std::vector<int8_t>& value,
                 const std::vector<int8_t>& initial_assignment) const
  {
    value = initial_assignment;
    for (size_t i = 0; i < compact_to_original.size(); ++i)
      value[compact_to_original[i]] = compact[i];
    for (auto it = extension.rbegin(); it != extension.rend(); ++it) {
      int required = no_required_value;
      for (const auto& clause : it->clauses) {
        bool satisfied = false;
        for (int lit : clause)
          if (LIT_VAR(lit) != it->variable && value[LIT_VAR(lit)] == LIT_POLARITY(lit))
            satisfied = true;
        if (satisfied) continue;
        for (int lit : clause)
          if (LIT_VAR(lit) == it->variable) {
            cuopt_assert(required == no_required_value || required == LIT_POLARITY(lit),
                         "BVE recovery requires conflicting values");
            required = LIT_POLARITY(lit);
          }
      }
      value[it->variable] =
        required != no_required_value ? required : initial_assignment[it->variable];
    }
  }
};
}  // namespace

template <typename coef_t>
sat_result_t fj_bin_sat_search(const fj_bin_problem_t<coef_t>& pb,
                               std::vector<int8_t>& assignment,
                               uint64_t seed,
                               const std::function<bool()>& stop,
                               int64_t& steps)
{
  std::vector<std::vector<int>> cnf;
  int variables              = 0;
  const auto encoding_result = encode_pb_model(pb, variables, cnf, stop);
  if (encoding_result != sat_result_t::successful) return encoding_result;

  sat_bve_t bve;
  if (!bve.presolve(cnf, variables, pb.n_variables, stop)) return sat_result_t::stopped;
  std::vector<int8_t> compact_initial_assignment;
  compact_initial_assignment.reserve(bve.model_variables);
  for (int k = 0; k < bve.model_variables; ++k)
    compact_initial_assignment.push_back(assignment[bve.compact_to_original[k]]);

  sat_t sat((int)bve.compact_to_original.size(), std::move(cnf), compact_initial_assignment, seed);
  const auto result = sat.solve(stop);
  steps             = sat.steps;
  if (result == sat_result_t::successful) {
    std::vector<int8_t> initial_assignment(variables, 0), full_value;
    std::copy(assignment.begin(), assignment.end(), initial_assignment.begin());
    bve.postsolve(sat.value, full_value, initial_assignment);
    for (int r = 0; r < pb.n_constraints; ++r) {
      [[maybe_unused]] int64_t lhs = 0;
      for (int p = pb.offsets[r]; p < pb.offsets[r + 1]; ++p)
        lhs += int64_t(pb.coefficients[p]) * full_value[pb.variables[p]];
      cuopt_assert(lhs <= pb.bound[r], "SAT assignment violates original constraint");
    }
    std::copy_n(full_value.begin(), pb.n_variables, assignment.begin());
  }
  return result;
}

template sat_result_t fj_bin_sat_search<int8_t>(const fj_bin_problem_t<int8_t>&,
                                                std::vector<int8_t>&,
                                                uint64_t,
                                                const std::function<bool()>&,
                                                int64_t&);

template sat_result_t fj_bin_sat_search<int16_t>(const fj_bin_problem_t<int16_t>&,
                                                 std::vector<int8_t>&,
                                                 uint64_t,
                                                 const std::function<bool()>&,
                                                 int64_t&);

}  // namespace cuopt::mathematical_optimization::mip
