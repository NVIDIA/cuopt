/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/optimization_problem.hpp>
#include <mip_heuristics/structural/arc_flow.cuh>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::test {

namespace {

// Enhanced arc-flow model of two identical parallel machines minimizing weighted completion time.
// A job arc advances one machine's clock from state q to q + p, costs w * q, and covers one unit
// of its job type's demand; a loss arc pads the tail of a machine's horizon.
struct job_type_t {
  int p;
  int w;
  int d;
};

struct built_model_t {
  std::vector<double> values;
  std::vector<int> indices;
  std::vector<int> offsets;
  std::vector<double> row_lb;
  std::vector<double> row_ub;
  std::vector<double> obj;
  std::vector<double> var_lb;
  std::vector<double> var_ub;
  std::vector<var_t> var_types;
  int n_rows{0};
  int n_cols{0};
  int horizon{0};
  int loss_first{0};
  int job_arcs{0};     // arcs surviving the reduction
  int used_states{0};  // states surviving the reduction
};

struct build_options_t {
  // Encode path termination as slack in the conservation row instead of an explicit loss arc,
  // which is what Papilo's singleton column substitution produces.
  bool row_slack_terminators{false};
  bool permute{false};
  double flow_row_factor{1.0};  // scale the second flow row and its bounds
  // Break the affine cost model by moving one of this type's interior arcs off its own line.  Named
  // by type rather than by column because which arcs survive the reduction is not obvious outside
  // the builder.
  int perturbed_cost_type{-1};
};

constexpr int n_machines = 2;

int eaf_horizon(const std::vector<job_type_t>& jobs)
{
  int total = 0;
  int p_max = 0;
  for (const auto& job : jobs) {
    total += job.p * job.d;
    p_max = std::max(p_max, job.p);
  }
  return (total + (n_machines - 1) * p_max) / n_machines;
}

int eaf_loss_first(const std::vector<job_type_t>& jobs)
{
  int p_max = 0;
  for (const auto& job : jobs) {
    p_max = std::max(p_max, job.p);
  }
  return eaf_horizon(jobs) - p_max;
}

// Types in Smith order: decreasing weight over processing time, ties by index so the reduction is
// deterministic.  An optimal schedule runs each machine's jobs in this order, so a path through the
// graph visits types in this order and no other sequence needs representing.
std::vector<int> smith_order(const std::vector<job_type_t>& jobs)
{
  std::vector<int> order(jobs.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    const long lhs = (long)jobs[a].w * jobs[b].p;
    const long rhs = (long)jobs[b].w * jobs[a].p;
    if (lhs != rhs) { return lhs > rhs; }
    return a < b;
  });
  return order;
}

// The reduced graph Kramer, Dell'Amico and Iori build rather than the straight one.  A type may
// only leave a state that a canonical path reaches, meaning one composed of types no later in Smith
// order, which drops both arcs and whole states.  Straight arc flow gives every type an arc at
// every feasible start, so the load table alone determines reachability and a search that ignored
// the graph would still pass; here it cannot.
struct reduced_graph_t {
  std::vector<std::pair<int, int>> arcs;  // (type, start state)
  std::vector<int> states;                // used states, ascending
  std::vector<int> row_of_state;          // state -> row, or -1 when the reduction dropped it
};

// A machine may be loaded to exactly the horizon, so the states run to it inclusive.  Stopping one
// short silently drops the schedules that fill a machine, which are optimal often enough that the
// graph would no longer contain the optimum for the reference to be compared against.
reduced_graph_t reduce_eaf(const std::vector<job_type_t>& jobs, int horizon, int loss_first)
{
  const int last = horizon;
  std::vector<char> reachable(last + 1, 0);
  reachable[0] = 1;

  reduced_graph_t graph;
  for (const int type : smith_order(jobs)) {
    const int p = jobs[type].p;
    // Copies of this type may precede an arc of it, so its own chains extend reachability first.
    const std::vector<char> before = reachable;
    for (int q = 0; q <= last; ++q) {
      if (!before[q]) { continue; }
      for (int copies = 1; copies <= jobs[type].d; ++copies) {
        const int target = q + copies * p;
        if (target > last) { break; }
        reachable[target] = 1;
      }
    }
    for (int q = 0; q + p <= last; ++q) {
      if (reachable[q]) { graph.arcs.push_back({type, q}); }
    }
  }

  std::vector<char> used(last + 1, 0);
  used[0] = 1;
  for (const auto& [type, start] : graph.arcs) {
    used[start]                = 1;
    used[start + jobs[type].p] = 1;
  }
  for (int q = loss_first; q <= last; ++q) {
    if (reachable[q]) { used[q] = 1; }
  }
  graph.row_of_state.assign(last + 1, -1);
  for (int q = 0; q <= last; ++q) {
    if (!used[q]) { continue; }
    graph.row_of_state[q] = graph.states.size();
    graph.states.push_back(q);
  }
  return graph;
}

built_model_t build_eaf(const std::vector<job_type_t>& jobs, const build_options_t& opts)
{
  const int horizon    = eaf_horizon(jobs);
  const int loss_first = eaf_loss_first(jobs);
  EXPECT_GT(loss_first, 0) << "the source state must not also be a terminator";

  const int n_types           = jobs.size();
  const reduced_graph_t graph = reduce_eaf(jobs, horizon, loss_first);
  const int n_states          = graph.states.size();
  const int n_rows            = n_states + n_types;
  const auto row_of           = [&](int state) {
    const int row = graph.row_of_state[state];
    EXPECT_GE(row, 0) << "an arc referenced a state the reduction dropped";
    return row;
  };

  // Columns: every surviving job arc, then the loss arcs when they are represented explicitly.
  struct column_t {
    std::vector<std::pair<int, double>> entries;
    double cost;
    double ub;
    int type{-1};
  };
  std::vector<column_t> columns;
  for (const auto& [type, start] : graph.arcs) {
    column_t col;
    col.entries = {
      {row_of(start), 1.0}, {row_of(start + jobs[type].p), -1.0}, {n_states + type, 1.0}};
    col.cost = (double)jobs[type].w * start;
    col.ub   = jobs[type].d;
    col.type = type;
    columns.push_back(std::move(col));
  }
  if (!opts.row_slack_terminators) {
    for (const int q : graph.states) {
      if (q >= loss_first) { columns.push_back(column_t{{{row_of(q), 1.0}}, 0.0, 1.0, -1}); }
    }
  }

  std::vector<double> row_lb(n_rows, 0.0);
  std::vector<double> row_ub(n_rows, 0.0);
  row_lb[row_of(0)] = row_ub[row_of(0)] = n_machines;
  if (opts.row_slack_terminators) {
    for (const int q : graph.states) {
      if (q < loss_first) { continue; }
      row_lb[row_of(q)] = -1.0;
      row_ub[row_of(q)] = 0.0;
    }
  }
  for (int j = 0; j < n_types; ++j) {
    row_lb[n_states + j] = jobs[j].d;
    row_ub[n_states + j] = std::numeric_limits<double>::infinity();
  }

  if (opts.perturbed_cost_type >= 0) {
    std::vector<int> of_type;
    for (int c = 0; c < (int)columns.size(); ++c) {
      if (columns[c].type == opts.perturbed_cost_type) { of_type.push_back(c); }
    }
    // The slope is fitted from the label's extreme arcs, so only an interior arc is off the fitted
    // line and reachable solely by the residual check over every arc.
    EXPECT_GE(of_type.size(), 3u) << "an interior arc needs a type with at least three of them";
    columns[of_type[of_type.size() / 2]].cost += 1.0;
  }

  const int n_cols = columns.size();
  std::vector<int> row_perm(n_rows);
  std::vector<int> col_perm(n_cols);
  std::iota(row_perm.begin(), row_perm.end(), 0);
  std::iota(col_perm.begin(), col_perm.end(), 0);
  if (opts.permute) {
    std::reverse(row_perm.begin(), row_perm.end());
    for (int i = 0; i + 1 < n_cols; i += 2) {
      std::swap(col_perm[i], col_perm[i + 1]);
    }
  }

  built_model_t model;
  model.n_rows      = n_rows;
  model.n_cols      = n_cols;
  model.horizon     = horizon;
  model.loss_first  = loss_first;
  model.job_arcs    = graph.arcs.size();
  model.used_states = n_states;
  model.obj.assign(n_cols, 0.0);
  model.var_lb.assign(n_cols, 0.0);
  model.var_ub.assign(n_cols, 0.0);
  model.var_types.assign(n_cols, var_t::INTEGER);
  model.row_lb.assign(n_rows, 0.0);
  model.row_ub.assign(n_rows, 0.0);

  for (int r = 0; r < n_rows; ++r) {
    model.row_lb[row_perm[r]] = row_lb[r];
    model.row_ub[row_perm[r]] = row_ub[r];
  }

  std::vector<std::vector<std::pair<int, double>>> by_row(n_rows);
  for (int c = 0; c < n_cols; ++c) {
    const auto& col      = columns[c];
    const int mapped     = col_perm[c];
    model.obj[mapped]    = col.cost;
    model.var_ub[mapped] = col.ub;
    for (const auto& [row, value] : col.entries) {
      by_row[row_perm[row]].emplace_back(mapped, value);
    }
  }
  if (opts.flow_row_factor != 1.0) {
    const int scaled = row_perm[1];
    for (auto& [col, value] : by_row[scaled]) {
      value *= opts.flow_row_factor;
    }
    model.row_lb[scaled] *= opts.flow_row_factor;
    model.row_ub[scaled] *= opts.flow_row_factor;
  }

  model.offsets.push_back(0);
  for (int r = 0; r < n_rows; ++r) {
    std::sort(by_row[r].begin(), by_row[r].end());
    for (const auto& [col, value] : by_row[r]) {
      model.indices.push_back(col);
      model.values.push_back(value);
    }
    model.offsets.push_back(model.indices.size());
  }
  return model;
}

// Optimal schedule by exhaustive assignment.  Smith's rule makes weighted shortest processing
// time optimal per machine, so sequencing each machine that way gives the true optimum.
double brute_force_optimum(const std::vector<job_type_t>& jobs)
{
  const int horizon    = eaf_horizon(jobs);
  const int loss_first = eaf_loss_first(jobs);
  std::vector<std::pair<int, int>> expanded;  // (p, w)
  for (const auto& job : jobs) {
    for (int i = 0; i < job.d; ++i) {
      expanded.emplace_back(job.p, job.w);
    }
  }
  const int n = expanded.size();
  double best = std::numeric_limits<double>::infinity();
  for (int mask = 0; mask < (1 << n); ++mask) {
    std::array<std::vector<std::pair<int, int>>, n_machines> machine;
    for (int i = 0; i < n; ++i) {
      machine[(mask >> i) & 1].push_back(expanded[i]);
    }
    double cost   = 0.0;
    bool feasible = true;
    for (auto& jobs_on_machine : machine) {
      std::stable_sort(jobs_on_machine.begin(),
                       jobs_on_machine.end(),
                       [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                         return (long)a.second * b.first > (long)b.second * a.first;
                       });
      int clock = 0;
      for (const auto& [p, w] : jobs_on_machine) {
        cost += (double)w * clock;
        clock += p;
      }
      if (clock < loss_first || clock > horizon) { feasible = false; }
    }
    if (feasible) { best = std::min(best, cost); }
  }
  return best;
}

struct run_outcome_t {
  bool prescreened{false};
  bool found{false};
  bool exact{false};
  double objective{0.0};
  std::vector<double> assignment;
};

run_outcome_t run_heuristic(const built_model_t& model)
{
  const raft::handle_t handle{};
  optimization_problem_t<int, double> problem(&handle);
  auto values  = model.values;
  auto indices = model.indices;
  auto offsets = model.offsets;
  auto obj     = model.obj;
  auto var_lb  = model.var_lb;
  auto var_ub  = model.var_ub;
  auto types   = model.var_types;
  auto row_lb  = model.row_lb;
  auto row_ub  = model.row_ub;
  problem.set_csr_constraint_matrix(
    values.data(), values.size(), indices.data(), indices.size(), offsets.data(), offsets.size());
  problem.set_objective_coefficients(obj.data(), obj.size());
  problem.set_variable_lower_bounds(var_lb.data(), var_lb.size());
  problem.set_variable_upper_bounds(var_ub.data(), var_ub.size());
  problem.set_variable_types(types.data(), types.size());
  problem.set_constraint_lower_bounds(row_lb.data(), row_lb.size());
  problem.set_constraint_upper_bounds(row_ub.data(), row_ub.size());

  mip_solver_settings_t<int, double> settings;
  run_outcome_t outcome;
  mip::arc_flow_t<int, double> heuristic;
  outcome.prescreened = heuristic.recognize(problem, settings.get_tolerances());
  if (!outcome.prescreened) { return outcome; }

  std::atomic<bool> preemption{false};
  const auto status =
    heuristic.solve(settings.get_tolerances(), preemption, 0.0, outcome.assignment);
  outcome.found = status == mip::structural_outcome_t::constructed;
  outcome.exact = heuristic.search_was_exact();
  // The dispatcher would take this from the solver-space problem; the models here are minimize
  // with no offset, so the two agree.
  if (outcome.found) {
    outcome.objective = 0.0;
    for (size_t j = 0; j < outcome.assignment.size(); ++j) {
      outcome.objective += model.obj[j] * outcome.assignment[j];
    }
  }
  return outcome;
}

const std::vector<job_type_t>& small_instance()
{
  static const std::vector<job_type_t> jobs = {{1, 3, 2}, {2, 1, 1}, {3, 2, 1}};
  return jobs;
}

// Processing times that do not tile the horizon, so the reduction leaves gaps: states 1, 3, 8 and
// 13 are unreachable by any canonical path and are absent from the model entirely.
const std::vector<job_type_t>& gapped_instance()
{
  static const std::vector<job_type_t> jobs = {{2, 9, 2}, {5, 4, 2}, {7, 3, 1}};
  return jobs;
}

// The heaviest type leads the Smith order and fills six of nine units, so the reduction leaves it a
// single arc out of the source and its cost slope has no second point to be fitted from.
const std::vector<job_type_t>& single_arc_label_instance()
{
  static const std::vector<job_type_t> jobs = {{2, 1, 1}, {5, 1, 1}, {6, 4, 1}};
  return jobs;
}

}  // namespace

TEST(arc_flow, matches_brute_force_optimum)
{
  const auto model   = build_eaf(small_instance(), {});
  const auto outcome = run_heuristic(model);
  ASSERT_TRUE(outcome.prescreened);
  ASSERT_TRUE(outcome.found);
  EXPECT_DOUBLE_EQ(outcome.objective, brute_force_optimum(small_instance()));
  // A search this small is nowhere near the history budget, so beaming it would mean the budget
  // is being converted into a width up front instead of charged as it accumulates.
  EXPECT_TRUE(outcome.exact);
}

// The reduction is what separates the enhanced graph from the straight one, so the fixture has to
// exercise it: a graph with an arc at every feasible start makes reachability a function of the
// load alone, and a search that never consulted the arc set would pass anyway.
TEST(arc_flow, reduced_graph_omits_states_and_arcs)
{
  const auto model = build_eaf(gapped_instance(), {});
  int straight     = 0;
  for (const auto& job : gapped_instance()) {
    straight += std::max(0, eaf_horizon(gapped_instance()) - job.p + 1);
  }
  EXPECT_LT(model.job_arcs, straight) << "the reduction dropped no arc";
  EXPECT_LT(model.used_states, model.horizon + 1) << "the reduction dropped no state";
}

// Normal patterns keep at least one optimal schedule, so the reduced graph must still reach the
// optimum the reference finds by exhaustive assignment.
TEST(arc_flow, matches_brute_force_optimum_on_reduced_graph)
{
  const auto outcome = run_heuristic(build_eaf(gapped_instance(), {}));
  ASSERT_TRUE(outcome.prescreened);
  ASSERT_TRUE(outcome.found);
  EXPECT_DOUBLE_EQ(outcome.objective, brute_force_optimum(gapped_instance()));
}

// A label with one arc is ordered by where that arc can go rather than by its ratio, which is a
// position the model determines but not the Smith one.  The point stays usable; what must not
// happen is the pass reporting it as the optimum of an order it did not actually search.
TEST(arc_flow, single_arc_label_is_ordered_but_not_exact)
{
  const auto outcome = run_heuristic(build_eaf(single_arc_label_instance(), {}));
  ASSERT_TRUE(outcome.prescreened);
  ASSERT_TRUE(outcome.found);
  EXPECT_FALSE(outcome.exact);
  EXPECT_DOUBLE_EQ(outcome.objective, brute_force_optimum(single_arc_label_instance()));
}

// The detector reads only permutation invariant data, so reordering rows and columns must not
// change what it finds.  This is the property that keeps it from leaning on model index order.
TEST(arc_flow, invariant_under_row_and_column_permutation)
{
  build_options_t permuted;
  permuted.permute    = true;
  const auto plain    = run_heuristic(build_eaf(small_instance(), {}));
  const auto shuffled = run_heuristic(build_eaf(small_instance(), permuted));
  ASSERT_TRUE(plain.found);
  ASSERT_TRUE(shuffled.found);
  EXPECT_DOUBLE_EQ(plain.objective, shuffled.objective);
}

// Papilo substitutes a bounded singleton column out of its equality and leaves the row as an
// inequality, so a loss arc reaches the second early heuristic slot as conservation row slack.
TEST(arc_flow, accepts_row_slack_terminators)
{
  build_options_t slack;
  slack.row_slack_terminators = true;
  const auto outcome          = run_heuristic(build_eaf(small_instance(), slack));
  ASSERT_TRUE(outcome.prescreened);
  ASSERT_TRUE(outcome.found);
  EXPECT_DOUBLE_EQ(outcome.objective, brute_force_optimum(small_instance()));
}

// MIP scaling applies power of two row factors before this heuristic runs, so the unit incidence
// pattern is only recoverable after normalizing each row by its coefficient magnitude.
TEST(arc_flow, tolerates_row_scaling)
{
  build_options_t scaled;
  scaled.flow_row_factor = 4.0;
  const auto outcome     = run_heuristic(build_eaf(small_instance(), scaled));
  ASSERT_TRUE(outcome.prescreened);
  ASSERT_TRUE(outcome.found);
  EXPECT_DOUBLE_EQ(outcome.objective, brute_force_optimum(small_instance()));
}

TEST(arc_flow, rejects_non_affine_costs)
{
  // The slope is fitted from the label's extreme arcs, so moving one arc off the line is only
  // caught by the residual check that revisits every arc.
  build_options_t perturbed;
  perturbed.perturbed_cost_type = 1;
  const auto outcome            = run_heuristic(build_eaf(small_instance(), perturbed));
  EXPECT_FALSE(outcome.found);
}

// The construction consumes exactly the demanded units, so it cannot discover that oversatisfying
// a covering row pays.  A negative cost slope is where that would happen, and the detector has to
// refuse the model rather than return a point it has no argument for.
TEST(arc_flow, rejects_negative_cost_slope)
{
  const std::vector<job_type_t> jobs = {{1, 3, 2}, {2, -1, 1}, {3, 2, 1}};
  const auto outcome                 = run_heuristic(build_eaf(jobs, {}));
  EXPECT_FALSE(outcome.found);
}

TEST(arc_flow, rejects_model_without_unit_incidence)
{
  built_model_t knapsack;
  knapsack.n_rows    = 1;
  knapsack.n_cols    = 2;
  knapsack.values    = {2.0, 3.0};
  knapsack.indices   = {0, 1};
  knapsack.offsets   = {0, 2};
  knapsack.row_lb    = {1.0};
  knapsack.row_ub    = {std::numeric_limits<double>::infinity()};
  knapsack.obj       = {1.0, 1.0};
  knapsack.var_lb    = {0.0, 0.0};
  knapsack.var_ub    = {1.0, 1.0};
  knapsack.var_types = {var_t::INTEGER, var_t::INTEGER};
  const auto outcome = run_heuristic(knapsack);
  EXPECT_FALSE(outcome.prescreened);
  EXPECT_FALSE(outcome.found);
}

TEST(arc_flow, is_reproducible)
{
  const auto model  = build_eaf(small_instance(), {});
  const auto first  = run_heuristic(model);
  const auto second = run_heuristic(model);
  ASSERT_TRUE(first.found);
  ASSERT_TRUE(second.found);
  EXPECT_DOUBLE_EQ(first.objective, second.objective);
  EXPECT_EQ(first.assignment, second.assignment);
}

}  // namespace cuopt::mathematical_optimization::test
