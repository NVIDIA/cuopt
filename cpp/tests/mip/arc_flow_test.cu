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
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::test {

namespace {

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
  int horizon{0};
  int job_arcs{0};
  int used_states{0};
};

struct build_options_t {
  bool row_slack_terminators{false};
  bool permute{false};
  double flow_row_factor{1.0};
  double cost_intercept{0.0};
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

struct reduced_graph_t {
  std::vector<std::pair<int, int>> arcs;
  std::vector<int> states;
  std::vector<int> row_of_state;
};

reduced_graph_t reduce_eaf(const std::vector<job_type_t>& jobs, int horizon, int loss_first)
{
  const int last = horizon;
  std::vector<char> reachable(last + 1, 0);
  reachable[0] = 1;

  reduced_graph_t graph;
  for (const int type : smith_order(jobs)) {
    const int p = jobs[type].p;
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
    col.cost = (double)jobs[type].w * start + opts.cost_intercept;
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
  model.horizon     = horizon;
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

double brute_force_optimum(const std::vector<job_type_t>& jobs)
{
  const int horizon    = eaf_horizon(jobs);
  const int loss_first = eaf_loss_first(jobs);
  std::vector<std::pair<int, int>> expanded;
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

struct input_options_t {
  bool set_lower_bounds{true};
  bool set_upper_bounds{true};
};

void expect_feasible(const built_model_t& model, const std::vector<double>& assignment)
{
  ASSERT_EQ(assignment.size(), model.obj.size());
  for (size_t j = 0; j < assignment.size(); ++j) {
    EXPECT_GE(assignment[j], model.var_lb[j]);
    EXPECT_LE(assignment[j], model.var_ub[j]);
    if (model.var_types[j] == var_t::INTEGER) {
      EXPECT_DOUBLE_EQ(assignment[j], std::round(assignment[j]));
    }
  }
  for (size_t r = 0; r < model.row_lb.size(); ++r) {
    double activity = 0.0;
    for (int k = model.offsets[r]; k < model.offsets[r + 1]; ++k) {
      activity += model.values[k] * assignment[model.indices[k]];
    }
    EXPECT_GE(activity, model.row_lb[r]);
    EXPECT_LE(activity, model.row_ub[r]);
  }
}

run_outcome_t run_heuristic(const built_model_t& model, input_options_t options = {})
{
  const raft::handle_t handle{};
  optimization_problem_t<int, double> problem(&handle);
  problem.set_csr_constraint_matrix(
    model.values.data(),
    model.values.size(),
    model.indices.data(),
    model.indices.size(),
    model.offsets.data(),
    model.offsets.size());
  problem.set_objective_coefficients(model.obj.data(), model.obj.size());
  if (options.set_lower_bounds) {
    problem.set_variable_lower_bounds(model.var_lb.data(), model.var_lb.size());
  }
  if (options.set_upper_bounds) {
    problem.set_variable_upper_bounds(model.var_ub.data(), model.var_ub.size());
  }
  problem.set_variable_types(model.var_types.data(), model.var_types.size());
  problem.set_constraint_lower_bounds(model.row_lb.data(), model.row_lb.size());
  problem.set_constraint_upper_bounds(model.row_ub.data(), model.row_ub.size());

  mip_solver_settings_t<int, double> settings;
  run_outcome_t outcome;
  mip::arc_flow_t<int, double> heuristic;
  outcome.prescreened = heuristic.recognize(problem, settings.get_tolerances());
  if (!outcome.prescreened) { return outcome; }

  std::atomic<bool> preemption{false};
  const auto status = heuristic.solve(settings.get_tolerances(), preemption, outcome.assignment);
  outcome.found = status == mip::structural_outcome_t::constructed;
  outcome.exact = heuristic.search_was_exact();
  if (outcome.found) {
    expect_feasible(model, outcome.assignment);
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

const std::vector<job_type_t>& gapped_instance()
{
  static const std::vector<job_type_t> jobs = {{2, 9, 2}, {5, 4, 2}, {7, 3, 1}};
  return jobs;
}

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
  EXPECT_TRUE(outcome.exact);
}

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

TEST(arc_flow, matches_brute_force_optimum_on_reduced_graph)
{
  const auto outcome = run_heuristic(build_eaf(gapped_instance(), {}));
  ASSERT_TRUE(outcome.prescreened);
  ASSERT_TRUE(outcome.found);
  EXPECT_DOUBLE_EQ(outcome.objective, brute_force_optimum(gapped_instance()));
}

TEST(arc_flow, single_arc_label_is_ordered_but_not_exact)
{
  const auto outcome = run_heuristic(build_eaf(single_arc_label_instance(), {}));
  ASSERT_TRUE(outcome.prescreened);
  ASSERT_TRUE(outcome.found);
  EXPECT_FALSE(outcome.exact);
  EXPECT_DOUBLE_EQ(outcome.objective, brute_force_optimum(single_arc_label_instance()));
}

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

TEST(arc_flow, accepts_row_slack_terminators)
{
  build_options_t slack;
  slack.row_slack_terminators = true;
  const auto outcome          = run_heuristic(build_eaf(small_instance(), slack));
  ASSERT_TRUE(outcome.prescreened);
  ASSERT_TRUE(outcome.found);
  EXPECT_DOUBLE_EQ(outcome.objective, brute_force_optimum(small_instance()));
}

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
  build_options_t perturbed;
  perturbed.perturbed_cost_type = 1;
  const auto outcome            = run_heuristic(build_eaf(small_instance(), perturbed));
  EXPECT_FALSE(outcome.found);
}

TEST(arc_flow, rejects_negative_cost_slope)
{
  const std::vector<job_type_t> jobs = {{1, 3, 2}, {2, -1, 1}, {3, 2, 1}};
  const auto outcome                 = run_heuristic(build_eaf(jobs, {}));
  EXPECT_FALSE(outcome.found);
}

TEST(arc_flow, rejects_model_without_unit_incidence)
{
  built_model_t knapsack;
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

TEST(arc_flow, accepts_implicit_zero_lower_bounds)
{
  input_options_t options;
  options.set_lower_bounds = false;
  const auto outcome       = run_heuristic(build_eaf(small_instance(), {}), options);
  ASSERT_TRUE(outcome.prescreened);
  ASSERT_TRUE(outcome.found);
}

TEST(arc_flow, rejects_implicit_infinite_upper_bounds)
{
  input_options_t options;
  options.set_upper_bounds = false;
  const auto outcome       = run_heuristic(build_eaf(small_instance(), {}), options);
  EXPECT_FALSE(outcome.prescreened);
  EXPECT_FALSE(outcome.found);
}

TEST(arc_flow, accepts_large_finite_capacities)
{
  auto model = build_eaf(small_instance(), {});
  std::fill(model.var_ub.begin(), model.var_ub.end(), std::numeric_limits<double>::max());
  const auto outcome = run_heuristic(model);
  ASSERT_TRUE(outcome.prescreened);
  ASSERT_TRUE(outcome.found);
}

TEST(arc_flow, accepts_affine_cost_intercept)
{
  build_options_t options;
  options.cost_intercept = 7.0;
  const auto outcome     = run_heuristic(build_eaf(small_instance(), options));
  ASSERT_TRUE(outcome.prescreened);
  ASSERT_TRUE(outcome.found);
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
