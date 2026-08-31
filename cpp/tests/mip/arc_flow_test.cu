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
};

struct build_options_t {
  bool row_slack_terminators{false};
  bool permute{false};
  bool large_finite_capacities{false};
  double flow_row_factor{1.0};
  double cost_intercept{0.0};
  int perturbed_cost_label{-1};
  int negative_cost_label{-1};
  int single_arc_label{-1};
};

enum state_t : int { t0, t1, t2, t3, t5, n_states };

struct arc_t {
  state_t from;
  state_t to;
  int label;
  double cost;
};

constexpr int n_labels = 3;
constexpr int n_paths  = 2;
constexpr double expected_objective = 6.0;
constexpr std::array<int, n_labels> label_demand{2, 1, 1};
constexpr int demanded_arcs = label_demand[0] + label_demand[1] + label_demand[2];
constexpr std::array<state_t, 3> terminator_states{t2, t3, t5};
constexpr std::array<arc_t, 8> arcs{{{t0, t1, 0, 0.0},
                                     {t1, t2, 0, 3.0},
                                     {t2, t3, 0, 6.0},
                                     {t0, t3, 1, 0.0},
                                     {t2, t5, 1, 4.0},
                                     {t0, t2, 2, 0.0},
                                     {t1, t3, 2, 1.0},
                                     {t3, t5, 2, 3.0}}};

built_model_t build_arc_flow(const build_options_t& opts = {})
{
  struct column_t {
    std::vector<std::pair<int, double>> entries;
    double cost;
    double ub;
  };

  std::vector<column_t> columns;
  for (const auto& arc : arcs) {
    if (arc.label == opts.single_arc_label && arc.from != t0) { continue; }
    double cost = arc.cost + opts.cost_intercept;
    if (arc.label == opts.negative_cost_label) { cost = -arc.cost; }
    if (arc.label == opts.perturbed_cost_label && arc.from == t1) { cost += 1.0; }
    const double ub = opts.large_finite_capacities ? std::numeric_limits<double>::max()
                                                   : label_demand[arc.label];
    columns.push_back(
      column_t{{{arc.from, 1.0}, {arc.to, -1.0}, {n_states + arc.label, 1.0}}, cost, ub});
  }
  if (!opts.row_slack_terminators) {
    const double ub =
      opts.large_finite_capacities ? std::numeric_limits<double>::max() : 1.0;
    for (const state_t state : terminator_states) {
      columns.push_back(column_t{{{state, 1.0}}, 0.0, ub});
    }
  }

  const int n_rows = n_states + n_labels;
  std::vector<double> row_lb(n_rows, 0.0);
  std::vector<double> row_ub(n_rows, 0.0);
  row_lb[t0] = row_ub[t0] = n_paths;
  if (opts.row_slack_terminators) {
    for (const state_t state : terminator_states) {
      row_lb[state] = -1.0;
      row_ub[state] = 0.0;
    }
  }
  for (int label = 0; label < n_labels; ++label) {
    row_lb[n_states + label] = label_demand[label];
    row_ub[n_states + label] = std::numeric_limits<double>::infinity();
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

struct run_outcome_t {
  bool prescreened{false};
  bool found{false};
  double objective{0.0};
  std::vector<double> assignment;
};

struct input_options_t {
  bool set_lower_bounds{true};
  bool set_upper_bounds{true};
  bool maximize{false};
  bool use_internal_problem{false};
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
  problem.set_maximize(options.maximize);

  mip_solver_settings_t<int, double> settings;
  run_outcome_t outcome;
  mip::arc_flow_t<int, double> heuristic;
  if (options.use_internal_problem) {
    mip::problem_t<int, double> internal_problem(problem, settings.get_tolerances(), false);
    outcome.prescreened = heuristic.recognize(internal_problem, settings.get_tolerances());
  } else {
    outcome.prescreened = heuristic.recognize(problem, settings.get_tolerances());
  }
  if (!outcome.prescreened) { return outcome; }

  std::atomic<bool> preemption{false};
  outcome.found = heuristic.solve(settings.get_tolerances(), preemption, outcome.assignment);
  if (outcome.found) {
    expect_feasible(model, outcome.assignment);
    outcome.objective = 0.0;
    for (size_t j = 0; j < outcome.assignment.size(); ++j) {
      outcome.objective += model.obj[j] * outcome.assignment[j];
    }
  }
  return outcome;
}

}  // namespace

TEST(arc_flow, finds_exact_optimum_on_reduced_graph)
{
  const auto outcome = run_heuristic(build_arc_flow());
  ASSERT_TRUE(outcome.prescreened);
  ASSERT_TRUE(outcome.found);
  EXPECT_DOUBLE_EQ(outcome.objective, expected_objective);
}

TEST(arc_flow, handles_maximization_in_both_recognizers)
{
  auto model = build_arc_flow();
  for (auto& coefficient : model.obj) {
    coefficient = -coefficient;
  }
  const auto user_problem = run_heuristic(model, {.maximize = true});
  const auto internal_problem =
    run_heuristic(model, {.maximize = true, .use_internal_problem = true});
  ASSERT_TRUE(user_problem.prescreened);
  ASSERT_TRUE(user_problem.found);
  ASSERT_TRUE(internal_problem.prescreened);
  ASSERT_TRUE(internal_problem.found);
  EXPECT_DOUBLE_EQ(user_problem.objective, -expected_objective);
  EXPECT_DOUBLE_EQ(internal_problem.objective, -expected_objective);
}

TEST(arc_flow, recognizes_large_scaled_demand_with_relative_tolerance)
{
  auto model                         = build_arc_flow();
  constexpr int cover_row            = n_states;
  constexpr int large_demand         = 19998;
  constexpr double row_scale         = 0.1;
  constexpr double normalized_offset = 1e-5;
  const double scaled_demand         = (large_demand + normalized_offset) * row_scale;
  model.row_lb[cover_row]             = scaled_demand;
  for (int k = model.offsets[cover_row]; k < model.offsets[cover_row + 1]; ++k) {
    model.values[k] = row_scale;
  }
  std::fill(model.var_ub.begin(), model.var_ub.end(), large_demand);

  const double normalized_demand = scaled_demand / row_scale;
  EXPECT_GT(std::abs(normalized_demand - std::round(normalized_demand)), 1e-9);

  const raft::handle_t handle{};
  optimization_problem_t<int, double> problem(&handle);
  problem.set_csr_constraint_matrix(model.values.data(),
                                    model.values.size(),
                                    model.indices.data(),
                                    model.indices.size(),
                                    model.offsets.data(),
                                    model.offsets.size());
  problem.set_objective_coefficients(model.obj.data(), model.obj.size());
  problem.set_variable_lower_bounds(model.var_lb.data(), model.var_lb.size());
  problem.set_variable_upper_bounds(model.var_ub.data(), model.var_ub.size());
  problem.set_variable_types(model.var_types.data(), model.var_types.size());
  problem.set_constraint_lower_bounds(model.row_lb.data(), model.row_lb.size());
  problem.set_constraint_upper_bounds(model.row_ub.data(), model.row_ub.size());

  mip_solver_settings_t<int, double> settings;
  mip::arc_flow_t<int, double> heuristic;
  EXPECT_TRUE(heuristic.recognize(problem, settings.get_tolerances()));
}

TEST(arc_flow, single_arc_label_is_ordered_but_not_exact)
{
  const auto outcome = run_heuristic(build_arc_flow({.single_arc_label = 1}));
  ASSERT_TRUE(outcome.prescreened);
  ASSERT_TRUE(outcome.found);
  EXPECT_DOUBLE_EQ(outcome.objective, expected_objective);
}

TEST(arc_flow, accepts_supported_variants)
{
  struct test_case_t {
    const char* name;
    build_options_t build_options;
    input_options_t input_options;
    double objective;
  };
  const std::array<test_case_t, 6> cases{{
    {.name = "permutation", .build_options = {.permute = true}, .objective = expected_objective},
    {.name          = "row scaling",
     .build_options = {.flow_row_factor = 4.0},
     .objective     = expected_objective},
    {.name          = "row-slack termination",
     .build_options = {.row_slack_terminators = true},
     .objective     = expected_objective},
    {.name          = "implicit zero lower bounds",
     .input_options = {.set_lower_bounds = false},
     .objective     = expected_objective},
    {.name          = "large finite capacities",
     .build_options = {.large_finite_capacities = true},
     .objective     = expected_objective},
    {.name          = "affine cost intercept",
     .build_options = {.cost_intercept = 7.0},
     .objective     = expected_objective + 7.0 * demanded_arcs},
  }};
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const auto outcome =
      run_heuristic(build_arc_flow(test_case.build_options), test_case.input_options);
    EXPECT_TRUE(outcome.prescreened);
    EXPECT_TRUE(outcome.found);
    if (outcome.found) { EXPECT_DOUBLE_EQ(outcome.objective, test_case.objective); }
  }
}

TEST(arc_flow, rejects_non_affine_costs)
{
  const auto outcome = run_heuristic(build_arc_flow({.perturbed_cost_label = 0}));
  EXPECT_FALSE(outcome.found);
}

TEST(arc_flow, rejects_negative_cost_slope)
{
  const auto outcome = run_heuristic(build_arc_flow({.negative_cost_label = 2}));
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

TEST(arc_flow, rejects_implicit_infinite_upper_bounds)
{
  const auto outcome = run_heuristic(build_arc_flow(), {.set_upper_bounds = false});
  EXPECT_FALSE(outcome.prescreened);
  EXPECT_FALSE(outcome.found);
}

TEST(arc_flow, is_reproducible)
{
  const auto model  = build_arc_flow();
  const auto first  = run_heuristic(model);
  const auto second = run_heuristic(model);
  ASSERT_TRUE(first.found);
  ASSERT_TRUE(second.found);
  EXPECT_DOUBLE_EQ(first.objective, second.objective);
  EXPECT_EQ(first.assignment, second.assignment);
}

}  // namespace cuopt::mathematical_optimization::test
