/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/optimization_problem.hpp>
#include <mip_heuristics/structural/markshare.cuh>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

namespace {

using solver_t = markshare_t<int, double>;

constexpr int n_rows            = 3;
constexpr int n_binaries        = 6;
constexpr double infinity_value = std::numeric_limits<double>::infinity();

// The coefficient block, row major: three knapsacks over six binaries.
const std::vector<std::vector<double>> coefficients = {
  {3, 5, 7, 2, 4, 6},
  {2, 8, 1, 9, 3, 5},
  {6, 4, 9, 3, 7, 2},
};
const std::vector<double> rhs = {15, 7, 22};

// Column layout is [slack_0 .. slack_{m-1}] [partner_0 .. partner_{m-1}]? [binary_0 ..], mirroring
// the two shapes the MIPLIB files use: markshare_4_0 has a single slack per row, while markshare1
// and markshare2 spell the slack out as a pair whose second half is fixed at zero.
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
  bool fixed_partner{false};  // the markshare1 / markshare2 slack-pair shape
  bool finite_slack_bound{false};
  bool binary_carries_cost{false};
  bool slack_coefficient_two{false};
  bool inequality_row{false};
  bool fractional_coefficient{false};
};

built_model_t build_market_split(build_options_t options = {})
{
  const int n_slacks  = options.fixed_partner ? 2 * n_rows : n_rows;
  const int n_columns = n_slacks + n_binaries;

  built_model_t model;
  model.offsets.push_back(0);
  for (int k = 0; k < n_rows; ++k) {
    // The row's own slack, then its fixed partner when the pair shape is requested.
    model.indices.push_back(k);
    model.values.push_back(options.slack_coefficient_two ? 2.0 : 1.0);
    if (options.fixed_partner) {
      model.indices.push_back(n_rows + k);
      model.values.push_back(1.0);
    }
    for (int j = 0; j < n_binaries; ++j) {
      model.indices.push_back(n_slacks + j);
      double coefficient = coefficients[k][j];
      if (options.fractional_coefficient && k == 0 && j == 0) { coefficient += 0.5; }
      model.values.push_back(coefficient);
    }
    model.offsets.push_back((int)model.indices.size());
  }

  model.row_ub = rhs;
  model.row_lb = rhs;
  if (options.inequality_row) { model.row_lb[0] = -infinity_value; }

  model.obj.assign(n_columns, 0.0);
  model.var_lb.assign(n_columns, 0.0);
  model.var_ub.assign(n_columns, 1.0);
  model.var_types.assign(n_columns, var_t::INTEGER);

  const double slack_bound =
    options.finite_slack_bound ? std::accumulate(rhs.begin(), rhs.end(), 0.0) : infinity_value;
  for (int k = 0; k < n_rows; ++k) {
    model.obj[k]       = 1.0;
    model.var_ub[k]    = slack_bound;
    model.var_types[k] = var_t::CONTINUOUS;
    if (options.fixed_partner) {
      model.obj[n_rows + k]       = -1.0;
      model.var_ub[n_rows + k]    = 0.0;
      model.var_types[n_rows + k] = var_t::CONTINUOUS;
    }
  }
  if (options.binary_carries_cost) { model.obj[n_slacks] = 1.0; }

  return model;
}

// Minimum total slack over all 2^n binary assignments that keep every row activity at or below
// its right hand side. This is the reference the search has to match exactly -- it is what makes
// the level ascent an optimality proof rather than just a feasibility claim.
double brute_force_optimum()
{
  double best = infinity_value;
  for (int mask = 0; mask < (1 << n_binaries); ++mask) {
    double total    = 0.0;
    bool admissible = true;
    for (int k = 0; k < n_rows && admissible; ++k) {
      double activity = 0.0;
      for (int j = 0; j < n_binaries; ++j) {
        if ((mask >> j & 1) != 0) { activity += coefficients[k][j]; }
      }
      if (activity > rhs[k]) {
        admissible = false;
      } else {
        total += rhs[k] - activity;
      }
    }
    if (admissible) { best = std::min(best, total); }
  }
  return best;
}

struct run_outcome_t {
  bool recognized{false};
  bool found{false};
  std::vector<double> assignment;
  double objective{0.0};
};

run_outcome_t run_heuristic(const built_model_t& model)
{
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
  run_outcome_t outcome;
  solver_t heuristic;
  outcome.recognized = heuristic.recognize(problem, settings.get_tolerances());
  if (!outcome.recognized) { return outcome; }

  std::atomic<bool> preemption{false};
  outcome.found = heuristic.solve(settings.get_tolerances(), preemption, outcome.assignment);
  if (outcome.found) {
    for (size_t j = 0; j < outcome.assignment.size(); ++j) {
      outcome.objective += model.obj[j] * outcome.assignment[j];
    }
  }
  return outcome;
}

void expect_feasible(const built_model_t& model, const std::vector<double>& assignment)
{
  ASSERT_EQ(assignment.size(), model.obj.size());
  for (size_t j = 0; j < assignment.size(); ++j) {
    EXPECT_GE(assignment[j], model.var_lb[j] - 1e-9) << "column " << j;
    EXPECT_LE(assignment[j], model.var_ub[j] + 1e-9) << "column " << j;
    if (model.var_types[j] == var_t::INTEGER) {
      EXPECT_NEAR(assignment[j], std::round(assignment[j]), 1e-9) << "column " << j;
    }
  }
  for (int k = 0; k < n_rows; ++k) {
    double activity = 0.0;
    for (int e = model.offsets[k]; e < model.offsets[k + 1]; ++e) {
      activity += model.values[e] * assignment[model.indices[e]];
    }
    EXPECT_NEAR(activity, model.row_ub[k], 1e-6) << "row " << k;
  }
}

}  // namespace

TEST(markshare, solves_the_single_slack_shape)
{
  const auto model   = build_market_split();
  const auto outcome = run_heuristic(model);
  ASSERT_TRUE(outcome.recognized);
  ASSERT_TRUE(outcome.found);
  expect_feasible(model, outcome.assignment);
  EXPECT_DOUBLE_EQ(outcome.objective, brute_force_optimum());
}

TEST(markshare, solves_the_fixed_partner_shape)
{
  const auto model   = build_market_split({.fixed_partner = true});
  const auto outcome = run_heuristic(model);
  ASSERT_TRUE(outcome.recognized);
  ASSERT_TRUE(outcome.found);
  expect_feasible(model, outcome.assignment);
  EXPECT_DOUBLE_EQ(outcome.objective, brute_force_optimum());
}

TEST(markshare, accepts_a_finite_slack_upper_bound)
{
  const auto model   = build_market_split({.finite_slack_bound = true});
  const auto outcome = run_heuristic(model);
  ASSERT_TRUE(outcome.recognized);
  ASSERT_TRUE(outcome.found);
  expect_feasible(model, outcome.assignment);
  EXPECT_DOUBLE_EQ(outcome.objective, brute_force_optimum());
}

TEST(markshare, rejects_a_binary_carrying_objective_cost)
{
  EXPECT_FALSE(run_heuristic(build_market_split({.binary_carries_cost = true})).recognized);
}

TEST(markshare, rejects_a_non_unit_slack)
{
  EXPECT_FALSE(run_heuristic(build_market_split({.slack_coefficient_two = true})).recognized);
}

TEST(markshare, rejects_an_inequality_row)
{
  EXPECT_FALSE(run_heuristic(build_market_split({.inequality_row = true})).recognized);
}

TEST(markshare, rejects_a_fractional_coefficient)
{
  EXPECT_FALSE(run_heuristic(build_market_split({.fractional_coefficient = true})).recognized);
}

TEST(markshare, declares_itself_exclusive)
{
  solver_t heuristic;
  EXPECT_TRUE(heuristic.exclusive());
}

}  // namespace cuopt::mathematical_optimization::mip
