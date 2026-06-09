/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/linear_programming/io/parser.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <mip_heuristics/presolve/trivial_presolve.cuh>
#include <mip_heuristics/relaxed_lp/relaxed_lp.cuh>
#include <pdlp/pdlp.cuh>
#include <pdlp/utilities/problem_checking.cuh>
#include <utilities/common_utils.hpp>
#include <utilities/error.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

constexpr double default_time_limit    = 10;
constexpr bool default_heuristics_only = true;

namespace {

optimization_problem_t<int, double> make_concurrent_root_infeasible_problem(
  raft::handle_t const* handle)
{
  constexpr int n = 120;

  optimization_problem_t<int, double> problem(handle);
  std::vector<double> coefficients;
  std::vector<int> indices;
  coefficients.reserve(2 * n);
  indices.reserve(2 * n);
  std::vector<int> offsets = {0, n, 2 * n};
  std::vector<double> objective(n, 1.0);
  std::vector<double> var_lower(n, 0.0);
  std::vector<double> var_upper(n, 1.0);
  std::vector<var_t> var_types(n, var_t::INTEGER);

  double sum_a = 0.0;
  double sum_b = 0.0;
  for (int i = 0; i < n; ++i) {
    const double a = 1.0 + static_cast<double>((i * 37) % 90) / 10.0;
    const double b = a + (static_cast<double>((i * 17) % 11) - 5.0) / 20.0;
    coefficients.push_back(a);
    indices.push_back(i);
    sum_a += a;
    sum_b += b;
  }
  for (int i = 0; i < n; ++i) {
    const double a = 1.0 + static_cast<double>((i * 37) % 90) / 10.0;
    const double b = a + (static_cast<double>((i * 17) % 11) - 5.0) / 20.0;
    coefficients.push_back(b);
    indices.push_back(i);
  }

  const std::vector<double> row_lower = {0.80 * sum_a, -std::numeric_limits<double>::infinity()};
  const std::vector<double> row_upper = {std::numeric_limits<double>::infinity(), 0.30 * sum_b};

  problem.set_csr_constraint_matrix(coefficients.data(),
                                    coefficients.size(),
                                    indices.data(),
                                    indices.size(),
                                    offsets.data(),
                                    offsets.size());
  problem.set_constraint_lower_bounds(row_lower.data(), row_lower.size());
  problem.set_constraint_upper_bounds(row_upper.data(), row_upper.size());
  problem.set_objective_coefficients(objective.data(), objective.size());
  problem.set_variable_lower_bounds(var_lower.data(), var_lower.size());
  problem.set_variable_upper_bounds(var_upper.data(), var_upper.size());
  problem.set_variable_types(var_types.data(), var_types.size());

  return problem;
}

}  // namespace

TEST(termination_status, trivial_presolve_optimality_test)
{
  auto [termination_status, obj_val, lb] = test_mps_file(
    "mip/trivial-presolve-optimality.mps", default_time_limit, default_heuristics_only);
  EXPECT_EQ(termination_status, mip_termination_status_t::Optimal);
  EXPECT_EQ(obj_val, -1);
}

TEST(termination_status, trivial_presolve_no_obj_vars_test)
{
  auto [termination_status, obj_val, lb] = test_mps_file(
    "mip/trivial-presolve-no-obj-vars.mps", default_time_limit, default_heuristics_only);
  EXPECT_EQ(termination_status, mip_termination_status_t::Optimal);
  EXPECT_EQ(obj_val, 0);
}

TEST(termination_status, presolve_optimality_test)
{
  auto [termination_status, obj_val, lb] =
    test_mps_file("mip/sudoku.mps", default_time_limit, default_heuristics_only);
  EXPECT_EQ(termination_status, mip_termination_status_t::Optimal);
  EXPECT_EQ(obj_val, 0);
}

TEST(termination_status, presolve_infeasible_test)
{
  auto [termination_status, obj_val, lb] =
    test_mps_file("mip/presolve-infeasible.mps", default_time_limit, default_heuristics_only);
  EXPECT_EQ(termination_status, mip_termination_status_t::Infeasible);
}

TEST(termination_status, feasible_found_test)
{
  auto [termination_status, obj_val, lb] =
    test_mps_file("mip/gen-ip054.mps", default_time_limit, default_heuristics_only);
  EXPECT_EQ(termination_status, mip_termination_status_t::FeasibleFound);
}

TEST(termination_status, timeout_test)
{
  auto [termination_status, obj_val, lb] =
    test_mps_file("mip/stein9inf.mps", default_time_limit, default_heuristics_only);
  EXPECT_EQ(termination_status, mip_termination_status_t::TimeLimit);
}

TEST(termination_status, optimality_test)
{
  auto [termination_status, obj_val, lb] =
    test_mps_file("mip/bb_optimality.mps", default_time_limit, false);
  EXPECT_EQ(termination_status, mip_termination_status_t::Optimal);
  EXPECT_NEAR(obj_val, 2, 1e-6);
}

// Ensure the lower bound on maximization problems when BB times out has the right sign
TEST(termination_status, lower_bound_bb_timeout)
{
  auto [termination_status, obj_val, lb] = test_mps_file("mip/cod105_max.mps", 5.0, false);
  EXPECT_EQ(termination_status, mip_termination_status_t::FeasibleFound);
  EXPECT_GE(obj_val, 6);
  EXPECT_GE(lb, obj_val);
}

TEST(termination_status, crossing_bounds_infeasible)
{
  auto [termination_status, obj_val, lb] = test_mps_file("mip/crossing_var_bounds.mps", 0.5, false);
  EXPECT_EQ(termination_status, mip_termination_status_t::Infeasible);
}

TEST(termination_status, gf2_presolve_optimal)
{
  auto [termination_status, obj_val, lb] = test_mps_file("mip/enlight_hard.mps", 0.5, true);
  EXPECT_EQ(termination_status, mip_termination_status_t::Optimal);
}

TEST(termination_status, gf2_presolve_infeasible)
{
  auto [termination_status, obj_val, lb] = test_mps_file("mip/enlight11.mps", 0.5, true);
  EXPECT_EQ(termination_status, mip_termination_status_t::Infeasible);
}

TEST(termination_status, bb_infeasible_test)
{
  // First, check that presolve doesn't reduce the problem to infeasibility
  {
    auto [termination_status, obj_val, lb] = test_mps_file("mip/stein9inf.mps", 1, true);
    EXPECT_EQ(termination_status, mip_termination_status_t::TimeLimit);
  }
  // Ensure that B&B proves the MIP infeasible
  {
    auto [termination_status, obj_val, lb] = test_mps_file("mip/stein9inf.mps", 30, false);
    EXPECT_EQ(termination_status, mip_termination_status_t::Infeasible);
  }
}

TEST(termination_status, concurrent_root_infeasible_returns_status)
{
  const raft::handle_t handle_{};
  auto problem = make_concurrent_root_infeasible_problem(&handle_);
  handle_.sync_stream();

  mip_solver_settings_t<int, double> settings;
  settings.time_limit       = 15.0;
  settings.determinism_mode = CUOPT_MODE_OPPORTUNISTIC;
  settings.num_cpu_threads  = 8;

  auto solution = solve_mip(&handle_, problem, settings);
  EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Infeasible);
}

}  // namespace cuopt::linear_programming::test
