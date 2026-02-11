/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/error.hpp>
#include <utilities/seed_generator.cuh>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

namespace {

void expect_solutions_bitwise_equal(const mip_solution_t<int, double>& sol1,
                                    const mip_solution_t<int, double>& sol2,
                                    raft::handle_t& handle,
                                    const std::string& label = "")
{
  auto x1 = cuopt::host_copy(sol1.get_solution(), handle.get_stream());
  auto x2 = cuopt::host_copy(sol2.get_solution(), handle.get_stream());

  ASSERT_EQ(x1.size(), x2.size()) << label << "Solution sizes differ";
  for (size_t i = 0; i < x1.size(); ++i) {
    EXPECT_EQ(x1[i], x2[i]) << label << "Variable " << i << " differs";
  }
}

}  // namespace

class DeterministicBBTest : public ::testing::Test {
 protected:
  raft::handle_t handle_;
};

// Test that multiple runs with deterministic mode produce identical objective values
TEST_F(DeterministicBBTest, reproducible_objective)
{
  auto path    = make_path_absolute("/mip/gen-ip054.mps");
  auto problem = mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();

  mip_solver_settings_t<int, double> settings;
  settings.time_limit       = 60.0;
  settings.determinism_mode = CUOPT_MODE_DETERMINISTIC;
  settings.num_cpu_threads  = 8;
  settings.work_limit       = 4;

  // Ensure seed is positive int32_t
  auto seed = std::random_device{}() & 0x7fffffff;
  std::cout << "Tested with seed " << seed << "\n";
  settings.seed = seed;

  auto solution1 = solve_mip(&handle_, problem, settings);
  double obj1    = solution1.get_objective_value();
  auto status1   = solution1.get_termination_status();

  for (int i = 2; i <= 10; ++i) {
    auto solution = solve_mip(&handle_, problem, settings);
    double obj    = solution.get_objective_value();
    auto status   = solution.get_termination_status();

    EXPECT_EQ(status1, status) << "Termination status differs on run " << i;
    ASSERT_EQ(obj1, obj) << "Objective value differs on run " << i;
    expect_solutions_bitwise_equal(solution1, solution, handle_);
  }
}

TEST_F(DeterministicBBTest, reproducible_infeasibility)
{
  auto path    = make_path_absolute("/mip/stein9inf.mps");
  auto problem = mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();

  mip_solver_settings_t<int, double> settings;
  settings.time_limit       = 60.0;
  settings.determinism_mode = CUOPT_MODE_DETERMINISTIC;
  settings.num_cpu_threads  = 8;
  settings.work_limit       = 100;  // High enough to fully explore

  auto seed = std::random_device{}() & 0x7fffffff;
  std::cout << "Tested with seed " << seed << "\n";
  settings.seed = seed;

  auto solution1 = solve_mip(&handle_, problem, settings);
  auto status1   = solution1.get_termination_status();
  EXPECT_EQ(status1, mip_termination_status_t::Infeasible)
    << "First run should detect infeasibility";

  for (int i = 2; i <= 5; ++i) {
    auto solution = solve_mip(&handle_, problem, settings);
    auto status   = solution.get_termination_status();

    EXPECT_EQ(status1, status) << "Termination status differs on run " << i;
    EXPECT_EQ(status, mip_termination_status_t::Infeasible)
      << "Run " << i << " should detect infeasibility";
  }
}

// Test determinism under high thread contention
TEST_F(DeterministicBBTest, reproducible_high_contention)
{
  auto path    = make_path_absolute("/mip/gen-ip054.mps");
  auto problem = mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();

  mip_solver_settings_t<int, double> settings;
  settings.time_limit       = 60.0;
  settings.determinism_mode = CUOPT_MODE_DETERMINISTIC;
  settings.num_cpu_threads  = 128;  // High thread count to stress contention
  settings.work_limit       = 1;

  auto seed = std::random_device{}() & 0x7fffffff;

  std::cout << "Tested with seed " << seed << "\n";
  settings.seed = seed;

  std::vector<mip_solution_t<int, double>> solutions;

  constexpr int num_runs = 3;
  for (int run = 0; run < num_runs; ++run) {
    solutions.push_back(solve_mip(&handle_, problem, settings));
  }

  for (int i = 1; i < num_runs; ++i) {
    EXPECT_EQ(solutions[0].get_termination_status(), solutions[i].get_termination_status())
      << "Run " << i << " termination status differs from run 0";
    EXPECT_DOUBLE_EQ(solutions[0].get_objective_value(), solutions[i].get_objective_value())
      << "Run " << i << " objective differs from run 0";
    expect_solutions_bitwise_equal(
      solutions[0], solutions[i], handle_, "Run " + std::to_string(i) + " vs run 0: ");
  }
}

// Test that solution vectors are bitwise identical across runs
TEST_F(DeterministicBBTest, reproducible_solution_vector)
{
  auto path    = make_path_absolute("/mip/swath1.mps");
  auto problem = mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();

  mip_solver_settings_t<int, double> settings;
  settings.time_limit       = 60.0;
  settings.determinism_mode = CUOPT_MODE_DETERMINISTIC;
  settings.num_cpu_threads  = 8;
  settings.work_limit       = 2;

  auto seed = std::random_device{}() & 0x7fffffff;

  std::cout << "Tested with seed " << seed << "\n";
  settings.seed = seed;

  auto solution1 = solve_mip(&handle_, problem, settings);
  auto solution2 = solve_mip(&handle_, problem, settings);

  EXPECT_EQ(solution1.get_termination_status(), solution2.get_termination_status());
  EXPECT_DOUBLE_EQ(solution1.get_objective_value(), solution2.get_objective_value());
  expect_solutions_bitwise_equal(solution1, solution2, handle_);
}

// Parameterized test for different problem instances
class DeterministicBBInstanceTest
  : public ::testing::TestWithParam<std::tuple<std::string, int, double, int>> {
 protected:
  raft::handle_t handle_;
};

TEST_P(DeterministicBBInstanceTest, deterministic_across_runs)
{
  auto [instance_path, num_threads, time_limit, work_limit] = GetParam();
  auto path                                                 = make_path_absolute(instance_path);
  auto problem = mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();

  // Get a random seed for each run
  auto seed = std::random_device{}() & 0x7fffffff;

  std::cout << "Tested with seed " << seed << "\n";

  mip_solver_settings_t<int, double> settings;
  settings.time_limit       = time_limit;
  settings.determinism_mode = CUOPT_MODE_DETERMINISTIC;
  settings.num_cpu_threads  = num_threads;
  settings.work_limit       = work_limit;
  settings.seed             = seed;

  cuopt::seed_generator::set_seed(seed);
  auto solution1 = solve_mip(&handle_, problem, settings);
  cuopt::seed_generator::set_seed(seed);
  auto solution2 = solve_mip(&handle_, problem, settings);
  cuopt::seed_generator::set_seed(seed);
  auto solution3 = solve_mip(&handle_, problem, settings);

  EXPECT_EQ(solution1.get_termination_status(), solution2.get_termination_status());
  EXPECT_EQ(solution1.get_termination_status(), solution3.get_termination_status());

  EXPECT_DOUBLE_EQ(solution1.get_objective_value(), solution2.get_objective_value());
  EXPECT_DOUBLE_EQ(solution1.get_objective_value(), solution3.get_objective_value());

  EXPECT_DOUBLE_EQ(solution1.get_solution_bound(), solution2.get_solution_bound());
  EXPECT_DOUBLE_EQ(solution1.get_solution_bound(), solution3.get_solution_bound());

  expect_solutions_bitwise_equal(solution1, solution2, handle_, "Run 1 vs 2: ");
  expect_solutions_bitwise_equal(solution1, solution3, handle_, "Run 1 vs 3: ");
}

INSTANTIATE_TEST_SUITE_P(
  DeterministicBB,
  DeterministicBBInstanceTest,
  ::testing::Values(
    // Instance, threads, time_limit
    std::make_tuple("/mip/gen-ip054.mps", 4, 60.0, 4),
    std::make_tuple("/mip/swath1.mps", 8, 60.0, 4),
    std::make_tuple("/mip/gen-ip054.mps", 128, 120.0, 1),
    std::make_tuple("/mip/bb_optimality.mps", 4, 60.0, 4),
    std::make_tuple("/mip/neos5.mps", 16, 60.0, 1),
    std::make_tuple("/mip/seymour1.mps", 16, 60.0, 1),
    // too heavy for CI
    // std::make_tuple("/mip/n2seq36q.mps", 16, 60.0, 4),
    std::make_tuple("/mip/gmu-35-50.mps", 32, 60.0, 3)),
  [](const ::testing::TestParamInfo<DeterministicBBInstanceTest::ParamType>& info) {
    const auto& path = std::get<0>(info.param);
    int threads      = std::get<1>(info.param);
    std::string name = path.substr(path.rfind('/') + 1);
    name             = name.substr(0, name.rfind('.'));
    std::replace(name.begin(), name.end(), '-', '_');
    return name + "_threads" + std::to_string(threads);
  });

}  // namespace cuopt::linear_programming::test
