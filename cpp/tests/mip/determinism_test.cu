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
#include <cuopt/linear_programming/utilities/internals.hpp>
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
#include <cstdlib>
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

struct callback_solution_t {
  std::vector<double> assignment;
  double objective{};
  double solution_bound{};
  internals::mip_solution_origin_t origin{internals::mip_solution_origin_t::UNKNOWN};
};

class first_n_get_solution_callback_t : public cuopt::internals::get_solution_callback_ext_t {
 public:
  first_n_get_solution_callback_t(std::vector<callback_solution_t>& solutions_in,
                                  int n_variables_,
                                  size_t max_solutions_,
                                  void* expected_user_data_)
    : solutions(solutions_in),
      expected_user_data(expected_user_data_),
      n_variables(n_variables_),
      max_solutions(max_solutions_)
  {
  }

  void get_solution(void* data,
                    void* cost,
                    void* solution_bound,
                    const internals::mip_solution_callback_info_t* callback_info,
                    void* user_data) override
  {
    EXPECT_EQ(user_data, expected_user_data);
    ASSERT_NE(callback_info, nullptr);
    n_calls++;

    auto assignment_ptr     = static_cast<double*>(data);
    auto objective_ptr      = static_cast<double*>(cost);
    auto solution_bound_ptr = static_cast<double*>(solution_bound);
    EXPECT_FALSE(std::isnan(objective_ptr[0]));
    EXPECT_FALSE(std::isnan(solution_bound_ptr[0]));

    if (solutions.size() >= max_solutions) { return; }

    callback_solution_t callback_solution;
    callback_solution.assignment.assign(assignment_ptr, assignment_ptr + n_variables);
    callback_solution.objective      = objective_ptr[0];
    callback_solution.solution_bound = solution_bound_ptr[0];
    callback_solution.origin         = (internals::mip_solution_origin_t)callback_info->origin;
    solutions.push_back(std::move(callback_solution));
  }

  std::vector<callback_solution_t>& solutions;
  void* expected_user_data;
  int n_calls{0};
  int n_variables;
  size_t max_solutions;
};

bool is_gpu_callback_origin(internals::mip_solution_origin_t origin)
{
  switch (origin) {
    case internals::mip_solution_origin_t::FEASIBILITY_JUMP:
    case internals::mip_solution_origin_t::LOCAL_SEARCH:
    case internals::mip_solution_origin_t::LP_ROUNDING:
    case internals::mip_solution_origin_t::RECOMBINATION:
    case internals::mip_solution_origin_t::SUB_MIP: return true;
    default: return false;
  }
}

size_t count_callbacks_with_origin(const std::vector<callback_solution_t>& callbacks,
                                   internals::mip_solution_origin_t origin)
{
  return std::count_if(callbacks.begin(),
                       callbacks.end(),
                       [origin](const callback_solution_t& sol) { return sol.origin == origin; });
}

size_t count_gpu_callbacks(const std::vector<callback_solution_t>& callbacks)
{
  return std::count_if(callbacks.begin(), callbacks.end(), [](const callback_solution_t& sol) {
    return is_gpu_callback_origin(sol.origin);
  });
}

size_t count_branch_and_bound_callbacks(const std::vector<callback_solution_t>& callbacks)
{
  return std::count_if(callbacks.begin(), callbacks.end(), [](const callback_solution_t& sol) {
    return sol.origin == internals::mip_solution_origin_t::BRANCH_AND_BOUND_NODE ||
           sol.origin == internals::mip_solution_origin_t::BRANCH_AND_BOUND_DIVING;
  });
}

void expect_callback_prefixes_bitwise_equal(const std::vector<callback_solution_t>& lhs,
                                            const std::vector<callback_solution_t>& rhs,
                                            size_t prefix_size,
                                            const std::string& label)
{
  ASSERT_GE(lhs.size(), prefix_size) << label << "Left callback prefix missing entries";
  ASSERT_GE(rhs.size(), prefix_size) << label << "Right callback prefix missing entries";
  for (size_t i = 0; i < prefix_size; ++i) {
    EXPECT_EQ(lhs[i].objective, rhs[i].objective)
      << label << "Callback objective differs at index " << i;
    EXPECT_EQ(lhs[i].solution_bound, rhs[i].solution_bound)
      << label << "Callback bound differs at index " << i;
    EXPECT_EQ(lhs[i].origin, rhs[i].origin) << label << "Callback origin differs at index " << i;
    ASSERT_EQ(lhs[i].assignment.size(), rhs[i].assignment.size())
      << label << "Callback assignment size differs at index " << i;
    for (size_t j = 0; j < lhs[i].assignment.size(); ++j) {
      EXPECT_EQ(lhs[i].assignment[j], rhs[i].assignment[j])
        << label << "Callback assignment differs at callback " << i << " variable " << j;
    }
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
  settings.determinism_mode = CUOPT_MODE_DETERMINISTIC_BB;
  settings.num_cpu_threads  = 8;
  settings.work_limit       = 2;

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
  settings.determinism_mode = CUOPT_MODE_DETERMINISTIC_BB;
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
  settings.determinism_mode = CUOPT_MODE_DETERMINISTIC_BB;
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
  settings.determinism_mode = CUOPT_MODE_DETERMINISTIC_BB;
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

TEST_F(DeterministicBBTest, deterministic_callback_sequence_reproducible_with_gpu_pipeline)
{
  constexpr size_t callback_compare_count = 5;
  constexpr size_t callback_capture_limit = 32;
  constexpr size_t min_gpu_callback_count = 3;

  auto path    = make_path_absolute("/mip/50v-10.mps");
  auto problem = mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();

  mip_solver_settings_t<int, double> settings;
  settings.time_limit               = 360.0;
  settings.determinism_mode         = CUOPT_MODE_DETERMINISTIC;
  settings.num_cpu_threads          = 2;
  settings.work_limit               = 4;
  settings.bb_work_unit_scale       = 2.0;
  settings.gpu_heur_work_unit_scale = 1.0;
  settings.cpufj_work_unit_scale    = 1.0;

  auto seed = std::random_device{}() & 0x7fffffff;
  std::cout << "Tested with seed " << seed << "\n";
  settings.seed = seed;

  const int n_variables = problem.get_variable_lower_bounds().size();
  int user_data         = 7;

  std::vector<callback_solution_t> callbacks_run1;
  first_n_get_solution_callback_t callback_run1(
    callbacks_run1, n_variables, callback_capture_limit, &user_data);
  auto settings_run1 = settings;
  settings_run1.set_mip_callback(&callback_run1, &user_data);
  cuopt::seed_generator::set_seed(seed);
  auto solution1 = solve_mip(&handle_, problem, settings_run1);

  std::vector<callback_solution_t> callbacks_run2;
  first_n_get_solution_callback_t callback_run2(
    callbacks_run2, n_variables, callback_capture_limit, &user_data);
  auto settings_run2 = settings;
  settings_run2.set_mip_callback(&callback_run2, &user_data);
  cuopt::seed_generator::set_seed(seed);
  auto solution2 = solve_mip(&handle_, problem, settings_run2);

  EXPECT_EQ(solution1.get_termination_status(), solution2.get_termination_status());
  EXPECT_GE(callback_run1.n_calls, (int)callback_compare_count);
  EXPECT_GE(callback_run2.n_calls, (int)callback_compare_count);
  ASSERT_GE(callbacks_run1.size(), callback_compare_count);
  ASSERT_GE(callbacks_run2.size(), callback_compare_count);

  EXPECT_GE(count_gpu_callbacks(callbacks_run1), min_gpu_callback_count);
  EXPECT_GE(count_gpu_callbacks(callbacks_run2), min_gpu_callback_count);

  expect_callback_prefixes_bitwise_equal(
    callbacks_run1, callbacks_run2, callback_compare_count, "Deterministic callback run 1 vs 2: ");
}

class DeterministicGpuHeuristicsInstanceTest : public ::testing::TestWithParam<std::string> {
 protected:
  raft::handle_t handle_;
};

TEST_P(DeterministicGpuHeuristicsInstanceTest, reproducible_with_gpu_heuristics)
{
  auto path    = make_path_absolute(GetParam());
  auto problem = mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();

  mip_solver_settings_t<int, double> settings;
  settings.time_limit       = 60.0;
  settings.determinism_mode = CUOPT_MODE_DETERMINISTIC;
  settings.num_cpu_threads  = 8;
  settings.work_limit       = 5;

  auto seed = std::random_device{}() & 0x7fffffff;
  std::cout << "Tested with seed " << seed << "\n";
  settings.seed = seed;

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

  expect_solutions_bitwise_equal(solution1, solution2, handle_, "GPU heur run 1 vs 2: ");
  expect_solutions_bitwise_equal(solution1, solution3, handle_, "GPU heur run 1 vs 3: ");
}

INSTANTIATE_TEST_SUITE_P(
  DeterministicGpuHeuristics,
  DeterministicGpuHeuristicsInstanceTest,
  ::testing::Values(std::string("/mip/gen-ip054.mps"),
                    std::string("/mip/pk1.mps"),
                    // std::string("/mip/sct2.mps"),
                    // std::string("/mip/thor50dday.mps"),
                    std::string("/mip/neos5.mps")),
  [](const ::testing::TestParamInfo<DeterministicGpuHeuristicsInstanceTest::ParamType>& info) {
    std::string name = info.param.substr(info.param.rfind('/') + 1);
    name             = name.substr(0, name.rfind('.'));
    std::replace(name.begin(), name.end(), '-', '_');
    return name;
  });

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
    // Instance, threads, time_limit, work limit
    std::make_tuple("/mip/gen-ip054.mps", 4, 60.0, 4),
    std::make_tuple("/mip/swath1.mps", 8, 60.0, 4),
    std::make_tuple("/mip/50v-10.mps", 8, 60.0, 4),
    std::make_tuple("/mip/gen-ip054.mps", 128, 120.0, 1),
    std::make_tuple("/mip/bb_optimality.mps", 4, 60.0, 4),
    std::make_tuple("/mip/neos5.mps", 16, 60.0, 1),
    std::make_tuple("/mip/seymour1.mps", 16, 120.0, 1),
    // too heavy for CI
    // std::make_tuple("/mip/n2seq36q.mps", 16, 60.0, 4),
    std::make_tuple("/mip/gmu-35-50.mps", 32, 60.0, 2)),
  [](const ::testing::TestParamInfo<DeterministicBBInstanceTest::ParamType>& info) {
    const auto& path = std::get<0>(info.param);
    int threads      = std::get<1>(info.param);
    std::string name = path.substr(path.rfind('/') + 1);
    name             = name.substr(0, name.rfind('.'));
    std::replace(name.begin(), name.end(), '-', '_');
    return name + "_threads" + std::to_string(threads);
  });

}  // namespace cuopt::linear_programming::test
