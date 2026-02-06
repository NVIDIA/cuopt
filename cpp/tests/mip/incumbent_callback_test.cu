/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/error.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

class test_set_solution_callback_t : public cuopt::internals::set_solution_callback_t {
 public:
  test_set_solution_callback_t(std::vector<std::pair<std::vector<double>, double>>& solutions_,
                               void* expected_user_data_)
    : solutions(solutions_), expected_user_data(expected_user_data_), n_calls(0)
  {
  }
  // This will check that the we are able to recompute our own solution
  void set_solution(void* data, void* cost, void* solution_bound, void* user_data) override
  {
    EXPECT_EQ(user_data, expected_user_data);
    n_calls++;
    auto bound_ptr = static_cast<double*>(solution_bound);
    EXPECT_FALSE(std::isnan(bound_ptr[0]));
    auto assignment = static_cast<double*>(data);
    auto cost_ptr   = static_cast<double*>(cost);
    if (solutions.empty()) { return; }

    auto const& [last_assignment, last_cost] = solutions.back();
    std::copy(last_assignment.begin(), last_assignment.end(), assignment);
    *cost_ptr = last_cost;
  }
  std::vector<std::pair<std::vector<double>, double>>& solutions;
  void* expected_user_data;
  int n_calls;
};

class test_get_solution_callback_t : public cuopt::internals::get_solution_callback_t {
 public:
  test_get_solution_callback_t(std::vector<std::pair<std::vector<double>, double>>& solutions_in,
                               int n_variables_,
                               void* expected_user_data_)
    : solutions(solutions_in),
      expected_user_data(expected_user_data_),
      n_calls(0),
      n_variables(n_variables_)
  {
  }
  void get_solution(void* data, void* cost, void* solution_bound, void* user_data) override
  {
    EXPECT_EQ(user_data, expected_user_data);
    n_calls++;
    auto bound_ptr = static_cast<double*>(solution_bound);
    EXPECT_FALSE(std::isnan(bound_ptr[0]));
    auto assignment_ptr = static_cast<double*>(data);
    auto cost_ptr       = static_cast<double*>(cost);
    std::vector<double> assignment(assignment_ptr, assignment_ptr + n_variables);
    solutions.push_back(std::make_pair(std::move(assignment), *cost_ptr));
  }
  std::vector<std::pair<std::vector<double>, double>>& solutions;
  void* expected_user_data;
  int n_calls;
  int n_variables;
};

void check_solutions(const test_get_solution_callback_t& get_solution_callback,
                     const cuopt::mps_parser::mps_data_model_t<int, double>& op_problem,
                     const cuopt::linear_programming::mip_solver_settings_t<int, double>& settings)
{
  for (const auto& solution : get_solution_callback.solutions) {
    EXPECT_EQ(solution.first.size(), op_problem.get_variable_lower_bounds().size());
    test_variable_bounds(op_problem, solution.first, settings);
    const double unscaled_acceptable_tol = 0.1;
    test_constraint_sanity_per_row(
      op_problem,
      solution.first,
      // because of scaling the values are not as accurate, so add more relative tolerance
      unscaled_acceptable_tol,
      settings.tolerances.relative_tolerance);
    test_objective_sanity(op_problem, solution.first, solution.second, 1e-4);
  }
}

void test_incumbent_callback(std::string test_instance, bool include_set_callback)
{
  const raft::handle_t handle_{};
  std::cout << "Running: " << test_instance << std::endl;
  auto path = make_path_absolute(test_instance);
  cuopt::mps_parser::mps_data_model_t<int, double> mps_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);

  auto settings       = mip_solver_settings_t<int, double>{};
  settings.time_limit = 30.;
  settings.presolve   = true;
  int user_data       = 42;
  std::vector<std::pair<std::vector<double>, double>> solutions;
  test_get_solution_callback_t get_solution_callback(
    solutions, op_problem.get_n_variables(), &user_data);
  settings.set_mip_callback(&get_solution_callback, &user_data);
  std::unique_ptr<test_set_solution_callback_t> set_solution_callback;
  if (include_set_callback) {
    set_solution_callback = std::make_unique<test_set_solution_callback_t>(solutions, &user_data);
    settings.set_mip_callback(set_solution_callback.get(), &user_data);
  }
  auto solution = solve_mip(op_problem, settings);
  EXPECT_GE(get_solution_callback.n_calls, 1);
  if (include_set_callback) { EXPECT_GE(set_solution_callback->n_calls, 1); }
  check_solutions(get_solution_callback, mps_problem, settings);
}

TEST(mip_solve, incumbent_get_callback_test)
{
  std::vector<std::string> test_instances = {
    "mip/50v-10.mps", "mip/neos5-free-bound.mps", "mip/swath1.mps"};
  for (const auto& test_instance : test_instances) {
    test_incumbent_callback(test_instance, false);
  }
}

TEST(mip_solve, incumbent_get_set_callback_test)
{
  std::vector<std::string> test_instances = {
    "mip/50v-10.mps", "mip/neos5-free-bound.mps", "mip/swath1.mps"};
  for (const auto& test_instance : test_instances) {
    test_incumbent_callback(test_instance, true);
  }
}

}  // namespace cuopt::linear_programming::test
