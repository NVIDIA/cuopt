/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "cuopt/linear_programming/mip/solver_settings.hpp"
#include "dual_simplex/branch_and_bound.hpp"
#include "dual_simplex/simplex_solver_settings.hpp"
#include "mip_utils.cuh"

#include <cuopt/linear_programming/solve.hpp>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

struct result_map_t {
  std::string file;
  double cost;
};

void test_miplib_file(result_map_t test_instance, mip_solver_settings_t<int, double> settings)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute(test_instance.file);
  cuopt::mps_parser::mps_data_model_t<int, double> problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  // set the time limit depending on we are in assert mode or not
#ifdef ASSERT_MODE
  constexpr double test_time_limit = 60.;
#else
  constexpr double test_time_limit = 30.;
#endif

  settings.time_limit                  = test_time_limit;
  mip_solution_t<int, double> solution = solve_mip(&handle_, problem, settings);
  bool is_feasible = solution.get_termination_status() == mip_termination_status_t::FeasibleFound ||
                     solution.get_termination_status() == mip_termination_status_t::Optimal;
  EXPECT_TRUE(is_feasible);
  double obj_val = solution.get_objective_value();
  // for now keep a 100% error rate
  EXPECT_NEAR(test_instance.cost, obj_val, test_instance.cost);
  test_variable_bounds(problem, solution.get_solution(), settings);
  // TODO test integrality as well
}

static void test_variable_bounds(const cuopt::mps_parser::mps_data_model_t<int, double>& problem,
                                 const rmm::device_uvector<double>& solution,
                                 double integrality_tolerance)
{
  const double* lower_bound_ptr = problem.get_variable_lower_bounds().data();
  const double* upper_bound_ptr = problem.get_variable_upper_bounds().data();
  auto host_assignment          = cuopt::host_copy(solution);
  double* assignment_ptr        = host_assignment.data();
  cuopt_assert(host_assignment.size() == problem.get_variable_lower_bounds().size(), "");
  cuopt_assert(host_assignment.size() == problem.get_variable_upper_bounds().size(), "");
  std::vector<int> indices(host_assignment.size());

  std::iota(indices.begin(), indices.end(), 0);
  bool result = std::all_of(indices.begin(), indices.end(), [=](int idx) {
    bool res = true;
    if (lower_bound_ptr != nullptr) {
      res = res && (assignment_ptr[idx] >= lower_bound_ptr[idx] - integrality_tolerance);
    }
    if (upper_bound_ptr != nullptr) {
      res = res && (assignment_ptr[idx] <= upper_bound_ptr[idx] + integrality_tolerance);
    }
    return res;
  });
  EXPECT_TRUE(result);
}

void test_branch_and_bound_file(result_map_t test_instance,
                                dual_simplex::simplex_solver_settings_t<int, double> settings)
{
  auto path = make_path_absolute(test_instance.file);
  cuopt::mps_parser::mps_data_model_t<int, double> problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);

  // set the time limit depending on we are in assert mode or not
#ifdef ASSERT_MODE
  constexpr double test_time_limit = 60.;
#else
  constexpr double test_time_limit = 30.;
#endif

  settings.time_limit = test_time_limit;
  mip_solution_t<int, double> solution;

  dual_simplex::branch_and_bound_t<int, double> branch_and_bound(problem, settings);
  dual_simplex::mip_status_t status = branch_and_bound.solve(solution);
  bool is_feasible = solution.get_termination_status() == mip_termination_status_t::FeasibleFound ||
                     solution.get_termination_status() == mip_termination_status_t::Optimal;
  EXPECT_TRUE(is_feasible);
  double obj_val = solution.get_objective_value();
  EXPECT_NEAR(test_instance.cost, obj_val, test_instance.cost);
  test_variable_bounds(problem, solution.get_solution(), 1.0e-5);
}

TEST(mip_solve, run_small_tests)
{
  mip_solver_settings_t<int, double> settings;
  std::vector<result_map_t> test_instances = {
    {"mip/50v-10.mps", 11311031.}, {"mip/neos5.mps", 15.}, {"mip/swath1.mps", 1300.}};
  for (const auto& test_instance : test_instances) {
    test_miplib_file(test_instance, settings);
  }
}

TEST(mip_solve, branch_and_bound_test)
{
  dual_simplex::simplex_solver_settings_t<int, double> settings;
  std::vector<result_map_t> test_instances = {
    {"mip/50v-10.mps", 11311031.}, {"mip/neos5.mps", 15.}, {"mip/swath1.mps", 1300.}};

  std::vector<dual_simplex::search_strategy_t> search_strategies = {
    dual_simplex::search_strategy_t::BEST_FIRST,
    dual_simplex::search_strategy_t::DEPTH_FIRST,
    dual_simplex::search_strategy_t::MULTITHREADED_BEST_FIRST_WITH_DIVING};

  for (const auto& search_strategy : search_strategies) {
    settings.bnb_search_strategy = search_strategy;
    for (const auto& test_instance : test_instances) {
      test_branch_and_bound_file(test_instance, settings);
    }
  }
}
}  // namespace cuopt::linear_programming::test
