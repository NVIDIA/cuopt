/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

class test_set_solution_callback_t : public cuopt::internals::set_solution_callback_t {
 public:
  test_set_solution_callback_t(
    std::vector<std::pair<rmm::device_uvector<double>, double>>& solutions_)
    : solutions(solutions_), n_calls(0)
  {
  }
  // This will check that the we are able to recompute our own solution
  void set_solution(void* data, void* cost) override
  {
    n_calls++;
    rmm::cuda_stream_view stream{};
    auto assignment = static_cast<double*>(data);
    auto cost_ptr   = static_cast<double*>(cost);
    if (solutions.empty()) { return; }

    auto const& [last_assignment, last_cost] = solutions.back();
    raft::copy(assignment, last_assignment.data(), last_assignment.size(), stream);
    raft::copy(cost_ptr, &last_cost, 1, stream);
    stream.synchronize();
  }
  std::vector<std::pair<rmm::device_uvector<double>, double>>& solutions;
  int n_calls;
};

class test_get_solution_callback_t : public cuopt::internals::get_solution_callback_t {
 public:
  test_get_solution_callback_t(
    std::vector<std::pair<rmm::device_uvector<double>, double>>& solutions_in, int n_variables_)
    : solutions(solutions_in), n_calls(0), n_variables(n_variables_)
  {
  }
  void get_solution(void* data, void* cost) override
  {
    n_calls++;
    rmm::cuda_stream_view stream{};
    rmm::device_uvector<double> assignment(n_variables, stream);
    raft::copy(assignment.data(), static_cast<double*>(data), n_variables, stream);
    auto h_cost = 0.;
    raft::copy(&h_cost, static_cast<double*>(cost), 1, stream);
    stream.synchronize();
    solutions.push_back(std::make_pair(std::move(assignment), h_cost));
  }
  std::vector<std::pair<rmm::device_uvector<double>, double>>& solutions;
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

void test_incumbent_callback(std::string test_instance)
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
  std::vector<std::pair<rmm::device_uvector<double>, double>> solutions;
  test_get_solution_callback_t get_solution_callback(solutions, op_problem.get_n_variables());
  test_set_solution_callback_t set_solution_callback(solutions);
  settings.set_mip_callback(&get_solution_callback);
  settings.set_mip_callback(&set_solution_callback);
  auto solution = solve_mip(op_problem, settings);
  EXPECT_GE(get_solution_callback.n_calls, 1);
  EXPECT_GE(set_solution_callback.n_calls, 1);
  check_solutions(get_solution_callback, mps_problem, settings);
}

TEST(mip_solve, incumbent_callback_test)
{
  std::vector<std::string> test_instances = {
    "mip/50v-10.mps", "mip/neos5-free-bound.mps", "mip/swath1.mps"};
  for (const auto& test_instance : test_instances) {
    test_incumbent_callback(test_instance);
  }
}

}  // namespace cuopt::linear_programming::test
