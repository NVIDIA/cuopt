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

void test_mps_file(result_map_t test_instance)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute(test_instance.file);
  cuopt::mps_parser::mps_data_model_t<int, double> problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  mip_solver_settings_t<int, double> settings;
  // set the time limit depending on we are in assert mode or not
#ifdef ASSERT_MODE
  constexpr double test_time_limit = 60.;
#else
  constexpr double test_time_limit = 30.;
#endif

  settings.time_limit                  = test_time_limit;
  mip_solution_t<int, double> solution = solve_mip(&handle_, problem, settings);
  EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::FeasibleFound);
  double obj_val = solution.get_objective_value();
  // for now keep a 100% error rate
  EXPECT_NEAR(test_instance.cost, obj_val, test_instance.cost);
  test_variable_bounds(problem, solution.get_solution(), settings);
  // TODO test integrality as well
}

TEST(mip_solve, run_small_tests)
{
  std::vector<result_map_t> test_instances = {
    {"mip/50v-10.mps", 11311031.}, {"mip/neos5.mps", 15.}, {"mip/swath1.mps", 1300.}};
  for (const auto& test_instance : test_instances) {
    test_mps_file(test_instance);
  }
}

}  // namespace cuopt::linear_programming::test
