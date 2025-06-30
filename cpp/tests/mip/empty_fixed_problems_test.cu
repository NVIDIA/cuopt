/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <linear_programming/pdlp.cuh>
#include <linear_programming/utilities/problem_checking.cuh>
#include <mip/presolve/trivial_presolve.cuh>
#include <mip/relaxed_lp/relaxed_lp.cuh>
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

static std::pair<mip_termination_status_t, double> test_mps_file(std::string test_instance)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute(test_instance);
  cuopt::mps_parser::mps_data_model_t<int, double> problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  mip_solver_settings_t<int, double> settings;
  settings.time_limit                  = 1;
  settings.heuristics_only             = true;
  mip_solution_t<int, double> solution = solve_mip(&handle_, problem, settings);
  return std::make_pair(solution.get_termination_status(), solution.get_objective_value());
}

TEST(mip_solve, fixed_problem_test)
{
  auto [termination_status, obj_val] = test_mps_file("mip/fixed-problem.mps");
  EXPECT_EQ(termination_status, mip_termination_status_t::Optimal);
  EXPECT_NEAR(obj_val, 65, 1e-5);
}

TEST(mip_solve, fixed_problem_infeasible_test)
{
  auto [termination_status, obj_val] = test_mps_file("mip/fixed-problem-infeas.mps");
  EXPECT_EQ(termination_status, mip_termination_status_t::Infeasible);
}
TEST(mip_solve, empty_problem_test)
{
  auto [termination_status, obj_val] = test_mps_file("mip/empty-problem-obj.mps");
  EXPECT_EQ(termination_status, mip_termination_status_t::Optimal);
  EXPECT_NEAR(obj_val, 81, 1e-5);
}

TEST(mip_solve, empty_problem_with_objective_test)
{
  auto [termination_status, obj_val] = test_mps_file("mip/empty-problem-objective-vars.mps");
  EXPECT_EQ(termination_status, mip_termination_status_t::Optimal);
  EXPECT_NEAR(obj_val, -2, 1e-5);
}

TEST(mip_solve, empty_max_problem_with_objective_test)
{
  auto [termination_status, obj_val] = test_mps_file("mip/empty-max-problem-objective-vars.mps");
  EXPECT_EQ(termination_status, mip_termination_status_t::Optimal);
  EXPECT_NEAR(obj_val, 11, 1e-5);
}

}  // namespace cuopt::linear_programming::test
