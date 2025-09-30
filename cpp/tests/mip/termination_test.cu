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

constexpr double default_time_limit    = 10;
constexpr bool default_heuristics_only = true;

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
  EXPECT_EQ(obj_val, 2);
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

}  // namespace cuopt::linear_programming::test
