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

TEST(mip_solve, fixed_problem_test)
{
  auto [termination_status, obj_val, lb] = test_mps_file("mip/fixed-problem.mps");
  EXPECT_EQ(termination_status, mip_termination_status_t::Optimal);
  EXPECT_NEAR(obj_val, 65, 1e-5);
}

TEST(mip_solve, fixed_problem_infeasible_test)
{
  auto [termination_status, obj_val, lb] = test_mps_file("mip/fixed-problem-infeas.mps");
  EXPECT_EQ(termination_status, mip_termination_status_t::Infeasible);
}
TEST(mip_solve, empty_problem_test)
{
  auto [termination_status, obj_val, lb] = test_mps_file("mip/empty-problem-obj.mps");
  EXPECT_EQ(termination_status, mip_termination_status_t::Optimal);
  EXPECT_NEAR(obj_val, 81, 1e-5);
}

TEST(mip_solve, empty_problem_with_objective_test)
{
  auto [termination_status, obj_val, lb] = test_mps_file("mip/empty-problem-objective-vars.mps");
  EXPECT_EQ(termination_status, mip_termination_status_t::Optimal);
  EXPECT_NEAR(obj_val, -2, 1e-5);
}

TEST(mip_solve, empty_max_problem_with_objective_test)
{
  auto [termination_status, obj_val, lb] =
    test_mps_file("mip/empty-max-problem-objective-vars.mps");
  EXPECT_EQ(termination_status, mip_termination_status_t::Optimal);
  EXPECT_NEAR(obj_val, 11, 1e-5);
}

}  // namespace cuopt::linear_programming::test
