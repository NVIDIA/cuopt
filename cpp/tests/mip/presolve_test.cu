/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/linear_programming/solve.hpp>
#include <linear_programming/utils.cuh>
#include <mip/presolve/third_party_presolve.hpp>
#include <mip/problem/problem.cuh>
#include <mps_parser/mps_data_model.hpp>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

TEST(problem, find_implied_integers)
{
  const raft::handle_t handle_{};

  auto path           = make_path_absolute("mip/fiball.mps");
  auto mps_data_model = cuopt::mps_parser::parse_mps<int, double>(path, false);
  auto op_problem     = mps_data_model_to_optimization_problem(&handle_, mps_data_model);
  auto presolver      = std::make_unique<detail::third_party_presolve_t<int, double>>();
  auto result         = presolver->apply(
    op_problem, cuopt::linear_programming::problem_category_t::MIP, false, 1e-6, 1e-12, 20, 1);
  ASSERT_TRUE(result.has_value());

  auto problem = detail::problem_t<int, double>(result->reduced_problem);
  problem.set_implied_integers(result->implied_integer_indices);
  ASSERT_TRUE(result->implied_integer_indices.size() > 0);
  auto var_types = host_copy(problem.variable_types);
  // Find the index of the one continuous variable
  auto it = std::find_if(var_types.begin(), var_types.end(), [](var_t var_type) {
    return var_type == var_t::CONTINUOUS;
  });
  ASSERT_NE(it, var_types.end());
  ASSERT_EQ(problem.presolve_data.var_flags.size(), var_types.size());
  // Ensure it is an implied integer
  EXPECT_EQ(problem.presolve_data.var_flags.element(it - var_types.begin(), handle_.get_stream()),
            ((int)detail::problem_t<int, double>::var_flags_t::VAR_IMPLIED_INTEGER));
}

}  // namespace cuopt::linear_programming::test
