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
#include "mip_utils.cuh"

#include <cuopt/linear_programming/solve.hpp>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/error.hpp>

namespace cuopt::linear_programming::test {
TEST(mip_solve, integer_with_real_bounds_test)
{
  auto time_limit      = 1;
  auto heuristics_only = true;
  auto presolve        = false;
  auto [termination_status, obj_val, lb] =
    test_mps_file("mip/integer-with-real-bounds.mps", time_limit, heuristics_only, presolve);
  EXPECT_EQ(termination_status, mip_termination_status_t::Optimal);
  EXPECT_NEAR(obj_val, 4, 1e-5);
}
}  // namespace cuopt::linear_programming::test
