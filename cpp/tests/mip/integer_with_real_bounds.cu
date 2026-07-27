/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/error.hpp>

namespace cuopt::mathematical_optimization::test {
TEST(mip_solve, integer_with_real_bounds_test)
{
  auto time_limit      = 1;
  auto heuristics_only = true;
  auto presolver       = cuopt::mathematical_optimization::presolver_t::None;
  auto [termination_status, obj_val, lb] =
    test_mps_file("mip/integer-with-real-bounds.mps", time_limit, heuristics_only, presolver);
  EXPECT_EQ(termination_status, mip_termination_status_t::Optimal);
  EXPECT_NEAR(obj_val, 4, 1e-5);
}
}  // namespace cuopt::mathematical_optimization::test
