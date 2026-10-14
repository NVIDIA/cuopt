/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/mathematical_optimization/mip/solver_solution.hpp>
#include <mip_heuristics/presolve/trivial_presolve.cuh>
#include <mip_heuristics/relaxed_lp/relaxed_lp.cuh>
#include <pdlp/pdlp.cuh>
#include <pdlp/utilities/problem_checking.cuh>
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

namespace cuopt::mathematical_optimization::test {

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
// Papilo proves this problem infeasible almost immediately, while the early heuristics started
// before presolve may still be queued. Each of them has to be asked to exit before any of them is
// waited for; otherwise the wait can run another heuristic's queued task to completion on the
// calling thread and the solve never returns. The hang is a race, so the problem is solved
// repeatedly.
TEST(mip_solve, presolve_infeasible_repeated_solves)
{
  for (int i = 0; i < 100; ++i) {
    auto [termination_status, obj_val, lb] = test_mps_file("mip/presolve-infeasible-small.mps");
    ASSERT_EQ(termination_status, mip_termination_status_t::Infeasible) << "solve " << i;
  }
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

TEST(mip_solve, mip_presolved_to_lp)
{
  auto [termination_status, obj_val, lb] = test_mps_file("mip/mip-presolved-to-lp.mps", 5, false);
  EXPECT_EQ(termination_status, mip_termination_status_t::Optimal);
  EXPECT_NEAR(obj_val, 0, 1e-5);
}

}  // namespace cuopt::mathematical_optimization::test
