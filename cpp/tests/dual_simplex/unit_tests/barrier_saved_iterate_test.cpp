/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <barrier/saved_iterate.hpp>

#include <gtest/gtest.h>

#include <initializer_list>
#include <limits>

namespace cuopt::mathematical_optimization::barrier::test {
namespace {

using residuals_t = iterate_residuals_t<double>;

constexpr residuals_t tolerance{1e-6, 1e-6, 1e-6, 1e-6};
// A QP trajectory can improve its objective gap while primal and dual residuals increase.
constexpr residuals_t before_gap_convergence{1.70e-8, 1.42e-13, 9.76e-7, 5.87e-2};
constexpr residuals_t relaxed_solution{2.62e-7, 9.31e-10, 3.37e-13, 3.50e-8};

}  // namespace

TEST(BarrierSavedIterate, SavesRelaxedSolutionDespiteLargerFeasibilityResiduals)
{
  EXPECT_TRUE(should_save_iterate(relaxed_solution, before_gap_convergence, tolerance, true));
}

TEST(BarrierSavedIterate, RetainsRelaxedSolutionWhenObjectiveGapDeteriorates)
{
  const residuals_t better_residuals_bad_gap{1e-8, 1e-12, 1e-14, 1e-5};
  EXPECT_FALSE(should_save_iterate(better_residuals_bad_gap, relaxed_solution, tolerance, true));
}

TEST(BarrierSavedIterate, RequiresAllResidualsToImproveBeforeGapConvergence)
{
  const residuals_t all_improve{1e-9, 1e-14, 1e-8, 1e-2};
  const residuals_t worse_primal{2e-8, 1e-14, 1e-8, 1e-2};
  EXPECT_TRUE(should_save_iterate(all_improve, before_gap_convergence, tolerance, true));
  EXPECT_FALSE(should_save_iterate(worse_primal, before_gap_convergence, tolerance, true));
}

TEST(BarrierSavedIterate, RequiresAllResidualsToImproveBetweenRelaxedSolutions)
{
  const residuals_t all_improve{1e-8, 1e-12, 1e-14, 1e-8};
  const residuals_t worse_primal{3e-7, 1e-12, 1e-14, 1e-9};
  EXPECT_TRUE(should_save_iterate(all_improve, relaxed_solution, tolerance, true));
  EXPECT_FALSE(should_save_iterate(worse_primal, relaxed_solution, tolerance, true));
  EXPECT_FALSE(should_save_iterate(relaxed_solution, relaxed_solution, tolerance, true));
}

TEST(BarrierSavedIterate, LinearProblemPreservesResidualSelectionWithoutGapCheck)
{
  const residuals_t all_improve_large_gap{1e-8, 1e-12, 1e-14, 1.0};
  EXPECT_TRUE(should_save_iterate(all_improve_large_gap, relaxed_solution, tolerance, false));
  EXPECT_FALSE(should_save_iterate(relaxed_solution, before_gap_convergence, tolerance, false));
}

TEST(BarrierSavedIterate, SavesFirstFiniteCandidate)
{
  constexpr double inf = std::numeric_limits<double>::infinity();
  const residuals_t empty_saved{inf, inf, inf, inf};
  EXPECT_TRUE(should_save_iterate(relaxed_solution, empty_saved, tolerance, true));
  EXPECT_TRUE(should_save_iterate(before_gap_convergence, empty_saved, tolerance, true));
}

TEST(BarrierSavedIterate, RejectsCandidateOutsideRelaxedResidualTolerances)
{
  residuals_t current = relaxed_solution;
  current.primal      = tolerance.primal;
  EXPECT_FALSE(should_save_iterate(current, before_gap_convergence, tolerance, true));
  current      = relaxed_solution;
  current.dual = tolerance.dual;
  EXPECT_FALSE(should_save_iterate(current, before_gap_convergence, tolerance, true));
  current                 = relaxed_solution;
  current.complementarity = tolerance.complementarity;
  EXPECT_FALSE(should_save_iterate(current, before_gap_convergence, tolerance, true));
}

TEST(BarrierSavedIterate, ObjectiveGapAtToleranceDoesNotReplaceRelaxedSolution)
{
  const residuals_t better_residuals{1e-8, 1e-12, 1e-14, tolerance.objective_gap};
  EXPECT_FALSE(should_save_iterate(better_residuals, relaxed_solution, tolerance, true));
}

TEST(BarrierSavedIterate, RejectsNonfiniteObjectiveGapWhenRequired)
{
  for (double gap : {std::numeric_limits<double>::quiet_NaN(),
                     std::numeric_limits<double>::infinity(),
                     -std::numeric_limits<double>::infinity()}) {
    const residuals_t current{1e-8, 1e-12, 1e-14, gap};
    EXPECT_FALSE(should_save_iterate(current, relaxed_solution, tolerance, true));
    EXPECT_TRUE(should_save_iterate(current, relaxed_solution, tolerance, false));
  }
}

TEST(BarrierSavedIterate, RejectsNonfiniteResiduals)
{
  for (double residual :
       {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    residuals_t current = relaxed_solution;
    current.primal      = residual;
    EXPECT_FALSE(should_save_iterate(current, before_gap_convergence, tolerance, true));
    current      = relaxed_solution;
    current.dual = residual;
    EXPECT_FALSE(should_save_iterate(current, before_gap_convergence, tolerance, true));
    current                 = relaxed_solution;
    current.complementarity = residual;
    EXPECT_FALSE(should_save_iterate(current, before_gap_convergence, tolerance, true));
  }
}

}  // namespace cuopt::mathematical_optimization::barrier::test
