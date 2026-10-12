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
constexpr double residuals_t::* residual_members[]{&residuals_t::primal,
                                                   &residuals_t::dual,
                                                   &residuals_t::complementarity,
                                                   &residuals_t::objective_gap};

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

TEST(BarrierSavedIterate, RequiresGapConvergenceEvenWhenWorstMetricImproves)
{
  const residuals_t all_improve{1e-9, 1e-14, 1e-8, 1e-2};
  const residuals_t worse_primal{2e-8, 1e-14, 1e-8, 1e-2};
  EXPECT_FALSE(should_save_iterate(all_improve, before_gap_convergence, tolerance, true));
  EXPECT_FALSE(should_save_iterate(worse_primal, before_gap_convergence, tolerance, true));
  EXPECT_TRUE(should_save_iterate(all_improve, before_gap_convergence, tolerance, false));
  EXPECT_TRUE(should_save_iterate(worse_primal, before_gap_convergence, tolerance, false));
  const residuals_t better_residuals_worse_gap{1e-9, 1e-14, 1e-8, 1e-1};
  EXPECT_FALSE(
    should_save_iterate(better_residuals_worse_gap, before_gap_convergence, tolerance, true));
}

TEST(BarrierSavedIterate, RequiresWorstMetricToImproveBetweenRelaxedSolutions)
{
  const residuals_t all_improve{1e-8, 1e-12, 1e-14, 1e-8};
  const residuals_t worse_primal{3e-7, 1e-12, 1e-14, 1e-9};
  EXPECT_TRUE(should_save_iterate(all_improve, relaxed_solution, tolerance, true));
  EXPECT_FALSE(should_save_iterate(worse_primal, relaxed_solution, tolerance, true));
  EXPECT_FALSE(should_save_iterate(relaxed_solution, relaxed_solution, tolerance, true));
}

TEST(BarrierSavedIterate, SavesImprovedWorstMetricDespiteLargerOtherResiduals)
{
  const residuals_t saved{8e-7, 2e-7, 1e-7, 3e-7};
  const residuals_t current{6e-7, 4e-7, 2e-7, 5e-7};
  EXPECT_TRUE(should_save_iterate(current, saved, tolerance, true));
}

TEST(BarrierSavedIterate, RejectsWorseMeritEvenWhenAllMetricsPassRelaxedTolerances)
{
  const residuals_t saved{2e-7, 2e-7, 2e-7, 4e-7};
  const residuals_t current{1e-7, 1e-7, 1e-7, 6e-7};
  EXPECT_FALSE(should_save_iterate(current, saved, tolerance, true));
}

TEST(BarrierSavedIterate, SavesObjectiveGapImprovementWithUnchangedOtherResiduals)
{
  const residuals_t saved{1e-7, 1e-7, 1e-7, 8e-7};
  const residuals_t current{1e-7, 1e-7, 1e-7, 4e-7};
  EXPECT_TRUE(should_save_iterate(current, saved, tolerance, true));
}

TEST(BarrierSavedIterate, RetainsSavedIterateOnEqualMerit)
{
  const residuals_t saved{4e-7, 2e-7, 1e-7, 2e-7};
  const residuals_t current{3e-7, 1e-7, 5e-8, 4e-7};
  EXPECT_FALSE(should_save_iterate(current, saved, tolerance, true));
}

TEST(BarrierSavedIterate, NormalizesEachMetricByItsRelaxedTolerance)
{
  const residuals_t unequal_tolerances{1e-4, 1e-6, 1e-3, 1e-8};
  const residuals_t saved{5e-5, 5e-7, 5e-4, 5e-9};
  // The raw maximum decreases, but the gap uses more of its allowed tolerance.
  const residuals_t worse_normalized_gap{4e-5, 4e-7, 4e-4, 8e-9};
  EXPECT_FALSE(should_save_iterate(worse_normalized_gap, saved, unequal_tolerances, true));
  // The raw maximum increases, but the largest fraction of a tolerance decreases.
  const residuals_t better_normalized_merit{3e-5, 3e-7, 6e-4, 6e-9};
  EXPECT_TRUE(
    should_save_iterate(better_normalized_merit, worse_normalized_gap, unequal_tolerances, true));
}

TEST(BarrierSavedIterate, PreservesRecoverableIterateWithUnequalTolerances)
{
  const residuals_t unequal_tolerances{1e-4, 1e-6, 1e-3, 1e-8};
  const residuals_t saved{5e-5, 5e-7, 5e-4, 5e-9};
  for (double gap : {1e-8, 2e-8}) {
    const residuals_t current{4e-5, 4e-7, 4e-4, gap};
    EXPECT_FALSE(should_save_iterate(current, saved, unequal_tolerances, true));
    EXPECT_TRUE(should_save_iterate(saved, current, unequal_tolerances, true));
  }
}

TEST(BarrierSavedIterate, LinearProblemUsesWorstResidualWithoutGapCheck)
{
  const residuals_t all_improve_large_gap{1e-8, 1e-12, 1e-14, 1.0};
  EXPECT_TRUE(should_save_iterate(all_improve_large_gap, relaxed_solution, tolerance, false));
  EXPECT_TRUE(should_save_iterate(relaxed_solution, before_gap_convergence, tolerance, false));
  const residuals_t worse_primal_smaller_other_residuals{3e-7, 1e-12, 1e-14, 0.0};
  EXPECT_FALSE(
    should_save_iterate(worse_primal_smaller_other_residuals, relaxed_solution, tolerance, false));
}

TEST(BarrierSavedIterate, SavesFirstCandidatePassingAllEnabledTolerances)
{
  constexpr double inf = std::numeric_limits<double>::infinity();
  const residuals_t empty_saved{inf, inf, inf, inf};
  EXPECT_TRUE(should_save_iterate(relaxed_solution, empty_saved, tolerance, true));
  EXPECT_FALSE(should_save_iterate(before_gap_convergence, empty_saved, tolerance, true));
  EXPECT_TRUE(should_save_iterate(before_gap_convergence, empty_saved, tolerance, false));
}

TEST(BarrierSavedIterate, AppliesSameRelaxedToleranceBoundaryToEveryEnabledMetric)
{
  constexpr double inf = std::numeric_limits<double>::infinity();
  const residuals_t empty_saved{inf, inf, inf, inf};
  for (bool check_objective_gap : {false, true}) {
    for (auto member : residual_members) {
      for (double factor : {0.5, 1.0, 2.0}) {
        residuals_t current = relaxed_solution;
        current.*member     = factor * (tolerance.*member);
        const bool ignored  = !check_objective_gap && member == &residuals_t::objective_gap;
        EXPECT_EQ(should_save_iterate(current, empty_saved, tolerance, check_objective_gap),
                  ignored || factor < 1.0);
      }
    }
  }
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
  for (double residual : {std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::infinity(),
                          -std::numeric_limits<double>::infinity()}) {
    for (auto member : residual_members) {
      residuals_t current = relaxed_solution;
      current.*member     = residual;
      EXPECT_FALSE(should_save_iterate(current, before_gap_convergence, tolerance, true));
      EXPECT_EQ(should_save_iterate(current, before_gap_convergence, tolerance, false),
                member == &residuals_t::objective_gap);
    }
  }
}

TEST(BarrierSavedIterate, RejectsInvalidToleranceForEveryEnabledMetric)
{
  for (double invalid : {0.0,
                         -1e-6,
                         std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()}) {
    for (auto member : residual_members) {
      residuals_t invalid_tolerance = tolerance;
      invalid_tolerance.*member     = invalid;
      const residuals_t current{1e-8, 1e-12, 1e-14, 0.0};
      EXPECT_FALSE(should_save_iterate(current, relaxed_solution, invalid_tolerance, true));
      EXPECT_EQ(should_save_iterate(current, relaxed_solution, invalid_tolerance, false),
                member == &residuals_t::objective_gap);
    }
  }
}

}  // namespace cuopt::mathematical_optimization::barrier::test
