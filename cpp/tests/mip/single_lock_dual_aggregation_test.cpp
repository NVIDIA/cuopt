/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/inline_lp_test_utils.hpp>

#include <cuopt/mathematical_optimization/solve.hpp>
#include <mip_heuristics/presolve/single_lock_dual_aggregation.hpp>
#include <mip_heuristics/presolve/third_party_presolve.hpp>

#include <papilo/core/ProblemBuilder.hpp>
#include <papilo/core/ProblemUpdate.hpp>
#include <papilo/core/Reductions.hpp>
#include <papilo/core/Statistics.hpp>
#include <papilo/core/postsolve/PostsolveStorage.hpp>
#include <papilo/io/Message.hpp>
#include <papilo/misc/Timer.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace cuopt::mathematical_optimization::mip::test {

// Must match third_party_presolve.cpp; Papilo's own default is 1e-6.
constexpr double kProductionFeasTol = 1e-5;

struct papilo_harness_t {
  papilo::Num<double> num;
  papilo::Reductions<double> reductions;
  papilo::Statistics stats;
  papilo::PresolveOptions options;
  papilo::Message msg;
  double timer_acc{0};

  papilo_harness_t()
  {
    options.tlim = std::numeric_limits<double>::max();
    num.setFeasTol(kProductionFeasTol);
  }

  papilo::PresolveStatus run(papilo::Problem<double>& problem,
                             SingleLockDualAggregation<double>& presolver)
  {
    papilo::PostsolveStorage<double> postsolve(problem.getNRows(), problem.getNCols());
    papilo::ProblemUpdate<double> update(problem, postsolve, stats, options, num, msg);
    papilo::Timer timer(timer_acc);
    int reason = 0;
    return presolver.execute(problem, update, num, reductions, timer, reason);
  }

  bool has_replace(int col1, int col2, double factor, double offset) const
  {
    const auto& txns = reductions.getTransactions();
    const auto& reds = reductions.getReductions();
    for (const auto& t : txns) {
      const int payload_start = t.start + t.nlocks;
      if (t.end - payload_start != 2) continue;
      const auto& r0 = reds[payload_start];
      const auto& r1 = reds[payload_start + 1];
      if (r0.row == papilo::ColReduction::REPLACE && r0.col == col1 &&
          std::abs(r0.newval - factor) < 1e-9 && r1.col == col2 &&
          std::abs(r1.newval - offset) < 1e-9)
        return true;
    }
    return false;
  }

  bool replacement_locks_dependencies(int col1,
                                      int col2,
                                      int row,
                                      const std::unordered_set<int>& bound_cols) const
  {
    const auto& txns = reductions.getTransactions();
    const auto& reds = reductions.getReductions();
    for (const auto& t : txns) {
      const int payload_start = t.start + t.nlocks;
      if (t.end - payload_start != 2) continue;
      const auto& replacement = reds[payload_start];
      const auto& master      = reds[payload_start + 1];
      if (replacement.row != papilo::ColReduction::REPLACE || replacement.col != col1 ||
          master.col != col2)
        continue;

      bool col1_locked = false;
      bool col2_locked = false;
      bool row_locked  = false;
      std::unordered_set<int> locked_bound_cols;
      for (int i = t.start; i < payload_start; ++i) {
        const auto& lock = reds[i];
        if (lock.row == papilo::ColReduction::LOCKED) {
          col1_locked |= lock.col == col1;
          col2_locked |= lock.col == col2;
        } else if (lock.row == papilo::ColReduction::BOUNDS_LOCKED) {
          locked_bound_cols.insert(lock.col);
        } else if (lock.row == row && lock.col == papilo::RowReduction::LOCKED) {
          row_locked = true;
        }
      }
      return col1_locked && col2_locked && row_locked && locked_bound_cols == bound_cols;
    }
    return false;
  }
};

papilo::Problem<double> build_problem(int nrows,
                                      int ncols,
                                      const std::vector<std::tuple<int, int, double>>& entries,
                                      const std::vector<double>& obj,
                                      const std::vector<double>& lb,
                                      const std::vector<double>& ub,
                                      const std::vector<bool>& integer,
                                      const std::vector<double>& row_lhs,
                                      const std::vector<double>& row_rhs,
                                      const std::vector<bool>& lhs_inf,
                                      const std::vector<bool>& rhs_inf)
{
  papilo::ProblemBuilder<double> builder;
  builder.setNumRows(nrows);
  builder.setNumCols(ncols);
  for (auto& [r, c, v] : entries)
    builder.addEntry(r, c, v);
  for (int c = 0; c < ncols; ++c) {
    builder.setObj(c, obj[c]);
    builder.setColLb(c, lb[c]);
    builder.setColUb(c, ub[c]);
    builder.setColIntegral(c, integer[c]);
  }
  for (int r = 0; r < nrows; ++r) {
    builder.setRowLhs(r, row_lhs[r]);
    builder.setRowRhs(r, row_rhs[r]);
    builder.setRowLhsInf(r, lhs_inf[r]);
    builder.setRowRhsInf(r, rhs_inf[r]);
  }
  return builder.build();
}

// x has one up-lock in a LEQ row. Probe proves y=0 => x=0 via activity.
// Favorable-state check passes. Result: x = y (direct substitution).
//
//   min -x
//   s.t.  3x - 4y <= 1     (the locking row)
//         x + y   >= 0     (GEQ slack: positive coeffs add down-locks only)
//         x, y in {0,1}
//
// On the locking row:
// A_min=-4, A_max=3. Probe(x=1,y=0): probed_min = -4-0-(-4)+3 = 3 > 1 => proven.
// Favorable(y=1): A_max - max(0,-4) + (-4) = 3-0-4 = -1 <= 1 => safe.
TEST(SingleLockDualAggregation, DirectSubstitution)
{
  auto problem = build_problem(2,
                               2,
                               {{0, 0, 3.0}, {0, 1, -4.0}, {1, 0, 1.0}, {1, 1, 1.0}},
                               {-1.0, 0.0},
                               {0.0, 0.0},
                               {1.0, 1.0},
                               {true, true},
                               {0.0, 0.0},
                               {1.0, 0.0},
                               {true, false},
                               {false, true});

  SingleLockDualAggregation<double> presolver;
  papilo_harness_t h;
  auto status = h.run(problem, presolver);

  EXPECT_EQ(status, papilo::PresolveStatus::kReduced);
  EXPECT_TRUE(h.has_replace(0, 1, 1.0, 0.0));  // x = y
  EXPECT_TRUE(h.replacement_locks_dependencies(0, 1, 0, {0, 1}));
}

// Direct master has no negative binary coeff, so direct probe fails.
// Anti probe (y=1 unfavorable for upward) succeeds. Result: x = 1 - y.
//
//   min -x
//   s.t.  3x + 4y <= 5     (the locking row)
//         x + y   >= 0     (GEQ slack: positive coeffs add down-locks only)
//         x, y in {0,1}
//
// On the locking row:
// A_min=0, A_max=7. Direct: no neg binary master. Anti master: y (coeff +4).
// Probe(x=1,y=1): probed_min = 0-0-0+(3+4) = 7 > 5 => proven.
// Favorable(y=0 for anti-upward): A_max - max(0,4) + 0 = 7-4 = 3 <= 5 => safe.
TEST(SingleLockDualAggregation, AntiSubstitution)
{
  auto problem = build_problem(2,
                               2,
                               {{0, 0, 3.0}, {0, 1, 4.0}, {1, 0, 1.0}, {1, 1, 1.0}},
                               {-1.0, 0.0},
                               {0.0, 0.0},
                               {1.0, 1.0},
                               {true, true},
                               {0.0, 0.0},
                               {5.0, 0.0},
                               {true, false},
                               {false, true});

  SingleLockDualAggregation<double> presolver;
  papilo_harness_t h;
  auto status = h.run(problem, presolver);

  EXPECT_EQ(status, papilo::PresolveStatus::kReduced);
  EXPECT_TRUE(h.has_replace(0, 1, -1.0, 1.0));  // x = 1 - y
}

// A free row (both sides infinite) produces no locks and no candidates.
TEST(SingleLockDualAggregation, FreeRowNonCase)
{
  auto problem = build_problem(1,
                               2,
                               {{0, 0, 2.0}, {0, 1, 3.0}},
                               {-1.0, 0.0},
                               {0.0, 0.0},
                               {1.0, 1.0},
                               {true, true},
                               {0.0},
                               {0.0},
                               {true},
                               {true});

  SingleLockDualAggregation<double> presolver;
  papilo_harness_t h;
  auto status = h.run(problem, presolver);

  EXPECT_EQ(status, papilo::PresolveStatus::kUnchanged);
  EXPECT_EQ(h.reductions.size(), 0u);
}

// Probe proves the implication but the favorable-state check fails,
// so the substitution is correctly rejected.
//
//   min -x
//   s.t.  3x - 2y <= 0     (the locking row)
//         x + y   >= 0     (GEQ slack: positive coeffs add down-locks only)
//         x, y in {0,1}
//
// A_min=-2, A_max=3. Probe(x=1,y=0): probed_min = -2-0-(-2)+3 = 3 > 0 => proven.
// Favorable(y=1): A_max - max(0,-2) + (-2) = 3-0-2 = 1 > 0 => FAILS.
TEST(SingleLockDualAggregation, FavorableStateRejects)
{
  auto problem = build_problem(2,
                               2,
                               {{0, 0, 3.0}, {0, 1, -2.0}, {1, 0, 1.0}, {1, 1, 1.0}},
                               {-1.0, 0.0},
                               {0.0, 0.0},
                               {1.0, 1.0},
                               {true, true},
                               {0.0, 0.0},
                               {0.0, 0.0},
                               {true, false},
                               {false, true});

  SingleLockDualAggregation<double> presolver;
  papilo_harness_t h;
  auto status = h.run(problem, presolver);

  EXPECT_EQ(status, papilo::PresolveStatus::kUnchanged);
  EXPECT_EQ(h.reductions.size(), 0u);
}

// x has one down-lock in a GEQ row and a non-negative cost. The probe proves y=1 => x=1
// from the maximum activity.
//
//   min x
//   s.t.  3x - 4y >= -1    (the locking row)
//         x + y   <= 2     (LEQ slack: positive coeffs add up-locks only)
//         x, y in {0,1}
//
// A_min=-4, A_max=3. Probe(x=0,y=1): probed_max = 3-3-0+(-4) = -4 < -1 => proven.
// Favorable(y=0): A_min - min(0,-4) + 0 = -4+4 = 0 >= -1 => safe.
TEST(SingleLockDualAggregation, DownwardSubstitution)
{
  auto problem = build_problem(2,
                               2,
                               {{0, 0, 3.0}, {0, 1, -4.0}, {1, 0, 1.0}, {1, 1, 1.0}},
                               {1.0, 0.0},
                               {0.0, 0.0},
                               {1.0, 1.0},
                               {true, true},
                               {-1.0, 0.0},
                               {0.0, 2.0},
                               {false, true},
                               {true, false});

  SingleLockDualAggregation<double> presolver;
  papilo_harness_t h;
  auto status = h.run(problem, presolver);

  EXPECT_EQ(status, papilo::PresolveStatus::kReduced);
  EXPECT_TRUE(h.has_replace(0, 1, 1.0, 0.0));  // x = y
}

// A two-sided row runs both halves of the favorable-state check.
//
//   min -x
//   s.t.  -10 <= 3x - 4y <= 1    (the locking row)
//         x + y >= 0
//         x, y in {0,1}
//
// Exactly feasible: (0,0), (0,1), (1,1); optimum (1,1). Both favorable-state halves pass
// (upper: 3-0-4 = -1 <= 1, lower: -4+4-4 = -4 >= -10), and x=y keeps the optimum.
TEST(SingleLockDualAggregation, RangedRowSubstitution)
{
  auto problem = build_problem(2,
                               2,
                               {{0, 0, 3.0}, {0, 1, -4.0}, {1, 0, 1.0}, {1, 1, 1.0}},
                               {-1.0, 0.0},
                               {0.0, 0.0},
                               {1.0, 1.0},
                               {true, true},
                               {-10.0, 0.0},
                               {1.0, 0.0},
                               {false, false},
                               {false, true});

  SingleLockDualAggregation<double> presolver;
  papilo_harness_t h;
  auto status = h.run(problem, presolver);

  EXPECT_EQ(status, papilo::PresolveStatus::kReduced);
  EXPECT_TRUE(h.has_replace(0, 1, 1.0, 0.0));  // x = y
}

// Equality rows lock both directions, so a variable confined to them is never single-locked.
//
//   min -x
//   s.t.  3x - 4y = 1
//         x + y   = 1
//         x, y in {0,1}
TEST(SingleLockDualAggregation, EqualityRowsLockBothDirections)
{
  auto problem = build_problem(2,
                               2,
                               {{0, 0, 3.0}, {0, 1, -4.0}, {1, 0, 1.0}, {1, 1, 1.0}},
                               {-1.0, 0.0},
                               {0.0, 0.0},
                               {1.0, 1.0},
                               {true, true},
                               {1.0, 1.0},
                               {1.0, 1.0},
                               {false, false},
                               {false, false});

  SingleLockDualAggregation<double> presolver;
  papilo_harness_t h;
  auto status = h.run(problem, presolver);

  EXPECT_EQ(status, papilo::PresolveStatus::kUnchanged);
  EXPECT_EQ(h.reductions.size(), 0u);
}

// The favorable state violates the row by 5e-6, under the 1e-5 feasibility tolerance.
// Exactly, x must be 0, so x=y would leave the problem with no exact optimum.
//
//   min -x
//   s.t.  2x - 1.000001y <= 0.999994    (the locking row)
//         x + y >= 0
//         x, y in {0,1}
//
// Probe(x=1,y=0): probed_min = 2 > 0.999994 => proven.
// Favorable(y=1): A_max - max(0,-1.000001) + (-1.000001) = 0.999999 > 0.999994 => reject.
TEST(SingleLockDualAggregation, FavorableStateRejectsSubToleranceViolation)
{
  auto problem = build_problem(2,
                               2,
                               {{0, 0, 2.0}, {0, 1, -1.000001}, {1, 0, 1.0}, {1, 1, 1.0}},
                               {-1.0, 0.0},
                               {0.0, 0.0},
                               {1.0, 1.0},
                               {true, true},
                               {0.0, 0.0},
                               {0.999994, 0.0},
                               {true, false},
                               {false, true});

  SingleLockDualAggregation<double> presolver;
  papilo_harness_t h;
  auto status = h.run(problem, presolver);

  EXPECT_EQ(status, papilo::PresolveStatus::kUnchanged);
  EXPECT_EQ(h.reductions.size(), 0u);
}

namespace {

papilo::Problem<double> build_direct_substitution_problem(double candidate_objective)
{
  return build_problem(2,
                       2,
                       {{0, 0, 3.0}, {0, 1, -4.0}, {1, 0, 1.0}, {1, 1, 1.0}},
                       {candidate_objective, 0.0},
                       {0.0, 0.0},
                       {1.0, 1.0},
                       {true, true},
                       {0.0, 0.0},
                       {1.0, 0.0},
                       {true, false},
                       {false, true});
}

}  // namespace

TEST(SingleLockDualAggregation, DualredsZeroDisables)
{
  auto problem = build_direct_substitution_problem(-1.0);

  SingleLockDualAggregation<double> presolver;
  papilo_harness_t h;
  h.options.dualreds = 0;
  auto status        = h.run(problem, presolver);

  EXPECT_EQ(status, papilo::PresolveStatus::kUnchanged);
  EXPECT_EQ(h.reductions.size(), 0u);
}

// A zero objective coefficient makes x=y discard alternative optima, which needs dualreds 2.
TEST(SingleLockDualAggregation, ZeroObjectiveNeedsFullDualreds)
{
  {
    auto problem = build_direct_substitution_problem(0.0);
    SingleLockDualAggregation<double> presolver;
    papilo_harness_t h;
    h.options.dualreds = 1;
    EXPECT_EQ(h.run(problem, presolver), papilo::PresolveStatus::kUnchanged);
    EXPECT_EQ(h.reductions.size(), 0u);
  }
  {
    auto problem = build_direct_substitution_problem(0.0);
    SingleLockDualAggregation<double> presolver;
    papilo_harness_t h;
    h.options.dualreds = 2;
    EXPECT_EQ(h.run(problem, presolver), papilo::PresolveStatus::kReduced);
    EXPECT_TRUE(h.has_replace(0, 1, 1.0, 0.0));
  }
}

TEST(SingleLockDualAggregation, PreventsSubstitutionChains)
{
  auto problem = build_problem(2,
                               3,
                               {{0, 0, 3.0}, {0, 1, -3.0}, {1, 0, -4.0}, {1, 1, 4.0}, {1, 2, -5.0}},
                               {-1.0, -1.0, 0.0},
                               {0.0, 0.0, 0.0},
                               {1.0, 1.0, 1.0},
                               {true, true, true},
                               {0.0, 0.0},
                               {1.0, -1.0},
                               {true, true},
                               {false, false});

  SingleLockDualAggregation<double> presolver;
  papilo_harness_t h;
  auto status = h.run(problem, presolver);

  EXPECT_EQ(status, papilo::PresolveStatus::kReduced);
  EXPECT_TRUE(h.has_replace(0, 1, 1.0, 0.0));
  EXPECT_FALSE(h.has_replace(1, 2, 1.0, 0.0));
}

third_party_presolve_device_result_t<int, double> static run_presolve(
  std::string_view lp_text, std::unordered_set<std::string> allowlist)
{
  const raft::handle_t handle{};
  auto mps_data_model = cuopt::test::parse_inline_lp(lp_text);
  auto op_problem     = mps_data_model_to_optimization_problem(&handle, mps_data_model);
  auto presolver      = std::make_unique<third_party_presolve_t<int, double>>();
  presolver->set_reduction_allowlist(std::move(allowlist));
  return presolver->apply_presolve_from_op_problem(
    op_problem, problem_category_t::MIP, presolver_t::Papilo, false, 1e-6, 1e-12, 20, 1);
}

TEST(SingleLockDualAggregation, AppliesOnlyDisjointSubstitutions)
{
  auto result = run_presolve(R"LP(
Minimize
  obj: - x - y
Subject To
  r0: 3 x - 3 y <= 1
  r1: -4 x + 4 y - 5 z <= -1
Binaries
  x
  y
  z
End
)LP",
                             {"single_lock_dual_aggregation"});

  EXPECT_TRUE(result.status == third_party_presolve_status_t::REDUCED ||
              result.status == third_party_presolve_status_t::OPTIMAL);
}

}  // namespace cuopt::mathematical_optimization::mip::test
