/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/presolve/single_lock_dual_aggregation.hpp>

#include <papilo/core/ProblemBuilder.hpp>
#include <papilo/core/ProblemUpdate.hpp>
#include <papilo/core/Reductions.hpp>
#include <papilo/core/Statistics.hpp>
#include <papilo/core/postsolve/PostsolveStorage.hpp>
#include <papilo/io/Message.hpp>
#include <papilo/misc/Timer.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace cuopt::mathematical_optimization::mip::test {

struct papilo_harness_t {
  papilo::Num<double> num;
  papilo::Reductions<double> reductions;
  papilo::Statistics stats;
  papilo::PresolveOptions options;
  papilo::Message msg;
  double timer_acc{0};

  papilo_harness_t() { options.tlim = std::numeric_limits<double>::max(); }

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
      if (t.end - t.start != 2) continue;
      const auto& r0 = reds[t.start];
      const auto& r1 = reds[t.start + 1];
      if (r0.row == papilo::ColReduction::REPLACE && r0.col == col1 &&
          std::abs(r0.newval - factor) < 1e-9 && r1.col == col2 &&
          std::abs(r1.newval - offset) < 1e-9)
        return true;
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
//   s.t.  3x - 2y <= 0
//         x, y in {0,1}
//
// A_min=-2, A_max=3. Probe(x=1,y=0): probed_min = -2-0-(-2)+3 = 3 > 0 => proven.
// Favorable(y=1): A_max - max(0,-2) + (-2) = 3-0-2 = 1 > 0 => FAILS.
TEST(SingleLockDualAggregation, FavorableStateRejects)
{
  auto problem = build_problem(1,
                               2,
                               {{0, 0, 3.0}, {0, 1, -2.0}},
                               {-1.0, 0.0},
                               {0.0, 0.0},
                               {1.0, 1.0},
                               {true, true},
                               {0.0},
                               {0.0},
                               {true},
                               {false});

  SingleLockDualAggregation<double> presolver;
  papilo_harness_t h;
  auto status = h.run(problem, presolver);

  EXPECT_EQ(status, papilo::PresolveStatus::kUnchanged);
  EXPECT_EQ(h.reductions.size(), 0u);
}

}  // namespace cuopt::mathematical_optimization::mip::test
