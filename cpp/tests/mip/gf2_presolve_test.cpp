/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/presolve/gf2_presolve.hpp>

#include <papilo/core/PresolveOptions.hpp>
#include <papilo/core/ProblemBuilder.hpp>
#include <papilo/core/ProblemUpdate.hpp>
#include <papilo/core/Reductions.hpp>
#include <papilo/core/Statistics.hpp>
#include <papilo/core/postsolve/PostsolveStorage.hpp>
#include <papilo/io/Message.hpp>
#include <papilo/misc/Num.hpp>
#include <papilo/misc/Timer.hpp>

#include <gtest/gtest.h>

#include <tuple>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::test {

TEST(gf2_presolve, uses_compact_constraint_indices)
{
  constexpr int num_non_gf2_constraints = 128;
  constexpr int num_rows                = num_non_gf2_constraints + 2;

  std::vector<std::tuple<int, int, double>> entries;
  entries.reserve(2 * num_non_gf2_constraints + 5);

  // These duplicate fractional equalities are deliberately not GF2 constraints. The last two
  // rows mirror the x + 2y = k structure in the enlight instances, so their raw row indices (128
  // and 129) differ from their compact GF2 ordinals (0 and 1).
  for (int row = 0; row < num_non_gf2_constraints; ++row) {
    entries.emplace_back(row, 0, 1.0);
    entries.emplace_back(row, 1, 1.0);
  }
  entries.emplace_back(num_non_gf2_constraints, 2, 1.0);
  entries.emplace_back(num_non_gf2_constraints, 3, 1.0);
  entries.emplace_back(num_non_gf2_constraints, 4, 2.0);
  entries.emplace_back(num_non_gf2_constraints + 1, 3, 1.0);
  entries.emplace_back(num_non_gf2_constraints + 1, 5, 2.0);

  std::vector<double> row_bounds(num_non_gf2_constraints, 0.5);
  row_bounds.insert(row_bounds.end(), {1.0, 0.0});

  papilo::ProblemBuilder<double> builder;
  builder.reserve(entries.size(), num_rows, 6);
  builder.setNumRows(num_rows);
  builder.setNumCols(6);
  builder.setObjAll({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  builder.setColLbAll({-1.0, -1.0, 0.0, 0.0, -1.0, -1.0});
  builder.setColUbAll({1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  builder.setColIntegralAll({0, 0, 1, 1, 1, 1});
  builder.setRowLhsAll(row_bounds);
  builder.setRowRhsAll(std::move(row_bounds));
  builder.addEntryAll(std::move(entries));
  auto problem = builder.build();

  papilo::Num<double> num;
  papilo::PresolveOptions options;
  papilo::Statistics statistics;
  papilo::Message message;
  papilo::PostsolveStorage<double> postsolve(problem, num, options);
  papilo::ProblemUpdate<double> problem_update(
    problem, postsolve, statistics, options, num, message);
  papilo::Reductions<double> reductions;
  double elapsed_time = 0.0;
  papilo::Timer timer(elapsed_time);
  int reason_of_infeasibility = 0;

  mip::GF2Presolve<double> presolver;
  auto status =
    presolver.execute(problem, problem_update, num, reductions, timer, reason_of_infeasibility);

  EXPECT_EQ(status, papilo::PresolveStatus::kReduced);
}

}  // namespace cuopt::mathematical_optimization::test
