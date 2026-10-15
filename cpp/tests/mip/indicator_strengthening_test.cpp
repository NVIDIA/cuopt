/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/presolve/indicator_strengthening.hpp>

#include <papilo/core/ProblemBuilder.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip::test {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kTol = 1e-9;

using entries_t = std::vector<std::pair<int, double>>;

struct row_t {
  entries_t entries;
  double lhs;
  double rhs;
};

// Every column has a lower bound of 0. A row side of +-kInf is left unbounded.
papilo::Problem<double> build_problem(const std::vector<int>& upper_bounds,
                                      const std::vector<bool>& integral,
                                      const std::vector<row_t>& rows)
{
  const int num_cols = upper_bounds.size();
  const int num_rows = rows.size();

  papilo::ProblemBuilder<double> builder;
  builder.setNumCols(num_cols);
  builder.setNumRows(num_rows);
  for (int col = 0; col < num_cols; ++col) {
    builder.setObj(col, 0.0);
    builder.setColLb(col, 0.0);
    builder.setColUb(col, upper_bounds[col]);
    builder.setColIntegral(col, integral[col]);
  }
  for (int row = 0; row < num_rows; ++row) {
    for (const auto& [col, value] : rows[row].entries) {
      builder.addEntry(row, col, value);
    }
    builder.setRowLhsInf(row, rows[row].lhs == -kInf);
    builder.setRowRhsInf(row, rows[row].rhs == kInf);
    if (rows[row].lhs != -kInf) { builder.setRowLhs(row, rows[row].lhs); }
    if (rows[row].rhs != kInf) { builder.setRowRhs(row, rows[row].rhs); }
  }
  return builder.build();
}

// The stored (column, value) pairs of a row, in storage order.
entries_t row_entries(const papilo::Problem<double>& problem, int row)
{
  const auto coefficients = problem.getConstraintMatrix().getRowCoefficients(row);
  const int len           = coefficients.getLength();
  entries_t entries;
  entries.reserve(len);
  for (int p = 0; p < len; ++p) {
    entries.emplace_back(coefficients.getIndices()[p], coefficients.getValues()[p]);
  }
  return entries;
}

double row_activity(const papilo::Problem<double>& problem,
                    int row,
                    const std::vector<double>& point)
{
  double activity = 0.0;
  for (const auto& [col, value] : row_entries(problem, row)) {
    activity += value * point[col];
  }
  return activity;
}

bool is_feasible(const papilo::Problem<double>& problem, const std::vector<double>& point)
{
  const auto& matrix = problem.getConstraintMatrix();
  const auto& flags  = matrix.getRowFlags();
  for (int row = 0; row < matrix.getNRows(); ++row) {
    const double activity = row_activity(problem, row, point);
    if (!flags[row].test(papilo::RowFlag::kLhsInf) &&
        activity < matrix.getLeftHandSides()[row] - kTol) {
      return false;
    }
    if (!flags[row].test(papilo::RowFlag::kRhsInf) &&
        activity > matrix.getRightHandSides()[row] + kTol) {
      return false;
    }
  }
  return true;
}

// Every integer point in the box [0, upper_bounds]. The continuous columns in these tests only meet
// integral row sides, so the integer grid reaches every vertex of their feasible intervals.
std::vector<std::vector<double>> integer_points(const std::vector<int>& upper_bounds)
{
  const int num_cols = upper_bounds.size();
  size_t num_points  = 1;
  for (int upper_bound : upper_bounds) {
    num_points *= upper_bound + 1;
  }
  std::vector<std::vector<double>> points(num_points, std::vector<double>(num_cols));
  for (size_t k = 0; k < num_points; ++k) {
    size_t rest = k;
    for (int col = 0; col < num_cols; ++col) {
      points[k][col] = rest % (upper_bounds[col] + 1);
      rest /= upper_bounds[col] + 1;
    }
  }
  return points;
}

}  // namespace

// Reduction 1 alone. Two groups, z0 owning {x0, x1} and z1 owning {x2, x3}.
//
//   y0 <= x0 + x1 + x2         members span {z0, z1}  =>  y0 <= z0 + z1
//   x2 + x3 - y1 >= 0          members span {z1}      =>  y1 <= z1 (stored as >=)
//   y2 <= x0 + x2              one indicator per member, nothing to aggregate
TEST(IndicatorStrengthening, ImpliedIndicatorRows)
{
  constexpr int z0 = 0;
  constexpr int z1 = 1;
  constexpr int x0 = 2;
  constexpr int x1 = 3;
  constexpr int x2 = 4;
  constexpr int x3 = 5;
  constexpr int y0 = 6;
  constexpr int y1 = 7;
  constexpr int y2 = 8;

  const std::vector<int> upper_bounds(9, 1);
  const std::vector<bool> integral(9, true);
  const std::vector<row_t> rows{
    {{{x0, 1.0}, {z0, -1.0}}, -kInf, 0.0},
    {{{x1, 1.0}, {z0, -1.0}}, -kInf, 0.0},
    {{{x2, 1.0}, {z1, -1.0}}, -kInf, 0.0},
    {{{x3, 1.0}, {z1, -1.0}}, -kInf, 0.0},
    {{{y0, 1.0}, {x0, -1.0}, {x1, -1.0}, {x2, -1.0}}, -kInf, 0.0},
    {{{x2, 1.0}, {x3, 1.0}, {y1, -1.0}}, 0.0, kInf},
    {{{y2, 1.0}, {x0, -1.0}, {x2, -1.0}}, -kInf, 0.0},
  };
  const int num_rows = rows.size();

  const auto original = build_problem(upper_bounds, integral, rows);
  auto strengthened   = build_problem(upper_bounds, integral, rows);
  strengthen_indicators<int, double>(strengthened);

  const auto& matrix = strengthened.getConstraintMatrix();
  ASSERT_EQ(matrix.getNRows(), num_rows + 2);
  for (int row = 0; row < num_rows; ++row) {
    EXPECT_EQ(row_entries(strengthened, row), row_entries(original, row)) << "row " << row;
  }

  const int implied_y0 = num_rows;
  const int implied_y1 = num_rows + 1;
  EXPECT_EQ(row_entries(strengthened, implied_y0), (entries_t{{z0, -1.0}, {z1, -1.0}, {y0, 1.0}}));
  EXPECT_EQ(row_entries(strengthened, implied_y1), (entries_t{{z1, -1.0}, {y1, 1.0}}));
  for (int row : {implied_y0, implied_y1}) {
    EXPECT_TRUE(matrix.getRowFlags()[row].test(papilo::RowFlag::kLhsInf)) << "row " << row;
    EXPECT_FALSE(matrix.getRowFlags()[row].test(papilo::RowFlag::kRhsInf)) << "row " << row;
    EXPECT_EQ(matrix.getRightHandSides()[row], 0.0) << "row " << row;
  }

  for (const auto& point : integer_points(upper_bounds)) {
    EXPECT_EQ(is_feasible(strengthened, point), is_feasible(original, point));
  }

  // LP point: y0 = 1 is covered by x0 + x1 + x2 = 1, but z0 + z1 = 2/3.
  std::vector<double> fractional(9, 0.0);
  fractional[z0] = fractional[z1] = 1.0 / 3.0;
  fractional[x0] = fractional[x1] = fractional[x2] = 1.0 / 3.0;
  fractional[y0]                                   = 1.0;
  EXPECT_TRUE(is_feasible(original, fractional));
  EXPECT_GT(row_activity(strengthened, implied_y0, fractional), kTol);
}

// Reduction 2 alone. z0 owns {x0, x1, x2}, z1 owns {x3}, s is a continuous slack in [0, 2].
//
//   x0 + x1 + x2 - s <= 1      =>  x0 + x1 + x2 - s <= z0
//   -x0 - x1 - x2 >= -2        =>  -x0 - x1 - x2 + 2 z0 >= 0
//   x0 + x3 <= 1               members share no indicator, unchanged
//   x0 + x1 <= 2               capacity is not binding, unchanged
TEST(IndicatorStrengthening, LiftedCapacityRows)
{
  constexpr int z0 = 0;
  constexpr int z1 = 1;
  constexpr int x0 = 2;
  constexpr int x1 = 3;
  constexpr int x2 = 4;
  constexpr int x3 = 5;
  constexpr int s  = 6;

  const std::vector<int> upper_bounds{1, 1, 1, 1, 1, 1, 2};
  const std::vector<bool> integral{true, true, true, true, true, true, false};
  const std::vector<row_t> rows{
    {{{x0, 1.0}, {z0, -1.0}}, -kInf, 0.0},
    {{{x1, 1.0}, {z0, -1.0}}, -kInf, 0.0},
    {{{x2, 1.0}, {z0, -1.0}}, -kInf, 0.0},
    {{{x3, 1.0}, {z1, -1.0}}, -kInf, 0.0},
    {{{x0, 1.0}, {x1, 1.0}, {x2, 1.0}, {s, -1.0}}, -kInf, 1.0},
    {{{x0, -1.0}, {x1, -1.0}, {x2, -1.0}}, -2.0, kInf},
    {{{x0, 1.0}, {x3, 1.0}}, -kInf, 1.0},
    {{{x0, 1.0}, {x1, 1.0}}, -kInf, 2.0},
  };
  const int num_rows  = rows.size();
  const int lifted_le = 4;
  const int lifted_ge = 5;

  const auto original = build_problem(upper_bounds, integral, rows);
  auto strengthened   = build_problem(upper_bounds, integral, rows);
  strengthen_indicators<int, double>(strengthened);

  const auto& matrix = strengthened.getConstraintMatrix();
  ASSERT_EQ(matrix.getNRows(), num_rows);
  for (int row = 0; row < num_rows; ++row) {
    if (row == lifted_le || row == lifted_ge) { continue; }
    EXPECT_EQ(row_entries(strengthened, row), row_entries(original, row)) << "row " << row;
    EXPECT_EQ(matrix.getRightHandSides()[row],
              original.getConstraintMatrix().getRightHandSides()[row])
      << "row " << row;
  }

  EXPECT_EQ(row_entries(strengthened, lifted_le),
            (entries_t{{z0, -1.0}, {x0, 1.0}, {x1, 1.0}, {x2, 1.0}, {s, -1.0}}));
  EXPECT_TRUE(matrix.getRowFlags()[lifted_le].test(papilo::RowFlag::kLhsInf));
  EXPECT_EQ(matrix.getRightHandSides()[lifted_le], 0.0);

  EXPECT_EQ(row_entries(strengthened, lifted_ge),
            (entries_t{{z0, 2.0}, {x0, -1.0}, {x1, -1.0}, {x2, -1.0}}));
  EXPECT_TRUE(matrix.getRowFlags()[lifted_ge].test(papilo::RowFlag::kRhsInf));
  EXPECT_EQ(matrix.getLeftHandSides()[lifted_ge], 0.0);

  for (const auto& point : integer_points(upper_bounds)) {
    EXPECT_EQ(is_feasible(strengthened, point), is_feasible(original, point));
  }

  // LP point: x0 + x1 + x2 = 1 fits the capacity, but z0 = 1/3 does not pay for it.
  std::vector<double> fractional(7, 0.0);
  fractional[z0] = 1.0 / 3.0;
  fractional[x0] = fractional[x1] = fractional[x2] = 1.0 / 3.0;
  EXPECT_TRUE(is_feasible(original, fractional));
  EXPECT_GT(row_activity(strengthened, lifted_le, fractional), kTol);
  EXPECT_LT(row_activity(strengthened, lifted_ge, fractional), -kTol);
}

// Both reductions on one set-cover-like model: z0 owns {x0, x1}, z1 owns {x2, x3}, y must be
// covered by one of the members, and each group has a capacity row.
//
//   y <= x0 + x1 + x2 + x3     =>  appended y <= z0 + z1
//   x0 + x1 - s <= 1           =>  x0 + x1 - s <= z0
//   x2 + x3 <= 1               =>  x2 + x3 <= z1
TEST(IndicatorStrengthening, ImpliedAndLiftedRows)
{
  constexpr int z0 = 0;
  constexpr int z1 = 1;
  constexpr int x0 = 2;
  constexpr int x1 = 3;
  constexpr int x2 = 4;
  constexpr int x3 = 5;
  constexpr int y  = 6;
  constexpr int s  = 7;

  const std::vector<int> upper_bounds{1, 1, 1, 1, 1, 1, 1, 2};
  const std::vector<bool> integral{true, true, true, true, true, true, true, false};
  const std::vector<row_t> rows{
    {{{x0, 1.0}, {z0, -1.0}}, -kInf, 0.0},
    {{{x1, 1.0}, {z0, -1.0}}, -kInf, 0.0},
    {{{x2, 1.0}, {z1, -1.0}}, -kInf, 0.0},
    {{{x3, 1.0}, {z1, -1.0}}, -kInf, 0.0},
    {{{y, 1.0}, {x0, -1.0}, {x1, -1.0}, {x2, -1.0}, {x3, -1.0}}, -kInf, 0.0},
    {{{x0, 1.0}, {x1, 1.0}, {s, -1.0}}, -kInf, 1.0},
    {{{x2, 1.0}, {x3, 1.0}}, -kInf, 1.0},
  };
  const int num_rows  = rows.size();
  const int lifted_z0 = 5;
  const int lifted_z1 = 6;
  const int implied_y = num_rows;

  const auto original = build_problem(upper_bounds, integral, rows);
  auto strengthened   = build_problem(upper_bounds, integral, rows);
  strengthen_indicators<int, double>(strengthened);

  const auto& matrix = strengthened.getConstraintMatrix();
  ASSERT_EQ(matrix.getNRows(), num_rows + 1);
  for (int row = 0; row < lifted_z0; ++row) {
    EXPECT_EQ(row_entries(strengthened, row), row_entries(original, row)) << "row " << row;
  }

  EXPECT_EQ(row_entries(strengthened, lifted_z0),
            (entries_t{{z0, -1.0}, {x0, 1.0}, {x1, 1.0}, {s, -1.0}}));
  EXPECT_EQ(row_entries(strengthened, lifted_z1), (entries_t{{z1, -1.0}, {x2, 1.0}, {x3, 1.0}}));
  EXPECT_EQ(row_entries(strengthened, implied_y), (entries_t{{z0, -1.0}, {z1, -1.0}, {y, 1.0}}));
  for (int row : {lifted_z0, lifted_z1, implied_y}) {
    EXPECT_TRUE(matrix.getRowFlags()[row].test(papilo::RowFlag::kLhsInf)) << "row " << row;
    EXPECT_FALSE(matrix.getRowFlags()[row].test(papilo::RowFlag::kRhsInf)) << "row " << row;
    EXPECT_EQ(matrix.getRightHandSides()[row], 0.0) << "row " << row;
  }

  for (const auto& point : integer_points(upper_bounds)) {
    EXPECT_EQ(is_feasible(strengthened, point), is_feasible(original, point));
  }

  // LP point: every original row holds, and each of the three new rows is violated by 1/3.
  std::vector<double> fractional(8, 0.0);
  fractional[z0] = fractional[z1] = 1.0 / 3.0;
  fractional[x0] = fractional[x1] = fractional[x2] = fractional[x3] = 1.0 / 3.0;
  fractional[y]                                                     = 1.0;
  EXPECT_TRUE(is_feasible(original, fractional));
  for (int row : {lifted_z0, lifted_z1, implied_y}) {
    EXPECT_GT(row_activity(strengthened, row, fractional), kTol) << "row " << row;
  }
}

}  // namespace cuopt::mathematical_optimization::mip::test
