/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/presolve/bhw_coeff_reduce.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace cuopt::mathematical_optimization::test {

using mip::BHW_MAX_LEN;
using mip::bhw_reduce_row;
using mip::bhw_row_rewrite_t;
using mip::bhw_shape_cache_t;

namespace {

bhw_row_rewrite_t reduce(const std::vector<double>& coefficients,
                         double side,
                         int direction            = 1,
                         bhw_shape_cache_t* cache = nullptr)
{
  return bhw_reduce_row<double>(
    coefficients.data(), (int)coefficients.size(), side, direction, cache);
}

// The whole point of the pass: the rewritten row must accept exactly the same 0/1 points as the
// original. Checked over every point, independently of the extremal-point test the search uses.
bool same_feasible_set(const std::vector<double>& coefficients,
                       double side,
                       int direction,
                       const bhw_row_rewrite_t& rewrite)
{
  constexpr double tol = 1e-9;
  const int k          = (int)coefficients.size();
  for (uint32_t mask = 0; mask < (1u << k); ++mask) {
    double original   = 0.0;
    int64_t rewritten = 0;
    for (int i = 0; i < k; ++i) {
      if ((mask >> i & 1u) == 0u) continue;
      original += coefficients[i];
      rewritten += rewrite.coefficients[i];
    }
    const bool original_ok = direction == 1 ? original <= side + tol : original >= side - tol;
    const bool rewritten_ok =
      direction == 1 ? rewritten <= rewrite.side : rewritten >= rewrite.side;
    if (original_ok != rewritten_ok) return false;
  }
  return true;
}

}  // namespace

// Bradley, Hammer and Wolsey (1974) open with this row and reduce it to 4,4,2,2,1,1,1,0 <= 5. That
// rewrite enlarges the LP relaxation (it admits a fractional point of activity 91.75 against a
// right-hand side of 80), so the LP-strength check rejects it and no smaller equivalent form
// survives.
TEST(bhw_coeff_reduce, rejects_a_rewrite_that_weakens_the_relaxation)
{
  EXPECT_FALSE(reduce({65, 64, 41, 22, 13, 12, 8, 2}, 80).accepted);
}

// The shape that motivated the pass: a 9/10 coefficient against thirds, with a fractional
// right-hand side. Integerizing and reducing lands it in int8 range.
TEST(bhw_coeff_reduce, rational_row_integerizes_and_reduces)
{
  const std::vector<double> row{0.9, 1.0 / 3, 1.0 / 3, 1.0 / 3};
  const auto reduced = reduce(row, 2.0 / 3);
  ASSERT_TRUE(reduced.accepted);
  EXPECT_EQ(reduced.coefficients, std::vector<int64_t>({3, 1, 1, 1}));
  EXPECT_EQ(reduced.side, 2);
  EXPECT_TRUE(same_feasible_set(row, 2.0 / 3, 1, reduced));
}

// The >= orientation is normalized by negation, so the same row negated must come back negated.
TEST(bhw_coeff_reduce, greater_equal_row_keeps_its_orientation)
{
  const std::vector<double> row{-0.9, -1.0 / 3, -1.0 / 3, -1.0 / 3};
  const auto reduced = reduce(row, -2.0 / 3, -1);
  ASSERT_TRUE(reduced.accepted);
  EXPECT_EQ(reduced.coefficients, std::vector<int64_t>({-3, -1, -1, -1}));
  EXPECT_EQ(reduced.side, -2);
  EXPECT_TRUE(same_feasible_set(row, -2.0 / 3, -1, reduced));
}

TEST(bhw_coeff_reduce, rejects_rows_with_nothing_to_give_back)
{
  // Already at unit magnitude.
  EXPECT_FALSE(reduce({1, 1, 1, 1}, 2).accepted);
  // Does not integerize within the rational cap.
  EXPECT_FALSE(reduce({M_PI, 1, 1}, 2).accepted);
  // Outside the enumerable width.
  EXPECT_FALSE(reduce({5}, 2).accepted);
  EXPECT_FALSE(reduce(std::vector<double>(BHW_MAX_LEN + 1, 3.0), 5).accepted);
  // Every point feasible, so there is nothing to separate.
  EXPECT_FALSE(reduce({3, 2, 2}, 100).accepted);
  // No point feasible.
  EXPECT_FALSE(reduce({3, 2, 2}, -1).accepted);
}

TEST(bhw_coeff_reduce, rejects_rows_with_a_zero_coefficient)
{
  EXPECT_FALSE(reduce({65, 64, 41, 22, 13, 12, 8, 2, 0}, 80).accepted);
  EXPECT_FALSE(reduce({0, 9, 7, 6, 6, 4}, 20).accepted);
  EXPECT_FALSE(reduce({6, 0}, 5).accepted);
}

TEST(bhw_coeff_reduce, memoized_result_matches_the_uncached_one)
{
  bhw_shape_cache_t cache;
  const std::vector<std::vector<double>> rows{
    {0.9, 1.0 / 3, 1.0 / 3, 1.0 / 3}, {6, 4, 3, 2}, {-6, 4, 3, -2}, {9, 7, 6, 6, 4}};
  for (const auto& row : rows) {
    for (int repeat = 0; repeat < 2; ++repeat) {
      const auto cached   = reduce(row, 12, 1, &cache);
      const auto uncached = reduce(row, 12, 1, nullptr);
      EXPECT_EQ(cached.accepted, uncached.accepted);
      EXPECT_EQ(cached.coefficients, uncached.coefficients);
      EXPECT_EQ(cached.side, uncached.side);
    }
  }
}

// The invariant that matters, over mixed signs, both orientations and rational coefficients: an
// accepted rewrite never changes which 0/1 points satisfy the row.
TEST(bhw_coeff_reduce, accepted_rewrites_preserve_the_feasible_set)
{
  std::mt19937_64 rng(20260805);
  bhw_shape_cache_t cache;
  const int denominators[] = {1, 2, 3, 4, 5, 6, 8, 10, 12, 16};
  int accepted             = 0;

  for (int trial = 0; trial < 20000; ++trial) {
    const int len         = 2 + (int)(rng() % 7);
    const int direction   = (rng() & 1u) != 0u ? 1 : -1;
    const int denominator = denominators[rng() % 10];

    std::vector<double> row(len);
    double positive_sum = 0.0;
    double negative_sum = 0.0;
    for (int i = 0; i < len; ++i) {
      const int64_t numerator = 1 + (int64_t)(rng() % 30);
      row[i]                  = (double)numerator / denominator * ((rng() & 3u) == 0u ? -1.0 : 1.0);
      if (row[i] > 0.0)
        positive_sum += row[i];
      else
        negative_sum += row[i];
    }
    // Put the side inside the activity range so the row is not trivially satisfied or violated.
    double side = negative_sum + (positive_sum - negative_sum) * (double)(rng() % 1001) / 1000.0;
    side        = std::round(side * denominator) / denominator;
    if (direction == -1) {
      for (double& value : row)
        value = -value;
      side = -side;
    }

    const auto reduced = reduce(row, side, direction, &cache);
    if (!reduced.accepted) continue;
    ++accepted;

    ASSERT_EQ((int)reduced.coefficients.size(), len);
    ASSERT_TRUE(same_feasible_set(row, side, direction, reduced))
      << "rewrite changed the 0/1 feasible set on trial " << trial;
  }
  // Guards against the generator drifting into a corner where nothing is ever reduced.
  EXPECT_GT(accepted, 1000);
}

}  // namespace cuopt::mathematical_optimization::test
