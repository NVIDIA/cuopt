/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuts/cuts.hpp>
#include <presolve/probing_cache.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::mip::test {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
}  // namespace

// x_0 is an integer in (-inf, 1] when probed, so the cache holds {x_0 = 1} first and
// {x_0 <= 0} second. Presolve then tightens x_0 to [0, 1], making it a binary in B&B.
TEST(ProbingImpliedBound, UpperBoundedIntegerTightenedToBinary)
{
  simplex::user_problem_t<int, double> problem(nullptr);
  problem.num_cols = 2;
  problem.lower    = {0.0, 0.0};
  problem.upper    = {1.0, 10.0};

  probing_cache_t<int, double> cache;
  auto& entries                         = cache.probing_cache[0];
  entries[0].val_interval               = {1.0, interval_type_t::EQUALS};
  entries[0].var_to_cached_bound_map[1] = {5.0, 10.0};
  entries[1].val_interval               = {0.0, interval_type_t::LEQ};
  entries[1].var_to_cached_bound_map[1] = {0.0, 2.0};

  probing_implied_bound_t<int, double> implied;
  extract_probing_implied_bounds<int, double>({}, problem, cache, implied);

  ASSERT_EQ(implied.zero_offsets, (std::vector<int>{0, 1, 1}));
  EXPECT_EQ(implied.zero_variables, (std::vector<int>{1}));
  EXPECT_EQ(implied.zero_lower_bound, (std::vector<double>{0.0}));
  EXPECT_EQ(implied.zero_upper_bound, (std::vector<double>{2.0}));

  ASSERT_EQ(implied.one_offsets, (std::vector<int>{0, 1, 1}));
  EXPECT_EQ(implied.one_variables, (std::vector<int>{1}));
  EXPECT_EQ(implied.one_lower_bound, (std::vector<double>{5.0}));
  EXPECT_EQ(implied.one_upper_bound, (std::vector<double>{10.0}));
}

// Only the probe x_0 = 1 changed any bound, so it is the only entry and sits in the first slot.
TEST(ProbingImpliedBound, SingleCachedProbeOnOne)
{
  simplex::user_problem_t<int, double> problem(nullptr);
  problem.num_cols = 2;
  problem.lower    = {0.0, 0.0};
  problem.upper    = {1.0, 10.0};

  probing_cache_t<int, double> cache;
  auto& entries                         = cache.probing_cache[0];
  entries[0].val_interval               = {1.0, interval_type_t::EQUALS};
  entries[0].var_to_cached_bound_map[1] = {5.0, 10.0};

  probing_implied_bound_t<int, double> implied;
  extract_probing_implied_bounds<int, double>({}, problem, cache, implied);

  EXPECT_EQ(implied.zero_offsets, (std::vector<int>{0, 0, 0}));
  ASSERT_EQ(implied.one_offsets, (std::vector<int>{0, 1, 1}));
  EXPECT_EQ(implied.one_variables, (std::vector<int>{1}));
  EXPECT_EQ(implied.one_lower_bound, (std::vector<double>{5.0}));
  EXPECT_EQ(implied.one_upper_bound, (std::vector<double>{10.0}));
}

// Probes covering both values of x_0 (x_0 <= 1 or x_0 >= 0 on [0, 1]) imply nothing for a branch.
TEST(ProbingImpliedBound, ProbeCoveringBothValuesIsSkipped)
{
  simplex::user_problem_t<int, double> problem(nullptr);
  problem.num_cols = 2;
  problem.lower    = {0.0, 0.0};
  problem.upper    = {1.0, kInf};

  probing_cache_t<int, double> cache;
  auto& entries                         = cache.probing_cache[0];
  entries[0].val_interval               = {1.0, interval_type_t::LEQ};
  entries[0].var_to_cached_bound_map[1] = {0.0, 3.0};
  entries[1].val_interval               = {0.0, interval_type_t::GEQ};
  entries[1].var_to_cached_bound_map[1] = {4.0, kInf};

  probing_implied_bound_t<int, double> implied;
  extract_probing_implied_bounds<int, double>({}, problem, cache, implied);

  EXPECT_EQ(implied.zero_offsets, (std::vector<int>{0, 0, 0}));
  EXPECT_EQ(implied.one_offsets, (std::vector<int>{0, 0, 0}));
}

// Regular binary probing: {x_0 = 0} first, {x_0 = 1} second.
TEST(ProbingImpliedBound, BinaryProbeKeepsOrientation)
{
  simplex::user_problem_t<int, double> problem(nullptr);
  problem.num_cols = 2;
  problem.lower    = {0.0, 0.0};
  problem.upper    = {1.0, 10.0};

  probing_cache_t<int, double> cache;
  auto& entries                         = cache.probing_cache[0];
  entries[0].val_interval               = {0.0, interval_type_t::EQUALS};
  entries[0].var_to_cached_bound_map[1] = {0.0, 2.0};
  entries[1].val_interval               = {1.0, interval_type_t::EQUALS};
  entries[1].var_to_cached_bound_map[1] = {5.0, 10.0};

  probing_implied_bound_t<int, double> implied;
  extract_probing_implied_bounds<int, double>({}, problem, cache, implied);

  ASSERT_EQ(implied.zero_offsets, (std::vector<int>{0, 1, 1}));
  EXPECT_EQ(implied.zero_upper_bound, (std::vector<double>{2.0}));
  ASSERT_EQ(implied.one_offsets, (std::vector<int>{0, 1, 1}));
  EXPECT_EQ(implied.one_lower_bound, (std::vector<double>{5.0}));
}

}  // namespace cuopt::mathematical_optimization::mip::test
