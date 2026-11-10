/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mip_heuristics/diversity/fix_propagate_host.hpp>

#include <gtest/gtest.h>

#include <random>

namespace cuopt::mathematical_optimization::test {
namespace {
using mip::fix_propagate_preferred_value;
using mip::fix_propagate_value_strategy_t;

constexpr double integrality_tolerance = 1e-6;
constexpr double infinity              = std::numeric_limits<double>::infinity();

double value(fix_propagate_value_strategy_t strategy,
             double reference_value,
             double uniform_draw = 0.,
             double value_param  = 0.)
{
  return fix_propagate_preferred_value<double>(
    strategy, reference_value, 0., infinity, 0., value_param, integrality_tolerance, uniform_draw);
}

}  // namespace

TEST(FixPropagateValue, RoundUp)
{
  EXPECT_EQ(value(fix_propagate_value_strategy_t::ROUND_UP, 2.3), 3.);
  EXPECT_EQ(value(fix_propagate_value_strategy_t::ROUND_UP, 2.0), 2.);
  EXPECT_EQ(value(fix_propagate_value_strategy_t::ROUND_UP, 2.0 - 1e-9), 2.);
}

TEST(FixPropagateValue, LpRandomizedUsesTheFractionalPart)
{
  const auto strategy = fix_propagate_value_strategy_t::LP_RANDOMIZED;
  EXPECT_EQ(value(strategy, 2.3, 0.29), 3.);
  EXPECT_EQ(value(strategy, 2.3, 0.31), 2.);
  // integral reference values are never moved
  for (double uniform_draw : {0., 0.5, 0.999}) {
    EXPECT_EQ(value(strategy, 4.0, uniform_draw), 4.);
    EXPECT_EQ(value(strategy, 4.0 - 1e-9, uniform_draw), 4.);
  }
}

TEST(FixPropagateValue, LpRandomizedRoundsUpWithTheFractionalProbability)
{
  std::mt19937 generator(7);
  std::uniform_real_distribution<double> uniform(0., 1.);
  for (double fractional_part : {0.05, 0.25, 0.5, 0.9}) {
    const int n_draws = 20000;
    int n_rounded_up  = 0;
    for (int i = 0; i < n_draws; ++i) {
      n_rounded_up += value(fix_propagate_value_strategy_t::LP_RANDOMIZED,
                            1. + fractional_part,
                            uniform(generator)) == 2.;
    }
    EXPECT_NEAR(double(n_rounded_up) / n_draws, fractional_part, 0.015)
      << "fractional part " << fractional_part;
  }
}

TEST(FixPropagateValue, AbsoluteReducedCostOrderRanksBothSigns)
{
  using mip::fix_propagate_order_key;
  using mip::fix_propagate_variable_strategy_t;
  const auto strategy = fix_propagate_variable_strategy_t::REDUCED_COST_ABS;
  EXPECT_GT((fix_propagate_order_key<double>(strategy, 0.5, 5.)),
            (fix_propagate_order_key<double>(strategy, 0.5, -1.)));
  EXPECT_GT((fix_propagate_order_key<double>(strategy, 0.5, -5.)),
            (fix_propagate_order_key<double>(strategy, 0.5, 1.)));
  // TYPE keeps the formulation order (equal keys, stable sort)
  EXPECT_EQ((fix_propagate_order_key<double>(fix_propagate_variable_strategy_t::TYPE, 0.2, 3.)),
            (fix_propagate_order_key<double>(fix_propagate_variable_strategy_t::TYPE, 0.9, -3.)));
}

TEST(FixPropagateValue, ObjectiveAndRandomBoundValues)
{
  auto preferred = [](fix_propagate_value_strategy_t strategy,
                      double objective_coefficient,
                      double uniform_draw = 0.,
                      double ub           = 4.) {
    return fix_propagate_preferred_value<double>(
      strategy, 1.5, 0., ub, 0., 0., integrality_tolerance, uniform_draw, objective_coefficient);
  };
  EXPECT_EQ(preferred(fix_propagate_value_strategy_t::GOOD_OBJ, -2.),
            4.);  // a negative cost is better at the top
  EXPECT_EQ(preferred(fix_propagate_value_strategy_t::GOOD_OBJ, 2.), 0.);
  EXPECT_EQ(preferred(fix_propagate_value_strategy_t::GOOD_OBJ, 0.),
            0.);  // no objective: the lower bound
  EXPECT_EQ(preferred(fix_propagate_value_strategy_t::BAD_OBJ, 2.), 4.);
  EXPECT_EQ(preferred(fix_propagate_value_strategy_t::BAD_OBJ, -2.), 0.);
  EXPECT_EQ(preferred(fix_propagate_value_strategy_t::RANDOM_BOUND, 0., 0.2), 4.);
  EXPECT_EQ(preferred(fix_propagate_value_strategy_t::RANDOM_BOUND, 0., 0.7), 0.);
  // an infinite chosen bound falls back to rounding the reference value up
  EXPECT_EQ(preferred(fix_propagate_value_strategy_t::GOOD_OBJ, -2., 0., infinity), 2.);
  EXPECT_EQ(preferred(fix_propagate_value_strategy_t::UPPER_BOUND, 0., 0., infinity), 2.);
}

}  // namespace cuopt::mathematical_optimization::test
