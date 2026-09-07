/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace cuopt_bench {

struct row_sum_t {
  double activity;
  double positive_activity;
  double error_bound;
};

template <typename value_t, typename index_t, typename solution_t>
row_sum_t row_sum_fp64(const value_t* values,
                       const index_t* indices,
                       int64_t begin,
                       int64_t end,
                       const solution_t* solution)
{
  double activity = 0.0;
  double positive = 0.0;
  double abs_sum  = 0.0;
  for (int64_t entry = begin; entry < end; ++entry) {
    const double term = (double)values[entry] * (double)solution[indices[entry]];
    activity += term;
    abs_sum += std::fabs(term);
    if (term > 0.0) positive += term;
  }
  const double width = (double)(end - begin);
  return {activity, positive, (width + 1.0) * std::numeric_limits<double>::epsilon() * abs_sum};
}

struct row_sum_wide_t {
  _Float128 activity;
  _Float128 positive_activity;
};

template <typename value_t, typename index_t, typename solution_t>
row_sum_wide_t row_sum_exact(const value_t* values,
                             const index_t* indices,
                             int64_t begin,
                             int64_t end,
                             const solution_t* solution)
{
  _Float128 activity = 0;
  _Float128 positive = 0;
  for (int64_t entry = begin; entry < end; ++entry) {
    const _Float128 term = (_Float128)values[entry] * (_Float128)solution[indices[entry]];
    activity += term;
    if (term > 0) positive += term;
  }
  return {activity, positive};
}

inline double scaled_tolerance(double absolute_tolerance, double first, double second)
{
  return absolute_tolerance * std::max({1.0, std::fabs(first), std::fabs(second)});
}

inline std::pair<double, double> scaled_row_limits(double absolute_tolerance,
                                                   double positive_activity,
                                                   double lower_bound,
                                                   double upper_bound)
{
  return {lower_bound - scaled_tolerance(absolute_tolerance, positive_activity, lower_bound),
          upper_bound + scaled_tolerance(absolute_tolerance, positive_activity, upper_bound)};
}

inline std::pair<double, double> scaled_bound_limits(double absolute_tolerance,
                                                     double value,
                                                     double lower_bound,
                                                     double upper_bound)
{
  return {lower_bound - scaled_tolerance(absolute_tolerance, lower_bound, value),
          upper_bound + scaled_tolerance(absolute_tolerance, upper_bound, value)};
}

inline double bound_excess(double value,
                           double lower_bound,
                           double upper_bound,
                           double absolute_tolerance)
{
  const auto limits = scaled_bound_limits(absolute_tolerance, value, lower_bound, upper_bound);
  return std::max(std::max(limits.first - value, value - limits.second), 0.0);
}

struct row_verdict_t {
  double activity;
  double excess;
  double raw_excess;
  double lower_limit;
  double upper_limit;
  bool escalated;
};

template <typename value_t, typename index_t, typename solution_t, typename limits_t>
row_verdict_t check_row(const value_t* values,
                        const index_t* indices,
                        int64_t begin,
                        int64_t end,
                        const solution_t* solution,
                        double lower_bound,
                        double upper_bound,
                        limits_t limits)
{
  const row_sum_t sum = row_sum_fp64(values, indices, begin, end, solution);
  auto bound          = limits(sum.positive_activity);
  const bool satisfied =
    sum.activity - sum.error_bound >= bound.first && sum.activity + sum.error_bound <= bound.second;
  const bool undecided =
    sum.activity + sum.error_bound >= bound.first && sum.activity - sum.error_bound <= bound.second;

  if (satisfied || !undecided) {
    const double excess =
      std::max(std::max(bound.first - sum.activity, sum.activity - bound.second), 0.0);
    const double raw_excess =
      std::max(std::max(lower_bound - sum.activity, sum.activity - upper_bound), 0.0);
    return {sum.activity, excess, raw_excess, bound.first, bound.second, false};
  }

  const row_sum_wide_t wide = row_sum_exact(values, indices, begin, end, solution);
  bound                     = limits((double)wide.positive_activity);
  const _Float128 zero      = 0;
  const double excess       = (double)std::max(
    std::max((_Float128)bound.first - wide.activity, wide.activity - (_Float128)bound.second),
    zero);
  const double raw_excess = (double)std::max(
    std::max((_Float128)lower_bound - wide.activity, wide.activity - (_Float128)upper_bound), zero);
  return {(double)wide.activity, excess, raw_excess, bound.first, bound.second, true};
}

}  // namespace cuopt_bench
