/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <utilities/macros.cuh>

namespace cuopt {

namespace detail {

// Best rational approximation p/q to x with q <= max_denom, via continued fractions. Returns the
// last valid convergent if the denominator limit is reached.
inline std::pair<int64_t, int64_t> rational_approximation(double x,
                                                          int64_t max_denom,
                                                          double epsilon)
{
  cuopt_assert(std::isfinite(x), "non-finite coefficient");
  if (!std::isfinite(x)) return {0, 0};

  double ax = std::abs(x);
  if (ax < epsilon) { return {0, 1}; }

  if (x < 0) {
    auto [p, q] = rational_approximation(-x, max_denom, epsilon);
    return {-p, q};
  }

  const double integer_part = std::floor(x);
  if (integer_part >= (double)std::numeric_limits<int64_t>::max()) return {0, 0};

  int64_t p_prev2 = 1, q_prev2 = 0;
  int64_t p_prev1 = (int64_t)integer_part, q_prev1 = 1;

  double remainder = x - integer_part;

  for (int iter = 0; iter < 100; ++iter) {
    if (std::abs(remainder) < 1e-15) break;

    remainder             = 1.0 / remainder;
    const double quotient = std::floor(remainder);
    if (!std::isfinite(quotient) || quotient >= (double)std::numeric_limits<int64_t>::max()) {
      return {0, 0};
    }
    int64_t a = (int64_t)quotient;
    remainder -= a;

    int64_t p_product;
    int64_t q_product;
    int64_t p_curr;
    int64_t q_curr;
    if (__builtin_mul_overflow(a, p_prev1, &p_product) ||
        __builtin_add_overflow(p_product, p_prev2, &p_curr) ||
        __builtin_mul_overflow(a, q_prev1, &q_product) ||
        __builtin_add_overflow(q_product, q_prev2, &q_curr)) {
      return {0, 0};
    }

    if (q_curr > max_denom) break;

    p_prev2 = p_prev1;
    q_prev2 = q_prev1;
    p_prev1 = p_curr;
    q_prev1 = q_curr;

    double approx_err = x - (double)p_curr / (double)q_curr;
    if (std::abs(approx_err) < epsilon) break;
  }

  return {p_prev1, q_prev1};
}

// Brute-force: try scalars 1..max_brute and return the smallest that makes all coefficients
// integral.
inline double find_scaling_brute_force(const std::vector<double>& coefficients,
                                       int max_brute = 100,
                                       double tol    = 1e-6)
{
  for (int s = 1; s <= max_brute; ++s) {
    bool ok = true;
    for (double c : coefficients) {
      cuopt_assert(std::isfinite(c), "non-finite coefficient");
      if (!std::isfinite(c)) return std::numeric_limits<double>::quiet_NaN();
      double scaled = s * c;
      if (!std::isfinite(scaled) || std::abs(scaled - std::round(scaled)) > tol) {
        ok = false;
        break;
      }
    }
    if (ok) return (double)s;
  }
  return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace detail

// Continued-fractions approach: rationalize each coefficient, compute scm/gcd incrementally.
// Returns the smallest positive multiplier s such that s * c is (near-)integer for every c, or NaN
// if no such multiplier exists within the caps.
inline double find_scaling_rational(const std::vector<double>& coefficients,
                                    double maxscale     = 1e6,
                                    int64_t maxdnom     = 10000000,
                                    double maxfinal     = 10000,
                                    double intcheck_tol = 1e-6)
{
  constexpr double no_scaling = std::numeric_limits<double>::quiet_NaN();
  double epsilon              = 1.0 / maxscale;

  int64_t gcd = 0;
  int64_t scm = 1;

  for (double c : coefficients) {
    auto [num, den] = detail::rational_approximation(c, maxdnom, epsilon);
    if (den == 0) return no_scaling;
    if (num == 0) continue;

    if (num == std::numeric_limits<int64_t>::min()) return no_scaling;
    int64_t abs_num = std::abs(num);
    if (gcd == 0) {
      gcd = abs_num;
      scm = den;
    } else {
      gcd            = std::gcd(gcd, abs_num);
      int64_t factor = den / std::gcd(scm, den);
      int64_t new_scm;
      if (__builtin_mul_overflow(scm, factor, &new_scm)) return no_scaling;
      scm = new_scm;
    }

    if ((double)scm / (double)gcd > maxscale) return no_scaling;
  }

  if (gcd == 0) return 1.0;

  double intscalar = (double)scm / (double)gcd;
  if (intscalar > maxfinal) return no_scaling;

  for (double c : coefficients) {
    double scaled = intscalar * c;
    if (!std::isfinite(scaled) || std::abs(scaled - std::round(scaled)) > intcheck_tol)
      return no_scaling;
  }

  return intscalar;
}

// Finds the smallest integer scaling factor s such that s * c_i is integral for all i. Tries a
// brute-force sweep first (cheap, numerically robust), then falls back to continued fractions for
// larger scalars.
inline double find_objective_scaling_factor(const std::vector<double>& coefficients)
{
  double s = detail::find_scaling_brute_force(coefficients);
  if (!std::isnan(s)) return s;
  return find_scaling_rational(coefficients);
}

}  // namespace cuopt
