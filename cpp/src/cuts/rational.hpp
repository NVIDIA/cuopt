/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

// Best rational approximation p/q to x with q <= max_denominator, via continued fractions.
// On success, returns true with numerator/denominator set to a rational within 1e-14 of x.
// On failure (overflow, or no approximation within 1e-14 for q <= max_denominator), returns
// false; the out parameters hold the best convergent found, or are left untouched on early
// overflow.
template <typename f_t>
bool rational_approximation(f_t x,
                            int64_t max_denominator,
                            int64_t& numerator,
                            int64_t& denominator)
{
  int64_t a, p0 = 0, q0 = 1, p1 = 1, q1 = 0;
  f_t val       = x;
  bool negative = false;

  if (x < 0) {
    negative = true;
    val      = -val;
  }

  while (1) {
    a = (int64_t)std::floor(val);
    if (a < 0 || a > INT64_MAX) { return false; }  // Protect against overflow
    int64_t p2 = a * p1 + p0;
    int64_t q2 = a * q1 + q0;
    if (q2 > max_denominator) { break; }
    p0 = p1;
    q0 = q1;
    p1 = p2;
    q1 = q2;

    f_t rem = val - a;
    if (rem < 1e-14) { break; }
    val = 1.0 / rem;
  }

  numerator   = negative ? -p1 : p1;
  denominator = q1;

  f_t approx = static_cast<f_t>(numerator) / static_cast<f_t>(denominator);
  f_t err    = std::abs(approx - x);
  return err <= 1e-14;
}

// Rational number: p/q with q > 0.
// Together p and q occupy 128 bits, so rational128_t represents a rational with 128 bits of
// precision. The class is templated on the floating-point type f_t used for conversions to and
// from the rational representation; the internal numerator/denominator are always int64_t.
template <typename f_t>
struct rational128_t {
  int64_t p{0};
  int64_t q{1};

  static rational128_t from_floating_point(f_t x, int64_t max_d = 10000000)
  {
    int64_t num = 0, den = 1;
    // Ignore the success bool: even when rational_approximation reports the approximation
    // is not exact within 1e-14, the out-parameters still hold the best convergent found.
    // On the early-overflow path the out-parameters are left at our 0/1 default, which is a
    // safe placeholder representing 0.
    rational_approximation(x, max_d, num, den);
    if (den <= 0) {
      den = 1;
      num = static_cast<int64_t>(std::llround(static_cast<double>(x)));
    }
    return {num, den};
  }

  // Safe version: returns {0, 1} if the rational approximation is not accurate within 1e-14.
  // This allows callers to detect failure by testing is_zero() on the result.
  static rational128_t safe_from_floating_point(f_t x, int64_t max_d = 10000000)
  {
    int64_t num = 0, den = 1;
    if (!rational_approximation(x, max_d, num, den)) return {0, 1};
    if (den <= 0) return {0, 1};
    return {num, den};
  }

  // Convert to floating-point in double precision, then cast to f_t.
  // The intermediate double avoids precision loss when f_t is float.
  f_t to_floating_point() const
  {
    return static_cast<f_t>(static_cast<double>(p) / static_cast<double>(q));
  }

  // Reduce to lowest terms
  rational128_t reduced() const
  {
    if (p == 0) return {0, 1};
    int64_t g = std::gcd(std::abs(p), q);
    return {p / g, q / g};
  }

  rational128_t operator*(const rational128_t& o) const
  {
    // Multiply with cross-cancellation to reduce overflow
    int64_t g1 = std::gcd(std::abs(p), o.q);
    int64_t g2 = std::gcd(std::abs(o.p), q);
    return rational128_t{(p / g1) * (o.p / g2), (q / g2) * (o.q / g1)};
  }

  rational128_t operator/(const rational128_t& o) const
  {
    if (o.p == 0) return {0, 1};
    rational128_t inv = (o.p > 0) ? rational128_t{o.q, o.p} : rational128_t{-o.q, -o.p};
    return *this * inv;
  }

  rational128_t operator+(const rational128_t& o) const
  {
    int64_t g  = std::gcd(q, o.q);
    int64_t lq = q / g;
    return rational128_t{p * (o.q / g) + o.p * lq, lq * o.q}.reduced();
  }

  rational128_t operator-(const rational128_t& o) const { return *this + rational128_t{-o.p, o.q}; }

  rational128_t abs() const { return {std::abs(p), q}; }

  bool is_zero() const { return p == 0; }
};

// GCD of two rationals: gcd(a/b, c/d) = gcd(a*d, c*b) / (b*d).
// Treats both operands as non-negative (operates on their absolute values).
// Intended to be found via Argument-Dependent Lookup on the rational128_t<f_t> argument type.
template <typename f_t>
rational128_t<f_t> gcd(rational128_t<f_t> a, rational128_t<f_t> b)
{
  if (a.is_zero()) return b.abs();
  if (b.is_zero()) return a.abs();
  a           = a.abs();
  b           = b.abs();
  int64_t num = std::gcd(a.p * b.q, b.p * a.q);
  int64_t den = a.q * b.q;
  int64_t g   = std::gcd(num, den);
  return {num / g, den / g};
}

// GCD of two floating-point values, computed by rationalizing each operand and taking the
// rational GCD, then converting back to f_t. Useful for computing the step size of an
// objective whose nonzero coefficients are integer (or implied-integer) multiples of a
// common rational.
template <typename f_t>
f_t gcd_floating_point(f_t a, f_t b)
{
  // Short-circuit near-zero inputs. rational128_t<f_t>::from_floating_point uses a 1e-14
  // internal tolerance for rational_approximation; checking against a coarser eps here
  // preserves the historical behavior of treating very small inputs as zero.
  constexpr f_t eps = static_cast<f_t>(1e-9);
  if (std::abs(a) < eps) return std::abs(b);
  if (std::abs(b) < eps) return std::abs(a);
  auto ra = rational128_t<f_t>::from_floating_point(std::abs(a));
  auto rb = rational128_t<f_t>::from_floating_point(std::abs(b));
  return gcd(ra, rb).to_floating_point();
}

// GCD of the absolute values of a vector of floats whose entries are close to integers.
// Returns 0 if every entry rounds to zero (or the vector is empty).
template <typename f_t>
f_t gcd_of_integer_values(const std::vector<f_t>& values)
{
  int64_t g = 0;
  for (f_t v : values) {
    int64_t iv = std::llround(std::abs(v));
    if (iv == 0) continue;
    g = (g == 0) ? iv : std::gcd(g, iv);
  }
  return static_cast<f_t>(g);
}

}  // namespace cuopt::linear_programming::dual_simplex
