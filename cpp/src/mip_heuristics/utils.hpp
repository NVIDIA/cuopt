/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

#ifdef __CUDACC__
#define CUOPT_MIP_HOST_DEVICE inline __host__ __device__
#else
#define CUOPT_MIP_HOST_DEVICE inline
#endif

namespace cuopt::mathematical_optimization::mip {

// checks if a given float value can be exactly represented as an integer of type int_t.
template <typename int_t, typename f_t>
inline bool is_exactly_representable(f_t value)
{
  static_assert(std::is_integral_v<int_t>);
  static_assert(std::is_floating_point_v<f_t>);

  if constexpr (std::numeric_limits<f_t>::digits < std::numeric_limits<int_t>::digits) {
    return false;
  } else {
    return std::isfinite(value) && std::trunc(value) == value &&
           value >= (f_t)std::numeric_limits<int_t>::lowest() &&
           value <= (f_t)std::numeric_limits<int_t>::max();
  }
}

// Ogita-Rump-Oishi Dot2. TwoProduct recovers the rounding of each coefficient-value product, which
// a compensated summation over already-multiplied terms cannot see.
// https://epubs.siam.org/doi/abs/10.1137/030601818
template <typename UIt, typename VIt>
__attribute__((optimize("no-fast-math"))) CUOPT_MIP_HOST_DEVICE auto compensated_dot2(UIt u_it,
                                                                                      VIt v_it,
                                                                                      std::size_t n)
{
  using f_t = decltype(u_it[0] * v_it[0]);
  f_t p     = 0;
  f_t s     = 0;
  for (std::size_t k = 0; k < n; ++k) {
    const f_t u = u_it[k];
    const f_t v = v_it[k];
    const f_t h = u * v;
    const f_t t = p + h;
    const f_t z = t - p;
    s += ((p - (t - z)) + (h - z)) + fma(u, v, -h);
    p = t;
  }
  return p + s;
}

}  // namespace cuopt::mathematical_optimization::mip

#undef CUOPT_MIP_HOST_DEVICE
