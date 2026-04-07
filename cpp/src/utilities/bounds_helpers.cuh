/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/macros.cuh>

#include <cmath>

namespace cuopt::detail {

template <typename f_t>
HDI constexpr f_t abs_no_raffunc(f_t value)
{
  return value < f_t(0) ? -value : value;
}

template <typename f_t>
struct combine_finite_abs_bounds {
  HDI f_t operator()(f_t lower, f_t upper) const
  {
    f_t value = f_t(0);
    if (isfinite(upper)) { value = abs_no_raffunc(upper); }
    if (isfinite(lower)) { value = max(value, abs_no_raffunc(lower)); }
    return value;
  }
};

}  // namespace cuopt::detail

namespace cuopt::linear_programming::detail {

template <typename f_t>
using combine_finite_abs_bounds = ::cuopt::detail::combine_finite_abs_bounds<f_t>;

}  // namespace cuopt::linear_programming::detail
