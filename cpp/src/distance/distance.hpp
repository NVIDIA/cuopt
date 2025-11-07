/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <routing/routing_details.hpp>
#include <routing/structures.hpp>

#include <cuda_fp16.h>

namespace cuopt {
namespace routing {
namespace detail {

template <typename f_t>
constexpr f_t distance(const f_t px1, const f_t py1, const f_t px2, const f_t py2)
{
  f_t diff_x = (px1 - px2);
  f_t diff_y = (py1 - py2);
  return sqrtf(diff_x * diff_x + diff_y * diff_y);
}

template <typename i_t, typename f_t>
constexpr f_t euclidean_dist(const f_t* px, const f_t* py, const i_t a, const i_t b)
{
  f_t px_a   = static_cast<f_t>(px[a]);
  f_t px_b   = static_cast<f_t>(px[b]);
  f_t py_a   = static_cast<f_t>(py[a]);
  f_t py_b   = static_cast<f_t>(py[b]);
  f_t diff_x = (px_a - px_b);
  f_t diff_y = (py_a - py_b);
  return sqrt(diff_x * diff_x + diff_y * diff_y);
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
