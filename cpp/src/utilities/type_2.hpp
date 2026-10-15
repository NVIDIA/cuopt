/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector_types.h>

namespace cuopt {

template <typename T>
struct type_2 {
  using type = void;
};

template <>
struct type_2<int> {
  using type = int2;
};

template <>
struct type_2<float> {
  using type = float2;
};

template <>
struct type_2<double> {
  using type = double2;
};

#if defined(__CUDACC__)
#define CUOPT_TYPE_2_HOST_DEVICE inline __host__ __device__
#else
#define CUOPT_TYPE_2_HOST_DEVICE inline
#endif

template <typename f_t2>
CUOPT_TYPE_2_HOST_DEVICE auto& get_lower(f_t2& value)
{
  return value.x;
}

template <typename f_t2>
CUOPT_TYPE_2_HOST_DEVICE auto& get_upper(f_t2& value)
{
  return value.y;
}

#undef CUOPT_TYPE_2_HOST_DEVICE

}  // namespace cuopt
