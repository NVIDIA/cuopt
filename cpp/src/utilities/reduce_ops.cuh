/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <raft/util/cuda_dev_essentials.cuh>

namespace cuopt {

// abs(value); used as a cub/thrust transform-iterator input op ahead of an inf-norm reduce.
template <typename f_t>
struct abs_value_transform_t {
  __device__ f_t operator()(f_t value) const { return raft::abs(value); }
};

template <typename item_t>
struct max_op_t {
  __host__ __device__ item_t operator()(const item_t& lhs, const item_t& rhs) const
  {
    return lhs > rhs ? lhs : rhs;
  }
};

template <typename item_t>
struct min_op_t {
  __host__ __device__ item_t operator()(const item_t& lhs, const item_t& rhs) const
  {
    return lhs < rhs ? lhs : rhs;
  }
};

}  // namespace cuopt
