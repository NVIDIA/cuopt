/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <mip/problem/problem.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t>
__device__ __forceinline__ void detect_range_sub_warp(i_t* id_warp_beg,
                                                      i_t* id_range_end,
                                                      i_t* threads_per_item,
                                                      raft::device_span<i_t> warp_offsets,
                                                      raft::device_span<i_t> bin_offsets)
{
  i_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / raft::WarpSize;
  i_t lane_id = threadIdx.x & 31;
  bool pred   = false;
  if (lane_id < warp_offsets.size()) { pred = (warp_id >= warp_offsets[lane_id]); }
  unsigned int m  = __ballot_sync(0xffffffff, pred);
  i_t seg         = 31 - __clz(m);
  i_t it_per_warp = (1 << (5 - seg));  // item per warp = raft::WarpSize/(2^seg)
  if (5 - seg < 0) {
    *threads_per_item = 0;
    return;
  }
  i_t beg           = bin_offsets[seg] + (warp_id - warp_offsets[seg]) * it_per_warp;
  i_t end           = bin_offsets[seg + 1];
  *id_warp_beg      = beg;
  *id_range_end     = end;
  *threads_per_item = (1 << seg);
}

}  // namespace cuopt::linear_programming::detail
