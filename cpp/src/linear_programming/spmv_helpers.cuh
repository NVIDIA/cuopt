/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "spmv_kernels.cuh"

#include <raft/core/device_span.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct identity_functor {
  __host__ __device__ __forceinline__ void operator()(i_t idx,
                                                      f_t x,
                                                      raft::device_span<f_t> output) const
  {
    output[idx] = x;
  }
};

template <typename i_t,
          typename f_t,
          typename view_t,
          typename functor_t = identity_functor<i_t, f_t>>
void spmv_call(rmm::cuda_stream_view stream,
               view_t view,
               raft::device_span<f_t> input,
               raft::device_span<f_t> output,
               raft::device_span<f_t> tmp_out,
               i_t item_sub_warp_count,
               i_t item_blocks_count,
               i_t heavy_block_count,
               i_t heavy_items_beg_id,
               i_t num_heavy_items,
               i_t heavy_work_per_block,
               const rmm::device_uvector<i_t>& warp_item_offsets,
               const rmm::device_uvector<i_t>& warp_item_id_offsets,
               const rmm::device_uvector<i_t>& heavy_items_vertex_ids,
               const rmm::device_uvector<i_t>& heavy_items_pseudo_block_ids,
               const rmm::device_uvector<i_t>& heavy_items_block_segments,
               functor_t functor = identity_functor<i_t, f_t>())
{
  constexpr i_t block_size = 256;
  i_t num_sub_warp_blocks  = raft::ceildiv(item_sub_warp_count * raft::WarpSize, block_size);
  spmv_kernel<i_t, f_t, block_size>
    <<<item_blocks_count + heavy_block_count, block_size, 0, stream>>>(
      view,
      input,
      output,
      tmp_out,
      num_sub_warp_blocks,
      item_blocks_count,
      heavy_items_beg_id,
      heavy_work_per_block,
      make_span(warp_item_offsets),
      make_span(warp_item_id_offsets),
      make_span(heavy_items_vertex_ids),
      make_span(heavy_items_pseudo_block_ids),
      functor);
  if (heavy_block_count != 0) {
    finalize_spmv_kernel<i_t, f_t><<<num_heavy_items, 32, 0, stream>>>(
      heavy_items_beg_id, make_span(heavy_items_block_segments), tmp_out, view, output, functor);
  }
}

}  // namespace cuopt::linear_programming::detail
