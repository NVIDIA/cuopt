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

#include <mip/problem/problem.cuh>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t>
struct calc_blocks_per_item_t {
  calc_blocks_per_item_t(raft::device_span<const i_t> offsets_, i_t work_per_block_)
    : offsets(offsets_), work_per_block(work_per_block_)
  {
  }
  raft::device_span<const i_t> offsets;
  i_t work_per_block;
  __device__ __forceinline__ i_t operator()(i_t item_id) const
  {
    i_t work_per_vertex = (offsets[item_id + 1] - offsets[item_id]);
    return raft::ceildiv<i_t>(work_per_vertex, work_per_block);
  }
};

template <typename i_t>
struct heavy_vertex_meta_t {
  heavy_vertex_meta_t(raft::device_span<const i_t> offsets_,
                      raft::device_span<i_t> vertex_id_,
                      raft::device_span<i_t> pseudo_block_id_)
    : offsets(offsets_), vertex_id(vertex_id_), pseudo_block_id(pseudo_block_id_)
  {
  }

  raft::device_span<const i_t> offsets;
  raft::device_span<i_t> vertex_id;
  raft::device_span<i_t> pseudo_block_id;

  __device__ __forceinline__ void operator()(i_t id) const
  {
    vertex_id[offsets[id]] = id;
    if (id != 0) {
      pseudo_block_id[offsets[id]] = offsets[id - 1] - offsets[id] + 1;
    } else {
      pseudo_block_id[offsets[0]] = 0;
    }
  }
};

template <typename i_t, typename f_t>
__global__ void graph_data_copy(raft::device_span<i_t> reorg_ids,
                                raft::device_span<i_t> offsets,
                                raft::device_span<f_t> coeff,
                                raft::device_span<i_t> edge,
                                raft::device_span<i_t> pb_offsets,
                                raft::device_span<f_t> pb_coeff,
                                raft::device_span<i_t> pb_edge)
{
  i_t new_idx = blockIdx.x;
  i_t idx     = reorg_ids[new_idx];

  auto read_beg = pb_offsets[idx];
  auto read_end = pb_offsets[idx + 1];

  auto write_beg = offsets[new_idx];

  for (i_t i = threadIdx.x; i < (read_end - read_beg); i += blockDim.x) {
    coeff[i + write_beg] = pb_coeff[i + read_beg];
    edge[i + write_beg]  = pb_edge[i + read_beg];
  }
}

template <typename i_t, typename f_t>
__global__ void check_data(raft::device_span<i_t> reorg_ids,
                           raft::device_span<i_t> offsets,
                           raft::device_span<f_t> coeff,
                           raft::device_span<i_t> edge,
                           raft::device_span<i_t> pb_offsets,
                           raft::device_span<f_t> pb_coeff,
                           raft::device_span<i_t> pb_edge,
                           i_t* errors)
{
  i_t new_idx = blockIdx.x;
  i_t idx     = reorg_ids[new_idx];

  auto src_read_beg = pb_offsets[idx];
  auto src_read_end = pb_offsets[idx + 1];
  auto dst_read_beg = offsets[new_idx];

  for (i_t i = threadIdx.x; i < (src_read_end - src_read_beg); i += blockDim.x) {
    if (coeff[i + dst_read_beg] != pb_coeff[i + src_read_beg]) {
      printf("coeff mismatch vertex id %d orig id %d at edge index %d\n", new_idx, idx, i);
      atomicAdd(errors, 1);
    }
    if (edge[i + dst_read_beg] != pb_edge[i + src_read_beg]) {
      printf("edge mismatch vertex id %d orig id %d at edge index %d\n", new_idx, idx, i);
      atomicAdd(errors, 1);
    }
    if (edge[i + dst_read_beg] >= pb_edge.size()) { printf("oob\n"); }
  }
}

template <typename i_t, typename f_t>
void create_graph(const raft::handle_t* handle_ptr,
                  rmm::device_uvector<i_t>& reorg_ids,
                  rmm::device_uvector<i_t>& offsets,
                  rmm::device_uvector<f_t>& coeff,
                  rmm::device_uvector<i_t>& edge,
                  rmm::device_uvector<i_t>& pb_offsets,
                  rmm::device_uvector<f_t>& pb_coeff,
                  rmm::device_uvector<i_t>& pb_edge,
                  bool debug)
{
  //  calculate degree and store in offsets
  thrust::transform(
    handle_ptr->get_thrust_policy(),
    reorg_ids.begin(),
    reorg_ids.end(),
    offsets.begin(),
    [off = make_span(pb_offsets)] __device__(auto id) { return off[id + 1] - off[id]; });

  //  create offsets
  thrust::exclusive_scan(
    handle_ptr->get_thrust_policy(), offsets.begin(), offsets.end(), offsets.begin());

  // copy adjacency lists and vertex properties
  graph_data_copy<i_t, f_t>
    <<<reorg_ids.size(), 256, 0, handle_ptr->get_stream()>>>(make_span(reorg_ids),
                                                             make_span(offsets),
                                                             make_span(coeff),
                                                             make_span(edge),
                                                             make_span(pb_offsets),
                                                             make_span(pb_coeff),
                                                             make_span(pb_edge));

  if (debug) {
    rmm::device_scalar<i_t> errors(0, handle_ptr->get_stream());
    check_data<i_t, f_t>
      <<<reorg_ids.size(), 256, 0, handle_ptr->get_stream()>>>(make_span(reorg_ids),
                                                               make_span(offsets),
                                                               make_span(coeff),
                                                               make_span(edge),
                                                               make_span(pb_offsets),
                                                               make_span(pb_coeff),
                                                               make_span(pb_edge),
                                                               errors.data());
    i_t error_count = errors.value(handle_ptr->get_stream());
    if (error_count != 0) { std::cerr << "adjacency list copy mismatch\n"; }
  }
}

template <typename i_t>
std::tuple<i_t, i_t> create_heavy_item_block_segments(rmm::cuda_stream_view stream,
                                                      rmm::device_uvector<i_t>& vertex_id,
                                                      rmm::device_uvector<i_t>& pseudo_block_id,
                                                      rmm::device_uvector<i_t>& item_block_segments,
                                                      const i_t heavy_degree_cutoff,
                                                      const std::vector<i_t>& bin_offsets,
                                                      rmm::device_uvector<i_t> const& offsets)
{
  auto heavy_id_beg   = bin_offsets[std::log2(heavy_degree_cutoff) + 1];
  auto n_items        = offsets.size() - 1;
  auto heavy_id_count = n_items - heavy_id_beg;
  item_block_segments.resize(1 + heavy_id_count, stream);

  // Amount of blocks to be launched for each item (constraint or variable).
  auto work_per_block              = heavy_degree_cutoff / 2;
  auto calc_blocks_per_vertex_iter = thrust::make_transform_iterator(
    thrust::make_counting_iterator<i_t>(heavy_id_beg),
    calc_blocks_per_item_t<i_t>{make_span(offsets), work_per_block});

  // Inclusive scan so that each block can determine which item it belongs to
  item_block_segments.set_element_to_zero_async(0, stream);
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         calc_blocks_per_vertex_iter,
                         calc_blocks_per_vertex_iter + heavy_id_count,
                         item_block_segments.begin() + 1);
  auto num_blocks = item_block_segments.back_element(stream);
  if (num_blocks > 0) {
    vertex_id.resize(num_blocks, stream);
    pseudo_block_id.resize(num_blocks, stream);
    thrust::fill(rmm::exec_policy(stream), vertex_id.begin(), vertex_id.end(), i_t{-1});
    thrust::fill(rmm::exec_policy(stream), pseudo_block_id.begin(), pseudo_block_id.end(), i_t{1});
    thrust::for_each(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<i_t>(0),
      thrust::make_counting_iterator<i_t>(item_block_segments.size() - 1),
      heavy_vertex_meta_t<i_t>{
        make_span(item_block_segments), make_span(vertex_id), make_span(pseudo_block_id)});
    thrust::inclusive_scan(rmm::exec_policy(stream),
                           vertex_id.begin(),
                           vertex_id.end(),
                           vertex_id.begin(),
                           thrust::maximum<i_t>{});
    thrust::inclusive_scan(rmm::exec_policy(stream),
                           pseudo_block_id.begin(),
                           pseudo_block_id.end(),
                           pseudo_block_id.begin(),
                           thrust::plus<i_t>{});
  }
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  // Total number of blocks that have to be launched
  return std::make_tuple(num_blocks, heavy_id_beg);
}

template <typename i_t>
std::tuple<i_t, i_t> block_meta(rmm::cuda_stream_view stream,
                                i_t heavy_id_beg,
                                rmm::device_uvector<i_t>& d_warp_offsets,
                                rmm::device_uvector<i_t>& d_warp_id_offsets,
                                rmm::device_uvector<i_t>& d_block_offsets,
                                rmm::device_uvector<i_t>& d_block_id_offsets,
                                const std::vector<i_t>& bin_offsets,
                                i_t w_t_r,
                                i_t heavy_w_cut_off,
                                bool debug = false)
{
  i_t block_size = 256;

  std::vector<i_t> warp_offsets;
  std::vector<i_t> warp_id_offsets;
  warp_offsets.reserve(32);
  warp_id_offsets.reserve(32);

  for (i_t t_p_v = 1; t_p_v <= 16 * 2; t_p_v *= 2) {
    warp_id_offsets.push_back(bin_offsets[std::log2(t_p_v * w_t_r) + 1]);
  }

  // start with non-zero vertices
  warp_offsets.push_back(0);
  warp_id_offsets[0] = bin_offsets[0];
  for (i_t t_p_v = 1; t_p_v <= 16; t_p_v *= 2) {
    auto num_items  = warp_id_offsets[std::log2(t_p_v) + 1] - warp_id_offsets[std::log2(t_p_v)];
    auto warp_count = raft::ceildiv<i_t>(num_items * t_p_v, raft::WarpSize);
    warp_offsets.push_back(warp_count + warp_offsets.back());
  }

  auto num_sub_warps       = warp_offsets.back();
  auto num_sub_warp_blocks = raft::ceildiv(raft::WarpSize * num_sub_warps, block_size);

  //[128, 256]
  std::vector<i_t> block_id_offsets;
  std::vector<i_t> block_offsets;
  block_id_offsets.push_back(bin_offsets[std::log2(16 * 2 * w_t_r) + 1]);
  block_offsets.push_back(num_sub_warp_blocks);

  block_id_offsets.push_back(bin_offsets[std::log2(16 * 2 * w_t_r) + 3]);
  block_offsets.push_back(block_offsets.back() +
                          raft::ceildiv(bin_offsets[std::log2(16 * 2 * w_t_r) + 3] -
                                          bin_offsets[std::log2(16 * 2 * w_t_r) + 1],
                                        block_size / 64));

  //[512, heavy_degree_cutoff/2]
  block_id_offsets.push_back(heavy_id_beg);
  block_offsets.push_back(block_offsets.back() + heavy_id_beg -
                          bin_offsets[std::log2(16 * 2 * w_t_r) + 3]);

  expand_device_copy(d_warp_offsets, warp_offsets, stream);
  expand_device_copy(d_warp_id_offsets, warp_id_offsets, stream);
  expand_device_copy(d_block_offsets, block_offsets, stream);
  expand_device_copy(d_block_id_offsets, block_id_offsets, stream);
  return std::make_tuple(num_sub_warps, block_offsets.back());
}

}  // namespace cuopt::linear_programming::detail
