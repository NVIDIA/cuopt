/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <mip/problem/problem.cuh>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {

template <typename DryRunFunc, typename CaptureGraphFunc>
bool build_graph(managed_stream_pool& streams,
                 const raft::handle_t* handle_ptr,
                 cudaGraph_t& graph,
                 cudaGraphExec_t& graph_exec,
                 DryRunFunc d_func,
                 CaptureGraphFunc g_func)
{
  bool graph_created = false;
  cudaGraphCreate(&graph, 0);
  cudaEvent_t fork_stream_event;
  cudaEventCreate(&fork_stream_event);

  cudaStreamBeginCapture(handle_ptr->get_stream(), cudaStreamCaptureModeGlobal);
  cudaEventRecord(fork_stream_event, handle_ptr->get_stream());

  // dry-run - managed pool tracks how many streams were issued
  d_func();
  streams.wait_issued_on_event(fork_stream_event);
  streams.reset_issued();

  g_func();
  auto activity_done = streams.create_events_on_issued();
  streams.reset_issued();
  for (auto& e : activity_done) {
    cudaStreamWaitEvent(handle_ptr->get_stream(), e);
  }

  cudaStreamEndCapture(handle_ptr->get_stream(), &graph);
  RAFT_CHECK_CUDA(handle_ptr->get_stream());

  if (graph_exec != nullptr) {
    cudaGraphExecDestroy(graph_exec);
    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    RAFT_CHECK_CUDA(handle_ptr->get_stream());
  } else {
    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    RAFT_CHECK_CUDA(handle_ptr->get_stream());
  }

  cudaGraphDestroy(graph);
  graph_created = true;

  handle_ptr->get_stream().synchronize();
  RAFT_CHECK_CUDA(handle_ptr->get_stream());

  return graph_created;
}

template <typename i_t>
struct calc_blocks_per_item_t : public thrust::unary_function<i_t, i_t> {
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
struct heavy_vertex_meta_t : public thrust::unary_function<i_t, i_t> {
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

template <typename i_t>
void compact_bins(std::vector<i_t>& bins, i_t num_items)
{
  auto found_last_bin  = std::lower_bound(bins.begin(), bins.end(), num_items) - bins.begin();
  auto max_degree_cnst = 2 << (found_last_bin - 3);
  if (max_degree_cnst > 256) { found_last_bin = 10; }
  for (int i = 2; i <= found_last_bin - 1; ++i) {
    bins[i] = bins[1];
  }
  for (size_t i = found_last_bin; i < bins.size(); ++i) {
    bins[i] = num_items;
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
i_t create_heavy_item_block_segments(rmm::cuda_stream_view stream,
                                     rmm::device_uvector<i_t>& vertex_id,
                                     rmm::device_uvector<i_t>& pseudo_block_id,
                                     rmm::device_uvector<i_t>& item_block_segments,
                                     const i_t heavy_degree_cutoff,
                                     const std::vector<i_t>& bin_offsets,
                                     rmm::device_uvector<i_t> const& offsets)
{
  // TODO : assert that bin_offsets.back() == offsets.size() - 1
  auto heavy_id_beg   = bin_offsets[ceil_log_2(heavy_degree_cutoff)];
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
      thrust::make_counting_iterator<i_t>(item_block_segments.size()),
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
  // Total number of blocks that have to be launched
  return num_blocks;
}

template <typename i_t>
std::pair<bool, i_t> sub_warp_meta(rmm::cuda_stream_view stream,
                                   rmm::device_uvector<i_t>& d_warp_offsets,
                                   rmm::device_uvector<i_t>& d_warp_id_offsets,
                                   const std::vector<i_t>& bin_offsets,
                                   i_t w_t_r)
{
  // 1, 2, 4, 8, 16
  auto sub_warp_bin_count = 5;
  std::vector<i_t> warp_counts(sub_warp_bin_count);

  std::vector<i_t> warp_offsets(warp_counts.size() + 1);
  std::vector<i_t> warp_id_offsets(warp_counts.size() + 1);

  for (size_t i = 0; i < warp_id_offsets.size(); ++i) {
    warp_id_offsets[i] = bin_offsets[i + std::log2(w_t_r) + 1];
  }
  warp_id_offsets[0] = bin_offsets[0];

  i_t non_empty_bin_count = 0;
  for (size_t i = 0; i < warp_counts.size(); ++i) {
    warp_counts[i] =
      raft::ceildiv<i_t>((warp_id_offsets[i + 1] - warp_id_offsets[i]) * (1 << i), raft::WarpSize);
    if (warp_counts[i] != 0) { non_empty_bin_count++; }
  }

  warp_offsets[0] = 0;
  for (size_t i = 1; i < warp_offsets.size(); ++i) {
    warp_offsets[i] = warp_offsets[i - 1] + warp_counts[i - 1];
  }
  expand_device_copy(d_warp_offsets, warp_offsets, stream);
  expand_device_copy(d_warp_id_offsets, warp_id_offsets, stream);

  // If there is only 1 bin active, then there is no need to add logic to determine which warps work
  // on which bin
  return std::make_pair(non_empty_bin_count == 1, warp_offsets.back());
}

}  // namespace cuopt::linear_programming::detail
