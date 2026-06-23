/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/copy_helpers.hpp>
#include <utilities/macros.cuh>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/core/device_span.hpp>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <std::integral i_t,
          typename value_t,
          int warps_per_cta,
          typename InputIt,
          typename OutputIt>
__global__ void __launch_bounds__(warps_per_cta* raft::WarpSize)
  warp_per_cone_reduce_kernel(InputIt input,
                              raft::device_span<const i_t> small_cone_ids,
                              raft::device_span<const std::size_t> cone_offsets,
                              OutputIt output,
                              value_t init);

/**
 * Segmented-sum dispatcher for packed second-order cone vectors.
 *
 * Cone dimensions are fixed for a solve, so the constructor partitions cone
 * ids once by reduction strategy. Each call then reuses those partitions:
 * small cones use one warp per cone, medium cones use CUB DeviceSegmentedReduce,
 * and large cones use CUB DeviceReduce one cone at a time. The object owns the
 * CUB workspace for those medium/large paths. Call `prepare_workspace` once
 * before using a CUB-backed path.
 */
template <std::integral i_t, i_t warp_cone_dim = 64, i_t large_cone_cutoff = 32768>
struct segmented_sum_t {
  static_assert(warp_cone_dim > 0);
  static_assert(large_cone_cutoff > warp_cone_dim);

  raft::device_span<const std::size_t> cone_offsets;
  rmm::device_uvector<i_t> small_cone_ids;   // cone dimension <= warp_cone_dim
  rmm::device_uvector<i_t> medium_cone_ids;  // warp_cone_dim < cone dimension <= large_cone_cutoff

  std::vector<std::size_t> large_cone_offsets;
  std::vector<i_t> large_cone_ids;
  std::vector<i_t> large_cone_dimensions;

  // Maximum CUB temporary storage needed by prepared medium/large reductions.
  std::size_t cub_workspace_bytes = 0;
  rmm::device_buffer cub_workspace;

 private:
  template <typename value_t>
  void prepare_workspace_for_type(rmm::cuda_stream_view stream)
  {
    auto input  = thrust::make_constant_iterator(value_t{});
    auto output = thrust::make_discard_iterator();

    if (!medium_cone_ids.is_empty()) {
      const auto medium_begin_offsets =
        thrust::make_permutation_iterator(cone_offsets.data(), medium_cone_ids.begin());
      const auto medium_end_offsets =
        thrust::make_permutation_iterator(cone_offsets.data() + 1, medium_cone_ids.begin());

      std::size_t temp_storage_bytes = 0;
      RAFT_CUDA_TRY(cub::DeviceSegmentedReduce::Sum(nullptr,
                                                    temp_storage_bytes,
                                                    input,
                                                    output,
                                                    medium_cone_ids.size(),
                                                    medium_begin_offsets,
                                                    medium_end_offsets,
                                                    stream.value()));
      cub_workspace_bytes = std::max(cub_workspace_bytes, temp_storage_bytes);
    }

    for (std::size_t i = 0; i < large_cone_ids.size(); ++i) {
      std::size_t temp_storage_bytes = 0;
      RAFT_CUDA_TRY(cub::DeviceReduce::Sum(nullptr,
                                           temp_storage_bytes,
                                           input + large_cone_offsets[i],
                                           output + large_cone_ids[i],
                                           large_cone_dimensions[i],
                                           stream.value()));
      cub_workspace_bytes = std::max(cub_workspace_bytes, temp_storage_bytes);
    }

    if (cub_workspace.size() < cub_workspace_bytes) {
      cub_workspace.resize(cub_workspace_bytes, stream);
    }
  }

 public:
  template <typename value_t, typename... rest_t>
  void prepare_workspace(rmm::cuda_stream_view stream)
  {
    prepare_workspace_for_type<value_t>(stream);
    (prepare_workspace_for_type<rest_t>(stream), ...);
  }

  template <typename value_t, typename InputIt, typename OutputIt, int warps_per_cta = 8>
  void operator()(InputIt input, OutputIt output, value_t init, rmm::cuda_stream_view stream)
  {
    if (!small_cone_ids.is_empty()) {
      // Each warp reduces one small cone. `warps_per_cta` only controls how
      // many independent cone reductions are packed into one CTA; the default
      // of 8 gives a conventional 256-thread block.
      const auto n_small = small_cone_ids.size();
      const auto grid    = (n_small + warps_per_cta - 1) / warps_per_cta;
      warp_per_cone_reduce_kernel<i_t, value_t, warps_per_cta>
        <<<grid, warps_per_cta * raft::WarpSize, 0, stream.value()>>>(
          input, cuopt::make_span(small_cone_ids), cone_offsets, output, init);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    if (!medium_cone_ids.is_empty()) {
      cuopt_assert(cub_workspace_bytes > 0 && cub_workspace.size() >= cub_workspace_bytes,
                   "segmented_sum_t::prepare_workspace must be called before reducing medium or "
                   "large cones");

      const auto medium_output = thrust::make_permutation_iterator(output, medium_cone_ids.begin());
      const auto medium_begin_offsets =
        thrust::make_permutation_iterator(cone_offsets.data(), medium_cone_ids.begin());
      const auto medium_end_offsets =
        thrust::make_permutation_iterator(cone_offsets.data() + 1, medium_cone_ids.begin());

      std::size_t temp_storage_bytes = cub_workspace_bytes;
      RAFT_CUDA_TRY(cub::DeviceSegmentedReduce::Sum(cub_workspace.data(),
                                                    temp_storage_bytes,
                                                    input,
                                                    medium_output,
                                                    medium_cone_ids.size(),
                                                    medium_begin_offsets,
                                                    medium_end_offsets,
                                                    stream.value()));
    }

    if (!large_cone_ids.empty()) {
      cuopt_assert(cub_workspace_bytes > 0 && cub_workspace.size() >= cub_workspace_bytes,
                   "segmented_sum_t::prepare_workspace must be called before reducing medium or "
                   "large cones");

      for (std::size_t i = 0; i < large_cone_ids.size(); ++i) {
        std::size_t temp_storage_bytes = cub_workspace_bytes;
        RAFT_CUDA_TRY(cub::DeviceReduce::Sum(cub_workspace.data(),
                                             temp_storage_bytes,
                                             input + large_cone_offsets[i],
                                             output + large_cone_ids[i],
                                             large_cone_dimensions[i],
                                             stream.value()));
      }
    }
  }

  template <std::floating_point f_t, typename InputIt>
  void operator()(InputIt input, raft::device_span<f_t> output, rmm::cuda_stream_view stream)
  {
    operator()(input, output.data(), f_t{0}, stream);
  }

  segmented_sum_t(std::span<const i_t> cone_dimensions_host,
                  raft::device_span<const std::size_t> cone_offsets_in,
                  rmm::cuda_stream_view stream)
    : cone_offsets(cone_offsets_in),
      small_cone_ids(0, stream),
      medium_cone_ids(0, stream),
      cub_workspace(0, stream)
  {
    std::vector<i_t> small_cone_ids_host;
    std::vector<i_t> medium_cone_ids_host;

    std::size_t cone_offset = 0;
    i_t cone                = 0;
    for (const auto cone_dimension : cone_dimensions_host) {
      if (cone_dimension <= warp_cone_dim) {
        small_cone_ids_host.push_back(cone);
      } else if (cone_dimension <= large_cone_cutoff) {
        medium_cone_ids_host.push_back(cone);
      } else {
        large_cone_ids.push_back(cone);
        large_cone_offsets.push_back(cone_offset);
        large_cone_dimensions.push_back(cone_dimension);
      }
      cone_offset += cone_dimension;
      ++cone;
    }

    bool need_sync = false;
    if (!small_cone_ids_host.empty()) {
      cuopt::device_copy(small_cone_ids, small_cone_ids_host, stream);
      need_sync = true;
    }
    if (!medium_cone_ids_host.empty()) {
      cuopt::device_copy(medium_cone_ids, medium_cone_ids_host, stream);
      need_sync = true;
    }
    if (need_sync) { stream.synchronize(); }
  }
};

template <std::integral i_t,
          typename value_t,
          int warps_per_cta,
          typename InputIt,
          typename OutputIt>
__global__ void __launch_bounds__(warps_per_cta* raft::WarpSize)
  warp_per_cone_reduce_kernel(InputIt input,
                              raft::device_span<const i_t> small_cone_ids,
                              raft::device_span<const std::size_t> cone_offsets,
                              OutputIt output,
                              value_t init)
{
  static_assert(warps_per_cta > 0);
  static_assert(warps_per_cta * raft::WarpSize <= 1024);

  using warp_reduce_t = cub::WarpReduce<value_t, raft::WarpSize>;
  __shared__ typename warp_reduce_t::TempStorage temp_storage[warps_per_cta];

  const auto lane_id  = raft::laneId();
  const auto warp_idx = threadIdx.x / raft::WarpSize;
  const auto slot     = blockIdx.x * warps_per_cta + warp_idx;
  if (slot >= small_cone_ids.size()) { return; }

  const auto cone = small_cone_ids[slot];
  const auto off  = cone_offsets[cone];
  const auto dim  = cone_offsets[cone + 1] - off;

  auto sum = init;
  for (std::size_t i = lane_id; i < dim; i += raft::WarpSize) {
    sum = sum + input[off + i];
  }

  sum = warp_reduce_t(temp_storage[warp_idx]).Sum(sum);
  if (lane_id == 0) { output[cone] = sum; }
}

}  // namespace cuopt::linear_programming::dual_simplex
