/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <raft/core/handle.hpp>
#include <raft/util/cuda_dev_essentials.cuh>

#include <utility>

namespace cuopt::linear_programming::detail {

#define FIRST_THREAD (threadIdx.x == 0 && blockIdx.x == 0)
#define TH_ID_X      threadIdx.x + blockIdx.x* blockDim.x
#define GRID_STRIDE  gridDim.x* blockDim.x

// compute the grid and block size maximizing occupancy for a given kernel and TPB
inline std::pair<dim3, dim3> get_launch_dims_max_occupancy(void* kernel_address,
                                                           int TPB,
                                                           const raft::handle_t* handle_ptr)
{
  int num_blocks_per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel_address, TPB, 0);
  int n_blocks = handle_ptr->get_device_properties().multiProcessorCount * num_blocks_per_sm;
  dim3 dim_block(TPB, 1, 1);
  dim3 dim_grid(n_blocks, 1, 1);

  return {dim_grid, dim_block};
}

}  // namespace cuopt::linear_programming::detail
