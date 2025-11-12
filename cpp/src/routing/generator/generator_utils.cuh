/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <routing/arc_value.hpp>

#include <raft/random/rng.cuh>

#define CUDA_MAX_BLOCKS_2D 256

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
__global__ void build_cost_matrix(
  f_t* mat, f_t const* x_pos, f_t const* y_pos, i_t size, bool asymmetric, f_t asymmetry_scalar)
{
  for (i_t i = blockIdx.y * blockDim.y + threadIdx.y; i < size; i += blockDim.y * gridDim.y) {
    for (i_t j = blockIdx.x * blockDim.x + threadIdx.x; j < size; j += blockDim.x * gridDim.x) {
      if (i <= j) {
        f_t x1            = x_pos[i];
        f_t y1            = y_pos[i];
        f_t x2            = x_pos[j];
        f_t y2            = y_pos[j];
        f_t val           = euclidean_distance(x1, y1, x2, y2);
        mat[i * size + j] = val;
        mat[j * size + i] = asymmetric ? val * (1.0 - asymmetry_scalar) : val;
      }
    }
  }
}

template <typename i_t, typename f_t>
__global__ void fill_time_windows(
  f_t* mat, i_t* earliest_time, i_t* latest_time, f_t const tw_tightness, i_t row_size)
{
  raft::random::PCGenerator block_rng(0, uint64_t(blockIdx.x), 0);
  int node   = blockIdx.x;
  auto depot = 0;
  if (node == depot) {
    earliest_time[depot] = 0;
    latest_time[depot]   = INT32_MAX;
    return;
  }
  __shared__ f_t sh_time_node_furthest_node;
  __shared__ i_t sh_furthest_node;
  __shared__ i_t sh_lock;
  if (threadIdx.x == 0) {
    sh_time_node_furthest_node = 0.f;
    sh_furthest_node           = 0;
    sh_lock                    = 0;
  }
  __syncthreads();

  f_t increasing_coef = 1.f + tw_tightness;
  f_t decreasing_coef = 1.f - tw_tightness;
  f_t time_depot_node = mat[depot * row_size + node];
  for (i_t i = threadIdx.x; i < row_size; i += blockDim.x) {
    if (i == depot) { continue; }
    f_t time = mat[node * row_size + i];
    if (sh_time_node_furthest_node < time) {
      while (atomicCAS_block(&sh_lock, 0, 1)) {}
      __threadfence_block();
      if (sh_time_node_furthest_node < time) {
        sh_time_node_furthest_node = time;
        sh_furthest_node           = i;
      }
      __threadfence_block();
      sh_lock = 0;
    }
  }
  __syncthreads();

  // Earliest time we can serve customer is Depot -> node time
  // Latest time we can serve customer is Depot -> node -> furthest node -> Depot
  // Add randomness and coefficients
  if (threadIdx.x == 0) {
    f_t time_furthest_node_depot = mat[sh_furthest_node * row_size + depot];
    i_t latest          = time_depot_node + sh_time_node_furthest_node + time_furthest_node_depot;
    auto randint        = block_rng.next_u32() % (latest / 2);
    i_t earliest        = latest - (randint + latest / 2);
    earliest_time[node] = earliest * increasing_coef;
    randint             = block_rng.next_u32() % ((latest - earliest_time[node]) / 2 + 1);
    latest_time[node]   = max(earliest_time[node] + 1, (i_t)((latest - randint) * decreasing_coef));
    cuopt_assert(earliest_time[node] < latest_time[node],
                 "Earliest time should be less than latest time");
  }
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
