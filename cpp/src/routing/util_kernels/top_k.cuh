/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <routing/local_search/cycle_finder/cycle_finder.hpp>
#include <utilities/macros.cuh>

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_shuffle.cuh>
#include <cub/block/block_store.cuh>
#include <cub/cub.cuh>

#include <cuda/std/type_traits>

namespace cuopt {
namespace routing {
namespace detail {

template <typename cand_t>
struct decomposer_t {
  __device__ ::cuda::std::tuple<double&> operator()(cand_t& key) const
  {
    return {key.selection_delta};
  }
};

template <typename output_t>
constexpr auto get_default()
{
  if constexpr (::cuda::std::is_same_v<output_t, double>) {
    return std::numeric_limits<double>::max();
  } else {
    return output_t::init_data();
  }
}

template <typename i_t,
          typename output_t,
          int k,
          int TPB,
          bool write_diagonal,
          int items_per_thread = 16>
DI int top_k_indices_per_row(i_t row_id,
                             raft::device_span<const output_t> row_costs,
                             raft::device_span<output_t> out_costs,
                             raft::device_span<i_t> out_indices)
{
  static_assert(TPB * items_per_thread == k * 2, "Invalid launch configration");
  cuopt_assert(out_costs.size() == k, "Unexpected row size");
  cuopt_assert(out_indices.size() == k, "Unexpected row size");

  constexpr int loads_per_thread         = items_per_thread / 2;
  constexpr int load_items_per_iteration = loads_per_thread * TPB;
  constexpr int sort_items_per_iteration = items_per_thread * TPB;

  // TODO : shared memory bank size 8 bytes
  using block_sort          = cub::BlockRadixSort<output_t, TPB, items_per_thread, int>;
  using temp_sort_storage_t = typename block_sort::TempStorage;

  using block_load_sort          = cub::BlockLoad<output_t, TPB, items_per_thread>;
  using temp_load_sort_storage_t = typename block_load_sort::TempStorage;

  using block_load          = cub::BlockLoad<output_t, TPB, loads_per_thread>;
  using temp_load_storage_t = typename block_load::TempStorage;

  using block_shuffle       = cub::BlockShuffle<output_t, TPB>;
  using temp_shfl_storage_t = typename block_shuffle::TempStorage;

  using block_store_key    = cub::BlockStore<output_t, TPB, items_per_thread>;
  using temp_key_storage_t = typename block_store_key::TempStorage;

  using block_store_val    = cub::BlockStore<int, TPB, items_per_thread>;
  using temp_val_storage_t = typename block_store_val::TempStorage;

  using block_reduce          = cub::BlockReduce<int, TPB>;
  using temp_reduce_storage_t = typename block_reduce::TempStorage;

  __shared__ union {
    temp_load_storage_t load;
    temp_load_sort_storage_t load_sort;
    temp_sort_storage_t sort;
    temp_shfl_storage_t shfl;
    temp_key_storage_t store_key;
    temp_val_storage_t store_val;
    temp_reduce_storage_t reduce;
  } temp_storage;

  output_t sort_cost[items_per_thread];
  int col_id[items_per_thread];

  i_t num_cols      = row_costs.size();
  auto iter_per_row = raft::ceildiv<i_t>(num_cols, load_items_per_iteration);
  i_t col_offset    = 0;

  if (iter_per_row == 0) { return 0; }

  // The first step tries to load a maximum of sort_items_per_iteration
  // which is set to 2*k
  auto load_len = min(sort_items_per_iteration, num_cols);

  block_load_sort(temp_storage.load_sort)
    .Load(row_costs.data() + col_offset, sort_cost, load_len, get_default<output_t>());

  // If the column index matches the row id then the element we are dealing with
  // lies on the diagonal of the matrix. We want these costs to be double max()
  if constexpr (write_diagonal) {
    if (threadIdx.x == ((row_id - col_offset) / items_per_thread)) {
      sort_cost[(row_id - col_offset) % items_per_thread] = get_default<output_t>();
    }
  }
  // Populate column ids based on thread id
  for (int i = 0; i < items_per_thread; ++i) {
    col_id[i] = col_offset + items_per_thread * threadIdx.x + i;
  }
  col_offset += load_len;

  // while col_offset is not needed in the code above it has been kept to explain
  // the logic of the code below

  __syncthreads();

  // Sort the loaded data

  if constexpr (::cuda::std::is_same_v<output_t, double>) {
    block_sort(temp_storage.sort).Sort(sort_cost, col_id);
  } else {
    block_sort(temp_storage.sort).Sort(sort_cost, col_id, decomposer_t<output_t>{});
  }

  __syncthreads();

  // This section is active when there are more columns to deal with after the
  // first stage.
  for (int iter = 2; iter < iter_per_row; ++iter) {
    output_t load_cost[loads_per_thread];
    // This step tries to load a maximum of load_items_per_iteration
    // which is set to k
    load_len = min(load_items_per_iteration, num_cols - col_offset);
    block_load(temp_storage.load)
      .Load(row_costs.data() + col_offset, load_cost, load_len, get_default<output_t>());

    // If the column index matches the row id then the element we are dealing with
    // lies on the diagonal of the matrix. We want these costs to be double max()
    if constexpr (write_diagonal) {
      if (threadIdx.x == ((row_id - col_offset) / loads_per_thread)) {
        load_cost[(row_id - col_offset) % loads_per_thread] = get_default<output_t>();
      }
    }
    __syncthreads();

    // overwrite undesirable elements

    // every thread (out of 128) maintains 16 key value pairs
    //(sort_cost and col_id). After block sort is called over these elements,
    // we over write the data of the last 64 threads with new data from the row.
    if (threadIdx.x >= blockDim.x / 2) {
      for (int i = 0; i < loads_per_thread; ++i) {
        sort_cost[i + loads_per_thread] = load_cost[i];
        col_id[i + loads_per_thread]    = col_offset + loads_per_thread * threadIdx.x + i;
        col_id[i] = col_offset + loads_per_thread * threadIdx.x + i - load_items_per_iteration / 2;
      }
    }

    __syncthreads();
    // get items loaded from first half of threads
    for (int i = 0; i < loads_per_thread; ++i) {
      block_shuffle(temp_storage.shfl).Offset(load_cost[i], sort_cost[i], -(blockDim.x / 2));
      __syncthreads();
    }

    // Sort the loaded data
    if constexpr (::cuda::std::is_same_v<output_t, double>) {
      block_sort(temp_storage.sort).Sort(sort_cost, col_id);
    } else {
      block_sort(temp_storage.sort).Sort(sort_cost, col_id, decomposer_t<output_t>{});
    }
    __syncthreads();

    col_offset += load_items_per_iteration;
  }

  // After the thread block iterates over the entire row, it ends up with the key
  // value pairs stored in sort_cost and col_id for which sort_cost is the least
  // across the entire row.

  // block_store
  int valid_cost_count = 0;
  for (int i = 0; i < items_per_thread; ++i) {
    valid_cost_count += (sort_cost[i] != get_default<output_t>());
  }
  block_store_key(temp_storage.store_key).Store(out_costs.data(), sort_cost, out_costs.size());
  __syncthreads();

  block_store_val(temp_storage.store_val).Store(out_indices.data(), col_id, out_indices.size());

  __syncthreads();

  // we can store a maximum of k elements in the graph.
  return min(block_reduce(temp_storage.reduce).Sum(valid_cost_count),
             static_cast<int>(out_costs.size()));
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
