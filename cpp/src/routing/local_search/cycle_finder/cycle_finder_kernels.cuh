/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../../routing_helpers.cuh"
#include "cycle_finder.hpp"
#include "routing/utilities/constants.hpp"
#include "routing/utilities/cuopt_utils.cuh"

namespace cuopt {
namespace routing {
namespace detail {

// since we have many kernels declare the shmem here
extern __shared__ double sh_buf[];

template <size_t max_routes>
__global__ void clamp_occupied(typename device_map_t<key_t<max_routes>, double>::view_t curr_path)
{
  if (curr_path.occupied[0] > curr_path.max_size) { curr_path.occupied[0] = curr_path.max_size; }
}

template <typename map_key_t, typename value_t>
__global__ void clear_map(typename device_map_t<map_key_t, value_t>::view_t d_valid_paths)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < d_valid_paths.max_available * d_valid_paths.max_level;
       i += blockDim.x * gridDim.x) {
    d_valid_paths.locks[i]        = 0;
    d_valid_paths.keys[i]         = map_key_t();
    d_valid_paths.values[i]       = std::numeric_limits<value_t>::max();
    d_valid_paths.predecessors[i] = std::numeric_limits<uint32_t>::max();
    if (i < d_valid_paths.max_level * d_valid_paths.max_size) {
      d_valid_paths.occupied_indices[i] = {-1, std::numeric_limits<int>::max()};
      d_valid_paths.size_per_head[i]    = 0;
    }
    if (i < d_valid_paths.max_level) {
      d_valid_paths.occupied[i]       = 0;
      d_valid_paths.stop_inserting[i] = 0;
    }
  }
}

template <typename map_key_t, typename value_t>
__global__ void test_empty(typename device_map_t<map_key_t, value_t>::view_t d_valid_paths)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < d_valid_paths.max_available;
       i += blockDim.x * gridDim.x) {
    cuopt_assert(d_valid_paths.locks[i] == 0, "");
    cuopt_assert(d_valid_paths.keys[i] == map_key_t(), "");
    cuopt_assert(d_valid_paths.values[i] == std::numeric_limits<value_t>::max(), "");
    cuopt_assert(d_valid_paths.predecessors[i] == std::numeric_limits<uint32_t>::max(), "");
    if (i < d_valid_paths.max_size) {
      int2 occ = {-1, std::numeric_limits<int>::max()};
      cuopt_assert(d_valid_paths.occupied_indices[i].x == occ.x, "");
      cuopt_assert(d_valid_paths.occupied_indices[i].y == occ.y, "");
      cuopt_assert(d_valid_paths.size_per_head[i] == 0, "");
    }
    if (i < 1) {
      cuopt_assert(d_valid_paths.occupied[i] == 0, "");
      cuopt_assert(d_valid_paths.stop_inserting[i] == 0, "");
    }
  }
}

template <typename i_t, typename f_t, size_t max_routes>
__global__ void init_cycle(typename ret_cycles_t<i_t, f_t>::view_t ret,
                           typename path_t<i_t, f_t, max_routes>::view_t d_best)
{
  ret.push_back(d_best.key_ptr[0].head);
}

template <typename i_t, typename f_t, size_t max_routes>
__global__ void close_cycle(typename ret_cycles_t<i_t, f_t>::view_t ret,
                            typename path_t<i_t, f_t, max_routes>::view_t d_best,
                            int curr_cycle_size)
{
  ret.curr_cycle_size = curr_cycle_size;
  ret.push_back(d_best.key_ptr[0].tail);
  ret.append_cycle(ret.curr_cycle_size);
}

template <typename i_t, typename f_t, size_t max_routes>
__global__ void extend_cycle(
  typename graph_t<i_t, f_t>::view_t const graph,
  typename device_map_t<key_t<max_routes>, double>::view_t const d_valid_paths,
  typename path_t<i_t, f_t, max_routes>::view_t d_best,
  typename ret_cycles_t<i_t, f_t>::view_t ret,
  int level,
  int curr_cycle_size)
{
  ret.curr_cycle_size = curr_cycle_size;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < d_valid_paths.max_available;
       i += blockDim.x * gridDim.x) {
    if (d_valid_paths.keys[i] == d_best.key_ptr[0]) {
      // acquire
      auto route_id = graph.route_ids[d_best.key_ptr[0].head];
      // Set las bit to false "cutting the head"
      d_best.key_ptr[0].label.set(route_id, false);
      // Set new head
      auto pred              = d_valid_paths.predecessors[i];
      d_best.key_ptr[0].head = pred;
      ret.push_back(d_best.key_ptr[0].head);
    }
  }
}

template <typename i_t, typename f_t, size_t max_routes>
__global__ void init_kernel(typename graph_t<i_t, f_t>::view_t const graph_view,
                            typename device_map_t<key_t<max_routes>, double>::view_t curr_map)
{
  int tail      = blockIdx.x;
  auto row_size = graph_view.row_sizes[tail];
  if (row_size == 0) return;
  int src_route = graph_view.route_ids[tail];
  int offset    = tail * max_graph_nodes_per_row;

  auto weights = raft::device_span<double>{sh_buf, (size_t)row_size};
  auto indices = raft::device_span<i_t>{(i_t*)&(weights.data()[row_size]), (size_t)row_size};

  // fill the shared array find the
  __shared__ uint32_t num_negative, row_offset;
  if (threadIdx.x == 0) {
    // the best case where all are negative
    if (graph_view.weights[offset + row_size - 1] < -EPSILON) {
      num_negative = row_size;
    }
    // if there is any negative, this will be changed
    else {
      num_negative = 0;
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
    auto global_idx  = offset + i;
    auto curr_weight = graph_view.weights[global_idx];
    indices[i]       = graph_view.indices[global_idx];
    weights[i]       = curr_weight;
    if (i != row_size - 1) {
      auto next_weight = graph_view.weights[global_idx + 1];
      // only one thread will enter here as weights are sorted
      if (curr_weight < -EPSILON && next_weight >= -EPSILON) { num_negative = i + 1; }
    }
  }
  __syncthreads();
  if (num_negative == 0) return;
  __syncthreads();
  // get the global offset to start writing the negative weights to the hashmap
  if (threadIdx.x == 0) {
    row_offset = atomicAdd(curr_map.occupied, num_negative);
    // if we exceed the size of the map reduce num_negative to only accommodate the first couple
    if (row_offset + num_negative > curr_map.max_size) {
      num_negative = max(0, (int)curr_map.max_size - (int)row_offset);
    }
  }
  __syncthreads();
  // copy each row to the internal memory of hashmap
  for (int col = threadIdx.x; col < num_negative; col += blockDim.x) {
    if (row_offset + col < curr_map.max_size) {
      int head      = indices[col];
      double weight = weights[col];
      int dst_route = graph_view.route_ids[head];

      key_t<max_routes> new_key(tail, head);
      new_key.label.set(src_route, true);
      new_key.label.set(dst_route, true);
      cuopt_assert(src_route != dst_route, "src_route cannot be same as dst_route");
      // directly copy to the internals of the hashmap, we are using it as an array here
      curr_map.keys[row_offset + col]             = new_key;
      curr_map.values[row_offset + col]           = weight;
      curr_map.predecessors[row_offset + col]     = tail;
      curr_map.occupied_indices[row_offset + col] = {(int)row_offset + col, head};
      atomicAdd(&curr_map.size_per_head[head], 1);
    } else {
      cuopt_assert(false, "cannot go out of bounds");
    }
  }
}

template <typename i_t, typename f_t, size_t max_routes, bool last_level>
__global__ void find_kernel(
  int level,
  typename graph_t<i_t, f_t>::view_t graph_view,
  typename device_map_t<key_t<max_routes>, double>::view_t const prev_map,
  typename device_map_t<key_t<max_routes>, double>::view_t curr_map,
  typename cycle_candidates_t<i_t, f_t, max_routes>::view_t cycle_candidates,
  bool depot_included)
{
  __shared__ int reduction_index;

  int head = blockIdx.x;
  // size per map is scanned, so it contains the offsets
  int occupied_begin      = prev_map.size_per_head[blockIdx.x];
  int occupied_end        = prev_map.size_per_head[blockIdx.x + 1];
  int n_of_items_per_head = (occupied_end - occupied_begin);
  if (n_of_items_per_head == 0) return;
  auto row_size = graph_view.row_sizes[head];
  if (row_size == 0) return;
  int offset = head * max_graph_nodes_per_row;
  cuopt_assert(row_size < graph_view.get_num_vertices(), "row_size should not exceed vertex size");
  auto weights = raft::device_span<double>{(double*)&sh_buf[2 * warp_size], (size_t)row_size};
  auto indices = raft::device_span<i_t>{(i_t*)&weights.data()[row_size], (size_t)row_size};

  for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
    auto global_idx = offset + i;
    indices[i]      = graph_view.indices[global_idx];
    weights[i]      = graph_view.weights[global_idx];
  }
  __syncthreads();

  key_t<max_routes> th_best_key = key_t<max_routes>();
  double th_best_cost           = std::numeric_limits<double>::max();

  // each thread tries to expland the path from the previous hashmaps occupied indices
  for (int i = threadIdx.x; i < n_of_items_per_head; i += blockDim.x) {
    cuopt_assert(i + occupied_begin < prev_map.max_available,
                 "index should be smaller than max_available");
    auto key_val_index = prev_map.occupied_indices[i + occupied_begin].x;
    cuopt_assert(key_val_index >= 0, "Occupied index should be positive");
    cuopt_assert(key_val_index < prev_map.keys.size(), "Wrong occupied index");
    key_t<max_routes> old_tmp = prev_map.keys[key_val_index];
    double value              = prev_map.values[key_val_index];
    cuopt_assert(old_tmp.head == head, "Wrong key head");
    cuopt_assert(prev_map.occupied_indices[i + occupied_begin].y == head, "Wrong key head");
    // Process old head edges
    for (int col = 0; col < row_size; ++col) {
      int dst       = indices[col];
      double weight = weights[col];
      cuopt_assert(dst < graph_view.get_num_vertices(), "dst should be smaller than num_vertices");
      int dst_route          = graph_view.route_ids[dst];
      double main_cost_after = weight + value;
      if (main_cost_after >= -EPSILON) {
        // early break because the weights are sorted
        break;
      } else {
        // A cycle was found
        if (old_tmp.tail == dst) {
          if (main_cost_after < th_best_cost) {
            th_best_cost = main_cost_after;
            th_best_key  = old_tmp;
            cuopt_assert(th_best_cost == std::numeric_limits<double>::max() ||
                           th_best_key.head != std::numeric_limits<uint32_t>::max(),
                         "Invalid cost and head");
          }
        } else if (!old_tmp.label.test(dst_route)) {
          if constexpr (!last_level) {
            key_t<max_routes> tmp = old_tmp;
            tmp.label.set(dst_route);
            tmp.head = dst;
            cuopt_assert(graph_view.route_ids[tmp.tail] != dst_route,
                         "Route id of tail cannot be equal to head!");
            cuopt_assert(graph_view.route_ids[head] != dst_route,
                         "Route id of tail cannot be equal to head!");
            curr_map.add(tmp, main_cost_after, head);
          }
        }
      }
    }
  }

  int thread_best_idx = threadIdx.x;
  // Record block best
  routing::detail::block_reduce_ranked(th_best_cost, thread_best_idx, sh_buf, &reduction_index);
  if (threadIdx.x == reduction_index) {
    cycle_candidates.keys[blockIdx.x]      = th_best_key;
    cycle_candidates.costs[blockIdx.x]     = sh_buf[0];
    cycle_candidates.level_vec[blockIdx.x] = level;
    cuopt_assert(!depot_included ||
                   ((sh_buf[0] == std::numeric_limits<double>::max()) || (th_best_key.head != 0)),
                 "");
  }
}

// this kernel accepts the best cycles in a sorted order
// it updates the best cycles if the route is not occupied
// it expands the cycle if a longer cycle with lower cost is found and the extended nodes are not
// forbidden
template <typename i_t, typename f_t, size_t max_routes>
__global__ void record_best_cycles(
  int total_size,
  typename graph_t<i_t, f_t>::view_t graph,
  typename cycle_candidates_t<i_t, f_t, max_routes>::view_t cycle_candidates,
  typename path_t<i_t, f_t, max_routes>::view_t best_cycles,
  i_t* sorted_key_indices)

{
  if (threadIdx.x == 0) {
    const auto empty_mask = device_bitset_t<max_routes>{};
    auto special_node_bit =
      device_bitset_t<max_routes>{}.set(graph.route_ids[graph.special_index], true);
    for (int i = 0; i < total_size; ++i) {
      double cost = cycle_candidates.costs[i];
      if (cost == std::numeric_limits<double>::max()) { break; }
      i_t key_idx   = sorted_key_indices[i];
      auto key      = cycle_candidates.keys[key_idx];
      auto level    = cycle_candidates.level_vec[key_idx];
      i_t n_cycles  = *best_cycles.n_cycles;
      auto all_mask = *best_cycles.all_mask;
      // test whether this cycle can be inserted to current set of cycles as independent
      if ((all_mask & key.label) == empty_mask) {
        // we can independently insert
        best_cycles.key_ptr[n_cycles]   = key;
        best_cycles.cost_ptr[n_cycles]  = cost;
        best_cycles.level_ptr[n_cycles] = level - 1;
        *best_cycles.all_mask           = all_mask | key.label;
        *best_cycles.n_cycles           = n_cycles + 1;
      }
      // if special node is marked, unmark it to have more relocate cycles
      best_cycles.all_mask->set(graph.route_ids[graph.special_index], false);
    }
    // check if all cycles are found
    if ((*best_cycles.all_mask) == (~special_node_bit)) { *best_cycles.all_found = true; }
  }
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
