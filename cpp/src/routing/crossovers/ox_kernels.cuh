/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../local_search/cycle_finder/cycle_graph.hpp"
#include "../solution/solution.cuh"
#include "routing/utilities/cuopt_utils.cuh"

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
__global__ void transpose_graph_kernel(typename ox_graph_t<i_t, f_t>::view_t graph,
                                       typename ox_graph_t<i_t, f_t>::view_t transpose_graph,
                                       i_t max_route_len)
{
  auto bucket        = blockIdx.x / graph.get_num_vertices();
  auto vertex_id     = blockIdx.x % graph.get_num_vertices();
  auto global_offset = bucket * graph.get_num_vertices() * graph.get_max_nodes_per_row() +
                       vertex_id * graph.get_max_nodes_per_row();
  auto row_len = graph.row_sizes[bucket * graph.get_num_vertices() + vertex_id];
  cuopt_assert(graph.get_max_nodes_per_row() == transpose_graph.get_max_nodes_per_row(),
               "Mismatch max graph nodes");

  for (i_t i = threadIdx.x; i < row_len; i += blockDim.x) {
    auto edge   = graph.indices[global_offset + i];
    auto weight = graph.weights[global_offset + i];
    auto veh    = graph.buckets[global_offset + i];
    auto transpose_global_offset =
      edge * transpose_graph.n_buckets * transpose_graph.get_max_nodes_per_row();
    auto transpose_edge_idx = atomicAdd(&transpose_graph.row_sizes[edge], 1);
    transpose_graph.indices[transpose_global_offset + transpose_edge_idx] = vertex_id;
    transpose_graph.weights[transpose_global_offset + transpose_edge_idx] = weight;
    transpose_graph.buckets[transpose_global_offset + transpose_edge_idx] = veh;
  }
}

template <typename i_t, typename f_t>
__global__ void bellman_ford_init(raft::device_span<f_t> path_cost,
                                  raft::device_span<i_t> predecessor,
                                  raft::device_span<i_t> predecessor_vehicle)
{
  if (threadIdx.x == 0) {
    path_cost[0]           = 0.;
    predecessor[0]         = 0;
    predecessor_vehicle[0] = 0;
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void bellman_ford_kernel(const typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                                    typename ox_graph_t<i_t, f_t>::view_t transpose_graph,
                                    raft::device_span<double> path_cost,
                                    raft::device_span<i_t> predecessor,
                                    raft::device_span<i_t> predecessor_vehicle,
                                    raft::device_span<i_t> vehicle_availability,
                                    i_t row_size,
                                    int i,
                                    bool run_heuristic)
{
  __shared__ i_t reduction_index;
  __shared__ double reduction_buf[2 * warp_size];

  i_t vertex         = blockIdx.x + i;
  i_t previous_level = i - 1;

  i_t best_thread_edge      = -1;
  double best_thread_update = std::numeric_limits<double>::max();
  i_t best_thread_veh       = -1;

  auto n_edges = transpose_graph.row_sizes[vertex];

  auto transpose_offset =
    vertex * transpose_graph.n_buckets * transpose_graph.get_max_nodes_per_row();

  int veh_counter;

  // n_buckets * n_edges threads
  for (i_t tid = threadIdx.x; tid < n_edges; tid += blockDim.x) {
    auto edge         = transpose_graph.indices[transpose_offset + tid];
    auto path_cost_ij = path_cost[previous_level * row_size + edge];
    auto veh          = transpose_graph.buckets[transpose_offset + tid];
    if (path_cost_ij == std::numeric_limits<double>::max()) { continue; }

    if (run_heuristic) {
      veh_counter = vehicle_availability[veh];
      int tmp     = edge;
      int i_tmp   = previous_level;
      while (tmp > 0) {
        auto pred_veh = predecessor_vehicle[i_tmp * row_size + tmp];
        if (pred_veh == veh) { --veh_counter; }
        tmp = predecessor[i_tmp * row_size + tmp];
        --i_tmp;
      }
    }
    auto weight = transpose_graph.weights[transpose_offset + tid];

    if (!run_heuristic || veh_counter > 0) {
      if (path_cost_ij + weight < best_thread_update) {
        best_thread_edge   = edge;
        best_thread_update = path_cost_ij + weight;
        best_thread_veh    = veh;
      }
    }
  }

  i_t curr_thread = threadIdx.x;
  block_reduce_ranked(best_thread_update, curr_thread, reduction_buf, &reduction_index);
  if (reduction_index == threadIdx.x) {
    if (reduction_buf[0] != std::numeric_limits<double>::max()) {
      path_cost[i * row_size + vertex]           = reduction_buf[0];
      predecessor[i * row_size + vertex]         = best_thread_edge;
      predecessor_vehicle[i * row_size + vertex] = best_thread_veh;
    }
  }
}

/**
 * @brief This method fills the graph edges
 *
 * @tparam i_t
 * @tparam f_t
 * @tparam REQUEST
 * @param solution
 * @param graph
 * @param offspring
 * @param max_route_len
 * @param weights
 * @return
 */
template <typename i_t, typename f_t, request_t REQUEST>
__global__ void calculate_edge_costs_kernel(
  const typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  typename ox_graph_t<i_t, f_t>::view_t graph,
  raft::device_span<i_t> offspring,
  raft::device_span<i_t> vehicle_id_per_bucket,
  i_t max_route_len,
  const infeasible_cost_t weights)
{
  auto bucket = blockIdx.x / (offspring.size() - 1);
  auto i      = blockIdx.x % (offspring.size() - 1);

  // TODO: here we should find optimal DEPOT for each path. This will require storing tuples
  // instead of pairs that contain start_depot, end_depot
  int depot       = 0;
  auto vehicle_id = vehicle_id_per_bucket[bucket];

  // FIXME: get_vehicle_info(bucket)
  auto vehicle_info = solution.problem.fleet_info.get_vehicle_info(vehicle_id);
  extern __shared__ i_t shmem[];
  raft::device_span<double> row_value((double*)shmem, max_route_len);
  raft::device_span<i_t> row_edge((i_t*)&row_value.data()[row_value.size()], max_route_len);

  __shared__ i_t row_n_edges;

  if (threadIdx.x == 0) {
    row_n_edges = 0;
    atomicExch(&offspring[0], depot);
  }
  __syncthreads();

  // Each block handles a row
  auto global_offset = bucket * graph.get_num_vertices() * graph.get_max_nodes_per_row() +
                       i * graph.get_max_nodes_per_row();
  cuopt_assert(graph.get_max_nodes_per_row() == max_route_len, "Wrong max graph nodes");

  if (threadIdx.x == 0) {
    const auto& dimensions_info = solution.problem.dimensions_info;

    double cost = 0.;
    double optimal_vehicle_fixed_cost =
      dimensions_info.has_dimension(dim_t::VEHICLE_FIXED_COST) ? vehicle_info.fixed_cost : 0.;

    node_t<i_t, f_t, REQUEST> helper_nodes[3] = {node_t<i_t, f_t, REQUEST>(dimensions_info),
                                                 node_t<i_t, f_t, REQUEST>(dimensions_info),
                                                 node_t<i_t, f_t, REQUEST>(dimensions_info)};

    // FIXME: get_start_depot_node_info(bucket)
    auto start_depot_node_info  = solution.problem.get_start_depot_node_info(vehicle_id);
    auto return_depot_node_info = solution.problem.get_return_depot_node_info(vehicle_id);
    helper_nodes[2]             = create_depot_node<i_t, f_t, REQUEST>(
      solution.problem, start_depot_node_info, return_depot_node_info, vehicle_id);
    helper_nodes[0] = create_node<i_t, f_t, REQUEST>(solution.problem, offspring[i + 1]);
    helper_nodes[2].calculate_forward_all(helper_nodes[0], vehicle_info);
    bool b = 0;

    // Calculate forward all loop has to be executed by a single thread so we keep this
    // sequential
    for (int j = i + 1; j < offspring.size(); ++j) {
      if (j - i > max_route_len) { break; }

      cost = node_t<i_t, f_t, REQUEST>::cost_combine(
        helper_nodes[b], helper_nodes[2], vehicle_info, true, weights, d_zero_cost, d_zero_cost);

      row_value[row_n_edges] = optimal_vehicle_fixed_cost + cost;
      row_edge[row_n_edges]  = j;
      ++row_n_edges;

      if (j + 1 == offspring.size()) { break; }

      helper_nodes[!b] = create_node<i_t, f_t, REQUEST>(solution.problem, offspring[j + 1]);
      helper_nodes[b].calculate_forward_all(helper_nodes[!b], vehicle_info);
      b = !b;
    }

    graph.row_sizes[bucket * graph.get_num_vertices() + i] = row_n_edges;
  }
  __syncthreads();

  for (i_t i = threadIdx.x; i < row_n_edges; i += blockDim.x) {
    graph.weights[global_offset + i] = row_value[i];
    graph.indices[global_offset + i] = row_edge[i];
    graph.buckets[global_offset + i] = bucket;
  }
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
