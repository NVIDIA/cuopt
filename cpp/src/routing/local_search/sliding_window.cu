/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <utilities/cuda_helpers.cuh>
#include "../solution/solution.cuh"
#include "../utilities/cuopt_utils.cuh"
#include "local_search.cuh"
#include "permutation_helper.cuh"

#include <thrust/fill.h>
#include <thrust/remove.h>

namespace cuopt {
namespace routing {
namespace detail {

constexpr int max_range_size = 200;

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          int window_size,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
__device__ std::pair<i_t, i_t> get_start_end_indices(
  raft::device_span<node_t<i_t, f_t, REQUEST>> nodes,
  i_t intra_idx,
  const typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
  i_t route_length)
{
  // clamp the range to +- 100 to intra_idx to save from run time
  i_t start_idx               = max(intra_idx - max_range_size / 2, 0);
  i_t end_idx                 = min(intra_idx + max_range_size / 2, route_length);
  std::pair<i_t, i_t> indices = {start_idx, end_idx};
  return indices;
}

// Returns the min and max indices for the browsing of the route with the sliding window
// Browsing cannot start before a certain pickup and after a certain delivery
template <typename i_t,
          typename f_t,
          request_t REQUEST,
          int window_size,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
__device__ std::pair<i_t, i_t> get_start_end_indices(
  raft::device_span<node_t<i_t, f_t, REQUEST>> nodes,
  i_t intra_idx,
  const typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
  i_t route_length)
{
  std::pair<i_t, i_t> indices = {0, route_length};
  for (int i = 0; i < window_size; ++i) {
    const auto& brother_info = nodes[i].request.brother_info;
    const auto [brother_route_id, brother_intra_idx] =
      solution.route_node_map.get_route_id_and_intra_idx(brother_info);
    if (!nodes[i].request.info.is_pickup()) {
      indices.first = max(indices.first, brother_intra_idx);
    } else {
      indices.second = min(indices.second, brother_intra_idx);
    }
  }
  i_t curr_range_size = indices.second - indices.first;
  if (curr_range_size > max_range_size) {
    i_t diff = curr_range_size - max_range_size;
    // clamp the range to +- 100 to intra_idx to save from run time
    indices.first  = indices.first + diff / 2;
    indices.second = indices.second - diff / 2;
  }

  return indices;
}

template <typename i_t, typename f_t, request_t REQUEST, int window_size>
__device__ void try_permutations(
  found_sliding_solution_t<i_t>& found_sliding_solution,
  i_t permutation_index,
  const i_t permutation_array[window_size],
  i_t window_start_idx,
  const typename route_t<i_t, f_t, REQUEST>::view_t& s_route,
  const typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
  const typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
  double excess_limit)
{
  static_assert(window_size >= min_permutations<REQUEST>());

  // Filtration phase

  // Discard cases that overstep on the end of the route
  // There are handled by another thread block anyway
  if (window_start_idx + window_size > s_route.get_num_nodes()) { return; }

  // Initialize window

  // Get window based on premutation index

  const auto& dimensions_info                     = solution.problem.dimensions_info;
  constexpr size_t nodes_size                     = max_permutation_intra + 1;
  node_t<i_t, f_t, REQUEST> arr_nodes[nodes_size] = {node_t<i_t, f_t, REQUEST>(dimensions_info),
                                                     node_t<i_t, f_t, REQUEST>(dimensions_info),
                                                     node_t<i_t, f_t, REQUEST>(dimensions_info),
                                                     node_t<i_t, f_t, REQUEST>(dimensions_info),
                                                     node_t<i_t, f_t, REQUEST>(dimensions_info),
                                                     node_t<i_t, f_t, REQUEST>(dimensions_info)};
  auto nodes = raft::device_span<node_t<i_t, f_t, REQUEST>>(arr_nodes, window_size);
  for (int i = 0; i < window_size; ++i) {
    nodes[permutation_array[i]] = s_route.get_node(window_start_idx + i);
  }

  // Discard permutations creating an invalid route
  // (If route contains delivery before pickup pair)
  if constexpr (REQUEST == request_t::PDP) {
    for (int i = 1; i < window_size; ++i) {
      if (nodes[i].request.is_pickup()) {
        const int delivery_id = nodes[i].request.brother_info.node();
        for (int j = i - 1; j >= 0; --j)
          if (nodes[j].request.info.node() == delivery_id) return;
      }
    }
  }

  // Try inserting the window along the whole route
  //
  // For PDP we don't need to go from 0 until route_length:
  // Start : highest possible pickup pair from delivery in window (can't insert a delivery before
  // a pickup) End : lowest possible delivery pair from pickup in window (can't insert a pickup
  // after a delivery)

  const auto [start_idx, end_idx] = get_start_end_indices<i_t, f_t, REQUEST, window_size>(
    nodes, window_start_idx, solution, s_route.get_num_nodes());

  // Handle the three cases, first inplace, then left part : shifting the window left; then right
  // part : shifting the window right

  // -- In place --

  // Abtritrally choose to propagate forward data (could have been on backward starting from the
  // end)

  if (!forward_fragment_update<i_t, f_t, REQUEST>(s_route.get_node(window_start_idx - 1),
                                                  s_route,
                                                  nodes.data(),
                                                  window_size,
                                                  move_candidates.weights,
                                                  excess_limit)) {
    return;
  }

  // Just for time filtration
  if (dimensions_info.has_dimension(dim_t::TIME)) {
    auto next_node = s_route.get_node(window_start_idx + window_size);

    loop_over_constrained_dimensions(dimensions_info, [&] __device__(auto I) {
      get_dimension_of<I>(nodes[window_size - 1])
        .calculate_forward(
          get_dimension_of<I>(next_node),
          get_arc_of_dimension<i_t, f_t, I>(
            nodes[window_size - 1].request.info, next_node.request.info, s_route.vehicle_info()));
    });

    bool valid = true;
    loop_over_constrained_dimensions(dimensions_info, [&] __device__(auto I) {
      valid &= get_dimension_of<I>(next_node).forward_feasible(
        s_route.vehicle_info(), move_candidates.weights[I], excess_limit);
    });
    if (!valid) { return; }
  }

  auto next_node = s_route.get_node(window_start_idx + window_size);

  // Do a cost combine on right part
  // Its backward info is valid and forward has just been updated
  // Store the candidate if the different is better

  const double delta = node_t<i_t, f_t, REQUEST>::cost_combine(nodes[window_size - 1],
                                                               next_node,
                                                               s_route.vehicle_info(),
                                                               move_candidates.include_objective,
                                                               move_candidates.weights,
                                                               s_route.get_objective_cost(),
                                                               s_route.get_infeasibility_cost());
  if (delta < -EPSILON) {
    const double selection_delta =
      node_t<i_t, f_t, REQUEST>::cost_combine(nodes[window_size - 1],
                                              next_node,
                                              s_route.vehicle_info(),
                                              move_candidates.include_objective,
                                              move_candidates.selection_weights,
                                              s_route.get_objective_cost(),
                                              s_route.get_infeasibility_cost());
    if (delta < found_sliding_solution.delta) {
      found_sliding_solution.delta                 = selection_delta;
      found_sliding_solution.window_size           = window_size;
      found_sliding_solution.intra_insertion_index = window_start_idx - 1;
      found_sliding_solution.permutation_index     = permutation_index;
      found_sliding_solution.window_start          = window_start_idx;
    }
  }

  // -- Left part / shifting --

  auto curr_node = s_route.get_node(window_start_idx - 1);

  // curr_node has incorrect backward data since before, window was between this node and the next
  // node (after window)
  s_route.get_node(window_start_idx + window_size)
    .calculate_backward_all(curr_node, s_route.vehicle_info());

  if (dimensions_info.has_dimension(dim_t::TIME) &&
      !curr_node.time_dim.backward_feasible(
        s_route.vehicle_info(), move_candidates.weights[dim_t::TIME], excess_limit)) {
    return;
  }

  // > start_idx (and not >=) because we try to insert until after this point
  for (i_t i = window_start_idx - 1; i > start_idx; --i) {
    // Propagate the updated backward info to end of the window

    if (!backward_fragment_update<i_t, f_t, REQUEST>(
          curr_node, s_route, nodes.data(), window_size, move_candidates.weights, excess_limit)) {
      break;
    }

    // Just for time filtration
    if (dimensions_info.has_dimension(dim_t::TIME)) {
      auto previous_node = s_route.get_node(i - 1);

      nodes[0].time_dim.calculate_backward(
        previous_node.time_dim,
        get_transit_time(
          previous_node.request.info, nodes[0].request.info, s_route.vehicle_info(), true));
      if (!previous_node.time_dim.backward_feasible(
            s_route.vehicle_info(), move_candidates.weights[dim_t::TIME], excess_limit)) {
        break;
      }
    }

    auto previous_node = s_route.get_node(i - 1);

    // Do a cost combine on left part, its forward info is valid

    const double delta = node_t<i_t, f_t, REQUEST>::cost_combine(previous_node,
                                                                 nodes[0],
                                                                 s_route.vehicle_info(),
                                                                 move_candidates.include_objective,
                                                                 move_candidates.weights,
                                                                 s_route.get_objective_cost(),
                                                                 s_route.get_infeasibility_cost());
    // i - 1 because we insert before current node
    // (intra_insertion_index store the index after which we insert)
    if (delta < -EPSILON) {
      const double selection_delta =
        node_t<i_t, f_t, REQUEST>::cost_combine(previous_node,
                                                nodes[0],
                                                s_route.vehicle_info(),
                                                move_candidates.include_objective,
                                                move_candidates.selection_weights,
                                                s_route.get_objective_cost(),
                                                s_route.get_infeasibility_cost());
      if (delta < found_sliding_solution.delta) {
        found_sliding_solution.delta                 = selection_delta;
        found_sliding_solution.window_size           = window_size;
        found_sliding_solution.intra_insertion_index = i - 1;
        found_sliding_solution.permutation_index     = permutation_index;
        found_sliding_solution.window_start          = window_start_idx;
      }
    }

    // Use current node correct backward info to update the previous node

    // Only update if not at the end do avoid useless computation
    if (i - 1 > start_idx) {
      curr_node.calculate_backward_all(previous_node, s_route.vehicle_info());

      if (dimensions_info.has_dimension(dim_t::TIME) &&
          !previous_node.time_dim.backward_feasible(
            s_route.vehicle_info(), move_candidates.weights[dim_t::TIME], excess_limit)) {
        break;
      }

      // Move backward
      curr_node = previous_node;
    }
  }

  // -- Right part / shifting --

  curr_node = s_route.get_node(window_start_idx + window_size);

  // curr_node has incorrect forward data since before, window was between this node and the
  // previous node before the window
  s_route.get_node(window_start_idx - 1).calculate_forward_all(curr_node, s_route.vehicle_info());

  if (dimensions_info.has_dimension(dim_t::TIME) &&
      !curr_node.time_dim.forward_feasible(
        s_route.vehicle_info(), move_candidates.weights[dim_t::TIME], excess_limit)) {
    return;
  }

  for (i_t i = window_start_idx + window_size; i < end_idx; ++i) {
    // Propagate the updated forward info to the beginning of the window
    if (!forward_fragment_update<i_t, f_t, REQUEST>(
          curr_node, s_route, nodes.data(), window_size, move_candidates.weights, excess_limit)) {
      return;
    }

    // Just for time filtration
    if (dimensions_info.has_dimension(dim_t::TIME)) {
      auto next_node = s_route.get_node(i + 1);

      loop_over_constrained_dimensions(dimensions_info, [&] __device__(auto I) {
        get_dimension_of<I>(nodes[window_size - 1])
          .calculate_forward(
            get_dimension_of<I>(next_node),
            get_arc_of_dimension<i_t, f_t, I>(
              nodes[window_size - 1].request.info, next_node.request.info, s_route.vehicle_info()));
      });

      bool valid = true;
      loop_over_constrained_dimensions(dimensions_info, [&] __device__(auto I) {
        valid &= get_dimension_of<I>(next_node).forward_feasible(
          s_route.vehicle_info(), move_candidates.weights[I], excess_limit);
      });
      if (!valid) { return; }
    }

    auto next_node = s_route.get_node(i + 1);

    // Do a cost combine on right part, its backward info is valid and update if the different is
    // better

    const double delta = node_t<i_t, f_t, REQUEST>::cost_combine(nodes[window_size - 1],
                                                                 next_node,
                                                                 s_route.vehicle_info(),
                                                                 move_candidates.include_objective,
                                                                 move_candidates.weights,
                                                                 s_route.get_objective_cost(),
                                                                 s_route.get_infeasibility_cost());

    if (delta < -EPSILON) {
      const double selection_delta =
        node_t<i_t, f_t, REQUEST>::cost_combine(nodes[window_size - 1],
                                                next_node,
                                                s_route.vehicle_info(),
                                                move_candidates.include_objective,
                                                move_candidates.selection_weights,
                                                s_route.get_objective_cost(),
                                                s_route.get_infeasibility_cost());
      if (delta < found_sliding_solution.delta) {
        found_sliding_solution.delta                 = selection_delta;
        found_sliding_solution.window_size           = window_size;
        found_sliding_solution.intra_insertion_index = i;
        found_sliding_solution.permutation_index     = permutation_index;
        found_sliding_solution.window_start          = window_start_idx;
      }
    }

    // Use current node correct backward info to update the next node

    // Only update if not at the end do avoid useless computation
    if (i + 1 < end_idx) {
      curr_node.calculate_forward_all(next_node, s_route.vehicle_info());

      if (dimensions_info.has_dimension(dim_t::TIME) &&
          !next_node.time_dim.forward_feasible(
            s_route.vehicle_info(), move_candidates.weights[dim_t::TIME], excess_limit)) {
        return;
      }

      // Move forward
      curr_node = next_node;
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST, int window_size>
__device__ void try_permutations_cvrp(
  found_sliding_solution_t<i_t>& found_sliding_solution,
  i_t permutation_index,
  const i_t permutation_array[window_size],
  i_t window_start_idx,
  const typename route_t<i_t, f_t, REQUEST>::view_t& s_route,
  const typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
  const typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
  double excess_limit)
{
  static_assert(window_size >= min_permutations<REQUEST>());
  // if (window_size != 1) { return; }
  // printf("Processing window size: %i, window_start_idx: %i\n", window_size, window_start_idx);

  // Filtration phase

  // Discard cases that overstep on the end of the route
  // There are handled by another thread block anyway
  if (window_start_idx + window_size > s_route.get_num_nodes()) { return; }

  // Initialize window

  // Get window based on premutation index

  const auto& dimensions_info                     = solution.problem.dimensions_info;
  constexpr size_t nodes_size                     = max_permutation_intra + 1;
  node_t<i_t, f_t, REQUEST> arr_nodes[nodes_size] = {node_t<i_t, f_t, REQUEST>(dimensions_info),
                                                     node_t<i_t, f_t, REQUEST>(dimensions_info),
                                                     node_t<i_t, f_t, REQUEST>(dimensions_info),
                                                     node_t<i_t, f_t, REQUEST>(dimensions_info),
                                                     node_t<i_t, f_t, REQUEST>(dimensions_info),
                                                     node_t<i_t, f_t, REQUEST>(dimensions_info)};

  auto nodes = raft::device_span<node_t<i_t, f_t, REQUEST>>(arr_nodes, window_size);
  for (int i = 0; i < window_size; ++i) {
    nodes[permutation_array[i]] = s_route.get_node(window_start_idx + i);
  }

  // Try inserting the window along the whole route
  //
  // For PDP we don't need to go from 0 until route_length:
  // Start : highest possible pickup pair from delivery in window (can't insert a delivery before
  // a pickup) End : lowest possible delivery pair from pickup in window (can't insert a pickup
  // after a delivery)

  const auto [start_idx, end_idx] = get_start_end_indices<i_t, f_t, REQUEST, window_size>(
    nodes, window_start_idx, solution, s_route.get_num_nodes());

  // pre-compute fragment cost
  f_t fragment_dist   = 0.;
  f_t fragment_demand = nodes[0].capacity_dim.demand[0];
  for (int i = 1; i < window_size; ++i) {
    fragment_dist += get_arc_of_dimension<i_t, f_t, dim_t::DIST, true>(
      nodes[i - 1].request.info, nodes[i].request.info, s_route.vehicle_info());
    fragment_demand += nodes[i].capacity_dim.demand[0];
  }
  // printf("start_idx: %i, end_idx: %i\n", start_idx, end_idx);

  // Handle the three cases, first inplace, then left part : shifting the window left; then right
  // part : shifting the window right

  // -- In place --

  // Abtritrally choose to propagate forward data (could have been on backward starting from the
  // end)

  if (!forward_fragment_update_cvrp<i_t, f_t, REQUEST>(s_route.get_node(window_start_idx - 1),
                                                       s_route,
                                                       nodes.data(),
                                                       window_size,
                                                       fragment_dist,
                                                       fragment_demand,
                                                       move_candidates.weights,
                                                       excess_limit)) {
    return;
  }

  auto next_node = s_route.get_node(window_start_idx + window_size);

  // Do a cost combine on right part
  // Its backward info is valid and forward has just been updated
  // Store the candidate if the different is better

  const double delta = node_t<i_t, f_t, REQUEST>::cost_combine(nodes[window_size - 1],
                                                               next_node,
                                                               s_route.vehicle_info(),
                                                               move_candidates.include_objective,
                                                               move_candidates.weights,
                                                               s_route.get_objective_cost(),
                                                               s_route.get_infeasibility_cost());
  if (delta < -EPSILON) {
    const double selection_delta =
      node_t<i_t, f_t, REQUEST>::cost_combine(nodes[window_size - 1],
                                              next_node,
                                              s_route.vehicle_info(),
                                              move_candidates.include_objective,
                                              move_candidates.selection_weights,
                                              s_route.get_objective_cost(),
                                              s_route.get_infeasibility_cost());
    if (delta < found_sliding_solution.delta) {
      found_sliding_solution.delta                 = selection_delta;
      found_sliding_solution.window_size           = window_size;
      found_sliding_solution.intra_insertion_index = window_start_idx - 1;
      found_sliding_solution.permutation_index     = permutation_index;
      found_sliding_solution.window_start          = window_start_idx;
    }
  }
  // printf("Left part shifing\n");

  // -- Left part / shifting --

  auto curr_node = s_route.get_node(window_start_idx - 1);

  // curr_node has incorrect backward data since before, window was between this node and the next
  // node (after window)
  s_route.get_node(window_start_idx + window_size)
    .calculate_backward_all(curr_node, s_route.vehicle_info());

  // > start_idx (and not >=) because we try to insert until after this point
  for (i_t i = window_start_idx - 1; i > start_idx; --i) {
    // Propagate the updated backward info to end of the window

    if (!backward_fragment_update_cvrp<i_t, f_t, REQUEST>(curr_node,
                                                          s_route,
                                                          nodes.data(),
                                                          window_size,
                                                          fragment_dist,
                                                          fragment_demand,
                                                          move_candidates.weights,
                                                          excess_limit)) {
      break;
    }

    auto previous_node = s_route.get_node(i - 1);

    // Do a cost combine on left part, its forward info is valid

    const double delta = node_t<i_t, f_t, REQUEST>::cost_combine(previous_node,
                                                                 nodes[0],
                                                                 s_route.vehicle_info(),
                                                                 move_candidates.include_objective,
                                                                 move_candidates.weights,
                                                                 s_route.get_objective_cost(),
                                                                 s_route.get_infeasibility_cost());
    // i - 1 because we insert before current node
    // (intra_insertion_index store the index after which we insert)
    if (delta < -EPSILON) {
      const double selection_delta =
        node_t<i_t, f_t, REQUEST>::cost_combine(previous_node,
                                                nodes[0],
                                                s_route.vehicle_info(),
                                                move_candidates.include_objective,
                                                move_candidates.selection_weights,
                                                s_route.get_objective_cost(),
                                                s_route.get_infeasibility_cost());
      if (delta < found_sliding_solution.delta) {
        found_sliding_solution.delta                 = selection_delta;
        found_sliding_solution.window_size           = window_size;
        found_sliding_solution.intra_insertion_index = i - 1;
        found_sliding_solution.permutation_index     = permutation_index;
        found_sliding_solution.window_start          = window_start_idx;
      }
    }

    // Use current node correct backward info to update the previous node

    // Only update if not at the end do avoid useless computation
    if (i - 1 > start_idx) {
      curr_node.calculate_backward_all(previous_node, s_route.vehicle_info());

      if (dimensions_info.has_dimension(dim_t::TIME) &&
          !previous_node.time_dim.backward_feasible(
            s_route.vehicle_info(), move_candidates.weights[dim_t::TIME], excess_limit)) {
        break;
      }

      // Move backward
      curr_node = previous_node;
    }
  }

  // -- Right part / shifting --

  curr_node = s_route.get_node(window_start_idx + window_size);

  // curr_node has incorrect forward data since before, window was between this node and the
  // previous node before the window
  s_route.get_node(window_start_idx - 1).calculate_forward_all(curr_node, s_route.vehicle_info());

  for (i_t i = window_start_idx + window_size; i < end_idx; ++i) {
    // printf("Right shift: %i\n", i);
    // Propagate the updated forward info to the beginning of the window
    if (!forward_fragment_update_cvrp<i_t, f_t, REQUEST>(curr_node,
                                                         s_route,
                                                         nodes.data(),
                                                         window_size,
                                                         fragment_dist,
                                                         fragment_demand,
                                                         move_candidates.weights,
                                                         excess_limit)) {
      return;
    }

    auto next_node = s_route.get_node(i + 1);

    // Do a cost combine on right part, its backward info is valid and update if the different is
    // better

    const double delta = node_t<i_t, f_t, REQUEST>::cost_combine(nodes[window_size - 1],
                                                                 next_node,
                                                                 s_route.vehicle_info(),
                                                                 move_candidates.include_objective,
                                                                 move_candidates.weights,
                                                                 s_route.get_objective_cost(),
                                                                 s_route.get_infeasibility_cost());
    // printf("delta: %f, ls_epsilon: %f\n", delta, move_candidates.ls_epsilon);

    if (delta < -EPSILON) {
      const double selection_delta =
        node_t<i_t, f_t, REQUEST>::cost_combine(nodes[window_size - 1],
                                                next_node,
                                                s_route.vehicle_info(),
                                                move_candidates.include_objective,
                                                move_candidates.selection_weights,
                                                s_route.get_objective_cost(),
                                                s_route.get_infeasibility_cost());
      if (delta < found_sliding_solution.delta) {
        found_sliding_solution.delta                 = selection_delta;
        found_sliding_solution.window_size           = window_size;
        found_sliding_solution.intra_insertion_index = i;
        found_sliding_solution.permutation_index     = permutation_index;
        found_sliding_solution.window_start          = window_start_idx;
      }
    }

    // Use current node correct backward info to update the next node

    // Only update if not at the end do avoid useless computation
    if (i + 1 < end_idx) {
      curr_node.calculate_forward_all(next_node, s_route.vehicle_info());

      if (dimensions_info.has_dimension(dim_t::TIME) &&
          !next_node.time_dim.forward_feasible(
            s_route.vehicle_info(), move_candidates.weights[dim_t::TIME], excess_limit)) {
        return;
      }

      // Move forward
      curr_node = next_node;
    }
  }
}

// Handle permutation cases from permut_idx to max_permut (avoids having to write many ifs + is
// generic)
template <typename i_t,
          typename f_t,
          request_t REQUEST,
          int permut_idx,
          int max_permut,
          bool is_cvrp>
DI void permutation_cases(int i,
                          found_sliding_solution_t<i_t>& found_sliding_solution,
                          i_t intra_idx,
                          const typename route_t<i_t, f_t, REQUEST>::view_t& s_route,
                          const typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                          const typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
                          double excess_limit)
{
  if constexpr (permut_idx <= max_permut) {
    // When getting the nth permutation it's necessary to scale back down the indices to go from
    // [0
    // - fact(n)]
    if (i < sum_factorial<REQUEST>(permut_idx)) {
      constexpr int window_size = permut_idx;
      i_t permutation_array[window_size];
      const int scaled_down_permutation_index =
        i -
        ((permut_idx == min_permutations<REQUEST>()) ? 0 : sum_factorial<REQUEST>(permut_idx - 1));
      get_nth_permutation<i_t, window_size, REQUEST>(permutation_array,
                                                     scaled_down_permutation_index);

      if constexpr (is_cvrp) {
        try_permutations_cvrp<i_t, f_t, REQUEST, window_size>(found_sliding_solution,
                                                              scaled_down_permutation_index,
                                                              permutation_array,
                                                              intra_idx,
                                                              s_route,
                                                              solution,
                                                              move_candidates,
                                                              excess_limit);
      } else {
        try_permutations<i_t, f_t, REQUEST, window_size>(found_sliding_solution,
                                                         scaled_down_permutation_index,
                                                         permutation_array,
                                                         intra_idx,
                                                         s_route,
                                                         solution,
                                                         move_candidates,
                                                         excess_limit);
      }
    } else
      permutation_cases<i_t, f_t, REQUEST, permut_idx + 1, max_permut, is_cvrp>(
        i, found_sliding_solution, intra_idx, s_route, solution, move_candidates, excess_limit);
  }
}

template <typename i_t, typename f_t, request_t REQUEST, bool is_cvrp = false>
__global__ void kernel_perform_sliding_window(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  found_sliding_solution_t<i_t>* best_candidates,
  typename move_candidates_t<i_t, f_t>::view_t move_candidates,
  int* locks,
  int blocks_per_node)
{
  extern __shared__ i_t shmem[];
  // Each block handles a different starting point for the window
  // +1 to skip depot
  const bool depot_included = solution.problem.order_info.depot_included;
  const i_t node_idx        = blockIdx.x / blocks_per_node;
  const auto node_info      = move_candidates.nodes_to_search.sampled_nodes_to_search[node_idx];

  cuopt_assert(node_info.node() <
                 solution.get_num_orders() + solution.n_routes * after_depot_insertion_multiplier,
               "Invalid node id");
  // special node that represent after depot insertion is ignored
  if (node_info.node() >= solution.get_num_orders()) { return; }

  // Retrive associated node info

  const auto [route_id, intra_idx] =
    solution.route_node_map.get_route_id_and_intra_idx(node_info.node());

  if (route_id == -1)  // Handle unrouted node case for GES
    return;

  cuopt_assert(route_id >= 0, "Invalid route id");
  cuopt_assert(route_id < solution.n_routes, "Invalid route id");

  const auto& orginal_route = solution.routes[route_id];
  const auto route_length   = orginal_route.get_num_nodes();
  cuopt_assert(route_length > 1, "Invalid route length");

  cuopt_assert(intra_idx > 0, "Invalid intra_idx");
  cuopt_assert(intra_idx < route_length, "Invalid intra_idx");

  // Copy route to shared

  auto s_route =
    route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(shmem, orginal_route, route_length);
  __syncthreads();

  s_route.copy_from(orginal_route);
  __syncthreads();

  // Find what is the best (lowest cost) window insertion
  //
  // All permutations of windows (size 2 to max_permutation_intra) are tryed along the whole route
  // Insertion with the lowest cost is recorded globally

  found_sliding_solution_t<i_t> found_sliding_solution =
    is_sliding_uinitialized_t<i_t>::init_data();

  const double excess_limit =
    s_route.get_weighted_excess(move_candidates.weights) * ls_excess_multiplier_route;

  // Each thread handle one window size + permutation case
  // Block-stride loop pattern is used to go through all possible cases

  // Last "cycle" instead of having idle threads, they will be working on the few first
  // permutations
  // of window of size of max_permutation_intra + 1
  const int limit =
    ((sum_factorial<REQUEST>(max_permutation_intra) + blockDim.x - 1) / blockDim.x) * blockDim.x;

  const int permutes_per_block = (limit + blocks_per_node - 1) / blocks_per_node;
  const int offset             = (blockIdx.x % blocks_per_node) * permutes_per_block;
  for (int i = threadIdx.x; i < permutes_per_block; i += blockDim.x) {
    permutation_cases<i_t,
                      f_t,
                      REQUEST,
                      min_permutations<REQUEST>(),
                      max_permutation_intra + 1,
                      is_cvrp>(i + offset,
                               found_sliding_solution,
                               intra_idx,
                               s_route,
                               solution,
                               move_candidates,
                               excess_limit);
  }

  __shared__ int reduction_index;
  __shared__ double shbuf[warp_size * 2];

  int idx = threadIdx.x;
  // block_reduce_ranked changes found_sliding_solution
  double saved_cost = found_sliding_solution.delta;
  block_reduce_ranked(saved_cost, idx, shbuf, &reduction_index);

  // Elected thread write globally its delta
  // Handle non found case
  if (shbuf[0] != std::numeric_limits<double>::max() && shbuf[0] < -EPSILON &&
      reduction_index == threadIdx.x) {
    // move_candidates.nodes_to_search.active_nodes_impacted[node_info.node()] = 1;
    while (atomicCAS(&locks[route_id], 0, 1))
      ;
    // Acquire
    __threadfence();
    if (found_sliding_solution.delta < best_candidates[route_id].delta)
      best_candidates[route_id] = found_sliding_solution;
    __threadfence();
    // Release
    locks[route_id] = 0;
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
DI bool check_route(const typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                    const typename route_t<i_t, f_t, REQUEST>::view_t& s_route,
                    const bool include_objective,
                    const infeasible_cost_t& weights)
{
  __syncthreads();
  // Check cost reduction
  cuopt_assert(
    abs(s_route.get_cost(include_objective, weights) -
        solution.routes[s_route.get_id()].get_cost(include_objective, weights)) > EPSILON,
    "Cost should be lower");

  for (int i = threadIdx.x + 1; i < s_route.get_num_nodes(); i += blockDim.x) {
    const auto& node = s_route.get_node(i);

    if (!node.node_info().is_break()) {
      auto [route_id, intra_route_id] =
        solution.route_node_map.get_route_id_and_intra_idx(node.request.info);

      cuopt_assert(route_id == s_route.get_id(), "Bad node route id");
      cuopt_assert(i == intra_route_id, "Bad intra_route_idx");
    }
  }

  // Check that all nodes that were here are still there
  for (int i = threadIdx.x + 1; i < s_route.get_num_nodes(); i += blockDim.x) {
    const int id                = solution.routes[s_route.get_id()].requests().node_info[i].node();
    [[maybe_unused]] bool found = false;
    for (int j = 1; j < s_route.get_num_nodes(); ++j) {
      if (s_route.requests().node_info[j].node() == id) {
        found = true;
        break;
      }
    }

    cuopt_assert(found, "Node not found");
  }

  // Check that all deliveries are after pikcups
  if constexpr (REQUEST == request_t::PDP) {
    for (int i = threadIdx.x + 1; i < s_route.get_num_nodes(); i += blockDim.x) {
      const auto& curr_node = s_route.get_node(i);
      if (curr_node.node_info().is_service_node()) {
        const auto& brother_info = curr_node.request.brother_info;

        auto [brother_route_id, brother_intra_idx] =
          solution.route_node_map.get_route_id_and_intra_idx(brother_info);
        cuopt_assert(brother_route_id == s_route.get_id(),
                     "Node and brother should be in same route!");
        if (curr_node.request.is_pickup() && i > brother_intra_idx)
          cuopt_assert(false, "Bad order pickup delivery");
      }
    }
  }
  __syncthreads();

  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
DI void mark_impacted_nodes(const typename route_t<i_t, f_t, REQUEST>::view_t& route,
                            typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
                            const found_sliding_solution_t<i_t>& best_candidate,
                            i_t n_orders)
{
  // mark the window itself and also the surrounding positions
  if (best_candidate.window_start - 1 == 0 || best_candidate.intra_insertion_index == 0) {
    if (threadIdx.x == 0) {
      move_candidates.nodes_to_search.active_nodes_impacted[route.get_id() + n_orders] = 1;
    }
  }
  // add two more nodes
  i_t start = max(best_candidate.window_start - 1, 1);
  // add two more nodes
  i_t end =
    min(best_candidate.window_start + best_candidate.window_size + 1, route.get_num_nodes());
  for (i_t i = threadIdx.x + start; i < end; i += blockDim.x) {
    move_candidates.nodes_to_search.active_nodes_impacted[route.node_id(i)] = 1;
  }
  start = max(best_candidate.intra_insertion_index, 1);
  end   = min(best_candidate.intra_insertion_index + 2, route.get_num_nodes());
  // mark the surroundings of the new position
  for (i_t i = threadIdx.x + start; i < end; i += blockDim.x) {
    move_candidates.nodes_to_search.active_nodes_impacted[route.node_id(i)] = 1;
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void execute_sliding_move(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                                     found_sliding_solution_t<i_t>* best_candidates,
                                     typename move_candidates_t<i_t, f_t>::view_t move_candidates,
                                     // For debug pruposes only
                                     [[maybe_unused]] double* total_delta)
{
  const i_t route_id                                 = blockIdx.x;
  const found_sliding_solution_t<i_t> best_candidate = best_candidates[route_id];

  cuopt_assert(route_id >= 0, "Invalid route id");
  cuopt_assert(route_id < solution.n_routes, "Invalid route id");
  auto& orginal_route = solution.routes[route_id];

  if (is_sliding_uinitialized_t<i_t>{}(best_candidate)) {
    return;
  } else {
    mark_impacted_nodes<i_t, f_t, REQUEST>(
      orginal_route, move_candidates, best_candidate, solution.get_num_orders());
  }

  extern __shared__ i_t shmem[];
  node_t<i_t, f_t, REQUEST>* nodes = (node_t<i_t, f_t, REQUEST>*)shmem;

  const i_t aligned_bytes = raft::alignTo(
    sizeof(node_t<i_t, f_t, REQUEST>) * (max_permutation_intra + 1), sizeof(infeasible_cost_t));

  auto s_route = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    (i_t*)(((uint8_t*)shmem) + aligned_bytes), orginal_route, orginal_route.get_num_nodes());
  __syncthreads();

  s_route.copy_from(orginal_route);
  __syncthreads();

  // Update route : remove window, insert window, update infos

  const int window_size = best_candidate.window_size;
  // + 1 because we allow trailing thread to go over
  cuopt_assert(
    window_size >= min_permutations<REQUEST>() && window_size <= max_permutation_intra + 1,
    "Invalid found window_size");
  if (threadIdx.x == 0) {
    // Find what nodes should be inserted

    i_t permutation_array[max_permutation_intra + 1];
    i_t base_indices[max_permutation_intra + 1];

    // Init base indices & retrieve permutation
    for (int i = 0; i < window_size; ++i)
      base_indices[i] = i;
    get_nth_permutation<i_t, REQUEST>(
      permutation_array, best_candidate.permutation_index, base_indices, window_size);

    const int old_start_idx = best_candidate.window_start;
    cuopt_assert(old_start_idx > 0 && old_start_idx + window_size - 1 < s_route.get_num_nodes(),
                 "Invalid old start index");

    // Store
    for (int i = 0; i < window_size; ++i)
      nodes[permutation_array[i]] = s_route.get_node(i + old_start_idx);

    const int new_start_idx = best_candidate.intra_insertion_index;

    // In place insertion
    if (old_start_idx == new_start_idx + 1) {
      for (int i = 0; i < window_size; ++i)
        s_route.set_node(i + new_start_idx + 1, nodes[i]);
    } else {
      // Left shift
      for (int i = old_start_idx; (i + window_size) < s_route.get_num_nodes(); ++i)
        s_route.set_node(i, s_route.get_node(i + window_size));

      // Right shift nodes to leave room for the window insertion

      cuopt_assert(new_start_idx >= 0 && new_start_idx < s_route.get_num_nodes(),
                   "Invalid new start index");
      for (int i = s_route.get_num_nodes() - 1;
           i > new_start_idx + ((new_start_idx < old_start_idx) ? window_size : 0);
           --i)
        s_route.set_node(i, s_route.get_node(i - window_size));

      // Insert the nodes

      // + 1 because we insert after the node
      // Handle case where insertion is after initial position of window
      for (int i = 0; i < window_size; ++i)
        s_route.set_node(
          i + new_start_idx + 1 - ((new_start_idx > old_start_idx) ? window_size : 0), nodes[i]);
    }

    // Update info
    route_t<i_t, f_t, REQUEST>::view_t::compute_forward(s_route);
    route_t<i_t, f_t, REQUEST>::view_t::compute_backward(s_route);
    s_route.compute_cost();
    solution.routes_to_copy[route_id]   = 1;
    solution.routes_to_search[route_id] = 1;
  }

  __syncthreads();

  // Update intra_route_idx_per_node
  for (int i = threadIdx.x + 1; i < s_route.get_num_nodes(); i += blockDim.x) {
    const auto& node_info = s_route.requests().node_info[i];
    solution.route_node_map.set_intra_route_idx(node_info, i);
  }
  __syncthreads();

  // Copy back to global
  cuopt_assert((check_route<i_t, f_t, REQUEST>(
                 solution, s_route, move_candidates.include_objective, move_candidates.weights)),
               "_");
  // (..., true) allow to execute instruction only in cuopt_assrt (assert) mode
  if (threadIdx.x == 0)
    cuopt_assert(
      (atomicAdd(total_delta,
                 abs(s_route.get_cost(move_candidates.include_objective, move_candidates.weights) -
                     solution.routes[s_route.get_id()].get_cost(move_candidates.include_objective,
                                                                move_candidates.weights))),
       true),
      "_");
  solution.routes[s_route.get_id()].copy_from(s_route);
}

template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::fill_pdp_considered_nodes(
  solution_t<i_t, f_t, REQUEST>& solution, move_candidates_t<i_t, f_t>& move_candidates)
{
  std::vector<NodeInfo<i_t>> node_id_vector;
  for (i_t i = solution.problem_ptr->order_info.depot_included_; i < solution.get_num_orders();
       ++i) {
    node_id_vector.push_back(solution.problem_ptr->get_node_info_of_node(i));
  }
  move_candidates.nodes_to_search.n_sampled_nodes = node_id_vector.size();
  raft::copy(move_candidates.nodes_to_search.sampled_nodes_to_search.data(),
             node_id_vector.data(),
             node_id_vector.size(),
             solution.sol_handle->get_stream());
  solution.sol_handle->sync_stream();
}

template <typename i_t, typename f_t, request_t REQUEST>
bool local_search_t<i_t, f_t, REQUEST>::perform_sliding_window(
  solution_t<i_t, f_t, REQUEST>& solution, move_candidates_t<i_t, f_t>& move_candidates)
{
  raft::common::nvtx::range fun_scope("run_sliding_window");
  i_t n_moves_found           = 0;
  size_t shared_for_tmp_route = 0;
  solution.compute_max_active();

  if (REQUEST == request_t::PDP) { fill_pdp_considered_nodes(solution, move_candidates); }

  // FIXME:: we should be using is_cvrp_intra(), but other dimensions are not handled correctly in
  // this case, need to replicate what we do in the two opt here
  auto is_cvrp = solution.problem_ptr->is_cvrp();

  sliding_cuda_graph.start_capture(solution.sol_handle->get_stream());
  async_fill(found_sliding_solution_data_,
             is_sliding_uinitialized_t<i_t>::init_data(),
             solution.sol_handle->get_stream());
  async_fill(locks_, 0, solution.sol_handle->get_stream());
  // So that it only trigger in debug
  cuopt_func_call(
    move_candidates.debug_delta.set_value_to_zero_async(solution.sol_handle->get_stream()));

  constexpr i_t thread_per_block = 64;
  shared_for_tmp_route           = solution.get_temp_route_shared_size();
  if ((is_cvrp && !set_shmem_of_kernel(kernel_perform_sliding_window<i_t, f_t, REQUEST, true>,
                                       shared_for_tmp_route)) ||
      (!is_cvrp && !set_shmem_of_kernel(kernel_perform_sliding_window<i_t, f_t, REQUEST, false>,
                                        shared_for_tmp_route))) {
    sliding_cuda_graph.end_capture(solution.sol_handle->get_stream());
    return false;
  }
  int ideal_blocks    = 4 * solution.sol_handle->get_num_sms();
  int blocks_per_node = std::max(ideal_blocks / move_candidates.nodes_to_search.n_sampled_nodes, 1);

  auto n_blocks = move_candidates.nodes_to_search.n_sampled_nodes * blocks_per_node;
  cuopt_assert(n_blocks > 0, "n_blocks should be positive");
  cuopt_expects(n_blocks > 0, error_type_t::RuntimeError, "A runtime error occurred!");
  if (is_cvrp) {
    kernel_perform_sliding_window<i_t, f_t, REQUEST, true>
      <<<n_blocks,  // One block for each node
         thread_per_block,
         shared_for_tmp_route,
         solution.sol_handle->get_stream()>>>(solution.view(),
                                              found_sliding_solution_data_.data(),
                                              move_candidates.view(),
                                              locks_.data(),
                                              blocks_per_node);
  } else {
    kernel_perform_sliding_window<i_t, f_t, REQUEST, false>
      <<<n_blocks,  // One block for each node
         thread_per_block,
         shared_for_tmp_route,
         solution.sol_handle->get_stream()>>>(solution.view(),
                                              found_sliding_solution_data_.data(),
                                              move_candidates.view(),
                                              locks_.data(),
                                              blocks_per_node);
  }
  sliding_cuda_graph.end_capture(solution.sol_handle->get_stream());
  sliding_cuda_graph.launch_graph(solution.sol_handle->get_stream());
  RAFT_CHECK_CUDA(solution.sol_handle->get_stream());
  n_moves_found = thrust::count_if(solution.sol_handle->get_thrust_policy(),
                                   found_sliding_solution_data_.begin(),
                                   found_sliding_solution_data_.end(),
                                   is_sliding_initialized_t<i_t>());

  if (n_moves_found == 0) { return false; }

  const i_t aligned_bytes = raft::alignTo(
    sizeof(node_t<i_t, f_t, REQUEST>) * (max_permutation_intra + 1), sizeof(infeasible_cost_t));
  const size_t aligned_shared_size = aligned_bytes + shared_for_tmp_route;
  if (!set_shmem_of_kernel(execute_sliding_move<i_t, f_t, REQUEST>, aligned_shared_size)) {
    return false;
  }
  [[maybe_unused]] double cost_before = 0., cost_after = 0.;
  cuopt_func_call(solution.compute_cost());
  cuopt_func_call(cost_before =
                    solution.get_cost(move_candidates.include_objective, move_candidates.weights));

  // One block for each found route
  execute_sliding_move<i_t, f_t, REQUEST>
    <<<solution.n_routes, 256, aligned_shared_size, solution.sol_handle->get_stream()>>>(
      solution.view(),
      found_sliding_solution_data_.data(),
      move_candidates.view(),
      move_candidates.debug_delta.data());
  RAFT_CHECK_CUDA(solution.sol_handle->get_stream());
  cuopt_func_call(solution.compute_cost());
  cuopt_func_call(cost_after =
                    solution.get_cost(move_candidates.include_objective, move_candidates.weights));

  cuopt_assert(cost_before - cost_after >= EPSILON, "Cost should improve!");
  cuopt_assert(abs((cost_before - cost_after) -
                     move_candidates.debug_delta.value(solution.sol_handle->get_stream()) <
                   EPSILON),
               "Cost mismatch on cross costs!");
  return true;
}

template bool local_search_t<int, float, request_t::PDP>::perform_sliding_window(
  solution_t<int, float, request_t::PDP>& solution, move_candidates_t<int, float>& move_candidates);
template bool local_search_t<int, float, request_t::VRP>::perform_sliding_window(
  solution_t<int, float, request_t::VRP>& solution, move_candidates_t<int, float>& move_candidates);

template void local_search_t<int, float, request_t::PDP>::fill_pdp_considered_nodes(
  solution_t<int, float, request_t::PDP>& solution, move_candidates_t<int, float>& move_candidates);
template void local_search_t<int, float, request_t::VRP>::fill_pdp_considered_nodes(
  solution_t<int, float, request_t::VRP>& solution, move_candidates_t<int, float>& move_candidates);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
