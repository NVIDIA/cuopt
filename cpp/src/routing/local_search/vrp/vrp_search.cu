/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "vrp_execute.cuh"
#include "vrp_search.cuh"

namespace cuopt {
namespace routing {
namespace detail {

// FIXME get rid of share memory completely. (Akif has a commit for this somewhere)
#ifdef BENCHMARK
constexpr int max_n_neighbors = 128;
#else
constexpr int max_n_neighbors = 96;
#endif

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void compute_reverse_distances(typename solution_t<i_t, f_t, REQUEST>::view_t solution)
{
  extern __shared__ double shmem[];

  if (threadIdx.x == 0) {
    auto route    = solution.routes[blockIdx.x];
    auto route_id = route.get_id();
    auto n_nodes  = route.get_num_nodes();

    route.dimensions.distance_dim.reverse_distance[n_nodes] = 0.;
    for (int i = n_nodes - 1; i >= 0; i--) {
      double dist = get_arc_of_dimension<i_t, f_t, dim_t::DIST>(
        route.get_node(i + 1).node_info(), route.get_node(i).node_info(), route.vehicle_info());
      route.dimensions.distance_dim.reverse_distance[i] =
        dist + route.dimensions.distance_dim.reverse_distance[i + 1];
    }
  }
}

/**
 * @brief CVRP overload, no fragment stored in shared memory
 *
 * @tparam i_t
 * @tparam f_t
 * @tparam REQUEST
 * @param solution
 * @param move_candidates
 * @param route_1
 * @param route_2
 * @param search_data
 * @param excess_limit
 * @return
 */

template <typename i_t, typename f_t, request_t REQUEST>
DI thrust::pair<double, double> evaluate_move(
  typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
  typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
  const typename route_t<i_t, f_t, REQUEST>::view_t& route_1,
  const typename route_t<i_t, f_t, REQUEST>::view_t& route_2,
  search_data_t<i_t>& search_data,
  double excess_limit)
{
  auto [inf_delta_1, inf_selection_delta_1] =
    evaluate_cap_infeasibility<i_t, f_t, REQUEST>(solution,
                                                  move_candidates,
                                                  route_1,
                                                  search_data.start_idx_1,
                                                  search_data.frag_size_1,
                                                  route_2,
                                                  search_data.start_idx_2,
                                                  search_data.frag_size_2,
                                                  excess_limit);
  if (inf_delta_1 == std::numeric_limits<double>::max()) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }

  auto [delta_1, selection_delta_1] =
    evaluate_fragment<i_t, f_t, REQUEST>(solution,
                                         move_candidates,
                                         route_1,
                                         search_data.start_idx_1,
                                         search_data.frag_size_1,
                                         route_2,
                                         search_data.start_idx_2,
                                         search_data.frag_size_2,
                                         excess_limit,
                                         search_data.reversed_frag_2);

  auto [inf_delta_2, inf_selection_delta_2] =
    evaluate_cap_infeasibility<i_t, f_t, REQUEST>(solution,
                                                  move_candidates,
                                                  route_2,
                                                  search_data.start_idx_2,
                                                  search_data.frag_size_2,
                                                  route_1,
                                                  search_data.start_idx_1,
                                                  search_data.frag_size_1,
                                                  excess_limit);
  if (inf_delta_2 == std::numeric_limits<double>::max()) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }

  auto [delta_2, selection_delta_2] =
    evaluate_fragment<i_t, f_t, REQUEST>(solution,
                                         move_candidates,
                                         route_2,
                                         search_data.start_idx_2,
                                         search_data.frag_size_2,
                                         route_1,
                                         search_data.start_idx_1,
                                         search_data.frag_size_1,
                                         excess_limit,
                                         search_data.reversed_frag_1);
  return {delta_1 + delta_2 + inf_delta_1 + inf_delta_2,
          selection_delta_1 + selection_delta_2 + inf_selection_delta_1 + inf_selection_delta_2};
}

template <typename i_t, typename f_t, request_t REQUEST>
DI thrust::pair<double, double> evaluate_move(
  typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
  typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
  const typename route_t<i_t, f_t, REQUEST>::view_t& route_1,
  const typename route_t<i_t, f_t, REQUEST>::view_t& route_2,
  search_data_t<i_t>& search_data,
  typename dimensions_route_t<i_t, f_t, REQUEST>::view_t& fragment,
  double excess_limit)
{
  fragment.copy_nodes_from(
    0, route_2, search_data.start_idx_2 + 1, search_data.frag_size_2, search_data.reversed_frag_2);
  auto [delta_1, selection_delta_1] =
    evaluate_fragment<i_t, f_t, REQUEST>(solution,
                                         move_candidates,
                                         route_1,
                                         search_data.start_idx_1,
                                         search_data.start_idx_1 + search_data.frag_size_1 + 1,
                                         search_data.frag_size_2,
                                         fragment,
                                         excess_limit);
  if (delta_1 == std::numeric_limits<double>::max()) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }
  i_t start_idx;
  i_t end_idx;
  i_t frag_size;
  if (search_data.offset == 1) {
    fragment.copy_nodes_from(0, route_2, search_data.start_idx_2 + search_data.frag_size_2 + 1, 1);
    fragment.copy_nodes_from(1,
                             route_1,
                             search_data.start_idx_1 + 1,
                             search_data.frag_size_1,
                             search_data.reversed_frag_1);
    start_idx = search_data.start_idx_2;
    end_idx   = search_data.start_idx_2 + search_data.frag_size_2 + 2;
    frag_size = search_data.frag_size_1 + 1;
  } else if (search_data.offset == 0) {
    fragment.copy_nodes_from(0,
                             route_1,
                             search_data.start_idx_1 + 1,
                             search_data.frag_size_1,
                             search_data.reversed_frag_1);
    start_idx = search_data.start_idx_2;
    end_idx   = search_data.start_idx_2 + search_data.frag_size_2 + 1;
    frag_size = search_data.frag_size_1;
  } else {
    fragment.copy_nodes_from(0,
                             route_1,
                             search_data.start_idx_1 + 1,
                             search_data.frag_size_1,
                             search_data.reversed_frag_1);
    fragment.copy_nodes_from(search_data.frag_size_1, route_2, search_data.start_idx_2, 1);
    start_idx = search_data.start_idx_2 - 1;
    end_idx   = search_data.start_idx_2 + search_data.frag_size_2 + 1;
    frag_size = search_data.frag_size_1 + 1;
  }

  if (start_idx < 0 || end_idx > route_2.get_num_nodes()) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }

  auto [delta_2, selection_delta_2] = evaluate_fragment<i_t, f_t, REQUEST>(
    solution, move_candidates, route_2, start_idx, end_idx, frag_size, fragment, excess_limit);
  if (delta_2 == std::numeric_limits<double>::max()) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }
  return {delta_1 + delta_2, selection_delta_1 + selection_delta_2};
}

template <typename i_t, typename f_t, request_t REQUEST>
DI thrust::pair<double, double> evaluate_2_opt_route(
  typename solution_t<i_t, f_t, REQUEST>::view_t const& solution,
  typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
  typename route_t<i_t, f_t, REQUEST>::view_t const& route_1,
  typename route_t<i_t, f_t, REQUEST>::view_t const& route_2,
  VehicleInfo<f_t> const& vehicle_info,
  i_t vehicle_id,
  i_t route_1_end,
  i_t route_2_start,
  double excess_limit)
{
  auto start_depot_node_info  = solution.problem.get_start_depot_node_info(vehicle_id);
  auto return_depot_node_info = solution.problem.get_return_depot_node_info(vehicle_id);
  auto depot_node             = create_depot_node<i_t, f_t, REQUEST>(
    solution.problem, start_depot_node_info, return_depot_node_info, vehicle_id);
  auto start_depot_node = create_depot_node<i_t, f_t, REQUEST>(
    solution.problem, start_depot_node_info, return_depot_node_info, vehicle_id);
  auto temp_node = start_depot_node;

  for (i_t i = 0; i < route_1_end; ++i) {
    auto next_node = route_1.get_node(i + 1);
    if (!next_node.node_info().is_service_node()) {
      return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    }
    temp_node.calculate_forward_all(next_node, vehicle_info);
    if (!next_node.forward_feasible(vehicle_info, move_candidates.weights, excess_limit)) {
      return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    }
    temp_node = next_node;
  }

  for (i_t i = route_2_start; i < route_2.get_num_nodes() - 1; ++i) {
    auto next_node = route_2.get_node(i + 1);
    if (!next_node.node_info().is_service_node()) {
      return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    }
    temp_node.calculate_forward_all(next_node, vehicle_info);
    if (!next_node.forward_feasible(vehicle_info, move_candidates.weights, excess_limit)) {
      return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    }
    temp_node = next_node;
  }

  auto return_depot_node = create_depot_node<i_t, f_t, REQUEST>(
    solution.problem, return_depot_node_info, start_depot_node_info, vehicle_id);
  double delta = temp_node.calculate_forward_all_and_delta(return_depot_node,
                                                           vehicle_info,
                                                           move_candidates.include_objective,
                                                           move_candidates.weights,
                                                           route_1.get_objective_cost(),
                                                           route_1.get_infeasibility_cost());

  if (!return_depot_node.feasible(vehicle_info, move_candidates.weights, excess_limit)) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }

  double selection_delta =
    temp_node.calculate_forward_all_and_delta(return_depot_node,
                                              vehicle_info,
                                              move_candidates.include_objective,
                                              move_candidates.selection_weights,
                                              route_1.get_objective_cost(),
                                              route_1.get_infeasibility_cost());

  auto vehicle_fixed_cost_delta = vehicle_info.fixed_cost - route_1.vehicle_info().fixed_cost;

  return {delta + vehicle_fixed_cost_delta, selection_delta + vehicle_fixed_cost_delta};
}

template <typename i_t, typename f_t, request_t REQUEST>
DI thrust::pair<double, double> evaluate_2_opt_star_move_hvrp(
  typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
  typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
  const typename route_t<i_t, f_t, REQUEST>::view_t& route_1,
  const typename route_t<i_t, f_t, REQUEST>::view_t& route_2,
  search_data_t<i_t>& search_data,
  double excess_limit)
{
  // prevent 2-opt* from creating an empty route
  if (search_data.start_idx_1 == 0 &&
      search_data.start_idx_2 + search_data.frag_size_2 == route_2.get_num_nodes()) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }
  if (search_data.start_idx_2 == 0 &&
      search_data.start_idx_1 + search_data.frag_size_1 == route_1.get_num_nodes()) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }

  auto swap_types = search_data.move_type % 2 == 1;
  if (swap_types && solution.problem.fleet_info.buckets[route_1.get_vehicle_id()] ==
                      solution.problem.fleet_info.buckets[route_2.get_vehicle_id()]) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }

  auto const& first_vehicle_type = swap_types ? route_2 : route_1;
  auto [delta_1, selection_delta_1] =
    evaluate_2_opt_route<i_t, f_t, REQUEST>(solution,
                                            move_candidates,
                                            route_1,
                                            route_2,
                                            first_vehicle_type.vehicle_info(),
                                            first_vehicle_type.get_vehicle_id(),
                                            search_data.start_idx_1,
                                            search_data.start_idx_2,
                                            excess_limit);

  if (delta_1 == std::numeric_limits<double>::max()) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }

  auto const& second_vehicle_type = swap_types ? route_1 : route_2;
  auto [delta_2, selection_delta_2] =
    evaluate_2_opt_route<i_t, f_t, REQUEST>(solution,
                                            move_candidates,
                                            route_2,
                                            route_1,
                                            second_vehicle_type.vehicle_info(),
                                            second_vehicle_type.get_vehicle_id(),
                                            search_data.start_idx_2,
                                            search_data.start_idx_1,
                                            excess_limit);
  if (delta_2 == std::numeric_limits<double>::max()) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }

  return {delta_1 + delta_2, selection_delta_1 + selection_delta_2};
}

template <typename i_t, typename f_t, request_t REQUEST>
DI thrust::pair<double, double> evaluate_2_opt_star_move_homogenous(
  typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
  typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
  const typename route_t<i_t, f_t, REQUEST>::view_t& route_1,
  const typename route_t<i_t, f_t, REQUEST>::view_t& route_2,
  search_data_t<i_t>& search_data,
  double excess_limit)
{
  // prevent 2-opt* from creating an empty route
  if (search_data.start_idx_1 == 0 &&
      search_data.start_idx_2 + search_data.frag_size_2 == route_2.get_num_nodes()) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }
  if (search_data.start_idx_2 == 0 &&
      search_data.start_idx_1 + search_data.frag_size_1 == route_1.get_num_nodes()) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }
  double delta_1 = node_t<i_t, f_t, REQUEST>::cost_combine(
    route_1.get_node(search_data.start_idx_1),
    route_2.get_node(search_data.start_idx_2 + search_data.frag_size_2),
    route_1.vehicle_info(),
    move_candidates.include_objective,
    move_candidates.weights,
    route_1.get_objective_cost(),
    route_1.get_infeasibility_cost());
  if (delta_1 == std::numeric_limits<double>::max()) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }

  double delta_2 = node_t<i_t, f_t, REQUEST>::cost_combine(
    route_2.get_node(search_data.start_idx_2),
    route_1.get_node(search_data.start_idx_1 + search_data.frag_size_1),
    route_2.vehicle_info(),
    move_candidates.include_objective,
    move_candidates.weights,
    route_2.get_objective_cost(),
    route_2.get_infeasibility_cost());
  if (delta_2 == std::numeric_limits<double>::max()) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }
  double selection_delta_1 = node_t<i_t, f_t, REQUEST>::cost_combine(
    route_1.get_node(search_data.start_idx_1),
    route_2.get_node(search_data.start_idx_2 + search_data.frag_size_2),
    route_1.vehicle_info(),
    move_candidates.include_objective,
    move_candidates.selection_weights,
    route_1.get_objective_cost(),
    route_1.get_infeasibility_cost());
  double selection_delta_2 = node_t<i_t, f_t, REQUEST>::cost_combine(
    route_2.get_node(search_data.start_idx_2),
    route_1.get_node(search_data.start_idx_1 + search_data.frag_size_1),
    route_2.vehicle_info(),
    move_candidates.include_objective,
    move_candidates.selection_weights,
    route_2.get_objective_cost(),
    route_2.get_infeasibility_cost());
  return {delta_1 + delta_2, selection_delta_1 + selection_delta_2};
}

template <typename i_t, typename f_t, request_t REQUEST>
DI bool get_nodes_to_consider(typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                              typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
                              search_data_t<i_t>& search_data,
                              bool outgoing_direction,
                              bool recycle)
{
  constexpr bool exclude_self_in_neighbors = true;
  const int max_neighbors                  = min(max_n_neighbors, solution.get_num_orders());
  if (search_data.block_node_id >= solution.get_num_orders()) {
    search_data.start_idx_1       = 0;
    i_t depot_after_special_index = search_data.block_node_id - solution.get_num_orders();
    i_t route_id                  = depot_after_special_index / after_depot_insertion_multiplier;
    i_t neighbor_batch            = depot_after_special_index % after_depot_insertion_multiplier;
    // reduce it to a unique id again, for move recording and consistency
    search_data.block_node_id = solution.get_num_orders() + route_id;
    // We shouldn't have empty routes, however python(hetero) tests frequently produces empty routes
    if (solution.routes[route_id].get_num_nodes() <= 1) { return false; }
    auto first_node_of_route = solution.routes[route_id].node_info(1);
    if (!first_node_of_route.is_service_node()) { return false; }
    if (!recycle) {
      search_data.nodes_to_consider =
        move_candidates.viables.get_viable_to_pickups(first_node_of_route.node(),
                                                      solution.get_num_requests(),
                                                      max_neighbors,
                                                      exclude_self_in_neighbors,
                                                      neighbor_batch);
      // there is no relocate out for the depot location
      if (outgoing_direction || threadIdx.x >= search_data.nodes_to_consider.size()) {
        return false;
      }
      search_data.node_id_2 = search_data.nodes_to_consider[threadIdx.x];
    }
  } else {
    if (!recycle) {
      if (outgoing_direction) {
        search_data.nodes_to_consider =
          move_candidates.viables.get_viable_to_pickups(search_data.block_node_id,
                                                        solution.get_num_requests(),
                                                        max_neighbors,
                                                        exclude_self_in_neighbors);
      } else {
        search_data.nodes_to_consider =
          move_candidates.viables.get_viable_from_pickups(search_data.block_node_id,
                                                          solution.get_num_requests(),
                                                          max_neighbors,
                                                          exclude_self_in_neighbors);
      }
      if (threadIdx.x >= search_data.nodes_to_consider.size()) { return false; }

      search_data.node_id_2 = search_data.nodes_to_consider[threadIdx.x];
      if (outgoing_direction) { raft::swapVals(search_data.block_node_id, search_data.node_id_2); }
    }
    search_data.start_idx_1 =
      solution.route_node_map.intra_route_idx_per_node[search_data.block_node_id];
  }
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
DI bool get_work_config(typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                        typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
                        search_data_t<i_t>& search_data,
                        bool recycle)
{
  // each move type considers all nodes + insertion after depot
  auto searched_nodes = move_candidates.nodes_to_search;
  NodeInfo<i_t> node_info;
  i_t move_category;
  if (recycle) {
    i_t gl_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    // tailing unused threads
    if (gl_thread_id >= (i_t)vrp_move_t::SIZE * searched_nodes.n_sampled_nodes) { return true; }
    move_category  = gl_thread_id % (i_t)vrp_move_t::SIZE;
    int2 node_pair = searched_nodes.recycled_node_pairs[gl_thread_id / (i_t)vrp_move_t::SIZE];
    search_data.block_node_id = node_pair.x;
    search_data.node_id_2     = node_pair.y;
  } else {
    const i_t n_blocks_per_move_type = searched_nodes.n_sampled_nodes;
    node_info     = searched_nodes.sampled_nodes_to_search[blockIdx.x / (i_t)vrp_move_t::SIZE];
    move_category = blockIdx.x % (i_t)vrp_move_t::SIZE;
    search_data.block_node_id = node_info.node();
  }

  if (move_category <= (i_t)vrp_move_t::CROSS) {
    constexpr bool outgoing_relocate = false;
    bool valid_work_load             = get_nodes_to_consider<i_t, f_t, REQUEST>(
      solution, move_candidates, search_data, outgoing_relocate, recycle);
    if (!valid_work_load) { return true; }
    search_data.move_type      = move_category;
    const i_t n_reverse_types  = 4;
    const i_t n_fragment_sizes = max_cross_size * max_cross_size;
    const i_t n_fragment_types = n_reverse_types * n_fragment_sizes;
    const i_t n_offset         = 3;
    i_t offset_type            = move_category / n_fragment_types;
    i_t fragment_type          = move_category % n_fragment_types;
    i_t fragment_size_type     = fragment_type / n_reverse_types;
    i_t reverse_type           = fragment_type % n_reverse_types;
    search_data.offset         = -1 + (offset_type % n_offset);
    if (solution.problem.is_cvrp() && search_data.offset != 0) { return true; }
    search_data.frag_size_1 = (fragment_size_type / max_cross_size) + 1;
    search_data.frag_size_2 = (fragment_size_type % max_cross_size) + 1;
    if (reverse_type == 0) {
      search_data.reversed_frag_1 = false;
      search_data.reversed_frag_2 = false;
      // we need to subrtract 1, in order to include the node_id_2 in the fragmnet
      search_data.start_idx_2 =
        solution.route_node_map.intra_route_idx_per_node[search_data.node_id_2] - 1;
    } else if (reverse_type == 1) {
      search_data.reversed_frag_1 = true;
      search_data.reversed_frag_2 = false;
      // we need to subrtract 1, in order to include the node_id_2 in the fragmnet
      search_data.start_idx_2 =
        solution.route_node_map.intra_route_idx_per_node[search_data.node_id_2] - 1;
    } else if (reverse_type == 2) {
      search_data.reversed_frag_1 = false;
      search_data.reversed_frag_2 = true;
      search_data.start_idx_2 =
        solution.route_node_map.intra_route_idx_per_node[search_data.node_id_2] -
        search_data.frag_size_2;
    } else if (reverse_type == 3) {
      search_data.reversed_frag_1 = true;
      search_data.reversed_frag_2 = true;
      search_data.start_idx_2 =
        solution.route_node_map.intra_route_idx_per_node[search_data.node_id_2] -
        search_data.frag_size_2;
    }
  }
  // NOTE: if the shared memory used by the fragment is too large, we can separate this
  // into a separate kernel as this is the only part where we use 6 sized fragments
  // in fact the fragment usage can be completely eliminated by having a non-copying view to the
  // route and changing the forward values of temp nodes on the fly
  else if (move_category <= (i_t)vrp_move_t::RELOCATE) {
    search_data.move_type  = move_category;
    i_t move_type          = move_category - ((i_t)vrp_move_t::CROSS + 1);
    const i_t n_directions = 2;
    bool outgoing_relocate = move_type % n_directions;
    bool valid_work_load   = get_nodes_to_consider<i_t, f_t, REQUEST>(
      solution, move_candidates, search_data, outgoing_relocate, recycle);
    if (!valid_work_load) { return true; }
    const i_t n_reverse_types = 2;
    i_t direction_move_type   = move_type / 2;
    i_t fragment_size_type    = direction_move_type / n_reverse_types;
    i_t reverse_type          = direction_move_type % n_reverse_types;
    cuopt_assert(fragment_size_type < max_relocate_size,
                 "Fragment size should be smaller than the max_relocate_size size");
    search_data.offset      = 0;
    search_data.frag_size_1 = 0;
    search_data.frag_size_2 = fragment_size_type + 1;
    if (reverse_type == 0) {
      search_data.reversed_frag_1 = false;
      search_data.reversed_frag_2 = false;
      // we need to subrtract 1, in order to include the node_id_2 in the fragmnet
      search_data.start_idx_2 =
        solution.route_node_map.intra_route_idx_per_node[search_data.node_id_2] - 1;
    } else if (reverse_type == 1) {
      search_data.reversed_frag_1 = false;
      search_data.reversed_frag_2 = true;
      // we need to subrtract 1, in order to include the node_id_2 in the fragmnet
      search_data.start_idx_2 =
        solution.route_node_map.intra_route_idx_per_node[search_data.node_id_2] -
        search_data.frag_size_2;
    }
  } else if (move_category <= (i_t)vrp_move_t::TWO_OPT_STAR) {
    // Don't run two opt star if there are non uniform breaks
    if (solution.problem.has_non_uniform_breaks()) { return true; }
    constexpr bool outgoing_relocate = false;
    bool valid_work_load             = get_nodes_to_consider<i_t, f_t, REQUEST>(
      solution, move_candidates, search_data, outgoing_relocate, recycle);
    if (!valid_work_load) { return true; }
    search_data.move_type   = move_category;
    search_data.offset      = 0;
    search_data.frag_size_1 = 1;
    search_data.frag_size_2 = 1;
    search_data.start_idx_2 =
      solution.route_node_map.intra_route_idx_per_node[search_data.node_id_2] - 1;
    search_data.reversed_frag_1 = false;
    search_data.reversed_frag_2 = false;
  } else {
    cuopt_assert(false, "Invalid number of blocks!");
  }
  return false;
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void find_vrp_moves_kernel(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                                      typename move_candidates_t<i_t, f_t>::view_t move_candidates,
                                      bool recycle)
{
  extern __shared__ double shmem[];
  search_data_t<i_t> search_data;
  bool early_exit =
    get_work_config<i_t, f_t, REQUEST>(solution, move_candidates, search_data, recycle);
  if (early_exit) return;
  i_t r_id_1;
  if (search_data.block_node_id >= solution.get_num_orders()) {
    r_id_1 = search_data.block_node_id - solution.get_num_orders();
  } else {
    r_id_1 = solution.route_node_map.route_id_per_node[search_data.block_node_id];
  }

  i_t r_id_2 = solution.route_node_map.route_id_per_node[search_data.node_id_2];
  if (r_id_1 == -1 || r_id_2 == -1 || r_id_1 == r_id_2) return;
  cuopt_assert(r_id_1 < solution.routes.size(), "route id should be in range!");
  auto route_1 = solution.routes[r_id_1];
  auto route_2 = solution.routes[r_id_2];
  // if the end index of a fragment is above(or below when reversed) the route boundaries
  if (search_data.start_idx_1 + search_data.frag_size_1 >= route_1.get_num_nodes() ||
      search_data.start_idx_2 + search_data.frag_size_2 >= route_2.get_num_nodes() ||
      search_data.start_idx_2 < 0 || route_1.get_num_nodes() <= 1 || route_2.get_num_nodes() <= 1) {
    return;
  }
  size_t size_of_frag = dimensions_route_t<i_t, f_t, REQUEST>::get_shared_size(
    max_fragment_size, solution.problem.dimensions_info);
  auto frag_ptr = (i_t*)(((uint8_t*)shmem) + size_of_frag * threadIdx.x);
  typename dimensions_route_t<i_t, f_t, REQUEST>::view_t fragment;
  // max_fragment_size-1, because the create shared route adds one more already
  thrust::tie(fragment, frag_ptr) =
    dimensions_route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
      frag_ptr, solution.problem.dimensions_info, max_fragment_size - 1);
  double excess_route_1 = route_1.get_weighted_excess(move_candidates.weights);
  double excess_route_2 = route_2.get_weighted_excess(move_candidates.weights);
  double excess_limit   = (excess_route_1 + excess_route_2) * ls_excess_multiplier_route;
  double cost_delta, selection_delta;
  if (search_data.move_type > (i_t)vrp_move_t::RELOCATE) {
    // 2-opt* doesn't work with special nodes
    if (solution.problem.has_special_nodes()) { return; }
    if (solution.problem.fleet_info.is_homogenous_fleet()) {
      thrust::tie(cost_delta, selection_delta) =
        evaluate_2_opt_star_move_homogenous<i_t, f_t, REQUEST>(
          solution, move_candidates, route_1, route_2, search_data, excess_limit);
    } else {
      thrust::tie(cost_delta, selection_delta) = evaluate_2_opt_star_move_hvrp<i_t, f_t, REQUEST>(
        solution, move_candidates, route_1, route_2, search_data, excess_limit);
    }
  } else {
    if (solution.problem.is_cvrp()) {
      thrust::tie(cost_delta, selection_delta) = evaluate_move<i_t, f_t, REQUEST>(
        solution, move_candidates, route_1, route_2, search_data, excess_limit);
    } else {
      thrust::tie(cost_delta, selection_delta) = evaluate_move<i_t, f_t, REQUEST>(
        solution, move_candidates, route_1, route_2, search_data, fragment, excess_limit);
    }
    // embed the information of reverse fragments in the sizes (-1 for reversed fragment)
    if (search_data.reversed_frag_1) { search_data.frag_size_1 = -search_data.frag_size_1; }
    if (search_data.reversed_frag_2) { search_data.frag_size_2 = -search_data.frag_size_2; }
  }
  i_t route_pair_idx =
    move_candidates.vrp_move_candidates.get_route_pair_idx(r_id_1, r_id_2, solution.n_routes);

  if (cost_delta > -EPSILON) return;
  // for VRP and sliding kernels we record only negative moves with working weights but execute with
  // the alpha and beta
  move_candidates.vrp_move_candidates.record_candidate(
    route_pair_idx,
    search_data.block_node_id,
    search_data.node_id_2,
    search_data.frag_size_1,
    search_data.frag_size_2,
    search_data.move_type,
    search_data.offset,
    selection_delta,
    move_candidates.nodes_to_search.active_nodes_impacted);
}

template <typename i_t, typename f_t, request_t REQUEST>
bool find_vrp_moves(solution_t<i_t, f_t, REQUEST>& sol,
                    move_candidates_t<i_t, f_t>& move_candidates,
                    bool recycle = false)
{
  raft::common::nvtx::range fun_scope("find_vrp_moves");
  if (sol.n_routes < 2) { return false; }

  if (sol.problem_ptr->is_cvrp()) {
    compute_reverse_distances<i_t, f_t, REQUEST>
      <<<sol.get_n_routes(), 32, 0, sol.sol_handle->get_stream()>>>(sol.view());
  }
  i_t TPB             = std::min(max_n_neighbors, sol.problem_ptr->get_num_orders());
  size_t size_of_frag = dimensions_route_t<i_t, f_t, REQUEST>::get_shared_size(
    max_fragment_size, sol.problem_ptr->dimensions_info);
  size_t sh_size           = size_of_frag * TPB;
  i_t n_of_nodes_to_search = move_candidates.nodes_to_search.n_sampled_nodes;
  i_t n_blocks;
  if (recycle) {
    i_t n_threads = n_of_nodes_to_search * (i_t)vrp_move_t::SIZE;
    n_blocks      = (n_threads + TPB - 1) / TPB;
  } else {
    n_blocks = n_of_nodes_to_search * (i_t)vrp_move_t::SIZE;
  }
  cuopt_assert(n_blocks > 0, "n_blocks should be positive");
  cuopt_expects(n_blocks > 0, error_type_t::RuntimeError, "A runtime error occurred!");
  if (!set_shmem_of_kernel(find_vrp_moves_kernel<i_t, f_t, REQUEST>, sh_size)) { return false; }
  move_candidates.vrp_move_candidates.find_kernel_graph.start_capture(sol.sol_handle->get_stream());
  move_candidates.vrp_move_candidates.reset(sol.sol_handle);
  find_vrp_moves_kernel<i_t, f_t, REQUEST>
    <<<n_blocks, TPB, sh_size, sol.sol_handle->get_stream()>>>(
      sol.view(), move_candidates.view(), recycle);
  move_candidates.vrp_move_candidates.find_kernel_graph.end_capture(sol.sol_handle->get_stream());
  move_candidates.vrp_move_candidates.find_kernel_graph.launch_graph(sol.sol_handle->get_stream());
  sol.sol_handle->sync_stream();
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
bool recycle_unused_moves(solution_t<i_t, f_t, REQUEST>& sol,
                          move_candidates_t<i_t, f_t>& move_candidates)
{
  raft::common::nvtx::range fun_scope("recycle_unused_moves");
  auto& nodes_to_search  = move_candidates.nodes_to_search;
  constexpr bool recycle = true;
  bool nodes_remained    = nodes_to_search.sample_nodes_for_recycle(sol, move_candidates);
  if (!nodes_remained) { return false; }
  if (!find_vrp_moves(sol, move_candidates, recycle)) { return false; }
  bool move_found = select_and_execute_vrp_move(sol, move_candidates);
  return move_found;
}

template <typename i_t, typename f_t, request_t REQUEST>
bool perform_vrp_search(solution_t<i_t, f_t, REQUEST>& sol,
                        move_candidates_t<i_t, f_t>& move_candidates)
{
  raft::common::nvtx::range fun_scope("perform_vrp_search");
  cuopt_func_call(sol.check_cost_coherence(move_candidates.weights));
  if (!find_vrp_moves(sol, move_candidates)) { return false; }
  bool move_found = select_and_execute_vrp_move(sol, move_candidates);
  if (move_found) {
    // copy the current nodes to search beforehand, so sliding can search for it again
    auto copy_sampled_nodes = move_candidates.nodes_to_search.h_sampled_nodes;
    // do a single iteration as more iterations doesn't find more moves
    recycle_unused_moves(sol, move_candidates);
    move_candidates.nodes_to_search.h_sampled_nodes = copy_sampled_nodes;
    move_candidates.nodes_to_search.n_sampled_nodes = copy_sampled_nodes.size();
  }
  return move_found;
}

template bool perform_vrp_search<int, float, request_t::VRP>(
  solution_t<int, float, request_t::VRP>& sol, move_candidates_t<int, float>& move_candidates);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
