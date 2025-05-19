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

#pragma once

#include "../../solution/solution.cuh"
#include "../move_candidates/move_candidates.cuh"

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
DI thrust::pair<double, double> evaluate_cap_infeasibility(
  typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
  typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
  const typename route_t<i_t, f_t, REQUEST>::view_t& route_1,
  i_t start_idx_1,
  i_t frag_size_1,
  const typename route_t<i_t, f_t, REQUEST>::view_t& route_2,
  i_t start_idx_2,
  i_t frag_size_2,
  double excess_limit)
{
  auto computed_demand_relocating =
    route_2.get_node(start_idx_2 + frag_size_2).capacity_dim.max_to_node[0] -
    route_2.get_node(start_idx_2).capacity_dim.max_to_node[0];
  auto computed_demand_original =
    (route_1.get_node(start_idx_1 + frag_size_1).capacity_dim.max_to_node[0] -
     route_1.get_node(start_idx_1).capacity_dim.max_to_node[0]);
  auto diff   = computed_demand_relocating - computed_demand_original;
  auto excess = max(0,
                    route_1.get_node(route_1.get_num_nodes()).capacity_dim.max_to_node[0] + diff -
                      route_1.vehicle_info().capacities[0]);
  if (excess * move_candidates.weights[dim_t::CAP] > excess_limit) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }
  double inf_delta           = 0.;
  double selection_inf_delta = 0.;
  auto curr_inf              = route_1.get_infeasibility_cost();
  curr_inf[dim_t::CAP]       = excess;

  inf_delta = infeasible_cost_t::dot(
    move_candidates.weights,
    infeasible_cost_t::nominal_diff(curr_inf, route_1.get_infeasibility_cost()));

  selection_inf_delta = infeasible_cost_t::dot(
    move_candidates.selection_weights,
    infeasible_cost_t::nominal_diff(curr_inf, route_1.get_infeasibility_cost()));
  return {inf_delta, selection_inf_delta};
}

template <typename i_t, typename f_t, request_t REQUEST>
DI thrust::pair<double, double> evaluate_fragment(
  typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
  typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
  const typename route_t<i_t, f_t, REQUEST>::view_t& route_1,
  i_t start_idx_1,
  i_t frag_size_1,
  const typename route_t<i_t, f_t, REQUEST>::view_t& route_2,
  i_t start_idx_2,
  i_t frag_size_2,
  double excess_limit,
  bool reverse)
{
  // disallow empty routes
  if (frag_size_2 == 0 && route_1.get_num_service_nodes() == frag_size_1) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }

  if (!move_candidates.include_objective) { return {0, 0}; }
  // cost check
  double obj_delta = 0.;
  double all_forward_1 =
    route_1.get_node(start_idx_1 + 1 + frag_size_1).distance_dim.distance_forward -
    route_1.get_node(start_idx_1).distance_dim.distance_forward;
  if (frag_size_2 == 0) {
    auto direct = get_arc_of_dimension<i_t, f_t, dim_t::DIST>(
      route_1.get_node(start_idx_1).node_info(),
      route_1.get_node(start_idx_1 + 1 + frag_size_1).node_info(),
      route_1.vehicle_info());
    return {direct - all_forward_1, direct - all_forward_1};
  }

  if (!reverse) {
    double sd1_sd2_1 =
      get_arc_of_dimension<i_t, f_t, dim_t::DIST>(route_1.get_node(start_idx_1).node_info(),
                                                  route_2.get_node(start_idx_2 + 1).node_info(),
                                                  route_1.vehicle_info());

    double end_node_2_end_node_1 = get_arc_of_dimension<i_t, f_t, dim_t::DIST>(
      route_2.get_node(start_idx_2 + frag_size_2).node_info(),
      route_1.get_node(start_idx_1 + frag_size_1 + 1).node_info(),
      route_1.vehicle_info());
    double frag_dist = route_2.get_node(start_idx_2 + frag_size_2).distance_dim.distance_forward -
                       route_2.get_node(start_idx_2 + 1).distance_dim.distance_forward;
    obj_delta = sd1_sd2_1 + frag_dist + end_node_2_end_node_1 - all_forward_1;
  } else {
    double sd1_end_frag_2 = get_arc_of_dimension<i_t, f_t, dim_t::DIST>(
      route_1.get_node(start_idx_1).node_info(),
      route_2.get_node(start_idx_2 + frag_size_2).node_info(),
      route_1.vehicle_info());

    double sd2_1_end_node_1 = get_arc_of_dimension<i_t, f_t, dim_t::DIST>(
      route_2.get_node(start_idx_2 + 1).node_info(),
      route_1.get_node(start_idx_1 + frag_size_1 + 1).node_info(),
      route_1.vehicle_info());
    double frag_dist =
      route_2.dimensions.distance_dim.reverse_distance[(start_idx_2 + 1)] -
      route_2.dimensions.distance_dim.reverse_distance[(start_idx_2 + frag_size_2)];
    obj_delta = sd1_end_frag_2 + frag_dist + sd2_1_end_node_1 - all_forward_1;
  }

  return {obj_delta, obj_delta};
}

template <typename i_t, typename f_t, request_t REQUEST>
DI thrust::pair<double, double> evaluate_fragment(
  typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
  typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
  const typename route_t<i_t, f_t, REQUEST>::view_t& route,
  i_t start_idx,
  i_t end_idx,
  i_t frag_size,
  typename dimensions_route_t<i_t, f_t, REQUEST>::view_t const& fragment,
  double excess_limit,
  i_t frag_start = 0)
{
  // TODO create forward/backward nodes and only get forward node
  auto temp_node = route.get_node(start_idx);
  for (i_t i = frag_start; i < frag_start + frag_size; ++i) {
    auto next_node = fragment.get_node(i);
    if (!next_node.node_info().is_service_node()) {
      return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    }
    temp_node.calculate_forward_all(next_node, route.vehicle_info());
    if (!next_node.forward_feasible(route.vehicle_info(), move_candidates.weights, excess_limit)) {
      return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    }
    temp_node = next_node;
  }

  // check if all the service nodes are being ejected and disallow such moves so that we don't end
  // up with an empty route
  if (frag_size == 0 && route.get_num_service_nodes() == end_idx - start_idx - 1) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }
  auto end_node = route.get_node(end_idx);
  double delta  = temp_node.calculate_forward_all_and_delta(end_node,
                                                           route.vehicle_info(),
                                                           move_candidates.include_objective,
                                                           move_candidates.weights,
                                                           route.get_objective_cost(),
                                                           route.get_infeasibility_cost());
  if (!end_node.feasible(route.vehicle_info(), move_candidates.weights, excess_limit)) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }
  double selection_delta =
    temp_node.calculate_forward_all_and_delta(end_node,
                                              route.vehicle_info(),
                                              move_candidates.include_objective,
                                              move_candidates.selection_weights,
                                              route.get_objective_cost(),
                                              route.get_infeasibility_cost());
  // if move insertion is feasible compute cost with alpha and beta too and return the pair
  return {delta, selection_delta};
}
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
