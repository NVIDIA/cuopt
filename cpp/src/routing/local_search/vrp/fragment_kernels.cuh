/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <thrust/pair.h>
#include "../../solution/solution.cuh"
#include "../move_candidates/move_candidates.cuh"

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
DI thrust::pair<double, double> compute_distance_delta_from_totals(
  typename move_candidates_t<i_t, f_t>::view_t const& move_candidates,
  const typename route_t<i_t, f_t, REQUEST>::view_t& route,
  double new_total_cost,
  double new_total_distance)
{
  auto new_obj_cost = route.get_objective_cost();
  auto new_inf_cost = route.get_infeasibility_cost();

  if (route.vehicle_info().has_distance_tiers()) {
    const auto old_total_cost     = route.get_node(route.get_num_nodes()).cost_dim.cost_forward;
    const auto old_total_distance = route.get_node(route.get_num_nodes()).cost_dim.distance_forward;
    new_obj_cost[objective_t::COST] = route.vehicle_info().compute_distance_cost_from_delta(
      old_total_distance,
      old_total_cost,
      route.get_objective_cost()[objective_t::COST],
      new_total_distance,
      new_total_cost,
      route.get_active_distance_tier());
  } else {
    new_obj_cost[objective_t::COST] = new_total_cost;
  }
  new_inf_cost[dim_t::COST] =
    route.template get_dim<dim_t::COST>().dim_info.has_max_constraint
      ? route.vehicle_info().compute_distance_excess(new_total_distance) +
          max(0., new_obj_cost[objective_t::COST] - route.vehicle_info().max_cost)
      : route.vehicle_info().compute_distance_excess(new_total_distance);

  double delta = infeasible_cost_t::dot(
    move_candidates.weights,
    infeasible_cost_t::nominal_diff(new_inf_cost, route.get_infeasibility_cost()));
  double selection_delta = infeasible_cost_t::dot(
    move_candidates.selection_weights,
    infeasible_cost_t::nominal_diff(new_inf_cost, route.get_infeasibility_cost()));

  if (move_candidates.include_objective) {
    auto obj_weights = route.dimensions_info().objective_weights;
    delta += objective_cost_t::dot(obj_weights, new_obj_cost - route.get_objective_cost());
    selection_delta +=
      objective_cost_t::dot(obj_weights, new_obj_cost - route.get_objective_cost());
  }

  return {delta, selection_delta};
}

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

  double cost_delta       = 0.;
  double distance_delta   = 0.;
  double all_forward_cost = route_1.get_node(start_idx_1 + 1 + frag_size_1).cost_dim.cost_forward -
                            route_1.get_node(start_idx_1).cost_dim.cost_forward;
  double all_forward_distance =
    route_1.get_node(start_idx_1 + 1 + frag_size_1).cost_dim.distance_forward -
    route_1.get_node(start_idx_1).cost_dim.distance_forward;
  if (frag_size_2 == 0) {
    auto direct_cost = get_arc_cost(route_1.get_node(start_idx_1).node_info(),
                                    route_1.get_node(start_idx_1 + 1 + frag_size_1).node_info(),
                                    route_1.vehicle_info());
    auto direct_distance =
      get_travel_distance(route_1.get_node(start_idx_1).node_info(),
                          route_1.get_node(start_idx_1 + 1 + frag_size_1).node_info(),
                          route_1.vehicle_info());
    auto new_total_cost = route_1.get_node(route_1.get_num_nodes()).cost_dim.cost_forward +
                          (direct_cost - all_forward_cost);
    auto new_total_distance = route_1.get_node(route_1.get_num_nodes()).cost_dim.distance_forward +
                              (direct_distance - all_forward_distance);
    return compute_distance_delta_from_totals<i_t, f_t, REQUEST>(
      move_candidates, route_1, new_total_cost, new_total_distance);
  }

  if (!reverse) {
    double sd1_sd2_1_cost     = get_arc_cost(route_1.get_node(start_idx_1).node_info(),
                                         route_2.get_node(start_idx_2 + 1).node_info(),
                                         route_1.vehicle_info());
    double sd1_sd2_1_distance = get_travel_distance(route_1.get_node(start_idx_1).node_info(),
                                                    route_2.get_node(start_idx_2 + 1).node_info(),
                                                    route_1.vehicle_info());

    double end_node_2_end_node_1_cost =
      get_arc_cost(route_2.get_node(start_idx_2 + frag_size_2).node_info(),
                   route_1.get_node(start_idx_1 + frag_size_1 + 1).node_info(),
                   route_1.vehicle_info());
    double end_node_2_end_node_1_distance =
      get_travel_distance(route_2.get_node(start_idx_2 + frag_size_2).node_info(),
                          route_1.get_node(start_idx_1 + frag_size_1 + 1).node_info(),
                          route_1.vehicle_info());
    double frag_cost = route_2.get_node(start_idx_2 + frag_size_2).cost_dim.cost_forward -
                       route_2.get_node(start_idx_2 + 1).cost_dim.cost_forward;
    double frag_distance = route_2.get_node(start_idx_2 + frag_size_2).cost_dim.distance_forward -
                           route_2.get_node(start_idx_2 + 1).cost_dim.distance_forward;
    cost_delta = sd1_sd2_1_cost + frag_cost + end_node_2_end_node_1_cost - all_forward_cost;
    distance_delta =
      sd1_sd2_1_distance + frag_distance + end_node_2_end_node_1_distance - all_forward_distance;
  } else {
    double sd1_end_frag_2_cost =
      get_arc_cost(route_1.get_node(start_idx_1).node_info(),
                   route_2.get_node(start_idx_2 + frag_size_2).node_info(),
                   route_1.vehicle_info());
    double sd1_end_frag_2_distance =
      get_travel_distance(route_1.get_node(start_idx_1).node_info(),
                          route_2.get_node(start_idx_2 + frag_size_2).node_info(),
                          route_1.vehicle_info());

    double sd2_1_end_node_1_cost =
      get_arc_cost(route_2.get_node(start_idx_2 + 1).node_info(),
                   route_1.get_node(start_idx_1 + frag_size_1 + 1).node_info(),
                   route_1.vehicle_info());
    double sd2_1_end_node_1_distance =
      get_travel_distance(route_2.get_node(start_idx_2 + 1).node_info(),
                          route_1.get_node(start_idx_1 + frag_size_1 + 1).node_info(),
                          route_1.vehicle_info());
    double frag_cost = route_2.dimensions.cost_dim.reverse_cost[(start_idx_2 + 1)] -
                       route_2.dimensions.cost_dim.reverse_cost[(start_idx_2 + frag_size_2)];
    double frag_distance =
      route_2.dimensions.cost_dim.reverse_distance[(start_idx_2 + 1)] -
      route_2.dimensions.cost_dim.reverse_distance[(start_idx_2 + frag_size_2)];
    cost_delta = sd1_end_frag_2_cost + frag_cost + sd2_1_end_node_1_cost - all_forward_cost;
    distance_delta =
      sd1_end_frag_2_distance + frag_distance + sd2_1_end_node_1_distance - all_forward_distance;
  }

  auto new_total_cost =
    route_1.get_node(route_1.get_num_nodes()).cost_dim.cost_forward + cost_delta;
  auto new_total_distance =
    route_1.get_node(route_1.get_num_nodes()).cost_dim.distance_forward + distance_delta;
  return compute_distance_delta_from_totals<i_t, f_t, REQUEST>(
    move_candidates, route_1, new_total_cost, new_total_distance);
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
