/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/cuda_helpers.cuh>
#include "../solution/solution.cuh"

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
__device__ void set_nodes_data_of_single_route(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  const typename problem_t<i_t, f_t>::view_t problem,
  i_t route_id)
{
  auto& route = solution.routes[route_id];
  set_nodes_data_of_single_route<i_t, f_t, REQUEST>(solution, problem, route);
}

template <typename i_t, typename f_t, request_t REQUEST>
__device__ void set_route_data(typename problem_t<i_t, f_t>::view_t const& problem,
                               typename route_t<i_t, f_t, REQUEST>::view_t& route)

{
  const auto& order_info = problem.order_info;
  const auto& fleet_info = problem.fleet_info;
  i_t n_nodes_route      = route.get_num_nodes();
  const i_t vehicle_id   = route.get_vehicle_id();
  if (threadIdx.x == 0) {
    i_t earliest, latest;
    if (order_info.depot_included) {
      earliest = max(problem.order_info.earliest_time[DEPOT], fleet_info.earliest_time[vehicle_id]);
      latest   = min(problem.order_info.latest_time[DEPOT], fleet_info.latest_time[vehicle_id]);
    } else {
      earliest = fleet_info.earliest_time[vehicle_id];
      latest   = fleet_info.latest_time[vehicle_id];
    }
    if (problem.dimensions_info.has_dimension(dim_t::TIME)) {
      auto& time_route                             = route.template get_dim<dim_t::TIME>();
      time_route.departure_backward[n_nodes_route] = latest;
      time_route.departure_forward[0]              = earliest;
      time_route.excess_forward[0]                 = 0.f;
      time_route.excess_backward[n_nodes_route]    = 0.f;
      if (time_route.dim_info.should_compute_travel_time()) {
        time_route.latest_arrival_forward[0]                = latest;
        time_route.earliest_arrival_backward[n_nodes_route] = earliest;
      }
      cuopt_assert(abs(route.template get_dim<dim_t::TIME>().excess_forward[route.get_num_nodes()] -
                       route.template get_dim<dim_t::TIME>().excess_backward[0]) < 0.0001,
                   "Backward forward mismatch!");
    }
    route.template get_dim<dim_t::DIST>().distance_backward[n_nodes_route] = 0.f;
    route.template get_dim<dim_t::DIST>().distance_forward[0]              = 0.f;
    if (problem.dimensions_info.has_dimension(dim_t::CAP)) {
      route.template get_dim<dim_t::CAP>().max_to_node[0]           = 0;
      route.template get_dim<dim_t::CAP>().gathered[0]              = 0;
      route.template get_dim<dim_t::CAP>().max_after[n_nodes_route] = 0;
    }
  }
}

// sets the node data of the route that has the node_ids
template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
__device__ void set_nodes_data_of_single_route(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  const typename problem_t<i_t, f_t>::view_t problem,
  typename route_t<i_t, f_t, REQUEST>::view_t& route)
{
  const auto& order_info = problem.order_info;
  const auto& fleet_info = problem.fleet_info;
  i_t n_nodes_route      = route.get_num_nodes();
  const i_t vehicle_id   = route.get_vehicle_id();
  const i_t route_id     = route.get_id();
  auto request_size      = request_info_t<i_t, REQUEST>::size();
  cuopt_assert(route.get_id() == route_id, "Route id mismatch!");
  for (i_t i = threadIdx.x; i <= n_nodes_route; i += blockDim.x) {
    NodeInfo<> node_info;

    if (i == 0) {
      node_info = problem.get_start_depot_node_info(vehicle_id);
    } else if (i == n_nodes_route) {
      node_info = problem.get_return_depot_node_info(vehicle_id);
    } else {
      i_t node_id       = route.node_id(i);
      i_t node_location = order_info.get_order_location(node_id);
      node_info         = NodeInfo<>{node_id, node_location, node_type_t::DELIVERY};
    }
    auto brother_info = node_info;
    auto curr_node =
      node_info.is_depot()
        ? create_depot_node<i_t, f_t, REQUEST>(problem, node_info, brother_info, vehicle_id)
        : create_node<i_t, f_t, REQUEST>(problem, node_info, brother_info);
    route.set_node(i, curr_node);
    solution.route_node_map.set_route_id_and_intra_idx(node_info, route_id, i);
  }
  __syncthreads();
  // create_depot_node only sets fixed route data. Here, we initialize backward and forward data for
  // depots
  set_route_data<i_t, f_t, REQUEST>(problem, route);
}

// sets the node data of the route that has the node_ids
template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
__device__ void set_nodes_data_of_single_route(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  const typename problem_t<i_t, f_t>::view_t problem,
  typename route_t<i_t, f_t, REQUEST>::view_t& route)
{
  const auto& order_info = problem.order_info;
  const auto& fleet_info = problem.fleet_info;
  i_t n_nodes_route      = route.get_num_nodes();
  const i_t vehicle_id   = route.get_vehicle_id();
  const i_t route_id     = route.get_id();
  // we might add incomplete pdp solutions from recombiners
  cuopt_assert(route.get_id() == route_id, "Route id mismatch!");
  for (i_t i = threadIdx.x; i <= n_nodes_route; i += blockDim.x) {
    NodeInfo<> node_info, brother_info;

    if (i == 0) {
      node_info    = problem.get_start_depot_node_info(vehicle_id);
      brother_info = problem.get_return_depot_node_info(vehicle_id);
    } else if (i == n_nodes_route) {
      node_info    = problem.get_return_depot_node_info(vehicle_id);
      brother_info = problem.get_start_depot_node_info(vehicle_id);
    } else {
      i_t node_id    = route.node_id(i);
      i_t brother_id = order_info.pair_indices[node_id];
      node_type_t node_type =
        order_info.is_pickup_index[node_id] ? node_type_t::PICKUP : node_type_t::DELIVERY;
      node_type_t brother_type =
        node_type == node_type_t::PICKUP ? node_type_t::DELIVERY : node_type_t::PICKUP;
      i_t node_location    = order_info.get_order_location(node_id);
      i_t brother_location = order_info.get_order_location(brother_id);
      node_info            = NodeInfo<i_t>(node_id, node_location, node_type);
      brother_info         = NodeInfo<i_t>(brother_id, brother_location, brother_type);
    }

    auto curr_node =
      node_info.is_depot()
        ? create_depot_node<i_t, f_t, REQUEST>(problem, node_info, brother_info, vehicle_id)
        : create_node<i_t, f_t, REQUEST>(problem, node_info, brother_info);
    route.set_node(i, curr_node);
    solution.route_node_map.set_route_id_and_intra_idx(node_info, route_id, i);
  }
  __syncthreads();
  set_route_data<i_t, f_t, REQUEST>(problem, route);
}

/**
 * @brief All break nodes are pre-allocated and created in problem.cu
 * Changing the vehicle id and depots is not sufficient to evaluate the new routes.
 * When breaks are set. We need to retrieve the correct break node based on the new vehicle id.
 *
 */
template <typename i_t, typename f_t, request_t REQUEST>
DI void set_break_node_info(typename problem_t<i_t, f_t>::view_t const& problem,
                            typename route_t<i_t, f_t, REQUEST>::view_t& route)
{
  if (!problem.dimensions_info.has_dimension(dim_t::BREAK)) { return; }

  for (auto i = threadIdx.x; i < route.get_num_nodes(); i += blockDim.x) {
    auto node = route.get_node(i);
    if (node.node_info().is_break()) {
      auto node_break_dim = node.node_info().node();
      auto break_node_id  = problem.special_nodes.get_break_loc_idx(node.node_info());
      auto break_nodes    = problem.special_nodes.subset(route.get_vehicle_id(), node_break_dim);
      auto new_break_node =
        create_break_node<i_t, f_t, REQUEST>(break_nodes, break_node_id, problem.dimensions_info);
      route.set_node(i, new_break_node);
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
DI void set_route_info(typename problem_t<i_t, f_t>::view_t const& problem,
                       typename route_t<i_t, f_t, REQUEST>::view_t& route)
{
  const i_t vehicle_id = route.get_vehicle_id();
  NodeInfo<> node_info, brother_info;
  if (threadIdx.x == 0) {
    node_info = problem.get_start_depot_node_info(vehicle_id);
    if constexpr (REQUEST == request_t::PDP) {
      brother_info = problem.get_return_depot_node_info(vehicle_id);
    } else {
      brother_info = node_info;
    }
    auto start_depot_node =
      create_depot_node<i_t, f_t, REQUEST>(problem, node_info, brother_info, vehicle_id);
    route.set_node(0, start_depot_node);
  } else if (threadIdx.x == blockDim.x - 1) {
    node_info = problem.get_return_depot_node_info(vehicle_id);
    if constexpr (REQUEST == request_t::PDP) {
      brother_info = problem.get_start_depot_node_info(vehicle_id);
    } else {
      brother_info = node_info;
    }
    auto end_depot_node =
      create_depot_node<i_t, f_t, REQUEST>(problem, node_info, brother_info, vehicle_id);
    route.set_node(route.get_num_nodes(), end_depot_node);
  }
  __syncthreads();
  set_route_data<i_t, f_t, REQUEST>(problem, route);
}

template <typename i_t, typename f_t, request_t REQUEST>
DI void reset_vehicle_id(typename problem_t<i_t, f_t>::view_t const& problem,
                         typename route_t<i_t, f_t, REQUEST>::view_t& route,
                         i_t vehicle_id)
{
  if (threadIdx.x == 0) { route.set_vehicle_id(vehicle_id); }
  __syncthreads();
  set_route_info<i_t, f_t, REQUEST>(problem, route);
  __syncthreads();
  set_break_node_info<i_t, f_t, REQUEST>(problem, route);
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
