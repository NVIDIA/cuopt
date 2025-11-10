/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#include <utilities/cuda_helpers.cuh>
#include "../solution/solution.cuh"
#include "set_nodes_data.cuh"

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
__global__ void set_initial_nodes_kernel(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                                         const typename problem_t<i_t, f_t>::view_t problem,
                                         const i_t* map_indices)
{
  auto th                = threadIdx.x + blockIdx.x * blockDim.x;
  const auto& order_info = problem.order_info;
  const auto& fleet_info = problem.fleet_info;

  if (th < solution.n_routes) {
    auto& curr_route = solution.routes[th];

    i_t vehicle_id              = curr_route.get_vehicle_id();
    auto start_depot_node_info  = problem.get_start_depot_node_info(vehicle_id);
    auto return_depot_node_info = problem.get_return_depot_node_info(vehicle_id);

    auto start_depot_node = create_depot_node<i_t, f_t, REQUEST>(
      problem, start_depot_node_info, return_depot_node_info, vehicle_id);

    auto return_depot_node = create_depot_node<i_t, f_t, REQUEST>(
      problem, return_depot_node_info, start_depot_node_info, vehicle_id);

    i_t earliest, latest;
    if (order_info.depot_included) {
      earliest = max(problem.order_info.earliest_time[DEPOT], fleet_info.earliest_time[vehicle_id]);
      latest   = min(problem.order_info.latest_time[DEPOT], fleet_info.latest_time[vehicle_id]);
    } else {
      earliest = fleet_info.earliest_time[vehicle_id];
      latest   = fleet_info.latest_time[vehicle_id];
    }

    if (th < solution.get_num_requests()) {
      auto idx        = map_indices[th];
      auto request_id = solution.get_request(idx);

      auto node_info = NodeInfo<i_t>(
        request_id.id(), order_info.get_order_location(request_id.id()), node_type_t::DELIVERY);

      auto node = create_node<i_t, f_t, REQUEST>(problem, node_info, node_info);

      solution.route_node_map.set_route_id_and_intra_idx(node_info, th, 1);

      curr_route.set_node(0, start_depot_node);
      curr_route.set_node(1, node);
      curr_route.set_node(2, return_depot_node);
      curr_route.set_num_nodes(2);
      curr_route.set_id(th);
      if (problem.dimensions_info.has_dimension(dim_t::TIME)) {
        auto& time_route = curr_route.template get_dim<dim_t::TIME>();
        // set DEPOT nodes departure backward to latest
        time_route.departure_backward[2] = latest;
        time_route.excess_backward[2]    = 0.f;
        time_route.departure_forward[0]  = earliest;
        time_route.excess_forward[0]     = 0.f;
        if (time_route.dim_info.should_compute_travel_time()) {
          time_route.latest_arrival_forward[0]    = latest;
          time_route.earliest_arrival_backward[2] = earliest;
        }
      }
    } else {
      curr_route.set_node(0, start_depot_node);
      curr_route.set_node(1, return_depot_node);
      curr_route.set_num_nodes(1);
      curr_route.set_id(th);
      if (problem.dimensions_info.has_dimension(dim_t::TIME)) {
        auto& time_route = curr_route.template get_dim<dim_t::TIME>();
        // set DEPOT nodes departure backward to latest
        time_route.departure_backward[1] = latest;
        time_route.excess_backward[1]    = 0.f;
        time_route.departure_forward[0]  = earliest;
        time_route.excess_forward[0]     = 0.f;
        if (time_route.dim_info.should_compute_travel_time()) {
          time_route.latest_arrival_forward[0]    = latest;
          time_route.earliest_arrival_backward[1] = earliest;
        }
      }
    }
  }
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
__global__ void set_initial_nodes_kernel(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                                         const typename problem_t<i_t, f_t>::view_t problem,
                                         const i_t* map_indices)
{
  auto th                = threadIdx.x + blockIdx.x * blockDim.x;
  const auto& order_info = problem.order_info;
  const auto& fleet_info = problem.fleet_info;
  if (th < solution.n_routes) {
    auto& curr_route = solution.routes[th];

    i_t vehicle_id              = curr_route.get_vehicle_id();
    auto start_depot_node_info  = problem.get_start_depot_node_info(vehicle_id);
    auto return_depot_node_info = problem.get_return_depot_node_info(vehicle_id);
    auto start_depot_node       = create_depot_node<i_t, f_t, REQUEST>(
      problem, start_depot_node_info, return_depot_node_info, vehicle_id);

    auto return_depot_node = create_depot_node<i_t, f_t, REQUEST>(
      problem, return_depot_node_info, start_depot_node_info, vehicle_id);

    i_t earliest, latest;
    if (order_info.depot_included) {
      earliest = max(problem.order_info.earliest_time[DEPOT], fleet_info.earliest_time[vehicle_id]);
      latest   = min(problem.order_info.latest_time[DEPOT], fleet_info.latest_time[vehicle_id]);
    } else {
      earliest = fleet_info.earliest_time[vehicle_id];
      latest   = fleet_info.latest_time[vehicle_id];
    }

    if (th < solution.get_num_requests()) {
      auto idx          = map_indices[th];
      auto pickup_idx   = problem.pickup_indices[idx];
      auto delivery_idx = problem.delivery_indices[idx];
      auto pickup_node_info =
        NodeInfo<i_t>(pickup_idx, order_info.get_order_location(pickup_idx), node_type_t::PICKUP);
      auto delivery_node_info = NodeInfo<i_t>(
        delivery_idx, order_info.get_order_location(delivery_idx), node_type_t::DELIVERY);

      auto pickup_node =
        create_node<i_t, f_t, REQUEST>(problem, pickup_node_info, delivery_node_info);
      auto delivery_node =
        create_node<i_t, f_t, REQUEST>(problem, delivery_node_info, pickup_node_info);

      solution.route_node_map.set_route_id_and_intra_idx(pickup_node_info, th, 1);
      solution.route_node_map.set_route_id_and_intra_idx(delivery_node_info, th, 2);

      curr_route.set_node(0, start_depot_node);
      curr_route.set_node(1, pickup_node);
      curr_route.set_node(2, delivery_node);
      curr_route.set_node(3, return_depot_node);
      curr_route.set_num_nodes(3);
      curr_route.set_id(th);
      if (problem.dimensions_info.has_dimension(dim_t::TIME)) {
        auto& time_route = curr_route.template get_dim<dim_t::TIME>();
        // set DEPOT nodes departure backward to latest
        time_route.departure_backward[3] = latest;
        time_route.excess_backward[3]    = 0.f;
        time_route.departure_forward[0]  = earliest;
        time_route.excess_forward[0]     = 0.f;
        if (time_route.dim_info.should_compute_travel_time()) {
          time_route.latest_arrival_forward[0]    = latest;
          time_route.earliest_arrival_backward[3] = earliest;
        }
      }
    } else {
      curr_route.set_node(0, start_depot_node);
      curr_route.set_node(1, return_depot_node);
      curr_route.set_num_nodes(1);
      curr_route.set_id(th);
      if (problem.dimensions_info.has_dimension(dim_t::TIME)) {
        auto& time_route = curr_route.template get_dim<dim_t::TIME>();
        // set DEPOT nodes departure backward to latest
        time_route.departure_backward[1] = latest;
        time_route.excess_backward[1]    = 0.f;
        time_route.departure_forward[0]  = earliest;
        time_route.excess_forward[0]     = 0.f;
        if (time_route.dim_info.should_compute_travel_time()) {
          time_route.latest_arrival_forward[0]    = latest;
          time_route.earliest_arrival_backward[1] = earliest;
        }
      }
    }
  }
}

// sets the node data of the route that has the node_ids
template <typename i_t, typename f_t, request_t REQUEST>
__global__ void set_nodes_data_of_route_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  const typename problem_t<i_t, f_t>::view_t problem,
  i_t route_id)
{
  set_nodes_data_of_single_route<i_t, f_t, REQUEST>(solution, problem, route_id);
}

// sets the node data of the route that has the node_ids
template <typename i_t, typename f_t, request_t REQUEST>
__global__ void set_nodes_data_of_solution_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  const typename problem_t<i_t, f_t>::view_t problem)
{
  auto route_id = blockIdx.x;
  set_nodes_data_of_single_route<i_t, f_t, REQUEST>(solution, problem, route_id);
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void set_nodes_data_of_new_routes_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  const typename problem_t<i_t, f_t>::view_t problem,
  i_t starting_route_id)
{
  auto route_id = starting_route_id + blockIdx.x;
  set_nodes_data_of_single_route<i_t, f_t, REQUEST>(solution, problem, route_id);
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::set_initial_nodes(const rmm::device_uvector<i_t>& d_indices,
                                                      i_t desired_n_routes)
{
  thrust::fill(sol_handle->get_thrust_policy(),
               route_node_map.route_id_per_node.begin(),
               route_node_map.route_id_per_node.end(),
               -1);
  thrust::fill(sol_handle->get_thrust_policy(),
               route_node_map.intra_route_idx_per_node.begin(),
               route_node_map.intra_route_idx_per_node.end(),
               -1);
  constexpr i_t TPB = 32;
  i_t n_blocks      = (desired_n_routes + TPB - 1) / TPB;
  set_initial_nodes_kernel<i_t, f_t, REQUEST>
    <<<n_blocks, TPB, 0, sol_handle->get_stream()>>>(view(), problem_ptr->view(), d_indices.data());

  sol_handle->get_stream().synchronize();
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::set_nodes_data_of_solution()
{
  constexpr i_t TPB = 32;
  i_t n_blocks      = n_routes;
  set_nodes_data_of_solution_kernel<i_t, f_t, REQUEST>
    <<<n_blocks, TPB, 0, sol_handle->get_stream()>>>(view(), problem_ptr->view());
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::set_nodes_data_of_route(i_t route_id)
{
  constexpr i_t TPB = 32;
  set_nodes_data_of_route_kernel<i_t, f_t, REQUEST>
    <<<1, TPB, 0, sol_handle->get_stream()>>>(view(), problem_ptr->view(), route_id);
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::set_nodes_data_of_new_routes(i_t added_routes,
                                                                 i_t prev_route_size)
{
  constexpr i_t TPB     = 32;
  i_t starting_route_id = prev_route_size;
  set_nodes_data_of_new_routes_kernel<i_t, f_t, REQUEST>
    <<<added_routes, TPB, 0, sol_handle->get_stream()>>>(
      view(), problem_ptr->view(), starting_route_id);
}

template class solution_t<int, float, request_t::PDP>;
template class solution_t<int, float, request_t::VRP>;

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
