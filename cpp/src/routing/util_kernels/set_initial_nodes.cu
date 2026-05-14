/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#include <utilities/cuda_helpers.cuh>
#include "../solution/solution.cuh"
#include "set_nodes_data.cuh"

#include <algorithm>
#include <numeric>

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

// ---------------------------------------------------------------------------
// Greedy qtime-priority initialisation (VRP only).
//
// Each block handles one route.  Thread 0 sequentially places k service nodes
// (the k requests assigned to this route) in the order they appear in the flat
// assignment array (already sorted by due-time / weight on the host).
// Routes with no assigned requests are initialised as depot-only.
// ---------------------------------------------------------------------------
template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
__global__ void due_time_greedy_init_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  const typename problem_t<i_t, f_t>::view_t problem,
  const i_t* assignments,  // flat: [route_id * max_per_route + slot] = request_idx
  const i_t* n_per_route,  // number of requests assigned to each route
  i_t max_per_route)
{
  i_t route_id = blockIdx.x;
  if (route_id >= solution.n_routes || threadIdx.x != 0) return;

  auto& curr_route            = solution.routes[route_id];
  i_t vehicle_id              = curr_route.get_vehicle_id();
  const auto& order_info      = problem.order_info;
  const auto& fleet_info      = problem.fleet_info;
  auto start_depot_node_info  = problem.get_start_depot_node_info(vehicle_id);
  auto return_depot_node_info = problem.get_return_depot_node_info(vehicle_id);
  auto start_depot_node       = create_depot_node<i_t, f_t, REQUEST>(
    problem, start_depot_node_info, return_depot_node_info, vehicle_id);
  auto return_depot_node = create_depot_node<i_t, f_t, REQUEST>(
    problem, return_depot_node_info, start_depot_node_info, vehicle_id);

  i_t earliest, latest;
  if (order_info.depot_included) {
    earliest = max(order_info.earliest_time[DEPOT], fleet_info.earliest_time[vehicle_id]);
    latest   = min(order_info.latest_time[DEPOT], fleet_info.latest_time[vehicle_id]);
  } else {
    earliest = fleet_info.earliest_time[vehicle_id];
    latest   = fleet_info.latest_time[vehicle_id];
  }

  i_t k = n_per_route[route_id];  // service nodes assigned to this route
  curr_route.set_node(0, start_depot_node);
  curr_route.set_id(route_id);

  for (i_t slot = 0; slot < k; ++slot) {
    i_t req_idx     = assignments[route_id * max_per_route + slot];
    auto request_id = solution.get_request(req_idx);
    i_t order_id    = request_id.id();
    auto node_info =
      NodeInfo<i_t>(order_id, order_info.get_order_location(order_id), node_type_t::DELIVERY);
    auto node = create_node<i_t, f_t, REQUEST>(problem, node_info, node_info);
    curr_route.set_node(slot + 1, node);
    solution.route_node_map.set_route_id_and_intra_idx(node_info, route_id, slot + 1);
  }

  // return depot sits at position k+1; n_nodes = k+1 (start depot + k service nodes)
  i_t n_nodes = k + 1;
  curr_route.set_node(n_nodes, return_depot_node);
  curr_route.set_num_nodes(n_nodes);

  if (problem.dimensions_info.has_dimension(dim_t::TIME)) {
    auto& time_route                       = curr_route.template get_dim<dim_t::TIME>();
    time_route.departure_backward[n_nodes] = latest;
    time_route.excess_backward[n_nodes]    = 0.f;
    time_route.departure_forward[0]        = earliest;
    time_route.excess_forward[0]           = 0.f;
    if (time_route.dim_info.should_compute_travel_time()) {
      time_route.latest_arrival_forward[0]          = latest;
      time_route.earliest_arrival_backward[n_nodes] = earliest;
    }
  }
}

// PDP stub — not implemented; falls back to random_init_routes.
template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
__global__ void due_time_greedy_init_kernel(typename solution_t<i_t, f_t, REQUEST>::view_t,
                                            const typename problem_t<i_t, f_t>::view_t,
                                            const i_t*,
                                            const i_t*,
                                            i_t)
{
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::due_time_greedy_init_routes()
{
  raft::common::nvtx::range fun_scope("due_time_greedy_init_routes");
  auto stream = sol_handle->get_stream();
  stream.synchronize();

  // PDP is not yet supported; fall back to random init.
  if constexpr (REQUEST == request_t::PDP) {
    random_init_routes();
    return;
  }

  i_t n_requests         = get_num_requests();
  const auto& order_info = problem_ptr->order_info;

  bool has_due_times = !order_info.v_order_due_times_.is_empty();
  bool has_weights   = !order_info.v_order_weights_.is_empty();

  // Copy due-times and weights to host (small arrays — lot counts, not fleet).
  std::vector<i_t> h_due_times, h_weights;
  if (has_due_times) {
    h_due_times.resize(order_info.v_order_due_times_.size());
    raft::copy(
      h_due_times.data(), order_info.v_order_due_times_.data(), h_due_times.size(), stream);
  }
  if (has_weights) {
    h_weights.resize(order_info.v_order_weights_.size());
    raft::copy(h_weights.data(), order_info.v_order_weights_.data(), h_weights.size(), stream);
  }
  stream.synchronize();

  // Sort request indices: smallest due-time first; break ties by largest weight.
  std::vector<i_t> sorted(n_requests);
  std::iota(sorted.begin(), sorted.end(), 0);
  if (has_due_times || has_weights) {
    // For VRP, request_idx → order_id = idx + depot_included_offset.
    i_t depot_offset = order_info.depot_included_ ? 1 : 0;
    std::stable_sort(sorted.begin(), sorted.end(), [&](i_t a, i_t b) {
      i_t oa = a + depot_offset;
      i_t ob = b + depot_offset;
      i_t da = has_due_times ? h_due_times[oa] : 0;
      i_t db = has_due_times ? h_due_times[ob] : 0;
      i_t wa = has_weights ? h_weights[oa] : 0;
      i_t wb = has_weights ? h_weights[ob] : 0;
      if (da != db) return da < db;  // tighter due-time first
      return wa > wb;                // heavier weight first on tie
    });
  }

  // Round-robin assignment: sorted request i → route (i % n_routes).
  i_t max_per_route = (n_requests + n_routes - 1) / n_routes;
  std::vector<i_t> h_assignments(n_routes * max_per_route, -1);
  std::vector<i_t> h_n_per_route(n_routes, 0);

  for (i_t i = 0; i < n_requests; ++i) {
    i_t r                                   = i % n_routes;
    i_t slot                                = h_n_per_route[r]++;
    h_assignments[r * max_per_route + slot] = sorted[i];
  }

  // Resize route buffers if max_per_route + 2 (depots) exceeds current capacity.
  i_t needed_capacity = max_per_route + 2;
  if (needed_capacity > max_nodes_per_route) {
    resize_routes(raft::alignTo(needed_capacity, base_route_size));
  }

  // Initialise route n_nodes on device before launching the kernel.
  for (i_t r = 0; r < n_routes; ++r) {
    i_t n_nodes = h_n_per_route[r] + 1;  // start-depot + service nodes
    routes[r].n_nodes.set_value_async(n_nodes, stream);
  }

  rmm::device_uvector<i_t> d_assignments(h_assignments.size(), stream);
  rmm::device_uvector<i_t> d_n_per_route(h_n_per_route.size(), stream);
  raft::copy(d_assignments.data(), h_assignments.data(), h_assignments.size(), stream);
  raft::copy(d_n_per_route.data(), h_n_per_route.data(), h_n_per_route.size(), stream);

  thrust::fill(sol_handle->get_thrust_policy(),
               route_node_map.route_id_per_node.begin(),
               route_node_map.route_id_per_node.end(),
               -1);
  thrust::fill(sol_handle->get_thrust_policy(),
               route_node_map.intra_route_idx_per_node.begin(),
               route_node_map.intra_route_idx_per_node.end(),
               -1);

  due_time_greedy_init_kernel<i_t, f_t, REQUEST><<<n_routes, 1, 0, stream>>>(
    view(), problem_ptr->view(), d_assignments.data(), d_n_per_route.data(), max_per_route);

  stream.synchronize();

  // Mark solution as found.
  const i_t one = 1;
  d_sol_found.set_value_async(one, stream);
  stream.synchronize();
}

template class solution_t<int, float, request_t::PDP>;
template class solution_t<int, float, request_t::VRP>;

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
