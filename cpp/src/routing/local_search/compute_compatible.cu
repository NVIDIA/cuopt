/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/cuda_helpers.cuh>
#include "compute_compatible.cuh"
#include "local_search.cuh"

#include <thrust/reduce.h>

#include <raft/util/cudart_utils.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
DI bool check_route_possible_for_given_vehicle(
  typename problem_t<i_t, f_t>::view_t problem,
  i_t vehicle_id,
  i_t i1,
  i_t i2,
  const typename solution_t<i_t, f_t, REQUEST>::view_t& sol,
  bool is_problem_run)
{
  auto start_depot_info  = problem.get_start_depot_node_info(vehicle_id);
  auto return_depot_info = problem.get_return_depot_node_info(vehicle_id);
  auto const& order_info = problem.order_info;

  // FIXME: Check with respect to vehicle zero for now. THis is wrong, we need to check for every
  // vehicle
  auto vehicle_info = problem.fleet_info.get_vehicle_info(vehicle_id);
  // Type doesnt' matter
  auto i1_info = NodeInfo<i_t>(i1, order_info.get_order_location(i1), node_type_t::PICKUP);
  auto i2_info = NodeInfo<i_t>(i2, order_info.get_order_location(i2), node_type_t::PICKUP);

  auto first_node =
    create_depot_node<i_t, f_t, REQUEST>(problem, start_depot_info, start_depot_info, vehicle_id);
  auto second_node = create_node<i_t, f_t, REQUEST>(problem, i1_info, i1_info);
  if (is_problem_run) {
    first_node.time_dim.calculate_forward(
      second_node.time_dim, get_transit_time(start_depot_info, i1_info, vehicle_info, true));
  } else {
    auto [route_id, intra_idx] = sol.route_node_map.get_route_id_and_intra_idx(i1);
    // intra_idx -1 comes from prize collection
    if (intra_idx == -1) {
      first_node.time_dim.calculate_forward(
        second_node.time_dim, get_transit_time(start_depot_info, i1_info, vehicle_info, true));
    } else {
      second_node = sol.routes[route_id].get_node(intra_idx);
    }
  }

  first_node = create_node<i_t, f_t, REQUEST>(problem, i2_info, i2_info);

  second_node.time_dim.calculate_forward(first_node.time_dim,
                                         get_transit_time(i1_info, i2_info, vehicle_info, true));

  second_node =
    create_depot_node<i_t, f_t, REQUEST>(problem, return_depot_info, return_depot_info, vehicle_id);
  first_node.time_dim.calculate_forward(
    second_node.time_dim, get_transit_time(i2_info, return_depot_info, vehicle_info, true));

  return second_node.time_dim.forward_feasible(VehicleInfo<f_t>());
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
DI bool check_route_possible_for_given_vehicle(
  typename problem_t<i_t, f_t>::view_t problem,
  i_t vehicle_id,
  i_t i1,
  i_t i2,
  i_t i3,
  i_t i4,
  const typename solution_t<i_t, f_t, REQUEST>::view_t& sol,
  bool is_problem_run)
{
  auto start_depot_info  = problem.get_start_depot_node_info(vehicle_id);
  auto return_depot_info = problem.get_return_depot_node_info(vehicle_id);

  auto const& order_info = problem.order_info;
  auto vehicle_info      = problem.fleet_info.get_vehicle_info(vehicle_id);
  i_t i1_brother = 0, i2_brother = 0, i3_brother = 0, i4_brother = 0;
  // Type doesnt' matter
  auto i1_info = NodeInfo<i_t>(i1, order_info.get_order_location(i1), node_type_t::PICKUP);
  auto i2_info = NodeInfo<i_t>(i2, order_info.get_order_location(i2), node_type_t::PICKUP);
  auto i3_info = NodeInfo<i_t>(i3, order_info.get_order_location(i3), node_type_t::PICKUP);
  auto i4_info = NodeInfo<i_t>(i4, order_info.get_order_location(i4), node_type_t::PICKUP);

  if constexpr (REQUEST == request_t::PDP) {
    i1_brother = problem.order_info.pair_indices[i1];
    i2_brother = problem.order_info.pair_indices[i2];
    i3_brother = problem.order_info.pair_indices[i3];
    i4_brother = problem.order_info.pair_indices[i4];
  }
  auto i1_brother_info =
    NodeInfo<i_t>(i1_brother, order_info.get_order_location(i1_brother), node_type_t::PICKUP);
  auto i2_brother_info =
    NodeInfo<i_t>(i2_brother, order_info.get_order_location(i2_brother), node_type_t::PICKUP);
  auto i3_brother_info =
    NodeInfo<i_t>(i3_brother, order_info.get_order_location(i3_brother), node_type_t::PICKUP);
  auto i4_brother_info =
    NodeInfo<i_t>(i4_brother, order_info.get_order_location(i4_brother), node_type_t::PICKUP);

  auto first_node =
    create_depot_node<i_t, f_t, REQUEST>(problem, start_depot_info, return_depot_info, vehicle_id);
  auto second_node = create_node<i_t, f_t, REQUEST>(problem, i1_info, i1_brother_info);
  if (is_problem_run) {
    first_node.time_dim.calculate_forward(
      second_node.time_dim, get_transit_time(start_depot_info, i1_info, vehicle_info, true));
  } else {
    auto [route_id, intra_idx] = sol.route_node_map.get_route_id_and_intra_idx(i1);
    // intra_idx -1 comes from prize collection
    if (intra_idx == -1) {
      first_node.time_dim.calculate_forward(
        second_node.time_dim, get_transit_time(start_depot_info, i1_info, vehicle_info, true));
    } else {
      second_node = sol.routes[route_id].get_node(intra_idx);
    }
  }

  first_node = create_node<i_t, f_t, REQUEST>(problem, i2_info, i2_brother_info);
  second_node.time_dim.calculate_forward(first_node.time_dim,
                                         get_transit_time(i1_info, i2_info, vehicle_info, true));

  second_node = create_node<i_t, f_t, REQUEST>(problem, i3_info, i3_brother_info);
  first_node.time_dim.calculate_forward(second_node.time_dim,
                                        get_transit_time(i2_info, i3_info, vehicle_info, true));

  first_node = create_node<i_t, f_t, REQUEST>(problem, i4_info, i4_brother_info);
  second_node.time_dim.calculate_forward(first_node.time_dim,
                                         get_transit_time(i3_info, i4_info, vehicle_info, true));

  second_node =
    create_depot_node<i_t, f_t, REQUEST>(problem, return_depot_info, start_depot_info, vehicle_id);
  first_node.time_dim.calculate_forward(
    second_node.time_dim, get_transit_time(i4_info, return_depot_info, vehicle_info, true));

  return second_node.time_dim.forward_feasible(VehicleInfo<f_t>());
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
DI bool check_route_possible(typename problem_t<i_t, f_t>::view_t problem,
                             i_t i1,
                             i_t i2,
                             const typename solution_t<i_t, f_t, REQUEST>::view_t& sol,
                             bool is_problem_run = true)
{
  // If any of the vehicle serve this request, return true
  i_t n_vehicles = problem.fleet_info.get_num_vehicles();
  for (i_t vid = 0; vid < n_vehicles; ++vid) {
    if (check_route_possible_for_given_vehicle<i_t, f_t, REQUEST>(
          problem, vid, i1, i2, sol, is_problem_run)) {
      return true;
    }
  }
  return false;
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
DI bool check_route_possible(typename problem_t<i_t, f_t>::view_t problem,
                             i_t i1,
                             i_t i2,
                             i_t i3,
                             i_t i4,
                             const typename solution_t<i_t, f_t, REQUEST>::view_t& sol,
                             bool is_problem_run = true)
{
  // If any of the vehicle serve this request, return true
  i_t n_vehicles = problem.fleet_info.get_num_vehicles();
  for (i_t vid = 0; vid < n_vehicles; ++vid) {
    if (check_route_possible_for_given_vehicle<i_t, f_t, REQUEST>(
          problem, vid, i1, i2, i3, i4, sol, is_problem_run)) {
      return true;
    }
  }
  return false;
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
DI bool check_compatible(typename problem_t<i_t, f_t>::view_t problem,
                         i_t p1,
                         i_t p2,
                         const typename solution_t<i_t, f_t, REQUEST>::view_t& sol)
{
  return check_route_possible<i_t, f_t, REQUEST>(problem, p1, p2, sol) ||
         check_route_possible<i_t, f_t, REQUEST>(problem, p2, p1, sol);
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
DI bool check_compatible(typename problem_t<i_t, f_t>::view_t problem,
                         i_t p1,
                         i_t p2,
                         const typename solution_t<i_t, f_t, REQUEST>::view_t& sol)
{
  i_t d1 = problem.order_info.pair_indices[p1];
  i_t d2 = problem.order_info.pair_indices[p2];

  return check_route_possible<i_t, f_t, REQUEST>(problem, p1, p2, d1, d2, sol) ||
         check_route_possible<i_t, f_t, REQUEST>(problem, p1, p2, d2, d1, sol) ||
         check_route_possible<i_t, f_t, REQUEST>(problem, p1, d1, p2, d2, sol) ||

         check_route_possible<i_t, f_t, REQUEST>(problem, p2, p1, d1, d2, sol) ||
         check_route_possible<i_t, f_t, REQUEST>(problem, p2, p1, d2, d1, sol) ||
         check_route_possible<i_t, f_t, REQUEST>(problem, p2, d2, p1, d1, sol);
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void initialize_incompatible_kernel(
  typename problem_t<i_t, f_t>::view_t problem,
  uint8_t* compatibility_matrix,
  const typename solution_t<i_t, f_t, REQUEST>::view_t sol)
{
  i_t t_id       = threadIdx.x + blockDim.x * blockIdx.x;
  i_t n_requests = problem.order_info.get_num_requests();

  if (t_id >= n_requests * n_requests) return;

  i_t i = t_id % n_requests;
  i_t j = t_id / n_requests;

  i_t p_i = REQUEST == request_t::PDP ? problem.pickup_indices[i]
                                      : i + (int)problem.order_info.depot_included;
  i_t p_j = REQUEST == request_t::PDP ? problem.pickup_indices[j]
                                      : j + (int)problem.order_info.depot_included;

  uint8_t b = check_compatible<i_t, f_t, REQUEST>(problem, p_i, p_j, sol);
  compatibility_matrix[p_i * problem.order_info.get_num_orders() + p_j] = b;
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
__global__ void initialize_viable_kernel(typename problem_t<i_t, f_t>::view_t problem,
                                         i_t* viable_to_pickups,
                                         i_t* viable_from_pickups,
                                         i_t* n_viable_to_pickups,
                                         i_t* n_viable_from_pickups,
                                         i_t* viable_to_deliveries,
                                         i_t* viable_from_deliveries,
                                         i_t* n_viable_to_deliveries,
                                         i_t* n_viable_from_deliveries,
                                         const typename solution_t<i_t, f_t, REQUEST>::view_t sol,
                                         bool is_problem_run)
{
  i_t t_id     = threadIdx.x + blockDim.x * blockIdx.x;
  i_t n_orders = problem.order_info.get_num_orders();
  if (t_id >= n_orders * n_orders) return;

  i_t i = t_id % n_orders;
  i_t j = t_id / n_orders;

  const bool depot_included = problem.order_info.depot_included;

  if (depot_included) {
    // we don't access any of them
    if (i == 0 || j == 0) { return; }
  }

  // check inserting j after i
  if (check_route_possible<i_t, f_t, REQUEST>(problem, i, j, sol, is_problem_run)) {
    i_t offset = atomicAdd(n_viable_from_pickups + i, 1);
    viable_from_pickups[i * problem.order_info.get_num_requests() + offset] = j;
  }

  // check inserting j before i
  if (check_route_possible<i_t, f_t, REQUEST>(problem, j, i, sol, is_problem_run)) {
    i_t offset = atomicAdd(n_viable_to_pickups + i, 1);
    viable_to_pickups[i * problem.order_info.get_num_requests() + offset] = j;
  }
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
__global__ void initialize_viable_kernel(typename problem_t<i_t, f_t>::view_t problem,
                                         i_t* viable_to_pickups,
                                         i_t* viable_from_pickups,
                                         i_t* n_viable_to_pickups,
                                         i_t* n_viable_from_pickups,
                                         i_t* viable_to_deliveries,
                                         i_t* viable_from_deliveries,
                                         i_t* n_viable_to_deliveries,
                                         i_t* n_viable_from_deliveries,
                                         const typename solution_t<i_t, f_t, REQUEST>::view_t sol,
                                         bool is_problem_run)
{
  i_t t_id     = threadIdx.x + blockDim.x * blockIdx.x;
  i_t n_orders = problem.order_info.get_num_orders();
  if (t_id >= n_orders * n_orders) return;

  const bool depot_included = problem.order_info.depot_included;
  i_t i                     = t_id % n_orders;
  i_t j                     = t_id / n_orders;

  if (depot_included) {
    // we don't access any of them
    if (i == 0 || j == 0) { return; }
  }

  i_t bi = problem.order_info.pair_indices[i];
  i_t bj = problem.order_info.pair_indices[j];

  // check insertion of pickups(i.e cases where j is a pickup)
  if (problem.order_info.is_pickup_index[j]) {
    // check inserting j after i
    if (problem.order_info.is_pickup_index[i]) {
      if (check_route_possible<i_t, f_t, REQUEST>(problem, i, j, bi, bj, sol, is_problem_run) ||
          check_route_possible<i_t, f_t, REQUEST>(problem, i, j, bj, bi, sol, is_problem_run)) {
        i_t offset = atomicAdd(n_viable_from_pickups + i, 1);
        viable_from_pickups[i * problem.order_info.get_num_requests() + offset] = j;
      }

    } else {
      if (check_route_possible<i_t, f_t, REQUEST>(problem, bi, i, j, bj, sol, is_problem_run)) {
        i_t offset = atomicAdd(n_viable_from_pickups + i, 1);
        viable_from_pickups[i * problem.order_info.get_num_requests() + offset] = j;
      }
    }

    // check inserting j before i
    if (problem.order_info.is_pickup_index[i]) {
      if (check_route_possible<i_t, f_t, REQUEST>(problem, j, i, bi, bj, sol, is_problem_run) ||
          check_route_possible<i_t, f_t, REQUEST>(problem, j, i, bj, bi, sol, is_problem_run) ||
          check_route_possible<i_t, f_t, REQUEST>(problem, j, bj, i, bi, sol, is_problem_run)) {
        i_t offset = atomicAdd(n_viable_to_pickups + i, 1);
        viable_to_pickups[i * problem.order_info.get_num_requests() + offset] = j;
      }

    } else {
      if (check_route_possible<i_t, f_t, REQUEST>(problem, bi, j, i, bj, sol, is_problem_run) ||
          check_route_possible<i_t, f_t, REQUEST>(problem, bi, j, bj, i, sol, is_problem_run)) {
        i_t offset = atomicAdd(n_viable_to_pickups + i, 1);
        viable_to_pickups[i * problem.order_info.get_num_requests() + offset] = j;
      }
    }
  }
  // check insertion of deliveries(i.e cases where j is a delivery)
  else {
    // check inserting j after i
    if (problem.order_info.is_pickup_index[i]) {
      if (check_route_possible<i_t, f_t, REQUEST>(problem, bj, i, j, bi, sol, is_problem_run) ||
          check_route_possible<i_t, f_t, REQUEST>(problem, i, bj, j, bi, sol, is_problem_run)) {
        i_t offset = atomicAdd(n_viable_from_deliveries + i, 1);
        viable_from_deliveries[i * problem.order_info.get_num_requests() + offset] = j;
      }

    } else {
      if (check_route_possible<i_t, f_t, REQUEST>(problem, bi, i, bj, j, sol, is_problem_run) ||
          check_route_possible<i_t, f_t, REQUEST>(problem, bi, bj, i, j, sol, is_problem_run) ||
          check_route_possible<i_t, f_t, REQUEST>(problem, bj, bi, i, j, sol, is_problem_run)) {
        i_t offset = atomicAdd(n_viable_from_deliveries + i, 1);
        viable_from_deliveries[i * problem.order_info.get_num_requests() + offset] = j;
      }
    }

    // check inserting j before i
    if (problem.order_info.is_pickup_index[i]) {
      if (check_route_possible<i_t, f_t, REQUEST>(problem, bj, j, i, bi, sol, is_problem_run)) {
        i_t offset = atomicAdd(n_viable_to_deliveries + i, 1);
        viable_to_deliveries[i * problem.order_info.get_num_requests() + offset] = j;
      }

    } else {
      if (check_route_possible<i_t, f_t, REQUEST>(problem, bi, bj, j, i, sol, is_problem_run) ||
          check_route_possible<i_t, f_t, REQUEST>(problem, bj, bi, j, i, sol, is_problem_run)) {
        i_t offset = atomicAdd(n_viable_to_deliveries + i, 1);
        viable_to_deliveries[i * problem.order_info.get_num_requests() + offset] = j;
      }
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void calculate_route_compatibility_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  uint8_t* route_compatibility,
  const uint8_t* compatibility_matrix)
{
  const i_t n_requests = solution.get_num_requests();
  const i_t n_orders   = solution.get_num_orders();
  i_t route_id         = blockIdx.x / n_requests;
  i_t request_id       = blockIdx.x % n_requests;

  i_t p_i = solution.get_request(request_id).id();

  if (!solution.route_node_map.is_node_served(p_i)) { return; }

  const auto route             = solution.routes[route_id];
  uint32_t thread_incompatible = 0;
  // loop over all the nodes and check compatibility matrix
  for (i_t z = threadIdx.x + 1; z < route.get_num_nodes(); z += blockDim.x) {
    if constexpr (REQUEST == request_t::PDP) {
      if (route.requests().node_info[z].is_pickup()) {
        i_t node_id = route.requests().node_info[z].node();
        if (!compatibility_matrix[node_id * n_orders + p_i]) ++thread_incompatible;
      }
    } else {
      if (route.requests().node_info[z].is_service_node()) {
        i_t node_id = route.requests().node_info[z].node();
        if (!compatibility_matrix[node_id * n_orders + p_i]) ++thread_incompatible;
      }
    }
  }
  __syncthreads();
  __shared__ uint32_t sh_buf[raft::WarpSize];
  uint32_t result = raft::blockReduce(thread_incompatible, (char*)sh_buf);
  __syncthreads();
  if (threadIdx.x == 0) { route_compatibility[route_id * n_orders + p_i] = min(result, 2); }
}

template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::calculate_route_compatibility(
  solution_t<i_t, f_t, REQUEST>& sol)
{
  raft::common::nvtx::range fun_scope("calculate_route_compatibility");
  // note that this was allocated for the max size
  // reset all even though we are not using the full array.
  thrust::fill(sol.sol_handle->get_thrust_policy(),
               move_candidates.route_compatibility.begin(),
               move_candidates.route_compatibility.end(),
               uint8_t(2));
  i_t TPB      = 128;
  i_t n_blocks = sol.n_routes * sol.get_num_requests();
  calculate_route_compatibility_kernel<i_t, f_t, REQUEST>
    <<<n_blocks, TPB, 0, sol.sol_handle->get_stream()>>>(
      sol.view(),
      move_candidates.route_compatibility.data(),
      move_candidates.viables.compatibility_matrix.data());
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
}

// sort the viable matrix according to the distance after the insertion
template <typename i_t, typename f_t>
void problem_t<i_t, f_t>::sort_viable_matrix(rmm::device_uvector<i_t>& viable_from_matrix,
                                             rmm::device_uvector<i_t>& viable_to_matrix)
{
  raft::common::nvtx::range fun_scope("sort_viable_matrix");
  rmm::device_uvector<i_t> segments(get_num_orders() * get_num_requests(),
                                    handle_ptr->get_stream());
  const i_t l_n_requests = get_num_requests();

  // FIXME: doing it only for first vehicle, this is wrong
  const auto l_vehicle_info = fleet_info.get_vehicle_info(0, handle_ptr->get_stream());
  auto order_info_view      = this->order_info.view();
  // create segments
  thrust::transform(handle_ptr->get_thrust_policy(),
                    thrust::counting_iterator<i_t>(0),
                    thrust::counting_iterator<i_t>(0) + get_num_orders() * get_num_requests(),
                    segments.begin(),
                    [l_n_requests] __device__(i_t idx) -> i_t {
                      i_t segment = idx / l_n_requests;
                      return segment;
                    });
  // sort according to distance
  thrust::stable_sort(
    handle_ptr->get_thrust_policy(),
    thrust::make_zip_iterator(viable_from_matrix.begin(), segments.begin()),
    thrust::make_zip_iterator(viable_from_matrix.end(), segments.end()),
    [l_vehicle_info, order_info_view] __device__(auto first, auto second) -> bool {
      i_t to_node_1   = thrust::get<0>(first);
      i_t to_node_2   = thrust::get<0>(second);
      i_t from_node_1 = thrust::get<1>(first);
      i_t from_node_2 = thrust::get<1>(second);
      if (to_node_1 == -1) return false;
      if (to_node_2 == -1) return true;
      auto dist_between_1 = get_distance(
        NodeInfo<i_t>(
          from_node_1, order_info_view.get_order_location(from_node_1), node_type_t::PICKUP),
        NodeInfo<i_t>(
          to_node_1, order_info_view.get_order_location(to_node_1), node_type_t::PICKUP),
        l_vehicle_info);
      auto dist_between_2 = get_distance(
        NodeInfo<i_t>(
          from_node_2, order_info_view.get_order_location(from_node_2), node_type_t::PICKUP),
        NodeInfo<i_t>(
          to_node_2, order_info_view.get_order_location(to_node_2), node_type_t::PICKUP),
        l_vehicle_info);
      return dist_between_1 < dist_between_2;
    });
  // sort the segments
  thrust::stable_sort(handle_ptr->get_thrust_policy(),
                      thrust::make_zip_iterator(viable_from_matrix.begin(), segments.begin()),
                      thrust::make_zip_iterator(viable_from_matrix.end(), segments.end()),
                      [] __device__(auto first, auto second) -> bool {
                        i_t from_node_1 = thrust::get<1>(first);
                        i_t from_node_2 = thrust::get<1>(second);
                        return from_node_1 < from_node_2;
                      });
  // create segments again
  thrust::transform(handle_ptr->get_thrust_policy(),
                    thrust::counting_iterator<i_t>(0),
                    thrust::counting_iterator<i_t>(0) + get_num_orders() * get_num_requests(),
                    segments.begin(),
                    [l_n_requests] __device__(i_t idx) -> i_t {
                      i_t segment = idx / l_n_requests;
                      return segment;
                    });
  // sort according to distance
  thrust::stable_sort(
    handle_ptr->get_thrust_policy(),
    thrust::make_zip_iterator(viable_to_matrix.begin(), segments.begin()),
    thrust::make_zip_iterator(viable_to_matrix.end(), segments.end()),
    [l_vehicle_info, order_info_view] __device__(auto first, auto second) -> bool {
      i_t from_node_1 = thrust::get<0>(first);
      i_t from_node_2 = thrust::get<0>(second);
      i_t to_node_1   = thrust::get<1>(first);
      i_t to_node_2   = thrust::get<1>(second);
      if (from_node_1 == -1) return false;
      if (from_node_2 == -1) return true;
      auto dist_between_1 = get_distance(
        NodeInfo<i_t>(
          from_node_1, order_info_view.get_order_location(from_node_1), node_type_t::PICKUP),
        NodeInfo<i_t>(
          to_node_1, order_info_view.get_order_location(to_node_1), node_type_t::PICKUP),
        l_vehicle_info);
      auto dist_between_2 = get_distance(
        NodeInfo<i_t>(
          from_node_2, order_info_view.get_order_location(from_node_2), node_type_t::PICKUP),
        NodeInfo<i_t>(
          to_node_2, order_info_view.get_order_location(to_node_2), node_type_t::PICKUP),
        l_vehicle_info);
      return dist_between_1 < dist_between_2;
    });
  // sort the segments again to get back the segmented sorted
  thrust::stable_sort(handle_ptr->get_thrust_policy(),
                      thrust::make_zip_iterator(viable_to_matrix.begin(), segments.begin()),
                      thrust::make_zip_iterator(viable_to_matrix.end(), segments.end()),
                      [] __device__(auto first, auto second) -> bool {
                        i_t to_node_1 = thrust::get<1>(first);
                        i_t to_node_2 = thrust::get<1>(second);
                        return to_node_1 < to_node_2;
                      });
  // raft::print_device_vector(
  //   "viable_to_matrix:", viable_to_matrix.data(), viable_to_matrix.size(),
  //   std::cout);
  // raft::print_device_vector(
  //   "viable_from_matrix:", viable_from_matrix.data(), viable_from_matrix.size(),
  //   std::cout);
  //   printf("\n");
  handle_ptr->sync_stream();
}

template <typename i_t, typename f_t, request_t REQUEST>
void initialize_incompatible(problem_t<i_t, f_t>& problem, solution_t<i_t, f_t, REQUEST>* sol_ptr)
{
  raft::common::nvtx::range fun_scope("initialize_incompatible");
  typename solution_t<i_t, f_t, REQUEST>::view_t sol_view;
  bool is_problem_run              = true;
  raft::handle_t const* handle_ptr = problem.handle_ptr;
  if (sol_ptr != nullptr) {
    sol_view       = sol_ptr->view();
    is_problem_run = false;
  }
  auto& viables = problem.viables;
  i_t n_items   = problem.get_num_orders() * problem.get_num_orders();
  viables.compatibility_matrix.resize(n_items, handle_ptr->get_stream());
  viables.viable_to_pickups.resize(problem.get_num_orders() * problem.get_num_requests(),
                                   handle_ptr->get_stream());
  viables.viable_from_pickups.resize(problem.get_num_orders() * problem.get_num_requests(),
                                     handle_ptr->get_stream());
  viables.n_viable_to_pickups.resize(problem.get_num_orders(), handle_ptr->get_stream());
  viables.n_viable_from_pickups.resize(problem.get_num_orders(), handle_ptr->get_stream());
  viables.viable_to_deliveries.resize(problem.get_num_orders() * problem.get_num_requests(),
                                      handle_ptr->get_stream());
  viables.viable_from_deliveries.resize(problem.get_num_orders() * problem.get_num_requests(),
                                        handle_ptr->get_stream());
  viables.n_viable_to_deliveries.resize(problem.get_num_orders(), handle_ptr->get_stream());
  viables.n_viable_from_deliveries.resize(problem.get_num_orders(), handle_ptr->get_stream());
  // initialize as always compatible
  thrust::fill(handle_ptr->get_thrust_policy(),
               viables.compatibility_matrix.begin(),
               viables.compatibility_matrix.end(),
               uint8_t(1));

  thrust::fill(handle_ptr->get_thrust_policy(),
               viables.viable_to_pickups.begin(),
               viables.viable_to_pickups.end(),
               -1);
  thrust::fill(handle_ptr->get_thrust_policy(),
               viables.n_viable_to_pickups.begin(),
               viables.n_viable_to_pickups.end(),
               0);
  thrust::fill(handle_ptr->get_thrust_policy(),
               viables.viable_from_pickups.begin(),
               viables.viable_from_pickups.end(),
               -1);
  thrust::fill(handle_ptr->get_thrust_policy(),
               viables.n_viable_from_pickups.begin(),
               viables.n_viable_from_pickups.end(),
               0);

  thrust::fill(handle_ptr->get_thrust_policy(),
               viables.viable_to_deliveries.begin(),
               viables.viable_to_deliveries.end(),
               -1);
  thrust::fill(handle_ptr->get_thrust_policy(),
               viables.n_viable_to_deliveries.begin(),
               viables.n_viable_to_deliveries.end(),
               0);
  thrust::fill(handle_ptr->get_thrust_policy(),
               viables.viable_from_deliveries.begin(),
               viables.viable_from_deliveries.end(),
               -1);
  thrust::fill(handle_ptr->get_thrust_policy(),
               viables.n_viable_from_deliveries.begin(),
               viables.n_viable_from_deliveries.end(),
               0);
  i_t TPB      = 256;
  i_t n_blocks = (problem.get_num_requests() * problem.get_num_requests() - 1 + TPB) / TPB;
  if constexpr (REQUEST == request_t::PDP) {
    initialize_incompatible_kernel<i_t, f_t, request_t::PDP>
      <<<n_blocks, TPB, 0, handle_ptr->get_stream()>>>(
        problem.view(), viables.compatibility_matrix.data(), sol_view);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_ptr->get_stream()));
    n_blocks = (problem.get_num_orders() * problem.get_num_orders() - 1 + TPB) / TPB;
    initialize_viable_kernel<i_t, f_t, request_t::PDP>
      <<<n_blocks, TPB, 0, handle_ptr->get_stream()>>>(problem.view(),
                                                       viables.viable_to_pickups.data(),
                                                       viables.viable_from_pickups.data(),
                                                       viables.n_viable_to_pickups.data(),
                                                       viables.n_viable_from_pickups.data(),
                                                       viables.viable_to_deliveries.data(),
                                                       viables.viable_from_deliveries.data(),
                                                       viables.n_viable_to_deliveries.data(),
                                                       viables.n_viable_from_deliveries.data(),
                                                       sol_view,
                                                       is_problem_run);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_ptr->get_stream()));
  } else {
    initialize_incompatible_kernel<i_t, f_t, request_t::VRP>
      <<<n_blocks, TPB, 0, handle_ptr->get_stream()>>>(
        problem.view(), viables.compatibility_matrix.data(), sol_view);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_ptr->get_stream()));
    n_blocks = (problem.get_num_orders() * problem.get_num_orders() - 1 + TPB) / TPB;
    initialize_viable_kernel<i_t, f_t, request_t::VRP>
      <<<n_blocks, TPB, 0, handle_ptr->get_stream()>>>(problem.view(),
                                                       viables.viable_to_pickups.data(),
                                                       viables.viable_from_pickups.data(),
                                                       viables.n_viable_to_pickups.data(),
                                                       viables.n_viable_from_pickups.data(),
                                                       viables.viable_to_deliveries.data(),
                                                       viables.viable_from_deliveries.data(),
                                                       viables.n_viable_to_deliveries.data(),
                                                       viables.n_viable_from_deliveries.data(),
                                                       sol_view,
                                                       is_problem_run);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_ptr->get_stream()));
  }
  problem.sort_viable_matrix(viables.viable_to_pickups, viables.viable_from_pickups);
  problem.sort_viable_matrix(viables.viable_to_deliveries, viables.viable_from_deliveries);
  i_t max_viable_to_pickup     = thrust::reduce(handle_ptr->get_thrust_policy(),
                                            viables.n_viable_to_pickups.begin(),
                                            viables.n_viable_to_pickups.end(),
                                            0,
                                            thrust::maximum<i_t>());
  i_t max_viable_from_pickup   = thrust::reduce(handle_ptr->get_thrust_policy(),
                                              viables.n_viable_from_pickups.begin(),
                                              viables.n_viable_from_pickups.end(),
                                              0,
                                              thrust::maximum<i_t>());
  i_t max_viable_to_delivery   = thrust::reduce(handle_ptr->get_thrust_policy(),
                                              viables.n_viable_to_deliveries.begin(),
                                              viables.n_viable_to_deliveries.end(),
                                              0,
                                              thrust::maximum<i_t>());
  i_t max_viable_from_delivery = thrust::reduce(handle_ptr->get_thrust_policy(),
                                                viables.n_viable_from_deliveries.begin(),
                                                viables.n_viable_from_deliveries.end(),
                                                0,
                                                thrust::maximum<i_t>());
  viables.max_viable_row_size  = std::max({max_viable_to_pickup,
                                           max_viable_from_pickup,
                                           max_viable_to_delivery,
                                           max_viable_from_delivery});
}
template void initialize_incompatible<int, float, request_t::VRP>(
  problem_t<int, float>& problem, solution_t<int, float, request_t::VRP>* sol_ptr);
template void initialize_incompatible<int, float, request_t::PDP>(
  problem_t<int, float>& problem, solution_t<int, float, request_t::PDP>* sol_ptr);
template void local_search_t<int, float, request_t::PDP>::calculate_route_compatibility(
  solution_t<int, float, request_t::PDP>& sol);
template void local_search_t<int, float, request_t::VRP>::calculate_route_compatibility(
  solution_t<int, float, request_t::VRP>& sol);
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
