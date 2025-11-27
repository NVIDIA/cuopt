/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <raft/core/device_span.hpp>
#include "routing/utilities/cuopt_utils.cuh"
#include "solution.cuh"

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
__device__ void set_deleted_route_data(
  typename solution_t<i_t, f_t, REQUEST>::view_t sol,
  raft::device_span<typename route_t<i_t, f_t, REQUEST>::view_t> routes,
  i_t route_id,
  i_t route_size,
  raft::device_span<request_info_t<i_t, REQUEST>> stack = {})
{
  routes[route_id].reset();
  i_t count = 0;

  for (i_t i = 1; i != route_size; ++i) {
    auto node_info = routes[route_id].requests().node_info[i];
    if (!node_info.is_break()) {
      request_id_t<REQUEST> request_id(node_info.node());
      if (count < stack.size()) {
        stack[count] = create_request<i_t, f_t, REQUEST>(sol.problem, request_id);
      }

      sol.route_node_map.reset_node(node_info);
      ++count;
    }
  }
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
__device__ void set_deleted_route_data(
  typename solution_t<i_t, f_t, REQUEST>::view_t sol,
  raft::device_span<typename route_t<i_t, f_t, REQUEST>::view_t> routes,
  i_t route_id,
  i_t route_size,
  raft::device_span<request_info_t<i_t, REQUEST>> stack = {})
{
  routes[route_id].reset();
  i_t count = 0;

  auto& route_node_map = sol.route_node_map;
  for (i_t i = 1; i != route_size; ++i) {
    auto node_info = routes[route_id].requests().node_info[i];
    if (node_info.is_pickup()) {
      const auto pickup_node_info   = node_info;
      const auto delivery_node_info = routes[route_id].requests().brother_info[i];
      request_id_t<REQUEST> request_id(pickup_node_info.node(), delivery_node_info.node());
      if (count < stack.size()) {
        stack[count] = create_request<i_t, f_t, REQUEST>(sol.problem, request_id);
      }
      // Mark those nodes as deactivated
      route_node_map.reset_node(pickup_node_info);
      route_node_map.reset_node(delivery_node_info);
      ++count;
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void set_deleted_routes_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t sol,
  raft::device_span<typename route_t<i_t, f_t, REQUEST>::view_t> routes,
  raft::device_span<i_t> route_ids,
  raft::device_span<request_info_t<i_t, REQUEST>> stack,
  i_t* thread_block_output_offset)
{
  cuopt_assert(route_ids.size() == gridDim.x, "route_ids size mismatch!");
  i_t route_id = route_ids[blockIdx.x];

  cuopt_assert(routes.size() != 0, "routes length mismatch!");
  cuopt_assert(routes[route_id].get_num_nodes_ptr() != nullptr, "route n_nodes is nullptr!");
  const i_t route_size        = routes[route_id].get_num_nodes();
  const i_t num_service_nodes = routes[route_id].get_num_service_nodes();
  cuopt_assert(route_size > 0, "Route size should be positive");

  auto write_stack = stack.subspan(atomicAdd(
    thread_block_output_offset, num_service_nodes / request_info_t<i_t, REQUEST>::size()));
  set_deleted_route_data<i_t, f_t, REQUEST>(sol, routes, route_id, route_size, write_stack);
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void set_deleted_routes_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t sol,
  raft::device_span<typename route_t<i_t, f_t, REQUEST>::view_t> routes,
  raft::device_span<i_t> route_ids)
{
  cuopt_assert(route_ids.size() == gridDim.x, "route_ids size mismatch!");
  i_t route_id = route_ids[blockIdx.x];

  cuopt_assert(routes.size() != 0, "routes length mismatch!");
  cuopt_assert(routes[route_id].get_num_nodes_ptr() != nullptr, "route n_nodes is nullptr!");
  const i_t route_size = routes[route_id].get_num_nodes();
  cuopt_assert(route_size > 0, "Route size should be positive");

  set_deleted_route_data<i_t, f_t, REQUEST>(sol, routes, route_id, route_size);
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void compute_route_id_kernel(typename route_t<i_t, f_t, REQUEST>::view_t* routes,
                                        typename route_node_map_t<i_t>::view_t route_node_map)
{
  auto& route = routes[blockIdx.x];
  for (i_t id = threadIdx.x + 1; id < route.get_num_nodes(); id += blockDim.x) {
    route_node_map.set_route_id_and_intra_idx(route.requests().node_info[id], blockIdx.x, id);
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void insert_nodes_to_route_kernel(typename solution_t<i_t, f_t, REQUEST>::view_t sol,
                                             i_t route_id,
                                             i_t intra_idx,
                                             i_t n_nodes_to_insert,
                                             NodeInfo<>* d_nodes_to_insert)
{
  extern __shared__ i_t shmem[];
  cuopt_assert(gridDim.x == 1, "Kernel should only have one block");
  const auto& order_info = sol.problem.order_info;
  auto& global_route     = sol.routes[route_id];
  auto route             = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    shmem, global_route, global_route.get_num_nodes() + n_nodes_to_insert);
  __syncthreads();
  route.copy_from(global_route);
  __syncthreads();
  if (threadIdx.x == 0) {
    for (i_t i = 0; i < n_nodes_to_insert; ++i) {
      auto node_id      = d_nodes_to_insert[i].node();
      auto brother_id   = node_id;
      auto node_type    = node_type_t::DELIVERY;
      auto brother_type = node_type_t::DELIVERY;
      if constexpr (REQUEST == request_t::PDP) {
        brother_id   = order_info.pair_indices[node_id];
        node_type    = sol.problem.order_info.is_pickup_index[node_id]
                         ? request_info_t<i_t, REQUEST>::primary_node_type()
                         : node_type_t::DELIVERY;
        brother_type = sol.problem.order_info.is_pickup_index[brother_id]
                         ? request_info_t<i_t, REQUEST>::primary_node_type()
                         : node_type_t::DELIVERY;

        cuopt_assert(node_type == d_nodes_to_insert[i].node_type(), "Node type mismatch!");
      }

      auto node_info = NodeInfo(node_id, order_info.get_order_location(node_id), node_type);
      auto brother_info =
        NodeInfo(brother_id, order_info.get_order_location(brother_id), brother_type);
      auto node = create_node<i_t, f_t, REQUEST>(sol.problem, node_info, brother_info);
      route.insert_node(intra_idx + i, node, sol.route_node_map);
    }
    route_t<i_t, f_t, REQUEST>::view_t::compute_forward(route);
    route_t<i_t, f_t, REQUEST>::view_t::compute_backward(route);
    route.compute_cost();
    sol.routes_to_copy[route_id] = 1;
  }
  __syncthreads();
  global_route.copy_from(route);
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void insert_node_to_best_kernel(typename solution_t<i_t, f_t, REQUEST>::view_t sol,
                                           NodeInfo<> node,
                                           const bool include_objective,
                                           const infeasible_cost_t weights)
{
  // This is only called for PDP use case to handle cluster infeasibility caused by
  // assymetric and symmetric EAX
  cuopt_assert(REQUEST == request_t::PDP,
               "insert_node_to_best_kernel should only be called for PDP use cases!");
  extern __shared__ i_t shmem[];
  __shared__ double reduction_buf[2 * raft::WarpSize];
  __shared__ i_t reduction_idx;
  const auto& order_info     = sol.problem.order_info;
  const auto& route_node_map = sol.route_node_map;

  i_t route_id   = -1;
  i_t node_id    = node.node();
  i_t brother_id = node_id;
  [[maybe_unused]] i_t brother_intra_idx;
  if constexpr (REQUEST == request_t::PDP) {
    brother_id = order_info.pair_indices[node_id];
    i_t brother_route_id;
    thrust::tie(brother_route_id, brother_intra_idx) =
      route_node_map.get_route_id_and_intra_idx(brother_id);
    cuopt_assert(brother_intra_idx != -1, "Brother should be routed!");
    cuopt_assert(brother_route_id != -1, "Brother should be routed!");
    cuopt_assert(route_id == -1, "Node should not have been routed!");

    // Inserting node to it's brother's route
    route_id = brother_route_id;
  }

  cuopt_assert(route_id >= 0 && route_id < sol.n_routes,
               "route to be inserted should be non-negative!");
  auto& global_route = sol.routes[route_id];
  auto route         = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    shmem, global_route, global_route.get_num_nodes() + 1);
  __syncthreads();
  route.copy_from(global_route);
  __syncthreads();
  double thread_best_cost = std::numeric_limits<double>::max();
  i_t thread_best_idx     = -1;
  bool is_brother_pickup =
    REQUEST == request_t::PDP ? order_info.is_pickup_index[brother_id] : true;
  auto node_type    = is_brother_pickup ? node_type_t::DELIVERY : node_type_t::PICKUP;
  auto brother_type = is_brother_pickup ? node_type_t::PICKUP : node_type_t::DELIVERY;

  cuopt_assert(node.node_type() == node_type, "Type mismatch!");
  auto node_info = NodeInfo(node_id, order_info.get_order_location(node_id), node_type);
  cuopt_assert(node_info == node, "Mismatch of node values!");
  auto brother_info = NodeInfo(brother_id, order_info.get_order_location(brother_id), brother_type);
  auto node_to_insert = create_node<i_t, f_t, REQUEST>(sol.problem, node_info, brother_info);
  i_t start, end;
  if constexpr (REQUEST == request_t::PDP) {
    start = is_brother_pickup ? brother_intra_idx : 0;
    end   = is_brother_pickup ? route.get_num_nodes() : brother_intra_idx;
  } else {
    start = 0;
    end   = route.get_num_nodes();
  }
  const auto route_objective_cost     = route.get_objective_cost();
  const auto route_infeasibility_cost = route.get_infeasibility_cost();
  cuopt_assert(start < end, "Start should be smaller than the end!");
  for (i_t i = start + threadIdx.x; i < end; i += blockDim.x) {
    auto next_node    = route.get_node(i + 1);
    double cost_delta = node_to_insert.calculate_forward_all_and_delta(next_node,
                                                                       route.vehicle_info(),
                                                                       include_objective,
                                                                       weights,
                                                                       route_objective_cost,
                                                                       route_infeasibility_cost);
    if (cost_delta < thread_best_cost) {
      thread_best_cost = cost_delta;
      thread_best_idx  = i;
    }
  }
  i_t t_id = threadIdx.x;
  block_reduce_ranked(thread_best_cost, t_id, reduction_buf, &reduction_idx);
  // there is a sync inside reduce
  if (threadIdx.x == reduction_idx) {
    cuopt_assert(thread_best_idx != -1, "");
    route.insert_node(thread_best_idx, node_to_insert, sol.route_node_map);
    route_t<i_t, f_t, REQUEST>::view_t::compute_forward(route);
    route_t<i_t, f_t, REQUEST>::view_t::compute_backward(route);
    route.compute_cost();
    sol.routes_to_copy[route_id]   = 1;
    sol.routes_to_search[route_id] = 1;
  }
  __syncthreads();
  global_route.copy_from(route);
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void remove_nodes_kernel(typename solution_t<i_t, f_t, REQUEST>::view_t sol,
                                    i_t n_nodes_to_eject,
                                    NodeInfo<>* d_nodes_to_eject,
                                    i_t* empty_route_produced)
{
  extern __shared__ i_t shmem[];

  auto& route_node_map = sol.route_node_map;
  cuopt_assert(gridDim.x == 1, "Kernel should only have one block");
  init_block_shmem(shmem, 0, sol.n_routes);
  __syncthreads();
  // first check if we generate empty routes if we eject
  for (i_t i = threadIdx.x; i < n_nodes_to_eject; i += blockDim.x) {
    i_t node_to_eject = d_nodes_to_eject[i].node();
    i_t route_id      = route_node_map.get_route_id(node_to_eject);
    cuopt_assert(route_id != -1, "Node should be part of a route!");
    atomicAdd(&shmem[route_id], 1);
  }
  __syncthreads();
  for (i_t i = 0; i < n_nodes_to_eject; ++i) {
    i_t node_to_eject = d_nodes_to_eject[i].node();
    i_t route_id      = route_node_map.get_route_id(node_to_eject);
    cuopt_assert(route_id != -1, "Node should be part of a route!");
    if ((sol.routes[route_id].get_num_nodes() - 1) <= shmem[route_id]) {
      if (threadIdx.x == 0) *empty_route_produced = 1;
      return;
    }
  }
  __syncthreads();

  for (i_t i = 0; i < n_nodes_to_eject; ++i) {
    i_t node_to_eject          = d_nodes_to_eject[i].node();
    auto [route_id, intra_idx] = route_node_map.get_route_id_and_intra_idx(d_nodes_to_eject[i]);
    cuopt_assert(route_id != -1, "Node should be part of a route!");
    cuopt_assert(intra_idx != -1, "Node should be part of a route!");
    auto& global_route = sol.routes[route_id];
    auto route         = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
      shmem, global_route, global_route.get_num_nodes());
    __syncthreads();
    route.copy_from(global_route);
    __syncthreads();
    if (threadIdx.x == 0) {
      route.eject_node(intra_idx, sol.route_node_map);
      route_t<i_t, f_t, REQUEST>::view_t::compute_forward(route);
      route_t<i_t, f_t, REQUEST>::view_t::compute_backward(route);
      route.compute_cost();
      sol.routes_to_copy[route_id]   = 1;
      sol.routes_to_search[route_id] = 1;
    }
    __syncthreads();
    global_route.copy_from(route);
    __syncthreads();
  }
  if (threadIdx.x == 0) { *empty_route_produced = 0; }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void remap_route_nodes(typename route_t<i_t, f_t, REQUEST>::view_t* routes,
                                  typename route_node_map_t<i_t>::view_t route_node_map,
                                  i_t* deleted_route_ids,
                                  i_t n_deleted)
{
  cuopt_assert(deleted_route_ids[0] > -1, "starting_route should be greater than -1!");
  __shared__ i_t n_deleted_before;
  i_t min_deleted_route_id = deleted_route_ids[0];
  // start from the min
  i_t block_route_id = min_deleted_route_id + blockIdx.x;
  if (threadIdx.x == 0) {
    n_deleted_before = 0;
    for (i_t i = 0; i < n_deleted; ++i) {
      if (deleted_route_ids[i] < block_route_id) {
        ++n_deleted_before;
      } else if (deleted_route_ids[i] == block_route_id) {
        n_deleted_before = -1;
        break;
      }
    }
  }
  __syncthreads();
  // this block handles an already deleted route
  if (n_deleted_before == -1) return;
  cuopt_assert(n_deleted_before > 0, "at least one route should be deleted before!");
  auto& route      = routes[block_route_id];
  i_t new_route_id = block_route_id - n_deleted_before;
  __syncthreads();
  // +1 to skip depot
  for (i_t id = threadIdx.x + 1; id < route.get_num_nodes(); id += blockDim.x) {
    // Do not touch break nodes
    if (!route.requests().node_info[id].is_break()) {
      cuopt_assert(route_node_map.get_route_id(route.requests().node_info[id]) > 0,
                   "route_id could not be 0");
      route_node_map.set_route_id(route.requests().node_info[id], new_route_id);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    route.set_id(new_route_id);
    cuopt_assert(new_route_id >= min_deleted_route_id,
                 "new_route_id should be greater than min_deleted_route_id!");
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void shift_routes_kernel(typename solution_t<i_t, f_t, REQUEST>::view_t sol,
                                    i_t* deleted_route_ids,
                                    i_t n_deleted)
{
  i_t route_shift_count = 0;
  i_t curr_route_idx    = deleted_route_ids[0];
  while (curr_route_idx < sol.n_routes) {
    if (route_shift_count < n_deleted && curr_route_idx == deleted_route_ids[route_shift_count]) {
      ++route_shift_count;
    } else {
      sol.routes[curr_route_idx - route_shift_count] = sol.routes[curr_route_idx];
    }
    ++curr_route_idx;
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void copy_routes(typename solution_t<i_t, f_t, REQUEST>::view_t dst_sol,
                            const typename solution_t<i_t, f_t, REQUEST>::view_t src_sol)
{
  const auto route_id = blockIdx.x;
  const auto n_nodes  = src_sol.routes[route_id].get_num_nodes();
  if (n_nodes > 0) { dst_sol.routes[route_id].copy_from(src_sol.routes[route_id]); }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void compute_cost_kernel(typename solution_t<i_t, f_t, REQUEST>::view_t sol)
{
  const auto route_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (route_id < sol.n_routes) {
    infeasible_cost_t route_inf_cost;
    objective_cost_t route_obj_cost;
    thrust::tie(route_obj_cost, route_inf_cost) = sol.routes[route_id].compute_cost(false);

    objective_cost_t* obj_cost  = sol.objective_cost;
    infeasible_cost_t* inf_cost = sol.infeasibility_cost;

    obj_cost->atomic_add(route_obj_cost);
    inf_cost->atomic_add(route_inf_cost);
    if (!sol.routes[route_id].is_feasible()) { atomicAdd(sol.n_infeasible_routes, 1); }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void compute_max_active_kernel(typename solution_t<i_t, f_t, REQUEST>::view_t sol)
{
  __shared__ i_t shmem[warp_size];
  i_t max_per_thread = 0;
  for (i_t i = threadIdx.x; i < sol.n_routes; i += blockDim.x) {
    // we calculate depot twice because this is used in alocations
    max_per_thread = max(sol.routes[i].get_num_nodes() + 1, max_per_thread);
  }
  // reduction is min, so convert it to max
  block_reduce(-max_per_thread, shmem);
  if (threadIdx.x == 0) { *sol.max_active_nodes_for_all_routes = -shmem[0]; }
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
