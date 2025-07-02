/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "../solution/solution.cuh"

namespace cuopt {
namespace routing {
namespace detail {

// the reason we have this function is that intra_route_idx_per_node might change in parallel during
// a move so we manually find the location of the node as if it is invalid
template <typename i_t, typename f_t, request_t REQUEST>
DI i_t find_intra_idx(const typename route_t<i_t, f_t, REQUEST>::view_t& route, i_t node_id)
{
  i_t n_nodes = route.get_num_nodes();
  for (i_t i = 1; i < n_nodes; ++i) {
    if (node_id == route.requests().node_info[i].node() &&
        route.requests().node_info[i].is_service_node()) {
      return i;
    }
  }
  cuopt_assert(false, "Failed to find the node in this route");
  return -1;
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
DI void compute_temp_route(typename route_t<i_t, f_t, REQUEST>::view_t& temp_route,
                           const typename route_t<i_t, f_t, REQUEST>::view_t& route,
                           const i_t n_nodes_route,
                           typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                           request_id_t<REQUEST> const& request_id,
                           i_t intra_idx = -1)
{
  if (threadIdx.x == 0) {
    temp_route.set_num_nodes(route.get_num_nodes() - request_info_t<i_t, REQUEST>::size());
  }
  if (intra_idx == -1) { intra_idx = find_intra_idx<i_t, f_t, REQUEST>(route, request_id.id()); }
  __syncthreads();
  // copies the fixed route data (i.e tw values, demand, node_id etc.)
  temp_route.template copy_route_data_after_ejection<REQUEST>(route, intra_idx, true);
  __syncthreads();
  temp_route.copy_backward_data(route, intra_idx + 1, n_nodes_route + 1, intra_idx);
  __syncthreads();
  if (threadIdx.x == 0) {
    auto prev_node = temp_route.get_node(intra_idx - 1);
    auto next_node = temp_route.get_node(intra_idx);
    prev_node.calculate_forward_all(next_node, temp_route.vehicle_info());
    temp_route.set_node(intra_idx, next_node);
    next_node.calculate_backward_all(prev_node, temp_route.vehicle_info());
    temp_route.set_node(intra_idx - 1, prev_node);
    // forward pass after node to the end
    route_t<i_t, f_t, REQUEST>::view_t::compute_forward(temp_route, intra_idx);
    // backward pass until the beginning
    if (intra_idx > 1) {
      route_t<i_t, f_t, REQUEST>::view_t::compute_backward_in_between(temp_route, 0, intra_idx - 1);
    }
    temp_route.compute_cost();
  }
  __syncthreads();

  cuopt_assert(
    !temp_route.dimensions_info().has_dimension(dim_t::TIME) ||
      abs(temp_route.template get_dim<dim_t::TIME>().excess_forward[temp_route.get_num_nodes()] -
            temp_route.template get_dim<dim_t::TIME>().excess_backward[0] <
          0.01),
    "Excess issue");
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
DI void compute_temp_route(typename route_t<i_t, f_t, REQUEST>::view_t& temp_route,
                           const typename route_t<i_t, f_t, REQUEST>::view_t& route,
                           const i_t n_nodes_route,
                           typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                           request_id_t<REQUEST> const& request_id,
                           i_t intra_idx = -1)
{
  i_t pickup_id   = request_id.pickup;
  i_t delivery_id = request_id.delivery;
  i_t delivery_intra_idx;
  if (threadIdx.x == 0) {
    temp_route.set_num_nodes(route.get_num_nodes() - request_info_t<i_t, REQUEST>::size());
  }
  if (intra_idx == -1) {
    intra_idx          = find_intra_idx<i_t, f_t, REQUEST>(route, pickup_id);
    delivery_intra_idx = find_intra_idx<i_t, f_t, REQUEST>(route, delivery_id);
  } else {
    delivery_intra_idx = solution.route_node_map.get_intra_route_idx(request_id.delivery);
  }
  // copies the fixed route data (i.e tw values, demand, node_id etc.)
  __syncthreads();
  temp_route.template copy_route_data_after_ejection<REQUEST>(
    route, intra_idx, delivery_intra_idx, true);
  __syncthreads();
  if (threadIdx.x == 0) {
    auto prev_node = temp_route.get_node(intra_idx - 1);
    auto next_node = temp_route.get_node(intra_idx);
    prev_node.calculate_forward_all(next_node, temp_route.vehicle_info());
    temp_route.set_node(intra_idx, next_node);
    // if pickup and delivery are not together
    if (intra_idx + 1 != delivery_intra_idx) {
      // forward pass from the insertion node till the delivery node
      route_t<i_t, f_t, REQUEST>::view_t::compute_forward_in_between(
        temp_route,
        intra_idx,
        delivery_intra_idx - 2 /* -1 because we already removed the pickup node*/);
      prev_node = temp_route.get_node(delivery_intra_idx - 2);
      next_node = temp_route.get_node(delivery_intra_idx - 1);
      prev_node.calculate_forward_all(next_node, temp_route.vehicle_info());
      temp_route.set_node(delivery_intra_idx - 1, next_node);
    }
    // forward pass after the delivery node to the end
    route_t<i_t, f_t, REQUEST>::view_t::compute_forward_in_between(
      temp_route, delivery_intra_idx - 1, temp_route.get_num_nodes());
  }
  __syncthreads();
  temp_route.copy_backward_data(
    route, delivery_intra_idx + 1, n_nodes_route + 1, delivery_intra_idx - 1);
  __syncthreads();

  if (threadIdx.x == 0) {
    auto next_node = temp_route.get_node(delivery_intra_idx - 1);
    auto prev_node = temp_route.get_node(delivery_intra_idx - 2);
    next_node.calculate_backward_all(prev_node, temp_route.vehicle_info());
    temp_route.set_node(delivery_intra_idx - 2, prev_node);
    // backward pass until the beginning, unless delivery was at the beginning
    if (delivery_intra_idx > 2) {
      route_t<i_t, f_t, REQUEST>::view_t::compute_backward_in_between(
        temp_route, 0, delivery_intra_idx - 2);
    }
    temp_route.compute_cost();
  }
  __syncthreads();
  cuopt_assert(
    !temp_route.dimensions_info().has_dimension(dim_t::TIME) ||
      abs(temp_route.template get_dim<dim_t::TIME>().excess_forward[temp_route.get_num_nodes()] -
            temp_route.template get_dim<dim_t::TIME>().excess_backward[0] <
          0.01),
    "Excess issue");
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
