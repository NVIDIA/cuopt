/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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
#include "found_solution.cuh"

namespace cuopt {
namespace routing {
namespace detail {

template <request_t REQUEST>
DI auto get_request_locations(found_sol_t selected_candidate)
{
  if constexpr (REQUEST == request_t::PDP) {
    return request_id_t<REQUEST>(selected_candidate.pickup_location,
                                 selected_candidate.delivery_location);
  } else {
    return request_id_t<REQUEST>(selected_candidate.pickup_location);
  }
}

// Insert both pickup & delivery in given route
// Update forward / backward data
// Route size is also update
// Single threaded for now
template <typename i_t, typename f_t, request_t REQUEST>
DI void execute_insert(typename solution_t<i_t, f_t, REQUEST>::view_t& view,
                       typename route_t<i_t, f_t, REQUEST>::view_t& route_to_modify,
                       request_id_t<REQUEST> const& request_location,
                       const request_info_t<i_t, REQUEST>* request_id)
{
  const auto& dimensions_info = view.problem.dimensions_info;
  cuopt_assert(__popc(__activemask() == 1), "execute_insert should be called by a single thread");
  auto request_node = view.get_request(request_id);

  if constexpr (REQUEST == request_t::PDP) {
    cuopt_assert(request_location.pickup >= 0 && request_location.delivery >= 0,
                 "Pickup and delivery locations should be positive");
    cuopt_assert(request_location.pickup <= request_location.delivery,
                 "Pickup should be smaller than delivery");
  }

  route_to_modify.template insert_request<REQUEST>(
    request_location, request_node, view.route_node_map);

  route_t<i_t, f_t, REQUEST>::view_t::compute_forward(route_to_modify);
  route_t<i_t, f_t, REQUEST>::view_t::compute_backward(route_to_modify);
  route_to_modify.compute_cost();
  view.routes_to_copy[route_to_modify.get_id()] = 1;
}

// Function used for pickup/delivery insertion after no ejection
// Compact then pick one move randomly
template <int BLOCK_SIZE, typename i_t, typename f_t, request_t REQUEST>
__global__ void select_feasible_insert(typename solution_t<i_t, f_t, REQUEST>::view_t view,
                                       uint64_t* __restrict__ feasible_candidates,
                                       found_sol_t* __restrict__ selected_candidate);

// Function used for pickup/delivery insertion after no ejection
// Execute move selected by select_feasible_insert
template <typename i_t, typename f_t, request_t REQUEST>
__global__ void execute_feasible_insert(typename solution_t<i_t, f_t, REQUEST>::view_t view,
                                        const request_info_t<i_t, REQUEST>* request_id,
                                        found_sol_t selected_candidate);

// Add ejected nodes to ejection pool
// Select temp route, execute the request insertion and write it back to route global memory
template <int BLOCK_SIZE, typename i_t, typename f_t, request_t REQUEST>
__global__ void select_tmp_and_execute_insert(
  typename solution_t<i_t, f_t, REQUEST>::view_t view,
  const request_info_t<i_t, REQUEST>* __restrict__ request_id,
  uint64_t* __restrict__ feasible_candidates,
  typename ejection_pool_t<request_info_t<i_t, REQUEST>>::view_t EP,
  i_t fragment_step,
  i_t fragment_size);
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
