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

#include "compute_ejections.cuh"
#include "compute_insertions.cuh"
#include "delivery_insertion.cuh"

#include <utilities/seed_generator.cuh>
#include "routing/utilities/cuopt_utils.cuh"

#include "../routing_helpers.cuh"

#include <raft/util/cudart_utils.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <request_t REQUEST>
constexpr int get_n_viable()
{
  if constexpr (REQUEST == request_t::VRP) {
    return 256;
  } else {
    return 64;
  }
}

template <typename i_t, typename f_t, typename T>
__device__ bool print_filtered(i_t curr_node,
                               T* pickup_ids,
                               i_t n_viable,
                               VehicleInfo<f_t> vehicle_info)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("n_viable %d \n", n_viable);
    for (int i = 0; i < n_viable; ++i) {
      auto info_1     = NodeInfo<i_t>(curr_node, curr_node, node_type_t::PICKUP);
      auto info_2     = NodeInfo<i_t>((int)pickup_ids[i], (int)pickup_ids[i], node_type_t::PICKUP);
      double distance = get_transit_time(info_1, info_2, vehicle_info, true);
      printf("Distance from %d to %d is %f\n", curr_node, pickup_ids[i], distance);
    }
    printf("\n\n\n");
  }
  return true;
}

template <typename T>
__device__ bool print_array(T* array, int size)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("size of array %d \n", size);
    for (int i = 0; i < size; ++i) {
      printf("%d  ", array[i]);
    }
    printf("\n\n\n");
  }
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST, search_type_t search_type, bool is_pickup>
DI void time_filter_of_request(const typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                               const typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
                               const typename route_t<i_t, f_t, REQUEST>::view_t& route,
                               i_t request_idx,
                               i_t node_id_at_consideration,
                               i_t insertion_idx,
                               i_t route_id,
                               i_t route_id_of_inserted,
                               i_t intra_idx_of_inserted,
                               double excess_curr_route,
                               raft::device_span<uint16_t>& compacted_requests)
{
  if constexpr (REQUEST == request_t::PDP) {
    auto prev_node = route.get_node(insertion_idx);
    auto next_node = route.get_node(insertion_idx + 1);
    // if the performance is bad just take this relative index computation out in an array
    // const auto& route_of_inserted = solution.routes[route_id_of_inserted];

    // now check if the forward filtration works
    auto node_to_insert =
      route_id_of_inserted >= 0
        ? solution.routes[route_id_of_inserted].get_node(intra_idx_of_inserted)
        : create_node<i_t, f_t, REQUEST>(solution.problem, node_id_at_consideration);

    // compute excess limit of current route (whether same of different route)
    double excess_limit = get_excess_limit<i_t, f_t, REQUEST, search_type>(
      solution, move_candidates, route_id, excess_curr_route, route_id_of_inserted);

    if constexpr (is_pickup) {
      prev_node.calculate_forward_all(node_to_insert, route.vehicle_info());
      // checks whether there is an excess or combination fails
      bool pickup_filtration =
        node_to_insert.time_dim.forward_feasible(
          route.vehicle_info(), move_candidates.weights[dim_t::TIME], excess_limit) &&
        node_t<i_t, f_t, REQUEST>::time_combine(
          node_to_insert, next_node, route.vehicle_info(), move_candidates.weights, excess_limit);
      if (pickup_filtration) { compacted_requests[request_idx] = 1; }
    } else {
      next_node.calculate_backward_all(node_to_insert, route.vehicle_info());
      // checks whether there is an excess or combination fails
      bool delivery_filtration =
        node_to_insert.time_dim.backward_feasible(
          route.vehicle_info(), move_candidates.weights[dim_t::TIME], excess_limit) &&
        node_t<i_t, f_t, REQUEST>::time_combine(
          prev_node, node_to_insert, route.vehicle_info(), move_candidates.weights, excess_limit);
      if (delivery_filtration) { compacted_requests[request_idx] = 1; }
    }
  } else {
    compacted_requests[request_idx] = 1;
  }
}

// in the local search, each block handles a route and a pickup insertion position
// this kernel filters the viable pickup requests and their insertion positions in different
// routes we create a viable requests of 128 that can be inserted after each route(ejected and
// unejected)/pickup position the number of possible insertion positions in a loop will be limited
// to max 40 we will calculate the relative index of the inserted request in the new route:
// (ejected_pickup_index/ejected_route_size)*inserted_route_size we will try to insert the pickup
// request around this relative index +-20 positions (in total 40)
template <typename i_t,
          typename f_t,
          request_t REQUEST,
          search_type_t search_type,
          bool is_pickup,
          bool insert_unserviced>
__device__ i_t
filter_viable_requests(const typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                       const typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
                       const typename route_t<i_t, f_t, REQUEST>::view_t& route,
                       i_t insertion_idx,
                       i_t ejected_pickup_id,
                       raft::device_span<uint16_t>& compacted_requests,
                       raft::device_span<uint16_t>& sh_buf_temp)
{
  auto& route_node_map       = solution.route_node_map;
  const bool is_ejected      = ejected_pickup_id < solution.get_num_orders();
  const i_t n_nodes_of_route = route.get_num_nodes();
  const i_t route_id         = route.get_id();
  cuopt_assert(!is_ejected || route_id == route_node_map.get_route_id(ejected_pickup_id),
               "Wrong route id");
  double excess_curr_route = solution.routes[route_id].get_weighted_excess(move_candidates.weights);
  init_block_shmem(sh_buf_temp.data(), static_cast<uint16_t>(0), solution.get_num_requests());
  init_block_shmem(
    compacted_requests.data(), static_cast<uint16_t>(0), solution.get_num_requests());
  __syncthreads();
  auto prev_node_info = route.requests().node_info[insertion_idx];
  auto next_node_info = route.requests().node_info[insertion_idx + 1];

  // FIXME:: Treat break/special nodes as regular nodes
  if (n_nodes_of_route == 1 ||
      (!prev_node_info.is_service_node() && !next_node_info.is_service_node())) {
    // if pickup is unrouted, delivery is also assumed to be unrouted
    for (i_t j = threadIdx.x; j < solution.get_num_requests(); j += blockDim.x) {
      auto curr_req = solution.get_request(j);
      if constexpr (!is_pickup && REQUEST == request_t::PDP) {
        raft::swapVals(curr_req.pickup, curr_req.delivery);
      }
      auto pickup_id = curr_req.id();
      // Don't allow inserting of nodes from same route
      if ((insert_unserviced && !route_node_map.is_node_served(pickup_id)) ||
          (!insert_unserviced && route_node_map.is_node_served(pickup_id) &&
           route_id != route_node_map.get_route_id(pickup_id))) {
        compacted_requests[j] = 1;
      }
    }
    __syncthreads();
    // now we have the compatible requests marked, compact them and get the size
    block_inclusive_scan(
      sh_buf_temp.data(), compacted_requests.data(), solution.get_num_requests());
    __syncthreads();
    const uint16_t size_of_compacted_requests = sh_buf_temp[solution.get_num_requests() - 1];
    init_block_shmem(
      compacted_requests.data(), static_cast<uint16_t>(0), solution.get_num_requests());
    // there is not a single compatible request
    __syncthreads();
    // compact request indices
    for (i_t i = threadIdx.x + 1; i < solution.get_num_requests(); i += blockDim.x) {
      // only record the indices where the scan is changed
      if (sh_buf_temp[i] > sh_buf_temp[i - 1]) {
        auto curr_req = solution.get_request(i);
        if constexpr (!is_pickup && REQUEST == request_t::PDP) {
          raft::swapVals(curr_req.pickup, curr_req.delivery);
        }
        compacted_requests[sh_buf_temp[i] - 1] = curr_req.id();
      }
    }
    if (sh_buf_temp[0] == 1) {
      auto curr_req = solution.get_request(0);
      if constexpr (!is_pickup && REQUEST == request_t::PDP) {
        raft::swapVals(curr_req.pickup, curr_req.delivery);
      }
      compacted_requests[0] = curr_req.id();
    }
    __syncthreads();
    // store first 128 or less, intra was checked here too
    return size_of_compacted_requests;
  } else {
    bool is_prev_not_service = !prev_node_info.is_service_node();
    i_t node_id_at_insertion = is_prev_not_service
                                 ? route.requests().node_info[insertion_idx + 1].node()
                                 : route.requests().node_info[insertion_idx].node();

    // this is a sorted viable list
    i_t n_viable_requests     = 0;
    const i_t* viable_row_ptr = nullptr;
    if constexpr (is_pickup) {
      n_viable_requests = is_prev_not_service
                            ? move_candidates.viables.n_viable_to_pickups[node_id_at_insertion]
                            : move_candidates.viables.n_viable_from_pickups[node_id_at_insertion];
      viable_row_ptr =
        is_prev_not_service
          ? &move_candidates.viables
               .viable_to_pickups[node_id_at_insertion * solution.get_num_requests()]
          : &move_candidates.viables
               .viable_from_pickups[node_id_at_insertion * solution.get_num_requests()];
    } else {
      n_viable_requests =
        is_prev_not_service
          ? move_candidates.viables.n_viable_to_deliveries[node_id_at_insertion]
          : move_candidates.viables.n_viable_from_deliveries[node_id_at_insertion];

      viable_row_ptr =
        is_prev_not_service
          ? &move_candidates.viables
               .viable_to_deliveries[node_id_at_insertion * solution.get_num_requests()]
          : &move_candidates.viables
               .viable_from_deliveries[node_id_at_insertion * solution.get_num_requests()];
    }

    for (i_t j = threadIdx.x; j < n_viable_requests; j += blockDim.x) {
      i_t node_id_at_consideration = viable_row_ptr[j];
      cuopt_assert(node_id_at_consideration != -1, "Node id considered cannot be -1");

      auto [route_id_of_inserted, intra_idx_of_inserted] =
        solution.route_node_map.get_route_id_and_intra_idx(node_id_at_consideration);
      i_t pickup_id_of_viable_node = node_id_at_consideration;
      if constexpr (!is_pickup) {
        pickup_id_of_viable_node =
          solution.problem.order_info.pair_indices[node_id_at_consideration];
      }

      if ((!insert_unserviced && route_id_of_inserted == -1) ||
          // for prize collection, don't consider already serviced nodes
          (insert_unserviced && route_id_of_inserted >= 0) ||
          (route_id_of_inserted == route_id && pickup_id_of_viable_node != ejected_pickup_id)) {
        continue;
      }

      uint8_t is_node_compatible_with_its_route =
        route_id_of_inserted >= 0
          ? move_candidates.route_compatibility[route_id_of_inserted * solution.get_num_orders() +
                                                pickup_id_of_viable_node]
          : true;
      // get the route compatibility value of this node_id
      uint8_t comp_val =
        move_candidates
          .route_compatibility[route_id * solution.get_num_orders() + pickup_id_of_viable_node];
      bool compatible = false;
      // the ejected request was incompatible
      if (is_ejected &&
          move_candidates
              .route_compatibility[route_id * solution.get_num_orders() + ejected_pickup_id] > 0) {
        compatible = true;
      }
      // fully compatible
      else if ((comp_val == 0 || is_node_compatible_with_its_route > 0)) {
        compatible = true;
      }
      // 1 request not compatible and we are checking whether that request is ejected
      else if ((is_ejected && (comp_val == 1) &&
                !move_candidates.viables
                   .compatibility_matrix[pickup_id_of_viable_node * solution.get_num_orders() +
                                         ejected_pickup_id])) {
        compatible = true;
      } else if (ejected_pickup_id == pickup_id_of_viable_node) {
        compatible = true;
      }

      if (!compatible) {
        continue;
      } else {
        time_filter_of_request<i_t, f_t, REQUEST, search_type, is_pickup>(solution,
                                                                          move_candidates,
                                                                          route,
                                                                          j,
                                                                          node_id_at_consideration,
                                                                          insertion_idx,
                                                                          route_id,
                                                                          route_id_of_inserted,
                                                                          intra_idx_of_inserted,
                                                                          excess_curr_route,
                                                                          compacted_requests);
      }
    }
    __syncthreads();
    // now we have the compatible requests marked, compact them and get the size
    block_inclusive_scan(
      sh_buf_temp.data(), compacted_requests.data(), solution.get_num_requests());
    __syncthreads();
    // print_array(sh_buf_temp.data(), solution.get_num_requests());
    const i_t size_of_compacted_requests = sh_buf_temp[solution.get_num_requests() - 1];
    init_block_shmem(
      compacted_requests.data(), static_cast<uint16_t>(0), solution.get_num_requests());
    // there is not a single compatible request
    if (size_of_compacted_requests == 0) { return 0; }
    __syncthreads();
    // compact request indices
    for (i_t i = threadIdx.x + 1; i < n_viable_requests; i += blockDim.x) {
      // only record the indices where the scan is changed
      if (sh_buf_temp[i] > sh_buf_temp[i - 1]) {
        compacted_requests[sh_buf_temp[i] - 1] = viable_row_ptr[i];
      }
    }
    if (threadIdx.x == 0 && sh_buf_temp[0] == 1) { compacted_requests[0] = viable_row_ptr[0]; }
    __syncthreads();
    return size_of_compacted_requests;
  }
}

DI void shuffle_compacted_requests(raft::device_span<uint16_t>& compacted_requests,
                                   raft::random::PCGenerator& rng,
                                   int size_of_compacted_requests)
{
  if (threadIdx.x == 0) {
    for (int i = 0; i < size_of_compacted_requests / 4; ++i) {
      int idx1 = rng.next_u32() % size_of_compacted_requests;
      int idx2 = rng.next_u32() % size_of_compacted_requests;
      if (idx1 != idx2) raft::swapVals(compacted_requests[idx1], compacted_requests[idx2]);
    }
  }
  __syncthreads();
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          search_type_t search_type,
          bool pickup_first,
          bool insert_unserviced>
__device__ i_t find_request_insertion(typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                                      typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
                                      i_t ejected_pickup_id,
                                      i_t node_insertion,
                                      const typename route_t<i_t, f_t, REQUEST>::view_t& route,
                                      cand_t* candidate,
                                      raft::device_span<uint16_t>& compacted_requests,
                                      raft::device_span<uint16_t>& sh_buf_temp,
                                      double excess_curr_route,
                                      [[maybe_unused]] raft::random::PCGenerator* rng = nullptr)
{
  i_t size_of_compacted_requests =
    filter_viable_requests<i_t, f_t, REQUEST, search_type, pickup_first, insert_unserviced>(
      solution,
      move_candidates,
      route,
      node_insertion,
      ejected_pickup_id,
      compacted_requests,
      sh_buf_temp);
  if constexpr (search_type == search_type_t::RANDOM) {
    // as an heuristic add some noise to compacted requests when it is random
    if (size_of_compacted_requests > get_n_viable<REQUEST>()) {
      shuffle_compacted_requests(compacted_requests, *rng, size_of_compacted_requests);
    }
  }

  size_of_compacted_requests = min(size_of_compacted_requests, get_n_viable<REQUEST>());
  cuopt_assert(size_of_compacted_requests <= blockDim.x,
               "size_of_compacted_requests should be smaller!");
  // only continue if the thread is working on a valid request
  if (threadIdx.x < size_of_compacted_requests) {
    i_t node_id = compacted_requests[threadIdx.x];
    request_id_t<REQUEST> request_id;
    if constexpr (REQUEST == request_t::PDP) {
      request_id =
        request_id_t<REQUEST>(node_id, solution.problem.order_info.pair_indices[node_id]);
    } else {
      request_id = request_id_t<REQUEST>(node_id);
    }
    auto other_route_id = solution.route_node_map.get_route_id(node_id);
    cuopt_assert(other_route_id >= 0 || insert_unserviced,
                 "Other route id cannot be -1, it must have been filtered!");
    const auto& dimensions_info = solution.problem.dimensions_info;

    auto request_node =
      other_route_id >= 0
        ? solution.routes[other_route_id].get_request_node(solution.route_node_map, request_id)
        : create_request_node<i_t, f_t, REQUEST>(solution.problem, request_id);
    if constexpr (pickup_first) {
      const auto prev_node = route.get_node(node_insertion);
      prev_node.calculate_forward_all(request_node.node(), route.vehicle_info());
    } else if constexpr (REQUEST == request_t::PDP) {
      raft::swapVals(request_node.pickup, request_node.delivery);
      const auto next_node = route.get_node(node_insertion + 1);
      next_node.calculate_backward_all(request_node.delivery, route.vehicle_info());
    }

    // compute excess limit of current route (whether same of different route)
    double excess_limit = get_excess_limit<i_t, f_t, REQUEST, search_type>(
      solution, move_candidates, route.get_id(), excess_curr_route, other_route_id);

    if constexpr (search_type != search_type_t::RANDOM) {
      find_brother_insertion<i_t, f_t, REQUEST, pick_mode_t::COST_DELTA, pickup_first>(
        solution,
        request_node,
        node_insertion,
        route,
        move_candidates.include_objective,
        move_candidates.weights,
        excess_limit,
        candidate);
    } else {
      find_brother_insertion<i_t, f_t, REQUEST, pick_mode_t::PROBABILITY, pickup_first>(
        solution,
        request_node,
        node_insertion,
        route,
        move_candidates.include_objective,
        move_candidates.weights,
        excess_limit,
        candidate,
        rng);
    }
  }
  return size_of_compacted_requests;
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          search_type_t search_type,
          bool insert_unserviced>
__device__ void record_found_request_insertion(
  typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
  typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
  i_t ejected_pickup_id,
  i_t node_insertion,
  i_t sink_node_id,
  const typename route_t<i_t, f_t, REQUEST>::view_t& route,
  raft::device_span<uint16_t>& compacted_requests,
  raft::device_span<uint16_t>& sh_buf_temp,
  double excess_curr_route,
  [[maybe_unused]] raft::random::PCGenerator* rng = nullptr)
{
  if constexpr (REQUEST == request_t::VRP) {
    cand_t candidate = cand_t::create<search_type>();
    auto size_of_compacted_requests =
      find_request_insertion<i_t, f_t, REQUEST, search_type, true, insert_unserviced>(
        solution,
        move_candidates,
        ejected_pickup_id,
        node_insertion,
        route,
        &candidate,
        compacted_requests,
        sh_buf_temp,
        excess_curr_route,
        rng);
    if (threadIdx.x < size_of_compacted_requests && candidate.is_valid(search_type)) {
      move_candidates.record_candidate(candidate, compacted_requests[threadIdx.x], sink_node_id);
    }
  } else {
    cand_t candidate = cand_t::create<search_type>();
    auto size_of_compacted_requests =
      find_request_insertion<i_t, f_t, REQUEST, search_type, true, insert_unserviced>(
        solution,
        move_candidates,
        ejected_pickup_id,
        node_insertion,
        route,
        &candidate,
        compacted_requests,
        sh_buf_temp,
        excess_curr_route,
        rng);
    // the compacted requests here and the next line are completely different here they are pickups,
    // the other deliveries
    if (threadIdx.x < size_of_compacted_requests && candidate.is_valid(search_type)) {
      move_candidates.record_candidate(candidate, compacted_requests[threadIdx.x], sink_node_id);
    }
    __syncthreads();
    candidate = cand_t::create<search_type>();
    size_of_compacted_requests =
      find_request_insertion<i_t, f_t, REQUEST, search_type, false, insert_unserviced>(
        solution,
        move_candidates,
        ejected_pickup_id,
        node_insertion,
        route,
        &candidate,
        compacted_requests,
        sh_buf_temp,
        excess_curr_route,
        rng);
    if (threadIdx.x < size_of_compacted_requests && candidate.is_valid(search_type)) {
      move_candidates.record_candidate(candidate, compacted_requests[threadIdx.x], sink_node_id);
    }
  }
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          search_type_t search_type,
          bool insert_unserviced>
__device__ void find_cross_relocate_insertion(
  typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
  typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
  const request_id_t<REQUEST>& ejected_id,
  const typename route_t<i_t, f_t, REQUEST>::view_t& route,
  raft::device_span<uint16_t>& compacted_requests,
  raft::device_span<uint16_t>& sh_buf_temp,
  double excess_curr_route,
  bool is_ejected_route,
  [[maybe_unused]] raft::random::PCGenerator* thread_rng = nullptr)
{
  if (is_ejected_route) {
    // we want to insert just after
    i_t ejected_pickup_idx   = solution.route_node_map.get_intra_route_idx(ejected_id.id()) - 1;
    i_t ejected_delivery_idx = ejected_pickup_idx;
    if constexpr (REQUEST == request_t::PDP) {
      ejected_delivery_idx = solution.route_node_map.get_intra_route_idx(ejected_id.delivery) - 2;
    }
    cuopt_assert(ejected_pickup_idx >= 0 && ejected_delivery_idx >= 0,
                 "Indices should be positive");
    cuopt_assert(ejected_pickup_idx <= ejected_delivery_idx, "Delivery should be after pickup");
    record_found_request_insertion<i_t, f_t, REQUEST, search_type, insert_unserviced>(
      solution,
      move_candidates,
      ejected_id.id(),
      ejected_pickup_idx,
      ejected_id.id(),
      route,
      compacted_requests,
      sh_buf_temp,
      excess_curr_route,
      thread_rng);
    __syncthreads();  // do we need this?
    // second part of insertion to ejected delivery position only happens in the case of PDP
    if constexpr (REQUEST == request_t::PDP) {
      record_found_request_insertion<i_t, f_t, REQUEST, search_type, insert_unserviced>(
        solution,
        move_candidates,
        ejected_id.id(),
        ejected_delivery_idx,
        ejected_id.delivery,
        route,
        compacted_requests,
        sh_buf_temp,
        excess_curr_route,
        thread_rng);
      __syncthreads();  // do we need this?
    }
  } else {
    // find the pickup insertion index according to the blockid
    // to keep the load balance, loop over 4 positions
    i_t relocate_route_start_idx =
      (blockIdx.x - solution.get_num_requests()) % move_candidates.number_of_blocks_per_ls_route;
    const i_t n_nodes_route = route.get_num_nodes();
    for (i_t pickup_insertion = relocate_route_start_idx; pickup_insertion < n_nodes_route;
         pickup_insertion += move_candidates.number_of_blocks_per_ls_route) {
      cand_t candidate = cand_t::create<search_type>();
      auto size_of_compacted_requests =
        find_request_insertion<i_t, f_t, REQUEST, search_type, true, insert_unserviced>(
          solution,
          move_candidates,
          ejected_id.id(),
          pickup_insertion,
          route,
          &candidate,
          compacted_requests,
          sh_buf_temp,
          excess_curr_route,
          thread_rng);
      if (threadIdx.x < size_of_compacted_requests && candidate.is_valid(search_type)) {
        cuopt_assert(ejected_id.id() != solution.get_num_orders() + solution.n_routes,
                     "Special node cannot participate in a move");
        // record the moves found in this block. we need critical section here because requets from
        // multiple blocks will try to save to the same pseudo route
        move_candidates.record_candidate_thread_safe(
          candidate, compacted_requests[threadIdx.x], ejected_id.id());
      }
      __syncthreads();
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST, search_type_t search_type>
DI bool is_ejected_route(typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                         typename move_candidates_t<i_t, f_t>::view_t& move_candidates)
{
  if constexpr (search_type == search_type_t::IMPROVE) {
    return blockIdx.x < solution.get_num_requests() * move_candidates.number_of_blocks_per_ls_route;
  } else {
    return blockIdx.x < solution.get_num_requests();
  }
}

template <typename i_t, typename f_t, request_t REQUEST, search_type_t search_type>
DI auto get_ejected_id(typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                       typename move_candidates_t<i_t, f_t>::view_t& move_candidates)
{
  if constexpr (search_type == search_type_t::IMPROVE) {
    return solution.get_request(blockIdx.x / move_candidates.number_of_blocks_per_ls_route);
  } else {
    return solution.get_request(blockIdx.x);
  }
}

template <typename i_t, typename f_t, request_t REQUEST, search_type_t search_type>
DI i_t get_unejected_route_id(typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                              typename move_candidates_t<i_t, f_t>::view_t& move_candidates)
{
  if constexpr (search_type == search_type_t::IMPROVE) {
    cuopt_assert(blockIdx.x < (solution.get_num_requests() + solution.n_routes) *
                                move_candidates.number_of_blocks_per_ls_route,
                 "Too many blocks launched for find_insertions!");
    return (blockIdx.x -
            solution.get_num_requests() * move_candidates.number_of_blocks_per_ls_route) /
           move_candidates.number_of_blocks_per_ls_route;

  } else {
    cuopt_assert(blockIdx.x < solution.get_num_requests() +
                                solution.n_routes * move_candidates.number_of_blocks_per_ls_route,
                 "Too many blocks launched for find_insertions!");
    return (blockIdx.x - solution.get_num_requests()) /
           move_candidates.number_of_blocks_per_ls_route;
  }
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          search_type_t search_type,
          bool insert_unserviced>
__global__ void find_insertions_kernel(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                                       typename move_candidates_t<i_t, f_t>::view_t move_candidates,
                                       int64_t seed)
{
  extern __shared__ i_t shmem[];

  i_t route_idx;
  request_id_t<REQUEST> ejected_id;

  [[maybe_unused]] raft::random::PCGenerator thread_rng(
    seed + (threadIdx.x + blockIdx.x * blockDim.x),
    uint64_t(solution.solution_id * (threadIdx.x + blockIdx.x * blockDim.x)),
    0);

  auto compacted_requests =
    raft::device_span<uint16_t>((uint16_t*)(shmem), solution.get_num_requests());
  auto sh_buf_temp = raft::device_span<uint16_t>(
    (uint16_t*)(compacted_requests.data() + solution.get_num_requests()),
    solution.get_num_requests());

  i_t aligned_bytes = raft::alignTo((sizeof(uint16_t) * (2 * solution.get_num_requests())),
                                    sizeof(infeasible_cost_t));
  __syncthreads();

  typename route_t<i_t, f_t, REQUEST>::view_t route;
  bool is_ejected = is_ejected_route<i_t, f_t, REQUEST, search_type>(solution, move_candidates);

  auto& route_node_map = solution.route_node_map;
  // if the route is an ejected route
  if (is_ejected) {
    ejected_id = get_ejected_id<i_t, f_t, REQUEST, search_type>(solution, move_candidates);
    route_idx  = route_node_map.get_route_id(ejected_id.id());
    if (route_idx == -1) {
      if constexpr (REQUEST == request_t::PDP) {
        cuopt_assert(route_node_map.get_route_id(ejected_id.delivery) == -1,
                     "If pickup is ejected, delivery should also be ejected");
      }
      return;
    }
    cuopt_assert(route_node_map.get_route_id(ejected_id.id()) == route_idx,
                 "Pickup / Delivery should be on the same route");
    const auto global_route = solution.routes[route_idx];
    route                   = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
      (i_t*)(((uint8_t*)shmem) + aligned_bytes), global_route, global_route.get_num_nodes());
    __syncthreads();
    // get the temp route that has 1 ejected request in it
    compute_temp_route<i_t, f_t, REQUEST>(
      route,
      global_route,
      global_route.get_num_nodes(),
      solution,
      ejected_id,
      solution.route_node_map.get_intra_route_idx(ejected_id.id()));
    __syncthreads();
    // record ejection costs of each PD request as a move from the special node to the ejected
    // node we only record the routes having at least l request in it, to prevent route reduction
    if (route.get_num_service_nodes() >= request_info_t<i_t, REQUEST>::size()) {
      const auto special_node_id = solution.get_num_orders() + solution.n_routes;
      const auto cost_difference =
        route.get_cost(move_candidates.include_objective, move_candidates.weights) -
        global_route.get_cost(move_candidates.include_objective, move_candidates.weights);

      cand_t curr_cand = move_candidates_t<i_t, f_t>::make_candidate(0, 0, 0, 0, cost_difference);
      if (threadIdx.x == 0) {
        cuopt_assert(ejected_id.id() != solution.get_num_orders() + solution.n_routes,
                     "Special node cannot participate in a move");
        // since we use load balancing and multiple blocks can handle one route
        // we need to do a threadsafe recording. the values will be exactly the same but we avoid
        // race conditions
        move_candidates.record_candidate_thread_safe(curr_cand, special_node_id, ejected_id.id());
      }
    }
  }
  // if the route is not ejected, i.e the ejected_node is a pseudo_node
  else {
    route_idx = get_unejected_route_id<i_t, f_t, REQUEST, search_type>(solution, move_candidates);
    // pseudo_node ids start from the
    ejected_id.id()         = route_idx + solution.get_num_orders();
    const auto global_route = solution.routes[route_idx];
    route                   = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
      (i_t*)(((uint8_t*)shmem) + aligned_bytes), global_route, global_route.get_num_nodes());
    __syncthreads();
    route.copy_from(global_route);
  }
  __syncthreads();

  const double excess_curr_route =
    solution.routes[route_idx].get_weighted_excess(move_candidates.weights);

  // for CROSS moves, we will do only 4 checks in total:
  //  1) insert pickup node in place of pickup ejection and find best delivery insertion
  //  2) insert delivery node in place of pickup ejection and find best pickup insertion
  //  3) insert pickup node in place of delivery ejection and find best delivery insertion
  //  4) insert delivery node in place of delivery ejection and find best pickup insertion
  // for RANDOM and IMPROVE, we will do a full route search
  if constexpr (search_type == search_type_t::CROSS || search_type == search_type_t::RANDOM) {
    // we want to insert just after
    find_cross_relocate_insertion<i_t, f_t, REQUEST, search_type, insert_unserviced>(
      solution,
      move_candidates,
      ejected_id,
      route,
      compacted_requests,
      sh_buf_temp,
      excess_curr_route,
      is_ejected,
      &thread_rng);
  }
  // do a full search
  else {
    const i_t n_nodes_route = route.get_num_nodes();
    i_t start_idx           = blockIdx.x % move_candidates.number_of_blocks_per_ls_route;
    for (i_t pickup_insertion = start_idx; pickup_insertion < n_nodes_route;
         pickup_insertion += move_candidates.number_of_blocks_per_ls_route) {
      cand_t candidate = cand_t::create<search_type>();
      auto size_of_compacted_requests =
        find_request_insertion<i_t, f_t, REQUEST, search_type, true, insert_unserviced>(
          solution,
          move_candidates,
          ejected_id.id(),
          pickup_insertion,
          route,
          &candidate,
          compacted_requests,
          sh_buf_temp,
          excess_curr_route,
          &thread_rng);
      if (threadIdx.x < size_of_compacted_requests && candidate.is_valid(search_type)) {
        cuopt_assert(ejected_id.id() != solution.get_num_orders() + solution.n_routes,
                     "Special node cannot participate in a move");
        // for non cross moves we only save pickup id moves as they represent what is inserted and
        // what is ejected(in terms of route)
        if (move_candidates.number_of_blocks_per_ls_route == 1) {
          move_candidates.record_if_better(
            candidate, compacted_requests[threadIdx.x], ejected_id.id());
        } else {
          move_candidates.record_candidate_thread_safe(
            candidate, compacted_requests[threadIdx.x], ejected_id.id());
        }
      }
      __syncthreads();
    }
  }
}

template <typename i_t, request_t REQUEST>
i_t get_number_of_blocks_per_ls_route(i_t n_routes)
{
  if (REQUEST == request_t::VRP) {
    return 1;
  } else {
    return std::max<i_t>(1, std::ceil<i_t>(-0.09 * n_routes + 11.1));
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
size_t get_sh_size_for_compute_insertions(solution_t<i_t, f_t, REQUEST> const& sol)
{
  size_t aligned_bytes =
    raft::alignTo((sizeof(uint16_t) * (sol.get_num_requests() * 2)), sizeof(infeasible_cost_t));
  return aligned_bytes + sol.get_temp_route_shared_size();
}

template <typename i_t, typename f_t, request_t REQUEST>
void find_insertions(solution_t<i_t, f_t, REQUEST>& sol,
                     move_candidates_t<i_t, f_t>& move_candidates,
                     search_type_t search_type)
{
  auto name = "find_insertions";
  if (search_type == search_type_t::RANDOM) { name = "random_find_insertions"; }

  constexpr bool insert_unserviced = false;
  raft::common::nvtx::range fun_scope(name);
  i_t TPB = get_n_viable<REQUEST>();
  // the formula is obtained with a linear regression across different instances
  move_candidates.number_of_blocks_per_ls_route =
    get_number_of_blocks_per_ls_route<i_t, REQUEST>(sol.get_n_routes());
  i_t n_blocks =
    move_candidates.number_of_blocks_per_ls_route * (sol.get_n_routes() + sol.get_num_requests());

  sol.check_routes_can_insert_and_get_sh_size();

  size_t shared_size = get_sh_size_for_compute_insertions(sol);

  if (search_type == search_type_t::IMPROVE) {
    bool is_set = set_shmem_of_kernel(
      find_insertions_kernel<i_t, f_t, REQUEST, search_type_t::IMPROVE, insert_unserviced>,
      shared_size);
    cuopt_assert(is_set,
                 "Not enough shared memory on device for computing local search insertions!");
    cuopt_expects(is_set, error_type_t::OutOfMemoryError, "Not enough shared memory on device");
    find_insertions_kernel<i_t, f_t, REQUEST, search_type_t::IMPROVE, insert_unserviced>
      <<<n_blocks, TPB, shared_size, sol.sol_handle->get_stream()>>>(
        sol.view(), move_candidates.view(), seed_generator::get_seed());
  } else {
    // for cross the load-balance factor is always 4
    move_candidates.number_of_blocks_per_ls_route =
      std::max(1, sol.get_max_active_nodes_for_all_routes() / 4);
    if (search_type == search_type_t::CROSS) {
      n_blocks =
        move_candidates.number_of_blocks_per_ls_route * sol.get_n_routes() + sol.get_num_requests();
      bool is_set = set_shmem_of_kernel(
        find_insertions_kernel<i_t, f_t, REQUEST, search_type_t::CROSS, insert_unserviced>,
        shared_size);
      cuopt_assert(is_set,
                   "Not enough shared memory on device for computing local search insertions!");
      cuopt_expects(is_set, error_type_t::OutOfMemoryError, "Not enough shared memory on device");
      find_insertions_kernel<i_t, f_t, REQUEST, search_type_t::CROSS, insert_unserviced>
        <<<n_blocks, TPB, shared_size, sol.sol_handle->get_stream()>>>(
          sol.view(), move_candidates.view(), seed_generator::get_seed());
    } else if (search_type == search_type_t::RANDOM) {
      // we don't search for relocates in random.
      n_blocks    = sol.get_num_requests();
      bool is_set = set_shmem_of_kernel(
        find_insertions_kernel<i_t, f_t, REQUEST, search_type_t::RANDOM, insert_unserviced>,
        shared_size);
      cuopt_assert(is_set,
                   "Not enough shared memory on device for computing local search insertions!");
      cuopt_expects(is_set, error_type_t::OutOfMemoryError, "Not enough shared memory on device");
      find_insertions_kernel<i_t, f_t, REQUEST, search_type_t::RANDOM, insert_unserviced>
        <<<n_blocks, TPB, shared_size, sol.sol_handle->get_stream()>>>(
          sol.view(), move_candidates.view(), seed_generator::get_seed());
    }
  }
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
  sol.sol_handle->sync_stream();
}

template <typename i_t, typename f_t, request_t REQUEST>
void find_unserviced_insertions(solution_t<i_t, f_t, REQUEST>& sol,
                                move_candidates_t<i_t, f_t>& move_candidates)
{
  constexpr bool insert_unserviced = true;

  auto name = "find_unserviced_insertions";
  raft::common::nvtx::range fun_scope(name);
  constexpr auto const TPB = get_n_viable<REQUEST>();
  // the formula is obtained with a linear regression across different instances
  move_candidates.number_of_blocks_per_ls_route =
    get_number_of_blocks_per_ls_route<i_t, REQUEST>(sol.get_n_routes());
  i_t n_blocks =
    move_candidates.number_of_blocks_per_ls_route * (sol.get_n_routes() + sol.get_num_requests());

  sol.check_routes_can_insert_and_get_sh_size();

  size_t shared_size = get_sh_size_for_compute_insertions(sol);

  bool is_set = set_shmem_of_kernel(
    find_insertions_kernel<i_t, f_t, REQUEST, search_type_t::IMPROVE, insert_unserviced>,
    shared_size);
  cuopt_assert(is_set, "Not enough shared memory on device for computing local search insertions!");
  cuopt_expects(is_set, error_type_t::OutOfMemoryError, "Not enough shared memory on device");
  find_insertions_kernel<i_t, f_t, REQUEST, search_type_t::IMPROVE, insert_unserviced>
    <<<n_blocks, TPB, shared_size, sol.sol_handle->get_stream()>>>(
      sol.view(), move_candidates.view(), seed_generator::get_seed());
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
  sol.sol_handle->sync_stream();
}

template void find_insertions<int, float>(solution_t<int, float, request_t::PDP>& sol,
                                          move_candidates_t<int, float>& move_candidates,
                                          search_type_t);
template void find_insertions<int, float>(solution_t<int, float, request_t::VRP>& sol,
                                          move_candidates_t<int, float>& move_candidates,
                                          search_type_t);

template void find_unserviced_insertions<int, float>(
  solution_t<int, float, request_t::PDP>& sol, move_candidates_t<int, float>& move_candidates);
template void find_unserviced_insertions<int, float>(
  solution_t<int, float, request_t::VRP>& sol, move_candidates_t<int, float>& move_candidates);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
