/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../solution/solution.cuh"
#include "found_solution.cuh"

namespace cuopt {
namespace routing {
namespace detail {

/* The goal of this kernel is to have each block, delete consecutive requests from a route
 * To then insert the given request
 *
 * Each block handles a starting request (pickup)
 * Each thread then counts the amount of requests (pickups) it can delete on its right part of the
 * route (starting at its startin request) If it can delete enought (with respect to fragement
 * size), it stores the consecutive indices it should delete Then a tmp shared route is created
 * containing all nodes apart from the one we elected for deletion Then we try all insertion
 * possible of the request
 */
template <int BLOCK_SIZE, typename i_t, typename f_t, request_t REQUEST>
__global__ void kernel_get_best_insertion_ejection_solution(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  const request_info_t<i_t, REQUEST>* request_id,
  i_t* p_scores,
  i_t fragment_size,
  i_t fragment_step,
  feasible_move_t feasible_candidates,
  int64_t seed);

template <int BLOCK_SIZE,
          typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
DI i_t fill_to_delete(const typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                      const typename route_t<i_t, f_t, REQUEST>::view_t& route,
                      i_t intra_pickup_id,
                      i_t* to_delete,
                      i_t fragment_size,
                      [[maybe_unused]] i_t* p_scores = nullptr)
{
  cuopt_assert(BLOCK_SIZE % raft::WarpSize == 0, "Block size should be modulo of warp size");

  const auto route_length = route.get_num_nodes();
  cuopt_assert(route_length >= 1, "Route lenght should be stricly positive");

  if (intra_pickup_id + fragment_size >= route_length) {
    // Not enough to delete vrp case
    return -1;
  }

  // Browse throught the mask, starting at the intra_pickup_id to get the fragement
  __shared__ i_t p_score;
  // TODO: For now single threaded, should be a parallel compact
  if (threadIdx.x == 0) {
    p_score     = 0;
    i_t counter = 0;
    for (int i = intra_pickup_id; counter < fragment_size * request_info_t<i_t, REQUEST>::size();
         ++i) {
      // if there is a break node, don't consider this fragment
      const auto& node_info = route.requests().node_info[i];
      if (node_info.is_break()) {
        p_score = -1;
        break;
      }

      cuopt_assert(i < route_length, "Intra pickup id can only be smaller than route size");
      cuopt_assert(i > 0, "Intra pickup id can only be stricly positive");
      if (p_scores != nullptr) p_score += p_scores[node_info.node()];
      to_delete[counter] = i;
      ++counter;
    }
    cuopt_assert(p_score == -1 || counter == (fragment_size * request_info_t<i_t, REQUEST>::size()),
                 "Not enough elements got inserted");
    if (p_scores != nullptr && p_score != -1) {
      cuopt_assert(p_score >= (fragment_size / request_info_t<i_t, REQUEST>::size()),
                   "Inital p value is 1, total should be at least "
                   "fragment_size/request_info_t<i_t, REQUEST>::size()");
    }
  }

  __syncthreads();
  return p_score;
}

template <int BLOCK_SIZE,
          typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
DI i_t fill_to_delete(const typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                      const typename route_t<i_t, f_t, REQUEST>::view_t& route,
                      i_t intra_pickup_id,
                      i_t* to_delete,
                      i_t fragment_size,
                      [[maybe_unused]] i_t* p_scores = nullptr)
{
  cuopt_assert(BLOCK_SIZE % raft::WarpSize == 0, "Block size should be modulo of warp size");

  const auto route_length = route.get_num_nodes();
  cuopt_assert(route_length >= 1, "Route lenght should be stricly positive");

  // Check for fragement spanning to large (no looping around)
  // Range covered goes from intra_pickup_id to intra_pickup_id + blockDim.x
  // Not include threadIdx.x in loop to always have full warps for __ballot_sync and shared writes
  __shared__ uint32_t s_pickups[BLOCK_SIZE / raft::WarpSize];
  uint32_t n_deletable_requests = 0;
  for (i_t i = intra_pickup_id; i < route_length; i += blockDim.x) {
    int is_pickup = 0;
    if (threadIdx.x + i < route_length) {
      cuopt_assert(threadIdx.x + i < route.max_nodes_per_route(),
                   "Indexing should not be greater than max size");
      is_pickup = route.requests().is_pickup_node(threadIdx.x + i);
    }
    const uint32_t pickups = __ballot_sync(~0, is_pickup);
    if (threadIdx.x % raft::WarpSize == 0) s_pickups[threadIdx.x / raft::WarpSize] = pickups;
    __syncthreads();
// Compute amount of requests on my right side including mine
#pragma unroll
    for (int j = 0; j < BLOCK_SIZE / raft::WarpSize; ++j)
      n_deletable_requests += __popc(s_pickups[j]);
    __syncthreads();  // To avoid race condition on next loop write
  }

  cuopt_assert(n_deletable_requests > 0, "There should be at least one pickup");
  // Discard the block if not enough to delete by returning a special val
  if (n_deletable_requests < fragment_size) { return -1; }

  // Browse throught the mask, starting at the intra_pickup_id to get the fragement
  __shared__ i_t p_score;
  if (threadIdx.x == 0) {
    p_score     = 0;
    i_t counter = 0;
    for (int i = intra_pickup_id; counter < fragment_size * 2; ++i) {
      // if there is a break node, don't consider this fragment
      const auto& node_info = route.requests().node_info[i];
      if (node_info.is_break()) {
        p_score = -1;
        break;
      }

      if (route.requests().is_pickup_node(i) && i != route_length) {
        to_delete[counter] = i;
        cuopt_assert(i < route_length, "Intra pickup id can only be smaller than route size");
        cuopt_assert(i > 0, "Intra pickup id can only be stricly positive");
        cuopt_assert(route.requests().is_pickup_node(i), "Node should be pickup");
        cuopt_assert(route.get_node(i).request.is_pickup(), "Node should be pickup");
        if (p_scores != nullptr) { p_score += p_scores[route.node_id(i)]; }
        ++counter;
        // TODO: should we store brother node and intra index in shared ?
        i_t brother_intra_idx =
          solution.route_node_map.get_intra_route_idx(route.requests().brother_info[i]);
        to_delete[counter] = brother_intra_idx;
        cuopt_assert(brother_intra_idx < route.get_num_nodes(),
                     "Intra delivery id can only be smaller than route size");
        cuopt_assert(brother_intra_idx > 0, "Intra delivery id can only be stricly positive");
        ++counter;
      }
    }
    cuopt_assert(p_score == -1 || counter == fragment_size * 2, "Not enough elements got inserted");
    if (p_scores != nullptr && p_score != -1) {
      cuopt_assert(p_score >= fragment_size / 2,
                   "Inital p value is 1, total should be at least fragment_size/2");
    }
  }

  __syncthreads();
  return p_score;
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
