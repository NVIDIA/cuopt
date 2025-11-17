/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/cuda_helpers.cuh>
#include "../solution/solution.cuh"
#include "compute_delivery_insertions.cuh"
#include "compute_fragment_ejections.cuh"
#include "found_solution.cuh"

namespace cuopt {
namespace routing {
namespace detail {

template <int BLOCK_SIZE, typename i_t, typename f_t, request_t REQUEST>
__global__ void kernel_get_best_insertion_ejection_solution(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  const request_info_t<i_t, REQUEST>* request_info,
  i_t* p_scores,
  i_t fragment_size,
  i_t fragment_step,
  feasible_move_t feasible_candidates,
  int64_t seed)
{
  // TODO @hugo: this might cause misaligned access with VRP, as shmem is i_t and it will might end
  // at 4 bytes alignment for PDP it is always multiples of 2, that's why it ends at 8 byte
  // alignment
  extern __shared__ i_t shmem[];
  cuopt_assert(fragment_size > 0, "There should be at least one node for ejection");
  cuopt_assert(request_info != nullptr, "Request id should not be nullptr");
  cuopt_assert(request_info->is_valid(solution.problem.order_info.depot_included),
               "Request id should be positive");
  cuopt_assert(fragment_size >= fragment_step, "Fragement size should be bigger or equal to step");

  // First request of the fragment

  const i_t frag_to_delete = blockIdx.x % fragment_step + (fragment_size - fragment_step) + 1;
  cuopt_assert(frag_to_delete > 0 && frag_to_delete <= fragment_size, "Invalid frag to delete");
  const auto request_id = solution.get_request(blockIdx.x / fragment_step);
  request_id.check(solution.get_num_orders());

  const auto [route_id, intra_pickup_id] =
    solution.route_node_map.get_route_id_and_intra_idx(request_id.id());

  cuopt_assert(route_id >= 0 || route_id == -1,
               "Route id from route_id_per_node should be positive or flagged");
  cuopt_assert(route_id < solution.n_routes, "Route id should be smalled than number of routes");
  // Discard deleted requests
  if (route_id == -1) return;

  auto& route             = solution.routes[route_id];
  const auto route_length = route.get_num_nodes();

  cuopt_assert(route_length > 1, "Route length should be greated than one");
  cuopt_assert(intra_pickup_id > 0, "Intra pickup id must be stricly positive");
  cuopt_assert(intra_pickup_id < route_length, "Intra pickup id must be inferior to route length");
  if constexpr (REQUEST == request_t::PDP) {
    cuopt_assert(route.requests().is_pickup_node(intra_pickup_id),
                 "Intra pickup doesn't correspond to a pickup");
  }
  i_t* to_delete = shmem;

  const i_t p_score = fill_to_delete<BLOCK_SIZE, i_t, f_t, REQUEST>(
    solution, route, intra_pickup_id, to_delete, frag_to_delete, p_scores);
  // Not enough to delete, discard the block
  // or if the p_score found is bigger than the p_score of the request
  if (p_score == -1 || p_score >= p_scores[request_info->info.node()]) return;
  // Sync to_delete array before sorting it
  __syncthreads();
  cuopt_assert(is_positive(to_delete, frag_to_delete * request_info_t<i_t, REQUEST>::size()),
               "Array contains negative or 0 values");

  sort_to_delete<BLOCK_SIZE, i_t, REQUEST>(to_delete, frag_to_delete);
  cuopt_assert(is_sorted(to_delete, frag_to_delete * request_info_t<i_t, REQUEST>::size()),
               "Array is not sorted after calling sort_to_delete");
  cuopt_assert(is_positive(to_delete, frag_to_delete * request_info_t<i_t, REQUEST>::size()),
               "Array contains negative or 0 values");

  // Copy only the segements that are not concerned by the deletion
  auto aligned_sz =
    raft::alignTo((size_t)(fragment_size * request_info_t<i_t, REQUEST>::size() * sizeof(i_t)),
                  sizeof(infeasible_cost_t));

  auto temp_route = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    (i_t*)(((uint8_t*)shmem) + aligned_sz),
    route,
    route_length - request_info_t<i_t, REQUEST>::size() * frag_to_delete);
  __syncthreads();
  temp_route.copy_route_data_after_ejections(
    route, to_delete, frag_to_delete * request_info_t<i_t, REQUEST>::size());
  __syncthreads();

  // Recompute data on the new route
  if (threadIdx.x == 0) {
    route_t<i_t, f_t, REQUEST>::view_t::compute_forward_in_between(
      temp_route, 0, route_length - request_info_t<i_t, REQUEST>::size() * frag_to_delete);
    route_t<i_t, f_t, REQUEST>::view_t::compute_backward_in_between(
      temp_route, 0, route_length - request_info_t<i_t, REQUEST>::size() * frag_to_delete);
  }

  __syncthreads();

  find_all_delivery_insertions<BLOCK_SIZE, false, i_t, f_t, REQUEST>(solution,
                                                                     temp_route,
                                                                     request_info,
                                                                     feasible_candidates,
                                                                     seed,
                                                                     p_score,
                                                                     frag_to_delete,
                                                                     fragment_step);
}

template __global__ void
kernel_get_best_insertion_ejection_solution<32, int, float, request_t::PDP>(
  typename solution_t<int, float, request_t::PDP>::view_t solution,
  const request_info_t<int, request_t::PDP>* request_id,
  int* p_scores,
  int fragment_size,
  int fragment_step,
  feasible_move_t feasible_candidates,
  int64_t seed);
template __global__ void
kernel_get_best_insertion_ejection_solution<64, int, float, request_t::PDP>(
  typename solution_t<int, float, request_t::PDP>::view_t solution,
  const request_info_t<int, request_t::PDP>* request_id,
  int* p_scores,
  int fragment_size,
  int fragment_step,
  feasible_move_t feasible_candidates,
  int64_t seed);
template __global__ void
kernel_get_best_insertion_ejection_solution<128, int, float, request_t::PDP>(
  typename solution_t<int, float, request_t::PDP>::view_t solution,
  const request_info_t<int, request_t::PDP>* request_id,
  int* p_scores,
  int fragment_size,
  int fragment_step,
  feasible_move_t feasible_candidates,
  int64_t seed);
template __global__ void
kernel_get_best_insertion_ejection_solution<512, int, float, request_t::PDP>(
  typename solution_t<int, float, request_t::PDP>::view_t solution,
  const request_info_t<int, request_t::PDP>* request_id,
  int* p_scores,
  int fragment_size,
  int fragment_step,
  feasible_move_t feasible_candidates,
  int64_t seed);

template __global__ void
kernel_get_best_insertion_ejection_solution<32, int, float, request_t::VRP>(
  typename solution_t<int, float, request_t::VRP>::view_t solution,
  const request_info_t<int, request_t::VRP>* request_id,
  int* p_scores,
  int fragment_size,
  int fragment_step,
  feasible_move_t feasible_candidates,
  int64_t seed);
template __global__ void
kernel_get_best_insertion_ejection_solution<64, int, float, request_t::VRP>(
  typename solution_t<int, float, request_t::VRP>::view_t solution,
  const request_info_t<int, request_t::VRP>* request_id,
  int* p_scores,
  int fragment_size,
  int fragment_step,
  feasible_move_t feasible_candidates,
  int64_t seed);
template __global__ void
kernel_get_best_insertion_ejection_solution<128, int, float, request_t::VRP>(
  typename solution_t<int, float, request_t::VRP>::view_t solution,
  const request_info_t<int, request_t::VRP>* request_id,
  int* p_scores,
  int fragment_size,
  int fragment_step,
  feasible_move_t feasible_candidates,
  int64_t seed);
template __global__ void
kernel_get_best_insertion_ejection_solution<512, int, float, request_t::VRP>(
  typename solution_t<int, float, request_t::VRP>::view_t solution,
  const request_info_t<int, request_t::VRP>* request_id,
  int* p_scores,
  int fragment_size,
  int fragment_step,
  feasible_move_t feasible_candidates,
  int64_t seed);
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
