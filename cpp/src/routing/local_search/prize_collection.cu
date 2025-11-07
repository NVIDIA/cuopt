/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/cuda_helpers.cuh>
#include "../ges/execute_insertion.cuh"
#include "../solution/solution.cuh"
#include "../utilities/cuopt_utils.cuh"
#include "compute_ejections.cuh"
#include "compute_insertions.cuh"
#include "local_search.cuh"
#include "routing/utilities/cuopt_utils.cuh"

#include <thrust/fill.h>
#include <thrust/remove.h>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void get_best_move_per_route(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  typename move_candidates_t<i_t, f_t>::view_t move_candidates)
{
  auto& best_cand_per_route = move_candidates.prize_move_candidates.best_cand_per_route;
  auto& route_locks         = move_candidates.prize_move_candidates.locks_per_route;

  i_t matrix_width  = move_candidates.cand_matrix.matrix_width;
  i_t matrix_height = move_candidates.cand_matrix.matrix_height;
  size_t n_moves    = matrix_width * matrix_height;
  for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n_moves; i += blockDim.x * gridDim.x) {
    const auto& curr_cand = move_candidates.cand_matrix.get_candidate(i);
    if (curr_cand.cost_counter.cost < -EPSILON) {
      i_t source = i / matrix_width;
      i_t sink   = i % matrix_width;

      i_t ejected_node_id   = sink;
      i_t insertion_node_id = source;

      i_t route_id = -1;

      bool is_ejected = ejected_node_id < solution.get_num_orders();
      if (is_ejected) {
        route_id = solution.route_node_map.get_route_id(ejected_node_id);
      } else {
        route_id        = ejected_node_id - solution.get_num_orders();
        ejected_node_id = solution.get_num_orders();
      }

      cuopt_assert(route_id >= 0 && route_id < solution.n_routes,
                   "route id corresponding to move should be in range!");

      // add the info of which node has to be ejected
      i_t pickup_insertion, delivery_insertion, dummy1, dummy2;
      double cost_delta;
      move_candidates_t<i_t, f_t>::get_candidate(
        curr_cand, pickup_insertion, delivery_insertion, dummy1, dummy2, cost_delta);

      auto updated_cand = prize_cand_t(
        pickup_insertion, delivery_insertion, ejected_node_id, insertion_node_id, cost_delta);

      acquire_lock(&route_locks[route_id]);
      if (curr_cand.cost_counter.cost < best_cand_per_route[route_id].cost) {
        best_cand_per_route[route_id] = updated_cand;
      }
      release_lock(&route_locks[route_id]);
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
DI bool is_candidate_valid(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                           raft::device_span<prize_cand_t> best_cand_per_route,
                           i_t route_id)
{
  auto cand_i  = best_cand_per_route[route_id];
  i_t p1       = cand_i.pickup_insertion;
  i_t d1       = cand_i.delivery_insertion;
  i_t e1       = cand_i.ejected_node_id;
  i_t i1       = cand_i.inserted_node_id;
  double cost1 = cand_i.cost;

  if (i1 < solution.get_num_orders()) {
    __shared__ int num_routes_with_same_insertion;
    if (threadIdx.x == 0) { num_routes_with_same_insertion = 0; }
    __syncthreads();

    for (int j = threadIdx.x; j < solution.n_routes; j += blockDim.x) {
      // take an early exit
      if (num_routes_with_same_insertion > 0) { break; }
      if (route_id == j) continue;
      auto cand_j = best_cand_per_route[j];

      i_t p2       = cand_j.pickup_insertion;
      i_t d2       = cand_j.delivery_insertion;
      i_t e2       = cand_j.ejected_node_id;
      i_t i2       = cand_j.inserted_node_id;
      double cost2 = cand_j.cost;

      if (i2 < solution.get_num_orders() && i2 == i1 && cost1 < 0 && cost2 < 0) {
        if (cost2 < cost1 || (cost1 == cost2 && j < route_id)) {
          atomicAdd_block(&num_routes_with_same_insertion, 1);
          __threadfence_block();
          break;
        }
      }
    }

    __syncthreads();
    return num_routes_with_same_insertion == 0;
  }

  // pure ejection is always valid
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void execute_moves(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                              typename move_candidates_t<i_t, f_t>::view_t move_candidates)
{
  auto& best_cand_per_route = move_candidates.prize_move_candidates.best_cand_per_route;

  i_t route_id         = blockIdx.x;
  auto curr_route_cand = best_cand_per_route[route_id];

  if (curr_route_cand.cost >= 0) { return; }

  if (!is_candidate_valid<i_t, f_t, REQUEST>(solution, best_cand_per_route, route_id)) { return; }

  extern __shared__ i_t shmem[];

  auto original_route = solution.routes[route_id];

  auto sh_route = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    shmem, original_route, original_route.get_num_nodes() + request_info_t<i_t, REQUEST>::size());

  request_id_t<REQUEST> ejected_request;
  request_id_t<REQUEST> insertion_locations;
  request_id_t<REQUEST> insertion_request;

  insertion_locations.id() = curr_route_cand.pickup_insertion;
  ejected_request.id()     = curr_route_cand.ejected_node_id;
  insertion_request.id()   = curr_route_cand.inserted_node_id;
  double cost              = curr_route_cand.cost;
  if constexpr (REQUEST == request_t::PDP) {
    if (ejected_request.pickup < solution.get_num_orders()) {
      ejected_request.delivery = solution.problem.order_info.pair_indices[ejected_request.pickup];
    }
    if (insertion_request.pickup < solution.get_num_orders()) {
      insertion_request.delivery =
        solution.problem.order_info.pair_indices[insertion_request.pickup];
    }
    insertion_locations.delivery = curr_route_cand.delivery_insertion;
  }

  cuopt_assert(ejected_request.id() < solution.get_num_orders() ||
                 insertion_request.id() < solution.get_num_orders(),
               "There should be at least one insertion or one ejection!");

  // If the move is related to an ejected route
  if (ejected_request.id() < solution.get_num_orders()) {
    cuopt_assert(solution.route_node_map.is_node_served(ejected_request.id()),
                 "we should not be ejecting nodes that are unserviced!");
    cuopt_assert(solution.route_node_map.get_route_id(ejected_request.id()) == route_id,
                 "ejected node should belong to this route!");

    compute_temp_route<i_t, f_t, REQUEST>(
      sh_route,
      original_route,
      original_route.get_num_nodes(),
      solution,
      ejected_request,
      solution.route_node_map.get_intra_route_idx(ejected_request.id()));

    solution.route_node_map.reset_node(ejected_request.id());
    if constexpr (REQUEST == request_t::PDP) {
      solution.route_node_map.reset_node(ejected_request.delivery);
    }

    __syncthreads();
    sh_route.compute_intra_indices(solution.route_node_map);
  } else {
    sh_route.copy_from(original_route);
  }

  __syncthreads();

  // If the move has insertions
  if (insertion_request.id() < solution.get_num_orders()) {
    cuopt_assert(!solution.route_node_map.is_node_served(insertion_request.id()),
                 "we should not be inserting nodes that are already served!");
    if (threadIdx.x == 0) {
      const auto request = create_request<i_t, f_t, REQUEST>(solution.problem, insertion_request);
      execute_insert<i_t, f_t, REQUEST>(solution, sh_route, insertion_locations, &request);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    solution.routes_to_copy[route_id]   = 1;
    solution.routes_to_search[route_id] = 1;
  }
  original_route.copy_from(sh_route);
}

template <typename i_t, typename f_t, request_t REQUEST>
bool local_search_t<i_t, f_t, REQUEST>::perform_prize_collection(solution_t<i_t, f_t, REQUEST>& sol)
{
  // need to reset whole cand matrix, because we are filling it before getting best cand per route
  move_candidates.reset(sol.sol_handle);

  sol.global_runtime_checks(false, false, "perform_prize_collection_begin");

  calculate_route_compatibility(sol);
  find_unserviced_insertions<i_t, f_t, REQUEST>(sol, move_candidates);

  move_candidates.prize_move_candidates.reset(sol.sol_handle);

  constexpr const auto TPB = 128;
  size_t n_moves =
    move_candidates.cand_matrix.matrix_width * move_candidates.cand_matrix.matrix_height;

  size_t n_blocks    = std::min((n_moves + TPB - 1) / TPB, (size_t)65536);
  size_t shared_size = 0;

  get_best_move_per_route<i_t, f_t, REQUEST>
    <<<n_blocks, TPB, shared_size, sol.sol_handle->get_stream()>>>(sol.view(),
                                                                   move_candidates.view());
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());

  if (!move_candidates.prize_move_candidates.has_improving_routes(sol.sol_handle)) { return false; }

  n_blocks    = sol.get_n_routes();
  shared_size = sol.check_routes_can_insert_and_get_sh_size(request_info_t<i_t, REQUEST>::size());
  if (!set_shmem_of_kernel(execute_moves<i_t, f_t, REQUEST>, shared_size)) { return false; }
  execute_moves<i_t, f_t, REQUEST><<<n_blocks, TPB, shared_size, sol.sol_handle->get_stream()>>>(
    sol.view(), move_candidates.view());
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());

  sol.compute_cost();

  sol.global_runtime_checks(false, false, "perform_prize_collection_end");
  return true;
}

template bool local_search_t<int, float, request_t::PDP>::perform_prize_collection(
  solution_t<int, float, request_t::PDP>& sol);
template bool local_search_t<int, float, request_t::VRP>::perform_prize_collection(
  solution_t<int, float, request_t::VRP>& sol);
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
