/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../local_search/compute_ejections.cuh"
#include "compute_delivery_insertions.cuh"
#include "execute_insertion.cuh"
#include "guided_ejection_search.cuh"

#include <routing/utilities/cuopt_utils.cuh>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
__device__ void find_squeeze_pos(typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                                 const request_info_t<i_t, REQUEST>* request,
                                 cand_t* block_best_result,
                                 i_t* shmem,
                                 bool include_objective,
                                 infeasible_cost_t const& weights,
                                 double excess_limit,
                                 i_t route_id)
{
  __shared__ double reduction_buf[2 * raft::WarpSize];
  __shared__ i_t reduction_idx;
  const auto& dimensions_info = solution.problem.dimensions_info;
  auto request_node           = solution.get_request(request);
  auto gl_route               = solution.routes[route_id];
  auto sh_route               = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    shmem, gl_route, gl_route.get_num_nodes());
  __syncthreads();
  sh_route.copy_from(gl_route);
  __syncthreads();
  auto best_move = cand_t{0, 0, std::numeric_limits<double>::max()};
  for (i_t node_insertion_idx = threadIdx.x; node_insertion_idx < gl_route.get_num_nodes();
       node_insertion_idx += blockDim.x) {
    auto feasible_move = cand_t{0, 0, std::numeric_limits<double>::max()};
    find_request_insertion<i_t, f_t, REQUEST, pick_mode_t::COST_DELTA, insert_mode_t::SQUEEZE>(
      solution,
      sh_route,
      request_node,
      node_insertion_idx,
      include_objective,
      weights,
      excess_limit,
      &feasible_move,
      nullptr);
    if (feasible_move < best_move) { best_move = feasible_move; }
  }
  // get best move for this route/block
  double thread_value = best_move.cost_counter.cost;
  i_t thread_id       = threadIdx.x;

  block_reduce_ranked(thread_value, thread_id, reduction_buf, &reduction_idx);
  if (threadIdx.x == reduction_idx) {
    uint pair_2 = (route_id << 16) | (request_node.node().request.info.node());
    // use the second location to mark the route id
    best_move.pair_2   = pair_2;
    *block_best_result = best_move;
  }
  __syncthreads();
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void find_best_empty_route_move(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                                           cand_t* global_best,
                                           bool include_objective,
                                           infeasible_cost_t weights,
                                           double excess_limit)
{
  extern __shared__ i_t shbuf[];
  cand_t* cand = (cand_t*)shbuf;
  i_t* shmem   = (i_t*)&cand[1];

  auto n_routes    = solution.n_routes;
  auto request_id  = blockIdx.x / n_routes;
  auto to_route_id = blockIdx.x % n_routes;

  cuopt_assert(request_id < solution.get_num_requests(), "Invalid request id");

  auto to_route = solution.routes[to_route_id];
  // We want to relocate to empty routes only
  if (to_route.get_num_service_nodes() > 0) { return; }

  auto const tmp_request_id = solution.get_request(request_id);
  auto const request        = create_request<i_t, f_t, REQUEST>(solution.problem, tmp_request_id);
  const auto [from_route_id, _] =
    solution.route_node_map.get_route_id_and_intra_idx(request.info.node());

  // In case of prize collection, some nodes might not have been serviced
  if (from_route_id >= 0) {
    auto from_route = solution.routes[from_route_id];

    // Cannot relocate from a route that has one request or none
    if (from_route.get_num_service_nodes() <= request_info_t<i_t, REQUEST>::size()) { return; }
  }

  // Try to insert request to empty route. Sub optimal since we don't consider ejection of the
  // request. We are just trying to fill missing vehicles.
  find_squeeze_pos<i_t, f_t, REQUEST>(
    solution, &request, cand, shmem, include_objective, weights, excess_limit, to_route_id);

  if (threadIdx.x == 0) {
    acquire_lock(solution.lock);
    if (cand->cost_counter.cost < global_best->cost_counter.cost) { *global_best = *cand; }
    release_lock(solution.lock);
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void execute_best_empty_route_move(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution, cand_t* global_best)
{
  if (global_best->cost_counter.cost == std::numeric_limits<double>::max()) { return; }

  extern __shared__ i_t shmem[];

  // Eject request from it's original route and insert it to the best empty route

  i_t insertion_1, insertion_2, route_id, pickup_id;
  double cost_delta;
  move_candidates_t<i_t, f_t>::get_candidate(
    *global_best, insertion_1, insertion_2, route_id, pickup_id, cost_delta);
  const auto [original_route_id, intra_route_idx] =
    solution.route_node_map.get_route_id_and_intra_idx(pickup_id);

  request_id_t<REQUEST> request_id;
  if constexpr (REQUEST == request_t::PDP) {
    request_id =
      request_id_t<REQUEST>(pickup_id, solution.problem.order_info.pair_indices[pickup_id]);
  } else {
    request_id = request_id_t<REQUEST>(pickup_id);
  }

  // In case of prize collection, we can have unserviced nodes
  if (original_route_id >= 0) {
    auto original_route = solution.routes[original_route_id];

    auto temp_route = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
      shmem, original_route, original_route.get_num_nodes());
    __syncthreads();

    compute_temp_route<i_t, f_t, REQUEST>(
      temp_route,
      original_route,
      original_route.get_num_nodes(),
      solution,
      request_id,
      solution.route_node_map.get_intra_route_idx(request_id.id()));
    __syncthreads();
    temp_route.compute_intra_indices(solution.route_node_map);
    __syncthreads();
    solution.routes[temp_route.get_id()].copy_from(temp_route);
  }

  auto inserting_route = solution.routes[route_id];
  if (threadIdx.x == 0) {
    request_id_t<REQUEST> request_locations;
    if constexpr (REQUEST == request_t::PDP)
      request_locations = request_id_t<REQUEST>(insertion_1, insertion_2);
    else {
      request_locations = request_id_t<REQUEST>(insertion_1);
    }

    auto const request = create_request<i_t, f_t, REQUEST>(solution.problem, request_id);
    execute_insert<i_t, f_t, REQUEST>(solution, inserting_route, request_locations, &request);
  }
}

// each block checks a route
template <typename i_t, typename f_t, request_t REQUEST>
__global__ void find_best_squeeze_pos(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                                      const request_info_t<i_t, REQUEST>* request,
                                      cand_t* global_best,
                                      bool include_objective,
                                      infeasible_cost_t weights,
                                      i_t route_id = -1)
{
  extern __shared__ i_t shbuf[];
  cand_t* cand = (cand_t*)shbuf;
  i_t* shmem   = (i_t*)&cand[1];

  if (route_id == -1) { route_id = blockIdx.x; }
  find_squeeze_pos<i_t, f_t, REQUEST>(solution,
                                      request,
                                      cand,
                                      shmem,
                                      include_objective,
                                      weights,
                                      std::numeric_limits<f_t>::max(),
                                      route_id);
  if (threadIdx.x == 0) {
    // add the route id to which
    acquire_lock(solution.lock);
    if (cand->cost_counter.cost < global_best->cost_counter.cost) { *global_best = *cand; }
    release_lock(solution.lock);
  }
}

template <typename i_t, typename f_t, request_t REQUEST, bool squeeze_mode>
__global__ void find_all_squeeze_pos(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  typename ejection_pool_t<request_info_t<i_t, REQUEST>>::view_t EP,
  raft::device_span<cand_t> best_per_request,
  raft::device_span<cand_t> best_per_route,
  bool include_objective,
  infeasible_cost_t weights,
  double excess_limit,
  i_t n_insertions,
  i_t* inserted_requests)
{
  extern __shared__ i_t shbuf[];
  cand_t* cand = (cand_t*)shbuf;
  i_t* shmem   = (i_t*)&cand[1];

  i_t ep_idx   = blockIdx.x % n_insertions;
  i_t route_id = blockIdx.x / n_insertions;
  auto offset  = EP.size() - n_insertions;
  cuopt_assert(offset >= 0, "Offset should be positive");
  auto request = EP.stack_[offset + ep_idx];
  if (inserted_requests[request.info.node()]) { return; }
  find_squeeze_pos<i_t, f_t, REQUEST>(
    solution, &request, cand, shmem, include_objective, weights, excess_limit, route_id);
  if (threadIdx.x == 0) {
    // add the route id to which
    if constexpr (squeeze_mode) {
      acquire_lock(&(solution.lock_per_order[ep_idx]));
      if (cand->cost_counter.cost < best_per_request[ep_idx].cost_counter.cost) {
        best_per_request[ep_idx] = *cand;
      }
      release_lock(&(solution.lock_per_order[ep_idx]));
    } else {
      acquire_lock(&(solution.lock_per_route[route_id]));
      if (cand->cost_counter.cost < best_per_route[route_id].cost_counter.cost) {
        best_per_route[route_id] = *cand;
      }
      release_lock(&(solution.lock_per_route[route_id]));
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void extract_best_per_route(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                                       raft::device_span<cand_t> best_per_request,
                                       raft::device_span<cand_t> best_per_route)
{
  auto best_cand = best_per_request[blockIdx.x];
  i_t insertion_1, insertion_2, route_id, pickup_id;
  double cost_delta;
  move_candidates_t<i_t, f_t>::get_candidate(
    best_cand, insertion_1, insertion_2, route_id, pickup_id, cost_delta);

  // Update global best per route
  if (threadIdx.x == 0) {
    if (best_cand.cost_counter.cost < best_per_route[route_id].cost_counter.cost) {
      acquire_lock(&(solution.lock_per_route[route_id]));
      if (best_cand.cost_counter.cost < best_per_route[route_id].cost_counter.cost) {
        best_per_route[route_id] = best_cand;
      }
      release_lock(&(solution.lock_per_route[route_id]));
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void execute_move(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                             const request_info_t<i_t, REQUEST>* request,
                             cand_t* global_best)
{
  cuopt_assert(blockDim.x == 1 && gridDim.x == 1, "This should be a single threaded kernel");
  i_t insertion_1, insertion_2, route_id, pickup_id;
  double cost_delta;
  move_candidates_t<i_t, f_t>::get_candidate(
    *global_best, insertion_1, insertion_2, route_id, pickup_id, cost_delta);
  auto orginal_route = solution.routes[route_id];
  request_id_t<REQUEST> request_locations;
  if constexpr (REQUEST == request_t::PDP)
    request_locations = request_id_t<REQUEST>(insertion_1, insertion_2);
  else {
    request_locations = request_id_t<REQUEST>(insertion_1);
  }
  execute_insert<i_t, f_t, REQUEST>(solution, orginal_route, request_locations, request);
  cuopt_assert(!orginal_route.dimensions_info().has_dimension(dim_t::TIME) ||
                 abs(orginal_route.template get_dim<dim_t::TIME>()
                       .excess_forward[orginal_route.get_num_nodes()] -
                     orginal_route.template get_dim<dim_t::TIME>().excess_backward[0]) < 0.01,
               "Backward forward mismatch!");
}

template <typename i_t, typename f_t, request_t REQUEST, bool squeeze_mode>
__device__ bool reject_move(cand_t const& cand,
                            request_id_t<REQUEST> const& request_id,
                            i_t route_id,
                            raft::device_span<cand_t> best_per_request,
                            raft::device_span<cand_t> best_per_route,
                            i_t* inserted_requests)
{
  if (cand.cost_counter.cost == std::numeric_limits<double>::max()) { return true; }
  if constexpr (!squeeze_mode) {
    __shared__ i_t sh_process_request;
    if (threadIdx.x == 0) { sh_process_request = false; }
    __syncthreads();

    if (threadIdx.x == 0) {
      if (atomicCAS(&inserted_requests[request_id.id()], 0, 1) == 0) { sh_process_request = true; }
    }
    __syncthreads();

    return !sh_process_request;
  }
  return false;
}

template <typename i_t, typename f_t, request_t REQUEST, bool squeeze_mode>
__global__ void execute_all_move(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                                 raft::device_span<cand_t> best_per_request,
                                 raft::device_span<cand_t> best_per_route,
                                 i_t* inserted_requests,
                                 i_t* number_of_inserted)
{
  extern __shared__ i_t shmem[];
  cand_t cand = best_per_route[blockIdx.x];

  i_t insertion_1, insertion_2, route_id;
  request_id_t<REQUEST> request_id;
  double cost_delta;
  move_candidates_t<i_t, f_t>::get_candidate(
    cand, insertion_1, insertion_2, route_id, request_id.id(), cost_delta);

  if (reject_move<i_t, f_t, REQUEST, squeeze_mode>(
        cand, request_id, route_id, best_per_request, best_per_route, inserted_requests)) {
    return;
  }

  auto orginal_route = solution.routes[route_id];

  auto sh_route = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    shmem, orginal_route, orginal_route.get_num_nodes() + request_info_t<i_t, REQUEST>::size());
  __syncthreads();
  sh_route.copy_from(orginal_route);
  __syncthreads();

  if (threadIdx.x == 0) {
    if constexpr (REQUEST == request_t::PDP) {
      request_id.delivery = solution.problem.order_info.pair_indices[request_id.id()];
    }

    request_id_t<REQUEST> request_locations;
    if constexpr (REQUEST == request_t::PDP)
      request_locations = request_id_t<REQUEST>(insertion_1, insertion_2);
    else {
      request_locations = request_id_t<REQUEST>(insertion_1);
    }

    auto request = create_request<i_t, f_t, REQUEST>(solution.problem, request_id);
    execute_insert<i_t, f_t, REQUEST>(solution, sh_route, request_locations, &request);
    inserted_requests[request_id.id()] = 1;
    atomicAdd(number_of_inserted, 1);
    cuopt_assert(!orginal_route.dimensions_info().has_dimension(dim_t::TIME) ||
                   abs(orginal_route.template get_dim<dim_t::TIME>()
                         .excess_forward[orginal_route.get_num_nodes()] -
                       orginal_route.template get_dim<dim_t::TIME>().excess_backward[0]) < 0.01,
                 "Backward forward mismatch!");
  }
  __syncthreads();
  // Copy back to global
  solution.routes[sh_route.get_id()].copy_from(sh_route);
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void increase_multiple_p_scores(
  typename ejection_pool_t<request_info_t<i_t, REQUEST>>::view_t EP,
  i_t* p_scores,
  i_t* inserted_requests,
  i_t n_insertions)
{
  auto offset = EP.size() - n_insertions;
  for (i_t i = threadIdx.x; i < n_insertions; i += blockDim.x) {
    auto req = EP.at(offset + i);
    // Potential use of dynamic parallelism here
    if (!inserted_requests[req.info.node()]) { p_scores[req.info.node()] += 1; }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void eject_inserted_requests(
  typename ejection_pool_t<request_info_t<i_t, REQUEST>>::view_t EP,
  i_t* inserted_requests,
  i_t n_insertions)
{
  if (threadIdx.x == 0) {
    auto starting_size = EP.size();
    for (int i = 0; i < n_insertions; ++i) {
      auto req = EP.at(starting_size - 1 - i);
      if (inserted_requests[req.info.node()]) { EP.pop(req); }
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void squeeze_breaks_kernel(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                                      const bool include_objective,
                                      infeasible_cost_t weights)
{
  extern __shared__ i_t shmem[];

  i_t route_id = blockIdx.x;

  auto global_route      = solution.routes[route_id];
  const int n_break_dims = solution.problem.get_break_dimensions(global_route.get_vehicle_id());

  typename route_t<i_t, f_t, REQUEST>::view_t sh_route;
  sh_route = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    shmem, global_route, global_route.get_num_nodes() + n_break_dims);
  sh_route.copy_from(global_route);
  __syncthreads();

  auto break_dim_counters =
    raft::device_span<i_t>(reinterpret_cast<i_t*>(sh_route.shared_end_address()), n_break_dims);
  for (i_t tid = threadIdx.x; tid < n_break_dims; tid += blockDim.x) {
    break_dim_counters[tid] = 0;
  }
  __syncthreads();

  for (i_t tid = threadIdx.x; tid < sh_route.get_num_nodes(); tid += blockDim.x) {
    auto node = sh_route.get_node(tid);
    if (node.node_info().is_break()) {
      cuopt_assert(node.node_info().break_dim() >= 0 && node.node_info().break_dim() < n_break_dims,
                   "Break dimension out of bounds");
      break_dim_counters[node.node_info().break_dim()] = 1;
    }
  }
  __syncthreads();

  for (int break_dim_idx = 0; break_dim_idx < n_break_dims; ++break_dim_idx) {
    if (break_dim_counters[break_dim_idx] == 1) { continue; }
    const auto old_objective_cost    = sh_route.get_objective_cost();
    const auto old_infeasbility_cost = sh_route.get_infeasibility_cost();
    auto break_nodes =
      solution.problem.special_nodes.subset(sh_route.get_vehicle_id(), break_dim_idx);

    i_t route_size                = sh_route.get_num_nodes();
    i_t num_break_nodes           = break_nodes.size();
    double thread_best_cost       = std::numeric_limits<double>::max();
    i_t thread_best_idx           = -1;
    i_t thread_best_break_node_id = -1;
    for (int index = threadIdx.x; index < route_size * num_break_nodes; index += blockDim.x) {
      i_t break_node_id = index / route_size;
      i_t insertion_idx = index % route_size;

      auto break_node = create_break_node<i_t, f_t, REQUEST>(
        break_nodes, break_node_id, solution.problem.dimensions_info);

      auto curr_node = sh_route.get_node(insertion_idx);
      auto next_node = sh_route.get_node(insertion_idx + 1);

      curr_node.calculate_forward_all(break_node, sh_route.vehicle_info());

      double cost_difference = break_node.calculate_forward_all_and_delta(next_node,
                                                                          sh_route.vehicle_info(),
                                                                          include_objective,
                                                                          weights,
                                                                          old_objective_cost,
                                                                          old_infeasbility_cost);
      if (cost_difference < thread_best_cost) {
        thread_best_cost          = cost_difference;
        thread_best_idx           = insertion_idx;
        thread_best_break_node_id = break_node_id;
      }
    }

    i_t t_id = threadIdx.x;
    __shared__ i_t reduction_idx;
    __shared__ double reduction_buf[2 * raft::WarpSize];
    block_reduce_ranked(thread_best_cost, t_id, reduction_buf, &reduction_idx);

    if (threadIdx.x == reduction_idx) {
      auto break_node = create_break_node<i_t, f_t, REQUEST>(
        break_nodes, thread_best_break_node_id, solution.problem.dimensions_info);
      // do not update the intra indices yet
      sh_route.insert_node(thread_best_idx, break_node, solution.route_node_map, false);
      route_t<i_t, f_t, REQUEST>::view_t::compute_forward(sh_route);
      route_t<i_t, f_t, REQUEST>::view_t::compute_backward(sh_route);
      sh_route.compute_cost();
    }

    __syncthreads();
  }

  if (threadIdx.x == 0) { solution.routes_to_copy[sh_route.get_id()] = 1; }

  sh_route.compute_intra_indices(solution.route_node_map);
  __syncthreads();
  global_route.copy_from(sh_route);
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
