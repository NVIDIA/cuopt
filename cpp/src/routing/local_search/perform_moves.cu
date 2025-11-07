/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/cuda_helpers.cuh>
#include "compute_ejections.cuh"
#include "local_search.cuh"
#include "permutation_helper.cuh"

namespace cuopt {
namespace routing {
namespace detail {

// TODO get this by the function param
constexpr int n_insertions_per_move = 1;

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
DI auto create_request_node(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                            request_id_t<REQUEST> const& request_id)
{
  const auto& order_info = solution.problem.order_info;
  const auto node_info   = NodeInfo<i_t>(
    request_id.id(), order_info.get_order_location(request_id.id()), node_type_t::DELIVERY);
  const auto node = create_node<i_t, f_t, REQUEST>(solution.problem, node_info, node_info);
  return request_node_t<i_t, f_t, REQUEST>(node);
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
DI auto create_request_node(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                            request_id_t<REQUEST> const& request_id)
{
  const auto& order_info      = solution.problem.order_info;
  const auto pickup_node_info = NodeInfo<i_t>(
    request_id.pickup, order_info.get_order_location(request_id.pickup), node_type_t::PICKUP);
  const auto delivery_node_info = NodeInfo<i_t>(
    request_id.delivery, order_info.get_order_location(request_id.delivery), node_type_t::DELIVERY);
  const auto pickup_node =
    create_node<i_t, f_t, REQUEST>(solution.problem, pickup_node_info, delivery_node_info);
  const auto delivery_node =
    create_node<i_t, f_t, REQUEST>(solution.problem, delivery_node_info, pickup_node_info);
  return request_node_t<i_t, f_t, REQUEST>(pickup_node, delivery_node);
}

/**
 * @brief Inserts cycles and paths found in cycle finder graph.
 * We use temp routes to do the insertions and then copy them to the original routes.
 *
 * @param solution Solution representation
 * @param path The path of moves that needs to be inserted
 */
template <typename i_t, typename f_t, request_t REQUEST>
__global__ void insert_graph_nodes_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  typename move_candidates_t<i_t, f_t>::view_t move_candidates)
{
  __shared__ extern i_t shmem[];
  const auto& order_info = solution.problem.order_info;
  auto& route_node_map   = solution.route_node_map;

  const auto path = move_candidates.move_path;
  // each block does a single insertion
  cand_t insertion_item = path.path[blockIdx.x];
  request_id_t<REQUEST> ejected_request_id;
  request_id_t<REQUEST> request_id;
  request_id_t<REQUEST> request_location;
  move_path_t<i_t, f_t>::get_cycle_edge(
    insertion_item, request_id.id(), ejected_request_id.id(), request_location);

  cuopt_assert(request_id.id() < solution.get_num_orders(),
               "Pickup id should be smaller than n_orders");
  typename route_t<i_t, f_t, REQUEST>::view_t route;
  typename route_t<i_t, f_t, REQUEST>::view_t global_route;
  bool is_pseudo_node = ejected_request_id.id() >= solution.get_num_orders();
  // pseudo node
  if (is_pseudo_node) {
    cuopt_assert(ejected_request_id.id() != solution.get_num_orders() + solution.n_routes,
                 "Special node cannot participate in a move");
    auto pseudo_route_id = ejected_request_id.id() - solution.get_num_orders();
    global_route         = solution.routes[pseudo_route_id];
    route                = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
      shmem,
      global_route,
      global_route.get_num_nodes() + n_insertions_per_move * request_info_t<i_t, REQUEST>::size());
    __syncthreads();
    route.copy_from(global_route);
  }
  // ejected node
  else {
    // we recompute the route_id_per_node after the kernel so there are no race condiitons here
    auto route_id = route_node_map.get_route_id(ejected_request_id.id());
    global_route  = solution.routes[route_id];
    route         = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
      shmem, global_route, global_route.get_num_nodes());
    __syncthreads();
    if constexpr (REQUEST == request_t::PDP) {
      ejected_request_id.delivery = order_info.pair_indices[ejected_request_id.id()];
    }
    // get the temp route that has 1 ejected request in it
    compute_temp_route<i_t, f_t, REQUEST>(
      route, global_route, global_route.get_num_nodes(), solution, ejected_request_id);
  }
  // read other route id before we change the route_id_per_node
  const i_t other_route_id = route_node_map.get_route_id(request_id.id());
  if constexpr (REQUEST == request_t::PDP) {
    request_id.delivery = order_info.pair_indices[request_id.id()];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    auto request_node = create_request_node<i_t, f_t, REQUEST>(solution, request_id);
    route.insert_request(request_location, request_node, solution.route_node_map, false);
    route_t<i_t, f_t, REQUEST>::view_t::compute_forward(route);
    route_t<i_t, f_t, REQUEST>::view_t::compute_backward(route);
    route.compute_cost();
    solution.routes_to_copy[route.get_id()]   = 1;
    solution.routes_to_search[route.get_id()] = 1;
  }
  __syncthreads();
  cuopt_assert(global_route.get_id() == route.get_id(), "Route ids don't match in perform");
  cuopt_assert(
    !is_pseudo_node || (global_route.get_num_nodes() + request_info_t<i_t, REQUEST>::size() ==
                        route.get_num_nodes()),
    "Node count is not properly increased");
  global_route.copy_from(route);
  __syncthreads();
  // check if the cycle closes on the route of the ejected nodes
  // if the cycle does not close, copy the ejected(from which the interted requests are ejected)
  // temp route to actual route

  if (!path.loop_closed[other_route_id]) {
    auto& other_route        = solution.routes[other_route_id];
    auto sh_other_temp_route = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
      shmem, other_route, other_route.get_num_nodes());
    __syncthreads();
    compute_temp_route<i_t, f_t, REQUEST>(
      sh_other_temp_route, other_route, other_route.get_num_nodes(), solution, request_id);
    __syncthreads();
    cuopt_assert((sh_other_temp_route.get_num_nodes() + request_info_t<i_t, REQUEST>::size()) ==
                   other_route.get_num_nodes(),
                 "Node count is not properly decreased");
    cuopt_assert(sh_other_temp_route.get_id() == other_route.get_id(),
                 "Route ids don't match in perform");
    other_route.copy_from(sh_other_temp_route);
    if (threadIdx.x == 0) {
      solution.routes_to_copy[other_route_id]   = 1;
      solution.routes_to_search[other_route_id] = 1;
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__device__ void add_edge_to_move_path(typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
                                      typename solution_t<i_t, f_t, REQUEST>::view_t& sol,
                                      i_t src_id,
                                      i_t dst_id)
{
  auto move = move_candidates.cand_matrix.get_candidate(src_id, dst_id);
  i_t insertion_1, insertion_2, insertion_3, insertion_4;
  double cost_delta;
  move_candidates_t<i_t, f_t>::get_candidate(
    move, insertion_1, insertion_2, insertion_3, insertion_4, cost_delta);
  cuopt_assert(dst_id != sol.get_num_orders() + sol.n_routes,
               "Special node cannot participate in a move");
  auto cand     = move_path_t<i_t, f_t>::make_cycle_edge(src_id,  // inserting node
                                                     dst_id,  // ejecting node
                                                     insertion_1,
                                                     insertion_2);
  i_t write_pos = atomicAdd(move_candidates.move_path.n_insertions, 1);
  move_candidates.move_path.path[write_pos] = cand;
}

// each thread block(warp) handles a cycle and fills the move path accordingly
template <typename i_t, typename f_t, request_t REQUEST>
__global__ void populate_move_path_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t sol,
  typename move_candidates_t<i_t, f_t>::view_t move_candidates)
{
  if (threadIdx.x == 0) {
    auto& route_node_map = sol.route_node_map;
    auto cycle_id        = blockIdx.x;
    auto offset_beg      = move_candidates.cycles.offsets[cycle_id];
    auto offset_end      = move_candidates.cycles.offsets[cycle_id + 1];
    auto cycle_size      = offset_end - offset_beg;
    auto curr_cycle      = move_candidates.cycles.paths.subspan(offset_beg, cycle_size);
    i_t src_id           = curr_cycle[cycle_size - 1];
    i_t dst_id;
    for (i_t i = cycle_size - 2; i >= -1; --i) {
      dst_id = curr_cycle[(i + cycle_size) % cycle_size];
      // if the edge is from a special node to any node, break the edge and mark the dst node as
      // loop not closed
      if (src_id == sol.get_num_orders() + sol.n_routes) {
        // this means that we need to write the ejected route as it is later
        i_t route_id                                       = route_node_map.get_route_id(dst_id);
        move_candidates.move_path.loop_closed[route_id]    = 0;
        move_candidates.move_path.changed_routes[route_id] = 1;
      }
      // all the other edges
      else if (src_id < sol.get_num_orders()) {
        add_edge_to_move_path<i_t, f_t, REQUEST>(move_candidates, sol, src_id, dst_id);
        move_candidates.move_path.changed_routes[route_node_map.get_route_id(src_id)] = 1;
        i_t dest_route_id = dst_id < sol.get_num_orders() ? route_node_map.get_route_id(dst_id)
                                                          : dst_id - sol.get_num_orders();
        move_candidates.move_path.changed_routes[dest_route_id] = 1;
      }
      // else = the edge is from pseudo node to special node, then break the edge and do nothing
      src_id = dst_id;
    }
  }
}

// each thread block(warp) checks a route, if it is not changed adds the intra move
template <typename i_t, typename f_t, request_t REQUEST>
__global__ void populate_intra_candidates(
  typename solution_t<i_t, f_t, REQUEST>::view_t sol,
  typename move_candidates_t<i_t, f_t>::view_t move_candidates)
{
  // visit the diagonal of the candidate matrix which contains the intra candidates
  for (i_t route_id = threadIdx.x; route_id < sol.n_routes; route_id += blockDim.x) {
    if (move_candidates.move_path.changed_routes[route_id]) continue;
    auto intra_move = move_candidates.cand_matrix.get_intra_candidate(route_id);
    if (intra_move.cost_counter.cost > -EPSILON) { continue; }
    i_t insertion_1, insertion_2, insertion_3, insertion_4;
    double cost_delta;
    i_t src_id = intra_move.pair_1;
    auto cand  = move_candidates.cand_matrix.get_candidate(src_id, src_id);
    move_candidates_t<i_t, f_t>::get_candidate(
      cand, insertion_1, insertion_2, insertion_3, insertion_4, cost_delta);
    // if it is best cycle and the cost is positive. or there is no move there
    i_t write_pos = atomicAdd_block(move_candidates.move_path.n_insertions, 1);
    auto move     = move_path_t<i_t, f_t>::make_cycle_edge(src_id,  // inserting node
                                                       src_id,  // ejecting node
                                                       insertion_1,
                                                       insertion_2);
    move_candidates.move_path.path[write_pos] = move;
  }
}

// this kernel populates route pairs
template <typename i_t, typename f_t, request_t REQUEST>
__global__ void populate_cross_list_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t sol,
  typename move_candidates_t<i_t, f_t>::view_t move_candidates)
{
  extern __shared__ cross_cand_t best_values_per_route[];
  auto scross_cands    = move_candidates.scross_move_candidates;
  i_t* locks_per_route = (i_t*)&best_values_per_route[sol.n_routes];
  init_block_shmem(best_values_per_route,
                   cross_cand_t{0, 0, std::numeric_limits<double>::max(), 0, 0},
                   sol.n_routes);
  init_block_shmem(locks_per_route, 0, sol.n_routes);
  __syncthreads();

  auto& route_node_map = sol.route_node_map;
  i_t first_node       = blockIdx.x + 1;
  i_t first_route      = route_node_map.get_route_id(first_node);
  // we use incomplete LS in try_squeeze
  if (first_route == -1) return;
  const i_t special_node_id = sol.n_routes + sol.get_num_orders();
  // we can have all combinations at cross, delivery ejection and delivery insertion too
  const i_t row_size = (sol.get_num_orders() - 1) + sol.n_routes;
  for (i_t i = threadIdx.x; i < row_size; i += blockDim.x) {
    i_t second_node = i + 1;
    i_t second_route;
    const auto first_cand = move_candidates.cand_matrix.get_candidate(first_node, second_node);
    cand_t second_cand;
    if (first_cand.cost_counter.cost == std::numeric_limits<double>::max()) continue;
    // for cross
    if (second_node < sol.get_num_orders()) {
      second_route = route_node_map.get_route_id(second_node);
      // only consider the upper triangular matrix, as we reach the lower with the second half of
      // cross
      if (first_route >= second_route) continue;
      cuopt_assert(second_route != -1, "Node should be assigned");
      second_cand = move_candidates.cand_matrix.get_candidate(second_node, first_node);
      if (second_cand.cost_counter.cost == std::numeric_limits<double>::max()) continue;
    }
    // for relocate
    else {
      second_route = second_node - sol.get_num_orders();
      if (second_route == first_route) continue;
      // get the cost for the ejection of first node, for the candidate
      second_cand = move_candidates.cand_matrix.get_candidate(special_node_id, first_node);
      if (second_cand.cost_counter.cost == std::numeric_limits<double>::max()) continue;
    }

    const auto cross_cand = move_candidates_t<i_t, f_t>::make_cross_candidate(
      first_cand, second_cand, first_node, second_node);

    if (cross_cand.cost_counter.cost < best_values_per_route[second_route].cost_counter.cost) {
      acquire_lock_block(&locks_per_route[second_route]);
      if (cross_cand.cost_counter.cost < best_values_per_route[second_route].cost_counter.cost) {
        best_values_per_route[second_route] = cross_cand;
      }
      release_lock_block(&locks_per_route[second_route]);
    }
  }
  __syncthreads();
  for (i_t second_route = threadIdx.x; second_route < sol.n_routes; second_route += blockDim.x) {
    const i_t route_pair_idx  = first_route * sol.n_routes + second_route;
    const auto best_per_route = best_values_per_route[second_route];
    // acquire the critical section for this route pair
    acquire_lock(&scross_cands.route_pair_locks[route_pair_idx]);
    scross_cands.insert_best_scross_candidate(route_pair_idx, best_per_route);
    release_lock(&scross_cands.route_pair_locks[route_pair_idx]);
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void populate_cross_moves_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t sol,
  typename move_candidates_t<i_t, f_t>::view_t move_candidates)
{
  auto scross_cands         = move_candidates.scross_move_candidates;
  const i_t matrix_size     = sol.n_routes * sol.n_routes;
  const i_t special_node_id = sol.n_routes + sol.get_num_orders();
  extern __shared__ i_t changed_routes[];
  i_t* route_pair_indices = &changed_routes[sol.n_routes];
  init_block_shmem(changed_routes, 0, sol.n_routes);
  block_sequence(route_pair_indices, sol.n_routes * sol.n_routes);
  __syncthreads();
  if (threadIdx.x == 0) {
    raft::random::PCGenerator thread_rng(
      clock64(), uint64_t((threadIdx.x + blockIdx.x * blockDim.x)), 0);
    random_shuffle(route_pair_indices, sol.n_routes * sol.n_routes, thread_rng);
    i_t n_changed_routes = 0;
    for (i_t i = 0; i < matrix_size; ++i) {
      i_t curr_idx    = route_pair_indices[i];
      const auto cand = scross_cands.scross_best_cand_list[curr_idx];
      if (n_changed_routes >= sol.n_routes - 1) break;
      if (cand.cost_counter.cost > -EPSILON) continue;
      i_t first_node                  = cand.id_1;
      i_t second_node                 = cand.id_2;
      bool is_ejected                 = second_node < sol.get_num_orders();
      [[maybe_unused]] i_t matrix_idx = is_ejected ? second_node : special_node_id;
      cuopt_assert(
        cand.cost_counter.cost == (move_candidates.cand_matrix.get_cost(first_node, second_node) +
                                   move_candidates.cand_matrix.get_cost(matrix_idx, first_node)),
        "Cost mismatch!");
      if constexpr (REQUEST == request_t::PDP) {
        // we only have cross on ejected node
        if (is_ejected) {
          if (!sol.problem.order_info.is_pickup_index[first_node]) {
            first_node = sol.problem.order_info.pair_indices[first_node];
          }
          if (!sol.problem.order_info.is_pickup_index[second_node]) {
            second_node = sol.problem.order_info.pair_indices[second_node];
          }
        }
      }

      i_t first_route  = sol.route_node_map.get_route_id(first_node);
      i_t second_route = is_ejected ? sol.route_node_map.get_route_id(second_node)
                                    : second_node - sol.get_num_orders();
      cuopt_assert(first_route != -1 && second_route != -1, "Unrouted node in candidates!");
      if (changed_routes[first_route] || changed_routes[second_route]) continue;
      i_t insertion_1, insertion_2, insertion_3, insertion_4;
      double cost_delta;
      move_candidates_t<i_t, f_t>::get_candidate(
        cand, insertion_1, insertion_2, insertion_3, insertion_4, cost_delta);
      cuopt_func_call(*move_candidates.debug_delta -= cost_delta);
      i_t n_insertions = *move_candidates.move_path.n_insertions;
      auto move_1      = move_path_t<i_t, f_t>::make_cycle_edge(first_node,   // inserting node
                                                           second_node,  // ejecting node
                                                           insertion_1,
                                                           insertion_2);
      move_candidates.move_path.path[n_insertions] = move_1;
      *move_candidates.move_path.n_insertions      = n_insertions + 1;
      // for cross
      if (is_ejected) {
        auto move_2 = move_path_t<i_t, f_t>::make_cycle_edge(second_node,  // inserting node
                                                             first_node,   // ejecting node
                                                             insertion_3,
                                                             insertion_4);
        move_candidates.move_path.path[n_insertions + 1] = move_2;
        *move_candidates.move_path.n_insertions          = n_insertions + 2;
      }
      // for relcoate
      else {
        // mark loop as not closed, so that the ejected route will be saved to global
        move_candidates.move_path.loop_closed[first_route] = 0;
      }
      changed_routes[first_route]  = 1;
      changed_routes[second_route] = 1;
      n_changed_routes += 2;
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::reset_cross_vectors(solution_t<i_t, f_t, REQUEST>& solution)
{
  auto& scross_cands = move_candidates.scross_move_candidates;
  if (scross_cands.route_pair_locks.size() < size_t(solution.n_routes * solution.n_routes)) {
    scross_cands.route_pair_locks.resize(solution.n_routes * solution.n_routes,
                                         solution.sol_handle->get_stream());
    scross_cands.scross_best_cand_list.resize(solution.n_routes * solution.n_routes,
                                              solution.sol_handle->get_stream());
  }
  scross_cands.reset(solution.sol_handle);
}

template <typename i_t, typename f_t, request_t REQUEST>
bool local_search_t<i_t, f_t, REQUEST>::populate_cross_moves(
  solution_t<i_t, f_t, REQUEST>& solution, move_candidates_t<i_t, f_t>& move_candidates)
{
  raft::common::nvtx::range fun_scope("populate_cross_moves");
  reset_cross_vectors(solution);
  const i_t TPB  = 256;
  size_t sh_size = solution.n_routes * (sizeof(i_t) + sizeof(cross_cand_t));
  if (!set_shmem_of_kernel(populate_cross_list_kernel<i_t, f_t, REQUEST>, sh_size)) {
    return false;
  }
  populate_cross_list_kernel<i_t, f_t, REQUEST>
    <<<solution.get_num_orders() - 1, TPB, sh_size, solution.sol_handle->get_stream()>>>(
      solution.view(), move_candidates.view());

  sh_size = sizeof(i_t) * (solution.n_routes + 1) * solution.n_routes;

  if (!set_shmem_of_kernel(populate_cross_moves_kernel<i_t, f_t, REQUEST>, sh_size)) {
    return false;
  }
  populate_cross_moves_kernel<i_t, f_t, REQUEST>
    <<<1, TPB, sh_size, solution.sol_handle->get_stream()>>>(solution.view(),
                                                             move_candidates.view());
  solution.sol_handle->sync_stream();
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::populate_move_path(
  solution_t<i_t, f_t, REQUEST>& solution, move_candidates_t<i_t, f_t>& move_candidates)
{
  raft::common::nvtx::range fun_scope("populate_move_path");
  auto n_cycles = move_candidates.cycles.n_cycles_.value(solution.sol_handle->get_stream());
  if (n_cycles) {
    populate_move_path_kernel<i_t, f_t, REQUEST>
      <<<n_cycles, 32, 0, solution.sol_handle->get_stream()>>>(solution.view(),
                                                               move_candidates.view());
  }
  populate_intra_candidates<i_t, f_t, REQUEST>
    <<<1, 128, 0, solution.sol_handle->get_stream()>>>(solution.view(), move_candidates.view());
}

template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::perform_moves(solution_t<i_t, f_t, REQUEST>& solution,
                                                      move_candidates_t<i_t, f_t>& move_candidates)
{
  raft::common::nvtx::range fun_scope("perform_moves");
  solution.global_runtime_checks(false, false, "perform_moves_start");
  auto stream        = solution.sol_handle->get_stream();
  constexpr i_t TPB  = 32;
  const i_t n_blocks = move_candidates.move_path.n_insertions.value(stream);
  size_t shared_size = solution.check_routes_can_insert_and_get_sh_size(
    n_insertions_per_move * request_info_t<i_t, REQUEST>::size());
  bool is_set = set_shmem_of_kernel(insert_graph_nodes_kernel<i_t, f_t, REQUEST>, shared_size);
  cuopt_assert(is_set, "Not enough shared memory on device for performing the local search move!");
  cuopt_expects(is_set, error_type_t::OutOfMemoryError, "Not enough shared memory on device");
  insert_graph_nodes_kernel<i_t, f_t, REQUEST>
    <<<n_blocks, TPB, shared_size, stream>>>(solution.view(), move_candidates.view());
  solution.compute_route_id_per_node();
  solution.compute_cost();
  solution.global_runtime_checks(false, false, "perform_moves_end");
}

template void local_search_t<int, float, request_t::PDP>::perform_moves(
  solution_t<int, float, request_t::PDP>& solution, move_candidates_t<int, float>& move_candidates);
template void local_search_t<int, float, request_t::PDP>::populate_move_path(
  solution_t<int, float, request_t::PDP>& solution, move_candidates_t<int, float>& move_candidates);
template bool local_search_t<int, float, request_t::PDP>::populate_cross_moves(
  solution_t<int, float, request_t::PDP>& solution, move_candidates_t<int, float>& move_candidates);
template bool local_search_t<int, float, request_t::VRP>::populate_cross_moves(
  solution_t<int, float, request_t::VRP>& solution, move_candidates_t<int, float>& move_candidates);

template void local_search_t<int, float, request_t::VRP>::perform_moves(
  solution_t<int, float, request_t::VRP>& solution, move_candidates_t<int, float>& move_candidates);
template void local_search_t<int, float, request_t::VRP>::populate_move_path(
  solution_t<int, float, request_t::VRP>& solution, move_candidates_t<int, float>& move_candidates);
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
