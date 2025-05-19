/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <utilities/cuda_helpers.cuh>
#include "../solution/solution.cuh"
#include "../util_kernels/top_k.cuh"
#include "../utilities/cuopt_utils.cuh"
#include "local_search.cuh"
#include "vrp/fragment_kernels.cuh"

namespace cuopt {
namespace routing {
namespace detail {

auto constexpr items_per_thread = 2;
auto constexpr top_k_candidates = 64;
auto constexpr write_diagonal   = false;

template <typename i_t, typename f_t, request_t REQUEST>
DI thrust::pair<double, double> evaluate_two_opt_move(
  typename solution_t<i_t, f_t, REQUEST>::view_t& sol,
  typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
  const typename route_t<i_t, f_t, REQUEST>::view_t& route,
  i_t first,
  i_t second,
  typename dimensions_route_t<i_t, f_t, REQUEST>::view_t const& fragment,
  double excess_limit)
{
  auto frag_size                = second - first;
  auto frag_start               = route.get_num_nodes() - second;
  auto [delta, selection_delta] = evaluate_fragment<i_t, f_t, REQUEST>(
    sol, move_candidates, route, first, second + 1, frag_size, fragment, excess_limit, frag_start);

  if (delta == std::numeric_limits<double>::max()) {
    return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  }
  return {delta, selection_delta};
}

template <typename i_t, typename f_t, request_t REQUEST>
DI thrust::pair<double, double> evaluate_two_opt_cvrp_move(
  typename solution_t<i_t, f_t, REQUEST>::view_t& sol,
  typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
  const typename route_t<i_t, f_t, REQUEST>::view_t& route,
  typename dimensions_route_t<i_t, f_t, REQUEST>::view_t const& reverse_route,
  i_t first,
  i_t second)
{
  auto n_nodes         = route.get_num_nodes();
  double frag_backward = reverse_route.distance_dim.distance_forward[n_nodes - (first + 1)] -
                         reverse_route.distance_dim.distance_forward[n_nodes - second];
  double forward_sum = route.get_node(second + 1).distance_dim.distance_forward -
                       route.get_node(first).distance_dim.distance_forward;

  double first_second = get_arc_of_dimension<i_t, f_t, dim_t::DIST>(
    route.get_node(first).node_info(), route.get_node(second).node_info(), route.vehicle_info());

  double first_next_second_next =
    get_arc_of_dimension<i_t, f_t, dim_t::DIST>(route.get_node(first + 1).node_info(),
                                                route.get_node(second + 1).node_info(),
                                                route.vehicle_info());

  double delta = (first_second + frag_backward + first_next_second_next) - forward_sum;
  return {delta, delta};
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void find_two_opt_moves(typename solution_t<i_t, f_t, REQUEST>::view_t sol,
                                   typename move_candidates_t<i_t, f_t>::view_t move_candidates,
                                   raft::device_span<two_opt_cand_t<i_t>> best_candidates,
                                   raft::device_span<two_opt_cand_t<i_t>> sampled_nodes_data,
                                   raft::device_span<i_t> locks)
{
  extern __shared__ double shmem[];

  const auto node_info = move_candidates.nodes_to_search.sampled_nodes_to_search[blockIdx.x];
  cuopt_assert(
    node_info.node() < sol.get_num_orders() + sol.n_routes * after_depot_insertion_multiplier,
    "Invalid node id");
  // special node that represent after depot insertion is ignored
  if (node_info.node() >= sol.get_num_orders()) { return; }

  // Retrive associated node info

  const auto [route_id, intra_idx] =
    sol.route_node_map.get_route_id_and_intra_idx(node_info.node());

  if (route_id == -1)  // Handle unrouted node case for GES
    return;

  cuopt_assert(route_id >= 0, "Invalid route id");
  cuopt_assert(route_id < sol.n_routes, "Invalid route id");
  auto route = sol.routes[route_id];

  const auto route_length = route.get_num_nodes();
  cuopt_assert(route_length > 1, "Invalid route length");

  cuopt_assert(intra_idx > 0, "Invalid intra_idx");
  cuopt_assert(intra_idx < route_length, "Invalid intra_idx");

  const double excess_limit =
    route.get_weighted_excess(move_candidates.weights) * ls_excess_multiplier_route;

  auto sh_reverse_route = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    (int*)shmem, route, route.get_num_nodes());
  __syncthreads();
  for (auto i = threadIdx.x; i <= route.get_num_nodes(); i += blockDim.x) {
    auto node      = route.get_node(i);
    auto intra_idx = route.get_num_nodes() - i;
    sh_reverse_route.set_node(intra_idx, node);
  }
  __syncthreads();
  if (sol.problem.is_cvrp_intra()) {
    if (threadIdx.x == 0) { route_t<i_t, f_t, REQUEST>::view_t::compute_forward(sh_reverse_route); }
    __syncthreads();
  }

  two_opt_cand_t<i_t> two_opt_cand = is_two_opt_uinitialized_t<i_t>::init_data;
  double cost_delta, selection_delta;
  auto nodes = route.get_num_nodes() - intra_idx;
  auto first = intra_idx;

  for (auto tid = threadIdx.x; tid < nodes; tid += blockDim.x) {
    auto second = tid + intra_idx;
    // skip non valid indices
    if (second <= first + 1) { continue; }

    if (sol.problem.is_cvrp_intra()) {
      thrust::tie(cost_delta, selection_delta) = evaluate_two_opt_cvrp_move<i_t, f_t, REQUEST>(
        sol, move_candidates, route, sh_reverse_route.dimensions, first, second);
    } else {
      thrust::tie(cost_delta, selection_delta) = evaluate_two_opt_move<i_t, f_t, REQUEST>(
        sol, move_candidates, route, first, second, sh_reverse_route.dimensions, excess_limit);
    }

    if (cost_delta > -EPSILON) { continue; }

    if (selection_delta < two_opt_cand.selection_delta) {
      two_opt_cand.first           = first;
      two_opt_cand.second          = second;
      two_opt_cand.selection_delta = selection_delta;
    }
  }

  __shared__ int reduction_index;
  __shared__ double shbuf[warp_size * 2];

  int idx = threadIdx.x;
  // block_reduce_ranked changes two_opt_cand
  double saved_cost = two_opt_cand.selection_delta;
  block_reduce_ranked(saved_cost, idx, shbuf, &reduction_index);
  if (!sol.problem.is_cvrp_intra()) {
    if (shbuf[0] != std::numeric_limits<double>::max() && shbuf[0] < -EPSILON &&
        reduction_index == threadIdx.x) {
      if (two_opt_cand.selection_delta < best_candidates[route_id].selection_delta) {
        acquire_lock(&locks[route_id]);
        if (two_opt_cand.selection_delta < best_candidates[route_id].selection_delta) {
          best_candidates[route_id] = two_opt_cand;
        }
        release_lock(&locks[route_id]);
      }
    }
  } else {
    if (threadIdx.x == reduction_index) {
      sampled_nodes_data[route_id * sol.get_num_orders() + node_info.node()] = two_opt_cand;
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
DI void mark_impacted_nodes(typename route_t<i_t, f_t, REQUEST>::view_t const& route,
                            typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
                            raft::device_span<i_t> moved_regions,
                            i_t start_idx,
                            i_t inserted_frag_size,
                            i_t max_active,
                            bool is_cvrp)
{
  // add two more neighbors as there might be more slack around them
  i_t start     = max(start_idx, 1);
  i_t end       = min(start_idx + inserted_frag_size + 1, route.get_num_nodes());
  auto route_id = blockIdx.x;
  for (i_t i = threadIdx.x + start; i < end; i += blockDim.x) {
    move_candidates.nodes_to_search.active_nodes_impacted[route.node_id(i)] = 1;
    if (is_cvrp) {
      cuopt_assert(!moved_regions[route_id * max_active + i], "Node was already moved");
      moved_regions[route_id * max_active + i] = 1;
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void execute_two_opt_moves(typename solution_t<i_t, f_t, REQUEST>::view_t sol,
                                      typename move_candidates_t<i_t, f_t>::view_t move_candidates,
                                      raft::device_span<two_opt_cand_t<i_t>> best_candidates,
                                      raft::device_span<i_t> moved_regions)
{
  extern __shared__ double shmem[];
  auto route_id   = blockIdx.x;
  auto route      = sol.routes[route_id];
  auto max_active = sol.get_max_active_nodes_for_all_routes();
  auto cand       = best_candidates[route_id];
  auto frag_size  = cand.second - (cand.first);

  if (cand.selection_delta == std::numeric_limits<double>::max()) { return; }
  cuopt_func_call(
    if (threadIdx.x == 0) { atomicAdd(move_candidates.debug_delta, cand.selection_delta); });

  auto s_route = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    (i_t*)shmem, route, route.get_num_nodes());
  __syncthreads();
  s_route.copy_from(route);
  __syncthreads();

  typename dimensions_route_t<i_t, f_t, REQUEST>::view_t fragment;
  i_t* dummy;
  // max_fragment_size-1, because the create shared route adds one more already
  thrust::tie(fragment, dummy) = dimensions_route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    reinterpret_cast<i_t*>(raft::alignTo(s_route.shared_end_address(), sizeof(double))),
    sol.problem.dimensions_info,
    max_active - 1);
  __syncthreads();
  fragment.parallel_copy_nodes_from(0, s_route, cand.first + 1, frag_size, true);
  __syncthreads();

  for (auto i = threadIdx.x; i < frag_size; i += blockDim.x) {
    auto node      = fragment.get_node(i);
    auto intra_idx = cand.first + 1 + i;
    s_route.set_node(intra_idx, node);
    sol.route_node_map.set_intra_route_idx(node.node_info(), intra_idx);
  }
  __syncthreads();

  mark_impacted_nodes<i_t, f_t, REQUEST>(s_route,
                                         move_candidates,
                                         moved_regions,
                                         cand.first,
                                         frag_size,
                                         max_active,
                                         sol.problem.is_cvrp_intra());
  route_t<i_t, f_t, REQUEST>::view_t::compute_forward_backward_cost(s_route);
  __syncthreads();
  if (threadIdx.x == 0) {
    s_route.compute_cost();
    sol.routes_to_copy[s_route.get_id()]   = 1;
    sol.routes_to_search[s_route.get_id()] = 1;
  }
  __syncthreads();

  route.copy_from(s_route);
}

template <typename i_t, typename f_t, request_t REQUEST, int BLOCK_SIZE>
__global__ void execute_recycle(typename solution_t<i_t, f_t, REQUEST>::view_t sol,
                                typename move_candidates_t<i_t, f_t>::view_t move_candidates,
                                raft::device_span<two_opt_cand_t<i_t>> sampled_nodes_data,
                                raft::device_span<i_t> moved_regions)
{
  extern __shared__ double shmem[];
  __shared__ i_t sh_overlaps;

  auto route_id = blockIdx.x;
  __shared__ two_opt_cand_t<i_t> sh_out_costs[64];
  __shared__ i_t sh_out_indices[64];

  auto out_costs              = raft::device_span<two_opt_cand_t<i_t>>(sh_out_costs, 64);
  auto out_indices            = raft::device_span<i_t>(sh_out_indices, 64);
  auto tmp_sampled_nodes_data = raft::device_span<const two_opt_cand_t<i_t>>(
    reinterpret_cast<const two_opt_cand_t<i_t>*>(sampled_nodes_data.data() +
                                                 route_id * sol.get_num_orders()),
    sol.get_num_orders());

  top_k_indices_per_row<i_t,
                        two_opt_cand_t<i_t>,
                        top_k_candidates,
                        BLOCK_SIZE,
                        write_diagonal,
                        items_per_thread>(route_id, tmp_sampled_nodes_data, out_costs, out_indices);

  auto cand = out_costs[0];
  if (cand.selection_delta == std::numeric_limits<double>::max()) { return; }

  auto max_active = sol.get_max_active_nodes_for_all_routes();
  auto route      = sol.routes[route_id];
  auto frag_size  = cand.second - (cand.first);
  auto s_route    = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    (i_t*)shmem, route, route.get_num_nodes());
  __syncthreads();
  s_route.copy_from(route);
  __syncthreads();

  typename dimensions_route_t<i_t, f_t, REQUEST>::view_t fragment;
  i_t* dummy;
  // max_fragment_size-1, because the create shared route adds one more already
  thrust::tie(fragment, dummy) = dimensions_route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    reinterpret_cast<i_t*>(raft::alignTo(s_route.shared_end_address(), sizeof(double))),
    sol.problem.dimensions_info,
    max_active - 1);
  __syncthreads();

  // Use best move for cvrptw
  auto n_iter = top_k_candidates;
  for (i_t x = 0; x < n_iter; ++x) {
    __syncthreads();

    if (threadIdx.x == 0) { sh_overlaps = 0; }
    __syncthreads();

    auto cand = out_costs[x];
    if (cand.selection_delta == std::numeric_limits<double>::max()) { break; }

    auto intra_idx = cand.first;
    auto frag_size = cand.second - cand.first;
    // previous node can be impacted
    i_t start = max(intra_idx - 1, 1);
    i_t end   = min(intra_idx + frag_size + 2, route.get_num_nodes());
    for (i_t i = threadIdx.x + start; i < end; i += blockDim.x) {
      if (moved_regions[route_id * max_active + i]) { sh_overlaps = 1; }
    }
    __syncthreads();
    if (sh_overlaps) { continue; }

    cuopt_func_call(
      if (threadIdx.x == 0) { atomicAdd(move_candidates.debug_delta, cand.selection_delta); });

    fragment.parallel_copy_nodes_from(0, s_route, cand.first + 1, frag_size, true);
    __syncthreads();

    for (auto i = threadIdx.x; i < frag_size; i += blockDim.x) {
      auto node      = fragment.get_node(i);
      auto intra_idx = cand.first + 1 + i;
      s_route.set_node(intra_idx, node);
      sol.route_node_map.set_intra_route_idx(node.node_info(), intra_idx);
    }
    __syncthreads();

    mark_impacted_nodes<i_t, f_t, REQUEST>(s_route,
                                           move_candidates,
                                           moved_regions,
                                           cand.first,
                                           frag_size,
                                           max_active,
                                           sol.problem.is_cvrp_intra());
    __syncthreads();
  }

  route_t<i_t, f_t, REQUEST>::view_t::compute_forward_backward_cost(s_route);
  __syncthreads();
  if (threadIdx.x == 0) {
    s_route.compute_cost();
    sol.routes_to_copy[s_route.get_id()]   = 1;
    sol.routes_to_search[s_route.get_id()] = 1;
  }
  __syncthreads();

  route.copy_from(s_route);
}

template <typename i_t, typename f_t, request_t REQUEST>
bool local_search_t<i_t, f_t, REQUEST>::perform_two_opt(
  solution_t<i_t, f_t, REQUEST>& sol, move_candidates_t<i_t, f_t>& move_candidates)
{
  raft::common::nvtx::range fun_scope("run_two_opt");
  sol.global_runtime_checks(false, false, "two_opt_start");
  i_t n_moves_found = 0;
  if (!move_candidates.include_objective) { return false; }

  [[maybe_unused]] double cost_before = 0., cost_after = 0.;

  auto constexpr const n_threads = 64;

  auto n_blocks = move_candidates.nodes_to_search.n_sampled_nodes;
  cuopt_assert(n_blocks > 0, "n_blocks should be positive");
  cuopt_expects(n_blocks > 0, error_type_t::RuntimeError, "A runtime error occurred!");

  if (sol.problem_ptr->is_cvrp_intra()) {
    sampled_nodes_data_.resize(sol.get_n_routes() * sol.get_num_orders(),
                               sol.sol_handle->get_stream());
    async_fill(
      sampled_nodes_data_, is_two_opt_uinitialized_t<i_t>::init_data, sol.sol_handle->get_stream());
  } else {
    two_opt_cand_data_.resize(sol.get_n_routes(), sol.sol_handle->get_stream());
    async_fill(locks_, 0, sol.sol_handle->get_stream());
  }

  auto sh_size = sol.check_routes_can_insert_and_get_sh_size(0);

  if (!set_shmem_of_kernel(find_two_opt_moves<i_t, f_t, REQUEST>, sh_size)) { return false; }

  find_two_opt_moves<i_t, f_t, REQUEST>
    <<<n_blocks, n_threads, sh_size, sol.sol_handle->get_stream()>>>(
      sol.view(),
      move_candidates.view(),
      cuopt::make_span(two_opt_cand_data_),
      cuopt::make_span(sampled_nodes_data_),
      cuopt::make_span(locks_));
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());

  n_moves_found = thrust::count_if(sol.sol_handle->get_thrust_policy(),
                                   sampled_nodes_data_.begin(),
                                   sampled_nodes_data_.end(),
                                   is_two_opt_initialized_t<i_t>());
  if (!n_moves_found) { return false; }

  sol.compute_max_active();

  cuopt_func_call(
    move_candidates.debug_delta.set_value_to_zero_async(sol.sol_handle->get_stream()));
  cuopt_func_call(sol.compute_cost());
  cuopt_func_call(cost_before =
                    sol.problem_ptr->is_cvrp_intra()
                      ? sol.get_total_cost(move_candidates.weights) -
                          sol.get_cost(false, move_candidates.weights)
                      : sol.get_cost(move_candidates.include_objective, move_candidates.weights));

  auto shared_route_size =
    raft::alignTo(sol.check_routes_can_insert_and_get_sh_size(0), sizeof(double));
  auto max_active   = sol.get_max_active_nodes_for_all_routes();
  auto size_of_frag = dimensions_route_t<i_t, f_t, REQUEST>::get_shared_size(
    max_active, sol.problem_ptr->dimensions_info);
  sh_size = shared_route_size + size_of_frag;

  if (sol.problem_ptr->is_cvrp_intra()) {
    if (!set_shmem_of_kernel(execute_recycle<i_t, f_t, REQUEST, n_threads>, sh_size)) {
      return false;
    }
    // Dynamic resizing due to breaks
    moved_regions_.resize(sol.get_n_routes() * sol.get_max_active_nodes_for_all_routes(),
                          sol.sol_handle->get_stream());
    async_fill(moved_regions_, 0, sol.sol_handle->get_stream());
    execute_recycle<i_t, f_t, REQUEST, n_threads>
      <<<sol.get_n_routes(), n_threads, sh_size, sol.sol_handle->get_stream()>>>(
        sol.view(),
        move_candidates.view(),
        cuopt::make_span(sampled_nodes_data_),
        cuopt::make_span(moved_regions_));
  } else {
    if (!set_shmem_of_kernel(execute_two_opt_moves<i_t, f_t, REQUEST>, sh_size)) { return false; }
    execute_two_opt_moves<i_t, f_t, REQUEST>
      <<<sol.get_n_routes(), n_threads, sh_size, sol.sol_handle->get_stream()>>>(
        sol.view(),
        move_candidates.view(),
        cuopt::make_span(two_opt_cand_data_),
        cuopt::make_span(moved_regions_));
  }
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());

  cuopt_func_call(sol.compute_cost());
  cuopt_func_call(cost_after =
                    sol.problem_ptr->is_cvrp_intra()
                      ? sol.get_total_cost(move_candidates.weights) -
                          sol.get_cost(false, move_candidates.weights)
                      : sol.get_cost(move_candidates.include_objective, move_candidates.weights));

  cuopt_assert(abs((cost_before - cost_after) +

                     move_candidates.debug_delta.value(sol.sol_handle->get_stream()) <
                   EPSILON * (1 + abs(cost_before))),
               "Cost mismatch on two_opt costs!");
  cuopt_assert(cost_before - cost_after >= EPSILON, "Cost should improve!");
  sol.global_runtime_checks(false, false, "two_opt_end");
  return true;
}

template bool local_search_t<int, float, request_t::VRP>::perform_two_opt(
  solution_t<int, float, request_t::VRP>& solution, move_candidates_t<int, float>& move_candidates);

template bool local_search_t<int, float, request_t::PDP>::perform_two_opt(
  solution_t<int, float, request_t::PDP>& solution, move_candidates_t<int, float>& move_candidates);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
