/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include "../utilities/cuopt_utils.cuh"
#include "local_search.cuh"

#include <cub/cub.cuh>

namespace cuopt {
namespace routing {
namespace detail {

auto constexpr const max_window_size = 20;
constexpr int max_n_neighbors        = 128;

template <typename i_t, typename f_t, request_t REQUEST>
DI thrust::pair<double, double> eval_move(
  typename solution_t<i_t, f_t, REQUEST>::view_t& sol,
  typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
  const typename route_t<i_t, f_t, REQUEST>::view_t& s_route,
  raft::device_span<double> sh_reverse_dist,
  i_t intra_idx,
  i_t insertion_pos,
  i_t window_size,
  i_t route_max_window_size,
  bool reverse)
{
  auto original_window_dist =
    s_route.dimensions.distance_dim.distance_forward[intra_idx + window_size - 1] -
    s_route.dimensions.distance_dim.distance_forward[intra_idx];
  auto new_window_dist = reverse ? sh_reverse_dist[route_max_window_size - 1] -
                                     sh_reverse_dist[route_max_window_size - window_size]
                                 : original_window_dist;

  auto original_previous_intra_frag_next =
    s_route.dimensions.distance_dim.distance_forward[intra_idx + window_size] -
    s_route.dimensions.distance_dim.distance_forward[intra_idx - 1];

  auto frag_begin = reverse ? intra_idx + window_size - 1 : intra_idx;
  auto frag_end   = reverse ? intra_idx : intra_idx + window_size - 1;
  auto insertion_pos_frag_begin =
    get_arc_of_dimension<i_t, f_t, dim_t::DIST>(s_route.get_node(insertion_pos).node_info(),
                                                s_route.get_node(frag_begin).node_info(),
                                                s_route.vehicle_info());

  // in-place
  if (insertion_pos == intra_idx - 1) {
    auto frag_end_frag_next = get_arc_of_dimension<i_t, f_t, dim_t::DIST>(
      s_route.get_node(frag_end).node_info(),
      s_route.get_node(intra_idx + window_size).node_info(),
      s_route.vehicle_info());
    auto delta = insertion_pos_frag_begin + new_window_dist + frag_end_frag_next -
                 original_previous_intra_frag_next;
    return {delta, delta};
  }

  auto frag_end_insertion_pos_next =
    get_arc_of_dimension<i_t, f_t, dim_t::DIST>(s_route.get_node(frag_end).node_info(),
                                                s_route.get_node(insertion_pos + 1).node_info(),
                                                s_route.vehicle_info());

  auto previous_intra_frag_next = get_arc_of_dimension<i_t, f_t, dim_t::DIST>(
    s_route.get_node(intra_idx - 1).node_info(),
    s_route.get_node(intra_idx + window_size).node_info(),
    s_route.vehicle_info());
  auto insertion_pos_insertion_pos_next =
    get_arc_of_dimension<i_t, f_t, dim_t::DIST>(s_route.get_node(insertion_pos).node_info(),
                                                s_route.get_node(insertion_pos + 1).node_info(),
                                                s_route.vehicle_info());
  auto delta = previous_intra_frag_next + insertion_pos_frag_begin + new_window_dist +
               frag_end_insertion_pos_next - insertion_pos_insertion_pos_next -
               original_previous_intra_frag_next;
  return {delta, delta};
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void set_moved_regions_kernel(typename solution_t<i_t, f_t, REQUEST>::view_t sol,
                                         raft::device_span<i_t> moved_regions)
{
  auto route_id   = blockIdx.x;
  auto route      = sol.routes[route_id];
  auto max_active = sol.get_max_active_nodes_for_all_routes();
  for (i_t i = threadIdx.x; i < route.get_num_nodes(); i += blockDim.x) {
    moved_regions[route_id * max_active + i] = route.get_node(i).node_info().node();
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void find_sliding_moves_tsp(
  typename solution_t<i_t, f_t, REQUEST>::view_t sol,
  typename move_candidates_t<i_t, f_t>::view_t move_candidates,
  raft::device_span<sliding_tsp_cand_t<i_t>> sampled_nodes_data,
  raft::device_span<i_t> locks)
{
  extern __shared__ double shmem[];

  const i_t node_idx   = blockIdx.x;
  const auto node_info = move_candidates.nodes_to_search.sampled_nodes_to_search[node_idx];
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

  auto route_max_window_size = min(route.get_num_nodes() - intra_idx, max_window_size);

  auto s_route = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    (i_t*)shmem, route, route.get_num_nodes(), true);
  auto sh_reverse_dist = raft::device_span<double>(
    reinterpret_cast<double*>(raft::alignTo(s_route.shared_end_address(), sizeof(double))),
    route_max_window_size);
  s_route.copy_from(route);
  __syncthreads();

  // reverse and non reverse fragment
  for (i_t tid = threadIdx.x; tid < route_max_window_size; tid += blockDim.x) {
    sh_reverse_dist[tid] =
      route.dimensions.distance_dim
        .reverse_distance[route.get_num_nodes() - intra_idx - (route_max_window_size - 1) + tid];
  }
  __syncthreads();

  const double excess_limit =
    s_route.get_weighted_excess(move_candidates.weights) * ls_excess_multiplier_route;

  sliding_tsp_cand_t<i_t> sliding_tsp_cand = is_sliding_tsp_uinitialized_t<i_t>::init_data();
  double cost_delta, selection_delta;

  constexpr bool exclude_self_in_neighbors = false;  // for reverse op
  const int max_neighbors                  = min(max_n_neighbors, sol.get_num_orders());
  auto nodes_to_consider                   = move_candidates.viables.get_viable_to_pickups(
    node_info.node(), sol.get_num_requests(), max_neighbors, exclude_self_in_neighbors);

  auto n_reverse_types = 2;
  auto n_insertion_pos = nodes_to_consider.size();
  auto total_permut    = route_max_window_size * n_reverse_types *
                      n_insertion_pos;  // forward, backward at every node pos
  for (i_t tid = threadIdx.x; tid < total_permut; tid += blockDim.x) {
    auto node_choice   = nodes_to_consider[tid % n_insertion_pos];
    auto insertion_pos = sol.route_node_map.intra_route_idx_per_node[node_choice];
    auto frag_size     = tid / n_insertion_pos;
    auto window_size   = (frag_size % route_max_window_size) + 1;
    auto reverse       = frag_size / route_max_window_size;

    cuopt_assert(insertion_pos < s_route.get_num_nodes(), "Wrong insertion pos");
    cuopt_assert(reverse == 0 || reverse == 1, "Wrong reverse val");

    if (intra_idx + window_size > s_route.get_num_nodes()) { continue; }

    // overlap
    if (!reverse && insertion_pos >= intra_idx - 1 && insertion_pos < intra_idx + window_size) {
      continue;
    }

    if (reverse && window_size == 1) { continue; }

    // authorize reverse in place (no -1)
    if (reverse && insertion_pos >= intra_idx && insertion_pos < intra_idx + window_size) {
      continue;
    }

    thrust::tie(cost_delta, selection_delta) = eval_move<i_t, f_t, REQUEST>(sol,
                                                                            move_candidates,
                                                                            s_route,
                                                                            sh_reverse_dist,
                                                                            intra_idx,
                                                                            insertion_pos,
                                                                            window_size,
                                                                            route_max_window_size,
                                                                            reverse);

    if (cost_delta > -EPSILON) { continue; }

    if (selection_delta < sliding_tsp_cand.selection_delta) {
      sliding_tsp_cand.insertion_pos   = insertion_pos;
      sliding_tsp_cand.window_size     = window_size;
      sliding_tsp_cand.window_start    = intra_idx;
      sliding_tsp_cand.reverse         = reverse;
      sliding_tsp_cand.selection_delta = selection_delta;
    }
  }
  // reduction
  __shared__ int reduction_index;
  __shared__ double shbuf[warp_size * 2];

  int idx = threadIdx.x;
  // block_reduce_ranked changes sliding_tsp_cand
  double saved_cost = sliding_tsp_cand.selection_delta;
  block_reduce_ranked(saved_cost, idx, shbuf, &reduction_index);

  if (threadIdx.x == reduction_index) { sampled_nodes_data[node_info.node()] = sliding_tsp_cand; }
}

template <typename i_t, typename f_t, request_t REQUEST>
DI void mark_impacted_nodes(const typename route_t<i_t, f_t, REQUEST>::view_t& route,
                            typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
                            const sliding_tsp_cand_t<i_t>& best_candidate,
                            raft::device_span<i_t> moved_regions,
                            i_t n_orders,
                            i_t max_active)
{
  auto route_id = blockIdx.x;
  // mark the window itself and also the surrounding positions
  if (best_candidate.window_start - 1 == 0 || best_candidate.insertion_pos == 0) {
    if (threadIdx.x == 0) {
      move_candidates.nodes_to_search.active_nodes_impacted[route.get_id() + n_orders] = 1;
    }
  }
  // add two more nodes
  i_t start = max(best_candidate.window_start - 1, 1);
  // add two more nodes
  i_t end =
    min(best_candidate.window_start + best_candidate.window_size + 1, route.get_num_nodes());
  for (i_t i = threadIdx.x + start; i < end; i += blockDim.x) {
    cuopt_assert(moved_regions[route_id * max_active + i] != -1, "Node was already moved");
    move_candidates.nodes_to_search.active_nodes_impacted[route.node_id(i)] = 1;
    moved_regions[route_id * max_active + i]                                = -1;
  }

  start = max(best_candidate.insertion_pos, 1);
  end   = min(best_candidate.insertion_pos + 2, route.get_num_nodes());
  // mark the surroundings of the new position
  for (i_t i = threadIdx.x + start; i < end; i += blockDim.x) {
    move_candidates.nodes_to_search.active_nodes_impacted[route.node_id(i)] = 1;
    moved_regions[route_id * max_active + i]                                = -1;
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void execute_sliding_moves_tsp(
  typename solution_t<i_t, f_t, REQUEST>::view_t sol,
  typename move_candidates_t<i_t, f_t>::view_t move_candidates,
  raft::device_span<sliding_tsp_cand_t<i_t>> sampled_nodes_data,
  raft::device_span<i_t> moved_regions)
{
  extern __shared__ double shmem[];
  auto route_id = blockIdx.x;

  auto cand = sampled_nodes_data[0];
  if (cand.selection_delta == std::numeric_limits<double>::max()) { return; }

  auto max_active = sol.get_max_active_nodes_for_all_routes();
  auto route      = sol.routes[route_id];
  auto s_route    = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    (i_t*)shmem, route, route.get_num_nodes(), true);
  s_route.copy_from(route);
  __syncthreads();

  s_route.copy_to_tsp_route();

  __shared__ i_t sh_overlaps;

  for (i_t x = 0; x < sampled_nodes_data.size(); ++x) {
    __syncthreads();

    auto cand = sampled_nodes_data[x];

    if (cand.selection_delta == std::numeric_limits<double>::max()) { break; }

    if (threadIdx.x == 0) { sh_overlaps = 0; }
    __syncthreads();

    // add two more nodes
    i_t start = max(cand.window_start - 1, 1);
    // add two more nodes
    i_t end = min(cand.window_start + cand.window_size + 1, s_route.get_num_nodes());
    for (i_t i = threadIdx.x + start; i < end; i += blockDim.x) {
      if (moved_regions[route_id * max_active + i] == -1) { sh_overlaps = 1; }
    }
    start = max(cand.insertion_pos - 1, 1);
    end   = min(cand.insertion_pos + 2, s_route.get_num_nodes());
    // mark the surroundings of the new position
    for (i_t i = threadIdx.x + start; i < end; i += blockDim.x) {
      if (moved_regions[route_id * max_active + i] == -1) { sh_overlaps = 1; }
    }
    __syncthreads();

    if (sh_overlaps) { continue; }

    cuopt_func_call(
      if (threadIdx.x == 0) { atomicAdd(move_candidates.debug_delta, cand.selection_delta); });

    auto original_node_id = moved_regions[route_id * max_active + cand.window_start];
    auto original_fragment_end_node_id =
      moved_regions[route_id * max_active + cand.window_start + cand.window_size - 1];
    auto original_node_insertion = moved_regions[route_id * max_active + cand.insertion_pos];
    cuopt_assert(original_node_id >= 0, "Moved region node id should be positive");
    cuopt_assert(original_node_insertion >= 0, "Moved region node id should be positive");

    mark_impacted_nodes<i_t, f_t, REQUEST>(
      route, move_candidates, cand, moved_regions, sol.get_num_orders(), max_active);
    __syncthreads();

    if (threadIdx.x == 0) {
      auto original_node_info =
        NodeInfo<i_t>(original_node_id,
                      sol.problem.order_info.get_order_location(original_node_id),
                      node_type_t::DELIVERY);
      auto original_fragment_end_node_info =
        NodeInfo<i_t>(original_fragment_end_node_id,
                      sol.problem.order_info.get_order_location(original_fragment_end_node_id),
                      node_type_t::DELIVERY);
      auto original_node_insertion_info =
        NodeInfo<i_t>(original_node_insertion,
                      sol.problem.order_info.get_order_location(original_node_insertion),
                      node_type_t::DELIVERY);

      auto node_before_fragment = s_route.dimensions.requests.tsp_requests.pred[original_node_id];
      auto node_after_fragment =
        s_route.dimensions.requests.tsp_requests.succ[original_fragment_end_node_id];
      auto fragment_start     = original_node_info;
      auto fragment_end       = original_fragment_end_node_info;
      auto node_insertion_pos = original_node_insertion_info;
      auto node_after_insertion_pos =
        s_route.dimensions.requests.tsp_requests.succ[original_node_insertion];

      s_route.dimensions.requests.tsp_requests.succ[node_before_fragment.node()] =
        node_after_fragment;
      s_route.dimensions.requests.tsp_requests.pred[node_after_fragment.node()] =
        node_before_fragment;

      s_route.dimensions.requests.tsp_requests.succ[node_insertion_pos.node()] = fragment_start;
      s_route.dimensions.requests.tsp_requests.pred[node_after_insertion_pos.node()] = fragment_end;

      s_route.dimensions.requests.tsp_requests.succ[fragment_end.node()] = node_after_insertion_pos;
      s_route.dimensions.requests.tsp_requests.pred[fragment_start.node()] = node_insertion_pos;

      if (cand.reverse) {
        auto start = fragment_start;
        for (i_t i = 0; i < cand.window_size; ++i) {
          raft::swapVals(s_route.dimensions.requests.tsp_requests.pred[start.node()],
                         s_route.dimensions.requests.tsp_requests.succ[start.node()]);
          start = s_route.dimensions.requests.tsp_requests.pred[start.node()];
        }

        // Connecting node links after reverse
        if (cand.window_start == cand.insertion_pos + 1) {  // in place
          s_route.dimensions.requests.tsp_requests.succ[fragment_start.node()] =
            node_after_fragment;
          s_route.dimensions.requests.tsp_requests.pred[node_after_fragment.node()] =
            fragment_start;
        } else {
          s_route.dimensions.requests.tsp_requests.succ[fragment_start.node()] =
            node_after_insertion_pos;
          s_route.dimensions.requests.tsp_requests.pred[node_after_insertion_pos.node()] =
            fragment_start;
        }
        s_route.dimensions.requests.tsp_requests.pred[fragment_end.node()] = node_insertion_pos;
        s_route.dimensions.requests.tsp_requests.succ[node_insertion_pos.node()] = fragment_end;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    auto start = s_route.dimensions.requests.tsp_requests.start;
    for (i_t i = 1; i < s_route.get_num_nodes(); ++i) {
      start          = s_route.dimensions.requests.tsp_requests.succ[start.node()];
      auto intra_idx = sol.route_node_map.get_intra_route_idx(start.node());
      auto node      = route.get_node(intra_idx);
      s_route.set_node(i, node);
      sol.route_node_map.set_intra_route_idx(start, i);
    }

    sol.routes_to_copy[route_id]   = 1;
    sol.routes_to_search[route_id] = 1;
  }
  __syncthreads();

  route.copy_from(s_route);
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void fill_reverse_distances_kernel(typename solution_t<i_t, f_t, REQUEST>::view_t sol)
{
  auto route             = sol.routes[0];
  auto n_nodes           = route.get_num_nodes();
  auto reverse_distances = route.dimensions.distance_dim.reverse_distance;
  for (i_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n_nodes;
       tid += blockDim.x * gridDim.x) {
    reverse_distances[tid] =
      get_arc_of_dimension<i_t, f_t, dim_t::DIST>(route.get_node(n_nodes - tid).node_info(),
                                                  route.get_node(n_nodes - 1 - tid).node_info(),
                                                  route.vehicle_info());
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void fill_forward_distances_kernel(typename solution_t<i_t, f_t, REQUEST>::view_t sol)
{
  auto route             = sol.routes[0];
  auto n_nodes           = route.get_num_nodes();
  auto forward_distances = route.dimensions.distance_dim.distance_forward;
  for (i_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n_nodes;
       tid += blockDim.x * gridDim.x) {
    forward_distances[tid] = get_arc_of_dimension<i_t, f_t, dim_t::DIST>(
      route.get_node(tid).node_info(), route.get_node(tid + 1).node_info(), route.vehicle_info());
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
void resize_temp_storage(solution_t<i_t, f_t, REQUEST>& sol,
                         move_candidates_t<i_t, f_t>& move_candidates,
                         i_t n_nodes,
                         size_t& temp_storage_bytes)
{
  auto distances_ptr = sol.get_route(0).dimensions.distance_dim.distance_forward.data();
  cub::DeviceScan::ExclusiveSum(static_cast<void*>(nullptr),
                                temp_storage_bytes,
                                distances_ptr,
                                distances_ptr,
                                n_nodes + 1,
                                sol.sol_handle->get_stream());

  if (temp_storage_bytes > 0) {
    move_candidates.temp_storage.resize(temp_storage_bytes, sol.sol_handle->get_stream());
  }
}

template <typename i_t, typename f_t, request_t REQUEST, bool reverse>
void compute_cumulative_distances(solution_t<i_t, f_t, REQUEST>& sol,
                                  move_candidates_t<i_t, f_t>& move_candidates,
                                  i_t n_nodes,
                                  i_t n_threads,
                                  size_t temp_storage_bytes)
{
  auto distances_ptr = reverse ? sol.get_route(0).dimensions.distance_dim.reverse_distance.data()
                               : sol.get_route(0).dimensions.distance_dim.distance_forward.data();
  auto n_fill_blocks = (sol.get_num_orders() + n_threads - 1) / n_threads;
  if (reverse) {
    fill_reverse_distances_kernel<i_t, f_t, REQUEST>
      <<<n_fill_blocks, n_threads, 0, sol.sol_handle->get_stream()>>>(sol.view());
    RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
  } else {
    fill_forward_distances_kernel<i_t, f_t, REQUEST>
      <<<n_fill_blocks, n_threads, 0, sol.sol_handle->get_stream()>>>(sol.view());
    RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
  }

  size_t n_temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(static_cast<void*>(nullptr),
                                n_temp_storage_bytes,
                                distances_ptr,
                                distances_ptr,
                                n_nodes + 2,
                                sol.sol_handle->get_stream());

  if (n_temp_storage_bytes > 0) {
    cuopt_expects(n_temp_storage_bytes == temp_storage_bytes,
                  cuopt::error_type_t::RuntimeError,
                  "Cannot resize when using cuda graphs");
  }

  cub::DeviceScan::ExclusiveSum(move_candidates.temp_storage.data(),
                                temp_storage_bytes,
                                distances_ptr,
                                distances_ptr,
                                n_nodes + 2,
                                sol.sol_handle->get_stream());
}

template <typename i_t, typename f_t, request_t REQUEST>
bool local_search_t<i_t, f_t, REQUEST>::perform_sliding_tsp(
  solution_t<i_t, f_t, REQUEST>& sol, move_candidates_t<i_t, f_t>& move_candidates)
{
  raft::common::nvtx::range fun_scope("perform_sliding_tsp");
  sol.global_runtime_checks(false, false, "sliding_tsp_start");
  i_t n_moves_found = 0;
  if (!move_candidates.include_objective) { return false; }
  [[maybe_unused]] double cost_before = 0., cost_after = 0.;

  auto constexpr const n_threads = 128;

  auto shared_route_size = sol.check_routes_can_insert_and_get_sh_size(0);
  sol.compute_max_active();
  moved_regions_.resize(sol.get_n_routes() * sol.get_max_active_nodes_for_all_routes(),
                        sol.sol_handle->get_stream());
  auto n_nodes              = sol.problem_ptr->order_info.get_num_depot_excluded_orders();
  size_t temp_storage_bytes = 0;
  resize_temp_storage<i_t, f_t, REQUEST>(sol, move_candidates, n_nodes, temp_storage_bytes);

  compute_cumulative_distances<i_t, f_t, REQUEST, true>(
    sol, move_candidates, n_nodes, n_threads, temp_storage_bytes);

  auto n_blocks = move_candidates.nodes_to_search.n_sampled_nodes;
  async_fill(sampled_tsp_data_,
             is_sliding_tsp_uinitialized_t<i_t>::init_data(),
             sol.sol_handle->get_stream());

  auto sh_size =
    raft::alignTo(shared_route_size, sizeof(double)) + max_window_size * sizeof(double);

  if (!set_shmem_of_kernel(find_sliding_moves_tsp<i_t, f_t, REQUEST>, sh_size)) { return false; }

  find_sliding_moves_tsp<i_t, f_t, REQUEST>
    <<<n_blocks, n_threads, sh_size, sol.sol_handle->get_stream()>>>(
      sol.view(),
      move_candidates.view(),
      cuopt::make_span(sampled_tsp_data_),
      cuopt::make_span(locks_));
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());

  n_moves_found = thrust::count_if(rmm::exec_policy(sol.sol_handle->get_stream()),
                                   sampled_tsp_data_.begin(),
                                   sampled_tsp_data_.end(),
                                   is_sliding_tsp_initialized_t<i_t>());
  if (!n_moves_found) { return false; }

  async_fill(moved_regions_, 1, sol.sol_handle->get_stream());

  set_moved_regions_kernel<i_t, f_t, REQUEST>
    <<<sol.get_n_routes(), 64, 0, sol.sol_handle->get_stream()>>>(sol.view(),
                                                                  cuopt::make_span(moved_regions_));
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());

  cuopt_func_call(
    move_candidates.debug_delta.set_value_to_zero_async(sol.sol_handle->get_stream()));
  cuopt_func_call(sol.compute_cost());
  cuopt_func_call(cost_before = sol.get_total_cost(move_candidates.weights) -
                                sol.get_cost(false, move_candidates.weights));

  sh_size = shared_route_size;

  if (!set_shmem_of_kernel(execute_sliding_moves_tsp<i_t, f_t, REQUEST>, sh_size)) { return false; }

  thrust::sort(rmm::exec_policy(sol.sol_handle->get_stream()),
               sampled_tsp_data_.begin(),
               sampled_tsp_data_.end(),
               [] __device__(sliding_tsp_cand_t<i_t> cand1, sliding_tsp_cand_t<i_t> cand2) -> bool {
                 return cand1.selection_delta < cand2.selection_delta;
               });

  execute_sliding_moves_tsp<i_t, f_t, REQUEST>
    <<<sol.get_n_routes(), n_threads, sh_size, sol.sol_handle->get_stream()>>>(
      sol.view(),
      move_candidates.view(),
      cuopt::make_span(sampled_tsp_data_),
      cuopt::make_span(moved_regions_));
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());

  compute_cumulative_distances<i_t, f_t, REQUEST, false>(
    sol, move_candidates, n_nodes, n_threads, temp_storage_bytes);

  cuopt_func_call(sol.compute_cost());
  cuopt_func_call(cost_after = sol.get_total_cost(move_candidates.weights) -
                               sol.get_cost(false, move_candidates.weights));

  cuopt_assert(abs((cost_before - cost_after) +
                     move_candidates.debug_delta.value(sol.sol_handle->get_stream()) <
                   EPSILON * (1 + abs(cost_before))),
               "Cost mismatch on sliding_tsp costs!");
  cuopt_assert(cost_before - cost_after >= EPSILON, "Cost should improve!");

  sol.global_runtime_checks(false, false, "sliding_tsp_end");
  return true;
}

template bool local_search_t<int, float, request_t::PDP>::perform_sliding_tsp(
  solution_t<int, float, request_t::PDP>& solution, move_candidates_t<int, float>& move_candidates);
template bool local_search_t<int, float, request_t::VRP>::perform_sliding_tsp(
  solution_t<int, float, request_t::VRP>& solution, move_candidates_t<int, float>& move_candidates);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
