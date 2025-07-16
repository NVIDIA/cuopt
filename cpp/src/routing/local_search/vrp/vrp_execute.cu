/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "../../util_kernels/set_nodes_data.cuh"
#include "../permutation_helper.cuh"
#include "vrp_execute.cuh"
#include "vrp_search.cuh"

#include <cooperative_groups.h>

namespace cuopt {
namespace routing {
namespace detail {

namespace cg = cooperative_groups;
extern __shared__ double shmem[];

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void compact_best_route_pair_moves(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  typename move_candidates_t<i_t, f_t>::view_t move_candidates)
{
  i_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= solution.n_routes * solution.n_routes) return;
  auto& vrp_candidates = move_candidates.vrp_move_candidates;
  if (vrp_candidates.cost_delta[tid] == std::numeric_limits<double>::max()) return;
  i_t offset = atomicAdd(vrp_candidates.n_best_route_pair_moves, 1);
  vrp_candidates.compacted_move_indices[offset] = tid;
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void extract_non_overlapping_moves_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  typename move_candidates_t<i_t, f_t>::view_t move_candidates,
  uint64_t seed)
{
  auto& vrp_candidates = move_candidates.vrp_move_candidates;
  i_t n_best_route_pair_moves =
    min(*vrp_candidates.n_best_route_pair_moves, max_n_best_route_pair_moves);
  auto changed_routes = raft::device_span<i_t>{(i_t*)shmem, (size_t)solution.n_routes};
  auto shuffled_route_pair_indices =
    raft::device_span<i_t>{changed_routes.end(), (size_t)n_best_route_pair_moves};
  auto sh_best_route_pairs =
    raft::device_span<i_t>{shuffled_route_pair_indices.end(), (size_t)n_best_route_pair_moves};
  init_block_shmem(changed_routes, 0);
  block_sequence(shuffled_route_pair_indices.data(), shuffled_route_pair_indices.size());
  block_copy(sh_best_route_pairs.data(),
             vrp_candidates.compacted_move_indices.data(),
             n_best_route_pair_moves);
  __syncthreads();
  raft::random::PCGenerator thread_rng(seed + (threadIdx.x + blockIdx.x * blockDim.x),
                                       uint64_t((threadIdx.x + blockIdx.x * blockDim.x)),
                                       0);
  if (threadIdx.x == 0) {
    random_shuffle(
      shuffled_route_pair_indices.data(), shuffled_route_pair_indices.size(), thread_rng);
    i_t n_moves_found = 0;
    for (i_t i = 0; i < shuffled_route_pair_indices.size(); ++i) {
      i_t random_idx     = shuffled_route_pair_indices[i];
      i_t route_pair_idx = sh_best_route_pairs[random_idx];
      i_t r_1            = route_pair_idx / solution.n_routes;
      i_t r_2            = route_pair_idx % solution.n_routes;
      cuopt_assert(r_1 != r_2, "Route ids cannot be the same!");
      cuopt_assert(r_1 < r_2, "Only upper triangular matrix!");
      if (changed_routes[r_1] || changed_routes[r_2]) continue;
      if (n_moves_found >= solution.n_routes / 2) break;
      changed_routes[r_1] = 1;
      changed_routes[r_2] = 1;
      vrp_candidates.record_move(route_pair_idx, n_moves_found);
      ++n_moves_found;
      cuopt_func_call(*move_candidates.debug_delta += vrp_candidates.cost_delta[route_pair_idx]);
    }
    vrp_candidates.set_n_changed_routes(n_moves_found);
  }
}

// this function inserts a fragment into a route gap
// the route gap considers the ejected fragment as well as the inserted fragment
template <typename i_t, typename f_t, request_t REQUEST>
DI void insert_fragment_to_route_gap(
  typename route_t<i_t, f_t, REQUEST>::view_t& route,
  i_t insert_pos,
  typename dimensions_route_t<i_t, f_t, REQUEST>::view_t& fragment,
  i_t frag_size)
{
  for (i_t i = threadIdx.x; i < frag_size; i += blockDim.x) {
    route.set_node(i + insert_pos, fragment.get_node(i));
  }
  __syncthreads();
  route_t<i_t, f_t, REQUEST>::view_t::compute_forward_backward_cost(route);
  __syncthreads();
}

template <typename i_t, typename f_t, request_t REQUEST>
DI void prepare_fragment(typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                         typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
                         typename route_t<i_t, f_t, REQUEST>::view_t& route_1,
                         typename route_t<i_t, f_t, REQUEST>::view_t& s_route,
                         typename route_t<i_t, f_t, REQUEST>::view_t& route_eject_buffer,
                         typename dimensions_route_t<i_t, f_t, REQUEST>::view_t& fragment,
                         search_data_t<i_t>& search_data,
                         i_t& start_idx,
                         i_t& frag_size)
{
  i_t move_id                = blockIdx.x / 2;
  i_t is_first_route         = blockIdx.x % 2;
  const auto& vrp_candidates = move_candidates.vrp_move_candidates;
  i_t cand_idx               = vrp_candidates.selected_move_indices[move_id];
  search_data.block_node_id  = vrp_candidates.node_id_1[cand_idx];
  search_data.node_id_2      = vrp_candidates.node_id_2[cand_idx];
  search_data.frag_size_1    = vrp_candidates.frag_size_1[cand_idx];
  search_data.frag_size_2    = vrp_candidates.frag_size_2[cand_idx];
  search_data.move_type      = vrp_candidates.move_type[cand_idx];
  search_data.offset         = vrp_candidates.insert_offset[cand_idx];
  if (search_data.block_node_id >= solution.get_num_orders()) {
    search_data.start_idx_1 = 0;
  } else {
    search_data.start_idx_1 =
      solution.route_node_map.intra_route_idx_per_node[search_data.block_node_id];
  }
  search_data.start_idx_2 =
    solution.route_node_map.intra_route_idx_per_node[search_data.node_id_2] - 1;
  search_data.reversed_frag_1 = false;
  search_data.reversed_frag_2 = false;
  if (search_data.frag_size_1 < 0) {
    search_data.reversed_frag_1 = true;
    search_data.frag_size_1     = -search_data.frag_size_1;
  }
  if (search_data.frag_size_2 < 0) {
    search_data.reversed_frag_2 = true;
    search_data.frag_size_2     = -search_data.frag_size_2;
    search_data.start_idx_2 =
      solution.route_node_map.intra_route_idx_per_node[search_data.node_id_2] -
      search_data.frag_size_2;
  }
  i_t route_id_1;
  if (search_data.block_node_id >= solution.get_num_orders()) {
    route_id_1 = search_data.block_node_id - solution.get_num_orders();
  } else {
    route_id_1 = solution.route_node_map.route_id_per_node[search_data.block_node_id];
  }

  i_t route_id_2 = solution.route_node_map.route_id_per_node[search_data.node_id_2];
  // even blocks execute first part, odd blocks execute second part
  if (!is_first_route) {
    raft::swapVals(search_data.start_idx_1, search_data.start_idx_2);
    raft::swapVals(search_data.block_node_id, search_data.node_id_2);
    raft::swapVals(search_data.frag_size_1, search_data.frag_size_2);
    raft::swapVals(search_data.reversed_frag_1, search_data.reversed_frag_2);
    raft::swapVals(route_id_1, route_id_2);
  }

  cuopt_assert(route_id_1 != route_id_2, "Route ids cannot be the same");
  route_1       = solution.routes[route_id_1];
  auto& route_2 = solution.routes[route_id_2];
  // load fragments from the routes
  i_t depot_excluded_max_route_size =
    solution.get_max_active_nodes_for_all_routes() + *vrp_candidates.max_added_size - 1;
  s_route = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    (i_t*)shmem, route_1, depot_excluded_max_route_size);
  // remove
  __syncthreads();
  route_eject_buffer = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    reinterpret_cast<i_t*>(raft::alignTo(s_route.shared_end_address(), sizeof(double))),
    route_1,
    depot_excluded_max_route_size);
  // remove
  __syncthreads();
  i_t* dummy;
  thrust::tie(fragment, dummy) = dimensions_route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    reinterpret_cast<i_t*>(raft::alignTo(route_eject_buffer.shared_end_address(), sizeof(double))),
    solution.problem.dimensions_info,
    max_fragment_size - 1);
  __syncthreads();
  s_route.copy_from(route_1);
  // two opt start does not load the fragment, we will directly copy the other route to
  // route_eject_buffer since we have grid sync, the other route will not have changed
  if (search_data.move_type > (i_t)vrp_move_t::RELOCATE) {
    route_eject_buffer.copy_from(route_2);
  }
  // this section is valid for fragment exchanges (cross, relocate)
  else {
    if (!is_first_route) {
      if (search_data.offset == 1) {
        fragment.parallel_copy_nodes_from(
          0, route_1, search_data.start_idx_1 + search_data.frag_size_1 + 1, 1);
        __syncthreads();
        fragment.parallel_copy_nodes_from(1,
                                          route_2,
                                          search_data.start_idx_2 + 1,
                                          search_data.frag_size_2,
                                          search_data.reversed_frag_2);
        start_idx = search_data.start_idx_1;
        frag_size = search_data.frag_size_2 + 1;
      } else if (search_data.offset == 0) {
        fragment.parallel_copy_nodes_from(0,
                                          route_2,
                                          search_data.start_idx_2 + 1,
                                          search_data.frag_size_2,
                                          search_data.reversed_frag_2);
        start_idx = search_data.start_idx_1;
        frag_size = search_data.frag_size_2;
      } else {
        fragment.parallel_copy_nodes_from(0,
                                          route_2,
                                          search_data.start_idx_2 + 1,
                                          search_data.frag_size_2,
                                          search_data.reversed_frag_2);
        __syncthreads();
        fragment.parallel_copy_nodes_from(
          search_data.frag_size_2, route_1, search_data.start_idx_1, 1);
        start_idx = search_data.start_idx_1 - 1;
        frag_size = search_data.frag_size_2 + 1;
      }
    } else {
      fragment.parallel_copy_nodes_from(0,
                                        route_2,
                                        search_data.start_idx_2 + 1,
                                        search_data.frag_size_2,
                                        search_data.reversed_frag_2);
      start_idx = search_data.start_idx_1;
      frag_size = search_data.frag_size_2;
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
DI void mark_impacted_nodes(const typename route_t<i_t, f_t, REQUEST>::view_t& route,
                            typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
                            i_t start_idx,
                            i_t inserted_frag_size,
                            i_t n_orders)
{
  // mark the window itself and also the surrounding positions
  if (start_idx == 0 && threadIdx.x == 0) {
    move_candidates.nodes_to_search.active_nodes_impacted[route.get_id() + n_orders] = 1;
  }
  // add two more neighbors as there might be more slack around them
  i_t start = max(start_idx, 1);
  i_t end   = min(start_idx + inserted_frag_size + 1, route.get_num_nodes());
  for (i_t i = threadIdx.x + start; i < end; i += blockDim.x) {
    move_candidates.nodes_to_search.active_nodes_impacted[route.node_id(i)] = 1;
  }
}

// The local search is expected to be pretty fast so that this kernel also need to be fast
template <typename i_t, typename f_t, request_t REQUEST>
__global__ void execute_vrp_moves_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  typename move_candidates_t<i_t, f_t>::view_t move_candidates)
{
  typename route_t<i_t, f_t, REQUEST>::view_t route_1;
  typename route_t<i_t, f_t, REQUEST>::view_t s_route;
  typename route_t<i_t, f_t, REQUEST>::view_t route_eject_buffer;
  typename dimensions_route_t<i_t, f_t, REQUEST>::view_t fragment;
  search_data_t<i_t> search_data;
  i_t start_idx;
  i_t frag_size;

  prepare_fragment<i_t, f_t, REQUEST>(solution,
                                      move_candidates,
                                      route_1,
                                      s_route,
                                      route_eject_buffer,
                                      fragment,
                                      search_data,
                                      start_idx,
                                      frag_size);
  // After loading the fragments, do a grid sync
  cg::this_grid().sync();

  cuopt_func_call(double cost_before =
                    route_1.get_cost(move_candidates.include_objective, move_candidates.weights));
  cuopt_func_call(__syncthreads());
  if (search_data.move_type > (i_t)vrp_move_t::RELOCATE) {
    auto swap_types = search_data.move_type % 2 == 1;
    s_route.copy_from(route_eject_buffer,
                      search_data.start_idx_2 + 1,
                      route_eject_buffer.get_num_nodes() + 1,
                      search_data.start_idx_1 + 1);

    if (threadIdx.x == 0) {
      i_t new_route_size =
        (route_eject_buffer.get_num_nodes() - search_data.start_idx_2) + search_data.start_idx_1;
      s_route.set_num_nodes(new_route_size);
    }
    __syncthreads();

    // For the non swap case, only resetting the end depot node data would be enough.
    if (!solution.problem.fleet_info.is_homogenous_fleet()) {
      auto vehicle_id = swap_types ? route_eject_buffer.get_vehicle_id() : s_route.get_vehicle_id();
      __syncthreads();
      reset_vehicle_id<i_t, f_t, REQUEST>(solution.problem, s_route, vehicle_id);
      __syncthreads();
    }
    mark_impacted_nodes<i_t, f_t, REQUEST>(
      s_route, move_candidates, search_data.start_idx_1, 3, solution.get_num_orders());
    route_t<i_t, f_t, REQUEST>::view_t::compute_forward_backward_cost(s_route);
    __syncthreads();
  } else {
    if (search_data.frag_size_1 > search_data.frag_size_2) {
      s_route.parallel_eject_node(route_eject_buffer,
                                  start_idx + 1,
                                  solution.route_node_map,
                                  false,
                                  search_data.frag_size_1 - search_data.frag_size_2);
    } else {
      s_route.parallel_insert_node(route_eject_buffer,
                                   start_idx,
                                   fragment,
                                   solution.route_node_map,
                                   false,
                                   search_data.frag_size_2 - search_data.frag_size_1);
    }
    __syncthreads();
    cuopt_assert(s_route.get_num_nodes() - (search_data.frag_size_2 - search_data.frag_size_1) ==
                   route_1.get_num_nodes(),
                 "Number of nodes should be consistent");
    cuopt_assert(s_route.get_num_nodes() > 1, "Route cannot be empty!");
    insert_fragment_to_route_gap<i_t, f_t, REQUEST>(s_route, start_idx + 1, fragment, frag_size);
    mark_impacted_nodes<i_t, f_t, REQUEST>(
      s_route, move_candidates, start_idx, search_data.frag_size_2, solution.get_num_orders());
  }

  route_1.copy_from(s_route);

  if (threadIdx.x == 0) { solution.routes_to_copy[s_route.get_id()] = 1; }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void find_max_added_size_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  typename move_candidates_t<i_t, f_t>::view_t move_candidates)
{
  const auto& vrp_candidates = move_candidates.vrp_move_candidates;
  i_t cand_idx               = vrp_candidates.selected_move_indices[blockIdx.x];
  search_data_t<i_t> search_data;
  search_data.block_node_id = vrp_candidates.node_id_1[cand_idx];
  search_data.node_id_2     = vrp_candidates.node_id_2[cand_idx];
  search_data.move_type     = vrp_candidates.move_type[cand_idx];
  if (search_data.move_type <= (i_t)vrp_move_t::RELOCATE) { return; }
  if (search_data.block_node_id >= solution.get_num_orders()) {
    search_data.start_idx_1 = 0;
  } else {
    search_data.start_idx_1 =
      solution.route_node_map.intra_route_idx_per_node[search_data.block_node_id];
  }
  search_data.start_idx_2 =
    solution.route_node_map.intra_route_idx_per_node[search_data.node_id_2] - 1;
  if (threadIdx.x == 0) {
    i_t route_id_1;
    if (search_data.block_node_id >= solution.get_num_orders()) {
      route_id_1 = search_data.block_node_id - solution.get_num_orders();
    } else {
      route_id_1 = solution.route_node_map.route_id_per_node[search_data.block_node_id];
    }
    i_t route_id_2 = solution.route_node_map.route_id_per_node[search_data.node_id_2];
    auto& route_1  = solution.routes[route_id_1];
    auto& route_2  = solution.routes[route_id_2];
    i_t max_route_diff_1 =
      ((route_2.get_num_nodes() - search_data.start_idx_2) + search_data.start_idx_1) -
      route_1.get_num_nodes();
    i_t max_route_diff_2 =
      ((route_1.get_num_nodes() - search_data.start_idx_1) + search_data.start_idx_2) -
      route_2.get_num_nodes();
    atomicMax(vrp_candidates.max_added_size, max(max_route_diff_1, max_route_diff_2));
  }
}

// Finds the non-overlapping route-pair moves in random order
template <typename i_t, typename f_t, request_t REQUEST>
i_t extract_non_overlapping_moves(solution_t<i_t, f_t, REQUEST>& sol,
                                  move_candidates_t<i_t, f_t>& move_candidates)
{
  raft::common::nvtx::range fun_scope("extract_non_overlapping_moves");
  i_t TPB                  = 128;
  i_t n_blocks_for_compact = (sol.n_routes * sol.n_routes + TPB - 1) / TPB;
  compact_best_route_pair_moves<i_t, f_t, REQUEST>
    <<<n_blocks_for_compact, TPB, 0, sol.sol_handle->get_stream()>>>(sol.view(),
                                                                     move_candidates.view());
  i_t n_best_route_pair_moves =
    move_candidates.vrp_move_candidates.n_best_route_pair_moves.value(sol.sol_handle->get_stream());
  n_best_route_pair_moves = std::min(n_best_route_pair_moves, max_n_best_route_pair_moves);
  if (n_best_route_pair_moves == 0) { return 0; }
  size_t sh_size = sizeof(i_t) * (sol.get_n_routes() + n_best_route_pair_moves * 2);
  bool is_set =
    set_shmem_of_kernel(extract_non_overlapping_moves_kernel<i_t, f_t, REQUEST>, sh_size);
  cuopt_assert(is_set,
               "Not enough shared memory on device for extract_non_overlapping_moves_kernel!");
  cuopt_expects(is_set, error_type_t::OutOfMemoryError, "Not enough shared memory on device");
  extract_non_overlapping_moves_kernel<i_t, f_t, REQUEST>
    <<<1, TPB, sh_size, sol.sol_handle->get_stream()>>>(
      sol.view(), move_candidates.view(), seed_generator::get_seed());
  return move_candidates.vrp_move_candidates.n_of_selected_moves.value(
    sol.sol_handle->get_stream());
}

template <typename i_t, typename f_t, request_t REQUEST>
void find_max_added_size(solution_t<i_t, f_t, REQUEST>& sol,
                         move_candidates_t<i_t, f_t>& move_candidates,
                         i_t n_moves_found)
{
  i_t TPB      = 32;
  i_t n_blocks = n_moves_found;
  find_max_added_size_kernel<i_t, f_t, REQUEST>
    <<<n_blocks, TPB, 0, sol.sol_handle->get_stream()>>>(sol.view(), move_candidates.view());
}

template <typename i_t, typename f_t, request_t REQUEST>
bool execute_vrp_moves(solution_t<i_t, f_t, REQUEST>& sol,
                       move_candidates_t<i_t, f_t>& move_candidates,
                       i_t n_moves_found)
{
  raft::common::nvtx::range fun_scope("execute_vrp_moves");
  cuopt_func_call(sol.compute_cost());
  cuopt_func_call(double cost_before =
                    sol.get_cost(move_candidates.include_objective, move_candidates.weights));
  sol.global_runtime_checks(false, false, "execute_vrp_moves_kernel_begin");
  find_max_added_size<i_t, f_t, REQUEST>(sol, move_candidates, n_moves_found);
  i_t max_added_size =
    move_candidates.vrp_move_candidates.max_added_size.value(sol.sol_handle->get_stream());
  size_t shared_route_size =
    raft::alignTo(sol.check_routes_can_insert_and_get_sh_size(max_added_size), sizeof(double));
  size_t size_of_frag = dimensions_route_t<i_t, f_t, REQUEST>::get_shared_size(
    max_fragment_size, sol.problem_ptr->dimensions_info);
  size_t sh_size     = shared_route_size * 2 + size_of_frag;
  i_t TPB            = 64;
  i_t n_blocks       = n_moves_found * 2;
  i_t numBlocksPerSm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocksPerSm, execute_vrp_moves_kernel<i_t, f_t, REQUEST>, TPB, 0);
  // if the number of blocks are larger than the gpu can hold, only execute the max fitting moves
  n_blocks            = std::min(n_blocks,
                      sol.sol_handle->get_device_properties().multiProcessorCount * numBlocksPerSm);
  auto sol_view       = sol.view();
  auto move_cand_view = move_candidates.view();
  // launch
  void* kernelArgs[] = {&sol_view, &move_cand_view};
  dim3 dimBlock(TPB, 1, 1);
  dim3 dimGrid(n_blocks, 1, 1);
  bool is_set = set_shmem_of_kernel(execute_vrp_moves_kernel<i_t, f_t, REQUEST>, sh_size);
  if (!is_set) { return false; }

  cuopt_assert(is_set, "Not enough shared memory on device for execute_vrp_moves_kernel!");
  cuopt_expects(is_set, error_type_t::OutOfMemoryError, "Not enough shared memory on device");
  // FIXME:: Cuda graph is turned off for now because of the assertions triggering in CUDA 12 builds
  // running on V100 move_candidates.vrp_execute_graph.start_capture(sol.sol_handle->get_stream());
  cudaLaunchCooperativeKernel((void*)execute_vrp_moves_kernel<i_t, f_t, REQUEST>,
                              dimGrid,
                              dimBlock,
                              kernelArgs,
                              sh_size,
                              sol.sol_handle->get_stream());
  sol.compute_route_id_per_node();
  sol.compute_cost();
  // move_candidates.vrp_execute_graph.end_capture(sol.sol_handle->get_stream());
  // move_candidates.vrp_execute_graph.launch_graph(sol.sol_handle->get_stream());

  sol.global_runtime_checks(false, false, "execute_vrp_moves_kernel_end");
  cuopt_func_call(double cost_after =
                    sol.get_cost(move_candidates.include_objective, move_candidates.weights));
  cuopt_assert(cost_before - cost_after > EPSILON, "Cost should improve!");
  cuopt_assert(abs((cost_before - cost_after) +
                     move_candidates.debug_delta.value(sol.sol_handle->get_stream()) <
                   EPSILON * (1 + abs(cost_before))),
               "Cost mismatch on vrp costs!");
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
bool select_and_execute_vrp_move(solution_t<i_t, f_t, REQUEST>& sol,
                                 move_candidates_t<i_t, f_t>& move_candidates)
{
  raft::common::nvtx::range fun_scope("select_and_execute_vrp_move");
  cuopt_func_call(
    move_candidates.debug_delta.set_value_to_zero_async(sol.sol_handle->get_stream()));
  i_t n_moves_found = extract_non_overlapping_moves(sol, move_candidates);
  if (n_moves_found == 0) { return false; }
  bool success = execute_vrp_moves(sol, move_candidates, n_moves_found);
  if (!success) { return false; }
  return true;
}

template bool select_and_execute_vrp_move<int, float, request_t::VRP>(
  solution_t<int, float, request_t::VRP>& sol, move_candidates_t<int, float>& move_candidates);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
