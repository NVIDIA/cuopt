/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../solution/solution.cuh"
#include "local_search.cuh"

#include <cuda/std/atomic>

#include <chrono>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
DI void add_random_move_thread_safe(const typename solution_t<i_t, f_t, REQUEST>::view_t& sol,
                                    typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
                                    i_t first_node,
                                    i_t second_node,
                                    cand_t cand)
{
  i_t insertion_1, insertion_2, dump;
  double cost_delta;
  if constexpr (REQUEST == request_t::PDP) {
    // we only have cross on ejected node
    if (!sol.problem.order_info.is_pickup_index[first_node]) {
      first_node = sol.problem.order_info.pair_indices[first_node];
    }
    if (!sol.problem.order_info.is_pickup_index[second_node]) {
      second_node = sol.problem.order_info.pair_indices[second_node];
    }
  }
  move_candidates_t<i_t, f_t>::get_candidate(
    cand, insertion_1, insertion_2, dump, dump, cost_delta);
  cuopt_assert(cost_delta != std::numeric_limits<double>::max(), "Candidate must be valid!");
  i_t n_insertions = atomicAdd(move_candidates.move_path.n_insertions, 1);
  auto move        = move_path_t<i_t, f_t>::make_cycle_edge(first_node,   // inserting node
                                                     second_node,  // ejecting node
                                                     insertion_1,
                                                     insertion_2);

  move_candidates.move_path.path[n_insertions] = move;
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void fill_random_route_pair_moves(
  typename solution_t<i_t, f_t, REQUEST>::view_t sol,
  typename move_candidates_t<i_t, f_t>::view_t move_candidates)
{
  i_t first_node_id  = blockIdx.x + (i_t)sol.problem.order_info.depot_included;
  i_t first_route_id = sol.route_node_map.route_id_per_node[first_node_id];
  if (first_route_id == -1) { return; }
  i_t row_size = sol.get_num_orders();

  for (i_t i = threadIdx.x + (i_t)sol.problem.order_info.depot_included; i < row_size;
       i += blockDim.x) {
    i_t second_route_id = sol.route_node_map.route_id_per_node[i];
    // continue if unrouted. non-service nodes (i.e breaks are at the end anyway)
    // we also ignore lower triangular part to avoid duplicates
    if (second_route_id == -1 || first_route_id >= second_route_id) { continue; }
    // get first candidate
    auto cand = move_candidates.cand_matrix.get_candidate(first_node_id, i);
    if (cand.cost_counter.cost == std::numeric_limits<double>::max()) { continue; }
    // get second candidate
    cand = move_candidates.cand_matrix.get_candidate(i, first_node_id);
    if (cand.cost_counter.cost == std::numeric_limits<double>::max()) { continue; }
    // valid candidate found. increment counter for route pair and reserve the index
    i_t route_pair_idx = move_candidates.vrp_move_candidates.get_route_pair_idx(
      first_route_id, second_route_id, sol.n_routes);
    i_t curr_idx = atomicAdd(move_candidates.random_move_candidates.n_moves, 1);
    i_t cand_idx = first_node_id * move_candidates.cand_matrix.matrix_width + i;
    move_candidates.random_move_candidates.moves_per_route_pair[curr_idx] =
      int2{cand_idx, route_pair_idx};
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void extract_offsets_kernel(typename solution_t<i_t, f_t, REQUEST>::view_t sol,
                                       typename move_candidates_t<i_t, f_t>::view_t move_candidates,
                                       i_t n_random_moves)
{
  i_t th_id = threadIdx.x + blockIdx.x * blockDim.x;
  // n_random_moves + 1 to mark the last segment where we end all segments
  if (n_random_moves < th_id || th_id == 0) { return; }
  i_t curr_route_pair_idx = move_candidates.random_move_candidates.moves_per_route_pair[th_id].y;
  i_t prev_route_pair_idx =
    move_candidates.random_move_candidates.moves_per_route_pair[th_id - 1].y;

  cuopt_assert(prev_route_pair_idx != std::numeric_limits<int>::max(),
               "Route pair index should be valid!");
  // if there is no move for a particular route pair, their offsets will remain as 0
  if (curr_route_pair_idx != prev_route_pair_idx) {
    move_candidates.random_move_candidates.move_end_offset[prev_route_pair_idx] = th_id;
    // don't write the end index as there is no segment after this
    if (th_id != n_random_moves) {
      move_candidates.random_move_candidates.move_begin_offset[curr_route_pair_idx] = th_id;
    } else {
      cuopt_assert(curr_route_pair_idx == std::numeric_limits<int>::max(),
                   "Route pair index issue!");
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void pick_random_move_per_route_pair_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t sol,
  typename move_candidates_t<i_t, f_t>::view_t move_candidates,
  i_t seed)
{
  i_t th_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (sol.n_routes * sol.n_routes <= th_id) { return; }
  i_t offset_begin = move_candidates.random_move_candidates.move_begin_offset[th_id];
  i_t offset_end   = move_candidates.random_move_candidates.move_end_offset[th_id];
  cuopt_assert(offset_end >= offset_begin, "Offset end cannot be smaller!");
  i_t n_moves_per_this_route_pair = offset_end - offset_begin;
  if (n_moves_per_this_route_pair == 0) {
    cuopt_assert(offset_begin == 0 && offset_end == 0, "Offsets should zero if not written!");
    return;
  }
  raft::random::PCGenerator thread_rng(seed + (threadIdx.x + blockIdx.x * blockDim.x), 0, 0);
  i_t selected_local_idx = thread_rng.next_u32() % n_moves_per_this_route_pair;
  i_t idx_to_write       = atomicAdd(move_candidates.random_move_candidates.n_selected_moves, 1);
  i_t selected_cand_idx =
    move_candidates.random_move_candidates.moves_per_route_pair[selected_local_idx + offset_begin]
      .x;
  move_candidates.random_move_candidates.selected_move_indices[idx_to_write] = selected_cand_idx;
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void select_random_route_pairs_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t sol,
  typename move_candidates_t<i_t, f_t>::view_t move_candidates,
  i_t seed)
{
  extern __shared__ i_t shmem[];
  __shared__ i_t found_moves;
  auto changed_routes = raft::device_span<i_t>{(i_t*)shmem, (size_t)sol.n_routes};
  // size can be routes/2, but n_routes will never exceed shared memory, keep it like this
  auto executed_move_indices = raft::device_span<i_t>{changed_routes.end(), (size_t)sol.n_routes};
  init_block_shmem(changed_routes, 0);
  init_shmem(found_moves, 0);
  // no need to initialize executed_move_indices as we always access write and read
  __syncthreads();
  if (threadIdx.x == 0) {
    i_t remaining_moves = *move_candidates.random_move_candidates.n_selected_moves;

    raft::random::PCGenerator thread_rng(seed + (threadIdx.x + blockIdx.x * blockDim.x), 0, 0);
    // we want this loop fast, don't do actual candidate insertion here
    while (found_moves < sol.n_routes / 2 && remaining_moves > 0) {
      i_t curr_idx    = thread_rng.next_u32() % remaining_moves;
      i_t cand_idx    = move_candidates.random_move_candidates.selected_move_indices[curr_idx];
      i_t first_node  = cand_idx / move_candidates.cand_matrix.matrix_width;
      i_t second_node = cand_idx % move_candidates.cand_matrix.matrix_width;
      // we could do the insertion path filling here, but since this is a sequential loop just
      // select the moves to execute we will later populate moves in parallel
      i_t first_route_id  = sol.route_node_map.route_id_per_node[first_node];
      i_t second_route_id = sol.route_node_map.route_id_per_node[second_node];
      if (!changed_routes[first_route_id] && !changed_routes[second_route_id]) {
        // we will populate these in parallel later
        executed_move_indices[found_moves++] = cand_idx;
        changed_routes[first_route_id]       = 1;
        changed_routes[second_route_id]      = 1;
      }
      // override curr index with last index and shrink remaning_moves
      move_candidates.random_move_candidates.selected_move_indices[curr_idx] =
        move_candidates.random_move_candidates.selected_move_indices[remaining_moves - 1];
      remaining_moves--;
    }
  }
  __syncthreads();
  // in parallel populate the moves
  for (i_t i = threadIdx.x; i < found_moves; i += blockDim.x) {
    i_t th_cand_idx = executed_move_indices[i];
    i_t first_node  = th_cand_idx / move_candidates.cand_matrix.matrix_width;
    i_t second_node = th_cand_idx % move_candidates.cand_matrix.matrix_width;

    auto cand = move_candidates.cand_matrix.get_candidate(th_cand_idx);
    add_random_move_thread_safe<i_t, f_t, REQUEST>(
      sol, move_candidates, first_node, second_node, cand);
    raft::swapVals(first_node, second_node);
    // reverse candidates
    th_cand_idx = first_node * move_candidates.cand_matrix.matrix_width + second_node;
    cand        = move_candidates.cand_matrix.get_candidate(th_cand_idx);
    add_random_move_thread_safe<i_t, f_t, REQUEST>(
      sol, move_candidates, first_node, second_node, cand);
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
void select_random_route_pairs(solution_t<i_t, f_t, REQUEST>& sol,
                               move_candidates_t<i_t, f_t>& move_candidates)
{
  constexpr i_t nthreads = 256;
  auto nblocks           = 1;
  size_t sh_size         = 2 * sol.n_routes * sizeof(i_t);

  if (!set_shmem_of_kernel(select_random_route_pairs_kernel<i_t, f_t, REQUEST>, sh_size)) {
    cuopt_assert(false, "Not enough shared memory in select_random_route_pairs");
    return;
  }
  select_random_route_pairs_kernel<i_t, f_t, REQUEST>
    <<<nblocks, nthreads, sh_size, sol.sol_handle->get_stream()>>>(
      sol.view(), move_candidates.view(), seed_generator::get_seed());
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
}

template <typename i_t, typename f_t, request_t REQUEST>
void pick_random_move_per_route_pair(solution_t<i_t, f_t, REQUEST>& sol,
                                     move_candidates_t<i_t, f_t>& move_candidates)
{
  constexpr i_t nthreads = 256;
  i_t n_route_pair       = sol.n_routes * sol.n_routes;
  auto nblocks           = (n_route_pair + nthreads - 1) / nthreads;
  pick_random_move_per_route_pair_kernel<i_t, f_t, REQUEST>
    <<<nblocks, nthreads, 0, sol.sol_handle->get_stream()>>>(
      sol.view(), move_candidates.view(), seed_generator::get_seed());
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
}

template <typename i_t, typename f_t, request_t REQUEST>
void get_offsets_of_route_pairs(solution_t<i_t, f_t, REQUEST>& sol,
                                move_candidates_t<i_t, f_t>& move_candidates,
                                i_t n_random_moves)
{
  constexpr i_t nthreads = 256;
  auto nblocks           = ((n_random_moves + 1) + nthreads - 1) / nthreads;
  extract_offsets_kernel<i_t, f_t, REQUEST><<<nblocks, nthreads, 0, sol.sol_handle->get_stream()>>>(
    sol.view(), move_candidates.view(), n_random_moves);
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
}

template <typename i_t, typename f_t, request_t REQUEST>
i_t sort_random_moves_by_route_pair_idx(solution_t<i_t, f_t, REQUEST>& sol,
                                        move_candidates_t<i_t, f_t>& move_candidates)
{
  auto& random_candidates = move_candidates.random_move_candidates;
  // sort the candidates by the route_pair_index
  i_t n_random_moves = random_candidates.n_moves.value(sol.sol_handle->get_stream());
  if (n_random_moves == 0) { return n_random_moves; }
  size_t temp_storage_bytes = 0;
  cub::DeviceMergeSort::SortKeys(
    static_cast<void*>(nullptr),
    temp_storage_bytes,
    random_candidates.moves_per_route_pair.data(),
    n_random_moves,
    [] __device__(int2 a, int2 b) -> bool { return a.y < b.y; },
    sol.sol_handle->get_stream());
  // Allocate temporary storage
  if (random_candidates.d_cub_storage_bytes.size() < temp_storage_bytes) {
    random_candidates.d_cub_storage_bytes.resize(temp_storage_bytes, sol.sol_handle->get_stream());
  }
  // Run sorting operation
  cub::DeviceMergeSort::SortKeys(
    random_candidates.d_cub_storage_bytes.data(),
    temp_storage_bytes,
    random_candidates.moves_per_route_pair.data(),
    n_random_moves,
    [] __device__(int2 a, int2 b) -> bool { return a.y < b.y; },
    sol.sol_handle->get_stream());
  return n_random_moves;
}

template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::populate_random_moves(solution_t<i_t, f_t, REQUEST>& sol)
{
  raft::common::nvtx::range fun_scope("populate_random_moves");
  // extract valid moves
  constexpr i_t nthreads = 256;
  auto nblocks           = sol.get_num_depot_excluded_orders();
  fill_random_route_pair_moves<i_t, f_t, REQUEST>
    <<<nblocks, nthreads, 0, sol.sol_handle->get_stream()>>>(sol.view(), move_candidates.view());
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
  // sort valid moves by route pair index
  i_t n_random_moves = sort_random_moves_by_route_pair_idx(sol, move_candidates);
  if (n_random_moves == 0) return;
  // get the offsets of route pair indices
  get_offsets_of_route_pairs(sol, move_candidates, n_random_moves);
  // pick a random move for each route pair index
  pick_random_move_per_route_pair(sol, move_candidates);
  // select a random route pair to execute
  select_random_route_pairs(sol, move_candidates);
}

template void local_search_t<int, float, request_t::PDP>::populate_random_moves(
  solution_t<int, float, request_t::PDP>&);
template void local_search_t<int, float, request_t::VRP>::populate_random_moves(
  solution_t<int, float, request_t::VRP>&);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
