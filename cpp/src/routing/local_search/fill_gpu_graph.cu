/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../solution/solution.cuh"
#include "local_search.cuh"

#include <utilities/seed_generator.cuh>
#include "../util_kernels/top_k.cuh"
#include "cycle_finder/cycle_graph.hpp"
#include "routing/utilities/cuopt_utils.cuh"

namespace cuopt {
namespace routing {
namespace detail {

auto constexpr write_diagonal = true;

template <typename i_t, typename f_t, request_t REQUEST, int TPB>
__global__ void fill_graph_kernel(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                                  typename move_candidates_t<i_t, f_t>::view_t move_candidates)
{
  // each thread group handles a row in the full move_cand matrix

  i_t inserting_route_id;
  request_id_t<REQUEST> inserted_request_id;
  // set special node id
  if (blockIdx.x == solution.get_num_requests()) {
    inserted_request_id.id() = move_candidates.graph.special_index;
    inserting_route_id       = solution.n_routes;
  } else {
    inserted_request_id = solution.get_request(blockIdx.x);
    inserting_route_id  = solution.route_node_map.get_route_id(inserted_request_id.id());
  }

  cuopt_assert(TPB == blockDim.x, "unexpected block dimension");
  cuopt_assert(inserted_request_id.id() >= 0,
               "inserting_request id should be bigger than or equal to 0!");
  cuopt_assert(inserted_request_id.id() <= (solution.get_num_orders() + solution.n_routes),
               "inserting_request id is invalid at candidate!");
  cuopt_assert(inserted_request_id.id() <= move_candidates.graph.route_ids.size(),
               "inserting_request id is invalid at candidate!");
  cuopt_assert(inserted_request_id.id() <= move_candidates.graph.row_sizes.size(),
               "inserting_request id is invalid at candidate!");

  if (inserting_route_id == -1) {
    if (threadIdx.x == 0) {
      move_candidates.graph.route_ids[inserted_request_id.id()] = inserting_route_id;
      move_candidates.graph.row_sizes[inserted_request_id.id()] = 0;
    }
    return;
  }

  // TODO : if sizeof(cost_counter_t) == sizeof(double) is not true anymore
  // then create static array of cost_counter_t within top_k_indices_per_row and extract cost
  static_assert(sizeof(cost_counter_t) == sizeof(double), "cost_counter_t size mismatch");

  auto row_id                               = inserted_request_id.id();
  raft::device_span<const double> row_costs = raft::device_span<const double>{
    reinterpret_cast<const double*>(move_candidates.cand_matrix.cost_counter.data() +
                                    row_id * move_candidates.cand_matrix.matrix_width),
    static_cast<size_t>(move_candidates.cand_matrix.matrix_width)};

  using ::cuopt::routing::detail::max_graph_nodes_per_row;
  i_t n_items_in_row =
    top_k_indices_per_row<i_t, double, max_graph_nodes_per_row, TPB, write_diagonal>(
      row_id,
      row_costs,
      move_candidates.graph.weights.subspan(row_id * max_graph_nodes_per_row,
                                            max_graph_nodes_per_row),
      move_candidates.graph.indices.subspan(row_id * max_graph_nodes_per_row,
                                            max_graph_nodes_per_row));

  if (threadIdx.x == 0) {
    move_candidates.graph.route_ids[row_id] = inserting_route_id;
    move_candidates.graph.row_sizes[row_id] = n_items_in_row;
  }
}

// loop over diagonal of the matrix and find the best intra index per route
template <typename i_t, typename f_t, request_t REQUEST>
__global__ void fill_intra_candidates(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                                      typename move_candidates_t<i_t, f_t>::view_t move_candidates,
                                      int64_t seed)
{
  __shared__ double shmem[raft::WarpSize * 2];
  __shared__ i_t reduction_idx;
  i_t route_id = blockIdx.x;
  raft::random::PCGenerator thread_rng(seed + (threadIdx.x + blockIdx.x * blockDim.x),
                                       uint64_t(route_id * (threadIdx.x + blockIdx.x * blockDim.x)),
                                       0);
  double thread_best_cost = std::numeric_limits<double>::max();
  i_t thread_best_node_id = -1;
  i_t counter             = 1;

  auto& route_node_map = solution.route_node_map;
  auto& graph          = move_candidates.graph;

  if (threadIdx.x == 0) {
    graph.row_sizes[solution.get_num_orders() + route_id]                           = 1;
    graph.route_ids[solution.get_num_orders() + route_id]                           = route_id;
    graph.weights[(solution.get_num_orders() + route_id) * max_graph_nodes_per_row] = 0.;
    graph.indices[(solution.get_num_orders() + route_id) * max_graph_nodes_per_row] =
      graph.special_index;
  }

  // loop over the diagonal
  for (i_t i = threadIdx.x; i < solution.get_num_requests(); i += blockDim.x) {
    request_id_t<REQUEST> ejected_request_id = solution.get_request(i);
    i_t cand_route_id = solution.route_node_map.get_route_id(ejected_request_id.id());
    if (cand_route_id == route_id) {
      i_t insertion_1, insertion_2, insertion_3, insertion_4;
      double cost_delta;
      const auto cand =
        move_candidates.cand_matrix.get_candidate(ejected_request_id.id(), ejected_request_id.id());
      move_candidates_t<i_t, f_t>::get_candidate(
        cand, insertion_1, insertion_2, insertion_3, insertion_4, cost_delta);
      if (cost_delta == std::numeric_limits<double>::max()) continue;
      // if the insertion position is the same skip this intra move
      const auto pickup_intra = route_node_map.get_intra_route_idx(ejected_request_id.id());
      if constexpr (REQUEST == request_t::PDP) {
        const auto delivery_intra = route_node_map.get_intra_route_idx(ejected_request_id.delivery);
        // if we are inserting back into the same route and same position
        if (insertion_1 == (pickup_intra - 1) && insertion_2 == (delivery_intra - 2)) { continue; }
      } else if constexpr (REQUEST == request_t::VRP) {
        if (insertion_1 == (pickup_intra - 1)) { continue; }
      }
      // cand does not contain information about which nodes should be inseted
      // that informarion is on the matrix indices
      // i am not putting the node id in the candidate structure because we might reuse this later
      // for 4 node insertions
      cuopt_assert(ejected_request_id.id() < solution.get_num_orders(),
                   "Should be a valid VRP node");
      if (cost_delta < thread_best_cost) {
        thread_best_cost    = cost_delta;
        thread_best_node_id = ejected_request_id.id();
      }
    }
  }

  i_t best_idx = threadIdx.x;
  block_reduce_ranked(thread_best_cost, best_idx, shmem, &reduction_idx);
  if (shmem[0] != std::numeric_limits<double>::max() && reduction_idx == threadIdx.x) {
    auto cost_and_id = cand_t{uint(thread_best_node_id), uint(thread_best_node_id), shmem[0]};
    move_candidates.cand_matrix.set_intra_candidate(cost_and_id, route_id);
  }
}

// fills the CSR graph from a full matrix
template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::fill_gpu_graph(solution_t<i_t, f_t, REQUEST>& solution)
{
  raft::common::nvtx::range fun_scope("fill_gpu_graph");
  constexpr i_t TPB = 128;
  solution.sol_handle->sync_stream();
  const auto stream                   = solution.sol_handle->get_stream();
  move_candidates.graph.special_index = solution.get_num_orders() + solution.n_routes;
  fill_intra_candidates<i_t, f_t, REQUEST><<<solution.n_routes, TPB, 0, stream>>>(
    solution.view(), move_candidates.view(), seed_generator::get_seed());
  // +1 for special node
  i_t n_blocks = solution.get_num_requests() + 1;
  fill_graph_kernel<i_t, f_t, REQUEST, TPB>
    <<<n_blocks, TPB, 0, stream>>>(solution.view(), move_candidates.view());
  stream.synchronize();
}
template void local_search_t<int, float, request_t::PDP>::fill_gpu_graph(
  solution_t<int, float, request_t::PDP>&);
template void local_search_t<int, float, request_t::VRP>::fill_gpu_graph(
  solution_t<int, float, request_t::VRP>&);
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
