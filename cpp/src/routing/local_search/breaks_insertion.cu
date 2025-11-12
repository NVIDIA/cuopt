/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "local_search.cuh"

#include "../solution/solution.cuh"
#include "move_candidates/move_candidates.cuh"
#include "routing/utilities/cuopt_utils.cuh"

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void find_break_insertions_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  const bool include_objective,
  infeasible_cost_t weights,
  typename breaks_move_candidates_t<i_t, f_t>::view_t breaks_move_candidates)
{
  extern __shared__ i_t shmem[];
  const int n_max_break_dims = solution.problem.get_max_break_dimensions();

  i_t ejected_intra_idx = -1;

  auto& locks_per_route     = breaks_move_candidates.locks_per_route;
  auto& best_cand_per_route = breaks_move_candidates.best_cand_per_route;

  i_t route_id          = blockIdx.x / n_max_break_dims;
  i_t ejected_break_dim = blockIdx.x % n_max_break_dims;
  auto global_route     = solution.routes[route_id];
  if (ejected_break_dim >= global_route.get_num_breaks()) { return; }

  for (int i = 0; i < global_route.get_num_nodes(); ++i) {
    auto node = global_route.get_node(i);
    if (node.node_info().is_break() && node.node_info().break_dim() == ejected_break_dim) {
      ejected_intra_idx = i;
      break;
    }
  }

  if (ejected_intra_idx == -1) { return; }

  i_t inserted_break_dim = ejected_break_dim;

  auto route_excess = ls_excess_multiplier_route * global_route.get_weighted_excess(weights);

  const auto old_objective_cost    = global_route.get_objective_cost();
  const auto old_infeasbility_cost = global_route.get_infeasibility_cost();

  typename route_t<i_t, f_t, REQUEST>::view_t sh_route;
  sh_route = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    shmem, global_route, global_route.get_num_nodes());
  sh_route.copy_from(global_route);

  __syncthreads();

  if (threadIdx.x == 0) {
    // don't update node route maps
    sh_route.eject_node(ejected_intra_idx, solution.route_node_map, false);
    route_t<i_t, f_t, REQUEST>::view_t::compute_forward(sh_route);
    route_t<i_t, f_t, REQUEST>::view_t::compute_backward(sh_route);
    sh_route.compute_cost();
  }
  __syncthreads();

  if (inserted_break_dim >= 0) {
    auto break_nodes =
      solution.problem.special_nodes.subset(sh_route.get_vehicle_id(), inserted_break_dim);

    break_cand_t thread_best_cand;

    i_t route_size      = sh_route.get_num_nodes();
    i_t num_break_nodes = break_nodes.size();
    // FIXME:: loop only in between previous and next break dim
    for (int index = threadIdx.x; index < route_size * num_break_nodes; index += blockDim.x) {
      i_t break_node_id = index / route_size;
      i_t insertion_idx = index % route_size;

      auto break_node = create_break_node<i_t, f_t, REQUEST>(
        break_nodes, break_node_id, solution.problem.dimensions_info);

      auto curr_node = sh_route.get_node(insertion_idx);
      auto next_node = sh_route.get_node(insertion_idx + 1);

      curr_node.calculate_forward_all(break_node, sh_route.vehicle_info());

      if (!break_node.forward_feasible(sh_route.vehicle_info(), weights, route_excess)) {
        continue;
      }

      const bool feasible_combine = node_t<i_t, f_t, REQUEST>::combine(
        break_node, next_node, sh_route.vehicle_info(), weights, route_excess);

      if (feasible_combine) {
        double cost_difference = break_node.calculate_forward_all_and_delta(next_node,
                                                                            sh_route.vehicle_info(),
                                                                            include_objective,
                                                                            weights,
                                                                            old_objective_cost,
                                                                            old_infeasbility_cost);

        if (cost_difference < thread_best_cand.cost) {
          thread_best_cand.ejection_idx        = ejected_intra_idx;
          thread_best_cand.insertion_idx       = insertion_idx;
          thread_best_cand.inserting_break_dim = inserted_break_dim;
          thread_best_cand.break_node_idx      = break_node_id;
          thread_best_cand.cost                = cost_difference;
        }
      }
    }

    i_t t_id = threadIdx.x;
    __shared__ i_t reduction_idx;
    __shared__ double reduction_buf[2 * raft::WarpSize];
    double temp_cost = thread_best_cand.cost;
    // FIXME:: Change the block_reduce_ranked so that the first argument is const
    block_reduce_ranked(temp_cost, t_id, reduction_buf, &reduction_idx);

    if (threadIdx.x == reduction_idx) {
      acquire_lock(&locks_per_route[sh_route.get_id()]);
      if (reduction_buf[0] < best_cand_per_route[sh_route.get_id()].cost) {
        best_cand_per_route[sh_route.get_id()] = thread_best_cand;
      }
      release_lock(&locks_per_route[sh_route.get_id()]);
    }
  } else {
    // if you are only ejecting but not inserting
    // note:: this should never happen with the current state of the code
    if (threadIdx.x == 0) {
      acquire_lock(&locks_per_route[sh_route.get_id()]);
      double cost_difference =
        sh_route.get_cost(true, weights) - global_route.get_cost(true, weights);
      if (cost_difference < best_cand_per_route[sh_route.get_id()].cost) {
        break_cand_t cand;
        cand.ejection_idx                      = ejected_intra_idx;
        cand.cost                              = cost_difference;
        best_cand_per_route[sh_route.get_id()] = cand;
      }
      release_lock(&locks_per_route[sh_route.get_id()]);
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
void find_break_insertions(solution_t<i_t, f_t, REQUEST>& sol,
                           move_candidates_t<i_t, f_t>& move_candidates)
{
  if (!sol.problem_ptr->special_nodes.is_empty()) {
    i_t TPB      = 256;
    i_t n_blocks = sol.get_n_routes() * sol.problem_ptr->get_max_break_dimensions();

    sol.compute_max_active();

    move_candidates.breaks_move_candidates.reset(sol.sol_handle);

    // We are only exchanging and not inserting new ones, so we don't need additional memory
    size_t sh_size = sol.get_temp_route_shared_size(0);

    if (!set_shmem_of_kernel(find_break_insertions_kernel<i_t, f_t, REQUEST>, sh_size)) {
      cuopt_assert(false, "Not enough shared memory in find_break_insertions_kernel");
      return;
    }

    find_break_insertions_kernel<i_t, f_t, REQUEST>
      <<<n_blocks, TPB, sh_size, sol.sol_handle->get_stream()>>>(
        sol.view(),
        move_candidates.include_objective,
        move_candidates.weights,
        move_candidates.breaks_move_candidates.view());
    RAFT_CUDA_TRY(cudaStreamSynchronize(sol.sol_handle->get_stream()));
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void execute_break_moves(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                                    typename move_candidates_t<i_t, f_t>::view_t move_candidates)
{
  auto& best_cand_per_route = move_candidates.breaks_move_candidates.best_cand_per_route;

  i_t route_id         = blockIdx.x;
  auto curr_route_cand = best_cand_per_route[route_id];

  if (curr_route_cand.cost > 0.) { return; }

  extern __shared__ i_t shmem[];

  cuopt_assert(route_id < solution.n_routes, "route id should be less than num routes!");
  auto original_route = solution.routes[route_id];

  // at any time, we are exactly deleting one break and inserting one break
  auto sh_route = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    shmem, original_route, original_route.get_num_nodes());
  sh_route.copy_from(original_route);
  __syncthreads();

  i_t ejection_idx = curr_route_cand.ejection_idx;

  if (ejection_idx > 0) {
    if (threadIdx.x == 0) { sh_route.eject_node(ejection_idx, solution.route_node_map); }
    __syncthreads();
  }

  i_t insertion_idx = curr_route_cand.insertion_idx;
  if (insertion_idx >= 0) {
    if (threadIdx.x == 0) {
      auto break_nodes = solution.problem.special_nodes.subset(sh_route.get_vehicle_id(),
                                                               curr_route_cand.inserting_break_dim);
      auto break_node  = create_break_node<i_t, f_t, REQUEST>(
        break_nodes, curr_route_cand.break_node_idx, solution.problem.dimensions_info);

      sh_route.insert_node(insertion_idx, &break_node, solution.route_node_map);
    }

    __syncthreads();
  }

  if (threadIdx.x == 0) {
    route_t<i_t, f_t, REQUEST>::view_t::compute_forward(sh_route);
    route_t<i_t, f_t, REQUEST>::view_t::compute_backward(sh_route);
    solution.routes_to_copy[route_id]   = 1;
    solution.routes_to_search[route_id] = 1;
  }

  __syncthreads();
  original_route.copy_from(sh_route);
}

template <typename i_t, typename f_t, request_t REQUEST>
bool local_search_t<i_t, f_t, REQUEST>::perform_break_moves(solution_t<i_t, f_t, REQUEST>& sol)
{
  move_candidates.reset(sol.sol_handle);
  move_candidates.breaks_move_candidates.reset(sol.sol_handle);

  sol.global_runtime_checks(false, false, "perform_break_moves_begin");

  // We are not leveraging route compatibility for now, when we allow moving breaks across routes,
  // we should use this. So keeping this here commented
  // calculate_route_compatibility(sol);
  find_break_insertions<i_t, f_t, REQUEST>(sol, move_candidates);

  if (!move_candidates.breaks_move_candidates.has_improving_routes(sol.sol_handle)) {
    return false;
  }

  i_t TPB         = 256;
  size_t n_blocks = sol.get_n_routes();

  // We are only exchanging breaks and not inserting any new ones, so we don't need extra memory
  size_t shared_size = sol.check_routes_can_insert_and_get_sh_size(0);
  if (!set_shmem_of_kernel(execute_break_moves<i_t, f_t, REQUEST>, shared_size)) { return false; }
  execute_break_moves<i_t, f_t, REQUEST>
    <<<n_blocks, TPB, shared_size, sol.sol_handle->get_stream()>>>(sol.view(),
                                                                   move_candidates.view());
  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());

  sol.compute_cost();
  sol.sol_handle->sync_stream();

  sol.global_runtime_checks(false, false, "perform_break_moves_end");
  return true;
}

template bool local_search_t<int, float, request_t::PDP>::perform_break_moves(
  solution_t<int, float, request_t::PDP>& sol);
template bool local_search_t<int, float, request_t::VRP>::perform_break_moves(
  solution_t<int, float, request_t::VRP>& sol);
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
