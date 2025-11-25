/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../solution/solution.cuh"

#include <utilities/copy_helpers.hpp>
#include <utilities/seed_generator.cuh>
#include "compute_delivery_insertions.cuh"
#include "compute_fragment_ejections.cuh"
#include "ejection_pool.cuh"
#include "execute_insertion.cuh"
#include "found_solution.cuh"
#include "guided_ejection_search.cuh"

#include <utilities/cuda_helpers.cuh>
#include "../node/node.cuh"
#include "../route/route.cuh"

#include <thrust/copy.h>
#include <cuda/atomic>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void __launch_bounds__(1024, 1)
  execute_feasible_insert(typename solution_t<i_t, f_t, REQUEST>::view_t view,
                          const request_info_t<i_t, REQUEST>* request_id,
                          found_sol_t selected_candidate)
{
  cuopt_assert(request_id != nullptr, "Request id should not be nullptr");
  cuopt_assert(request_id->is_valid(view.problem.order_info.depot_included),
               "Request id should be positive");
  selected_candidate.is_valid(view.n_routes,
                              view.routes[selected_candidate.route_id].get_num_nodes());

  auto& orginal_route = view.routes[selected_candidate.route_id];
  extern __shared__ i_t shmem[];

  auto s_route = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    shmem, orginal_route, orginal_route.get_num_nodes() + request_info_t<i_t, REQUEST>::size());
  __syncthreads();

  s_route.copy_from(orginal_route);
  __syncthreads();

  // Execute insert on temp route (single threaded)
  if (threadIdx.x == 0) {
    auto request_loc = get_request_locations<REQUEST>(selected_candidate);
    execute_insert<i_t, f_t, REQUEST>(view, s_route, request_loc, request_id);
  }
  __syncthreads();
  cuopt_assert(s_route.is_feasible(), "Route should be feasible after node insertion");

  // Copy back to global
  view.routes[s_route.get_id()].copy_from(s_route);
}

template <int BLOCK_SIZE, typename i_t, typename f_t, request_t REQUEST>
__global__ void get_all_feasible_insertion(typename solution_t<i_t, f_t, REQUEST>::view_t view,
                                           const request_info_t<i_t, REQUEST>* request_id,
                                           feasible_move_t feasible_candidates,
                                           int64_t seed)
{
  cuopt_assert(request_id != nullptr, "Request id should not be nullptr");
  cuopt_assert(request_id->is_valid(view.problem.order_info.depot_included),
               "Request id should be positive");
  cuopt_assert(request_id->is_valid(view.get_num_orders()),
               "Request id should be inferior to number of orders");

  extern __shared__ i_t shmem[];
  const i_t route_id = blockIdx.x;
  cuopt_assert(route_id < view.n_routes,
               "Number of blocks (route_id) should be inferior to total number of routes");

  auto& route = view.routes[route_id];

  auto shared_route =
    route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(shmem, route, route.get_num_nodes());

  __syncthreads();
  shared_route.copy_from(route);
  __syncthreads();

  find_all_delivery_insertions<BLOCK_SIZE, true, i_t, f_t, REQUEST>(
    view, shared_route, request_id, feasible_candidates, seed);
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
DI void delete_request(typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                       typename ejection_pool_t<request_info_t<i_t, REQUEST>>::view_t& EP,
                       typename request_route_t<i_t, f_t, REQUEST>::view_t rdim,
                       i_t* to_delete,
                       i_t i)
{
  request_id_t<REQUEST> request_id(rdim.node_id(to_delete[i]));
  EP.push(create_request<i_t, f_t, REQUEST>(solution.problem, request_id));
  solution.route_node_map.reset_node(rdim.node_info[to_delete[i]]);
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
DI void delete_request(typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                       typename ejection_pool_t<request_info_t<i_t, REQUEST>>::view_t& EP,
                       typename request_route_t<i_t, f_t, REQUEST>::view_t rdim,
                       i_t* to_delete,
                       i_t i)
{
  request_id_t<REQUEST> request_id(rdim.node_id(to_delete[i]), rdim.node_id(to_delete[i + 1]));
  EP.push(create_request<i_t, f_t, REQUEST>(solution.problem, request_id));

  solution.route_node_map.reset_node(rdim.node_info[to_delete[i]]);
  solution.route_node_map.reset_node(rdim.node_info[to_delete[i + 1]]);
}

template <int BLOCK_SIZE, typename i_t, typename f_t, request_t REQUEST>
__global__ void __launch_bounds__(1024, 1)
  select_tmp_and_execute_insert(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                                const request_info_t<i_t, REQUEST>* __restrict__ request_info,
                                uint64_t* __restrict__ feasible_candidates,
                                typename ejection_pool_t<request_info_t<i_t, REQUEST>>::view_t EP,
                                i_t fragment_step,
                                i_t fragment_size)
{
  cuopt_assert(*feasible_candidates != static_cast<uint64_t>(-1),
               "Execute insert from tmp route should only happen if there is a feasible candidate");
  cuopt_assert(gridDim.x == 1,
               "select_tmp_and_execute_insert should be executed by only one block");
  cuopt_assert(request_info != nullptr, "Request id should not be nullptr");
  cuopt_assert(request_info->is_valid(solution.problem.order_info.depot_included),
               "Request id should be positive");
  cuopt_assert(fragment_step > 0, "There should be at least one node for ejection");
  cuopt_assert(fragment_size > 0, "There should be at least one node for ejection");

  // Feasible candidate is stored in the first cell since there can be only one (selected through
  // atomicMin)
  found_sol_t selected_move = bit_cast<found_sol_t, uint64_t>(*feasible_candidates);

  // Find concerned route
  // Here route_id contains the global index of the request
  // Only toggle the first 14th bits to find the request
  const i_t found_block_id = selected_move.route_id & 16383;
  const auto request_id    = solution.get_request(found_block_id);
  const i_t frag_to_delete = (selected_move.route_id >> 14) + (fragment_size - fragment_step) + 1;
  cuopt_assert(frag_to_delete >= 1 && frag_to_delete <= fragment_size, "Invalid frag to delete");
  const i_t route_id = solution.route_node_map.get_route_id(request_id.id());
  cuopt_assert(route_id >= 0, "Selected route should be active");
  cuopt_assert(route_id < solution.n_routes,
               "Elected route_id should be smaller than number of routes");
  auto& orginal_route             = solution.routes[route_id];
  const auto orginal_route_length = orginal_route.get_num_nodes();
  cuopt_assert(orginal_route_length > 0, "Route length should be stricly positive");

  // Load orignal route to shared to first fill ejection pool
  extern __shared__ i_t shmem[];
  i_t* to_delete = shmem;
  auto aligned_sz =
    raft::alignTo((size_t)(fragment_size * request_info_t<i_t, REQUEST>::size() * sizeof(i_t)),
                  sizeof(infeasible_cost_t));

  auto s_route = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    (i_t*)(((uint8_t*)shmem) + aligned_sz), orginal_route, orginal_route_length);
  // TODO: needed ?
  __syncthreads();
  s_route.copy_from(orginal_route);
  __syncthreads();

  const i_t intra_pickup_id = solution.route_node_map.get_intra_route_idx(request_id.id());
  cuopt_assert(intra_pickup_id >= 0, "Intra pickup id should be positive");
  cuopt_assert(intra_pickup_id < orginal_route_length,
               "Intra pickup id should inferior to route length");

  fill_to_delete<BLOCK_SIZE, i_t, f_t, REQUEST>(
    solution, s_route, intra_pickup_id, to_delete, frag_to_delete);

  __syncthreads();

  // In contrary to last time, we don't want to sort because we want to keep
  // Pickup & delivery pairs together

  // Locally copy request (corresponding to stack top) since stack will be rewritten
  request_info_t<i_t, REQUEST> _request_info;

  // Add deleted nodes to EP & update globals indices
  if (threadIdx.x == 0) {
    _request_info    = *request_info;
    const auto& rdim = s_route.requests();
    for (int i = 0; i < frag_to_delete * request_info_t<i_t, REQUEST>::size();
         i += request_info_t<i_t, REQUEST>::size()) {
      delete_request<i_t, f_t, REQUEST>(solution, EP, rdim, to_delete, i);
    }
  }

  // Make sure we are done before rewriting shared memory
  __syncthreads();
  sort_to_delete<BLOCK_SIZE, i_t, REQUEST>(to_delete, frag_to_delete);
  cuopt_assert(is_sorted(to_delete, frag_to_delete * request_info_t<i_t, REQUEST>::size()),
               "Array is not sorted after calling sort_to_delete");
  cuopt_assert(is_positive(to_delete, frag_to_delete * request_info_t<i_t, REQUEST>::size()),
               "Array contains negative or 0 values");
  __syncthreads();
  s_route.copy_route_data_after_ejections(
    orginal_route, to_delete, frag_to_delete * request_info_t<i_t, REQUEST>::size());
  __syncthreads();
  // Recompute data on the new route
  if (threadIdx.x == 0) {
    route_t<i_t, f_t, REQUEST>::view_t::compute_forward_in_between(
      s_route, 0, orginal_route_length - request_info_t<i_t, REQUEST>::size() * frag_to_delete);
    route_t<i_t, f_t, REQUEST>::view_t::compute_backward_in_between(
      s_route, 0, orginal_route_length - request_info_t<i_t, REQUEST>::size() * frag_to_delete);
    auto request_loc = get_request_locations<REQUEST>(selected_move);
    execute_insert<i_t, f_t, REQUEST>(solution, s_route, request_loc, &_request_info);
  }
  __syncthreads();
  s_route.compute_intra_indices(solution.route_node_map);
  __syncthreads();
  // Copy back to global
  solution.routes[s_route.get_id()].copy_from(s_route);
}

/**
 * @brief Finds the best feasible insertion of current request considering all insertion-ejection
 * combination
 *
 * Best refers to the feasible solution with the lowest p score
 */
template <typename i_t, typename f_t, request_t REQUEST>
bool guided_ejection_search_t<i_t, f_t, REQUEST>::execute_best_insertion_ejection_solution(
  request_info_t<i_t, REQUEST>* d_request, i_t& counter)
{
  raft::common::nvtx::range fun_scope("execute_best_insertion_ejection_solution");
  counter++;
  constexpr uint64_t unset_val = static_cast<uint64_t>(-1);

  // TODO: constexpr for now, might be different
  constexpr i_t threads_per_block = 64;
  constexpr i_t fragment_step     = solution_t<i_t, f_t, REQUEST>::fragment_step;
  constexpr i_t max_fragment_size = 10;
  cuopt_assert(fragment_step <= 4, "Only 2 bits are supported so 4 max");
  static_assert(max_fragment_size % fragment_step == 0);
  const i_t block_size = solution_ptr->get_num_requests();
  size_t shared_for_tmp_route =
    solution_ptr->check_routes_can_insert_and_get_sh_size(request_info_t<i_t, REQUEST>::size());
  size_t shared_for_delete_array = 0;

  // One block per pickup (request, if node is in ejection pool it's discarded) in the route. Will
  // be the starting point of the deleting fragement window One thread per pickup location in that
  // route No need to check for reallocation here because we delete at least 1 request before
  // inserting
  i_t fragment_size = config.frag_eject_first ? 0 : request_info_t<i_t, REQUEST>::size();
  for (; bit_cast<unsigned long long int, found_sol_t>(feasible_candidates_data_.front_element(
           solution_ptr->sol_handle->get_stream())) == unset_val &&
         !time_stop_condition_reached() && fragment_size + fragment_step <= max_fragment_size;) {
    RAFT_CHECK_CUDA(solution_ptr->sol_handle->get_stream());
    // Increment here and not in for loop to not have it incremented if conditions are not met
    fragment_size += fragment_step;
    shared_for_delete_array =
      raft::alignTo(fragment_size * request_info_t<i_t, REQUEST>::size() * sizeof(i_t),
                    sizeof(infeasible_cost_t));
    if (!set_shmem_of_kernel(
          kernel_get_best_insertion_ejection_solution<threads_per_block, i_t, f_t, REQUEST>,
          shared_for_delete_array + shared_for_tmp_route)) {
      return false;
    }
    kernel_get_best_insertion_ejection_solution<threads_per_block, i_t, f_t, REQUEST>
      <<<solution_ptr->get_num_requests() * fragment_step,
         threads_per_block,
         shared_for_delete_array + shared_for_tmp_route,
         solution_ptr->sol_handle->get_stream()>>>(
        solution_ptr->view(),
        d_request,
        p_scores_.data(),
        fragment_size,
        fragment_step,
        feasible_move_t(cuopt::make_span(feasible_candidates_data_),
                        feasible_candidates_size_.data(),
                        solution_ptr->get_num_orders(),
                        solution_ptr->problem_ptr->get_max_break_dimensions(),
                        solution_ptr->get_n_routes()),
        seed_generator::get_seed());
    RAFT_CHECK_CUDA(solution_ptr->sol_handle->get_stream());
  }

  // Didn't manage to insert even with deleting
  if (fragment_size >= max_fragment_size || time_stop_condition_reached()) { return false; }
  if (!set_shmem_of_kernel(select_tmp_and_execute_insert<1024, i_t, f_t, REQUEST>,
                           shared_for_delete_array + shared_for_tmp_route)) {
    return false;
  }
  // Kernel almost single threaded, more threads just helps to copy route faster
  select_tmp_and_execute_insert<1024, i_t, f_t, REQUEST>
    <<<1,
       1024,
       shared_for_delete_array + shared_for_tmp_route,
       solution_ptr->sol_handle->get_stream()>>>(solution_ptr->view(),
                                                 d_request,
                                                 (uint64_t*)feasible_candidates_data_.data(),
                                                 EP.view(),
                                                 fragment_step,
                                                 fragment_size);
  RAFT_CHECK_CUDA(solution_ptr->sol_handle->get_stream());
  // Update EP index, route_id contains the amount we deleted
  found_sol_t selected_move =
    feasible_candidates_data_.element(0, solution_ptr->sol_handle->get_stream());
  const i_t deleted_frag = (selected_move.route_id >> 14) + fragment_size - fragment_step + 1;
  EP.index_ += deleted_frag;
  solution_ptr->global_runtime_checks(false, true, "execute_best_insertion_ejection_solution");
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
found_sol_t select_random_initialized(rmm::device_uvector<found_sol_t>& feasible_candidates,
                                      solution_t<i_t, f_t, REQUEST>* solution_ptr)
{
  rmm::device_scalar<int> flag(solution_ptr->sol_handle->get_stream());
  flag.set_value_to_zero_async(solution_ptr->sol_handle->get_stream());

  rmm::device_scalar<found_sol_t> random_selected_candidate(solution_ptr->sol_handle->get_stream());

  uint64_t* feasible_cand = reinterpret_cast<uint64_t*>(feasible_candidates.data());
  thrust::for_each(solution_ptr->sol_handle->get_thrust_policy(),
                   feasible_cand,
                   feasible_cand + feasible_candidates.size(),
                   [output_ptr = reinterpret_cast<uint64_t*>(random_selected_candidate.data()),
                    flag       = flag.data()] __device__(auto data) {
                     int updated = 1;
                     if (data != found_sol_t::uninitialized) {
                       if (*flag == 0) { updated = atomicCAS(flag, 0, 1); }
                     }
                     if (!updated) { *output_ptr = data; }
                   });
  RAFT_CHECK_CUDA(solution_ptr->sol_handle->get_stream());
  return random_selected_candidate.value(solution_ptr->sol_handle->get_stream());
}

template <typename i_t, typename f_t, request_t REQUEST>
bool guided_ejection_search_t<i_t, f_t, REQUEST>::perform_insertion(
  const request_info_t<i_t, REQUEST>* request)
{
  raft::common::nvtx::range fun_scope("perform_insertion");
  // Picks a random one and execute the move
  const found_sol_t selected_candidate =
    select_random_initialized(feasible_candidates_data_, solution_ptr);

  size_t shared_for_tmp_route =
    solution_ptr->check_routes_can_insert_and_get_sh_size(request_info_t<i_t, REQUEST>::size());
  if (!set_shmem_of_kernel(execute_feasible_insert<i_t, f_t, REQUEST>, shared_for_tmp_route)) {
    return false;
  }

  execute_feasible_insert<i_t, f_t, REQUEST>
    <<<1, 1024, shared_for_tmp_route, solution_ptr->sol_handle->get_stream()>>>(
      solution_ptr->view(), request, selected_candidate);
  RAFT_CHECK_CUDA(solution_ptr->sol_handle->get_stream());
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
i_t guided_ejection_search_t<i_t, f_t, REQUEST>::find_single_insertion(
  const request_info_t<i_t, REQUEST>* request)
{
  raft::common::nvtx::range fun_scope("find_single_insertion");
  constexpr i_t threads_per_block = 64;
  i_t grid_size                   = solution_ptr->get_n_routes();
  cuopt_assert(grid_size > 0, "Number of route in solution should be positive.");

  // Init feasible candidate
  constexpr i_t init_size = 0;
  feasible_candidates_size_.set_value_async(init_size, solution_ptr->sol_handle->get_stream());
  thrust::uninitialized_fill(
    solution_ptr->sol_handle->get_thrust_policy(),
    (uint64_t*)feasible_candidates_data_.data(),
    (uint64_t*)(feasible_candidates_data_.data() + feasible_candidates_data_.size()),
    found_sol_t::uninitialized);

  size_t shared_for_tmp_route = solution_ptr->check_routes_can_insert_and_get_sh_size();
  if (!set_shmem_of_kernel(get_all_feasible_insertion<threads_per_block, i_t, f_t, REQUEST>,
                           shared_for_tmp_route)) {
    return {};
  }
  get_all_feasible_insertion<threads_per_block, i_t, f_t, REQUEST>
    <<<grid_size,
       threads_per_block,
       shared_for_tmp_route,
       solution_ptr->sol_handle->get_stream()>>>(
      solution_ptr->view(),
      request,
      feasible_move_t(cuopt::make_span(feasible_candidates_data_),
                      feasible_candidates_size_.data(),
                      solution_ptr->get_num_orders(),
                      solution_ptr->problem_ptr->get_max_break_dimensions(),
                      solution_ptr->get_n_routes()),
      seed_generator::get_seed());

  RAFT_CHECK_CUDA(solution_ptr->sol_handle->get_stream());

  return feasible_candidates_size_.value(solution_ptr->sol_handle->get_stream());
}

// tries a single insertion in a while loop while perturbating
// if an insertion is found we return
template <typename i_t, typename f_t, request_t REQUEST>
bool guided_ejection_search_t<i_t, f_t, REQUEST>::try_single_insert_with_perturbation(
  const request_info_t<i_t, REQUEST>* request)
{
  raft::common::nvtx::range fun_scope("try_single_insert_with_perturbation");
  i_t const_1, const_2;
  const_1                = 1;
  const_2                = 8;
  i_t perturbation_count = std::max(const_1, std::min(100 / solution_ptr->n_routes, const_2));
  for (i_t i = 0; i < perturbation_count; ++i) {
    solution_ptr->global_runtime_checks(false, false, "try_single_insert_with_perturbation");
    local_search_ptr_->run_random_local_search(*solution_ptr, false);
  }
  auto n_found_candidates = find_single_insertion(request);
  if (n_found_candidates != 0) { return perform_insertion(request); }
  return false;
}

template bool
guided_ejection_search_t<int, float, request_t::PDP>::execute_best_insertion_ejection_solution(
  request_info_t<int, request_t::PDP>* d_request, int& counter);

template bool
guided_ejection_search_t<int, float, request_t::PDP>::try_single_insert_with_perturbation(
  const request_info_t<int, request_t::PDP>* d_request);

template bool
guided_ejection_search_t<int, float, request_t::VRP>::execute_best_insertion_ejection_solution(
  request_info_t<int, request_t::VRP>* d_request, int& counter);

template bool
guided_ejection_search_t<int, float, request_t::VRP>::try_single_insert_with_perturbation(
  const request_info_t<int, request_t::VRP>* d_request);
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
