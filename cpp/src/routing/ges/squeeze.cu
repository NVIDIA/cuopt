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

#include "squeeze.cuh"

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
bool guided_ejection_search_t<i_t, f_t, REQUEST>::repair_empty_routes()
{
  int counter       = 0;
  auto min_vehicles = solution_ptr->problem_ptr->data_view_ptr->get_min_vehicles();
  rmm::device_scalar<cand_t> best_move({0, 0, std::numeric_limits<double>::max()},
                                       solution_ptr->sol_handle->get_stream());
  // Try every request to non empty route
  auto const n_blocks       = solution_ptr->get_num_requests() * solution_ptr->get_n_routes();
  auto const n_empty_routes = solution_ptr->get_num_empty_vehicles();
  solution_ptr->d_lock.set_value_to_zero_async(solution_ptr->sol_handle->get_stream());
  cand_t uninit_cand{0, 0, std::numeric_limits<double>::max()};

  auto include_objective = true;
  auto excess_limit      = std::numeric_limits<f_t>::epsilon();
  while (get_num_non_empty_vehicles() < min_vehicles && counter < n_empty_routes * 2) {
    auto const sh_route = solution_ptr->check_routes_can_insert_and_get_sh_size();
    const i_t TPB       = std::min(
      128, raft::alignTo(solution_ptr->get_max_active_nodes_for_all_routes(), raft::WarpSize));
    auto const sh_find = sh_route + sizeof(cand_t);

    if (!set_shmem_of_kernel(find_best_empty_route_move<i_t, f_t, REQUEST>, sh_find)) { break; }
    // reset the best move stored
    best_move.set_value_async(uninit_cand, solution_ptr->sol_handle->get_stream());
    find_best_empty_route_move<i_t, f_t, REQUEST>
      <<<n_blocks, TPB, sh_find, solution_ptr->sol_handle->get_stream()>>>(
        solution_ptr->view(), best_move.data(), include_objective, default_weights, excess_limit);
    RAFT_CHECK_CUDA(solution_ptr->sol_handle->get_stream());

    // If unable to find feasible moves, switch to least excess moves
    cand_t best_move_h = best_move.value(solution_ptr->sol_handle->get_stream());
    if (best_move_h.cost_counter.cost == std::numeric_limits<double>::max()) {
      include_objective = false;
      excess_limit      = std::numeric_limits<f_t>::max();
      continue;
    }

    if (!set_shmem_of_kernel(execute_best_empty_route_move<i_t, f_t, REQUEST>, sh_route)) { break; }
    execute_best_empty_route_move<i_t, f_t, REQUEST>
      <<<1, TPB, sh_route, solution_ptr->sol_handle->get_stream()>>>(solution_ptr->view(),
                                                                     best_move.data());
    RAFT_CHECK_CUDA(solution_ptr->sol_handle->get_stream());
    ++counter;
  }
  solution_ptr->sol_handle->sync_stream();
  if (get_num_non_empty_vehicles() < min_vehicles) {
    cuopt_assert(false, "Min vehicles constraint was not repaired, sol is infeasible");
    return false;
  }
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
template <bool squeeze_mode>
i_t guided_ejection_search_t<i_t, f_t, REQUEST>::try_multiple_insert(i_t n_insertions,
                                                                     infeasible_cost_t weights,
                                                                     double excess_limit,
                                                                     bool include_objective)
{
  auto stream        = solution_ptr->sol_handle->get_stream();
  const i_t n_blocks = solution_ptr->n_routes * n_insertions;

  solution_ptr->d_lock.set_value_to_zero_async(stream);
  int counter = 0;
  while (counter != n_insertions) {
    number_of_inserted.set_value_to_zero_async(stream);
    size_t shmem_for_route = solution_ptr->check_routes_can_insert_and_get_sh_size();
    size_t sh_size         = shmem_for_route + sizeof(cand_t);
    const i_t TPB          = std::min(
      128, raft::alignTo(solution_ptr->get_max_active_nodes_for_all_routes(), raft::WarpSize));
    cuopt_expects(
      n_blocks > 0, error_type_t::ValidationError, "Number of blocks should be greater than 0!");
    cuopt_expects(
      TPB > 0, error_type_t::ValidationError, "Number of threads should be greater than 0!");

    async_fill(best_squeeze_per_cand, cand_t{0, 0, std::numeric_limits<double>::max()}, stream);
    async_fill(best_squeeze_per_route, cand_t{0, 0, std::numeric_limits<double>::max()}, stream);

    bool is_set =
      set_shmem_of_kernel(find_all_squeeze_pos<i_t, f_t, REQUEST, squeeze_mode>, sh_size);
    cuopt_expects(is_set, error_type_t::OutOfMemoryError, "Not enough shared memory on device");
    // insert the request greedily to a position that will generate the least excess
    find_all_squeeze_pos<i_t, f_t, REQUEST, squeeze_mode>
      <<<n_blocks, TPB, sh_size, stream>>>(solution_ptr->view(),
                                           EP.view(),
                                           cuopt::make_span(best_squeeze_per_cand),
                                           cuopt::make_span(best_squeeze_per_route),
                                           include_objective,
                                           weights,
                                           excess_limit,
                                           n_insertions,
                                           inserted_requests.data());
    RAFT_CHECK_CUDA(solution_ptr->sol_handle->get_stream());

    if constexpr (squeeze_mode) {
      size_t move_blocks = solution_ptr->get_num_requests();
      extract_best_per_route<i_t, f_t, REQUEST>
        <<<move_blocks, TPB, 0, stream>>>(solution_ptr->view(),
                                          cuopt::make_span(best_squeeze_per_cand),
                                          cuopt::make_span(best_squeeze_per_route));
      RAFT_CHECK_CUDA(solution_ptr->sol_handle->get_stream());
    }

    size_t move_blocks = solution_ptr->get_n_routes();
    is_set =
      set_shmem_of_kernel(execute_all_move<i_t, f_t, REQUEST, squeeze_mode>, shmem_for_route);
    cuopt_expects(is_set, error_type_t::OutOfMemoryError, "Not enough shared memory on device");
    // execute squeeze moves
    execute_all_move<i_t, f_t, REQUEST, squeeze_mode>
      <<<move_blocks, TPB, shmem_for_route, stream>>>(solution_ptr->view(),
                                                      cuopt::make_span(best_squeeze_per_cand),
                                                      cuopt::make_span(best_squeeze_per_route),
                                                      inserted_requests.data(),
                                                      number_of_inserted.data());
    RAFT_CHECK_CUDA(stream);
    auto n_inserted = number_of_inserted.value(stream);

    if (n_inserted == 0) {
      //  Some of the attempted requests could not be inserted in this call or following ones
      //  after perturbations
      increase_multiple_p_scores<i_t, f_t, REQUEST>
        <<<1, 64, 0, stream>>>(EP.view(), p_scores_.data(), inserted_requests.data(), n_insertions);
      break;
    }
    counter += n_inserted;
  }

  solution_ptr->compute_cost();
  solution_ptr->global_runtime_checks(false, false, "try_multiple_insert_end");
  stream.synchronize();
  return counter;
}

template <typename i_t, typename f_t, request_t REQUEST>
i_t guided_ejection_search_t<i_t, f_t, REQUEST>::try_multiple_feasible_insertions(
  i_t n_insertions, const bool enable_perturbation)
{
  raft::common::nvtx::range fun_scope("try_multiple_insert");

  if (enable_perturbation) {
    i_t const_1, const_2;
    const_1                = 1;
    const_2                = 8;
    i_t perturbation_count = std::max(const_1, std::min(100 / solution_ptr->n_routes, const_2));
    for (i_t i = 0; i < perturbation_count; ++i) {
      solution_ptr->global_runtime_checks(false, true, "try_multiple_insert_with_perturbation_1");
      local_search_ptr_->run_random_local_search(*solution_ptr, false);
    }
  }

  constexpr auto const include_objective = true;
  // We should go with the best possible request insertion
  constexpr auto const squeeze_mode = true;

  async_fill(inserted_requests, 0, solution_ptr->sol_handle->get_stream());

  i_t successful_insertions = try_multiple_insert<squeeze_mode>(
    n_insertions, default_weights, std::numeric_limits<f_t>::epsilon(), include_objective);

  eject_inserted_requests<i_t, f_t, REQUEST><<<1, 32, 0, solution_ptr->sol_handle->get_stream()>>>(
    EP.view(), inserted_requests.data(), n_insertions);
  RAFT_CHECK_CUDA(solution_ptr->sol_handle->get_stream());

  // Index is not updated in device view
  EP.index_ -= successful_insertions;
  solution_ptr->global_runtime_checks(false, true, "try_multiple_insert_with_perturbation_end");
  return successful_insertions;
}

template <typename i_t, typename f_t, request_t REQUEST>
void guided_ejection_search_t<i_t, f_t, REQUEST>::squeeze_all_ep()
{
  raft::common::nvtx::range fun_scope("squeeze_all_ep");

  constexpr auto const squeeze_size      = 100;
  constexpr auto const include_objective = false;
  auto run_batches                       = true;

  async_fill(inserted_requests, 0, solution_ptr->sol_handle->get_stream());

  while (run_batches) {
    // Limit the search to number of routes
    auto const batch_size = std::min(solution_ptr->get_n_routes(), EP.size());
    i_t successful_insertions;
    if (EP.size() < squeeze_size) {
      successful_insertions = try_multiple_insert<true>(batch_size,
                                                        local_search_ptr_->move_candidates.weights,
                                                        std::numeric_limits<f_t>::max(),
                                                        include_objective);
    } else {
      successful_insertions = try_multiple_insert<false>(batch_size,
                                                         local_search_ptr_->move_candidates.weights,
                                                         std::numeric_limits<f_t>::max(),
                                                         include_objective);
    }
    if (successful_insertions == 0) { run_batches = false; }

    eject_inserted_requests<i_t, f_t, REQUEST>
      <<<1, 32, 0, solution_ptr->sol_handle->get_stream()>>>(
        EP.view(), inserted_requests.data(), batch_size);
    RAFT_CHECK_CUDA(solution_ptr->sol_handle->get_stream());

    // Index is not updated in device view
    EP.index_ -= successful_insertions;
  }

  solution_ptr->global_runtime_checks(false, false, "squeeze_all_ep_end");
  // cuopt_assert(EP.size() == 0, "EP should be empty at the end of squeeze_all_ep");
  cuopt_expects(EP.size() == 0, error_type_t::RuntimeError, "An internal error occured!");
}

// squeeze all requests do a local search and then compare the excess, if it is better then save
// if feasibilized then return true
template <typename i_t, typename f_t, request_t REQUEST>
bool guided_ejection_search_t<i_t, f_t, REQUEST>::squeeze_all_and_save()
{
  raft::common::nvtx::range fun_scope("squeeze_all_and_save");
  auto stream = solution_ptr->sol_handle->get_stream();
  // copy current solution state
  squeeze_save_state.copy_device_solution(*solution_ptr);
  auto save_ep_size = EP.size();
  squeeze_all_ep();
  constexpr double ls_weights_after_squeeze[] = {1., 1., 1., 1., 1., 1., 1., 1., 1.};
  static_assert(sizeof(ls_weights_after_squeeze) / sizeof(double) == (size_t)dim_t::SIZE);
  constexpr bool include_objective = false;
  auto original_weights            = local_search_ptr_->move_candidates.weights;
  auto original_incl_objective     = local_search_ptr_->move_candidates.include_objective;

  // do local search with high weights on excess and run infeasible search
  double total_excess = solution_ptr->get_total_excess(ls_weights_after_squeeze);
  local_search_ptr_->set_active_weights(ls_weights_after_squeeze, include_objective);

  local_search_ptr_->start_timer(remaining_time());

  solution_ptr->global_runtime_checks(true, false, "squeeze_all_and_save_before_ls");
  const bool consider_unserviced = false;
  const bool enable_time_limit   = true;
  const bool enable_cycle_finder = false;
  local_search_ptr_->run_best_local_search(
    *solution_ptr, consider_unserviced, enable_time_limit, enable_cycle_finder);

  // reset the weights
  total_excess = solution_ptr->get_total_excess(original_weights);
  local_search_ptr_->set_active_weights(original_weights, original_incl_objective);
  // check if solution is feasible at the end
  bool feasibilized = solution_ptr->is_feasible();
  // if the route is feasibled return true
  if (feasibilized) {
    solution_ptr->global_runtime_checks(true, true, "squeeze_all_and_save_after_feasibilized");
    return true;
  }
  // else restore the solution and return false
  else {
    solution_ptr->global_runtime_checks(
      true, false, "squeeze_all_and_save_after_feasibilized_failed");
    // get the excess after local search and save only if it is lower than the one we saved
    if (total_excess < min_excess_overall) {
      min_excess_overall = total_excess;
      // return unsearched squeeze result
      ges_loop_save_state.copy_device_solution(*solution_ptr);
    }
    solution_ptr->copy_device_solution(squeeze_save_state);
    EP.index_ += save_ep_size;
    return false;
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
void guided_ejection_search_t<i_t, f_t, REQUEST>::squeeze(
  const request_info_t<i_t, REQUEST>* request, bool random_route)
{
  raft::common::nvtx::range fun_scope("squeeze");
  auto stream        = solution_ptr->sol_handle->get_stream();
  const i_t n_blocks = solution_ptr->n_routes;
  size_t sh_size     = solution_ptr->check_routes_can_insert_and_get_sh_size() + sizeof(cand_t);
  const i_t TPB      = std::min(
    128, raft::alignTo(solution_ptr->get_max_active_nodes_for_all_routes(), raft::WarpSize));
  rmm::device_scalar<cand_t> best_move({0, 0, std::numeric_limits<double>::max()}, stream);

  solution_ptr->d_lock.set_value_to_zero_async(stream);
  bool is_set = set_shmem_of_kernel(find_best_squeeze_pos<i_t, f_t, REQUEST>, sh_size);
  cuopt_expects(is_set, error_type_t::OutOfMemoryError, "Not enough shared memory on device");
  constexpr bool include_objective = false;
  if (random_route) {
    i_t route_id = dist_candidate(gen_candidate) % solution_ptr->get_n_routes();
    // insert the request greedily to a position that will generate the least excess
    find_best_squeeze_pos<i_t, f_t, REQUEST>
      <<<1, TPB, sh_size, stream>>>(solution_ptr->view(),
                                    request,
                                    best_move.data(),
                                    include_objective,
                                    local_search_ptr_->move_candidates.weights,
                                    route_id);
    RAFT_CHECK_CUDA(solution_ptr->sol_handle->get_stream());
  } else {
    find_best_squeeze_pos<i_t, f_t, REQUEST>
      <<<n_blocks, TPB, sh_size, stream>>>(solution_ptr->view(),
                                           request,
                                           best_move.data(),
                                           include_objective,
                                           local_search_ptr_->move_candidates.weights);
    RAFT_CHECK_CUDA(solution_ptr->sol_handle->get_stream());
  }
  cuopt_assert(best_move.value(stream).cost_counter.cost != std::numeric_limits<double>::max(),
               "At least a move should be found in squeeze");
  // execute squeeze
  execute_move<i_t, f_t><<<1, 1, 0, stream>>>(solution_ptr->view(), request, best_move.data());
  solution_ptr->compute_cost();
  solution_ptr->global_runtime_checks(false, false, "squeeze");
  stream.synchronize();
}

template <typename i_t, typename f_t, request_t REQUEST>
bool guided_ejection_search_t<i_t, f_t, REQUEST>::try_squeeze_feasible(
  const request_info_t<i_t, REQUEST>* request, bool random_route)
{
  raft::common::nvtx::range fun_scope("try_squeeze");
  auto stream = solution_ptr->sol_handle->get_stream();
  // copy current solution state
  squeeze_save_state.copy_device_solution(*solution_ptr);
  squeeze(request, random_route);
  constexpr bool include_objective = false;
  auto original_incl_objective     = local_search_ptr_->move_candidates.include_objective;
  local_search_ptr_->set_active_weights(local_search_ptr_->move_candidates.weights,
                                        include_objective);

  local_search_ptr_->start_timer(remaining_time());

  const bool consider_unserviced = false;
  const bool enable_time_limit   = true;
  const bool enable_cycle_finder = false;
  local_search_ptr_->run_best_local_search(
    *solution_ptr, consider_unserviced, enable_time_limit, enable_cycle_finder);
  // check if solution is feasible at the end
  bool feasibilized = solution_ptr->is_feasible();
  local_search_ptr_->set_active_weights(local_search_ptr_->move_candidates.weights,
                                        original_incl_objective);

  // if the route is feasibled return true
  if (feasibilized) {
    return true;
  }
  // else restore the solution and return false
  else {
    solution_ptr->copy_device_solution(squeeze_save_state);
    return false;
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
void guided_ejection_search_t<i_t, f_t, REQUEST>::squeeze_breaks()
{
  raft::common::nvtx::range fun_scope("squeeze_breaks");
  solution_ptr->global_runtime_checks(false, false, "squeeze_breaks_begin");
  auto stream         = solution_ptr->sol_handle->get_stream();
  size_t n_break_dims = solution_ptr->problem_ptr->get_max_break_dimensions();
  size_t n_blocks     = solution_ptr->n_routes;
  size_t sh_size      = solution_ptr->check_routes_can_insert_and_get_sh_size(n_break_dims) +
                   sizeof(i_t) * n_break_dims;
  size_t TPB = 128;

  if (!set_shmem_of_kernel(squeeze_breaks_kernel<i_t, f_t, REQUEST>, sh_size)) {
    cuopt_assert(false, "Not enough shared memory in squeeze_breaks_kernel");
    return;
  }

  squeeze_breaks_kernel<i_t, f_t, REQUEST><<<n_blocks, TPB, sh_size, stream>>>(
    solution_ptr->view(), false, local_search_ptr_->move_candidates.weights);
  RAFT_CHECK_CUDA(solution_ptr->sol_handle->get_stream());
  solution_ptr->compute_cost();
  solution_ptr->global_runtime_checks(false, false, "squeeze_breaks_end");
  return;
}

template <typename i_t, typename f_t, request_t REQUEST>
bool guided_ejection_search_t<i_t, f_t, REQUEST>::try_squeeze_breaks_feasible()
{
  raft::common::nvtx::range fun_scope("try_squeeze_breaks_feasible");

  size_t n_break_dims = solution_ptr->problem_ptr->get_max_break_dimensions();
  if (n_break_dims == 0) { return solution_ptr->is_feasible(); }
  auto stream = solution_ptr->sol_handle->get_stream();

  squeeze_breaks();

  if (solution_ptr->is_feasible()) { return true; }

  local_search_ptr_->start_timer(remaining_time());

  auto original_incl_objective = local_search_ptr_->move_candidates.include_objective;
  local_search_ptr_->set_active_weights(local_search_ptr_->move_candidates.weights, false);
  const bool consider_unserviced = false;
  const bool enable_time_limit   = true;
  const bool enable_cycle_finder = false;
  local_search_ptr_->run_best_local_search(
    *solution_ptr, consider_unserviced, enable_time_limit, enable_cycle_finder);

  local_search_ptr_->set_active_weights(local_search_ptr_->move_candidates.weights,
                                        original_incl_objective);
  return solution_ptr->is_feasible();
}

template bool guided_ejection_search_t<int, float, request_t::PDP>::repair_empty_routes();
template bool guided_ejection_search_t<int, float, request_t::VRP>::repair_empty_routes();

template bool guided_ejection_search_t<int, float, request_t::PDP>::squeeze_all_and_save();
template bool guided_ejection_search_t<int, float, request_t::VRP>::squeeze_all_and_save();
template void guided_ejection_search_t<int, float, request_t::PDP>::squeeze_all_ep();
template void guided_ejection_search_t<int, float, request_t::VRP>::squeeze_all_ep();

template bool guided_ejection_search_t<int, float, request_t::PDP>::try_squeeze_feasible(
  const request_info_t<int, request_t::PDP>* request, bool random_route);
template bool guided_ejection_search_t<int, float, request_t::VRP>::try_squeeze_feasible(
  const request_info_t<int, request_t::VRP>* request, bool random_route);

template int guided_ejection_search_t<int, float, request_t::PDP>::try_multiple_feasible_insertions(
  int, bool);
template int guided_ejection_search_t<int, float, request_t::VRP>::try_multiple_feasible_insertions(
  int, bool);

template bool guided_ejection_search_t<int, float, request_t::PDP>::try_squeeze_breaks_feasible();
template bool guided_ejection_search_t<int, float, request_t::VRP>::try_squeeze_breaks_feasible();
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
