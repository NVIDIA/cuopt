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

#include <cuopt/error.hpp>

#include "guided_ejection_search.cuh"

#include <utilities/cuda_helpers.cuh>
#include "../local_search/local_search.cuh"
#include "../node/node.cuh"
#include "../routing_helpers.cuh"
#include "../solution/pool_allocator.cuh"
#include "../solution/solution.cuh"
#include "../utilities/env_utils.hpp"
#include "compute_delivery_insertions.cuh"
#include "compute_fragment_ejections.cuh"
#include "ejection_pool.cuh"
#include "execute_insertion.cuh"
#include "found_solution.cuh"

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>

#include <cuda_profiler_api.h>

#include <omp.h>

#include <chrono>
#include <fstream>
#include <limits>
#include <random>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
guided_ejection_search_t<i_t, f_t, REQUEST>::guided_ejection_search_t(
  solution_t<i_t, f_t, REQUEST>& solution,
  local_search_t<i_t, f_t, REQUEST>* local_search_ptr,
  std::ofstream* file)
  : solution_ptr(&solution),        //  ref
    ges_loop_save_state(solution),  // cpy ctr
    squeeze_save_state(solution),   // cpy ctr
    local_search_ptr_(local_search_ptr),
    // One unique for each pickup/delivery id pairs + handle route beginning + handle break node
    feasible_candidates_data_(
      solution.get_num_orders() *
          (solution.get_num_orders() + solution.problem_ptr->get_max_break_dimensions()) +
        solution.problem_ptr->get_fleet_size() *
          (1 + solution.problem_ptr->get_max_break_dimensions()) *
          (solution.get_num_orders() + solution.problem_ptr->get_max_break_dimensions()),
      solution.sol_handle->get_stream()),
    feasible_candidates_size_(solution.sol_handle->get_stream()),
    gen_candidate(seed_generator::get_seed()),
    p_scores_(solution.get_num_orders(), solution.sol_handle->get_stream()),
    inserted_requests(solution.get_num_orders(), solution.sol_handle->get_stream()),
    best_squeeze_per_cand(solution.get_num_requests(), solution.sol_handle->get_stream()),
    best_squeeze_per_route(solution.problem_ptr->get_fleet_size(),
                           solution.sol_handle->get_stream()),
    number_of_inserted(solution.sol_handle->get_stream()),
    global_min_p_(solution.sol_handle->get_stream()),
    global_random_counter_(solution.sol_handle->get_stream()),
    // considering max k_max is 6
    global_sequence_(2 * allowed_max_k_max + lexico_result_buffer_size,
                     solution.sol_handle->get_stream()),
    EP(solution.get_num_orders(), solution.sol_handle->get_stream()),
    intermediate_file(file),
    dump_intermediate(false)
{
  raft::common::nvtx::range fun_scope("guided_ejection_search_t");
  reset_p_scores();
}

template <typename i_t, typename f_t, request_t REQUEST>
void guided_ejection_search_t<i_t, f_t, REQUEST>::set_solution_ptr(
  solution_t<i_t, f_t, REQUEST>* solution_ptr_, bool clear_scores)
{
  solution_ptr       = solution_ptr_;
  min_excess_overall = std::numeric_limits<f_t>::max();
  min_ep_size        = std::numeric_limits<i_t>::max();
  EP.index_          = -1;
  if (clear_scores) reset_p_scores();
}

template <typename i_t, typename f_t, request_t REQUEST>
void guided_ejection_search_t<i_t, f_t, REQUEST>::start_timer(
  std::chrono::time_point<std::chrono::steady_clock> start_time, f_t intial_sol_time_limit)
{
  start      = start_time;
  time_limit = intial_sol_time_limit;
}

template <typename i_t, typename f_t, request_t REQUEST>
void guided_ejection_search_t<i_t, f_t, REQUEST>::dump_to_file(std::string msg)
{
  f_t seconds_elapsed =
    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
      .count() /
    1000.f;
  *intermediate_file << msg << " Time : " << seconds_elapsed << std::endl;
  intermediate_file->flush();
}

template <typename i_t, typename f_t, request_t REQUEST>
f_t guided_ejection_search_t<i_t, f_t, REQUEST>::remaining_time() const
{
  f_t seconds_elapsed =
    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
      .count() /
    1000.f;

  return std::max(time_limit - seconds_elapsed, f_t{});
}

template <typename i_t, typename f_t, request_t REQUEST>
bool guided_ejection_search_t<i_t, f_t, REQUEST>::time_stop_condition_reached()
{
  f_t seconds_elapsed =
    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
      .count() /
    1000.f;
  bool finished = seconds_elapsed > time_limit;
  if (dump_intermediate && (seconds_elapsed > dump_interval * dump_iter)) {
    *intermediate_file << solution_ptr->sol_handle->get_device() << "," << seconds_elapsed << ","
                       << (solution_ptr->n_routes + 1) << "," << EP.size() << std::endl;
    intermediate_file->flush();
    dump_iter++;
  }
  // static bool profiler_started = false;
  // if (seconds_elapsed > 180.f && !profiler_started && seconds_elapsed < 209.f) {
  //   cudaProfilerStart();
  //   profiler_started = true;
  // }
  // if (seconds_elapsed > 210.f && profiler_started) {
  //   cudaProfilerStop();
  //   profiler_started = false;
  // }

  return finished;
}

template <typename i_t, typename f_t, request_t REQUEST>
template <request_t r_t, std::enable_if_t<r_t == request_t::VRP, bool>>
void guided_ejection_search_t<i_t, f_t, REQUEST>::reset_p_scores()
{
  thrust::fill(
    solution_ptr->sol_handle->get_thrust_policy(), p_scores_.begin(), p_scores_.end(), 1);
}

template <typename i_t, typename f_t, request_t REQUEST>
template <request_t r_t, std::enable_if_t<r_t == request_t::PDP, bool>>
void guided_ejection_search_t<i_t, f_t, REQUEST>::reset_p_scores()
{
  thrust::uninitialized_fill(
    solution_ptr->sol_handle->get_thrust_policy(), p_scores_.begin(), p_scores_.end(), 0);
  const auto [pickup_indices, _] =
    solution_ptr->problem_ptr->data_view_ptr->get_pickup_delivery_pair();
  thrust::scatter(solution_ptr->sol_handle->get_thrust_policy(),
                  thrust::make_constant_iterator(1),
                  thrust::make_constant_iterator(1) + solution_ptr->get_num_requests(),
                  pickup_indices,
                  p_scores_.begin());
}

template <typename i_t, request_t REQUEST>
__global__ void incr_p_scores(const request_info_t<i_t, REQUEST>* request_id,
                              i_t* p_scores,
                              const bool depot_included)
{
  cuopt_assert(request_id != nullptr, "Request id should not be nullptr");
  // If depot is included (i.e, order locations is not used) node should be at least one. Otherwise
  // it can be zero
  cuopt_assert(request_id->info.node() >= depot_included, "Request id should be positive");
  cuopt_assert(p_scores[request_id->info.node()] > 0, "Inital p score value should be positive");
  p_scores[request_id->info.node()] += 1;
}

template <typename i_t, typename f_t, request_t REQUEST>
void guided_ejection_search_t<i_t, f_t, REQUEST>::shuffle_pool()
{
  raft::common::nvtx::range fun_scope("shuffle_pool");
  // include the ejected request in shuffle
  ++EP.index_;
  EP.random_shuffle();
  --EP.index_;
  if (dump_intermediate) { dump_to_file("Shuffle"); }
}

template <typename i_t, typename f_t, request_t REQUEST>
bool guided_ejection_search_t<i_t, f_t, REQUEST>::guided_ejection_search_loop(i_t& counter,
                                                                              bool minimize_routes,
                                                                              i_t desired_ep_size)
{
  raft::common::nvtx::range fun_scope("guided_ejection_search_loop");
  i_t iteration_limit              = 500000;
  i_t ges_loop_iterations          = 0;
  i_t consecutive_ejection_failure = 1;

  // When running route minimizer, if it is difficult to remove a particular route, we would want to
  // continue with removing another route, so having shorter time is preferable. In case of fixed
  // route loop, we want to run this loop until the time limit
  if (minimize_routes) {
    // this estimation is taken from the following paper
    // Nalepa, J., & Blocho, M. (2016). Enhanced Guided Ejection Search for the Pickup and Delivery
    // Problem with Time Windows. Lecture Notes in Computer Science, 388–398.
    // doi:10.1007/978-3-662-49381-6_37
    // As the number of routes decreases, it becomes increasingly difficult to find feasible
    // insertions, accordingly the number of iterations is increased
    i_t K = solution_ptr->get_n_routes();
    cuopt_assert(K > 0, "number of routes should be positive!");
    i_t N           = solution_ptr->get_num_requests();
    i_t cM          = N * N / K;
    iteration_limit = min(iteration_limit, cM);
  }

  i_t const n_max_multiple_insertions =
    std::max(1, (i_t)(solution_ptr->get_num_requests() * insertion_rate));
  i_t n_insertions = std::min(solution_ptr->get_n_routes(), n_max_multiple_insertions);

  const bool depot_included = solution_ptr->problem_ptr->order_info.depot_included_;

  min_ep_size = min(EP.size(), min_ep_size);

  while (EP.size() > desired_ep_size) {
    solution_ptr->global_runtime_checks(false, true, "ges_while_loop_begin");
    if (time_stop_condition_reached() || ges_loop_iterations == iteration_limit) {
      if (dump_intermediate) {
        dump_to_file("Iteration or time limit exhausted! Trying another route!");
      }
      return false;
    }
    ++ges_loop_iterations;

    if (n_insertions > 1) {
      n_insertions                   = std::min(n_insertions, EP.size());
      const bool enable_perturbation = true;
      n_insertions = try_multiple_feasible_insertions(n_insertions, enable_perturbation);
      continue;
    }

    // LIFO pop
    const auto request          = EP.pop();
    bool single_insertion_found = try_single_insert_with_perturbation(request);
    if (single_insertion_found) { continue; }

    // if that was the last request in the pool, try to squeeze it
    if (EP.size() == 0) {
      if (try_squeeze_feasible(request)) { return true; }
    } else {
      // push the last one back because we will try to insert all
      EP.push_back_last();
      if (!minimize_routes && ((EP.size() <= 15 && EP.size() < min_ep_size) ||
                               (EP.size() <= 5 && EP.size() <= min_ep_size) ||
                               (EP.size() <= 80 && EP.size() < min_ep_size - 10))) {
        min_ep_size = EP.size();
        if (squeeze_all_and_save()) { return true; }
      }
      EP.pop();
    }

    // Increase penalty counter for this request
    incr_p_scores<i_t><<<1, 1, 0, solution_ptr->sol_handle->get_stream()>>>(
      request, p_scores_.data(), depot_included);

    RAFT_CHECK_CUDA(solution_ptr->sol_handle->get_stream());
    bool move_executed = config.frag_eject_first
                           ? execute_best_insertion_ejection_solution(request, counter)
                           : run_lexicographic_search(request);

    if (!move_executed) {
      move_executed = !config.frag_eject_first
                        ? execute_best_insertion_ejection_solution(request, counter)
                        : run_lexicographic_search(request);
    }

    if (!move_executed) {
      if (consecutive_ejection_failure % shuffle_interval == 0) {
        bool squeeze_found = try_squeeze_feasible(request);
        if (!squeeze_found) {
          // if cannot squeeze shuffle
          shuffle_pool();
        } else {
          move_executed                = true;
          consecutive_ejection_failure = 1;
          continue;
        }
      }
      // return only we are minimizing the routes in order to eject another route to the EP
      // for fixed route, the EP is already full of requests
      if (minimize_routes && consecutive_ejection_failure > eject_new_route_threshold) {
        if (dump_intermediate) {
          dump_to_file("Consecutive ejection failure! Trying another route!");
        }
        EP.push_back_last();
        return false;
      }

      RAFT_CHECK_CUDA(solution_ptr->sol_handle->get_stream());
      // reinsert the request and increase the ejection failure counter
      EP.push_back_last();
      consecutive_ejection_failure++;
    } else {
      consecutive_ejection_failure = 1;
    }
    solution_ptr->global_runtime_checks(false, true, "ges_while_loop_end");
  }

  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
bool guided_ejection_search_t<i_t, f_t, REQUEST>::greedy_insert(bool insert_all)
{
  raft::common::nvtx::range fun_scope("greedy_insert");

  i_t const n_max_multiple_insertions = std::max(
    2 * solution_ptr->get_n_routes(), (i_t)(solution_ptr->get_num_requests() * insertion_rate));
  i_t n_insertions = std::min(solution_ptr->get_n_routes(), n_max_multiple_insertions);

  while (EP.size() > 0) {
    solution_ptr->global_runtime_checks(false, true, "greedy_insert_loop");

    n_insertions = std::min(n_insertions, EP.size());
    if (insert_all) { n_insertions = EP.size(); }
    const bool enable_perturbation = false;
    const int succesful_insertions =
      try_multiple_feasible_insertions(n_insertions, enable_perturbation);

    if (succesful_insertions == 0) { return false; }
  }

  return true;
}

// gather all the -1 id'd pickup items into ejection pool
template <typename i_t, typename f_t, request_t REQUEST>
void guided_ejection_search_t<i_t, f_t, REQUEST>::init_ejection_pool()
{
  raft::common::nvtx::range fun_scope("init_ejection_pool");
  solution_ptr->populate_ep_with_unserved(EP);
}

// tries to reach the fixed number of routes
// if we can't reach within the given time an infeasible solution is returned
// @todo provide an option to keep the feasible and return feasible solution too
template <typename i_t, typename f_t, request_t REQUEST>
bool guided_ejection_search_t<i_t, f_t, REQUEST>::fixed_route_loop()
{
  raft::common::nvtx::range fun_scope("fixed_route_loop");
  i_t counter = 0;

  bool all_inserted = greedy_insert();

  // run guided ejection search with a very large EP
  if (!all_inserted && !guided_ejection_search_loop(counter, false)) {
    solution_ptr->global_runtime_checks(false, true, "fixed_route_loop");
    solution_ptr->sol_handle->sync_stream();
    bool success = squeeze_all_and_save();
    if (!success) {
      // In case of prize collection, we are ok with partial solutions. In the absence of prize
      // collection, we require full solution even if it is infeasible and the diversity manager can
      // make the solution feasible
      if (!solution_ptr->problem_ptr->has_prize_collection()) {
        solution_ptr->copy_device_solution(ges_loop_save_state);
      } else {
        // In case of prize collection, sometimes we have infeasible nodes and we could be stuck
        // trying to insert those nodes while there are easy ones and empty routes. This can
        // particularly happen because we give far less time for GES when prize collection is
        // present
        if (!EP.empty()) { greedy_insert(true); }
      }
    } else {
      solution_ptr->global_runtime_checks(true, true, "fixed_route_loop_done");
    }

    EP.clear();
    return success;
  }

  solution_ptr->sol_handle->sync_stream();
  solution_ptr->global_runtime_checks(true, true, "fixed_route_loop_done");
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
i_t guided_ejection_search_t<i_t, f_t, REQUEST>::get_num_non_empty_vehicles()
{
  return solution_ptr->get_n_routes() - solution_ptr->get_num_empty_vehicles();
}

template <typename i_t, typename f_t, request_t REQUEST>
bool guided_ejection_search_t<i_t, f_t, REQUEST>::construct_feasible_solution()
{
  auto& problem  = *(solution_ptr->problem_ptr);
  auto& dim_info = problem.dimensions_info;

  // Clear EP and remove all routes and start from scratch
  EP.clear();
  solution_ptr->populate_ep_with_unserved(EP);

  const auto preferred_order_of_vehicles = problem.get_preferred_order_of_vehicles();

  std::vector<int> vehicle_priority(problem.get_fleet_size());
  for (size_t i = 0; i < preferred_order_of_vehicles.size(); ++i) {
    vehicle_priority[preferred_order_of_vehicles[i]] = i;
  }

  auto unused_vehicles = solution_ptr->get_unused_vehicles();
  std::sort(unused_vehicles.begin(), unused_vehicles.end(), [&](auto& a, auto& b) {
    return vehicle_priority[a] < vehicle_priority[b];
  });

  size_t total_vehicles     = problem.get_fleet_size();
  size_t remaining_vehicles = unused_vehicles.size();
  size_t avg_route_size     = 50;
  // try to have at least 50 requests per route
  while (!EP.empty() && remaining_vehicles > 0) {
    size_t num_new_vehicles = (EP.size() + avg_route_size - 1) / avg_route_size;
    num_new_vehicles        = std::min(remaining_vehicles, num_new_vehicles);
    if (remaining_vehicles == total_vehicles) {
      num_new_vehicles =
        std::max(num_new_vehicles, (size_t)problem.data_view_ptr->get_min_vehicles());
    }

    std::vector<std::pair<int, std::vector<NodeInfo<>>>> new_routes(num_new_vehicles);
    for (size_t i = 0; i < num_new_vehicles; ++i) {
      new_routes[i].first = unused_vehicles[i];
    }
    solution_ptr->add_routes(new_routes);
    // permutate the EP for randomness
    EP.random_shuffle();
    bool all_inserted = greedy_insert();
    if (!all_inserted) { local_search_ptr_->perturb_solution(*solution_ptr); }

    if (dim_info.has_dimension(dim_t::BREAK)) {
      bool breaks_squeeze_success = try_squeeze_breaks_feasible();
      if (!breaks_squeeze_success) {
        solution_ptr->eject_until_feasible();
        EP.clear();
        solution_ptr->populate_ep_with_unserved(EP);
      }
    }

    unused_vehicles = solution_ptr->get_unused_vehicles();
    std::sort(unused_vehicles.begin(), unused_vehicles.end(), [&](auto& a, auto& b) {
      return vehicle_priority[a] < vehicle_priority[b];
    });

    remaining_vehicles = unused_vehicles.size();
    auto used_vehicles = total_vehicles - remaining_vehicles;
    avg_route_size     = std::ceil((problem.get_num_requests() - EP.size()) / used_vehicles);
    // If no request is inserted because of constraints, we need to insert different vehicles
    if (avg_route_size == 0) { avg_route_size = 50; }

    if (EP.empty()) {
      repair_empty_routes();
      solution_ptr->global_runtime_checks(true, true, "construct_feasible_solution_end");
      return true;
    }
  }

  // If we still have unserviced nodes after adding all vehicles, try to insert all the requests
  // greedily. In the default greedy insertions, we only try to insert the last n_insertions from
  // the ejection pool. There is some likelihood of missing attempting to insert some nodes because
  // of that.
  if (!EP.empty()) { greedy_insert(true); }

  solution_ptr->global_runtime_checks(false, true, "construct_feasible_solution_end");
  return false;
}

template <typename i_t, typename f_t, request_t REQUEST>
void guided_ejection_search_t<i_t, f_t, REQUEST>::route_minimizer_loop()
{
  raft::common::nvtx::range fun_scope("route_minimizer_loop");

  // If ejection pool is not empty first run fixed route loop with
  if (!EP.empty()) {
    bool success = fixed_route_loop();
    // If we can't get feasible solution with full fleet, we can't run the minimizer loop
    if (!success) {
      // If fixed route loop is not succesful, (i.e. there is no feasible solution that can serve
      // all nodes), no point in doing route minimization. But we need to remove empty vehicles
      // In case of success, we will use route minimization that will remove these empty vehicles
      // later
      if (solution_ptr->problem_ptr->has_prize_collection()) {
        cuopt_assert(
          solution_ptr->is_feasible(),
          "prize collection should always return feasible solutions, they can be partial "
          "solutions though!");
      }

      // remove empty routes. If there are still empty routes while some orders are not being
      // served, that means they can never serve these orders, could be because of constraints like
      // order vehicle match
      solution_ptr->remove_empty_routes();

      return;
    }
  }

  const auto stream = solution_ptr->sol_handle->get_stream();

  next_route_id_t<i_t, f_t, REQUEST> next_route_id(solution_ptr);

  i_t counter = 0;
  while (!time_stop_condition_reached() &&
         get_num_non_empty_vehicles() >
           std::max(1, solution_ptr->problem_ptr->data_view_ptr->get_min_vehicles())) {
    i_t vehicle_id, random_route_id;
    std::tie(vehicle_id, random_route_id) = next_route_id();
    if (random_route_id < 0) { break; }
    // Save solution state before ges loop in case of route restoration
    stream.synchronize();
    ges_loop_save_state.copy_device_solution(*solution_ptr);
    stream.synchronize();
    solution_ptr->remove_routes(EP, std::vector<i_t>{random_route_id});

    // Routes can be empty when number of vehicles is more than number of requests
    if (EP.empty()) { continue; }

    // If ges loop left early, restore state
    if (!guided_ejection_search_loop(counter, true)) {
      stream.synchronize();
      solution_ptr->copy_device_solution(ges_loop_save_state);
      stream.synchronize();
    }
    solution_ptr->global_runtime_checks(true, true, "route_minimizer_loop");
  }

  // remove empty routes
  // Note, we could technically remove the empty routes upfront, however, we want to eject the
  // lowest priority (highest in numerical value) first. This logic has to be added to route
  // minimization loop
  solution_ptr->remove_empty_routes();
}

ges_config_t get_config()
{
  ges_config_t config;
  set_if_env_set(config.frag_eject_first, "GES_FRAGMENT_FIRST");
  set_if_env_set(config.pert, "GES_PERT");
  set_if_env_set(config.k_max, "GES_K_MAX");
  return config;
}

template class guided_ejection_search_t<int, float, request_t::PDP>;
template class guided_ejection_search_t<int, float, request_t::VRP>;

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
