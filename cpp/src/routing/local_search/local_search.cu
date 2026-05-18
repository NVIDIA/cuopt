/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// 1 = std::cout timing/cost logs in run_fast_search; 0 = off. Uses #if (not #ifdef).
#define CUOPT_PROFILE_FAST_SEARCH 1

#include "local_search.cuh"

#include <utilities/cuda_helpers.cuh>
#include "compute_ejections.cuh"
#include "compute_insertions.cuh"
#include "vrp/nodes_to_search.cuh"
#include "vrp/vrp_search.cuh"

#include <routing/utilities/cuopt_utils.cuh>
#include <utilities/copy_helpers.hpp>

#include <thrust/fill.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <unordered_set>
#include <utility>

namespace cuopt {
namespace routing {
namespace detail {

#if CUOPT_PROFILE_FAST_SEARCH
namespace {
inline const char* fast_operator_label(fast_operators_t op)
{
  switch (op) {
    case fast_operators_t::SLIDING: return "SLIDING";
    case fast_operators_t::VRP: return "VRP";
    case fast_operators_t::CROSS: return "CROSS";
    case fast_operators_t::REGRET: return "REGRET";
    case fast_operators_t::TWO_OPT: return "TWO_OPT";
    default: return "?";
  }
}

template <typename Sol, typename MC>
double cuopt_fast_search_sync_and_cost(Sol& sol, MC& move_candidates)
{
  sol.sol_handle->sync_stream();
  cuopt_func_call(sol.compute_cost());
  return sol.get_cost(move_candidates.include_objective, move_candidates.weights);
}
}  // namespace
#endif

template <typename i_t, typename f_t, request_t REQUEST>
local_search_t<i_t, f_t, REQUEST>::local_search_t(const solution_handle_t<i_t, f_t>* sol_handle_,
                                                  i_t n_orders,
                                                  i_t max_routes,
                                                  bool depot_included,
                                                  const viables_t<i_t, f_t>& viables_)
  : cycle_finder_small(sol_handle_, depot_included, 5, 50000),
    cycle_finder_big(sol_handle_, depot_included, 5, 50000),
    move_candidates(n_orders, max_routes, sol_handle_, viables_),
    vehicle_assignment(sol_handle_),
    two_opt_cand_data_(0, sol_handle_->get_stream()),
    sampled_nodes_data_(0, sol_handle_->get_stream()),
    found_sliding_solution_data_(max_routes, sol_handle_->get_stream()),
    sampled_tsp_data_(max_routes * n_orders, sol_handle_->get_stream()),
    moved_regions_(0, sol_handle_->get_stream()),
    locks_(max_routes, sol_handle_->get_stream())
{
  raft::common::nvtx::range fun_scope("local_search_t");
}
// sets the search weights and excess values for the local search
template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::set_active_weights(const infeasible_cost_t weights_,
                                                           bool include_objective)
{
  move_candidates.weights.copy_from(weights_);
  move_candidates.include_objective = include_objective;
}

template <typename i_t, typename f_t, request_t REQUEST>
bool local_search_t<i_t, f_t, REQUEST>::run_two_opt_search(solution_t<i_t, f_t, REQUEST>& sol)
{
  raft::common::nvtx::range fun_scope("run_two_opt_search");
  bool move_found = perform_two_opt(sol, move_candidates);
  sol.sol_handle->sync_stream();
  if (move_found) {
    sol.compute_cost();
    cuopt_func_call(sol.check_cost_coherence(move_candidates.weights));
    return true;
  }
  return false;
}

template <typename i_t, typename f_t, request_t REQUEST>
bool local_search_t<i_t, f_t, REQUEST>::run_sliding_search(solution_t<i_t, f_t, REQUEST>& sol)
{
  raft::common::nvtx::range fun_scope("run_sliding_search");
  bool move_found = sol.problem_ptr->is_tsp ? perform_sliding_tsp(sol, move_candidates)
                                            : perform_sliding_window(sol, move_candidates);
  sol.sol_handle->sync_stream();
  if (move_found) {
    sol.compute_cost();
    cuopt_func_call(sol.check_cost_coherence(move_candidates.weights));
    return true;
  }
  return false;
}

template <typename i_t, typename f_t, request_t REQUEST>
bool local_search_t<i_t, f_t, REQUEST>::run_collect_prizes(solution_t<i_t, f_t, REQUEST>& sol)
{
  raft::common::nvtx::range fun_scope("run_collect_prizes");
  // don't run prize collection if there is no prize dimension
  if (!sol.problem_ptr->dimensions_info.has_dimension(dim_t::PRIZE)) { return false; }

  bool move_found = perform_prize_collection(sol);
  sol.sol_handle->sync_stream();
  if (move_found) {
    sol.compute_cost();
    cuopt_func_call(sol.check_cost_coherence(move_candidates.weights));
    return true;
  }
  return false;
}

template <typename i_t, typename f_t, request_t REQUEST>
bool local_search_t<i_t, f_t, REQUEST>::run_cross_search(solution_t<i_t, f_t, REQUEST>& sol)
{
  raft::common::nvtx::range fun_scope("run_cross_search");
  if (sol.n_routes < 2) { return false; }
  // independent thread scheduling is not present in pascal and we use it in populate_cross_list
  // kernel
  if (sol.sol_handle->get_device_properties().major < 7) { return false; }
  move_candidates.reset(sol.sol_handle);
  calculate_route_compatibility(sol);
  [[maybe_unused]] double cost_before = 0., cost_after = 0.;
  cuopt_func_call(sol.compute_cost());
  cuopt_func_call(cost_before =
                    sol.get_cost(move_candidates.include_objective, move_candidates.weights));
  find_insertions<i_t, f_t, REQUEST>(sol, move_candidates, search_type_t::CROSS);
  // directly populate move path after a cross move
  auto success = populate_cross_moves(sol, move_candidates);
  sol.sol_handle->sync_stream();
  // if we found a move, perform moves and continue
  if (success && move_candidates.move_path.n_insertions.value(sol.sol_handle->get_stream()) != 0) {
    perform_moves(sol, move_candidates);
    cuopt_func_call(sol.check_cost_coherence(move_candidates.weights));
    cuopt_func_call(cost_after =
                      sol.get_cost(move_candidates.include_objective, move_candidates.weights));

    cuopt_assert(cost_before - cost_after > EPSILON, "Cost should improve!");
    cuopt_assert(abs((cost_before - cost_after) -
                     move_candidates.debug_delta.value(sol.sol_handle->get_stream())) <
                   EPSILON * (1 + abs(cost_before)),
                 "Cost mismatch on cross costs!");
    return true;
  }
  return false;
}

template <typename i_t, typename f_t, request_t REQUEST>
template <request_t r_t, std::enable_if_t<r_t == request_t::PDP, bool>>
bool local_search_t<i_t, f_t, REQUEST>::run_fast_search(solution_t<i_t, f_t, r_t>& sol,
                                                        bool full_set)
{
  raft::common::nvtx::range fun_scope("run_fast_search");
#if CUOPT_PROFILE_FAST_SEARCH
  const auto t_entry      = std::chrono::steady_clock::now();
  const double cost_entry = cuopt_fast_search_sync_and_cost(sol, move_candidates);
#endif

  std::vector<fast_operators_t> fast_operators{fast_operators_t::SLIDING, fast_operators_t::CROSS};
  if (false && !sol.problem_ptr->fleet_info.is_homogenous_ &&
      !sol.problem_ptr->has_non_uniform_breaks()) {
    fast_operators.push_back(fast_operators_t::REGRET);
  }

  std::shuffle(fast_operators.begin(), fast_operators.end(), rng);

  // In PDP mode we do not run the full set
  for (auto const& op : fast_operators) {
    switch (op) {
      case fast_operators_t::SLIDING: {
#if CUOPT_PROFILE_FAST_SEARCH
        const auto t0 = std::chrono::steady_clock::now();
#endif
        if (run_sliding_search(sol)) {
#if CUOPT_PROFILE_FAST_SEARCH
          const auto t1           = std::chrono::steady_clock::now();
          const double cost_after = cuopt_fast_search_sync_and_cost(sol, move_candidates);
          std::cout << "[run_fast_search PDP] " << fast_operator_label(op) << " wall=" << std::fixed
                    << std::setprecision(3)
                    << std::chrono::duration<double, std::milli>(t1 - t0).count()
                    << " ms cost_entry=" << std::setprecision(12) << cost_entry
                    << " cost_after=" << cost_after << " delta=" << (cost_after - cost_entry)
                    << std::defaultfloat << '\n';
          std::cout << "[run_fast_search PDP] total wall=" << std::fixed << std::setprecision(3)
                    << std::chrono::duration<double, std::milli>(t1 - t_entry).count()
                    << " ms (improved)\n"
                    << std::defaultfloat;
#endif
          return true;
        }
#if CUOPT_PROFILE_FAST_SEARCH
        {
          const auto t1 = std::chrono::steady_clock::now();
          std::cout << "[run_fast_search PDP] " << fast_operator_label(op) << " wall=" << std::fixed
                    << std::setprecision(3)
                    << std::chrono::duration<double, std::milli>(t1 - t0).count()
                    << " ms (no move)\n"
                    << std::defaultfloat;
        }
#endif
        break;
      }
      case fast_operators_t::VRP: {
        break;
      }
      case fast_operators_t::REGRET: {
#if CUOPT_PROFILE_FAST_SEARCH
        const auto t0 = std::chrono::steady_clock::now();
#endif
        if (run_vehicle_assignment<i_t, f_t, REQUEST>(sol, move_candidates, vehicle_assignment)) {
#if CUOPT_PROFILE_FAST_SEARCH
          const auto t1           = std::chrono::steady_clock::now();
          const double cost_after = cuopt_fast_search_sync_and_cost(sol, move_candidates);
          std::cout << "[run_fast_search PDP] " << fast_operator_label(op) << " wall=" << std::fixed
                    << std::setprecision(3)
                    << std::chrono::duration<double, std::milli>(t1 - t0).count()
                    << " ms cost_entry=" << std::setprecision(12) << cost_entry
                    << " cost_after=" << cost_after << " delta=" << (cost_after - cost_entry)
                    << std::defaultfloat << '\n';
          std::cout << "[run_fast_search PDP] total wall=" << std::fixed << std::setprecision(3)
                    << std::chrono::duration<double, std::milli>(t1 - t_entry).count()
                    << " ms (improved)\n"
                    << std::defaultfloat;
#endif
          return true;
        }
#if CUOPT_PROFILE_FAST_SEARCH
        {
          const auto t1 = std::chrono::steady_clock::now();
          std::cout << "[run_fast_search PDP] " << fast_operator_label(op) << " wall=" << std::fixed
                    << std::setprecision(3)
                    << std::chrono::duration<double, std::milli>(t1 - t0).count()
                    << " ms (no move)\n"
                    << std::defaultfloat;
        }
#endif
        break;
      }
      case fast_operators_t::TWO_OPT: {
        break;
      }
      case fast_operators_t::CROSS: {
#if CUOPT_PROFILE_FAST_SEARCH
        const auto t0 = std::chrono::steady_clock::now();
#endif
        if (run_cross_search(sol)) {
#if CUOPT_PROFILE_FAST_SEARCH
          const auto t1           = std::chrono::steady_clock::now();
          const double cost_after = cuopt_fast_search_sync_and_cost(sol, move_candidates);
          std::cout << "[run_fast_search PDP] " << fast_operator_label(op) << " wall=" << std::fixed
                    << std::setprecision(3)
                    << std::chrono::duration<double, std::milli>(t1 - t0).count()
                    << " ms cost_entry=" << std::setprecision(12) << cost_entry
                    << " cost_after=" << cost_after << " delta=" << (cost_after - cost_entry)
                    << std::defaultfloat << '\n';
          std::cout << "[run_fast_search PDP] total wall=" << std::fixed << std::setprecision(3)
                    << std::chrono::duration<double, std::milli>(t1 - t_entry).count()
                    << " ms (improved)\n"
                    << std::defaultfloat;
#endif
          return true;
        }
#if CUOPT_PROFILE_FAST_SEARCH
        {
          const auto t1 = std::chrono::steady_clock::now();
          std::cout << "[run_fast_search PDP] " << fast_operator_label(op) << " wall=" << std::fixed
                    << std::setprecision(3)
                    << std::chrono::duration<double, std::milli>(t1 - t0).count()
                    << " ms (no move)\n"
                    << std::defaultfloat;
        }
#endif
        break;
      }
    }
  }
#if CUOPT_PROFILE_FAST_SEARCH
  {
    const auto t_exit      = std::chrono::steady_clock::now();
    const double cost_exit = cuopt_fast_search_sync_and_cost(sol, move_candidates);
    std::cout << "[run_fast_search PDP] exit (no improvement) total_wall=" << std::fixed
              << std::setprecision(3)
              << std::chrono::duration<double, std::milli>(t_exit - t_entry).count()
              << " ms cost_entry=" << std::setprecision(12) << cost_entry
              << " cost_exit=" << cost_exit << " delta=" << (cost_exit - cost_entry)
              << std::defaultfloat << '\n';
  }
#endif
  return false;
}

template <typename i_t, typename f_t, request_t REQUEST>
bool local_search_t<i_t, f_t, REQUEST>::run_vrp_search_tier(solution_t<i_t, f_t, REQUEST>& sol,
                                                            bool full_set,
                                                            int tier)
{
  if constexpr (REQUEST != request_t::VRP) {
    (void)sol;
    (void)full_set;
    (void)tier;
    return false;
  } else {
    raft::common::nvtx::range fun_scope("run_vrp_search_tier");
    if (tier != 0 && tier != 1) { return false; }
    const bool tier_full_set = full_set;

    if (tier == 1 && sol.problem_ptr->is_tsp) {
      // Medium tier is empty for TSP (no VRP operator).
      return false;
    }

    std::vector<fast_operators_t> operators;
    if (tier == 0) {
      operators.push_back(fast_operators_t::TWO_OPT);
      operators.push_back(fast_operators_t::SLIDING);
      if (false && !sol.problem_ptr->fleet_info.is_homogenous_ &&
          !sol.problem_ptr->has_non_uniform_breaks()) {
        operators.push_back(fast_operators_t::REGRET);
      }
    } else {
      operators.push_back(fast_operators_t::VRP);
    }

    std::shuffle(operators.begin(), operators.end(), rng);

    auto& nodes_to_search = move_candidates.nodes_to_search;
    if (tier_full_set) {
      sol.set_routes_to_search();
      extract_nodes_to_search(sol, move_candidates);
    } else if (nodes_to_search.h_nodes_to_search.empty()) {
      // Fast tier sampling removes nodes from h_nodes_to_search; restore_found_nodes may not
      // refill enough before medium (VRP) runs — repopulate from the solution when empty.
      extract_nodes_to_search(sol, move_candidates);
    }
    if (!nodes_to_search.sample_nodes_to_search(sol, rng, tier_full_set)) {
      std::cout << "run_vrp_search_tier t=" << tier
                << " nodes_to_search.sample_nodes_to_search failed" << std::endl;
      exit(1);
      return false;
    }

#if CUOPT_PROFILE_FAST_SEARCH
    const auto t_entry      = std::chrono::steady_clock::now();
    const double cost_entry = cuopt_fast_search_sync_and_cost(sol, move_candidates);
#endif

    bool move_found = false;
    for (auto const& op : operators) {
#if CUOPT_PROFILE_FAST_SEARCH
      const auto t_op0 = std::chrono::steady_clock::now();
#endif
      switch (op) {
        case fast_operators_t::SLIDING: {
          move_found = run_sliding_search(sol) || move_found;
          break;
        }
        case fast_operators_t::VRP: {
          move_found = perform_vrp_search(sol, move_candidates) || move_found;
          break;
        }
        case fast_operators_t::REGRET: {
          move_found =
            run_vehicle_assignment<i_t, f_t, REQUEST>(sol, move_candidates, vehicle_assignment) ||
            move_found;
          break;
        }
        case fast_operators_t::TWO_OPT: {
          move_found = run_two_opt_search(sol) || move_found;
          break;
        }
        case fast_operators_t::CROSS: {
          break;
        }
      }
#if CUOPT_PROFILE_FAST_SEARCH
      {
        const auto t_op1           = std::chrono::steady_clock::now();
        const double cost_after_op = cuopt_fast_search_sync_and_cost(sol, move_candidates);
        std::cout << "[run_vrp_search_tier t=" << tier << "] op=" << fast_operator_label(op)
                  << " wall_op=" << std::fixed << std::setprecision(3)
                  << std::chrono::duration<double, std::milli>(t_op1 - t_op0).count()
                  << " ms cost=" << std::setprecision(12) << cost_after_op
                  << " (vs_entry_delta=" << (cost_after_op - cost_entry)
                  << ") move_found_accum=" << static_cast<int>(move_found) << std::defaultfloat
                  << '\n';
      }
#endif
    }

    move_candidates.nodes_to_search.restore_found_nodes(sol);
#if CUOPT_PROFILE_FAST_SEARCH
    {
      const auto t_exit      = std::chrono::steady_clock::now();
      const double cost_exit = cuopt_fast_search_sync_and_cost(sol, move_candidates);
      std::cout << "[run_vrp_search_tier t=" << tier
                << "] exit full_set=" << static_cast<int>(tier_full_set)
                << " move_found_accum=" << static_cast<int>(move_found)
                << " cost_entry=" << std::fixed << std::setprecision(12) << cost_entry
                << " cost_exit=" << cost_exit << " delta=" << (cost_exit - cost_entry)
                << " total_wall=" << std::setprecision(3)
                << std::chrono::duration<double, std::milli>(t_exit - t_entry).count() << " ms\n"
                << std::defaultfloat;
    }
#endif

    if (tier == 0) { return move_found; }
    if (tier == 1) { return move_found; }
    if (tier_full_set) { return move_found; }
    return true;
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
bool local_search_t<i_t, f_t, REQUEST>::try_negative_cycle_improvement(
  solution_t<i_t, f_t, REQUEST>& sol,
  bool should_all_nodes_be_served,
  bool run_cycle_finder_enabled)
{
  if (!run_cycle_finder_enabled || sol.n_routes > 1023) { return false; }
  if constexpr (REQUEST == request_t::VRP) {
    if (sol.n_routes < 2) { return false; }
  }

  move_candidates.reset(sol.sol_handle);
  calculate_route_compatibility(sol);
  find_insertions<i_t, f_t, REQUEST>(sol, move_candidates, search_type_t::IMPROVE);

  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
  sol.sol_handle->sync_stream();
  fill_gpu_graph(sol);

  move_candidates.find_best_negative_cycles(
    sol.n_routes, cycle_finder_small, cycle_finder_big, sol.sol_handle);
  [[maybe_unused]] double cost_before = 0., cost_after = 0.;
  cuopt_func_call(cost_before =
                    sol.get_cost(move_candidates.include_objective, move_candidates.weights));

  populate_move_path(sol, move_candidates);

  bool improved = move_candidates.move_path.n_insertions.value(sol.sol_handle->get_stream()) != 0;

  if (improved) {
    sol.unset_routes_to_search();
    perform_moves(sol, move_candidates);
    cuopt_func_call(sol.check_cost_coherence(move_candidates.weights));
    sol.global_runtime_checks(should_all_nodes_be_served, false, "run_best_local_search_end");
    cuopt_func_call(sol.compute_cost());
    cuopt_func_call(cost_after =
                      sol.get_cost(move_candidates.include_objective, move_candidates.weights));
    cuopt_assert((cost_after - cost_before) - move_candidates.cycles.total_cycle_cost < 1.,
                 "Cost mismatch after a move");
    sol.sol_handle->sync_stream();
  }

  return improved;
}

template <typename i_t, typename f_t, request_t REQUEST>
template <request_t r_t, std::enable_if_t<r_t == request_t::VRP, bool>>
bool local_search_t<i_t, f_t, REQUEST>::run_fast_search(solution_t<i_t, f_t, r_t>& sol,
                                                        bool full_set)
{
  return run_vrp_search_tier(sol, full_set, 0);
}

template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::run_best_local_search(solution_t<i_t, f_t, REQUEST>& sol,
                                                              const bool consider_unserviced,
                                                              const bool time_limit_enabled,
                                                              const bool run_cycle_finder)
{
  // Handle a corner case when there is no single task that is feasible
  if (sol.n_routes == 0) { return; }
  // for production use working weights
  move_candidates.selection_weights = move_candidates.weights;
  // for benchmarks use low random weights
  benchmark_call(move_candidates.set_random_selection_weights(rng));

  // Run regret heuristic upfront to get correct assignment of vehicles
  if (!sol.problem_ptr->fleet_info.is_homogenous_ && !sol.problem_ptr->has_non_uniform_breaks()) {
    run_vehicle_assignment(sol, move_candidates, vehicle_assignment);
  }

  i_t iter = 0;
  sol.sol_handle->sync_stream();
  sol.compute_cost();
  const i_t iter_limit = max_iterations;
  const bool should_all_nodes_be_served =
    consider_unserviced && !sol.problem_ptr->has_prize_collection();
  sol.global_runtime_checks(should_all_nodes_be_served, false, "run_best_local_search_begin");
  while (iter < iter_limit) {
    if constexpr (REQUEST == request_t::VRP) { extract_nodes_to_search(sol, move_candidates); }
    iter++;
    // Fast / medium / slow (negative-cycle) search. PDP: fast = existing operators; medium empty;
    // slow = try_negative_cycle_improvement.
    while (true) {
      if (time_limit_enabled && local_search_t<i_t, f_t, REQUEST>::check_time_limit()) { break; }
      iter++;
      if constexpr (REQUEST == request_t::VRP) {
        while (run_vrp_search_tier(sol, sol.problem_ptr->is_tsp && iter == 2, 0)) {
          if (time_limit_enabled && local_search_t<i_t, f_t, REQUEST>::check_time_limit()) {
            break;
          }
          iter++;
        }
        if (time_limit_enabled && local_search_t<i_t, f_t, REQUEST>::check_time_limit()) { break; }
        if (run_vrp_search_tier(sol, sol.problem_ptr->is_tsp && iter == 2, 1)) { continue; }
        if (try_negative_cycle_improvement(sol, should_all_nodes_be_served, run_cycle_finder)) {
          continue;
        }
      } else {
        while (run_fast_search(sol, sol.problem_ptr->is_tsp && iter == 2)) {
          if (time_limit_enabled && local_search_t<i_t, f_t, REQUEST>::check_time_limit()) {
            break;
          }
          iter++;
        }
        if (time_limit_enabled && local_search_t<i_t, f_t, REQUEST>::check_time_limit()) { break; }
        if (try_negative_cycle_improvement(sol, should_all_nodes_be_served, run_cycle_finder)) {
          continue;
        }
      }
      if (consider_unserviced && sol.problem_ptr->has_prize_collection() &&
          run_collect_prizes(sol)) {
        continue;
      }
      if (!sol.problem_ptr->special_nodes.is_empty() && perform_break_moves(sol)) { continue; }
      break;
    }

    sol.global_runtime_checks(
      should_all_nodes_be_served, false, "run_best_local_search_after_fast_search");

    if (time_limit_enabled && local_search_t<i_t, f_t, REQUEST>::check_time_limit()) {
      cuopt_func_call(sol.check_cost_coherence(move_candidates.weights));
      break;
    }
    break;
  }
  // reset it, so that next time all routes will be searched unless otherwise is specified
  sol.set_routes_to_search();
}

template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::run_random_local_search(solution_t<i_t, f_t, REQUEST>& sol,
                                                                bool time_limit_enabled)
{
  // Handle a corner case when there is no single task that is feasible
  if (sol.n_routes == 0) { return; }
  if (sol.n_routes > 1024) { return; }
  sol.sol_handle->sync_stream();

  // The weights for best local search must be set outside before calling this function
  set_active_weights(move_candidates.weights, move_candidates.include_objective);
  bool is_originally_feasible = sol.is_feasible();
  sol.global_runtime_checks(false, is_originally_feasible, "run_random_local_search_begin");

  move_candidates.reset(sol.sol_handle);
  move_candidates.random_move_candidates.reset(sol.sol_handle);
  calculate_route_compatibility(sol);
  find_insertions<i_t, f_t, REQUEST>(sol, move_candidates, search_type_t::RANDOM);

  RAFT_CHECK_CUDA(sol.sol_handle->get_stream());
  sol.sol_handle->sync_stream();
  populate_random_moves(sol);

  bool time_limit_reached =
    (time_limit_enabled && local_search_t<i_t, f_t, REQUEST>::check_time_limit());
  // if there is no more insertions found
  if (move_candidates.move_path.n_insertions.value(sol.sol_handle->get_stream()) == 0 ||
      time_limit_reached) {
    return;
  }
  perform_moves(sol, move_candidates);

  sol.global_runtime_checks(false, is_originally_feasible, "run_random_local_search_end");
  sol.sol_handle->sync_stream();

  set_active_weights(move_candidates.weights, move_candidates.include_objective);
}

template <typename i_t, typename f_t, request_t REQUEST>
void local_search_t<i_t, f_t, REQUEST>::perturb_solution(solution_t<i_t, f_t, REQUEST>& sol,
                                                         i_t perturb_count)
{
  if (perturb_count <= 0) {
    i_t min_count = 1;
    i_t max_count = 8;
    i_t n_routes  = std::max(1, sol.get_n_routes());
    perturb_count = std::max(min_count, std::min(100 / n_routes, max_count));
  }

  for (i_t i = 0; i < perturb_count; ++i) {
    run_random_local_search(sol, false);
  }
}

template class local_search_t<int, float, request_t::PDP>;
template class local_search_t<int, float, request_t::VRP>;

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
