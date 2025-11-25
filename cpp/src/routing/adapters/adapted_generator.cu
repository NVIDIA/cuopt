/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "adapted_generator.cuh"

#include "../ges/guided_ejection_search.cuh"

namespace cuopt::routing::detail {

// generator is a class can generate solutions (guided ejection search)
// TODO pass a GES object and keep it alive and reuse that GES object for different solutions
// keep a vector of GES objects for parallel working
template <typename i_t, typename f_t, request_t REQUEST>
adapted_generator_t<i_t, f_t, REQUEST>::adapted_generator_t(const problem_t<i_t, f_t>& problem_,
                                                            allocator& pool_allocator_)
  : problem(problem_), pool_allocator(pool_allocator_)
{
}

// make feasible ejects nodes until it is feasible, put them into EP
// then run GES to make it feasible
// if this can't make it feasible, we squeeze the request
template <typename i_t, typename f_t, request_t REQUEST>
bool adapted_generator_t<i_t, f_t, REQUEST>::make_feasible(
  adapted_sol_t<i_t, f_t, REQUEST>& adapted_solution,
  f_t time_limit,
  costs const& weight,
  bool clear_scores)
{
  raft::common::nvtx::range fun_scope("make_feasible");
  auto [resource, index] = pool_allocator.resource_pool->acquire();
  resource.ges.set_solution_ptr(&adapted_solution.sol, clear_scores);
  resource.ges.start_timer(std::chrono::steady_clock::now(), time_limit);
  auto gpu_weight = get_cuopt_cost(weight);
  resource.ls.set_active_weights(gpu_weight, std::numeric_limits<f_t>::max());
  // fprintf(
  //   f.file_ptr,
  //   "Number of infeasible routes: %d \n",
  //   adapted_solution.sol.n_infeasible_routes.value(adapted_solution.sol.sol_handle->get_stream()));
  constexpr bool add_slack_to_sol = true;
  adapted_solution.sol.eject_until_feasible(add_slack_to_sol);

  adapted_solution.sol.populate_ep_with_unserved(resource.ges.EP);
  // fprintf(f.file_ptr, "EP size after ejection: %d \n", resource.ges.EP.size());
  constexpr i_t perturbation_count = 1;
  for (i_t i = 0; i < perturbation_count; ++i) {
    resource.ls.run_random_local_search(adapted_solution.sol, false);
  }
  resource.ges.fixed_route_loop();

  // If the breaks are ejected, we need to squeeze them back
  resource.ges.try_squeeze_breaks_feasible();

  adapted_solution.populate_host_data(true);
  adapted_solution.check_device_host_coherence();
  cuopt_func_call(adapted_solution.sol.check_cost_coherence(gpu_weight));
  pool_allocator.resource_pool->release(index);
  return adapted_solution.sol.is_feasible();
}

template <typename i_t, typename f_t, request_t REQUEST>
void generate_tsp_solution(adapted_sol_t<i_t, f_t, REQUEST>& sol,
                           const std::vector<i_t>& desired_vehicle_ids)
{
  sol.clear_solution(desired_vehicle_ids);
  sol.sol.n_routes = 0;
  std::vector<NodeInfo<>> node_infos(sol.sol.get_num_depot_excluded_orders());
  for (i_t i = 0; i < (i_t)node_infos.size(); ++i) {
    node_infos[i] = sol.problem->get_node_info_of_node(i + sol.problem->order_info.depot_included_);
  }
  std::mt19937 rng(seed_generator::get_seed());
  std::shuffle(node_infos.begin(), node_infos.end(), rng);
  std::vector<std::pair<i_t, std::vector<NodeInfo<>>>> routes_to_add;
  routes_to_add.push_back({0, node_infos});
  sol.add_new_routes(routes_to_add);
}

// this generates a pool of solutions and returns a vector of solutions structure
// if feasible_only is false, we can squeeze the rest of the EP to the solution
template <typename i_t, typename f_t, request_t REQUEST>
void adapted_generator_t<i_t, f_t, REQUEST>::generate_solution(
  adapted_sol_t<i_t, f_t, REQUEST>& sol,
  const std::vector<i_t>& desired_vehicle_ids,
  f_t time_limit,
  const costs& weight,
  const timer_t& timer)
{
  raft::common::nvtx::range fun_scope("generate_solution");
  if (sol.problem->is_tsp) {
    generate_tsp_solution<i_t, f_t, REQUEST>(sol, desired_vehicle_ids);
    return;
  }

  f_t ges_time_limit       = timer.clamp_remaining_time(time_limit);
  auto [resource, index]   = pool_allocator.resource_pool->acquire();
  const auto start_time    = std::chrono::steady_clock::now();
  auto gpu_weight          = get_cuopt_cost(weight);
  bool run_route_minimizer = desired_vehicle_ids.empty();
  resource.ls.set_active_weights(gpu_weight, std::numeric_limits<f_t>::max());
  i_t n_routes =
    run_route_minimizer ? sol.sol.problem_ptr->get_fleet_size() : desired_vehicle_ids.size();

  resource.ges.set_solution_ptr(&sol.sol, true);
  resource.ges.start_timer(start_time, ges_time_limit);
  cuopt_assert(n_routes > 0, "Number of routes cannot be zero.");

  const auto& dim_info = sol.sol.problem_ptr->dimensions_info;

  if (run_route_minimizer) {
    resource.ges.construct_feasible_solution();
    resource.ges.route_minimizer_loop();
  } else {
    // FIXME:: We can do better
    sol.clear_solution(desired_vehicle_ids);
    sol.sol.random_init_routes();
    sol.sol.compute_initial_data();
    sol.sol.eject_until_feasible();
    resource.ges.init_ejection_pool();
    resource.ges.fixed_route_loop();
    if (dim_info.has_dimension(dim_t::BREAK)) { resource.ges.try_squeeze_breaks_feasible(); }
  }

  resource.ges.repair_empty_routes();

  sol.populate_host_data(true);
  cuopt_func_call(sol.check_device_host_coherence());
  cuopt_func_call(sol.sol.check_cost_coherence(gpu_weight));
  pool_allocator.resource_pool->release(index);
  pool_allocator.sync_all_streams();
}

template struct adapted_generator_t<int, float, request_t::PDP>;
template struct adapted_generator_t<int, float, request_t::VRP>;

}  // namespace cuopt::routing::detail
