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

#include <utilities/seed_generator.cuh>
#include "../diversity/helpers.hpp"
#include "../ges/guided_ejection_search.cuh"

#include "adapted_modifier.cuh"
#include "adapted_sol.cuh"

namespace cuopt::routing::detail {

template <typename i_t, typename f_t, request_t REQUEST>
adapted_modifier_t<i_t, f_t, REQUEST>::adapted_modifier_t(allocator& pool_allocator_)
  : pool_allocator(pool_allocator_),
    helper_nodes(pool_allocator_.problem.get_num_orders()),
    optimal_cycles(pool_allocator_)
{
}

template <typename i_t, typename f_t, request_t REQUEST>
void adapted_modifier_t<i_t, f_t, REQUEST>::perturbate(
  adapted_sol_t<i_t, f_t, REQUEST>& adapted_solution, costs weight, i_t perturbation_count)
{
  raft::common::nvtx::range fun_scope("perturbate");
  auto [resource, index] = pool_allocator.resource_pool->acquire();
  auto gpu_weight        = get_cuopt_cost(weight);
  // temporarily set it to double max. it is not used anymore
  // another PR removes it completely
  resource.ls.set_active_weights(gpu_weight, std::numeric_limits<double>::max());
  for (i_t i = 0; i < perturbation_count; ++i) {
    resource.ls.run_random_local_search(adapted_solution.sol, false);
  }
  adapted_solution.populate_host_data(true);
  adapted_solution.check_device_host_coherence();
  cuopt_func_call(adapted_solution.sol.check_cost_coherence(gpu_weight));
  pool_allocator.resource_pool->release(index);
}

template <typename i_t, typename f_t, request_t REQUEST>
void adapted_modifier_t<i_t, f_t, REQUEST>::improve(
  adapted_sol_t<i_t, f_t, REQUEST>& adapted_solution,
  costs weight,
  f_t time_limit,
  bool run_cycle_finder)
{
  raft::common::nvtx::range fun_scope("improve");
  auto [resource, index] = pool_allocator.resource_pool->acquire();
  // set the excess limit to to the total excess with some multiplier
  auto gpu_weight          = get_cuopt_cost(weight);
  bool consider_unserviced = true;
  bool time_limit_enabled  = true;

  resource.ls.set_active_weights(gpu_weight);
  resource.ls.start_timer(time_limit);
  resource.ls.run_best_local_search(
    adapted_solution.sol, consider_unserviced, time_limit_enabled, run_cycle_finder);
  adapted_solution.populate_host_data();
  adapted_solution.check_device_host_coherence();
  cuopt_func_call(adapted_solution.sol.check_cost_coherence(gpu_weight));
  pool_allocator.resource_pool->release(index);
}

// add unserviced pdp requests to the solution
template <typename i_t, typename f_t, request_t REQUEST>
void adapted_modifier_t<i_t, f_t, REQUEST>::add_unserviced_request(
  adapted_sol_t<i_t, f_t, REQUEST>& adapted_solution, costs final_weight)
{
  raft::common::nvtx::range fun_scope("add_unserviced_request");
  // find unserviced requests
  auto [resource, index] = pool_allocator.resource_pool->acquire();
  resource.ges.set_solution_ptr(&adapted_solution.sol);
  auto gpu_weight = get_cuopt_cost(final_weight);
  resource.ls.set_active_weights(gpu_weight, std::numeric_limits<f_t>::max());
  adapted_solution.sol.populate_ep_with_unserved(resource.ges.EP);
  resource.ges.EP.random_shuffle();
  resource.ges.squeeze_all_ep();
  adapted_solution.populate_host_data();
  adapted_solution.check_device_host_coherence();
  cuopt_func_call(adapted_solution.sol.check_cost_coherence(gpu_weight));
  pool_allocator.resource_pool->release(index);
  adapted_solution.sol.global_runtime_checks(true, false, "add_unserviced_request");
}

// add selected unserviced requests to the solution
template <typename i_t, typename f_t, request_t REQUEST>
void adapted_modifier_t<i_t, f_t, REQUEST>::add_selected_unserviced_requests(
  adapted_sol_t<i_t, f_t, REQUEST>& adapted_solution,
  const std::vector<i_t>& unserviced_nodes,
  costs final_weight)
{
  if (unserviced_nodes.empty()) { return; }
  raft::common::nvtx::range fun_scope("add_selected_unserviced_request");
  // find unserviced requests
  auto [resource, index] = pool_allocator.resource_pool->acquire();
  resource.ges.set_solution_ptr(&adapted_solution.sol);
  auto gpu_weight = get_cuopt_cost(final_weight);
  resource.ls.set_active_weights(gpu_weight, std::numeric_limits<f_t>::max());
  adapted_solution.sol.populate_ep_with_selected_unserved(resource.ges.EP, unserviced_nodes);
  resource.ges.EP.random_shuffle();
  resource.ges.squeeze_all_ep();
  adapted_solution.populate_host_data();
  adapted_solution.check_device_host_coherence();
  cuopt_func_call(adapted_solution.sol.check_cost_coherence(gpu_weight));
  pool_allocator.resource_pool->release(index);
  adapted_solution.sol.global_runtime_checks(false, false, "add_selected_unserviced_request");
}

/*! \brief { Prepares both solutions for EAX.
 * Make sure that both solutions serve the same nodes and have equal number of
 * routes which is an algorithmic requirement for running EAX. } */
template <typename i_t, typename f_t, request_t REQUEST>
void adapted_modifier_t<i_t, f_t, REQUEST>::equalize_routes_and_nodes(
  adapted_sol_t<i_t, f_t, REQUEST>& sol_a,
  adapted_sol_t<i_t, f_t, REQUEST>& sol_b,
  costs final_weight,
  bool skip_adding_nodes_to_a)
{
  raft::common::nvtx::range fun_scope("equalize_routes_and_nodes");

  // If both solutions have no unserviced nodes (default behavior)
  // and if they have equal route count we should exit early
  if ((sol_a.sol.get_n_routes() == sol_b.sol.get_n_routes()) && !sol_a.has_unserviced_nodes &&
      !sol_b.has_unserviced_nodes) {
    return;
  }

  std::vector<int> missing_in_a, missing_in_b;
  auto& nodes_a = sol_a.nodes;
  auto& nodes_b = sol_b.nodes;
  missing_in_a.reserve(nodes_a.size());
  missing_in_b.reserve(nodes_b.size());
  for (size_t i = 0; i < nodes_a.size(); ++i) {
    bool a_unrouted = nodes_a[i].r_id == std::numeric_limits<size_t>::max();
    bool b_unrouted = nodes_b[i].r_id == std::numeric_limits<size_t>::max();
    if (a_unrouted && !b_unrouted) {
      missing_in_a.push_back(i);
    } else if (!a_unrouted && b_unrouted) {
      missing_in_b.push_back(i);
    }
  }

  if (!skip_adding_nodes_to_a && sol_a.sol.get_n_routes() > sol_b.sol.get_n_routes()) {
    auto removed_nodes = sol_a.priority_remove_diff_routes(sol_b);
    missing_in_a.insert(missing_in_a.end(), removed_nodes.begin(), removed_nodes.end());
  } else if (sol_b.sol.get_n_routes() > sol_a.sol.get_n_routes()) {
    auto removed_nodes = sol_b.priority_remove_diff_routes(sol_a);
    missing_in_b.insert(missing_in_b.end(), removed_nodes.begin(), removed_nodes.end());
  }

  // If there are no mutually exclusive requests, take an early exit
  if (missing_in_a.empty() && missing_in_b.empty()) { return; }

  if (!skip_adding_nodes_to_a) {
    add_selected_unserviced_requests(sol_a, missing_in_a, final_weight);
  }
  add_selected_unserviced_requests(sol_b, missing_in_b, final_weight);
}

/*! \brief { Greedily remove Cluster/Order infeasiblity. It may introduce time/cap infesibility and
 * reinsert brother nodes to the solution. } */
template <typename i_t, typename f_t, request_t REQUEST>
bool adapted_modifier_t<i_t, f_t, REQUEST>::eject_request_infeasible_nodes(
  adapted_sol_t<i_t, f_t, REQUEST>& sol)
{
  raft::common::nvtx::range fun_scope("eject_request_infeasible_nodes");
  helper_set.clear();
  helper_nodes.clear();
  for (int i = 0; i < (int)sol.problem->get_num_orders(); i++) {
    NodeInfo<> node = sol.problem->get_node_info_of_node(i);
    if (node.is_depot()) { continue; }
    if (helper_set.count(node) != 0) { continue; }
    if (sol.unserviced(node.node())) { continue; }

    auto to_remove     = node;
    NodeInfo<> brother = sol.problem->get_brother_node_info(node);
    if (helper_set.count(brother) != 0) continue;
    // If the cluster/order is correct continue
    if (sol.nodes[node.node()].r_id == sol.nodes[brother.node()].r_id) {
      if (node.is_pickup() && sol.nodes[node.node()].r_index < sol.nodes[brother.node()].r_index)
        continue;
      if (!node.is_pickup() && sol.nodes[node.node()].r_index > sol.nodes[brother.node()].r_index)
        continue;
    }

    to_remove = (next_random() % 2) ? node : brother;
    if (abs(sol.routes[sol.nodes[node.node()].r_id].length -
            sol.routes[sol.nodes[brother.node()].r_id].length) > 10) {
      to_remove = (sol.routes[sol.nodes[node.node()].r_id].length <
                   sol.routes[sol.nodes[brother.node()].r_id].length)
                    ? brother
                    : node;
    }
    cuopt_assert(!sol.unserviced(to_remove.node()), "Node to remove should be routed");
    helper_set.insert(to_remove);
    helper_nodes.push_back(to_remove);
  }
  // boolean value checks whether we ended up with an empty route
  bool remove_success = sol.remove_nodes(helper_nodes);
  if (remove_success) {
    // Reinsert remaining nodes without a pair (assuming that brother node is already included)
    std::shuffle(helper_nodes.begin(), helper_nodes.end(), next_random_object());
    return true;
  }
  return false;
}

/*! \brief { Greedily remove Cluster/Order infeasiblity. It may introduce time/cap infesibility and
 * reinsert brother nodes to the solution. } */
template <typename i_t, typename f_t, request_t REQUEST>
void adapted_modifier_t<i_t, f_t, REQUEST>::insert_infeasible_nodes(
  adapted_sol_t<i_t, f_t, REQUEST>& sol, costs& weights)
{
  raft::common::nvtx::range fun_scope("insert_infeasible_nodes");
  // add cluster/order infeasible nodes to the solution
  sol.add_nodes_to_best(helper_nodes, weights);
}

template <typename i_t, typename f_t, request_t REQUEST>
bool adapted_modifier_t<i_t, f_t, REQUEST>::make_cluster_order_feasible_request(
  adapted_sol_t<i_t, f_t, REQUEST>& sol, costs weights)
{
  raft::common::nvtx::range fun_scope("make_cluster_order_feasible_request");
  if constexpr (REQUEST == request_t::VRP) { return true; }
  if (eject_request_infeasible_nodes(sol)) {
    insert_infeasible_nodes(sol, weights);
    return true;
  }
  return false;
}

/*! \brief { Add all cycles to the solution. Every cycle may be rotated during the procedure. We
 * find optimal route to satisfy clustering requirement, then optimal cycle rotation to satisfy
 * order requirement, then insert to optimal place in the route with respect to (order_violation,
 * distance) requirement. } */
template <typename i_t, typename f_t, request_t REQUEST>
bool adapted_modifier_t<i_t, f_t, REQUEST>::add_cycles_request(
  adapted_sol_t<i_t, f_t, REQUEST>& a,
  std::vector<std::vector<NodeInfo<>>>& cycles,
  costs final_weight)
{
  return optimal_cycles.add_cycles_request(a, cycles, final_weight);
}

template <typename i_t, typename f_t, request_t REQUEST>
void adapted_modifier_t<i_t, f_t, REQUEST>::squeeze_breaks(adapted_sol_t<i_t, f_t, REQUEST>& sol,
                                                           costs& weights)
{
  auto [resource, index] = pool_allocator.resource_pool->acquire();
  resource.ges.set_solution_ptr(&sol.sol);
  auto gpu_weight = get_cuopt_cost(weights);
  resource.ls.set_active_weights(gpu_weight, std::numeric_limits<f_t>::max());
  resource.ges.squeeze_breaks();

  sol.populate_host_data();
  sol.check_device_host_coherence();
  cuopt_func_call(sol.sol.check_cost_coherence(gpu_weight));

  pool_allocator.resource_pool->release(index);
  auto all_nodes_should_be_served = !sol.problem->has_prize_collection();
  sol.sol.global_runtime_checks(all_nodes_should_be_served, false, "squeeze_breaks");
}

template struct adapted_modifier_t<int, float, request_t::PDP>;
template struct adapted_modifier_t<int, float, request_t::VRP>;

}  // namespace cuopt::routing::detail
