/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../ges/found_solution.cuh"
#include "../node/node.cuh"
#include "../optional.cuh"
#include "../solution/solution.cuh"
#include "compute_insertions.cuh"
#include "move_candidates/move_candidates.cuh"

#include <raft/random/rng_device.cuh>

#include <cstdint>

namespace cuopt {
namespace routing {
namespace detail {

enum class pick_mode_t { COST_DELTA, PROBABILITY };

// keep this function in this file as this is included/used in many local search files
template <typename i_t, typename f_t, request_t REQUEST, search_type_t search_type>
DI double get_excess_limit(const typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                           const typename move_candidates_t<i_t, f_t>::view_t& move_candidates,
                           i_t route_id,
                           double excess_curr_route,
                           i_t route_id_of_inserted)
{
  // compute excess limit of current route (whether same of different route)
  double excess_limit;
  excess_limit = excess_curr_route;
  if (route_id_of_inserted >= 0 && route_id_of_inserted != route_id) {
    auto& other_route = solution.routes[route_id_of_inserted];
    excess_limit += other_route.get_weighted_excess(move_candidates.weights);
  }
  excess_limit = excess_limit * ls_excess_multiplier_route;
  return excess_limit;
}

template <typename i_t, typename f_t>
DI void update_random_cand(raft::random::PCGenerator* rng,
                           cand_t* feasible_move,
                           i_t pickup_insertion,
                           i_t delivery_insertion)
{
  auto counter = feasible_move->cost_counter.counter;
  if (rng->next_u32() % counter == 0) {
    *feasible_move = move_candidates_t<i_t, f_t>::make_candidate(pickup_insertion,
                                                                 delivery_insertion,
                                                                 std::numeric_limits<short>::max(),
                                                                 std::numeric_limits<short>::max(),
                                                                 0);  // we will fill counter later
  }
  feasible_move->cost_counter.counter = counter + 1;
}

template <typename i_t, typename f_t>
DI void update_best_cand(double delta,
                         cand_t* feasible_move,
                         i_t pickup_insertion,
                         i_t delivery_insertion)
{
  if (delta < feasible_move->cost_counter.cost) {
    *feasible_move = move_candidates_t<i_t, f_t>::make_candidate(pickup_insertion,
                                                                 delivery_insertion,
                                                                 std::numeric_limits<short>::max(),
                                                                 std::numeric_limits<short>::max(),
                                                                 delta);
  }
}

template <typename i_t, typename f_t, request_t REQUEST, pick_mode_t pick_mode>
DI void find_delivery_insertion(typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                                const node_t<i_t, f_t, REQUEST>& pickup_node,
                                node_t<i_t, f_t, REQUEST>& delivery_node,
                                i_t pickup_insertion,
                                const typename route_t<i_t, f_t, REQUEST>::view_t& route,
                                bool include_objective,
                                infeasible_cost_t const& weights,
                                double excess_limit,
                                cand_t* feasible_move                           = nullptr,
                                [[maybe_unused]] raft::random::PCGenerator* rng = nullptr,
                                bool check_single_location                      = false)
{
  const i_t n_nodes_route = route.get_num_nodes();
  const i_t route_id      = route.get_id();

  // Combine 2 fragments
  node_t<i_t, f_t, REQUEST> last =
    check_single_location ? route.get_node(pickup_insertion) : pickup_node;
  node_t<i_t, f_t, REQUEST> current = route.get_node(pickup_insertion + 1);
  // we need to get he original routes cost here.
  // calculate_forward_all will return the difference of new
  // (ejected/inserted route) - (original_route)
  const auto old_objective_cost    = solution.routes[route_id].get_objective_cost();
  const auto old_infeasbility_cost = solution.routes[route_id].get_infeasibility_cost();

  // Insert delivery after pickup
  last.calculate_forward_all(delivery_node, route.vehicle_info());
  // Forward time filtration
  if (!delivery_node.time_dim.forward_feasible(
        route.vehicle_info(), weights[dim_t::TIME], excess_limit)) {
    return;
  }

  bool fragment_combine_feasible = node_t<i_t, f_t, REQUEST>::combine(
    delivery_node, current, route.vehicle_info(), weights, excess_limit);

  if (fragment_combine_feasible) {
    // Feasible insertion found
    if constexpr (pick_mode == pick_mode_t::COST_DELTA) {
      double insertion_cost_delta =
        delivery_node.calculate_forward_all_and_delta(current,
                                                      route.vehicle_info(),
                                                      include_objective,
                                                      weights,
                                                      old_objective_cost,
                                                      old_infeasbility_cost);
      // atomically update the feasible_move for this request
      update_best_cand<i_t, f_t>(
        insertion_cost_delta, feasible_move, pickup_insertion, pickup_insertion);
    } else if constexpr (pick_mode == pick_mode_t::PROBABILITY) {
      update_random_cand<i_t, f_t>(rng, feasible_move, pickup_insertion, pickup_insertion);
    }
  }

  if (check_single_location) { return; }

  for (int i = pickup_insertion + 1; i < n_nodes_route; ++i) {
    // Insert current after last
    last.calculate_forward_all(current, route.vehicle_info());
    if (!current.forward_feasible(route.vehicle_info(), weights, excess_limit)) { return; }

    // Insert delivery_node after current
    current.calculate_forward_all(delivery_node, route.vehicle_info());
    if (!delivery_node.time_dim.forward_feasible(
          route.vehicle_info(), weights[dim_t::TIME], excess_limit) ||
        delivery_node.time_dim.excess_forward > current.time_dim.excess_forward) {
      return;
    }

    // Move forward
    last    = current;
    current = route.get_node(i + 1);

    fragment_combine_feasible = node_t<i_t, f_t, REQUEST>::combine(
      delivery_node, current, route.vehicle_info(), weights, excess_limit);

    // Combine 2 fragments
    if (fragment_combine_feasible) {
      if constexpr (pick_mode == pick_mode_t::COST_DELTA) {
        double insertion_cost_delta =
          delivery_node.calculate_forward_all_and_delta(current,
                                                        route.vehicle_info(),
                                                        include_objective,
                                                        weights,
                                                        old_objective_cost,
                                                        old_infeasbility_cost);
        // atomically update the feasible_move for this request
        update_best_cand<i_t, f_t>(insertion_cost_delta, feasible_move, pickup_insertion, i);
      } else if constexpr (pick_mode == pick_mode_t::PROBABILITY) {
        update_random_cand<i_t, f_t>(rng, feasible_move, pickup_insertion, i);
      }
    }
  }
}

template <typename i_t, typename f_t, request_t REQUEST, pick_mode_t pick_mode>
DI void find_pickup_insertion(typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                              node_t<i_t, f_t, REQUEST>& pickup_node,
                              const node_t<i_t, f_t, REQUEST>& delivery_node,
                              i_t delivery_insertion,
                              const typename route_t<i_t, f_t, REQUEST>::view_t& route,
                              bool include_objective,
                              infeasible_cost_t const& weights,
                              double excess_limit,
                              cand_t* feasible_move                           = nullptr,
                              [[maybe_unused]] raft::random::PCGenerator* rng = nullptr,
                              bool check_single_location                      = false)
{
  const i_t route_id = route.get_id();

  node_t<i_t, f_t, REQUEST> last =
    check_single_location ? route.get_node(delivery_insertion + 1) : delivery_node;
  node_t<i_t, f_t, REQUEST> current = route.get_node(delivery_insertion);

  // we need to get he original routes cost here.
  // calculate_forward_all will return the difference of new
  // (ejected/inserted route) - (original_route)
  const auto old_objective_cost    = solution.routes[route_id].get_objective_cost();
  const auto old_infeasbility_cost = solution.routes[route_id].get_infeasibility_cost();

  // Insert delivery after pickup
  last.calculate_backward_all(pickup_node, route.vehicle_info());
  // Forward time filtration
  if (!pickup_node.time_dim.backward_feasible(
        route.vehicle_info(), weights[dim_t::TIME], excess_limit)) {
    return;
  }

  bool fragment_combine_feasible = node_t<i_t, f_t, REQUEST>::combine(
    current, pickup_node, route.vehicle_info(), weights, excess_limit);
  if (fragment_combine_feasible) {
    // Feasible insertion found
    if constexpr (pick_mode == pick_mode_t::COST_DELTA) {
      double insertion_cost_delta =
        pickup_node.calculate_backward_all_and_delta(current,
                                                     route.vehicle_info(),
                                                     include_objective,
                                                     weights,
                                                     old_objective_cost,
                                                     old_infeasbility_cost);
      update_best_cand<i_t, f_t>(
        insertion_cost_delta, feasible_move, delivery_insertion, delivery_insertion);
    } else if constexpr (pick_mode == pick_mode_t::PROBABILITY) {
      update_random_cand<i_t, f_t>(rng, feasible_move, delivery_insertion, delivery_insertion);
    }
  }

  if (check_single_location) { return; }

  for (int i = delivery_insertion; i > 0; --i) {
    // Insert current before last
    last.calculate_backward_all(current, route.vehicle_info());
    if (!current.backward_feasible(route.vehicle_info(), weights, excess_limit)) { return; }

    // Insert pickup_node before current
    current.calculate_backward_all(pickup_node, route.vehicle_info());

    if (!pickup_node.time_dim.backward_feasible(
          route.vehicle_info(), weights[dim_t::TIME], excess_limit) ||
        pickup_node.time_dim.excess_backward > current.time_dim.excess_backward) {
      return;
    }

    // Move backward
    last    = current;
    current = route.get_node(i - 1);

    fragment_combine_feasible = node_t<i_t, f_t, REQUEST>::combine(
      current, pickup_node, route.vehicle_info(), weights, excess_limit);

    // Combine 2 fragments
    if (fragment_combine_feasible) {
      if constexpr (pick_mode == pick_mode_t::COST_DELTA) {
        double insertion_cost_delta =
          pickup_node.calculate_backward_all_and_delta(current,
                                                       route.vehicle_info(),
                                                       include_objective,
                                                       weights,
                                                       old_objective_cost,
                                                       old_infeasbility_cost);
        update_best_cand<i_t, f_t>(insertion_cost_delta, feasible_move, i - 1, delivery_insertion);
      } else if constexpr (pick_mode == pick_mode_t::PROBABILITY) {
        update_random_cand<i_t, f_t>(rng, feasible_move, i - 1, delivery_insertion);
      }
    }
  }
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          pick_mode_t pick_mode,
          bool is_delivery,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
DI void find_brother_insertion(typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                               request_node_t<i_t, f_t, REQUEST>& request_node,
                               i_t node_insertion_idx,
                               const typename route_t<i_t, f_t, REQUEST>::view_t& curr_route,
                               bool include_objective,
                               infeasible_cost_t const& weights,
                               double excess_limit,
                               cand_t* feasible_move                           = nullptr,
                               [[maybe_unused]] raft::random::PCGenerator* rng = nullptr)
{
  auto node = request_node.node();
  // Combine 2 fragments
  node_t<i_t, f_t, REQUEST> current = curr_route.get_node(node_insertion_idx + 1);
  bool fragment_combine_feasible    = node_t<i_t, f_t, REQUEST>::combine(
    node, current, curr_route.vehicle_info(), weights, excess_limit);

  const i_t route_id = curr_route.get_id();

  const auto old_objective_cost     = solution.routes[route_id].get_objective_cost();
  const auto old_infeasibility_cost = solution.routes[route_id].get_infeasibility_cost();
  if (fragment_combine_feasible) {
    // Feasible insertion found
    if constexpr (pick_mode == pick_mode_t::COST_DELTA) {
      double insertion_cost_delta = node.calculate_forward_all_and_delta(current,
                                                                         curr_route.vehicle_info(),
                                                                         include_objective,
                                                                         weights,
                                                                         old_objective_cost,
                                                                         old_infeasibility_cost);
      // atomically update the feasible_move for this request
      update_best_cand<i_t, f_t>(
        insertion_cost_delta, feasible_move, node_insertion_idx, node_insertion_idx);
    } else if constexpr (pick_mode == pick_mode_t::PROBABILITY) {
      update_random_cand<i_t, f_t>(rng, feasible_move, node_insertion_idx, node_insertion_idx);
    }
  }
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          pick_mode_t pick_mode,
          bool is_delivery,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
DI void find_brother_insertion(typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                               request_node_t<i_t, f_t, REQUEST>& request_node,
                               i_t node_insertion,
                               const typename route_t<i_t, f_t, REQUEST>::view_t& route,
                               bool include_objective,
                               infeasible_cost_t weights,
                               double excess_limit,
                               cand_t* feasible_move                           = nullptr,
                               [[maybe_unused]] raft::random::PCGenerator* rng = nullptr,
                               bool check_single_location                      = false)
{
  cuopt_assert(solution.problem.order_info.is_pickup_index[request_node.pickup.id()],
               "Not a pickup idx!");
  cuopt_assert(!solution.problem.order_info.is_pickup_index[request_node.delivery.id()],
               "Not a delivery idx!");
  if constexpr (REQUEST == request_t::PDP) {
    cuopt_assert(request_node.pickup.id() == request_node.delivery.request.brother_info.node(),
                 "Pair mismatch!");
    cuopt_assert(request_node.delivery.id() == request_node.pickup.request.brother_info.node(),
                 "Pair mismatch!");
  }
  if constexpr (is_delivery) {
    find_delivery_insertion<i_t, f_t, REQUEST, pick_mode>(solution,
                                                          request_node.pickup,
                                                          request_node.delivery,
                                                          node_insertion,
                                                          route,
                                                          include_objective,
                                                          weights,
                                                          excess_limit,
                                                          feasible_move,
                                                          rng,
                                                          check_single_location);
  } else {
    find_pickup_insertion<i_t, f_t, REQUEST, pick_mode>(solution,
                                                        request_node.pickup,
                                                        request_node.delivery,
                                                        node_insertion,
                                                        route,
                                                        include_objective,
                                                        weights,
                                                        excess_limit,
                                                        feasible_move,
                                                        rng,
                                                        check_single_location);
  }
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
