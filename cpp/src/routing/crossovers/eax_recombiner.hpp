/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <routing/structures.hpp>

#include "../diversity/helpers.hpp"
#include "ab_cycle.hpp"

#include <raft/common/nvtx.hpp>

#include <climits>
#include <cmath>
#include <optional>
#include <unordered_set>

constexpr int max_eax_cycle_length = 64;

namespace cuopt {
namespace routing {

// #define EAX_DIAGNOSTICS

/*! \brief { Recombine two solutions. One of input solutions may be overwritten and child may
 * contain lower number of nodes. } */
template <class Solution>
struct a_eax {
  ESET_graph eset;
  std::vector<std::vector<detail::NodeInfo<>>> cycles;

  std::vector<std::vector<detail::NodeInfo<>>> ab_graph;
  std::unordered_set<detail::NodeInfo<>, detail::NodeInfoHash> route_start_ids;

  std::vector<detail::NodeInfo<>> route_helper;
  std::vector<detail::NodeInfo<>> succesor_cpy;
  std::vector<ESET_graph::edge> cycle;

  std::optional<detail::NodeInfo<>> single_depot_node;

  a_eax(size_t nodes_number) : eset(nodes_number), succesor_cpy(nodes_number, detail::NodeInfo<>())
  {
    ab_graph.assign(nodes_number, std::vector<detail::NodeInfo<>>());
    cycle.reserve(1000);
  }

  /*! \brief { Recombine solutions a and b. The output will be stored in a (it will be overwritten)
   * } */
  bool recombine(Solution& a,
                 const Solution& b,
                 bool asymmetric              = true,
                 bool use_perfect_edges_limit = true)
  {
    raft::common::nvtx::range fun_scope("eax");
    // THIS IS TEMPORARY UNTIL EMPTY ROUTES ARE FIXED
    if (check_if_routes_empty(a) || check_if_routes_empty(b)) return false;
    single_depot_node = a.problem->get_single_depot();

    // different routes from a & b > 1. Set min routes
    if (a.routes.size() != b.routes.size() || !single_depot_node.has_value()) return false;

    eset.clear();
    fill_eset(a, b);
    if (eset.total_edges == 0) return false;

#ifdef EAX_DIAGNOSTICS
    printf("Eset edges: %d\n", eset.total_edges);
#endif

    std::vector<ESET_graph::edge> backup;
    const size_t max_size    = 128;
    int perfect_edges_number = eset.total_edges / 2;
    if (use_perfect_edges_limit) {
      perfect_edges_number = std::min<int>(eset.total_edges / 2, max_eax_cycle_length);
    }

    while (eset.find_cycle(asymmetric)) {
      if (eset.helper.size() <= max_size) {
        if (std::fabs((int)backup.size() - perfect_edges_number) >
            std::fabs((int)eset.helper.size() - perfect_edges_number)) {
          backup = eset.helper;
        }
      }
    }

    if (backup.empty()) { return false; }

    eset.helper = backup;

#ifdef EAX_DIAGNOSTICS
    printf("Cycle found : %lu\n", eset.helper.size());
#endif

    cycles.clear();
    populate_helper_to_graph(a, asymmetric);
    ab_graph_to_solution(a, b, asymmetric);

    return true;
  }

  void reset_ab()
  {
    for (auto& a : ab_graph)
      a.clear();
  }

  void fill_eset(Solution& a, const Solution& b)
  {
    // from_sol = 0 <=> solution = a
    // from_sol = 1 <=> solution = b
    eset.clear();
    auto& a00 = eset.eset[0][0];
    auto& a01 = eset.eset[0][1];
    auto& a10 = eset.eset[1][0];
    auto& a11 = eset.eset[1][1];
    for (auto& route : a.routes) {
      detail::NodeInfo<> start = route.start;
      cuopt_expects(!route.is_empty(), error_type_t::ValidationError, "Route cannot be empty");
      while (!start.is_depot()) {
        if (b.pred[start.node()] != a.pred[start.node()]) {
          a01[a.pred[start.node()].node()].emplace_back(a.pred[start.node()], start, 0);
          a00[start.node()].emplace_back(a.pred[start.node()], start, 0);
        }
        start = a.succ[start.node()];
      }
      detail::NodeInfo<> end = route.end;
      // record last endge of the route
      if (b.succ[end.node()] != a.succ[end.node()]) {
        a01[end.node()].emplace_back(end, a.succ[end.node()], 0);
        a00[a.succ[end.node()].node()].emplace_back(end, a.succ[end.node()], 0);
      }
    }

    for (auto& route : b.routes) {
      cuopt_expects(!route.is_empty(), error_type_t::ValidationError, "Route cannot be empty");
      auto start = route.start;
      while (!start.is_depot()) {
        if (a.pred[start.node()] != b.pred[start.node()]) {
          a11[b.pred[start.node()].node()].emplace_back(b.pred[start.node()], start, 1);
          a10[start.node()].emplace_back(b.pred[start.node()], start, 1);
        }
        start = b.succ[start.node()];
      }
      auto end = route.end;
      // record last edge of the route
      if (b.succ[end.node()] != a.succ[end.node()]) {
        a11[end.node()].emplace_back(end, b.succ[end.node()], 1);
        a10[b.succ[end.node()].node()].emplace_back(end, b.succ[end.node()], 1);
      }
    }

    eset.calculate_edges();
  }

  void populate_helper_to_graph(Solution& a, bool asymmetric)
  {
    reset_ab();
    route_start_ids.clear();
    // Fill start nodes of routes affected by the recombination
    for (auto& edge : eset.helper) {
      if (!edge.from.is_depot()) {
        int route_id = a.nodes[edge.from.node()].r_id;
        cuopt_assert(route_id >= 0 && route_id < a.routes.size(), "route id should be in range");
        route_start_ids.insert(a.routes[route_id].start);
      }
      if (!edge.to.is_depot()) {
        int route_id = a.nodes[edge.to.node()].r_id;
        cuopt_assert(route_id >= 0 && route_id < a.routes.size(), "route id should be in range");
        route_start_ids.insert(a.routes[route_id].start);
      }
    }

    // Fill the eset structure with edges from affected routes from a
    for (auto start : route_start_ids) {
      detail::NodeInfo<> node = a.pred[start.node()];
      ab_graph[node.node()].push_back(start);
      if (!asymmetric) ab_graph[start.node()].push_back(node);
      while (!start.is_depot()) {
        node = a.succ[start.node()];
        ab_graph[start.node()].push_back(node);
        if (!asymmetric) ab_graph[node.node()].push_back(start);
        start = node;
      }
    }

    // Fill the eset structure with edges from the cycle that belong to solution b and erase those
    // from a
    for (auto& edge : eset.helper) {
      // from_sol = 1 <=> solution = b
      if (edge.from_sol) {
        ab_graph[edge.from.node()].push_back(edge.to);
        if (!asymmetric) ab_graph[edge.to.node()].push_back(edge.from);
      } else {
        find_and_pop(ab_graph[edge.from.node()], edge.to);

        if (!asymmetric) find_and_pop(ab_graph[edge.to.node()], edge.from);
      }
    }

    // Now ab_graph contains directed or undirected cycles that can be inserted to a solution. Those
    // containing depo will be inserted as routes
  }

  void ab_graph_to_solution(Solution& a,
                            const Solution& b,
                            bool asymmetric,
                            bool majority_vote = true)
  {
    if (!asymmetric && majority_vote) succesor_cpy = a.succ;

    std::vector<int> routes_to_remove;
    std::vector<int> removed_vehicle_ids;

    // remove affected routes from a
    for (auto node : route_start_ids) {
      cuopt_assert(!node.is_break(), "route should not be starting with break nodes");
      auto id         = a.nodes[node.node()].r_id;
      auto vehicle_id = a.nodes[node.node()].v_id;
      routes_to_remove.push_back(id);
      removed_vehicle_ids.push_back(vehicle_id);
    }
    a.remove_routes(routes_to_remove);

    std::vector<std::pair<int, std::vector<detail::NodeInfo<>>>> routes_to_add;
    // insert routes from ab_graph to solution and pop from ab_graph structure
    while (!ab_graph[single_depot_node.value().node()].empty()) {
      route_helper.clear();
      detail::NodeInfo<> prev_node = single_depot_node.value();
      detail::NodeInfo<> node      = pop_random(ab_graph[single_depot_node.value().node()]);

      assert(!removed_vehicle_ids.empty());
      // FIXME:: use heuristics to figure out which vehicle to use
      int vehicle_id = pop_random(removed_vehicle_ids);
      if (!asymmetric) { find_and_pop(ab_graph[node.node()], prev_node); }
      int no_revert = 0, revert = 0;
      while (!node.is_depot()) {
        route_helper.push_back(node);
        prev_node = node;
        node      = pop_random(ab_graph[node.node()]);
        if (!asymmetric && majority_vote) {
          if (succesor_cpy[prev_node.node()] == node || b.succ[prev_node.node()] == node)
            no_revert++;
          else
            revert++;
        }

        if (!asymmetric) find_and_pop(ab_graph[node.node()], prev_node);
      }

      if (!asymmetric && majority_vote && revert > no_revert) {
        std::reverse(route_helper.begin(), route_helper.end());
#ifdef EAX_DIAGNOSTICS
        printf(" Route reversed by majority vote: %d %d \n", revert, no_revert);
#endif
      }
      routes_to_add.push_back({vehicle_id, route_helper});
    }

    assert(routes_to_add.size() == routes_to_remove.size());
    a.add_new_routes(routes_to_add);
    // identify and insert cycles if some are left
    for (int i = 0; i < (int)ab_graph.size(); i++) {
      if (!ab_graph[i].empty()) {
        route_helper.clear();
        detail::NodeInfo<> node_i    = a.problem->get_node_info_of_node(i);
        detail::NodeInfo<> prev_node = a.problem->get_node_info_of_node(i);

        route_helper.push_back(prev_node);
        detail::NodeInfo<> node = pop_random(ab_graph[i]);
        if (!asymmetric) find_and_pop(ab_graph[node.node()], prev_node);

        while (node != node_i) {
          route_helper.push_back(node);
          prev_node = node;
          node      = pop_random(ab_graph[node.node()]);

          if (!asymmetric) find_and_pop(ab_graph[node.node()], prev_node);
        }
        cycles.push_back(route_helper);
      }
    }
#ifdef EAX_DIAGNOSTICS
    printf("cycle lengths: ");
    for (size_t i = 0; i < cycles.size(); ++i) {
      printf("%lu\t", cycles[i].size());
    }
    printf("\n");
#endif
  }
};

}  // namespace routing
}  // namespace cuopt
