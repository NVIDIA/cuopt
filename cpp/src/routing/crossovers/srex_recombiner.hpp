/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../diversity/helpers.hpp"
#include "set_covering.hpp"

#include <unordered_set>

namespace cuopt {
namespace routing {

/*! \brief { Recombine two solutions. One of input solutions may be overwritten and child may
 * contain lower number of nodes. } */
template <class Solution>
struct srex {
  set_covering s;

  std::vector<int> ids_a;
  std::vector<int> ids_b;

  std::vector<detail::NodeInfo<>> helper;
  std::vector<int> tmp_visited;
  std::vector<std::pair<int, std::vector<detail::NodeInfo<>>>> tmp_routes;
  std::unordered_set<int> tmp_route_ids;
  std::unordered_set<int> tmp_vehicle_ids;

  srex(size_t nodes_number) : s(nodes_number)
  {
    ids_a.reserve(200);
    ids_b.reserve(200);
    helper.reserve(200);
    tmp_routes.reserve(200);
  }

  /*! \brief { Recombine solutions a and b. The output will be stored in a (it will be overwritten)
   * } */
  bool recombine(Solution& a, Solution& b, bool& ret)
  {
    raft::common::nvtx::range fun_scope("srex");
    // different routes from a & b > 1. Set min routes
    if (a.routes.size() <= 1 || b.routes.size() <= 1) return false;
    if (check_if_routes_empty(a) || check_if_routes_empty(b)) return false;
    // THIS IS TEMPORARY UNTIL EMPTY ROUTES ARE FIXED
    // cuopt_func_call(check_if_routes_empty(a));
    // cuopt_func_call(check_if_routes_empty(b));
    helper.clear();
    // We omit common routes
    a.different_route_ids(ids_a, b);
    b.different_route_ids(ids_b, a);

    if (ids_a.size() <= 1 || ids_b.size() <= 1) return false;

    if (s.a_routes.size() < ids_a.size()) s.a_routes.resize(ids_a.size());
    if (s.b_routes.size() < ids_b.size()) s.b_routes.resize(ids_b.size());
    s.a_size = ids_a.size();
    s.b_size = ids_b.size();

    fill_scp_a(a);
    fill_scp_b(b);
    size_t target_number = std::min<size_t>(ids_a.size(), ids_b.size());
    s.solve(target_number);

    ret           = (s.from_a.size() > s.from_b.size()) ? false : true;
    auto& guiding = (ret == false) ? a : b;
    auto& other   = (ret == true) ? a : b;

    auto& guiding_ids = (ret == false) ? s.from_a : s.from_b;
    auto& other_ids   = (ret == true) ? s.from_a : s.from_b;

    auto& guiding_routes = (ret == false) ? s.a_routes : s.b_routes;
    auto& other_routes   = (ret == true) ? s.a_routes : s.b_routes;

    tmp_visited.assign(guiding.nodes.size(), -1);
    tmp_routes.clear();

    // Add routes found by scp solver. If node is already assigned it is omited ( overassigned are
    // assigned according to greedy strategy )
    while (guiding_ids.size() > 0 || other_ids.size() > 0) {
      auto index = next_random() % (guiding_ids.size() + other_ids.size());
      if (index < guiding_ids.size()) {
        auto a_id = guiding_ids[index];
        helper.clear();
        for (auto node : guiding_routes[a_id])
          if (tmp_visited[node.node()] == -1) {
            helper.push_back(node);
            tmp_visited[node.node()] = 1;  // mark with something other than -1
          }

        if (helper.empty()) {
          //                    printf(" Empty Route\n");
          return false;
        }

        tmp_routes.emplace_back(0, helper);
        guiding_ids[index] = guiding_ids.back();
        guiding_ids.pop_back();
      } else {
        index     = index - guiding_ids.size();
        auto b_id = other_ids[index];

        helper.clear();
        for (auto node : other_routes[b_id]) {
          if (tmp_visited[node.node()] == -1) {
            helper.push_back(node);
            tmp_visited[node.node()] = 1;  // mark with something other than -1
          }
        }

        if (helper.empty()) {
          //                    printf(" Empty Route\n");
          return false;
        }
        tmp_routes.emplace_back(0, helper);
        other_ids[index] = other_ids.back();
        other_ids.pop_back();
      }
    }
    // nodes to cover at the end contains the uncovered nodes
    std::unordered_set<int> unserviced_nodes(s.nodes_to_cover.begin(), s.nodes_to_cover.end());
    // Eliminating already existing routes
    size_t j = 0;
    while (j < tmp_routes.size()) {
      if (exist_equivalent(guiding, tmp_routes[j].second)) {
        for (auto n : tmp_routes[j].second) {
          if (unserviced_nodes.count(n.node()) != 0) { unserviced_nodes.erase(n.node()); }
        }
        tmp_routes[j] = tmp_routes.back();
        tmp_routes.pop_back();
      } else {
        j++;
      }
    }
    tmp_route_ids.clear();
    tmp_vehicle_ids.clear();
    for (auto& a : tmp_routes) {
      for (auto& b : a.second) {
        if (!guiding.unserviced(b.node())) {
          if (unserviced_nodes.count(b.node()) != 0) { unserviced_nodes.erase(b.node()); }
          tmp_route_ids.insert(guiding.nodes[b.node()].r_id);
          tmp_vehicle_ids.insert(guiding.nodes[b.node()].v_id);
        }
      }
    }
    // remove the routes of the remaining uncovered nodes
    for (auto& a : unserviced_nodes) {
      if (!guiding.unserviced(a)) {
        tmp_route_ids.insert(guiding.nodes[a].r_id);
        tmp_vehicle_ids.insert(guiding.nodes[a].v_id);
      }
    }
    if (tmp_route_ids.size() == 0 || tmp_routes.size() == 0) { return false; }

    // populate vehicle ids randomly
    std::vector<int> tmp_vehicle_ids_vec;
    tmp_vehicle_ids_vec.insert(
      tmp_vehicle_ids_vec.end(), tmp_vehicle_ids.begin(), tmp_vehicle_ids.end());
    for (auto& r : tmp_routes) {
      r.first = pop_random(tmp_vehicle_ids_vec);
    }
    std::vector<int> routes_to_remove;
    routes_to_remove.insert(routes_to_remove.end(), tmp_route_ids.begin(), tmp_route_ids.end());
    guiding.remove_routes(routes_to_remove);

    guiding.add_new_routes(tmp_routes);
    return true;
  }

  bool exist_equivalent(const Solution& S, const std::vector<detail::NodeInfo<>>& route)
  {
    if (S.pred[route[0].node()].node() != DEPO) { return false; }
    if (S.succ[route.back().node()].node() != DEPO) { return false; }
    // When the route size is 1, the following for loop will not be executed,
    // so we have to check if the node is serviced or not
    if (S.unserviced(route[0].node())) { return false; }

    for (size_t i = 0; i < route.size() - 1; i++) {
      if (S.succ[route[i].node()] != route[i + 1]) { return false; }
    }
    return true;
  }

  void fill_scp_a(const Solution& a)
  {
    size_t id = 0;
    for (auto& r_id : ids_a) {
      auto& a_routes = s.a_routes[id];
      a_routes.clear();
      detail::NodeInfo<> start = a.routes[r_id].start;
      while (!start.is_depot()) {
        a_routes.push_back(start);
        start = a.succ[start.node()];
      }
      id++;
    }
  }

  void fill_scp_b(const Solution& b)
  {
    size_t id = 0;
    for (auto& r_id : ids_b) {
      auto& b_routes = s.b_routes[id];
      b_routes.clear();
      detail::NodeInfo<> start = b.routes[r_id].start;
      while (!start.is_depot()) {
        b_routes.push_back(start);
        start = b.succ[start.node()];
      }
      id++;
    }
  }
};

}  // namespace routing
}  // namespace cuopt
