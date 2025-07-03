/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include "../diversity/helpers.hpp"

#include <vector>

namespace cuopt {
namespace routing {

/*! \brief { Recombine two solutions. One of input solutions may be overwritten and child may
 * contain lower number of nodes. } */
template <class Solution>
struct inversion {
  //! Routes from A sorted with respect to B
  std::array<std::vector<detail::NodeInfo<>>, 6> routes;

  //! Node positions in the routes. The position can be either nominal or relative to route length.
  //! Vectors are overwritten at each recombination
  std::vector<double> B_node_index;

  //! Helper for route remove
  std::vector<detail::NodeInfo<>> route_starts_helper;
  //! Helper for route add
  std::vector<std::pair<int, std::vector<detail::NodeInfo<>>>> routes_to_add;
  //! Removing the bulk removes
  std::vector<int> routes_to_remove;
  //! Different route ids
  std::vector<size_t> different_routes;

  //! Helpers to store routes for calculate inversions
  std::vector<int> helper;
  std::vector<int> helper_tmp;

  std::vector<int> ids_a;
  std::vector<std::pair<int, int>> a_inversions;

  inversion(size_t nodes_number) : B_node_index(nodes_number, 0.0), helper(nodes_number)
  {
    for (auto& a : routes)
      a.reserve(128);
  }

  const size_t route_size_limit = 60;

  /*! \brief { Recombine solutions a and b. The output will be stored in a (it will be overwritten)
   * } */
  bool recombine(Solution& a, const Solution& b)
  {
    raft::common::nvtx::range fun_scope("inversion");
    // THIS IS TEMPORARY UNTIL EMPTY ROUTES ARE FIXED
    if (check_if_routes_empty(a) || check_if_routes_empty(b)) return false;
    bool relative = next_random() % 2;
    bool tsp      = a.get_routes().size() == 1;
    route_starts_helper.clear();
    different_routes.clear();
    routes_to_add.clear();
    routes_to_remove.clear();

    std::vector<int> removed_vehicles;
    for (auto& route : routes) {
      route.clear();
    }

    fill_positions(b, relative);

    if (!find_routes(a, b)) return false;

    for (size_t i = 0; i < route_starts_helper.size(); i++) {
      /*
            printf(" \n");
            for (size_t j=0 ; j < routes[i].size() ; j++ )
                printf(" %d ", routes[i][j]);
*/
      int start_index;
      int end_index;
      if (routes[i].size() <= route_size_limit && !tsp) {
        start_index = 0;
        end_index   = routes[i].size();
      } else if (!tsp) {
        start_index = next_random() % ((int)routes[i].size() - route_size_limit);
        end_index   = start_index + route_size_limit;
      } else {
        start_index = next_random() % (std::max<int>(1, (int)routes[i].size() / 2));
        end_index   = (int)routes[i].size() / 2 + start_index;
      }
      std::sort(routes[i].begin() + start_index,
                routes[i].begin() + end_index,
                [this](const auto& a, const auto& b) {
                  return (B_node_index[a.node()] < B_node_index[b.node()]);
                });
      /*
                  printf(" \n");
                  for (size_t j=0 ; j < routes[i].size() ; j++ )
                      printf(" %d ", routes[i][j]);
      */
    }

    //  printf(" \n");

    for (size_t i = 0; i < route_starts_helper.size(); i++) {
      auto node = route_starts_helper[i];
      cuopt_assert(!node.is_break(), "route should not be starting with break nodes");
      auto id        = a.nodes[node.node()].r_id;
      int vehicle_id = a.nodes[node.node()].v_id;
      routes_to_remove.push_back(id);
      removed_vehicles.push_back(vehicle_id);
    }
    a.remove_routes(routes_to_remove);

    for (size_t i = 0; i < route_starts_helper.size(); i++) {
      // do a random assignment of vehicles
      // FIXME:: use heuristics to figure out which vehicle to use
      int vehicle_id = pop_random(removed_vehicles);
      routes_to_add.push_back({vehicle_id, routes[i]});
    }
    a.add_new_routes(routes_to_add);

    return true;
  }

  bool find_routes(const Solution& a, const Solution& b)
  {
    a.different_route_ids(ids_a, b);
    if (ids_a.empty()) { return false; }

    a_inversions.clear();
    int max_inversions = 0;
    for (size_t index = 0; index < ids_a.size(); index++) {
      helper.clear();
      helper_tmp.clear();
      int r_id = ids_a[index];

      int start = a.get_routes()[r_id].start.node();
      int end   = a.get_routes()[r_id].end.node();

      while (start != end) {
        helper.push_back(start);
        helper_tmp.push_back(start);
        start = a.succ[start].node();
      }
      helper.push_back(end);
      helper_tmp.push_back(end);

      if (helper.size() > 1) {
        auto inversions = calculate_inversions(0, helper.size() - 1);
        a_inversions.emplace_back(ids_a[index], inversions);
        if (inversions > max_inversions) max_inversions = inversions;
      } else
        a_inversions.emplace_back(ids_a[index], 0);
    }

    if (max_inversions < 2) return false;

    // Reverse sort wrt inversions number
    std::sort(a_inversions.begin(),
              a_inversions.end(),
              [](std::pair<int, int>& a, std::pair<int, int>& b) { return (a.second > b.second); });

    int i = 0;
    while (size_t(i) < a_inversions.size()) {
      if (a_inversions[i].second < 2) { break; }
      ++i;
    }

    if (a_inversions.size() > 10)
      a_inversions.erase(a_inversions.begin() + std::min<int>(i, 10), a_inversions.end());
    /*
    printf("\n");
    for ( auto&a : a_inversions )
        printf("%d ", a.second);
    printf("\n");
    */

    different_routes.clear();
    for (auto& a : a_inversions) {
      different_routes.push_back(a.first);
    }
    int routes_number             = 0;
    int total_length              = 0;
    const int max_inverted_routes = std::max(1, std::min((int)different_routes.size() / 2, 5));
    while (total_length < 60 && routes_number < max_inverted_routes &&
           different_routes.size() > 0) {
      size_t initial_id            = next_random() % (different_routes.size());
      int initial                  = different_routes[initial_id];
      different_routes[initial_id] = different_routes.back();
      different_routes.pop_back();

      load_route(a, routes_number, initial);
      total_length += routes[routes_number++].size();
    }

    return true;
  }

  void load_route(const Solution& a, size_t route_index_in_routes, size_t route_position)
  {
    detail::NodeInfo<> start = a.routes[route_position].start;
    route_starts_helper.push_back(start);

    while (!start.is_depot()) {
      routes[route_index_in_routes].push_back(start);
      start = a.succ[start.node()];
    }
  }

  void fill_positions(const Solution& b, bool relative)
  {
    bool out_of_bounds =
      std::any_of(b.nodes.begin(), b.nodes.end(), [bounds = B_node_index.size()](auto node) {
        return node.node_id() >= static_cast<int>(bounds);
      });
    cuopt_assert(!out_of_bounds, "out of bounds access in fill_positions!");
    cuopt_expects(!out_of_bounds, error_type_t::RuntimeError, "A runtime error occurred!");
    for (auto& node : b.nodes) {
      if (node.is_depot() || b.unserviced(node.node_id())) continue;

      B_node_index[node.node_id()] = (relative)
                                       ? (double)b.nodes[node.node_id()].r_index /
                                           (double)b.routes[b.nodes[node.node_id()].r_id].length
                                       : (double)b.nodes[node.node_id()].r_index;
    }
  }

  /*! \brief { Mergesort based inversion calculator. }*/
  int calculate_inversions(int left, int right)
  {
    int middle, inversions = 0;
    if (right > left) {
      middle = (right + left) / 2;

      inversions += calculate_inversions(left, middle);
      inversions += calculate_inversions(middle + 1, right);
      inversions += merge(left, middle + 1, right);
    }
    return inversions;
  }

  /*! \brief { Merge arrays while calculating the number of inversions.  }*/
  int merge(int left, int middle, int right)
  {
    int i = left, j = middle, k = left;
    int inversions = 0;

    while ((i < middle) && (j <= right)) {
      if (B_node_index[helper[i]] > B_node_index[helper[j]]) {
        helper_tmp[k++] = helper[j++];
        // All in range i-middle are an inversion with respect to j
        inversions += (middle - i);
      } else
        helper_tmp[k++] = helper[i++];
    }

    while (i < middle)
      helper_tmp[k++] = helper[i++];

    while (j <= right)
      helper_tmp[k++] = helper[j++];

    for (i = left; i <= right; i++)
      helper[i] = helper_tmp[i];

    return inversions;
  }
};

}  // namespace routing
}  // namespace cuopt
