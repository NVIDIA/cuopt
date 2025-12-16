/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../diversity/helpers.hpp"

#include <memory>
#include <numeric>
#include <vector>

namespace cuopt {
namespace routing {

/*! \brief { Set covering solver } */
struct set_covering {
  // Here the set data is represented
  //! The total number of nodes in the VRP problem
  size_t nodes_number;

  //! Nodes to be covered. The set size here may be smaller then the number of nodes in the VRP
  //! problem as some routes may overlap and should be excluded from the search.
  std::vector<int> nodes_to_cover;

  std::vector<std::vector<detail::NodeInfo<>>> a_routes;
  size_t a_size;
  std::vector<std::vector<detail::NodeInfo<>>> b_routes;
  size_t b_size;

  //! The indexes here correspond to a_routes
  std::vector<int> node_route_map_a;
  //! The indexes here correspond to b_routes
  std::vector<int> node_route_map_b;

  //! Each entry common[i][j] = |A_i \intersection B_i |
  std::vector<std::vector<int>> common_nodes;

  // Here the actual solution is stored
  //! Routes from solution a/b in the present solution (indexing of a_routes/b_routes)
  std::vector<int> from_a;
  //! Indexing hash
  std::vector<char> route_from_a_in;
  std::vector<int> from_b;
  std::vector<char> route_from_b_in;

  //! Helpers for const time evaluation. Contains the nodes number in the completition of
  //! intersection of each set with the current solution
  std::vector<int> a_route_intersection;
  std::vector<int> b_route_intersection;

  //! The minimum number of routes from solution a in the scp solution
  size_t min_routes_from_a{2};
  size_t min_routes_from_b{2};

  //! the quality of solution
  int covered;

  //! helpers to generate initial
  std::vector<int> a_used_routes;
  std::vector<int> b_used_routes;
#ifdef TEST_COVERING
  std::vector<char> covering;
#endif
 private:
  size_t max_r_num{200};
  size_t max_r_len{200};

 public:
  //! \param[in] nodes_number_ {The total number of nodes in the vrp problem }
  set_covering(size_t nodes_number_) : nodes_number(nodes_number_)
  {
    nodes_to_cover.reserve(nodes_number);
#ifdef TEST_COVERING
    covering.assign(nodes_number, 0);
#endif
    // Pre assign anticipated max routes number (this may be expanded during the initialization
    // procedure).
    a_routes.assign(max_r_num, std::vector<detail::NodeInfo<>>());
    for (auto& a : a_routes)
      a.reserve(max_r_len);
    b_routes.assign(max_r_num, std::vector<detail::NodeInfo<>>());
    for (auto& b : b_routes)
      b.reserve(max_r_len);
    a_size = 0;
    b_size = 0;

    common_nodes.assign(max_r_num, std::vector<int>(max_r_num, 0));

    from_a.reserve(max_r_num);
    from_b.reserve(max_r_num);

    a_used_routes.reserve(max_r_num);
    b_used_routes.reserve(max_r_num);
    route_from_a_in.assign(max_r_num, 0);
    route_from_b_in.assign(max_r_num, 0);

    node_route_map_a.assign(nodes_number, -1);
    node_route_map_b.assign(nodes_number, -1);
  }

  /*! \brief { Solve set covering problem. We assume that a_routes, b_routes, a_size, b_size have
   * been filled by outer entity} */
  void solve(int target_route_number)
  {
    recalculate_problem_data();
    create_random_initial(target_route_number);
    perform_search();
  }

 private:
  /*! \brief { Here we assume that  a_routes, b_routes, a_size, b_size have been filled by outer
   * entity. The node to route map and common_nodes are filled (and sizes adjusted if necessary).
   *           Every node has to be covered by at least 1 route!.
   *           from_a/b route_from_a/b_in cleared } */
  inline void recalculate_problem_data() noexcept
  {
    // EXPECT every node is covered by at least one route

    // Zero the common nodes count
    for (size_t j = 0; j < std::min<size_t>(common_nodes.size(), a_routes.size()); j++) {
      auto& a = common_nodes[j];
      // The rest will be truncated anyway
      size_t max_index = std::min<size_t>(b_size, a.size());
      for (size_t i = 0; i < max_index; i++)
        a[i] = 0;
    }
    // Resize if necessary (and zero new elements):
    if (common_nodes.size() < std::max(a_size, b_size)) {
      max_r_num = std::max(a_size, b_size);
      for (auto& a : common_nodes)
        a.resize(max_r_num, 0);
      common_nodes.resize(max_r_num, std::vector<int>(max_r_num, 0));
    }

    std::fill(node_route_map_a.begin(), node_route_map_a.end(), -1);
    std::fill(node_route_map_b.begin(), node_route_map_b.end(), -1);

    int max_a_node_id = -1, max_b_node_id = -1;
    for (size_t a_index = 0; a_index < a_size; a_index++) {
      for (auto& a : a_routes[a_index]) {
        max_a_node_id = std::max(max_a_node_id, a.node());
      }
    }

    for (size_t b_index = 0; b_index < b_size; b_index++) {
      for (auto& b : b_routes[b_index]) {
        max_b_node_id = std::max(max_b_node_id, b.node());
      }
    }

    cuopt_assert(static_cast<size_t>(max_a_node_id) < node_route_map_a.size(),
                 "out of bounds access in recalculate_problem_data!");
    cuopt_assert(static_cast<size_t>(max_b_node_id) < node_route_map_b.size(),
                 "out of bounds access in recalculate_problem_data!");
    cuopt_expects(static_cast<size_t>(max_a_node_id) < node_route_map_a.size(),
                  error_type_t::RuntimeError,
                  "out of bounds access");
    cuopt_expects(static_cast<size_t>(max_b_node_id) < node_route_map_b.size(),
                  error_type_t::RuntimeError,
                  "out of bounds access");

    // Fill node - route mapping
    for (size_t a_index = 0; a_index < a_size; a_index++) {
      for (auto& a : a_routes[a_index]) {
        node_route_map_a[a.node()] = (int)a_index;
      }
    }

    for (size_t b_index = 0; b_index < b_size; b_index++) {
      for (auto& b : b_routes[b_index]) {
        node_route_map_b[b.node()] = (int)b_index;
      }
    }

    nodes_to_cover.clear();
    for (size_t index = 0; index < nodes_number; index++)
      if (node_route_map_a[index] != -1 || node_route_map_b[index] != -1)
        nodes_to_cover.push_back((int)index);

    // Fill common nodes
    for (auto& node : nodes_to_cover) {
      int route_a = node_route_map_a[node];
      int route_b = node_route_map_b[node];
      // Check necessary only if not every node is double covered
      if (route_a >= 0 && route_b >= 0) common_nodes[route_a][route_b]++;
    }

    route_from_a_in.resize(a_size, 0);
    route_from_b_in.resize(b_size, 0);
    a_route_intersection.resize(a_size, 0);
    b_route_intersection.resize(b_size, 0);

    std::fill(route_from_a_in.begin(), route_from_a_in.end(), 0);
    std::fill(route_from_b_in.begin(), route_from_b_in.end(), 0);
    std::fill(a_route_intersection.begin(), a_route_intersection.end(), 0);
    std::fill(b_route_intersection.begin(), b_route_intersection.end(), 0);
  }

  /*! \brief {Create random initial solution. } */
  inline void create_random_initial(int target_route_number) noexcept
  {
    // Clear solution content
    from_a.clear();
    from_b.clear();

    // Assign a random solution with ~ 1/2 routes from 'a' and 1/2 from 'b'
    // ASSERT taget_routes/2 < a_used , b_used
    a_used_routes.resize(a_size);
    b_used_routes.resize(b_size);
    std::iota(a_used_routes.begin(), a_used_routes.end(), 0);
    std::iota(b_used_routes.begin(), b_used_routes.end(), 0);
    std::shuffle(a_used_routes.begin(), a_used_routes.end(), next_random_object());
    std::shuffle(b_used_routes.begin(), b_used_routes.end(), next_random_object());

    int num_a, num_b;

    if (a_used_routes.size() < b_used_routes.size()) {
      num_a = std::min<int>(std::max<int>(1, target_route_number / 2), a_used_routes.size());
      num_b = target_route_number - num_a;
    } else {
      num_b = std::min<int>(std::max<int>(1, target_route_number / 2), b_used_routes.size());
      num_a = target_route_number - num_b;
    }

    std::copy(a_used_routes.begin(), a_used_routes.begin() + num_a, std::back_inserter(from_a));
    std::copy(b_used_routes.begin(), b_used_routes.begin() + num_b, std::back_inserter(from_b));
    for (auto& a : from_a)
      route_from_a_in[a] = 1;
    for (auto& a : from_b)
      route_from_b_in[a] = 1;

    // Calculating intersection of routes with created solution (and covered nodes)

    covered = 0;
    for (auto& node : nodes_to_cover) {
      auto a_id = node_route_map_a[node];
      auto b_id = node_route_map_b[node];

      if (a_id == -1) {  // Node not covered by a
        if (route_from_b_in[b_id]) covered++;
        b_route_intersection[b_id]++;
      } else if (b_id == -1) {  // Node not covered by b
        if (route_from_a_in[a_id]) covered++;
        a_route_intersection[a_id]++;
      } else if (route_from_a_in[a_id] == 0 || route_from_b_in[b_id] == 0) {
        if (route_from_a_in[a_id]) {
          covered++;
          a_route_intersection[a_id]++;
        } else if (route_from_b_in[b_id]) {
          covered++;
          b_route_intersection[b_id]++;
        } else {
          a_route_intersection[a_id]++;
          b_route_intersection[b_id]++;
        }
      } else
        covered++;
    }
    // TEST intersection aqui
    // TEST covered
  }

  /*! \brief { Find best add move better then best_delta and better then 0. } */
  template <bool rem_a, bool add_a>
  inline bool find_best_exchange(int rem_id, int& best_delta, int& best_add, int& random_index)
  {
    // Add from a
    bool ret     = false;
    auto& from_x = (add_a) ? route_from_a_in : route_from_b_in;
    for (size_t i = 0; i < from_x.size(); i++) {
      if (from_x[i] == 0) {
        auto delta = evaluate_route_swap<rem_a, add_a>(rem_id, i);
        if (delta > best_delta) {
          random_index = 2;
          best_delta   = delta;
          best_add     = i;
          ret          = true;
        } else if (delta == best_delta && delta > 0 && next_random() % random_index == 0) {
          random_index++;
          best_delta = delta;
          best_add   = i;
          ret        = true;
        }
      }
    }
    return ret;
  }

  /*! \brief { Perform Local Search on the solution } */
  void perform_search()
  {
    bool performed = true;

    while (performed) {
      performed = false;

      int random_index   = 2;
      int best_delta     = 0;
      int best_add_index = -1;
      // Fill availible indices
      if (from_a.size() > min_routes_from_a) {
        a_used_routes.resize(from_a.size());
        std::iota(a_used_routes.begin(), a_used_routes.end(), 0);
      } else
        a_used_routes.clear();

      if (from_b.size() > min_routes_from_b) {
        b_used_routes.resize(from_b.size());
        std::iota(b_used_routes.begin(), b_used_routes.end(), 0);
      } else
        b_used_routes.clear();

      // Iterate the routes in solution in random order to find a swap move
      while (a_used_routes.size() + b_used_routes.size() > 0) {
        std::pair<bool, bool> move_type;
        auto u_index = next_random() % (a_used_routes.size() + b_used_routes.size());

        int index;
        if (u_index < a_used_routes.size()) {
          index                  = a_used_routes[u_index];
          a_used_routes[u_index] = a_used_routes.back();
          a_used_routes.pop_back();

          auto a_id_rem = from_a[index];

          if (find_best_exchange<true, true>(a_id_rem, best_delta, best_add_index, random_index))
            move_type = {true, true};
          if (from_a.size() > min_routes_from_a &&
              find_best_exchange<true, false>(a_id_rem, best_delta, best_add_index, random_index))
            move_type = {true, false};

        } else {
          u_index = u_index - a_used_routes.size();

          index                  = b_used_routes[u_index];
          b_used_routes[u_index] = b_used_routes.back();
          b_used_routes.pop_back();

          auto a_id_rem = from_b[index];

          if (from_b.size() > min_routes_from_b &&
              find_best_exchange<false, true>(a_id_rem, best_delta, best_add_index, random_index))
            move_type = {false, true};
          if (find_best_exchange<false, false>(a_id_rem, best_delta, best_add_index, random_index))
            move_type = {false, false};
        }

        if (best_delta > 0) {
          performed = true;
          if (move_type.first && move_type.second)
            perform_move<true, true>(index, best_add_index);
          else if (move_type.first && !move_type.second)
            perform_move<true, false>(index, best_add_index);
          else if (!move_type.first && move_type.second)
            perform_move<false, true>(index, best_add_index);
          else
            perform_move<false, false>(index, best_add_index);
          break;
        }
      }
    }
  }

  /*! \brief { Evaluate a swap move} */
  template <bool is_rem_a, bool is_add_a>
  inline int evaluate_route_swap(int id_rem, int id_add) noexcept
  {
    if constexpr (is_rem_a && is_add_a)
      return a_route_intersection[id_add] - a_route_intersection[id_rem];
    if constexpr (!is_rem_a && !is_add_a)
      return b_route_intersection[id_add] - b_route_intersection[id_rem];
    if constexpr (is_rem_a && !is_add_a)
      return b_route_intersection[id_add] - a_route_intersection[id_rem] +
             common_nodes[id_rem][id_add];
    if constexpr (!is_rem_a && is_add_a)
      return a_route_intersection[id_add] - b_route_intersection[id_rem] +
             common_nodes[id_add][id_rem];
  }

  /*! \brief { Perform a swap move} */
  template <bool is_rem_a, bool is_add_a>
  inline void perform_move(int f_id_rem, int id_add) noexcept
  {
    auto& from_rem    = (is_rem_a) ? from_a : from_b;
    auto& from_add    = (is_add_a) ? from_a : from_b;
    auto& from_rem_in = (is_rem_a) ? route_from_a_in : route_from_b_in;
    auto& from_add_in = (is_add_a) ? route_from_a_in : route_from_b_in;
    auto& rem_routes  = (is_rem_a) ? a_routes : b_routes;
    auto& add_routes  = (is_add_a) ? a_routes : b_routes;

    auto id_rem = from_rem[f_id_rem];

    covered += evaluate_route_swap<is_rem_a, is_add_a>(id_rem, id_add);

    // Iterate through nodes of removed route and adjust counters (could be done in a route loop
    // optimal choice depends on which is greater: route length or routes number)
    for (auto node : rem_routes[id_rem]) {
      int id = (is_rem_a) ? node_route_map_b[node.node()] : node_route_map_a[node.node()];
      if (id >= 0) {
        if constexpr (is_rem_a) b_route_intersection[id]++;
        if constexpr (!is_rem_a) a_route_intersection[id]++;
      }
    }

    // remove route from solution
    from_rem_in[id_rem] = 0;
    from_rem[f_id_rem]  = from_rem.back();
    from_rem.pop_back();

    // Iterate through nodes of added route and adjust counters
    for (auto node : add_routes[id_add]) {
      int id = (is_add_a) ? node_route_map_b[node.node()] : node_route_map_a[node.node()];
      if (id >= 0) {
        if constexpr (is_add_a) {
          // if ( route_from_b_in[node_route_map_b[node]])
          b_route_intersection[id]--;
        }
        if constexpr (!is_add_a) {
          // if ( route_from_a_in[node_route_map_a[node]])
          a_route_intersection[id]--;
        }
      }
    }

    // remove route from solution
    from_add_in[id_add] = 1;
    from_add.push_back(id_add);

#ifdef TEST_COVERING
    if (!test_covered()) printf(" Covered number invalid\n");
    fflush(stdout);
#endif
  }

  bool test_covered()
  {
#ifdef TEST_COVERING
    std::fill(covering.begin(), covering.end(), 0);
    for (auto a_id : from_a)
      for (auto node : a_routes[a_id])
        covering[node] = 1;

    for (auto b_id : from_b)
      for (auto node : b_routes[b_id])
        covering[node] = 1;
    int cov = 0;
    for (auto a : covering)
      if (a == 1) cov++;
    if (cov == covered) return true;
#endif
    return false;
  }
};

}  // namespace routing
}  // namespace cuopt
