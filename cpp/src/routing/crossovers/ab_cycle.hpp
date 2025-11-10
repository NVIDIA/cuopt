/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../diversity/helpers.hpp"

#include <algorithm>
#include <vector>

namespace cuopt {
namespace routing {

//! The ESET structure needed by the asymmetric EAX algorithm ( Nagata Braysy Duallert ). It
//! represents the graph consisting of edges from the symmetric difference of two input VRP
//! solutions. As both directions of edges may appear the graph representation is a list
//! representation of undirected graph. We separate edges from two distinct solutions as they appear
//! interchangably in the AB cycles (the first index) and from/to edges (for asymetric cycles).
struct ESET_graph {
  struct edge {
    detail::NodeInfo<> from;
    detail::NodeInfo<> to;
    bool from_sol;
    edge(detail::NodeInfo<> from_, detail::NodeInfo<> to_, bool from_sol_)
      : from(from_), to(to_), from_sol(from_sol_)
    {
    }
    bool operator==(const edge& e) const
    {
      return (from == e.from) && (e.to == to) && (from_sol == e.from_sol);
    }
  };
  //! [ solution index <0,1> ][is edge from the node <0,1>][vrp node index <0,problem_size)]
  std::vector<std::vector<edge>> eset[2][2];

  std::vector<edge> helper;

  int total_edges;

  ESET_graph(size_t nodes_number)
  {
    eset[0][0].assign(nodes_number, std::vector<edge>());
    eset[1][0].assign(nodes_number, std::vector<edge>());
    eset[0][1].assign(nodes_number, std::vector<edge>());
    eset[1][1].assign(nodes_number, std::vector<edge>());
    helper.reserve(1024);
  }

  void clear()
  {
    total_edges = 0;
    helper.clear();
    for (auto& a : eset[0][0])
      a.clear();
    for (auto& b : eset[1][0])
      b.clear();
    for (auto& a : eset[0][1])
      a.clear();
    for (auto& b : eset[1][1])
      b.clear();
  }

  inline int degree(size_t index) noexcept
  {
    return eset[0][0][index].size() + eset[0][1][index].size() + eset[1][0][index].size() +
           eset[1][1][index].size();
  }

  void calculate_edges()
  {
    total_edges = 0;
    for (size_t j = 0; j < eset[0][0].size(); j++)
      total_edges +=
        eset[0][0][j].size() + eset[0][1][j].size() + eset[1][0][j].size() + eset[1][1][j].size();
    // Every edge is represented twice
    total_edges /= 2;
  }

  //! The eset has to be filled by outside entity. Then the method should fill the helper vector
  //! with a random AB cycle.
  bool find_cycle(bool asymmetric, int index_hint = -1, bool expand = false)
  {
    // Clear if not expanding
    if (!expand) helper.clear();

    int index = (index_hint == -1) ? non_empty_index() : index_hint;
    if (index < 0) return false;

    // Choose randomly starting solution
    bool solution_index = next_random() % 2;
    // If 0 empty then take 1
    bool from_node = eset[solution_index][0][index].empty();
    // If 0 is non-empty and 1 is non-empty randomly pick between either of them
    if (!eset[solution_index][!from_node][index].empty()) from_node = next_random() % 2;

    int cycle_length = 0;
    auto first_index = index;

    while (!eset[solution_index][from_node][index].empty()) {
      cycle_length++;
      auto& vec = eset[solution_index][from_node][index];
      cuopt_assert(vec.size() != 0, "Vector size cannot be 0");
      size_t edge_index = next_random() % vec.size();
      edge edge_        = vec[edge_index];
      helper.push_back(edge_);

      // Pop first representation of edge_
      vec[edge_index] = vec.back();
      vec.pop_back();

      // Find next index
      index = (from_node) ? edge_.to.node() : edge_.from.node();

      // Pop second representation of edge_
      auto& second_representation = eset[solution_index][!from_node][index];
      auto it    = std::find(second_representation.begin(), second_representation.end(), edge_);
      edge_index = it - second_representation.begin();
      second_representation[edge_index] = second_representation.back();
      second_representation.pop_back();

      if (index == first_index && cycle_length % 2 == 0) break;

      solution_index = !solution_index;
      // We change from node ( always if asymmetric , randomly if symmetric )
      if (asymmetric)
        from_node = !from_node;
      else {
        from_node = eset[solution_index][0][index].empty();
        if (!eset[solution_index][!from_node][index].empty()) from_node = next_random() % 2;
      }
    }
    return true;
  }

  //! The cycle in helper will be expanded by blocking ( if possible ). We expand the cycle in the
  //! helper structure
  bool expand_cycle(bool asymmetric)
  {
    for (auto& a : helper) {
      if (degree((int)a.from.node()) > 0) {
        find_cycle(asymmetric, a.from.node(), true);
        // Assert helper size increased
        return true;
      }
    }
    return false;
  }

  const std::vector<edge>& get_cycle() { return helper; }

 private:
  int non_empty_index()
  {
    int num_nodes_with_edges = 0;

    for (size_t j = 0; j < eset[0][0].size(); j++) {
      auto& a = eset[0][0][j];
      auto& b = eset[1][0][j];
      auto& c = eset[0][1][j];
      auto& d = eset[1][1][j];

      if (a.size() > 0 || b.size() > 0 || c.size() > 0 || d.size() > 0) num_nodes_with_edges++;
    }

    if (num_nodes_with_edges == 0) return -1;

    int counter = next_random() % num_nodes_with_edges;

    for (int index = 0; index < (int)eset[0][0].size(); index++) {
      if (eset[0][0][index].size() > 0 || eset[0][1][index].size() > 0 ||
          eset[1][0][index].size() > 0 || eset[1][1][index].size() > 0)
        counter--;
      if (counter < 0) return index;
    }
    // Shouldn't get here
    return -1;
  }
};

}  // namespace routing
}  // namespace cuopt
