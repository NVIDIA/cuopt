/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <dual_simplex/user_problem.hpp>

#include <utilities/timer.hpp>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cuopt::linear_programming::detail {

struct clique_config_t {
  int min_clique_size               = 512;
  int max_clique_size_for_extension = 128;
};

template <typename i_t, typename f_t>
struct entry_t {
  i_t col;
  f_t val;
  bool operator<(const entry_t& other) const { return val < other.val; }
  bool operator<(double other) const { return val < other; }
};

template <typename i_t, typename f_t>
struct knapsack_constraint_t {
  std::vector<entry_t<i_t, f_t>> entries;
  f_t rhs;
  i_t cstr_idx;
  bool is_set_packing      = false;
  bool is_set_partitioning = false;
};

template <typename i_t, typename f_t>
struct addtl_clique_t {
  i_t vertex_idx;
  i_t clique_idx;
  i_t start_pos_on_clique;
};

template <typename i_t, typename f_t>
struct clique_table_t {
  clique_table_t(i_t n_vertices, i_t min_clique_size_, i_t max_clique_size_for_extension_)
    : min_clique_size(min_clique_size_),
      max_clique_size_for_extension(max_clique_size_for_extension_),
      var_clique_map_first(n_vertices),
      var_clique_map_addtl(n_vertices),
      adj_list_small_cliques(n_vertices),
      var_degrees(n_vertices, -1),
      n_variables(n_vertices / 2)
  {
  }

  std::unordered_set<i_t> get_adj_set_of_var(i_t var_idx);
  i_t get_degree_of_var(i_t var_idx);
  bool check_adjacency(i_t var_idx1, i_t var_idx2);

  // keeps the large cliques in each constraint
  std::vector<std::vector<i_t>> first;
  // keeps the additional cliques
  std::vector<addtl_clique_t<i_t, f_t>> addtl_cliques;
  // TODO figure out the performance of lookup for the following: unordered_set vs vector
  // keeps the indices of original(first) cliques that contain variable x
  std::vector<std::unordered_set<i_t>> var_clique_map_first;
  // keeps the indices of additional cliques that contain variable x
  std::vector<std::unordered_set<i_t>> var_clique_map_addtl;
  // adjacency list to keep small cliques, this basically keeps the vars share a small clique
  // constraint
  std::unordered_map<i_t, std::unordered_set<i_t>> adj_list_small_cliques;
  // degrees of each vertex
  std::vector<i_t> var_degrees;
  // number of variables in the original problem
  const i_t n_variables;
  const i_t min_clique_size;
  const i_t max_clique_size_for_extension;
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances;
};

template <typename i_t, typename f_t>
void find_initial_cliques(dual_simplex::user_problem_t<i_t, f_t>& problem,
                          typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances,
                          cuopt::timer_t& timer);

}  // namespace cuopt::linear_programming::detail

// Possible application to rounding procedure, keeping it as reference

// fix set of variables x_1, x_2, x_3,... in a bulk. Consider sorting according largest size GUB
// constraint(or some other criteria).

// compute new activities on changed constraints, given that x_1=v_1, x_2=v_2, x_3=v_3:

// 	if the current constraint is GUB

// 		if at least two binary vars(note that some can be full integer) are common: (needs
// binary_vars_in_bulk^2 number of checks)

// 			return infeasible

// 		else

// 			set L_r to 1.

// 	else(non-GUB constraints)

// 		greedy clique partitioning algorithm:

// 			set L_r = sum(all positive coefficients on binary vars) + sum(min_activity contribution on
// non-binary vars) # note that the paper doesn't contain this part, since it only deals with binary

// 			# iterate only on binary variables(i.e. vertices of B- and complements of B+)

// 			start with highest weight vertex (v) among unmarked and mark it

// 			find maximal clique among unmarked containing the vertex: (there are various algorithms to
// find maximal clique)

// 				max_clique = {v}

// 				L_r -= w_v

// 				# prioritization is on higher weight vertex when there are equivalent max cliques?
//                 # we could try BFS to search multiple greedy paths
// 				for each unmarked vertex(w):

// 					counter = 0

// 					for each vertex(k) in max_clique:

// 						if(check_if_pair_shares_an_edge(w,k))

// 							counter++

// 					if counter == max_clique.size()

// 						max_clique = max_clique U {w}

// 						mark w as marked

// 			if(L_r > UB) return infeasible

// remove all fixed variables(original and newly propagated) from the conflict graph. !!!!!! still a
// bit unclear how to remove it from the adjaceny list data structure since it only supports
// additions!!!!

// add newly discovered GUB constraints into dynamic adjacency list

// do double probing to infer new edges(we need a heuristic to choose which pairs to probe)

// check_if_pair_shares_an_edge(w,v):

// 	check GUB constraints by traversing the double linked list:

// 		on the column of variable w:

// 		for each row:

// 			if v is contained on the row

// 				return true

// 	check added edges on adjacency list:

// 		k <- last[w]

// 		while k != 0

// 			if(adj[k] == v)

// 				return true

// 			k <-next[k]

// 	return false
