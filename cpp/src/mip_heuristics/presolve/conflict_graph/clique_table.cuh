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

#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <dual_simplex/user_problem.hpp>

#include <memory>
#include <utilities/timer.hpp>

#include <algorithm>
#include <atomic>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

struct clique_config_t {
  int min_clique_size               = 512;
  int max_clique_size_for_extension = 128;
  // extend_cliques work budget; one unit ≈ one hash/scan op in extend_clique.
  // Soft floor before honoring cut-gen signal; hard ceiling.
  double min_extend_work = 1e7;
  double max_extend_work = 2e9;
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

// CSR per-vertex map: for v in [0, n_vertices), `indices[offsets[v] ..
// offsets[v+1])` is a sorted slice. Build protocol: callers push (src, value)
// pairs and call `finalize_from_unsorted_pairs`.
template <typename i_t>
struct csr_var_map_t {
  std::vector<i_t> offsets;  // size: n_vertices + 1; offsets[v] is the start in `indices`
  std::vector<i_t> indices;  // sorted within each [offsets[v], offsets[v+1]) slice

  void clear_and_resize(i_t n_vertices)
  {
    offsets.assign(n_vertices + 1, 0);
    indices.clear();
  }
  i_t n_keys() const { return offsets.empty() ? 0 : static_cast<i_t>(offsets.size() - 1); }
  i_t slice_size(i_t v) const { return offsets[v + 1] - offsets[v]; }
  // Range-for friendly view of the sorted slice for vertex v.
  std::span<const i_t> slice(i_t v) const
  {
    return {indices.data() + offsets[v], (size_t)(offsets[v + 1] - offsets[v])};
  }
  // O(1) summary used by cut/extension cost-budget heuristics.
  double avg_slice_size() const
  {
    const i_t k = n_keys();
    return k > 0 ? static_cast<double>(indices.size()) / static_cast<double>(k) : 0.0;
  }
  bool slice_contains(i_t v, i_t value) const
  {
    auto s = slice(v);
    return std::binary_search(s.begin(), s.end(), value);
  }

  // Build CSR from unsorted (src, value) pairs. Each output slice is sorted
  // and deduplicated. Caller must keep p.first in [0, n_vertices).
  void finalize_from_unsorted_pairs(i_t n_vertices, std::vector<std::pair<i_t, i_t>>& pairs)
  {
    offsets.assign(n_vertices + 1, 0);
    for (const auto& p : pairs) {
      offsets[p.first + 1]++;
    }
    for (i_t v = 1; v <= n_vertices; ++v) {
      offsets[v] += offsets[v - 1];
    }
    indices.assign(static_cast<size_t>(offsets.back()), i_t{0});
    std::vector<i_t> head(n_vertices, 0);
    for (const auto& p : pairs) {
      indices[offsets[p.first] + head[p.first]++] = p.second;
    }
    for (i_t v = 0; v < n_vertices; ++v) {
      auto* b = indices.data() + offsets[v];
      auto* e = indices.data() + offsets[v] + head[v];
      std::sort(b, e);
      auto* new_end = std::unique(b, e);
      head[v]       = static_cast<i_t>(new_end - b);
    }
    // Compact away dedupe holes.
    std::vector<i_t> new_offsets(n_vertices + 1, 0);
    for (i_t v = 0; v < n_vertices; ++v) {
      new_offsets[v + 1] = new_offsets[v] + head[v];
    }
    if (new_offsets.back() != offsets.back()) {
      std::vector<i_t> new_indices(static_cast<size_t>(new_offsets.back()));
      for (i_t v = 0; v < n_vertices; ++v) {
        std::copy(indices.data() + offsets[v],
                  indices.data() + offsets[v] + head[v],
                  new_indices.data() + new_offsets[v]);
      }
      offsets = std::move(new_offsets);
      indices = std::move(new_indices);
    } else {
      offsets = std::move(new_offsets);
    }
  }
};

template <typename i_t, typename f_t>
struct clique_table_t {
  clique_table_t(i_t n_vertices, i_t min_clique_size_, i_t max_clique_size_for_extension_)
    : min_clique_size(min_clique_size_),
      max_clique_size_for_extension(max_clique_size_for_extension_),
      var_degrees(n_vertices, -1),
      n_variables(n_vertices / 2)
  {
    var_clique_first.clear_and_resize(n_vertices);
    var_clique_addtl.clear_and_resize(n_vertices);
    small_clique_adj.clear_and_resize(n_vertices);
  }

  // Copy disabled; move provided so tests can return by value.
  // Move-assign omitted because of const members.
  clique_table_t(const clique_table_t&)            = delete;
  clique_table_t& operator=(const clique_table_t&) = delete;

  clique_table_t(clique_table_t&& other) noexcept
    : first(std::move(other.first)),
      addtl_cliques(std::move(other.addtl_cliques)),
      var_clique_first(std::move(other.var_clique_first)),
      var_clique_addtl(std::move(other.var_clique_addtl)),
      first_var_positions(std::move(other.first_var_positions)),
      small_clique_adj(std::move(other.small_clique_adj)),
      var_degrees(std::move(other.var_degrees)),
      n_variables(other.n_variables),
      min_clique_size(other.min_clique_size),
      max_clique_size_for_extension(other.max_clique_size_for_extension),
      tolerances(other.tolerances)
  {
  }

  clique_table_t& operator=(clique_table_t&&) = delete;

  std::unordered_set<i_t> get_adj_set_of_var(i_t var_idx) const;
  i_t get_degree_of_var(i_t var_idx);
  bool check_adjacency(i_t var_idx1, i_t var_idx2) const;
  bool empty() const
  {
    return first.empty() && addtl_cliques.empty() && small_clique_adj.indices.empty();
  }

  void set_small_clique_adj_for_test(const std::unordered_map<i_t, std::unordered_set<i_t>>& edges);

  // keeps the large cliques in each constraint
  std::vector<std::vector<i_t>> first;
  // keeps the additional cliques
  std::vector<addtl_clique_t<i_t, f_t>> addtl_cliques;
  csr_var_map_t<i_t> var_clique_first;
  csr_var_map_t<i_t> var_clique_addtl;
  // var_idx -> position mapping for each first clique, enabling O(1) membership/position checks
  std::vector<std::unordered_map<i_t, i_t>> first_var_positions;
  csr_var_map_t<i_t> small_clique_adj;
  // degrees of each vertex
  std::vector<i_t> var_degrees;
  // number of variables in the original problem
  const i_t n_variables;
  const i_t min_clique_size;
  const i_t max_clique_size_for_extension;
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances;
};

// Builds the conflict-graph clique table for `problem`. The base cliques are
// published to `clique_table_out` before the (optional, signal-gated) extension
// phase begins, so cut generation can pick up the table while extension keeps
// running concurrently. Consumers MUST set `*signal_extend` and join the
// producing task before reading the table (see prepare_fractional_sub_conflict_graph), since
// the extension phase keeps mutating the same object after it is published.
template <typename i_t, typename f_t>
void find_initial_cliques(simplex::user_problem_t<i_t, f_t>& problem,
                          typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances,
                          std::shared_ptr<clique_table_t<i_t, f_t>>& clique_table_out,
                          cuopt::timer_t& timer,
                          omp_atomic_t<bool>* signal_extend = nullptr);

template <typename i_t, typename f_t>
void build_clique_table(const simplex::user_problem_t<i_t, f_t>& problem,
                        clique_table_t<i_t, f_t>& clique_table,
                        typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances,
                        bool remove_small_cliques,
                        bool fill_var_clique_maps,
                        cuopt::timer_t& timer);

template <typename i_t, typename f_t>
void fill_var_clique_maps(clique_table_t<i_t, f_t>& clique_table);

}  // namespace cuopt::mathematical_optimization::mip

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
