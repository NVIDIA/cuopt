/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuts/cuts.hpp>
#include <math_optimization/tic_toc.hpp>
#include <presolve/conflict_graph/clique_table.cuh>
#include <utilities/macros.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

enum class clique_cut_build_status_t : int8_t { NO_CUT = 0, CUT_ADDED = 1, INFEASIBLE = 2 };

// Pick the seed vertex with the smallest conflict-graph degree. Returns -1 if
// the seed is empty or the time limit is hit while scanning.
template <typename i_t, typename f_t>
i_t min_degree_anchor(const std::vector<i_t>& seed,
                      mip::clique_table_t<i_t, f_t>& graph,
                      f_t start_time,
                      f_t time_limit)
{
  i_t smallest_degree     = std::numeric_limits<i_t>::max();
  i_t smallest_degree_var = -1;
  for (const i_t v : seed) {
    if (toc(start_time) >= time_limit) { return -1; }
    i_t degree = graph.get_degree_of_var(v);
    if (degree < smallest_degree) {
      smallest_degree     = degree;
      smallest_degree_var = v;
    }
  }
  return smallest_degree_var;
}

// Reduced-cost key for a CG vertex. A complement vertex (idx >= num_vars) maps
// to the original variable and flips the sign. Sorting candidates by this key
// keeps xstar minimally disturbed so the resulting cut stays binding and the
// dual simplex resolve stays cheap.
template <typename i_t, typename f_t>
f_t cg_reduced_cost(i_t vertex_idx, const std::vector<f_t>& reduced_costs, i_t num_vars)
{
  i_t var_idx = vertex_idx % num_vars;
  cuopt_assert(var_idx >= 0 && var_idx < static_cast<i_t>(reduced_costs.size()),
               "Reduced cost index out of range");
  f_t rc = reduced_costs[var_idx];
  if (!std::isfinite(rc)) { rc = 0.0; }
  return vertex_idx >= num_vars ? -rc : rc;
}

template <typename i_t, typename f_t>
void sort_candidates_by_reduced_cost(std::vector<i_t>& candidates,
                                     const std::vector<f_t>& reduced_costs,
                                     i_t num_vars)
{
  std::sort(candidates.begin(), candidates.end(), [&](i_t a, i_t b) {
    return cg_reduced_cost(a, reduced_costs, num_vars) <
           cg_reduced_cost(b, reduced_costs, num_vars);
  });
}

// Greedily grow `selected` by appending candidates (assumed already ordered by
// reduced cost) that are adjacent to every current member of `selected`. The
// resulting `selected` is therefore a clique. Stops early when the time or work
// budget is exhausted.
template <typename i_t, typename f_t>
void greedy_extend_clique(std::vector<i_t>& selected,
                          const std::vector<i_t>& candidates,
                          mip::clique_table_t<i_t, f_t>& graph,
                          f_t adj_check_cost,
                          f_t start_time,
                          f_t time_limit,
                          f_t* work_estimate,
                          f_t max_work_estimate)
{
  for (const i_t candidate : candidates) {
    if (toc(start_time) >= time_limit) { return; }
    bool add   = true;
    i_t checks = 0;
    for (const i_t v : selected) {
      checks++;
      if (!graph.check_adjacency(candidate, v)) {
        add = false;
        break;
      }
    }
    if (add_work_estimate(adj_check_cost * checks, work_estimate, max_work_estimate)) { break; }
    if (add) { selected.push_back(candidate); }
  }
}

}  // namespace cuopt::mathematical_optimization::mip
