/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "clique_activity_corrections.cuh"

#include <mip_heuristics/mip_constants.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/logger.hpp>
#include <utilities/macros.cuh>

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
typename clique_group_table_t<i_t, f_t>::view_t clique_group_table_t<i_t, f_t>::view()
{
  view_t v;
  v.group_constraint_ids     = make_span(group_constraint_ids);
  v.group_member_offsets     = make_span(group_member_offsets);
  v.group_member_vars        = make_span(group_member_vars);
  v.group_member_coeffs      = make_span(group_member_coeffs);
  v.constraint_group_offsets = make_span(constraint_group_offsets);
  v.reverse_group_id         = make_span(reverse_group_id);
  v.n_groups                 = n_groups;
  return v;
}

template <typename i_t, typename f_t>
void clique_group_table_t<i_t, f_t>::build_from_host(problem_t<i_t, f_t>& problem,
                                                     clique_table_t<i_t, f_t>& clique_table)
{
  n_groups = 0;

  const bool has_large_or_addtl =
    !clique_table.first.empty() || !clique_table.addtl_cliques.empty();
  const bool has_small_adj = !clique_table.adj_list_small_cliques.empty();
  if (!has_large_or_addtl && !has_small_adj) {
    CUOPT_LOG_TRACE("clique_group_table_t::build_from_host: no cliques, skipping");
    return;
  }

  const i_t n_vars        = problem.n_variables;
  const i_t n_constraints = problem.n_constraints;
  const i_t nnz           = problem.nnz;
  cuopt_assert(n_vars > 0, "problem has no variables");
  cuopt_assert(n_constraints >= 0, "n_constraints must be non-negative");
  cuopt_assert(nnz >= 0, "nnz must be non-negative");
  cuopt_assert((i_t)clique_table.var_clique_map_first.size() >= n_vars ||
                 clique_table.var_clique_map_first.empty(),
               "var_clique_map_first sized inconsistently with problem");

  // --- Pull problem structure to host ---------------------------------------
  auto& handle_ptr = problem.handle_ptr;
  auto stream      = handle_ptr->get_stream();

  std::vector<i_t> h_offsets         = host_copy(problem.offsets, stream);
  std::vector<i_t> h_variables       = host_copy(problem.variables, stream);
  std::vector<f_t> h_coefficients    = host_copy(problem.coefficients, stream);
  std::vector<i_t> h_reverse_offsets = host_copy(problem.reverse_offsets, stream);
  std::vector<i_t> h_reverse_constr  = host_copy(problem.reverse_constraints, stream);
  std::vector<i_t> h_is_binary       = host_copy(problem.is_binary_variable, stream);

  // --- Materialize ALL explicit cliques (large + addtl) into a flat list ----
  //
  // Each entry is a full clique as a list of positive-literal vertex indices.
  // Cliques that contain any complement literal (vertex >= n_vars) are dropped
  // for this first implementation — the correction math below assumes positive
  // literals.
  //
  // Small cliques from adj_list_small_cliques are handled separately per
  // constraint, because they are stored only as pairwise edges and we need to
  // re-extract maximal cliques from the induced subgraph.

  std::vector<std::vector<i_t>> all_cliques;
  all_cliques.reserve(clique_table.first.size() + clique_table.addtl_cliques.size());

  // Stats counters for input cliques dropped due to complement literals;
  // reported alongside the group-table summary at the end of this function.
  i_t large_cliques_dropped_compl = 0;
  i_t addtl_cliques_dropped_compl = 0;

  auto is_all_positive = [&](const std::vector<i_t>& c) {
    for (i_t v : c)
      if (v >= n_vars) return false;
    return true;
  };

  // Large cliques
  for (auto const& clique : clique_table.first) {
    if (!is_all_positive(clique)) {
      ++large_cliques_dropped_compl;
      continue;
    }
    all_cliques.push_back(clique);
  }
  // Additional cliques = {vertex_idx} ∪ first[clique_idx][start_pos_on_clique:]
  for (auto const& addtl : clique_table.addtl_cliques) {
    auto const& base = clique_table.first[addtl.clique_idx];
    if (addtl.vertex_idx >= n_vars) {
      ++addtl_cliques_dropped_compl;
      continue;
    }
    bool skip = false;
    for (i_t i = addtl.start_pos_on_clique; i < (i_t)base.size(); ++i) {
      if (base[i] >= n_vars) {
        skip = true;
        break;
      }
    }
    if (skip) {
      ++addtl_cliques_dropped_compl;
      continue;
    }
    std::vector<i_t> mat;
    mat.reserve(1 + base.size() - addtl.start_pos_on_clique);
    mat.push_back(addtl.vertex_idx);
    for (i_t i = addtl.start_pos_on_clique; i < (i_t)base.size(); ++i) {
      mat.push_back(base[i]);
    }
    all_cliques.push_back(std::move(mat));
  }

  // Reverse index: var → indices into all_cliques
  std::vector<std::vector<i_t>> var_to_cliques(n_vars);
  for (i_t ci = 0; ci < (i_t)all_cliques.size(); ++ci) {
    for (i_t v : all_cliques[ci]) {
      var_to_cliques[v].push_back(ci);
    }
  }

  // --- Small-clique adjacency helper ----------------------------------------
  //
  // Returns true iff there's an adjacency edge (u, v) in adj_list_small_cliques.
  // Adjacency is symmetric, so we can look up either direction.
  auto has_small_edge = [&](i_t u, i_t v) -> bool {
    auto it = clique_table.adj_list_small_cliques.find(u);
    if (it == clique_table.adj_list_small_cliques.end()) return false;
    return it->second.count(v) > 0;
  };

  // --- Group building (host) -------------------------------------------------
  //
  // For each constraint c, greedily partition its binary variables into
  // non-overlapping clique groups:
  //   1) First, try the explicit large/addtl cliques sorted by size descending.
  //   2) Then, extract small cliques from the pairwise adjacency list via a
  //      greedy maximal-clique heuristic over unassigned members.
  //
  // Groups are emitted in constraint-id order, giving deterministic downstream
  // summation without needing a sort.

  struct group_t {
    i_t cnst_idx;
    std::vector<i_t> vars;
    std::vector<f_t> coeffs;
  };
  std::vector<group_t> groups;
  groups.reserve(all_cliques.size() * 2);

  // Stats: number of groups emitted by phase
  i_t phase1_groups = 0;
  i_t phase2_groups = 0;

  std::unordered_map<i_t, f_t> var_to_coeff;
  std::unordered_set<i_t> relevant_clique_idx;
  std::vector<i_t> sorted_cliques;
  std::unordered_set<i_t> assigned_vars;
  std::vector<i_t> unassigned_binaries;

  auto emit_group_from_members = [&](i_t c, const std::vector<i_t>& members) -> bool {
    if (members.size() < 2) return false;
    group_t g;
    g.cnst_idx = c;
    g.vars.reserve(members.size());
    g.coeffs.reserve(members.size());
    for (i_t v : members) {
      g.vars.push_back(v);
      g.coeffs.push_back(var_to_coeff[v]);
      assigned_vars.insert(v);
    }
    groups.push_back(std::move(g));
    return true;
  };

  for (i_t c = 0; c < n_constraints; ++c) {
    var_to_coeff.clear();
    relevant_clique_idx.clear();
    sorted_cliques.clear();
    assigned_vars.clear();

    // (var → coeff) for this constraint
    i_t row_begin = h_offsets[c];
    i_t row_end   = h_offsets[c + 1];
    for (i_t i = row_begin; i < row_end; ++i) {
      i_t var = h_variables[i];
      var_to_coeff.emplace(var, h_coefficients[i]);
    }

    // (1) Large/addtl cliques: gather those that have any member in this cnst
    for (auto& [var, coeff] : var_to_coeff) {
      if (!h_is_binary[var]) continue;
      for (i_t ci : var_to_cliques[var])
        relevant_clique_idx.insert(ci);
    }
    if (!relevant_clique_idx.empty()) {
      sorted_cliques.assign(relevant_clique_idx.begin(), relevant_clique_idx.end());
      std::sort(sorted_cliques.begin(), sorted_cliques.end(), [&](i_t a, i_t b) {
        return all_cliques[a].size() > all_cliques[b].size();
      });

      for (i_t ci : sorted_cliques) {
        std::vector<i_t> members;
        members.reserve(all_cliques[ci].size());
        for (i_t var : all_cliques[ci]) {
          auto it = var_to_coeff.find(var);
          if (it == var_to_coeff.end()) continue;
          if (assigned_vars.count(var)) continue;
          members.push_back(var);
        }
        if (emit_group_from_members(c, members)) ++phase1_groups;
      }
    }

    // (2) Small cliques: greedy maximal-clique extraction over remaining
    //     unassigned binary variables using adj_list_small_cliques.
    if (has_small_adj) {
      unassigned_binaries.clear();
      for (auto& [var, _coeff] : var_to_coeff) {
        if (h_is_binary[var] && !assigned_vars.count(var)) { unassigned_binaries.push_back(var); }
      }
      // Iterate a snapshot; emit_group_from_members mutates assigned_vars.
      for (i_t seed : unassigned_binaries) {
        if (assigned_vars.count(seed)) continue;
        auto adj_it = clique_table.adj_list_small_cliques.find(seed);
        if (adj_it == clique_table.adj_list_small_cliques.end()) continue;

        // Candidates: binary vars in this constraint, adjacent to seed,
        // still unassigned.
        std::vector<i_t> cands;
        for (i_t w : adj_it->second) {
          if (w == seed) continue;
          if (!var_to_coeff.count(w)) continue;
          if (w >= n_vars) continue;  // skip complement literals
          if (!h_is_binary[w]) continue;
          if (assigned_vars.count(w)) continue;
          cands.push_back(w);
        }
        if (cands.empty()) continue;

        // Greedy extension: add w if adjacent to ALL members already in clique.
        std::vector<i_t> clique{seed};
        for (i_t w : cands) {
          bool ok = true;
          for (i_t u : clique) {
            if (!has_small_edge(u, w)) {
              ok = false;
              break;
            }
          }
          if (ok) clique.push_back(w);
        }
        if (emit_group_from_members(c, clique)) ++phase2_groups;
      }
    }
  }

  n_groups = static_cast<i_t>(groups.size());
  if (n_groups == 0) {
    CUOPT_LOG_TRACE(
      "clique_group_table_t::build_from_host: no (cnst, clique) pairs with ≥2 members");
    return;
  }

  // --- Flatten into CSR arrays (still host) ---------------------------------
  i_t total_members = 0;
  for (auto const& g : groups)
    total_members += static_cast<i_t>(g.vars.size());

  std::vector<i_t> h_group_constraint_ids(n_groups);
  std::vector<i_t> h_group_member_offsets(n_groups + 1, 0);
  std::vector<i_t> h_group_member_vars(total_members);
  std::vector<f_t> h_group_member_coeffs(total_members);

  i_t member_cursor = 0;
  for (i_t g = 0; g < n_groups; ++g) {
    h_group_constraint_ids[g] = groups[g].cnst_idx;
    h_group_member_offsets[g] = member_cursor;
    for (std::size_t k = 0; k < groups[g].vars.size(); ++k) {
      h_group_member_vars[member_cursor]   = groups[g].vars[k];
      h_group_member_coeffs[member_cursor] = groups[g].coeffs[k];
      ++member_cursor;
    }
  }
  h_group_member_offsets[n_groups] = total_members;

  // constraint_group_offsets: groups are sorted by constraint_id (we iterated
  // c in order). Compute counts then prefix-sum.
  std::vector<i_t> h_constraint_group_offsets(n_constraints + 1, 0);
  for (i_t g = 0; g < n_groups; ++g) {
    h_constraint_group_offsets[h_group_constraint_ids[g] + 1]++;
  }
  for (i_t c = 1; c <= n_constraints; ++c) {
    h_constraint_group_offsets[c] += h_constraint_group_offsets[c - 1];
  }

  // reverse_group_id: one entry per nnz reverse-CSR slot, initialized to -1.
  // For each group g, for each member var, find the position in the reverse
  // CSR where reverse_constraints[pos] == g.cnst_idx, and write g there.
  std::vector<i_t> h_reverse_group_id(nnz, -1);
  for (i_t g = 0; g < n_groups; ++g) {
    i_t c = h_group_constraint_ids[g];
    for (i_t k = h_group_member_offsets[g]; k < h_group_member_offsets[g + 1]; ++k) {
      i_t var    = h_group_member_vars[k];
      i_t rv_beg = h_reverse_offsets[var];
      i_t rv_end = h_reverse_offsets[var + 1];
      for (i_t r = rv_beg; r < rv_end; ++r) {
        if (h_reverse_constr[r] == c) {
          h_reverse_group_id[r] = g;
          break;
        }
      }
    }
  }

  // --- Host-side invariant checks -------------------------------------------
  cuopt_assert((i_t)h_group_constraint_ids.size() == n_groups, "group_constraint_ids bad size");
  cuopt_assert((i_t)h_group_member_offsets.size() == n_groups + 1, "group_member_offsets bad size");
  cuopt_assert((i_t)h_group_member_vars.size() == total_members, "group_member_vars bad size");
  cuopt_assert((i_t)h_group_member_coeffs.size() == total_members, "group_member_coeffs bad size");
  cuopt_assert(h_group_member_offsets.front() == 0, "group_member_offsets[0] != 0");
  cuopt_assert(h_group_member_offsets.back() == total_members,
               "group_member_offsets[n_groups] != total_members");
  cuopt_assert((i_t)h_constraint_group_offsets.size() == n_constraints + 1,
               "constraint_group_offsets bad size");
  cuopt_assert(h_constraint_group_offsets.front() == 0, "constraint_group_offsets[0] != 0");
  cuopt_assert(h_constraint_group_offsets.back() == n_groups,
               "constraint_group_offsets[n_constraints] != n_groups");
  cuopt_assert((i_t)h_reverse_group_id.size() == nnz, "reverse_group_id bad size");
  // Check monotonicity of member offsets and membership of group ids
  for (i_t g = 0; g < n_groups; ++g) {
    cuopt_assert(h_group_member_offsets[g] <= h_group_member_offsets[g + 1],
                 "group_member_offsets not monotonic");
    cuopt_assert(h_group_member_offsets[g + 1] - h_group_member_offsets[g] >= 2,
                 "each group must have >=2 members");
    cuopt_assert(h_group_constraint_ids[g] >= 0 && h_group_constraint_ids[g] < n_constraints,
                 "group constraint id out of range");
    for (i_t k = h_group_member_offsets[g]; k < h_group_member_offsets[g + 1]; ++k) {
      cuopt_assert(h_group_member_vars[k] >= 0 && h_group_member_vars[k] < n_vars,
                   "group member var out of range");
    }
  }
  // Groups are emitted in ascending constraint order
  for (i_t g = 1; g < n_groups; ++g) {
    cuopt_assert(h_group_constraint_ids[g - 1] <= h_group_constraint_ids[g],
                 "groups not sorted by constraint id");
  }

  // --- Copy to device --------------------------------------------------------
  group_constraint_ids     = device_copy(h_group_constraint_ids, stream);
  group_member_offsets     = device_copy(h_group_member_offsets, stream);
  group_member_vars        = device_copy(h_group_member_vars, stream);
  group_member_coeffs      = device_copy(h_group_member_coeffs, stream);
  constraint_group_offsets = device_copy(h_constraint_group_offsets, stream);
  reverse_group_id         = device_copy(h_reverse_group_id, stream);

  handle_ptr->sync_stream();

  // --- Stats ---------------------------------------------------------------
  //
  // Print a summary of the built group table so the user can see, at a glance:
  //   - how many (cnst, clique) groups survived and how members are distributed
  //   - how much coverage we got per constraint
  //   - how many source cliques came from each kind of clique table entry
  //   - whether explicit cliques or the small-adj heuristic drove the result
  {
    i_t min_size = std::numeric_limits<i_t>::max();
    i_t max_size = 0;
    i_t size_eq2 = 0, size_3_5 = 0, size_6_20 = 0, size_ge21 = 0;
    for (i_t g = 0; g < n_groups; ++g) {
      i_t sz   = h_group_member_offsets[g + 1] - h_group_member_offsets[g];
      min_size = std::min(min_size, sz);
      max_size = std::max(max_size, sz);
      if (sz == 2)
        ++size_eq2;
      else if (sz <= 5)
        ++size_3_5;
      else if (sz <= 20)
        ++size_6_20;
      else
        ++size_ge21;
    }
    const double avg_size = (double)total_members / (double)n_groups;

    i_t constraints_with_groups = 0;
    i_t max_groups_per_cnst     = 0;
    for (i_t c = 0; c < n_constraints; ++c) {
      i_t ng = h_constraint_group_offsets[c + 1] - h_constraint_group_offsets[c];
      if (ng > 0) ++constraints_with_groups;
      max_groups_per_cnst = std::max(max_groups_per_cnst, ng);
    }

    CUOPT_LOG_INFO(
      "clique_group_table_t::build_from_host: n_groups=%d total_members=%d avg_size=%.2f "
      "(min=%d max=%d) | size buckets: =2:%d, 3-5:%d, 6-20:%d, >=21:%d | "
      "constraints_with_groups=%d/%d max_groups/cnst=%d | "
      "phase1 (explicit cliques)=%d phase2 (small-adj)=%d",
      n_groups,
      total_members,
      avg_size,
      (n_groups > 0 ? min_size : 0),
      max_size,
      size_eq2,
      size_3_5,
      size_6_20,
      size_ge21,
      constraints_with_groups,
      n_constraints,
      max_groups_per_cnst,
      phase1_groups,
      phase2_groups);
    CUOPT_LOG_INFO(
      "clique_group_table_t::build_from_host: sources: large=%zu (dropped compl=%d), "
      "addtl=%zu (dropped compl=%d), all_positive_materialized=%zu, small_adj_vars=%zu",
      clique_table.first.size(),
      large_cliques_dropped_compl,
      clique_table.addtl_cliques.size(),
      addtl_cliques_dropped_compl,
      all_cliques.size(),
      clique_table.adj_list_small_cliques.size());
  }
}

#if MIP_INSTANTIATE_FLOAT
template class clique_group_table_t<int, float>;
#endif
#if MIP_INSTANTIATE_DOUBLE
template class clique_group_table_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
