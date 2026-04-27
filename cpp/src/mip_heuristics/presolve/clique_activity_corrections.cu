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
#include <numeric>
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
  v.reverse_member_sign      = make_span(reverse_member_sign);
  v.n_groups                 = n_groups;
  return v;
}

template <typename i_t, typename f_t>
void clique_group_table_t<i_t, f_t>::build_from_host(
  problem_t<i_t, f_t>& problem,
  const std::vector<i_t>& primary_reverse_original_ids,
  clique_table_t<i_t, f_t>& clique_table)
{
  n_groups = 0;

  const bool has_large_or_addtl =
    !clique_table.first.empty() || !clique_table.addtl_cliques.empty();
  const bool has_small_adj = !clique_table.small_clique_adj.indices.empty();
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
  cuopt_assert(clique_table.var_clique_first.n_keys() >= n_vars ||
                 clique_table.var_clique_first.indices.empty(),
               "var_clique_first sized inconsistently with problem");

  // Build clique-build-space → pb-space map. Identity for root /
  // +objective-cut; remap via primary_reverse_original_ids[problem.original_ids]
  // for fixed sub-problems. -1 means the build-var was fixed/removed.
  const i_t n_build_vars = clique_table.n_variables;
  const bool ids_match   = (n_build_vars == n_vars);
  std::vector<i_t> build_var_to_pb;
  if (ids_match) {
    build_var_to_pb.resize(n_build_vars);
    std::iota(build_var_to_pb.begin(), build_var_to_pb.end(), i_t{0});
  } else {
    cuopt_assert(
      !primary_reverse_original_ids.empty(),
      "clique_table.n_variables differs from problem.n_variables but the caller provided no "
      "primary_reverse_original_ids — cannot safely remap clique-build ids to this sub-problem.");
    cuopt_assert((i_t)problem.original_ids.size() == n_vars,
                 "problem.original_ids must be sized to problem.n_variables to run clique-aware "
                 "propagation on a sub-problem");
    const i_t n_input = static_cast<i_t>(primary_reverse_original_ids.size());
    build_var_to_pb.assign(n_build_vars, -1);
    for (i_t p = 0; p < n_vars; ++p) {
      i_t input_idx = problem.original_ids[p];
      if (input_idx < 0 || input_idx >= n_input) continue;
      i_t build_idx = primary_reverse_original_ids[input_idx];
      if (build_idx < 0 || build_idx >= n_build_vars) continue;
      cuopt_assert(build_var_to_pb[build_idx] == -1,
                   "Duplicate forward remap entry for a clique-build var");
      build_var_to_pb[build_idx] = p;
    }
  }

  // Remap literal vertex from clique-build to pb literal space; -1 if removed.
  auto remap_vertex_to_pb = [&](i_t vertex) -> i_t {
    cuopt_assert(vertex >= 0 && vertex < 2 * n_build_vars,
                 "vertex out of clique-build literal range");
    const bool neg      = (vertex >= n_build_vars);
    const i_t build_var = neg ? (vertex - n_build_vars) : vertex;
    const i_t pb_var    = build_var_to_pb[build_var];
    if (pb_var < 0) return -1;
    return neg ? (n_vars + pb_var) : pb_var;
  };

  // --- Pull problem structure to host ---------------------------------------
  auto& handle_ptr = problem.handle_ptr;
  auto stream      = handle_ptr->get_stream();

  std::vector<i_t> h_offsets         = host_copy(problem.offsets, stream);
  std::vector<i_t> h_variables       = host_copy(problem.variables, stream);
  std::vector<f_t> h_coefficients    = host_copy(problem.coefficients, stream);
  std::vector<i_t> h_reverse_offsets = host_copy(problem.reverse_offsets, stream);
  std::vector<i_t> h_reverse_constr  = host_copy(problem.reverse_constraints, stream);
  std::vector<i_t> h_is_binary       = host_copy(problem.is_binary_variable, stream);

  // Materialize explicit (large + addtl) cliques as flat literal-vertex lists.
  // Small cliques are handled separately per constraint via the pairwise CSR.
  // Storing the effective literal coeff b_j = sign_j * a_j lets the kernel
  // treat positive and complement literals uniformly.
  std::vector<std::vector<i_t>> all_cliques;
  all_cliques.reserve(clique_table.first.size() + clique_table.addtl_cliques.size());

  auto underlying_var = [&](i_t vertex) -> i_t {
    return vertex < n_vars ? vertex : (vertex - n_vars);
  };
  auto literal_sign = [&](i_t vertex) -> i_t { return vertex < n_vars ? i_t{+1} : i_t{-1}; };

  // Remap clique to pb literal space; drop members removed since clique-build.
  // Cliques with <2 surviving members are dropped (trivially non-tightening).
  auto push_remapped_clique = [&](const std::vector<i_t>& src) {
    std::vector<i_t> dst;
    dst.reserve(src.size());
    for (i_t v_build : src) {
      i_t v_pb = remap_vertex_to_pb(v_build);
      if (v_pb >= 0) dst.push_back(v_pb);
    }
    if (dst.size() >= 2) all_cliques.push_back(std::move(dst));
  };

  for (auto const& clique : clique_table.first) {
    push_remapped_clique(clique);
  }
  // Each addtl = {vertex_idx} ∪ first[clique_idx][start_pos_on_clique:]
  for (auto const& addtl : clique_table.addtl_cliques) {
    auto const& base = clique_table.first[addtl.clique_idx];
    std::vector<i_t> mat;
    mat.reserve(1 + base.size() - addtl.start_pos_on_clique);
    mat.push_back(addtl.vertex_idx);
    for (i_t i = addtl.start_pos_on_clique; i < (i_t)base.size(); ++i) {
      mat.push_back(base[i]);
    }
    push_remapped_clique(mat);
  }

  // Reverse index: underlying var → indices into all_cliques (pb space).
  std::vector<std::vector<i_t>> var_to_cliques(n_vars);
  for (i_t ci = 0; ci < (i_t)all_cliques.size(); ++ci) {
    for (i_t vertex : all_cliques[ci]) {
      var_to_cliques[underlying_var(vertex)].push_back(ci);
    }
  }

  // Small-clique adjacency: query the CSR directly when id spaces match,
  // otherwise build a pb-space hash shadow (sub-problem path is cold).
  std::unordered_map<i_t, std::unordered_set<i_t>> adj_list_shadow;
  if (!ids_match && has_small_adj) {
    const auto& sc    = clique_table.small_clique_adj;
    const i_t sc_keys = sc.n_keys();
    for (i_t u_build = 0; u_build < sc_keys; ++u_build) {
      const i_t u_slice = sc.slice_size(u_build);
      if (u_slice == 0) continue;
      i_t u_pb = remap_vertex_to_pb(u_build);
      if (u_pb < 0) continue;
      auto& set = adj_list_shadow[u_pb];
      for (const i_t* it = sc.slice_begin(u_build); it != sc.slice_end(u_build); ++it) {
        i_t v_pb = remap_vertex_to_pb(*it);
        if (v_pb >= 0) set.insert(v_pb);
      }
      if (set.empty()) adj_list_shadow.erase(u_pb);
    }
  }
  const bool has_small_adj_for_build = ids_match ? has_small_adj : !adj_list_shadow.empty();

  auto has_small_edge = [&](i_t u, i_t v) -> bool {
    if (ids_match) { return clique_table.small_clique_adj.slice_contains(u, v); }
    auto it = adj_list_shadow.find(u);
    if (it == adj_list_shadow.end()) return false;
    return it->second.count(v) > 0;
  };

  // Copy neighbors of seed_lit into `out`; returns false if seed has no
  // neighbors (CSR slice for ids_match, unordered_set otherwise).
  auto collect_neighbors = [&](i_t seed_lit, std::vector<i_t>& out) -> bool {
    out.clear();
    if (ids_match) {
      const auto& sc = clique_table.small_clique_adj;
      const i_t k    = sc.slice_size(seed_lit);
      if (k == 0) return false;
      out.assign(sc.slice_begin(seed_lit), sc.slice_end(seed_lit));
      return true;
    }
    auto it = adj_list_shadow.find(seed_lit);
    if (it == adj_list_shadow.end() || it->second.empty()) return false;
    out.assign(it->second.begin(), it->second.end());
    return true;
  };

  // For each constraint, greedily partition its binary vars into groups:
  // (1) explicit large/addtl cliques first, sorted largest-first;
  // (2) then small-adj greedy maximal-clique extraction over unassigned binaries.
  // Emit in constraint-id order for deterministic downstream summation.

  struct group_t {
    i_t cnst_idx;
    std::vector<i_t> vars;
    std::vector<f_t> coeffs;
    std::vector<i_t> signs;  // +/-1, parallel to vars/coeffs
  };
  std::vector<group_t> groups;
  groups.reserve(all_cliques.size() * 2);

  i_t phase1_groups = 0;
  i_t phase2_groups = 0;

  std::unordered_map<i_t, f_t> var_to_coeff;
  std::unordered_set<i_t> relevant_clique_idx;
  std::vector<i_t> sorted_cliques;
  std::unordered_set<i_t> assigned_vars;
  std::vector<i_t> unassigned_binaries;

  // Emit a group, storing the effective literal coeff sign * a_var.
  auto emit_group_from_members =
    [&](i_t c, const std::vector<i_t>& members, const std::vector<i_t>& signs) -> bool {
    if (members.size() < 2) return false;
    cuopt_assert(members.size() == signs.size(), "members/signs size mismatch");
    group_t g;
    g.cnst_idx = c;
    g.vars.reserve(members.size());
    g.coeffs.reserve(members.size());
    g.signs.reserve(members.size());
    for (std::size_t k = 0; k < members.size(); ++k) {
      i_t v = members[k];
      i_t s = signs[k];
      g.vars.push_back(v);
      g.coeffs.push_back(static_cast<f_t>(s) * var_to_coeff[v]);
      g.signs.push_back(s);
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

    i_t row_begin = h_offsets[c];
    i_t row_end   = h_offsets[c + 1];
    for (i_t i = row_begin; i < row_end; ++i) {
      i_t var = h_variables[i];
      var_to_coeff.emplace(var, h_coefficients[i]);
    }

    // (1) Large/addtl cliques touching this constraint, largest-first.
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
        std::vector<i_t> signs;
        members.reserve(all_cliques[ci].size());
        signs.reserve(all_cliques[ci].size());
        for (i_t vertex : all_cliques[ci]) {
          i_t var  = underlying_var(vertex);
          i_t sign = literal_sign(vertex);
          auto it  = var_to_coeff.find(var);
          if (it == var_to_coeff.end()) continue;
          if (!h_is_binary[var]) continue;
          if (assigned_vars.count(var)) continue;
          // Reject a clique containing both x and ~x (tautology).
          bool already_in = false;
          for (i_t uv : members) {
            if (uv == var) {
              already_in = true;
              break;
            }
          }
          if (already_in) continue;
          members.push_back(var);
          signs.push_back(sign);
        }
        if (emit_group_from_members(c, members, signs)) ++phase1_groups;
      }
    }

    // (2) Small-adj greedy maximal-clique extraction over remaining
    // unassigned binaries. Nodes are literals; each underlying var appears
    // at most once per group.
    if (has_small_adj_for_build) {
      unassigned_binaries.clear();
      for (auto& [var, _coeff] : var_to_coeff) {
        if (h_is_binary[var] && !assigned_vars.count(var)) { unassigned_binaries.push_back(var); }
      }
      for (i_t seed_var : unassigned_binaries) {
        if (assigned_vars.count(seed_var)) continue;
        // Try both polarities; the second short-circuits if the first emitted.
        for (i_t seed_sign : {i_t{+1}, i_t{-1}}) {
          if (assigned_vars.count(seed_var)) break;
          i_t seed_lit = (seed_sign == +1) ? seed_var : (n_vars + seed_var);
          std::vector<i_t> seed_neighbors;
          if (!collect_neighbors(seed_lit, seed_neighbors)) continue;

          std::vector<i_t> cand_lits;
          for (i_t w_lit : seed_neighbors) {
            if (w_lit == seed_lit) continue;
            i_t w_var = underlying_var(w_lit);
            if (!var_to_coeff.count(w_var)) continue;
            if (!h_is_binary[w_var]) continue;
            if (assigned_vars.count(w_var)) continue;
            if (w_var == seed_var) continue;  // reject x / ~x tautology
            cand_lits.push_back(w_lit);
          }
          if (cand_lits.empty()) continue;

          std::vector<i_t> clique_lits{seed_lit};
          std::vector<i_t> clique_vars{seed_var};
          std::vector<i_t> clique_signs{seed_sign};
          for (i_t w_lit : cand_lits) {
            i_t w_var  = underlying_var(w_lit);
            i_t w_sign = literal_sign(w_lit);
            bool dup   = false;
            for (i_t uv : clique_vars) {
              if (uv == w_var) {
                dup = true;
                break;
              }
            }
            if (dup) continue;
            bool ok = true;
            for (i_t u_lit : clique_lits) {
              if (!has_small_edge(u_lit, w_lit)) {
                ok = false;
                break;
              }
            }
            if (ok) {
              clique_lits.push_back(w_lit);
              clique_vars.push_back(w_var);
              clique_signs.push_back(w_sign);
            }
          }
          if (emit_group_from_members(c, clique_vars, clique_signs)) ++phase2_groups;
        }
      }
    }
  }

  n_groups = static_cast<i_t>(groups.size());
  if (n_groups == 0) {
    CUOPT_LOG_TRACE(
      "clique_group_table_t::build_from_host: no (cnst, clique) pairs with ≥2 members");
    return;
  }

  // Flatten into host CSR arrays.
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

  // constraint_group_offsets: count + prefix-sum (groups already sorted by c).
  std::vector<i_t> h_constraint_group_offsets(n_constraints + 1, 0);
  for (i_t g = 0; g < n_groups; ++g) {
    h_constraint_group_offsets[h_group_constraint_ids[g] + 1]++;
  }
  for (i_t c = 1; c <= n_constraints; ++c) {
    h_constraint_group_offsets[c] += h_constraint_group_offsets[c - 1];
  }

  // reverse_group_id / reverse_member_sign: one entry per nnz reverse-CSR slot.
  // group_id == -1 for non-members (sign 0); otherwise sign is +/-1.
  std::vector<i_t> h_reverse_group_id(nnz, -1);
  std::vector<i_t> h_reverse_member_sign(nnz, 0);
  for (i_t g = 0; g < n_groups; ++g) {
    i_t c = h_group_constraint_ids[g];
    for (i_t k = h_group_member_offsets[g]; k < h_group_member_offsets[g + 1]; ++k) {
      i_t var    = h_group_member_vars[k];
      i_t local  = k - h_group_member_offsets[g];
      i_t sign   = groups[g].signs[local];
      i_t rv_beg = h_reverse_offsets[var];
      i_t rv_end = h_reverse_offsets[var + 1];
      for (i_t r = rv_beg; r < rv_end; ++r) {
        if (h_reverse_constr[r] == c) {
          h_reverse_group_id[r]    = g;
          h_reverse_member_sign[r] = sign;
          break;
        }
      }
    }
  }

  // Host-side invariant checks.
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
  cuopt_assert((i_t)h_reverse_member_sign.size() == nnz, "reverse_member_sign bad size");
  for (i_t r = 0; r < nnz; ++r) {
    if (h_reverse_group_id[r] == -1) {
      cuopt_assert(h_reverse_member_sign[r] == 0,
                   "reverse_member_sign must be 0 when not a member");
    } else {
      cuopt_assert(h_reverse_member_sign[r] == 1 || h_reverse_member_sign[r] == -1,
                   "reverse_member_sign must be +/-1 when a member");
    }
  }
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
  for (i_t g = 1; g < n_groups; ++g) {
    cuopt_assert(h_group_constraint_ids[g - 1] <= h_group_constraint_ids[g],
                 "groups not sorted by constraint id");
  }

  group_constraint_ids     = device_copy(h_group_constraint_ids, stream);
  group_member_offsets     = device_copy(h_group_member_offsets, stream);
  group_member_vars        = device_copy(h_group_member_vars, stream);
  group_member_coeffs      = device_copy(h_group_member_coeffs, stream);
  constraint_group_offsets = device_copy(h_constraint_group_offsets, stream);
  reverse_group_id         = device_copy(h_reverse_group_id, stream);
  reverse_member_sign      = device_copy(h_reverse_member_sign, stream);

  handle_ptr->sync_stream();

  // Summary stats.
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
      "clique_group_table_t::build_from_host: sources: large=%zu, addtl=%zu, "
      "materialized=%zu, small_adj_edges=%zu",
      clique_table.first.size(),
      clique_table.addtl_cliques.size(),
      all_cliques.size(),
      clique_table.small_clique_adj.indices.size());
  }
}

#if MIP_INSTANTIATE_FLOAT
template class clique_group_table_t<int, float>;
#endif
#if MIP_INSTANTIATE_DOUBLE
template class clique_group_table_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
