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

  // --- Clique-build idx space vs. problem idx space -------------------------
  //
  // The clique_table encodes literal vertices in [0, 2 * M) where
  // M = clique_table.n_variables = problem.n_variables at the moment
  // find_initial_cliques ran (the "clique-build snapshot", in M-space).
  // Heuristics may drive this function against:
  //   (a) the root problem (M == n_vars): identity map; or
  //   (b) problem_with_objective_cut (same n_vars, extra row): identity; or
  //   (c) a further-reduced sub-problem from solution_t::fix_variables
  //       used by fp_recombiner / bp_recombiner / sub_mip. Its
  //       `original_ids` lives in user-input N-space (not M-space) because
  //       fix_variables composes with the parent problem's original_ids.
  //
  // For (c) we must translate clique-build vertex ids into the sub-problem's
  // own idx space via the two-step chain
  //     M-idx  ← clique_table.build_reverse_original_ids[N-idx] ←
  //     N-idx  ← problem.original_ids[sub-idx] ← sub-idx
  // and then drop clique members whose underlying var was fixed/removed.
  //
  // Builds a forward map `build_var_to_pb[m] = p` (or -1 if m was removed)
  // so the rest of this function can operate entirely in pb-space and the
  // closures below use n_vars without caring about M.
  const i_t n_build_vars = clique_table.n_variables;
  const bool ids_match   = (n_build_vars == n_vars);
  std::vector<i_t> build_var_to_pb;
  if (ids_match) {
    // Fast path: same id space (root / +objective-cut problem). Identity.
    build_var_to_pb.resize(n_build_vars);
    std::iota(build_var_to_pb.begin(), build_var_to_pb.end(), i_t{0});
  } else {
    // Sub-problem path. Requires the snapshot recorded in
    // diversity_manager_t::run_presolve when the clique table was attached.
    cuopt_assert(
      !clique_table.build_reverse_original_ids.empty(),
      "clique_table.n_variables differs from problem.n_variables but the table has no "
      "build_reverse_original_ids snapshot — the clique table was attached without recording its "
      "id space; cannot safely remap to this sub-problem.");
    cuopt_assert((i_t)problem.original_ids.size() == n_vars,
                 "problem.original_ids must be sized to problem.n_variables to run clique-aware "
                 "propagation on a sub-problem");
    const i_t n_input = static_cast<i_t>(clique_table.build_reverse_original_ids.size());
    build_var_to_pb.assign(n_build_vars, -1);
    for (i_t p = 0; p < n_vars; ++p) {
      i_t input_idx = problem.original_ids[p];
      if (input_idx < 0 || input_idx >= n_input) continue;
      i_t build_idx = clique_table.build_reverse_original_ids[input_idx];
      if (build_idx < 0 || build_idx >= n_build_vars) continue;
      // fix_variables is injective after restriction, so each clique-build
      // var maps to at most one pb var.
      cuopt_assert(build_var_to_pb[build_idx] == -1,
                   "Duplicate forward remap entry for a clique-build var");
      build_var_to_pb[build_idx] = p;
    }
  }

  // Translate a literal vertex id from clique-build space to pb literal
  // space. Returns -1 if the underlying var is not in pb.
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

  // --- Materialize ALL explicit cliques (large + addtl) into a flat list ----
  //
  // Each entry is a full clique as a list of literal vertex indices in the
  // conflict graph: `v < n_vars` is the positive literal of var `v`, and
  // `v >= n_vars` is the complement literal of var `v - n_vars`. Cliques may
  // freely mix positive and complement literals.
  //
  // Small cliques from adj_list_small_cliques are handled separately per
  // constraint, because they are stored only as pairwise edges and we need to
  // re-extract maximal cliques from the induced subgraph.
  //
  // Clique-aware activity correction with mixed literals:
  //   For a clique {L_j} (at most one literal true), substitute x_j = z_j for
  //   positive literals and x_j = 1 - z_j for complement literals where z_j is
  //   the literal indicator. The constraint's row then contributes
  //     (const) + sum_j b_j z_j,   with   b_j := sign_j * a_j
  //   where sign_j = +1 for positive literal, -1 for complement.
  //   The constant offset `sum_{j ∈ Q-} a_j` absorbs into min/max_activity
  //   uniformly, and — critically — cancels out of the stock-minus-true
  //   correction. So storing the effective literal coefficient `b_j` as the
  //   member coeff lets the existing kernel compute the right correction with
  //   no changes.

  std::vector<std::vector<i_t>> all_cliques;
  all_cliques.reserve(clique_table.first.size() + clique_table.addtl_cliques.size());

  auto underlying_var = [&](i_t vertex) -> i_t {
    return vertex < n_vars ? vertex : (vertex - n_vars);
  };
  auto literal_sign = [&](i_t vertex) -> i_t { return vertex < n_vars ? i_t{+1} : i_t{-1}; };

  // Remap source cliques into pb literal space. Members whose underlying var
  // was removed between clique-build time and now are dropped; cliques that
  // shrink below 2 members are discarded wholesale (trivially non-tightening).
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
  // Additional cliques = {vertex_idx} ∪ first[clique_idx][start_pos_on_clique:]
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

  // Reverse index: underlying var → indices into all_cliques. All vertex ids
  // in all_cliques are already in pb literal space after the remap above.
  std::vector<std::vector<i_t>> var_to_cliques(n_vars);
  for (i_t ci = 0; ci < (i_t)all_cliques.size(); ++ci) {
    for (i_t vertex : all_cliques[ci]) {
      var_to_cliques[underlying_var(vertex)].push_back(ci);
    }
  }

  // --- Small-clique adjacency (remapped if needed) --------------------------
  //
  // When the id spaces match, we query clique_table.adj_list_small_cliques
  // directly to avoid copying. Otherwise we build a pb-space shadow so the
  // rest of the loop below needs no further translation.
  std::unordered_map<i_t, std::unordered_set<i_t>> adj_list_shadow;
  if (!ids_match && has_small_adj) {
    for (auto const& [u_build, neighbors_build] : clique_table.adj_list_small_cliques) {
      i_t u_pb = remap_vertex_to_pb(u_build);
      if (u_pb < 0) continue;
      auto& set = adj_list_shadow[u_pb];
      for (i_t v_build : neighbors_build) {
        i_t v_pb = remap_vertex_to_pb(v_build);
        if (v_pb >= 0) set.insert(v_pb);
      }
      if (set.empty()) adj_list_shadow.erase(u_pb);
    }
  }
  const auto& adj_for_build = ids_match ? clique_table.adj_list_small_cliques : adj_list_shadow;
  const bool has_small_adj_for_build = ids_match ? has_small_adj : !adj_list_shadow.empty();

  // Returns true iff there's an adjacency edge (u, v) in adj_for_build.
  // Adjacency is symmetric, so either direction works.
  auto has_small_edge = [&](i_t u, i_t v) -> bool {
    auto it = adj_for_build.find(u);
    if (it == adj_for_build.end()) return false;
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
    // Parallel to vars/coeffs: +1 for positive literal member, -1 for
    // complement. Consumed only when building reverse_member_sign below.
    std::vector<i_t> signs;
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

  // Emit a group given the underlying-var list and the parallel literal-sign
  // list (+1 for positive, -1 for complement). We store the effective literal
  // coefficient `sign * a_var` rather than the raw constraint coefficient — see
  // the derivation at the top of this function.
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
          // Guard against a clique that accidentally contains both literals of
          // the same var (trivial tautology, non-tightening): keep only the
          // first occurrence.
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

    // (2) Small cliques: greedy maximal-clique extraction over remaining
    //     unassigned binary variables using adj_list_small_cliques. Nodes in
    //     the adjacency graph are literals (positive or complement), and an
    //     edge between two literals means "at most one of them is true" — so
    //     the greedy walk works on literals directly, with the extra
    //     bookkeeping that each underlying var may appear at most once per
    //     group.
    if (has_small_adj_for_build) {
      unassigned_binaries.clear();
      for (auto& [var, _coeff] : var_to_coeff) {
        if (h_is_binary[var] && !assigned_vars.count(var)) { unassigned_binaries.push_back(var); }
      }
      // Iterate a snapshot; emit_group_from_members mutates assigned_vars.
      for (i_t seed_var : unassigned_binaries) {
        if (assigned_vars.count(seed_var)) continue;
        // Try both literal polarities as the seed. If the positive-literal
        // seed produces a group, `seed_var` becomes assigned and the second
        // attempt short-circuits. This is the cheapest way to let the greedy
        // search find cliques anchored at either ~x or x.
        for (i_t seed_sign : {i_t{+1}, i_t{-1}}) {
          if (assigned_vars.count(seed_var)) break;
          i_t seed_lit = (seed_sign == +1) ? seed_var : (n_vars + seed_var);
          auto adj_it  = adj_for_build.find(seed_lit);
          if (adj_it == adj_for_build.end()) continue;

          // Candidate literals adjacent to the seed whose underlying var is in
          // this constraint, binary, and still unassigned.
          std::vector<i_t> cand_lits;
          for (i_t w_lit : adj_it->second) {
            if (w_lit == seed_lit) continue;
            i_t w_var = underlying_var(w_lit);
            if (!var_to_coeff.count(w_var)) continue;
            if (!h_is_binary[w_var]) continue;
            if (assigned_vars.count(w_var)) continue;
            if (w_var == seed_var) continue;  // reject x / ~x pair (tautology)
            cand_lits.push_back(w_lit);
          }
          if (cand_lits.empty()) continue;

          // Greedy extension on literals, tracking underlying vars to enforce
          // the "≤1 appearance per group" invariant.
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
  // reverse_member_sign: parallel array initialized to 0 (not a member); for
  // each member, we record its literal sign so the per-var adjustment in
  // update_bounds_per_cnst_cliq can work on b_v = sign_v * a_v instead of the
  // raw a_v. See derivation in bounds_update_helpers.cuh.
  // For each group g, for each member var, find the position in the reverse
  // CSR where reverse_constraints[pos] == g.cnst_idx, and write g / sign there.
  std::vector<i_t> h_reverse_group_id(nnz, -1);
  std::vector<i_t> h_reverse_member_sign(nnz, 0);
  for (i_t g = 0; g < n_groups; ++g) {
    i_t c = h_group_constraint_ids[g];
    for (i_t k = h_group_member_offsets[g]; k < h_group_member_offsets[g + 1]; ++k) {
      i_t var = h_group_member_vars[k];
      // Recover the literal sign from the group's dense arrays. groups[g] is
      // still in host memory; we iterate (k - member_offsets[g]) within it.
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
  cuopt_assert((i_t)h_reverse_member_sign.size() == nnz, "reverse_member_sign bad size");
  // reverse_member_sign must match reverse_group_id: 0 iff group_id == -1;
  // otherwise ±1.
  for (i_t r = 0; r < nnz; ++r) {
    if (h_reverse_group_id[r] == -1) {
      cuopt_assert(h_reverse_member_sign[r] == 0,
                   "reverse_member_sign must be 0 when not a member");
    } else {
      cuopt_assert(h_reverse_member_sign[r] == 1 || h_reverse_member_sign[r] == -1,
                   "reverse_member_sign must be +/-1 when a member");
    }
  }
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
  reverse_member_sign      = device_copy(h_reverse_member_sign, stream);

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
      "clique_group_table_t::build_from_host: sources: large=%zu, addtl=%zu, "
      "materialized=%zu, small_adj_vars=%zu",
      clique_table.first.size(),
      clique_table.addtl_cliques.size(),
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
