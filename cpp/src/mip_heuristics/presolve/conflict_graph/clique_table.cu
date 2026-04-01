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

#define DEBUG_KNAPSACK_CONSTRAINTS 0

#include "clique_table.cuh"

#include <algorithm>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/sparse_vector.hpp>
#include <limits>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/utils.cuh>
#include <utilities/logger.hpp>
#include <utilities/macros.cuh>
#include <utilities/timer.hpp>

namespace cuopt::linear_programming::detail {

// do constraints with only binary variables.
template <typename i_t, typename f_t>
void find_cliques_from_constraint(const knapsack_constraint_t<i_t, f_t>& kc,
                                  clique_table_t<i_t, f_t>& clique_table,
                                  cuopt::timer_t& timer)
{
  i_t size = kc.entries.size();
  cuopt_assert(size > 1, "Constraint has not enough variables");
  if (kc.entries[size - 1].val + kc.entries[size - 2].val <= kc.rhs) { return; }

  std::vector<i_t> clique;
  i_t k = size - 1;
  // find the first clique, which is the largest
  // FIXME: do binary search
  // require k >= 1 so kc.entries[k-1] is always valid
  while (k >= 1 && kc.entries[k].val + kc.entries[k - 1].val > kc.rhs) {
    k--;
  }
  for (i_t idx = k; idx < size; idx++) {
    clique.push_back(kc.entries[idx].col);
  }
  clique_table.first.push_back(clique);
  const i_t original_clique_start_idx = k;
  // find the additional cliques
  k--;
  while (k >= 0) {
    if (timer.check_time_limit()) { return; }
    f_t curr_val = kc.entries[k].val;
    i_t curr_col = kc.entries[k].col;
    // do a binary search in the clique coefficients to find f, such that coeff_k + coeff_f > rhs
    // this means that we get a subset of the original clique and extend it with a variable
    f_t val_to_find = kc.rhs - curr_val + clique_table.tolerances.absolute_tolerance;
    auto it         = std::lower_bound(
      kc.entries.begin() + original_clique_start_idx, kc.entries.end(), val_to_find);
    if (it != kc.entries.end()) {
      i_t position_on_knapsack_constraint = std::distance(kc.entries.begin(), it);
      i_t start_pos_on_clique = position_on_knapsack_constraint - original_clique_start_idx;
      cuopt_assert(start_pos_on_clique >= 1, "Start position on clique is negative");
      cuopt_assert(it->val + curr_val > kc.rhs, "RHS mismatch");
#if DEBUG_KNAPSACK_CONSTRAINTS
      CUOPT_LOG_DEBUG("Found additional clique: %d, %d, %d",
                      curr_col,
                      clique_table.first.size() - 1,
                      start_pos_on_clique);
#endif
      clique_table.addtl_cliques.push_back(
        {curr_col, (i_t)clique_table.first.size() - 1, start_pos_on_clique});
    } else {
      break;
    }
    k--;
  }
}

// sort CSR by constraint coefficients
template <typename i_t, typename f_t>
void sort_csr_by_constraint_coefficients(
  std::vector<knapsack_constraint_t<i_t, f_t>>& knapsack_constraints)
{
  // sort the rows of the CSR matrix by the coefficients of the constraint
  for (auto& knapsack_constraint : knapsack_constraints) {
    std::sort(knapsack_constraint.entries.begin(), knapsack_constraint.entries.end());
  }
}

template <typename i_t, typename f_t>
void make_coeff_positive_knapsack_constraint(
  const dual_simplex::user_problem_t<i_t, f_t>& problem,
  std::vector<knapsack_constraint_t<i_t, f_t>>& knapsack_constraints,
  std::unordered_set<i_t>& set_packing_constraints,
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances)
{
  for (i_t i = 0; i < (i_t)knapsack_constraints.size(); i++) {
    auto& knapsack_constraint = knapsack_constraints[i];
    f_t rhs_offset            = 0;
    bool all_coeff_are_equal  = true;
    f_t first_coeff           = std::abs(knapsack_constraint.entries[0].val);
    for (auto& entry : knapsack_constraint.entries) {
      if (entry.val < 0) {
        entry.val = -entry.val;
        rhs_offset += entry.val;
        // negation of a variable is var + num_cols
        entry.col = entry.col + problem.num_cols;
      }
      if (!integer_equal<f_t>(entry.val, first_coeff, tolerances.absolute_tolerance)) {
        all_coeff_are_equal = false;
      }
    }
    knapsack_constraint.rhs += rhs_offset;
    if (!integer_equal<f_t>(knapsack_constraint.rhs, first_coeff, tolerances.absolute_tolerance)) {
      all_coeff_are_equal = false;
    }
    knapsack_constraint.is_set_packing = all_coeff_are_equal;
    if (!all_coeff_are_equal) { knapsack_constraint.is_set_partitioning = false; }
    if (knapsack_constraint.is_set_packing) { set_packing_constraints.insert(i); }
    cuopt_assert(knapsack_constraint.rhs >= 0, "RHS must be non-negative");
  }
}

// convert all the knapsack constraints
// if a binary variable has a negative coefficient, put its negation in the constraint
template <typename i_t, typename f_t>
void fill_knapsack_constraints(const dual_simplex::user_problem_t<i_t, f_t>& problem,
                               std::vector<knapsack_constraint_t<i_t, f_t>>& knapsack_constraints,
                               dual_simplex::csr_matrix_t<i_t, f_t>& A)
{
  // we might add additional constraints for the equality constraints
  i_t added_constraints = 0;
  // in user problems, ranged constraint ids monotonically increase.
  // when a row sense is "E", check if it is ranged constraint and treat accordingly
  i_t ranged_constraint_counter = 0;
  for (i_t i = 0; i < A.m; i++) {
    std::pair<i_t, i_t> constraint_range = A.get_constraint_range(i);
    if (constraint_range.second - constraint_range.first < 2) {
      CUOPT_LOG_DEBUG("Constraint %d has less than 2 variables, skipping", i);
      if (problem.row_sense[i] == 'E' && ranged_constraint_counter < problem.num_range_rows &&
          problem.range_rows[ranged_constraint_counter] == i) {
        ranged_constraint_counter++;
      }
      continue;
    }
    bool all_binary = true;
    // check if all variables are binary (any non-continuous with bounds [0,1])
    for (i_t j = constraint_range.first; j < constraint_range.second; j++) {
      if (problem.var_types[A.j[j]] == dual_simplex::variable_type_t::CONTINUOUS ||
          problem.lower[A.j[j]] != 0 || problem.upper[A.j[j]] != 1) {
        all_binary = false;
        break;
      }
    }
    // if all variables are binary, convert the constraint to a knapsack constraint
    if (!all_binary) {
      if (problem.row_sense[i] == 'E' && ranged_constraint_counter < problem.num_range_rows &&
          problem.range_rows[ranged_constraint_counter] == i) {
        ranged_constraint_counter++;
      }
      continue;
    }
    knapsack_constraint_t<i_t, f_t> knapsack_constraint;

    knapsack_constraint.cstr_idx = i;
    if (problem.row_sense[i] == 'L') {
      knapsack_constraint.rhs = problem.rhs[i];
      for (i_t j = constraint_range.first; j < constraint_range.second; j++) {
        knapsack_constraint.entries.push_back({A.j[j], A.x[j]});
      }
    } else if (problem.row_sense[i] == 'G') {
      knapsack_constraint.rhs = -problem.rhs[i];
      for (i_t j = constraint_range.first; j < constraint_range.second; j++) {
        knapsack_constraint.entries.push_back({A.j[j], -A.x[j]});
      }
    }
    // equality part
    else {
      // For equality rows, partitioning status should not depend on raw rhs scale here.
      // The exact set-packing/partitioning check is finalized later in
      // make_coeff_positive_knapsack_constraint after coefficient normalization.
      bool is_set_partitioning = true;
      bool ranged_constraint   = ranged_constraint_counter < problem.num_range_rows &&
                               problem.range_rows[ranged_constraint_counter] == i;
      // less than part
      knapsack_constraint.rhs = problem.rhs[i];
      if (ranged_constraint) {
        knapsack_constraint.rhs += problem.range_value[ranged_constraint_counter];
        is_set_partitioning = problem.range_value[ranged_constraint_counter] == 0.;
        ranged_constraint_counter++;
      }
      for (i_t j = constraint_range.first; j < constraint_range.second; j++) {
        knapsack_constraint.entries.push_back({A.j[j], A.x[j]});
      }
      // greater than part: convert it to less than
      knapsack_constraint_t<i_t, f_t> knapsack_constraint2;
      // Mark synthetic rows from equality splitting with negative ids so they never alias real row
      // indices (including rows appended later by clique extension).
      knapsack_constraint2.cstr_idx = -(added_constraints + 1);
      added_constraints++;
      knapsack_constraint2.rhs = -problem.rhs[i];
      for (i_t j = constraint_range.first; j < constraint_range.second; j++) {
        knapsack_constraint2.entries.push_back({A.j[j], -A.x[j]});
      }
      knapsack_constraint.is_set_partitioning  = is_set_partitioning;
      knapsack_constraint2.is_set_partitioning = is_set_partitioning;
      knapsack_constraints.push_back(knapsack_constraint2);
    }
    knapsack_constraints.push_back(knapsack_constraint);
  }
  CUOPT_LOG_DEBUG("Number of knapsack constraints: %d added %d constraints",
                  knapsack_constraints.size(),
                  added_constraints);
}

template <typename i_t, typename f_t>
void remove_small_cliques(clique_table_t<i_t, f_t>& clique_table, cuopt::timer_t& timer)
{
  i_t num_removed_first = 0;
  i_t num_removed_addtl = 0;
  std::vector<bool> to_delete(clique_table.first.size(), false);
  // if a clique is small, we remove it from the cliques and add it to adjlist
  for (size_t clique_idx = 0; clique_idx < clique_table.first.size(); clique_idx++) {
    if (timer.check_time_limit()) { return; }
    const auto& clique = clique_table.first[clique_idx];
    if (clique.size() <= (size_t)clique_table.min_clique_size) {
      for (size_t i = 0; i < clique.size(); i++) {
        for (size_t j = 0; j < clique.size(); j++) {
          if (i == j) { continue; }
          clique_table.adj_list_small_cliques[clique[i]].insert(clique[j]);
        }
      }
      num_removed_first++;
      to_delete[clique_idx] = true;
    }
  }
  for (size_t addtl_c = 0; addtl_c < clique_table.addtl_cliques.size(); addtl_c++) {
    const auto& addtl_clique   = clique_table.addtl_cliques[addtl_c];
    const auto base_clique_idx = static_cast<size_t>(addtl_clique.clique_idx);
    cuopt_assert(base_clique_idx < to_delete.size(),
                 "Additional clique points to invalid base clique index");
    // Remove additional cliques whose base clique is scheduled for deletion.
    if (to_delete[base_clique_idx]) {
      // Materialize conflicts represented by:
      //   addtl_clique.vertex_idx + first[base_clique_idx][start_pos_on_clique:]
      // before deleting both the additional and base clique entries.
      for (size_t i = addtl_clique.start_pos_on_clique;
           i < clique_table.first[base_clique_idx].size();
           i++) {
        clique_table.adj_list_small_cliques[clique_table.first[base_clique_idx][i]].insert(
          addtl_clique.vertex_idx);
        clique_table.adj_list_small_cliques[addtl_clique.vertex_idx].insert(
          clique_table.first[base_clique_idx][i]);
      }
      clique_table.addtl_cliques.erase(clique_table.addtl_cliques.begin() + addtl_c);
      addtl_c--;
      num_removed_addtl++;
      continue;
    }
    i_t size_of_clique =
      clique_table.first[base_clique_idx].size() - addtl_clique.start_pos_on_clique + 1;
    if (size_of_clique < clique_table.min_clique_size) {
      // the items from first clique are already added to the adjlist
      // only add the items that are coming from the new var in the additional clique
      for (size_t i = addtl_clique.start_pos_on_clique;
           i < clique_table.first[base_clique_idx].size();
           i++) {
        // insert conflicts both way
        clique_table.adj_list_small_cliques[clique_table.first[base_clique_idx][i]].insert(
          addtl_clique.vertex_idx);
        clique_table.adj_list_small_cliques[addtl_clique.vertex_idx].insert(
          clique_table.first[base_clique_idx][i]);
      }
      clique_table.addtl_cliques.erase(clique_table.addtl_cliques.begin() + addtl_c);
      addtl_c--;
      num_removed_addtl++;
    }
  }
  CUOPT_LOG_DEBUG("Number of removed cliques from first: %d, additional: %d",
                  num_removed_first,
                  num_removed_addtl);
  size_t i       = 0;
  size_t old_idx = 0;
  std::vector<i_t> index_mapping(clique_table.first.size(), -1);
  auto it = std::remove_if(clique_table.first.begin(), clique_table.first.end(), [&](auto& clique) {
    bool res = false;
    if (to_delete[old_idx]) {
      res = true;
    } else {
      index_mapping[old_idx] = i++;
    }
    old_idx++;
    return res;
  });
  clique_table.first.erase(it, clique_table.first.end());
  // renumber the reference indices in the additional cliques, since we removed some cliques
  for (size_t addtl_c = 0; addtl_c < clique_table.addtl_cliques.size(); addtl_c++) {
    i_t new_clique_idx = index_mapping[clique_table.addtl_cliques[addtl_c].clique_idx];
    cuopt_assert(new_clique_idx != -1, "New clique index is -1");
    clique_table.addtl_cliques[addtl_c].clique_idx = new_clique_idx;
    cuopt_assert(clique_table.first[new_clique_idx].size() -
                     clique_table.addtl_cliques[addtl_c].start_pos_on_clique + 1 >=
                   (size_t)clique_table.min_clique_size,
                 "A small clique remained after removing small cliques");
  }
  // Clique removals/edge materialization can change degrees; force recompute on next query.
  std::fill(clique_table.var_degrees.begin(), clique_table.var_degrees.end(), -1);
}

template <typename i_t, typename f_t>
std::unordered_set<i_t> clique_table_t<i_t, f_t>::get_adj_set_of_var(i_t var_idx)
{
  std::unordered_set<i_t> adj_set;
  for (const auto& clique_idx : var_clique_map_first[var_idx]) {
    adj_set.insert(first[clique_idx].begin(), first[clique_idx].end());
  }

  for (const auto& addtl_clique_idx : var_clique_map_addtl[var_idx]) {
    adj_set.insert(addtl_cliques[addtl_clique_idx].vertex_idx);
    adj_set.insert(first[addtl_cliques[addtl_clique_idx].clique_idx].begin() +
                     addtl_cliques[addtl_clique_idx].start_pos_on_clique,
                   first[addtl_cliques[addtl_clique_idx].clique_idx].end());
  }
  // Reverse lookup for additional cliques using position map:
  // if var_idx is in first[clique_idx][start_pos_on_clique:], it is adjacent to vertex_idx.
  for (const auto& addtl : addtl_cliques) {
    if (addtl.vertex_idx == var_idx) { continue; }
    if (static_cast<size_t>(addtl.clique_idx) < first_var_positions.size()) {
      const auto& pos_map = first_var_positions[addtl.clique_idx];
      auto it             = pos_map.find(var_idx);
      if (it != pos_map.end() && it->second >= addtl.start_pos_on_clique) {
        adj_set.insert(addtl.vertex_idx);
      }
    }
  }

  for (const auto& adj_vertex : adj_list_small_cliques[var_idx]) {
    adj_set.insert(adj_vertex);
  }
  // Add the complement of var_idx to the adjacency set
  i_t complement_idx = (var_idx >= n_variables) ? (var_idx - n_variables) : (var_idx + n_variables);
  adj_set.insert(complement_idx);
  adj_set.erase(var_idx);
  return adj_set;
}

template <typename i_t, typename f_t>
i_t clique_table_t<i_t, f_t>::get_degree_of_var(i_t var_idx)
{
  // if it is not already computed, compute it and return
  if (var_degrees[var_idx] == -1) { var_degrees[var_idx] = get_adj_set_of_var(var_idx).size(); }
  return var_degrees[var_idx];
}

template <typename i_t, typename f_t>
bool clique_table_t<i_t, f_t>::check_adjacency(i_t var_idx1, i_t var_idx2)
{
  if (var_idx1 == var_idx2) { return false; }
  if (var_idx1 % n_variables == var_idx2 % n_variables) { return true; }

  {
    auto it = adj_list_small_cliques.find(var_idx1);
    if (it != adj_list_small_cliques.end() && it->second.count(var_idx2) > 0) { return true; }
  }

  // Iterate whichever variable belongs to fewer first-cliques
  {
    i_t probe_var  = var_idx1;
    i_t target_var = var_idx2;
    if (var_clique_map_first[var_idx1].size() > var_clique_map_first[var_idx2].size()) {
      probe_var  = var_idx2;
      target_var = var_idx1;
    }
    for (const auto& clique_idx : var_clique_map_first[probe_var]) {
      if (first_var_positions[clique_idx].count(target_var) > 0) { return true; }
    }
  }

  for (const auto& addtl_idx : var_clique_map_addtl[var_idx1]) {
    const auto& addtl   = addtl_cliques[addtl_idx];
    const auto& pos_map = first_var_positions[addtl.clique_idx];
    auto it             = pos_map.find(var_idx2);
    if (it != pos_map.end() && it->second >= addtl.start_pos_on_clique) { return true; }
  }

  for (const auto& addtl_idx : var_clique_map_addtl[var_idx2]) {
    const auto& addtl   = addtl_cliques[addtl_idx];
    const auto& pos_map = first_var_positions[addtl.clique_idx];
    auto it             = pos_map.find(var_idx1);
    if (it != pos_map.end() && it->second >= addtl.start_pos_on_clique) { return true; }
  }

  return false;
}

// this function should only be called within extend clique
// if this is called outside extend clique, csr matrix should be converted into csc and copied into
// problem because the problem is partly modified
template <typename i_t, typename f_t>
void insert_clique_into_problem(const std::vector<i_t>& clique,
                                dual_simplex::user_problem_t<i_t, f_t>& problem,
                                dual_simplex::csr_matrix_t<i_t, f_t>& A,
                                f_t coeff_scale)
{
  // convert vertices into original vars
  f_t rhs_offset = 0.;
  std::vector<i_t> new_vars;
  std::vector<f_t> new_coeffs;
  for (size_t i = 0; i < clique.size(); i++) {
    f_t coeff   = coeff_scale;
    i_t var_idx = clique[i];
    if (var_idx >= problem.num_cols) {
      coeff   = -coeff_scale;
      var_idx = var_idx - problem.num_cols;
      rhs_offset += coeff_scale;
    }
    new_vars.push_back(var_idx);
    new_coeffs.push_back(coeff);
  }
  //   coeff_scale * (1 - x) = coeff_scale - coeff_scale * x
  // Move constants to the right, so rhs must decrease by rhs_offset.
  f_t rhs = coeff_scale - rhs_offset;
  // insert the new clique into the problem as a new constraint
  dual_simplex::sparse_vector_t<i_t, f_t> new_row(A.n, new_vars.size());
  new_row.i = std::move(new_vars);
  new_row.x = std::move(new_coeffs);
  A.append_row(new_row);
  problem.row_sense.push_back('L');
  problem.rhs.push_back(rhs);
  problem.row_names.push_back("Clique" + std::to_string(problem.row_names.size()));
}

template <typename i_t, typename f_t>
bool extend_clique(const std::vector<i_t>& clique,
                   clique_table_t<i_t, f_t>& clique_table,
                   dual_simplex::user_problem_t<i_t, f_t>& problem,
                   dual_simplex::csr_matrix_t<i_t, f_t>& A,
                   f_t coeff_scale,
                   bool modify_problem,
                   i_t min_extension_gain,
                   i_t remaining_rows_budget,
                   i_t remaining_nnz_budget,
                   i_t& inserted_row_nnz)
{
  inserted_row_nnz        = 0;
  i_t smallest_degree     = std::numeric_limits<i_t>::max();
  i_t smallest_degree_var = -1;
  // find smallest degree vertex in the current set packing constraint
  for (size_t idx = 0; idx < clique.size(); idx++) {
    i_t var_idx = clique[idx];
    i_t degree  = clique_table.get_degree_of_var(var_idx);
    if (degree < smallest_degree) {
      smallest_degree     = degree;
      smallest_degree_var = var_idx;
    }
  }
  std::vector<i_t> extension_candidates;
  auto smallest_degree_adj_set = clique_table.get_adj_set_of_var(smallest_degree_var);
  std::unordered_set<i_t> clique_members(clique.begin(), clique.end());
  for (const auto& candidate : smallest_degree_adj_set) {
    if (clique_members.find(candidate) == clique_members.end()) {
      extension_candidates.push_back(candidate);
    }
  }
  std::sort(extension_candidates.begin(), extension_candidates.end(), [&](i_t a, i_t b) {
    return clique_table.get_degree_of_var(a) > clique_table.get_degree_of_var(b);
  });
  auto new_clique               = clique;
  i_t n_of_complement_conflicts = 0;
  i_t complement_conflict_var   = -1;
  for (size_t idx = 0; idx < extension_candidates.size(); idx++) {
    i_t var_idx                 = extension_candidates[idx];
    bool add                    = true;
    bool complement_conflict    = false;
    i_t complement_conflict_idx = -1;
    for (size_t i = 0; i < new_clique.size(); i++) {
      if (var_idx % clique_table.n_variables == new_clique[i] % clique_table.n_variables) {
        complement_conflict     = true;
        complement_conflict_idx = var_idx % clique_table.n_variables;
      }
      // check if the tested variable conflicts with all vars in the new clique
      if (!clique_table.check_adjacency(var_idx, new_clique[i])) {
        add = false;
        break;
      }
    }
    if (add) {
      new_clique.push_back(var_idx);
      if (complement_conflict) {
        n_of_complement_conflicts++;
        complement_conflict_var = complement_conflict_idx;
      }
    }
  }
  // if we found a larger cliqe, insert it into the formulation
  if (new_clique.size() > clique.size()) {
    if (n_of_complement_conflicts > 0) {
      CUOPT_LOG_DEBUG("Found %d complement conflicts on var %d",
                      n_of_complement_conflicts,
                      complement_conflict_var);
      cuopt_assert(n_of_complement_conflicts == 1, "There can only be one complement conflict");
      // Keep the discovered extension in the clique table for downstream dominance checks.
      clique_table.first.push_back(new_clique);
      for (const auto& var_idx : new_clique) {
        clique_table.var_degrees[var_idx] = -1;
      }
      if (modify_problem) {
        // fix all other variables other than complementing var
        for (size_t i = 0; i < new_clique.size(); i++) {
          if (new_clique[i] % clique_table.n_variables != complement_conflict_var) {
            CUOPT_LOG_DEBUG("Fixing variable %d", new_clique[i]);
            if (new_clique[i] >= problem.num_cols) {
              cuopt_assert(problem.lower[new_clique[i] - problem.num_cols] != 0 ||
                             problem.upper[new_clique[i] - problem.num_cols] != 0,
                           "Variable is fixed to other side");
              problem.lower[new_clique[i] - problem.num_cols] = 1;
              problem.upper[new_clique[i] - problem.num_cols] = 1;
            } else {
              cuopt_assert(problem.lower[new_clique[i]] != 1 || problem.upper[new_clique[i]] != 1,
                           "Variable is fixed to other side");
              problem.lower[new_clique[i]] = 0;
              problem.upper[new_clique[i]] = 0;
            }
          }
        }
      }
      return true;
    } else {
      // Keep the discovered extension in the clique table even when row insertion is skipped by
      // row/nnz budgets.
      clique_table.first.push_back(new_clique);
      for (const auto& var_idx : new_clique) {
        clique_table.var_degrees[var_idx] = -1;
      }
#if DEBUG_KNAPSACK_CONSTRAINTS
      CUOPT_LOG_DEBUG("Extended clique: %lu from %lu", new_clique.size(), clique.size());
#endif
      i_t extension_gain = static_cast<i_t>(new_clique.size() - clique.size());
      if (extension_gain < min_extension_gain) { return true; }
      if (remaining_rows_budget <= 0 ||
          remaining_nnz_budget < static_cast<i_t>(new_clique.size())) {
        return true;
      }
      // Row insertion is now deferred until dominance is confirmed against model rows.
      // This keeps extension and replacement sequential: detect dominance first, then replace.
      inserted_row_nnz = 0;
    }
  }
  return new_clique.size() > clique.size();
}

template <typename i_t>
struct clique_sig_t {
  i_t knapsack_idx;
  i_t size;
  long long signature;
};

template <typename i_t>
struct extension_candidate_t {
  i_t knapsack_idx;
  i_t estimated_gain;
  i_t clique_size;
};

template <typename i_t>
bool compare_clique_sig(const clique_sig_t<i_t>& a, const clique_sig_t<i_t>& b)
{
  if (a.signature != b.signature) { return a.signature < b.signature; }
  return a.size < b.size;
}

template <typename i_t>
bool compare_signature_value(long long value, const clique_sig_t<i_t>& a)
{
  return value < a.signature;
}

template <typename i_t>
bool compare_extension_candidate(const extension_candidate_t<i_t>& a,
                                 const extension_candidate_t<i_t>& b)
{
  if (a.estimated_gain != b.estimated_gain) { return a.estimated_gain > b.estimated_gain; }
  if (a.clique_size != b.clique_size) { return a.clique_size < b.clique_size; }
  return a.knapsack_idx < b.knapsack_idx;
}

template <typename i_t>
bool is_sorted_subset(const std::vector<i_t>& a, const std::vector<i_t>& b)
{
  size_t i = 0;
  size_t j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] == b[j]) {
      i++;
      j++;
    } else if (a[i] > b[j]) {
      j++;
    } else {
      return false;
    }
  }
  return i == a.size();
}

template <typename i_t, typename f_t>
void fix_difference(const std::vector<i_t>& superset,
                    const std::vector<i_t>& subset,
                    dual_simplex::user_problem_t<i_t, f_t>& problem)
{
  cuopt_assert(std::is_sorted(subset.begin(), subset.end()),
               "subset vector passed to fix_difference is not sorted");
  for (auto var_idx : superset) {
    if (std::binary_search(subset.begin(), subset.end(), var_idx)) { continue; }
    if (var_idx >= problem.num_cols) {
      i_t orig_idx = var_idx - problem.num_cols;
      CUOPT_LOG_DEBUG("Fixing variable %d", orig_idx);
      cuopt_assert(problem.lower[orig_idx] != 0 || problem.upper[orig_idx] != 0,
                   "Variable is fixed to other side");
      problem.lower[orig_idx] = 1;
      problem.upper[orig_idx] = 1;
    } else {
      CUOPT_LOG_DEBUG("Fixing variable %d", var_idx);
      cuopt_assert(problem.lower[var_idx] != 1 || problem.upper[var_idx] != 1,
                   "Variable is fixed to other side");
      problem.lower[var_idx] = 0;
      problem.upper[var_idx] = 0;
    }
  }
}

template <typename i_t, typename T>
void remove_marked_elements(std::vector<T>& vec, const std::vector<i_t>& removal_marker)
{
  size_t write_idx = 0;
  for (size_t i = 0; i < vec.size(); i++) {
    if (!removal_marker[i]) {
      if (write_idx != i) { vec[write_idx] = std::move(vec[i]); }
      write_idx++;
    }
  }
  vec.resize(write_idx);
}

template <typename i_t, typename f_t>
void remove_dominated_cliques_in_problem_for_single_extended_clique(
  const std::vector<i_t>& curr_clique,
  f_t coeff_scale,
  i_t remaining_rows_budget,
  i_t remaining_nnz_budget,
  i_t& inserted_row_nnz,
  const std::vector<clique_sig_t<i_t>>& sp_sigs,
  const std::vector<std::vector<i_t>>& cstr_vars,
  const std::vector<knapsack_constraint_t<i_t, f_t>>& knapsack_constraints,
  std::vector<i_t>& original_to_current_row_idx,
  dual_simplex::user_problem_t<i_t, f_t>& problem,
  dual_simplex::csr_matrix_t<i_t, f_t>& A,
  cuopt::timer_t& timer)
{
  inserted_row_nnz = 0;
  if (curr_clique.empty() || sp_sigs.empty()) { return; }
  std::vector<i_t> curr_clique_vars(curr_clique.begin(), curr_clique.end());
  std::sort(curr_clique_vars.begin(), curr_clique_vars.end());
  curr_clique_vars.erase(std::unique(curr_clique_vars.begin(), curr_clique_vars.end()),
                         curr_clique_vars.end());
  long long signature = 0;
  for (auto v : curr_clique_vars) {
    signature += static_cast<long long>(v);
  }
  constexpr size_t dominance_window = 20000;
  auto end_it =
    std::upper_bound(sp_sigs.begin(), sp_sigs.end(), signature, compare_signature_value<i_t>);
  size_t end   = static_cast<size_t>(std::distance(sp_sigs.begin(), end_it));
  size_t start = (end > dominance_window) ? (end - dominance_window) : 0;
  std::vector<i_t> rows_to_remove;
  bool covering_clique_implied_by_partitioning = false;
  for (size_t idx = end; idx > start; idx--) {
    if (timer.check_time_limit()) { break; }
    const auto& sp      = sp_sigs[idx - 1];
    const auto& vars_sp = cstr_vars[sp.knapsack_idx];
    if (vars_sp.size() > curr_clique_vars.size()) { continue; }
    cuopt_assert(std::is_sorted(vars_sp.begin(), vars_sp.end()),
                 "vars_sp vector passed to is_sorted_subset is not sorted");
    if (!is_sorted_subset(vars_sp, curr_clique_vars)) { continue; }
    if (knapsack_constraints[sp.knapsack_idx].is_set_partitioning) {
      if (vars_sp.size() != curr_clique_vars.size()) {
        fix_difference(curr_clique_vars, vars_sp, problem);
        covering_clique_implied_by_partitioning = true;
      }
      continue;
    }
    i_t original_row_idx = knapsack_constraints[sp.knapsack_idx].cstr_idx;
    if (original_row_idx < 0) { continue; }
    cuopt_assert(original_row_idx < static_cast<i_t>(original_to_current_row_idx.size()),
                 "Invalid original row index in knapsack constraint");
    i_t current_row_idx = original_to_current_row_idx[original_row_idx];
    if (current_row_idx < 0) { continue; }
    cuopt_assert(current_row_idx < static_cast<i_t>(problem.row_sense.size()),
                 "Invalid current row index in row mapping");
    rows_to_remove.push_back(current_row_idx);
  }
  if (rows_to_remove.empty()) { return; }
  std::sort(rows_to_remove.begin(), rows_to_remove.end());
  rows_to_remove.erase(std::unique(rows_to_remove.begin(), rows_to_remove.end()),
                       rows_to_remove.end());
  if (!covering_clique_implied_by_partitioning) {
    if (remaining_rows_budget <= 0 ||
        remaining_nnz_budget < static_cast<i_t>(curr_clique_vars.size())) {
      return;
    }
    insert_clique_into_problem(curr_clique_vars, problem, A, coeff_scale);
    inserted_row_nnz = static_cast<i_t>(curr_clique_vars.size());
  }
  std::vector<i_t> removal_marker(problem.row_sense.size(), 0);
  for (auto row_idx : rows_to_remove) {
    cuopt_assert(row_idx >= 0 && row_idx < static_cast<i_t>(removal_marker.size()),
                 "Invalid dominated row index");
    CUOPT_LOG_DEBUG("Removing dominated row %d", row_idx);
    removal_marker[row_idx] = true;
  }
  dual_simplex::csr_matrix_t<i_t, f_t> A_removed(0, 0, 0);
  A.remove_rows(removal_marker, A_removed);
  A                = std::move(A_removed);
  problem.num_rows = A.m;
  remove_marked_elements(problem.row_sense, removal_marker);
  remove_marked_elements(problem.rhs, removal_marker);
  remove_marked_elements(problem.row_names, removal_marker);
  cuopt_assert(problem.rhs.size() == problem.row_sense.size(), "rhs and row sense size mismatch");
  cuopt_assert(problem.row_names.size() == problem.rhs.size(), "row names and rhs size mismatch");
  cuopt_assert(problem.num_rows == static_cast<i_t>(problem.rhs.size()),
               "matrix and num rows mismatch after removal");
  if (!problem.range_rows.empty()) {
    std::vector<i_t> old_to_new_indices;
    old_to_new_indices.reserve(removal_marker.size());
    i_t new_idx = 0;
    for (size_t i = 0; i < removal_marker.size(); ++i) {
      if (!removal_marker[i]) {
        old_to_new_indices.push_back(new_idx++);
      } else {
        old_to_new_indices.push_back(-1);
      }
    }
    std::vector<i_t> new_range_rows;
    std::vector<f_t> new_range_values;
    for (size_t i = 0; i < problem.range_rows.size(); ++i) {
      i_t old_row = problem.range_rows[i];
      cuopt_assert(old_row >= 0 && old_row < static_cast<i_t>(removal_marker.size()),
                   "Invalid row index in range_rows");
      if (!removal_marker[old_row]) {
        i_t new_row = old_to_new_indices[old_row];
        cuopt_assert(new_row != -1, "Invalid new row index for ranged row renumbering");
        new_range_rows.push_back(new_row);
        new_range_values.push_back(problem.range_value[i]);
      }
    }
    problem.range_rows  = std::move(new_range_rows);
    problem.range_value = std::move(new_range_values);
  }
  problem.num_range_rows = static_cast<i_t>(problem.range_rows.size());
  std::vector<i_t> removed_prefix(removal_marker.size() + 1, 0);
  for (size_t row_idx = 0; row_idx < removal_marker.size(); row_idx++) {
    removed_prefix[row_idx + 1] =
      removed_prefix[row_idx] + static_cast<i_t>(removal_marker[row_idx]);
  }
  for (i_t row_idx = 0; row_idx < static_cast<i_t>(original_to_current_row_idx.size()); row_idx++) {
    i_t current_row_idx = original_to_current_row_idx[row_idx];
    if (current_row_idx < 0) { continue; }
    cuopt_assert(current_row_idx < static_cast<i_t>(removal_marker.size()),
                 "Row index map is out of bounds");
    if (removal_marker[current_row_idx]) {
      original_to_current_row_idx[row_idx] = -1;
    } else {
      original_to_current_row_idx[row_idx] = current_row_idx - removed_prefix[current_row_idx];
    }
  }
}

// Also known as clique merging. Infer larger clique constraints which allows inclusion of vars from
// other constraints. This only extends the original cliques in the formulation for now.
// TODO: consider a heuristic on how much of the cliques derived from knapsacks to include here
template <typename i_t, typename f_t>
i_t extend_cliques(const std::vector<knapsack_constraint_t<i_t, f_t>>& knapsack_constraints,
                   const std::unordered_set<i_t>& set_packing_constraints,
                   clique_table_t<i_t, f_t>& clique_table,
                   dual_simplex::user_problem_t<i_t, f_t>& problem,
                   dual_simplex::csr_matrix_t<i_t, f_t>& A,
                   bool modify_problem,
                   cuopt::timer_t& timer,
                   double* work_estimate_out,
                   double max_work_estimate)
{
  constexpr i_t min_extension_gain       = 2;
  constexpr i_t extension_yield_window   = 64;
  constexpr i_t min_successes_per_window = 1;

  double local_work = 0.0;
  double& work      = work_estimate_out ? *work_estimate_out : local_work;

  i_t base_rows      = A.m;
  i_t base_nnz       = A.row_start[A.m];
  i_t max_added_rows = std::max<i_t>(8, base_rows / 50);
  i_t max_added_nnz  = std::max<i_t>(8 * clique_table.max_clique_size_for_extension, base_nnz / 50);

  i_t added_rows       = 0;
  i_t added_nnz        = 0;
  i_t window_attempts  = 0;
  i_t window_successes = 0;

  CUOPT_LOG_DEBUG("Clique extension heuristics: min_gain=%d row_budget=%d nnz_budget=%d",
                  min_extension_gain,
                  max_added_rows,
                  max_added_nnz);
  std::vector<std::vector<i_t>> cstr_vars(knapsack_constraints.size());
  std::vector<clique_sig_t<i_t>> sp_sigs;
  sp_sigs.reserve(set_packing_constraints.size());
  for (const auto knapsack_idx : set_packing_constraints) {
    cuopt_assert(knapsack_idx >= 0 && knapsack_idx < static_cast<i_t>(knapsack_constraints.size()),
                 "Invalid set packing constraint index");
    const auto& vars = knapsack_constraints[knapsack_idx].entries;
    cstr_vars[knapsack_idx].reserve(vars.size());
    for (const auto& entry : vars) {
      cstr_vars[knapsack_idx].push_back(entry.col);
    }
    std::sort(cstr_vars[knapsack_idx].begin(), cstr_vars[knapsack_idx].end());
    cstr_vars[knapsack_idx].erase(
      std::unique(cstr_vars[knapsack_idx].begin(), cstr_vars[knapsack_idx].end()),
      cstr_vars[knapsack_idx].end());
    long long signature = 0;
    for (auto v : cstr_vars[knapsack_idx]) {
      signature += static_cast<long long>(v);
    }
    sp_sigs.push_back({knapsack_idx, static_cast<i_t>(cstr_vars[knapsack_idx].size()), signature});
    work += cstr_vars[knapsack_idx].size();
  }
  if (work > max_work_estimate) { return 0; }
  std::sort(sp_sigs.begin(), sp_sigs.end(), compare_clique_sig<i_t>);
  std::vector<i_t> original_to_current_row_idx(problem.row_sense.size(), -1);
  for (i_t row_idx = 0; row_idx < static_cast<i_t>(original_to_current_row_idx.size()); row_idx++) {
    original_to_current_row_idx[row_idx] = row_idx;
  }
  std::vector<extension_candidate_t<i_t>> extension_worklist;
  extension_worklist.reserve(knapsack_constraints.size());
  for (i_t knapsack_idx = 0; knapsack_idx < static_cast<i_t>(knapsack_constraints.size());
       knapsack_idx++) {
    if (timer.check_time_limit()) { break; }
    if (work > max_work_estimate) { break; }
    const auto& knapsack_constraint = knapsack_constraints[knapsack_idx];
    if (!knapsack_constraint.is_set_packing) { continue; }
    i_t clique_size = static_cast<i_t>(knapsack_constraint.entries.size());
    if (clique_size >= clique_table.max_clique_size_for_extension) { continue; }
    i_t smallest_degree = std::numeric_limits<i_t>::max();
    for (const auto& entry : knapsack_constraint.entries) {
      smallest_degree = std::min(smallest_degree, clique_table.get_degree_of_var(entry.col));
    }
    i_t estimated_gain = std::max<i_t>(0, smallest_degree - (clique_size - 1));
    if (estimated_gain < min_extension_gain) { continue; }
    extension_worklist.push_back({knapsack_idx, estimated_gain, clique_size});
    work += knapsack_constraint.entries.size();
  }
  std::stable_sort(
    extension_worklist.begin(), extension_worklist.end(), compare_extension_candidate<i_t>);
  CUOPT_LOG_DEBUG("Clique extension candidates after scoring: %zu", extension_worklist.size());

  i_t n_extended_cliques = 0;
  for (const auto& candidate : extension_worklist) {
    if (timer.check_time_limit()) { break; }
    if (work > max_work_estimate) { break; }
    if (added_rows >= max_added_rows || added_nnz >= max_added_nnz) {
      CUOPT_LOG_DEBUG(
        "Stopping clique extension: budget reached (rows=%d nnz=%d)", added_rows, added_nnz);
      break;
    }
    window_attempts++;
    const auto& knapsack_constraint = knapsack_constraints[candidate.knapsack_idx];
    std::vector<i_t> clique;
    for (const auto& entry : knapsack_constraint.entries) {
      clique.push_back(entry.col);
    }
    i_t inserted_row_nnz = 0;
    f_t coeff_scale      = knapsack_constraint.entries[0].val;
    bool extended_clique = extend_clique(clique,
                                         clique_table,
                                         problem,
                                         A,
                                         coeff_scale,
                                         modify_problem,
                                         min_extension_gain,
                                         max_added_rows - added_rows,
                                         max_added_nnz - added_nnz,
                                         inserted_row_nnz);
    work += clique.size() * clique.size();
    if (extended_clique) {
      n_extended_cliques++;
      i_t replacement_row_nnz = 0;
      if (modify_problem) {
        remove_dominated_cliques_in_problem_for_single_extended_clique(clique_table.first.back(),
                                                                       coeff_scale,
                                                                       max_added_rows - added_rows,
                                                                       max_added_nnz - added_nnz,
                                                                       replacement_row_nnz,
                                                                       sp_sigs,
                                                                       cstr_vars,
                                                                       knapsack_constraints,
                                                                       original_to_current_row_idx,
                                                                       problem,
                                                                       A,
                                                                       timer);
      }
      if (replacement_row_nnz > 0) {
        window_successes++;
        added_rows++;
        added_nnz += replacement_row_nnz;
      }
    }
    if (window_attempts >= extension_yield_window) {
      if (window_successes < min_successes_per_window) {
        CUOPT_LOG_DEBUG(
          "Stopping clique extension: low yield (%d/%d)", window_successes, window_attempts);
        break;
      }
      window_attempts  = 0;
      window_successes = 0;
    }
  }
  if (modify_problem) {
    // copy modified matrix back to problem
    A.to_compressed_col(problem.A);
  }
  CUOPT_LOG_DEBUG("Number of extended cliques: %d", n_extended_cliques);
  return n_extended_cliques;
}

template <typename i_t, typename f_t>
void fill_var_clique_maps(clique_table_t<i_t, f_t>& clique_table)
{
  clique_table.first_var_positions.resize(clique_table.first.size());
  for (size_t clique_idx = 0; clique_idx < clique_table.first.size(); clique_idx++) {
    const auto& clique = clique_table.first[clique_idx];
    auto& pos_map      = clique_table.first_var_positions[clique_idx];
    pos_map.reserve(clique.size());
    for (size_t idx = 0; idx < clique.size(); idx++) {
      i_t var_idx = clique[idx];
      clique_table.var_clique_map_first[var_idx].insert(clique_idx);
      pos_map[var_idx] = static_cast<i_t>(idx);
    }
  }
  for (size_t addtl_c = 0; addtl_c < clique_table.addtl_cliques.size(); addtl_c++) {
    const auto& addtl_clique = clique_table.addtl_cliques[addtl_c];
    clique_table.var_clique_map_addtl[addtl_clique.vertex_idx].insert(addtl_c);
  }
}

template <typename i_t, typename f_t>
void build_clique_table(const dual_simplex::user_problem_t<i_t, f_t>& problem,
                        clique_table_t<i_t, f_t>& clique_table,
                        typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances,
                        bool remove_small_cliques_flag,
                        bool fill_var_clique_maps_flag,
                        cuopt::timer_t& timer)
{
  if (timer.check_time_limit()) { return; }
  cuopt_assert(clique_table.n_variables == problem.num_cols, "Clique table size mismatch");
  cuopt_assert(problem.var_types.size() == static_cast<size_t>(problem.num_cols),
               "Problem variable types size mismatch");
  std::vector<knapsack_constraint_t<i_t, f_t>> knapsack_constraints;
  std::unordered_set<i_t> set_packing_constraints;
  dual_simplex::csr_matrix_t<i_t, f_t> A(problem.num_rows, problem.num_cols, 0);
  problem.A.to_compressed_row(A);
  fill_knapsack_constraints(problem, knapsack_constraints, A);
  make_coeff_positive_knapsack_constraint(
    problem, knapsack_constraints, set_packing_constraints, tolerances);
  sort_csr_by_constraint_coefficients(knapsack_constraints);
  clique_table.tolerances = tolerances;
  for (const auto& knapsack_constraint : knapsack_constraints) {
    if (timer.check_time_limit()) { return; }
    find_cliques_from_constraint(knapsack_constraint, clique_table, timer);
  }
  if (timer.check_time_limit()) { return; }
  if (remove_small_cliques_flag) { remove_small_cliques(clique_table, timer); }
  if (timer.check_time_limit()) { return; }
  if (fill_var_clique_maps_flag) { fill_var_clique_maps(clique_table); }
}

template <typename i_t, typename f_t>
void print_knapsack_constraints(
  const std::vector<knapsack_constraint_t<i_t, f_t>>& knapsack_constraints,
  bool print_only_set_packing = false)
{
#if DEBUG_KNAPSACK_CONSTRAINTS
  std::cout << "Number of knapsack constraints: " << knapsack_constraints.size() << "\n";
  for (const auto& knapsack : knapsack_constraints) {
    if (print_only_set_packing && !knapsack.is_set_packing) { continue; }
    std::cout << "Knapsack constraint idx: " << knapsack.cstr_idx << "\n";
    std::cout << "  RHS: " << knapsack.rhs << "\n";
    std::cout << "  Is set packing: " << knapsack.is_set_packing << "\n";
    std::cout << "  Entries:\n";
    for (const auto& entry : knapsack.entries) {
      std::cout << "    col: " << entry.col << ", val: " << entry.val << "\n";
    }
    std::cout << "----------\n";
  }
#endif
}

template <typename i_t, typename f_t>
void print_clique_table(const clique_table_t<i_t, f_t>& clique_table)
{
#if DEBUG_KNAPSACK_CONSTRAINTS
  std::cout << "Number of cliques: " << clique_table.first.size() << "\n";
  for (const auto& clique : clique_table.first) {
    std::cout << "Clique: ";
    for (const auto& var : clique) {
      std::cout << var << " ";
    }
  }
  std::cout << "Number of additional cliques: " << clique_table.addtl_cliques.size() << "\n";
  for (const auto& addtl_clique : clique_table.addtl_cliques) {
    std::cout << "Additional clique: " << addtl_clique.vertex_idx << ", " << addtl_clique.clique_idx
              << ", " << addtl_clique.start_pos_on_clique << "\n";
  }
#endif
}

template <typename i_t, typename f_t>
void find_initial_cliques(dual_simplex::user_problem_t<i_t, f_t>& problem,
                          typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances,
                          std::shared_ptr<clique_table_t<i_t, f_t>>* clique_table_out,
                          cuopt::timer_t& timer,
                          bool modify_problem,
                          std::atomic<bool>* signal_extend)
{
  cuopt::timer_t stage_timer(std::numeric_limits<double>::infinity());
#ifdef DEBUG_CLIQUE_TABLE
  double t_fill   = 0.;
  double t_coeff  = 0.;
  double t_sort   = 0.;
  double t_find   = 0.;
  double t_small  = 0.;
  double t_maps   = 0.;
  double t_extend = 0.;
  double t_remove = 0.;
#endif
  std::vector<knapsack_constraint_t<i_t, f_t>> knapsack_constraints;
  std::unordered_set<i_t> set_packing_constraints;
  dual_simplex::csr_matrix_t<i_t, f_t> A(problem.num_rows, problem.num_cols, 0);
  problem.A.to_compressed_row(A);
  fill_knapsack_constraints(problem, knapsack_constraints, A);
#ifdef DEBUG_CLIQUE_TABLE
  t_fill = stage_timer.elapsed_time();
#endif
  make_coeff_positive_knapsack_constraint(
    problem, knapsack_constraints, set_packing_constraints, tolerances);
#ifdef DEBUG_CLIQUE_TABLE
  t_coeff = stage_timer.elapsed_time();
#endif
  sort_csr_by_constraint_coefficients(knapsack_constraints);
#ifdef DEBUG_CLIQUE_TABLE
  t_sort = stage_timer.elapsed_time();
#endif
  clique_config_t clique_config;
  std::shared_ptr<clique_table_t<i_t, f_t>> clique_table_shared;
  clique_table_t<i_t, f_t> clique_table_local(2 * problem.num_cols,
                                              clique_config.min_clique_size,
                                              clique_config.max_clique_size_for_extension);
  clique_table_t<i_t, f_t>* clique_table_ptr = &clique_table_local;
  if (clique_table_out != nullptr) {
    clique_table_shared =
      std::make_shared<clique_table_t<i_t, f_t>>(2 * problem.num_cols,
                                                 clique_config.min_clique_size,
                                                 clique_config.max_clique_size_for_extension);
    clique_table_ptr = clique_table_shared.get();
  }
  clique_table_ptr->tolerances             = tolerances;
  double time_limit_for_additional_cliques = timer.remaining_time() / 2;
  cuopt::timer_t additional_cliques_timer(time_limit_for_additional_cliques);
  double find_work_estimate = 0.0;
  for (const auto& knapsack_constraint : knapsack_constraints) {
    if (timer.check_time_limit()) { break; }
    if (signal_extend && signal_extend->load(std::memory_order_acquire)) { break; }
    find_cliques_from_constraint(knapsack_constraint, *clique_table_ptr, additional_cliques_timer);
    find_work_estimate += knapsack_constraint.entries.size();
  }
#ifdef DEBUG_CLIQUE_TABLE
  t_find = stage_timer.elapsed_time();
#endif
  CUOPT_LOG_DEBUG("Number of cliques: %d, additional cliques: %d, find_work=%.0f",
                  clique_table_ptr->first.size(),
                  clique_table_ptr->addtl_cliques.size(),
                  find_work_estimate);
  remove_small_cliques(*clique_table_ptr, timer);
#ifdef DEBUG_CLIQUE_TABLE
  t_small = stage_timer.elapsed_time();
#endif
  fill_var_clique_maps(*clique_table_ptr);
#ifdef DEBUG_CLIQUE_TABLE
  t_maps = stage_timer.elapsed_time();
#endif
  if (clique_table_out != nullptr) { *clique_table_out = std::move(clique_table_shared); }
  double extend_work               = 0.0;
  constexpr double max_extend_work = 2e9;
  i_t n_extended_cliques           = extend_cliques(knapsack_constraints,
                                          set_packing_constraints,
                                          *clique_table_ptr,
                                          problem,
                                          A,
                                          modify_problem,
                                          timer,
                                          &extend_work,
                                          max_extend_work);
#ifdef DEBUG_CLIQUE_TABLE
  t_extend = stage_timer.elapsed_time();
  CUOPT_LOG_DEBUG(
    "Clique table timing (s): fill=%.6f coeff=%.6f sort=%.6f find=%.6f small=%.6f maps=%.6f "
    "extend=%.6f total=%.6f find_work=%.0f extend_work=%.0f",
    t_fill,
    t_coeff - t_fill,
    t_sort - t_coeff,
    t_find - t_sort,
    t_small - t_find,
    t_maps - t_small,
    t_extend - t_maps,
    t_extend,
    find_work_estimate,
    extend_work);
#endif
}

#define INSTANTIATE(F_TYPE)                                               \
  template void find_initial_cliques<int, F_TYPE>(                        \
    dual_simplex::user_problem_t<int, F_TYPE> & problem,                  \
    typename mip_solver_settings_t<int, F_TYPE>::tolerances_t tolerances, \
    std::shared_ptr<clique_table_t<int, F_TYPE>> * clique_table_out,      \
    cuopt::timer_t & timer,                                               \
    bool modify_problem,                                                  \
    std::atomic<bool>* signal_extend);                                    \
  template void build_clique_table<int, F_TYPE>(                          \
    const dual_simplex::user_problem_t<int, F_TYPE>& problem,             \
    clique_table_t<int, F_TYPE>& clique_table,                            \
    typename mip_solver_settings_t<int, F_TYPE>::tolerances_t tolerances, \
    bool remove_small_cliques_flag,                                       \
    bool fill_var_clique_maps_flag,                                       \
    cuopt::timer_t& timer);                                               \
  template class clique_table_t<int, F_TYPE>;

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif
#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif
#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
