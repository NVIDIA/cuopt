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

#define DEBUG_KNAPSACK_CONSTRAINTS 1

#include "clique_table.cuh"

#include <algorithm>
#include <dual_simplex/sparse_matrix.hpp>
#include <limits>
#include <mip/mip_constants.hpp>
#include <mip/utils.cuh>
#include <utilities/logger.hpp>
#include <utilities/macros.cuh>
#include <utilities/timer.hpp>

namespace cuopt::linear_programming::detail {

// do constraints with only binary variables.
template <typename i_t, typename f_t>
void find_cliques_from_constraint(const knapsack_constraint_t<i_t, f_t>& kc,
                                  clique_table_t<i_t, f_t>& clique_table)
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
    if (!all_binary) { continue; }
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
      bool is_set_partitioning = problem.rhs[i] == 1.;
      bool ranged_constraint   = ranged_constraint_counter < problem.num_range_rows &&
                               problem.range_rows[ranged_constraint_counter] == i;
      // less than part
      knapsack_constraint.rhs = problem.rhs[i];
      if (ranged_constraint) {
        knapsack_constraint.rhs += problem.range_value[ranged_constraint_counter];
        is_set_partitioning =
          problem.range_value[ranged_constraint_counter] == 0. && problem.rhs[i] == 1.;
        ranged_constraint_counter++;
      }
      for (i_t j = constraint_range.first; j < constraint_range.second; j++) {
        knapsack_constraint.entries.push_back({A.j[j], A.x[j]});
      }
      // greater than part: convert it to less than
      knapsack_constraint_t<i_t, f_t> knapsack_constraint2;
      knapsack_constraint2.cstr_idx = A.m + added_constraints++;
      knapsack_constraint2.rhs      = -problem.rhs[i];
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
void remove_small_cliques(clique_table_t<i_t, f_t>& clique_table)
{
  i_t num_removed_first = 0;
  i_t num_removed_addtl = 0;
  std::vector<bool> to_delete(clique_table.first.size(), false);
  // if a clique is small, we remove it from the cliques and add it to adjlist
  for (size_t clique_idx = 0; clique_idx < clique_table.first.size(); clique_idx++) {
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
    const auto& addtl_clique = clique_table.addtl_cliques[addtl_c];
    i_t size_of_clique =
      clique_table.first[addtl_clique.clique_idx].size() - addtl_clique.start_pos_on_clique + 1;
    if (size_of_clique < clique_table.min_clique_size) {
      // the items from first clique are already added to the adjlist
      // only add the items that are coming from the new var in the additional clique
      for (size_t i = addtl_clique.start_pos_on_clique;
           i < clique_table.first[addtl_clique.clique_idx].size();
           i++) {
        // insert conflicts both way
        clique_table.adj_list_small_cliques[clique_table.first[addtl_clique.clique_idx][i]].insert(
          addtl_clique.vertex_idx);
        clique_table.adj_list_small_cliques[addtl_clique.vertex_idx].insert(
          clique_table.first[addtl_clique.clique_idx][i]);
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
}

template <typename i_t, typename f_t>
std::unordered_set<i_t> clique_table_t<i_t, f_t>::get_adj_set_of_var(i_t var_idx)
{
  std::unordered_set<i_t> adj_set;
  for (const auto& clique_idx : var_clique_map_first[var_idx]) {
    adj_set.insert(first[clique_idx].begin(), first[clique_idx].end());
  }

  for (const auto& addtl_clique_idx : var_clique_map_addtl[var_idx]) {
    adj_set.insert(first[addtl_cliques[addtl_clique_idx].clique_idx].begin(),
                   first[addtl_cliques[addtl_clique_idx].clique_idx].end());
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
  // if passed same variable
  if (var_idx1 == var_idx2) { return false; }
  // in case they are complements of each other
  if (var_idx1 % n_variables == var_idx2 % n_variables) { return true; }
  if (adj_list_small_cliques[var_idx1].count(var_idx2) > 0) { return true; }
  // Check first cliques: var_clique_map_first stores clique indices
  for (const auto& clique_idx : var_clique_map_first[var_idx1]) {
    const auto& clique = first[clique_idx];
    // TODO: we can also keep a set of the clique if the memory allows, instead of doing linear
    // search
    if (std::find(clique.begin(), clique.end(), var_idx2) != clique.end()) { return true; }
  }

  // Check additional cliques: var_clique_map_addtl stores indices into addtl_cliques
  for (const auto& addtl_idx : var_clique_map_addtl[var_idx1]) {
    const auto& addtl  = addtl_cliques[addtl_idx];
    const auto& clique = first[addtl.clique_idx];
    // addtl clique is: vertex_idx + first[clique_idx][start_pos_on_clique:]
    if (addtl.vertex_idx == var_idx2) { return true; }
    if (addtl.start_pos_on_clique < static_cast<i_t>(clique.size())) {
      if (std::find(clique.begin() + addtl.start_pos_on_clique, clique.end(), var_idx2) !=
          clique.end()) {
        return true;
      }
    }
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
  f_t rhs = coeff_scale + rhs_offset;
  // insert the new clique into the problem as a new constraint
  A.insert_row(new_vars, new_coeffs);
  problem.row_sense.push_back('L');
  problem.rhs.push_back(rhs);
}

template <typename i_t, typename f_t>
bool extend_clique(const std::vector<i_t>& clique,
                   clique_table_t<i_t, f_t>& clique_table,
                   dual_simplex::user_problem_t<i_t, f_t>& problem,
                   dual_simplex::csr_matrix_t<i_t, f_t>& A,
                   f_t coeff_scale)
{
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
      return false;
    } else {
      clique_table.first.push_back(new_clique);
#if DEBUG_KNAPSACK_CONSTRAINTS
      CUOPT_LOG_DEBUG("Extended clique: %lu from %lu", new_clique.size(), clique.size());
#endif
      // insert the new clique into the problem as a new constraint
      insert_clique_into_problem(new_clique, problem, A, coeff_scale);
    }
  }
  return new_clique.size() > clique.size();
}

// Also known as clique merging. Infer larger clique constraints which allows inclusion of vars from
// other constraints. This only extends the original cliques in the formulation for now.
// TODO: consider a heuristic on how much of the cliques derived from knapsacks to include here
template <typename i_t, typename f_t>
i_t extend_cliques(const std::vector<knapsack_constraint_t<i_t, f_t>>& knapsack_constraints,
                   clique_table_t<i_t, f_t>& clique_table,
                   dual_simplex::user_problem_t<i_t, f_t>& problem,
                   dual_simplex::csr_matrix_t<i_t, f_t>& A,
                   cuopt::timer_t& timer)
{
  i_t n_extended_cliques = 0;
  // we try extending cliques on set packing constraints
  for (const auto& knapsack_constraint : knapsack_constraints) {
    if (timer.check_time_limit()) { break; }
    if (!knapsack_constraint.is_set_packing) { continue; }
    if (knapsack_constraint.entries.size() < (size_t)clique_table.max_clique_size_for_extension) {
      std::vector<i_t> clique;
      for (const auto& entry : knapsack_constraint.entries) {
        clique.push_back(entry.col);
      }
      f_t coeff_scale      = knapsack_constraint.entries[0].val;
      bool extended_clique = extend_clique(clique, clique_table, problem, A, coeff_scale);
      if (extended_clique) { n_extended_cliques++; }
    }
  }
  // problem.A.check_matrix();
  // copy modified matrix back to problem
  A.to_compressed_col(problem.A);
  return n_extended_cliques;
}

template <typename i_t, typename f_t>
void fill_var_clique_maps(clique_table_t<i_t, f_t>& clique_table)
{
  for (size_t clique_idx = 0; clique_idx < clique_table.first.size(); clique_idx++) {
    const auto& clique = clique_table.first[clique_idx];
    for (size_t idx = 0; idx < clique.size(); idx++) {
      i_t var_idx = clique[idx];
      clique_table.var_clique_map_first[var_idx].insert(clique_idx);
    }
  }
  for (size_t addtl_c = 0; addtl_c < clique_table.addtl_cliques.size(); addtl_c++) {
    const auto& addtl_clique = clique_table.addtl_cliques[addtl_c];
    clique_table.var_clique_map_addtl[addtl_clique.vertex_idx].insert(addtl_c);
  }
}

// we want to remove constraints that are covered by extended cliques
// for set partitioning constraints, we will keep the constraint on original problem but fix
// extended vars to zero For a set partitioning constraint: v1+v2+...+vk = 1 and discovered:
// v1+v2+...+vk  + vl1+vl2 +...+vli <= 1
// Then substituting the first to the second you have:
// 1  + vl1+vl2 +...+vli <= 1
// Which is simply:
// vl1+vl2 +...+vli <= 0
// so we can fix them
template <typename i_t, typename f_t>
void remove_dominated_cliques(
  dual_simplex::user_problem_t<i_t, f_t>& problem,
  dual_simplex::csr_matrix_t<i_t, f_t>& A,
  clique_table_t<i_t, f_t>& clique_table,
  std::unordered_set<i_t>& set_packing_constraints,
  const std::vector<knapsack_constraint_t<i_t, f_t>>& knapsack_constraints,
  i_t n_extended_cliques,
  cuopt::timer_t& timer)
{
  // TODO check if we need to add the dominance for the table itself
  i_t extended_clique_start_idx = clique_table.first.size() - n_extended_cliques;
  CUOPT_LOG_DEBUG("Number of extended cliques: %d", n_extended_cliques);
  std::vector<i_t> removal_marker(problem.row_sense.size(), 0);
  std::vector<std::vector<i_t>> cstr_vars(knapsack_constraints.size());
  bool time_limit_reached = timer.check_time_limit();
  for (const auto knapsack_idx : set_packing_constraints) {
    if (timer.check_time_limit()) {
      time_limit_reached = true;
      break;
    }
    cuopt_assert(knapsack_constraints[knapsack_idx].is_set_packing,
                 "Set packing constraint is not a set packing constraint");
    const auto& vars = knapsack_constraints[knapsack_idx].entries;
    cstr_vars[knapsack_idx].reserve(vars.size());
    for (const auto& entry : vars) {
      cstr_vars[knapsack_idx].push_back(entry.col);
    }
    std::sort(cstr_vars[knapsack_idx].begin(), cstr_vars[knapsack_idx].end());
  }
  if (!time_limit_reached) {
    CUOPT_LOG_DEBUG("Constraint variable lists built: %zu", set_packing_constraints.size());
    constexpr size_t dominance_window = 100;
    struct clique_sig_t {
      i_t knapsack_idx;
      i_t row_idx;
      i_t size;
      long long signature;
    };
    std::vector<clique_sig_t> sp_sigs;
    sp_sigs.reserve(set_packing_constraints.size());
    CUOPT_LOG_DEBUG("Building set packing signatures");
    for (const auto knapsack_idx : set_packing_constraints) {
      if (timer.check_time_limit()) {
        time_limit_reached = true;
        break;
      }
      const auto& vars = cstr_vars[knapsack_idx];
      if (vars.empty()) { continue; }
      long long signature = 0;
      for (auto v : vars) {
        signature += static_cast<long long>(v);
      }
      sp_sigs.push_back({knapsack_idx,
                         knapsack_constraints[knapsack_idx].cstr_idx,
                         static_cast<i_t>(vars.size()),
                         signature});
    }
    CUOPT_LOG_DEBUG("Sorting signatures: %zu", sp_sigs.size());
    std::sort(sp_sigs.begin(), sp_sigs.end(), [](const auto& a, const auto& b) {
      if (a.signature != b.signature) { return a.signature < b.signature; }
      return a.size < b.size;
    });
    auto is_subset = [](const std::vector<i_t>& a, const std::vector<i_t>& b) {
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
    };
    auto fix_difference = [&](const std::vector<i_t>& superset, const std::vector<i_t>& subset) {
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
    };
    auto find_window_start = [&](long long signature) {
      auto it = std::lower_bound(
        sp_sigs.begin(), sp_sigs.end(), signature, [](const auto& a, long long value) {
          return a.signature < value;
        });
      return static_cast<size_t>(std::distance(sp_sigs.begin(), it));
    };
    CUOPT_LOG_DEBUG("Scanning extended cliques for dominance");
    for (i_t i = 0; i < n_extended_cliques; i++) {
      // Break here so that the discovered dominance is applied
      if (timer.check_time_limit()) {
        time_limit_reached = true;
        break;
      }
      i_t clique_idx          = extended_clique_start_idx + i;
      const auto& curr_clique = clique_table.first[clique_idx];
      if (curr_clique.empty()) { continue; }
      std::vector<i_t> curr_clique_vars(curr_clique.begin(), curr_clique.end());
      std::sort(curr_clique_vars.begin(), curr_clique_vars.end());
      cuopt_assert(
        std::unique(curr_clique_vars.begin(), curr_clique_vars.end()) == curr_clique_vars.end(),
        "Clique variables are not unique");
      long long signature = 0;
      for (auto v : curr_clique_vars) {
        signature += static_cast<long long>(v);
      }
      size_t start = find_window_start(signature);
      size_t end   = std::min(sp_sigs.size(), start + dominance_window);
      for (size_t idx = start; idx < end; idx++) {
        const auto& sp = sp_sigs[idx];
        if (sp.row_idx >= 0 && sp.row_idx < static_cast<i_t>(removal_marker.size()) &&
            removal_marker[sp.row_idx]) {
          continue;
        }
        const auto& vars_sp = cstr_vars[sp.knapsack_idx];
        if (vars_sp.size() > curr_clique_vars.size()) { continue; }
        if (!is_subset(vars_sp, curr_clique_vars)) { continue; }
        if (knapsack_constraints[sp.knapsack_idx].is_set_partitioning) {
          CUOPT_LOG_DEBUG("Fixing difference between clique %d and set packing constraint %d",
                          clique_idx,
                          sp.row_idx);
          // note that we never deleter set partitioning constraints but it fixes some other
          // variables
          fix_difference(curr_clique_vars, vars_sp);
        } else {
          if (sp.row_idx >= 0 && sp.row_idx < A.m) { removal_marker[sp.row_idx] = true; }
        }
      }
      if ((i % 128) == 0) {
        CUOPT_LOG_TRACE("Processed extended clique %d/%d", i + 1, n_extended_cliques);
      }
    }
    CUOPT_LOG_DEBUG("Dominance scan complete");
  }
  // TODO if more row removal is needed somewher else(e.g another presolve), standardize this
  dual_simplex::csr_matrix_t<i_t, f_t> A_removed(0, 0, 0);
  CUOPT_LOG_DEBUG("Removing dominated rows");
  A.remove_rows(removal_marker, A_removed);
  CUOPT_LOG_DEBUG("Rows removed, updating problem");
  A_removed.to_compressed_col(problem.A);
  problem.num_rows = A_removed.m;
  cuopt_assert(problem.rhs.size() == problem.row_sense.size(), "rhs and row sense size mismatch");
  i_t n = 0;
  // Remove problem.row_sense entries for which removal_marker is true, using remove_if
  auto new_end = std::remove_if(
    problem.row_sense.begin(), problem.row_sense.end(), [&removal_marker, &n](char) mutable {
      return removal_marker[n++];
    });
  // Compute count before erase invalidates the iterator
  size_t n_of_removed_constraints = std::distance(new_end, problem.row_sense.end());
  problem.row_sense.erase(new_end, problem.row_sense.end());
  n = 0;
  // Remove problem.rhs entries for which removal_marker is true, using remove_if
  auto new_end_rhs =
    std::remove_if(problem.rhs.begin(), problem.rhs.end(), [&removal_marker, &n](f_t) mutable {
      return removal_marker[n++];
    });
  problem.rhs.erase(new_end_rhs, problem.rhs.end());
  CUOPT_LOG_DEBUG("Number of removed constraints by clique covering: %d", n_of_removed_constraints);
  cuopt_assert(problem.rhs.size() == problem.row_sense.size(), "rhs and row sense size mismatch");
  cuopt_assert(problem.A.m == problem.rhs.size(), "matrix and num rows mismatch after removal");
  // Renumber the ranged row indices in problem.range_rows to ensure consistency after constraint
  // removals. Create a mapping from old indices to new indices.
  if (!problem.range_rows.empty()) {
    std::vector<i_t> old_to_new_indices;
    old_to_new_indices.reserve(removal_marker.size());
    i_t new_idx = 0;
    for (size_t i = 0; i < removal_marker.size(); ++i) {
      if (!removal_marker[i]) {
        old_to_new_indices.push_back(new_idx++);
      } else {
        old_to_new_indices.push_back(-1);  // removed constraint
      }
    }
    // Remove entries from range_rows and range_value where the underlying row has been removed.
    std::vector<i_t> new_range_rows;
    std::vector<f_t> new_range_values;
    for (size_t i = 0; i < problem.range_rows.size(); ++i) {
      i_t old_row = problem.range_rows[i];
      if (old_row >= 0 && old_row < (i_t)removal_marker.size() && !removal_marker[old_row]) {
        i_t new_row = old_to_new_indices[old_row];
        cuopt_assert(new_row != -1, "Invalid new row index for ranged row renumbering");
        new_range_rows.push_back(new_row);
        new_range_values.push_back(problem.range_value[i]);
      }
      // else: the ranged row was removed, so we skip it
    }
    problem.range_rows  = std::move(new_range_rows);
    problem.range_value = std::move(new_range_values);
  }
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
                          cuopt::timer_t& timer)
{
  cuopt::timer_t stage_timer(std::numeric_limits<double>::infinity());
  double t_fill   = 0.;
  double t_coeff  = 0.;
  double t_sort   = 0.;
  double t_find   = 0.;
  double t_small  = 0.;
  double t_maps   = 0.;
  double t_extend = 0.;
  double t_remove = 0.;
  std::vector<knapsack_constraint_t<i_t, f_t>> knapsack_constraints;
  std::unordered_set<i_t> set_packing_constraints;
  dual_simplex::csr_matrix_t<i_t, f_t> A(problem.num_rows, problem.num_cols, 0);
  problem.A.to_compressed_row(A);
  fill_knapsack_constraints(problem, knapsack_constraints, A);
  t_fill = stage_timer.elapsed_time();
  make_coeff_positive_knapsack_constraint(
    problem, knapsack_constraints, set_packing_constraints, tolerances);
  t_coeff = stage_timer.elapsed_time();
  sort_csr_by_constraint_coefficients(knapsack_constraints);
  t_sort = stage_timer.elapsed_time();
  // print_knapsack_constraints(knapsack_constraints);
  // TODO think about getting min_clique_size according to some problem property
  clique_config_t clique_config;
  clique_table_t<i_t, f_t> clique_table(2 * problem.num_cols,
                                        clique_config.min_clique_size,
                                        clique_config.max_clique_size_for_extension);
  clique_table.tolerances = tolerances;
  for (const auto& knapsack_constraint : knapsack_constraints) {
    if (timer.check_time_limit()) { break; }
    find_cliques_from_constraint(knapsack_constraint, clique_table);
  }
  if (timer.check_time_limit()) { return; }
  t_find = stage_timer.elapsed_time();
  CUOPT_LOG_DEBUG("Number of cliques: %d, additional cliques: %d",
                  clique_table.first.size(),
                  clique_table.addtl_cliques.size());
  // print_clique_table(clique_table);
  // remove small cliques and add them to adj_list
  remove_small_cliques(clique_table);
  t_small = stage_timer.elapsed_time();
  // fill var clique maps
  fill_var_clique_maps(clique_table);
  t_maps                 = stage_timer.elapsed_time();
  i_t n_extended_cliques = extend_cliques(knapsack_constraints, clique_table, problem, A, timer);
  t_extend               = stage_timer.elapsed_time();
  remove_dominated_cliques(problem,
                           A,
                           clique_table,
                           set_packing_constraints,
                           knapsack_constraints,
                           n_extended_cliques,
                           timer);
  t_remove = stage_timer.elapsed_time();
  CUOPT_LOG_DEBUG(
    "Clique table timing (s): fill=%.6f coeff=%.6f sort=%.6f find=%.6f small=%.6f maps=%.6f "
    "extend=%.6f remove=%.6f total=%.6f",
    t_fill,
    t_coeff - t_fill,
    t_sort - t_coeff,
    t_find - t_sort,
    t_small - t_find,
    t_maps - t_small,
    t_extend - t_maps,
    t_remove - t_extend,
    t_remove);
  // exit(0);
}

#define INSTANTIATE(F_TYPE)                                               \
  template void find_initial_cliques<int, F_TYPE>(                        \
    dual_simplex::user_problem_t<int, F_TYPE> & problem,                  \
    typename mip_solver_settings_t<int, F_TYPE>::tolerances_t tolerances, \
    cuopt::timer_t & timer);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif
#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif
#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
