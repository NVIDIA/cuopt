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
#include <cmath>
#include <limits>
#include <linear_algebra/sparse_matrix.hpp>
#include <linear_algebra/sparse_vector.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/utils.cuh>
#include <utilities/logger.hpp>
#include <utilities/macros.cuh>
#include <utilities/timer.hpp>

namespace cuopt::mathematical_optimization::mip {

using simplex::user_problem_t;

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
  const user_problem_t<i_t, f_t>& problem,
  std::vector<knapsack_constraint_t<i_t, f_t>>& knapsack_constraints,
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
    cuopt_assert(knapsack_constraint.rhs >= 0, "RHS must be non-negative");
  }
}

// convert all the knapsack constraints
// if a binary variable has a negative coefficient, put its negation in the constraint
template <typename i_t, typename f_t>
void fill_knapsack_constraints(const user_problem_t<i_t, f_t>& problem,
                               std::vector<knapsack_constraint_t<i_t, f_t>>& knapsack_constraints,
                               csr_matrix_t<i_t, f_t>& A)
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
      if (problem.var_types[A.j[j]] == simplex::variable_type_t::CONTINUOUS ||
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
      // Final partitioning check is done after coefficient normalization in
      // make_coeff_positive_knapsack_constraint.
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
      // Negative ids prevent aliasing with real row indices.
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
  std::vector<std::pair<i_t, i_t>> small_edges;

  // Demote sub-threshold first-cliques into pairwise edges.
  for (size_t clique_idx = 0; clique_idx < clique_table.first.size(); clique_idx++) {
    if (timer.check_time_limit()) { return; }
    const auto& clique = clique_table.first[clique_idx];
    if (clique.size() <= (size_t)clique_table.min_clique_size) {
      for (size_t i = 0; i < clique.size(); i++) {
        for (size_t j = 0; j < clique.size(); j++) {
          if (i == j) { continue; }
          small_edges.emplace_back(clique[i], clique[j]);
        }
      }
      num_removed_first++;
      to_delete[clique_idx] = true;
    }
  }
  std::vector<bool> addtl_to_delete(clique_table.addtl_cliques.size(), false);
  for (size_t addtl_c = 0; addtl_c < clique_table.addtl_cliques.size(); addtl_c++) {
    const auto& addtl_clique   = clique_table.addtl_cliques[addtl_c];
    const auto base_clique_idx = static_cast<size_t>(addtl_clique.clique_idx);
    cuopt_assert(base_clique_idx < to_delete.size(),
                 "Additional clique points to invalid base clique index");
    const bool drop_because_base = to_delete[base_clique_idx];
    const i_t extended_size =
      clique_table.first[base_clique_idx].size() - addtl_clique.start_pos_on_clique + 1;
    const bool drop_because_small = extended_size < clique_table.min_clique_size;
    if (!drop_because_base && !drop_because_small) { continue; }

    for (size_t i = addtl_clique.start_pos_on_clique;
         i < clique_table.first[base_clique_idx].size();
         i++) {
      const i_t base_member = clique_table.first[base_clique_idx][i];
      small_edges.emplace_back(base_member, addtl_clique.vertex_idx);
      small_edges.emplace_back(addtl_clique.vertex_idx, base_member);
    }
    addtl_to_delete[addtl_c] = true;
    num_removed_addtl++;
  }
  {
    size_t old_addtl_idx = 0;
    auto addtl_it        = std::remove_if(clique_table.addtl_cliques.begin(),
                                   clique_table.addtl_cliques.end(),
                                   [&](const auto&) { return addtl_to_delete[old_addtl_idx++]; });
    clique_table.addtl_cliques.erase(addtl_it, clique_table.addtl_cliques.end());
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
  clique_table.small_clique_adj.finalize_from_unsorted_pairs(2 * clique_table.n_variables,
                                                             small_edges);
  // Force degree recompute after structural changes.
  std::fill(clique_table.var_degrees.begin(), clique_table.var_degrees.end(), -1);
}

template <typename i_t, typename f_t>
std::unordered_set<i_t> clique_table_t<i_t, f_t>::get_adj_set_of_var(i_t var_idx) const
{
  std::unordered_set<i_t> adj_set;

  // First-clique edges: every member of each first-clique containing var_idx.
  for (i_t clique_idx : var_clique_first.slice(var_idx)) {
    const auto& c = first[clique_idx];
    adj_set.insert(c.begin(), c.end());
  }

  // Addtl-clique edges.
  for (i_t addtl_idx : var_clique_addtl.slice(var_idx)) {
    const auto& a = addtl_cliques[addtl_idx];
    if (a.vertex_idx == var_idx) {
      // var_idx is the extension vertex; new neighbors are the base suffix.
      const auto& base = first[a.clique_idx];
      adj_set.insert(base.begin() + a.start_pos_on_clique, base.end());
    } else {
      // var_idx is a base member; only new edge is to the extension vertex.
      adj_set.insert(a.vertex_idx);
    }
  }

  for (i_t adj_var : small_clique_adj.slice(var_idx)) {
    adj_set.insert(adj_var);
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
bool clique_table_t<i_t, f_t>::check_adjacency(i_t var_idx1, i_t var_idx2) const
{
  if (var_idx1 == var_idx2) { return false; }
  if (var_idx1 % n_variables == var_idx2 % n_variables) { return true; }

  // small_clique_adj is symmetric, so probe either direction.
  if (small_clique_adj.slice_contains(var_idx1, var_idx2)) { return true; }

  // Probe through the var with the smaller var_clique_first slice.
  {
    i_t probe_var  = var_idx1;
    i_t target_var = var_idx2;
    if (var_clique_first.slice_size(var_idx1) > var_clique_first.slice_size(var_idx2)) {
      probe_var  = var_idx2;
      target_var = var_idx1;
    }
    for (i_t clique_idx : var_clique_first.slice(probe_var)) {
      if (first_var_positions[clique_idx].count(target_var) > 0) { return true; }
    }
  }

  for (i_t addtl_idx : var_clique_addtl.slice(var_idx1)) {
    const auto& addtl   = addtl_cliques[addtl_idx];
    const auto& pos_map = first_var_positions[addtl.clique_idx];
    auto pos_it         = pos_map.find(var_idx2);
    if (pos_it != pos_map.end() && pos_it->second >= addtl.start_pos_on_clique) { return true; }
  }
  for (i_t addtl_idx : var_clique_addtl.slice(var_idx2)) {
    const auto& addtl   = addtl_cliques[addtl_idx];
    const auto& pos_map = first_var_positions[addtl.clique_idx];
    auto pos_it         = pos_map.find(var_idx1);
    if (pos_it != pos_map.end() && pos_it->second >= addtl.start_pos_on_clique) { return true; }
  }

  return false;
}

// Returns true on success; `work_out` accumulates scan/hash ops as a
// near-uniform wall-time proxy.
template <typename i_t, typename f_t>
bool extend_clique(const std::vector<i_t>& clique,
                   clique_table_t<i_t, f_t>& clique_table,
                   double& work_out)
{
  i_t smallest_degree     = std::numeric_limits<i_t>::max();
  i_t smallest_degree_var = -1;
  for (size_t idx = 0; idx < clique.size(); idx++) {
    i_t var_idx = clique[idx];
    i_t degree  = clique_table.get_degree_of_var(var_idx);
    if (degree < smallest_degree) {
      smallest_degree     = degree;
      smallest_degree_var = var_idx;
    }
  }
  work_out += static_cast<double>(clique.size());

  auto smallest_degree_adj_set = clique_table.get_adj_set_of_var(smallest_degree_var);
  const double D               = static_cast<double>(smallest_degree_adj_set.size());
  work_out += D;

  std::unordered_set<i_t> clique_members(clique.begin(), clique.end());
  work_out += static_cast<double>(clique.size());

  std::vector<i_t> extension_candidates;
  extension_candidates.reserve(smallest_degree_adj_set.size());
  for (const auto& candidate : smallest_degree_adj_set) {
    if (clique_members.find(candidate) == clique_members.end()) {
      extension_candidates.push_back(candidate);
    }
  }
  work_out += D;

  std::sort(extension_candidates.begin(), extension_candidates.end(), [&](i_t a, i_t b) {
    return clique_table.get_degree_of_var(a) > clique_table.get_degree_of_var(b);
  });
  const double C = static_cast<double>(extension_candidates.size());
  if (C > 1.0) { work_out += C * std::log2(C); }

  auto new_clique = clique;
  for (size_t idx = 0; idx < extension_candidates.size(); idx++) {
    i_t var_idx = extension_candidates[idx];
    bool add    = true;
    for (size_t i = 0; i < new_clique.size(); i++) {
      work_out += 1.0;
      if (!clique_table.check_adjacency(var_idx, new_clique[i])) {
        add = false;
        break;
      }
    }
    if (add) { new_clique.push_back(var_idx); }
  }

  if (new_clique.size() > clique.size()) {
    clique_table.first.push_back(new_clique);
    for (const auto& var_idx : new_clique) {
      clique_table.var_degrees[var_idx] = -1;
    }
    work_out += static_cast<double>(new_clique.size());
#if DEBUG_KNAPSACK_CONSTRAINTS
    CUOPT_LOG_DEBUG("Extended clique: %lu from %lu", new_clique.size(), clique.size());
#endif
    return true;
  }
  return false;
}

template <typename i_t>
struct extension_candidate_t {
  i_t knapsack_idx;
  i_t estimated_gain;
  i_t clique_size;
};

template <typename i_t>
bool compare_extension_candidate(const extension_candidate_t<i_t>& a,
                                 const extension_candidate_t<i_t>& b)
{
  if (a.estimated_gain != b.estimated_gain) { return a.estimated_gain > b.estimated_gain; }
  if (a.clique_size != b.clique_size) { return a.clique_size < b.clique_size; }
  return a.knapsack_idx < b.knapsack_idx;
}

// Extends set-packing cliques. Soft floor: min_work; hard ceiling: max_work
// or `timer`. signal_extend only honored after min_work.
template <typename i_t, typename f_t>
i_t extend_cliques(const std::vector<knapsack_constraint_t<i_t, f_t>>& knapsack_constraints,
                   clique_table_t<i_t, f_t>& clique_table,
                   cuopt::timer_t& timer,
                   double* work_estimate_out,
                   double min_work,
                   double max_work,
                   omp_atomic_t<bool>* signal_extend)
{
  constexpr i_t min_extension_gain = 2;

  double local_work = 0.0;
  double& work      = work_estimate_out ? *work_estimate_out : local_work;

  std::vector<extension_candidate_t<i_t>> extension_worklist;
  extension_worklist.reserve(knapsack_constraints.size());
  for (i_t knapsack_idx = 0; knapsack_idx < static_cast<i_t>(knapsack_constraints.size());
       knapsack_idx++) {
    if (timer.check_time_limit()) { break; }
    if (work >= max_work) { break; }
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
    work += static_cast<double>(knapsack_constraint.entries.size());
  }
  std::stable_sort(
    extension_worklist.begin(), extension_worklist.end(), compare_extension_candidate<i_t>);
  if (!extension_worklist.empty()) {
    work += static_cast<double>(extension_worklist.size()) *
            std::log2(static_cast<double>(extension_worklist.size()));
  }
  CUOPT_LOG_DEBUG("Clique extension candidates after scoring: %zu", extension_worklist.size());

  i_t n_extended_cliques = 0;
  for (const auto& candidate : extension_worklist) {
    if (timer.check_time_limit()) { break; }
    if (work >= min_work) {
      if (work >= max_work) { break; }
      if (signal_extend && signal_extend->load(std::memory_order_acquire)) {
        CUOPT_LOG_DEBUG("Stopping clique extension: cut-pass signal received (work=%.0f)", work);
        break;
      }
    }
    const auto& knapsack_constraint = knapsack_constraints[candidate.knapsack_idx];
    std::vector<i_t> clique;
    clique.reserve(knapsack_constraint.entries.size());
    for (const auto& entry : knapsack_constraint.entries) {
      clique.push_back(entry.col);
    }
    if (extend_clique(clique, clique_table, work)) { n_extended_cliques++; }
  }
  CUOPT_LOG_DEBUG("Number of extended cliques: %d (work=%.0f)", n_extended_cliques, work);
  return n_extended_cliques;
}

template <typename i_t, typename f_t>
void fill_var_clique_maps(clique_table_t<i_t, f_t>& clique_table)
{
  const i_t n_vertices = 2 * clique_table.n_variables;

  // first_var_positions: per-clique hash map (cliques small ⇒ hash beats binary search).
  clique_table.first_var_positions.assign(clique_table.first.size(), {});

  std::vector<std::pair<i_t, i_t>> first_pairs;
  size_t total_first_members = 0;
  for (const auto& c : clique_table.first) {
    total_first_members += c.size();
  }
  first_pairs.reserve(total_first_members);

  for (size_t clique_idx = 0; clique_idx < clique_table.first.size(); clique_idx++) {
    const auto& clique = clique_table.first[clique_idx];
    auto& pos_map      = clique_table.first_var_positions[clique_idx];
    pos_map.reserve(clique.size());
    for (size_t idx = 0; idx < clique.size(); idx++) {
      const i_t var_idx = clique[idx];
      first_pairs.emplace_back(var_idx, static_cast<i_t>(clique_idx));
      pos_map[var_idx] = static_cast<i_t>(idx);
    }
  }
  clique_table.var_clique_first.finalize_from_unsorted_pairs(n_vertices, first_pairs);

  std::vector<std::pair<i_t, i_t>> addtl_pairs;
  for (size_t addtl_c = 0; addtl_c < clique_table.addtl_cliques.size(); ++addtl_c) {
    const auto& a = clique_table.addtl_cliques[addtl_c];
    addtl_pairs.emplace_back(a.vertex_idx, static_cast<i_t>(addtl_c));
    const auto& base = clique_table.first[a.clique_idx];
    for (i_t pos = a.start_pos_on_clique; pos < static_cast<i_t>(base.size()); ++pos) {
      addtl_pairs.emplace_back(base[pos], static_cast<i_t>(addtl_c));
    }
  }
  clique_table.var_clique_addtl.finalize_from_unsorted_pairs(n_vertices, addtl_pairs);
}

template <typename i_t, typename f_t>
void clique_table_t<i_t, f_t>::set_small_clique_adj_for_test(
  const std::unordered_map<i_t, std::unordered_set<i_t>>& edges)
{
  std::vector<std::pair<i_t, i_t>> pairs;
  size_t total = 0;
  for (const auto& kv : edges) {
    total += kv.second.size();
  }
  pairs.reserve(total);
  for (const auto& kv : edges) {
    for (const auto& v : kv.second) {
      pairs.emplace_back(kv.first, v);
    }
  }
  small_clique_adj.finalize_from_unsorted_pairs(2 * n_variables, pairs);
}

template <typename i_t, typename f_t>
void build_clique_table(const user_problem_t<i_t, f_t>& problem,
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
  csr_matrix_t<i_t, f_t> A(problem.num_rows, problem.num_cols, 0);
  problem.A.to_compressed_row(A);
  fill_knapsack_constraints(problem, knapsack_constraints, A);
  make_coeff_positive_knapsack_constraint(problem, knapsack_constraints, tolerances);
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
void find_initial_cliques(user_problem_t<i_t, f_t>& problem,
                          typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances,
                          std::shared_ptr<clique_table_t<i_t, f_t>>& clique_table_out,
                          cuopt::timer_t& timer,
                          omp_atomic_t<bool>* signal_extend)
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
  csr_matrix_t<i_t, f_t> A(problem.num_rows, problem.num_cols, 0);
  problem.A.to_compressed_row(A);
  fill_knapsack_constraints(problem, knapsack_constraints, A);
#ifdef DEBUG_CLIQUE_TABLE
  t_fill = stage_timer.elapsed_time();
#endif
  make_coeff_positive_knapsack_constraint(problem, knapsack_constraints, tolerances);
#ifdef DEBUG_CLIQUE_TABLE
  t_coeff = stage_timer.elapsed_time();
#endif
  sort_csr_by_constraint_coefficients(knapsack_constraints);
#ifdef DEBUG_CLIQUE_TABLE
  t_sort = stage_timer.elapsed_time();
#endif
  clique_config_t clique_config;
  auto clique_table =
    std::make_shared<clique_table_t<i_t, f_t>>(2 * problem.num_cols,
                                               clique_config.min_clique_size,
                                               clique_config.max_clique_size_for_extension);
  clique_table->tolerances                 = tolerances;
  double time_limit_for_additional_cliques = timer.remaining_time() / 2;
  cuopt::timer_t additional_cliques_timer(time_limit_for_additional_cliques);
  double find_work_estimate = 0.0;
  // Always build base cliques in full; signal_extend only gates the extension phase.
  for (const auto& knapsack_constraint : knapsack_constraints) {
    if (timer.check_time_limit()) { break; }
    find_cliques_from_constraint(knapsack_constraint, *clique_table, additional_cliques_timer);
    find_work_estimate += knapsack_constraint.entries.size();
  }
#ifdef DEBUG_CLIQUE_TABLE
  t_find = stage_timer.elapsed_time();
#endif
  CUOPT_LOG_DEBUG("Number of cliques: %d, additional cliques: %d, find_work=%.0f",
                  clique_table->first.size(),
                  clique_table->addtl_cliques.size(),
                  find_work_estimate);
  remove_small_cliques(*clique_table, timer);
#ifdef DEBUG_CLIQUE_TABLE
  t_small = stage_timer.elapsed_time();
#endif
  fill_var_clique_maps(*clique_table);
#ifdef DEBUG_CLIQUE_TABLE
  t_maps = stage_timer.elapsed_time();
#endif
  // Publish the base table so cut generation can start using it; the extension
  // phase below keeps mutating *clique_table, so the consumer must signal this
  // task to stop and join it (taskwait) before reading the table.
  clique_table_out       = clique_table;
  double extend_work     = 0.0;
  i_t n_extended_cliques = extend_cliques(knapsack_constraints,
                                          *clique_table,
                                          timer,
                                          &extend_work,
                                          clique_config.min_extend_work,
                                          clique_config.max_extend_work,
                                          signal_extend);
  if (n_extended_cliques > 0) { fill_var_clique_maps(*clique_table); }
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

#define INSTANTIATE(F_TYPE)                                                                    \
  template void find_initial_cliques<int, F_TYPE>(                                             \
    user_problem_t<int, F_TYPE> & problem,                                                     \
    typename mip_solver_settings_t<int, F_TYPE>::tolerances_t tolerances,                      \
    std::shared_ptr<clique_table_t<int, F_TYPE>> & clique_table_out,                           \
    cuopt::timer_t & timer,                                                                    \
    omp_atomic_t<bool> * signal_extend);                                                       \
  template void build_clique_table<int, F_TYPE>(                                               \
    const user_problem_t<int, F_TYPE>& problem,                                                \
    clique_table_t<int, F_TYPE>& clique_table,                                                 \
    typename mip_solver_settings_t<int, F_TYPE>::tolerances_t tolerances,                      \
    bool remove_small_cliques_flag,                                                            \
    bool fill_var_clique_maps_flag,                                                            \
    cuopt::timer_t& timer);                                                                    \
  template void fill_var_clique_maps<int, F_TYPE>(clique_table_t<int, F_TYPE> & clique_table); \
  template class clique_table_t<int, F_TYPE>;

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif
#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif
#undef INSTANTIATE

}  // namespace cuopt::mathematical_optimization::mip
