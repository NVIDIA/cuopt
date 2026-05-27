/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuts/objective_step.hpp>
#include <cuts/rational.hpp>

#include <utilities/macros.cuh>

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

namespace {

// Solve constraint i for solve_for under rational arithmetic and return the resulting
// lattice (step, bias) for solve_for. The RHS accumulator b starts at rhs_i and is
// reduced by every other variable's known lattice contribution, leaving only solve_for's
// term on the LHS; step_sum accumulates a gcd over those contributions.
//
// Returns {0, 0} if solve_for cannot be determined (its coefficient in constraint i is
// zero, or no other variable in the constraint has a known lattice). The caller treats
// a zero step as "no discovery" and skips the update.
template <typename i_t, typename f_t>
std::pair<rational128_t<f_t>, rational128_t<f_t>> compute_lattice_for_unknown(
  i_t i,
  i_t solve_for,
  const std::vector<i_t>& offsets,
  const std::vector<i_t>& variables,
  const std::vector<rational128_t<f_t>>& coef_r,
  const std::vector<rational128_t<f_t>>& step_r,
  const std::vector<rational128_t<f_t>>& bias_r,
  const rational128_t<f_t>& rhs_i)
{
  using R     = rational128_t<f_t>;
  R a_unknown = {0, 1};
  R b         = rhs_i;
  R step_sum  = {0, 1};

  for (i_t p = offsets[i]; p < offsets[i + 1]; ++p) {
    i_t j = variables[p];

    if (j == solve_for) {
      a_unknown = coef_r[p];
      continue;
    }
    if (step_r[j].is_zero()) continue;

    b        = b - coef_r[p] * bias_r[j];
    step_sum = gcd(step_sum, (coef_r[p] * step_r[j]).abs());
  }

  if (a_unknown.is_zero()) return {R{0, 1}, R{0, 1}};

  return {step_sum / a_unknown.abs(), b / a_unknown};
}

}  // namespace

template <typename i_t, typename f_t>
bool propagate_lattice(i_t n_vars,
                       i_t n_cons,
                       const std::vector<i_t>& offsets,
                       const std::vector<i_t>& variables,
                       const std::vector<f_t>& coefficients,
                       const std::vector<f_t>& con_lb,
                       const std::vector<f_t>& con_ub,
                       const std::vector<f_t>& var_lb,
                       const std::vector<bool>& is_lattice_known_initially,
                       std::vector<f_t>& lattice_step,
                       std::vector<f_t>& lattice_bias)
{
  constexpr f_t eq_tol = 1e-8;

  // Track lattice as rationals: step_r[j] and bias_r[j]
  // step_r[j].p == 0 means unknown
  std::vector<rational128_t<f_t>> step_r(n_vars);
  std::vector<rational128_t<f_t>> bias_r(n_vars);

  for (i_t j = 0; j < n_vars; ++j) {
    if (is_lattice_known_initially[j]) {
      step_r[j] = {1, 1};
      bias_r[j] = rational128_t<f_t>::from_floating_point(var_lb[j]);
    }
  }

  // Rationalize all matrix coefficients once
  std::vector<rational128_t<f_t>> coef_r(coefficients.size());
  for (size_t p = 0; p < coefficients.size(); ++p) {
    coef_r[p] = rational128_t<f_t>::from_floating_point(coefficients[p]);
  }

  // Per-constraint RHS (rationalized) for equality rows, plus a flag marking each row
  // as an equality or an inequality. Lattice propagation is only sound on equality rows
  // where exactly one unknown remains: the row then pins that unknown's value modulo
  // a lattice. Inequality rows carry an implicit slack that bounds rather than pins
  // the chosen variable, so we exclude them from the worklist below. The caller must guarantee no
  // constraint is free, i.e. every row has at least one finite bound, and the equality
  // check ensures that finite bound exists on both sides.
  std::vector<rational128_t<f_t>> rhs(n_cons);
  std::vector<bool> is_equality(n_cons);
  for (i_t i = 0; i < n_cons; ++i) {
    bool lb_finite = std::isfinite(con_lb[i]);
    bool ub_finite = std::isfinite(con_ub[i]);
    cuopt_assert(lb_finite || ub_finite, "propagate_lattice: free constraints are not supported");

    if (lb_finite && ub_finite && std::abs(con_lb[i] - con_ub[i]) < eq_tol) {
      rhs[i]         = rational128_t<f_t>::from_floating_point(con_lb[i]);
      is_equality[i] = true;
    }
  }

  // Track how many constraints each variable appears in.
  std::vector<i_t> constraints_per_variable(n_vars, 0);
  for (i_t i = 0; i < n_cons; ++i) {
    for (i_t p = offsets[i]; p < offsets[i + 1]; ++p) {
      constraints_per_variable[variables[p]]++;
    }
  }

  // Convert the CSR representation to CSC for fast access. Don't store nonzero values
  // Just store the adjacency information / sparsity pattern.
  struct csc_adjacency_t {
    std::vector<i_t> col_start;
    std::vector<i_t> i;
  };
  csc_adjacency_t A_col;
  // Compute the col_start array by taking the cumulative sum of constraints_per_variable.
  A_col.col_start.assign(n_vars + 1, 0);
  for (i_t j = 0; j < n_vars; ++j) {
    A_col.col_start[j + 1] = A_col.col_start[j] + constraints_per_variable[j];
  }
  // Use the col_start array to populate the i array. This shifts col_start by one to the right
  A_col.i.resize(A_col.col_start[n_vars]);
  for (i_t i = 0; i < n_cons; ++i) {
    for (i_t p = offsets[i]; p < offsets[i + 1]; ++p) {
      A_col.i[A_col.col_start[variables[p]]++] = i;
    }
  }
  // Restore the col_start array
  i_t carry = 0;
  for (i_t j = 0; j <= n_vars; ++j) {
    const i_t next     = A_col.col_start[j];
    A_col.col_start[j] = carry;
    carry              = next;
  }

  // Number of currently-unknown variables in each constraint. Initialized once and
  // maintained incrementally: on each discovery we decrement unknown_count_per_constraint
  // for every constraint containing the discovered variable. A constraint becomes
  // productive once this drops to exactly 1 (and the row is an equality).
  std::vector<i_t> unknown_count_per_constraint(n_cons, 0);
  i_t max_unknown_count = 0;
  for (i_t i = 0; i < n_cons; ++i) {
    for (i_t p = offsets[i]; p < offsets[i + 1]; ++p) {
      i_t j = variables[p];
      if (step_r[j].is_zero()) { unknown_count_per_constraint[i]++; }
    }
    if (unknown_count_per_constraint[i] > max_unknown_count) {
      max_unknown_count = unknown_count_per_constraint[i];
    }
  }

  // Iteratively propagate using a worklist of constraints that may yield a new lattice
  // discovery on the next pass. We seed it with every equality row. Each pass scans the worklist,
  // collects the variables whose lattices were newly determined, then rebuilds the worklist as the
  // union (deduplicated) of equality rows touched by any of those variables.
  bool any_discovered = false;

  // Worklist buffers
  std::vector<i_t> changed_constraints(n_cons);
  std::vector<i_t> next_changed_constraints(n_cons);
  std::vector<i_t> sorted_constraints(n_cons);
  i_t num_changed = 0;
  for (i_t i = 0; i < n_cons; ++i) {
    if (is_equality[i]) { changed_constraints[num_changed++] = i; }
  }

  // Reused across passes to avoid reallocation.
  std::vector<i_t> discovered_variables;
  std::vector<i_t> in_next_pass(n_cons, 0);

  // bucket_offset[k] is first the number of constraints with k unknowns and then, after
  // the prefix sum, the running insertion position for bucket k in sorted_constraints.
  // Sized once using the fact that unknown counts only decrease over time.
  std::vector<i_t> bucket_offset(max_unknown_count + 1, 0);

  while (num_changed > 0) {
    // Counting-sort the active prefix of changed_constraints by unknown_count_per_constraint
    // ascending. Sorting is linear in num_changed + max_unknown_count and lets us process
    // constraints with few unknowns first, since those are the ones most likely to
    // discover a variable and unblock other constraints in the same pass.
    for (i_t k = 0; k <= max_unknown_count; ++k) {
      bucket_offset[k] = 0;
    }
    for (i_t k = 0; k < num_changed; ++k) {
      bucket_offset[unknown_count_per_constraint[changed_constraints[k]]]++;
    }
    // Prefix sum: bucket_offset[k] becomes the start of bucket k in the sorted output.
    i_t running = 0;
    for (i_t k = 0; k <= max_unknown_count; ++k) {
      i_t cnt          = bucket_offset[k];
      bucket_offset[k] = running;
      running += cnt;
    }
    for (i_t k = 0; k < num_changed; ++k) {
      i_t i                                                                = changed_constraints[k];
      sorted_constraints[bucket_offset[unknown_count_per_constraint[i]]++] = i;
    }
    changed_constraints.swap(sorted_constraints);

    discovered_variables.clear();

    for (i_t k = 0; k < num_changed; ++k) {
      i_t i = changed_constraints[k];
      // The worklist contains only equality rows and an
      // equality row is productive exactly when one unknown remains: that unknown is
      // then determined by the row given the known variables' lattices.
      if (unknown_count_per_constraint[i] != 1) continue;

      i_t j = -1;
      for (i_t p = offsets[i]; p < offsets[i + 1]; ++p) {
        const i_t candidate = variables[p];
        if (step_r[candidate].is_zero()) {
          j = candidate;
          break;
        }
      }
      if (j < 0) continue;

      auto [new_step, new_bias] = compute_lattice_for_unknown<i_t, f_t>(
        i, j, offsets, variables, coef_r, step_r, bias_r, rhs[i]);

      if (!new_step.is_zero()) {
        step_r[j] = new_step.reduced();
        bias_r[j] = new_bias.reduced();
        discovered_variables.push_back(j);
        // Every constraint touching j now has one fewer unknown.
        const i_t col_start = A_col.col_start[j];
        const i_t col_end   = A_col.col_start[j + 1];
        for (i_t p = col_start; p < col_end; ++p) {
          unknown_count_per_constraint[A_col.i[p]]--;
        }
        // j had step_r zero, which means is_lattice_known_initially[j]
        // was false: any discovery here is therefore of an originally-unknown variable.
        any_discovered = true;
      }
    }

    // Build next pass's worklist as the union of equality rows touched by any
    // discovered variable, deduplicated through the in_next_pass flag array. Inequality
    // rows are filtered out as well.
    i_t num_next = 0;
    for (i_t j : discovered_variables) {
      const i_t col_start = A_col.col_start[j];
      const i_t col_end   = A_col.col_start[j + 1];
      for (i_t p = col_start; p < col_end; ++p) {
        const i_t i = A_col.i[p];
        if (!is_equality[i]) continue;
        if (!in_next_pass[i]) {
          in_next_pass[i]                      = true;
          next_changed_constraints[num_next++] = i;
        }
      }
    }
    // Clear in_next_pass
    for (i_t k = 0; k < num_next; ++k) {
      in_next_pass[next_changed_constraints[k]] = false;
    }

    changed_constraints.swap(next_changed_constraints);
    num_changed = num_next;
  }

  // Convert back to f_t
  lattice_step.assign(n_vars, f_t(0));
  lattice_bias.assign(n_vars, f_t(0));
  for (i_t j = 0; j < n_vars; ++j) {
    if (!step_r[j].is_zero()) {
      lattice_step[j] = step_r[j].to_floating_point();
      lattice_bias[j] = bias_r[j].to_floating_point();
    }
  }

  return any_discovered;
}

template <typename i_t, typename f_t>
objective_step_t<f_t> compute_objective_step_info(
  const std::vector<f_t>& obj_coefs,
  const std::vector<f_t>& var_lb,
  const std::vector<bool>& is_lattice_known_initially,
  const std::vector<i_t>& offsets,
  const std::vector<i_t>& variables,
  const std::vector<f_t>& coefficients,
  const std::vector<f_t>& con_lb,
  const std::vector<f_t>& con_ub)
{
  const i_t n_variables = static_cast<i_t>(obj_coefs.size());

  // Caller is expected to have checked the all-lattice-known fast path; here we run
  // lattice propagation on the equality rows of the constraint matrix to discover the
  // lattice of any remaining continuous (and not implied-integer) objective variables.
  std::vector<f_t> lattice_step, lattice_bias;
  bool discovered = propagate_lattice<i_t, f_t>(n_variables,
                                                static_cast<i_t>(con_lb.size()),
                                                offsets,
                                                variables,
                                                coefficients,
                                                con_lb,
                                                con_ub,
                                                var_lb,
                                                is_lattice_known_initially,
                                                lattice_step,
                                                lattice_bias);

  if (!discovered) return {};

  // Combine using rational arithmetic to compute objective step and bias.
  // gcd(a/b, c/d) = gcd(a*d, c*b) / (b*d) -- implemented by gcd_floating_point.
  f_t obj_step = 0;
  f_t obj_bias = 0;

  for (i_t i = 0; i < n_variables; ++i) {
    if (obj_coefs[i] == 0) continue;
    if (lattice_step[i] == 0) return {};  // Still unknown -- give up.

    f_t coef = obj_coefs[i];
    obj_step = gcd_floating_point<f_t>(obj_step, std::abs(coef * lattice_step[i]));
    obj_bias += coef * lattice_bias[i];
  }

  if (obj_step > 1e-12) {
    objective_step_t<f_t> result;
    result.step_size = obj_step;
    result.bias      = std::fmod(obj_bias, obj_step);
    if (result.bias < 0) result.bias += obj_step;
    return result;
  }
  return {};
}

// Explicit instantiations for the (i_t, f_t) combinations used by problem_t.
template bool propagate_lattice<int, double>(int,
                                             int,
                                             const std::vector<int>&,
                                             const std::vector<int>&,
                                             const std::vector<double>&,
                                             const std::vector<double>&,
                                             const std::vector<double>&,
                                             const std::vector<double>&,
                                             const std::vector<bool>&,
                                             std::vector<double>&,
                                             std::vector<double>&);

template bool propagate_lattice<int, float>(int,
                                            int,
                                            const std::vector<int>&,
                                            const std::vector<int>&,
                                            const std::vector<float>&,
                                            const std::vector<float>&,
                                            const std::vector<float>&,
                                            const std::vector<float>&,
                                            const std::vector<bool>&,
                                            std::vector<float>&,
                                            std::vector<float>&);

template objective_step_t<double> compute_objective_step_info<int, double>(
  const std::vector<double>&,
  const std::vector<double>&,
  const std::vector<bool>&,
  const std::vector<int>&,
  const std::vector<int>&,
  const std::vector<double>&,
  const std::vector<double>&,
  const std::vector<double>&);

template objective_step_t<float> compute_objective_step_info<int, float>(const std::vector<float>&,
                                                                         const std::vector<float>&,
                                                                         const std::vector<bool>&,
                                                                         const std::vector<int>&,
                                                                         const std::vector<int>&,
                                                                         const std::vector<float>&,
                                                                         const std::vector<float>&,
                                                                         const std::vector<float>&);

}  // namespace cuopt::linear_programming::dual_simplex
