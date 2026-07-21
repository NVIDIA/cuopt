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
#include <vector>

namespace cuopt::mathematical_optimization::mip {

using simplex::objective_step_t;

namespace {

// Compute the lattice step for an unknown variable in constraint i, given the known
// variables' steps. The step is: gcd(|a_k * step_k| for known vars) / |a_unknown|.
// Returns {0, 1} if solve_for cannot be determined (its coefficient is zero, or no
// other variable in the constraint has a known lattice).
template <typename i_t, typename f_t>
rational128_t<f_t> compute_step_for_unknown(i_t i,
                                            i_t solve_for,
                                            const std::vector<i_t>& offsets,
                                            const std::vector<i_t>& variables,
                                            const std::vector<rational128_t<f_t>>& coef_r,
                                            const std::vector<rational128_t<f_t>>& step_r)
{
  rational128_t<f_t> a_unknown = {0, 1};
  rational128_t<f_t> step_sum  = {0, 1};

  for (i_t p = offsets[i]; p < offsets[i + 1]; ++p) {
    i_t j = variables[p];

    if (j == solve_for) {
      a_unknown = coef_r[p];
      continue;
    }
    if (step_r[j].is_zero()) continue;

    step_sum = gcd(step_sum, (coef_r[p] * step_r[j]).abs());
    // Check for overflow in intermediate GCD computation
    if (step_sum.q <= 0) return {0, 1};
  }

  if (a_unknown.is_zero()) return {0, 1};

  rational128_t<f_t> result = step_sum / a_unknown.abs();
  // Check for overflow in the division
  if (result.q <= 0) return {0, 1};
  return result;
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
                       const std::vector<bool>& is_lattice_known_initially,
                       const std::vector<f_t>& obj_coefs,
                       std::vector<f_t>& lattice_step)
{
  constexpr f_t eq_tol = 1e-8;

  // Track lattice step as rationals. step_r[j].p == 0 means unknown.
  std::vector<rational128_t<f_t>> step_r(n_vars);

  for (i_t j = 0; j < n_vars; ++j) {
    if (is_lattice_known_initially[j]) { step_r[j] = {1, 1}; }
  }

  // Rationalize all matrix coefficients once. Track which constraints have at least one
  // non-rationalizable coefficient so we can exclude them from propagation.
  std::vector<rational128_t<f_t>> coef_r(coefficients.size());
  std::vector<bool> constraint_has_bad_coef(n_cons, false);
  for (i_t i = 0; i < n_cons; ++i) {
    for (i_t p = offsets[i]; p < offsets[i + 1]; ++p) {
      coef_r[p] = rational128_t<f_t>::safe_from_floating_point(coefficients[p]);
      if (coef_r[p].is_zero() && coefficients[p] != 0) { constraint_has_bad_coef[i] = true; }
    }
  }

  // Identify equality rows. Lattice propagation through equalities is sound when exactly
  // one unknown remains in the row.
  std::vector<bool> is_equality(n_cons);
  for (i_t i = 0; i < n_cons; ++i) {
    if (constraint_has_bad_coef[i]) continue;
    bool lb_finite = std::isfinite(con_lb[i]);
    bool ub_finite = std::isfinite(con_ub[i]);
    cuopt_assert(lb_finite || ub_finite, "propagate_lattice: free constraints are not supported");

    if (lb_finite && ub_finite && std::abs(con_lb[i] - con_ub[i]) < eq_tol) {
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

  // Convert the CSR representation to CSC for fast column access.
  struct csc_adjacency_t {
    std::vector<i_t> col_start;
    std::vector<i_t> i;
  };
  csc_adjacency_t A_col;
  A_col.col_start.assign(n_vars + 1, 0);
  for (i_t j = 0; j < n_vars; ++j) {
    A_col.col_start[j + 1] = A_col.col_start[j] + constraints_per_variable[j];
  }
  A_col.i.resize(A_col.col_start[n_vars]);
  for (i_t i = 0; i < n_cons; ++i) {
    for (i_t p = offsets[i]; p < offsets[i + 1]; ++p) {
      A_col.i[A_col.col_start[variables[p]]++] = i;
    }
  }
  // Restore col_start
  i_t carry = 0;
  for (i_t j = 0; j <= n_vars; ++j) {
    const i_t next     = A_col.col_start[j];
    A_col.col_start[j] = carry;
    carry              = next;
  }

  // Number of currently-unknown variables in each constraint.
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

  // Iteratively propagate through equality rows using a worklist.
  bool any_discovered = false;

  std::vector<i_t> changed_constraints(n_cons);
  std::vector<i_t> next_changed_constraints(n_cons);
  std::vector<i_t> sorted_constraints(n_cons);
  i_t num_changed = 0;
  for (i_t i = 0; i < n_cons; ++i) {
    if (is_equality[i]) { changed_constraints[num_changed++] = i; }
  }

  std::vector<i_t> discovered_variables;
  std::vector<i_t> in_next_pass(n_cons, 0);
  std::vector<i_t> bucket_offset(max_unknown_count + 1, 0);

  while (num_changed > 0) {
    // Counting-sort by unknown count ascending.
    for (i_t k = 0; k <= max_unknown_count; ++k) {
      bucket_offset[k] = 0;
    }
    for (i_t k = 0; k < num_changed; ++k) {
      bucket_offset[unknown_count_per_constraint[changed_constraints[k]]]++;
    }
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
      if (unknown_count_per_constraint[i] != 1) continue;

      i_t j = -1;
      for (i_t p = offsets[i]; p < offsets[i + 1]; ++p) {
        if (step_r[variables[p]].is_zero()) {
          j = variables[p];
          break;
        }
      }
      if (j < 0) continue;

      rational128_t<f_t> new_step =
        compute_step_for_unknown<i_t, f_t>(i, j, offsets, variables, coef_r, step_r);

      if (!new_step.is_zero() && new_step.q > 0) {
        step_r[j] = new_step.reduced();
        discovered_variables.push_back(j);
        const i_t col_start = A_col.col_start[j];
        const i_t col_end   = A_col.col_start[j + 1];
        for (i_t p = col_start; p < col_end; ++p) {
          unknown_count_per_constraint[A_col.i[p]]--;
        }
        any_discovered = true;
      }
    }

    // Build next worklist from equality rows touched by discovered variables.
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
    for (i_t k = 0; k < num_next; ++k) {
      in_next_pass[next_changed_constraints[k]] = false;
    }

    changed_constraints.swap(next_changed_constraints);
    num_changed = num_next;
  }

  // --- Inequality propagation for single-variable objectives ---
  // If exactly one objective variable remains unknown, use the objective-direction argument:
  // at optimum (minimization), at least one inequality bounding the variable in the
  // improving direction must be tight. We take the GCD of steps from all qualifying
  // inequalities.
  {
    i_t obj_var       = -1;
    f_t obj_coef      = 0;
    i_t n_unknown_obj = 0;
    for (i_t j = 0; j < n_vars; ++j) {
      if (obj_coefs[j] == 0) continue;
      if (step_r[j].is_zero()) {
        obj_var  = j;
        obj_coef = obj_coefs[j];
        n_unknown_obj++;
      }
    }

    if (n_unknown_obj == 1 && obj_var >= 0) {
      rational128_t<f_t> combined_step = {0, 1};
      bool found_any                   = false;

      const i_t col_start = A_col.col_start[obj_var];
      const i_t col_end   = A_col.col_start[obj_var + 1];
      for (i_t cp = col_start; cp < col_end; ++cp) {
        i_t i = A_col.i[cp];

        if (is_equality[i]) continue;
        if (constraint_has_bad_coef[i]) continue;
        if (unknown_count_per_constraint[i] != 1) continue;

        // Find the coefficient of obj_var in this constraint
        f_t obj_var_coef_in_row = 0;
        for (i_t p = offsets[i]; p < offsets[i + 1]; ++p) {
          if (variables[p] == obj_var) {
            obj_var_coef_in_row = coefficients[p];
            break;
          }
        }

        // Check objective direction: minimization pushes obj_var in direction -sign(obj_coef).
        // For LHS <= ub: tightens when obj_var_coef_in_row * obj_coef < 0
        // For LHS >= lb: tightens when obj_var_coef_in_row * obj_coef > 0
        bool lb_finite  = std::isfinite(con_lb[i]);
        bool ub_finite  = std::isfinite(con_ub[i]);
        bool qualifying = false;

        if (ub_finite && (obj_var_coef_in_row * obj_coef < 0)) {
          qualifying = true;
        } else if (lb_finite && (obj_var_coef_in_row * obj_coef > 0)) {
          qualifying = true;
        }

        if (!qualifying) continue;

        rational128_t<f_t> new_step =
          compute_step_for_unknown<i_t, f_t>(i, obj_var, offsets, variables, coef_r, step_r);

        if (new_step.is_zero() || new_step.q <= 0) continue;

        if (!found_any) {
          combined_step = new_step;
          found_any     = true;
        } else {
          combined_step = gcd(combined_step, new_step);
          // If the GCD has become negligibly small, the constraints do not share a
          // meaningful common lattice. Bail out to avoid overflow in rational arithmetic.
          if (combined_step.q < 0 || std::abs(combined_step.to_floating_point()) < 1e-10) {
            found_any = false;
            break;
          }
        }
      }

      if (found_any && !combined_step.is_zero()) {
        step_r[obj_var] = combined_step.reduced();
        any_discovered  = true;
      }
    }
  }

  // Convert back to f_t
  lattice_step.assign(n_vars, f_t(0));
  for (i_t j = 0; j < n_vars; ++j) {
    if (!step_r[j].is_zero()) { lattice_step[j] = step_r[j].to_floating_point(); }
  }

  return any_discovered;
}

template <typename i_t, typename f_t>
objective_step_t<f_t> compute_objective_step_info(
  const std::vector<f_t>& obj_coefs,
  const std::vector<bool>& is_lattice_known_initially,
  const std::vector<i_t>& offsets,
  const std::vector<i_t>& variables,
  const std::vector<f_t>& coefficients,
  const std::vector<f_t>& con_lb,
  const std::vector<f_t>& con_ub)
{
  const i_t n_variables   = static_cast<i_t>(obj_coefs.size());
  const i_t n_constraints = static_cast<i_t>(con_lb.size());

  std::vector<f_t> lattice_step;
  bool discovered = propagate_lattice<i_t, f_t>(n_variables,
                                                n_constraints,
                                                offsets,
                                                variables,
                                                coefficients,
                                                con_lb,
                                                con_ub,
                                                is_lattice_known_initially,
                                                obj_coefs,
                                                lattice_step);

  if (!discovered) return {};

  // Objective step = gcd(|c_j * step_j|) over all objective variables.
  f_t obj_step = 0;
  for (i_t i = 0; i < n_variables; ++i) {
    if (obj_coefs[i] == 0) continue;
    if (lattice_step[i] == 0) return {};  // Still unknown -- give up.
    obj_step = gcd_floating_point<f_t>(obj_step, std::abs(obj_coefs[i] * lattice_step[i]));
  }

  if (obj_step > 1e-12) {
    objective_step_t<f_t> result;
    result.step_size = obj_step;
    return result;
  }
  return {};
}

// Explicit instantiations
template bool propagate_lattice<int, double>(int,
                                             int,
                                             const std::vector<int>&,
                                             const std::vector<int>&,
                                             const std::vector<double>&,
                                             const std::vector<double>&,
                                             const std::vector<double>&,
                                             const std::vector<bool>&,
                                             const std::vector<double>&,
                                             std::vector<double>&);

template bool propagate_lattice<int, float>(int,
                                            int,
                                            const std::vector<int>&,
                                            const std::vector<int>&,
                                            const std::vector<float>&,
                                            const std::vector<float>&,
                                            const std::vector<float>&,
                                            const std::vector<bool>&,
                                            const std::vector<float>&,
                                            std::vector<float>&);

template objective_step_t<double> compute_objective_step_info<int, double>(
  const std::vector<double>&,
  const std::vector<bool>&,
  const std::vector<int>&,
  const std::vector<int>&,
  const std::vector<double>&,
  const std::vector<double>&,
  const std::vector<double>&);

template objective_step_t<float> compute_objective_step_info<int, float>(const std::vector<float>&,
                                                                         const std::vector<bool>&,
                                                                         const std::vector<int>&,
                                                                         const std::vector<int>&,
                                                                         const std::vector<float>&,
                                                                         const std::vector<float>&,
                                                                         const std::vector<float>&);

}  // namespace cuopt::mathematical_optimization::mip
