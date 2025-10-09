/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <dual_simplex/presolve.hpp>

#include <dual_simplex/folding.hpp>
#include <dual_simplex/right_looking_lu.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/tic_toc.hpp>

#include <cmath>
#include <iostream>
#include <numeric>

namespace cuopt::linear_programming::dual_simplex {

template <typename f_t>
static inline f_t update_lb(f_t curr_lb, f_t coeff, f_t delta_min_act, f_t delta_max_act)
{
  auto comp_bnd = (coeff < 0.) ? delta_min_act / coeff : delta_max_act / coeff;
  return std::max(curr_lb, comp_bnd);
}

template <typename f_t>
static inline f_t update_ub(f_t curr_ub, f_t coeff, f_t delta_min_act, f_t delta_max_act)
{
  auto comp_bnd = (coeff < 0.) ? delta_max_act / coeff : delta_min_act / coeff;
  return std::min(curr_ub, comp_bnd);
}

template <typename i_t, typename f_t>
static inline bool check_infeasibility(f_t min_a, f_t max_a, f_t cnst_lb, f_t cnst_ub, f_t eps)
{
  return (min_a > cnst_ub + eps) || (max_a < cnst_lb - eps);
}

#define DEBUG_BOUND_STRENGTHENING 0

template <typename i_t, typename f_t>
void print_bounds_stats(const std::vector<f_t>& lower,
                        const std::vector<f_t>& upper,
                        const simplex_solver_settings_t<i_t, f_t>& settings,
                        const std::string msg)
{
#if DEBUG_BOUND_STRENGTHENING
  f_t lb_norm = 0.0;
  f_t ub_norm = 0.0;

  i_t sz = lower.size();
  for (i_t i = 0; i < sz; ++i) {
    if (std::isfinite(lower[i])) { lb_norm += abs(lower[i]); }
    if (std::isfinite(upper[i])) { ub_norm += abs(upper[i]); }
  }
  settings.log.printf("%s :: lb norm %e, ub norm %e\n", msg.c_str(), lb_norm, ub_norm);
#endif
}

template <typename i_t, typename f_t>
bool bound_strengthening(const std::vector<char>& row_sense,
                         const simplex_solver_settings_t<i_t, f_t>& settings,
                         lp_problem_t<i_t, f_t>& problem,
                         const csc_matrix_t<i_t, f_t>& Arow,
                         const std::vector<variable_type_t>& var_types,
                         const std::vector<bool>& bounds_changed)
{
  const i_t m = problem.num_rows;
  const i_t n = problem.num_cols;

  std::vector<f_t> delta_min_activity(m);
  std::vector<f_t> delta_max_activity(m);
  std::vector<f_t> constraint_lb(m);
  std::vector<f_t> constraint_ub(m);

  // FIXME:: Instead of initializing constraint_changed to true, we can only look
  // at the constraints corresponding to branched variable in branch and bound
  // This is because, the parent LP already checked for feasibility of the constraints
  // without the branched variable bounds
  std::vector<bool> constraint_changed(m, true);
  std::vector<bool> variable_changed(n, false);
  std::vector<bool> constraint_changed_next(m, false);

  if (false && !bounds_changed.empty()) {
    std::fill(constraint_changed.begin(), constraint_changed.end(), false);
    for (i_t i = 0; i < n; ++i) {
      if (bounds_changed[i]) {
        const i_t row_start = problem.A.col_start[i];
        const i_t row_end   = problem.A.col_start[i + 1];
        for (i_t p = row_start; p < row_end; ++p) {
          const i_t j           = problem.A.i[p];
          constraint_changed[j] = true;
        }
      }
    }
  }

  const bool is_row_sense_empty = row_sense.empty();
  if (is_row_sense_empty) {
    std::copy(problem.rhs.begin(), problem.rhs.end(), constraint_lb.begin());
    std::copy(problem.rhs.begin(), problem.rhs.end(), constraint_ub.begin());
  } else {
    //  Set the constraint bounds
    for (i_t i = 0; i < m; ++i) {
      if (row_sense[i] == 'E') {
        constraint_lb[i] = problem.rhs[i];
        constraint_ub[i] = problem.rhs[i];
      } else if (row_sense[i] == 'L') {
        constraint_ub[i] = problem.rhs[i];
        constraint_lb[i] = -inf;
      } else {
        constraint_lb[i] = problem.rhs[i];
        constraint_ub[i] = inf;
      }
    }
  }

  std::vector<f_t> lower = problem.lower;
  std::vector<f_t> upper = problem.upper;
  print_bounds_stats(lower, upper, settings, "Initial bounds");

  i_t iter             = 0;
  const i_t iter_limit = 10;
  while (iter < iter_limit) {
    for (i_t i = 0; i < m; ++i) {
      if (!constraint_changed[i]) { continue; }
      const i_t row_start = Arow.col_start[i];
      const i_t row_end   = Arow.col_start[i + 1];

      f_t min_a = 0.0;
      f_t max_a = 0.0;
      for (i_t p = row_start; p < row_end; ++p) {
        const i_t j    = Arow.i[p];
        const f_t a_ij = Arow.x[p];

        variable_changed[j] = true;
        if (a_ij > 0) {
          min_a += a_ij * lower[j];
          max_a += a_ij * upper[j];
        } else if (a_ij < 0) {
          min_a += a_ij * upper[j];
          max_a += a_ij * lower[j];
        }
        if (upper[j] == inf && a_ij > 0) { max_a = inf; }
        if (lower[j] == -inf && a_ij < 0) { max_a = inf; }

        if (lower[j] == -inf && a_ij > 0) { min_a = -inf; }
        if (upper[j] == inf && a_ij < 0) { min_a = -inf; }
      }

      f_t cnst_lb = constraint_lb[i];
      f_t cnst_ub = constraint_ub[i];
      bool is_infeasible =
        check_infeasibility<i_t, f_t>(min_a, max_a, cnst_lb, cnst_ub, settings.primal_tol);
      if (is_infeasible) {
        settings.log.printf(
          "Iter:: %d, Infeasible constraint %d, cnst_lb %e, cnst_ub %e, min_a %e, max_a %e\n",
          iter,
          i,
          cnst_lb,
          cnst_ub,
          min_a,
          max_a);
        return false;
      }

      delta_min_activity[i] = cnst_ub - min_a;
      delta_max_activity[i] = cnst_lb - max_a;
    }

    i_t num_bounds_changed = 0;

    for (i_t k = 0; k < n; ++k) {
      if (!variable_changed[k]) { continue; }
      f_t old_lb = lower[k];
      f_t old_ub = upper[k];

      f_t new_lb = old_lb;
      f_t new_ub = old_ub;

      const i_t row_start = problem.A.col_start[k];
      const i_t row_end   = problem.A.col_start[k + 1];
      for (i_t p = row_start; p < row_end; ++p) {
        const i_t i = problem.A.i[p];

        if (!constraint_changed[i]) { continue; }
        const f_t a_ik = problem.A.x[p];

        f_t delta_min_act = delta_min_activity[i];
        f_t delta_max_act = delta_max_activity[i];

        delta_min_act += (a_ik < 0) ? a_ik * old_ub : a_ik * old_lb;
        delta_max_act += (a_ik > 0) ? a_ik * old_ub : a_ik * old_lb;

        new_lb = std::max(new_lb, update_lb(old_lb, a_ik, delta_min_act, delta_max_act));
        new_ub = std::min(new_ub, update_ub(old_ub, a_ik, delta_min_act, delta_max_act));
      }

      // Integer rounding
      if (!var_types.empty() &&
          (var_types[k] == variable_type_t::INTEGER || var_types[k] == variable_type_t::BINARY)) {
        new_lb = std::ceil(new_lb - settings.integer_tol);
        new_ub = std::floor(new_ub + settings.integer_tol);
      }

      bool lb_updated = abs(new_lb - old_lb) > 1e3 * settings.primal_tol;
      bool ub_updated = abs(new_ub - old_ub) > 1e3 * settings.primal_tol;

      new_lb = std::max(new_lb, problem.lower[k]);
      new_ub = std::min(new_ub, problem.upper[k]);

      if (new_lb > new_ub + 1e-6) {
        settings.log.printf(
          "Iter:: %d, Infeasible variable after update %d, %e > %e\n", iter, k, new_lb, new_ub);
        return false;
      }
      if (new_lb != old_lb || new_ub != old_ub) {
        for (i_t p = row_start; p < row_end; ++p) {
          const i_t i                = problem.A.i[p];
          constraint_changed_next[i] = true;
        }
      }

      lower[k] = std::min(new_lb, new_ub);
      upper[k] = std::max(new_lb, new_ub);

      bool bounds_changed = lb_updated || ub_updated;
      if (bounds_changed) { num_bounds_changed++; }
    }

    if (num_bounds_changed == 0) { break; }

    std::swap(constraint_changed, constraint_changed_next);
    std::fill(constraint_changed_next.begin(), constraint_changed_next.end(), false);
    std::fill(variable_changed.begin(), variable_changed.end(), false);

    iter++;
  }

  // settings.log.printf("Total strengthened variables %d\n", total_strengthened_variables);

#if DEBUG_BOUND_STRENGTHENING
  f_t lb_change      = 0.0;
  f_t ub_change      = 0.0;
  int num_lb_changed = 0;
  int num_ub_changed = 0;

  for (i_t i = 0; i < n; ++i) {
    if (lower[i] > problem.lower[i] + settings.primal_tol ||
        (!std::isfinite(problem.lower[i]) && std::isfinite(lower[i]))) {
      num_lb_changed++;
      lb_change +=
        std::isfinite(problem.lower[i])
          ? (lower[i] - problem.lower[i]) / (1e-6 + std::max(abs(lower[i]), abs(problem.lower[i])))
          : 1.0;
    }
    if (upper[i] < problem.upper[i] - settings.primal_tol ||
        (!std::isfinite(problem.upper[i]) && std::isfinite(upper[i]))) {
      num_ub_changed++;
      ub_change +=
        std::isfinite(problem.upper[i])
          ? (problem.upper[i] - upper[i]) / (1e-6 + std::max(abs(problem.upper[i]), abs(upper[i])))
          : 1.0;
    }
  }

  if (num_lb_changed > 0 || num_ub_changed > 0) {
    settings.log.printf(
      "lb change %e, ub change %e, num lb changed %d, num ub changed %d, iter %d\n",
      100 * lb_change / std::max(1, num_lb_changed),
      100 * ub_change / std::max(1, num_ub_changed),
      num_lb_changed,
      num_ub_changed,
      iter);
  }
  print_bounds_stats(lower, upper, settings, "Final bounds");
#endif

  problem.lower = lower;
  problem.upper = upper;

  return true;
}

template <typename i_t, typename f_t>
i_t remove_empty_cols(lp_problem_t<i_t, f_t>& problem,
                      i_t& num_empty_cols,
                      presolve_info_t<i_t, f_t>& presolve_info)
{
  constexpr bool verbose = false;
  if (verbose) { printf("Removing %d empty columns\n", num_empty_cols); }
  // We have a variable x_j that does not appear in any rows
  // The cost function
  // sum_{k != j} c_k * x_k + c_j * x_j
  // becomes
  // sum_{k != j} c_k * x_k + c_j * l_j if c_j > 0
  // or
  // sum_{k != j} c_k * x_k + c_j * u_j if c_j < 0
  presolve_info.removed_variables.reserve(num_empty_cols);
  presolve_info.removed_values.reserve(num_empty_cols);
  presolve_info.removed_reduced_costs.reserve(num_empty_cols);
  std::vector<i_t> col_marker(problem.num_cols);
  i_t new_cols = 0;
  for (i_t j = 0; j < problem.num_cols; ++j) {
    bool remove_var = false;
    if ((problem.A.col_start[j + 1] - problem.A.col_start[j]) == 0) {
      if (problem.objective[j] >= 0 && problem.lower[j] > -inf) {
        presolve_info.removed_values.push_back(problem.lower[j]);
        problem.obj_constant += problem.objective[j] * problem.lower[j];
        remove_var = true;
      } else if (problem.objective[j] <= 0 && problem.upper[j] < inf) {
        presolve_info.removed_values.push_back(problem.upper[j]);
        problem.obj_constant += problem.objective[j] * problem.upper[j];
        remove_var = true;
      }
    }

    if (remove_var) {
      col_marker[j] = 1;
      presolve_info.removed_variables.push_back(j);
      presolve_info.removed_reduced_costs.push_back(problem.objective[j]);
    } else {
      col_marker[j] = 0;
      new_cols++;
    }
  }
  presolve_info.remaining_variables.reserve(new_cols);

  problem.A.remove_columns(col_marker);
  // Clean up objective, lower, upper, and col_names
  assert(new_cols == problem.A.n);
  std::vector<f_t> objective(new_cols);
  std::vector<f_t> lower(new_cols, -INFINITY);
  std::vector<f_t> upper(new_cols, INFINITY);

  int new_j = 0;
  for (i_t j = 0; j < problem.num_cols; ++j) {
    if (!col_marker[j]) {
      objective[new_j] = problem.objective[j];
      lower[new_j]     = problem.lower[j];
      upper[new_j]     = problem.upper[j];
      presolve_info.remaining_variables.push_back(j);
      new_j++;
    } else {
      num_empty_cols--;
    }
  }
  problem.objective = objective;
  problem.lower     = lower;
  problem.upper     = upper;
  problem.num_cols  = new_cols;
  return 0;
}

template <typename i_t, typename f_t>
i_t remove_rows(lp_problem_t<i_t, f_t>& problem,
                const std::vector<char>& row_sense,
                csr_matrix_t<i_t, f_t>& Arow,
                std::vector<i_t>& row_marker,
                bool error_on_nonzero_rhs)
{
  constexpr bool verbose = true;
  if (verbose) { printf("Removing rows %d %ld\n", Arow.m, row_marker.size()); }
  csr_matrix_t<i_t, f_t> Aout(0, 0, 0);
  Arow.remove_rows(row_marker, Aout);
  i_t new_rows = Aout.m;
  if (verbose) { printf("Cleaning up rhs. New rows %d\n", new_rows); }
  std::vector<char> new_row_sense(new_rows);
  std::vector<f_t> new_rhs(new_rows);
  i_t row_count = 0;
  for (i_t i = 0; i < problem.num_rows; ++i) {
    if (!row_marker[i]) {
      new_row_sense[row_count] = row_sense[i];
      new_rhs[row_count]       = problem.rhs[i];
      row_count++;
    } else {
      if (error_on_nonzero_rhs && problem.rhs[i] != 0.0) {
        if (verbose) {
          printf(
            "Error nonzero rhs %e for zero row %d sense %c\n", problem.rhs[i], i, row_sense[i]);
        }
        return i + 1;
      }
    }
  }
  problem.rhs = new_rhs;
  Aout.to_compressed_col(problem.A);
  assert(problem.A.m == new_rows);
  problem.num_rows = problem.A.m;
  return 0;
}

template <typename i_t, typename f_t>
i_t remove_empty_rows(lp_problem_t<i_t, f_t>& problem,
                      std::vector<char>& row_sense,
                      i_t& num_empty_rows,
                      presolve_info_t<i_t, f_t>& presolve_info)
{
  constexpr bool verbose = false;
  if (verbose) { printf("Problem has %d empty rows\n", num_empty_rows); }
  csr_matrix_t<i_t, f_t> Arow(0, 0, 0);
  problem.A.to_compressed_row(Arow);
  std::vector<i_t> row_marker(problem.num_rows);
  presolve_info.removed_constraints.reserve(num_empty_rows);
  presolve_info.remaining_constraints.reserve(problem.num_rows - num_empty_rows);
  for (i_t i = 0; i < problem.num_rows; ++i) {
    if ((Arow.row_start[i + 1] - Arow.row_start[i]) == 0) {
      row_marker[i] = 1;
      presolve_info.removed_constraints.push_back(i);
      if (verbose) {
        printf("Empty row %d start %d end %d\n", i, Arow.row_start[i], Arow.row_start[i + 1]);
      }
    } else {
      presolve_info.remaining_constraints.push_back(i);
      row_marker[i] = 0;
    }
  }
  const i_t retval = remove_rows(problem, row_sense, Arow, row_marker, true);
  return retval;
}

template <typename i_t, typename f_t>
i_t remove_fixed_variables(f_t fixed_tolerance,
                           lp_problem_t<i_t, f_t>& problem,
                           i_t& fixed_variables)
{
  constexpr bool verbose = false;
  if (verbose) { printf("Removing %d fixed variables\n", fixed_variables); }
  // We have a variable with l_j = x_j = u_j
  // Constraints of the form
  //
  // sum_{k != j} a_ik * x_k + a_ij * x_j {=, <=} beta
  // become
  // sum_{k != j} a_ik * x_k {=, <=} beta - a_ij * l_j
  //
  // The cost function
  // sum_{k != j} c_k * x_k + c_j * x_j
  // becomes
  // sum_{k != j} c_k * x_k + c_j l_j

  std::vector<i_t> col_marker(problem.num_cols);
  for (i_t j = 0; j < problem.num_cols; ++j) {
    if (std::abs(problem.upper[j] - problem.lower[j]) < fixed_tolerance) {
      col_marker[j] = 1;
      for (i_t p = problem.A.col_start[j]; p < problem.A.col_start[j + 1]; ++p) {
        const i_t i   = problem.A.i[p];
        const f_t aij = problem.A.x[p];
        problem.rhs[i] -= aij * problem.lower[j];
      }
      problem.obj_constant += problem.objective[j] * problem.lower[j];
    } else {
      col_marker[j] = 0;
    }
  }

  problem.A.remove_columns(col_marker);

  // Clean up objective, lower, upper, and col_names
  i_t new_cols = problem.A.n;
  if (verbose) { printf("new cols %d\n", new_cols); }
  std::vector<f_t> objective(new_cols);
  std::vector<f_t> lower(new_cols);
  std::vector<f_t> upper(new_cols);
  i_t new_j = 0;
  for (i_t j = 0; j < problem.num_cols; ++j) {
    if (!col_marker[j]) {
      objective[new_j] = problem.objective[j];
      lower[new_j]     = problem.lower[j];
      upper[new_j]     = problem.upper[j];
      new_j++;
      fixed_variables--;
    }
  }
  problem.objective = objective;
  problem.lower     = lower;
  problem.upper     = upper;
  problem.num_cols  = problem.A.n;
  if (verbose) { printf("Finishing fixed columns\n"); }
  return 0;
}

template <typename i_t, typename f_t>
i_t convert_less_than_to_equal(const user_problem_t<i_t, f_t>& user_problem,
                               std::vector<char>& row_sense,
                               lp_problem_t<i_t, f_t>& problem,
                               i_t& less_rows,
                               std::vector<i_t>& new_slacks)
{
  constexpr bool verbose = false;
  if (verbose) { printf("Converting %d less than inequalities to equalities\n", less_rows); }
  // We must convert rows in the form: a_i^T x <= beta
  // into: a_i^T x + s_i = beta, s_i >= 0

  csr_matrix_t<i_t, f_t> Arow(0, 0, 0);
  problem.A.to_compressed_row(Arow);
  i_t num_cols = problem.num_cols + less_rows;
  i_t nnz      = problem.A.col_start[problem.num_cols] + less_rows;
  problem.A.col_start.resize(num_cols + 1);
  problem.A.i.resize(nnz);
  problem.A.x.resize(nnz);
  problem.lower.resize(num_cols);
  problem.upper.resize(num_cols);
  problem.objective.resize(num_cols);

  i_t p = problem.A.col_start[problem.num_cols];
  i_t j = problem.num_cols;
  for (i_t i = 0; i < problem.num_rows; i++) {
    if (row_sense[i] == 'L') {
      problem.lower[j]     = 0.0;
      problem.upper[j]     = INFINITY;
      problem.objective[j] = 0.0;
      problem.A.i[p]       = i;
      problem.A.x[p]       = 1.0;
      new_slacks.push_back(j);
      problem.A.col_start[j++] = p++;
      row_sense[i]             = 'E';
      less_rows--;
    }
  }
  problem.A.col_start[num_cols] = p;
  assert(less_rows == 0);
  assert(p == nnz);
  problem.A.n      = num_cols;
  problem.num_cols = num_cols;

  return 0;
}

template <typename i_t, typename f_t>
i_t convert_greater_to_less(const user_problem_t<i_t, f_t>& user_problem,
                            std::vector<char>& row_sense,
                            lp_problem_t<i_t, f_t>& problem,
                            i_t& greater_rows,
                            i_t& less_rows)
{
  constexpr bool verbose = false;
  if (verbose) {
    printf("Transforming %d greater than constraints into less than constraints\n", greater_rows);
  }
  // We have a constraint in the form
  // sum_{j : a_ij != 0} a_ij * x_j >= beta
  // We transform this into the constraint
  // sum_{j : a_ij != 0} -a_ij * x_j <= -beta

  // First construct a compressed sparse row representation of the A matrix
  csr_matrix_t<i_t, f_t> Arow(0, 0, 0);
  problem.A.to_compressed_row(Arow);

  for (i_t i = 0; i < problem.num_rows; i++) {
    if (row_sense[i] == 'G') {
      i_t row_start = Arow.row_start[i];
      i_t row_end   = Arow.row_start[i + 1];
      for (i_t p = Arow.row_start[i]; p < row_end; p++) {
        Arow.x[p] *= -1;
      }
      problem.rhs[i] *= -1;
      row_sense[i] = 'L';
      greater_rows--;
      less_rows++;
    }
  }

  // Now convert the compressed sparse row representation back to compressed
  // sparse column
  Arow.to_compressed_col(problem.A);

  return 0;
}

template <typename i_t, typename f_t>
i_t convert_range_rows(const user_problem_t<i_t, f_t>& user_problem,
                       std::vector<char>& row_sense,
                       lp_problem_t<i_t, f_t>& problem,
                       i_t& less_rows,
                       i_t& equal_rows,
                       i_t& greater_rows,
                       std::vector<i_t>& new_slacks)
{
  // A range row has the format h_i <= a_i^T x <= u_i
  // We must convert this into the constraint
  // a_i^T x - s_i = 0
  // h_i <= s_i <= u_i
  // by adding a new slack variable s_i
  //
  // The values of h_i and u_i are determined by the b_i (RHS) and r_i (RANGES)
  // associated with the ith constraint as well as the row sense
  i_t num_cols       = problem.num_cols + user_problem.num_range_rows;
  i_t num_range_rows = user_problem.num_range_rows;
  i_t nnz            = problem.A.col_start[problem.num_cols] + num_range_rows;
  problem.A.col_start.resize(num_cols + 1);
  problem.A.i.resize(nnz);
  problem.A.x.resize(nnz);
  problem.lower.resize(num_cols);
  problem.upper.resize(num_cols);
  problem.objective.resize(num_cols);

  i_t p = problem.A.col_start[problem.num_cols];
  i_t j = problem.num_cols;
  for (i_t k = 0; k < num_range_rows; k++) {
    const i_t i = user_problem.range_rows[k];
    const f_t r = user_problem.range_value[k];
    const f_t b = problem.rhs[i];
    f_t h;
    f_t u;
    if (row_sense[i] == 'L') {
      h = b - std::abs(r);
      u = b;
      less_rows--;
      equal_rows++;
    } else if (row_sense[i] == 'G') {
      h = b;
      u = b + std::abs(r);
      greater_rows--;
      equal_rows++;
    } else if (row_sense[i] == 'E') {
      if (r > 0) {
        h = b;
        u = b + std::abs(r);
      } else {
        h = b - std::abs(r);
        u = b;
      }
    }
    problem.lower[j]     = h;
    problem.upper[j]     = u;
    problem.objective[j] = 0.0;
    problem.A.i[p]       = i;
    problem.A.x[p]       = -1.0;
    new_slacks.push_back(j);
    problem.A.col_start[j++] = p++;
    problem.rhs[i]           = 0.0;
    row_sense[i]             = 'E';
  }
  problem.A.col_start[num_cols] = p;
  assert(p == nnz);
  problem.A.n      = num_cols;
  problem.num_cols = num_cols;

  return 0;
}

template <typename i_t, typename f_t>
i_t find_dependent_rows(lp_problem_t<i_t, f_t>& problem,
                        const simplex_solver_settings_t<i_t, f_t>& settings,
                        std::vector<i_t>& dependent_rows,
                        i_t& infeasible)
{
  i_t m  = problem.num_rows;
  i_t n  = problem.num_cols;
  i_t nz = problem.A.col_start[n];
  assert(m == problem.A.m);
  assert(n == problem.A.n);
  dependent_rows.resize(m);

  infeasible = -1;

  // Form C = A'
  csc_matrix_t<i_t, f_t> C(n, m, 1);
  problem.A.transpose(C);
  assert(C.col_start[m] == nz);

  // Calculate L*U = C(p, :)
  csc_matrix_t<i_t, f_t> L(n, m, nz);
  csc_matrix_t<i_t, f_t> U(m, m, nz);
  std::vector<i_t> pinv(n);
  std::vector<i_t> q(m);

  i_t pivots = right_looking_lu_row_permutation_only(C, settings, 1e-13, tic(), q, pinv);

  if (pivots < m) {
    settings.log.printf("Found %d dependent rows\n", m - pivots);
    const i_t num_dependent = m - pivots;
    std::vector<f_t> independent_rhs(pivots);
    std::vector<f_t> dependent_rhs(num_dependent);
    std::vector<i_t> dependent_row_list(num_dependent);
    i_t ind_count = 0;
    i_t dep_count = 0;
    for (i_t i = 0; i < m; ++i) {
      i_t row = q[i];
      if (i < pivots) {
        dependent_rows[row]          = 0;
        independent_rhs[ind_count++] = problem.rhs[row];
      } else {
        dependent_rows[row]             = 1;
        dependent_rhs[dep_count]        = problem.rhs[row];
        dependent_row_list[dep_count++] = row;
      }
    }

#if 0
    std::vector<f_t> z = independent_rhs;
    // Solve U1^T z = independent_rhs
    for (i_t k = 0; k < pivots; ++k) {
      const i_t col_start = U.col_start[k];
      const i_t col_end   = U.col_start[k + 1];
      for (i_t p = col_start; p < col_end; ++p) {
        z[k] -= U.x[p] * z[U.i[p]];
      }
      z[k] /= U.x[col_end];
    }

    // Compute compare_dependent = U2^T z
    std::vector<f_t> compare_dependent(num_dependent);
    for (i_t k = pivots; k < m; ++k) {
      f_t dot             = 0.0;
      const i_t col_start = U.col_start[k];
      const i_t col_end   = U.col_start[k + 1];
      for (i_t p = col_start; p < col_end; ++p) {
        dot += z[U.i[p]] * U.x[p];
      }
      compare_dependent[k - pivots] = dot;
    }

    for (i_t k = 0; k < m - pivots; ++k) {
      if (std::abs(compare_dependent[k] - dependent_rhs[k]) > 1e-6) {
        infeasible = dependent_row_list[k];
        break;
      } else {
        problem.rhs[dependent_row_list[k]] = 0.0;
      }
    }
#endif
  } else {
    settings.log.printf("No dependent rows found\n");
  }
  return pivots;
}

template <typename i_t, typename f_t>
i_t add_artifical_variables(lp_problem_t<i_t, f_t>& problem,
                            std::vector<i_t>& equality_rows,
                            std::vector<i_t>& new_slacks)
{
  const i_t n        = problem.num_cols;
  const i_t m        = problem.num_rows;
  const i_t num_cols = n + equality_rows.size();
  const i_t nnz      = problem.A.col_start[n] + equality_rows.size();
  problem.A.col_start.resize(num_cols + 1);
  problem.A.i.resize(nnz);
  problem.A.x.resize(nnz);
  problem.lower.resize(num_cols);
  problem.upper.resize(num_cols);
  problem.objective.resize(num_cols);

  i_t p = problem.A.col_start[n];
  i_t j = n;
  for (i_t i : equality_rows) {
    // Add an artifical variable z to the equation a_i^T x == b
    // This now becomes a_i^T x + z == b,   0 <= z =< 0
    problem.A.col_start[j] = p;
    problem.A.i[p]         = i;
    problem.A.x[p]         = 1.0;
    problem.lower[j]       = 0.0;
    problem.upper[j]       = 0.0;
    problem.objective[j]   = 0.0;
    new_slacks.push_back(j);
    p++;
    j++;
  }
  problem.A.col_start[num_cols] = p;
  assert(j == num_cols);
  assert(p == nnz);
  constexpr bool verbose = false;
  if (verbose) { printf("Added %d artificial variables\n", num_cols - n); }
  problem.A.n      = num_cols;
  problem.num_cols = num_cols;
  return 0;
}

template <typename i_t, typename f_t>
void convert_user_problem(const user_problem_t<i_t, f_t>& user_problem,
                          const simplex_solver_settings_t<i_t, f_t>& settings,
                          lp_problem_t<i_t, f_t>& problem,
                          std::vector<i_t>& new_slacks,
                          dualize_info_t<i_t, f_t>& dualize_info)
{
  constexpr bool verbose = false;
  if (verbose) {
    printf("Converting problem with %d rows and %d columns and %d nonzeros\n",
           user_problem.num_rows,
           user_problem.num_cols,
           user_problem.A.col_start[user_problem.num_cols]);
  }

  // Copy info from user_problem to problem
  problem.num_rows     = user_problem.num_rows;
  problem.num_cols     = user_problem.num_cols;
  problem.A            = user_problem.A;
  problem.objective    = user_problem.objective;
  problem.obj_scale    = user_problem.obj_scale;
  problem.obj_constant = user_problem.obj_constant;
  problem.rhs          = user_problem.rhs;
  problem.lower        = user_problem.lower;
  problem.upper        = user_problem.upper;

  // Make a copy of row_sense so we can modify it
  std::vector<char> row_sense = user_problem.row_sense;

  // The original problem can have constraints in the form
  // a_i^T x >= b, a_i^T x <= b, and a_i^T x == b
  //
  // we first restrict these to just
  // a_i^T x <= b and a_i^T x == b
  //
  // We do this by working with the A matrix in csr format
  // and negating coefficents in rows with >= or 'G' row sense
  i_t greater_rows = 0;
  i_t less_rows    = 0;
  i_t equal_rows   = 0;
  std::vector<i_t> equality_rows;
  for (i_t i = 0; i < user_problem.num_rows; ++i) {
    if (row_sense[i] == 'G') {
      greater_rows++;
    } else if (row_sense[i] == 'L') {
      less_rows++;
    } else {
      equal_rows++;
      equality_rows.push_back(i);
    }
  }
  if (verbose) { printf("Constraints < %d = %d > %d\n", less_rows, equal_rows, greater_rows); }

  if (user_problem.num_range_rows > 0) {
    if (verbose) { printf("Problem has %d range rows\n", user_problem.num_range_rows); }
    convert_range_rows(
      user_problem, row_sense, problem, less_rows, equal_rows, greater_rows, new_slacks);
  }

  if (greater_rows > 0) {
    convert_greater_to_less(user_problem, row_sense, problem, greater_rows, less_rows);
  }

  // At this point the problem representation is in the form: A*x {<=, =} b
  // This is the time to run bound strengthening
  constexpr bool run_bound_strengthening = false;
  if constexpr (run_bound_strengthening) {
    settings.log.printf("Running bound strengthening\n");
    csc_matrix_t<i_t, f_t> Arow(1, 1, 1);
    problem.A.transpose(Arow);
    bound_strengthening(row_sense, settings, problem, Arow);
  }
  settings.log.debug(
    "equality rows %d less rows %d columns %d\n", equal_rows, less_rows, problem.num_cols);
  if (settings.barrier && settings.dualize != 0 &&
      (settings.dualize == 1 ||
       (settings.dualize == -1 && less_rows > 1.2 * problem.num_cols && equal_rows < 2e4))) {
    settings.log.debug("Dualizing in presolve\n");

    i_t num_upper_bounds = 0;
    std::vector<i_t> vars_with_upper_bounds;
    vars_with_upper_bounds.reserve(problem.num_cols);
    bool can_dualize = true;
    for (i_t j = 0; j < problem.num_cols; j++) {
      if (problem.lower[j] != 0.0) {
        settings.log.debug("Variable %d has a nonzero lower bound %e\n", j, problem.lower[j]);
        can_dualize = false;
        break;
      }
      if (problem.upper[j] < inf) {
        num_upper_bounds++;
        vars_with_upper_bounds.push_back(j);
      }
    }

    i_t max_column_nz = 0;
    for (i_t j = 0; j < problem.num_cols; j++) {
      const i_t col_nz = problem.A.col_start[j + 1] - problem.A.col_start[j];
      max_column_nz    = std::max(col_nz, max_column_nz);
    }

    std::vector<i_t> row_degree(problem.num_rows, 0);
    for (i_t j = 0; j < problem.num_cols; j++) {
      const i_t col_start = problem.A.col_start[j];
      const i_t col_end   = problem.A.col_start[j + 1];
      for (i_t p = col_start; p < col_end; p++) {
        row_degree[problem.A.i[p]]++;
      }
    }

    i_t max_row_nz = 0;
    for (i_t i = 0; i < problem.num_rows; i++) {
      max_row_nz = std::max(row_degree[i], max_row_nz);
    }
    settings.log.debug("max row nz %d max col nz %d\n", max_row_nz, max_column_nz);

    if (settings.dualize == -1 && max_row_nz > 1e4 && max_column_nz < max_row_nz) {
      can_dualize = false;
    }

    if (can_dualize) {
      // The problem is in the form
      // minimize   c^T x
      // subject to A_in * x <= b_in        : y_in
      //            A_eq * x == b_eq        : y_eq
      //            0 <= x                  : z_l
      //            x_j <= u_j, for j in U  : z_u
      //
      // The dual is of the form
      // maximize    -b_in^T y_in - b_eq^T y_eq + 0^T z_l - u^T z_u
      // subject to  -A_in^T y_in - A_eq^T y_eq + z_l - z_u = c
      //             y_in >= 0
      //             y_eq free
      //             z_l >= 0
      //             z_u >= 0
      //
      // Since the solvers expect the problem to be in minimization form,
      // we convert this to
      //
      // minimize    b_in^T y_in + b_eq^T y_eq - 0^T z_l + u^T z_u
      // subject to  -A_in^T y_in - A_eq^T y_eq + z_l - z_u = c  : x
      //             y_in >= 0 : x_in
      //             y_eq free
      //             z_l >= 0 : x_l
      //             z_u >= 0 : x_u
      //
      // The dual of this problem is of the form
      //
      // maximize    -c^T x
      // subject to   A_in * x + x_in = b_in   <=> A_in * x <= b_in
      //              A_eq * x = b_eq
      //              x + x_u = u              <=> x <= u
      //              x = x_l                  <=> x >= 0
      //              x free, x_in >= 0, x_l >- 0, x_u >= 0
      i_t dual_rows = problem.num_cols;
      i_t dual_cols = problem.num_rows + problem.num_cols + num_upper_bounds;
      lp_problem_t<i_t, f_t> dual_problem(problem.handle_ptr, 1, 1, 0);
      csc_matrix_t<i_t, f_t> dual_constraint_matrix(1, 1, 0);
      problem.A.transpose(dual_constraint_matrix);
      // dual_constraint_matrix <- [-A^T I I]
      dual_constraint_matrix.m = dual_rows;
      dual_constraint_matrix.n = dual_cols;
      i_t nnz                  = dual_constraint_matrix.col_start[problem.num_rows];
      i_t new_nnz              = nnz + problem.num_cols + num_upper_bounds;
      dual_constraint_matrix.col_start.resize(dual_cols + 1);
      dual_constraint_matrix.i.resize(new_nnz);
      dual_constraint_matrix.x.resize(new_nnz);
      for (i_t p = 0; p < nnz; p++) {
        dual_constraint_matrix.x[p] *= -1.0;
      }
      i_t i = 0;
      for (i_t j = problem.num_rows; j < problem.num_rows + problem.num_cols; j++) {
        dual_constraint_matrix.col_start[j] = nnz;
        dual_constraint_matrix.i[nnz]       = i++;
        dual_constraint_matrix.x[nnz]       = 1.0;
        nnz++;
      }
      for (i_t k = 0; k < num_upper_bounds; k++) {
        i_t p                               = problem.num_rows + problem.num_cols + k;
        dual_constraint_matrix.col_start[p] = nnz;
        dual_constraint_matrix.i[nnz]       = vars_with_upper_bounds[k];
        dual_constraint_matrix.x[nnz]       = -1.0;
        nnz++;
      }
      dual_constraint_matrix.col_start[dual_cols] = nnz;
      settings.log.debug("dual_constraint_matrix nnz %d predicted %d\n", nnz, new_nnz);
      dual_problem.num_rows = dual_rows;
      dual_problem.num_cols = dual_cols;
      dual_problem.objective.resize(dual_cols, 0.0);
      for (i_t j = 0; j < problem.num_rows; j++) {
        dual_problem.objective[j] = problem.rhs[j];
      }
      for (i_t k = 0; k < num_upper_bounds; k++) {
        i_t j                     = problem.num_rows + problem.num_cols + k;
        dual_problem.objective[j] = problem.upper[vars_with_upper_bounds[k]];
      }
      dual_problem.A     = dual_constraint_matrix;
      dual_problem.rhs   = problem.objective;
      dual_problem.lower = std::vector<f_t>(dual_cols, 0.0);
      dual_problem.upper = std::vector<f_t>(dual_cols, inf);
      for (i_t j : equality_rows) {
        dual_problem.lower[j] = -inf;
      }
      dual_problem.obj_constant = 0.0;
      dual_problem.obj_scale    = -1.0;

      equal_rows = problem.num_cols;
      less_rows  = 0;

      dualize_info.vars_with_upper_bounds = vars_with_upper_bounds;
      dualize_info.zl_start               = problem.num_rows;
      dualize_info.zu_start               = problem.num_rows + problem.num_cols;
      dualize_info.equality_rows          = equality_rows;
      dualize_info.primal_problem         = problem;
      dualize_info.solving_dual           = true;

      problem = dual_problem;

      settings.log.printf("Solving the dual\n");
    }
  }

  if (less_rows > 0) {
    convert_less_than_to_equal(user_problem, row_sense, problem, less_rows, new_slacks);
  }

  // Add artifical variables
  if (!settings.barrier_presolve) { add_artifical_variables(problem, equality_rows, new_slacks); }
}

template <typename i_t, typename f_t>
i_t presolve(const lp_problem_t<i_t, f_t>& original,
             const simplex_solver_settings_t<i_t, f_t>& settings,
             lp_problem_t<i_t, f_t>& problem,
             presolve_info_t<i_t, f_t>& presolve_info)
{
  problem = original;
  std::vector<char> row_sense(problem.num_rows, '=');

  // The original problem may have a variable without a lower bound
  // but a finite upper bound
  // -inf < x_j <= u_j
  i_t no_lower_bound = 0;
  for (i_t j = 0; j < problem.num_cols; j++) {
    if (problem.lower[j] == -inf && problem.upper[j] < inf) { no_lower_bound++; }
  }
#ifdef PRINT_INFO
  settings.log.printf("%d variables with no lower bound\n", no_lower_bound);
#endif
  // The original problem may have nonzero lower bounds
  // 0 != l_j <= x_j <= u_j
  i_t nonzero_lower_bounds = 0;
  for (i_t j = 0; j < problem.num_cols; j++) {
    if (problem.lower[j] != 0.0 && problem.lower[j] > -inf) { nonzero_lower_bounds++; }
  }
  if (settings.barrier_presolve && nonzero_lower_bounds > 0) {
    settings.log.printf("Transforming %ld nonzero lower bound\n", nonzero_lower_bounds);
    presolve_info.removed_lower_bounds.resize(problem.num_cols);
    // We can construct a new variable: x'_j = x_j - l_j or x_j = x'_j + l_j
    // than we have 0 <= x'_j <= u_j - l_j
    // Constraints in the form:
    //  sum_{k != j} a_ik x_k + a_ij * x_j {=, <=} beta_i
    //  become
    //  sum_{k != j} a_ik x_k + a_ij * (x'_j + l_j) {=, <=} beta_i
    //  or
    //  sum_{k != j} a_ik x_k + a_ij * x'_j {=, <=} beta_i - a_{ij} l_j
    //
    // the cost function
    // sum_{k != j} c_k x_k + c_j * x_j
    // becomes
    // sum_{k != j} c_k x_k + c_j (x'_j + l_j)
    //
    // so we get the constant term c_j * l_j
    for (i_t j = 0; j < problem.num_cols; j++) {
      if (problem.lower[j] != 0.0 && problem.lower[j] > -inf) {
        for (i_t p = problem.A.col_start[j]; p < problem.A.col_start[j + 1]; p++) {
          i_t i   = problem.A.i[p];
          f_t aij = problem.A.x[p];
          problem.rhs[i] -= aij * problem.lower[j];
        }
        problem.obj_constant += problem.objective[j] * problem.lower[j];
        problem.upper[j] -= problem.lower[j];
        presolve_info.removed_lower_bounds[j] = problem.lower[j];
        problem.lower[j]                      = 0.0;
      }
    }
  }

  // Check for empty rows
  i_t num_empty_rows = 0;
  {
    csr_matrix_t<i_t, f_t> Arow(0, 0, 0);
    problem.A.to_compressed_row(Arow);
    for (i_t i = 0; i < problem.num_rows; i++) {
      if (Arow.row_start[i + 1] - Arow.row_start[i] == 0) { num_empty_rows++; }
    }
  }
  if (num_empty_rows > 0) {
    settings.log.printf("Presolve removing %d empty rows\n", num_empty_rows);
    i_t i = remove_empty_rows(problem, row_sense, num_empty_rows, presolve_info);
    if (i != 0) { return -1; }
  }

  // Check for empty cols
  i_t num_empty_cols = 0;
  {
    for (i_t j = 0; j < problem.num_cols; ++j) {
      if ((problem.A.col_start[j + 1] - problem.A.col_start[j]) == 0) { num_empty_cols++; }
    }
  }
  if (num_empty_cols > 0) {
    settings.log.printf("Presolve attempt to remove %d empty cols\n", num_empty_cols);
    remove_empty_cols(problem, num_empty_cols, presolve_info);
  }

  // Check for free variables
  i_t free_variables = 0;
  for (i_t j = 0; j < problem.num_cols; j++) {
    if (problem.lower[j] == -inf && problem.upper[j] == inf) { free_variables++; }
  }
  if (settings.barrier_presolve && free_variables > 0) {
#ifdef PRINT_INFO
    settings.log.printf("%d free variables\n", free_variables);
#endif

    // We have a variable x_j: with -inf < x_j < inf
    // we create new variables v and w with 0 <= v, w and x_j = v - w
    // Constraints
    // sum_{k != j} a_ik x_k + a_ij x_j {=, <=} beta
    // become
    // sum_{k != j} a_ik x_k + aij v - a_ij w {=, <=} beta
    //
    // The cost function
    // sum_{k != j} c_k x_k + c_j x_j
    // becomes
    // sum_{k != j} c_k x_k + c_j v - c_j w

    i_t num_cols = problem.num_cols + free_variables;
    i_t nnz      = problem.A.col_start[problem.num_cols];
    for (i_t j = 0; j < problem.num_cols; j++) {
      if (problem.lower[j] == -inf && problem.upper[j] == inf) {
        nnz += (problem.A.col_start[j + 1] - problem.A.col_start[j]);
      }
    }

    problem.A.col_start.resize(num_cols + 1);
    problem.A.i.resize(nnz);
    problem.A.x.resize(nnz);
    problem.lower.resize(num_cols);
    problem.upper.resize(num_cols);
    problem.objective.resize(num_cols);

    presolve_info.free_variable_pairs.resize(free_variables * 2);
    i_t pair_count = 0;
    i_t q          = problem.A.col_start[problem.num_cols];
    i_t col        = problem.num_cols;
    for (i_t j = 0; j < problem.num_cols; j++) {
      if (problem.lower[j] == -inf && problem.upper[j] == inf) {
        for (i_t p = problem.A.col_start[j]; p < problem.A.col_start[j + 1]; p++) {
          i_t i          = problem.A.i[p];
          f_t aij        = problem.A.x[p];
          problem.A.i[q] = i;
          problem.A.x[q] = -aij;
          q++;
        }
        problem.lower[col]                              = 0.0;
        problem.upper[col]                              = inf;
        problem.objective[col]                          = -problem.objective[j];
        presolve_info.free_variable_pairs[pair_count++] = j;
        presolve_info.free_variable_pairs[pair_count++] = col;
        problem.A.col_start[++col]                      = q;
        problem.lower[j]                                = 0.0;
      }
    }
    // assert(problem.A.p[num_cols] == nnz);
    problem.A.n      = num_cols;
    problem.num_cols = num_cols;
  }

  if (settings.barrier_presolve && settings.folding != 0) {
    folding(problem, settings, presolve_info);
  }

  // Check for dependent rows
  bool check_dependent_rows = false;  // settings.barrier;
  if (check_dependent_rows) {
    std::vector<i_t> dependent_rows;
    constexpr i_t kOk = -1;
    i_t infeasible;
    f_t dependent_row_start    = tic();
    const i_t independent_rows = find_dependent_rows(problem, settings, dependent_rows, infeasible);
    if (infeasible != kOk) {
      settings.log.printf("Found problem infeasible in presolve\n");
      return -1;
    }
    if (independent_rows < problem.num_rows) {
      const i_t num_dependent_rows = problem.num_rows - independent_rows;
      settings.log.printf("%d dependent rows\n", num_dependent_rows);
      csr_matrix_t<i_t, f_t> Arow(0, 0, 0);
      problem.A.to_compressed_row(Arow);
      remove_rows(problem, row_sense, Arow, dependent_rows, false);
    }
    settings.log.printf("Dependent row check in %.2fs\n", toc(dependent_row_start));
  }
  assert(problem.num_rows == problem.A.m);
  assert(problem.num_cols == problem.A.n);
  if (settings.print_presolve_stats && problem.A.m < original.A.m) {
    settings.log.printf("Presolve eliminated %d constraints\n", original.A.m - problem.A.m);
  }
  if (settings.print_presolve_stats && problem.A.n < original.A.n) {
    settings.log.printf("Presolve eliminated %d variables\n", original.A.n - problem.A.n);
  }
  if (settings.print_presolve_stats) {
    settings.log.printf("Presolved problem: %d constraints %d variables %d nonzeros\n",
                        problem.A.m,
                        problem.A.n,
                        problem.A.col_start[problem.A.n]);
  }
  assert(problem.rhs.size() == problem.A.m);
  return 0;
}

template <typename i_t, typename f_t>
void convert_user_lp_with_guess(const user_problem_t<i_t, f_t>& user_problem,
                                const lp_solution_t<i_t, f_t>& initial_solution,
                                const std::vector<f_t>& initial_slack,
                                lp_problem_t<i_t, f_t>& problem,
                                lp_solution_t<i_t, f_t>& converted_solution)
{
  std::vector<i_t> new_slacks;
  simplex_solver_settings_t<i_t, f_t> settings;
  dualize_info_t<i_t, f_t> dualize_info;
  convert_user_problem(user_problem, settings, problem, new_slacks, dualize_info);
  crush_primal_solution_with_slack(
    user_problem, problem, initial_solution.x, initial_slack, new_slacks, converted_solution.x);
  crush_dual_solution(user_problem,
                      problem,
                      new_slacks,
                      initial_solution.y,
                      initial_solution.z,
                      converted_solution.y,
                      converted_solution.z);
}

template <typename i_t, typename f_t>
void crush_primal_solution(const user_problem_t<i_t, f_t>& user_problem,
                           const lp_problem_t<i_t, f_t>& problem,
                           const std::vector<f_t>& user_solution,
                           const std::vector<i_t>& new_slacks,
                           std::vector<f_t>& solution)
{
  solution.resize(problem.num_cols, 0.0);
  for (i_t j = 0; j < user_problem.num_cols; j++) {
    solution[j] = user_solution[j];
  }

  std::vector<f_t> primal_residual(problem.num_rows);
  // Compute r = A*x
  matrix_vector_multiply(problem.A, 1.0, solution, 0.0, primal_residual);

  // Compute the value for each of the added slack variables
  for (i_t j : new_slacks) {
    const i_t col_start = problem.A.col_start[j];
    const i_t col_end   = problem.A.col_start[j + 1];
    const i_t diff      = col_end - col_start;
    assert(diff == 1);
    const i_t i = problem.A.i[col_start];
    assert(solution[j] == 0.0);
    const f_t beta  = problem.rhs[i];
    const f_t alpha = problem.A.x[col_start];
    assert(alpha == 1.0 || alpha == -1.0);
    const f_t slack_computed = (beta - primal_residual[i]) / alpha;
    solution[j] = std::max(problem.lower[j], std::min(slack_computed, problem.upper[j]));
  }

  primal_residual = problem.rhs;
  matrix_vector_multiply(problem.A, 1.0, solution, -1.0, primal_residual);
  const f_t primal_res   = vector_norm_inf<i_t, f_t>(primal_residual);
  constexpr bool verbose = false;
  if (verbose) { printf("Converted solution || A*x - b || %e\n", primal_res); }
}

template <typename i_t, typename f_t>
void crush_primal_solution_with_slack(const user_problem_t<i_t, f_t>& user_problem,
                                      const lp_problem_t<i_t, f_t>& problem,
                                      const std::vector<f_t>& user_solution,
                                      const std::vector<f_t>& user_slack,
                                      const std::vector<i_t>& new_slacks,
                                      std::vector<f_t>& solution)
{
  solution.resize(problem.num_cols, 0.0);
  for (i_t j = 0; j < user_problem.num_cols; j++) {
    solution[j] = user_solution[j];
  }

  std::vector<f_t> primal_residual(problem.num_rows);
  // Compute r = A*x
  matrix_vector_multiply(problem.A, 1.0, solution, 0.0, primal_residual);

  constexpr bool verbose = false;
  // Compute the value for each of the added slack variables
  for (i_t j : new_slacks) {
    const i_t col_start = problem.A.col_start[j];
    const i_t col_end   = problem.A.col_start[j + 1];
    const i_t diff      = col_end - col_start;
    assert(diff == 1);
    const i_t i = problem.A.i[col_start];
    assert(solution[j] == 0.0);
    const f_t si    = user_slack[i];
    const f_t beta  = problem.rhs[i];
    const f_t alpha = problem.A.x[col_start];
    assert(alpha == 1.0 || alpha == -1.0);
    const f_t slack_computed = (beta - primal_residual[i]) / alpha;
    if (std::abs(si - slack_computed) > 1e-6) {
      if (verbose) { printf("Slacks differ %d %e %e\n", j, si, slack_computed); }
    }
    solution[j] = si;
  }

  primal_residual = problem.rhs;
  matrix_vector_multiply(problem.A, 1.0, solution, -1.0, primal_residual);
  const f_t primal_res = vector_norm_inf<i_t, f_t>(primal_residual);
  if (verbose) { printf("Converted solution || A*x - b || %e\n", primal_res); }
  assert(primal_res < 1e-6);
}

template <typename i_t, typename f_t>
void crush_dual_solution(const user_problem_t<i_t, f_t>& user_problem,
                         const lp_problem_t<i_t, f_t>& problem,
                         const std::vector<i_t>& new_slacks,
                         const std::vector<f_t>& user_y,
                         const std::vector<f_t>& user_z,
                         std::vector<f_t>& y,
                         std::vector<f_t>& z)
{
  y.resize(problem.num_rows);
  for (i_t i = 0; i < user_problem.num_rows; i++) {
    y[i] = user_y[i];
  }
  z.resize(problem.num_cols);
  for (i_t j = 0; j < user_problem.num_cols; j++) {
    z[j] = user_z[j];
  }

  for (i_t j : new_slacks) {
    const i_t col_start = problem.A.col_start[j];
    const i_t col_end   = problem.A.col_start[j + 1];
    const i_t diff      = col_end - col_start;
    assert(diff == 1);
    const i_t i = problem.A.i[col_start];

    // A^T y + z = c
    // e_i^T y + z_j = c_j = 0
    // y_i + z_j = 0
    // z_j = - y_i;
    z[j] = -y[i];
  }

  // A^T y + z = c or A^T y + z - c = 0
  std::vector<f_t> dual_residual = z;
  for (i_t j = 0; j < problem.num_cols; j++) {
    dual_residual[j] -= problem.objective[j];
  }
  matrix_transpose_vector_multiply(problem.A, 1.0, y, 1.0, dual_residual);
  constexpr bool verbose = false;
  if (verbose) {
    printf("Converted solution || A^T y + z - c || %e\n", vector_norm_inf<i_t, f_t>(dual_residual));
  }
  for (i_t j = 0; j < problem.num_cols; ++j) {
    if (std::abs(dual_residual[j]) > 1e-6) {
      f_t ajty            = 0;
      const i_t col_start = problem.A.col_start[j];
      const i_t col_end   = problem.A.col_start[j + 1];
      for (i_t p = col_start; p < col_end; ++p) {
        const i_t i = problem.A.i[p];
        ajty += problem.A.x[p] * y[i];
        if (verbose) {
          printf("y %d %s %e Aij %e\n", i, user_problem.row_names[i].c_str(), y[i], problem.A.x[p]);
        }
      }
      if (verbose) {
        printf("dual res %d %e aty %e z %e c %e \n",
               j,
               dual_residual[j],
               ajty,
               z[j],
               problem.objective[j]);
      }
    }
  }
  const f_t dual_res_inf = vector_norm_inf<i_t, f_t>(dual_residual);
  assert(dual_res_inf < 1e-6);
}

template <typename i_t, typename f_t>
void uncrush_primal_solution(const user_problem_t<i_t, f_t>& user_problem,
                             const lp_problem_t<i_t, f_t>& problem,
                             const std::vector<f_t>& solution,
                             std::vector<f_t>& user_solution)
{
  user_solution.resize(user_problem.num_cols);
  assert(problem.num_cols >= user_problem.num_cols);
  std::copy(solution.begin(), solution.begin() + user_problem.num_cols, user_solution.data());
}

template <typename i_t, typename f_t>
void uncrush_dual_solution(const user_problem_t<i_t, f_t>& user_problem,
                           const lp_problem_t<i_t, f_t>& problem,
                           const std::vector<f_t>& y,
                           const std::vector<f_t>& z,
                           std::vector<f_t>& user_y,
                           std::vector<f_t>& user_z)
{
  user_y.resize(user_problem.num_rows);
  // Reduced costs are uncrushed just like the primal solution
  uncrush_primal_solution(user_problem, problem, z, user_z);

  // Adjust the sign of the dual variables y
  // We should have A^T y + z = c
  // In convert_user_problem, we converted >= to <=, so we need to adjust the sign of the dual
  // variables
  for (i_t i = 0; i < user_problem.num_rows; i++) {
    if (user_problem.row_sense[i] == 'G') {
      user_y[i] = -y[i];
    } else {
      user_y[i] = y[i];
    }
  }
}

template <typename i_t, typename f_t>
void uncrush_solution(const presolve_info_t<i_t, f_t>& presolve_info,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      const std::vector<f_t>& crushed_x,
                      const std::vector<f_t>& crushed_y,
                      const std::vector<f_t>& crushed_z,
                      std::vector<f_t>& uncrushed_x,
                      std::vector<f_t>& uncrushed_y,
                      std::vector<f_t>& uncrushed_z)
{
  std::vector<f_t> input_x = crushed_x;
  std::vector<f_t> input_y = crushed_y;
  std::vector<f_t> input_z = crushed_z;
  if (presolve_info.folding_info.is_folded) {
    // We solved a foled problem in the form
    // minimize c_prime^T x_prime
    // subject to A_prime x_prime = b_prime
    // x_prime >= 0
    //
    // where A_prime = C^s A D
    // and c_prime = D^T c
    // and b_prime = C^s b

    // We need to map this solution back to the converted problem
    //
    // minimize c^T x
    // subject to A * x = b
    //            x_j + w_j = u_j, j in U
    //            0 <= x,
    //            0 <= w

    i_t reduced_cols  = presolve_info.folding_info.D.n;
    i_t previous_cols = presolve_info.folding_info.D.m;
    i_t reduced_rows  = presolve_info.folding_info.C_s.m;
    i_t previous_rows = presolve_info.folding_info.C_s.n;

    std::vector<f_t> xtilde(previous_cols);
    std::vector<f_t> ytilde(previous_rows);
    std::vector<f_t> ztilde(previous_cols);

    matrix_vector_multiply(presolve_info.folding_info.D, 1.0, crushed_x, 0.0, xtilde);
    matrix_transpose_vector_multiply(presolve_info.folding_info.C_s, 1.0, crushed_y, 0.0, ytilde);
    matrix_transpose_vector_multiply(presolve_info.folding_info.D_s, 1.0, crushed_z, 0.0, ztilde);

    settings.log.debug("|| y ||_2 = %e\n", vector_norm2<i_t, f_t>(ytilde));
    settings.log.debug("|| z ||_2 = %e\n", vector_norm2<i_t, f_t>(ztilde));
    std::vector<f_t> dual_residual(previous_cols);
    for (i_t j = 0; j < previous_cols; j++) {
      dual_residual[j] = ztilde[j] - presolve_info.folding_info.c_tilde[j];
    }
    matrix_transpose_vector_multiply(
      presolve_info.folding_info.A_tilde, 1.0, ytilde, 1.0, dual_residual);
    settings.log.printf("Unfolded dual residual = %e\n", vector_norm_inf<i_t, f_t>(dual_residual));

    // Now we need to map the solution back to the original problem
    // minimize c^T x
    // subject to A * x = b
    //           0 <= x,
    //           x_j <= u_j, j in U
    input_x = xtilde;
    input_x.resize(previous_cols - presolve_info.folding_info.num_upper_bounds);
    input_y = ytilde;
    input_y.resize(previous_rows - presolve_info.folding_info.num_upper_bounds);
    input_z = ztilde;
    input_z.resize(previous_cols - presolve_info.folding_info.num_upper_bounds);
  }

  const i_t num_free_variables = presolve_info.free_variable_pairs.size() / 2;
  if (num_free_variables > 0) {
    settings.log.printf("Post-solve: Handling free variables %d\n", num_free_variables);
    // We added free variables so we need to map the crushed solution back to the original variables
    for (i_t k = 0; k < 2 * num_free_variables; k += 2) {
      const i_t u = presolve_info.free_variable_pairs[k];
      const i_t v = presolve_info.free_variable_pairs[k + 1];
      input_x[u] -= input_x[v];
    }
    input_z.resize(input_z.size() - num_free_variables);
    input_x.resize(input_x.size() - num_free_variables);
  }

  if (presolve_info.removed_variables.size() > 0) {
    settings.log.printf("Post-solve: Handling removed variables %d\n",
                        presolve_info.removed_variables.size());
    // We removed some variables, so we need to map the crushed solution back to the original
    // variables
    const i_t n = presolve_info.removed_variables.size() + presolve_info.remaining_variables.size();
    std::vector<f_t> input_x_copy = input_x;
    std::vector<f_t> input_z_copy = input_z;
    input_x_copy.resize(n);
    input_z_copy.resize(n);

    i_t k = 0;
    for (const i_t j : presolve_info.remaining_variables) {
      input_x_copy[j] = input_x[k];
      input_z_copy[j] = input_z[k];
      k++;
    }

    k = 0;
    for (const i_t j : presolve_info.removed_variables) {
      input_x_copy[j] = presolve_info.removed_values[k];
      input_z_copy[j] = presolve_info.removed_reduced_costs[k];
      k++;
    }
    input_x = input_x_copy;
    input_z = input_z_copy;
  }

  if (presolve_info.removed_constraints.size() > 0) {
    settings.log.printf("Post-solve: Handling removed constraints %d\n",
                        presolve_info.removed_constraints.size());
    // We removed some constraints, so we need to map the crushed solution back to the original
    // constraints
    const i_t m =
      presolve_info.removed_constraints.size() + presolve_info.remaining_constraints.size();
    std::vector<f_t> input_y_copy = input_y;
    input_y_copy.resize(m);

    i_t k = 0;
    for (const i_t i : presolve_info.remaining_constraints) {
      input_y_copy[i] = input_y[k];
      k++;
    }
    for (const i_t i : presolve_info.removed_constraints) {
      input_y_copy[i] = 0.0;
    }
    input_y = input_y_copy;
  }

  if (presolve_info.removed_lower_bounds.size() > 0) {
    settings.log.printf("Post-solve: Handling removed lower bounds %d\n",
                        presolve_info.removed_lower_bounds.size());
    // We removed some lower bounds so we need to map the crushed solution back to the original
    // variables
    for (i_t j = 0; j < input_x.size(); j++) {
      input_x[j] += presolve_info.removed_lower_bounds[j];
    }
  }
  assert(uncrushed_x.size() == input_x.size());
  assert(uncrushed_y.size() == input_y.size());
  assert(uncrushed_z.size() == input_z.size());

  uncrushed_x = input_x;
  uncrushed_y = input_y;
  uncrushed_z = input_z;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template void convert_user_problem<int, double>(
  const user_problem_t<int, double>& user_problem,
  const simplex_solver_settings_t<int, double>& settings,
  lp_problem_t<int, double>& problem,
  std::vector<int>& new_slacks,
  dualize_info_t<int, double>& dualize_info);

template void convert_user_lp_with_guess<int, double>(
  const user_problem_t<int, double>& user_problem,
  const lp_solution_t<int, double>& initial_solution,
  const std::vector<double>& initial_slack,
  lp_problem_t<int, double>& lp,
  lp_solution_t<int, double>& converted_solution);

template int presolve<int, double>(const lp_problem_t<int, double>& original,
                                   const simplex_solver_settings_t<int, double>& settings,
                                   lp_problem_t<int, double>& presolved,
                                   presolve_info_t<int, double>& presolve_info);

template void crush_primal_solution<int, double>(const user_problem_t<int, double>& user_problem,
                                                 const lp_problem_t<int, double>& problem,
                                                 const std::vector<double>& user_solution,
                                                 const std::vector<int>& new_slacks,
                                                 std::vector<double>& solution);

template void uncrush_primal_solution<int, double>(const user_problem_t<int, double>& user_problem,
                                                   const lp_problem_t<int, double>& problem,
                                                   const std::vector<double>& solution,
                                                   std::vector<double>& user_solution);

template void uncrush_dual_solution<int, double>(const user_problem_t<int, double>& user_problem,
                                                 const lp_problem_t<int, double>& problem,
                                                 const std::vector<double>& y,
                                                 const std::vector<double>& z,
                                                 std::vector<double>& user_y,
                                                 std::vector<double>& user_z);

template void uncrush_solution<int, double>(const presolve_info_t<int, double>& presolve_info,
                                            const simplex_solver_settings_t<int, double>& settings,
                                            const std::vector<double>& crushed_x,
                                            const std::vector<double>& crushed_y,
                                            const std::vector<double>& crushed_z,
                                            std::vector<double>& uncrushed_x,
                                            std::vector<double>& uncrushed_y,
                                            std::vector<double>& uncrushed_z);

template bool bound_strengthening<int, double>(
  const std::vector<char>& row_sense,
  const simplex_solver_settings_t<int, double>& settings,
  lp_problem_t<int, double>& problem,
  const csc_matrix_t<int, double>& Arow,
  const std::vector<variable_type_t>& var_types,
  const std::vector<bool>& bounds_changed);
#endif

}  // namespace cuopt::linear_programming::dual_simplex
