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

#include <dual_simplex/right_looking_lu.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/tic_toc.hpp>

#include <cmath>
#include <iostream>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>

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

template <typename i_t>
struct color_t {
  color_t(int8_t row_or_column, int8_t active, i_t color, std::vector<i_t> vertices)
    : row_or_column(row_or_column),
      active(active),
      color(color),
      vertices(vertices.begin(), vertices.end())
  {
  }
  int8_t row_or_column;
  int8_t active;
  i_t color;
  std::unordered_set<i_t> vertices;
};

constexpr int8_t kRow      = 0;
constexpr int8_t kCol      = 1;
constexpr int8_t kActive   = 1;
constexpr int8_t kInactive = 0;

template <typename i_t>
void find_vertices_to_refine(const std::unordered_set<i_t>& refining_color_vertices,
                             const std::vector<i_t>& offset,
                             const std::vector<i_t>& vertex_list,
                             const std::vector<i_t>& color_map,
                             std::vector<i_t>& marked_vertices,
                             std::vector<i_t>& vertices_to_refine,
                             std::vector<std::vector<i_t>>& vertices_to_refine_by_color,
                             std::vector<i_t>& marked_colors,
                             std::vector<i_t>& colors_to_update)
{
  for (i_t u : refining_color_vertices) {
    const i_t start = offset[u];
    const i_t end   = offset[u + 1];
    for (i_t p = start; p < end; p++) {
      const i_t v     = vertex_list[p];
      const i_t color = color_map[v];
      if (marked_vertices[v] == 0) {
        marked_vertices[v] = 1;
        vertices_to_refine.push_back(v);
        vertices_to_refine_by_color[color].push_back(v);
      }
      if (marked_colors[color] == 0) {
        marked_colors[color] = 1;
        colors_to_update.push_back(color);
      }
    }
  }
  for (i_t v : vertices_to_refine) {
    marked_vertices[v] = 0;
  }
  for (i_t color : colors_to_update) {
    marked_colors[color] = 0;
  }
}

template <typename i_t, typename f_t>
void compute_sums_of_refined_vertices(i_t refining_color,
                                      const std::unordered_set<i_t>& refining_color_vertices,
                                      const std::vector<i_t>& vertices_to_refine,
                                      const std::vector<i_t>& offsets,
                                      const std::vector<i_t>& vertex_list,
                                      const std::vector<f_t>& weight_list,
                                      const std::vector<i_t>& color_map,
                                      std::vector<f_t>& vertex_to_sum,
                                      std::vector<f_t>& max_sum_by_color)
{
  for (i_t v : refining_color_vertices) {
    const i_t start = offsets[v];
    const i_t end   = offsets[v + 1];
    for (i_t p = start; p < end; p++) {
      const i_t u = vertex_list[p];
      vertex_to_sum[u] += weight_list[p];
    }
  }

  for (i_t v : vertices_to_refine) {
    const i_t c = color_map[v];
    if (std::isnan(max_sum_by_color[c])) {
      max_sum_by_color[c] = vertex_to_sum[v];
    } else {
      max_sum_by_color[c] = std::max(max_sum_by_color[c], vertex_to_sum[v]);
    }
  }
}

template <typename i_t, typename f_t>
void compute_sums(const csc_matrix_t<i_t, f_t>& A,
                  const csr_matrix_t<i_t, f_t>& Arow,
                  i_t num_row_colors,
                  i_t num_col_colors,
                  i_t total_colors_seen,
                  const std::vector<i_t>& row_color_map,
                  const std::vector<i_t>& col_color_map,
                  const color_t<i_t>& refining_color,
                  std::vector<i_t>& colors_to_update,
                  std::vector<i_t>& marked_colors,
                  std::vector<i_t>& vertices_to_refine,
                  std::vector<i_t>& marked_vertices,
                  std::vector<std::vector<i_t>>& vertices_to_refine_by_color,
                  std::vector<f_t>& vertex_to_sum,
                  std::vector<f_t>& max_sum_by_color)
{
  i_t num_colors = num_row_colors + num_col_colors;
  colors_to_update.clear();
  vertices_to_refine.clear();
  if (refining_color.row_or_column == kRow) {
    // The refining color is a row color
    // Find all vertices (columns) that have a neighbor in the refining color
    colors_to_update.reserve(num_col_colors);
    find_vertices_to_refine(refining_color.vertices,
                            Arow.row_start,
                            Arow.j,
                            col_color_map,
                            marked_vertices,
                            vertices_to_refine,
                            vertices_to_refine_by_color,
                            marked_colors,
                            colors_to_update);

    // Let R be the refining color, R is a subset of the rows
    // We need to compute
    // sum_{v in R} A_vw for all w such that A_vw != 0
    // Note that if A_vw != 0 and A_vw' != 0, we may
    // have that w and w' are in a different (column) color
    compute_sums_of_refined_vertices(refining_color.color,
                                     refining_color.vertices,
                                     vertices_to_refine,
                                     Arow.row_start,
                                     Arow.j,
                                     Arow.x,
                                     col_color_map,
                                     vertex_to_sum,
                                     max_sum_by_color);
  } else {
    // The refining color is a column color
    // Find all vertices (rows) that have a neighbor in the refining color
    colors_to_update.reserve(num_row_colors);
    find_vertices_to_refine(refining_color.vertices,
                            A.col_start,
                            A.i,
                            row_color_map,
                            marked_vertices,
                            vertices_to_refine,
                            vertices_to_refine_by_color,
                            marked_colors,
                            colors_to_update);

    // Let Q be the refining color, Q is a subset of the columns
    // We need to compute
    // sum_{w in Q} A_vw for all v such that A_vw != 0
    // Note that if A_vw != 0 and A_v'w != 0, we may
    // have that v and v' are in a different (row) color
    compute_sums_of_refined_vertices(refining_color.color,
                                     refining_color.vertices,
                                     vertices_to_refine,
                                     A.col_start,
                                     A.i,
                                     A.x,
                                     row_color_map,
                                     vertex_to_sum,
                                     max_sum_by_color);
  }
}

template <typename i_t, typename f_t>
void find_colors_to_split(const std::vector<i_t> colors_to_update,
                          const std::vector<color_t<i_t>>& colors,
                          const std::vector<std::vector<i_t>>& vertices_to_refine_by_color,
                          const std::vector<f_t>& vertex_to_sum,
                          std::vector<f_t>& max_sum_by_color,
                          std::vector<f_t>& min_sum_by_color,
                          std::vector<i_t>& colors_to_split)
{
  for (i_t color : colors_to_update) {
    min_sum_by_color[color] = max_sum_by_color[color];
    for (i_t v : vertices_to_refine_by_color[color]) {
      if (vertex_to_sum[v] < min_sum_by_color[color]) {
        min_sum_by_color[color] = vertex_to_sum[v];
      }
    }

    if (vertices_to_refine_by_color[color].size() != colors[color].vertices.size()) {
      // We didn't touch all the vertices in the color. These vertices must have a sum of 0.0
      if (0.0 > max_sum_by_color[color]) { max_sum_by_color[color] = 0.0; }
      if (0.0 < min_sum_by_color[color]) { min_sum_by_color[color] = 0.0; }
    } else {
      // We touched all the vertices in the color. Compute the minimum sum seen
    }
  }

  colors_to_split.clear();
  for (i_t color : colors_to_update) {
    if (max_sum_by_color[color] < min_sum_by_color[color]) {
      printf("Color %d has max sum %e < min sum %e\n",
             color,
             max_sum_by_color[color],
             min_sum_by_color[color]);
      exit(1);
    }
    if (min_sum_by_color[color] < max_sum_by_color[color]) { colors_to_split.push_back(color); }
  }
}

template <typename i_t, typename f_t>
void split_colors(i_t color,
                  i_t refining_color,
                  int8_t side_being_split,
                  std::vector<f_t>& vertex_to_sum,
                  std::unordered_map<f_t, std::vector<i_t>>& color_sums,
                  std::unordered_map<f_t, i_t>& sum_to_sizes,
                  std::vector<color_t<i_t>>& colors,
                  std::vector<i_t>& color_stack,
                  std::vector<i_t>& color_in_stack,
                  std::vector<i_t>& color_map_B,
                  std::vector<i_t>& marked_vertices,
                  std::vector<std::vector<i_t>>& vertices_to_refine_by_color,
                  std::vector<f_t>& min_sum_by_color,
                  std::vector<f_t>& max_sum_by_color,
                  i_t& num_colors,
                  i_t& num_side_colors,
                  i_t& total_colors_seen)
{
  bool in_stack = color_in_stack[color];

  sum_to_sizes.clear();
  color_sums.clear();
  for (i_t v : vertices_to_refine_by_color[color]) {
    sum_to_sizes[vertex_to_sum[v]]++;
  }
  if (vertices_to_refine_by_color[color].size() != colors[color].vertices.size()) {
    const i_t remaining_size =
      colors[color].vertices.size() - vertices_to_refine_by_color[color].size();
    if (remaining_size < 0) {
      printf("Negative remaining size %d\n", remaining_size);

      printf("Color %d vertices\n");
      for (i_t v : colors[color].vertices) {
        printf("Vertex %d\n", v);
      }
      printf("Vertices to refine by color %d\n", color);
      for (i_t v : vertices_to_refine_by_color[color]) {
        printf("Vertex %d\n", v);
      }
      exit(1);
    }
    sum_to_sizes[0.0] += remaining_size;
    color_sums[0.0] = std::vector<i_t>();
  }

  i_t max_size = -1;
  f_t max_sum  = std::numeric_limits<f_t>::quiet_NaN();
  for (auto& [sum, size] : sum_to_sizes) {
    if (size > max_size) {
      max_size = size;
      max_sum  = sum;
    }
  }

  for (i_t v : vertices_to_refine_by_color[color]) {
    color_sums[vertex_to_sum[v]].push_back(v);
  }
  bool only_one = sum_to_sizes.size() == 1;
  if (only_one) {
    printf("Color %d has only one sum. color_sums size %ld. In stack %d\n",
           color,
           color_sums.size(),
           in_stack);
    exit(1);
  }

  i_t vertices_considered = 0;
  for (auto& [sum, vertices] : color_sums) {
    i_t size = vertices.size();
    if (sum == 0.0) {
      const i_t additional_size =
        (colors[color].vertices.size() - vertices_to_refine_by_color[color].size());
      size += additional_size;
    }

    vertices_considered += size;
    if (sum == 0.0) {
      // Push the current color back onto the stack
      if (!in_stack && max_size != size && max_sum != sum) {
        color_stack.push_back(color);
        color_in_stack[color] = 1;
      }
    } else {
      // Create a new color
      colors.emplace_back(side_being_split, kActive, total_colors_seen, vertices);

      // Push the new color onto the stack
      if (in_stack || !(max_size == size && max_sum == sum)) {
        color_stack.push_back(total_colors_seen);
        color_in_stack[total_colors_seen] = 1;
      }

      for (i_t v : vertices) {
        color_map_B[v] = total_colors_seen;
      }

      total_colors_seen++;
      num_colors++;
      num_side_colors++;
    }
  }
  if (vertices_considered != colors[color].vertices.size()) {
    printf("Vertices considered %d does not match color size %ld\n",
           vertices_considered,
           colors[color].vertices.size());
    exit(1);
  }

  for (i_t v : vertices_to_refine_by_color[color]) {
    if (color_map_B[v] != color) { colors[color].vertices.erase(v); }
  }
  if (colors[color].vertices.size() == 0) {
    colors[color].active = kInactive;
    num_colors--;
    num_side_colors--;
  }

  vertices_to_refine_by_color[color].clear();
  max_sum_by_color[color] = std::numeric_limits<f_t>::quiet_NaN();
  min_sum_by_color[color] = std::numeric_limits<f_t>::quiet_NaN();
}

template <typename i_t, typename f_t>
void color_lower_bounds(const csc_matrix_t<i_t, f_t>& A,
                        const csr_matrix_t<i_t, f_t>& Arow,
                        i_t& row_lower_bound,
                        i_t& col_lower_bound)
{
  // Look at the number of unique row sums and column sum.
  // This should be a lower bound on the number of colors.
  const i_t m = A.m;
  const i_t n = A.n;
  std::vector<f_t> row_sums(m, 0.0);
  std::vector<f_t> col_sums(n, 0.0);
  for (i_t i = 0; i < m; i++) {
    const i_t row_start = Arow.row_start[i];
    const i_t row_end   = Arow.row_start[i + 1];
    f_t sum             = 0.0;
    for (i_t p = row_start; p < row_end; p++) {
      sum += Arow.x[p];
    }
    row_sums[i] = sum;
  }
  for (i_t j = 0; j < n; j++) {
    const i_t col_start = A.col_start[j];
    const i_t col_end   = A.col_start[j + 1];
    f_t sum             = 0.0;
    for (i_t p = col_start; p < col_end; p++) {
      sum += A.x[p];
    }
    col_sums[j] = sum;
  }

  std::unordered_map<f_t, i_t> unique_row_sums;
  std::unordered_map<f_t, i_t> unique_col_sums;
  for (i_t i = 0; i < m; i++) {
    unique_row_sums[row_sums[i]]++;
  }
  for (i_t j = 0; j < n; j++) {
    unique_col_sums[col_sums[j]]++;
  }

  row_lower_bound = static_cast<i_t>(unique_row_sums.size());
  col_lower_bound = static_cast<i_t>(unique_col_sums.size());
}

template <typename i_t, typename f_t>
i_t color_graph(const csc_matrix_t<i_t, f_t>& A,
                const simplex_solver_settings_t<i_t, f_t>& settings,
                std::vector<color_t<i_t>>& colors,
                i_t row_threshold,
                i_t col_threshold,
                i_t& num_row_colors,
                i_t& num_col_colors,
                i_t& num_colors,
                i_t& total_colors_seen)
{
  f_t start_time    = tic();
  f_t last_log_time = start_time;
  const i_t m       = A.m;
  const i_t n       = A.n;
  csr_matrix_t<i_t, f_t> A_row(m, n, 1);
  A.to_compressed_row(A_row);

  i_t row_lower_bound = 0;
  i_t col_lower_bound = 0;

  color_lower_bounds(A, A_row, row_lower_bound, col_lower_bound);
  settings.log.debug("Folding: Row lower bound %d / %d col lower bound %d / %d\n",
                     row_lower_bound,
                     row_threshold,
                     col_lower_bound,
                     col_threshold);
  if (row_lower_bound > row_threshold || col_lower_bound > col_threshold) {
    settings.log.debug(
      "Folding: Row lower bound %d is greater than row threshold %d or col lower bound %d is "
      "greater than col threshold %d\n",
      row_lower_bound,
      row_threshold,
      col_lower_bound,
      col_threshold);
    return -1;
  }

  std::vector<i_t> all_rows_vertices(m);
  std::iota(all_rows_vertices.begin(), all_rows_vertices.end(), 0);
  color_t<i_t> all_rows(kRow, kActive, 0, all_rows_vertices);

  std::vector<i_t> all_cols_vertices(n);
  std::iota(all_cols_vertices.begin(), all_cols_vertices.end(), 0);
  color_t<i_t> all_cols(kCol, kActive, 1, all_cols_vertices);

  std::vector<i_t> color_stack;
  color_stack.push_back(0);
  color_stack.push_back(1);

  std::vector<i_t> row_color_map(m, 0);
  std::vector<i_t> col_color_map(n, 1);

  num_row_colors    = 1;
  num_col_colors    = 1;
  num_colors        = num_row_colors + num_col_colors;
  total_colors_seen = 2;  // The total colors seen includes inactive colors

  colors.emplace_back(all_rows);
  colors.emplace_back(all_cols);

  const i_t max_vertices = std::max(m, n);
  std::vector<f_t> vertex_to_sum(max_vertices, 0.0);
  std::vector<i_t> vertices_to_refine;
  vertices_to_refine.reserve(max_vertices);
  std::vector<i_t> marked_vertices(max_vertices, 0);

  i_t max_colors = m + n;
  std::vector<i_t> color_in_stack(max_colors, 0);
  color_in_stack[0] = 1;
  color_in_stack[1] = 1;

  std::unordered_map<f_t, std::vector<i_t>> color_sums;
  std::unordered_map<f_t, i_t> sum_to_sizes;

  std::vector<std::vector<i_t>> vertices_to_refine_by_color(max_colors);
  std::vector<f_t> max_sum_by_color(max_colors, std::numeric_limits<f_t>::quiet_NaN());
  std::vector<f_t> min_sum_by_color(max_colors, std::numeric_limits<f_t>::quiet_NaN());
  std::vector<i_t> marked_colors(max_colors, 0);

  std::vector<i_t> colors_to_split;
  colors_to_split.reserve(max_vertices);
  std::vector<i_t> colors_to_update;
  colors_to_update.reserve(max_vertices);

  i_t num_refinements       = 0;
  f_t colors_per_refinement = 0.0;
  while (!color_stack.empty()) {
    num_refinements++;
    i_t refining_color_index           = color_stack.back();
    const color_t<i_t>& refining_color = colors[refining_color_index];
    color_stack.pop_back();  // Can pop since refining color is no longer needed. New colors will be
                             // added to the stack.
    color_in_stack[refining_color_index] = 0;
    colors_to_update.clear();
    compute_sums(A,
                 A_row,
                 num_row_colors,
                 num_col_colors,
                 total_colors_seen,
                 row_color_map,
                 col_color_map,
                 refining_color,
                 colors_to_update,
                 marked_colors,
                 vertices_to_refine,
                 marked_vertices,
                 vertices_to_refine_by_color,
                 vertex_to_sum,
                 max_sum_by_color);

    colors_to_split.clear();
    find_colors_to_split(colors_to_update,
                         colors,
                         vertices_to_refine_by_color,
                         vertex_to_sum,
                         max_sum_by_color,
                         min_sum_by_color,
                         colors_to_split);
    // We now need to split the current colors into new colors
    if (refining_color.row_or_column == kRow) {
      // Refining color is a row color.
      // See if we need to split the column colors
      for (i_t color : colors_to_split) {
        split_colors(color,
                     refining_color.color,
                     kCol,
                     vertex_to_sum,
                     color_sums,
                     sum_to_sizes,
                     colors,
                     color_stack,
                     color_in_stack,
                     col_color_map,
                     marked_vertices,
                     vertices_to_refine_by_color,
                     min_sum_by_color,
                     max_sum_by_color,
                     num_colors,
                     num_col_colors,
                     total_colors_seen);
      }
    } else {
      // Refining color is a column color.
      // See if we need to split the row colors
      for (i_t color : colors_to_split) {
        split_colors(color,
                     refining_color.color,
                     kRow,
                     vertex_to_sum,
                     color_sums,
                     sum_to_sizes,
                     colors,
                     color_stack,
                     color_in_stack,
                     row_color_map,
                     marked_vertices,
                     vertices_to_refine_by_color,
                     min_sum_by_color,
                     max_sum_by_color,
                     num_colors,
                     num_row_colors,
                     total_colors_seen);
      }
    }

    for (i_t v : vertices_to_refine) {
      vertex_to_sum[v] = 0.0;
    }

    colors_per_refinement =
      static_cast<f_t>(num_row_colors + num_col_colors) / static_cast<f_t>(num_refinements);
    i_t projected_colors =
      num_row_colors + num_col_colors +
      static_cast<i_t>(colors_per_refinement * static_cast<f_t>(color_stack.size()));

    if (total_colors_seen >= max_colors - 10) {
      settings.log.debug(
        "Folding: Increase max colors from %d to %d\n", max_colors, max_colors * 2);
      max_colors *= 2;
      color_in_stack.resize(max_colors, 0);
      vertices_to_refine_by_color.resize(max_colors);
      max_sum_by_color.resize(max_colors);
      min_sum_by_color.resize(max_colors);
      marked_colors.resize(max_colors, 0);
    }

#ifdef DEBUG
    for (i_t k = 0; k < max_vertices; k++) {
      if (vertex_to_sum[k] != 0.0) {
        settings.log.printf("Folding: Vertex %d has sum %e\n", k, vertex_to_sum[k]);
        return -2;
      }
    }
#endif

    for (i_t color : colors_to_update) {
      vertices_to_refine_by_color[color].clear();
      max_sum_by_color[color] = std::numeric_limits<f_t>::quiet_NaN();
      min_sum_by_color[color] = std::numeric_limits<f_t>::quiet_NaN();
    }

#ifdef DEBUG
    for (i_t k = 0; k < total_colors_seen; k++) {
      if (vertices_to_refine_by_color[k].size() != 0) {
        settings.log.printf("Folding: Color %d has %ld vertices to refine. Not cleared\n",
                            k,
                            vertices_to_refine_by_color[k].size());
        return -2;
      }
    }
#endif

#ifdef DEBUG
    for (i_t i = 0; i < m; i++) {
      if (row_color_map[i] >= total_colors_seen) {
        settings.log.printf("Folding: Row color %d is not in the colors vector\n",
                            row_color_map[i]);
        return -2;
      }
    }
    for (i_t j = 0; j < n; j++) {
      if (col_color_map[j] >= total_colors_seen) {
        settings.log.printf("Folding: Column color %d is not in the colors vector. %d\n",
                            col_color_map[j],
                            num_colors);
        return -2;
      }
    }
#endif

#ifdef DEBUG
    // Count the number of active colors
    i_t num_active_colors     = 0;
    i_t num_active_row_colors = 0;
    i_t num_active_col_colors = 0;
    for (color_t<i_t>& color : colors) {
      if (color.active == kActive) {
        num_active_colors++;
        if (color.row_or_column == kRow) {
          num_active_row_colors++;
          for (i_t v : color.vertices) {
            if (row_color_map[v] != color.color) {
              settings.log.printf(
                "Folding: Row color map %d does not match color %d for vertex %d\n",
                row_color_map[v],
                color.color,
                v);
              return -2;
            }
          }
        } else {
          num_active_col_colors++;
          for (i_t v : color.vertices) {
            if (col_color_map[v] != color.color) {
              settings.log.printf(
                "Folding: Column color map %d does not match color %d for vertex %d\n",
                col_color_map[v],
                color.color,
                v);
              return -2;
            }
          }
        }
      }
    }
    // printf("Number of active colors: %d\n", num_active_colors);
    if (num_active_colors != num_colors) {
      settings.log.printf("Folding: Number of active colors does not match number of colors\n");
      return -2;
    }
    // printf("Number of active row colors: %d\n", num_active_row_colors);
    if (num_active_row_colors != num_row_colors) {
      settings.log.printf(
        "Folding: Number of active row colors does not match number of row colors\n");
      return -2;
    }
    // printf("Number of active column colors: %d\n", num_active_col_colors);
    if (num_active_col_colors != num_col_colors) {
      settings.log.printf(
        "Folding: Number of active column colors does not match number of column colors\n");
      return -2;
    }
#endif

    if (toc(last_log_time) > 1.0) {
      last_log_time = tic();
      settings.log.printf("Folding: %d refinements %d colors in %.2fs",
                          num_refinements,
                          num_row_colors + num_col_colors,
                          toc(start_time));
#ifdef PRINT_INFO
      settings.log.debug(
        "Number of refinements %8d. Number of colors %d (row colors %d, col colors %d) stack size "
        "%ld colors per "
        "refinement %.2f projected colors %d in %.2f seconds\n",
        num_refinements,
        num_row_colors + num_col_colors,
        num_row_colors,
        num_col_colors,
        color_stack.size(),
        colors_per_refinement,
        projected_colors,
        toc(start_time));
#endif
    }
    if (num_row_colors >= max_vertices) {
      settings.log.printf("Folding: Too many row colors %d max %d\n", num_row_colors, max_vertices);
      return -2;
    }
    if (num_col_colors >= max_vertices) {
      settings.log.printf(
        "Folding: Too many column colors %d max %d\n", num_col_colors, max_vertices);
      return -2;
    }

    if (num_row_colors > row_threshold || num_col_colors > col_threshold) {
      settings.log.printf("Folding: Number of colors exceeds threshold");
      return -1;
    }
  }
  settings.log.printf(
    "Folding: Colors %d. Refinements: %d\n", num_row_colors + num_col_colors, num_refinements);

  return 0;
}

template <typename i_t, typename f_t>
void folding(lp_problem_t<i_t, f_t>& problem,
             const simplex_solver_settings_t<i_t, f_t>& settings,
             presolve_info_t<i_t, f_t>& presolve_info)
{
  // Handle linear programs in the form
  // minimize c^T x
  // subject to A * x = b
  //            0 <= x,
  //              x_j <= u_j,   j in U

  // These can be converted into the form
  // minimize c^T x
  // subject to A * x = b
  //            x_j + w_j = u_j, j in U
  //            0 <= x,
  //            0 <= w
  //
  // We can then construct the augmented matrix
  //
  // [ A 0   b]
  // [ I I   u]
  // [ c 0 inf]
  f_t start_time = tic();

  i_t m = problem.num_rows;
  i_t n = problem.num_cols;

  if (settings.folding == -1 && (m > 1e6 || n > 1e6)) {
    settings.log.printf("Folding: Skipping\n");
    return;
  }

  i_t nz_obj = 0;
  for (i_t j = 0; j < n; j++) {
    if (problem.objective[j] != 0.0) { nz_obj++; }
  }
  i_t nz_rhs = 0;
  for (i_t i = 0; i < m; i++) {
    if (problem.rhs[i] != 0.0) { nz_rhs++; }
  }
  i_t nz_lb = 0;
  for (i_t j = 0; j < n; j++) {
    if (problem.lower[j] != 0.0) { nz_lb++; }
  }

  std::vector<f_t> finite_upper_bounds;
  finite_upper_bounds.reserve(n);
  i_t nz_ub = 0;
  for (i_t j = 0; j < n; j++) {
    if (problem.upper[j] != inf) {
      nz_ub++;
      finite_upper_bounds.push_back(problem.upper[j]);
    }
  }

  if (nz_lb > 0) {
    settings.log.printf("Folding: Can't handle problems with nonzero lower bounds\n");
    return;
  }

  i_t m_prime      = m + 1 + nz_ub;
  i_t n_prime      = n + nz_ub + 1;
  i_t augmented_nz = problem.A.col_start[n] + nz_obj + nz_rhs + 3 * nz_ub + 1;
  settings.log.debug("Folding: Augmented matrix has %d rows, %d columns, %d nonzeros\n",
                     m_prime,
                     n_prime,
                     augmented_nz);

  csc_matrix_t<i_t, f_t> augmented(m_prime, n_prime, augmented_nz);
  i_t nnz         = 0;
  i_t upper_count = 0;
  for (i_t j = 0; j < n; j++) {
    augmented.col_start[j] = nnz;
    // A
    const i_t col_start = problem.A.col_start[j];
    const i_t col_end   = problem.A.col_start[j + 1];
    for (i_t p = col_start; p < col_end; p++) {
      augmented.i[nnz] = problem.A.i[p];
      augmented.x[nnz] = problem.A.x[p];
      nnz++;
    }
    // I
    if (problem.upper[j] != inf) {
      augmented.i[nnz] = m + upper_count;
      augmented.x[nnz] = 1.0;
      upper_count++;
      nnz++;
    }
    // c
    if (problem.objective[j] != 0.0) {
      augmented.i[nnz] = m + nz_ub;
      augmented.x[nnz] = problem.objective[j];
      nnz++;
    }
  }
  // Column [ 0; I; 0]
  for (i_t j = n; j < n + nz_ub; j++) {
    augmented.col_start[j] = nnz;
    i_t k                  = j - n;
    augmented.i[nnz]       = m + k;
    augmented.x[nnz]       = 1.0;
    nnz++;
  }
  // Final column [b; u; inf]
  augmented.col_start[n + nz_ub] = nnz;
  for (i_t i = 0; i < m; i++) {
    if (problem.rhs[i] != 0.0) {
      augmented.i[nnz] = i;
      augmented.x[nnz] = problem.rhs[i];
      nnz++;
    }
  }
  upper_count = 0;
  for (i_t j = 0; j < n; j++) {
    if (problem.upper[j] != inf) {
      augmented.i[nnz] = m + upper_count;
      augmented.x[nnz] = problem.upper[j];
      upper_count++;
      nnz++;
    }
  }
  augmented.i[nnz] = m + nz_ub;
  augmented.x[nnz] = inf;
  nnz++;
  augmented.col_start[n + nz_ub + 1] = nnz;  // Finalize the matrix
  settings.log.debug("Folding: Augmented matrix has %d nonzeros predicted %d\n", nnz, augmented_nz);

  // Ensure only 1 inf in the augmented matrice
  i_t num_inf = 0;
  for (i_t p = 0; p < augmented_nz; p++) {
    if (augmented.x[p] == inf) { num_inf++; }
  }
  settings.log.debug("Folding: Augmented matrix has %d infs\n", num_inf);
  if (num_inf != 1) {
    settings.log.printf("Folding: Augmented matrix has %d infs, expected 1\n", num_inf);
    return;
  }

#ifdef WRITE_AUGMENTED
  {
    FILE* fid;
    fid = fopen("augmented.mtx", "w");
    augmented.write_matrix_market(fid);
    fclose(fid);
  }
#endif

  std::vector<color_t<i_t>> colors;
  i_t num_row_colors;
  i_t num_col_colors;
  i_t num_colors;
  i_t total_colors_seen;
  f_t color_start_time = tic();
  f_t fold_threshold   = settings.folding == -1 ? 0.50 : 1.0;
  i_t row_threshold    = static_cast<i_t>(fold_threshold * static_cast<f_t>(m));
  i_t col_threshold    = static_cast<i_t>(fold_threshold * static_cast<f_t>(n));
  i_t status           = color_graph(augmented,
                           settings,
                           colors,
                           row_threshold,
                           col_threshold,
                           num_row_colors,
                           num_col_colors,
                           num_colors,
                           total_colors_seen);
  if (status != 0) {
    settings.log.printf("Folding: Coloring aborted in %.2f seconds\n", toc(color_start_time));
    return;
  }
  settings.log.printf("Folding: Coloring time %.2f seconds\n", toc(color_start_time));

  // Go through the active colors and ensure that the row corresponding to the objective is its own
  // color
  std::vector<f_t> full_rhs(m_prime, 0.0);
  for (i_t i = 0; i < m; i++) {
    full_rhs[i] = problem.rhs[i];
  }
  upper_count = 0;
  for (i_t j = 0; j < n; j++) {
    if (problem.upper[j] != inf) {
      full_rhs[m + upper_count] = problem.upper[j];
      upper_count++;
    }
  }
  full_rhs[m + nz_ub] = inf;

  std::vector<i_t> row_colors;
  row_colors.reserve(num_row_colors);

  bool found_objective_color = false;
  i_t objective_color        = -1;
  i_t color_count            = 0;
  for (const color_t<i_t>& color : colors) {
    if (color.active == kActive) {
      if (color.row_or_column == kRow) {
        if (color.vertices.size() == 1) {
          if (*color.vertices.begin() == m + nz_ub) {
            settings.log.debug("Folding: Row color %d is the objective color\n", color.color);
            found_objective_color = true;
            objective_color       = color_count;
          } else {
            row_colors.push_back(color_count);
          }
        } else {
          row_colors.push_back(color_count);
#ifdef ROW_RHS_CHECK
          // Check that all vertices in the same row color have the same rhs value
          auto it       = color.vertices.begin();
          f_t rhs_value = full_rhs[*it];
          for (++it; it != color.vertices.end(); ++it) {
            if (full_rhs[*it] != rhs_value) {
              settings.log.printf(
                "Folding: RHS value for vertex %d is %e, but should be %e. Difference is %e\n",
                *it,
                full_rhs[*it],
                rhs_value,
                full_rhs[*it] - rhs_value);
              return;
            }
          }
#endif
        }
      }
    }
    color_count++;
  }

  if (!found_objective_color) {
    settings.log.printf("Folding: Objective color not found\n");
    return;
  }

  // Go through the active colors and ensure that the column corresponding to the rhs is its own
  // color
  bool found_rhs_color = false;
  i_t rhs_color        = -1;
  std::vector<f_t> full_objective(n_prime, 0.0);
  for (i_t j = 0; j < n; j++) {
    full_objective[j] = problem.objective[j];
  }
  full_objective[n_prime - 1] = inf;

  std::vector<i_t> col_colors;
  col_colors.reserve(num_col_colors - 1);
  color_count = 0;
  for (const color_t<i_t>& color : colors) {
    if (color.active == kActive) {
      if (color.row_or_column == kCol) {
        if (color.vertices.size() == 1) {
          if (*color.vertices.begin() == n_prime - 1) {
            settings.log.debug("Folding: Column color %d is the rhs color\n", color.color);
            found_rhs_color = true;
            rhs_color       = color_count;
          } else {
            col_colors.push_back(color_count);
          }
        } else {
          col_colors.push_back(color_count);
#ifdef COL_OBJ_CHECK
          // Check that all vertices in the same column color have the same objective value
          auto it             = color.vertices.begin();
          f_t objective_value = full_objective[*it];
          for (; it != color.vertices.end(); ++it) {
            if (full_objective[*it] != objective_value) {
              settings.log.printf(
                "Folding: Objective value for vertex %d is %e, but should be %e. Difference is "
                "%e\n",
                *it,
                full_objective[*it],
                objective_value,
                full_objective[*it] - objective_value);
              return;
            }
          }
#endif
        }
      }
    }
    color_count++;
  }

  if (!found_rhs_color) {
    settings.log.printf("Folding: RHS color not found\n");
    return;
  }

  // The original problem is in the form
  // minimize c^T x
  // subject to A x = b
  // x >= 0
  //
  // Let A_prime = C^s A D
  // b_prime = C^s b
  // c_prime = D^T c
  //
  // where C = Pi_P
  // and D = Pi_Q
  //
  // We will construct the new problem
  //
  // minimize c_prime^T x_prime
  // subject to A_prime x_prime = b_prime
  // x_prime >= 0
  //

  i_t previous_rows = m + nz_ub;
  i_t reduced_rows  = num_row_colors - 1;
  settings.log.debug("Folding: previous_rows %d reduced_rows %d\n", previous_rows, reduced_rows);

  // Construct the matrix Pi_P
  // Pi_vP = { 1 if v in P
  //         { 0 otherwise
  settings.log.debug("Folding: Constructing Pi_P\n");
  csc_matrix_t<i_t, f_t> Pi_P(previous_rows, reduced_rows, previous_rows);
  nnz = 0;
  for (i_t k = 0; k < reduced_rows; k++) {
    Pi_P.col_start[k]         = nnz;
    const i_t color_index     = row_colors[k];
    const color_t<i_t>& color = colors[color_index];
    for (i_t v : color.vertices) {
      Pi_P.i[nnz] = v;
      Pi_P.x[nnz] = 1.0;
      nnz++;
    }
  }
  Pi_P.col_start[reduced_rows] = nnz;
  settings.log.debug("Folding: Pi_P nz %d predicted %d\n", nnz, previous_rows);
  if (nnz != previous_rows) {
    settings.log.printf("Folding: Pi_P nz %d predicted %d\n", nnz, previous_rows);
    return;
  }
#ifdef WRITE_PI_P
  FILE* fid = fopen("Pi_P.txt", "w");
  Pi_P.write_matrix_market(fid);
  fclose(fid);
#endif

  // Start by constructing the matrix C^s
  // C^s = Pi^s_P
  // C^s_tv = Pi_vt / sum_v' Pi_v't
  // We have that sum_v' Pi_v't = | T |
  // C^s_tv = Pi_vt / | T | if t corresponds to color T
  // We have that Pi_vT = {1 if v in color T, 0 otherwiseS
  // C^s_tv = { 1/|T| if v in color T
  //         { 0
  settings.log.debug("Folding: Constructing C^s row\n");
  csr_matrix_t<i_t, f_t> C_s_row(reduced_rows, previous_rows, previous_rows);
  nnz = 0;
  settings.log.debug(
    "Folding: row_colors size %ld reduced_rows %d\n", row_colors.size(), reduced_rows);
  if (row_colors.size() != reduced_rows) {
    settings.log.printf("Folding: Bad row colors\n");
    return;
  }
  for (i_t k = 0; k < reduced_rows; k++) {
    C_s_row.row_start[k]  = nnz;
    const i_t color_index = row_colors[k];
    if (color_index < 0) {
      settings.log.printf("Folding: Bad row colors\n");
      return;
    }
    const color_t<i_t>& color = colors[color_index];
    const i_t color_size      = color.vertices.size();
    for (i_t v : color.vertices) {
      C_s_row.j[nnz] = v;
      C_s_row.x[nnz] = 1.0 / static_cast<f_t>(color_size);
      nnz++;
    }
  }
  C_s_row.row_start[reduced_rows] = nnz;
  settings.log.debug("Folding: C_s nz %d predicted %d\n", nnz, previous_rows);
  settings.log.debug("Folding: Converting C^s row to compressed column\n");

  // csc_matrix_t<i_t, f_t> C_s(reduced_rows, previous_rows, 1);
  presolve_info.folding_info.C_s.resize(reduced_rows, previous_rows, 1);
  csc_matrix_t<i_t, f_t>& C_s = presolve_info.folding_info.C_s;
  C_s_row.to_compressed_col(C_s);
  settings.log.debug("Folding: Completed C^s\n");
#ifdef DEBUG
  fid = fopen("C_s.txt", "w");
  C_s.write_matrix_market(fid);
  fclose(fid);

  // Verify that C^s Pi_P = I
  csc_matrix_t<i_t, f_t> product(reduced_rows, reduced_rows, 1);
  multiply(C_s, Pi_P, product);
  csc_matrix_t<i_t, f_t> identity(reduced_rows, reduced_rows, reduced_rows);
  for (i_t i = 0; i < reduced_rows; i++) {
    identity.col_start[i] = i;
    identity.i[i]         = i;
    identity.x[i]         = 1.0;
  }
  identity.col_start[reduced_rows] = reduced_rows;
  csc_matrix_t<i_t, f_t> error(reduced_rows, reduced_rows, 1);
  add(product, identity, 1.0, -1.0, error);
  printf("|| C^s Pi_P - I ||_1 = %f\n", error.norm1());
  if (error.norm1() > 1e-6) { exit(1); }
#endif

  // Construct that matrix D
  // D = Pi_Q
  // D_vQ = { 1 if v in Q
  //        { 0 otherwise
  settings.log.debug("Folding: Constructing D\n");
  i_t previous_cols = n + nz_ub;
  i_t reduced_cols  = num_col_colors - 1;
  settings.log.debug(
    "Folding: previous columns %d reduced columns %d\n", previous_cols, reduced_cols);
  presolve_info.folding_info.D.resize(previous_cols, reduced_cols, previous_cols);
  csc_matrix_t<i_t, f_t>& D = presolve_info.folding_info.D;
  nnz                       = 0;
  for (i_t k = 0; k < reduced_cols; k++) {
    D.col_start[k]        = nnz;
    const i_t color_index = col_colors[k];
    // printf("column color %d index %d colors size %ld\n", k, color_index, colors.size());
    if (color_index < 0) {
      settings.log.printf("Folding: Bad column colors\n");
      return;
    }
    const color_t<i_t>& color = colors[color_index];
    for (const i_t v : color.vertices) {
      D.i[nnz] = v;
      D.x[nnz] = 1.0;
      nnz++;
    }
  }
  D.col_start[reduced_cols] = nnz;
  settings.log.debug("Folding: D nz %d predicted %d\n", nnz, previous_cols);
#ifdef WRITE_D
  fid = fopen("D.txt", "w");
  D.write_matrix_market(fid);
  fclose(fid);
#endif

  // Construct D^s_tv
  // D^s_Tv = D_vT / sum_v' D_v'T
  csr_matrix_t<i_t, f_t> D_s_row(reduced_cols, previous_cols, previous_cols);
  nnz = 0;
  for (i_t k = 0; k < reduced_cols; k++) {
    D_s_row.row_start[k]      = nnz;
    const i_t color_index     = col_colors[k];
    const color_t<i_t>& color = colors[color_index];
    const i_t color_size      = color.vertices.size();
    for (i_t v : color.vertices) {
      D_s_row.j[nnz] = v;
      D_s_row.x[nnz] = 1.0 / static_cast<f_t>(color_size);
      nnz++;
    }
  }
  D_s_row.row_start[reduced_cols] = nnz;
  settings.log.debug("Folding: D^s row nz %d predicted %d\n", nnz, previous_cols);
  settings.log.debug("Folding: Converting D^s row to compressed column\n");
  // csc_matrix_t<i_t, f_t> D_s(reduced_cols, previous_cols, 1);
  presolve_info.folding_info.D_s.resize(reduced_cols, previous_cols, 1);
  csc_matrix_t<i_t, f_t>& D_s = presolve_info.folding_info.D_s;
  D_s_row.to_compressed_col(D_s);
#ifdef WRITE_DS
  settings.log.printf("Folding: Writing D^s\n");
  fid = fopen("D_s.txt", "w");
  D_s.write_matrix_market(fid);
  fclose(fid);
#endif

#ifdef DEBUG
  // Verify that D^s D = I
  csc_matrix_t<i_t, f_t> D_product(reduced_cols, reduced_cols, 1);
  multiply(D_s, D, D_product);
  csc_matrix_t<i_t, f_t> D_identity(reduced_cols, reduced_cols, reduced_cols);
  for (i_t i = 0; i < reduced_cols; i++) {
    D_identity.col_start[i] = i;
    D_identity.i[i]         = i;
    D_identity.x[i]         = 1.0;
  }
  D_identity.col_start[reduced_cols] = reduced_cols;
  csc_matrix_t<i_t, f_t> D_error(reduced_cols, reduced_cols, 1);
  add(D_product, D_identity, 1.0, -1.0, D_error);
  settings.log.debug("Folding: || D^s D - I ||_1 = %f\n", D_error.norm1());
  if (D_error.norm1() > 1e-6) {
    settings.log.printf("Folding: || D^s D - I ||_1 = %f\n", D_error.norm1());
    return;
  }

  // Construct the matrix X
  // X = C C^s
  settings.log.debug("Folding: Constructing X\n");
  csc_matrix_t<i_t, f_t> X(previous_rows, previous_rows, 1);
  multiply(Pi_P, C_s, X);
  settings.log.debug("Folding: Completed X\n");
  std::vector<f_t> X_col_sums(previous_rows);
  for (i_t j = 0; j < previous_rows; j++) {
    X_col_sums[j] = 0.0;
    for (i_t p = X.col_start[j]; p < X.col_start[j + 1]; p++) {
      X_col_sums[j] += X.x[p];
    }
    if (std::abs(X_col_sums[j] - 1.0) > 1e-6) {
      settings.log.printf("Folding: X_col_sums[%d] = %f\n", j, X_col_sums[j]);
      return;
    }
  }
  csr_matrix_t<i_t, f_t> X_row(previous_rows, previous_rows, 1);
  X.to_compressed_row(X_row);
  std::vector<f_t> X_row_sums(previous_rows);
  for (i_t i = 0; i < previous_rows; i++) {
    X_row_sums[i] = 0.0;
    for (i_t p = X_row.row_start[i]; p < X_row.row_start[i + 1]; p++) {
      X_row_sums[i] += X_row.x[p];
    }
    if (std::abs(X_row_sums[i] - 1.0) > 1e-6) {
      settings.log.printf("Folding: X_row_sums[%d] = %f\n", i, X_row_sums[i]);
      return;
    }
  }
  settings.log.printf("Folding: Verified X is doubly stochastic\n");

  // Construct the matrix Y
  // Y = D D^s
  settings.log.printf("Folding: Constructing Y\n");
  csc_matrix_t<i_t, f_t> Y(previous_cols, previous_cols, 1);
  multiply(D, D_s, Y);
  settings.log.debug("Folding: Completed Y\n");

  std::vector<f_t> Y_col_sums(previous_cols);
  for (i_t j = 0; j < previous_cols; j++) {
    Y_col_sums[j] = 0.0;
    for (i_t p = Y.col_start[j]; p < Y.col_start[j + 1]; p++) {
      Y_col_sums[j] += Y.x[p];
    }
    if (std::abs(Y_col_sums[j] - 1.0) > 1e-6) {
      settings.log.printf("Folding: Y_col_sums[%d] = %f\n", j, Y_col_sums[j]);
      return;
    }
  }
  csr_matrix_t<i_t, f_t> Y_row(previous_cols, previous_cols, 1);
  Y.to_compressed_row(Y_row);
  std::vector<f_t> Y_row_sums(previous_cols);
  for (i_t i = 0; i < previous_cols; i++) {
    Y_row_sums[i] = 0.0;
    for (i_t p = Y_row.row_start[i]; p < Y_row.row_start[i + 1]; p++) {
      Y_row_sums[i] += Y_row.x[p];
    }
    if (std::abs(Y_row_sums[i] - 1.0) > 1e-6) {
      settings.log.printf("Folding: Y_row_sums[%d] = %f\n", i, Y_row_sums[i]);
      return;
    }
  }
  settings.log.printf("Folding: Verified Y is doubly stochastic\n");
#endif
  // Construct the matrix A_tilde
  settings.log.debug("Folding: Constructing A_tilde\n");
  i_t A_nnz                      = problem.A.col_start[n];
  csc_matrix_t<i_t, f_t> A_tilde = augmented;
  A_tilde.remove_row(m + nz_ub);
  A_tilde.m--;
  A_tilde.remove_column(n + nz_ub);
  A_tilde.n--;
#ifdef WRITE_A_TILDE
  {
    FILE* fid;
    fid = fopen("A_tilde.txt", "w");
    A_tilde.write_matrix_market(fid);
    fclose(fid);
  }
#endif

  csr_matrix_t<i_t, f_t> A_tilde_row(A_tilde.m, A_tilde.n, A_tilde.col_start[A_tilde.n]);
  A_tilde.to_compressed_row(A_tilde_row);

#ifdef DEBUG
  std::vector<i_t> row_to_color(A_tilde.m, -1);
  for (i_t k = 0; k < total_colors_seen; k++) {
    const color_t<i_t>& row_color = colors[k];
    if (k == objective_color) continue;
    if (row_color.active == kActive && row_color.row_or_column == kRow) {
      for (i_t u : row_color.vertices) {
        row_to_color[u] = k;
      }
    }
  }
  std::vector<i_t> col_to_color(A_tilde.n, -1);
  for (i_t k = 0; k < total_colors_seen; k++) {
    const color_t<i_t>& col_color = colors[k];
    if (k == rhs_color) continue;
    if (col_color.active == kActive && col_color.row_or_column == kCol) {
      for (i_t v : col_color.vertices) {
        col_to_color[v] = k;
        // printf("Col %d assigned to color %d =? %d\n", v, k, col_color.color);
      }
    }
  }

  // Check that the partition is equitable
  for (i_t k = 0; k < total_colors_seen; k++) {
    const color_t<i_t> col_color = colors[k];
    if (col_color.active == kActive) {
      if (col_color.row_or_column == kCol) {
        // Check sum_{w in color} Avw = sum_{w in color} Avprimew for all (v, vprime) in row color P
        for (i_t h = 0; h < total_colors_seen; h++) {
          const color_t<i_t> row_color = colors[h];
          if (row_color.active == kActive && row_color.row_or_column == kRow) {
            for (i_t u : row_color.vertices) {
              for (i_t v : row_color.vertices) {
                if (u != v) {
                  f_t sum_Av      = 0.0;
                  f_t sum_Avprime = 0.0;
                  for (i_t p = A_tilde_row.row_start[u]; p < A_tilde_row.row_start[u + 1]; p++) {
                    const i_t j = A_tilde_row.j[p];
                    if (col_to_color[j] == k) { sum_Av += A_tilde_row.x[p]; }
                  }
                  for (i_t p = A_tilde_row.row_start[v]; p < A_tilde_row.row_start[v + 1]; p++) {
                    const i_t j = A_tilde_row.j[p];
                    if (col_to_color[j] == k) { sum_Avprime += A_tilde_row.x[p]; }
                  }
                  if (std::abs(sum_Av - sum_Avprime) > 1e-12) {
                    settings.log.printf(
                      "Folding: u %d v %d row color %d sum_A%d: %f sum_A%d: = %f\n",
                      u,
                      v,
                      h,
                      u,
                      sum_Av,
                      v,
                      sum_Avprime);
                    settings.log.printf("Folding: row color %d vertices: ", h);
                    for (i_t u : row_color.vertices) {
                      settings.log.printf("%d(%d) ", u, row_to_color[u]);
                    }
                    settings.log.printf("\n");
                    settings.log.printf("col color %d vertices: ", k);
                    for (i_t v : col_color.vertices) {
                      settings.log.printf("%d(%d) ", v, col_to_color[v]);
                    }
                    settings.log.printf("\n");
                    settings.log.printf("row %d\n", u);
                    for (i_t p = A_tilde_row.row_start[u]; p < A_tilde_row.row_start[u + 1]; p++) {
                      const i_t j = A_tilde_row.j[p];
                      settings.log.printf("row %d col %d column color %d value %e\n",
                                          u,
                                          j,
                                          col_to_color[j],
                                          A_tilde_row.x[p]);
                      if (col_to_color[j] == k) { sum_Av += A_tilde_row.x[p]; }
                    }
                    settings.log.printf("row %d\n", v);
                    for (i_t p = A_tilde_row.row_start[v]; p < A_tilde_row.row_start[v + 1]; p++) {
                      const i_t j = A_tilde_row.j[p];
                      settings.log.printf("row %d col %d column color %d value %e\n",
                                          v,
                                          j,
                                          col_to_color[j],
                                          A_tilde_row.x[p]);
                      if (col_to_color[j] == k) { sum_Avprime += A_tilde_row.x[p]; }
                    }
                    settings.log.printf("total colors seen %d\n", total_colors_seen);
                    return;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  settings.log.printf("Folding: Verified that the column partition is equitable\n");

  for (i_t k = 0; k < num_colors; k++) {
    const color_t<i_t>& row_color = colors[k];
    if (row_color.active == kActive) {
      if (row_color.row_or_column == kRow) {
        for (i_t h = 0; h < num_colors; h++) {
          const color_t<i_t>& col_color = colors[h];
          if (col_color.active == kActive && col_color.row_or_column == kCol) {
            for (i_t u : col_color.vertices) {
              for (i_t v : col_color.vertices) {
                if (u != v) {
                  f_t sum_A_u = 0.0;
                  f_t sum_A_v = 0.0;
                  for (i_t p = A_tilde.col_start[u]; p < A_tilde.col_start[u + 1]; p++) {
                    const i_t i = A_tilde.i[p];
                    if (row_to_color[i] == k) { sum_A_u += A_tilde.x[p]; }
                  }
                  for (i_t p = A_tilde.col_start[v]; p < A_tilde.col_start[v + 1]; p++) {
                    const i_t i = A_tilde.i[p];
                    if (row_to_color[i] == k) { sum_A_v += A_tilde.x[p]; }
                  }
                  if (std::abs(sum_A_u - sum_A_v) > 1e-12) {
                    settings.log.printf(
                      "Folding: u %d v %d row color %d sum_A%d: %f sum_A%d: = %f\n",
                      u,
                      v,
                      k,
                      u,
                      sum_A_u,
                      v,
                      sum_A_v);
                    return;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  settings.log.printf("Folding: Verified that the row partition is equitable\n");

  fid = fopen("X.txt", "w");
  X.write_matrix_market(fid);
  fclose(fid);
  fid = fopen("Y.txt", "w");
  Y.write_matrix_market(fid);
  fclose(fid);
#endif

  if (A_tilde.m != previous_rows || A_tilde.n != previous_cols) {
    settings.log.printf("Folding: A_tilde has %d rows and %d cols, expected %d and %d\n",
                        A_tilde.m,
                        A_tilde.n,
                        previous_rows,
                        previous_cols);
    return;
  }

  settings.log.debug("Folding: partial A_tilde nz %d predicted %d\n", nnz, A_nnz + 2 * nz_ub);

#ifdef DEBUG
  // Verify that XA = AY
  csc_matrix_t<i_t, f_t> XA(previous_rows, previous_cols, 1);
  multiply(X, A_tilde, XA);
  fid = fopen("XA.txt", "w");
  XA.write_matrix_market(fid);
  fclose(fid);
  csc_matrix_t<i_t, f_t> AY(previous_rows, previous_cols, 1);
  multiply(A_tilde, Y, AY);
  fid = fopen("AY.txt", "w");
  AY.write_matrix_market(fid);
  fclose(fid);
  csc_matrix_t<i_t, f_t> XA_AY_error(previous_rows, previous_cols, 1);
  add(XA, AY, 1.0, -1.0, XA_AY_error);
  printf("|| XA - AY ||_1 = %f\n", XA_AY_error.norm1());
#endif
  // Construct the matrix A_prime
  settings.log.debug("Folding: Constructing A_prime\n");
  csc_matrix_t<i_t, f_t> A_prime(reduced_rows, reduced_cols, 1);
  csc_matrix_t<i_t, f_t> AD(previous_rows, reduced_cols, 1);
  settings.log.debug("Folding: Multiplying A_tilde and D\n");
  multiply(A_tilde, D, AD);
  settings.log.debug("Folding: Multiplying C_s and AD\n");
  multiply(C_s, AD, A_prime);

  // Construct the vector b_prime
  settings.log.debug("Folding: Constructing b_prime\n");
  std::vector<f_t> b_tilde(previous_rows);
  for (i_t i = 0; i < m; i++) {
    b_tilde[i] = problem.rhs[i];
  }
  for (i_t i = m; i < m + nz_ub; i++) {
    b_tilde[i] = finite_upper_bounds[i - m];
  }
  std::vector<f_t> b_prime(reduced_rows);
  matrix_vector_multiply(C_s, 1.0, b_tilde, 0.0, b_prime);

  // Construct the vector c_prime
  settings.log.debug("Folding: Constructing c_prime\n");
  std::vector<f_t> c_tilde(previous_cols);
  for (i_t j = 0; j < n; j++) {
    c_tilde[j] = problem.objective[j];
  }
  for (i_t j = n; j < n + nz_ub; j++) {
    c_tilde[j] = 0.0;
  }
  std::vector<f_t> c_prime(reduced_cols);
  matrix_transpose_vector_multiply(D, 1.0, c_tilde, 0.0, c_prime);

  if (reduced_rows > reduced_cols) {
    settings.log.printf("Folding: Reduced rows %d > reduced cols %d\n", reduced_rows, reduced_cols);
    return;
  }

  // Construct a new problem
  settings.log.printf(
    "Folding: Constructing reduced problem: %d constraints %d variables and %d nonzeros\n",
    reduced_rows,
    reduced_cols,
    A_prime.col_start[reduced_cols]);

#ifdef SOLVE_REDUCED_PROBLEM
  user_problem_t<i_t, f_t> reduced_problem(problem.handle_ptr);
  reduced_problem.num_rows       = reduced_rows;
  reduced_problem.num_cols       = reduced_cols;
  reduced_problem.A              = A_prime;
  reduced_problem.objective      = c_prime;
  reduced_problem.rhs            = b_prime;
  reduced_problem.lower          = std::vector<f_t>(reduced_cols, 0.0);
  reduced_problem.upper          = std::vector<f_t>(reduced_cols, inf);
  reduced_problem.obj_constant   = 0.0;
  reduced_problem.obj_scale      = 1.0;
  reduced_problem.num_range_rows = 0;
  reduced_problem.row_sense      = std::vector<char>(reduced_rows, 'E');
  reduced_problem.var_types =
    std::vector<variable_type_t>(reduced_cols, variable_type_t::CONTINUOUS);
#endif

  problem.num_rows     = reduced_rows;
  problem.num_cols     = reduced_cols;
  problem.objective    = c_prime;
  problem.A            = A_prime;
  problem.rhs          = b_prime;
  problem.lower        = std::vector<f_t>(reduced_cols, 0.0);
  problem.upper        = std::vector<f_t>(reduced_cols, inf);
  problem.obj_constant = 0.0;
  problem.obj_scale    = 1.0;

  presolve_info.folding_info.c_tilde          = c_tilde;
  presolve_info.folding_info.A_tilde          = A_tilde;
  presolve_info.folding_info.is_folded        = true;
  presolve_info.folding_info.num_upper_bounds = nz_ub;

  settings.log.printf("Folding: time %.2f seconds\n", toc(start_time));

#ifdef SOLVE_REDUCED_PROBLEM
  // Solve the reduced problem
  lp_solution_t<i_t, f_t> reduced_solution(reduced_rows, reduced_cols);
  simplex_solver_settings_t<i_t, f_t> reduced_settings;
  reduced_settings.folding          = false;
  reduced_settings.barrier          = true;
  reduced_settings.barrier_presolve = true;
  reduced_settings.log.log          = true;

  solve_linear_program_with_barrier(reduced_problem, reduced_settings, reduced_solution);

  std::vector<f_t>& x_prime = reduced_solution.x;
  std::vector<f_t>& y_prime = reduced_solution.y;
  std::vector<f_t>& z_prime = reduced_solution.z;
  settings.log.printf("Folding: Reduced objective = %e\n", reduced_solution.objective);

  std::vector<f_t> x(previous_cols);
  std::vector<f_t> y(previous_rows);
  std::vector<f_t> z(previous_cols);
  matrix_vector_multiply(D, 1.0, x_prime, 0.0, x);
  matrix_transpose_vector_multiply(C_s, 1.0, y_prime, 0.0, y);
  matrix_transpose_vector_multiply(D_s, 1.0, z_prime, 0.0, z);

  settings.log.printf("Folding: Original primal objective = %e\n", dot<i_t, f_t>(c_tilde, x));
  settings.log.printf("Folding: Original dual objective   = %e\n", dot<i_t, f_t>(b_tilde, y));

  settings.log.printf("Folding: || y ||_2 = %e\n", vector_norm2<i_t, f_t>(y));
  printf("|| z ||_2 = %e\n", vector_norm2<i_t, f_t>(z));

  std::vector<f_t> dual_residual(previous_cols);
  for (i_t j = 0; j < previous_cols; j++) {
    dual_residual[j] = z[j] - c_tilde[j];
  }
  matrix_transpose_vector_multiply(A_tilde, 1.0, y, 1.0, dual_residual);
  settings.log.printf("Folding: Original dual residual = %e\n",
                      vector_norm_inf<i_t, f_t>(dual_residual));
#endif
}

template <typename i_t, typename f_t>
void convert_user_problem(const user_problem_t<i_t, f_t>& user_problem,
                          const simplex_solver_settings_t<i_t, f_t>& settings,
                          lp_problem_t<i_t, f_t>& problem,
                          std::vector<i_t>& new_slacks)
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

  if (less_rows > 0) {
    convert_less_than_to_equal(user_problem, row_sense, problem, less_rows, new_slacks);
  }

  // Add artifical variables
  if (!settings.barrier_presolve) {
    printf("Adding artifical variables\n");
    add_artifical_variables(problem, equality_rows, new_slacks);
  }
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

  // Check for free variables
  i_t free_variables = 0;
  for (i_t j = 0; j < problem.num_cols; j++) {
    if (problem.lower[j] == -inf && problem.upper[j] == inf) { free_variables++; }
  }
  if (free_variables > 0) {
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
  convert_user_problem(user_problem, settings, problem, new_slacks);
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

    printf("|| y ||_2 = %e\n", vector_norm2<i_t, f_t>(ytilde));
    printf("|| z ||_2 = %e\n", vector_norm2<i_t, f_t>(ztilde));
    std::vector<f_t> dual_residual(previous_cols);
    for (i_t j = 0; j < previous_cols; j++) {
      dual_residual[j] = ztilde[j] - presolve_info.folding_info.c_tilde[j];
    }
    matrix_transpose_vector_multiply(
      presolve_info.folding_info.A_tilde, 1.0, ytilde, 1.0, dual_residual);
    printf("Unfolded dual residual = %e\n", vector_norm_inf<i_t, f_t>(dual_residual));

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

  if (presolve_info.removed_constraints.size() == 0) {
    uncrushed_y = input_y;
  } else {
    printf("Handling removed constraints %d\n", presolve_info.removed_constraints.size());
    // We removed some constraints, so we need to map the crushed solution back to the original
    // constraints
    const i_t m =
      presolve_info.removed_constraints.size() + presolve_info.remaining_constraints.size();
    uncrushed_y.resize(m);

    i_t k = 0;
    for (const i_t i : presolve_info.remaining_constraints) {
      uncrushed_y[i] = input_y[k];
      k++;
    }
    for (const i_t i : presolve_info.removed_constraints) {
      uncrushed_y[i] = 0.0;
    }
  }

  if (presolve_info.removed_variables.size() == 0) {
    uncrushed_x = input_x;
    uncrushed_z = input_z;
  } else {
    printf("Handling removed variables %d\n", presolve_info.removed_variables.size());
    // We removed some variables, so we need to map the crushed solution back to the original
    // variables
    const i_t n = presolve_info.removed_variables.size() + presolve_info.remaining_variables.size();
    uncrushed_x.resize(n);
    uncrushed_z.resize(n);

    i_t k = 0;
    for (const i_t j : presolve_info.remaining_variables) {
      uncrushed_x[j] = input_x[k];
      uncrushed_z[j] = input_z[k];
      k++;
    }

    k = 0;
    for (const i_t j : presolve_info.removed_variables) {
      uncrushed_x[j] = presolve_info.removed_values[k];
      uncrushed_z[j] = presolve_info.removed_reduced_costs[k];
      k++;
    }
  }

  const i_t num_free_variables = presolve_info.free_variable_pairs.size() / 2;
  if (num_free_variables > 0) {
    printf("Handling free variables %d\n", num_free_variables);
    // We added free variables so we need to map the crushed solution back to the original variables
    for (i_t k = 0; k < 2 * num_free_variables; k += 2) {
      const i_t u = presolve_info.free_variable_pairs[k];
      const i_t v = presolve_info.free_variable_pairs[k + 1];
      uncrushed_x[u] -= uncrushed_x[v];
    }
    const i_t n = uncrushed_x.size();
    uncrushed_x.resize(n - num_free_variables);
    uncrushed_z.resize(n - num_free_variables);
  }

  if (presolve_info.removed_lower_bounds.size() > 0) {
    printf("Handling removed lower bounds %d\n", presolve_info.removed_lower_bounds.size());
    // We removed some lower bounds so we need to map the crushed solution back to the original
    // variables
    for (i_t j = 0; j < uncrushed_x.size(); j++) {
      uncrushed_x[j] += presolve_info.removed_lower_bounds[j];
    }
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template void convert_user_problem<int, double>(
  const user_problem_t<int, double>& user_problem,
  const simplex_solver_settings_t<int, double>& settings,
  lp_problem_t<int, double>& problem,
  std::vector<int>& new_slacks);

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
