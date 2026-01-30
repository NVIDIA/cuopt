/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/presolve.hpp>

#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/folding.hpp>
#include <dual_simplex/right_looking_lu.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/tic_toc.hpp>

#include <cmath>
#include <iostream>
#include <numeric>

namespace cuopt::linear_programming::dual_simplex {

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

  // Check to see if a variable participates in a quadratic objective
  std::vector<bool> has_quadratic_term(problem.num_cols, false);

  if (problem.Q.n > 0) {
    for (i_t j = 0; j < problem.num_cols; ++j) {
      const i_t row_start = problem.Q.row_start[j];
      const i_t row_end   = problem.Q.row_start[j + 1];
      if (row_end - row_start == 0) { continue; }
      // Q is symmetric, so its sufficient to check only the row size
      has_quadratic_term[j] = true;
    }
  }

  std::vector<i_t> col_marker(problem.num_cols);
  i_t new_cols = 0;
  for (i_t j = 0; j < problem.num_cols; ++j) {
    bool remove_var = false;
    if ((problem.A.col_start[j + 1] - problem.A.col_start[j]) == 0) {
      if (problem.objective[j] >= 0 && problem.lower[j] > -inf && !has_quadratic_term[j]) {
        presolve_info.removed_values.push_back(problem.lower[j]);
        problem.obj_constant += problem.objective[j] * problem.lower[j];
        remove_var = true;
      } else if (problem.objective[j] <= 0 && problem.upper[j] < inf && !has_quadratic_term[j]) {
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

  std::vector<i_t> col_old_to_new(problem.num_cols, -1);
  int new_j = 0;
  for (i_t j = 0; j < problem.num_cols; ++j) {
    if (!col_marker[j]) {
      objective[new_j] = problem.objective[j];
      lower[new_j]     = problem.lower[j];
      upper[new_j]     = problem.upper[j];
      presolve_info.remaining_variables.push_back(j);
      col_old_to_new[j] = new_j;
      new_j++;
    } else {
      num_empty_cols--;
    }
  }
  if (problem.Q.n > 0) {
    // There would not have been any non zero entry corresponding to the removed variables in the Q
    // matrix So we can just copy the row_start array and change the column indices to the new
    // indices
    for (i_t j = 0; j < problem.num_cols; ++j) {
      i_t new_j = col_old_to_new[j];
      assert(new_j <= j);
      if (new_j != -1) { problem.Q.row_start[new_j] = problem.Q.row_start[j]; }
    }
    problem.Q.row_start[new_cols] = problem.Q.row_start[problem.num_cols];
    problem.Q.row_start.resize(new_cols + 1);

    i_t Q_nnz = problem.Q.j.size();
    for (i_t jj = 0; jj < Q_nnz; ++jj) {
      i_t old_col = problem.Q.j[jj];
      i_t new_col = col_old_to_new[old_col];
      assert(new_col != -1);
      problem.Q.j[jj] = new_col;
    }
    problem.Q.m = new_cols;
    problem.Q.n = new_cols;
    problem.Q.check_matrix("After removing empty columns");
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
  constexpr bool verbose = false;
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
  // No need to clean up the Q matrix since we are not removing any columns
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
  if (verbose) {
    CUOPT_LOG_DEBUG("Converting %d less than inequalities to equalities\n", less_rows);
  }
  // We must convert rows in the form: a_i^T x <= beta
  // into: a_i^T x + s_i = beta, s_i >= 0

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
  if (pivots == CONCURRENT_HALT_RETURN) { return CONCURRENT_HALT_RETURN; }
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
                            const std::vector<i_t>& range_rows,
                            const std::vector<i_t>& equality_rows,
                            std::vector<i_t>& new_slacks)
{
  const i_t n                   = problem.num_cols;
  const i_t m                   = problem.num_rows;
  const i_t num_artificial_vars = equality_rows.size() - range_rows.size();
  const i_t num_cols            = n + num_artificial_vars;
  i_t nnz                       = problem.A.col_start[n] + num_artificial_vars;
  problem.A.col_start.resize(num_cols + 1);
  problem.A.i.resize(nnz);
  problem.A.x.resize(nnz);
  problem.lower.resize(num_cols);
  problem.upper.resize(num_cols);
  problem.objective.resize(num_cols);

  std::vector<bool> is_range_row(problem.num_rows, false);
  for (i_t i : range_rows) {
    is_range_row[i] = true;
  }

  i_t p = problem.A.col_start[n];
  i_t j = n;
  for (i_t i : equality_rows) {
    if (is_range_row[i]) { continue; }
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
  if (verbose) { printf("Added %d artificial variables\n", num_artificial_vars); }
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
    settings.log.printf("Converting problem with %d rows and %d columns and %d nonzeros\n",
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
  if (verbose) {
    settings.log.printf("Constraints < %d = %d > %d\n", less_rows, equal_rows, greater_rows);
  }

  if (user_problem.num_range_rows > 0) {
    if (verbose) { printf("Problem has %d range rows\n", user_problem.num_range_rows); }
    convert_range_rows(
      user_problem, row_sense, problem, less_rows, equal_rows, greater_rows, new_slacks);
  }

  if (greater_rows > 0) {
    convert_greater_to_less(user_problem, row_sense, problem, greater_rows, less_rows);
  }

  constexpr bool run_bounds_strengthening = false;
  if (run_bounds_strengthening) {
    csr_matrix_t<i_t, f_t> Arow(1, 1, 1);
    problem.A.to_compressed_row(Arow);

    settings.log.printf("Running bound strengthening\n");

    // Empty var_types means that all variables are continuous
    bounds_strengthening_t<i_t, f_t> strengthening(problem, Arow, row_sense, {});
    std::vector<bool> bounds_changed(problem.num_cols, true);
    strengthening.bounds_strengthening(settings, bounds_changed, problem.lower, problem.upper);
  }

  settings.log.debug(
    "equality rows %d less rows %d columns %d\n", equal_rows, less_rows, problem.num_cols);
  if (settings.barrier && settings.dualize != 0 && user_problem.Q_values.size() == 0 &&
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

  if (user_problem.Q_values.size() > 0) {
    settings.log.debug("Converting problem with %d quadratic nonzeros\n",
                       user_problem.Q_values.size());
    settings.log.debug(
      "problem.num_cols: %d user_problem.num_cols: %d\n", problem.num_cols, user_problem.num_cols);
    problem.Q.m      = problem.num_cols;
    problem.Q.n      = problem.num_cols;
    problem.Q.nz_max = user_problem.Q_values.size();
    problem.Q.row_start.resize(problem.num_cols + 1);
    for (i_t j = 0; j < user_problem.num_cols; j++) {
      problem.Q.row_start[j] = user_problem.Q_offsets[j];
    }
    i_t nz = user_problem.Q_offsets[user_problem.num_cols];
    for (i_t j = user_problem.num_cols; j <= problem.num_cols; j++) {
      problem.Q.row_start[j] = nz;
    }
    problem.Q.j = user_problem.Q_indices;
    problem.Q.x = user_problem.Q_values;
  }

  // Add artifical variables
  if (!settings.barrier_presolve) {
    add_artifical_variables(problem, user_problem.range_rows, equality_rows, new_slacks);
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

  if (no_lower_bound > 0) {
    settings.log.printf("%d variables with no lower bound\n", no_lower_bound);
  }

  // FIXME:: handle no lower bound case for barrier presolve

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

    std::vector<bool> lower_bounds_removed(problem.num_cols, false);
    for (i_t j = 0; j < problem.num_cols; j++) {
      if (problem.lower[j] != 0.0 && problem.lower[j] > -inf) {
        lower_bounds_removed[j]               = true;
        presolve_info.removed_lower_bounds[j] = problem.lower[j];
      }
    }

    auto old_objective = problem.objective;
    if (problem.Q.n > 0) {
      for (i_t row = 0; row < problem.num_cols; row++) {
        i_t row_start = problem.Q.row_start[row];
        i_t row_end   = problem.Q.row_start[row + 1];
        for (i_t p = row_start; p < row_end; p++) {
          i_t col = problem.Q.j[p];
          f_t qij = problem.Q.x[p];

          if (lower_bounds_removed[row]) {
            problem.objective[col] += 0.5 * qij * problem.lower[row];
          }
          if (lower_bounds_removed[col]) {
            problem.objective[row] += 0.5 * qij * problem.lower[col];
          }
          if (lower_bounds_removed[row] && lower_bounds_removed[col]) {
            problem.obj_constant += 0.5 * qij * problem.lower[row] * problem.lower[col];
          }
        }
      }
    }

    for (i_t j = 0; j < problem.num_cols; j++) {
      if (lower_bounds_removed[j]) {
        i_t col_start = problem.A.col_start[j];
        i_t col_end   = problem.A.col_start[j + 1];
        for (i_t p = col_start; p < col_end; p++) {
          i_t i   = problem.A.i[p];
          f_t aij = problem.A.x[p];
          problem.rhs[i] -= aij * problem.lower[j];
        }
        problem.obj_constant += old_objective[j] * problem.lower[j];
        problem.upper[j] -= problem.lower[j];
        problem.lower[j] = 0.0;
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

  problem.Q.check_matrix("Before free variable expansion");

  if (settings.barrier_presolve && free_variables > 0) {
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

    std::vector<i_t> pair_index(problem.num_cols, -1);
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
        pair_index[j]                                   = col;
        problem.A.col_start[++col]                      = q;
        problem.lower[j]                                = 0.0;
      }
    }

    if (problem.Q.n > 0) {
      std::vector<i_t> row_counts(num_cols, 0);
      i_t nz_count = problem.Q.row_start[problem.num_cols];
      for (i_t row = 0; row < problem.Q.n; row++) {
        i_t q_start     = problem.Q.row_start[row];
        i_t q_end       = problem.Q.row_start[row + 1];
        row_counts[row] = q_end - q_start;
        for (i_t qj = q_start; qj < q_end; qj++) {
          i_t col = problem.Q.j[qj];
          if (pair_index[row] != -1 && pair_index[col] != -1) {
            assert(pair_index[row] >= problem.num_cols);
            assert(pair_index[col] >= problem.num_cols);
            row_counts[row]++;
            row_counts[pair_index[row]] += 2;
            nz_count += 3;
          } else if (pair_index[col] != -1) {
            assert(pair_index[col] >= problem.num_cols);
            row_counts[row]++;
            nz_count++;
          } else if (pair_index[row] != -1) {
            assert(pair_index[row] >= problem.num_cols);
            row_counts[pair_index[row]]++;
            nz_count++;
          }
        }
      }

      std::vector<i_t> Q_row_start(num_cols + 1);
      Q_row_start[0] = 0;
      for (i_t row = 0; row < num_cols; row++) {
        Q_row_start[row + 1] = Q_row_start[row] + row_counts[row];
      }
      std::vector<i_t> Q_j(nz_count);
      std::vector<f_t> Q_x(nz_count);
      auto row_starts = Q_row_start;
      // First copy the original Q ma
      for (i_t row = 0; row < problem.Q.n; row++) {
        i_t q_start = problem.Q.row_start[row];
        i_t q_end   = problem.Q.row_start[row + 1];
        i_t q_nz    = Q_row_start[row];
        for (i_t qj = q_start; qj < q_end; qj++) {
          i_t col   = problem.Q.j[qj];
          f_t qij   = problem.Q.x[qj];
          Q_j[q_nz] = col;
          Q_x[q_nz] = qij;
          q_nz++;
        }
        row_starts[row] = q_nz;
      }

      // Expand the Q matrix for the free variables
      for (i_t row = 0; row < problem.Q.n; row++) {
        i_t q_start = problem.Q.row_start[row];
        i_t q_end   = problem.Q.row_start[row + 1];
        for (i_t qj = q_start; qj < q_end; qj++) {
          i_t col = problem.Q.j[qj];
          f_t qij = problem.Q.x[qj];
          if (pair_index[row] != -1 && pair_index[col] != -1) {
            Q_j[row_starts[row]] = pair_index[col];
            Q_x[row_starts[row]] = -qij;
            row_starts[row]++;

            Q_j[row_starts[pair_index[row]]] = col;
            Q_x[row_starts[pair_index[row]]] = -qij;
            row_starts[pair_index[row]]++;

            Q_j[row_starts[pair_index[row]]] = pair_index[col];
            Q_x[row_starts[pair_index[row]]] = qij;
            row_starts[pair_index[row]]++;
          } else if (pair_index[col] != -1) {
            Q_j[row_starts[row]] = pair_index[col];
            Q_x[row_starts[row]] = -qij;
            row_starts[row]++;
          } else if (pair_index[row] != -1) {
            Q_j[row_starts[pair_index[row]]] = col;
            Q_x[row_starts[pair_index[row]]] = -qij;
            row_starts[pair_index[row]]++;
          }
        }
      }

      problem.Q.m = problem.Q.n = num_cols;
      problem.Q.nz_max          = Q_row_start[num_cols];
      problem.Q.row_start       = Q_row_start;
      problem.Q.j               = Q_j;
      problem.Q.x               = Q_x;
      problem.Q.check_matrix("After free variable expansion");
    }

    // assert(problem.A.p[num_cols] == nnz);
    problem.A.n      = num_cols;
    problem.num_cols = num_cols;
  }

  if (settings.barrier_presolve && settings.folding != 0 && problem.Q.n == 0) {
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
    if (independent_rows == CONCURRENT_HALT_RETURN) { return CONCURRENT_HALT_RETURN; }
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
f_t crush_dual_solution(const user_problem_t<i_t, f_t>& user_problem,
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

  std::vector<bool> is_range_row(problem.num_rows, false);
  for (i_t i = 0; i < user_problem.range_rows.size(); i++) {
    is_range_row[user_problem.range_rows[i]] = true;
  }
  assert(user_problem.num_rows == problem.num_rows);

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
    if (is_range_row[i]) {
      z[j] = y[i];
    } else {
      z[j] = -y[i];
    }
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
  // TODO: fix me! In test ./cpp/build/tests/linear_programming/C_API_TEST
  // c_api/TimeLimitTestFixture.time_limit/2 this is crashing. It is crashing only if it is run as
  // whole in sequence and not filtering the respective test. Crash could be observed in previous
  // versions by setting probing cache time to zero. assert(dual_res_inf < 1e-6);
  return dual_res_inf;
}

template <typename i_t, typename f_t>
void uncrush_primal_solution(const user_problem_t<i_t, f_t>& user_problem,
                             const lp_problem_t<i_t, f_t>& problem,
                             const std::vector<f_t>& solution,
                             std::vector<f_t>& user_solution)
{
  user_solution.resize(user_problem.num_cols);
  assert(problem.num_cols >= user_problem.num_cols);
  assert(solution.size() >= user_problem.num_cols);
  std::copy(solution.begin(),
            solution.begin() + std::min((i_t)solution.size(), user_problem.num_cols),
            user_solution.data());
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
  std::vector<f_t> input_x             = crushed_x;
  std::vector<f_t> input_y             = crushed_y;
  std::vector<f_t> input_z             = crushed_z;
  std::vector<i_t> free_variable_pairs = presolve_info.free_variable_pairs;
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

    // If the original problem had free variables we need to reinstate them
    free_variable_pairs = presolve_info.folding_info.previous_free_variable_pairs;
  }

  const i_t num_free_variables = free_variable_pairs.size() / 2;
  if (num_free_variables > 0) {
    settings.log.printf("Post-solve: Handling free variables %d\n", num_free_variables);
    // We added free variables so we need to map the crushed solution back to the original variables
    for (i_t k = 0; k < 2 * num_free_variables; k += 2) {
      const i_t u = free_variable_pairs[k];
      const i_t v = free_variable_pairs[k + 1];
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

template double crush_dual_solution<int, double>(const user_problem_t<int, double>& user_problem,
                                                 const lp_problem_t<int, double>& problem,
                                                 const std::vector<int>& new_slacks,
                                                 const std::vector<double>& user_y,
                                                 const std::vector<double>& user_z,
                                                 std::vector<double>& y,
                                                 std::vector<double>& z);

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

#endif

}  // namespace cuopt::linear_programming::dual_simplex
