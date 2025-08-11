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

#include <cmath>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
void bound_strengthening(const std::vector<char>& row_sense,
                         const simplex_solver_settings_t<i_t, f_t>& settings,
                         lp_problem_t<i_t, f_t>& problem)
{
  const i_t m = problem.num_rows;
  const i_t n = problem.num_cols;

  std::vector<f_t> constraint_lower(m);
  std::vector<i_t> num_lower_infinity(m);
  std::vector<i_t> num_upper_infinity(m);

  csc_matrix_t<i_t, f_t> Arow(1, 1, 1);
  problem.A.transpose(Arow);

  std::vector<i_t> less_rows;
  less_rows.reserve(m);

  for (i_t i = 0; i < m; ++i) {
    if (row_sense[i] == 'L') { less_rows.push_back(i); }
  }

  std::vector<f_t> lower = problem.lower;
  std::vector<f_t> upper = problem.upper;

  std::vector<i_t> updated_variables_list;
  updated_variables_list.reserve(n);
  std::vector<i_t> updated_variables_mark(n, 0);

  i_t iter                         = 0;
  const i_t iter_limit             = 10;
  i_t total_strengthened_variables = 0;
  settings.log.printf("Less equal rows %d\n", less_rows.size());
  while (iter < iter_limit && less_rows.size() > 0) {
    // Derive bounds on the constraints
    settings.log.printf("Running bound strengthening on %d rows\n",
                        static_cast<i_t>(less_rows.size()));
    for (i_t i : less_rows) {
      const i_t row_start   = Arow.col_start[i];
      const i_t row_end     = Arow.col_start[i + 1];
      num_lower_infinity[i] = 0;
      num_upper_infinity[i] = 0;

      f_t lower_limit = 0.0;
      for (i_t p = row_start; p < row_end; ++p) {
        const i_t j    = Arow.i[p];
        const f_t a_ij = Arow.x[p];
        if (a_ij > 0) {
          lower_limit += a_ij * lower[j];
        } else if (a_ij < 0) {
          lower_limit += a_ij * upper[j];
        }
        if (lower[j] == -inf && a_ij > 0) {
          num_lower_infinity[i]++;
          lower_limit = -inf;
        }
        if (upper[j] == inf && a_ij < 0) {
          num_lower_infinity[i]++;
          lower_limit = -inf;
        }
      }
      constraint_lower[i] = lower_limit;
    }

    // Use the constraint bounds to derive new bounds on the variables
    for (i_t i : less_rows) {
      if (std::isfinite(constraint_lower[i]) && num_lower_infinity[i] == 0) {
        const i_t row_start = Arow.col_start[i];
        const i_t row_end   = Arow.col_start[i + 1];
        for (i_t p = row_start; p < row_end; ++p) {
          const i_t k    = Arow.i[p];
          const f_t a_ik = Arow.x[p];
          if (a_ik > 0) {
            const f_t new_upper = lower[k] + (problem.rhs[i] - constraint_lower[i]) / a_ik;
            if (new_upper < upper[k]) {
              upper[k] = new_upper;
              if (lower[k] > upper[k]) {
                settings.log.printf(
                  "\t INFEASIBLE!!!!!!!!!!!!!!!!! constraint_lower %e lower %e rhs %e\n",
                  constraint_lower[i],
                  lower[k],
                  problem.rhs[i]);
              }
              if (!updated_variables_mark[k]) { updated_variables_list.push_back(k); }
            }
          } else if (a_ik < 0) {
            const f_t new_lower = upper[k] + (problem.rhs[i] - constraint_lower[i]) / a_ik;
            if (new_lower > lower[k]) {
              lower[k] = new_lower;
              if (lower[k] > upper[k]) {
                settings.log.printf("\t INFEASIBLE !!!!!!!!!!!!!!!!!!1\n");
              }
              if (!updated_variables_mark[k]) { updated_variables_list.push_back(k); }
            }
          }
        }
      }
    }
    less_rows.clear();

    // Update the bounds on the constraints
    settings.log.printf("Round %d: Strengthend %d variables\n",
                        iter,
                        static_cast<i_t>(updated_variables_list.size()));
    total_strengthened_variables += updated_variables_list.size();
    for (i_t j : updated_variables_list) {
      updated_variables_mark[j] = 0;
      const i_t col_start       = problem.A.col_start[j];
      const i_t col_end         = problem.A.col_start[j + 1];
      for (i_t p = col_start; p < col_end; ++p) {
        const i_t i = problem.A.i[p];
        less_rows.push_back(i);
      }
    }
    updated_variables_list.clear();
    iter++;
  }
  settings.log.printf("Total strengthened variables %d\n", total_strengthened_variables);
  problem.lower = lower;
  problem.upper = upper;
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
                std::vector<i_t>& row_marker)
{
  constexpr bool verbose = false;
  if (verbose) { printf("Removing rows %d %ld\n", Arow.m, row_marker.size()); }
  csr_matrix_t<i_t, f_t> Aout;
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
      if (problem.rhs[i] != 0.0) {
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
                      i_t& num_empty_rows)
{
  constexpr bool verbose = false;
  if (verbose) { printf("Problem has %d empty rows\n", num_empty_rows); }
  csr_matrix_t<i_t, f_t> Arow;
  problem.A.to_compressed_row(Arow);
  std::vector<i_t> row_marker(problem.num_rows);

  for (i_t i = 0; i < problem.num_rows; ++i) {
    if ((Arow.row_start[i + 1] - Arow.row_start[i]) == 0) {
      row_marker[i] = 1;
      if (verbose) {
        printf("Empty row %d start %d end %d\n", i, Arow.row_start[i], Arow.row_start[i + 1]);
      }
    } else {
      row_marker[i] = 0;
    }
  }
  const i_t retval = remove_rows(problem, row_sense, Arow, row_marker);
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

  csr_matrix_t<i_t, f_t> Arow;
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
  csr_matrix_t<i_t, f_t> Arow;
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
  for (i_t i = 0; i < m; ++i) {
    q[i] = i;
  }
  std::optional<std::vector<i_t>> optional_q = q;
  // TODO: Replace with right looking LU in crossover PR
  // i_t pivots = left_looking_lu(C, settings, 1e-13, optional_q, L, U, pinv);
  i_t pivots = 0;
  if (pivots < m) {
    const i_t num_dependent = m - pivots;
    std::vector<f_t> independent_rhs(pivots);
    std::vector<f_t> dependent_rhs(num_dependent);
    std::vector<i_t> dependent_row_list(num_dependent);
    i_t ind_count = 0;
    i_t dep_count = 0;
    for (i_t i = 0; i < m; ++i) {
      i_t row = (*optional_q)[i];
      if (i < pivots) {
        dependent_rows[row]          = 0;
        independent_rhs[ind_count++] = problem.rhs[row];
      } else {
        dependent_rows[row]             = 1;
        dependent_rhs[dep_count]        = problem.rhs[row];
        dependent_row_list[dep_count++] = row;
      }
    }

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
    bound_strengthening(row_sense, settings, problem);
  }

  // The original problem may have a variable without a lower bound
  // but a finite upper bound
  // -inf < x_j <= u_j
  i_t no_lower_bound = 0;
  for (i_t j = 0; j < problem.num_cols; j++) {
    if (problem.lower[j] == -INFINITY && problem.upper[j] < INFINITY) { no_lower_bound++; }
  }

  // The original problem may have nonzero lower bounds
  // 0 != l_j <= x_j <= u_j
  i_t nonzero_lower_bounds = 0;
  for (i_t j = 0; j < problem.num_cols; j++) {
    if (problem.lower[j] != 0.0 && problem.lower[j] > -INFINITY) { nonzero_lower_bounds++; }
  }

  if (less_rows > 0) {
    convert_less_than_to_equal(user_problem, row_sense, problem, less_rows, new_slacks);
  }

  // Add artifical variables
  add_artifical_variables(problem, equality_rows, new_slacks);
}

template <typename i_t, typename f_t>
i_t presolve(const lp_problem_t<i_t, f_t>& original,
             const simplex_solver_settings_t<i_t, f_t>& settings,
             lp_problem_t<i_t, f_t>& problem,
             presolve_info_t<i_t, f_t>& presolve_info)
{
  problem = original;
  std::vector<char> row_sense(problem.num_rows, '=');

  i_t free_variables = 0;
  for (i_t j = 0; j < problem.num_cols; j++) {
    if (problem.lower[j] == -INFINITY && problem.upper[j] == INFINITY) { free_variables++; }
  }
  if (free_variables > 0) { settings.log.printf("%d free variables\n", free_variables); }

  // Check for empty rows
  i_t num_empty_rows = 0;
  {
    csr_matrix_t<i_t, f_t> Arow;
    problem.A.to_compressed_row(Arow);
    for (i_t i = 0; i < problem.num_rows; i++) {
      if (Arow.row_start[i + 1] - Arow.row_start[i] == 0) { num_empty_rows++; }
    }
  }
  if (num_empty_rows > 0) {
    settings.log.printf("Presolve removing %d empty rows\n", num_empty_rows);
    i_t i = remove_empty_rows(problem, row_sense, num_empty_rows);
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
  constexpr bool check_dependent_rows = false;
  if (check_dependent_rows) {
    std::vector<i_t> dependent_rows;
    constexpr i_t kOk = -1;
    i_t infeasible;
    const i_t independent_rows = find_dependent_rows(problem, settings, dependent_rows, infeasible);
    if (infeasible != kOk) {
      settings.log.printf("Found problem infeasible in presolve\n");
      return -1;
    }
    if (independent_rows < problem.num_rows) {
      const i_t num_dependent_rows = problem.num_rows - independent_rows;
      settings.log.printf("%d dependent rows\n", num_dependent_rows);
      csr_matrix_t<i_t, f_t> Arow;
      problem.A.to_compressed_row(Arow);
      remove_rows(problem, row_sense, Arow, dependent_rows);
    }
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
                      const std::vector<f_t>& crushed_z,
                      std::vector<f_t>& uncrushed_x,
                      std::vector<f_t>& uncrushed_z)
{
  if (presolve_info.removed_variables.size() == 0) {
    uncrushed_x = crushed_x;
    uncrushed_z = crushed_z;
    return;
  }

  const i_t n = presolve_info.removed_variables.size() + presolve_info.remaining_variables.size();
  uncrushed_x.resize(n);
  uncrushed_z.resize(n);

  i_t k = 0;
  for (const i_t j : presolve_info.remaining_variables) {
    uncrushed_x[j] = crushed_x[k];
    uncrushed_z[j] = crushed_z[k];
    k++;
  }

  k = 0;
  for (const i_t j : presolve_info.removed_variables) {
    uncrushed_x[j] = presolve_info.removed_values[k];
    uncrushed_z[j] = presolve_info.removed_reduced_costs[k];
    k++;
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
                                            const std::vector<double>& crushed_z,
                                            std::vector<double>& uncrushed_x,
                                            std::vector<double>& uncrushed_z);
#endif

}  // namespace cuopt::linear_programming::dual_simplex
