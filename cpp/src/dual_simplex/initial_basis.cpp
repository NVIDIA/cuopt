/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/initial_basis.hpp>

#include <dual_simplex/right_looking_lu.hpp>
#include <dual_simplex/singletons.hpp>
#include <dual_simplex/tic_toc.hpp>

#include <cassert>
#include <cmath>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
i_t initial_basis_selection(const lp_problem_t<i_t, f_t>& problem,
                            const simplex_solver_settings_t<i_t, f_t>& settings,
                            const std::vector<i_t>& candidate_columns,
                            f_t start_time,
                            std::vector<variable_status_t>& vstatus,
                            std::vector<i_t>& dependent_rows)
{
  i_t m  = problem.num_rows;
  i_t n  = problem.num_cols;
  i_t nz = problem.A.col_start[n];
  i_t N  = candidate_columns.size();
  assert(m == problem.A.m);
  assert(n == problem.A.n);
  assert(N == vstatus.size());

  i_t Cnz = 0;
  {
    for (i_t k = 0; k < N; ++k) {
      const i_t j         = candidate_columns[k];
      const i_t col_start = problem.A.col_start[j];
      const i_t col_end   = problem.A.col_start[j + 1];
      Cnz += col_end - col_start;
    }
  }
  csc_matrix_t<i_t, f_t> CT(m, N, Cnz);
  i_t Cq = 0;
  for (i_t k = 0; k < N; ++k) {
    const i_t j         = candidate_columns[k];
    const i_t col_start = problem.A.col_start[j];
    const i_t col_end   = problem.A.col_start[j + 1];
    CT.col_start[k]     = Cq;
    for (i_t p = col_start; p < col_end; ++p) {
      CT.i[Cq] = problem.A.i[p];
      CT.x[Cq] = problem.A.x[p];
      Cq++;
    }
  }
  CT.col_start[N] = Cq;
  assert(Cq == Cnz);

  // Form C = A(:, candidate_cols)'
  csc_matrix_t<i_t, f_t> C(N, m, 1);
  CT.transpose(C);
  assert(C.col_start[m] == Cnz);

  // Calculate L*U = C(p, :)
  csc_matrix_t<i_t, f_t> L(N, m, nz);
  csc_matrix_t<i_t, f_t> U(m, m, nz);
  std::vector<i_t> pinv(N);
  std::vector<i_t> q(m);

  i_t pivots;
  f_t factorization_start   = tic();
  bool need_factorization   = true;
  bool eliminate_singletons = settings.eliminate_singletons;
  if (eliminate_singletons) {
    i_t row_singletons;
    i_t col_singletons;
    std::vector<i_t> row_perm(N);
    find_singletons(C, row_singletons, row_perm, col_singletons, q);
    std::vector<i_t> row_perm_inv(N);
    inverse_permutation(row_perm, row_perm_inv);

    i_t num_singletons = row_singletons + col_singletons;

    // P C Q = [ U_11  U_12 U_13 ]
    //         [      L_22      ]
    //         [      L_32  S   ]
    //
    // where U_11 is col_singletons x col_singletons and upper triangular
    //       L_22 is row_singletons x row_singletons and lower triangular
    //       U_12 is col_singletons x row_singletons
    //       U_13 is col_singletons x (m - col_singletons - row_singletons)
    //       L_32 is (N - col_singletons - row_singletons) x row_singletons
    //       S    is (N - col_singletons - row_singletons) x (m - col_singletons - row_singletons)

    if (num_singletons > 0) {
      settings.log.debug(
        "Singletons found %d (row %d col %d)\n", num_singletons, row_singletons, col_singletons);

      i_t S_rows = N - num_singletons;
      i_t S_cols = m - num_singletons;
      settings.log.debug("S_rows %d S_cols %d\n", S_rows, S_cols);
      i_t S_pivots = 0;
      if (S_cols > 0) {
        csc_matrix_t<i_t, f_t> S(S_rows, S_cols, 1);
        S.i.clear();
        S.x.clear();

        // Build S
        i_t Snz = 0;
        for (i_t k = num_singletons; k < m; ++k) {
          S.col_start[k - num_singletons] = Snz;
          const i_t j                     = q[k];
          const i_t col_start             = C.col_start[j];
          const i_t col_end               = C.col_start[j + 1];
          for (i_t p = col_start; p < col_end; ++p) {
            const i_t i = row_perm_inv[C.i[p]];
            if (i >= num_singletons) {
              const i_t row_i = i - num_singletons;
              assert(row_i < S_rows);
              S.i.push_back(row_i);
              S.x.push_back(C.x[p]);
              Snz++;
            }
          }
        }
        S.col_start[S_cols] = Snz;
        settings.log.debug("Snz %d\n", Snz);
        std::vector<i_t> S_p_inv(S_rows);
        std::vector<i_t> S_q(S_cols);
        if (Snz > 0) {
          // Factorize S
          S_pivots =
            right_looking_lu_row_permutation_only(S, settings, 1e-12, start_time, S_q, S_p_inv);
          if (S_pivots < 0) {
            settings.log.printf("Aborting: right looking LU factorization\n");
            return S_pivots;
          }
        } else {
          for (i_t k = 0; k < S_cols; ++k) {
            S_q[k] = k;
          }
          for (i_t k = 0; k < S_rows; ++k) {
            S_p_inv[k] = k;
          }
        }
        settings.log.debug("S_pivots %d\n", S_pivots);
        // Fix up column permutations
        std::vector<i_t> col_perm_sav(m - num_singletons);
        i_t q_j = 0;
        for (i_t h = num_singletons; h < m; ++h) {
          col_perm_sav[q_j] = q[h];
          q_j++;
        }
        q_j = 0;
        for (i_t h = num_singletons; h < m; ++h) {
          q[h] = col_perm_sav[S_q[q_j]];
          q_j++;
        }

        // Fix up row permutations
        std::vector<i_t> last_perm(S_rows);
        std::vector<i_t> S_p(S_rows);
        inverse_permutation(S_p_inv, S_p);
        for (i_t k = 0; k < S_rows; ++k) {
          last_perm[k] = row_perm[num_singletons + k];
        }
        for (i_t k = 0; k < S_rows; ++k) {
          row_perm[num_singletons + k] = last_perm[S_p[k]];
        }
      }
      pivots = num_singletons + S_pivots;
      inverse_permutation(row_perm, pinv);
      need_factorization = false;
    }
  }

  if (need_factorization) {
    pivots = right_looking_lu_row_permutation_only(C, settings, 1e-12, start_time, q, pinv);
  }

  f_t factorization_time = toc(factorization_start);
  settings.log.printf("Initial basis factorization time: %.2f seconds\n", factorization_time);
  settings.log.debug("pivots %d m %d\n", pivots, m);

  if (pivots < 0) {
    settings.log.printf("Aborting: right looking LU factorization\n");
    return pivots;
  }

  if (pivots < m) {
    for (i_t i = 0; i < m; ++i) {
      i_t row = q[i];
      if (i >= pivots) { dependent_rows.push_back(row); }
    }
  }
  // Construct the permutation vector from it's inverse
  std::vector<i_t> p(N);
  for (i_t j = 0; j < N; ++j) {
    assert(pinv[j] != -1);
    assert(pinv[j] < N);
    p[pinv[j]] = j;
  }
  // The first pivots variables are in the basis. If pivots = m, B will be
  // non-singular
  for (i_t k = 0; k < pivots; ++k) {
    const i_t t = p[k];
    vstatus[t]  = variable_status_t::BASIC;  // variable x_j is basic
  }
  settings.log.debug(
    "%d basic variables set. Setting remaining %d to nonbasic\n", pivots, N - pivots);
  for (i_t k = pivots; k < N; ++k) {
    const i_t t = p[k];
    const i_t j = candidate_columns[t];
    if (problem.lower[j] > -INFINITY) {
      vstatus[t] = variable_status_t::NONBASIC_LOWER;  // variable x_j is nonbasic
                                                       // on lower bound
    } else if (problem.upper[j] < INFINITY) {
      vstatus[t] = variable_status_t::NONBASIC_UPPER;  // variable x_j is nonbasic
                                                       // on upper bound
    } else {
      vstatus[t] = variable_status_t::NONBASIC_FREE;  // variable x_j is nonbasic
                                                      // free at value 0
    }
  }

  return pivots;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template int initial_basis_selection<int, double>(
  const lp_problem_t<int, double>& problem,
  const simplex_solver_settings_t<int, double>& settings,
  const std::vector<int>& candidate_columns,
  double start_time,
  std::vector<variable_status_t>& vstatus,
  std::vector<int>& dependent_rows);

#endif

}  // namespace cuopt::linear_programming::dual_simplex
