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

#include <dual_simplex/basis_solves.hpp>

#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/right_looking_lu.hpp>
#include <dual_simplex/singletons.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <dual_simplex/triangle_solve.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t>
i_t reorder_basic_list(const std::vector<i_t>& q, std::vector<i_t>& basic_list)
{
  const i_t m                     = basic_list.size();
  std::vector<i_t> basic_list_old = basic_list;
  for (i_t k = 0; k < m; k++) {
    basic_list[k] = basic_list_old[q[k]];
  }
  return 0;
}

template <typename i_t>
void get_basis_from_vstatus(i_t m,
                            const std::vector<variable_status_t>& vstatus,
                            std::vector<i_t>& basis_list,
                            std::vector<i_t>& nonbasic_list,
                            std::vector<i_t>& superbasic_list)
{
  i_t n             = vstatus.size();
  i_t num_basic     = 0;
  i_t num_non_basic = 0;
  for (i_t j = 0; j < n; ++j) {
    if (vstatus[j] == variable_status_t::BASIC) {
      basis_list[num_basic++] = j;
      assert(num_basic <= m);
    } else if (vstatus[j] == variable_status_t::NONBASIC_LOWER ||
               vstatus[j] == variable_status_t::NONBASIC_UPPER ||
               vstatus[j] == variable_status_t::NONBASIC_FREE ||
               vstatus[j] == variable_status_t::NONBASIC_FIXED) {
      nonbasic_list.push_back(j);
      num_non_basic++;
      assert(num_non_basic <= n - m);
    } else if (vstatus[j] == variable_status_t::SUPERBASIC) {
      superbasic_list.push_back(j);
    }
  }
  i_t num_super_basic = superbasic_list.size();
  assert(num_basic == m);
}

namespace {

template <typename i_t, typename f_t>
void write_singleton_info(i_t m,
                          i_t col_singletons,
                          i_t row_singletons,
                          const csc_matrix_t<i_t, f_t>& B,
                          const std::vector<i_t>& row_perm,
                          const std::vector<i_t>& row_perm_inv,
                          const std::vector<i_t>& col_perm)
{
  FILE* file = fopen("singleton_debug.m", "w");
  if (file != NULL) {
    fprintf(file,
            "m = %d; col_singletons = %d; row_singletons = %d;\n",
            m,
            col_singletons,
            row_singletons);
    B.print_matrix(file);
    fprintf(file, "B = sparse(ijx(:,1), ijx(:,2), ijx(:,3), m, m);\n");
    fprintf(file, "row_perm = [\n");
    for (i_t i = 0; i < m; ++i) {
      fprintf(file, "%d\n", row_perm[i] + 1);
    }
    fprintf(file, "];\n");
    fprintf(file, "col_perm=[\n");
    for (i_t i = 0; i < m; ++i) {
      fprintf(file, "%d\n", col_perm[i] + 1);
    }
    fprintf(file, "];\n;");
    fprintf(file, "row_perm_inv = [\n");
    for (i_t i = 0; i < m; ++i) {
      fprintf(file, "%d\n", row_perm_inv[i] + 1);
    }
    fprintf(file, "];\n");
  }
  fclose(file);
}

template <typename i_t, typename f_t>
void write_factor_info(const char* filename,
                       i_t m,
                       i_t row_singletons,
                       i_t col_singletons,
                       const csc_matrix_t<i_t, f_t>& B,
                       const csc_matrix_t<i_t, f_t>& C,
                       const csc_matrix_t<i_t, f_t>& D,
                       const csc_matrix_t<i_t, f_t>& L,
                       const csc_matrix_t<i_t, f_t>& U,
                       const std::vector<i_t>& row_perm,
                       const std::vector<i_t>& col_perm)
{
  FILE* file = fopen(filename, "w");
  if (file != NULL) {
    fprintf(file,
            "m = %d; row_singletons = %d; col_singletons = %d;\n",
            m,
            row_singletons,
            col_singletons);
    B.print_matrix(file);
    fprintf(file, "B = sparse(ijx(:, 1), ijx(:, 2), ijx(:,3), m, m);\n");
    C.print_matrix(file);
    fprintf(file, "C = sparse(ijx(:,1), ijx(:,2), ijx(:,3), m, m);\n");
    D.print_matrix(file);
    fprintf(file, "D = sparse(ijx(:,1), ijx(:,2), ijx(:,3), m, m);\n");
    L.print_matrix(file);
    fprintf(file, "L = sparse(ijx(:,1), ijx(:, 2), ijx(:,3), m, m);\n");
    U.print_matrix(file);
    fprintf(file, "U = sparse(ijx(:,1), ijx(:,2), ijx(:,3), m, m);\n");
    fprintf(file, "row_perm_1 = [\n");
    for (i_t i = 0; i < m; i++) {
      fprintf(file, "%d;\n", row_perm[i] + 1);
    }
    fprintf(file, "];\n");
    fprintf(file, "col_perm = [\n");
    for (i_t j = 0; j < m; ++j) {
      fprintf(file, "%d;\n", col_perm[j] + 1);
    }
    fprintf(file, "];\n");
  }
  fclose(file);
}

template <typename i_t, typename f_t>
void write_basis_info(const csc_matrix_t<i_t, f_t>& B)
{
  FILE* file = fopen("basis.m", "w");
  if (file != NULL) {
    i_t m = B.m;
    fprintf(file, "m = %d;\n", m);
    B.print_matrix(file);
    fprintf(file, "B = sparse(ijx(:, 1), ijx(:, 2), ijx(:,3), m, m);\n");
  }
  fclose(file);
}

}  // namespace

template <typename i_t, typename f_t>
i_t factorize_basis(const csc_matrix_t<i_t, f_t>& A,
                    const simplex_solver_settings_t<i_t, f_t>& settings,
                    const std::vector<i_t>& basic_list,
                    csc_matrix_t<i_t, f_t>& L,
                    csc_matrix_t<i_t, f_t>& U,
                    std::vector<i_t>& p,
                    std::vector<i_t>& pinv,
                    std::vector<i_t>& q,
                    std::vector<i_t>& deficient,
                    std::vector<i_t>& slacks_needed)
{
  const i_t m              = basic_list.size();
  constexpr f_t medium_tol = 1e-12;

  const bool eliminate_singletons = settings.eliminate_singletons;
  constexpr bool verbose          = false;
  if (eliminate_singletons) {
    // TODO: We should see if we can find the singletons without explictly forming the matrix B
    f_t fact_start = tic();
    csc_matrix_t<i_t, f_t> B(A.m, A.m, 1);
    form_b(A, basic_list, B);
    std::vector<i_t> row_perm(m);
    std::vector<i_t> col_perm(m);
    i_t row_singletons;
    i_t col_singletons;
    find_singletons(B, row_singletons, row_perm, col_singletons, col_perm);
    std::vector<i_t> row_perm_inv(m);
    inverse_permutation(row_perm, row_perm_inv);

#ifdef PRINT_SINGLETONS
    printf("Singletons row %d col %d num %d\n",
           row_singletons,
           col_singletons,
           row_singletons + col_singletons);
#endif
    constexpr bool write_out = false;
    if (write_out) {
      write_singleton_info(m, col_singletons, row_singletons, B, row_perm, row_perm_inv, col_perm);
    }

    // P B Q = [ U_11  U_12 U_13 ]
    //         [       L_22      ]
    //         [       L_32  S   ]
    //
    // where U_11 is col_singletons x col_singletons and upper triangular
    //       L_22 is row_singletons x row_singletons and lower triangular
    //       U_12 is col_singletons x row_singletons
    //       U_13 is col_singletons x (m - col_singletons - row_singletons)
    //       L_32 is (m - col_singletons - row_singletons) x row_singletons
    //       S    is (m - col_singletons - row_singletons) x (m - col_singletons - row_singletons)

    // This permutation already provides a partial LU factorization since
    //
    // P B Q = [ U_11 U_12 U_13 ]  = [ I             ] [ U_11 U_12  U_13 ]
    //         [      L_22      ]    [    L_22       ] [      I          ]
    //         [      L_32  S   ]    [    L_32  L_33 ] [            U_33 ]
    // where L_33 * U_33 = S

    if ((col_singletons + row_singletons) > 0) {
      const i_t Bnz = B.col_start[m];
      L.reallocate(Bnz);
      U.reallocate(Bnz);
      i_t Lnz = 0;
      // Fill in L(:, 0:col_singletons-1) with I
      for (i_t k = 0; k < col_singletons; ++k) {
        L.col_start[k] = Lnz;
        L.i[Lnz]       = k;
        L.x[Lnz]       = 1.0;
        Lnz++;
        assert(Lnz <= Bnz);
      }

      i_t Unz = 0;
      // Fill in U(:, 0:col_singletons-1) with U_11
      for (i_t k = 0; k < col_singletons; ++k) {
        const i_t j         = col_perm[k];
        U.col_start[k]      = Unz;
        const i_t col_start = B.col_start[j];
        const i_t col_end   = B.col_start[j + 1];
        for (i_t p = col_start; p < col_end; ++p) {
          U.i[Unz] = row_perm_inv[B.i[p]];
          U.x[Unz] = B.x[p];
          Unz++;
          assert(Unz <= Bnz);
        }
      }
      if (col_singletons > 0) { U.col_start[col_singletons] = Unz; }

      // Ensure U(i, i) is at the end of column i for U_11
      for (i_t k = 0; k < col_singletons; ++k) {
        const i_t col_start      = U.col_start[k];
        const i_t col_before_end = U.col_start[k + 1] - 1;
        for (i_t p = col_start; p < col_before_end; ++p) {
          if (U.i[p] == k) {
            const f_t tmp_x     = U.x[p];
            U.i[p]              = U.i[col_before_end];
            U.x[p]              = U.x[col_before_end];
            U.i[col_before_end] = k;
            U.x[col_before_end] = tmp_x;
            break;
          }
        }
      }

      // Fill in L(:, col_singletons:col_singletons+row_singletons-1) with L_22 and L_32
      //     and U(:, col_singletons:col_singletons+row_singletons-1) with U_12 and I
      const i_t num_singletons = col_singletons + row_singletons;
      for (i_t k = col_singletons; k < num_singletons; ++k) {
        const i_t j         = col_perm[k];
        L.col_start[k]      = Lnz;
        U.col_start[k]      = Unz;
        const i_t col_start = B.col_start[j];
        const i_t col_end   = B.col_start[j + 1];
        for (i_t p = col_start; p < col_end; ++p) {
          const i_t i = row_perm_inv[B.i[p]];
          if (i >= col_singletons) {
            L.i[Lnz] = i;
            L.x[Lnz] = B.x[p];
            Lnz++;
            assert(Lnz <= Bnz);
          } else {
            U.i[Unz] = i;
            U.x[Unz] = B.x[p];
            Unz++;
            assert(Unz <= Bnz);
          }
        }
        // add in the identity in U
        U.i[Unz] = k;
        U.x[Unz] = 1.0;
        Unz++;
        assert(Unz <= Bnz);
      }
      L.col_start[num_singletons] = Lnz;

      // Ensure L(i, i) is at the beginning of column i for L_22 and L32
      for (i_t k = col_singletons; k < num_singletons; ++k) {
        const i_t col_start = L.col_start[k];
        const i_t col_end   = L.col_start[k + 1];
        if (L.i[col_start] == k) { continue; }
        bool found_diag = false;
        for (i_t p = col_start; p < col_end; ++p) {
          if (L.i[p] == k) {
            const i_t tmp_i = L.i[col_start];
            const f_t tmp_x = L.x[col_start];
            L.i[col_start]  = k;
            L.x[col_start]  = L.x[p];
            L.i[p]          = tmp_i;
            L.x[p]          = tmp_x;
            found_diag      = true;
            break;
          }
        }
        assert(found_diag);
      }

      // Compute how many nonzeros in B we have used so far so we know
      // how many nonzeros are in S
      const i_t Bnz_used = (Lnz - col_singletons) + (Unz - row_singletons);
      const i_t Snz_max  = Bnz - Bnz_used;
      const i_t Sdim     = m - col_singletons - row_singletons;
      i_t Srank          = 0;
      f_t actual_factor  = 0;
      if (Sdim > 0) {
        csc_matrix_t<i_t, f_t> S(Sdim, Sdim, Snz_max);

        // Build S
        i_t Snz = 0;
        for (i_t k = num_singletons; k < m; ++k) {
          S.col_start[k - num_singletons] = Snz;
          const i_t j                     = col_perm[k];
          const i_t col_start             = B.col_start[j];
          const i_t col_end               = B.col_start[j + 1];
          for (i_t p = col_start; p < col_end; ++p) {
            const i_t i = row_perm_inv[B.i[p]];
            if (i >= num_singletons) {
              const i_t row_i = i - num_singletons;
              assert(row_i < Sdim);
              S.i[Snz] = row_i;
              S.x[Snz] = B.x[p];
              Snz++;
              assert(Snz <= Snz_max);
            }
          }
        }
        S.col_start[Sdim] = Snz;  // Finalize S

        csc_matrix_t<i_t, f_t> SL(Sdim, Sdim, Snz);
        csc_matrix_t<i_t, f_t> SU(Sdim, Sdim, Snz);
        // Factorize S
        std::vector<i_t> S_perm_inv(Sdim);
        std::optional<std::vector<i_t>> empty = std::nullopt;
        f_t actual_factor_start               = tic();

        std::vector<i_t> S_col_perm(Sdim);
        std::vector<i_t> identity(Sdim);
        for (i_t h = 0; h < Sdim; ++h) {
          identity[h] = h;
        }
        Srank = right_looking_lu(
          S, settings.threshold_partial_pivoting_tol, identity, S_col_perm, SL, SU, S_perm_inv);
        if (Srank != Sdim) {
          // Get the rank deficient columns
          deficient.clear();
          deficient.resize(Sdim - Srank);
          for (i_t h = Srank; h < Sdim; ++h) {
            deficient[h - Srank] = col_perm[num_singletons + S_col_perm[h]];
          }
          // Get S_perm
          std::vector<i_t> S_perm(Sdim);
          inverse_permutation(S_perm_inv, S_perm);
          // Get the slacks needed
          slacks_needed.resize(Sdim - Srank);
          for (i_t h = Srank; h < Sdim; ++h) {
            slacks_needed[h - Srank] = row_perm[num_singletons + S_perm[h]];
          }

          return -1;
        }

        // Need to permute col_perm[k] according to q
        std::vector<i_t> col_perm_sav(m - num_singletons);
        i_t q_j = 0;
        for (i_t h = num_singletons; h < m; ++h) {
          col_perm_sav[q_j] = col_perm[h];
          q_j++;
        }
        q_j = 0;
        for (i_t h = num_singletons; h < m; ++h) {
          col_perm[h] = col_perm_sav[S_col_perm[q_j]];
          q_j++;
        }

        std::vector<i_t> S_perm(m);
        inverse_permutation(S_perm_inv, S_perm);
        actual_factor = toc(actual_factor_start);

        // Permute the rows of L_32 according to S_perm_inv
        for (i_t k = col_singletons; k < num_singletons; ++k) {
          const i_t col_start = L.col_start[k];
          const i_t col_end   = L.col_start[k + 1];
          for (i_t p = col_start; p < col_end; ++p) {
            const i_t i = L.i[p];
            if (i >= num_singletons) {
              const i_t new_i = num_singletons + S_perm_inv[i - num_singletons];
              L.i[p]          = new_i;
            }
          }
        }

        const i_t SLnz    = SL.col_start[Sdim];
        const i_t Lnz_max = Lnz + SLnz;
        if (Lnz_max > Bnz) { L.reallocate(Lnz_max); }

        // Fill in L(:, num_singletons:m-1) with L_33
        for (i_t k = num_singletons; k < m; ++k) {
          L.col_start[k]      = Lnz;
          const i_t j         = k - num_singletons;
          const i_t col_start = SL.col_start[j];
          const i_t col_end   = SL.col_start[j + 1];
          for (i_t p = col_start; p < col_end; ++p) {
            const i_t i = num_singletons + SL.i[p];
            L.i[Lnz]    = i;
            L.x[Lnz]    = SL.x[p];
            Lnz++;
            assert(Lnz <= Lnz_max);
          }
        }
        assert(Lnz == Lnz_max);
        L.col_start[m] = Lnz;  // Finalize L

        const i_t SUnz    = SU.col_start[Sdim];
        const i_t Unz_max = Unz + SUnz + (Bnz - Bnz_used);
        if (Unz_max > Bnz) { U.reallocate(Unz_max); }

        // Fill in U(:, num_singletons:m-1) with U_13 and U_33
        for (i_t k = num_singletons; k < m; ++k) {
          U.col_start[k] = Unz;

          // U_13
          const i_t j           = col_perm[k];
          const i_t B_col_start = B.col_start[j];
          const i_t B_col_end   = B.col_start[j + 1];
          for (i_t p = B_col_start; p < B_col_end; ++p) {
            const i_t i = row_perm_inv[B.i[p]];
            if (i < num_singletons) {
              U.i[Unz] = i;
              U.x[Unz] = B.x[p];
              Unz++;
              assert(Unz <= Unz_max);
            }
          }

          // U_33
          const i_t l           = k - num_singletons;
          const i_t U_col_start = SU.col_start[l];
          const i_t U_col_end   = SU.col_start[l + 1];
          for (i_t p = U_col_start; p < U_col_end; ++p) {
            const i_t i = num_singletons + SU.i[p];
            U.i[Unz]    = i;
            U.x[Unz]    = SU.x[p];
            Unz++;
            assert(Unz <= Unz_max);
          }
        }
        assert(Unz <= Unz_max);
        U.col_start[m] = Unz;  // Finalize U

        std::vector<i_t> last_perm(Sdim);
        for (i_t k = 0; k < Sdim; ++k) {
          last_perm[k] = row_perm[num_singletons + k];
        }

        // Fix up row permutations
        for (i_t k = 0; k < Sdim; ++k) {
          row_perm[num_singletons + k] = last_perm[S_perm[k]];
        }
        inverse_permutation(row_perm, row_perm_inv);
      } else {
        L.col_start[m] = Lnz;  // Finalize L
        U.col_start[m] = Unz;  // Finalize U
      }

      constexpr bool check_singleton = false;
      if (check_singleton) {
        // Check the diagonal entries of L
        for (i_t k = 0; k < m; ++k) {
          const i_t col_start = L.col_start[k];
          if (L.i[col_start] != k) {
            printf("col %d Li %d col singletons %d num singletons %d\n",
                   k,
                   L.i[col_start],
                   col_singletons,
                   num_singletons);
          }
          assert(L.i[col_start] == k);
        }

        // Check the diagonal entries of U
        for (i_t k = 0; k < m; ++k) {
          const i_t col_end = U.col_start[k + 1] - 1;
          assert(U.i[col_end] == k);
        }

        // Check L*U = B(row_perm, col_perm)
        csc_matrix_t<i_t, f_t> C(m, m, 1);
        multiply(L, U, C);

        csc_matrix_t<i_t, f_t> D(m, m, 1);
        B.permute_rows_and_cols(row_perm_inv, col_perm, D);

        csc_matrix_t<i_t, f_t> E(m, m, 1);
        add(C, D, 1.0, -1.0, E);

        write_factor_info("singleton_factor.m",
                          m,
                          row_singletons,
                          col_singletons,
                          B,
                          C,
                          D,
                          L,
                          U,
                          row_perm,
                          col_perm);

        const f_t norm_diff = E.norm1();
        printf(
          "|| L*U - B(row_perm, col_perm) || %e. m %d row singletons %d col singletons %d Sdim "
          "%d\n",
          norm_diff,
          m,
          row_singletons,
          col_singletons,
          Sdim);
        assert(norm_diff < 1e-3);
      }
      p    = row_perm;
      pinv = row_perm_inv;
      q    = col_perm;
      assert(p.size() == m);
      assert(pinv.size() == m);
      assert(q.size() == m);
      assert(L.m == m);
      assert(L.n == m);
      assert(U.m == m);
      assert(U.n == m);
      return Srank + num_singletons;
    }
  }

  i_t rank                   = -1;
  constexpr bool write_basis = false;

  if (write_basis) {
    csc_matrix_t<i_t, f_t> B(m, m, 1);
    form_b(A, basic_list, B);
    write_basis_info(B);
  }
  q.resize(m);
  f_t fact_start = tic();
  right_looking_lu(A, medium_tol, basic_list, q, L, U, pinv);
  if (verbose) {
    printf("Right Lnz+Unz %d t %.3f\n", L.col_start[m] + U.col_start[m], toc(fact_start));
  }
  inverse_permutation(pinv, p);
  rank                    = m;
  constexpr bool check_lu = false;
  if (check_lu) {
    csc_matrix_t<i_t, f_t> C(m, m, 1);
    multiply(L, U, C);

    csc_matrix_t<i_t, f_t> B(m, m, 1);
    form_b(A, basic_list, B);
    csc_matrix_t<i_t, f_t> D(m, m, 1);
    B.permute_rows_and_cols(pinv, q, D);

    csc_matrix_t<i_t, f_t> E(m, m, 1);
    add(C, D, 1.0, -1.0, E);

    write_factor_info("rightlu_factor.m", m, 0, 0, B, C, D, L, U, p, q);

    const f_t norm_diff = E.norm1();
    printf("|| L*U - B(row_perm, col_perm) || %e. m %d\n", norm_diff, m);
    assert(norm_diff < 1e-3);
  }

  return rank;
}

template <typename i_t, typename f_t>
i_t basis_repair(const csc_matrix_t<i_t, f_t>& A,
                 const simplex_solver_settings_t<i_t, f_t>& settings,
                 const std::vector<i_t>& deficient,
                 const std::vector<i_t>& slacks_needed,
                 std::vector<i_t>& basis_list,
                 std::vector<i_t>& nonbasic_list,
                 std::vector<variable_status_t>& vstatus)
{
  const i_t m = A.m;
  const i_t n = A.n;
  assert(basis_list.size() == m);
  assert(nonbasic_list.size() == n - m);

  // Create slack_map
  std::vector<i_t> slack_map(m);  // slack_map[i] = j if column j is e_i
  i_t slacks_found = 0;
  for (i_t j = n - 1; j >= n - m; j--) {
    const i_t col_start = A.col_start[j];
    const i_t col_end   = A.col_start[j + 1];
    const i_t col_nz    = col_end - col_start;
    if (col_nz == 1 && std::abs(A.x[col_start]) == 1.0) {
      const i_t i  = A.i[col_start];
      slack_map[i] = j;
      slacks_found++;
    }
  }

  assert(slacks_found == m);

  // Create nonbasic_map
  std::vector<i_t> nonbasic_map(n);  // nonbasic_map[j] = p if nonbasic[p] = j, -1 if j is basic
  for (i_t k = 0; k < m; ++k) {
    nonbasic_map[basis_list[k]] = -1;
  }
  for (i_t k = 0; k < n - m; ++k) {
    nonbasic_map[nonbasic_list[k]] = k;
  }

  const i_t columns_to_replace = deficient.size();
  for (i_t k = 0; k < columns_to_replace; ++k) {
    const i_t bad_j          = basis_list[deficient[k]];
    const i_t replace_i      = slacks_needed[k];
    const i_t replace_j      = slack_map[replace_i];
    basis_list[deficient[k]] = replace_j;
    assert(nonbasic_map[replace_j] != -1);
    nonbasic_list[nonbasic_map[replace_j]] = bad_j;
    vstatus[replace_j]                     = variable_status_t::BASIC;
    // This is the main issue. What value should bad_j take on.
    vstatus[bad_j] = variable_status_t::NONBASIC_FREE;
  }

  return 0;
}

template <typename i_t, typename f_t>
i_t form_b(const csc_matrix_t<i_t, f_t>& A,
           const std::vector<i_t>& basic_list,
           csc_matrix_t<i_t, f_t>& B)
{
  const i_t m = A.m;
  i_t Bnz     = 0;
  for (i_t k = 0; k < m; ++k) {
    const i_t j         = basic_list[k];
    const i_t col_start = A.col_start[j];
    const i_t col_end   = A.col_start[j + 1];
    Bnz += (col_end - col_start);
  }
  B.reallocate(Bnz);
  const i_t Bnz_check = Bnz;
  Bnz                 = 0;
  for (i_t k = 0; k < m; ++k) {
    B.col_start[k]      = Bnz;
    const i_t j         = basic_list[k];
    const i_t col_start = A.col_start[j];
    const i_t col_end   = A.col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      B.i[Bnz] = A.i[p];
      B.x[Bnz] = A.x[p];
      Bnz++;
    }
  }
  B.col_start[m] = Bnz;
  assert(Bnz_check == Bnz);
  return 0;
}

// y = B*x = sum_{j in basis} A(:, j) * x(k)
template <typename i_t, typename f_t>
i_t b_multiply(const lp_problem_t<i_t, f_t>& lp,
               const std::vector<i_t>& basic_list,
               const std::vector<f_t>& x,
               std::vector<f_t>& y)
{
  const i_t m = lp.num_rows;
  std::fill(y.begin(), y.end(), 0.0);
  for (i_t k = 0; k < m; ++k) {
    const i_t j         = basic_list[k];
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    const f_t xk        = x[k];
    for (i_t p = col_start; p < col_end; ++p) {
      y[lp.A.i[p]] += xk * lp.A.x[p];
    }
  }
  return 0;
}

// y = B'*x. y_j = A(:, j)'*x for all j
template <typename i_t, typename f_t>
i_t b_transpose_multiply(const lp_problem_t<i_t, f_t>& lp,
                         const std::vector<i_t>& basic_list,
                         const std::vector<f_t>& x,
                         std::vector<f_t>& y)
{
  const i_t m = lp.num_rows;
  std::fill(y.begin(), y.end(), 0.0);
  for (i_t k = 0; k < m; ++k) {
    const i_t j         = basic_list[k];
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    f_t dot             = 0;
    for (i_t p = col_start; p < col_end; ++p) {
      dot += x[lp.A.i[p]] * lp.A.x[p];
    }
    y[k] = dot;
  }
  return 0;
}

// Solves the system B'*y = c, given L*U = B(p, :)
template <typename i_t, typename f_t>
i_t b_transpose_solve(const csc_matrix_t<i_t, f_t>& L,
                      const csc_matrix_t<i_t, f_t>& U,
                      const std::vector<i_t>& p,
                      const std::vector<f_t>& rhs,
                      std::vector<f_t>& solution)
{
  // P*B = L*U
  // B'*P' = U'*L'
  // B'*y = c
  // y = P'*w
  // B'*P'*w = U'*L'*w = c
  // U'*r = c
  // L'*w = r

  // Solve for r such that U'*r = c
  std::vector<f_t> r = rhs;
  upper_triangular_transpose_solve(U, r);
#ifdef BASIS_DEBUG
  // err = norm(U'*r - c, inf)
  std::vector<f_t> residual = rhs;
  matrix_transpose_vector_multiply(U, 1.0, r, -1.0, residual);
  f_t err = vector_norm_inf(residual);
  assert(err < 1e-12);
  printf("|| U'*r - c || %e\n", err);
  std::vector<f_t> residual2 = r;
#endif

  // Solve for w such that L'*w = r
  lower_triangular_transpose_solve(L, r);
#ifdef BASIS_DEBUG
  // err2 = norm(L'*w -r, inf)
  matrix_transpose_vector_multiply(L, 1.0, r, -1.0, residual2);
  f_t err2 = vector_norm_inf(residual2);
  printf("|| L'*w - r|| %e\n", err2);
  assert(err2 < 1e-9);
#endif

  // y = P'*w
  inverse_permute_vector(p, r, solution);

  return 0;
}

// Solves the system B*x = b, given L*U = B(p, :)
template <typename i_t, typename f_t>
i_t b_solve(const csc_matrix_t<i_t, f_t>& L,
            const csc_matrix_t<i_t, f_t>& U,
            const std::vector<i_t>& p,
            const std::vector<f_t>& rhs,
            std::vector<f_t>& solution)
{
  const i_t m = L.m;
  assert(p.size() == m);
  assert(rhs.size() == m);
  assert(solution.size() == m);
  // P*B = L*U
  // B*x = b
  // P*B*x = P*b = b'
  permute_vector(p, rhs, solution);

  // Solve for v such that L*v = b'
  lower_triangular_solve(L, solution);

#ifdef BASIS_DEBUG
  std::vector<f_t> residual1(m);
  permute_vector(p, rhs, residual1);
  matrix_vector_multiply(L, 1.0, solution, -1.0, residual1);
  const f_t err = vector_norm_inf(residual1);
  printf("|| L*v - bprime ||_inf %e\n", err);
  std::vector<f_t> residual2 = solution;
  assert(err < 1e-12);
#endif

  // Solve for x such that U*x = v
  upper_triangular_solve(U, solution);
#ifdef BASIS_DEBUG
  matrix_vector_multiply(U, 1.0, solution, -1.0, residual2);
  const f_t err2 = vector_norm_inf(residual2);
  printf("|| U*x - v ||_inf %e\n", err2);
  assert(err2 < 1e-10);
#endif

  return 0;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template int reorder_basic_list<int>(const std::vector<int>& q, std::vector<int>& basic_list);

template void get_basis_from_vstatus<int>(int m,
                                          const std::vector<variable_status_t>& vstatus,
                                          std::vector<int>& basis_list,
                                          std::vector<int>& nonbasic_list,
                                          std::vector<int>& superbasic_list);

template int factorize_basis<int>(const csc_matrix_t<int, double>& A,
                                  const simplex_solver_settings_t<int, double>& settings,
                                  const std::vector<int>& basis_list,
                                  csc_matrix_t<int, double>& L,
                                  csc_matrix_t<int, double>& U,
                                  std::vector<int>& p,
                                  std::vector<int>& pinv,
                                  std::vector<int>& q,
                                  std::vector<int>& deficient,
                                  std::vector<int>& slacks_needed);

template int basis_repair<int, double>(const csc_matrix_t<int, double>& A,
                                       const simplex_solver_settings_t<int, double>& settings,
                                       const std::vector<int>& deficient,
                                       const std::vector<int>& slacks_needed,
                                       std::vector<int>& basis_list,
                                       std::vector<int>& nonbasic_list,
                                       std::vector<variable_status_t>& vstatus);

template int form_b<int, double>(const csc_matrix_t<int, double>& A,
                                 const std::vector<int>& basic_list,
                                 csc_matrix_t<int, double>& B);

template int b_multiply<int, double>(const lp_problem_t<int, double>& lp,
                                     const std::vector<int>& basic_list,
                                     const std::vector<double>& x,
                                     std::vector<double>& y);

template int b_transpose_multiply<int, double>(const lp_problem_t<int, double>& lp,
                                               const std::vector<int>& basic_list,
                                               const std::vector<double>& x,
                                               std::vector<double>& y);

// Solves B'*y = c, given L*U = B(p, :). This version supports a dense vector
template int b_transpose_solve<int, double>(const csc_matrix_t<int, double>& L,
                                            const csc_matrix_t<int, double>& U,
                                            const std::vector<int>& p,
                                            const std::vector<double>& rhs,
                                            std::vector<double>& solution);

// Solves the system B*x = b, given L*U = B(p, :)
template int b_solve<int, double>(const csc_matrix_t<int, double>& L,
                                  const csc_matrix_t<int, double>& U,
                                  const std::vector<int>& p,
                                  const std::vector<double>& rhs,
                                  std::vector<double>& solution);
#endif

}  // namespace cuopt::linear_programming::dual_simplex
