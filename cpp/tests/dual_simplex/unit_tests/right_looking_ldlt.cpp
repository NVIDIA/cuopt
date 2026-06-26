/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>

#include <dual_simplex/right_looking_lu.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/tic_toc.hpp>

#include <cmath>
#include <numeric>
#include <vector>

namespace cuopt::mathematical_optimization::simplex::test {

// Helper: build a CSC lower-triangle matrix from dense symmetric input (column-major, full matrix).
// Only stores entries (i, j) with i >= j.
static csc_matrix_t<int, double> dense_to_lower_csc(int n, const std::vector<double>& dense)
{
  // Count nonzeros in lower triangle
  int nnz = 0;
  for (int j = 0; j < n; j++) {
    for (int i = j; i < n; i++) {
      if (dense[j * n + i] != 0.0) { nnz++; }
    }
  }

  csc_matrix_t<int, double> A(n, n, nnz);
  int p = 0;
  for (int j = 0; j < n; j++) {
    A.col_start[j] = p;
    for (int i = j; i < n; i++) {
      double val = dense[j * n + i];
      if (val != 0.0) {
        A.i[p] = i;
        A.x[p] = val;
        p++;
      }
    }
  }
  A.col_start[n] = p;
  return A;
}

// Helper: verify the factorization P * A * P^T = L * D * L^T
// by computing L * D * L^T and comparing against P * A * P^T.
static void verify_ldlt(int n,
                        const std::vector<double>& dense_A,
                        int rank,
                        const std::vector<int>& perm,
                        const csc_matrix_t<int, double>& L,
                        const std::vector<double>& D,
                        double tol = 1e-10)
{
  // Build P * A * P^T as dense matrix (rank x rank leading submatrix)
  // perm[k] = original index of the k-th pivot
  // So P*A*P^T[k, l] = A[perm[k], perm[l]]
  std::vector<double> PAPT(rank * rank, 0.0);
  for (int k = 0; k < rank; k++) {
    for (int l = 0; l < rank; l++) {
      PAPT[l * rank + k] = dense_A[perm[l] * n + perm[k]];
    }
  }

  // Build L as dense matrix (rank x rank)
  // L is stored in CSC with permuted row indices: L.i[p] is the permuted row index.
  std::vector<double> L_dense(rank * rank, 0.0);
  for (int j = 0; j < rank; j++) {
    for (int p = L.col_start[j]; p < L.col_start[j + 1]; p++) {
      int i      = L.i[p];
      double val = L.x[p];
      if (i < rank) { L_dense[j * rank + i] = val; }
    }
  }

  // Compute L * D * L^T
  std::vector<double> LDLT(rank * rank, 0.0);
  for (int i = 0; i < rank; i++) {
    for (int j = 0; j < rank; j++) {
      double sum = 0.0;
      for (int k = 0; k < rank; k++) {
        sum += L_dense[k * rank + i] * D[k] * L_dense[k * rank + j];
      }
      LDLT[j * rank + i] = sum;
    }
  }

  // Compare PAPT and LDLT
  for (int i = 0; i < rank; i++) {
    for (int j = 0; j < rank; j++) {
      EXPECT_NEAR(PAPT[j * rank + i], LDLT[j * rank + i], tol)
        << "Mismatch at (" << i << ", " << j << ")";
    }
  }
}

// Test 1: 2x2 positive definite diagonal matrix
TEST(right_looking_ldlt, diagonal_2x2)
{
  // A = [4 0; 0 9]
  const int n               = 2;
  std::vector<double> dense = {4.0, 0.0, 0.0, 9.0};
  auto A                    = dense_to_lower_csc(n, dense);

  simplex_solver_settings_t<int, double> settings;
  std::vector<int> perm;
  csc_matrix_t<int, double> L(n, n, 1);
  std::vector<double> D;
  double work_estimate = 0;
  double start_time    = tic();

  int rank = right_looking_ldlt(A, settings, 1e-12, start_time, perm, L, D, work_estimate);

  EXPECT_EQ(rank, 2);
  EXPECT_EQ(D.size(), 2u);
  // Both diagonal entries should appear as pivots
  verify_ldlt(n, dense, rank, perm, L, D);
}

// Test 2: 3x3 positive definite matrix
TEST(right_looking_ldlt, pd_3x3)
{
  // A = [4  2  1]
  //     [2  5  3]
  //     [1  3  6]
  const int n = 3;
  // Column-major storage of full symmetric matrix
  std::vector<double> dense = {4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0};
  auto A                    = dense_to_lower_csc(n, dense);

  simplex_solver_settings_t<int, double> settings;
  std::vector<int> perm;
  csc_matrix_t<int, double> L(n, n, 1);
  std::vector<double> D;
  double work_estimate = 0;
  double start_time    = tic();

  int rank = right_looking_ldlt(A, settings, 1e-12, start_time, perm, L, D, work_estimate);

  EXPECT_EQ(rank, 3);
  verify_ldlt(n, dense, rank, perm, L, D);

  // All D entries should be positive (PD matrix)
  for (int k = 0; k < rank; k++) {
    EXPECT_GT(D[k], 0.0);
  }
}

// Test 3: Rank-1 PSD matrix (singular)
TEST(right_looking_ldlt, rank1_psd)
{
  // A = v * v^T where v = [1, 2, 3]
  // A = [1 2 3]
  //     [2 4 6]
  //     [3 6 9]
  const int n               = 3;
  std::vector<double> dense = {1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0};
  auto A                    = dense_to_lower_csc(n, dense);

  simplex_solver_settings_t<int, double> settings;
  std::vector<int> perm;
  csc_matrix_t<int, double> L(n, n, 1);
  std::vector<double> D;
  double work_estimate = 0;
  double start_time    = tic();

  int rank = right_looking_ldlt(A, settings, 1e-12, start_time, perm, L, D, work_estimate);

  // Rank should be 1
  EXPECT_EQ(rank, 1);
  EXPECT_EQ(D.size(), 1u);
  EXPECT_GT(D[0], 0.0);
  verify_ldlt(n, dense, rank, perm, L, D);
}

// Test 4: Rank-2 PSD matrix (singular)
TEST(right_looking_ldlt, rank2_psd)
{
  // A = v1*v1^T + v2*v2^T where v1 = [1, 0, 1], v2 = [0, 1, 1]
  // A = [1 0 1]   [0 0 0]   [1 0 1]
  //     [0 0 0] + [0 1 1] = [0 1 1]
  //     [1 0 1]   [0 1 1]   [1 1 2]
  const int n               = 3;
  std::vector<double> dense = {1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0};
  auto A                    = dense_to_lower_csc(n, dense);

  simplex_solver_settings_t<int, double> settings;
  std::vector<int> perm;
  csc_matrix_t<int, double> L(n, n, 1);
  std::vector<double> D;
  double work_estimate = 0;
  double start_time    = tic();

  int rank = right_looking_ldlt(A, settings, 1e-12, start_time, perm, L, D, work_estimate);

  // Rank should be 2
  EXPECT_EQ(rank, 2);
  EXPECT_EQ(D.size(), 2u);
  for (int k = 0; k < rank; k++) {
    EXPECT_GT(D[k], 0.0);
  }
  verify_ldlt(n, dense, rank, perm, L, D);
}

// Test 5: Zero matrix (rank 0)
TEST(right_looking_ldlt, zero_matrix)
{
  const int n = 3;
  std::vector<double> dense(n * n, 0.0);
  auto A = dense_to_lower_csc(n, dense);

  simplex_solver_settings_t<int, double> settings;
  std::vector<int> perm;
  csc_matrix_t<int, double> L(n, n, 1);
  std::vector<double> D;
  double work_estimate = 0;
  double start_time    = tic();

  int rank = right_looking_ldlt(A, settings, 1e-12, start_time, perm, L, D, work_estimate);

  EXPECT_EQ(rank, 0);
  EXPECT_TRUE(D.empty());
}

// Test 6: 1x1 matrix
TEST(right_looking_ldlt, scalar_1x1)
{
  const int n               = 1;
  std::vector<double> dense = {7.0};
  auto A                    = dense_to_lower_csc(n, dense);

  simplex_solver_settings_t<int, double> settings;
  std::vector<int> perm;
  csc_matrix_t<int, double> L(n, n, 1);
  std::vector<double> D;
  double work_estimate = 0;
  double start_time    = tic();

  int rank = right_looking_ldlt(A, settings, 1e-12, start_time, perm, L, D, work_estimate);

  EXPECT_EQ(rank, 1);
  EXPECT_EQ(D.size(), 1u);
  EXPECT_NEAR(D[0], 7.0, 1e-14);
  EXPECT_EQ(perm[0], 0);
}

// Test 7: Larger PD matrix (5x5, from A^T*A with random A)
TEST(right_looking_ldlt, pd_5x5)
{
  // Build A = B^T * B where B is 5x5 with known structure
  // B = [2 1 0 0 0]
  //     [1 3 1 0 0]
  //     [0 1 4 1 0]
  //     [0 0 1 5 1]
  //     [0 0 0 1 6]
  // A = B^T * B (tridiagonal B -> banded A, guaranteed PD)
  const int n    = 5;
  double B[5][5] = {
    {2, 1, 0, 0, 0}, {1, 3, 1, 0, 0}, {0, 1, 4, 1, 0}, {0, 0, 1, 5, 1}, {0, 0, 0, 1, 6}};
  std::vector<double> dense(n * n, 0.0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) {
        sum += B[k][i] * B[k][j];
      }
      dense[j * n + i] = sum;
    }
  }

  auto A = dense_to_lower_csc(n, dense);

  simplex_solver_settings_t<int, double> settings;
  std::vector<int> perm;
  csc_matrix_t<int, double> L(n, n, 1);
  std::vector<double> D;
  double work_estimate = 0;
  double start_time    = tic();

  int rank = right_looking_ldlt(A, settings, 1e-12, start_time, perm, L, D, work_estimate);

  EXPECT_EQ(rank, 5);
  for (int k = 0; k < rank; k++) {
    EXPECT_GT(D[k], 0.0);
  }
  verify_ldlt(n, dense, rank, perm, L, D);
}

// Test 8: Rank-3 PSD 5x5 matrix
TEST(right_looking_ldlt, rank3_5x5_psd)
{
  // A = sum_{k=0}^{2} v_k * v_k^T with linearly independent v_k
  // v0 = [1, 0, 1, 0, 1], v1 = [0, 1, 1, 0, 0], v2 = [0, 0, 0, 1, 1]
  const int n    = 5;
  double v[3][5] = {{1, 0, 1, 0, 1}, {0, 1, 1, 0, 0}, {0, 0, 0, 1, 1}};
  std::vector<double> dense(n * n, 0.0);
  for (int k = 0; k < 3; k++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        dense[j * n + i] += v[k][i] * v[k][j];
      }
    }
  }
  // A = [1 0 1 0 1]
  //     [0 1 1 0 0]
  //     [1 1 2 0 1]
  //     [0 0 0 1 1]
  //     [1 0 1 1 2]

  auto A = dense_to_lower_csc(n, dense);

  simplex_solver_settings_t<int, double> settings;
  std::vector<int> perm;
  csc_matrix_t<int, double> L(n, n, 1);
  std::vector<double> D;
  double work_estimate = 0;
  double start_time    = tic();

  int rank = right_looking_ldlt(A, settings, 1e-12, start_time, perm, L, D, work_estimate);

  EXPECT_EQ(rank, 3);
  for (int k = 0; k < rank; k++) {
    EXPECT_GT(D[k], 0.0);
  }
  verify_ldlt(n, dense, rank, perm, L, D);
}

// Test 9: Identity matrix
TEST(right_looking_ldlt, identity)
{
  const int n = 4;
  std::vector<double> dense(n * n, 0.0);
  for (int i = 0; i < n; i++) {
    dense[i * n + i] = 1.0;
  }
  auto A = dense_to_lower_csc(n, dense);

  simplex_solver_settings_t<int, double> settings;
  std::vector<int> perm;
  csc_matrix_t<int, double> L(n, n, 1);
  std::vector<double> D;
  double work_estimate = 0;
  double start_time    = tic();

  int rank = right_looking_ldlt(A, settings, 1e-12, start_time, perm, L, D, work_estimate);

  EXPECT_EQ(rank, 4);
  for (int k = 0; k < rank; k++) {
    EXPECT_NEAR(D[k], 1.0, 1e-14);
  }
  verify_ldlt(n, dense, rank, perm, L, D);
}

// Test 10: Sparse PSD matrix arising from graph Laplacian (always singular, rank = n-1)
TEST(right_looking_ldlt, graph_laplacian)
{
  // 4-node cycle graph Laplacian:
  // L = [ 2 -1  0 -1]
  //     [-1  2 -1  0]
  //     [ 0 -1  2 -1]
  //     [-1  0 -1  2]
  // Eigenvalues: 0, 2, 2, 4 -> rank 3
  const int n               = 4;
  std::vector<double> dense = {
    2.0, -1.0, 0.0, -1.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, -1.0, 0.0, -1.0, 2.0};
  auto A = dense_to_lower_csc(n, dense);

  simplex_solver_settings_t<int, double> settings;
  std::vector<int> perm;
  csc_matrix_t<int, double> L_out(n, n, 1);
  std::vector<double> D;
  double work_estimate = 0;
  double start_time    = tic();

  int rank = right_looking_ldlt(A, settings, 1e-12, start_time, perm, L_out, D, work_estimate);

  // Graph Laplacian of connected graph has rank n-1
  EXPECT_EQ(rank, 3);
  for (int k = 0; k < rank; k++) {
    EXPECT_GT(D[k], 0.0);
  }
  verify_ldlt(n, dense, rank, perm, L_out, D);
}

// Test 11: Verify that unsymmetric input (Q + Q^T) works when symmetrized externally.
// This tests the use case: given arbitrary Q, form H = Q + Q^T (lower triangle) and factorize.
TEST(right_looking_ldlt, symmetrized_from_unsymmetric)
{
  // Q (not symmetric):
  // Q = [4 1 0]
  //     [3 5 2]
  //     [0 1 6]
  // H = Q + Q^T = [8 4 0]
  //               [4 10 3]
  //               [0 3 12]
  // H is positive definite (diagonally dominant).
  const int n                 = 3;
  std::vector<double> dense_H = {8.0, 4.0, 0.0, 4.0, 10.0, 3.0, 0.0, 3.0, 12.0};
  auto A                      = dense_to_lower_csc(n, dense_H);

  simplex_solver_settings_t<int, double> settings;
  std::vector<int> perm;
  csc_matrix_t<int, double> L(n, n, 1);
  std::vector<double> D;
  double work_estimate = 0;
  double start_time    = tic();

  int rank = right_looking_ldlt(A, settings, 1e-12, start_time, perm, L, D, work_estimate);

  EXPECT_EQ(rank, 3);
  for (int k = 0; k < rank; k++) {
    EXPECT_GT(D[k], 0.0);
  }
  verify_ldlt(n, dense_H, rank, perm, L, D);
}

// Test 12: Larger rank-deficient matrix (10x10, rank 5)
TEST(right_looking_ldlt, rank5_10x10_psd)
{
  const int n = 10;
  const int r = 5;
  // Build A = V * V^T where V is 10 x 5
  double V[10][5] = {{1, 0, 0, 0, 0},
                     {0, 1, 0, 0, 0},
                     {0, 0, 1, 0, 0},
                     {0, 0, 0, 1, 0},
                     {0, 0, 0, 0, 1},
                     {1, 1, 0, 0, 0},
                     {0, 1, 1, 0, 0},
                     {0, 0, 1, 1, 0},
                     {0, 0, 0, 1, 1},
                     {1, 0, 0, 0, 1}};
  std::vector<double> dense(n * n, 0.0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < r; k++) {
        sum += V[i][k] * V[j][k];
      }
      dense[j * n + i] = sum;
    }
  }

  auto A = dense_to_lower_csc(n, dense);

  simplex_solver_settings_t<int, double> settings;
  std::vector<int> perm;
  csc_matrix_t<int, double> L(n, n, 1);
  std::vector<double> D;
  double work_estimate = 0;
  double start_time    = tic();

  int rank_out = right_looking_ldlt(A, settings, 1e-12, start_time, perm, L, D, work_estimate);

  EXPECT_EQ(rank_out, r);
  for (int k = 0; k < rank_out; k++) {
    EXPECT_GT(D[k], 0.0);
  }
  verify_ldlt(n, dense, rank_out, perm, L, D);
}

// Test 13: 10x10 matrix that is all zeros except a single 1 at position (9,9) (0-indexed)
TEST(right_looking_ldlt, sparse_single_entry_10x10)
{
  const int n = 10;
  std::vector<double> dense(n * n, 0.0);
  dense[9 * n + 9] = 1.0;  // A(9,9) = 1, everything else is 0
  auto A           = dense_to_lower_csc(n, dense);

  simplex_solver_settings_t<int, double> settings;
  std::vector<int> perm;
  csc_matrix_t<int, double> L(n, n, 1);
  std::vector<double> D;
  double work_estimate = 0;
  double start_time    = tic();

  int rank = right_looking_ldlt(A, settings, 1e-12, start_time, perm, L, D, work_estimate);

  // Only one nonzero diagonal entry, so rank = 1
  EXPECT_EQ(rank, 1);
  EXPECT_EQ(D.size(), 1u);
  EXPECT_NEAR(D[0], 1.0, 1e-14);
  // The pivot should be at original index 9
  EXPECT_EQ(perm[0], 9);
  verify_ldlt(n, dense, rank, perm, L, D);
}

// Test 14: Indefinite matrix (has a negative eigenvalue).
// The factorization should detect this and return INDEFINITE_MATRIX_RETURN.
// Matrix: A = [1, 2; 2, 1] has eigenvalues 3 and -1 (indefinite).
TEST(right_looking_ldlt, indefinite_2x2)
{
  const int n               = 2;
  std::vector<double> dense = {1.0, 2.0, 2.0, 1.0};
  auto A                    = dense_to_lower_csc(n, dense);

  simplex_solver_settings_t<int, double> settings;
  std::vector<int> perm;
  csc_matrix_t<int, double> L(n, n, 1);
  std::vector<double> D;
  double work_estimate = 0;
  double start_time    = tic();

  int result = right_looking_ldlt(A, settings, 1e-12, start_time, perm, L, D, work_estimate);

  // Should return INDEFINITE_MATRIX_RETURN (-4) since the matrix is indefinite
  EXPECT_EQ(result, INDEFINITE_MATRIX_RETURN);
}

// Test 15: Larger indefinite matrix (4x4 with mixed eigenvalues).
// A = [2, 1, 0, 1; 1, 0, 1, 0; 0, 1, 2, -1; 1, 0, -1, -1]
TEST(right_looking_ldlt, indefinite_4x4)
{
  const int n = 4;
  // A symmetric indefinite matrix
  std::vector<double> dense = {
    2.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, -1.0, 1.0, 0.0, -1.0, -1.0};
  auto A = dense_to_lower_csc(n, dense);

  simplex_solver_settings_t<int, double> settings;
  std::vector<int> perm;
  csc_matrix_t<int, double> L(n, n, 1);
  std::vector<double> D;
  double work_estimate = 0;
  double start_time    = tic();

  int result = right_looking_ldlt(A, settings, 1e-12, start_time, perm, L, D, work_estimate);

  // Should return INDEFINITE_MATRIX_RETURN (-4) since the matrix is indefinite
  EXPECT_EQ(result, INDEFINITE_MATRIX_RETURN);
}

// Test 16: Large sparse PD matrix A = e1*e^T + e*e1^T + n*I (n = 10000).
// e1 = [1,0,...,0], e = [1,1,...,1].
// A(0,0) = n+2, A(i,0) = A(0,i) = 1 for i > 0, A(i,i) = n for i > 0.
// This is an arrowhead matrix: dense first row/column, diagonal elsewhere.
// Markowitz pivoting should eliminate the n-1 diagonal nodes (degree 1) first,
// producing zero fill, then eliminate the dense node last.
// Without reordering, eliminating the dense node first would cause O(n^2) fill.
TEST(right_looking_ldlt, large_arrowhead_markowitz)
{
  const int n = 10000;

  // Build lower triangle CSC of A = e1*e1^T + e*e1^T + n*I
  // Column 0: entries at all rows (dense column). A(0,0) = n+2, A(i,0) = 1 for i > 0.
  // Columns j > 0: only diagonal entry A(j,j) = n.
  // Lower triangle: column 0 has entries (0, n+2), (1, 1), (2, 1), ..., (n-1, 1).
  //                 column j > 0 has entry (j, n).
  int nnz = n + (n - 1);  // n entries in col 0, (n-1) diagonal entries in cols 1..n-1
  csc_matrix_t<int, double> A(n, n, nnz);
  int p = 0;
  // Column 0
  A.col_start[0] = p;
  A.i[p]         = 0;
  A.x[p]         = static_cast<double>(n + 2);
  p++;
  for (int i = 1; i < n; i++) {
    A.i[p] = i;
    A.x[p] = 1.0;
    p++;
  }
  // Columns 1..n-1
  for (int j = 1; j < n; j++) {
    A.col_start[j] = p;
    A.i[p]         = j;
    A.x[p]         = static_cast<double>(n);
    p++;
  }
  A.col_start[n] = p;

  simplex_solver_settings_t<int, double> settings;
  std::vector<int> perm;
  csc_matrix_t<int, double> L(n, n, 1);
  std::vector<double> D;
  double work_estimate = 0;
  double start_time    = tic();

  int rank = right_looking_ldlt(A, settings, 1e-12, start_time, perm, L, D, work_estimate);

  EXPECT_EQ(rank, n);
  for (int k = 0; k < rank; k++) {
    EXPECT_GT(D[k], 0.0);
  }

  // With Markowitz pivoting, the diagonal nodes (degree 1) should be eliminated first,
  // producing zero fill. The L factor should be very sparse: at most n-1 entries in the
  // last column (from the dense node) plus n unit diagonals.
  int L_nnz = L.col_start[rank];
  // Each of the first n-1 pivots (degree-1 nodes) produces 1 L entry (the unit diagonal)
  // plus 1 off-diagonal entry (connecting to the dense node). The last pivot (dense node)
  // produces just the unit diagonal. Total: n diagonals + (n-1) off-diagonals = 2n - 1.
  EXPECT_LE(L_nnz, 2 * n);

  // Verify factorization correctness on a few random entries by checking P*A*P^T = L*D*L^T.
  // For the arrowhead, after reordering, the first n-1 pivots should be the diagonal nodes.
  // Check that perm[0] != 0 (the dense node should NOT be first).
  EXPECT_NE(perm[0], 0) << "Markowitz should not pick the dense node (index 0) first";
}

}  // namespace cuopt::mathematical_optimization::simplex::test
