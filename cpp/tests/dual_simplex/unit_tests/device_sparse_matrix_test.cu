/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <barrier/device_sparse_matrix.cuh>
#include <linear_algebra/sparse_matrix.hpp>

#include <cuda/stream>

#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace cuopt::mathematical_optimization::barrier::test {

namespace {

// A distinct value encoding the entry's position, so a mis-permutation names the entry it came
// from in the failure output. Integral, so host and device copies compare bit-exact.
double value_at(int row, int col)
{
  const double magnitude = 1000.0 * (row + 1) + (col + 1);
  return col % 2 == 0 ? magnitude : -magnitude;
}

// Host CSC with the given row indices per column, in the order given (deliberately unsorted).
csc_matrix_t<int, double> make_csc(int m, const std::vector<std::vector<int>>& rows_per_col)
{
  const int n = static_cast<int>(rows_per_col.size());
  int nz      = 0;
  for (const auto& rows : rows_per_col) {
    nz += static_cast<int>(rows.size());
  }

  csc_matrix_t<int, double> A(m, n, nz);
  int p          = 0;
  A.col_start[0] = 0;
  for (int j = 0; j < n; ++j) {
    for (int r : rows_per_col[j]) {
      A.i[p] = r;
      A.x[p] = value_at(r, j);
      ++p;
    }
    A.col_start[j + 1] = p;
  }
  return A;
}

// The device conversion must reproduce the host reference exactly.
void expect_device_matches_host(const csc_matrix_t<int, double>& A)
{
  auto stream = cuda::stream_ref{cudaStream_t{cudaStreamDefault}};

  csr_matrix_t<int, double> expected(A.m, A.n, A.col_start[A.n]);
  A.to_compressed_row(expected);

  device_csc_matrix_t<int, double> d_A(A, stream);
  device_csr_matrix_t<int, double> d_Arow(stream);
  d_A.to_compressed_row(d_Arow, stream);
  auto got = d_Arow.to_host(stream);

  ASSERT_EQ(got.m, expected.m);
  ASSERT_EQ(got.n, expected.n);
  EXPECT_EQ(got.row_start, expected.row_start);
  EXPECT_EQ(got.j, expected.j);
  EXPECT_EQ(got.x, expected.x);

  // The transpose shares the conversion, and CSC(A^T) holds the same arrays as CSR(A).
  csc_matrix_t<int, double> expected_t(1, 1, 1);
  A.transpose(expected_t);

  device_csc_matrix_t<int, double> d_AT(stream);
  d_A.transpose(d_AT, stream);
  auto got_t = d_AT.to_host(stream);

  ASSERT_EQ(got_t.m, expected_t.m);
  ASSERT_EQ(got_t.n, expected_t.n);
  EXPECT_EQ(got_t.col_start, expected_t.col_start);
  EXPECT_EQ(got_t.i, expected_t.i);
  EXPECT_EQ(got_t.x, expected_t.x);
}

}  // namespace

TEST(device_sparse_matrix, csc_to_csr_empty_rows_and_columns)
{
  // 9 x 6 sparsity pattern:
  //        c0 c1 c2 c3 c4 c5
  //   r0    .  .  .  .  .  .
  //   r1    x  .  .  x  .  x
  //   r2    .  .  .  .  .  .
  //   r3    .  .  x  .  .  x
  //   r4    .  .  .  .  .  .
  //   r5    x  .  .  .  .  x
  //   r6    .  .  .  .  .  .
  //   r7    x  .  x  .  .  x
  //   r8    .  .  .  .  .  .
  //
  // Rows 0, 2, 4, 6 and 8 hold no entries, giving leading, interior and trailing empty CSR rows,
  // i.e. zero-length sort segments. Columns 1 and 4 are empty, so their scatter blocks do no work.
  // Every row indices list is out of order, and rows 1, 3, 5 and 7 each hold more than one entry,
  // so the segmented sort has to restore column order rather than inherit it from the input.
  const std::vector<std::vector<int>> rows_per_col = {
    {5, 1, 7},     // c0
    {},            // c1, empty
    {7, 3},        // c2
    {1},           // c3
    {},            // c4, empty
    {3, 7, 5, 1},  // c5
  };

  expect_device_matches_host(make_csc(9, rows_per_col));
}

TEST(device_sparse_matrix, csc_to_csr_dense_column)
{
  constexpr int m = 1000;

  // Column 2 holds every row, in descending order. The scatter kernel gives each column one
  // 256-thread block, so this column makes its strided loop wrap several times.
  std::vector<int> all_rows_descending;
  all_rows_descending.reserve(m);
  for (int r = m - 1; r >= 0; --r) {
    all_rows_descending.push_back(r);
  }

  // The short columns repeat rows 0 and 999 so the first and last CSR rows hold several entries
  // rather than just the one the dense column contributes.
  const std::vector<std::vector<int>> rows_per_col = {
    {900, 4, 500},        // c0
    {0, 999},             // c1
    all_rows_descending,  // c2
    {999, 0, 7},          // c3
    {251, 250},           // c4
  };

  expect_device_matches_host(make_csc(m, rows_per_col));
}

TEST(device_sparse_matrix, csc_to_csr_empty_matrix)
{
  // No nonzeros at all: the conversion takes its early return, which zeroes the offsets and
  // launches no kernel.
  expect_device_matches_host(make_csc(4, {{}, {}, {}}));
}

TEST(device_sparse_matrix, csc_to_csr_single_entry)
{
  expect_device_matches_host(make_csc(1, {{0}}));
}

}  // namespace cuopt::mathematical_optimization::barrier::test
