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

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace cuopt::mathematical_optimization::barrier::test {

namespace {

// Host CSC with the given row indices per column, in the order given (deliberately unsorted).
csc_matrix_t<int, double> make_csc(int m,
                                   const std::vector<std::vector<int>>& rows_per_col,
                                   std::mt19937& rng)
{
  const int n = static_cast<int>(rows_per_col.size());
  int nz      = 0;
  for (const auto& rows : rows_per_col) {
    nz += static_cast<int>(rows.size());
  }

  csc_matrix_t<int, double> A(m, n, nz);
  std::uniform_real_distribution<double> value(-10.0, 10.0);
  int p          = 0;
  A.col_start[0] = 0;
  for (int j = 0; j < n; ++j) {
    for (int r : rows_per_col[j]) {
      A.i[p] = r;
      A.x[p] = value(rng);
      ++p;
    }
    A.col_start[j + 1] = p;
  }
  return A;
}

// A random subset of `count` distinct rows drawn from `candidates`, in shuffled order.
std::vector<int> random_rows(const std::vector<int>& candidates, int count, std::mt19937& rng)
{
  std::vector<int> rows(candidates);
  std::shuffle(rows.begin(), rows.end(), rng);
  rows.resize(count);
  return rows;
}

// The device conversion must reproduce the host reference exactly and be run-to-run stable.
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

  device_csr_matrix_t<int, double> d_again(stream);
  d_A.to_compressed_row(d_again, stream);
  auto again = d_again.to_host(stream);
  EXPECT_EQ(again.row_start, got.row_start);
  EXPECT_EQ(again.j, got.j);
  EXPECT_EQ(again.x, got.x);

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

TEST(device_sparse_matrix, csc_to_csr_random_with_empty_rows_and_columns)
{
  std::mt19937 rng(42);
  const int m = 37;
  const int n = 29;

  // Rows that are multiples of 5 never appear, so the CSR has interior empty rows.
  std::vector<int> candidates;
  for (int r = 0; r < m; ++r) {
    if (r % 5 != 0) { candidates.push_back(r); }
  }

  std::vector<std::vector<int>> rows_per_col(n);
  std::uniform_int_distribution<int> length(1, 6);
  for (int j = 0; j < n; ++j) {
    if (j % 7 == 3) { continue; }  // empty column
    rows_per_col[j] = random_rows(candidates, length(rng), rng);
  }

  expect_device_matches_host(make_csc(m, rows_per_col, rng));
}

TEST(device_sparse_matrix, csc_to_csr_dense_column)
{
  std::mt19937 rng(7);
  const int m = 1000;
  const int n = 5;

  std::vector<int> all_rows(m);
  std::iota(all_rows.begin(), all_rows.end(), 0);

  // Column 2 holds every row (longer than one thread block), the rest are short.
  std::vector<std::vector<int>> rows_per_col(n);
  for (int j = 0; j < n; ++j) {
    rows_per_col[j] = random_rows(all_rows, j == 2 ? m : 3, rng);
  }

  expect_device_matches_host(make_csc(m, rows_per_col, rng));
}

TEST(device_sparse_matrix, csc_to_csr_empty_matrix)
{
  std::mt19937 rng(1);
  expect_device_matches_host(make_csc(4, std::vector<std::vector<int>>(3), rng));
}

TEST(device_sparse_matrix, csc_to_csr_single_entry)
{
  std::mt19937 rng(3);
  expect_device_matches_host(make_csc(1, {{0}}, rng));
}

}  // namespace cuopt::mathematical_optimization::barrier::test
