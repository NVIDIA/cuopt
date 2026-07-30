/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <barrier/csr_kkt_build.cuh>
#include <barrier/device_sparse_matrix.cuh>
#include <barrier/second_order_cone_kernels.cuh>

#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>

#include <thrust/fill.h>

#include <cmath>
#include <vector>

namespace cuopt::mathematical_optimization::barrier::test {

namespace {

// Packed Hs_diag reference: eta^2 on every entry, head scaled by rank-2 corner d.
std::vector<double> expected_Hs_diag(const cone_data_t<int, double>& cones,
                                     rmm::cuda_stream_view stream)
{
  const int E                    = static_cast<int>(cones.n_sparse_cone_entries);
  auto d_host                    = cuopt::host_copy(cones.d, stream);
  auto eta_host                  = cuopt::host_copy(cones.eta, stream);
  auto sparse_cone_ids_host      = cuopt::host_copy(cones.sparse_cone_ids, stream);
  auto sparse_entry_offsets_host = cuopt::host_copy(cones.sparse_entry_offsets, stream);

  std::vector<double> hs(E);
  for (int s = 0; s < cones.n_sparse_cones; ++s) {
    const int head      = sparse_entry_offsets_host[s];
    const int end       = sparse_entry_offsets_host[s + 1];
    const int cone_idx  = sparse_cone_ids_host[s];
    const double eta_sq = eta_host[cone_idx] * eta_host[cone_idx];
    for (int e = head; e < end; ++e) {
      hs[e] = (e == head) ? eta_sq * d_host[s] : eta_sq;
    }
  }
  return hs;
}

}  // namespace

TEST(sparse_augmented_kkt, cone_counts_and_expansion_size)
{
  auto stream = rmm::cuda_stream_default;

  std::vector<int> cone_dimensions{3, 6, 5};
  rmm::device_uvector<double> x(14, stream);
  rmm::device_uvector<double> z(14, stream);

  cone_data_t<int, double> cones(
    cone_dimensions, cuopt::make_span(x), cuopt::make_span(z), stream, /*soc_threshold=*/4);

  EXPECT_EQ(cones.n_sparse_cones, 2);
  EXPECT_EQ(cones.n_dense_cones(), 1);
  EXPECT_EQ(cones.expansion_var_count(), 4);
  EXPECT_EQ(cones.n_sparse_cone_entries, 11u);
}

TEST(sparse_augmented_kkt, scatter_sparse_hessian_into_augmented)
{
  auto stream = rmm::cuda_stream_default;

  // Two sparse cones so the fused entry-parallel kernel has more than one sparse-cone
  // boundary to get right.
  std::vector<int> cone_dimensions{6, 5};
  rmm::device_uvector<double> x(11, stream);
  rmm::device_uvector<double> z(11, stream);

  cone_data_t<int, double> cones(
    cone_dimensions, cuopt::make_span(x), cuopt::make_span(z), stream, /*soc_threshold=*/4);

  ASSERT_EQ(cones.n_sparse_cones, 2);
  ASSERT_EQ(cones.n_sparse_cone_entries, 11u);
  ASSERT_EQ(cones.expansion_var_count(), 4);

  std::vector<double> x_host(11, 0.0);
  std::vector<double> z_host(11, 0.0);
  x_host[0] = 2.0;
  z_host[0] = 1.5;
  for (int j = 1; j < 6; ++j) {
    x_host[j] = 0.1 * j;
    z_host[j] = 0.08 * j;
  }
  x_host[6] = 1.8;
  z_host[6] = 1.4;
  for (int j = 1; j < 5; ++j) {
    x_host[6 + j] = 0.09 * j;
    z_host[6 + j] = 0.07 * j;
  }

  raft::copy(cones.x.data(), x_host.data(), x_host.size(), stream);
  raft::copy(cones.z.data(), z_host.data(), z_host.size(), stream);

  launch_nt_scaling(cones, stream);
  launch_update_scaling_sparse(cones, stream);

  const int E               = static_cast<int>(cones.n_sparse_cone_entries);
  const double dual_perturb = 0.02;
  const auto hs_expected    = expected_Hs_diag(cones, stream);

  // Distinct augmented-value slots per packed entry: hessian diag, then the four rank-2
  // couplings, then two expansion diagonals per sparse cone.
  std::vector<int> hessian_diag_csr(E);
  std::vector<double> q_values(E);
  std::vector<int> exp_v_col(E);
  std::vector<int> exp_u_col(E);
  std::vector<int> exp_v_row(E);
  std::vector<int> exp_u_row(E);
  for (int e = 0; e < E; ++e) {
    hessian_diag_csr[e] = e;
    q_values[e]         = 0.01 * (e + 1);
    exp_v_col[e]        = 11 + e;
    exp_u_col[e]        = 22 + e;
    exp_v_row[e]        = 33 + e;
    exp_u_row[e]        = 44 + e;
  }
  std::vector<int> sparse_expansion_D{55, 56, 57, 58};
  const int nnz = 59;

  auto d_hessian_diag_csr   = cuopt::device_copy(hessian_diag_csr, stream);
  auto d_q_values           = cuopt::device_copy(q_values, stream);
  auto d_exp_v_col          = cuopt::device_copy(exp_v_col, stream);
  auto d_exp_u_col          = cuopt::device_copy(exp_u_col, stream);
  auto d_exp_v_row          = cuopt::device_copy(exp_v_row, stream);
  auto d_exp_u_row          = cuopt::device_copy(exp_u_row, stream);
  auto d_sparse_expansion_D = cuopt::device_copy(sparse_expansion_D, stream);

  rmm::device_uvector<double> augmented_x(nnz, stream);
  thrust::fill(rmm::exec_policy(stream), augmented_x.begin(), augmented_x.end(), 0.0);
  rmm::device_uvector<double> d_hs_actual(E, stream);

  scatter_sparse_hessian_into_augmented(cones,
                                        augmented_x,
                                        d_hs_actual,
                                        d_hessian_diag_csr,
                                        d_q_values,
                                        d_exp_v_col,
                                        d_exp_u_col,
                                        d_exp_v_row,
                                        d_exp_u_row,
                                        d_sparse_expansion_D,
                                        stream,
                                        dual_perturb);

  auto hs_actual_host       = cuopt::host_copy(d_hs_actual, stream);
  auto aug_host             = cuopt::host_copy(augmented_x, stream);
  auto v_host               = cuopt::host_copy(cones.sparse_v, stream);
  auto u_host               = cuopt::host_copy(cones.sparse_u, stream);
  auto eta_host             = cuopt::host_copy(cones.eta, stream);
  auto sparse_cone_ids_host = cuopt::host_copy(cones.sparse_cone_ids, stream);

  for (int e = 0; e < E; ++e) {
    EXPECT_NEAR(hs_actual_host[e], hs_expected[e], 1e-10) << "Hs_diag entry " << e;
    EXPECT_NEAR(aug_host[e], -hs_actual_host[e] - q_values[e] - dual_perturb, 1e-10)
      << "hessian diag " << e;
    EXPECT_NEAR(aug_host[11 + e], v_host[e], 1e-10) << "v col " << e;
    EXPECT_NEAR(aug_host[22 + e], u_host[e], 1e-10) << "u col " << e;
    EXPECT_NEAR(aug_host[33 + e], v_host[e], 1e-10) << "v row " << e;
    EXPECT_NEAR(aug_host[44 + e], u_host[e], 1e-10) << "u row " << e;
  }

  for (int s = 0; s < cones.n_sparse_cones; ++s) {
    const int cone_idx  = sparse_cone_ids_host[s];
    const double eta_sq = eta_host[cone_idx] * eta_host[cone_idx];
    EXPECT_NEAR(aug_host[55 + 2 * s], -(eta_sq + dual_perturb), 1e-10) << "expansion v " << s;
    EXPECT_NEAR(aug_host[55 + 2 * s + 1], eta_sq + dual_perturb, 1e-10) << "expansion u " << s;
  }
}

TEST(sparse_augmented_kkt, sparse_augmented_matvec)
{
  auto stream = rmm::cuda_stream_default;

  std::vector<int> cone_dimensions{6};
  rmm::device_uvector<double> x(6, stream);
  rmm::device_uvector<double> z(6, stream);

  cone_data_t<int, double> cones(
    cone_dimensions, cuopt::make_span(x), cuopt::make_span(z), stream, /*soc_threshold=*/4);

  ASSERT_EQ(cones.n_sparse_cones, 1);
  ASSERT_EQ(cones.expansion_var_count(), 2);

  std::vector<double> x_host{2.0, 0.2, 0.3, 0.4, 0.5, 0.6};
  std::vector<double> z_host{1.5, 0.1, 0.15, 0.2, 0.25, 0.3};
  raft::copy(cones.x.data(), x_host.data(), x_host.size(), stream);
  raft::copy(cones.z.data(), z_host.data(), z_host.size(), stream);

  launch_nt_scaling(cones, stream);
  launch_update_scaling_sparse(cones, stream);

  const int n_primal = 6;
  const int m_rows   = 1;
  const int p        = cones.expansion_var_count();
  const int sys_size = n_primal + m_rows + p;

  std::vector<double> x_vec(sys_size, 0.0);
  x_vec[0]                     = 1.1;
  x_vec[1]                     = 0.3;
  x_vec[2]                     = 0.4;
  x_vec[3]                     = 0.2;
  x_vec[4]                     = 0.5;
  x_vec[5]                     = 0.6;
  x_vec[n_primal + m_rows]     = 0.25;   // expansion v
  x_vec[n_primal + m_rows + 1] = -0.15;  // expansion u

  const auto hs_host = expected_Hs_diag(cones, stream);
  auto d_hs          = cuopt::device_copy(hs_host, stream);

  rmm::device_uvector<double> d_x(sys_size, stream);
  rmm::device_uvector<double> d_r1(n_primal, stream);
  rmm::device_uvector<double> d_y_exp(p, stream);

  raft::copy(d_x.data(), x_vec.data(), sys_size, stream);
  thrust::fill(rmm::exec_policy(stream), d_r1.begin(), d_r1.end(), 0.0);
  thrust::fill(rmm::exec_policy(stream), d_y_exp.begin(), d_y_exp.end(), 0.0);

  launch_sparse_augmented_matvec(raft::device_span<const double>(d_x.data(), d_x.size()),
                                 raft::device_span<double>(d_r1.data(), d_r1.size()),
                                 raft::device_span<double>(d_y_exp.data(), d_y_exp.size()),
                                 cones,
                                 raft::device_span<const double>(d_hs.data(), d_hs.size()),
                                 /*cone_var_start=*/0,
                                 n_primal,
                                 m_rows,
                                 stream);

  auto r1_host   = cuopt::host_copy(d_r1, stream);
  auto yexp_host = cuopt::host_copy(d_y_exp, stream);
  auto v_host    = cuopt::host_copy(cones.sparse_v, stream);
  auto u_host    = cuopt::host_copy(cones.sparse_u, stream);
  auto eta_host  = cuopt::host_copy(cones.eta, stream);

  const double eta_sq = eta_host[0] * eta_host[0];
  double dot_v        = 0.0;
  double dot_u        = 0.0;
  for (int j = 0; j < 6; ++j) {
    dot_v += v_host[j] * x_vec[j];
    dot_u += u_host[j] * x_vec[j];
    const double expected = hs_host[j] * x_vec[j] - v_host[j] * x_vec[n_primal + m_rows] -
                            u_host[j] * x_vec[n_primal + m_rows + 1];
    EXPECT_NEAR(r1_host[j], expected, 1e-10) << "primal row " << j;
  }

  EXPECT_NEAR(yexp_host[0], -eta_sq * x_vec[n_primal + m_rows] + dot_v, 1e-10);
  EXPECT_NEAR(yexp_host[1], eta_sq * x_vec[n_primal + m_rows + 1] + dot_u, 1e-10);
}

TEST(sparse_augmented_kkt, update_scaling_sparse_dim_1000)
{
  auto stream = rmm::cuda_stream_default;

  std::vector<int> cone_dimensions{1000};
  rmm::device_uvector<double> x(1000, stream);
  rmm::device_uvector<double> z(1000, stream);

  cone_data_t<int, double> cones(
    cone_dimensions, cuopt::make_span(x), cuopt::make_span(z), stream, /*soc_threshold=*/5);

  ASSERT_EQ(cones.n_sparse_cones, 1);
  ASSERT_EQ(cones.n_sparse_cone_entries, 1000u);
  ASSERT_EQ(cones.expansion_var_count(), 2);

  std::vector<double> x_host(1000);
  std::vector<double> z_host(1000);
  x_host[0] = 100.0;
  z_host[0] = 80.0;
  for (int j = 1; j < 1000; ++j) {
    x_host[j] = 0.001 * ((j % 5) + 1);
    z_host[j] = 0.0015 * ((j % 7) + 1);
  }

  raft::copy(cones.x.data(), x_host.data(), x_host.size(), stream);
  raft::copy(cones.z.data(), z_host.data(), z_host.size(), stream);

  launch_nt_scaling(cones, stream);
  launch_update_scaling_sparse(cones, stream);

  auto d_host   = cuopt::host_copy(cones.d, stream);
  auto v_host   = cuopt::host_copy(cones.sparse_v, stream);
  auto u_host   = cuopt::host_copy(cones.sparse_u, stream);
  auto eta_host = cuopt::host_copy(cones.eta, stream);

  EXPECT_GT(d_host[0], 0.0);
  EXPECT_GT(eta_host[0], 0.0);
  for (int j = 0; j < 1000; ++j) {
    EXPECT_TRUE(std::isfinite(v_host[j])) << "v entry " << j;
    EXPECT_TRUE(std::isfinite(u_host[j])) << "u entry " << j;
  }

  const auto hs_host = expected_Hs_diag(cones, stream);
  auto d_hs          = cuopt::device_copy(hs_host, stream);

  const int n_primal = 1000;
  const int m_rows   = 1;
  const int p        = cones.expansion_var_count();
  const int sys_size = n_primal + m_rows + p;

  std::vector<double> x_vec(sys_size, 0.0);
  for (int j = 0; j < 1000; ++j) {
    x_vec[j] = x_host[j];
  }
  x_vec[n_primal + m_rows]     = 0.25;
  x_vec[n_primal + m_rows + 1] = -0.15;

  rmm::device_uvector<double> d_x(sys_size, stream);
  rmm::device_uvector<double> d_r1(n_primal, stream);
  rmm::device_uvector<double> d_y_exp(p, stream);

  raft::copy(d_x.data(), x_vec.data(), sys_size, stream);
  thrust::fill(rmm::exec_policy(stream), d_r1.begin(), d_r1.end(), 0.0);
  thrust::fill(rmm::exec_policy(stream), d_y_exp.begin(), d_y_exp.end(), 0.0);

  launch_sparse_augmented_matvec(raft::device_span<const double>(d_x.data(), d_x.size()),
                                 raft::device_span<double>(d_r1.data(), d_r1.size()),
                                 raft::device_span<double>(d_y_exp.data(), d_y_exp.size()),
                                 cones,
                                 raft::device_span<const double>(d_hs.data(), d_hs.size()),
                                 /*cone_var_start=*/0,
                                 n_primal,
                                 m_rows,
                                 stream);

  auto r1_host   = cuopt::host_copy(d_r1, stream);
  auto yexp_host = cuopt::host_copy(d_y_exp, stream);

  const double eta_sq  = eta_host[0] * eta_host[0];
  const double x_exp_v = x_vec[n_primal + m_rows];
  const double x_exp_u = x_vec[n_primal + m_rows + 1];
  double dot_v         = 0.0;
  double dot_u         = 0.0;
  for (int j = 0; j < 1000; ++j) {
    dot_v += v_host[j] * x_vec[j];
    dot_u += u_host[j] * x_vec[j];
    const double expected = hs_host[j] * x_vec[j] - v_host[j] * x_exp_v - u_host[j] * x_exp_u;
    EXPECT_NEAR(r1_host[j], expected, 1e-9) << "primal row " << j;
  }

  EXPECT_NEAR(yexp_host[0], -eta_sq * x_exp_v + dot_v, 1e-9);
  EXPECT_NEAR(yexp_host[1], eta_sq * x_exp_u + dot_u, 1e-9);
}

TEST(sparse_augmented_kkt, gpu_augmented_csr_metadata_matches_host)
{
  auto stream = rmm::cuda_stream_default;

  std::vector<int> cone_dimensions{3, 6, 5};
  rmm::device_uvector<double> x(14, stream);
  rmm::device_uvector<double> z(14, stream);
  cone_data_t<int, double> cones(
    cone_dimensions, cuopt::make_span(x), cuopt::make_span(z), stream, /*soc_threshold=*/4);

  cone_kkt_data_t<int, double> metadata(stream);
  build_augmented_csr_metadata(cones, metadata, stream);

  auto sparse_idx_host    = cuopt::host_copy(metadata.sparse_ids_by_cone, stream);
  auto dense_idx_host     = cuopt::host_copy(metadata.dense_ids_by_cone, stream);
  auto dense_ids_host     = cuopt::host_copy(metadata.dense_cone_ids, stream);
  auto block_offsets_host = cuopt::host_copy(metadata.dense_block_offsets, stream);

  std::vector<int> expected_sparse_idx(cones.n_cones, -1);
  auto sparse_cone_ids_host = cuopt::host_copy(cones.sparse_cone_ids, stream);
  for (int s = 0; s < cones.n_sparse_cones; ++s) {
    expected_sparse_idx[sparse_cone_ids_host[s]] = s;
  }
  ASSERT_EQ(expected_sparse_idx.size(), sparse_idx_host.size());
  for (size_t e = 0; e < expected_sparse_idx.size(); ++e) {
    EXPECT_EQ(expected_sparse_idx[e], sparse_idx_host[e]) << "sparse_ids_by_cone index " << e;
  }

  int dense_count          = 0;
  auto cone_is_sparse_host = cuopt::host_copy(cones.cone_is_sparse, stream);
  for (int k = 0; k < cones.n_cones; ++k) {
    if (cone_is_sparse_host[k] == 0) {
      EXPECT_EQ(dense_idx_host[k], dense_count);
      ASSERT_LT(dense_count, static_cast<int>(dense_ids_host.size()));
      EXPECT_EQ(dense_ids_host[dense_count], k);
      ++dense_count;
    } else {
      EXPECT_EQ(dense_idx_host[k], -1);
    }
  }
  EXPECT_EQ(dense_count, cones.n_dense_cones());
  EXPECT_EQ(metadata.dense_soc_kkt_nnz, block_offsets_host[dense_count]);
}

// Verifies the GPU-built augmented KKT CSR structure (row_start + column
// indices) and values for a concrete mixed SOCP: one linear variable, one
// dense Q^3 cone, one sparse Q^4 cone, quadratic cost, and two constraints.
// The augmented CSR is emitted column-sorted per row.
TEST(sparse_augmented_kkt, augmented_csr_indices_mixed_dense_sparse_qp)
{
  // Augmented KKT for the QP-SOCP:
  //
  //   minimize    (1/2) x^T Q x
  //   subject to  x_0 + x_4 = b_0                    (constraint 0)
  //               x_1 + x_6 = b_1                    (constraint 1)
  //               (x_1, x_2, x_3)      in Q^3        (dense cone)
  //               (x_4, x_5, x_6, x_7) in Q^4        (sparse cone)
  //
  // Q = diag(2, 3, ..., 9) with symmetric off-diagonals Q[0,4]=0.5, Q[1,7]=0.25.
  // This checks only the built KKT sparsity/values, so the objective linear term
  // and the right-hand side b are irrelevant and left unset.
  using i_t   = int;
  using f_t   = double;
  auto stream = rmm::cuda_stream_default;

  // Layout: 1 linear var, dense Q^3 cone (cols [1,4)), sparse Q^4 cone (cols
  // [4,8)), 2 constraints. Factorization size = n + m + p = 8 + 2 + 2 = 12.
  constexpr i_t n_linear   = 1;
  constexpr i_t n          = 8;  // 1 linear + 3 (dense) + 4 (sparse)
  constexpr i_t m          = 2;
  constexpr i_t cone_start = n_linear;
  constexpr i_t m_c        = 7;  // 3 + 4 cone entries
  const f_t dual_perturb   = 0.01;
  const f_t primal_perturb = 0.001;

  // Constraint matrix A (m x n, CSC by variable column): var0 -> c0, var1 -> c1,
  // var4 -> c0, var6 -> c1.
  csc_matrix_t<i_t, f_t> A(m, n, 4);
  A.col_start = {0, 1, 2, 2, 2, 3, 3, 4, 4};
  A.i         = {0, 1, 0, 1};
  A.x         = {1.0, 1.0, 1.0, 1.0};

  csc_matrix_t<i_t, f_t> AT(n, m, 4);
  A.transpose(AT);

  // Quadratic cost Q (n x n, symmetric CSC): diagonal plus off-diagonal pairs
  // (0,4) and (1,7) to exercise sorted insertion of the diagonal and of Q
  // entries outside the cone block.
  csc_matrix_t<i_t, f_t> Q(n, n, 12);
  Q.col_start    = {0, 2, 4, 5, 6, 8, 9, 10, 12};
  Q.i            = {0, 4, 1, 7, 2, 3, 0, 4, 5, 6, 1, 7};
  Q.x            = {2.0, 0.5, 3.0, 0.25, 4.0, 5.0, 0.5, 6.0, 7.0, 8.0, 0.25, 9.0};
  const i_t nnzQ = Q.col_start[n];

  std::vector<f_t> diag(n, 0.1);

  device_csc_matrix_t<i_t, f_t> d_A(A, stream);
  device_csc_matrix_t<i_t, f_t> d_AT(AT, stream);
  device_csc_matrix_t<i_t, f_t> d_Q(Q, stream);
  auto d_diag = cuopt::device_copy(diag, stream);

  rmm::device_uvector<f_t> cone_x(m_c, stream);
  rmm::device_uvector<f_t> cone_z(m_c, stream);
  std::vector<i_t> cone_dimensions{3, 4};
  cone_data_t<i_t, f_t> cones(cone_dimensions,
                              cuopt::make_span(cone_x),
                              cuopt::make_span(cone_z),
                              stream,
                              /*soc_threshold=*/3);
  const i_t p = static_cast<i_t>(cones.expansion_var_count());
  ASSERT_EQ(cones.n_sparse_cones, 1);
  ASSERT_EQ(p, 2);
  ASSERT_EQ(cones.n_sparse_cone_entries, 4u);

  cone_kkt_data_t<i_t, f_t> cone_data(stream);
  build_augmented_csr_metadata(cones, cone_data, stream);

  device_csr_matrix_t<i_t, f_t> device_augmented(stream);
  rmm::device_uvector<i_t> d_augmented_diagonal_indices(0, stream);

  sparse_cone_views_t<i_t, f_t> cone_views{
    raft::device_span<const i_t>{cones.element_cone_ids.data(), cones.element_cone_ids.size()},
    raft::device_span<const size_t>{cones.cone_offsets.data(), cones.cone_offsets.size()},
    raft::device_span<const i_t>{cones.sparse_cone_ids.data(), cones.sparse_cone_ids.size()},
    raft::device_span<const i_t>{cones.sparse_entry_offsets.data(),
                                 cones.sparse_entry_offsets.size()},
    cones.n_sparse_cone_entries};

  const i_t total_nnz =
    build_augmented_csr_on_device(n,
                                  m,
                                  p,
                                  cone_start,
                                  m_c,
                                  nnzQ,
                                  dual_perturb,
                                  primal_perturb,
                                  d_A,
                                  d_Q,
                                  d_AT,
                                  raft::device_span<const f_t>{d_diag.data(), d_diag.size()},
                                  cone_views,
                                  cone_data,
                                  d_augmented_diagonal_indices,
                                  device_augmented,
                                  stream);

  // Expected column-sorted CSR. Row blocks: primal [0,8), dual [8,10),
  // expansion [10,12). Column blocks: primal [0,8), dual [8,10), expansion cols
  // {10 (v), 11 (u)}.
  const std::vector<i_t> expected_row_start{0, 3, 8, 11, 14, 19, 22, 26, 30, 33, 36, 41, 46};
  const std::vector<i_t> expected_j{
    0, 4,  8,            // row0  linear var0: diag, Q(0,4), A^T(c0)
    1, 2,  3,  7,  9,    // row1  dense r0: block[1..3], Q(1,7), A^T(c1)
    1, 2,  3,            // row2  dense r1
    1, 2,  3,            // row3  dense r2
    0, 4,  8,  10, 11,   // row4  sparse r0: Q(4,0), diag, A^T(c0), exp v, exp u
    5, 10, 11,           // row5  sparse r1
    6, 9,  10, 11,       // row6  sparse r2: diag, A^T(c1), exp v, exp u
    1, 7,  10, 11,       // row7  sparse r3: Q(7,1), diag, exp v, exp u
    0, 4,  8,            // row8  dual c0: A^T rows {0,4}, diag
    1, 6,  9,            // row9  dual c1: A^T rows {1,6}, diag
    4, 5,  6,  7,  10,   // row10 expansion v: cone cols, diag
    4, 5,  6,  7,  11};  // row11 expansion u: cone cols, diag
  const std::vector<f_t> expected_x{-2.11, -0.5,  1.0,                 // row0
                                    -3.01, 0.0,   0.0,   -0.25, 1.0,   // row1
                                    0.0,   -4.01, 0.0,                 // row2
                                    0.0,   0.0,   -5.01,               // row3
                                    -0.5,  -6.01, 1.0,   0.0,   0.0,   // row4
                                    -7.01, 0.0,   0.0,                 // row5
                                    -8.01, 1.0,   0.0,   0.0,          // row6
                                    -0.25, -9.01, 0.0,   0.0,          // row7
                                    1.0,   1.0,   0.001,               // row8
                                    1.0,   1.0,   0.001,               // row9
                                    0.0,   0.0,   0.0,   0.0,   0.0,   // row10
                                    0.0,   0.0,   0.0,   0.0,   0.0};  // row11

  EXPECT_EQ(total_nnz, 46);

  auto row_start_host = cuopt::host_copy(device_augmented.row_start, stream);
  auto j_host         = cuopt::host_copy(device_augmented.j, stream);
  auto x_host         = cuopt::host_copy(device_augmented.x, stream);

  ASSERT_EQ(row_start_host.size(), expected_row_start.size());
  for (size_t r = 0; r < expected_row_start.size(); ++r) {
    EXPECT_EQ(row_start_host[r], expected_row_start[r]) << "row_start[" << r << "]";
  }

  ASSERT_EQ(j_host.size(), expected_j.size());
  ASSERT_EQ(x_host.size(), expected_x.size());
  for (size_t e = 0; e < expected_j.size(); ++e) {
    EXPECT_EQ(j_host[e], expected_j[e]) << "j[" << e << "]";
    EXPECT_NEAR(x_host[e], expected_x[e], 1e-12) << "x[" << e << "]";
  }
}

}  // namespace cuopt::mathematical_optimization::barrier::test
