/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <barrier/second_order_cone_kernels.cuh>

#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>

#include <thrust/fill.h>

#include <cmath>
#include <vector>

namespace cuopt::mathematical_optimization::barrier::test {

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

TEST(sparse_augmented_kkt, launch_get_Hs_sparse)
{
  auto stream = rmm::cuda_stream_default;

  std::vector<int> cone_dimensions{3, 6};
  rmm::device_uvector<double> x(9, stream);
  rmm::device_uvector<double> z(9, stream);

  cone_data_t<int, double> cones(
    cone_dimensions, cuopt::make_span(x), cuopt::make_span(z), stream, /*soc_threshold=*/4);

  ASSERT_EQ(cones.n_sparse_cones, 1);

  std::vector<double> x_host(9, 0.0);
  std::vector<double> z_host(9, 0.0);
  x_host[3] = 2.0;
  z_host[3] = 1.5;
  for (int j = 1; j < 6; ++j) {
    x_host[3 + j] = 0.1 * j;
    z_host[3 + j] = 0.08 * j;
  }
  x_host[0] = 1.5;
  z_host[0] = 1.2;
  for (int j = 1; j < 3; ++j) {
    x_host[j] = 0.05;
    z_host[j] = 0.04;
  }

  raft::copy(cones.x.data(), x_host.data(), x_host.size(), stream);
  raft::copy(cones.z.data(), z_host.data(), z_host.size(), stream);

  launch_nt_scaling(cones, stream);
  launch_update_scaling_sparse(cones, stream);

  rmm::device_uvector<double> Hs_diag(cones.n_sparse_cone_entries, stream);
  launch_get_Hs_sparse(cones, Hs_diag, stream);

  auto d_host   = cuopt::host_copy(cones.d, stream);
  auto eta_host = cuopt::host_copy(cones.eta, stream);
  auto hs_host  = cuopt::host_copy(Hs_diag, stream);

  const int sparse_cone = 1;
  const double eta_sq   = eta_host[sparse_cone] * eta_host[sparse_cone];

  EXPECT_NEAR(hs_host[0], eta_sq * d_host[0], 1e-10);
  for (int j = 1; j < 6; ++j) {
    EXPECT_NEAR(hs_host[j], eta_sq, 1e-10) << "tail index " << j;
  }
}

TEST(sparse_augmented_kkt, scatter_and_update_sparse_expansion)
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

  const int nnz = 32;

  rmm::device_uvector<double> augmented_x(nnz, stream);
  thrust::fill(rmm::exec_policy(stream), augmented_x.begin(), augmented_x.end(), 0.0);

  // One coupling/Hessian CSR entry per sparse-cone element, matching production where every
  // *_col/*_row index array has size n_sparse_cone_entries (== cones.sparse_v.size()).
  const int q_dim = 6;
  ASSERT_EQ(cones.n_sparse_cone_entries, static_cast<std::size_t>(q_dim));

  std::vector<int> sparse_hessian_diag(q_dim);
  std::vector<double> sparse_hessian_Q(q_dim, 0.0);
  std::vector<int> sparse_exp_v_col(q_dim);
  std::vector<int> sparse_exp_u_col(q_dim);
  std::vector<int> sparse_exp_v_row(q_dim);
  std::vector<int> sparse_exp_u_row(q_dim);
  std::vector<int> sparse_expansion_D{30, 31};
  for (int j = 0; j < q_dim; ++j) {
    sparse_hessian_diag[j] = j;       // CSR slots 0..5
    sparse_exp_v_col[j]    = 6 + j;   // 6..11
    sparse_exp_u_col[j]    = 12 + j;  // 12..17
    sparse_exp_v_row[j]    = 18 + j;  // 18..23
    sparse_exp_u_row[j]    = 24 + j;  // 24..29
  }

  auto d_sparse_hessian_diag = cuopt::device_copy(sparse_hessian_diag, stream);
  auto d_sparse_hessian_Q    = cuopt::device_copy(sparse_hessian_Q, stream);
  auto d_sparse_exp_v_col    = cuopt::device_copy(sparse_exp_v_col, stream);
  auto d_sparse_exp_u_col    = cuopt::device_copy(sparse_exp_u_col, stream);
  auto d_sparse_exp_v_row    = cuopt::device_copy(sparse_exp_v_row, stream);
  auto d_sparse_exp_u_row    = cuopt::device_copy(sparse_exp_u_row, stream);
  auto d_sparse_expansion_D  = cuopt::device_copy(sparse_expansion_D, stream);
  rmm::device_uvector<double> d_sparse_Hs_diag(cones.n_sparse_cone_entries, stream);

  launch_get_Hs_sparse(cones, d_sparse_Hs_diag, stream);
  scatter_sparse_hessian_diag_into_augmented(
    augmented_x, d_sparse_hessian_diag, d_sparse_Hs_diag, d_sparse_hessian_Q, stream, 0.0);
  update_sparse_expansion_in_augmented(augmented_x,
                                       d_sparse_exp_v_col,
                                       d_sparse_exp_u_col,
                                       d_sparse_exp_v_row,
                                       d_sparse_exp_u_row,
                                       d_sparse_expansion_D,
                                       cones.sparse_v,
                                       cones.sparse_u,
                                       cones.eta,
                                       cones.sparse_cone_ids,
                                       stream,
                                       0.0);

  auto aug_host = cuopt::host_copy(augmented_x, stream);
  auto v_host   = cuopt::host_copy(cones.sparse_v, stream);
  auto u_host   = cuopt::host_copy(cones.sparse_u, stream);
  auto hs_host  = cuopt::host_copy(d_sparse_Hs_diag, stream);
  auto eta_host = cuopt::host_copy(cones.eta, stream);

  for (int j = 0; j < q_dim; ++j) {
    EXPECT_NEAR(aug_host[j], -hs_host[j], 1e-10) << "hessian diag " << j;
    EXPECT_NEAR(aug_host[6 + j], v_host[j], 1e-10) << "v col " << j;
    EXPECT_NEAR(aug_host[12 + j], u_host[j], 1e-10) << "u col " << j;
    EXPECT_NEAR(aug_host[18 + j], v_host[j], 1e-10) << "v row " << j;
    EXPECT_NEAR(aug_host[24 + j], u_host[j], 1e-10) << "u row " << j;
  }

  const double eta_sq = eta_host[0] * eta_host[0];
  EXPECT_NEAR(aug_host[30], -eta_sq, 1e-10);
  EXPECT_NEAR(aug_host[31], eta_sq, 1e-10);
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

  rmm::device_uvector<double> d_x(sys_size, stream);
  rmm::device_uvector<double> d_r1(n_primal, stream);
  rmm::device_uvector<double> d_y_exp(p, stream);
  rmm::device_uvector<double> d_hs(cones.n_sparse_cone_entries, stream);

  raft::copy(d_x.data(), x_vec.data(), sys_size, stream);
  thrust::fill(rmm::exec_policy(stream), d_r1.begin(), d_r1.end(), 0.0);
  thrust::fill(rmm::exec_policy(stream), d_y_exp.begin(), d_y_exp.end(), 0.0);

  launch_get_Hs_sparse(cones, d_hs, stream);
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
  auto hs_host   = cuopt::host_copy(d_hs, stream);
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

  rmm::device_uvector<double> Hs_diag(cones.n_sparse_cone_entries, stream);
  launch_get_Hs_sparse(cones, Hs_diag, stream);

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
                                 raft::device_span<const double>(Hs_diag.data(), Hs_diag.size()),
                                 /*cone_var_start=*/0,
                                 n_primal,
                                 m_rows,
                                 stream);

  auto r1_host   = cuopt::host_copy(d_r1, stream);
  auto yexp_host = cuopt::host_copy(d_y_exp, stream);
  auto hs_host   = cuopt::host_copy(Hs_diag, stream);

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

}  // namespace cuopt::mathematical_optimization::barrier::test
