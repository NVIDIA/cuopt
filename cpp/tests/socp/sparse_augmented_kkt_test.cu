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

  std::vector<int> cone_dimensions{3, 6};
  rmm::device_uvector<double> x(9, stream);
  rmm::device_uvector<double> z(9, stream);

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

}  // namespace cuopt::mathematical_optimization::barrier::test
