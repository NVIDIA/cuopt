/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <barrier/second_order_cone_kernels.cuh>

#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace cuopt::linear_programming::dual_simplex::test {

// Mirrors `cone_step_length_from_scalars` in second_order_cone_kernels.cuh (device).
double host_cone_step_length_from_scalars(double u0,
                                          double du0,
                                          double du_tail_sq,
                                          double u_tail_du_tail,
                                          double u_tail_sq,
                                          double alpha_max)
{
  const auto a     = du0 * du0 - du_tail_sq;
  const auto b     = u0 * du0 - u_tail_du_tail;
  const auto c_raw = u0 * u0 - u_tail_sq;
  const auto c     = c_raw > 0.0 ? c_raw : 0.0;
  const auto disc  = b * b - a * c;
  auto alpha       = alpha_max;

  if (u0 >= 0.0 && du0 < 0.0) { alpha = std::min(alpha, -u0 / du0); }

  if ((a > 0.0 && b > 0.0) || disc < 0.0) { return alpha; }

  if (a == 0.0) {
    return alpha;
  } else if (c == 0.0) {
    alpha = a >= 0.0 ? alpha : 0.0;
  } else {
    const auto t = -(b + std::copysign(std::sqrt(disc), b));
    auto r1      = c / t;
    auto r2      = t / a;
    if (r1 < 0.0) { r1 = alpha; }
    if (r2 < 0.0) { r2 = alpha; }
    alpha = std::min(alpha, std::min(r1, r2));
  }

  return alpha;
}

TEST(second_order_cone_kernels, topology_and_scratch_layout)
{
  auto stream = rmm::cuda_stream_default;

  std::vector<int> cone_dimensions{3, 2, 5};
  rmm::device_uvector<double> x(10, stream);
  rmm::device_uvector<double> z(10, stream);

  cone_data_t<int, double> cones(cone_dimensions, cuopt::make_span(x), cuopt::make_span(z), stream);

  EXPECT_EQ(cones.n_cones, std::size_t{3});
  EXPECT_EQ(cones.n_cone_entries, std::size_t{10});
  EXPECT_EQ(cones.x.data(), x.data());
  EXPECT_EQ(cones.z.data(), z.data());

  EXPECT_EQ(cuopt::host_copy(cones.cone_offsets, stream), (std::vector<std::size_t>{0, 3, 5, 10}));
  EXPECT_EQ(cuopt::host_copy(cones.cone_dimensions, stream), cone_dimensions);
  EXPECT_EQ(cuopt::host_copy(cones.element_cone_ids, stream),
            (std::vector<int>{0, 0, 0, 1, 1, 2, 2, 2, 2, 2}));
  EXPECT_EQ(cuopt::host_copy(cones.segmented_sum.small_cone_ids, stream),
            (std::vector<int>{0, 1, 2}));
  EXPECT_TRUE(cuopt::host_copy(cones.segmented_sum.medium_cone_ids, stream).empty());
  EXPECT_TRUE(cones.segmented_sum.large_cone_ids.empty());
  EXPECT_TRUE(cones.segmented_sum.large_cone_offsets.empty());
  EXPECT_TRUE(cones.segmented_sum.large_cone_dimensions.empty());

  EXPECT_EQ(cones.eta.size(), 3);
  EXPECT_EQ(cones.w.size(), 10);

  auto& scratch = cones.scratch;
  EXPECT_EQ(scratch.n_cones, cones.n_cones);
  EXPECT_EQ(scratch.n_cone_entries, cones.n_cone_entries);
  EXPECT_EQ(scratch.slots.size(), 3 * cone_dimensions.size());
  EXPECT_EQ(scratch.step_alpha_primal.size(), cone_dimensions.size());
  EXPECT_EQ(scratch.step_alpha_dual.size(), cone_dimensions.size());
  EXPECT_EQ(scratch.temp_cone.size(), x.size());

  EXPECT_EQ(scratch.get_slot<0>().size(), cone_dimensions.size());
  EXPECT_EQ(scratch.get_slot<1>().data(), scratch.get_slot<0>().data() + cones.n_cones);
  EXPECT_EQ(scratch.get_slot<2>().data(), scratch.get_slot<1>().data() + cones.n_cones);
}

TEST(second_order_cone_kernels, segmented_sum_uses_all_cone_size_buckets)
{
  auto stream = rmm::cuda_stream_default;

  std::vector<int> cone_dimensions{65, 3, 66, 32769};
  rmm::device_uvector<double> x(32903, stream);
  rmm::device_uvector<double> z(32903, stream);
  cone_data_t<int, double> cones(cone_dimensions, cuopt::make_span(x), cuopt::make_span(z), stream);

  EXPECT_EQ(cuopt::host_copy(cones.segmented_sum.small_cone_ids, stream), (std::vector<int>{1}));
  EXPECT_EQ(cuopt::host_copy(cones.segmented_sum.medium_cone_ids, stream),
            (std::vector<int>{0, 2}));
  EXPECT_EQ(cones.segmented_sum.large_cone_ids, (std::vector<int>{3}));
  EXPECT_EQ(cones.segmented_sum.large_cone_offsets, (std::vector<std::size_t>{134}));
  EXPECT_EQ(cones.segmented_sum.large_cone_dimensions, (std::vector<int>{32769}));

  std::vector<double> values_host(cones.n_cone_entries, 1.0);
  rmm::device_uvector<double> values(values_host.size(), stream);
  rmm::device_uvector<double> sums(cone_dimensions.size(), stream);
  raft::copy(values.data(), values_host.data(), values_host.size(), stream);

  EXPECT_GT(cones.segmented_sum.cub_workspace_bytes, 0);
  const auto workspace_size = cones.segmented_sum.cub_workspace.size();
  EXPECT_GT(workspace_size, 0);

  cones.segmented_sum(values.data(), cuopt::make_span(sums), stream);

  EXPECT_EQ(cuopt::host_copy(sums, stream), (std::vector<double>{65.0, 3.0, 66.0, 32769.0}));
  EXPECT_EQ(cones.segmented_sum.cub_workspace.size(), workspace_size);
}

TEST(second_order_cone_kernels, nt_scaling_matches_host_reference)
{
  auto stream = rmm::cuda_stream_default;

  std::vector<int> cone_dimensions{3, 65, 32769};
  std::size_t n_cone_entries = 0;
  for (const auto dim : cone_dimensions) {
    n_cone_entries += static_cast<std::size_t>(dim);
  }

  std::vector<double> x_host(n_cone_entries);
  std::vector<double> z_host(n_cone_entries);
  std::size_t offset = 0;
  for (std::size_t cone = 0; cone < cone_dimensions.size(); ++cone) {
    const auto dim = cone_dimensions[cone];
    x_host[offset] = 100.0 + static_cast<double>(cone);
    z_host[offset] = 80.0 + static_cast<double>(cone);
    for (int local_idx = 1; local_idx < dim; ++local_idx) {
      x_host[offset + local_idx] = 0.001 * static_cast<double>((local_idx % 5) + 1);
      z_host[offset + local_idx] = 0.0015 * static_cast<double>((local_idx % 7) + 1);
    }
    offset += static_cast<std::size_t>(dim);
  }

  auto x = cuopt::device_copy(x_host, stream);
  auto z = cuopt::device_copy(z_host, stream);
  cone_data_t<int, double> cones(cone_dimensions, cuopt::make_span(x), cuopt::make_span(z), stream);
  const auto workspace_size = cones.segmented_sum.cub_workspace.size();
  EXPECT_GT(workspace_size, 0);

  launch_nt_scaling(cones, stream);
  EXPECT_EQ(cones.segmented_sum.cub_workspace.size(), workspace_size);

  auto eta_host = cuopt::host_copy(cones.eta, stream);
  auto w_host   = cuopt::host_copy(cones.w, stream);

  std::vector<double> expected_eta(cone_dimensions.size());
  std::vector<double> expected_w(n_cone_entries);

  offset = 0;
  for (std::size_t cone = 0; cone < cone_dimensions.size(); ++cone) {
    const auto dim = cone_dimensions[cone];

    double x_tail_sq = 0.0;
    double z_tail_sq = 0.0;
    for (int local_idx = 1; local_idx < dim; ++local_idx) {
      const auto idx = offset + local_idx;
      x_tail_sq += x_host[idx] * x_host[idx];
      z_tail_sq += z_host[idx] * z_host[idx];
    }

    const auto x_tail_norm = std::sqrt(x_tail_sq);
    const auto z_tail_norm = std::sqrt(z_tail_sq);
    const auto x_det       = (x_host[offset] - x_tail_norm) * (x_host[offset] + x_tail_norm);
    const auto z_det       = (z_host[offset] - z_tail_norm) * (z_host[offset] + z_tail_norm);
    ASSERT_GT(x_det, 0.0) << "cone " << cone;
    ASSERT_GT(z_det, 0.0) << "cone " << cone;

    const auto x_scale = std::sqrt(x_det);
    const auto z_scale = std::sqrt(z_det);

    expected_eta[cone] = std::sqrt(z_scale / x_scale);

    double normalized_xz_dot = 0.0;
    for (int local_idx = 0; local_idx < dim; ++local_idx) {
      const auto idx = offset + local_idx;
      normalized_xz_dot += x_host[idx] * z_host[idx] / (x_scale * z_scale);
    }
    const auto w_det = 2.0 + 2.0 * normalized_xz_dot;
    ASSERT_GT(w_det, 0.0) << "cone " << cone;
    const auto w_scale = std::sqrt(w_det);

    expected_w[offset] = 0.0;
    for (int local_idx = 1; local_idx < dim; ++local_idx) {
      const auto idx  = offset + local_idx;
      expected_w[idx] = (z_host[idx] / z_scale - x_host[idx] / x_scale) / w_scale;
    }

    double normalized_tail_sq = 0.0;
    for (int local_idx = 1; local_idx < dim; ++local_idx) {
      const auto idx = offset + local_idx;
      normalized_tail_sq += expected_w[idx] * expected_w[idx];
    }
    expected_w[offset] = std::sqrt(1.0 + normalized_tail_sq);

    offset += static_cast<std::size_t>(dim);
  }

  for (std::size_t i = 0; i < expected_eta.size(); ++i) {
    EXPECT_NEAR(eta_host[i], expected_eta[i], 1e-10) << "cone " << i;
  }

  for (std::size_t i = 0; i < expected_w.size(); ++i) {
    EXPECT_NEAR(w_host[i], expected_w[i], 1e-10) << "entry " << i;
  }

  offset = 0;
  for (std::size_t cone = 0; cone < cone_dimensions.size(); ++cone) {
    const auto dim = cone_dimensions[cone];

    double tail_sq = 0.0;
    for (int local_idx = 1; local_idx < dim; ++local_idx) {
      const auto idx = offset + local_idx;
      tail_sq += w_host[idx] * w_host[idx];
    }

    EXPECT_NEAR(w_host[offset] * w_host[offset] - tail_sq, 1.0, 1e-10) << "cone " << cone;
    offset += static_cast<std::size_t>(dim);
  }
}

TEST(second_order_cone_kernels, cone_step_length_matches_host_reference)
{
  auto stream = rmm::cuda_stream_default;

  std::vector<int> cone_dimensions{3, 65, 32769};
  std::size_t n_cone_entries = 0;
  for (const auto dim : cone_dimensions) {
    n_cone_entries += static_cast<std::size_t>(dim);
  }

  std::vector<double> x_host(n_cone_entries);
  std::vector<double> z_host(n_cone_entries);
  std::vector<double> dx_host(n_cone_entries);
  std::vector<double> dz_host(n_cone_entries);

  std::size_t offset = 0;
  for (std::size_t cone = 0; cone < cone_dimensions.size(); ++cone) {
    const auto dim = cone_dimensions[cone];

    x_host[offset]  = 12.0 + static_cast<double>(cone);
    z_host[offset]  = 14.0 + static_cast<double>(cone);
    dx_host[offset] = (cone == 0) ? -30.0 : 0.2;
    dz_host[offset] = (cone == 1) ? -25.0 : 0.15;

    for (int local_idx = 1; local_idx < dim; ++local_idx) {
      const auto idx = offset + local_idx;

      x_host[idx]  = 0.001 * static_cast<double>((local_idx % 5) + 1);
      z_host[idx]  = 0.0015 * static_cast<double>((local_idx % 7) + 1);
      dx_host[idx] = 0.02 * static_cast<double>((local_idx % 5) - 2);
      dz_host[idx] = -0.015 * static_cast<double>((local_idx % 7) - 3);
    }

    offset += static_cast<std::size_t>(dim);
  }

  auto compute_expected_step = [&](std::vector<double> const& u,
                                   std::vector<double> const& du,
                                   double alpha_max,
                                   std::vector<double>& per_cone_alpha) {
    auto global_alpha = alpha_max;
    std::size_t off   = 0;

    for (std::size_t cone = 0; cone < cone_dimensions.size(); ++cone) {
      const auto dim = cone_dimensions[cone];

      double du_tail_sq     = 0.0;
      double u_tail_du_tail = 0.0;
      double u_tail_sq      = 0.0;
      for (int local_idx = 1; local_idx < dim; ++local_idx) {
        const auto idx = off + local_idx;
        du_tail_sq += du[idx] * du[idx];
        u_tail_du_tail += u[idx] * du[idx];
        u_tail_sq += u[idx] * u[idx];
      }

      per_cone_alpha[cone] = host_cone_step_length_from_scalars(
        u[off], du[off], du_tail_sq, u_tail_du_tail, u_tail_sq, alpha_max);
      global_alpha = std::min(global_alpha, per_cone_alpha[cone]);

      off += static_cast<std::size_t>(dim);
    }

    return global_alpha;
  };

  constexpr double alpha_max = 0.99;
  std::vector<double> expected_primal_per_cone(cone_dimensions.size());
  std::vector<double> expected_dual_per_cone(cone_dimensions.size());
  const auto expected_primal =
    compute_expected_step(x_host, dx_host, alpha_max, expected_primal_per_cone);
  const auto expected_dual =
    compute_expected_step(z_host, dz_host, alpha_max, expected_dual_per_cone);

  auto x  = cuopt::device_copy(x_host, stream);
  auto z  = cuopt::device_copy(z_host, stream);
  auto dx = cuopt::device_copy(dx_host, stream);
  auto dz = cuopt::device_copy(dz_host, stream);

  cone_data_t<int, double> cones(cone_dimensions, cuopt::make_span(x), cuopt::make_span(z), stream);
  const auto [step_primal, step_dual] =
    compute_cone_step_length(cones,
                             raft::device_span<const double>(dx.data(), dx.size()),
                             raft::device_span<const double>(dz.data(), dz.size()),
                             alpha_max,
                             stream);

  EXPECT_LT(expected_primal, alpha_max);
  EXPECT_LT(expected_dual, alpha_max);
  EXPECT_NEAR(step_primal, expected_primal, 1e-12);
  EXPECT_NEAR(step_dual, expected_dual, 1e-12);

  const auto primal_per_cone = cuopt::host_copy(cones.scratch.step_alpha_primal, stream);
  const auto dual_per_cone   = cuopt::host_copy(cones.scratch.step_alpha_dual, stream);
  for (std::size_t cone = 0; cone < cone_dimensions.size(); ++cone) {
    EXPECT_NEAR(primal_per_cone[cone], expected_primal_per_cone[cone], 1e-12)
      << "primal cone " << cone;
    EXPECT_NEAR(dual_per_cone[cone], expected_dual_per_cone[cone], 1e-12) << "dual cone " << cone;
  }
}

TEST(second_order_cone_kernels, scaling_operators_match_host_reference)
{
  auto stream = rmm::cuda_stream_default;

  std::vector<int> cone_dimensions{3, 65, 32769};
  std::size_t n_cone_entries = 0;
  for (const auto dim : cone_dimensions) {
    n_cone_entries += static_cast<std::size_t>(dim);
  }

  std::vector<double> x_host(n_cone_entries);
  std::vector<double> z_host(n_cone_entries);
  std::vector<double> v_host(n_cone_entries);
  std::vector<double> cone_target_host(n_cone_entries);
  std::vector<double> accum_initial_host(n_cone_entries);
  std::size_t offset = 0;
  for (std::size_t cone = 0; cone < cone_dimensions.size(); ++cone) {
    const auto dim             = cone_dimensions[cone];
    x_host[offset]             = 100.0 + static_cast<double>(cone);
    z_host[offset]             = 80.0 + static_cast<double>(cone);
    v_host[offset]             = 0.75 + 0.1 * static_cast<double>(cone);
    cone_target_host[offset]   = 0.4 + 0.03 * static_cast<double>(cone);
    accum_initial_host[offset] = -0.2 + 0.02 * static_cast<double>(cone);
    for (int local_idx = 1; local_idx < dim; ++local_idx) {
      const auto idx          = offset + local_idx;
      x_host[idx]             = 0.001 * static_cast<double>((local_idx % 5) + 1);
      z_host[idx]             = 0.0015 * static_cast<double>((local_idx % 7) + 1);
      v_host[idx]             = 0.002 * static_cast<double>((local_idx % 11) - 5);
      cone_target_host[idx]   = 0.003 * static_cast<double>((local_idx % 13) - 6);
      accum_initial_host[idx] = 0.004 * static_cast<double>((local_idx % 17) - 8);
    }
    offset += static_cast<std::size_t>(dim);
  }

  auto x           = cuopt::device_copy(x_host, stream);
  auto z           = cuopt::device_copy(z_host, stream);
  auto v           = cuopt::device_copy(v_host, stream);
  auto cone_target = cuopt::device_copy(cone_target_host, stream);
  auto accum       = cuopt::device_copy(accum_initial_host, stream);
  rmm::device_uvector<double> w_out(n_cone_entries, stream);
  rmm::device_uvector<double> w_inv_out(n_cone_entries, stream);
  rmm::device_uvector<double> h_out(n_cone_entries, stream);
  rmm::device_uvector<double> w_tmp(n_cone_entries, stream);
  // apply_w then apply_w on same v: should match apply_hessian (H = W^2 for symmetric NT W).
  rmm::device_uvector<double> w_squared_v(n_cone_entries, stream);
  rmm::device_uvector<double> recovered_dz(n_cone_entries, stream);

  cone_data_t<int, double> cones(cone_dimensions, cuopt::make_span(x), cuopt::make_span(z), stream);
  launch_nt_scaling(cones, stream);

  auto v_span = raft::device_span<const double>(v.data(), v.size());
  apply_w(v_span, cuopt::make_span(w_out), cones, stream);
  apply_w_inv(v_span, cuopt::make_span(w_inv_out), cones, stream);
  apply_hessian(v_span, cuopt::make_span(h_out), cones, stream);
  recover_cone_dz_from_target(
    v_span,
    cones,
    raft::device_span<const double>(cone_target.data(), cone_target.size()),
    cuopt::make_span(recovered_dz),
    stream);
  accumulate_cone_hessian_matvec(v_span, cones, cuopt::make_span(accum), stream);
  apply_w(v_span, cuopt::make_span(w_tmp), cones, stream);
  apply_w(raft::device_span<const double>(w_tmp.data(), w_tmp.size()),
          cuopt::make_span(w_squared_v),
          cones,
          stream);

  auto eta_host          = cuopt::host_copy(cones.eta, stream);
  auto w_host            = cuopt::host_copy(cones.w, stream);
  auto w_out_host        = cuopt::host_copy(w_out, stream);
  auto w_inv_out_host    = cuopt::host_copy(w_inv_out, stream);
  auto h_out_host        = cuopt::host_copy(h_out, stream);
  auto w_squared_v_host  = cuopt::host_copy(w_squared_v, stream);
  auto recovered_dz_host = cuopt::host_copy(recovered_dz, stream);
  auto accum_host        = cuopt::host_copy(accum, stream);

  std::vector<double> expected_w(n_cone_entries);
  std::vector<double> expected_w_inv(n_cone_entries);
  std::vector<double> expected_h(n_cone_entries);
  std::vector<double> expected_h_unscaled(n_cone_entries);

  offset = 0;
  for (std::size_t cone = 0; cone < cone_dimensions.size(); ++cone) {
    const auto dim = cone_dimensions[cone];
    const auto w0  = w_host[offset];
    const auto v0  = v_host[offset];
    const auto eta = eta_host[cone];

    double tail_dot = 0.0;
    for (int local_idx = 1; local_idx < dim; ++local_idx) {
      const auto idx = offset + local_idx;
      tail_dot += w_host[idx] * v_host[idx];
    }

    expected_w[offset]     = eta * (w0 * v0 + tail_dot);
    expected_w_inv[offset] = (w0 * v0 - tail_dot) / eta;

    const auto rho              = w0 * v0 + tail_dot;
    expected_h_unscaled[offset] = (eta * eta) * (2.0 * w0 * rho - v0);
    expected_h[offset]          = expected_h_unscaled[offset];

    for (int local_idx = 1; local_idx < dim; ++local_idx) {
      const auto idx = offset + local_idx;

      expected_w[idx]          = eta * (v_host[idx] + (v0 + tail_dot / (1.0 + w0)) * w_host[idx]);
      expected_w_inv[idx]      = (v_host[idx] + (-v0 + tail_dot / (1.0 + w0)) * w_host[idx]) / eta;
      expected_h_unscaled[idx] = (eta * eta) * (2.0 * w_host[idx] * rho + v_host[idx]);
      expected_h[idx]          = expected_h_unscaled[idx];
    }

    offset += static_cast<std::size_t>(dim);
  }

  for (std::size_t i = 0; i < n_cone_entries; ++i) {
    EXPECT_NEAR(w_out_host[i], expected_w[i], 1e-9) << "W entry " << i;
    EXPECT_NEAR(w_inv_out_host[i], expected_w_inv[i], 1e-9) << "W inverse entry " << i;
    EXPECT_NEAR(h_out_host[i], expected_h[i], 1e-9) << "H entry " << i;
    EXPECT_NEAR(h_out_host[i], w_squared_v_host[i], 1e-9) << "W^2 v vs H v entry " << i;
    EXPECT_NEAR(recovered_dz_host[i], cone_target_host[i] - expected_h_unscaled[i], 1e-9)
      << "recovered dz entry " << i;
    EXPECT_NEAR(accum_host[i], accum_initial_host[i] + expected_h_unscaled[i], 1e-9)
      << "accumulated H entry " << i;
  }
}

TEST(second_order_cone_kernels, combined_cone_rhs_matches_host_reference)
{
  auto stream = rmm::cuda_stream_default;

  std::vector<int> cone_dimensions{3, 65, 32769};
  std::size_t n_cone_entries = 0;
  for (const auto dim : cone_dimensions) {
    n_cone_entries += static_cast<std::size_t>(dim);
  }

  std::vector<double> x_host(n_cone_entries);
  std::vector<double> z_host(n_cone_entries);
  std::vector<double> dx_aff_host(n_cone_entries);
  std::vector<double> dz_aff_host(n_cone_entries);
  std::size_t offset = 0;
  for (std::size_t cone = 0; cone < cone_dimensions.size(); ++cone) {
    const auto dim      = cone_dimensions[cone];
    x_host[offset]      = 120.0 + static_cast<double>(cone);
    z_host[offset]      = 90.0 + static_cast<double>(cone);
    dx_aff_host[offset] = 0.25 + 0.05 * static_cast<double>(cone);
    dz_aff_host[offset] = -0.3 + 0.04 * static_cast<double>(cone);
    for (int local_idx = 1; local_idx < dim; ++local_idx) {
      const auto idx   = offset + local_idx;
      x_host[idx]      = 0.001 * static_cast<double>((local_idx % 5) + 1);
      z_host[idx]      = 0.0015 * static_cast<double>((local_idx % 7) + 1);
      dx_aff_host[idx] = 0.002 * static_cast<double>((local_idx % 11) - 5);
      dz_aff_host[idx] = 0.001 * static_cast<double>((local_idx % 13) - 6);
    }
    offset += static_cast<std::size_t>(dim);
  }

  auto x      = cuopt::device_copy(x_host, stream);
  auto z      = cuopt::device_copy(z_host, stream);
  auto dx_aff = cuopt::device_copy(dx_aff_host, stream);
  auto dz_aff = cuopt::device_copy(dz_aff_host, stream);
  rmm::device_uvector<double> out(n_cone_entries, stream);

  cone_data_t<int, double> cones(cone_dimensions, cuopt::make_span(x), cuopt::make_span(z), stream);
  launch_nt_scaling(cones, stream);

  constexpr double sigma_mu = 0.37;
  compute_combined_cone_rhs_term(raft::device_span<const double>(dx_aff.data(), dx_aff.size()),
                                 raft::device_span<const double>(dz_aff.data(), dz_aff.size()),
                                 cones,
                                 sigma_mu,
                                 cuopt::make_span(out),
                                 stream);

  auto eta_host = cuopt::host_copy(cones.eta, stream);
  auto w_host   = cuopt::host_copy(cones.w, stream);
  auto out_host = cuopt::host_copy(out, stream);

  auto apply_w_ref = [&](std::vector<double> const& v) {
    std::vector<double> result(n_cone_entries);
    std::size_t off = 0;
    for (std::size_t cone = 0; cone < cone_dimensions.size(); ++cone) {
      const auto dim = cone_dimensions[cone];
      const auto w0  = w_host[off];
      const auto v0  = v[off];

      double tail_dot = 0.0;
      for (int local_idx = 1; local_idx < dim; ++local_idx) {
        const auto idx = off + local_idx;
        tail_dot += w_host[idx] * v[idx];
      }

      result[off] = eta_host[cone] * (w0 * v0 + tail_dot);
      for (int local_idx = 1; local_idx < dim; ++local_idx) {
        const auto idx = off + local_idx;
        result[idx]    = eta_host[cone] * (v[idx] + (v0 + tail_dot / (1.0 + w0)) * w_host[idx]);
      }

      off += static_cast<std::size_t>(dim);
    }
    return result;
  };

  auto apply_w_inv_ref = [&](std::vector<double> const& v) {
    std::vector<double> result(n_cone_entries);
    std::size_t off = 0;
    for (std::size_t cone = 0; cone < cone_dimensions.size(); ++cone) {
      const auto dim = cone_dimensions[cone];
      const auto w0  = w_host[off];
      const auto v0  = v[off];

      double tail_dot = 0.0;
      for (int local_idx = 1; local_idx < dim; ++local_idx) {
        const auto idx = off + local_idx;
        tail_dot += w_host[idx] * v[idx];
      }

      result[off] = (w0 * v0 - tail_dot) / eta_host[cone];
      for (int local_idx = 1; local_idx < dim; ++local_idx) {
        const auto idx = off + local_idx;
        result[idx]    = (v[idx] + (-v0 + tail_dot / (1.0 + w0)) * w_host[idx]) / eta_host[cone];
      }

      off += static_cast<std::size_t>(dim);
    }
    return result;
  };

  // Same order as compute_combined_cone_rhs_term: apply_w(dx_aff), apply_w_inv(dz_aff).
  auto scaled_dx = apply_w_ref(dx_aff_host);
  auto scaled_dz = apply_w_inv_ref(dz_aff_host);
  auto nt_point  = apply_w_inv_ref(z_host);

  std::vector<double> shift(n_cone_entries);
  offset = 0;
  for (std::size_t cone = 0; cone < cone_dimensions.size(); ++cone) {
    const auto dim = cone_dimensions[cone];

    double head_dot = 0.0;
    for (int local_idx = 0; local_idx < dim; ++local_idx) {
      const auto idx = offset + local_idx;
      head_dot += scaled_dx[idx] * scaled_dz[idx];
    }

    shift[offset] = head_dot - sigma_mu;
    for (int local_idx = 1; local_idx < dim; ++local_idx) {
      const auto idx = offset + local_idx;
      shift[idx]     = scaled_dx[offset] * scaled_dz[idx] + scaled_dz[offset] * scaled_dx[idx];
    }

    offset += static_cast<std::size_t>(dim);
  }

  std::vector<double> minus_p(n_cone_entries);
  offset = 0;
  for (std::size_t cone = 0; cone < cone_dimensions.size(); ++cone) {
    const auto dim     = cone_dimensions[cone];
    const auto lambda0 = nt_point[offset];

    double lambda_tail_dot = 0.0;
    double lambda_tail_sq  = 0.0;
    for (int local_idx = 1; local_idx < dim; ++local_idx) {
      const auto idx = offset + local_idx;
      lambda_tail_dot += nt_point[idx] * shift[idx];
      lambda_tail_sq += nt_point[idx] * nt_point[idx];
    }

    const auto lambda_tail_norm = std::sqrt(lambda_tail_sq);
    const auto det_lambda       = (lambda0 - lambda_tail_norm) * (lambda0 + lambda_tail_norm);
    ASSERT_GT(lambda0, 0.0) << "cone " << cone;
    ASSERT_GT(det_lambda, 0.0) << "cone " << cone;

    const auto p_head = (lambda0 * shift[offset] - lambda_tail_dot) / det_lambda;
    minus_p[offset]   = -p_head;
    for (int local_idx = 1; local_idx < dim; ++local_idx) {
      const auto idx = offset + local_idx;
      minus_p[idx]   = (p_head * nt_point[idx] - shift[idx]) / lambda0;
    }

    offset += static_cast<std::size_t>(dim);
  }

  auto expected = apply_w_ref(minus_p);
  for (std::size_t i = 0; i < n_cone_entries; ++i) {
    EXPECT_NEAR(out_host[i], expected[i], 1e-8) << "entry " << i;
  }
}

}  // namespace cuopt::linear_programming::dual_simplex::test
