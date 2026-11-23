/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <barrier/augmented_matvec.cuh>
#include <barrier/iterative_refinement.hpp>
#include <linear_algebra/vector_math.cuh>
#include <utilities/copy_helpers.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cub/device/device_transform.cuh>

#include <optional>
#include <vector>

namespace cuopt::mathematical_optimization::simplex::test {

TEST(barrier, fused_augmented_matvec_preserves_segments_and_free_variables)
{
  raft::handle_t handle;
  const auto stream = handle.get_stream();
  constexpr int n = 257, m = 513, linear_n = 250;
  for (int p : {0, 3}) {
    const int size        = n + m + p;
    const int x2_offset   = n;
    const int y1_offset   = n + m;
    const int y2_offset   = 2 * n + m;
    const int r1_offset   = 2 * n + 2 * m;
    const int exp_offset  = 3 * n + 2 * m;
    const int orig_offset = exp_offset + p;
    std::vector<double> x(size), y(size), diag(n, 2.0), scratch(orig_offset + p, -99.0);
    std::vector<int> free_linear(n);
    for (int i = 0; i < size; ++i) {
      x[i] = i + 1;
      y[i] = -i - 2;
    }
    for (int i = 0; i < n; ++i) {
      free_linear[i] = i % 3 == 0;
    }
    rmm::device_uvector<double> dx(size, stream), dy(size, stream), dd(n, stream);
    rmm::device_uvector<double> work(scratch.size(), stream);
    rmm::device_uvector<int> mask(n, stream);
    raft::copy(dx.data(), x.data(), size, stream);
    raft::copy(dy.data(), y.data(), size, stream);
    raft::copy(dd.data(), diag.data(), n, stream);
    raft::copy(mask.data(), free_linear.data(), n, stream);
    raft::copy(work.data(), scratch.data(), scratch.size(), stream);
    auto* w = work.data();
    barrier::prepare_augmented_matvec<int, double>
      <<<3, 256, 0, stream.get()>>>(raft::device_span<const double>(dx.data(), size),
                                    raft::device_span<const double>(dy.data(), size),
                                    raft::device_span<const double>(dd.data(), n),
                                    raft::device_span<const int>(mask.data(), n),
                                    raft::device_span<double>(w, n),
                                    raft::device_span<double>(w + x2_offset, m),
                                    raft::device_span<double>(w + y1_offset, n),
                                    raft::device_span<double>(w + y2_offset, m),
                                    raft::device_span<double>(w + r1_offset, n),
                                    raft::device_span<double>(w + exp_offset, p),
                                    raft::device_span<double>(w + orig_offset, p),
                                    linear_n);
    RAFT_CUDA_TRY(cudaGetLastError());
    raft::copy(scratch.data(), w, scratch.size(), stream);
    stream.sync();
    for (int i = 0; i < n; ++i) {
      EXPECT_DOUBLE_EQ(scratch[i], x[i]);
      EXPECT_DOUBLE_EQ(scratch[y1_offset + i], y[i]);
      EXPECT_DOUBLE_EQ(scratch[r1_offset + i], i < linear_n && !free_linear[i] ? 2.0 * x[i] : 0.0);
    }
    for (int i = 0; i < m; ++i) {
      EXPECT_DOUBLE_EQ(scratch[x2_offset + i], x[n + i]);
      EXPECT_DOUBLE_EQ(scratch[y2_offset + i], y[n + i]);
    }
    for (int i = 0; i < p; ++i) {
      EXPECT_DOUBLE_EQ(scratch[exp_offset + i], 0.0);
      EXPECT_DOUBLE_EQ(scratch[orig_offset + i], y[n + m + i]);
      scratch[exp_offset + i] = i + 5;
    }
    raft::copy(w, scratch.data(), scratch.size(), stream);
    barrier::finish_augmented_matvec<int, double>
      <<<3, 256, 0, stream.get()>>>(raft::device_span<double>(dy.data(), size),
                                    raft::device_span<const double>(w + y1_offset, n),
                                    raft::device_span<const double>(w + y2_offset, m),
                                    raft::device_span<const double>(w + exp_offset, p),
                                    raft::device_span<const double>(w + orig_offset, p),
                                    2.0,
                                    -3.0);
    RAFT_CUDA_TRY(cudaGetLastError());
    std::vector<double> result(size);
    raft::copy(result.data(), dy.data(), size, stream);
    stream.sync();
    for (int i = 0; i < n + m; ++i) {
      EXPECT_DOUBLE_EQ(result[i], y[i]);
    }
    for (int i = 0; i < p; ++i) {
      EXPECT_DOUBLE_EQ(result[n + m + i], 2.0 * (i + 5) - 3.0 * y[n + m + i]);
    }
  }
}

TEST(barrier, device_arnoldi_column_matches_sequential_orthogonalization)
{
  raft::handle_t handle;
  const auto stream = handle.get_stream();
  constexpr int n   = 257;
  barrier::gmres_workspace_t<double> workspace(n, stream);
  for (int k : {0, 2}) {
    std::vector<std::vector<double>> basis(k + 1, std::vector<double>(n));
    std::vector<double> expected(n), actual(n), z(n, 2.0);
    for (int i = 0; i < n; ++i) {
      expected[i] = 0.1 + 0.01 * (i % 17);
    }
    raft::copy(workspace.V[k + 1].data(), expected.data(), n, stream);
    raft::copy(workspace.Z[k].data(), z.data(), n, stream);
    std::vector<double> dots(k + 1);
    for (int j = 0; j <= k; ++j) {
      for (int i = 0; i < n; ++i) {
        basis[j][i] = 0.01 * (1 + ((i + j) % 7));
      }
      raft::copy(workspace.V[j].data(), basis[j].data(), n, stream);
      for (int i = 0; i < n; ++i) {
        dots[j] += expected[i] * basis[j][i];
      }
      for (int i = 0; i < n; ++i) {
        expected[i] -= dots[j] * basis[j][i];
      }
    }
    workspace.check_preconditioner_async(workspace.Z[k]);
    workspace.orthogonalize(k);
    EXPECT_DOUBLE_EQ(workspace.h_column[workspace.dimension + 1], 2.0);
    for (int j = 0; j <= k; ++j) {
      EXPECT_NEAR(workspace.h_column[j], dots[j], 1e-12);
    }
    double norm_squared = 0;
    for (double value : expected) {
      norm_squared += value * value;
    }
    EXPECT_NEAR(workspace.h_column[k + 1], norm_squared, 1e-12);
    raft::copy(actual.data(), workspace.V[k + 1].data(), n, stream);
    stream.sync();
    for (int i = 0; i < n; ++i) {
      EXPECT_NEAR(actual[i], expected[i], 1e-12);
    }
  }
}

struct refinement_test_operator_t {
  struct data_t {
    raft::handle_t const* handle_ptr;
    std::optional<barrier::gmres_workspace_t<double>> gmres_workspace_;
    data_t(raft::handle_t const* handle, size_t n)
      : handle_ptr(handle), gmres_workspace_(std::in_place, n, handle->get_stream())
    {
    }
  } data_;

  refinement_test_operator_t(raft::handle_t const* handle, size_t n) : data_(handle, n) {}

  void a_multiply(double alpha,
                  const rmm::device_uvector<double>& x,
                  double beta,
                  rmm::device_uvector<double>& y)
  {
    const double* xp = x.data();
    double* yp       = y.data();
    cub::DeviceTransform::Transform(
      thrust::make_counting_iterator<int>(0),
      y.data(),
      y.size(),
      [xp, yp, alpha, beta] __device__(int i) -> double {
        return alpha * (1.0 + 0.01 * (i % 7)) * xp[i] + beta * yp[i];
      },
      data_.handle_ptr->get_stream().get());
  }

  void solve(rmm::device_uvector<double>& b, rmm::device_uvector<double>& x)
  {
    cub::DeviceTransform::Transform(
      b.data(),
      x.data(),
      b.size(),
      [] __device__(double value) -> double { return value / 0.25; },
      data_.handle_ptr->get_stream().get());
  }
};

TEST(barrier, refinement_reuses_workspace_and_checks_true_residual)
{
  raft::handle_t handle;
  constexpr int n = 257;
  // Solve two right-hand sides with the same fixed-size GMRES workspace.
  refinement_test_operator_t op(&handle, n);
  const auto* saved_r = op.data_.gmres_workspace_->r.data();
  const auto* saved_v = op.data_.gmres_workspace_->V[0].data();
  for (double rhs_value : {1.0, 2.0}) {
    std::vector<double> rhs(n, rhs_value), zero(n, 0.0), result(n);
    auto b             = cuopt::device_copy(rhs, handle.get_stream());
    auto x             = cuopt::device_copy(zero, handle.get_stream());
    const double error = barrier::iterative_refinement<int, double>(op, b, x, 1e-10);
    EXPECT_LE(error, 1e-10);
    EXPECT_EQ(op.data_.gmres_workspace_->r.data(), saved_r);
    EXPECT_EQ(op.data_.gmres_workspace_->V[0].data(), saved_v);
    raft::copy(result.data(), x.data(), n, handle.get_stream());
    handle.get_stream().sync();
    for (int i = 0; i < n; ++i) {
      EXPECT_NEAR((1.0 + 0.01 * (i % 7)) * result[i], rhs[i], 1e-10);
    }
  }
}

TEST(barrier, gmres_update_handles_all_counts_and_in_place_reuse)
{
  raft::handle_t handle;
  const auto stream       = handle.get_stream();
  constexpr int n         = 3;
  constexpr int dimension = barrier::gmres_workspace_t<double>::dimension;
  barrier::gmres_workspace_t<double> workspace(n, stream);
  rmm::device_uvector<double> d_x(n, stream);
  // Z[j] = (j + 1) * {1, 2, 3}, with alternating +1/-1 coefficients.
  // For example, count=2 adds {-1, -2, -3} to x on each update.
  const std::vector<double> h_initial{10.0, 20.0, 30.0};
  std::vector<double> h_actual(n), h_coefficients(dimension);
  std::vector<std::vector<double>> h_basis(dimension);
  for (int j = 0; j < dimension; ++j) {
    h_coefficients[j] = j % 2 == 0 ? 1.0 : -1.0;
    h_basis[j]        = {1.0 * (j + 1), 2.0 * (j + 1), 3.0 * (j + 1)};
    raft::copy(workspace.Z[j].data(), h_basis[j].data(), n, stream);
  }
  double delta = 0.0;
  for (int count = 0; count <= dimension; ++count) {
    SCOPED_TRACE(count);
    if (count > 0) { delta += h_coefficients[count - 1] * count; }
    raft::copy(d_x.data(), h_initial.data(), n, stream);
    for (int update = 1; update <= 2; ++update) {
      SCOPED_TRACE(update);
      workspace.update(d_x, h_coefficients, count);
      RAFT_CUDA_TRY(cudaGetLastError());
      raft::copy(h_actual.data(), d_x.data(), n, stream);
      stream.sync();
      for (int i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(h_actual[i], h_initial[i] + update * delta * (i + 1));
      }
    }
  }
}

struct reduction_test_square_op {
  __host__ __device__ double operator()(double value) const { return value * value; }
};

TEST(barrier, shared_reductions_reuse_storage_and_preserve_async_outputs)
{
  raft::handle_t handle;
  const auto stream = handle.get_stream();
  reduction_workspace_t<double> reductions(stream);
  const std::vector<double> host_values{-4.0, 3.0};
  auto values = cuopt::device_copy(host_values, stream);
  rmm::device_uvector<double> empty(0, stream), outputs(2, stream);

  EXPECT_DOUBLE_EQ(vector_norm_inf(values), 4.0);
  EXPECT_DOUBLE_EQ((device_vector_norm_inf<int, double>(values, stream)), 4.0);
  EXPECT_DOUBLE_EQ((vector_norm_inf<int, double>(host_values, stream)), 4.0);
  const auto squares = thrust::make_transform_iterator(values.data(), reduction_test_square_op{});
  EXPECT_DOUBLE_EQ((device_custom_vector_norm_inf<int, double>(squares, values.size(), stream)),
                   16.0);
  EXPECT_DOUBLE_EQ((device_custom_vector_norm_inf<int, double>(squares, 0, stream)), 0.0);
  EXPECT_DOUBLE_EQ(vector_norm_inf(empty), 0.0);

  EXPECT_DOUBLE_EQ(vector_norm_inf(values, reductions), 4.0);
  EXPECT_DOUBLE_EQ(vector_norm2(values, reductions), 5.0);
  EXPECT_DOUBLE_EQ(vector_norm_inf(empty, reductions), 0.0);
  EXPECT_DOUBLE_EQ(vector_norm2(empty, reductions), 0.0);

  std::vector<double> results(2);
  const void* scratch = nullptr;
  size_t capacity     = 0;
  for (bool is_empty : {false, true, false}) {
    SCOPED_TRACE(is_empty);
    const auto& input = is_empty ? empty : values;
    // Queue both caller-owned outputs before reading either one back.
    vector_norm_inf_async(raft::device_span<const double>(input.data(), input.size()),
                          raft::device_span<double>(outputs.data(), 1),
                          reductions,
                          stream);
    reductions.transform_reduce_async(input.data(),
                                      thrust::plus<double>{},
                                      reduction_test_square_op{},
                                      5.0,
                                      input.size(),
                                      raft::device_span<double>(outputs.data() + 1, 1),
                                      stream);
    raft::copy(results.data(), outputs.data(), results.size(), stream);
    stream.sync();
    EXPECT_DOUBLE_EQ(results[0], is_empty ? 0.0 : 4.0);
    EXPECT_DOUBLE_EQ(results[1], is_empty ? 5.0 : 30.0);

    if (scratch == nullptr) {
      // Warm both reduction operators before checking scratch reuse.
      scratch  = reductions.buffer_data.data();
      capacity = reductions.buffer_data.capacity();
      ASSERT_NE(scratch, nullptr);
    } else {
      EXPECT_EQ(reductions.buffer_data.data(), scratch);
      EXPECT_EQ(reductions.buffer_data.capacity(), capacity);
    }
  }
}

}  // namespace cuopt::mathematical_optimization::simplex::test
