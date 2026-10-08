/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>
#include <barrier/augmented_matvec.cuh>

#include <cuopt/mathematical_optimization/constants.h>
#include <barrier/iterative_refinement.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <cuopt/mathematical_optimization/utilities/barrier_cache.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/scaling.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/user_problem.hpp>
#include <linear_algebra/sparse_matrix.hpp>
#include <linear_algebra/vector_math.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>

#include <cub/device/device_transform.cuh>

#include <cmath>
#include <limits>
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
  double preconditioner;

  refinement_test_operator_t(raft::handle_t const* handle, size_t n, double preconditioner_value)
    : data_(handle, n), preconditioner(preconditioner_value)
  {
  }

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
    const double scale = preconditioner;
    cub::DeviceTransform::Transform(
      b.data(),
      x.data(),
      b.size(),
      [scale] __device__(double value) -> double { return value / scale; },
      data_.handle_ptr->get_stream().get());
  }
};

TEST(barrier, refinement_reuses_workspace_and_checks_true_residual)
{
  raft::handle_t handle;
  // Each linear system owns fixed-size storage reused for different right-hand sides.
  for (double preconditioner : {1.0, 0.25}) {
    for (int n : {128, 257}) {
      refinement_test_operator_t op(&handle, n, preconditioner);
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
  }
}

// This serves as both a warm up but also a mandatory initial call to setup cuSparse and cuBLAS
static void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream().get()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(handle_ptr->get_cusparse_handle(),
                                                                 CUSPARSE_POINTER_MODE_DEVICE,
                                                                 handle_ptr->get_stream().get()));
}

// Hub-and-spoke style chain: two free integrator variables plus a quadratic on w.
//
// minimize  0.5 w^2
// s.t.      y1 + w     = 1
//           -y1 + y2   = 0
//           -y2 + w    = 1
//            t  - w    = 0
//           (t, u) in Q^2
//
// Unique primal: w = t = 1, y1 = y2 = u = 0.
static user_problem_t<int, double> make_free_substitution_qp(raft::handle_t* handle)
{
  user_problem_t<int, double> user_problem(handle);

  constexpr int m  = 4;
  constexpr int n  = 5;
  constexpr int nz = 8;

  user_problem.num_rows = m;
  user_problem.num_cols = n;
  user_problem.objective.assign(n, 0.0);

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Columns: y1, y2, w, t, u
  user_problem.A.col_start = {0, 2, 4, 7, 8, 8};
  user_problem.A.i         = {0, 1, 1, 2, 0, 2, 3, 3};
  user_problem.A.x         = {1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0};

  user_problem.rhs       = {1.0, 0.0, 1.0, 0.0};
  user_problem.row_sense = {'E', 'E', 'E', 'E'};
  // Keep y1, y2, and w free so bound strengthening cannot pin the integrator
  // chain before substitution. w is skipped later because it appears in Q.
  user_problem.lower = {-inf, -inf, -inf, 0.0, 0.0};
  user_problem.upper.assign(n, inf);

  user_problem.Q_offsets = {0, 0, 0, 1, 1, 1};
  user_problem.Q_indices = {2};
  user_problem.Q_values  = {1.0};

  user_problem.num_range_rows         = 0;
  user_problem.problem_name           = "free_substitution_qp";
  user_problem.cone_var_start         = 3;
  user_problem.second_order_cone_dims = {2};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);
  return user_problem;
}

static void dual_residual(const lp_problem_t<int, double>& lp,
                          const std::vector<double>& x,
                          const std::vector<double>& y,
                          const std::vector<double>& z,
                          std::vector<double>& residual)
{
  residual = z;
  for (int j = 0; j < lp.num_cols; ++j) {
    residual[j] -= lp.objective[j];
  }
  if (lp.Q.n > 0) { matrix_vector_multiply(lp.Q, -1.0, x, 1.0, residual); }
  matrix_transpose_vector_multiply(lp.A, 1.0, y, 1.0, residual);
}

TEST(barrier, cone_metadata_reindexed_when_slack_is_inserted_before_cones)
{
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m       = 1;
  constexpr int n       = 5;
  constexpr int nz      = 5;
  user_problem.num_rows = m;
  user_problem.num_cols = n;
  user_problem.objective.assign(n, 0.0);
  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start.resize(n + 1);
  for (int j = 0; j < n; ++j) {
    user_problem.A.col_start[j] = j;
    user_problem.A.i[j]         = 0;
    user_problem.A.x[j]         = 1.0;
  }
  user_problem.A.col_start[n] = nz;
  user_problem.rhs            = {1.0};
  user_problem.row_sense      = {'L'};
  user_problem.lower.assign(n, 0.0);
  user_problem.upper.assign(n, inf);
  user_problem.num_range_rows         = 0;
  user_problem.second_order_cone_dims = {2, 2};
  user_problem.cone_var_start         = 1;

  simplex_solver_settings_t<int, double> settings;
  settings.barrier       = true;
  settings.dualize       = 0;
  settings.scale_columns = false;

  std::vector<int> new_slacks;
  dualize_info_t<int, double> dualize_info;
  lp_problem_t<int, double> original_lp(user_problem.handle_ptr, 1, 1, 1);
  convert_user_problem(user_problem, settings, original_lp, new_slacks, dualize_info);

  ASSERT_EQ(new_slacks.size(), 1);
  EXPECT_EQ(new_slacks[0], 1);
  EXPECT_EQ(original_lp.num_cols, 6);
  EXPECT_EQ(original_lp.second_order_cone_dims, user_problem.second_order_cone_dims);
  EXPECT_EQ(original_lp.cone_var_start, 2);

  lp_problem_t<int, double> barrier_lp(user_problem.handle_ptr,
                                       original_lp.num_rows,
                                       original_lp.num_cols,
                                       original_lp.A.col_start[original_lp.num_cols]);
  std::vector<double> column_scales;
  std::vector<double> row_scales;
  scaling(original_lp, settings, barrier_lp, column_scales, row_scales);

  EXPECT_EQ(barrier_lp.second_order_cone_dims, user_problem.second_order_cone_dims);
  EXPECT_EQ(barrier_lp.cone_var_start, 2);
}

TEST(barrier, range_and_less_rows_with_soc_block)
{
  // Variables ordered as [a, b | t, u, v], where (t, u, v) \in Q^3.
  //
  // minimize   -a + t
  // subject to a + b <= 3        (<= row)
  //            b - u  = 0
  //            1 <= b <= 2       (range row)
  //            (t, u, v) in Q^3
  //
  // The <= row and the range row each add a slack column, and both must land in front of the
  // cone block. With v = 0 and t = u = b the objective is -a + b, and a <= 3 - b gives
  // -3 + 2b, minimized at the range row's lower end b = 1.
  //
  // Optimal: a* = 2, b* = 1, t* = 1, u* = 1, v* = 0, obj* = -1.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 3;
  constexpr int n  = 5;
  constexpr int nz = 5;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {-1.0, 0.0, 1.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Columns: a, b, t, u, v
  user_problem.A.col_start = {0, 1, 4, 4, 5, 5};
  user_problem.A.i         = {0, 0, 1, 2, 1};
  user_problem.A.x         = {1.0, 1.0, 1.0, 1.0, -1.0};

  // Range rows arrive as 'E' rows with rhs = lower bound and range = upper - lower.
  user_problem.rhs       = {3.0, 0.0, 1.0};
  user_problem.row_sense = {'L', 'E', 'E'};
  user_problem.lower.assign(n, 0.0);
  user_problem.upper.assign(n, inf);

  user_problem.num_range_rows         = 1;
  user_problem.range_rows             = {2};
  user_problem.range_value            = {1.0};
  user_problem.problem_name           = "range_and_less_rows_with_soc_block";
  user_problem.cone_var_start         = 2;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, -1.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 2.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[3], 1.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[4]), 0.0, 1e-4);
}

TEST(barrier, qp_with_range_and_less_rows_and_soc_block)
{
  // Variables ordered as [l | t, u, v], where (t, u, v) \in Q^3.
  //
  // minimize   0.5 l^2 + 0.5 t^2
  // subject to l - u <= -1       (<= row)
  //            2 <= l + u <= 4   (range row)
  //            (t, u, v) in Q^3
  //
  // Q has an entry on the cone head t, so it has to move along with the cone block when the two
  // slack columns are inserted in front of it. Since t >= |u|, v = 0 and t = u at the optimum,
  // leaving 0.5 l^2 + 0.5 u^2. The unconstrained point l = u = 1 violates l - u <= -1, so the
  // optimum sits where u - l = 1 meets the range row's lower end l + u = 2 (multipliers 1 and
  // 0.5, both strictly positive).
  //
  // Optimal: l* = 0.5, t* = 1.5, u* = 1.5, v* = 0, obj* = 1.25.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 2;
  constexpr int n  = 4;
  constexpr int nz = 4;

  user_problem.num_rows = m;
  user_problem.num_cols = n;
  user_problem.objective.assign(n, 0.0);

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Columns: l, t, u, v
  user_problem.A.col_start = {0, 2, 2, 4, 4};
  user_problem.A.i         = {0, 1, 0, 1};
  user_problem.A.x         = {1.0, 1.0, -1.0, 1.0};

  user_problem.rhs       = {-1.0, 2.0};
  user_problem.row_sense = {'L', 'E'};
  user_problem.lower.assign(n, 0.0);
  user_problem.upper.assign(n, inf);

  user_problem.Q_offsets = {0, 1, 2, 2, 2};
  user_problem.Q_indices = {0, 1};
  user_problem.Q_values  = {1.0, 1.0};

  user_problem.num_range_rows         = 1;
  user_problem.range_rows             = {1};
  user_problem.range_value            = {2.0};
  user_problem.problem_name           = "qp_with_range_and_less_rows_and_soc_block";
  user_problem.cone_var_start         = 1;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 1.25, 1e-4);
  EXPECT_NEAR(solution.x[0], 0.5, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.5, 1e-4);
  EXPECT_NEAR(solution.x[2], 1.5, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[3]), 0.0, 1e-4);
}

TEST(barrier, presolve_reindexes_cone_start_after_empty_column_removal)
{
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 1;
  constexpr int n  = 4;
  constexpr int nz = 3;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {1.0, 0.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 0, 1, 2, 3};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 0;
  user_problem.A.x[1]      = -1.0;
  user_problem.A.i[2]      = 0;
  user_problem.A.x[2]      = 0.5;

  user_problem.rhs       = {1.0};
  user_problem.row_sense = {'E'};
  user_problem.lower.assign(n, 0.0);
  user_problem.upper.assign(n, inf);
  user_problem.num_range_rows         = 0;
  user_problem.cone_var_start         = 1;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;
  settings.scale_columns    = false;

  std::vector<int> new_slacks;
  dualize_info_t<int, double> dualize_info;
  lp_problem_t<int, double> original_lp(user_problem.handle_ptr, 1, 1, 1);
  convert_user_problem(user_problem, settings, original_lp, new_slacks, dualize_info);

  presolve_info_t<int, double> presolve_info;
  lp_problem_t<int, double> presolved_lp(user_problem.handle_ptr, 1, 1, 1);
  ASSERT_EQ(presolve(original_lp, settings, presolved_lp, presolve_info), 0);

  EXPECT_EQ(presolved_lp.num_cols, 3);
  EXPECT_EQ(presolved_lp.second_order_cone_dims, std::vector<int>({3}));
  EXPECT_EQ(presolved_lp.cone_var_start, 0);

  lp_problem_t<int, double> barrier_lp(user_problem.handle_ptr,
                                       presolved_lp.num_rows,
                                       presolved_lp.num_cols,
                                       presolved_lp.A.col_start[presolved_lp.num_cols]);
  std::vector<double> column_scales;
  std::vector<double> row_scales;
  ASSERT_EQ(scaling(presolved_lp, settings, barrier_lp, column_scales, row_scales), 0);
  EXPECT_EQ(barrier_lp.cone_var_start, 0);
}

TEST(barrier, presolve_keeps_direct_free_variables_before_cones)
{
  // Layout: [x0, x1 | cone x2, x3, x4] with x0, x1 free and a 3-dimensional SOC block.
  // Free linear columns are not split into v - w. Zero-cost free x0 is substituted from
  // the singleton equality (the pivot may contain cone columns); the leftover free x1
  // then has an empty column and is fixed at 0. The cone block stays trailing.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 1;
  constexpr int n  = 5;
  constexpr int nz = 5;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 0.0, 0.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 1, 2, 3, 4, 5};
  for (int j = 0; j < n; ++j) {
    user_problem.A.i[j] = 0;
    user_problem.A.x[j] = 1.0;
  }

  user_problem.rhs       = {1.0};
  user_problem.row_sense = {'E'};
  user_problem.lower     = {-inf, -inf, 0.0, 0.0, 0.0};
  user_problem.upper.assign(n, inf);
  user_problem.num_range_rows         = 0;
  user_problem.cone_var_start         = 2;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;
  settings.scale_columns    = false;

  std::vector<int> new_slacks;
  dualize_info_t<int, double> dualize_info;
  lp_problem_t<int, double> original_lp(user_problem.handle_ptr, 1, 1, 1);
  convert_user_problem(user_problem, settings, original_lp, new_slacks, dualize_info);

  presolve_info_t<int, double> presolve_info;
  lp_problem_t<int, double> presolved_lp(user_problem.handle_ptr, 1, 1, 1);
  ASSERT_EQ(presolve(original_lp, settings, presolved_lp, presolve_info), 0);

  EXPECT_EQ(presolved_lp.num_rows, 0);
  EXPECT_EQ(presolved_lp.num_cols, 3);
  EXPECT_EQ(presolved_lp.cone_var_start, 0);
  EXPECT_EQ(presolved_lp.second_order_cone_dims, std::vector<int>({3}));
  EXPECT_TRUE(presolve_info.free_variable_pairs.empty());
  EXPECT_TRUE(presolve_info.direct_free_variables.empty());
  ASSERT_EQ(presolve_info.free_variable_eliminations.size(), 2u);
  EXPECT_EQ(presolve_info.free_variable_eliminations[0].variable, 0);
  EXPECT_EQ(presolve_info.free_variable_eliminations[0].pivot_row, 0);
  EXPECT_EQ(presolve_info.free_variable_eliminations[1].variable, 1);
  EXPECT_EQ(presolve_info.free_variable_eliminations[1].pivot_row, -1);
  ASSERT_EQ(presolve_info.free_elimination_remaining_variables.size(), 3u);
  EXPECT_EQ(presolve_info.free_elimination_remaining_variables[0], 2);
  EXPECT_EQ(presolve_info.free_elimination_remaining_variables[1], 3);
  EXPECT_EQ(presolve_info.free_elimination_remaining_variables[2], 4);
}

TEST(barrier, presolve_skips_free_elimination_on_unstable_pivot)
{
  // Layout: [x0, x1 | cone x2, x3, x4] with only x1 free and zero-cost. Both rows holding x1
  // carry it with a 1e-10 coefficient next to O(1) entries, so substituting it would scatter
  // the pivot row amplified by 1e10. Every candidate pivot must fail the row threshold test
  // and x1 must survive as a direct free variable.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m       = 2;
  constexpr int n       = 5;
  constexpr int nz      = 6;
  constexpr double tiny = 1e-10;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 0.0, 0.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // x0 + tiny*x1 + x2 = 1, tiny*x1 + x3 + x4 = 1
  user_problem.A.col_start               = {0, 1, 3, 4, 5, 6};
  const std::vector<int> rows_of_entries = {0, 0, 1, 0, 1, 1};
  const std::vector<double> entry_values = {1.0, tiny, tiny, 1.0, 1.0, 1.0};
  for (int p = 0; p < nz; ++p) {
    user_problem.A.i[p] = rows_of_entries[p];
    user_problem.A.x[p] = entry_values[p];
  }

  user_problem.rhs       = {1.0, 1.0};
  user_problem.row_sense = {'E', 'E'};
  user_problem.lower     = {0.0, -inf, 0.0, 0.0, 0.0};
  user_problem.upper.assign(n, inf);
  user_problem.num_range_rows         = 0;
  user_problem.cone_var_start         = 2;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;
  settings.scale_columns    = false;

  std::vector<int> new_slacks;
  dualize_info_t<int, double> dualize_info;
  lp_problem_t<int, double> original_lp(user_problem.handle_ptr, 1, 1, 1);
  convert_user_problem(user_problem, settings, original_lp, new_slacks, dualize_info);

  presolve_info_t<int, double> presolve_info;
  lp_problem_t<int, double> presolved_lp(user_problem.handle_ptr, 1, 1, 1);
  ASSERT_EQ(presolve(original_lp, settings, presolved_lp, presolve_info), 0);

  EXPECT_EQ(presolved_lp.num_rows, m);
  EXPECT_EQ(presolved_lp.num_cols, n);
  EXPECT_EQ(presolved_lp.cone_var_start, 2);
  EXPECT_TRUE(presolve_info.free_variable_eliminations.empty());
  ASSERT_EQ(presolve_info.direct_free_variables.size(), 1u);
  EXPECT_EQ(presolve_info.direct_free_variables[0], 1);
}

TEST(barrier, rejects_middle_cone_input_before_barrier)
{
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 3;
  constexpr int n  = 5;
  constexpr int nz = 3;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {1.0, 0.0, 0.0, 0.0, 1.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 1, 1, 2, 2, 3};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 1;
  user_problem.A.x[1]      = 1.0;
  user_problem.A.i[2]      = 2;
  user_problem.A.x[2]      = 1.0;

  user_problem.rhs       = {2.0, 1.0, 3.0};
  user_problem.row_sense = {'E', 'E', 'E'};
  user_problem.lower.assign(n, 0.0);
  user_problem.upper.assign(n, inf);
  user_problem.num_range_rows         = 0;
  user_problem.cone_var_start         = 1;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier = true;
  settings.dualize = 0;
  lp_solution_t<int, double> solution(m, n);

  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);
  EXPECT_EQ(status, lp_status_t::NUMERICAL_ISSUES);
}

TEST(barrier, socp_min_x0_subject_to_norm_constraint)
{
  // minimize x_0
  // subject to x_1 = 1
  //            (x_0, x_1, x_2) in Q^3
  //
  // Optimal: x* = (1, 1, 0), obj* = 1

  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 1;
  constexpr int n  = 3;
  constexpr int nz = 1;

  user_problem.num_rows = m;
  user_problem.num_cols = n;

  user_problem.objective = {1.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 0, 1, 1};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;

  user_problem.rhs       = {1.0};
  user_problem.row_sense = {'E'};

  user_problem.lower = {0.0, 0.0, 0.0};
  user_problem.upper = {inf, inf, inf};

  user_problem.num_range_rows = 0;
  user_problem.problem_name   = "socp_norm_cone";

  user_problem.cone_var_start         = 0;
  user_problem.second_order_cone_dims = {3};

  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);
  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 1.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[2]), 0.0, 1e-4);
}

TEST(barrier, mixed_linear_and_soc_block)
{
  // Variables ordered as [l | t, u, v], where (t, u, v) \in Q^3.
  //
  // minimize   l
  // subject to l - t = 0
  //            u     = 1
  //            (t, u, v) in Q^3
  //
  // Optimal: l* = 1, t* = 1, u* = 1, v* = 0, obj* = 1.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 2;
  constexpr int n  = 4;
  constexpr int nz = 4;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {1.0, 0.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Columns: l, t, u, v
  user_problem.A.col_start = {0, 1, 2, 3, 3};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 0;
  user_problem.A.x[1]      = -1.0;
  user_problem.A.i[2]      = 1;
  user_problem.A.x[2]      = 1.0;

  user_problem.rhs       = {0.0, 1.0};
  user_problem.row_sense = {'E', 'E'};

  user_problem.lower = {0.0, 0.0, 0.0, 0.0};
  user_problem.upper = {inf, inf, inf, inf};

  user_problem.num_range_rows = 0;
  user_problem.problem_name   = "mixed_linear_and_soc_block";

  user_problem.cone_var_start         = 1;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 1.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 1.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[3]), 0.0, 1e-4);
}

TEST(barrier, mixed_linear_and_soc_tail_coupling)
{
  // Variables ordered as [l | t, u, v], where (t, u, v) \in Q^3.
  //
  // minimize   t
  // subject to l - u = 0
  //            l + u = 2
  //            (t, u, v) in Q^3
  //
  // Optimal: l* = 1, t* = 1, u* = 1, v* = 0, obj* = 1.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 2;
  constexpr int n  = 4;
  constexpr int nz = 4;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 1.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Columns: l, t, u, v
  user_problem.A.col_start = {0, 2, 2, 4, 4};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 1;
  user_problem.A.x[1]      = 1.0;
  user_problem.A.i[2]      = 0;
  user_problem.A.x[2]      = -1.0;
  user_problem.A.i[3]      = 1;
  user_problem.A.x[3]      = 1.0;

  user_problem.rhs       = {0.0, 2.0};
  user_problem.row_sense = {'E', 'E'};
  user_problem.lower     = {0.0, 0.0, 0.0, 0.0};
  user_problem.upper     = {inf, inf, inf, inf};

  user_problem.num_range_rows         = 0;
  user_problem.problem_name           = "mixed_linear_and_soc_tail_coupling";
  user_problem.cone_var_start         = 1;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;
  settings.scale_columns    = true;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 1.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 1.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[3]), 0.0, 1e-4);
}

TEST(barrier, mixed_linear_and_soc_tail_coupling_with_inequality)
{
  // Variables ordered as [l | t, u, v], where (t, u, v) \in Q^3.
  //
  // minimize   t
  // subject to l - u = 0
  //            l + u >= 2
  //            (t, u, v) in Q^3
  //
  // Optimal: l* = 1, t* = 1, u* = 1, v* = 0, obj* = 1.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 2;
  constexpr int n  = 4;
  constexpr int nz = 4;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 1.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Columns: l, t, u, v
  user_problem.A.col_start = {0, 2, 2, 4, 4};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 1;
  user_problem.A.x[1]      = 1.0;
  user_problem.A.i[2]      = 0;
  user_problem.A.x[2]      = -1.0;
  user_problem.A.i[3]      = 1;
  user_problem.A.x[3]      = 1.0;

  user_problem.rhs       = {0.0, 2.0};
  user_problem.row_sense = {'E', 'G'};
  user_problem.lower     = {0.0, 0.0, 0.0, 0.0};
  user_problem.upper     = {inf, inf, inf, inf};

  user_problem.num_range_rows         = 0;
  user_problem.problem_name           = "mixed_linear_and_soc_tail_coupling_with_inequality";
  user_problem.cone_var_start         = 1;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;
  settings.scale_columns    = true;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 1.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 1.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[3]), 0.0, 1e-4);
}

TEST(barrier, mixed_linear_and_two_soc_blocks)
{
  // Variables ordered as [l1, l2 | t1, u1, v1 | t2, u2, v2],
  // where (t1, u1, v1), (t2, u2, v2) \in Q^3.
  //
  // minimize   t1 + t2
  // subject to l1 - u1 = 0
  //            l2 - u2 = 0
  //            l1 + l2 = 3
  //            l1 - l2 = 1
  //
  // Optimal: l1* = 2, l2* = 1, t1* = 2, u1* = 2, v1* = 0,
  //          t2* = 1, u2* = 1, v2* = 0, obj* = 3.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 4;
  constexpr int n  = 8;
  constexpr int nz = 8;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Columns: l1, l2, t1, u1, v1, t2, u2, v2
  user_problem.A.col_start = {0, 3, 6, 6, 7, 7, 7, 8, 8};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 2;
  user_problem.A.x[1]      = 1.0;
  user_problem.A.i[2]      = 3;
  user_problem.A.x[2]      = 1.0;
  user_problem.A.i[3]      = 1;
  user_problem.A.x[3]      = 1.0;
  user_problem.A.i[4]      = 2;
  user_problem.A.x[4]      = 1.0;
  user_problem.A.i[5]      = 3;
  user_problem.A.x[5]      = -1.0;
  user_problem.A.i[6]      = 0;
  user_problem.A.x[6]      = -1.0;
  user_problem.A.i[7]      = 1;
  user_problem.A.x[7]      = -1.0;

  user_problem.rhs       = {0.0, 0.0, 3.0, 1.0};
  user_problem.row_sense = {'E', 'E', 'E', 'E'};
  user_problem.lower.assign(n, 0.0);
  user_problem.upper.assign(n, inf);

  user_problem.num_range_rows         = 0;
  user_problem.problem_name           = "mixed_linear_and_two_soc_blocks";
  user_problem.cone_var_start         = 2;
  user_problem.second_order_cone_dims = {3, 3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 3.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 2.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 2.0, 1e-4);
  EXPECT_NEAR(solution.x[3], 2.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[4]), 0.0, 1e-4);
  EXPECT_NEAR(solution.x[5], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[6], 1.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[7]), 0.0, 1e-4);
}

TEST(barrier, mixed_linear_and_two_soc_blocks_with_inequality)
{
  // Variables ordered as [l1, l2 | t1, u1, v1 | t2, u2, v2],
  // where (t1, u1, v1), (t2, u2, v2) \in Q^3.
  //
  // minimize   t1 + t2
  // subject to l1 - u1 = 0
  //            l2 - u2 = 0
  //            l1 + l2 >= 3
  //            l1 - l2 = 1
  //
  // Optimal: l1* = 2, l2* = 1, t1* = 2, u1* = 2, v1* = 0,
  //          t2* = 1, u2* = 1, v2* = 0, obj* = 3.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 4;
  constexpr int n  = 8;
  constexpr int nz = 8;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Columns: l1, l2, t1, u1, v1, t2, u2, v2
  user_problem.A.col_start = {0, 3, 6, 6, 7, 7, 7, 8, 8};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 2;
  user_problem.A.x[1]      = 1.0;
  user_problem.A.i[2]      = 3;
  user_problem.A.x[2]      = 1.0;
  user_problem.A.i[3]      = 1;
  user_problem.A.x[3]      = 1.0;
  user_problem.A.i[4]      = 2;
  user_problem.A.x[4]      = 1.0;
  user_problem.A.i[5]      = 3;
  user_problem.A.x[5]      = -1.0;
  user_problem.A.i[6]      = 0;
  user_problem.A.x[6]      = -1.0;
  user_problem.A.i[7]      = 1;
  user_problem.A.x[7]      = -1.0;

  user_problem.rhs       = {0.0, 0.0, 3.0, 1.0};
  user_problem.row_sense = {'E', 'E', 'G', 'E'};
  user_problem.lower.assign(n, 0.0);
  user_problem.upper.assign(n, inf);

  user_problem.num_range_rows         = 0;
  user_problem.problem_name           = "mixed_linear_and_two_soc_blocks_with_inequality";
  user_problem.cone_var_start         = 2;
  user_problem.second_order_cone_dims = {3, 3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;
  settings.scale_columns    = true;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 3.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 2.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 2.0, 1e-4);
  EXPECT_NEAR(solution.x[3], 2.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[4]), 0.0, 1e-4);
  EXPECT_NEAR(solution.x[5], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[6], 1.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[7]), 0.0, 1e-4);
}

TEST(barrier, free_linear_prefix_is_uncrushed_correctly_with_soc_block)
{
  // Variables ordered as [l | t, u, v], where (t, u, v) \in Q^3 and l is free.
  //
  // minimize   t
  // subject to l - u = 0
  //            u     = 1
  //            (t, u, v) in Q^3
  //
  // Direct free variable l is kept through presolve; end-to-end solve returns
  // l* = 1, t* = 1, u* = 1, v* = 0, obj* = 1.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 2;
  constexpr int n  = 4;
  constexpr int nz = 3;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 1.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Columns: l, t, u, v
  user_problem.A.col_start = {0, 1, 1, 3, 3};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 0;
  user_problem.A.x[1]      = -1.0;
  user_problem.A.i[2]      = 1;
  user_problem.A.x[2]      = 1.0;

  user_problem.rhs       = {0.0, 1.0};
  user_problem.row_sense = {'E', 'E'};
  user_problem.lower     = {-inf, 0.0, 0.0, 0.0};
  user_problem.upper     = {inf, inf, inf, inf};

  user_problem.num_range_rows         = 0;
  user_problem.problem_name           = "free_linear_prefix_is_uncrushed_correctly_with_soc_block";
  user_problem.cone_var_start         = 1;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 1.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 1.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[3]), 0.0, 1e-4);
}

TEST(barrier, qp_with_soc_block)
{
  // Variables ordered as [l | t, u, v], where (t, u, v) \in Q^3.
  //
  // minimize   0.5 l^2 + t
  // subject to l + u = 2
  //            (t, u, v) in Q^3
  //
  // Since t >= |u| and u = 2 - l with l >= 0, the objective becomes
  // 0.5 l^2 + |2 - l|, which is minimized at l* = 1, u* = 1, t* = 1, v* = 0.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 1;
  constexpr int n  = 4;
  constexpr int nz = 2;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 1.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Columns: l, t, u, v
  user_problem.A.col_start = {0, 1, 1, 2, 2};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 0;
  user_problem.A.x[1]      = 1.0;

  user_problem.rhs       = {2.0};
  user_problem.row_sense = {'E'};
  user_problem.lower.assign(n, 0.0);
  user_problem.upper.assign(n, inf);

  user_problem.Q_offsets = {0, 1, 1, 1, 1};
  user_problem.Q_indices = {0};
  user_problem.Q_values  = {1.0};

  user_problem.num_range_rows         = 0;
  user_problem.problem_name           = "qp_with_soc_block";
  user_problem.cone_var_start         = 1;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 1.5, 1e-4);
  EXPECT_NEAR(solution.x[0], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 1.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[3]), 0.0, 1e-4);
}

TEST(barrier, sparse_soc_expansion_solves_large_single_cone)
{
  // minimize x_0
  // subject to x_1 = 1
  //            (x_0, ..., x_5) in Q^6
  //
  // Optimal: x* = (1, 1, 0, 0, 0, 0), obj* = 1

  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 1;
  constexpr int n  = 6;
  constexpr int nz = 1;

  user_problem.num_rows = m;
  user_problem.num_cols = n;
  user_problem.objective.assign(n, 0.0);
  user_problem.objective[0] = 1.0;

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 0, 1, 1, 1, 1, 1};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;

  user_problem.rhs       = {1.0};
  user_problem.row_sense = {'E'};
  user_problem.lower.assign(n, 0.0);
  user_problem.upper.assign(n, inf);

  user_problem.num_range_rows         = 0;
  user_problem.problem_name           = "sparse_soc_single_large_cone";
  user_problem.cone_var_start         = 0;
  user_problem.second_order_cone_dims = {6};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier                      = true;
  settings.barrier_presolve             = true;
  settings.dualize                      = 0;
  settings.barrier_soc_threshold        = 4;
  settings.barrier_iterative_refinement = true;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);
  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 1.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  for (int j = 2; j < n; ++j) {
    EXPECT_NEAR(std::abs(solution.x[j]), 0.0, 1e-4) << "index " << j;
  }
}

TEST(barrier, mixed_dense_and_sparse_soc_blocks)
{
  // Variables ordered as [l1, l2 | dense Q^3 | sparse Q^6].
  //
  // minimize   t_dense + t_sparse
  // subject to l1 - u_dense = 0
  //            l2 - u_sparse = 0
  //            l1 + l2 >= 3
  //            l1 - l2 = 1
  //
  // Same optimum as mixed_linear_and_two_soc_blocks_with_inequality: obj* = 3.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 4;
  constexpr int n  = 11;
  constexpr int nz = 8;

  user_problem.num_rows = m;
  user_problem.num_cols = n;
  user_problem.objective.assign(n, 0.0);
  user_problem.objective[2] = 1.0;  // t_dense (head of dense Q^3 block)
  user_problem.objective[5] = 1.0;  // t_sparse (head of sparse Q^6 block)

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Columns: l1, l2, t_d, u_d, v_d, t_s, u_s, v_s, w_s, y_s, z_s
  user_problem.A.col_start = {0, 3, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 2;
  user_problem.A.x[1]      = 1.0;
  user_problem.A.i[2]      = 3;
  user_problem.A.x[2]      = 1.0;
  user_problem.A.i[3]      = 1;
  user_problem.A.x[3]      = 1.0;
  user_problem.A.i[4]      = 2;
  user_problem.A.x[4]      = 1.0;
  user_problem.A.i[5]      = 3;
  user_problem.A.x[5]      = -1.0;
  user_problem.A.i[6]      = 0;
  user_problem.A.x[6]      = -1.0;
  user_problem.A.i[7]      = 1;
  user_problem.A.x[7]      = -1.0;

  user_problem.rhs       = {0.0, 0.0, 3.0, 1.0};
  user_problem.row_sense = {'E', 'E', 'G', 'E'};
  user_problem.lower.assign(n, 0.0);
  user_problem.upper.assign(n, inf);

  user_problem.num_range_rows         = 0;
  user_problem.problem_name           = "mixed_dense_and_sparse_soc_blocks";
  user_problem.cone_var_start         = 2;
  user_problem.second_order_cone_dims = {3, 6};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier                      = true;
  settings.barrier_presolve             = true;
  settings.dualize                      = 0;
  settings.scale_columns                = true;
  settings.barrier_soc_threshold        = 4;
  settings.barrier_iterative_refinement = true;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 3.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 2.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 2.0, 1e-4);
  EXPECT_NEAR(solution.x[3], 2.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[4]), 0.0, 1e-4);
  EXPECT_NEAR(solution.x[5], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[6], 1.0, 1e-4);
  for (int j = 7; j < n; ++j) {
    EXPECT_NEAR(std::abs(solution.x[j]), 0.0, 1e-4) << "index " << j;
  }
}

TEST(barrier, sparse_soc_expansion_solves_dim_500_cone)
{
  // minimize x_0
  // subject to x_1 = 1
  //            (x_0, ..., x_499) in Q^500
  //
  // Optimal: x* = (1, 1, 0, ..., 0), obj* = 1
  // Sparse cone (dim 500 > default threshold 5) with barrier IR enabled.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 1;
  constexpr int n  = 500;
  constexpr int nz = 1;

  user_problem.num_rows = m;
  user_problem.num_cols = n;
  user_problem.objective.assign(n, 0.0);
  user_problem.objective[0] = 1.0;

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // x_1 = 1: nonzero in column 1.
  user_problem.A.col_start.assign(n + 1, 1);
  user_problem.A.col_start[0] = 0;
  user_problem.A.col_start[1] = 0;
  user_problem.A.i[0]         = 0;
  user_problem.A.x[0]         = 1.0;

  user_problem.rhs       = {1.0};
  user_problem.row_sense = {'E'};
  user_problem.lower.assign(n, 0.0);
  user_problem.upper.assign(n, inf);

  user_problem.num_range_rows         = 0;
  user_problem.problem_name           = "sparse_soc_dim_500";
  user_problem.cone_var_start         = 0;
  user_problem.second_order_cone_dims = {500};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier                      = true;
  settings.barrier_presolve             = true;
  settings.dualize                      = 0;
  settings.barrier_soc_threshold        = 5;
  settings.barrier_iterative_refinement = true;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 1.0, 1e-3);
  EXPECT_NEAR(solution.x[0], 1.0, 1e-3);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-3);
  for (int j = 2; j < n; ++j) {
    EXPECT_NEAR(std::abs(solution.x[j]), 0.0, 1e-3) << "index " << j;
  }
}

TEST(barrier, free_variable_substitution_postsolve_kkt)
{
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem = make_free_substitution_qp(&handle);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;
  settings.scale_columns    = false;
  settings.postsolve_info   = 1;

  std::vector<int> new_slacks;
  dualize_info_t<int, double> dualize_info;
  lp_problem_t<int, double> original_lp(user_problem.handle_ptr, 1, 1, 1);
  convert_user_problem(user_problem, settings, original_lp, new_slacks, dualize_info);

  presolve_info_t<int, double> presolve_info;
  lp_problem_t<int, double> presolved_lp(user_problem.handle_ptr, 1, 1, 1);
  ASSERT_EQ(presolve(original_lp, settings, presolved_lp, presolve_info), 0);
  ASSERT_EQ(presolve_info.free_variable_eliminations.size(), 2u);
  ASSERT_EQ(presolved_lp.num_cols, original_lp.num_cols - 2);
  ASSERT_EQ(presolved_lp.num_rows, original_lp.num_rows - 2);

  // Known feasible point on the original problem, restricted to remaining columns.
  const std::vector<double> original_x = {0.0, 0.0, 1.0, 1.0, 0.0};
  std::vector<double> crushed_x(presolved_lp.num_cols);
  for (int k = 0; k < presolved_lp.num_cols; ++k) {
    crushed_x[k] = original_x[presolve_info.free_elimination_remaining_variables[k]];
  }
  std::vector<double> crushed_y(presolved_lp.num_rows, 0.25);
  std::vector<double> crushed_z(presolved_lp.num_cols, 0.0);
  std::vector<double> reduced_dual;
  dual_residual(presolved_lp, crushed_x, crushed_y, crushed_z, reduced_dual);
  for (int j = 0; j < presolved_lp.num_cols; ++j) {
    crushed_z[j] -= reduced_dual[j];
  }
  dual_residual(presolved_lp, crushed_x, crushed_y, crushed_z, reduced_dual);
  ASSERT_NEAR((vector_norm_inf<int, double>(reduced_dual)), 0.0, 1e-12);

  std::vector<double> uncrushed_x(original_lp.num_cols);
  std::vector<double> uncrushed_y(original_lp.num_rows);
  std::vector<double> uncrushed_z(original_lp.num_cols);
  uncrush_solution(presolve_info,
                   settings,
                   original_lp,
                   crushed_x,
                   crushed_y,
                   crushed_z,
                   uncrushed_x,
                   uncrushed_y,
                   uncrushed_z);

  ASSERT_EQ(uncrushed_x.size(), static_cast<size_t>(original_lp.num_cols));
  EXPECT_NEAR(uncrushed_x[0], 0.0, 1e-12);
  EXPECT_NEAR(uncrushed_x[1], 0.0, 1e-12);
  EXPECT_NEAR(uncrushed_x[2], 1.0, 1e-12);
  EXPECT_NEAR(uncrushed_x[3], 1.0, 1e-12);
  EXPECT_NEAR(std::abs(uncrushed_z[0]), 0.0, 1e-12);
  EXPECT_NEAR(std::abs(uncrushed_z[1]), 0.0, 1e-12);

  std::vector<double> primal_residual = original_lp.rhs;
  matrix_vector_multiply(original_lp.A, 1.0, uncrushed_x, -1.0, primal_residual);
  EXPECT_NEAR((vector_norm_inf<int, double>(primal_residual)), 0.0, 1e-12);

  std::vector<double> uncrushed_dual;
  dual_residual(original_lp, uncrushed_x, uncrushed_y, uncrushed_z, uncrushed_dual);
  EXPECT_NEAR((vector_norm_inf<int, double>(uncrushed_dual)), 0.0, 1e-10);

  lp_solution_t<int, double> solution(user_problem.num_rows, user_problem.num_cols);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);
  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 0.5, 1e-4);
  EXPECT_NEAR(solution.x[0], 0.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 0.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[3], 1.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.z[0]), 0.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.z[1]), 0.0, 1e-4);

  std::vector<double> solved_primal = original_lp.rhs;
  matrix_vector_multiply(original_lp.A, 1.0, solution.x, -1.0, solved_primal);
  EXPECT_NEAR((vector_norm_inf<int, double>(solved_primal)), 0.0, 1e-5);

  std::vector<double> solved_dual;
  dual_residual(original_lp, solution.x, solution.y, solution.z, solved_dual);
  EXPECT_NEAR((vector_norm_inf<int, double>(solved_dual)), 0.0, 1e-5);
}

TEST(barrier, cached_solve_can_enable_iterative_refinement)
{
  // The first solve has no GMRES workspace. Enabling refinement on a cached solve must
  // create it for the retained system, including the ADAT case whose size is only m.
  for (int augmented : {0, 1}) {
    SCOPED_TRACE(augmented);
    auto cache = barrier_cache_t::create(cudaStreamNonBlocking);
    init_handler(cache->handle_ptr());
    user_problem_t<int, double> user_problem(cache->handle_ptr());
    user_problem.num_range_rows = 0;

    // Minimize 0.5 * (x0^2 + x1^2), subject to x0 + x1 = rhs and 0 <= x <= 10.
    // The unique solution is x0 = x1 = rhs / 2, with objective rhs^2 / 4.
    constexpr int m = 1, n = 2, nz = 2;
    user_problem.num_rows = m;
    user_problem.num_cols = n;
    user_problem.objective.assign(n, 0.0);
    user_problem.A.m      = m;
    user_problem.A.n      = n;
    user_problem.A.nz_max = nz;
    user_problem.A.reallocate(nz);
    user_problem.A.col_start = {0, 1, 2};
    user_problem.A.i         = {0, 0};
    user_problem.A.x         = {1.0, 1.0};
    user_problem.rhs         = {2.0};
    user_problem.row_sense   = {'E'};
    user_problem.lower.assign(n, 0.0);
    user_problem.upper.assign(n, 10.0);
    user_problem.Q_offsets = {0, 1, 2};
    user_problem.Q_indices = {0, 1};
    user_problem.Q_values  = {1.0, 1.0};
    user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);
    user_problem.problem_name = "cached_solve_can_enable_iterative_refinement";

    simplex_solver_settings_t<int, double> settings;
    settings.barrier                               = true;
    settings.barrier_presolve                      = false;
    settings.barrier_presolve_bound_free_variables = 0;
    settings.dualize                               = 0;
    settings.augmented                             = augmented;
    settings.barrier_iterative_refinement          = false;
    lp_solution_t<int, double> solution(m, n);
    ASSERT_EQ(solve_linear_program_with_barrier(user_problem, settings, solution, cache.get()),
              lp_status_t::OPTIMAL);
    ASSERT_NE(cache->transform(), nullptr);
    EXPECT_NEAR(solution.objective, 1.0, 1e-4);

    settings.barrier_iterative_refinement = true;
    for (double rhs : {4.0, 6.0}) {
      SCOPED_TRACE(rhs);
      // Update only the cache: rebuilding from user_problem would still solve RHS 2
      // and fail the expected solution/objective checks below.
      cache->update_rhs(&rhs, m);
      ASSERT_TRUE(cache->dirty());
      const auto status =
        solve_linear_program_with_barrier(user_problem, settings, solution, cache.get());
      ASSERT_EQ(status, lp_status_t::OPTIMAL);
      EXPECT_FALSE(cache->dirty());
      EXPECT_NEAR(solution.x[0], rhs / 2, 1e-4);
      EXPECT_NEAR(solution.x[1], rhs / 2, 1e-4);
      EXPECT_NEAR(solution.objective, rhs * rhs / 4, 1e-4);
    }
  }
}

double gmres_update_basis_value(int i, int j, bool cancellation)
{
  if (!cancellation) { return ((3 * i + 7 * j) % 37 - 18) / 16.0; }
  // Adjacent terms nearly cancel, with a small, exactly representable remainder.
  double value = 1048576.0 + ((i + j / 2) % 17) / 8.0;
  if (j % 2 != 0) { value += (1 + i % 13 + j) / 1024.0; }
  return i % 2 == 0 ? value : -value;
}

TEST(barrier, gmres_update_handles_all_counts_and_in_place_reuse)
{
  raft::handle_t handle;
  const auto stream       = handle.get_stream();
  constexpr int dimension = barrier::gmres_workspace_t<double>::dimension;
  for (int n : {0, 1, 31, 33, 257, 4096}) {
    barrier::gmres_workspace_t<double> workspace(n, stream);
    rmm::device_uvector<double> d_x(n, stream);
    std::vector<double> h_initial(n), h_actual(n);
    for (int i = 0; i < n; ++i) {
      h_initial[i] = (i % 19 - 9) / 8.0;
    }
    for (bool cancellation : {false, true}) {
      for (int count = 0; count <= dimension; ++count) {
        SCOPED_TRACE(::testing::Message()
                     << "n=" << n << " count=" << count << " cancellation=" << cancellation);
        std::vector<double> h_coefficients(count), h_delta(n, 0.0), h_abs_sum(n, 0.0);
        std::vector<std::vector<double>> h_basis(
          dimension, std::vector<double>(n, std::numeric_limits<double>::quiet_NaN()));
        for (int j = 0; j < count; ++j) {
          h_coefficients[j] = (j % 2 == 0 ? 1.0 : -1.0) * (j / 2 + 1) / 8.0;
          for (int i = 0; i < n; ++i) {
            h_basis[j][i]     = gmres_update_basis_value(i, j, cancellation);
            const double term = h_coefficients[j] * h_basis[j][i];
            // These dyadic products and their short sums are exact in double.
            h_delta[i] += term;
            h_abs_sum[i] += std::abs(term);
          }
        }
        if (n > 0) {
          raft::copy(d_x.data(), h_initial.data(), n, stream);
          for (int j = 0; j < dimension; ++j) {
            // Poison inactive columns so accidentally using them cannot pass silently.
            raft::copy(workspace.Z[j].data(), h_basis[j].data(), n, stream);
          }
        }
        for (int update = 1; update <= 2; ++update) {
          SCOPED_TRACE(::testing::Message() << "update=" << update);
          workspace.update(d_x, h_coefficients, count);
          RAFT_CUDA_TRY(cudaGetLastError());
          if (n > 0) { raft::copy(h_actual.data(), d_x.data(), n, stream); }
          stream.sync();
          for (int i = 0; i < n; ++i) {
            const double expected = h_initial[i] + update * h_delta[i];
            // Bound by input magnitudes, not the potentially tiny cancellation result.
            const double tolerance = 32 * std::numeric_limits<double>::epsilon() *
                                     (std::abs(h_initial[i]) + update * h_abs_sum[i]);
            ASSERT_TRUE(std::isfinite(h_actual[i])) << "i=" << i;
            EXPECT_NEAR(h_actual[i], expected, tolerance) << "i=" << i;
          }
        }
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
  transform_reduce_helper_t<double> reductions(stream);
  rmm::device_uvector<double> empty(0, stream);
  auto singleton  = cuopt::device_copy(std::vector<double>{-7.0}, stream);
  constexpr int n = 4096;
  auto large      = cuopt::device_copy(std::vector<double>(n, -3.0), stream);

  EXPECT_DOUBLE_EQ(vector_norm_inf(empty, reductions), 0.0);
  EXPECT_DOUBLE_EQ(vector_norm2(empty, reductions), 0.0);
  EXPECT_DOUBLE_EQ(vector_norm_inf(singleton, reductions), 7.0);
  EXPECT_DOUBLE_EQ(vector_norm2(singleton, reductions), 7.0);
  EXPECT_DOUBLE_EQ(vector_norm_inf(large, reductions), 3.0);
  EXPECT_DOUBLE_EQ(vector_norm2(large, reductions), 192.0);

  rmm::device_uvector<double> outputs(2, stream);
  std::vector<double> results(2);
  const void* scratch = nullptr;
  size_t capacity     = 0;
  for (int size : {n, 1, 0, n}) {
    SCOPED_TRACE(size);
    const auto& input          = size == n ? large : (size == 1 ? singleton : empty);
    const double expected_norm = size == n ? 3.0 : (size == 1 ? 7.0 : 0.0);
    // Queue two reductions into distinct caller-owned slots before a single readback.
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
    EXPECT_DOUBLE_EQ(results[0], expected_norm);
    EXPECT_DOUBLE_EQ(results[1], 5.0 + size * expected_norm * expected_norm);

    if (scratch == nullptr) {
      // Warm both reduction operators before checking large-to-small-to-large reuse.
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
