/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// scaling_ruiz_gpu() is a step-for-step port of the Ruiz branch of scaling(). The two run on
// disjoint problem sizes in production (solve.cpp picks the GPU path above
// gpu_ruiz_nnz_threshold), so nothing else exercises them against each other.

#include <gtest/gtest.h>

#include <barrier/device_sparse_matrix.cuh>
#include <barrier/scaling_gpu.cuh>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/scaling.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <linear_algebra/sparse_matrix.hpp>

#include <raft/core/handle.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace cuopt::mathematical_optimization::barrier::test {

using simplex::lp_problem_t;
using simplex::simplex_solver_settings_t;

using i_t = int;
using f_t = double;

namespace {

// QP whose variable bounds span 1e2..1e11 while A's coefficients stay O(1) -- the case the
// bound-magnitude column pre-scaling exists for.
lp_problem_t<i_t, f_t> make_wide_bound_qp(const raft::handle_t* handle)
{
  constexpr i_t m  = 3;
  constexpr i_t n  = 4;
  constexpr i_t nz = 9;

  lp_problem_t<i_t, f_t> lp(handle, m, n, nz);
  lp.obj_scale = 1.0;
  lp.objective = {1.0, 2.0, 3.0, 4.0};
  lp.rhs       = {1.0, 1.0, 1.0};

  lp.A.col_start = {0, 2, 4, 6, 9};
  lp.A.i         = {0, 1, 0, 2, 1, 2, 0, 1, 2};
  lp.A.x         = {1.0, 2.0, 3.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0};

  lp.lower = {0.0, 0.0, -1e9, 1e2};
  lp.upper = {1e4, 1e7, 1e9, 1e11};

  // Symmetric Q (CSR): diag(2, 3, 1, 4) with Q(0,1) = Q(1,0) = 0.5.
  lp.Q           = csr_matrix_t<i_t, f_t>(n, n, 6);
  lp.Q.row_start = {0, 2, 4, 5, 6};
  lp.Q.j         = {0, 1, 0, 1, 2, 3};
  lp.Q.x         = {2.0, 0.5, 0.5, 3.0, 1.0, 4.0};
  return lp;
}

void expect_vectors_near(const std::vector<f_t>& cpu,
                         const std::vector<f_t>& gpu,
                         f_t tol,
                         const char* what)
{
  ASSERT_EQ(cpu.size(), gpu.size()) << what << " length";
  for (size_t k = 0; k < cpu.size(); ++k) {
    const f_t magnitude = std::max(f_t(1), std::abs(cpu[k]));
    EXPECT_NEAR(cpu[k], gpu[k], tol * magnitude) << what << " mismatch at index " << k;
  }
}

// Per-column bound magnitude, defined exactly as the pre-scaling defines it: the geometric
// mean of the finite nonzero bounds, or the single finite side, or 0 when the column is free.
f_t column_bound_magnitude(f_t lower, f_t upper)
{
  const f_t lo                 = std::abs(lower);
  const f_t hi                 = std::abs(upper);
  const bool finite_nonzero_lo = lower > -1e20 && lo > 0;
  const bool finite_nonzero_hi = upper < 1e20 && hi > 0;
  if (finite_nonzero_lo && finite_nonzero_hi) return std::sqrt(lo * hi);
  if (finite_nonzero_lo) return lo;
  if (finite_nonzero_hi) return hi;
  return 0;
}

// Spread of those magnitudes across columns -- what the pre-scaling collapses toward 1. It
// cannot compress a single column's own lower..upper range, so measuring every bound
// individually would not show the effect.
f_t column_magnitude_spread(const std::vector<f_t>& lower, const std::vector<f_t>& upper)
{
  f_t lo = std::numeric_limits<f_t>::max();
  f_t hi = 0;
  for (size_t j = 0; j < lower.size(); ++j) {
    const f_t mag = column_bound_magnitude(lower[j], upper[j]);
    if (mag > 0) {
      lo = std::min(lo, mag);
      hi = std::max(hi, mag);
    }
  }
  return hi / lo;
}

}  // namespace

TEST(scaling_gpu_parity, wide_bound_qp_matches_cpu)
{
  raft::handle_t handle{};
  const lp_problem_t<i_t, f_t> unscaled = make_wide_bound_qp(&handle);

  simplex_solver_settings_t<i_t, f_t> settings;
  // Force the Ruiz branch so neither path takes the skip heuristic's early return.
  settings.qcqp_ruiz_equilibration = 1;

  lp_problem_t<i_t, f_t> cpu_scaled(&handle, 1, 1, 1);
  std::vector<f_t> cpu_col_scaling;
  std::vector<f_t> cpu_row_scaling;
  ASSERT_EQ(simplex::scaling(unscaled, settings, cpu_scaled, cpu_col_scaling, cpu_row_scaling), 0);

  lp_problem_t<i_t, f_t> gpu_scaled(&handle, 1, 1, 1);
  std::vector<f_t> gpu_col_scaling;
  std::vector<f_t> gpu_row_scaling;
  device_csc_matrix_ptr_t<i_t, f_t> device_A;
  device_csc_matrix_ptr_t<i_t, f_t> device_Q;
  ASSERT_EQ(simplex::scaling_ruiz_gpu(
              unscaled, settings, gpu_scaled, gpu_col_scaling, gpu_row_scaling, device_A, device_Q),
            0);

  // Both paths do the same multiplications in the same order; only the geometric-mean
  // reduction differs in summation order, so the two should agree to near machine precision.
  constexpr f_t tol = 1e-9;
  expect_vectors_near(cpu_col_scaling, gpu_col_scaling, tol, "column_scaling");
  expect_vectors_near(cpu_row_scaling, gpu_row_scaling, tol, "row_scaling");
  expect_vectors_near(cpu_scaled.objective, gpu_scaled.objective, tol, "objective");
  expect_vectors_near(cpu_scaled.lower, gpu_scaled.lower, tol, "lower");
  expect_vectors_near(cpu_scaled.upper, gpu_scaled.upper, tol, "upper");
  expect_vectors_near(cpu_scaled.rhs, gpu_scaled.rhs, tol, "rhs");
  expect_vectors_near(cpu_scaled.A.x, gpu_scaled.A.x, tol, "A values");
  expect_vectors_near(cpu_scaled.Q.x, gpu_scaled.Q.x, tol, "Q values");
}

// Ruiz re-equilibrates coefficients after the pre-scaling and largely cancels it, so the net
// compression is modest -- on this problem the spread goes 1.0e5 -> 6.6e4 with the pre-scaling
// and 1.0e5 -> 1.1e5 (i.e. worse than untouched) without it. Requiring only "no worse than
// unscaled" therefore still pins the pre-scaling down on both paths, without baking in a
// threshold that Ruiz's iteration count could shift.
TEST(scaling_gpu_parity, pre_scaling_does_not_leave_bounds_more_spread_than_unscaled)
{
  raft::handle_t handle{};
  const lp_problem_t<i_t, f_t> unscaled = make_wide_bound_qp(&handle);
  const f_t unscaled_spread             = column_magnitude_spread(unscaled.lower, unscaled.upper);

  simplex_solver_settings_t<i_t, f_t> settings;
  settings.qcqp_ruiz_equilibration = 1;

  lp_problem_t<i_t, f_t> cpu_scaled(&handle, 1, 1, 1);
  std::vector<f_t> cpu_col_scaling;
  std::vector<f_t> cpu_row_scaling;
  ASSERT_EQ(simplex::scaling(unscaled, settings, cpu_scaled, cpu_col_scaling, cpu_row_scaling), 0);

  lp_problem_t<i_t, f_t> gpu_scaled(&handle, 1, 1, 1);
  std::vector<f_t> gpu_col_scaling;
  std::vector<f_t> gpu_row_scaling;
  device_csc_matrix_ptr_t<i_t, f_t> device_A;
  device_csc_matrix_ptr_t<i_t, f_t> device_Q;
  ASSERT_EQ(simplex::scaling_ruiz_gpu(
              unscaled, settings, gpu_scaled, gpu_col_scaling, gpu_row_scaling, device_A, device_Q),
            0);

  EXPECT_LT(column_magnitude_spread(cpu_scaled.lower, cpu_scaled.upper), unscaled_spread);
  EXPECT_LT(column_magnitude_spread(gpu_scaled.lower, gpu_scaled.upper), unscaled_spread);
}

}  // namespace cuopt::mathematical_optimization::barrier::test
