/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <rmm/device_uvector.hpp>
#include <vector>
#include "dual_simplex/dense_vector.hpp"
#include "dual_simplex/simplex_solver_settings.hpp"
#include "dual_simplex/types.hpp"
#include "dual_simplex/vector_math.hpp"

namespace cuopt::linear_programming::dual_simplex {

// Functors for device operations (defined at namespace scope to avoid CUDA lambda restrictions)
template <typename T>
struct scale_op {
  T scale;
  __host__ __device__ T operator()(T val) const { return val * scale; }
};

template <typename T>
struct multiply_op {
  __host__ __device__ T operator()(T a, T b) const { return a * b; }
};

template <typename T>
struct axpy_op {
  T alpha;
  __host__ __device__ T operator()(T x, T y) const { return x + alpha * y; }
};

template <typename T>
struct subtract_scaled_op {
  T scale;
  __host__ __device__ T operator()(T a, T b) const { return a - scale * b; }
};

template <typename f_t>
f_t vector_norm_inf(const rmm::device_uvector<f_t>& x)
{
  auto begin   = x.data();
  auto end     = x.data() + x.size();
  auto max_abs = thrust::transform_reduce(
    rmm::exec_policy(x.stream()),
    begin,
    end,
    [] __host__ __device__(f_t val) { return abs(val); },
    static_cast<f_t>(0),
    thrust::maximum<f_t>{});
  RAFT_CHECK_CUDA(x.stream());
  return max_abs;
}

template <typename f_t>
f_t vector_norm2(const rmm::device_uvector<f_t>& x)
{
  auto begin          = x.data();
  auto end            = x.data() + x.size();
  auto sum_of_squares = thrust::transform_reduce(
    rmm::exec_policy(x.stream()),
    begin,
    end,
    [] __host__ __device__(f_t val) { return val * val; },
    f_t(0),
    thrust::plus<f_t>{});
  RAFT_CHECK_CUDA(x.stream());
  return std::sqrt(sum_of_squares);
}

template <typename i_t, typename f_t, typename T>
f_t iterative_refinement_simple(T& op,
                                const rmm::device_uvector<f_t>& b,
                                rmm::device_uvector<f_t>& x)
{
  rmm::device_uvector<f_t> x_sav(x, x.stream());

  const bool show_iterative_refinement_info = false;

  // r = b - Ax
  rmm::device_uvector<f_t> r(b, b.stream());
  op.a_multiply(-1.0, x, 1.0, r);

  f_t error = vector_norm_inf<f_t>(r);
  if (show_iterative_refinement_info) {
    CUOPT_LOG_INFO(
      "Iterative refinement. Initial error %e || x || %.16e", error, vector_norm2<f_t>(x));
  }
  rmm::device_uvector<f_t> delta_x(x.size(), op.data_.handle_ptr->get_stream());
  i_t iter = 0;
  while (error > 1e-8 && iter < 30) {
    thrust::fill(op.data_.handle_ptr->get_thrust_policy(),
                 delta_x.data(),
                 delta_x.data() + delta_x.size(),
                 0.0);
    RAFT_CHECK_CUDA(op.data_.handle_ptr->get_stream());
    op.solve(r, delta_x);

    thrust::transform(op.data_.handle_ptr->get_thrust_policy(),
                      x.data(),
                      x.data() + x.size(),
                      delta_x.data(),
                      x.data(),
                      thrust::plus<f_t>());
    RAFT_CHECK_CUDA(op.data_.handle_ptr->get_stream());
    // r = b - Ax
    raft::copy(r.data(), b.data(), b.size(), x.stream());
    op.a_multiply(-1.0, x, 1.0, r);

    f_t new_error = vector_norm_inf<f_t>(r);
    if (new_error > error) {
      raft::copy(x.data(), x_sav.data(), x.size(), x.stream());
      if (show_iterative_refinement_info) {
        CUOPT_LOG_INFO(
          "Iterative refinement. Iter %d error increased %e %e. Stopping", iter, error, new_error);
      }
      break;
    }
    error = new_error;
    raft::copy(x_sav.data(), x.data(), x.size(), x.stream());
    iter++;
    if (show_iterative_refinement_info) {
      CUOPT_LOG_INFO(
        "Iterative refinement. Iter %d error %e. || x || %.16e || dx || %.16e Continuing",
        iter,
        error,
        vector_norm2<f_t>(x),
        vector_norm2<f_t>(delta_x));
    }
  }
  return error;
}

/**
@brief Iterative refinement with GMRES as solver
 */
template <typename i_t, typename f_t, typename T>
f_t iterative_refinement_gmres(T& op,
                               const rmm::device_uvector<f_t>& b,
                               rmm::device_uvector<f_t>& x)
{
  // Parameters
  // Ideally, we do not need to restart here. But having restarts helps as a checkpoint to get
  // better solutions in case of true residual is far from the measured residual and true residuals
  // are not converging after some point
  const int max_restarts = 3;
  const int m            = 10;  // Krylov space dimension
  const f_t tol          = 1e-8;

  rmm::device_uvector<f_t> r(x.size(), x.stream());
  rmm::device_uvector<f_t> x_sav(x, x.stream());
  rmm::device_uvector<f_t> delta_x(x.size(), x.stream());

  // Host workspace for the Hessenberg matrix and other small arrays
  std::vector<std::vector<f_t>> H(m + 1, std::vector<f_t>(m, 0.0));
  std::vector<f_t> cs(m, 0.0);
  std::vector<f_t> sn(m, 0.0);
  std::vector<f_t> e1(m + 1, 0.0);
  std::vector<f_t> y(m, 0.0);

  bool show_info = false;

  f_t bnorm      = std::max(1.0, vector_norm_inf<f_t>(b));
  f_t rel_res    = 1.0;
  int outer_iter = 0;

  // r = b - A*x
  raft::copy(r.data(), b.data(), b.size(), x.stream());
  op.a_multiply(-1.0, x, 1.0, r);

  f_t norm_r = vector_norm_inf<f_t>(r);
  if (show_info) { CUOPT_LOG_INFO("GMRES IR: initial residual = %e, |b| = %e", norm_r, bnorm); }
  if (norm_r <= 1e-8) { return norm_r; }

  f_t residual      = norm_r;
  f_t best_residual = norm_r;

  // Main loop
  while (residual > tol && outer_iter < max_restarts) {
    // For right preconditioning: Apply preconditioner on Krylov directions, not on the residual.
    // So, start GMRES on r = b - A*x. v0 = r / ||r||
    std::vector<rmm::device_uvector<f_t>> V;
    std::vector<rmm::device_uvector<f_t>> Z;  // Store preconditioned vectors Z[k] = M^{-1} V[k]
    for (int k = 0; k < m + 1; ++k) {
      V.emplace_back(x.size(), x.stream());
      Z.emplace_back(x.size(), x.stream());
    }
    // v0 = r / ||r||
    f_t rnorm     = vector_norm2<f_t>(r);
    f_t inv_rnorm = (rnorm > 0) ? (f_t(1) / rnorm) : f_t(1);

    raft::copy(V[0].data(), r.data(), r.size(), x.stream());
    thrust::transform(op.data_.handle_ptr->get_thrust_policy(),
                      V[0].data(),
                      V[0].data() + V[0].size(),
                      V[0].data(),
                      scale_op<f_t>{inv_rnorm});
    RAFT_CHECK_CUDA(op.data_.handle_ptr->get_stream());
    e1.assign(m + 1, 0.0);
    e1[0] = rnorm;

    // Hessenberg building
    int k = 0;
    for (; k < m; ++k) {
      // Z[k] = M^{-1} V[k], i.e., apply right preconditioner and store
      op.solve(V[k], Z[k]);

      // Check if solve produced NaN (indicates cuDSS failure)
      f_t z_norm = vector_norm_inf<f_t>(Z[k]);
      if (!std::isfinite(z_norm)) {
        CUOPT_LOG_INFO("GMRES IR: solve at k=%d produced NaN, terminating", k);
        return std::numeric_limits<f_t>::quiet_NaN();
      }

      // w = A * Z[k]
      op.a_multiply(1.0, Z[k], 0.0, V[k + 1]);

      // Modified Gram-Schmidt orthogonalization
      for (int j = 0; j <= k; ++j) {
        // H[j][k] = dot(w, V[j])
        f_t hij = thrust::inner_product(op.data_.handle_ptr->get_thrust_policy(),
                                        V[k + 1].data(),
                                        V[k + 1].data() + x.size(),
                                        V[j].data(),
                                        f_t(0));
        RAFT_CHECK_CUDA(op.data_.handle_ptr->get_stream());
        H[j][k] = hij;
        // w -= H[j][k] * V[j]
        thrust::transform(op.data_.handle_ptr->get_thrust_policy(),
                          V[k + 1].data(),
                          V[k + 1].data() + x.size(),
                          V[j].data(),
                          V[k + 1].data(),
                          subtract_scaled_op<f_t>{hij});
        RAFT_CHECK_CUDA(op.data_.handle_ptr->get_stream());
      }

      // H[k+1][k] = ||w||
      f_t h_k1k = vector_norm2<f_t>(V[k + 1]);

      // Check for "lucky breakdown" BEFORE using h_k1k - Krylov subspace has converged
      // When h_k1k is zero, very small, or NaN, V[k+1] is in span of previous V's
      // Must check before storing in H to avoid NaN propagation in Givens rotations
      if (!std::isfinite(h_k1k) || h_k1k < 1e-14) {
        if (show_info) {
          CUOPT_LOG_INFO("GMRES IR: lucky breakdown at k=%d, h_k1k=%e (before Givens)", k, h_k1k);
        }
        // Don't store NaN in H, don't update Givens - just exit with current solution
        // k iterations are already complete and usable
        break;
      }

      H[k + 1][k] = h_k1k;

      // V[k+1] = V[k+1] / H[k+1][k]
      f_t inv_h = f_t(1) / h_k1k;
      thrust::transform(op.data_.handle_ptr->get_thrust_policy(),
                        V[k + 1].data(),
                        V[k + 1].data() + x.size(),
                        V[k + 1].data(),
                        scale_op<f_t>{inv_h});
      RAFT_CHECK_CUDA(op.data_.handle_ptr->get_stream());

      // Apply Given's rotations to new column
      for (int i = 0; i < k; ++i) {
        f_t temp    = cs[i] * H[i][k] + sn[i] * H[i + 1][k];
        H[i + 1][k] = -sn[i] * H[i][k] + cs[i] * H[i + 1][k];
        H[i][k]     = temp;
      }
      // Compute k-th Given's rotation
      f_t delta   = std::sqrt(H[k][k] * H[k][k] + H[k + 1][k] * H[k + 1][k]);
      cs[k]       = (delta == 0) ? 1.0 : H[k][k] / delta;
      sn[k]       = (delta == 0) ? 0.0 : H[k + 1][k] / delta;
      H[k][k]     = cs[k] * H[k][k] + sn[k] * H[k + 1][k];
      H[k + 1][k] = 0.0;

      // Update the residual norm
      f_t temp_e = cs[k] * e1[k] + sn[k] * e1[k + 1];
      e1[k + 1]  = -sn[k] * e1[k] + cs[k] * e1[k + 1];
      e1[k]      = temp_e;

      rel_res = std::abs(e1[k + 1]);  // / bnorm;
      if (show_info) { CUOPT_LOG_INFO("GMRES IR: iter %d residual = %e", k + 1, rel_res); }

      if (rel_res < tol) {
        k++;  // reached convergence
        break;
      }
    }  // end Arnoldi loop

    // Solve least squares H y = e
    // Back-substitution (H is (k+1)xk upper Hessenberg, cs/sin already applied)
    std::fill(y.begin(), y.end(), 0.0);
    for (int i = k - 1; i >= 0; --i) {
      f_t s = e1[i];
      for (int j = i + 1; j < k; ++j) {
        s -= H[i][j] * y[j];
      }
      // avoid inf/nan breakdown
      if (H[i][i] == 0.0) {
        y[i] = 0.0;
        break;
      } else {
        y[i] = s / H[i][i];
      }
    }

    // Compute GMRES update: delta_x = sum_j y_j * Z[j], where Z[j] = M^{-1} V[j]
    thrust::fill(op.data_.handle_ptr->get_thrust_policy(),
                 delta_x.data(),
                 delta_x.data() + delta_x.size(),
                 0.0);
    RAFT_CHECK_CUDA(op.data_.handle_ptr->get_stream());
    for (int j = 0; j < k; ++j) {
      thrust::transform(op.data_.handle_ptr->get_thrust_policy(),
                        delta_x.data(),
                        delta_x.data() + delta_x.size(),
                        Z[j].data(),
                        delta_x.data(),
                        axpy_op<f_t>{y[j]});
      RAFT_CHECK_CUDA(op.data_.handle_ptr->get_stream());
    }

    // Update x = x + delta_x
    thrust::transform(op.data_.handle_ptr->get_thrust_policy(),
                      x.data(),
                      x.data() + x.size(),
                      delta_x.data(),
                      x.data(),
                      thrust::plus<f_t>());
    RAFT_CHECK_CUDA(op.data_.handle_ptr->get_stream());
    // r = b - A*x
    raft::copy(r.data(), b.data(), b.size(), x.stream());
    op.a_multiply(-1.0, x, 1.0, r);

    residual = vector_norm_inf<f_t>(r);

    if (show_info) {
      auto l2_residual = vector_norm2<f_t>(r);
      CUOPT_LOG_INFO("GMRES IR: after outer_iter %d residual = %e, l2_residual = %e",
                     outer_iter,
                     residual,
                     l2_residual);
    }

    // Track best solution
    if (residual < best_residual) {
      best_residual = residual;
      raft::copy(x_sav.data(), x.data(), x.size(), x.stream());
    } else {
      // Residual increased or stagnated, restore best and stop
      if (show_info) {
        CUOPT_LOG_INFO(
          "GMRES IR: residual increased from %e to %e, stopping", best_residual, residual);
      }
      raft::copy(x.data(), x_sav.data(), x.size(), x.stream());
      break;
    }

    ++outer_iter;
  }
  return best_residual;
}

template <typename i_t, typename f_t, typename T>
f_t iterative_refinement(T& op, const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& x)
{
  rmm::device_uvector<f_t> d_b(b.size(), op.data_.handle_ptr->get_stream());
  raft::copy(d_b.data(), b.data(), b.size(), op.data_.handle_ptr->get_stream());
  rmm::device_uvector<f_t> d_x(x.size(), op.data_.handle_ptr->get_stream());
  raft::copy(d_x.data(), x.data(), x.size(), op.data_.handle_ptr->get_stream());
  auto err = iterative_refinement_gmres<i_t, f_t, T>(op, d_b, d_x);

  raft::copy(x.data(), d_x.data(), x.size(), op.data_.handle_ptr->get_stream());

  RAFT_CUDA_TRY(cudaStreamSynchronize(op.data_.handle_ptr->get_stream()));
  return err;
}

template <typename i_t, typename f_t, typename T>
f_t iterative_refinement(T& op, const rmm::device_uvector<f_t>& b, rmm::device_uvector<f_t>& x)
{
  return iterative_refinement_gmres<i_t, f_t, T>(op, b, x);
}

}  // namespace cuopt::linear_programming::dual_simplex
