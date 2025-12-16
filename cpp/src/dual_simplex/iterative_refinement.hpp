/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include "dual_simplex/dense_vector.hpp"
#include "dual_simplex/simplex_solver_settings.hpp"
#include "dual_simplex/types.hpp"
#include "dual_simplex/vector_math.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t, typename T>
void iterative_refinement_simple(T& op,
                                 const dense_vector_t<i_t, f_t>& b,
                                 dense_vector_t<i_t, f_t>& x)
{
  dense_vector_t<i_t, f_t> x_sav            = x;
  dense_vector_t<i_t, f_t> r                = b;
  const bool show_iterative_refinement_info = false;

  op.a_multiply(-1.0, x, 1.0, r);

  f_t error = vector_norm_inf<i_t, f_t>(r);
  if (show_iterative_refinement_info) {
    CUOPT_LOG_INFO(
      "Iterative refinement. Initial error %e || x || %.16e", error, vector_norm2<i_t, f_t>(x));
  }
  dense_vector_t<i_t, f_t> delta_x(x.size());
  i_t iter = 0;
  while (error > 1e-8 && iter < 30) {
    delta_x.set_scalar(0.0);
    op.solve(r, delta_x);

    x.axpy(1.0, delta_x, 1.0);

    r = b;
    op.a_multiply(-1.0, x, 1.0, r);

    f_t new_error = vector_norm_inf<i_t, f_t>(r);
    if (new_error > error) {
      x = x_sav;
      if (show_iterative_refinement_info) {
        CUOPT_LOG_INFO(
          "Iterative refinement. Iter %d error increased %e %e. Stopping", iter, error, new_error);
      }
      break;
    }
    error = new_error;
    x_sav = x;
    iter++;
    if (show_iterative_refinement_info) {
      CUOPT_LOG_INFO(
        "Iterative refinement. Iter %d error %e. || x || %.16e || dx || %.16e Continuing",
        iter,
        error,
        vector_norm2<i_t, f_t>(x),
        vector_norm2<i_t, f_t>(delta_x));
    }
  }
}

/**
@brief Iterative refinement with GMRES as solver
 */
template <typename i_t, typename f_t, typename T>
void iterative_refinement_gmres(T& op,
                                const dense_vector_t<i_t, f_t>& b,
                                dense_vector_t<i_t, f_t>& x)
{
  // Parameters
  // Ideally, we do not need to restart here. But having restarts helps as a checkpoint to get
  // better solutions in case of true residual is far from the measured residual and true residuals
  // are not converging after some point
  const int max_restarts = 3;
  const int m            = 10;  // Krylov space dimension
  const f_t tol          = 1e-8;

  dense_vector_t<i_t, f_t> r(x.size());
  dense_vector_t<i_t, f_t> x_sav = x;
  dense_vector_t<i_t, f_t> delta_x(x.size());

  // Host workspace for the Hessenberg matrix and other small arrays
  std::vector<std::vector<f_t>> H(m + 1, std::vector<f_t>(m, 0.0));
  std::vector<f_t> cs(m, 0.0);
  std::vector<f_t> sn(m, 0.0);
  std::vector<f_t> e1(m + 1, 0.0);
  std::vector<f_t> y(m, 0.0);

  bool show_info = false;

  f_t bnorm      = max(1.0, vector_norm_inf<i_t, f_t>(b));
  f_t rel_res    = 1.0;
  int outer_iter = 0;

  // r = b - A*x
  r = b;
  op.a_multiply(-1.0, x, 1.0, r);

  f_t norm_r = vector_norm_inf<i_t, f_t>(r);
  if (show_info) { CUOPT_LOG_INFO("GMRES IR: initial residual = %e, |b| = %e", norm_r, bnorm); }
  if (norm_r <= 1e-8) { return; }

  f_t residual      = norm_r;
  f_t best_residual = norm_r;

  // Main loop
  while (residual > tol && outer_iter < max_restarts) {
    // For right preconditioning: Apply preconditioner on Krylov directions, not on the residual.
    // So, start GMRES on r = b - A*x. v0 = r / ||r||
    std::vector<dense_vector_t<i_t, f_t>> V;
    std::vector<dense_vector_t<i_t, f_t>> Z;  // Store preconditioned vectors Z[k] = M^{-1} V[k]
    for (int k = 0; k < m + 1; ++k) {
      V.emplace_back(x.size());
      Z.emplace_back(x.size());
    }
    // v0 = r / ||r||
    f_t rnorm     = vector_norm2<i_t, f_t>(r);
    f_t inv_rnorm = (rnorm > 0) ? (f_t(1) / rnorm) : f_t(1);
    V[0]          = r;
    V[0].multiply_scalar(inv_rnorm);
    e1.assign(m + 1, 0.0);
    e1[0] = rnorm;

    // Hessenberg building
    int k = 0;
    for (; k < m; ++k) {
      // Z[k] = M^{-1} V[k], i.e., apply right preconditioner and store
      op.solve(V[k], Z[k]);

      // w = A * Z[k]
      op.a_multiply(1.0, Z[k], 0.0, V[k + 1]);

      // Modified Gram-Schmidt orthogonalization
      for (int j = 0; j <= k; ++j) {
        // H[j][k] = dot(w, V[j])
        f_t hij = V[k + 1].inner_product(V[j]);
        H[j][k] = hij;
        // w -= H[j][k] * V[j]
        V[k + 1].axpy(-hij, V[j], 1.0);
      }

      // H[k+1][k] = ||w||
      f_t h_k1k   = vector_norm2<i_t, f_t>(V[k + 1]);
      H[k + 1][k] = h_k1k;
      if (h_k1k != 0.0) {
        // V[k+1] = V[k+1] / H[k+1][k]
        f_t inv_h = f_t(1) / h_k1k;
        V[k + 1].multiply_scalar(inv_h);
      }

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
      y[i] = s / H[i][i];
    }

    // Compute GMRES update: delta_x = sum_j y_j * Z[j], where Z[j] = M^{-1} V[j]
    std::fill(delta_x.begin(), delta_x.end(), 0.0);
    for (int j = 0; j < k; ++j) {
      delta_x.axpy(y[j], Z[j], 1.0);
    }

    // Update x = x + delta_x
    x.axpy(1.0, delta_x, 1.0);

    // r = b - A*x
    r = b;
    op.a_multiply(-1.0, x, 1.0, r);

    residual = vector_norm_inf<i_t, f_t>(r);

    if (show_info) {
      auto l2_residual = vector_norm2<i_t, f_t>(r);
      CUOPT_LOG_INFO("GMRES IR: after outer_iter %d residual = %e, l2_residual = %e",
                     outer_iter,
                     residual,
                     l2_residual);
    }

    // Track best solution
    if (residual < best_residual) {
      best_residual = residual;
      x_sav         = x;
    } else {
      // Residual increased or stagnated, restore best and stop
      if (show_info) {
        CUOPT_LOG_INFO(
          "GMRES IR: residual increased from %e to %e, stopping", best_residual, residual);
      }
      x = x_sav;
      break;
    }

    ++outer_iter;
  }
}

template <typename i_t, typename f_t, typename T>
void iterative_refinement(T& op, const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& x)
{
  const bool is_qp = op.data_.Q.n > 0;
  if (is_qp) {
    iterative_refinement_gmres(op, b, x);
  } else {
    iterative_refinement_simple(op, b, x);
  }
  return;
}

}  // namespace cuopt::linear_programming::dual_simplex
