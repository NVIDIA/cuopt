/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <barrier/device_sparse_matrix.cuh>
#include <linear_algebra/dense_vector.hpp>

#include <dual_simplex/simplex_solver_settings.hpp>
#include <linear_algebra/vector_math.cuh>
#include <linear_algebra/vector_math.hpp>
#include <math_optimization/types.hpp>

#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::barrier {

template <typename f_t, int dimension>
struct gmres_update_op {
  const f_t* x;
  const f_t* basis[dimension];
  f_t coefficients[dimension];
  int count;

  __device__ f_t operator()(size_t i) const
  {
    f_t delta = 0;
    for (int j = 0; j < count; ++j) {
      delta += coefficients[j] * basis[j][i];
    }
    return x[i] + delta;
  }
};

// Reuse device storage across the affine/corrector solves and GMRES restarts.
template <typename f_t>
struct gmres_workspace_t {
  static constexpr int dimension = 10;
  rmm::device_uvector<f_t> r, x_sav;
  std::vector<rmm::device_uvector<f_t>> V, Z;
  transform_reduce_helper_t<f_t> reductions;
  rmm::device_uvector<f_t> d_column;
  std::array<f_t, dimension + 2> h_column;

  explicit gmres_workspace_t(cuda::stream_ref stream)
    : r(0, stream), x_sav(0, stream), reductions(stream), d_column(dimension + 2, stream)
  {
  }

  void resize(size_t n, cuda::stream_ref stream)
  {
    r.resize(n, stream);
    x_sav.resize(n, stream);
  }

  void resize_basis(size_t n, cuda::stream_ref stream)
  {
    if (V.empty()) {
      for (int k = 0; k <= dimension; ++k) {
        V.emplace_back(n, stream);
        Z.emplace_back(n, stream);
      }
    } else {
      for (int k = 0; k <= dimension; ++k) {
        V[k].resize(n, stream);
        Z[k].resize(n, stream);
      }
    }
  }

  f_t norm_inf(const rmm::device_uvector<f_t>& x)
  {
    return reductions.transform_reduce(
      x.data(),
      thrust::maximum<f_t>{},
      [] __host__ __device__(f_t v) { return abs(v); },
      f_t(0),
      x.size(),
      x.stream());
  }

  f_t norm2(const rmm::device_uvector<f_t>& x)
  {
    return std::sqrt(reductions.transform_reduce(
      x.data(),
      thrust::plus<f_t>{},
      [] __host__ __device__(f_t v) { return v * v; },
      f_t(0),
      x.size(),
      x.stream()));
  }

  void update(rmm::device_uvector<f_t>& x, const std::vector<f_t>& coefficients, int count)
  {
    gmres_update_op<f_t, dimension> op{};
    op.x     = x.data();
    op.count = count;
    for (int j = 0; j < count; ++j) {
      op.basis[j]        = Z[j].data();
      op.coefficients[j] = coefficients[j];
    }
    auto first = thrust::make_counting_iterator<size_t>(0);
    thrust::transform(rmm::exec_policy_nosync(x.stream()), first, first + x.size(), x.begin(), op);
  }

  template <typename Input, typename Reduce, typename Transform>
  void reduce_async(
    Input input, Reduce reduce, Transform transform, size_t n, f_t* output, cuda::stream_ref stream)
  {
    cub::DeviceReduce::TransformReduce(
      nullptr, reductions.buffer_size, input, output, n, reduce, transform, f_t(0), stream.get());
    reductions.buffer_data.resize(reductions.buffer_size, stream);
    cub::DeviceReduce::TransformReduce(reductions.buffer_data.data(),
                                       reductions.buffer_size,
                                       input,
                                       output,
                                       n,
                                       reduce,
                                       transform,
                                       f_t(0),
                                       stream.get());
  }

  void check_preconditioner_async(const rmm::device_uvector<f_t>& z)
  {
    reduce_async(
      z.data(),
      thrust::maximum<f_t>{},
      [] __host__ __device__(f_t v) { return abs(v); },
      z.size(),
      d_column.data() + dimension + 1,
      z.stream());
  }

  void orthogonalize(int k)
  {
    auto& w           = V[k + 1];
    const auto stream = w.stream();
    for (int j = 0; j <= k; ++j) {
      const f_t* wp  = w.data();
      const f_t* vp  = V[j].data();
      const f_t* hij = d_column.data() + j;
      reduce_async(
        thrust::make_counting_iterator<size_t>(0),
        thrust::plus<f_t>{},
        [wp, vp] __device__(size_t i) -> f_t { return wp[i] * vp[i]; },
        w.size(),
        d_column.data() + j,
        stream);
      thrust::transform(rmm::exec_policy_nosync(stream),
                        w.begin(),
                        w.end(),
                        V[j].begin(),
                        w.begin(),
                        [hij] __device__(f_t a, f_t b) { return a - *hij * b; });
    }
    reduce_async(
      w.data(),
      thrust::plus<f_t>{},
      [] __host__ __device__(f_t v) { return v * v; },
      w.size(),
      d_column.data() + k + 1,
      stream);
    raft::copy(h_column.data(), d_column.data(), h_column.size(), stream);
    stream.sync();
  }
};

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

template <typename i_t, typename f_t, typename T>
f_t iterative_refinement_simple(T& op,
                                const rmm::device_uvector<f_t>& b,
                                rmm::device_uvector<f_t>& x,
                                f_t tol)
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
  while (error > tol && iter < 30) {
    thrust::fill(rmm::exec_policy_nosync(op.data_.handle_ptr->get_stream()),
                 delta_x.data(),
                 delta_x.data() + delta_x.size(),
                 0.0);
    RAFT_CHECK_CUDA(op.data_.handle_ptr->get_stream().get());
    op.solve(r, delta_x);

    thrust::transform(rmm::exec_policy_nosync(op.data_.handle_ptr->get_stream()),
                      x.data(),
                      x.data() + x.size(),
                      delta_x.data(),
                      x.data(),
                      thrust::plus<f_t>());
    RAFT_CHECK_CUDA(op.data_.handle_ptr->get_stream().get());
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
                               rmm::device_uvector<f_t>& x,
                               f_t tol)
{
  // Parameters
  // Ideally, we do not need to restart here. But having restarts helps as a checkpoint to get
  // better solutions in case of true residual is far from the measured residual and true residuals
  // are not converging after some point
  const int max_restarts = 3;
  const int m            = gmres_workspace_t<f_t>::dimension;
  auto& workspace        = op.data_.gmres_workspace_;
  workspace.resize(x.size(), x.stream());
  auto& r     = workspace.r;
  auto& x_sav = workspace.x_sav;
  raft::copy(x_sav.data(), x.data(), x.size(), x.stream());

  // Host workspace for the Hessenberg matrix and other small arrays
  std::vector<std::vector<f_t>> H(m + 1, std::vector<f_t>(m, 0.0));
  std::vector<f_t> cs(m, 0.0);
  std::vector<f_t> sn(m, 0.0);
  std::vector<f_t> e1(m + 1, 0.0);
  std::vector<f_t> y(m, 0.0);

  bool show_info = false;

  f_t stop_ratio = 5.0;
  f_t bnorm      = show_info ? std::max(f_t(1), workspace.norm_inf(b)) : f_t(1);
  f_t rel_res    = 1.0;
  int outer_iter = 0;

  // r = b - A*x
  raft::copy(r.data(), b.data(), b.size(), x.stream());
  op.a_multiply(-1.0, x, 1.0, r);

  f_t norm_r = workspace.norm_inf(r);
  if (show_info) { CUOPT_LOG_INFO("GMRES IR: initial residual = %e, |b| = %e", norm_r, bnorm); }
  if (norm_r <= tol) { return norm_r; }

  f_t residual      = norm_r;
  f_t best_residual = norm_r;

  // Main loop
  while (residual > tol && outer_iter < max_restarts) {
    // For right preconditioning: Apply preconditioner on Krylov directions, not on the residual.
    // So, start GMRES on r = b - A*x. v0 = r / ||r||
    workspace.resize_basis(x.size(), x.stream());
    auto& V = workspace.V;
    auto& Z = workspace.Z;
    // v0 = r / ||r||
    f_t rnorm     = workspace.norm2(r);
    f_t inv_rnorm = (rnorm > 0) ? (f_t(1) / rnorm) : f_t(1);

    thrust::transform(rmm::exec_policy_nosync(op.data_.handle_ptr->get_stream()),
                      r.data(),
                      r.data() + r.size(),
                      V[0].data(),
                      scale_op<f_t>{inv_rnorm});
    RAFT_CHECK_CUDA(op.data_.handle_ptr->get_stream().get());
    e1.assign(m + 1, 0.0);
    e1[0] = rnorm;

    // Hessenberg building
    int k = 0;
    for (; k < m; ++k) {
      // Z[k] = M^{-1} V[k], i.e., apply right preconditioner and store
      op.solve(V[k], Z[k]);

      // Check if solve produced NaN (indicates cuDSS failure)
      workspace.check_preconditioner_async(Z[k]);

      // Stream-ordered dot/update pairs retain modified Gram-Schmidt while
      // deferring host scalar reads until the complete column is available.
      op.a_multiply(1.0, Z[k], 0.0, V[k + 1]);
      workspace.orthogonalize(k);
      f_t z_norm = workspace.h_column[m + 1];
      if (!std::isfinite(z_norm)) {
        CUOPT_LOG_INFO("GMRES IR: solve at k=%d produced NaN, terminating", k);
        return std::numeric_limits<f_t>::quiet_NaN();
      }

      for (int j = 0; j <= k; ++j) {
        H[j][k] = workspace.h_column[j];
      }

      // H[k+1][k] = ||w||
      f_t h_k1k = std::sqrt(workspace.h_column[k + 1]);

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
      thrust::transform(rmm::exec_policy_nosync(op.data_.handle_ptr->get_stream()),
                        V[k + 1].data(),
                        V[k + 1].data() + x.size(),
                        V[k + 1].data(),
                        scale_op<f_t>{inv_h});
      RAFT_CHECK_CUDA(op.data_.handle_ptr->get_stream().get());

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

    // Preserve the sequential basis accumulation without a temporary vector.
    workspace.update(x, y, k);
    RAFT_CHECK_CUDA(op.data_.handle_ptr->get_stream().get());
    // r = b - A*x
    raft::copy(r.data(), b.data(), b.size(), x.stream());
    op.a_multiply(-1.0, x, 1.0, r);

    residual = workspace.norm_inf(r);

    if (show_info) {
      auto l2_residual = workspace.norm2(r);
      CUOPT_LOG_INFO("GMRES IR: after outer_iter %d residual = %e, l2_residual = %e",
                     outer_iter,
                     residual,
                     l2_residual);
    }

    f_t improvement_ratio = best_residual / residual;
    // Track best solution
    if (improvement_ratio >= stop_ratio) {
      best_residual = residual;
      raft::copy(x_sav.data(), x.data(), x.size(), x.stream());
    } else if (improvement_ratio < stop_ratio && improvement_ratio > 1.0) {
      best_residual = residual;
      raft::copy(x_sav.data(), x.data(), x.size(), x.stream());
      // Residual decreased, but not enough, continue
      if (show_info) {
        CUOPT_LOG_INFO("GMRES IR: improvement ratio %e is less than %e, breaking early",
                       improvement_ratio,
                       stop_ratio);
      }
      break;
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
f_t iterative_refinement(T& op,
                         const dense_vector_t<i_t, f_t>& b,
                         dense_vector_t<i_t, f_t>& x,
                         f_t tol = 1e-8)
{
  rmm::device_uvector<f_t> d_b(b.size(), op.data_.handle_ptr->get_stream());
  raft::copy(d_b.data(), b.data(), b.size(), op.data_.handle_ptr->get_stream());
  rmm::device_uvector<f_t> d_x(x.size(), op.data_.handle_ptr->get_stream());
  raft::copy(d_x.data(), x.data(), x.size(), op.data_.handle_ptr->get_stream());
  auto err = iterative_refinement_gmres<i_t, f_t, T>(op, d_b, d_x, tol);

  raft::copy(x.data(), d_x.data(), x.size(), op.data_.handle_ptr->get_stream());

  op.data_.handle_ptr->get_stream().sync();
  return err;
}

template <typename i_t, typename f_t, typename T>
f_t iterative_refinement(T& op,
                         const rmm::device_uvector<f_t>& b,
                         rmm::device_uvector<f_t>& x,
                         f_t tol)
{
  return iterative_refinement_gmres<i_t, f_t, T>(op, b, x, tol);
}

}  // namespace cuopt::mathematical_optimization::barrier
