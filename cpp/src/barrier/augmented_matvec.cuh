/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/device_span.hpp>

#include <cuda_runtime.h>

namespace cuopt::mathematical_optimization::barrier {

template <typename i_t, typename f_t>
__global__ void prepare_augmented_matvec(raft::device_span<const f_t> x,
                                         raft::device_span<const f_t> y,
                                         raft::device_span<const f_t> diag,
                                         raft::device_span<const i_t> free_linear,
                                         raft::device_span<f_t> x1,
                                         raft::device_span<f_t> x2,
                                         raft::device_span<f_t> y1,
                                         raft::device_span<f_t> y2,
                                         raft::device_span<f_t> r1,
                                         raft::device_span<f_t> y_exp,
                                         raft::device_span<f_t> y_exp_orig,
                                         i_t linear_n,
                                         f_t beta)
{
  // beta == 0: y is output-only (it may be uninitialized, e.g. a fresh GMRES Krylov vector),
  // so never read it; 0 * NaN would otherwise poison the product.
  const bool read_y = beta != f_t(0);
  const i_t n       = static_cast<i_t>(x1.size());
  const i_t m       = static_cast<i_t>(x2.size());
  const i_t p       = static_cast<i_t>(y_exp.size());
  const i_t i       = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x1[i] = x[i];
    y1[i] = read_y ? y[i] : f_t(0);
    r1[i] = i < linear_n && !free_linear[i] ? x[i] * diag[i] : f_t(0);
  }
  if (i < m) {
    x2[i] = x[n + i];
    y2[i] = read_y ? y[n + i] : f_t(0);
  }
  if (i < p) {
    y_exp[i]      = f_t(0);
    y_exp_orig[i] = read_y ? y[n + m + i] : f_t(0);
  }
}

template <typename i_t, typename f_t>
__global__ void finish_augmented_matvec(raft::device_span<f_t> y,
                                        raft::device_span<const f_t> y1,
                                        raft::device_span<const f_t> y2,
                                        raft::device_span<const f_t> y_exp,
                                        raft::device_span<const f_t> y_exp_orig,
                                        f_t alpha,
                                        f_t beta)
{
  const i_t n = static_cast<i_t>(y1.size());
  const i_t m = static_cast<i_t>(y2.size());
  const i_t p = static_cast<i_t>(y_exp.size());
  const i_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { y[i] = y1[i]; }
  if (i < m) { y[n + i] = y2[i]; }
  if (i < p) { y[n + m + i] = alpha * y_exp[i] + beta * y_exp_orig[i]; }
}

}  // namespace cuopt::mathematical_optimization::barrier
