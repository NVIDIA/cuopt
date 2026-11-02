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
                                         i_t linear_n)
{
  const i_t n = static_cast<i_t>(x1.size());
  const i_t m = static_cast<i_t>(x2.size());
  const i_t p = static_cast<i_t>(y_exp.size());
  const i_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x1[i] = x[i];
    y1[i] = y[i];
    r1[i] = i < linear_n && !free_linear[i] ? x[i] * diag[i] : f_t(0);
  }
  if (i < m) {
    x2[i] = x[n + i];
    y2[i] = y[n + i];
  }
  if (i < p) {
    y_exp[i]      = f_t(0);
    y_exp_orig[i] = y[n + m + i];
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
