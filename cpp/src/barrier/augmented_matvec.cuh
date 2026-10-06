/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime.h>

namespace cuopt::mathematical_optimization::barrier {

template <typename i_t, typename f_t>
__global__ void prepare_augmented_matvec(const f_t* x,
                                         const f_t* y,
                                         const f_t* diag,
                                         const i_t* free_linear,
                                         f_t* x1,
                                         f_t* x2,
                                         f_t* y1,
                                         f_t* y2,
                                         f_t* r1,
                                         f_t* y_exp,
                                         f_t* y_exp_orig,
                                         i_t n,
                                         i_t m,
                                         i_t p,
                                         i_t linear_n)
{
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
__global__ void finish_augmented_matvec(f_t* y,
                                        const f_t* y1,
                                        const f_t* y2,
                                        const f_t* y_exp,
                                        const f_t* y_exp_orig,
                                        f_t alpha,
                                        f_t beta,
                                        i_t n,
                                        i_t m,
                                        i_t p)
{
  const i_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { y[i] = y1[i]; }
  if (i < m) { y[n + i] = y2[i]; }
  if (i < p) { y[n + m + i] = alpha * y_exp[i] + beta * y_exp_orig[i]; }
}

}  // namespace cuopt::mathematical_optimization::barrier
