/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
i_t preconditioned_conjugate_gradient(const T& op,
                                      const simplex_solver_settings_t<i_t, f_t>& settings,
                                      const dense_vector_t<i_t, f_t>& b,
                                      f_t tolerance,
                                      dense_vector_t<i_t, f_t>& xinout)
{
  const bool show_pcg_info          = false;
  dense_vector_t<i_t, f_t> residual = b;
  dense_vector_t<i_t, f_t> y(b.size());

  dense_vector_t<i_t, f_t> x = xinout;

  // r = A*x - b
  op.a_multiply(1.0, x, -1.0, residual);

  // Solve M y = r for y
  op.m_solve(residual, y);

  dense_vector_t<i_t, f_t> p = y;
  p.multiply_scalar(-1.0);

  dense_vector_t<i_t, f_t> Ap(b.size());
  i_t iter                  = 0;
  f_t norm_residual         = vector_norm2<i_t, f_t>(residual);
  f_t initial_norm_residual = norm_residual;
  if (show_pcg_info) {
    settings.log.printf("PCG initial residual 2-norm %e inf-norm %e\n",
                        norm_residual,
                        vector_norm_inf<i_t, f_t>(residual));
  }

  f_t rTy = residual.inner_product(y);

  while (norm_residual > tolerance && iter < 100) {
    // Compute Ap = A * p
    op.a_multiply(1.0, p, 0.0, Ap);

    // Compute alpha = (r^T * y) / (p^T * Ap)
    f_t alpha = rTy / p.inner_product(Ap);

    // Update residual = residual + alpha * Ap
    residual.axpy(alpha, Ap, 1.0);

    f_t new_residual = vector_norm2<i_t, f_t>(residual);
    if (new_residual > 1.1 * norm_residual || new_residual > 1.1 * initial_norm_residual) {
      if (show_pcg_info) {
        settings.log.printf(
          "PCG residual increased by more than 10%%. New %e > %e\n", new_residual, norm_residual);
      }
      break;
    }
    norm_residual = new_residual;

    // Update x = x + alpha * p
    x.axpy(alpha, p, 1.0);

    // residual = A*x - b
    residual = b;
    op.a_multiply(1.0, x, -1.0, residual);
    norm_residual = vector_norm2<i_t, f_t>(residual);

    // Solve M y = r for y
    op.m_solve(residual, y);

    // Compute beta = (r_+^T y_+) / (r^T y)
    f_t r1Ty1 = residual.inner_product(y);
    f_t beta  = r1Ty1 / rTy;

    rTy = r1Ty1;

    // Update p = -y + beta * p
    p.axpy(-1.0, y, beta);

    iter++;

    if (show_pcg_info) {
      settings.log.printf("PCG iter %3d 2-norm_residual %.2e inf-norm_residual %.2e\n",
                          iter,
                          norm_residual,
                          vector_norm_inf<i_t, f_t>(residual));
    }
  }

  residual = b;
  op.a_multiply(1.0, x, -1.0, residual);
  norm_residual = vector_norm2<i_t, f_t>(residual);
  if (norm_residual < initial_norm_residual) {
    if (show_pcg_info) {
      settings.log.printf("PCG improved residual 2-norm %.2e/%.2e in %d iterations\n",
                          norm_residual,
                          initial_norm_residual,
                          iter);
    }
    xinout = x;
  } else {
    if (show_pcg_info) { settings.log.printf("Rejecting PCG solution\n"); }
  }

  return iter;
}

}  // namespace cuopt::linear_programming::dual_simplex
