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
void iterative_refinement(const T& op,
                          const dense_vector_t<i_t, f_t>& b,
                          dense_vector_t<i_t, f_t>& x)
{
  dense_vector_t<i_t, f_t> x_sav            = x;
  dense_vector_t<i_t, f_t> r                = b;
  const bool show_iterative_refinement_info = false;

  op.a_multiply(-1.0, x, 1.0, r);

  f_t error = vector_norm_inf<i_t, f_t>(r);
  if (show_iterative_refinement_info) {
    printf(
      "Iterative refinement. Initial error %e || x || %.16e\n", error, vector_norm2<i_t, f_t>(x));
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
        printf("%d Iterative refinement error increased %e %e. Stopping\n", iter, error, new_error);
      }
      break;
    }
    error = new_error;
    x_sav = x;
    iter++;
    if (show_iterative_refinement_info) {
      printf("%d Iterative refinement error %e. || x || %.16e || dx || %.16e Continuing\n",
             iter,
             error,
             vector_norm2<i_t, f_t>(x),
             vector_norm2<i_t, f_t>(delta_x));
    }
  }
}

}  // namespace cuopt::linear_programming::dual_simplex
