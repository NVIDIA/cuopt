/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <dual_simplex/scaling.hpp>
#include <dual_simplex/sparse_matrix.hpp>

#include <cmath>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
i_t column_scaling(const lp_problem_t<i_t, f_t>& unscaled,
                   const simplex_solver_settings_t<i_t, f_t>& settings,
                   lp_problem_t<i_t, f_t>& scaled,
                   std::vector<f_t>& column_scaling)
{
  scaled = unscaled;
  i_t m  = scaled.num_rows;
  i_t n  = scaled.num_cols;

  if (!settings.scale_columns) {
    settings.log.printf("Skipping column scaling\n");
    column_scaling.resize(n, 1.0);
    return 0;
  }

  column_scaling.resize(n);
  f_t max = 0;
  for (i_t j = 0; j < n; ++j) {
    const i_t col_start = scaled.A.col_start[j];
    const i_t col_end   = scaled.A.col_start[j + 1];
    f_t sum             = 0.0;
    for (i_t p = col_start; p < col_end; ++p) {
      const f_t x = scaled.A.x[p];
      sum += x * x;
    }
    f_t col_norm_j = column_scaling[j] = sum > 0 ? std::sqrt(sum) : 1.0;
    max                                = std::max(col_norm_j, max);
  }
  settings.log.printf("Scaling matrix. Maximum column norm %e\n", max);
  // C(j, j) = 1/column_scaling(j)

  // scaled_A = unscaled_A * C
  for (i_t j = 0; j < n; ++j) {
    const i_t col_start = scaled.A.col_start[j];
    const i_t col_end   = scaled.A.col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      scaled.A.x[p] /= column_scaling[j];
    }
  }
  // scaled_obj = C*unscaled_obj
  for (i_t j = 0; j < n; ++j) {
    scaled.objective[j] /= column_scaling[j];
  }
  // scaled_lower = C^{-1} * unscaled_lower
  // scaled_upper = C^{-1} * unscaled_upper
  for (i_t j = 0; j < n; ++j) {
    scaled.lower[j] *= column_scaling[j];
    scaled.upper[j] *= column_scaling[j];
  }

  return 0;
}

template <typename i_t, typename f_t>
void unscale_solution(const std::vector<f_t>& column_scaling,
                      const std::vector<f_t>& scaled_x,
                      const std::vector<f_t>& scaled_z,
                      std::vector<f_t>& unscaled_x,
                      std::vector<f_t>& unscaled_z)
{
  const i_t n = scaled_x.size();
  unscaled_x.resize(n);
  unscaled_z.resize(n);
  for (i_t j = 0; j < n; ++j) {
    unscaled_x[j] = scaled_x[j] / column_scaling[j];
    unscaled_z[j] = scaled_z[j] / column_scaling[j];
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template int column_scaling<int, double>(const lp_problem_t<int, double>& unscaled,
                                         const simplex_solver_settings_t<int, double>& settings,
                                         lp_problem_t<int, double>& scaled,
                                         std::vector<double>& column_scaling);

template void unscale_solution<int, double>(const std::vector<double>& column_scaling,
                                            const std::vector<double>& scaled_x,
                                            const std::vector<double>& scaled_z,
                                            std::vector<double>& unscaled_x,
                                            std::vector<double>& unscaled_z);

#endif

}  // namespace cuopt::linear_programming::dual_simplex
