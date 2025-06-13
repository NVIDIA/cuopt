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

#pragma once

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/types.hpp>

#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
i_t column_scaling(const lp_problem_t<i_t, f_t>& unscaled,
                   const simplex_solver_settings_t<i_t, f_t>& settings,
                   lp_problem_t<i_t, f_t>& scaled,
                   std::vector<f_t>& column_scaling);

template <typename i_t, typename f_t>
void unscale_solution(const std::vector<f_t>& column_scaling,
                      const std::vector<f_t>& scaled_x,
                      const std::vector<f_t>& scaled_z,
                      std::vector<f_t>& unscaled_x,
                      std::vector<f_t>& unscaled_z);

}  // namespace cuopt::linear_programming::dual_simplex
