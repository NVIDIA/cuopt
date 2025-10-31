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

#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/sparse_matrix.hpp>

#include <optional>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
i_t right_looking_lu(const csc_matrix_t<i_t, f_t>& A,
                     const simplex_solver_settings_t<i_t, f_t>& settings,
                     f_t tol,
                     const std::vector<i_t>& column_list,
                     std::vector<i_t>& q,
                     csc_matrix_t<i_t, f_t>& L,
                     csc_matrix_t<i_t, f_t>& U,
                     std::vector<i_t>& pinv);

template <typename i_t, typename f_t>
i_t right_looking_lu_row_permutation_only(const csc_matrix_t<i_t, f_t>& A,
                                          const simplex_solver_settings_t<i_t, f_t>& settings,
                                          f_t tol,
                                          f_t start_time,
                                          std::vector<i_t>& q,
                                          std::vector<i_t>& pinv);

}  // namespace cuopt::linear_programming::dual_simplex
