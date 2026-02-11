/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/sparse_matrix.hpp>

#include <optional>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t, typename VectorI>
i_t right_looking_lu(const csc_matrix_t<i_t, f_t>& A,
                     const simplex_solver_settings_t<i_t, f_t>& settings,
                     f_t tol,
                     const VectorI& column_list,
                     VectorI& q,
                     csc_matrix_t<i_t, f_t>& L,
                     csc_matrix_t<i_t, f_t>& U,
                     VectorI& pinv);

template <typename i_t, typename f_t>
i_t right_looking_lu_row_permutation_only(const csc_matrix_t<i_t, f_t>& A,
                                          const simplex_solver_settings_t<i_t, f_t>& settings,
                                          f_t tol,
                                          f_t start_time,
                                          std::vector<i_t>& q,
                                          std::vector<i_t>& pinv);

}  // namespace cuopt::linear_programming::dual_simplex
