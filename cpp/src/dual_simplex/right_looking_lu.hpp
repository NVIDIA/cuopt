/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/simplex_solver_settings.hpp>
#include <linear_algebra/sparse_matrix.hpp>

#include <optional>

namespace cuopt::mathematical_optimization::simplex {

template <typename i_t, typename f_t>
i_t right_looking_lu(const csc_matrix_t<i_t, f_t>& A,
                     const simplex_solver_settings_t<i_t, f_t>& settings,
                     f_t tol,
                     const std::vector<i_t>& column_list,
                     f_t start_time,
                     std::vector<i_t>& q,
                     csc_matrix_t<i_t, f_t>& L,
                     csc_matrix_t<i_t, f_t>& U,
                     std::vector<i_t>& pinv,
                     f_t& work_estimate);

template <typename i_t, typename f_t>
i_t right_looking_lu_row_permutation_only(const csc_matrix_t<i_t, f_t>& A,
                                          const simplex_solver_settings_t<i_t, f_t>& settings,
                                          f_t tol,
                                          f_t start_time,
                                          std::vector<i_t>& q,
                                          std::vector<i_t>& pinv);

// Sparse LDL^T factorization with symmetric Markowitz pivoting.
// Computes P * A * P^T = L * D * L^T for a symmetric positive semidefinite matrix A.
// Input: A is n x n symmetric PSD, stored as lower triangle only in CSC format.
// Output:
//   perm[k] = original index of the k-th pivot (length = rank)
//   L = unit lower triangular factor (CSC, with permuted row indices)
//   D = diagonal factor (length = rank)
// Returns:
//   rank >= 0: number of successful pivots with D(k,k) >= pivot_tol (PSD case).
//   INDEFINITE_MATRIX_RETURN (-4): a negative pivot was encountered, or the matrix is nonzero
//     but no acceptable diagonal pivot was found (matrix is not PSD).
//   CONCURRENT_HALT_RETURN (-2): concurrent halt requested.
//   TIME_LIMIT_RETURN (-3): time limit exceeded.
template <typename i_t, typename f_t>
i_t right_looking_ldlt(const csc_matrix_t<i_t, f_t>& A,
                       const simplex_solver_settings_t<i_t, f_t>& settings,
                       f_t pivot_tol,
                       f_t start_time,
                       std::vector<i_t>& perm,
                       csc_matrix_t<i_t, f_t>& L,
                       std::vector<f_t>& D,
                       f_t& work_estimate);

}  // namespace cuopt::mathematical_optimization::simplex
