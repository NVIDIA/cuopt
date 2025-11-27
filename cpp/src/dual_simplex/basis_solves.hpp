/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/types.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t>
i_t reorder_basic_list(const std::vector<i_t>& q, std::vector<i_t>& basic_list);

// Get basis_list and nonbasic_list from vstatus
template <typename i_t>
void get_basis_from_vstatus(i_t m,
                            const std::vector<variable_status_t>& vstatus,
                            std::vector<i_t>& basis_list,
                            std::vector<i_t>& nonbasic_list,
                            std::vector<i_t>& superbasic_list);

// Factorize the basis matrix B = A(:, basis_list). P*B*Q = L*U
template <typename i_t, typename f_t>
i_t factorize_basis(const csc_matrix_t<i_t, f_t>& A,
                    const simplex_solver_settings_t<i_t, f_t>& settings,
                    const std::vector<i_t>& basis_list,
                    csc_matrix_t<i_t, f_t>& L,
                    csc_matrix_t<i_t, f_t>& U,
                    std::vector<i_t>& p,
                    std::vector<i_t>& pinv,
                    std::vector<i_t>& q,
                    std::vector<i_t>& deficient,
                    std::vector<i_t>& slacks_need);

// Repair the basis by bringing in slacks
template <typename i_t, typename f_t>
i_t basis_repair(const csc_matrix_t<i_t, f_t>& A,
                 const simplex_solver_settings_t<i_t, f_t>& settings,
                 const std::vector<i_t>& deficient,
                 const std::vector<i_t>& slacks_needed,
                 std::vector<i_t>& basis_list,
                 std::vector<i_t>& nonbasic_list,
                 std::vector<variable_status_t>& vstatus);

// Form the basis matrix B = A(:, basic_list)
template <typename i_t, typename f_t>
i_t form_b(const csc_matrix_t<i_t, f_t>& A,
           const std::vector<i_t>& basic_list,
           csc_matrix_t<i_t, f_t>& B);

// y = B*x = sum_{j in basis} A(:, j) * x(k)
template <typename i_t, typename f_t>
i_t b_multiply(const lp_problem_t<i_t, f_t>& lp,
               const std::vector<i_t>& basic_list,
               const std::vector<f_t>& x,
               std::vector<f_t>& y);

// y = B'*x. y_j = A(:, j)'*x for all j
template <typename i_t, typename f_t>
i_t b_transpose_multiply(const lp_problem_t<i_t, f_t>& lp,
                         const std::vector<i_t>& basic_list,
                         const std::vector<f_t>& x,
                         std::vector<f_t>& y);

// Solves B'*y = c, given L*U = B(p, :). This version supports a dense vector
template <typename i_t, typename f_t>
i_t b_transpose_solve(const csc_matrix_t<i_t, f_t>& L,
                      const csc_matrix_t<i_t, f_t>& U,
                      const std::vector<i_t>& p,
                      const std::vector<f_t>& rhs,
                      std::vector<f_t>& solution);

// Solves the system B*x = b, given L*U = B(p, :)
template <typename i_t, typename f_t>
i_t b_solve(const csc_matrix_t<i_t, f_t>& L,
            const csc_matrix_t<i_t, f_t>& U,
            const std::vector<i_t>& p,
            const std::vector<f_t>& rhs,
            std::vector<f_t>& solution);

}  // namespace cuopt::linear_programming::dual_simplex
