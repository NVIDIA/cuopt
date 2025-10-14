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

#include <dual_simplex/basis_solves.hpp>
#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/bound_flipping_ratio_test.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/phase1.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/random.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/tic_toc.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <ctime>

namespace cuopt::linear_programming::dual_simplex {

namespace phase2 {

// Computes vectors farkas_y, farkas_zl, farkas_zu that satisfy
//
// A'*farkas_y + farkas_zl - farkas_zu ~= 0
// farkas_zl, farkas_zu >= 0,
// b'*farkas_y + l'*farkas_zl - u'*farkas_zu = farkas_constant > 0
//
// This is a Farkas certificate for the infeasibility of the primal problem
//
// A*x = b, l <= x <= u
template <typename i_t, typename f_t>
void compute_farkas_certificate(const lp_problem_t<i_t, f_t>& lp,
                                const simplex_solver_settings_t<i_t, f_t>& settings,
                                const std::vector<variable_status_t>& vstatus,
                                const std::vector<f_t>& x,
                                const std::vector<f_t>& y,
                                const std::vector<f_t>& z,
                                const std::vector<f_t>& delta_y,
                                const std::vector<f_t>& delta_z,
                                i_t direction,
                                i_t leaving_index,
                                f_t obj_val,
                                std::vector<f_t>& farkas_y,
                                std::vector<f_t>& farkas_zl,
                                std::vector<f_t>& farkas_zu,
                                f_t& farkas_constant)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;

  std::vector<f_t> original_residual = z;
  matrix_transpose_vector_multiply(lp.A, 1.0, y, 1.0, original_residual);
  for (i_t j = 0; j < n; ++j) {
    original_residual[j] -= lp.objective[j];
  }
  const f_t original_residual_norm = vector_norm2<i_t, f_t>(original_residual);
  settings.log.printf("|| A'*y + z - c || = %e\n", original_residual_norm);

  std::vector<f_t> zl(n);
  std::vector<f_t> zu(n);
  for (i_t j = 0; j < n; ++j) {
    zl[j] = std::max(0.0, z[j]);
    zu[j] = -std::min(0.0, z[j]);
  }

  original_residual = zl;
  matrix_transpose_vector_multiply(lp.A, 1.0, y, 1.0, original_residual);
  for (i_t j = 0; j < n; ++j) {
    original_residual[j] -= (zu[j] + lp.objective[j]);
  }
  const f_t original_residual_2 = vector_norm2<i_t, f_t>(original_residual);
  settings.log.printf("|| A'*y + zl - zu - c || = %e\n", original_residual_2);

  std::vector<f_t> search_dir_residual = delta_z;
  matrix_transpose_vector_multiply(lp.A, 1.0, delta_y, 1.0, search_dir_residual);
  settings.log.printf("|| A'*delta_y + delta_z || = %e\n",
                      vector_norm2<i_t, f_t>(search_dir_residual));

  std::vector<f_t> y_bar(m);
  for (i_t i = 0; i < m; ++i) {
    y_bar[i] = y[i] + delta_y[i];
  }
  original_residual = z;
  matrix_transpose_vector_multiply(lp.A, 1.0, y_bar, 1.0, original_residual);
  for (i_t j = 0; j < n; ++j) {
    original_residual[j] += (delta_z[j] - lp.objective[j]);
  }
  const f_t original_residual_3 = vector_norm2<i_t, f_t>(original_residual);
  settings.log.printf("|| A'*(y + delta_y) + (z + delta_z) - c || = %e\n", original_residual_3);

  farkas_y.resize(m);
  farkas_zl.resize(n);
  farkas_zu.resize(n);

  f_t gamma = 0.0;
  for (i_t j = 0; j < n; ++j) {
    const f_t cj    = lp.objective[j];
    const f_t lower = lp.lower[j];
    const f_t upper = lp.upper[j];
    if (lower > -inf) { gamma -= lower * std::min(0.0, cj); }
    if (upper < inf) { gamma -= upper * std::max(0.0, cj); }
  }
  printf("gamma = %e\n", gamma);

  const f_t threshold          = 1.0;
  const f_t positive_threshold = std::max(-gamma, 0.0) + threshold;
  printf("positive_threshold = %e\n", positive_threshold);

  // We need to increase the dual objective to positive threshold
  f_t alpha        = threshold;
  const f_t infeas = (direction == 1) ? (lp.lower[leaving_index] - x[leaving_index])
                                      : (x[leaving_index] - lp.upper[leaving_index]);
  // We need the new objective to be at least positive_threshold
  // positive_threshold = obj_val+ alpha * infeas
  // infeas > 0, alpha > 0, positive_threshold > 0
  printf("direction = %d\n", direction);
  printf(
    "lower %e x %e upper %d\n", lp.lower[leaving_index], x[leaving_index], lp.upper[leaving_index]);
  printf("infeas = %e\n", infeas);
  printf("obj_val = %e\n", obj_val);
  alpha = std::max(threshold, (positive_threshold - obj_val) / infeas);
  printf("alpha = %e\n", alpha);

  std::vector<f_t> y_prime(m);
  std::vector<f_t> zl_prime(n);
  std::vector<f_t> zu_prime(n);

  // farkas_y = y + alpha * delta_y
  for (i_t i = 0; i < m; ++i) {
    farkas_y[i] = y[i] + alpha * delta_y[i];
    y_prime[i]  = y[i] + alpha * delta_y[i];
  }
  // farkas_zl = z + alpha * delta_z  - c-
  for (i_t j = 0; j < n; ++j) {
    const f_t cj        = lp.objective[j];
    const f_t z_j       = z[j];
    const f_t delta_z_j = delta_z[j];
    farkas_zl[j] = std::max(0.0, z_j) + alpha * std::max(0.0, delta_z_j) + -std::min(0.0, cj);
    zl_prime[j]  = zl[j] + alpha * std::max(0.0, delta_z_j);
  }

  // farkas_zu = z + alpha * delta_z + c+
  for (i_t j = 0; j < n; ++j) {
    const f_t cj        = lp.objective[j];
    const f_t z_j       = z[j];
    const f_t delta_z_j = delta_z[j];
    farkas_zu[j] = -std::min(0.0, z_j) - alpha * std::min(0.0, delta_z_j) + std::max(0.0, cj);
    zu_prime[j]  = zu[j] + alpha * (-std::min(0.0, delta_z_j));
  }

  // farkas_constant = b'*farkas_y + l'*farkas_zl - u'*farkas_zu
  farkas_constant   = 0.0;
  f_t test_constant = 0.0;
  f_t test_3        = 0.0;
  for (i_t i = 0; i < m; ++i) {
    farkas_constant += lp.rhs[i] * farkas_y[i];
    test_constant += lp.rhs[i] * y_prime[i];
    test_3 += lp.rhs[i] * delta_y[i];
  }
  printf("b'*delta_y = %e\n", test_3);
  printf("|| b || %e\n", vector_norm_inf<i_t, f_t>(lp.rhs));
  printf("|| delta y || %e\n", vector_norm_inf<i_t, f_t>(delta_y));
  for (i_t j = 0; j < n; ++j) {
    const f_t lower = lp.lower[j];
    const f_t upper = lp.upper[j];
    if (lower > -inf) {
      farkas_constant += lower * farkas_zl[j];
      test_constant += lower * zl_prime[j];
      const f_t delta_z_l_j = std::max(delta_z[j], 0.0);
      test_3 += lower * delta_z_l_j;
    }
    if (upper < inf) {
      farkas_constant -= upper * farkas_zu[j];
      test_constant -= upper * zu_prime[j];
      const f_t delta_z_u_j = -std::min(delta_z[j], 0.0);
      test_3 -= upper * delta_z_u_j;
    }
  }

  // Verify that the Farkas certificate is valid
  std::vector<f_t> residual = farkas_zl;
  matrix_transpose_vector_multiply(lp.A, 1.0, farkas_y, 1.0, residual);
  for (i_t j = 0; j < n; ++j) {
    residual[j] -= farkas_zu[j];
  }
  const f_t residual_norm = vector_norm2<i_t, f_t>(residual);

  f_t zl_min = 0.0;
  for (i_t j = 0; j < n; ++j) {
    zl_min = std::min(zl_min, farkas_zl[j]);
  }
  settings.log.printf("farkas_zl_min = %e\n", zl_min);
  f_t zu_min = 0.0;
  for (i_t j = 0; j < n; ++j) {
    zu_min = std::min(zu_min, farkas_zu[j]);
  }
  settings.log.printf("farkas_zu_min = %e\n", zu_min);

  settings.log.printf("|| A'*farkas_y + farkas_zl - farkas_zu || = %e\n", residual_norm);
  settings.log.printf("b'*farkas_y + l'*farkas_zl - u'*farkas_zu = %e\n", farkas_constant);

  if (residual_norm < 1e-6 && farkas_constant > 0.0 && zl_min >= 0.0 && zu_min >= 0.0) {
    settings.log.printf("Farkas certificate of infeasibility constructed\n");
  }
}

template <typename i_t, typename f_t>
void initial_perturbation(const lp_problem_t<i_t, f_t>& lp,
                          const simplex_solver_settings_t<i_t, f_t>& settings,
                          const std::vector<variable_status_t>& vstatus,
                          std::vector<f_t>& objective)
{
  const i_t m           = lp.num_rows;
  const i_t n           = lp.num_cols;
  f_t max_abs_obj_coeff = 0.0;
  for (i_t j = 0; j < n; ++j) {
    max_abs_obj_coeff = std::max(max_abs_obj_coeff, std::abs(lp.objective[j]));
  }

  const f_t dual_tol = settings.dual_tol;

  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  objective.resize(n);
  f_t sum_perturb = 0.0;
  i_t num_perturb = 0;

  random_t<i_t, f_t> random(settings.seed);
  for (i_t j = 0; j < n; ++j) {
    f_t obj = objective[j] = lp.objective[j];

    const f_t lower = lp.lower[j];
    const f_t upper = lp.upper[j];
    if (vstatus[j] == variable_status_t::NONBASIC_FIXED ||
        vstatus[j] == variable_status_t::NONBASIC_FREE || lower == upper ||
        lower == -inf && upper == inf) {
      continue;
    }

    const f_t rand_val = random.random();
    const f_t perturb =
      (1e-5 * std::abs(obj) + 1e-7 * max_abs_obj_coeff + 10 * dual_tol) * (1.0 + rand_val);

    if (vstatus[j] == variable_status_t::NONBASIC_LOWER || lower > -inf && upper < inf && obj > 0) {
      objective[j] = obj + perturb;
      sum_perturb += perturb;
      num_perturb++;
    } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER ||
               lower > -inf && upper < inf && obj < 0) {
      objective[j] = obj - perturb;
      sum_perturb += perturb;
      num_perturb++;
    }
  }

  settings.log.printf("Applied initial perturbation of %e to %d/%d objective coefficients\n",
                      sum_perturb,
                      num_perturb,
                      n);
}

template <typename i_t, typename f_t>
void compute_reduced_cost_update(const lp_problem_t<i_t, f_t>& lp,
                                 const std::vector<i_t>& basic_list,
                                 const std::vector<i_t>& nonbasic_list,
                                 const std::vector<f_t>& delta_y,
                                 i_t leaving_index,
                                 i_t direction,
                                 std::vector<i_t>& delta_z_mark,
                                 std::vector<i_t>& delta_z_indices,
                                 std::vector<f_t>& delta_z)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;

  // delta_zB = sigma*ei
  for (i_t k = 0; k < m; k++) {
    const i_t j = basic_list[k];
    delta_z[j]  = 0;
  }
  delta_z[leaving_index] = direction;
  // delta_zN = -N'*delta_y
  for (i_t k = 0; k < n - m; k++) {
    const i_t j = nonbasic_list[k];
    // z_j <- -A(:, j)'*delta_y
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    f_t dot             = 0.0;
    for (i_t p = col_start; p < col_end; ++p) {
      dot += lp.A.x[p] * delta_y[lp.A.i[p]];
    }
    delta_z[j] = -dot;
    if (dot != 0.0) {
      delta_z_indices.push_back(j);  // Note delta_z_indices has n elements reserved
      delta_z_mark[j] = 1;
    }
  }
}

template <typename i_t, typename f_t>
void compute_delta_z(const csc_matrix_t<i_t, f_t>& A_transpose,
                     const sparse_vector_t<i_t, f_t>& delta_y,
                     i_t leaving_index,
                     i_t direction,
                     std::vector<i_t>& nonbasic_mark,
                     std::vector<i_t>& delta_z_mark,
                     std::vector<i_t>& delta_z_indices,
                     std::vector<f_t>& delta_z)
{
  // delta_zN = - N'*delta_y
  const i_t nz_delta_y = delta_y.i.size();
  for (i_t k = 0; k < nz_delta_y; k++) {
    const i_t i         = delta_y.i[k];
    const f_t delta_y_i = delta_y.x[k];
    if (std::abs(delta_y_i) < 1e-12) { continue; }
    const i_t row_start = A_transpose.col_start[i];
    const i_t row_end   = A_transpose.col_start[i + 1];
    for (i_t p = row_start; p < row_end; ++p) {
      const i_t j = A_transpose.i[p];
      if (nonbasic_mark[j] >= 0) {
        delta_z[j] -= delta_y_i * A_transpose.x[p];
        if (!delta_z_mark[j]) {
          delta_z_mark[j] = 1;
          delta_z_indices.push_back(j);  // Note delta_z_indices has n elements reserved
        }
      }
    }
  }

  // delta_zB = sigma*ei
  delta_z[leaving_index] = direction;

#ifdef CHECK_CHANGE_IN_REDUCED_COST
  delta_y_sparse.to_dense(delta_y);
  std::vector<f_t> delta_z_check(n);
  std::vector<i_t> delta_z_mark_check(n, 0);
  std::vector<i_t> delta_z_indices_check;
  phase2::compute_reduced_cost_update(lp,
                                      basic_list,
                                      nonbasic_list,
                                      delta_y,
                                      leaving_index,
                                      direction,
                                      delta_z_mark_check,
                                      delta_z_indices_check,
                                      delta_z_check);
  f_t error_check = 0.0;
  for (i_t k = 0; k < n; ++k) {
    const f_t diff = std::abs(delta_z[k] - delta_z_check[k]);
    if (diff > 1e-6) {
      printf("delta_z error %d transpose %e no transpose %e diff %e\n",
             k,
             delta_z[k],
             delta_z_check[k],
             diff);
    }
    error_check = std::max(error_check, diff);
  }
  if (error_check > 1e-6) { printf("delta_z error %e\n", error_check); }
#endif
}

template <typename i_t, typename f_t>
void compute_reduced_costs(const std::vector<f_t>& objective,
                           const csc_matrix_t<i_t, f_t>& A,
                           const std::vector<f_t>& y,
                           const std::vector<i_t>& basic_list,
                           const std::vector<i_t>& nonbasic_list,
                           std::vector<f_t>& z)
{
  const i_t m = A.m;
  const i_t n = A.n;
  // zN = cN - N'*y
  for (i_t k = 0; k < n - m; k++) {
    const i_t j = nonbasic_list[k];
    // z_j <- c_j
    z[j] = objective[j];

    // z_j <- z_j - A(:, j)'*y
    const i_t col_start = A.col_start[j];
    const i_t col_end   = A.col_start[j + 1];
    f_t dot             = 0.0;
    for (i_t p = col_start; p < col_end; ++p) {
      dot += A.x[p] * y[A.i[p]];
    }
    z[j] -= dot;
  }
  // zB = 0
  for (i_t k = 0; k < m; ++k) {
    z[basic_list[k]] = 0.0;
  }
}

template <typename i_t, typename f_t>
void compute_primal_variables(const basis_update_mpf_t<i_t, f_t>& ft,
                              const std::vector<f_t>& lp_rhs,
                              const csc_matrix_t<i_t, f_t>& A,
                              const std::vector<i_t>& basic_list,
                              const std::vector<i_t>& nonbasic_list,
                              f_t tight_tol,
                              std::vector<f_t>& x)
{
  const i_t m          = A.m;
  const i_t n          = A.n;
  std::vector<f_t> rhs = lp_rhs;
  // rhs = b - sum_{j : x_j = l_j} A(:, j) * l(j)
  //         - sum_{j : x_j = u_j} A(:, j) * u(j)
  for (i_t k = 0; k < n - m; ++k) {
    const i_t j         = nonbasic_list[k];
    const i_t col_start = A.col_start[j];
    const i_t col_end   = A.col_start[j + 1];
    const f_t xj        = x[j];
    if (std::abs(xj) < tight_tol * 10) continue;
    for (i_t p = col_start; p < col_end; ++p) {
      rhs[A.i[p]] -= xj * A.x[p];
    }
  }

  std::vector<f_t> xB(m);
  ft.b_solve(rhs, xB);

  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    x[j]        = xB[k];
  }
}

template <typename i_t, typename f_t>
void clear_delta_z(i_t entering_index,
                   i_t leaving_index,
                   std::vector<i_t>& delta_z_mark,
                   std::vector<i_t>& delta_z_indices,
                   std::vector<f_t>& delta_z)
{
  for (i_t k = 0; k < delta_z_indices.size(); k++) {
    const i_t j     = delta_z_indices[k];
    delta_z[j]      = 0.0;
    delta_z_mark[j] = 0;
  }
  if (entering_index != -1) { delta_z[entering_index] = 0.0; }
  delta_z[leaving_index] = 0.0;
  delta_z_indices.clear();
}

template <typename i_t, typename f_t>
void clear_delta_x(const std::vector<i_t>& basic_list,
                   i_t entering_index,
                   sparse_vector_t<i_t, f_t>& scaled_delta_xB_sparse,
                   std::vector<f_t>& delta_x)
{
  const i_t scaled_delta_xB_nz = scaled_delta_xB_sparse.i.size();
  for (i_t k = 0; k < scaled_delta_xB_nz; ++k) {
    const i_t j = basic_list[scaled_delta_xB_sparse.i[k]];
    delta_x[j]  = 0.0;
  }
  // Leaving index already included above
  delta_x[entering_index] = 0.0;
  scaled_delta_xB_sparse.i.clear();
  scaled_delta_xB_sparse.x.clear();
}

template <typename i_t, typename f_t>
void compute_dual_residual(const csc_matrix_t<i_t, f_t>& A,
                           const std::vector<f_t>& objective,
                           const std::vector<f_t>& y,
                           const std::vector<f_t>& z,
                           std::vector<f_t>& dual_residual)
{
  dual_residual = z;
  const i_t n   = A.n;
  // r = A'*y + z  - c
  for (i_t j = 0; j < n; ++j) {
    dual_residual[j] -= objective[j];
  }
  matrix_transpose_vector_multiply(A, 1.0, y, 1.0, dual_residual);
}

template <typename i_t, typename f_t>
f_t l2_dual_residual(const lp_problem_t<i_t, f_t>& lp, const lp_solution_t<i_t, f_t>& solution)
{
  std::vector<f_t> dual_residual;
  compute_dual_residual(lp.A, lp.objective, solution.y, solution.z, dual_residual);
  return vector_norm2<i_t, f_t>(dual_residual);
}

template <typename i_t, typename f_t>
f_t l2_primal_residual(const lp_problem_t<i_t, f_t>& lp, const lp_solution_t<i_t, f_t>& solution)
{
  std::vector<f_t> primal_residual = lp.rhs;
  matrix_vector_multiply(lp.A, 1.0, solution.x, -1.0, primal_residual);
  return vector_norm2<i_t, f_t>(primal_residual);
}

template <typename i_t, typename f_t>
void vstatus_changes(const std::vector<variable_status_t>& vstatus,
                     const std::vector<variable_status_t>& vstatus_old,
                     const std::vector<f_t>& z,
                     const std::vector<f_t>& z_old,
                     i_t& num_vstatus_changes,
                     i_t& num_z_changes)
{
  num_vstatus_changes = 0;
  num_z_changes       = 0;
  const i_t n         = vstatus.size();
  for (i_t j = 0; j < n; ++j) {
    if (vstatus[j] != vstatus_old[j]) { num_vstatus_changes++; }
    if (std::abs(z[j] - z_old[j]) > 1e-6) { num_z_changes++; }
  }
}

template <typename f_t>
void compute_bounded_info(const std::vector<f_t>& lower,
                          const std::vector<f_t>& upper,
                          std::vector<uint8_t>& bounded_variables)
{
  const size_t n = lower.size();
  for (size_t j = 0; j < n; j++) {
    const bool bounded   = (lower[j] > -inf) && (upper[j] < inf) && (lower[j] != upper[j]);
    bounded_variables[j] = static_cast<uint8_t>(bounded);
  }
}

template <typename i_t, typename f_t>
void compute_dual_solution_from_basis(const lp_problem_t<i_t, f_t>& lp,
                                      basis_update_mpf_t<i_t, f_t>& ft,
                                      const std::vector<i_t>& basic_list,
                                      const std::vector<i_t>& nonbasic_list,
                                      std::vector<f_t>& y,
                                      std::vector<f_t>& z)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;

  y.resize(m);
  std::vector<f_t> cB(m);
  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    cB[k]       = lp.objective[j];
  }
  ft.b_transpose_solve(cB, y);

  // We want A'y + z = c
  // A = [ B N ]
  // B' y = c_B, z_B = 0
  // N' y + z_N = c_N
  z.resize(n);
  // zN = cN - N'*y
  for (i_t k = 0; k < n - m; k++) {
    const i_t j = nonbasic_list[k];
    // z_j <- c_j
    z[j] = lp.objective[j];

    // z_j <- z_j - A(:, j)'*y
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    f_t dot             = 0.0;
    for (i_t p = col_start; p < col_end; ++p) {
      dot += lp.A.x[p] * y[lp.A.i[p]];
    }
    z[j] -= dot;
  }
  // zB = 0
  for (i_t k = 0; k < m; ++k) {
    z[basic_list[k]] = 0.0;
  }
}

template <typename i_t, typename f_t>
i_t compute_primal_solution_from_basis(const lp_problem_t<i_t, f_t>& lp,
                                       basis_update_mpf_t<i_t, f_t>& ft,
                                       const std::vector<i_t>& basic_list,
                                       const std::vector<i_t>& nonbasic_list,
                                       const std::vector<variable_status_t>& vstatus,
                                       std::vector<f_t>& x)
{
  const i_t m          = lp.num_rows;
  const i_t n          = lp.num_cols;
  std::vector<f_t> rhs = lp.rhs;

  for (i_t k = 0; k < n - m; ++k) {
    const i_t j = nonbasic_list[k];
    if (vstatus[j] == variable_status_t::NONBASIC_LOWER ||
        vstatus[j] == variable_status_t::NONBASIC_FIXED) {
      x[j] = lp.lower[j];
    } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER) {
      x[j] = lp.upper[j];
    } else if (vstatus[j] == variable_status_t::NONBASIC_FREE) {
      x[j] = 0.0;
    }
  }

  // rhs = b - sum_{j : x_j = l_j} A(:, j) l(j) - sum_{j : x_j = u_j} A(:, j) *
  // u(j)
  for (i_t k = 0; k < n - m; ++k) {
    const i_t j         = nonbasic_list[k];
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    const f_t xj        = x[j];
    for (i_t p = col_start; p < col_end; ++p) {
      rhs[lp.A.i[p]] -= xj * lp.A.x[p];
    }
  }

  std::vector<f_t> xB(m);
  ft.b_solve(rhs, xB);

  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    x[j]        = xB[k];
  }
  return 0;
}

template <typename i_t, typename f_t>
f_t compute_initial_primal_infeasibilities(const lp_problem_t<i_t, f_t>& lp,
                                           const simplex_solver_settings_t<i_t, f_t>& settings,
                                           const std::vector<i_t>& basic_list,
                                           const std::vector<f_t>& x,
                                           std::vector<f_t>& squared_infeasibilities,
                                           std::vector<i_t>& infeasibility_indices)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;
  squared_infeasibilities.resize(n, 0.0);
  infeasibility_indices.reserve(n);
  infeasibility_indices.clear();
  f_t primal_inf = 0.0;
  for (i_t k = 0; k < m; ++k) {
    const i_t j            = basic_list[k];
    const f_t lower_infeas = lp.lower[j] - x[j];
    const f_t upper_infeas = x[j] - lp.upper[j];
    const f_t infeas       = std::max(lower_infeas, upper_infeas);
    if (infeas > settings.primal_tol) {
      const f_t square_infeas    = infeas * infeas;
      squared_infeasibilities[j] = square_infeas;
      infeasibility_indices.push_back(j);
      primal_inf += square_infeas;
    }
  }
  return primal_inf;
}

template <typename i_t, typename f_t>
void update_single_primal_infeasibility(const std::vector<f_t>& lower,
                                        const std::vector<f_t>& upper,
                                        const std::vector<f_t>& x,
                                        f_t primal_tol,
                                        std::vector<f_t>& squared_infeasibilities,
                                        std::vector<i_t>& infeasibility_indices,
                                        i_t j,
                                        f_t& primal_inf)
{
  const f_t old_val = squared_infeasibilities[j];
  // x_j < l_j - epsilon => -x_j + l_j > epsilon
  const f_t lower_infeas = lower[j] - x[j];
  // x_j > u_j + epsilon => x_j - u_j > epsilon
  const f_t upper_infeas = x[j] - upper[j];
  const f_t infeas       = std::max(lower_infeas, upper_infeas);
  const f_t new_val      = infeas * infeas;
  if (infeas > primal_tol) {
    primal_inf = std::max(0.0, primal_inf + (new_val - old_val));
    // We are infeasible w.r.t the tolerance
    if (old_val == 0.0) {
      // This is a new infeasibility
      // We need to add it to the list
      infeasibility_indices.push_back(j);
    } else {
      // Already infeasible
    }
    squared_infeasibilities[j] = new_val;
  } else {
    // We are feasible w.r.t the tolerance
    if (old_val != 0.0) {
      // We were previously infeasible,
      primal_inf                 = std::max(0.0, primal_inf - old_val);
      squared_infeasibilities[j] = 0.0;
    } else {
      // Still feasible
    }
  }
}

template <typename i_t, typename f_t>
void update_primal_infeasibilities(const lp_problem_t<i_t, f_t>& lp,
                                   const simplex_solver_settings_t<i_t, f_t>& settings,
                                   const std::vector<i_t>& basic_list,
                                   const std::vector<f_t>& x,
                                   i_t entering_index,
                                   i_t leaving_index,
                                   std::vector<i_t>& basic_change_list,
                                   std::vector<f_t>& squared_infeasibilities,
                                   std::vector<i_t>& infeasibility_indices,
                                   f_t& primal_inf)
{
  const f_t primal_tol = settings.primal_tol;
  const i_t nz         = basic_change_list.size();
  for (i_t k = 0; k < nz; ++k) {
    const i_t j = basic_list[basic_change_list[k]];
    // The change list will contain the leaving variable,
    // But not the entering variable.

    if (j == leaving_index) {
      // Force the leaving variable to be feasible
      const f_t old_val          = squared_infeasibilities[j];
      squared_infeasibilities[j] = 0.0;
      primal_inf                 = std::max(0.0, primal_inf - old_val);
      continue;
    }
    update_single_primal_infeasibility(lp.lower,
                                       lp.upper,
                                       x,
                                       primal_tol,
                                       squared_infeasibilities,
                                       infeasibility_indices,
                                       j,
                                       primal_inf);
  }
}

template <typename i_t, typename f_t>
void clean_up_infeasibilities(std::vector<f_t>& squared_infeasibilities,
                              std::vector<i_t>& infeasibility_indices)
{
  bool needs_clean_up = false;
  for (i_t k = 0; k < infeasibility_indices.size(); ++k) {
    const i_t j              = infeasibility_indices[k];
    const f_t squared_infeas = squared_infeasibilities[j];
    if (squared_infeas == 0.0) { needs_clean_up = true; }
  }

  if (needs_clean_up) {
    for (i_t k = 0; k < infeasibility_indices.size(); ++k) {
      const i_t j              = infeasibility_indices[k];
      const f_t squared_infeas = squared_infeasibilities[j];
      if (squared_infeas == 0.0) {
        // Set to the last element
        const i_t new_j          = infeasibility_indices.back();
        infeasibility_indices[k] = new_j;
        infeasibility_indices.pop_back();
        if (squared_infeasibilities[new_j] == 0.0) { k--; }
      }
    }
  }
}

template <typename i_t, typename f_t>
i_t steepest_edge_pricing_with_infeasibilities(const lp_problem_t<i_t, f_t>& lp,
                                               const simplex_solver_settings_t<i_t, f_t>& settings,
                                               const std::vector<f_t>& x,
                                               const std::vector<f_t>& dy_steepest_edge,
                                               const std::vector<i_t>& basic_mark,
                                               std::vector<f_t>& squared_infeasibilities,
                                               std::vector<i_t>& infeasibility_indices,
                                               i_t& direction,
                                               i_t& basic_leaving,
                                               f_t& max_val)
{
  max_val           = 0.0;
  i_t leaving_index = -1;
  const i_t nz      = infeasibility_indices.size();
  for (i_t k = 0; k < nz; ++k) {
    const i_t j              = infeasibility_indices[k];
    const f_t squared_infeas = squared_infeasibilities[j];
    const f_t val            = squared_infeas / dy_steepest_edge[j];
    if (val > max_val || val == max_val && j > leaving_index) {
      max_val                = val;
      leaving_index          = j;
      const f_t lower_infeas = lp.lower[j] - x[j];
      const f_t upper_infeas = x[j] - lp.upper[j];
      direction              = lower_infeas >= upper_infeas ? 1 : -1;
    }
  }

  basic_leaving = leaving_index >= 0 ? basic_mark[leaving_index] : -1;
  return leaving_index;
}

template <typename i_t, typename f_t>
i_t steepest_edge_pricing(const lp_problem_t<i_t, f_t>& lp,
                          const simplex_solver_settings_t<i_t, f_t>& settings,
                          const std::vector<f_t>& x,
                          const std::vector<f_t>& dy_steepest_edge,
                          const std::vector<i_t>& basic_list,
                          i_t& direction,
                          i_t& basic_leaving,
                          f_t& primal_inf,
                          f_t& max_val)
{
  const i_t m          = lp.num_rows;
  max_val              = 0.0;
  i_t leaving_index    = -1;
  const f_t primal_tol = settings.primal_tol;
  primal_inf           = 0;
  i_t num_candidates   = 0;
  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    if (x[j] < lp.lower[j] - primal_tol) {
      num_candidates++;
      // x_j < l_j => -x_j > -l_j => -x_j + l_j > 0
      const f_t infeas = -x[j] + lp.lower[j];
      primal_inf += infeas;
      const f_t val = (infeas * infeas) / dy_steepest_edge[j];
#ifdef DEBUG_PRICE
      settings.log.printf("price %d x %e lo %e infeas %e val %e se %e\n",
                          j,
                          x[j],
                          lp.lower[j],
                          infeas,
                          val,
                          dy_steepest_edge[j]);
#endif
      assert(val > 0.0);
      if (val > max_val) {
        max_val       = val;
        leaving_index = j;
        basic_leaving = k;
        direction     = 1;
      }
    }
    if (x[j] > lp.upper[j] + primal_tol) {
      num_candidates++;
      // x_j > u_j => x_j - u_j > 0
      const f_t infeas = x[j] - lp.upper[j];
      primal_inf += infeas;
      const f_t val = (infeas * infeas) / dy_steepest_edge[j];
#ifdef DEBUG_PRICE
      settings.log.printf("price %d x %e up %e infeas %e val %e se %e\n",
                          j,
                          x[j],
                          lp.upper[j],
                          infeas,
                          val,
                          dy_steepest_edge[j]);
#endif
      assert(val > 0.0);
      if (val > max_val) {
        max_val       = val;
        leaving_index = j;
        basic_leaving = k;
        direction     = -1;
      }
    }
  }
  return leaving_index;
}

// Maximum infeasibility
template <typename i_t, typename f_t>
i_t phase2_pricing(const lp_problem_t<i_t, f_t>& lp,
                   const simplex_solver_settings_t<i_t, f_t>& settings,
                   const std::vector<f_t>& x,
                   const std::vector<i_t>& basic_list,
                   i_t& direction,
                   i_t& basic_leaving,
                   f_t& primal_inf)
{
  const i_t m          = lp.num_rows;
  f_t max_val          = 0.0;
  i_t leaving_index    = -1;
  const f_t primal_tol = settings.primal_tol / 10;
  primal_inf           = 0;
  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    if (x[j] < lp.lower[j] - primal_tol) {
      // x_j < l_j => -x_j > -l_j => -x_j + l_j > 0
      const f_t val = -x[j] + lp.lower[j];
      assert(val > 0.0);
      primal_inf += val;
      if (val > max_val) {
        max_val       = val;
        leaving_index = j;
        basic_leaving = k;
        direction     = 1;
      }
    }
    if (x[j] > lp.upper[j] + primal_tol) {
      // x_j > u_j => x_j - u_j > 0
      const f_t val = x[j] - lp.upper[j];
      assert(val > 0.0);
      primal_inf += val;
      if (val > max_val) {
        max_val       = val;
        leaving_index = j;
        basic_leaving = k;
        direction     = -1;
      }
    }
  }
  return leaving_index;
}

template <typename i_t, typename f_t>
f_t first_stage_harris(const lp_problem_t<i_t, f_t>& lp,
                       const std::vector<variable_status_t>& vstatus,
                       const std::vector<i_t>& nonbasic_list,
                       std::vector<f_t>& z,
                       std::vector<f_t>& delta_z)
{
  const i_t n             = lp.num_cols;
  const i_t m             = lp.num_rows;
  constexpr f_t pivot_tol = 1e-7;
  constexpr f_t dual_tol  = 1e-7;
  f_t min_val             = inf;
  f_t step_length         = -inf;

  for (i_t k = 0; k < n - m; ++k) {
    const i_t j = nonbasic_list[k];
    if (vstatus[j] == variable_status_t::NONBASIC_LOWER && delta_z[j] < -pivot_tol) {
      const f_t ratio = (-dual_tol - z[j]) / delta_z[j];
      if (ratio < min_val) {
        min_val     = ratio;
        step_length = ratio;
      }
    }
    if (vstatus[j] == variable_status_t::NONBASIC_UPPER && delta_z[j] > pivot_tol) {
      const f_t ratio = (dual_tol - z[j]) / delta_z[j];
      if (ratio < min_val) {
        min_val     = ratio;
        step_length = ratio;
      }
    }
  }
  return step_length;
}

template <typename i_t, typename f_t>
i_t second_stage_harris(const lp_problem_t<i_t, f_t>& lp,
                        const std::vector<variable_status_t>& vstatus,
                        const std::vector<i_t>& nonbasic_list,
                        const std::vector<f_t>& z,
                        const std::vector<f_t>& delta_z,
                        f_t max_step_length,
                        f_t& step_length,
                        i_t& nonbasic_entering)
{
  const i_t n        = lp.num_cols;
  const i_t m        = lp.num_rows;
  i_t entering_index = -1;
  f_t max_val        = 0;
  for (i_t k = 0; k < n - m; ++k) {
    const i_t j = nonbasic_list[k];
    if (vstatus[j] == variable_status_t::NONBASIC_LOWER && delta_z[j] < 0) {
      // z_j + alpha delta_z_j >= 0, delta_z_j < 0
      // alpha delta_z_j >= -z_j
      // alpha <= -z_j/delta_z_j
      const f_t ratio = -z[j] / delta_z[j];
      if (ratio < max_step_length && std::abs(delta_z[j]) > max_val) {
        step_length       = ratio;
        max_val           = std::abs(delta_z[j]);
        entering_index    = j;
        nonbasic_entering = k;
      }
    } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER && delta_z[j] > 0) {
      // z_j + alpha delta_z_j <= 0, delta_z_j > 0
      // alpha <= -z_j/delta_z_j
      const f_t ratio = -z[j] / delta_z[j];
      if (ratio < max_step_length && std::abs(delta_z[j]) > max_val) {
        step_length       = ratio;
        max_val           = std::abs(delta_z[j]);
        entering_index    = j;
        nonbasic_entering = k;
      }
    }
  }
  return entering_index;
}

template <typename i_t, typename f_t>
i_t phase2_ratio_test(const lp_problem_t<i_t, f_t>& lp,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      const std::vector<variable_status_t>& vstatus,
                      const std::vector<i_t>& nonbasic_list,
                      std::vector<f_t>& z,
                      std::vector<f_t>& delta_z,
                      f_t& step_length,
                      i_t& nonbasic_entering)
{
  i_t entering_index  = -1;
  const i_t n         = lp.num_cols;
  const i_t m         = lp.num_rows;
  const f_t pivot_tol = settings.pivot_tol;
  const f_t dual_tol  = settings.dual_tol / 10;
  const f_t zero_tol  = settings.zero_tol;
  f_t min_val         = inf;

  for (i_t k = 0; k < n - m; ++k) {
    const i_t j = nonbasic_list[k];
    if (vstatus[j] == variable_status_t::NONBASIC_FIXED) { continue; }
    if (vstatus[j] == variable_status_t::NONBASIC_LOWER && delta_z[j] < -pivot_tol) {
      const f_t ratio = (-dual_tol - z[j]) / delta_z[j];
      if (ratio < min_val) {
        min_val           = ratio;
        entering_index    = j;
        step_length       = ratio;
        nonbasic_entering = k;
      } else if (ratio < min_val + zero_tol && std::abs(z[j]) > std::abs(z[entering_index])) {
        min_val           = ratio;
        entering_index    = j;
        step_length       = ratio;
        nonbasic_entering = k;
      }
    }
    if (vstatus[j] == variable_status_t::NONBASIC_UPPER && delta_z[j] > pivot_tol) {
      const f_t ratio = (dual_tol - z[j]) / delta_z[j];
      if (ratio < min_val) {
        min_val           = ratio;
        entering_index    = j;
        step_length       = ratio;
        nonbasic_entering = k;
      } else if (ratio < min_val + zero_tol && std::abs(z[j]) > std::abs(z[entering_index])) {
        min_val           = ratio;
        entering_index    = j;
        step_length       = ratio;
        nonbasic_entering = k;
      }
    }
  }
  return entering_index;
}

template <typename i_t, typename f_t>
i_t flip_bounds(const lp_problem_t<i_t, f_t>& lp,
                const simplex_solver_settings_t<i_t, f_t>& settings,
                const std::vector<uint8_t>& bounded_variables,
                const std::vector<f_t>& objective,
                const std::vector<f_t>& z,
                const std::vector<i_t>& delta_z_indices,
                const std::vector<i_t>& nonbasic_list,
                i_t entering_index,
                std::vector<variable_status_t>& vstatus,
                std::vector<f_t>& delta_x,
                std::vector<i_t>& mark,
                std::vector<f_t>& atilde,
                std::vector<i_t>& atilde_index)
{
  i_t num_flipped = 0;
  for (i_t j : delta_z_indices) {
    if (j == entering_index) { continue; }
    if (!bounded_variables[j]) { continue; }
    // x_j is now a nonbasic bounded variable that will not enter the basis this
    // iteration
    const f_t dual_tol =
      settings.dual_tol;  // lower to 1e-7 or less will cause 25fv47 and d2q06c to cycle
    if (vstatus[j] == variable_status_t::NONBASIC_LOWER && z[j] < -dual_tol) {
      const f_t delta = lp.upper[j] - lp.lower[j];
      scatter_dense(lp.A, j, -delta, atilde, mark, atilde_index);
      delta_x[j] += delta;
      vstatus[j] = variable_status_t::NONBASIC_UPPER;
#ifdef BOUND_FLIP_DEBUG
      settings.log.printf(
        "Flipping nonbasic %d from lo %e to up %e. z %e\n", j, lp.lower[j], lp.upper[j], z[j]);
#endif
      num_flipped++;
    } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER && z[j] > dual_tol) {
      const f_t delta = lp.lower[j] - lp.upper[j];
      scatter_dense(lp.A, j, -delta, atilde, mark, atilde_index);
      delta_x[j] += delta;
      vstatus[j] = variable_status_t::NONBASIC_LOWER;
#ifdef BOUND_FLIP_DEBUG
      settings.log.printf(
        "Flipping nonbasic %d from up %e to lo %e. z %e\n", j, lp.upper[j], lp.lower[j], z[j]);
#endif
      num_flipped++;
    }
  }
  return num_flipped;
}

template <typename i_t, typename f_t>
void initialize_steepest_edge_norms_from_slack_basis(const std::vector<i_t>& basic_list,
                                                     const std::vector<i_t>& nonbasic_list,
                                                     std::vector<f_t>& delta_y_steepest_edge)
{
  const i_t m = basic_list.size();
  const i_t n = delta_y_steepest_edge.size();
  for (i_t k = 0; k < m; ++k) {
    const i_t j              = basic_list[k];
    delta_y_steepest_edge[j] = 1.0;
  }
  const i_t n_minus_m = n - m;
  for (i_t k = 0; k < n_minus_m; ++k) {
    const i_t j              = nonbasic_list[k];
    delta_y_steepest_edge[j] = 1e-4;
  }
}

template <typename i_t, typename f_t>
i_t initialize_steepest_edge_norms(const lp_problem_t<i_t, f_t>& lp,
                                   const simplex_solver_settings_t<i_t, f_t>& settings,
                                   const f_t start_time,
                                   const std::vector<i_t>& basic_list,
                                   basis_update_mpf_t<i_t, f_t>& ft,
                                   std::vector<f_t>& delta_y_steepest_edge)
{
  const i_t m = basic_list.size();

  // We want to compute B^T delta_y_i = -e_i
  // If there is a column u of B^T such that B^T(:, u) = alpha * e_i than the
  // solve delta_y_i = -1/alpha * e_u
  // So we need to find columns of B^T (or rows of B) with only a single non-zero entry
  f_t start_singleton_rows = tic();
  std::vector<i_t> row_degree(m, 0);
  std::vector<i_t> mapping(m, -1);
  std::vector<f_t> coeff(m, 0.0);

  for (i_t k = 0; k < m; ++k) {
    const i_t j         = basic_list[k];
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      const i_t i = lp.A.i[p];
      row_degree[i]++;
      // column j of A is column k of B
      mapping[k] = i;
      coeff[k]   = lp.A.x[p];
    }
  }

#ifdef CHECK_SINGLETON_ROWS
  csc_matrix_t<i_t, f_t> B(m, m, 0);
  form_b(lp.A, basic_list, B);
  csc_matrix_t<i_t, f_t> B_transpose(m, m, 0);
  B.transpose(B_transpose);
#endif

  i_t num_singleton_rows = 0;
  for (i_t i = 0; i < m; ++i) {
    if (row_degree[i] == 1) {
      num_singleton_rows++;
#ifdef CHECK_SINGLETON_ROWS
      const i_t col_start = B_transpose.col_start[i];
      const i_t col_end   = B_transpose.col_start[i + 1];
      if (col_end - col_start != 1) {
        settings.log.printf("Singleton row %d has %d non-zero entries\n", i, col_end - col_start);
      }
#endif
    }
  }

  if (num_singleton_rows > 0) {
    settings.log.printf("Found %d singleton rows for steepest edge norms in %.2fs\n",
                        num_singleton_rows,
                        toc(start_singleton_rows));
  }

  f_t last_log = tic();
  for (i_t k = 0; k < m; ++k) {
    sparse_vector_t<i_t, f_t> sparse_ei(m, 1);
    sparse_ei.x[0] = -1.0;
    sparse_ei.i[0] = k;
    const i_t j    = basic_list[k];
    f_t init       = -1.0;
    if (row_degree[mapping[k]] == 1) {
      const i_t u     = mapping[k];
      const f_t alpha = coeff[k];
      // dy[u] = -1.0 / alpha;
      f_t my_init = 1.0 / (alpha * alpha);
      init        = my_init;
#ifdef CHECK_HYPERSPARSE
      std::vector<f_t> residual(m);
      b_transpose_multiply(lp, basic_list, dy, residual);
      float error = 0;
      for (i_t h = 0; h < m; ++h) {
        const f_t error_component = std::abs(residual[h] - ei[h]);
        error += error_component;
        if (error_component > 1e-12) {
          settings.log.printf("Singleton row %d component %d error %e residual %e ei %e\n",
                              k,
                              h,
                              error_component,
                              residual[h],
                              ei[h]);
        }
      }
      if (error > 1e-12) { settings.log.printf("Singleton row %d error %e\n", k, error); }
#endif

#ifdef CHECK_HYPERSPARSE
      dy[u] = 0.0;
      ft.b_transpose_solve(ei, dy);
      init = vector_norm2_squared<i_t, f_t>(dy);
      if (init != my_init) {
        settings.log.printf("Singleton row %d error %.16e init %.16e my_init %.16e\n",
                            k,
                            std::abs(init - my_init),
                            init,
                            my_init);
      }
#endif
    } else {
#if COMPARE_WITH_DENSE
      ft.b_transpose_solve(ei, dy);
      init = vector_norm2_squared<i_t, f_t>(dy);
#else
      sparse_vector_t<i_t, f_t> sparse_dy(m, 0);
      ft.b_transpose_solve(sparse_ei, sparse_dy);
      f_t my_init = 0.0;
      for (i_t p = 0; p < sparse_dy.x.size(); ++p) {
        my_init += sparse_dy.x[p] * sparse_dy.x[p];
      }
#endif
#if COMPARE_WITH_DENSE
      if (std::abs(init - my_init) > 1e-12) {
        settings.log.printf("Singleton row %d error %.16e init %.16e my_init %.16e\n",
                            k,
                            std::abs(init - my_init),
                            init,
                            my_init);
      }
#endif
      init = my_init;
    }
    // ei[k]          = 0.0;
    // init = vector_norm2_squared<i_t, f_t>(dy);
    assert(init > 0);
    delta_y_steepest_edge[j] = init;

    f_t now            = toc(start_time);
    f_t time_since_log = toc(last_log);
    if (time_since_log > 10) {
      last_log = tic();
      settings.log.printf("Initialized %d of %d steepest edge norms in %.2fs\n", k, m, now);
    }
    if (toc(start_time) > settings.time_limit) { return -1; }
    if (settings.concurrent_halt != nullptr && *settings.concurrent_halt == 1) { return -1; }
  }
  return 0;
}

template <typename i_t, typename f_t>
i_t update_steepest_edge_norms(const simplex_solver_settings_t<i_t, f_t>& settings,
                               const std::vector<i_t>& basic_list,
                               const basis_update_mpf_t<i_t, f_t>& ft,
                               i_t direction,
                               const sparse_vector_t<i_t, f_t>& delta_y_sparse,
                               f_t dy_norm_squared,
                               const sparse_vector_t<i_t, f_t>& scaled_delta_xB,
                               i_t basic_leaving_index,
                               i_t entering_index,
                               std::vector<f_t>& v,
                               std::vector<f_t>& delta_y_steepest_edge)
{
  i_t m                = basic_list.size();
  const i_t delta_y_nz = delta_y_sparse.i.size();
  sparse_vector_t<i_t, f_t> v_sparse(m, 0);
  // B^T delta_y = - direction * e_basic_leaving_index
  // We want B v =  - B^{-T} e_basic_leaving_index
  ft.b_solve(delta_y_sparse, v_sparse);
  if (direction == -1) { v_sparse.negate(); }
  v_sparse.scatter(v);

  const i_t leaving_index        = basic_list[basic_leaving_index];
  const f_t prev_dy_norm_squared = delta_y_steepest_edge[leaving_index];
#ifdef STEEPEST_EDGE_DEBUG
  const f_t err = std::abs(dy_norm_squared - prev_dy_norm_squared) / (1.0 + dy_norm_squared);
  if (err > 1e-3) {
    settings.log.printf("i %d j %d leaving norm error %e computed %e previous estimate %e\n",
                        basic_leaving_index,
                        leaving_index,
                        err,
                        dy_norm_squared,
                        prev_dy_norm_squared);
  }
#endif

  // B*w = A(:, leaving_index)
  // B*scaled_delta_xB = -A(:, leaving_index) so w = -scaled_delta_xB
  const f_t wr = -scaled_delta_xB.find_coefficient(basic_leaving_index);
  if (wr == 0) { return -1; }
  const f_t omegar             = dy_norm_squared / (wr * wr);
  const i_t scaled_delta_xB_nz = scaled_delta_xB.i.size();
  for (i_t h = 0; h < scaled_delta_xB_nz; ++h) {
    const i_t k = scaled_delta_xB.i[h];
    const i_t j = basic_list[k];
    if (k == basic_leaving_index) {
      const f_t w_squared      = scaled_delta_xB.x[h] * scaled_delta_xB.x[h];
      delta_y_steepest_edge[j] = (1.0 / w_squared) * dy_norm_squared;
    } else {
      const f_t wk = -scaled_delta_xB.x[h];
      f_t new_val  = delta_y_steepest_edge[j] + wk * (2.0 * v[k] / wr + wk * omegar);
      new_val      = std::max(new_val, 1e-4);
#ifdef STEEPEST_EDGE_DEBUG
      if (!(new_val >= 0)) {
        settings.log.printf("new val %e\n", new_val);
        settings.log.printf("k %d j %d norm old %e wk %e vk %e wr %e omegar %e\n",
                            k,
                            j,
                            delta_y_steepest_edge[j],
                            wk,
                            v[k],
                            wr,
                            omegar);
      }
#endif
      assert(new_val >= 0.0);
      delta_y_steepest_edge[j] = new_val;
    }
  }

  const i_t v_nz = v_sparse.i.size();
  for (i_t k = 0; k < v_nz; ++k) {
    v[v_sparse.i[k]] = 0.0;
  }

  return 0;
}

// Compute steepest edge info for entering variable
template <typename i_t, typename f_t>
i_t compute_steepest_edge_norm_entering(const simplex_solver_settings_t<i_t, f_t>& settings,
                                        i_t m,
                                        const basis_update_mpf_t<i_t, f_t>& ft,
                                        i_t basic_leaving_index,
                                        i_t entering_index,
                                        std::vector<f_t>& steepest_edge_norms)
{
  sparse_vector_t<i_t, f_t> es_sparse(m, 1);
  es_sparse.i[0] = basic_leaving_index;
  es_sparse.x[0] = -1.0;
  sparse_vector_t<i_t, f_t> delta_ys_sparse(m, 0);
  ft.b_transpose_solve(es_sparse, delta_ys_sparse);
  steepest_edge_norms[entering_index] = delta_ys_sparse.norm2_squared();

#ifdef STEEPEST_EDGE_DEBUG
  settings.log.printf("Steepest edge norm %e for entering j %d at i %d\n",
                      steepest_edge_norms[entering_index],
                      entering_index,
                      basic_leaving_index);
#endif
  return 0;
}

template <typename i_t, typename f_t>
i_t check_steepest_edge_norms(const simplex_solver_settings_t<i_t, f_t>& settings,
                              const std::vector<i_t>& basic_list,
                              const basis_update_mpf_t<i_t, f_t>& ft,
                              const std::vector<f_t>& delta_y_steepest_edge)
{
  const i_t m = basic_list.size();
  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    std::vector<f_t> ei(m);
    ei[k] = -1.0;
    std::vector<f_t> delta_yi(m);
    ft.b_transpose_solve(ei, delta_yi);
    const f_t computed_norm = vector_norm2_squared(delta_yi);
    const f_t updated_norm  = delta_y_steepest_edge[j];
    const f_t err = std::abs(computed_norm - updated_norm) / (1 + std::abs(computed_norm));
    if (err > 1e-3) {
      settings.log.printf(
        "i %d j %d computed %e updated %e err %e\n", k, j, computed_norm, updated_norm, err);
    }
  }
  return 0;
}

template <typename i_t, typename f_t>
i_t compute_perturbation(const lp_problem_t<i_t, f_t>& lp,
                         const simplex_solver_settings_t<i_t, f_t>& settings,
                         const std::vector<i_t>& delta_z_indices,
                         std::vector<f_t>& z,
                         std::vector<f_t>& objective,
                         f_t& sum_perturb)
{
  const i_t n         = lp.num_cols;
  const i_t m         = lp.num_rows;
  const f_t tight_tol = settings.tight_tol;
  i_t num_perturb     = 0;
  sum_perturb         = 0.0;
  for (i_t k = 0; k < delta_z_indices.size(); ++k) {
    const i_t j = delta_z_indices[k];
    if (lp.upper[j] == inf && lp.lower[j] > -inf && z[j] < -tight_tol) {
      const f_t violation = -z[j];
      z[j] += violation;  // z[j] <- 0
      objective[j] += violation;
      num_perturb++;
      sum_perturb += violation;
#ifdef PERTURBATION_DEBUG
      if (violation > 1e-1) {
        settings.log.printf(
          "perturbation: violation %e j %d lower %e\n", violation, j, lp.lower[j]);
      }
#endif
    } else if (lp.lower[j] == -inf && lp.upper[j] < inf && z[j] > tight_tol) {
      const f_t violation = z[j];
      z[j] -= violation;  // z[j] <- 0
      objective[j] -= violation;
      num_perturb++;
      sum_perturb += violation;
#ifdef PERTURBATION_DEWBUG
      if (violation > 1e-1) {
        settings.log.printf(
          "perturbation: violation %e j %d upper %e\n", violation, j, lp.upper[j]);
      }
#endif
    }
  }
#ifdef PERTURBATION_DEBUG
  if (num_perturb > 0) {
    settings.log.printf("Perturbed %d dual variables by %e\n", num_perturb, sum_perturb);
  }
#endif
  return 0;
}

template <typename i_t>
void reset_basis_mark(const std::vector<i_t>& basic_list,
                      const std::vector<i_t>& nonbasic_list,
                      std::vector<i_t>& basic_mark,
                      std::vector<i_t>& nonbasic_mark)
{
  const i_t m         = basic_list.size();
  const i_t n         = nonbasic_mark.size();
  const i_t n_minus_m = n - m;

  for (i_t k = 0; k < n; k++) {
    basic_mark[k] = -1;
  }

  for (i_t k = 0; k < n; k++) {
    nonbasic_mark[k] = -1;
  }

  for (i_t k = 0; k < n_minus_m; k++) {
    nonbasic_mark[nonbasic_list[k]] = k;
  }

  for (i_t k = 0; k < m; k++) {
    basic_mark[basic_list[k]] = k;
  }
}

template <typename i_t, typename f_t>
void compute_delta_y(const basis_update_mpf_t<i_t, f_t>& ft,
                     i_t basic_leaving_index,
                     i_t direction,
                     sparse_vector_t<i_t, f_t>& delta_y_sparse,
                     sparse_vector_t<i_t, f_t>& UTsol_sparse)
{
  const i_t m = delta_y_sparse.n;
  // BT*delta_y = -delta_zB = -sigma*ei
  sparse_vector_t<i_t, f_t> ei_sparse(m, 1);
  ei_sparse.i[0] = basic_leaving_index;
  ei_sparse.x[0] = -direction;
  ft.b_transpose_solve(ei_sparse, delta_y_sparse, UTsol_sparse);

  if (direction != -1) {
    // We solved BT*delta_y = -sigma*ei, but for the update we need
    // UT*etilde = ei. So we need to flip the sign of the solution
    // in the case that sigma == 1.
    UTsol_sparse.negate();
  }

#ifdef CHECK_B_TRANSPOSE_SOLVE
  std::vector<f_t> delta_y_sparse_vector_check(m);
  delta_y_sparse.to_dense(delta_y_sparse_vector_check);
  f_t error_check = 0.0;
  for (i_t k = 0; k < m; ++k) {
    if (std::abs(delta_y[k] - delta_y_sparse_vector_check[k]) > 1e-6) {
      settings.log.printf(
        "\tBTranspose error %d %e %e\n", k, delta_y[k], delta_y_sparse_vector_check[k]);
    }
    error_check += std::abs(delta_y[k] - delta_y_sparse_vector_check[k]);
  }
  if (error_check > 1e-6) { settings.log.printf("BTranspose error %e\n", error_check); }
  std::vector<f_t> residual(m);
  b_transpose_multiply(lp, basic_list, delta_y_sparse_vector_check, residual);
  for (i_t k = 0; k < m; ++k) {
    if (std::abs(residual[k] - ei[k]) > 1e-6) {
      settings.log.printf("\tBTranspose multiply error %d %e %e\n", k, residual[k], ei[k]);
    }
  }
#endif
}

template <typename i_t, typename f_t>
i_t update_dual_variables(const sparse_vector_t<i_t, f_t>& delta_y_sparse,
                          const std::vector<i_t>& delta_z_indices,
                          const std::vector<f_t>& delta_z,
                          f_t step_length,
                          i_t leaving_index,
                          std::vector<f_t>& y,
                          std::vector<f_t>& z)
{
  // Update dual variables
  // y <- y + steplength * delta_y
  const i_t delta_y_nz = delta_y_sparse.i.size();
  for (i_t k = 0; k < delta_y_nz; ++k) {
    const i_t i = delta_y_sparse.i[k];
    y[i] += step_length * delta_y_sparse.x[k];
  }
  // z <- z + steplength * delta_z
  const i_t delta_z_nz = delta_z_indices.size();
  for (i_t k = 0; k < delta_z_nz; ++k) {
    const i_t j = delta_z_indices[k];
    z[j] += step_length * delta_z[j];
  }
  z[leaving_index] += step_length * delta_z[leaving_index];
  return 0;
}

template <typename i_t, typename f_t>
void adjust_for_flips(const basis_update_mpf_t<i_t, f_t>& ft,
                      const std::vector<i_t>& basic_list,
                      const std::vector<i_t>& delta_z_indices,
                      std::vector<i_t>& atilde_index,
                      std::vector<f_t>& atilde,
                      std::vector<i_t>& atilde_mark,
                      sparse_vector_t<i_t, f_t>& delta_xB_0_sparse,
                      std::vector<f_t>& delta_x_flip,
                      std::vector<f_t>& x)
{
  const i_t m         = basic_list.size();
  const i_t atilde_nz = atilde_index.size();
  // B*delta_xB_0 = atilde
  sparse_vector_t<i_t, f_t> atilde_sparse(m, atilde_nz);
  for (i_t k = 0; k < atilde_nz; ++k) {
    atilde_sparse.i[k] = atilde_index[k];
    atilde_sparse.x[k] = atilde[atilde_index[k]];
  }
  ft.b_solve(atilde_sparse, delta_xB_0_sparse);
  const i_t delta_xB_0_nz = delta_xB_0_sparse.i.size();
  for (i_t k = 0; k < delta_xB_0_nz; ++k) {
    const i_t j = basic_list[delta_xB_0_sparse.i[k]];
    x[j] += delta_xB_0_sparse.x[k];
  }

  for (i_t j : delta_z_indices) {
    x[j] += delta_x_flip[j];
    delta_x_flip[j] = 0.0;
  }

  // Clear atilde
  for (i_t k = 0; k < atilde_index.size(); ++k) {
    atilde[atilde_index[k]] = 0.0;
  }
  // Clear atilde_mark
  for (i_t k = 0; k < atilde_mark.size(); ++k) {
    atilde_mark[k] = 0;
  }
  atilde_index.clear();
}

template <typename i_t, typename f_t>
i_t compute_delta_x(const lp_problem_t<i_t, f_t>& lp,
                    const basis_update_mpf_t<i_t, f_t>& ft,
                    i_t entering_index,
                    i_t leaving_index,
                    i_t basic_leaving_index,
                    i_t direction,
                    const std::vector<i_t>& basic_list,
                    const std::vector<f_t>& delta_x_flip,
                    const sparse_vector_t<i_t, f_t>& rhs_sparse,
                    const std::vector<f_t>& x,
                    sparse_vector_t<i_t, f_t>& utilde_sparse,
                    sparse_vector_t<i_t, f_t>& scaled_delta_xB_sparse,
                    std::vector<f_t>& delta_x)
{
  f_t delta_x_leaving = direction == 1 ? lp.lower[leaving_index] - x[leaving_index]
                                       : lp.upper[leaving_index] - x[leaving_index];
  // B*w = -A(:, entering)
  ft.b_solve(rhs_sparse, scaled_delta_xB_sparse, utilde_sparse);
  scaled_delta_xB_sparse.negate();

#ifdef CHECK_B_SOLVE
  std::vector<f_t> scaled_delta_xB(m);
  {
    std::vector<f_t> residual_B(m);
    b_multiply(lp, basic_list, scaled_delta_xB, residual_B);
    f_t err_max = 0;
    for (i_t k = 0; k < m; ++k) {
      const f_t err = std::abs(rhs[k] + residual_B[k]);
      if (err >= 1e-6) {
        settings.log.printf(
          "Bsolve diff %d %e rhs %e residual %e\n", k, err, rhs[k], residual_B[k]);
      }
      err_max = std::max(err_max, err);
    }
    if (err_max > 1e-6) { settings.log.printf("B multiply error %e\n", err_max); }
  }
#endif

  f_t scale = scaled_delta_xB_sparse.find_coefficient(basic_leaving_index);
  if (scale != scale) {
    // We couldn't find a coefficient for the basic leaving index.
    // The coefficient might be very small. Switch to a regular solve and try to recover.
    std::vector<f_t> rhs;
    rhs_sparse.to_dense(rhs);
    const i_t m = basic_list.size();
    std::vector<f_t> scaled_delta_xB(m);
    ft.b_solve(rhs, scaled_delta_xB);
    if (scaled_delta_xB[basic_leaving_index] != 0.0 &&
        !std::isnan(scaled_delta_xB[basic_leaving_index])) {
      scaled_delta_xB_sparse.from_dense(scaled_delta_xB);
      scaled_delta_xB_sparse.negate();
      scale = -scaled_delta_xB[basic_leaving_index];
    } else {
      return -1;
    }
  }
  const f_t primal_step_length = delta_x_leaving / scale;
  const i_t scaled_delta_xB_nz = scaled_delta_xB_sparse.i.size();
  for (i_t k = 0; k < scaled_delta_xB_nz; ++k) {
    const i_t j = basic_list[scaled_delta_xB_sparse.i[k]];
    delta_x[j]  = primal_step_length * scaled_delta_xB_sparse.x[k];
  }
  delta_x[leaving_index]  = delta_x_leaving;
  delta_x[entering_index] = primal_step_length;
  return 0;
}

template <typename i_t, typename f_t>
void update_primal_variables(const sparse_vector_t<i_t, f_t>& scaled_delta_xB_sparse,
                             const std::vector<i_t>& basic_list,
                             const std::vector<f_t>& delta_x,
                             i_t entering_index,
                             std::vector<f_t>& x)
{
  // x <- x + delta_x
  const i_t scaled_delta_xB_nz = scaled_delta_xB_sparse.i.size();
  for (i_t k = 0; k < scaled_delta_xB_nz; ++k) {
    const i_t j = basic_list[scaled_delta_xB_sparse.i[k]];
    x[j] += delta_x[j];
  }
  // Leaving index already included above
  x[entering_index] += delta_x[entering_index];
}

template <typename i_t, typename f_t>
void update_objective(const std::vector<i_t>& basic_list,
                      const std::vector<i_t>& changed_basic_indices,
                      const std::vector<f_t>& objective,
                      const std::vector<f_t>& delta_x,
                      i_t entering_index,
                      f_t& obj)
{
  const i_t changed_basic_nz = changed_basic_indices.size();
  for (i_t k = 0; k < changed_basic_nz; ++k) {
    const i_t j = basic_list[changed_basic_indices[k]];
    obj += delta_x[j] * objective[j];
  }
  // Leaving index already included above
  obj += delta_x[entering_index] * objective[entering_index];
}

template <typename i_t, typename f_t>
f_t dual_infeasibility(const lp_problem_t<i_t, f_t>& lp,
                       const simplex_solver_settings_t<i_t, f_t>& settings,
                       const std::vector<variable_status_t>& vstatus,
                       const std::vector<f_t>& z,
                       f_t tight_tol,
                       f_t dual_tol)
{
  const i_t n             = lp.num_cols;
  const i_t m             = lp.num_rows;
  i_t num_infeasible      = 0;
  f_t sum_infeasible      = 0.0;
  i_t lower_bound_inf     = 0;
  i_t upper_bound_inf     = 0;
  i_t free_inf            = 0;
  i_t non_basic_lower_inf = 0;
  i_t non_basic_upper_inf = 0;

  for (i_t j = 0; j < n; ++j) {
    if (vstatus[j] == variable_status_t::NONBASIC_FIXED) { continue; }
    if (lp.upper[j] == inf && lp.lower[j] > -inf && z[j] < -tight_tol) {
      // -inf < l_j <= x_j < inf, so need z_j > 0 to be feasible
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
      lower_bound_inf++;
      settings.log.debug("lower_bound_inf %d lower %e upper %e z %e vstatus %d\n",
                         j,
                         lp.lower[j],
                         lp.upper[j],
                         z[j],
                         static_cast<int>(vstatus[j]));
    } else if (lp.lower[j] == -inf && lp.upper[j] < inf && z[j] > tight_tol) {
      // -inf < x_j <= u_j < inf, so need z_j < 0 to be feasible
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
      upper_bound_inf++;
      settings.log.debug("upper_bound_inf %d upper %e lower %e z %e vstatus %d\n",
                         j,
                         lp.upper[j],
                         lp.lower[j],
                         z[j],
                         static_cast<int>(vstatus[j]));
    } else if (lp.lower[j] == -inf && lp.upper[j] == inf && z[j] > tight_tol) {
      // -inf < x_j < inf, so need z_j = 0 to be feasible
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
      free_inf++;
    } else if (lp.lower[j] == -inf && lp.upper[j] == inf && z[j] < -tight_tol) {
      // -inf < x_j < inf, so need z_j = 0 to be feasible
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
      free_inf++;
    } else if (vstatus[j] == variable_status_t::NONBASIC_LOWER && z[j] < -dual_tol) {
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
      non_basic_lower_inf++;
    } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER && z[j] > dual_tol) {
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
      non_basic_upper_inf++;
    }
  }

#ifdef DUAL_INFEASIBILE_DEBUG
  if (num_infeasible > 0) {
    settings.log.printf(
      "Infeasibilities %e: lower %d upper %d free %d nonbasic lower %d "
      "nonbasic upper %d\n",
      sum_infeasible,
      lower_bound_inf,
      upper_bound_inf,
      free_inf,
      non_basic_lower_inf,
      non_basic_upper_inf);
    settings.log.printf("num infeasible %d\n", num_infeasible);
  }
#endif
  return sum_infeasible;
}

template <typename i_t, typename f_t>
f_t primal_infeasibility(const lp_problem_t<i_t, f_t>& lp,
                         const simplex_solver_settings_t<i_t, f_t>& settings,
                         const std::vector<variable_status_t>& vstatus,
                         const std::vector<f_t>& x)
{
  const i_t n    = lp.num_cols;
  f_t primal_inf = 0;
  for (i_t j = 0; j < n; ++j) {
    if (x[j] < lp.lower[j]) {
      // x_j < l_j => -x_j > -l_j => -x_j + l_j > 0
      const f_t infeas = -x[j] + lp.lower[j];
      primal_inf += infeas;
#ifdef PRIMAL_INFEASIBLE_DEBUG
      if (infeas > settings.primal_tol) {
        settings.log.printf("x %d infeas %e lo %e val %e up %e vstatus %d\n",
                            j,
                            infeas,
                            lp.lower[j],
                            x[j],
                            lp.upper[j],
                            static_cast<int>(vstatus[j]));
      }
#endif
    }
    if (x[j] > lp.upper[j]) {
      // x_j > u_j => x_j - u_j > 0
      const f_t infeas = x[j] - lp.upper[j];
      primal_inf += infeas;
#ifdef PRIMAL_INFEASIBLE_DEBUG
      if (infeas > settings.primal_tol) {
        settings.log.printf("x %d infeas %e lo %e val %e up %e vstatus %d\n",
                            j,
                            infeas,
                            lp.lower[j],
                            x[j],
                            lp.upper[j],
                            static_cast<int>(vstatus[j]));
      }
#endif
    }
  }
  return primal_inf;
}

template <typename i_t, typename f_t>
void check_primal_infeasibilities(const lp_problem_t<i_t, f_t>& lp,
                                  const simplex_solver_settings_t<i_t, f_t>& settings,
                                  const std::vector<i_t>& basic_list,
                                  const std::vector<f_t>& x,
                                  const std::vector<f_t>& squared_infeasibilities,
                                  const std::vector<i_t>& infeasibility_indices)
{
  const i_t m = basic_list.size();
  for (i_t k = 0; k < m; ++k) {
    const i_t j            = basic_list[k];
    const f_t lower_infeas = lp.lower[j] - x[j];
    const f_t upper_infeas = x[j] - lp.upper[j];
    const f_t infeas       = std::max(lower_infeas, upper_infeas);
    if (infeas > settings.primal_tol) {
      const f_t square_infeas = infeas * infeas;
      if (square_infeas != squared_infeasibilities[j]) {
        settings.log.printf("Primal infeasibility mismatch %d %e != %e\n",
                            j,
                            square_infeas,
                            squared_infeasibilities[j]);
      }
      bool found = false;
      for (i_t h = 0; h < infeasibility_indices.size(); ++h) {
        if (infeasibility_indices[h] == j) {
          found = true;
          break;
        }
      }
      if (!found) { settings.log.printf("Infeasibility index not found %d\n", j); }
    }
  }
}

template <typename i_t>
void check_basic_infeasibilities(const std::vector<i_t>& basic_list,
                                 const std::vector<i_t>& basic_mark,
                                 const std::vector<i_t>& infeasibility_indices,
                                 i_t info)
{
  for (i_t k = 0; k < infeasibility_indices.size(); ++k) {
    const i_t j = infeasibility_indices[k];
    if (basic_mark[j] < 0) { printf("%d basic_infeasibilities basic_mark[%d] < 0\n", info, j); }
  }
}

template <typename i_t, typename f_t>
void check_update(const lp_problem_t<i_t, f_t>& lp,
                  const simplex_solver_settings_t<i_t, f_t>& settings,
                  const basis_update_t<i_t, f_t>& ft,
                  const std::vector<i_t>& basic_list,
                  const std::vector<i_t>& basic_leaving_index)
{
  const i_t m = basic_list.size();
  csc_matrix_t<i_t, f_t> Btest(m, m, 1);
  ft.multiply_lu(Btest);
  {
    csc_matrix_t<i_t, f_t> B(m, m, 1);
    form_b(lp.A, basic_list, B);
    csc_matrix_t<i_t, f_t> Diff(m, m, 1);
    add(Btest, B, 1.0, -1.0, Diff);
    const f_t err = Diff.norm1();
    if (err > settings.primal_tol) { settings.log.printf("|| B - L*U || %e\n", Diff.norm1()); }
    if (err > settings.primal_tol) {
      for (i_t j = 0; j < m; ++j) {
        for (i_t p = Diff.col_start[j]; p < Diff.col_start[j + 1]; ++p) {
          const i_t i = Diff.i[p];
          if (Diff.x[p] != 0.0) { settings.log.printf("Diff %d %d %e\n", j, i, Diff.x[p]); }
        }
      }
    }
    settings.log.printf("basic leaving index %d\n", basic_leaving_index);
    assert(err < settings.primal_tol);
  }
}

template <typename i_t, typename f_t>
void check_basis_mark(const simplex_solver_settings_t<i_t, f_t>& settings,
                      const std::vector<i_t>& basic_list,
                      const std::vector<i_t>& nonbasic_list,
                      const std::vector<i_t>& basic_mark,
                      const std::vector<i_t>& nonbasic_mark)
{
  const i_t m = basic_list.size();
  const i_t n = basic_mark.size();
  for (i_t k = 0; k < m; k++) {
    if (basic_mark[basic_list[k]] != k) {
      settings.log.printf("Basic mark %d %d\n", basic_list[k], k);
    }
  }
  for (i_t k = 0; k < n - m; k++) {
    if (nonbasic_mark[nonbasic_list[k]] != k) {
      settings.log.printf("Nonbasic mark %d %d\n", nonbasic_list[k], k);
    }
  }
}

template <typename i_t, typename f_t>
void bound_info(const lp_problem_t<i_t, f_t>& lp,
                const simplex_solver_settings_t<i_t, f_t>& settings)
{
  i_t n                 = lp.num_cols;
  i_t num_free          = 0;
  i_t num_boxed         = 0;
  i_t num_lower_bounded = 0;
  i_t num_upper_bounded = 0;
  i_t num_fixed         = 0;
  for (i_t j = 0; j < n; ++j) {
    if (lp.lower[j] == lp.upper[j]) {
      num_fixed++;
    } else if (lp.lower[j] > -inf && lp.upper[j] < inf) {
      num_boxed++;
    } else if (lp.lower[j] > -inf && lp.upper[j] == inf) {
      num_lower_bounded++;
    } else if (lp.lower[j] == -inf && lp.upper[j] < inf) {
      num_upper_bounded++;
    } else if (lp.lower[j] == -inf && lp.upper[j] == inf) {
      num_free++;
    }
  }
  settings.log.debug("Fixed %d Free %d Boxed %d Lower %d Upper %d\n",
                     num_fixed,
                     num_free,
                     num_boxed,
                     num_lower_bounded,
                     num_upper_bounded);
}

template <typename i_t, typename f_t>
void set_primal_variables_on_bounds(const lp_problem_t<i_t, f_t>& lp,
                                    const simplex_solver_settings_t<i_t, f_t>& settings,
                                    const std::vector<f_t>& z,
                                    std::vector<variable_status_t>& vstatus,
                                    std::vector<f_t>& x)
{
  const i_t n = lp.num_cols;
  for (i_t j = 0; j < n; ++j) {
    // We set z_j = 0 for basic variables
    // But we explicitally skip setting basic variables here
    if (vstatus[j] == variable_status_t::BASIC) { continue; }
    // We will flip the status of variables between nonbasic lower and nonbasic
    // upper here to improve dual feasibility
    const f_t fixed_tolerance = settings.fixed_tol;
    if (std::abs(lp.lower[j] - lp.upper[j]) < fixed_tolerance) {
      if (vstatus[j] != variable_status_t::NONBASIC_FIXED) {
        settings.log.debug("Setting fixed variable %d to %e (current %e). vstatus %d\n",
                           j,
                           lp.lower[j],
                           x[j],
                           static_cast<int>(vstatus[j]));
      }
      x[j]       = lp.lower[j];
      vstatus[j] = variable_status_t::NONBASIC_FIXED;
    } else if (z[j] == 0 && lp.lower[j] > -inf && vstatus[j] == variable_status_t::NONBASIC_LOWER) {
      x[j] = lp.lower[j];
    } else if (z[j] == 0 && lp.upper[j] < inf && vstatus[j] == variable_status_t::NONBASIC_UPPER) {
      x[j] = lp.upper[j];
    } else if (z[j] >= 0 && lp.lower[j] > -inf) {
      if (vstatus[j] != variable_status_t::NONBASIC_LOWER) {
        settings.log.debug(
          "Setting nonbasic lower variable (zj %e) %d to %e (current %e). vstatus %d\n",
          z[j],
          j,
          lp.lower[j],
          x[j],
          static_cast<int>(vstatus[j]));
      }
      x[j]       = lp.lower[j];
      vstatus[j] = variable_status_t::NONBASIC_LOWER;
    } else if (z[j] <= 0 && lp.upper[j] < inf) {
      if (vstatus[j] != variable_status_t::NONBASIC_UPPER) {
        settings.log.debug(
          "Setting nonbasic upper variable (zj %e) %d to %e (current %e). vstatus %d\n",
          z[j],
          j,
          lp.upper[j],
          x[j],
          static_cast<int>(vstatus[j]));
      }
      x[j]       = lp.upper[j];
      vstatus[j] = variable_status_t::NONBASIC_UPPER;
    } else if (lp.upper[j] == inf && lp.lower[j] > -inf && z[j] < 0) {
      // dual infeasible
      if (vstatus[j] != variable_status_t::NONBASIC_LOWER) {
        settings.log.debug("Setting nonbasic lower variable %d to %e (current %e). vstatus %d\n",
                           j,
                           lp.lower[j],
                           x[j],
                           static_cast<int>(vstatus[j]));
      }
      x[j]       = lp.lower[j];
      vstatus[j] = variable_status_t::NONBASIC_LOWER;
    } else if (lp.lower[j] == -inf && lp.upper[j] < inf && z[j] > 0) {
      // dual infeasible
      if (vstatus[j] != variable_status_t::NONBASIC_UPPER) {
        settings.log.debug("Setting nonbasic upper variable %d to %e (current %e). vstatus %d\n",
                           j,
                           lp.upper[j],
                           x[j],
                           static_cast<int>(vstatus[j]));
      }
      x[j]       = lp.upper[j];
      vstatus[j] = variable_status_t::NONBASIC_UPPER;
    } else if (lp.lower[j] == -inf && lp.upper[j] == inf) {
      x[j] = 0;  // Set nonbasic free variables to 0 this overwrites previous lines
      if (vstatus[j] != variable_status_t::NONBASIC_FREE) {
        settings.log.debug(
          "Setting free variable %d to %e. vstatus %d\n", j, 0, static_cast<int>(vstatus[j]));
      }
      vstatus[j] = variable_status_t::NONBASIC_FREE;
      settings.log.printf("Setting free variable %d as nonbasic at 0\n", j);
    } else {
      assert(1 == 0);
    }
  }
}

template <typename f_t>
f_t compute_perturbed_objective(const std::vector<f_t>& objective, const std::vector<f_t>& x)
{
  const size_t n = objective.size();
  f_t obj_val    = 0.0;
  for (size_t j = 0; j < n; ++j) {
    obj_val += objective[j] * x[j];
  }
  return obj_val;
}

template <typename i_t, typename f_t>
f_t amount_of_perturbation(const lp_problem_t<i_t, f_t>& lp, const std::vector<f_t>& objective)
{
  f_t perturbation = 0.0;
  const i_t n      = lp.num_cols;
  for (i_t j = 0; j < n; ++j) {
    perturbation += std::abs(lp.objective[j] - objective[j]);
  }
  return perturbation;
}

template <typename i_t, typename f_t>
void prepare_optimality(const lp_problem_t<i_t, f_t>& lp,
                        const simplex_solver_settings_t<i_t, f_t>& settings,
                        basis_update_mpf_t<i_t, f_t>& ft,
                        const std::vector<f_t>& objective,
                        const std::vector<i_t>& basic_list,
                        const std::vector<i_t>& nonbasic_list,
                        const std::vector<variable_status_t>& vstatus,
                        int phase,
                        f_t start_time,
                        f_t max_val,
                        i_t iter,
                        const std::vector<f_t>& x,
                        std::vector<f_t>& y,
                        std::vector<f_t>& z,
                        lp_solution_t<i_t, f_t>& sol)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;

  sol.objective      = compute_objective(lp, sol.x);
  sol.user_objective = compute_user_objective(lp, sol.objective);
  f_t perturbation   = phase2::amount_of_perturbation(lp, objective);
  if (perturbation > 1e-6 && phase == 2) {
    // Try to remove perturbation
    std::vector<f_t> unperturbed_y(m);
    std::vector<f_t> unperturbed_z(n);
    phase2::compute_dual_solution_from_basis(
      lp, ft, basic_list, nonbasic_list, unperturbed_y, unperturbed_z);
    {
      const f_t dual_infeas = phase2::dual_infeasibility(
        lp, settings, vstatus, unperturbed_z, settings.tight_tol, settings.dual_tol);
      if (dual_infeas <= settings.dual_tol) {
        settings.log.printf("Removed perturbation of %.2e.\n", perturbation);
        z            = unperturbed_z;
        y            = unperturbed_y;
        perturbation = 0.0;
      } else {
        settings.log.printf("Failed to remove perturbation of %.2e.\n", perturbation);
      }
    }
  }

  sol.l2_primal_residual  = l2_primal_residual(lp, sol);
  sol.l2_dual_residual    = l2_dual_residual(lp, sol);
  const f_t dual_infeas   = phase2::dual_infeasibility(lp, settings, vstatus, z, 0.0, 0.0);
  const f_t primal_infeas = phase2::primal_infeasibility(lp, settings, vstatus, x);
  if (phase == 1 && iter > 0) {
    settings.log.printf("Dual phase I complete. Iterations %d. Time %.2f\n", iter, toc(start_time));
  }
  if (phase == 2) {
    if (!settings.inside_mip) {
      settings.log.printf("\n");
      settings.log.printf(
        "Optimal solution found in %d iterations and %.2fs\n", iter, toc(start_time));
      settings.log.printf("Objective %+.8e\n", sol.user_objective);
      settings.log.printf("\n");
      settings.log.printf("Primal infeasibility (abs): %.2e\n", primal_infeas);
      settings.log.printf("Dual infeasibility (abs):   %.2e\n", dual_infeas);
      settings.log.printf("Perturbation:               %.2e\n", perturbation);
    } else {
      settings.log.printf("\n");
      settings.log.printf(
        "Root relaxation solution found in %d iterations and %.2fs\n", iter, toc(start_time));
      settings.log.printf("Root relaxation objective %+.8e\n", sol.user_objective);
      settings.log.printf("\n");
    }
  }
}

template <typename i_t, typename f_t>
class phase2_timers_t {
 public:
  phase2_timers_t(bool should_time)
    : record_time(should_time),
      bfrt_time(0),
      pricing_time(0),
      btran_time(0),
      ftran_time(0),
      flip_time(0),
      delta_z_time(0),
      se_norms_time(0),
      se_entering_time(0),
      lu_update_time(0),
      perturb_time(0),
      vector_time(0),
      objective_time(0),
      update_infeasibility_time(0)
  {
  }

  void start_timer()
  {
    if (!record_time) { return; }
    start_time = tic();
  }

  f_t stop_timer()
  {
    if (!record_time) { return 0.0; }
    return toc(start_time);
  }

  void print_timers(const simplex_solver_settings_t<i_t, f_t>& settings) const
  {
    if (!record_time) { return; }
    const f_t total_time = bfrt_time + pricing_time + btran_time + ftran_time + flip_time +
                           delta_z_time + lu_update_time + se_norms_time + se_entering_time +
                           perturb_time + vector_time + objective_time + update_infeasibility_time;
    // clang-format off
    settings.log.printf("BFRT time       %.2fs %4.1f%\n", bfrt_time, 100.0 * bfrt_time / total_time);
    settings.log.printf("Pricing time    %.2fs %4.1f%\n", pricing_time, 100.0 * pricing_time / total_time);
    settings.log.printf("BTran time      %.2fs %4.1f%\n", btran_time, 100.0 * btran_time / total_time);
    settings.log.printf("FTran time      %.2fs %4.1f%\n", ftran_time, 100.0 * ftran_time / total_time);
    settings.log.printf("Flip time       %.2fs %4.1f%\n", flip_time, 100.0 * flip_time / total_time);
    settings.log.printf("Delta_z time    %.2fs %4.1f%\n", delta_z_time, 100.0 * delta_z_time / total_time);
    settings.log.printf("LU update time  %.2fs %4.1f%\n", lu_update_time, 100.0 * lu_update_time / total_time);
    settings.log.printf("SE norms time   %.2fs %4.1f%\n", se_norms_time, 100.0 * se_norms_time / total_time);
    settings.log.printf("SE enter time   %.2fs %4.1f%\n", se_entering_time, 100.0 * se_entering_time / total_time);
    settings.log.printf("Perturb time    %.2fs %4.1f%\n", perturb_time, 100.0 * perturb_time / total_time);
    settings.log.printf("Vector time     %.2fs %4.1f%\n", vector_time, 100.0 * vector_time / total_time);
    settings.log.printf("Objective time  %.2fs %4.1f%\n", objective_time, 100.0 * objective_time / total_time);
    settings.log.printf("Inf update time %.2fs %4.1f%\n", update_infeasibility_time, 100.0 * update_infeasibility_time / total_time);
    settings.log.printf("Sum             %.2fs\n", total_time);
    // clang-format on
  }
  f_t bfrt_time;
  f_t pricing_time;
  f_t btran_time;
  f_t ftran_time;
  f_t flip_time;
  f_t delta_z_time;
  f_t se_norms_time;
  f_t se_entering_time;
  f_t lu_update_time;
  f_t perturb_time;
  f_t vector_time;
  f_t objective_time;
  f_t update_infeasibility_time;

 private:
  f_t start_time;
  bool record_time;
};

}  // namespace phase2

template <typename i_t, typename f_t>
dual::status_t dual_phase2(i_t phase,
                           i_t slack_basis,
                           f_t start_time,
                           const lp_problem_t<i_t, f_t>& lp,
                           const simplex_solver_settings_t<i_t, f_t>& settings,
                           std::vector<variable_status_t>& vstatus,
                           lp_solution_t<i_t, f_t>& sol,
                           i_t& iter,
                           std::vector<f_t>& delta_y_steepest_edge)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;
  assert(m <= n);
  assert(vstatus.size() == n);
  assert(lp.A.m == m);
  assert(lp.A.n == n);
  assert(lp.objective.size() == n);
  assert(lp.lower.size() == n);
  assert(lp.upper.size() == n);
  assert(lp.rhs.size() == m);
  std::vector<i_t> basic_list(m);
  std::vector<i_t> nonbasic_list;
  std::vector<i_t> superbasic_list;

  std::vector<f_t>& x = sol.x;
  std::vector<f_t>& y = sol.y;
  std::vector<f_t>& z = sol.z;

  dual::status_t status = dual::status_t::UNSET;

  // Perturbed objective
  std::vector<f_t> objective = lp.objective;

  settings.log.printf("Dual Simplex Phase %d\n", phase);
  std::vector<variable_status_t> vstatus_old = vstatus;
  std::vector<f_t> z_old                     = z;

  phase2::bound_info(lp, settings);
  get_basis_from_vstatus(m, vstatus, basic_list, nonbasic_list, superbasic_list);
  assert(superbasic_list.size() == 0);
  assert(nonbasic_list.size() == n - m);

  // Compute L*U = A(p, basic_list)
  csc_matrix_t<i_t, f_t> L(m, m, 1);
  csc_matrix_t<i_t, f_t> U(m, m, 1);
  std::vector<i_t> pinv(m);
  std::vector<i_t> p;
  std::vector<i_t> q;
  std::vector<i_t> deficient;
  std::vector<i_t> slacks_needed;

  if (factorize_basis(lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed) ==
      -1) {
    settings.log.debug("Initial factorization failed\n");
    basis_repair(lp.A, settings, deficient, slacks_needed, basic_list, nonbasic_list, vstatus);
    if (factorize_basis(lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed) ==
        -1) {
      return dual::status_t::NUMERICAL;
    }
    settings.log.printf("Basis repaired\n");
  }
  if (toc(start_time) > settings.time_limit) { return dual::status_t::TIME_LIMIT; }
  assert(q.size() == m);
  reorder_basic_list(q, basic_list);
  basis_update_mpf_t<i_t, f_t> ft(L, U, p, settings.refactor_frequency);

  std::vector<f_t> c_basic(m);
  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    c_basic[k]  = objective[j];
  }

  // Solve B'*y = cB
  ft.b_transpose_solve(c_basic, y);
  if (toc(start_time) > settings.time_limit) { return dual::status_t::TIME_LIMIT; }
  constexpr bool print_norms = false;
  if constexpr (print_norms) {
    settings.log.printf(
      "|| y || %e || cB || %e\n", vector_norm_inf<i_t, f_t>(y), vector_norm_inf<i_t, f_t>(c_basic));
  }

  phase2::compute_reduced_costs(objective, lp.A, y, basic_list, nonbasic_list, z);
  if constexpr (print_norms) { settings.log.printf("|| z || %e\n", vector_norm_inf<i_t, f_t>(z)); }

#ifdef COMPUTE_DUAL_RESIDUAL
  std::vector<f_t> dual_res1;
  compute_dual_residual(lp.A, objective, y, z, dual_res1);
  f_t dual_res_norm = vector_norm_inf<i_t, f_t>(dual_res1);
  if (dual_res_norm > settings.tight_tol) {
    settings.log.printf("|| A'*y + z - c || %e\n", dual_res_norm);
  }
  assert(dual_res_norm < 1e-3);
#endif

  phase2::set_primal_variables_on_bounds(lp, settings, z, vstatus, x);

#ifdef PRINT_VSTATUS_CHANGES
  i_t num_vstatus_changes;
  i_t num_z_changes;
  phase2::vstatus_changes(vstatus, vstatus_old, z, z_old, num_vstatus_changes, num_z_changes);
  settings.log.printf("Number of vstatus changes %d\n", num_vstatus_changes);
  settings.log.printf("Number of z changes %d\n", num_z_changes);
#endif

  const f_t init_dual_inf =
    phase2::dual_infeasibility(lp, settings, vstatus, z, settings.tight_tol, settings.dual_tol);
  if (init_dual_inf > settings.dual_tol) {
    settings.log.printf("Initial dual infeasibility %e\n", init_dual_inf);
  }

  for (i_t j = 0; j < n; ++j) {
    if (lp.lower[j] == -inf && lp.upper[j] == inf && vstatus[j] != variable_status_t::BASIC) {
      settings.log.printf("Free variable %d vstatus %d\n", j, vstatus[j]);
    }
  }

  phase2::compute_primal_variables(
    ft, lp.rhs, lp.A, basic_list, nonbasic_list, settings.tight_tol, x);

  if (toc(start_time) > settings.time_limit) { return dual::status_t::TIME_LIMIT; }
  if (print_norms) { settings.log.printf("|| x || %e\n", vector_norm2<i_t, f_t>(x)); }

#ifdef COMPUTE_PRIMAL_RESIDUAL
  std::vector<f_t> residual = lp.rhs;
  matrix_vector_multiply(lp.A, 1.0, x, -1.0, residual);
  f_t primal_residual = vector_norm_inf<i_t, f_t>(residual);
  if (primal_residual > settings.primal_tol) {
    settings.log.printf("|| A*x - b || %e\n", primal_residual);
  }
#endif

  if (delta_y_steepest_edge.size() == 0) {
    delta_y_steepest_edge.resize(n);
    if (slack_basis) {
      phase2::initialize_steepest_edge_norms_from_slack_basis(
        basic_list, nonbasic_list, delta_y_steepest_edge);
    } else {
      std::fill(delta_y_steepest_edge.begin(), delta_y_steepest_edge.end(), -1);
      if (phase2::initialize_steepest_edge_norms(
            lp, settings, start_time, basic_list, ft, delta_y_steepest_edge) == -1) {
        return dual::status_t::TIME_LIMIT;
      }
    }
  } else {
    settings.log.printf("using exisiting steepest edge %e\n",
                        vector_norm2<i_t, f_t>(delta_y_steepest_edge));
  }

  if (phase == 2) {
    settings.log.printf(" Iter     Objective           Num Inf.  Sum Inf.     Perturb  Time\n");
  }

  const i_t iter_limit = settings.iteration_limit;
  std::vector<f_t> delta_y(m, 0.0);
  std::vector<f_t> delta_z(n, 0.0);
  std::vector<f_t> delta_x(n, 0.0);
  std::vector<f_t> delta_x_flip(n, 0.0);
  std::vector<f_t> atilde(m, 0.0);
  std::vector<i_t> atilde_mark(m, 0);
  std::vector<i_t> atilde_index;
  std::vector<i_t> nonbasic_mark(n);
  std::vector<i_t> basic_mark(n);
  std::vector<i_t> delta_z_mark(n, 0);
  std::vector<i_t> delta_z_indices;
  std::vector<f_t> v(m, 0.0);
  std::vector<f_t> squared_infeasibilities;
  std::vector<i_t> infeasibility_indices;

  delta_z_indices.reserve(n);

  phase2::reset_basis_mark(basic_list, nonbasic_list, basic_mark, nonbasic_mark);

  std::vector<uint8_t> bounded_variables(n, 0);
  phase2::compute_bounded_info(lp.lower, lp.upper, bounded_variables);

  f_t primal_infeasibility = phase2::compute_initial_primal_infeasibilities(
    lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);

#ifdef CHECK_BASIC_INFEASIBILITIES
  phase2::check_basic_infeasibilities(basic_list, basic_mark, infeasibility_indices, 0);
#endif

  csc_matrix_t<i_t, f_t> A_transpose(1, 1, 0);
  lp.A.transpose(A_transpose);

  f_t obj              = compute_objective(lp, x);
  const i_t start_iter = iter;

  i_t sparse_delta_z = 0;
  i_t dense_delta_z  = 0;
  phase2::phase2_timers_t<i_t, f_t> timers(false);

  while (iter < iter_limit) {
    // Pricing
    i_t direction           = 0;
    i_t basic_leaving_index = -1;
    i_t leaving_index       = -1;
    f_t max_val;
    timers.start_timer();
    if (settings.use_steepest_edge_pricing) {
      leaving_index = phase2::steepest_edge_pricing_with_infeasibilities(lp,
                                                                         settings,
                                                                         x,
                                                                         delta_y_steepest_edge,
                                                                         basic_mark,
                                                                         squared_infeasibilities,
                                                                         infeasibility_indices,
                                                                         direction,
                                                                         basic_leaving_index,
                                                                         max_val);
    } else {
      // Max infeasibility pricing
      leaving_index = phase2::phase2_pricing(
        lp, settings, x, basic_list, direction, basic_leaving_index, primal_infeasibility);
    }
    timers.pricing_time += timers.stop_timer();
    if (leaving_index == -1) {
      phase2::prepare_optimality(lp,
                                 settings,
                                 ft,
                                 objective,
                                 basic_list,
                                 nonbasic_list,
                                 vstatus,
                                 phase,
                                 start_time,
                                 max_val,
                                 iter,
                                 x,
                                 y,
                                 z,
                                 sol);
      status = dual::status_t::OPTIMAL;
      break;
    }

    // BTran
    // BT*delta_y = -delta_zB = -sigma*ei
    timers.start_timer();
    sparse_vector_t<i_t, f_t> delta_y_sparse(m, 0);
    sparse_vector_t<i_t, f_t> UTsol_sparse(m, 0);
    phase2::compute_delta_y(ft, basic_leaving_index, direction, delta_y_sparse, UTsol_sparse);
    timers.btran_time += timers.stop_timer();

    const f_t steepest_edge_norm_check = delta_y_sparse.norm2_squared();
    if (delta_y_steepest_edge[leaving_index] <
        settings.steepest_edge_ratio * steepest_edge_norm_check) {
      constexpr bool verbose = false;
      if constexpr (verbose) {
        settings.log.printf(
          "iteration restart due to steepest edge. Leaving %d. Actual %.2e "
          "from update %.2e\n",
          leaving_index,
          steepest_edge_norm_check,
          delta_y_steepest_edge[leaving_index]);
      }
      delta_y_steepest_edge[leaving_index] = steepest_edge_norm_check;
      continue;
    }

    timers.start_timer();
    i_t delta_y_nz0      = 0;
    const i_t nz_delta_y = delta_y_sparse.i.size();
    for (i_t k = 0; k < nz_delta_y; k++) {
      if (std::abs(delta_y_sparse.x[k]) > 1e-12) { delta_y_nz0++; }
    }
    const f_t delta_y_nz_percentage = delta_y_nz0 / static_cast<f_t>(m) * 100.0;
    const bool use_transpose        = delta_y_nz_percentage <= 30.0;
    if (use_transpose) {
      sparse_delta_z++;
      phase2::compute_delta_z(A_transpose,
                              delta_y_sparse,
                              leaving_index,
                              direction,
                              nonbasic_mark,
                              delta_z_mark,
                              delta_z_indices,
                              delta_z);
    } else {
      dense_delta_z++;
      // delta_zB = sigma*ei
      delta_y_sparse.to_dense(delta_y);
      phase2::compute_reduced_cost_update(lp,
                                          basic_list,
                                          nonbasic_list,
                                          delta_y,
                                          leaving_index,
                                          direction,
                                          delta_z_mark,
                                          delta_z_indices,
                                          delta_z);
    }
    timers.delta_z_time += timers.stop_timer();

#ifdef COMPUTE_DUAL_RESIDUAL
    std::vector<f_t> dual_residual;
    std::vector<f_t> zeros(n, 0.0);
    phase2::compute_dual_residual(lp.A, zeros, delta_y, delta_z, dual_residual);
    // || A'*delta_y + delta_z ||_inf
    f_t dual_residual_norm = vector_norm_inf<i_t, f_t>(dual_residual);
    settings.log.printf(
      "|| A'*dy - dz || %e use transpose %d\n", dual_residual_norm, use_transpose);
#endif

    // Ratio test
    f_t step_length;
    i_t entering_index          = -1;
    i_t nonbasic_entering_index = -1;
    const bool harris_ratio     = settings.use_harris_ratio;
    const bool bound_flip_ratio = settings.use_bound_flip_ratio;
    if (harris_ratio) {
      f_t max_step_length = phase2::first_stage_harris(lp, vstatus, nonbasic_list, z, delta_z);
      entering_index      = phase2::second_stage_harris(lp,
                                                   vstatus,
                                                   nonbasic_list,
                                                   z,
                                                   delta_z,
                                                   max_step_length,
                                                   step_length,
                                                   nonbasic_entering_index);
    } else if (bound_flip_ratio) {
      timers.start_timer();
      f_t slope = direction == 1 ? (lp.lower[leaving_index] - x[leaving_index])
                                 : (x[leaving_index] - lp.upper[leaving_index]);
      bound_flipping_ratio_test_t<i_t, f_t> bfrt(settings,
                                                 start_time,
                                                 m,
                                                 n,
                                                 slope,
                                                 lp.lower,
                                                 lp.upper,
                                                 bounded_variables,
                                                 vstatus,
                                                 nonbasic_list,
                                                 z,
                                                 delta_z,
                                                 delta_z_indices,
                                                 nonbasic_mark);
      entering_index = bfrt.compute_step_length(step_length, nonbasic_entering_index);
      if (entering_index == -4) {
        settings.log.printf("Numerical issues encountered in ratio test.\n");
        return dual::status_t::NUMERICAL;
      }
      timers.bfrt_time += timers.stop_timer();
    } else {
      entering_index = phase2::phase2_ratio_test(
        lp, settings, vstatus, nonbasic_list, z, delta_z, step_length, nonbasic_entering_index);
    }
    if (entering_index == -2) { return dual::status_t::TIME_LIMIT; }
    if (entering_index == -3) { return dual::status_t::CONCURRENT_LIMIT; }
    if (entering_index == -1) {
      settings.log.printf("No entering variable found. Iter %d\n", iter);
      settings.log.printf("Scaled infeasibility %e\n", max_val);
      f_t perturbation = phase2::amount_of_perturbation(lp, objective);

      if (perturbation > 0.0 && phase == 2) {
        // Try to remove perturbation
        std::vector<f_t> unperturbed_y(m);
        std::vector<f_t> unperturbed_z(n);
        phase2::compute_dual_solution_from_basis(
          lp, ft, basic_list, nonbasic_list, unperturbed_y, unperturbed_z);
        {
          const f_t dual_infeas = phase2::dual_infeasibility(
            lp, settings, vstatus, unperturbed_z, settings.tight_tol, settings.dual_tol);
          settings.log.printf("Dual infeasibility after removing perturbation %e\n", dual_infeas);
          if (dual_infeas <= settings.dual_tol) {
            settings.log.printf("Removed perturbation of %.2e.\n", perturbation);
            z            = unperturbed_z;
            y            = unperturbed_y;
            perturbation = 0.0;

            std::vector<f_t> unperturbed_x(n);
            phase2::compute_primal_solution_from_basis(
              lp, ft, basic_list, nonbasic_list, vstatus, unperturbed_x);
            x                    = unperturbed_x;
            primal_infeasibility = phase2::compute_initial_primal_infeasibilities(
              lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);
            settings.log.printf("Updated primal infeasibility: %e\n", primal_infeasibility);

            objective = lp.objective;
            // Need to reset the objective value, since we have recomputed x
            obj = phase2::compute_perturbed_objective(objective, x);
            if (dual_infeas <= settings.dual_tol && primal_infeasibility <= settings.primal_tol) {
              phase2::prepare_optimality(lp,
                                         settings,
                                         ft,
                                         objective,
                                         basic_list,
                                         nonbasic_list,
                                         vstatus,
                                         phase,
                                         start_time,
                                         max_val,
                                         iter,
                                         x,
                                         y,
                                         z,
                                         sol);
              status = dual::status_t::OPTIMAL;
              break;
            }
            settings.log.printf(
              "Continuing with perturbation removed and steepest edge norms reset\n");
            // Clear delta_z before restarting the iteration
            phase2::clear_delta_z(
              entering_index, leaving_index, delta_z_mark, delta_z_indices, delta_z);
            continue;
          } else {
            std::vector<f_t> unperturbed_x(n);
            phase2::compute_primal_solution_from_basis(
              lp, ft, basic_list, nonbasic_list, vstatus, unperturbed_x);
            x                    = unperturbed_x;
            primal_infeasibility = phase2::compute_initial_primal_infeasibilities(
              lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);

            const f_t orig_dual_infeas = phase2::dual_infeasibility(
              lp, settings, vstatus, z, settings.tight_tol, settings.dual_tol);

            if (primal_infeasibility <= settings.primal_tol &&
                orig_dual_infeas <= settings.dual_tol) {
              phase2::prepare_optimality(lp,
                                         settings,
                                         ft,
                                         objective,
                                         basic_list,
                                         nonbasic_list,
                                         vstatus,
                                         phase,
                                         start_time,
                                         max_val,
                                         iter,
                                         x,
                                         y,
                                         z,
                                         sol);
              status = dual::status_t::OPTIMAL;
              break;
            }
            settings.log.printf("Failed to remove perturbation of %.2e.\n", perturbation);
          }
        }
      }

      if (perturbation == 0.0 && phase == 2) {
        constexpr bool use_farkas = false;
        if constexpr (use_farkas) {
          std::vector<f_t> farkas_y;
          std::vector<f_t> farkas_zl;
          std::vector<f_t> farkas_zu;
          f_t farkas_constant;
          std::vector<f_t> my_delta_y;
          delta_y_sparse.to_dense(my_delta_y);

          // TODO(CMM): Do I use the perturbed or unperturbed objective?
          const f_t obj_val = phase2::compute_perturbed_objective(objective, x);
          phase2::compute_farkas_certificate(lp,
                                             settings,
                                             vstatus,
                                             x,
                                             y,
                                             z,
                                             my_delta_y,
                                             delta_z,
                                             direction,
                                             leaving_index,
                                             obj_val,
                                             farkas_y,
                                             farkas_zl,
                                             farkas_zu,
                                             farkas_constant);
        }
      }

      const f_t dual_infeas =
        phase2::dual_infeasibility(lp, settings, vstatus, z, settings.tight_tol, settings.dual_tol);
      settings.log.printf("Dual infeasibility %e\n", dual_infeas);
      const f_t primal_inf = phase2::primal_infeasibility(lp, settings, vstatus, x);
      settings.log.printf("Primal infeasibility %e\n", primal_inf);
      settings.log.printf("Updates %d\n", ft.num_updates());
      settings.log.printf("Steepest edge %e\n", max_val);
      if (dual_infeas > settings.dual_tol) {
        settings.log.printf(
          "Numerical issues encountered. No entering variable found with large infeasibility.\n");
        return dual::status_t::NUMERICAL;
      }
      return dual::status_t::DUAL_UNBOUNDED;
    }

    timers.start_timer();
    // Update dual variables
    // y <- y + steplength * delta_y
    // z <- z + steplength * delta_z
    i_t update_dual_variables_status = phase2::update_dual_variables(
      delta_y_sparse, delta_z_indices, delta_z, step_length, leaving_index, y, z);
    if (update_dual_variables_status == -1) {
      settings.log.printf("Numerical issues encountered in update_dual_variables.\n");
      return dual::status_t::NUMERICAL;
    }
    timers.vector_time += timers.stop_timer();

#ifdef COMPUTE_DUAL_RESIDUAL
    phase2::compute_dual_residual(lp.A, objective, y, z, dual_res1);
    f_t dual_res_norm = vector_norm_inf<i_t, f_t>(dual_res1);
    if (dual_res_norm > settings.dual_tol) {
      settings.log.printf("|| A'*y + z - c || %e steplength %e\n", dual_res_norm, step_length);
    }
#endif

    timers.start_timer();
    // Update primal variable
    const i_t num_flipped = phase2::flip_bounds(lp,
                                                settings,
                                                bounded_variables,
                                                objective,
                                                z,
                                                delta_z_indices,
                                                nonbasic_list,
                                                entering_index,
                                                vstatus,
                                                delta_x_flip,
                                                atilde_mark,
                                                atilde,
                                                atilde_index);

    timers.flip_time += timers.stop_timer();

    sparse_vector_t<i_t, f_t> delta_xB_0_sparse(m, 0);
    if (num_flipped > 0) {
      timers.start_timer();
      phase2::adjust_for_flips(ft,
                               basic_list,
                               delta_z_indices,
                               atilde_index,
                               atilde,
                               atilde_mark,
                               delta_xB_0_sparse,
                               delta_x_flip,
                               x);
      timers.ftran_time += timers.stop_timer();
    }

    timers.start_timer();
    sparse_vector_t<i_t, f_t> utilde_sparse(m, 0);
    sparse_vector_t<i_t, f_t> scaled_delta_xB_sparse(m, 0);
    sparse_vector_t<i_t, f_t> rhs_sparse(lp.A, entering_index);
    if (phase2::compute_delta_x(lp,
                                ft,
                                entering_index,
                                leaving_index,
                                basic_leaving_index,
                                direction,
                                basic_list,
                                delta_x_flip,
                                rhs_sparse,
                                x,
                                utilde_sparse,
                                scaled_delta_xB_sparse,
                                delta_x) == -1) {
      settings.log.printf("Failed to compute delta_x. Iter %d\n", iter);
      return dual::status_t::NUMERICAL;
    }

    timers.ftran_time += timers.stop_timer();

#ifdef CHECK_PRIMAL_STEP
    std::vector<f_t> residual(m);
    matrix_vector_multiply(lp.A, 1.0, delta_x, 1.0, residual);
    f_t primal_step_err = vector_norm_inf<i_t, f_t>(residual);
    if (primal_step_err > 1e-4) { settings.log.printf("|| A * dx || %e\n", primal_step_err); }
#endif

    timers.start_timer();
    const i_t steepest_edge_status = phase2::update_steepest_edge_norms(settings,
                                                                        basic_list,
                                                                        ft,
                                                                        direction,
                                                                        delta_y_sparse,
                                                                        steepest_edge_norm_check,
                                                                        scaled_delta_xB_sparse,
                                                                        basic_leaving_index,
                                                                        entering_index,
                                                                        v,
                                                                        delta_y_steepest_edge);
#ifdef STEEPEST_EDGE_DEBUG
    if (steepest_edge_status == -1) {
      settings.log.printf("Num updates %d\n", ft.num_updates());
      settings.log.printf("|| rhs || %e\n", vector_norm_inf(rhs));
    }
#endif
    assert(steepest_edge_status == 0);
    timers.se_norms_time += timers.stop_timer();

    timers.start_timer();
    // x <- x + delta_x
    phase2::update_primal_variables(scaled_delta_xB_sparse, basic_list, delta_x, entering_index, x);
    timers.vector_time += timers.stop_timer();

#ifdef COMPUTE_PRIMAL_RESIDUAL
    residual = lp.rhs;
    matrix_vector_multiply(lp.A, 1.0, x, -1.0, residual);
    primal_residual = vector_norm_inf<i_t, f_t>(residual);
    if (iter % 100 == 0 && primal_residual > 10 * settings.primal_tol) {
      settings.log.printf("|| A*x - b || %e\n", primal_residual);
    }
#endif

    timers.start_timer();
    // TODO(CMM): Do I also need to update the objective due to the bound flips?
    // TODO(CMM): I'm using the unperturbed objective here, should this be the perturbed objective?
    phase2::update_objective(
      basic_list, scaled_delta_xB_sparse.i, lp.objective, delta_x, entering_index, obj);
    timers.objective_time += timers.stop_timer();

    timers.start_timer();
    // Update primal infeasibilities due to changes in basic variables
    // from flipping bounds
#ifdef CHECK_BASIC_INFEASIBILITIES
    phase2::check_basic_infeasibilities(basic_list, basic_mark, infeasibility_indices, 2);
#endif
    phase2::update_primal_infeasibilities(lp,
                                          settings,
                                          basic_list,
                                          x,
                                          entering_index,
                                          leaving_index,
                                          delta_xB_0_sparse.i,
                                          squared_infeasibilities,
                                          infeasibility_indices,
                                          primal_infeasibility);
    // Update primal infeasibilities due to changes in basic variables
    // from the leaving and entering variables
    phase2::update_primal_infeasibilities(lp,
                                          settings,
                                          basic_list,
                                          x,
                                          entering_index,
                                          leaving_index,
                                          scaled_delta_xB_sparse.i,
                                          squared_infeasibilities,
                                          infeasibility_indices,
                                          primal_infeasibility);
    // Update the entering variable
    phase2::update_single_primal_infeasibility(lp.lower,
                                               lp.upper,
                                               x,
                                               settings.primal_tol,
                                               squared_infeasibilities,
                                               infeasibility_indices,
                                               entering_index,
                                               primal_infeasibility);

    phase2::clean_up_infeasibilities(squared_infeasibilities, infeasibility_indices);

#if CHECK_PRIMAL_INFEASIBILITIES
    phase2::check_primal_infeasibilities(
      lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);
#endif
    timers.update_infeasibility_time += timers.stop_timer();

    // Clear delta_x
    phase2::clear_delta_x(basic_list, entering_index, scaled_delta_xB_sparse, delta_x);

    timers.start_timer();
    f_t sum_perturb = 0.0;
    phase2::compute_perturbation(lp, settings, delta_z_indices, z, objective, sum_perturb);
    timers.perturb_time += timers.stop_timer();

    // Update basis information
    vstatus[entering_index] = variable_status_t::BASIC;
    if (lp.lower[leaving_index] != lp.upper[leaving_index]) {
      vstatus[leaving_index] = static_cast<variable_status_t>(-direction);
    } else {
      vstatus[leaving_index] = variable_status_t::NONBASIC_FIXED;
    }
    basic_list[basic_leaving_index]        = entering_index;
    nonbasic_list[nonbasic_entering_index] = leaving_index;
    nonbasic_mark[entering_index]          = -1;
    nonbasic_mark[leaving_index]           = nonbasic_entering_index;
    basic_mark[leaving_index]              = -1;
    basic_mark[entering_index]             = basic_leaving_index;

#ifdef CHECK_BASIC_INFEASIBILITIES
    phase2::check_basic_infeasibilities(basic_list, basic_mark, infeasibility_indices, 5);
#endif

    timers.start_timer();
    // Refactor or update the basis factorization
    bool should_refactor = ft.num_updates() > settings.refactor_frequency;
    if (!should_refactor) {
      i_t recommend_refactor = ft.update(utilde_sparse, UTsol_sparse, basic_leaving_index);
#ifdef CHECK_UPDATE
      phase2::check_update(lp, settings, ft, basic_list, basic_leaving_index);
#endif
      should_refactor = recommend_refactor == 1;
    }

#ifdef CHECK_BASIC_INFEASIBILITIES
    phase2::check_basic_infeasibilities(basic_list, basic_mark, infeasibility_indices, 6);
#endif
    if (should_refactor) {
      bool should_recompute_x = false;
      if (factorize_basis(lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed) ==
          -1) {
        should_recompute_x = true;
        settings.log.printf("Failed to factorize basis. Iteration %d\n", iter);
        if (toc(start_time) > settings.time_limit) { return dual::status_t::TIME_LIMIT; }
        basis_repair(lp.A, settings, deficient, slacks_needed, basic_list, nonbasic_list, vstatus);
        i_t count = 0;
        while (factorize_basis(
                 lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed) == -1) {
          settings.log.printf("Failed to repair basis. Iteration %d. %d deficient columns.\n",
                              iter,
                              static_cast<int>(deficient.size()));
#ifdef CHECK_L_FACTOR
          if (L.check_matrix() == -1) { settings.log.printf("Bad L after basis repair\n"); }
#endif
          if (toc(start_time) > settings.time_limit) { return dual::status_t::TIME_LIMIT; }
          settings.threshold_partial_pivoting_tol = 1.0;
          count++;
          if (count > 10) { return dual::status_t::NUMERICAL; }
          basis_repair(
            lp.A, settings, deficient, slacks_needed, basic_list, nonbasic_list, vstatus);

#ifdef CHECK_BASIS_REPAIR
          csc_matrix_t<i_t, f_t> B(m, m, 0);
          form_b(lp.A, basic_list, B);
          for (i_t k = 0; k < deficient.size(); ++k) {
            const i_t j         = deficient[k];
            const i_t col_start = B.col_start[j];
            const i_t col_end   = B.col_start[j + 1];
            const i_t col_nz    = col_end - col_start;
            if (col_nz != 1) {
              settings.log.printf("Deficient column %d has %d nonzeros\n", j, col_nz);
            }
            const i_t i = B.i[col_start];
            if (i != slacks_needed[k]) {
              settings.log.printf("Slack %d needed but found %d instead\n", slacks_needed[k], i);
            }
          }
#endif
        }

        settings.log.printf("Successfully repaired basis. Iteration %d\n", iter);
      }
      reorder_basic_list(q, basic_list);
#ifdef CHECK_L_FACTOR
      if (L.check_matrix() == -1) { settings.log.printf("Bad L factor\n"); }
#endif
      ft.reset(L, U, p);
      phase2::reset_basis_mark(basic_list, nonbasic_list, basic_mark, nonbasic_mark);
      if (should_recompute_x) {
        std::vector<f_t> unperturbed_x(n);
        phase2::compute_primal_solution_from_basis(
          lp, ft, basic_list, nonbasic_list, vstatus, unperturbed_x);
        x = unperturbed_x;
      }
      phase2::compute_initial_primal_infeasibilities(
        lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);
    }
#ifdef CHECK_BASIC_INFEASIBILITIES
    phase2::check_basic_infeasibilities(basic_list, basic_mark, infeasibility_indices, 7);
#endif
    timers.lu_update_time += timers.stop_timer();

    timers.start_timer();
    phase2::compute_steepest_edge_norm_entering(
      settings, m, ft, basic_leaving_index, entering_index, delta_y_steepest_edge);
    timers.se_entering_time += timers.stop_timer();

#ifdef STEEPEST_EDGE_DEBUG
    if (iter < 100 || iter % 100 == 0))
    {
      phase2::check_steepest_edge_norms(settings, basic_list, ft, delta_y_steepest_edge);
    }
#endif

#ifdef CHECK_BASIS_MARK
    phase2::check_basis_mark(settings, basic_list, nonbasic_list, basic_mark, nonbasic_mark);
#endif

    iter++;

    // Clear delta_z
    phase2::clear_delta_z(entering_index, leaving_index, delta_z_mark, delta_z_indices, delta_z);

    f_t now = toc(start_time);
    if ((iter - start_iter) < settings.first_iteration_log ||
        (iter % settings.iteration_log_frequency) == 0) {
      if (phase == 1 && iter == 1) {
        settings.log.printf(" Iter     Objective           Num Inf.  Sum Inf.     Perturb  Time\n");
      }
      settings.log.printf("%5d %+.16e %7d %.8e %.2e %.2f\n",
                          iter,
                          compute_user_objective(lp, obj),
                          infeasibility_indices.size(),
                          primal_infeasibility,
                          sum_perturb,
                          now);
    }

    if (obj >= settings.cut_off) {
      settings.log.printf("Solve cutoff. Current objecive %e. Cutoff %e\n", obj, settings.cut_off);
      return dual::status_t::CUTOFF;
    }

    if (now > settings.time_limit) { return dual::status_t::TIME_LIMIT; }

    if (settings.concurrent_halt != nullptr && *settings.concurrent_halt == 1) {
      return dual::status_t::CONCURRENT_LIMIT;
    }
  }
  if (iter >= iter_limit) { status = dual::status_t::ITERATION_LIMIT; }

  if (phase == 2) {
    timers.print_timers(settings);
    constexpr bool print_stats = false;
    if constexpr (print_stats) {
      settings.log.printf("Sparse delta_z %8d %8.2f%\n",
                          sparse_delta_z,
                          100.0 * sparse_delta_z / (sparse_delta_z + dense_delta_z));
      settings.log.printf("Dense delta_z  %8d %8.2f%\n",
                          dense_delta_z,
                          100.0 * dense_delta_z / (sparse_delta_z + dense_delta_z));
      ft.print_stats();
    }
  }
  return status;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template dual::status_t dual_phase2<int, double>(
  int phase,
  int slack_basis,
  double start_time,
  const lp_problem_t<int, double>& lp,
  const simplex_solver_settings_t<int, double>& settings,
  std::vector<variable_status_t>& vstatus,
  lp_solution_t<int, double>& sol,
  int& iter,
  std::vector<double>& steepest_edge_norms);

#endif

}  // namespace cuopt::linear_programming::dual_simplex
