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
#include <dual_simplex/solve.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/tic_toc.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <limits>
#include <list>

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
  for (i_t j = 0; j < n; ++j)
  {
    original_residual[j] -= lp.objective[j];
  }
  const f_t original_residual_norm = vector_norm2<i_t, f_t>(original_residual);
  settings.log.printf("|| A'*y + z - c || = %e\n", original_residual_norm);


  std::vector<f_t> zl(n);
  std::vector<f_t> zu(n);
  for (i_t j = 0; j < n; ++j)
  {
    zl[j] = std::max(0.0, z[j]);
    zu[j] = -std::min(0.0, z[j]);
  }

  original_residual = zl;
  matrix_transpose_vector_multiply(lp.A, 1.0, y, 1.0, original_residual);
  for (i_t j = 0; j < n; ++j)
  {
    original_residual[j] -= (zu[j] + lp.objective[j]);
  }
  const f_t original_residual_2 = vector_norm2<i_t, f_t>(original_residual);
  settings.log.printf("|| A'*y + zl - zu - c || = %e\n", original_residual_2);


  std::vector<f_t> search_dir_residual = delta_z;
  matrix_transpose_vector_multiply(lp.A, 1.0, delta_y, 1.0, search_dir_residual);
  settings.log.printf("|| A'*delta_y + delta_z || = %e\n", vector_norm2<i_t, f_t>(search_dir_residual));

  std::vector<f_t> y_bar(m);
  for (i_t i = 0; i < m; ++i)
  {
    y_bar[i] = y[i] + delta_y[i];
  }
  original_residual = z;
  matrix_transpose_vector_multiply(lp.A, 1.0, y_bar, 1.0, original_residual);
  for (i_t j = 0; j < n; ++j)
  {
    original_residual[j] += (delta_z[j] - lp.objective[j]);
  }
  const f_t original_residual_3 = vector_norm2<i_t, f_t>(original_residual);
  settings.log.printf("|| A'*(y + delta_y) + (z + delta_z) - c || = %e\n", original_residual_3);



  farkas_y.resize(m);
  farkas_zl.resize(n);
  farkas_zu.resize(n);

  f_t gamma = 0.0;
  for (i_t j = 0; j < n; ++j)
  {
    const f_t cj = lp.objective[j];
    const f_t lower = lp.lower[j];
    const f_t upper = lp.upper[j];
    if (lower > -inf)
    {
      gamma -= lower * std::min(0.0, cj);
    }
    if (upper < inf)
    {
      gamma -= upper * std::max(0.0, cj);
    }
  }
  printf("gamma = %e\n", gamma);

  const f_t threshold = 1.0;
  const f_t positive_threshold = std::max(-gamma, 0.0) + threshold;
  printf("positive_threshold = %e\n", positive_threshold);

  // We need to increase the dual objective to positive threshold
  f_t alpha = threshold;
  const f_t infeas = (direction == 1) ? (lp.lower[leaving_index] - x[leaving_index]) : (x[leaving_index] - lp.upper[leaving_index]);
  // We need the new objective to be at least positive_threshold
  // positive_threshold = obj_val+ alpha * infeas
  // infeas > 0, alpha > 0, positive_threshold > 0
  printf("direction = %d\n", direction);
  printf("lower %e x %e upper %d\n", lp.lower[leaving_index], x[leaving_index], lp.upper[leaving_index]);
  printf("infeas = %e\n", infeas);
  printf("obj_val = %e\n", obj_val);
  alpha = std::max(threshold,(positive_threshold - obj_val) / infeas);
  printf("alpha = %e\n", alpha);

  std::vector<f_t> y_prime(m);
  std::vector<f_t> zl_prime(n);
  std::vector<f_t> zu_prime(n);

  // farkas_y = y + alpha * delta_y
  for (i_t i = 0; i < m; ++i)
  {
    farkas_y[i] = y[i] + alpha * delta_y[i];
    y_prime[i] = y[i] + alpha * delta_y[i];
  }
  // farkas_zl = z + alpha * delta_z  - c-
  for (i_t j = 0; j < n; ++j)
  {
    const f_t cj = lp.objective[j];
    const f_t z_j = z[j];
    const f_t delta_z_j = delta_z[j];
    farkas_zl[j] = std::max(0.0, z_j) + alpha * std::max(0.0, delta_z_j) + -std::min(0.0, cj);
    zl_prime[j] = zl[j] + alpha * std::max(0.0, delta_z_j);
  }

  // farkas_zu = z + alpha * delta_z + c+
  for (i_t j = 0; j < n; ++j)
  {
    const f_t cj = lp.objective[j];
    const f_t z_j = z[j];
    const f_t delta_z_j = delta_z[j];
    farkas_zu[j] = -std::min(0.0, z_j) - alpha * std::min(0.0, delta_z_j) + std::max(0.0, cj);
    zu_prime[j] = zu[j] + alpha * (-std::min(0.0, delta_z_j));
  }

  // farkas_constant = b'*farkas_y + l'*farkas_zl - u'*farkas_zu
  farkas_constant = 0.0;
  f_t test_constant = 0.0;
  f_t test_3 = 0.0;
  for (i_t i = 0; i < m; ++i)
  {
    farkas_constant += lp.rhs[i] * farkas_y[i];
    test_constant += lp.rhs[i] * y_prime[i];
    test_3 += lp.rhs[i] * delta_y[i];
  }
  printf("b'*delta_y = %e\n", test_3);
  printf("|| b || %e\n", vector_norm_inf<i_t, f_t>(lp.rhs));
  printf("|| delta y || %e\n", vector_norm_inf<i_t, f_t>(delta_y));
  for (i_t j = 0; j < n; ++j)
  {
    const f_t lower = lp.lower[j];
    const f_t upper = lp.upper[j];
    if (lower > -inf)
    {
      farkas_constant += lower * farkas_zl[j];
      test_constant += lower * zl_prime[j];
      const f_t delta_z_l_j = std::max(delta_z[j], 0.0);
      test_3 += lower * delta_z_l_j;
    }
    if (upper < inf)
    {
      farkas_constant -= upper * farkas_zu[j];
      test_constant -= upper * zu_prime[j];
      const f_t delta_z_u_j = -std::min(delta_z[j], 0.0);
      test_3 -= upper * delta_z_u_j;
    }
  }


  // Verify that the Farkas certificate is valid
  std::vector<f_t> residual = farkas_zl;
  matrix_transpose_vector_multiply(lp.A, 1.0, farkas_y, 1.0, residual);
  for (i_t j = 0; j < n; ++j)
  {
    residual[j] -= farkas_zu[j];
  }
  const f_t residual_norm = vector_norm2<i_t, f_t>(residual);

  f_t zl_min = 0.0;
  for (i_t j = 0; j < n; ++j)
  {
    zl_min = std::min(zl_min, farkas_zl[j]);
  }
  settings.log.printf("farkas_zl_min = %e\n", zl_min);
  f_t zu_min = 0.0;
  for (i_t j = 0; j < n; ++j)
  {
    zu_min = std::min(zu_min, farkas_zu[j]);
  }
  settings.log.printf("farkas_zu_min = %e\n", zu_min);

  settings.log.printf("|| A'*farkas_y + farkas_zl - farkas_zu || = %e\n", residual_norm);
  settings.log.printf("b'*farkas_y + l'*farkas_zl - u'*farkas_zu = %e\n", farkas_constant);

  if (residual_norm < 1e-6 && farkas_constant > 0.0 && zl_min >= 0.0 && zu_min >= 0.0)
  {
    settings.log.printf("Farkas certificate of infeasibility constructed\n");
  }
}




template <typename i_t, typename f_t>
void initial_perturbation(const lp_problem_t<i_t, f_t>& lp,
                          const simplex_solver_settings_t<i_t, f_t>& settings,
                          const std::vector<variable_status_t>& vstatus,
                          std::vector<f_t>& objective)
{

  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;
  f_t max_abs_obj_coeff = 0.0;
  for (i_t j = 0; j < n; ++j)
  {
    max_abs_obj_coeff = std::max(max_abs_obj_coeff, std::abs(lp.objective[j]));
  }

  const f_t dual_tol = settings.dual_tol;

  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  objective.resize(n);
  f_t sum_perturb = 0.0;
  i_t num_perturb = 0;
  for (i_t j = 0; j < n; ++j)
  {
    f_t obj = objective[j] = lp.objective[j];

    const f_t lower = lp.lower[j];
    const f_t upper = lp.upper[j];
    if (vstatus[j] == variable_status_t::NONBASIC_FIXED ||
        vstatus[j] == variable_status_t::NONBASIC_FREE || lower == upper ||
        lower == -inf && upper == inf) {
      continue;
    }

    const f_t rand_val = static_cast<f_t>(std::rand() / (RAND_MAX + 1.0));
    const f_t perturb = (1e-5 * std::abs(obj) + 1e-7 * max_abs_obj_coeff + 10 * dual_tol) * (1.0 + rand_val);

    if (vstatus[j] == variable_status_t::NONBASIC_LOWER || lower > -inf && upper < inf && obj > 0)
    {
      objective[j] = obj + perturb;
      sum_perturb += perturb;
      num_perturb++;
    } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER || lower > -inf && upper < inf && obj < 0)
    {
      objective[j] = obj - perturb;
      sum_perturb += perturb;
      num_perturb++;
    }
  }

  settings.log.printf("Applied initial perturbation of %e to %d/%d objective coefficients\n", sum_perturb, num_perturb, n);
}



template <typename i_t, typename f_t>
bool use_transpose_for_delta_z(const lp_problem_t<i_t, f_t>& lp,
                          const csc_matrix_t<i_t, f_t>& A_transpose,
                          const sparse_vector_t<i_t, f_t>& delta_y,
                          const std::vector<i_t>& nonbasic_list)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;
  const i_t nz_delta_y = delta_y.i.size();
  f_t transpose_ops = 0;
  for (i_t k = 0; k < nz_delta_y; k++)
  {
    const i_t i = delta_y.i[k];
    const f_t delta_y_i = delta_y.x[k];
    if (std::abs(delta_y_i) < 1e-12) {
      continue;
    }
    const i_t row_start = A_transpose.col_start[i];
    const i_t row_end = A_transpose.col_start[i + 1];
    transpose_ops += row_end - row_start;
  }

  f_t regular_ops = 0;
   for (i_t k = 0; k < n - m; k++) {
    const i_t j = nonbasic_list[k];
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    regular_ops += col_end - col_start;
   }

  const bool use_transpose = transpose_ops < regular_ops;
  return use_transpose;
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
  for (i_t k = 0; k < nz_delta_y; k++)
  {
    const i_t i = delta_y.i[k];
    const f_t delta_y_i = delta_y.x[k];
    if (0 && std::abs(delta_y_i) < 1e-12) {
      continue;
    }
    const i_t row_start = A_transpose.col_start[i];
    const i_t row_end = A_transpose.col_start[i + 1];
    for (i_t p = row_start; p < row_end; ++p)
    {
      const i_t j = A_transpose.i[p];
      if (nonbasic_mark[j] >= 0)
      {
        delta_z[j] -= delta_y_i * A_transpose.x[p];
        if (!delta_z_mark[j])
        {
          delta_z_mark[j] = 1;
          delta_z_indices.push_back(j);
        }
      }
    }
  }

  // delta_zB = sigma*ei
  delta_z[leaving_index] = direction;
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
    if (dot != 0.0)
    {
      delta_z_indices.push_back(j);
      delta_z_mark[j] = 1;
    }
  }
}


template <typename i_t, typename f_t>
void clear_delta_z(i_t entering_index,
                   i_t leaving_index,
                   std::vector<i_t>& delta_z_mark,
                   std::vector<i_t>& delta_z_indices,
                   std::vector<f_t>& delta_z)
{
  for (i_t k = 0; k < delta_z_indices.size(); k++)
  {
    const i_t j = delta_z_indices[k];
    delta_z[j] = 0.0;
    delta_z_mark[j] = 0;
  }
  if (entering_index != -1) { delta_z[entering_index] = 0.0; }
  delta_z[leaving_index] = 0.0;
  delta_z_indices.clear();
}



template <typename i_t, typename f_t>
f_t l2_dual_residual(const lp_problem_t<i_t, f_t>& lp, const lp_solution_t<i_t, f_t>& solution)
{
  std::vector<f_t> dual_residual = solution.z;
  const i_t n                    = lp.num_cols;
  // dual_residual <- z - c
  for (i_t j = 0; j < n; j++) {
    dual_residual[j] -= lp.objective[j];
  }
  // dual_residual <- 1.0*A'*y + 1.0*(z - c)
  matrix_transpose_vector_multiply(lp.A, 1.0, solution.y, 1.0, dual_residual);
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
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;
  std::vector<f_t> rhs = lp.rhs;

  for (i_t k = 0; k < n - m; ++k)
  {
    const i_t j = nonbasic_list[k];
    if (vstatus[j] == variable_status_t::NONBASIC_LOWER || vstatus[j] == variable_status_t::NONBASIC_FIXED)
    {
      x[j] = lp.lower[j];
    }
    else if (vstatus[j] == variable_status_t::NONBASIC_UPPER)
    {
      x[j] = lp.upper[j];
    }
    else if (vstatus[j] == variable_status_t::NONBASIC_FREE)
    {
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
    const i_t j = basic_list[k];
    const f_t lower_infeas = lp.lower[j] - x[j];
    const f_t upper_infeas = x[j] - lp.upper[j];
    const f_t infeas = std::max(lower_infeas, upper_infeas);
    if (infeas > settings.primal_tol) {
      const f_t square_infeas = infeas * infeas;
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
  const f_t now_feasible = std::numeric_limits<f_t>::denorm_min();
  const f_t old_val = squared_infeasibilities[j];
  // x_j < l_j - epsilon => -x_j + l_j > epsilon
  const f_t lower_infeas = lower[j] - x[j];
  // x_j > u_j + epsilon => x_j - u_j > epsilon
  const f_t upper_infeas = x[j] - upper[j];
  const f_t infeas = std::max(lower_infeas, upper_infeas);
  const f_t new_val = infeas * infeas;
  if (infeas > primal_tol) {
    primal_inf = std::max(0.0, primal_inf + (new_val - old_val));
    // We are infeasible w.r.t the tolerance
    if (old_val == 0.0) {
      //printf("New infeasibility %d %e\n", j, infeas);
      // This is a new infeasibility
      // We need to add it to the list
      infeasibility_indices.push_back(j);
    } else {
      //printf("Already infeasible %d %e\n", j, infeas);
    }
    squared_infeasibilities[j] = new_val;
  } else {
    // We are feasible w.r.t the tolerance
    if (old_val != 0.0) {
      // We were previously infeasible,
      primal_inf = std::max(0.0, primal_inf - old_val);
      //printf("Now feasible %d %e\n", j, infeas);
      squared_infeasibilities[j] = now_feasible;
    } else {
      //printf("Still feasible %d %e\n", j, infeas);
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
  const f_t now_feasible = std::numeric_limits<f_t>::denorm_min();
  const f_t primal_tol = settings.primal_tol;
  const i_t nz = basic_change_list.size();
  for (i_t k = 0; k < nz; ++k) {
    const i_t j = basic_list[basic_change_list[k]];
    // The change list will contain the leaving variable,
    // But not the entering variable.

    if (j == leaving_index) {
      // Force the leaving variable to be feasible
      const f_t old_val = squared_infeasibilities[j];
      squared_infeasibilities[j] = now_feasible;
      primal_inf = std::max(0.0, primal_inf - old_val);
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

  // Update the entering variable
  update_single_primal_infeasibility(lp.lower,
                                     lp.upper,
                                     x,
                                     primal_tol,
                                     squared_infeasibilities,
                                     infeasibility_indices,
                                     entering_index,
                                     primal_inf);
}

template <typename i_t, typename f_t>
void clean_up_infeasibilities(std::vector<f_t>& squared_infeasibilities,
                              std::vector<i_t>& infeasibility_indices)
{
  const f_t now_feasible = std::numeric_limits<f_t>::denorm_min();
  bool needs_clean_up = false;
  for (i_t k = 0; k < infeasibility_indices.size(); ++k) {
    const i_t j = infeasibility_indices[k];
    const f_t squared_infeas = squared_infeasibilities[j];
    if (squared_infeas == now_feasible) {
      needs_clean_up = true;
    }
  }

   if (needs_clean_up) {
    for (i_t k = 0; k < infeasibility_indices.size(); ++k) {
      const i_t j = infeasibility_indices[k];
      const f_t squared_infeas = squared_infeasibilities[j];
      if (squared_infeas == now_feasible) {
        // Set to the last element
        const i_t sz = infeasibility_indices.size();
        infeasibility_indices[k] = infeasibility_indices[sz - 1];
        infeasibility_indices.pop_back();
        squared_infeasibilities[j] = 0.0;
        i_t new_j = infeasibility_indices[k];
        if (squared_infeasibilities[new_j] == now_feasible) {
          k--;
        }
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
  const f_t now_feasible = std::numeric_limits<f_t>::denorm_min();
  max_val = 0.0;
  i_t leaving_index = -1;
  bool needs_clean_up = false;
  const i_t nz = infeasibility_indices.size();
  for (i_t k = 0; k < nz; ++k) {
    const i_t j = infeasibility_indices[k];
    const f_t squared_infeas = squared_infeasibilities[j];
#if 0
    if (squared_infeas == now_feasible)
    {
      needs_clean_up = true;
      continue;
    }
#endif
    const f_t val = squared_infeas / dy_steepest_edge[j];
    if (val > max_val || val == max_val && j > leaving_index) {
      max_val = val;
      leaving_index = j;
      const f_t lower_infeas = lp.lower[j] - x[j];
      const f_t upper_infeas = x[j] - lp.upper[j];
      direction = lower_infeas >= upper_infeas ? 1 : -1;
    }
  }
#if 0
  if (needs_clean_up) {
    for (i_t k = 0; k < infeasibility_indices.size(); ++k) {
      const i_t j = infeasibility_indices[k];
      const f_t squared_infeas = squared_infeasibilities[j];
      if (squared_infeas == now_feasible) {
        // Set to the last element
        const i_t sz = infeasibility_indices.size();
        infeasibility_indices[k] = infeasibility_indices[sz - 1];
        infeasibility_indices.pop_back();
        squared_infeasibilities[j] = 0.0;
      }
    }
  }
#endif

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
i_t bound_flipping_ratio_test(const lp_problem_t<i_t, f_t>& lp,
                              const simplex_solver_settings_t<i_t, f_t>& settings,
                              f_t start_time,
                              const std::vector<variable_status_t>& vstatus,
                              const std::vector<i_t>& nonbasic_list,
                              const std::vector<f_t>& x,
                              std::vector<f_t>& z,
                              std::vector<f_t>& delta_z,
                              i_t direction,
                              i_t leaving_index,
                              f_t& step_length,
                              i_t& nonbasic_entering)
{
  const i_t n = lp.num_cols;
  const i_t m = lp.num_rows;

  f_t slope = direction == 1 ? (lp.lower[leaving_index] - x[leaving_index])
                             : (x[leaving_index] - lp.upper[leaving_index]);
  assert(slope > 0);

  const f_t pivot_tol         = settings.pivot_tol;
  const f_t relaxed_pivot_tol = settings.pivot_tol;
  const f_t zero_tol          = settings.zero_tol;
  std::list<i_t> q_pos;
  assert(nonbasic_list.size() == n - m);
  for (i_t k = 0; k < n - m; ++k) {
    const i_t j = nonbasic_list[k];
    if (vstatus[j] == variable_status_t::NONBASIC_FIXED) { continue; }
    if (vstatus[j] == variable_status_t::NONBASIC_LOWER && delta_z[j] < -pivot_tol) {
      q_pos.push_back(k);
    } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER && delta_z[j] > pivot_tol) {
      q_pos.push_back(k);
    }
  }
  i_t entering_index = -1;
  step_length        = inf;
  const f_t dual_tol = settings.dual_tol / 10;
  while (q_pos.size() > 0 && slope > 0) {
    // Find the minimum ratio for nonbasic variables in q_pos
    f_t min_val = inf;
    typename std::list<i_t>::iterator q_index;
    i_t candidate = -1;
    for (typename std::list<i_t>::iterator it = q_pos.begin(); it != q_pos.end(); ++it) {
      const i_t k = *it;
      const i_t j = nonbasic_list[k];
      f_t ratio   = inf;
      if (vstatus[j] == variable_status_t::NONBASIC_LOWER && delta_z[j] < -pivot_tol) {
        ratio = (-dual_tol - z[j]) / delta_z[j];
      } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER && delta_z[j] > pivot_tol) {
        ratio = (dual_tol - z[j]) / delta_z[j];
      } else if (min_val != inf) {
        // We've already found something just continue;
      } else if (vstatus[j] == variable_status_t::NONBASIC_LOWER) {
        ratio = (-dual_tol - z[j]) / delta_z[j];
      } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER) {
        ratio = (dual_tol - z[j]) / delta_z[j];
      } else {
        assert(1 == 0);
      }

      ratio = std::max(ratio, 0.0);

      if (ratio < min_val) {
        min_val = ratio;
        q_index = it;  // Save the iterator so we can remove the element it
                       // points to from the q_pos list later (if it corresponds
                       // to a bounded variable)
        candidate = j;
      } else if (ratio < min_val + zero_tol &&
                 std::abs(delta_z[j]) > std::abs(delta_z[candidate])) {
        min_val   = ratio;
        q_index   = it;
        candidate = j;
      }
    }
    step_length       = min_val;  // Save the step length
    nonbasic_entering = *q_index;
    const i_t j = entering_index = nonbasic_list[nonbasic_entering];
    if (lp.lower[j] > -inf && lp.upper[j] < inf && lp.lower[j] != lp.upper[j]) {
      const f_t interval    = lp.upper[j] - lp.lower[j];
      const f_t delta_slope = std::abs(delta_z[j]) * interval;
#ifdef BOUND_FLIP_DEBUG
      if (slope - delta_slope > 0) {
        settings.log.printf(
          "Bound flip %d slope change %e prev slope %e slope %e. curr step "
          "length %e\n",
          j,
          delta_slope,
          slope,
          slope - delta_slope,
          step_length);
      }
#endif
      slope -= delta_slope;
      q_pos.erase(q_index);
    } else {
      // we hit a variable that is not bounded. Exit
      break;
    }

    if (toc(start_time) > settings.time_limit) { return -2; }
    if (settings.concurrent_halt != nullptr &&
        settings.concurrent_halt->load(std::memory_order_acquire) == 1) {
      return -3;
    }
  }
  // step_length, nonbasic_entering, and entering_index are defined after the
  // while loop
  assert(step_length >= 0);

  return entering_index;
}

template <typename i_t, typename f_t>
i_t flip_bounds(const lp_problem_t<i_t, f_t>& lp,
                const simplex_solver_settings_t<i_t, f_t>& settings,
                const std::vector<bool>& bounded_variables,
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
  //f_t delta_obj = 0;
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
      //delta_obj += delta * objective[j];
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
      //delta_obj += delta * objective[j];
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
i_t initialize_steepest_edge_norms(const lp_problem_t<i_t, f_t>& lp,
                                   const simplex_solver_settings_t<i_t, f_t>& settings,
                                   const f_t start_time,
                                   const std::vector<i_t>& basic_list,
                                   basis_update_mpf_t<i_t, f_t>& ft,
                                   std::vector<f_t>& delta_y_steepest_edge)
{
  // TODO: Skip this initialization when starting from a slack basis
  //       Or skip individual columns corresponding to slack variables

  const i_t m  = basic_list.size();

  // We want to compute B^T delta_y_i = -e_i
  // If there is a column u of B^T such that B^T(:, u) = alpha * e_i than the
  // solve delta_y_i = -1/alpha * e_u
  // So we need to find columns of B^T (or rows of B) with only a single non-zero entry
  f_t start_singleton_rows = tic();
  std::vector<i_t> row_degree(m, 0);
  std::vector<i_t> mapping(m, -1);
  std::vector<f_t> coeff(m, 0.0);
#if 1
  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end = lp.A.col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      const i_t i = lp.A.i[p];
      row_degree[i]++;
      // column j of A is column k of B
      mapping[k] = i;
      coeff[k] = lp.A.x[p];
    }
  }

  csc_matrix_t<i_t, f_t> B(m, m, 0);
  form_b(lp.A, basic_list, B);
  csc_matrix_t<i_t, f_t> B_transpose(m, m, 0);
  B.transpose(B_transpose);

  i_t num_singleton_rows = 0;
  for (i_t i = 0; i < m; ++i) {
    if (row_degree[i] == 1) {
      num_singleton_rows++;
      const i_t col_start = B_transpose.col_start[i];
      const i_t col_end = B_transpose.col_start[i + 1];
      if (col_end - col_start != 1) {
        settings.log.printf("Singleton row %d has %d non-zero entries\n", i, col_end - col_start);
      }
    }
  }



  settings.log.printf("Found %d singleton rows in %.2fs\n", num_singleton_rows, toc(start_singleton_rows));

  //ft.compute_transposes();
#endif

  f_t last_log = tic();
  for (i_t k = 0; k < m; ++k) {
#if 0
    std::vector<f_t> ei(m);
    ei[k]       = -1.0;
    std::vector<f_t> dy(m, 0.0);
#else
    sparse_vector_t<i_t, f_t> sparse_ei(m, 1);
    sparse_ei.x[0] = -1.0;
    sparse_ei.i[0] = k;
#endif
    const i_t j = basic_list[k];
    f_t init = -1.0;
    if (row_degree[mapping[k]] == 1) {
      const i_t u = mapping[k];
      //settings.log.printf("Singleton row %d u %d\n", k, u);
      const f_t alpha = coeff[k];
      //dy[u] = -1.0 / alpha;
      f_t my_init = 1.0 / (alpha * alpha);
      init = my_init;
#ifdef CHECK_HYPERSPARSE
      std::vector<f_t> residual(m);
      b_transpose_multiply(lp, basic_list, dy, residual);
      float error = 0;
      for (i_t h = 0; h < m; ++h) {
        const f_t error_component = std::abs(residual[h] - ei[h]);
        error += error_component;
        if (error_component > 1e-12) {
          settings.log.printf("Singleton row %d component %d error %e residual %e ei %e\n", k, h, error_component, residual[h], ei[h]);
        }
      }
      if (error > 1e-12) {
        settings.log.printf("Singleton row %d error %e\n", k, error);
      }
#endif

#ifdef CHECK_HYPERSPARSE
      dy[u] = 0.0;
      ft.b_transpose_solve(ei, dy);
      init = vector_norm2_squared<i_t, f_t>(dy);
      if (init != my_init) {
        settings.log.printf("Singleton row %d error %.16e init %.16e my_init %.16e\n", k, std::abs(init - my_init), init, my_init);
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
        settings.log.printf("Singleton row %d error %.16e init %.16e my_init %.16e\n", k, std::abs(init - my_init), init, my_init);
      }
#endif
      init = my_init;
    }
    //ei[k]          = 0.0;
    //init = vector_norm2_squared<i_t, f_t>(dy);
    assert(init > 0);
    delta_y_steepest_edge[j] = init;

    f_t now            = toc(start_time);
    f_t time_since_log = toc(last_log);
    if (time_since_log > 10) {
      last_log = tic();
      settings.log.printf("Initialized %d of %d steepest edge norms in %.2fs\n", k, m, now);
    }
    if (toc(start_time) > settings.time_limit) { return -1; }
    if (settings.concurrent_halt != nullptr &&
        settings.concurrent_halt->load(std::memory_order_acquire) == 1) {
      return -1;
    }
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
  i_t m = basic_list.size();

  //sparse_vector_t<i_t, f_t> delta_y_sparse(delta_y);
  const i_t delta_y_nz = delta_y_sparse.i.size();
  sparse_vector_t<i_t, f_t> v_sparse(m, 0);

  if (0)
  {
    // B^T delta_y = - direction * e_basic_leaving_index
    // We want B v =  - B^{-T} e_basic_leaving_index
    std::vector<f_t> delta_y;
    delta_y_sparse.to_dense(delta_y);
    ft.b_solve(delta_y, v);
    // if direction = -1 we need to scale v
    if (direction == -1) {
      for (i_t k = 0; k < m; ++k) {
        v[k] *= -1;
      }
    }
  }
  else
  {
    ft.b_solve(delta_y_sparse, v_sparse);
    if (direction == -1) {
      for (i_t k = 0; k < v_sparse.i.size(); ++k) {
        v_sparse.x[k] *= -1;
      }
    }
    v_sparse.scatter(v);
  }

  //const f_t dy_norm_squared      = delta_y_sparse.norm2_squared();
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
  f_t scale;
  const i_t scaled_delta_xB_nz = scaled_delta_xB.i.size();
  for (i_t h = 0; h < scaled_delta_xB_nz; ++h) {
    const i_t k = scaled_delta_xB.i[h];
    if (k == basic_leaving_index) {
      scale = scaled_delta_xB.x[h];
      break;
    }
  }
  const f_t wr = -scale;
  //const f_t wr = -scaled_delta_xB.x[basic_leaving_index];
  if (wr == 0) { return -1; }
  const f_t omegar = dy_norm_squared / (wr * wr);
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
                                        f_t b_transpose_density,
                                        std::vector<f_t>& steepest_edge_norms)
{
  if (0) {
    std::vector<f_t> es(m);
    es[basic_leaving_index] = -1.0;
    std::vector<f_t> delta_ys(m);
    ft.b_transpose_solve(es, delta_ys);
    steepest_edge_norms[entering_index] = vector_norm2_squared<i_t, f_t>(delta_ys);
  } else {
    sparse_vector_t<i_t, f_t> es_sparse(m, 1);
    es_sparse.i[0] = basic_leaving_index;
    es_sparse.x[0] = -1.0;
    sparse_vector_t<i_t, f_t> delta_ys_sparse(m, 0);
    ft.b_transpose_solve(es_sparse, delta_ys_sparse);
    steepest_edge_norms[entering_index] = delta_ys_sparse.norm2_squared();
  }

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
  //for (i_t j = 0; j < n; ++j) {
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
  f_t perturbation   = 0.0;
  for (i_t j = 0; j < n; ++j) {
    perturbation += std::abs(lp.objective[j] - objective[j]);
  }

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
      }
      else {
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
      settings.log.printf("Max steepest edge norm:     %.2e\n", max_val);
    } else {
      settings.log.printf("\n");
      settings.log.printf(
        "Root relaxation solution found in %d iterations and %.2fs\n", iter, toc(start_time));
      settings.log.printf("Root relaxation objective %+.8e\n", sol.user_objective);
      settings.log.printf("\n");
    }
  }
}

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
  if (print_norms) {
    settings.log.printf(
      "|| y || %e || cB || %e\n", vector_norm_inf<i_t, f_t>(y), vector_norm_inf<i_t, f_t>(c_basic));
  }

  // zN = cN - N'*y
  for (i_t k = 0; k < n - m; k++) {
    const i_t j = nonbasic_list[k];
    // z_j <- c_j
    z[j] = objective[j];

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
  if (print_norms) { settings.log.printf("|| z || %e\n", vector_norm_inf<i_t, f_t>(z)); }

#ifdef COMPUTE_DUAL_RESIDUAL
  // || A'*y + z  - c||_inf
  std::vector<f_t> dual_res1 = z;
  for (i_t j = 0; j < n; ++j) {
    dual_res1[j] -= objective[j];
  }
  matrix_transpose_vector_multiply(lp.A, 1.0, y, 1.0, dual_res1);
  f_t dual_res_norm = vector_norm_inf<i_t, f_t>(dual_res1);
  if (1 || dual_res_norm > settings.tight_tol) {
    settings.log.printf("|| A'*y + z - c || %e\n", dual_res_norm);
  }
  assert(dual_res_norm < 1e-3);
#endif

  phase2::set_primal_variables_on_bounds(lp, settings, z, vstatus, x);

#ifdef PRINT_VSTATUS_CHANGES
  i_t num_vstatus_changes = 0;
  i_t num_z_changes       = 0;
  for (i_t j = 0; j < n; ++j) {
    if (vstatus[j] != vstatus_old[j]) { num_vstatus_changes++; }
    if (std::abs(z[j] - z_old[j]) > 1e-6) { num_z_changes++; }
  }

  printf("Number of vstatus changes %d\n", num_vstatus_changes);
  printf("Number of z changes %d\n", num_z_changes);
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

  std::vector<f_t> rhs = lp.rhs;
  // rhs = b - sum_{j : x_j = l_j} A(:, j) l(j) - sum_{j : x_j = u_j} A(:, j) *
  // u(j)
  for (i_t k = 0; k < n - m; ++k) {
    const i_t j         = nonbasic_list[k];
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    const f_t xj        = x[j];
    if (std::abs(xj) < settings.tight_tol * 10) continue;
    for (i_t p = col_start; p < col_end; ++p) {
      rhs[lp.A.i[p]] -= xj * lp.A.x[p];
    }
  }

  std::vector<f_t> xB(m);
  ft.b_solve(rhs, xB);
  if (toc(start_time) > settings.time_limit) { return dual::status_t::TIME_LIMIT; }

  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    x[j]        = xB[k];
  }
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
      for (i_t k = 0; k < m; ++k) {
        const i_t j              = basic_list[k];
        delta_y_steepest_edge[j] = 1.0;
      }
      for (i_t k = 0; k < n - m; ++k) {
        const i_t j              = nonbasic_list[k];
        delta_y_steepest_edge[j] = 1e-4;
      }
    } else {
      std::fill(delta_y_steepest_edge.begin(), delta_y_steepest_edge.end(), -1);
      if (phase2::initialize_steepest_edge_norms(lp,
            settings, start_time, basic_list, ft, delta_y_steepest_edge) == -1) {
        return dual::status_t::TIME_LIMIT;
      }
    }
  } else {
    settings.log.printf("using exisiting steepest edge %e\n",
                        vector_norm2<i_t, f_t>(delta_y_steepest_edge));
  }

  if (phase == 2) { settings.log.printf(" Iter     Objective   Primal Infeas  Perturb  Time\n"); }

  const i_t iter_limit = settings.iteration_limit;
  std::vector<f_t> delta_y(m, 0.0);
  std::vector<f_t> delta_z(n, 0.0);
  std::vector<f_t> delta_x(n, 0.0);
  std::vector<f_t> delta_x_flip(n, 0.0);
  std::vector<f_t> atilde(m, 0.0);
  std::vector<i_t> atilde_mark(m, 0);
  std::vector<i_t> atilde_index;
  std::vector<i_t> nonbasic_mark(n, -1);
  std::vector<i_t> basic_mark(n, -1);
  std::vector<i_t> delta_z_mark(n, 0);
  std::vector<i_t> delta_z_indices;
  std::vector<f_t> v(m, 0.0);
  std::vector<f_t> squared_infeasibilities;
  std::vector<i_t> infeasibility_indices;

  for (i_t k = 0; k < n - m; k++) {
    nonbasic_mark[nonbasic_list[k]] = k;
  }

  for (i_t k = 0; k < m; k++) {
    basic_mark[basic_list[k]] = k;
  }

  std::vector<bool> bounded_variables(n, false);
  for (i_t j = 0; j < n; j++) {
    const bool bounded =
      (lp.lower[j] > -inf) && (lp.upper[j] < inf) && (lp.lower[j] != lp.upper[j]);
    bounded_variables[j] = bounded;
  }

  f_t primal_infeasibility = phase2::compute_initial_primal_infeasibilities(
    lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);


  csc_matrix_t<i_t, f_t> A_transpose(1, 1, 0);
  lp.A.transpose(A_transpose);


  f_t obj = compute_objective(lp, x);
  settings.log.printf("Initial objective %e\n", obj);

  const i_t start_iter = iter;

  f_t b_transpose_solve_density = 0.0;
  f_t b_solve_density = 0.0;

  i_t sparse_delta_z = 0;
  i_t dense_delta_z = 0;

  f_t bfrt_time        = 0;
  f_t pricing_time     = 0;
  f_t btran_time       = 0;
  f_t ftran_time       = 0;
  f_t flip_time        = 0;
  f_t delta_z_time     = 0;
  f_t se_norms_time    = 0;
  f_t se_entering_time = 0;
  f_t lu_update_time = 0;
  f_t perturb_time   = 0;
  f_t vector_time    = 0;
  f_t objective_time = 0;
  f_t update_infeasibility_time = 0;
  bool restart_steepest_edge = true;

  while (iter < iter_limit) {
    // Pricing
    i_t direction = 0;
    i_t basic_leaving_index = -1;
    i_t leaving_index = -1;
    f_t max_val;
    f_t price_start_time = tic();
    if (settings.use_steepest_edge_pricing) {
#if 0
      i_t direction_junk = 0;
      i_t leaving_index_junk = -1;
      f_t max_val_junk = 0.0;
      f_t primal_inf_junk = 0;
      i_t basic_leaving_index_junk = -1;
      leaving_index_junk = phase2::steepest_edge_pricing(lp,
                                                    settings,
                                                    x,
                                                    delta_y_steepest_edge,
                                                    basic_list,
                                                    direction_junk,
                                                    basic_leaving_index_junk,
                                                    primal_inf_junk,
                                                    max_val_junk);
#else
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
#endif
#if 0
      if (leaving_index != leaving_index_junk || basic_leaving_index != basic_leaving_index_junk || max_val != max_val_junk || direction != direction_junk) {
        printf("Leaving index %d %d Basic leaving index %d %d max_val %e %e\n", leaving_index, leaving_index_junk, basic_leaving_index, basic_leaving_index_junk, max_val, max_val_junk);
        printf("Direction %d %d\n", direction, direction_junk);

        if (leaving_index >= 0 && leaving_index_junk >= 0) {
          printf("Squared infeasibilities %d %e %d %e\n", leaving_index, squared_infeasibilities[leaving_index] / delta_y_steepest_edge[leaving_index], leaving_index_junk, squared_infeasibilities[leaving_index_junk] / delta_y_steepest_edge[leaving_index_junk]);
        }
        else
        {
          printf("Trying to print bad stuff\n");
        }
      }
     // printf("Leaving index %d\n", leaving_index);
#endif
    } else {
      // Max infeasibility pricing
      leaving_index = phase2::phase2_pricing(
        lp, settings, x, basic_list, direction, basic_leaving_index, primal_infeasibility);
    }
    pricing_time += toc(price_start_time);
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
    f_t btran_start_time = tic();
    sparse_vector_t<i_t, f_t> delta_y_sparse(m, 0);
    sparse_vector_t<i_t, f_t> UTsol_sparse(m, 0);
    if (0) {
      std::vector<f_t> ei(m, 0.0);
      ei[basic_leaving_index] = -direction;

      std::vector<f_t> UTsol;
      ft.b_transpose_solve(ei, delta_y, UTsol);

      if (ei[basic_leaving_index] != 1.0) {
        // Need to flip the sign of UTsol
        for (i_t k = 0; k < m; ++k) {
          UTsol[k] *= -1.0;
        }
      }
      sparse_vector_t<i_t, f_t> dy_sparse(delta_y);
      sparse_vector_t<i_t, f_t> UT_sparse(UTsol);
      delta_y_sparse = dy_sparse;
      UTsol_sparse = UT_sparse;
      b_transpose_solve_density = delta_y_sparse.i.size() / static_cast<f_t>(m);
    } else {
      sparse_vector_t<i_t, f_t> ei_sparse(m, 1);
      ei_sparse.i[0] = basic_leaving_index;
      ei_sparse.x[0] = -direction;
      ft.b_transpose_solve(ei_sparse, delta_y_sparse, UTsol_sparse);

      if (direction != -1) {
        // We solved BT*delta_y = -sigma*ei, but for the update we need
        // UT*etilde = ei. So we need to flip the sign of the solution
        // in the case that sigma == 1.
        for (i_t k = 0; k < UTsol_sparse.x.size(); ++k) {
          UTsol_sparse.x[k] *= -1.0;
        }
      }
    }

#if 0
    std::vector<f_t> delta_y_sparse_vector_check(m);
    delta_y_sparse.to_dense(delta_y_sparse_vector_check);
    f_t error_check = 0.0;
    for (i_t k = 0; k < m; ++k)
    {
      if (std::abs(delta_y[k] - delta_y_sparse_vector_check[k]) > 1e-6)
      {
        settings.log.printf("\tBTranspose error %d %e %e\n", k, delta_y[k], delta_y_sparse_vector_check[k]);
      }
      error_check += std::abs(delta_y[k] - delta_y_sparse_vector_check[k]);
    }
    if (error_check > 1e-6) {
      settings.log.printf("BTranspose error %e\n", error_check);
    }
    std::vector<f_t> residual(m);
    b_transpose_multiply(lp, basic_list, delta_y_sparse_vector_check, residual);
    for (i_t k = 0; k < m; ++k)
    {
      if (std::abs(residual[k] - ei[k]) > 1e-6)
      {
        settings.log.printf("\tBTranspose multiply error %d %e %e\n", k, residual[k], ei[k]);
      }
    }
#endif
#if 1
    const f_t steepest_edge_norm_check = delta_y_sparse.norm2_squared();
#else
    f_t steepest_edge_norm_check = vector_norm2_squared<i_t, f_t>(delta_y);
#endif
    if (restart_steepest_edge && delta_y_steepest_edge[leaving_index] <
        settings.steepest_edge_ratio * steepest_edge_norm_check) {
      constexpr bool verbose = false;
      if (verbose) {
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

    btran_time += toc(btran_start_time);

#ifdef COMPUTE_BTRANSPOSE_RESIDUAL
    {
      std::vector<f_t> res(m);
      b_transpose_multiply(lp, basic_list, delta_y, res);
      f_t max_err = 0.0;
      for (i_t k = 0; k < m; k++) {
        const f_t err = std::abs(res[k] - ei[k]);
        if (err > 1e-4) { settings.log.printf("BT err %d %e\n", k, err); }
        max_err = std::max(max_err, err);
      }
      printf("BTranspose multiply error %e\n", max_err);
    }
#endif

    f_t delta_z_start_time = tic();

    i_t delta_y_nz0 = 0;
    const i_t nz_delta_y = delta_y_sparse.i.size();
    for (i_t k = 0; k < nz_delta_y; k++) {
      if (std::abs(delta_y_sparse.x[k]) > 1e-12) {
        delta_y_nz0++;
      }
    }
    const f_t dy_percent = static_cast<f_t>(delta_y_nz0) / static_cast<f_t>(nz_delta_y) * 100.0;
    if (dy_percent < 10.0) {
      //settings.log.printf("delta_y_nz0 %d nz_delta_y %d percentage %.1f\n", delta_y_nz0, nz_delta_y, dy_percent);
    }
    const f_t delta_y_nz_percentage = delta_y_nz0 / static_cast<f_t>(m) * 100.0;
    //const bool use_transpose = phase2::use_transpose_for_delta_z(lp, A_transpose, delta_y_sparse, nonbasic_list);
    const bool use_transpose = delta_y_nz_percentage <= 30.0;
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

#if 0
    if (use_transpose)
    {
      delta_y_sparse.to_dense(delta_y);
      std::vector<f_t> delta_z_check(n);
      std::vector<i_t> delta_z_mark_check(n, 0);
      std::vector<i_t> delta_z_indices_check  ;
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
          settings.log.printf("delta_z error %d transpose %e no transpose %e diff %e\n", k, delta_z[k], delta_z_check[k], diff);
        }
        error_check = std::max(error_check, diff);
      }
      if (error_check > 1e-6) { settings.log.printf("delta_z error %e\n", error_check); }
    }
#endif
    delta_z_time += toc(delta_z_start_time);

#ifdef COMPUTE_DUAL_RESIDUAL
    std::vector<f_t> dual_residual = delta_z;
    // || A'*delta_y + delta_z ||_inf
    matrix_transpose_vector_multiply(lp.A, 1.0, delta_y, 1.0, dual_residual);
    f_t dual_residual_norm = vector_norm_inf<i_t, f_t>(dual_residual);
    settings.log.printf("|| A'*dy - dz || %e use transpose %d\n", dual_residual_norm, use_transpose);
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
      f_t bfrt_start = tic();
#if 1
      f_t slope = direction == 1 ? (lp.lower[leaving_index] - x[leaving_index])
                             : (x[leaving_index] - lp.upper[leaving_index]);
      bound_flipping_ratio_test_t<i_t, f_t> bfrt(settings, start_time, m, n, slope, lp.lower, lp.upper, bounded_variables, vstatus, nonbasic_list, z, delta_z, delta_z_indices, nonbasic_mark);
      entering_index = bfrt.compute_step_length(step_length, nonbasic_entering_index);
      if constexpr (0)
      {
        f_t shadow_step_length;
        i_t shadow_nonbasic_entering_index;
        i_t shadow_entering_index = phase2::bound_flipping_ratio_test(lp,
                                                         settings,
                                                         start_time,
                                                         vstatus,
                                                         nonbasic_list,
                                                         x,
                                                         z,
                                                         delta_z,
                                                         direction,
                                                         leaving_index,
                                                         shadow_step_length,
                                                         shadow_nonbasic_entering_index);
        if (shadow_nonbasic_entering_index != nonbasic_entering_index)
        {
          settings.log.printf(
            "step diff %e shadow step length %e step length %e shadow nonbasic entering %d "
            "nonbasic entering %d\n",
            step_length - shadow_step_length,
            shadow_step_length,
            step_length,
            shadow_nonbasic_entering_index,
            nonbasic_entering_index);
        }
      }
#else
      entering_index = phase2::bound_flipping_ratio_test(lp,
                                                         settings,
                                                         start_time,
                                                         vstatus,
                                                         nonbasic_list,
                                                         x,
                                                         z,
                                                         delta_z,
                                                         direction,
                                                         leaving_index,
                                                         step_length,
                                                         nonbasic_entering_index);
#endif
      bfrt_time += toc(bfrt_start);
    } else {
      entering_index = phase2::phase2_ratio_test(
        lp, settings, vstatus, nonbasic_list, z, delta_z, step_length, nonbasic_entering_index);
    }
    if (entering_index == -2) { return dual::status_t::TIME_LIMIT; }
    if (entering_index == -3) { return dual::status_t::CONCURRENT_LIMIT; }
    if (entering_index == -1) {
      settings.log.printf("No entering variable found. Iter %d\n", iter);
      settings.log.printf("Scaled infeasibility %e\n", max_val);


      f_t primal_inf_check = 0.0;
      i_t num_infeasible = 0;
      f_t max_primal_infeas = 0.0;
      primal_infeasibility = 0.0;
      for (i_t k = 0; k < m; ++k) {
        const i_t j = basic_list[k];
        const f_t lower_infeas = lp.lower[j] - x[j];
        const f_t upper_infeas = x[j] - lp.upper[j];
        const f_t infeas = std::max(lower_infeas, upper_infeas);
        if (infeas > settings.primal_tol) {
          primal_inf_check += infeas;
          num_infeasible++;
          primal_infeasibility += infeas * infeas;
          squared_infeasibilities[j] = infeas * infeas;
          max_primal_infeas = std::max(max_primal_infeas, infeas);
        }
      }

      for (i_t j = 0; j < n; ++j)
      {
        delta_y_steepest_edge[j] = 1.0;
      }

      restart_steepest_edge = false;
      settings.log.printf("Max Primal infeasibility %e Sum Primal infeasibility %e Num infeasible %d\n", max_primal_infeas, primal_inf_check, num_infeasible);
      f_t perturbation = 0.0;
      for (i_t j = 0; j < n; ++j) {
        perturbation += std::abs(lp.objective[j] - objective[j]);
      }
      settings.log.printf("Perturbation %e\n", perturbation);

      if (perturbation > 0.0 && phase == 2) {
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

            std::vector<f_t> unperturbed_x(n);
            phase2::compute_primal_solution_from_basis(lp, ft, basic_list, nonbasic_list, vstatus, unperturbed_x);
            x = unperturbed_x;
            primal_infeasibility = phase2::compute_initial_primal_infeasibilities(
              lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);
            settings.log.printf("Updated primal infeasibility: %e\n", primal_infeasibility);

            objective = lp.objective;

            obj = 0.0;
            for (i_t j = 0; j < n; ++j)
            {
              obj += objective[j] * x[j];
            }

            if (dual_infeas <= settings.dual_tol && primal_infeasibility <= settings.primal_tol)
            {
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


            settings.log.printf("Continuing with perturbation removed and steepest edge norms reset\n");
             // Clear delta_z
            phase2::clear_delta_z(entering_index, leaving_index, delta_z_mark, delta_z_indices, delta_z);
            continue;
          } else {
            settings.log.printf("Failed to remove perturbation of %.2e.\n", perturbation);
          }
        }
      }

      if (perturbation == 0.0 && phase == 2)
      {

            constexpr bool use_farkas = true;

            if constexpr (use_farkas) {
              std::vector<f_t> farkas_y;
              std::vector<f_t> farkas_zl;
              std::vector<f_t> farkas_zu;
              f_t farkas_constant;
              std::vector<f_t> my_delta_y;
              delta_y_sparse.to_dense(my_delta_y);


              f_t obj_val = 0.0;
              for (i_t j = 0; j < n; ++j)
              {
               obj_val += objective[j] * x[j];
              }
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

      if (max_val < 2e-8) {
        // We could be done
        settings.log.printf("Exiting due to small primal infeasibility se %e\n", max_val);
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


    f_t vector_y_z_start_time = tic();
    // Update dual variables


  #if 1
    const i_t delta_y_nz = delta_y_sparse.i.size();
    for (i_t k = 0; k < delta_y_nz; ++k) {
      const i_t i = delta_y_sparse.i[k];
      y[i] += step_length * delta_y_sparse.x[k];
    }
    const i_t delta_z_nz = delta_z_indices.size();
    for (i_t k = 0; k < delta_z_nz; ++k) {
      const i_t j = delta_z_indices[k];
      z[j] += step_length * delta_z[j];
    }
    z[leaving_index] += step_length * delta_z[leaving_index];
  #else

    // y <- y + steplength * delta_y
    for (i_t i = 0; i < m; ++i) {
      y[i] += step_length * delta_y[i];
    }
    // z <- z + steplength * delta_z
    for (i_t j = 0; j < n; ++j) {
      z[j] += step_length * delta_z[j];
    }
#endif
    vector_time += toc(vector_y_z_start_time);

#ifdef COMPUTE_DUAL_RESIDUAL
    dual_res1 = z;
    for (i_t j = 0; j < n; ++j) {
      dual_res1[j] -= objective[j];
    }
    matrix_transpose_vector_multiply(lp.A, 1.0, y, 1.0, dual_res1);
    f_t dual_res_norm = vector_norm_inf<i_t, f_t>(dual_res1);
    if (dual_res_norm > settings.dual_tol) {
      settings.log.printf("|| A'*y + z - c || %e steplength %e\n", dual_res_norm, step_length);
    }
#endif

    f_t flip_start_time = tic();
    // Update primal variable

    const i_t num_flipped = phase2::flip_bounds(
      lp, settings, bounded_variables, objective, z, delta_z_indices, nonbasic_list, entering_index, vstatus, delta_x_flip, atilde_mark, atilde, atilde_index);

    flip_time += toc(flip_start_time);

    sparse_vector_t<i_t, f_t> delta_xB_0_sparse(m, 0);

    f_t ftran_start_time = tic();

    if (num_flipped > 0) {
      //settings.log.printf("Flipped %6d bounds. Dz nz %.2f Atilde nz %6d  %.2f %\n", num_flipped, static_cast<f_t>(delta_z_indices.size()) / static_cast<f_t>(n -m) * 100.0, atilde_index.size(), static_cast<f_t>(atilde_index.size()) / static_cast<f_t>(m) * 100.0);
      const i_t atilde_nz = atilde_index.size();
      if (1) {
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
      } else {
        // B*delta_xB_0 = atilde
        std::vector<f_t> delta_xB_0(m);
        ft.b_solve(atilde, delta_xB_0);
        for (i_t k = 0; k < m; ++k) {
          const i_t j = basic_list[k];
          x[j] += delta_xB_0[k];
        }
      }

#if 1
      for (i_t j : delta_z_indices) {
        x[j] += delta_x_flip[j];
        delta_x_flip[j] = 0.0;
      }
#else
      for (i_t k = 0; k < n - m; ++k) {
        const i_t j = nonbasic_list[k];
        x[j] += delta_x_flip[j];
      }
#endif

      // Clear atilde
      for (i_t k = 0; k < atilde_index.size(); ++k)
      {
        atilde[atilde_index[k]] = 0.0;
      }
      // Clear atilde_mark
      for (i_t k = 0; k < atilde_mark.size(); ++k)
      {
        atilde_mark[k] = 0;
      }
      atilde_index.clear();
    }

    f_t delta_x_leaving;
    if (direction == 1) {
      delta_x_leaving = lp.lower[leaving_index] - x[leaving_index];
    } else {
      delta_x_leaving = lp.upper[leaving_index] - x[leaving_index];
    }
    // B*w = -A(:, entering)
    std::vector<f_t> scaled_delta_xB(m);
    const i_t col_nz = lp.A.col_start[entering_index + 1] - lp.A.col_start[entering_index];
    std::vector<f_t> utilde(m);
    sparse_vector_t<i_t, f_t> utilde_sparse(m, 0);
    sparse_vector_t<i_t, f_t> scaled_delta_xB_sparse(m, 0);
    if (0)
    {
      std::fill(rhs.begin(), rhs.end(), 0.0);
      lp.A.load_a_column(entering_index, rhs);
      ft.b_solve(rhs, scaled_delta_xB, utilde);
      for (i_t i = 0; i < m; ++i) {
        scaled_delta_xB[i] *= -1.0;
      }
      sparse_vector_t<i_t, f_t> dxB_sparse(scaled_delta_xB);
      sparse_vector_t<i_t, f_t> ut_sparse(utilde);
      scaled_delta_xB_sparse = dxB_sparse;
      utilde_sparse = ut_sparse;
      b_solve_density = scaled_delta_xB_sparse.i.size() / static_cast<f_t>(m);
    }
    else
    {
      sparse_vector_t<i_t, f_t> rhs_sparse(lp.A, entering_index);
      ft.b_solve(rhs_sparse, scaled_delta_xB_sparse, utilde_sparse);
      const i_t xB_nz = scaled_delta_xB_sparse.i.size();
      for (i_t k = 0; k < xB_nz; ++k)
      {
        scaled_delta_xB_sparse.x[k] *= -1.0;
      }
      scaled_delta_xB_sparse.to_dense(scaled_delta_xB);
      utilde_sparse.to_dense(utilde);
      b_solve_density = static_cast<f_t>(xB_nz) / static_cast<f_t>(m);
#if 0
      rhs_sparse.to_dense(rhs);
#endif
    }

#if 0
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
      if (err_max > 1e-6)
      {
        printf("B multiply error %e\n", err_max);
      }
    }
#endif

    ftran_time += toc(ftran_start_time);

    f_t delta_x_change_start_time = tic();

#if 1
  f_t scale;
  const i_t scaled_delta_xB_nz = scaled_delta_xB_sparse.i.size();
  for (i_t k = 0; k < scaled_delta_xB_nz; ++k) {
    if (scaled_delta_xB_sparse.i[k] == basic_leaving_index) {
      scale = scaled_delta_xB_sparse.x[k];
      break;
    }
  }
  f_t primal_step_length = delta_x_leaving / scale;
  for (i_t k = 0; k < scaled_delta_xB_nz; ++k) {
    const i_t j = basic_list[scaled_delta_xB_sparse.i[k]];
    delta_x[j]  = primal_step_length * scaled_delta_xB_sparse.x[k];
  }
  delta_x[leaving_index] = delta_x_leaving;
  delta_x[entering_index] = primal_step_length;
#else
   f_t primal_step_length = delta_x_leaving / scaled_delta_xB[basic_leaving_index];
   std::vector<f_t> delta_x(n, 0.0);
    for (i_t k = 0; k < m; ++k) {
      const i_t j = basic_list[k];
      delta_x[j]  = primal_step_length * scaled_delta_xB[k];
    }
    delta_x[leaving_index] = delta_x_leaving;
    for (i_t k = 0; k < n - m; k++) {
      const i_t j = nonbasic_list[k];
      delta_x[j]  = 0.0;
    }
    delta_x[entering_index] = primal_step_length;
#endif
    vector_time += toc(delta_x_change_start_time);

#if 0
    std::vector<f_t> residual(m);
    matrix_vector_multiply(lp.A, 1.0, delta_x, 1.0, residual);
    f_t primal_step_err = vector_norm_inf<i_t, f_t>(residual);
    if (primal_step_err > 1e-4) { settings.log.printf("|| A * dx || %e\n", primal_step_err); }
#endif


    f_t steepest_edge_norms_start_time = tic();
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
      settings.log.printf(" Primal step length %e\n", primal_step_length);
      settings.log.printf("|| delta_xB || %e\n", vector_norm_inf(scaled_delta_xB));
      settings.log.printf("|| rhs || %e\n", vector_norm_inf(rhs));
    }
#endif
    assert(steepest_edge_status == 0);

    se_norms_time += toc(steepest_edge_norms_start_time);

    f_t vector_x_start_time = tic();
    // x <- x + delta_x
#if 1

  //std::vector<f_t> x_check = x;
  for (i_t k = 0; k < scaled_delta_xB_nz; ++k) {
    const i_t j = basic_list[scaled_delta_xB_sparse.i[k]];
    x[j] += delta_x[j];
  }
  // Leaving index already included above
  x[entering_index] += delta_x[entering_index];
#else

  for (i_t j = 0; j < n; ++j) {
    x[j] += delta_x[j];
  }

#endif
    vector_time += toc(vector_x_start_time);

#ifdef COMPUTE_PRIMAL_RESIDUAL
    residual = lp.rhs;
    matrix_vector_multiply(lp.A, 1.0, x, -1.0, residual);
    primal_residual = vector_norm_inf<i_t, f_t>(residual);
    if (iter % 100 == 0 && primal_residual > 10 * settings.primal_tol) {
      settings.log.printf("|| A*x - b || %e\n", primal_residual);
    }
#endif


    f_t objective_start_time = tic();
#if 1
    for (i_t k = 0; k < scaled_delta_xB_nz; ++k) {
      const i_t j = basic_list[scaled_delta_xB_sparse.i[k]];
      obj += delta_x[j] * lp.objective[j];
    }
    // Leaving index already included above
    obj += delta_x[entering_index] * lp.objective[entering_index];

    //const f_t obj_check  = compute_objective(lp, x);
    //if (std::abs(obj - obj_check) > 1e-5) {
    //  settings.log.printf("Objective error %e: %e %e\n", std::abs(obj - obj_check), obj, obj_check);
    //}
#endif
    objective_time += toc(objective_start_time);

#if 1
    f_t update_infeasibility_start_time = tic();
    // Update primal infeasibilities
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

    if (primal_infeasibility < 0.0) {
      settings.log.printf("!!!!! Negative primal infeasibility %e\n", primal_infeasibility);
    }

    phase2::clean_up_infeasibilities(squared_infeasibilities, infeasibility_indices);
#endif

#if CHECK_PRIMAL_INFEASIBILITIES
    // Check primal infeasibilities
    {
      for (i_t k = 0; k < m; ++k)
      {
        const i_t j = basic_list[k];
        const f_t lower_infeas = lp.lower[j] - x[j];
        const f_t upper_infeas = x[j] - lp.upper[j];
        const f_t infeas = std::max(lower_infeas, upper_infeas);
        if (infeas > settings.primal_tol) {
          const f_t square_infeas = infeas * infeas;
          if (square_infeas != squared_infeasibilities[j]) {
            settings.log.printf("Primal infeasibility mismatch %d %e != %e\n", j, square_infeas, squared_infeasibilities[j]);
          }
          bool found = false;
          for (i_t h = 0; h < infeasibility_indices.size(); ++h) {
            if (infeasibility_indices[h] == j) {
              found = true;
              break;
            }
          }
          if (!found) {
            settings.log.printf("Infeasibility index not found %d\n", j);
          }
        }
      }
    }
#endif

#if 1
    update_infeasibility_time += toc(update_infeasibility_start_time);
#endif

    // Clear delta_x
    for (i_t k = 0; k < scaled_delta_xB_nz; ++k) {
      const i_t j = basic_list[scaled_delta_xB_sparse.i[k]];
      delta_x[j] = 0.0;
    }
    // Leaving index already included above
    delta_x[entering_index] = 0.0;
    scaled_delta_xB_sparse.i.clear();
    scaled_delta_xB_sparse.x.clear();


    f_t perturb_start_time = tic();
    f_t sum_perturb = 0.0;
    phase2::compute_perturbation(lp, settings, delta_z_indices, z, objective, sum_perturb);
    perturb_time += toc(perturb_start_time);

    // Update basis
    vstatus[entering_index] = variable_status_t::BASIC;
    if (lp.lower[leaving_index] != lp.upper[leaving_index]) {
      vstatus[leaving_index] = static_cast<variable_status_t>(-direction);
    } else {
      vstatus[leaving_index] = variable_status_t::NONBASIC_FIXED;
    }
    basic_list[basic_leaving_index]        = entering_index;
    nonbasic_list[nonbasic_entering_index] = leaving_index;
    nonbasic_mark[entering_index] = -1;
    nonbasic_mark[leaving_index] = nonbasic_entering_index;
    basic_mark[leaving_index] = -1;
    basic_mark[entering_index] = basic_leaving_index;

    f_t lu_update_start_time = tic();
    // Refactor or Update
    bool should_refactor = ft.num_updates() > settings.refactor_frequency;
    if (!should_refactor) {
      i_t recommend_refactor = ft.update(utilde_sparse, UTsol_sparse, basic_leaving_index);
      //i_t recommend_refactor = ft.update(utilde, UTsol, basic_leaving_index);
#ifdef CHECK_UPDATE
      {
        csc_matrix_t<i_t, f_t> Btest(m, m, 1);
        ft.multiply_lu(Btest);
        {
          csc_matrix_t<i_t, f_t> B(m, m, 1);
          form_b(lp.A, basic_list, B);
          csc_matrix_t<i_t, f_t> Diff(m, m, 1);
          add(Btest, B, 1.0, -1.0, Diff);
          const f_t err = Diff.norm1();
          if (err > settings.primal_tol) {
            settings.log.printf("|| B - L*U || %e\n", Diff.norm1());
          }
          if (err > settings.primal_tol)
          {
            for (i_t j = 0; j < m; ++j)
            {
              for (i_t p = Diff.col_start[j]; p < Diff.col_start[j + 1]; ++p)
              {
                const i_t i = Diff.i[p];
                if (Diff.x[p] != 0.0)
                {
                  settings.log.printf("Diff %d %d %e\n", j, i, Diff.x[p]);
                }
              }
            }
          }
          settings.log.printf("basic leaving index %d\n", basic_leaving_index);
          assert(err < settings.primal_tol);
        }
      }
#endif
      should_refactor = recommend_refactor == 1;
    }

    if (should_refactor) {
      if (factorize_basis(lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed) ==
          -1) {
        basis_repair(lp.A, settings, deficient, slacks_needed, basic_list, nonbasic_list, vstatus);
        if (factorize_basis(
              lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed) == -1) {
          return dual::status_t::NUMERICAL;
        }
      }
      reorder_basic_list(q, basic_list);
      ft.reset(L, U, p);
      for (i_t k = 0; k < n; k++) {
        basic_mark[k]    = -1;
        nonbasic_mark[k] = -1;
      }
      for (i_t k = 0; k < m; k++) {
        basic_mark[basic_list[k]] = k;
      }
      for (i_t k = 0; k < n - m; k++) {
        nonbasic_mark[nonbasic_list[k]] = k;
      }
    }

    lu_update_time += toc(lu_update_start_time);

    f_t steepest_edge_entering_start_time = tic();
    phase2::compute_steepest_edge_norm_entering(
      settings, m, ft, basic_leaving_index, entering_index, b_transpose_solve_density, delta_y_steepest_edge);
    se_entering_time += toc(steepest_edge_entering_start_time);

#ifdef STEEPEST_EDGE_DEBUG
    if (iter < 100 || iter % 100 == 0))
        {
            phase2::check_steepest_edge_norms(settings, basic_list, ft, delta_y_steepest_edge);
        }
#endif

#if 0
    for (i_t k = 0; k < m; k++) {
      if (basic_mark[basic_list[k]] != k) {
        printf("Basic mark %d %d\n", basic_list[k], k);
      }
    }
    for (i_t k = 0; k < n - m; k++) {
      if (nonbasic_mark[nonbasic_list[k]] != k) {
        printf("Nonbasic mark %d %d\n", nonbasic_list[k], k);
      }
    }
#endif

    iter++;

#if 1
    // Clear delta_y
    //const i_t nz_dy = delta_y_sparse.i.size();
    //for (i_t k = 0; k < nz_dy; ++k) {
    // delta_y[delta_y_sparse.i[k]] = 0.0;
    //}

    // Clear delta_z
    phase2::clear_delta_z(entering_index, leaving_index, delta_z_mark, delta_z_indices, delta_z);


#endif

    f_t now       = toc(start_time);
    if ((iter - start_iter) < settings.first_iteration_log ||
        (iter % settings.iteration_log_frequency) == 0) {
      if (phase == 1 && iter == 1) {
        settings.log.printf(" Iter     Objective   Primal Infeas  Perturb  Time\n");
      }
      settings.log.printf("%5d %+.16e %8d %.8e %.2e %.2e %.2f\n",
                          iter,
                          compute_user_objective(lp, obj),
                          infeasibility_indices.size(),
                          primal_infeasibility,
                          sum_perturb,
                          step_length,
                          now);
    }

    if (obj >= settings.cut_off) {
      settings.log.printf("Solve cutoff. Current objecive %e. Cutoff %e\n", obj, settings.cut_off);
      return dual::status_t::CUTOFF;
    }

    if (now > settings.time_limit) { return dual::status_t::TIME_LIMIT; }

    if (settings.concurrent_halt != nullptr &&
        settings.concurrent_halt->load(std::memory_order_acquire) == 1) {
      return dual::status_t::CONCURRENT_LIMIT;
    }
  }
  if (iter >= iter_limit) { status = dual::status_t::ITERATION_LIMIT; }

  if (phase == 2) {
    const f_t total_time = bfrt_time + pricing_time + btran_time + ftran_time + flip_time +
                          delta_z_time + lu_update_time + se_norms_time + se_entering_time +
                          perturb_time + vector_time + objective_time + update_infeasibility_time;
    settings.log.printf("BFRT time       %.2f %4.1f%\n", bfrt_time, 100.0 * bfrt_time / total_time);
    settings.log.printf("Pricing time    %.2f %4.1f%\n", pricing_time, 100.0 * pricing_time / total_time);
    settings.log.printf("BTran time      %.2f %4.1f%\n", btran_time, 100.0 * btran_time / total_time);
    settings.log.printf("FTran time      %.2f %4.1f%\n", ftran_time, 100.0 * ftran_time / total_time);
    settings.log.printf("Flip time       %.2f %4.1f%\n", flip_time, 100.0 * flip_time / total_time);
    settings.log.printf("Delta_z time    %.2f %4.1f%\n", delta_z_time, 100.0 * delta_z_time / total_time);
    settings.log.printf("LU update time  %.2f %4.1f%\n", lu_update_time, 100.0 * lu_update_time / total_time);
    settings.log.printf("SE norms time   %.2f %4.1f%\n", se_norms_time, 100.0 * se_norms_time / total_time);
    settings.log.printf("SE enter time   %.2f %4.1f%\n", se_entering_time, 100.0 * se_entering_time / total_time);
    settings.log.printf("Perturb time    %.2f %4.1f%\n", perturb_time, 100.0 * perturb_time / total_time);
    settings.log.printf("Vector time     %.2f %4.1f%\n", vector_time, 100.0 * vector_time / total_time);
    settings.log.printf("Objective time  %.2f %4.1f%\n", objective_time, 100.0 * objective_time / total_time);
    settings.log.printf("Inf update time %.2f %4.1f%\n", update_infeasibility_time, 100.0 * update_infeasibility_time / total_time);
    settings.log.printf("Sum             %.2f\n", total_time);

    settings.log.printf("Sparse delta_z %8d %8.2f%\n", sparse_delta_z, 100.0 * sparse_delta_z / (sparse_delta_z + dense_delta_z));
    settings.log.printf("Dense delta_z  %8d %8.2f%\n", dense_delta_z, 100.0 * dense_delta_z / (sparse_delta_z + dense_delta_z));
    ft.print_stats();
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
