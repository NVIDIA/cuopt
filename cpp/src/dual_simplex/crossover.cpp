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

#include <dual_simplex/crossover.hpp>

#include <dual_simplex/basis_solves.hpp>
#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/primal.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/tic_toc.hpp>

#include <array>

namespace cuopt::linear_programming::dual_simplex {

namespace {

crossover_status_t return_to_status(int status)
{
  if (status == -1) {
    return crossover_status_t::TIME_LIMIT;
  } else if (status == -2) {
    return crossover_status_t::CONCURRENT_LIMIT;
  } else {
    return crossover_status_t::NUMERICAL_ISSUES;
  }
}

template <typename i_t, typename f_t>
void verify_basis(i_t m, i_t n, const std::vector<variable_status_t>& vstatus)
{
  i_t num_basic      = 0;
  i_t num_nonbasic   = 0;
  i_t num_superbasic = 0;
  {
    for (i_t j = 0; j < n; ++j) {
      if (vstatus[j] == variable_status_t::BASIC) {
        num_basic++;
      } else if (vstatus[j] == variable_status_t::NONBASIC_LOWER ||
                 vstatus[j] == variable_status_t::NONBASIC_UPPER ||
                 vstatus[j] == variable_status_t::NONBASIC_FREE ||
                 vstatus[j] == variable_status_t::NONBASIC_FIXED) {
        num_nonbasic++;
      } else {
        num_superbasic++;
      }
    }
  }
  assert(num_nonbasic == n - m);
  assert(num_basic == m);
  assert(num_superbasic == 0);
}

template <typename i_t, typename f_t>
void compare_vstatus_with_lists(i_t m,
                                i_t n,
                                const std::vector<i_t>& basic_list,
                                const std::vector<i_t>& nonbasic_list,
                                const std::vector<variable_status_t>& vstatus)
{
  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    assert(vstatus[j] == variable_status_t::BASIC);
  }
  for (i_t k = 0; k < std::min(static_cast<i_t>(nonbasic_list.size()), n - m); ++k) {
    const i_t j = nonbasic_list[k];
    assert(vstatus[j] == variable_status_t::NONBASIC_LOWER ||
           vstatus[j] == variable_status_t::NONBASIC_UPPER ||
           vstatus[j] == variable_status_t::NONBASIC_FREE ||
           vstatus[j] == variable_status_t::NONBASIC_FIXED);
  }
}

template <typename i_t, typename f_t>
f_t dual_infeasibility(const lp_problem_t<i_t, f_t>& lp,
                       const simplex_solver_settings_t<i_t, f_t>& settings,
                       const std::vector<variable_status_t>& vstatus,
                       const std::vector<f_t>& z)
{
  const i_t n             = lp.num_cols;
  const i_t m             = lp.num_rows;
  i_t num_infeasible      = 0;
  f_t sum_infeasible      = 0.0;
  constexpr f_t tight_tol = 1e-6;
  i_t lower_bound_inf     = 0;
  i_t upper_bound_inf     = 0;
  i_t free_inf            = 0;
  i_t non_basic_lower_inf = 0;
  i_t non_basic_upper_inf = 0;

  for (i_t j = 0; j < n; ++j) {
    if (vstatus[j] == variable_status_t::NONBASIC_FIXED) {
      continue;  // Is it correct to ignore fixed variables?
    }
    if (lp.upper[j] == inf && lp.lower[j] > -inf && z[j] < -tight_tol) {
      // -inf < l_j <= x_j < inf, so need z_j > 0 to be feasible
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
      settings.log.debug("lower bound infeasible %d z %e lower %e upper %e vstatus %d\n",
                         j,
                         z[j],
                         lp.lower[j],
                         lp.upper[j],
                         static_cast<int>(vstatus[j]));
      lower_bound_inf++;
    } else if (lp.lower[j] == -inf && lp.upper[j] < inf && z[j] > tight_tol) {
      // -inf < x_j <= u_j < inf, so need z_j < 0 to be feasible
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
      settings.log.debug("upper bound infeasible %d z %e lower %e upper %e vstatus %d\n",
                         j,
                         z[j],
                         lp.lower[j],
                         lp.upper[j],
                         static_cast<int>(vstatus[j]));
      upper_bound_inf++;
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
    } else if (vstatus[j] == variable_status_t::NONBASIC_LOWER && z[j] < -tight_tol) {
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
      non_basic_lower_inf++;
      settings.log.debug(
        "nonbasic lower infeasible %d z %e lower %e upper %e\n", j, z[j], lp.lower[j], lp.upper[j]);
    } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER && z[j] > tight_tol) {
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
      non_basic_upper_inf++;
    }
  }
  settings.log.debug(
    "num infeasible %d lower_bound_inf %d upper_bound_inf %d free_inf %d non_basic_lower_inf %d "
    "non_basic_upper_inf %d\n",
    num_infeasible,
    lower_bound_inf,
    upper_bound_inf,
    free_inf,
    non_basic_lower_inf,
    non_basic_upper_inf);

  return sum_infeasible;
}

template <typename i_t, typename f_t>
f_t primal_infeasibility(const lp_problem_t<i_t, f_t>& lp,
                         const simplex_solver_settings_t<i_t, f_t>& settings,
                         const std::vector<variable_status_t>& vstatus,
                         const std::vector<f_t>& x)
{
  const i_t n              = lp.num_cols;
  f_t primal_inf           = 0;
  constexpr bool verbose   = false;
  constexpr f_t infeas_tol = 1e-3;
  for (i_t j = 0; j < n; ++j) {
    if (x[j] < lp.lower[j]) {
      // x_j < l_j => -x_j > -l_j => -x_j + l_j > 0
      const f_t infeas = -x[j] + lp.lower[j];
      primal_inf += infeas;
      if (verbose && infeas > infeas_tol) {
        settings.log.debug("x %d infeas %e lo %e val %e up %e vstatus %hhd\n",
                           j,
                           infeas,
                           lp.lower[j],
                           x[j],
                           lp.upper[j],
                           vstatus[j]);
      }
    }
    if (x[j] > lp.upper[j]) {
      // x_j > u_j => x_j - u_j > 0
      const f_t infeas = x[j] - lp.upper[j];
      primal_inf += infeas;
      if (verbose && infeas > infeas_tol) {
        settings.log.debug("x %d infeas %e lo %e val %e up %e vstatus %hhd\n",
                           j,
                           infeas,
                           lp.lower[j],
                           x[j],
                           lp.upper[j],
                           vstatus[j]);
      }
    }
  }
  return primal_inf;
}

template <typename i_t, typename f_t>
f_t dual_residual(const lp_problem_t<i_t, f_t>& lp, const lp_solution_t<i_t, f_t>& solution)
{
  std::vector<f_t> dual_residual = solution.z;
  const i_t n                    = lp.num_cols;
  // dual_residual <- z - c
  for (i_t j = 0; j < n; j++) {
    dual_residual[j] -= lp.objective[j];
  }
  // dual_residual <- 1.0*A'*y + 1.0*(z - c)
  matrix_transpose_vector_multiply(lp.A, 1.0, solution.y, 1.0, dual_residual);
  return vector_norm_inf<i_t, f_t>(dual_residual);
}

template <typename i_t, typename f_t>
f_t dual_ratio_test(const lp_problem_t<i_t, f_t>& lp,
                    const simplex_solver_settings_t<i_t, f_t>& settings,
                    const std::vector<i_t>& nonbasic_list,
                    const std::vector<variable_status_t>& vstatus,
                    const std::vector<f_t>& z,
                    const std::vector<f_t>& delta_zN,
                    i_t& entering_index,
                    i_t& nonbasic_entering_index)
{
  i_t m                   = lp.num_rows;
  i_t n                   = lp.num_cols;
  f_t step_length         = 1.0;
  entering_index          = -1;
  nonbasic_entering_index = -1;
  for (i_t k = 0; k < n - m; ++k) {
    const i_t j             = nonbasic_list[k];
    const f_t zj            = z[j];
    const f_t dz            = delta_zN[k];
    constexpr f_t pivot_tol = 1e-9;
    constexpr f_t dual_tol  = 1e-6;
    if (vstatus[j] == variable_status_t::NONBASIC_FIXED) { continue; }
    if (vstatus[j] == variable_status_t::NONBASIC_LOWER && z[j] > -pivot_tol && dz < -pivot_tol) {
      const f_t ratio = (-dual_tol - zj) / dz;
      if (ratio < step_length) {
        step_length             = ratio;
        entering_index          = j;
        nonbasic_entering_index = k;
        if (step_length < 0) {
          settings.log.debug(
            "Step length %e is negative. NONBASIC_LOWER z %e dz %e\n", step_length, zj, dz);
        }
      }
    }
    if (vstatus[j] == variable_status_t::NONBASIC_UPPER && z[j] < pivot_tol && dz > pivot_tol) {
      const f_t ratio = (dual_tol - zj) / dz;
      if (ratio < step_length) {
        step_length             = ratio;
        entering_index          = j;
        nonbasic_entering_index = k;
        if (step_length < 0) {
          settings.log.debug(
            "Step length %e is negative. NONBASIC_UPPER z %e dz %e\n", step_length, zj, dz);
        }
      }
    }
    if (vstatus[j] == variable_status_t::NONBASIC_FREE && std::abs(z[j]) < pivot_tol &&
        std::abs(dz) > pivot_tol) {
      // -dual_tol <= zj + step_length * dz <= dual_tol
      // step_length  <= (dual_tol - zj) / dz  if dz > 0
      // step_length <= (-dual_tol - zj) / dz if dz < 0
      const f_t ratio = dz > 0 ? (dual_tol - zj) / dz : (-dual_tol - zj) / dz;
      if (ratio < step_length) {
        step_length             = ratio;
        entering_index          = j;
        nonbasic_entering_index = k;
        if (step_length < 0) {
          settings.log.debug(
            "Step length %e is negative. NONBASIC_FREE z %e dz %e\n", step_length, zj, dz);
        }
      }
    }
  }
  return step_length;
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

  y.resize(m, 0.0);
  std::vector<f_t> cB(m);
  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    cB[k]       = lp.objective[j];
  }
  sparse_vector_t<i_t, f_t> cB_sparse(cB);
  sparse_vector_t<i_t, f_t> y_sparse(m, 1);
  ft.b_transpose_solve(cB_sparse, y_sparse);
  y_sparse.scatter(y);

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
i_t dual_push(const lp_problem_t<i_t, f_t>& lp,
              const simplex_solver_settings_t<i_t, f_t>& settings,
              f_t start_time,
              lp_solution_t<i_t, f_t>& solution,
              basis_update_mpf_t<i_t, f_t>& ft,
              std::vector<i_t>& basic_list,
              std::vector<i_t>& nonbasic_list,
              std::vector<i_t>& superbasic_list,
              std::vector<variable_status_t>& vstatus)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;

  superbasic_list.clear();
  std::vector<i_t> superbasic_list_index;
  for (i_t k = 0; k < m; ++k) {
    const i_t j       = basic_list[k];
    const f_t zj      = solution.z[j];
    constexpr f_t tol = 1e-6;
    if (std::abs(zj) > tol) {
      superbasic_list.push_back(j);
      superbasic_list_index.push_back(k);
    }
  }

  for (i_t k = 0; k < n - m; ++k) {
    const i_t j       = nonbasic_list[k];
    const f_t zj      = solution.z[j];
    constexpr f_t tol = 1e-3;
    if (vstatus[j] == variable_status_t::NONBASIC_LOWER && zj < -tol) {
      settings.log.debug("infeasible z %d on lower. zj %e xj %e lo %e up %e\n",
                         j,
                         zj,
                         solution.x[j],
                         lp.lower[j],
                         lp.upper[j]);
    } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER && zj > tol) {
      settings.log.debug("infeasible z %d on upper. zj %e xj %e lo %e up %e\n",
                         j,
                         zj,
                         solution.x[j],
                         lp.lower[j],
                         lp.upper[j]);
    } else if (vstatus[j] == variable_status_t::NONBASIC_FREE && std::abs(zj) > tol) {
      settings.log.debug("infeasible z %d free. zj %e\n", j, zj);
    }
  }

  i_t total_superbasics = superbasic_list.size();
  settings.log.debug("Dual push: superbasics %ld\n", total_superbasics);
  verify_basis<i_t, f_t>(m, n, vstatus);
  compare_vstatus_with_lists<i_t, f_t>(m, n, basic_list, nonbasic_list, vstatus);

  std::vector<f_t>& z       = solution.z;
  std::vector<f_t>& y       = solution.y;
  const std::vector<f_t>& x = solution.x;
  i_t num_pushes            = 0;
  while (superbasic_list.size() > 0) {
    const i_t s                   = superbasic_list.back();
    const i_t basic_leaving_index = superbasic_list_index.back();

    // Remove superbasic variable
    superbasic_list.pop_back();
    superbasic_list_index.pop_back();

    f_t delta_zs = -z[s];
    sparse_vector_t<i_t, f_t> es_sparse(m, 1);
    es_sparse.i[0] = basic_leaving_index;
    es_sparse.x[0] = -delta_zs;

    // B^T delta_y = -delta_zs*es
    std::vector<f_t> delta_y(m);
    sparse_vector_t<i_t, f_t> delta_y_sparse(m, 1);
    sparse_vector_t<i_t, f_t> UTsol_sparse(m, 1);
    ft.b_transpose_solve(es_sparse, delta_y_sparse, UTsol_sparse);
    delta_y_sparse.scatter(delta_y);

    // We solved B^T delta_y = -delta_zs*es, but for the update we need
    // U^T*etilde = es.
    // We have that B^T = U^T*L^T so B^T delta_y = U^T*etilde = -delta_zs*es
    // So we need to divide by -delta_zs
    for (i_t k = 0; k < UTsol_sparse.i.size(); ++k) {
      UTsol_sparse.x[k] /= -delta_zs;
    }

    // delta_zN = -N^T delta_y
    std::vector<f_t> delta_zN(n - m);
    for (i_t k = 0; k < n - m; ++k) {
      const i_t j         = nonbasic_list[k];
      const i_t col_start = lp.A.col_start[j];
      const i_t col_end   = lp.A.col_start[j + 1];
      f_t dot             = 0.0;
      for (i_t p = col_start; p < col_end; ++p) {
        dot += lp.A.x[p] * delta_y[lp.A.i[p]];
      }
      delta_zN[k] = -dot;
    }

    i_t entering_index          = -1;
    i_t nonbasic_entering_index = -1;
    f_t step_length             = dual_ratio_test(
      lp, settings, nonbasic_list, vstatus, z, delta_zN, entering_index, nonbasic_entering_index);
    assert(step_length >= -1e-6);

    // y <- y + step_length * delta_y
    for (i_t i = 0; i < m; ++i) {
      y[i] += step_length * delta_y[i];
    }

    // z <- z + step_length * delta z
    for (i_t k = 0; k < n - m; ++k) {
      const i_t j = nonbasic_list[k];
      z[j] += step_length * delta_zN[k];
    }
    z[s] += step_length * delta_zs;

    if (entering_index != -1) {
      // Update the basis
      assert(std::abs(z[entering_index]) < 1e-4);
      z[entering_index] = 0.0;
      assert(basic_list[basic_leaving_index] == s);
      basic_list[basic_leaving_index]        = entering_index;
      nonbasic_list[nonbasic_entering_index] = s;
      // Set vstatus
      vstatus[entering_index] = variable_status_t::BASIC;
      const f_t lower_slack   = x[s] - lp.lower[s];
      const f_t upper_slack   = lp.upper[s] - x[s];
      constexpr f_t tol       = 1e-6;
      const f_t fixed_tol     = settings.fixed_tol;
      if (std::abs(lp.lower[s] - lp.upper[s]) < fixed_tol) {
        vstatus[s] = variable_status_t::NONBASIC_FIXED;
      } else if (lower_slack < tol && lp.lower[s] > -inf) {
        vstatus[s] = variable_status_t::NONBASIC_LOWER;
      } else if (upper_slack < tol && lp.upper[s] < inf) {
        vstatus[s] = variable_status_t::NONBASIC_UPPER;
      } else if (upper_slack < lower_slack) {
        vstatus[s] = variable_status_t::NONBASIC_UPPER;
      } else {
        vstatus[s] = variable_status_t::NONBASIC_LOWER;
      }
#ifdef PARANOID
      printf(
        "Set superbasic %d to nonbasic with status %d. lower %e x %e upper %e gap %e zj %e infeas "
        "%d\n",
        s,
        vstatus[s],
        lp.lower[s],
        x[s],
        lp.upper[s],
        lp.upper[s] - lp.lower[s],
        z[s],
        vstatus[s] == variable_status_t::NONBASIC_LOWER
          ? z[s] < -1e-6
          : (vstatus[s] == variable_status_t::NONBASIC_UPPER ? z[s] > 1e-6 : 0));
#endif
      // Refactor or Update
      bool should_refactor = ft.num_updates() > settings.refactor_frequency;
      if (!should_refactor) {
        sparse_vector_t<i_t, f_t> abar_sparse(lp.A, entering_index);
        sparse_vector_t<i_t, f_t> utilde_sparse(m, 1);
        // permute abar_sparse and store in utilde_sparse
        abar_sparse.inverse_permute_vector(ft.inverse_row_permutation(), utilde_sparse);
        ft.l_solve(utilde_sparse);
        i_t recommend_refactor = ft.update(utilde_sparse, UTsol_sparse, basic_leaving_index);
        should_refactor        = recommend_refactor == 1;
      }

      if (should_refactor) {
        csc_matrix_t<i_t, f_t> L(m, m, 1);
        csc_matrix_t<i_t, f_t> U(m, m, 1);
        std::vector<i_t> p(m);
        std::vector<i_t> pinv(m);
        std::vector<i_t> q(m);
        std::vector<i_t> deficient;
        std::vector<i_t> slacks_needed;
        factorize_basis(lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed);
        reorder_basic_list(q, basic_list);
        // Reordering the basic list causes us to mess up the superbasic list index
        // so we need to update it
        superbasic_list.clear();
        superbasic_list_index.clear();
        for (i_t k = 0; k < m; ++k) {
          const i_t j       = basic_list[k];
          const f_t zj      = solution.z[j];
          constexpr f_t tol = 1e-6;
          if (std::abs(zj) > tol) {
            superbasic_list.push_back(j);
            superbasic_list_index.push_back(k);
          }
        }
        ft.reset(L, U, p);
      }

    } else {
      // Basis remains unchanged
    }

    num_pushes++;
    if (num_pushes % settings.iteration_log_frequency == 0 || superbasic_list.size() == 0) {
      settings.log.printf(
        "%d of %d dual pushes in %.2fs\n", num_pushes, total_superbasics, toc(start_time));
    }
    if (toc(start_time) > settings.time_limit) {
      settings.log.printf("Crossover time exceeded\n");
      return -1;
    }
    if (settings.concurrent_halt != nullptr && *settings.concurrent_halt == 1) {
      settings.log.printf("Concurrent halt\n");
      return -2;
    }
  }

  verify_basis<i_t, f_t>(m, n, vstatus);

  std::vector<f_t> y_test;
  std::vector<f_t> z_test;
  compute_dual_solution_from_basis(lp, ft, basic_list, nonbasic_list, y_test, z_test);

  solution.y = y_test;
  solution.z = z_test;
  solution.iterations += num_pushes;

  return 0;
}

template <typename i_t, typename f_t>
f_t primal_residual(const lp_problem_t<i_t, f_t>& lp, const lp_solution_t<i_t, f_t>& solution)
{
  std::vector<f_t> primal_residual = lp.rhs;
  matrix_vector_multiply(lp.A, 1.0, solution.x, -1.0, primal_residual);
  return vector_norm_inf<i_t, f_t>(primal_residual);
}

template <typename i_t, typename f_t>
f_t primal_ratio_test(const lp_problem_t<i_t, f_t>& lp,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      const std::vector<i_t>& basic_list,
                      const std::vector<f_t>& x,
                      const sparse_vector_t<i_t, f_t>& delta_xB,
                      i_t& leaving_index,
                      i_t& basic_leaving_index,
                      i_t& bound)
{
  const i_t m             = lp.num_rows;
  const i_t n             = lp.num_cols;
  f_t step_length         = 1.0;
  constexpr f_t pivot_tol = 1e-9;
  for (i_t k = 0; k < delta_xB.i.size(); ++k) {
    const i_t j = basic_list[delta_xB.i[k]];
    if (x[j] <= lp.upper[j] && delta_xB.x[k] > pivot_tol && lp.upper[j] < inf) {
      const f_t ratio = (lp.upper[j] - x[j]) / delta_xB.x[k];
      if (ratio < step_length) {
        step_length         = ratio;
        leaving_index       = j;
        basic_leaving_index = delta_xB.i[k];
        bound               = 1;
        if (step_length < 0) {
          settings.log.debug("Step length %e is negative. delta x %e upper %e x %e\n",
                             step_length,
                             delta_xB.x[k],
                             lp.upper[j],
                             x[j]);
        }
      }
    } else if (x[j] >= lp.lower[j] && delta_xB.x[k] < -pivot_tol && lp.lower[j] > -inf) {
      const f_t ratio = (lp.lower[j] - x[j]) / delta_xB.x[k];
      if (ratio < step_length) {
        step_length         = ratio;
        leaving_index       = j;
        basic_leaving_index = delta_xB.i[k];
        bound               = -1;
        if (step_length < 0) {
          settings.log.debug("Step length %e is negative. delta x %e lower %e x %e\n",
                             step_length,
                             delta_xB.x[k],
                             lp.lower[j],
                             x[j]);
        }
      }
    }
  }
  return step_length;
}

template <typename i_t, typename f_t>
i_t primal_push(const lp_problem_t<i_t, f_t>& lp,
                const simplex_solver_settings_t<i_t, f_t>& settings,
                f_t start_time,
                lp_solution_t<i_t, f_t>& solution,
                basis_update_mpf_t<i_t, f_t>& ft,
                std::vector<i_t>& basic_list,
                std::vector<i_t>& nonbasic_list,
                std::vector<i_t>& superbasic_list,
                std::vector<variable_status_t>& vstatus)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;

  settings.log.debug("Primal push: superbasic %ld\n", superbasic_list.size());

  std::vector<f_t>& x = solution.x;
  std::vector<f_t>& y = solution.y;
  std::vector<f_t>& z = solution.z;

  f_t last_print_time         = tic();
  const i_t total_superbasics = superbasic_list.size();
  i_t num_pushes              = 0;
  while (superbasic_list.size() > 0) {
    const i_t s = superbasic_list.back();

    // Load A(:, s) into As
    sparse_vector_t<i_t, f_t> As_sparse(lp.A, s);
    sparse_vector_t<i_t, f_t> w_sparse(m, 0);
    sparse_vector_t<i_t, f_t> utilde_sparse(m, 0);
    // Solve B*w = As, w = -delta_xB/delta_xs
    ft.b_solve(As_sparse, w_sparse, utilde_sparse);
#ifdef CHECK_RESIDUAL
    {
      std::vector<f_t> w(m);
      w_sparse.to_dense(w);
      std::vector<f_t> r(m);
      b_multiply(lp, basic_list, w, r);
      std::vector<f_t> As(m);
      lp.A.load_a_column(s, As);
      for (i_t i = 0; i < m; ++i) {
        r[i] -= As[i];
      }
      printf("|| B*w - As || %e\n", vector_norm_inf<i_t, f_t>(r));
    }
#endif

    // Perform two ratio tests
    std::array<sparse_vector_t<i_t, f_t>, 2> delta_xB_trials;
    std::array<f_t, 2> delta_xs_trials;
    std::array<f_t, 2> step_length_trials;
    std::array<i_t, 2> leaving_index_trials;
    std::array<i_t, 2> basic_leaving_index_trials;
    std::array<i_t, 2> bound_trials;
    i_t best = -1;
    for (i_t push = 0; push < 2; ++push) {
      if (push == 0 && lp.lower[s] == -inf) {
        best = 1;
        continue;
      }
      if (push == 1 && lp.upper[s] == inf) {
        // We can't do this push
        best = 0;
        continue;
      }
      delta_xs_trials[push] = push == 0 ? lp.lower[s] - x[s] : lp.upper[s] - x[s];
      const f_t delta_xs    = delta_xs_trials[push];
      // Compute delta_xB_trial = -delta_xs * w
      sparse_vector_t<i_t, f_t>& delta_xB_trial = delta_xB_trials[push];
      delta_xB_trial                            = w_sparse;
      for (i_t k = 0; k < w_sparse.i.size(); ++k) {
        delta_xB_trial.x[k] = -w_sparse.x[k] * delta_xs;
      }
      leaving_index_trials[push]       = -1;
      basic_leaving_index_trials[push] = -1;
      bound_trials[push]               = 0;
      step_length_trials[push]         = primal_ratio_test(lp,
                                                   settings,
                                                   basic_list,
                                                   x,
                                                   delta_xB_trial,
                                                   leaving_index_trials[push],
                                                   basic_leaving_index_trials[push],
                                                   bound_trials[push]);
    }

    if (best == -1) { best = step_length_trials[0] > step_length_trials[1] ? 0 : 1; }
    assert(best != -1);

    f_t delta_xs                        = delta_xs_trials[best];
    sparse_vector_t<i_t, f_t>& delta_xB = delta_xB_trials[best];
    i_t leaving_index                   = leaving_index_trials[best];
    i_t basic_leaving_index             = basic_leaving_index_trials[best];
    f_t step_length                     = step_length_trials[best];
    i_t bound                           = bound_trials[best];

#ifdef CHECK_DIRECTION
    {
      std::vector<f_t> delta_x(n, 0.0);
      for (Int k = 0; k < m; ++k) {
        const Int j = basic_list[k];
        delta_x[j]  = delta_xB[k];
      }
      delta_x[s] = delta_xs;
      std::vector<f_t> nullspace(m);
      matrix_vector_multiply(lp.A, 1.0, delta_x, 0.0, nullspace);
      printf("|| A * dx || %e\n", vector_norm_inf(nullspace));
    }
#endif

    // xB <- xB + step_length * delta_xB
    for (i_t k = 0; k < delta_xB.i.size(); ++k) {
      const i_t j = basic_list[delta_xB.i[k]];
      x[j] += step_length * delta_xB.x[k];
    }
    // x_s <- x_s + step_length * delta_xs
    x[s] += step_length * delta_xs;

#ifdef COMPUTE_RESIDUAL
    // Compute r = b - A*x
    std::vector<f_t> residual = lp.rhs;
    matrix_vector_multiply(lp.A, -1.0, x, 1.0, residual);
    f_t primal_residual = vector_norm_inf<i_t, f_t>(residual);
#endif

    if (leaving_index != -1) {
      // Move superbasic variable into the basis
      vstatus[s]          = variable_status_t::BASIC;
      const f_t fixed_tol = settings.fixed_tol;
      if (std::abs(lp.lower[leaving_index] - lp.upper[leaving_index]) > fixed_tol) {
        vstatus[leaving_index] =
          bound == -1 ? variable_status_t::NONBASIC_LOWER : variable_status_t::NONBASIC_UPPER;
      } else {
        vstatus[leaving_index] = variable_status_t::NONBASIC_FIXED;
      }

      if (std::abs(z[s]) > 1e-6 ||
          (vstatus[leaving_index] == variable_status_t::NONBASIC_LOWER && z[leaving_index] < 0) ||
          (vstatus[leaving_index] == variable_status_t::NONBASIC_UPPER && z[leaving_index] > 0)) {
        settings.log.debug("Variable %d now basic with z %e. Variable %d now %d with z %e\n",
                           s,
                           z[s],
                           leaving_index,
                           static_cast<int>(vstatus[leaving_index]),
                           z[leaving_index]);
      }
      basic_list[basic_leaving_index] = s;
      nonbasic_list.push_back(leaving_index);

      // Refactor or Update
      bool should_refactor = ft.num_updates() > 100;
      if (!should_refactor) {
        sparse_vector_t<i_t, f_t> es_sparse(m, 1);
        es_sparse.i[0] = basic_leaving_index;
        es_sparse.x[0] = 1.0;
        sparse_vector_t<i_t, f_t> UTsol_sparse(m, 1);
        sparse_vector_t<i_t, f_t> solution_sparse(m, 1);
        ft.b_transpose_solve(es_sparse, solution_sparse, UTsol_sparse);
        i_t recommend_refactor = ft.update(utilde_sparse, UTsol_sparse, basic_leaving_index);
        should_refactor        = recommend_refactor == 1;
      }

      if (should_refactor) {
        csc_matrix_t<i_t, f_t> L(m, m, 1);
        csc_matrix_t<i_t, f_t> U(m, m, 1);
        std::vector<i_t> p(m);
        std::vector<i_t> pinv(m);
        std::vector<i_t> q(m);
        std::vector<i_t> deficient;
        std::vector<i_t> slacks_needed;
        i_t rank =
          factorize_basis(lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed);
        if (rank != m) {
          settings.log.debug("Failed to factorize basis. rank %d m %d\n", rank, m);
          basis_repair(
            lp.A, settings, deficient, slacks_needed, basic_list, nonbasic_list, vstatus);
          if (factorize_basis(
                lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed) == -1) {
            settings.log.printf("Failed to factorize basis after repair. rank %d m %d\n", rank, m);
            return -1;
          } else {
            settings.log.debug("Basis repaired\n");
          }
        }
        reorder_basic_list(q, basic_list);
        ft.reset(L, U, p);
      }
    } else {
      // Move superbasic variable into the nonbasic variables
      if (best == 0) {
        vstatus[s] = variable_status_t::NONBASIC_LOWER;
        nonbasic_list.push_back(s);
      } else {
        vstatus[s] = variable_status_t::NONBASIC_UPPER;
        nonbasic_list.push_back(s);
      }
    }

    // Remove superbasic variable
    superbasic_list.pop_back();

    num_pushes++;
    if (num_pushes % settings.iteration_log_frequency == 0 || toc(last_print_time) > 10.0 ||
        superbasic_list.size() == 0) {
      settings.log.printf(
        "%d of %d primal pushes in %.2f seconds\n", num_pushes, total_superbasics, toc(start_time));
      last_print_time = tic();
    }

    if (toc(start_time) > settings.time_limit) {
      settings.log.printf("Crossover time limit exceeded\n");
      return -1;
    }
    if (settings.concurrent_halt != nullptr && *settings.concurrent_halt == 1) {
      settings.log.printf("Concurrent halt\n");
      return -2;
    }
  }

  verify_basis<i_t, f_t>(m, n, vstatus);

  // Solve for xB such that B*xB = b - N*xN
  std::vector<f_t> rhs = lp.rhs;
  settings.log.debug("n %d m %d basic %ld nonbasic %ld n-m %d\n",
                     n,
                     m,
                     basic_list.size(),
                     nonbasic_list.size(),
                     n - m);
  assert(nonbasic_list.size() == n - m);
  for (i_t k = 0; k < n - m; ++k) {
    const i_t j         = nonbasic_list[k];
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      const i_t i = lp.A.i[p];
      rhs[i] -= lp.A.x[p] * solution.x[j];
    }
  }
  std::vector<f_t> xB(m);
  ft.b_solve(rhs, xB);
  std::vector<f_t> x_compare(n);
  for (i_t k = 0; k < m; ++k) {
    const i_t j  = basic_list[k];
    x_compare[j] = xB[k];
  }
  for (i_t k = 0; k < n - m; ++k) {
    const i_t j  = nonbasic_list[k];
    x_compare[j] = solution.x[j];
  }
  solution.x = x_compare;
  solution.iterations += num_pushes;
  return 0;
}

template <typename i_t, typename f_t>
i_t find_candidate_columns(const lp_problem_t<i_t, f_t>& lp,
                           const simplex_solver_settings_t<i_t, f_t>& settings,
                           const lp_solution_t<i_t, f_t>& solution,
                           std::vector<variable_status_t>& vstatus,
                           std::vector<i_t>& candidate_columns,
                           std::vector<bool>& column_included)
{
  const i_t n         = lp.num_cols;
  const i_t m         = lp.num_rows;
  f_t basis_threshold = 1e-10;
  while (candidate_columns.size() < m && basis_threshold < 1.0) {
    basis_threshold *= 10.0;
    for (i_t j = 0; j < n; ++j) {
      const f_t lower_bound_slack = solution.x[j] - lp.lower[j];
      const f_t upper_bound_slack = lp.upper[j] - solution.x[j];
      if (lower_bound_slack < basis_threshold && lp.lower[j] > -inf) {
        vstatus[j] = variable_status_t::NONBASIC_LOWER;
      } else if (upper_bound_slack < basis_threshold && lp.upper[j] < inf) {
        vstatus[j] = variable_status_t::NONBASIC_UPPER;
      } else if (solution.z[j] > basis_threshold && lp.lower[j] > -inf) {
        vstatus[j] = variable_status_t::NONBASIC_LOWER;
      } else if (solution.z[j] < -basis_threshold && lp.upper[j] < inf) {
        vstatus[j] = variable_status_t::NONBASIC_UPPER;
      } else if (std::abs(lp.lower[j] - lp.upper[j]) < 1e-7) {
        vstatus[j] = variable_status_t::NONBASIC_FIXED;
      } else {
        if (!column_included[j]) {
          candidate_columns.push_back(j);
          column_included[j] = true;
          vstatus[j]         = variable_status_t::SUPERBASIC;
        }
      }
    }
    settings.log.debug("basis threshold %e candidate columns %ld m %d\n",
                       basis_threshold,
                       candidate_columns.size(),
                       m);
  }
  return 0;
}

template <typename i_t, typename f_t>
i_t add_slacks_to_basis(const lp_problem_t<i_t, f_t>& lp,
                        const std::vector<i_t>& dependent_rows,
                        std::vector<variable_status_t>& vstatus)
{
  const i_t n   = lp.num_cols;
  i_t num_basic = 0;
  // Add a slack to the basis for each dependent row
  for (i_t i : dependent_rows) {
    for (i_t j = n - 1; j >= 0; --j) {
      const i_t col_start = lp.A.col_start[j];
      const i_t col_end   = lp.A.col_start[j + 1];
      const i_t nz        = col_end - col_start;
      if (nz == 1) {
        if (i == lp.A.i[col_start]) {
          vstatus[j] = variable_status_t::BASIC;
          num_basic++;
          break;
        }
      }
    }
  }
  return num_basic;
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
      x[j] = lp.lower[j];
      if (vstatus[j] != variable_status_t::NONBASIC_FIXED) {
        settings.log.debug("Setting fixed variable %d to %e. vstatus %d\n",
                           j,
                           lp.lower[j],
                           static_cast<int>(vstatus[j]));
      }
      vstatus[j] = variable_status_t::NONBASIC_FIXED;
    } else if (z[j] >= 0 && lp.lower[j] > -inf) {
      x[j] = lp.lower[j];
      if (vstatus[j] != variable_status_t::NONBASIC_LOWER) {
        settings.log.debug("Setting nonbasic lower variable %d to %e. vstatus %d\n",
                           j,
                           lp.lower[j],
                           static_cast<int>(vstatus[j]));
      }
      vstatus[j] = variable_status_t::NONBASIC_LOWER;
    } else if (z[j] <= 0 && lp.upper[j] < inf) {
      x[j] = lp.upper[j];
      if (vstatus[j] != variable_status_t::NONBASIC_UPPER) {
        settings.log.debug("Setting nonbasic upper variable %d to %e. vstatus %d\n",
                           j,
                           lp.upper[j],
                           static_cast<int>(vstatus[j]));
      }
      vstatus[j] = variable_status_t::NONBASIC_UPPER;
    } else if (lp.upper[j] == inf && lp.lower[j] > -inf && z[j] < 0) {
      // dual infeasible
      x[j] = lp.lower[j];
      if (vstatus[j] != variable_status_t::NONBASIC_LOWER) {
        settings.log.debug("Setting nonbasic lower variable %d to %e. vstatus %d\n",
                           j,
                           lp.lower[j],
                           static_cast<int>(vstatus[j]));
      }
      vstatus[j] = variable_status_t::NONBASIC_LOWER;
    } else if (lp.lower[j] == -inf && lp.upper[j] < inf && z[j] > 0) {
      // dual infeasible
      x[j] = lp.upper[j];
      if (vstatus[j] != variable_status_t::NONBASIC_UPPER) {
        settings.log.debug("Setting nonbasic upper variable %d to %e. vstatus %d\n",
                           j,
                           lp.upper[j],
                           static_cast<int>(vstatus[j]));
      }
      vstatus[j] = variable_status_t::NONBASIC_UPPER;
    } else if (lp.lower[j] == -inf && lp.upper[j] == inf) {
      x[j] = 0;  // Set nonbasic free variables to 0 this overwrites previous lines
      if (vstatus[j] != variable_status_t::NONBASIC_FREE) {
        settings.log.debug(
          "Setting free variable %d to %e. vstatus %d\n", j, 0, static_cast<int>(vstatus[j]));
      }
      vstatus[j] = variable_status_t::NONBASIC_FREE;
      settings.log.debug("Setting free variable %d as nonbasic at 0\n", j);
    } else {
      assert(1 == 0);
    }
  }
}

template <typename i_t, typename f_t>
void print_crossover_info(const lp_problem_t<i_t, f_t>& lp,
                          const simplex_solver_settings_t<i_t, f_t>& settings,
                          const std::vector<variable_status_t>& vstatus,
                          const lp_solution_t<i_t, f_t>& solution,
                          const std::string& prefix)
{
  const f_t primal_res    = primal_residual(lp, solution);
  const f_t dual_res      = dual_residual(lp, solution);
  const f_t primal_infeas = primal_infeasibility(lp, settings, vstatus, solution.x);
  const f_t dual_infeas   = dual_infeasibility(lp, settings, vstatus, solution.z);
  const f_t obj           = compute_objective(lp, solution.x);
  const f_t user_obj      = compute_user_objective(lp, obj);
  settings.log.printf("\n");
  settings.log.printf("%20s. Primal residual (abs): %.2e Primal infeasibility (abs): %.2e\n",
                      prefix.c_str(),
                      primal_res,
                      primal_infeas);
  settings.log.printf("%20s. Dual   residual (abs): %.2e Dual   infeasibility (abs): %.2e\n",
                      prefix.c_str(),
                      dual_res,
                      dual_infeas);
  settings.log.printf("%20s. Objective %+.8e\n", prefix.c_str(), user_obj);
  settings.log.printf("\n");
}

}  // namespace

template <typename i_t, typename f_t>
crossover_status_t crossover(const lp_problem_t<i_t, f_t>& lp,
                             const simplex_solver_settings_t<i_t, f_t>& settings,
                             const lp_solution_t<i_t, f_t>& initial_solution,
                             f_t start_time,
                             lp_solution_t<i_t, f_t>& solution,
                             std::vector<variable_status_t>& vstatus)
{
  const i_t m         = lp.num_rows;
  const i_t n         = lp.num_cols;
  f_t crossover_start = tic();

  settings.log.printf("\n");
  settings.log.printf("Starting crossover\n");

  vstatus.resize(n, variable_status_t::SUPERBASIC);
  std::vector<i_t> candidate_columns;
  candidate_columns.reserve(n);
  std::vector<bool> column_included(n, false);

  find_candidate_columns(
    lp, settings, initial_solution, vstatus, candidate_columns, column_included);

  std::vector<i_t> dependent_rows;
  std::vector<variable_status_t> vstatus_for_candidates(candidate_columns.size());
  settings.log.debug("m %d candidate columns %ld\n", m, candidate_columns.size());
  i_t rank = initial_basis_selection(
    lp, settings, candidate_columns, start_time, vstatus_for_candidates, dependent_rows);
  if (rank < 0) {
    settings.log.printf("Aborting: initial basis selection\n");
    return return_to_status(rank);
  }

  i_t num_basic = 0;
  if (rank < m) {
    num_basic = add_slacks_to_basis(lp, dependent_rows, vstatus);
    settings.log.debug("num basic %d from slacks\n", num_basic);
  }

  for (i_t k = 0; k < candidate_columns.size(); k++) {
    const i_t j = candidate_columns[k];
    vstatus[j]  = vstatus_for_candidates[k];
    if (vstatus[j] == variable_status_t::BASIC) { num_basic++; }
  }
  assert(num_basic == m);

  assert(initial_solution.x.size() == solution.x.size());
  assert(initial_solution.y.size() == solution.y.size());
  assert(initial_solution.z.size() == solution.z.size());

  solution.x          = initial_solution.x;
  solution.y          = initial_solution.y;
  solution.z          = initial_solution.z;
  solution.iterations = initial_solution.iterations;

  const f_t fixed_tolerance     = settings.fixed_tol;
  constexpr f_t basis_threshold = 1e-6;
  for (i_t j = 0; j < n; ++j) {
    if (vstatus[j] != variable_status_t::BASIC) {
      const f_t lower_bound_slack = initial_solution.x[j] - lp.lower[j];
      const f_t upper_bound_slack = lp.upper[j] - initial_solution.x[j];
      if (std::abs(lp.lower[j] - lp.upper[j]) < fixed_tolerance) {
        vstatus[j] = variable_status_t::NONBASIC_FIXED;
      } else if (solution.z[j] > -basis_threshold && lp.lower[j] > -inf) {
        vstatus[j] = variable_status_t::NONBASIC_LOWER;
      } else if (solution.z[j] < basis_threshold && lp.upper[j] < inf) {
        vstatus[j] = variable_status_t::NONBASIC_UPPER;
      }
    }
  }

  print_crossover_info(lp, settings, vstatus, solution, "Crossover start");

  std::vector<i_t> basic_list(m);
  std::vector<i_t> nonbasic_list;
  std::vector<i_t> superbasic_list;

  get_basis_from_vstatus(m, vstatus, basic_list, nonbasic_list, superbasic_list);
  verify_basis<i_t, f_t>(m, n, vstatus);
  compare_vstatus_with_lists<i_t, f_t>(m, n, basic_list, nonbasic_list, vstatus);
  settings.log.debug("basic list size %ld m %d\n", basic_list.size(), m);
  settings.log.debug("nonbasic list size %ld n - m %d\n", nonbasic_list.size(), n - m);
  settings.log.debug("superbasic list size %ld\n", superbasic_list.size());
  // Factorize the basis matrix B = A(:, basis_list). P*B*Q = L*U
  csc_matrix_t<i_t, f_t> L(m, m, 1);
  csc_matrix_t<i_t, f_t> U(m, m, 1);
  std::vector<i_t> p(m);
  std::vector<i_t> pinv(m);
  std::vector<i_t> q(m);
  std::vector<i_t> deficient;
  std::vector<i_t> slacks_needed;

  rank = factorize_basis(lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed);
  if (rank != m) {
    settings.log.debug("Failed to factorize basis. rank %d m %d\n", rank, m);
    basis_repair(lp.A, settings, deficient, slacks_needed, basic_list, nonbasic_list, vstatus);
    if (factorize_basis(lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed) ==
        -1) {
      settings.log.printf("Failed to factorize basis after repair. rank %d m %d\n", rank, m);
      return crossover_status_t::NUMERICAL_ISSUES;
    } else {
      settings.log.debug("Basis repaired\n");
    }
  }
  reorder_basic_list(q, basic_list);

  if (toc(start_time) > settings.time_limit) {
    settings.log.printf("Time limit exceeded\n");
    return crossover_status_t::TIME_LIMIT;
  }
  if (settings.concurrent_halt != nullptr && *settings.concurrent_halt == 1) {
    settings.log.printf("Concurrent halt\n");
    return crossover_status_t::CONCURRENT_LIMIT;
  }

  basis_update_mpf_t ft(L, U, p, settings.refactor_frequency);
  verify_basis<i_t, f_t>(m, n, vstatus);
  compare_vstatus_with_lists<i_t, f_t>(m, n, basic_list, nonbasic_list, vstatus);
  i_t dual_push_status = dual_push(
    lp, settings, start_time, solution, ft, basic_list, nonbasic_list, superbasic_list, vstatus);
  if (dual_push_status < 0) { return return_to_status(dual_push_status); }
  settings.log.debug("basic list size %ld m %d\n", basic_list.size(), m);
  settings.log.debug("nonbasic list size %ld n - m %d\n", nonbasic_list.size(), n - m);
  print_crossover_info(lp, settings, vstatus, solution, "Dual push complete");

  nonbasic_list.clear();
  superbasic_list.clear();

  for (i_t j = 0; j < n; ++j) {
    if (vstatus[j] != variable_status_t::BASIC) {
      const f_t lower_infeas      = lp.lower[j] - initial_solution.x[j];
      const f_t lower_bound_slack = initial_solution.x[j] - lp.lower[j];
      const f_t upper_infeas      = initial_solution.x[j] - lp.upper[j];
      const f_t upper_bound_slack = lp.upper[j] - initial_solution.x[j];
      if (std::abs(lp.lower[j] - lp.upper[j]) < fixed_tolerance) {
        vstatus[j] = variable_status_t::NONBASIC_FIXED;
        nonbasic_list.push_back(j);
      } else if (lower_infeas > 0 && lp.lower[j] > -inf) {
        vstatus[j]    = variable_status_t::NONBASIC_LOWER;
        solution.x[j] = lp.lower[j];
        nonbasic_list.push_back(j);
      } else if (upper_infeas > 0 && lp.upper[j] < inf) {
        vstatus[j]    = variable_status_t::NONBASIC_UPPER;
        solution.x[j] = lp.upper[j];
        nonbasic_list.push_back(j);
      } else if (lower_bound_slack < basis_threshold && lp.lower[j] > -inf) {
        vstatus[j] = variable_status_t::NONBASIC_LOWER;
        nonbasic_list.push_back(j);
      } else if (upper_bound_slack < basis_threshold && lp.upper[j] < inf) {
        vstatus[j] = variable_status_t::NONBASIC_UPPER;
        nonbasic_list.push_back(j);
      } else if (lp.lower[j] == -inf && lp.upper[j] == inf) {
        vstatus[j] = variable_status_t::NONBASIC_FREE;
        nonbasic_list.push_back(j);
      } else {
        vstatus[j] = variable_status_t::SUPERBASIC;
        superbasic_list.push_back(j);
      }
    }
  }

  if (superbasic_list.size() > 0) {
    std::vector<f_t> save_x = solution.x;
    i_t primal_push_status  = primal_push(
      lp, settings, start_time, solution, ft, basic_list, nonbasic_list, superbasic_list, vstatus);
    if (primal_push_status < 0) { return return_to_status(primal_push_status); }
    print_crossover_info(lp, settings, vstatus, solution, "Primal push complete");
  } else {
    settings.log.printf("No primal push needed. No superbasic variables\n");
  }

  f_t primal_infeas = primal_infeasibility(lp, settings, vstatus, solution.x);
  f_t dual_infeas   = dual_infeasibility(lp, settings, vstatus, solution.z);
  f_t obj           = compute_objective(lp, solution.x);
  f_t primal_res    = primal_residual(lp, solution);
  f_t dual_res      = dual_residual(lp, solution);

  const f_t primal_tol = settings.primal_tol;
  const f_t dual_tol   = settings.dual_tol;

  bool primal_feasible = primal_infeas <= primal_tol && primal_res <= primal_tol;
  bool dual_feasible   = dual_infeas <= dual_tol && dual_res <= dual_tol;

  if (primal_feasible && dual_feasible) {
    solution.objective      = compute_objective(lp, solution.x);
    solution.user_objective = compute_user_objective(lp, solution.objective);
    settings.log.printf("Skipping clean up phase\n");
  } else if (dual_feasible && !primal_feasible) {
    i_t dual_iter = 0;
    std::vector<f_t> edge_norms;
    dual::status_t status =
      dual_phase2(2, 0, start_time, lp, settings, vstatus, solution, dual_iter, edge_norms);
    if (toc(start_time) > settings.time_limit) {
      settings.log.printf("Time limit exceeded\n");
      return crossover_status_t::TIME_LIMIT;
    }
    if (settings.concurrent_halt != nullptr && *settings.concurrent_halt == 1) {
      settings.log.printf("Concurrent halt\n");
      return crossover_status_t::CONCURRENT_LIMIT;
    }
    primal_infeas = primal_infeasibility(lp, settings, vstatus, solution.x);
    dual_infeas   = dual_infeasibility(lp, settings, vstatus, solution.z);
    primal_res    = primal_residual(lp, solution);
    dual_res      = dual_residual(lp, solution);
    if (status != dual::status_t::OPTIMAL) {
      print_crossover_info(lp, settings, vstatus, solution, "Dual phase 2 complete");
    }
    solution.iterations += dual_iter;
    primal_feasible = primal_infeas <= primal_tol && primal_res <= primal_tol;
    dual_feasible   = dual_infeas <= dual_tol && dual_res <= dual_tol;
  } else {
    lp_problem_t<i_t, f_t> phase1_problem(lp.handle_ptr, 1, 1, 1);
    create_phase1_problem(lp, phase1_problem);
    std::vector<variable_status_t> phase1_vstatus(n);
    i_t num_basic_phase1    = 0;
    i_t num_nonbasic_phase1 = 0;
    for (i_t j = 0; j < n; ++j) {
      if (lp.lower[j] > -inf && lp.upper[j] < inf && vstatus[j] != variable_status_t::BASIC) {
        phase1_vstatus[j] = variable_status_t::NONBASIC_FIXED;
      } else {
        phase1_vstatus[j] = vstatus[j];
      }
      if (phase1_vstatus[j] == variable_status_t::BASIC) {
        num_basic_phase1++;
      } else {
        num_nonbasic_phase1++;
      }
    }
    settings.log.debug("num basic phase 1 %d (m %d) num nonbasic phase 1 %d (n - m %d)\n",
                       num_basic_phase1,
                       m,
                       num_nonbasic_phase1,
                       n - m);
    i_t iter = 0;
    lp_solution_t<i_t, f_t> phase1_solution(phase1_problem.num_rows, phase1_problem.num_cols);
    std::vector<f_t> junk;
    dual::status_t phase1_status = dual_phase2(
      1, 1, start_time, phase1_problem, settings, phase1_vstatus, phase1_solution, iter, junk);
    if (phase1_status == dual::status_t::NUMERICAL ||
        phase1_status == dual::status_t::DUAL_UNBOUNDED) {
      settings.log.printf("Failed in Phase 1\n");
      phase1_solution.objective = -std::numeric_limits<f_t>::infinity();
    }
    f_t phase1_obj = phase1_solution.objective;
    if (phase1_obj > -1e-3) {
      const f_t dual_tol           = settings.dual_tol;
      i_t num_changes_from_phase_1 = 0;
      for (i_t j = 0; j < n; ++j) {
        if (lp.lower[j] > -inf && lp.upper[j] < inf &&
            phase1_vstatus[j] != variable_status_t::BASIC) {
          // set vstatus from z value in phase 1
          if (std::abs(lp.upper[j] - lp.lower[j]) < settings.tight_tol) {
            vstatus[j] = variable_status_t::NONBASIC_FIXED;
          } else if (phase1_solution.z[j] >= dual_tol) {
            vstatus[j] = variable_status_t::NONBASIC_LOWER;
            num_changes_from_phase_1++;
          } else if (phase1_solution.z[j] <= -dual_tol) {
            vstatus[j] = variable_status_t::NONBASIC_UPPER;
            num_changes_from_phase_1++;
          } else {
            // -epsilon <= z[j] <= epsilon
            // We want to choose based on xl
            const f_t lower_slack = phase1_solution.x[j] - lp.lower[j];
            const f_t upper_slack = lp.upper[j] - phase1_solution.x[j];
            if (upper_slack < lower_slack) {
              vstatus[j] = variable_status_t::NONBASIC_UPPER;
              num_changes_from_phase_1++;
            } else {
              vstatus[j] = variable_status_t::NONBASIC_LOWER;
              num_changes_from_phase_1++;
            }
          }
        } else {
          // vstatus from phase 1
          vstatus[j] = phase1_vstatus[j];
        }
      }
      settings.log.debug("num changes from phase 1 %d\n", num_changes_from_phase_1);
      nonbasic_list.clear();
      superbasic_list.clear();
      get_basis_from_vstatus(m, vstatus, basic_list, nonbasic_list, superbasic_list);
      rank =
        factorize_basis(lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed);
      if (rank != m) {
        settings.log.debug("Failed to factorize basis. rank %d m %d\n", rank, m);
        basis_repair(lp.A, settings, deficient, slacks_needed, basic_list, nonbasic_list, vstatus);
        if (factorize_basis(
              lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed) == -1) {
          settings.log.printf("Failed to factorize basis after repair. rank %d m %d\n", rank, m);
          return crossover_status_t::NUMERICAL_ISSUES;
        } else {
          settings.log.debug("Basis repaired\n");
        }
      }
      reorder_basic_list(q, basic_list);
      ft.reset(L, U, p);

      compute_dual_solution_from_basis(lp, ft, basic_list, nonbasic_list, solution.y, solution.z);

      i_t num_flips = 0;
      for (i_t j = 0; j < n; ++j) {
        if (vstatus[j] == variable_status_t::BASIC) { continue; }
        if (lp.lower[j] > -inf && lp.upper[j] < inf) {
          if (vstatus[j] == variable_status_t::NONBASIC_LOWER &&
              solution.z[j] < -settings.dual_tol) {
            vstatus[j] = variable_status_t::NONBASIC_UPPER;
            num_flips++;
          } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER &&
                     solution.z[j] > settings.dual_tol) {
            vstatus[j] = variable_status_t::NONBASIC_LOWER;
            num_flips++;
          }
        }
      }
      settings.log.debug("Num flips %d\n", num_flips);
      solution = phase1_solution;
      print_crossover_info(lp, settings, vstatus, solution, "Dual phase 1 complete");
      std::vector<f_t> edge_norms;
      dual::status_t status = dual_phase2(
        2, iter == 0 ? 1 : 0, start_time, lp, settings, vstatus, solution, iter, edge_norms);
      if (toc(start_time) > settings.time_limit) {
        settings.log.printf("Time limit exceeded\n");
        return crossover_status_t::TIME_LIMIT;
      }
      if (settings.concurrent_halt != nullptr && *settings.concurrent_halt == 1) {
        settings.log.printf("Concurrent halt\n");
        return crossover_status_t::CONCURRENT_LIMIT;
      }
      solution.iterations += iter;
      primal_infeas = primal_infeasibility(lp, settings, vstatus, solution.x);
      dual_infeas   = dual_infeasibility(lp, settings, vstatus, solution.z);
      primal_res    = primal_residual(lp, solution);
      dual_res      = dual_residual(lp, solution);
      if (status != dual::status_t::OPTIMAL) {
        print_crossover_info(lp, settings, vstatus, solution, "Dual phase 2 complete");
      }
      primal_feasible = primal_infeas <= primal_tol && primal_res <= primal_tol;
      dual_feasible   = dual_infeas <= dual_tol && dual_res <= dual_tol;
    } else {
      settings.log.printf("Unable to find feasible solution in Phase 1\n");
    }
  }

  settings.log.printf("Crossover time %.2f seconds\n", toc(crossover_start));
  settings.log.printf("Total time %.2f seconds\n", toc(start_time));

  crossover_status_t status = crossover_status_t::NUMERICAL_ISSUES;
  if (dual_feasible) { status = crossover_status_t::DUAL_FEASIBLE; }
  if (primal_feasible) { status = crossover_status_t::PRIMAL_FEASIBLE; }
  if (primal_feasible && dual_feasible) { status = crossover_status_t::OPTIMAL; }
  return status;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template crossover_status_t crossover<int, double>(
  const lp_problem_t<int, double>& problem,
  const simplex_solver_settings_t<int, double>& settings,
  const lp_solution_t<int, double>& initial_solution,
  double start_time,
  lp_solution_t<int, double>& solution,
  std::vector<variable_status_t>& vstatus);

#endif

}  // namespace cuopt::linear_programming::dual_simplex
