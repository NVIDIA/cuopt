/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/primal.hpp>

#include <dual_simplex/basis_solves.hpp>
#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/phase1.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/tic_toc.hpp>

namespace cuopt::linear_programming::dual_simplex {

namespace {

template <typename i_t, typename f_t>
void set_primal_variables_on_bounds(const lp_problem_t<i_t, f_t>& lp,
                                    const simplex_solver_settings_t<i_t, f_t>& settings,
                                    const std::vector<f_t>& z,
                                    std::vector<variable_status_t>& vstatus,
                                    std::vector<f_t>& x)
{
  const i_t n            = lp.num_cols;
  constexpr f_t diff_tol = 1e-6;
  for (i_t j = 0; j < n; ++j) {
    if (vstatus[j] == variable_status_t::BASIC) { continue; }
    if (vstatus[j] == variable_status_t::NONBASIC_FIXED) {
      if (std::abs(lp.lower[j] - x[j]) > diff_tol) {
        settings.log.debug("Changing x %d from %e to %e. Nonbasic fixed\n", j, x[j], lp.lower[j]);
      }
      x[j] = lp.lower[j];
    } else if (vstatus[j] == variable_status_t::NONBASIC_LOWER) {
      if (std::abs(lp.lower[j] - x[j]) > diff_tol) {
        settings.log.debug("Changing x %d from %e to %e. Nonbasic lower\n", j, x[j], lp.lower[j]);
      }
      x[j] = lp.lower[j];
    } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER) {
      if (std::abs(lp.upper[j] - x[j]) > diff_tol) {
        settings.log.debug("Changing x %d from %e to %e. Nonbasic upper\n", j, x[j], lp.upper[j]);
      }
      x[j] = lp.upper[j];
    } else if (vstatus[j] == variable_status_t::NONBASIC_FREE) {
      if (std::abs(x[j]) > diff_tol) {
        settings.log.debug("Changing x %d from %e to %e. Nonbasic free\n", j, x[j], 0.0);
      }
      x[j] = 0;  // Set nonbasic free variables to 0 this overwrites previous lines
    } else {
      assert(1 == 0);
    }
  }
}

template <typename i_t, typename f_t>
f_t dual_infeasibility(const lp_problem_t<i_t, f_t>& lp,
                       const std::vector<variable_status_t>& vstatus,
                       const std::vector<f_t>& z)
{
  const i_t n             = lp.num_cols;
  const i_t m             = lp.num_rows;
  i_t num_infeasible      = 0;
  f_t sum_infeasible      = 0.0;
  constexpr f_t tight_tol = 0;
  i_t lower_bound_inf     = 0;
  i_t upper_bound_inf     = 0;
  i_t free_inf            = 0;
  i_t non_basic_lower_inf = 0;
  i_t non_basic_upper_inf = 0;

  for (i_t j = 0; j < n; ++j) {
    if (lp.upper[j] == inf && lp.lower[j] > -inf && z[j] < -tight_tol) {
      // -inf < l_j <= x_j < inf, so need z_j > 0 to be feasible
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
      lower_bound_inf++;
    } else if (lp.lower[j] == -inf && lp.upper[j] < inf && z[j] > tight_tol) {
      // -inf < x_j <= u_j < inf, so need z_j < 0 to be feasible
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
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
    } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER && z[j] > tight_tol) {
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
      non_basic_upper_inf++;
    }
  }

  return sum_infeasible;
}

template <typename i_t, typename f_t>
i_t phase2_pricing(const lp_problem_t<i_t, f_t>& lp,
                   const std::vector<f_t>& z,
                   const std::vector<i_t>& nonbasic_list,
                   const std::vector<variable_status_t>& vstatus,
                   i_t& direction,
                   i_t& basic_entering,
                   f_t& dual_inf)
{
  const i_t m        = lp.num_rows;
  const i_t n        = lp.num_cols;
  i_t entering_index = -1;
  f_t max_infeas     = 0.0;
  dual_inf           = 0.0;
  for (i_t k = 0; k < n - m; ++k) {
    const i_t j            = nonbasic_list[k];
    constexpr f_t dual_tol = 1e-6;
    if (vstatus[j] == variable_status_t::NONBASIC_FIXED) { continue; }
    if ((vstatus[j] == variable_status_t::NONBASIC_LOWER ||
         vstatus[j] == variable_status_t::NONBASIC_FREE) &&
        z[j] < -dual_tol) {
      const f_t infeas = -z[j];
      dual_inf += infeas;
      if (max_infeas < infeas) {
        max_infeas     = infeas;
        basic_entering = k;
        entering_index = j;
        direction      = 1;
      }
    } else if ((vstatus[j] == variable_status_t::NONBASIC_UPPER ||
                vstatus[j] == variable_status_t::NONBASIC_FREE) &&
               z[j] > dual_tol) {
      const f_t infeas = z[j];
      dual_inf += infeas;
      if (max_infeas < infeas) {
        max_infeas     = infeas;
        basic_entering = k;
        entering_index = j;
        direction      = -1;
      }
    }
  }
  return entering_index;
}

template <typename i_t, typename f_t>
i_t ratio_test(const lp_problem_t<i_t, f_t>& lp,
               const std::vector<variable_status_t>& vstatus,
               const std::vector<i_t>& basic_list,
               std::vector<f_t>& x,
               std::vector<f_t>& delta_x,
               f_t& step_length,
               i_t& basic_leaving)
{
  const i_t m             = lp.num_rows;
  const i_t n             = lp.num_cols;
  basic_leaving           = -1;
  i_t leaving_index       = -1;
  f_t min_val             = inf;
  constexpr f_t pivot_tol = 1e-8;
  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    if (delta_x[j] == 0.0) { continue; }
    if (lp.lower[j] > -inf && x[j] >= lp.lower[j] && delta_x[j] < -pivot_tol) {
      // xj + step * delta_x[j] >= lp.lower[j]
      // step * delta_x[j] >= lp.lower[j] - x[j]
      // step <= (lp.lower[j] - x[j]) / delta_x[j], delta_x[j] < 0
      const f_t neum = lp.lower[j] - x[j];
      f_t ratio      = neum / delta_x[j];
      if (ratio < min_val) {
        min_val       = ratio;
        basic_leaving = k;
        leaving_index = j;
      }
    }
    if (lp.upper[j] < inf && x[j] <= lp.upper[j] && delta_x[j] > pivot_tol) {
      // xj + step * delta_x[j] <= lp.upper[j]
      // step * delta_x[j] <= lp.upper[j] - x[j]
      // step <= (lp.upper[j] - x[j]) / delta_x[j], delta_x[j] > 0
      const f_t neum = lp.upper[j] - x[j];
      f_t ratio      = neum / delta_x[j];
      if (ratio < min_val) {
        min_val       = ratio;
        basic_leaving = k;
        leaving_index = j;
      }
    }
  }
  step_length = min_val;
  return leaving_index;
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
      if (infeas > 1e-6) {
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
      if (infeas > 1e-6) {
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

}  // namespace

// Note this implementation of primal simplex is experimental
// It is meant only to serve as a method to remove the perturbation to the objective
// after dual simplex has found a primal feasible solution
// The implementation currently cycles. So is not enabled at this time.
template <typename i_t, typename f_t>
primal::status_t primal_phase2(i_t phase,
                               f_t start_time,
                               const lp_problem_t<i_t, f_t>& lp,
                               const simplex_solver_settings_t<i_t, f_t>& settings,
                               std::vector<variable_status_t>& vstatus,
                               lp_solution_t<i_t, f_t>& sol,
                               i_t& iter)
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
  std::vector<i_t> bound_info(n - m);

  std::vector<f_t>& x = sol.x;
  std::vector<f_t>& y = sol.y;
  std::vector<f_t>& z = sol.z;

  std::vector<f_t> incoming_x                     = x;
  std::vector<variable_status_t> incoming_vstatus = vstatus;

  settings.log.printf("Primal Simplex Phase %d\n", phase);
  settings.log.printf("Solving a problem with %d constraints %d variables %d nonzeros\n",
                      lp.num_rows,
                      lp.num_cols,
                      lp.A.col_start[lp.num_cols]);

  get_basis_from_vstatus(m, vstatus, basic_list, nonbasic_list, superbasic_list);
  assert(superbasic_list.size() == 0);
  assert(nonbasic_list.size() == n - m);

  // Compute L*U = A(p, basic_list)
  csc_matrix_t<i_t, f_t> L(m, m, 1);
  csc_matrix_t<i_t, f_t> U(m, m, 1);
  std::vector<i_t> pinv(m);
  std::vector<i_t> p(m);
  std::vector<i_t> q(m);
  std::vector<i_t> deficient;
  std::vector<i_t> slacks_needed;
  i_t rank =
    factorize_basis(lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed);
  if (rank != m) {
    settings.log.debug("Failed to factorize basis. rank %d m %d\n", rank, m);
    basis_repair(lp.A, settings, deficient, slacks_needed, basic_list, nonbasic_list, vstatus);
    if (factorize_basis(lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed) ==
        -1) {
      settings.log.printf("Failed to factorize basis after repair. rank %d m %d\n", rank, m);
      return primal::status_t::NUMERICAL;
    } else {
      settings.log.debug("Basis repaired\n");
    }
  }
  reorder_basic_list(q, basic_list);
  reorder_basic_list(q, basic_list);
  basis_update_t ft(L, U, p);

  std::vector<f_t> c_basic(m);
  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    c_basic[k]  = lp.objective[j];
  }

  // Solve B'*y = cB
  ft.b_transpose_solve(c_basic, y);
  settings.log.printf(
    "|| y || %e || cB || %e\n", vector_norm_inf<i_t, f_t>(y), vector_norm_inf<i_t, f_t>(c_basic));

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
  settings.log.printf("|| z || %e\n", vector_norm_inf<i_t, f_t>(z));

  set_primal_variables_on_bounds(lp, settings, z, vstatus, x);

  const f_t init_dual_inf = dual_infeasibility(lp, vstatus, z);
  settings.log.printf("Initial dual infeasibility %e\n", init_dual_inf);

  std::vector<f_t> rhs = lp.rhs;
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
  settings.log.printf("|| x || %e\n", vector_norm2<i_t, f_t>(x));

  std::vector<f_t> residual = lp.rhs;
  matrix_vector_multiply(lp.A, 1.0, x, -1.0, residual);
  f_t primal_residual = vector_norm_inf<i_t, f_t>(residual);
  if (primal_residual > 1e-6) { settings.log.printf("|| A*x - b || %e\n", primal_residual); }
  f_t primal_inf = primal_infeasibility(lp, settings, vstatus, x);
  settings.log.printf("Initial primal infeasibility %e\n", primal_inf);

  const i_t iter_limit = iter + 1000;
  std::vector<f_t> delta_y(m);
  std::vector<f_t> delta_z(n);
  std::vector<f_t> delta_x(n);

  settings.log.printf("Iter Objective       Primal inf  Dual Inf.  Step  Entering  Leaving\n");
  while (iter < iter_limit) {
    i_t nonbasic_entering = -1;
    f_t dual_inf;
    i_t direction;
    i_t entering_index =
      phase2_pricing(lp, z, nonbasic_list, vstatus, direction, nonbasic_entering, dual_inf);
    if (entering_index == -1) {
      f_t obj        = compute_objective(lp, x);
      f_t primal_inf = primal_infeasibility(lp, settings, vstatus, x);
      settings.log.printf(
        "Optimal solution found. Objective %e. Dual infeas %e. Primal "
        "infeasibility %e. Iterations %d\n",
        compute_user_objective(lp, obj),
        dual_inf,
        primal_inf,
        iter);
      return primal::status_t::OPTIMAL;
    }

    std::vector<f_t> scaled_delta_xB(m);
    std::vector<f_t> rhs(m);
    const i_t col_start = lp.A.col_start[entering_index];
    const i_t col_end   = lp.A.col_start[entering_index + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      rhs[lp.A.i[p]] = lp.A.x[p];
    }
    std::vector<f_t> utilde(m);
    ft.b_solve(rhs, scaled_delta_xB, utilde);

    for (i_t k = 0; k < m; ++k) {
      const i_t j = basic_list[k];
      delta_x[j]  = -direction * scaled_delta_xB[k];
    }
    for (i_t k = 0; k < n - m; ++k) {
      const i_t j = nonbasic_list[k];
      delta_x[j]  = 0.0;
    }
    delta_x[entering_index] = direction;

    std::vector<f_t> residual(m);
    matrix_vector_multiply(lp.A, 1.0, delta_x, 1.0, residual);
    f_t primal_step_err = vector_norm_inf<i_t, f_t>(residual);
    if (primal_step_err > 1e-3) { printf("|| A * dx || %e\n", primal_step_err); }

    i_t basic_leaving;
    f_t step_length;
    i_t leaving_index = ratio_test(lp, vstatus, basic_list, x, delta_x, step_length, basic_leaving);
    if (leaving_index == -1) {
      settings.log.printf("No leaving variable. Primal unbounded?\n");
      return primal::status_t::PRIMAL_UNBOUNDED;
    }
    assert(step_length >= 0.0);

    // Update the primal variables
    for (i_t j = 0; j < n; ++j) {
      x[j] += step_length * delta_x[j];
    }

    // Update the factorization
    ft.update(utilde, basic_leaving);

    // Update the basis
    basic_list[basic_leaving]        = entering_index;
    nonbasic_list[nonbasic_entering] = leaving_index;
    vstatus[entering_index]          = variable_status_t::BASIC;
    if (std::abs(lp.upper[leaving_index] - lp.lower[leaving_index]) < 1e-12) {
      vstatus[leaving_index] = variable_status_t::NONBASIC_FIXED;
    } else if (direction == 1) {
      vstatus[leaving_index] = variable_status_t::NONBASIC_LOWER;
    } else {
      vstatus[leaving_index] = variable_status_t::NONBASIC_UPPER;
    }

    // Solve for y such that B'*y = c_B
    for (i_t k = 0; k < m; ++k) {
      const i_t j = basic_list[k];
      c_basic[k]  = lp.objective[j];
    }
    ft.b_transpose_solve(y, c_basic);
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

    const f_t obj        = compute_objective(lp, x);
    const f_t primal_inf = primal_infeasibility(lp, settings, vstatus, x);
    settings.log.printf("%3d %.10e %.2e %.2e %.2e %d %d\n",
                        iter,
                        compute_user_objective(lp, obj),
                        primal_inf,
                        dual_inf,
                        step_length,
                        entering_index,
                        leaving_index);

    iter++;
  }

  if (iter == iter_limit) { return primal::status_t::ITERATION_LIMIT; }

  return primal::status_t::NUMERICAL;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template primal::status_t primal_phase2<int, double>(
  int phase,
  double start_time,
  const lp_problem_t<int, double>& lp,
  const simplex_solver_settings_t<int, double>& settings,
  std::vector<variable_status_t>& vstatus,
  lp_solution_t<int, double>& sol,
  int& iter);

#endif

}  // namespace cuopt::linear_programming::dual_simplex
