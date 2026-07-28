/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/primal.hpp>

#include <dual_simplex/basis_solves.hpp>
#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/phase1.hpp>
#include <dual_simplex/solve.hpp>
#include <math_optimization/tic_toc.hpp>

#include <algorithm>

namespace cuopt::mathematical_optimization::simplex {

namespace {

template <typename i_t, typename f_t>
void set_primal_variables_on_bounds(const lp_problem_t<i_t, f_t>& lp,
                                    const simplex_solver_settings_t<i_t, f_t>& settings,
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
                       const std::vector<f_t>& z,
                       f_t tight_tol,
                       i_t& num_infeasible)
{
  const i_t n             = lp.num_cols;
  num_infeasible          = 0;
  f_t sum_infeasible      = 0.0;
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
                   f_t dual_tol,
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
    const i_t j = nonbasic_list[k];
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
f_t primal_infeasibility(const lp_problem_t<i_t, f_t>& lp,
                         const simplex_solver_settings_t<i_t, f_t>& settings,
                         const std::vector<variable_status_t>& vstatus,
                         const std::vector<f_t>& x,
                         i_t& num_infeasible)
{
  const i_t n    = lp.num_cols;
  f_t primal_inf = 0;
  num_infeasible = 0;
  for (i_t j = 0; j < n; ++j) {
    // Nonbasics are pinned to a bound; only basics can be (legitimately) infeasible.
    if (vstatus[j] != variable_status_t::BASIC) { continue; }
    if (x[j] < lp.lower[j] - settings.primal_tol) {
      // x_j < l_j => -x_j > -l_j => -x_j + l_j > 0
      const f_t infeas = -x[j] + lp.lower[j];
      primal_inf += infeas;
      num_infeasible++;
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
    if (x[j] > lp.upper[j] + settings.primal_tol) {
      // x_j > u_j => x_j - u_j > 0
      const f_t infeas = x[j] - lp.upper[j];
      primal_inf += infeas;
      num_infeasible++;
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

template <typename i_t, typename f_t>
f_t primal_infeasibility(const lp_problem_t<i_t, f_t>& lp,
                         const simplex_solver_settings_t<i_t, f_t>& settings,
                         const std::vector<variable_status_t>& vstatus,
                         const std::vector<f_t>& x)
{
  i_t num_infeasible = 0;
  return primal_infeasibility(lp, settings, vstatus, x, num_infeasible);
}

template <typename i_t, typename f_t>
void compute_phase1_objective(const lp_problem_t<i_t, f_t>& lp,
                              const simplex_solver_settings_t<i_t, f_t>& settings,
                              const std::vector<variable_status_t>& vstatus,
                              const std::vector<f_t>& x,
                              std::vector<f_t>& objective)
{
  const i_t n = lp.num_cols;
  for (i_t j = 0; j < n; ++j) {
    if (vstatus[j] != variable_status_t::BASIC) {
      objective[j] = 0.0;
    } else if (x[j] < lp.lower[j] - settings.primal_tol) {
      objective[j] = -1.0;
    } else if (x[j] > lp.upper[j] + settings.primal_tol) {
      objective[j] = 1.0;
    } else {
      objective[j] = 0.0;
    }
  }
}

template <typename i_t, typename f_t>
void compute_delta_y(const basis_update_mpf_t<i_t, f_t>& basis_update,
                     i_t basic_leaving,
                     sparse_vector_t<i_t, f_t>& delta_y,
                     sparse_vector_t<i_t, f_t>& etilde)
{
  const i_t m = delta_y.n;
  sparse_vector_t<i_t, f_t> ei(m, 1);
  ei.i[0] = basic_leaving;
  ei.x[0] = 1.0;
  delta_y.clear();
  etilde.clear();
  basis_update.b_transpose_solve(ei, delta_y, etilde);
}

template <typename i_t, typename f_t>
void compute_delta_z(const csr_matrix_t<i_t, f_t>& Arow,
                     const std::vector<variable_status_t>& vstatus,
                     const sparse_vector_t<i_t, f_t>& delta_y,
                     std::vector<f_t>& delta_z)
{
  // A^T delta_y + delta_z = 0
  // delta_z = -A^T delta_y = - sum_i A(i, :) * delta_y_i
  std::fill(delta_z.begin(), delta_z.end(), 0.0);
  for (i_t k = 0; k < static_cast<i_t>(delta_y.i.size()); ++k) {
    const i_t i         = delta_y.i[k];
    const f_t delta_y_i = delta_y.x[k];
    const i_t row_start = Arow.row_start[i];
    const i_t row_end   = Arow.row_start[i + 1];
    for (i_t p = row_start; p < row_end; ++p) {
      const i_t j = Arow.j[p];
      if (vstatus[j] != variable_status_t::BASIC) { delta_z[j] -= Arow.x[p] * delta_y_i; }
    }
  }
}

template <typename f_t>
f_t compute_dual_step_length(f_t entering_reduced_cost, f_t pivot)
{
  assert(pivot != 0.0);
  return entering_reduced_cost / pivot;
}

template <typename i_t, typename f_t>
void update_y(f_t dual_step_length, const sparse_vector_t<i_t, f_t>& delta_y, std::vector<f_t>& y)
{
  for (i_t k = 0; k < static_cast<i_t>(delta_y.i.size()); ++k) {
    const i_t i = delta_y.i[k];
    y[i] += dual_step_length * delta_y.x[k];
  }
}

template <typename i_t, typename f_t>
void update_z(f_t dual_step_length,
              const std::vector<i_t>& nonbasic_list,
              i_t entering_index,
              const std::vector<f_t>& delta_z,
              std::vector<f_t>& z)
{
  for (i_t k = 0; k < static_cast<i_t>(nonbasic_list.size()); ++k) {
    const i_t j = nonbasic_list[k];
    z[j] += dual_step_length * delta_z[j];
  }
  z[entering_index] = 0.0;
}

template <typename i_t, typename f_t>
void compute_dual_variables(const lp_problem_t<i_t, f_t>& lp,
                            const simplex_solver_settings_t<i_t, f_t>& settings,
                            const std::vector<f_t>& objective,
                            const std::vector<i_t>& basic_list,
                            const std::vector<i_t>& nonbasic_list,
                            basis_update_mpf_t<i_t, f_t>& ft,
                            std::vector<f_t>& c_basic,
                            std::vector<f_t>& y,
                            std::vector<f_t>& z)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;
  // Solve for y such that B'*y = c_B
  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    c_basic[k]  = objective[j];
  }
  ft.b_transpose_solve(c_basic, y);
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
}

template <typename i_t, typename f_t>
void compute_basic_primal_variables(const lp_problem_t<i_t, f_t>& lp,
                                    const basis_update_mpf_t<i_t, f_t>& basis_update,
                                    const std::vector<i_t>& basic_list,
                                    const std::vector<i_t>& nonbasic_list,
                                    std::vector<f_t>& x)
{
  const i_t m          = lp.num_rows;
  const i_t n          = lp.num_cols;
  std::vector<f_t> rhs = lp.rhs;
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
  basis_update.b_solve(rhs, xB);
  for (i_t k = 0; k < m; ++k) {
    x[basic_list[k]] = xB[k];
  }
}

template <typename i_t, typename f_t>
f_t primal_constraint_residual(const lp_problem_t<i_t, f_t>& lp, const std::vector<f_t>& x)
{
  std::vector<f_t> residual = lp.rhs;
  matrix_vector_multiply(lp.A, 1.0, x, -1.0, residual);
  return vector_norm_inf<i_t, f_t>(residual);
}

}  // namespace


template <typename i_t, typename f_t>
i_t primal_ratio_test(const lp_problem_t<i_t, f_t>& lp,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      const std::vector<variable_status_t>& vstatus,
                      const std::vector<i_t>& basic_list,
                      std::vector<f_t>& x,
                      std::vector<f_t>& delta_x,
                      f_t& step_length,
                      i_t& basic_leaving,
                      i_t entering_index,
                      i_t direction)
{
  const i_t m             = lp.num_rows;
  const i_t n             = lp.num_cols;
  basic_leaving           = -1;
  i_t leaving_index       = -1;
  f_t min_val             = inf;
  f_t current_dx          = 0.0;
  constexpr f_t pivot_tol = 1e-8;

  // Entering variable can hit its opposite bound: limit step by that
  if (direction > 0 && lp.upper[entering_index] < inf) {
    const f_t limit = lp.upper[entering_index] - x[entering_index];
    if (limit >= 0 && limit < min_val) {
      min_val       = limit;
      leaving_index = -1;  // no basic leaves; will be handled by caller
      basic_leaving = -1;
    }
  } else if (direction < 0 && lp.lower[entering_index] > -inf) {
    const f_t limit = x[entering_index] - lp.lower[entering_index];
    if (limit >= 0 && limit < min_val) {
      min_val       = limit;
      leaving_index = -1;
      basic_leaving = -1;
    }
  }

  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    if (delta_x[j] == 0.0) { continue; }

    // Already below lower and moving back up: stop when we reach the lower bound.
    // Without this, phase I can take an unbounded step (false unbounded) or skip the
    // breakpoint of the piecewise phase-I objective and stall still infeasible.
    if (x[j] < lp.lower[j] && delta_x[j] > pivot_tol && lp.lower[j] > -inf) {
      const f_t ratio = (lp.lower[j] - x[j]) / delta_x[j];
      if (ratio >= 0 && ratio < min_val) {
        min_val       = ratio;
        basic_leaving = k;
        leaving_index = j;
        current_dx    = delta_x[j];
      }
    }
    // Already above upper and moving back down: stop when we reach the upper bound.
    if (x[j] > lp.upper[j] && delta_x[j] < -pivot_tol && lp.upper[j] < inf) {
      const f_t ratio = (lp.upper[j] - x[j]) / delta_x[j];
      if (ratio >= 0 && ratio < min_val) {
        min_val       = ratio;
        basic_leaving = k;
        leaving_index = j;
        current_dx    = -delta_x[j];
      }
    }

    if (lp.lower[j] > -inf && delta_x[j] < -pivot_tol) {
      // xj + step * delta_x[j] >= lp.lower[j]
      // step * delta_x[j] >= lp.lower[j] - x[j]
      // step <= (lp.lower[j] - x[j]) / delta_x[j], delta_x[j] < 0
      f_t neum = lp.lower[j] - x[j];
      // A basic sitting below its bound (within the primal tolerance) is on
      // the bound numerically, but gives a tiny negative ratio. Dropping it lets
      // the step run straight through the bound, so treat it as a zero-length
      // block. A genuine violation is left to the branches above, which stop at
      // the bound when the variable moves back toward it.
      if (neum > 0 && neum <= settings.primal_tol) { neum = 0.0; }
      f_t ratio = neum / delta_x[j];
      if (ratio >= 0 && ratio < min_val) {
        min_val       = ratio;
        basic_leaving = k;
        leaving_index = j;
        current_dx    = -delta_x[j];
      } else if (ratio >= 0 && ratio < min_val + 1e-9 && -delta_x[j] > current_dx) {
        min_val       = ratio;
        basic_leaving = k;
        leaving_index = j;
        current_dx    = -delta_x[j];
      }
    }
    if (lp.upper[j] < inf && delta_x[j] > pivot_tol) {
      // xj + step * delta_x[j] <= lp.upper[j]
      // step * delta_x[j] <= lp.upper[j] - x[j]
      // step <= (lp.upper[j] - x[j]) / delta_x[j], delta_x[j] > 0
      f_t neum = lp.upper[j] - x[j];
      // Mirror of the lower bound case: slightly above the bound is considered on the bound.
      if (neum < 0 && -neum <= settings.primal_tol) { neum = 0.0; }
      f_t ratio = neum / delta_x[j];
      if (ratio >= 0 && ratio < min_val) {
        min_val       = ratio;
        basic_leaving = k;
        leaving_index = j;
        current_dx    = delta_x[j];
      } else if (ratio >= 0 && ratio < min_val + 1e-9 && delta_x[j] > current_dx) {
        min_val       = ratio;
        basic_leaving = k;
        leaving_index = j;
        current_dx    = delta_x[j];
      }
    }
  }
  step_length = min_val;
  return leaving_index;
}


template <typename i_t, typename f_t>
primal_status_t primal_phase2(i_t phase,
                              f_t start_time,
                              const lp_problem_t<i_t, f_t>& lp,
                              const simplex_solver_settings_t<i_t, f_t>& settings,
                              std::vector<variable_status_t>& vstatus,
                              lp_solution_t<i_t, f_t>& sol,
                              i_t& iter)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;

  f_t work_estimate = 0;
  std::vector<i_t> basic_list(m);
  std::vector<i_t> nonbasic_list;
  std::vector<i_t> superbasic_list;

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
  i_t rank = factorize_basis(lp.A,
                             settings,
                             basic_list,
                             start_time,
                             L,
                             U,
                             p,
                             pinv,
                             q,
                             deficient,
                             slacks_needed,
                             work_estimate);
  if (rank == CONCURRENT_HALT_RETURN) {
    settings.log.printf("Concurrent halt in primal phase2\n");
    return primal_status_t::CONCURRENT_LIMIT;
  } else if (rank == TIME_LIMIT_RETURN) {
    return primal_status_t::TIME_LIMIT;
  } else if (rank < 0) {
    return toc(start_time) > settings.time_limit ? primal_status_t::TIME_LIMIT
                                                 : primal_status_t::NUMERICAL;
  } else if (rank != m) {
    settings.log.debug("Failed to factorize basis. rank %d m %d\n", rank, m);
    basis_repair(lp.A,
                 settings,
                 lp.lower,
                 lp.upper,
                 deficient,
                 slacks_needed,
                 basic_list,
                 nonbasic_list,
                 superbasic_list,
                 vstatus,
                 work_estimate);
    rank = factorize_basis(lp.A,
                           settings,
                           basic_list,
                           start_time,
                           L,
                           U,
                           p,
                           pinv,
                           q,
                           deficient,
                           slacks_needed,
                           work_estimate);
    if (rank == CONCURRENT_HALT_RETURN) {
      return primal_status_t::CONCURRENT_LIMIT;
    } else if (rank == TIME_LIMIT_RETURN) {
      return primal_status_t::TIME_LIMIT;
    } else if (rank < 0) {
      settings.log.printf("Failed to factorize basis after repair. rank %d m %d\n", rank, m);
      return toc(start_time) > settings.time_limit ? primal_status_t::TIME_LIMIT
                                                   : primal_status_t::NUMERICAL;
    } else {
      settings.log.debug("Basis repaired\n");
    }
  }
  reorder_basic_list(q, basic_list);
  basis_update_mpf_t<i_t, f_t> ft(L, U, p, settings.refactor_frequency);

  return primal_phase2_with_advanced_basis(phase,
                                           start_time,
                                           lp,
                                           settings,
                                           vstatus,
                                           ft,
                                           basic_list,
                                           nonbasic_list,
                                           sol,
                                           iter,
                                           work_estimate);
}
// Note this implementation of primal simplex is experimental
// It is meant only to serve as a method to remove the perturbation to the objective
// after dual simplex has found a primal feasible solution
template <typename i_t, typename f_t>
primal_status_t primal_phase2_with_advanced_basis(
  i_t phase,
  f_t start_time,
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  std::vector<variable_status_t>& vstatus,
  basis_update_mpf_t<i_t, f_t>& basis_update,
  std::vector<i_t>& basic_list,
  std::vector<i_t>& nonbasic_list,
  lp_solution_t<i_t, f_t>& sol,
  i_t& iter,
  f_t& work_estimate,
  bool print_summary)
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

  std::vector<f_t>& x = sol.x;
  std::vector<f_t>& y = sol.y;
  std::vector<f_t>& z = sol.z;

  std::vector<f_t> incoming_x                     = x;
  std::vector<variable_status_t> incoming_vstatus = vstatus;
  settings.log.printf("Primal Simplex\n");
  // Nonbasics must be on their bounds before forming B x_B = b - A_N x_N.
  // Setting them after the solve leaves ||A*x - b|| large whenever x_N != 0.
  set_primal_variables_on_bounds(lp, settings, vstatus, x);

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
  basis_update.b_solve(rhs, xB);

  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    x[j]        = xB[k];
  }
  constexpr bool print_norms = false;
  if constexpr (print_norms) {
    settings.log.printf("|| x || %e\n", vector_norm2<i_t, f_t>(x));
  }

  std::vector<f_t> residual = lp.rhs;
  matrix_vector_multiply(lp.A, 1.0, x, -1.0, residual);
  f_t primal_residual = vector_norm_inf<i_t, f_t>(residual);
  if (primal_residual > settings.primal_tol) {
    settings.log.printf("|| A*x - b || %e\n", primal_residual);
  }
 
 
  std::vector<f_t> objective = lp.objective;
  const f_t primal_tol       = settings.primal_tol;
  f_t primal_inf = primal_infeasibility(lp, settings, vstatus, x);
  if (primal_inf > primal_tol) {
    // We are primal infeasible. Switch to phase 1
    compute_phase1_objective(lp, settings, vstatus, x, objective);
    settings.log.printf("Phase 1\n");
    settings.log.printf("Initial primal infeasibility %e\n", primal_inf);
    phase = 1;
  } else {
    settings.log.printf("Phase 2\n");
    phase = 2;
  }

  std::vector<f_t> c_basic(m);
  compute_dual_variables(
    lp, settings, objective, basic_list, nonbasic_list, basis_update, c_basic, y, z);
  if constexpr (print_norms) {
    settings.log.printf("|| z || %e\n", vector_norm_inf<i_t, f_t>(z));
  }

  i_t num_dual_inf        = 0;
  i_t num_primal_inf      = 0;
  const f_t init_dual_inf =
    dual_infeasibility(lp, vstatus, z, settings.dual_tol, num_dual_inf);
  if (num_dual_inf > 0) {
    settings.log.printf("Initial dual infeasibility %e\n", init_dual_inf);
  }

  csr_matrix_t<i_t, f_t> Arow(m, n, lp.A.nnz());
  lp.A.to_compressed_row(Arow);

  const i_t iter_limit = settings.iteration_limit;
  const i_t start_iter = iter;
  sparse_vector_t<i_t, f_t> delta_y(m, 0);
  sparse_vector_t<i_t, f_t> etilde(m, 0);
  std::vector<f_t> delta_z(n);
  std::vector<f_t> delta_x(n);

  f_t dual_inf = init_dual_inf;
  f_t obj      = compute_objective(lp, x);
  f_t pricing_dual_tol = settings.dual_tol;
  primal_inf           = primal_infeasibility(lp, settings, vstatus, x, num_primal_inf);
  settings.log.printf(" Iter     Objective           Num Inf.  Sum Inf.       Time\n");
  settings.log.printf("%5d %+.16e %7d %.8e %.2f\n",
                      iter,
                      compute_user_objective(lp, obj),
                      phase == 1 ? num_primal_inf : num_dual_inf,
                      phase == 1 ? primal_inf : dual_inf,
                      toc(start_time));
  bool switched_phase = false;
  while (iter < iter_limit) {
    i_t nonbasic_entering = -1;
    i_t direction;
    i_t entering_index = phase2_pricing(
      lp, z, nonbasic_list, vstatus, pricing_dual_tol, direction, nonbasic_entering, dual_inf);
    if (entering_index == -1) {
      if (phase == 2) {
        // Verify optimality with a consistent basic solution: refactor, put
        // nonbasics exactly on their status bounds, rebuild x_B so Ax = b, and
        // refresh duals. If that point is not primal/dual feasible, continue.
        if (basis_update.num_updates() > 0) {
          i_t rank = basis_update.refactor_basis(
            lp.A, settings, lp.lower, lp.upper, start_time, basic_list, nonbasic_list, vstatus);
          if (rank == CONCURRENT_HALT_RETURN) { return primal_status_t::CONCURRENT_LIMIT; }
          if (rank == TIME_LIMIT_RETURN) { return primal_status_t::TIME_LIMIT; }
          if (rank != 0) {
            settings.log.printf("Failed to refactor basis at optimality check. Iteration %d\n",
                                iter);
            return primal_status_t::NUMERICAL;
          }
          work_estimate = basis_update.work_estimate();
        }
        set_primal_variables_on_bounds(lp, settings, vstatus, x);
        compute_basic_primal_variables(lp, basis_update, basic_list, nonbasic_list, x);
        compute_dual_variables(
          lp, settings, objective, basic_list, nonbasic_list, basis_update, c_basic, y, z);
        primal_inf = primal_infeasibility(lp, settings, vstatus, x, num_primal_inf);
        dual_inf =
          dual_infeasibility(lp, vstatus, z, pricing_dual_tol, num_dual_inf);
        if (primal_inf > primal_tol) {
          compute_phase1_objective(lp, settings, vstatus, x, objective);
          phase            = 1;
          pricing_dual_tol = settings.dual_tol;
          compute_dual_variables(
            lp, settings, objective, basic_list, nonbasic_list, basis_update, c_basic, y, z);
          settings.log.printf(
            "Switching to Primal Simplex Phase 1 after optimality refresh. "
            "Primal infeasibility %e\n",
            primal_inf);
          settings.log.printf(" Iter     Objective           Num Inf.  Sum Inf.       Time\n");
          switched_phase = true;
          continue;
        }
        if (num_dual_inf > 0) {
          // The refreshed reduced costs contain a candidate visible at the active
          // pricing tolerance.
          continue;
        }

        i_t num_tight_dual_inf = 0;
        const f_t tight_dual_inf =
          dual_infeasibility(lp, vstatus, z, f_t(0.0), num_tight_dual_inf);
        if (tight_dual_inf > settings.dual_tol) {
          // No candidate is visible at the active pricing tolerance, but the
          // zero-tolerance residual is still material. Try tighter pricing before
          // accepting optimality. This is needed for problems such as cycle,
          // where many small reduced-cost violations lead to improving pivots.
          f_t retry_dual_tol = pricing_dual_tol;
          f_t retry_dual_inf = 0.0;
          i_t retry_entering = -1;
          while (retry_entering == -1 && retry_dual_tol > f_t(1e-10)) {
            retry_dual_tol *= f_t(0.1);
            retry_entering = phase2_pricing(lp,
                                            z,
                                            nonbasic_list,
                                            vstatus,
                                            retry_dual_tol,
                                            direction,
                                            nonbasic_entering,
                                            retry_dual_inf);
          }
          if (retry_entering != -1) {
            pricing_dual_tol = retry_dual_tol;
            continue;
          }
        }
        // Report the unfiltered residual at the accepted solution.
        dual_inf       = tight_dual_inf;
        num_dual_inf   = num_tight_dual_inf;
        obj                = compute_objective(lp, x);
        sol.objective      = obj;
        sol.user_objective = compute_user_objective(lp, obj);
        if (!settings.inside_mip && print_summary) {
          settings.log.printf("\n");
          settings.log.printf(
            "Optimal solution found in %d iterations and %.2fs\n", iter, toc(start_time));
          settings.log.printf("Objective %+.8e\n", sol.user_objective);
          settings.log.printf("\n");
          settings.log.printf("Primal infeasibility (abs): %.2e\n", primal_inf);
          settings.log.printf("Dual infeasibility   (abs): %.2e\n", dual_inf);
          settings.log.printf("Primal residual ||Ax-b||:   %.2e\n",
                              primal_constraint_residual(lp, x));
        }
        return primal_status_t::OPTIMAL;
      } else {
        primal_inf = primal_infeasibility(lp, settings, vstatus, x, num_primal_inf);

        if (primal_inf > primal_tol) {
          // Incremental duals may be stale relative to the current phase-I
          // objective. Refresh objective and duals, then retry pricing with
          // successively tighter dual tolerances.
          settings.log.printf("Refreshing phase-I objective and duals. Num updates %d. Iter %d\n", 
            basis_update.num_updates(), iter);
          compute_phase1_objective(lp, settings, vstatus, x, objective);
          compute_dual_variables(
            lp, settings, objective, basic_list, nonbasic_list, basis_update, c_basic, y, z);
          f_t retry_dual_tol = pricing_dual_tol;
          while (entering_index == -1 && retry_dual_tol > f_t(1e-10)) {
            retry_dual_tol *= f_t(0.1);
            settings.log.printf("Retrying phase-I pricing with dual_tol %e\n", retry_dual_tol);
            entering_index = phase2_pricing(lp,
                                            z,
                                            nonbasic_list,
                                            vstatus,
                                            retry_dual_tol,
                                            direction,
                                            nonbasic_entering,
                                            dual_inf);
          }
          if (entering_index == -1) {
            settings.log.printf(
              "Numerical issues encountered. No entering variable found with large "
              "infeasibility %e (%d).\n",
              primal_inf,
              num_primal_inf);
            return primal_status_t::NUMERICAL;
          }
          pricing_dual_tol = retry_dual_tol;
        } else {
          // Restore the objective to the original objective
          objective        = lp.objective;
          phase            = 2;
          pricing_dual_tol = settings.dual_tol;
          settings.log.printf(
            "Primal phase I complete. Iterations %d. Time %.2f\n", iter, toc(start_time));
          settings.log.printf(" Iter     Objective           Num Inf.  Sum Inf.       Time\n");
          compute_dual_variables(
            lp, settings, objective, basic_list, nonbasic_list, basis_update, c_basic, y, z);
          obj      = compute_objective(lp, x);
          dual_inf =
            dual_infeasibility(lp, vstatus, z, settings.dual_tol, num_dual_inf);
          iter++;
          // Print here: continue may hit dual-optimal Phase 2 and return before
          // the end-of-loop log checks switched_phase.
          settings.log.printf("%5d %+.16e %7d %.8e %.2f\n",
                              iter,
                              compute_user_objective(lp, obj),
                              num_dual_inf,
                              dual_inf,
                              toc(start_time));
          continue;
        }
      }
    }

    sparse_vector_t<i_t, f_t> rhs_sparse(lp.A, entering_index);
    sparse_vector_t<i_t, f_t> scaled_delta_xB_sparse(m, 0);
    sparse_vector_t<i_t, f_t> utilde_sparse(m, 0);
    basis_update.b_solve(rhs_sparse, scaled_delta_xB_sparse, utilde_sparse);
    std::vector<f_t> scaled_delta_xB(m);
    scaled_delta_xB_sparse.to_dense(scaled_delta_xB);

    for (i_t k = 0; k < m; ++k) {
      const i_t j = basic_list[k];
      delta_x[j]  = -direction * scaled_delta_xB[k];
    }
    for (i_t k = 0; k < n - m; ++k) {
      const i_t j = nonbasic_list[k];
      delta_x[j]  = 0.0;
    }
    delta_x[entering_index] = direction;

#ifdef CHECK_NULLSPACE
    std::vector<f_t> residual(m, 0.0);
    matrix_vector_multiply(lp.A, 1.0, delta_x, 0.0, residual);
    f_t primal_step_err = vector_norm_inf<i_t, f_t>(residual);
    if (primal_step_err > 1e-3) {
      settings.log.printf("|| A * dx || %e at iter %d (updates %d)\n",
                          primal_step_err,
                          iter,
                          basis_update.num_updates());
    }
#endif

    i_t basic_leaving;
    f_t step_length;
    i_t leaving_index = primal_ratio_test(lp,
                                          settings,
                                          vstatus,
                                          basic_list,
                                          x,
                                          delta_x,
                                          step_length,
                                          basic_leaving,
                                          entering_index,
                                          direction);
    if (leaving_index == -1 && step_length >= inf) {
      settings.log.printf("No leaving variable. Primal unbounded?\n");
      return primal_status_t::PRIMAL_UNBOUNDED;
    }

    const bool basis_updated = (leaving_index != -1);
    bool recompute_duals     = false;
    for (i_t j = 0; j < n; ++j) {
      x[j] += step_length * delta_x[j];
    }

#ifdef COMPUTE_RESIDUAL
    f_t debug_primal_residual = primal_constraint_residual(lp, x);
    if (debug_primal_residual > 1e-6) {
      settings.log.printf("|| A * x - b || %e at iteration %d (updates %d)\n", debug_primal_residual, iter, basis_update.num_updates());
    }
#endif 


    if (basis_updated) {
      assert(step_length >= 0.0);

      bool should_refactor = basis_update.num_updates() > settings.refactor_frequency;
      f_t dual_step_length = 0.0;
      if (!should_refactor) {
        compute_delta_y(basis_update, basic_leaving, delta_y, etilde);
        const f_t pivot  = scaled_delta_xB[basic_leaving];
        dual_step_length = compute_dual_step_length(z[entering_index], pivot);
      }

      basic_list[basic_leaving]        = entering_index;
      nonbasic_list[nonbasic_entering] = leaving_index;
      vstatus[entering_index]          = variable_status_t::BASIC;
      // Place the leaver on its leaving bound. If that bound is far from the
      // current value (typical after a zero-step leave of an already-infeasible
      // basic), rebuild x_B after the factor matches the new basis so Ax = b;
      // phase handling below may then (re)enter Phase I if basics are infeasible.
      bool rebuild_x_after_bound_snap = false;
      f_t leave_bound                 = 0.0;
      if (std::abs(lp.upper[leaving_index] - lp.lower[leaving_index]) < 1e-12) {
        vstatus[leaving_index] = variable_status_t::NONBASIC_FIXED;
        leave_bound            = lp.lower[leaving_index];
      } else {
        // Classify by which bound was hit. Using sign(delta_x) is wrong when the
        // variable approached the bound from the infeasible side (phase I).
        const f_t x_leave       = x[leaving_index];
        const f_t dist_to_lower = std::abs(x_leave - lp.lower[leaving_index]);
        const f_t dist_to_upper = std::abs(x_leave - lp.upper[leaving_index]);
        if (lp.lower[leaving_index] > -inf &&
            (lp.upper[leaving_index] >= inf || dist_to_lower <= dist_to_upper)) {
          vstatus[leaving_index] = variable_status_t::NONBASIC_LOWER;
          leave_bound            = lp.lower[leaving_index];
        } else {
          vstatus[leaving_index] = variable_status_t::NONBASIC_UPPER;
          leave_bound            = lp.upper[leaving_index];
        }
      }
      if (std::abs(x[leaving_index] - leave_bound) > settings.primal_tol) {
        rebuild_x_after_bound_snap = true;
      }
      x[leaving_index] = leave_bound;

      if (!should_refactor) {
        compute_delta_z(Arow, vstatus, delta_y, delta_z);
        update_y(dual_step_length, delta_y, y);
        update_z(dual_step_length, nonbasic_list, entering_index, delta_z, z);
        should_refactor = basis_update.update(utilde_sparse, etilde, basic_leaving) == 1;
      }
      if (should_refactor) {
        i_t rank = basis_update.refactor_basis(
          lp.A, settings, lp.lower, lp.upper, start_time, basic_list, nonbasic_list, vstatus);
        if (rank == CONCURRENT_HALT_RETURN) { return primal_status_t::CONCURRENT_LIMIT; }
        if (rank == TIME_LIMIT_RETURN) { return primal_status_t::TIME_LIMIT; }
        if (rank != 0) {
          settings.log.printf("Failed to refactor basis. Iteration %d\n", iter);
          return primal_status_t::NUMERICAL;
        }
        work_estimate   = basis_update.work_estimate();
        recompute_duals = true;
        // Factor matches basic_list: rebuild x_B so Ax = b exactly.
        set_primal_variables_on_bounds(lp, settings, vstatus, x);
        compute_basic_primal_variables(lp, basis_update, basic_list, nonbasic_list, x);
      } else if (rebuild_x_after_bound_snap) {
        // FT update already matches the new basis; recompute x_B with the leaving variable
        // snapped onto its bound.
        compute_basic_primal_variables(lp, basis_update, basic_list, nonbasic_list, x);
      }
    } else {
      if (direction > 0) {
        vstatus[entering_index] = variable_status_t::NONBASIC_UPPER;
        x[entering_index]       = lp.upper[entering_index];
      } else {
        vstatus[entering_index] = variable_status_t::NONBASIC_LOWER;
        x[entering_index]       = lp.lower[entering_index];
      }
    }

    primal_inf = primal_infeasibility(lp, settings, vstatus, x, num_primal_inf);
    if (primal_inf > primal_tol) {
      if (phase != 1) {
        settings.log.printf(
          "Switching to Primal Simplex Phase 1. Iteration %d. Primal infeasibility %e\n",
          iter,
          primal_inf);
        settings.log.printf(" Iter     Objective           Num Inf.  Sum Inf.       Time\n");
        switched_phase = true;
      }
      compute_phase1_objective(lp, settings, vstatus, x, objective);
      phase           = 1;
      recompute_duals = true;
    } else if (phase == 1) {
      objective        = lp.objective;
      phase            = 2;
      pricing_dual_tol = settings.dual_tol;
      recompute_duals  = true;
      settings.log.printf(
        "Primal phase I complete. Iterations %d. Time %.2f\n", iter, toc(start_time));
      settings.log.printf(" Iter     Objective           Num Inf.  Sum Inf.       Time\n");
      switched_phase = true;
    }

    if (recompute_duals) {
      compute_dual_variables(
        lp, settings, objective, basic_list, nonbasic_list, basis_update, c_basic, y, z);
    }

    obj      = compute_objective(lp, x);
    dual_inf =
      dual_infeasibility(lp, vstatus, z, pricing_dual_tol, num_dual_inf);

    iter++;

    f_t now = toc(start_time);
    if (0|| (iter - start_iter) < settings.first_iteration_log ||
        (iter % settings.iteration_log_frequency) == 0 || switched_phase) {
      const f_t user_obj = compute_user_objective(lp, obj);
      settings.log.printf("%5d %+.16e %7d %.8e %.2f\n",
                          iter,
                          user_obj,
                          phase == 1 ? num_primal_inf : num_dual_inf,
                          phase == 1 ? primal_inf : dual_inf,
                          now);
      switched_phase = false;
    }
  }

  if (iter == iter_limit) { return primal_status_t::ITERATION_LIMIT; }

  return primal_status_t::NUMERICAL;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template
int primal_ratio_test(const lp_problem_t<int, double>& lp,
                      const simplex_solver_settings_t<int, double>& settings,
                      const std::vector<variable_status_t>& vstatus,
                      const std::vector<int>& basic_list,
                      std::vector<double>& x,
                      std::vector<double>& delta_x,
                      double& step_length,
                      int& basic_leaving,
                      int entering_index,
                      int direction);

template primal_status_t primal_phase2<int, double>(
  int phase,
  double start_time,
  const lp_problem_t<int, double>& lp,
  const simplex_solver_settings_t<int, double>& settings,
  std::vector<variable_status_t>& vstatus,
  lp_solution_t<int, double>& sol,
  int& iter);

template primal_status_t primal_phase2_with_advanced_basis<int, double>(
  int phase,
  double start_time,
  const lp_problem_t<int, double>& lp,
  const simplex_solver_settings_t<int, double>& settings,
  std::vector<variable_status_t>& vstatus,
  basis_update_mpf_t<int, double>& basis_update,
  std::vector<int>& basic_list,
  std::vector<int>& nonbasic_list,
  lp_solution_t<int, double>& sol,
  int& iter,
  double& work_estimate,
  bool print_summary);

#endif

}  // namespace cuopt::mathematical_optimization::simplex
