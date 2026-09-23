/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <branch_and_bound/degenerate_pivots.hpp>
#include <branch_and_bound/fractional.hpp>

#include <dual_simplex/basis_solves.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/primal.hpp>
#include <dual_simplex/random.hpp>
#include <linear_algebra/vector_math.hpp>
#include <math_optimization/tic_toc.hpp>

#include <algorithm>

namespace cuopt::mathematical_optimization::mip {

using simplex::lp_solution_t;
using simplex::simplex_solver_settings_t;
using simplex::variable_status_t;
using simplex::variable_type_t;

template <typename i_t, typename f_t>
bool check_for_dual_degeneracy(const simplex::lp_solution_t<i_t, f_t>& solution,
                               const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                               const std::vector<i_t>& nonbasic_list,
                               std::vector<i_t>& zero_reduced_costs_vars,
                               std::vector<i_t>& zero_reduced_costs_vars_nonbasic_index)
{
  const i_t num_nonbasics = nonbasic_list.size();
  for (i_t k = 0; k < num_nonbasics; k++) {
    const i_t j = nonbasic_list[k];
    if (std::abs(solution.z[j]) <= settings.tight_tol) {
      zero_reduced_costs_vars.push_back(j);
      zero_reduced_costs_vars_nonbasic_index.push_back(k);
    }
  }
  return !zero_reduced_costs_vars.empty();
}

template <typename i_t, typename f_t>
void dual_degenerate_feasibility_pump(const simplex::lp_problem_t<i_t, f_t>& lp,
                                      const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                                      const std::vector<simplex::variable_type_t>& var_types,
                                      const std::vector<f_t>& edge_norms,
                                      f_t root_relax_work_estimate,
                                      f_t start_time,
                                      std::vector<i_t>& basic_list,
                                      std::vector<i_t>& nonbasic_list,
                                      std::vector<simplex::variable_status_t>& vstatus,
                                      simplex::lp_solution_t<i_t, f_t>& soln,
                                      simplex::basis_update_mpf_t<i_t, f_t>& basis_update,
                                      i_t& num_fractional,
                                      std::vector<i_t>& fractional)
{
  f_t dual_degenerate_feasibility_pump_start_time = tic();
  std::vector<i_t> zero_reduced_costs_vars;
  std::vector<i_t> zero_reduced_costs_vars_nonbasic_index;
  bool dual_degenerate = check_for_dual_degeneracy(
    soln, settings, nonbasic_list, zero_reduced_costs_vars, zero_reduced_costs_vars_nonbasic_index);
  if (!dual_degenerate) { return; }

  // Construct a new LP problem
  // minimize p^T x
  // subject to B x_B + N_z x_z = b - N x_N
  //            l_B <= x_B <= u_B
  //            l_z <= x_z <= u_z
  //
  // where B is the basic matrix, N is the nonbasic matrix, b is the right-hand side,

  const i_t m = lp.num_rows;
  const i_t n = lp.num_rows + zero_reduced_costs_vars.size();

  i_t nnz = 0;
  for (i_t j = 0; j < lp.num_cols; j++) {
    if (vstatus[j] == variable_status_t::BASIC || std::abs(soln.z[j]) <= settings.tight_tol) {
      nnz += lp.A.col_start[j + 1] - lp.A.col_start[j];
    }
  }
  simplex::lp_problem_t<i_t, f_t> lp_reduced(lp.handle_ptr, m, n, nnz);
  csc_matrix_t<i_t, f_t>& A_reduced = lp_reduced.A;
  std::vector<i_t> original_col_to_reduced_col(lp.num_cols, -1);
  i_t nz          = 0;
  i_t reduced_col = 0;
  for (i_t j = 0; j < lp.num_cols; j++) {
    if (vstatus[j] == variable_status_t::BASIC || std::abs(soln.z[j]) <= settings.tight_tol) {
      original_col_to_reduced_col[j]   = reduced_col;
      A_reduced.col_start[reduced_col] = nz;
      const i_t col_start              = lp.A.col_start[j];
      const i_t col_end                = lp.A.col_start[j + 1];
      for (i_t p = col_start; p < col_end; p++) {
        const i_t i     = lp.A.i[p];
        const f_t value = lp.A.x[p];
        A_reduced.i[nz] = i;
        A_reduced.x[nz] = value;
        nz++;
      }
      lp_reduced.lower[reduced_col] = lp.lower[j];
      lp_reduced.upper[reduced_col] = lp.upper[j];
      reduced_col++;
    }
  }
  A_reduced.col_start[reduced_col] = nz;

  std::vector<f_t> b_reduced = lp.rhs;
  for (i_t j = 0; j < lp.num_cols; j++) {
    if (vstatus[j] == variable_status_t::BASIC || std::abs(soln.z[j]) <= settings.tight_tol) {
      // PASS
    } else {
      const i_t col_start = lp.A.col_start[j];
      const i_t col_end   = lp.A.col_start[j + 1];
      for (i_t p = col_start; p < col_end; p++) {
        const i_t i     = lp.A.i[p];
        const f_t value = lp.A.x[p];
        b_reduced[i] -= value * soln.x[j];
      }
    }
  }
  lp_reduced.rhs       = b_reduced;
  lp_reduced.obj_scale = 1.0;

  settings.log.printf(
    "Constructed dual degenerate feasibility pump LP with %d rows and %d columns\n", m, n);

  std::vector<i_t> reduced_basic_list(m);
  std::vector<i_t> reduced_nonbasic_list(zero_reduced_costs_vars.size());
  std::vector<variable_status_t> reduced_vstatus(n);
  i_t num_basic    = 0;
  i_t num_nonbasic = 0;
  reduced_col      = 0;
  for (i_t j = 0; j < lp.num_cols; j++) {
    if (vstatus[j] == variable_status_t::BASIC) {
      reduced_vstatus[reduced_col++] = variable_status_t::BASIC;
    } else if (std::abs(soln.z[j]) <= 1e-10) {
      reduced_nonbasic_list[num_nonbasic++] =
        reduced_col;  // Does ordering of nonbasic variables matter?
      reduced_vstatus[reduced_col++] = vstatus[j];
    }
  }

  simplex::lp_solution_t<i_t, f_t> reduced_solution(m, n);
  reduced_col = 0;
  for (i_t j = 0; j < lp.num_cols; j++) {
    if (vstatus[j] == variable_status_t::BASIC || std::abs(soln.z[j]) <= settings.tight_tol) {
      reduced_solution.x[reduced_col++] = soln.x[j];
    }
  }

  std::vector<f_t> reduced_edge_norms(n);
  reduced_col = 0;
  for (i_t j = 0; j < lp.num_cols; j++) {
    if (vstatus[j] == variable_status_t::BASIC || std::abs(soln.z[j]) <= settings.tight_tol) {
      reduced_edge_norms[reduced_col++] = edge_norms[j];
    }
  }

  simplex::basis_update_mpf_t<i_t, f_t> reduced_basis_update = basis_update;
  reduced_basis_update.clear_work_estimate();
  for (i_t k = 0; k < m; k++) {
    reduced_basic_list[k] = original_col_to_reduced_col[basic_list[k]];
  }

  f_t primal_work_estimate = 0.0;
  i_t iter                 = 0;
  i_t max_pump_iter        = 10;
  simplex::random_t<i_t, f_t> rng(settings.random_seed);
  i_t best_num_fractional = num_fractional;
  std::vector<variable_status_t> best_reduced_vstatus(n);
  bool stalled = false;
  for (i_t pump_iter = 0; pump_iter < max_pump_iter; pump_iter++) {
    reduced_col = 0;
    for (i_t j = 0; j < lp.num_cols; j++) {
      if (vstatus[j] == variable_status_t::BASIC || std::abs(soln.z[j]) <= settings.tight_tol) {
        lp_reduced.objective[reduced_col] = 0;
        if (var_types[j] == variable_type_t::INTEGER) {
          if (is_fractional(reduced_solution.x[reduced_col], var_types[j], settings.integer_tol)) {
            // Default to the exact nearest-integer rounding. Only perturb the
            // rounding direction when the previous pass made no progress (a
            // zero-pivot solve), to break out of the stall.
            const f_t random_value =
              stalled ? 0.25 * (2.0 * rng.random() - 1.0) : 0.0;  // [-0.25, 0.25]
            if (reduced_solution.x[reduced_col] + random_value <
                std::floor(reduced_solution.x[reduced_col]) + 0.5) {
              lp_reduced.objective[reduced_col] = 1;
            } else {
              lp_reduced.objective[reduced_col] = -1;
            }
          } else if (reduced_vstatus[reduced_col] == variable_status_t::NONBASIC_LOWER) {
            lp_reduced.objective[reduced_col] = 0.1;
          } else if (reduced_vstatus[reduced_col] == variable_status_t::NONBASIC_UPPER) {
            lp_reduced.objective[reduced_col] = -0.1;
          }
        }
        reduced_col++;
      }
    }

    // Check reduced costs before calling primal simplex.
    // Compute y = B^{-T} * c_B (BTRAN with the pump objective on basic variables)
    std::vector<f_t> c_basic_pump(m, 0.0);
    for (i_t k = 0; k < m; k++) {
      c_basic_pump[k] = lp_reduced.objective[reduced_basic_list[k]];
    }
    std::vector<f_t> y_pump(m);
    reduced_basis_update.b_transpose_solve(c_basic_pump, y_pump);

    // Check if any nonbasic has a violated reduced cost
    i_t num_violated                = 0;
    f_t max_violation               = 0.0;
    const i_t num_nonbasics_reduced = reduced_nonbasic_list.size();
    for (i_t k = 0; k < num_nonbasics_reduced; k++) {
      const i_t j = reduced_nonbasic_list[k];
      // z[j] = c[j] - y^T * A(:,j)
      f_t zj              = lp_reduced.objective[j];
      const i_t col_start = A_reduced.col_start[j];
      const i_t col_end   = A_reduced.col_start[j + 1];
      for (i_t p = col_start; p < col_end; p++) {
        zj -= y_pump[A_reduced.i[p]] * A_reduced.x[p];
      }
      // Check pricing condition
      bool violated = false;
      if (reduced_vstatus[j] == variable_status_t::NONBASIC_LOWER ||
          reduced_vstatus[j] == variable_status_t::NONBASIC_FIXED) {
        if (zj < -settings.dual_tol) { violated = true; }
      } else if (reduced_vstatus[j] == variable_status_t::NONBASIC_UPPER) {
        if (zj > settings.dual_tol) { violated = true; }
      }
      if (violated) {
        num_violated++;
        max_violation = std::max(max_violation, std::abs(zj));
      }
    }

    if (num_violated == 0) {
      settings.log.printf(
        "Degenerate feasibility pump (%d/%d): skipping primal simplex, no violated reduced costs "
        "(%d nonbasics checked)\n",
        pump_iter,
        max_pump_iter,
        num_nonbasics_reduced);
      primal_work_estimate += reduced_basis_update.work_estimate();
      reduced_basis_update.clear_work_estimate();
      // Don't count this as a pump iteration, but break if we've skipped twice
      // in a row (perturbation isn't helping)
      if (stalled) { break; }
      stalled = true;
      pump_iter--;
      continue;
    }
    settings.log.printf(
      "Degenerate feasibility pump (%d/%d): %d violated reduced costs (max %.2e) out of %d "
      "nonbasics\n",
      pump_iter,
      max_pump_iter,
      num_violated,
      max_violation,
      num_nonbasics_reduced);

    bool recompute_basis                                = false;
    const i_t iter_before                               = iter;
    f_t primal_work_before                              = primal_work_estimate;
    f_t pump_call_start_time                            = tic();
    simplex_solver_settings_t<i_t, f_t> primal_settings = settings;
    primal_settings.log.log                             = false;
    primal_settings.time_limit                          = settings.time_limit;
    primal_settings.work_limit                          = root_relax_work_estimate / 10;
    settings.log.printf(
      "Degenerate feasibility pump: calling primal simplex with %d rows, %d cols, %d nnz, "
      "%d basis updates, work_limit %.2e, primal_work_estimate %.2e\n",
      m,
      n,
      A_reduced.col_start[n],
      reduced_basis_update.num_updates(),
      primal_settings.work_limit,
      primal_work_estimate);
    simplex::primal_status_t lp_status =
      simplex::primal_phase2_with_advanced_basis(2,
                                                 start_time,
                                                 lp_reduced,
                                                 primal_settings,
                                                 reduced_vstatus,
                                                 reduced_basis_update,
                                                 reduced_basic_list,
                                                 reduced_nonbasic_list,
                                                 reduced_solution,
                                                 iter,
                                                 primal_work_estimate);
    f_t pump_call_time  = toc(pump_call_start_time);
    f_t pump_call_work  = primal_work_estimate - primal_work_before;
    i_t pump_call_iters = iter - iter_before;
    settings.log.printf(
      "Degenerate feasibility pump: primal simplex returned status %d, %d iters, "
      "work %.2e (%.2e/iter), time %.2f (%.2e work/s)\n",
      static_cast<int>(lp_status),
      pump_call_iters,
      pump_call_work,
      pump_call_iters > 0 ? pump_call_work / pump_call_iters : 0.0,
      pump_call_time,
      pump_call_time > 0 ? pump_call_work / pump_call_time : 0.0);
    // Detect a stall: the solve made no pivots, so the incumbent vertex was
    // already optimal for this objective and x did not move. Perturb next pass.
    stalled = (iter == iter_before);

    if (lp_status == simplex::primal_status_t::OPTIMAL) {
      std::vector<f_t> adjusted_solution(lp.num_cols, 0.0);
      reduced_col = 0;
      for (i_t j = 0; j < lp.num_cols; j++) {
        if (vstatus[j] == variable_status_t::BASIC || std::abs(soln.z[j]) <= settings.tight_tol) {
          adjusted_solution[j] = reduced_solution.x[reduced_col++];
        } else {
          adjusted_solution[j] = soln.x[j];
        }
      }

      // Verify the solution is primal feasible
      std::vector<f_t> residual = lp.rhs;
      matrix_vector_multiply(lp.A, 1.0, adjusted_solution, -1.0, residual);
      const f_t primal_residual = vector_norm_inf<i_t, f_t>(residual);

      if (primal_residual > 1e-6) {
        settings.log.printf("Reduced LP residual|| A*x  - b ||_inf = %.4e\n", primal_residual);
      }

      std::vector<i_t> tmp_fractional;
      i_t num_fractional_reduced =
        fractional_variables(settings, adjusted_solution, var_types, tmp_fractional);
      settings.log.printf(
        "Degenerate feasibility pump (%d/%d): primal work estimate %.2e, iter %d, fractional "
        "variables %d/%d. Time %.2f\n",
        pump_iter,
        max_pump_iter,
        primal_work_estimate,
        iter,
        num_fractional_reduced,
        num_fractional,
        toc(dual_degenerate_feasibility_pump_start_time));
      // Also treat a pass that fails to improve the best as a stall, so we perturb
      // the next pass even when the solve pivoted (moved) without reducing the count.
      stalled = stalled || (num_fractional_reduced >= best_num_fractional);
      if (num_fractional_reduced < best_num_fractional) {
        best_num_fractional  = num_fractional_reduced;
        best_reduced_vstatus = reduced_vstatus;
      }
    } else {
      settings.log.printf(
        "Degenerate feasibility pump: primal simplex returned non-optimal status %d at pump_iter "
        "%d. Work estimate %.2e\n",
        static_cast<int>(lp_status),
        pump_iter,
        primal_work_estimate);
      // Even if we hit work/time limit, the solution may have improved.
      // Check fractional count before breaking.
      if (lp_status == simplex::primal_status_t::WORK_LIMIT ||
          lp_status == simplex::primal_status_t::TIME_LIMIT) {
        std::vector<f_t> adjusted_solution(lp.num_cols, 0.0);
        reduced_col = 0;
        for (i_t j = 0; j < lp.num_cols; j++) {
          if (vstatus[j] == variable_status_t::BASIC || std::abs(soln.z[j]) <= settings.tight_tol) {
            adjusted_solution[j] = reduced_solution.x[reduced_col++];
          } else {
            adjusted_solution[j] = soln.x[j];
          }
        }
        std::vector<f_t> residual = lp.rhs;
        matrix_vector_multiply(lp.A, 1.0, adjusted_solution, -1.0, residual);
        const f_t primal_residual = vector_norm_inf<i_t, f_t>(residual);
        if (primal_residual <= 1e-6) {
          std::vector<i_t> tmp_fractional;
          i_t num_fractional_reduced =
            fractional_variables(settings, adjusted_solution, var_types, tmp_fractional);
          settings.log.printf(
            "Degenerate feasibility pump (%d/%d): after work/time limit, fractional "
            "variables %d/%d\n",
            pump_iter,
            max_pump_iter,
            num_fractional_reduced,
            num_fractional);
          if (num_fractional_reduced < best_num_fractional) {
            best_num_fractional  = num_fractional_reduced;
            best_reduced_vstatus = reduced_vstatus;
          }
        }
      }
      break;
    }
  }

  settings.log.printf(
    "Degenerate feasibility pump: Simplex iterations %d, Best number of fractional variables "
    "%d/%d. Work estimate %.2e, Time %.2f, Basis updates %d\n",
    iter,
    best_num_fractional,
    num_fractional,
    primal_work_estimate,
    toc(dual_degenerate_feasibility_pump_start_time),
    reduced_basis_update.num_updates());
  if (best_num_fractional < num_fractional) {
    // Translate the vstatus from the reduced problem to the vstatus for the original problem
    i_t reduced_cols = 0;
    for (i_t j = 0; j < lp.num_cols; j++) {
      if (vstatus[j] == variable_status_t::BASIC || std::abs(soln.z[j]) <= settings.tight_tol) {
        vstatus[j] = best_reduced_vstatus[reduced_cols++];
      }
    }

    std::vector<i_t> superbasic_list;
    nonbasic_list.clear();
    simplex::get_basis_from_vstatus(m, vstatus, basic_list, nonbasic_list, superbasic_list);
    assert(superbasic_list.empty());
    i_t deficient_repaired    = 0;
    const i_t refactor_status = basis_update.refactor_basis(lp.A,
                                                            settings,
                                                            lp.lower,
                                                            lp.upper,
                                                            start_time,
                                                            basic_list,
                                                            nonbasic_list,
                                                            vstatus,
                                                            deficient_repaired);
    if (refactor_status == CONCURRENT_HALT_RETURN || refactor_status == TIME_LIMIT_RETURN) {
      return;
    }
    if (refactor_status != 0) {
      // TODO: On failure vstatus, basic_list, and nonbasic_list are in a bad state.
      // We should save copies before the failure and restore them after the failure.
      settings.log.printf(
        "Failed to refactor basis after dual degenerate feasibility pump. "
        "%d deficient columns.\n",
        refactor_status);
      return;
    }

    // Update the solution
    // First set the nonbasic variables on their bounds
    for (i_t k = 0; k < lp.num_cols - lp.num_rows; k++) {
      const i_t j = nonbasic_list[k];
      if (vstatus[j] == variable_status_t::NONBASIC_LOWER ||
          vstatus[j] == variable_status_t::NONBASIC_FIXED) {
        soln.x[j] = lp.lower[j];
      } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER) {
        soln.x[j] = lp.upper[j];
      } else {
        soln.x[j] = 0;
      }
    }
    // Then compute the effective rhs
    std::vector<f_t> rhs = lp.rhs;
    for (i_t j = 0; j < lp.num_cols; j++) {
      if (vstatus[j] == variable_status_t::BASIC) { continue; }
      const i_t col_start = lp.A.col_start[j];
      const i_t col_end   = lp.A.col_start[j + 1];

      const f_t x_j = soln.x[j];
      for (i_t p = col_start; p < col_end; p++) {
        const i_t i   = lp.A.i[p];
        const f_t aij = lp.A.x[p];
        rhs[i] -= aij * x_j;
      }
    }

    // Then solve B xB = rhs
    std::vector<f_t> xB(lp.num_rows);
    basis_update.b_solve(rhs, xB);

    // Then update the basic variables
    for (i_t k = 0; k < lp.num_rows; k++) {
      soln.x[basic_list[k]] = xB[k];
    }

    fractional.clear();
    num_fractional = fractional_variables(settings, soln.x, var_types, fractional);
  }
}

template <typename i_t, typename f_t>
i_t apply_delta_x_for_integer_pivot(const simplex::lp_problem_t<i_t, f_t>& lp,
                                    const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                                    i_t entering_index,
                                    i_t nonbasic_entering,
                                    i_t direction,
                                    const sparse_vector_t<i_t, f_t>& delta_x,
                                    const std::vector<simplex::variable_type_t>& var_types,
                                    f_t start_time,
                                    std::vector<i_t>& basic_list,
                                    std::vector<i_t>& nonbasic_list,
                                    std::vector<i_t>& nonbasic_index,
                                    std::vector<i_t>& variable_to_basic,
                                    std::vector<simplex::variable_status_t>& vstatus,
                                    sparse_vector_t<i_t, f_t>& utilde_sparse,
                                    simplex::lp_solution_t<i_t, f_t>& solution,
                                    simplex::basis_update_mpf_t<i_t, f_t>& basis_update,
                                    f_t& work_estimate)
{
  // Keep the existing dense ratio test and full integrality scan.
  std::vector<f_t> delta_x_dense;
  delta_x.to_dense(delta_x_dense);
  f_t step_length;
  i_t basic_leaving;
  const i_t leaving_index = simplex::primal_ratio_test(lp,
                                                       settings,
                                                       vstatus,
                                                       basic_list,
                                                       solution.x,
                                                       delta_x_dense,
                                                       step_length,
                                                       basic_leaving,
                                                       entering_index,
                                                       direction,
                                                       work_estimate);
  bool binding_integer =
    leaving_index != -1 &&
    is_fractional(solution.x[leaving_index], var_types[leaving_index], settings.integer_tol);
  if (!binding_integer) {
    if (leaving_index == -1) {
      return -4;  // unbounded or entering hit its own bound
    } else if (var_types[leaving_index] != variable_type_t::INTEGER) {
      return -5;  // continuous variable won ratio test
    } else {
      return -6;  // integer variable won but it's not fractional (already at integer value)
    }
  }

  std::vector<f_t> test_x = solution.x;
  i_t integer_destroyed   = 0;
  for (i_t h = 0; h < lp.num_cols; ++h) {
    test_x[h] += step_length * delta_x_dense[h];
    if (var_types[h] != variable_type_t::INTEGER) { continue; }
    const bool was_fractional = is_fractional(solution.x[h], var_types[h], settings.integer_tol);
    const bool now_fractional = is_fractional(test_x[h], var_types[h], settings.integer_tol);
    if (now_fractional && !was_fractional) {
      integer_destroyed++;
    } else if (!now_fractional && was_fractional) {
      integer_destroyed--;
    }
  }
  // Require a strict net decrease in fractional integers.
  if (integer_destroyed >= 0) { return -2; }

  if (utilde_sparse.i.empty()) {
    // Recover B^{-1} abar from the direction before changing the basis:
    // delta_x[basic_list[h]] = -direction * (B^{-1} abar)[h].
    // In MPF, utilde = U0 * (B^{-1} abar), since all updates are absorbed into L.
    sparse_vector_t<i_t, f_t> b_inv_abar(lp.num_rows, 0);
    b_inv_abar.i.reserve(delta_x.i.size());
    b_inv_abar.x.reserve(delta_x.x.size());
    const i_t nz = delta_x.i.size();
    for (i_t k = 0; k < nz; ++k) {
      const i_t h = variable_to_basic[delta_x.i[k]];
      if (h >= 0 && delta_x.x[k] != 0) {
        b_inv_abar.i.push_back(h);
        b_inv_abar.x.push_back(-direction * delta_x.x[k]);
      }
    }
    basis_update.u_multiply(b_inv_abar, utilde_sparse);
  }

  solution.x                        = test_x;
  basic_list[basic_leaving]         = entering_index;
  variable_to_basic[entering_index] = basic_leaving;
  variable_to_basic[leaving_index]  = -1;
  nonbasic_list[nonbasic_entering]  = leaving_index;
  vstatus[entering_index]           = variable_status_t::BASIC;
  if (std::abs(lp.upper[leaving_index] - lp.lower[leaving_index]) < 1e-12) {
    vstatus[leaving_index] = variable_status_t::NONBASIC_FIXED;
  } else if (delta_x_dense[leaving_index] < 0) {
    vstatus[leaving_index] = variable_status_t::NONBASIC_LOWER;
  } else {
    vstatus[leaving_index] = variable_status_t::NONBASIC_UPPER;
  }

  // Keep nonbasic_index consistent with nonbasic_list: entering_index is now basic,
  // and leaving_index has taken its slot in nonbasic_list.
  nonbasic_index[entering_index] = -1;
  nonbasic_index[leaving_index]  = nonbasic_entering;

  const i_t m = lp.num_rows;
  sparse_vector_t<i_t, f_t> es_sparse(m, 1);
  es_sparse.i[0] = basic_leaving;
  es_sparse.x[0] = 1.0;
  sparse_vector_t<i_t, f_t> UTsol_sparse(m, 1);
  sparse_vector_t<i_t, f_t> solution_sparse(m, 1);
  basis_update.b_transpose_solve(es_sparse, solution_sparse, UTsol_sparse);
  const i_t recommend_refactor = basis_update.update(utilde_sparse, UTsol_sparse, basic_leaving);
  if (recommend_refactor == 1) {
    csc_matrix_t<i_t, f_t> L(m, m, 1);
    csc_matrix_t<i_t, f_t> U(m, m, 1);
    std::vector<i_t> pinv(m);
    std::vector<i_t> p(m);
    std::vector<i_t> q(m);
    std::vector<i_t> deficient;
    std::vector<i_t> slacks_needed;
    f_t factorize_work_estimate = 0.0;
    const i_t rank              = factorize_basis(lp.A,
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
                                     factorize_work_estimate);
    if (rank == CONCURRENT_HALT_RETURN || rank == TIME_LIMIT_RETURN) { return -3; }
    if (rank < 0 || rank != lp.num_rows) { return -3; }
    simplex::reorder_basic_list(q, basic_list);
    for (i_t k = 0; k < m; ++k) {
      variable_to_basic[basic_list[k]] = k;
    }
    basis_update.reset(L, U, p);
  }

  return 0;
}

template <typename i_t, typename f_t>
void fast_slack_integer_pivots(const simplex::lp_problem_t<i_t, f_t>& lp,
                               const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                               const std::vector<i_t>& fractional,
                               const std::vector<i_t>& row_to_slack,
                               const simplex::lp_solution_t<i_t, f_t>& solution,
                               const std::vector<simplex::variable_type_t>& var_types,
                               f_t start_time,
                               std::vector<i_t>& basic_list,
                               std::vector<i_t>& nonbasic_list,
                               std::vector<i_t>& nonbasic_index,
                               std::vector<i_t>& variable_to_basic,
                               std::vector<simplex::variable_status_t>& vstatus,
                               simplex::lp_solution_t<i_t, f_t>& soln,
                               simplex::basis_update_mpf_t<i_t, f_t>& basis_update,
                               f_t& work_estimate)
{
  std::vector<i_t> fast_candidates;
  std::vector<i_t> fast_rows;
  std::vector<i_t> fast_nonbasic_slacks;
  for (i_t j : fractional) {
    const i_t col_start                            = lp.A.col_start[j];
    const i_t col_end                              = lp.A.col_start[j + 1];
    const i_t num_rows                             = col_end - col_start;
    i_t num_basic_slacks                           = 0;
    i_t num_nonbasic_slacks_with_reduced_cost_zero = 0;
    i_t nonbasic_slack                             = -1;
    i_t slack_row                                  = -1;
    for (i_t p = col_start; p < col_end; p++) {
      const i_t i     = lp.A.i[p];
      const i_t slack = row_to_slack[i];
      if (slack >= 0) {
        if (vstatus[slack] == variable_status_t::BASIC) {
          num_basic_slacks++;
        } else if (std::abs(solution.z[slack]) <= 1e-10) {
          num_nonbasic_slacks_with_reduced_cost_zero++;
          nonbasic_slack = slack;
          slack_row      = i;
        }
      }
    }
    if (num_basic_slacks == num_rows - 1 && num_nonbasic_slacks_with_reduced_cost_zero == 1) {
      fast_candidates.push_back(j);
      fast_rows.push_back(slack_row);
      fast_nonbasic_slacks.push_back(nonbasic_slack);
    }
  }

  if (fast_candidates.size() > 0 && settings.inside_mip < 2) {
    settings.log.printf("Found %ld fast candidates for pivot out integer variables\n",
                        fast_candidates.size());
  }

  // Build a reverse index nonbasic_index[v] = position of v in nonbasic_list, or -1 if not
  // present. Used to locate the entering variable's slot in the fast-candidate path.
  // apply_delta_x_for_integer_pivot keeps this index consistent by applying an O(1) fix-up
  // on each successful pivot; the two variables whose (non)basic status changes are the only
  // entries that need to be updated.
  nonbasic_index.assign(lp.num_cols, -1);
  for (i_t p = 0; p < static_cast<i_t>(nonbasic_list.size()); ++p) {
    nonbasic_index[nonbasic_list[p]] = p;
  }

  const i_t num_candidates = fast_candidates.size();
  f_t last_log             = tic();
  f_t loop_start           = tic();
  for (i_t k = 0; k < num_candidates; k++) {
    const i_t j              = fast_candidates[k];
    const i_t row            = fast_rows[k];
    const i_t nonbasic_slack = fast_nonbasic_slacks[k];
    // Skip if state changed by a prior successful pivot.
    if (vstatus[j] != variable_status_t::BASIC) { continue; }
    if (vstatus[nonbasic_slack] == variable_status_t::BASIC) { continue; }
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    f_t a_ij            = 0.0;
    for (i_t p = col_start; p < col_end; p++) {
      const i_t i = lp.A.i[p];
      if (i == row) {
        a_ij = lp.A.x[p];
        break;
      }
    }
    f_t bound = a_ij > 0 ? lp.lower[j] : lp.upper[j];
    if (std::abs(bound) == inf) { continue; }

    const f_t delta_xj = bound - soln.x[j];
    const f_t scale    = -delta_xj * a_ij;
    if (std::abs(scale) <= 1e-12) { continue; }

    // Build delta_x describing "move x[j] to its bound, let the basic slacks compensate to
    // keep A*x = b". This is a feasible direction (A*delta_x = 0). The nonzero pattern lives
    // on the entries of column A(:, j) plus j itself. In the nonbasic_slack slot,
    // delta_x[nonbasic_slack] = -delta_xj * a_ij > 0, i.e. the entering slack moves up from
    // its lower bound 0. We build the sparse version to feed the feasibility scan, then
    // normalize so that delta_x[nonbasic_slack] == 1 (the convention primal_ratio_test expects
    // for entering variables).
    sparse_vector_t<i_t, f_t> delta_x_sparse;
    delta_x_sparse.n = lp.num_cols;
    delta_x_sparse.i.reserve(col_end - col_start + 1);
    delta_x_sparse.x.reserve(col_end - col_start + 1);
    delta_x_sparse.i.push_back(j);
    delta_x_sparse.x.push_back(delta_xj);
    for (i_t p = col_start; p < col_end; p++) {
      const i_t r             = lp.A.i[p];
      const f_t a_rj          = lp.A.x[p];
      const f_t delta_slack_r = -delta_xj * a_rj;
      delta_x_sparse.i.push_back(row_to_slack[r]);
      delta_x_sparse.x.push_back(delta_slack_r);
    }

    // Reject if the full unit step would drive any basic slack below zero.
    bool ok       = true;
    const i_t ndx = delta_x_sparse.i.size();
    for (i_t h = 0; h < ndx; h++) {
      const i_t jj = delta_x_sparse.i[h];
      if (jj == j) continue;
      const f_t val         = delta_x_sparse.x[h];
      const f_t slack_value = soln.x[jj];
      if (val < -slack_value) {
        ok = false;
        break;
      }
    }
    if (!ok) { continue; }

    // Normalize so that delta_x[nonbasic_slack] == 1 (the standard entering-direction
    // convention). Done on the sparse vector, after the feasibility scan above, which reads
    // the unnormalized values.
    for (f_t& val : delta_x_sparse.x) {
      val /= scale;
    }

    // Entering variable is the nonbasic slack, moving up from its lower bound 0.
    const i_t entering_index    = nonbasic_slack;
    const i_t nonbasic_entering = nonbasic_index[nonbasic_slack];
    if (nonbasic_entering < 0) { continue; }
    const i_t direction = 1;

    // The common helper computes utilde only if the pivot is accepted.
    sparse_vector_t<i_t, f_t> utilde_sparse;

    i_t error = apply_delta_x_for_integer_pivot(lp,
                                                settings,
                                                entering_index,
                                                nonbasic_entering,
                                                direction,
                                                delta_x_sparse,
                                                var_types,
                                                start_time,
                                                basic_list,
                                                nonbasic_list,
                                                nonbasic_index,
                                                variable_to_basic,
                                                vstatus,
                                                utilde_sparse,
                                                soln,
                                                basis_update,
                                                work_estimate);
    // apply_delta_x_for_integer_pivot only mutates vstatus when the pivot actually fires,
    // so entering_index transitioning to BASIC is a reliable success signal.
    if (!error && settings.inside_mip < 2) {
      settings.log.printf(
        "Fast candidate pivot succeeded: j=%d entering slack=%d row=%d\n", j, entering_index, row);
    }

    if (settings.inside_mip < 2 && toc(last_log) > 1.0) {
      settings.log.printf("Fast candidates %d/%d processed in %.2f seconds\n",
                          k + 1,
                          num_candidates,
                          toc(loop_start));
      last_log = tic();
    }
  }
  if (settings.inside_mip < 2) {
    settings.log.printf("Fast candidates: %d/%d processed in %.2f seconds\n",
                        num_candidates,
                        num_candidates,
                        toc(loop_start));
  }
}

template <typename i_t, typename f_t>
i_t pivot_out_integer_variables(const simplex::lp_problem_t<i_t, f_t>& lp,
                                const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                                const std::vector<i_t>& new_slacks,
                                const std::vector<simplex::variable_type_t>& var_types,
                                f_t start_time,
                                std::vector<i_t>& basic_list,
                                std::vector<i_t>& nonbasic_list,
                                std::vector<simplex::variable_status_t>& vstatus,
                                simplex::lp_solution_t<i_t, f_t>& solution,
                                simplex::basis_update_mpf_t<i_t, f_t>& basis_update,
                                i_t& num_fractional,
                                std::vector<i_t>& fractional)
{
  if (num_fractional == 0) { return 0; }
  f_t pivot_out_integer_variables_start_time = tic();
  std::vector<i_t> zero_reduced_costs_vars;
  std::vector<i_t> zero_reduced_costs_vars_nonbasic_index;
  bool dual_degenerate = check_for_dual_degeneracy(solution,
                                                   settings,
                                                   nonbasic_list,
                                                   zero_reduced_costs_vars,
                                                   zero_reduced_costs_vars_nonbasic_index);
  if (!dual_degenerate) { return 0; }

  lp_solution_t<i_t, f_t> soln_copy                       = solution;
  std::vector<i_t> basic_list_copy                        = basic_list;
  std::vector<i_t> nonbasic_list_copy                     = nonbasic_list;
  std::vector<variable_status_t> vstatus_copy             = vstatus;
  simplex::basis_update_mpf_t<i_t, f_t> basis_update_copy = basis_update;

  const i_t start_num_fractional = num_fractional;

  const i_t num_zero_reduced_costs_vars = zero_reduced_costs_vars.size();

  std::vector<i_t> row_to_slack(lp.num_rows, -1);
  for (i_t j : new_slacks) {
    if (lp.lower[j] != 0 || lp.upper[j] != inf) { continue; }
    const i_t p             = lp.A.col_start[j];
    row_to_slack[lp.A.i[p]] = j;
  }

  f_t work_estimate = 0.0;

  // Count primal degenerate basic variables
  i_t num_degenerate            = 0;
  i_t num_degenerate_continuous = 0;
  i_t num_degenerate_integer    = 0;
  for (i_t k = 0; k < lp.num_rows; k++) {
    const i_t j              = basic_list_copy[k];
    const f_t slack_to_lower = soln_copy.x[j] - lp.lower[j];
    const f_t slack_to_upper = lp.upper[j] - soln_copy.x[j];
    if (slack_to_lower <= settings.primal_tol || slack_to_upper <= settings.primal_tol) {
      num_degenerate++;
      if (var_types[j] == variable_type_t::INTEGER) {
        num_degenerate_integer++;
      } else {
        num_degenerate_continuous++;
      }
    }
  }
  const f_t degeneracy_fraction = static_cast<f_t>(num_degenerate) / lp.num_rows;
  if (settings.inside_mip < 2 && settings.inside_submip == 0) {
    settings.log.printf(
      "Primal degeneracy: %d/%d basic variables are degenerate (%.1f%%), "
      "continuous=%d, integer=%d\n",
      num_degenerate,
      lp.num_rows,
      100.0 * degeneracy_fraction,
      num_degenerate_continuous,
      num_degenerate_integer);
  }

  // Skip pivot_out entirely if primal degeneracy is too high — the ratio test
  // will almost always be won by a degenerate variable, making pivots hopeless.
  if (degeneracy_fraction > 0.5) {
    if (settings.inside_mip < 2) {
      settings.log.printf("Skipping pivot_out_integer_variables: degeneracy %.1f%% > 50%%\n",
                          100.0 * degeneracy_fraction);
    }
    return 0;
  }

  std::vector<i_t> nonbasic_index;
  std::vector<i_t> variable_to_basic(lp.num_cols, -1);
  for (i_t k = 0; k < lp.num_rows; k++) {
    variable_to_basic[basic_list_copy[k]] = k;
  }
  fast_slack_integer_pivots(lp,
                            settings,
                            fractional,
                            row_to_slack,
                            solution,
                            var_types,
                            start_time,
                            basic_list_copy,
                            nonbasic_list_copy,
                            nonbasic_index,
                            variable_to_basic,
                            vstatus_copy,
                            soln_copy,
                            basis_update_copy,
                            work_estimate);

  std::vector<i_t> work_list = fractional;

  sparse_vector_t<i_t, f_t> ep;
  ep.n = lp.num_rows;
  ep.i.resize(1);
  ep.x.resize(1);
  ep.x[0] = 1.0;

  std::vector<f_t> delta_y_dense(lp.num_rows, 0.0);

  // Track which entering variables are actually tried (to detect duplication)
  std::vector<i_t> entering_tried_count(lp.num_cols, 0);

  i_t worklist_total_processed   = 0;
  i_t worklist_skipped           = 0;
  i_t worklist_btran_done        = 0;
  i_t worklist_ftran_done        = 0;
  i_t worklist_pivots_succeeded  = 0;
  i_t worklist_readded           = 0;
  f_t worklist_btran_time        = 0.0;
  f_t worklist_dot_time          = 0.0;
  f_t worklist_ftran_time        = 0.0;
  i_t worklist_no_candidates     = 0;  // target had no nonzero dot_q
  i_t worklist_ratio_test_fail   = 0;  // ratio test didn't pick a fractional integer (error -1)
  i_t worklist_net_increase_fail = 0;  // pivot would net-increase fractionals (error -2)
  i_t worklist_unbounded         = 0;  // entering hit its own bound or unbounded (error -4)
  i_t worklist_continuous_won    = 0;  // continuous variable won ratio test (error -5)
  i_t worklist_nonfrac_int_won   = 0;  // non-fractional integer won ratio test (error -6)

  f_t worklist_loop_start = tic();
  f_t worklist_last_log   = tic();

  while (!work_list.empty()) {
    const i_t j = work_list.back();
    const i_t p = variable_to_basic[j];
    work_list.pop_back();
    worklist_total_processed++;

    // Skip if j is no longer basic and fractional (may have been fixed by a prior pivot)
    if (p < 0) {
      worklist_skipped++;
      continue;
    }
    if (vstatus_copy[j] != variable_status_t::BASIC) {
      worklist_skipped++;
      continue;
    }
    if (!is_fractional(soln_copy.x[j], var_types[j], settings.integer_tol)) {
      worklist_skipped++;
      continue;
    }

    // We want to pivot variable j out of the basis.
    // We solve B^T * delta_y = e_p, where p is the position of j in the basis.
    // Or delta_y = B^{-T} e_p, or delta_y^T = e_p^T B^{-T}

    ep.i[0] = p;
    sparse_vector_t<i_t, f_t> delta_y_sparse;
    sparse_vector_t<i_t, f_t> UTsol_sparse;
    f_t btran_start = tic();
    basis_update_copy.b_transpose_solve(ep, delta_y_sparse, UTsol_sparse);
    worklist_btran_time += toc(btran_start);
    worklist_btran_done++;

    // Scatter delta_y_sparse into dense workspace for dot product computation
    const i_t delta_y_nz = delta_y_sparse.i.size();
    for (i_t h = 0; h < delta_y_nz; h++) {
      delta_y_dense[delta_y_sparse.i[h]] = delta_y_sparse.x[h];
    }

    // We also have that
    // B*delta_xB + N*delta_xN = 0
    // So delta_xB = -B^{-1} N * delta_xN
    // And delta_xB[p] = e_p^T * delta_xB = -e_p^T B^{-1} N * delta_xN
    //                                    = -delta_y^T N * delta_xN
    // Recall that delta_xN = e_q where q is the entering variables
    // So delta_xB[p] = -delta_y^T A(:, q)
    //
    // For p to be the leaving variable, we need it to be the binding
    // member in the ratio test
    // x_B + alpha * delta_xB >= l_B
    // x_B + alpha * delta_xB <= u_B
    //
    // Or alpha <= (l_B[p] - x_B[p]) / delta_xB[p] when delta_xB[p] < 0
    // Or alpha <= (u_B[p] - x_B[p]) / delta_xB[p] when delta_xB[p] > 0
    //
    // Thus, if we want to push x_B[p] up to u_B[p], we want
    // alpha = (u_B[p] - x_B[p]) / delta_xB[p] to be small
    // And if we want to push x_B[p] down to l_B[p], we want
    // alpha = (l_B[p] - x_B[p]) / delta_xB[p] to be small
    //
    // Or equivalently, we want delta_xB[p] to be large

    // Find top 3 candidates by merit = |dot_q| / nnz(A(:,q))
    // Large |dot_q| means the target moves a lot (small step to hit bound).
    // Small nnz means the FTRAN result is likely sparse, so fewer competing
    // basic variables will have nonzero delta_xB components to block the target.
    // Skip entering variables that have already been tried (and failed) by prior targets.
    f_t values[3]  = {0.0, 0.0, 0.0};
    i_t indices[3] = {-1, -1, -1};
    f_t dot_start  = tic();
    for (i_t q : zero_reduced_costs_vars) {
      if (var_types[q] == variable_type_t::INTEGER) { continue; }
      if (nonbasic_index[q] < 0) { continue; }
      if (entering_tried_count[q] > 0) { continue; }
      // Compute dot_q = delta_y^T * A(:, q) using dense delta_y
      const i_t col_start = lp.A.col_start[q];
      const i_t col_end   = lp.A.col_start[q + 1];
      const i_t col_nnz   = col_end - col_start;
      f_t dot_q           = 0.0;
      for (i_t pp = col_start; pp < col_end; pp++) {
        dot_q += delta_y_dense[lp.A.i[pp]] * lp.A.x[pp];
      }
      const f_t abs_dot_q = std::abs(dot_q);
      if (abs_dot_q <= 1e-12) { continue; }
      const f_t merit = abs_dot_q / static_cast<f_t>(col_nnz);

      if (merit > values[0]) {
        indices[2] = indices[1];
        values[2]  = values[1];
        indices[1] = indices[0];
        values[1]  = values[0];
        indices[0] = q;
        values[0]  = merit;
      } else if (merit > values[1]) {
        indices[2] = indices[1];
        values[2]  = values[1];
        indices[1] = q;
        values[1]  = merit;
      } else if (merit > values[2]) {
        indices[2] = q;
        values[2]  = merit;
      }
    }
    worklist_dot_time += toc(dot_start);

    if (indices[0] == -1) { worklist_no_candidates++; }

    // Try the top 3 candidates
    for (i_t h = 0; h < 3; h++) {
      if (indices[h] == -1) break;

      const i_t q                 = indices[h];
      const i_t entering_index    = q;
      const i_t nonbasic_entering = nonbasic_index[q];
      if (nonbasic_entering < 0) { continue; }
      entering_tried_count[q]++;

      // Determine direction based on entering variable's status
      const i_t direction = (vstatus_copy[q] == variable_status_t::NONBASIC_LOWER ||
                             vstatus_copy[q] == variable_status_t::NONBASIC_FIXED)
                              ? 1
                              : -1;

      //  Solve B * delta_xB = A(:, q) so utilde is valid for the MPF update.
      sparse_vector_t<i_t, f_t> rhs(lp.A, q);
      sparse_vector_t<i_t, f_t> delta_xB;
      sparse_vector_t<i_t, f_t> utilde_sparse;
      f_t ftran_start = tic();
      basis_update_copy.b_solve(rhs, delta_xB, utilde_sparse);
      worklist_ftran_time += toc(ftran_start);
      worklist_ftran_done++;

      sparse_vector_t<i_t, f_t> delta_x(lp.num_cols, 0);
      delta_x.i.reserve(delta_xB.i.size() + 1);
      delta_x.x.reserve(delta_xB.x.size() + 1);
      const i_t nz = delta_xB.i.size();
      for (i_t k = 0; k < nz; ++k) {
        delta_x.i.push_back(basic_list_copy[delta_xB.i[k]]);
        delta_x.x.push_back(-direction * delta_xB.x[k]);
      }
      delta_x.i.push_back(q);
      delta_x.x.push_back(direction);

      i_t error = apply_delta_x_for_integer_pivot(lp,
                                                  settings,
                                                  entering_index,
                                                  nonbasic_entering,
                                                  direction,
                                                  delta_x,
                                                  var_types,
                                                  start_time,
                                                  basic_list_copy,
                                                  nonbasic_list_copy,
                                                  nonbasic_index,
                                                  variable_to_basic,
                                                  vstatus_copy,
                                                  utilde_sparse,
                                                  soln_copy,
                                                  basis_update_copy,
                                                  work_estimate);

      if (error == -2) { worklist_net_increase_fail++; }
      if (error == -4) {
        worklist_unbounded++;
        worklist_ratio_test_fail++;
      }
      if (error == -5) {
        worklist_continuous_won++;
        worklist_ratio_test_fail++;
      }
      if (error == -6) {
        worklist_nonfrac_int_won++;
        worklist_ratio_test_fail++;
      }

      if (!error) {
        worklist_pivots_succeeded++;
#ifdef READD_TO_WORKLIST
        // We did a successful pivot; add fractional variables whose values changed to work list.
        std::vector<f_t> delta_x_dense;
        delta_x.to_dense(delta_x_dense);
        for (i_t k : fractional) {
          if (vstatus_copy[k] != variable_status_t::BASIC) { continue; }
          if (std::abs(delta_x_dense[k]) > settings.zero_tol) {
            work_list.push_back(k);
            worklist_readded++;
          }
        }
#endif
        break;
      }
    }

    // Clear dense workspace for next target
    for (i_t h = 0; h < delta_y_nz; h++) {
      delta_y_dense[delta_y_sparse.i[h]] = 0.0;
    }

    if (toc(worklist_last_log) > 1.0) {
      if (settings.inside_mip < 2) {
        settings.log.printf(
          "Worklist progress: %d/%d processed, %d pivots, %d ratio_fail (unb=%d cont=%d nfint=%d), "
          "%d net_inc_fail, %d no_cand, %.2f seconds\n",
          worklist_total_processed,
          static_cast<i_t>(fractional.size()),
          worklist_pivots_succeeded,
          worklist_ratio_test_fail,
          worklist_unbounded,
          worklist_continuous_won,
          worklist_nonfrac_int_won,
          worklist_net_increase_fail,
          worklist_no_candidates,
          toc(worklist_loop_start));
      }
      worklist_last_log = tic();
    }
  }

  // Count unique entering variables and duplication
  i_t unique_entering         = 0;
  i_t max_entering_count      = 0;
  i_t entering_tried_once     = 0;
  i_t entering_tried_multiple = 0;
  for (i_t q = 0; q < lp.num_cols; q++) {
    if (entering_tried_count[q] > 0) {
      unique_entering++;
      max_entering_count = std::max(max_entering_count, entering_tried_count[q]);
      if (entering_tried_count[q] == 1) {
        entering_tried_once++;
      } else {
        entering_tried_multiple++;
      }
    }
  }
  if (settings.inside_mip < 2) {
    settings.log.printf(
      "Worklist entering stats: unique=%d, tried_once=%d, tried_multiple=%d, "
      "max_count=%d, total_ftran=%d, duplication_ratio=%.1fx\n",
      unique_entering,
      entering_tried_once,
      entering_tried_multiple,
      max_entering_count,
      worklist_ftran_done,
      worklist_ftran_done / std::max(1.0, static_cast<double>(unique_entering)));

    settings.log.printf(
      "Worklist stats: processed=%d skipped=%d btran=%d ftran=%d pivots=%d readded=%d "
      "btran_time=%.2f dot_time=%.2f ftran_time=%.2f zero_rc_vars=%d "
      "no_candidates=%d ratio_test_fail=%d (unbounded=%d continuous_won=%d nonfrac_int_won=%d) "
      "net_increase_fail=%d\n",
      worklist_total_processed,
      worklist_skipped,
      worklist_btran_done,
      worklist_ftran_done,
      worklist_pivots_succeeded,
      worklist_readded,
      worklist_btran_time,
      worklist_dot_time,
      worklist_ftran_time,
      num_zero_reduced_costs_vars,
      worklist_no_candidates,
      worklist_ratio_test_fail,
      worklist_unbounded,
      worklist_continuous_won,
      worklist_nonfrac_int_won,
      worklist_net_increase_fail);
  }

  std::vector<i_t> new_fractional;
  const i_t num_new_fractional =
    fractional_variables(settings, soln_copy.x, var_types, new_fractional);
  if (num_new_fractional < start_num_fractional) {
    i_t num_integer_increased = start_num_fractional - num_new_fractional;
#if 0
    settings.log.printf("Pivoted out %d integer variables: %d -> %d in %.2f\n",
                         num_integer_increased,
                         start_num_fractional,
                         num_new_fractional,
                         toc(pivot_out_integer_variables_start_time));
#endif
    num_fractional = num_new_fractional;
    fractional     = new_fractional;
    basic_list     = basic_list_copy;
    nonbasic_list  = nonbasic_list_copy;
    vstatus        = vstatus_copy;
    basis_update   = basis_update_copy;
    solution       = soln_copy;
    return num_integer_increased;
  }
  return 0;
}

template bool check_for_dual_degeneracy<int, double>(
  const simplex::lp_solution_t<int, double>&,
  const simplex::simplex_solver_settings_t<int, double>&,
  const std::vector<int>&,
  std::vector<int>&,
  std::vector<int>&);

template void fast_slack_integer_pivots<int, double>(
  const simplex::lp_problem_t<int, double>&,
  const simplex::simplex_solver_settings_t<int, double>&,
  const std::vector<int>&,
  const std::vector<int>&,
  const simplex::lp_solution_t<int, double>&,
  const std::vector<simplex::variable_type_t>&,
  double,
  std::vector<int>&,
  std::vector<int>&,
  std::vector<int>&,
  std::vector<int>&,
  std::vector<simplex::variable_status_t>&,
  simplex::lp_solution_t<int, double>&,
  simplex::basis_update_mpf_t<int, double>&,
  double&);

template int pivot_out_integer_variables<int, double>(
  const simplex::lp_problem_t<int, double>&,
  const simplex::simplex_solver_settings_t<int, double>&,
  const std::vector<int>&,
  const std::vector<simplex::variable_type_t>&,
  double,
  std::vector<int>&,
  std::vector<int>&,
  std::vector<simplex::variable_status_t>&,
  simplex::lp_solution_t<int, double>&,
  simplex::basis_update_mpf_t<int, double>&,
  int&,
  std::vector<int>&);

template int apply_delta_x_for_integer_pivot<int, double>(
  const simplex::lp_problem_t<int, double>&,
  const simplex::simplex_solver_settings_t<int, double>&,
  int,
  int,
  int,
  const sparse_vector_t<int, double>&,
  const std::vector<simplex::variable_type_t>&,
  double,
  std::vector<int>&,
  std::vector<int>&,
  std::vector<int>&,
  std::vector<int>&,
  std::vector<simplex::variable_status_t>&,
  sparse_vector_t<int, double>&,
  simplex::lp_solution_t<int, double>&,
  simplex::basis_update_mpf_t<int, double>&,
  double&);

template void dual_degenerate_feasibility_pump<int, double>(
  const simplex::lp_problem_t<int, double>&,
  const simplex::simplex_solver_settings_t<int, double>&,
  const std::vector<simplex::variable_type_t>&,
  const std::vector<double>&,
  double,
  double,
  std::vector<int>&,
  std::vector<int>&,
  std::vector<simplex::variable_status_t>&,
  simplex::lp_solution_t<int, double>&,
  simplex::basis_update_mpf_t<int, double>&,
  int&,
  std::vector<int>&);

}  // namespace cuopt::mathematical_optimization::mip
