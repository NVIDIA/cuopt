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

#include <dual_simplex/solve.hpp>

#include <dual_simplex/barrier.hpp>
#include <dual_simplex/branch_and_bound.hpp>
#include <dual_simplex/crossover.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/phase1.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/primal.hpp>
#include <dual_simplex/scaling.hpp>
#include <dual_simplex/singletons.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <dual_simplex/triangle_solve.hpp>
#include <dual_simplex/types.hpp>
#include <dual_simplex/user_problem.hpp>

#include <cstdio>
#include <cstdlib>
#include <queue>
#include <string>

namespace cuopt::linear_programming::dual_simplex {

namespace {

template <typename i_t, typename f_t>
void write_matlab(const std::string& filename, const dual_simplex::lp_problem_t<i_t, f_t>& lp)
{
  FILE* fid = fopen(filename.c_str(), "w");
  if (fid == NULL) { printf("Can't open file %s\n", filename.c_str()); }
  fprintf(fid, "m = %d; n = %d\n", lp.num_rows, lp.num_cols);
  lp.A.print_matrix(fid);
  fprintf(fid, "clu = [");
  for (int32_t j = 0; j < lp.num_cols; ++j) {
    fprintf(fid, "%e %e %e\n", lp.objective[j], lp.lower[j], lp.upper[j]);
  }
  fprintf(fid, "];\n");
  fprintf(fid, "b = [\n");
  for (int32_t i = 0; i < lp.num_rows; ++i) {
    fprintf(fid, "%e\n", lp.rhs[i]);
  }
  fprintf(fid, "];\n");
  fprintf(fid, "A = sparse(ijx(:, 1), ijx(:, 2), ijx(:, 3), m, n);\n");
  fprintf(fid, "c = clu(:, 1);\n");
  fprintf(fid, "l = clu(:, 2);\n");
  fprintf(fid, "u = clu(:, 3);\n");
  fclose(fid);
}

}  // namespace

template <typename i_t, typename f_t>
bool is_mip(const user_problem_t<i_t, f_t>& problem)
{
  bool found_integer = false;
  const i_t n        = problem.num_cols;
  for (i_t j = 0; j < n; ++j) {
    if (problem.var_types[j] != variable_type_t::CONTINUOUS) {
      found_integer = true;
      break;
    }
  }
  return found_integer;
}

template <typename i_t, typename f_t>
f_t compute_objective(const lp_problem_t<i_t, f_t>& problem, const std::vector<f_t>& x)
{
  const i_t n = problem.num_cols;
  assert(x.size() == problem.num_cols);
  f_t obj = 0.0;
  for (i_t j = 0; j < n; ++j) {
    obj += problem.objective[j] * x[j];
  }
  return obj;
}

template <typename i_t, typename f_t>
f_t compute_user_objective(const lp_problem_t<i_t, f_t>& lp, const std::vector<f_t>& x)
{
  const f_t obj      = compute_objective(lp, x);
  const f_t user_obj = compute_user_objective(lp, obj);
  return user_obj;
}

template <typename i_t, typename f_t>
f_t compute_user_objective(const lp_problem_t<i_t, f_t>& lp, f_t obj)
{
  const f_t user_obj = lp.obj_scale * (obj + lp.obj_constant);
  return user_obj;
}

template <typename i_t, typename f_t>
lp_status_t solve_linear_program_advanced(const lp_problem_t<i_t, f_t>& original_lp,
                                          const f_t start_time,
                                          const simplex_solver_settings_t<i_t, f_t>& settings,
                                          lp_solution_t<i_t, f_t>& original_solution,
                                          std::vector<variable_status_t>& vstatus,
                                          std::vector<f_t>& edge_norms)
{
  const i_t m = original_lp.num_rows;
  const i_t n = original_lp.num_cols;
  assert(m <= n);
  std::vector<i_t> basic_list(m);
  std::vector<i_t> nonbasic_list;
  basis_update_mpf_t<i_t, f_t> ft(m, settings.refactor_frequency);
  return solve_linear_program_with_advanced_basis(original_lp,
                                                  start_time,
                                                  settings,
                                                  original_solution,
                                                  ft,
                                                  basic_list,
                                                  nonbasic_list,
                                                  vstatus,
                                                  edge_norms);
}

template <typename i_t, typename f_t>
lp_status_t solve_linear_program_with_advanced_basis(
  const lp_problem_t<i_t, f_t>& original_lp,
  const f_t start_time,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  lp_solution_t<i_t, f_t>& original_solution,
  basis_update_mpf_t<i_t, f_t>& ft,
  std::vector<i_t>& basic_list,
  std::vector<i_t>& nonbasic_list,
  std::vector<variable_status_t>& vstatus,
  std::vector<f_t>& edge_norms)
{
  lp_status_t lp_status = lp_status_t::UNSET;
  lp_problem_t<i_t, f_t> presolved_lp(original_lp.handle_ptr, 1, 1, 1);
  presolve_info_t<i_t, f_t> presolve_info;
  const i_t ok = presolve(original_lp, settings, presolved_lp, presolve_info);
  if (ok == -1) { return lp_status_t::INFEASIBLE; }

  constexpr bool write_out_matlab = false;
  if (write_out_matlab) {
    std::string matlab_file = "presolved.m";
    settings.log.printf("Writing %s\n", matlab_file.c_str());
    write_matlab(matlab_file, presolved_lp);
  }

  lp_problem_t<i_t, f_t> lp(original_lp.handle_ptr,
                            presolved_lp.num_rows,
                            presolved_lp.num_cols,
                            presolved_lp.A.col_start[presolved_lp.num_cols]);
  std::vector<f_t> column_scales;
  column_scaling(presolved_lp, settings, lp, column_scales);
  assert(presolved_lp.num_cols == lp.num_cols);
  lp_problem_t<i_t, f_t> phase1_problem(original_lp.handle_ptr, 1, 1, 1);
  std::vector<variable_status_t> phase1_vstatus;
  f_t phase1_obj = -inf;
  create_phase1_problem(lp, phase1_problem);
  assert(phase1_problem.num_cols == presolved_lp.num_cols);

  // Set the vstatus for the phase1 problem based on a slack basis
  phase1_vstatus.resize(phase1_problem.num_cols);
  std::fill(phase1_vstatus.begin(), phase1_vstatus.end(), variable_status_t::NONBASIC_LOWER);
  i_t num_basic = 0;
  for (i_t j = phase1_problem.num_cols - 1; j >= 0; --j) {
    const i_t col_start = phase1_problem.A.col_start[j];
    const i_t col_end   = phase1_problem.A.col_start[j + 1];
    const i_t nz        = col_end - col_start;
    if (nz == 1 && std::abs(phase1_problem.A.x[col_start]) == 1.0) {
      phase1_vstatus[j] = variable_status_t::BASIC;
      num_basic++;
    }
    if (num_basic == phase1_problem.num_rows) { break; }
  }
  assert(num_basic == phase1_problem.num_rows);
  i_t iter = 0;
  lp_solution_t<i_t, f_t> phase1_solution(phase1_problem.num_rows, phase1_problem.num_cols);
  edge_norms.clear();
  dual::status_t phase1_status = dual_phase2(
    1, 1, start_time, phase1_problem, settings, phase1_vstatus, phase1_solution, iter, edge_norms);
  if (phase1_status == dual::status_t::NUMERICAL ||
      phase1_status == dual::status_t::DUAL_UNBOUNDED) {
    settings.log.printf("Failed in Phase 1\n");
    return lp_status_t::NUMERICAL_ISSUES;
  }
  if (phase1_status == dual::status_t::TIME_LIMIT) { return lp_status_t::TIME_LIMIT; }
  if (phase1_status == dual::status_t::ITERATION_LIMIT) { return lp_status_t::ITERATION_LIMIT; }
  if (phase1_status == dual::status_t::CONCURRENT_LIMIT) { return lp_status_t::CONCURRENT_LIMIT; }
  phase1_obj = phase1_solution.objective;
  if (phase1_obj > -settings.primal_tol) {
    settings.log.printf("Dual feasible solution found.\n");
    lp_solution_t<i_t, f_t> solution(lp.num_rows, lp.num_cols);
    assert(lp.num_cols == phase1_problem.num_cols);
    assert(solution.x.size() == lp.num_cols);
    vstatus = phase1_vstatus;
    edge_norms.clear();
    bool initialize_basis_update = true;
    dual::status_t status        = dual_phase2_with_advanced_basis(2,
                                                            iter == 0 ? 1 : 0,
                                                            initialize_basis_update,
                                                            start_time,
                                                            lp,
                                                            settings,
                                                            vstatus,
                                                            ft,
                                                            basic_list,
                                                            nonbasic_list,
                                                            solution,
                                                            iter,
                                                            edge_norms);
    if (status == dual::status_t::NUMERICAL) {
      // Became dual infeasible. Try phase 1 again
      phase1_vstatus = vstatus;
      settings.log.printf("Running Phase 1 again\n");
      edge_norms.clear();
      initialize_basis_update = false;
      dual_phase2_with_advanced_basis(1,
                                      0,
                                      initialize_basis_update,
                                      start_time,
                                      phase1_problem,
                                      settings,
                                      phase1_vstatus,
                                      ft,
                                      basic_list,
                                      nonbasic_list,
                                      phase1_solution,
                                      iter,
                                      edge_norms);
      vstatus = phase1_vstatus;
      edge_norms.clear();
      status = dual_phase2_with_advanced_basis(2,
                                               0,
                                               initialize_basis_update,
                                               start_time,
                                               lp,
                                               settings,
                                               vstatus,
                                               ft,
                                               basic_list,
                                               nonbasic_list,
                                               solution,
                                               iter,
                                               edge_norms);
    }
    constexpr bool primal_cleanup = false;
    if (status == dual::status_t::OPTIMAL && primal_cleanup) {
      primal_phase2(2, start_time, lp, settings, vstatus, solution, iter);
    }
    if (status == dual::status_t::OPTIMAL) {
      std::vector<f_t> unscaled_x(lp.num_cols);
      std::vector<f_t> unscaled_z(lp.num_cols);
      unscale_solution<i_t, f_t>(column_scales, solution.x, solution.z, unscaled_x, unscaled_z);
      uncrush_solution(presolve_info,
                       settings,
                       unscaled_x,
                       solution.y,
                       unscaled_z,
                       original_solution.x,
                       original_solution.y,
                       original_solution.z);
      original_solution.objective          = solution.objective;
      original_solution.user_objective     = solution.user_objective;
      original_solution.l2_primal_residual = solution.l2_primal_residual;
      original_solution.l2_dual_residual   = solution.l2_dual_residual;
      lp_status                            = lp_status_t::OPTIMAL;
    }
    if (status == dual::status_t::DUAL_UNBOUNDED) { lp_status = lp_status_t::INFEASIBLE; }
    if (status == dual::status_t::TIME_LIMIT) { lp_status = lp_status_t::TIME_LIMIT; }
    if (status == dual::status_t::ITERATION_LIMIT) { lp_status = lp_status_t::ITERATION_LIMIT; }
    if (status == dual::status_t::CONCURRENT_LIMIT) { lp_status = lp_status_t::CONCURRENT_LIMIT; }
    if (status == dual::status_t::NUMERICAL) { lp_status = lp_status_t::NUMERICAL_ISSUES; }
    if (status == dual::status_t::CUTOFF) { lp_status = lp_status_t::CUTOFF; }
    original_solution.iterations = iter;
  } else {
    // Dual infeasible -> Primal unbounded
    settings.log.printf("Dual infeasible\n");
    original_solution.objective = -inf;
    if (lp.obj_scale == 1.0) {
      // Objective for unbounded minimization is -inf
      original_solution.user_objective = -inf;
    } else {
      // Objective for unbounded maximization is inf
      original_solution.user_objective = inf;
    }
    original_solution.iterations = iter;
    return lp_status_t::UNBOUNDED;
  }
  return lp_status;
}

template <typename i_t, typename f_t>
lp_status_t solve_linear_program_with_barrier(const user_problem_t<i_t, f_t>& user_problem,
                                              const simplex_solver_settings_t<i_t, f_t>& settings,
                                              lp_solution_t<i_t, f_t>& solution)
{
  f_t start_time     = tic();
  lp_status_t status = lp_status_t::UNSET;
  lp_problem_t<i_t, f_t> original_lp(user_problem.handle_ptr, 1, 1, 1);

  // Convert the user problem to a linear program with only equality constraints
  std::vector<i_t> new_slacks;
  simplex_solver_settings_t<i_t, f_t> barrier_settings = settings;
  barrier_settings.barrier_presolve                    = true;
  dualize_info_t<i_t, f_t> dualize_info;
  convert_user_problem(user_problem, barrier_settings, original_lp, new_slacks, dualize_info);
  lp_solution_t<i_t, f_t> lp_solution(original_lp.num_rows, original_lp.num_cols);

  // Presolve the linear program
  presolve_info_t<i_t, f_t> presolve_info;
  lp_problem_t<i_t, f_t> presolved_lp(user_problem.handle_ptr, 1, 1, 1);
  const i_t ok = presolve(original_lp, barrier_settings, presolved_lp, presolve_info);
  if (ok == -1) { return lp_status_t::INFEASIBLE; }

  // Apply columns scaling to the presolve LP
  lp_problem_t<i_t, f_t> barrier_lp(user_problem.handle_ptr,
                                    presolved_lp.num_rows,
                                    presolved_lp.num_cols,
                                    presolved_lp.A.col_start[presolved_lp.num_cols]);
  std::vector<f_t> column_scales;
  column_scaling(presolved_lp, barrier_settings, barrier_lp, column_scales);

  // Solve using barrier
  lp_solution_t<i_t, f_t> barrier_solution(barrier_lp.num_rows, barrier_lp.num_cols);
  barrier_solver_t<i_t, f_t> barrier_solver(barrier_lp, presolve_info, barrier_settings);
  barrier_solver_settings_t<i_t, f_t> barrier_solver_settings;
  lp_status_t barrier_status =
    barrier_solver.solve(start_time, barrier_solver_settings, barrier_solution);
  if (barrier_status == lp_status_t::OPTIMAL) {
#ifdef COMPUTE_SCALED_RESIDUALS
    std::vector<f_t> scaled_residual = barrier_lp.rhs;
    matrix_vector_multiply(barrier_lp.A, 1.0, barrier_solution.x, -1.0, scaled_residual);
    f_t scaled_primal_residual = vector_norm_inf<i_t, f_t>(scaled_residual);
    settings.log.printf("Scaled Primal residual: %e\n", scaled_primal_residual);

    std::vector<f_t> scaled_dual_residual = barrier_solution.z;
    for (i_t j = 0; j < scaled_dual_residual.size(); ++j) {
      scaled_dual_residual[j] -= barrier_lp.objective[j];
    }
    matrix_transpose_vector_multiply(
      barrier_lp.A, 1.0, barrier_solution.y, 1.0, scaled_dual_residual);
    f_t scaled_dual_residual_norm = vector_norm_inf<i_t, f_t>(scaled_dual_residual);
    settings.log.printf("Scaled Dual residual: %e\n", scaled_dual_residual_norm);
#endif
    // Unscale the solution
    std::vector<f_t> unscaled_x(barrier_lp.num_cols);
    std::vector<f_t> unscaled_z(barrier_lp.num_cols);
    unscale_solution<i_t, f_t>(
      column_scales, barrier_solution.x, barrier_solution.z, unscaled_x, unscaled_z);

    std::vector<f_t> residual = presolved_lp.rhs;
    matrix_vector_multiply(presolved_lp.A, 1.0, unscaled_x, -1.0, residual);
    f_t primal_residual = vector_norm_inf<i_t, f_t>(residual);
    settings.log.printf("Unscaled Primal infeasibility   (abs/rel): %.2e/%.2e\n",
                        primal_residual,
                        primal_residual / (1.0 + vector_norm_inf<i_t, f_t>(presolved_lp.rhs)));

    std::vector<f_t> unscaled_dual_residual = unscaled_z;
    for (i_t j = 0; j < unscaled_dual_residual.size(); ++j) {
      unscaled_dual_residual[j] -= presolved_lp.objective[j];
    }
    matrix_transpose_vector_multiply(
      presolved_lp.A, 1.0, barrier_solution.y, 1.0, unscaled_dual_residual);
    f_t unscaled_dual_residual_norm = vector_norm_inf<i_t, f_t>(unscaled_dual_residual);
    settings.log.printf(
      "Unscaled Dual infeasibility     (abs/rel): %.2e/%.2e\n",
      unscaled_dual_residual_norm,
      unscaled_dual_residual_norm / (1.0 + vector_norm_inf<i_t, f_t>(presolved_lp.objective)));

    // Undo presolve
    uncrush_solution(presolve_info,
                     barrier_settings,
                     unscaled_x,
                     barrier_solution.y,
                     unscaled_z,
                     lp_solution.x,
                     lp_solution.y,
                     lp_solution.z);

    std::vector<f_t> post_solve_residual = original_lp.rhs;
    matrix_vector_multiply(original_lp.A, 1.0, lp_solution.x, -1.0, post_solve_residual);
    f_t post_solve_primal_residual = vector_norm_inf<i_t, f_t>(post_solve_residual);
    settings.log.printf(
      "Post-solve Primal infeasibility (abs/rel): %.2e/%.2e\n",
      post_solve_primal_residual,
      post_solve_primal_residual / (1.0 + vector_norm_inf<i_t, f_t>(original_lp.rhs)));

    std::vector<f_t> post_solve_dual_residual = lp_solution.z;
    for (i_t j = 0; j < post_solve_dual_residual.size(); ++j) {
      post_solve_dual_residual[j] -= original_lp.objective[j];
    }
    matrix_transpose_vector_multiply(
      original_lp.A, 1.0, lp_solution.y, 1.0, post_solve_dual_residual);
    f_t post_solve_dual_residual_norm = vector_norm_inf<i_t, f_t>(post_solve_dual_residual);
    settings.log.printf(
      "Post-solve Dual infeasibility   (abs/rel): %.2e/%.2e\n",
      post_solve_dual_residual_norm,
      post_solve_dual_residual_norm / (1.0 + vector_norm_inf<i_t, f_t>(original_lp.objective)));

    if (dualize_info.solving_dual) {
      lp_solution_t<i_t, f_t> primal_solution(dualize_info.primal_problem.num_rows,
                                              dualize_info.primal_problem.num_cols);
      std::copy(lp_solution.y.begin(),
                lp_solution.y.begin() + dualize_info.primal_problem.num_cols,
                primal_solution.x.data());

      // Negate x
      for (i_t i = 0; i < dualize_info.primal_problem.num_cols; ++i) {
        primal_solution.x[i] *= -1.0;
      }
      std::copy(lp_solution.x.begin(),
                lp_solution.x.begin() + dualize_info.primal_problem.num_rows,
                primal_solution.y.data());
      // Negate y
      for (i_t i = 0; i < dualize_info.primal_problem.num_rows; ++i) {
        primal_solution.y[i] *= -1.0;
      }

      std::vector<f_t>& z = primal_solution.z;
      for (i_t j = 0; j < dualize_info.primal_problem.num_cols; ++j) {
        const i_t u = dualize_info.zl_start + j;
        z[j]        = lp_solution.x[u];
      }
      i_t k = 0;
      for (i_t j : dualize_info.vars_with_upper_bounds) {
        const i_t v = dualize_info.zu_start + k;
        z[j] -= lp_solution.x[v];
        k++;
      }

      // Check the objective and residuals on the primal problem.
      settings.log.printf("Primal objective: %e\n",
                          dot<i_t, f_t>(dualize_info.primal_problem.objective, primal_solution.x));

      std::vector<f_t> primal_residual = dualize_info.primal_problem.rhs;
      matrix_vector_multiply(
        dualize_info.primal_problem.A, 1.0, primal_solution.x, -1.0, primal_residual);
      std::vector<i_t> inequality_rows(dualize_info.primal_problem.num_rows, 1);
      for (i_t i : dualize_info.equality_rows) {
        inequality_rows[i] = 0;
      }
      for (i_t i = 0; i < dualize_info.primal_problem.num_rows; ++i) {
        if (inequality_rows[i] == 1) {
          primal_residual[i] = std::max(primal_residual[i], 0.0);  // a_i^T x - b_i <= 0
        }
      }
      f_t primal_residual_norm     = vector_norm_inf<i_t, f_t>(primal_residual);
      const f_t norm_b             = vector_norm_inf<i_t, f_t>(dualize_info.primal_problem.rhs);
      f_t primal_relative_residual = primal_residual_norm / (1.0 + norm_b);
      settings.log.printf(
        "Primal residual (abs/rel): %e/%e\n", primal_residual_norm, primal_relative_residual);

      std::vector<f_t> dual_residual = dualize_info.primal_problem.objective;
      for (i_t j = 0; j < dualize_info.primal_problem.num_cols; ++j) {
        dual_residual[j] -= z[j];
      }
      matrix_transpose_vector_multiply(
        dualize_info.primal_problem.A, 1.0, primal_solution.y, -1.0, dual_residual);
      f_t dual_residual_norm     = vector_norm_inf<i_t, f_t>(dual_residual);
      const f_t norm_c           = vector_norm_inf<i_t, f_t>(dualize_info.primal_problem.objective);
      f_t dual_relative_residual = dual_residual_norm / (1.0 + norm_c);
      settings.log.printf(
        "Dual residual (abs/rel): %e/%e\n", dual_residual_norm, dual_relative_residual);

      original_lp = dualize_info.primal_problem;
      lp_solution = primal_solution;
    }

    uncrush_primal_solution(user_problem, original_lp, lp_solution.x, solution.x);
    uncrush_dual_solution(
      user_problem, original_lp, lp_solution.y, lp_solution.z, solution.y, solution.z);
    solution.objective          = barrier_solution.objective;
    solution.user_objective     = barrier_solution.user_objective;
    solution.l2_primal_residual = barrier_solution.l2_primal_residual;
    solution.l2_dual_residual   = barrier_solution.l2_dual_residual;
    solution.iterations         = barrier_solution.iterations;
  }

  // If we aren't doing crossover, we're done
  if (!settings.crossover) { return barrier_status; }

  if (settings.crossover && barrier_status == lp_status_t::OPTIMAL) {
    // Check to see if we need to add artifical variables
    std::vector<i_t> artificial_variables;
    artificial_variables.reserve(original_lp.num_rows);
    for (i_t i = 0; i < original_lp.num_rows; ++i) {
      artificial_variables.push_back(i);
    }
    if (artificial_variables.size() > 0) {
      settings.log.printf("Adding %ld artificial variables\n", artificial_variables.size());
      const i_t additional_vars = artificial_variables.size();
      const i_t new_cols        = original_lp.num_cols + additional_vars;
      i_t col                   = original_lp.num_cols;
      i_t nz                    = original_lp.A.col_start[col];
      const i_t new_nz          = nz + additional_vars;
      original_lp.A.col_start.resize(new_cols + 1);
      original_lp.A.x.resize(new_nz);
      original_lp.A.i.resize(new_nz);
      original_lp.objective.resize(new_cols);
      original_lp.lower.resize(new_cols);
      original_lp.upper.resize(new_cols);
      lp_solution.x.resize(new_cols);
      lp_solution.z.resize(new_cols);
      for (i_t i : artificial_variables) {
        original_lp.A.x[nz]        = 1.0;
        original_lp.A.i[nz]        = i;
        original_lp.objective[col] = 0.0;
        original_lp.lower[col]     = 0.0;
        original_lp.upper[col]     = 0.0;
        lp_solution.x[col]         = 0.0;
        lp_solution.z[col]         = -lp_solution.y[i];
        nz++;
        col++;
        original_lp.A.col_start[col] = nz;
      }
      original_lp.A.n      = new_cols;
      original_lp.num_cols = new_cols;
#ifdef PRINT_INFO
      printf("nz %d =? new_nz %d =? Acol %d, num_cols %d =? new_cols %d x size %ld z size %ld\n",
             nz,
             new_nz,
             original_lp.A.col_start[original_lp.num_cols],
             original_lp.num_cols,
             new_cols,
             lp_solution.x.size(),
             lp_solution.z.size());
#endif
    }

    // Run crossover
    lp_solution_t<i_t, f_t> crossover_solution(original_lp.num_rows, original_lp.num_cols);
    std::vector<variable_status_t> vstatus(original_lp.num_cols);
    crossover_status_t crossover_status = crossover(
      original_lp, barrier_settings, lp_solution, start_time, crossover_solution, vstatus);
    settings.log.printf("Crossover status: %d\n", crossover_status);
    if (crossover_status == crossover_status_t::OPTIMAL) { barrier_status = lp_status_t::OPTIMAL; }
  }
  return barrier_status;
}

template <typename i_t, typename f_t>
lp_status_t solve_linear_program(const user_problem_t<i_t, f_t>& user_problem,
                                 const simplex_solver_settings_t<i_t, f_t>& settings,
                                 lp_solution_t<i_t, f_t>& solution)
{
  f_t start_time = tic();
  lp_problem_t<i_t, f_t> original_lp(user_problem.handle_ptr, 1, 1, 1);
  std::vector<i_t> new_slacks;
  dualize_info_t<i_t, f_t> dualize_info;
  convert_user_problem(user_problem, settings, original_lp, new_slacks, dualize_info);
  solution.resize(user_problem.num_rows, user_problem.num_cols);
  lp_solution_t<i_t, f_t> lp_solution(original_lp.num_rows, original_lp.num_cols);
  std::vector<variable_status_t> vstatus;
  std::vector<f_t> edge_norms;
  lp_status_t status = solve_linear_program_advanced(
    original_lp, start_time, settings, lp_solution, vstatus, edge_norms);
  uncrush_primal_solution(user_problem, original_lp, lp_solution.x, solution.x);
  uncrush_dual_solution(
    user_problem, original_lp, lp_solution.y, lp_solution.z, solution.y, solution.z);
  solution.objective          = lp_solution.objective;
  solution.user_objective     = lp_solution.user_objective;
  solution.iterations         = lp_solution.iterations;
  solution.l2_primal_residual = lp_solution.l2_primal_residual;
  solution.l2_dual_residual   = lp_solution.l2_dual_residual;
  return status;
}

template <typename i_t, typename f_t>
i_t solve(const user_problem_t<i_t, f_t>& problem,
          const simplex_solver_settings_t<i_t, f_t>& settings,
          std::vector<f_t>& primal_solution)
{
  i_t status;
  if (is_mip(problem) && !settings.relaxation) {
    branch_and_bound_t branch_and_bound(problem, settings);
    mip_solution_t<i_t, f_t> mip_solution(problem.num_cols);
    mip_status_t mip_status = branch_and_bound.solve(mip_solution);
    if (mip_status == mip_status_t::OPTIMAL) {
      status = 0;
    } else {
      status = -1;
    }
    primal_solution = mip_solution.x;
  } else {
    f_t start_time = tic();
    lp_problem_t<i_t, f_t> original_lp(
      problem.handle_ptr, problem.num_rows, problem.num_cols, problem.A.col_start[problem.A.n]);
    std::vector<i_t> new_slacks;
    dualize_info_t<i_t, f_t> dualize_info;
    convert_user_problem(problem, settings, original_lp, new_slacks, dualize_info);
    lp_solution_t<i_t, f_t> solution(original_lp.num_rows, original_lp.num_cols);
    std::vector<variable_status_t> vstatus;
    std::vector<f_t> edge_norms;
    lp_status_t lp_status = solve_linear_program_advanced(
      original_lp, start_time, settings, solution, vstatus, edge_norms);
    primal_solution = solution.x;
    if (lp_status == lp_status_t::OPTIMAL) {
      status = 0;
    } else {
      status = -1;
    }
  }
  return status;
}

template <typename i_t, typename f_t>
i_t solve_mip_with_guess(const user_problem_t<i_t, f_t>& problem,
                         const simplex_solver_settings_t<i_t, f_t>& settings,
                         const std::vector<f_t>& guess,
                         mip_solution_t<i_t, f_t>& solution)
{
  i_t status;
  if (is_mip(problem)) {
    branch_and_bound_t branch_and_bound(problem, settings);
    branch_and_bound.set_initial_guess(guess);
    mip_status_t mip_status = branch_and_bound.solve(solution);
    if (mip_status == mip_status_t::OPTIMAL) {
      status = 0;
    } else {
      status = -1;
    }
  } else {
    settings.log.printf("Not a MIP\n");
    status = -1;
  }
  return status;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template bool is_mip<int, double>(const user_problem_t<int, double>& problem);

template double compute_objective<int, double>(const lp_problem_t<int, double>& problem,
                                               const std::vector<double>& x);

template double compute_user_objective<int, double>(const lp_problem_t<int, double>& lp,
                                                    const std::vector<double>& x);

template double compute_user_objective(const lp_problem_t<int, double>& lp, double obj);

template lp_status_t solve_linear_program_advanced(
  const lp_problem_t<int, double>& original_lp,
  const double start_time,
  const simplex_solver_settings_t<int, double>& settings,
  lp_solution_t<int, double>& original_solution,
  std::vector<variable_status_t>& vstatus,
  std::vector<double>& edge_norms);

template lp_status_t solve_linear_program_with_advanced_basis(
  const lp_problem_t<int, double>& original_lp,
  const double start_time,
  const simplex_solver_settings_t<int, double>& settings,
  lp_solution_t<int, double>& original_solution,
  basis_update_mpf_t<int, double>& ft,
  std::vector<int>& basic_list,
  std::vector<int>& nonbasic_list,
  std::vector<variable_status_t>& vstatus,
  std::vector<double>& edge_norms);

template lp_status_t solve_linear_program_with_barrier(
  const user_problem_t<int, double>& user_problem,
  const simplex_solver_settings_t<int, double>& settings,
  lp_solution_t<int, double>& solution);

template lp_status_t solve_linear_program(const user_problem_t<int, double>& user_problem,
                                          const simplex_solver_settings_t<int, double>& settings,
                                          lp_solution_t<int, double>& solution);

template int solve<int, double>(const user_problem_t<int, double>& user_problem,
                                const simplex_solver_settings_t<int, double>& settings,
                                std::vector<double>& primal_solution);

template int solve_mip_with_guess<int, double>(
  const user_problem_t<int, double>& problem,
  const simplex_solver_settings_t<int, double>& settings,
  const std::vector<double>& guess,
  mip_solution_t<int, double>& solution);

#endif

}  // namespace cuopt::linear_programming::dual_simplex
