/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <branch_and_bound/pseudo_costs.hpp>
#include <branch_and_bound/shared_strong_branching_context.hpp>

#include <dual_simplex/phase2.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/tic_toc.hpp>

#include <pdlp/pdlp_constants.hpp>

#include <cuopt/linear_programming/solve.hpp>

#include <utilities/copy_helpers.hpp>

#include <raft/core/nvtx.hpp>

#include <omp.h>

namespace cuopt::linear_programming::dual_simplex {

namespace {

static bool is_dual_simplex_done(dual::status_t status)
{
  return status == dual::status_t::DUAL_UNBOUNDED || status == dual::status_t::OPTIMAL ||
         status == dual::status_t::ITERATION_LIMIT || status == dual::status_t::CUTOFF;
}

template <typename f_t>
struct objective_change_estimate_t {
  f_t down_obj_change;
  f_t up_obj_change;
};

template <typename i_t, typename f_t>
f_t compute_step_length(const simplex_solver_settings_t<i_t, f_t>& settings,
                        const std::vector<variable_status_t>& vstatus,
                        const std::vector<f_t>& z,
                        const std::vector<f_t>& delta_z,
                        const std::vector<i_t>& delta_z_indices)
{
  f_t step_length = inf;
  f_t pivot_tol   = settings.pivot_tol;
  const i_t nz    = delta_z_indices.size();
  for (i_t h = 0; h < nz; h++) {
    const i_t j = delta_z_indices[h];
    if (vstatus[j] == variable_status_t::NONBASIC_LOWER && delta_z[j] < -pivot_tol) {
      const f_t ratio = -z[j] / delta_z[j];
      if (ratio < step_length) { step_length = ratio; }
    } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER && delta_z[j] > pivot_tol) {
      const f_t ratio = -z[j] / delta_z[j];
      if (ratio < step_length) { step_length = ratio; }
    }
  }
  return step_length;
}

template <typename i_t, typename f_t>
objective_change_estimate_t<f_t> single_pivot_objective_change_estimate(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const csc_matrix_t<i_t, f_t>& A_transpose,
  const std::vector<variable_status_t>& vstatus,
  i_t variable_j,
  i_t basic_j,
  const lp_solution_t<i_t, f_t>& lp_solution,
  const std::vector<i_t>& basic_list,
  const std::vector<i_t>& nonbasic_list,
  const std::vector<i_t>& nonbasic_mark,
  basis_update_mpf_t<i_t, f_t>& basis_factors,
  std::vector<i_t>& workspace,
  std::vector<f_t>& delta_z,
  f_t& work_estimate)
{
  // Compute the objective estimate for the down and up branches of variable j
  assert(variable_j >= 0);
  assert(basic_j >= 0);

  // Down branch
  i_t direction = -1;
  sparse_vector_t<i_t, f_t> e_k(lp.num_rows, 0);
  e_k.i.push_back(basic_j);
  e_k.x.push_back(-f_t(direction));

  sparse_vector_t<i_t, f_t> delta_y(lp.num_rows, 0);
  basis_factors.b_transpose_solve(e_k, delta_y);

  // Compute delta_z_N = -N^T * delta_y
  i_t delta_y_nz0      = 0;
  const i_t nz_delta_y = delta_y.i.size();
  for (i_t k = 0; k < nz_delta_y; k++) {
    if (std::abs(delta_y.x[k]) > settings.zero_tol) { delta_y_nz0++; }
  }
  work_estimate += nz_delta_y;
  const f_t delta_y_nz_percentage = delta_y_nz0 / static_cast<f_t>(lp.num_rows) * 100.0;
  const bool use_transpose        = delta_y_nz_percentage <= 30.0;
  std::vector<i_t> delta_z_indices;
  // delta_z starts out all zero
  if (use_transpose) {
    compute_delta_z(A_transpose,
                    delta_y,
                    variable_j,
                    direction,
                    nonbasic_mark,
                    workspace,
                    delta_z_indices,
                    delta_z,
                    work_estimate);
  } else {
    std::vector<f_t> delta_y_dense(lp.num_rows, 0);
    delta_y.to_dense(delta_y_dense);
    compute_reduced_cost_update(lp,
                                basic_list,
                                nonbasic_list,
                                delta_y_dense,
                                variable_j,
                                direction,
                                workspace,
                                delta_z_indices,
                                delta_z,
                                work_estimate);
  }

  // Verify dual feasibility
#ifdef CHECK_DUAL_FEASIBILITY
  {
    std::vector<f_t> dual_residual = lp_solution.z;
    for (i_t j = 0; j < lp.num_cols; j++) {
      dual_residual[j] -= lp.objective[j];
    }
    matrix_transpose_vector_multiply(lp.A, 1.0, lp_solution.y, 1.0, dual_residual);
    f_t dual_residual_norm = vector_norm_inf<i_t, f_t>(dual_residual);
    settings.log.printf("Dual residual norm: %e\n", dual_residual_norm);
  }
#endif

  // Compute the step-length
  f_t step_length = compute_step_length(settings, vstatus, lp_solution.z, delta_z, delta_z_indices);

  // Handle the leaving variable case

  f_t delta_obj_down =
    step_length * (lp_solution.x[variable_j] - std::floor(lp_solution.x[variable_j]));
#ifdef CHECK_DELTA_OBJ
  f_t delta_obj_check = 0.0;
  for (i_t k = 0; k < delta_y.i.size(); k++) {
    delta_obj_check += lp.rhs[delta_y.i[k]] * delta_y.x[k];
  }
  for (i_t h = 0; h < delta_z_indices.size(); h++) {
    const i_t j = delta_z_indices[h];
    if (vstatus[j] == variable_status_t::NONBASIC_LOWER) {
      delta_obj_check += lp.lower[j] * delta_z[j];
    } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER) {
      delta_obj_check += lp.upper[j] * delta_z[j];
    }
  }
  delta_obj_check += std::floor(lp_solution.x[variable_j]) * delta_z[variable_j];
  delta_obj_check *= step_length;
  if (std::abs(delta_obj_check - delta_obj) > 1e-6) {
    settings.log.printf("Delta obj check %e. Delta obj %e. Step length %e.\n",
                        delta_obj_check,
                        delta_obj,
                        step_length);
  }
#endif

  settings.log.debug(
    "Down branch %d. Step length: %e. Delta obj: %e. \n", variable_j, step_length, delta_obj_down);

  // Up branch
  direction = 1;
  // Negate delta_z
  for (i_t j : delta_z_indices) {
    delta_z[j] *= -1.0;
  }

  // Compute the step-length
  step_length = compute_step_length(settings, vstatus, lp_solution.z, delta_z, delta_z_indices);

  f_t delta_obj_up =
    step_length * (std::ceil(lp_solution.x[variable_j]) - lp_solution.x[variable_j]);
  settings.log.debug(
    "Up branch %d. Step length: %e. Delta obj: %e.\n", variable_j, step_length, delta_obj_up);

  delta_z_indices.push_back(variable_j);

  // Clear delta_z
  for (i_t j : delta_z_indices) {
    delta_z[j]   = 0.0;
    workspace[j] = 0;
  }

#ifdef CHECK_DELTA_Z
  for (i_t j = 0; j < lp.num_cols; j++) {
    if (delta_z[j] != 0.0) { settings.log.printf("Delta z %d: %e\n", j, delta_z[j]); }
  }
  for (i_t j = 0; j < lp.num_cols; j++) {
    if (workspace[j] != 0) { settings.log.printf("Workspace %d: %d\n", j, workspace[j]); }
  }
#endif

  return {.down_obj_change = std::max<f_t>(delta_obj_down, 0),
          .up_obj_change   = std::max<f_t>(delta_obj_up, 0)};
}

template <typename i_t, typename f_t>
void initialize_pseudo_costs_with_estimate(const lp_problem_t<i_t, f_t>& lp,
                                           const simplex_solver_settings_t<i_t, f_t>& settings,
                                           const std::vector<variable_status_t>& vstatus,
                                           const lp_solution_t<i_t, f_t>& lp_solution,
                                           const std::vector<i_t>& basic_list,
                                           const std::vector<i_t>& nonbasic_list,
                                           const std::vector<i_t>& fractional,
                                           basis_update_mpf_t<i_t, f_t>& basis_factors,
                                           pseudo_costs_t<i_t, f_t>& pc)
{
  i_t m = lp.num_rows;
  i_t n = lp.num_cols;

  std::vector<f_t> delta_z(n, 0);
  std::vector<i_t> workspace(n, 0);

  f_t work_estimate = 0;

  std::vector<i_t> basic_map(n, -1);
  for (i_t i = 0; i < m; i++) {
    basic_map[basic_list[i]] = i;
  }

  std::vector<i_t> nonbasic_mark(n, -1);
  for (i_t i = 0; i < n - m; i++) {
    nonbasic_mark[nonbasic_list[i]] = i;
  }

  for (i_t k = 0; k < fractional.size(); k++) {
    const i_t j = fractional[k];
    assert(j >= 0);

    objective_change_estimate_t<f_t> estimate =
      single_pivot_objective_change_estimate(lp,
                                             settings,
                                             pc.AT,
                                             vstatus,
                                             j,
                                             basic_map[j],
                                             lp_solution,
                                             basic_list,
                                             nonbasic_list,
                                             nonbasic_mark,
                                             basis_factors,
                                             workspace,
                                             delta_z,
                                             work_estimate);
    pc.strong_branch_down[k] = estimate.down_obj_change;
    pc.strong_branch_up[k]   = estimate.up_obj_change;
  }
}

template <typename i_t, typename f_t>
f_t objective_upper_bound(const lp_problem_t<i_t, f_t>& lp, f_t upper_bound, f_t dual_tol)
{
  f_t cut_off = 0;

  if (std::isfinite(upper_bound)) {
    cut_off = upper_bound + dual_tol;
  } else {
    cut_off = 0;
    for (i_t j = 0; j < lp.num_cols; ++j) {
      if (lp.objective[j] > 0) {
        cut_off += lp.objective[j] * lp.upper[j];
      } else if (lp.objective[j] < 0) {
        cut_off += lp.objective[j] * lp.lower[j];
      }
    }
  }

  return cut_off;
}

template <typename i_t, typename f_t>
void strong_branch_helper(i_t start,
                          i_t end,
                          f_t start_time,
                          const lp_problem_t<i_t, f_t>& original_lp,
                          const simplex_solver_settings_t<i_t, f_t>& settings,
                          const std::vector<variable_type_t>& var_types,
                          const std::vector<i_t>& fractional,
                          const std::vector<f_t>& root_soln,
                          const std::vector<variable_status_t>& root_vstatus,
                          const std::vector<f_t>& edge_norms,
                          f_t root_obj,
                          f_t upper_bound,
                          i_t iter_limit,
                          pseudo_costs_t<i_t, f_t>& pc,
                          std::vector<f_t>& dual_simplex_obj_down,
                          std::vector<f_t>& dual_simplex_obj_up,
                          std::vector<dual::status_t>& dual_simplex_status_down,
                          std::vector<dual::status_t>& dual_simplex_status_up,
                          shared_strong_branching_context_view_t<i_t, f_t>& sb_view)
{
  raft::common::nvtx::range scope("BB::strong_branch_helper");
  lp_problem_t child_problem = original_lp;

  constexpr bool verbose = false;
  f_t last_log           = tic();
  i_t thread_id          = omp_get_thread_num();
  for (i_t k = start; k < end; ++k) {
    const i_t j = fractional[k];

    for (i_t branch = 0; branch < 2; branch++) {
      // Do the down branch
      const i_t shared_idx = (branch == 0) ? k : k + static_cast<i_t>(fractional.size());
      // Batch PDLP has already solved this subproblem, skip it
      if (sb_view.is_valid() && sb_view.is_solved(shared_idx)) {
        if (verbose) {
          settings.log.printf(
            "[COOP SB] DS thread %d skipping variable %d branch %s (shared_idx %d): already solved "
            "by PDLP\n",
            thread_id,
            j,
            branch == 0 ? "down" : "up",
            shared_idx);
        }
        continue;
      }

      if (branch == 0) {
        child_problem.lower[j] = original_lp.lower[j];
        child_problem.upper[j] = std::floor(root_soln[j]);
      } else {
        child_problem.lower[j] = std::ceil(root_soln[j]);
        child_problem.upper[j] = original_lp.upper[j];
      }

      simplex_solver_settings_t<i_t, f_t> child_settings = settings;
      child_settings.set_log(false);
      f_t lp_start_time = tic();
      f_t elapsed_time  = toc(start_time);
      if (elapsed_time > settings.time_limit) { break; }
      child_settings.time_limit      = std::max(0.0, settings.time_limit - elapsed_time);
      child_settings.iteration_limit = iter_limit;
      child_settings.cut_off =
        objective_upper_bound(child_problem, upper_bound, child_settings.dual_tol);

      lp_solution_t<i_t, f_t> solution(original_lp.num_rows, original_lp.num_cols);
      i_t iter                               = 0;
      std::vector<variable_status_t> vstatus = root_vstatus;
      std::vector<f_t> child_edge_norms      = edge_norms;
      dual::status_t status                  = dual_phase2(2,
                                          0,
                                          lp_start_time,
                                          child_problem,
                                          child_settings,
                                          vstatus,
                                          solution,
                                          iter,
                                          child_edge_norms);

      f_t obj = std::numeric_limits<f_t>::quiet_NaN();
      if (status == dual::status_t::DUAL_UNBOUNDED) {
        // LP was infeasible
        obj = std::numeric_limits<f_t>::infinity();
      } else if (status == dual::status_t::OPTIMAL || status == dual::status_t::ITERATION_LIMIT ||
                 status == dual::status_t::CUTOFF) {
        obj = compute_objective(child_problem, solution.x);
      } else {
        settings.log.debug("Thread id %2d remaining %d variable %d branch %d status %d\n",
                           thread_id,
                           end - 1 - k,
                           j,
                           branch,
                           status);
      }

      if (branch == 0) {
        pc.strong_branch_down[k]    = std::max(obj - root_obj, 0.0);
        dual_simplex_obj_down[k]    = std::max(obj - root_obj, 0.0);
        dual_simplex_status_down[k] = status;
        if (verbose) {
          settings.log.printf("Thread id %2d remaining %d variable %d branch %d obj %e time %.2f\n",
                              thread_id,
                              end - 1 - k,
                              j,
                              branch,
                              obj,
                              toc(start_time));
        }
      } else {
        pc.strong_branch_up[k]    = std::max(obj - root_obj, 0.0);
        dual_simplex_obj_up[k]    = std::max(obj - root_obj, 0.0);
        dual_simplex_status_up[k] = status;
        if (verbose) {
          settings.log.printf(
            "Thread id %2d remaining %d variable %d branch %d obj %e change down %e change up %e "
            "time %.2f\n",
            thread_id,
            end - 1 - k,
            j,
            branch,
            obj,
            dual_simplex_obj_down[k],
            dual_simplex_obj_up[k],
            toc(start_time));
        }
      }
      // Mark the subproblem as solved so that batch PDLP removes it from the batch
      if (sb_view.is_valid()) {
        // We could not mark as solved nodes hitting iteration limit in DS
        if ((branch == 0 && is_dual_simplex_done(dual_simplex_status_down[k])) ||
            (branch == 1 && is_dual_simplex_done(dual_simplex_status_up[k]))) {
          sb_view.mark_solved(shared_idx);
          if (verbose) {
            settings.log.printf(
              "[COOP SB] DS thread %d solved variable %d branch %s (shared_idx %d), marking in "
              "shared context\n",
              thread_id,
              j,
              branch == 0 ? "down" : "up",
              shared_idx);
          }
        }
      }
      if (toc(start_time) > settings.time_limit) { break; }
    }
    if (toc(start_time) > settings.time_limit) { break; }

    const i_t completed = pc.num_strong_branches_completed++;

    if (thread_id == 0 && toc(last_log) > 10) {
      last_log = tic();
      settings.log.printf("%d of %ld strong branches completed in %.1fs\n",
                          completed,
                          fractional.size(),
                          toc(start_time));
    }

    child_problem.lower[j] = original_lp.lower[j];
    child_problem.upper[j] = original_lp.upper[j];

    if (toc(start_time) > settings.time_limit) { break; }
  }
}

template <typename i_t, typename f_t>
std::pair<f_t, dual::status_t> trial_branching(const lp_problem_t<i_t, f_t>& original_lp,
                                               const simplex_solver_settings_t<i_t, f_t>& settings,
                                               const std::vector<variable_type_t>& var_types,
                                               const std::vector<variable_status_t>& vstatus,
                                               const std::vector<f_t>& edge_norms,
                                               const basis_update_mpf_t<i_t, f_t>& basis_factors,
                                               const std::vector<i_t>& basic_list,
                                               const std::vector<i_t>& nonbasic_list,
                                               i_t branch_var,
                                               f_t branch_var_lower,
                                               f_t branch_var_upper,
                                               f_t upper_bound,
                                               f_t start_time,
                                               i_t iter_limit,
                                               omp_atomic_t<int64_t>& total_lp_iter)
{
  lp_problem_t child_problem      = original_lp;
  child_problem.lower[branch_var] = branch_var_lower;
  child_problem.upper[branch_var] = branch_var_upper;

  const bool initialize_basis                        = false;
  simplex_solver_settings_t<i_t, f_t> child_settings = settings;
  child_settings.set_log(false);
  child_settings.iteration_limit = iter_limit;
  child_settings.inside_mip      = 2;
  child_settings.scale_columns   = false;
  child_settings.cut_off =
    objective_upper_bound(child_problem, upper_bound, child_settings.dual_tol);

  lp_solution_t<i_t, f_t> solution(original_lp.num_rows, original_lp.num_cols);
  i_t iter                                         = 0;
  std::vector<variable_status_t> child_vstatus     = vstatus;
  std::vector<f_t> child_edge_norms                = edge_norms;
  std::vector<i_t> child_basic_list                = basic_list;
  std::vector<i_t> child_nonbasic_list             = nonbasic_list;
  basis_update_mpf_t<i_t, f_t> child_basis_factors = basis_factors;

  // Only refactor the basis if we encounter numerical issues.
  child_basis_factors.set_refactor_frequency(iter_limit);

  dual::status_t status = dual_phase2_with_advanced_basis(2,
                                                          0,
                                                          initialize_basis,
                                                          start_time,
                                                          child_problem,
                                                          child_settings,
                                                          child_vstatus,
                                                          child_basis_factors,
                                                          child_basic_list,
                                                          child_nonbasic_list,
                                                          solution,
                                                          iter,
                                                          child_edge_norms);
  total_lp_iter += iter;
  settings.log.debug("Trial branching on variable %d. Lo: %e Up: %e. Iter %d. Status %s. Obj %e\n",
                     branch_var,
                     child_problem.lower[branch_var],
                     child_problem.upper[branch_var],
                     iter,
                     dual::status_to_string(status).c_str(),
                     compute_objective(child_problem, solution.x));

  if (status == dual::status_t::DUAL_UNBOUNDED) {
    // LP was infeasible
    return {std::numeric_limits<f_t>::infinity(), dual::status_t::DUAL_UNBOUNDED};
  } else if (status == dual::status_t::OPTIMAL || status == dual::status_t::ITERATION_LIMIT ||
             status == dual::status_t::CUTOFF) {
    return {compute_objective(child_problem, solution.x), status};
  } else {
    return {std::numeric_limits<f_t>::quiet_NaN(), dual::status_t::NUMERICAL};
  }
}

}  // namespace

template <typename i_t, typename f_t>
static cuopt::mps_parser::mps_data_model_t<i_t, f_t> simplex_problem_to_mps_data_model(
  const dual_simplex::lp_problem_t<i_t, f_t>& lp,
  const std::vector<i_t>& new_slacks,
  const std::vector<f_t>& root_soln,
  std::vector<f_t>& original_root_soln_x)
{
  // Branch and bound has a problem of the form:
  // minimize c^T x
  // subject to A*x + Es = b
  //            l <= x <= u
  //            E_{jj} = sigma_j, where sigma_j is +1 or -1

  // We need to convert this into a problem that is better for PDLP
  // to solve. PDLP perfers inequality constraints. Thus, we want
  // to convert the above into the problem:
  // minimize c^T x
  // subject to  lb <= A*x <= ub
  //             l <= x <= u

  cuopt::mps_parser::mps_data_model_t<i_t, f_t> mps_model;
  int m = lp.num_rows;
  int n = lp.num_cols - new_slacks.size();
  original_root_soln_x.resize(n);

  // Remove slacks from A
  dual_simplex::csc_matrix_t<i_t, f_t> A_no_slacks = lp.A;
  std::vector<i_t> cols_to_remove(lp.A.n, 0);
  for (i_t j : new_slacks) {
    cols_to_remove[j] = 1;
  }
  A_no_slacks.remove_columns(cols_to_remove);

  for (i_t j = 0; j < n; j++) {
    original_root_soln_x[j] = root_soln[j];
  }

  // Convert CSC to CSR using built-in method
  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, 0);
  A_no_slacks.to_compressed_row(csr_A);

  int nz = csr_A.row_start[m];

  // Set CSR constraint matrix
  mps_model.set_csr_constraint_matrix(
    csr_A.x.data(), nz, csr_A.j.data(), nz, csr_A.row_start.data(), m + 1);

  // Set objective coefficients
  mps_model.set_objective_coefficients(lp.objective.data(), n);

  // The LP is already in minimization form (objective negated for max problems).
  // Pass identity scaling so PDLP returns the raw DS-space objective directly.
  mps_model.set_objective_scaling_factor(f_t(1.0));
  mps_model.set_objective_offset(f_t(0.0));

  // Set variable bounds
  mps_model.set_variable_lower_bounds(lp.lower.data(), n);
  mps_model.set_variable_upper_bounds(lp.upper.data(), n);

  // Convert row sense and RHS to constraint bounds
  std::vector<f_t> constraint_lower(m);
  std::vector<f_t> constraint_upper(m);

  std::vector<i_t> slack_map(m, -1);
  for (i_t j : new_slacks) {
    const i_t col_start = lp.A.col_start[j];
    const i_t i         = lp.A.i[col_start];
    slack_map[i]        = j;
  }

  for (i_t i = 0; i < m; ++i) {
    // Each row is of the form a_i^T x + sigma * s_i = b_i
    // with sigma = +1 or -1
    // and l_i <= s_i <= u_i
    // We have that a_i^T x - b_i = -sigma * s_i
    // If sigma = -1, then we have
    //    a_i^T x - b_i = s_i
    //  l_i <= a_i^T x - b_i <= u_i
    //  l_i + b_i <= a_i^T x <= u_i + b_i
    //
    // If sigma = +1, then we have
    //    a_i^T x - b_i = -s_i
    //   -a_i^T x + b_i = s_i
    //  l_i <= -a_i^T x + b_i <= u_i
    //  l_i - b_i <= -a_i^T x <= u_i - b_i
    //  -u_i + b_i <= a_i^T x <= -l_i + b_i

    const i_t slack = slack_map[i];
    assert(slack != -1);
    const i_t col_start   = lp.A.col_start[slack];
    const f_t sigma       = lp.A.x[col_start];
    const f_t slack_lower = lp.lower[slack];
    const f_t slack_upper = lp.upper[slack];

    if (sigma == -1) {
      constraint_lower[i] = slack_lower + lp.rhs[i];
      constraint_upper[i] = slack_upper + lp.rhs[i];
    } else if (sigma == 1) {
      constraint_lower[i] = -slack_upper + lp.rhs[i];
      constraint_upper[i] = -slack_lower + lp.rhs[i];
    } else {
      assert(sigma == 1.0 || sigma == -1.0);
    }
  }

  mps_model.set_constraint_lower_bounds(constraint_lower.data(), m);
  mps_model.set_constraint_upper_bounds(constraint_upper.data(), m);
  mps_model.set_maximize(false);

  return mps_model;
}

enum class sb_source_t { DUAL_SIMPLEX, PDLP, NONE };

// Merge a single strong branching result from Dual Simplex and PDLP.
// Rules:
//   1. If both found optimal   -> keep DS (higher quality vertex solution)
//   2. Else if Dual Simplex found infeasible -> declare infeasible
//   3. Else if one is optimal -> keep the optimal one
//   4. Else if Dual Simplex hit iteration limit -> keep DS
//   5. Else if none converged -> NaN (original objective)
template <typename i_t, typename f_t>
static std::pair<f_t, sb_source_t> merge_sb_result(f_t dual_simplex_val,
                                                   dual::status_t dual_simplex_status,
                                                   f_t pdlp_dual_obj,
                                                   bool pdlp_optimal)
{
  // Dual simplex always maintains dual feasibility, so OPTIMAL and ITERATION_LIMIT both qualify

  // Rule 1: Both optimal -> keep DS
  if (dual_simplex_status == dual::status_t::OPTIMAL && pdlp_optimal) {
    return {dual_simplex_val, sb_source_t::DUAL_SIMPLEX};
  }

  // Rule 2: Dual Simplex found infeasible -> declare infeasible
  if (dual_simplex_status == dual::status_t::DUAL_UNBOUNDED) {
    return {std::numeric_limits<f_t>::infinity(), sb_source_t::DUAL_SIMPLEX};
  }

  // Rule 3: Only one converged -> keep that
  if (dual_simplex_status == dual::status_t::OPTIMAL && !pdlp_optimal) {
    return {dual_simplex_val, sb_source_t::DUAL_SIMPLEX};
  }
  if (pdlp_optimal && dual_simplex_status != dual::status_t::OPTIMAL) {
    return {pdlp_dual_obj, sb_source_t::PDLP};
  }

  // Rule 4: Dual Simplex hit iteration limit or work limit or cutoff -> keep DS
  if (dual_simplex_status == dual::status_t::ITERATION_LIMIT ||
      dual_simplex_status == dual::status_t::WORK_LIMIT ||
      dual_simplex_status == dual::status_t::CUTOFF) {
    return {dual_simplex_val, sb_source_t::DUAL_SIMPLEX};
  }

  // Rule 5: None converged -> NaN
  return {std::numeric_limits<f_t>::quiet_NaN(), sb_source_t::NONE};
}

template <typename i_t, typename f_t>
static void batch_pdlp_strong_branching_task(
  const simplex_solver_settings_t<i_t, f_t>& settings,
  i_t effective_batch_pdlp,
  f_t start_time,
  std::atomic<int>& concurrent_halt,
  const lp_problem_t<i_t, f_t>& original_lp,
  const std::vector<i_t>& new_slacks,
  const std::vector<f_t>& root_soln,
  const std::vector<i_t>& fractional,
  f_t root_obj,
  pseudo_costs_t<i_t, f_t>& pc,
  shared_strong_branching_context_view_t<i_t, f_t>& sb_view,
  std::vector<f_t>& pdlp_obj_down,
  std::vector<f_t>& pdlp_obj_up)
{
  constexpr bool verbose = false;

  settings.log.printf(effective_batch_pdlp == 2
                        ? "Batch PDLP only for strong branching\n"
                        : "Cooperative batch PDLP and Dual Simplex for strong branching\n");

  f_t start_batch = tic();
  std::vector<f_t> original_root_soln_x;

  if (concurrent_halt.load() == 1) { return; }

  const auto mps_model =
    simplex_problem_to_mps_data_model(original_lp, new_slacks, root_soln, original_root_soln_x);

  std::vector<f_t> fraction_values;

  std::vector<f_t> original_root_soln_y, original_root_soln_z;
  // TODO put back later once Chris has this part
  /*uncrush_dual_solution(
    original_problem, original_lp, root_soln_y, root_soln_z, original_root_soln_y,
    original_root_soln_z);*/

  for (i_t k = 0; k < fractional.size(); k++) {
    const i_t j = fractional[k];
    fraction_values.push_back(original_root_soln_x[j]);
  }

  if (concurrent_halt.load() == 1) { return; }

  f_t batch_elapsed_time = toc(start_time);
  const f_t warm_start_remaining_time =
    std::max(static_cast<f_t>(0.0), settings.time_limit - batch_elapsed_time);
  if (warm_start_remaining_time <= 0.0) { return; }

  assert(!pc.pdlp_warm_cache.populated && "PDLP warm cache should not be populated at this point");

  if (!pc.pdlp_warm_cache.populated) {
    pdlp_solver_settings_t<i_t, f_t> ws_settings;
    ws_settings.method               = method_t::PDLP;
    ws_settings.presolver            = presolver_t::None;
    ws_settings.pdlp_solver_mode     = pdlp_solver_mode_t::Stable3;
    ws_settings.detect_infeasibility = false;
    // Since the warm start will be used over and over again we want to maximize the chance of
    // convergeance Batch PDLP is very compute intensive so we want to minimize the number of
    // iterations
    constexpr int warm_start_iteration_limit         = 500000;
    ws_settings.iteration_limit                      = warm_start_iteration_limit;
    ws_settings.time_limit                           = warm_start_remaining_time;
    constexpr f_t pdlp_tolerance                     = 1e-5;
    ws_settings.tolerances.relative_dual_tolerance   = pdlp_tolerance;
    ws_settings.tolerances.absolute_dual_tolerance   = pdlp_tolerance;
    ws_settings.tolerances.relative_primal_tolerance = pdlp_tolerance;
    ws_settings.tolerances.absolute_primal_tolerance = pdlp_tolerance;
    ws_settings.tolerances.relative_gap_tolerance    = pdlp_tolerance;
    ws_settings.tolerances.absolute_gap_tolerance    = pdlp_tolerance;
    ws_settings.inside_mip                           = true;
    if (effective_batch_pdlp == 1) { ws_settings.concurrent_halt = &concurrent_halt; }

    auto start_time = std::chrono::high_resolution_clock::now();

    auto ws_solution = solve_lp(&pc.pdlp_warm_cache.batch_pdlp_handle, mps_model, ws_settings);

    if (verbose) {
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
      settings.log.printf(
        "Original problem solved in %d milliseconds"
        " and iterations: %d\n",
        duration,
        ws_solution.get_pdlp_warm_start_data().total_pdlp_iterations_);
    }

    if (ws_solution.get_termination_status() == pdlp_termination_status_t::Optimal) {
      auto& cache           = pc.pdlp_warm_cache;
      const auto& ws_primal = ws_solution.get_primal_solution();
      const auto& ws_dual   = ws_solution.get_dual_solution();
      // Need to use the pc steam since the batch pdlp handle will get destroyed after the warm
      // start
      cache.initial_primal = rmm::device_uvector<f_t>(ws_primal, ws_primal.stream());
      cache.initial_dual   = rmm::device_uvector<f_t>(ws_dual, ws_dual.stream());
      cache.step_size      = ws_solution.get_pdlp_warm_start_data().initial_step_size_;
      cache.primal_weight  = ws_solution.get_pdlp_warm_start_data().initial_primal_weight_;
      cache.pdlp_iteration = ws_solution.get_pdlp_warm_start_data().total_pdlp_iterations_;
      cache.populated      = true;

      if (verbose) {
        settings.log.printf(
          "Cached PDLP warm start: primal=%zu dual=%zu step_size=%e primal_weight=%e iters=%d\n",
          cache.initial_primal.size(),
          cache.initial_dual.size(),
          cache.step_size,
          cache.primal_weight,
          cache.pdlp_iteration);
      }
    } else {
      if (verbose) {
        settings.log.printf(
          "PDLP warm start solve did not reach optimality (%s), skipping cache and batch PDLP\n",
          ws_solution.get_termination_status_string().c_str());
      }
      return;
    }
  }

  if (concurrent_halt.load() == 1) { return; }

  pdlp_solver_settings_t<i_t, f_t> pdlp_settings;
  if (effective_batch_pdlp == 1) {
    pdlp_settings.concurrent_halt  = &concurrent_halt;
    pdlp_settings.shared_sb_solved = sb_view.solved;
  }

  batch_elapsed_time = toc(start_time);
  const f_t batch_remaining_time =
    std::max(static_cast<f_t>(0.0), settings.time_limit - batch_elapsed_time);
  if (batch_remaining_time <= 0.0) { return; }
  pdlp_settings.time_limit = batch_remaining_time;

  if (pc.pdlp_warm_cache.populated) {
    auto& cache = pc.pdlp_warm_cache;
    pdlp_settings.set_initial_primal_solution(cache.initial_primal.data(),
                                              cache.initial_primal.size(),
                                              cache.batch_pdlp_handle.get_stream());
    pdlp_settings.set_initial_dual_solution(
      cache.initial_dual.data(), cache.initial_dual.size(), cache.batch_pdlp_handle.get_stream());
    pdlp_settings.set_initial_step_size(cache.step_size);
    pdlp_settings.set_initial_primal_weight(cache.primal_weight);
    pdlp_settings.set_initial_pdlp_iteration(cache.pdlp_iteration);
  }

  if (concurrent_halt.load() == 1) { return; }

  const auto solutions = batch_pdlp_solve(
    &pc.pdlp_warm_cache.batch_pdlp_handle, mps_model, fractional, fraction_values, pdlp_settings);
  f_t batch_pdlp_strong_branching_time = toc(start_batch);

  // Fail safe in case the batch PDLP failed and produced no solutions
  if (solutions.get_additional_termination_informations().size() != fractional.size() * 2) {
    if (verbose) { settings.log.printf("Batch PDLP failed and produced no solutions\n"); }
    return;
  }

  // Find max iteration on how many are done accross the batch
  i_t max_iterations = 0;
  i_t amount_done    = 0;
  for (i_t k = 0; k < solutions.get_additional_termination_informations().size(); k++) {
    max_iterations = std::max(
      max_iterations, solutions.get_additional_termination_information(k).number_of_steps_taken);
    // TODO batch mode infeasible: should also count as done if infeasible
    if (solutions.get_termination_status(k) == pdlp_termination_status_t::Optimal) {
      amount_done++;
    }
  }

  if (verbose) {
    settings.log.printf(
      "Batch PDLP strong branching completed in %.2fs. Solved %d/%d with max %d iterations\n",
      batch_pdlp_strong_branching_time,
      amount_done,
      fractional.size() * 2,
      max_iterations);
  }

  for (i_t k = 0; k < fractional.size(); k++) {
    if (solutions.get_termination_status(k) == pdlp_termination_status_t::Optimal) {
      pdlp_obj_down[k] = std::max(solutions.get_dual_objective_value(k) - root_obj, f_t(0.0));
    }
    if (solutions.get_termination_status(k + fractional.size()) ==
        pdlp_termination_status_t::Optimal) {
      pdlp_obj_up[k] =
        std::max(solutions.get_dual_objective_value(k + fractional.size()) - root_obj, f_t(0.0));
    }
  }
}

template <typename i_t, typename f_t>
static void batch_pdlp_reliability_branching_task(
  logger_t& log,
  i_t rb_mode,
  i_t num_candidates,
  f_t start_time,
  std::atomic<int>& concurrent_halt,
  const lp_problem_t<i_t, f_t>& original_lp,
  const std::vector<i_t>& new_slacks,
  const std::vector<f_t>& solution,
  branch_and_bound_worker_t<i_t, f_t>* worker,
  const std::vector<i_t>& candidate_vars,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  shared_strong_branching_context_view_t<i_t, f_t>& sb_view,
  batch_pdlp_warm_cache_t<i_t, f_t>& pdlp_warm_cache,
  std::vector<f_t>& pdlp_obj_down,
  std::vector<f_t>& pdlp_obj_up)
{
  log.printf(rb_mode == 2 ? "RB batch PDLP only for %d candidates\n"
                          : "RB cooperative batch PDLP and DS for %d candidates\n",
             num_candidates);

  f_t start_batch = tic();

  std::vector<f_t> original_soln_x;

  if (concurrent_halt.load() == 1) { return; }

  auto mps_model =
    simplex_problem_to_mps_data_model(original_lp, new_slacks, solution, original_soln_x);
  {
    const i_t n_orig = original_lp.num_cols - new_slacks.size();
    for (i_t j = 0; j < n_orig; j++) {
      mps_model.variable_lower_bounds_[j] = worker->leaf_problem.lower[j];
      mps_model.variable_upper_bounds_[j] = worker->leaf_problem.upper[j];
    }
  }

  std::vector<f_t> fraction_values;
  fraction_values.reserve(num_candidates);
  for (i_t j : candidate_vars) {
    fraction_values.push_back(original_soln_x[j]);
  }

  if (concurrent_halt.load() == 1) { return; }

  const f_t batch_elapsed_time = toc(start_time);
  const f_t batch_remaining_time =
    std::max(static_cast<f_t>(0.0), settings.time_limit - batch_elapsed_time);
  if (batch_remaining_time <= 0.0) { return; }

  // One handle per batch PDLP since there can be concurrent calls
  const raft::handle_t batch_pdlp_handle;

  pdlp_solver_settings_t<i_t, f_t> pdlp_settings;
  if (rb_mode == 1) {
    pdlp_settings.concurrent_halt  = &concurrent_halt;
    pdlp_settings.shared_sb_solved = sb_view.solved;
  }
  pdlp_settings.time_limit = batch_remaining_time;

  if (pdlp_warm_cache.populated) {
    auto& cache = pdlp_warm_cache;
    pdlp_settings.set_initial_primal_solution(
      cache.initial_primal.data(), cache.initial_primal.size(), batch_pdlp_handle.get_stream());
    pdlp_settings.set_initial_dual_solution(
      cache.initial_dual.data(), cache.initial_dual.size(), batch_pdlp_handle.get_stream());
    pdlp_settings.set_initial_step_size(cache.step_size);
    pdlp_settings.set_initial_primal_weight(cache.primal_weight);
    pdlp_settings.set_initial_pdlp_iteration(cache.pdlp_iteration);
  }

  if (concurrent_halt.load() == 1) { return; }

  const auto solutions =
    batch_pdlp_solve(&batch_pdlp_handle, mps_model, candidate_vars, fraction_values, pdlp_settings);

  f_t batch_pdlp_time = toc(start_batch);

  if (solutions.get_additional_termination_informations().size() !=
      static_cast<size_t>(num_candidates) * 2) {
    log.printf("RB batch PDLP failed and produced no solutions\n");
    return;
  }

  i_t amount_done = 0;
  for (i_t k = 0; k < num_candidates * 2; k++) {
    if (solutions.get_termination_status(k) == pdlp_termination_status_t::Optimal) {
      amount_done++;
    }
  }

  log.printf("RB batch PDLP completed in %.2fs. Solved %d/%d\n",
             batch_pdlp_time,
             amount_done,
             num_candidates * 2);

  for (i_t k = 0; k < num_candidates; k++) {
    if (solutions.get_termination_status(k) == pdlp_termination_status_t::Optimal) {
      pdlp_obj_down[k] = solutions.get_dual_objective_value(k);
    }
    if (solutions.get_termination_status(k + num_candidates) ==
        pdlp_termination_status_t::Optimal) {
      pdlp_obj_up[k] = solutions.get_dual_objective_value(k + num_candidates);
    }
  }
}

template <typename i_t, typename f_t>
void strong_branching(const lp_problem_t<i_t, f_t>& original_lp,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      f_t start_time,
                      const std::vector<i_t>& new_slacks,
                      const std::vector<variable_type_t>& var_types,
                      const lp_solution_t<i_t, f_t>& root_solution,
                      const std::vector<i_t>& fractional,
                      f_t root_obj,
                      f_t upper_bound,
                      const std::vector<variable_status_t>& root_vstatus,
                      const std::vector<f_t>& edge_norms,
                      const std::vector<i_t>& basic_list,
                      const std::vector<i_t>& nonbasic_list,
                      basis_update_mpf_t<i_t, f_t>& basis_factors,
                      pseudo_costs_t<i_t, f_t>& pc)
{
  constexpr bool verbose = false;

  pc.resize(original_lp.num_cols);
  pc.strong_branch_down.assign(fractional.size(), 0);
  pc.strong_branch_up.assign(fractional.size(), 0);
  pc.num_strong_branches_completed = 0;

  const f_t elapsed_time = toc(start_time);
  if (elapsed_time > settings.time_limit) { return; }

  // 0: no batch PDLP, 1: cooperative batch PDLP and DS, 2: batch PDLP only
  const i_t effective_batch_pdlp =
    (settings.sub_mip || (settings.deterministic && settings.mip_batch_pdlp_strong_branching == 1))
      ? 0
      : settings.mip_batch_pdlp_strong_branching;

  if (settings.mip_batch_pdlp_strong_branching != 0 &&
      (settings.sub_mip || settings.deterministic)) {
    settings.log.printf(
      "Batch PDLP strong branching is disabled because sub-MIP or deterministic mode is enabled\n");
  }

  settings.log.printf("Strong branching using %d threads and %ld fractional variables\n",
                      settings.num_threads,
                      fractional.size());

  // Cooperative DS + PDLP: shared context tracks which subproblems are solved
  shared_strong_branching_context_t<i_t, f_t> shared_ctx(2 * fractional.size());
  shared_strong_branching_context_view_t<i_t, f_t> sb_view(shared_ctx.solved);

  std::atomic<int> concurrent_halt{0};

  std::vector<f_t> pdlp_obj_down(fractional.size(), std::numeric_limits<f_t>::quiet_NaN());
  std::vector<f_t> pdlp_obj_up(fractional.size(), std::numeric_limits<f_t>::quiet_NaN());

  std::vector<dual::status_t> dual_simplex_status_down(fractional.size(), dual::status_t::UNSET);
  std::vector<dual::status_t> dual_simplex_status_up(fractional.size(), dual::status_t::UNSET);
  std::vector<f_t> dual_simplex_obj_down(fractional.size(), std::numeric_limits<f_t>::quiet_NaN());
  std::vector<f_t> dual_simplex_obj_up(fractional.size(), std::numeric_limits<f_t>::quiet_NaN());
  f_t strong_branching_start_time = tic();
  i_t simplex_iteration_limit     = settings.strong_branching_simplex_iteration_limit;

  if (simplex_iteration_limit < 1) {
    initialize_pseudo_costs_with_estimate(original_lp,
                                          settings,
                                          root_vstatus,
                                          root_solution,
                                          basic_list,
                                          nonbasic_list,
                                          fractional,
                                          basis_factors,
                                          pc);
  } else {
#pragma omp parallel num_threads(settings.num_threads)
    {
#pragma omp single nowait
      {
        if (effective_batch_pdlp != 0) {
#pragma omp task
          batch_pdlp_strong_branching_task(settings,
                                           effective_batch_pdlp,
                                           start_time,
                                           concurrent_halt,
                                           original_lp,
                                           new_slacks,
                                           root_solution.x,
                                           fractional,
                                           root_obj,
                                           pc,
                                           sb_view,
                                           pdlp_obj_down,
                                           pdlp_obj_up);
        }

        if (effective_batch_pdlp != 2) {
          i_t n = std::min<i_t>(4 * settings.num_threads, fractional.size());
// Here we are creating more tasks than the number of threads
// such that they can be scheduled dynamically to the threads.
#pragma omp taskloop num_tasks(n)
          for (i_t k = 0; k < n; k++) {
            i_t start = std::floor(k * fractional.size() / n);
            i_t end   = std::floor((k + 1) * fractional.size() / n);

            constexpr bool verbose = false;
            if (verbose) {
              settings.log.printf("Thread id %d task id %d start %d end %d. size %d\n",
                                  omp_get_thread_num(),
                                  k,
                                  start,
                                  end,
                                  end - start);
            }

            strong_branch_helper(start,
                                 end,
                                 start_time,
                                 original_lp,
                                 settings,
                                 var_types,
                                 fractional,
                                 root_solution.x,
                                 root_vstatus,
                                 edge_norms,
                                 root_obj,
                                 upper_bound,
                                 simplex_iteration_limit,
                                 pc,
                                 dual_simplex_obj_down,
                                 dual_simplex_obj_up,
                                 dual_simplex_status_down,
                                 dual_simplex_status_up,
                                 sb_view);
          }
          // DS done: signal PDLP to stop (time-limit or all work done) and wait
          if (effective_batch_pdlp == 1) { concurrent_halt.store(1); }
        }
      }
    }
  }

  settings.log.printf("Strong branching completed in %.2fs\n", toc(strong_branching_start_time));

  if (verbose) {
    // Collect Dual Simplex statistics
    i_t dual_simplex_optimal = 0, dual_simplex_infeasible = 0, dual_simplex_iter_limit = 0;
    i_t dual_simplex_numerical = 0, dual_simplex_cutoff = 0, dual_simplex_time_limit = 0;
    i_t dual_simplex_concurrent = 0, dual_simplex_work_limit = 0, dual_simplex_unset = 0;
    const i_t total_subproblems = fractional.size() * 2;
    for (i_t k = 0; k < fractional.size(); k++) {
      for (auto st : {dual_simplex_status_down[k], dual_simplex_status_up[k]}) {
        switch (st) {
          case dual::status_t::OPTIMAL: dual_simplex_optimal++; break;
          case dual::status_t::DUAL_UNBOUNDED: dual_simplex_infeasible++; break;
          case dual::status_t::ITERATION_LIMIT: dual_simplex_iter_limit++; break;
          case dual::status_t::NUMERICAL: dual_simplex_numerical++; break;
          case dual::status_t::CUTOFF: dual_simplex_cutoff++; break;
          case dual::status_t::TIME_LIMIT: dual_simplex_time_limit++; break;
          case dual::status_t::CONCURRENT_LIMIT: dual_simplex_concurrent++; break;
          case dual::status_t::WORK_LIMIT: dual_simplex_work_limit++; break;
          case dual::status_t::UNSET: dual_simplex_unset++; break;
        }
      }
    }

    settings.log.printf("Dual Simplex: %d/%d optimal, %d infeasible, %d iter-limit",
                        dual_simplex_optimal,
                        total_subproblems,
                        dual_simplex_infeasible,
                        dual_simplex_iter_limit);
    if (dual_simplex_cutoff) settings.log.printf(", %d cutoff", dual_simplex_cutoff);
    if (dual_simplex_time_limit) settings.log.printf(", %d time-limit", dual_simplex_time_limit);
    if (dual_simplex_numerical) settings.log.printf(", %d numerical", dual_simplex_numerical);
    if (dual_simplex_concurrent)
      settings.log.printf(", %d concurrent-halt", dual_simplex_concurrent);
    if (dual_simplex_work_limit) settings.log.printf(", %d work-limit", dual_simplex_work_limit);
    if (dual_simplex_unset) settings.log.printf(", %d unset/skipped", dual_simplex_unset);
    settings.log.printf("\n");
  }

  if (effective_batch_pdlp != 0 && verbose) {
    i_t pdlp_optimal_count = 0;
    for (i_t k = 0; k < fractional.size(); k++) {
      if (!std::isnan(pdlp_obj_down[k])) pdlp_optimal_count++;
      if (!std::isnan(pdlp_obj_up[k])) pdlp_optimal_count++;
    }

    settings.log.printf("Batch PDLP found %d/%d optimal solutions\n",
                        pdlp_optimal_count,
                        static_cast<int>(fractional.size() * 2));
  }

  if (effective_batch_pdlp != 0) {
    i_t merged_from_ds   = 0;
    i_t merged_from_pdlp = 0;
    i_t merged_nan       = 0;
    i_t solved_by_both   = 0;
    for (i_t k = 0; k < fractional.size(); k++) {
      for (i_t branch = 0; branch < 2; branch++) {
        const bool is_down = (branch == 0);
        f_t& sb_dest       = is_down ? pc.strong_branch_down[k] : pc.strong_branch_up[k];
        f_t ds_obj         = is_down ? dual_simplex_obj_down[k] : dual_simplex_obj_up[k];
        dual::status_t ds_status =
          is_down ? dual_simplex_status_down[k] : dual_simplex_status_up[k];
        f_t pdlp_obj  = is_down ? pdlp_obj_down[k] : pdlp_obj_up[k];
        bool pdlp_has = !std::isnan(pdlp_obj);
        bool ds_has   = ds_status != dual::status_t::UNSET;

        const auto [value, source] =
          merge_sb_result<i_t, f_t>(ds_obj, ds_status, pdlp_obj, pdlp_has);

        if (source == sb_source_t::PDLP || effective_batch_pdlp == 2) { sb_dest = value; }

        if (source == sb_source_t::DUAL_SIMPLEX)
          merged_from_ds++;
        else if (source == sb_source_t::PDLP)
          merged_from_pdlp++;
        else
          merged_nan++;

        if (ds_has && pdlp_has && verbose) {
          solved_by_both++;
          settings.log.printf(
            "[COOP SB] Merge: variable %d %s solved by BOTH (DS=%e PDLP=%e) -> kept %s\n",
            fractional[k],
            is_down ? "DOWN" : "UP",
            ds_obj,
            pdlp_obj,
            source == sb_source_t::DUAL_SIMPLEX ? "DS" : "PDLP");
        }
      }
    }

    pc.pdlp_warm_cache.percent_solved_by_batch_pdlp_at_root =
      (f_t(merged_from_pdlp) / f_t(fractional.size() * 2)) * 100.0;
    if (verbose) {
      settings.log.printf(
        "Batch PDLP for strong branching. Percent solved by batch PDLP at root: %f\n",
        pc.pdlp_warm_cache.percent_solved_by_batch_pdlp_at_root);
      settings.log.printf(
        "Merged results: %d from DS, %d from PDLP, %d unresolved (NaN), %d solved by both\n",
        merged_from_ds,
        merged_from_pdlp,
        merged_nan,
        solved_by_both);
    }
  }

  pc.update_pseudo_costs_from_strong_branching(fractional, root_solution.x);
}

template <typename i_t, typename f_t>
f_t pseudo_costs_t<i_t, f_t>::calculate_pseudocost_score(i_t j,
                                                         const std::vector<f_t>& solution,
                                                         f_t pseudo_cost_up_avg,
                                                         f_t pseudo_cost_down_avg) const
{
  constexpr f_t eps = 1e-6;
  i_t num_up        = pseudo_cost_num_up[j];
  i_t num_down      = pseudo_cost_num_down[j];
  f_t pc_up         = num_up > 0 ? pseudo_cost_sum_up[j] / num_up : pseudo_cost_up_avg;
  f_t pc_down       = num_down > 0 ? pseudo_cost_sum_down[j] / num_down : pseudo_cost_down_avg;
  f_t f_down        = solution[j] - std::floor(solution[j]);
  f_t f_up          = std::ceil(solution[j]) - solution[j];
  return std::max(f_down * pc_down, eps) * std::max(f_up * pc_up, eps);
}

template <typename i_t, typename f_t>
void pseudo_costs_t<i_t, f_t>::update_pseudo_costs(mip_node_t<i_t, f_t>* node_ptr,
                                                   f_t leaf_objective)
{
  const f_t change_in_obj = std::max(leaf_objective - node_ptr->lower_bound, 0.0);
  const f_t frac          = node_ptr->branch_dir == rounding_direction_t::DOWN
                              ? node_ptr->fractional_val - std::floor(node_ptr->fractional_val)
                              : std::ceil(node_ptr->fractional_val) - node_ptr->fractional_val;

  if (node_ptr->branch_dir == rounding_direction_t::DOWN) {
    pseudo_cost_sum_down[node_ptr->branch_var] += change_in_obj / frac;
    pseudo_cost_num_down[node_ptr->branch_var]++;
  } else {
    pseudo_cost_sum_up[node_ptr->branch_var] += change_in_obj / frac;
    pseudo_cost_num_up[node_ptr->branch_var]++;
  }
}

template <typename i_t, typename f_t>
void pseudo_costs_t<i_t, f_t>::initialized(i_t& num_initialized_down,
                                           i_t& num_initialized_up,
                                           f_t& pseudo_cost_down_avg,
                                           f_t& pseudo_cost_up_avg) const
{
  auto avgs            = compute_pseudo_cost_averages(pseudo_cost_sum_down.data(),
                                           pseudo_cost_sum_up.data(),
                                           pseudo_cost_num_down.data(),
                                           pseudo_cost_num_up.data(),
                                           pseudo_cost_sum_down.size());
  pseudo_cost_down_avg = avgs.down_avg;
  pseudo_cost_up_avg   = avgs.up_avg;
}

template <typename i_t, typename f_t>
i_t pseudo_costs_t<i_t, f_t>::variable_selection(const std::vector<i_t>& fractional,
                                                 const std::vector<f_t>& solution,
                                                 logger_t& log)
{
  i_t branch_var = fractional[0];
  f_t max_score  = -1;
  i_t num_initialized_down;
  i_t num_initialized_up;
  f_t pseudo_cost_down_avg;
  f_t pseudo_cost_up_avg;

  initialized(num_initialized_down, num_initialized_up, pseudo_cost_down_avg, pseudo_cost_up_avg);

  log.printf("PC: num initialized down %d up %d avg down %e up %e\n",
             num_initialized_down,
             num_initialized_up,
             pseudo_cost_down_avg,
             pseudo_cost_up_avg);

  for (i_t j : fractional) {
    f_t score = calculate_pseudocost_score(j, solution, pseudo_cost_up_avg, pseudo_cost_down_avg);

    if (score > max_score) {
      max_score  = score;
      branch_var = j;
    }
  }

  log.debug("Pseudocost branching on %d. Value %e. Score %e.\n",
            branch_var,
            solution[branch_var],
            max_score);

  return branch_var;
}

template <typename i_t, typename f_t>
i_t pseudo_costs_t<i_t, f_t>::reliable_variable_selection(
  const mip_node_t<i_t, f_t>* node_ptr,
  const std::vector<i_t>& fractional,
  branch_and_bound_worker_t<i_t, f_t>* worker,
  const std::vector<variable_type_t>& var_types,
  const branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  f_t upper_bound,
  int max_num_tasks,
  logger_t& log,
  const std::vector<i_t>& new_slacks,
  const lp_problem_t<i_t, f_t>& original_lp)
{
  constexpr f_t eps                      = 1e-6;
  f_t start_time                         = bnb_stats.start_time;
  i_t branch_var                         = fractional[0];
  f_t max_score                          = -1;
  f_t pseudo_cost_down_avg               = -1;
  f_t pseudo_cost_up_avg                 = -1;
  lp_solution_t<i_t, f_t>& leaf_solution = worker->leaf_solution;

  const int64_t branch_and_bound_lp_iters = bnb_stats.total_lp_iters;
  const i_t branch_and_bound_lp_iter_per_node =
    bnb_stats.nodes_explored > 0 ? branch_and_bound_lp_iters / bnb_stats.nodes_explored : 0;
  const i_t iter_limit_per_trial = std::clamp(2 * branch_and_bound_lp_iter_per_node,
                                              reliability_branching_settings.lower_max_lp_iter,
                                              reliability_branching_settings.upper_max_lp_iter);

  i_t reliable_threshold = settings.reliability_branching;
  if (reliable_threshold < 0) {
    const i_t max_threshold            = reliability_branching_settings.max_reliable_threshold;
    const i_t min_threshold            = reliability_branching_settings.min_reliable_threshold;
    const f_t iter_factor              = reliability_branching_settings.bnb_lp_factor;
    const i_t iter_offset              = reliability_branching_settings.bnb_lp_offset;
    const int64_t alpha                = iter_factor * branch_and_bound_lp_iters;
    const int64_t max_reliability_iter = alpha + reliability_branching_settings.bnb_lp_offset;

    f_t iter_fraction =
      (max_reliability_iter - strong_branching_lp_iter) / (strong_branching_lp_iter + 1.0);
    iter_fraction = std::min(1.0, iter_fraction);
    iter_fraction = std::max((alpha - strong_branching_lp_iter) / (strong_branching_lp_iter + 1.0),
                             iter_fraction);
    reliable_threshold = (1 - iter_fraction) * min_threshold + iter_fraction * max_threshold;
    reliable_threshold = strong_branching_lp_iter < max_reliability_iter ? reliable_threshold : 0;
  }

  // If `reliable_threshold == 0`, then we set the uninitialized pseudocosts to the average.
  // Otherwise, the best ones are initialized via strong branching, while the other are ignored.  //
  // In the latter, we are not using the average pseudocost (which calculated in the `initialized`
  // method).
  if (reliable_threshold == 0) {
    i_t num_initialized_up;
    i_t num_initialized_down;
    initialized(num_initialized_down, num_initialized_up, pseudo_cost_down_avg, pseudo_cost_up_avg);
    log.printf("PC: num initialized down %d up %d avg down %e up %e\n",
               num_initialized_down,
               num_initialized_up,
               pseudo_cost_down_avg,
               pseudo_cost_up_avg);
  }

  std::vector<std::pair<f_t, i_t>> unreliable_list;
  omp_mutex_t score_mutex;

  for (i_t j : fractional) {
    if (pseudo_cost_num_down[j] < reliable_threshold ||
        pseudo_cost_num_up[j] < reliable_threshold) {
      unreliable_list.push_back(std::make_pair(-1, j));
      continue;
    }
    f_t score =
      calculate_pseudocost_score(j, leaf_solution.x, pseudo_cost_up_avg, pseudo_cost_down_avg);

    if (score > max_score) {
      max_score  = score;
      branch_var = j;
    }
  }

  if (unreliable_list.empty()) {
    log.printf("pc branching on %d. Value %e. Score %e\n",
               branch_var,
               leaf_solution.x[branch_var],
               max_score);

    return branch_var;
  }

  // 0: no batch PDLP, 1: cooperative batch PDLP and DS, 2: batch PDLP only
  const i_t rb_mode = settings.mip_batch_pdlp_reliability_branching;
  // We don't use batch PDLP in reliability branching if the PDLP warm start data was not filled
  // This indicates that PDLP alone (not batched) couldn't even run at the root node
  // So it will most likely perform poorly compared to DS
  // It is also off if the number of candidate is very small
  // If warm start could run but almost none of the BPDLP results were used, we also want to avoid
  // using batch PDLP
  constexpr i_t min_num_candidates_for_pdlp                       = 5;
  constexpr f_t min_percent_solved_by_batch_pdlp_at_root_for_pdlp = 5.0;
  // Batch PDLP is either forced or we use the heuristic to decide if it should be used
  const bool use_pdlp = (rb_mode == 2) || (rb_mode != 0 && !settings.sub_mip &&
                                           !settings.deterministic && pdlp_warm_cache.populated &&
                                           unreliable_list.size() > min_num_candidates_for_pdlp &&
                                           pdlp_warm_cache.percent_solved_by_batch_pdlp_at_root >
                                             min_percent_solved_by_batch_pdlp_at_root_for_pdlp);

  if (rb_mode != 0 && !pdlp_warm_cache.populated) {
    log.printf("PDLP warm start data not populated, using DS only\n");
  } else if (rb_mode != 0 && settings.sub_mip) {
    log.printf("Batch PDLP reliability branching is disabled because sub-MIP is enabled\n");
  } else if (rb_mode != 0 && settings.deterministic) {
    log.printf(
      "Batch PDLP reliability branching is disabled because deterministic mode is enabled\n");
  } else if (rb_mode != 0 && unreliable_list.size() < min_num_candidates_for_pdlp) {
    log.printf("Not enough candidates to use batch PDLP, using DS only\n");
  } else if (rb_mode != 0 && pdlp_warm_cache.percent_solved_by_batch_pdlp_at_root < 5.0) {
    log.printf("Percent solved by batch PDLP at root is too low, using DS only\n");
  } else if (use_pdlp) {
    log.printf(
      "Using batch PDLP because populated, unreliable list size is %d (> %d), and percent solved "
      "by batch PDLP at root is %f%% (> %f%%)\n",
      static_cast<i_t>(unreliable_list.size()),
      min_num_candidates_for_pdlp,
      pdlp_warm_cache.percent_solved_by_batch_pdlp_at_root,
      min_percent_solved_by_batch_pdlp_at_root_for_pdlp);
  }

  const int num_tasks     = std::max(max_num_tasks, 10);
  const int task_priority = reliability_branching_settings.task_priority;
  // If both batch PDLP and DS are used we double the max number of candidates
  const i_t max_num_candidates = use_pdlp ? 2 * reliability_branching_settings.max_num_candidates
                                          : reliability_branching_settings.max_num_candidates;
  const i_t num_candidates     = std::min<size_t>(unreliable_list.size(), max_num_candidates);

  assert(task_priority > 0);
  assert(max_num_candidates > 0);
  assert(num_candidates > 0);
  assert(num_tasks > 0);

  log.printf(
    "RB iters = %d, B&B iters = %d, unreliable = %d, num_tasks = %d, reliable_threshold = %d\n",
    strong_branching_lp_iter.load(),
    branch_and_bound_lp_iters,
    unreliable_list.size(),
    num_tasks,
    reliable_threshold);

  if (unreliable_list.size() > max_num_candidates) {
    if (reliability_branching_settings.rank_candidates_with_dual_pivot) {
      i_t m             = worker->leaf_problem.num_rows;
      i_t n             = worker->leaf_problem.num_cols;
      f_t work_estimate = 0;

      std::vector<f_t> delta_z(n, 0);
      std::vector<i_t> workspace(n, 0);

      std::vector<i_t> basic_map(n, -1);
      for (i_t i = 0; i < m; i++) {
        basic_map[worker->basic_list[i]] = i;
      }

      std::vector<i_t> nonbasic_mark(n, -1);
      for (i_t i = 0; i < n - m; i++) {
        nonbasic_mark[worker->nonbasic_list[i]] = i;
      }

      for (auto& [score, j] : unreliable_list) {
        if (pseudo_cost_num_down[j] == 0 || pseudo_cost_num_up[j] == 0) {
          // Estimate the objective change by performing a single pivot of dual simplex.
          objective_change_estimate_t<f_t> estimate =
            single_pivot_objective_change_estimate(worker->leaf_problem,
                                                   settings,
                                                   AT,
                                                   node_ptr->vstatus,
                                                   j,
                                                   basic_map[j],
                                                   leaf_solution,
                                                   worker->basic_list,
                                                   worker->nonbasic_list,
                                                   nonbasic_mark,
                                                   worker->basis_factors,
                                                   workspace,
                                                   delta_z,
                                                   work_estimate);

          score = std::max(estimate.up_obj_change, eps) * std::max(estimate.down_obj_change, eps);
        } else {
          // Use the previous score, even if it is unreliable
          score = calculate_pseudocost_score(
            j, leaf_solution.x, pseudo_cost_up_avg, pseudo_cost_down_avg);
        }
      }
    } else {
      f_t high = max_score > 0 ? max_score : 1;
      f_t low  = 0;

      for (auto& [score, j] : unreliable_list) {
        if (score == -1) { score = worker->rng.uniform(low, high); }
      }
    }

    // We only need to get the top-k elements in the list, where
    // k = num_candidates
    std::partial_sort(unreliable_list.begin(),
                      unreliable_list.begin() + num_candidates,
                      unreliable_list.end(),
                      [](auto el1, auto el2) { return el1.first > el2.first; });
  }

  // Both DS and PDLP work on the same candidate set
  std::vector<i_t> candidate_vars(num_candidates);
  for (i_t i = 0; i < num_candidates; i++) {
    candidate_vars[i] = unreliable_list[i].second;
  }

  // Shared context for cooperative work-stealing (mode 1)
  // [0..num_candidates) = down, [num_candidates..2*num_candidates) = up
  shared_strong_branching_context_t<i_t, f_t> shared_ctx(2 * num_candidates);
  shared_strong_branching_context_view_t<i_t, f_t> sb_view(shared_ctx.solved);

  std::vector<f_t> pdlp_obj_down(num_candidates, std::numeric_limits<f_t>::quiet_NaN());
  std::vector<f_t> pdlp_obj_up(num_candidates, std::numeric_limits<f_t>::quiet_NaN());

  std::atomic<int> concurrent_halt{0};

  if (use_pdlp) {
#pragma omp task default(shared)
    batch_pdlp_reliability_branching_task(log,
                                          rb_mode,
                                          num_candidates,
                                          start_time,
                                          concurrent_halt,
                                          original_lp,
                                          new_slacks,
                                          leaf_solution.x,
                                          worker,
                                          candidate_vars,
                                          settings,
                                          sb_view,
                                          pdlp_warm_cache,
                                          pdlp_obj_down,
                                          pdlp_obj_up);
  }

  if (toc(start_time) > settings.time_limit) {
    log.printf("Time limit reached\n");
    if (use_pdlp) {
      concurrent_halt.store(1);
#pragma omp taskwait
    }
    return branch_var;
  }

  std::vector<f_t> dual_simplex_obj_down(num_candidates, std::numeric_limits<f_t>::quiet_NaN());
  std::vector<f_t> dual_simplex_obj_up(num_candidates, std::numeric_limits<f_t>::quiet_NaN());
  std::vector<dual::status_t> dual_simplex_status_down(num_candidates, dual::status_t::UNSET);
  std::vector<dual::status_t> dual_simplex_status_up(num_candidates, dual::status_t::UNSET);

  f_t dual_simplex_start_time = tic();

  if (rb_mode != 2) {
#pragma omp taskloop if (num_tasks > 1) priority(task_priority) num_tasks(num_tasks) \
  shared(score_mutex,                                                                \
           sb_view,                                                                  \
           dual_simplex_obj_down,                                                    \
           dual_simplex_obj_up,                                                      \
           dual_simplex_status_down,                                                 \
           dual_simplex_status_up,                                                   \
           unreliable_list)
    for (i_t i = 0; i < num_candidates; ++i) {
      auto [score, j] = unreliable_list[i];

      if (toc(start_time) > settings.time_limit) { continue; }

      if (rb_mode == 1 && sb_view.is_solved(i)) {
        log.printf(
          "DS skipping variable %d branch down (shared_idx %d): already solved by PDLP\n", j, i);
      } else {
        pseudo_cost_mutex_down[j].lock();
        if (pseudo_cost_num_down[j] < reliable_threshold) {
          // Do trial branching on the down branch
          const auto [obj, status] = trial_branching(worker->leaf_problem,
                                                     settings,
                                                     var_types,
                                                     node_ptr->vstatus,
                                                     worker->leaf_edge_norms,
                                                     worker->basis_factors,
                                                     worker->basic_list,
                                                     worker->nonbasic_list,
                                                     j,
                                                     worker->leaf_problem.lower[j],
                                                     std::floor(leaf_solution.x[j]),
                                                     upper_bound,
                                                     start_time,
                                                     iter_limit_per_trial,
                                                     strong_branching_lp_iter);

          dual_simplex_obj_down[i]    = obj;
          dual_simplex_status_down[i] = status;
          if (!std::isnan(obj)) {
            f_t change_in_obj = std::max(obj - node_ptr->lower_bound, eps);
            f_t change_in_x   = leaf_solution.x[j] - std::floor(leaf_solution.x[j]);
            pseudo_cost_sum_down[j] += change_in_obj / change_in_x;
            pseudo_cost_num_down[j]++;
            // Should be valid if were are already here
            if (rb_mode == 1 && is_dual_simplex_done(status)) { sb_view.mark_solved(i); }
          }
        } else {
          // Variable became reliable, make it as solved so that batch PDLP does not solve it again
          if (rb_mode == 1) sb_view.mark_solved(i);
        }
        pseudo_cost_mutex_down[j].unlock();
      }

      if (toc(start_time) > settings.time_limit) { continue; }

      const i_t shared_idx = i + num_candidates;
      if (rb_mode == 1 && sb_view.is_solved(shared_idx)) {
        log.printf("DS skipping variable %d branch up (shared_idx %d): already solved by PDLP\n",
                   j,
                   shared_idx);
      } else {
        pseudo_cost_mutex_up[j].lock();
        if (pseudo_cost_num_up[j] < reliable_threshold) {
          const auto [obj, status] = trial_branching(worker->leaf_problem,
                                                     settings,
                                                     var_types,
                                                     node_ptr->vstatus,
                                                     worker->leaf_edge_norms,
                                                     worker->basis_factors,
                                                     worker->basic_list,
                                                     worker->nonbasic_list,
                                                     j,
                                                     std::ceil(leaf_solution.x[j]),
                                                     worker->leaf_problem.upper[j],
                                                     upper_bound,
                                                     start_time,
                                                     iter_limit_per_trial,
                                                     strong_branching_lp_iter);

          dual_simplex_obj_up[i]    = obj;
          dual_simplex_status_up[i] = status;
          if (!std::isnan(obj)) {
            f_t change_in_obj = std::max(obj - node_ptr->lower_bound, eps);
            f_t change_in_x   = std::ceil(leaf_solution.x[j]) - leaf_solution.x[j];
            pseudo_cost_sum_up[j] += change_in_obj / change_in_x;
            pseudo_cost_num_up[j]++;
            // Should be valid if were are already here
            if (rb_mode == 1 && is_dual_simplex_done(status)) { sb_view.mark_solved(shared_idx); }
          }
        } else {
          // Variable became reliable, make it as solved so that batch PDLP does not solve it again
          if (rb_mode == 1) sb_view.mark_solved(shared_idx);
        }
        pseudo_cost_mutex_up[j].unlock();
      }

      if (toc(start_time) > settings.time_limit) { continue; }

      score =
        calculate_pseudocost_score(j, leaf_solution.x, pseudo_cost_up_avg, pseudo_cost_down_avg);

      score_mutex.lock();
      if (score > max_score) {
        max_score  = score;
        branch_var = j;
      }
      score_mutex.unlock();
    }

    concurrent_halt.store(1);
  }

  f_t dual_simplex_elapsed = toc(dual_simplex_start_time);

  // TODO put back
  // if (rb_mode != 2) {
  //  if (rb_mode == 1) {
  //    log.printf(
  //      "RB Dual Simplex: %d candidates, %d/%d optimal, %d/%d infeasible, %d/%d failed, %d skipped
  //      (PDLP) in %.2fs\n", num_candidates, dual_simplex_optimal.load(), num_candidates * 2,
  //      dual_simplex_infeasible.load(), num_candidates * 2,
  //      dual_simplex_failed.load(), num_candidates * 2,
  //      dual_simplex_skipped.load(), dual_simplex_elapsed);
  //  } else {
  //    log.printf(
  //      "RB Dual Simplex: %d candidates, %d/%d optimal, %d/%d infeasible, %d/%d failed in
  //      %.2fs\n", num_candidates, dual_simplex_optimal.load(), num_candidates * 2,
  //      dual_simplex_infeasible.load(), num_candidates * 2, dual_simplex_failed.load(),
  //      num_candidates * 2, dual_simplex_elapsed);
  //  }
  //}

  if (use_pdlp) {
#pragma omp taskwait

    i_t pdlp_applied = 0;
    i_t pdlp_optimal = 0;
    for (i_t i = 0; i < num_candidates; i++) {
      const i_t j = candidate_vars[i];

      // Down: check if PDLP should override DS
      if (!std::isnan(pdlp_obj_down[i])) {
        pdlp_optimal++;
        const auto [merged_obj, source] = merge_sb_result<i_t, f_t>(
          dual_simplex_obj_down[i], dual_simplex_status_down[i], pdlp_obj_down[i], true);
        // PDLP won the merge, update the pseudo-cost only if node is still unreliable (concurrent
        // calls may have made it reliable)
        if (source == sb_source_t::PDLP) {
          pseudo_cost_mutex_down[j].lock();
          if (pseudo_cost_num_down[j] < reliable_threshold) {
            f_t change_in_obj = std::max(merged_obj - node_ptr->lower_bound, eps);
            f_t change_in_x   = leaf_solution.x[j] - std::floor(leaf_solution.x[j]);
            pseudo_cost_sum_down[j] += change_in_obj / change_in_x;
            pseudo_cost_num_down[j]++;
            pdlp_applied++;
          }
          pseudo_cost_mutex_down[j].unlock();
        }
      }

      // Up: check if PDLP should override DS
      if (!std::isnan(pdlp_obj_up[i])) {
        pdlp_optimal++;
        const auto [merged_obj, source] = merge_sb_result<i_t, f_t>(
          dual_simplex_obj_up[i], dual_simplex_status_up[i], pdlp_obj_up[i], true);
        // PDLP won the merge, update the pseudo-cost only if node is still unreliable (concurrent
        // calls may have made it reliable)
        if (source == sb_source_t::PDLP) {
          pseudo_cost_mutex_up[j].lock();
          if (pseudo_cost_num_up[j] < reliable_threshold) {
            f_t change_in_obj = std::max(merged_obj - node_ptr->lower_bound, eps);
            f_t change_in_x   = std::ceil(leaf_solution.x[j]) - leaf_solution.x[j];
            pseudo_cost_sum_up[j] += change_in_obj / change_in_x;
            pseudo_cost_num_up[j]++;
            pdlp_applied++;
          }
          pseudo_cost_mutex_up[j].unlock();
        }
      }

      f_t score =
        calculate_pseudocost_score(j, leaf_solution.x, pseudo_cost_up_avg, pseudo_cost_down_avg);
      if (score > max_score) {
        max_score  = score;
        branch_var = j;
      }
    }

    log.printf("RB batch PDLP: %d candidates, %d/%d optimal, %d applied to pseudo-costs\n",
               num_candidates,
               pdlp_optimal,
               num_candidates * 2,
               pdlp_applied);
  }

  log.printf(
    "pc branching on %d. Value %e. Score %e\n", branch_var, leaf_solution.x[branch_var], max_score);

  return branch_var;
}

template <typename i_t, typename f_t>
f_t pseudo_costs_t<i_t, f_t>::obj_estimate(const std::vector<i_t>& fractional,
                                           const std::vector<f_t>& solution,
                                           f_t lower_bound,
                                           logger_t& log)
{
  const i_t num_fractional = fractional.size();
  f_t estimate             = lower_bound;

  i_t num_initialized_down;
  i_t num_initialized_up;
  f_t pseudo_cost_down_avg;
  f_t pseudo_cost_up_avg;

  initialized(num_initialized_down, num_initialized_up, pseudo_cost_down_avg, pseudo_cost_up_avg);

  for (i_t j : fractional) {
    constexpr f_t eps = 1e-6;
    i_t num_up        = pseudo_cost_num_up[j];
    i_t num_down      = pseudo_cost_num_down[j];
    f_t pc_up         = num_up > 0 ? pseudo_cost_sum_up[j] / num_up : pseudo_cost_up_avg;
    f_t pc_down       = num_down > 0 ? pseudo_cost_sum_down[j] / num_down : pseudo_cost_down_avg;
    f_t f_down        = solution[j] - std::floor(solution[j]);
    f_t f_up          = std::ceil(solution[j]) - solution[j];
    estimate += std::min(pc_down * f_down, pc_up * f_up);
  }

  log.printf("pseudocost estimate = %e\n", estimate);
  return estimate;
}

template <typename i_t, typename f_t>
void pseudo_costs_t<i_t, f_t>::update_pseudo_costs_from_strong_branching(
  const std::vector<i_t>& fractional, const std::vector<f_t>& root_soln)
{
  for (i_t k = 0; k < fractional.size(); k++) {
    const i_t j = fractional[k];
    for (i_t branch = 0; branch < 2; branch++) {
      if (branch == 0) {
        f_t change_in_obj = strong_branch_down[k];
        if (std::isnan(change_in_obj)) { continue; }
        f_t frac = root_soln[j] - std::floor(root_soln[j]);
        pseudo_cost_sum_down[j] += change_in_obj / frac;
        pseudo_cost_num_down[j]++;
      } else {
        f_t change_in_obj = strong_branch_up[k];
        if (std::isnan(change_in_obj)) { continue; }
        f_t frac = std::ceil(root_soln[j]) - root_soln[j];
        pseudo_cost_sum_up[j] += change_in_obj / frac;
        pseudo_cost_num_up[j]++;
      }
    }
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template class pseudo_costs_t<int, double>;

template void strong_branching<int, double>(const lp_problem_t<int, double>& original_lp,
                                            const simplex_solver_settings_t<int, double>& settings,
                                            double start_time,
                                            const std::vector<int>& new_slacks,
                                            const std::vector<variable_type_t>& var_types,
                                            const lp_solution_t<int, double>& root_solution,
                                            const std::vector<int>& fractional,
                                            double root_obj,
                                            double upper_bound,
                                            const std::vector<variable_status_t>& root_vstatus,
                                            const std::vector<double>& edge_norms,
                                            const std::vector<int>& basic_list,
                                            const std::vector<int>& nonbasic_list,
                                            basis_update_mpf_t<int, double>& basis_factors,
                                            pseudo_costs_t<int, double>& pc);

#endif

}  // namespace cuopt::linear_programming::dual_simplex
