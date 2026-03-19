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

#include <cuopt/linear_programming/solve.hpp>

#include <utilities/copy_helpers.hpp>

#include <raft/core/nvtx.hpp>

#include <omp.h>

namespace cuopt::linear_programming::dual_simplex {

namespace {

template <typename i_t, typename f_t>
void strong_branch_helper(i_t start,
                          i_t end,
                          f_t start_time,
                          const lp_problem_t<i_t, f_t>& original_lp,
                          const simplex_solver_settings_t<i_t, f_t>& settings,
                          const std::vector<variable_type_t>& var_types,
                          const std::vector<i_t>& fractional,
                          f_t root_obj,
                          const std::vector<f_t>& root_soln,
                          const std::vector<variable_status_t>& root_vstatus,
                          const std::vector<f_t>& edge_norms,
                          pseudo_costs_t<i_t, f_t>& pc,
                          std::vector<f_t>& ds_obj_down,
                          std::vector<f_t>& ds_obj_up,
                          std::vector<dual::status_t>& ds_status_down,
                          std::vector<dual::status_t>& ds_status_up,
                          std::atomic<int>* concurrent_halt,
                          shared_strong_branching_context_view_t<i_t, f_t>& sb_view)
{
  raft::common::nvtx::range scope("BB::strong_branch_helper");
  lp_problem_t child_problem = original_lp;

  assert(concurrent_halt != nullptr && "Concurrent halt pointer cannot be nullptr");

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
        settings.log.printf(
          "[COOP SB] DS thread %d skipping variable %d branch %s (shared_idx %d): already solved by PDLP\n",
          thread_id, j, branch == 0 ? "down" : "up", shared_idx);
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
      if (elapsed_time > settings.time_limit || *concurrent_halt == 1) { break; }
      child_settings.time_limit      = std::max(0.0, settings.time_limit - elapsed_time);
      child_settings.iteration_limit = 200;
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
      } else if (status == dual::status_t::OPTIMAL || status == dual::status_t::ITERATION_LIMIT) {
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
        pc.strong_branch_down[k] = std::max(obj - root_obj, 0.0);
        ds_obj_down[k] = std::max(obj - root_obj, 0.0);
        ds_status_down[k]        = status;
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
        pc.strong_branch_up[k] = std::max(obj - root_obj, 0.0);
        ds_obj_up[k] = std::max(obj - root_obj, 0.0);
        ds_status_up[k]        = status;
        if (verbose) {
          settings.log.printf(
            "Thread id %2d remaining %d variable %d branch %d obj %e change down %e change up %e "
            "time %.2f\n",
            thread_id,
            end - 1 - k,
            j,
            branch,
            obj,
            ds_obj_down[k],
            ds_obj_up[k],
            toc(start_time));
        }
      }
      // Mark the subproblem as solved so that batch PDLP removes it from the batch
      if (sb_view.is_valid()) {
        sb_view.mark_solved(shared_idx);
        settings.log.printf(
          "[COOP SB] DS thread %d solved variable %d branch %s (shared_idx %d), marking in shared context\n",
          thread_id, j, branch == 0 ? "down" : "up", shared_idx);
      }
      if (toc(start_time) > settings.time_limit || *concurrent_halt == 1) {
        break; 
      }
    }
    if (toc(start_time) > settings.time_limit || *concurrent_halt == 1) {
      break; 
    }

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

    if (toc(start_time) > settings.time_limit || *concurrent_halt == 1) {
      break; 
    }
  }
}

template <typename i_t, typename f_t>
f_t trial_branching(const lp_problem_t<i_t, f_t>& original_lp,
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
                    i_t bnb_lp_iter_per_node,
                    f_t start_time,
                    i_t upper_max_lp_iter,
                    i_t lower_max_lp_iter,
                    omp_atomic_t<int64_t>& total_lp_iter)
{
  lp_problem_t child_problem      = original_lp;
  child_problem.lower[branch_var] = branch_var_lower;
  child_problem.upper[branch_var] = branch_var_upper;

  const bool initialize_basis                        = false;
  simplex_solver_settings_t<i_t, f_t> child_settings = settings;
  child_settings.set_log(false);
  i_t lp_iter_upper              = upper_max_lp_iter;
  i_t lp_iter_lower              = lower_max_lp_iter;
  child_settings.iteration_limit = std::clamp(bnb_lp_iter_per_node, lp_iter_lower, lp_iter_upper);
  child_settings.cut_off         = upper_bound + settings.dual_tol;
  child_settings.inside_mip      = 2;
  child_settings.scale_columns   = false;

  lp_solution_t<i_t, f_t> solution(original_lp.num_rows, original_lp.num_cols);
  i_t iter                                         = 0;
  std::vector<variable_status_t> child_vstatus     = vstatus;
  std::vector<f_t> child_edge_norms                = edge_norms;
  std::vector<i_t> child_basic_list                = basic_list;
  std::vector<i_t> child_nonbasic_list             = nonbasic_list;
  basis_update_mpf_t<i_t, f_t> child_basis_factors = basis_factors;

  // Only refactor the basis if we encounter numerical issues.
  child_basis_factors.set_refactor_frequency(upper_max_lp_iter);

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
    return std::numeric_limits<f_t>::infinity();
  } else if (status == dual::status_t::OPTIMAL || status == dual::status_t::ITERATION_LIMIT ||
             status == dual::status_t::CUTOFF) {
    return compute_objective(child_problem, solution.x);
  } else {
    return std::numeric_limits<f_t>::quiet_NaN();
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

  // Set objective scaling and offset
  mps_model.set_objective_scaling_factor(lp.obj_scale);
  mps_model.set_objective_offset(lp.obj_constant);

  // Set variable bounds
  mps_model.set_variable_lower_bounds(lp.lower.data(), n);
  mps_model.set_variable_upper_bounds(lp.upper.data(), n);

  // Convert row sense and RHS to constraint bounds
  std::vector<f_t> constraint_lower(m);
  std::vector<f_t> constraint_upper(m);

  std::vector<i_t> slack_map(m, -1);
  for (i_t j : new_slacks) {
    const i_t col_start = lp.A.col_start[j];
    const i_t i = lp.A.i[col_start];
    slack_map[i] = j;
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
    const i_t col_start = lp.A.col_start[slack];
    const f_t sigma = lp.A.x[col_start];
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
  mps_model.set_maximize(lp.obj_scale < 0);

  return mps_model;
}

// Merge a single strong branching result from Dual Simplex and PDLP.
// Rules:
//   1. If both found optimal   -> keep DS (higher quality vertex solution)
//   2. Else if Dual Simplex found infeasible -> declare infeasible
//   3. Else if one is optimal -> keep the optimal one
//   4. Else if Dual Simplex hit iteration limit -> keep DS
//   5. Else if none converged -> NaN (original objective)
// Return {value, source} where source is 0 if Dual Simplex, 1 if PDLP, 2 if both
template <typename i_t, typename f_t>
static std::pair<f_t, i_t> merge_sb_result(f_t ds_val,
                    dual::status_t ds_status,
                    f_t pdlp_dual_obj,
                    bool pdlp_optimal)
{
  // Dual simplex always maintains dual feasibility, so OPTIMAL and ITERATION_LIMIT both qualify

  // Rule 1: Both optimal -> keep DS
  if (ds_status == dual::status_t::OPTIMAL && pdlp_optimal) { return {ds_val, 0}; }

  // Rule 2: Dual Simplex found infeasible -> declare infeasible
  if (ds_status == dual::status_t::DUAL_UNBOUNDED) { return {std::numeric_limits<f_t>::infinity(), 0}; }

  // Rule 3: Only one converged -> keep that
  if (ds_status == dual::status_t::OPTIMAL && !pdlp_optimal) { return {ds_val, 0}; }
  if (pdlp_optimal && ds_status != dual::status_t::OPTIMAL) { return {pdlp_dual_obj, 1}; }

  // Rule 4: Dual Simplex hit iteration limit -> keep DS
  if (ds_status == dual::status_t::ITERATION_LIMIT) { return {ds_val, 0}; }

  // Rule 5: None converged -> NaN
  return {std::numeric_limits<f_t>::quiet_NaN(), 2};
}


template <typename i_t, typename f_t>
void strong_branching(const lp_problem_t<i_t, f_t>& original_lp,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      f_t start_time,
                      const std::vector<i_t>& new_slacks,
                      const std::vector<variable_type_t>& var_types,
                      const std::vector<f_t>& root_soln,
                      const std::vector<f_t>& root_soln_y,
                      const std::vector<f_t>& root_soln_z,
                      const std::vector<i_t>& fractional,
                      f_t root_obj,
                      const std::vector<variable_status_t>& root_vstatus,
                      const std::vector<f_t>& edge_norms,
                      pseudo_costs_t<i_t, f_t>& pc)
{
  pc.resize(original_lp.num_cols);
  pc.strong_branch_down.assign(fractional.size(), 0);
  pc.strong_branch_up.assign(fractional.size(), 0);
  pc.num_strong_branches_completed = 0;

  settings.log.printf("Strong branching using %d threads and %ld fractional variables\n",
                      settings.num_threads,
                      fractional.size());

  // Cooperative DS + PDLP: shared context tracks which subproblems are solved
  shared_strong_branching_context_t<i_t, f_t> shared_ctx(2 * fractional.size());
  shared_strong_branching_context_view_t<i_t, f_t> sb_view(std::span(shared_ctx.solved));

  std::atomic<int> concurrent_halt{0};

  std::vector<f_t> pdlp_obj_down(fractional.size(), std::numeric_limits<f_t>::quiet_NaN());
  std::vector<f_t> pdlp_obj_up(fractional.size(), std::numeric_limits<f_t>::quiet_NaN());

  auto pdlp_thread = std::thread([&]() {
    
    if (settings.mip_batch_pdlp_strong_branching == 0)
     return;
  
    settings.log.printf(settings.mip_batch_pdlp_strong_branching == 2
      ? "Batch PDLP only for strong branching\n"
      : "Cooperative batch PDLP and Dual Simplex for strong branching\n");

    f_t start_batch = tic();
    std::vector<f_t> original_root_soln_x;

    const auto mps_model         = simplex_problem_to_mps_data_model(original_lp, new_slacks, root_soln, original_root_soln_x);


    std::vector<f_t> fraction_values;

    std::vector<f_t> original_root_soln_y, original_root_soln_z;
    // TODO put back later once Chris has this part
    /*uncrush_dual_solution(
      original_problem, original_lp, root_soln_y, root_soln_z, original_root_soln_y, original_root_soln_z);*/

    for (i_t k = 0; k < fractional.size(); k++) {
      const i_t j = fractional[k];
      fraction_values.push_back(original_root_soln_x[j]);
    }

    const f_t batch_elapsed_time = toc(start_time);
    const f_t batch_remaining_time =
      std::max(static_cast<f_t>(0.0), settings.time_limit - batch_elapsed_time);
    if (batch_remaining_time <= 0.0) { return; }

    pdlp_solver_settings_t<i_t, f_t> pdlp_settings;
    if (settings.mip_batch_pdlp_strong_branching == 1) {
      pdlp_settings.concurrent_halt = &concurrent_halt;
      pdlp_settings.shared_sb_view  = sb_view;
    }

    pdlp_settings.time_limit = batch_remaining_time;
    const raft::handle_t batch_pdlp_handle;
    constexpr bool dual_simplex_primal_dual = false;
    if (dual_simplex_primal_dual) {
      pdlp_settings.set_initial_primal_solution(
        original_root_soln_x.data(), original_root_soln_x.size(), batch_pdlp_handle.get_stream());
      pdlp_settings.set_initial_dual_solution(
        original_root_soln_y.data(), original_root_soln_y.size(), batch_pdlp_handle.get_stream());
    }
    const auto solutions =
      batch_pdlp_solve(&batch_pdlp_handle, mps_model, fractional, fraction_values, pdlp_settings);
    f_t batch_pdlp_strong_branching_time = toc(start_batch);

    // Fail safe in case the batch PDLP failed and produced no solutions
    if (solutions.get_additional_termination_informations().size() != fractional.size() * 2) {
      settings.log.printf("Batch PDLP failed and produced no solutions\n");
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

    settings.log.printf(
      "Batch PDLP strong branching completed in %.2fs. Solved %d/%d with max %d iterations\n",
      batch_pdlp_strong_branching_time,
      amount_done,
      fractional.size() * 2,
      max_iterations);

    for (i_t k = 0; k < fractional.size(); k++) {
      // Call BatchLP solver. Solve 2*fractional.size() subproblems.
      // Let j = fractional[k]. We want to solve the two trial branching problems
      // Branch down:
      // minimize c^T x
      // subject to lb <= A*x <= ub
      // x_j <= floor(root_soln[j])
      // l <= x < u
      // Let the optimal objective value of thie problem be obj_down
      f_t obj_down = (solutions.get_termination_status(k) == pdlp_termination_status_t::Optimal)
                       ? solutions.get_dual_objective_value(k)
                       : std::numeric_limits<f_t>::quiet_NaN();

      // Branch up:
      // minimize c^T x
      // subject to lb <= A*x <= ub
      // x_j >= ceil(root_soln[j])
      // Let the optimal objective value of thie problem be obj_up
      f_t obj_up = (solutions.get_termination_status(k + fractional.size()) ==
                    pdlp_termination_status_t::Optimal)
                     ? solutions.get_dual_objective_value(k + fractional.size())
                     : std::numeric_limits<f_t>::quiet_NaN();

      pdlp_obj_down[k] = std::max(obj_down - root_obj, f_t(0.0));
      pdlp_obj_up[k]   = std::max(obj_up - root_obj, f_t(0.0));
    }

  });
  
  std::vector<dual::status_t> ds_status_down(fractional.size(), dual::status_t::UNSET);
  std::vector<dual::status_t> ds_status_up(fractional.size(), dual::status_t::UNSET);
  std::vector<f_t> ds_obj_down(fractional.size(), std::numeric_limits<f_t>::quiet_NaN());
  std::vector<f_t> ds_obj_up(fractional.size(), std::numeric_limits<f_t>::quiet_NaN());
  f_t dual_simplex_strong_branching_time = tic();

  if (settings.mip_batch_pdlp_strong_branching != 2) {
#pragma omp parallel num_threads(settings.num_threads)
    {
      i_t n = std::min<i_t>(4 * settings.num_threads, fractional.size());

      // Here we are creating more tasks than the number of threads
      // such that they can be scheduled dynamically to the threads.
#pragma omp for schedule(dynamic, 1)
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
                             root_obj,
                             root_soln,
                             root_vstatus,
                             edge_norms,
                             pc,
                             ds_obj_down,
                             ds_obj_up,
                             ds_status_down,
                             ds_status_up,
                             &concurrent_halt,
                             sb_view);
      }
    }

  // DS done: signal PDLP to stop (time-limit or all work done) and wait
  concurrent_halt.store(1);
  }

  pdlp_thread.join();

  settings.log.printf("Strong branching took %.2fs\n", toc(dual_simplex_strong_branching_time));


  // Collect Dual Simplex statistics
  i_t ds_optimal = 0, ds_infeasible = 0, ds_iter_limit = 0;
  i_t ds_numerical = 0, ds_cutoff = 0, ds_time_limit = 0;
  i_t ds_concurrent = 0, ds_work_limit = 0, ds_unset = 0;
  const i_t total_subproblems = fractional.size() * 2;
  for (i_t k = 0; k < fractional.size(); k++) {
    for (auto st : {ds_status_down[k], ds_status_up[k]}) {
      switch (st) {
        case dual::status_t::OPTIMAL:          ds_optimal++;     break;
        case dual::status_t::DUAL_UNBOUNDED:   ds_infeasible++;  break;
        case dual::status_t::ITERATION_LIMIT:  ds_iter_limit++;  break;
        case dual::status_t::NUMERICAL:        ds_numerical++;   break;
        case dual::status_t::CUTOFF:           ds_cutoff++;      break;
        case dual::status_t::TIME_LIMIT:       ds_time_limit++;  break;
        case dual::status_t::CONCURRENT_LIMIT: ds_concurrent++;  break;
        case dual::status_t::WORK_LIMIT:       ds_work_limit++;  break;
        case dual::status_t::UNSET:            ds_unset++;       break;
      }
    }
  }

  settings.log.printf("Dual Simplex: %d/%d optimal, %d infeasible, %d iter-limit",
    ds_optimal, total_subproblems, ds_infeasible, ds_iter_limit);
  if (ds_cutoff)     settings.log.printf(", %d cutoff", ds_cutoff);
  if (ds_time_limit) settings.log.printf(", %d time-limit", ds_time_limit);
  if (ds_numerical)  settings.log.printf(", %d numerical", ds_numerical);
  if (ds_concurrent) settings.log.printf(", %d concurrent-halt", ds_concurrent);
  if (ds_work_limit) settings.log.printf(", %d work-limit", ds_work_limit);
  if (ds_unset)      settings.log.printf(", %d unset/skipped", ds_unset);
  settings.log.printf("\n");

  if (settings.mip_batch_pdlp_strong_branching != 0) {
    i_t pdlp_optimal_count = 0;
    for (i_t k = 0; k < fractional.size(); k++) {
      if (!std::isnan(pdlp_obj_down[k])) pdlp_optimal_count++;
      if (!std::isnan(pdlp_obj_up[k])) pdlp_optimal_count++;
    }

    settings.log.printf(
      "Batch PDLP found %d/%d optimal solutions\n",
      pdlp_optimal_count,
      fractional.size() * 2);
  }

  i_t merged_from_ds = 0;
  i_t merged_from_pdlp = 0;
  i_t merged_nan = 0;
  i_t solved_by_both_down = 0;
  i_t solved_by_both_up = 0;
  for (i_t k = 0; k < fractional.size(); k++) {
    bool ds_has_down   = ds_status_down[k] != dual::status_t::UNSET;
    bool pdlp_has_down = !std::isnan(pdlp_obj_down[k]);
    const auto [value_down, source_down] = merge_sb_result<i_t, f_t>(ds_obj_down[k], ds_status_down[k], pdlp_obj_down[k], pdlp_has_down);
    pc.strong_branch_down[k] = value_down;
    if (source_down == 0) merged_from_ds++;
    else if (source_down == 1) merged_from_pdlp++;
    else merged_nan++;
    if (ds_has_down && pdlp_has_down) {
      solved_by_both_down++;
      settings.log.printf(
        "[COOP SB] Merge: variable %d DOWN solved by BOTH (DS=%e PDLP=%e) -> kept %s\n",
        fractional[k], ds_obj_down[k], pdlp_obj_down[k], source_down == 0 ? "DS" : "PDLP");
    }

    bool ds_has_up   = ds_status_up[k] != dual::status_t::UNSET;
    bool pdlp_has_up = !std::isnan(pdlp_obj_up[k]);
    const auto [value_up, source_up] = merge_sb_result<i_t, f_t>(ds_obj_up[k], ds_status_up[k], pdlp_obj_up[k], pdlp_has_up);
    pc.strong_branch_up[k] = value_up;
    if (source_up == 0) merged_from_ds++;
    else if (source_up == 1) merged_from_pdlp++;
    else merged_nan++;
    if (ds_has_up && pdlp_has_up) {
      solved_by_both_up++;
      settings.log.printf(
        "[COOP SB] Merge: variable %d UP solved by BOTH (DS=%e PDLP=%e) -> kept %s\n",
        fractional[k], ds_obj_up[k], pdlp_obj_up[k], source_up == 0 ? "DS" : "PDLP");
    }
  }

  if (settings.mip_batch_pdlp_strong_branching != 0) {
    settings.log.printf(
      "Merged results: %d from DS, %d from PDLP, %d unresolved (NaN), %d/%d solved by both (down/up)\n",
      merged_from_ds,
      merged_from_pdlp,
      merged_nan,
      solved_by_both_down,
      solved_by_both_up);
  }

  pc.update_pseudo_costs_from_strong_branching(fractional, root_soln);
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
  mip_node_t<i_t, f_t>* node_ptr,
  const std::vector<i_t>& fractional,
  const std::vector<f_t>& solution,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const std::vector<variable_type_t>& var_types,
  branch_and_bound_worker_t<i_t, f_t>* worker,
  const branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
  f_t upper_bound,
  int max_num_tasks,
  logger_t& log,
  const std::vector<i_t>& new_slacks,
  const lp_problem_t<i_t, f_t>& original_lp)
{
  constexpr f_t eps = 1e-6;
  f_t start_time    = bnb_stats.start_time;
  i_t branch_var    = fractional[0];
  f_t max_score     = -1;
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

  const int64_t branch_and_bound_lp_iters = bnb_stats.total_lp_iters;
  const int64_t branch_and_bound_explored = bnb_stats.nodes_explored;
  const i_t branch_and_bound_lp_iter_per_node =
    branch_and_bound_lp_iters / bnb_stats.nodes_explored;

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

  std::vector<i_t> unreliable_list;
  omp_mutex_t score_mutex;

  for (i_t j : fractional) {
    if (pseudo_cost_num_down[j] < reliable_threshold ||
        pseudo_cost_num_up[j] < reliable_threshold) {
      unreliable_list.push_back(j);
      continue;
    }

    f_t score = calculate_pseudocost_score(j, solution, pseudo_cost_up_avg, pseudo_cost_down_avg);

    if (score > max_score) {
      max_score  = score;
      branch_var = j;
    }
  }

  if (unreliable_list.empty()) {
    log.printf(
      "pc branching on %d. Value %e. Score %e\n", branch_var, solution[branch_var], max_score);

    return branch_var;
  }

  const int num_tasks          = std::max(max_num_tasks, 1);
  const int task_priority      = reliability_branching_settings.task_priority;
  const i_t max_num_candidates = reliability_branching_settings.max_num_candidates;
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

  // Shuffle the unreliable list so every variable has the same chance to be selected.
  if (unreliable_list.size() > max_num_candidates) { worker->rng.shuffle(unreliable_list); }

  // Variables beyond num_candidates are solved by batch PDLP instead of Dual Simplex
  std::vector<i_t> pdlp_overflow_list;
  bool use_pdlp = settings.mip_batch_pdlp_reliability_branching == 1 &&
                  static_cast<i_t>(unreliable_list.size()) > num_candidates;
  if (use_pdlp) {
    pdlp_overflow_list.assign(unreliable_list.begin() + num_candidates, unreliable_list.end());
  }

  const i_t num_pdlp_vars = pdlp_overflow_list.size();
  std::vector<f_t> pdlp_obj_down(num_pdlp_vars, std::numeric_limits<f_t>::quiet_NaN());
  std::vector<f_t> pdlp_obj_up(num_pdlp_vars, std::numeric_limits<f_t>::quiet_NaN());

  // DS can halt PDLP via concurrent_halt, but not the other way around
  std::atomic<int> concurrent_halt{0};
  std::thread pdlp_thread;

  if (use_pdlp) {
    pdlp_thread = std::thread([&]() {
      log.printf("RB batch PDLP: solving %d overflow unreliable variables\n", num_pdlp_vars);

      f_t start_batch = tic();

      std::vector<f_t> original_soln_x;
      // Convert the original_lp that has cuts to a problem that is better for PDLP
      auto mps_model = simplex_problem_to_mps_data_model(
        original_lp, new_slacks, solution, original_soln_x);
      // Apply the bounds of the current leaf problem
      {
        const i_t n_orig = original_lp.num_cols - new_slacks.size();
        for (i_t j = 0; j < n_orig; j++) {
          mps_model.variable_lower_bounds_[j] = worker->leaf_problem.lower[j];
          mps_model.variable_upper_bounds_[j] = worker->leaf_problem.upper[j];
        }
      }

      std::vector<f_t> fraction_values;
      fraction_values.reserve(num_pdlp_vars);
      for (i_t j : pdlp_overflow_list) {
        fraction_values.push_back(original_soln_x[j]);
      }

      const f_t batch_elapsed_time    = toc(start_time);
      const f_t batch_remaining_time =
        std::max(static_cast<f_t>(0.0), settings.time_limit - batch_elapsed_time);
      if (batch_remaining_time <= 0.0) { return; }

      pdlp_solver_settings_t<i_t, f_t> pdlp_settings;
      pdlp_settings.concurrent_halt = &concurrent_halt;
      pdlp_settings.time_limit      = batch_remaining_time;

      const raft::handle_t batch_pdlp_handle;
      const auto solutions = batch_pdlp_solve(
        &batch_pdlp_handle, mps_model, pdlp_overflow_list, fraction_values, pdlp_settings);

      f_t batch_pdlp_time = toc(start_batch);

      if (solutions.get_additional_termination_informations().size() !=
          static_cast<size_t>(num_pdlp_vars) * 2) {
        log.printf("RB batch PDLP failed and produced no solutions\n");
        return;
      }

      i_t amount_done = 0;
      for (i_t k = 0; k < num_pdlp_vars * 2; k++) {
        if (solutions.get_termination_status(k) == pdlp_termination_status_t::Optimal) {
          amount_done++;
        }
      }

      log.printf("RB batch PDLP completed in %.2fs. Solved %d/%d in %.2fs\n",
                 batch_pdlp_time,
                 amount_done,
                 num_pdlp_vars * 2,
                 toc(start_batch));

      for (i_t k = 0; k < num_pdlp_vars; k++) {
        if (solutions.get_termination_status(k) == pdlp_termination_status_t::Optimal) {
          pdlp_obj_down[k] = solutions.get_dual_objective_value(k);
        }
        if (solutions.get_termination_status(k + num_pdlp_vars) ==
            pdlp_termination_status_t::Optimal) {
          pdlp_obj_up[k] = solutions.get_dual_objective_value(k + num_pdlp_vars);
        }
      }
    });
  }

  if (toc(start_time) > settings.time_limit) {
    log.printf("Time limit reached");
    if (use_pdlp) {
      concurrent_halt.store(1);
      pdlp_thread.join();
    }
    return branch_var;
  }

  omp_atomic_t<i_t> ds_optimal{0};
  omp_atomic_t<i_t> ds_infeasible{0};
  omp_atomic_t<i_t> ds_failed{0};
  f_t ds_start_time = tic();

#pragma omp taskloop if (num_tasks > 1) priority(task_priority) num_tasks(num_tasks) \
  shared(score_mutex, ds_optimal, ds_infeasible, ds_failed)
  for (i_t i = 0; i < num_candidates; ++i) {
    const i_t j = unreliable_list[i];

    if (toc(start_time) > settings.time_limit) { continue; }

    pseudo_cost_mutex_down[j].lock();
    if (pseudo_cost_num_down[j] < reliable_threshold) {
      // Do trial branching on the down branch
      f_t obj = trial_branching(worker->leaf_problem,
                                settings,
                                var_types,
                                node_ptr->vstatus,
                                worker->leaf_edge_norms,
                                worker->basis_factors,
                                worker->basic_list,
                                worker->nonbasic_list,
                                j,
                                worker->leaf_problem.lower[j],
                                std::floor(solution[j]),
                                upper_bound,
                                branch_and_bound_lp_iter_per_node,
                                start_time,
                                reliability_branching_settings.upper_max_lp_iter,
                                reliability_branching_settings.lower_max_lp_iter,
                                strong_branching_lp_iter);

      if (std::isnan(obj)) {
        ds_failed++;
      } else if (std::isinf(obj)) {
        ds_infeasible++;
        f_t change_in_obj = std::max(obj - node_ptr->lower_bound, eps);
        f_t change_in_x   = solution[j] - std::floor(solution[j]);
        pseudo_cost_sum_down[j] += change_in_obj / change_in_x;
        pseudo_cost_num_down[j]++;
      } else {
        ds_optimal++;
        f_t change_in_obj = std::max(obj - node_ptr->lower_bound, eps);
        f_t change_in_x   = solution[j] - std::floor(solution[j]);
        pseudo_cost_sum_down[j] += change_in_obj / change_in_x;
        pseudo_cost_num_down[j]++;
      }
    }
    pseudo_cost_mutex_down[j].unlock();

    if (toc(start_time) > settings.time_limit) { continue; }

    pseudo_cost_mutex_up[j].lock();
    if (pseudo_cost_num_up[j] < reliable_threshold) {
      f_t obj = trial_branching(worker->leaf_problem,
                                settings,
                                var_types,
                                node_ptr->vstatus,
                                worker->leaf_edge_norms,
                                worker->basis_factors,
                                worker->basic_list,
                                worker->nonbasic_list,
                                j,
                                std::ceil(solution[j]),
                                worker->leaf_problem.upper[j],
                                upper_bound,
                                branch_and_bound_lp_iter_per_node,
                                start_time,
                                reliability_branching_settings.upper_max_lp_iter,
                                reliability_branching_settings.lower_max_lp_iter,
                                strong_branching_lp_iter);

      if (std::isnan(obj)) {
        ds_failed++;
      } else if (std::isinf(obj)) {
        // Is it ok to process infinity obj like this?
        ds_infeasible++;
        f_t change_in_obj = std::max(obj - node_ptr->lower_bound, eps);
        f_t change_in_x   = std::ceil(solution[j]) - solution[j];
        pseudo_cost_sum_up[j] += change_in_obj / change_in_x;
        pseudo_cost_num_up[j]++;
      } else {
        ds_optimal++;
        f_t change_in_obj = std::max(obj - node_ptr->lower_bound, eps);
        f_t change_in_x   = std::ceil(solution[j]) - solution[j];
        pseudo_cost_sum_up[j] += change_in_obj / change_in_x;
        pseudo_cost_num_up[j]++;
      }
    }
    pseudo_cost_mutex_up[j].unlock();

    if (toc(start_time) > settings.time_limit) { continue; }

    f_t score = calculate_pseudocost_score(j, solution, pseudo_cost_up_avg, pseudo_cost_down_avg);

    score_mutex.lock();
    if (score > max_score) {
      max_score  = score;
      branch_var = j;
    }
    score_mutex.unlock();
  }

  f_t ds_elapsed = toc(ds_start_time);
  log.printf(
    "RB Dual Simplex: %d candidates, %d/%d optimal/dual-feasible, %d/%d infeasible, "
    "%d/%d failed in %.2fs\n",
    num_candidates,
    ds_optimal.load(),
    num_candidates * 2,
    ds_infeasible.load(),
    num_candidates * 2,
    ds_failed.load(),
    num_candidates * 2,
    ds_elapsed);

  if (use_pdlp) {
    // Dual Simplex is done on the main thread, telling Batch PDLP to stop
    concurrent_halt.store(1);
    pdlp_thread.join();

    i_t pdlp_optimal  = 0;
    for (i_t k = 0; k < num_pdlp_vars; k++) {
      const i_t j = pdlp_overflow_list[k];

      pseudo_cost_mutex_down[j].lock();
      if (!std::isnan(pdlp_obj_down[k])) {
        f_t change_in_obj = std::max(pdlp_obj_down[k] - node_ptr->lower_bound, eps);
        f_t change_in_x   = solution[j] - std::floor(solution[j]);
        pseudo_cost_sum_down[j] += change_in_obj / change_in_x;
        pseudo_cost_num_down[j]++;
        pdlp_optimal++;
      }
      pseudo_cost_mutex_down[j].unlock();

      pseudo_cost_mutex_up[j].lock();
      if (!std::isnan(pdlp_obj_up[k])) {
        f_t change_in_obj = std::max(pdlp_obj_up[k] - node_ptr->lower_bound, eps);
        f_t change_in_x   = std::ceil(solution[j]) - solution[j];
        pseudo_cost_sum_up[j] += change_in_obj / change_in_x;
        pseudo_cost_num_up[j]++;
        pdlp_optimal++;
      }
      pseudo_cost_mutex_up[j].unlock();

      f_t score =
        calculate_pseudocost_score(j, solution, pseudo_cost_up_avg, pseudo_cost_down_avg);
      if (score > max_score) {
        max_score  = score;
        branch_var = j;
      }
    }

    log.printf(
      "RB batch PDLP: %d candidates, %d/%d optimal\n",
      num_pdlp_vars,
      pdlp_optimal,
      num_pdlp_vars * 2);
  }

  log.printf(
    "pc branching on %d. Value %e. Score %e\n", branch_var, solution[branch_var], max_score);

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
                                            const std::vector<double>& root_soln,
                                            const std::vector<double>& root_soln_y,
                                            const std::vector<double>& root_soln_z,
                                            const std::vector<int>& fractional,
                                            double root_obj,
                                            const std::vector<variable_status_t>& root_vstatus,
                                            const std::vector<double>& edge_norms,
                                            pseudo_costs_t<int, double>& pc);

#endif

}  // namespace cuopt::linear_programming::dual_simplex
