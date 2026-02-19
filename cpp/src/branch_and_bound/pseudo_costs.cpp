/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <branch_and_bound/pseudo_costs.hpp>

#include <dual_simplex/phase2.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/tic_toc.hpp>

#include <cuopt/linear_programming/solve.hpp>

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
                          pseudo_costs_t<i_t, f_t>& pc)
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
        if (verbose) {
          settings.log.printf(
            "Thread id %2d remaining %d variable %d branch %d obj %e change down %e change up %e "
            "time %.2f\n",
            thread_id,
            end - 1 - k,
            j,
            branch,
            obj,
            pc.strong_branch_down[k],
            pc.strong_branch_up[k],
            toc(start_time));
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
  const dual_simplex::user_problem_t<i_t, f_t>& user_problem)
{
  cuopt::mps_parser::mps_data_model_t<i_t, f_t> mps_model;
  int m = user_problem.num_rows;
  int n = user_problem.num_cols;

  // Convert CSC to CSR using built-in method
  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, 0);
  user_problem.A.to_compressed_row(csr_A);

  int nz = csr_A.row_start[m];

  // Set CSR constraint matrix
  mps_model.set_csr_constraint_matrix(
    csr_A.x.data(), nz, csr_A.j.data(), nz, csr_A.row_start.data(), m + 1);

  // Set objective coefficients
  mps_model.set_objective_coefficients(user_problem.objective.data(), n);

  // Set objective scaling and offset
  mps_model.set_objective_scaling_factor(user_problem.obj_scale);
  mps_model.set_objective_offset(user_problem.obj_constant);

  // Set variable bounds
  mps_model.set_variable_lower_bounds(user_problem.lower.data(), n);
  mps_model.set_variable_upper_bounds(user_problem.upper.data(), n);

  // Convert row sense and RHS to constraint bounds
  std::vector<f_t> constraint_lower(m);
  std::vector<f_t> constraint_upper(m);

  for (i_t i = 0; i < m; ++i) {
    if (user_problem.row_sense[i] == 'L') {
      constraint_lower[i] = -std::numeric_limits<f_t>::infinity();
      constraint_upper[i] = user_problem.rhs[i];
    } else if (user_problem.row_sense[i] == 'G') {
      constraint_lower[i] = user_problem.rhs[i];
      constraint_upper[i] = std::numeric_limits<f_t>::infinity();
    } else {
      constraint_lower[i] = user_problem.rhs[i];
      constraint_upper[i] = user_problem.rhs[i];
    }
  }

  for (i_t k = 0; k < user_problem.num_range_rows; ++k) {
    i_t i = user_problem.range_rows[k];
    f_t r = user_problem.range_value[k];
    f_t b = user_problem.rhs[i];
    f_t h = -std::numeric_limits<f_t>::infinity();
    f_t u = std::numeric_limits<f_t>::infinity();
    if (user_problem.row_sense[i] == 'L') {
      h = b - std::abs(r);
      u = b;
    } else if (user_problem.row_sense[i] == 'G') {
      h = b;
      u = b + std::abs(r);
    } else if (user_problem.row_sense[i] == 'E') {
      if (r > 0) {
        h = b;
        u = b + std::abs(r);
      } else {
        h = b - std::abs(r);
        u = b;
      }
    }
    constraint_lower[i] = h;
    constraint_upper[i] = u;
  }

  mps_model.set_constraint_lower_bounds(constraint_lower.data(), m);
  mps_model.set_constraint_upper_bounds(constraint_upper.data(), m);
  mps_model.set_maximize(user_problem.obj_scale < 0);

  return mps_model;
}

template <typename i_t, typename f_t>
void strong_branching(const user_problem_t<i_t, f_t>& original_problem,
                      const lp_problem_t<i_t, f_t>& original_lp,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      f_t start_time,
                      const std::vector<variable_type_t>& var_types,
                      const std::vector<f_t> root_soln,
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

  if (settings.mip_batch_pdlp_strong_branching) {
    settings.log.printf("Batch PDLP strong branching enabled\n");

    f_t start_batch = tic();

    // Use original_problem to create the BatchLP problem
    csr_matrix_t<i_t, f_t> A_row(original_problem.A.m, original_problem.A.n, 0);
    original_problem.A.to_compressed_row(A_row);

    // Convert the root_soln to the original problem space
    std::vector<f_t> original_root_soln_x;
    uncrush_primal_solution(original_problem, original_lp, root_soln, original_root_soln_x);

    std::vector<f_t> fraction_values;

    for (i_t k = 0; k < fractional.size(); k++) {
      const i_t j = fractional[k];
      fraction_values.push_back(original_root_soln_x[j]);
    }

    const auto mps_model = simplex_problem_to_mps_data_model(original_problem);
    const raft::handle_t batch_pdlp_handle;
    const auto solutions =
      batch_pdlp_solve(&batch_pdlp_handle, mps_model, fractional, fraction_values);
    f_t batch_pdlp_strong_branching_time = toc(start_batch);

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

      pc.strong_branch_down[k] = obj_down - root_obj;
      pc.strong_branch_up[k]   = obj_up - root_obj;
    }
  } else {
    settings.log.printf("Strong branching using %d threads and %ld fractional variables\n",
                        settings.num_threads,
                        fractional.size());
    f_t strong_branching_start_time = tic();

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
                             pc);
      }
    }
    settings.log.printf("Strong branching completed in %.2fs\n", toc(strong_branching_start_time));
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
  logger_t& log)
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

  if (toc(start_time) > settings.time_limit) {
    log.printf("Time limit reached");
    return branch_var;
  }

#pragma omp taskloop if (num_tasks > 1) priority(task_priority) num_tasks(num_tasks) \
  shared(score_mutex)
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

      if (!std::isnan(obj)) {
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

      if (!std::isnan(obj)) {
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

template void strong_branching<int, double>(const user_problem_t<int, double>& original_problem,
                                            const lp_problem_t<int, double>& original_lp,
                                            const simplex_solver_settings_t<int, double>& settings,
                                            double start_time,
                                            const std::vector<variable_type_t>& var_types,
                                            const std::vector<double> root_soln,
                                            const std::vector<int>& fractional,
                                            double root_obj,
                                            const std::vector<variable_status_t>& root_vstatus,
                                            const std::vector<double>& edge_norms,
                                            pseudo_costs_t<int, double>& pc);

#endif

}  // namespace cuopt::linear_programming::dual_simplex
