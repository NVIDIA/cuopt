/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <mip/diversity/lns/rins.cuh>

#include <mip/diversity/diversity_manager.cuh>
#include <mip/feasibility_jump/fj_cpu.cuh>
#include <mip/mip_constants.hpp>
#include <mip/presolve/trivial_presolve.cuh>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
rins_t<i_t, f_t>::rins_t(mip_solver_context_t<i_t, f_t>& context_,
                         diversity_manager_t<i_t, f_t>& dm_,
                         rins_settings_t settings_)
  : context(context_), problem_ptr(context.problem_ptr), dm(dm_), settings(settings_)
{
  fixrate    = settings.default_fixrate;
  time_limit = settings.default_time_limit;
}

template <typename i_t, typename f_t>
rins_thread_t<i_t, f_t>::~rins_thread_t()
{
  this->request_termination();
}

template <typename i_t, typename f_t>
void rins_thread_t<i_t, f_t>::run_worker()
{
  raft::common::nvtx::range fun_scope("Running RINS");
  rins_ptr->run_rins();
}

template <typename i_t, typename f_t>
void rins_t<i_t, f_t>::new_best_incumbent_callback(const std::vector<f_t>& solution)
{
  node_count_at_last_improvement = node_count.load();
}

template <typename i_t, typename f_t>
void rins_t<i_t, f_t>::node_callback(const std::vector<f_t>& solution, f_t objective)
{
  if (!enabled) return;

  node_count++;

  if (node_count - node_count_at_last_improvement < settings.nodes_after_later_improvement) return;

  if (node_count - node_count_at_last_rins > settings.node_freq) {
    // opportunistic early test w/ atomic to avoid having to take the lock
    if (!rins_thread->cpu_thread_done) return;
    std::lock_guard<std::mutex> lock(rins_mutex);
    if (rins_thread->cpu_thread_done && dm.population.current_size() > 0 &&
        dm.population.is_feasible()) {
      lp_optimal_solution = solution;
      rins_thread->start_cpu_solver();
    }
  }
}

template <typename i_t, typename f_t>
void rins_t<i_t, f_t>::enable()
{
  rins_thread           = std::make_unique<rins_thread_t<i_t, f_t>>();
  rins_thread->rins_ptr = this;
  seed                  = cuopt::seed_generator::get_seed();
  problem_ptr->handle_ptr->sync_stream();
  problem_copy = std::make_unique<problem_t<i_t, f_t>>(*problem_ptr, &rins_handle);
  enabled      = true;
}

template <typename i_t, typename f_t>
void rins_t<i_t, f_t>::stop_rins()
{
  enabled = false;
  if (rins_thread) rins_thread->request_termination();
  rins_thread.reset();
}

template <typename i_t, typename f_t>
void rins_t<i_t, f_t>::run_rins()
{
  if (total_calls == 0) RAFT_CUDA_TRY(cudaSetDevice(context.handle_ptr->get_device()));

  if (!dm.population.is_feasible()) return;

  cuopt_assert(lp_optimal_solution.size() == problem_copy->n_variables, "Assignment size mismatch");
  cuopt_assert(problem_copy->handle_ptr == &rins_handle, "Handle mismatch");
  // Do not make assertions based on problem_ptr. The original problem may have been modified within
  // the FP loop relaxing integers cuopt_assert(problem_copy->n_variables ==
  // problem_ptr->n_variables, "Problem size mismatch"); cuopt_assert(problem_copy->n_constraints ==
  // problem_ptr->n_constraints, "Problem size mismatch"); cuopt_assert(problem_copy->n_integer_vars
  // == problem_ptr->n_integer_vars,
  //              "Problem size mismatch");
  // cuopt_assert(problem_copy->n_binary_vars == problem_ptr->n_binary_vars, "Problem size
  // mismatch");
  cuopt_assert(dm.population.current_size() > 0, "No solutions in population");

  solution_t<i_t, f_t> best_sol(*problem_copy);
  rins_handle.sync_stream();
  // copy the best from the population into a solution_t in the RINS stream
  {
    std::lock_guard<std::recursive_mutex> lock(dm.population.write_mutex);
    auto& best_feasible_ref = dm.population.best_feasible();
    cuopt_assert(best_feasible_ref.assignment.size() == best_sol.assignment.size(),
                 "Assignment size mismatch");
    cuopt_assert(best_feasible_ref.get_feasible(), "Best feasible is not feasible");
    expand_device_copy(best_sol.assignment, best_feasible_ref.assignment, rins_handle.get_stream());
    best_sol.handle_ptr  = &rins_handle;
    best_sol.problem_ptr = problem_copy.get();
    best_sol.compute_feasibility();
  }
  cuopt_assert(best_sol.handle_ptr == &rins_handle, "Handle mismatch");

  cuopt_assert(best_sol.get_feasible(), "Best solution is not feasible");
  if (!best_sol.get_feasible()) { return; }

  i_t sol_size_before_rins = best_sol.assignment.size();
  auto lp_opt_device = cuopt::device_copy(this->lp_optimal_solution, rins_handle.get_stream());
  cuopt_assert(lp_opt_device.size() == problem_copy->n_variables, "Assignment size mismatch");
  cuopt_assert(best_sol.assignment.size() == problem_copy->n_variables, "Assignment size mismatch");

  rmm::device_uvector<i_t> vars_to_fix(problem_copy->n_integer_vars, rins_handle.get_stream());
  auto end =
    thrust::copy_if(rins_handle.get_thrust_policy(),
                    problem_copy->integer_indices.begin(),
                    problem_copy->integer_indices.end(),
                    vars_to_fix.begin(),
                    [lpopt     = lp_opt_device.data(),
                     pb        = problem_copy->view(),
                     incumbent = best_sol.assignment.data()] __device__(i_t var_idx) -> bool {
                      return pb.integer_equal(lpopt[var_idx], incumbent[var_idx]);
                    });
  vars_to_fix.resize(end - vars_to_fix.begin(), rins_handle.get_stream());
  f_t fractional_ratio = (f_t)(vars_to_fix.size()) / (f_t)problem_copy->n_integer_vars;

  // abort if the fractional ratio is too low
  if (fractional_ratio < settings.min_fractional_ratio) {
    CUOPT_LOG_TRACE("RINS fractional ratio too low, aborting");
    return;
  }

  thrust::default_random_engine g(seed + node_count);

  // shuffle fixing order
  thrust::shuffle(rins_handle.get_thrust_policy(), vars_to_fix.begin(), vars_to_fix.end(), g);

  // fix n first according to fractional ratio
  f_t rins_ratio = fixrate;
  i_t n_to_fix   = std::max((int)(vars_to_fix.size() * rins_ratio), 0);
  vars_to_fix.resize(n_to_fix, rins_handle.get_stream());
  thrust::sort(rins_handle.get_thrust_policy(), vars_to_fix.begin(), vars_to_fix.end());

  cuopt_assert(thrust::all_of(rins_handle.get_thrust_policy(),
                              vars_to_fix.begin(),
                              vars_to_fix.end(),
                              [pb = problem_copy->view()] __device__(i_t var_idx) -> bool {
                                return pb.is_integer_var(var_idx);
                              }),
               "All variables to fix must be integer variables");

  if (n_to_fix == 0) {
    CUOPT_LOG_DEBUG("RINS no variables to fix");
    return;
  }

  total_calls++;
  node_count_at_last_rins = node_count.load();
  time_limit              = std::min(time_limit, dm.timer.remaining_time());
  CUOPT_LOG_DEBUG("Running RINS on solution with objective %g, fixing %d/%d",
                  best_sol.get_user_objective(),
                  vars_to_fix.size(),
                  problem_copy->n_integer_vars);
  CUOPT_LOG_DEBUG("RINS fixrate %g time limit %g", fixrate, time_limit);
  CUOPT_LOG_DEBUG("RINS fractional ratio %g%%", fractional_ratio * 100);

  f_t prev_obj = best_sol.get_user_objective();

  auto [fixed_problem, fixed_assignment, variable_map] = best_sol.fix_variables(vars_to_fix);
  CUOPT_LOG_DEBUG(
    "new var count %d var_count %d", fixed_problem.n_variables, problem_copy->n_integer_vars);

  // should probably just do an spmv to get the objective instead. ugly mess of copies
  solution_t<i_t, f_t> best_sol_fixed_space(fixed_problem);
  cuopt_assert(best_sol_fixed_space.handle_ptr == &rins_handle, "Handle mismatch");
  best_sol_fixed_space.copy_new_assignment(
    cuopt::host_copy(fixed_assignment, rins_handle.get_stream()));
  best_sol_fixed_space.compute_feasibility();
  CUOPT_LOG_DEBUG("RINS best sol fixed space objective %g",
                  best_sol_fixed_space.get_user_objective());

  if (settings.objective_cut) {
    f_t objective_cut =
      best_sol_fixed_space.get_objective() -
      std::max(std::abs(0.001 * best_sol_fixed_space.get_objective()), OBJECTIVE_EPSILON);
    fixed_problem.add_cutting_plane_at_objective(objective_cut);
  }

  fixed_problem.presolve_data.reset_additional_vars(fixed_problem, &rins_handle);
  fixed_problem.presolve_data.initialize_var_mapping(fixed_problem, &rins_handle);
  trivial_presolve(fixed_problem);
  fixed_problem.check_problem_representation(true);

  std::vector<std::vector<f_t>> rins_solution_queue;

  mip_solver_context_t<i_t, f_t> fj_context(
    &rins_handle, &fixed_problem, context.settings, context.scaling);
  fj_t<i_t, f_t> fj(fj_context);
  solution_t<i_t, f_t> fj_solution(fixed_problem);
  fj_solution.copy_new_assignment(cuopt::host_copy(fixed_assignment, rins_handle.get_stream()));
  std::vector<f_t> default_weights(fixed_problem.n_constraints, 1.);
  cpu_fj_thread_t<i_t, f_t> cpu_fj_thread;
  cpu_fj_thread.fj_cpu             = fj.create_cpu_climber(fj_solution,
                                               default_weights,
                                               default_weights,
                                               0.,
                                               context.preempt_heuristic_solver_,
                                               fj_settings_t{},
                                               true);
  cpu_fj_thread.fj_ptr             = &fj;
  cpu_fj_thread.fj_cpu->log_prefix = "[RINS] ";
  cpu_fj_thread.time_limit         = time_limit;
  cpu_fj_thread.start_cpu_solver();

  f_t lower_bound = context.branch_and_bound_ptr ? context.branch_and_bound_ptr->get_lower_bound()
                                                 : -std::numeric_limits<f_t>::infinity();
  f_t current_mip_gap = compute_rel_mip_gap(prev_obj, lower_bound);

  // run sub-mip
  namespace dual_simplex = cuopt::linear_programming::dual_simplex;
  dual_simplex::user_problem_t<i_t, f_t> branch_and_bound_problem(&rins_handle);
  dual_simplex::simplex_solver_settings_t<i_t, f_t> branch_and_bound_settings;
  dual_simplex::mip_solution_t<i_t, f_t> branch_and_bound_solution(1);
  dual_simplex::mip_status_t branch_and_bound_status = dual_simplex::mip_status_t::UNSET;
  fixed_problem.get_host_user_problem(branch_and_bound_problem);
  branch_and_bound_solution.resize(branch_and_bound_problem.num_cols);
  // Fill in the settings for branch and bound
  branch_and_bound_settings.time_limit = time_limit;
  // branch_and_bound_settings.node_limit = 5000 + node_count / 100;  // try harder as time goes
  // on
  branch_and_bound_settings.print_presolve_stats = false;
  branch_and_bound_settings.absolute_mip_gap_tol = context.settings.tolerances.absolute_mip_gap;
  branch_and_bound_settings.relative_mip_gap_tol =
    std::min(current_mip_gap, (f_t)settings.target_mip_gap);
  branch_and_bound_settings.integer_tol     = context.settings.tolerances.integrality_tolerance;
  branch_and_bound_settings.num_threads     = 2;
  branch_and_bound_settings.num_bfs_workers = 1;

  // In the future, let RINS use all the diving heuristics. For now,
  // restricting to guided diving.
  branch_and_bound_settings.diving_settings.num_diving_workers = 1;
  branch_and_bound_settings.diving_settings.line_search_diving = 0;
  branch_and_bound_settings.diving_settings.coefficient_diving = 0;
  branch_and_bound_settings.diving_settings.pseudocost_diving  = 0;
  branch_and_bound_settings.log.log                            = false;
  branch_and_bound_settings.log.log_prefix                     = "[RINS] ";
  branch_and_bound_settings.solution_callback = [&rins_solution_queue](std::vector<f_t>& solution,
                                                                       f_t objective) {
    rins_solution_queue.push_back(solution);
  };
  dual_simplex::branch_and_bound_t<i_t, f_t> branch_and_bound(branch_and_bound_problem,
                                                              branch_and_bound_settings);
  branch_and_bound.set_initial_guess(cuopt::host_copy(fixed_assignment, rins_handle.get_stream()));
  branch_and_bound_status = branch_and_bound.solve(branch_and_bound_solution);

  if (!std::isnan(branch_and_bound_solution.objective)) {
    CUOPT_LOG_DEBUG("RINS submip solution found. Objective %.16e. Status %d",
                    branch_and_bound_solution.objective,
                    int(branch_and_bound_status));
    // RINS submip may have just proved the initial guess is the optimal, therefore the queue might
    // be empty in that case
  }
  if (branch_and_bound_status == dual_simplex::mip_status_t::OPTIMAL) {
    CUOPT_LOG_DEBUG("RINS submip optimal");
    // do goldilocks update
    fixrate    = std::max(fixrate - 0.05, settings.min_fixrate);
    time_limit = std::max(time_limit - 2, settings.min_time_limit);
  } else if (branch_and_bound_status == dual_simplex::mip_status_t::TIME_LIMIT) {
    CUOPT_LOG_DEBUG("RINS submip time limit");
    // do goldilocks update
    fixrate    = std::min(fixrate + 0.05, settings.max_fixrate);
    time_limit = std::min(time_limit + 2, settings.max_time_limit);
  } else if (branch_and_bound_status == dual_simplex::mip_status_t::INFEASIBLE) {
    CUOPT_LOG_DEBUG("RINS submip infeasible");
    // do goldilocks update, decreasing fixrate
    fixrate = std::max(fixrate - 0.05, settings.min_fixrate);
  } else {
    CUOPT_LOG_DEBUG("RINS solution not found");
    // do goldilocks update
    fixrate    = std::min(fixrate + 0.05, settings.max_fixrate);
    time_limit = std::min(time_limit + 2, settings.max_time_limit);
  }

  cpu_fj_thread.stop_cpu_solver();
  bool fj_solution_found = cpu_fj_thread.wait_for_cpu_solver();
  CUOPT_LOG_DEBUG("RINS FJ ran for %d iterations", cpu_fj_thread.fj_cpu->iterations);
  if (fj_solution_found) {
    CUOPT_LOG_DEBUG("RINS FJ solution found. Objective %.16e",
                    cpu_fj_thread.fj_cpu->h_best_objective);
    rins_solution_queue.push_back(cpu_fj_thread.fj_cpu->h_best_assignment);
  }
  // Thread will be automatically terminated and joined by destructor

  bool improvement_found = false;
  for (auto& fixed_sol : rins_solution_queue) {
    cuopt_assert(fixed_assignment.size() == fixed_sol.size(), "Assignment size mismatch");
    rmm::device_uvector<f_t> post_processed_solution(fixed_sol.size(), rins_handle.get_stream());
    raft::copy(
      post_processed_solution.data(), fixed_sol.data(), fixed_sol.size(), rins_handle.get_stream());
    fixed_problem.post_process_assignment(post_processed_solution, false);
    cuopt_assert(post_processed_solution.size() == fixed_assignment.size(),
                 "Assignment size mismatch");
    rins_handle.sync_stream();

    rmm::device_uvector<f_t> unfixed_assignment(post_processed_solution.size(),
                                                rins_handle.get_stream());
    raft::copy(unfixed_assignment.data(),
               post_processed_solution.data(),
               post_processed_solution.size(),
               rins_handle.get_stream());
    best_sol.unfix_variables(unfixed_assignment, variable_map);
    best_sol.compute_feasibility();

    if (best_sol.get_feasible()) {
      cuopt_assert(best_sol.test_number_all_integer(), "All must be integers after RINS");
      if (best_sol.get_user_objective() < prev_obj) { improvement_found = true; }
      cuopt_assert(best_sol.assignment.size() == sol_size_before_rins, "Assignment size mismatch");
      cuopt_assert(best_sol.assignment.size() == problem_copy->n_variables,
                   "Assignment size mismatch");
      dm.population.add_external_solution(
        best_sol.get_host_assignment(), best_sol.get_objective(), solution_origin_t::RINS);
    }
  }

  if (improvement_found) total_success++;
  CUOPT_LOG_DEBUG("RINS calls/successes %d/%d", total_calls, total_success);
}

#if MIP_INSTANTIATE_FLOAT
template class rins_thread_t<int, float>;
template class rins_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class rins_thread_t<int, double>;
template class rins_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
