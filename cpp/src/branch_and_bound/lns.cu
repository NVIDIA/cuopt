/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "lns.cuh"

#include <branch_and_bound/branch_and_bound.hpp>
#include <branch_and_bound/fractional.hpp>
#include <branch_and_bound/utils.hpp>
#include <dual_simplex/presolve.hpp>
#include <mip_heuristics/feasibility_jump/fj_cpu_worker.cuh>
#include <presolve/third_party_presolve.hpp>
#include <utilities/scope_guard.hpp>

#include <raft/core/nvtx.hpp>

#define SUBMIP_VERBOSE false
#if SUBMIP_VERBOSE
#define DEBUG_SUBMIP(settings, fmt, ...) settings.log.print_format(fmt, __VA_ARGS__);
#else
#define DEBUG_SUBMIP(settings, fmt, ...)
#endif

namespace cuopt::mathematical_optimization::mip {

using simplex::domain_t;
using simplex::lp_problem_t;
using simplex::simplex_solver_settings_t;
using simplex::user_problem_t;
using simplex::variable_type_t;

namespace {

template <typename i_t, typename f_t>
f_t submip_get_max_fixrate(const submip_stats_t& stats,
                           const mip_hyper_params_t<i_t, f_t>& submip_settings,
                           pcgenerator_t& rng)
{
  // Adaptive fix rate based on previous successes and failures.
  f_t low  = submip_settings.submip_base_target_fixrate;
  f_t high = submip_settings.submip_base_target_fixrate;

  if (stats.total_infeasible > 0) {
    f_t infeasible_avg_fixrate = stats.average_infeasible_fixrate();
    high                       = 0.8 * infeasible_avg_fixrate;
    low                        = std::min(low, high);
  }

  if (stats.total_success > 0) {
    f_t success_avg_fixrate = stats.average_success_fixrate();
    low                     = std::min(low, 0.8 * success_avg_fixrate);
    high                    = std::max(high, 1.05 * success_avg_fixrate);
  }

  high        = std::min(high, 0.9);
  low         = std::max(low, 0.1);
  f_t fixrate = high > low ? rng.uniform(low, high) : low;
  return fixrate;
}

template <typename i_t, typename f_t>
void get_unfixed_integer_variables(const std::vector<f_t>& lower,
                                   const std::vector<f_t>& upper,
                                   const std::vector<variable_type_t>& var_types,
                                   f_t fixed_tol,
                                   std::vector<i_t>& integer_list)
{
  const i_t num_cols = var_types.size();
  for (i_t j = 0; j < num_cols; ++j) {
    if (var_types[j] == variable_type_t::CONTINUOUS) { continue; }
    if (std::abs(lower[j] - upper[j]) <= fixed_tol) { continue; }
    integer_list.push_back(j);
  }

  assert(!integer_list.empty() && "The integer list cannot be empty!");
}

template <typename i_t, typename f_t>
i_t calculate_fixrate(const std::vector<variable_type_t>& var_types,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      const simplex::domain_t<i_t, f_t>& domain,
                      const size_t start)
{
  i_t num_fixed = 0;

  for (size_t k = start; k < domain.size(); ++k) {
    const simplex::bound_change_t<i_t, f_t>& change = domain[k];
    if (var_types[change.var] != variable_type_t::CONTINUOUS &&
        change.old_upper - change.old_lower > settings.fixed_tol &&
        change.new_upper - change.new_lower <= settings.fixed_tol) {
      ++num_fixed;
    }
  }

  return num_fixed;
}

template <typename i_t, typename f_t>
i_t extend_variable_fixings(const csr_matrix_t<i_t, f_t>& Arow,
                            const std::vector<variable_type_t>& var_types,
                            const simplex_solver_settings_t<i_t, f_t>& settings,
                            const std::vector<f_t>& obj_coeffs,
                            const std::vector<i_t>& fractional,
                            const std::vector<f_t>& current_sol,
                            const std::vector<f_t>& root_solution,
                            i_t target_num_fixed,
                            f_t start_time,
                            lp_problem_t<i_t, f_t>& lp,
                            domain_t<i_t, f_t>& domain)
{
  std::vector<std::tuple<f_t, i_t, f_t>> candidates;
  for (i_t j : fractional) {
    if (std::abs(lp.lower[j] - lp.upper[j]) <= settings.fixed_tol) { continue; }
    if (toc(start_time) >= settings.time_limit || settings.received_halt_signal()) return -1;

    f_t root_change = current_sol[j] - root_solution[j];
    f_t obj_coeff   = obj_coeffs[j];
    f_t fixed_val   = 0;

    if (root_change >= 0.4) {
      fixed_val = std::ceil(current_sol[j]);
    } else if (root_change <= -0.4) {
      fixed_val = std::floor(current_sol[j]);
    } else if (obj_coeff > settings.zero_tol) {
      fixed_val = std::ceil(current_sol[j]);
    } else if (obj_coeff < -settings.zero_tol) {
      fixed_val = std::floor(current_sol[j]);
    } else {
      fixed_val = std::round(current_sol[j]);
    }

    candidates.push_back(std::make_tuple(std::abs(fixed_val - current_sol[j]), j, fixed_val));
  }

  std::sort(candidates.begin(), candidates.end(), [](auto a, auto b) {
    return std::get<0>(a) < std::get<0>(b);
  });

  i_t num_fixed         = 0;
  f_t accumulate_change = 0;

  for (auto [dist, j, fixed_val] : candidates) {
    if (num_fixed >= target_num_fixed) break;
    if (toc(start_time) >= settings.time_limit || settings.received_halt_signal()) return -1;

    fixed_val = std::clamp(fixed_val, lp.lower[j], lp.upper[j]);
    i_t start = domain.size();
    simplex::bound_change_t<i_t, f_t> bound_change{
      .var       = j,
      .new_upper = fixed_val,
      .new_lower = fixed_val,
      .origin    = simplex::bound_change_origin_t::BRANCH};
    if (!domain.apply_and_propagate(Arow, var_types, settings, bound_change, lp)) { return -1; }
    num_fixed += calculate_fixrate(var_types, settings, domain, start);

    // Limit the amount of fixing to the current LP.
    accumulate_change += dist;
    if (accumulate_change >= 0.5) break;
  }

  return num_fixed;
}

template <typename i_t, typename f_t>
bool solve_lp(const branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
              const simplex_solver_settings_t<i_t, f_t>& settings,
              diving_worker_t<i_t, f_t>* worker,
              branch_and_bound_stats_t<i_t, f_t>& stats,
              std::vector<i_t>& fractional)
{
  branch_and_bound_t<i_t, f_t>* bnb_ptr = worker->branch_and_bound_ptr;
  mip_node_t<i_t, f_t>* node            = &worker->start_node;

  // After fixing the variables, re-solve the LP relaxation. We use the optimal solution
  // in the next iteration to find additional variable fixings.
  // We continue to do this until enough variables were fixed or no variable is left to fix.
  simplex::logger_t log;
  log.log = false;

  int64_t iter_offset =
    settings.inside_submip ? 0 : settings.mip_hyper_params.submip_iteration_limit_offset;
  int64_t simplex_iter       = bnb_stats.total_simplex_iters;
  f_t iter_ratio             = settings.mip_hyper_params.submip_iteration_limit_ratio;
  int64_t simplex_iter_limit = iter_offset + simplex_iter * iter_ratio;
  int64_t max_iter           = simplex_iter_limit - stats.total_simplex_iters;
  if (max_iter <= 0) {
    DEBUG_SUBMIP(settings,
                 "{}Max iteration reached! {}/{}",
                 settings.log.log_prefix,
                 stats.total_simplex_iters.load(std::memory_order::relaxed),
                 simplex_iter_limit)
    return false;
  }

  f_t leaf_obj;
  f_t upper_bound = bnb_ptr->get_upper_bound();
  simplex::dual_status_t lp_status =
    worker->solve_node_lp(settings, node, stats, log, max_iter, bnb_stats.start_time, upper_bound);

  node_status_t node_status =
    worker->classify_node(node, lp_status, upper_bound, settings, fractional, leaf_obj);

  if (node_status == node_status_t::FATHOMED) {
    DEBUG_SUBMIP(settings,
                 "{}Round {}: simplex returned {}. leaf_obj={:.4g}, upper_bound={:.4g}",
                 settings.log.log_prefix,
                 round,
                 simplex::dual_status_to_string(lp_status),
                 leaf_obj,
                 upper_bound)
    return false;
  }

  // We found a feasible solution when fixing the variables in RINS/RENS.
  if (node_status == node_status_t::INTEGER_FEASIBLE) {
    bnb_ptr->set_solution_from_worker(
      worker->leaf_problem, leaf_obj, worker->leaf_solution.x, -1, worker->search_strategy);
    DEBUG_SUBMIP(settings,
                 "{}Round {}: found a solution with obj={:.4g}. upper_bound={:.4g}",
                 settings.log.log_prefix,
                 round,
                 leaf_obj,
                 bnb_ptr->get_upper_bound())
    return false;
  }

  worker->recompute_basis = false;
  return true;
}

}  // namespace

template <typename i_t, typename f_t>
void solve_submip(diving_worker_t<i_t, f_t>* worker,
                  branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
                  const user_problem_t<i_t, f_t>& original_problem,
                  submip_stats_t& submip_stats,
                  f_t fixrate,
                  int64_t simplex_iter_used,
                  simplex_solver_settings_t<i_t, f_t> submip_settings)
{
  f_t time_limit = submip_settings.time_limit - toc(bnb_stats.start_time);
  if (time_limit <= 0) { return; }

  double start_time                     = tic();
  branch_and_bound_t<i_t, f_t>* bnb_ptr = worker->branch_and_bound_ptr;
  bool inside_submip                    = submip_settings.inside_submip;

  // If not enough variables was fixed (the neighbourhood is too loose) or the sub-MIP already
  // found a solution that improved the incumbent, then do a DFS with a backtrack_limit of 5
  // levels up to try to find a feasible solution quickly from the neighbourhood.
  if (fixrate < submip_settings.mip_hyper_params.submip_min_fixrate_cap ||
      (inside_submip && submip_stats.total_success != 0)) {
    worker->start_node.packed_vstatus = simplex::compress_vstatus(worker->leaf_vstatus);

    fj_cpu_worker_t<i_t, f_t> submip_fj_cpu_worker;
    if (submip_settings.mip_hyper_params.submip_enable_cpufj) {
      submip_fj_cpu_worker.improvement_callback =
        [bnb_ptr](f_t obj, const std::vector<f_t>& assignment, double work_units) {
          bnb_ptr->set_solution_from_cpu_fj(obj, assignment, work_units);
        };

      f_t work_limit = 1.0;
      submip_fj_cpu_worker.create_worker(worker->leaf_problem,
                                         worker->var_types,
                                         original_problem.num_cols,
                                         worker->leaf_solution.x,
                                         submip_settings,
                                         std::format("{} [CPU FJ]", submip_settings.log.log_prefix),
                                         worker->rng.next_i64());
      submip_fj_cpu_worker.run_sync(time_limit, work_limit);
    }

    DEBUG_SUBMIP(submip_settings,
                 "{}Running a quick DFS. fixrate={:.4g}",
                 submip_settings.log.log_prefix,
                 fixrate);

    simplex_solver_settings_t<i_t, f_t> dfs_settings = submip_settings;
    dfs_settings.mip_hyper_params.diving_backtrack_limit =
      dfs_settings.mip_hyper_params.submip_dfs_max_backtrack;
    worker->domain.clear();
    worker->dive(dfs_settings, bnb_stats);
    return;
  }

  bool is_root_heuristic = submip_settings.inside_root_node;
  i_t submip_level       = submip_settings.mip_hyper_params.submip_level + 1;
  f_t upper_bound        = bnb_ptr->get_upper_bound();
  f_t user_lower         = compute_user_objective(worker->leaf_problem, bnb_ptr->get_lower_bound());
  f_t user_obj           = compute_user_objective(worker->leaf_problem, upper_bound);
  f_t rel_gap            = user_relative_gap(user_obj, user_lower);
  i_t explored           = bnb_stats.nodes_explored;

  bool max_recursion = submip_level > submip_settings.mip_hyper_params.submip_max_level;
  submip_settings.mip_hyper_params.submip_rins =
    submip_settings.mip_hyper_params.submip_rins != 0 && !max_recursion;
  submip_settings.mip_hyper_params.submip_rens =
    submip_settings.mip_hyper_params.submip_rens != 0 && !max_recursion;

  submip_settings.node_limit =
    submip_settings.mip_hyper_params.submip_node_limit_offset + explored / 20;

  // Add offset only on the top call, we want number of simplex iteration to decay
  // as we go down the recursion to avoid spending too much time in the deeper levels.
  int64_t iter_offset =
    inside_submip ? 0 : submip_settings.mip_hyper_params.submip_iteration_limit_offset;
  int64_t simplex_iter = bnb_stats.total_simplex_iters;
  f_t iter_ratio       = submip_settings.mip_hyper_params.submip_iteration_limit_ratio;

  submip_settings.mip_hyper_params.bnb_simplex_iteration_limit =
    iter_offset + simplex_iter * iter_ratio - simplex_iter_used;
  if (submip_settings.mip_hyper_params.bnb_simplex_iteration_limit <= 0) { return; }

  submip_settings.time_limit = time_limit;

  submip_settings.relative_mip_gap_tol =
    std::min(submip_settings.mip_hyper_params.submip_target_mip_gap, rel_gap);

  submip_settings.print_presolve_stats                     = false;
  submip_settings.num_threads                              = 1;
  submip_settings.reliability_branching                    = 0;
  submip_settings.clique_cuts                              = 0;
  submip_settings.zero_half_cuts                           = 0;
  submip_settings.inside_submip                            = 1;
  submip_settings.strong_branching_simplex_iteration_limit = 50;
  submip_settings.inside_root_node                         = 0;
  submip_settings.mip_hyper_params.submip_level            = submip_level;
  submip_settings.benchmark_info_ptr                       = nullptr;
  submip_settings.log.log                                  = SUBMIP_VERBOSE;
  std::string_view log_prefix                              = submip_settings.log.log_prefix;

  DEBUG_SUBMIP(submip_settings, "{}Sub-MIP: fixrate={:.2f}", log_prefix, fixrate)
  DEBUG_SUBMIP(
    submip_settings,
    "{}Sub-MIP solve settings: time_limit={:.2f}, node_limit={}, iter_limit={} (current_iter={}), "
    "tol={:g}",
    log_prefix,
    submip_settings.time_limit,
    submip_settings.node_limit,
    submip_settings.mip_hyper_params.bnb_simplex_iteration_limit,
    bnb_stats.total_simplex_iters.load(std::memory_order::relaxed),
    submip_settings.relative_mip_gap_tol);

  // The `worker->leaf_problem` is directly converted to an `user_problem_t`, meaning that
  // there is only equality rows (the range row vector is empty) and it contains
  // structural + slacks + cuts constraints/variables.
  user_problem_t<i_t, f_t> submip_problem(original_problem.handle_ptr);
  simplex::convert_lp_to_user_problem(
    worker->leaf_problem, worker->var_types, submip_settings, submip_problem);

  third_party_presolve_t<i_t, f_t> presolver;
  f_t presolve_time_limit = std::min(0.1 * submip_settings.time_limit, 60.0);
  third_party_presolve_status_t presolver_status =
    presolver.apply_to_subproblem(submip_problem, submip_settings, presolve_time_limit, 1);

  if (presolver_status == third_party_presolve_status_t::INFEASIBLE ||
      presolver_status == third_party_presolve_status_t::UNBNDORINFEAS ||
      presolver_status == third_party_presolve_status_t::UNBOUNDED) {
    DEBUG_SUBMIP(submip_settings, "{}Presolve detected infeasibility", log_prefix);
    submip_stats.save_infeasible(fixrate);
    return;
  }

  // Also handle optimal
  if (submip_problem.num_rows == 0 || submip_problem.num_cols == 0) {
    DEBUG_SUBMIP(submip_settings,
                 "{}Reduced to a trivial {} x {} problem; solving by bound pushing",
                 log_prefix,
                 submip_problem.num_rows,
                 submip_problem.num_cols);
    submip_stats.save_empty();
    return;
  }

  double presolve_time = toc(start_time);

  if (submip_settings.received_halt_signal()) { return; }

  submip_settings.heuristic_preemption_callback   = nullptr;
  submip_settings.dual_simplex_objective_callback = nullptr;
  submip_settings.set_simplex_solution_callback   = nullptr;
  submip_settings.solution_callback =
    [bnb_ptr, &presolver, fixrate, &submip_stats, log_prefix, worker](
      const std::vector<f_t>& solution, f_t obj) {
      bnb_ptr->set_solution_from_submip(
        worker->leaf_problem, solution, presolver, submip_stats, fixrate, log_prefix);
    };

  DEBUG_SUBMIP(submip_settings,
               "{}Sub-MIP: {} constraints, {} variables, {} nonzeros\n",
               log_prefix,
               submip_problem.num_rows,
               submip_problem.num_cols,
               submip_problem.A.nnz());

  probing_implied_bound_t<i_t, f_t> empty_probing(submip_problem.num_cols);
  submip_settings.time_limit -= presolve_time;
  if (submip_settings.time_limit <= 0.0) { return; }

  branch_and_bound_t submip_bnb(submip_problem, submip_settings, tic(), empty_probing);
  simplex::mip_solution_t<i_t, f_t> submip_solution(submip_problem.num_cols);

  std::vector<f_t> presolved_incumbent;

  // We do not have an incumbent yet, so skip the initial guess.
  if (!worker->current_incumbent.empty()) {
    // Crush the incumbent to presolve space. It may not be valid for the sub-MIP since we
    // may fix integer variables that does not match the current incumbent to reach the target
    // fix rate.
    presolver.crush_primal_solution(submip_problem, worker->current_incumbent, presolved_incumbent);
    submip_bnb.set_initial_guess(presolved_incumbent);
  }

  // Even if we do not have a valid incumbent now, the upper bound can still be set by the early
  // heuristics.
  upper_bound = bnb_ptr->get_upper_bound();
  if (std::isfinite(upper_bound)) {
    const f_t user_upper    = compute_user_objective(worker->leaf_problem, upper_bound);
    const f_t submip_cutoff = compute_presolved_objective(submip_bnb.original_lp_, user_upper);
    submip_bnb.set_initial_upper_bound(submip_cutoff);
  }

  if (!is_root_heuristic)
    submip_bnb.set_initial_pseudocost(bnb_ptr->pc_, presolver.get_reduced_to_original_map());

  if (bnb_ptr->submip_halt_callback_) {
    // Copy the halt callback to the deeper level.
    submip_bnb.set_submip_halt_callback(bnb_ptr->submip_halt_callback_);
  } else {
    // This should only be called by the main solver.
    submip_bnb.set_submip_halt_callback([bnb_ptr](f_t, f_t submip_lower_bound) {
      f_t user_upper = compute_user_objective(bnb_ptr->original_lp_, bnb_ptr->get_upper_bound());
      bool is_cutoff = bnb_ptr->original_lp_.obj_scale > 0 ? submip_lower_bound > user_upper
                                                           : user_upper > submip_lower_bound;
      bool is_solver_running = !bnb_ptr->has_solver_stopped();
      return is_cutoff || !is_solver_running || bnb_ptr->received_halt_signal();
    });
  }

  fj_cpu_worker_t<i_t, f_t> submip_fj_cpu_worker;

  if (submip_settings.mip_hyper_params.submip_enable_cpufj) {
    // Since we do not have an incumbent, use the LP solution of the last round of variable fixing
    // in RENS.
    if (worker->search_strategy == search_strategy_t::RENS) {
      presolver.crush_primal_solution(submip_problem, worker->leaf_solution.x, presolved_incumbent);
    }

    // Launch a CPU FJ worker on the presolved sub-MIP with a fixed budget (in terms of work units)
    // to run in parallel with the cut-and-branch algorithm with the goal of finding a quick
    // feasible solution for the sub-MIP problem. The CPU FJ uses the current incumbent (crushed
    // into the presolved space) as initial guess. The worker is automatically stop when we go out
    // of the scope.
    std::vector<f_t> initial_guess;
    crush_primal_solution(submip_problem,
                          submip_bnb.original_lp_,
                          presolved_incumbent,
                          submip_bnb.new_slacks_,
                          initial_guess);

    submip_fj_cpu_worker.improvement_callback =
      [&submip_bnb](f_t obj, const std::vector<f_t>& solution, double work_units) {
        submip_bnb.set_solution_from_cpu_fj(obj, solution, work_units);
      };

    f_t work_limit = 1.0;
    submip_fj_cpu_worker.create_worker(submip_bnb.original_lp_,
                                       submip_bnb.var_types_,
                                       submip_bnb.original_problem_.num_cols,
                                       initial_guess,
                                       submip_bnb.settings_,
                                       std::format("{} [CPU FJ]", log_prefix),
                                       worker->rng.next_i64());
    submip_fj_cpu_worker.run_async(submip_settings.time_limit, work_limit);
  }

  bnb_ptr->active_submip_solvers_mutex_.lock();
  bnb_ptr->active_submip_solvers_[worker->worker_id] = &submip_bnb;
  const bool halted = bnb_ptr->node_concurrent_halt_.load(std::memory_order::acquire);
  bnb_ptr->active_submip_solvers_mutex_.unlock();

  scope_guard solver_scope([&]() {
    bnb_ptr->active_submip_solvers_mutex_.lock();
    bnb_ptr->active_submip_solvers_[worker->worker_id] = nullptr;
    bnb_ptr->active_submip_solvers_mutex_.unlock();
  });

  if (halted) { submip_bnb.halt_solver(); }

  mip_status_t submip_status = submip_bnb.solve(submip_solution);
  f_t submip_time            = toc(start_time);

  DEBUG_SUBMIP(
    submip_settings,
    "{}Sub-MIP: status={}, iterations={} (total={}), presolve_time={:.2f}, total_time={:.2f} \n",
    log_prefix,
    mip_status_to_string(submip_status),
    submip_solution.simplex_iterations,
    bnb_stats.total_simplex_iters.load(std::memory_order::relaxed),
    presolve_time,
    submip_time);

  if (submip_status == mip_status_t::NUMERICAL) { return; }
  if (submip_status == mip_status_t::INFEASIBLE || submip_status == mip_status_t::UNBOUNDED) {
    submip_stats.save_infeasible(fixrate);
    return;
  }

  if (submip_solution.has_incumbent) {
    bnb_ptr->set_solution_from_submip(
      worker->leaf_problem, submip_solution.x, presolver, submip_stats, fixrate, log_prefix);
  }

  // Accumulate simplex iterations to determine when to stop exploring the sub-MIP
  if (inside_submip) { bnb_stats.total_simplex_iters += submip_solution.simplex_iterations; }
}

template <typename i_t, typename f_t>
void mutation(diving_worker_t<i_t, f_t>* worker,
              branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
              const user_problem_t<i_t, f_t>& original_problem,
              submip_stats_t& submip_stats,
              simplex_solver_settings_t<i_t, f_t> settings)
{
  raft::common::nvtx::range scope("BB::mutation_thread");
  if (worker->orbital_fixing) { worker->orbital_fixing->disable(); }

  i_t submip_level = settings.mip_hyper_params.submip_level + 1;
  settings.log.log_prefix =
    std::format("[{} {}] ", search_strategy_to_string(worker->search_strategy), submip_level);

  assert((worker->search_strategy == search_strategy_t::MUTATION) &&
         "Sub-MIP worker must be set to MUTATION type");

  ++submip_stats.total_calls;

  lp_problem_t<i_t, f_t>& lp  = worker->leaf_problem;
  std::vector<f_t>& incumbent = worker->current_incumbent;
  domain_t<i_t, f_t>& domain  = worker->domain;

  std::vector<i_t> integer_list;
  get_unfixed_integer_variables(
    lp.lower, lp.upper, worker->var_types, settings.fixed_tol, integer_list);
  worker->rng.shuffle(integer_list);
  worker->domain.compute_activities(worker->Arow, worker->leaf_problem);

  f_t target_fixrate = submip_get_max_fixrate(submip_stats, settings.mip_hyper_params, worker->rng);
  i_t target_num_fixed = target_fixrate * integer_list.size();
  i_t num_fixed        = 0;

  for (i_t j : integer_list) {
    if (toc(bnb_stats.start_time) >= settings.time_limit) {
      DEBUG_SUBMIP(settings, "{}Time limit reached.", settings.log.log_prefix)
      return;
    }

    if (settings.received_halt_signal()) {
      DEBUG_SUBMIP(settings, "{}Halt signal received.", settings.log.log_prefix)
      return;
    }

    if (num_fixed >= target_num_fixed) break;
    if (std::abs(lp.lower[j] - lp.upper[j]) <= settings.fixed_tol) continue;

    f_t fixed_val = std::clamp(std::round(incumbent[j]), lp.lower[j], lp.upper[j]);
    i_t start     = domain.size();
    simplex::bound_change_t<i_t, f_t> bound_change{
      .var       = j,
      .new_upper = fixed_val,
      .new_lower = fixed_val,
      .origin    = simplex::bound_change_origin_t::BRANCH};

    if (!domain.apply_and_propagate(worker->Arow, worker->var_types, settings, bound_change, lp)) {
      DEBUG_SUBMIP(settings,
                   "{}Bound propagation detected infeasibility. fixed={}/{}, target={} ({:.4g})",
                   settings.log.log_prefix,
                   num_fixed,
                   integer_list.size(),
                   target_num_fixed,
                   target_fixrate)
      return;
    }

    num_fixed += calculate_fixrate(worker->var_types, settings, domain, start);
  }

  f_t fixrate = static_cast<f_t>(num_fixed) / integer_list.size();

  DEBUG_SUBMIP(settings,
               "{} fixed variables = {:.0f} ({:.2f}), target fixrate = {} ({:.2f})",
               settings.log.log_prefix,
               fixrate * integer_list.size(),
               fixrate,
               target_num_fixed,
               target_fixrate);

  solve_submip(worker, bnb_stats, original_problem, submip_stats, fixrate, 0, settings);

  DEBUG_SUBMIP(
    settings,
    "{}success={}, infeasible={}, calls={}, fixrate={:.4g} ({:.0f}/{}), max_fixrate={:.4g}\n",
    settings.log.log_prefix,
    submip_stats.total_success.load(std::memory_order::relaxed),
    submip_stats.total_infeasible.load(std::memory_order::relaxed),
    submip_stats.total_calls.load(std::memory_order::relaxed),
    fixrate,
    fixrate * integer_list.size(),
    integer_list.size(),
    target_fixrate);
}

template <typename i_t, typename f_t>
void rins(diving_worker_t<i_t, f_t>* worker,
          branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
          const user_problem_t<i_t, f_t>& original_problem,
          submip_stats_t& submip_stats,
          simplex_solver_settings_t<i_t, f_t> settings)
{
  raft::common::nvtx::range scope("BB::submip_thread");
  if (worker->orbital_fixing) { worker->orbital_fixing->disable(); }

  i_t submip_level = settings.mip_hyper_params.submip_level + 1;
  settings.log.log_prefix =
    std::format("[{} {}] ", search_strategy_to_string(worker->search_strategy), submip_level);
  branch_and_bound_t<i_t, f_t>* bnb_ptr = worker->branch_and_bound_ptr;

  assert(worker->search_strategy == search_strategy_t::RINS &&
         "Sub-MIP worker must be set to RINS type");

  ++submip_stats.total_calls;

  bool has_submip         = false;
  worker->recompute_basis = true;

  branch_and_bound_stats_t<i_t, f_t> stats;
  lp_problem_t<i_t, f_t>& lp    = worker->leaf_problem;
  std::vector<f_t>& incumbent   = worker->current_incumbent;
  domain_t<i_t, f_t>& domain    = worker->domain;
  std::vector<f_t>& current_sol = worker->leaf_solution.x;
  worker->domain.compute_activities(worker->Arow, worker->leaf_problem);

  std::vector<i_t> fractional;
  fractional_variables(settings, current_sol, worker->var_types, fractional);

  std::vector<i_t> integer_list;
  get_unfixed_integer_variables(
    lp.lower, lp.upper, worker->var_types, settings.fixed_tol, integer_list);

  i_t num_integers = integer_list.size();
  f_t max_fixrate  = submip_get_max_fixrate(submip_stats, settings.mip_hyper_params, worker->rng);
  f_t min_fixrate  = settings.mip_hyper_params.submip_min_fixrate;
  f_t fixrate      = 0;
  i_t num_fixed    = 0;
  bool has_stopped = false;

  i_t round = 0;

  while (!bnb_ptr->has_solver_stopped() && !settings.received_halt_signal()) {
    f_t prev_stack_size = worker->domain.size();
    f_t prev_fixrate    = fixrate;
    f_t target_fixrate  = max_fixrate;
    i_t new_fixings     = 0;
    i_t round_num_fixed = (target_fixrate - prev_fixrate) * num_integers;

    if (round_num_fixed == 0) {
      has_submip = fixrate > 0;
      break;
    }

    // Shuffle the fractional and integer list, so every variable has the same chance to the picked
    // (we iterate the list in order).
    worker->rng.shuffle(integer_list);
    worker->rng.shuffle(fractional);

    // RINS neighbourhood: Fix all the integer variables where the current solution matches the
    // incumbent. We are using the `max_fixrate` here to allow RINS to fix all integer variables
    // that it can within our budget.
    for (i_t j : integer_list) {
      if (toc(bnb_stats.start_time) >= settings.time_limit) {
        DEBUG_SUBMIP(settings, "{}Time limit reached.", settings.log.log_prefix)
        has_stopped = true;
        break;
      }

      if (settings.received_halt_signal()) {
        DEBUG_SUBMIP(settings, "{}Halt signal received.", settings.log.log_prefix)
        has_stopped = true;
        break;
      }

      if (new_fixings >= round_num_fixed) break;
      if (std::abs(lp.lower[j] - lp.upper[j]) <= settings.fixed_tol) continue;
      if (std::abs(current_sol[j] - incumbent[j]) <= settings.integer_tol) {
        f_t fixed_val = std::clamp(std::round(incumbent[j]), lp.lower[j], lp.upper[j]);
        i_t start     = domain.size();
        simplex::bound_change_t<i_t, f_t> bound_change{
          .var       = j,
          .new_upper = fixed_val,
          .new_lower = fixed_val,
          .origin    = simplex::bound_change_origin_t::BRANCH};
        if (!domain.apply_and_propagate(
              worker->Arow, worker->var_types, settings, bound_change, lp)) {
          DEBUG_SUBMIP(settings,
                       "{}Bound propagation detected infeasibility. fixed={}/{}, "
                       "max_fixrate={:.4g}, min_fixrate={:.4g}",
                       settings.log.log_prefix,
                       num_fixed + new_fixings,
                       num_integers,
                       max_fixrate,
                       min_fixrate)
          has_stopped = true;
          break;
        }

        new_fixings += calculate_fixrate(worker->var_types, settings, domain, start);
      }
    }

    if (has_stopped) { break; }

    num_fixed += new_fixings;
    fixrate = static_cast<f_t>(num_fixed) / integer_list.size();

    // The RINS neighbourhood ran dry. If it is already tight enough, take it rather than
    // diluting it with fixings that do not agree with the incumbent.
    if (prev_fixrate == fixrate && fixrate >= min_fixrate) {
      has_submip = true;
      break;
    }

    // Even considering the entire integer list, we were unable to fix a single variable in this
    // iteration. Iterate over the fractional variables again and fixing those that closest to
    // an integer solution first in order to reach the fixing threshold.
    if (fixrate == prev_fixrate && prev_stack_size == worker->domain.size()) {
      new_fixings = extend_variable_fixings(worker->Arow,
                                            worker->var_types,
                                            settings,
                                            worker->leaf_problem.objective,
                                            fractional,
                                            current_sol,
                                            worker->root_solution,
                                            round_num_fixed,
                                            bnb_stats.start_time,
                                            worker->leaf_problem,
                                            worker->domain);
      if (new_fixings < 0) {
        DEBUG_SUBMIP(settings,
                     "{}Round {}: stopped fixing (infeasible, halted or time limit reached).",
                     settings.log.log_prefix,
                     round)
        break;
      }

      num_fixed += new_fixings;
      fixrate = static_cast<f_t>(num_fixed) / integer_list.size();

      if (fixrate >= min_fixrate) {
        has_submip = true;
        break;
      }
    }

    DEBUG_SUBMIP(
      settings,
      "{}Round {}: fixed {:.0f} ({:.2f}) -> {:.0f} ({:.2f}) variables ({:+}). round target = {} "
      "({:.2f}). "
      "max fixrate = {:.4g}",
      settings.log.log_prefix,
      round,
      prev_fixrate * num_integers,
      prev_fixrate,
      fixrate * num_integers,
      fixrate,
      new_fixings,
      round_num_fixed,
      target_fixrate,
      max_fixrate);

    if (fixrate >= max_fixrate) {
      has_submip = true;
      break;
    }

    if (toc(bnb_stats.start_time) >= settings.time_limit) { break; }

    bool continue_dive = solve_lp(bnb_stats, settings, worker, stats, fractional);
    if (!continue_dive) { break; }
    ++round;
  }

  // Accumulate the iterations for sub-MIP so it stops when it reaches the allocated budget.
  if (settings.inside_submip) { bnb_stats.total_simplex_iters += stats.total_simplex_iters; }

  if (has_submip) {
    solve_submip(worker,
                 bnb_stats,
                 original_problem,
                 submip_stats,
                 fixrate,
                 stats.total_simplex_iters,
                 settings);
  }

  DEBUG_SUBMIP(
    settings,
    "{}success={}, infeasible={}, calls={}, fixrate={:.4g} ({:.0f}/{}), max_fixrate={:.4g}, "
    "min_fixrate={:.4g}\n",
    settings.log.log_prefix,
    submip_stats.total_success.load(std::memory_order::relaxed),
    submip_stats.total_infeasible.load(std::memory_order::relaxed),
    submip_stats.total_calls.load(std::memory_order::relaxed),
    fixrate,
    fixrate * num_integers,
    num_integers,
    max_fixrate,
    min_fixrate);
}

template <typename i_t, typename f_t>
void rens(diving_worker_t<i_t, f_t>* worker,
          branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
          const user_problem_t<i_t, f_t>& original_problem,
          submip_stats_t& submip_stats,
          simplex_solver_settings_t<i_t, f_t> settings)
{
  raft::common::nvtx::range scope("BB::submip_thread");
  if (worker->orbital_fixing) { worker->orbital_fixing->disable(); }

  i_t submip_level = settings.mip_hyper_params.submip_level + 1;
  settings.log.log_prefix =
    std::format("[{} {}] ", search_strategy_to_string(worker->search_strategy), submip_level);
  branch_and_bound_t<i_t, f_t>* bnb_ptr = worker->branch_and_bound_ptr;

  assert(worker->search_strategy == search_strategy_t::RENS &&
         "Sub-MIP worker must be set to RENS type");

  ++submip_stats.total_calls;

  bool has_submip         = false;
  worker->recompute_basis = true;

  branch_and_bound_stats_t<i_t, f_t> stats;
  lp_problem_t<i_t, f_t>& lp    = worker->leaf_problem;
  domain_t<i_t, f_t>& domain    = worker->domain;
  std::vector<f_t>& current_sol = worker->leaf_solution.x;
  worker->domain.compute_activities(worker->Arow, worker->leaf_problem);

  std::vector<i_t> fractional;
  fractional_variables(settings, current_sol, worker->var_types, fractional);

  std::vector<i_t> integer_list;
  get_unfixed_integer_variables(
    lp.lower, lp.upper, worker->var_types, settings.fixed_tol, integer_list);

  i_t num_integers = integer_list.size();
  f_t max_fixrate  = submip_get_max_fixrate(submip_stats, settings.mip_hyper_params, worker->rng);
  f_t min_fixrate  = settings.mip_hyper_params.submip_min_fixrate;
  f_t fixrate      = 0;
  i_t num_fixed    = 0;
  f_t close_ratio  = settings.mip_hyper_params.submip_round_close_ratio;
  bool has_stopped = false;

  i_t round = 0;

  while (!bnb_ptr->has_solver_stopped() && !settings.received_halt_signal()) {
    f_t prev_stack_size = worker->domain.size();
    f_t prev_fixrate    = fixrate;
    f_t target_fixrate  = std::min(1.0 - (1.0 - fixrate) * close_ratio, max_fixrate);
    i_t new_fixings     = 0;

    i_t round_num_fixed = (target_fixrate - prev_fixrate) * num_integers;
    if (round_num_fixed == 0) {
      target_fixrate  = max_fixrate;
      round_num_fixed = (target_fixrate - prev_fixrate) * num_integers;
      if (round_num_fixed == 0) {
        has_submip = fixrate > 0;
        break;
      }
    }

    // Shuffle the fractional and integer list, so every variable has the same chance to the picked
    // (we iterate the list in order).
    worker->rng.shuffle(integer_list);
    worker->rng.shuffle(fractional);

    for (i_t j : integer_list) {
      if (toc(bnb_stats.start_time) >= settings.time_limit) {
        DEBUG_SUBMIP(settings, "{}Time limit reached.", settings.log.log_prefix)
        has_stopped = true;
        break;
      }

      if (settings.received_halt_signal()) {
        DEBUG_SUBMIP(settings, "{}Halt signal received.", settings.log.log_prefix)
        has_stopped = true;
        break;
      }

      if (new_fixings >= round_num_fixed) break;
      if (std::abs(lp.lower[j] - lp.upper[j]) <= settings.fixed_tol) continue;

      simplex::bound_change_t<i_t, f_t> bound_change{
        .var = j, .origin = simplex::bound_change_origin_t::BRANCH};
      bound_change.new_lower = std::clamp(std::floor(current_sol[j]), lp.lower[j], lp.upper[j]);
      bound_change.new_upper = std::clamp(std::ceil(current_sol[j]), lp.lower[j], lp.upper[j]);

      if (bound_change.new_lower != lp.lower[j] || bound_change.new_upper != lp.upper[j]) {
        i_t start = domain.size();
        if (!domain.apply_and_propagate(
              worker->Arow, worker->var_types, settings, bound_change, lp)) {
          DEBUG_SUBMIP(settings,
                       "{}Bound propagation detected infeasibility. fixed={}/{}, "
                       "max_fixrate={:.4g}, min_fixrate={:.4g}",
                       settings.log.log_prefix,
                       num_fixed + new_fixings,
                       num_integers,
                       max_fixrate,
                       min_fixrate)
          has_stopped = true;
          break;
        }

        new_fixings += calculate_fixrate(worker->var_types, settings, domain, start);
      }
    }

    if (has_stopped) { break; }

    num_fixed += new_fixings;
    fixrate = static_cast<f_t>(num_fixed) / integer_list.size();

    // Even considering the entire integer list, we were unable to fix a single variable in this
    // iteration. Iterate over the fractional variables again and fixing those that closest to
    // an integer solution first in order to reach the fixing threshold.
    if (fixrate == prev_fixrate && prev_stack_size == worker->domain.size()) {
      new_fixings = extend_variable_fixings(worker->Arow,
                                            worker->var_types,
                                            settings,
                                            worker->leaf_problem.objective,
                                            fractional,
                                            current_sol,
                                            worker->root_solution,
                                            round_num_fixed,
                                            bnb_stats.start_time,
                                            worker->leaf_problem,
                                            worker->domain);
      if (new_fixings < 0) {
        DEBUG_SUBMIP(settings,
                     "{}Round {}: stopped fixing (infeasible, halted or time limit reached).",
                     settings.log.log_prefix,
                     round)
        break;
      }

      num_fixed += new_fixings;
      fixrate = static_cast<f_t>(num_fixed) / integer_list.size();

      if (fixrate >= min_fixrate) {
        has_submip = true;
        break;
      }
    }

    DEBUG_SUBMIP(
      settings,
      "{}Round {}: fixed {:.0f} ({:.2f}) -> {:.0f} ({:.2f}) variables ({:+}). round target = {} "
      "({:.2f}). "
      "max fixrate = {:.4g}",
      settings.log.log_prefix,
      round,
      prev_fixrate * num_integers,
      prev_fixrate,
      fixrate * num_integers,
      fixrate,
      new_fixings,
      round_num_fixed,
      target_fixrate,
      max_fixrate);

    if (fixrate >= max_fixrate) {
      has_submip = true;
      break;
    }

    if (toc(bnb_stats.start_time) >= settings.time_limit) { break; }

    bool continue_dive = solve_lp(bnb_stats, settings, worker, stats, fractional);
    if (!continue_dive) { break; }
    ++round;
  }

  // Accumulate the iterations for sub-MIP so it stops when it reaches the allocated budget.
  if (settings.inside_submip) { bnb_stats.total_simplex_iters += stats.total_simplex_iters; }

  if (has_submip) {
    solve_submip(worker,
                 bnb_stats,
                 original_problem,
                 submip_stats,
                 fixrate,
                 stats.total_simplex_iters,
                 settings);
  }

  DEBUG_SUBMIP(
    settings,
    "{}success={}, infeasible={}, calls={}, fixrate={:.4g} ({:.0f}/{}), max_fixrate={:.4g}, "
    "min_fixrate={:.4g}\n",
    settings.log.log_prefix,
    submip_stats.total_success.load(std::memory_order::relaxed),
    submip_stats.total_infeasible.load(std::memory_order::relaxed),
    submip_stats.total_calls.load(std::memory_order::relaxed),
    fixrate,
    fixrate * num_integers,
    num_integers,
    max_fixrate,
    min_fixrate);
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template void solve_submip(diving_worker_t<int, double>* worker,
                           branch_and_bound_stats_t<int, double>& bnb_stats,
                           const user_problem_t<int, double>& original_problem,
                           submip_stats_t& submip_stats,
                           double fixrate,
                           int64_t simplex_iter_used,
                           simplex_solver_settings_t<int, double> submip_settings);

template void mutation(diving_worker_t<int, double>* worker,
                       branch_and_bound_stats_t<int, double>& bnb_stats,
                       const user_problem_t<int, double>& original_problem,
                       submip_stats_t& submip_stats,
                       simplex_solver_settings_t<int, double> submip_settings);

template void rins(diving_worker_t<int, double>* worker,
                   branch_and_bound_stats_t<int, double>& bnb_stats,
                   const user_problem_t<int, double>& original_problem,
                   submip_stats_t& submip_stats,
                   simplex_solver_settings_t<int, double> settings);

template void rens(diving_worker_t<int, double>* worker,
                   branch_and_bound_stats_t<int, double>& bnb_stats,
                   const user_problem_t<int, double>& original_problem,
                   submip_stats_t& submip_stats,
                   simplex_solver_settings_t<int, double> settings);

#endif

}  // namespace cuopt::mathematical_optimization::mip
