/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/solve_remote.hpp>

#include <mip_heuristics/feasibility_jump/early_cpufj.cuh>
#include <mip_heuristics/feasibility_jump/early_gpufj.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/mip_scaling_strategy.cuh>
#include <mip_heuristics/presolve/third_party_presolve.hpp>
#include <mip_heuristics/presolve/trivial_presolve.cuh>
#include <mip_heuristics/solver.cuh>
#include <mip_heuristics/utilities/sort_csr.cuh>
#include <mip_heuristics/utils.cuh>

#include <pdlp/pdlp.cuh>
#include <pdlp/restart_strategy/pdlp_restart_strategy.cuh>
#include <pdlp/step_size_strategy/adaptive_step_size_strategy.hpp>
#include <pdlp/utilities/problem_checking.cuh>
#include <pdlp/utils.cuh>
#include <utilities/logger.hpp>
#include <utilities/seed_generator.cuh>
#include <utilities/version_info.hpp>

#include <cuopt/linear_programming/backend_selection.hpp>
#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_solution.hpp>
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>

#include <mps_parser/mps_data_model.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_profiler_api.h>

namespace cuopt::linear_programming {

// This serves as both a warm up but also a mandatory initial call to setup cuSparse and cuBLAS
static void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

template <typename f_t>
static void invoke_solution_callbacks(
  const std::vector<internals::base_solution_callback_t*>& mip_callbacks,
  f_t objective,
  std::vector<f_t>& assignment,
  f_t bound)
{
  std::vector<f_t> obj_vec   = {objective};
  std::vector<f_t> bound_vec = {bound};
  for (auto callback : mip_callbacks) {
    if (callback != nullptr &&
        callback->get_type() == internals::base_solution_callback_type::GET_SOLUTION) {
      auto get_sol_callback = static_cast<internals::get_solution_callback_t*>(callback);
      get_sol_callback->get_solution(
        assignment.data(), obj_vec.data(), bound_vec.data(), get_sol_callback->get_user_data());
    }
  }
}

template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> run_mip(detail::problem_t<i_t, f_t>& problem,
                                 mip_solver_settings_t<i_t, f_t> const& settings,
                                 timer_t& timer,
                                 f_t initial_cutoff = std::numeric_limits<f_t>::infinity())
{
  try {
    raft::common::nvtx::range fun_scope("run_mip");
    if (settings.get_mip_callbacks().size() > 0) {
      auto callback_num_variables = problem.original_problem_ptr->get_n_variables();
      if (problem.has_papilo_presolve_data()) {
        callback_num_variables = problem.get_papilo_original_num_variables();
      }
      for (auto callback : settings.get_mip_callbacks()) {
        callback->template setup<f_t>(callback_num_variables);
      }
    }
    // if the input problem is empty: early exit
    if (problem.empty) {
      detail::solution_t<i_t, f_t> solution(problem);
      problem.preprocess_problem();
      thrust::for_each(
        problem.handle_ptr->get_thrust_policy(),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(problem.n_variables),
        [sol = solution.assignment.data(), pb = problem.view()] __device__(i_t index) {
          auto bounds = pb.variable_bounds[index];
          sol[index] = pb.objective_coefficients[index] > 0 ? get_lower(bounds) : get_upper(bounds);
        });
      problem.post_process_solution(solution);
      solution.compute_objective();  // just to ensure h_user_obj is set
      auto stats = solver_stats_t<i_t, f_t>{};
      stats.set_solution_bound(solution.get_user_objective());
      // log the objective for scripts which need it
      CUOPT_LOG_INFO("Best feasible: %f", solution.get_user_objective());
      for (auto callback : settings.get_mip_callbacks()) {
        if (callback->get_type() == internals::base_solution_callback_type::GET_SOLUTION) {
          auto temp_sol(solution);
          auto get_sol_callback = static_cast<internals::get_solution_callback_t*>(callback);
          std::vector<f_t> user_objective_vec(1);
          std::vector<f_t> user_bound_vec(1);
          user_objective_vec[0] = solution.get_user_objective();
          user_bound_vec[0]     = stats.get_solution_bound();
          if (problem.has_papilo_presolve_data()) {
            problem.papilo_uncrush_assignment(temp_sol.assignment);
          }
          std::vector<f_t> user_assignment_vec(temp_sol.assignment.size());
          raft::copy(user_assignment_vec.data(),
                     temp_sol.assignment.data(),
                     temp_sol.assignment.size(),
                     temp_sol.handle_ptr->get_stream());
          solution.handle_ptr->sync_stream();
          get_sol_callback->get_solution(user_assignment_vec.data(),
                                         user_objective_vec.data(),
                                         user_bound_vec.data(),
                                         get_sol_callback->get_user_data());
        }
      }
      return solution.get_solution(true, stats, false);
    }
    // problem contains unpreprocessed data
    detail::problem_t<i_t, f_t> scaled_problem(problem);
    cuopt_func_call(auto saved_problem = scaled_problem);
    CUOPT_LOG_INFO("Objective offset %f scaling_factor %f",
                   problem.presolve_data.objective_offset,
                   problem.presolve_data.objective_scaling_factor);
    CUOPT_LOG_INFO("Model fingerprint: 0x%x", problem.get_fingerprint());
    cuopt_assert(problem.original_problem_ptr->get_n_variables() == scaled_problem.n_variables,
                 "Size mismatch");
    cuopt_assert(problem.original_problem_ptr->get_n_constraints() == scaled_problem.n_constraints,
                 "Size mismatch");
    // only call preprocess on scaled problem, so we can compute feasibility on the original problem
    scaled_problem.preprocess_problem();
    scaled_problem.related_vars_time_limit = settings.heuristic_params.related_vars_time_limit;
    detail::trivial_presolve(scaled_problem);

    detail::mip_solver_t<i_t, f_t> solver(scaled_problem, settings, timer);
    // initial_cutoff is in user-space (representation-invariant).
    // It will be converted to the target solver-space at each consumption point.
    solver.context.initial_cutoff = initial_cutoff;
    if (timer.check_time_limit()) {
      CUOPT_LOG_INFO("Time limit reached before main solve");
      detail::solution_t<i_t, f_t> sol(problem);
      auto stats                 = solver.get_solver_stats();
      stats.total_solve_time     = timer.elapsed_time();
      sol.post_process_completed = true;
      return sol.get_solution(false, stats, false);
    }

    // Run early CPUFJ on papilo-presolved problem during cuOpt presolve (probing cache).
    // Stopped by run_solver after presolve completes; its best objective feeds into initial_cutoff.
    // This CPUFJ operates on *problem.original_problem_ptr (papilo-presolved
    // optimization_problem_t). Its solver-space differs from both the first-pass FJ (original
    // problem) and B&B (post-trivial- presolve), so initial_cutoff (user-space) is converted via
    // problem.get_solver_obj_from_user_obj.
    std::unique_ptr<detail::early_cpufj_t<i_t, f_t>> early_cpufj;
    bool run_early_cpufj = problem.has_papilo_presolve_data() &&
                           settings.determinism_mode != CUOPT_MODE_DETERMINISTIC &&
                           problem.original_problem_ptr->get_n_integers() > 0;
    if (run_early_cpufj) {
      auto* presolver_ptr = problem.presolve_data.papilo_presolve_ptr;
      auto mip_callbacks  = settings.get_mip_callbacks();
      f_t no_bound = problem.presolve_data.objective_scaling_factor >= 0 ? (f_t)-1e20 : (f_t)1e20;
      auto incumbent_callback =
        [presolver_ptr, mip_callbacks, no_bound](
          f_t solver_obj, f_t user_obj, const std::vector<f_t>& assignment) {
          std::vector<f_t> user_assignment;
          presolver_ptr->uncrush_primal_solution(assignment, user_assignment);
          invoke_solution_callbacks(mip_callbacks, user_obj, user_assignment, no_bound);
        };
      early_cpufj = std::make_unique<detail::early_cpufj_t<i_t, f_t>>(
        *problem.original_problem_ptr, settings.get_tolerances(), incumbent_callback);
      // Convert initial_cutoff from user-space to the CPUFJ's solver-space (papilo-presolved).
      // problem.get_solver_obj_from_user_obj uses the papilo offset/scale (matching the CPUFJ).
      if (std::isfinite(initial_cutoff)) {
        early_cpufj->set_best_objective(problem.get_solver_obj_from_user_obj(initial_cutoff));
      }
      early_cpufj->start();
      solver.context.early_cpufj_ptr = early_cpufj.get();
      CUOPT_LOG_DEBUG("Started early CPUFJ on papilo-presolved problem during cuOpt presolve");
    }

    auto presolved_sol            = solver.run_solver();
    bool is_feasible_on_presolved = presolved_sol.get_feasible();
    presolved_sol.problem_ptr     = &problem;
    // at this point we need to compute the feasibility on the original problem not the presolved
    // one
    bool is_feasible_on_original = presolved_sol.compute_feasibility();
    if (!scaled_problem.empty && is_feasible_on_presolved != is_feasible_on_original) {
      CUOPT_LOG_WARN(
        "The feasibility does not match on presolved and original problems. To overcome this "
        "issue, "
        "please provide a more numerically stable problem.");
    }

    auto sol = presolved_sol.get_solution(
      is_feasible_on_presolved || is_feasible_on_original, solver.get_solver_stats(), false);

    int hidesol =
      std::getenv("CUOPT_MIP_HIDE_SOLUTION") ? atoi(std::getenv("CUOPT_MIP_HIDE_SOLUTION")) : 0;
    if (!hidesol) { detail::print_solution(scaled_problem.handle_ptr, sol.get_solution()); }
    return sol;
  } catch (const std::exception& e) {
    CUOPT_LOG_ERROR("Unexpected error in run_mip: %s", e.what());
    throw;
  }
}

template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip(optimization_problem_t<i_t, f_t>& op_problem,
                                   mip_solver_settings_t<i_t, f_t> const& settings_const)
{
  try {
    mip_solver_settings_t<i_t, f_t> settings(settings_const);
    if (settings.presolver == presolver_t::Default || settings.presolver == presolver_t::PSLP) {
      if (settings.presolver == presolver_t::PSLP) {
        CUOPT_LOG_INFO(
          "PSLP presolver is not supported for MIP problems, using Papilo presolver instead");
      }
      settings.presolver = presolver_t::Papilo;
    }
    constexpr f_t max_time_limit = 1000000000;
    f_t time_limit =
      (settings.time_limit == 0 || settings.time_limit == std::numeric_limits<f_t>::infinity() ||
       settings.time_limit == std::numeric_limits<f_t>::max())
        ? max_time_limit
        : settings.time_limit;

    // Create log stream for file logging and add it to default logger
    init_logger_t log(settings.log_file, settings.log_to_console);
    // Init libraies before to not include it in solve time
    // This needs to be called before pdlp is initialized
    init_handler(op_problem.get_handle_ptr());

    print_version_info();

    // Initialize seed generator if a specific seed is requested
    if (settings.seed >= 0) { cuopt::seed_generator::set_seed(settings.seed); }

    raft::common::nvtx::range fun_scope("Running solver");

    // This is required as user might forget to set some fields
    problem_checking_t<i_t, f_t>::check_problem_representation(op_problem);
    problem_checking_t<i_t, f_t>::check_initial_solution_representation(op_problem, settings);

    CUOPT_LOG_INFO(
      "Solving a problem with %d constraints, %d variables (%d integers), and %d nonzeros",
      op_problem.get_n_constraints(),
      op_problem.get_n_variables(),
      op_problem.get_n_integers(),
      op_problem.get_nnz());
    op_problem.print_scaling_information();

    // Check for crossing bounds. Return infeasible if there are any
    if (problem_checking_t<i_t, f_t>::has_crossing_bounds(op_problem)) {
      return mip_solution_t<i_t, f_t>(mip_termination_status_t::Infeasible,
                                      solver_stats_t<i_t, f_t>{},
                                      op_problem.get_handle_ptr()->get_stream());
    }

    for (auto callback : settings.get_mip_callbacks()) {
      callback->template setup<f_t>(op_problem.get_n_variables());
    }

    auto timer = timer_t(time_limit);
    if (settings.mip_scaling != CUOPT_MIP_SCALING_OFF) {
      detail::mip_scaling_strategy_t<i_t, f_t> scaling(op_problem);
      scaling.scale_problem(settings.mip_scaling != CUOPT_MIP_SCALING_NO_OBJECTIVE);
    }
    double presolve_time = 0.0;
    std::unique_ptr<detail::third_party_presolve_t<i_t, f_t>> presolver;
    std::optional<detail::third_party_presolve_result_t<i_t, f_t>> presolve_result_opt;
    detail::problem_t<i_t, f_t> problem(
      op_problem, settings.get_tolerances(), settings.determinism_mode == CUOPT_MODE_DETERMINISTIC);

    auto run_presolve              = settings.presolver != presolver_t::None;
    run_presolve                   = run_presolve && settings.initial_solutions.size() == 0;
    bool has_set_solution_callback = false;
    for (auto callback : settings.get_mip_callbacks()) {
      if (callback != nullptr &&
          callback->get_type() == internals::base_solution_callback_type::SET_SOLUTION) {
        has_set_solution_callback = true;
        break;
      }
    }
    if (run_presolve && has_set_solution_callback) {
      CUOPT_LOG_WARN("Presolve is disabled because set_solution callbacks are provided.");
      run_presolve = false;
    }

    if (!run_presolve) { CUOPT_LOG_INFO("Presolve is disabled, skipping"); }

    // Start early FJ (CPU and GPU) during presolve to find incumbents ASAP
    // Only run if presolve is enabled (gives FJ time to find solutions)
    // and we're not in deterministic mode
    std::unique_ptr<detail::early_cpufj_t<i_t, f_t>> early_cpufj;
    std::unique_ptr<detail::early_gpufj_t<i_t, f_t>> early_gpufj;

    // Track best incumbent found during presolve (shared across CPU and GPU FJ).
    // early_best_objective is in the original problem's solver-space (always minimization),
    // used for fast comparison in the callback.
    // early_best_user_obj is the corresponding user-space objective,
    // passed to run_mip for correct cross-space conversion.
    std::atomic<f_t> early_best_objective{std::numeric_limits<f_t>::infinity()};
    f_t early_best_user_obj{std::numeric_limits<f_t>::infinity()};
    std::mutex early_callback_mutex;

    bool run_early_fj = run_presolve && settings.determinism_mode != CUOPT_MODE_DETERMINISTIC &&
                        op_problem.get_n_integers() > 0 && op_problem.get_n_constraints() > 0;
    f_t no_bound = problem.presolve_data.objective_scaling_factor >= 0 ? (f_t)-1e20 : (f_t)1e20;
    if (run_early_fj) {
      auto early_fj_callback = [&early_best_objective,
                                &early_best_user_obj,
                                &early_callback_mutex,
                                mip_callbacks = settings.get_mip_callbacks(),
                                no_bound](
                                 f_t solver_obj, f_t user_obj, const std::vector<f_t>& assignment) {
        std::lock_guard<std::mutex> lock(early_callback_mutex);
        if (solver_obj >= early_best_objective.load()) { return; }
        early_best_objective.store(solver_obj);
        early_best_user_obj  = user_obj;
        auto user_assignment = assignment;
        invoke_solution_callbacks(mip_callbacks, user_obj, user_assignment, no_bound);
      };

      // Start early CPUFJ on original problem (will restart on presolved problem after Papilo)
      early_cpufj = std::make_unique<detail::early_cpufj_t<i_t, f_t>>(
        op_problem, settings.get_tolerances(), early_fj_callback);
      early_cpufj->start();
      CUOPT_LOG_INFO("Started early CPUFJ on original problem");

      // Start early GPU FJ (uses GPU while CPU is busy with Papilo)
      early_gpufj =
        std::make_unique<detail::early_gpufj_t<i_t, f_t>>(op_problem, settings, early_fj_callback);
      early_gpufj->start();
      CUOPT_LOG_INFO("Started early GPUFJ during presolve");
    }

    auto constexpr const dual_postsolve = false;
    if (run_presolve) {
      detail::sort_csr(op_problem);
      // allocate not more than 10% of the time limit to presolve.
      // Note that this is not the presolve time, but the time limit for presolve.
      const auto& hp = settings.heuristic_params;
      double presolve_time_limit =
        std::min(hp.presolve_time_ratio * time_limit, hp.presolve_max_time);
      if (settings.determinism_mode == CUOPT_MODE_DETERMINISTIC) {
        presolve_time_limit = std::numeric_limits<double>::infinity();
      }
      presolver   = std::make_unique<detail::third_party_presolve_t<i_t, f_t>>();
      auto result = presolver->apply(op_problem,
                                     cuopt::linear_programming::problem_category_t::MIP,
                                     settings.presolver,
                                     dual_postsolve,
                                     settings.tolerances.absolute_tolerance,
                                     settings.tolerances.relative_tolerance,
                                     presolve_time_limit,
                                     settings.num_cpu_threads);

      if (result.status == detail::third_party_presolve_status_t::INFEASIBLE) {
        return mip_solution_t<i_t, f_t>(mip_termination_status_t::Infeasible,
                                        solver_stats_t<i_t, f_t>{},
                                        op_problem.get_handle_ptr()->get_stream());
      }
      if (result.status == detail::third_party_presolve_status_t::UNBNDORINFEAS) {
        return mip_solution_t<i_t, f_t>(mip_termination_status_t::UnboundedOrInfeasible,
                                        solver_stats_t<i_t, f_t>{},
                                        op_problem.get_handle_ptr()->get_stream());
      }
      if (result.status == detail::third_party_presolve_status_t::UNBOUNDED) {
        return mip_solution_t<i_t, f_t>(mip_termination_status_t::Unbounded,
                                        solver_stats_t<i_t, f_t>{},
                                        op_problem.get_handle_ptr()->get_stream());
      }
      presolve_result_opt.emplace(std::move(result));

      problem = detail::problem_t<i_t, f_t>(presolve_result_opt->reduced_problem);
      problem.set_papilo_presolve_data(presolver.get(),
                                       presolve_result_opt->reduced_to_original_map,
                                       presolve_result_opt->original_to_reduced_map,
                                       op_problem.get_n_variables());
      problem.set_implied_integers(presolve_result_opt->implied_integer_indices);
      presolve_time = timer.elapsed_time();
      if (presolve_result_opt->implied_integer_indices.size() > 0) {
        CUOPT_LOG_INFO("%d implied integers", presolve_result_opt->implied_integer_indices.size());
      }
      CUOPT_LOG_INFO("Papilo presolve time: %.2f", presolve_time);
    }

    // Stop early GPU FJ now that Papilo presolve is complete
    if (early_gpufj) {
      early_gpufj->stop();
      if (early_gpufj->solution_found()) {
        CUOPT_LOG_INFO("Early GPU FJ found incumbent with objective %.6e during presolve",
                       early_gpufj->get_best_objective());
      }
      early_gpufj.reset();  // Free GPU memory
    }

    if (early_cpufj && run_presolve && presolve_result_opt.has_value()) {
      early_cpufj->stop();
      if (early_cpufj->solution_found()) {
        CUOPT_LOG_INFO("Early CPUFJ (original) found incumbent with objective %.6e",
                       early_cpufj->get_best_objective());
      }
      early_cpufj.reset();
    }

    if (settings.user_problem_file != "") {
      CUOPT_LOG_INFO("Writing user problem to file: %s", settings.user_problem_file.c_str());
      op_problem.write_to_mps(settings.user_problem_file);
    }
    if (run_presolve && presolve_result_opt.has_value() && settings.presolve_file != "") {
      CUOPT_LOG_INFO("Writing presolved problem to file: %s", settings.presolve_file.c_str());
      presolve_result_opt->reduced_problem.write_to_mps(settings.presolve_file);
    }

    // early_best_user_obj is in user-space.
    // run_mip stores it in context.initial_cutoff and converts to target spaces as needed.
    auto sol = run_mip(problem, settings, timer, early_best_user_obj);

    if (run_presolve) {
      auto status_to_skip = sol.get_termination_status() == mip_termination_status_t::TimeLimit ||
                            sol.get_termination_status() == mip_termination_status_t::WorkLimit ||
                            sol.get_termination_status() == mip_termination_status_t::Infeasible;
      auto primal_solution =
        cuopt::device_copy(sol.get_solution(), op_problem.get_handle_ptr()->get_stream());
      rmm::device_uvector<f_t> dual_solution(0, op_problem.get_handle_ptr()->get_stream());
      rmm::device_uvector<f_t> reduced_costs(0, op_problem.get_handle_ptr()->get_stream());
      presolver->undo(primal_solution,
                      dual_solution,
                      reduced_costs,
                      cuopt::linear_programming::problem_category_t::MIP,
                      status_to_skip,
                      dual_postsolve,
                      op_problem.get_handle_ptr()->get_stream());
      if (!status_to_skip) {
        thrust::fill(rmm::exec_policy(op_problem.get_handle_ptr()->get_stream()),
                     dual_solution.data(),
                     dual_solution.data() + dual_solution.size(),
                     std::numeric_limits<f_t>::signaling_NaN());
        thrust::fill(rmm::exec_policy(op_problem.get_handle_ptr()->get_stream()),
                     reduced_costs.data(),
                     reduced_costs.data() + reduced_costs.size(),
                     std::numeric_limits<f_t>::signaling_NaN());
        detail::problem_t<i_t, f_t> full_problem(op_problem);
        detail::solution_t<i_t, f_t> full_sol(full_problem);
        full_sol.copy_new_assignment(
          cuopt::host_copy(primal_solution, op_problem.get_handle_ptr()->get_stream()));
        full_sol.compute_feasibility();
        if (!full_sol.get_feasible()) {
          CUOPT_LOG_WARN("The solution is not feasible after post solve");
        }

        auto full_stats = sol.get_stats();
        // add third party presolve time to cuopt presolve time
        full_stats.presolve_time += presolve_time;

        // FIXME:: reduced_solution.get_stats() is not correct, we need to compute the stats for
        // the full problem
        full_sol.post_process_completed = true;  // hack
        sol                             = full_sol.get_solution(true, full_stats);
      }
    }

    if (settings.sol_file != "") {
      CUOPT_LOG_INFO("Writing solution to file %s", settings.sol_file.c_str());
      sol.write_to_sol_file(settings.sol_file, op_problem.get_handle_ptr()->get_stream());
    }
    return sol;
  } catch (const cuopt::logic_error& e) {
    CUOPT_LOG_ERROR("Error in solve_mip: %s", e.what());
    return mip_solution_t<i_t, f_t>{e, op_problem.get_handle_ptr()->get_stream()};
  } catch (const std::bad_alloc& e) {
    CUOPT_LOG_ERROR("Error in solve_mip: %s", e.what());
    return mip_solution_t<i_t, f_t>{
      cuopt::logic_error("Memory allocation failed", cuopt::error_type_t::RuntimeError),
      op_problem.get_handle_ptr()->get_stream()};
  } catch (const std::exception& e) {
    CUOPT_LOG_ERROR("Unexpected error in solve_mip: %s", e.what());
    throw;
  }
}

template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip(
  raft::handle_t const* handle_ptr,
  const cuopt::mps_parser::mps_data_model_t<i_t, f_t>& mps_data_model,
  mip_solver_settings_t<i_t, f_t> const& settings)
{
  auto op_problem = mps_data_model_to_optimization_problem(handle_ptr, mps_data_model);
  return solve_mip(op_problem, settings);
}

// ============================================================================
// CPU problem overload (convert to GPU, solve, convert solution back)
// ============================================================================

template <typename i_t, typename f_t>
std::unique_ptr<mip_solution_interface_t<i_t, f_t>> solve_mip(
  cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
  mip_solver_settings_t<i_t, f_t> const& settings)
{
  // Create CUDA resources for the conversion
  rmm::cuda_stream stream;
  raft::handle_t handle(stream);

  // Convert CPU problem to GPU problem
  auto gpu_problem = cpu_problem.to_optimization_problem(&handle);

  // Synchronize before solving to ensure conversion is complete
  stream.synchronize();

  // Solve on GPU
  auto gpu_solution = solve_mip<i_t, f_t>(*gpu_problem, settings);

  // Ensure all GPU work from the solve is complete before D2H copies in to_cpu_solution(),
  // which uses rmm::cuda_stream_per_thread (a different stream than the solver used).
  stream.synchronize();

  // Convert GPU solution back to CPU
  gpu_mip_solution_t<i_t, f_t> gpu_sol_interface(std::move(gpu_solution));
  return gpu_sol_interface.to_cpu_solution();
}

// ============================================================================
// Interface-based solve overload with remote execution support
// ============================================================================

template <typename i_t, typename f_t>
std::unique_ptr<mip_solution_interface_t<i_t, f_t>> solve_mip(
  optimization_problem_interface_t<i_t, f_t>* problem_interface,
  mip_solver_settings_t<i_t, f_t> const& settings)
{
  cuopt_expects(problem_interface != nullptr,
                error_type_t::ValidationError,
                "problem_interface cannot be null");

  try {
    // Check if remote execution is enabled (always uses CPU backend)
    if (is_remote_execution_enabled()) {
      auto* cpu_prob = dynamic_cast<cpu_optimization_problem_t<i_t, f_t>*>(problem_interface);
      cuopt_expects(cpu_prob != nullptr,
                    error_type_t::ValidationError,
                    "Remote execution requires CPU memory backend");
      return solve_mip_remote(*cpu_prob, settings);
    }

    // Local execution - dispatch to appropriate overload based on problem type
    auto* cpu_prob = dynamic_cast<cpu_optimization_problem_t<i_t, f_t>*>(problem_interface);
    if (cpu_prob != nullptr) { return solve_mip(*cpu_prob, settings); }

    // GPU problem: call GPU solver directly
    auto* gpu_prob = dynamic_cast<optimization_problem_t<i_t, f_t>*>(problem_interface);
    cuopt_expects(gpu_prob != nullptr,
                  error_type_t::ValidationError,
                  "problem_interface must be either a CPU or GPU optimization problem");
    auto gpu_solution = solve_mip<i_t, f_t>(*gpu_prob, settings);
    return std::make_unique<gpu_mip_solution_t<i_t, f_t>>(std::move(gpu_solution));
  } catch (const cuopt::logic_error& e) {
    CUOPT_LOG_ERROR("Error in solve_mip (interface): %s", e.what());
    throw;
  } catch (const std::bad_alloc& e) {
    CUOPT_LOG_ERROR("Error in solve_mip (interface): %s", e.what());
    throw cuopt::logic_error("Memory allocation failed", cuopt::error_type_t::RuntimeError);
  }
}

#define INSTANTIATE(F_TYPE)                                                               \
  template mip_solution_t<int, F_TYPE> solve_mip(                                         \
    optimization_problem_t<int, F_TYPE>& op_problem,                                      \
    mip_solver_settings_t<int, F_TYPE> const& settings);                                  \
                                                                                          \
  template mip_solution_t<int, F_TYPE> solve_mip(                                         \
    raft::handle_t const* handle_ptr,                                                     \
    const cuopt::mps_parser::mps_data_model_t<int, F_TYPE>& mps_data_model,               \
    mip_solver_settings_t<int, F_TYPE> const& settings);                                  \
                                                                                          \
  template std::unique_ptr<mip_solution_interface_t<int, F_TYPE>> solve_mip(              \
    cpu_optimization_problem_t<int, F_TYPE>&, mip_solver_settings_t<int, F_TYPE> const&); \
                                                                                          \
  template std::unique_ptr<mip_solution_interface_t<int, F_TYPE>> solve_mip(              \
    optimization_problem_interface_t<int, F_TYPE>*, mip_solver_settings_t<int, F_TYPE> const&);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

}  // namespace cuopt::linear_programming
