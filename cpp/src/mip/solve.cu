/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>

#include <mip/mip_constants.hpp>
#include <mip/presolve/third_party_presolve.hpp>
#include <mip/presolve/trivial_presolve.cuh>
#include <mip/solver.cuh>
#include <mip/utilities/sort_csr.cuh>
#include <mip/utils.cuh>

#include <linear_programming/initial_scaling_strategy/initial_scaling.cuh>
#include <linear_programming/pdlp.cuh>
#include <linear_programming/restart_strategy/pdlp_restart_strategy.cuh>
#include <linear_programming/step_size_strategy/adaptive_step_size_strategy.hpp>
#include <linear_programming/utilities/problem_checking.cuh>
#include <linear_programming/utils.cuh>
#include <utilities/logger.hpp>
#include <utilities/timer.hpp>
#include <utilities/version_info.hpp>

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>

#include <mps_parser/mps_data_model.hpp>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/common/nvtx.hpp>
#include <raft/core/handle.hpp>

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

template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> run_mip(detail::problem_t<i_t, f_t>& problem,
                                 mip_solver_settings_t<i_t, f_t> const& settings,
                                 cuopt::timer_t& timer)
{
  raft::common::nvtx::range fun_scope("run_mip");
  auto constexpr const running_mip = true;

  // TODO ask Akif and Alice how was this passed down?
  auto hyper_params                                     = settings.hyper_params;
  hyper_params.update_primal_weight_on_initial_solution = false;
  hyper_params.update_step_size_on_initial_solution     = true;
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
    thrust::for_each(problem.handle_ptr->get_thrust_policy(),
                     thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(problem.n_variables),
                     [sol = solution.assignment.data(), pb = problem.view()] __device__(i_t index) {
                       auto bounds = pb.variable_bounds[index];
                       sol[index]  = pb.objective_coefficients[index] > 0 ? get_lower(bounds)
                                                                          : get_upper(bounds);
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

  CUOPT_LOG_INFO("Objective offset %f scaling_factor %f",
                 problem.presolve_data.objective_offset,
                 problem.presolve_data.objective_scaling_factor);
  cuopt_assert(problem.original_problem_ptr->get_n_variables() == scaled_problem.n_variables,
               "Size mismatch");
  cuopt_assert(problem.original_problem_ptr->get_n_constraints() == scaled_problem.n_constraints,
               "Size mismatch");
  detail::pdlp_initial_scaling_strategy_t<i_t, f_t> scaling(
    scaled_problem.handle_ptr,
    scaled_problem,
    hyper_params.default_l_inf_ruiz_iterations,
    (f_t)hyper_params.default_alpha_pock_chambolle_rescaling,
    scaled_problem.reverse_coefficients,
    scaled_problem.reverse_offsets,
    scaled_problem.reverse_constraints,
    nullptr,
    hyper_params,
    running_mip);

  cuopt_func_call(auto saved_problem = scaled_problem);
  if (settings.mip_scaling) {
    scaling.scale_problem();
    if (settings.initial_solutions.size() > 0) {
      for (const auto& initial_solution : settings.initial_solutions) {
        scaling.scale_primal(*initial_solution);
      }
    }
  }
  // only call preprocess on scaled problem, so we can compute feasibility on the original problem
  scaled_problem.preprocess_problem();
  // cuopt_func_call((check_scaled_problem<i_t, f_t>(scaled_problem, saved_problem)));
  detail::trivial_presolve(scaled_problem);

  detail::mip_solver_t<i_t, f_t> solver(scaled_problem, settings, scaling, timer);
  auto scaled_sol                 = solver.run_solver();
  bool is_feasible_before_scaling = scaled_sol.get_feasible();
  scaled_sol.problem_ptr          = &problem;
  if (settings.mip_scaling) { scaling.unscale_solutions(scaled_sol); }
  // at this point we need to compute the feasibility on the original problem not the presolved one
  bool is_feasible_after_unscaling = scaled_sol.compute_feasibility();
  if (!scaled_problem.empty && is_feasible_before_scaling != is_feasible_after_unscaling) {
    CUOPT_LOG_WARN(
      "The feasibility does not match on scaled and unscaled problems. To overcome this issue, "
      "please provide a more numerically stable problem.");
  }

  auto sol = scaled_sol.get_solution(
    is_feasible_before_scaling || is_feasible_after_unscaling, solver.get_solver_stats(), false);
  detail::print_solution(scaled_problem.handle_ptr, sol.get_solution());
  return sol;
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

    auto timer           = cuopt::timer_t(time_limit);
    double presolve_time = 0.0;
    std::unique_ptr<detail::third_party_presolve_t<i_t, f_t>> presolver;
    std::optional<detail::third_party_presolve_result_t<i_t, f_t>> presolve_result;
    detail::problem_t<i_t, f_t> problem(op_problem, settings.get_tolerances());

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

    auto constexpr const dual_postsolve = false;
    if (run_presolve) {
      detail::sort_csr(op_problem);
      // allocate not more than 10% of the time limit to presolve.
      // Note that this is not the presolve time, but the time limit for presolve.
      const double presolve_time_limit = std::min(0.1 * time_limit, 60.0);
      presolver   = std::make_unique<detail::third_party_presolve_t<i_t, f_t>>();
      auto result = presolver->apply(op_problem,
                                     cuopt::linear_programming::problem_category_t::MIP,
                                     settings.presolver,
                                     dual_postsolve,
                                     settings.tolerances.absolute_tolerance,
                                     settings.tolerances.relative_tolerance,
                                     presolve_time_limit,
                                     settings.num_cpu_threads);
      if (!result.has_value()) {
        return mip_solution_t<i_t, f_t>(mip_termination_status_t::Infeasible,
                                        solver_stats_t<i_t, f_t>{},
                                        op_problem.get_handle_ptr()->get_stream());
      }
      presolve_result.emplace(std::move(*result));

      problem = detail::problem_t<i_t, f_t>(presolve_result->reduced_problem);
      problem.set_papilo_presolve_data(presolver.get(),
                                       presolve_result->reduced_to_original_map,
                                       presolve_result->original_to_reduced_map,
                                       op_problem.get_n_variables());
      problem.set_implied_integers(presolve_result->implied_integer_indices);
      presolve_time = timer.elapsed_time();
      if (presolve_result->implied_integer_indices.size() > 0) {
        CUOPT_LOG_INFO("%d implied integers", presolve_result->implied_integer_indices.size());
      }
      if (problem.is_objective_integral()) { CUOPT_LOG_INFO("Objective function is integral"); }
      CUOPT_LOG_INFO("Papilo presolve time: %.2f", presolve_time);
    }
    if (settings.user_problem_file != "") {
      CUOPT_LOG_INFO("Writing user problem to file: %s", settings.user_problem_file.c_str());
      op_problem.write_to_mps(settings.user_problem_file);
    }

    auto sol = run_mip(problem, settings, timer);

    if (run_presolve) {
      auto status_to_skip = sol.get_termination_status() == mip_termination_status_t::TimeLimit ||
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

#define INSTANTIATE(F_TYPE)                                                 \
  template mip_solution_t<int, F_TYPE> solve_mip(                           \
    optimization_problem_t<int, F_TYPE>& op_problem,                        \
    mip_solver_settings_t<int, F_TYPE> const& settings);                    \
                                                                            \
  template mip_solution_t<int, F_TYPE> solve_mip(                           \
    raft::handle_t const* handle_ptr,                                       \
    const cuopt::mps_parser::mps_data_model_t<int, F_TYPE>& mps_data_model, \
    mip_solver_settings_t<int, F_TYPE> const& settings);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

}  // namespace cuopt::linear_programming
