/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/error.hpp>
#include <linear_programming/pdlp.cuh>
#include <linear_programming/restart_strategy/pdlp_restart_strategy.cuh>
#include <linear_programming/step_size_strategy/adaptive_step_size_strategy.hpp>
#include <linear_programming/translate.hpp>
#include <linear_programming/utilities/logger_init.hpp>
#include <linear_programming/utilities/problem_checking.cuh>
#include <linear_programming/utils.cuh>

#include <mip/mip_constants.hpp>
#include <mip/presolve/trivial_presolve.cuh>
#include <mip/solver.cuh>

#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/solve.hpp>

#include <mps_parser/mps_data_model.hpp>
#include <utilities/copy_helpers.hpp>

#include <dual_simplex/crossover.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <linear_programming/utilities/problem_checking.cuh>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/common/nvtx.hpp>
#include <raft/core/handle.hpp>

#include <thread>  // For std::thread

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

// Corresponds to the first good general settings we found
// It's what was used for the GTC results
static void set_Stable1()
{
  pdlp_hyper_params::initial_step_size_scaling                                  = 1.6;
  pdlp_hyper_params::default_l_inf_ruiz_iterations                              = 1;
  pdlp_hyper_params::do_pock_chambolle_scaling                                  = true;
  pdlp_hyper_params::do_ruiz_scaling                                            = true;
  pdlp_hyper_params::default_alpha_pock_chambolle_rescaling                     = 1.3;
  pdlp_hyper_params::default_artificial_restart_threshold                       = 0.5;
  pdlp_hyper_params::compute_initial_step_size_before_scaling                   = false;
  pdlp_hyper_params::compute_initial_primal_weight_before_scaling               = true;
  pdlp_hyper_params::initial_primal_weight_c_scaling                            = 2.2;
  pdlp_hyper_params::initial_primal_weight_b_scaling                            = 4.6;
  pdlp_hyper_params::major_iteration                                            = 52;
  pdlp_hyper_params::min_iteration_restart                                      = 0;
  pdlp_hyper_params::restart_strategy                                           = 1;
  pdlp_hyper_params::never_restart_to_average                                   = false;
  pdlp_hyper_params::host_default_reduction_exponent                            = 0.5;
  pdlp_hyper_params::host_default_growth_exponent                               = 0.9;
  pdlp_hyper_params::host_default_primal_weight_update_smoothing                = 0.3;
  pdlp_hyper_params::host_default_sufficient_reduction_for_restart              = 0.2;
  pdlp_hyper_params::host_default_necessary_reduction_for_restart               = 0.5;
  pdlp_hyper_params::host_primal_importance                                     = 1.8;
  pdlp_hyper_params::host_primal_distance_smoothing                             = 0.6;
  pdlp_hyper_params::host_dual_distance_smoothing                               = 0.2;
  pdlp_hyper_params::compute_last_restart_before_new_primal_weight              = false;
  pdlp_hyper_params::artificial_restart_in_main_loop                            = false;
  pdlp_hyper_params::rescale_for_restart                                        = false;
  pdlp_hyper_params::update_primal_weight_on_initial_solution                   = false;
  pdlp_hyper_params::update_step_size_on_initial_solution                       = false;
  pdlp_hyper_params::handle_some_primal_gradients_on_finite_bounds_as_residuals = true;
  pdlp_hyper_params::project_initial_primal                                     = false;
}

// Even better general setting due to proper primal gradient handling for KKT restart and initial
// projection
static void set_Stable2()
{
  pdlp_hyper_params::initial_step_size_scaling                                  = 1.0;
  pdlp_hyper_params::default_l_inf_ruiz_iterations                              = 10;
  pdlp_hyper_params::do_pock_chambolle_scaling                                  = true;
  pdlp_hyper_params::do_ruiz_scaling                                            = true;
  pdlp_hyper_params::default_alpha_pock_chambolle_rescaling                     = 1.0;
  pdlp_hyper_params::default_artificial_restart_threshold                       = 0.36;
  pdlp_hyper_params::compute_initial_step_size_before_scaling                   = false;
  pdlp_hyper_params::compute_initial_primal_weight_before_scaling               = false;
  pdlp_hyper_params::initial_primal_weight_c_scaling                            = 1.0;
  pdlp_hyper_params::initial_primal_weight_b_scaling                            = 1.0;
  pdlp_hyper_params::major_iteration                                            = 40;
  pdlp_hyper_params::min_iteration_restart                                      = 10;
  pdlp_hyper_params::restart_strategy                                           = 1;
  pdlp_hyper_params::never_restart_to_average                                   = false;
  pdlp_hyper_params::host_default_reduction_exponent                            = 0.3;
  pdlp_hyper_params::host_default_growth_exponent                               = 0.6;
  pdlp_hyper_params::host_default_primal_weight_update_smoothing                = 0.5;
  pdlp_hyper_params::host_default_sufficient_reduction_for_restart              = 0.2;
  pdlp_hyper_params::host_default_necessary_reduction_for_restart               = 0.8;
  pdlp_hyper_params::host_primal_importance                                     = 1.0;
  pdlp_hyper_params::host_primal_distance_smoothing                             = 0.5;
  pdlp_hyper_params::host_dual_distance_smoothing                               = 0.5;
  pdlp_hyper_params::compute_last_restart_before_new_primal_weight              = true;
  pdlp_hyper_params::artificial_restart_in_main_loop                            = false;
  pdlp_hyper_params::rescale_for_restart                                        = true;
  pdlp_hyper_params::update_primal_weight_on_initial_solution                   = false;
  pdlp_hyper_params::update_step_size_on_initial_solution                       = false;
  pdlp_hyper_params::handle_some_primal_gradients_on_finite_bounds_as_residuals = false;
  pdlp_hyper_params::project_initial_primal                                     = true;
}

// Legacy/Original/Initial PDLP settings
static void set_Methodical1()
{
  pdlp_hyper_params::initial_step_size_scaling                                  = 1.0;
  pdlp_hyper_params::default_l_inf_ruiz_iterations                              = 5;
  pdlp_hyper_params::do_pock_chambolle_scaling                                  = true;
  pdlp_hyper_params::do_ruiz_scaling                                            = true;
  pdlp_hyper_params::default_alpha_pock_chambolle_rescaling                     = 1.0;
  pdlp_hyper_params::default_artificial_restart_threshold                       = 0.5;
  pdlp_hyper_params::compute_initial_step_size_before_scaling                   = false;
  pdlp_hyper_params::compute_initial_primal_weight_before_scaling               = false;
  pdlp_hyper_params::initial_primal_weight_c_scaling                            = 1.0;
  pdlp_hyper_params::initial_primal_weight_b_scaling                            = 1.0;
  pdlp_hyper_params::major_iteration                                            = 64;
  pdlp_hyper_params::min_iteration_restart                                      = 0;
  pdlp_hyper_params::restart_strategy                                           = 2;
  pdlp_hyper_params::never_restart_to_average                                   = false;
  pdlp_hyper_params::host_default_reduction_exponent                            = 0.3;
  pdlp_hyper_params::host_default_growth_exponent                               = 0.6;
  pdlp_hyper_params::host_default_primal_weight_update_smoothing                = 0.5;
  pdlp_hyper_params::host_default_sufficient_reduction_for_restart              = 0.1;
  pdlp_hyper_params::host_default_necessary_reduction_for_restart               = 0.9;
  pdlp_hyper_params::host_primal_importance                                     = 1.0;
  pdlp_hyper_params::host_primal_distance_smoothing                             = 0.5;
  pdlp_hyper_params::host_dual_distance_smoothing                               = 0.5;
  pdlp_hyper_params::compute_last_restart_before_new_primal_weight              = true;
  pdlp_hyper_params::artificial_restart_in_main_loop                            = false;
  pdlp_hyper_params::rescale_for_restart                                        = false;
  pdlp_hyper_params::update_primal_weight_on_initial_solution                   = false;
  pdlp_hyper_params::update_step_size_on_initial_solution                       = false;
  pdlp_hyper_params::handle_some_primal_gradients_on_finite_bounds_as_residuals = true;
  pdlp_hyper_params::project_initial_primal                                     = false;
}

// Can be extremly faster but usually leads to more divergence
// Used for the blog post results
static void set_Fast1()
{
  pdlp_hyper_params::initial_step_size_scaling                                  = 0.8;
  pdlp_hyper_params::default_l_inf_ruiz_iterations                              = 6;
  pdlp_hyper_params::do_pock_chambolle_scaling                                  = true;
  pdlp_hyper_params::do_ruiz_scaling                                            = false;
  pdlp_hyper_params::default_alpha_pock_chambolle_rescaling                     = 2.0;
  pdlp_hyper_params::default_artificial_restart_threshold                       = 0.3;
  pdlp_hyper_params::compute_initial_step_size_before_scaling                   = false;
  pdlp_hyper_params::compute_initial_primal_weight_before_scaling               = true;
  pdlp_hyper_params::initial_primal_weight_c_scaling                            = 1.2;
  pdlp_hyper_params::initial_primal_weight_b_scaling                            = 1.2;
  pdlp_hyper_params::major_iteration                                            = 76;
  pdlp_hyper_params::min_iteration_restart                                      = 6;
  pdlp_hyper_params::restart_strategy                                           = 1;
  pdlp_hyper_params::never_restart_to_average                                   = true;
  pdlp_hyper_params::host_default_reduction_exponent                            = 0.4;
  pdlp_hyper_params::host_default_growth_exponent                               = 0.6;
  pdlp_hyper_params::host_default_primal_weight_update_smoothing                = 0.5;
  pdlp_hyper_params::host_default_sufficient_reduction_for_restart              = 0.3;
  pdlp_hyper_params::host_default_necessary_reduction_for_restart               = 0.9;
  pdlp_hyper_params::host_primal_importance                                     = 0.8;
  pdlp_hyper_params::host_primal_distance_smoothing                             = 0.8;
  pdlp_hyper_params::host_dual_distance_smoothing                               = 0.3;
  pdlp_hyper_params::compute_last_restart_before_new_primal_weight              = true;
  pdlp_hyper_params::artificial_restart_in_main_loop                            = true;
  pdlp_hyper_params::rescale_for_restart                                        = true;
  pdlp_hyper_params::update_primal_weight_on_initial_solution                   = false;
  pdlp_hyper_params::update_step_size_on_initial_solution                       = false;
  pdlp_hyper_params::handle_some_primal_gradients_on_finite_bounds_as_residuals = true;
  pdlp_hyper_params::project_initial_primal                                     = false;
}

template <typename i_t, typename f_t>
void set_pdlp_solver_mode(pdlp_solver_settings_t<i_t, f_t> const& settings)
{
  if (settings.pdlp_solver_mode == pdlp_solver_mode_t::Stable2)
    set_Stable2();
  else if (settings.pdlp_solver_mode == pdlp_solver_mode_t::Stable1)
    set_Stable1();
  else if (settings.pdlp_solver_mode == pdlp_solver_mode_t::Methodical1)
    set_Methodical1();
  else if (settings.pdlp_solver_mode == pdlp_solver_mode_t::Fast1)
    set_Fast1();
}

void setup_device_symbols(rmm::cuda_stream_view stream_view)
{
  raft::common::nvtx::range fun_scope("Setting device symbol");
  detail::set_adaptive_step_size_hyper_parameters(stream_view);
  detail::set_restart_hyper_parameters(stream_view);
  detail::set_pdlp_hyper_parameters(stream_view);
}

std::atomic<int> global_concurrent_halt;

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> convert_dual_simplex_sol(
  detail::problem_t<i_t, f_t>& problem,
  const dual_simplex::lp_solution_t<i_t, f_t>& solution,
  dual_simplex::lp_status_t status,
  f_t duration,
  f_t norm_user_objective,
  f_t norm_rhs)
{
  auto to_termination_status = [](dual_simplex::lp_status_t status) {
    switch (status) {
      case dual_simplex::lp_status_t::OPTIMAL: return pdlp_termination_status_t::Optimal;
      case dual_simplex::lp_status_t::INFEASIBLE:
        return pdlp_termination_status_t::PrimalInfeasible;
      case dual_simplex::lp_status_t::UNBOUNDED: return pdlp_termination_status_t::DualInfeasible;
      case dual_simplex::lp_status_t::TIME_LIMIT: return pdlp_termination_status_t::TimeLimit;
      case dual_simplex::lp_status_t::ITERATION_LIMIT:
        return pdlp_termination_status_t::IterationLimit;
      case dual_simplex::lp_status_t::CONCURRENT_LIMIT:
        return pdlp_termination_status_t::ConcurrentLimit;
      default: return pdlp_termination_status_t::NumericalError;
    }
  };

  rmm::device_uvector<f_t> final_primal_solution =
    cuopt::device_copy(solution.x, problem.handle_ptr->get_stream());
  rmm::device_uvector<f_t> final_dual_solution =
    cuopt::device_copy(solution.y, problem.handle_ptr->get_stream());
  rmm::device_uvector<f_t> final_reduced_cost =
    cuopt::device_copy(solution.z, problem.handle_ptr->get_stream());
  problem.handle_ptr->sync_stream();

  // Should be filled with more information from dual simplex
  typename optimization_problem_solution_t<i_t, f_t>::additional_termination_information_t info;
  info.solved_by_pdlp                  = false;
  info.primal_objective                = solution.user_objective;
  info.dual_objective                  = solution.user_objective;
  info.gap                             = 0.0;
  info.relative_gap                    = 0.0;
  info.solve_time                      = duration;
  info.number_of_steps_taken           = solution.iterations;
  info.total_number_of_attempted_steps = solution.iterations;
  info.l2_primal_residual              = solution.l2_primal_residual;
  info.l2_dual_residual                = solution.l2_dual_residual;
  info.l2_relative_primal_residual     = solution.l2_primal_residual / (1.0 + norm_user_objective);
  info.l2_relative_dual_residual       = solution.l2_dual_residual / (1.0 + norm_rhs);
  info.max_primal_ray_infeasibility    = 0.0;
  info.primal_ray_linear_objective     = 0.0;
  info.max_dual_ray_infeasibility      = 0.0;
  info.dual_ray_linear_objective       = 0.0;

  pdlp_termination_status_t termination_status = to_termination_status(status);
  auto sol = optimization_problem_solution_t<i_t, f_t>(final_primal_solution,
                                                       final_dual_solution,
                                                       final_reduced_cost,
                                                       problem.objective_name,
                                                       problem.var_names,
                                                       problem.row_names,
                                                       info,
                                                       termination_status);

  if (termination_status != pdlp_termination_status_t::Optimal &&
      termination_status != pdlp_termination_status_t::TimeLimit &&
      termination_status != pdlp_termination_status_t::ConcurrentLimit) {
    CUOPT_LOG_INFO("Dual simplex status %s", sol.get_termination_status_string().c_str());
  }

  problem.handle_ptr->sync_stream();
  return sol;
}

template <typename i_t, typename f_t>
std::tuple<dual_simplex::lp_solution_t<i_t, f_t>, dual_simplex::lp_status_t, f_t, f_t, f_t>
run_dual_simplex(dual_simplex::user_problem_t<i_t, f_t>& user_problem,
                 pdlp_solver_settings_t<i_t, f_t> const& settings)
{
  auto start_solver = std::chrono::high_resolution_clock::now();

  f_t norm_user_objective = dual_simplex::vector_norm2<i_t, f_t>(user_problem.objective);
  f_t norm_rhs            = dual_simplex::vector_norm2<i_t, f_t>(user_problem.rhs);

  dual_simplex::simplex_solver_settings_t<i_t, f_t> dual_simplex_settings;
  dual_simplex_settings.time_limit      = settings.time_limit;
  dual_simplex_settings.iteration_limit = settings.iteration_limit;
  dual_simplex_settings.concurrent_halt = settings.concurrent_halt;
  if (dual_simplex_settings.concurrent_halt != nullptr) {
    // Don't show the dual simplex log in concurrent mode. Show the PDLP log instead
    dual_simplex_settings.log.log = false;
  }

  dual_simplex::lp_solution_t<i_t, f_t> solution(user_problem.num_rows, user_problem.num_cols);
  auto status =
    dual_simplex::solve_linear_program<i_t, f_t>(user_problem, dual_simplex_settings, solution);

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_solver);

  CUOPT_LOG_INFO("Dual simplex finished in %.2f seconds", duration.count() / 1000.0);

  if (settings.concurrent_halt != nullptr && (status == dual_simplex::lp_status_t::OPTIMAL ||
                                              status == dual_simplex::lp_status_t::UNBOUNDED ||
                                              status == dual_simplex::lp_status_t::INFEASIBLE)) {
    // We finished. Tell PDLP to stop if it is still running.
    settings.concurrent_halt->store(1, std::memory_order_release);
  }

  return {std::move(solution), status, duration.count() / 1000.0, norm_user_objective, norm_rhs};
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> run_dual_simplex(
  detail::problem_t<i_t, f_t>& problem, pdlp_solver_settings_t<i_t, f_t> const& settings)
{
  // Convert data structures to dual simplex format and back
  dual_simplex::user_problem_t<i_t, f_t> dual_simplex_problem =
    cuopt_problem_to_simplex_problem<i_t, f_t>(problem);
  auto sol_dual_simplex = run_dual_simplex(dual_simplex_problem, settings);
  return convert_dual_simplex_sol(problem,
                                  std::get<0>(sol_dual_simplex),
                                  std::get<1>(sol_dual_simplex),
                                  std::get<2>(sol_dual_simplex),
                                  std::get<3>(sol_dual_simplex),
                                  std::get<4>(sol_dual_simplex));
}

template <typename i_t, typename f_t>
static optimization_problem_solution_t<i_t, f_t> run_pdlp_solver(
  detail::problem_t<i_t, f_t>& problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  const std::chrono::high_resolution_clock::time_point& start_time,
  bool is_batch_mode)
{
  if (problem.n_constraints == 0) {
    CUOPT_LOG_INFO("No constraints in the problem: PDLP can't be run, use Dual Simplex instead.");
    return optimization_problem_solution_t<i_t, f_t>{pdlp_termination_status_t::NumericalError,
                                                     problem.handle_ptr->get_stream()};
  }
  detail::pdlp_solver_t<i_t, f_t> solver(problem, settings, is_batch_mode);
  return solver.run_solver(start_time);
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> run_pdlp(detail::problem_t<i_t, f_t>& problem,
                                                   pdlp_solver_settings_t<i_t, f_t> const& settings,
                                                   bool is_batch_mode)
{
  auto start_solver = std::chrono::high_resolution_clock::now();
  f_t start_time    = dual_simplex::tic();
  auto sol          = run_pdlp_solver(problem, settings, start_solver, is_batch_mode);
  auto end          = std::chrono::high_resolution_clock::now();
  auto duration     = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_solver);
  sol.set_solve_time(duration.count() / 1000.0);
  CUOPT_LOG_INFO("PDLP finished");
  if (sol.get_termination_status() != pdlp_termination_status_t::ConcurrentLimit) {
    CUOPT_LOG_INFO("Status: %s   Objective: %.8e  Iterations: %d  Time: %.3fs",
                   sol.get_termination_status_string().c_str(),
                   sol.get_objective_value(),
                   sol.get_additional_termination_information().number_of_steps_taken,
                   sol.get_solve_time());
  }

  const bool do_crossover = settings.crossover;
  i_t crossover_info      = 0;
  if (do_crossover && sol.get_termination_status() == pdlp_termination_status_t::Optimal) {
    crossover_info = -1;

    dual_simplex::lp_problem_t<i_t, f_t> lp(1, 1, 1);
    dual_simplex::lp_solution_t<i_t, f_t> initial_solution(1, 1);
    translate_to_crossover_problem(problem, sol, lp, initial_solution);
    dual_simplex::simplex_solver_settings_t<i_t, f_t> dual_simplex_settings;
    dual_simplex_settings.time_limit      = settings.time_limit;
    dual_simplex_settings.iteration_limit = settings.iteration_limit;
    dual_simplex_settings.concurrent_halt = settings.concurrent_halt;
    dual_simplex::lp_solution_t<i_t, f_t> vertex_solution(lp.num_rows, lp.num_cols);
    std::vector<dual_simplex::variable_status_t> vstatus(lp.num_cols);
    dual_simplex::crossover_status_t crossover_status = dual_simplex::crossover(
      lp, dual_simplex_settings, initial_solution, start_time, vertex_solution, vstatus);
    pdlp_termination_status_t termination_status = pdlp_termination_status_t::TimeLimit;
    auto to_termination_status                   = [](dual_simplex::crossover_status_t status) {
      switch (status) {
        case dual_simplex::crossover_status_t::OPTIMAL: return pdlp_termination_status_t::Optimal;
        case dual_simplex::crossover_status_t::PRIMAL_FEASIBLE:
          return pdlp_termination_status_t::PrimalFeasible;
        case dual_simplex::crossover_status_t::DUAL_FEASIBLE:
          return pdlp_termination_status_t::NumericalError;
        case dual_simplex::crossover_status_t::NUMERICAL_ISSUES:
          return pdlp_termination_status_t::NumericalError;
        case dual_simplex::crossover_status_t::CONCURRENT_LIMIT:
          return pdlp_termination_status_t::ConcurrentLimit;
        case dual_simplex::crossover_status_t::TIME_LIMIT:
          return pdlp_termination_status_t::TimeLimit;
        default: return pdlp_termination_status_t::NumericalError;
      }
    };
    termination_status = to_termination_status(crossover_status);
    if (crossover_status == dual_simplex::crossover_status_t::OPTIMAL) { crossover_info = 0; }
    rmm::device_uvector<f_t> final_primal_solution =
      cuopt::device_copy(vertex_solution.x, problem.handle_ptr->get_stream());
    rmm::device_uvector<f_t> final_dual_solution =
      cuopt::device_copy(vertex_solution.y, problem.handle_ptr->get_stream());
    rmm::device_uvector<f_t> final_reduced_cost =
      cuopt::device_copy(vertex_solution.z, problem.handle_ptr->get_stream());

    // Should be filled with more information from dual simplex
    typename optimization_problem_solution_t<i_t, f_t>::additional_termination_information_t info;
    info.primal_objective      = vertex_solution.user_objective;
    info.number_of_steps_taken = vertex_solution.iterations;
    auto crossover_end         = std::chrono::high_resolution_clock::now();
    auto crossover_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(crossover_end - start_solver);
    info.solve_time    = crossover_duration.count() / 1000.0;
    auto sol_crossover = optimization_problem_solution_t<i_t, f_t>(final_primal_solution,
                                                                   final_dual_solution,
                                                                   final_reduced_cost,
                                                                   problem.objective_name,
                                                                   problem.var_names,
                                                                   problem.row_names,
                                                                   info,
                                                                   termination_status);
    sol.copy_from(problem.handle_ptr, sol_crossover);
    CUOPT_LOG_INFO("Crossover status %s", sol.get_termination_status_string().c_str());
  }
  if (settings.concurrent_halt != nullptr && crossover_info == 0 &&
      sol.get_termination_status() == pdlp_termination_status_t::Optimal) {
    // We finished. Tell dual simplex to stop if it is still running.
    settings.concurrent_halt->store(1, std::memory_order_release);
  }
  return sol;
}

template <typename i_t, typename f_t>
void run_dual_simplex_thread(
  dual_simplex::user_problem_t<i_t, f_t>& problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  std::unique_ptr<
    std::tuple<dual_simplex::lp_solution_t<i_t, f_t>, dual_simplex::lp_status_t, f_t, f_t, f_t>>&
    sol_ptr)
{
  // We will return the solution from the thread as a unique_ptr
  sol_ptr = std::make_unique<
    std::tuple<dual_simplex::lp_solution_t<i_t, f_t>, dual_simplex::lp_status_t, f_t, f_t, f_t>>(
    run_dual_simplex(problem, settings));
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> run_concurrent(
  const optimization_problem_t<i_t, f_t>& op_problem,
  detail::problem_t<i_t, f_t>& problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  bool is_batch_mode)
{
  CUOPT_LOG_INFO("Running concurrent\n");
  f_t start_time = dual_simplex::tic();

  // Copy the settings so that we can set the concurrent halt pointer
  pdlp_solver_settings_t<i_t, f_t> settings_pdlp(settings,
                                                 op_problem.get_handle_ptr()->get_stream());

  // Set the concurrent halt pointer
  global_concurrent_halt.store(0, std::memory_order_release);
  settings_pdlp.concurrent_halt = &global_concurrent_halt;

  // Initialize the dual simplex structures before we run PDLP.
  // Otherwise, CUDA API calls to the problem stream may occur in both threads and throw graph
  // capture off
  dual_simplex::user_problem_t<i_t, f_t> dual_simplex_problem =
    cuopt_problem_to_simplex_problem<i_t, f_t>(problem);
  // Create a thread for dual simplex
  std::unique_ptr<
    std::tuple<dual_simplex::lp_solution_t<i_t, f_t>, dual_simplex::lp_status_t, f_t, f_t, f_t>>
    sol_dual_simplex_ptr;
  std::thread dual_simplex_thread(run_dual_simplex_thread<i_t, f_t>,
                                  std::ref(dual_simplex_problem),
                                  std::ref(settings_pdlp),
                                  std::ref(sol_dual_simplex_ptr));

  // Run pdlp in the main thread
  auto sol_pdlp = run_pdlp(problem, settings_pdlp, is_batch_mode);

  // Wait for dual simplex thread to finish
  dual_simplex_thread.join();

  // copy the dual simplex solution to the device
  auto sol_dual_simplex = convert_dual_simplex_sol(problem,
                                                   std::get<0>(*sol_dual_simplex_ptr),
                                                   std::get<1>(*sol_dual_simplex_ptr),
                                                   std::get<2>(*sol_dual_simplex_ptr),
                                                   std::get<3>(*sol_dual_simplex_ptr),
                                                   std::get<4>(*sol_dual_simplex_ptr));

  f_t end_time = dual_simplex::toc(start_time);
  CUOPT_LOG_INFO("Concurrent time:  %.3fs", end_time);
  // Check status to see if we should return the pdlp solution or the dual simplex solution
  if (sol_dual_simplex.get_termination_status() == pdlp_termination_status_t::Optimal ||
      sol_dual_simplex.get_termination_status() == pdlp_termination_status_t::PrimalInfeasible ||
      sol_dual_simplex.get_termination_status() == pdlp_termination_status_t::DualInfeasible) {
    CUOPT_LOG_INFO("Solved with dual simplex");
    sol_pdlp.copy_from(op_problem.get_handle_ptr(), sol_dual_simplex);
    sol_pdlp.set_solve_time(end_time);
    CUOPT_LOG_INFO("Status: %s   Objective: %.8e  Iterations: %d  Time: %.3fs",
                   sol_pdlp.get_termination_status_string().c_str(),
                   sol_pdlp.get_objective_value(),
                   sol_pdlp.get_additional_termination_information().number_of_steps_taken,
                   end_time);
    return sol_pdlp;
  } else if (sol_pdlp.get_termination_status() == pdlp_termination_status_t::Optimal) {
    CUOPT_LOG_INFO("Solved with PDLP");
    return sol_pdlp;
  } else if (sol_pdlp.get_termination_status() == pdlp_termination_status_t::ConcurrentLimit) {
    CUOPT_LOG_INFO("Using dual simplex solve info");
    return sol_dual_simplex;
  } else {
    CUOPT_LOG_INFO("Using PDLP solve info");
    return sol_pdlp;
  }
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp_with_method(
  const optimization_problem_t<i_t, f_t>& op_problem,
  detail::problem_t<i_t, f_t>& problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  bool is_batch_mode)
{
  if (settings.method == method_t::DualSimplex) {
    return run_dual_simplex(problem, settings);
  } else if (settings.method == method_t::Concurrent) {
    return run_concurrent(op_problem, problem, settings, is_batch_mode);
  } else {
    return run_pdlp(problem, settings, is_batch_mode);
  }
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp(optimization_problem_t<i_t, f_t>& op_problem,
                                                   pdlp_solver_settings_t<i_t, f_t> const& settings,
                                                   bool problem_checking,
                                                   bool use_pdlp_solver_mode,
                                                   bool is_batch_mode)
{
  try {
    // Create log stream for file logging and add it to default logger
    init_logger_t log(settings.log_file, settings.log_to_console);

    // Init libraies before to not include it in solve time
    // This needs to be called before pdlp is initialized
    init_handler(op_problem.get_handle_ptr());

    raft::common::nvtx::range fun_scope("Running solver");

    if (problem_checking) {
      raft::common::nvtx::range fun_scope("Check problem representation");
      // This is required as user might forget to set some fields
      problem_checking_t<i_t, f_t>::check_problem_representation(op_problem);
      problem_checking_t<i_t, f_t>::check_initial_solution_representation(op_problem, settings);
    }
    detail::problem_t<i_t, f_t> problem(op_problem);
    CUOPT_LOG_INFO(
      "Solving a problem with %d constraints %d variables (%d integers) and %d nonzeros",
      problem.n_constraints,
      problem.n_variables,
      problem.n_integer_vars,
      problem.nnz);
    CUOPT_LOG_INFO("Objective offset %f scaling_factor %f",
                   problem.presolve_data.objective_offset,
                   problem.presolve_data.objective_scaling_factor);

    if (settings.user_problem_file != "") {
      CUOPT_LOG_INFO("Writing user problem to file: %s", settings.user_problem_file.c_str());
      problem.write_as_mps(settings.user_problem_file);
    }

    // Set the hyper-parameters based on the solver_settings
    if (use_pdlp_solver_mode) { set_pdlp_solver_mode(settings); }

    setup_device_symbols(op_problem.get_handle_ptr()->get_stream());

    auto sol = solve_lp_with_method(op_problem, problem, settings, is_batch_mode);

    if (settings.sol_file != "") {
      CUOPT_LOG_INFO("Writing solution to file %s", settings.sol_file.c_str());
      sol.write_to_sol_file(settings.sol_file, op_problem.get_handle_ptr()->get_stream());
    }

    return sol;
  } catch (const cuopt::logic_error& e) {
    CUOPT_LOG_ERROR("Error in solve_lp: %s", e.what());
    return optimization_problem_solution_t<i_t, f_t>{e, op_problem.get_handle_ptr()->get_stream()};
  } catch (const std::bad_alloc& e) {
    CUOPT_LOG_ERROR("Error in solve_lp: %s", e.what());
    return optimization_problem_solution_t<i_t, f_t>{
      cuopt::logic_error("Memory allocation failed", cuopt::error_type_t::RuntimeError),
      op_problem.get_handle_ptr()->get_stream()};
  }
}

template <typename i_t, typename f_t>
cuopt::linear_programming::optimization_problem_t<i_t, f_t> mps_data_model_to_optimization_problem(
  raft::handle_t const* handle_ptr, const cuopt::mps_parser::mps_data_model_t<i_t, f_t>& data_model)
{
  cuopt::linear_programming::optimization_problem_t<i_t, f_t> op_problem(handle_ptr);
  op_problem.set_maximize(data_model.get_sense());

  op_problem.set_csr_constraint_matrix(data_model.get_constraint_matrix_values().data(),
                                       data_model.get_constraint_matrix_values().size(),
                                       data_model.get_constraint_matrix_indices().data(),
                                       data_model.get_constraint_matrix_indices().size(),
                                       data_model.get_constraint_matrix_offsets().data(),
                                       data_model.get_constraint_matrix_offsets().size());

  if (data_model.get_constraint_bounds().size() != 0) {
    op_problem.set_constraint_bounds(data_model.get_constraint_bounds().data(),
                                     data_model.get_constraint_bounds().size());
  }
  if (data_model.get_objective_coefficients().size() != 0) {
    op_problem.set_objective_coefficients(data_model.get_objective_coefficients().data(),
                                          data_model.get_objective_coefficients().size());
  }
  op_problem.set_objective_scaling_factor(data_model.get_objective_scaling_factor());
  op_problem.set_objective_offset(data_model.get_objective_offset());
  if (data_model.get_variable_lower_bounds().size() != 0) {
    op_problem.set_variable_lower_bounds(data_model.get_variable_lower_bounds().data(),
                                         data_model.get_variable_lower_bounds().size());
  }
  if (data_model.get_variable_upper_bounds().size() != 0) {
    op_problem.set_variable_upper_bounds(data_model.get_variable_upper_bounds().data(),
                                         data_model.get_variable_upper_bounds().size());
  }
  if (data_model.get_variable_types().size() != 0) {
    std::vector<var_t> enum_variable_types(data_model.get_variable_types().size());
    std::transform(
      data_model.get_variable_types().cbegin(),
      data_model.get_variable_types().cend(),
      enum_variable_types.begin(),
      [](const auto val) -> var_t { return val == 'I' ? var_t::INTEGER : var_t::CONTINUOUS; });
    op_problem.set_variable_types(enum_variable_types.data(), enum_variable_types.size());
  }

  if (data_model.get_row_types().size() != 0) {
    op_problem.set_row_types(data_model.get_row_types().data(), data_model.get_row_types().size());
  }
  if (data_model.get_constraint_lower_bounds().size() != 0) {
    op_problem.set_constraint_lower_bounds(data_model.get_constraint_lower_bounds().data(),
                                           data_model.get_constraint_lower_bounds().size());
  }
  if (data_model.get_constraint_upper_bounds().size() != 0) {
    op_problem.set_constraint_upper_bounds(data_model.get_constraint_upper_bounds().data(),
                                           data_model.get_constraint_upper_bounds().size());
  }

  if (data_model.get_objective_name().size() != 0) {
    op_problem.set_objective_name(data_model.get_objective_name());
  }
  if (data_model.get_problem_name().size() != 0) {
    op_problem.set_problem_name(data_model.get_problem_name().data());
  }
  if (data_model.get_variable_names().size() != 0) {
    op_problem.set_variable_names(data_model.get_variable_names());
  }
  if (data_model.get_row_names().size() != 0) {
    op_problem.set_row_names(data_model.get_row_names());
  }

  return op_problem;
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp(
  raft::handle_t const* handle_ptr,
  const cuopt::mps_parser::mps_data_model_t<i_t, f_t>& mps_data_model,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  bool problem_checking,
  bool use_pdlp_solver_mode)
{
  auto op_problem = mps_data_model_to_optimization_problem(handle_ptr, mps_data_model);
  return solve_lp(op_problem, settings, problem_checking, use_pdlp_solver_mode);
}

#define INSTANTIATE(F_TYPE)                                                            \
  template optimization_problem_solution_t<int, F_TYPE> solve_lp(                      \
    optimization_problem_t<int, F_TYPE>& op_problem,                                   \
    pdlp_solver_settings_t<int, F_TYPE> const& settings,                               \
    bool problem_checking,                                                             \
    bool use_pdlp_solver_mode,                                                         \
    bool is_batch_mode);                                                               \
                                                                                       \
  template optimization_problem_solution_t<int, F_TYPE> solve_lp(                      \
    raft::handle_t const* handle_ptr,                                                  \
    const cuopt::mps_parser::mps_data_model_t<int, F_TYPE>& mps_data_model,            \
    pdlp_solver_settings_t<int, F_TYPE> const& settings,                               \
    bool problem_checking,                                                             \
    bool use_pdlp_solver_mode);                                                        \
                                                                                       \
  template optimization_problem_solution_t<int, F_TYPE> solve_lp_with_method(          \
    const optimization_problem_t<int, F_TYPE>& op_problem,                             \
    detail::problem_t<int, F_TYPE>& problem,                                           \
    pdlp_solver_settings_t<int, F_TYPE> const& settings,                               \
    bool is_batch_mode = false);                                                       \
                                                                                       \
  template optimization_problem_t<int, F_TYPE> mps_data_model_to_optimization_problem( \
    raft::handle_t const* handle_ptr,                                                  \
    const cuopt::mps_parser::mps_data_model_t<int, F_TYPE>& data_model);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

}  // namespace cuopt::linear_programming
