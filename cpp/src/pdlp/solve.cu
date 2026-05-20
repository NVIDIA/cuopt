/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/solve_remote.hpp>
#include <pdlp/cusparse_view.hpp>
#include <pdlp/optimal_batch_size_handler/optimal_batch_size_handler.hpp>
#include <pdlp/pdlp.cuh>
#include <pdlp/pdlp_constants.hpp>
#include <pdlp/restart_strategy/pdlp_restart_strategy.cuh>
#include <pdlp/step_size_strategy/adaptive_step_size_strategy.hpp>
#include <pdlp/translate.hpp>
#include <pdlp/utilities/ping_pong_graph.cuh>
#include <pdlp/utilities/problem_checking.cuh>
#include <pdlp/utils.cuh>
#include <utilities/logger.hpp>

#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/presolve/third_party_presolve.hpp>
#include <mip_heuristics/presolve/trivial_presolve.cuh>
#include <mip_heuristics/solver.cuh>
#include <mip_heuristics/utilities/sort_csr.cuh>

#include <cuopt/linear_programming/backend_selection.hpp>
#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_solution.hpp>
#include <cuopt/linear_programming/optimization_problem_utils.hpp>
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/solve.hpp>

#include <cuopt/linear_programming/io/mps_data_model.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/omp_helpers.hpp>
#include <utilities/version_info.hpp>

#include <barrier/sparse_cholesky.cuh>

#include <dual_simplex/crossover.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <pdlp/utilities/problem_checking.cuh>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/core/device_setter.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>

#include <rmm/cuda_stream.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <set>
#include <tuple>

#define CUOPT_LOG_CONDITIONAL_INFO(condition, ...) \
  if ((condition)) { CUOPT_LOG_INFO(__VA_ARGS__); }

namespace cuopt::linear_programming {

template <typename From, typename To>
extern rmm::device_uvector<To> gpu_cast(const rmm::device_uvector<From>& src,
                                        rmm::cuda_stream_view stream);

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
static void set_Stable1(pdlp_hyper_params::pdlp_hyper_params_t& hyper_params)
{
  hyper_params.initial_step_size_scaling                                  = 1.6;
  hyper_params.default_l_inf_ruiz_iterations                              = 1;
  hyper_params.do_pock_chambolle_scaling                                  = true;
  hyper_params.do_ruiz_scaling                                            = true;
  hyper_params.default_alpha_pock_chambolle_rescaling                     = 1.3;
  hyper_params.default_artificial_restart_threshold                       = 0.5;
  hyper_params.compute_initial_step_size_before_scaling                   = false;
  hyper_params.compute_initial_primal_weight_before_scaling               = true;
  hyper_params.initial_primal_weight_c_scaling                            = 2.2;
  hyper_params.initial_primal_weight_b_scaling                            = 4.6;
  hyper_params.major_iteration                                            = 52;
  hyper_params.min_iteration_restart                                      = 0;
  hyper_params.restart_strategy                                           = 1;
  hyper_params.never_restart_to_average                                   = false;
  hyper_params.reduction_exponent                                         = 0.5;
  hyper_params.growth_exponent                                            = 0.9;
  hyper_params.primal_weight_update_smoothing                             = 0.3;
  hyper_params.sufficient_reduction_for_restart                           = 0.2;
  hyper_params.necessary_reduction_for_restart                            = 0.5;
  hyper_params.primal_importance                                          = 1.8;
  hyper_params.primal_distance_smoothing                                  = 0.6;
  hyper_params.dual_distance_smoothing                                    = 0.2;
  hyper_params.compute_last_restart_before_new_primal_weight              = false;
  hyper_params.artificial_restart_in_main_loop                            = false;
  hyper_params.rescale_for_restart                                        = false;
  hyper_params.update_primal_weight_on_initial_solution                   = false;
  hyper_params.update_step_size_on_initial_solution                       = false;
  hyper_params.handle_some_primal_gradients_on_finite_bounds_as_residuals = true;
  hyper_params.project_initial_primal                                     = false;
  hyper_params.use_adaptive_step_size_strategy                            = true;
  hyper_params.initial_step_size_max_singular_value                       = false;
  hyper_params.initial_primal_weight_combined_bounds                      = true;
  hyper_params.bound_objective_rescaling                                  = false;
  hyper_params.use_reflected_primal_dual                                  = false;
  hyper_params.use_fixed_point_error                                      = false;
  hyper_params.reflection_coefficient = 1.0;  // TODO test with other values
  hyper_params.use_conditional_major  = false;
}

// Even better general setting due to proper primal gradient handling for KKT restart and initial
// projection
static void set_Stable2(pdlp_hyper_params::pdlp_hyper_params_t& hyper_params)
{
  hyper_params.initial_step_size_scaling                                  = 1.0;
  hyper_params.default_l_inf_ruiz_iterations                              = 10;
  hyper_params.do_pock_chambolle_scaling                                  = true;
  hyper_params.do_ruiz_scaling                                            = true;
  hyper_params.default_alpha_pock_chambolle_rescaling                     = 1.0;
  hyper_params.default_artificial_restart_threshold                       = 0.36;
  hyper_params.compute_initial_step_size_before_scaling                   = false;
  hyper_params.compute_initial_primal_weight_before_scaling               = false;
  hyper_params.initial_primal_weight_c_scaling                            = 1.0;
  hyper_params.initial_primal_weight_b_scaling                            = 1.0;
  hyper_params.major_iteration                                            = 40;
  hyper_params.min_iteration_restart                                      = 10;
  hyper_params.restart_strategy                                           = 1;
  hyper_params.never_restart_to_average                                   = false;
  hyper_params.reduction_exponent                                         = 0.3;
  hyper_params.growth_exponent                                            = 0.6;
  hyper_params.primal_weight_update_smoothing                             = 0.5;
  hyper_params.sufficient_reduction_for_restart                           = 0.2;
  hyper_params.necessary_reduction_for_restart                            = 0.8;
  hyper_params.primal_importance                                          = 1.0;
  hyper_params.primal_distance_smoothing                                  = 0.5;
  hyper_params.dual_distance_smoothing                                    = 0.5;
  hyper_params.compute_last_restart_before_new_primal_weight              = true;
  hyper_params.artificial_restart_in_main_loop                            = false;
  hyper_params.rescale_for_restart                                        = true;
  hyper_params.update_primal_weight_on_initial_solution                   = false;
  hyper_params.update_step_size_on_initial_solution                       = false;
  hyper_params.handle_some_primal_gradients_on_finite_bounds_as_residuals = false;
  hyper_params.project_initial_primal                                     = true;
  hyper_params.use_adaptive_step_size_strategy                            = true;
  hyper_params.initial_step_size_max_singular_value                       = false;
  hyper_params.initial_primal_weight_combined_bounds                      = true;
  hyper_params.bound_objective_rescaling                                  = false;
  hyper_params.use_reflected_primal_dual                                  = false;
  hyper_params.use_fixed_point_error                                      = false;
  hyper_params.reflection_coefficient                                     = 1.0;
  hyper_params.use_conditional_major                                      = false;
}

/* 1 - 1 mapping of cuPDLPx(+) function from Haihao and al.
 * For more information please read:
 * @article{lu2025cupdlpx,
 *   title={cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for Linear Programming},
 *   author={Lu, Haihao and Peng, Zedong and Yang, Jinwen},
 *   journal={arXiv preprint arXiv:2507.14051},
 *   year={2025}
 * }
 *
 * @article{lu2024restarted,
 *   title={Restarted Halpern PDHG for linear programming},
 *   author={Lu, Haihao and Yang, Jinwen},
 *   journal={arXiv preprint arXiv:2407.16144},
 *   year={2024}
 * }
 */
static void set_Stable3(pdlp_hyper_params::pdlp_hyper_params_t& hyper_params)
{
  hyper_params.initial_step_size_scaling                = 1.0;
  hyper_params.default_l_inf_ruiz_iterations            = 10;
  hyper_params.do_pock_chambolle_scaling                = true;
  hyper_params.do_ruiz_scaling                          = true;
  hyper_params.default_alpha_pock_chambolle_rescaling   = 1.0;
  hyper_params.default_artificial_restart_threshold     = 0.36;
  hyper_params.compute_initial_step_size_before_scaling = false;
  hyper_params.compute_initial_primal_weight_before_scaling =
    true;  // TODO this is maybe why he disabled primal weight when bound rescaling is on, because
           // TODO try with false
  hyper_params.initial_primal_weight_c_scaling  = 1.0;
  hyper_params.initial_primal_weight_b_scaling  = 1.0;
  hyper_params.major_iteration                  = 200;  // TODO Try with something smaller
  hyper_params.min_iteration_restart            = 0;
  hyper_params.restart_strategy                 = 3;
  hyper_params.never_restart_to_average         = true;
  hyper_params.reduction_exponent               = 0.3;
  hyper_params.growth_exponent                  = 0.6;
  hyper_params.primal_weight_update_smoothing   = 0.5;
  hyper_params.sufficient_reduction_for_restart = 0.2;
  hyper_params.necessary_reduction_for_restart  = 0.8;
  hyper_params.primal_importance                = 1.0;
  hyper_params.primal_distance_smoothing        = 0.5;
  hyper_params.dual_distance_smoothing          = 0.5;
  hyper_params.compute_last_restart_before_new_primal_weight              = true;
  hyper_params.artificial_restart_in_main_loop                            = false;
  hyper_params.rescale_for_restart                                        = true;
  hyper_params.update_primal_weight_on_initial_solution                   = false;
  hyper_params.update_step_size_on_initial_solution                       = false;
  hyper_params.handle_some_primal_gradients_on_finite_bounds_as_residuals = false;
  hyper_params.project_initial_primal          = true;  // TODO I think he doesn't do it anymore
  hyper_params.use_adaptive_step_size_strategy = false;
  hyper_params.initial_step_size_max_singular_value  = true;
  hyper_params.initial_primal_weight_combined_bounds = false;
  hyper_params.bound_objective_rescaling             = true;
  hyper_params.use_reflected_primal_dual             = true;
  hyper_params.use_fixed_point_error                 = true;
  hyper_params.use_conditional_major                 = true;
}

// Legacy/Original/Initial PDLP settings
static void set_Methodical1(pdlp_hyper_params::pdlp_hyper_params_t& hyper_params)
{
  hyper_params.initial_step_size_scaling                                  = 1.0;
  hyper_params.default_l_inf_ruiz_iterations                              = 5;
  hyper_params.do_pock_chambolle_scaling                                  = true;
  hyper_params.do_ruiz_scaling                                            = true;
  hyper_params.default_alpha_pock_chambolle_rescaling                     = 1.0;
  hyper_params.default_artificial_restart_threshold                       = 0.5;
  hyper_params.compute_initial_step_size_before_scaling                   = false;
  hyper_params.compute_initial_primal_weight_before_scaling               = false;
  hyper_params.initial_primal_weight_c_scaling                            = 1.0;
  hyper_params.initial_primal_weight_b_scaling                            = 1.0;
  hyper_params.major_iteration                                            = 64;
  hyper_params.min_iteration_restart                                      = 0;
  hyper_params.restart_strategy                                           = 2;
  hyper_params.never_restart_to_average                                   = false;
  hyper_params.reduction_exponent                                         = 0.3;
  hyper_params.growth_exponent                                            = 0.6;
  hyper_params.primal_weight_update_smoothing                             = 0.5;
  hyper_params.sufficient_reduction_for_restart                           = 0.1;
  hyper_params.necessary_reduction_for_restart                            = 0.9;
  hyper_params.primal_importance                                          = 1.0;
  hyper_params.primal_distance_smoothing                                  = 0.5;
  hyper_params.dual_distance_smoothing                                    = 0.5;
  hyper_params.compute_last_restart_before_new_primal_weight              = true;
  hyper_params.artificial_restart_in_main_loop                            = false;
  hyper_params.rescale_for_restart                                        = false;
  hyper_params.update_primal_weight_on_initial_solution                   = false;
  hyper_params.update_step_size_on_initial_solution                       = false;
  hyper_params.handle_some_primal_gradients_on_finite_bounds_as_residuals = true;
  hyper_params.project_initial_primal                                     = false;
  hyper_params.use_adaptive_step_size_strategy                            = true;
  hyper_params.initial_step_size_max_singular_value                       = false;
  hyper_params.initial_primal_weight_combined_bounds                      = true;
  hyper_params.bound_objective_rescaling                                  = false;
  hyper_params.use_reflected_primal_dual                                  = false;
  hyper_params.use_fixed_point_error                                      = false;
  hyper_params.reflection_coefficient                                     = 1.0;
  hyper_params.use_conditional_major                                      = false;
}

// Can be extremly faster but usually leads to more divergence
// Used for the blog post results
static void set_Fast1(pdlp_hyper_params::pdlp_hyper_params_t& hyper_params)
{
  hyper_params.initial_step_size_scaling                                  = 0.8;
  hyper_params.default_l_inf_ruiz_iterations                              = 6;
  hyper_params.do_pock_chambolle_scaling                                  = true;
  hyper_params.do_ruiz_scaling                                            = false;
  hyper_params.default_alpha_pock_chambolle_rescaling                     = 2.0;
  hyper_params.default_artificial_restart_threshold                       = 0.3;
  hyper_params.compute_initial_step_size_before_scaling                   = false;
  hyper_params.compute_initial_primal_weight_before_scaling               = true;
  hyper_params.initial_primal_weight_c_scaling                            = 1.2;
  hyper_params.initial_primal_weight_b_scaling                            = 1.2;
  hyper_params.major_iteration                                            = 76;
  hyper_params.min_iteration_restart                                      = 6;
  hyper_params.restart_strategy                                           = 1;
  hyper_params.never_restart_to_average                                   = true;
  hyper_params.reduction_exponent                                         = 0.4;
  hyper_params.growth_exponent                                            = 0.6;
  hyper_params.primal_weight_update_smoothing                             = 0.5;
  hyper_params.sufficient_reduction_for_restart                           = 0.3;
  hyper_params.necessary_reduction_for_restart                            = 0.9;
  hyper_params.primal_importance                                          = 0.8;
  hyper_params.primal_distance_smoothing                                  = 0.8;
  hyper_params.dual_distance_smoothing                                    = 0.3;
  hyper_params.compute_last_restart_before_new_primal_weight              = true;
  hyper_params.artificial_restart_in_main_loop                            = true;
  hyper_params.rescale_for_restart                                        = true;
  hyper_params.update_primal_weight_on_initial_solution                   = false;
  hyper_params.update_step_size_on_initial_solution                       = false;
  hyper_params.handle_some_primal_gradients_on_finite_bounds_as_residuals = true;
  hyper_params.project_initial_primal                                     = false;
  hyper_params.use_adaptive_step_size_strategy                            = true;
  hyper_params.initial_step_size_max_singular_value                       = false;
  hyper_params.initial_primal_weight_combined_bounds                      = true;
  hyper_params.bound_objective_rescaling                                  = false;
  hyper_params.use_reflected_primal_dual                                  = false;
  hyper_params.use_fixed_point_error                                      = false;
  hyper_params.reflection_coefficient                                     = 1.0;
  hyper_params.use_conditional_major                                      = false;
}

template <typename i_t, typename f_t>
void set_pdlp_solver_mode(pdlp_solver_settings_t<i_t, f_t>& settings)
{
  if (settings.pdlp_solver_mode == pdlp_solver_mode_t::Stable2)
    set_Stable2(settings.hyper_params);
  else if (settings.pdlp_solver_mode == pdlp_solver_mode_t::Stable1)
    set_Stable1(settings.hyper_params);
  else if (settings.pdlp_solver_mode == pdlp_solver_mode_t::Methodical1)
    set_Methodical1(settings.hyper_params);
  else if (settings.pdlp_solver_mode == pdlp_solver_mode_t::Fast1)
    set_Fast1(settings.hyper_params);
  else if (settings.pdlp_solver_mode == pdlp_solver_mode_t::Stable3)
    set_Stable3(settings.hyper_params);
}

std::atomic<int> global_concurrent_halt{0};

template <typename f_t>
void adjust_dual_solution_and_reduced_cost(rmm::device_uvector<f_t>& dual_solution,
                                           rmm::device_uvector<f_t>& reduced_cost,
                                           rmm::cuda_stream_view stream_view)
{
  // y <- -y
  cub::DeviceTransform::Transform(
    dual_solution.data(),
    dual_solution.data(),
    dual_solution.size(),
    [] HD(f_t dual) { return -dual; },
    stream_view);

  // z <- -z
  cub::DeviceTransform::Transform(
    reduced_cost.data(),
    reduced_cost.data(),
    reduced_cost.size(),
    [] HD(f_t reduced_cost) { return -reduced_cost; },
    stream_view);
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> convert_dual_simplex_sol(
  const dual_simplex::lp_solution_t<i_t, f_t>& solution,
  raft::handle_t const* handle_ptr,
  std::string const& objective_name,
  std::vector<std::string> const& var_names,
  std::vector<std::string> const& row_names,
  bool maximize,
  dual_simplex::lp_status_t status,
  f_t duration,
  f_t norm_user_objective,
  f_t norm_rhs,
  method_t method)
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
      case dual_simplex::lp_status_t::UNBOUNDED_OR_INFEASIBLE:
        return pdlp_termination_status_t::UnboundedOrInfeasible;
      default: return pdlp_termination_status_t::NumericalError;
    }
  };

  rmm::device_uvector<f_t> final_primal_solution =
    cuopt::device_copy(solution.x, handle_ptr->get_stream());
  rmm::device_uvector<f_t> final_dual_solution =
    cuopt::device_copy(solution.y, handle_ptr->get_stream());
  rmm::device_uvector<f_t> final_reduced_cost =
    cuopt::device_copy(solution.z, handle_ptr->get_stream());
  handle_ptr->sync_stream();

  // Negate dual variables and reduced costs for maximization problems
  if (maximize) {
    adjust_dual_solution_and_reduced_cost(
      final_dual_solution, final_reduced_cost, handle_ptr->get_stream());
    handle_ptr->sync_stream();
  }

  // Should be filled with more information from dual simplex
  std::vector<
    typename optimization_problem_solution_t<i_t, f_t>::additional_termination_information_t>
    info(1);
  info[0].solved_by                       = method;
  info[0].primal_objective                = solution.user_objective;
  info[0].dual_objective                  = solution.user_objective;
  info[0].gap                             = 0.0;
  info[0].relative_gap                    = 0.0;
  info[0].solve_time                      = duration;
  info[0].number_of_steps_taken           = solution.iterations;
  info[0].total_number_of_attempted_steps = solution.iterations;
  info[0].l2_primal_residual              = solution.l2_primal_residual;
  info[0].l2_dual_residual                = solution.l2_dual_residual;
  info[0].l2_relative_primal_residual  = solution.l2_primal_residual / (1.0 + norm_user_objective);
  info[0].l2_relative_dual_residual    = solution.l2_dual_residual / (1.0 + norm_rhs);
  info[0].max_primal_ray_infeasibility = 0.0;
  info[0].primal_ray_linear_objective  = 0.0;
  info[0].max_dual_ray_infeasibility   = 0.0;
  info[0].dual_ray_linear_objective    = 0.0;

  pdlp_termination_status_t termination_status = to_termination_status(status);
  auto sol = optimization_problem_solution_t<i_t, f_t>(final_primal_solution,
                                                       final_dual_solution,
                                                       final_reduced_cost,
                                                       objective_name,
                                                       var_names,
                                                       row_names,
                                                       std::move(info),
                                                       {termination_status});

  if (termination_status != pdlp_termination_status_t::Optimal &&
      termination_status != pdlp_termination_status_t::TimeLimit &&
      termination_status != pdlp_termination_status_t::ConcurrentLimit) {
    CUOPT_LOG_INFO("%s Solve status %s",
                   method == method_t::DualSimplex ? "Dual Simplex" : "Barrier",
                   sol.get_termination_status_string().c_str());
  }

  handle_ptr->sync_stream();
  return sol;
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> convert_dual_simplex_sol(
  detail::problem_t<i_t, f_t>& problem,
  const dual_simplex::lp_solution_t<i_t, f_t>& solution,
  dual_simplex::lp_status_t status,
  f_t duration,
  f_t norm_user_objective,
  f_t norm_rhs,
  method_t method)
{
  return convert_dual_simplex_sol(solution,
                                  problem.handle_ptr,
                                  problem.objective_name,
                                  problem.var_names,
                                  problem.row_names,
                                  problem.maximize,
                                  status,
                                  duration,
                                  norm_user_objective,
                                  norm_rhs,
                                  method);
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> convert_dual_simplex_sol(
  optimization_problem_t<i_t, f_t>& op_problem,
  const dual_simplex::lp_solution_t<i_t, f_t>& solution,
  dual_simplex::lp_status_t status,
  f_t duration,
  f_t norm_user_objective,
  f_t norm_rhs,
  method_t method)
{
  return convert_dual_simplex_sol(solution,
                                  op_problem.get_handle_ptr(),
                                  op_problem.get_objective_name(),
                                  op_problem.get_variable_names(),
                                  op_problem.get_row_names(),
                                  op_problem.get_sense(),
                                  status,
                                  duration,
                                  norm_user_objective,
                                  norm_rhs,
                                  method);
}

template <typename i_t, typename f_t>
std::tuple<dual_simplex::lp_solution_t<i_t, f_t>, dual_simplex::lp_status_t, f_t, f_t, f_t>
run_barrier(dual_simplex::user_problem_t<i_t, f_t>& user_problem,
            pdlp_solver_settings_t<i_t, f_t> const& settings,
            const timer_t& timer)
{
  f_t norm_user_objective = dual_simplex::vector_norm2<i_t, f_t>(user_problem.objective);
  f_t norm_rhs            = dual_simplex::vector_norm2<i_t, f_t>(user_problem.rhs);

  dual_simplex::simplex_solver_settings_t<i_t, f_t> barrier_settings;
  barrier_settings.num_gpus                        = settings.num_gpus;
  barrier_settings.time_limit                      = settings.time_limit;
  barrier_settings.iteration_limit                 = settings.iteration_limit;
  barrier_settings.concurrent_halt                 = settings.concurrent_halt;
  barrier_settings.folding                         = settings.folding;
  barrier_settings.augmented                       = settings.augmented;
  barrier_settings.dualize                         = settings.dualize;
  barrier_settings.ordering                        = settings.ordering;
  barrier_settings.barrier_dual_initial_point      = settings.barrier_dual_initial_point;
  barrier_settings.barrier                         = true;
  barrier_settings.crossover                       = settings.crossover;
  barrier_settings.eliminate_dense_columns         = settings.eliminate_dense_columns;
  barrier_settings.barrier_iterative_refinement    = settings.barrier_iterative_refinement;
  barrier_settings.barrier_step_scale              = settings.barrier_step_scale;
  barrier_settings.cudss_deterministic             = settings.cudss_deterministic;
  barrier_settings.barrier_relaxed_feasibility_tol = settings.tolerances.relative_primal_tolerance;
  barrier_settings.barrier_relaxed_optimality_tol  = settings.tolerances.relative_dual_tolerance;
  barrier_settings.barrier_relaxed_complementarity_tol = settings.tolerances.relative_gap_tolerance;
  if (barrier_settings.concurrent_halt != nullptr) {
    // Don't show the barrier log in concurrent mode. Show the PDLP log instead
    barrier_settings.log.log = false;
  }

  dual_simplex::lp_solution_t<i_t, f_t> solution(user_problem.num_rows, user_problem.num_cols);
  auto status = dual_simplex::solve_linear_program_with_barrier<i_t, f_t>(
    user_problem, barrier_settings, timer.get_tic_start(), solution);

  CUOPT_LOG_CONDITIONAL_INFO(
    !settings.inside_mip, "Barrier finished in %.2f seconds", timer.elapsed_time());

  if (settings.concurrent_halt != nullptr &&
      (status == dual_simplex::lp_status_t::OPTIMAL ||
       status == dual_simplex::lp_status_t::UNBOUNDED ||
       status == dual_simplex::lp_status_t::INFEASIBLE ||
       status == dual_simplex::lp_status_t::UNBOUNDED_OR_INFEASIBLE)) {
    // We finished. Tell PDLP to stop if it is still running.
    *settings.concurrent_halt = 1;
  }

  return {std::move(solution), status, timer.elapsed_time(), norm_user_objective, norm_rhs};
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> run_barrier(
  detail::problem_t<i_t, f_t>& problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  const timer_t& timer)
{
  // Convert data structures to dual simplex format and back
  dual_simplex::user_problem_t<i_t, f_t> dual_simplex_problem =
    cuopt_problem_to_user_problem<i_t, f_t>(problem.handle_ptr, problem);
  auto sol_dual_simplex = run_barrier(dual_simplex_problem, settings, timer);
  return convert_dual_simplex_sol(problem,
                                  std::get<0>(sol_dual_simplex),
                                  std::get<1>(sol_dual_simplex),
                                  std::get<2>(sol_dual_simplex),
                                  std::get<3>(sol_dual_simplex),
                                  std::get<4>(sol_dual_simplex),
                                  method_t::Barrier);
}

template <typename i_t, typename f_t>
void run_barrier_thread(
  dual_simplex::user_problem_t<i_t, f_t>& problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  std::unique_ptr<
    std::tuple<dual_simplex::lp_solution_t<i_t, f_t>, dual_simplex::lp_status_t, f_t, f_t, f_t>>&
    sol_ptr,
  const timer_t& timer)
{
  // We will return the solution from the thread as a unique_ptr
  sol_ptr = std::make_unique<
    std::tuple<dual_simplex::lp_solution_t<i_t, f_t>, dual_simplex::lp_status_t, f_t, f_t, f_t>>(
    run_barrier(problem, settings, timer));

  // Wait for barrier thread to finish
  problem.handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
std::tuple<dual_simplex::lp_solution_t<i_t, f_t>, dual_simplex::lp_status_t, f_t, f_t, f_t>
run_dual_simplex(dual_simplex::user_problem_t<i_t, f_t>& user_problem,
                 pdlp_solver_settings_t<i_t, f_t> const& settings,
                 const timer_t& timer)
{
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
  auto status = dual_simplex::solve_linear_program<i_t, f_t>(
    user_problem, dual_simplex_settings, timer.get_tic_start(), solution);

  CUOPT_LOG_CONDITIONAL_INFO(
    !settings.inside_mip, "Dual simplex finished in %.2f seconds", timer.elapsed_time());

  if (settings.concurrent_halt != nullptr &&
      (status == dual_simplex::lp_status_t::OPTIMAL ||
       status == dual_simplex::lp_status_t::UNBOUNDED ||
       status == dual_simplex::lp_status_t::INFEASIBLE ||
       status == dual_simplex::lp_status_t::UNBOUNDED_OR_INFEASIBLE)) {
    // We finished. Tell PDLP to stop if it is still running.
    *settings.concurrent_halt = 1;
  }

  return {std::move(solution), status, timer.elapsed_time(), norm_user_objective, norm_rhs};
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> run_dual_simplex(
  detail::problem_t<i_t, f_t>& problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  const timer_t& timer)
{
  // Convert data structures to dual simplex format and back
  dual_simplex::user_problem_t<i_t, f_t> dual_simplex_problem =
    cuopt_problem_to_user_problem<i_t, f_t>(problem.handle_ptr, problem);
  auto sol_dual_simplex = run_dual_simplex(dual_simplex_problem, settings, timer);
  return convert_dual_simplex_sol(problem,
                                  std::get<0>(sol_dual_simplex),
                                  std::get<1>(sol_dual_simplex),
                                  std::get<2>(sol_dual_simplex),
                                  std::get<3>(sol_dual_simplex),
                                  std::get<4>(sol_dual_simplex),
                                  method_t::DualSimplex);
}

#if PDLP_INSTANTIATE_FLOAT || CUOPT_INSTANTIATE_FLOAT

template <typename i_t>
static optimization_problem_solution_t<i_t, double> run_pdlp_solver_in_fp32(
  detail::problem_t<i_t, double>& problem,
  pdlp_solver_settings_t<i_t, double> const& settings,
  const timer_t& timer,
  bool is_batch_mode)
{
  CUOPT_LOG_CONDITIONAL_INFO(!settings.inside_mip, "Running PDLP in FP32 precision");
  auto stream = problem.handle_ptr->get_stream();

  // Convert the optimization problem stored inside problem_t to float
  auto float_op = problem.original_problem_ptr->template convert_to_other_prec<float>(stream);
  float_op.set_objective_offset(static_cast<float>(problem.presolve_data.objective_offset));
  float_op.set_objective_scaling_factor(
    static_cast<float>(problem.presolve_data.objective_scaling_factor));

  detail::problem_t<i_t, float> float_problem(float_op);

  auto objective_name = problem.objective_name;
  auto var_names      = problem.var_names;
  auto row_names      = problem.row_names;
  // When crossover is off, free double-precision GPU memory to reduce peak usage.
  // When crossover is on, run_pdlp needs the problem data after we return.
  if (!settings.crossover) {
    {
      [[maybe_unused]] auto discard = detail::problem_t<i_t, double>(std::move(problem));
    }
  }

  // Create float settings from double settings
  pdlp_solver_settings_t<i_t, float> fs;
  fs.tolerances.absolute_dual_tolerance =
    static_cast<float>(settings.tolerances.absolute_dual_tolerance);
  fs.tolerances.relative_dual_tolerance =
    static_cast<float>(settings.tolerances.relative_dual_tolerance);
  fs.tolerances.absolute_primal_tolerance =
    static_cast<float>(settings.tolerances.absolute_primal_tolerance);
  fs.tolerances.relative_primal_tolerance =
    static_cast<float>(settings.tolerances.relative_primal_tolerance);
  fs.tolerances.absolute_gap_tolerance =
    static_cast<float>(settings.tolerances.absolute_gap_tolerance);
  fs.tolerances.relative_gap_tolerance =
    static_cast<float>(settings.tolerances.relative_gap_tolerance);
  fs.tolerances.primal_infeasible_tolerance =
    static_cast<float>(settings.tolerances.primal_infeasible_tolerance);
  fs.tolerances.dual_infeasible_tolerance =
    static_cast<float>(settings.tolerances.dual_infeasible_tolerance);
  fs.detect_infeasibility         = settings.detect_infeasibility;
  fs.strict_infeasibility         = settings.strict_infeasibility;
  fs.iteration_limit              = settings.iteration_limit;
  fs.time_limit                   = static_cast<float>(settings.time_limit);
  fs.pdlp_solver_mode             = settings.pdlp_solver_mode;
  fs.log_to_console               = settings.log_to_console;
  fs.log_file                     = settings.log_file;
  fs.per_constraint_residual      = settings.per_constraint_residual;
  fs.save_best_primal_so_far      = settings.save_best_primal_so_far;
  fs.first_primal_feasible        = settings.first_primal_feasible;
  fs.all_primal_feasible          = settings.all_primal_feasible;
  fs.eliminate_dense_columns      = settings.eliminate_dense_columns;
  fs.barrier_iterative_refinement = settings.barrier_iterative_refinement;
  fs.barrier_step_scale           = settings.barrier_step_scale;
  fs.pdlp_precision               = pdlp_precision_t::DefaultPrecision;
  fs.method                       = method_t::PDLP;
  fs.inside_mip                   = settings.inside_mip;
  fs.hyper_params                 = settings.hyper_params;
  fs.presolver                    = settings.presolver;
  fs.num_gpus                     = settings.num_gpus;
  fs.concurrent_halt              = settings.concurrent_halt;

  detail::pdlp_solver_t<i_t, float> solver(float_problem, fs, is_batch_mode);
  if (settings.inside_mip) { solver.set_inside_mip(true); }
  auto float_sol = solver.run_solver(timer);

  // Convert float solution back to double on GPU (gpu_cast defined in optimization_problem.cu)
  auto dev_primal  = gpu_cast<float, double>(float_sol.get_primal_solution(), stream);
  auto dev_dual    = gpu_cast<float, double>(float_sol.get_dual_solution(), stream);
  auto dev_reduced = gpu_cast<float, double>(float_sol.get_reduced_cost(), stream);

  // Convert termination info (small host-side struct, stays on CPU)
  auto float_term_infos = float_sol.get_additional_termination_informations();
  using double_term_info_t =
    typename optimization_problem_solution_t<i_t, double>::additional_termination_information_t;
  std::vector<double_term_info_t> term_infos;
  for (auto& fi : float_term_infos) {
    double_term_info_t di;
    di.number_of_steps_taken           = fi.number_of_steps_taken;
    di.total_number_of_attempted_steps = fi.total_number_of_attempted_steps;
    di.l2_primal_residual              = static_cast<double>(fi.l2_primal_residual);
    di.l2_relative_primal_residual     = static_cast<double>(fi.l2_relative_primal_residual);
    di.l2_dual_residual                = static_cast<double>(fi.l2_dual_residual);
    di.l2_relative_dual_residual       = static_cast<double>(fi.l2_relative_dual_residual);
    di.primal_objective                = static_cast<double>(fi.primal_objective);
    di.dual_objective                  = static_cast<double>(fi.dual_objective);
    di.gap                             = static_cast<double>(fi.gap);
    di.relative_gap                    = static_cast<double>(fi.relative_gap);
    di.max_primal_ray_infeasibility    = static_cast<double>(fi.max_primal_ray_infeasibility);
    di.primal_ray_linear_objective     = static_cast<double>(fi.primal_ray_linear_objective);
    di.max_dual_ray_infeasibility      = static_cast<double>(fi.max_dual_ray_infeasibility);
    di.dual_ray_linear_objective       = static_cast<double>(fi.dual_ray_linear_objective);
    di.solve_time                      = fi.solve_time;
    di.solved_by                       = fi.solved_by;
    term_infos.push_back(di);
  }

  auto status_vec = float_sol.get_terminations_status();

  return optimization_problem_solution_t<i_t, double>(dev_primal,
                                                      dev_dual,
                                                      dev_reduced,
                                                      objective_name,
                                                      var_names,
                                                      row_names,
                                                      std::move(term_infos),
                                                      std::move(status_vec));
}
#endif

template <typename i_t, typename f_t>
static optimization_problem_solution_t<i_t, f_t> run_pdlp_solver(
  detail::problem_t<i_t, f_t>& problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  const timer_t& timer,
  bool is_batch_mode)
{
  if (problem.n_constraints == 0) {
    CUOPT_LOG_CONDITIONAL_INFO(
      !settings.inside_mip,
      "No constraints in the problem: PDLP can't be run, use Dual Simplex instead.");
    return optimization_problem_solution_t<i_t, f_t>{pdlp_termination_status_t::NumericalError,
                                                     problem.handle_ptr->get_stream()};
  }
#if PDLP_INSTANTIATE_FLOAT || CUOPT_INSTANTIATE_FLOAT
  if constexpr (std::is_same_v<f_t, double>) {
    if (settings.pdlp_precision == pdlp_precision_t::SinglePrecision) {
      return run_pdlp_solver_in_fp32(problem, settings, timer, is_batch_mode);
    }
  }
#endif
  detail::pdlp_solver_t<i_t, f_t> solver(problem, settings, is_batch_mode);
  if (settings.inside_mip) { solver.set_inside_mip(true); }
  return solver.run_solver(timer);
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> run_pdlp(detail::problem_t<i_t, f_t>& problem,
                                                   pdlp_solver_settings_t<i_t, f_t> const& settings,
                                                   const timer_t& timer,
                                                   bool is_batch_mode)
{
  if constexpr (!std::is_same_v<f_t, double>) {
    cuopt_expects(!is_batch_mode,
                  error_type_t::ValidationError,
                  "PDLP batch mode is not supported for float precision. Use double precision.");
  }
  cuopt_expects(!(settings.pdlp_precision == pdlp_precision_t::MixedPrecision &&
                  !detail::is_cusparse_runtime_mixed_precision_supported()),
                error_type_t::ValidationError,
                "Mixed-precision SpMV requires cuSPARSE runtime 12.5 or later.");
  cuopt_expects(
    !(is_batch_mode && settings.pdlp_precision == pdlp_precision_t::MixedPrecision),
    error_type_t::ValidationError,
    "Mixed-precision SpMV is not supported in batch mode. Set pdlp_precision=-1 (default) "
    "or disable batch mode.");
  cuopt_expects(!(settings.pdlp_precision == pdlp_precision_t::SinglePrecision && is_batch_mode),
                error_type_t::ValidationError,
                "Single-precision PDLP is not supported in batch mode.");

  auto start_solver = std::chrono::high_resolution_clock::now();
  timer_t timer_pdlp(timer.remaining_time());
  auto sol = run_pdlp_solver(problem, settings, timer, is_batch_mode);
  // Negate dual variables and reduced costs for maximization problems
  if (problem.maximize) {
    adjust_dual_solution_and_reduced_cost(
      sol.get_dual_solution(), sol.get_reduced_cost(), problem.handle_ptr->get_stream());
    problem.handle_ptr->sync_stream();
  }
  auto pdlp_solve_time = timer_pdlp.elapsed_time();
  sol.set_solve_time(timer.elapsed_time());
  CUOPT_LOG_CONDITIONAL_INFO(!settings.inside_mip, "PDLP finished");
  if (sol.get_termination_status() != pdlp_termination_status_t::ConcurrentLimit) {
    CUOPT_LOG_CONDITIONAL_INFO(!settings.inside_mip,
                               "Status: %s   Objective: %.8e  Iterations: %d  Time: %.3fs",
                               sol.get_termination_status_string().c_str(),
                               sol.get_objective_value(),
                               sol.get_additional_termination_information().number_of_steps_taken,
                               sol.get_solve_time());
  }

  if constexpr (std::is_same_v<f_t, double>) {
    const bool do_crossover = settings.crossover;
    i_t crossover_info      = 0;
    if (do_crossover && sol.get_termination_status() == pdlp_termination_status_t::Optimal) {
      crossover_info = -1;

      dual_simplex::lp_problem_t<i_t, f_t> lp(problem.handle_ptr, 1, 1, 1);
      dual_simplex::lp_solution_t<i_t, f_t> initial_solution(1, 1);
      translate_to_crossover_problem(problem, sol, lp, initial_solution);
      dual_simplex::simplex_solver_settings_t<i_t, f_t> dual_simplex_settings;
      dual_simplex_settings.time_limit      = settings.time_limit;
      dual_simplex_settings.iteration_limit = settings.iteration_limit;
      dual_simplex_settings.concurrent_halt = settings.concurrent_halt;
      dual_simplex::lp_solution_t<i_t, f_t> vertex_solution(lp.num_rows, lp.num_cols);
      std::vector<dual_simplex::variable_status_t> vstatus(lp.num_cols);
      dual_simplex::crossover_status_t crossover_status =
        dual_simplex::crossover(lp,
                                dual_simplex_settings,
                                initial_solution,
                                timer.get_tic_start(),
                                vertex_solution,
                                vstatus);
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
      problem.handle_ptr->sync_stream();
      // Negate dual variables and reduced costs for maximization problems
      if (problem.maximize) {
        adjust_dual_solution_and_reduced_cost(
          final_dual_solution, final_reduced_cost, problem.handle_ptr->get_stream());
        problem.handle_ptr->sync_stream();
      }

      // Should be filled with more information from dual simplex
      std::vector<
        typename optimization_problem_solution_t<i_t, f_t>::additional_termination_information_t>
        info(1);
      info[0].primal_objective      = vertex_solution.user_objective;
      info[0].number_of_steps_taken = vertex_solution.iterations;
      auto crossover_end            = std::chrono::high_resolution_clock::now();
      auto crossover_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(crossover_end - start_solver);
      info[0].solve_time = crossover_duration.count() / 1000.0;
      auto sol_crossover = optimization_problem_solution_t<i_t, f_t>(final_primal_solution,
                                                                     final_dual_solution,
                                                                     final_reduced_cost,
                                                                     problem.objective_name,
                                                                     problem.var_names,
                                                                     problem.row_names,
                                                                     std::move(info),
                                                                     {termination_status});
      sol.copy_from(problem.handle_ptr, sol_crossover);
      CUOPT_LOG_CONDITIONAL_INFO(
        !settings.inside_mip, "Crossover status %s", sol.get_termination_status_string().c_str());
    }
    if (settings.method == method_t::Concurrent && settings.concurrent_halt != nullptr &&
        crossover_info == 0 && sol.get_termination_status() == pdlp_termination_status_t::Optimal) {
      // We finished. Tell dual simplex to stop if it is still running.
      CUOPT_LOG_CONDITIONAL_INFO(!settings.inside_mip, "PDLP finished. Telling others to stop");
      *settings.concurrent_halt = 1;
    }
  }
  return sol;
}

// Compute in double as some cases overflow when using size_t
//
// `per_climber_objectives` / `per_climber_constraint_bounds` tell the estimator whether the caller
// will expand these fields to (trial_batch_size * n_{vars,constraints}).
template <typename i_t, typename f_t>
static double batch_pdlp_memory_estimator(const optimization_problem_t<i_t, f_t>& problem,
                                          double trial_batch_size,
                                          bool per_climber_objectives        = false,
                                          bool per_climber_constraint_bounds = false,
                                          bool collect_solutions             = false)
{
  double total_memory = 0.0;
  // In PDLP we store the scaled version of the problem which contains all of those
  total_memory += problem.get_constraint_matrix_indices().size() * sizeof(i_t);
  total_memory += problem.get_constraint_matrix_offsets().size() * sizeof(i_t);
  total_memory += problem.get_constraint_matrix_values().size() * sizeof(f_t);
  total_memory *= 2.0;  // To account for the A_t matrix

  // Internally we always use have a scaled and an unscaled version of the objective coefficients
  if (per_climber_objectives) {
    total_memory += 2.0 * trial_batch_size * problem.get_n_variables() * sizeof(f_t);
  } else {
    total_memory += 2.0 * problem.get_objective_coefficients().size() * sizeof(f_t);
  }

  total_memory += problem.get_constraint_bounds().size() * sizeof(f_t);
  total_memory += problem.get_variable_lower_bounds().size() * sizeof(f_t);
  total_memory += problem.get_variable_upper_bounds().size() * sizeof(f_t);

  // Per-climber constraint bounds expansion adds 2 * trial_batch_size * n_constraints. Strong
  // branching never expands these, so the flag guards the cost.
  // 2.0 because we have scaled and unscaled
  if (per_climber_constraint_bounds) {
    total_memory +=
      2.0 * trial_batch_size * problem.get_constraint_lower_bounds().size() * sizeof(f_t);
    total_memory +=
      2.0 * trial_batch_size * problem.get_constraint_upper_bounds().size() * sizeof(f_t);
  } else {
    total_memory += 2.0 * problem.get_constraint_lower_bounds().size() * sizeof(f_t);
    total_memory += 2.0 * problem.get_constraint_upper_bounds().size() * sizeof(f_t);
  }

  // Batch data estimator

  // Data from PDHG
  total_memory += trial_batch_size * problem.get_n_variables() * sizeof(f_t);
  total_memory += trial_batch_size * problem.get_n_constraints() * sizeof(f_t);
  total_memory += trial_batch_size * problem.get_n_variables() * sizeof(f_t);
  total_memory += trial_batch_size * problem.get_n_constraints() * sizeof(f_t);
  total_memory += trial_batch_size * problem.get_n_variables() * sizeof(f_t);
  total_memory += trial_batch_size * problem.get_n_variables() * sizeof(f_t);
  total_memory += trial_batch_size * problem.get_n_constraints() * sizeof(f_t);

  // Data from the saddle point state
  total_memory += trial_batch_size * problem.get_n_variables() * sizeof(f_t);
  total_memory += trial_batch_size * problem.get_n_constraints() * sizeof(f_t);
  total_memory += trial_batch_size * problem.get_n_variables() * sizeof(f_t);
  total_memory += trial_batch_size * problem.get_n_constraints() * sizeof(f_t);
  total_memory += trial_batch_size * problem.get_n_constraints() * sizeof(f_t);
  total_memory += trial_batch_size * problem.get_n_variables() * sizeof(f_t);
  total_memory += trial_batch_size * problem.get_n_variables() * sizeof(f_t);

  // Data for the convergeance information
  total_memory += trial_batch_size * problem.get_n_constraints() * sizeof(f_t);
  total_memory += trial_batch_size * problem.get_n_variables() * sizeof(f_t);
  total_memory += trial_batch_size * problem.get_n_variables() * sizeof(f_t);
  total_memory += trial_batch_size * problem.get_n_constraints() * sizeof(f_t);

  // Data for the localized duality gap container
  total_memory += trial_batch_size * problem.get_n_variables() * sizeof(f_t);
  total_memory += trial_batch_size * problem.get_n_constraints() * sizeof(f_t);

  // Data for the solution (only allocated when collect_solutions is true)
  if (collect_solutions) {
    total_memory += problem.get_n_variables() * trial_batch_size * sizeof(f_t);
    total_memory += problem.get_n_constraints() * trial_batch_size * sizeof(f_t);
    total_memory += problem.get_n_variables() * trial_batch_size * sizeof(f_t);
  }

  // Add a 70% overhead to make sure we have enough memory considering other parts of the solver may
  // need memory later while the batch PDLP is running
  total_memory *= 1.7;

  // Data from saddle point state
  return total_memory;
}

// We need to custom craft a solver settings for the batch mode as we need a specific set of values
// We override iteration limit and pdlp tolerance unless the user has specified otherwise
template <typename i_t, typename f_t>
static void apply_batch_settings_overrides(
  const pdlp_solver_settings_t<i_t, f_t>& original_settings,
  pdlp_solver_settings_t<i_t, f_t>& batch_settings)
{
  constexpr int batch_iteration_limit = 100000;
  constexpr f_t pdlp_tolerance        = 1e-4;

  const pdlp_solver_settings_t<i_t, f_t> default_settings{};

  auto override_or_keep_given =
    [&](const auto& given_value, const auto& default_value, const auto& override_value) {
      return given_value == default_value ? override_value : given_value;
    };

  batch_settings.method               = cuopt::linear_programming::method_t::PDLP;
  batch_settings.presolver            = presolver_t::None;
  batch_settings.pdlp_solver_mode     = pdlp_solver_mode_t::Stable3;
  batch_settings.detect_infeasibility = false;
  batch_settings.iteration_limit      = override_or_keep_given(
    original_settings.iteration_limit, default_settings.iteration_limit, batch_iteration_limit);
  batch_settings.inside_mip = true;
  // Override the tolerances unless the user has specified otherwise
  // Only risk is overriding a user intentionnaly wanting to use numeric_limits<f_t>::max() as an
  // iteration limit
  batch_settings.tolerances.absolute_dual_tolerance =
    override_or_keep_given(original_settings.tolerances.absolute_dual_tolerance,
                           default_settings.tolerances.absolute_dual_tolerance,
                           pdlp_tolerance);
  batch_settings.tolerances.relative_dual_tolerance =
    override_or_keep_given(original_settings.tolerances.relative_dual_tolerance,
                           default_settings.tolerances.relative_dual_tolerance,
                           pdlp_tolerance);
  batch_settings.tolerances.absolute_primal_tolerance =
    override_or_keep_given(original_settings.tolerances.absolute_primal_tolerance,
                           default_settings.tolerances.absolute_primal_tolerance,
                           pdlp_tolerance);
  batch_settings.tolerances.relative_primal_tolerance =
    override_or_keep_given(original_settings.tolerances.relative_primal_tolerance,
                           default_settings.tolerances.relative_primal_tolerance,
                           pdlp_tolerance);
  batch_settings.tolerances.absolute_gap_tolerance =
    override_or_keep_given(original_settings.tolerances.absolute_gap_tolerance,
                           default_settings.tolerances.absolute_gap_tolerance,
                           pdlp_tolerance);
  batch_settings.tolerances.relative_gap_tolerance =
    override_or_keep_given(original_settings.tolerances.relative_gap_tolerance,
                           default_settings.tolerances.relative_gap_tolerance,
                           pdlp_tolerance);

  constexpr bool pdlp_primal_dual_init       = true;
  constexpr bool primal_weight_init          = true;
  constexpr bool use_initial_pdlp_iterations = false;
  if (original_settings.has_initial_primal_solution() && pdlp_primal_dual_init) {
    batch_settings.set_initial_primal_solution(
      original_settings.get_initial_primal_solution().data(),
      original_settings.get_initial_primal_solution().size(),
      original_settings.get_initial_primal_solution().stream());
  }
  if (original_settings.has_initial_dual_solution() && pdlp_primal_dual_init) {
    batch_settings.set_initial_dual_solution(
      original_settings.get_initial_dual_solution().data(),
      original_settings.get_initial_dual_solution().size(),
      original_settings.get_initial_dual_solution().stream());
  }
  // Step size doesn't change anyways, just to save the compute
  if (original_settings.get_initial_step_size().has_value()) {
    batch_settings.set_initial_step_size(original_settings.get_initial_step_size().value());
  }
  if (original_settings.get_initial_primal_weight().has_value() && primal_weight_init) {
    batch_settings.set_initial_primal_weight(original_settings.get_initial_primal_weight().value());
  }
  if (original_settings.get_initial_pdlp_iteration().has_value() && use_initial_pdlp_iterations) {
    batch_settings.set_initial_pdlp_iteration(
      original_settings.get_initial_pdlp_iteration().value());
  }
}

// Fixed-path helper: caller pre-sized the batch via fixed_batch_size and pre-expanded any
// per-climber problem fields directly on the optimization_problem_t (objective_coefficients,
// constraint_lower_bounds, constraint_upper_bounds, batch_objective_offsets_). A single
// solve_lp call runs the batch — no memory heuristics, no sub-batching.
template <typename i_t, typename f_t>
static optimization_problem_solution_t<i_t, f_t> run_batch_pdlp_fixed(
  optimization_problem_t<i_t, f_t>& problem, pdlp_solver_settings_t<i_t, f_t> const& settings)
{
  cuopt_expects(settings.fixed_batch_size > 0,
                error_type_t::ValidationError,
                "run_batch_pdlp_fixed requires fixed_batch_size > 0");

  const size_t n_vars        = static_cast<size_t>(problem.get_n_variables());
  const size_t n_constraints = static_cast<size_t>(problem.get_n_constraints());
  const size_t bs            = static_cast<size_t>(settings.fixed_batch_size);

  const size_t obj_size = problem.get_objective_coefficients().size();
  const size_t clb_size = problem.get_constraint_lower_bounds().size();
  const size_t cub_size = problem.get_constraint_upper_bounds().size();
  const size_t off_size = problem.get_batch_objective_offsets().size();

  cuopt_expects(
    obj_size == n_vars || obj_size == bs * n_vars,
    error_type_t::ValidationError,
    "run_batch_pdlp fixed path: objective_coefficients size (%zu) must equal n_variables "
    "(%zu, shared across climbers) or fixed_batch_size * n_variables (%zu, per-climber).",
    obj_size,
    n_vars,
    bs * n_vars);

  cuopt_expects(
    clb_size == n_constraints || clb_size == bs * n_constraints,
    error_type_t::ValidationError,
    "run_batch_pdlp fixed path: constraint_lower_bounds size (%zu) must equal n_constraints "
    "(%zu, shared across climbers) or fixed_batch_size * n_constraints (%zu, per-climber).",
    clb_size,
    n_constraints,
    bs * n_constraints);

  cuopt_expects(
    cub_size == n_constraints || cub_size == bs * n_constraints,
    error_type_t::ValidationError,
    "run_batch_pdlp fixed path: constraint_upper_bounds size (%zu) must equal n_constraints "
    "(%zu, shared across climbers) or fixed_batch_size * n_constraints (%zu, per-climber).",
    cub_size,
    n_constraints,
    bs * n_constraints);

  // The lower/upper sweep in pdhg.cu (`if (constraint_lower_bounds.size() > dual_size_h_)`) keys
  // off the lower-bound array only and assumes the upper-bound array follows. Reject any layout
  // where one is shared and the other is per-climber.
  cuopt_expects(clb_size == cub_size,
                error_type_t::ValidationError,
                "run_batch_pdlp fixed path: constraint_lower_bounds (%zu) and "
                "constraint_upper_bounds (%zu) must have the same size (both shared or both "
                "per-climber).",
                clb_size,
                cub_size);

  cuopt_expects(off_size == 0 || off_size == bs,
                error_type_t::ValidationError,
                "run_batch_pdlp fixed path: batch_objective_offsets size (%zu) must be 0 (no "
                "per-climber offsets) or fixed_batch_size (%zu).",
                off_size,
                bs);

  pdlp_solver_settings_t<i_t, f_t> batch_settings = settings;
  apply_batch_settings_overrides(settings, batch_settings);

  return solve_lp(problem,
                  batch_settings,
                  /*problem_checking=*/false,
                  /*use_pdlp_solver_mode=*/true,
                  /*is_batch_mode=*/true);
}

template <typename i_t, typename f_t>
static void validate_new_bounds(const optimization_problem_t<i_t, f_t>& problem,
                                pdlp_solver_settings_t<i_t, f_t> const& settings)
{
  std::set<std::pair<i_t, i_t>> seen_bounds;
  i_t last_climber_id = -1;
  for (const auto& new_bound : settings.new_bounds) {
    const auto climber_id = std::get<0>(new_bound);
    const auto var_idx    = std::get<1>(new_bound);
    const auto lower      = std::get<2>(new_bound);
    const auto upper      = std::get<3>(new_bound);

    cuopt_expects(
      climber_id >= 0, error_type_t::ValidationError, "new_bounds climber_id must be non-negative");
    if (settings.fixed_batch_size > 0) {
      cuopt_expects(climber_id < settings.fixed_batch_size,
                    error_type_t::ValidationError,
                    "new_bounds climber_id must be less than fixed_batch_size");
    }
    if (climber_id != last_climber_id) {
      cuopt_expects(climber_id > last_climber_id,
                    error_type_t::ValidationError,
                    "new_bounds climber_id entries must be sorted ascending and grouped");
      last_climber_id = climber_id;
    }
    cuopt_expects(var_idx >= 0 && var_idx < problem.get_n_variables(),
                  error_type_t::ValidationError,
                  "new_bounds variable_index must be in [0, n_variables)");
    cuopt_expects(!std::isnan(lower) && !std::isnan(upper),
                  error_type_t::ValidationError,
                  "new_bounds lower and upper bounds must not be NaN");
    cuopt_expects(lower <= upper,
                  error_type_t::ValidationError,
                  "new_bounds lower bound must be less than or equal to upper bound");
    cuopt_expects(seen_bounds.insert({climber_id, var_idx}).second,
                  error_type_t::ValidationError,
                  "new_bounds cannot contain duplicate (climber_id, variable_index) entries");
  }
}

// Returns the batch size implied by per-climber variable-bound overrides.
template <typename i_t, typename f_t>
static size_t new_bounds_batch_size(const std::vector<std::tuple<i_t, i_t, f_t, f_t>>& new_bounds)
{
  cuopt_assert(!new_bounds.empty(), "Batch size should be greater than 0");
  i_t max_climber_id = 0;
  for (const auto& new_bound : new_bounds) {
    const auto climber_id = std::get<0>(new_bound);
    cuopt_assert(climber_id >= 0, "new_bounds climber_id must be non-negative");
    max_climber_id = std::max(max_climber_id, climber_id);
  }
  return static_cast<size_t>(max_climber_id) + 1;
}

template <typename i_t, typename f_t>
static void validate_splitting_new_bounds(
  const std::vector<std::tuple<i_t, i_t, f_t, f_t>>& new_bounds, size_t batch_size)
{
  cuopt_expects(new_bounds.size() == batch_size,
                error_type_t::ValidationError,
                "run_batch_pdlp splitting path requires exactly one new_bounds entry per climber");
  for (size_t i = 0; i < batch_size; ++i) {
    cuopt_expects(std::get<0>(new_bounds[i]) == static_cast<i_t>(i),
                  error_type_t::ValidationError,
                  "run_batch_pdlp splitting path requires new_bounds sorted by climber_id with no "
                  "missing climbers");
  }
}

template <typename i_t, typename f_t>
static size_t max_memory_batch_size(const optimization_problem_t<i_t, f_t>& problem,
                                    bool per_climber_objectives,
                                    bool per_climber_constraint_bounds,
                                    bool collect_solutions,
                                    size_t memory_max_batch_size)
{
  size_t st_free_mem, st_total_mem;
  RAFT_CUDA_TRY(cudaMemGetInfo(&st_free_mem, &st_total_mem));
  const double free_mem  = static_cast<double>(st_free_mem);
  const double total_mem = static_cast<double>(st_total_mem);

  while (memory_max_batch_size > 0) {
    const double mem_est = batch_pdlp_memory_estimator(problem,
                                                       memory_max_batch_size,
                                                       per_climber_objectives,
                                                       per_climber_constraint_bounds,
                                                       collect_solutions);
    if (mem_est <= free_mem) { break; }
#ifdef BATCH_VERBOSE_MODE
    std::cout << "Memory estimate: " << mem_est << std::endl;
    std::cout << "Memory max batch size: " << memory_max_batch_size << std::endl;
    std::cout << "Free memory: " << free_mem << std::endl;
    std::cout << "Total memory: " << total_mem << std::endl;
    std::cout << "--------------------------------" << std::endl;
#endif
    memory_max_batch_size--;
  }
  return memory_max_batch_size;
}

// Splitting-path helper: strong-branching flow.
// By default will try to run with the full batch size
// If the memory is too high, it will use the optimal batch size heuristic and split the batch into
// sub-batches
template <typename i_t, typename f_t>
static optimization_problem_solution_t<i_t, f_t> run_batch_pdlp_splitting(
  optimization_problem_t<i_t, f_t>& problem, pdlp_solver_settings_t<i_t, f_t> const& settings)
{
  rmm::cuda_stream_view stream = problem.get_handle_ptr()->get_stream();
  const i_t n_vars             = problem.get_n_variables();
  const i_t n_constraints      = problem.get_n_constraints();

  // Splitting path only supports un-expanded problems + per-climber variable-bound overrides.
  cuopt_expects(problem.get_objective_coefficients().size() == static_cast<size_t>(n_vars),
                error_type_t::ValidationError,
                "run_batch_pdlp splitting path requires un-expanded objective_coefficients "
                "(size == n_variables). Set fixed_batch_size and pre-expand on the "
                "optimization_problem_t to use the fixed path for per-climber problem data.");
  cuopt_expects(problem.get_constraint_lower_bounds().size() == static_cast<size_t>(n_constraints),
                error_type_t::ValidationError,
                "run_batch_pdlp splitting path requires un-expanded constraint_lower_bounds "
                "(size == n_constraints).");
  cuopt_expects(problem.get_constraint_upper_bounds().size() == static_cast<size_t>(n_constraints),
                error_type_t::ValidationError,
                "run_batch_pdlp splitting path requires un-expanded constraint_upper_bounds "
                "(size == n_constraints).");
  cuopt_expects(problem.get_batch_objective_offsets().size() == 0,
                error_type_t::ValidationError,
                "run_batch_pdlp splitting path does not support per-climber objective offsets. "
                "Use the fixed path (set fixed_batch_size) instead.");

  cuopt_assert(settings.new_bounds.size() > 0, "Batch size should be greater than 0");
  const size_t max_batch_size  = new_bounds_batch_size(settings.new_bounds);
  size_t memory_max_batch_size = max_batch_size;
  validate_splitting_new_bounds(settings.new_bounds, max_batch_size);

  const bool collect_solutions = settings.generate_batch_primal_dual_solution;
  // Strong branching never expands per-climber objectives or constraint bounds.
  const double memory_estimate =
    batch_pdlp_memory_estimator(problem,
                                max_batch_size,
                                /*per_climber_objectives=*/false,
                                /*per_climber_constraint_bounds=*/false,
                                collect_solutions);
  size_t st_free_mem, st_total_mem;
  RAFT_CUDA_TRY(cudaMemGetInfo(&st_free_mem, &st_total_mem));
  const double free_mem  = static_cast<double>(st_free_mem);
  const double total_mem = static_cast<double>(st_total_mem);

#ifdef BATCH_VERBOSE_MODE
  std::cout << "Memory estimate: " << memory_estimate << std::endl;
  std::cout << "Free memory: " << free_mem << std::endl;
  std::cout << "Total memory: " << total_mem << std::endl;
#endif

  bool use_optimal_batch_size = false;
  // If the memory estimate is too high, we need to use the optimal batch size heuristic
  if (memory_estimate > free_mem) {
    use_optimal_batch_size = true;
    memory_max_batch_size  = max_memory_batch_size(problem,
                                                  /*per_climber_objectives=*/false,
                                                  /*per_climber_constraint_bounds=*/false,
                                                  collect_solutions,
                                                  memory_max_batch_size);
    // Can't even fit one PDLP
    if (memory_max_batch_size == 0) {
      return optimization_problem_solution_t<i_t, f_t>(pdlp_termination_status_t::NumericalError,
                                                       stream);
    }
  }

  size_t optimal_batch_size = use_optimal_batch_size
                                ? detail::optimal_batch_size_handler(problem, memory_max_batch_size)
                                : max_batch_size;
  if (settings.fixed_batch_size > 0) { optimal_batch_size = settings.fixed_batch_size; }
  cuopt_assert(optimal_batch_size != 0 && optimal_batch_size <= max_batch_size,
               "Optimal batch size should be between 1 and max batch size");

  rmm::device_uvector<f_t> full_primal_solution(
    (collect_solutions) ? problem.get_n_variables() * max_batch_size : 0, stream);
  rmm::device_uvector<f_t> full_dual_solution(
    (collect_solutions) ? problem.get_n_constraints() * max_batch_size : 0, stream);
  rmm::device_uvector<f_t> full_reduced_cost(
    (collect_solutions) ? problem.get_n_variables() * max_batch_size : 0, stream);

  std::vector<
    typename optimization_problem_solution_t<i_t, f_t>::additional_termination_information_t>
    full_info;
  std::vector<pdlp_termination_status_t> full_status;

  pdlp_solver_settings_t<i_t, f_t> batch_settings = settings;
  const auto original_new_bounds                  = batch_settings.new_bounds;
  apply_batch_settings_overrides(settings, batch_settings);

  for (size_t i = 0; i < max_batch_size; i += optimal_batch_size) {
    const size_t current_batch_size = std::min(optimal_batch_size, max_batch_size - i);
    batch_settings.new_bounds.clear();
    for (size_t c = 0; c < current_batch_size; ++c) {
      const auto& new_bound = original_new_bounds[i + c];
      batch_settings.new_bounds.emplace_back(static_cast<i_t>(c),
                                             std::get<1>(new_bound),
                                             std::get<2>(new_bound),
                                             std::get<3>(new_bound));
    }

    if (!settings.shared_sb_solved.empty()) {
      batch_settings.shared_sb_solved = settings.shared_sb_solved.subspan(i, current_batch_size);
    }

    auto sol = solve_lp(problem,
                        batch_settings,
                        /*problem_checking=*/false,
                        /*use_pdlp_solver_mode=*/true,
                        /*is_batch_mode=*/true);

    // solve_lp swallows cuopt::logic_error and surfaces it via error_status on the returned
    // solution. If we kept aggregating, the final batched solution we build below would be
    // constructed without forwarding that error_status, silently dropping the error
    if (sol.get_error_status().get_error_type() != error_type_t::Success) { return sol; }

    if (collect_solutions) {
      raft::copy(full_primal_solution.data() + i * problem.get_n_variables(),
                 sol.get_primal_solution().data(),
                 sol.get_primal_solution().size(),
                 stream);
      raft::copy(full_dual_solution.data() + i * problem.get_n_constraints(),
                 sol.get_dual_solution().data(),
                 sol.get_dual_solution().size(),
                 stream);
      raft::copy(full_reduced_cost.data() + i * problem.get_n_variables(),
                 sol.get_reduced_cost().data(),
                 sol.get_reduced_cost().size(),
                 stream);
    }
    auto info = sol.get_additional_termination_informations();
    full_info.insert(full_info.end(), info.begin(), info.end());

    auto status = sol.get_terminations_status();
    full_status.insert(full_status.end(), status.begin(), status.end());
  }

  return optimization_problem_solution_t<i_t, f_t>(full_primal_solution,
                                                   full_dual_solution,
                                                   full_reduced_cost,
                                                   problem.get_objective_name(),
                                                   problem.get_variable_names(),
                                                   problem.get_row_names(),
                                                   std::move(full_info),
                                                   std::move(full_status));
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> run_batch_pdlp(
  optimization_problem_t<i_t, f_t>& problem, pdlp_solver_settings_t<i_t, f_t> const& settings)
{
  validate_new_bounds(problem, settings);

  // Fixed path: caller has pre-sized the batch (via fixed_batch_size) and pre-expanded any
  // per-climber problem fields directly on the optimization_problem_t. One solve_lp, no memory
  // heuristics.
  if (settings.fixed_batch_size > 0) { return run_batch_pdlp_fixed(problem, settings); }
  // Splitting path: strong-branching flow. Auto-picks batch size and sub-batches based on memory.
  return run_batch_pdlp_splitting(problem, settings);
}

// At this stage, the problem shouldn't already be expanded
// The results of this function should be used as the settings.fixed_batch_size, to expand the
// problem fields and call run_batch_pdlp
template <typename i_t, typename f_t>
size_t compute_optimal_batch_size(const optimization_problem_t<i_t, f_t>& problem,
                                  bool per_climber_objectives,
                                  bool per_climber_constraint_bounds,
                                  bool collect_solutions)
{
  // Find the maximum batch size that can be used without exceeding the free memory

  // Since we decerement iteratively, we don't want to use std::numeric_limits<size_t>::max()
  // Even if 20K fits in memory it will never be an optimal batch size,  it's just to have a
  // reasonable upper bound
  constexpr size_t max_batch_size    = 20000;
  const size_t memory_max_batch_size = max_memory_batch_size(problem,
                                                             per_climber_objectives,
                                                             per_climber_constraint_bounds,
                                                             collect_solutions,
                                                             max_batch_size);
#ifdef BATCH_VERBOSE_MODE
  std::cout << "Memory max batch size: " << memory_max_batch_size << std::endl;
#endif

  // We now know the maximum batch size that can be used without exceeding the free memory
  // Now find the optimal batch size [0, memory_max_batch_size]

  const size_t optimal_batch_size = static_cast<size_t>(
    detail::optimal_batch_size_handler(problem, static_cast<int>(memory_max_batch_size)));
#ifdef BATCH_VERBOSE_MODE
  std::cout << "Optimal batch size: " << optimal_batch_size << std::endl;
#endif
  return optimal_batch_size;
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> batch_pdlp_solve(
  raft::handle_t const* handle_ptr,
  const cuopt::linear_programming::io::mps_data_model_t<i_t, f_t>& mps_model,
  const std::vector<i_t>& fractional,
  const std::vector<f_t>& root_soln_x,
  pdlp_solver_settings_t<i_t, f_t> const& settings_const)
{
  cuopt_expects(fractional.size() == root_soln_x.size(),
                error_type_t::ValidationError,
                "Fractional and root solution must have the same size");
  cuopt_expects(settings_const.new_bounds.empty(),
                error_type_t::ValidationError,
                "Settings must not have new bounds");

  pdlp_solver_settings_t<i_t, f_t> settings(settings_const);

  // Lower bounds can sometimes generate infeasible instances that we struggle to detect
  constexpr bool only_upper = false;

  for (size_t i = 0; i < fractional.size(); ++i)
    settings.new_bounds.push_back({static_cast<i_t>(i),
                                   fractional[i],
                                   mps_model.get_variable_lower_bounds()[fractional[i]],
                                   std::floor(root_soln_x[i])});
  if (!only_upper) {
    for (size_t i = 0; i < fractional.size(); i++)
      settings.new_bounds.push_back({static_cast<i_t>(i + fractional.size()),
                                     fractional[i],
                                     std::ceil(root_soln_x[i]),
                                     mps_model.get_variable_upper_bounds()[fractional[i]]});
  }

  optimization_problem_t<i_t, f_t> op_problem =
    mps_data_model_to_optimization_problem(handle_ptr, mps_model);

  return run_batch_pdlp(op_problem, settings);
}

template <typename i_t, typename f_t>
void run_dual_simplex_thread(
  dual_simplex::user_problem_t<i_t, f_t>& problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  std::unique_ptr<
    std::tuple<dual_simplex::lp_solution_t<i_t, f_t>, dual_simplex::lp_status_t, f_t, f_t, f_t>>&
    sol_ptr,
  const timer_t& timer)
{
  // We will return the solution from the thread as a unique_ptr
  sol_ptr = std::make_unique<
    std::tuple<dual_simplex::lp_solution_t<i_t, f_t>, dual_simplex::lp_status_t, f_t, f_t, f_t>>(
    run_dual_simplex(problem, settings, timer));
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> run_concurrent(
  detail::problem_t<i_t, f_t>& problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  const timer_t& timer,
  bool is_batch_mode)
{
  CUOPT_LOG_CONDITIONAL_INFO(!settings.inside_mip, "Running concurrent (showing only PDLP log)\n");
  timer_t timer_concurrent(timer.remaining_time());

  // Copy the settings so that we can set the concurrent halt pointer
  pdlp_solver_settings_t<i_t, f_t> settings_pdlp(settings);

  // Use a local halt flag only when the caller did not provide one.
  if (settings_pdlp.concurrent_halt == nullptr) {
    global_concurrent_halt        = 0;
    settings_pdlp.concurrent_halt = &global_concurrent_halt;
  }

  // Make sure allocations are done on the original stream
  problem.handle_ptr->sync_stream();

  // Stand-alone LP always runs all three concurrently. MIP gates the barrier so we don't
  // overshoot num_cpu_threads (need 1 PDLP + 1 dual simplex + 1 barrier).
  const int available_threads = omp_in_parallel() ? omp_get_num_threads() : omp_get_max_threads();
  const bool enable_barrier =
    !settings.inside_mip || available_threads >= CUOPT_CONCURRENT_LP_BARRIER_REQUIRED_THREAD_COUNT;

  if (settings.num_gpus > 1) {
    int device_count = raft::device_setter::get_device_count();
    CUOPT_LOG_CONDITIONAL_INFO(!settings.inside_mip,
                               "Running PDLP%s on %d GPUs",
                               enable_barrier ? " and Barrier" : "",
                               device_count);
    cuopt_expects(
      device_count > 1, error_type_t::RuntimeError, "Multi-GPU mode requires at least 2 GPUs");
  }

  // Initialize the dual simplex structures before we run PDLP.
  // Otherwise, CUDA API calls to the problem stream may occur in both threads and throw graph
  // capture off
  dual_simplex::user_problem_t<i_t, f_t> dual_simplex_problem =
    cuopt_problem_to_user_problem<i_t, f_t>(problem.handle_ptr, problem);
  // Dual simplex / barrier results — written by tasks, read after the taskgroup barrier.
  std::unique_ptr<
    std::tuple<dual_simplex::lp_solution_t<i_t, f_t>, dual_simplex::lp_status_t, f_t, f_t, f_t>>
    sol_dual_simplex_ptr;
  std::exception_ptr dual_simplex_exception;
  auto request_concurrent_halt = [&settings_pdlp]() {
    if (settings_pdlp.concurrent_halt != nullptr) { settings_pdlp.concurrent_halt->store(1); }
  };
  // Owned at parent scope so its destructor runs on the dispatching thread after the taskgroup
  // joins every spawned task — cublasDestroy internally calls cudaDeviceSynchronize, which is
  // globally forbidden while any stream is in graph capture mode. Construction happens inside
  // the barrier task body below: capture invalidation caused by another thread's first-use
  // library init is now recovered by manual_cuda_graph_t::run, so the previous main-thread
  // preflight (eager handle construction + cuDSS warmup) is no longer needed.
  std::unique_ptr<raft::handle_t> barrier_handle_ptr;
  if (!enable_barrier) {
    CUOPT_LOG_DEBUG("MIP: skipping concurrent barrier, %d threads available < %d required.",
                    available_threads,
                    CUOPT_CONCURRENT_LP_BARRIER_REQUIRED_THREAD_COUNT);
  }

  // Dispatch barrier + dual simplex as OMP tasks (not std::threads) so they consume slots from
  // the upstream MIP OMP team and respect num_cpu_threads. PDLP runs synchronously on the
  // dispatching thread; the taskgroup implicit barrier joins the tasks.
  std::unique_ptr<
    std::tuple<dual_simplex::lp_solution_t<i_t, f_t>, dual_simplex::lp_status_t, f_t, f_t, f_t>>
    sol_barrier_ptr;
  std::exception_ptr barrier_exception;
  std::exception_ptr pdlp_exception;
  optimization_problem_solution_t<i_t, f_t> sol_pdlp{pdlp_termination_status_t::NumericalError,
                                                     problem.handle_ptr->get_stream()};

  auto dispatch_concurrent_solvers = [&]() {
#pragma omp taskgroup
    {
      // Barrier task — always on for stand-alone LP, gated on enable_barrier for MIP.
      if (enable_barrier) {
#pragma omp task default(shared)
        {
          try {
            auto call_barrier_thread = [&]() {
              rmm::cuda_stream_view barrier_stream = rmm::cuda_stream_per_thread;
              barrier_handle_ptr         = std::make_unique<raft::handle_t>(barrier_stream);
              auto barrier_problem       = dual_simplex_problem;
              barrier_problem.handle_ptr = barrier_handle_ptr.get();
              run_barrier_thread<i_t, f_t>(barrier_problem, settings_pdlp, sol_barrier_ptr, timer);
            };
            if (settings.num_gpus > 1) {
              problem.handle_ptr->sync_stream();
              raft::device_setter device_setter(1);  // Scoped variable
              CUOPT_LOG_DEBUG("Barrier device: %d", device_setter.get_current_device());
              call_barrier_thread();
            } else {
              call_barrier_thread();
            }
          } catch (...) {
            barrier_exception = std::current_exception();
            request_concurrent_halt();
          }
        }
      }

      // Dual simplex task — skipped from MIP (B&B already drives it separately).
      if (!settings.inside_mip) {
#pragma omp task default(shared)
        {
          try {
            run_dual_simplex_thread<i_t, f_t>(
              dual_simplex_problem, settings_pdlp, sol_dual_simplex_ptr, timer);
          } catch (...) {
            dual_simplex_exception = std::current_exception();
            request_concurrent_halt();
          }
        }
      }

      if (settings.num_gpus > 1) {
        CUOPT_LOG_DEBUG("PDLP device: %d", raft::device_setter::get_current_device());
      }

      // PDLP runs synchronously on the dispatcher, concurrently with the queued tasks.
      try {
        sol_pdlp = run_pdlp(problem, settings_pdlp, timer, is_batch_mode);
      } catch (...) {
        pdlp_exception = std::current_exception();
        request_concurrent_halt();
      }
      // Implicit taskgroup barrier joins all spawned tasks below.
    }
  };

  if (omp_in_parallel()) {
    // Reuse the upstream OMP team (e.g. solve_mip's outer parallel region).
    dispatch_concurrent_solvers();
  } else {
    // Stand-alone LP: stand up a local team sized for 1 dispatcher + 1 per spawned task.
    const int num_workers = 1 + (settings.inside_mip ? 0 : 1) + (enable_barrier ? 1 : 0);
#pragma omp parallel num_threads(num_workers) default(shared)
    {
#pragma omp single
      {
        dispatch_concurrent_solvers();
      }
    }
  }

  // Destroy on the dispatching thread, post-join: cublasDestroy → cudaDeviceSynchronize must
  // not fire during any graph capture.
  barrier_handle_ptr.reset();

  if (pdlp_exception) { std::rethrow_exception(pdlp_exception); }
  if (dual_simplex_exception) { std::rethrow_exception(dual_simplex_exception); }
  if (barrier_exception) { std::rethrow_exception(barrier_exception); }

  // copy the dual simplex solution to the device
  auto sol_dual_simplex =
    !settings.inside_mip
      ? convert_dual_simplex_sol(problem,
                                 std::get<0>(*sol_dual_simplex_ptr),
                                 std::get<1>(*sol_dual_simplex_ptr),
                                 std::get<2>(*sol_dual_simplex_ptr),
                                 std::get<3>(*sol_dual_simplex_ptr),
                                 std::get<4>(*sol_dual_simplex_ptr),
                                 method_t::DualSimplex)
      : optimization_problem_solution_t<i_t, f_t>{pdlp_termination_status_t::ConcurrentLimit,
                                                  problem.handle_ptr->get_stream()};

  // copy the barrier solution to the device (sentinel when the barrier task was skipped).
  auto sol_barrier = enable_barrier ? convert_dual_simplex_sol(problem,
                                                               std::get<0>(*sol_barrier_ptr),
                                                               std::get<1>(*sol_barrier_ptr),
                                                               std::get<2>(*sol_barrier_ptr),
                                                               std::get<3>(*sol_barrier_ptr),
                                                               std::get<4>(*sol_barrier_ptr),
                                                               method_t::Barrier)
                                    : optimization_problem_solution_t<i_t, f_t>{
                                        pdlp_termination_status_t::ConcurrentLimit,
                                        problem.handle_ptr->get_stream()};

  f_t end_time = timer.elapsed_time();
  CUOPT_LOG_CONDITIONAL_INFO(!settings.inside_mip, "Concurrent time: %.3fs", end_time);
  // Check status to see if we should return the pdlp solution or the dual simplex solution
  if (!settings.inside_mip &&
      (sol_dual_simplex.get_termination_status() == pdlp_termination_status_t::Optimal ||
       sol_dual_simplex.get_termination_status() == pdlp_termination_status_t::PrimalInfeasible ||
       sol_dual_simplex.get_termination_status() == pdlp_termination_status_t::DualInfeasible)) {
    CUOPT_LOG_CONDITIONAL_INFO(!settings.inside_mip, "Solved with dual simplex");
    sol_pdlp.copy_from(problem.handle_ptr, sol_dual_simplex);
    sol_pdlp.set_solve_time(end_time);
    CUOPT_LOG_CONDITIONAL_INFO(
      !settings.inside_mip,
      "Status: %s   Objective: %.8e  Iterations: %d  Time: %.3fs",
      sol_pdlp.get_termination_status_string().c_str(),
      sol_pdlp.get_objective_value(),
      sol_pdlp.get_additional_termination_information().number_of_steps_taken,
      end_time);
    CUOPT_LOG_CONDITIONAL_INFO(
      !settings.inside_mip,
      "Primal residual (abs/rel): %8.2e/%8.2e",
      sol_pdlp.get_additional_termination_information().l2_primal_residual,
      sol_pdlp.get_additional_termination_information().l2_relative_primal_residual);
    CUOPT_LOG_CONDITIONAL_INFO(
      !settings.inside_mip,
      "Dual   residual (abs/rel): %8.2e/%8.2e",
      sol_pdlp.get_additional_termination_information().l2_dual_residual,
      sol_pdlp.get_additional_termination_information().l2_relative_dual_residual);
    return sol_pdlp;
  } else if (sol_barrier.get_termination_status() == pdlp_termination_status_t::Optimal) {
    CUOPT_LOG_CONDITIONAL_INFO(!settings.inside_mip, "Solved with barrier");
    sol_pdlp.copy_from(problem.handle_ptr, sol_barrier);
    sol_pdlp.set_solve_time(end_time);
    CUOPT_LOG_CONDITIONAL_INFO(
      !settings.inside_mip,
      "Status: %s   Objective: %.8e  Iterations: %d  Time: %.3fs",
      sol_pdlp.get_termination_status_string().c_str(),
      sol_pdlp.get_objective_value(),
      sol_pdlp.get_additional_termination_information().number_of_steps_taken,
      end_time);
    CUOPT_LOG_CONDITIONAL_INFO(
      !settings.inside_mip,
      "Primal residual (abs/rel): %8.2e/%8.2e",
      sol_pdlp.get_additional_termination_information().l2_primal_residual,
      sol_pdlp.get_additional_termination_information().l2_relative_primal_residual);
    CUOPT_LOG_CONDITIONAL_INFO(
      !settings.inside_mip,
      "Dual   residual (abs/rel): %8.2e/%8.2e",
      sol_pdlp.get_additional_termination_information().l2_dual_residual,
      sol_pdlp.get_additional_termination_information().l2_relative_dual_residual);
    return sol_pdlp;
  } else if (sol_pdlp.get_termination_status() == pdlp_termination_status_t::Optimal) {
    CUOPT_LOG_CONDITIONAL_INFO(!settings.inside_mip, "Solved with PDLP");
    return sol_pdlp;
  } else if (!settings.inside_mip &&
             sol_pdlp.get_termination_status() == pdlp_termination_status_t::ConcurrentLimit) {
    CUOPT_LOG_CONDITIONAL_INFO(!settings.inside_mip, "Using dual simplex solve info");
    return sol_dual_simplex;
  } else {
    CUOPT_LOG_CONDITIONAL_INFO(!settings.inside_mip, "Using PDLP solve info");
    return sol_pdlp;
  }
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp_with_method(
  detail::problem_t<i_t, f_t>& problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  const timer_t& timer,
  bool is_batch_mode)
{
  if constexpr (std::is_same_v<f_t, double>) {
    if (settings.method == method_t::DualSimplex) {
      return run_dual_simplex(problem, settings, timer);
    } else if (settings.method == method_t::Barrier) {
      return run_barrier(problem, settings, timer);
    } else if (settings.method == method_t::Concurrent) {
      return run_concurrent(problem, settings, timer, is_batch_mode);
    } else {
      return run_pdlp(problem, settings, timer, is_batch_mode);
    }
  } else {
    // Float precision only supports PDLP without presolve/crossover
    cuopt_expects(settings.method == method_t::PDLP,
                  error_type_t::ValidationError,
                  "Float precision only supports PDLP method. DualSimplex, Barrier, and Concurrent "
                  "require double precision.");
    return run_pdlp(problem, settings, timer, is_batch_mode);
  }
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_qp(optimization_problem_t<i_t, f_t>& op_problem,
                                                   pdlp_solver_settings_t<i_t, f_t> const& settings,
                                                   bool problem_checking)
{
  try {
    print_version_info();
    // Create log stream for file logging and add it to default logger
    init_logger_t log(settings.log_file, settings.log_to_console);

    // Init libraries before to not include it in solve time
    init_handler(op_problem.get_handle_ptr());

    auto qp_timer = cuopt::timer_t(settings.time_limit);

    raft::common::nvtx::range fun_scope("Running QP solver");
    if (settings.user_problem_file != "") {
      CUOPT_LOG_INFO("Writing user problem to file: %s", settings.user_problem_file.c_str());
      op_problem.write_to_mps(settings.user_problem_file);
    }
    // Convert data structures to dual simplex format and back
    dual_simplex::user_problem_t<i_t, f_t> dual_simplex_problem =
      cuopt_optimization_problem_to_user_problem<i_t, f_t>(op_problem.get_handle_ptr(), op_problem);
    auto sol_dual_simplex = run_barrier(dual_simplex_problem, settings, qp_timer);
    auto solution         = convert_dual_simplex_sol(op_problem,
                                             std::get<0>(sol_dual_simplex),
                                             std::get<1>(sol_dual_simplex),
                                             std::get<2>(sol_dual_simplex),
                                             std::get<3>(sol_dual_simplex),
                                             std::get<4>(sol_dual_simplex),
                                             method_t::Barrier);
    if (settings.sol_file != "") {
      CUOPT_LOG_INFO("Writing solution to file %s", settings.sol_file.c_str());
      solution.write_to_sol_file(settings.sol_file, op_problem.get_handle_ptr()->get_stream());
    }
    return solution;
  } catch (const cuopt::logic_error& e) {
    CUOPT_LOG_ERROR("Error in solve_qp: %s", e.what());
    return optimization_problem_solution_t<i_t, f_t>{e, op_problem.get_handle_ptr()->get_stream()};
  } catch (const std::bad_alloc& e) {
    CUOPT_LOG_ERROR("Error in solve_qp: %s", e.what());
    return optimization_problem_solution_t<i_t, f_t>{
      cuopt::logic_error("Memory allocation failed", cuopt::error_type_t::RuntimeError),
      op_problem.get_handle_ptr()->get_stream()};
  }
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp(
  optimization_problem_t<i_t, f_t>& op_problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings_const,
  bool problem_checking,
  bool use_pdlp_solver_mode,
  bool is_batch_mode)
{
  if (op_problem.has_quadratic_objective()) {
    return solve_qp(op_problem, settings_const, problem_checking);
  }

  try {
    if (!settings_const.inside_mip) print_version_info();

    pdlp_solver_settings_t<i_t, f_t> settings(settings_const);
    // Create log stream for file logging and add it to default logger
    init_logger_t log(settings.log_file, settings.log_to_console);

    // Init libraries before to not include it in solve time
    // This needs to be called before pdlp is initialized
    init_handler(op_problem.get_handle_ptr());

    raft::common::nvtx::range fun_scope("Running solver");

    if (problem_checking) {
      raft::common::nvtx::range fun_scope("Check problem representation");
      // This is required as user might forget to set some fields
      problem_checking_t<i_t, f_t>::check_problem_representation(op_problem);
      // In batch PDLP for strong branching, the initial solutions will be by design out of bounds.
      // Batch mode also disables this check: fixed_batch_size > 0 means the caller has already
      // expanded per-climber fields on the problem, which would fail single-problem size checks.
      if (settings.new_bounds.size() == 0 && settings.fixed_batch_size == 0)
        problem_checking_t<i_t, f_t>::check_initial_solution_representation(op_problem, settings);
    }

    if (!settings_const.inside_mip) {
      CUOPT_LOG_INFO(
        "Solving a problem with %d constraints, %d variables (%d integers), and %d nonzeros",
        op_problem.get_n_constraints(),
        op_problem.get_n_variables(),
        0,
        op_problem.get_nnz());
      op_problem.print_scaling_information();
    }

    // Check for crossing bounds. Return infeasible if there are any
    if (problem_checking_t<i_t, f_t>::has_crossing_bounds(op_problem)) {
      return optimization_problem_solution_t<i_t, f_t>(pdlp_termination_status_t::PrimalInfeasible,
                                                       op_problem.get_handle_ptr()->get_stream());
    }
    validate_new_bounds(op_problem, settings);

    auto lp_timer = cuopt::timer_t(settings.time_limit);
    detail::problem_t<i_t, f_t> problem(op_problem);
    // handle default presolve
    if (settings.presolver == presolver_t::Default) {
      constexpr i_t presolve_nnz_threshold = 8000;
      const bool skip_presolve_for_small_dual_simplex =
        settings.method == method_t::DualSimplex && op_problem.get_nnz() < presolve_nnz_threshold;
      if (skip_presolve_for_small_dual_simplex) {
        // Skip presolve for small dual-simplex problems where the fixed overhead
        // (~20-30ms) exceeds the simplex solve time. Based on Netlib benchmarks,
        // problems with fewer than 8000 nonzeros never benefit from PSLP presolve.
        settings.presolver = presolver_t::None;
        CUOPT_LOG_INFO("Skipping presolve for small problem (nnz=%d < %d)",
                       op_problem.get_nnz(),
                       presolve_nnz_threshold);
      } else {
        settings.presolver = presolver_t::PSLP;
        CUOPT_LOG_INFO("Using PSLP presolver");
      }
    }

    [[maybe_unused]] double presolve_time = 0.0;
    std::unique_ptr<detail::third_party_presolve_t<i_t, f_t>> presolver;
    auto run_presolve = settings.presolver != presolver_t::None;
    run_presolve = run_presolve && settings.get_pdlp_warm_start_data().total_pdlp_iterations_ == -1;

    // Declare result at outer scope so that result.reduced_problem (which may be
    // referenced by problem.original_problem_ptr) remains alive through the solve.
    std::optional<detail::third_party_presolve_result_t<i_t, f_t>> result;

    if (run_presolve) {
      detail::sort_csr(op_problem);
      // allocate no more than 10% of the time limit to presolve.
      // Note that this is not the presolve time, but the time limit for presolve.
      // But no less than 1 second, to avoid early timeout triggering known crashes
      const double presolve_time_limit =
        std::max(1.0, std::min(0.1 * lp_timer.remaining_time(), 60.0));
      presolver = std::make_unique<detail::third_party_presolve_t<i_t, f_t>>();
      result    = presolver->apply(op_problem,
                                cuopt::linear_programming::problem_category_t::LP,
                                settings.presolver,
                                settings.dual_postsolve,
                                settings.tolerances.absolute_primal_tolerance,
                                settings.tolerances.relative_primal_tolerance,
                                presolve_time_limit);
      if (result->status == detail::third_party_presolve_status_t::INFEASIBLE) {
        return optimization_problem_solution_t<i_t, f_t>(
          pdlp_termination_status_t::PrimalInfeasible, op_problem.get_handle_ptr()->get_stream());
      }
      if (result->status == detail::third_party_presolve_status_t::UNBNDORINFEAS) {
        return optimization_problem_solution_t<i_t, f_t>(
          pdlp_termination_status_t::UnboundedOrInfeasible,
          op_problem.get_handle_ptr()->get_stream());
      }
      if (result->status == detail::third_party_presolve_status_t::UNBOUNDED) {
        return optimization_problem_solution_t<i_t, f_t>(pdlp_termination_status_t::DualInfeasible,
                                                         op_problem.get_handle_ptr()->get_stream());
      }

      // Handle case where presolve completely solved the problem (reduced to 0 rows/cols)
      // Must check before constructing problem_t since it fails on empty problems
      if (result->reduced_problem.get_n_variables() == 0 &&
          result->reduced_problem.get_n_constraints() == 0) {
        CUOPT_LOG_INFO("Presolve completely solved the problem");
        presolve_time = lp_timer.elapsed_time();
        CUOPT_LOG_INFO("%s presolve time: %.2fs",
                       settings.presolver == presolver_t::PSLP ? "PSLP" : "Papilo",
                       presolve_time);

        // Create empty solution vectors for the reduced problem
        rmm::device_uvector<f_t> empty_primal(0, op_problem.get_handle_ptr()->get_stream());
        rmm::device_uvector<f_t> empty_dual(0, op_problem.get_handle_ptr()->get_stream());
        rmm::device_uvector<f_t> empty_reduced_costs(0, op_problem.get_handle_ptr()->get_stream());

        // Run postsolve to get the full solution
        presolver->undo(empty_primal,
                        empty_dual,
                        empty_reduced_costs,
                        cuopt::linear_programming::problem_category_t::LP,
                        false,  // status_to_skip
                        settings.dual_postsolve,
                        op_problem.get_handle_ptr()->get_stream());

        // Create termination info with the objective from presolve
        typename optimization_problem_solution_t<i_t, f_t>::additional_termination_information_t
          term_info;
        term_info.primal_objective      = result->reduced_problem.get_objective_offset();
        term_info.dual_objective        = result->reduced_problem.get_objective_offset();
        term_info.number_of_steps_taken = 0;
        term_info.solve_time            = presolve_time;
        term_info.l2_primal_residual    = 0.0;
        term_info.l2_dual_residual      = 0.0;
        term_info.gap                   = 0.0;

        std::vector<
          typename optimization_problem_solution_t<i_t, f_t>::additional_termination_information_t>
          term_vec{term_info};
        std::vector<pdlp_termination_status_t> status_vec{pdlp_termination_status_t::Optimal};

        CUOPT_LOG_INFO("Status: Optimal  Objective: %f", term_info.primal_objective);
        return optimization_problem_solution_t<i_t, f_t>(empty_primal,
                                                         empty_dual,
                                                         empty_reduced_costs,
                                                         op_problem.get_objective_name(),
                                                         op_problem.get_variable_names(),
                                                         op_problem.get_row_names(),
                                                         std::move(term_vec),
                                                         std::move(status_vec));
      }

      problem       = detail::problem_t<i_t, f_t>(result->reduced_problem);
      presolve_time = lp_timer.elapsed_time();
      CUOPT_LOG_INFO("%s presolve time: %.2fs",
                     settings.presolver == presolver_t::PSLP ? "PSLP" : "Papilo",
                     presolve_time);
    }

    if (!settings_const.inside_mip) {
      CUOPT_LOG_INFO("Objective offset %f scaling_factor %f",
                     problem.presolve_data.objective_offset,
                     problem.presolve_data.objective_scaling_factor);
    }

    if (settings.user_problem_file != "") {
      CUOPT_LOG_INFO("Writing user problem to file: %s", settings.user_problem_file.c_str());
      op_problem.write_to_mps(settings.user_problem_file);
    }
    if (run_presolve && settings.presolve_file != "") {
      CUOPT_LOG_INFO("Writing presolved problem to file: %s", settings.presolve_file.c_str());
      result->reduced_problem.write_to_mps(settings.presolve_file);
    }

    // Set the hyper-parameters based on the solver_settings
    if (use_pdlp_solver_mode) { set_pdlp_solver_mode(settings); }

    auto solution = solve_lp_with_method(problem, settings, lp_timer, is_batch_mode);

    if (run_presolve) {
      auto primal_solution = cuopt::device_copy(solution.get_primal_solution(),
                                                op_problem.get_handle_ptr()->get_stream());
      auto dual_solution =
        cuopt::device_copy(solution.get_dual_solution(), op_problem.get_handle_ptr()->get_stream());
      auto reduced_costs =
        cuopt::device_copy(solution.get_reduced_cost(), op_problem.get_handle_ptr()->get_stream());
      bool status_to_skip = false;

      presolver->undo(primal_solution,
                      dual_solution,
                      reduced_costs,
                      cuopt::linear_programming::problem_category_t::LP,
                      status_to_skip,
                      settings.dual_postsolve,
                      op_problem.get_handle_ptr()->get_stream());

      std::vector<
        typename optimization_problem_solution_t<i_t, f_t>::additional_termination_information_t>
        term_vec = solution.get_additional_termination_informations();
      std::vector<pdlp_termination_status_t> status_vec = solution.get_terminations_status();

      // Create a new solution with the full problem solution
      solution =
        optimization_problem_solution_t<i_t, f_t>(primal_solution,
                                                  dual_solution,
                                                  reduced_costs,
                                                  std::move(solution.get_pdlp_warm_start_data()),
                                                  op_problem.get_objective_name(),
                                                  op_problem.get_variable_names(),
                                                  op_problem.get_row_names(),
                                                  std::move(term_vec),
                                                  std::move(status_vec));
    }

    if (settings.sol_file != "") {
      CUOPT_LOG_INFO("Writing solution to file %s", settings.sol_file.c_str());
      solution.write_to_sol_file(settings.sol_file, op_problem.get_handle_ptr()->get_stream());
    }

    return solution;
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
  raft::handle_t const* handle_ptr,
  const cuopt::linear_programming::io::mps_data_model_t<i_t, f_t>& data_model)
{
  cuopt_expects(handle_ptr != nullptr,
                error_type_t::ValidationError,
                "handle_ptr must not be null for GPU-backed problem construction");
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
    std::transform(data_model.get_variable_types().cbegin(),
                   data_model.get_variable_types().cend(),
                   enum_variable_types.begin(),
                   detail::char_to_var_type);
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
  auto problem_name = data_model.get_problem_name();
  op_problem.set_problem_name(problem_name);
  if (data_model.get_variable_names().size() != 0) {
    op_problem.set_variable_names(data_model.get_variable_names());
  }
  if (data_model.get_row_names().size() != 0) {
    op_problem.set_row_names(data_model.get_row_names());
  }

  if (data_model.get_quadratic_objective_values().size() != 0) {
    const std::vector<f_t> Q_values  = data_model.get_quadratic_objective_values();
    const std::vector<i_t> Q_indices = data_model.get_quadratic_objective_indices();
    const std::vector<i_t> Q_offsets = data_model.get_quadratic_objective_offsets();
    op_problem.set_quadratic_objective_matrix(Q_values.data(),
                                              Q_values.size(),
                                              Q_indices.data(),
                                              Q_indices.size(),
                                              Q_offsets.data(),
                                              Q_offsets.size());
  }

  return op_problem;
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp(
  raft::handle_t const* handle_ptr,
  const cuopt::linear_programming::io::mps_data_model_t<i_t, f_t>& mps_data_model,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  bool problem_checking,
  bool use_pdlp_solver_mode)
{
  auto op_problem = mps_data_model_to_optimization_problem(handle_ptr, mps_data_model);
  return solve_lp(op_problem, settings, problem_checking, use_pdlp_solver_mode);
}

// ============================================================================
// CPU problem overloads (convert to GPU, solve, convert solution back)
// ============================================================================

template <typename i_t, typename f_t>
std::unique_ptr<lp_solution_interface_t<i_t, f_t>> solve_lp(
  cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  bool problem_checking,
  bool use_pdlp_solver_mode,
  bool is_batch_mode)
{
  // Create CUDA resources for the conversion
  rmm::cuda_stream stream;
  raft::handle_t handle(stream);

  // Convert CPU problem to GPU problem
  auto gpu_problem = cpu_problem.to_optimization_problem(&handle);

  // Synchronize before solving to ensure conversion is complete
  stream.synchronize();

  // Solve on GPU
  auto gpu_solution = solve_lp<i_t, f_t>(
    *gpu_problem, settings, problem_checking, use_pdlp_solver_mode, is_batch_mode);

  // Ensure all GPU work from the solve is complete before D2H copies in to_cpu_solution(),
  // which uses rmm::cuda_stream_per_thread (a different stream than the solver used).
  stream.synchronize();

  // Convert GPU solution back to CPU
  gpu_lp_solution_t<i_t, f_t> gpu_sol_interface(std::move(gpu_solution));
  return gpu_sol_interface.to_cpu_solution();
}

// ============================================================================
// Interface-based solve overloads with remote execution support
// ============================================================================

template <typename i_t, typename f_t>
std::unique_ptr<lp_solution_interface_t<i_t, f_t>> solve_lp(
  optimization_problem_interface_t<i_t, f_t>* problem_interface,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  bool problem_checking,
  bool use_pdlp_solver_mode,
  bool is_batch_mode)
{
  cuopt_expects(problem_interface != nullptr,
                error_type_t::ValidationError,
                "problem_interface cannot be null");

  // Check if remote execution is enabled (always uses CPU backend)
#ifdef CUOPT_ENABLE_GRPC
  if (is_remote_execution_enabled()) {
    cuopt_expects(!is_batch_mode,
                  error_type_t::ValidationError,
                  "Batch mode with remote execution is not supported via this entry point. "
                  "Use solve_batch_remote() instead.");
    auto* cpu_prob = dynamic_cast<cpu_optimization_problem_t<i_t, f_t>*>(problem_interface);
    cuopt_expects(cpu_prob != nullptr,
                  error_type_t::ValidationError,
                  "Remote execution requires CPU memory backend");
    return solve_lp_remote(*cpu_prob, settings);
  }
#else
  cuopt_expects(!is_remote_execution_enabled(),
                error_type_t::ValidationError,
                "Remote execution was requested, but this build was compiled without gRPC support");
#endif

  // Local execution - dispatch to appropriate overload based on problem type
  auto* cpu_prob = dynamic_cast<cpu_optimization_problem_t<i_t, f_t>*>(problem_interface);
  if (cpu_prob != nullptr) {
    return solve_lp(*cpu_prob, settings, problem_checking, use_pdlp_solver_mode, is_batch_mode);
  }

  // GPU problem: call GPU solver directly
  auto* gpu_prob = dynamic_cast<optimization_problem_t<i_t, f_t>*>(problem_interface);
  cuopt_expects(gpu_prob != nullptr,
                error_type_t::ValidationError,
                "problem_interface must be either a CPU or GPU optimization problem");
  auto gpu_solution =
    solve_lp<i_t, f_t>(*gpu_prob, settings, problem_checking, use_pdlp_solver_mode, is_batch_mode);
  return std::make_unique<gpu_lp_solution_t<i_t, f_t>>(std::move(gpu_solution));
}

#define INSTANTIATE(F_TYPE)                                                                      \
  template optimization_problem_solution_t<int, F_TYPE> solve_lp(                                \
    optimization_problem_t<int, F_TYPE>& op_problem,                                             \
    pdlp_solver_settings_t<int, F_TYPE> const& settings,                                         \
    bool problem_checking,                                                                       \
    bool use_pdlp_solver_mode,                                                                   \
    bool is_batch_mode);                                                                         \
                                                                                                 \
  template optimization_problem_solution_t<int, F_TYPE> solve_lp(                                \
    raft::handle_t const* handle_ptr,                                                            \
    const cuopt::linear_programming::io::mps_data_model_t<int, F_TYPE>& mps_data_model,          \
    pdlp_solver_settings_t<int, F_TYPE> const& settings,                                         \
    bool problem_checking,                                                                       \
    bool use_pdlp_solver_mode);                                                                  \
                                                                                                 \
  template std::unique_ptr<lp_solution_interface_t<int, F_TYPE>> solve_lp(                       \
    cpu_optimization_problem_t<int, F_TYPE>&,                                                    \
    pdlp_solver_settings_t<int, F_TYPE> const&,                                                  \
    bool,                                                                                        \
    bool,                                                                                        \
    bool);                                                                                       \
                                                                                                 \
  template std::unique_ptr<lp_solution_interface_t<int, F_TYPE>> solve_lp(                       \
    optimization_problem_interface_t<int, F_TYPE>*,                                              \
    pdlp_solver_settings_t<int, F_TYPE> const&,                                                  \
    bool,                                                                                        \
    bool,                                                                                        \
    bool);                                                                                       \
                                                                                                 \
  template optimization_problem_solution_t<int, F_TYPE> solve_lp_with_method(                    \
    detail::problem_t<int, F_TYPE>& problem,                                                     \
    pdlp_solver_settings_t<int, F_TYPE> const& settings,                                         \
    const timer_t& timer,                                                                        \
    bool is_batch_mode);                                                                         \
                                                                                                 \
  template optimization_problem_solution_t<int, F_TYPE> batch_pdlp_solve(                        \
    raft::handle_t const* handle_ptr,                                                            \
    const cuopt::linear_programming::io::mps_data_model_t<int, F_TYPE>& mps_data_model,          \
    const std::vector<int>& fractional,                                                          \
    const std::vector<F_TYPE>& root_soln_x,                                                      \
    pdlp_solver_settings_t<int, F_TYPE> const& settings);                                        \
                                                                                                 \
  template optimization_problem_solution_t<int, F_TYPE> run_batch_pdlp(                          \
    optimization_problem_t<int, F_TYPE>& problem,                                                \
    pdlp_solver_settings_t<int, F_TYPE> const& settings);                                        \
                                                                                                 \
  template size_t compute_optimal_batch_size(const optimization_problem_t<int, F_TYPE>& problem, \
                                             bool per_climber_objectives,                        \
                                             bool per_climber_constraint_bounds,                 \
                                             bool collect_solutions);                            \
                                                                                                 \
  template optimization_problem_t<int, F_TYPE> mps_data_model_to_optimization_problem(           \
    raft::handle_t const* handle_ptr,                                                            \
    const cuopt::linear_programming::io::mps_data_model_t<int, F_TYPE>& data_model);             \
  template void set_pdlp_solver_mode(pdlp_solver_settings_t<int, F_TYPE>& settings);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

}  // namespace cuopt::linear_programming
