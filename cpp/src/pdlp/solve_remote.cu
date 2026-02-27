/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/cpu_pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/gpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <utilities/logger.hpp>

namespace cuopt::linear_programming {

// ============================================================================
// Remote execution stubs (placeholder implementations)
// ============================================================================

template <typename i_t, typename f_t>
std::unique_ptr<lp_solution_interface_t<i_t, f_t>> solve_lp_remote(
  cpu_optimization_problem_t<i_t, f_t> const& cpu_problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  bool problem_checking,
  bool use_pdlp_solver_mode)
{
  init_logger_t log(settings.log_file, settings.log_to_console);
  CUOPT_LOG_INFO(
    "solve_lp_remote (CPU problem) stub called - returning dummy solution for testing");

  // TODO: Implement actual remote LP solving via gRPC
  // For now, return a dummy solution with fake data (allows testing the full flow)
  i_t n_vars        = cpu_problem.get_n_variables();
  i_t n_constraints = cpu_problem.get_n_constraints();

  std::vector<f_t> primal_solution(n_vars, 0.0);
  std::vector<f_t> dual_solution(n_constraints, 0.0);
  std::vector<f_t> reduced_cost(n_vars, 0.0);

  // Create fake warm start data struct with recognizable non-zero values for testing
  cpu_pdlp_warm_start_data_t<i_t, f_t> warmstart;
  warmstart.current_primal_solution_                  = std::vector<f_t>(n_vars, 1.1);
  warmstart.current_dual_solution_                    = std::vector<f_t>(n_constraints, 2.2);
  warmstart.initial_primal_average_                   = std::vector<f_t>(n_vars, 3.3);
  warmstart.initial_dual_average_                     = std::vector<f_t>(n_constraints, 4.4);
  warmstart.current_ATY_                              = std::vector<f_t>(n_vars, 5.5);
  warmstart.sum_primal_solutions_                     = std::vector<f_t>(n_vars, 6.6);
  warmstart.sum_dual_solutions_                       = std::vector<f_t>(n_constraints, 7.7);
  warmstart.last_restart_duality_gap_primal_solution_ = std::vector<f_t>(n_vars, 8.8);
  warmstart.last_restart_duality_gap_dual_solution_   = std::vector<f_t>(n_constraints, 9.9);
  warmstart.initial_primal_weight_                    = 99.1;
  warmstart.initial_step_size_                        = 99.2;
  warmstart.total_pdlp_iterations_                    = 100;
  warmstart.total_pdhg_iterations_                    = 200;
  warmstart.last_candidate_kkt_score_                 = 99.3;
  warmstart.last_restart_kkt_score_                   = 99.4;
  warmstart.sum_solution_weight_                      = 99.5;
  warmstart.iterations_since_last_restart_            = 10;

  auto solution = std::make_unique<cpu_lp_solution_t<i_t, f_t>>(
    std::move(primal_solution),
    std::move(dual_solution),
    std::move(reduced_cost),
    pdlp_termination_status_t::Optimal,  // Fake optimal status
    0.0,                                 // Primal objective (zero solution)
    0.0,                                 // Dual objective (zero solution)
    0.01,                                // Dummy solve time
    0.001,                               // l2_primal_residual
    0.002,                               // l2_dual_residual
    0.003,                               // gap
    42,                                  // num_iterations
    true,                                // solved_by_pdlp
    std::move(warmstart)                 // warmstart data
  );

  return solution;
}

template <typename i_t, typename f_t>
std::unique_ptr<mip_solution_interface_t<i_t, f_t>> solve_mip_remote(
  cpu_optimization_problem_t<i_t, f_t> const& cpu_problem,
  mip_solver_settings_t<i_t, f_t> const& settings)
{
  init_logger_t log(settings.log_file, settings.log_to_console);
  CUOPT_LOG_INFO(
    "solve_mip_remote (CPU problem) stub called - returning dummy solution for testing");

  // TODO: Implement actual remote MIP solving via gRPC
  // For now, return a dummy solution with fake data (allows testing the full flow)
  i_t n_vars = cpu_problem.get_n_variables();

  std::vector<f_t> solution(n_vars, 0.0);
  auto mip_solution = std::make_unique<cpu_mip_solution_t<i_t, f_t>>(
    std::move(solution),
    mip_termination_status_t::Optimal,  // Fake optimal status
    0.0,                                // Objective value (zero solution)
    0.0,                                // MIP gap
    0.0,                                // Solution bound
    0.01,                               // Total solve time
    0.0,                                // Presolve time
    0.0,                                // Max constraint violation
    0.0,                                // Max int violation
    0.0,                                // Max variable bound violation
    0,                                  // Number of nodes
    0);                                 // Number of simplex iterations

  return mip_solution;
}

// ============================================================================
// Helper: Convert GPU problem to CPU problem (copies all device data to host)
// ============================================================================

template <typename i_t, typename f_t>
cpu_optimization_problem_t<i_t, f_t> gpu_problem_to_cpu(
  gpu_optimization_problem_t<i_t, f_t> const& gpu_problem)
{
  auto cpu_problem = cpu_optimization_problem_t<i_t, f_t>(nullptr);  // No CUDA resources for remote

  // Copy scalar properties
  cpu_problem.set_maximize(gpu_problem.get_sense());
  cpu_problem.set_objective_offset(gpu_problem.get_objective_offset());
  cpu_problem.set_objective_scaling_factor(gpu_problem.get_objective_scaling_factor());
  cpu_problem.set_problem_category(gpu_problem.get_problem_category());

  // Copy names
  cpu_problem.set_problem_name(gpu_problem.get_problem_name());
  cpu_problem.set_objective_name(gpu_problem.get_objective_name());
  cpu_problem.set_variable_names(gpu_problem.get_variable_names());
  cpu_problem.set_row_names(gpu_problem.get_row_names());

  // Copy objective coefficients
  auto obj_coeffs = gpu_problem.get_objective_coefficients_host();
  if (!obj_coeffs.empty()) {
    cpu_problem.set_objective_coefficients(obj_coeffs.data(), obj_coeffs.size());
  }

  // Copy constraint matrix (CSR format)
  auto matrix_values  = gpu_problem.get_constraint_matrix_values_host();
  auto matrix_indices = gpu_problem.get_constraint_matrix_indices_host();
  auto matrix_offsets = gpu_problem.get_constraint_matrix_offsets_host();
  if (!matrix_values.empty()) {
    cpu_problem.set_csr_constraint_matrix(matrix_values.data(),
                                          matrix_values.size(),
                                          matrix_indices.data(),
                                          matrix_indices.size(),
                                          matrix_offsets.data(),
                                          matrix_offsets.size());
  }

  // Copy constraint bounds (single RHS b-vector)
  auto constraint_bounds = gpu_problem.get_constraint_bounds_host();
  if (!constraint_bounds.empty()) {
    cpu_problem.set_constraint_bounds(constraint_bounds.data(), constraint_bounds.size());
  }

  // Copy constraint lower/upper bounds
  auto constraint_lb = gpu_problem.get_constraint_lower_bounds_host();
  auto constraint_ub = gpu_problem.get_constraint_upper_bounds_host();
  if (!constraint_lb.empty()) {
    cpu_problem.set_constraint_lower_bounds(constraint_lb.data(), constraint_lb.size());
  }
  if (!constraint_ub.empty()) {
    cpu_problem.set_constraint_upper_bounds(constraint_ub.data(), constraint_ub.size());
  }

  // Copy row types (constraint sense: E/L/G)
  auto row_types = gpu_problem.get_row_types_host();
  if (!row_types.empty()) { cpu_problem.set_row_types(row_types.data(), row_types.size()); }

  // Copy variable bounds
  auto var_lb = gpu_problem.get_variable_lower_bounds_host();
  auto var_ub = gpu_problem.get_variable_upper_bounds_host();
  if (!var_lb.empty()) { cpu_problem.set_variable_lower_bounds(var_lb.data(), var_lb.size()); }
  if (!var_ub.empty()) { cpu_problem.set_variable_upper_bounds(var_ub.data(), var_ub.size()); }

  // Copy variable types
  auto var_types = gpu_problem.get_variable_types_host();
  if (!var_types.empty()) { cpu_problem.set_variable_types(var_types.data(), var_types.size()); }

  // Copy quadratic objective if present
  if (gpu_problem.has_quadratic_objective()) {
    auto quad_offsets = gpu_problem.get_quadratic_objective_offsets();
    auto quad_indices = gpu_problem.get_quadratic_objective_indices();
    auto quad_values  = gpu_problem.get_quadratic_objective_values();
    cpu_problem.set_quadratic_objective_matrix(quad_values.data(),
                                               quad_values.size(),
                                               quad_indices.data(),
                                               quad_indices.size(),
                                               quad_offsets.data(),
                                               quad_offsets.size());
  }

  return cpu_problem;
}

// ============================================================================
// Remote execution for GPU problems (converts to CPU then calls CPU remote)
// ============================================================================

template <typename i_t, typename f_t>
std::unique_ptr<lp_solution_interface_t<i_t, f_t>> solve_lp_remote(
  gpu_optimization_problem_t<i_t, f_t> const& gpu_problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  bool problem_checking,
  bool use_pdlp_solver_mode)
{
  init_logger_t log(settings.log_file, settings.log_to_console);
  CUOPT_LOG_INFO("solve_lp_remote (GPU problem) - converting to CPU for remote execution");

  auto cpu_problem = gpu_problem_to_cpu(gpu_problem);

  // Call CPU remote solver (returns unique_ptr<lp_solution_interface_t>)
  auto cpu_solution_interface =
    solve_lp_remote(cpu_problem, settings, problem_checking, use_pdlp_solver_mode);

  // Convert CPU solution back to GPU solution (since we started with a GPU problem)
  auto gpu_solution = cpu_solution_interface->to_gpu_solution(rmm::cuda_stream_per_thread);
  return std::make_unique<gpu_lp_solution_t<i_t, f_t>>(std::move(gpu_solution));
}

template <typename i_t, typename f_t>
std::unique_ptr<mip_solution_interface_t<i_t, f_t>> solve_mip_remote(
  gpu_optimization_problem_t<i_t, f_t> const& gpu_problem,
  mip_solver_settings_t<i_t, f_t> const& settings)
{
  init_logger_t log(settings.log_file, settings.log_to_console);
  CUOPT_LOG_INFO("solve_mip_remote (GPU problem) - converting to CPU for remote execution");

  auto cpu_problem = gpu_problem_to_cpu(gpu_problem);

  // Call CPU remote solver (returns unique_ptr<mip_solution_interface_t>)
  auto cpu_solution_interface = solve_mip_remote(cpu_problem, settings);

  // Convert CPU solution back to GPU solution (since we started with a GPU problem)
  auto gpu_solution = cpu_solution_interface->to_gpu_solution(rmm::cuda_stream_per_thread);
  return std::make_unique<gpu_mip_solution_t<i_t, f_t>>(std::move(gpu_solution));
}

// Explicit template instantiations for remote execution stubs
template std::unique_ptr<lp_solution_interface_t<int, double>> solve_lp_remote(
  cpu_optimization_problem_t<int, double> const&,
  pdlp_solver_settings_t<int, double> const&,
  bool,
  bool);

template std::unique_ptr<mip_solution_interface_t<int, double>> solve_mip_remote(
  cpu_optimization_problem_t<int, double> const&, mip_solver_settings_t<int, double> const&);

template std::unique_ptr<lp_solution_interface_t<int, double>> solve_lp_remote(
  gpu_optimization_problem_t<int, double> const&,
  pdlp_solver_settings_t<int, double> const&,
  bool,
  bool);

template std::unique_ptr<mip_solution_interface_t<int, double>> solve_mip_remote(
  gpu_optimization_problem_t<int, double> const&, mip_solver_settings_t<int, double> const&);

}  // namespace cuopt::linear_programming
