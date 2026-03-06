/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/cpu_pdlp_warm_start_data.hpp>
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

// Explicit template instantiations for remote execution stubs
template std::unique_ptr<lp_solution_interface_t<int, double>> solve_lp_remote(
  cpu_optimization_problem_t<int, double> const&,
  pdlp_solver_settings_t<int, double> const&,
  bool,
  bool);

template std::unique_ptr<mip_solution_interface_t<int, double>> solve_mip_remote(
  cpu_optimization_problem_t<int, double> const&, mip_solver_settings_t<int, double> const&);

}  // namespace cuopt::linear_programming
