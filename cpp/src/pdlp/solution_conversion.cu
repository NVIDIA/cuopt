/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

/**
 * @file solution_conversion.cu
 * @brief Implementations of conversion methods from solution classes to Cython ret structs
 */

#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/gpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/utilities/cython_solve.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming {

// ===========================
// GPU LP Solution Conversion
// ===========================

template <typename i_t, typename f_t>
cuopt::cython::linear_programming_ret_t
gpu_lp_solution_t<i_t, f_t>::to_linear_programming_ret_t() &&
{
  cuopt::cython::linear_programming_ret_t ret;

  // Move GPU solution data into device_buffer wrappers
  // This is zero-copy - we're just transferring ownership
  auto& sol = solution_;

  // Main solution vectors
  ret.primal_solution_ =
    std::make_unique<rmm::device_buffer>(std::move(sol.get_primal_solution()).release());
  ret.dual_solution_ =
    std::make_unique<rmm::device_buffer>(std::move(sol.get_dual_solution()).release());
  ret.reduced_cost_ =
    std::make_unique<rmm::device_buffer>(std::move(sol.get_reduced_cost()).release());

  // Warm start data
  auto& ws = sol.get_pdlp_warm_start_data();
  if (ws.current_primal_solution_.size() > 0) {
    ret.current_primal_solution_ =
      std::make_unique<rmm::device_buffer>(std::move(ws.current_primal_solution_).release());
    ret.current_dual_solution_ =
      std::make_unique<rmm::device_buffer>(std::move(ws.current_dual_solution_).release());
    ret.initial_primal_average_ =
      std::make_unique<rmm::device_buffer>(std::move(ws.initial_primal_average_).release());
    ret.initial_dual_average_ =
      std::make_unique<rmm::device_buffer>(std::move(ws.initial_dual_average_).release());
    ret.current_ATY_ = std::make_unique<rmm::device_buffer>(std::move(ws.current_ATY_).release());
    ret.sum_primal_solutions_ =
      std::make_unique<rmm::device_buffer>(std::move(ws.sum_primal_solutions_).release());
    ret.sum_dual_solutions_ =
      std::make_unique<rmm::device_buffer>(std::move(ws.sum_dual_solutions_).release());
    ret.last_restart_duality_gap_primal_solution_ = std::make_unique<rmm::device_buffer>(
      std::move(ws.last_restart_duality_gap_primal_solution_).release());
    ret.last_restart_duality_gap_dual_solution_ = std::make_unique<rmm::device_buffer>(
      std::move(ws.last_restart_duality_gap_dual_solution_).release());

    // Scalar warm start data
    ret.initial_primal_weight_         = ws.initial_primal_weight_;
    ret.initial_step_size_             = ws.initial_step_size_;
    ret.total_pdlp_iterations_         = ws.total_pdlp_iterations_;
    ret.total_pdhg_iterations_         = ws.total_pdhg_iterations_;
    ret.last_candidate_kkt_score_      = ws.last_candidate_kkt_score_;
    ret.last_restart_kkt_score_        = ws.last_restart_kkt_score_;
    ret.sum_solution_weight_           = ws.sum_solution_weight_;
    ret.iterations_since_last_restart_ = ws.iterations_since_last_restart_;
  } else {
    // No warm start data - set empty buffers and default values
    ret.current_primal_solution_                  = std::make_unique<rmm::device_buffer>();
    ret.current_dual_solution_                    = std::make_unique<rmm::device_buffer>();
    ret.initial_primal_average_                   = std::make_unique<rmm::device_buffer>();
    ret.initial_dual_average_                     = std::make_unique<rmm::device_buffer>();
    ret.current_ATY_                              = std::make_unique<rmm::device_buffer>();
    ret.sum_primal_solutions_                     = std::make_unique<rmm::device_buffer>();
    ret.sum_dual_solutions_                       = std::make_unique<rmm::device_buffer>();
    ret.last_restart_duality_gap_primal_solution_ = std::make_unique<rmm::device_buffer>();
    ret.last_restart_duality_gap_dual_solution_   = std::make_unique<rmm::device_buffer>();

    ret.initial_primal_weight_         = 0.0;
    ret.initial_step_size_             = 0.0;
    ret.total_pdlp_iterations_         = 0;
    ret.total_pdhg_iterations_         = 0;
    ret.last_candidate_kkt_score_      = 0.0;
    ret.last_restart_kkt_score_        = 0.0;
    ret.sum_solution_weight_           = 0.0;
    ret.iterations_since_last_restart_ = 0;
  }

  // Metadata and termination stats
  ret.termination_status_ = solution_.get_termination_status(0);
  ret.error_status_       = solution_.get_error_status().get_error_type();
  ret.error_message_      = std::string(solution_.get_error_status().what());

  auto& term_infos = solution_.get_additional_termination_informations();
  if (!term_infos.empty()) {
    auto& term_info         = term_infos[0];
    ret.l2_primal_residual_ = term_info.l2_primal_residual;
    ret.l2_dual_residual_   = term_info.l2_dual_residual;
    ret.primal_objective_   = term_info.primal_objective;
    ret.dual_objective_     = term_info.dual_objective;
    ret.gap_                = term_info.gap;
    ret.nb_iterations_      = term_info.number_of_steps_taken;
    ret.solve_time_         = term_info.solve_time;
    ret.solved_by_pdlp_     = term_info.solved_by_pdlp;
  } else {
    ret.l2_primal_residual_ = 0.0;
    ret.l2_dual_residual_   = 0.0;
    ret.primal_objective_   = 0.0;
    ret.dual_objective_     = 0.0;
    ret.gap_                = 0.0;
    ret.nb_iterations_      = 0;
    ret.solve_time_         = 0.0;
    ret.solved_by_pdlp_     = false;
  }

  return ret;
}

// ===========================
// GPU MIP Solution Conversion
// ===========================

template <typename i_t, typename f_t>
cuopt::cython::mip_ret_t gpu_mip_solution_t<i_t, f_t>::to_mip_ret_t() &&
{
  cuopt::cython::mip_ret_t ret;

  // Move GPU solution data into device_buffer wrapper
  ret.solution_ =
    std::make_unique<rmm::device_buffer>(std::move(solution_.get_solution()).release());

  // Metadata and termination stats
  ret.termination_status_           = solution_.get_termination_status();
  ret.error_status_                 = solution_.get_error_status().get_error_type();
  ret.error_message_                = std::string(solution_.get_error_status().what());
  ret.objective_                    = solution_.get_objective_value();
  ret.mip_gap_                      = solution_.get_mip_gap();
  ret.solution_bound_               = solution_.get_solution_bound();
  ret.total_solve_time_             = solution_.get_total_solve_time();
  ret.presolve_time_                = solution_.get_presolve_time();
  ret.max_constraint_violation_     = solution_.get_max_constraint_violation();
  ret.max_int_violation_            = solution_.get_max_int_violation();
  ret.max_variable_bound_violation_ = solution_.get_max_variable_bound_violation();
  ret.nodes_                        = solution_.get_num_nodes();
  ret.simplex_iterations_           = solution_.get_num_simplex_iterations();

  return ret;
}

// ===========================
// CPU LP Solution Conversion
// ===========================

template <typename i_t, typename f_t>
cuopt::cython::cpu_linear_programming_ret_t
cpu_lp_solution_t<i_t, f_t>::to_cpu_linear_programming_ret_t() &&
{
  cuopt::cython::cpu_linear_programming_ret_t ret;

  // Move CPU solution data (std::vector move is zero-copy)
  ret.primal_solution_ = std::move(primal_solution_);
  ret.dual_solution_   = std::move(dual_solution_);
  ret.reduced_cost_    = std::move(reduced_cost_);

  // Warm start data (only move if present)
  if (!pdlp_warm_start_data_.current_primal_solution_.empty()) {
    ret.current_primal_solution_ = std::move(pdlp_warm_start_data_.current_primal_solution_);
    ret.current_dual_solution_   = std::move(pdlp_warm_start_data_.current_dual_solution_);
    ret.initial_primal_average_  = std::move(pdlp_warm_start_data_.initial_primal_average_);
    ret.initial_dual_average_    = std::move(pdlp_warm_start_data_.initial_dual_average_);
    ret.current_ATY_             = std::move(pdlp_warm_start_data_.current_ATY_);
    ret.sum_primal_solutions_    = std::move(pdlp_warm_start_data_.sum_primal_solutions_);
    ret.sum_dual_solutions_      = std::move(pdlp_warm_start_data_.sum_dual_solutions_);
    ret.last_restart_duality_gap_primal_solution_ =
      std::move(pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_);
    ret.last_restart_duality_gap_dual_solution_ =
      std::move(pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_);

    // Scalar warm start data
    ret.initial_primal_weight_         = pdlp_warm_start_data_.initial_primal_weight_;
    ret.initial_step_size_             = pdlp_warm_start_data_.initial_step_size_;
    ret.total_pdlp_iterations_         = pdlp_warm_start_data_.total_pdlp_iterations_;
    ret.total_pdhg_iterations_         = pdlp_warm_start_data_.total_pdhg_iterations_;
    ret.last_candidate_kkt_score_      = pdlp_warm_start_data_.last_candidate_kkt_score_;
    ret.last_restart_kkt_score_        = pdlp_warm_start_data_.last_restart_kkt_score_;
    ret.sum_solution_weight_           = pdlp_warm_start_data_.sum_solution_weight_;
    ret.iterations_since_last_restart_ = pdlp_warm_start_data_.iterations_since_last_restart_;
  }

  // Metadata and termination stats
  ret.termination_status_ = termination_status_;
  ret.error_status_       = error_status_.get_error_type();
  ret.error_message_      = std::string(error_status_.what());
  ret.l2_primal_residual_ = l2_primal_residual_;
  ret.l2_dual_residual_   = l2_dual_residual_;
  ret.primal_objective_   = primal_objective_;
  ret.dual_objective_     = dual_objective_;
  ret.gap_                = gap_;
  ret.nb_iterations_      = num_iterations_;
  ret.solve_time_         = solve_time_;
  ret.solved_by_pdlp_     = solved_by_pdlp_;

  return ret;
}

// ===========================
// CPU MIP Solution Conversion
// ===========================

template <typename i_t, typename f_t>
cuopt::cython::cpu_mip_ret_t cpu_mip_solution_t<i_t, f_t>::to_cpu_mip_ret_t() &&
{
  cuopt::cython::cpu_mip_ret_t ret;

  // Move CPU solution data (std::vector move is zero-copy)
  ret.solution_ = std::move(solution_);

  // Metadata and termination stats
  ret.termination_status_           = termination_status_;
  ret.error_status_                 = error_status_.get_error_type();
  ret.error_message_                = std::string(error_status_.what());
  ret.objective_                    = objective_;
  ret.mip_gap_                      = mip_gap_;
  ret.solution_bound_               = solution_bound_;
  ret.total_solve_time_             = total_solve_time_;
  ret.presolve_time_                = presolve_time_;
  ret.max_constraint_violation_     = max_constraint_violation_;
  ret.max_int_violation_            = max_int_violation_;
  ret.max_variable_bound_violation_ = max_variable_bound_violation_;
  ret.nodes_                        = num_nodes_;
  ret.simplex_iterations_           = num_simplex_iterations_;

  return ret;
}

// Explicit template instantiations
template cuopt::cython::linear_programming_ret_t
gpu_lp_solution_t<int, double>::to_linear_programming_ret_t() &&;
template cuopt::cython::mip_ret_t gpu_mip_solution_t<int, double>::to_mip_ret_t() &&;
template cuopt::cython::cpu_linear_programming_ret_t
cpu_lp_solution_t<int, double>::to_cpu_linear_programming_ret_t() &&;
template cuopt::cython::cpu_mip_ret_t cpu_mip_solution_t<int, double>::to_cpu_mip_ret_t() &&;

}  // namespace cuopt::linear_programming
