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

#include <cuopt/export.hpp>
#include <cuopt/mathematical_optimization/cpu_optimization_problem_solution.hpp>
#include <cuopt/mathematical_optimization/optimization_problem_solution.hpp>
#include <cuopt/mathematical_optimization/utilities/cython_solve.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::mathematical_optimization {

// ===========================
// GPU LP Solution Conversion
// ===========================

template <typename i_t, typename f_t>
cuopt::cython::linear_programming_ret_t gpu_lp_solution_t<i_t, f_t>::to_linear_programming_ret_t()
{
  using gpu_solutions_t = cuopt::cython::linear_programming_ret_t::gpu_solutions_t;
  cuopt::cython::linear_programming_ret_t ret;

  auto& sol = solution_;
  gpu_solutions_t gpu;

  gpu.primal_solution_ =
    std::make_unique<rmm::device_buffer>(std::move(sol.get_primal_solution()).release());
  gpu.dual_solution_ =
    std::make_unique<rmm::device_buffer>(std::move(sol.get_dual_solution()).release());
  gpu.reduced_cost_ =
    std::make_unique<rmm::device_buffer>(std::move(sol.get_reduced_cost()).release());

  auto& ws = sol.get_pdlp_warm_start_data();
  if (ws.current_primal_solution_.size() > 0) {
    gpu.current_primal_solution_ =
      std::make_unique<rmm::device_buffer>(std::move(ws.current_primal_solution_).release());
    gpu.current_dual_solution_ =
      std::make_unique<rmm::device_buffer>(std::move(ws.current_dual_solution_).release());
    gpu.initial_primal_average_ =
      std::make_unique<rmm::device_buffer>(std::move(ws.initial_primal_average_).release());
    gpu.initial_dual_average_ =
      std::make_unique<rmm::device_buffer>(std::move(ws.initial_dual_average_).release());
    gpu.current_ATY_ = std::make_unique<rmm::device_buffer>(std::move(ws.current_ATY_).release());
    gpu.sum_primal_solutions_ =
      std::make_unique<rmm::device_buffer>(std::move(ws.sum_primal_solutions_).release());
    gpu.sum_dual_solutions_ =
      std::make_unique<rmm::device_buffer>(std::move(ws.sum_dual_solutions_).release());
    gpu.last_restart_duality_gap_primal_solution_ = std::make_unique<rmm::device_buffer>(
      std::move(ws.last_restart_duality_gap_primal_solution_).release());
    gpu.last_restart_duality_gap_dual_solution_ = std::make_unique<rmm::device_buffer>(
      std::move(ws.last_restart_duality_gap_dual_solution_).release());

    ret.initial_primal_weight_         = ws.initial_primal_weight_;
    ret.initial_step_size_             = ws.initial_step_size_;
    ret.total_pdlp_iterations_         = ws.total_pdlp_iterations_;
    ret.total_pdhg_iterations_         = ws.total_pdhg_iterations_;
    ret.last_candidate_kkt_score_      = ws.last_candidate_kkt_score_;
    ret.last_restart_kkt_score_        = ws.last_restart_kkt_score_;
    ret.sum_solution_weight_           = ws.sum_solution_weight_;
    ret.iterations_since_last_restart_ = ws.iterations_since_last_restart_;
  } else {
    gpu.current_primal_solution_                  = std::make_unique<rmm::device_buffer>();
    gpu.current_dual_solution_                    = std::make_unique<rmm::device_buffer>();
    gpu.initial_primal_average_                   = std::make_unique<rmm::device_buffer>();
    gpu.initial_dual_average_                     = std::make_unique<rmm::device_buffer>();
    gpu.current_ATY_                              = std::make_unique<rmm::device_buffer>();
    gpu.sum_primal_solutions_                     = std::make_unique<rmm::device_buffer>();
    gpu.sum_dual_solutions_                       = std::make_unique<rmm::device_buffer>();
    gpu.last_restart_duality_gap_primal_solution_ = std::make_unique<rmm::device_buffer>();
    gpu.last_restart_duality_gap_dual_solution_   = std::make_unique<rmm::device_buffer>();
  }

  ret.solutions_ = std::move(gpu);

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
    ret.solved_by_          = term_info.solved_by;
  }

  return ret;
}

// ===========================
// GPU MIP Solution Conversion
// ===========================

template <typename i_t, typename f_t>
cuopt::cython::mip_ret_t gpu_mip_solution_t<i_t, f_t>::to_mip_ret_t()
{
  cuopt::cython::mip_ret_t ret;

  ret.solution_ =
    std::make_unique<rmm::device_buffer>(std::move(solution_.get_solution()).release());

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
// Explicit template instantiations
template CUOPT_EXPORT cuopt::cython::linear_programming_ret_t
gpu_lp_solution_t<int, double>::to_linear_programming_ret_t();
template CUOPT_EXPORT cuopt::cython::mip_ret_t gpu_mip_solution_t<int, double>::to_mip_ret_t();

}  // namespace cuopt::mathematical_optimization
