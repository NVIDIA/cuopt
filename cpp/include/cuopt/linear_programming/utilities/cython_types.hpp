/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>

#include <rmm/device_buffer.hpp>

#include <memory>
#include <string>
#include <vector>

namespace cuopt {
namespace cython {

// GPU-backed LP solution struct (uses device memory)
struct linear_programming_ret_t {
  std::unique_ptr<rmm::device_buffer> primal_solution_;
  std::unique_ptr<rmm::device_buffer> dual_solution_;
  std::unique_ptr<rmm::device_buffer> reduced_cost_;
  /* -- PDLP Warm Start Data -- */
  std::unique_ptr<rmm::device_buffer> current_primal_solution_;
  std::unique_ptr<rmm::device_buffer> current_dual_solution_;
  std::unique_ptr<rmm::device_buffer> initial_primal_average_;
  std::unique_ptr<rmm::device_buffer> initial_dual_average_;
  std::unique_ptr<rmm::device_buffer> current_ATY_;
  std::unique_ptr<rmm::device_buffer> sum_primal_solutions_;
  std::unique_ptr<rmm::device_buffer> sum_dual_solutions_;
  std::unique_ptr<rmm::device_buffer> last_restart_duality_gap_primal_solution_;
  std::unique_ptr<rmm::device_buffer> last_restart_duality_gap_dual_solution_;
  double initial_primal_weight_;
  double initial_step_size_;
  int total_pdlp_iterations_;
  int total_pdhg_iterations_;
  double last_candidate_kkt_score_;
  double last_restart_kkt_score_;
  double sum_solution_weight_;
  int iterations_since_last_restart_;
  /* -- /PDLP Warm Start Data -- */

  linear_programming::pdlp_termination_status_t termination_status_;
  error_type_t error_status_;
  std::string error_message_;

  /*Termination stats*/
  double l2_primal_residual_;
  double l2_dual_residual_;
  double primal_objective_;
  double dual_objective_;
  double gap_;
  int nb_iterations_;
  double solve_time_;
  bool solved_by_pdlp_;
};

// CPU-backed LP solution struct (uses host memory)
struct cpu_linear_programming_ret_t {
  std::vector<double> primal_solution_;
  std::vector<double> dual_solution_;
  std::vector<double> reduced_cost_;
  /* -- PDLP Warm Start Data -- */
  std::vector<double> current_primal_solution_;
  std::vector<double> current_dual_solution_;
  std::vector<double> initial_primal_average_;
  std::vector<double> initial_dual_average_;
  std::vector<double> current_ATY_;
  std::vector<double> sum_primal_solutions_;
  std::vector<double> sum_dual_solutions_;
  std::vector<double> last_restart_duality_gap_primal_solution_;
  std::vector<double> last_restart_duality_gap_dual_solution_;
  double initial_primal_weight_;
  double initial_step_size_;
  int total_pdlp_iterations_;
  int total_pdhg_iterations_;
  double last_candidate_kkt_score_;
  double last_restart_kkt_score_;
  double sum_solution_weight_;
  int iterations_since_last_restart_;
  /* -- /PDLP Warm Start Data -- */

  linear_programming::pdlp_termination_status_t termination_status_;
  error_type_t error_status_;
  std::string error_message_;

  /*Termination stats*/
  double l2_primal_residual_;
  double l2_dual_residual_;
  double primal_objective_;
  double dual_objective_;
  double gap_;
  int nb_iterations_;
  double solve_time_;
  bool solved_by_pdlp_;
};

// GPU-backed MIP solution struct (uses device memory)
struct mip_ret_t {
  std::unique_ptr<rmm::device_buffer> solution_;

  linear_programming::mip_termination_status_t termination_status_;
  error_type_t error_status_;
  std::string error_message_;

  /*Termination stats*/
  double objective_;
  double mip_gap_;
  double solution_bound_;
  double total_solve_time_;
  double presolve_time_;
  double max_constraint_violation_;
  double max_int_violation_;
  double max_variable_bound_violation_;
  int nodes_;
  int simplex_iterations_;
};

// CPU-backed MIP solution struct (uses host memory)
struct cpu_mip_ret_t {
  std::vector<double> solution_;

  linear_programming::mip_termination_status_t termination_status_;
  error_type_t error_status_;
  std::string error_message_;

  /*Termination stats*/
  double objective_;
  double mip_gap_;
  double solution_bound_;
  double total_solve_time_;
  double presolve_time_;
  double max_constraint_violation_;
  double max_int_violation_;
  double max_variable_bound_violation_;
  int nodes_;
  int simplex_iterations_;
};

}  // namespace cython
}  // namespace cuopt
