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
#include <variant>
#include <vector>

namespace cuopt {
namespace cython {

using gpu_buffer = std::unique_ptr<rmm::device_buffer>;
using cpu_buffer = std::vector<double>;

// LP solution struct — GPU and CPU solutions use the same struct, differing only in the
// vector storage type (device_buffer vs std::vector).  The solutions_ variant holds all
// buffer/vector fields; shared scalar fields live directly on the struct.
struct linear_programming_ret_t {
  struct gpu_solutions_t {
    gpu_buffer primal_solution_;
    gpu_buffer dual_solution_;
    gpu_buffer reduced_cost_;
    gpu_buffer current_primal_solution_;
    gpu_buffer current_dual_solution_;
    gpu_buffer initial_primal_average_;
    gpu_buffer initial_dual_average_;
    gpu_buffer current_ATY_;
    gpu_buffer sum_primal_solutions_;
    gpu_buffer sum_dual_solutions_;
    gpu_buffer last_restart_duality_gap_primal_solution_;
    gpu_buffer last_restart_duality_gap_dual_solution_;
  };

  struct cpu_solutions_t {
    cpu_buffer primal_solution_;
    cpu_buffer dual_solution_;
    cpu_buffer reduced_cost_;
    cpu_buffer current_primal_solution_;
    cpu_buffer current_dual_solution_;
    cpu_buffer initial_primal_average_;
    cpu_buffer initial_dual_average_;
    cpu_buffer current_ATY_;
    cpu_buffer sum_primal_solutions_;
    cpu_buffer sum_dual_solutions_;
    cpu_buffer last_restart_duality_gap_primal_solution_;
    cpu_buffer last_restart_duality_gap_dual_solution_;
  };

  std::variant<gpu_solutions_t, cpu_solutions_t> solutions_;

  /* -- PDLP Warm Start Scalars -- */
  double initial_primal_weight_{};
  double initial_step_size_{};
  int total_pdlp_iterations_{};
  int total_pdhg_iterations_{};
  double last_candidate_kkt_score_{};
  double last_restart_kkt_score_{};
  double sum_solution_weight_{};
  int iterations_since_last_restart_{};
  /* -- /PDLP Warm Start Scalars -- */

  linear_programming::pdlp_termination_status_t termination_status_{};
  error_type_t error_status_{};
  std::string error_message_;

  /*Termination stats*/
  double l2_primal_residual_{};
  double l2_dual_residual_{};
  double primal_objective_{};
  double dual_objective_{};
  double gap_{};
  int nb_iterations_{};
  double solve_time_{};
  bool solved_by_pdlp_{};

  bool is_gpu() const { return std::holds_alternative<gpu_solutions_t>(solutions_); }
};

// MIP solution struct — GPU and CPU solutions use the same struct, differing only in the
// solution vector storage type.
struct mip_ret_t {
  std::variant<gpu_buffer, cpu_buffer> solution_;

  linear_programming::mip_termination_status_t termination_status_{};
  error_type_t error_status_{};
  std::string error_message_;

  /*Termination stats*/
  double objective_{};
  double mip_gap_{};
  double solution_bound_{};
  double total_solve_time_{};
  double presolve_time_{};
  double max_constraint_violation_{};
  double max_int_violation_{};
  double max_variable_bound_violation_{};
  int nodes_{};
  int simplex_iterations_{};

  bool is_gpu() const { return std::holds_alternative<gpu_buffer>(solution_); }
};

}  // namespace cython
}  // namespace cuopt
