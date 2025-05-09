/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include <mps_parser/data_model_view.hpp>
#include <raft/core/handle.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace cuopt {
namespace cython {

// aggregate for call_solve() return type
// to be exposed to cython:
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

  /*Termination stats*/
  double l2_primal_residual_;
  double l2_dual_residual_;
  double primal_objective_;
  double dual_objective_;
  double gap_;
  int nb_iterations_;
  double solve_time_;
};

struct mip_ret_t {
  std::unique_ptr<rmm::device_buffer> solution_;

  linear_programming::mip_termination_status_t termination_status_;

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

struct solver_ret_t {
  linear_programming::problem_category_t problem_type;
  linear_programming_ret_t lp_ret;
  mip_ret_t mip_ret;
  // possibly add dual simplex return structure
};

// Wrapper for solve to expose the API to cython.

std::unique_ptr<solver_ret_t> call_solve(cuopt::mps_parser::data_model_view_t<int, double>*,
                                         linear_programming::solver_settings_t<int, double>*);

std::pair<std::vector<std::unique_ptr<solver_ret_t>>, double> call_batch_solve(
  std::vector<cuopt::mps_parser::data_model_view_t<int, double>*>,
  linear_programming::solver_settings_t<int, double>*);

}  // namespace cython
}  // namespace cuopt
