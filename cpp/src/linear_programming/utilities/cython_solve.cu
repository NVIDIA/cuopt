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

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <cuopt/linear_programming/utilities/cython_solve.hpp>
#include <mip/logger.hpp>
#include <mps_parser/data_model_view.hpp>

#include <raft/common/nvtx.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_buffer.hpp>

#include <utility>
#include <vector>

#include <unistd.h>

namespace cuopt {
namespace cython {

using cuopt::linear_programming::var_t;

static cuopt::linear_programming::optimization_problem_t<int, double>
data_model_to_optimization_problem(
  cuopt::mps_parser::data_model_view_t<int, double>* data_model,
  cuopt::linear_programming::solver_settings_t<int, double>* solver_settings,
  raft::handle_t const* handle_ptr)
{
  cuopt::linear_programming::optimization_problem_t<int, double> op_problem(handle_ptr);
  op_problem.set_maximize(data_model->get_sense());
  if (data_model->get_constraint_matrix_values().size() != 0 &&
      data_model->get_constraint_matrix_indices().size() != 0 &&
      data_model->get_constraint_matrix_offsets().size() != 0) {
    op_problem.set_csr_constraint_matrix(data_model->get_constraint_matrix_values().data(),
                                         data_model->get_constraint_matrix_values().size(),
                                         data_model->get_constraint_matrix_indices().data(),
                                         data_model->get_constraint_matrix_indices().size(),
                                         data_model->get_constraint_matrix_offsets().data(),
                                         data_model->get_constraint_matrix_offsets().size());
  }
  if (data_model->get_constraint_bounds().size() != 0) {
    op_problem.set_constraint_bounds(data_model->get_constraint_bounds().data(),
                                     data_model->get_constraint_bounds().size());
  }
  if (data_model->get_objective_coefficients().size() != 0) {
    op_problem.set_objective_coefficients(data_model->get_objective_coefficients().data(),
                                          data_model->get_objective_coefficients().size());
  }
  op_problem.set_objective_scaling_factor(data_model->get_objective_scaling_factor());
  op_problem.set_objective_offset(data_model->get_objective_offset());
  if (data_model->get_variable_lower_bounds().size() != 0) {
    op_problem.set_variable_lower_bounds(data_model->get_variable_lower_bounds().data(),
                                         data_model->get_variable_lower_bounds().size());
  }
  if (data_model->get_variable_upper_bounds().size() != 0) {
    op_problem.set_variable_upper_bounds(data_model->get_variable_upper_bounds().data(),
                                         data_model->get_variable_upper_bounds().size());
  }

  if (data_model->get_row_types().size() != 0) {
    op_problem.set_row_types(data_model->get_row_types().data(),
                             data_model->get_row_types().size());
  }
  if (data_model->get_constraint_lower_bounds().size() != 0) {
    op_problem.set_constraint_lower_bounds(data_model->get_constraint_lower_bounds().data(),
                                           data_model->get_constraint_lower_bounds().size());
  }
  if (data_model->get_constraint_upper_bounds().size() != 0) {
    op_problem.set_constraint_upper_bounds(data_model->get_constraint_upper_bounds().data(),
                                           data_model->get_constraint_upper_bounds().size());
  }

  if (solver_settings->get_pdlp_warm_start_data_view()
        .last_restart_duality_gap_dual_solution_.data() != nullptr) {
    // Moved inside
    cuopt::linear_programming::pdlp_warm_start_data_t<int, double> pdlp_warm_start_data(
      solver_settings->get_pdlp_warm_start_data_view(), handle_ptr->get_stream());
    solver_settings->get_pdlp_settings().set_pdlp_warm_start_data(pdlp_warm_start_data);
  }

  if (data_model->get_variable_types().size() != 0) {
    std::vector<var_t> enum_variable_types(data_model->get_variable_types().size());
    std::transform(
      data_model->get_variable_types().data(),
      data_model->get_variable_types().data() + data_model->get_variable_types().size(),
      enum_variable_types.begin(),
      [](const auto val) -> var_t { return val == 'I' ? var_t::INTEGER : var_t::CONTINUOUS; });
    op_problem.set_variable_types(enum_variable_types.data(), enum_variable_types.size());
  }

  return op_problem;
}

/**
 * @brief Wrapper for linear_programming to expose the API to cython
 *
 * @param data_model Composable data model object
 * @param solver_settings PDLP solver settings object
 * @return linear_programming_ret_t
 */
linear_programming_ret_t call_solve_lp(
  cuopt::linear_programming::optimization_problem_t<int, double>& op_problem,
  cuopt::linear_programming::pdlp_solver_settings_t<int, double>& solver_settings)
{
  raft::common::nvtx::range fun_scope("Call Solve");
  cuopt_expects(
    op_problem.get_problem_category() == cuopt::linear_programming::problem_category_t::LP,
    error_type_t::ValidationError,
    "LP solve cannot be called on a MIP problem!");
  auto solution = cuopt::linear_programming::solve_lp(op_problem, solver_settings);
  linear_programming_ret_t lp_ret{
    std::make_unique<rmm::device_buffer>(solution.get_primal_solution().release()),
    std::make_unique<rmm::device_buffer>(solution.get_dual_solution().release()),
    std::make_unique<rmm::device_buffer>(solution.get_reduced_cost().release()),
    std::make_unique<rmm::device_buffer>(
      solution.get_pdlp_warm_start_data().current_primal_solution_.release()),
    std::make_unique<rmm::device_buffer>(
      solution.get_pdlp_warm_start_data().current_dual_solution_.release()),
    std::make_unique<rmm::device_buffer>(
      solution.get_pdlp_warm_start_data().initial_primal_average_.release()),
    std::make_unique<rmm::device_buffer>(
      solution.get_pdlp_warm_start_data().initial_dual_average_.release()),
    std::make_unique<rmm::device_buffer>(
      solution.get_pdlp_warm_start_data().current_ATY_.release()),
    std::make_unique<rmm::device_buffer>(
      solution.get_pdlp_warm_start_data().sum_primal_solutions_.release()),
    std::make_unique<rmm::device_buffer>(
      solution.get_pdlp_warm_start_data().sum_dual_solutions_.release()),
    std::make_unique<rmm::device_buffer>(
      solution.get_pdlp_warm_start_data().last_restart_duality_gap_primal_solution_.release()),
    std::make_unique<rmm::device_buffer>(
      solution.get_pdlp_warm_start_data().last_restart_duality_gap_dual_solution_.release()),
    solution.get_pdlp_warm_start_data().initial_primal_weight_,
    solution.get_pdlp_warm_start_data().initial_step_size_,
    solution.get_pdlp_warm_start_data().total_pdlp_iterations_,
    solution.get_pdlp_warm_start_data().total_pdhg_iterations_,
    solution.get_pdlp_warm_start_data().last_candidate_kkt_score_,
    solution.get_pdlp_warm_start_data().last_restart_kkt_score_,
    solution.get_pdlp_warm_start_data().sum_solution_weight_,
    solution.get_pdlp_warm_start_data().iterations_since_last_restart_,
    solution.get_termination_status(),
    solution.get_error_status().get_error_type(),
    solution.get_error_status().what(),
    solution.get_additional_termination_information().l2_primal_residual,
    solution.get_additional_termination_information().l2_dual_residual,
    solution.get_additional_termination_information().primal_objective,
    solution.get_additional_termination_information().dual_objective,
    solution.get_additional_termination_information().gap,
    solution.get_additional_termination_information().number_of_steps_taken,
    solution.get_additional_termination_information().solve_time};

  return lp_ret;
}

/**
 * @brief Wrapper for linear_programming to expose the API to cython
 *
 * @param data_model Composable data model object
 * @param solver_settings MIP solver settings object
 * @return mip_ret_t
 */
mip_ret_t call_solve_mip(
  cuopt::linear_programming::optimization_problem_t<int, double>& op_problem,
  cuopt::linear_programming::mip_solver_settings_t<int, double>& solver_settings)
{
  raft::common::nvtx::range fun_scope("Call Solve");
  cuopt_expects(
    (op_problem.get_problem_category() == cuopt::linear_programming::problem_category_t::MIP) or
      (op_problem.get_problem_category() == cuopt::linear_programming::problem_category_t::IP),
    error_type_t::ValidationError,
    "MIP solve cannot be called on an LP problem!");
  auto solution = cuopt::linear_programming::solve_mip(op_problem, solver_settings);
  mip_ret_t mip_ret{std::make_unique<rmm::device_buffer>(solution.get_solution().release()),
                    solution.get_termination_status(),
                    solution.get_error_status().get_error_type(),
                    solution.get_error_status().what(),
                    solution.get_objective_value(),
                    solution.get_mip_gap(),
                    solution.get_solution_bound(),
                    solution.get_total_solve_time(),
                    solution.get_presolve_time(),
                    solution.get_max_constraint_violation(),
                    solution.get_max_int_violation(),
                    solution.get_max_variable_bound_violation(),
                    solution.get_num_nodes(),
                    solution.get_num_simplex_iterations()};
  return mip_ret;
}

std::unique_ptr<solver_ret_t> call_solve(
  cuopt::mps_parser::data_model_view_t<int, double>* data_model,
  cuopt::linear_programming::solver_settings_t<int, double>* solver_settings)
{
  raft::common::nvtx::range fun_scope("Call Solve");

  cudaStream_t stream;
  RAFT_CUDA_TRY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const raft::handle_t handle_{stream};

  auto op_problem = data_model_to_optimization_problem(data_model, solver_settings, &handle_);
  solver_ret_t response;
  if (op_problem.get_problem_category() == linear_programming::problem_category_t::LP) {
    response.lp_ret       = call_solve_lp(op_problem, solver_settings->get_pdlp_settings());
    response.problem_type = linear_programming::problem_category_t::LP;
  } else {
    response.mip_ret      = call_solve_mip(op_problem, solver_settings->get_mip_settings());
    response.problem_type = linear_programming::problem_category_t::MIP;
  }

  return std::make_unique<solver_ret_t>(std::move(response));
}

static int compute_max_thread(
  const std::vector<cuopt::mps_parser::data_model_view_t<int, double>*>& data_models)
{
  constexpr std::size_t max_total = 4;

  // Computing on the total_mem as LP is suppose to run on a single exclusive GPU
  std::size_t free_mem, total_mem;
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_mem, &total_mem));

  // Approximate the necessary memory for each problem
  std::size_t needed_memory = 0;
  for (const auto data_model : data_models) {
    const int nb_variables   = data_model->get_objective_coefficients().size();
    const int nb_constraints = data_model->get_constraint_bounds().size();
    // Currently we roughly need 8 times more memory than the size of each structure in the problem
    // representation
    needed_memory += ((nb_variables * 3 * sizeof(double)) + (nb_constraints * 3 * sizeof(double)) +
                      data_model->get_constraint_matrix_values().size() * sizeof(double) +
                      data_model->get_constraint_matrix_indices().size() * sizeof(int) +
                      data_model->get_constraint_matrix_offsets().size() * sizeof(int)) *
                     8;
  }

  const int res = std::min(max_total, std::min(total_mem / needed_memory, data_models.size()));
  cuopt_expects(
    res > 0, error_type_t::RuntimeError, "Problems too big to be solved in batch mode.");
  // A front end mecanism should prevent users to pick one or more problems so large that this would
  // return 0
  return res;
}

std::pair<std::vector<std::unique_ptr<solver_ret_t>>, double> call_batch_solve(
  std::vector<cuopt::mps_parser::data_model_view_t<int, double>*> data_models,
  cuopt::linear_programming::solver_settings_t<int, double>* solver_settings)
{
  raft::common::nvtx::range fun_scope("Call batch solve");

  const std::size_t size = data_models.size();

  std::vector<std::unique_ptr<solver_ret_t>> list(size);

  auto start_solver = std::chrono::high_resolution_clock::now();

  // Limit parallelism as too much stream overlap gets too slow
  const int max_thread = compute_max_thread(data_models);

  if (solver_settings->get_parameter<int>(CUOPT_METHOD) == CUOPT_METHOD_CONCURRENT) {
    CUOPT_LOG_INFO("Concurrent mode not supported for batch solve. Using PDLP instead. ");
    CUOPT_LOG_INFO(
      "Set the CUOPT_METHOD parameter to CUOPT_METHOD_PDLP or CUOPT_METHOD_DUAL_SIMPLEX to avoid "
      "this warning.");
    solver_settings->set_parameter(CUOPT_METHOD, CUOPT_METHOD_PDLP);
  }

#pragma omp parallel for num_threads(max_thread)
  for (std::size_t i = 0; i < size; ++i)
    list[i] = std::move(call_solve(data_models[i], solver_settings));

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_solver);

  return {std::move(list), duration.count() / 1000.0};
}

}  // namespace cython
}  // namespace cuopt
