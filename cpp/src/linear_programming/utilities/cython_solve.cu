/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_conversions.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <cuopt/linear_programming/utilities/cython_solve.hpp>
#include <cuopt/linear_programming/utilities/remote_solve.hpp>
#include <mip/logger.hpp>
#include <mps_parser/data_model_view.hpp>
#include <mps_parser/mps_data_model.hpp>
#include <mps_parser/writer.hpp>
#include <utilities/copy_helpers.hpp>

#include <raft/common/nvtx.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <utility>
#include <vector>

#include <unistd.h>

namespace cuopt {
namespace cython {

using cuopt::linear_programming::var_t;

// Note: data_model_to_optimization_problem() has been replaced with
// data_model_view_to_optimization_problem() from optimization_problem_conversions.hpp
// Warm start handling is now done inline where needed.

/**
 * @brief Wrapper for linear_programming to expose the API to cython
 *
 * @param data_model Composable data model object
 * @param solver_settings PDLP solver settings object
 * @return linear_programming_ret_t
 */
linear_programming_ret_t call_solve_lp(
  raft::handle_t const* handle_ptr,
  const cuopt::mps_parser::data_model_view_t<int, double>& view,
  cuopt::linear_programming::pdlp_solver_settings_t<int, double>& solver_settings,
  bool is_batch_mode)
{
  raft::common::nvtx::range fun_scope("Call Solve");

  // Validate that this is an LP problem (not MIP/IP)
  cuopt_expects(view.get_problem_category() == cuopt::linear_programming::problem_category_t::LP,
                cuopt::error_type_t::ValidationError,
                "LP solve cannot be called on a MIP problem!");

  const bool problem_checking     = true;
  const bool use_pdlp_solver_mode = true;

  // Call solver - handles remote/local branching and batch mode validation
  auto solution = cuopt::linear_programming::solve_lp(
    handle_ptr, view, solver_settings, problem_checking, use_pdlp_solver_mode, is_batch_mode);

  // Extract termination statistics (scalars - same for both device and host memory)
  linear_programming_ret_t lp_ret{};
  const auto& stats          = solution.get_additional_termination_information();
  auto& warm_start           = solution.get_pdlp_warm_start_data();
  lp_ret.termination_status_ = solution.get_termination_status();
  lp_ret.error_status_       = solution.get_error_status().get_error_type();
  lp_ret.error_message_      = solution.get_error_status().what();
  lp_ret.l2_primal_residual_ = stats.l2_primal_residual;
  lp_ret.l2_dual_residual_   = stats.l2_dual_residual;
  lp_ret.primal_objective_   = stats.primal_objective;
  lp_ret.dual_objective_     = stats.dual_objective;
  lp_ret.gap_                = stats.gap;
  lp_ret.nb_iterations_      = stats.number_of_steps_taken;
  lp_ret.solve_time_         = stats.solve_time;
  lp_ret.solved_by_pdlp_     = stats.solved_by_pdlp;

  // Extract warm-start data (scalars - same for both device and host memory)
  lp_ret.initial_primal_weight_         = warm_start.initial_primal_weight_;
  lp_ret.initial_step_size_             = warm_start.initial_step_size_;
  lp_ret.total_pdlp_iterations_         = warm_start.total_pdlp_iterations_;
  lp_ret.total_pdhg_iterations_         = warm_start.total_pdhg_iterations_;
  lp_ret.last_candidate_kkt_score_      = warm_start.last_candidate_kkt_score_;
  lp_ret.last_restart_kkt_score_        = warm_start.last_restart_kkt_score_;
  lp_ret.sum_solution_weight_           = warm_start.sum_solution_weight_;
  lp_ret.iterations_since_last_restart_ = warm_start.iterations_since_last_restart_;

  // Transfer solution data - either GPU or CPU depending on where it was solved
  if (solution.is_device_memory()) {
    // Local GPU solve: transfer device buffers
    lp_ret.primal_solution_ =
      std::make_unique<rmm::device_buffer>(solution.get_primal_solution().release());
    lp_ret.dual_solution_ =
      std::make_unique<rmm::device_buffer>(solution.get_dual_solution().release());
    lp_ret.reduced_cost_ =
      std::make_unique<rmm::device_buffer>(solution.get_reduced_cost().release());
    lp_ret.current_primal_solution_ =
      std::make_unique<rmm::device_buffer>(warm_start.current_primal_solution_.release());
    lp_ret.current_dual_solution_ =
      std::make_unique<rmm::device_buffer>(warm_start.current_dual_solution_.release());
    lp_ret.initial_primal_average_ =
      std::make_unique<rmm::device_buffer>(warm_start.initial_primal_average_.release());
    lp_ret.initial_dual_average_ =
      std::make_unique<rmm::device_buffer>(warm_start.initial_dual_average_.release());
    lp_ret.current_ATY_ = std::make_unique<rmm::device_buffer>(warm_start.current_ATY_.release());
    lp_ret.sum_primal_solutions_ =
      std::make_unique<rmm::device_buffer>(warm_start.sum_primal_solutions_.release());
    lp_ret.sum_dual_solutions_ =
      std::make_unique<rmm::device_buffer>(warm_start.sum_dual_solutions_.release());
    lp_ret.last_restart_duality_gap_primal_solution_ = std::make_unique<rmm::device_buffer>(
      warm_start.last_restart_duality_gap_primal_solution_.release());
    lp_ret.last_restart_duality_gap_dual_solution_ = std::make_unique<rmm::device_buffer>(
      warm_start.last_restart_duality_gap_dual_solution_.release());
    lp_ret.is_device_memory_ = true;
  } else {
    // Remote solve: use host vectors
    lp_ret.primal_solution_host_ = solution.get_primal_solution_host();
    lp_ret.dual_solution_host_   = solution.get_dual_solution_host();
    lp_ret.reduced_cost_host_    = solution.get_reduced_cost_host();
    lp_ret.is_device_memory_     = false;
  }

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
  raft::handle_t const* handle_ptr,
  const cuopt::mps_parser::data_model_view_t<int, double>& view,
  cuopt::linear_programming::mip_solver_settings_t<int, double>& solver_settings)
{
  raft::common::nvtx::range fun_scope("Call Solve");

  // Validate that this is a MIP or IP problem (not pure LP)
  cuopt_expects(
    (view.get_problem_category() == cuopt::linear_programming::problem_category_t::MIP) ||
      (view.get_problem_category() == cuopt::linear_programming::problem_category_t::IP),
    cuopt::error_type_t::ValidationError,
    "MIP solve cannot be called on an LP problem!");

  // Call solver - handles both remote and local solves
  // Remote: returns CPU solution, Local: returns GPU solution
  auto solution = cuopt::linear_programming::solve_mip(handle_ptr, view, solver_settings);

  // Extract solution statistics (scalars - same for both device and host memory)
  mip_ret_t mip_ret{};
  mip_ret.termination_status_           = solution.get_termination_status();
  mip_ret.error_status_                 = solution.get_error_status().get_error_type();
  mip_ret.error_message_                = solution.get_error_status().what();
  mip_ret.objective_                    = solution.get_objective_value();
  mip_ret.mip_gap_                      = solution.get_mip_gap();
  mip_ret.solution_bound_               = solution.get_solution_bound();
  mip_ret.total_solve_time_             = solution.get_total_solve_time();
  mip_ret.presolve_time_                = solution.get_presolve_time();
  mip_ret.max_constraint_violation_     = solution.get_max_constraint_violation();
  mip_ret.max_int_violation_            = solution.get_max_int_violation();
  mip_ret.max_variable_bound_violation_ = solution.get_max_variable_bound_violation();
  mip_ret.nodes_                        = solution.get_num_nodes();
  mip_ret.simplex_iterations_           = solution.get_num_simplex_iterations();

  // Transfer solution data - either GPU or CPU depending on where it was solved
  if (solution.is_device_memory()) {
    // Local GPU solve: transfer device buffer
    mip_ret.solution_ = std::make_unique<rmm::device_buffer>(solution.get_solution().release());
    mip_ret.is_device_memory_ = true;
  } else {
    // Remote solve: use host vector
    mip_ret.solution_host_    = solution.get_solution_host();
    mip_ret.is_device_memory_ = false;
  }
  return mip_ret;
}

std::unique_ptr<solver_ret_t> call_solve(
  cuopt::mps_parser::data_model_view_t<int, double>* data_model,
  cuopt::linear_programming::solver_settings_t<int, double>* solver_settings,
  unsigned int flags,
  bool is_batch_mode)
{
  raft::common::nvtx::range fun_scope("Call Solve");

  // Data from Python is always in CPU memory
  data_model->set_is_device_memory(false);

  // Determine if LP or MIP based on variable types (uses cached value)
  auto problem_category = data_model->get_problem_category();
  bool is_mip = (problem_category == cuopt::linear_programming::problem_category_t::MIP) ||
                (problem_category == cuopt::linear_programming::problem_category_t::IP);

  // Create handle for local solve (unused for remote solve)
  // Remote solve is detected by solve_lp/solve_mip via CUOPT_REMOTE_HOST/PORT env vars
  rmm::cuda_stream stream(static_cast<rmm::cuda_stream::flags>(flags));
  const raft::handle_t handle_{stream};
  const raft::handle_t* handle_ptr = &handle_;

  // Handle warm start data if present (only for local LP solves)
  if (!is_mip && solver_settings->get_pdlp_warm_start_data_view()
                     .last_restart_duality_gap_dual_solution_.data() != nullptr) {
    cuopt::linear_programming::pdlp_warm_start_data_t<int, double> pdlp_warm_start_data(
      solver_settings->get_pdlp_warm_start_data_view(), handle_.get_stream());
    solver_settings->get_pdlp_settings().set_pdlp_warm_start_data(pdlp_warm_start_data);
  }

  solver_ret_t response;

  if (!is_mip) {
    // LP solve
    response.lp_ret =
      call_solve_lp(handle_ptr, *data_model, solver_settings->get_pdlp_settings(), is_batch_mode);
    response.problem_type = linear_programming::problem_category_t::LP;
    if (response.lp_ret.is_device_memory_) {
      // Reset stream to per-thread default as non-blocking stream is out of scope after the
      // function returns.
      response.lp_ret.primal_solution_->set_stream(rmm::cuda_stream_per_thread);
      response.lp_ret.dual_solution_->set_stream(rmm::cuda_stream_per_thread);
      response.lp_ret.reduced_cost_->set_stream(rmm::cuda_stream_per_thread);
      response.lp_ret.current_primal_solution_->set_stream(rmm::cuda_stream_per_thread);
      response.lp_ret.current_dual_solution_->set_stream(rmm::cuda_stream_per_thread);
      response.lp_ret.initial_primal_average_->set_stream(rmm::cuda_stream_per_thread);
      response.lp_ret.initial_dual_average_->set_stream(rmm::cuda_stream_per_thread);
      response.lp_ret.current_ATY_->set_stream(rmm::cuda_stream_per_thread);
      response.lp_ret.sum_primal_solutions_->set_stream(rmm::cuda_stream_per_thread);
      response.lp_ret.sum_dual_solutions_->set_stream(rmm::cuda_stream_per_thread);
      response.lp_ret.last_restart_duality_gap_primal_solution_->set_stream(
        rmm::cuda_stream_per_thread);
      response.lp_ret.last_restart_duality_gap_dual_solution_->set_stream(
        rmm::cuda_stream_per_thread);
    }
  } else {
    // MIP solve
    response.mip_ret = call_solve_mip(handle_ptr, *data_model, solver_settings->get_mip_settings());
    response.problem_type = linear_programming::problem_category_t::MIP;
    if (response.mip_ret.is_device_memory_) {
      // Reset stream to per-thread default as non-blocking stream is out of scope after the
      // function returns.
      response.mip_ret.solution_->set_stream(rmm::cuda_stream_per_thread);
    }
  }

  // Reset warmstart data streams in solver_settings to per-thread default before destroying our
  // local stream. The warmstart data was created using our stream and its uvectors are associated
  // with it.
  auto& warmstart_data = solver_settings->get_pdlp_settings().get_pdlp_warm_start_data();
  if (warmstart_data.current_primal_solution_.size() > 0) {
    warmstart_data.current_primal_solution_.set_stream(rmm::cuda_stream_per_thread);
    warmstart_data.current_dual_solution_.set_stream(rmm::cuda_stream_per_thread);
    warmstart_data.initial_primal_average_.set_stream(rmm::cuda_stream_per_thread);
    warmstart_data.initial_dual_average_.set_stream(rmm::cuda_stream_per_thread);
    warmstart_data.current_ATY_.set_stream(rmm::cuda_stream_per_thread);
    warmstart_data.sum_primal_solutions_.set_stream(rmm::cuda_stream_per_thread);
    warmstart_data.sum_dual_solutions_.set_stream(rmm::cuda_stream_per_thread);
    warmstart_data.last_restart_duality_gap_primal_solution_.set_stream(
      rmm::cuda_stream_per_thread);
    warmstart_data.last_restart_duality_gap_dual_solution_.set_stream(rmm::cuda_stream_per_thread);
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
    // Currently we roughly need 8 times more memory than the size of each structure in the
    // problem representation
    needed_memory += ((nb_variables * 3 * sizeof(double)) + (nb_constraints * 3 * sizeof(double)) +
                      data_model->get_constraint_matrix_values().size() * sizeof(double) +
                      data_model->get_constraint_matrix_indices().size() * sizeof(int) +
                      data_model->get_constraint_matrix_offsets().size() * sizeof(int)) *
                     8;
  }

  const int res = std::min(max_total, std::min(total_mem / needed_memory, data_models.size()));
  cuopt_expects(
    res > 0, error_type_t::RuntimeError, "Problems too big to be solved in batch mode.");
  // A front end mecanism should prevent users to pick one or more problems so large that this
  // would return 0
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
  int max_thread = 1;
  if (linear_programming::is_remote_solve_enabled()) {
    // Cap parallelism for remote solve to avoid overwhelming the remote service.
    constexpr std::size_t max_total = 4;
    max_thread = static_cast<int>(std::min(std::max<std::size_t>(size, 1), max_total));
  } else {
    max_thread = compute_max_thread(data_models);
  }

  if (solver_settings->get_parameter<int>(CUOPT_METHOD) == CUOPT_METHOD_CONCURRENT) {
    CUOPT_LOG_INFO("Concurrent mode not supported for batch solve. Using PDLP instead. ");
    CUOPT_LOG_INFO(
      "Set the CUOPT_METHOD parameter to CUOPT_METHOD_PDLP or CUOPT_METHOD_DUAL_SIMPLEX to avoid "
      "this warning.");
    solver_settings->set_parameter(CUOPT_METHOD, CUOPT_METHOD_PDLP);
  }

  const bool is_batch_mode = true;

#pragma omp parallel for num_threads(max_thread)
  for (std::size_t i = 0; i < size; ++i)
    list[i] = call_solve(data_models[i], solver_settings, cudaStreamNonBlocking, is_batch_mode);

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_solver);

  return {std::move(list), duration.count() / 1000.0};
}

}  // namespace cython
}  // namespace cuopt
