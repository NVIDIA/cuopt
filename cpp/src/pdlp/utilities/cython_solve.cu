/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/backend_selection.hpp>
#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/gpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/optimization_problem_utils.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <cuopt/linear_programming/utilities/cython_solve.hpp>
#include <mip_heuristics/logger.hpp>
#include <mps_parser/data_model_view.hpp>
#include <mps_parser/mps_data_model.hpp>
#include <mps_parser/writer.hpp>
#include <utilities/copy_helpers.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>

#include <rmm/device_buffer.hpp>

#include <utility>
#include <vector>

#include <unistd.h>

namespace cuopt {
namespace cython {

/**
 * @brief Wrapper for linear_programming to expose the API to cython
 *
 * @param problem_interface Problem interface (GPU or CPU backend)
 * @param solver_settings PDLP solver settings object
 * @return lp_solution_interface_t pointer (raw pointer, caller owns)
 */
cuopt::linear_programming::lp_solution_interface_t<int, double>* call_solve_lp(
  cuopt::linear_programming::optimization_problem_interface_t<int, double>* problem_interface,
  cuopt::linear_programming::pdlp_solver_settings_t<int, double>& solver_settings,
  bool is_batch_mode)
{
  raft::common::nvtx::range fun_scope("Call Solve LP");
  cuopt_expects(
    problem_interface->get_problem_category() == cuopt::linear_programming::problem_category_t::LP,
    error_type_t::ValidationError,
    "LP solve cannot be called on a MIP problem!");
  const bool problem_checking     = true;
  const bool use_pdlp_solver_mode = true;

  // Solve returns unique_ptr<lp_solution_interface_t>
  auto solution_interface = cuopt::linear_programming::solve_lp(
    problem_interface, solver_settings, problem_checking, use_pdlp_solver_mode, is_batch_mode);

  // Return raw pointer (Python wrapper will own and manage lifecycle)
  return solution_interface.release();
}

/**
 * @brief Wrapper for linear_programming to expose the API to cython
 *
 * @param problem_interface Problem interface (GPU or CPU backend)
 * @param solver_settings MIP solver settings object
 * @return mip_solution_interface_t pointer (raw pointer, caller owns)
 */
cuopt::linear_programming::mip_solution_interface_t<int, double>* call_solve_mip(
  cuopt::linear_programming::optimization_problem_interface_t<int, double>* problem_interface,
  cuopt::linear_programming::mip_solver_settings_t<int, double>& solver_settings)
{
  raft::common::nvtx::range fun_scope("Call Solve MIP");
  cuopt_expects((problem_interface->get_problem_category() ==
                 cuopt::linear_programming::problem_category_t::MIP) or
                  (problem_interface->get_problem_category() ==
                   cuopt::linear_programming::problem_category_t::IP),
                error_type_t::ValidationError,
                "MIP solve cannot be called on an LP problem!");

  // Solve returns unique_ptr<mip_solution_interface_t>
  auto solution_interface =
    cuopt::linear_programming::solve_mip(problem_interface, solver_settings);

  // Return raw pointer (Python wrapper will own and manage lifecycle)
  return solution_interface.release();
}

std::unique_ptr<solver_ret_t> call_solve(
  cuopt::mps_parser::data_model_view_t<int, double>* data_model,
  cuopt::linear_programming::solver_settings_t<int, double>* solver_settings,
  unsigned int flags,
  bool is_batch_mode)
{
  raft::common::nvtx::range fun_scope("Call Solve");

  // Determine memory backend based on execution mode
  auto memory_backend = cuopt::linear_programming::get_memory_backend_type();

  solver_ret_t response;

  // Create problem instance and CUDA resources based on memory backend
  if (memory_backend == cuopt::linear_programming::memory_backend_t::GPU) {
    // GPU memory backend: Create CUDA resources and GPU problem
    rmm::cuda_stream stream(static_cast<rmm::cuda_stream::flags>(flags));
    const raft::handle_t handle_{stream};

    auto gpu_problem = cuopt::linear_programming::gpu_optimization_problem_t<int, double>(&handle_);
    cuopt::linear_programming::populate_from_data_model_view(
      &gpu_problem, data_model, solver_settings, &handle_);

    // Call appropriate solve function and convert to ret struct
    if (gpu_problem.get_problem_category() == linear_programming::problem_category_t::LP) {
      // Solve and get solution interface pointer
      auto lp_solution_ptr =
        std::unique_ptr<linear_programming::lp_solution_interface_t<int, double>>(
          call_solve_lp(&gpu_problem, solver_settings->get_pdlp_settings(), is_batch_mode));

      response.lp_ret       = lp_solution_ptr->to_python_lp_ret();
      response.problem_type = linear_programming::problem_category_t::LP;

      // The solve's local stream is destroyed when this function returns, so reassociate
      // all returned device_buffers with a long-lived stream for safe deallocation later.
      auto& gpu_sols =
        std::get<linear_programming_ret_t::gpu_solutions_t>(response.lp_ret.solutions_);
      gpu_sols.primal_solution_->set_stream(rmm::cuda_stream_per_thread);
      gpu_sols.dual_solution_->set_stream(rmm::cuda_stream_per_thread);
      gpu_sols.reduced_cost_->set_stream(rmm::cuda_stream_per_thread);
      gpu_sols.current_primal_solution_->set_stream(rmm::cuda_stream_per_thread);
      gpu_sols.current_dual_solution_->set_stream(rmm::cuda_stream_per_thread);
      gpu_sols.initial_primal_average_->set_stream(rmm::cuda_stream_per_thread);
      gpu_sols.initial_dual_average_->set_stream(rmm::cuda_stream_per_thread);
      gpu_sols.current_ATY_->set_stream(rmm::cuda_stream_per_thread);
      gpu_sols.sum_primal_solutions_->set_stream(rmm::cuda_stream_per_thread);
      gpu_sols.sum_dual_solutions_->set_stream(rmm::cuda_stream_per_thread);
      gpu_sols.last_restart_duality_gap_primal_solution_->set_stream(rmm::cuda_stream_per_thread);
      gpu_sols.last_restart_duality_gap_dual_solution_->set_stream(rmm::cuda_stream_per_thread);

    } else {
      // MIP solve
      auto mip_solution_ptr =
        std::unique_ptr<linear_programming::mip_solution_interface_t<int, double>>(
          call_solve_mip(&gpu_problem, solver_settings->get_mip_settings()));

      response.mip_ret      = mip_solution_ptr->to_python_mip_ret();
      response.problem_type = linear_programming::problem_category_t::MIP;

      // Same stream reassociation as the LP path above.
      auto& gpu_sol = std::get<gpu_buffer>(response.mip_ret.solution_);
      gpu_sol->set_stream(rmm::cuda_stream_per_thread);
    }

    // Reset warmstart data streams in solver_settings (skip in batch mode to avoid data race
    // on the shared solver_settings object accessed concurrently by multiple threads)
    if (!is_batch_mode) {
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
        warmstart_data.last_restart_duality_gap_dual_solution_.set_stream(
          rmm::cuda_stream_per_thread);
      }
    }

  } else {
    // CPU memory backend: No CUDA resources needed at problem-creation time
    auto cpu_problem = cuopt::linear_programming::cpu_optimization_problem_t<int, double>(nullptr);
    cuopt::linear_programming::populate_from_data_model_view(
      &cpu_problem, data_model, solver_settings, nullptr);

    // Call appropriate solve function and convert to ret struct
    if (cpu_problem.get_problem_category() == linear_programming::problem_category_t::LP) {
      auto lp_solution_ptr =
        std::unique_ptr<linear_programming::lp_solution_interface_t<int, double>>(
          call_solve_lp(&cpu_problem, solver_settings->get_pdlp_settings(), is_batch_mode));

      response.lp_ret       = lp_solution_ptr->to_python_lp_ret();
      response.problem_type = linear_programming::problem_category_t::LP;

    } else {
      auto mip_solution_ptr =
        std::unique_ptr<linear_programming::mip_solution_interface_t<int, double>>(
          call_solve_mip(&cpu_problem, solver_settings->get_mip_settings()));

      response.mip_ret      = mip_solution_ptr->to_python_mip_ret();
      response.problem_type = linear_programming::problem_category_t::MIP;
    }
  }

  return std::make_unique<solver_ret_t>(std::move(response));
}

static int compute_max_thread(
  const std::vector<cuopt::mps_parser::data_model_view_t<int, double>*>& data_models)
{
  constexpr std::size_t max_total = 4;

  // Computing on the total_mem as LP is suppose to run on a single exclusive GPU
  // On CPU-only hosts cudaMemGetInfo will fail; fall back to single-threaded batch.
  std::size_t free_mem, total_mem;
  auto cuda_err = cudaMemGetInfo(&free_mem, &total_mem);
  if (cuda_err != cudaSuccess) {
    cudaGetLastError();  // clear the error
    return 1;
  }

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

std::pair<std::vector<std::unique_ptr<solver_ret_t>>, double> solve_batch_remote(
  std::vector<cuopt::mps_parser::data_model_view_t<int, double>*> data_models,
  cuopt::linear_programming::solver_settings_t<int, double>* solver_settings)
{
  cuopt_expects(
    false,
    error_type_t::RuntimeError,
    "Remote batch solve is not yet implemented. "
    "Please use local batch solve or solve problems individually via remote execution.");
  return {};
}

std::pair<std::vector<std::unique_ptr<solver_ret_t>>, double> call_batch_solve(
  std::vector<cuopt::mps_parser::data_model_view_t<int, double>*> data_models,
  cuopt::linear_programming::solver_settings_t<int, double>* solver_settings)
{
  raft::common::nvtx::range fun_scope("Call batch solve");

  if (cuopt::linear_programming::is_remote_execution_enabled()) {
    return solve_batch_remote(data_models, solver_settings);
  }

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
