/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <linear_programming/pdlp.cuh>
#include <linear_programming/utils.cuh>
#include <mip/mip_constants.hpp>
#include "cuopt/linear_programming/pdlp/solver_solution.hpp"

#include <raft/common/nvtx.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/ternary_op.cuh>

#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>

#include <thrust/count.h>
#include <thrust/extrema.h>

#include <optional>

namespace cuopt::linear_programming::detail {

void set_pdlp_hyper_parameters(rmm::cuda_stream_view stream_view)
{
  RAFT_CUDA_TRY(cudaMemcpyToSymbolAsync(pdlp_hyper_params::primal_importance,
                                        &pdlp_hyper_params::host_primal_importance,
                                        sizeof(double),
                                        0,
                                        cudaMemcpyHostToDevice,
                                        stream_view));
}

template <typename i_t, typename f_t>
pdlp_solver_t<i_t, f_t>::pdlp_solver_t(problem_t<i_t, f_t>& op_problem,
                                       pdlp_solver_settings_t<i_t, f_t> const& settings)
  : handle_ptr_(op_problem.handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    problem_ptr(&op_problem),
    op_problem_scaled_(
      op_problem, false),  // False to call the PDLP custom version of the problem copy constructor
    unscaled_primal_avg_solution_{static_cast<size_t>(op_problem.n_variables), stream_view_},
    unscaled_dual_avg_solution_{static_cast<size_t>(op_problem.n_constraints), stream_view_},
    primal_size_h_(op_problem.n_variables),
    dual_size_h_(op_problem.n_constraints),
    primal_step_size_{stream_view_},
    dual_step_size_{stream_view_},
    primal_weight_{stream_view_},
    step_size_{(f_t)pdlp_hyper_params::initial_step_size_scaling, stream_view_},
    step_size_strategy_{handle_ptr_, &primal_weight_, &step_size_},
    pdhg_solver_{handle_ptr_, op_problem_scaled_},
    settings_(settings, stream_view_),
    initial_scaling_strategy_{handle_ptr_,
                              op_problem_scaled_,
                              pdlp_hyper_params::default_l_inf_ruiz_iterations,
                              (f_t)pdlp_hyper_params::default_alpha_pock_chambolle_rescaling,
                              pdhg_solver_,
                              op_problem_scaled_.reverse_coefficients,
                              op_problem_scaled_.reverse_offsets,
                              op_problem_scaled_.reverse_constraints},
    average_op_problem_evaluation_cusparse_view_{handle_ptr_,
                                                 op_problem,
                                                 unscaled_primal_avg_solution_,
                                                 unscaled_dual_avg_solution_,
                                                 pdhg_solver_.get_primal_tmp_resource(),
                                                 pdhg_solver_.get_dual_tmp_resource(),
                                                 op_problem.reverse_coefficients,
                                                 op_problem.reverse_offsets,
                                                 op_problem.reverse_constraints},
    current_op_problem_evaluation_cusparse_view_{handle_ptr_,
                                                 op_problem,
                                                 pdhg_solver_.get_primal_solution(),
                                                 pdhg_solver_.get_dual_solution(),
                                                 pdhg_solver_.get_primal_tmp_resource(),
                                                 pdhg_solver_.get_dual_tmp_resource(),
                                                 op_problem.reverse_coefficients,
                                                 op_problem.reverse_offsets,
                                                 op_problem.reverse_constraints},
    restart_strategy_{handle_ptr_,
                      op_problem,
                      average_op_problem_evaluation_cusparse_view_,
                      primal_size_h_,
                      dual_size_h_},
    average_termination_strategy_{handle_ptr_,
                                  op_problem,
                                  average_op_problem_evaluation_cusparse_view_,
                                  primal_size_h_,
                                  dual_size_h_,
                                  settings_},
    current_termination_strategy_{handle_ptr_,
                                  op_problem,
                                  current_op_problem_evaluation_cusparse_view_,
                                  primal_size_h_,
                                  dual_size_h_,
                                  settings_},
    initial_primal_{0, stream_view_},
    initial_dual_{0, stream_view_},

    best_primal_solution_so_far{pdlp_termination_status_t::TimeLimit, stream_view_},
    inside_mip_{false}
{
  if (settings.has_initial_primal_solution()) {
    auto& primal_sol = settings.get_initial_primal_solution();
    set_initial_primal_solution(primal_sol);
  }
  if (settings.has_initial_dual_solution()) {
    const auto& dual_sol = settings.get_initial_dual_solution();
    set_initial_dual_solution(dual_sol);
  }

  if (settings.get_pdlp_warm_start_data().last_restart_duality_gap_dual_solution_.size() != 0) {
    set_initial_primal_solution(settings.get_pdlp_warm_start_data().current_primal_solution_);
    set_initial_dual_solution(settings.get_pdlp_warm_start_data().current_dual_solution_);
    initial_step_size_     = settings.get_pdlp_warm_start_data().initial_step_size_;
    initial_primal_weight_ = settings.get_pdlp_warm_start_data().initial_primal_weight_;
    total_pdlp_iterations_ = settings.get_pdlp_warm_start_data().total_pdlp_iterations_;
    pdhg_solver_.total_pdhg_iterations_ =
      settings.get_pdlp_warm_start_data().total_pdhg_iterations_;
    pdhg_solver_.get_d_total_pdhg_iterations().set_value_async(
      settings.get_pdlp_warm_start_data().total_pdhg_iterations_, stream_view_);
    restart_strategy_.last_candidate_kkt_score =
      settings.get_pdlp_warm_start_data().last_candidate_kkt_score_;
    restart_strategy_.last_restart_kkt_score =
      settings.get_pdlp_warm_start_data().last_restart_kkt_score_;
    raft::copy(restart_strategy_.weighted_average_solution_.sum_primal_solutions_.data(),
               settings.get_pdlp_warm_start_data().sum_primal_solutions_.data(),
               settings.get_pdlp_warm_start_data().sum_primal_solutions_.size(),
               stream_view_);
    raft::copy(restart_strategy_.weighted_average_solution_.sum_dual_solutions_.data(),
               settings.get_pdlp_warm_start_data().sum_dual_solutions_.data(),
               settings.get_pdlp_warm_start_data().sum_dual_solutions_.size(),
               stream_view_);
    raft::copy(unscaled_primal_avg_solution_.data(),
               settings.get_pdlp_warm_start_data().initial_primal_average_.data(),
               settings.get_pdlp_warm_start_data().initial_primal_average_.size(),
               stream_view_);
    raft::copy(unscaled_dual_avg_solution_.data(),
               settings.get_pdlp_warm_start_data().initial_dual_average_.data(),
               settings.get_pdlp_warm_start_data().initial_dual_average_.size(),
               stream_view_);
    raft::copy(pdhg_solver_.get_saddle_point_state().get_current_AtY().data(),
               settings.get_pdlp_warm_start_data().current_ATY_.data(),
               settings.get_pdlp_warm_start_data().current_ATY_.size(),
               stream_view_);
    raft::copy(restart_strategy_.last_restart_duality_gap_.primal_solution_.data(),
               settings.get_pdlp_warm_start_data().last_restart_duality_gap_primal_solution_.data(),
               settings.get_pdlp_warm_start_data().last_restart_duality_gap_primal_solution_.size(),
               stream_view_);
    raft::copy(restart_strategy_.last_restart_duality_gap_.dual_solution_.data(),
               settings.get_pdlp_warm_start_data().last_restart_duality_gap_dual_solution_.data(),
               settings.get_pdlp_warm_start_data().last_restart_duality_gap_dual_solution_.size(),
               stream_view_);

    const auto value = settings.get_pdlp_warm_start_data().sum_solution_weight_;
    restart_strategy_.weighted_average_solution_.sum_primal_solution_weights_.set_value_async(
      value, stream_view_);
    restart_strategy_.weighted_average_solution_.sum_dual_solution_weights_.set_value_async(
      value, stream_view_);
    restart_strategy_.weighted_average_solution_.iterations_since_last_restart_ =
      settings.get_pdlp_warm_start_data().iterations_since_last_restart_;
  }
  // Checks performed below are assert only
  best_primal_quality_so_far_.primal_objective = (op_problem_scaled_.maximize)
                                                   ? -std::numeric_limits<f_t>::infinity()
                                                   : std::numeric_limits<f_t>::infinity();
  op_problem.check_problem_representation(true, false);
  op_problem_scaled_.check_problem_representation(true, false);
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::set_initial_primal_weight(f_t initial_primal_weight)
{
  initial_primal_weight_ = initial_primal_weight;
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::set_initial_step_size(f_t initial_step_size)
{
  initial_step_size_ = initial_step_size;
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::set_initial_k(i_t initial_k)
{
  initial_k_ = initial_k;
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::set_relative_dual_tolerance_factor(f_t dual_tolerance_factor)
{
  average_termination_strategy_.set_relative_dual_tolerance_factor(dual_tolerance_factor);
  current_termination_strategy_.set_relative_dual_tolerance_factor(dual_tolerance_factor);
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::set_relative_primal_tolerance_factor(f_t primal_tolerance_factor)
{
  average_termination_strategy_.set_relative_primal_tolerance_factor(primal_tolerance_factor);
  current_termination_strategy_.set_relative_primal_tolerance_factor(primal_tolerance_factor);
}

template <typename i_t, typename f_t>
f_t pdlp_solver_t<i_t, f_t>::get_relative_dual_tolerance_factor() const
{
  return current_termination_strategy_.get_relative_dual_tolerance_factor();
}

template <typename i_t, typename f_t>
f_t pdlp_solver_t<i_t, f_t>::get_relative_primal_tolerance_factor() const
{
  return current_termination_strategy_.get_relative_primal_tolerance_factor();
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::set_initial_primal_solution(
  const rmm::device_uvector<f_t>& initial_primal_solution)
{
  initial_primal_.resize(initial_primal_solution.size(), stream_view_);
  raft::copy(initial_primal_.data(),
             initial_primal_solution.data(),
             initial_primal_solution.size(),
             stream_view_);
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::set_initial_dual_solution(
  const rmm::device_uvector<f_t>& initial_dual_solution)
{
  initial_dual_.resize(initial_dual_solution.size(), stream_view_);
  raft::copy(
    initial_dual_.data(), initial_dual_solution.data(), initial_dual_solution.size(), stream_view_);
}

static bool time_limit_reached(const std::chrono::high_resolution_clock::time_point& start_time,
                               double seconds)
{
  auto current_time = std::chrono::high_resolution_clock::now();
  auto elapsed =
    std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();

  return elapsed >= (seconds * 1000.0);
}

template <typename i_t, typename f_t>
std::optional<optimization_problem_solution_t<i_t, f_t>> pdlp_solver_t<i_t, f_t>::check_limits(
  const std::chrono::high_resolution_clock::time_point& start_time)
{
  // Check for time limit
  if (time_limit_reached(start_time, settings_.time_limit)) {
    if (settings_.save_best_primal_so_far) {
#ifdef PDLP_VERBOSE_MODE
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      std::cout << "Time Limit reached, returning best primal so far" << std::endl;
#endif
      return std::move(best_primal_solution_so_far);
    }
#ifdef PDLP_VERBOSE_MODE
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    std::cout << "Time Limit reached, returning current solution" << std::endl;
#endif
    return current_termination_strategy_.fill_return_problem_solution(
      internal_solver_iterations_,
      pdhg_solver_,
      pdhg_solver_.get_primal_solution(),
      pdhg_solver_.get_dual_solution(),
      get_filled_warmed_start_data(),
      pdlp_termination_status_t::TimeLimit);
  }

  // Check for iteration limit
  if (internal_solver_iterations_ >= settings_.iteration_limit) {
    if (settings_.save_best_primal_so_far) {
#ifdef PDLP_VERBOSE_MODE
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      std::cout << "Iteration Limit reached, returning best primal so far" << std::endl;
#endif
      best_primal_solution_so_far.set_termination_status(pdlp_termination_status_t::IterationLimit);
      return std::move(best_primal_solution_so_far);
    }
#ifdef PDLP_VERBOSE_MODE
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    std::cout << "Iteration Limit reached, returning current solution" << std::endl;
#endif

    return current_termination_strategy_.fill_return_problem_solution(
      internal_solver_iterations_,
      pdhg_solver_,
      pdhg_solver_.get_primal_solution(),
      pdhg_solver_.get_dual_solution(),
      get_filled_warmed_start_data(),
      pdlp_termination_status_t::IterationLimit);
  }

  // Check for concurrent limit
  if (settings_.concurrent_halt != nullptr &&
      settings_.concurrent_halt->load(std::memory_order_acquire) == 1) {
#ifdef PDLP_VERBOSE_MODE
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    std::cout << "Concurrent Limit reached, returning current solution" << std::endl;
#endif
    return current_termination_strategy_.fill_return_problem_solution(
      internal_solver_iterations_,
      pdhg_solver_,
      pdhg_solver_.get_primal_solution(),
      pdhg_solver_.get_dual_solution(),
      get_filled_warmed_start_data(),
      pdlp_termination_status_t::ConcurrentLimit);
  }

  return std::nullopt;
}

// True if current has a better primal objective value based on if minimize or maximize
template <typename f_t>
static bool is_current_objective_better(f_t current_primal_objective,
                                        f_t other_primal_objective,
                                        bool maximize)
{
  const bool current_is_lower = current_primal_objective < other_primal_objective;
  return (!maximize && current_is_lower) || (maximize && !current_is_lower);
}

// Returns the solution with the best quality
template <typename i_t, typename f_t>
const pdlp_solver_t<i_t, f_t>::primal_quality_adapter_t& pdlp_solver_t<i_t, f_t>::get_best_quality(
  const pdlp_solver_t<i_t, f_t>::primal_quality_adapter_t& current,
  const pdlp_solver_t<i_t, f_t>::primal_quality_adapter_t& other)
{
#ifdef PDLP_DEBUG_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "  current.is_primal_feasible = " << current.is_primal_feasible << std::endl;
  std::cout << "  current.nb_violated_constraints = " << current.nb_violated_constraints
            << std::endl;
  std::cout << "  current.primal_residual = " << current.primal_residual << std::endl;
  std::cout << "  current.primal_objective = " << current.primal_objective << std::endl;
  std::cout << "  other.is_primal_feasible = " << other.is_primal_feasible << std::endl;
  std::cout << "  other.nb_violated_constraints = " << other.nb_violated_constraints << std::endl;
  std::cout << "  other.primal_residual = " << other.primal_residual << std::endl;
  std::cout << "  other.primal_objective = " << other.primal_objective << std::endl;
#endif

  // Primal feasiblity is best

  if (current.is_primal_feasible && !other.is_primal_feasible)
    return current;
  else if (!current.is_primal_feasible && other.is_primal_feasible)
    return other;
  else if (current.is_primal_feasible && other.is_primal_feasible) {
    // Then objective is best
    const bool current_objective_is_better = is_current_objective_better(
      current.primal_objective, other.primal_objective, op_problem_scaled_.maximize);
    return (current_objective_is_better) ? current : other;
  }

  // Both are not primal feasible

  // Prioritize least overall residual
  if (current.primal_residual < other.primal_residual)
    return current;
  else
    return other;
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::set_inside_mip(bool inside_mip)
{
  inside_mip_ = inside_mip;
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::record_best_primal_so_far(
  const detail::pdlp_termination_strategy_t<i_t, f_t>& current,
  const detail::pdlp_termination_strategy_t<i_t, f_t>& average,
  const pdlp_termination_status_t& termination_current,
  const pdlp_termination_status_t& termination_average)
{
#ifdef PDLP_VERBOSE_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "Recording best primal so far" << std::endl;
#endif

  // As this point, neither current or average are pdlp_termination_status_t::Optimal, else they
  // would have been returned
  cuopt_assert(termination_current != pdlp_termination_status_t::Optimal,
               "Solution can't be pdlp_termination_status_t::Optimal at this point");
  cuopt_assert(termination_average != pdlp_termination_status_t::Optimal,
               "Solution can't be pdlp_termination_status_t::Optimal at this point");

  // First find best between current and average

  const auto& current_quality = current.get_convergence_information().to_primal_quality_adapter(
    termination_current == pdlp_termination_status_t::PrimalFeasible);
  const auto& average_quality = average.get_convergence_information().to_primal_quality_adapter(
    termination_average == pdlp_termination_status_t::PrimalFeasible);
  const auto& best_candidate = get_best_quality(current_quality, average_quality);

  // Then best between last and best_candidate

  const auto& best_overall = get_best_quality(best_candidate, best_primal_quality_so_far_);

  // Best overall is different (better) than last found
  if (best_overall != best_primal_quality_so_far_) {
#ifdef PDLP_DEBUG_MODE
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    std::cout << "New best primal found" << std::endl;
#endif
    best_primal_quality_so_far_ = best_overall;

    // Record the new solution

    rmm::device_uvector<f_t>* primal_to_set;
    rmm::device_uvector<f_t>* dual_to_set;
    detail::pdlp_termination_strategy_t<i_t, f_t>* termination_strategy_to_use;
    std::string_view debug_string;

    if (best_overall == current_quality) {
      primal_to_set               = &pdhg_solver_.get_primal_solution();
      dual_to_set                 = &pdhg_solver_.get_dual_solution();
      termination_strategy_to_use = &current_termination_strategy_;
      debug_string                = "  current is better";
    } else {
      primal_to_set               = &unscaled_primal_avg_solution_;
      dual_to_set                 = &unscaled_dual_avg_solution_;
      termination_strategy_to_use = &average_termination_strategy_;
      debug_string                = "  average is better";
    }

#ifdef PDLP_DEBUG_MODE
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    std::cout << debug_string << std::endl;
#endif

    best_primal_solution_so_far = termination_strategy_to_use->fill_return_problem_solution(
      internal_solver_iterations_,
      pdhg_solver_,
      *primal_to_set,
      *dual_to_set,
      pdlp_termination_status_t::TimeLimit,
      true);
  } else {
#ifdef PDLP_DEBUG_MODE
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    std::cout << "Last best primal is still best" << std::endl;
#endif
  }
}

template <typename i_t, typename f_t>
pdlp_warm_start_data_t<i_t, f_t> pdlp_solver_t<i_t, f_t>::get_filled_warmed_start_data()
{
  return pdlp_warm_start_data_t<i_t, f_t>(
    pdhg_solver_.get_primal_solution(),
    pdhg_solver_.get_dual_solution(),
    unscaled_primal_avg_solution_,
    unscaled_dual_avg_solution_,
    pdhg_solver_.get_saddle_point_state().get_current_AtY(),
    restart_strategy_.weighted_average_solution_.sum_primal_solutions_,
    restart_strategy_.weighted_average_solution_.sum_dual_solutions_,
    restart_strategy_.last_restart_duality_gap_.primal_solution_,
    restart_strategy_.last_restart_duality_gap_.dual_solution_,
    get_primal_weight_h(),
    get_step_size_h(),
    total_pdlp_iterations_,
    pdhg_solver_.total_pdhg_iterations_,
    restart_strategy_.last_candidate_kkt_score,
    restart_strategy_.last_restart_kkt_score,
    restart_strategy_.weighted_average_solution_.sum_primal_solution_weights_.value(stream_view_),
    restart_strategy_.weighted_average_solution_.iterations_since_last_restart_);
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::print_termination_criteria(
  const std::chrono::high_resolution_clock::time_point& start_time, bool is_average)
{
  if (!inside_mip_) {
    const auto current_time = std::chrono::high_resolution_clock::now();
    const f_t elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count() /
      1000.0;
    if (is_average) {
      average_termination_strategy_.print_termination_criteria(total_pdlp_iterations_, elapsed);
    } else {
      current_termination_strategy_.print_termination_criteria(total_pdlp_iterations_, elapsed);
    }
  }
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::print_final_termination_criteria(
  const std::chrono::high_resolution_clock::time_point& start_time,
  const convergence_information_t<i_t, f_t>& convergence_information,
  const pdlp_termination_status_t& termination_status,
  bool is_average)
{
  if (!inside_mip_) {
    print_termination_criteria(start_time, is_average);
    CUOPT_LOG_INFO(
      "LP Solver status:                %s",
      optimization_problem_solution_t<i_t, f_t>::get_termination_status_string(termination_status)
        .c_str());
    CUOPT_LOG_INFO("Primal objective:                %+.8e",
                   convergence_information.get_primal_objective().value(stream_view_));
    CUOPT_LOG_INFO("Dual objective:                  %+.8e",
                   convergence_information.get_dual_objective().value(stream_view_));
    CUOPT_LOG_INFO("Duality gap (abs/rel):           %+.2e / %+.2e",
                   convergence_information.get_gap().value(stream_view_),
                   convergence_information.get_relative_gap_value());
    CUOPT_LOG_INFO("Primal infeasibility (abs/rel):  %+.2e / %+.2e",
                   convergence_information.get_l2_primal_residual().value(stream_view_),
                   convergence_information.get_relative_l2_primal_residual_value());
    CUOPT_LOG_INFO("Dual infeasibility (abs/rel):    %+.2e / %+.2e",
                   convergence_information.get_l2_dual_residual().value(stream_view_),
                   convergence_information.get_relative_l2_dual_residual_value());
  }
}

template <typename i_t, typename f_t>
std::optional<optimization_problem_solution_t<i_t, f_t>> pdlp_solver_t<i_t, f_t>::check_termination(
  const std::chrono::high_resolution_clock::time_point& start_time)
{
  raft::common::nvtx::range fun_scope("Check termination");

  // Still need to always compute the termination condition for current even if we don't check them
  // after for kkt restart
#ifdef PDLP_VERBOSE_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  printf("Termination criteria current\n");
  current_termination_strategy_.print_termination_criteria();
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
#endif
  pdlp_termination_status_t termination_current =
    current_termination_strategy_.evaluate_termination_criteria(
      pdhg_solver_,
      pdhg_solver_.get_primal_solution(),
      pdhg_solver_.get_dual_solution(),
      problem_ptr->combined_bounds,
      problem_ptr->objective_coefficients);

#ifdef PDLP_VERBOSE_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "Termination criteria average:" << std::endl;
  average_termination_strategy_.print_termination_criteria();
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
#endif

  // Check both average and current solution
  pdlp_termination_status_t termination_average =
    average_termination_strategy_.evaluate_termination_criteria(
      pdhg_solver_,
      unscaled_primal_avg_solution_,
      unscaled_dual_avg_solution_,
      problem_ptr->combined_bounds,
      problem_ptr->objective_coefficients);

  // We exit directly without checking the termination criteria as some problem can have a low
  // initial redidual + there is by definition 0 gap at first
  // To avoid that we allow at least two iterations at first before checking (in practice 0 wasn't
  // enough) We still need to check iteration and time limit prior without breaking the logic below
  // of first checking termination before the limit
  if (total_pdlp_iterations_ <= 1) {
    print_termination_criteria(start_time);
    return check_limits(start_time);
  }

  // First check for pdlp_termination_reason_t::Optimality and handle the first primal feasible case

  if (settings_.first_primal_feasible) {
    // Both primal feasible, return best objective
    if (termination_average == pdlp_termination_status_t::PrimalFeasible &&
        termination_current == pdlp_termination_status_t::PrimalFeasible) {
      const f_t current_overall_primal_residual =
        current_termination_strategy_.get_convergence_information().get_l2_primal_residual().value(
          stream_view_);
      const f_t average_overall_primal_residual =
        average_termination_strategy_.get_convergence_information().get_l2_primal_residual().value(
          stream_view_);
      if (current_overall_primal_residual < average_overall_primal_residual) {
        return current_termination_strategy_.fill_return_problem_solution(
          internal_solver_iterations_,
          pdhg_solver_,
          pdhg_solver_.get_primal_solution(),
          pdhg_solver_.get_dual_solution(),
          get_filled_warmed_start_data(),
          termination_current);
      } else  // Average has better overall residual
      {
        return average_termination_strategy_.fill_return_problem_solution(
          internal_solver_iterations_,
          pdhg_solver_,
          unscaled_primal_avg_solution_,
          unscaled_dual_avg_solution_,
          get_filled_warmed_start_data(),
          termination_average);
      }
    } else if (termination_current == pdlp_termination_status_t::PrimalFeasible) {
      return current_termination_strategy_.fill_return_problem_solution(
        internal_solver_iterations_,
        pdhg_solver_,
        pdhg_solver_.get_primal_solution(),
        pdhg_solver_.get_dual_solution(),
        get_filled_warmed_start_data(),
        termination_current);
    } else if (termination_average == pdlp_termination_status_t::PrimalFeasible) {
      return average_termination_strategy_.fill_return_problem_solution(
        internal_solver_iterations_,
        pdhg_solver_,
        unscaled_primal_avg_solution_,
        unscaled_dual_avg_solution_,
        get_filled_warmed_start_data(),
        termination_average);
    }
    // else neither of the two is primal feasible
  }

  // If both are pdlp_termination_status_t::Optimal, return the one with the lowest KKT score
  if (termination_average == pdlp_termination_status_t::Optimal &&
      termination_current == pdlp_termination_status_t::Optimal) {
    const f_t current_kkt_score = restart_strategy_.compute_kkt_score(
      current_termination_strategy_.get_convergence_information().get_l2_primal_residual(),
      current_termination_strategy_.get_convergence_information().get_l2_dual_residual(),
      current_termination_strategy_.get_convergence_information().get_gap(),
      primal_weight_);

    const f_t average_kkt_score = restart_strategy_.compute_kkt_score(
      average_termination_strategy_.get_convergence_information().get_l2_primal_residual(),
      average_termination_strategy_.get_convergence_information().get_l2_dual_residual(),
      average_termination_strategy_.get_convergence_information().get_gap(),
      primal_weight_);

    if (current_kkt_score < average_kkt_score) {
#ifdef PDLP_VERBOSE_MODE
      std::cout << "Optimal. End total number of iteration current=" << internal_solver_iterations_
                << std::endl;
#endif
      print_final_termination_criteria(start_time,
                                       current_termination_strategy_.get_convergence_information(),
                                       termination_current);
      return current_termination_strategy_.fill_return_problem_solution(
        internal_solver_iterations_,
        pdhg_solver_,
        pdhg_solver_.get_primal_solution(),
        pdhg_solver_.get_dual_solution(),
        get_filled_warmed_start_data(),
        termination_current);
    } else {
#ifdef PDLP_VERBOSE_MODE
      std::cout << "Optimal. End total number of iteration average=" << internal_solver_iterations_
                << std::endl;
#endif
      print_final_termination_criteria(start_time,
                                       average_termination_strategy_.get_convergence_information(),
                                       termination_average,
                                       true);
      return average_termination_strategy_.fill_return_problem_solution(
        internal_solver_iterations_,
        pdhg_solver_,
        unscaled_primal_avg_solution_,
        unscaled_dual_avg_solution_,
        get_filled_warmed_start_data(),
        termination_average);
    }
  }

  // If at least one is pdlp_termination_status_t::Optimal, return it
  if (termination_average == pdlp_termination_status_t::Optimal) {
#ifdef PDLP_VERBOSE_MODE
    std::cout << "Optimal. End total number of iteration average=" << internal_solver_iterations_
              << std::endl;
#endif
    print_final_termination_criteria(start_time,
                                     average_termination_strategy_.get_convergence_information(),
                                     termination_average,
                                     true);
    return average_termination_strategy_.fill_return_problem_solution(
      internal_solver_iterations_,
      pdhg_solver_,
      unscaled_primal_avg_solution_,
      unscaled_dual_avg_solution_,
      get_filled_warmed_start_data(),
      termination_average);
  }
  if (termination_current == pdlp_termination_status_t::Optimal) {
#ifdef PDLP_VERBOSE_MODE
    std::cout << "Optimal. End total number of iteration current=" << internal_solver_iterations_
              << std::endl;
#endif
    print_final_termination_criteria(
      start_time, current_termination_strategy_.get_convergence_information(), termination_current);
    return current_termination_strategy_.fill_return_problem_solution(
      internal_solver_iterations_,
      pdhg_solver_,
      pdhg_solver_.get_primal_solution(),
      pdhg_solver_.get_dual_solution(),
      get_filled_warmed_start_data(),
      termination_current);
  }

  // Check for infeasibility

  // If strict infeasibility, any infeasibility is detected, it is returned
  // Else both are needed
  // (If infeasibility_detection is not set, termination reason cannot be Infeasible)
  if (settings_.strict_infeasibility) {
    if (termination_current == pdlp_termination_status_t::PrimalInfeasible ||
        termination_current == pdlp_termination_status_t::DualInfeasible) {
#ifdef PDLP_VERBOSE_MODE
      std::cout << "Current Infeasible. End total number of iteration current="
                << internal_solver_iterations_ << std::endl;
#endif
      print_final_termination_criteria(start_time,
                                       current_termination_strategy_.get_convergence_information(),
                                       termination_current);
      return current_termination_strategy_.fill_return_problem_solution(
        internal_solver_iterations_,
        pdhg_solver_,
        pdhg_solver_.get_primal_solution(),
        pdhg_solver_.get_dual_solution(),
        termination_current);
    }
    if (termination_average == pdlp_termination_status_t::PrimalInfeasible ||
        termination_average == pdlp_termination_status_t::DualInfeasible) {
#ifdef PDLP_VERBOSE_MODE
      std::cout << "Average Infeasible. End total number of iteration current="
                << internal_solver_iterations_ << std::endl;
#endif
      print_final_termination_criteria(start_time,
                                       average_termination_strategy_.get_convergence_information(),
                                       termination_average,
                                       true);
      return average_termination_strategy_.fill_return_problem_solution(
        internal_solver_iterations_,
        pdhg_solver_,
        unscaled_primal_avg_solution_,
        unscaled_dual_avg_solution_,
        termination_average);
    }
  } else {
    if ((termination_current == pdlp_termination_status_t::PrimalInfeasible &&
         termination_average == pdlp_termination_status_t::PrimalInfeasible) ||
        (termination_current == pdlp_termination_status_t::DualInfeasible &&
         termination_average == pdlp_termination_status_t::DualInfeasible)) {
#ifdef PDLP_VERBOSE_MODE
      std::cout << "Infeasible. End total number of iteration current="
                << internal_solver_iterations_ << std::endl;
#endif
      print_final_termination_criteria(start_time,
                                       current_termination_strategy_.get_convergence_information(),
                                       termination_current);
      return current_termination_strategy_.fill_return_problem_solution(
        internal_solver_iterations_,
        pdhg_solver_,
        pdhg_solver_.get_primal_solution(),
        pdhg_solver_.get_dual_solution(),
        termination_current);
    }
  }

  // Numerical error has happend (movement is 0 and pdlp_termination_status_t::Optimality has not
  // been reached)
  if (step_size_strategy_.get_valid_step_size() == -1) {
#ifdef PDLP_VERBOSE_MODE
    std::cout << "Numerical Error. End total number of iteration current="
              << internal_solver_iterations_ << std::endl;
#endif
    print_final_termination_criteria(
      start_time, current_termination_strategy_.get_convergence_information(), termination_current);
    return optimization_problem_solution_t<i_t, f_t>{pdlp_termination_status_t::NumericalError,
                                                     stream_view_};
  }

  // If not infeasible and not pdlp_termination_status_t::Optimal and no error, record best so far
  // is toggle
  if (settings_.save_best_primal_so_far)
    record_best_primal_so_far(current_termination_strategy_,
                              average_termination_strategy_,
                              termination_current,
                              termination_average);
  if (total_pdlp_iterations_ % 1000 == 0) { print_termination_criteria(start_time); }

  // No reason to terminate
  return check_limits(start_time);
}

template <typename f_t>
static void compute_stats(const rmm::device_uvector<f_t>& vec,
                          f_t& smallest,
                          f_t& largest,
                          f_t& avg)
{
  auto abs_op      = [] __host__ __device__(f_t x) { return abs(x); };
  auto min_nonzero = [] __host__ __device__(f_t x) {
    return x == 0 ? std::numeric_limits<f_t>::max() : abs(x);
  };

  smallest = thrust::transform_reduce(rmm::exec_policy(vec.stream()),
                                      vec.begin(),
                                      vec.end(),
                                      min_nonzero,
                                      std::numeric_limits<f_t>::max(),
                                      thrust::minimum<f_t>());

  largest = thrust::transform_reduce(
    rmm::exec_policy(vec.stream()), vec.begin(), vec.end(), abs_op, 0.0f, thrust::maximum<f_t>());

  f_t sum = thrust::transform_reduce(
    rmm::exec_policy(vec.stream()), vec.begin(), vec.end(), abs_op, 0.0f, thrust::plus<f_t>());

  avg = sum / vec.size();
};

template <typename f_t>
static void print_problem_info(const rmm::device_uvector<f_t>& nonzero_coeffs,
                               const rmm::device_uvector<f_t>& objective_coeffs,
                               const rmm::device_uvector<f_t>& combined_bounds)
{
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  // Get stats for constraint matrix coefficients
  f_t smallest, largest, avg;
  compute_stats(nonzero_coeffs, smallest, largest, avg);
  std::cout << "Absolute value of nonzero constraint matrix elements: largest=" << largest
            << ", smallest=" << smallest << ", avg=" << avg << std::endl;

  // Get stats for objective coefficients
  compute_stats(objective_coeffs, smallest, largest, avg);
  std::cout << "Absolute value of objective vector elements: largest=" << largest
            << ", smallest=" << smallest << ", avg=" << avg << std::endl;

  // Get stats for combined bounds
  compute_stats(combined_bounds, smallest, largest, avg);
  std::cout << "Absolute value of rhs vector elements: largest=" << largest
            << ", smallest=" << smallest << ", avg=" << avg << std::endl;

  RAFT_CUDA_TRY(cudaDeviceSynchronize());
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::update_primal_dual_solutions(
  std::optional<const rmm::device_uvector<f_t>*> primal,
  std::optional<const rmm::device_uvector<f_t>*> dual)
{
#ifdef PDLP_DEBUG_MODE
  std::cout << "  Updating primal and dual solution" << std::endl;
#endif

  // Copy the initial solution in pdhg as a first solution
  if (primal) {
    raft::copy(pdhg_solver_.get_primal_solution().data(),
               primal.value()->data(),
               primal_size_h_,
               stream_view_);
  }
  if (dual) {
    raft::copy(
      pdhg_solver_.get_dual_solution().data(), dual.value()->data(), dual_size_h_, stream_view_);
  }

  // Handle initial step size if needed

  if (pdlp_hyper_params::update_step_size_on_initial_solution) {
#ifdef PDLP_DEBUG_MODE
    std::cout << "    Updating initial step size on initial solution" << std::endl;
#endif

    // Computing a new step size only make sense if both are set and not all 0
    const bool both_initial_set = primal && dual;
    const bool primal_not_all_zeros =
      both_initial_set && thrust::count(handle_ptr_->get_thrust_policy(),
                                        primal.value()->begin(),
                                        primal.value()->end(),
                                        f_t(0)) != static_cast<i_t>(primal.value()->size());
    const bool dual_not_all_zeros =
      both_initial_set && thrust::count(handle_ptr_->get_thrust_policy(),
                                        dual.value()->begin(),
                                        dual.value()->end(),
                                        f_t(0)) != static_cast<i_t>(dual.value()->size());

    if (both_initial_set && primal_not_all_zeros && dual_not_all_zeros) {
      // To compute an initial step size we use adaptative_step_size.compute_step_sizes
      // It requieres setting potential_next_dual_solution, current_Aty and both delta primal and
      // dual Since we want to mimick a movement from an all 0 solutions, we can simply set both
      // potential_next_dual_solution and our delta to our initial solution current_Aty to all 0

      auto& saddle = pdhg_solver_.get_saddle_point_state();

      // Set all 4 fields
      raft::copy(saddle.get_delta_primal().data(),
                 primal.value()->data(),
                 saddle.get_delta_primal().size(),
                 stream_view_);
      raft::copy(saddle.get_delta_dual().data(),
                 dual.value()->data(),
                 saddle.get_delta_dual().size(),
                 stream_view_);
      raft::copy(pdhg_solver_.get_potential_next_dual_solution().data(),
                 dual.value()->data(),
                 pdhg_solver_.get_potential_next_dual_solution().size(),
                 stream_view_);
      RAFT_CUDA_TRY(cudaMemsetAsync(saddle.get_current_AtY().data(),
                                    f_t(0.0),
                                    sizeof(f_t) * saddle.get_current_AtY().size(),
                                    stream_view_));

      // Scale if should compute initial step size after scaling
      if (!pdlp_hyper_params::compute_initial_step_size_before_scaling) {
#ifdef PDLP_DEBUG_MODE
        std::cout << "      Scaling before computing initial step size" << std::endl;
#endif
        initial_scaling_strategy_.scale_solutions(saddle.get_delta_primal(),
                                                  saddle.get_delta_dual());
        initial_scaling_strategy_.scale_dual(pdhg_solver_.get_potential_next_dual_solution());
      }

      // Compute an initial step size
      ++pdhg_solver_.total_pdhg_iterations_;  // Fake a first initial PDHG step, else it will break
                                              // the computation
      step_size_strategy_.compute_step_sizes(pdhg_solver_, primal_step_size_, dual_step_size_, 0);
      --pdhg_solver_.total_pdhg_iterations_;

      // Else scale after computing initial step size
      if (pdlp_hyper_params::compute_initial_step_size_before_scaling) {
#ifdef PDLP_DEBUG_MODE
        std::cout << "      Scaling after computing initial step size" << std::endl;
#endif
        initial_scaling_strategy_.scale_solutions(saddle.get_delta_primal(),
                                                  saddle.get_delta_dual());
        initial_scaling_strategy_.scale_dual(pdhg_solver_.get_potential_next_dual_solution());
      }
    }
  }

  // Handle initial primal weight if needed

  // We should always scale the initial solution
  // We scale here only if it is not done after if
  // compute_initial_primal_weight_before_scaling is true

  // Scale if should compute primal weight after scaling
  if (!pdlp_hyper_params::compute_initial_primal_weight_before_scaling) {
#ifdef PDLP_DEBUG_MODE
    std::cout << "      Scaling before computing initial primal weight:" << std::endl;
#endif
    initial_scaling_strategy_.scale_solutions(pdhg_solver_.get_primal_solution(),
                                              pdhg_solver_.get_dual_solution());
  }

  // If only primal or dual is set, the primal weight wont (as it can't) be updated
  if (pdlp_hyper_params::update_primal_weight_on_initial_solution) {
#ifdef PDLP_DEBUG_MODE
    std::cout << "      Updating the initial primal weight on initial solution" << std::endl;
#endif
    restart_strategy_.update_distance(
      pdhg_solver_, primal_weight_, primal_step_size_, dual_step_size_, step_size_);
  }

  // We scale here because it was not done previously
  if (pdlp_hyper_params::compute_initial_primal_weight_before_scaling) {
    initial_scaling_strategy_.scale_solutions(pdhg_solver_.get_primal_solution(),
                                              pdhg_solver_.get_dual_solution());
  }
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> pdlp_solver_t<i_t, f_t>::run_solver(
  const std::chrono::high_resolution_clock::time_point& start_time)
{
  bool verbose;
#ifdef PDLP_VERBOSE_MODE
  verbose = true;
#else
  verbose = false;
#endif

#ifdef PDLP_DEBUG_MODE
  std::cout << "Starting PDLP loop:" << std::endl;
#endif

  if (pdlp_hyper_params::compute_initial_step_size_before_scaling) compute_initial_step_size();
  if (pdlp_hyper_params::compute_initial_primal_weight_before_scaling)
    compute_initial_primal_weight();

  initial_scaling_strategy_.scale_problem();

  if (!pdlp_hyper_params::compute_initial_step_size_before_scaling) compute_initial_step_size();
  if (!pdlp_hyper_params::compute_initial_primal_weight_before_scaling)
    compute_initial_primal_weight();

#ifdef PDLP_DEBUG_MODE
  std::cout << "Initial Scaling done" << std::endl;
#endif

  // Needs to be performed here before the below line to make sure the initial primal_weight / step
  // size are used as previous point when potentially updating them in this next call
  if (initial_step_size_.has_value())
    step_size_.set_value_async(initial_step_size_.value(), stream_view_);
  if (initial_primal_weight_.has_value())
    primal_weight_.set_value_async(initial_primal_weight_.value(), stream_view_);
  if (initial_k_.has_value()) {
    pdhg_solver_.total_pdhg_iterations_ = initial_k_.value();
    pdhg_solver_.get_d_total_pdhg_iterations().set_value_async(initial_k_.value(), stream_view_);
  }

  // Only the primal_weight_ and step_size_ variables are initialized during the initial phase
  // The associated primal/dual step_size (computed using the two firstly mentionned) are not
  // initialized. This calls ensures the latter
  // In the event of a given primal and dual solutions and if the option is toggled, calling the
  // update primal_weight and step_size will also update the associated primal_step_size_,
  // dual_step_size_. In summary: the below call is only mandatory at the beginning when
  // computing/setting the initial primal weight and step size and if they are not recomputed later.
  step_size_strategy_.get_primal_and_dual_stepsizes(primal_step_size_, dual_step_size_);

  // If there is an initial primal or dual we should update the restart info as if there was a step
  // that has happend
  if (initial_primal_.size() != 0 || initial_dual_.size() != 0) {
    update_primal_dual_solutions(
      (initial_primal_.size() != 0) ? std::make_optional(&initial_primal_) : std::nullopt,
      (initial_dual_.size() != 0) ? std::make_optional(&initial_dual_) : std::nullopt);
  }

  // Project initial primal solution
  if (pdlp_hyper_params::project_initial_primal) {
    raft::linalg::ternaryOp(pdhg_solver_.get_primal_solution().data(),
                            pdhg_solver_.get_primal_solution().data(),
                            op_problem_scaled_.variable_lower_bounds.data(),
                            op_problem_scaled_.variable_upper_bounds.data(),
                            primal_size_h_,
                            clamp<f_t>(),
                            stream_view_);
    raft::linalg::ternaryOp(unscaled_primal_avg_solution_.data(),
                            unscaled_primal_avg_solution_.data(),
                            op_problem_scaled_.variable_lower_bounds.data(),
                            op_problem_scaled_.variable_upper_bounds.data(),
                            primal_size_h_,
                            clamp<f_t>(),
                            stream_view_);
  }

  if (verbose) {
    std::cout << "primal_size_h_ " << primal_size_h_ << " dual_size_h_ " << dual_size_h_ << " nnz "
              << problem_ptr->nnz << std::endl;
    std::cout << "Problem before scaling" << std::endl;
    print_problem_info<f_t>(
      problem_ptr->coefficients, problem_ptr->objective_coefficients, problem_ptr->combined_bounds);
    std::cout << "Problem after scaling" << std::endl;
    print_problem_info<f_t>(op_problem_scaled_.coefficients,
                            op_problem_scaled_.objective_coefficients,
                            op_problem_scaled_.combined_bounds);
    raft::print_device_vector("Initial step_size", step_size_.data(), 1, std::cout);
    raft::print_device_vector("Initial primal_weight", primal_weight_.data(), 1, std::cout);
    raft::print_device_vector("Initial primal_step_size", primal_step_size_.data(), 1, std::cout);
    raft::print_device_vector("Initial dual_step_size", dual_step_size_.data(), 1, std::cout);
  }

  bool warm_start_was_given =
    settings_.get_pdlp_warm_start_data().last_restart_duality_gap_dual_solution_.size() != 0;

  if (!inside_mip_) {
    CUOPT_LOG_INFO(
      "   Iter    Primal Obj.      Dual Obj.    Gap        Primal Res.  Dual Res.   Time");
  }
  while (true) {
    bool is_major_iteration = ((total_pdlp_iterations_ % pdlp_hyper_params::major_iteration == 0) &&
                               (total_pdlp_iterations_ > 0)) ||
                              (total_pdlp_iterations_ <= pdlp_hyper_params::min_iteration_restart);
    bool error_occured                      = (step_size_strategy_.get_valid_step_size() == -1);
    bool artificial_restart_check_main_loop = false;
    if (pdlp_hyper_params::artificial_restart_in_main_loop)
      artificial_restart_check_main_loop =
        restart_strategy_.should_do_artificial_restart(total_pdlp_iterations_);
    if (is_major_iteration || artificial_restart_check_main_loop || error_occured) {
      if (verbose) {
        std::cout << "-------------------------------" << std::endl;
        std::cout << internal_solver_iterations_ << std::endl;
        raft::print_device_vector("step_size", step_size_.data(), 1, std::cout);
        raft::print_device_vector("primal_weight", primal_weight_.data(), 1, std::cout);
        raft::print_device_vector("primal_step_size", primal_step_size_.data(), 1, std::cout);
        raft::print_device_vector("dual_step_size", dual_step_size_.data(), 1, std::cout);
      }

      // If a warm start is given and it's the first step, the average solutions were already filled
      bool no_rescale_average = internal_solver_iterations_ == 0 && warm_start_was_given;

      if (!no_rescale_average) {
        // Average in PDLP is scaled then unscaled which can create numerical innacuracies (a * x /
        // x can != x using float) This can create issues when comparing current and average kkt
        // scores, falsly assuming they are different while they should be equal They should be
        // equal:
        // 1. At the very beginning of the solver, when no steps have been taken yet
        // 2. After a single step, since average of one step is the same step
        if (internal_solver_iterations_ <= 1) {
          raft::copy(unscaled_primal_avg_solution_.data(),
                     pdhg_solver_.get_primal_solution().data(),
                     primal_size_h_,
                     stream_view_);
          raft::copy(unscaled_dual_avg_solution_.data(),
                     pdhg_solver_.get_dual_solution().data(),
                     dual_size_h_,
                     stream_view_);
        } else {
          restart_strategy_.get_average_solutions(unscaled_primal_avg_solution_,
                                                  unscaled_dual_avg_solution_);
        }
      }

      // We go back to the unscaled problem here. It ensures that we do not terminate 'too early'
      // because of the error margin being evaluated on the scaled problem

      // Evaluation is done on the unscaled problem and solutions

      // If warm start data was given, the average solutions were also already scaled
      if (!no_rescale_average) {
        initial_scaling_strategy_.unscale_solutions(unscaled_primal_avg_solution_,
                                                    unscaled_dual_avg_solution_);
      }
      initial_scaling_strategy_.unscale_solutions(pdhg_solver_.get_primal_solution(),
                                                  pdhg_solver_.get_dual_solution());

      // Check for termination
      std::optional<optimization_problem_solution_t<i_t, f_t>> solution =
        check_termination(start_time);

      if (solution.has_value()) { return std::move(solution.value()); }

      if (pdlp_hyper_params::rescale_for_restart) {
        initial_scaling_strategy_.scale_solutions(unscaled_primal_avg_solution_,
                                                  unscaled_dual_avg_solution_);
        initial_scaling_strategy_.scale_solutions(pdhg_solver_.get_primal_solution(),
                                                  pdhg_solver_.get_dual_solution());
      }

      if (pdlp_hyper_params::restart_strategy !=
          static_cast<int>(
            detail::pdlp_restart_strategy_t<i_t, f_t>::restart_strategy_t::NO_RESTART)) {
        restart_strategy_.compute_restart(
          pdhg_solver_,
          unscaled_primal_avg_solution_,
          unscaled_dual_avg_solution_,
          total_pdlp_iterations_,
          primal_step_size_,
          dual_step_size_,
          primal_weight_,
          step_size_,
          current_termination_strategy_.get_convergence_information(),  // Needed for KKT restart
          average_termination_strategy_.get_convergence_information()   // Needed for KKT restart
        );
      }

      if (!pdlp_hyper_params::rescale_for_restart) {
        // We don't need to rescale average because what matters is weighted_average_solution
        // getting the scaled accumulation
        // During the next iteration, unscaled_avg_solution will be overwritten again through
        // get_average_solutions
        initial_scaling_strategy_.scale_solutions(pdhg_solver_.get_primal_solution(),
                                                  pdhg_solver_.get_dual_solution());
      }
    }

    take_step(total_pdlp_iterations_);

    ++total_pdlp_iterations_;
    ++internal_solver_iterations_;
  }
  return optimization_problem_solution_t<i_t, f_t>{pdlp_termination_status_t::NumericalError,
                                                   stream_view_};
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::take_step(i_t total_pdlp_iterations)
{
  // continue testing stepsize until we find a valid one or encounter a numerical error
  step_size_strategy_.set_valid_step_size(0);

  while (step_size_strategy_.get_valid_step_size() == 0) {
#ifdef PDLP_DEBUG_MODE
    std::cout << "PDHG Iteration:\n"
              << "    primal_weight=" << primal_weight_.value(stream_view_) << "\n"
              << "    step_size=" << step_size_.value(stream_view_) << "\n"
              << "    primal_step_size=" << primal_step_size_.value(stream_view_) << "\n"
              << "    dual_step_size=" << dual_step_size_.value(stream_view_) << std::endl;
#endif
    pdhg_solver_.take_step(primal_step_size_,
                           dual_step_size_,
                           restart_strategy_.get_iterations_since_last_restart(),
                           restart_strategy_.get_last_restart_was_average(),
                           total_pdlp_iterations);

    step_size_strategy_.compute_step_sizes(
      pdhg_solver_, primal_step_size_, dual_step_size_, total_pdlp_iterations);
  }
#ifdef PDLP_DEBUG_MODE
  std::cout << "PDHG Iteration: valid step size found" << std::endl;
#endif

  // Valid state found, update internal solution state
  // Average is being added asynchronously on the GPU while the solution is being updated on the CPU
  restart_strategy_.add_current_solution_to_average_solution(
    pdhg_solver_.get_potential_next_primal_solution().data(),
    pdhg_solver_.get_potential_next_dual_solution().data(),
    step_size_,
    total_pdlp_iterations);
  pdhg_solver_.update_solution(current_op_problem_evaluation_cusparse_view_);
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::compute_initial_step_size()
{
  raft::common::nvtx::range fun_scope("compute_initial_step_size");

  // set stepsize relative to maximum absolute value of A
  rmm::device_scalar<f_t> abs_max_element{0.0, stream_view_};
  void* d_temp_storage      = NULL;
  size_t temp_storage_bytes = 0;

  detail::max_abs_value<f_t> red_op;
  cub::DeviceReduce::Reduce(d_temp_storage,
                            temp_storage_bytes,
                            op_problem_scaled_.coefficients.data(),
                            abs_max_element.data(),
                            op_problem_scaled_.nnz,
                            red_op,
                            0.0,
                            stream_view_);
  // Allocate temporary storage
  rmm::device_buffer cub_tmp{temp_storage_bytes, stream_view_};
  // Run max-reduction
  cub::DeviceReduce::Reduce(cub_tmp.data(),
                            temp_storage_bytes,
                            op_problem_scaled_.coefficients.data(),
                            abs_max_element.data(),
                            op_problem_scaled_.nnz,
                            red_op,
                            0.0,
                            stream_view_);
  raft::linalg::eltwiseDivideCheckZero(
    step_size_.data(), step_size_.data(), abs_max_element.data(), 1, stream_view_);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));
}

template <typename f_t>
__global__ void compute_weights_initial_primal_weight_from_squared_norms(const f_t* b_vec_norm,
                                                                         const f_t* c_vec_norm,
                                                                         f_t* primal_weight)
{
  if (threadIdx.x + blockIdx.x * blockDim.x > 0) { return; }
  f_t c_vec_norm_ = *c_vec_norm;
  f_t b_vec_norm_ = *b_vec_norm;

  if (b_vec_norm_ > 0.0 && c_vec_norm_ > 0.0) {
#ifdef PDLP_DEBUG_MODE
    printf("b_vec_norm_ %lf c_vec_norm_ %lf primal_importance %lf\n",
           b_vec_norm_,
           c_vec_norm_,
           pdlp_hyper_params::primal_importance);
#endif
    *primal_weight = pdlp_hyper_params::primal_importance * (c_vec_norm_ / b_vec_norm_);
  } else {
    *primal_weight = pdlp_hyper_params::primal_importance;
  }
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::compute_initial_primal_weight()
{
  // Here we use the combined bounds of the op_problem_scaled which may or may not be scaled yet
  // based on pdlp config
  detail::combine_constraint_bounds<i_t, f_t>(op_problem_scaled_,
                                              op_problem_scaled_.combined_bounds);

  // => same as sqrt(dot(b,b))
  rmm::device_scalar<f_t> b_vec_norm{0.0, stream_view_};
  rmm::device_scalar<f_t> c_vec_norm{0.0, stream_view_};

  detail::my_l2_weighted_norm<i_t, f_t>(op_problem_scaled_.combined_bounds,
                                        pdlp_hyper_params::initial_primal_weight_b_scaling,
                                        b_vec_norm,
                                        stream_view_);

  detail::my_l2_weighted_norm<i_t, f_t>(op_problem_scaled_.objective_coefficients,
                                        pdlp_hyper_params::initial_primal_weight_c_scaling,
                                        c_vec_norm,
                                        stream_view_);

  compute_weights_initial_primal_weight_from_squared_norms<<<1, 1, 0, stream_view_>>>(
    b_vec_norm.data(), c_vec_norm.data(), primal_weight_.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));
}

template <typename i_t, typename f_t>
f_t pdlp_solver_t<i_t, f_t>::get_primal_weight_h() const
{
  return primal_weight_.value(stream_view_);
}

template <typename i_t, typename f_t>
f_t pdlp_solver_t<i_t, f_t>::get_step_size_h() const
{
  return step_size_.value(stream_view_);
}

template <typename i_t, typename f_t>
i_t pdlp_solver_t<i_t, f_t>::get_total_pdhg_iterations() const
{
  return pdhg_solver_.total_pdhg_iterations_;
}

template <typename i_t, typename f_t>
detail::pdlp_termination_strategy_t<i_t, f_t>&
pdlp_solver_t<i_t, f_t>::get_current_termination_strategy()
{
  return current_termination_strategy_;
}

#if MIP_INSTANTIATE_FLOAT
template class pdlp_solver_t<int, float>;

template __global__ void compute_weights_initial_primal_weight_from_squared_norms<float>(
  const float* b_vec_norm, const float* c_vec_norm, float* primal_weight);
#endif

#if MIP_INSTANTIATE_DOUBLE
template class pdlp_solver_t<int, double>;

template __global__ void compute_weights_initial_primal_weight_from_squared_norms<double>(
  const double* b_vec_norm, const double* c_vec_norm, double* primal_weight);
#endif

}  // namespace cuopt::linear_programming::detail
