/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>

#include <pdlp/cusparse_view.hpp>
#include <pdlp/pdlp.cuh>
#include <pdlp/swap_and_resize_helper.cuh>
#include <pdlp/utils.cuh>

#include <mip_heuristics/mip_constants.hpp>
#include "cuopt/linear_programming/pdlp/solver_solution.hpp"

#include <utilities/copy_helpers.hpp>
#include <utilities/macros.cuh>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/ternary_op.cuh>

#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>

#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/logical.h>

#include <cmath>
#include <optional>
#include <unordered_set>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
static size_t batch_size_handler(const problem_t<i_t, f_t>& op_problem,
                                 const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  if (settings.new_bounds.empty()) { return 1; }
#ifdef BATCH_VERBOSE_MODE
  std::cout << "Running batch PDLP with " << settings.new_bounds.size() << " problems" << std::endl;
#endif
  return settings.new_bounds.size();
}

template <typename i_t, typename f_t>
pdlp_solver_t<i_t, f_t>::pdlp_solver_t(problem_t<i_t, f_t>& op_problem,
                                       pdlp_solver_settings_t<i_t, f_t> const& settings,
                                       bool is_legacy_batch_mode)
  : climber_strategies_(batch_size_handler(op_problem, settings)),
    batch_mode_(climber_strategies_.size() > 1),
    handle_ptr_(op_problem.handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    settings_(settings),
    problem_ptr(&op_problem),
    op_problem_scaled_(
      op_problem, false),  // False to call the PDLP custom version of the problem copy constructor
    unscaled_primal_avg_solution_{static_cast<size_t>(op_problem.n_variables), stream_view_},
    unscaled_dual_avg_solution_{static_cast<size_t>(op_problem.n_constraints), stream_view_},
    primal_size_h_(op_problem.n_variables),
    dual_size_h_(op_problem.n_constraints),
    primal_step_size_{climber_strategies_.size(), stream_view_},
    dual_step_size_{climber_strategies_.size(), stream_view_},
    primal_weight_{climber_strategies_.size(), stream_view_},
    best_primal_weight_{climber_strategies_.size(), stream_view_},
    step_size_{climber_strategies_.size(), stream_view_},
    step_size_strategy_{handle_ptr_,
                        &primal_weight_,
                        &step_size_,
                        is_legacy_batch_mode,
                        op_problem.n_variables,
                        op_problem.n_constraints,
                        climber_strategies_,
                        settings_.hyper_params},
    pdhg_solver_{handle_ptr_,
                 op_problem_scaled_,
                 is_legacy_batch_mode,
                 climber_strategies_,
                 settings_.hyper_params,
                 settings_.new_bounds},
    initial_scaling_strategy_{handle_ptr_,
                              op_problem_scaled_,
                              settings_.hyper_params.default_l_inf_ruiz_iterations,
                              (f_t)settings_.hyper_params.default_alpha_pock_chambolle_rescaling,
                              op_problem_scaled_.reverse_coefficients,
                              op_problem_scaled_.reverse_offsets,
                              op_problem_scaled_.reverse_constraints,
                              &pdhg_solver_,
                              settings_.hyper_params},
    average_op_problem_evaluation_cusparse_view_{handle_ptr_,
                                                 op_problem,
                                                 unscaled_primal_avg_solution_,
                                                 unscaled_dual_avg_solution_,
                                                 pdhg_solver_.get_primal_tmp_resource(),
                                                 pdhg_solver_.get_dual_tmp_resource(),
                                                 pdhg_solver_.get_potential_next_primal_solution(),
                                                 pdhg_solver_.get_potential_next_dual_solution(),
                                                 op_problem.reverse_coefficients,
                                                 op_problem.reverse_offsets,
                                                 op_problem.reverse_constraints,
                                                 climber_strategies_,
                                                 settings_.hyper_params},
    current_op_problem_evaluation_cusparse_view_{handle_ptr_,
                                                 op_problem,
                                                 pdhg_solver_.get_primal_solution(),
                                                 pdhg_solver_.get_dual_solution(),
                                                 pdhg_solver_.get_primal_tmp_resource(),
                                                 pdhg_solver_.get_dual_tmp_resource(),
                                                 pdhg_solver_.get_potential_next_primal_solution(),
                                                 pdhg_solver_.get_potential_next_dual_solution(),
                                                 op_problem.reverse_coefficients,
                                                 op_problem.reverse_offsets,
                                                 op_problem.reverse_constraints,
                                                 climber_strategies_,
                                                 settings_.hyper_params},
    restart_strategy_{handle_ptr_,
                      op_problem,
                      average_op_problem_evaluation_cusparse_view_,
                      primal_size_h_,
                      dual_size_h_,
                      is_legacy_batch_mode,
                      climber_strategies_,
                      settings_.hyper_params},
    average_termination_strategy_{handle_ptr_,
                                  op_problem,
                                  op_problem_scaled_,
                                  average_op_problem_evaluation_cusparse_view_,
                                  pdhg_solver_.get_cusparse_view(),
                                  primal_size_h_,
                                  dual_size_h_,
                                  initial_scaling_strategy_,
                                  settings_,
                                  climber_strategies_},
    current_termination_strategy_{handle_ptr_,
                                  op_problem,
                                  op_problem_scaled_,
                                  current_op_problem_evaluation_cusparse_view_,
                                  pdhg_solver_.get_cusparse_view(),
                                  primal_size_h_,
                                  dual_size_h_,
                                  initial_scaling_strategy_,
                                  settings_,
                                  climber_strategies_},
    initial_primal_{0, stream_view_},
    initial_dual_{0, stream_view_},
    reusable_device_scalar_value_1_{f_t(1.0), stream_view_},
    reusable_device_scalar_value_0_{f_t(0.0), stream_view_},
    batch_solution_to_return_{pdlp_termination_status_t::TimeLimit, stream_view_},
    best_primal_solution_so_far{pdlp_termination_status_t::TimeLimit, stream_view_},
    inside_mip_{false}
{
  // Set step_size initial scaling
  thrust::fill(handle_ptr_->get_thrust_policy(),
               step_size_.data(),
               step_size_.end(),
               (f_t)settings_.hyper_params.initial_step_size_scaling);

  if (settings_.has_initial_primal_solution()) {
    auto& primal_sol = settings_.get_initial_primal_solution();
    set_initial_primal_solution(primal_sol);
  }
  if (settings_.has_initial_dual_solution()) {
    const auto& dual_sol = settings_.get_initial_dual_solution();
    set_initial_dual_solution(dual_sol);
  }

  if (settings_.get_pdlp_warm_start_data().is_populated()) {
    cuopt_expects(
      !batch_mode_, error_type_t::ValidationError, "Batch mode not supported for warm start");
    cuopt_expects(settings_.pdlp_solver_mode == pdlp_solver_mode_t::Stable2,
                  error_type_t::ValidationError,
                  "Only Stable2 mode supported for warm start");
    set_initial_primal_solution(settings_.get_pdlp_warm_start_data().current_primal_solution_);
    set_initial_dual_solution(settings_.get_pdlp_warm_start_data().current_dual_solution_);
    initial_step_size_     = settings_.get_pdlp_warm_start_data().initial_step_size_;
    initial_primal_weight_ = settings_.get_pdlp_warm_start_data().initial_primal_weight_;
    total_pdlp_iterations_ = settings_.get_pdlp_warm_start_data().total_pdlp_iterations_;
    pdhg_solver_.total_pdhg_iterations_ =
      settings_.get_pdlp_warm_start_data().total_pdhg_iterations_;
    pdhg_solver_.get_d_total_pdhg_iterations().set_value_async(
      settings_.get_pdlp_warm_start_data().total_pdhg_iterations_, stream_view_);
    restart_strategy_.last_candidate_kkt_score =
      settings_.get_pdlp_warm_start_data().last_candidate_kkt_score_;
    restart_strategy_.last_restart_kkt_score =
      settings_.get_pdlp_warm_start_data().last_restart_kkt_score_;
    raft::copy(restart_strategy_.weighted_average_solution_.sum_primal_solutions_.data(),
               settings_.get_pdlp_warm_start_data().sum_primal_solutions_.data(),
               settings_.get_pdlp_warm_start_data().sum_primal_solutions_.size(),
               stream_view_);
    raft::copy(restart_strategy_.weighted_average_solution_.sum_dual_solutions_.data(),
               settings_.get_pdlp_warm_start_data().sum_dual_solutions_.data(),
               settings_.get_pdlp_warm_start_data().sum_dual_solutions_.size(),
               stream_view_);
    raft::copy(unscaled_primal_avg_solution_.data(),
               settings_.get_pdlp_warm_start_data().initial_primal_average_.data(),
               settings_.get_pdlp_warm_start_data().initial_primal_average_.size(),
               stream_view_);
    raft::copy(unscaled_dual_avg_solution_.data(),
               settings_.get_pdlp_warm_start_data().initial_dual_average_.data(),
               settings_.get_pdlp_warm_start_data().initial_dual_average_.size(),
               stream_view_);
    raft::copy(pdhg_solver_.get_saddle_point_state().get_current_AtY().data(),
               settings_.get_pdlp_warm_start_data().current_ATY_.data(),
               settings_.get_pdlp_warm_start_data().current_ATY_.size(),
               stream_view_);
    raft::copy(
      restart_strategy_.last_restart_duality_gap_.primal_solution_.data(),
      settings_.get_pdlp_warm_start_data().last_restart_duality_gap_primal_solution_.data(),
      settings_.get_pdlp_warm_start_data().last_restart_duality_gap_primal_solution_.size(),
      stream_view_);
    raft::copy(restart_strategy_.last_restart_duality_gap_.dual_solution_.data(),
               settings_.get_pdlp_warm_start_data().last_restart_duality_gap_dual_solution_.data(),
               settings_.get_pdlp_warm_start_data().last_restart_duality_gap_dual_solution_.size(),
               stream_view_);

    const auto value = settings_.get_pdlp_warm_start_data().sum_solution_weight_;
    restart_strategy_.weighted_average_solution_.sum_primal_solution_weights_.set_value_async(
      value, stream_view_);
    restart_strategy_.weighted_average_solution_.sum_dual_solution_weights_.set_value_async(
      value, stream_view_);
    restart_strategy_.weighted_average_solution_.iterations_since_last_restart_ =
      settings_.get_pdlp_warm_start_data().iterations_since_last_restart_;
  }
  // Checks performed below are assert only
  best_primal_quality_so_far_.primal_objective = (op_problem_scaled_.maximize)
                                                   ? -std::numeric_limits<f_t>::infinity()
                                                   : std::numeric_limits<f_t>::infinity();
  op_problem.check_problem_representation(true, false);
  op_problem_scaled_.check_problem_representation(true, false);

  if (settings_.new_bounds.size() > 0) {
    batch_solution_to_return_.get_additional_termination_informations().resize(
      settings_.new_bounds.size());
    batch_solution_to_return_.get_terminations_status().resize(settings_.new_bounds.size());
    batch_solution_to_return_.get_primal_solution().resize(
      op_problem.n_variables * settings_.new_bounds.size(), stream_view_);
    batch_solution_to_return_.get_dual_solution().resize(
      op_problem.n_constraints * settings_.new_bounds.size(), stream_view_);
    batch_solution_to_return_.get_reduced_cost().resize(
      op_problem.n_variables * settings_.new_bounds.size(), stream_view_);
  }
  for (size_t i = 0; i < climber_strategies_.size(); ++i) {
    climber_strategies_[i].original_index = static_cast<int>(i);
  }
  if (batch_mode_) {
    cuopt_assert(!settings_.detect_infeasibility,
                 "Detect infeasibility must be false in batch mode");
  }
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
  cuopt_assert(initial_primal_solution.size() % primal_size_h_ == 0,
               "Initial primal solution size must be divisible by primal_size_h_");
  initial_primal_.resize(primal_size_h_ * climber_strategies_.size(), stream_view_);
  // In batch case initial_primal_ can be larger than the given initial_primal_solution
  cub::DeviceTransform::Transform(problem_wrap_container(initial_primal_solution),
                                  initial_primal_.data(),
                                  initial_primal_.size(),
                                  cuda::std::identity{},
                                  stream_view_);
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::set_initial_dual_solution(
  const rmm::device_uvector<f_t>& initial_dual_solution)
{
  cuopt_assert(initial_dual_solution.size() % dual_size_h_ == 0,
               "Initial dual solution size must be divisible by dual_size_h_");
  initial_dual_.resize(dual_size_h_ * climber_strategies_.size(), stream_view_);
  cub::DeviceTransform::Transform(problem_wrap_container(initial_dual_solution),
                                  initial_dual_.data(),
                                  initial_dual_.size(),
                                  cuda::std::identity{},
                                  stream_view_);
}

static bool time_limit_reached(const timer_t& timer) { return timer.check_time_limit(); }

template <typename i_t, typename f_t>
std::optional<optimization_problem_solution_t<i_t, f_t>> pdlp_solver_t<i_t, f_t>::check_limits(
  const timer_t& timer)
{
  // Check for time limit
  if (time_limit_reached(timer)) {
    if (settings_.save_best_primal_so_far) {
#ifdef PDLP_VERBOSE_MODE
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      std::cout << "Time Limit reached, returning best primal so far" << std::endl;
#endif
      return std::move(best_primal_solution_so_far);
    }

    if (batch_mode_) {
      // Set the termination status to TimeLimit for all climbers appart from the potentially
      // already done ones
      for (size_t i = 0; i < batch_solution_to_return_.get_terminations_status().size(); ++i) {
        if (!current_termination_strategy_.is_done(
              current_termination_strategy_.get_termination_status(i))) {
          batch_solution_to_return_
            .get_terminations_status()[climber_strategies_[i].original_index] =
            pdlp_termination_status_t::TimeLimit;
        }
      }
      current_termination_strategy_.convert_gpu_terms_stats_to_host(
        batch_solution_to_return_.get_additional_termination_informations());
      return optimization_problem_solution_t<i_t, f_t>{
        batch_solution_to_return_.get_primal_solution(),
        batch_solution_to_return_.get_dual_solution(),
        batch_solution_to_return_.get_reduced_cost(),
        get_filled_warmed_start_data(),
        problem_ptr->objective_name,
        problem_ptr->var_names,
        problem_ptr->row_names,
        std::move(batch_solution_to_return_.get_additional_termination_informations()),
        std::move(batch_solution_to_return_.get_terminations_status())};
    }

#ifdef PDLP_VERBOSE_MODE
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    std::cout << "Time Limit reached, returning current solution" << std::endl;
#endif
    return current_termination_strategy_.fill_return_problem_solution(
      internal_solver_iterations_,
      pdhg_solver_,
      (settings_.hyper_params.use_adaptive_step_size_strategy)
        ? pdhg_solver_.get_primal_solution()
        : pdhg_solver_.get_potential_next_primal_solution(),
      (settings_.hyper_params.use_adaptive_step_size_strategy)
        ? pdhg_solver_.get_dual_solution()
        : pdhg_solver_.get_potential_next_dual_solution(),
      get_filled_warmed_start_data(),
      std::vector<pdlp_termination_status_t>(climber_strategies_.size(),
                                             pdlp_termination_status_t::TimeLimit));
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

    if (batch_mode_) {
      // Set the termination status to IterationLimit for all climbers appart from the potentially
      // already done ones
      for (size_t i = 0; i < batch_solution_to_return_.get_terminations_status().size(); ++i) {
        if (!current_termination_strategy_.is_done(
              current_termination_strategy_.get_termination_status(i))) {
          batch_solution_to_return_
            .get_terminations_status()[climber_strategies_[i].original_index] =
            pdlp_termination_status_t::IterationLimit;
        }
      }
      current_termination_strategy_.convert_gpu_terms_stats_to_host(
        batch_solution_to_return_.get_additional_termination_informations());
      return optimization_problem_solution_t<i_t, f_t>{
        batch_solution_to_return_.get_primal_solution(),
        batch_solution_to_return_.get_dual_solution(),
        batch_solution_to_return_.get_reduced_cost(),
        get_filled_warmed_start_data(),
        problem_ptr->objective_name,
        problem_ptr->var_names,
        problem_ptr->row_names,
        std::move(batch_solution_to_return_.get_additional_termination_informations()),
        std::move(batch_solution_to_return_.get_terminations_status())};
    }

    return current_termination_strategy_.fill_return_problem_solution(
      internal_solver_iterations_,
      pdhg_solver_,
      (settings_.hyper_params.use_adaptive_step_size_strategy)
        ? pdhg_solver_.get_primal_solution()
        : pdhg_solver_.get_potential_next_primal_solution(),
      (settings_.hyper_params.use_adaptive_step_size_strategy)
        ? pdhg_solver_.get_dual_solution()
        : pdhg_solver_.get_potential_next_dual_solution(),
      get_filled_warmed_start_data(),
      std::vector<pdlp_termination_status_t>(climber_strategies_.size(),
                                             pdlp_termination_status_t::IterationLimit));
  }

  // Check for concurrent limit
  if (settings_.method == method_t::Concurrent && settings_.concurrent_halt != nullptr &&
      *settings_.concurrent_halt == 1) {
#ifdef PDLP_VERBOSE_MODE
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    std::cout << "Concurrent Limit reached, returning current solution" << std::endl;
#endif

    if (batch_mode_) {
      // Set the termination status to ConcurrentLimit for all climbers appart from the potentially
      // already done ones
      for (size_t i = 0; i < batch_solution_to_return_.get_terminations_status().size(); ++i) {
        if (!current_termination_strategy_.is_done(
              current_termination_strategy_.get_termination_status(i))) {
          batch_solution_to_return_
            .get_terminations_status()[climber_strategies_[i].original_index] =
            pdlp_termination_status_t::ConcurrentLimit;
        }
      }
      current_termination_strategy_.convert_gpu_terms_stats_to_host(
        batch_solution_to_return_.get_additional_termination_informations());
      return optimization_problem_solution_t<i_t, f_t>{
        batch_solution_to_return_.get_primal_solution(),
        batch_solution_to_return_.get_dual_solution(),
        batch_solution_to_return_.get_reduced_cost(),
        get_filled_warmed_start_data(),
        problem_ptr->objective_name,
        problem_ptr->var_names,
        problem_ptr->row_names,
        std::move(batch_solution_to_return_.get_additional_termination_informations()),
        std::move(batch_solution_to_return_.get_terminations_status())};
    }

    return current_termination_strategy_.fill_return_problem_solution(
      internal_solver_iterations_,
      pdhg_solver_,
      (settings_.hyper_params.use_adaptive_step_size_strategy)
        ? pdhg_solver_.get_primal_solution()
        : pdhg_solver_.get_potential_next_primal_solution(),
      (settings_.hyper_params.use_adaptive_step_size_strategy)
        ? pdhg_solver_.get_dual_solution()
        : pdhg_solver_.get_potential_next_dual_solution(),
      get_filled_warmed_start_data(),
      std::vector<pdlp_termination_status_t>(climber_strategies_.size(),
                                             pdlp_termination_status_t::ConcurrentLimit));
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
      std::vector<pdlp_termination_status_t>(climber_strategies_.size(),
                                             pdlp_termination_status_t::TimeLimit),
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
  if (batch_mode_)
    return pdlp_warm_start_data_t<i_t, f_t>();
  else {
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
      get_primal_weight_h(0),
      get_step_size_h(0),
      total_pdlp_iterations_,
      pdhg_solver_.total_pdhg_iterations_,
      restart_strategy_.last_candidate_kkt_score,
      restart_strategy_.last_restart_kkt_score,
      restart_strategy_.weighted_average_solution_.sum_primal_solution_weights_.value(stream_view_),
      restart_strategy_.weighted_average_solution_.iterations_since_last_restart_);
  }
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::print_termination_criteria(const timer_t& timer, bool is_average)
{
  if (!inside_mip_) {
    auto elapsed = timer.elapsed_time();
    if (is_average) {
      average_termination_strategy_.print_termination_criteria(total_pdlp_iterations_, elapsed);
    } else {
      current_termination_strategy_.print_termination_criteria(total_pdlp_iterations_, elapsed);
    }
  }
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::print_final_termination_criteria(
  const timer_t& timer,
  const convergence_information_t<i_t, f_t>& convergence_information,
  const pdlp_termination_status_t& termination_status,
  bool is_average)
{
  if (!inside_mip_) {
    // TODO less critical batch mode: handle this
    print_termination_criteria(timer, is_average);
    CUOPT_LOG_INFO(
      "LP Solver status:                %s",
      optimization_problem_solution_t<i_t, f_t>::get_termination_status_string(termination_status)
        .c_str());
    CUOPT_LOG_INFO("Primal objective:                %+.8e",
                   convergence_information.get_primal_objective().element(0, stream_view_));
    CUOPT_LOG_INFO("Dual objective:                  %+.8e",
                   convergence_information.get_dual_objective().element(0, stream_view_));
    CUOPT_LOG_INFO("Duality gap (abs/rel):           %+.2e / %+.2e",
                   convergence_information.get_gap().element(0, stream_view_),
                   convergence_information.get_relative_gap_value());
    CUOPT_LOG_INFO("Primal infeasibility (abs/rel):  %+.2e / %+.2e",
                   convergence_information.get_l2_primal_residual().element(0, stream_view_),
                   convergence_information.get_relative_l2_primal_residual_value());
    CUOPT_LOG_INFO("Dual infeasibility (abs/rel):    %+.2e / %+.2e",
                   convergence_information.get_l2_dual_residual().element(0, stream_view_),
                   convergence_information.get_relative_l2_dual_residual_value());
  }
}

template <typename i_t, typename f_t>
std::optional<optimization_problem_solution_t<i_t, f_t>>
pdlp_solver_t<i_t, f_t>::check_batch_termination(const timer_t& timer)
{
  raft::common::nvtx::range fun_scope("check_batch_termination");

  // Forced to do it in two lines because of macro template interaction
  [[maybe_unused]] const bool is_cupdlpx = is_cupdlpx_restart<i_t, f_t>(settings_.hyper_params);
  cuopt_assert(is_cupdlpx, "Batch termination handling only supported with cuPDLPx restart");

#ifdef BATCH_VERBOSE_MODE
  for (size_t i = 0; i < current_termination_strategy_.get_terminations_status().size(); ++i) {
    const auto& term = current_termination_strategy_.get_termination_status(i);
    if (current_termination_strategy_.is_done(term)) {
      std::cout << "[BATCH MODE]: Climber " << i << " is done with "
                << optimization_problem_solution_t<i_t, f_t>::get_termination_status_string(term)
                << " at step " << total_pdlp_iterations_ << ". It's original index is "
                << climber_strategies_[i].original_index << std::endl;
    }
  }
#endif

  // All are optimal or infeasible
  if (current_termination_strategy_.all_done()) {
    const auto original_batch_size = settings_.new_bounds.size();
    // Some climber got removed from the batch while the optimization was running
    if (original_batch_size != climber_strategies_.size()) {
#ifdef BATCH_VERBOSE_MODE
      std::cout << "Original batch size was " << original_batch_size << " but is now "
                << climber_strategies_.size() << std::endl;
#endif
      cuopt_assert(current_termination_strategy_.get_terminations_status().size() ==
                     climber_strategies_.size(),
                   "Terminations status size mismatch");
      for (size_t i = 0; i < current_termination_strategy_.get_terminations_status().size(); ++i) {
        // Found one that is done
        cuopt_assert(current_termination_strategy_.is_done(
                       current_termination_strategy_.get_termination_status(i)),
                     "Climber should be done");
        // Copy current climber solution information
        batch_solution_to_return_.get_terminations_status()[climber_strategies_[i].original_index] =
          current_termination_strategy_.get_termination_status(i);
        raft::copy(batch_solution_to_return_.get_primal_solution().data() +
                     climber_strategies_[i].original_index * primal_size_h_,
                   pdhg_solver_.get_potential_next_primal_solution().data() + i * primal_size_h_,
                   primal_size_h_,
                   stream_view_);
        raft::copy(batch_solution_to_return_.get_dual_solution().data() +
                     climber_strategies_[i].original_index * dual_size_h_,
                   pdhg_solver_.get_potential_next_dual_solution().data() + i * dual_size_h_,
                   dual_size_h_,
                   stream_view_);
        raft::copy(
          batch_solution_to_return_.get_reduced_cost().data() +
            climber_strategies_[i].original_index * primal_size_h_,
          current_termination_strategy_.get_convergence_information().get_reduced_cost().data() +
            i * primal_size_h_,
          primal_size_h_,
          stream_view_);
        batch_solution_to_return_
          .get_additional_termination_informations()[climber_strategies_[i].original_index]
          .number_of_steps_taken = total_pdlp_iterations_;
        batch_solution_to_return_
          .get_additional_termination_informations()[climber_strategies_[i].original_index]
          .total_number_of_attempted_steps = pdhg_solver_.get_total_pdhg_iterations();
        batch_solution_to_return_
          .get_additional_termination_informations()[climber_strategies_[i].original_index]
          .solved_by_pdlp = (current_termination_strategy_.get_termination_status(i) !=
                             pdlp_termination_status_t::ConcurrentLimit);
      }
      current_termination_strategy_.fill_gpu_terms_stats(total_pdlp_iterations_);
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));
      current_termination_strategy_.convert_gpu_terms_stats_to_host(
        batch_solution_to_return_.get_additional_termination_informations());
      return optimization_problem_solution_t<i_t, f_t>{
        batch_solution_to_return_.get_primal_solution(),
        batch_solution_to_return_.get_dual_solution(),
        batch_solution_to_return_.get_reduced_cost(),
        get_filled_warmed_start_data(),
        problem_ptr->objective_name,
        problem_ptr->var_names,
        problem_ptr->row_names,
        std::move(batch_solution_to_return_.get_additional_termination_informations()),
        std::move(batch_solution_to_return_.get_terminations_status())};
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));
    return current_termination_strategy_.fill_return_problem_solution(
      internal_solver_iterations_,
      pdhg_solver_,
      pdhg_solver_.get_potential_next_primal_solution(),
      pdhg_solver_.get_potential_next_dual_solution(),
      get_filled_warmed_start_data(),
      std::move(current_termination_strategy_.get_terminations_status()));
  } else if (enable_batch_resizing)  // Some might be optimal, let's remove them from the batch
  {
    raft::common::nvtx::range fun_scope("remove_done_climbers");
    std::unordered_set<i_t> to_remove;
    for (size_t i = 0; i < current_termination_strategy_.get_terminations_status().size(); ++i) {
      // Found one that is done
      if (current_termination_strategy_.is_done(
            current_termination_strategy_.get_termination_status(i))) {
        raft::common::nvtx::range fun_scope("remove_done_climber");
#ifdef BATCH_VERBOSE_MODE
        std::cout << "Removing climber " << i << " because it is done. Its original index is "
                  << climber_strategies_[i].original_index << std::endl;
#endif
        to_remove.emplace(i);
        // Copy current climber solution information
        batch_solution_to_return_.get_terminations_status()[climber_strategies_[i].original_index] =
          current_termination_strategy_.get_termination_status(i);
        raft::copy(batch_solution_to_return_.get_primal_solution().data() +
                     climber_strategies_[i].original_index * primal_size_h_,
                   pdhg_solver_.get_potential_next_primal_solution().data() + i * primal_size_h_,
                   primal_size_h_,
                   stream_view_);
        raft::copy(batch_solution_to_return_.get_dual_solution().data() +
                     climber_strategies_[i].original_index * dual_size_h_,
                   pdhg_solver_.get_potential_next_dual_solution().data() + i * dual_size_h_,
                   dual_size_h_,
                   stream_view_);
        raft::copy(
          batch_solution_to_return_.get_reduced_cost().data() +
            climber_strategies_[i].original_index * primal_size_h_,
          current_termination_strategy_.get_convergence_information().get_reduced_cost().data() +
            i * primal_size_h_,
          primal_size_h_,
          stream_view_);
        batch_solution_to_return_
          .get_additional_termination_informations()[climber_strategies_[i].original_index]
          .number_of_steps_taken = total_pdlp_iterations_;
        batch_solution_to_return_
          .get_additional_termination_informations()[climber_strategies_[i].original_index]
          .total_number_of_attempted_steps = pdhg_solver_.get_total_pdhg_iterations();
        batch_solution_to_return_
          .get_additional_termination_informations()[climber_strategies_[i].original_index]
          .solved_by_pdlp = (current_termination_strategy_.get_termination_status(i) !=
                             pdlp_termination_status_t::ConcurrentLimit);
      }
    }
    if (to_remove.size() > 0) {
      current_termination_strategy_.fill_gpu_terms_stats(total_pdlp_iterations_);
#ifdef BATCH_VERBOSE_MODE
      std::cout << "Removing " << to_remove.size() << " climbers from the batch" << std::endl;
#endif
      resize_and_swap_all_context_loop(to_remove);
    }
  }

  return check_limits(timer);
}

template <typename i_t, typename f_t>
std::optional<optimization_problem_solution_t<i_t, f_t>> pdlp_solver_t<i_t, f_t>::check_termination(
  const timer_t& timer)
{
  raft::common::nvtx::range fun_scope("Check termination");

  // Still need to always compute the termination condition for current even if we don't check them
  // after for kkt restart
  current_termination_strategy_.evaluate_termination_criteria(
    pdhg_solver_,
    (settings_.hyper_params.use_adaptive_step_size_strategy)
      ? pdhg_solver_.get_primal_solution()
      : pdhg_solver_.get_potential_next_primal_solution(),
    (settings_.hyper_params.use_adaptive_step_size_strategy)
      ? pdhg_solver_.get_dual_solution()
      : pdhg_solver_.get_potential_next_dual_solution(),
    pdhg_solver_.get_dual_slack(),
    pdhg_solver_.get_saddle_point_state().get_delta_primal(),
    pdhg_solver_.get_saddle_point_state().get_delta_dual(),
    total_pdlp_iterations_,
    problem_ptr->combined_bounds,
    problem_ptr->objective_coefficients);
#ifdef PDLP_VERBOSE_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  printf("Termination criteria current\n");
  print_termination_criteria(timer, false);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
#endif

  // Check both average and current solution
  if (!settings_.hyper_params.never_restart_to_average) {
    average_termination_strategy_.evaluate_termination_criteria(
      pdhg_solver_,
      unscaled_primal_avg_solution_,
      unscaled_dual_avg_solution_,
      pdhg_solver_.get_dual_slack(),
      restart_strategy_.last_restart_duality_gap_
        .primal_solution_,  // Will not be used since average not used in batch mode
      restart_strategy_.last_restart_duality_gap_
        .dual_solution_,  // Will not be used since average not used in batch mode
      total_pdlp_iterations_,
      problem_ptr->combined_bounds,
      problem_ptr->objective_coefficients);
  }

#ifdef PDLP_VERBOSE_MODE
  if (!settings_.hyper_params.never_restart_to_average) {
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    std::cout << "Termination criteria average:" << std::endl;
    print_termination_criteria(timer, true);
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
  }
#endif

  // We exit directly without checking the termination criteria as some problem can have a low
  // initial redidual + there is by definition 0 gap at first
  // To avoid that we allow at least two iterations at first before checking (in practice 0 wasn't
  // enough) We still need to check iteration and time limit prior without breaking the logic below
  // of first checking termination before the limit
  if (total_pdlp_iterations_ <= 1) {
    print_termination_criteria(timer);
    return check_limits(timer);
  }

  // Handle the batch case separetly
  if (batch_mode_) { return check_batch_termination(timer); }

  // For non-batch mode
  pdlp_termination_status_t termination_current =
    current_termination_strategy_.get_termination_status(0);
  pdlp_termination_status_t termination_average =
    average_termination_strategy_.get_termination_status(0);

  // First check for pdlp_termination_reason_t::Optimality and handle the first primal feasible case

  if (settings_.first_primal_feasible) {
    // Both primal feasible, return best objective
    // TODO later batch mode: handle primal feasible here
    cuopt_expects(!batch_mode_,
                  error_type_t::ValidationError,
                  "First primal feasible is not supported in batch mode");
    if (termination_average == pdlp_termination_status_t::PrimalFeasible &&
        termination_current == pdlp_termination_status_t::PrimalFeasible) {
      const f_t current_overall_primal_residual =
        current_termination_strategy_.get_convergence_information()
          .get_l2_primal_residual()
          .element(0, stream_view_);
      const f_t average_overall_primal_residual =
        average_termination_strategy_.get_convergence_information()
          .get_l2_primal_residual()
          .element(0, stream_view_);
      if (current_overall_primal_residual < average_overall_primal_residual) {
        return current_termination_strategy_.fill_return_problem_solution(
          internal_solver_iterations_,
          pdhg_solver_,
          (settings_.hyper_params.use_adaptive_step_size_strategy)
            ? pdhg_solver_.get_primal_solution()
            : pdhg_solver_.get_potential_next_primal_solution(),
          (settings_.hyper_params.use_adaptive_step_size_strategy)
            ? pdhg_solver_.get_dual_solution()
            : pdhg_solver_.get_potential_next_dual_solution(),
          get_filled_warmed_start_data(),
          current_termination_strategy_.get_terminations_status());
      } else  // Average has better overall residual
      {
        return average_termination_strategy_.fill_return_problem_solution(
          internal_solver_iterations_,
          pdhg_solver_,
          unscaled_primal_avg_solution_,
          unscaled_dual_avg_solution_,
          get_filled_warmed_start_data(),
          {termination_average});
      }
    } else if (termination_current == pdlp_termination_status_t::PrimalFeasible) {
      return current_termination_strategy_.fill_return_problem_solution(
        internal_solver_iterations_,
        pdhg_solver_,
        (settings_.hyper_params.use_adaptive_step_size_strategy)
          ? pdhg_solver_.get_primal_solution()
          : pdhg_solver_.get_potential_next_primal_solution(),
        (settings_.hyper_params.use_adaptive_step_size_strategy)
          ? pdhg_solver_.get_dual_solution()
          : pdhg_solver_.get_potential_next_dual_solution(),
        get_filled_warmed_start_data(),
        {termination_current});
    } else if (termination_average == pdlp_termination_status_t::PrimalFeasible) {
      return average_termination_strategy_.fill_return_problem_solution(
        internal_solver_iterations_,
        pdhg_solver_,
        unscaled_primal_avg_solution_,
        unscaled_dual_avg_solution_,
        get_filled_warmed_start_data(),
        {termination_average});
    }
    // else neither of the two is primal feasible
  }

  // If both are pdlp_termination_status_t::Optimal, return the one with the lowest KKT score
  if (average_termination_strategy_.all_optimal_status() &&
      current_termination_strategy_.all_optimal_status()) {
    cuopt_assert(!batch_mode_,
                 "Should never have both current and average optimal in batch mode (since using "
                 "PDLP+ algorithm)");
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
      print_final_termination_criteria(
        timer, current_termination_strategy_.get_convergence_information(), termination_current);
      return current_termination_strategy_.fill_return_problem_solution(
        internal_solver_iterations_,
        pdhg_solver_,
        (settings_.hyper_params.use_adaptive_step_size_strategy)
          ? pdhg_solver_.get_primal_solution()
          : pdhg_solver_.get_potential_next_primal_solution(),
        (settings_.hyper_params.use_adaptive_step_size_strategy)
          ? pdhg_solver_.get_dual_solution()
          : pdhg_solver_.get_potential_next_dual_solution(),
        get_filled_warmed_start_data(),
        std::move(current_termination_strategy_.get_terminations_status()));
    } else {
#ifdef PDLP_VERBOSE_MODE
      std::cout << "Optimal. End total number of iteration average=" << internal_solver_iterations_
                << std::endl;
#endif
      print_final_termination_criteria(timer,
                                       average_termination_strategy_.get_convergence_information(),
                                       termination_average,
                                       true);
      return average_termination_strategy_.fill_return_problem_solution(
        internal_solver_iterations_,
        pdhg_solver_,
        unscaled_primal_avg_solution_,
        unscaled_dual_avg_solution_,
        get_filled_warmed_start_data(),
        std::move(average_termination_strategy_.get_terminations_status()));
    }
  }

  // If at least one is pdlp_termination_status_t::Optimal, return it
  if (average_termination_strategy_.all_optimal_status()) {
    cuopt_assert(!batch_mode_,
                 "Should never have average optimal in batch mode (since using PDLP+ algorithm)");
#ifdef PDLP_VERBOSE_MODE
    std::cout << "Optimal. End total number of iteration average=" << internal_solver_iterations_
              << std::endl;
#endif
    print_final_termination_criteria(timer,
                                     average_termination_strategy_.get_convergence_information(),
                                     termination_average,
                                     true);
    return average_termination_strategy_.fill_return_problem_solution(
      internal_solver_iterations_,
      pdhg_solver_,
      unscaled_primal_avg_solution_,
      unscaled_dual_avg_solution_,
      get_filled_warmed_start_data(),
      std::move(average_termination_strategy_.get_terminations_status()));
  }
  if (current_termination_strategy_.all_optimal_status()) {
#ifdef PDLP_VERBOSE_MODE
    std::cout << "Optimal. End total number of iteration current=" << internal_solver_iterations_
              << std::endl;
#endif
    print_final_termination_criteria(
      timer, current_termination_strategy_.get_convergence_information(), termination_current);
    return current_termination_strategy_.fill_return_problem_solution(
      internal_solver_iterations_,
      pdhg_solver_,
      (settings_.hyper_params.use_adaptive_step_size_strategy)
        ? pdhg_solver_.get_primal_solution()
        : pdhg_solver_.get_potential_next_primal_solution(),
      (settings_.hyper_params.use_adaptive_step_size_strategy)
        ? pdhg_solver_.get_dual_solution()
        : pdhg_solver_.get_potential_next_dual_solution(),
      get_filled_warmed_start_data(),
      std::move(current_termination_strategy_.get_terminations_status()));
  }

  // Check for infeasibility

  // If strict infeasibility, any infeasibility is detected, it is returned
  // Else both are needed (unless there is no average)
  // (If infeasibility_detection is not set, termination reason cannot be Infeasible)
  if (settings_.detect_infeasibility) {
    if (settings_.strict_infeasibility || settings_.hyper_params.never_restart_to_average) {
      if (termination_current == pdlp_termination_status_t::PrimalInfeasible ||
          termination_current == pdlp_termination_status_t::DualInfeasible) {
#ifdef PDLP_VERBOSE_MODE
        std::cout << "Current Infeasible. End total number of iteration current="
                  << internal_solver_iterations_ << std::endl;
#endif
        print_final_termination_criteria(
          timer, current_termination_strategy_.get_convergence_information(), termination_current);
        return current_termination_strategy_.fill_return_problem_solution(
          internal_solver_iterations_,
          pdhg_solver_,
          (settings_.hyper_params.use_adaptive_step_size_strategy)
            ? pdhg_solver_.get_primal_solution()
            : pdhg_solver_.get_potential_next_primal_solution(),
          (settings_.hyper_params.use_adaptive_step_size_strategy)
            ? pdhg_solver_.get_dual_solution()
            : pdhg_solver_.get_potential_next_dual_solution(),
          std::move(current_termination_strategy_.get_terminations_status()));
      }
      if (termination_average == pdlp_termination_status_t::PrimalInfeasible ||
          termination_average == pdlp_termination_status_t::DualInfeasible) {
#ifdef PDLP_VERBOSE_MODE
        std::cout << "Average Infeasible. End total number of iteration current="
                  << internal_solver_iterations_ << std::endl;
#endif
        print_final_termination_criteria(
          timer,
          average_termination_strategy_.get_convergence_information(),
          termination_average,
          true);
        return average_termination_strategy_.fill_return_problem_solution(
          internal_solver_iterations_,
          pdhg_solver_,
          unscaled_primal_avg_solution_,
          unscaled_dual_avg_solution_,
          std::move(average_termination_strategy_.get_terminations_status()));
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
        print_final_termination_criteria(
          timer, current_termination_strategy_.get_convergence_information(), termination_current);
        return current_termination_strategy_.fill_return_problem_solution(
          internal_solver_iterations_,
          pdhg_solver_,
          (settings_.hyper_params.use_adaptive_step_size_strategy)
            ? pdhg_solver_.get_primal_solution()
            : pdhg_solver_.get_potential_next_primal_solution(),
          (settings_.hyper_params.use_adaptive_step_size_strategy)
            ? pdhg_solver_.get_dual_solution()
            : pdhg_solver_.get_potential_next_dual_solution(),
          std::move(current_termination_strategy_.get_terminations_status()));
      }
    }
  }

  // Numerical error has happend (movement is 0 and pdlp_termination_status_t::Optimality has not
  // been reached)
  if (step_size_strategy_.get_valid_step_size() == -1) {
    cuopt_assert(!batch_mode_,
                 "Step size can never be invalid in bath mode (since using PDLP+ algorithm)");
#ifdef PDLP_VERBOSE_MODE
    std::cout << "Numerical Error. End total number of iteration current="
              << internal_solver_iterations_ << std::endl;
#endif
    print_final_termination_criteria(
      timer, current_termination_strategy_.get_convergence_information(), termination_current);
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
  if (total_pdlp_iterations_ % 1000 == 0) { print_termination_criteria(timer); }

  // No reason to terminate
  return check_limits(timer);
}

template <typename f_t>
static void compute_stats(const rmm::device_uvector<f_t>& vec,
                          f_t& smallest,
                          f_t& largest,
                          f_t& avg)
{
  auto abs_op      = [] __host__ __device__(f_t x) { return abs(x); };
  auto min_nonzero = [] __host__ __device__(f_t x)
    -> f_t { return x == 0 ? std::numeric_limits<f_t>::max() : abs(x); };

  cuopt_assert(vec.size() > 0, "Vector must not be empty");

  auto stream = vec.stream();
  size_t n    = vec.size();

  rmm::device_scalar<f_t> d_smallest(stream);
  rmm::device_scalar<f_t> d_largest(stream);
  rmm::device_scalar<f_t> d_sum(stream);

  auto min_nz_iter = thrust::make_transform_iterator(vec.cbegin(), min_nonzero);
  auto abs_iter    = thrust::make_transform_iterator(vec.cbegin(), abs_op);

  void* d_temp   = nullptr;
  size_t bytes_1 = 0, bytes_2 = 0, bytes_3 = 0;
  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(d_temp,
                                          bytes_1,
                                          min_nz_iter,
                                          d_smallest.data(),
                                          n,
                                          cuda::minimum<>{},
                                          std::numeric_limits<f_t>::max(),
                                          stream));
  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(
    d_temp, bytes_2, abs_iter, d_largest.data(), n, cuda::maximum<>{}, f_t(0), stream));
  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(
    d_temp, bytes_3, abs_iter, d_sum.data(), n, cuda::std::plus<>{}, f_t(0), stream));

  size_t max_bytes = std::max({bytes_1, bytes_2, bytes_3});
  rmm::device_buffer temp_buf(max_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(temp_buf.data(),
                                          bytes_1,
                                          min_nz_iter,
                                          d_smallest.data(),
                                          n,
                                          cuda::minimum<>{},
                                          std::numeric_limits<f_t>::max(),
                                          stream));
  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(
    temp_buf.data(), bytes_2, abs_iter, d_largest.data(), n, cuda::maximum<>{}, f_t(0), stream));
  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(
    temp_buf.data(), bytes_3, abs_iter, d_sum.data(), n, cuda::std::plus<>{}, f_t(0), stream));

  smallest = d_smallest.value(stream);
  largest  = d_largest.value(stream);
  avg      = d_sum.value(stream) / vec.size();
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
    cuopt_assert(pdhg_solver_.get_primal_solution().size() == primal.value()->size(),
                 "Both of those should have equal size");
    raft::copy(pdhg_solver_.get_primal_solution().data(),
               primal.value()->data(),
               pdhg_solver_.get_primal_solution().size(),
               stream_view_);
    if (settings_.hyper_params.use_reflected_primal_dual) {
      raft::copy(pdhg_solver_.get_potential_next_primal_solution().data(),
                 primal.value()->data(),
                 pdhg_solver_.get_potential_next_primal_solution().size(),
                 stream_view_);
      raft::copy(restart_strategy_.last_restart_duality_gap_.primal_solution_.data(),
                 primal.value()->data(),
                 restart_strategy_.last_restart_duality_gap_.primal_solution_.size(),
                 stream_view_);
    }
  }
  if (dual) {
    cuopt_assert(pdhg_solver_.get_dual_solution().size() == dual.value()->size(),
                 "Both of those should have equal size");
    raft::copy(pdhg_solver_.get_dual_solution().data(),
               dual.value()->data(),
               pdhg_solver_.get_dual_solution().size(),
               stream_view_);
    if (settings_.hyper_params.use_reflected_primal_dual) {
      raft::copy(pdhg_solver_.get_potential_next_dual_solution().data(),
                 dual.value()->data(),
                 pdhg_solver_.get_potential_next_dual_solution().size(),
                 stream_view_);
      raft::copy(restart_strategy_.last_restart_duality_gap_.dual_solution_.data(),
                 dual.value()->data(),
                 restart_strategy_.last_restart_duality_gap_.dual_solution_.size(),
                 stream_view_);
    }
  }

  // Handle initial step size if needed

  if (settings_.hyper_params.update_step_size_on_initial_solution) {
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
      if (!settings_.hyper_params.compute_initial_step_size_before_scaling) {
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
      if (settings_.hyper_params.compute_initial_step_size_before_scaling) {
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
  if (!settings_.hyper_params.compute_initial_primal_weight_before_scaling) {
#ifdef PDLP_DEBUG_MODE
    std::cout << "      Scaling before computing initial primal weight:" << std::endl;
#endif
    initial_scaling_strategy_.scale_solutions(pdhg_solver_.get_primal_solution(),
                                              pdhg_solver_.get_dual_solution());
    if (settings_.hyper_params.use_reflected_primal_dual) {
      initial_scaling_strategy_.scale_solutions(pdhg_solver_.get_potential_next_primal_solution(),
                                                pdhg_solver_.get_potential_next_dual_solution());
      initial_scaling_strategy_.scale_solutions(
        restart_strategy_.last_restart_duality_gap_.primal_solution_,
        restart_strategy_.last_restart_duality_gap_.dual_solution_);
    }
  }

  // If only primal or dual is set, the primal weight wont (as it can't) be updated
  if (settings_.hyper_params.update_primal_weight_on_initial_solution) {
#ifdef PDLP_DEBUG_MODE
    std::cout << "      Updating the initial primal weight on initial solution" << std::endl;
#endif
    restart_strategy_.update_distance(
      pdhg_solver_, primal_weight_, primal_step_size_, dual_step_size_, step_size_);
  }

  // We scale here because it was not done previously
  if (settings_.hyper_params.compute_initial_primal_weight_before_scaling) {
    initial_scaling_strategy_.scale_solutions(pdhg_solver_.get_primal_solution(),
                                              pdhg_solver_.get_dual_solution());
    if (settings_.hyper_params.use_reflected_primal_dual) {
      initial_scaling_strategy_.scale_solutions(pdhg_solver_.get_potential_next_primal_solution(),
                                                pdhg_solver_.get_potential_next_dual_solution());
      initial_scaling_strategy_.scale_solutions(
        restart_strategy_.last_restart_duality_gap_.primal_solution_,
        restart_strategy_.last_restart_duality_gap_.dual_solution_);
    }
  }
}

template <typename f_t>
HDI void fixed_error_computation(const f_t norm_squared_delta_primal,
                                 const f_t norm_squared_delta_dual,
                                 const f_t primal_weight,
                                 const f_t step_size,
                                 const f_t interaction,
                                 f_t* fixed_point_error)
{
  cuopt_assert(!isnan(norm_squared_delta_primal), "norm_squared_delta_primal must not be NaN");
  cuopt_assert(!isnan(norm_squared_delta_dual), "norm_squared_delta_dual must not be NaN");
  cuopt_assert(!isnan(primal_weight), "primal_weight must not be NaN");
  cuopt_assert(!isnan(step_size), "step_size must not be NaN");
  cuopt_assert(!isnan(interaction), "interaction must not be NaN");
  cuopt_assert(norm_squared_delta_primal >= f_t(0.0), "norm_squared_delta_primal must be >= 0");
  cuopt_assert(norm_squared_delta_dual >= f_t(0.0), "norm_squared_delta_dual must be >= 0");
  cuopt_assert(primal_weight > f_t(0.0), "primal_weight must be > 0");
  cuopt_assert(step_size > f_t(0.0), "step_size must be > 0");

  const f_t movement =
    norm_squared_delta_primal * primal_weight + norm_squared_delta_dual / primal_weight;
  const f_t computed_interaction = f_t(2.0) * interaction * step_size;

  cuopt_assert(movement + computed_interaction >= f_t(0.0),
               "Movement + computed interaction must be >= 0");

  // Clamp to 0 to avoid NaN
  *fixed_point_error = cuda::std::sqrt(cuda::std::max(f_t(0.0), movement + computed_interaction));

#ifdef CUPDLP_DEBUG_MODE
  printf("movement %lf\n", movement);
  printf("interaction %lf\n", interaction);
  printf("state->fixed_point_error %lf\n", *fixed_point_error);
#endif
}

template <typename f_t>
__global__ void kernel_compute_fixed_error(raft::device_span<const f_t> norm_squared_delta_primal,
                                           raft::device_span<const f_t> norm_squared_delta_dual,
                                           raft::device_span<const f_t> primal_weight,
                                           raft::device_span<const f_t> step_size,
                                           raft::device_span<const f_t> interaction,
                                           raft::device_span<f_t> fixed_point_error)
{
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  cuopt_assert(norm_squared_delta_primal.size() == norm_squared_delta_dual.size() &&
                 norm_squared_delta_primal.size() == primal_weight.size() &&
                 norm_squared_delta_primal.size() == step_size.size() &&
                 norm_squared_delta_primal.size() == interaction.size() &&
                 norm_squared_delta_primal.size() == fixed_point_error.size(),
               "All vectors must have the same size");
  if (index >= norm_squared_delta_primal.size()) { return; }
  fixed_error_computation<f_t>(norm_squared_delta_primal[index],
                               norm_squared_delta_dual[index],
                               primal_weight[index],
                               step_size[index],
                               interaction[index],
                               &fixed_point_error[index]);
}

template <typename i_t, typename f_t>
__global__ void pdlp_swap_device_vectors_kernel(const swap_pair_t<i_t>* swap_pairs,
                                                i_t swap_count,
                                                raft::device_span<f_t> primal_weight,
                                                raft::device_span<f_t> best_primal_weight,
                                                raft::device_span<f_t> step_size,
                                                raft::device_span<f_t> primal_step_size,
                                                raft::device_span<f_t> dual_step_size)
{
  const i_t idx = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= swap_count) { return; }

  const i_t left  = swap_pairs[idx].left;
  const i_t right = swap_pairs[idx].right;

  cuda::std::swap(primal_weight[left], primal_weight[right]);
  cuda::std::swap(best_primal_weight[left], best_primal_weight[right]);
  cuda::std::swap(step_size[left], step_size[right]);
  cuda::std::swap(primal_step_size[left], primal_step_size[right]);
  cuda::std::swap(dual_step_size[left], dual_step_size[right]);
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::swap_context(
  const thrust::universal_host_pinned_vector<swap_pair_t<i_t>>& swap_pairs)
{
  if (swap_pairs.empty()) { return; }

  const auto batch_size = static_cast<i_t>(primal_weight_.size());
  cuopt_assert(batch_size > 0, "Batch size must be greater than 0");
  for (const auto& pair : swap_pairs) {
    cuopt_assert(pair.left < pair.right, "Left swap index must be less than right swap index");
    cuopt_assert(pair.right < batch_size, "Right swap index is out of bounds");
  }

  const auto [grid_size, block_size] =
    kernel_config_from_batch_size(static_cast<i_t>(swap_pairs.size()));
  pdlp_swap_device_vectors_kernel<i_t, f_t>
    <<<grid_size, block_size, 0, stream_view_>>>(thrust::raw_pointer_cast(swap_pairs.data()),
                                                 static_cast<i_t>(swap_pairs.size()),
                                                 make_span(primal_weight_),
                                                 make_span(best_primal_weight_),
                                                 make_span(step_size_),
                                                 make_span(primal_step_size_),
                                                 make_span(dual_step_size_));
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::resize_context(i_t new_size)
{
  [[maybe_unused]] const auto batch_size = static_cast<i_t>(primal_weight_.size());
  cuopt_assert(batch_size > 0, "Batch size must be greater than 0");
  cuopt_assert(new_size > 0, "New size must be greater than 0");
  cuopt_assert(new_size < batch_size, "New size must be less than batch size");

  primal_weight_.resize(new_size, stream_view_);
  best_primal_weight_.resize(new_size, stream_view_);
  step_size_.resize(new_size, stream_view_);
  primal_step_size_.resize(new_size, stream_view_);
  dual_step_size_.resize(new_size, stream_view_);

  climber_strategies_.resize(new_size);
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::swap_all_context(
  const thrust::universal_host_pinned_vector<swap_pair_t<i_t>>& swap_pairs)
{
  if (swap_pairs.empty()) { return; }

  raft::common::nvtx::range fun_scope("swap_all_context");

  pdhg_solver_.swap_context(swap_pairs);
  restart_strategy_.swap_context(swap_pairs);
  swap_context(swap_pairs);
  step_size_strategy_.swap_context(swap_pairs);
  current_termination_strategy_.swap_context(swap_pairs);

  for (const auto& pair : swap_pairs) {
    host_vector_swap(climber_strategies_, pair.left, pair.right);
  }

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::resize_all_context(i_t new_size)
{
  raft::common::nvtx::range fun_scope("resize_all_context");

  // Resize PDHG, its saddle point and its new bounds
  pdhg_solver_.resize_context(new_size);
  // Resize restart strategy and its duality gap container
  restart_strategy_.resize_context(new_size);
  // Resize step size strategy
  step_size_strategy_.resize_context(new_size);
  // Resize current termination strategy and its convergence information
  current_termination_strategy_.resize_context(new_size);
  // Resize PDLP own context
  resize_context(new_size);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::resize_and_swap_all_context_loop(
  const std::unordered_set<i_t>& climber_strategies_to_remove)
{
  raft::common::nvtx::range fun_scope("resize_and_swap_all_context_loop");

  cuopt_assert(climber_strategies_to_remove.size() != climber_strategies_.size(),
               "We should never remove all climbers");
  cuopt_assert(
    climber_strategies_to_remove.size() < climber_strategies_.size(),
    "climber_strategies_to_remove size must be less than or equal to climber_strategies_.size()");

  thrust::universal_host_pinned_vector<swap_pair_t<i_t>> swap_pairs;

  // Here we accumulate all the swap pairs that need to be done to remove the climbers
  // Then execute all the swaps and resizes
  i_t last = climber_strategies_.size() - 1;
  for (i_t i = 0; i <= last; ++i) {
    // Not to remove, skip this id
    if (climber_strategies_to_remove.find(i) == climber_strategies_to_remove.end()) continue;

    // While the last id part of the remove set, we decrement
    // Climbers to remove that are in the end are already where they should be
    while (climber_strategies_to_remove.find(last) != climber_strategies_to_remove.end() &&
           last >= i)
      --last;

    // If both ids cross: all the to remove are in the end, we can break
    if (i >= last) break;

    swap_pairs.push_back({i, last});
    --last;
  }

  // No swap can happen if all climbers to remove are at the end
  if (!swap_pairs.empty()) { swap_all_context(swap_pairs); }

  cuopt_assert(
    last + 1 == climber_strategies_.size() - climber_strategies_to_remove.size(),
    "Last + 1 must be equal to climber_strategies_.size() - climber_strategies_to_remove.size()");
  resize_all_context(last + 1);

#ifdef BATCH_VERBOSE_MODE
  std::cout << "Batch size is now " << climber_strategies_.size() << ". Climbers left: ";
  for (size_t i = 0; i < climber_strategies_.size(); ++i) {
    std::cout << climber_strategies_[i].original_index << " ";
  }
  std::cout << std::endl;
#endif

  // Reset all cusparse view

  // Reset cuSparse views for PDHG
  auto& pdhg_cusparse_view = pdhg_solver_.get_cusparse_view();
  pdhg_cusparse_view.batch_dual_solutions.create(
    op_problem_scaled_.n_constraints,
    climber_strategies_.size(),
    climber_strategies_.size(),
    pdhg_solver_.get_saddle_point_state().get_dual_solution().data(),
    CUSPARSE_ORDER_ROW);
  pdhg_cusparse_view.batch_current_AtYs.create(
    op_problem_scaled_.n_variables,
    climber_strategies_.size(),
    climber_strategies_.size(),
    pdhg_solver_.get_saddle_point_state().get_current_AtY().data(),
    CUSPARSE_ORDER_ROW);
  pdhg_cusparse_view.batch_reflected_primal_solutions.create(
    op_problem_scaled_.n_variables,
    climber_strategies_.size(),
    climber_strategies_.size(),
    pdhg_solver_.get_reflected_primal().data(),
    CUSPARSE_ORDER_ROW);
  pdhg_cusparse_view.batch_dual_gradients.create(
    op_problem_scaled_.n_constraints,
    climber_strategies_.size(),
    climber_strategies_.size(),
    pdhg_solver_.get_saddle_point_state().get_dual_gradient().data(),
    CUSPARSE_ORDER_ROW);

  // Reset cusparse view used by adaptive step size strategy but owned by PDHG
  pdhg_cusparse_view.batch_potential_next_dual_solution.create(
    op_problem_scaled_.n_constraints,
    climber_strategies_.size(),
    op_problem_scaled_.n_constraints,
    pdhg_solver_.get_potential_next_dual_solution().data(),
    CUSPARSE_ORDER_COL);
  pdhg_cusparse_view.batch_next_AtYs.create(
    op_problem_scaled_.n_variables,
    climber_strategies_.size(),
    op_problem_scaled_.n_variables,
    pdhg_solver_.get_saddle_point_state().get_next_AtY().data(),
    CUSPARSE_ORDER_COL);

  // Reset cusparse view used by convergence information but owned by PDLP
  current_op_problem_evaluation_cusparse_view_.batch_primal_solutions.create(
    op_problem_scaled_.n_variables,
    climber_strategies_.size(),
    op_problem_scaled_.n_variables,
    pdhg_solver_.get_potential_next_primal_solution().data(),
    CUSPARSE_ORDER_COL);
  current_op_problem_evaluation_cusparse_view_.batch_dual_solutions.create(
    op_problem_scaled_.n_constraints,
    climber_strategies_.size(),
    op_problem_scaled_.n_constraints,
    pdhg_solver_.get_potential_next_dual_solution().data(),
    CUSPARSE_ORDER_COL);
  current_op_problem_evaluation_cusparse_view_.batch_tmp_duals.create(
    op_problem_scaled_.n_constraints,
    climber_strategies_.size(),
    op_problem_scaled_.n_constraints,
    pdhg_solver_.get_dual_tmp_resource().data(),
    CUSPARSE_ORDER_COL);
  current_op_problem_evaluation_cusparse_view_.batch_tmp_primals.create(
    op_problem_scaled_.n_variables,
    climber_strategies_.size(),
    op_problem_scaled_.n_variables,
    pdhg_solver_.get_primal_tmp_resource().data(),
    CUSPARSE_ORDER_COL);

  // Rerun preprocess

  // PDHG SpMM preprocess
#if CUDA_VER_12_4_UP
  my_cusparsespmm_preprocess(
    handle_ptr_->get_cusparse_handle(),
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    reusable_device_scalar_value_1_.data(),
    pdhg_cusparse_view.A_T,
    pdhg_cusparse_view.batch_dual_solutions,
    reusable_device_scalar_value_0_.data(),
    pdhg_cusparse_view.batch_current_AtYs,
    (deterministic_batch_pdlp) ? CUSPARSE_SPMM_CSR_ALG3 : CUSPARSE_SPMM_CSR_ALG2,
    pdhg_cusparse_view.buffer_transpose_batch_row_row_.data(),
    stream_view_);
  my_cusparsespmm_preprocess(
    handle_ptr_->get_cusparse_handle(),
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    reusable_device_scalar_value_1_.data(),
    pdhg_cusparse_view.A,
    pdhg_cusparse_view.batch_reflected_primal_solutions,
    reusable_device_scalar_value_0_.data(),
    pdhg_cusparse_view.batch_dual_gradients,
    (deterministic_batch_pdlp) ? CUSPARSE_SPMM_CSR_ALG3 : CUSPARSE_SPMM_CSR_ALG2,
    pdhg_cusparse_view.buffer_non_transpose_batch_row_row_.data(),
    stream_view_);

  // Adaptive step size strategy SpMM preprocess
  my_cusparsespmm_preprocess(handle_ptr_->get_cusparse_handle(),
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             reusable_device_scalar_value_1_.data(),
                             pdhg_cusparse_view.A_T,
                             pdhg_cusparse_view.batch_potential_next_dual_solution,
                             reusable_device_scalar_value_0_.data(),
                             pdhg_cusparse_view.batch_next_AtYs,
                             CUSPARSE_SPMM_CSR_ALG3,
                             (f_t*)pdhg_cusparse_view.buffer_transpose_batch.data(),
                             stream_view_);

  // Convergence information SpMM preprocess
  my_cusparsespmm_preprocess(
    handle_ptr_->get_cusparse_handle(),
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    reusable_device_scalar_value_1_.data(),
    current_op_problem_evaluation_cusparse_view_.A_T,
    current_op_problem_evaluation_cusparse_view_.batch_dual_solutions,
    reusable_device_scalar_value_0_.data(),
    current_op_problem_evaluation_cusparse_view_.batch_tmp_primals,
    CUSPARSE_SPMM_CSR_ALG3,
    (f_t*)current_op_problem_evaluation_cusparse_view_.buffer_transpose_batch.data(),
    stream_view_);

  my_cusparsespmm_preprocess(
    handle_ptr_->get_cusparse_handle(),
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    reusable_device_scalar_value_1_.data(),
    current_op_problem_evaluation_cusparse_view_.A,
    current_op_problem_evaluation_cusparse_view_.batch_primal_solutions,
    reusable_device_scalar_value_0_.data(),
    current_op_problem_evaluation_cusparse_view_.batch_tmp_duals,
    CUSPARSE_SPMM_CSR_ALG3,
    (f_t*)current_op_problem_evaluation_cusparse_view_.buffer_non_transpose_batch.data(),
    stream_view_);
#endif

  // Set PDHG graph to unitilized so that next call can start a new graph
  // Currently graph capture is not supported for cuSparse SpMM
  // TODO enable once cuSparse SpMM supports graph capture
  pdhg_solver_.get_graph_all() = ping_pong_graph_t<i_t>(stream_view_, true);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::compute_fixed_error(std::vector<int>& has_restarted)
{
  raft::common::nvtx::range fun_scope("compute_fixed_error");

#ifdef CUPDLP_DEBUG_MODE
  printf("Computing compute_fixed_point_error \n");
#endif
  cuopt_assert(
    pdhg_solver_.get_reflected_primal().size() == primal_size_h_ * climber_strategies_.size(),
    "reflected_primal_ size mismatch");
  cuopt_assert(
    pdhg_solver_.get_reflected_dual().size() == dual_size_h_ * climber_strategies_.size(),
    "reflected_dual_ size mismatch");
  cuopt_assert(
    pdhg_solver_.get_primal_solution().size() == primal_size_h_ * climber_strategies_.size(),
    "primal_solution_ size mismatch");
  cuopt_assert(pdhg_solver_.get_dual_solution().size() == dual_size_h_ * climber_strategies_.size(),
               "dual_solution_ size mismatch");
  cuopt_assert(pdhg_solver_.get_saddle_point_state().get_delta_primal().size() ==
                 primal_size_h_ * climber_strategies_.size(),
               "delta_primal_ size mismatch");
  cuopt_assert(pdhg_solver_.get_saddle_point_state().get_delta_dual().size() ==
                 dual_size_h_ * climber_strategies_.size(),
               "delta_dual_ size mismatch");

  // Computing the deltas
  // TODO batch mdoe: this only works if everyone restarts
  cub::DeviceTransform::Transform(cuda::std::make_tuple(pdhg_solver_.get_reflected_primal().data(),
                                                        pdhg_solver_.get_primal_solution().data()),
                                  pdhg_solver_.get_saddle_point_state().get_delta_primal().data(),
                                  pdhg_solver_.get_primal_solution().size(),
                                  cuda::std::minus<f_t>{},
                                  stream_view_.value());
  cub::DeviceTransform::Transform(cuda::std::make_tuple(pdhg_solver_.get_reflected_dual().data(),
                                                        pdhg_solver_.get_dual_solution().data()),
                                  pdhg_solver_.get_saddle_point_state().get_delta_dual().data(),
                                  pdhg_solver_.get_dual_solution().size(),
                                  cuda::std::minus<f_t>{},
                                  stream_view_.value());

  auto& cusparse_view = pdhg_solver_.get_cusparse_view();
  // Sync to make sure all previous cuSparse operations are finished before setting the
  // potential_next_dual_solution
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));

  // Make potential_next_dual_solution point towards reflected dual solution to reuse the code
  RAFT_CUSPARSE_TRY(cusparseDnVecSetValues(cusparse_view.potential_next_dual_solution,
                                           (void*)pdhg_solver_.get_reflected_dual().data()));

  if (batch_mode_)
    RAFT_CUSPARSE_TRY(cusparseDnMatSetValues(cusparse_view.batch_potential_next_dual_solution,
                                             (void*)pdhg_solver_.get_reflected_dual().data()));

  step_size_strategy_.compute_interaction_and_movement(
    pdhg_solver_.get_primal_tmp_resource(), cusparse_view, pdhg_solver_.get_saddle_point_state());

  if (batch_mode_) {
    const auto [grid_size, block_size] = kernel_config_from_batch_size(climber_strategies_.size());
    kernel_compute_fixed_error<f_t><<<grid_size, block_size, 0, stream_view_>>>(
      make_span(step_size_strategy_.get_norm_squared_delta_primal()),
      make_span(step_size_strategy_.get_norm_squared_delta_dual()),
      make_span(primal_weight_),
      make_span(step_size_),
      make_span(step_size_strategy_.get_interaction()),
      make_span(restart_strategy_.fixed_point_error_));
    RAFT_CUDA_TRY(cudaStreamSynchronize(
      stream_view_));  // To make sure all the data is written from device to host
    RAFT_CUDA_TRY(cudaPeekAtLastError());

#ifdef CUPDLP_DEBUG_MODE
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
#endif
  } else {
    fixed_error_computation<f_t>(step_size_strategy_.get_norm_squared_delta_primal(0),
                                 step_size_strategy_.get_norm_squared_delta_dual(0),
                                 primal_weight_.element(0, stream_view_),
                                 step_size_.element(0, stream_view_),
                                 step_size_strategy_.get_interaction(0),
                                 &restart_strategy_.fixed_point_error_[0]);
  }

  // Sync to make sure all previous cuSparse operations are finished before setting the
  // potential_next_dual_solution
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));

  // Put back
  RAFT_CUSPARSE_TRY(
    cusparseDnVecSetValues(cusparse_view.potential_next_dual_solution,
                           (void*)pdhg_solver_.get_potential_next_dual_solution().data()));

  if (batch_mode_) {
    RAFT_CUSPARSE_TRY(
      cusparseDnMatSetValues(cusparse_view.batch_potential_next_dual_solution,
                             (void*)pdhg_solver_.get_potential_next_dual_solution().data()));
  }

#ifdef CUPDLP_DEBUG_MODE
  for (size_t i = 0; i < climber_strategies_.size(); ++i) {
    printf("fixed_point_error %lf\n", restart_strategy_.fixed_point_error_[i]);
  }
#endif

  for (size_t i = 0; i < climber_strategies_.size(); ++i) {
    cuopt_assert(!std::isnan(restart_strategy_.fixed_point_error_[i]),
                 "fixed_point_error_ must not be NaN after compute_fixed_error");
    cuopt_assert(restart_strategy_.fixed_point_error_[i] >= f_t(0.0),
                 "fixed_point_error_ must be >= 0 after compute_fixed_error");
    if (has_restarted[i]) {
      restart_strategy_.initial_fixed_point_error_[i] = restart_strategy_.fixed_point_error_[i];
      cuopt_assert(!std::isnan(restart_strategy_.initial_fixed_point_error_[i]),
                   "initial_fixed_point_error_ must not be NaN after assignment");
      has_restarted[i] = false;
    }
  }
}

// Tranpose all the data we use in termination condition and restart:
// potential_next_primal_solution, potential_next_dual_solution, dual_slack
template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::transpose_primal_dual_to_row(
  rmm::device_uvector<f_t>& primal_to_transpose,
  rmm::device_uvector<f_t>& dual_to_transpose,
  rmm::device_uvector<f_t>& dual_slack_to_transpose)
{
  bool is_dual_slack_empty = dual_slack_to_transpose.size() == 0;
  rmm::device_uvector<f_t> primal_transposed(primal_size_h_ * climber_strategies_.size(),
                                             stream_view_);
  rmm::device_uvector<f_t> dual_transposed(dual_size_h_ * climber_strategies_.size(), stream_view_);
  rmm::device_uvector<f_t> dual_slack_transposed(
    is_dual_slack_empty ? 0 : primal_size_h_ * climber_strategies_.size(), stream_view_);

  RAFT_CUBLAS_TRY(cublasSetStream(handle_ptr_->get_cublas_handle(), stream_view_));
  CUBLAS_CHECK(cublasDgeam(handle_ptr_->get_cublas_handle(),
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           climber_strategies_.size(),
                           primal_size_h_,
                           reusable_device_scalar_value_1_.data(),
                           primal_to_transpose.data(),
                           primal_size_h_,
                           reusable_device_scalar_value_0_.data(),
                           nullptr,
                           climber_strategies_.size(),
                           primal_transposed.data(),
                           climber_strategies_.size()));

  if (!is_dual_slack_empty) {
    CUBLAS_CHECK(cublasDgeam(handle_ptr_->get_cublas_handle(),
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             climber_strategies_.size(),
                             primal_size_h_,
                             reusable_device_scalar_value_1_.data(),
                             dual_slack_to_transpose.data(),
                             primal_size_h_,
                             reusable_device_scalar_value_0_.data(),
                             nullptr,
                             climber_strategies_.size(),
                             dual_slack_transposed.data(),
                             climber_strategies_.size()));
  }
  CUBLAS_CHECK(cublasDgeam(handle_ptr_->get_cublas_handle(),
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           climber_strategies_.size(),
                           dual_size_h_,
                           reusable_device_scalar_value_1_.data(),
                           dual_to_transpose.data(),
                           dual_size_h_,
                           reusable_device_scalar_value_0_.data(),
                           nullptr,
                           climber_strategies_.size(),
                           dual_transposed.data(),
                           climber_strategies_.size()));

  // Copy that holds the tranpose to the original vector
  raft::copy(primal_to_transpose.data(),
             primal_transposed.data(),
             primal_size_h_ * climber_strategies_.size(),
             stream_view_);

  if (!is_dual_slack_empty) {
    raft::copy(dual_slack_to_transpose.data(),
               dual_slack_transposed.data(),
               primal_size_h_ * climber_strategies_.size(),
               stream_view_);
  }

  // Copy that holds the tranpose to the original vector
  raft::copy(dual_to_transpose.data(),
             dual_transposed.data(),
             dual_size_h_ * climber_strategies_.size(),
             stream_view_);
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::transpose_primal_dual_back_to_col(
  rmm::device_uvector<f_t>& primal_to_transpose,
  rmm::device_uvector<f_t>& dual_to_transpose,
  rmm::device_uvector<f_t>& dual_slack_to_transpose)
{
  bool is_dual_slack_empty = dual_slack_to_transpose.size() == 0;
  rmm::device_uvector<f_t> primal_transposed(primal_size_h_ * climber_strategies_.size(),
                                             stream_view_);
  rmm::device_uvector<f_t> dual_transposed(dual_size_h_ * climber_strategies_.size(), stream_view_);
  rmm::device_uvector<f_t> dual_slack_transposed(
    is_dual_slack_empty ? 0 : primal_size_h_ * climber_strategies_.size(), stream_view_);

  RAFT_CUBLAS_TRY(cublasSetStream(handle_ptr_->get_cublas_handle(), stream_view_));
  CUBLAS_CHECK(cublasDgeam(handle_ptr_->get_cublas_handle(),
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           primal_size_h_,
                           climber_strategies_.size(),
                           reusable_device_scalar_value_1_.data(),
                           primal_to_transpose.data(),
                           climber_strategies_.size(),
                           reusable_device_scalar_value_0_.data(),
                           nullptr,
                           primal_size_h_,
                           primal_transposed.data(),
                           primal_size_h_));

  if (!is_dual_slack_empty) {
    CUBLAS_CHECK(cublasDgeam(handle_ptr_->get_cublas_handle(),
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             primal_size_h_,
                             climber_strategies_.size(),
                             reusable_device_scalar_value_1_.data(),
                             dual_slack_to_transpose.data(),
                             climber_strategies_.size(),
                             reusable_device_scalar_value_0_.data(),
                             nullptr,
                             primal_size_h_,
                             dual_slack_transposed.data(),
                             primal_size_h_));
  }

  CUBLAS_CHECK(cublasDgeam(handle_ptr_->get_cublas_handle(),
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           dual_size_h_,
                           climber_strategies_.size(),
                           reusable_device_scalar_value_1_.data(),
                           dual_to_transpose.data(),
                           climber_strategies_.size(),
                           reusable_device_scalar_value_0_.data(),
                           nullptr,
                           dual_size_h_,
                           dual_transposed.data(),
                           dual_size_h_));

  // Copy that holds the tranpose to the original vector
  raft::copy(primal_to_transpose.data(),
             primal_transposed.data(),
             primal_size_h_ * climber_strategies_.size(),
             stream_view_);

  if (!is_dual_slack_empty) {
    raft::copy(dual_slack_to_transpose.data(),
               dual_slack_transposed.data(),
               primal_size_h_ * climber_strategies_.size(),
               stream_view_);
  }

  // Copy that holds the tranpose to the original vector
  raft::copy(dual_to_transpose.data(),
             dual_transposed.data(),
             dual_size_h_ * climber_strategies_.size(),
             stream_view_);
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> pdlp_solver_t<i_t, f_t>::run_solver(const timer_t& timer)
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

  // TODO handle that properly
  if (settings_.hyper_params.compute_initial_step_size_before_scaling &&
      !settings_.get_initial_step_size().has_value())
    compute_initial_step_size();
  if (settings_.hyper_params.compute_initial_primal_weight_before_scaling &&
      !settings_.get_initial_primal_weight().has_value())
    compute_initial_primal_weight();

  initial_scaling_strategy_.scale_problem();

  if (!settings_.hyper_params.compute_initial_step_size_before_scaling &&
      !settings_.get_initial_step_size().has_value())
    compute_initial_step_size();
  if (!settings_.hyper_params.compute_initial_primal_weight_before_scaling &&
      !settings_.get_initial_primal_weight().has_value())
    compute_initial_primal_weight();

#ifdef PDLP_DEBUG_MODE
  std::cout << "Initial Scaling done" << std::endl;
#endif

  // Needs to be performed here before the below line to make sure the initial primal_weight / step
  // size are used as previous point when potentially updating them in this next call
  if (settings_.get_initial_step_size().has_value() || initial_step_size_.has_value()) {
    if (initial_step_size_.has_value())
      thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                                 step_size_.begin(),
                                 step_size_.end(),
                                 initial_step_size_.value());
    else
      thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                                 step_size_.begin(),
                                 step_size_.end(),
                                 settings_.get_initial_step_size().value());
  }
  if (settings_.get_initial_primal_weight().has_value() || initial_primal_weight_.has_value()) {
    if (initial_primal_weight_.has_value()) {
      thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                                 primal_weight_.begin(),
                                 primal_weight_.end(),
                                 initial_primal_weight_.value());
      if (is_cupdlpx_restart<i_t, f_t>(settings_.hyper_params))
        thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                                   best_primal_weight_.begin(),
                                   best_primal_weight_.end(),
                                   initial_primal_weight_.value());
    } else {
      thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                                 primal_weight_.begin(),
                                 primal_weight_.end(),
                                 settings_.get_initial_primal_weight().value());
      if (is_cupdlpx_restart<i_t, f_t>(settings_.hyper_params))
        thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                                   best_primal_weight_.begin(),
                                   best_primal_weight_.end(),
                                   settings_.get_initial_primal_weight().value());
    }
  }
  if (initial_k_.has_value()) {
    pdhg_solver_.total_pdhg_iterations_ = initial_k_.value();
    pdhg_solver_.get_d_total_pdhg_iterations().set_value_async(initial_k_.value(), stream_view_);
  }

  // Only the primal_weight_ and step_size_ variables are initialized during the initial phase
  // The associated primal/dual step_size (computed using the two firstly mentionned) are not
  // initialized. This calls ensures the latter
  // In the event of a given primal and dual solutions and if the option is toggled, calling the
  // update primal_weight and step_size will also update the associated primal_step_size_,
  // dual_step_size_.
  // In summary: the below call is only mandatory at the beginning when
  // computing/setting the initial primal weight and step size and if they are not recomputed later.
  step_size_strategy_.get_primal_and_dual_stepsizes(primal_step_size_, dual_step_size_);

#ifdef CUPDLP_DEBUG_MODE
  if (initial_primal_.size() != 0 || initial_dual_.size() != 0) {
    std::cout << "Initial primal and dual solution before scaling" << std::endl;
    if (initial_primal_.size() != 0) { print("initial_primal_", initial_primal_); }
    if (initial_dual_.size() != 0) { print("initial_dual_", initial_dual_); }
  }
#endif

  // If there is an initial primal or dual we should update the restart info as if there was a step
  // that has happend
  if (initial_primal_.size() != 0 || initial_dual_.size() != 0) {
    update_primal_dual_solutions(
      (initial_primal_.size() != 0) ? std::make_optional(&initial_primal_) : std::nullopt,
      (initial_dual_.size() != 0) ? std::make_optional(&initial_dual_) : std::nullopt);
  }

#ifdef CUPDLP_DEBUG_MODE
  std::cout << "Solution before projection" << std::endl;
  print("pdhg_solver_.get_primal_solution()", pdhg_solver_.get_primal_solution());
  print("pdhg_solver_.get_dual_solution()", pdhg_solver_.get_dual_solution());
  print("pdhg_solver_.get_potential_next_primal_solution()",
        pdhg_solver_.get_potential_next_primal_solution());
  print("pdhg_solver_.get_potential_next_dual_solution()",
        pdhg_solver_.get_potential_next_dual_solution());
  print("restart_strategy_.last_restart_duality_gap_.primal_solution_",
        restart_strategy_.last_restart_duality_gap_.primal_solution_);
  print("restart_strategy_.last_restart_duality_gap_.dual_solution_",
        restart_strategy_.last_restart_duality_gap_.dual_solution_);
#endif

  // Project initial primal solution
  if (settings_.hyper_params.project_initial_primal) {
    using f_t2 = typename type_2<f_t>::type;
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(pdhg_solver_.get_primal_solution().data(),
                            problem_wrap_container(op_problem_scaled_.variable_bounds)),
      pdhg_solver_.get_primal_solution().data(),
      pdhg_solver_.get_primal_solution().size(),
      clamp<f_t, f_t2>(),
      stream_view_.value());

    pdhg_solver_.refine_initial_primal_projection();

    if (!settings_.hyper_params.never_restart_to_average) {
      cuopt_expects(!batch_mode_,
                    cuopt::error_type_t::ValidationError,
                    "Restart to average not supported in batch mode");
      cub::DeviceTransform::Transform(
        cuda::std::make_tuple(unscaled_primal_avg_solution_.data(),
                              op_problem_scaled_.variable_bounds.data()),
        unscaled_primal_avg_solution_.data(),
        primal_size_h_,
        clamp<f_t, f_t2>(),
        stream_view_.value());
    }
  }

#ifdef CUPDLP_DEBUG_MODE
  std::cout << "Solution after projection" << std::endl;
  print("pdhg_solver_.get_primal_solution()", pdhg_solver_.get_primal_solution());
  print("pdhg_solver_.get_dual_solution()", pdhg_solver_.get_dual_solution());
  print("pdhg_solver_.get_potential_next_primal_solution()",
        pdhg_solver_.get_potential_next_primal_solution());
  print("pdhg_solver_.get_potential_next_dual_solution()",
        pdhg_solver_.get_potential_next_dual_solution());
  print("restart_strategy_.last_restart_duality_gap_.primal_solution_",
        restart_strategy_.last_restart_duality_gap_.primal_solution_);
  print("restart_strategy_.last_restart_duality_gap_.dual_solution_",
        restart_strategy_.last_restart_duality_gap_.dual_solution_);
#endif

  // Need to to tranpose primal solution to row format as there might be initial values or clamping
  // Value may not be all 0
  if (batch_mode_) {
    rmm::device_uvector<f_t> dummy(0, stream_view_);
    transpose_primal_dual_to_row(
      pdhg_solver_.get_primal_solution(), pdhg_solver_.get_dual_solution(), dummy);
    if (settings_.hyper_params.use_reflected_primal_dual) {
      transpose_primal_dual_to_row(pdhg_solver_.get_potential_next_primal_solution(),
                                   pdhg_solver_.get_potential_next_dual_solution(),
                                   dummy);
      transpose_primal_dual_to_row(restart_strategy_.last_restart_duality_gap_.primal_solution_,
                                   restart_strategy_.last_restart_duality_gap_.dual_solution_,
                                   dummy);
    }
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
#ifdef CUPDLP_DEBUG_MODE
  raft::print_device_vector("Initial step_size", step_size_.data(), step_size_.size(), std::cout);
  raft::print_device_vector(
    "Initial primal_weight", primal_weight_.data(), primal_weight_.size(), std::cout);
#endif

  bool warm_start_was_given = settings_.get_pdlp_warm_start_data().is_populated();

  // In batch mode, before running the solver, we need to transpose the primal and dual solution to
  // row format
  if (batch_mode_)
    transpose_primal_dual_to_row(pdhg_solver_.get_potential_next_primal_solution(),
                                 pdhg_solver_.get_potential_next_dual_solution(),
                                 pdhg_solver_.get_dual_slack());

  if (!inside_mip_) {
    CUOPT_LOG_INFO(
      "   Iter    Primal Obj.      Dual Obj.    Gap        Primal Res.  Dual Res.   Time");
  }
  while (true) {
#ifdef CUPDLP_DEBUG_MODE
    printf("Step: %d\n", total_pdlp_iterations_);
#endif
    bool is_major_iteration =
      (((total_pdlp_iterations_) % settings_.hyper_params.major_iteration == 0) &&
       (total_pdlp_iterations_ > 0)) ||
      (total_pdlp_iterations_ <= settings_.hyper_params.min_iteration_restart);
    bool error_occured                      = (step_size_strategy_.get_valid_step_size() == -1);
    bool artificial_restart_check_main_loop = false;
    std::vector<int> has_restarted(climber_strategies_.size(), 0);
    bool is_conditional_major =
      (settings_.hyper_params.use_conditional_major)
        ? (total_pdlp_iterations_ % conditional_major<i_t>(total_pdlp_iterations_)) == 0
        : false;
    if (settings_.hyper_params.artificial_restart_in_main_loop)
      artificial_restart_check_main_loop =
        restart_strategy_.should_do_artificial_restart(total_pdlp_iterations_);
    if (is_major_iteration || artificial_restart_check_main_loop || error_occured ||
        is_conditional_major) {
      if (verbose) {
        std::cout << "-------------------------------" << std::endl;
        std::cout << internal_solver_iterations_ << std::endl;
        raft::print_device_vector("step_size", step_size_.data(), step_size_.size(), std::cout);
        raft::print_device_vector(
          "primal_weight", primal_weight_.data(), primal_weight_.size(), std::cout);
        raft::print_device_vector(
          "primal_step_size", primal_step_size_.data(), primal_step_size_.size(), std::cout);
        raft::print_device_vector(
          "dual_step_size", dual_step_size_.data(), dual_step_size_.size(), std::cout);
      }

      // If a warm start is given and it's the first step, the average solutions were already filled
      bool no_rescale_average = (internal_solver_iterations_ == 0 && warm_start_was_given) ||
                                settings_.hyper_params.never_restart_to_average;

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

#ifdef CUPDLP_DEBUG_MODE
      print("before scale slack", pdhg_solver_.get_dual_slack());
      print("before scale potential next primal",
            pdhg_solver_.get_potential_next_primal_solution());
      print("before scale potential next dual", pdhg_solver_.get_potential_next_dual_solution());
#endif

      // In case of batch mode, primal and dual matrices are in row format
      // We need to transpose them to column format before doing any checks
      if (batch_mode_) {
        rmm::device_uvector<f_t> dummy(0, stream_view_);
        transpose_primal_dual_back_to_col(pdhg_solver_.get_potential_next_primal_solution(),
                                          pdhg_solver_.get_potential_next_dual_solution(),
                                          pdhg_solver_.get_dual_slack());
        transpose_primal_dual_back_to_col(
          restart_strategy_.last_restart_duality_gap_.primal_solution_,
          restart_strategy_.last_restart_duality_gap_.dual_solution_,
          dummy);
        transpose_primal_dual_back_to_col(
          pdhg_solver_.get_primal_solution(), pdhg_solver_.get_dual_solution(), dummy);
      }

      // We go back to the unscaled problem here. It ensures that we do not terminate 'too early'
      // because of the error margin being evaluated on the scaled problem

      // Evaluation is done on the unscaled problem and solutions

      // If warm start data was given, the average solutions were also already scaled
      if (!no_rescale_average) {
        initial_scaling_strategy_.unscale_solutions(unscaled_primal_avg_solution_,
                                                    unscaled_dual_avg_solution_);
      }
      if (settings_.hyper_params.use_adaptive_step_size_strategy) {
        initial_scaling_strategy_.unscale_solutions(pdhg_solver_.get_primal_solution(),
                                                    pdhg_solver_.get_dual_solution());
      } else {
        initial_scaling_strategy_.unscale_solutions(
          pdhg_solver_.get_potential_next_primal_solution(),
          pdhg_solver_.get_potential_next_dual_solution(),
          pdhg_solver_.get_dual_slack());
      }

#ifdef CUPDLP_DEBUG_MODE
      print("after scale slack", pdhg_solver_.get_dual_slack());
      print("after scale potential next primal", pdhg_solver_.get_potential_next_primal_solution());
      print("after scale potential next dual", pdhg_solver_.get_potential_next_dual_solution());
#endif

#ifdef PDLP_DEBUG_MODE
      print("before check termination primal", pdhg_solver_.get_primal_solution());
      print("before check termination dual", pdhg_solver_.get_dual_solution());
      if (!settings_.hyper_params.never_restart_to_average) {
        print("before check termination average primal", unscaled_primal_avg_solution_);
        print("before check termination average dual", unscaled_dual_avg_solution_);
      }
#endif

      // Check for termination
      std::optional<optimization_problem_solution_t<i_t, f_t>> solution = check_termination(timer);

      if (solution.has_value()) { return std::move(solution.value()); }

      if (settings_.hyper_params.rescale_for_restart) {
        if (!settings_.hyper_params.never_restart_to_average) {
          initial_scaling_strategy_.scale_solutions(unscaled_primal_avg_solution_,
                                                    unscaled_dual_avg_solution_);
        }
        if (settings_.hyper_params.use_adaptive_step_size_strategy) {
          initial_scaling_strategy_.scale_solutions(pdhg_solver_.get_primal_solution(),
                                                    pdhg_solver_.get_dual_solution());
        } else {
          initial_scaling_strategy_.scale_solutions(
            pdhg_solver_.get_potential_next_primal_solution(),
            pdhg_solver_.get_potential_next_dual_solution(),
            pdhg_solver_.get_dual_slack());
        }
      }

      if (settings_.hyper_params.restart_strategy !=
            static_cast<int>(
              detail::pdlp_restart_strategy_t<i_t, f_t>::restart_strategy_t::NO_RESTART) &&
          (is_major_iteration || artificial_restart_check_main_loop)) {
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
          average_termination_strategy_.get_convergence_information(),  // Needed for KKT restart
          best_primal_weight_,  // Needed for cuPDLP+ restart
          has_restarted         // Needed for cuPDLP+ restart
        );
      }

      if (!settings_.hyper_params.rescale_for_restart) {
        // We don't need to rescale average because what matters is weighted_average_solution
        // getting the scaled accumulation
        // During the next iteration, unscaled_avg_solution will be overwritten again through
        // get_average_solutions
        if (settings_.hyper_params.use_adaptive_step_size_strategy) {
          initial_scaling_strategy_.scale_solutions(pdhg_solver_.get_primal_solution(),
                                                    pdhg_solver_.get_dual_solution());
        } else {
          initial_scaling_strategy_.scale_solutions(
            pdhg_solver_.get_potential_next_primal_solution(),
            pdhg_solver_.get_potential_next_dual_solution(),
            pdhg_solver_.get_dual_slack());
        }
      }

      // In batch mode, after having checked for termination and restart
      // We transpose back to row for the PDHG iterations
      if (batch_mode_) {
        transpose_primal_dual_to_row(pdhg_solver_.get_potential_next_primal_solution(),
                                     pdhg_solver_.get_potential_next_dual_solution(),
                                     pdhg_solver_.get_dual_slack());
        rmm::device_uvector<f_t> dummy(0, stream_view_);
        transpose_primal_dual_to_row(restart_strategy_.last_restart_duality_gap_.primal_solution_,
                                     restart_strategy_.last_restart_duality_gap_.dual_solution_,
                                     dummy);
        transpose_primal_dual_to_row(
          pdhg_solver_.get_primal_solution(), pdhg_solver_.get_dual_solution(), dummy);
      }
    }

#ifdef CUPDLP_DEBUG_MODE
    printf("Is Major %d\n",
           (total_pdlp_iterations_ + 1) % settings_.hyper_params.major_iteration == 0);
#endif
    take_step(total_pdlp_iterations_,
              (total_pdlp_iterations_ + 1) % settings_.hyper_params.major_iteration == 0);

    if (settings_.hyper_params.use_reflected_primal_dual) {
      if (settings_.hyper_params.use_fixed_point_error &&
          ((total_pdlp_iterations_ + 1) % settings_.hyper_params.major_iteration == 0 ||
           std::any_of(has_restarted.begin(), has_restarted.end(), [](int restarted) {
             return restarted == 1;
           }))) {
        // TODO later batch mode: remove this once if you have per climber restart
        if (std::any_of(has_restarted.begin(), has_restarted.end(), [](int restarted) {
              return restarted == 1;
            }))
          cuopt_assert(std::all_of(has_restarted.begin(),
                                   has_restarted.end(),
                                   [](int restarted) { return restarted == 1; }),
                       "If any, all should be true");
        if (batch_mode_) {
          rmm::device_uvector<f_t> dummy(0, stream_view_);
          transpose_primal_dual_back_to_col(
            pdhg_solver_.get_reflected_primal(),
            pdhg_solver_.get_reflected_dual(),
            pdhg_solver_.get_saddle_point_state().get_current_AtY());
          transpose_primal_dual_back_to_col(
            pdhg_solver_.get_primal_solution(), pdhg_solver_.get_dual_solution(), dummy);
        }
        compute_fixed_error(has_restarted);  // May set has_restarted to false
        if (batch_mode_) {
          rmm::device_uvector<f_t> dummy(0, stream_view_);
          transpose_primal_dual_to_row(pdhg_solver_.get_reflected_primal(),
                                       pdhg_solver_.get_reflected_dual(),
                                       pdhg_solver_.get_saddle_point_state().get_current_AtY());
          transpose_primal_dual_to_row(
            pdhg_solver_.get_primal_solution(), pdhg_solver_.get_dual_solution(), dummy);
        }
      }
      halpern_update();
    }

    ++total_pdlp_iterations_;
    ++internal_solver_iterations_;
    if (settings_.hyper_params.never_restart_to_average)
      restart_strategy_.increment_iteration_since_last_restart();
  }
  return optimization_problem_solution_t<i_t, f_t>{pdlp_termination_status_t::NumericalError,
                                                   stream_view_};
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::take_adaptive_step(i_t total_pdlp_iterations, bool is_major_iteration)
{
  // continue testing stepsize until we find a valid one or encounter a numerical error
  step_size_strategy_.set_valid_step_size(0);

  while (step_size_strategy_.get_valid_step_size() == 0) {
#ifdef PDLP_DEBUG_MODE
    std::cout << "PDHG Iteration:" << std::endl;
    print("primal_weight_", primal_weight_);
    print("step_size_", step_size_);
    print("primal_step_size_", primal_step_size_);
    print("dual_step_size_", dual_step_size_);
#endif
    pdhg_solver_.take_step(primal_step_size_,
                           dual_step_size_,
                           restart_strategy_.get_iterations_since_last_restart(),
                           restart_strategy_.get_last_restart_was_average(),
                           total_pdlp_iterations,
                           is_major_iteration);

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
void pdlp_solver_t<i_t, f_t>::take_constant_step(bool is_major_iteration)
{
  pdhg_solver_.take_step(
    primal_step_size_, dual_step_size_, 0, false, total_pdlp_iterations_, is_major_iteration);
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::halpern_update()
{
  raft::common::nvtx::range fun_scope("halpern_update");

  // TODO later batch mode: handle if element in the batch have different one if restart per climber
  const f_t weight =
    f_t(restart_strategy_.weighted_average_solution_.get_iterations_since_last_restart() + 1) /
    f_t(restart_strategy_.weighted_average_solution_.get_iterations_since_last_restart() + 2);

#ifdef CUPDLP_DEBUG_MODE
  printf("halper_update weight %lf\n", weight);
#endif

  // Update primal
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(pdhg_solver_.get_reflected_primal().data(),
                          pdhg_solver_.get_saddle_point_state().get_primal_solution().data(),
                          restart_strategy_.last_restart_duality_gap_.primal_solution_.data()),
    pdhg_solver_.get_saddle_point_state().get_primal_solution().data(),
    pdhg_solver_.get_saddle_point_state().get_primal_solution().size(),
    [weight, reflection_coefficient = settings_.hyper_params.reflection_coefficient] __device__(
      f_t reflected_primal, f_t current_primal, f_t initial_primal) {
      const f_t reflected = reflection_coefficient * reflected_primal +
                            (f_t(1.0) - reflection_coefficient) * current_primal;
      return weight * reflected + (f_t(1.0) - weight) * initial_primal;
    },
    stream_view_.value());

#ifdef CUPDLP_DEBUG_MODE
  print("pdhg_solver_.get_reflected_dual()", pdhg_solver_.get_reflected_dual());
  print("pdhg_solver_.get_saddle_point_state().get_dual_solution()",
        pdhg_solver_.get_saddle_point_state().get_dual_solution());
  print("restart_strategy_.last_restart_duality_gap_.dual_solution_",
        restart_strategy_.last_restart_duality_gap_.dual_solution_);

#endif

  // Update dual
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(pdhg_solver_.get_reflected_dual().data(),
                          pdhg_solver_.get_saddle_point_state().get_dual_solution().data(),
                          restart_strategy_.last_restart_duality_gap_.dual_solution_.data()),
    pdhg_solver_.get_saddle_point_state().get_dual_solution().data(),
    pdhg_solver_.get_saddle_point_state().get_dual_solution().size(),
    [weight, reflection_coefficient = settings_.hyper_params.reflection_coefficient] __device__(
      f_t reflected_dual, f_t current_dual, f_t initial_dual) {
      const f_t reflected = reflection_coefficient * reflected_dual +
                            (f_t(1.0) - reflection_coefficient) * current_dual;
      return weight * reflected + (f_t(1.0) - weight) * initial_dual;
    },
    stream_view_.value());

#ifdef CUPDLP_DEBUG_MODE
  print("halpen_update current primal",
        pdhg_solver_.get_saddle_point_state().get_primal_solution());
  print("halpen_update current dual", pdhg_solver_.get_saddle_point_state().get_dual_solution());
#endif
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::take_step([[maybe_unused]] i_t total_pdlp_iterations,
                                        [[maybe_unused]] bool is_major_iteration)
{
  if (settings_.hyper_params.use_adaptive_step_size_strategy) {
    cuopt_expects(!batch_mode_,
                  error_type_t::ValidationError,
                  "Batch mode not supported for use_adaptive_step_size_strategy mode");
    take_adaptive_step(total_pdlp_iterations, is_major_iteration);
  } else {
    cuopt_assert(total_pdlp_iterations == pdhg_solver_.get_total_pdhg_iterations(),
                 "In non adaptive step size mode, both pdlp and pdhg step should always be equal");
    take_constant_step(is_major_iteration);
  }

  // print("primal", pdhg_solver_.get_primal_solution());
  // print("dual", pdhg_solver_.get_dual_solution());
  // print("potential next primal", pdhg_solver_.get_potential_next_primal_solution());
  // print("potential next dual", pdhg_solver_.get_potential_next_dual_solution());
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::compute_initial_step_size()
{
  raft::common::nvtx::range fun_scope("compute_initial_step_size");

  if (!settings_.hyper_params.initial_step_size_max_singular_value) {
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

    // Sync since we are using local variable
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));
  } else {
    constexpr i_t max_iterations = 5000;
    constexpr f_t tolerance      = 1e-4;

    i_t m = op_problem_scaled_.n_constraints;
    i_t n = op_problem_scaled_.n_variables;

    std::vector<f_t> z(m);
    rmm::device_uvector<f_t> d_z(m, stream_view_);
    rmm::device_uvector<f_t> d_q(m, stream_view_);
    rmm::device_uvector<f_t> d_atq(n, stream_view_);

    std::mt19937 gen(1);
    std::normal_distribution<f_t> dist(f_t(0.0), f_t(1.0));

    for (int i = 0; i < m; ++i)
      z[i] = dist(gen);

    device_copy(d_z, z, stream_view_);

    rmm::device_scalar<f_t> norm_q(stream_view_);
    rmm::device_scalar<f_t> sigma_max_sq(stream_view_);
    rmm::device_scalar<f_t> residual_norm(stream_view_);
    rmm::device_scalar<f_t> reusable_device_scalar_value_1_(1, stream_view_);
    rmm::device_scalar<f_t> reusable_device_scalar_value_0_(0, stream_view_);

    cusparseDnVecDescr_t vecZ, vecQ, vecATQ;
    RAFT_CUSPARSE_TRY(
      raft::sparse::detail::cusparsecreatednvec(&vecZ, m, const_cast<f_t*>(d_z.data())));
    RAFT_CUSPARSE_TRY(
      raft::sparse::detail::cusparsecreatednvec(&vecQ, m, const_cast<f_t*>(d_q.data())));
    RAFT_CUSPARSE_TRY(
      raft::sparse::detail::cusparsecreatednvec(&vecATQ, n, const_cast<f_t*>(d_atq.data())));

    const auto& cusparse_view_ = pdhg_solver_.get_cusparse_view();

    [[maybe_unused]] int sing_iters = 0;
    for (int i = 0; i < max_iterations; ++i) {
      ++sing_iters;
      // d_q = d_z
      raft::copy(d_q.data(), d_z.data(), m, stream_view_);
      // norm_q = l2_norm(d_q)
      my_l2_norm<i_t, f_t>(d_q, norm_q, handle_ptr_);

      cuopt_assert(norm_q.value(stream_view_) != f_t(0), "norm q can't be 0");

      // d_q *= 1 / norm_q
      cub::DeviceTransform::Transform(
        d_q.data(),
        d_q.data(),
        d_q.size(),
        [norm_q = norm_q.data()] __device__(f_t d_q) { return d_q / *norm_q; },
        stream_view_.value());

      // A_t_q = A_t @ d_q
      RAFT_CUSPARSE_TRY(
        raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           reusable_device_scalar_value_1_.data(),
                                           cusparse_view_.A_T,
                                           vecQ,
                                           reusable_device_scalar_value_0_.data(),
                                           vecATQ,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           (f_t*)cusparse_view_.buffer_transpose.data(),
                                           stream_view_.value()));

      // z = A @ A_t_q
      RAFT_CUSPARSE_TRY(
        raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           reusable_device_scalar_value_1_.data(),  // 1
                                           cusparse_view_.A,
                                           vecATQ,
                                           reusable_device_scalar_value_0_.data(),  // 1
                                           vecZ,
                                           CUSPARSE_SPMV_CSR_ALG2,
                                           (f_t*)cusparse_view_.buffer_non_transpose.data(),
                                           stream_view_.value()));
      // sigma_max_sq = dot(q, z)
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
                                                      m,
                                                      d_q.data(),
                                                      primal_stride,
                                                      d_z.data(),
                                                      primal_stride,
                                                      sigma_max_sq.data(),
                                                      stream_view_.value()));

      cub::DeviceTransform::Transform(
        cuda::std::make_tuple(d_q.data(), d_z.data()),
        d_q.data(),
        d_q.size(),
        [sigma_max_sq = sigma_max_sq.data()] __device__(f_t d_q, f_t d_z) {
          return d_q * -(*sigma_max_sq) + d_z;
        },
        stream_view_.value());

      my_l2_norm<i_t, f_t>(d_q, residual_norm, handle_ptr_);

      if (residual_norm.value(stream_view_) < tolerance) break;
    }
#ifdef CUPDLP_DEBUG_MODE
    printf("iter_count %d\n", sing_iters);
#endif

    constexpr f_t scaling_factor = 0.998;
    const f_t step_size          = scaling_factor / std::sqrt(sigma_max_sq.value(stream_view_));
    thrust::uninitialized_fill(
      handle_ptr_->get_thrust_policy(), step_size_.begin(), step_size_.end(), step_size);

    // Sync since we are using local variable
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));
    RAFT_CUSPARSE_TRY(cusparseDestroyDnVec(vecZ));
    RAFT_CUSPARSE_TRY(cusparseDestroyDnVec(vecQ));
    RAFT_CUSPARSE_TRY(cusparseDestroyDnVec(vecATQ));
  }
}

template <typename i_t, typename f_t>
__global__ void compute_weights_initial_primal_weight_from_squared_norms(
  const f_t* b_vec_norm,
  const f_t* c_vec_norm,
  raft::device_span<f_t> primal_weight,
  raft::device_span<f_t> best_primal_weight,
  int batch_size,
  const pdlp_hyper_params::pdlp_hyper_params_t hyper_params)
{
  const int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= batch_size) { return; }
  f_t c_vec_norm_ = *c_vec_norm;
  f_t b_vec_norm_ = *b_vec_norm;

  if (b_vec_norm_ > f_t(0.0) && c_vec_norm_ > f_t(0.0)) {
#ifdef PDLP_DEBUG_MODE
    printf("b_vec_norm_ %lf c_vec_norm_ %lf primal_importance %lf\n",
           b_vec_norm_,
           c_vec_norm_,
           hyper_params.primal_importance);
#endif
    primal_weight[id]      = hyper_params.primal_importance * (c_vec_norm_ / b_vec_norm_);
    best_primal_weight[id] = primal_weight[id];
  } else {
    // It may be better to use this formula instead: primal_weight[id]      =
    // hyper_params.primal_importance * (c_vec_norm_ + (f_t(1.0))) / (b_vec_norm_ + f_t(1.0)); Not
    // doing so currently not to break backward compatibility and lack of examples to experiment
    // with it
    primal_weight[id]      = hyper_params.primal_importance;
    best_primal_weight[id] = primal_weight[id];
  }
}

template <typename i_t, typename f_t>
void pdlp_solver_t<i_t, f_t>::compute_initial_primal_weight()
{
  raft::common::nvtx::range fun_scope("compute_initial_primal_weight");

  // Here we use the combined bounds of the op_problem_scaled which may or may not be scaled yet
  // based on pdlp config
  // TODO later batch mode: handle per problem objective coefficients and rhs
  detail::combine_constraint_bounds<i_t, f_t>(op_problem_scaled_,
                                              op_problem_scaled_.combined_bounds);
  rmm::device_scalar<f_t> c_vec_norm{0.0, stream_view_};
  detail::my_l2_weighted_norm<i_t, f_t>(op_problem_scaled_.objective_coefficients,
                                        settings_.hyper_params.initial_primal_weight_c_scaling,
                                        c_vec_norm,
                                        stream_view_);

  rmm::device_scalar<f_t> b_vec_norm{0.0, stream_view_};
  if (settings_.hyper_params.initial_primal_weight_combined_bounds) {
    // => same as sqrt(dot(b,b))
    detail::my_l2_weighted_norm<i_t, f_t>(op_problem_scaled_.combined_bounds,
                                          settings_.hyper_params.initial_primal_weight_b_scaling,
                                          b_vec_norm,
                                          stream_view_);

  } else {
    if (settings_.hyper_params.bound_objective_rescaling) {
      constexpr f_t one = f_t(1.0);
      thrust::uninitialized_fill(
        handle_ptr_->get_thrust_policy(), primal_weight_.begin(), primal_weight_.end(), one);
      thrust::uninitialized_fill(handle_ptr_->get_thrust_policy(),
                                 best_primal_weight_.begin(),
                                 best_primal_weight_.end(),
                                 one);
      return;
    } else {
      cuopt_expects(settings_.hyper_params.initial_primal_weight_b_scaling == 1,
                    error_type_t::ValidationError,
                    "Passing a scaling is not supported for now");

      compute_sum_bounds(op_problem_scaled_.constraint_lower_bounds,
                         op_problem_scaled_.constraint_upper_bounds,
                         b_vec_norm,
                         stream_view_);
    }
  }

  const auto [grid_size, block_size] = kernel_config_from_batch_size(climber_strategies_.size());
  compute_weights_initial_primal_weight_from_squared_norms<i_t, f_t>
    <<<grid_size, block_size, 0, stream_view_>>>(b_vec_norm.data(),
                                                 c_vec_norm.data(),
                                                 make_span(primal_weight_),
                                                 make_span(best_primal_weight_),
                                                 climber_strategies_.size(),
                                                 settings_.hyper_params);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Sync since we are using local variable
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));
}

template <typename i_t, typename f_t>
f_t pdlp_solver_t<i_t, f_t>::get_primal_weight_h(i_t id) const
{
  cuopt_assert(id < primal_weight_.size(), "id is out of bounds");
  return primal_weight_.element(id, stream_view_);
}

template <typename i_t, typename f_t>
f_t pdlp_solver_t<i_t, f_t>::get_step_size_h(i_t id) const
{
  cuopt_assert(id < step_size_.size(), "id is out of bounds");
  return step_size_.element(id, stream_view_);
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
  const float* b_vec_norm,
  const float* c_vec_norm,
  raft::device_span<float> primal_weight,
  raft::device_span<float> best_primal_weight,
  int batch_size,
  const pdlp_hyper_params::pdlp_hyper_params_t hyper_params);
#endif

#if MIP_INSTANTIATE_DOUBLE
template class pdlp_solver_t<int, double>;

template __global__ void compute_weights_initial_primal_weight_from_squared_norms<double>(
  const double* b_vec_norm,
  const double* c_vec_norm,
  raft::device_span<double> primal_weight,
  raft::device_span<double> best_primal_weight,
  int batch_size,
  const pdlp_hyper_params::pdlp_hyper_params_t hyper_params);
#endif

}  // namespace cuopt::linear_programming::detail
