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

#pragma once

#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>

#include <linear_programming/cusparse_view.hpp>
#include <linear_programming/initial_scaling_strategy/initial_scaling.cuh>
#include <linear_programming/pdhg.hpp>
#include <linear_programming/restart_strategy/pdlp_restart_strategy.cuh>
#include <linear_programming/step_size_strategy/adaptive_step_size_strategy.hpp>
#include <linear_programming/termination_strategy/termination_strategy.hpp>

#include <mip/problem/problem.cuh>

#include <utilities/timer.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <optional>
#include "linear_programming/termination_strategy/convergence_information.hpp"

namespace cuopt::linear_programming::detail {
void set_pdlp_hyper_parameters(rmm::cuda_stream_view stream_view);
/**
 * @brief Solver for an optimization problem (Currently only linear program) to be solved,
 * pdlp_parameters and pdlp_internal_state
 *
 * @tparam i_t  Data type of indexes
 * @tparam f_t  Data type of the variables and their weights in the equations
 *
 */
template <typename i_t, typename f_t>
class pdlp_solver_t {
 public:
  static_assert(std::is_integral<i_t>::value,
                "'pdlp_solver_t' accepts only integer types for indexes");
  static_assert(std::is_floating_point<f_t>::value,
                "'pdlp_solver_t' accepts only floating point types for weights");

  /**
   * @brief An implementation of PDLP - First order solver for linear (and quadratic programming)
   *
   * For full description of algorithm, see https://arxiv.org/abs/2106.04756
   *
   * @param[in] op_problem An problem_t<i_t, f_t> object with a
   * representation of a linear program
   */
  pdlp_solver_t(
    problem_t<i_t, f_t>& op_problem,
    pdlp_solver_settings_t<i_t, f_t> const& settings = pdlp_solver_settings_t<i_t, f_t>{},
    bool is_batch_mode                               = false);

  optimization_problem_solution_t<i_t, f_t> run_solver(const timer_t& timer);

  f_t get_primal_weight_h() const;
  f_t get_step_size_h() const;
  i_t get_total_pdhg_iterations() const;
  f_t get_relative_dual_tolerance_factor() const;
  f_t get_relative_primal_tolerance_factor() const;
  detail::pdlp_termination_strategy_t<i_t, f_t>& get_current_termination_strategy();

  void set_problem_ptr(problem_t<i_t, f_t>* problem_ptr_);

  // Interface to let MIP set an initial solution
  // Users will keep on using the optimization_problem to provide an initial solution
  void set_initial_primal_solution(const rmm::device_uvector<f_t>& initial_primal_solution);
  void set_initial_dual_solution(const rmm::device_uvector<f_t>& initial_dual_solution);
  void set_initial_primal_weight(f_t initial_primal_weight);
  void set_initial_step_size(f_t initial_primal_weight);
  void set_initial_k(i_t initial_k);
  void set_relative_dual_tolerance_factor(f_t dual_tolerance_factor);
  void set_relative_primal_tolerance_factor(f_t primal_tolerance_factor);

  using primal_quality_adapter_t =
    typename convergence_information_t<i_t, f_t>::primal_quality_adapter_t;

  const primal_quality_adapter_t& get_best_quality(const primal_quality_adapter_t& current,
                                                   const primal_quality_adapter_t& other);

  void set_inside_mip(bool inside_mip);

  void compute_initial_step_size();
  void compute_initial_primal_weight();

 private:
  void print_termination_criteria(const timer_t& timer, bool is_average = false);
  void print_final_termination_criteria(
    const timer_t& timer,
    const convergence_information_t<i_t, f_t>& convergence_information,
    const pdlp_termination_status_t& termination_status,
    bool is_average = false);
  std::optional<optimization_problem_solution_t<i_t, f_t>> check_termination(const timer_t& timer);
  std::optional<optimization_problem_solution_t<i_t, f_t>> check_limits(const timer_t& timer);
  void record_best_primal_so_far(const detail::pdlp_termination_strategy_t<i_t, f_t>& current,
                                 const detail::pdlp_termination_strategy_t<i_t, f_t>& average,
                                 const pdlp_termination_status_t& termination_current,
                                 const pdlp_termination_status_t& termination_average);

  void take_step([[maybe_unused]] i_t total_pdlp_iterations,
                 [[maybe_unused]] bool is_major_iteration);
  void take_adaptive_step(i_t total_pdlp_iterations, bool is_major_iteration);
  void take_constant_step(bool is_major_iteration);

  /**
   * @brief Update current primal & dual solution by setting new solutions and triggering a
   * recomputation of the primal weight and step size
   *
   * @param primal Initial primal solution
   * @param dual Initial dual solution
   */
  void update_primal_dual_solutions(std::optional<const rmm::device_uvector<f_t>*> primal,
                                    std::optional<const rmm::device_uvector<f_t>*> dual);

  raft::handle_t const* handle_ptr_;
  rmm::cuda_stream_view stream_view_;

  problem_t<i_t, f_t>* problem_ptr;
  // Combined bounds in op_problem_scaled_ will only be scaled if
  // compute_initial_primal_weight_before_scaling is false because of compute_initial_primal_weight
  problem_t<i_t, f_t> op_problem_scaled_;

  rmm::device_uvector<f_t> unscaled_primal_avg_solution_;
  rmm::device_uvector<f_t> unscaled_dual_avg_solution_;

  i_t primal_size_h_;
  i_t dual_size_h_;

  rmm::device_scalar<f_t> primal_step_size_;
  rmm::device_scalar<f_t> dual_step_size_;

  /**
  The primal and dual step sizes are parameterized as:
    tau = primal_step_size = step_size / primal_weight
    sigma = dual_step_size = step_size * primal_weight.
  The primal_weight factor is named as such because this parameterization is
  equivalent to defining the Bregman divergences as:
  D_x(x, x bar) = 0.5 * primal_weight ||x - x bar||_2^2, and
  D_y(y, y bar) = 0.5 / primal_weight ||y - y bar||_2^2.

  The parameter primal_weight is adjusted smoothly at each restart; to balance the
  primal and dual distances traveled since the last restart.
  */
  rmm::device_scalar<f_t> primal_weight_;
  rmm::device_scalar<f_t> best_primal_weight_;
  rmm::device_scalar<f_t> step_size_;

  // Step size strategy
  detail::adaptive_step_size_strategy_t<i_t, f_t> step_size_strategy_;

 public:
  // Inner solver
  detail::pdhg_solver_t<i_t, f_t> pdhg_solver_;
  void halpern_update();

 private:
  // Intentionnaly take a copy to avoid an unintentional modification in the calling context
  const pdlp_solver_settings_t<i_t, f_t> settings_;

  void compute_fixed_error(bool& has_restarted);

  pdlp_warm_start_data_t<i_t, f_t> get_filled_warmed_start_data();

  // Initial scaling strategy
  detail::pdlp_initial_scaling_strategy_t<i_t, f_t> initial_scaling_strategy_;

  // For the average evaluation
  detail::cusparse_view_t<i_t, f_t> average_op_problem_evaluation_cusparse_view_;
  detail::cusparse_view_t<i_t, f_t> current_op_problem_evaluation_cusparse_view_;

  // Restart strategy
  detail::pdlp_restart_strategy_t<i_t, f_t> restart_strategy_;
  // Termination strategy
  detail::pdlp_termination_strategy_t<i_t, f_t> average_termination_strategy_;
  detail::pdlp_termination_strategy_t<i_t, f_t> current_termination_strategy_;

  /* Two counters are necessary because of the PDLP warm start data
   *  total_pdlp_iterations_: total, counting potential previous PDLP iterations
   *    Useful for:
   *      - Not triggerring on the min iteration restart
   *      - Not triggering a check_limits without optimality check
   *      - Correct restart information
   * internal_solver_iterations_: only current PDLP object iterations
   *    Useful for:
   *      - Returning the correct amount of iterations in the solution object
   *      - Correct iteration limit trigger
   */
  i_t total_pdlp_iterations_{0};
  i_t internal_solver_iterations_{0};

  // Initial solution
  rmm::device_uvector<f_t> initial_primal_;
  rmm::device_uvector<f_t> initial_dual_;
  // Used in the context of MIP to relaunch PDLP from a pseudo previous state
  std::optional<f_t> initial_primal_weight_;
  std::optional<f_t> initial_step_size_;
  std::optional<i_t> initial_k_;

  // Only used if save_best_primal_so_far is toggeled
  optimization_problem_solution_t<i_t, f_t> best_primal_solution_so_far;
  primal_quality_adapter_t best_primal_quality_so_far_;
  // Flag to indicate if solver is being called from MIP. No logging is done in this case.
  bool inside_mip_{false};
};

}  // namespace cuopt::linear_programming::detail
