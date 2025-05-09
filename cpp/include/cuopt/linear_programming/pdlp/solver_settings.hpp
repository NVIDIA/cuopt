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

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>
#include <optional>
#include <raft/core/device_span.hpp>
#include <rmm/device_uvector.hpp>

#include <atomic>

namespace cuopt::linear_programming {

// Forward declare solver_settings_t for friend class
template <typename i_t, typename f_t>
class solver_settings_t;

/**
 * @brief Enum representing the different solver modes under which PDLP can operate.
 *
 * Stable2: Best overall mode from experiments; balances speed and convergence
 * success. If you want to use the legacy version, use Stable1.
 * Methodical1: Usually leads to slower individual steps but fewer are needed to
 * converge. It uses from 1.3x up to 1.7x times more memory.
 * Fast1: Less convergence success but usually yields the highest speed
 *
 * @note Default mode is Stable2.
 */
// Forced to use an enum instead of an enum class for compatibility with the Cython layer
enum pdlp_solver_mode_t : int {
  Stable1     = CUOPT_PDLP_SOLVER_MODE_STABLE1,
  Stable2     = CUOPT_PDLP_SOLVER_MODE_STABLE2,
  Methodical1 = CUOPT_PDLP_SOLVER_MODE_METHODICAL1,
  Fast1       = CUOPT_PDLP_SOLVER_MODE_FAST1
};

/**
 * @brief Enum representing the different methods that can be used to solve the linear programming
 * problem.
 *
 * Concurrent: Use both PDLP and DualSimplex in parallel.
 * PDLP: Use the PDLP method.
 * DualSimplex: Use the dual simplex method.
 *
 * @note Default method is Concurrent.
 */
enum method_t : int {
  Concurrent  = CUOPT_METHOD_CONCURRENT,
  PDLP        = CUOPT_METHOD_PDLP,
  DualSimplex = CUOPT_METHOD_DUAL_SIMPLEX
};

template <typename i_t, typename f_t>
class pdlp_solver_settings_t {
 public:
  pdlp_solver_settings_t() = default;

  // Copy constructor for when copying in the PDLP object
  pdlp_solver_settings_t(const pdlp_solver_settings_t& other, rmm::cuda_stream_view stream_view);
  /**
   * @brief Set both absolute and relative tolerance on the primal feasibility, dual feasibility
   and gap.
   * Changing this value has a significant impact on accuracy and runtime.
   *
   * Optimality is computed as follows:
   * - dual_feasiblity < absolute_dual_tolerance + relative_dual_tolerance *
   norm_objective_coefficient (l2_norm(c))
   * - primal_feasiblity < absolute_primal_tolerance + relative_primal_tolerance *
   norm_constraint_bounds (l2_norm(b))
   * - duality_gap < absolute_gap_tolerance + relative_gap_tolerance *
   (|primal_objective| + |dual_objective|)
   *
   * If all three conditions hold, optimality is reached.
   *
   * @note Default value is 1e-4.
   *
   * To set each absolute and relative tolerance, use the provided setters.
   *
   * @param eps_optimal Tolerance to optimality
   */
  void set_optimality_tolerance(f_t eps_optimal);

  /**
   * @brief Set the absolute dual tolerance.
   * For more details on tolerance to optimality, see `set_optimality_tolerance` method.
   *
   * @note Default value is 1e-4.
   *
   * @param absolute_dual_tolerance Absolute dual tolerance
   */
  void set_absolute_dual_tolerance(f_t absolute_dual_tolerance);
  /**
   * @brief Set the relative dual tolerance.
   * For more details on tolerance to optimality, see `set_optimality_tolerance` method.
   *
   * @note Default value is 1e-4.
   *
   * @param relative_dual_tolerance Relative dual tolerance
   */
  void set_relative_dual_tolerance(f_t relative_dual_tolerance);
  /**
   * @brief Set the absolute primal tolerance.
   * For more details on tolerance to optimality, see `set_optimality_tolerance` method.
   *
   * @note Default value is 1e-4.
   *
   * @param absolute_primal_tolerance Absolute primal tolerance
   */
  void set_absolute_primal_tolerance(f_t absolute_primal_tolerance);
  /**
   * @brief Set the relative primal tolerance.
   * For more details on tolerance to optimality, see `set_optimality_tolerance` method.
   *
   * @note Default value is 1e-4.
   *
   * @param relative_primal_tolerance Relative primal tolerance
   */
  void set_relative_primal_tolerance(f_t relative_primal_tolerance);
  /**
   * @brief Set the absolute gap tolerance.
   * For more details on tolerance to gap, see `set_optimality_tolerance` method.
   *
   * @note Default value is 1e-4.
   *
   * @param absolute_gap_tolerance Absolute gap tolerance
   */
  void set_absolute_gap_tolerance(f_t absolute_gap_tolerance);
  /**
   * @brief Set the relative gap tolerance.
   * For more details on tolerance to gap, see `set_optimality_tolerance` method.
   *
   * @note Default value is 1e-4.
   *
   * @param relative_gap_tolerance Relative gap tolerance
   */
  void set_relative_gap_tolerance(f_t relative_gap_tolerance);

  /**
   * @brief Solver will detect and leave if the problem is detected as infeasible.
   *
   * @note By default the solver will not detect infeasibility.
   *
   * Some problems detected as infeasible may converge under a different tolerance factor.
   *
   * Detecting infeasibility consumes both more runtime and memory. The added runtime is between 3%
   * and 7%, added memory is between 10% and 20%.
   *
   * @param detect True to detect infeasibility, false to ignore it.
   */
  void set_infeasibility_detection(bool detect);

  /**
   * @brief In strict infeasibility mode, if current or average solution is detected as infeasible.
   * It will be returned. Else both need to be detected as infeasible
   *
   * @note By default this is set to false.
   *
   * @param strict_infeasibility Strict infeasibility val.
   */
  void set_strict_infeasibility(bool strict_infeasibility);

  /**
   * @brief Set the primal infeasible tolerance.
   *
   * @note Default value is 1e-8.
   * Higher values will detect infeasibility quicker but may trigger false positive.
   *
   * @param primal_infeasible_tolerance Primal infeasible tolerance
   */
  void set_primal_infeasible_tolerance(f_t primal_infeasible_tolerance);

  /**
   * @brief Set the dual infeasible tolerance.
   * Higher values will detect infeasibility quicker but may trigger false positive.
   *
   * @note Default value is 1e-8.
   *
   * @param dual_infeasible_tolerance dual infeasible tolerance
   */
  void set_dual_infeasible_tolerance(f_t dual_infeasible_tolerance);

  /**
   * @brief Set the iteration limit after which the solver will stop and return the current
   * solution.
   *
   * @note By default there is no iteration limit.
   * For performance reasons, cuOpt's does not constantly checks for iteration limit, thus, the
   * solver might run a few extra iterations over the limit.
   * If set along time limit, the first limit reached will exit.
   *
   * @param iteration_limit Iteration limit to set
   */
  void set_iteration_limit(i_t iteration_limit);

  /**
   * @brief Set the time limit in seconds after which the solver will stop and return the
   * current solution.
   *
   * @note By default there is no time limit.
   * For performance reasons, cuOpt's does not constantly checks for time limit, thus, the solver
   * might run slightly over the limit.
   * If set along iteration limit, the first limit reached will exit.
   *
   * @param time_limit Time limit to set in seconds
   */
  void set_time_limit(double time_limit);

  /**
   * @brief Set the mode under which PDLP should operate. The mode will change the way the
   * PDLP internally optimizes the problem. The mode choice can drastically impact how fast a
   * specific problem will be solved. Users are encouraged to test different modes to see which one
   * fits the best their problem. By default, the solver uses pdlp_solver_mode_t.Stable2, the best
   * overall mode from our experiments. For now, only three modes are available : [Stable2,
   * Methodical1, Fast1]
   *
   * @note For now, we don't offer any mechanism to know upfront which solver mode will be the best
   * for one specific problem.
   * Check the pdlp_solver_mode_t enum for more details on the different modes.
   *
   * @param solver_mode Solver mode to set
   */
  void set_pdlp_solver_mode(pdlp_solver_mode_t solver_mode);

  /**
   * @brief Set the method to solve the linear programming problem.
   * Three methods are available:
   * - Concurrent: Use both PDLP and DualSimplex in parallel.
   * - PDLP: Use the PDLP method.
   * - DualSimplex: Use the dual simplex method.
   *
   * @note Default method is Concurrent.
   *
   * @param method Method to set
   */
  void set_method(method_t method);

  /**
   * @brief Set the log file name that the default logger writes to.
   *
   * @param log_file log file
   */
  void set_log_file(std::string log_file);

  /**
   * @brief Set the generate logs flag
   *
   * @param log_to_console generate logs
   */
  void set_log_to_console(bool log_to_console);

  /**
   * @brief PDLP will compute the primal & dual residual per constraint instead of globally
   *
   * @param per_constraint_residual Mode to set
   */
  void set_per_constraint_residual(bool per_constraint_residual);

  /**
   * @brief Set the crossover mode
   *
   * @param crossover True to enable crossover, false to disable
   */
  void set_crossover(bool crossover);

  /**
   * @brief PDLP will save the best primal solution so far
   * Will always prioritize a primal feasible to a non primal feasible
   * If a new primal feasible is found, the one with the best primal objective will be kept
   * If no primal feasible was found, the one with the lowest primal residual will be kept
   * If two have the same primal residual, the one with the best objective will be kept
   *
   * @param save_best_primal_so_far Mode to set
   */
  void set_save_best_primal_so_far(bool save_best_primal_so_far);

  /**
   * @brief PDLP will return on the first primal feasible solution found
   *
   * @param first_primal_feasible Mode to set
   */
  void set_first_primal_feasible_encountered(bool first_primal_feasible);

  /**
   * @brief Set an initial primal solution.
   *
   * @note Default value is all 0.
   *
   * @param[in] initial_primal_solution Device or host memory pointer to a floating point array of
   * size size.
   * cuOpt copies this data. Copy happens on the stream of the raft:handler passed to the problem.
   * @param size Size of the initial_primal_solution array.
   */
  void set_initial_primal_solution(const f_t* initial_primal_solution,
                                   i_t size,
                                   rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  /**
   * @brief Set an initial dual solution.
   *
   * @note Default value is all 0.
   *
   * @param[in] initial_dual_solution Device or host memory pointer to a floating point array of
   * size size.
   * cuOpt copies this data. Copy happens on the stream of the raft:handler passed to the problem.
   * @param size Size of the initial_dual_solution array.
   */
  void set_initial_dual_solution(const f_t* initial_dual_solution,
                                 i_t size,
                                 rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  /**
   * @brief Set the pdlp warm start data. This allows to restart PDLP with a previous solution
   *
   * @note Interface for the C++ side. Only Stable2 and Fast1 are supported.
   *
   * @param pdlp_warm_start_data_view Pdlp warm start data from your solution object to warm start
   * from
   * @param var_mapping Variables indices to scatter to in case the new problem has less variables
   * @param constraint_mapping Constraints indices to scatter to in case the new problem has less
   * constraints
   */
  void set_pdlp_warm_start_data(pdlp_warm_start_data_t<i_t, f_t>& pdlp_warm_start_data_view,
                                const rmm::device_uvector<i_t>& var_mapping =
                                  rmm::device_uvector<i_t>{0, rmm::cuda_stream_default},
                                const rmm::device_uvector<i_t>& constraint_mapping =
                                  rmm::device_uvector<i_t>{0, rmm::cuda_stream_default});

  // Same but for the Cython interface
  void set_pdlp_warm_start_data(const f_t* current_primal_solution,
                                const f_t* current_dual_solution,
                                const f_t* initial_primal_average,
                                const f_t* initial_dual_average,
                                const f_t* current_ATY,
                                const f_t* sum_primal_solutions,
                                const f_t* sum_dual_solutions,
                                const f_t* last_restart_duality_gap_primal_solution,
                                const f_t* last_restart_duality_gap_dual_solution,
                                i_t primal_size,
                                i_t dual_size,
                                f_t initial_primal_weight_,
                                f_t initial_step_size_,
                                i_t total_pdlp_iterations_,
                                i_t total_pdhg_iterations_,
                                f_t last_candidate_kkt_score_,
                                f_t last_restart_kkt_score_,
                                f_t sum_solution_weight_,
                                i_t iterations_since_last_restart_);

  /**
   * @brief Set the pointer to the concurrent halt atomic
   *
   * @param concurrent_halt Concurrent halt
   */
  void set_concurrent_halt(std::atomic<i_t>* concurrent_halt) noexcept;

  /**
   * @brief Get the absolute dual tolerance.
   * For more details on tolerance to optimality, see `set_optimality_tolerance` method.
   */
  f_t get_absolute_dual_tolerance() const noexcept;
  /**
   * @brief Get the relative dual tolerance.
   * For more details on tolerance to optimality, see `set_optimality_tolerance` method.
   */
  f_t get_relative_dual_tolerance() const noexcept;
  /**
   * @brief Get the absolute primal tolerance.
   * For more details on tolerance to optimality, see `set_optimality_tolerance` method.
   */
  f_t get_absolute_primal_tolerance() const noexcept;
  /**
   * @brief Get the relative primal tolerance.
   * For more details on tolerance to optimality, see `set_optimality_tolerance` method.
   */
  f_t get_relative_primal_tolerance() const noexcept;
  /**
   * @brief Get the absolute gap tolerance.
   * For more details on tolerance to optimality, see `set_optimality_tolerance` method.
   */
  f_t get_absolute_gap_tolerance() const noexcept;
  /**
   * @brief Get the relative gap tolerance.
   * For more details on tolerance to gap, see `set_optimality_tolerance` method.
   */
  f_t get_relative_gap_tolerance() const noexcept;

  /**
   * @brief Get the status of detecting infeasibility
   *
   * @return Status of detecting infeasibility
   */
  bool get_infeasibility_detection() const noexcept;

  /**
   * @brief Get strict_inefasibility bool val
   *
   * @return strict infeasibility
   */
  bool get_strict_infeasibility() const noexcept;

  /**
   * @brief Get the primal infeasible tolerance.
   */
  f_t get_primal_infeasible_tolerance() const noexcept;

  /**
   * @brief Get the dual infeasible tolerance.
   */
  f_t get_dual_infeasible_tolerance() const noexcept;

  /**
   * @brief Get the iteration limit
   *
   * @return iteration limit
   */
  i_t get_iteration_limit() const noexcept;

  /**
   * @brief Get the time limit in ms
   *
   * @return time limit
   */
  std::optional<double> get_time_limit() const noexcept;

  /**
   * @brief Get the solver mode
   * For more details on solver mode, see `set_pdlp_solver_mode` method.
   *
   * @return solver mode
   */
  pdlp_solver_mode_t get_pdlp_solver_mode() const noexcept;

  /**
   * @brief Get the method
   *
   * @return method
   */
  method_t get_method() const noexcept;

  /**
   * @brief Get the log file name
   *
   * @return log file
   */
  std::string get_log_file() const noexcept;

  /**
   * @brief Get the write logs flag
   *
   * @return write logs
   */
  bool get_log_to_console() const noexcept;

  /**
   * @brief Get the crossover mode
   *
   * @return crossover mode
   */
  bool get_crossover() const noexcept;

  /**
   * @brief True if in per row residual mode, false if global
   *
   */
  bool get_per_constraint_residual() const noexcept;

  /**
   * @brief True if is saving the best primal so far
   *
   */
  bool get_save_best_primal_so_far() const noexcept;

  /**
   * @brief True if first primal feasible is enabled
   *
   */
  bool get_first_primal_feasible_encountered() const noexcept;

  /**
   * @brief Get the concurrent halt
   *
   * @return concurrent halt
   */
  std::atomic<i_t>* get_concurrent_halt() const noexcept;

  /**
   * @brief Get the pdlp warm start data
   *
   * @return pdlp warm start data
   */
  const pdlp_warm_start_data_t<i_t, f_t>& get_pdlp_warm_start_data() const noexcept;
  pdlp_warm_start_data_t<i_t, f_t>& get_pdlp_warm_start_data();
  const pdlp_warm_start_data_view_t<i_t, f_t>& get_pdlp_warm_start_data_view() const noexcept;

  const rmm::device_uvector<f_t>& get_initial_primal_solution() const;
  const rmm::device_uvector<f_t>& get_initial_dual_solution() const;

  bool has_initial_primal_solution() const;
  bool has_initial_dual_solution() const;

  struct tolerances_t {
    f_t absolute_dual_tolerance     = 1.0e-4;
    f_t relative_dual_tolerance     = 1.0e-4;
    f_t absolute_primal_tolerance   = 1.0e-4;
    f_t relative_primal_tolerance   = 1.0e-4;
    f_t absolute_gap_tolerance      = 1.0e-4;
    f_t relative_gap_tolerance      = 1.0e-4;
    f_t primal_infeasible_tolerance = 1.0e-8;
    f_t dual_infeasible_tolerance   = 1.0e-8;
  };

  tolerances_t get_tolerances() const noexcept;
  template <typename U, typename V>
  friend class problem_checking_t;

 private:
  tolerances_t tolerances;
  bool detect_infeasibility_{false};
  bool strict_infeasibility_{false};
  i_t iteration_limit_ = std::numeric_limits<i_t>::max();
  double time_limit_{std::numeric_limits<double>::infinity()};
  pdlp_solver_mode_t solver_mode_{pdlp_solver_mode_t::Stable2};
  bool log_to_console_{true};
  std::string log_file_{""};
  bool per_constraint_residual_{false};
  bool crossover_{false};
  bool save_best_primal_so_far_{false};
  bool first_primal_feasible_{false};
  method_t method_{method_t::Concurrent};
  /** Initial primal solution */
  std::shared_ptr<rmm::device_uvector<f_t>> initial_primal_solution_;
  /** Initial dual solution */
  std::shared_ptr<rmm::device_uvector<f_t>> initial_dual_solution_;
  // For the C++ interface
  pdlp_warm_start_data_t<i_t, f_t> pdlp_warm_start_data_;
  // For the Cython interface
  pdlp_warm_start_data_view_t<i_t, f_t> pdlp_warm_start_data_view_;
  // For concurrent termination
  std::atomic<i_t>* concurrent_halt_;
  static constexpr f_t minimal_absolute_tolerance = 1.0e-12;

  friend class solver_settings_t<i_t, f_t>;
};

}  // namespace cuopt::linear_programming
