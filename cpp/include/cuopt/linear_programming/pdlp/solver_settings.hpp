/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
 * @brief Enum representing the different solver modes under which PDLP can
 * operate.
 *
 * Stable2: Best overall mode from experiments; balances speed and convergence
 * success. If you want to use the legacy version, use Stable1.
 * Methodical1: Usually leads to slower individual steps but fewer are needed to
 * converge. It uses from 1.3x up to 1.7x times more memory.
 * Fast1: Less convergence success but usually yields the highest speed
 *
 * @note Default mode is Stable2.
 */
// Forced to use an enum instead of an enum class for compatibility with the
// Cython layer
enum pdlp_solver_mode_t : int {
  Stable1     = CUOPT_PDLP_SOLVER_MODE_STABLE1,
  Stable2     = CUOPT_PDLP_SOLVER_MODE_STABLE2,
  Methodical1 = CUOPT_PDLP_SOLVER_MODE_METHODICAL1,
  Fast1       = CUOPT_PDLP_SOLVER_MODE_FAST1
};

/**
 * @brief Enum representing the different methods that can be used to solve the
 * linear programming problem.
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
   * @brief Set both absolute and relative tolerance on the primal feasibility,
   dual feasibility and gap.
   * Changing this value has a significant impact on accuracy and runtime.
   *
   * Optimality is computed as follows:
   * - dual_feasiblity < absolute_dual_tolerance + relative_dual_tolerance *
   norm_objective_coefficient (l2_norm(c))
   * - primal_feasiblity < absolute_primal_tolerance + relative_primal_tolerance
   * norm_constraint_bounds (l2_norm(b))
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
   * @brief Set an initial primal solution.
   *
   * @note Default value is all 0.
   *
   * @param[in] initial_primal_solution Device or host memory pointer to a
   * floating point array of size size. cuOpt copies this data. Copy happens on
   * the stream of the raft:handler passed to the problem.
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
   * @param[in] initial_dual_solution Device or host memory pointer to a
   * floating point array of size size. cuOpt copies this data. Copy happens on
   * the stream of the raft:handler passed to the problem.
   * @param size Size of the initial_dual_solution array.
   */
  void set_initial_dual_solution(const f_t* initial_dual_solution,
                                 i_t size,
                                 rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  /**
   * @brief Set the pdlp warm start data. This allows to restart PDLP with a
   * previous solution
   *
   * @note Interface for the C++ side. Only Stable2 and Fast1 are supported.
   *
   * @param pdlp_warm_start_data_view Pdlp warm start data from your solution
   * object to warm start from
   * @param var_mapping Variables indices to scatter to in case the new problem
   * has less variables
   * @param constraint_mapping Constraints indices to scatter to in case the new
   * problem has less constraints
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

  tolerances_t tolerances;
  bool detect_infeasibility{false};
  bool strict_infeasibility{false};
  i_t iteration_limit{std::numeric_limits<i_t>::max()};
  double time_limit{std::numeric_limits<double>::infinity()};
  pdlp_solver_mode_t pdlp_solver_mode{pdlp_solver_mode_t::Stable2};
  bool log_to_console{true};
  std::string log_file{""};
  std::string sol_file{""};
  std::string user_problem_file{""};
  bool per_constraint_residual{false};
  bool crossover{false};
  bool save_best_primal_so_far{false};
  bool first_primal_feasible{false};
  bool presolve{false};
  method_t method{method_t::Concurrent};
  // For concurrent termination
  std::atomic<i_t>* concurrent_halt;
  static constexpr f_t minimal_absolute_tolerance = 1.0e-12;

 private:
  /** Initial primal solution */
  std::shared_ptr<rmm::device_uvector<f_t>> initial_primal_solution_;
  /** Initial dual solution */
  std::shared_ptr<rmm::device_uvector<f_t>> initial_dual_solution_;
  // For the C++ interface
  pdlp_warm_start_data_t<i_t, f_t> pdlp_warm_start_data_;
  // For the Cython interface
  pdlp_warm_start_data_view_t<i_t, f_t> pdlp_warm_start_data_view_;

  friend class solver_settings_t<i_t, f_t>;
};

}  // namespace cuopt::linear_programming
