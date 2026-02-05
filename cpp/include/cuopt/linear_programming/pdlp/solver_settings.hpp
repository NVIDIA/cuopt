/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>
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
 * Stable3: Best overall mode from experiments; balances speed and convergence
 * success. If you want to use the legacy version, use Stable2.
 * Methodical1: Usually leads to slower individual steps but fewer are needed to
 * converge. It uses from 1.3x up to 1.7x times more memory.
 * Fast1: Less convergence success but usually yields the highest speed
 *
 * @note Default mode is Stable3.
 */
// Forced to use an enum instead of an enum class for compatibility with the
// Cython layer
enum pdlp_solver_mode_t : int {
  Stable1     = CUOPT_PDLP_SOLVER_MODE_STABLE1,
  Stable2     = CUOPT_PDLP_SOLVER_MODE_STABLE2,
  Methodical1 = CUOPT_PDLP_SOLVER_MODE_METHODICAL1,
  Fast1       = CUOPT_PDLP_SOLVER_MODE_FAST1,
  Stable3     = CUOPT_PDLP_SOLVER_MODE_STABLE3
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
  DualSimplex = CUOPT_METHOD_DUAL_SIMPLEX,
  Barrier     = CUOPT_METHOD_BARRIER
};

template <typename i_t, typename f_t>
class pdlp_solver_settings_t {
 public:
  pdlp_solver_settings_t() = default;

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

  /** TODO batch mode: tmp
   * @brief Set an initial step size.
   *
   * @param[in] initial_step_size Initial step size.
   */
  // TODO batch mode: tmp
  void set_initial_step_size(f_t initial_step_size);
  /**
   * @brief Set an initial primal weight.
   *
   * @param[in] initial_primal_weight Initial primal weight.
   */
  void set_initial_primal_weight(f_t initial_primal_weight);

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
  // TODO batch mode: tmp
  std::optional<f_t> get_initial_step_size() const;
  // TODO batch mode: tmp
  std::optional<f_t> get_initial_primal_weight() const;

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
    f_t primal_infeasible_tolerance = 1.0e-10;
    f_t dual_infeasible_tolerance   = 1.0e-10;
  };

  tolerances_t get_tolerances() const noexcept;
  template <typename U, typename V>
  friend class problem_checking_t;

  tolerances_t tolerances;
  bool detect_infeasibility{false};
  bool strict_infeasibility{false};
  i_t iteration_limit{std::numeric_limits<i_t>::max()};
  double time_limit{std::numeric_limits<double>::infinity()};
  pdlp_solver_mode_t pdlp_solver_mode{pdlp_solver_mode_t::Stable3};
  bool log_to_console{true};
  std::string log_file{""};
  std::string sol_file{""};
  std::string user_problem_file{""};
  bool per_constraint_residual{false};
  bool crossover{false};
  bool cudss_deterministic{false};
  i_t folding{-1};
  i_t augmented{-1};
  i_t dualize{-1};
  i_t ordering{-1};
  i_t barrier_dual_initial_point{-1};
  bool eliminate_dense_columns{true};
  bool save_best_primal_so_far{false};
  bool first_primal_feasible{false};
  presolver_t presolver{presolver_t::Default};
  bool dual_postsolve{true};
  int num_gpus{1};
  method_t method{method_t::Concurrent};
  bool inside_mip{false};
  // For concurrent termination
  std::atomic<int>* concurrent_halt{nullptr};
  static constexpr f_t minimal_absolute_tolerance = 1.0e-12;
  pdlp_hyper_params::pdlp_hyper_params_t hyper_params;
  // Holds the information of new variable lower and upper bounds for each climber in the format:
  // (variable index, new lower bound, new upper bound)
  // For each entry in the vector, a new version of the problem (climber) will be solved
  // concurrently i.e. if new_bounds.size() == 2, then 2 versions of the problem with updated bounds
  // will be solved concurrently
  std::vector<std::tuple<i_t, f_t, f_t>> new_bounds;

 private:
  /** Initial primal solution */
  std::shared_ptr<rmm::device_uvector<f_t>> initial_primal_solution_;
  /** Initial dual solution */
  std::shared_ptr<rmm::device_uvector<f_t>> initial_dual_solution_;
  /** Initial step size */
  // TODO batch mode: tmp
  std::optional<f_t> initial_step_size_;
  /** Initial primal weight */
  // TODO batch mode: tmp
  std::optional<f_t> initial_primal_weight_;
  // For the C++ interface
  pdlp_warm_start_data_t<i_t, f_t> pdlp_warm_start_data_;
  // For the Cython interface
  pdlp_warm_start_data_view_t<i_t, f_t> pdlp_warm_start_data_view_;

  friend class solver_settings_t<i_t, f_t>;
};

}  // namespace cuopt::linear_programming
