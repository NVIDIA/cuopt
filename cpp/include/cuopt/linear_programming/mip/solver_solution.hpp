/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/constants.h>
#include <cuopt/error.hpp>
#include <cuopt/linear_programming/mip/solver_stats.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/core/handle.hpp>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace cuopt::linear_programming {

enum class mip_termination_status_t : int8_t {
  NoTermination = CUOPT_TERIMINATION_STATUS_NO_TERMINATION,
  Optimal       = CUOPT_TERIMINATION_STATUS_OPTIMAL,
  FeasibleFound = CUOPT_TERIMINATION_STATUS_FEASIBLE_FOUND,
  Infeasible    = CUOPT_TERIMINATION_STATUS_INFEASIBLE,
  Unbounded     = CUOPT_TERIMINATION_STATUS_UNBOUNDED,
  TimeLimit     = CUOPT_TERIMINATION_STATUS_TIME_LIMIT,
};

template <typename i_t, typename f_t>
class mip_solution_t : public base_solution_t {
 public:
  mip_solution_t(rmm::device_uvector<f_t> solution,
                 std::vector<std::string> var_names,
                 f_t objective,
                 f_t mip_gap,
                 mip_termination_status_t termination_status,
                 f_t max_constraint_violation,
                 f_t max_int_violation,
                 f_t max_variable_bound_violation,
                 solver_stats_t<i_t, f_t> stats,
                 std::vector<rmm::device_uvector<f_t>> solution_pool = {});

  mip_solution_t(mip_termination_status_t termination_status,
                 solver_stats_t<i_t, f_t> stats,
                 rmm::cuda_stream_view stream_view);
  mip_solution_t(const cuopt::logic_error& error_status, rmm::cuda_stream_view stream_view);

  // CPU-only constructors for remote solve
  mip_solution_t(std::vector<f_t> solution,
                 std::vector<std::string> var_names,
                 f_t objective,
                 f_t mip_gap,
                 mip_termination_status_t termination_status,
                 f_t max_constraint_violation,
                 f_t max_int_violation,
                 f_t max_variable_bound_violation,
                 solver_stats_t<i_t, f_t> stats);

  mip_solution_t(mip_termination_status_t termination_status, solver_stats_t<i_t, f_t> stats);
  mip_solution_t(const cuopt::logic_error& error_status);

  bool is_mip() const override { return true; }

  /**
   * @brief Check if solution data is stored in device (GPU) memory
   * @return true if data is in GPU memory, false if in CPU memory
   */
  bool is_device_memory() const;

  const rmm::device_uvector<f_t>& get_solution() const;
  rmm::device_uvector<f_t>& get_solution();

  /**
   * @brief Returns the solution in host (CPU) memory.
   * Only valid when is_device_memory() returns false.
   */
  std::vector<f_t>& get_solution_host();
  const std::vector<f_t>& get_solution_host() const;

  f_t get_objective_value() const;
  f_t get_mip_gap() const;
  f_t get_solution_bound() const;
  double get_total_solve_time() const;
  double get_presolve_time() const;
  mip_termination_status_t get_termination_status() const;
  static std::string get_termination_status_string(mip_termination_status_t termination_status);
  std::string get_termination_status_string() const;
  const cuopt::logic_error& get_error_status() const;
  f_t get_max_constraint_violation() const;
  f_t get_max_int_violation() const;
  f_t get_max_variable_bound_violation() const;
  solver_stats_t<i_t, f_t> get_stats() const;
  i_t get_num_nodes() const;
  i_t get_num_simplex_iterations() const;
  const std::vector<std::string>& get_variable_names() const;
  const std::vector<rmm::device_uvector<f_t>>& get_solution_pool() const;
  void write_to_sol_file(std::string_view filename, rmm::cuda_stream_view stream_view) const;
  void log_summary() const;

  //============================================================================
  // Setters for remote solve deserialization
  //============================================================================

  /**
   * @brief Set the solution in host memory
   * @param solution The solution vector
   */
  void set_solution_host(std::vector<f_t> solution);

  /**
   * @brief Set the objective value
   */
  void set_objective(f_t value);

  /**
   * @brief Set the MIP gap
   */
  void set_mip_gap(f_t value);

  /**
   * @brief Set the solution bound
   */
  void set_solution_bound(f_t value);

  /**
   * @brief Set total solve time
   */
  void set_total_solve_time(double value);

  /**
   * @brief Set presolve time
   */
  void set_presolve_time(double value);

  /**
   * @brief Set max constraint violation
   */
  void set_max_constraint_violation(f_t value);

  /**
   * @brief Set max integer violation
   */
  void set_max_int_violation(f_t value);

  /**
   * @brief Set max variable bound violation
   */
  void set_max_variable_bound_violation(f_t value);

  /**
   * @brief Set number of nodes
   */
  void set_nodes(i_t value);

  /**
   * @brief Set number of simplex iterations
   */
  void set_simplex_iterations(i_t value);

  /**
   * @brief Get error string
   */
  std::string get_error_string() const;

  /**
   * @brief Get number of nodes
   */
  i_t get_nodes() const;

  /**
   * @brief Get number of simplex iterations
   */
  i_t get_simplex_iterations() const;

  /**
   * @brief Copy solution data from GPU to CPU memory.
   *
   * After calling this method, is_device_memory() will return false and
   * the solution can be accessed via get_solution_host().
   * This is useful for remote solve scenarios where serialization requires
   * CPU-accessible data.
   *
   * If the solution is already in CPU memory, this is a no-op.
   *
   * @param stream_view The CUDA stream to use for the copy
   */
  void to_host(rmm::cuda_stream_view stream_view);

 private:
  // GPU (device) storage - populated for local GPU solves
  std::unique_ptr<rmm::device_uvector<f_t>> solution_;

  // CPU (host) storage - populated for remote solves
  std::unique_ptr<std::vector<f_t>> solution_host_;

  // Flag indicating where solution data is stored
  bool is_device_memory_ = true;

  std::vector<std::string> var_names_;
  f_t objective_;
  f_t mip_gap_;
  mip_termination_status_t termination_status_;
  cuopt::logic_error error_status_;
  f_t max_constraint_violation_;
  f_t max_int_violation_;
  f_t max_variable_bound_violation_;
  solver_stats_t<i_t, f_t> stats_;
  std::vector<rmm::device_uvector<f_t>> solution_pool_;
};

}  // namespace cuopt::linear_programming
