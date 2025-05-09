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

#include <cuopt/linear_programming/utilities/internals.hpp>

#include <raft/core/device_span.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming {

// Forward declare solver_settings_t for friend class
template <typename i_t, typename f_t>
class solver_settings_t;

template <typename i_t, typename f_t>
class mip_solver_settings_t {
 public:
  mip_solver_settings_t()                                      = default;
  mip_solver_settings_t(const mip_solver_settings_t& settings) = default;

  /**
   * @brief Set the absolute tolerance.
   *
   * @note Default value is 1e-4.
   *
   * @param absolute_tolerance Absolute tolerance
   */
  void set_absolute_tolerance(f_t absolute_tolerance);

  /**
   * @brief Set the relative tolerance.
   *
   * @note Default value is 1e-6.
   *
   * @param relative_tolerance Relative tolerance
   */
  void set_relative_tolerance(f_t relative_tolerance);

  /**
   * @brief Set the integrality tolerance.
   *
   * @note Default value is 1e-5.
   *
   * @param integrality_tolerance Integrality tolerance
   */
  void set_integrality_tolerance(f_t integrality_tolerance);

  /**
   * @brief Set the allowed MIP gap absolute difference.
   *
   * @note Default value is 0.
   *
   * @note This is used to terminate the solve if the absolute difference between
   * the lower bound (or upper bound when maximizing) and the best feasible solution is less than
   * this value.
   *
   * @param absolute_mip_gap MIP gap absolute tolerance
   */
  void set_absolute_mip_gap(f_t absolute_mip_gap);

  /**
   * @brief Set the allowed MIP gap relative difference.
   *
   * @note Default value is 1e-3
   *
   * @note This is used to terminate the solve if the relative difference between
   * the objective value and the best feasible solution is less than this value.
   *
   * @param relative_mip_gap MIP gap relative tolerance
   */
  void set_relative_mip_gap(f_t relative_mip_gap);

  /**
   * @brief Set the time limit in seconds
   */
  void set_time_limit(double time_limit) noexcept;

  /**
   * @brief Set the log file name that the default logger writes to.
   *
   * @param log_file log file
   */
  void set_log_file(std::string log_file) noexcept;

  /**
   * @brief Set the generate logs flag
   *
   * @param log_to_console generate logs
   */
  void set_log_to_console(bool log_to_console) noexcept;

  /**
   * @brief Set the heuristics only flag
   */
  void set_heuristics_only(bool heuristics_only) noexcept;

  /**
   * @brief Set the number of CPU threads
   */
  void set_num_cpu_threads(i_t num_cpu_threads) noexcept;

  /**
   * @brief Set the incumbent callback
   */
  void set_incumbent_solution_callback(internals::lp_incumbent_sol_callback_t* callback = nullptr);

  /**
   * @brief Set whether or not problem scaling is used for MIP
   */
  void set_mip_scaling(bool mip_scaling);

  /**
   * @brief Set an primal solution.
   *
   * @note Default value is all 0 or the LP optimal point.
   *
   * @param[in] initial_solution Device or host memory pointer to a floating point array of
   * size size.
   * cuOpt copies this data. Copy happens on the stream of the raft:handler passed to the problem.
   * @param size Size of the initial_solution array.
   */
  void set_initial_solution(const f_t* initial_solution,
                            i_t size,
                            rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  /**
   * @brief Get the absolute tolerance.
   */
  f_t get_absolute_tolerance() const noexcept;

  /**
   * @brief Get the relative tolerance.
   */
  f_t get_relative_tolerance() const noexcept;

  /**
   * @brief Get the integrality tolerance.
   */
  f_t get_integrality_tolerance() const noexcept;

  /**
   * @brief Get the MIP gap absolute tolerance.
   */
  f_t get_absolute_mip_gap() const noexcept;

  /**
   * @brief Get the MIP gap relative tolerance.
   */
  f_t get_relative_mip_gap() const noexcept;

  /**
   * @brief Get whether or not problem scaling is used for MIP
   */
  bool get_mip_scaling() const noexcept;

  /**
   * @brief Get the time limit in seconds
   *
   * @return time limit
   */
  double get_time_limit() const noexcept;

  /**
   * @brief Get the heuristics only flag
   *
   * @return heuristics only flag
   */
  bool get_heuristics_only() const noexcept;

  /**
   * @brief Get the number of CPU threads
   *
   * @return number of CPU threads
   */
  i_t get_num_cpu_threads() const noexcept;

  /**
   * @brief Get the log file name.
   *
   * @return log file
   */
  std::string get_log_file() const noexcept;

  /**
   * @brief Get the generate logs flag
   *
   * @return generate logs flag
   */
  bool get_log_to_console() const noexcept;

  /**
   * @brief Get the initial solution.
   *
   * @return Initial solution as a rmm::device_uvector<f_t>
   */
  rmm::device_uvector<f_t>& get_initial_solution() const;

  /**
   * @brief Get incumbent solution callback
   *
   * @return callback pointer
   */
  internals::lp_incumbent_sol_callback_t* get_incumbent_solution_callback() const;

  bool has_initial_solution() const;

  struct tolerances_t {
    f_t absolute_tolerance    = 1.0e-4;
    f_t relative_tolerance    = 1.0e-6;
    f_t integrality_tolerance = 1.0e-5;
    f_t absolute_mip_gap      = 1.0e-10;
    f_t relative_mip_gap      = 1.0e-4;
  };

  /**
   * @brief Get the tolerance settings as a single structure.
   */
  tolerances_t get_tolerances() const noexcept;

  template <typename U, typename V>
  friend class problem_checking_t;

 private:
  tolerances_t tolerances;
  f_t time_limit_       = 0.;
  bool heuristics_only_ = false;
  i_t num_cpu_threads_  = -1;  // -1 means use default number of threads in branch and bound
  bool log_to_console_  = true;
  std::string log_file_;
  /** Initial primal solution */
  std::shared_ptr<rmm::device_uvector<f_t>> initial_solution_;
  internals::lp_incumbent_sol_callback_t* incumbent_sol_callback_ = nullptr;
  bool mip_scaling_                                               = true;

  friend class solver_settings_t<i_t, f_t>;
};

}  // namespace cuopt::linear_programming
