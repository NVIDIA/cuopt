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
  mip_solver_settings_t() = default;

  /**
   * @brief Set the callback for the user solution
   */
  void set_mip_callback(internals::base_solution_callback_t* callback = nullptr);

  /**
   * @brief Set an primal solution.
   *
   * @note Default value is all 0 or the LP optimal point.
   *
   * @param[in] initial_solution Device or host memory pointer to a floating
   * point array of size size. cuOpt copies this data. Copy happens on the
   * stream of the raft:handler passed to the problem.
   * @param size Size of the initial_solution array.
   */
  void set_initial_solution(const f_t* initial_solution,
                            i_t size,
                            rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  /**
   * @brief Get the initial solution.
   *
   * @return Initial solution as a rmm::device_uvector<f_t>
   */
  rmm::device_uvector<f_t>& get_initial_solution() const;

  /**
   * @brief Get the callback for the user solution
   *
   * @return callback pointer
   */
  const std::vector<internals::base_solution_callback_t*> get_mip_callbacks() const;

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
  tolerances_t tolerances;

  f_t time_limit       = std::numeric_limits<f_t>::infinity();
  bool heuristics_only = false;
  i_t num_cpu_threads  = -1;  // -1 means use default number of threads in branch and bound
  bool log_to_console  = true;
  std::string log_file;
  std::string sol_file;

  /** Initial primal solution */
  std::shared_ptr<rmm::device_uvector<f_t>> initial_solution_;
  bool mip_scaling = true;

 private:
  std::vector<internals::base_solution_callback_t*> mip_callbacks_;

  friend class solver_settings_t<i_t, f_t>;
};

}  // namespace cuopt::linear_programming
