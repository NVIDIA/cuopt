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

#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>

#include <raft/core/device_span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>

#include <optional>
#include <vector>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
class solver_settings_t {
 public:
  solver_settings_t();

  // Delete copy constructor
  solver_settings_t(const solver_settings_t& settings) = delete;
  // Delete assignment operator
  solver_settings_t& operator=(const solver_settings_t& settings) = delete;
  // Delete move constructor
  solver_settings_t(solver_settings_t&& settings) = delete;
  // Delete move assignment operator
  solver_settings_t& operator=(solver_settings_t&& settings) = delete;

  void set_parameter_from_string(const std::string& name, const std::string& value);

  template <typename T>
  void set_parameter(const std::string& name, T value);

  template <typename T>
  T get_parameter(const std::string& name) const;

  std::string get_parameter_as_string(const std::string& name) const;

  void set_initial_pdlp_primal_solution(const f_t* initial_primal_solution,
                                        i_t size,
                                        rmm::cuda_stream_view stream = rmm::cuda_stream_default);
  void set_initial_pdlp_dual_solution(const f_t* initial_dual_solution,
                                      i_t size,
                                      rmm::cuda_stream_view stream = rmm::cuda_stream_default);
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

  const rmm::device_uvector<f_t>& get_initial_pdlp_primal_solution() const;
  const rmm::device_uvector<f_t>& get_initial_pdlp_dual_solution() const;

  // MIP Settings
  void set_initial_mip_solution(const f_t* initial_solution, i_t size);
  void set_mip_callback(internals::base_solution_callback_t* callback = nullptr);

  const rmm::device_uvector<f_t>& get_initial_mip_solution() const;
  const pdlp_warm_start_data_view_t<i_t, f_t>& get_pdlp_warm_start_data_view() const noexcept;
  const std::vector<internals::base_solution_callback_t*> get_mip_callbacks() const;

  pdlp_solver_settings_t<i_t, f_t>& get_pdlp_settings();
  mip_solver_settings_t<i_t, f_t>& get_mip_settings();

  const std::vector<parameter_info_t<f_t>>& get_float_parameters() const;
  const std::vector<parameter_info_t<i_t>>& get_int_parameters() const;
  const std::vector<parameter_info_t<bool>>& get_bool_parameters() const;
  const std::vector<parameter_info_t<std::string>>& get_string_parameters() const;

 private:
  pdlp_solver_settings_t<i_t, f_t> pdlp_settings;
  mip_solver_settings_t<i_t, f_t> mip_settings;

  std::vector<parameter_info_t<f_t>> float_parameters;
  std::vector<parameter_info_t<i_t>> int_parameters;
  std::vector<parameter_info_t<bool>> bool_parameters;
  std::vector<parameter_info_t<std::string>> string_parameters;
};

}  // namespace cuopt::linear_programming
