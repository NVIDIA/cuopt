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

#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>

#include <raft/core/device_span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include <optional>

namespace cuopt::linear_programming {

template <typename T>
struct parameter_info_t {
  parameter_info_t(std::string_view param_name, T* value, T min, T max, T def)
    : param_name(param_name), value_ptr(value), min_value(min), max_value(max), default_value(def)
  {
  }
  std::string param_name;
  T* value_ptr;
  T min_value;
  T max_value;
  T default_value;
};

template <>
struct parameter_info_t<bool> {
  parameter_info_t(std::string_view name, bool* value, bool def)
    : param_name(name), value_ptr(value), default_value(def)
  {
  }
  std::string param_name;
  bool* value_ptr;
  bool default_value;
};

template <>
struct parameter_info_t<std::string> {
  parameter_info_t(std::string_view name, std::string* value, std::string def)
    : param_name(name), value_ptr(value), default_value(def)
  {
  }
  std::string param_name;
  std::string* value_ptr;
  std::string default_value;
};

template <typename i_t, typename f_t>
class solver_settings_t {
 public:
  solver_settings_t() : pdlp_settings(), mip_settings()
  {
    // clang-format off
    // Float parameters
    float_parameters = {
      {CUOPT_TIME_LIMIT, &mip_settings.time_limit_, 0, std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()},
      {CUOPT_TIME_LIMIT, &pdlp_settings.time_limit_, 0, std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()},
      {CUOPT_ABSOLUTE_DUAL_TOLERANCE, &pdlp_settings.tolerances.absolute_dual_tolerance, 1e-12, 1e-1, 1e-4},
      {CUOPT_RELATIVE_DUAL_TOLERANCE, &pdlp_settings.tolerances.relative_dual_tolerance, 1e-12, 1e-1, 1e-4},
      {CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, &pdlp_settings.tolerances.absolute_primal_tolerance, 1e-12, 1e-1, 1e-4},
      {CUOPT_RELATIVE_PRIMAL_TOLERANCE, &pdlp_settings.tolerances.relative_primal_tolerance, 1e-12, 1e-1, 1e-4},
      {CUOPT_ABSOLUTE_GAP_TOLERANCE, &pdlp_settings.tolerances.absolute_gap_tolerance, 1e-12, 1e-1, 1e-4},
      {CUOPT_RELATIVE_GAP_TOLERANCE, &pdlp_settings.tolerances.relative_gap_tolerance, 1e-12, 1e-1, 1e-4},
      {CUOPT_MIP_ABSOLUTE_TOLERANCE, &mip_settings.tolerances.absolute_tolerance, 1e-12, 1e-1, 1e-4},
      {CUOPT_MIP_RELATIVE_TOLERANCE, &mip_settings.tolerances.relative_tolerance, 1e-12, 1e-1, 1e-6},
      {CUOPT_MIP_INTEGRALITY_TOLERANCE, &mip_settings.tolerances.integrality_tolerance, 1e-12, 1e-1, 1e-5}
    };

    // Int parameters
    int_parameters = {
      {CUOPT_ITERATION_LIMIT, &pdlp_settings.iteration_limit_, 1, std::numeric_limits<i_t>::max(), std::numeric_limits<i_t>::max()},
      {CUOPT_PDLP_SOLVER_MODE, reinterpret_cast<int*>(&pdlp_settings.solver_mode_), CUOPT_PDLP_SOLVER_MODE_STABLE1, CUOPT_PDLP_SOLVER_MODE_FAST1, CUOPT_PDLP_SOLVER_MODE_STABLE2},
      {CUOPT_METHOD, reinterpret_cast<int*>(&pdlp_settings.method_), CUOPT_METHOD_CONCURRENT, CUOPT_METHOD_DUAL_SIMPLEX, CUOPT_METHOD_CONCURRENT},
      {CUOPT_NUM_CPU_THREADS, &mip_settings.num_cpu_threads_, -1, std::numeric_limits<i_t>::max(), -1}
    };

    // Bool parameters
    bool_parameters = {
      {CUOPT_INFEASIBILITY_DETECTION, &pdlp_settings.detect_infeasibility_, true},
      {CUOPT_STRICT_INFEASIBILITY, &pdlp_settings.strict_infeasibility_, false},
      {CUOPT_PER_CONSTRAINT_RESIDUAL, &pdlp_settings.per_constraint_residual_, false},
      {CUOPT_SAVE_BEST_PRIMAL_SO_FAR, &pdlp_settings.save_best_primal_so_far_, false},
      {CUOPT_FIRST_PRIMAL_FEASIBLE, &pdlp_settings.first_primal_feasible_, false},
      {CUOPT_MIP_SCALING, &mip_settings.mip_scaling_, true},
      {CUOPT_MIP_HEURISTICS_ONLY, &mip_settings.heuristics_only_, false}
    };

    // String parameters
    string_parameters = {
      {CUOPT_LOG_FILE, &mip_settings.log_file_, ""}
    };
    // clang-format on
  }

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

  // PDLP Settings
  void set_optimality_tolerance(f_t eps_optimal);
  void set_absolute_dual_tolerance(f_t absolute_dual_tolerance);
  void set_relative_dual_tolerance(f_t relative_dual_tolerance);
  void set_absolute_primal_tolerance(f_t absolute_primal_tolerance);
  void set_relative_primal_tolerance(f_t relative_primal_tolerance);
  void set_absolute_gap_tolerance(f_t absolute_gap_tolerance);
  void set_relative_gap_tolerance(f_t relative_gap_tolerance);
  void set_infeasibility_detection(bool detect);
  void set_strict_infeasibility(bool strict_infeasibility);
  void set_primal_infeasible_tolerance(f_t primal_infeasible_tolerance);
  void set_dual_infeasible_tolerance(f_t dual_infeasible_tolerance);
  void set_iteration_limit(i_t iteration_limit);
  void set_time_limit(double time_limit);
  void set_pdlp_solver_mode(pdlp_solver_mode_t solver_mode);
  void set_method(method_t method);
  void set_crossover(bool crossover);
  void set_log_file(std::string log_file);
  void set_log_to_console(bool log_to_console);
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

  f_t get_absolute_dual_tolerance() const noexcept;
  f_t get_relative_dual_tolerance() const noexcept;
  f_t get_absolute_primal_tolerance() const noexcept;
  f_t get_relative_primal_tolerance() const noexcept;
  f_t get_absolute_gap_tolerance() const noexcept;
  f_t get_relative_gap_tolerance() const noexcept;
  bool get_infeasibility_detection() const noexcept;
  bool get_strict_infeasibility() const noexcept;
  f_t get_primal_infeasible_tolerance() const noexcept;
  f_t get_dual_infeasible_tolerance() const noexcept;
  i_t get_iteration_limit() const noexcept;
  double get_time_limit() const noexcept;
  pdlp_solver_mode_t get_pdlp_solver_mode() const noexcept;
  method_t get_method() const noexcept;
  bool get_crossover() const noexcept;
  std::string get_log_file() const noexcept;
  bool get_log_to_console() const noexcept;
  const rmm::device_uvector<f_t>& get_initial_pdlp_primal_solution() const;
  const rmm::device_uvector<f_t>& get_initial_pdlp_dual_solution() const;

  // MIP Settings
  void set_absolute_tolerance(f_t absolute_tolerance);
  void set_relative_tolerance(f_t relative_tolerance);
  void set_integrality_tolerance(f_t integrality_tolerance);
  void set_absolute_mip_gap(f_t absolute_mip_gap);
  void set_relative_mip_gap(f_t relative_mip_gap);
  void set_initial_mip_solution(const f_t* initial_solution, i_t size);
  void set_mip_incumbent_solution_callback(
    internals::lp_incumbent_sol_callback_t* callback = nullptr);
  void set_mip_scaling(bool mip_scaling);
  void set_mip_heuristics_only(bool heuristics_only);
  void set_mip_num_cpu_threads(i_t num_cpu_threads);
  bool get_mip_heuristics_only() const noexcept;
  i_t get_mip_num_cpu_threads() const noexcept;

  f_t get_absolute_tolerance() const noexcept;
  f_t get_relative_tolerance() const noexcept;
  f_t get_integrality_tolerance() const noexcept;
  f_t get_absolute_mip_gap() const noexcept;
  f_t get_relative_mip_gap() const noexcept;
  const rmm::device_uvector<f_t>& get_initial_mip_solution() const;
  const pdlp_warm_start_data_view_t<i_t, f_t>& get_pdlp_warm_start_data_view() const noexcept;
  const internals::lp_incumbent_sol_callback_t* get_mip_incumbent_solution_callback() const;

  pdlp_solver_settings_t<i_t, f_t>& get_pdlp_settings();
  mip_solver_settings_t<i_t, f_t>& get_mip_settings();

 private:
  pdlp_solver_settings_t<i_t, f_t> pdlp_settings;
  mip_solver_settings_t<i_t, f_t> mip_settings;

  std::vector<parameter_info_t<f_t>> float_parameters;
  std::vector<parameter_info_t<i_t>> int_parameters;
  std::vector<parameter_info_t<bool>> bool_parameters;
  std::vector<parameter_info_t<std::string>> string_parameters;
};

}  // namespace cuopt::linear_programming
