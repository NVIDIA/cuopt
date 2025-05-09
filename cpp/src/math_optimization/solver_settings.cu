/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <cuopt/linear_programming/solver_settings.hpp>
#include <mip/mip_constants.hpp>

namespace cuopt::linear_programming {

namespace {

bool string_to_int(const std::string& value, int& result)
{
  try {
    result = std::stoi(value);
    return true;
  } catch (const std::invalid_argument& e) {
    return false;
  }
}

template <typename f_t>
bool string_to_float(const std::string& value, f_t& result)
{
  try {
    if constexpr (std::is_same_v<f_t, float>) { result = std::stof(value); }
    if constexpr (std::is_same_v<f_t, double>) { result = std::stod(value); }
    return true;
  } catch (const std::invalid_argument& e) {
    return false;
  }
}

bool string_to_bool(const std::string& value, bool& result)
{
  if (value == "true" || value == "True" || value == "TRUE" || value == "1" || value == "t" ||
      value == "T") {
    result = true;
    return true;
  } else if (value == "false" || value == "False" || value == "FALSE" || value == "0" ||
             value == "f" || value == "F") {
    result = false;
    return true;
  } else {
    return false;
  }
}

}  // namespace

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_parameter_from_string(const std::string& name,
                                                            const std::string& value)
{
  bool found = false;
  for (auto& param : int_parameters) {
    if (param.param_name == name) {
      i_t value_int;
      if (string_to_int(value, value_int)) {
        if (value_int < param.min_value || value_int > param.max_value) {
          throw std::invalid_argument("Parameter " + name + " out of range");
        }
        *param.value_ptr = value_int;
        found            = true;
      } else {
        throw std::invalid_argument("Parameter " + name + " is not an integer");
      }
    }
  }
  for (auto& param : float_parameters) {
    if (param.param_name == name) {
      f_t value_float;
      if (string_to_float<f_t>(value, value_float)) {
        if (value_float < param.min_value || value_float > param.max_value) {
          throw std::invalid_argument("Parameter " + name + " out of range");
        }
        *param.value_ptr = value_float;
        found            = true;
      } else {
        throw std::invalid_argument("Parameter " + name + " is not a float");
      }
      return;
    }
  }
  for (auto& param : bool_parameters) {
    if (param.param_name == name) {
      bool value_bool;
      if (string_to_bool(value, value_bool)) {
        *param.value_ptr = value_bool;
        found            = true;
      } else {
        throw std::invalid_argument("Parameter " + name + " must be true or false");
      }
    }
  }

  for (auto& param : string_parameters) {
    if (param.param_name == name) {
      *param.value_ptr = value;
      found            = true;
    }
  }
  if (!found) { throw std::invalid_argument("Parameter " + name + " not found"); }
}

template <typename i_t, typename f_t>
template <typename T>
void solver_settings_t<i_t, f_t>::set_parameter(const std::string& name, T value)
{
  bool found = false;
  if constexpr (std::is_same_v<T, i_t>) {
    for (auto& param : int_parameters) {
      if (param.param_name == name) {
        if (value < param.min_value || value > param.max_value) {
          throw std::invalid_argument("Parameter " + name + " out of range");
        }
        *param.value_ptr = value;
        found            = true;
      }
    }
  }
  if constexpr (std::is_same_v<T, f_t>) {
    for (auto& param : float_parameters) {
      if (param.param_name == name) {
        if (value < param.min_value || value > param.max_value) {
          throw std::invalid_argument("Parameter " + name + " out of range");
        }
        *param.value_ptr = value;
        found            = true;
      }
    }
  }
  if constexpr (std::is_same_v<T, bool>) {
    for (auto& param : bool_parameters) {
      if (param.param_name == name) {
        *param.value_ptr = value;
        found            = true;
      }
    }
  }
  if constexpr (std::is_same_v<T, std::string>) {
    for (auto& param : string_parameters) {
      if (param.param_name == name) {
        *param.value_ptr = value;
        found            = true;
      }
    }
  }
  if (!found) { throw std::invalid_argument("Parameter " + name + " not found"); }
}

template <typename i_t, typename f_t>
template <typename T>
T solver_settings_t<i_t, f_t>::get_parameter(const std::string& name) const
{
  if constexpr (std::is_same_v<T, i_t>) {
    for (auto& param : int_parameters) {
      if (param.param_name == name) { return *param.value_ptr; }
    }
  }
  if constexpr (std::is_same_v<T, f_t>) {
    for (auto& param : float_parameters) {
      if (param.param_name == name) { return *param.value_ptr; }
    }
  }
  if constexpr (std::is_same_v<T, bool>) {
    for (auto& param : bool_parameters) {
      if (param.param_name == name) { return *param.value_ptr; }
    }
  }
  if constexpr (std::is_same_v<T, std::string>) {
    for (auto& param : string_parameters) {
      if (param.param_name == name) { return *param.value_ptr; }
    }
  }
  throw std::invalid_argument("Parameter " + name + " not found");
}

template <typename i_t, typename f_t>
std::string solver_settings_t<i_t, f_t>::get_parameter_as_string(const std::string& name) const
{
  for (auto& param : int_parameters) {
    if (param.param_name == name) { return std::to_string(*param.value_ptr); }
  }
  for (auto& param : float_parameters) {
    if (param.param_name == name) { return std::to_string(*param.value_ptr); }
  }
  for (auto& param : bool_parameters) {
    if (param.param_name == name) { return *param.value_ptr ? "true" : "false"; }
  }
  for (auto& param : string_parameters) {
    if (param.param_name == name) { return *param.value_ptr; }
  }
  throw std::invalid_argument("Parameter " + name + " not found");
}

// PDLP Settings Forwarding Implementations
template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_optimality_tolerance(f_t eps_optimal)
{
  pdlp_settings.set_optimality_tolerance(eps_optimal);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_absolute_dual_tolerance(f_t tol)
{
  pdlp_settings.set_absolute_dual_tolerance(tol);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_relative_dual_tolerance(f_t tol)
{
  pdlp_settings.set_relative_dual_tolerance(tol);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_absolute_primal_tolerance(f_t tol)
{
  pdlp_settings.set_absolute_primal_tolerance(tol);
  mip_settings.set_absolute_tolerance(tol);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_relative_primal_tolerance(f_t tol)
{
  pdlp_settings.set_relative_primal_tolerance(tol);
  mip_settings.set_relative_tolerance(tol);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_absolute_gap_tolerance(f_t tol)
{
  pdlp_settings.set_absolute_gap_tolerance(tol);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_relative_gap_tolerance(f_t tol)
{
  pdlp_settings.set_relative_gap_tolerance(tol);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_infeasibility_detection(bool detect)
{
  pdlp_settings.set_infeasibility_detection(detect);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_strict_infeasibility(bool strict)
{
  pdlp_settings.set_strict_infeasibility(strict);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_primal_infeasible_tolerance(f_t tol)
{
  pdlp_settings.set_primal_infeasible_tolerance(tol);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_dual_infeasible_tolerance(f_t tol)
{
  pdlp_settings.set_dual_infeasible_tolerance(tol);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_iteration_limit(i_t limit)
{
  pdlp_settings.set_iteration_limit(limit);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_time_limit(double limit)
{
  pdlp_settings.set_time_limit(limit);
  mip_settings.set_time_limit(limit);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_pdlp_solver_mode(pdlp_solver_mode_t mode)
{
  pdlp_settings.set_pdlp_solver_mode(mode);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_method(method_t method)
{
  pdlp_settings.set_method(method);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_crossover(bool crossover)
{
  pdlp_settings.set_crossover(crossover);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_log_file(std::string log_file)
{
  pdlp_settings.set_log_file(log_file);
  mip_settings.set_log_file(log_file);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_log_to_console(bool log_to_console)
{
  pdlp_settings.set_log_to_console(log_to_console);
  mip_settings.set_log_to_console(log_to_console);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_initial_pdlp_primal_solution(const f_t* solution,
                                                                   i_t size,
                                                                   rmm::cuda_stream_view stream)
{
  pdlp_settings.set_initial_primal_solution(solution, size, stream);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_initial_pdlp_dual_solution(const f_t* solution,
                                                                 i_t size,
                                                                 rmm::cuda_stream_view stream)
{
  pdlp_settings.set_initial_dual_solution(solution, size, stream);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_pdlp_warm_start_data(
  const f_t* current_primal_solution,
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
  f_t initial_primal_weight,
  f_t initial_step_size,
  i_t total_pdlp_iterations,
  i_t total_pdhg_iterations,
  f_t last_candidate_kkt_score,
  f_t last_restart_kkt_score,
  f_t sum_solution_weight,
  i_t iterations_since_last_restart)
{
  pdlp_settings.set_pdlp_warm_start_data(current_primal_solution,
                                         current_dual_solution,
                                         initial_primal_average,
                                         initial_dual_average,
                                         current_ATY,
                                         sum_primal_solutions,
                                         sum_dual_solutions,
                                         last_restart_duality_gap_primal_solution,
                                         last_restart_duality_gap_dual_solution,
                                         primal_size,
                                         dual_size,
                                         initial_primal_weight,
                                         initial_step_size,
                                         total_pdlp_iterations,
                                         total_pdhg_iterations,
                                         last_candidate_kkt_score,
                                         last_restart_kkt_score,
                                         sum_solution_weight,
                                         iterations_since_last_restart);
}

// PDLP Getters Implementations
template <typename i_t, typename f_t>
f_t solver_settings_t<i_t, f_t>::get_absolute_dual_tolerance() const noexcept
{
  return pdlp_settings.get_absolute_dual_tolerance();
}

template <typename i_t, typename f_t>
f_t solver_settings_t<i_t, f_t>::get_relative_dual_tolerance() const noexcept
{
  return pdlp_settings.get_relative_dual_tolerance();
}

template <typename i_t, typename f_t>
f_t solver_settings_t<i_t, f_t>::get_absolute_primal_tolerance() const noexcept
{
  return pdlp_settings.get_absolute_primal_tolerance();
}

template <typename i_t, typename f_t>
f_t solver_settings_t<i_t, f_t>::get_relative_primal_tolerance() const noexcept
{
  return pdlp_settings.get_relative_primal_tolerance();
}

template <typename i_t, typename f_t>
f_t solver_settings_t<i_t, f_t>::get_absolute_gap_tolerance() const noexcept
{
  return pdlp_settings.get_absolute_gap_tolerance();
}

template <typename i_t, typename f_t>
f_t solver_settings_t<i_t, f_t>::get_relative_gap_tolerance() const noexcept
{
  return pdlp_settings.get_relative_gap_tolerance();
}

template <typename i_t, typename f_t>
bool solver_settings_t<i_t, f_t>::get_infeasibility_detection() const noexcept
{
  return pdlp_settings.get_infeasibility_detection();
}

template <typename i_t, typename f_t>
bool solver_settings_t<i_t, f_t>::get_strict_infeasibility() const noexcept
{
  return pdlp_settings.get_strict_infeasibility();
}

template <typename i_t, typename f_t>
f_t solver_settings_t<i_t, f_t>::get_primal_infeasible_tolerance() const noexcept
{
  return pdlp_settings.get_primal_infeasible_tolerance();
}

template <typename i_t, typename f_t>
f_t solver_settings_t<i_t, f_t>::get_dual_infeasible_tolerance() const noexcept
{
  return pdlp_settings.get_dual_infeasible_tolerance();
}

template <typename i_t, typename f_t>
i_t solver_settings_t<i_t, f_t>::get_iteration_limit() const noexcept
{
  return pdlp_settings.get_iteration_limit();
}

template <typename i_t, typename f_t>
double solver_settings_t<i_t, f_t>::get_time_limit() const noexcept
{
  return mip_settings.get_time_limit();
}

template <typename i_t, typename f_t>
std::string solver_settings_t<i_t, f_t>::get_log_file() const noexcept
{
  return mip_settings.get_log_file();
}

template <typename i_t, typename f_t>
bool solver_settings_t<i_t, f_t>::get_log_to_console() const noexcept
{
  return mip_settings.get_log_to_console();
}

template <typename i_t, typename f_t>
pdlp_solver_mode_t solver_settings_t<i_t, f_t>::get_pdlp_solver_mode() const noexcept
{
  return pdlp_settings.get_pdlp_solver_mode();
}

template <typename i_t, typename f_t>
method_t solver_settings_t<i_t, f_t>::get_method() const noexcept
{
  return pdlp_settings.get_method();
}

template <typename i_t, typename f_t>
bool solver_settings_t<i_t, f_t>::get_crossover() const noexcept
{
  return pdlp_settings.get_crossover();
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& solver_settings_t<i_t, f_t>::get_initial_pdlp_primal_solution()
  const
{
  return pdlp_settings.get_initial_primal_solution();
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& solver_settings_t<i_t, f_t>::get_initial_pdlp_dual_solution() const
{
  return pdlp_settings.get_initial_dual_solution();
}

// MIP Settings Implementations
template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_absolute_tolerance(f_t tol)
{
  mip_settings.set_absolute_tolerance(tol);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_relative_tolerance(f_t tol)
{
  mip_settings.set_relative_tolerance(tol);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_integrality_tolerance(f_t tol)
{
  mip_settings.set_integrality_tolerance(tol);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_absolute_mip_gap(f_t tol)
{
  mip_settings.set_absolute_mip_gap(tol);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_relative_mip_gap(f_t tol)
{
  mip_settings.set_relative_mip_gap(tol);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_initial_mip_solution(const f_t* solution, i_t size)
{
  mip_settings.set_initial_solution(solution, size);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_mip_incumbent_solution_callback(
  internals::lp_incumbent_sol_callback_t* callback)
{
  mip_settings.set_incumbent_solution_callback(callback);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_mip_scaling(bool mip_scaling)
{
  mip_settings.set_mip_scaling(mip_scaling);
}

template <typename i_t, typename f_t>
f_t solver_settings_t<i_t, f_t>::get_absolute_tolerance() const noexcept
{
  return mip_settings.get_absolute_tolerance();
}

template <typename i_t, typename f_t>
f_t solver_settings_t<i_t, f_t>::get_relative_tolerance() const noexcept
{
  return mip_settings.get_relative_tolerance();
}

template <typename i_t, typename f_t>
f_t solver_settings_t<i_t, f_t>::get_integrality_tolerance() const noexcept
{
  return mip_settings.get_integrality_tolerance();
}

template <typename i_t, typename f_t>
f_t solver_settings_t<i_t, f_t>::get_absolute_mip_gap() const noexcept
{
  return mip_settings.get_absolute_mip_gap();
}

template <typename i_t, typename f_t>
f_t solver_settings_t<i_t, f_t>::get_relative_mip_gap() const noexcept
{
  return mip_settings.get_relative_mip_gap();
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& solver_settings_t<i_t, f_t>::get_initial_mip_solution() const
{
  return mip_settings.get_initial_solution();
}

template <typename i_t, typename f_t>
const internals::lp_incumbent_sol_callback_t*
solver_settings_t<i_t, f_t>::get_mip_incumbent_solution_callback() const
{
  return mip_settings.get_incumbent_solution_callback();
}

template <typename i_t, typename f_t>
bool solver_settings_t<i_t, f_t>::get_mip_heuristics_only() const noexcept
{
  return mip_settings.get_heuristics_only();
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_mip_heuristics_only(bool heuristics_only)
{
  mip_settings.set_heuristics_only(heuristics_only);
}

template <typename i_t, typename f_t>
i_t solver_settings_t<i_t, f_t>::get_mip_num_cpu_threads() const noexcept
{
  return mip_settings.get_num_cpu_threads();
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_mip_num_cpu_threads(i_t num_cpu_threads)
{
  mip_settings.set_num_cpu_threads(num_cpu_threads);
}

template <typename i_t, typename f_t>
pdlp_solver_settings_t<i_t, f_t>& solver_settings_t<i_t, f_t>::get_pdlp_settings()
{
  return pdlp_settings;
}

template <typename i_t, typename f_t>
mip_solver_settings_t<i_t, f_t>& solver_settings_t<i_t, f_t>::get_mip_settings()
{
  return mip_settings;
}

template <typename i_t, typename f_t>
const pdlp_warm_start_data_view_t<i_t, f_t>&
solver_settings_t<i_t, f_t>::get_pdlp_warm_start_data_view() const noexcept
{
  return pdlp_settings.get_pdlp_warm_start_data_view();
}

#if MIP_INSTANTIATE_FLOAT
template class solver_settings_t<int, float>;
template void solver_settings_t<int, float>::set_parameter(const std::string& name, int value);
template void solver_settings_t<int, float>::set_parameter(const std::string& name, float value);
template void solver_settings_t<int, float>::set_parameter(const std::string& name, bool value);
template int solver_settings_t<int, float>::get_parameter(const std::string& name) const;
template float solver_settings_t<int, float>::get_parameter(const std::string& name) const;
template bool solver_settings_t<int, float>::get_parameter(const std::string& name) const;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class solver_settings_t<int, double>;
template void solver_settings_t<int, double>::set_parameter(const std::string& name, int value);
template void solver_settings_t<int, double>::set_parameter(const std::string& name, double value);
template void solver_settings_t<int, double>::set_parameter(const std::string& name, bool value);
template int solver_settings_t<int, double>::get_parameter(const std::string& name) const;
template double solver_settings_t<int, double>::get_parameter(const std::string& name) const;
template bool solver_settings_t<int, double>::get_parameter(const std::string& name) const;
#endif

}  // namespace cuopt::linear_programming
