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
solver_settings_t<i_t, f_t>::solver_settings_t() : pdlp_settings(), mip_settings()
{
  // clang-format off
  // Float parameters
  float_parameters = {
    {CUOPT_TIME_LIMIT, &mip_settings.time_limit, 0.0, std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()},
    {CUOPT_TIME_LIMIT, &pdlp_settings.time_limit, 0.0, std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()},
    {CUOPT_ABSOLUTE_DUAL_TOLERANCE, &pdlp_settings.tolerances.absolute_dual_tolerance, 0.0, 1e-1, 1e-4},
    {CUOPT_RELATIVE_DUAL_TOLERANCE, &pdlp_settings.tolerances.relative_dual_tolerance, 0.0, 1e-1, 1e-4},
    {CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, &pdlp_settings.tolerances.absolute_primal_tolerance, 0.0, 1e-1, 1e-4},
    {CUOPT_RELATIVE_PRIMAL_TOLERANCE, &pdlp_settings.tolerances.relative_primal_tolerance, 0.0, 1e-1, 1e-4},
    {CUOPT_ABSOLUTE_GAP_TOLERANCE, &pdlp_settings.tolerances.absolute_gap_tolerance, 0.0, 1e-1, 1e-4},
    {CUOPT_RELATIVE_GAP_TOLERANCE, &pdlp_settings.tolerances.relative_gap_tolerance, 0.0, 1e-1, 1e-4},
    {CUOPT_MIP_ABSOLUTE_TOLERANCE, &mip_settings.tolerances.absolute_tolerance, 0.0, 1e-1, 1e-4},
    {CUOPT_MIP_RELATIVE_TOLERANCE, &mip_settings.tolerances.relative_tolerance, 0.0, 1e-1, 1e-4},
    {CUOPT_MIP_INTEGRALITY_TOLERANCE, &mip_settings.tolerances.integrality_tolerance, 0.0, 1e-1, 1e-5},
    {CUOPT_MIP_ABSOLUTE_GAP, &mip_settings.tolerances.absolute_mip_gap, 0.0, 1e-1, 1e-10},
    {CUOPT_MIP_RELATIVE_GAP, &mip_settings.tolerances.relative_mip_gap, 0.0, 1e-1, 1e-4},
    {CUOPT_PRIMAL_INFEASIBLE_TOLERANCE, &pdlp_settings.tolerances.primal_infeasible_tolerance, 0.0, 1e-1, 1e-8},
    {CUOPT_DUAL_INFEASIBLE_TOLERANCE, &pdlp_settings.tolerances.dual_infeasible_tolerance, 0.0, 1e-1, 1e-8}
   };

  // Int parameters
  int_parameters = {
    {CUOPT_ITERATION_LIMIT, &pdlp_settings.iteration_limit, 0, std::numeric_limits<i_t>::max(), std::numeric_limits<i_t>::max()},
    {CUOPT_PDLP_SOLVER_MODE, reinterpret_cast<int*>(&pdlp_settings.pdlp_solver_mode), CUOPT_PDLP_SOLVER_MODE_STABLE1, CUOPT_PDLP_SOLVER_MODE_FAST1, CUOPT_PDLP_SOLVER_MODE_STABLE2},
    {CUOPT_METHOD, reinterpret_cast<int*>(&pdlp_settings.method), CUOPT_METHOD_CONCURRENT, CUOPT_METHOD_DUAL_SIMPLEX, CUOPT_METHOD_CONCURRENT},
    {CUOPT_NUM_CPU_THREADS, &mip_settings.num_cpu_threads, -1, std::numeric_limits<i_t>::max(), -1}
  };

    // Bool parameters
  bool_parameters = {
    {CUOPT_INFEASIBILITY_DETECTION, &pdlp_settings.detect_infeasibility, false},
    {CUOPT_STRICT_INFEASIBILITY, &pdlp_settings.strict_infeasibility, false},
    {CUOPT_PER_CONSTRAINT_RESIDUAL, &pdlp_settings.per_constraint_residual, false},
    {CUOPT_SAVE_BEST_PRIMAL_SO_FAR, &pdlp_settings.save_best_primal_so_far, false},
    {CUOPT_FIRST_PRIMAL_FEASIBLE, &pdlp_settings.first_primal_feasible, false},
    {CUOPT_MIP_SCALING, &mip_settings.mip_scaling, true},
    {CUOPT_MIP_HEURISTICS_ONLY, &mip_settings.heuristics_only, false},
    {CUOPT_LOG_TO_CONSOLE, &pdlp_settings.log_to_console, true},
    {CUOPT_LOG_TO_CONSOLE, &mip_settings.log_to_console, true},
    {CUOPT_CROSSOVER, &pdlp_settings.crossover, false}
  };
  // String parameters
  string_parameters = {
    {CUOPT_LOG_FILE,  &mip_settings.log_file, ""},
    {CUOPT_LOG_FILE,  &pdlp_settings.log_file, ""},
    {CUOPT_SOL_FILE,  &mip_settings.sol_file, ""},
    {CUOPT_SOL_FILE,  &pdlp_settings.sol_file, ""}
  };
  // clang-format on
}

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
          throw std::invalid_argument("Parameter " + name + " value " + value + " out of range");
        }
        *param.value_ptr = value_int;
        found            = true;
      } else {
        throw std::invalid_argument("Parameter " + name + " value " + value + " is not an integer");
      }
    }
  }
  for (auto& param : float_parameters) {
    if (param.param_name == name) {
      f_t value_float;
      if (string_to_float<f_t>(value, value_float)) {
        if (value_float < param.min_value || value_float > param.max_value) {
          throw std::invalid_argument("Parameter " + name + " value " + value + " out of range");
        }
        *param.value_ptr = value_float;
        found            = true;
      } else {
        throw std::invalid_argument("Parameter " + name + " value " + value + " is not a float");
      }
    }
  }
  for (auto& param : bool_parameters) {
    if (param.param_name == name) {
      bool value_bool;
      if (string_to_bool(value, value_bool)) {
        *param.value_ptr = value_bool;
        found            = true;
      } else {
        throw std::invalid_argument("Parameter " + name + " value " + value +
                                    " must be true or false");
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

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_initial_mip_solution(const f_t* solution, i_t size)
{
  mip_settings.set_initial_solution(solution, size);
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::set_mip_callback(internals::base_solution_callback_t* callback)
{
  mip_settings.set_mip_callback(callback);
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& solver_settings_t<i_t, f_t>::get_initial_mip_solution() const
{
  return mip_settings.get_initial_solution();
}

template <typename i_t, typename f_t>
const std::vector<internals::base_solution_callback_t*>
solver_settings_t<i_t, f_t>::get_mip_callbacks() const
{
  return mip_settings.get_mip_callbacks();
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

template <typename i_t, typename f_t>
const std::vector<parameter_info_t<f_t>>& solver_settings_t<i_t, f_t>::get_float_parameters() const
{
  return float_parameters;
}

template <typename i_t, typename f_t>
const std::vector<parameter_info_t<i_t>>& solver_settings_t<i_t, f_t>::get_int_parameters() const
{
  return int_parameters;
}

template <typename i_t, typename f_t>
const std::vector<parameter_info_t<bool>>& solver_settings_t<i_t, f_t>::get_bool_parameters() const
{
  return bool_parameters;
}

template <typename i_t, typename f_t>
const std::vector<parameter_info_t<std::string>>&
solver_settings_t<i_t, f_t>::get_string_parameters() const
{
  return string_parameters;
}

#if MIP_INSTANTIATE_FLOAT
template class solver_settings_t<int, float>;
template void solver_settings_t<int, float>::set_parameter(const std::string& name, int value);
template void solver_settings_t<int, float>::set_parameter(const std::string& name, float value);
template void solver_settings_t<int, float>::set_parameter(const std::string& name, bool value);
template int solver_settings_t<int, float>::get_parameter(const std::string& name) const;
template float solver_settings_t<int, float>::get_parameter(const std::string& name) const;
template bool solver_settings_t<int, float>::get_parameter(const std::string& name) const;
template std::string solver_settings_t<int, float>::get_parameter(const std::string& name) const;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class solver_settings_t<int, double>;
template void solver_settings_t<int, double>::set_parameter(const std::string& name, int value);
template void solver_settings_t<int, double>::set_parameter(const std::string& name, double value);
template void solver_settings_t<int, double>::set_parameter(const std::string& name, bool value);
template int solver_settings_t<int, double>::get_parameter(const std::string& name) const;
template double solver_settings_t<int, double>::get_parameter(const std::string& name) const;
template bool solver_settings_t<int, double>::get_parameter(const std::string& name) const;
template std::string solver_settings_t<int, double>::get_parameter(const std::string& name) const;
#endif

}  // namespace cuopt::linear_programming
