/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <cuopt/export.hpp>
#include <cuopt/mathematical_optimization/solver_settings.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <utilities/logger.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace cuopt::mathematical_optimization {

namespace {

bool string_to_int(const std::string& value, int& result)
{
  try {
    size_t pos = 0;
    result     = std::stoi(value, &pos);
    return pos == value.size();
  } catch (const std::exception&) {
    return false;
  }
}

template <typename f_t>
bool string_to_float(const std::string& value, f_t& result)
{
  try {
    size_t pos = 0;
    if constexpr (std::is_same_v<f_t, float>) { result = std::stof(value, &pos); }
    if constexpr (std::is_same_v<f_t, double>) { result = std::stod(value, &pos); }
    if (std::isnan(result)) { return false; }
    return pos == value.size();
  } catch (const std::exception&) {
    return false;
  }
}

std::string quote_if_needed(const std::string& s)
{
  bool needs_quoting = s.empty() || s.find(' ') != std::string::npos ||
                       s.find('"') != std::string::npos || s.find('\t') != std::string::npos;
  if (!needs_quoting) return s;
  std::string out = "\"";
  for (char c : s) {
    if (c == '"')
      out += "\\\"";
    else
      out += c;
  }
  out += '"';
  return out;
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
  bool found  = false;
  bool output = false;
  for (auto& param : int_parameters) {
    if (param.param_name == name) {
      i_t value_int;
      if (string_to_int(value, value_int)) {
        if (value_int < param.min_value || value_int > param.max_value) {
          throw std::invalid_argument("Parameter " + name + " value " + value + " out of range");
        }
        *param.value_ptr = value_int;
        found            = true;
        if (!output) {
          CUOPT_LOG_INFO("Setting parameter %s to %d", name.c_str(), value_int);
          output = true;
        }
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
        if (!output) {
          CUOPT_LOG_INFO("Setting parameter %s to %e", name.c_str(), value_float);
          output = true;
        }
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
        if (!output) {
          CUOPT_LOG_INFO("Setting parameter %s to %s", name.c_str(), value_bool ? "true" : "false");
          output = true;
        }
      } else {
        throw std::invalid_argument("Parameter " + name + " value " + value +
                                    " must be true or false");
      }
    }
  }

  for (auto& param : string_parameters) {
    if (param.param_name == name) {
      *param.value_ptr = value;
      if (!output) {
        CUOPT_LOG_INFO("Setting parameter %s to %s", name.c_str(), value.c_str());
        output = true;
      }
      found = true;
    }
  }
  if (!found) { throw std::invalid_argument("Parameter " + name + " not found"); }
}

template <typename i_t, typename f_t>
template <typename T>
void solver_settings_t<i_t, f_t>::set_parameter(const std::string& name, T value)
{
  bool found  = false;
  bool output = false;
  if constexpr (std::is_same_v<T, i_t>) {
    for (auto& param : int_parameters) {
      if (param.param_name == name) {
        if (value < param.min_value || value > param.max_value) {
          throw std::invalid_argument("Parameter " + name + " out of range");
        }
        *param.value_ptr = value;
        if (!output) {
          CUOPT_LOG_INFO("Setting parameter %s to %d", name.c_str(), value);
          output = true;
        }
        found = true;
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
        if (!output) {
          CUOPT_LOG_INFO("Setting parameter %s to %e", name.c_str(), value);
          output = true;
        }
        found = true;
      }
    }
  }
  if constexpr (std::is_same_v<T, bool>) {
    for (auto& param : bool_parameters) {
      if (param.param_name == name) {
        *param.value_ptr = value;
        if (!output) {
          CUOPT_LOG_INFO("Setting parameter %s to %s", name.c_str(), value ? "true" : "false");
          output = true;
        }
        found = true;
      }
    }
  }
  if constexpr (std::is_same_v<T, std::string>) {
    for (auto& param : string_parameters) {
      if (param.param_name == name) {
        *param.value_ptr = value;
        if (!output) {
          CUOPT_LOG_INFO("Setting parameter %s to %s", name.c_str(), value.c_str());
          output = true;
        }
        found = true;
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
void solver_settings_t<i_t, f_t>::set_mip_callback(internals::base_solution_callback_t* callback,
                                                   void* user_data)
{
  mip_settings.set_mip_callback(callback, user_data);
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

template <typename i_t, typename f_t>
const std::vector<std::string> solver_settings_t<i_t, f_t>::get_parameter_names() const
{
  std::vector<std::string> parameter_names;
  for (auto& param : int_parameters) {
    parameter_names.push_back(param.param_name);
  }
  for (auto& param : float_parameters) {
    parameter_names.push_back(param.param_name);
  }
  for (auto& param : bool_parameters) {
    parameter_names.push_back(param.param_name);
  }
  for (auto& param : string_parameters) {
    parameter_names.push_back(param.param_name);
  }
  return parameter_names;
}

template <typename i_t, typename f_t>
void solver_settings_t<i_t, f_t>::load_parameters_from_file(const std::string& path)
{
  cuopt_expects(!std::filesystem::is_directory(path) && std::filesystem::exists(path),
                error_type_t::ValidationError,
                "Parameter config: not a valid file: %s",
                path.c_str());
  std::ifstream file(path);
  cuopt_expects(file.is_open(),
                error_type_t::ValidationError,
                "Parameter config: cannot open: %s",
                path.c_str());
  std::string line;
  while (std::getline(file, line)) {
    auto first_non_ws = std::find_if_not(line.begin(), line.end(), ::isspace);
    if (first_non_ws == line.end() || *first_non_ws == '#') continue;
    line.erase(line.begin(), first_non_ws);

    std::istringstream iss(line);
    std::string key;
    cuopt_expects(iss >> key >> std::ws && iss.get() == '=',
                  error_type_t::ValidationError,
                  "Parameter config: bad line: %s",
                  line.c_str());
    iss >> std::ws;
    cuopt_expects(!iss.eof(),
                  error_type_t::ValidationError,
                  "Parameter config: missing value: %s",
                  line.c_str());
    std::string val;
    if (iss.peek() == '"') {
      iss.get();
      val.clear();
      char ch;
      bool closed = false;
      while (iss.get(ch)) {
        if (ch == '\\' && iss.peek() == '"') {
          iss.get(ch);
          val += '"';
        } else if (ch == '"') {
          closed = true;
          break;
        } else {
          val += ch;
        }
      }
      cuopt_expects(closed,
                    error_type_t::ValidationError,
                    "Parameter config: unterminated quote: %s",
                    line.c_str());
    } else {
      iss >> val;
    }
    std::string trailing;
    cuopt_expects(!bool(iss >> trailing),
                  error_type_t::ValidationError,
                  "Parameter config: trailing junk: %s",
                  line.c_str());
    try {
      set_parameter_from_string(key, val);
    } catch (const std::invalid_argument& e) {
      cuopt_expects(false, error_type_t::ValidationError, "Parameter config: %s", e.what());
    }
  }
  CUOPT_LOG_INFO("Parameters loaded from: %s", path.c_str());
}

template <typename i_t, typename f_t>
bool solver_settings_t<i_t, f_t>::dump_parameters_to_file(const std::string& path,
                                                          bool hyperparameters_only) const
{
  std::ofstream file(path);
  if (!file.is_open()) {
    CUOPT_LOG_ERROR("Cannot open file for writing: %s", path.c_str());
    return false;
  }
  file << "# cuOpt parameter configuration (auto-generated)\n";
  file << "# Uncomment and change the values you wish to override.\n\n";
  for (const auto& p : int_parameters) {
    if (hyperparameters_only && p.param_name.find("hyper_") == std::string::npos) continue;
    if (p.description && p.description[0] != '\0')
      file << "# " << p.description << " (int, range: [" << p.min_value << ", " << p.max_value
           << "])\n";
    file << "# " << p.param_name << " = " << *p.value_ptr << "\n\n";
  }
  for (const auto& p : float_parameters) {
    if (hyperparameters_only && p.param_name.find("hyper_") == std::string::npos) continue;
    if (p.description && p.description[0] != '\0')
      file << "# " << p.description << " (double, range: [" << p.min_value << ", " << p.max_value
           << "])\n";
    file << "# " << p.param_name << " = " << *p.value_ptr << "\n\n";
  }
  for (const auto& p : bool_parameters) {
    if (hyperparameters_only && p.param_name.find("hyper_") == std::string::npos) continue;
    if (p.description && p.description[0] != '\0') file << "# " << p.description << " (bool)\n";
    file << "# " << p.param_name << " = " << (*p.value_ptr ? "true" : "false") << "\n\n";
  }
  for (const auto& p : string_parameters) {
    if (hyperparameters_only && p.param_name.find("hyper_") == std::string::npos) continue;
    if (p.description && p.description[0] != '\0') file << "# " << p.description << " (string)\n";
    file << "# " << p.param_name << " = " << quote_if_needed(*p.value_ptr) << "\n\n";
  }
  return true;
}

// NOTE: deliberately no `template class solver_settings_t<...>` here.
//
// That would instantiate every member, including the implicitly-defined constructor and
// copy constructor. Those construct a pdlp_solver_settings_t, which holds a
// pdlp_warm_start_data_t by value, whose default ctor lives in a CUDA translation unit --
// so the whole class instantiation drags a CUDA dependency into this CUDA-free library and
// leaves libcuopt_client.so with an undefined symbol. Members are therefore instantiated
// individually below; libcuopt emits the constructors via its own `template class`
// in solver_settings_gpu.cu.

#if MIP_INSTANTIATE_FLOAT
template CUOPT_EXPORT void solver_settings_t<int, float>::set_parameter_from_string(
  const std::string&, const std::string&);
template CUOPT_EXPORT std::string solver_settings_t<int, float>::get_parameter_as_string(
  const std::string&) const;
template CUOPT_EXPORT void solver_settings_t<int, float>::set_mip_callback(
  internals::base_solution_callback_t*, void*);
template CUOPT_EXPORT const std::vector<internals::base_solution_callback_t*>
solver_settings_t<int, float>::get_mip_callbacks() const;
template CUOPT_EXPORT pdlp_solver_settings_t<int, float>&
solver_settings_t<int, float>::get_pdlp_settings();
template CUOPT_EXPORT mip_solver_settings_t<int, float>&
solver_settings_t<int, float>::get_mip_settings();
template CUOPT_EXPORT const std::vector<parameter_info_t<float>>&
solver_settings_t<int, float>::get_float_parameters() const;
template CUOPT_EXPORT const std::vector<parameter_info_t<int>>&
solver_settings_t<int, float>::get_int_parameters() const;
template CUOPT_EXPORT const std::vector<parameter_info_t<bool>>&
solver_settings_t<int, float>::get_bool_parameters() const;
template CUOPT_EXPORT const std::vector<std::string>
solver_settings_t<int, float>::get_parameter_names() const;
template CUOPT_EXPORT const std::vector<parameter_info_t<std::string>>&
solver_settings_t<int, float>::get_string_parameters() const;
template CUOPT_EXPORT const pdlp_warm_start_data_view_t<int, float>&
solver_settings_t<int, float>::get_pdlp_warm_start_data_view() const noexcept;
template CUOPT_EXPORT void solver_settings_t<int, float>::load_parameters_from_file(
  const std::string&);
template CUOPT_EXPORT bool solver_settings_t<int, float>::dump_parameters_to_file(
  const std::string&, bool) const;
template CUOPT_EXPORT void solver_settings_t<int, float>::set_parameter(const std::string& name,
                                                                        int value);
template CUOPT_EXPORT void solver_settings_t<int, float>::set_parameter(const std::string& name,
                                                                        float value);
template CUOPT_EXPORT void solver_settings_t<int, float>::set_parameter(const std::string& name,
                                                                        bool value);
template CUOPT_EXPORT int solver_settings_t<int, float>::get_parameter(
  const std::string& name) const;
template CUOPT_EXPORT float solver_settings_t<int, float>::get_parameter(
  const std::string& name) const;
template CUOPT_EXPORT bool solver_settings_t<int, float>::get_parameter(
  const std::string& name) const;
template CUOPT_EXPORT std::string solver_settings_t<int, float>::get_parameter(
  const std::string& name) const;
#endif

#if MIP_INSTANTIATE_DOUBLE
template CUOPT_EXPORT void solver_settings_t<int, double>::set_parameter_from_string(
  const std::string&, const std::string&);
template CUOPT_EXPORT std::string solver_settings_t<int, double>::get_parameter_as_string(
  const std::string&) const;
template CUOPT_EXPORT void solver_settings_t<int, double>::set_mip_callback(
  internals::base_solution_callback_t*, void*);
template CUOPT_EXPORT const std::vector<internals::base_solution_callback_t*>
solver_settings_t<int, double>::get_mip_callbacks() const;
template CUOPT_EXPORT pdlp_solver_settings_t<int, double>&
solver_settings_t<int, double>::get_pdlp_settings();
template CUOPT_EXPORT mip_solver_settings_t<int, double>&
solver_settings_t<int, double>::get_mip_settings();
template CUOPT_EXPORT const std::vector<parameter_info_t<double>>&
solver_settings_t<int, double>::get_float_parameters() const;
template CUOPT_EXPORT const std::vector<parameter_info_t<int>>&
solver_settings_t<int, double>::get_int_parameters() const;
template CUOPT_EXPORT const std::vector<parameter_info_t<bool>>&
solver_settings_t<int, double>::get_bool_parameters() const;
template CUOPT_EXPORT const std::vector<std::string>
solver_settings_t<int, double>::get_parameter_names() const;
template CUOPT_EXPORT const std::vector<parameter_info_t<std::string>>&
solver_settings_t<int, double>::get_string_parameters() const;
template CUOPT_EXPORT const pdlp_warm_start_data_view_t<int, double>&
solver_settings_t<int, double>::get_pdlp_warm_start_data_view() const noexcept;
template CUOPT_EXPORT void solver_settings_t<int, double>::load_parameters_from_file(
  const std::string&);
template CUOPT_EXPORT bool solver_settings_t<int, double>::dump_parameters_to_file(
  const std::string&, bool) const;
template CUOPT_EXPORT void solver_settings_t<int, double>::set_parameter(const std::string& name,
                                                                         int value);
template CUOPT_EXPORT void solver_settings_t<int, double>::set_parameter(const std::string& name,
                                                                         double value);
template CUOPT_EXPORT void solver_settings_t<int, double>::set_parameter(const std::string& name,
                                                                         bool value);
template CUOPT_EXPORT int solver_settings_t<int, double>::get_parameter(
  const std::string& name) const;
template CUOPT_EXPORT double solver_settings_t<int, double>::get_parameter(
  const std::string& name) const;
template CUOPT_EXPORT bool solver_settings_t<int, double>::get_parameter(
  const std::string& name) const;
template CUOPT_EXPORT std::string solver_settings_t<int, double>::get_parameter(
  const std::string& name) const;
#endif

}  // namespace cuopt::mathematical_optimization
