/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/cpu_pdlp_warm_start_data.hpp>
#include <cuopt/mathematical_optimization/mip/mip_parameters.hpp>
#include <cuopt/mathematical_optimization/pdlp/pdlp_parameters.hpp>
#include <cuopt/mathematical_optimization/solver_parameter_registry.hpp>

#include <string>
#include <vector>

namespace cuopt {
namespace mathematical_optimization {

/**
 * @brief Solver settings with no GPU-backed members, the counterpart of cpu_optimization_problem_t.
 *
 * Same parameters, same names, defaults and ranges as solver_settings_t, because both bind the
 * same pdlp_parameters_t / mip_parameters_t through one solver_parameter_registry_t. What it
 * leaves out is what needs a GPU: device initial solutions, device warm start and callbacks.
 * The PDLP warm start is the CPU one, for remote solves.
 */
template <typename i_t, typename f_t>
class cpu_solver_settings_t {
 public:
  cpu_solver_settings_t() : registry_(pdlp_, mip_) {}

  cpu_solver_settings_t(const cpu_solver_settings_t&)            = delete;
  cpu_solver_settings_t& operator=(const cpu_solver_settings_t&) = delete;
  cpu_solver_settings_t(cpu_solver_settings_t&&)                 = delete;
  cpu_solver_settings_t& operator=(cpu_solver_settings_t&&)      = delete;

  void set_parameter_from_string(const std::string& name, const std::string& value)
  {
    registry_.set_parameter_from_string(name, value);
  }

  template <typename T>
  void set_parameter(const std::string& name, T value)
  {
    registry_.set_parameter(name, value);
  }

  template <typename T>
  T get_parameter(const std::string& name) const
  {
    return registry_.template get_parameter<T>(name);
  }

  std::string get_parameter_as_string(const std::string& name) const
  {
    return registry_.get_parameter_as_string(name);
  }

  const std::vector<parameter_info_t<f_t>>& get_float_parameters() const
  {
    return registry_.get_float_parameters();
  }
  const std::vector<parameter_info_t<i_t>>& get_int_parameters() const
  {
    return registry_.get_int_parameters();
  }
  const std::vector<parameter_info_t<bool>>& get_bool_parameters() const
  {
    return registry_.get_bool_parameters();
  }
  const std::vector<parameter_info_t<std::string>>& get_string_parameters() const
  {
    return registry_.get_string_parameters();
  }
  const std::vector<std::string> get_parameter_names() const
  {
    return registry_.get_parameter_names();
  }

  void load_parameters_from_file(const std::string& path)
  {
    registry_.load_parameters_from_file(path);
  }
  bool dump_parameters_to_file(const std::string& path, bool hyperparameters_only = true) const
  {
    return registry_.dump_parameters_to_file(path, hyperparameters_only);
  }

  pdlp_parameters_t<i_t, f_t>& get_pdlp_parameters() { return pdlp_; }
  const pdlp_parameters_t<i_t, f_t>& get_pdlp_parameters() const { return pdlp_; }
  mip_parameters_t<i_t, f_t>& get_mip_parameters() { return mip_; }
  const mip_parameters_t<i_t, f_t>& get_mip_parameters() const { return mip_; }

  /** LP only. Left unpopulated for a cold start. */
  cpu_pdlp_warm_start_data_t<i_t, f_t>& get_cpu_pdlp_warm_start_data() { return pdlp_warm_start_; }
  const cpu_pdlp_warm_start_data_t<i_t, f_t>& get_cpu_pdlp_warm_start_data() const
  {
    return pdlp_warm_start_;
  }

 private:
  pdlp_parameters_t<i_t, f_t> pdlp_;
  mip_parameters_t<i_t, f_t> mip_;
  cpu_pdlp_warm_start_data_t<i_t, f_t> pdlp_warm_start_;
  solver_parameter_registry_t<i_t, f_t> registry_;  // after pdlp_ and mip_, which it points into
};

}  // namespace mathematical_optimization
}  // namespace cuopt
