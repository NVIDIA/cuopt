/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/constants.h>
#include <cuopt/export.hpp>
#include <cuopt/mathematical_optimization/mip/mip_parameters.hpp>
#include <cuopt/mathematical_optimization/pdlp/pdlp_parameters.hpp>
#include <cuopt/mathematical_optimization/utilities/internals.hpp>

#include <string>
#include <vector>

namespace cuopt {
namespace CUOPT_EXPORT mathematical_optimization {

/**
 * @brief The solver parameters by name: the one table of names, defaults and ranges.
 *
 * Binds each CUOPT_* name to a member of a pdlp_parameters_t / mip_parameters_t and implements
 * set/get by name, parameter files and listing. solver_settings_t and cpu_solver_settings_t both
 * hold one, so the two cannot disagree on which parameters exist. Needs no CUDA, rmm or raft.
 *
 * The registry points into the two parameter objects it was built on, so it is neither copyable
 * nor movable and must not outlive them.
 */
template <typename i_t, typename f_t>
class solver_parameter_registry_t {
 public:
  solver_parameter_registry_t(pdlp_parameters_t<i_t, f_t>& pdlp, mip_parameters_t<i_t, f_t>& mip);

  solver_parameter_registry_t(const solver_parameter_registry_t&)            = delete;
  solver_parameter_registry_t& operator=(const solver_parameter_registry_t&) = delete;
  solver_parameter_registry_t(solver_parameter_registry_t&&)                 = delete;
  solver_parameter_registry_t& operator=(solver_parameter_registry_t&&)      = delete;

  void set_parameter_from_string(const std::string& name, const std::string& value);

  template <typename T>
  void set_parameter(const std::string& name, T value);

  template <typename T>
  T get_parameter(const std::string& name) const;

  std::string get_parameter_as_string(const std::string& name) const;

  const std::vector<parameter_info_t<f_t>>& get_float_parameters() const;
  const std::vector<parameter_info_t<i_t>>& get_int_parameters() const;
  const std::vector<parameter_info_t<bool>>& get_bool_parameters() const;
  const std::vector<parameter_info_t<std::string>>& get_string_parameters() const;
  const std::vector<std::string> get_parameter_names() const;

  void load_parameters_from_file(const std::string& path);
  bool dump_parameters_to_file(const std::string& path, bool hyperparameters_only = true) const;

 private:
  std::vector<parameter_info_t<f_t>> float_parameters;
  std::vector<parameter_info_t<i_t>> int_parameters;
  std::vector<parameter_info_t<bool>> bool_parameters;
  std::vector<parameter_info_t<std::string>> string_parameters;
};

}  // namespace CUOPT_EXPORT mathematical_optimization
}  // namespace cuopt
