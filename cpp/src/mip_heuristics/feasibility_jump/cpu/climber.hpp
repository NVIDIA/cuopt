/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "state.hpp"

namespace cuopt::mathematical_optimization::simplex {

template <typename i_t, typename f_t>
struct lp_problem_t;

template <typename i_t, typename f_t>
struct simplex_solver_settings_t;

enum class variable_type_t : int8_t;

}  // namespace cuopt::mathematical_optimization::simplex

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> init_fj_cpu_from_host_lp(
  const simplex::lp_problem_t<i_t, f_t>& problem,
  const std::vector<simplex::variable_type_t>& variable_types,
  i_t n_structural,
  const std::vector<f_t>& start_assignment,
  const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
  std::atomic<bool>& preemption_flag,
  int64_t seed);

template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> init_fj_cpu_from_host_model(
  i_t n_variables,
  i_t n_constraints,
  i_t nnz,
  bool maximize,
  f_t objective_scaling_factor,
  f_t objective_offset,
  std::vector<f_t> coefficients,
  std::vector<i_t> variables,
  std::vector<i_t> offsets,
  std::vector<f_t> objective_coefficients,
  std::vector<f_t> variable_lower_bounds,
  std::vector<f_t> variable_upper_bounds,
  std::vector<f_t> constraint_lower_bounds,
  std::vector<f_t> constraint_upper_bounds,
  std::vector<f_t> constraint_bounds,
  std::vector<char> row_types,
  std::vector<var_t> variable_types,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings);

template <typename i_t, typename f_t>
void finalize_fj_cpu_host_initialization(
  fj_cpu_climber_t<i_t, f_t>&,
  fj_cpu_problem_t<i_t, f_t>&,
  i_t,
  i_t,
  i_t,
  i_t,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t&);
}  // namespace cuopt::mathematical_optimization::mip
