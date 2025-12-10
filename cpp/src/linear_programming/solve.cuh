/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/optimization_problem.hpp>

#include <mps_parser/mps_data_model.hpp>

#include <raft/core/handle.hpp>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
cuopt::linear_programming::optimization_problem_t<i_t, f_t> mps_data_model_to_optimization_problem(
  raft::handle_t const* handle_ptr,
  const cuopt::mps_parser::mps_data_model_t<i_t, f_t>& data_model);

template <typename i_t, typename f_t>
cuopt::linear_programming::optimization_problem_solution_t<i_t, f_t> solve_lp_with_method(
  detail::problem_t<i_t, f_t>& problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  const timer_t& timer,
  bool is_batch_mode = false);

template <typename i_t, typename f_t>
void set_pdlp_solver_mode(pdlp_solver_settings_t<i_t, f_t> const& settings);

}  // namespace cuopt::linear_programming
