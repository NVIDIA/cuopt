/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>

#include <atomic>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
class problem_t;

template <typename i_t, typename f_t>
struct concurrent_root_solution_t {
  bool usable{false};
  bool optimal{false};
  std::vector<f_t> primal;
  std::vector<f_t> dual;
  std::vector<f_t> reduced_cost;
  f_t solver_objective{0};
  f_t user_objective{0};
  i_t iterations{0};
  method_t method{method_t::Unset};
};

// Shared PDLP/barrier settings for the MIP root LP. Callers that own the
// solve (B&B or the heuristics-only fallback) still apply time_limit and
// concurrent_halt for their own halt/timer.
template <typename i_t, typename f_t>
pdlp_solver_settings_t<i_t, f_t> make_mip_root_lp_settings(
  const mip_solver_settings_t<i_t, f_t>& mip_settings);

template <typename i_t, typename f_t>
concurrent_root_solution_t<i_t, f_t> solve_concurrent_root_relaxation(
  problem_t<i_t, f_t>* problem,
  const pdlp_solver_settings_t<i_t, f_t>& settings,
  f_t time_limit,
  std::atomic<int>* concurrent_halt);

}  // namespace cuopt::mathematical_optimization::mip
