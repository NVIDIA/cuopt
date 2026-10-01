/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Stable external include. CPU implementation details and state ownership live below cpu/.
#include <mip_heuristics/feasibility_jump/cpu/state.hpp>

namespace cuopt::mathematical_optimization {
template <typename i_t, typename f_t>
class optimization_problem_t;
}

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
class problem_t;

// CUDA-facing adapters. Their definitions own all interaction with the GPU-backed problem_t;
// cpu/engine.cpp only sees the resulting host climber state.
template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> init_fj_cpu_standalone(
  problem_t<i_t, f_t>& problem,
  std::atomic<bool>& preemption_flag,
  uint64_t seed,
  fj_settings_t settings = fj_settings_t{});

template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> init_fj_cpu_from_optimization_problem(
  const optimization_problem_t<i_t, f_t>& problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings = fj_settings_t{});

template <typename i_t, typename f_t>
void build_climber_portfolio(problem_t<i_t, f_t>& problem,
                             std::vector<std::atomic<bool>>& preemption_flags,
                             std::vector<std::unique_ptr<fj_cpu_climber_t<i_t, f_t>>>& climbers,
                             int64_t base_seed,
                             bool low_latency = false);

}  // namespace cuopt::mathematical_optimization::mip
