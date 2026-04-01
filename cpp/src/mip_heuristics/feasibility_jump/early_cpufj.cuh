/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/early_heuristic.cuh>

#include <atomic>
#include <memory>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct cpu_fj_thread_t;

template <typename i_t, typename f_t>
class early_cpufj_t : public early_heuristic_t<i_t, f_t, early_cpufj_t<i_t, f_t>> {
 public:
  early_cpufj_t(const optimization_problem_t<i_t, f_t>& op_problem,
                const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
                early_incumbent_callback_t<f_t> incumbent_callback);

  ~early_cpufj_t();

  static constexpr const char* name() { return "CPUFJ"; }

  void start();
  void stop();

 private:
  std::unique_ptr<cpu_fj_thread_t<i_t, f_t>> cpu_fj_thread_;
  std::atomic<bool> preemption_flag_{false};
};

}  // namespace cuopt::linear_programming::detail
