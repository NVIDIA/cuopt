/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/early_heuristic.cuh>
#include <mip_heuristics/feasibility_jump/fj_cpu.cuh>

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
class early_cpufj_t : public early_heuristic_t<i_t, f_t, early_cpufj_t<i_t, f_t>> {
 public:
  early_cpufj_t(const optimization_problem_t<i_t, f_t>& op_problem,
                const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
                early_incumbent_callback_t<f_t> incumbent_callback);

  ~early_cpufj_t();

  static constexpr const char* name() { return "CPUFJ"; }

  // Lanes are OMP tasks that never yield, so n_lanes threads are unavailable to anything else
  // until stop(). Callers sharing the team with other work size it accordingly.
  void start(int n_lanes);
  void stop();

  int lane_count() const { return (int)climbers_.size(); }

 private:
  friend class early_heuristic_t<i_t, f_t, early_cpufj_t<i_t, f_t>>;

  std::vector<f_t> to_user_assignment(const std::vector<f_t>& assignment);

  const optimization_problem_t<i_t, f_t>* problem_ptr_;
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances_;
  std::vector<std::unique_ptr<fj_cpu_climber_t<i_t, f_t>>> climbers_;
  std::atomic<bool> preemption_flag_{false};
  // try_update_best and the incumbent callback behind it are not thread-safe, and every lane
  // reports into them from its own task.
  std::mutex incumbent_mutex_;
};

}  // namespace cuopt::mathematical_optimization::mip
