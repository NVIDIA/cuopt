/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/early_heuristic.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solution/solution.cuh>

#include <memory>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
class fj_t;

template <typename i_t, typename f_t>
struct mip_solver_context_t;

template <typename i_t, typename f_t>
class early_gpufj_t : public early_heuristic_t<i_t, f_t, early_gpufj_t<i_t, f_t>> {
 public:
  early_gpufj_t(const optimization_problem_t<i_t, f_t>& op_problem,
                const mip_solver_settings_t<i_t, f_t>& settings,
                early_incumbent_callback_t<f_t> incumbent_callback);

  ~early_gpufj_t();

  static constexpr const char* name() { return "GPUFJ"; }

  void start();
  void stop();

 private:
  friend class early_heuristic_t<i_t, f_t, early_gpufj_t<i_t, f_t>>;

  std::vector<f_t> to_user_assignment(const std::vector<f_t>& assignment);

  int device_id_{0};

  // handle_ must be declared before problem_ptr_/solution_ptr_ so it outlives them
  // (C++ destroys members in reverse declaration order)
  raft::handle_t handle_;

  std::unique_ptr<problem_t<i_t, f_t>> problem_ptr_;
  std::unique_ptr<solution_t<i_t, f_t>> solution_ptr_;
  std::unique_ptr<mip_solver_context_t<i_t, f_t>> context_ptr_;
  std::unique_ptr<fj_t<i_t, f_t>> fj_ptr_;
};

}  // namespace cuopt::mathematical_optimization::mip
