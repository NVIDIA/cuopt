/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/early_heuristic.cuh>

#include <atomic>
#include <functional>
#include <memory>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
class structural_heuristic_t {
 public:
  virtual ~structural_heuristic_t() = default;

  virtual const char* name() const = 0;

  virtual bool recognize(
    const optimization_problem_t<i_t, f_t>& op_problem,
    const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances) = 0;

  virtual bool recognize(const problem_t<i_t, f_t>&,
                         const typename mip_solver_settings_t<i_t, f_t>::tolerances_t&)
  {
    return false;
  }

  virtual bool solve(
    const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
    std::atomic<bool>& preemption,
    std::vector<f_t>& assignment) = 0;
};

template <typename i_t, typename f_t>
class early_structural_t : public early_heuristic_t<i_t, f_t, early_structural_t<i_t, f_t>> {
 public:
  static std::unique_ptr<early_structural_t> create(
    const optimization_problem_t<i_t, f_t>& op_problem,
    const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
    early_incumbent_callback_t<f_t> incumbent_callback);

  ~early_structural_t();

  static constexpr const char* name() { return "Structural"; }

  void start();
  void stop();

 private:
  early_structural_t(const optimization_problem_t<i_t, f_t>& op_problem,
                     const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
                     early_incumbent_callback_t<f_t> incumbent_callback,
                     std::unique_ptr<structural_heuristic_t<i_t, f_t>> active);

  void run();

  bool preprocessing_is_identity() const;

  const optimization_problem_t<i_t, f_t>& op_problem_;
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances_;
  std::unique_ptr<structural_heuristic_t<i_t, f_t>> active_;
  std::atomic<bool> preemption_flag_{false};
  bool task_launched_{false};
};

template <typename f_t>
using structural_incumbent_callback_t =
  std::function<void(const std::vector<f_t>& assignment, f_t objective)>;

template <typename i_t, typename f_t>
class root_structural_t {
 public:
  root_structural_t(problem_t<i_t, f_t>& problem,
                    const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
                    std::atomic<bool>& preemption,
                    structural_incumbent_callback_t<f_t> incumbent_callback);

  ~root_structural_t();

  bool recognized() const { return active_ != nullptr; }

  void run();

 private:
  int device_id_{0};
  raft::handle_t handle_;
  std::unique_ptr<problem_t<i_t, f_t>> problem_;
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances_;
  std::atomic<bool>& preemption_;
  structural_incumbent_callback_t<f_t> incumbent_callback_;
  std::unique_ptr<structural_heuristic_t<i_t, f_t>> active_;
};

}  // namespace cuopt::mathematical_optimization::mip
