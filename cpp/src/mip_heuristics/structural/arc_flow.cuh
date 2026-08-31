/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/structural/early_structural.cuh>

namespace cuopt::mathematical_optimization::mip {

// Constructs paths through labelled arc-flow DAGs. Arc costs must be affine in tail potential;
// path termination may use explicit loss arcs or conservation-row slack.
template <typename i_t, typename f_t>
class arc_flow_t : public structural_heuristic_t<i_t, f_t> {
 public:
  arc_flow_t();
  ~arc_flow_t() override;

  const char* name() const override { return "ArcFlowDP"; }

  bool recognize(const optimization_problem_t<i_t, f_t>& op_problem,
                 const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances) override;
  bool recognize(const problem_t<i_t, f_t>& problem,
                 const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances) override;

  bool solve(
    const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
    std::atomic<bool>& preemption,
    std::vector<f_t>& assignment) override;

 private:
  struct host_state_t;

  std::unique_ptr<host_state_t> state_;
};

}  // namespace cuopt::mathematical_optimization::mip
