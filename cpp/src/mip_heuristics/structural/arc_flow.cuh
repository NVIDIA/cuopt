/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/structural/early_structural.cuh>

namespace cuopt::mathematical_optimization::mip {

// Constructive primal heuristic for enhanced arc-flow models: m source-to-sink paths through a
// DAG whose internal arcs each carry one covering-row label, in the Valerio de Carvalho arc-flow
// lineage.  It consumes the covering demands as an ordered token sequence and solves the resulting
// frontier dynamic program, which is optimal over that sequence unless the reconstruction history
// exceeds its budget and forces a beam.
//
// The recognized family is narrower than arc-flow in general.  The node potential and the token
// order are both recovered from the objective, so an arc's cost must be affine in the potential of
// its tail, and a negative fitted slope is rejected.  Fitting a slope needs two arcs at distinct
// potentials: a label that has fewer is placed by reachability instead, which is a position its arc
// set determines but not the one weighted shortest processing time would give, so the sequence
// searched is no longer the Smith-ordered one.  Path termination is recognized in either of its
// encodings, an explicit unlabelled arc or slack in the node's conservation row, which is what lets
// the same detector run before and after a presolve pass that substitutes bounded singleton columns
// out of their equality.  Other presolve reductions, notably row aggregation and coefficient
// strengthening, destroy the pattern and are not handled.
//
// Detection reads only permutation-invariant data: row bounds, coefficient patterns, the
// right-hand side, and statistics derived from the arc set.  Row order, column order and names are
// never consulted, so detection and the published objective are invariant under a permutation of
// the model.  The selected support is not, and cannot be where the model has automorphisms.
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

  structural_outcome_t solve(
    const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
    std::atomic<bool>& preemption,
    double work_budget,
    std::vector<f_t>& assignment) override;

  // True only when the constructed point is the optimum of the Smith-ordered family: false if the
  // history budget forced a beam, and false if any label had to be ordered by reachability because
  // its slope could not be fitted.
  bool search_was_exact() const { return search_was_exact_; }

 private:
  // Host mirror recovered by recognize(), consumed by solve().  Defined in the source, since its
  // shape is the detector's business alone.
  struct host_state_t;

  std::unique_ptr<host_state_t> state_;
  bool search_was_exact_{true};
};

}  // namespace cuopt::mathematical_optimization::mip
