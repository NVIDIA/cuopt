/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "early_structural.cuh"

#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/structural/arc_flow.cuh>

#include <utilities/copy_helpers.hpp>
#include <utilities/macros.cuh>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

namespace {

// Recomputed from the solver-space problem rather than from whatever structure a heuristic thinks
// it found, so a detection mistake cannot publish an infeasible point.  Feasibility is decided by
// the solver's own tolerances: the question is whether the solver would accept this point.
template <typename i_t, typename f_t>
bool validate(const problem_t<i_t, f_t>& problem,
              const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
              const std::vector<f_t>& assignment,
              f_t& objective)
{
  auto stream = problem.handle_ptr->get_stream();

  const auto csr_values  = cuopt::host_copy(problem.coefficients, stream);
  const auto csr_cols    = cuopt::host_copy(problem.variables, stream);
  const auto csr_offsets = cuopt::host_copy(problem.offsets, stream);
  const auto row_lb      = cuopt::host_copy(problem.constraint_lower_bounds, stream);
  const auto row_ub      = cuopt::host_copy(problem.constraint_upper_bounds, stream);
  const auto obj         = cuopt::host_copy(problem.objective_coefficients, stream);
  const auto var_types   = cuopt::host_copy(problem.variable_types, stream);
  const auto [var_lb, var_ub] =
    cuopt::extract_host_bounds<f_t>(problem.variable_bounds, problem.handle_ptr);

  if ((i_t)assignment.size() != problem.n_variables) { return false; }

  const double integrality = tolerances.integrality_tolerance;
  const double abs_tol     = tolerances.absolute_tolerance;
  const double rel_tol     = tolerances.relative_tolerance;

  double obj_value = 0.0;
  for (i_t j = 0; j < problem.n_variables; ++j) {
    const double x = assignment[j];
    if (var_types[j] == var_t::INTEGER && std::abs(x - std::round(x)) > integrality) {
      return false;
    }
    if (x < (double)var_lb[j] - abs_tol || x > (double)var_ub[j] + abs_tol) { return false; }
    obj_value += obj[j] * x;
  }

  for (i_t r = 0; r < problem.n_constraints; ++r) {
    double activity = 0.0;
    for (i_t k = csr_offsets[r]; k < csr_offsets[r + 1]; ++k) {
      activity += (double)csr_values[k] * (double)assignment[csr_cols[k]];
    }
    const double slack = abs_tol + rel_tol * std::max(1.0, std::abs(activity));
    const double lo    = row_lb[r];
    const double hi    = row_ub[r];
    if (std::isfinite(lo) && activity < lo - slack) { return false; }
    if (std::isfinite(hi) && activity > hi + slack) { return false; }
  }

  objective = obj_value;
  return true;
}

}  // namespace

template <typename i_t, typename f_t>
early_structural_t<i_t, f_t>::early_structural_t(
  const optimization_problem_t<i_t, f_t>& op_problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  early_incumbent_callback_t<f_t> incumbent_callback)
  : early_heuristic_t<i_t, f_t, early_structural_t<i_t, f_t>>(std::move(incumbent_callback)),
    op_problem_(op_problem),
    tolerances_(tolerances)
{
  auto arc_flow = std::make_unique<arc_flow_t<i_t, f_t>>();
  if (arc_flow->recognize(op_problem, tolerances)) { active_ = std::move(arc_flow); }

  // The framework's problem copy is the expensive part of construction, so it is built only once a
  // structure has been recognized.
  if (active_) {
    active_->set_lane_budget(omp_get_num_threads() - 1);
    CUOPT_LOG_DEBUG("[Early Structural] %s recognized the model", active_->name());
    this->initialize_problem(op_problem, tolerances);
  }
}

template <typename i_t, typename f_t>
early_structural_t<i_t, f_t>::~early_structural_t()
{
  stop();
}

template <typename i_t, typename f_t>
const char* early_structural_t<i_t, f_t>::recognized_name() const
{
  return active_ ? active_->name() : nullptr;
}

template <typename i_t, typename f_t>
void early_structural_t<i_t, f_t>::start()
{
  if (!active_ || task_launched_ ||
      omp_get_num_threads() < CUOPT_MIP_EARLY_STRUCTURAL_REQUIRED_THREAD_COUNT) {
    return;
  }

  preemption_flag_.store(false);
  this->start_time_ = std::chrono::steady_clock::now();
  task_launched_    = true;

  // A data member is not a valid depend list item, so the dependence is named through a
  // dereferenced pointer to it.  stop() names the same storage, which is what pairs the taskwait
  // with this task.
  auto* task_token = &preemption_flag_;
  CUOPT_LOG_DEBUG("Launching early structural task for %s", active_->name());
#pragma omp task priority(CUOPT_DEFAULT_TASK_PRIORITY) depend(out : *task_token)
  this->run();
}

template <typename i_t, typename f_t>
void early_structural_t<i_t, f_t>::stop()
{
  if (!task_launched_) { return; }

  auto* task_token = &preemption_flag_;
  preemption_flag_.store(true);
#pragma omp taskwait depend(in : *task_token)
  task_launched_ = false;

  CUOPT_LOG_DEBUG("[Early Structural] Stopped, solution_found=%d", (int)this->solution_found_);
}

// An assignment built in op_problem space can only be published through post_process_assignment if
// preprocessing left the columns where they were: same count, no shifted lower bound, no split.
template <typename i_t, typename f_t>
bool early_structural_t<i_t, f_t>::preprocessing_is_identity() const
{
  const auto& presolve_data = this->problem_ptr_->presolve_data;
  if (this->problem_ptr_->n_variables != op_problem_.get_n_variables()) { return false; }
  if ((i_t)presolve_data.variable_offsets.size() != this->problem_ptr_->n_variables) {
    return false;
  }
  for (const f_t offset : presolve_data.variable_offsets) {
    if (offset != f_t{0}) { return false; }
  }
  for (size_t j = 0; j < presolve_data.additional_var_used.size(); ++j) {
    if (presolve_data.additional_var_used[j]) { return false; }
  }
  return true;
}

template <typename i_t, typename f_t>
void early_structural_t<i_t, f_t>::run()
{
  cuopt_assert(active_ != nullptr, "task launched without a recognized structure");

  std::vector<f_t> assignment;
  const structural_outcome_t outcome =
    active_->solve(tolerances_, preemption_flag_, 0.0, assignment);
  if (outcome != structural_outcome_t::constructed) {
    CUOPT_LOG_DEBUG("[Early Structural] %s constructed nothing", active_->name());
    return;
  }
  if (preemption_flag_.load()) { return; }

  // Checked before validation rather than after: without column identity the assignment cannot be
  // read in solver space at all, which is the space both the validator and the publication use.
  if (!preprocessing_is_identity()) {
    CUOPT_LOG_DEBUG(
      "[Early Structural] %s constructed a point but preprocessing moved the columns, discarding",
      active_->name());
    return;
  }

  f_t objective{0};
  if (!validate(*this->problem_ptr_, tolerances_, assignment, objective)) {
    CUOPT_LOG_DEBUG("[Early Structural] %s constructed a point that failed validation, discarding",
                    active_->name());
    return;
  }
  this->try_update_best(objective, assignment, active_->name());
}

template <typename i_t, typename f_t>
root_structural_t<i_t, f_t>::root_structural_t(
  problem_t<i_t, f_t>& problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  std::atomic<bool>& preemption,
  const std::vector<std::vector<i_t>>* column_symmetry,
  int lane_budget)
  : problem_(problem), tolerances_(tolerances), preemption_(preemption)
{
  auto arc_flow = std::make_unique<arc_flow_t<i_t, f_t>>();
  if (arc_flow->recognize(problem, tolerances)) { active_ = std::move(arc_flow); }
  if (active_) {
    active_->set_lane_budget(lane_budget);
    active_->set_column_symmetry(column_symmetry);
    CUOPT_LOG_DEBUG("[Root Structural] %s recognized the model, %d lanes, %zu symmetry generators",
                    active_->name(),
                    lane_budget,
                    column_symmetry == nullptr ? size_t{0} : column_symmetry->size());
  }
}

template <typename i_t, typename f_t>
root_structural_t<i_t, f_t>::~root_structural_t() = default;

template <typename i_t, typename f_t>
const char* root_structural_t<i_t, f_t>::recognized_name() const
{
  return active_ ? active_->name() : nullptr;
}

template <typename i_t, typename f_t>
void root_structural_t<i_t, f_t>::run()
{
  if (!active_) { return; }
  if (!problem_.branch_and_bound_callback) {
    CUOPT_LOG_DEBUG("[Root Structural] no branch and bound to publish to, skipping");
    return;
  }

  // The recognizer read this problem, so the point comes back in the space B&B branches in and
  // needs no mapping.  It is still validated: a detection mistake must not reach the tree.
  std::vector<f_t> assignment;
  const structural_outcome_t outcome = active_->solve(tolerances_, preemption_, 0.0, assignment);
  if (outcome != structural_outcome_t::constructed) {
    CUOPT_LOG_DEBUG("[Root Structural] %s constructed nothing", active_->name());
    return;
  }
  if (preemption_.load()) { return; }

  f_t objective{0};
  if (!validate(problem_, tolerances_, assignment, objective)) {
    CUOPT_LOG_DEBUG("[Root Structural] %s constructed a point that failed validation, discarding",
                    active_->name());
    return;
  }

  const bool accepted =
    problem_.branch_and_bound_callback(assignment, heuristics_origin_t::HEURISTICS);
  CUOPT_LOG_DEBUG("[Root Structural] %s published objective %+.6e, accepted=%d",
                  active_->name(),
                  (double)problem_.get_user_obj_from_solver_obj(objective),
                  (int)accepted);
}

#if MIP_INSTANTIATE_FLOAT
template class early_structural_t<int, float>;
template class root_structural_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class early_structural_t<int, double>;
template class root_structural_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
