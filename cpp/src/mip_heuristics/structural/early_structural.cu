/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "early_structural.cuh"

#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/structural/arc_flow.cuh>
#include <mip_heuristics/utils.cuh>

#include <utilities/copy_helpers.hpp>
#include <utilities/macros.cuh>

#include <omp.h>

#include <cmath>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
static bool validate(const problem_t<i_t, f_t>& problem,
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

  const double integrality = problem.tolerances.integrality_tolerance;

  double obj_value = 0.0;
  for (i_t j = 0; j < problem.n_variables; ++j) {
    const double x = assignment[j];
    if (!std::isfinite(x)) { return false; }
    if (var_types[j] == var_t::INTEGER && std::abs(x - std::round(x)) > integrality) {
      return false;
    }
    if (x < (double)var_lb[j] - integrality || x > (double)var_ub[j] + integrality) {
      return false;
    }
    obj_value += obj[j] * x;
  }
  if (!std::isfinite(obj_value)) { return false; }

  for (i_t r = 0; r < problem.n_constraints; ++r) {
    double activity = 0.0;
    for (i_t k = csr_offsets[r]; k < csr_offsets[r + 1]; ++k) {
      activity += (double)csr_values[k] * (double)assignment[csr_cols[k]];
    }
    if (!std::isfinite(activity)) { return false; }
    const double lo    = row_lb[r];
    const double hi    = row_ub[r];
    const double slack = get_cstr_tolerance<i_t, double>(lo,
                                                         hi,
                                                         problem.tolerances.absolute_tolerance,
                                                         problem.tolerances.relative_tolerance);
    if (std::isfinite(lo) && activity < lo - slack) { return false; }
    if (std::isfinite(hi) && activity > hi + slack) { return false; }
  }

  objective = obj_value;
  return true;
}

template <typename i_t, typename f_t>
std::unique_ptr<early_structural_t<i_t, f_t>> early_structural_t<i_t, f_t>::create(
  const optimization_problem_t<i_t, f_t>& op_problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  early_incumbent_callback_t<f_t> incumbent_callback)
{
  if (omp_get_num_threads() < CUOPT_MIP_EARLY_STRUCTURAL_REQUIRED_THREAD_COUNT) { return nullptr; }
  auto active = std::make_unique<arc_flow_t<i_t, f_t>>();
  if (!active->recognize(op_problem, tolerances)) { return nullptr; }
  return std::unique_ptr<early_structural_t>(
    new early_structural_t(op_problem, tolerances, std::move(incumbent_callback), std::move(active)));
}

template <typename i_t, typename f_t>
early_structural_t<i_t, f_t>::early_structural_t(
  const optimization_problem_t<i_t, f_t>& op_problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  early_incumbent_callback_t<f_t> incumbent_callback,
  std::unique_ptr<structural_heuristic_t<i_t, f_t>> active)
  : early_heuristic_t<i_t, f_t, early_structural_t<i_t, f_t>>(
      op_problem, tolerances, std::move(incumbent_callback)),
    op_problem_(op_problem),
    tolerances_(tolerances),
    active_(std::move(active))
{
  cuopt_assert(active_ != nullptr, "missing structural heuristic");
  CUOPT_LOG_DEBUG("[Early Structural] %s recognized the model", active_->name());
}

template <typename i_t, typename f_t>
early_structural_t<i_t, f_t>::~early_structural_t()
{
  stop();
}

template <typename i_t, typename f_t>
void early_structural_t<i_t, f_t>::start()
{
  if (task_launched_) { return; }

  preemption_flag_.store(false);
  this->start_time_ = std::chrono::steady_clock::now();
  task_launched_    = true;

  // OpenMP depend clauses require a variable or array element.
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

template <typename i_t, typename f_t>
bool early_structural_t<i_t, f_t>::preprocessing_is_identity() const
{
  // Recognition produces assignments in the source problem's column space.
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
  const structural_outcome_t outcome = active_->solve(tolerances_, preemption_flag_, assignment);
  if (outcome != structural_outcome_t::constructed) {
    CUOPT_LOG_DEBUG("[Early Structural] %s constructed nothing", active_->name());
    return;
  }
  if (preemption_flag_.load()) { return; }

  if (!preprocessing_is_identity()) {
    CUOPT_LOG_DEBUG(
      "[Early Structural] %s constructed a point but preprocessing moved the columns, discarding",
      active_->name());
    return;
  }

  f_t objective{0};
  if (!validate(*this->problem_ptr_, assignment, objective)) {
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
  structural_incumbent_callback_t<f_t> incumbent_callback)
  : problem_(problem),
    tolerances_(tolerances),
    preemption_(preemption),
    incumbent_callback_(std::move(incumbent_callback))
{
  auto arc_flow = std::make_unique<arc_flow_t<i_t, f_t>>();
  if (arc_flow->recognize(problem, tolerances)) { active_ = std::move(arc_flow); }
  if (active_) { CUOPT_LOG_DEBUG("[Root Structural] %s recognized the model", active_->name()); }
}

template <typename i_t, typename f_t>
root_structural_t<i_t, f_t>::~root_structural_t() = default;

template <typename i_t, typename f_t>
void root_structural_t<i_t, f_t>::run()
{
  if (!active_) { return; }
  cuopt_assert(incumbent_callback_ != nullptr, "missing incumbent callback");

  std::vector<f_t> assignment;
  const structural_outcome_t outcome = active_->solve(tolerances_, preemption_, assignment);
  if (outcome != structural_outcome_t::constructed) {
    CUOPT_LOG_DEBUG("[Root Structural] %s constructed nothing", active_->name());
    return;
  }
  if (preemption_.load()) { return; }

  f_t objective{0};
  if (!validate(problem_, assignment, objective)) {
    CUOPT_LOG_DEBUG("[Root Structural] %s constructed a point that failed validation, discarding",
                    active_->name());
    return;
  }
  if (preemption_.load()) { return; }

  incumbent_callback_(assignment, objective);
  CUOPT_LOG_DEBUG("[Root Structural] %s queued objective %+.6e",
                  active_->name(),
                  (double)problem_.get_user_obj_from_solver_obj(objective));
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
