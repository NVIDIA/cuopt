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

#include <utilities/macros.cuh>

#include <omp.h>

#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
static bool validate(problem_t<i_t, f_t>& problem,
                     const std::vector<f_t>& assignment,
                     f_t& objective)
{
  if ((i_t)assignment.size() != problem.n_variables) { return false; }
  solution_t<i_t, f_t> solution(problem);
  solution.copy_new_assignment(assignment);
  if (has_variable_bounds_violation(problem.handle_ptr, solution.assignment, &problem) ||
      !solution.compute_feasibility()) {
    return false;
  }
  objective = solution.get_objective();
  return true;
}

template <typename i_t, typename f_t, typename model_t>
static std::unique_ptr<structural_heuristic_t<i_t, f_t>> make_structural_heuristic(
  const model_t& model, const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances)
{
  auto heuristic = std::make_unique<arc_flow_t<i_t, f_t>>();
  if (!heuristic->recognize(model, tolerances)) { return nullptr; }
  return heuristic;
}

template <typename i_t, typename f_t>
std::unique_ptr<early_structural_t<i_t, f_t>> early_structural_t<i_t, f_t>::create(
  const optimization_problem_t<i_t, f_t>& op_problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  early_incumbent_callback_t<f_t> incumbent_callback)
{
  if (omp_get_num_threads() < CUOPT_MIP_EARLY_STRUCTURAL_REQUIRED_THREAD_COUNT) { return nullptr; }
  auto active = make_structural_heuristic<i_t, f_t>(op_problem, tolerances);
  if (!active) { return nullptr; }
  return std::unique_ptr<early_structural_t>(new early_structural_t(
    op_problem, tolerances, std::move(incumbent_callback), std::move(active)));
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
  if (!active_->solve(tolerances_, preemption_flag_, assignment)) {
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
  : tolerances_(tolerances),
    preemption_(preemption),
    incumbent_callback_(std::move(incumbent_callback))
{
  RAFT_CUDA_TRY(cudaGetDevice(&device_id_));
  active_ = make_structural_heuristic<i_t, f_t>(problem, tolerances);
  if (!active_) { return; }
  problem.handle_ptr->sync_stream();
  problem_ = std::make_unique<problem_t<i_t, f_t>>(problem, &handle_);
  CUOPT_LOG_DEBUG("[Root Structural] %s recognized the model", active_->name());
}

template <typename i_t, typename f_t>
root_structural_t<i_t, f_t>::~root_structural_t() = default;

template <typename i_t, typename f_t>
void root_structural_t<i_t, f_t>::run()
{
  if (!active_) { return; }
  cuopt_assert(problem_ != nullptr, "missing structural problem");
  cuopt_assert(incumbent_callback_ != nullptr, "missing incumbent callback");

  std::vector<f_t> assignment;
  if (!active_->solve(tolerances_, preemption_, assignment)) {
    CUOPT_LOG_DEBUG("[Root Structural] %s constructed nothing", active_->name());
    return;
  }
  if (preemption_.load()) { return; }

  RAFT_CUDA_TRY(cudaSetDevice(device_id_));
  f_t objective{0};
  if (!validate(*problem_, assignment, objective)) {
    CUOPT_LOG_DEBUG("[Root Structural] %s constructed a point that failed validation, discarding",
                    active_->name());
    return;
  }
  if (preemption_.load()) { return; }

  incumbent_callback_(assignment, objective);
  CUOPT_LOG_DEBUG("[Root Structural] %s queued objective %+.6e",
                  active_->name(),
                  (double)problem_->get_user_obj_from_solver_obj(objective));
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
