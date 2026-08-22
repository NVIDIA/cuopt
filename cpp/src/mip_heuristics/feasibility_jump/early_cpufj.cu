/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "early_cpufj.cuh"

#include <mip_heuristics/mip_constants.hpp>
#include <utilities/seed_generator.cuh>

#include <algorithm>
#include <string>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
early_cpufj_t<i_t, f_t>::early_cpufj_t(
  const optimization_problem_t<i_t, f_t>& op_problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  early_incumbent_callback_t<f_t> incumbent_callback)
  : early_heuristic_t<i_t, f_t, early_cpufj_t<i_t, f_t>>(op_problem, std::move(incumbent_callback)),
    problem_ptr_(&op_problem),
    tolerances_(tolerances)
{
}

template <typename i_t, typename f_t>
early_cpufj_t<i_t, f_t>::~early_cpufj_t()
{
  stop();
}

template <typename i_t, typename f_t>
void early_cpufj_t<i_t, f_t>::start()
{
  // 1: presolve, 1: early GPU FJ, 1: early CPU FJ
  if (!climbers_.empty() || omp_get_num_threads() < CUOPT_MIP_EARLY_CPUFJ_REQUIRED_THREAD_COUNT) {
    return;
  }

  this->preemption_flag_.store(false);
  this->start_time_ = std::chrono::steady_clock::now();

  // Tasks are not preempted, so a lane posted beyond the team size would sit in the queue for the
  // whole of presolve without running an iteration.
  const int n_lanes       = std::min(CUOPT_MIP_EARLY_CPUFJ_MAX_CLIMBERS, omp_get_num_threads());
  const int64_t base_seed = cuopt::seed_generator::get_seed();
  climbers_.resize(n_lanes);

  auto report_incumbent = [this](f_t solver_obj, const std::vector<f_t>& assignment, double) {
    std::lock_guard<std::mutex> guard(incumbent_mutex_);
    this->try_update_best(solver_obj, assignment);
  };

  // Lane 0 builds the host problem representation and every other lane copies it. All of it
  // finishes before the first task is posted, so no lane reads a template another lane is running
  // on. seed_generator steps a non-atomic global, which is why the draws stay on this thread.
  for (int k = 0; k < n_lanes; ++k) {
    if (k == 0) {
      climbers_[0] =
        init_fj_cpu_from_optimization_problem(*this->problem_ptr_, tolerances_, preemption_flag_);
    } else {
      fj_settings_t settings;
      settings.seed = (int)cuopt::seed_generator::get_seed();
      climbers_[k]  = init_fj_cpu_clone(*climbers_[0], preemption_flag_, settings);
    }
    apply_lane_diversification<i_t, f_t>(*climbers_[k], k, base_seed);
    climbers_[k]->log_prefix           = "[Early CPUFJ " + std::to_string(k) + "] ";
    climbers_[k]->improvement_callback = report_incumbent;
  }

  auto shared = std::make_shared<fj_cpu_shared_incumbent_t<i_t, f_t>>();
  for (int k = 0; k < n_lanes; ++k)
    climbers_[k]->shared_incumbent = shared;

  CUOPT_LOG_DEBUG("Launching %d early CPUFJ tasks", n_lanes);
  for (int k = 0; k < n_lanes; ++k) {
    auto* climber = climbers_[k].get();
#pragma omp task firstprivate(climber) priority(CUOPT_DEFAULT_TASK_PRIORITY) \
  depend(out : *climber) default(none)
    cpufj_solve(climber);
  }
}

template <typename i_t, typename f_t>
void early_cpufj_t<i_t, f_t>::stop()
{
  if (climbers_.empty()) { return; }

  preemption_flag_.store(true);

  // Every lane is told to stop before any wait, otherwise the first wait blocks on a lane that has
  // not been asked to exit yet.
  for (auto& climber : climbers_) {
    climber->halted = true;
  }
  for (size_t k = 0; k < climbers_.size(); ++k) {
#pragma omp taskwait depend(in : *climbers_[k])  // Wait for each early CPUFJ task to finish
  }

  i_t total_iterations = 0;
  for (const auto& climber : climbers_) {
    total_iterations += climber->iterations;
  }

  CUOPT_LOG_DEBUG("[Early CPUFJ] Stopped after %d iterations over %d climbers, solution_found=%d",
                  total_iterations,
                  (int)climbers_.size(),
                  this->solution_found_);

  climbers_.clear();
}

template <typename i_t, typename f_t>
std::vector<f_t> early_cpufj_t<i_t, f_t>::to_user_assignment(const std::vector<f_t>& assignment)
{
  return assignment;
}

#if MIP_INSTANTIATE_FLOAT
template class early_cpufj_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class early_cpufj_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
