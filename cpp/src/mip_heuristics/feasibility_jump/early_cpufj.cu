/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "early_cpufj.cuh"

#include <mip_heuristics/mip_constants.hpp>

#include <omp.h>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
early_cpufj_t<i_t, f_t>::early_cpufj_t(
  const optimization_problem_t<i_t, f_t>& op_problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  early_incumbent_callback_t<f_t> incumbent_callback,
  uint64_t seed)
  : early_heuristic_t<i_t, f_t, early_cpufj_t<i_t, f_t>>(op_problem, std::move(incumbent_callback)),
    problem_ptr_(&op_problem),
    tolerances_(tolerances),
    seed_(seed)
{
}

template <typename i_t, typename f_t>
early_cpufj_t<i_t, f_t>::~early_cpufj_t()
{
  stop();
}

template <typename i_t, typename f_t>
void early_cpufj_t<i_t, f_t>::start(bool low_latency)
{
  const bool threaded = !omp_in_parallel();
  // 1: presolve, 1: early GPU FJ, 1: early CPU FJ
  if (climber_ ||
      (!threaded && omp_get_num_threads() < CUOPT_MIP_EARLY_CPUFJ_REQUIRED_THREAD_COUNT)) {
    return;
  }

  this->preemption_flag_.store(false);
  this->start_time_ = std::chrono::steady_clock::now();

  auto report_incumbent = [this](f_t solver_obj, const std::vector<f_t>& assignment, double) {
    this->try_update_best(solver_obj, assignment);
  };

  fj_settings_t settings;
  settings.seed = (int)seed_;
  climber_      = init_fj_cpu_from_optimization_problem(
    *this->problem_ptr_, tolerances_, preemption_flag_, settings);
  climber_->low_latency          = low_latency;
  climber_->log_prefix           = "[Early CPUFJ] ";
  climber_->improvement_callback = report_incumbent;

  CUOPT_LOG_DEBUG("Launching early CPUFJ %s", threaded ? "thread" : "task");
  auto* climber = climber_.get();
  if (threaded) {
    worker_ = std::thread([climber] { cpufj_solve(climber); });
    return;
  }
#pragma omp task firstprivate(climber) priority(CUOPT_DEFAULT_TASK_PRIORITY) \
  depend(out : *climber) default(none)
  cpufj_solve(climber);
}

template <typename i_t, typename f_t>
void early_cpufj_t<i_t, f_t>::stop()
{
  if (!climber_) { return; }

  preemption_flag_.store(true);
  climber_->halted = true;
  if (worker_.joinable()) {
    worker_.join();
  } else {
#pragma omp taskwait depend(in : *climber_)  // Wait for the early CPUFJ task to finish
  }

  CUOPT_LOG_DEBUG("[Early CPUFJ] Stopped after %d iterations, solution_found=%d",
                  climber_->iterations,
                  this->solution_found_);

  climber_.reset();
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
