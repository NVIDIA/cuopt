/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "early_cpufj.cuh"

#include <mip_heuristics/mip_constants.hpp>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
early_cpufj_t<i_t, f_t>::early_cpufj_t(
  const optimization_problem_t<i_t, f_t>& op_problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  early_incumbent_callback_t<f_t> incumbent_callback,
  std::atomic<bool>* cancel_requested)
  : early_heuristic_t<i_t, f_t, early_cpufj_t<i_t, f_t>>(
      op_problem, tolerances, std::move(incumbent_callback)),
    cancel_requested_(cancel_requested)
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
  if (fj_cpu_ || omp_get_num_threads() < CUOPT_MIP_EARLY_CPUFJ_REQUIRED_THREAD_COUNT) { return; }

  this->preemption_flag_.store(false);
  this->start_time_ = std::chrono::steady_clock::now();

  // Prefer the gRPC cancel flag as the climber's preemption signal so cancel
  // aborts early CPUFJ during Papilo without waiting for stop()/taskwait.
  std::atomic<bool>& preempt_ref =
    (cancel_requested_ != nullptr) ? *cancel_requested_ : preemption_flag_;

  fj_cpu_ = init_fj_cpu_standalone(*this->problem_ptr_, *this->solution_ptr_, preempt_ref);

  fj_cpu_->log_prefix = "[Early CPUFJ] ";

  fj_cpu_->improvement_callback = [this](f_t solver_obj,
                                         const std::vector<f_t>& assignment,
                                         double) { this->try_update_best(solver_obj, assignment); };

  CUOPT_LOG_DEBUG("Launching early CPUFJ task");
#pragma omp task shared(fj_cpu_) priority(CUOPT_DEFAULT_TASK_PRIORITY) \
  depend(out : *fj_cpu_) default(none)
  cpufj_solve(fj_cpu_.get());
}

template <typename i_t, typename f_t>
void early_cpufj_t<i_t, f_t>::stop()
{
  if (!fj_cpu_) { return; }

  preemption_flag_.store(true);
  // The climber exits when either its preempt atomic or halted is true. If
  // cancel_requested_ was wired as that preempt atomic, writing
  // preemption_flag_ here does not affect the climber; fj_cpu_->halted below
  // still stops it on the normal (non-cancel) path. On cancel,
  // *cancel_requested_ is already true so the climber is already exiting.

  fj_cpu_->halted = true;
#pragma omp taskwait depend(in : *fj_cpu_)  // Wait for the early CPUFJ task to finish

  CUOPT_LOG_DEBUG("[Early CPUFJ] Stopped after %d iterations, solution_found=%d",
                  fj_cpu_ ? fj_cpu_->iterations : 0,
                  this->solution_found_);

  fj_cpu_.reset();
}

#if MIP_INSTANTIATE_FLOAT
template class early_cpufj_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class early_cpufj_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
