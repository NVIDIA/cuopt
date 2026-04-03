/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "early_cpufj.cuh"

#include <mip_heuristics/feasibility_jump/fj_cpu.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <utilities/logger.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
early_cpufj_t<i_t, f_t>::early_cpufj_t(
  const optimization_problem_t<i_t, f_t>& op_problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  early_incumbent_callback_t<f_t> incumbent_callback)
  : early_heuristic_t<i_t, f_t, early_cpufj_t<i_t, f_t>>(
      op_problem, tolerances, std::move(incumbent_callback))
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
  if (cpu_fj_thread_) { return; }

  this->preemption_flag_.store(false);
  this->start_time_ = std::chrono::steady_clock::now();

  cpu_fj_thread_ = std::make_unique<cpu_fj_thread_t<i_t, f_t>>();
  cpu_fj_thread_->fj_cpu =
    init_fj_cpu_standalone(*this->problem_ptr_, *this->solution_ptr_, preemption_flag_);
  cpu_fj_thread_->time_limit = std::numeric_limits<f_t>::infinity();

  cpu_fj_thread_->fj_cpu->log_prefix = "[Early CPUFJ] ";

  cpu_fj_thread_->fj_cpu->improvement_callback =
    [this](f_t solver_obj, const std::vector<f_t>& assignment, double) {
      this->try_update_best(solver_obj, assignment);
    };

  cpu_fj_thread_->start_cpu_solver();
}

template <typename i_t, typename f_t>
void early_cpufj_t<i_t, f_t>::stop()
{
  if (!cpu_fj_thread_) { return; }

  preemption_flag_.store(true);
  cpu_fj_thread_->stop_cpu_solver();
  cpu_fj_thread_->wait_for_cpu_solver();

  CUOPT_LOG_DEBUG("[Early CPUFJ] Stopped after %d iterations, solution_found=%d",
                  cpu_fj_thread_->fj_cpu ? cpu_fj_thread_->fj_cpu->iterations : 0,
                  this->solution_found_);

  cpu_fj_thread_.reset();
}

#if MIP_INSTANTIATE_FLOAT
template class early_cpufj_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class early_cpufj_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
