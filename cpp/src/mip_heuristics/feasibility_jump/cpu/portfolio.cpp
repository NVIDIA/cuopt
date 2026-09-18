/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "climber.hpp"
#include "internal.hpp"
#include "problem.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::fj_cpu_deleter_t::operator()(fj_cpu_climber_t<i_t, f_t>* ptr) const
{
  delete ptr;
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::create_worker(
  const lp_problem_t<i_t, f_t>& problem,
  const std::vector<simplex::variable_type_t>& variable_types,
  i_t n_structural,
  const std::vector<f_t>& start_assignment,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  std::string log_prefix,
  int64_t seed)
{
  auto new_climber = init_fj_cpu_from_host_lp(
    problem, variable_types, n_structural, start_assignment, settings, preemption_flag, seed);
  fj_cpu.reset(new_climber.release());
  fj_cpu->log_prefix           = std::move(log_prefix);
  fj_cpu->improvement_callback = improvement_callback;
  fj_cpu->halted               = false;
  preemption_flag              = false;
  is_initialized               = true;
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::run_async(f_t time_limit, double work_unit_limit)
{
  if (!is_initialized) return;

  auto& fj_ptr = fj_cpu;
#pragma omp task shared(fj_cpu, is_initialized, fj_ptr) firstprivate(time_limit, work_unit_limit) \
  priority(CUOPT_DEFAULT_TASK_PRIORITY) default(none) depend(out : fj_ptr)
  {
    if (is_initialized) { cpufj_solve(fj_cpu.get(), time_limit, work_unit_limit); }
  }
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::run_sync(f_t time_limit, double work_unit_limit)
{
  if (!is_initialized) return;
  cpufj_solve(fj_cpu.get(), time_limit, work_unit_limit);
  is_initialized = false;
  fj_cpu.reset();
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::stop()
{
  if (!is_initialized) return;

  preemption_flag = true;

  auto& fj_ptr = fj_cpu;
#pragma omp taskwait depend(in : fj_ptr)
  is_initialized = false;
  fj_cpu.reset();
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::send_stop_signal()
{
  preemption_flag = true;
}

#if MIP_INSTANTIATE_FLOAT
template struct fj_cpu_worker_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template struct fj_cpu_worker_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
