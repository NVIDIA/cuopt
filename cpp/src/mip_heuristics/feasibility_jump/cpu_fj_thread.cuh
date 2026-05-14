/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>

#include <atomic>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct fj_cpu_climber_t;

template <typename i_t, typename f_t>
struct fj_cpu_task_t {
  struct fj_cpu_deleter_t {
    void operator()(fj_cpu_climber_t<i_t, f_t>* ptr) const;
  };
  std::atomic<bool> preemption_flag{false};
  std::unique_ptr<fj_cpu_climber_t<i_t, f_t>, fj_cpu_deleter_t> fj_cpu;
};

// `seed` selects the FJ RNG seed: pass a non-negative value for a deterministic seed,
// or -1 to draw from the global cuopt::seed_generator (the historical behavior).
// In deterministic mode the caller MUST pass an explicit seed, otherwise the underlying
// seed_generator::get_seed() racing with concurrent callers breaks reproducibility.
template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_task_t<i_t, f_t>> make_fj_cpu_task_from_host_lp(
  const dual_simplex::lp_problem_t<i_t, f_t>& problem,
  const std::vector<dual_simplex::variable_type_t>& variable_types,
  const std::vector<f_t>& seed_assignment,
  const dual_simplex::simplex_solver_settings_t<i_t, f_t>& settings,
  std::function<void(f_t, const std::vector<f_t>&, double)> improvement_callback,
  std::string log_prefix,
  int64_t seed = -1);

template <typename i_t, typename f_t>
void run_fj_cpu_task(fj_cpu_task_t<i_t, f_t>& task,
                     f_t time_limit         = std::numeric_limits<f_t>::infinity(),
                     double work_unit_limit = std::numeric_limits<double>::infinity());

template <typename i_t, typename f_t>
void stop_fj_cpu_task(fj_cpu_task_t<i_t, f_t>& task);

}  // namespace cuopt::linear_programming::detail
