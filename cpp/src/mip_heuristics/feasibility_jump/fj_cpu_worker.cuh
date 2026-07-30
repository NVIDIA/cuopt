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

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
struct fj_cpu_climber_t;

template <typename i_t, typename f_t>
struct fj_cpu_worker_t {
  // Custom deleter to avoid pulling the entire fj_cpu_climber_t class here.
  struct fj_cpu_deleter_t {
    void operator()(fj_cpu_climber_t<i_t, f_t>* ptr) const;
  };
  std::atomic<bool> preemption_flag{false};
  std::unique_ptr<fj_cpu_climber_t<i_t, f_t>, fj_cpu_deleter_t> fj_cpu;
  std::function<void(f_t, const std::vector<f_t>&, double)> improvement_callback;

  ~fj_cpu_worker_t() { stop(); }

  // `seed` selects the FJ RNG seed: pass a non-negative value for a deterministic seed,
  // or -1 to draw from the global cuopt::seed_generator (the historical behavior).
  // In deterministic mode the caller MUST pass an explicit seed, otherwise the underlying
  // seed_generator::get_seed() racing with concurrent callers breaks reproducibility.
  void create_worker(const simplex::lp_problem_t<i_t, f_t>& problem,
                     const std::vector<simplex::variable_type_t>& variable_types,
                     const std::vector<f_t>& seed_assignment,
                     const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                     std::string log_prefix,
                     int64_t seed = -1);

  // Run the worker asynchronously (i.e., launch an openmp task and then continue the
  // execution). Call `stop()` for stopping the worker
  void run_async(f_t time_limit         = std::numeric_limits<f_t>::infinity(),
                 double work_unit_limit = std::numeric_limits<double>::infinity());

  // Run the CPU FJ synchronously (i.e., wait for it to finish before proceeding)
  void run_sync(f_t time_limit         = std::numeric_limits<f_t>::infinity(),
                double work_unit_limit = std::numeric_limits<double>::infinity());

  void stop();
};

}  // namespace cuopt::mathematical_optimization::mip
