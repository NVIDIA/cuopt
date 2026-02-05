/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <atomic>
#include <limits>
namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
struct solver_stats_t {
  // Direction-neutral placeholder; solver_context initializes based on maximize/minimize.
  solver_stats_t() : solution_bound(std::numeric_limits<f_t>::infinity()) {}

  solver_stats_t(const solver_stats_t& other) { *this = other; }

  solver_stats_t& operator=(const solver_stats_t& other)
  {
    if (this == &other) { return *this; }
    total_solve_time = other.total_solve_time;
    presolve_time    = other.presolve_time;
    solution_bound.store(other.solution_bound.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
    num_nodes              = other.num_nodes;
    num_simplex_iterations = other.num_simplex_iterations;
    return *this;
  }

  f_t get_solution_bound() const { return solution_bound.load(std::memory_order_relaxed); }

  void set_solution_bound(f_t value) { solution_bound.store(value, std::memory_order_relaxed); }

  f_t total_solve_time = 0.;
  f_t presolve_time    = 0.;
  std::atomic<f_t> solution_bound;
  i_t num_nodes              = 0;
  i_t num_simplex_iterations = 0;
};

}  // namespace cuopt::linear_programming
