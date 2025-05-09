/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <dual_simplex/logger.hpp>
#include <dual_simplex/types.hpp>

#include <algorithm>
#include <atomic>
#include <functional>
#include <limits>
#include <thread>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
struct simplex_solver_settings_t {
 public:
  simplex_solver_settings_t()
    : iteration_limit(std::numeric_limits<i_t>::max()),
      time_limit(std::numeric_limits<f_t>::infinity()),
      absolute_mip_gap_tol(0.0),
      relative_mip_gap_tol(1e-3),
      integer_tol(1e-5),
      primal_tol(1e-6),
      dual_tol(1e-6),
      pivot_tol(1e-7),
      tight_tol(1e-10),
      fixed_tol(1e-10),
      zero_tol(1e-12),
      cut_off(std::numeric_limits<f_t>::infinity()),
      steepest_edge_ratio(0.5),
      steepest_edge_primal_tol(1e-9),
      use_steepest_edge_pricing(true),
      use_harris_ratio(false),
      use_bound_flip_ratio(true),
      scale_columns(true),
      relaxation(false),
      use_left_looking_lu(false),
      eliminate_singletons(true),
      print_presolve_stats(true),
      refactor_frequency(100),
      iteration_log_frequency(1000),
      first_iteration_log(2),
      num_threads(std::thread::hardware_concurrency() > 8
                    ? (std::thread::hardware_concurrency() / 8)
                    : std::thread::hardware_concurrency()),
      random_seed(0),
      inside_mip(0),
      solution_callback(nullptr),
      heuristic_preemption_callback(nullptr),
      concurrent_halt(nullptr)
  {
  }

  void set_log(bool logging) const { log.log = logging; }
  void enable_log_to_file() { log.enable_log_to_file(); }
  void set_log_filename(const std::string& log_filename) { log.set_log_file(log_filename); }
  void close_log_file() { log.close_log_file(); }
  i_t iteration_limit;
  f_t time_limit;
  f_t absolute_mip_gap_tol;  // Tolerance on mip gap to declare optimal
  f_t relative_mip_gap_tol;  // Tolerance on mip gap to declare optimal
  f_t integer_tol;           // Tolerance on integralitiy violation
  f_t primal_tol;            // Absolute primal infeasibility tolerance
  f_t dual_tol;              // Absolute dual infeasibility tolerance
  f_t pivot_tol;             // Simplex pivot tolerance
  f_t tight_tol;             // A tight tolerance used to check for infeasibility
  f_t fixed_tol;             // If l <= x <= u with u - l < fixed_tol a variable is consider fixed
  f_t zero_tol;              // Values below this tolerance are considered numerically zero
  f_t cut_off;               // If the dual objective is greater than the cutoff we stop
  f_t
    steepest_edge_ratio;  // the ratio of computed steepest edge mismatch from updated steepest edge
  f_t steepest_edge_primal_tol;    // Primal tolerance divided by steepest edge norm
  bool use_steepest_edge_pricing;  // true if using steepest edge pricing, false if using max
                                   // infeasibility pricing
  bool use_harris_ratio;           // true if using the harris ratio test
  bool use_bound_flip_ratio;       // true if using the bound flip ratio test
  bool scale_columns;              // true to scale the columns of A
  bool relaxation;                 // true to only solve the LP relaxation of a MIP
  bool
    use_left_looking_lu;  // true to use left looking LU factorization, false to use right looking
  bool eliminate_singletons;    // true to eliminate singletons from the basis
  bool print_presolve_stats;    // true to print presolve stats
  i_t refactor_frequency;       // number of basis updates before refactorization
  i_t iteration_log_frequency;  // number of iterations between log updates
  i_t first_iteration_log;      // number of iterations to log at beginning of solve
  i_t num_threads;              // number of threads to use
  i_t random_seed;              // random seed
  i_t inside_mip;  // 0 if outside MIP, 1 if inside MIP at root node, 2 if inside MIP at leaf node
  std::function<void(std::vector<f_t>&, f_t)> solution_callback;
  std::function<void()> heuristic_preemption_callback;
  mutable logger_t log;
  std::atomic<i_t>* concurrent_halt;  // if nullptr ignored, if !nullptr, 0 if solver should
                                      // continue, 1 if solver should halt
};

}  // namespace cuopt::linear_programming::dual_simplex
