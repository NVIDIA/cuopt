/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <cuopt/mathematical_optimization/optimization_problem.hpp>

#include <chrono>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename f_t>
using early_incumbent_callback_t = std::function<void(
  f_t solver_obj, f_t user_obj, const std::vector<f_t>& assignment, const char* heuristic_name)>;

// CRTP base for early heuristics that run on the original (or papilo-presolved) problem
// during presolve to find incumbents as early as possible.
// Derived classes implement start() and stop().
template <typename i_t, typename f_t, typename Derived>
class early_heuristic_t {
 public:
  early_heuristic_t(const optimization_problem_t<i_t, f_t>& op_problem,
                    early_incumbent_callback_t<f_t> incumbent_callback)
    : objective_scaling_factor_(op_problem.get_sense() ? -op_problem.get_objective_scaling_factor()
                                                       : op_problem.get_objective_scaling_factor()),
      objective_offset_(op_problem.get_sense() ? -op_problem.get_objective_offset()
                                               : op_problem.get_objective_offset()),
      incumbent_callback_(std::move(incumbent_callback))
  {
  }

  bool solution_found() const { return solution_found_; }
  f_t get_best_objective() const { return best_objective_; }
  // Return the best objective converted to user-space (sense-aware, offset-aware).
  f_t get_best_user_objective() const
  {
    return objective_scaling_factor_ * (best_objective_ + objective_offset_);
  }
  // Set the incumbent threshold.  `obj` must be in THIS heuristic's solver-space
  // (i.e. the space of its input problem).  Callers that hold a value from a
  // different problem representation (e.g., the original pre-presolve problem)
  // must convert it first, otherwise try_update_best will reject valid solutions.
  void set_best_objective(f_t obj) { best_objective_ = obj; }
  const std::vector<f_t>& get_best_assignment() const { return best_assignment_; }

 protected:
  ~early_heuristic_t() = default;

  // NOT thread-safe. solver_obj is in solver-space (always minimization).
  void try_update_best(f_t solver_obj, const std::vector<f_t>& assignment)
  {
    if (solver_obj >= best_objective_) { return; }
    best_objective_ = solver_obj;

    best_assignment_ = ((Derived*)this)->to_user_assignment(assignment);
    solution_found_  = true;
    f_t user_obj     = get_best_user_objective();
    // Log and callback are deferred to the shared incumbent_callback_ which enforces
    // global monotonicity across all early heuristic instances.
    if (incumbent_callback_) {
      incumbent_callback_(solver_obj, user_obj, best_assignment_, Derived::name());
    }
  }

  bool solution_found_{false};
  f_t best_objective_{std::numeric_limits<f_t>::infinity()};
  f_t objective_scaling_factor_;
  f_t objective_offset_;
  std::vector<f_t> best_assignment_;

  early_incumbent_callback_t<f_t> incumbent_callback_;
  std::chrono::steady_clock::time_point start_time_;
};

}  // namespace cuopt::mathematical_optimization::mip
