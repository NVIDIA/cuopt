/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <cuopt/mathematical_optimization/mip/solver_stats.hpp>
#include <cuopt/mathematical_optimization/utilities/internals.hpp>

#include <mip_heuristics/presolve/semi_continuous.cuh>
#include <mip_heuristics/problem/problem.cuh>

#include <utilities/copy_helpers.hpp>
#include <utilities/logger.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Single point at which MIP incumbents are reported to the user get-solution callbacks.
// The heuristic thread (through the population) and the branch-and-bound thread both publish
// here, so the guard on the last published objective is shared and every incumbent is reported
// once, at the moment it is found rather than when the heuristic thread next drains its queue.
template <typename i_t, typename f_t>
class solution_publication_t {
 public:
  solution_publication_t(const mip_solver_settings_t<i_t, f_t>& settings,
                         const solver_stats_t<i_t, f_t>& stats)
    : settings_(settings), stats_(stats)
  {
    if (has_get_solution_callback()) {
      RAFT_CUDA_TRY(cudaGetDevice(&device_id_));
      handle_ = std::make_unique<raft::handle_t>();
    }
  }

  // Whether any get-solution callback is registered. Callers can use this to skip assembling
  // the host assignment that publish_if_better would otherwise discard.
  bool enabled() const { return handle_ != nullptr; }

  // `assignment` and `solver_objective` are in problem_ptr's solver space, which is always
  // oriented as a minimization. Returns whether the incumbent was published.
  //
  // Post-processing runs on a private stream, so this is safe to call from the branch-and-bound
  // thread while the heuristic thread owns problem_ptr->handle_ptr's stream.
  bool publish_if_better(problem_t<i_t, f_t>* problem_ptr,
                         const std::vector<f_t>& assignment,
                         f_t solver_objective)
  {
    if (handle_ == nullptr) { return false; }
    cuopt_assert(problem_ptr != nullptr, "Publication problem pointer must not be null");
    cuopt_assert(std::isfinite(solver_objective), "Published objective must be finite");

    std::lock_guard<std::mutex> lock(mutex_);
    if (!(solver_objective < best_published_objective_)) { return false; }
    best_published_objective_ = solver_objective;

    const auto user_assignment = build_user_assignment(problem_ptr, assignment);
    const f_t user_objective   = problem_ptr->get_user_obj_from_solver_obj(solver_objective);
    const f_t user_bound       = stats_.get_solution_bound();
    CUOPT_LOG_DEBUG("Publishing incumbent: objective %g, %lu variables",
                    user_objective,
                    user_assignment.size());

    for (auto callback : settings_.get_mip_callbacks()) {
      if (callback == nullptr ||
          callback->get_type() != internals::base_solution_callback_type::GET_SOLUTION) {
        continue;
      }
      // Each callback gets its own copies: the interface hands out mutable pointers.
      std::vector<f_t> callback_assignment(user_assignment);
      std::vector<f_t> callback_objective(1, user_objective);
      std::vector<f_t> callback_bound(1, user_bound);
      auto get_sol_callback = static_cast<internals::get_solution_callback_t*>(callback);
      get_sol_callback->get_solution(callback_assignment.data(),
                                     callback_objective.data(),
                                     callback_bound.data(),
                                     get_sol_callback->get_user_data());
    }
    return true;
  }

 private:
  // Lifts a solver-space assignment into the space the callbacks were set up for.
  std::vector<f_t> build_user_assignment(problem_t<i_t, f_t>* problem_ptr,
                                         const std::vector<f_t>& assignment)
  {
    // The B&B thread may never have selected a device of its own.
    RAFT_CUDA_TRY(cudaSetDevice(device_id_));
    auto stream = handle_->get_stream();
    rmm::device_uvector<f_t> d_assignment(assignment.size(), stream);
    raft::copy(d_assignment.data(), assignment.data(), assignment.size(), stream);
    // post_process_assignment writes through problem_ptr->presolve_data.fixed_var_assignment,
    // which both publishing threads share: the caller's lock is what keeps them apart.
    problem_ptr->post_process_assignment(d_assignment, true, stream);
    if (problem_ptr->has_papilo_presolve_data()) {
      problem_ptr->papilo_uncrush_assignment(d_assignment, stream);
    }
    auto user_assignment = cuopt::host_copy(d_assignment, stream);
    if (mip_solver_settings_accessor<i_t, f_t>::has_semi_continuous_callback_translation(
          settings_)) {
      strip_semi_continuous_auxiliaries_from_assignment(
        user_assignment,
        mip_solver_settings_accessor<i_t, f_t>::get_semi_continuous_original_num_variables(
          settings_));
    }
    return user_assignment;
  }

  bool has_get_solution_callback() const
  {
    for (auto callback : settings_.get_mip_callbacks()) {
      if (callback != nullptr &&
          callback->get_type() == internals::base_solution_callback_type::GET_SOLUTION) {
        return true;
      }
    }
    return false;
  }

  const mip_solver_settings_t<i_t, f_t>& settings_;
  const solver_stats_t<i_t, f_t>& stats_;
  int device_id_{0};
  // Null when no get-solution callback is registered, which also disables publication.
  std::unique_ptr<raft::handle_t> handle_;
  std::mutex mutex_;
  f_t best_published_objective_{std::numeric_limits<f_t>::max()};
};

}  // namespace cuopt::mathematical_optimization::mip
