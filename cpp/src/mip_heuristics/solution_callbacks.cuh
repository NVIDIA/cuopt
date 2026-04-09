/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/mip/solver_stats.hpp>

#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solution/solution.cuh>

#include <limits>
#include <mutex>
#include <vector>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct solution_callback_payload_t {
  std::vector<f_t> assignment{};
  f_t user_objective{};
  f_t solver_objective{};
  internals::mip_solution_callback_info_t callback_info{};
};

template <typename f_t>
void dispatch_get_solution_callbacks(
  const std::vector<internals::base_solution_callback_t*>& user_callbacks,
  const std::vector<f_t>& assignment,
  f_t user_objective,
  f_t solution_bound,
  const internals::mip_solution_callback_info_t& callback_info)
{
  for (auto callback : user_callbacks) {
    if (callback->get_type() != internals::base_solution_callback_type::GET_SOLUTION_EXT &&
        callback->get_type() != internals::base_solution_callback_type::GET_SOLUTION) {
      continue;
    }

    std::vector<f_t> user_assignment(assignment);
    std::vector<f_t> user_objective_vec(1, user_objective);
    std::vector<f_t> user_bound_vec(1, solution_bound);
    if (callback->get_type() == internals::base_solution_callback_type::GET_SOLUTION_EXT) {
      auto get_sol_callback_ext = static_cast<internals::get_solution_callback_ext_t*>(callback);
      get_sol_callback_ext->get_solution(user_assignment.data(),
                                         user_objective_vec.data(),
                                         user_bound_vec.data(),
                                         &callback_info,
                                         get_sol_callback_ext->get_user_data());
    } else if (callback->get_type() == internals::base_solution_callback_type::GET_SOLUTION) {
      auto get_sol_callback = static_cast<internals::get_solution_callback_t*>(callback);
      get_sol_callback->get_solution(user_assignment.data(),
                                     user_objective_vec.data(),
                                     user_bound_vec.data(),
                                     get_sol_callback->get_user_data());
    }
  }
}

template <typename i_t, typename f_t>
class solution_publication_t {
 public:
  solution_publication_t(const mip_solver_settings_t<i_t, f_t>& settings_,
                         solver_stats_t<i_t, f_t>& stats_)
    : settings(settings_), stats(stats_)
  {
  }

  void reset_published_best(f_t objective = std::numeric_limits<f_t>::max())
  {
    best_callback_feasible_objective_ = objective;
  }

  solution_callback_payload_t<i_t, f_t> build_callback_payload(
    problem_t<i_t, f_t>* problem_ptr,
    solution_t<i_t, f_t>& sol,
    internals::mip_solution_origin_t origin,
    double work_timestamp)
  {
    cuopt_assert(problem_ptr != nullptr, "Callback payload problem pointer must not be null");
    cuopt_assert(work_timestamp >= 0.0, "work_timestamp must not be negative");
    solution_callback_payload_t<i_t, f_t> payload{};
    payload.user_objective               = sol.get_user_objective();
    payload.solver_objective             = sol.get_objective();
    payload.callback_info.origin         = (uint32_t)origin;
    payload.callback_info.work_timestamp = work_timestamp;
    solution_t<i_t, f_t> temp_sol(sol);
    CUOPT_LOG_DEBUG("build_callback_payload: pre_postprocess size=%zu handle=%p problem_handle=%p",
                    temp_sol.assignment.size(),
                    (void*)sol.handle_ptr,
                    (void*)problem_ptr->handle_ptr);
    problem_ptr->post_process_assignment(temp_sol.assignment, true, sol.handle_ptr);
    CUOPT_LOG_DEBUG("build_callback_payload: post_postprocess size=%zu",
                    temp_sol.assignment.size());
    if (problem_ptr->has_papilo_presolve_data()) {
      CUOPT_LOG_DEBUG("build_callback_payload: pre_papilo size=%zu papilo_reduced_size=%zu",
                      temp_sol.assignment.size(),
                      problem_ptr->get_papilo_original_num_variables());
      problem_ptr->papilo_uncrush_assignment(temp_sol.assignment, sol.handle_ptr);
      CUOPT_LOG_DEBUG("build_callback_payload: post_papilo size=%zu", temp_sol.assignment.size());
    }
    payload.assignment = cuopt::host_copy(temp_sol.assignment, temp_sol.handle_ptr->get_stream());
    CUOPT_LOG_DEBUG("build_callback_payload: final payload size=%zu obj=%.6g origin=%s",
                    payload.assignment.size(),
                    payload.user_objective,
                    internals::mip_solution_origin_to_string(origin));
    return payload;
  }

  bool publish_new_best_feasible(const solution_callback_payload_t<i_t, f_t>& payload,
                                 double elapsed_time = -1.0)
  {
    std::lock_guard<std::mutex> lock(solution_callback_mutex_);
    cuopt_assert(std::isfinite(payload.solver_objective),
                 "Feasible incumbent objective must be finite");
    if (!(payload.solver_objective < best_callback_feasible_objective_)) { return false; }

    best_callback_feasible_objective_ = payload.solver_objective;
    if (settings.benchmark_info_ptr != nullptr && elapsed_time >= 0.0) {
      settings.benchmark_info_ptr->last_improvement_of_best_feasible = elapsed_time;
    }
    invoke_get_solution_callbacks(payload);
    return true;
  }

 private:
  void invoke_get_solution_callbacks(const solution_callback_payload_t<i_t, f_t>& payload)
  {
    auto user_callbacks = settings.get_mip_callbacks();
    CUOPT_LOG_DEBUG("Publishing incumbent: obj=%g wut=%.6f origin=%s callbacks=%zu",
                    payload.user_objective,
                    payload.callback_info.work_timestamp,
                    internals::mip_solution_origin_to_string(
                      (internals::mip_solution_origin_t)payload.callback_info.origin),
                    user_callbacks.size());
    dispatch_get_solution_callbacks(user_callbacks,
                                    payload.assignment,
                                    payload.user_objective,
                                    stats.get_solution_bound(),
                                    payload.callback_info);
  }

  const mip_solver_settings_t<i_t, f_t>& settings;
  solver_stats_t<i_t, f_t>& stats;
  std::mutex solution_callback_mutex_;
  f_t best_callback_feasible_objective_{std::numeric_limits<f_t>::max()};
};

// Processes SET_SOLUTION user callbacks: invokes the callback, validates/scales/preprocesses
// the returned assignment, and returns it for the caller to reinject.
template <typename i_t, typename f_t>
class solution_injection_t {
 public:
  solution_injection_t(const mip_solver_settings_t<i_t, f_t>& settings_,
                       solver_stats_t<i_t, f_t>& stats_)
    : settings(settings_), stats(stats_)
  {
  }

  template <typename OnInjectedFn>
  void invoke_set_solution_callbacks(problem_t<i_t, f_t>* problem_ptr,
                                     solution_t<i_t, f_t>& current_incumbent,
                                     OnInjectedFn&& on_injected)
  {
    auto user_callbacks = settings.get_mip_callbacks();
    for (auto callback : user_callbacks) {
      if (callback->get_type() != internals::base_solution_callback_type::SET_SOLUTION) {
        continue;
      }
      auto set_sol_callback       = static_cast<internals::set_solution_callback_t*>(callback);
      f_t user_bound              = stats.get_solution_bound();
      auto callback_num_variables = problem_ptr->original_problem_ptr->get_n_variables();
      rmm::device_uvector<f_t> incumbent_assignment(callback_num_variables,
                                                    current_incumbent.handle_ptr->get_stream());
      auto inf = std::numeric_limits<f_t>::infinity();
      current_incumbent.handle_ptr->sync_stream();
      std::vector<f_t> h_incumbent_assignment(incumbent_assignment.size());
      std::vector<f_t> h_outside_sol_objective(1, inf);
      std::vector<f_t> h_user_bound(1, user_bound);
      set_sol_callback->set_solution(h_incumbent_assignment.data(),
                                     h_outside_sol_objective.data(),
                                     h_user_bound.data(),
                                     set_sol_callback->get_user_data());
      f_t outside_sol_objective = h_outside_sol_objective[0];
      if (outside_sol_objective == inf) { continue; }

      raft::copy(incumbent_assignment.data(),
                 h_incumbent_assignment.data(),
                 incumbent_assignment.size(),
                 current_incumbent.handle_ptr->get_stream());
      bool is_valid = problem_ptr->pre_process_assignment(incumbent_assignment);
      if (!is_valid) { continue; }

      solution_t<i_t, f_t> outside_sol(current_incumbent);
      cuopt_assert(outside_sol.assignment.size() == incumbent_assignment.size(),
                   "Incumbent assignment size mismatch");
      raft::copy(outside_sol.assignment.data(),
                 incumbent_assignment.data(),
                 incumbent_assignment.size(),
                 current_incumbent.handle_ptr->get_stream());
      outside_sol.compute_feasibility();

      CUOPT_LOG_DEBUG("Injected solution feasibility = %d objective = %g excess = %g",
                      outside_sol.get_feasible(),
                      outside_sol.get_user_objective(),
                      outside_sol.get_total_excess());
      cuopt_assert(std::abs(outside_sol.get_user_objective() - outside_sol_objective) <= 1e-6,
                   "External solution objective mismatch");
      on_injected(outside_sol.get_host_assignment(),
                  outside_sol.get_objective(),
                  internals::mip_solution_origin_t::USER_INJECTED);
    }
  }

 private:
  const mip_solver_settings_t<i_t, f_t>& settings;
  solver_stats_t<i_t, f_t>& stats;
};

}  // namespace cuopt::linear_programming::detail
