/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <branch_and_bound/concurrent_root_solver.hpp>
#include <utilities/timer.hpp>

#include <mip_heuristics/problem/problem.cuh>
#include <pdlp/solve.cuh>

#include <raft/util/cudart_utils.hpp>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
pdlp_solver_settings_t<i_t, f_t> make_mip_root_lp_settings(
  const mip_solver_settings_t<i_t, f_t>& mip_settings)
{
  pdlp_solver_settings_t<i_t, f_t> settings{};
  settings.tolerances.absolute_dual_tolerance   = mip_settings.tolerances.absolute_tolerance;
  settings.tolerances.relative_dual_tolerance   = mip_settings.tolerances.relative_tolerance;
  settings.tolerances.absolute_primal_tolerance = mip_settings.tolerances.absolute_tolerance;
  settings.tolerances.relative_primal_tolerance = mip_settings.tolerances.relative_tolerance;
  settings.first_primal_feasible                = false;
  settings.method                               = mip_settings.method;
  settings.inside_mip                           = true;
  settings.pdlp_solver_mode                     = pdlp_solver_mode_t::Stable2;
  settings.num_gpus                             = mip_settings.num_gpus;
  settings.presolver                            = presolver_t::None;
  settings.per_constraint_residual              = true;
  set_pdlp_solver_mode(settings);
  return settings;
}

template <typename i_t, typename f_t>
concurrent_root_solution_t<i_t, f_t> solve_concurrent_root_relaxation(
  problem_t<i_t, f_t>* problem,
  const pdlp_solver_settings_t<i_t, f_t>& settings,
  f_t time_limit,
  std::atomic<int>* concurrent_halt)
{
  concurrent_root_solution_t<i_t, f_t> result;
  auto root_settings            = settings;
  root_settings.time_limit      = time_limit;
  root_settings.concurrent_halt = concurrent_halt;

  timer_t root_timer(time_limit);
  auto lp_result    = solve_lp_with_method<i_t, f_t>(*problem, root_settings, root_timer);
  const auto status = lp_result.get_termination_status();
  result.usable =
    status != pdlp_termination_status_t::NumericalError &&
    status != pdlp_termination_status_t::ConcurrentLimit &&
    lp_result.get_primal_solution().size() == static_cast<size_t>(problem->n_variables) &&
    lp_result.get_dual_solution().size() == static_cast<size_t>(problem->n_constraints);
  result.optimal = status == pdlp_termination_status_t::Optimal;
  if (!result.usable) { return result; }

  auto& d_primal       = lp_result.get_primal_solution();
  auto& d_dual         = lp_result.get_dual_solution();
  auto& d_reduced_cost = lp_result.get_reduced_cost();
  result.primal.resize(d_primal.size());
  result.dual.resize(d_dual.size());
  result.reduced_cost.resize(d_reduced_cost.size());
  auto stream = problem->handle_ptr->get_stream();
  raft::copy(result.primal.data(), d_primal.data(), d_primal.size(), stream);
  raft::copy(result.dual.data(), d_dual.data(), d_dual.size(), stream);
  raft::copy(result.reduced_cost.data(), d_reduced_cost.data(), d_reduced_cost.size(), stream);
  problem->handle_ptr->sync_stream();

  result.user_objective   = lp_result.get_objective_value();
  result.solver_objective = problem->get_solver_obj_from_user_obj(result.user_objective);
  result.iterations = lp_result.get_additional_termination_information().number_of_steps_taken;
  result.method     = lp_result.get_additional_termination_information().solved_by;
  return result;
}

template pdlp_solver_settings_t<int, double> make_mip_root_lp_settings(
  const mip_solver_settings_t<int, double>&);

template concurrent_root_solution_t<int, double> solve_concurrent_root_relaxation(
  problem_t<int, double>*, const pdlp_solver_settings_t<int, double>&, double, std::atomic<int>*);

}  // namespace cuopt::mathematical_optimization::mip
