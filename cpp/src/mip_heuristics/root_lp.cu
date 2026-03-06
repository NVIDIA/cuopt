/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/problem/problem_helpers.cuh>
#include "root_lp.cuh"

#include <pdlp/pdlp.cuh>
#include <pdlp/solve.cuh>

#include <dual_simplex/types.hpp>
#include <raft/core/copy.hpp>
#include <utilities/timer.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
cuopt::linear_programming::dual_simplex::root_relaxation_first_solution_t<i_t, f_t>
run_pdlp_barrier_for_root_lp(problem_t<i_t, f_t>* problem,
                             f_t time_limit,
                             std::atomic<int>* concurrent_halt,
                             i_t num_gpus)
{
  convert_greater_to_less(*problem);
  f_t tolerance_divisor =
    problem->tolerances.absolute_tolerance /
    (problem->tolerances.relative_tolerance > 0 ? problem->tolerances.relative_tolerance : 1);
  pdlp_solver_settings_t<i_t, f_t> pdlp_settings{};
  pdlp_settings.tolerances.relative_primal_tolerance =
    problem->tolerances.absolute_tolerance / tolerance_divisor;
  pdlp_settings.tolerances.relative_dual_tolerance =
    problem->tolerances.absolute_tolerance / tolerance_divisor;
  pdlp_settings.time_limit            = time_limit;
  pdlp_settings.first_primal_feasible = false;
  pdlp_settings.concurrent_halt       = concurrent_halt;
  pdlp_settings.method                = method_t::Concurrent;
  pdlp_settings.inside_mip            = true;
  pdlp_settings.pdlp_solver_mode      = pdlp_solver_mode_t::Stable2;
  pdlp_settings.num_gpus              = num_gpus;
  pdlp_settings.presolver             = presolver_t::None;

  timer_t lp_timer(time_limit);
  auto lp_result = solve_lp_with_method<i_t, f_t>(*problem, pdlp_settings, lp_timer);

  cuopt::linear_programming::dual_simplex::root_relaxation_first_solution_t<i_t, f_t> result;
  auto stream = problem->handle_ptr->get_stream();
  result.primal.resize(lp_result.get_primal_solution().size());
  result.dual.resize(lp_result.get_dual_solution().size());
  result.reduced_costs.resize(lp_result.get_reduced_cost().size());
  raft::copy(
    result.primal.data(), lp_result.get_primal_solution().data(), result.primal.size(), stream);
  raft::copy(result.dual.data(), lp_result.get_dual_solution().data(), result.dual.size(), stream);
  raft::copy(result.reduced_costs.data(),
             lp_result.get_reduced_cost().data(),
             result.reduced_costs.size(),
             stream);
  problem->handle_ptr->sync_stream();
  result.objective      = problem->get_solver_obj_from_user_obj(lp_result.get_objective_value());
  result.user_objective = lp_result.get_objective_value();
  result.iterations     = lp_result.get_additional_termination_information().number_of_steps_taken;
  return result;
}

template cuopt::linear_programming::dual_simplex::root_relaxation_first_solution_t<int, double>
run_pdlp_barrier_for_root_lp<int, double>(problem_t<int, double>*, double, std::atomic<int>*, int);

template cuopt::linear_programming::dual_simplex::root_relaxation_first_solution_t<int, float>
run_pdlp_barrier_for_root_lp<int, float>(problem_t<int, float>*, float, std::atomic<int>*, int);

}  // namespace cuopt::linear_programming::detail
