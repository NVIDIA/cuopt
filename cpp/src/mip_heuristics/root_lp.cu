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

#include <dual_simplex/crossover.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/types.hpp>
#include <raft/core/copy.hpp>
#include <utilities/timer.hpp>

namespace cuopt::linear_programming::detail {

namespace {
template <typename i_t, typename f_t>
cuopt::linear_programming::dual_simplex::root_relaxation_first_solution_t<i_t, f_t>
copy_lp_result_to_root_solution(problem_t<i_t, f_t>* problem,
                                const optimization_problem_solution_t<i_t, f_t>& lp_result)
{
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

}  // namespace

template <typename i_t, typename f_t>
cuopt::linear_programming::dual_simplex::root_relaxation_first_solution_t<i_t, f_t>
run_solver_for_root_lp(problem_t<i_t, f_t>* problem,
                       f_t time_limit,
                       std::atomic<int>* concurrent_halt,
                       i_t num_gpus,
                       method_t method)
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
  pdlp_settings.halt_set_by_caller    = true;  // B&B sets halt only after crossover
  pdlp_settings.method                = method;
  pdlp_settings.inside_mip            = true;
  pdlp_settings.num_gpus              = num_gpus;
  pdlp_settings.presolver             = presolver_t::None;
  pdlp_settings.crossover             = false;  // B&B does crush + crossover for both paths
  if (method == method_t::PDLP) { pdlp_settings.pdlp_solver_mode = pdlp_solver_mode_t::Stable2; }

  timer_t lp_timer(time_limit);
  auto lp_result = solve_lp_with_method<i_t, f_t>(*problem, pdlp_settings, lp_timer);
  return copy_lp_result_to_root_solution(problem, lp_result);
}

template <typename i_t, typename f_t>
cuopt::linear_programming::dual_simplex::crossover_status_t run_crush_crossover_and_maybe_win(
  const cuopt::linear_programming::dual_simplex::root_relaxation_first_solution_t<i_t, f_t>& result,
  const cuopt::linear_programming::dual_simplex::user_problem_t<i_t, f_t>& original_problem,
  const cuopt::linear_programming::dual_simplex::lp_problem_t<i_t, f_t>& original_lp,
  const std::vector<i_t>& new_slacks,
  const cuopt::linear_programming::dual_simplex::simplex_solver_settings_t<i_t, f_t>&
    crossover_settings,
  f_t start_time,
  std::atomic<int>* concurrent_halt,
  std::function<void()> set_halter,
  std::function<void(
    const cuopt::linear_programming::dual_simplex::root_relaxation_first_solution_t<i_t, f_t>&)>
    on_first_lp_solution,
  std::mutex* first_solver_mutex,
  bool* first_solver_callback_done,
  std::mutex* first_result_mutex,
  std::atomic<int>* winner,
  int winner_id,
  cuopt::linear_programming::dual_simplex::crossover_status_t* first_crossover_status_out,
  cuopt::linear_programming::dual_simplex::lp_solution_t<i_t, f_t>* winner_crossover_soln,
  std::vector<cuopt::linear_programming::dual_simplex::variable_status_t>* winner_crossover_vstatus,
  f_t* winner_root_objective,
  const char* this_solver_name,
  std::string* winner_solver_name_out)
{
  using namespace cuopt::linear_programming::dual_simplex;
  if (on_first_lp_solution) {
    std::lock_guard<std::mutex> lock(*first_solver_mutex);
    if (!*first_solver_callback_done) {
      *first_solver_callback_done = true;
      on_first_lp_solution(result);
    }
  }
  lp_solution_t<i_t, f_t> soln(original_lp.num_rows, original_lp.num_cols);
  soln.x              = result.primal;
  soln.y              = result.dual;
  soln.z              = result.reduced_costs;
  soln.objective      = result.objective;
  soln.user_objective = result.user_objective;
  soln.iterations     = result.iterations;
  std::vector<f_t> crushed_x;
  crush_primal_solution(original_problem, original_lp, soln.x, new_slacks, crushed_x);
  std::vector<f_t> crushed_y;
  std::vector<f_t> crushed_z;
  (void)crush_dual_solution(
    original_problem, original_lp, new_slacks, soln.y, soln.z, crushed_y, crushed_z);
  soln.x = std::move(crushed_x);
  soln.y = std::move(crushed_y);
  soln.z = std::move(crushed_z);
  lp_solution_t<i_t, f_t> crossover_out(original_lp.num_rows, original_lp.num_cols);
  std::vector<variable_status_t> vstatus_out(original_lp.num_cols);
  auto root_crossover_settings = crossover_settings;
  root_crossover_settings.inside_mip =
    1;  // root LP crossover; dual_phase2 uses this to set concurrent_halt
  root_crossover_settings.log.log         = false;
  root_crossover_settings.concurrent_halt = concurrent_halt;
  crossover_status_t status =
    crossover(original_lp, root_crossover_settings, soln, start_time, crossover_out, vstatus_out);
  {
    std::lock_guard<std::mutex> lock(*first_result_mutex);
    int expected = 0;
    if (status == crossover_status_t::OPTIMAL &&
        winner->compare_exchange_strong(expected, winner_id, std::memory_order_acq_rel)) {
      *first_crossover_status_out = status;
      if (winner_solver_name_out) { *winner_solver_name_out = this_solver_name; }
      winner_crossover_soln->x              = std::move(crossover_out.x);
      winner_crossover_soln->y              = std::move(crossover_out.y);
      winner_crossover_soln->z              = std::move(crossover_out.z);
      winner_crossover_soln->objective      = result.objective;
      winner_crossover_soln->user_objective = result.user_objective;
      winner_crossover_soln->iterations     = result.iterations;
      *winner_root_objective                = result.objective;
      *winner_crossover_vstatus             = std::move(vstatus_out);
      set_halter();
    } else {
      if (winner->load(std::memory_order_acquire) != 0) { status = *first_crossover_status_out; }
    }
  }
  return status;
}

template cuopt::linear_programming::dual_simplex::root_relaxation_first_solution_t<int, double>
run_solver_for_root_lp<int, double>(
  problem_t<int, double>*, double, std::atomic<int>*, int, method_t);
template cuopt::linear_programming::dual_simplex::crossover_status_t
run_crush_crossover_and_maybe_win<int, double>(
  const cuopt::linear_programming::dual_simplex::root_relaxation_first_solution_t<int, double>&,
  const cuopt::linear_programming::dual_simplex::user_problem_t<int, double>&,
  const cuopt::linear_programming::dual_simplex::lp_problem_t<int, double>&,
  const std::vector<int>&,
  const cuopt::linear_programming::dual_simplex::simplex_solver_settings_t<int, double>&,
  double,
  std::atomic<int>*,
  std::function<void()>,
  std::function<void(
    const cuopt::linear_programming::dual_simplex::root_relaxation_first_solution_t<int, double>&)>,
  std::mutex*,
  bool*,
  std::mutex*,
  std::atomic<int>*,
  int,
  cuopt::linear_programming::dual_simplex::crossover_status_t*,
  cuopt::linear_programming::dual_simplex::lp_solution_t<int, double>*,
  std::vector<cuopt::linear_programming::dual_simplex::variable_status_t>*,
  double*,
  const char*,
  std::string*);

#ifdef MIP_INSTANTIATION_FLOAT
template cuopt::linear_programming::dual_simplex::root_relaxation_first_solution_t<int, float>
run_solver_for_root_lp<int, float>(problem_t<int, float>*, float, std::atomic<int>*, int, method_t);
template cuopt::linear_programming::dual_simplex::crossover_status_t
run_crush_crossover_and_maybe_win<int, float>(
  const cuopt::linear_programming::dual_simplex::root_relaxation_first_solution_t<int, float>&,
  const cuopt::linear_programming::dual_simplex::user_problem_t<int, float>&,
  const cuopt::linear_programming::dual_simplex::lp_problem_t<int, float>&,
  const std::vector<int>&,
  const cuopt::linear_programming::dual_simplex::simplex_solver_settings_t<int, float>&,
  float,
  std::atomic<int>*,
  std::function<void()>,
  std::function<void(
    const cuopt::linear_programming::dual_simplex::root_relaxation_first_solution_t<int, float>&)>,
  std::mutex*,
  bool*,
  std::mutex*,
  std::atomic<int>*,
  int,
  cuopt::linear_programming::dual_simplex::crossover_status_t*,
  cuopt::linear_programming::dual_simplex::lp_solution_t<int, float>*,
  std::vector<cuopt::linear_programming::dual_simplex::variable_status_t>*,
  float*,
  const char*,
  std::string*);
#endif
}  // namespace cuopt::linear_programming::detail
