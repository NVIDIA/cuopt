/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "relaxed_lp.cuh"

#include <cuopt/error.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/utils.cuh>
#include <pdlp/solve.cuh>

#include <pdlp/pdlp.cuh>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/linalg/binary_op.cuh>

#include <thrust/tabulate.h>

#include <atomic>

// enable to activate detailed determinism logs
#if 0
#undef CUOPT_DETERMINISM_LOG
#define CUOPT_DETERMINISM_LOG(...) \
  do {                             \
    CUOPT_LOG_INFO(__VA_ARGS__);   \
  } while (0)
#endif

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> get_relaxed_lp_solution(
  problem_t<i_t, f_t>& op_problem,
  solution_t<i_t, f_t>& solution,
  const relaxed_lp_settings_t& settings)
{
  return get_relaxed_lp_solution(op_problem, solution.assignment, solution.lp_state, settings);
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> get_relaxed_lp_solution(
  problem_t<i_t, f_t>& op_problem,
  rmm::device_uvector<f_t>& assignment,
  lp_state_t<i_t, f_t>& lp_state,
  const relaxed_lp_settings_t& settings)
{
  raft::common::nvtx::range fun_scope("get_relaxed_lp_solution");
  static thread_local uint64_t lp_call_counter{0};
  const uint64_t lp_call_id = lp_call_counter++;

  pdlp_solver_settings_t<i_t, f_t> pdlp_settings{};
  pdlp_settings.detect_infeasibility = settings.check_infeasibility;
  pdlp_settings.set_optimality_tolerance(settings.tolerance);
  f_t tolerance_divisor =
    op_problem.tolerances.absolute_tolerance / op_problem.tolerances.relative_tolerance;
  if (tolerance_divisor == 0) { tolerance_divisor = 1; }
  pdlp_settings.tolerances.relative_primal_tolerance = settings.tolerance / tolerance_divisor;
  pdlp_settings.tolerances.relative_dual_tolerance   = settings.tolerance / tolerance_divisor;
  pdlp_settings.time_limit                           = settings.time_limit;
  pdlp_settings.iteration_limit                      = settings.iteration_limit;

  const f_t work_limit                  = settings.work_limit;
  const bool determinism_mode           = std::isfinite(work_limit);
  pdlp_settings.concurrent_halt         = settings.concurrent_halt;
  pdlp_settings.per_constraint_residual = settings.per_constraint_residual;
  pdlp_settings.first_primal_feasible   = settings.return_first_feasible;
  pdlp_settings.pdlp_solver_mode        = pdlp_solver_mode_t::Stable2;
  int estim_iters                       = pdlp_settings.iteration_limit;
  if (determinism_mode) {
    // try to estimate the iteration count based on the requested work limit
    // TODO: replace with an actual model. this is a rather ugly hack to avoid having
    // to touch the PDLP code for this initial PR
    estim_iters = 100;
    if (!std::isinf(work_limit)) {
      do {
        // TODO: use an actual predictor model here
        double estim_ms = 313 + 200 * op_problem.n_variables - 400 * op_problem.n_constraints +
                          600 * op_problem.coefficients.size() + 7100 * estim_iters;
        estim_ms = std::max(0.0, estim_ms);
        if (estim_ms > work_limit * 1000) { break; }
        estim_iters += 100;
      } while (true);
    } else {
      estim_iters = std::numeric_limits<int>::max();
    }
    CUOPT_DETERMINISM_LOG(
      "estimated iterations %d for work limit %f", estim_iters, settings.work_limit);
    pdlp_settings.iteration_limit  = estim_iters;
    pdlp_settings.time_limit       = std::numeric_limits<double>::infinity();
    pdlp_settings.pdlp_solver_mode = pdlp_solver_mode_t::Stable2;
    pdlp_settings.presolver        = presolver_t::None;
  }
  CUOPT_DETERMINISM_LOG(
    "LP call %lu[%d] config: det=%d work_limit=%.6f time_limit=%.6f iter_limit=%d method=%d "
    "mode=%d "
    "presolver=%d save_state=%d has_initial=%d assignment_hash=0x%x",
    lp_call_id,
    gettid(),
    (int)determinism_mode,
    settings.work_limit,
    pdlp_settings.time_limit,
    pdlp_settings.iteration_limit,
    (int)pdlp_settings.method,
    (int)pdlp_settings.pdlp_solver_mode,
    (int)pdlp_settings.presolver,
    (int)settings.save_state,
    (int)settings.has_initial_primal,
    detail::compute_hash(assignment, op_problem.handle_ptr->get_stream()));
  set_pdlp_solver_mode(pdlp_settings);
  // TODO: set Stable3 here?
  pdlp_solver_t<i_t, f_t> lp_solver(op_problem, pdlp_settings);
  if (settings.has_initial_primal) {
    i_t prev_size = lp_state.prev_dual.size();
    CUOPT_LOG_TRACE(
      "setting initial primal solution of size %d dual size %d problem vars %d cstrs %d",
      assignment.size(),
      lp_state.prev_dual.size(),
      op_problem.n_variables,
      op_problem.n_constraints);
    lp_state.resize(op_problem, op_problem.handle_ptr->get_stream());
    clamp_within_var_bounds(assignment, &op_problem, op_problem.handle_ptr);
    // The previous dual sometimes contain invalid values w.r.t current problem
    // Adjust better dual values when we use warm start
    thrust::tabulate(op_problem.handle_ptr->get_thrust_policy(),
                     lp_state.prev_dual.data(),
                     lp_state.prev_dual.data() + op_problem.n_constraints,
                     [prev_size, dual = make_span(lp_state.prev_dual)] __device__(i_t i) {
                       // early exit to avoid a false positive in compute-sanitizer initcheck
                       if (i >= prev_size) { return 0.0; }
                       f_t x = dual[i];
                       if (!isfinite(x)) { return 0.0; }
                       return x;
                     });
    lp_solver.set_initial_primal_solution(assignment);
    lp_solver.set_initial_dual_solution(lp_state.prev_dual);
  }
  CUOPT_LOG_TRACE(
    "running LP with n_vars %d n_cstr %d", op_problem.n_variables, op_problem.n_constraints);
  // before LP flush the logs as it takes quite some time
  cuopt::default_logger().flush();
  // temporarily add timer
  auto start_time = timer_t(pdlp_settings.time_limit);
  lp_solver.set_inside_mip(true);
  CUOPT_DETERMINISM_LOG(
    "prev solution sizes primal=%lu dual=%lu", assignment.size(), lp_state.prev_dual.size());
  if (determinism_mode) {
    auto init_primal_hash =
      detail::compute_hash(make_span(assignment), op_problem.handle_ptr->get_stream());
    auto init_dual_hash =
      settings.has_initial_primal
        ? detail::compute_hash(make_span(lp_state.prev_dual), op_problem.handle_ptr->get_stream())
        : 0u;
    CUOPT_DETERMINISM_LOG("LP call %lu pre-solve state: init_primal_hash=0x%x init_dual_hash=0x%x",
                          lp_call_id,
                          init_primal_hash,
                          init_dual_hash);
  }
  auto solver_response = lp_solver.run_solver(start_time);
  CUOPT_DETERMINISM_LOG("post LP primal size %lu", solver_response.get_primal_solution().size());
  const int actual_iters =
    solver_response.get_additional_termination_information().number_of_steps_taken;
  CUOPT_DETERMINISM_LOG("LP call %lu result: status=%d iters=%d primal_hash=0x%x",
                        lp_call_id,
                        (int)solver_response.get_termination_status(),
                        actual_iters,
                        solver_response.get_primal_solution().size() != 0
                          ? detail::compute_hash(solver_response.get_primal_solution(),
                                                 op_problem.handle_ptr->get_stream())
                          : 0u);

  if (determinism_mode && settings.work_context != nullptr) {
    double work_to_record = settings.work_limit;
    if (estim_iters > 0) {
      work_to_record =
        settings.work_limit * std::clamp((double)actual_iters / (double)estim_iters, 0.0, 1.0);
    }
    CUOPT_DETERMINISM_LOG(
      "LP call %lu recording %.6fwu (actual_iters=%d estim_iters=%d requested=%.6f)",
      lp_call_id,
      work_to_record,
      actual_iters,
      estim_iters,
      settings.work_limit);
    settings.work_context->record_work_sync_on_horizon(work_to_record);
  }

  if (solver_response.get_primal_solution().size() != 0 &&
      solver_response.get_dual_solution().size() != 0 && settings.save_state) {
    CUOPT_LOG_TRACE("saving initial primal solution of size %d", lp_state.prev_primal.size());
    lp_state.set_state(solver_response.get_primal_solution(), solver_response.get_dual_solution());
  }
  if (solver_response.get_primal_solution().size() != 0) {
    // copy the solution no matter what, because in the worst case we are closer to the polytope
    raft::copy(assignment.data(),
               solver_response.get_primal_solution().data(),
               solver_response.get_primal_solution().size(),
               op_problem.handle_ptr->get_stream());
  }
  CUOPT_DETERMINISM_LOG("LP call %lu assignment_after_copy hash=0x%x",
                        lp_call_id,
                        detail::compute_hash(assignment, op_problem.handle_ptr->get_stream()));
  if (solver_response.get_termination_status() == pdlp_termination_status_t::Optimal) {
    CUOPT_LOG_TRACE("feasible solution found with LP objective %f",
                    solver_response.get_objective_value());
  } else {
    CUOPT_DETERMINISM_LOG(
      "LP returned with reason %d, %d iterations",
      solver_response.get_termination_status(),
      solver_response.get_additional_termination_information().number_of_steps_taken);
  }

  return solver_response;
}

// Run LP with variables fixed to specific values
template <typename i_t, typename f_t>
bool run_lp_with_vars_fixed(problem_t<i_t, f_t>& op_problem,
                            problem_t<i_t, f_t>& fixed_problem,
                            solution_t<i_t, f_t>& solution,
                            rmm::device_uvector<f_t>& fixed_assignment,
                            rmm::device_uvector<i_t>& variable_map,
                            relaxed_lp_settings_t& settings,
                            bound_presolve_t<i_t, f_t>* bound_presolve,
                            bool check_fixed_assignment_feasibility)
{
  if (check_fixed_assignment_feasibility) {
    solution_t<i_t, f_t> temp_solution(fixed_problem);
    raft::copy(temp_solution.assignment.data(),
               fixed_assignment.data(),
               fixed_assignment.size(),
               fixed_problem.handle_ptr->get_stream());
    bool temp_solution_feasible = temp_solution.compute_feasibility();
    if (!temp_solution_feasible) {
      CUOPT_LOG_DEBUG(
        "Infeasible solution detected with fixed vars LP. Sol excess %f fixed sol excess %f",
        solution.get_total_excess(),
        temp_solution.get_total_excess());
      settings.time_limit = 1;
    }
  }
  if (bound_presolve != nullptr) {
    bound_presolve->resize(fixed_problem);
    // run bounds prop to quickly discover inevitably infeasible
    bound_presolve->settings.time_limit = (settings.time_limit / 10);
    auto term_crit                      = bound_presolve->solve(fixed_problem);
    bound_presolve->settings            = {};
    if (bound_presolve->infeas_constraints_count > 0) {
      solution.unfix_variables(fixed_assignment, variable_map);
      bound_presolve->resize(op_problem);
      CUOPT_LOG_DEBUG("Infeasible problem detected with LP with fixed vars");
      return true;
    }
  }
  fixed_problem.check_problem_representation(true);
  // if we are on the original problem and fixing the integers, save the state
  // if we are in recombiners and on a smaller problem, don't update the state with integers fixed
  CUOPT_LOG_TRACE("save_state %d", settings.save_state);
  auto& lp_state = fixed_problem.lp_state;
  auto solver_response =
    get_relaxed_lp_solution(fixed_problem, fixed_assignment, lp_state, settings);
  // unfix the assignment on given result no matter if it is feasible
  solution.unfix_variables(fixed_assignment, variable_map);
  if (bound_presolve != nullptr) { bound_presolve->resize(op_problem); }
  return false;
}

// returns true if the problem is inevitably infeasible
template <typename i_t, typename f_t>
bool run_lp_with_vars_fixed(problem_t<i_t, f_t>& op_problem,
                            solution_t<i_t, f_t>& solution,
                            const rmm::device_uvector<i_t>& variables_to_fix,
                            relaxed_lp_settings_t& settings,
                            bound_presolve_t<i_t, f_t>* bound_presolve,
                            bool check_fixed_assignment_feasibility,
                            bool use_integer_fixed_problem)
{
  // if we are fixing all vars, there is no lp to be run
  if (variables_to_fix.size() == (size_t)op_problem.n_variables) { return true; }
  if (use_integer_fixed_problem) {
    op_problem.fill_integer_fixed_problem(solution.assignment, op_problem.handle_ptr);
    auto fixed_assignment =
      op_problem.get_fixed_assignment_from_integer_fixed_problem(solution.assignment);
    return run_lp_with_vars_fixed(op_problem,
                                  *op_problem.integer_fixed_problem,
                                  solution,
                                  fixed_assignment,
                                  op_problem.integer_fixed_variable_map,
                                  settings,
                                  bound_presolve,
                                  check_fixed_assignment_feasibility);
  } else {
    auto [fixed_problem, fixed_assignment, variable_map] = solution.fix_variables(variables_to_fix);
    return run_lp_with_vars_fixed(op_problem,
                                  fixed_problem,
                                  solution,
                                  fixed_assignment,
                                  variable_map,
                                  settings,
                                  bound_presolve,
                                  check_fixed_assignment_feasibility);
  }
}

#define INSTANTIATE(F_TYPE)                                                                   \
  template optimization_problem_solution_t<int, F_TYPE> get_relaxed_lp_solution<int, F_TYPE>( \
    problem_t<int, F_TYPE> & op_problem,                                                      \
    solution_t<int, F_TYPE> & solution,                                                       \
    const relaxed_lp_settings_t& settings);                                                   \
  template optimization_problem_solution_t<int, F_TYPE> get_relaxed_lp_solution<int, F_TYPE>( \
    problem_t<int, F_TYPE> & op_problem,                                                      \
    rmm::device_uvector<F_TYPE> & assignment,                                                 \
    lp_state_t<int, F_TYPE> & lp_state,                                                       \
    const relaxed_lp_settings_t& settings);                                                   \
  template bool run_lp_with_vars_fixed<int, F_TYPE>(                                          \
    problem_t<int, F_TYPE> & op_problem,                                                      \
    solution_t<int, F_TYPE> & solution,                                                       \
    const rmm::device_uvector<int>& variables_to_fix,                                         \
    relaxed_lp_settings_t& settings,                                                          \
    bound_presolve_t<int, F_TYPE>* bound_presolve,                                            \
    bool check_fixed_assignment_feasibility,                                                  \
    bool use_integer_fixed_problem);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
