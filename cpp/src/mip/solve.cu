/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <mip/mip_constants.hpp>
#include <mip/presolve/trivial_presolve.cuh>
#include <mip/solver.cuh>
#include <mip/utils.cuh>

#include <linear_programming/initial_scaling_strategy/initial_scaling.cuh>
#include <linear_programming/pdlp.cuh>
#include <linear_programming/restart_strategy/pdlp_restart_strategy.cuh>
#include <linear_programming/step_size_strategy/adaptive_step_size_strategy.hpp>
#include <linear_programming/utilities/logger_init.hpp>
#include <linear_programming/utilities/problem_checking.cuh>
#include <linear_programming/utils.cuh>
#include <utilities/error.hpp>
#include <utilities/timer.hpp>

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/solve.hpp>

#include <mps_parser/mps_data_model.hpp>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/common/nvtx.hpp>
#include <raft/core/handle.hpp>

namespace cuopt::linear_programming {

// This serves as both a warm up but also a mandatory initial call to setup cuSparse and cuBLAS
static void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

static void setup_device_symbols()
{
  raft::common::nvtx::range fun_scope("Setting device symbol");
  detail::set_adaptive_step_size_hyper_parameters();
  detail::set_restart_hyper_parameters();
  detail::set_pdlp_hyper_parameters();
}

template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> run_mip(detail::problem_t<i_t, f_t>& problem,
                                 mip_solver_settings_t<i_t, f_t> const& settings)
{
  const f_t time_limit =
    settings.get_time_limit() == 0 ? std::numeric_limits<f_t>::max() : settings.get_time_limit();
  if (settings.get_heuristics_only() && time_limit == std::numeric_limits<f_t>::max()) {
    CUOPT_LOG_ERROR("Time limit cannot be infinity when heuristics only is set");
    cuopt_expects(false,
                  error_type_t::RuntimeError,
                  "Time limit cannot be infinity when heuristics only is set");
  }
  auto timer                       = cuopt::timer_t(time_limit);
  auto constexpr const running_mip = true;

  pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
  pdlp_hyper_params::update_step_size_on_initial_solution     = true;
  // if the input problem is empty: early exit
  if (problem.empty) {
    detail::solution_t<i_t, f_t> solution(problem);
    problem.post_process_solution(solution);
    solution.compute_objective();  // just to ensure h_user_obj is set
    auto stats           = solver_stats_t<i_t, f_t>{};
    stats.solution_bound = solution.get_user_objective();
    return solution.get_solution(true, stats);
  }
  // problem contains unpreprocessed data
  detail::problem_t<i_t, f_t> scaled_problem(problem);
  CUOPT_LOG_INFO("Solving a problem with %d constraints %d variables (%d integers) and %d nonzeros",
                 problem.n_constraints,
                 problem.n_variables,
                 problem.n_integer_vars,
                 problem.nnz);
  CUOPT_LOG_INFO("Objective offset %f scaling_factor %f",
                 problem.presolve_data.objective_offset,
                 problem.presolve_data.objective_scaling_factor);
  cuopt_assert(problem.original_problem_ptr->get_n_variables() == scaled_problem.n_variables,
               "Size mismatch");
  cuopt_assert(problem.original_problem_ptr->get_n_constraints() == scaled_problem.n_constraints,
               "Size mismatch");
  detail::pdhg_solver_t<i_t, f_t> pdhg_solver(scaled_problem.handle_ptr, scaled_problem);
  detail::pdlp_initial_scaling_strategy_t<i_t, f_t> scaling(
    scaled_problem.handle_ptr,
    scaled_problem,
    pdlp_hyper_params::default_l_inf_ruiz_iterations,
    (f_t)pdlp_hyper_params::default_alpha_pock_chambolle_rescaling,
    pdhg_solver,
    scaled_problem.reverse_coefficients,
    scaled_problem.reverse_offsets,
    scaled_problem.reverse_constraints,
    running_mip);

  cuopt_func_call(auto saved_problem = scaled_problem);
  if (settings.get_mip_scaling()) { scaling.scale_problem(); }
  if (settings.has_initial_solution()) { scaling.scale_primal(settings.get_initial_solution()); }
  // only call preprocess on scaled problem, so we can compute feasibility on the original problem
  scaled_problem.preprocess_problem();
  // cuopt_func_call((check_scaled_problem<i_t, f_t>(scaled_problem, saved_problem)));
  detail::trivial_presolve(scaled_problem);
  detail::mip_solver_t<i_t, f_t> solver(scaled_problem, settings, scaling, timer);
  auto scaled_sol                 = solver.run_solver();
  bool is_feasible_before_scaling = scaled_sol.get_feasible();
  scaled_sol.problem_ptr          = &problem;
  if (settings.get_mip_scaling()) { scaling.unscale_solutions(scaled_sol); }
  // at this point we need to compute the feasibility on the original problem not the presolved one
  bool is_feasible_after_unscaling = scaled_sol.compute_feasibility();
  if (!scaled_problem.empty && is_feasible_before_scaling != is_feasible_after_unscaling) {
    CUOPT_LOG_WARN(
      "The feasibility does not match on scaled and unscaled problems. To overcome this issue, "
      "please provide a more numerically stable problem.");
  }
  // If the reduced problem is empty and feasible, then this solution is optimal
  // Set the solution bound appropriately
  if (scaled_problem.empty) {
    solver.get_solver_stats().solution_bound = scaled_sol.get_user_objective();
  }
  auto sol = scaled_sol.get_solution(is_feasible_before_scaling || is_feasible_after_unscaling,
                                     solver.get_solver_stats());
  detail::print_solution(sol.get_solution(), scaled_problem.handle_ptr);
  return sol;
}

template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip(optimization_problem_t<i_t, f_t>& op_problem,
                                   mip_solver_settings_t<i_t, f_t> const& settings)
{
  // Create log stream for file logging and add it to default logger
  init_logger_t log(settings.get_log_file(), settings.get_log_to_console());
#if CUOPT_LOG_ACTIVE_LEVEL >= RAPIDS_LOGGER_LOG_LEVEL_INFO
  cuopt::default_logger().set_pattern("%v");
#endif
  // Init libraies before to not include it in solve time
  // This needs to be called before pdlp is initialized
  init_handler(op_problem.get_handle_ptr());

  raft::common::nvtx::range fun_scope("Running solver");

  // This is required as user might forget to set some fields
  problem_checking_t<i_t, f_t>::check_problem_representation(op_problem);
  problem_checking_t<i_t, f_t>::check_initial_solution_representation(op_problem, settings);

  // have solve, problem, solution, utils etc. in common dir
  detail::problem_t<i_t, f_t> problem(op_problem);

  // this is for PDLP, i think this should be part of pdlp solver
  setup_device_symbols();

  return run_mip(problem, settings);
}

template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip(
  raft::handle_t const* handle_ptr,
  const cuopt::mps_parser::mps_data_model_t<i_t, f_t>& mps_data_model,
  mip_solver_settings_t<i_t, f_t> const& settings)
{
  auto op_problem = mps_data_model_to_optimization_problem(handle_ptr, mps_data_model);
  return solve_mip(op_problem, settings);
}

#define INSTANTIATE(F_TYPE)                                                 \
  template mip_solution_t<int, F_TYPE> solve_mip(                           \
    optimization_problem_t<int, F_TYPE>& op_problem,                        \
    mip_solver_settings_t<int, F_TYPE> const& settings);                    \
                                                                            \
  template mip_solution_t<int, F_TYPE> solve_mip(                           \
    raft::handle_t const* handle_ptr,                                       \
    const cuopt::mps_parser::mps_data_model_t<int, F_TYPE>& mps_data_model, \
    mip_solver_settings_t<int, F_TYPE> const& settings);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

}  // namespace cuopt::linear_programming
