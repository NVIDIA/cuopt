/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "feasibility_jump/feasibility_jump.cuh"

#include <mip/mip_constants.hpp>
#include "diversity/diversity_manager.cuh"
#include "local_search/local_search.cuh"
#include "local_search/rounding/simple_rounding.cuh"
#include "solver.cuh"

#include <dual_simplex/branch_and_bound.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solve.hpp>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>

#include <future>
#include <memory>
#include <thread>

namespace cuopt::linear_programming::detail {

// This serves as both a warm up but also a mandatory initial call to setup cuSparse and cuBLAS
static void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

template <typename i_t, typename f_t>
mip_solver_t<i_t, f_t>::mip_solver_t(const problem_t<i_t, f_t>& op_problem,
                                     const mip_solver_settings_t<i_t, f_t>& solver_settings,
                                     pdlp_initial_scaling_strategy_t<i_t, f_t>& scaling,
                                     timer_t timer)
  : op_problem_(op_problem),
    solver_settings_(solver_settings),
    context(op_problem.handle_ptr,
            const_cast<problem_t<i_t, f_t>*>(&op_problem),
            solver_settings,
            scaling),
    timer_(timer)
{
  init_handler(op_problem.handle_ptr);
}

template <typename i_t, typename f_t>
struct branch_and_bound_solution_helper_t {
  branch_and_bound_solution_helper_t(population_t<i_t, f_t>* population,
                                     dual_simplex::simplex_solver_settings_t<i_t, f_t>& settings)
    : population_ptr(population), settings_(settings) {};

  void solution_callback(std::vector<f_t>& solution, f_t objective)
  {
    population_ptr->add_external_solution(solution, objective);
  }

  void preempt_heuristic_solver() { population_ptr->preempt_heuristic_solver(); }
  population_t<i_t, f_t>* population_ptr;
  dual_simplex::simplex_solver_settings_t<i_t, f_t>& settings_;
};

template <typename i_t, typename f_t>
solution_t<i_t, f_t> mip_solver_t<i_t, f_t>::run_solver()
{
  if (context.settings.get_mip_callbacks().size() > 0) {
    for (auto callback : context.settings.get_mip_callbacks()) {
      callback->template setup<f_t>(context.problem_ptr->original_problem_ptr->get_n_variables());
    }
  }
  //  we need to keep original problem const
  cuopt_assert(context.problem_ptr != nullptr, "invalid problem pointer");
  context.problem_ptr->tolerances = context.settings.get_tolerances();
  cuopt_expects(context.problem_ptr->preprocess_called,
                error_type_t::RuntimeError,
                "preprocess_problem should be called before running the solver");

  if (context.problem_ptr->empty) {
    CUOPT_LOG_INFO("Problem fully reduced in presolve");
    solution_t<i_t, f_t> sol(*context.problem_ptr);
    sol.set_problem_fully_reduced();
    context.problem_ptr->post_process_solution(sol);
    return sol;
  }

  diversity_manager_t<i_t, f_t> dm(context);
  dm.timer              = timer_;
  bool presolve_success = dm.run_presolve(timer_.remaining_time());
  if (!presolve_success) {
    CUOPT_LOG_INFO("Problem proven infeasible in presolve");
    solution_t<i_t, f_t> sol(*context.problem_ptr);
    sol.set_problem_fully_reduced();
    context.problem_ptr->post_process_solution(sol);
    return sol;
  }
  if (context.problem_ptr->empty) {
    CUOPT_LOG_INFO("Problem full reduced in presolve");
    solution_t<i_t, f_t> sol(*context.problem_ptr);
    sol.set_problem_fully_reduced();
    context.problem_ptr->post_process_solution(sol);
    return sol;
  }

  namespace dual_simplex = cuopt::linear_programming::dual_simplex;
  std::future<dual_simplex::mip_status_t> branch_and_bound_status_future;
  dual_simplex::user_problem_t<i_t, f_t> branch_and_bound_problem;
  dual_simplex::simplex_solver_settings_t<i_t, f_t> branch_and_bound_settings;
  std::unique_ptr<dual_simplex::branch_and_bound_t<i_t, f_t>> branch_and_bound;
  branch_and_bound_solution_helper_t solution_helper(dm.get_population_pointer(),
                                                     branch_and_bound_settings);
  dual_simplex::mip_solution_t<i_t, f_t> branch_and_bound_solution(1);

  if (!context.settings.heuristics_only) {
    // Convert the presolved problem to dual_simplex::user_problem_t
    op_problem_.get_host_user_problem(branch_and_bound_problem);
    // Resize the solution now that we know the number of columns/variables
    branch_and_bound_solution.resize(branch_and_bound_problem.num_cols);

    // Fill in the settings for branch and bound
    branch_and_bound_settings.time_limit           = timer_.remaining_time();
    branch_and_bound_settings.print_presolve_stats = false;
    branch_and_bound_settings.absolute_mip_gap_tol = context.settings.tolerances.absolute_mip_gap;
    branch_and_bound_settings.relative_mip_gap_tol = context.settings.tolerances.relative_mip_gap;
    branch_and_bound_settings.integer_tol = context.settings.tolerances.integrality_tolerance;

    if (context.settings.num_cpu_threads != -1) {
      branch_and_bound_settings.num_threads = std::max(1, context.settings.num_cpu_threads);
    }

    // Set the branch and bound -> primal heuristics callback
    branch_and_bound_settings.solution_callback =
      std::bind(&branch_and_bound_solution_helper_t<i_t, f_t>::solution_callback,
                &solution_helper,
                std::placeholders::_1,
                std::placeholders::_2);
    branch_and_bound_settings.heuristic_preemption_callback = std::bind(
      &branch_and_bound_solution_helper_t<i_t, f_t>::preempt_heuristic_solver, &solution_helper);
    // Create the branch and bound object
    branch_and_bound = std::make_unique<dual_simplex::branch_and_bound_t<i_t, f_t>>(
      branch_and_bound_problem, branch_and_bound_settings);

    // Set the primal heuristics -> branch and bound callback
    context.problem_ptr->branch_and_bound_callback =
      std::bind(&dual_simplex::branch_and_bound_t<i_t, f_t>::set_new_solution,
                branch_and_bound.get(),
                std::placeholders::_1);

    // Fork a thread for branch and bound
    // std::async and std::future allow us to get the return value of bb::solve()
    // without having to manually manage the thread
    // std::future.get() performs a join() operation to wait until the return status is available
    branch_and_bound_status_future = std::async(std::launch::async,
                                                &dual_simplex::branch_and_bound_t<i_t, f_t>::solve,
                                                branch_and_bound.get(),
                                                std::ref(branch_and_bound_solution));
  }

  // Start the primal heuristics
  auto sol = dm.run_solver();
  if (!context.settings.heuristics_only) {
    // Wait for the branch and bound to finish
    auto bb_status = branch_and_bound_status_future.get();
    if (branch_and_bound_solution.lower_bound > -std::numeric_limits<f_t>::infinity()) {
      context.stats.solution_bound =
        context.problem_ptr->get_user_obj_from_solver_obj(branch_and_bound_solution.lower_bound);
    }
    if (bb_status == dual_simplex::mip_status_t::INFEASIBLE) { sol.set_problem_fully_reduced(); }
    context.stats.num_nodes              = branch_and_bound_solution.nodes_explored;
    context.stats.num_simplex_iterations = branch_and_bound_solution.simplex_iterations;
  }
  sol.compute_feasibility();
  rmm::device_scalar<i_t> is_feasible(sol.handle_ptr->get_stream());
  sol.test_variable_bounds(true, is_feasible.data());
  // test_variable_bounds clears is_feasible if the test is failed
  if (!is_feasible.value(sol.handle_ptr->get_stream())) {
    CUOPT_LOG_ERROR(
      "Solution is not feasible due to variable bounds, returning infeasible solution!");
    context.problem_ptr->post_process_solution(sol);
    return sol;
  }
  context.problem_ptr->post_process_solution(sol);
  return sol;
}

// Original feasibility jump has only double
#if MIP_INSTANTIATE_FLOAT
template class mip_solver_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class mip_solver_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
