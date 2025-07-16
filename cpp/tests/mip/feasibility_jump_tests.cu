/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include <linear_programming/utilities/problem_checking.cuh>
#include <mip/feasibility_jump/feasibility_jump.cuh>
#include <mip/solution/solution.cuh>
#include <mip/solver_context.cuh>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

struct fj_tweaks_t {
  double objective_weight = 0;
};

struct fj_state_t {
  detail::solution_t<int, double> solution;
  std::vector<double> solution_vector;
  int minimums;
  double incumbent_objective;
  double incumbent_violation;
};

// Helper function to setup MIP solver and run FJ with given settings and initial solution
static fj_state_t run_fj(std::string test_instance,
                         const detail::fj_settings_t& fj_settings,
                         fj_tweaks_t tweaks                   = {},
                         std::vector<double> initial_solution = {})
{
  const raft::handle_t handle_{};
  std::cout << "Running: " << test_instance << std::endl;

  auto path = cuopt::test::get_rapids_dataset_root_dir() + ("/mip/" + test_instance);
  cuopt::mps_parser::mps_data_model_t<int, double> mps_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);

  init_handler(op_problem.get_handle_ptr());
  // run the problem constructor of MIP, so that we do bounds standardization
  detail::problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();
  detail::pdhg_solver_t<int, double> pdhg_solver(problem.handle_ptr, problem);
  detail::pdlp_initial_scaling_strategy_t<int, double> scaling(&handle_,
                                                               problem,
                                                               10,
                                                               1.0,
                                                               pdhg_solver,
                                                               problem.reverse_coefficients,
                                                               problem.reverse_offsets,
                                                               problem.reverse_constraints,
                                                               true);

  auto settings       = mip_solver_settings_t<int, double>{};
  settings.time_limit = 30.;
  auto timer          = cuopt::timer_t(30);
  detail::mip_solver_t<int, double> solver(problem, settings, scaling, timer);

  detail::solution_t<int, double> solution(*solver.context.problem_ptr);
  if (initial_solution.size() > 0) {
    expand_device_copy(solution.assignment, initial_solution, solution.handle_ptr->get_stream());
  } else {
    thrust::fill(solution.handle_ptr->get_thrust_policy(),
                 solution.assignment.begin(),
                 solution.assignment.end(),
                 0.0);
  }
  solution.clamp_within_bounds();

  detail::fj_t<int, double> fj(solver.context, fj_settings);
  fj.reset_weights(solution.handle_ptr->get_stream(), 1.);
  fj.objective_weight.set_value_async(tweaks.objective_weight, solution.handle_ptr->get_stream());
  solution.handle_ptr->sync_stream();

  fj.solve(solution);
  auto solution_vector = host_copy(solution.assignment, solution.handle_ptr->get_stream());

  return {solution,
          solution_vector,
          fj.climbers[0]->local_minimums_reached.value(solution.handle_ptr->get_stream()),
          fj.climbers[0]->incumbent_objective.value(solution.handle_ptr->get_stream()),
          fj.climbers[0]->violation_score.value(solution.handle_ptr->get_stream())};
}

// FJ had a bug causing objective/violation values to explode in magnitude in certain scenarios.
// Ensure this is fixed on instances that historically triggered it.
static bool run_fj_check_no_obj_runoff(std::string test_instance)
{
  detail::fj_settings_t fj_settings;
  fj_settings.time_limit             = 30.;
  fj_settings.mode                   = detail::fj_mode_t::EXIT_NON_IMPROVING;
  fj_settings.n_of_minimums_for_exit = 20000 * 1000;
  fj_settings.update_weights         = true;
  fj_settings.feasibility_run        = false;
  fj_settings.termination     = detail::fj_termination_flags_t::FJ_TERMINATION_ITERATION_LIMIT;
  fj_settings.iteration_limit = 20000;

  auto state = run_fj(test_instance, fj_settings);

  // ensure that the objective and the violation in the FJ state are not too large (<1e60)
  EXPECT_LE(state.incumbent_violation, 1e60) << "FJ violation too large";
  EXPECT_LE(state.incumbent_objective, 1e60) << "FJ objective too large";

  return true;
}

static bool run_fj_check_objective(std::string test_instance, int iter_limit, double obj_target)
{
  // tolerance
  obj_target += 1e-4;

  detail::fj_settings_t fj_settings;
  fj_settings.time_limit             = 30.;
  fj_settings.mode                   = detail::fj_mode_t::EXIT_NON_IMPROVING;
  fj_settings.n_of_minimums_for_exit = 20000 * 1000;
  fj_settings.update_weights         = true;
  fj_settings.feasibility_run        = obj_target == +std::numeric_limits<double>::infinity();
  fj_settings.termination     = detail::fj_termination_flags_t::FJ_TERMINATION_ITERATION_LIMIT;
  fj_settings.iteration_limit = iter_limit;

  auto state     = run_fj(test_instance, fj_settings);
  auto& solution = state.solution;

  CUOPT_LOG_DEBUG("%s: Solution generated with FJ: is_feasible %d, objective %g (raw %g)",
                  test_instance.c_str(),
                  solution.get_feasible(),
                  solution.get_user_objective(),
                  solution.get_objective());

  EXPECT_TRUE(solution.get_feasible()) << test_instance << " is unexpectedly infeasible";
  EXPECT_LE(solution.get_user_objective(), obj_target)
    << test_instance << " objective " << solution.get_user_objective() << " exceeds target "
    << obj_target;

  return !solution.get_feasible() ? false : solution.get_user_objective() <= obj_target;
}

static bool run_fj_check_feasible(std::string test_instance)
{
  detail::fj_settings_t fj_settings;
  fj_settings.time_limit             = 30.;
  fj_settings.mode                   = detail::fj_mode_t::EXIT_NON_IMPROVING;
  fj_settings.n_of_minimums_for_exit = 20000 * 1000;
  fj_settings.update_weights         = true;
  fj_settings.feasibility_run        = false;
  fj_settings.termination     = detail::fj_termination_flags_t::FJ_TERMINATION_ITERATION_LIMIT;
  fj_settings.iteration_limit = 25000;

  auto state     = run_fj(test_instance, fj_settings);
  auto& solution = state.solution;

  bool previous_feasible = solution.get_feasible();
  double previous_obj    = solution.get_user_objective();

  EXPECT_TRUE(previous_feasible) << "Initial solution is unexpectedly infeasible";

  // again but with very large obj weight to force FJ into the infeasible region
  fj_tweaks_t tweaks;
  tweaks.objective_weight = 1e6;
  auto new_state          = run_fj(test_instance, fj_settings, tweaks, state.solution_vector);
  auto& new_solution      = new_state.solution;

  CUOPT_LOG_DEBUG("%s: Solution generated with FJ: is_feasible %d, objective %g (raw %g)",
                  test_instance.c_str(),
                  new_solution.get_feasible(),
                  new_solution.get_user_objective(),
                  new_solution.get_objective());

  // TODO: check neither worsens nor defeasibizes the solution
  EXPECT_TRUE(new_solution.get_feasible() == previous_feasible) << "FJ feasibility lost";
  EXPECT_LE(new_solution.get_user_objective(), previous_obj) << "FJ objective worsened";

  return true;
}

class MIPSolveParametricTest : public testing::TestWithParam<std::tuple<std::string, double, int>> {
};

TEST_P(MIPSolveParametricTest, feasibility_jump_obj_test)
{
  auto [instance, obj_target, iter_limit] = GetParam();
  EXPECT_TRUE(run_fj_check_objective(instance, iter_limit, obj_target));
}

INSTANTIATE_TEST_SUITE_P(
  MIPSolveTest,
  MIPSolveParametricTest,
  testing::Values(std::make_tuple("50v-10.mps", 7800, 100000),
                  std::make_tuple("fiball.mps", 140, 25000),
                  std::make_tuple("gen-ip054.mps", 7500, 20000),
                  std::make_tuple("sct2.mps", 100, 50000),
                  std::make_tuple("uccase9.mps", 4000000, 50000),
                  // unstable, prone to failure on slight weight changes
                  // std::make_tuple("drayage-25-23.mps", 300000, 50000),
                  std::make_tuple("tr12-30.mps", 300000, 50000),
                  std::make_tuple("neos-3004026-krka.mps",
                                  +std::numeric_limits<double>::infinity(),
                                  35000),  // feasibility
                  // std::make_tuple("nursesched-medium-hint03.mps", 12000, 50000), // too large
                  std::make_tuple("ns1208400.mps", 2, 60000),
                  std::make_tuple("gmu-35-50.mps", -2300000, 25000),
                  std::make_tuple("n2seq36q.mps", 158800, 25000),
                  std::make_tuple("seymour1.mps", 440, 50000),
                  std::make_tuple("rmatr200-p5.mps", 7000, 10000),
                  std::make_tuple("cvs16r128-89.mps", -50, 10000)
// TEMPORARY: occasional cusparse transpose issues on ARM in CI
#ifndef __aarch64__
                    ,
                  std::make_tuple("thor50dday.mps", 250000, 1000)
#endif
                    ));

TEST(mip_solve, feasibility_jump_feas_test)
{
  for (const auto& instance : {"tr12-30.mps",
                               "sct2.mps"
#ifndef __aarch64__
                               ,
                               "thor50dday.mps"
#endif
       }) {
    run_fj_check_feasible(instance);
  }
}

TEST(mip_solve, feasibility_jump_obj_runoff_test)
{
  for (const auto& instance : {"minrep_inf.mps", "sct2.mps", "uccase9.mps",
                               /*"buildingenergy.mps"*/}) {
    run_fj_check_no_obj_runoff(instance);
  }
}

}  // namespace cuopt::linear_programming::test
