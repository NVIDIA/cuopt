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

#include <linear_programming/pdlp.cuh>
#include <linear_programming/pdlp_constants.hpp>
#include <linear_programming/solve.cuh>
#include <linear_programming/utils.cuh>
#include <mps_parser.hpp>
#include "utilities/pdlp_test_utilities.cuh"

#include <utilities/base_fixture.hpp>
#include <utilities/common_utils.hpp>

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <mip/problem/problem.cuh>
#include <mps_parser/parser.hpp>

#include <utilities/copy_helpers.hpp>
#include <utilities/error.hpp>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/logical.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <vector>

namespace cuopt::linear_programming::test {

constexpr double afiro_primal_objective = -464;

// Accept a 1% error
static bool is_incorrect_objective(double reference, double objective)
{
  if (reference == 0) { return std::abs(objective) > 0.01; }
  if (objective == 0) { return std::abs(reference) > 0.01; }
  return std::abs((reference - objective) / reference) > 0.01;
}

TEST(pdlp_class, run_double)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  auto solver_settings   = pdlp_solver_settings_t<int, double>{};
  solver_settings.method = cuopt::linear_programming::method_t::PDLP;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution.get_additional_termination_information().primal_objective));
}

TEST(pdlp_class, run_double_very_low_accuracy)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings =
    cuopt::linear_programming::pdlp_solver_settings_t<int, double>{};
  // With all 0 afiro with return an error
  // Setting absolute tolerance to the minimal value of 1e-12 will make it work
  settings.tolerances.absolute_dual_tolerance   = settings.minimal_absolute_tolerance;
  settings.tolerances.relative_dual_tolerance   = 0.0;
  settings.tolerances.absolute_primal_tolerance = settings.minimal_absolute_tolerance;
  settings.tolerances.relative_primal_tolerance = 0.0;
  settings.tolerances.absolute_gap_tolerance    = settings.minimal_absolute_tolerance;
  settings.tolerances.relative_gap_tolerance    = 0.0;
  settings.method                               = cuopt::linear_programming::method_t::PDLP;

  optimization_problem_solution_t<int, double> solution = solve_lp(&handle_, op_problem, settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution.get_additional_termination_information().primal_objective));
}

TEST(pdlp_class, run_double_initial_solution)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  std::vector<double> inital_primal_sol(op_problem.get_n_variables());
  std::fill(inital_primal_sol.begin(), inital_primal_sol.end(), 1.0);
  op_problem.set_initial_primal_solution(inital_primal_sol.data(), inital_primal_sol.size());

  auto solver_settings   = pdlp_solver_settings_t<int, double>{};
  solver_settings.method = cuopt::linear_programming::method_t::PDLP;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution.get_additional_termination_information().primal_objective));
}

TEST(pdlp_class, run_iteration_limit)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings =
    cuopt::linear_programming::pdlp_solver_settings_t<int, double>{};

  settings.iteration_limit = 10;
  // To make sure it doesn't return before the iteration limit
  settings.set_optimality_tolerance(0);
  settings.method = cuopt::linear_programming::method_t::PDLP;

  optimization_problem_solution_t<int, double> solution = solve_lp(&handle_, op_problem, settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_ITERATION_LIMIT);
  // By default we would return all 0, we now return what we currently have so not all 0
  EXPECT_FALSE(thrust::all_of(handle_.get_thrust_policy(),
                              solution.get_primal_solution().begin(),
                              solution.get_primal_solution().end(),
                              thrust::placeholders::_1 == 0.0));
}

TEST(pdlp_class, run_time_limit)
{
  const raft::handle_t handle_{};
  auto path = make_path_absolute("linear_programming/savsched1/savsched1.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path);

  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings =
    cuopt::linear_programming::pdlp_solver_settings_t<int, double>{};

  // 200 ms
  constexpr double time_limit_seconds = 0.2;
  settings.time_limit                 = time_limit_seconds;
  // To make sure it doesn't return before the time limit
  settings.set_optimality_tolerance(0);
  settings.method = cuopt::linear_programming::method_t::PDLP;

  optimization_problem_solution_t<int, double> solution = solve_lp(&handle_, op_problem, settings);

  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_TIME_LIMIT);
  // By default we would return all 0, we now return what we currently have so not all 0
  EXPECT_FALSE(thrust::all_of(handle_.get_thrust_policy(),
                              solution.get_primal_solution().begin(),
                              solution.get_primal_solution().end(),
                              thrust::placeholders::_1 == 0.0));
  // Check that indeed it didn't run for more than x time
  EXPECT_TRUE(solution.get_additional_termination_information().solve_time <
              (time_limit_seconds * 5) * 1000);
}

TEST(pdlp_class, run_sub_mittleman)
{
  std::vector<std::pair<std::string,  // Instance name
                        double>>      // Expected objective value
    instances{{"graph40-40", -300.0},
              {"ex10", 100.0003411893773},
              {"datt256_lp", 255.9992298290425},
              {"woodlands09", 0.0},
              {"savsched1", 217.4054085795689},
              // {"nug08-3rd", 214.0141488989151}, // TODO: Fix this instance
              {"qap15", 1040.999546647414},
              {"scpm1", 413.7787723060584},
              // {"neos3", 27773.54059633068}, // TODO: Fix this instance
              {"a2864", -282.9962521965164}};

  for (const auto& entry : instances) {
    const auto& name                    = entry.first;
    const auto expected_objective_value = entry.second;

    auto path = make_path_absolute("linear_programming/" + name + "/" + name + ".mps");
    cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
      cuopt::mps_parser::parse_mps<int, double>(path);

    // Testing for each solver_mode is ok as it's parsing that is the bottleneck here, not
    // solving
    auto solver_mode_list = {
      cuopt::linear_programming::pdlp_solver_mode_t::Stable2,
      cuopt::linear_programming::pdlp_solver_mode_t::Methodical1,
      cuopt::linear_programming::pdlp_solver_mode_t::Fast1,
    };
    for (auto solver_mode : solver_mode_list) {
      auto settings             = pdlp_solver_settings_t<int, double>{};
      settings.pdlp_solver_mode = solver_mode;
      const raft::handle_t handle_{};
      optimization_problem_solution_t<int, double> solution =
        solve_lp(&handle_, op_problem, settings);
      EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
      EXPECT_FALSE(
        is_incorrect_objective(expected_objective_value,
                               solution.get_additional_termination_information().primal_objective));
      test_objective_sanity(op_problem,
                            solution.get_primal_solution(),
                            solution.get_additional_termination_information().primal_objective);
      test_constraint_sanity(op_problem, solution);
    }
  }
}

constexpr double initial_step_size_afiro     = 1.4893;
constexpr double initial_primal_weight_afiro = 0.0141652;
constexpr double factor_tolerance            = 1e-4f;

// Should be added to google test
#define EXPECT_NOT_NEAR(val1, val2, abs_error) \
  EXPECT_FALSE((std::abs((val1) - (val2)) <= (abs_error)))

TEST(pdlp_class, initial_solution_test)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> mps_data_model =
    cuopt::mps_parser::parse_mps<int, double>(path);

  auto op_problem = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
    &handle_, mps_data_model);
  cuopt::linear_programming::detail::problem_t<int, double> problem(op_problem);

  auto solver_settings = pdlp_solver_settings_t<int, double>{};
  // We are just testing initial scaling on initial solution scheme so we don't care about solver
  solver_settings.iteration_limit = 0;
  solver_settings.method          = cuopt::linear_programming::method_t::PDLP;
  // Empty call solve to set the parameters and init the handler since calling pdlp object directly
  // doesn't
  solver_settings.pdlp_solver_mode = cuopt::linear_programming::pdlp_solver_mode_t::Methodical1;
  solve_lp(op_problem, solver_settings);
  EXPECT_EQ(cuopt::linear_programming::pdlp_hyper_params::initial_step_size_scaling, 1);
  EXPECT_EQ(cuopt::linear_programming::pdlp_hyper_params::default_l_inf_ruiz_iterations, 5);
  EXPECT_TRUE(cuopt::linear_programming::pdlp_hyper_params::do_pock_chambolle_scaling);
  EXPECT_TRUE(cuopt::linear_programming::pdlp_hyper_params::do_ruiz_scaling);
  EXPECT_EQ(cuopt::linear_programming::pdlp_hyper_params::default_alpha_pock_chambolle_rescaling,
            1.0);

  EXPECT_FALSE(cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution);
  EXPECT_FALSE(
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution);

  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
  }

  // First add an initial primal then dual, then both, which shouldn't influence the values as the
  // scale on initial option is not toggled
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
  }
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
  }
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
  }

  // Toggle the scale on initial solution while not providing should yield the same
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = true;
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = false;
  }
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
  }
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution     = true;
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution     = false;
  }

  // Asking for initial scaling on step size with initial solution being only primal or only dual
  // should not break but not modify the step size
  {
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = false;
  }
  {
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = false;
  }

  // Asking for initial scaling on primal weight with initial solution being only primal or only
  // dual should *not* break but the primal weight should not change
  {
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    solver.run_solver(start_solver);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
  }
  {
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
  }

  // All 0 solution when given an initial primal and dual with scale on the step size should not
  // break but not change primal weight and step size
  {
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 0);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 0);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = false;
  }

  // All 0 solution when given an initial primal and/or dual with scale on the primal weight is
  // *not* an error but should not change primal weight and step size
  {
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 0);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    solver.run_solver(start_solver);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
  }
  {
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 0);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
  }
  {
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 0);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 0);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
  }

  // A non-all-0 vector for both initial primal and dual set should trigger a modification in primal
  // weight and step size
  {
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NOT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
  }
  {
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    EXPECT_NOT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = false;
  }
  {
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution     = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    EXPECT_NOT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NOT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution     = false;
  }
}

TEST(pdlp_class, initial_primal_weight_step_size_test)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> mps_data_model =
    cuopt::mps_parser::parse_mps<int, double>(path);

  auto op_problem = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
    &handle_, mps_data_model);
  cuopt::linear_programming::detail::problem_t<int, double> problem(op_problem);

  auto solver_settings = pdlp_solver_settings_t<int, double>{};
  // We are just testing initial scaling on initial solution scheme so we don't care about solver
  solver_settings.iteration_limit = 0;
  solver_settings.method          = cuopt::linear_programming::method_t::PDLP;
  // Select the default/legacy solver with no action upon the initial scaling on initial solution
  solver_settings.pdlp_solver_mode = cuopt::linear_programming::pdlp_solver_mode_t::Methodical1;
  EXPECT_FALSE(cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution);
  EXPECT_FALSE(
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution);

  // Check setting an initial primal weight and step size
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver                           = std::chrono::high_resolution_clock::now();
    constexpr double test_initial_step_size     = 1.0;
    constexpr double test_initial_primal_weight = 2.0;
    solver.set_initial_primal_weight(test_initial_primal_weight);
    solver.set_initial_step_size(test_initial_step_size);
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_EQ(test_initial_step_size, solver.get_step_size_h());
    EXPECT_EQ(test_initial_primal_weight, solver.get_primal_weight_h());
  }

  // Check that after setting an initial step size and primal weight, the computed one when adding
  // an initial primal / dual is indeed different
  {
    // Launching without an inital step size / primal weight and query the value
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution     = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    const double previous_step_size     = solver.get_step_size_h();
    const double previous_primal_weight = solver.get_primal_weight_h();

    // Start again but with an initial and check the impact
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver2(problem, solver_settings);
    start_solver                                = std::chrono::high_resolution_clock::now();
    constexpr double test_initial_step_size     = 1.0;
    constexpr double test_initial_primal_weight = 2.0;
    solver2.set_initial_primal_weight(test_initial_primal_weight);
    solver2.set_initial_step_size(test_initial_step_size);
    solver2.set_initial_primal_solution(d_initial_primal);
    solver2.set_initial_dual_solution(d_initial_dual);
    solver2.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    const double sovler2_step_size     = solver2.get_step_size_h();
    const double sovler2_primal_weight = solver2.get_primal_weight_h();
    EXPECT_NOT_NEAR(previous_step_size, sovler2_step_size, factor_tolerance);
    EXPECT_NOT_NEAR(previous_primal_weight, sovler2_primal_weight, factor_tolerance);

    // Again but with an initial k which should change the step size only, not the primal weight
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver3(problem, solver_settings);
    start_solver = std::chrono::high_resolution_clock::now();
    solver3.set_initial_primal_weight(test_initial_primal_weight);
    solver3.set_initial_step_size(test_initial_step_size);
    solver3.set_initial_primal_solution(d_initial_primal);
    solver3.set_initial_k(10000);
    solver3.set_initial_dual_solution(d_initial_dual);
    solver3.set_initial_dual_solution(d_initial_dual);
    solver3.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NOT_NEAR(sovler2_step_size, solver3.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(sovler2_primal_weight, solver3.get_primal_weight_h(), factor_tolerance);
  }
}

TEST(pdlp_class, initial_rhs_and_c)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> mps_data_model =
    cuopt::mps_parser::parse_mps<int, double>(path);

  auto op_problem = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
    &handle_, mps_data_model);
  cuopt::linear_programming::detail::problem_t<int, double> problem(op_problem);

  cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem);
  constexpr double test_initial_primal_factor = 1.0;
  constexpr double test_initial_dual_factor   = 2.0;
  solver.set_relative_dual_tolerance_factor(test_initial_dual_factor);
  solver.set_relative_primal_tolerance_factor(test_initial_primal_factor);

  EXPECT_EQ(solver.get_relative_dual_tolerance_factor(), test_initial_dual_factor);
  EXPECT_EQ(solver.get_relative_primal_tolerance_factor(), test_initial_primal_factor);
}

TEST(pdlp_class, per_constraint_test)
{
  /*
   * Define the following LP:
   * x1=0.01 <= 0
   * x2=0.01 <= 0
   * x3=0.1  <= 0
   *
   * With a tol of 0.1 per constraint will pass but the L2 version will not as L2 of primal residual
   * will be 0.1009
   */
  raft::handle_t handle;
  auto op_problem = optimization_problem_t<int, double>(&handle);

  std::vector<double> A_host           = {1.0, 1.0, 1.0};
  std::vector<int> indices_host        = {0, 1, 2};
  std::vector<int> offset_host         = {0, 1, 2, 3};
  std::vector<double> b_host           = {0.0, 0.0, 0.0};
  std::vector<double> h_initial_primal = {0.02, 0.03, 0.1};
  rmm::device_uvector<double> d_initial_primal(3, handle.get_stream());
  raft::copy(
    d_initial_primal.data(), h_initial_primal.data(), h_initial_primal.size(), handle.get_stream());

  op_problem.set_csr_constraint_matrix(A_host.data(),
                                       A_host.size(),
                                       indices_host.data(),
                                       indices_host.size(),
                                       offset_host.data(),
                                       offset_host.size());
  op_problem.set_constraint_lower_bounds(b_host.data(), b_host.size());
  op_problem.set_constraint_upper_bounds(b_host.data(), b_host.size());
  op_problem.set_objective_coefficients(b_host.data(), b_host.size());

  auto problem = cuopt::linear_programming::detail::problem_t<int, double>(op_problem);

  pdlp_solver_settings_t<int, double> solver_settings;
  solver_settings.tolerances.relative_primal_tolerance = 0;  // Shouldn't matter
  solver_settings.tolerances.absolute_primal_tolerance = 0.1;
  solver_settings.tolerances.relative_dual_tolerance   = 0;  // Shoudln't matter
  solver_settings.tolerances.absolute_dual_tolerance   = 0.1;
  solver_settings.method                               = cuopt::linear_programming::method_t::PDLP;

  // First solve without the per constraint and it should break
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);

    raft::copy(solver.pdhg_solver_.get_primal_solution().data(),
               d_initial_primal.data(),
               d_initial_primal.size(),
               handle.get_stream());

    auto& current_termination_strategy = solver.get_current_termination_strategy();
    pdlp_termination_status_t termination_average =
      current_termination_strategy.evaluate_termination_criteria(solver.pdhg_solver_,
                                                                 d_initial_primal,
                                                                 d_initial_primal,
                                                                 problem.combined_bounds,
                                                                 problem.objective_coefficients);

    EXPECT_TRUE(termination_average != pdlp_termination_status_t::Optimal);
  }
  {
    solver_settings.per_constraint_residual = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);

    raft::copy(solver.pdhg_solver_.get_primal_solution().data(),
               d_initial_primal.data(),
               d_initial_primal.size(),
               handle.get_stream());

    auto& current_termination_strategy = solver.get_current_termination_strategy();
    pdlp_termination_status_t termination_average =
      current_termination_strategy.evaluate_termination_criteria(solver.pdhg_solver_,
                                                                 d_initial_primal,
                                                                 d_initial_primal,
                                                                 problem.combined_bounds,
                                                                 problem.objective_coefficients);
    EXPECT_EQ(current_termination_strategy.get_convergence_information()
                .get_relative_linf_primal_residual()
                .value(handle.get_stream()),
              0.1);
  }
}

TEST(pdlp_class, best_primal_so_far_iteration)
{
  GTEST_SKIP() << "Skipping test: best_primal_so_far_iteration. Enable when ready to run.";
  const raft::handle_t handle1{};
  const raft::handle_t handle2{};

  auto path            = make_path_absolute("linear_programming/ns1687037/ns1687037.mps");
  auto solver_settings = pdlp_solver_settings_t<int, double>{};
  solver_settings.iteration_limit         = 3000;
  solver_settings.per_constraint_residual = true;
  solver_settings.method                  = cuopt::linear_programming::method_t::PDLP;

  cuopt::mps_parser::mps_data_model_t<int, double> op_problem1 =
    cuopt::mps_parser::parse_mps<int, double>(path);
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem2 =
    cuopt::mps_parser::parse_mps<int, double>(path);

  optimization_problem_solution_t<int, double> solution1 =
    solve_lp(&handle1, op_problem1, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  solver_settings.save_best_primal_so_far = true;
  optimization_problem_solution_t<int, double> solution2 =
    solve_lp(&handle2, op_problem2, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  EXPECT_TRUE(solution2.get_additional_termination_information().l2_primal_residual <
              solution1.get_additional_termination_information().l2_primal_residual);
}

TEST(pdlp_class, best_primal_so_far_time)
{
  GTEST_SKIP() << "Skipping test: best_primal_so_far_time. Enable when ready to run.";
  const raft::handle_t handle1{};
  const raft::handle_t handle2{};

  auto path                  = make_path_absolute("linear_programming/ns1687037/ns1687037.mps");
  auto solver_settings       = pdlp_solver_settings_t<int, double>{};
  solver_settings.time_limit = 2;
  solver_settings.per_constraint_residual = true;
  solver_settings.pdlp_solver_mode        = cuopt::linear_programming::pdlp_solver_mode_t::Stable1;
  solver_settings.method                  = cuopt::linear_programming::method_t::PDLP;

  cuopt::mps_parser::mps_data_model_t<int, double> op_problem1 =
    cuopt::mps_parser::parse_mps<int, double>(path);
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem2 =
    cuopt::mps_parser::parse_mps<int, double>(path);

  optimization_problem_solution_t<int, double> solution1 =
    solve_lp(&handle1, op_problem1, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  solver_settings.save_best_primal_so_far = true;
  optimization_problem_solution_t<int, double> solution2 =
    solve_lp(&handle2, op_problem2, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  EXPECT_TRUE(solution2.get_additional_termination_information().l2_primal_residual <
              solution1.get_additional_termination_information().l2_primal_residual);
}

TEST(pdlp_class, first_primal_feasible)
{
  GTEST_SKIP() << "Skipping test: first_primal_feasible. Enable when ready to run.";
  const raft::handle_t handle1{};
  const raft::handle_t handle2{};

  auto path            = make_path_absolute("linear_programming/ns1687037/ns1687037.mps");
  auto solver_settings = pdlp_solver_settings_t<int, double>{};
  solver_settings.iteration_limit         = 1000;
  solver_settings.per_constraint_residual = true;
  solver_settings.set_optimality_tolerance(1e-2);
  solver_settings.method = cuopt::linear_programming::method_t::PDLP;

  cuopt::mps_parser::mps_data_model_t<int, double> op_problem1 =
    cuopt::mps_parser::parse_mps<int, double>(path);
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem2 =
    cuopt::mps_parser::parse_mps<int, double>(path);

  optimization_problem_solution_t<int, double> solution1 =
    solve_lp(&handle1, op_problem1, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  solver_settings.first_primal_feasible = true;
  optimization_problem_solution_t<int, double> solution2 =
    solve_lp(&handle2, op_problem2, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  EXPECT_EQ(solution1.get_termination_status(), pdlp_termination_status_t::IterationLimit);
  EXPECT_EQ(solution2.get_termination_status(), pdlp_termination_status_t::PrimalFeasible);
}

TEST(pdlp_class, warm_start)
{
  std::vector<std::string> instance_names{"graph40-40",
                                          "ex10",
                                          "datt256_lp",
                                          "woodlands09",
                                          "savsched1",
                                          // "nug08-3rd", // TODO: Fix this instance
                                          "qap15",
                                          "scpm1",
                                          // "neos3", // TODO: Fix this instance
                                          "a2864"};
  for (auto instance_name : instance_names) {
    const raft::handle_t handle{};

    auto path =
      make_path_absolute("linear_programming/" + instance_name + "/" + instance_name + ".mps");
    auto solver_settings             = pdlp_solver_settings_t<int, double>{};
    solver_settings.pdlp_solver_mode = cuopt::linear_programming::pdlp_solver_mode_t::Stable2;
    solver_settings.set_optimality_tolerance(1e-2);
    solver_settings.detect_infeasibility = false;
    solver_settings.method               = cuopt::linear_programming::method_t::PDLP;

    cuopt::mps_parser::mps_data_model_t<int, double> mps_data_model =
      cuopt::mps_parser::parse_mps<int, double>(path);
    auto op_problem1 =
      cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
        &handle, mps_data_model);

    // Solving from scratch until 1e-2
    optimization_problem_solution_t<int, double> solution1 = solve_lp(op_problem1, solver_settings);

    // Solving until 1e-1 to use the result as a warm start
    solver_settings.set_optimality_tolerance(1e-1);
    auto op_problem2 =
      cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
        &handle, mps_data_model);
    optimization_problem_solution_t<int, double> solution2 = solve_lp(op_problem2, solver_settings);

    // Solving until 1e-2 using the previous state as a warm start
    solver_settings.set_optimality_tolerance(1e-2);
    auto op_problem3 =
      cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
        &handle, mps_data_model);
    solver_settings.set_pdlp_warm_start_data(solution2.get_pdlp_warm_start_data());
    optimization_problem_solution_t<int, double> solution3 = solve_lp(op_problem3, solver_settings);

    EXPECT_EQ(solution1.get_additional_termination_information().number_of_steps_taken,
              solution3.get_additional_termination_information().number_of_steps_taken +
                solution2.get_additional_termination_information().number_of_steps_taken);
  }
}

TEST(dual_simplex, afiro)
{
  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings =
    cuopt::linear_programming::pdlp_solver_settings_t<int, double>{};
  settings.method = cuopt::linear_programming::method_t::DualSimplex;

  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  optimization_problem_solution_t<int, double> solution = solve_lp(&handle_, op_problem, settings);
  EXPECT_EQ(solution.get_termination_status(), pdlp_termination_status_t::Optimal);
  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution.get_additional_termination_information().primal_objective));
}

// Should return a numerical error
TEST(pdlp_class, run_empty_matrix_pdlp)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/empty_matrix.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path);

  auto solver_settings   = pdlp_solver_settings_t<int, double>{};
  solver_settings.method = cuopt::linear_programming::method_t::PDLP;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_NUMERICAL_ERROR);
}

// Should run thanks to Dual Simplex
TEST(pdlp_class, run_empty_matrix_dual_simplex)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/empty_matrix.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path);

  auto solver_settings   = pdlp_solver_settings_t<int, double>{};
  solver_settings.method = cuopt::linear_programming::method_t::Concurrent;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(solution.get_additional_termination_information().solved_by_pdlp);
}

TEST(pdlp_class, test_max)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/good-max.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path);

  auto solver_settings   = pdlp_solver_settings_t<int, double>{};
  solver_settings.method = cuopt::linear_programming::method_t::PDLP;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_NEAR(
    solution.get_additional_termination_information().primal_objective, 17.0, factor_tolerance);
}

TEST(pdlp_class, test_max_with_offset)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/max_offset.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path);

  auto solver_settings   = pdlp_solver_settings_t<int, double>{};
  solver_settings.method = cuopt::linear_programming::method_t::PDLP;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_NEAR(
    solution.get_additional_termination_information().primal_objective, 0.0, factor_tolerance);
}

TEST(pdlp_class, test_lp_no_constraints)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/lp-model-no-constraints.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path);

  auto solver_settings = pdlp_solver_settings_t<int, double>{};

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_NEAR(
    solution.get_additional_termination_information().primal_objective, 1.0, factor_tolerance);
}

}  // namespace cuopt::linear_programming::test

CUOPT_TEST_PROGRAM_MAIN()
