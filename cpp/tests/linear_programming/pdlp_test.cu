/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

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

  constexpr double time_limit_seconds = 2;
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
      cuopt::linear_programming::pdlp_solver_mode_t::Stable3,
      cuopt::linear_programming::pdlp_solver_mode_t::Stable2,
      cuopt::linear_programming::pdlp_solver_mode_t::Stable1,
      cuopt::linear_programming::pdlp_solver_mode_t::Methodical1,
      cuopt::linear_programming::pdlp_solver_mode_t::Fast1,
    };
    for (auto solver_mode : solver_mode_list) {
      auto settings             = pdlp_solver_settings_t<int, double>{};
      settings.pdlp_solver_mode = solver_mode;
      settings.dual_postsolve   = false;
      for (auto [presolver, epsilon] :
           {std::pair{presolver_t::Papilo, 1e-1}, std::pair{presolver_t::None, 1e-6}}) {
        settings.presolver = presolver;
        settings.method    = cuopt::linear_programming::method_t::PDLP;
        const raft::handle_t handle_{};
        optimization_problem_solution_t<int, double> solution =
          solve_lp(&handle_, op_problem, settings);
        printf("running %s mode %d presolver %d\n",
               name.c_str(),
               (int)solver_mode,
               (int)settings.presolver);
        EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
        EXPECT_FALSE(is_incorrect_objective(
          expected_objective_value,
          solution.get_additional_termination_information().primal_objective));
        test_objective_sanity(op_problem,
                              solution.get_primal_solution(),
                              solution.get_additional_termination_information().primal_objective,
                              epsilon);
        test_constraint_sanity(op_problem,
                               solution.get_additional_termination_information(0),
                               solution.get_primal_solution(),
                               epsilon,
                               presolver);
      }
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
  set_pdlp_solver_mode(solver_settings);
  EXPECT_EQ(solver_settings.hyper_params.initial_step_size_scaling, 1);
  EXPECT_EQ(solver_settings.hyper_params.default_l_inf_ruiz_iterations, 5);
  EXPECT_TRUE(solver_settings.hyper_params.do_pock_chambolle_scaling);
  EXPECT_TRUE(solver_settings.hyper_params.do_ruiz_scaling);
  EXPECT_EQ(solver_settings.hyper_params.default_alpha_pock_chambolle_rescaling, 1.0);

  EXPECT_FALSE(solver_settings.hyper_params.update_step_size_on_initial_solution);
  EXPECT_FALSE(solver_settings.hyper_params.update_primal_weight_on_initial_solution);

  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    solver.run_solver(pdlp_timer);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(0), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(0), factor_tolerance);
  }

  // First add an initial primal then dual, then both, which shouldn't influence the values as the
  // scale on initial option is not toggled
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    solver.run_solver(pdlp_timer);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(0), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(0), factor_tolerance);
  }
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(pdlp_timer);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(0), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(0), factor_tolerance);
  }
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(pdlp_timer);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(0), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(0), factor_tolerance);
  }

  // Toggle the scale on initial solution while not providing should yield the same
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    solver_settings.hyper_params.update_step_size_on_initial_solution = true;
    solver.run_solver(pdlp_timer);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(0), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(0), factor_tolerance);
    solver_settings.hyper_params.update_step_size_on_initial_solution = false;
  }
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = true;
    solver.run_solver(pdlp_timer);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(0), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(0), factor_tolerance);
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = false;
  }
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = true;
    solver_settings.hyper_params.update_step_size_on_initial_solution     = true;
    solver.run_solver(pdlp_timer);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(0), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(0), factor_tolerance);
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = false;
    solver_settings.hyper_params.update_step_size_on_initial_solution     = false;
  }

  // Asking for initial scaling on step size with initial solution being only primal or only dual
  // should not break but not modify the step size
  {
    solver_settings.hyper_params.update_step_size_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    solver.run_solver(pdlp_timer);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(0), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(0), factor_tolerance);
    solver_settings.hyper_params.update_step_size_on_initial_solution = false;
  }
  {
    solver_settings.hyper_params.update_step_size_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(pdlp_timer);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(0), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(0), factor_tolerance);
    solver_settings.hyper_params.update_step_size_on_initial_solution = false;
  }

  // Asking for initial scaling on primal weight with initial solution being only primal or only
  // dual should *not* break but the primal weight should not change
  {
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    solver.run_solver(pdlp_timer);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(0), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(0), factor_tolerance);
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = false;
  }
  {
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(pdlp_timer);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(0), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(0), factor_tolerance);
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = false;
  }

  // All 0 solution when given an initial primal and dual with scale on the step size should not
  // break but not change primal weight and step size
  {
    solver_settings.hyper_params.update_step_size_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    std::vector<double> initial_primal(op_problem.get_n_variables(), 0);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 0);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(pdlp_timer);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(0), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(0), factor_tolerance);
    solver_settings.hyper_params.update_step_size_on_initial_solution = false;
  }

  // All 0 solution when given an initial primal and/or dual with scale on the primal weight is
  // *not* an error but should not change primal weight and step size
  {
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    std::vector<double> initial_primal(op_problem.get_n_variables(), 0);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    solver.run_solver(pdlp_timer);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(0), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(0), factor_tolerance);
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = false;
  }
  {
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 0);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(pdlp_timer);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(0), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(0), factor_tolerance);
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = false;
  }
  {
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    std::vector<double> initial_primal(op_problem.get_n_variables(), 0);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 0);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(pdlp_timer);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(0), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(0), factor_tolerance);
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = false;
  }

  // A non-all-0 vector for both initial primal and dual set should trigger a modification in primal
  // weight and step size
  {
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(pdlp_timer);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(0), factor_tolerance);
    EXPECT_NOT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(0), factor_tolerance);
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = false;
  }
  {
    solver_settings.hyper_params.update_step_size_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(pdlp_timer);
    EXPECT_NOT_NEAR(initial_step_size_afiro, solver.get_step_size_h(0), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(0), factor_tolerance);
    solver_settings.hyper_params.update_step_size_on_initial_solution = false;
  }
  {
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = true;
    solver_settings.hyper_params.update_step_size_on_initial_solution     = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(pdlp_timer);
    EXPECT_NOT_NEAR(initial_step_size_afiro, solver.get_step_size_h(0), factor_tolerance);
    EXPECT_NOT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(0), factor_tolerance);
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = false;
    solver_settings.hyper_params.update_step_size_on_initial_solution     = false;
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
  set_pdlp_solver_mode(solver_settings);
  EXPECT_FALSE(solver_settings.hyper_params.update_step_size_on_initial_solution);
  EXPECT_FALSE(solver_settings.hyper_params.update_primal_weight_on_initial_solution);

  // Check setting an initial primal weight and step size
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer                             = timer_t(solver_settings.time_limit);
    constexpr double test_initial_step_size     = 1.0;
    constexpr double test_initial_primal_weight = 2.0;
    solver.set_initial_primal_weight(test_initial_primal_weight);
    solver.set_initial_step_size(test_initial_step_size);
    solver.run_solver(pdlp_timer);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_EQ(test_initial_step_size, solver.get_step_size_h(0));
    EXPECT_EQ(test_initial_primal_weight, solver.get_primal_weight_h(0));
  }

  // Check that after setting an initial step size and primal weight, the computed one when adding
  // an initial primal / dual is indeed different
  {
    // Launching without an inital step size / primal weight and query the value
    solver_settings.hyper_params.update_primal_weight_on_initial_solution = true;
    solver_settings.hyper_params.update_step_size_on_initial_solution     = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto pdlp_timer = timer_t(solver_settings.time_limit);
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(pdlp_timer);
    const double previous_step_size     = solver.get_step_size_h(0);
    const double previous_primal_weight = solver.get_primal_weight_h(0);

    // Start again but with an initial and check the impact
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver2(problem, solver_settings);
    pdlp_timer                                  = timer_t(solver_settings.time_limit);
    constexpr double test_initial_step_size     = 1.0;
    constexpr double test_initial_primal_weight = 2.0;
    solver2.set_initial_primal_weight(test_initial_primal_weight);
    solver2.set_initial_step_size(test_initial_step_size);
    solver2.set_initial_primal_solution(d_initial_primal);
    solver2.set_initial_dual_solution(d_initial_dual);
    solver2.run_solver(pdlp_timer);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    const double sovler2_step_size     = solver2.get_step_size_h(0);
    const double sovler2_primal_weight = solver2.get_primal_weight_h(0);
    EXPECT_NOT_NEAR(previous_step_size, sovler2_step_size, factor_tolerance);
    EXPECT_NOT_NEAR(previous_primal_weight, sovler2_primal_weight, factor_tolerance);

    // Again but with an initial k which should change the step size only, not the primal weight
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver3(problem, solver_settings);
    pdlp_timer = timer_t(solver_settings.time_limit);
    solver3.set_initial_primal_weight(test_initial_primal_weight);
    solver3.set_initial_step_size(test_initial_step_size);
    solver3.set_initial_primal_solution(d_initial_primal);
    solver3.set_initial_k(10000);
    solver3.set_initial_dual_solution(d_initial_dual);
    solver3.set_initial_dual_solution(d_initial_dual);
    solver3.run_solver(pdlp_timer);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NOT_NEAR(sovler2_step_size, solver3.get_step_size_h(0), factor_tolerance);
    EXPECT_NEAR(sovler2_primal_weight, solver3.get_primal_weight_h(0), factor_tolerance);
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

  auto solver_settings = pdlp_solver_settings_t<int, double>{};
  cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
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
  solver_settings.pdlp_solver_mode =
    cuopt::linear_programming::pdlp_solver_mode_t::Stable2;  // Not supported for the default
                                                             // Stable3 for now
  set_pdlp_solver_mode(solver_settings);

  // First solve without the per constraint and it should break
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);

    raft::copy(solver.pdhg_solver_.get_primal_solution().data(),
               d_initial_primal.data(),
               d_initial_primal.size(),
               handle.get_stream());

    auto& current_termination_strategy = solver.get_current_termination_strategy();
    current_termination_strategy.evaluate_termination_criteria(solver.pdhg_solver_,
                                                               d_initial_primal,
                                                               d_initial_primal,
                                                               solver.pdhg_solver_.get_dual_slack(),
                                                               d_initial_primal,
                                                               d_initial_primal,
                                                               0,
                                                               problem.combined_bounds,
                                                               problem.objective_coefficients);
    pdlp_termination_status_t termination_current =
      current_termination_strategy.get_termination_status(0);

    EXPECT_TRUE(termination_current != pdlp_termination_status_t::Optimal);
  }
  {
    solver_settings.per_constraint_residual = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);

    raft::copy(solver.pdhg_solver_.get_primal_solution().data(),
               d_initial_primal.data(),
               d_initial_primal.size(),
               handle.get_stream());

    auto& current_termination_strategy = solver.get_current_termination_strategy();
    current_termination_strategy.evaluate_termination_criteria(solver.pdhg_solver_,
                                                               d_initial_primal,
                                                               d_initial_primal,
                                                               solver.pdhg_solver_.get_dual_slack(),
                                                               d_initial_primal,
                                                               d_initial_primal,
                                                               0,
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
  solver_settings.pdlp_solver_mode =
    cuopt::linear_programming::pdlp_solver_mode_t::Stable2;  // Not supported for the default
                                                             // Stable3 for now
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
    solver_settings.presolver            = presolver_t::None;

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

TEST(pdlp_class, warm_start_stable3_not_supported)
{
  const raft::handle_t handle{};

  auto path                        = make_path_absolute("linear_programming/afiro_original.mps");
  auto solver_settings             = pdlp_solver_settings_t<int, double>{};
  solver_settings.pdlp_solver_mode = cuopt::linear_programming::pdlp_solver_mode_t::Stable3;
  solver_settings.set_optimality_tolerance(1e-2);
  solver_settings.detect_infeasibility = false;
  solver_settings.method               = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver            = presolver_t::None;

  cuopt::mps_parser::mps_data_model_t<int, double> mps_data_model =
    cuopt::mps_parser::parse_mps<int, double>(path);
  auto op_problem = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
    &handle, mps_data_model);
  optimization_problem_solution_t<int, double> solution = solve_lp(op_problem, solver_settings);
  EXPECT_EQ(solution.get_termination_status(), pdlp_termination_status_t::Optimal);
  solver_settings.set_pdlp_warm_start_data(solution.get_pdlp_warm_start_data());
  optimization_problem_solution_t<int, double> solution2 = solve_lp(op_problem, solver_settings);
  EXPECT_EQ(solution2.get_termination_status(), pdlp_termination_status_t::NoTermination);
}

TEST(pdlp_class, dual_postsolve_size)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::Papilo;

  {
    solver_settings.dual_postsolve = true;
    optimization_problem_solution_t<int, double> solution =
      solve_lp(&handle_, op_problem, solver_settings);
    EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
    EXPECT_EQ(solution.get_dual_solution().size(), op_problem.get_n_constraints());
  }

  {
    solver_settings.dual_postsolve = false;
    optimization_problem_solution_t<int, double> solution =
      solve_lp(&handle_, op_problem, solver_settings);
    EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
    EXPECT_EQ(solution.get_dual_solution().size(), 0);
  }
}

TEST(dual_simplex, afiro)
{
  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings =
    cuopt::linear_programming::pdlp_solver_settings_t<int, double>{};
  settings.method    = cuopt::linear_programming::method_t::DualSimplex;
  settings.presolver = presolver_t::None;

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

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

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

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::Concurrent;
  solver_settings.presolver = presolver_t::None;

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

  auto solver_settings             = pdlp_solver_settings_t<int, double>{};
  solver_settings.method           = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode = cuopt::linear_programming::pdlp_solver_mode_t::Stable2;
  solver_settings.presolver        = presolver_t::None;

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

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

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

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.presolver = presolver_t::None;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_NEAR(
    solution.get_additional_termination_information().primal_objective, 1.0, factor_tolerance);
}

template <typename T>
rmm::device_uvector<T> extract_subvector(const rmm::device_uvector<T>& vector,
                                         size_t start,
                                         size_t length)
{
  rmm::device_uvector<T> subvector(length, vector.stream());
  raft::copy(subvector.data(), vector.data() + start, length, vector.stream());
  return subvector;
}

TEST(pdlp_class, simple_batch_afiro)
{
  const raft::handle_t handle_{};
  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

  constexpr int batch_size = 5;

  // Setup a larger batch afiro but with all same primal/dual bounds

  const auto& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const auto& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  for (size_t i = 0; i < batch_size; i++) {
    solver_settings.new_bounds.push_back({0, variable_lower_bounds[0], variable_upper_bounds[0]});
  }

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);

  // All should be optimal with the right objective
  for (size_t i = 0; i < batch_size; ++i) {
    EXPECT_EQ((int)solution.get_termination_status(i), CUOPT_TERIMINATION_STATUS_OPTIMAL);
    EXPECT_FALSE(is_incorrect_objective(
      afiro_primal_objective, solution.get_additional_termination_information(i).primal_objective));
  }

  // All should have the bitwise same primal/dual objective, termination reason, iterations,
  // residuals and primal/dual values compared to ref
  const auto ref_stats  = (int)solution.get_termination_status(0);
  const auto ref_primal = solution.get_additional_termination_information(0).primal_objective;
  const auto ref_dual   = solution.get_additional_termination_information(0).dual_objective;
  const auto ref_it     = solution.get_additional_termination_information(0).number_of_steps_taken;
  const auto ref_it_total =
    solution.get_additional_termination_information(0).total_number_of_attempted_steps;
  const auto ref_primal_residual =
    solution.get_additional_termination_information(0).l2_primal_residual;
  const auto ref_dual_residual =
    solution.get_additional_termination_information(0).l2_dual_residual;

  const auto ref_primal_solution =
    host_copy(solution.get_primal_solution(), solution.get_primal_solution().stream());
  const auto ref_dual_solution =
    host_copy(solution.get_dual_solution(), solution.get_dual_solution().stream());

  const size_t primal_size = ref_primal_solution.size() / batch_size;
  const size_t dual_size   = ref_dual_solution.size() / batch_size;

  for (size_t i = 1; i < batch_size; ++i) {
    EXPECT_EQ(ref_stats, (int)solution.get_termination_status(i));
    EXPECT_EQ(ref_primal, solution.get_additional_termination_information(i).primal_objective);
    EXPECT_EQ(ref_dual, solution.get_additional_termination_information(i).dual_objective);
    EXPECT_EQ(ref_it, solution.get_additional_termination_information(i).number_of_steps_taken);
    EXPECT_EQ(ref_it_total,
              solution.get_additional_termination_information(i).total_number_of_attempted_steps);
    EXPECT_EQ(ref_primal_residual,
              solution.get_additional_termination_information(i).l2_primal_residual);
    EXPECT_EQ(ref_dual_residual,
              solution.get_additional_termination_information(i).l2_dual_residual);
    // Direclty compare on ref since we just compare the first climber to the rest
    for (size_t p = 0; p < primal_size; ++p)
      EXPECT_EQ(ref_primal_solution[p], ref_primal_solution[p + i * primal_size]);
    for (size_t d = 0; d < dual_size; ++d)
      EXPECT_EQ(ref_dual_solution[d], ref_dual_solution[d + i * dual_size]);
  }

  const auto primal_solution = extract_subvector(solution.get_primal_solution(), 0, primal_size);

  test_objective_sanity(op_problem,
                        primal_solution,
                        solution.get_additional_termination_information(0).primal_objective);
  test_constraint_sanity(op_problem,
                         solution.get_additional_termination_information(0),
                         primal_solution,
                         tolerance,
                         false);
}

TEST(pdlp_class, simple_batch_different_bounds)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

  const std::vector<double>& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const std::vector<double>& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  // Solve alone to get ref
  auto op_problem_ref                           = op_problem;
  op_problem_ref.get_variable_lower_bounds()[5] = 4.0;
  op_problem_ref.get_variable_upper_bounds()[5] = 5.0;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem_ref, solver_settings);

  // Create new variable bounds for the first climber in the batch
  solver_settings.new_bounds.push_back({5, 4.0, 5.0});
  // The second climber has no changes
  solver_settings.new_bounds.push_back({0, variable_lower_bounds[0], variable_upper_bounds[0]});

  const auto new_primal = solution.get_additional_termination_information(0).primal_objective;

  // Now setup and solve batch
  optimization_problem_solution_t<int, double> solution2 =
    solve_lp(&handle_, op_problem, solver_settings);

  // Both should be optimal
  // Climber #0 should have same objective as ref and #1 as the usual
  EXPECT_EQ((int)solution2.get_termination_status(0), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(is_incorrect_objective(
    new_primal, solution2.get_additional_termination_information(0).primal_objective));
  EXPECT_EQ((int)solution2.get_termination_status(1), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution2.get_additional_termination_information(1).primal_objective));

  const auto primal_solution = extract_subvector(
    solution2.get_primal_solution(), 0, solution2.get_primal_solution().size() / 2);

  test_objective_sanity(op_problem_ref,
                        primal_solution,
                        solution2.get_additional_termination_information(0).primal_objective);
  test_constraint_sanity(op_problem_ref,
                         solution2.get_additional_termination_information(0),
                         primal_solution,
                         tolerance,
                         false);
}

TEST(pdlp_class, more_complex_batch_different_bounds)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

  constexpr int batch_size = 5;

  // Setup a larger batch afiro but with different bounds on climbers #1 and #3
  const std::vector<double>& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const std::vector<double>& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  // Get ref for climber #1
  auto op_problem_ref1                           = op_problem;
  op_problem_ref1.get_variable_lower_bounds()[5] = 4.0;
  op_problem_ref1.get_variable_upper_bounds()[5] = 5.0;
  optimization_problem_solution_t<int, double> solution1 =
    solve_lp(&handle_, op_problem_ref1, solver_settings);
  const auto first_new_primal =
    solution1.get_additional_termination_information(0).primal_objective;

  // Get ref for climber #3
  auto op_problem_ref3                           = op_problem;
  op_problem_ref3.get_variable_lower_bounds()[1] = -7.0;
  op_problem_ref3.get_variable_upper_bounds()[1] = 13.0;
  optimization_problem_solution_t<int, double> solution2 =
    solve_lp(&handle_, op_problem_ref3, solver_settings);
  const auto second_new_primal =
    solution2.get_additional_termination_information(0).primal_objective;

  // Climber #0: no-op
  solver_settings.new_bounds.push_back({0, variable_lower_bounds[0], variable_upper_bounds[0]});
  // Climber #1: var 5 -> [4.0, 5.0]
  solver_settings.new_bounds.push_back({5, 4.0, 5.0});
  // Climber #2: no-op
  solver_settings.new_bounds.push_back({0, variable_lower_bounds[0], variable_upper_bounds[0]});
  // Climber #3: var 1 -> [-7.0, 13.0]
  solver_settings.new_bounds.push_back({1, -7.0, 13.0});
  // Climber #4: no-op
  solver_settings.new_bounds.push_back({0, variable_lower_bounds[0], variable_upper_bounds[0]});

  // Setup and solve batch
  optimization_problem_solution_t<int, double> solution3 =
    solve_lp(&handle_, op_problem, solver_settings);

  // All should be optimal
  for (size_t i = 0; i < batch_size; ++i)
    EXPECT_EQ((int)solution3.get_termination_status(i), CUOPT_TERIMINATION_STATUS_OPTIMAL);

  // Climber #0 #2 #4 should have the same primal objective which is the unmodified one
  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution3.get_additional_termination_information(0).primal_objective));
  EXPECT_TRUE(solution3.get_additional_termination_information(0).primal_objective ==
                solution3.get_additional_termination_information(2).primal_objective &&
              solution3.get_additional_termination_information(2).primal_objective ==
                solution3.get_additional_termination_information(4).primal_objective);

  // Climber #1 and #3 should have same objective as to when ran alone
  EXPECT_FALSE(is_incorrect_objective(
    first_new_primal, solution3.get_additional_termination_information(1).primal_objective));

  EXPECT_FALSE(is_incorrect_objective(
    second_new_primal, solution3.get_additional_termination_information(3).primal_objective));

  const size_t primal_size = solution3.get_primal_solution().size() / batch_size;

  // Sanity checks for all climbers
  for (size_t i = 0; i < batch_size; ++i) {
    const auto current_primal_solution =
      extract_subvector(solution3.get_primal_solution(), i * primal_size, primal_size);
    const auto& current_info = solution3.get_additional_termination_information(i);

    if (i == 1) {
      test_objective_sanity(
        op_problem_ref1, current_primal_solution, current_info.primal_objective);
      test_constraint_sanity(
        op_problem_ref1, current_info, current_primal_solution, tolerance, false);
    } else if (i == 3) {
      test_objective_sanity(
        op_problem_ref3, current_primal_solution, current_info.primal_objective);
      test_constraint_sanity(
        op_problem_ref3, current_info, current_primal_solution, tolerance, false);
    } else {
      test_objective_sanity(op_problem, current_primal_solution, current_info.primal_objective);
      test_constraint_sanity(op_problem, current_info, current_primal_solution, tolerance, false);
    }
  }
}

TEST(pdlp_class, DISABLED_cupdlpx_infeasible_detection_afiro_new_bounds)
{
  const raft::handle_t handle_{};

  auto solver_settings                 = pdlp_solver_settings_t<int, double>{};
  solver_settings.method               = cuopt::linear_programming::method_t::PDLP;
  solver_settings.detect_infeasibility = true;

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  for (size_t i = 1; i < 8; ++i) {
    op_problem.get_variable_lower_bounds()[i] = 7.0;
    op_problem.get_variable_upper_bounds()[i] = 8.0;
  }
  for (size_t i = 13; i < 27; ++i) {
    op_problem.get_variable_lower_bounds()[i] = 1.0;
    op_problem.get_variable_upper_bounds()[i] = 5.0;
  }

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);

  EXPECT_EQ(solution.get_termination_status(0), pdlp_termination_status_t::PrimalInfeasible);
}

TEST(pdlp_class, DISABLED_cupdlpx_batch_infeasible_detection)
{
  const raft::handle_t handle_{};

  auto solver_settings                 = pdlp_solver_settings_t<int, double>{};
  solver_settings.method               = cuopt::linear_programming::method_t::PDLP;
  solver_settings.detect_infeasibility = true;

  constexpr int batch_size = 5;

  auto path = make_path_absolute("linear_programming/good-mps-fixed-ranges.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  const std::vector<double>& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const std::vector<double>& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  for (size_t i = 0; i < batch_size; i++) {
    solver_settings.new_bounds.push_back({0, variable_lower_bounds[0], variable_upper_bounds[0]});
  }

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);

  EXPECT_EQ(solution.get_termination_status(0), pdlp_termination_status_t::PrimalInfeasible);

  // All should have the bitwise same termination reason, and iterations
  const auto ref_stats = (int)solution.get_termination_status(0);
  const auto ref_it    = solution.get_additional_termination_information(0).number_of_steps_taken;
  const auto ref_it_total =
    solution.get_additional_termination_information(0).total_number_of_attempted_steps;

  for (size_t i = 1; i < batch_size; ++i) {
    EXPECT_EQ(ref_stats, (int)solution.get_termination_status(i));
    EXPECT_EQ(ref_it, solution.get_additional_termination_information(i).number_of_steps_taken);
    EXPECT_EQ(ref_it_total,
              solution.get_additional_termination_information(i).total_number_of_attempted_steps);
  }
}

// Disabled until we have a reliable way to detect infeasibility
TEST(pdlp_class, DISABLED_cupdlpx_infeasible_detection_batch_afiro_new_bounds)
{
  const raft::handle_t handle_{};

  auto solver_settings                 = pdlp_solver_settings_t<int, double>{};
  solver_settings.method               = cuopt::linear_programming::method_t::PDLP;
  solver_settings.detect_infeasibility = true;

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  // Use a ref problem that is infeasible
  auto op_problem_ref                           = op_problem;
  op_problem_ref.get_variable_lower_bounds()[1] = 7.0;
  op_problem_ref.get_variable_upper_bounds()[1] = 8.0;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem_ref, solver_settings);

  EXPECT_EQ(solution.get_termination_status(0), pdlp_termination_status_t::PrimalInfeasible);

  constexpr int batch_size = 5;

  const std::vector<double>& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const std::vector<double>& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  for (size_t i = 0; i < batch_size; i++) {
    solver_settings.new_bounds.push_back({1, 7.0, 8.0});
  }

  optimization_problem_solution_t<int, double> solution2 =
    solve_lp(&handle_, op_problem, solver_settings);

  // All should have the bitwise same termination reason, and iterations
  const auto ref_stats = (int)solution.get_termination_status(0);
  const auto ref_it    = solution.get_additional_termination_information(0).number_of_steps_taken;
  const auto ref_it_total =
    solution.get_additional_termination_information(0).total_number_of_attempted_steps;

  for (size_t i = 0; i < batch_size; ++i) {
    EXPECT_EQ(ref_stats, (int)solution2.get_termination_status(i));
    EXPECT_EQ(ref_it, solution2.get_additional_termination_information(i).number_of_steps_taken);
    EXPECT_EQ(ref_it_total,
              solution2.get_additional_termination_information(i).total_number_of_attempted_steps);
  }
}

TEST(pdlp_class, new_bounds)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

  // Manually changing the bounds and doing it through the solver settings should give the same
  // result

  solver_settings.new_bounds.push_back({0, 45.0, 55.0});

  optimization_problem_solution_t<int, double> solution1 =
    solve_lp(&handle_, op_problem, solver_settings);

  solver_settings.new_bounds.clear();

  std::vector<double>& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  std::vector<double>& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  variable_lower_bounds[0] = 45.0;
  variable_upper_bounds[0] = 55.0;

  optimization_problem_solution_t<int, double> solution2 =
    solve_lp(&handle_, op_problem, solver_settings);

  EXPECT_EQ(solution1.get_additional_termination_information(0).primal_objective,
            solution2.get_additional_termination_information(0).primal_objective);
  EXPECT_EQ(solution1.get_additional_termination_information(0).dual_objective,
            solution2.get_additional_termination_information(0).dual_objective);
  EXPECT_EQ(solution1.get_additional_termination_information(0).number_of_steps_taken,
            solution2.get_additional_termination_information(0).number_of_steps_taken);
  EXPECT_EQ(solution1.get_additional_termination_information(0).total_number_of_attempted_steps,
            solution2.get_additional_termination_information(0).total_number_of_attempted_steps);
  EXPECT_EQ(solution1.get_additional_termination_information(0).l2_primal_residual,
            solution2.get_additional_termination_information(0).l2_primal_residual);
  EXPECT_EQ(solution1.get_additional_termination_information(0).l2_dual_residual,
            solution2.get_additional_termination_information(0).l2_dual_residual);
}

TEST(pdlp_class, big_batch_afiro)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

  constexpr int batch_size = 1000;

  // Setup a larger batch afiro but with all same primal/dual bounds

  const std::vector<double>& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const std::vector<double>& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  for (size_t i = 0; i < batch_size; i++) {
    solver_settings.new_bounds.push_back({0, variable_lower_bounds[0], variable_upper_bounds[0]});
  }

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);

  // All should be optimal with
  for (size_t i = 0; i < batch_size; ++i) {
    EXPECT_EQ((int)solution.get_termination_status(i), CUOPT_TERIMINATION_STATUS_OPTIMAL);
    EXPECT_FALSE(is_incorrect_objective(
      afiro_primal_objective, solution.get_additional_termination_information(i).primal_objective));
  }

  // All should have the bitwise same primal/dual objective, termination reason, iterations,
  // residuals and primal/dual values compared to ref
  const auto ref_stats  = (int)solution.get_termination_status(0);
  const auto ref_primal = solution.get_additional_termination_information(0).primal_objective;
  const auto ref_dual   = solution.get_additional_termination_information(0).dual_objective;
  const auto ref_it     = solution.get_additional_termination_information(0).number_of_steps_taken;
  const auto ref_it_total =
    solution.get_additional_termination_information(0).total_number_of_attempted_steps;
  const auto ref_primal_residual =
    solution.get_additional_termination_information(0).l2_primal_residual;
  const auto ref_dual_residual =
    solution.get_additional_termination_information(0).l2_dual_residual;

  const auto ref_primal_solution =
    host_copy(solution.get_primal_solution(), solution.get_primal_solution().stream());
  const auto ref_dual_solution =
    host_copy(solution.get_dual_solution(), solution.get_dual_solution().stream());

  const size_t primal_size = ref_primal_solution.size() / batch_size;
  const size_t dual_size   = ref_dual_solution.size() / batch_size;

  for (size_t i = 1; i < batch_size; ++i) {
    EXPECT_EQ(ref_stats, (int)solution.get_termination_status(i));
    EXPECT_EQ(ref_primal, solution.get_additional_termination_information(i).primal_objective);
    EXPECT_EQ(ref_dual, solution.get_additional_termination_information(i).dual_objective);
    EXPECT_EQ(ref_it, solution.get_additional_termination_information(i).number_of_steps_taken);
    EXPECT_EQ(ref_it_total,
              solution.get_additional_termination_information(i).total_number_of_attempted_steps);
    EXPECT_EQ(ref_primal_residual,
              solution.get_additional_termination_information(i).l2_primal_residual);
    EXPECT_EQ(ref_dual_residual,
              solution.get_additional_termination_information(i).l2_dual_residual);
    // Direclty compare on ref since we just compare the first climber to the rest
    for (size_t p = 0; p < primal_size; ++p)
      EXPECT_EQ(ref_primal_solution[p], ref_primal_solution[p + i * primal_size]);
    for (size_t d = 0; d < dual_size; ++d)
      EXPECT_EQ(ref_dual_solution[d], ref_dual_solution[d + i * dual_size]);
  }

  const auto primal_solution =
    extract_subvector(solution.get_primal_solution(), primal_size * (batch_size - 1), primal_size);

  test_objective_sanity(
    op_problem,
    primal_solution,
    solution.get_additional_termination_information(batch_size - 1).primal_objective);
  test_constraint_sanity(op_problem,
                         solution.get_additional_termination_information(batch_size - 1),
                         primal_solution,
                         tolerance,
                         false);
}

// Disabled until we have a reliable way to detect infeasibility
TEST(pdlp_class, DISABLED_simple_batch_optimal_and_infeasible)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  auto solver_settings                 = pdlp_solver_settings_t<int, double>{};
  solver_settings.method               = cuopt::linear_programming::method_t::PDLP;
  solver_settings.detect_infeasibility = true;
  solver_settings.presolver            = presolver_t::None;

  const std::vector<double>& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const std::vector<double>& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  // Make the first problem infeasible while the second remains solvable
  solver_settings.new_bounds.push_back({1, 7.0, 8.0});
  // No change for the second
  solver_settings.new_bounds.push_back({0, variable_lower_bounds[0], variable_upper_bounds[0]});

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);

  // First should be primal infeasible and the second optimal with the correct
  EXPECT_EQ((int)solution.get_termination_status(0), CUOPT_TERIMINATION_STATUS_INFEASIBLE);
  EXPECT_EQ((int)solution.get_termination_status(1), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution.get_additional_termination_information(1).primal_objective));
}

// Disabled until we have a reliable way to detect infeasibility
TEST(pdlp_class, DISABLED_larger_batch_optimal_and_infeasible)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  auto solver_settings                 = pdlp_solver_settings_t<int, double>{};
  solver_settings.method               = cuopt::linear_programming::method_t::PDLP;
  solver_settings.detect_infeasibility = true;

  const std::vector<double>& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const std::vector<double>& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  // #0: no-op
  solver_settings.new_bounds.push_back({0, variable_lower_bounds[0], variable_upper_bounds[0]});
  // #1: var 1 -> [7.0, 8.0] (infeasible)
  solver_settings.new_bounds.push_back({1, 7.0, 8.0});
  // #2: no-op
  solver_settings.new_bounds.push_back({0, variable_lower_bounds[0], variable_upper_bounds[0]});
  // #3: var 1 -> [-11.0, -10.0] (infeasible)
  solver_settings.new_bounds.push_back({1, -11.0, -10.0});
  // #4: no-op
  solver_settings.new_bounds.push_back({0, variable_lower_bounds[0], variable_upper_bounds[0]});

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);

  // #1 and #3 should be infeasible
  EXPECT_EQ((int)solution.get_termination_status(1), CUOPT_TERIMINATION_STATUS_INFEASIBLE);
  EXPECT_EQ((int)solution.get_termination_status(3), CUOPT_TERIMINATION_STATUS_INFEASIBLE);

  // Rest should be feasible with the correct primal objective
  EXPECT_EQ((int)solution.get_termination_status(0), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_EQ((int)solution.get_termination_status(2), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_EQ((int)solution.get_termination_status(4), CUOPT_TERIMINATION_STATUS_OPTIMAL);

  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution.get_additional_termination_information(0).primal_objective));
  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution.get_additional_termination_information(2).primal_objective));
  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution.get_additional_termination_information(4).primal_objective));
}

TEST(pdlp_class, strong_branching_test)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  const std::vector<int> fractional     = {1, 2, 4};
  const std::vector<double> root_soln_x = {0.891, 0.109, 0.636429};

  auto solver_settings             = pdlp_solver_settings_t<int, double>{};
  solver_settings.method           = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode = pdlp_solver_mode_t::Stable3;
  solver_settings.presolver        = cuopt::linear_programming::presolver_t::None;

  const int n_fractional = fractional.size();
  const int batch_size   = n_fractional * 2;

  std::vector<double> ref_objectives(batch_size);
  std::vector<pdlp_termination_status_t> ref_statuses(batch_size);
  std::vector<cuopt::mps_parser::mps_data_model_t<int, double>> ref_problems;

  // Logic from batch_pdlp_solve in solve.cu:
  // Down branches first, then Up branches

  // Down branches
  for (int i = 0; i < n_fractional; ++i) {
    auto ref_prob                                 = op_problem;
    int var_idx                                   = fractional[i];
    ref_prob.get_variable_upper_bounds()[var_idx] = std::floor(root_soln_x[i]);
    ref_problems.push_back(ref_prob);
  }
  // Up branches
  for (int i = 0; i < n_fractional; ++i) {
    auto ref_prob                                 = op_problem;
    int var_idx                                   = fractional[i];
    ref_prob.get_variable_lower_bounds()[var_idx] = std::ceil(root_soln_x[i]);
    ref_problems.push_back(ref_prob);
  }

  // Solve references
  for (int i = 0; i < batch_size; ++i) {
    auto sol          = solve_lp(&handle_, ref_problems[i], solver_settings);
    ref_statuses[i]   = sol.get_termination_status(0);
    ref_objectives[i] = sol.get_additional_termination_information(0).primal_objective;
  }

  // Solve batch
  auto batch_sol = batch_pdlp_solve(&handle_, op_problem, fractional, root_soln_x, solver_settings);

  EXPECT_EQ((int)batch_sol.get_terminations_status().size(), batch_size);
  const size_t primal_size = op_problem.get_n_variables();

  for (int i = 0; i < batch_size; ++i) {
    EXPECT_EQ(batch_sol.get_termination_status(i), ref_statuses[i]);
    // Climber in the batch that have gained optimality can lose optimality while other are still
    // optimizing This can lead to differences in the objective values, so we allow for a small
    // tolerance
    EXPECT_NEAR(batch_sol.get_additional_termination_information(i).primal_objective,
                ref_objectives[i],
                1e-1);

    // Sanity checks
    const auto current_primal_solution =
      extract_subvector(batch_sol.get_primal_solution(), i * primal_size, primal_size);
    const auto& current_info = batch_sol.get_additional_termination_information(i);

    test_objective_sanity(ref_problems[i], current_primal_solution, current_info.primal_objective);
    test_constraint_sanity(
      ref_problems[i], current_info, current_primal_solution, tolerance, false);
  }

  // Now run again using the new_bounds API
  for (int i = 0; i < n_fractional; ++i) {
    solver_settings.new_bounds.push_back({fractional[i],
                                          op_problem.get_variable_lower_bounds()[fractional[i]],
                                          std::floor(root_soln_x[i])});
  }
  for (int i = 0; i < n_fractional; ++i) {
    solver_settings.new_bounds.push_back({fractional[i],
                                          std::ceil(root_soln_x[i]),
                                          op_problem.get_variable_upper_bounds()[fractional[i]]});
  }
  auto batch_sol2 = solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ(batch_sol2.get_terminations_status().size(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_EQ(batch_sol2.get_termination_status(i), batch_sol.get_termination_status(i));
    EXPECT_NEAR(batch_sol2.get_additional_termination_information(i).primal_objective,
                ref_objectives[i],
                1e-1);

    const auto current_primal_solution =
      extract_subvector(batch_sol2.get_primal_solution(), i * primal_size, primal_size);
    test_objective_sanity(ref_problems[i],
                          current_primal_solution,
                          batch_sol2.get_additional_termination_information(i).primal_objective);
    test_constraint_sanity(ref_problems[i],
                           batch_sol2.get_additional_termination_information(i),
                           current_primal_solution,
                           tolerance,
                           false);
  }
}

TEST(pdlp_class, many_different_bounds)
{
  constexpr double lower_bounds = -33.0;
  constexpr double upper_bounds = 10;

  const raft::handle_t handle_{};
  auto path = make_path_absolute("linear_programming/good-mps-some-var-bounds.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  const auto& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const auto& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  std::vector<std::tuple<int, double, double>> custom_bounds = {
    {0, lower_bounds - 100, upper_bounds},
    {0, variable_lower_bounds[0], variable_upper_bounds[0]},
    {0, lower_bounds - 100, upper_bounds},
    {0, variable_lower_bounds[0], variable_upper_bounds[0]},
    {0, lower_bounds - 150, upper_bounds},
    {0, lower_bounds - 200, upper_bounds},
    {0, variable_lower_bounds[0], variable_upper_bounds[0]},
    {0, lower_bounds - 1000, upper_bounds},
    {0, lower_bounds - 1000, upper_bounds},
    {0, lower_bounds - 1250, upper_bounds},
    {0, lower_bounds - 2500, upper_bounds},
    {0, variable_lower_bounds[0], variable_upper_bounds[0]},
    {0, variable_lower_bounds[0], variable_upper_bounds[0]},
  };
  const int batch_size = custom_bounds.size();
  std::vector<double> ref_objectives(batch_size);
  std::vector<pdlp_termination_status_t> ref_statuses(batch_size);
  std::vector<cuopt::mps_parser::mps_data_model_t<int, double>> ref_problems;
  std::vector<std::vector<double>> ref_primal_solutions(batch_size);

  // Solve each variant using PDLP
  for (int i = 0; i < batch_size; ++i) {
    const auto& bounds        = custom_bounds[i];
    auto solver_settings      = pdlp_solver_settings_t<int, double>{};
    solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
    solver_settings.presolver = presolver_t::None;
    auto ref_prob             = op_problem;
    ref_prob.get_variable_lower_bounds()[std::get<0>(bounds)] = std::get<1>(bounds);
    ref_prob.get_variable_upper_bounds()[std::get<0>(bounds)] = std::get<2>(bounds);
    ref_problems.push_back(ref_prob);
    auto solution     = solve_lp(&handle_, ref_prob, solver_settings);
    ref_statuses[i]   = solution.get_termination_status(0);
    ref_objectives[i] = solution.get_additional_termination_information(0).primal_objective;
    ref_primal_solutions[i] =
      host_copy(solution.get_primal_solution(), solution.get_primal_solution().stream());
  }

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;
  for (int i = 0; i < batch_size; ++i) {
    solver_settings.new_bounds.push_back(custom_bounds[i]);
  }

  optimization_problem_solution_t<int, double> batch_sol =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ(batch_sol.get_terminations_status().size(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    const size_t primal_size = op_problem.get_n_variables();
    EXPECT_EQ(batch_sol.get_termination_status(i), ref_statuses[i]);
    EXPECT_EQ(batch_sol.get_additional_termination_information(i).primal_objective,
              ref_objectives[i]);
    const auto current_primal_solution =
      extract_subvector(batch_sol.get_primal_solution(), i * primal_size, primal_size);
    const auto host_primal_solution =
      host_copy(extract_subvector(batch_sol.get_primal_solution(), i * primal_size, primal_size),
                batch_sol.get_primal_solution().stream());
    for (size_t p = 0; p < primal_size; ++p)
      EXPECT_EQ(host_primal_solution[p], ref_primal_solutions[i][p]);
    test_objective_sanity(ref_problems[i],
                          current_primal_solution,
                          batch_sol.get_additional_termination_information(i).primal_objective);
    // Here we can enforce very low tolerance because the problem is simple so the solution is exact
    // even accounting for scaling
    test_constraint_sanity(ref_problems[i],
                           batch_sol.get_additional_termination_information(i),
                           current_primal_solution,
                           1e-8,
                           false);
  }
}

TEST(pdlp_class, some_climber_hit_iteration_limit)
{
  // Same as above but with only two climber, one of wich should converge before iteration limit and
  // the other should hit it We should be able to retrieve the solution of the climber that was
  // optimal before iteration limit and correctly find iteration limit for the other climber

  constexpr double lower_bounds = -33.0;
  constexpr double upper_bounds = 10;

  const raft::handle_t handle_{};
  auto path = make_path_absolute("linear_programming/good-mps-some-var-bounds.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  const auto& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const auto& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  std::vector<std::tuple<int, double, double>> custom_bounds = {
    {0, lower_bounds - 2500, upper_bounds},
    {0, variable_lower_bounds[0], variable_upper_bounds[0]},
  };
  const int batch_size = custom_bounds.size();
  std::vector<double> ref_objectives(batch_size);
  std::vector<pdlp_termination_status_t> ref_statuses(batch_size);
  std::vector<cuopt::mps_parser::mps_data_model_t<int, double>> ref_problems;
  std::vector<std::vector<double>> ref_primal_solutions(batch_size);

  // Solve each variant using PDLP
  for (int i = 0; i < batch_size; ++i) {
    const auto& bounds              = custom_bounds[i];
    auto solver_settings            = pdlp_solver_settings_t<int, double>{};
    solver_settings.method          = cuopt::linear_programming::method_t::PDLP;
    solver_settings.iteration_limit = 500;
    solver_settings.presolver       = presolver_t::None;
    auto ref_prob                   = op_problem;
    ref_prob.get_variable_lower_bounds()[std::get<0>(bounds)] = std::get<1>(bounds);
    ref_prob.get_variable_upper_bounds()[std::get<0>(bounds)] = std::get<2>(bounds);
    ref_problems.push_back(ref_prob);
    auto solution     = solve_lp(&handle_, ref_prob, solver_settings);
    ref_statuses[i]   = solution.get_termination_status(0);
    ref_objectives[i] = solution.get_additional_termination_information(0).primal_objective;
    ref_primal_solutions[i] =
      host_copy(solution.get_primal_solution(), solution.get_primal_solution().stream());
  }

  auto solver_settings            = pdlp_solver_settings_t<int, double>{};
  solver_settings.method          = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver       = presolver_t::None;
  solver_settings.iteration_limit = 500;
  for (int i = 0; i < batch_size; ++i) {
    solver_settings.new_bounds.push_back(custom_bounds[i]);
  }

  optimization_problem_solution_t<int, double> batch_sol =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ(batch_sol.get_terminations_status().size(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    const size_t primal_size = op_problem.get_n_variables();
    EXPECT_EQ(batch_sol.get_termination_status(i), ref_statuses[i]);

    // Check other information only for the one that has converged before iteration limit
    if (ref_statuses[i] == pdlp_termination_status_t::Optimal) {
      EXPECT_EQ(batch_sol.get_additional_termination_information(i).primal_objective,
                ref_objectives[i]);
      const auto current_primal_solution =
        extract_subvector(batch_sol.get_primal_solution(), i * primal_size, primal_size);
      const auto host_primal_solution =
        host_copy(extract_subvector(batch_sol.get_primal_solution(), i * primal_size, primal_size),
                  batch_sol.get_primal_solution().stream());
      for (size_t p = 0; p < primal_size; ++p)
        EXPECT_EQ(host_primal_solution[p], ref_primal_solutions[i][p]);
      test_objective_sanity(ref_problems[i],
                            current_primal_solution,
                            batch_sol.get_additional_termination_information(i).primal_objective);
      // Here we can enforce very low tolerance because the problem is simple so the solution is
      // exact even accounting for scaling
      test_constraint_sanity(ref_problems[i],
                             batch_sol.get_additional_termination_information(i),
                             current_primal_solution,
                             1e-8,
                             false);
    }
  }
}

}  // namespace cuopt::linear_programming::test

CUOPT_TEST_PROGRAM_MAIN()
