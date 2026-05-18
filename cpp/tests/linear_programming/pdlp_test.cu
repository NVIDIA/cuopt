/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <branch_and_bound/shared_strong_branching_context.hpp>
#include <mps_parser_internal.hpp>
#include <pdlp/cusparse_view.hpp>
#include <pdlp/initial_scaling_strategy/initial_scaling.cuh>
#include <pdlp/pdlp.cuh>
#include <pdlp/pdlp_constants.hpp>
#include <pdlp/solve.cuh>
#include <pdlp/utils.cuh>

#include "utilities/pdlp_test_utilities.cuh"

#include "../mip/mip_utils.cuh"

#include <utilities/base_fixture.hpp>
#include <utilities/common_utils.hpp>

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/io/parser.hpp>
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/problem/problem.cuh>

#include <utilities/copy_helpers.hpp>
#include <utilities/error.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/logical.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace cuopt::linear_programming::test {

constexpr double afiro_primal_objective = -464.0;

template <typename T>
rmm::device_uvector<T> extract_subvector(const rmm::device_uvector<T>& vector,
                                         size_t start,
                                         size_t length);

// Accept a 1% error
template <typename f_t>
static bool is_incorrect_objective(f_t reference, f_t objective)
{
  if (reference == 0) { return std::abs(objective) > 0.01; }
  if (objective == 0) { return std::abs(reference) > 0.01; }
  return std::abs((reference - objective) / reference) > 0.01;
}

TEST(pdlp_class, run_double)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings   = pdlp_solver_settings_t<int, double>{};
  solver_settings.method = cuopt::linear_programming::method_t::PDLP;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution.get_additional_termination_information().primal_objective));
}

TEST(pdlp_class, precision_mixed)
{
  using namespace cuopt::linear_programming::detail;
  if (!is_cusparse_runtime_mixed_precision_supported()) {
    const raft::handle_t handle_{};
    auto path = make_path_absolute("linear_programming/afiro_original.mps");
    cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
      cuopt::linear_programming::io::parse_mps<int, double>(path, true);

    auto settings           = pdlp_solver_settings_t<int, double>{};
    settings.method         = cuopt::linear_programming::method_t::PDLP;
    settings.pdlp_precision = cuopt::linear_programming::pdlp_precision_t::MixedPrecision;

    optimization_problem_solution_t<int, double> solution =
      solve_lp(&handle_, op_problem, settings);
    EXPECT_EQ(solution.get_error_status().get_error_type(), cuopt::error_type_t::ValidationError);
    return;
  }

  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto settings_mixed           = pdlp_solver_settings_t<int, double>{};
  settings_mixed.method         = cuopt::linear_programming::method_t::PDLP;
  settings_mixed.pdlp_precision = cuopt::linear_programming::pdlp_precision_t::MixedPrecision;

  optimization_problem_solution_t<int, double> solution_mixed =
    solve_lp(&handle_, op_problem, settings_mixed);
  EXPECT_EQ((int)solution_mixed.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective,
    solution_mixed.get_additional_termination_information().primal_objective));

  auto settings_full           = pdlp_solver_settings_t<int, double>{};
  settings_full.method         = cuopt::linear_programming::method_t::PDLP;
  settings_full.pdlp_precision = cuopt::linear_programming::pdlp_precision_t::DefaultPrecision;

  optimization_problem_solution_t<int, double> solution_full =
    solve_lp(&handle_, op_problem, settings_full);
  EXPECT_EQ((int)solution_full.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective,
    solution_full.get_additional_termination_information().primal_objective));

  EXPECT_NEAR(solution_mixed.get_additional_termination_information().primal_objective,
              solution_full.get_additional_termination_information().primal_objective,
              1e-2);
}

TEST(pdlp_class, concurrent_pdlp_exception_joins_worker_threads)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto settings           = pdlp_solver_settings_t<int, double>{};
  settings.method         = cuopt::linear_programming::method_t::Concurrent;
  settings.presolver      = cuopt::linear_programming::presolver_t::None;
  settings.log_to_console = false;
  // In concurrent mode, dual simplex and barrier workers are started before PDLP validates that
  // all_primal_feasible is batch-only. This exercises the exception path with live worker threads.
  settings.all_primal_feasible = true;

  optimization_problem_solution_t<int, double> solution = solve_lp(&handle_, op_problem, settings);
  const auto error_status                               = solution.get_error_status();

  EXPECT_EQ(error_status.get_error_type(), cuopt::error_type_t::ValidationError);
  EXPECT_THAT(error_status.what(),
              testing::HasSubstr("all_primal_feasible only applies in batch mode"));
}

TEST(pdlp_class, run_double_very_low_accuracy)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

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
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution.get_additional_termination_information().primal_objective));
}

TEST(pdlp_class, run_double_initial_solution)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  std::vector<double> inital_primal_sol(op_problem.get_n_variables());
  std::fill(inital_primal_sol.begin(), inital_primal_sol.end(), 1.0);
  op_problem.set_initial_primal_solution(inital_primal_sol);

  auto solver_settings   = pdlp_solver_settings_t<int, double>{};
  solver_settings.method = cuopt::linear_programming::method_t::PDLP;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution.get_additional_termination_information().primal_objective));
}

TEST(pdlp_class, run_iteration_limit)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings =
    cuopt::linear_programming::pdlp_solver_settings_t<int, double>{};

  settings.iteration_limit = 10;
  // To make sure it doesn't return before the iteration limit
  settings.set_optimality_tolerance(0);
  settings.method = cuopt::linear_programming::method_t::PDLP;

  optimization_problem_solution_t<int, double> solution = solve_lp(&handle_, op_problem, settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_ITERATION_LIMIT);
  // By default we would return all 0, we now return what we currently have so not all 0
  EXPECT_FALSE(thrust::all_of(handle_.get_thrust_policy(),
                              solution.get_primal_solution().begin(),
                              solution.get_primal_solution().end(),
                              thrust::placeholders::_1 == 0.0));
}

TEST(pdlp_class, batch_iteration_limit_updates_additional_termination_stats)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto settings            = pdlp_solver_settings_t<int, double>{};
  settings.iteration_limit = 10;
  settings.set_optimality_tolerance(0);
  settings.method    = method_t::PDLP;
  settings.presolver = presolver_t::None;

  constexpr int batch_size = 2;
  auto solution            = solve_lp_batch_fixed<int, double>(
    &handle_, op_problem, settings, batch_size, {}, {}, {}, {}, true);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  const auto& statuses = solution.get_terminations_status();
  ASSERT_EQ(static_cast<int>(statuses.size()), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_EQ(statuses[i], pdlp_termination_status_t::IterationLimit) << "climber " << i;

    const auto info = solution.get_additional_termination_information(i);
    EXPECT_EQ(info.number_of_steps_taken, settings.iteration_limit) << "climber " << i;
    EXPECT_TRUE(std::isfinite(info.primal_objective)) << "climber " << i;
    EXPECT_TRUE(std::isfinite(info.l2_primal_residual)) << "climber " << i;
    EXPECT_TRUE(std::isfinite(info.l2_dual_residual)) << "climber " << i;
    EXPECT_EQ(info.solved_by, method_t::PDLP) << "climber " << i;
  }
}

TEST(pdlp_class, batch_settings_overrides_preserve_user_limits_and_tolerances)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  constexpr int batch_size           = 2;
  constexpr double tighter_tolerance = 1e-6;

  auto default_settings      = pdlp_solver_settings_t<int, double>{};
  default_settings.method    = method_t::PDLP;
  default_settings.presolver = presolver_t::None;

  auto default_solution =
    solve_lp_batch_fixed<int, double>(&handle_, op_problem, default_settings, batch_size);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  ASSERT_EQ(static_cast<int>(default_solution.get_terminations_status().size()), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_EQ(default_solution.get_termination_status(i), pdlp_termination_status_t::Optimal)
      << "climber " << i;
    auto primal_i = extract_subvector(default_solution.get_primal_solution(),
                                      i * op_problem.get_n_variables(),
                                      op_problem.get_n_variables());
    test_constraint_sanity(op_problem,
                           default_solution.get_additional_termination_information(i),
                           primal_i,
                           default_settings.tolerances.absolute_primal_tolerance);
    // By default we don't meet the 1e-6 relative primal tolerance
    EXPECT_GT(
      default_solution.get_additional_termination_information(i).l2_relative_primal_residual,
      tighter_tolerance)
      << "climber " << i;
  }

  auto tighter_tolerance_settings      = pdlp_solver_settings_t<int, double>{};
  tighter_tolerance_settings.method    = method_t::PDLP;
  tighter_tolerance_settings.presolver = presolver_t::None;
  tighter_tolerance_settings.set_optimality_tolerance(tighter_tolerance);

  auto tighter_tolerance_solution =
    solve_lp_batch_fixed<int, double>(&handle_, op_problem, tighter_tolerance_settings, batch_size);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  ASSERT_EQ(static_cast<int>(tighter_tolerance_solution.get_terminations_status().size()),
            batch_size);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_EQ(tighter_tolerance_solution.get_termination_status(i),
              pdlp_termination_status_t::Optimal)
      << "climber " << i;
    auto primal_i = extract_subvector(tighter_tolerance_solution.get_primal_solution(),
                                      i * op_problem.get_n_variables(),
                                      op_problem.get_n_variables());
    test_constraint_sanity(op_problem,
                           tighter_tolerance_solution.get_additional_termination_information(i),
                           primal_i,
                           tighter_tolerance);
    EXPECT_LE(tighter_tolerance_solution.get_additional_termination_information(i)
                .l2_relative_primal_residual,
              tighter_tolerance)
      << "climber " << i;
  }

  auto iteration_limit_settings            = pdlp_solver_settings_t<int, double>{};
  iteration_limit_settings.method          = method_t::PDLP;
  iteration_limit_settings.presolver       = presolver_t::None;
  iteration_limit_settings.iteration_limit = 10;
  iteration_limit_settings.set_optimality_tolerance(0);

  auto iteration_limit_solution =
    solve_lp_batch_fixed<int, double>(&handle_, op_problem, iteration_limit_settings, batch_size);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  ASSERT_EQ(static_cast<int>(iteration_limit_solution.get_terminations_status().size()),
            batch_size);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_EQ(iteration_limit_solution.get_termination_status(i),
              pdlp_termination_status_t::IterationLimit)
      << "climber " << i;
    EXPECT_EQ(
      iteration_limit_solution.get_additional_termination_information(i).number_of_steps_taken,
      iteration_limit_settings.iteration_limit)
      << "climber " << i;
  }
}

TEST(pdlp_class, run_time_limit)
{
  const raft::handle_t handle_{};
  auto path = make_path_absolute("linear_programming/savsched1/savsched1.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings =
    cuopt::linear_programming::pdlp_solver_settings_t<int, double>{};

  constexpr double time_limit_seconds = 2;
  settings.time_limit                 = time_limit_seconds;
  // To make sure it doesn't return before the time limit
  settings.set_optimality_tolerance(0);
  settings.method = cuopt::linear_programming::method_t::PDLP;

  optimization_problem_solution_t<int, double> solution = solve_lp(&handle_, op_problem, settings);

  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_TIME_LIMIT);
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
    cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
      cuopt::linear_programming::io::parse_mps<int, double>(path);

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
           {std::pair{presolver_t::Papilo, 1e-1}, std::pair{presolver_t::None, 1e-4}}) {
        settings.presolver = presolver;
        settings.method    = cuopt::linear_programming::method_t::PDLP;
        const raft::handle_t handle_{};
        optimization_problem_solution_t<int, double> solution =
          solve_lp(&handle_, op_problem, settings);
        printf("running %s mode %d presolver %d\n",
               name.c_str(),
               (int)solver_mode,
               (int)settings.presolver);
        EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);
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
                               presolver != presolver_t::None);
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
  cuopt::linear_programming::io::mps_data_model_t<int, double> mps_data_model =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

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
  cuopt::linear_programming::io::mps_data_model_t<int, double> mps_data_model =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

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
  solver_settings.pdlp_solver_mode = cuopt::linear_programming::pdlp_solver_mode_t::Stable2;
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
                .element(0, handle.get_stream()),
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
  solver_settings.pdlp_solver_mode        = cuopt::linear_programming::pdlp_solver_mode_t::Stable2;
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem1 =
    cuopt::linear_programming::io::parse_mps<int, double>(path);
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem2 =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

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

  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem1 =
    cuopt::linear_programming::io::parse_mps<int, double>(path);
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem2 =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

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

  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem1 =
    cuopt::linear_programming::io::parse_mps<int, double>(path);
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem2 =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

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

// -- Per constraints redisual, batch and non batch --

TEST(pdlp_class, per_constraint_residual_stable3)
{
  const raft::handle_t handle{};

  auto path                        = make_path_absolute("linear_programming/afiro_original.mps");
  auto solver_settings             = pdlp_solver_settings_t<int, double>{};
  solver_settings.pdlp_solver_mode = pdlp_solver_mode_t::Stable3;
  solver_settings.per_constraint_residual = true;
  solver_settings.presolver               = presolver_t::None;
  solver_settings.method                  = cuopt::linear_programming::method_t::PDLP;

  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  auto sol = solve_lp(&handle, op_problem, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  EXPECT_EQ(sol.get_termination_status(), pdlp_termination_status_t::Optimal);
  test_constraint_sanity_per_row(op_problem,
                                 sol.get_primal_solution(),
                                 solver_settings.tolerances.absolute_primal_tolerance,
                                 solver_settings.tolerances.relative_primal_tolerance);
}

TEST(pdlp_class, batch_per_constraint_residual_stable3)
{
  const raft::handle_t handle{};

  auto path                        = make_path_absolute("linear_programming/afiro_original.mps");
  auto solver_settings             = pdlp_solver_settings_t<int, double>{};
  solver_settings.pdlp_solver_mode = pdlp_solver_mode_t::Stable3;
  solver_settings.per_constraint_residual = true;
  solver_settings.presolver               = presolver_t::None;
  solver_settings.method                  = cuopt::linear_programming::method_t::PDLP;

  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  constexpr int batch_size = 2;

  // Mock a batch of size 2
  solver_settings.fixed_batch_size = batch_size;
  auto batch_sol                   = solve_lp<int, double>(&handle, op_problem, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  const auto& statuses = batch_sol.get_terminations_status();
  ASSERT_EQ((int)statuses.size(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_EQ(statuses[i], pdlp_termination_status_t::Optimal) << "climber " << i;
  }
  // Both iteration count should be the same
  EXPECT_EQ(batch_sol.get_additional_termination_information(0).number_of_steps_taken,
            batch_sol.get_additional_termination_information(1).number_of_steps_taken);

  const size_t primal_size = op_problem.get_n_variables();

  const auto primal_0 =
    extract_subvector(batch_sol.get_primal_solution(), 0 * primal_size, primal_size);
  test_constraint_sanity_per_row(op_problem,
                                 primal_0,
                                 solver_settings.tolerances.absolute_primal_tolerance,
                                 solver_settings.tolerances.relative_primal_tolerance);

  const auto primal_1 =
    extract_subvector(batch_sol.get_primal_solution(), 1 * primal_size, primal_size);
  test_constraint_sanity_per_row(op_problem,
                                 primal_1,
                                 solver_settings.tolerances.absolute_primal_tolerance,
                                 solver_settings.tolerances.relative_primal_tolerance);
}

TEST(pdlp_class, batch_per_constraint_residual_different_rhs_stable3)
{
  const raft::handle_t handle{};

  auto path                        = make_path_absolute("linear_programming/afiro_original.mps");
  auto solver_settings             = pdlp_solver_settings_t<int, double>{};
  solver_settings.pdlp_solver_mode = pdlp_solver_mode_t::Stable3;
  solver_settings.per_constraint_residual = true;
  solver_settings.presolver               = presolver_t::None;
  solver_settings.method                  = cuopt::linear_programming::method_t::PDLP;

  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  // Build two climbers that share A and variable bounds but differ on the constraint
  // lower/upper bounds (RHS): climber 0 keeps the original, climber 1 finite bounds get set to 100
  constexpr int batch_size          = 2;
  const std::vector<double> orig_lb = op_problem.get_constraint_lower_bounds();
  const std::vector<double> orig_ub = op_problem.get_constraint_upper_bounds();
  const size_t n_cons               = orig_lb.size();
  std::vector<double> climber1_lb   = orig_lb;
  std::vector<double> climber1_ub   = orig_ub;
  constexpr double new_rhs          = 100.0;
  for (size_t i = 0; i < n_cons; ++i) {
    if (std::isfinite(climber1_ub[i])) climber1_ub[i] = new_rhs;
  }

  // Expand the bounds on the mps_data_model_t before dispatching: solve_lp_batch_fixed
  // converts the model to an optimization_problem_t and resizes the device-side bound
  // vectors directly from these host arrays, so the expanded (batch_size * n_cons)
  // layout must already be present here.
  std::vector<double> per_climber_lb;
  std::vector<double> per_climber_ub;
  per_climber_lb.reserve(batch_size * n_cons);
  per_climber_ub.reserve(batch_size * n_cons);
  per_climber_lb.insert(per_climber_lb.end(), orig_lb.begin(), orig_lb.end());
  per_climber_ub.insert(per_climber_ub.end(), orig_ub.begin(), orig_ub.end());
  per_climber_lb.insert(per_climber_lb.end(), climber1_lb.begin(), climber1_lb.end());
  per_climber_ub.insert(per_climber_ub.end(), climber1_ub.begin(), climber1_ub.end());

  // Don't call set_constraint_lower_bounds and set_constraint_upper_bounds to avoid changing the
  // n_constraints_

  auto batch_sol = solve_lp_batch_fixed<int, double>(
    &handle, op_problem, solver_settings, batch_size, {}, per_climber_lb, per_climber_ub);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  const auto& statuses = batch_sol.get_terminations_status();
  ASSERT_EQ((int)statuses.size(), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_EQ(statuses[i], pdlp_termination_status_t::Optimal) << "climber " << i;
  }

  const size_t primal_size = op_problem.get_n_variables();

  // Reload the original (single-climber) problem and build per-climber views so the
  // per-row sanity check evaluates each solution against its own constraint bounds.
  auto climber0_problem = cuopt::linear_programming::io::parse_mps<int, double>(path);
  auto climber1_problem = cuopt::linear_programming::io::parse_mps<int, double>(path);
  climber1_problem.set_constraint_lower_bounds({climber1_lb.data(), climber1_lb.size()});
  climber1_problem.set_constraint_upper_bounds({climber1_ub.data(), climber1_ub.size()});

  const auto primal_0 =
    extract_subvector(batch_sol.get_primal_solution(), 0 * primal_size, primal_size);
  test_constraint_sanity_per_row(climber0_problem,
                                 primal_0,
                                 solver_settings.tolerances.absolute_primal_tolerance,
                                 solver_settings.tolerances.relative_primal_tolerance);

  const auto primal_1 =
    extract_subvector(batch_sol.get_primal_solution(), 1 * primal_size, primal_size);
  test_constraint_sanity_per_row(climber1_problem,
                                 primal_1,
                                 solver_settings.tolerances.absolute_primal_tolerance,
                                 solver_settings.tolerances.relative_primal_tolerance);
}

// -------------------------------------------------------------

// -- First primal feasible, batch and non batch --

TEST(pdlp_class, first_primal_feasible_stable3)
{
  const raft::handle_t handle{};

  auto path            = make_path_absolute("linear_programming/ns1687037/ns1687037.mps");
  auto solver_settings = pdlp_solver_settings_t<int, double>{};
  constexpr double kOptimalityTolerance = 1e-2;
  solver_settings.iteration_limit       = 1000;
  solver_settings.set_optimality_tolerance(kOptimalityTolerance);
  solver_settings.pdlp_solver_mode = pdlp_solver_mode_t::Stable3;
  solver_settings.method           = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver        = presolver_t::None;

  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  // Wihout first primal feasible we hit iteration limit
  auto sol_base = solve_lp(&handle, op_problem, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  EXPECT_EQ(sol_base.get_termination_status(), pdlp_termination_status_t::IterationLimit);

  solver_settings.first_primal_feasible = true;
  auto sol_fpf                          = solve_lp(&handle, op_problem, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  EXPECT_EQ(sol_fpf.get_termination_status(), pdlp_termination_status_t::PrimalFeasible);

  test_objective_sanity(op_problem,
                        sol_fpf.get_primal_solution(),
                        sol_fpf.get_additional_termination_information().primal_objective,
                        kOptimalityTolerance);
  test_constraint_sanity(op_problem,
                         sol_fpf.get_additional_termination_information(),
                         sol_fpf.get_primal_solution(),
                         kOptimalityTolerance);
}

TEST(pdlp_class, first_primal_feasible_batch_stable3)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/ns1687037/ns1687037.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  auto solver_settings                  = pdlp_solver_settings_t<int, double>{};
  solver_settings.method                = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode      = pdlp_solver_mode_t::Stable3;
  solver_settings.iteration_limit       = 1000;
  solver_settings.first_primal_feasible = true;
  constexpr double kOptimalityTolerance = 1e-2;
  solver_settings.set_optimality_tolerance(kOptimalityTolerance);
  solver_settings.presolver = presolver_t::None;

  constexpr int batch_size = 2;

  solver_settings.fixed_batch_size = batch_size;
  auto sol                         = solve_lp(&handle_, op_problem, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  const auto& statuses = sol.get_terminations_status();
  ASSERT_EQ((int)statuses.size(), batch_size);

  // All should be primal feasible
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_EQ(statuses[i], pdlp_termination_status_t::PrimalFeasible) << "climber " << i;
  }
  // Should have same number of steps taken
  EXPECT_EQ(sol.get_additional_termination_information(0).number_of_steps_taken,
            sol.get_additional_termination_information(1).number_of_steps_taken);

  // Should all respect the sanity checks
  for (int i = 0; i < batch_size; ++i) {
    auto primal_i = extract_subvector(
      sol.get_primal_solution(), i * op_problem.get_n_variables(), op_problem.get_n_variables());
    test_objective_sanity(op_problem,
                          primal_i,
                          sol.get_additional_termination_information(i).primal_objective,
                          kOptimalityTolerance);
    test_constraint_sanity(
      op_problem, sol.get_additional_termination_information(i), primal_i, kOptimalityTolerance);
  }
}

TEST(pdlp_class, first_primal_feasible_batch_different_rhs_stable3)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/ns1687037/ns1687037.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  auto solver_settings                  = pdlp_solver_settings_t<int, double>{};
  solver_settings.method                = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode      = pdlp_solver_mode_t::Stable3;
  solver_settings.iteration_limit       = 1000;
  solver_settings.first_primal_feasible = true;
  constexpr double kOptimalityTolerance = 1e-2;
  solver_settings.set_optimality_tolerance(kOptimalityTolerance);
  solver_settings.presolver = presolver_t::None;

  constexpr int batch_size = 2;

  std::vector<double> per_climber_lb;
  std::vector<double> per_climber_ub;
  per_climber_lb.resize(batch_size * op_problem.get_n_constraints());
  per_climber_ub.resize(batch_size * op_problem.get_n_constraints());
  std::copy(op_problem.get_constraint_lower_bounds().begin(),
            op_problem.get_constraint_lower_bounds().end(),
            per_climber_lb.begin());
  std::copy(op_problem.get_constraint_upper_bounds().begin(),
            op_problem.get_constraint_upper_bounds().end(),
            per_climber_ub.begin());
  // Make the second climber infeasible but since we stop at first primal feasible, it should be
  // fine
  std::fill(per_climber_lb.begin() + op_problem.get_n_constraints(), per_climber_lb.end(), 1000.0);
  std::fill(per_climber_ub.begin() + op_problem.get_n_constraints(), per_climber_ub.end(), 1000.0);

  auto sol = solve_lp_batch_fixed(&handle_,
                                  op_problem,
                                  solver_settings,
                                  batch_size,
                                  {},
                                  per_climber_lb,
                                  per_climber_ub,
                                  {},
                                  true);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  const auto& statuses = sol.get_terminations_status();
  ASSERT_EQ((int)statuses.size(), batch_size);

  // Climber one should be primal feasible, climber two should be no termination as we stop on first
  // primal feasible
  EXPECT_EQ(statuses[0], pdlp_termination_status_t::PrimalFeasible);
  EXPECT_EQ(statuses[1], pdlp_termination_status_t::NoTermination);

  // Should all respect the sanity checks
  auto primal_0 = extract_subvector(
    sol.get_primal_solution(), 0 * op_problem.get_n_variables(), op_problem.get_n_variables());
  test_objective_sanity(op_problem,
                        primal_0,
                        sol.get_additional_termination_information(0).primal_objective,
                        kOptimalityTolerance);
  test_constraint_sanity(
    op_problem, sol.get_additional_termination_information(0), primal_0, kOptimalityTolerance);
}

TEST(pdlp_class, all_primal_feasible_batch_different_rhs_stable3)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/ns1687037/ns1687037.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  auto solver_settings                  = pdlp_solver_settings_t<int, double>{};
  solver_settings.method                = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode      = pdlp_solver_mode_t::Stable3;
  solver_settings.iteration_limit       = 1000;
  solver_settings.all_primal_feasible   = true;
  constexpr double kOptimalityTolerance = 1e-2;
  solver_settings.set_optimality_tolerance(kOptimalityTolerance);
  solver_settings.presolver = presolver_t::None;

  constexpr int batch_size = 2;

  std::vector<double> per_climber_lb;
  std::vector<double> per_climber_ub;
  per_climber_lb.resize(batch_size * op_problem.get_n_constraints());
  per_climber_ub.resize(batch_size * op_problem.get_n_constraints());
  std::copy(op_problem.get_constraint_lower_bounds().begin(),
            op_problem.get_constraint_lower_bounds().end(),
            per_climber_lb.begin());
  std::copy(op_problem.get_constraint_upper_bounds().begin(),
            op_problem.get_constraint_upper_bounds().end(),
            per_climber_ub.begin());
  // Make the second climber infeasible but since we stop at first primal feasible, it should be
  // fine
  std::fill(per_climber_lb.begin() + op_problem.get_n_constraints(), per_climber_lb.end(), 1000.0);
  std::fill(per_climber_ub.begin() + op_problem.get_n_constraints(), per_climber_ub.end(), 1000.0);

  auto sol = solve_lp_batch_fixed(&handle_,
                                  op_problem,
                                  solver_settings,
                                  batch_size,
                                  {},
                                  per_climber_lb,
                                  per_climber_ub,
                                  {},
                                  true);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  const auto& statuses = sol.get_terminations_status();
  ASSERT_EQ((int)statuses.size(), batch_size);

  // Climber one should be primal feasible, climber two should be iteration limit
  EXPECT_EQ(statuses[0], pdlp_termination_status_t::PrimalFeasible);
  EXPECT_EQ(statuses[1], pdlp_termination_status_t::IterationLimit);

  // Should all respect the sanity checks
  auto primal_0 = extract_subvector(
    sol.get_primal_solution(), 0 * op_problem.get_n_variables(), op_problem.get_n_variables());
  test_objective_sanity(op_problem,
                        primal_0,
                        sol.get_additional_termination_information(0).primal_objective,
                        kOptimalityTolerance);
  test_constraint_sanity(
    op_problem, sol.get_additional_termination_information(0), primal_0, kOptimalityTolerance);
}

// -- First primal feasible and per constraint residual, batch and non batch --

TEST(pdlp_class, first_primal_feasible_and_per_constraint_residual_stable3)
{
  const raft::handle_t handle{};

  auto path            = make_path_absolute("linear_programming/ns1687037/ns1687037.mps");
  auto solver_settings = pdlp_solver_settings_t<int, double>{};
  solver_settings.pdlp_solver_mode        = pdlp_solver_mode_t::Stable3;
  solver_settings.first_primal_feasible   = true;
  solver_settings.per_constraint_residual = true;
  constexpr double kOptimalityTolerance   = 1e-2;
  solver_settings.set_optimality_tolerance(kOptimalityTolerance);
  solver_settings.presolver = presolver_t::None;
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;

  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  auto sol = solve_lp(&handle, op_problem, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  EXPECT_EQ(sol.get_termination_status(), pdlp_termination_status_t::PrimalFeasible);

  test_objective_sanity(op_problem,
                        sol.get_primal_solution(),
                        sol.get_additional_termination_information().primal_objective,
                        kOptimalityTolerance);
  test_constraint_sanity_per_row(op_problem,
                                 sol.get_primal_solution(),
                                 solver_settings.tolerances.absolute_primal_tolerance,
                                 solver_settings.tolerances.relative_primal_tolerance);
}

TEST(pdlp_class, first_primal_feasible_and_per_constraint_residual_batch_stable3)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/ns1687037/ns1687037.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  auto solver_settings                    = pdlp_solver_settings_t<int, double>{};
  solver_settings.method                  = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode        = pdlp_solver_mode_t::Stable3;
  solver_settings.iteration_limit         = 1000;
  solver_settings.first_primal_feasible   = true;
  solver_settings.per_constraint_residual = true;
  constexpr double kOptimalityTolerance   = 1e-2;
  solver_settings.set_optimality_tolerance(kOptimalityTolerance);
  solver_settings.presolver = presolver_t::None;

  constexpr int batch_size = 2;

  solver_settings.fixed_batch_size = batch_size;
  auto sol                         = solve_lp(&handle_, op_problem, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  const auto& statuses = sol.get_terminations_status();
  ASSERT_EQ((int)statuses.size(), batch_size);

  // Climber one should be primal feasible, climber two should be no termination as we stop on first
  // primal feasible
  EXPECT_EQ(statuses[0], pdlp_termination_status_t::PrimalFeasible);
  EXPECT_EQ(statuses[1], pdlp_termination_status_t::PrimalFeasible);

  // Should all respect the sanity checks
  for (int i = 0; i < batch_size; ++i) {
    auto primal_i = extract_subvector(
      sol.get_primal_solution(), i * op_problem.get_n_variables(), op_problem.get_n_variables());
    test_objective_sanity(op_problem,
                          primal_i,
                          sol.get_additional_termination_information(i).primal_objective,
                          kOptimalityTolerance);
    test_constraint_sanity_per_row(op_problem,
                                   primal_i,
                                   solver_settings.tolerances.absolute_primal_tolerance,
                                   solver_settings.tolerances.relative_primal_tolerance);
  }
}

TEST(pdlp_class, first_primal_feasible_and_per_constraint_residual_batch_different_rhs_stable3)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/ns1687037/ns1687037.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  auto solver_settings                    = pdlp_solver_settings_t<int, double>{};
  solver_settings.method                  = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode        = pdlp_solver_mode_t::Stable3;
  solver_settings.iteration_limit         = 1000;
  solver_settings.first_primal_feasible   = true;
  solver_settings.per_constraint_residual = true;
  constexpr double kOptimalityTolerance   = 1e-2;
  solver_settings.set_optimality_tolerance(kOptimalityTolerance);
  solver_settings.presolver = presolver_t::None;

  constexpr int batch_size = 2;

  std::vector<double> per_climber_lb;
  std::vector<double> per_climber_ub;
  per_climber_lb.resize(batch_size * op_problem.get_n_constraints());
  per_climber_ub.resize(batch_size * op_problem.get_n_constraints());
  std::copy(op_problem.get_constraint_lower_bounds().begin(),
            op_problem.get_constraint_lower_bounds().end(),
            per_climber_lb.begin());
  std::copy(op_problem.get_constraint_upper_bounds().begin(),
            op_problem.get_constraint_upper_bounds().end(),
            per_climber_ub.begin());
  // Make the second climber infeasible but since we stop at first primal feasible, it should be
  // fine
  std::fill(per_climber_lb.begin() + op_problem.get_n_constraints(), per_climber_lb.end(), 1000.0);
  std::fill(per_climber_ub.begin() + op_problem.get_n_constraints(), per_climber_ub.end(), 1000.0);

  auto sol = solve_lp_batch_fixed(&handle_,
                                  op_problem,
                                  solver_settings,
                                  batch_size,
                                  {},
                                  per_climber_lb,
                                  per_climber_ub,
                                  {},
                                  true);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  const auto& statuses = sol.get_terminations_status();
  ASSERT_EQ((int)statuses.size(), batch_size);

  // Climber one should be primal feasible, climber two should be no termination as we stop on first
  // primal feasible
  EXPECT_EQ(statuses[0], pdlp_termination_status_t::PrimalFeasible);
  EXPECT_EQ(statuses[1], pdlp_termination_status_t::NoTermination);

  // Should all respect the sanity checks
  auto primal_0 = extract_subvector(
    sol.get_primal_solution(), 0 * op_problem.get_n_variables(), op_problem.get_n_variables());
  test_objective_sanity(op_problem,
                        primal_0,
                        sol.get_additional_termination_information(0).primal_objective,
                        kOptimalityTolerance);
  test_constraint_sanity_per_row(op_problem,
                                 primal_0,
                                 solver_settings.tolerances.absolute_primal_tolerance,
                                 solver_settings.tolerances.relative_primal_tolerance);
}

TEST(pdlp_class, all_primal_feasible_and_per_constraint_residual_batch_different_rhs_stable3)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/ns1687037/ns1687037.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  auto solver_settings                    = pdlp_solver_settings_t<int, double>{};
  solver_settings.method                  = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode        = pdlp_solver_mode_t::Stable3;
  solver_settings.iteration_limit         = 1000;
  solver_settings.all_primal_feasible     = true;
  solver_settings.per_constraint_residual = true;
  constexpr double kOptimalityTolerance   = 1e-2;
  solver_settings.set_optimality_tolerance(kOptimalityTolerance);
  solver_settings.presolver = presolver_t::None;

  constexpr int batch_size = 2;

  std::vector<double> per_climber_lb;
  std::vector<double> per_climber_ub;
  per_climber_lb.resize(batch_size * op_problem.get_n_constraints());
  per_climber_ub.resize(batch_size * op_problem.get_n_constraints());
  std::copy(op_problem.get_constraint_lower_bounds().begin(),
            op_problem.get_constraint_lower_bounds().end(),
            per_climber_lb.begin());
  std::copy(op_problem.get_constraint_upper_bounds().begin(),
            op_problem.get_constraint_upper_bounds().end(),
            per_climber_ub.begin());
  // Make the second climber infeasible but since we stop at first primal feasible, it should be
  // fine
  std::fill(per_climber_lb.begin() + op_problem.get_n_constraints(), per_climber_lb.end(), 1000.0);
  std::fill(per_climber_ub.begin() + op_problem.get_n_constraints(), per_climber_ub.end(), 1000.0);

  auto sol = solve_lp_batch_fixed(&handle_,
                                  op_problem,
                                  solver_settings,
                                  batch_size,
                                  {},
                                  per_climber_lb,
                                  per_climber_ub,
                                  {},
                                  true);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  const auto& statuses = sol.get_terminations_status();
  ASSERT_EQ((int)statuses.size(), batch_size);

  // Climber one should be primal feasible, climber two should be no termination as we stop on first
  // primal feasible
  EXPECT_EQ(statuses[0], pdlp_termination_status_t::PrimalFeasible);
  EXPECT_EQ(statuses[1], pdlp_termination_status_t::IterationLimit);

  // Should all respect the sanity checks
  auto primal_0 = extract_subvector(
    sol.get_primal_solution(), 0 * op_problem.get_n_variables(), op_problem.get_n_variables());
  test_objective_sanity(op_problem,
                        primal_0,
                        sol.get_additional_termination_information(0).primal_objective,
                        kOptimalityTolerance);
  test_constraint_sanity_per_row(op_problem,
                                 primal_0,
                                 solver_settings.tolerances.absolute_primal_tolerance,
                                 solver_settings.tolerances.relative_primal_tolerance);
}

TEST(pdlp_class, all_primal_feasible_and_per_constraint_residual_batch_many_different_rhs_stable3_1)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/ns1687037/ns1687037.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  auto solver_settings                    = pdlp_solver_settings_t<int, double>{};
  solver_settings.method                  = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode        = pdlp_solver_mode_t::Stable3;
  solver_settings.iteration_limit         = 1000;
  solver_settings.all_primal_feasible     = true;
  solver_settings.per_constraint_residual = true;
  constexpr double kOptimalityTolerance   = 1e-2;
  solver_settings.set_optimality_tolerance(kOptimalityTolerance);
  solver_settings.presolver = presolver_t::None;

  const auto& original_lb    = op_problem.get_constraint_lower_bounds();
  const auto& original_ub    = op_problem.get_constraint_upper_bounds();
  const size_t n_constraints = op_problem.get_n_constraints();
  const size_t n_variables   = op_problem.get_n_variables();

  const std::vector<double> rhs_relaxations = {
    1000.0, 0.0, 2500.0, 1.0, 500.0, 250.0, 100.0, 10.0, 10000.0, 5000.0, 50.0};
  const int batch_size = static_cast<int>(rhs_relaxations.size());

  std::vector<double> per_climber_lb;
  std::vector<double> per_climber_ub;
  per_climber_lb.reserve(static_cast<size_t>(batch_size) * n_constraints);
  per_climber_ub.reserve(static_cast<size_t>(batch_size) * n_constraints);

  std::vector<double> ref_objectives(batch_size);
  std::vector<pdlp_termination_status_t> ref_statuses(batch_size);
  std::vector<std::vector<double>> ref_primal_solutions(batch_size);
  std::vector<int> ref_iteration_counts(batch_size);
  std::vector<cuopt::linear_programming::io::mps_data_model_t<int, double>> ref_problems;
  ref_problems.reserve(batch_size);

  auto ref_solver_settings                  = solver_settings;
  ref_solver_settings.all_primal_feasible   = false;
  ref_solver_settings.first_primal_feasible = true;

  for (int i = 0; i < batch_size; ++i) {
    std::vector<double> climber_lb = original_lb;
    std::vector<double> climber_ub = original_ub;
    const double relaxation        = rhs_relaxations[i];
    for (size_t c = 0; c < n_constraints; ++c) {
      if (std::isfinite(climber_lb[c])) { climber_lb[c] -= relaxation; }
      if (std::isfinite(climber_ub[c])) { climber_ub[c] += relaxation; }
    }

    auto ref_problem = op_problem;
    ref_problem.set_constraint_lower_bounds({climber_lb.data(), n_constraints});
    ref_problem.set_constraint_upper_bounds({climber_ub.data(), n_constraints});
    ref_problems.push_back(ref_problem);

    per_climber_lb.insert(per_climber_lb.end(), climber_lb.begin(), climber_lb.end());
    per_climber_ub.insert(per_climber_ub.end(), climber_ub.begin(), climber_ub.end());

    auto ref_solution = solve_lp(&handle_, ref_problems.back(), ref_solver_settings);
    ref_statuses[i]   = ref_solution.get_termination_status(0);
    ref_objectives[i] = ref_solution.get_additional_termination_information(0).primal_objective;
    ref_primal_solutions[i] =
      host_copy(ref_solution.get_primal_solution(), ref_solution.get_primal_solution().stream());
    ref_iteration_counts[i] =
      ref_solution.get_additional_termination_information(0).number_of_steps_taken;
    EXPECT_EQ(ref_statuses[i], pdlp_termination_status_t::PrimalFeasible) << "climber " << i;
  }

  auto batch_sol = solve_lp_batch_fixed(&handle_,
                                        op_problem,
                                        solver_settings,
                                        batch_size,
                                        {},
                                        per_climber_lb,
                                        per_climber_ub,
                                        {},
                                        true);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  ASSERT_EQ(static_cast<int>(batch_sol.get_terminations_status().size()), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_EQ(batch_sol.get_termination_status(i), ref_statuses[i]) << "climber " << i;
    EXPECT_NEAR(
      batch_sol.get_additional_termination_information(i).primal_objective, ref_objectives[i], 1e-4)
      << "climber " << i;
    // Same iteration count
    EXPECT_EQ(batch_sol.get_additional_termination_information(i).number_of_steps_taken,
              ref_iteration_counts[i]);

    auto primal_i =
      extract_subvector(batch_sol.get_primal_solution(), i * n_variables, n_variables);
    auto host_primal_i = host_copy(primal_i, primal_i.stream());
    ASSERT_EQ(host_primal_i.size(), ref_primal_solutions[i].size()) << "climber " << i;
    for (size_t p = 0; p < host_primal_i.size(); ++p) {
      EXPECT_NEAR(host_primal_i[p], ref_primal_solutions[i][p], 1e-4)
        << "climber " << i << ", primal index " << p;
    }

    test_objective_sanity(ref_problems[i],
                          primal_i,
                          batch_sol.get_additional_termination_information(i).primal_objective,
                          kOptimalityTolerance);
    test_constraint_sanity_per_row(ref_problems[i],
                                   primal_i,
                                   solver_settings.tolerances.absolute_primal_tolerance,
                                   solver_settings.tolerances.relative_primal_tolerance);
  }
}

TEST(pdlp_class, all_primal_feasible_and_per_constraint_residual_batch_many_different_rhs_stable3_2)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/ns1687037/ns1687037.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  auto solver_settings                    = pdlp_solver_settings_t<int, double>{};
  solver_settings.method                  = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode        = pdlp_solver_mode_t::Stable3;
  solver_settings.iteration_limit         = 1000;
  solver_settings.all_primal_feasible     = true;
  solver_settings.per_constraint_residual = true;
  constexpr double kOptimalityTolerance   = 1e-2;
  solver_settings.set_optimality_tolerance(kOptimalityTolerance);
  solver_settings.presolver = presolver_t::None;

  const auto& original_lb    = op_problem.get_constraint_lower_bounds();
  const auto& original_ub    = op_problem.get_constraint_upper_bounds();
  const size_t n_constraints = op_problem.get_n_constraints();
  const size_t n_variables   = op_problem.get_n_variables();

  const std::vector<double> rhs_relaxations = {
    0.0, 1.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0};
  const int batch_size = static_cast<int>(rhs_relaxations.size());

  std::vector<double> per_climber_lb;
  std::vector<double> per_climber_ub;
  per_climber_lb.reserve(static_cast<size_t>(batch_size) * n_constraints);
  per_climber_ub.reserve(static_cast<size_t>(batch_size) * n_constraints);

  std::vector<double> ref_objectives(batch_size);
  std::vector<pdlp_termination_status_t> ref_statuses(batch_size);
  std::vector<std::vector<double>> ref_primal_solutions(batch_size);
  std::vector<int> ref_iteration_counts(batch_size);
  std::vector<cuopt::linear_programming::io::mps_data_model_t<int, double>> ref_problems;
  ref_problems.reserve(batch_size);

  auto ref_solver_settings                  = solver_settings;
  ref_solver_settings.all_primal_feasible   = false;
  ref_solver_settings.first_primal_feasible = true;

  for (int i = 0; i < batch_size; ++i) {
    std::vector<double> climber_lb = original_lb;
    std::vector<double> climber_ub = original_ub;
    const double relaxation        = rhs_relaxations[i];
    for (size_t c = 0; c < n_constraints; ++c) {
      if (std::isfinite(climber_lb[c])) { climber_lb[c] -= relaxation; }
      if (std::isfinite(climber_ub[c])) { climber_ub[c] += relaxation; }
    }

    auto ref_problem = op_problem;
    ref_problem.set_constraint_lower_bounds({climber_lb.data(), n_constraints});
    ref_problem.set_constraint_upper_bounds({climber_ub.data(), n_constraints});
    ref_problems.push_back(ref_problem);

    per_climber_lb.insert(per_climber_lb.end(), climber_lb.begin(), climber_lb.end());
    per_climber_ub.insert(per_climber_ub.end(), climber_ub.begin(), climber_ub.end());

    auto ref_solution = solve_lp(&handle_, ref_problems.back(), ref_solver_settings);
    ref_statuses[i]   = ref_solution.get_termination_status(0);
    ref_objectives[i] = ref_solution.get_additional_termination_information(0).primal_objective;
    ref_primal_solutions[i] =
      host_copy(ref_solution.get_primal_solution(), ref_solution.get_primal_solution().stream());
    ref_iteration_counts[i] =
      ref_solution.get_additional_termination_information(0).number_of_steps_taken;
    EXPECT_EQ(ref_statuses[i], pdlp_termination_status_t::PrimalFeasible) << "climber " << i;
  }

  auto batch_sol = solve_lp_batch_fixed(&handle_,
                                        op_problem,
                                        solver_settings,
                                        batch_size,
                                        {},
                                        per_climber_lb,
                                        per_climber_ub,
                                        {},
                                        true);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  ASSERT_EQ(static_cast<int>(batch_sol.get_terminations_status().size()), batch_size);
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_EQ(batch_sol.get_termination_status(i), ref_statuses[i]) << "climber " << i;
    EXPECT_NEAR(
      batch_sol.get_additional_termination_information(i).primal_objective, ref_objectives[i], 1e-4)
      << "climber " << i;
    // Same iteration count
    EXPECT_EQ(batch_sol.get_additional_termination_information(i).number_of_steps_taken,
              ref_iteration_counts[i]);

    auto primal_i =
      extract_subvector(batch_sol.get_primal_solution(), i * n_variables, n_variables);
    auto host_primal_i = host_copy(primal_i, primal_i.stream());
    ASSERT_EQ(host_primal_i.size(), ref_primal_solutions[i].size()) << "climber " << i;
    for (size_t p = 0; p < host_primal_i.size(); ++p) {
      EXPECT_NEAR(host_primal_i[p], ref_primal_solutions[i][p], 1e-4)
        << "climber " << i << ", primal index " << p;
    }

    test_objective_sanity(ref_problems[i],
                          primal_i,
                          batch_sol.get_additional_termination_information(i).primal_objective,
                          kOptimalityTolerance);
    test_constraint_sanity_per_row(ref_problems[i],
                                   primal_i,
                                   solver_settings.tolerances.absolute_primal_tolerance,
                                   solver_settings.tolerances.relative_primal_tolerance);
  }
}

TEST(pdlp_class, batch_primal_feasible_non_batch_rejected)
{
  const raft::handle_t handle_{};
  auto path = make_path_absolute("linear_programming/ns1687037/ns1687037.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  auto solver_settings                = pdlp_solver_settings_t<int, double>{};
  solver_settings.method              = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode    = pdlp_solver_mode_t::Stable3;
  solver_settings.presolver           = presolver_t::None;
  solver_settings.all_primal_feasible = true;

  auto sol = solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ(sol.get_error_status().get_error_type(), cuopt::error_type_t::ValidationError);
}

TEST(pdlp_class, first_primal_feasible_and_batch_primal_feasible_rejected)
{
  const raft::handle_t handle_{};
  auto path = make_path_absolute("linear_programming/ns1687037/ns1687037.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  auto solver_settings                  = pdlp_solver_settings_t<int, double>{};
  solver_settings.method                = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode      = pdlp_solver_mode_t::Stable3;
  solver_settings.presolver             = presolver_t::None;
  solver_settings.first_primal_feasible = true;
  solver_settings.all_primal_feasible   = true;

  auto sol = solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ(sol.get_error_status().get_error_type(), cuopt::error_type_t::ValidationError);
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

    cuopt::linear_programming::io::mps_data_model_t<int, double> mps_data_model =
      cuopt::linear_programming::io::parse_mps<int, double>(path);
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

  cuopt::linear_programming::io::mps_data_model_t<int, double> mps_data_model =
    cuopt::linear_programming::io::parse_mps<int, double>(path);
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
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::Papilo;

  {
    solver_settings.dual_postsolve = true;
    optimization_problem_solution_t<int, double> solution =
      solve_lp(&handle_, op_problem, solver_settings);
    EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);
    EXPECT_EQ(solution.get_dual_solution().size(), op_problem.get_n_constraints());
  }

  {
    solver_settings.dual_postsolve = false;
    optimization_problem_solution_t<int, double> solution =
      solve_lp(&handle_, op_problem, solver_settings);
    EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);
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
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

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
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_NUMERICAL_ERROR);
}

// Should run thanks to Dual Simplex
TEST(pdlp_class, run_empty_matrix_dual_simplex)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/empty_matrix.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::Concurrent;
  solver_settings.presolver = presolver_t::None;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_EQ(solution.get_additional_termination_information().solved_by, method_t::DualSimplex);
}

TEST(pdlp_class, test_max)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/good-max.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  auto solver_settings             = pdlp_solver_settings_t<int, double>{};
  solver_settings.method           = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode = cuopt::linear_programming::pdlp_solver_mode_t::Stable2;
  solver_settings.presolver        = presolver_t::None;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_NEAR(
    solution.get_additional_termination_information().primal_objective, 17.0, factor_tolerance);
}

TEST(pdlp_class, test_max_with_offset)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/max_offset.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_NEAR(
    solution.get_additional_termination_information().primal_objective, 0.0, factor_tolerance);
}

TEST(pdlp_class, test_lp_no_constraints)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/lp-model-no-constraints.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.presolver = presolver_t::None;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);
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
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

  constexpr int batch_size = 5;

  // Setup a larger batch afiro but with all same primal/dual bounds

  const auto& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const auto& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  for (size_t i = 0; i < batch_size; i++) {
    solver_settings.new_bounds.push_back(
      {static_cast<int>(i), 0, variable_lower_bounds[0], variable_upper_bounds[0]});
  }

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);

  // All should be optimal with the right objective
  for (size_t i = 0; i < batch_size; ++i) {
    EXPECT_EQ((int)solution.get_termination_status(i), CUOPT_TERMINATION_STATUS_OPTIMAL);
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
  test_constraint_sanity(
    op_problem, solution.get_additional_termination_information(0), primal_solution, 1e-4, true);
}

TEST(pdlp_class, simple_batch_different_bounds)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

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
  solver_settings.new_bounds.push_back({0, 5, 4.0, 5.0});
  // The second climber has no changes
  solver_settings.new_bounds.push_back({1, 0, variable_lower_bounds[0], variable_upper_bounds[0]});

  const auto new_primal = solution.get_additional_termination_information(0).primal_objective;

  // Now setup and solve batch
  optimization_problem_solution_t<int, double> solution2 =
    solve_lp(&handle_, op_problem, solver_settings);

  // Both should be optimal
  // Climber #0 should have same objective as ref and #1 as the usual
  EXPECT_EQ((int)solution2.get_termination_status(0), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(is_incorrect_objective(
    new_primal, solution2.get_additional_termination_information(0).primal_objective));
  EXPECT_EQ((int)solution2.get_termination_status(1), CUOPT_TERMINATION_STATUS_OPTIMAL);
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
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

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
  solver_settings.new_bounds.push_back({0, 0, variable_lower_bounds[0], variable_upper_bounds[0]});
  // Climber #1: var 5 -> [4.0, 5.0]
  solver_settings.new_bounds.push_back({1, 5, 4.0, 5.0});
  // Climber #2: no-op
  solver_settings.new_bounds.push_back({2, 0, variable_lower_bounds[0], variable_upper_bounds[0]});
  // Climber #3: var 1 -> [-7.0, 13.0]
  solver_settings.new_bounds.push_back({3, 1, -7.0, 13.0});
  // Climber #4: no-op
  solver_settings.new_bounds.push_back({4, 0, variable_lower_bounds[0], variable_upper_bounds[0]});

  // Setup and solve batch
  optimization_problem_solution_t<int, double> solution3 =
    solve_lp(&handle_, op_problem, solver_settings);

  // All should be optimal
  for (size_t i = 0; i < batch_size; ++i)
    EXPECT_EQ((int)solution3.get_termination_status(i), CUOPT_TERMINATION_STATUS_OPTIMAL);

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
      test_constraint_sanity(op_problem_ref1, current_info, current_primal_solution, 1e-4, false);
    } else if (i == 3) {
      test_objective_sanity(
        op_problem_ref3, current_primal_solution, current_info.primal_objective);
      test_constraint_sanity(op_problem_ref3, current_info, current_primal_solution, 1e-4, false);
    } else {
      test_objective_sanity(op_problem, current_primal_solution, current_info.primal_objective);
      test_constraint_sanity(op_problem, current_info, current_primal_solution, 1e-4, false);
    }
  }
}

TEST(pdlp_class, simple_batch_different_objectives)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

  const int n_vars         = op_problem.get_n_variables();
  const auto& original_obj = op_problem.get_objective_coefficients();

  // Create a modified objective: scale by 2.0
  std::vector<double> modified_obj(original_obj.begin(), original_obj.end());
  for (auto& c : modified_obj)
    c *= 2.0;

  // Solve reference LPs individually
  // Ref 1: original objective
  auto ref_sol1         = solve_lp(&handle_, op_problem, solver_settings);
  const double ref_obj1 = ref_sol1.get_additional_termination_information(0).primal_objective;
  EXPECT_EQ((int)ref_sol1.get_termination_status(0), CUOPT_TERMINATION_STATUS_OPTIMAL);

  // Ref 2: modified objective
  auto op_problem_mod                         = op_problem;
  op_problem_mod.get_objective_coefficients() = modified_obj;
  auto ref_sol2                               = solve_lp(&handle_, op_problem_mod, solver_settings);
  const double ref_obj2 = ref_sol2.get_additional_termination_information(0).primal_objective;
  EXPECT_EQ((int)ref_sol2.get_termination_status(0), CUOPT_TERMINATION_STATUS_OPTIMAL);

  // Batch solve: fixed path with per-climber objective coefficients in COL-major layout
  // [climber0_all_vars, climber1_all_vars].
  std::vector<double> per_climber_objectives;
  per_climber_objectives.insert(
    per_climber_objectives.end(), original_obj.begin(), original_obj.end());
  per_climber_objectives.insert(
    per_climber_objectives.end(), modified_obj.begin(), modified_obj.end());

  auto batch_sol = solve_lp_batch_fixed(&handle_,
                                        op_problem,
                                        solver_settings,
                                        /*batch_size=*/2,
                                        per_climber_objectives);

  EXPECT_EQ((int)batch_sol.get_termination_status(0), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_EQ((int)batch_sol.get_termination_status(1), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(is_incorrect_objective(
    ref_obj1, batch_sol.get_additional_termination_information(0).primal_objective));
  EXPECT_FALSE(is_incorrect_objective(
    ref_obj2, batch_sol.get_additional_termination_information(1).primal_objective));

  // Extract per-climber solutions and validate
  const auto primal0 = extract_subvector(batch_sol.get_primal_solution(), 0, n_vars);
  test_objective_sanity(
    op_problem, primal0, batch_sol.get_additional_termination_information(0).primal_objective);
  test_constraint_sanity(
    op_problem, batch_sol.get_additional_termination_information(0), primal0, 1e-4, false);

  const auto primal1 = extract_subvector(batch_sol.get_primal_solution(), n_vars, n_vars);
  test_objective_sanity(
    op_problem_mod, primal1, batch_sol.get_additional_termination_information(1).primal_objective);
  test_constraint_sanity(
    op_problem_mod, batch_sol.get_additional_termination_information(1), primal1, 1e-4, false);
}

TEST(pdlp_class, simple_batch_different_offsets)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

  // Solve sequentially with different offsets
  const std::vector<double> offsets = {0.0, 10.0, -5.5};
  std::vector<double> ref_objectives;
  for (auto off : offsets) {
    auto op = op_problem;
    op.set_objective_offset(off);
    auto sol = solve_lp(&handle_, op, solver_settings);
    ASSERT_EQ((int)sol.get_termination_status(0), CUOPT_TERMINATION_STATUS_OPTIMAL);
    ref_objectives.push_back(sol.get_additional_termination_information(0).primal_objective);
  }

  // Solve as batch via fixed path with per-climber objective offsets.
  auto batch_sol = solve_lp_batch_fixed(&handle_,
                                        op_problem,
                                        solver_settings,
                                        /*batch_size=*/static_cast<int>(offsets.size()),
                                        /*per_climber_objective_coefficients=*/{},
                                        /*per_climber_constraint_lower_bounds=*/{},
                                        /*per_climber_constraint_upper_bounds=*/{},
                                        /*per_climber_objective_offsets=*/offsets);

  for (size_t i = 0; i < offsets.size(); ++i) {
    EXPECT_EQ((int)batch_sol.get_termination_status(i), CUOPT_TERMINATION_STATUS_OPTIMAL);
    EXPECT_FALSE(is_incorrect_objective(
      ref_objectives[i], batch_sol.get_additional_termination_information(i).primal_objective));
  }
}

TEST(pdlp_class, simple_batch_different_objectives_and_offsets)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

  const int n_vars         = op_problem.get_n_variables();
  const auto& original_obj = op_problem.get_objective_coefficients();

  // Two climbers: (original_obj, offset=3.5) and (2x objective, offset=-7.0)
  std::vector<double> obj_c1(original_obj.begin(), original_obj.end());
  std::vector<double> obj_c2(original_obj.begin(), original_obj.end());
  for (auto& c : obj_c2)
    c *= 2.0;
  const std::vector<double> offsets = {3.5, -7.0};

  // Solve sequentially as references
  auto ref_op1 = op_problem;
  ref_op1.set_objective_offset(offsets[0]);
  auto ref_sol1 = solve_lp(&handle_, ref_op1, solver_settings);
  ASSERT_EQ((int)ref_sol1.get_termination_status(0), CUOPT_TERMINATION_STATUS_OPTIMAL);
  const double ref_obj1 = ref_sol1.get_additional_termination_information(0).primal_objective;

  auto ref_op2                         = op_problem;
  ref_op2.get_objective_coefficients() = obj_c2;
  ref_op2.set_objective_offset(offsets[1]);
  auto ref_sol2 = solve_lp(&handle_, ref_op2, solver_settings);
  ASSERT_EQ((int)ref_sol2.get_termination_status(0), CUOPT_TERMINATION_STATUS_OPTIMAL);
  const double ref_obj2 = ref_sol2.get_additional_termination_information(0).primal_objective;

  // Batch solve via fixed path with both per-climber objectives and offsets.
  std::vector<double> per_climber_objectives;
  per_climber_objectives.insert(per_climber_objectives.end(), obj_c1.begin(), obj_c1.end());
  per_climber_objectives.insert(per_climber_objectives.end(), obj_c2.begin(), obj_c2.end());

  auto batch_sol = solve_lp_batch_fixed(&handle_,
                                        op_problem,
                                        solver_settings,
                                        /*batch_size=*/2,
                                        per_climber_objectives,
                                        /*per_climber_constraint_lower_bounds=*/{},
                                        /*per_climber_constraint_upper_bounds=*/{},
                                        offsets);

  EXPECT_EQ((int)batch_sol.get_termination_status(0), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_EQ((int)batch_sol.get_termination_status(1), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(is_incorrect_objective(
    ref_obj1, batch_sol.get_additional_termination_information(0).primal_objective));
  EXPECT_FALSE(is_incorrect_objective(
    ref_obj2, batch_sol.get_additional_termination_information(1).primal_objective));
}

TEST(pdlp_class, simple_batch_different_constraint_bounds)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

  const int n_constrs               = op_problem.get_n_constraints();
  const auto& original_lower_bounds = op_problem.get_constraint_lower_bounds();
  const auto& original_upper_bounds = op_problem.get_constraint_upper_bounds();

  // Build 3 climbers with perturbed bounds:
  //  - climber 0: unchanged (scale factor 1.0)
  //  - climber 1: tighten upper bounds by 5% where finite (scale 0.95 on finite upper)
  //  - climber 2: loosen upper bounds by 5% where finite (scale 1.05 on finite upper)
  const std::vector<double> upper_scales = {1.0, 0.95, 1.05};
  const size_t batch_size                = upper_scales.size();

  std::vector<double> all_new_lower;
  std::vector<double> all_new_upper;
  std::vector<std::vector<double>> per_climber_lower(batch_size);
  std::vector<std::vector<double>> per_climber_upper(batch_size);
  for (size_t c = 0; c < batch_size; ++c) {
    per_climber_lower[c] =
      std::vector<double>(original_lower_bounds.begin(), original_lower_bounds.end());
    per_climber_upper[c] =
      std::vector<double>(original_upper_bounds.begin(), original_upper_bounds.end());
    for (auto& v : per_climber_upper[c]) {
      if (std::isfinite(v)) v *= upper_scales[c];
    }
    all_new_lower.insert(
      all_new_lower.end(), per_climber_lower[c].begin(), per_climber_lower[c].end());
    all_new_upper.insert(
      all_new_upper.end(), per_climber_upper[c].begin(), per_climber_upper[c].end());
  }

  // Solve sequentially to get reference objectives
  std::vector<double> ref_objectives;
  for (size_t c = 0; c < batch_size; ++c) {
    auto op                          = op_problem;
    op.get_constraint_lower_bounds() = per_climber_lower[c];
    op.get_constraint_upper_bounds() = per_climber_upper[c];
    auto sol                         = solve_lp(&handle_, op, solver_settings);
    ASSERT_EQ((int)sol.get_termination_status(0), CUOPT_TERMINATION_STATUS_OPTIMAL);
    ref_objectives.push_back(sol.get_additional_termination_information(0).primal_objective);
  }

  // Solve as a batch via fixed path with per-climber constraint bounds.
  auto batch_sol = solve_lp_batch_fixed(&handle_,
                                        op_problem,
                                        solver_settings,
                                        /*batch_size=*/static_cast<int>(batch_size),
                                        /*per_climber_objective_coefficients=*/{},
                                        all_new_lower,
                                        all_new_upper);

  for (size_t i = 0; i < batch_size; ++i) {
    EXPECT_EQ((int)batch_sol.get_termination_status(i), CUOPT_TERMINATION_STATUS_OPTIMAL);
    EXPECT_FALSE(is_incorrect_objective(
      ref_objectives[i], batch_sol.get_additional_termination_information(i).primal_objective));
  }
}

TEST(pdlp_class, simple_batch_everything_different)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

  const int n_vars    = op_problem.get_n_variables();
  const int n_constrs = op_problem.get_n_constraints();

  const auto& original_obj          = op_problem.get_objective_coefficients();
  const auto& original_lower_bounds = op_problem.get_constraint_lower_bounds();
  const auto& original_upper_bounds = op_problem.get_constraint_upper_bounds();

  // Describe 2 climbers where EVERY per-climber field differs
  struct climber_spec {
    std::tuple<int, double, double> new_bound;  // (variable_idx, lower, upper)
    double obj_scale;                           // multiply objective coefficients
    double offset;                              // objective offset
    double constr_upper_scale;                  // multiply finite constraint upper bounds
  };
  const std::vector<climber_spec> specs = {
    // Climber 0: var 5 bounds [4.0,5.0], 1.5x obj, offset +7.5, constraint upper *1.02
    {{5, 4.0, 5.0}, 1.5, 7.5, 1.02},
    // Climber 1: var 1 bounds [-7.0,13.0], 2x obj, offset -3.25, constraint upper *0.95
    {{1, -7.0, 13.0}, 2.0, -3.25, 0.95},
  };
  const size_t batch_size = specs.size();

  // Build the per-climber objective/offset/constraint-bound vectors.
  std::vector<double> all_new_objectives;
  std::vector<double> all_new_objective_offsets;
  std::vector<double> all_new_constraint_lower;
  std::vector<double> all_new_constraint_upper;

  std::vector<std::vector<double>> per_climber_obj(batch_size);
  std::vector<std::vector<double>> per_climber_upper(batch_size);
  std::vector<std::vector<double>> per_climber_lower(batch_size);

  for (size_t c = 0; c < batch_size; ++c) {
    per_climber_obj[c] = std::vector<double>(original_obj.begin(), original_obj.end());
    for (auto& v : per_climber_obj[c])
      v *= specs[c].obj_scale;
    per_climber_lower[c] =
      std::vector<double>(original_lower_bounds.begin(), original_lower_bounds.end());
    per_climber_upper[c] =
      std::vector<double>(original_upper_bounds.begin(), original_upper_bounds.end());
    for (auto& v : per_climber_upper[c]) {
      if (std::isfinite(v)) v *= specs[c].constr_upper_scale;
    }
    all_new_objectives.insert(
      all_new_objectives.end(), per_climber_obj[c].begin(), per_climber_obj[c].end());
    all_new_objective_offsets.push_back(specs[c].offset);
    all_new_constraint_lower.insert(
      all_new_constraint_lower.end(), per_climber_lower[c].begin(), per_climber_lower[c].end());
    all_new_constraint_upper.insert(
      all_new_constraint_upper.end(), per_climber_upper[c].begin(), per_climber_upper[c].end());
  }

  // Sequential reference: solve each climber independently and capture its objective.
  std::vector<double> ref_objectives(batch_size);
  std::vector<cuopt::linear_programming::io::mps_data_model_t<int, double>> ref_problems;
  ref_problems.reserve(batch_size);
  for (size_t c = 0; c < batch_size; ++c) {
    auto ref_op                          = op_problem;
    ref_op.get_objective_coefficients()  = per_climber_obj[c];
    ref_op.get_constraint_lower_bounds() = per_climber_lower[c];
    ref_op.get_constraint_upper_bounds() = per_climber_upper[c];
    ref_op.get_variable_lower_bounds()[std::get<0>(specs[c].new_bound)] =
      std::get<1>(specs[c].new_bound);
    ref_op.get_variable_upper_bounds()[std::get<0>(specs[c].new_bound)] =
      std::get<2>(specs[c].new_bound);
    ref_op.set_objective_offset(specs[c].offset);
    ref_problems.push_back(ref_op);

    auto sol = solve_lp(&handle_, ref_problems.back(), solver_settings);
    ASSERT_EQ((int)sol.get_termination_status(0), CUOPT_TERMINATION_STATUS_OPTIMAL);
    ref_objectives[c] = sol.get_additional_termination_information(0).primal_objective;
  }

  // Now solve as a single batch via fixed path, combining new_bounds (per-climber variable-bound
  // overrides) with all the other per-climber problem fields expanded directly on the
  // optimization_problem_t.
  for (size_t c = 0; c < batch_size; ++c) {
    solver_settings.new_bounds.push_back({static_cast<int>(c),
                                          std::get<0>(specs[c].new_bound),
                                          std::get<1>(specs[c].new_bound),
                                          std::get<2>(specs[c].new_bound)});
  }

  auto batch_sol = solve_lp_batch_fixed(&handle_,
                                        op_problem,
                                        solver_settings,
                                        /*batch_size=*/static_cast<int>(batch_size),
                                        all_new_objectives,
                                        all_new_constraint_lower,
                                        all_new_constraint_upper,
                                        all_new_objective_offsets);

  for (size_t c = 0; c < batch_size; ++c) {
    EXPECT_EQ((int)batch_sol.get_termination_status(c), CUOPT_TERMINATION_STATUS_OPTIMAL);
    EXPECT_FALSE(is_incorrect_objective(
      ref_objectives[c], batch_sol.get_additional_termination_information(c).primal_objective));

    // Validate the per-climber primal solution matches the corresponding reference problem.
    // The solver's reported objective includes the offset; test_objective_sanity only computes
    // c^T * x, so subtract the offset to make the values comparable.
    const auto primal = extract_subvector(batch_sol.get_primal_solution(), c * n_vars, n_vars);
    const double reported_obj =
      batch_sol.get_additional_termination_information(c).primal_objective;
    test_objective_sanity(ref_problems[c], primal, reported_obj - specs[c].offset);
    test_constraint_sanity(
      ref_problems[c], batch_sol.get_additional_termination_information(c), primal, 1e-4, false);
  }
}

TEST(pdlp_class, run_batch_pdlp_fixed_rejects_partial_per_climber_expansion)
{
  const raft::handle_t handle_{};
  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  constexpr int batch_size = 3;
  const auto n_vars        = static_cast<size_t>(op_problem.get_n_variables());
  const auto n_cons        = static_cast<size_t>(op_problem.get_n_constraints());
  const auto stream        = handle_.get_stream();

  auto make_settings = []() {
    pdlp_solver_settings_t<int, double> s{};
    s.method                              = cuopt::linear_programming::method_t::PDLP;
    s.presolver                           = presolver_t::None;
    s.fixed_batch_size                    = batch_size;
    s.generate_batch_primal_dual_solution = true;
    return s;
  };

  auto expect_validation_error = [](auto&& fn) {
    try {
      fn();
      FAIL() << "expected cuopt::logic_error with ValidationError";
    } catch (const cuopt::logic_error& e) {
      EXPECT_EQ(e.get_error_type(), cuopt::error_type_t::ValidationError);
    }
  };

  // Case 1: objective_coefficients has an in-between size (batch_size * n_vars - 1).
  {
    auto gpu_op = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
      &handle_, op_problem);
    std::vector<double> bad_obj(batch_size * n_vars - 1, 0.0);
    assign_device_uvector_from_host(gpu_op.get_objective_coefficients(), bad_obj, stream);
    auto settings = make_settings();
    expect_validation_error([&]() { cuopt::linear_programming::run_batch_pdlp(gpu_op, settings); });
  }

  // Case 2: constraint_lower_bounds has an in-between size (batch_size * n_cons - 1).
  {
    auto gpu_op = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
      &handle_, op_problem);
    std::vector<double> bad_clb(batch_size * n_cons - 1, 0.0);
    assign_device_uvector_from_host(gpu_op.get_constraint_lower_bounds(), bad_clb, stream);
    auto settings = make_settings();
    expect_validation_error([&]() { cuopt::linear_programming::run_batch_pdlp(gpu_op, settings); });
  }

  // Case 3: constraint_upper_bounds has an in-between size (batch_size * n_cons - 1).
  {
    auto gpu_op = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
      &handle_, op_problem);
    std::vector<double> bad_cub(batch_size * n_cons - 1, 0.0);
    assign_device_uvector_from_host(gpu_op.get_constraint_upper_bounds(), bad_cub, stream);
    auto settings = make_settings();
    expect_validation_error([&]() { cuopt::linear_programming::run_batch_pdlp(gpu_op, settings); });
  }

  // Case 4: lower bounds expanded per-climber but upper bounds left shared (or vice versa).
  // pdhg.cu's swap path keys off the lower-bound size and assumes the upper follows.
  {
    auto gpu_op = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
      &handle_, op_problem);
    std::vector<double> per_climber_clb(batch_size * n_cons, 0.0);
    assign_device_uvector_from_host(gpu_op.get_constraint_lower_bounds(), per_climber_clb, stream);
    auto settings = make_settings();
    expect_validation_error([&]() { cuopt::linear_programming::run_batch_pdlp(gpu_op, settings); });
  }

  // Case 5: batch_objective_offsets has an unexpected size (not 0 and not fixed_batch_size).
  {
    auto gpu_op = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
      &handle_, op_problem);
    std::vector<double> bad_offsets(batch_size + 1, 0.0);
    gpu_op.set_batch_objective_offsets(bad_offsets);
    auto settings = make_settings();
    expect_validation_error([&]() { cuopt::linear_programming::run_batch_pdlp(gpu_op, settings); });
  }
}

TEST(pdlp_class, run_batch_pdlp_rejects_invalid_new_bounds)
{
  const raft::handle_t handle_{};
  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto expect_validation_error = [&](pdlp_solver_settings_t<int, double> settings) {
    auto gpu_op = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
      &handle_, op_problem);
    try {
      cuopt::linear_programming::run_batch_pdlp(gpu_op, settings);
      FAIL() << "expected cuopt::logic_error with ValidationError";
    } catch (const cuopt::logic_error& e) {
      EXPECT_EQ(e.get_error_type(), cuopt::error_type_t::ValidationError);
    }
  };

  auto make_settings = []() {
    pdlp_solver_settings_t<int, double> settings{};
    settings.method                              = cuopt::linear_programming::method_t::PDLP;
    settings.presolver                           = presolver_t::None;
    settings.generate_batch_primal_dual_solution = true;
    return settings;
  };

  {
    // Reversed bounds would make projection undefined for this climber.
    auto settings = make_settings();
    settings.new_bounds.push_back({0, 0, 2.0, 1.0});
    expect_validation_error(settings);
  }
  {
    // Variable indices must reference an existing variable.
    auto settings = make_settings();
    settings.new_bounds.push_back({0, static_cast<int>(op_problem.get_n_variables()), 0.0, 1.0});
    expect_validation_error(settings);
  }
  {
    // Negative variable indices cannot be mapped into the primal vector.
    auto settings = make_settings();
    settings.new_bounds.push_back({0, -1, 0.0, 1.0});
    expect_validation_error(settings);
  }
  {
    // A climber can only provide one override per variable.
    auto settings = make_settings();
    settings.new_bounds.push_back({0, 0, 0.0, 1.0});
    settings.new_bounds.push_back({0, 0, -1.0, 2.0});
    expect_validation_error(settings);
  }
  {
    // Climber entries must be sorted so sub-batching can split the flat list consistently.
    auto settings = make_settings();
    settings.new_bounds.push_back({1, 0, 0.0, 1.0});
    settings.new_bounds.push_back({0, 1, 0.0, 1.0});
    expect_validation_error(settings);
  }
  {
    // Reopening a climber after a later climber would make the flat layout non-contiguous.
    auto settings = make_settings();
    settings.new_bounds.push_back({0, 0, 0.0, 1.0});
    settings.new_bounds.push_back({1, 1, 0.0, 1.0});
    settings.new_bounds.push_back({0, 1, -1.0, 2.0});
    expect_validation_error(settings);
  }
  {
    // The run_batch_pdlp splitting path expects exactly one variable-bound override per climber.
    auto settings = make_settings();
    settings.new_bounds.push_back({0, 0, 0.0, 1.0});
    settings.new_bounds.push_back({0, 1, -1.0, 2.0});
    expect_validation_error(settings);
  }
  {
    // The run_batch_pdlp splitting path cannot skip climbers because it slices by batch slot.
    auto settings = make_settings();
    settings.new_bounds.push_back({0, 0, 0.0, 1.0});
    settings.new_bounds.push_back({2, 1, -1.0, 2.0});
    expect_validation_error(settings);
  }
  {
    // NaN bounds would poison the primal projection.
    auto settings = make_settings();
    settings.new_bounds.push_back({0, 0, std::numeric_limits<double>::quiet_NaN(), 1.0});
    expect_validation_error(settings);
  }
  {
    // Negative climber IDs cannot map to a batch slot.
    auto settings = make_settings();
    settings.new_bounds.push_back({-1, 0, 0.0, 1.0});
    expect_validation_error(settings);
  }
  {
    // Fixed-batch mode cannot reference climbers outside the declared batch.
    auto settings             = make_settings();
    settings.fixed_batch_size = 2;
    settings.new_bounds.push_back({2, 0, 0.0, 1.0});
    expect_validation_error(settings);
  }
  {
    // The solve_lp wrapper should reject invalid bounds before running PDLP as well.
    auto settings = make_settings();
    settings.new_bounds.push_back({0, 0, 2.0, 1.0});
    auto solution = solve_lp(&handle_, op_problem, settings);
    EXPECT_EQ(solution.get_error_status().get_error_type(), cuopt::error_type_t::ValidationError);
  }
}

TEST(pdlp_class, run_batch_pdlp_rejects_save_best_primal_so_far)
{
  const raft::handle_t handle_{};
  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  // Splitting path: trigger batch mode via a non-empty new_bounds list (size > 1).
  {
    auto gpu_op = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
      &handle_, op_problem);

    pdlp_solver_settings_t<int, double> settings{};
    settings.method                              = cuopt::linear_programming::method_t::PDLP;
    settings.presolver                           = presolver_t::None;
    settings.generate_batch_primal_dual_solution = true;
    settings.save_best_primal_so_far             = true;
    const int var_id                             = 0;
    settings.new_bounds.push_back({0,
                                   var_id,
                                   op_problem.get_variable_lower_bounds()[var_id] + 1.0,
                                   op_problem.get_variable_upper_bounds()[var_id]});
    settings.new_bounds.push_back({1,
                                   var_id,
                                   op_problem.get_variable_lower_bounds()[var_id] + 2.0,
                                   op_problem.get_variable_upper_bounds()[var_id]});

    auto sol = cuopt::linear_programming::run_batch_pdlp(gpu_op, settings);
    EXPECT_EQ(sol.get_error_status().get_error_type(), cuopt::error_type_t::ValidationError);
  }

  // Fixed-batch path: trigger batch mode via fixed_batch_size with shared (size == n) buffers.
  {
    auto gpu_op = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
      &handle_, op_problem);

    pdlp_solver_settings_t<int, double> settings{};
    settings.method                              = cuopt::linear_programming::method_t::PDLP;
    settings.presolver                           = presolver_t::None;
    settings.fixed_batch_size                    = 2;
    settings.generate_batch_primal_dual_solution = true;
    settings.save_best_primal_so_far             = true;

    auto sol = cuopt::linear_programming::run_batch_pdlp(gpu_op, settings);
    EXPECT_EQ(sol.get_error_status().get_error_type(), cuopt::error_type_t::ValidationError);
  }
}

TEST(pdlp_class, DISABLED_cupdlpx_infeasible_detection_afiro_new_bounds)
{
  const raft::handle_t handle_{};

  auto solver_settings                 = pdlp_solver_settings_t<int, double>{};
  solver_settings.method               = cuopt::linear_programming::method_t::PDLP;
  solver_settings.detect_infeasibility = true;

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

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
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  const std::vector<double>& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const std::vector<double>& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  for (size_t i = 0; i < batch_size; i++) {
    solver_settings.new_bounds.push_back(
      {static_cast<int>(i), 0, variable_lower_bounds[0], variable_upper_bounds[0]});
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
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

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
    solver_settings.new_bounds.push_back({static_cast<int>(i), 1, 7.0, 8.0});
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
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

  // Manually changing the bounds and doing it through the solver settings should give the same
  // result

  solver_settings.new_bounds.push_back({0, 0, 45.0, 55.0});

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
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

  constexpr int batch_size = 1000;

  // Setup a larger batch afiro but with all same primal/dual bounds

  const std::vector<double>& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const std::vector<double>& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  for (size_t i = 0; i < batch_size; i++) {
    solver_settings.new_bounds.push_back(
      {static_cast<int>(i), 0, variable_lower_bounds[0], variable_upper_bounds[0]});
  }

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);

  // All should be optimal with
  for (size_t i = 0; i < batch_size; ++i) {
    EXPECT_EQ((int)solution.get_termination_status(i), CUOPT_TERMINATION_STATUS_OPTIMAL);
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
                         1e-4,
                         false);
}

// Disabled until we have a reliable way to detect infeasibility
TEST(pdlp_class, DISABLED_simple_batch_optimal_and_infeasible)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings                 = pdlp_solver_settings_t<int, double>{};
  solver_settings.method               = cuopt::linear_programming::method_t::PDLP;
  solver_settings.detect_infeasibility = true;
  solver_settings.presolver            = presolver_t::None;

  const std::vector<double>& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const std::vector<double>& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  // Make the first problem infeasible while the second remains solvable
  solver_settings.new_bounds.push_back({0, 1, 7.0, 8.0});
  // No change for the second
  solver_settings.new_bounds.push_back({1, 0, variable_lower_bounds[0], variable_upper_bounds[0]});

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);

  // First should be primal infeasible and the second optimal with the correct
  EXPECT_EQ((int)solution.get_termination_status(0), CUOPT_TERMINATION_STATUS_INFEASIBLE);
  EXPECT_EQ((int)solution.get_termination_status(1), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution.get_additional_termination_information(1).primal_objective));
}

// Disabled until we have a reliable way to detect infeasibility
TEST(pdlp_class, DISABLED_larger_batch_optimal_and_infeasible)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings                 = pdlp_solver_settings_t<int, double>{};
  solver_settings.method               = cuopt::linear_programming::method_t::PDLP;
  solver_settings.detect_infeasibility = true;

  const std::vector<double>& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const std::vector<double>& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  // #0: no-op
  solver_settings.new_bounds.push_back({0, 0, variable_lower_bounds[0], variable_upper_bounds[0]});
  // #1: var 1 -> [7.0, 8.0] (infeasible)
  solver_settings.new_bounds.push_back({1, 1, 7.0, 8.0});
  // #2: no-op
  solver_settings.new_bounds.push_back({2, 0, variable_lower_bounds[0], variable_upper_bounds[0]});
  // #3: var 1 -> [-11.0, -10.0] (infeasible)
  solver_settings.new_bounds.push_back({3, 1, -11.0, -10.0});
  // #4: no-op
  solver_settings.new_bounds.push_back({4, 0, variable_lower_bounds[0], variable_upper_bounds[0]});

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);

  // #1 and #3 should be infeasible
  EXPECT_EQ((int)solution.get_termination_status(1), CUOPT_TERMINATION_STATUS_INFEASIBLE);
  EXPECT_EQ((int)solution.get_termination_status(3), CUOPT_TERMINATION_STATUS_INFEASIBLE);

  // Rest should be feasible with the correct primal objective
  EXPECT_EQ((int)solution.get_termination_status(0), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_EQ((int)solution.get_termination_status(2), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_EQ((int)solution.get_termination_status(4), CUOPT_TERMINATION_STATUS_OPTIMAL);

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
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  const std::vector<int> fractional     = {1, 2, 4};
  const std::vector<double> root_soln_x = {0.891, 0.109, 0.636429};

  auto solver_settings             = pdlp_solver_settings_t<int, double>{};
  solver_settings.method           = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode = pdlp_solver_mode_t::Stable3;
  solver_settings.presolver        = cuopt::linear_programming::presolver_t::None;
  solver_settings.generate_batch_primal_dual_solution = true;

  const int n_fractional = fractional.size();
  const int batch_size   = n_fractional * 2;

  std::vector<double> ref_objectives(batch_size);
  std::vector<pdlp_termination_status_t> ref_statuses(batch_size);
  std::vector<cuopt::linear_programming::io::mps_data_model_t<int, double>> ref_problems;

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
    test_constraint_sanity(ref_problems[i], current_info, current_primal_solution, 1e-4, false);
  }

  // Now run again using the new_bounds API
  for (int i = 0; i < n_fractional; ++i) {
    solver_settings.new_bounds.push_back({i,
                                          fractional[i],
                                          op_problem.get_variable_lower_bounds()[fractional[i]],
                                          std::floor(root_soln_x[i])});
  }
  for (int i = 0; i < n_fractional; ++i) {
    solver_settings.new_bounds.push_back({i + n_fractional,
                                          fractional[i],
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
                           1e-4,
                           false);
  }
}

TEST(pdlp_class, strong_branching_user_api)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  const std::vector<int> fractional     = {1, 2, 4};
  const std::vector<double> root_soln_x = {0.891, 0.109, 0.636429};

  auto solver_settings             = pdlp_solver_settings_t<int, double>{};
  solver_settings.method           = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode = pdlp_solver_mode_t::Stable3;
  solver_settings.presolver        = cuopt::linear_programming::presolver_t::None;
  solver_settings.generate_batch_primal_dual_solution = true;

  const int n_fractional = fractional.size();
  const int batch_size   = n_fractional * 2;

  std::vector<double> ref_objectives(batch_size);
  std::vector<pdlp_termination_status_t> ref_statuses(batch_size);
  std::vector<cuopt::linear_programming::io::mps_data_model_t<int, double>> ref_problems;

  // Down branches first, then Up branches.

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

  // Build per-climber variable bounds: down branches first, then up branches.
  for (int i = 0; i < n_fractional; ++i) {
    solver_settings.new_bounds.push_back({i,
                                          fractional[i],
                                          op_problem.get_variable_lower_bounds()[fractional[i]],
                                          std::floor(root_soln_x[i])});
  }
  for (int i = 0; i < n_fractional; ++i) {
    solver_settings.new_bounds.push_back({i + n_fractional,
                                          fractional[i],
                                          std::ceil(root_soln_x[i]),
                                          op_problem.get_variable_upper_bounds()[fractional[i]]});
  }

  // Solve batch via the run_batch_pdlp strong-branching path (auto batch sizing).
  auto gpu_op = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
    &handle_, op_problem);
  auto batch_sol = cuopt::linear_programming::run_batch_pdlp(gpu_op, solver_settings);

  EXPECT_EQ((int)batch_sol.get_terminations_status().size(), batch_size);
  const size_t primal_size = op_problem.get_n_variables();

  for (int i = 0; i < batch_size; ++i) {
    EXPECT_EQ(batch_sol.get_termination_status(i), ref_statuses[i]);
    // Climber in the batch that have gained optimality can lose optimality while other are still
    // optimizing This can lead to differences in the objective values, so we allow for a small
    // tolerance
    EXPECT_NEAR(batch_sol.get_additional_termination_information(i).primal_objective,
                ref_objectives[i],
                1e-4);

    const auto current_primal_solution =
      extract_subvector(batch_sol.get_primal_solution(), i * primal_size, primal_size);
    const auto& current_info = batch_sol.get_additional_termination_information(i);

    test_objective_sanity(ref_problems[i], current_primal_solution, current_info.primal_objective);
    test_constraint_sanity(ref_problems[i], current_info, current_primal_solution, 1e-4, false);
  }
}

TEST(pdlp_class, strong_branching_multi_bounds_per_climber)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings             = pdlp_solver_settings_t<int, double>{};
  solver_settings.method           = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode = pdlp_solver_mode_t::Stable3;
  solver_settings.presolver        = cuopt::linear_programming::presolver_t::None;
  solver_settings.generate_batch_primal_dual_solution = true;

  auto root_solution = solve_lp(&handle_, op_problem, solver_settings);
  ASSERT_EQ(root_solution.get_termination_status(0), pdlp_termination_status_t::Optimal);
  const auto root_primal =
    host_copy(root_solution.get_primal_solution(), root_solution.get_primal_solution().stream());

  const auto& original_lower = op_problem.get_variable_lower_bounds();
  const auto& original_upper = op_problem.get_variable_upper_bounds();
  auto tightened_bounds      = [&](int var_idx) {
    const double lower = std::max(original_lower[var_idx], std::floor(root_primal[var_idx]));
    const double upper = std::min(original_upper[var_idx], std::ceil(root_primal[var_idx]));
    return std::make_pair(lower, upper);
  };

  const std::vector<std::vector<int>> vars_by_climber = {
    {1, 2},
    {1, 4},
    {2, 4, 1},
    {4, 5},
  };
  const int batch_size = vars_by_climber.size();

  std::vector<std::tuple<int, int, double, double>> bound_specs;
  std::vector<double> ref_objectives(batch_size);
  std::vector<pdlp_termination_status_t> ref_statuses(batch_size);
  std::vector<cuopt::linear_programming::io::mps_data_model_t<int, double>> ref_problems;
  ref_problems.reserve(batch_size);

  for (int c = 0; c < batch_size; ++c) {
    auto ref_problem = op_problem;
    for (const auto var_idx : vars_by_climber[c]) {
      const auto [lower, upper]                        = tightened_bounds(var_idx);
      ref_problem.get_variable_lower_bounds()[var_idx] = lower;
      ref_problem.get_variable_upper_bounds()[var_idx] = upper;
      bound_specs.push_back({c, var_idx, lower, upper});
      solver_settings.new_bounds.push_back({c, var_idx, lower, upper});
    }
    ref_problems.push_back(ref_problem);

    auto ref_settings = solver_settings;
    ref_settings.new_bounds.clear();
    auto ref_solution = solve_lp(&handle_, ref_problems.back(), ref_settings);
    ref_statuses[c]   = ref_solution.get_termination_status(0);
    ASSERT_EQ(ref_statuses[c], pdlp_termination_status_t::Optimal);
    ref_objectives[c] = ref_solution.get_additional_termination_information(0).primal_objective;
  }

  auto batch_solution = solve_lp(&handle_, op_problem, solver_settings);

  ASSERT_EQ((int)batch_solution.get_terminations_status().size(), batch_size);
  const size_t primal_size = op_problem.get_n_variables();
  for (int c = 0; c < batch_size; ++c) {
    EXPECT_EQ(batch_solution.get_termination_status(c), ref_statuses[c]);
    EXPECT_NEAR(batch_solution.get_additional_termination_information(c).primal_objective,
                ref_objectives[c],
                1e-4);

    const auto current_primal_solution =
      extract_subvector(batch_solution.get_primal_solution(), c * primal_size, primal_size);
    const auto& current_info = batch_solution.get_additional_termination_information(c);
    test_objective_sanity(ref_problems[c], current_primal_solution, current_info.primal_objective);
    test_constraint_sanity(ref_problems[c], current_info, current_primal_solution, 1e-4, false);
  }
}

TEST(pdlp_class, run_batch_pdlp_many_different_bounds)
{
  constexpr double result_tolerance = 1e-8;

  const raft::handle_t handle_{};
  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  const auto& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const auto& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  auto regular_pdlp_settings             = pdlp_solver_settings_t<int, double>{};
  regular_pdlp_settings.method           = cuopt::linear_programming::method_t::PDLP;
  regular_pdlp_settings.pdlp_solver_mode = pdlp_solver_mode_t::Stable3;
  regular_pdlp_settings.presolver        = presolver_t::None;
  regular_pdlp_settings.set_optimality_tolerance(result_tolerance);

  const std::vector<std::vector<std::tuple<int, double, double>>> bound_offsets_by_climber = {
    {{1, 3.0, 7.0}},
    {{2, 5.0, 13.0}, {5, 17.0, 29.0}},
    {{3, 7.0, 17.0}, {6, 19.0, 31.0}, {10, 37.0, 47.0}},
    {{4, 11.0, 23.0}, {8, 29.0, 41.0}, {11, 43.0, 59.0}, {20, 67.0, 71.0}},
    {{1, 13.0, 29.0}, {13, 31.0, 53.0}},
    {{2, 17.0, 31.0}, {14, 37.0, 61.0}, {19, 53.0, 71.0}, {25, 83.0, 89.0}, {30, 97.0, 101.0}},
    {{5, 19.0, 37.0}, {16, 41.0, 67.0}, {21, 59.0, 83.0}},
    {{6, 23.0, 43.0},
     {18, 47.0, 71.0},
     {22, 67.0, 97.0},
     {29, 103.0, 107.0},
     {31, 109.0, 113.0},
     {7, 127.0, 131.0}},
    {{7, 29.0, 47.0}, {20, 53.0, 79.0}},
    {{8, 31.0, 53.0}, {12, 59.0, 83.0}, {26, 79.0, 103.0}, {31, 127.0, 131.0}, {4, 137.0, 139.0}},
    {{3, 37.0, 59.0},
     {11, 67.0, 89.0},
     {17, 83.0, 107.0},
     {28, 137.0, 139.0},
     {9, 149.0, 151.0},
     {15, 157.0, 163.0},
     {24, 167.0, 173.0}},
    {{4, 41.0, 61.0}, {10, 71.0, 97.0}, {15, 89.0, 109.0}},
  };
  const int batch_size = bound_offsets_by_climber.size();
  std::vector<std::vector<std::tuple<int, double, double>>> custom_bounds_by_climber(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    for (const auto& [var_idx, lower_offset, upper_offset] : bound_offsets_by_climber[i]) {
      const double lower = std::isfinite(variable_lower_bounds[var_idx])
                             ? variable_lower_bounds[var_idx] - lower_offset
                             : variable_lower_bounds[var_idx];
      const double upper = std::isfinite(variable_upper_bounds[var_idx])
                             ? variable_upper_bounds[var_idx] + upper_offset
                             : variable_upper_bounds[var_idx];
      custom_bounds_by_climber[i].push_back({var_idx, lower, upper});
    }
  }

  std::vector<double> ref_objectives(batch_size);
  std::vector<pdlp_termination_status_t> ref_statuses(batch_size);
  std::vector<cuopt::linear_programming::io::mps_data_model_t<int, double>> ref_problems;
  std::vector<std::tuple<int, int, double, double>> bound_specs;

  for (int i = 0; i < batch_size; ++i) {
    auto ref_problem = op_problem;
    for (const auto& bounds : custom_bounds_by_climber[i]) {
      ref_problem.get_variable_lower_bounds()[std::get<0>(bounds)] = std::get<1>(bounds);
      ref_problem.get_variable_upper_bounds()[std::get<0>(bounds)] = std::get<2>(bounds);
      bound_specs.push_back({i, std::get<0>(bounds), std::get<1>(bounds), std::get<2>(bounds)});
    }
    ref_problems.push_back(ref_problem);

    auto ref_solution = solve_lp(&handle_, ref_problems.back(), regular_pdlp_settings);
    ref_statuses[i]   = ref_solution.get_termination_status(0);
    ASSERT_EQ(ref_statuses[i], pdlp_termination_status_t::Optimal);
    ref_objectives[i] = ref_solution.get_additional_termination_information(0).primal_objective;
  }

  auto batch_settings                                = regular_pdlp_settings;
  batch_settings.generate_batch_primal_dual_solution = true;
  for (int i = 0; i < batch_size; ++i) {
    for (const auto& bounds : custom_bounds_by_climber[i]) {
      batch_settings.new_bounds.push_back(
        {i, std::get<0>(bounds), std::get<1>(bounds), std::get<2>(bounds)});
    }
  }

  batch_settings.set_optimality_tolerance(result_tolerance);
  optimization_problem_solution_t<int, double> batch_solution =
    solve_lp(&handle_, op_problem, batch_settings);

  ASSERT_EQ(batch_solution.get_terminations_status().size(), batch_size);
  const size_t primal_size = op_problem.get_n_variables();
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_EQ(batch_solution.get_termination_status(i), ref_statuses[i]);
    EXPECT_NEAR(batch_solution.get_additional_termination_information(i).primal_objective,
                ref_objectives[i],
                result_tolerance);

    const auto current_primal_solution =
      extract_subvector(batch_solution.get_primal_solution(), i * primal_size, primal_size);
    test_objective_sanity(
      ref_problems[i],
      current_primal_solution,
      batch_solution.get_additional_termination_information(i).primal_objective);
    test_constraint_sanity(ref_problems[i],
                           batch_solution.get_additional_termination_information(i),
                           current_primal_solution,
                           result_tolerance,
                           false);
  }
}

TEST(pdlp_class, run_batch_pdlp_many_different_bounds_good_mps_some_var_bounds)
{
  constexpr double lower_bounds    = -33.0;
  constexpr double upper_bounds    = 10.0;
  constexpr double exact_tolerance = 1e-8;

  const raft::handle_t handle_{};
  auto path = make_path_absolute("linear_programming/good-mps-some-var-bounds.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  const auto& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const auto& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  const std::vector<std::vector<std::tuple<int, double, double>>> custom_bounds_by_climber = {
    {{0, lower_bounds - 100.0, upper_bounds}},
    {{1, variable_lower_bounds[1] - 3.0, variable_upper_bounds[1] + 5.0}},
    {{0, lower_bounds - 150.0, upper_bounds + 1.0},
     {1, variable_lower_bounds[1] - 7.0, variable_upper_bounds[1] + 11.0}},
    {{0, lower_bounds - 200.0, upper_bounds + 2.0}},
    {{1, variable_lower_bounds[1] - 13.0, variable_upper_bounds[1] + 17.0}},
    {{0, lower_bounds - 500.0, upper_bounds + 3.0},
     {1, variable_lower_bounds[1] - 19.0, variable_upper_bounds[1] + 23.0}},
    {{0, lower_bounds - 750.0, upper_bounds + 5.0}},
    {{1, variable_lower_bounds[1] - 29.0, variable_upper_bounds[1] + 31.0}},
    {{0, lower_bounds - 1000.0, upper_bounds + 7.0},
     {1, variable_lower_bounds[1] - 37.0, variable_upper_bounds[1] + 41.0}},
    {{0, lower_bounds - 1250.0, upper_bounds + 11.0}},
    {{1, variable_lower_bounds[1] - 43.0, variable_upper_bounds[1] + 47.0}},
    {{0, lower_bounds - 2500.0, upper_bounds + 13.0},
     {1, variable_lower_bounds[1] - 53.0, variable_upper_bounds[1] + 59.0}},
  };
  const int batch_size = custom_bounds_by_climber.size();

  auto regular_pdlp_settings      = pdlp_solver_settings_t<int, double>{};
  regular_pdlp_settings.method    = cuopt::linear_programming::method_t::PDLP;
  regular_pdlp_settings.presolver = presolver_t::None;
  regular_pdlp_settings.set_optimality_tolerance(exact_tolerance);

  std::vector<double> ref_objectives(batch_size);
  std::vector<pdlp_termination_status_t> ref_statuses(batch_size);
  std::vector<cuopt::linear_programming::io::mps_data_model_t<int, double>> ref_problems;
  std::vector<std::vector<double>> ref_primal_solutions(batch_size);

  for (int i = 0; i < batch_size; ++i) {
    auto ref_problem = op_problem;
    for (const auto& bounds : custom_bounds_by_climber[i]) {
      ref_problem.get_variable_lower_bounds()[std::get<0>(bounds)] = std::get<1>(bounds);
      ref_problem.get_variable_upper_bounds()[std::get<0>(bounds)] = std::get<2>(bounds);
    }
    ref_problems.push_back(ref_problem);

    auto ref_solution = solve_lp(&handle_, ref_problems.back(), regular_pdlp_settings);
    ref_statuses[i]   = ref_solution.get_termination_status(0);
    ASSERT_EQ(ref_statuses[i], pdlp_termination_status_t::Optimal);
    ref_objectives[i] = ref_solution.get_additional_termination_information(0).primal_objective;
    ref_primal_solutions[i] =
      host_copy(ref_solution.get_primal_solution(), ref_solution.get_primal_solution().stream());
  }

  auto batch_settings                                = regular_pdlp_settings;
  batch_settings.generate_batch_primal_dual_solution = true;
  for (int i = 0; i < batch_size; ++i) {
    for (const auto& bounds : custom_bounds_by_climber[i]) {
      batch_settings.new_bounds.push_back(
        {i, std::get<0>(bounds), std::get<1>(bounds), std::get<2>(bounds)});
    }
  }

  auto batch_solution = solve_lp(&handle_, op_problem, batch_settings);

  ASSERT_EQ((int)batch_solution.get_terminations_status().size(), batch_size);
  const size_t primal_size = op_problem.get_n_variables();
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_EQ(batch_solution.get_termination_status(i), ref_statuses[i]);
    EXPECT_NEAR(batch_solution.get_additional_termination_information(i).primal_objective,
                ref_objectives[i],
                exact_tolerance);

    const auto current_primal_solution =
      extract_subvector(batch_solution.get_primal_solution(), i * primal_size, primal_size);
    const auto host_primal_solution =
      host_copy(current_primal_solution, batch_solution.get_primal_solution().stream());
    for (size_t p = 0; p < primal_size; ++p) {
      EXPECT_NEAR(host_primal_solution[p], ref_primal_solutions[i][p], exact_tolerance);
    }
    test_objective_sanity(
      ref_problems[i],
      current_primal_solution,
      batch_solution.get_additional_termination_information(i).primal_objective);
    test_constraint_sanity(ref_problems[i],
                           batch_solution.get_additional_termination_information(i),
                           current_primal_solution,
                           exact_tolerance,
                           false);
  }
}

TEST(pdlp_class, run_batch_fixed_api_many_different_bounds_good_mps_some_var_bounds)
{
  constexpr double lower_bounds    = -33.0;
  constexpr double upper_bounds    = 10.0;
  constexpr double exact_tolerance = 1e-8;

  const raft::handle_t handle_{};
  auto path = make_path_absolute("linear_programming/good-mps-some-var-bounds.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  const auto& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const auto& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  const std::vector<std::vector<std::tuple<int, double, double>>> custom_bounds_by_climber = {
    {{0, lower_bounds - 100.0, upper_bounds}},
    {{1, variable_lower_bounds[1] - 3.0, variable_upper_bounds[1] + 5.0}},
    {{0, lower_bounds - 150.0, upper_bounds + 1.0},
     {1, variable_lower_bounds[1] - 7.0, variable_upper_bounds[1] + 11.0}},
    {{0, lower_bounds - 200.0, upper_bounds + 2.0}},
    {{1, variable_lower_bounds[1] - 13.0, variable_upper_bounds[1] + 17.0}},
    {{0, lower_bounds - 500.0, upper_bounds + 3.0},
     {1, variable_lower_bounds[1] - 19.0, variable_upper_bounds[1] + 23.0}},
  };
  const int batch_size = custom_bounds_by_climber.size();

  auto regular_pdlp_settings      = pdlp_solver_settings_t<int, double>{};
  regular_pdlp_settings.method    = cuopt::linear_programming::method_t::PDLP;
  regular_pdlp_settings.presolver = presolver_t::None;
  regular_pdlp_settings.set_optimality_tolerance(exact_tolerance);

  std::vector<double> ref_objectives(batch_size);
  std::vector<pdlp_termination_status_t> ref_statuses(batch_size);
  std::vector<cuopt::linear_programming::io::mps_data_model_t<int, double>> ref_problems;
  std::vector<std::vector<double>> ref_primal_solutions(batch_size);

  for (int i = 0; i < batch_size; ++i) {
    auto ref_problem = op_problem;
    for (const auto& bounds : custom_bounds_by_climber[i]) {
      ref_problem.get_variable_lower_bounds()[std::get<0>(bounds)] = std::get<1>(bounds);
      ref_problem.get_variable_upper_bounds()[std::get<0>(bounds)] = std::get<2>(bounds);
    }
    ref_problems.push_back(ref_problem);

    auto ref_solution = solve_lp(&handle_, ref_problems.back(), regular_pdlp_settings);
    ref_statuses[i]   = ref_solution.get_termination_status(0);
    ASSERT_EQ(ref_statuses[i], pdlp_termination_status_t::Optimal);
    ref_objectives[i] = ref_solution.get_additional_termination_information(0).primal_objective;
    ref_primal_solutions[i] =
      host_copy(ref_solution.get_primal_solution(), ref_solution.get_primal_solution().stream());
  }

  auto batch_settings                                = regular_pdlp_settings;
  batch_settings.generate_batch_primal_dual_solution = true;
  batch_settings.fixed_batch_size                    = batch_size;
  for (int i = 0; i < batch_size; ++i) {
    for (const auto& bounds : custom_bounds_by_climber[i]) {
      batch_settings.new_bounds.push_back(
        {i, std::get<0>(bounds), std::get<1>(bounds), std::get<2>(bounds)});
    }
  }

  auto gpu_op = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
    &handle_, op_problem);
  auto batch_solution = cuopt::linear_programming::run_batch_pdlp(gpu_op, batch_settings);

  ASSERT_EQ((int)batch_solution.get_terminations_status().size(), batch_size);
  const size_t primal_size = op_problem.get_n_variables();
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_EQ(batch_solution.get_termination_status(i), ref_statuses[i]);
    EXPECT_NEAR(batch_solution.get_additional_termination_information(i).primal_objective,
                ref_objectives[i],
                exact_tolerance);

    const auto current_primal_solution =
      extract_subvector(batch_solution.get_primal_solution(), i * primal_size, primal_size);
    const auto host_primal_solution =
      host_copy(current_primal_solution, batch_solution.get_primal_solution().stream());
    for (size_t p = 0; p < primal_size; ++p) {
      EXPECT_NEAR(host_primal_solution[p], ref_primal_solutions[i][p], exact_tolerance);
    }
    test_objective_sanity(
      ref_problems[i],
      current_primal_solution,
      batch_solution.get_additional_termination_information(i).primal_objective);
    test_constraint_sanity(ref_problems[i],
                           batch_solution.get_additional_termination_information(i),
                           current_primal_solution,
                           exact_tolerance,
                           false);
  }
}

TEST(pdlp_class, many_different_bounds)
{
  constexpr double lower_bounds = -33.0;
  constexpr double upper_bounds = 10;

  const raft::handle_t handle_{};
  auto path = make_path_absolute("linear_programming/good-mps-some-var-bounds.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

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
  std::vector<cuopt::linear_programming::io::mps_data_model_t<int, double>> ref_problems;
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
    solver_settings.new_bounds.push_back({i,
                                          std::get<0>(custom_bounds[i]),
                                          std::get<1>(custom_bounds[i]),
                                          std::get<2>(custom_bounds[i])});
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
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  const auto& variable_lower_bounds = op_problem.get_variable_lower_bounds();
  const auto& variable_upper_bounds = op_problem.get_variable_upper_bounds();

  std::vector<std::tuple<int, double, double>> custom_bounds = {
    {0, lower_bounds - 2500, upper_bounds},
    {0, variable_lower_bounds[0], variable_upper_bounds[0]},
  };
  const int batch_size = custom_bounds.size();
  std::vector<double> ref_objectives(batch_size);
  std::vector<pdlp_termination_status_t> ref_statuses(batch_size);
  std::vector<cuopt::linear_programming::io::mps_data_model_t<int, double>> ref_problems;
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
    solver_settings.new_bounds.push_back({i,
                                          std::get<0>(custom_bounds[i]),
                                          std::get<1>(custom_bounds[i]),
                                          std::get<2>(custom_bounds[i])});
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

TEST(pdlp_class, precision_single)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings           = pdlp_solver_settings_t<int, double>{};
  solver_settings.method         = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_precision = cuopt::linear_programming::pdlp_precision_t::SinglePrecision;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);

  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution.get_additional_termination_information().primal_objective));
}

TEST(pdlp_class, precision_single_crossover)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings           = pdlp_solver_settings_t<int, double>{};
  solver_settings.method         = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_precision = cuopt::linear_programming::pdlp_precision_t::SinglePrecision;
  solver_settings.crossover      = true;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);

  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution.get_additional_termination_information().primal_objective));
}

TEST(pdlp_class, precision_single_concurrent)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings           = pdlp_solver_settings_t<int, double>{};
  solver_settings.method         = cuopt::linear_programming::method_t::Concurrent;
  solver_settings.pdlp_precision = cuopt::linear_programming::pdlp_precision_t::SinglePrecision;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);

  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution.get_additional_termination_information().primal_objective));
}

TEST(pdlp_class, precision_single_papilo_presolve)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings           = pdlp_solver_settings_t<int, double>{};
  solver_settings.method         = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_precision = cuopt::linear_programming::pdlp_precision_t::SinglePrecision;
  solver_settings.presolver      = cuopt::linear_programming::presolver_t::Papilo;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution.get_additional_termination_information().primal_objective));
}

TEST(pdlp_class, precision_single_pslp_presolve)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings           = pdlp_solver_settings_t<int, double>{};
  solver_settings.method         = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_precision = cuopt::linear_programming::pdlp_precision_t::SinglePrecision;
  solver_settings.presolver      = cuopt::linear_programming::presolver_t::PSLP;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution.get_additional_termination_information().primal_objective));
}

// ---------------------------------------------------------------------------
// Cooperative strong branching tests
// ---------------------------------------------------------------------------

TEST(pdlp_class, shared_sb_context_unit)
{
  using namespace cuopt::linear_programming::dual_simplex;

  constexpr int N = 10;
  shared_strong_branching_context_t<int, double> ctx(N);
  shared_strong_branching_context_view_t<int, double> view(ctx.solved);

  EXPECT_TRUE(view.is_valid());

  shared_strong_branching_context_view_t<int, double> empty_view;
  EXPECT_FALSE(empty_view.is_valid());

  for (int i = 0; i < N; ++i) {
    EXPECT_FALSE(view.is_solved(i));
  }

  view.mark_solved(0);
  view.mark_solved(3);
  view.mark_solved(7);

  EXPECT_TRUE(view.is_solved(0));
  EXPECT_FALSE(view.is_solved(1));
  EXPECT_FALSE(view.is_solved(2));
  EXPECT_TRUE(view.is_solved(3));
  EXPECT_FALSE(view.is_solved(4));
  EXPECT_FALSE(view.is_solved(5));
  EXPECT_FALSE(view.is_solved(6));
  EXPECT_TRUE(view.is_solved(7));
  EXPECT_FALSE(view.is_solved(8));
  EXPECT_FALSE(view.is_solved(9));

  // subview(2, 5) covers global indices [2..6]
  auto sv = view.subview(2, 5);
  EXPECT_TRUE(sv.is_valid());
  EXPECT_FALSE(sv.is_solved(0));  // global 2
  EXPECT_TRUE(sv.is_solved(1));   // global 3
  EXPECT_FALSE(sv.is_solved(2));  // global 4
  EXPECT_FALSE(sv.is_solved(3));  // global 5
  EXPECT_FALSE(sv.is_solved(4));  // global 6

  // Mark through subview: local 4 -> global 6
  sv.mark_solved(4);
  EXPECT_TRUE(view.is_solved(6));
  EXPECT_TRUE(sv.is_solved(4));
}

TEST(pdlp_class, shared_sb_view_batch_pre_solved)
{
  using namespace cuopt::linear_programming::dual_simplex;

  const raft::handle_t handle_{};
  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  const std::vector<int> fractional     = {1, 2, 4};
  const std::vector<double> root_soln_x = {0.891, 0.109, 0.636429};
  const int n_fractional                = fractional.size();
  const int batch_size                  = n_fractional * 2;  // 6

  auto solver_settings             = pdlp_solver_settings_t<int, double>{};
  solver_settings.method           = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode = pdlp_solver_mode_t::Stable3;
  solver_settings.presolver        = cuopt::linear_programming::presolver_t::None;

  // Build new_bounds: down branches [0..2], up branches [3..5]
  for (int i = 0; i < n_fractional; ++i)
    solver_settings.new_bounds.push_back({i,
                                          fractional[i],
                                          op_problem.get_variable_lower_bounds()[fractional[i]],
                                          std::floor(root_soln_x[i])});
  for (int i = 0; i < n_fractional; ++i)
    solver_settings.new_bounds.push_back({i + n_fractional,
                                          fractional[i],
                                          std::ceil(root_soln_x[i]),
                                          op_problem.get_variable_upper_bounds()[fractional[i]]});

  shared_strong_branching_context_t<int, double> shared_ctx(batch_size);
  shared_strong_branching_context_view_t<int, double> sb_view(shared_ctx.solved);

  // Pre-mark entries 1 and 4 as solved (simulating DS)
  sb_view.mark_solved(1);
  sb_view.mark_solved(4);

  solver_settings.shared_sb_solved = sb_view.solved;

  auto solution = solve_lp(&handle_, op_problem, solver_settings);

  ASSERT_EQ(solution.get_terminations_status().size(), batch_size);

  // Pre-solved entries should have ConcurrentLimit
  EXPECT_EQ(solution.get_termination_status(1), pdlp_termination_status_t::ConcurrentLimit);
  EXPECT_EQ(solution.get_termination_status(4), pdlp_termination_status_t::ConcurrentLimit);

  // Others should be Optimal
  EXPECT_EQ(solution.get_termination_status(0), pdlp_termination_status_t::Optimal);
  EXPECT_EQ(solution.get_termination_status(2), pdlp_termination_status_t::Optimal);
  EXPECT_EQ(solution.get_termination_status(3), pdlp_termination_status_t::Optimal);
  EXPECT_EQ(solution.get_termination_status(5), pdlp_termination_status_t::Optimal);

  // All entries should now be marked solved in the shared context
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_TRUE(sb_view.is_solved(i)) << "Entry " << i << " should be solved";
  }
}

TEST(pdlp_class, shared_sb_view_concurrent_mark)
{
  using namespace cuopt::linear_programming::dual_simplex;

  const raft::handle_t handle_{};
  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  const std::vector<int> fractional     = {1, 2, 4};
  const std::vector<double> root_soln_x = {0.891, 0.109, 0.636429};
  const int n_fractional                = fractional.size();
  const int batch_size                  = n_fractional * 2;

  auto solver_settings             = pdlp_solver_settings_t<int, double>{};
  solver_settings.method           = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode = pdlp_solver_mode_t::Stable3;
  solver_settings.presolver        = cuopt::linear_programming::presolver_t::None;
  solver_settings.iteration_limit  = 1000000;

  for (int i = 0; i < n_fractional; ++i)
    solver_settings.new_bounds.push_back({i, fractional[0], -5, -5});

  for (int i = 0; i < n_fractional; ++i)
    solver_settings.new_bounds.push_back({i + n_fractional,
                                          fractional[i],
                                          std::ceil(root_soln_x[i]),
                                          op_problem.get_variable_upper_bounds()[fractional[i]]});

  shared_strong_branching_context_t<int, double> shared_ctx(batch_size);
  shared_strong_branching_context_view_t<int, double> sb_view(shared_ctx.solved);

  solver_settings.shared_sb_solved = sb_view.solved;

  optimization_problem_solution_t<int, double>* result_ptr = nullptr;

  auto pdlp_thread = std::thread([&]() {
    auto sol = new optimization_problem_solution_t<int, double>(
      solve_lp(&handle_, op_problem, solver_settings));
    result_ptr = sol;
  });

  // Wait a bit then mark entries 0, 2, 4 as solved (simulating DS)
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  for (int i = 0; i < n_fractional; ++i)
    sb_view.mark_solved(i);

  pdlp_thread.join();

  ASSERT_NE(result_ptr, nullptr);
  auto& solution = *result_ptr;

  ASSERT_EQ(solution.get_terminations_status().size(), batch_size);

  for (int i = 0; i < batch_size; ++i) {
    auto status = solution.get_termination_status(i);
    // Each entry should be either Optimal (PDLP solved it first) or ConcurrentLimit (DS marked it)
    EXPECT_TRUE(status == pdlp_termination_status_t::Optimal ||
                status == pdlp_termination_status_t::ConcurrentLimit)
      << "Entry " << i << " has unexpected status "
      << cuopt::linear_programming::optimization_problem_solution_t<int, double>::
           get_termination_status_string(status);
  }

  // All entries should end up marked solved
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_TRUE(sb_view.is_solved(i)) << "Entry " << i << " should be solved";
  }

  delete result_ptr;
}

TEST(pdlp_class, shared_sb_view_all_infeasible)
{
  using namespace cuopt::linear_programming::dual_simplex;

  const raft::handle_t handle_{};
  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  const std::vector<int> fractional     = {1, 2, 4};
  const std::vector<double> root_soln_x = {0.891, 0.109, 0.636429};
  const int n_fractional                = fractional.size();
  const int batch_size                  = n_fractional;

  auto solver_settings             = pdlp_solver_settings_t<int, double>{};
  solver_settings.method           = cuopt::linear_programming::method_t::PDLP;
  solver_settings.pdlp_solver_mode = pdlp_solver_mode_t::Stable3;
  solver_settings.presolver        = cuopt::linear_programming::presolver_t::None;
  solver_settings.iteration_limit  = 1000000;

  for (int i = 0; i < n_fractional; ++i)
    solver_settings.new_bounds.push_back({i, fractional[0], -5, -5});

  shared_strong_branching_context_t<int, double> shared_ctx(batch_size);
  shared_strong_branching_context_view_t<int, double> sb_view(shared_ctx.solved);

  solver_settings.shared_sb_solved = sb_view.solved;

  optimization_problem_solution_t<int, double>* result_ptr = nullptr;

  auto pdlp_thread = std::thread([&]() {
    auto sol = new optimization_problem_solution_t<int, double>(
      solve_lp(&handle_, op_problem, solver_settings));
    result_ptr = sol;
  });

  // Wait a bit then mark entries 0, 2, 4 as solved (simulating DS)
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  for (int i = 0; i < n_fractional; ++i)
    sb_view.mark_solved(i);

  pdlp_thread.join();

  ASSERT_NE(result_ptr, nullptr);
  auto& solution = *result_ptr;

  ASSERT_EQ(solution.get_terminations_status().size(), batch_size);

  for (int i = 0; i < batch_size; ++i) {
    auto status = solution.get_termination_status(i);
    // Each entry should be either Optimal (PDLP solved it first) or ConcurrentLimit (DS marked it)
    EXPECT_TRUE(status == pdlp_termination_status_t::ConcurrentLimit)
      << "Entry " << i << " has unexpected status "
      << cuopt::linear_programming::optimization_problem_solution_t<int, double>::
           get_termination_status_string(status);
  }

  // All entries should end up marked solved
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_TRUE(sb_view.is_solved(i)) << "Entry " << i << " should be solved";
  }

  delete result_ptr;
}

// Stress test: fixed path with all per-climber fields expanded at maximum safe scale.
// All climbers are identical: the point is to verify the fixed path doesn't crash at scale
// and produces bitwise-identical results.
TEST(pdlp_class, big_batch_fixed_path)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

  const int n_vars    = op_problem.get_n_variables();
  const int n_constrs = op_problem.get_n_constraints();

  const auto& original_obj     = op_problem.get_objective_coefficients();
  const auto& original_lb      = op_problem.get_constraint_lower_bounds();
  const auto& original_ub      = op_problem.get_constraint_upper_bounds();
  const auto& variable_lb      = op_problem.get_variable_lower_bounds();
  const auto& variable_ub      = op_problem.get_variable_upper_bounds();
  const double original_offset = op_problem.get_objective_offset();

  // Query optimal batch size on the unexpanded problem, then expand to that size.
  auto gpu_op = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
    &handle_, op_problem);
  const size_t batch_size =
    cuopt::linear_programming::compute_optimal_batch_size(gpu_op, true, true, true);
  ASSERT_GT(batch_size, 0u);

  // Build expanded arrays: replicate identical per-climber fields × batch_size
  std::vector<double> all_objectives;
  std::vector<double> all_constraint_lower;
  std::vector<double> all_constraint_upper;
  std::vector<double> all_offsets;
  all_objectives.reserve(batch_size * n_vars);
  all_constraint_lower.reserve(batch_size * n_constrs);
  all_constraint_upper.reserve(batch_size * n_constrs);
  all_offsets.reserve(batch_size);

  for (size_t i = 0; i < batch_size; ++i) {
    all_objectives.insert(all_objectives.end(), original_obj.begin(), original_obj.end());
    all_constraint_lower.insert(all_constraint_lower.end(), original_lb.begin(), original_lb.end());
    all_constraint_upper.insert(all_constraint_upper.end(), original_ub.begin(), original_ub.end());
    all_offsets.push_back(original_offset);
    solver_settings.new_bounds.push_back({static_cast<int>(i), 0, variable_lb[0], variable_ub[0]});
  }

  auto stream = handle_.get_stream();
  assign_device_uvector_from_host(gpu_op.get_objective_coefficients(), all_objectives, stream);
  assign_device_uvector_from_host(
    gpu_op.get_constraint_lower_bounds(), all_constraint_lower, stream);
  assign_device_uvector_from_host(
    gpu_op.get_constraint_upper_bounds(), all_constraint_upper, stream);
  gpu_op.set_batch_objective_offsets(all_offsets);

  solver_settings.generate_batch_primal_dual_solution = true;
  solver_settings.fixed_batch_size                    = static_cast<int>(batch_size);

  auto solution = cuopt::linear_programming::run_batch_pdlp(gpu_op, solver_settings);

  // All should be optimal
  for (size_t i = 0; i < batch_size; ++i) {
    EXPECT_EQ((int)solution.get_termination_status(i), CUOPT_TERMINATION_STATUS_OPTIMAL);
    EXPECT_FALSE(is_incorrect_objective(
      afiro_primal_objective, solution.get_additional_termination_information(i).primal_objective));
  }

  // All should be bitwise identical
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
                         1e-4,
                         false);
}

TEST(pdlp_class, batch_bound_objective_rescaling_factors_match_input_expansion)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  constexpr int batch_size = 3;
  const int n_vars         = op_problem.get_n_variables();
  const int n_constrs      = op_problem.get_n_constraints();
  const auto& original_obj = op_problem.get_objective_coefficients();
  const auto& original_lb  = op_problem.get_constraint_lower_bounds();
  const auto& original_ub  = op_problem.get_constraint_upper_bounds();

  auto compute_rescaling = [&](std::vector<double> const& objectives,
                               std::vector<double> const& constraint_lower,
                               std::vector<double> const& constraint_upper) {
    auto gpu_op = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
      &handle_, op_problem);
    auto stream = handle_.get_stream();
    assign_device_uvector_from_host(gpu_op.get_objective_coefficients(), objectives, stream);
    assign_device_uvector_from_host(gpu_op.get_constraint_lower_bounds(), constraint_lower, stream);
    assign_device_uvector_from_host(gpu_op.get_constraint_upper_bounds(), constraint_upper, stream);

    pdlp_hyper_params::pdlp_hyper_params_t hyper_params{};
    hyper_params.do_ruiz_scaling           = false;
    hyper_params.do_pock_chambolle_scaling = false;
    hyper_params.bound_objective_rescaling = true;

    cuopt::linear_programming::detail::problem_t<int, double> problem(gpu_op);
    cuopt::linear_programming::detail::pdlp_initial_scaling_strategy_t<int, double> scaling(
      &handle_,
      problem,
      hyper_params.default_l_inf_ruiz_iterations,
      hyper_params.default_alpha_pock_chambolle_rescaling,
      problem.reverse_coefficients,
      problem.reverse_offsets,
      problem.reverse_constraints,
      nullptr,
      hyper_params,
      batch_size,
      true);

    scaling.bound_objective_rescaling();
    return std::make_pair(host_copy(scaling.get_bound_rescaling_vector(), stream),
                          host_copy(scaling.get_objective_rescaling_vector(), stream));
  };

  enum class field_layout_t { UNEXPANDED, EXPANDED_SAME, EXPANDED_DIFFERENT };

  auto build_case = [&](field_layout_t objective_layout, field_layout_t rhs_layout) {
    std::vector<double> objectives;
    std::vector<double> constraint_lower;
    std::vector<double> constraint_upper;

    const int objective_segments = objective_layout == field_layout_t::UNEXPANDED ? 1 : batch_size;
    objectives.reserve(static_cast<size_t>(objective_segments) * n_vars);
    for (int climber = 0; climber < objective_segments; ++climber) {
      const double objective_scale =
        objective_layout == field_layout_t::EXPANDED_DIFFERENT ? std::pow(2.0, climber) : 1.0;

      for (double v : original_obj) {
        objectives.push_back(v * objective_scale);
      }
    }

    const int rhs_segments = rhs_layout == field_layout_t::UNEXPANDED ? 1 : batch_size;
    constraint_lower.reserve(static_cast<size_t>(rhs_segments) * n_constrs);
    constraint_upper.reserve(static_cast<size_t>(rhs_segments) * n_constrs);
    for (int climber = 0; climber < rhs_segments; ++climber) {
      const double rhs_scale =
        rhs_layout == field_layout_t::EXPANDED_DIFFERENT ? std::pow(2.0, climber) : 1.0;

      for (double v : original_lb) {
        constraint_lower.push_back(std::isfinite(v) ? v * rhs_scale : v);
      }
      for (double v : original_ub) {
        constraint_upper.push_back(std::isfinite(v) ? v * rhs_scale : v);
      }
    }
    return compute_rescaling(objectives, constraint_lower, constraint_upper);
  };

  auto expect_rescaling_equal = [=](const std::vector<double>& scaling) {
    ASSERT_EQ(scaling.size(), static_cast<size_t>(batch_size));
    for (int climber = 1; climber < batch_size; ++climber) {
      EXPECT_EQ(scaling[0], scaling[climber]);
    }
  };
  auto expect_rescaling_different = [=](const std::vector<double>& scaling) {
    ASSERT_EQ(scaling.size(), static_cast<size_t>(batch_size));
    for (int climber = 1; climber < batch_size; ++climber) {
      EXPECT_NE(scaling[0], scaling[climber]);
    }
  };

  {
    auto [bound_rescaling, objective_rescaling] =
      build_case(field_layout_t::EXPANDED_SAME, field_layout_t::EXPANDED_SAME);
    expect_rescaling_equal(bound_rescaling);
    expect_rescaling_equal(objective_rescaling);
  }
  {
    auto [bound_rescaling, objective_rescaling] =
      build_case(field_layout_t::EXPANDED_DIFFERENT, field_layout_t::EXPANDED_SAME);
    expect_rescaling_equal(bound_rescaling);
    expect_rescaling_different(objective_rescaling);
  }
  {
    auto [bound_rescaling, objective_rescaling] =
      build_case(field_layout_t::EXPANDED_SAME, field_layout_t::EXPANDED_DIFFERENT);
    expect_rescaling_different(bound_rescaling);
    expect_rescaling_equal(objective_rescaling);
  }
  {
    auto [bound_rescaling, objective_rescaling] =
      build_case(field_layout_t::EXPANDED_DIFFERENT, field_layout_t::EXPANDED_DIFFERENT);
    expect_rescaling_different(bound_rescaling);
    expect_rescaling_different(objective_rescaling);
  }
  {
    auto [bound_rescaling, objective_rescaling] =
      build_case(field_layout_t::UNEXPANDED, field_layout_t::UNEXPANDED);
    expect_rescaling_equal(bound_rescaling);
    expect_rescaling_equal(objective_rescaling);
  }
  {
    auto [bound_rescaling, objective_rescaling] =
      build_case(field_layout_t::UNEXPANDED, field_layout_t::EXPANDED_DIFFERENT);
    expect_rescaling_different(bound_rescaling);
    expect_rescaling_equal(objective_rescaling);
  }
  {
    auto [bound_rescaling, objective_rescaling] =
      build_case(field_layout_t::EXPANDED_DIFFERENT, field_layout_t::UNEXPANDED);
    expect_rescaling_equal(bound_rescaling);
    expect_rescaling_different(objective_rescaling);
  }
}

// Tests the compute_optimal_batch_size → run_batch_pdlp two-step API.
// First queries the optimal batch size, then builds that many climbers with different
// objectives, constraint bounds, and offsets then solves.
TEST(pdlp_class, batch_with_optimal_size_query)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::linear_programming::io::mps_data_model_t<int, double> op_problem =
    cuopt::linear_programming::io::parse_mps<int, double>(path, true);

  auto solver_settings      = pdlp_solver_settings_t<int, double>{};
  solver_settings.method    = cuopt::linear_programming::method_t::PDLP;
  solver_settings.presolver = presolver_t::None;

  const int n_vars    = op_problem.get_n_variables();
  const int n_constrs = op_problem.get_n_constraints();

  const auto& original_obj = op_problem.get_objective_coefficients();
  const auto& original_lb  = op_problem.get_constraint_lower_bounds();
  const auto& original_ub  = op_problem.get_constraint_upper_bounds();
  const auto& variable_lb  = op_problem.get_variable_lower_bounds();
  const auto& variable_ub  = op_problem.get_variable_upper_bounds();

  // Step 1: query optimal batch size on the unexpanded problem.
  auto gpu_op = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
    &handle_, op_problem);
  const size_t batch_size =
    cuopt::linear_programming::compute_optimal_batch_size(gpu_op, true, true, true);
  ASSERT_GT(batch_size, 0u);

  // Step 2: build per-climber expanded arrays sized to batch_size.
  // Each climber gets a different objective scale, offset, and constraint upper scale.
  // Cycle through a small set of variations.
  struct climber_spec {
    double obj_scale;
    double offset;
    double constr_upper_val;
  };
  const std::vector<climber_spec> variations = {
    {1.0, 0.0, 10},
    {1.5, 7.5, 1000},
    {2.0, -3.25, 10000},
  };

  std::vector<double> all_objectives;
  std::vector<double> all_offsets;
  std::vector<double> all_constraint_lower;
  std::vector<double> all_constraint_upper;

  std::vector<std::vector<double>> per_climber_obj(batch_size);
  std::vector<std::vector<double>> per_climber_lower(batch_size);
  std::vector<std::vector<double>> per_climber_upper(batch_size);
  std::vector<climber_spec> specs(batch_size);

  for (size_t c = 0; c < batch_size; ++c) {
    specs[c]           = variations[c % variations.size()];
    per_climber_obj[c] = std::vector<double>(original_obj.begin(), original_obj.end());
    for (auto& v : per_climber_obj[c])
      v *= specs[c].obj_scale;
    per_climber_lower[c] = std::vector<double>(original_lb.begin(), original_lb.end());
    per_climber_upper[c] = std::vector<double>(original_ub.begin(), original_ub.end());
    for (auto& v : per_climber_upper[c]) {
      if (std::isfinite(v)) v = specs[c].constr_upper_val;
    }
    all_objectives.insert(
      all_objectives.end(), per_climber_obj[c].begin(), per_climber_obj[c].end());
    all_offsets.push_back(specs[c].offset);
    all_constraint_lower.insert(
      all_constraint_lower.end(), per_climber_lower[c].begin(), per_climber_lower[c].end());
    all_constraint_upper.insert(
      all_constraint_upper.end(), per_climber_upper[c].begin(), per_climber_upper[c].end());
  }

  // Sequential reference: solve one instance of each unique variation independently.
  const size_t n_variations = variations.size();
  std::vector<double> ref_objectives(n_variations);
  std::vector<cuopt::linear_programming::io::mps_data_model_t<int, double>> ref_problems;
  ref_problems.reserve(n_variations);
  for (size_t v = 0; v < n_variations; ++v) {
    auto ref_op                           = op_problem;
    ref_op.get_objective_coefficients()   = per_climber_obj[v];
    ref_op.get_constraint_lower_bounds()  = per_climber_lower[v];
    ref_op.get_constraint_upper_bounds()  = per_climber_upper[v];
    ref_op.get_variable_lower_bounds()[0] = variable_lb[0];
    ref_op.get_variable_upper_bounds()[0] = variable_ub[0];
    ref_op.set_objective_offset(variations[v].offset);
    ref_problems.push_back(ref_op);

    auto sol = solve_lp(&handle_, ref_problems.back(), solver_settings);
    ASSERT_EQ((int)sol.get_termination_status(0), CUOPT_TERMINATION_STATUS_OPTIMAL);
    ref_objectives[v] = sol.get_additional_termination_information(0).primal_objective;
  }

  // Step 3: expand the problem fields on gpu_op and call run_batch_pdlp.
  auto stream = handle_.get_stream();
  assign_device_uvector_from_host(gpu_op.get_objective_coefficients(), all_objectives, stream);
  assign_device_uvector_from_host(
    gpu_op.get_constraint_lower_bounds(), all_constraint_lower, stream);
  assign_device_uvector_from_host(
    gpu_op.get_constraint_upper_bounds(), all_constraint_upper, stream);
  gpu_op.set_batch_objective_offsets(all_offsets);

  solver_settings.generate_batch_primal_dual_solution = true;
  solver_settings.fixed_batch_size                    = static_cast<int>(batch_size);

  auto batch_sol = cuopt::linear_programming::run_batch_pdlp(gpu_op, solver_settings);

  // Compare each climber to the reference for its variation.
  for (size_t c = 0; c < batch_size; ++c) {
    const size_t v = c % n_variations;
    EXPECT_EQ((int)batch_sol.get_termination_status(c), CUOPT_TERMINATION_STATUS_OPTIMAL);
    EXPECT_FALSE(is_incorrect_objective(
      ref_objectives[v], batch_sol.get_additional_termination_information(c).primal_objective));

    const auto primal = extract_subvector(batch_sol.get_primal_solution(), c * n_vars, n_vars);
    const double reported_obj =
      batch_sol.get_additional_termination_information(c).primal_objective;
    test_objective_sanity(ref_problems[v], primal, reported_obj - specs[c].offset);
    test_constraint_sanity(
      ref_problems[v], batch_sol.get_additional_termination_information(c), primal, 1e-4, false);
  }
}

}  // namespace cuopt::linear_programming::test

CUOPT_TEST_PROGRAM_MAIN()
