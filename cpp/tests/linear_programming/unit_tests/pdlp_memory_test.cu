/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <pdlp/pdlp.cuh>
#include <pdlp/problem_memory.cuh>
#include <pdlp/solve.cuh>
#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::test {
namespace {

optimization_problem_t<int, double> make_memory_test_problem(const raft::handle_t& handle)
{
  optimization_problem_t<int, double> op(&handle);
  const std::vector<double> values{1, 2, 3, 4};
  const std::vector<int> columns{0, 1, 1, 2}, offsets{0, 2, 4};
  const std::vector<double> objective{1, 2, 3}, lower{0, 0, 0}, upper{10, 10, 10};
  const std::vector<double> row_lower{2, 3}, row_upper{2, 3};
  op.set_csr_constraint_matrix(
    values.data(), values.size(), columns.data(), columns.size(), offsets.data(), offsets.size());
  op.set_objective_coefficients(objective.data(), objective.size());
  op.set_variable_lower_bounds(lower.data(), lower.size());
  op.set_variable_upper_bounds(upper.data(), upper.size());
  op.set_constraint_lower_bounds(row_lower.data(), row_lower.size());
  op.set_constraint_upper_bounds(row_upper.data(), row_upper.size());
  return op;
}

using memory_test_params = std::tuple<pdlp_solver_mode_t, bool>;

class PdlpMemory : public testing::TestWithParam<memory_test_params> {};

TEST_P(PdlpMemory, OnlyAllocateWorkspacesUsedBySelectedMode)
{
  raft::handle_t handle;
  auto op = make_memory_test_problem(handle);
  mip::problem_t<int, double> problem(op);
  problem.compute_transpose_of_problem();
  auto settings                 = pdlp_solver_settings_t<int, double>{};
  settings.pdlp_solver_mode     = std::get<0>(GetParam());
  settings.detect_infeasibility = std::get<1>(GetParam());
  settings.iteration_limit      = 2000;
  settings.set_optimality_tolerance(1e-6);
  set_pdlp_solver_mode(settings);

  pdlp::pdlp_solver_t<int, double> solver(problem, settings);
  EXPECT_EQ(solver.pdhg_solver_.get_saddle_point_state().get_next_AtY().size(),
            settings.hyper_params.use_adaptive_step_size_strategy ? 3 : 0);
  auto& scaling = solver.get_initial_scaling_strategy();
  // resize(0) alone would pass a size check but still pin the entire allocation.
  EXPECT_EQ(scaling.get_iteration_variable_scaling().capacity(), 0);
  EXPECT_EQ(scaling.get_iteration_constraint_matrix_scaling().capacity(), 0);
  const auto convergence =
    solver.get_current_termination_strategy().get_convergence_information().view();
  EXPECT_EQ(convergence.bound_value.size(),
            settings.hyper_params.use_reflected_primal_dual ? 0 : 3);

  // Construct the same infeasibility workspace directly to inspect its device view.
  std::vector<pdlp_climber_strategy_t> climbers(1);
  auto& sparse = solver.pdhg_solver_.get_cusparse_view();
  {
    pdlp::infeasibility_information_t<int, double> infeasibility(&handle,
                                                                 problem,
                                                                 scaling.get_scaled_op_problem(),
                                                                 sparse,
                                                                 sparse,
                                                                 3,
                                                                 2,
                                                                 scaling,
                                                                 settings.detect_infeasibility,
                                                                 climbers,
                                                                 settings.hyper_params);
    const auto view        = infeasibility.view();
    const bool uses_legacy = settings.detect_infeasibility &&
                             !pdlp::is_cupdlpx_restart<int, double>(settings.hyper_params);
    EXPECT_EQ(view.homogenous_primal_residual != nullptr, uses_legacy);
    EXPECT_EQ(view.homogenous_dual_residual != nullptr, uses_legacy);
    EXPECT_EQ(view.reduced_cost != nullptr, uses_legacy);
    if (settings.detect_infeasibility) {
      // Exercise both certificate paths, including SpMM when cuPDLPx needs it.
      infeasibility.compute_infeasibility_information(
        solver.pdhg_solver_,
        solver.pdhg_solver_.get_saddle_point_state().get_delta_primal(),
        solver.pdhg_solver_.get_saddle_point_state().get_delta_dual());
      double primal_violation, dual_violation;
      raft::update_host(
        &primal_violation, view.max_primal_ray_infeasibility.data(), 1, handle.get_stream());
      raft::update_host(
        &dual_violation, view.max_dual_ray_infeasibility.data(), 1, handle.get_stream());
      handle.sync_stream();
      EXPECT_DOUBLE_EQ(primal_violation, 0);
      EXPECT_DOUBLE_EQ(dual_violation, 0);
    }
  }

  auto result                = solver.run_solver(timer_t(30));
  const auto& scaled_problem = scaling.get_scaled_op_problem();
  EXPECT_EQ(scaled_problem.variables.capacity(), 0);
  EXPECT_EQ(scaled_problem.offsets.capacity(), 0);
  EXPECT_EQ(scaled_problem.reverse_constraints.capacity(), 0);
  EXPECT_EQ(scaled_problem.reverse_offsets.capacity(), 0);
  EXPECT_EQ(result.get_termination_status(), pdlp_termination_status_t::Optimal);
  EXPECT_NEAR(result.get_additional_termination_information().primal_objective, 2.0, 1e-5);
  handle.sync_stream();
}

INSTANTIATE_TEST_SUITE_P(SolverModes,
                         PdlpMemory,
                         testing::Combine(testing::Values(pdlp_solver_mode_t::Stable2,
                                                          pdlp_solver_mode_t::Stable3,
                                                          pdlp_solver_mode_t::Methodical1,
                                                          pdlp_solver_mode_t::Fast1),
                                          testing::Bool()));

TEST(PdlpMemoryScaling, DeferredScalingKeepsScratchUntilExplicitRelease)
{
  raft::handle_t handle;
  auto op = make_memory_test_problem(handle);
  mip::problem_t<int, double> problem(op);
  problem.compute_transpose_of_problem();
  auto settings             = pdlp_solver_settings_t<int, double>{};
  settings.pdlp_solver_mode = pdlp_solver_mode_t::Stable3;
  set_pdlp_solver_mode(settings);
  pdlp::pdlp_solver_t<int, double> solver(problem, settings, false, true);
  auto& scaling = solver.get_initial_scaling_strategy();
  EXPECT_EQ(scaling.get_iteration_variable_scaling().size(), 3);
  EXPECT_EQ(scaling.get_iteration_constraint_matrix_scaling().size(), 2);
  scaling.compute_scaling_vectors(2, 1.0);
  const auto factors = host_copy(scaling.get_variable_scaling_vector(), handle.get_stream());
  for (auto value : factors) {
    EXPECT_GT(value, 0);
  }
  scaling.release_iteration_scratch();
  EXPECT_EQ(scaling.get_iteration_variable_scaling().capacity(), 0);
  EXPECT_EQ(scaling.get_iteration_constraint_matrix_scaling().capacity(), 0);
  scaling.compute_scaling_vectors(2, 1.0);
  EXPECT_EQ(scaling.get_iteration_variable_scaling().size(), 3);
  EXPECT_EQ(scaling.get_iteration_constraint_matrix_scaling().size(), 2);
  scaling.release_iteration_scratch();
  handle.sync_stream();
}

void expect_no_mip_workspace(const mip::problem_t<int, double>& problem)
{
  EXPECT_EQ(problem.integer_fixed_variable_map.capacity(), 0);
  EXPECT_EQ(problem.related_variables.capacity(), 0);
  EXPECT_EQ(problem.related_variables_offsets.capacity(), 0);
  EXPECT_EQ(problem.lp_state.prev_primal.capacity(), 0);
  EXPECT_EQ(problem.lp_state.prev_dual.capacity(), 0);
  EXPECT_EQ(problem.fixing_helpers.reduction_in_rhs.capacity(), 0);
  EXPECT_EQ(problem.fixing_helpers.variable_fix_mask.capacity(), 0);
}

TEST(PdlpMemoryProblem, ReleaseOwnedWorkspacePreservesModelAndCallerState)
{
  raft::handle_t handle;
  auto op = make_memory_test_problem(handle);
  mip::problem_t<int, double> problem(op);
  const auto* caller_primal = problem.lp_state.prev_primal.data();
  const auto* caller_mask   = problem.fixing_helpers.variable_fix_mask.data();
  {
    auto scaled = pdlp::make_scaled_pdlp_problem(problem);
    expect_no_mip_workspace(scaled);
    EXPECT_EQ(problem.lp_state.prev_primal.data(), caller_primal);
    EXPECT_EQ(problem.fixing_helpers.variable_fix_mask.data(), caller_mask);
    EXPECT_EQ(problem.lp_state.prev_primal.size(), 3);
    EXPECT_EQ(problem.fixing_helpers.variable_fix_mask.size(), 3);
    EXPECT_NE(scaled.coefficients.data(), problem.coefficients.data());
    EXPECT_EQ(host_copy(scaled.coefficients, handle.get_stream()),
              host_copy(problem.coefficients, handle.get_stream()));
  }
  const auto* matrix = problem.coefficients.data();
  const auto* bounds = problem.variable_bounds.data();
  pdlp::release_mip_only_workspace(problem);
  expect_no_mip_workspace(problem);
  EXPECT_EQ(problem.coefficients.data(), matrix);
  EXPECT_EQ(problem.variable_bounds.data(), bounds);
  EXPECT_EQ(problem.n_variables, 3);
  EXPECT_EQ(problem.n_constraints, 2);
  EXPECT_EQ(problem.nnz, 4);
  EXPECT_EQ(host_copy(problem.objective_coefficients, handle.get_stream()),
            (std::vector<double>{1, 2, 3}));
  handle.sync_stream();
}

using warm_start_t = pdlp_warm_start_data_t<int, double>;

constexpr std::array warm_start_vectors{&warm_start_t::current_primal_solution_,
                                        &warm_start_t::current_dual_solution_,
                                        &warm_start_t::initial_primal_average_,
                                        &warm_start_t::initial_dual_average_,
                                        &warm_start_t::current_ATY_,
                                        &warm_start_t::sum_primal_solutions_,
                                        &warm_start_t::sum_dual_solutions_,
                                        &warm_start_t::last_restart_duality_gap_primal_solution_,
                                        &warm_start_t::last_restart_duality_gap_dual_solution_};

warm_start_t make_memory_test_warm_start(cuda::stream_ref stream)
{
  warm_start_t data;
  for (auto member : warm_start_vectors) {
    data.*member = device_copy(std::vector<double>{1, 2, 3}, stream);
  }
  data.initial_primal_weight_         = 2;
  data.initial_step_size_             = 0.25;
  data.total_pdlp_iterations_         = 10;
  data.total_pdhg_iterations_         = 12;
  data.last_candidate_kkt_score_      = 0.5;
  data.last_restart_kkt_score_        = 0.75;
  data.sum_solution_weight_           = 3;
  data.iterations_since_last_restart_ = 4;
  stream.sync();
  return data;
}

auto warm_start_storage(const warm_start_t& data)
{
  std::array<const double*, warm_start_vectors.size()> pointers;
  for (size_t i = 0; i < warm_start_vectors.size(); ++i) {
    pointers[i] = (data.*warm_start_vectors[i]).data();
  }
  return pointers;
}

void expect_memory_test_warm_start(const warm_start_t& data, cuda::stream_ref stream)
{
  EXPECT_TRUE(data.is_populated());
  for (auto member : warm_start_vectors) {
    EXPECT_EQ(host_copy(data.*member, stream), (std::vector<double>{1, 2, 3}));
  }
  EXPECT_DOUBLE_EQ(data.initial_primal_weight_, 2);
  EXPECT_DOUBLE_EQ(data.initial_step_size_, 0.25);
  EXPECT_EQ(data.total_pdlp_iterations_, 10);
  EXPECT_EQ(data.total_pdhg_iterations_, 12);
  EXPECT_DOUBLE_EQ(data.last_candidate_kkt_score_, 0.5);
  EXPECT_DOUBLE_EQ(data.last_restart_kkt_score_, 0.75);
  EXPECT_DOUBLE_EQ(data.sum_solution_weight_, 3);
  EXPECT_EQ(data.iterations_since_last_restart_, 4);
}

TEST(PdlpMemoryWarmStart, MoveConstructionTransfersDeviceBuffers)
{
  raft::handle_t handle;
  auto data           = make_memory_test_warm_start(handle.get_stream());
  const auto pointers = warm_start_storage(data);
  warm_start_t moved(std::move(data));
  EXPECT_TRUE((std::is_nothrow_move_constructible_v<warm_start_t>));
  EXPECT_EQ(warm_start_storage(moved), pointers);
  for (auto member : warm_start_vectors) {
    EXPECT_TRUE((data.*member).is_empty());
  }
  expect_memory_test_warm_start(moved, handle.get_stream());
}

TEST(PdlpMemoryWarmStart, SolutionReturnPreservesDeviceBufferOwnership)
{
  raft::handle_t handle;
  auto stream         = handle.get_stream();
  auto data           = make_memory_test_warm_start(stream);
  const auto pointers = warm_start_storage(data);
  auto primal         = device_copy(std::vector<double>{1, 2, 3}, stream);
  auto dual           = device_copy(std::vector<double>{4, 5, 6}, stream);
  auto reduced_cost   = device_copy(std::vector<double>{7, 8, 9}, stream);
  using solution_t    = optimization_problem_solution_t<int, double>;
  solution_t solution(primal,
                      dual,
                      reduced_cost,
                      std::move(data),
                      "",
                      {},
                      {},
                      std::vector<solution_t::additional_termination_information_t>(1),
                      std::vector{pdlp_termination_status_t::IterationLimit});
  EXPECT_EQ(warm_start_storage(solution.get_pdlp_warm_start_data()), pointers);
  for (auto member : warm_start_vectors) {
    EXPECT_TRUE((data.*member).is_empty());
  }
  auto returned = std::move(solution);
  EXPECT_EQ(warm_start_storage(returned.get_pdlp_warm_start_data()), pointers);
  EXPECT_EQ(returned.get_termination_status(), pdlp_termination_status_t::IterationLimit);
  expect_memory_test_warm_start(returned.get_pdlp_warm_start_data(), stream);
}

TEST(PdlpMemoryWarmStart, ExplicitCopyStillOwnsIndependentBuffers)
{
  raft::handle_t handle;
  const auto original = make_memory_test_warm_start(handle.get_stream());
  warm_start_t copy(original);
  for (auto member : warm_start_vectors) {
    EXPECT_NE((copy.*member).data(), (original.*member).data());
  }
  expect_memory_test_warm_start(original, handle.get_stream());
  expect_memory_test_warm_start(copy, handle.get_stream());
}

}  // namespace
}  // namespace cuopt::mathematical_optimization::test
