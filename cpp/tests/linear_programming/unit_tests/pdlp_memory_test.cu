/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <pdlp/pdlp.cuh>
#include <pdlp/solve.cuh>
#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <unistd.h>

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

TEST(PdlpMemoryProblem, ClearPresolvedProblemPreservesSolverCopyAndMetadata)
{
  raft::handle_t handle;
  auto op = make_memory_test_problem(handle);
  mip::problem_t<int, double> problem(op);
  const auto values  = host_copy(problem.coefficients, handle.get_stream());
  const auto columns = host_copy(problem.variables, handle.get_stream());
  const auto offsets = host_copy(problem.offsets, handle.get_stream());
  op.clear();
  EXPECT_EQ(op.get_constraint_matrix_values().capacity(), 0);
  EXPECT_EQ(op.get_constraint_matrix_indices().capacity(), 0);
  EXPECT_EQ(op.get_constraint_matrix_offsets().capacity(), 0);
  EXPECT_EQ(problem.original_problem_ptr, &op);
  EXPECT_EQ(op.get_n_variables(), 3);
  EXPECT_EQ(op.get_n_constraints(), 2);
  EXPECT_EQ(op.get_objective_coefficients().capacity(), 0);
  EXPECT_EQ(op.get_variable_lower_bounds().capacity(), 0);
  EXPECT_EQ(op.get_variable_upper_bounds().capacity(), 0);
  EXPECT_EQ(op.get_constraint_lower_bounds().capacity(), 0);
  EXPECT_EQ(op.get_constraint_upper_bounds().capacity(), 0);
  EXPECT_EQ(host_copy(problem.objective_coefficients, handle.get_stream()),
            (std::vector<double>{1, 2, 3}));
  EXPECT_EQ(host_copy(problem.coefficients, handle.get_stream()), values);
  EXPECT_EQ(host_copy(problem.variables, handle.get_stream()), columns);
  EXPECT_EQ(host_copy(problem.offsets, handle.get_stream()), offsets);
  auto settings            = pdlp_solver_settings_t<int, double>{};
  settings.iteration_limit = 2000;
  settings.set_optimality_tolerance(1e-6);
  set_pdlp_solver_mode(settings);
  pdlp::pdlp_solver_t<int, double> solver(problem, settings);
  auto result = solver.run_solver(timer_t(30));
  EXPECT_EQ(result.get_termination_status(), pdlp_termination_status_t::Optimal);
  EXPECT_NEAR(result.get_objective_value(), 2.0, 1e-5);
  handle.sync_stream();
}

TEST(PdlpMemoryProblem, ClearReleasesAllDeviceCapacityAndIsIdempotent)
{
  raft::handle_t handle;
  auto op = make_memory_test_problem(handle);
  const std::vector<double> rhs{2, 3};
  const std::vector<char> row_types{'E', 'E'};
  const std::vector<var_t> variable_types(3, var_t::CONTINUOUS);
  const std::vector<std::string> variable_names{"x", "y", "z"}, row_names{"a", "b"};
  op.set_constraint_bounds(rhs.data(), rhs.size());
  op.set_row_types(row_types.data(), row_types.size());
  op.set_variable_types(variable_types.data(), variable_types.size());
  op.set_variable_names(variable_names);
  op.set_row_names(row_names);
  op.set_objective_offset(4);
  op.set_objective_scaling_factor(2);
  op.set_batch_objective_offsets({4, 5});
  op.set_objective_name("cost");
  op.set_problem_name("presolved");
  op.set_maximize(true);
  for (int repeat = 0; repeat < 2; ++repeat) {
    op.clear();
    EXPECT_EQ(op.get_constraint_matrix_values().capacity(), 0);
    EXPECT_EQ(op.get_constraint_matrix_indices().capacity(), 0);
    EXPECT_EQ(op.get_constraint_matrix_offsets().capacity(), 0);
    EXPECT_EQ(op.get_constraint_bounds().capacity(), 0);
    EXPECT_EQ(op.get_objective_coefficients().capacity(), 0);
    EXPECT_EQ(op.get_variable_lower_bounds().capacity(), 0);
    EXPECT_EQ(op.get_variable_upper_bounds().capacity(), 0);
    EXPECT_EQ(op.get_constraint_lower_bounds().capacity(), 0);
    EXPECT_EQ(op.get_constraint_upper_bounds().capacity(), 0);
    EXPECT_EQ(op.get_row_types().capacity(), 0);
    EXPECT_EQ(op.get_variable_types().capacity(), 0);
    EXPECT_EQ(op.get_handle_ptr(), &handle);
    EXPECT_EQ(op.get_n_variables(), 3);
    EXPECT_EQ(op.get_n_constraints(), 2);
    EXPECT_EQ(op.get_objective_offset(), 4);
    EXPECT_EQ(op.get_objective_scaling_factor(), 2);
    EXPECT_EQ(op.get_batch_objective_offsets(), (std::vector<double>{4, 5}));
    EXPECT_EQ(op.get_objective_name(), "cost");
    EXPECT_EQ(op.get_problem_name(), "presolved");
    EXPECT_EQ(op.get_variable_names(), variable_names);
    EXPECT_EQ(op.get_row_names(), row_names);
    EXPECT_TRUE(op.get_sense());
  }
  handle.sync_stream();
}

optimization_problem_t<int, double> make_presolved_csr_test_problem(const raft::handle_t& handle)
{
  optimization_problem_t<int, double> op(&handle);
  const std::vector<double> values{1, 1, 1, 1, 1, 1, 1};
  const std::vector<int> columns{0, 1, 3, 1, 2, 0, 2}, offsets{0, 3, 5, 7};
  const std::vector<double> objective{1, 1, 1, 3}, lower{0, 0, 0, 2}, upper{1, 1, 1, 2};
  const std::vector<double> row_lower{3, 1, 1};
  const std::vector<double> row_upper(3, std::numeric_limits<double>::infinity());
  op.set_csr_constraint_matrix(
    values.data(), values.size(), columns.data(), columns.size(), offsets.data(), offsets.size());
  op.set_objective_coefficients(objective.data(), objective.size());
  op.set_variable_lower_bounds(lower.data(), lower.size());
  op.set_variable_upper_bounds(upper.data(), upper.size());
  op.set_constraint_lower_bounds(row_lower.data(), row_lower.size());
  op.set_constraint_upper_bounds(row_upper.data(), row_upper.size());
  return op;
}

class PdlpMemoryPresolvedCsr : public testing::TestWithParam<std::tuple<presolver_t, method_t>> {
 protected:
  void SetUp() override
  {
    char path[]  = "/tmp/cuopt-presolved-csr-XXXXXX.mps";
    const int fd = mkstemps(path, 4);
    ASSERT_GE(fd, 0);
    close(fd);
    presolve_file_ = path;
  }

  void TearDown() override
  {
    if (!presolve_file_.empty()) { std::remove(presolve_file_.c_str()); }
  }

  std::string presolve_file_;
};

TEST_P(PdlpMemoryPresolvedCsr, PostsolveAndMpsOutputPreserveCallerMatrix)
{
  raft::handle_t handle;
  auto op                  = make_presolved_csr_test_problem(handle);
  const auto values        = op.get_constraint_matrix_values_host();
  const auto columns       = op.get_constraint_matrix_indices_host();
  const auto offsets       = op.get_constraint_matrix_offsets_host();
  auto settings            = pdlp_solver_settings_t<int, double>{};
  settings.presolver       = std::get<0>(GetParam());
  settings.method          = std::get<1>(GetParam());
  settings.presolve_file   = presolve_file_;
  settings.crossover       = false;
  settings.time_limit      = 30;
  settings.iteration_limit = 5000;
  settings.set_optimality_tolerance(1e-6);
  // Reusing the same caller model also checks that only solver-owned CSR is released.
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto result = solve_lp(op, settings);
    ASSERT_EQ(result.get_termination_status(), pdlp_termination_status_t::Optimal);
    EXPECT_NEAR(result.get_objective_value(), 7.5, 1e-5);
    const auto primal = host_copy(result.get_primal_solution(), handle.get_stream());
    ASSERT_EQ(primal.size(), 4);
    EXPECT_NEAR(primal[0], 0.5, 1e-5);
    EXPECT_NEAR(primal[1], 0.5, 1e-5);
    EXPECT_NEAR(primal[2], 0.5, 1e-5);
    EXPECT_NEAR(primal[3], 2.0, 1e-5);
    EXPECT_EQ(op.get_constraint_matrix_values_host(), values);
    EXPECT_EQ(op.get_constraint_matrix_indices_host(), columns);
    EXPECT_EQ(op.get_constraint_matrix_offsets_host(), offsets);
    const auto presolved = io::read<int, double>(presolve_file_);
    // The fixed variable is removed, but a nonempty CSR must reach the solver.
    EXPECT_EQ(presolved.get_objective_coefficients().size(), 3);
    EXPECT_GT(presolved.get_constraint_matrix_values().size(), 0);
  }
}

INSTANTIATE_TEST_SUITE_P(PresolversAndMethods,
                         PdlpMemoryPresolvedCsr,
                         testing::Combine(testing::Values(presolver_t::PSLP, presolver_t::Papilo),
                                          testing::Values(method_t::PDLP,
                                                          method_t::Concurrent,
                                                          method_t::DualSimplex,
                                                          method_t::Barrier,
                                                          method_t::Primal)));

TEST(PdlpMemoryProblem, PresolvedCsrRemainsAvailableForFp32Conversion)
{
  raft::handle_t handle;
  auto op                  = make_presolved_csr_test_problem(handle);
  auto settings            = pdlp_solver_settings_t<int, double>{};
  settings.method          = method_t::PDLP;
  settings.presolver       = presolver_t::PSLP;
  settings.pdlp_precision  = pdlp_precision_t::SinglePrecision;
  settings.crossover       = false;
  settings.time_limit      = 30;
  settings.iteration_limit = 5000;
  settings.set_optimality_tolerance(1e-5);
  auto result = solve_lp(op, settings);
  ASSERT_EQ(result.get_termination_status(), pdlp_termination_status_t::Optimal);
  EXPECT_NEAR(result.get_objective_value(), 7.5, 1e-4);
  EXPECT_EQ(result.get_primal_solution().size(), 4);
  EXPECT_EQ(op.get_nnz(), 7);
}
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
  EXPECT_EQ(solver.pdhg_solver_.get_saddle_point_state().get_next_AtY().size(), 3);
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

TEST(PdlpMemoryProblem, LpConstructionOmitsMipWorkspace)
{
  raft::handle_t handle;
  auto op = make_memory_test_problem(handle);
  ASSERT_EQ(op.get_problem_category(), problem_category_t::LP);
  mip::problem_t<int, double> problem(op);
  expect_no_mip_workspace(problem);
  EXPECT_EQ(problem.n_variables, 3);
  EXPECT_EQ(problem.n_constraints, 2);
  EXPECT_EQ(problem.nnz, 4);
  EXPECT_EQ(host_copy(problem.objective_coefficients, handle.get_stream()),
            (std::vector<double>{1, 2, 3}));
  mip::problem_t<int, double> copy(problem);
  expect_no_mip_workspace(copy);
  mip::problem_t<int, double> scaled(problem, false);
  expect_no_mip_workspace(scaled);
  EXPECT_NE(scaled.coefficients.data(), problem.coefficients.data());
  EXPECT_EQ(host_copy(scaled.coefficients, handle.get_stream()),
            host_copy(problem.coefficients, handle.get_stream()));
  handle.sync_stream();
}

TEST(PdlpMemoryProblem, ExplicitRelaxationOmitsWorkspaceAndPreservesMetadata)
{
  raft::handle_t handle;
  auto op = make_memory_test_problem(handle);
  const std::vector<var_t> types{var_t::INTEGER, var_t::CONTINUOUS, var_t::CONTINUOUS};
  op.set_variable_types(types.data(), types.size());
  // Disable only MIP workspace, not the model's integer-variable metadata.
  mip::problem_t<int, double> relaxation(op, {}, false, false);
  expect_no_mip_workspace(relaxation);
  EXPECT_EQ(relaxation.n_integer_vars, 1);
  EXPECT_EQ(host_copy(relaxation.variable_types, handle.get_stream()), types);
  for (auto presolver : {presolver_t::None, presolver_t::PSLP}) {
    auto settings            = pdlp_solver_settings_t<int, double>{};
    settings.method          = method_t::PDLP;
    settings.presolver       = presolver;
    settings.iteration_limit = 2000;
    settings.set_optimality_tolerance(1e-6);
    auto result = solve_lp(op, settings);
    EXPECT_EQ(result.get_termination_status(), pdlp_termination_status_t::Optimal);
    EXPECT_NEAR(result.get_additional_termination_information().primal_objective, 2.0, 1e-5);
    EXPECT_EQ(op.get_problem_category(), problem_category_t::MIP);
    EXPECT_EQ(host_copy(op.get_variable_types(), handle.get_stream()), types);
  }
  handle.sync_stream();
}

void expect_mip_workspace(const mip::problem_t<int, double>& problem)
{
  EXPECT_EQ(problem.integer_fixed_variable_map.size(), 3);
  EXPECT_EQ(problem.related_variables_offsets.size(), 3);
  EXPECT_EQ(problem.lp_state.prev_primal.size(), 3);
  EXPECT_EQ(problem.lp_state.prev_dual.size(), 2);
  EXPECT_EQ(problem.fixing_helpers.reduction_in_rhs.size(), 2);
  EXPECT_EQ(problem.fixing_helpers.variable_fix_mask.size(), 3);
}

TEST(PdlpMemoryProblem, MipConstructionAndCopiesPreserveWorkspace)
{
  raft::handle_t handle;
  for (bool all_integer : {false, true}) {
    auto op = make_memory_test_problem(handle);
    const std::vector<var_t> types{var_t::INTEGER,
                                   all_integer ? var_t::INTEGER : var_t::CONTINUOUS,
                                   all_integer ? var_t::INTEGER : var_t::CONTINUOUS};
    op.set_variable_types(types.data(), types.size());
    ASSERT_NE(op.get_problem_category(), problem_category_t::LP);
    mip::problem_t<int, double> problem(op);
    expect_mip_workspace(problem);
    EXPECT_EQ(host_copy(problem.lp_state.prev_primal, handle.get_stream()),
              (std::vector<double>{0, 0, 0}));
    EXPECT_EQ(host_copy(problem.lp_state.prev_dual, handle.get_stream()),
              (std::vector<double>{0, 0}));
    problem.related_variables    = device_copy(std::vector<int>{1, 2}, handle.get_stream());
    problem.lp_state.prev_primal = device_copy(std::vector<double>{1, 2, 3}, handle.get_stream());
    mip::problem_t<int, double> copy(problem);
    mip::problem_t<int, double> stream_copy(problem, &handle);
    mip::problem_t<int, double> fixing_copy(problem, true);
    for (const auto* copied : {&copy, &stream_copy, &fixing_copy}) {
      expect_mip_workspace(*copied);
      EXPECT_NE(copied->lp_state.prev_primal.data(), problem.lp_state.prev_primal.data());
      EXPECT_EQ(host_copy(copied->lp_state.prev_primal, handle.get_stream()),
                (std::vector<double>{1, 2, 3}));
      EXPECT_EQ(host_copy(copied->related_variables, handle.get_stream()),
                (std::vector<int>{1, 2}));
    }
    // PDLP's copy must omit MIP state even when its source has populated buffers.
    mip::problem_t<int, double> scaled(problem, false);
    expect_no_mip_workspace(scaled);
    expect_mip_workspace(problem);
    EXPECT_EQ(host_copy(problem.lp_state.prev_primal, handle.get_stream()),
              (std::vector<double>{1, 2, 3}));
    EXPECT_NE(scaled.coefficients.data(), problem.coefficients.data());
    EXPECT_EQ(host_copy(scaled.coefficients, handle.get_stream()),
              host_copy(problem.coefficients, handle.get_stream()));
  }
  handle.sync_stream();
}

TEST(PdlpMemoryProblem, PdlpSolvePreservesCallerMipWorkspace)
{
  raft::handle_t handle;
  auto op = make_memory_test_problem(handle);
  const std::vector<var_t> types{var_t::INTEGER, var_t::CONTINUOUS, var_t::CONTINUOUS};
  op.set_variable_types(types.data(), types.size());
  mip::problem_t<int, double> problem(op);
  const auto* caller_primal = problem.lp_state.prev_primal.data();
  const auto* caller_mask   = problem.fixing_helpers.variable_fix_mask.data();
  auto settings             = pdlp_solver_settings_t<int, double>{};
  settings.inside_mip       = true;
  settings.iteration_limit  = 2000;
  settings.set_optimality_tolerance(1e-6);
  set_pdlp_solver_mode(settings);
  pdlp::pdlp_solver_t<int, double> solver(problem, settings);
  expect_no_mip_workspace(solver.get_initial_scaling_strategy().get_scaled_op_problem());
  auto result = solver.run_solver(timer_t(30));
  EXPECT_EQ(result.get_termination_status(), pdlp_termination_status_t::Optimal);
  EXPECT_NEAR(result.get_additional_termination_information().primal_objective, 2.0, 1e-5);
  expect_mip_workspace(problem);
  EXPECT_EQ(problem.lp_state.prev_primal.data(), caller_primal);
  EXPECT_EQ(problem.fixing_helpers.variable_fix_mask.data(), caller_mask);
  EXPECT_EQ(host_copy(problem.lp_state.prev_primal, handle.get_stream()),
            (std::vector<double>{0, 0, 0}));
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
