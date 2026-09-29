/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mip_heuristics/diversity/diversity_manager.cuh>

#include <gtest/gtest.h>

#include <vector>

namespace cuopt::mathematical_optimization::test {
namespace opt = cuopt::mathematical_optimization;

namespace {

void init_population_test_problem(opt::optimization_problem_t<int, double>& op)
{
  const std::vector<double> coefficients{1, 1}, lower{0, 0}, upper{1, 1}, objective{1, 2};
  const std::vector<double> row_lower{0}, row_upper{2};
  const std::vector<int> columns{0, 1}, offsets{0, 2};
  // Population hashing needs an integer column; keep x continuous for the
  // fractional FIFO/objective markers and use the second column as integer.
  const std::vector<opt::var_t> types{opt::var_t::CONTINUOUS, opt::var_t::INTEGER};
  op.set_csr_constraint_matrix(coefficients.data(), 2, columns.data(), 2, offsets.data(), 2);
  op.set_variable_lower_bounds(lower.data(), 2);
  op.set_variable_upper_bounds(upper.data(), 2);
  op.set_variable_types(types.data(), 2);
  op.set_objective_coefficients(objective.data(), 2);
  op.set_constraint_lower_bounds(row_lower.data(), 1);
  op.set_constraint_upper_bounds(row_upper.data(), 1);
}

}  // namespace

TEST(Population, ExternalQueueMaterializesBoundedFifoBatches)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_population_test_problem(op);
  opt::mip_solver_settings_t<int, double> settings;
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  mip::diversity_manager_t<int, double> dm(context);
  dm.population.initialize_population();
  dm.population.allocate_solutions();

  // B&B-origin candidates use the general queue, which can contain a long
  // backlog while the GPU consumer is busy. Materialize only a bounded batch.
  for (int i = 0; i < 25; ++i) {
    const double value = static_cast<double>(i) / 32;
    dm.population.add_external_solution(
      {value, 0}, value, mip::solution_origin_t::BRANCH_AND_BOUND);
  }
  size_t consumed = 0;
  for (const size_t expected_size : {10, 10, 5}) {
    auto batch = dm.population.get_external_solutions();
    ASSERT_EQ(batch.size(), expected_size);
    for (auto& candidate : batch) {
      ASSERT_TRUE(candidate.get_feasible());
      EXPECT_EQ(candidate.get_host_assignment(),
                (std::vector<double>{static_cast<double>(consumed) / 32, 0}));
      ++consumed;
    }
    EXPECT_EQ(dm.population.get_external_solution_size(), 25 - consumed);
  }
  EXPECT_EQ(consumed, 25);
  EXPECT_TRUE(dm.population.get_external_solutions().empty());
}

TEST(Population, ExternalQueueBatchesAdvanceBothOriginsDespiteGeneralReplenishment)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_population_test_problem(op);
  opt::mip_solver_settings_t<int, double> settings;
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  mip::diversity_manager_t<int, double> dm(context);
  dm.population.initialize_population();
  dm.population.allocate_solutions();
  auto append_general = [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      const double value = static_cast<double>(i) / 64;
      dm.population.add_external_solution(
        {value, 0}, value, mip::solution_origin_t::BRANCH_AND_BOUND);
    }
  };
  append_general(0, 20);
  for (int i = 32; i < 42; ++i) {
    const double value = static_cast<double>(i) / 64;
    dm.population.add_external_solution({value, 0}, value, mip::solution_origin_t::CPUFJ);
  }
  int consumed_general = 0;
  int consumed_cpufj   = 0;
  auto consume_batch   = [&] {
    auto batch = dm.population.get_external_solutions();
    EXPECT_EQ(batch.size(), 10);
    for (auto& candidate : batch) {
      EXPECT_TRUE(candidate.get_feasible());
      const auto assignment = candidate.get_host_assignment();
      EXPECT_EQ(assignment[1], 0);
      if (assignment[0] < 0.5) {
        EXPECT_EQ(assignment[0], static_cast<double>(consumed_general++) / 64);
      } else {
        EXPECT_EQ(assignment[0], static_cast<double>(32 + consumed_cpufj++) / 64);
      }
    }
  };
  consume_batch();
  EXPECT_EQ(consumed_general, 5);
  EXPECT_EQ(consumed_cpufj, 5);
  EXPECT_TRUE(dm.population.solutions_in_external_queue_.load());

  // More general traffic must not put older CPUFJ submissions behind it.
  append_general(20, 30);
  consume_batch();
  EXPECT_EQ(consumed_general, 10);
  EXPECT_EQ(consumed_cpufj, 10);
  EXPECT_TRUE(dm.population.solutions_in_external_queue_.load());
  EXPECT_EQ(dm.population.get_external_solution_size(), 20);

  consume_batch();
  EXPECT_TRUE(dm.population.solutions_in_external_queue_.load());
  consume_batch();
  EXPECT_EQ(consumed_general, 30);
  EXPECT_EQ(consumed_cpufj, 10);
  EXPECT_FALSE(dm.population.solutions_in_external_queue_.load());
  EXPECT_TRUE(dm.population.get_external_solutions().empty());
}

TEST(Population, ExternalQueueDrainFindsValidTailAfterRejectedLowObjectives)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_population_test_problem(op);
  const double row_lower = 1;
  op.set_constraint_lower_bounds(&row_lower, 1);
  opt::mip_solver_settings_t<int, double> settings;
  settings.tolerances.absolute_tolerance = 3e-7;
  settings.tolerances.relative_tolerance = 4e-8;
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  mip::diversity_manager_t<int, double> dm(context);
  dm.population.initialize_population();
  dm.population.allocate_solutions();

  // The first full batch is invalid despite its attractive reported objective.
  // A valid improvement past two batch boundaries must still be consumed.
  for (int i = 0; i < 12; ++i) {
    dm.population.add_external_solution({0, 0}, -1e9, mip::solution_origin_t::BRANCH_AND_BOUND);
  }
  for (int i = 0; i < 12; ++i) {
    dm.population.add_external_solution({1, 1}, 3, mip::solution_origin_t::BRANCH_AND_BOUND);
  }
  dm.population.add_external_solution({1, 0}, 1, mip::solution_origin_t::BRANCH_AND_BOUND);
  dm.population.add_external_solutions_to_population();

  EXPECT_EQ(dm.population.get_external_solution_size(), 0);
  ASSERT_TRUE(dm.population.is_feasible());
  EXPECT_TRUE(dm.population.best_feasible().compute_feasibility());
  EXPECT_EQ(dm.population.best_feasible().get_objective(), 1);
  EXPECT_EQ(dm.population.best_feasible().get_host_assignment(), (std::vector<double>{1, 0}));
}

TEST(Population, ExternalQueueDrainAllowsReentrantProducerAndLeavesNewWorkPending)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_population_test_problem(op);
  opt::mip_solver_settings_t<int, double> settings;
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  mip::diversity_manager_t<int, double> dm(context);
  dm.population.initialize_population();
  dm.population.allocate_solutions();

  bool produced                     = false;
  problem.branch_and_bound_callback = [&](const auto&, auto) {
    if (produced) return true;
    produced = true;
    for (int i = 20; i > 0; --i) {
      const double value = static_cast<double>(i) / 64;
      dm.population.add_external_solution({value, 0}, value, mip::solution_origin_t::EXTERNAL);
    }
    return true;
  };
  dm.population.add_external_solution({0.75, 0}, 0.75, mip::solution_origin_t::EXTERNAL);
  dm.population.add_external_solutions_to_population();
  ASSERT_TRUE(produced);
  // The drain processes its initial workload without holding the producer lock
  // across callbacks or extending its budget to follow newly produced traffic.
  EXPECT_EQ(dm.population.best_feasible().get_objective(), 0.75);
  EXPECT_EQ(dm.population.get_external_solution_size(), 20);

  problem.branch_and_bound_callback = {};
  dm.population.add_external_solutions_to_population();
  EXPECT_EQ(dm.population.get_external_solution_size(), 0);
  EXPECT_EQ(dm.population.best_feasible().get_objective(), 1.0 / 64);
}

}  // namespace cuopt::mathematical_optimization::test
