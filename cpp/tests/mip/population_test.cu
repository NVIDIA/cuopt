/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mip_heuristics/diversity/diversity_manager.cuh>
#include <mip_heuristics/lns/population_feed.cuh>
#include <mip_heuristics/utils.cuh>

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
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
  // fractional objective markers and use the second column as integer.
  const std::vector<opt::var_t> types{opt::var_t::CONTINUOUS, opt::var_t::INTEGER};
  op.set_csr_constraint_matrix(coefficients.data(), 2, columns.data(), 2, offsets.data(), 2);
  op.set_variable_lower_bounds(lower.data(), 2);
  op.set_variable_upper_bounds(upper.data(), 2);
  op.set_variable_types(types.data(), 2);
  op.set_objective_coefficients(objective.data(), 2);
  op.set_constraint_lower_bounds(row_lower.data(), 1);
  op.set_constraint_upper_bounds(row_upper.data(), 1);
}

class origin_callback_t : public cuopt::internals::get_solution_callback_with_data_t {
 public:
  void get_solution_with_data(
    void* data,
    void* objective_value,
    void* solution_bound,
    void* user_data,
    const cuopt::internals::solution_callback_data_t& callback_data) override
  {
    EXPECT_EQ(user_data, this);
    const auto* assignment = static_cast<double*>(data);
    EXPECT_DOUBLE_EQ(assignment[0] + 2 * assignment[1], *static_cast<double*>(objective_value));
    origins.push_back(callback_data.from_lns);
  }

  std::vector<int> origins;
};

}  // namespace

TEST(Population, ExplicitLnsOriginSurvivesPublicationAndQueueDrain)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_population_test_problem(op);
  opt::mip_solver_settings_t<int, double> settings;
  origin_callback_t callback;
  settings.set_mip_callback(&callback, &callback);
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  mip::diversity_manager_t<int, double> dm(context);
  dm.population.initialize_population();
  dm.population.allocate_solutions();

  // Identical queue origins still distinguish LNS from ordinary CPUFJ. All four
  // callbacks run on this thread, independently of its name or which worker drains the queue.
  dm.population.add_external_solution({0.75, 0}, 0.75, mip::solution_origin_t::CPUFJ, true);
  dm.population.add_external_solution({0.5, 0}, 0.5, mip::solution_origin_t::CPUFJ);
  dm.population.add_external_solution({0.25, 0}, 0.25, mip::solution_origin_t::CPUFJ, true);
  dm.population.add_external_solution({0.125, 0}, 0.125, mip::solution_origin_t::CPUFJ);
  EXPECT_EQ(callback.origins, (std::vector<int>{1, 0, 1, 0}));
  dm.population.add_external_solutions_to_population();
  EXPECT_EQ(callback.origins, (std::vector<int>{1, 0, 1, 0}));

  // A subsequent population improvement must not inherit an earlier LNS label.
  mip::solution_t<int, double> improved(problem);
  improved.copy_new_assignment(std::vector<double>{0, 0});
  ASSERT_TRUE(improved.compute_feasibility());
  dm.population.add_solution(std::move(improved));
  EXPECT_EQ(callback.origins, (std::vector<int>{1, 0, 1, 0, 0}));
}

TEST(Population, ExternalQueueKeepsGlobalBestFiftyAcrossOrigins)
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

  const std::vector<mip::solution_origin_t> origins{mip::solution_origin_t::CPUFJ,
                                                    mip::solution_origin_t::BRANCH_AND_BOUND,
                                                    mip::solution_origin_t::EXTERNAL};
  // Reuse the queue after each drain, with ascending, descending, and permuted arrivals.
  for (int order = 0; order < 3; ++order) {
    for (int i = 0; i < 100; ++i) {
      const int index    = order == 0 ? i : (order == 1 ? 99 - i : (i * 37) % 100);
      const double value = static_cast<double>(index) / 128;
      dm.population.add_external_solution({value, 0}, value, origins[index % origins.size()]);
      EXPECT_EQ(dm.population.get_external_solution_size(), std::min(i + 1, 50));
    }
    EXPECT_TRUE(dm.population.solutions_in_external_queue_.load());
    auto candidates = dm.population.get_external_solutions();
    ASSERT_EQ(candidates.size(), 50);
    // The retained set includes 17 CPUFJ entries, with no separate per-origin cap.
    for (size_t i = 0; i < candidates.size(); ++i) {
      EXPECT_TRUE(candidates[i].get_feasible());
      EXPECT_EQ(candidates[i].get_host_assignment(),
                (std::vector<double>{static_cast<double>(i) / 128, 0}));
    }
    EXPECT_EQ(dm.population.get_external_solution_size(), 0);
    EXPECT_FALSE(dm.population.solutions_in_external_queue_.load());
    EXPECT_TRUE(dm.population.get_external_solutions().empty());
  }
}

TEST(Population, ExternalQueueRejectsNonfiniteObjectivesAndNonimprovingOverflow)
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

  const double inf = std::numeric_limits<double>::infinity();
  const std::vector<double> nonfinite{std::numeric_limits<double>::quiet_NaN(), inf, -inf};
  for (double objective : nonfinite) {
    dm.population.add_external_solution({0.75, 0}, objective, mip::solution_origin_t::CPUFJ);
    EXPECT_EQ(dm.population.get_external_solution_size(), 0);
  }
  EXPECT_FALSE(dm.population.solutions_in_external_queue_.load());
  for (int i = 0; i < 50; ++i) {
    const double value = static_cast<double>(i) / 128;
    dm.population.add_external_solution({value, 0}, value, mip::solution_origin_t::EXTERNAL);
  }
  for (int i = 0; i < 100; ++i) {
    for (double objective : nonfinite) {
      dm.population.add_external_solution(
        {0.75, 0}, objective, mip::solution_origin_t::BRANCH_AND_BOUND);
    }
    // Distinct assignments expose accidental replacement on a tied objective.
    dm.population.add_external_solution({0.75, 0}, 49.0 / 128, mip::solution_origin_t::EXTERNAL);
    dm.population.add_external_solution({0.75, 0}, 0.75, mip::solution_origin_t::EXTERNAL);
    EXPECT_EQ(dm.population.get_external_solution_size(), 50);
  }
  auto candidates = dm.population.get_external_solutions();
  ASSERT_EQ(candidates.size(), 50);
  for (size_t i = 0; i < candidates.size(); ++i) {
    EXPECT_TRUE(candidates[i].get_feasible());
    EXPECT_EQ(candidates[i].get_host_assignment(),
              (std::vector<double>{static_cast<double>(i) / 128, 0}));
  }
  EXPECT_EQ(dm.population.get_external_solution_size(), 0);
}

TEST(Population, ExternalQueueValidatesWithSolverTolerancesBeforeReplacingIncumbent)
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

  dm.population.add_external_solution({1, 1}, 3, mip::solution_origin_t::BRANCH_AND_BOUND);
  dm.population.add_external_solutions_to_population();
  ASSERT_TRUE(dm.population.is_feasible());
  ASSERT_EQ(dm.population.best_feasible().get_objective(), 3);

  // Ranking uses reported objectives, but those reports cannot replace a validated incumbent.
  for (int i = 0; i < 50; ++i) {
    dm.population.add_external_solution({0, 0}, -1e9, mip::solution_origin_t::BRANCH_AND_BOUND);
  }
  dm.population.add_external_solution({1, 0}, 1, mip::solution_origin_t::BRANCH_AND_BOUND);
  dm.population.add_external_solutions_to_population();
  EXPECT_EQ(dm.population.get_external_solution_size(), 0);
  EXPECT_TRUE(dm.population.best_feasible().compute_feasibility());
  EXPECT_EQ(dm.population.best_feasible().get_objective(), 3);

  const double tolerance = mip::get_cstr_tolerance<int, double>(
    row_lower, 2, settings.tolerances.absolute_tolerance, settings.tolerances.relative_tolerance);
  const double outside = 1 - 2 * tolerance;
  const double inside  = 1 - 0.5 * tolerance;
  dm.population.add_external_solution({outside, 0}, outside, mip::solution_origin_t::EXTERNAL);
  dm.population.add_external_solution({inside, 0}, inside, mip::solution_origin_t::EXTERNAL);
  {
    auto candidates = dm.population.get_external_solutions();
    ASSERT_EQ(candidates.size(), 2);
    EXPECT_FALSE(candidates[0].get_feasible());
    EXPECT_TRUE(candidates[1].get_feasible());
  }
  dm.population.add_external_solution({inside, 0}, inside, mip::solution_origin_t::EXTERNAL);
  dm.population.add_external_solutions_to_population();
  EXPECT_EQ(dm.population.get_external_solution_size(), 0);
  EXPECT_TRUE(dm.population.best_feasible().compute_feasibility());
  EXPECT_EQ(dm.population.best_feasible().get_objective(), inside);
  EXPECT_EQ(dm.population.best_feasible().get_host_assignment(), (std::vector<double>{inside, 0}));
}

TEST(Population, ExternalQueueDrainAllowsReentrantProducerAndLeavesNewHeapPending)
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
    EXPECT_EQ(dm.population.get_external_solution_size(), 0);
    for (int i = 60; i > 0; --i) {
      const double value = static_cast<double>(i) / 128;
      dm.population.add_external_solution({value, 0}, value, mip::solution_origin_t::EXTERNAL);
    }
    return true;
  };
  dm.population.add_external_solution({0.75, 0}, 0.75, mip::solution_origin_t::EXTERNAL);
  dm.population.add_external_solutions_to_population();
  ASSERT_TRUE(produced);
  // The swapped heap is drained without the producer lock; callback traffic waits for next time.
  EXPECT_EQ(dm.population.best_feasible().get_objective(), 0.75);
  EXPECT_EQ(dm.population.get_external_solution_size(), 50);
  EXPECT_TRUE(dm.population.solutions_in_external_queue_.load());

  problem.branch_and_bound_callback = {};
  dm.population.add_external_solutions_to_population();
  EXPECT_EQ(dm.population.get_external_solution_size(), 0);
  EXPECT_FALSE(dm.population.solutions_in_external_queue_.load());
  EXPECT_EQ(dm.population.best_feasible().get_objective(), 1.0 / 128);
}

TEST(Population, LnsFeedTracksOnlyBestSlotAndDetachesOnDestruction)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_population_test_problem(op);
  const double objective[] = {1, -0.5};
  op.set_objective_coefficients(objective, 2);
  opt::mip_solver_settings_t<int, double> settings;
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  mip::diversity_manager_t<int, double> dm(context);
  dm.population.initialize_population();
  dm.population.allocate_solutions();

  dm.population.add_external_solution({0.75, 0}, 0.75, mip::solution_origin_t::EXTERNAL);
  dm.population.add_external_solutions_to_population();
  {
    mip::lns_population_feed_t<int, double> feed(dm.population);
    std::vector<double> best(2);
    ASSERT_TRUE(feed.best_feasible(best));
    EXPECT_EQ(best, (std::vector<double>{0.75, 0}));
    for (int i = 10; i > 0; --i) {
      const double value = i / 16.0;
      dm.population.add_external_solution({value, 0}, value, mip::solution_origin_t::EXTERNAL);
      dm.population.add_external_solutions_to_population();
    }
    ASSERT_TRUE(feed.best_feasible(best));
    EXPECT_EQ(best, (std::vector<double>{1.0 / 16, 0}));

    // A diverse accepted member can have a slightly lower objective without
    // passing the population's best-slot improvement margin. Keep that margin.
    const double value = 1.0 / 16 - mip::OBJECTIVE_EPSILON / 2;
    const std::vector<double> nearby{value + 0.5, 1};
    dm.population.add_external_solution(nearby, value, mip::solution_origin_t::EXTERNAL);
    dm.population.add_external_solutions_to_population();
    EXPECT_TRUE(std::any_of(
      dm.population.solutions.begin(), dm.population.solutions.end(), [&](auto& member) {
        return member.first && member.second.get_host_assignment() == nearby;
      }));
    EXPECT_EQ(dm.population.best_feasible().get_objective(), 1.0 / 16);
    ASSERT_TRUE(feed.best_feasible(best));
    EXPECT_EQ(best, (std::vector<double>{1.0 / 16, 0}));
  }
  {
    mip::lns_population_feed_t<int, double> feed(dm.population);
    std::vector<double> best(2);
    ASSERT_TRUE(feed.best_feasible(best));
    EXPECT_EQ(best, (std::vector<double>{1.0 / 16, 0}));
  }
  // Reattaching would fail if the destroyed feed had left its callback installed.
  std::vector<std::vector<double>> received;
  dm.population.set_feasible_solution_callback(
    [&received](const auto& assignment, double, double) { received.push_back(assignment); });
  ASSERT_EQ(received.size(), 1);
  EXPECT_EQ(received[0], (std::vector<double>{1.0 / 16, 0}));
  received.clear();
  dm.population.add_external_solution({0, 0}, 0, mip::solution_origin_t::EXTERNAL);
  dm.population.add_external_solutions_to_population();
  ASSERT_EQ(received.size(), 1);
  EXPECT_EQ(received[0], (std::vector<double>{0, 0}));
  dm.population.clear_feasible_solution_callback();
}

}  // namespace cuopt::mathematical_optimization::test
