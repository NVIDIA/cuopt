/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../benchmarks/linear_programming/cuopt/c_api_check.hpp"

#include <mip_heuristics/diversity/diversity_manager.cuh>
#include <mip_heuristics/feasibility_jump/cpu/climber.hpp>
#include <mip_heuristics/feasibility_jump/early_cpufj.cuh>
#include <mip_heuristics/feasibility_jump/feasibility_jump.cuh>
#include <mip_heuristics/lns/cpufj.cuh>
#include <mip_heuristics/lns/cpufj_validation.cuh>
#include <mip_heuristics/lns/thread_budget.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <exception>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

namespace cuopt::lns::test {
namespace mip = cuopt::mathematical_optimization::mip;
namespace opt = cuopt::mathematical_optimization;

using tolerances_t = opt::mip_solver_settings_t<int, double>::tolerances_t;

struct host_model_t {
  std::vector<double> coefficients, lower, upper, objective, row_lower, row_upper;
  std::vector<int> columns, offsets;
  std::vector<opt::var_t> types;
};

std::unique_ptr<mip::fj_cpu_climber_t<int, double>> make_anchor(const host_model_t& m,
                                                                std::atomic<bool>& preemption,
                                                                tolerances_t tolerances)
{
  return mip::init_fj_cpu_from_host_model<int, double>((int)m.lower.size(),
                                                       (int)m.row_lower.size(),
                                                       (int)m.coefficients.size(),
                                                       false,
                                                       1.0,
                                                       0.0,
                                                       m.coefficients,
                                                       m.columns,
                                                       m.offsets,
                                                       m.objective,
                                                       m.lower,
                                                       m.upper,
                                                       m.row_lower,
                                                       m.row_upper,
                                                       {},
                                                       {},
                                                       m.types,
                                                       tolerances,
                                                       preemption,
                                                       {});
}

tolerances_t test_tolerances()
{
  tolerances_t tolerances;
  tolerances.absolute_tolerance    = 1e-7;
  tolerances.relative_tolerance    = 2e-8;
  tolerances.integrality_tolerance = 1e-4;
  return tolerances;
}

host_model_t single_variable(
  bool integer, double lower, double upper, double row_lower, double row_upper)
{
  return {{1.0},
          {lower},
          {upper},
          {1.0},
          {row_lower},
          {row_upper},
          {0},
          {0, 1},
          {integer ? opt::var_t::INTEGER : opt::var_t::CONTINUOUS}};
}

TEST(Lns, PopulationSeedUsesModelBoundsBeyondCpufjIntegerCap)
{
  std::atomic<bool> preemption{false};
  const auto model = single_variable(true, 0, 4e7, 0, 4e7);
  auto anchor      = make_anchor(model, preemption, test_tolerances());
  ASSERT_EQ(get_upper(anchor->h_var_bounds[0].get()), 1e7);
  const std::vector<double2> bounds{make_double2(0, 4e7)};
  const auto model_feasible = [&](const std::vector<double>& x) {
    return mip::verify_cpufj_lns_feasible(*anchor->problem, bounds, model.types, x);
  };
  EXPECT_TRUE(model_feasible({2e7}));
  EXPECT_FALSE(model_feasible({4e7 + 1}));
  EXPECT_FALSE(model_feasible({2e7 + .5}));

  auto seed = std::vector<double>{2e7};
  ASSERT_TRUE(mip::clamp_and_validate_cpufj_lns_seed(*anchor, seed, model_feasible));
  EXPECT_EQ(seed, (std::vector<double>{1e7}));
  seed = {4e7 + 1};
  EXPECT_FALSE(mip::clamp_and_validate_cpufj_lns_seed(*anchor, seed, model_feasible));
  // The CPUFJ worker still searches its private restricted domain.
  EXPECT_EQ(get_upper(anchor->h_var_bounds[0].get()), 1e7);
}

TEST(Lns, PopulationSeedKeepsOriginalContinuousTypes)
{
  std::atomic<bool> preemption{false};
  // CPUFJ strengthens this pair to integers because an integral optimum exists,
  // although fractional feasible population seeds are valid for the solver model.
  const host_model_t model{{1, -1},
                           {0, 0},
                           {10, 10},
                           {1, 1},
                           {1},
                           {1},
                           {0, 1},
                           {0, 2},
                           {opt::var_t::CONTINUOUS, opt::var_t::CONTINUOUS}};
  auto anchor = make_anchor(model, preemption, test_tolerances());
  ASSERT_EQ(anchor->problem->h_var_types,
            (std::vector<opt::var_t>{opt::var_t::INTEGER, opt::var_t::INTEGER}));
  const std::vector<double2> bounds{make_double2(0, 10), make_double2(0, 10)};
  const auto model_feasible = [&](const std::vector<double>& x) {
    return mip::verify_cpufj_lns_feasible(*anchor->problem, bounds, model.types, x);
  };
  const std::vector<double> source{1.5, .5};
  ASSERT_TRUE(model_feasible(source));
  auto normalized = source;
  ASSERT_TRUE(
    mip::clamp_and_validate_cpufj_lns_seed(*anchor->problem, bounds, model.types, normalized));
  EXPECT_EQ(normalized, source);
  EXPECT_FALSE(model_feasible({1.5, .6}));

  ASSERT_TRUE(mip::clamp_and_validate_cpufj_lns_seed(*anchor, normalized, model_feasible));
  EXPECT_EQ(normalized, (std::vector<double>{2, 1}));
  normalized = {1.5, .6};
  EXPECT_FALSE(mip::clamp_and_validate_cpufj_lns_seed(*anchor, normalized, model_feasible));
}

TEST(Lns, CpufjWorkerSkipsRepeatedIncompatiblePopulationSeed)
{
  std::atomic<bool> preemption{false};
  const host_model_t model{{1, -.5},
                           {0, 0},
                           {4e7, 8e7},
                           {1, 1},
                           {0},
                           {0},
                           {0, 1},
                           {0, 2},
                           {opt::var_t::INTEGER, opt::var_t::INTEGER}};
  auto anchor = make_anchor(model, preemption, test_tolerances());
  const std::vector<double2> bounds{make_double2(0, 4e7), make_double2(0, 8e7)};
  const auto model_feasible = [&](const std::vector<double>& x) {
    return mip::verify_cpufj_lns_feasible(*anchor->problem, bounds, model.types, x);
  };
  const std::vector<double> incompatible{2e7, 4e7}, compatible{1, 2};
  ASSERT_TRUE(model_feasible(incompatible));
  auto projected = incompatible;
  EXPECT_FALSE(mip::clamp_and_validate_cpufj_lns_seed(*anchor, projected, model_feasible));
  // Independent clamping to the CPUFJ caps breaks x0 - .5*x1 = 0.
  EXPECT_FALSE(model_feasible(projected));

  int snapshots = 0, validations = 0;
  mip::run_cpufj_lns_ruin_repair<int, double>(
    anchor.get(),
    [&](auto& x) {
      ++snapshots;
      x = snapshots < 4 ? incompatible : compatible;
      if (snapshots == 4) preemption = true;
      return true;
    },
    [&](const auto& x) {
      ++validations;
      return model_feasible(x);
    });
  EXPECT_EQ(snapshots, 4);
  // One model validation per distinct seed; the unchanged replacement needs no second check.
  EXPECT_EQ(validations, 2);
  ASSERT_TRUE(anchor->feasible_found);
  EXPECT_TRUE(model_feasible(anchor->h_best_assignment.underlying()));
}

TEST(Lns, MainSolveBudgetMatchesEnabledWorkers)
{
  for (int size = 2; size <= 48; ++size) {
    const int workers = mip::lns_worker_count(size, false);
    EXPECT_EQ(workers, size >= 7 ? 1 : 0);
    EXPECT_EQ(mip::lns_worker_count(size, true), 0);
  }
}

TEST(Lns, CpufjLnsRevalidatesWithSolverTolerances)
{
  mip::fj_cpu_problem_t<int, double> problem;
  problem.n_variables = problem.n_constraints = 1;
  problem.offsets                             = {0, 1};
  problem.variables                           = {0};
  problem.coefficients = problem.h_obj_coeffs = {1.0};
  problem.h_var_types                         = {opt::var_t::CONTINUOUS};
  problem.cstr_lb = problem.cstr_ub        = {1e6};
  problem.tolerances.absolute_tolerance    = 1e-7;
  problem.tolerances.relative_tolerance    = 2e-8;
  problem.tolerances.integrality_tolerance = 1e-4;
  std::vector<double2> bounds{make_double2(1e6, 1e6 + 1)};
  EXPECT_TRUE(mip::verify_cpufj_lns_feasible(problem, bounds, {1e6 + .01}));
  EXPECT_FALSE(mip::verify_cpufj_lns_feasible(problem, bounds, {1e6 + .03}));
  EXPECT_FALSE(
    mip::verify_cpufj_lns_feasible(problem, bounds, {std::numeric_limits<double>::quiet_NaN()}));
  problem.h_var_types = {opt::var_t::INTEGER};
  bounds              = {make_double2(0, 1)};
  problem.cstr_lb     = {0};
  problem.cstr_ub     = {2};
  EXPECT_TRUE(mip::verify_cpufj_lns_feasible(problem, bounds, {1 + 5e-5}));
  EXPECT_FALSE(mip::verify_cpufj_lns_feasible(problem, bounds, {1 + 2e-4}));
  auto seed = std::vector<double>{1 - 5e-5};
  ASSERT_TRUE(mip::clamp_and_validate_cpufj_lns_seed(problem, bounds, seed));
  EXPECT_EQ(seed[0], 1);
  problem.cstr_lb = problem.cstr_ub = {1 - 5e-5};
  seed                              = {1 - 5e-5};
  EXPECT_FALSE(mip::clamp_and_validate_cpufj_lns_seed(problem, bounds, seed));
  bounds          = {make_double2(.2, .99999)};
  problem.cstr_lb = {0};
  problem.cstr_ub = {2};
  seed            = {.99999};
  EXPECT_FALSE(mip::clamp_and_validate_cpufj_lns_seed(problem, bounds, seed));
}

void init_test_problem(opt::optimization_problem_t<int, double>& op,
                       bool integer           = false,
                       double row_lower_bound = 0)
{
  const std::vector<double> coefficients{1, 1}, lower{0, 0}, upper{1, 1}, objective{1, 2};
  const std::vector<double> row_lower{row_lower_bound}, row_upper{2};
  const std::vector<int> columns{0, 1}, offsets{0, 2};
  const std::vector<opt::var_t> types(2, integer ? opt::var_t::INTEGER : opt::var_t::CONTINUOUS);
  op.set_csr_constraint_matrix(coefficients.data(), 2, columns.data(), 2, offsets.data(), 2);
  op.set_variable_lower_bounds(lower.data(), 2);
  op.set_variable_upper_bounds(upper.data(), 2);
  op.set_variable_types(types.data(), 2);
  op.set_objective_coefficients(objective.data(), 2);
  op.set_constraint_lower_bounds(row_lower.data(), 1);
  op.set_constraint_upper_bounds(row_upper.data(), 1);
}

class lns_origin_callback_t : public cuopt::internals::get_solution_callback_with_data_t {
 public:
  void get_solution_with_data(void* data,
                              void* objective_value,
                              void* solution_bound,
                              void* user_data,
                              const cuOptMIPCallbackData& callback_data) override
  {
    EXPECT_EQ(user_data, this);
    const auto* assignment = static_cast<double*>(data);
    EXPECT_DOUBLE_EQ(assignment[0] + 2 * assignment[1], *static_cast<double*>(objective_value));
    origins.push_back(callback_data.from_lns);
  }

  std::vector<int> origins;
};

TEST(Lns, ActiveWorkerCallbackPublishesExplicitOrigin)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_test_problem(op, true);
  opt::mip_solver_settings_t<int, double> settings;
  lns_origin_callback_t callback;
  settings.set_mip_callback(&callback, &callback);
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  std::exception_ptr task_exception;
  context.task_exception = &task_exception;
  // Install the production callback, but stop the task before it can search or publish.
  context.preempt_heuristic_solver_ = true;
  mip::diversity_manager_t<int, double> dm(context);
  dm.population.initialize_population();
  dm.population.allocate_solutions();

#pragma omp parallel num_threads(7)
  {
#pragma omp masked
    {
#pragma omp taskgroup
      {
        dm.ls.start_cpufj_lns_improvement_thread(dm.population);
      }
    }
  }
  ASSERT_FALSE(task_exception);
  ASSERT_NE(dm.ls.scratch_cpu_fj_lns, nullptr);
  ASSERT_TRUE(dm.ls.scratch_cpu_fj_lns->improvement_callback);
  EXPECT_TRUE(callback.origins.empty());

  // Invoke the installed producer only after its task has joined, on the same thread
  // as ordinary CPUFJ publication. A missing producer flag must fail this test.
  dm.ls.scratch_cpu_fj_lns->improvement_callback(3, {1, 1}, 0);
  dm.population.add_external_solution({0, 1}, 2, mip::solution_origin_t::CPUFJ);
  dm.ls.scratch_cpu_fj_lns->improvement_callback(1, {1, 0}, 0);
  dm.population.add_external_solution({0, 0}, 0, mip::solution_origin_t::CPUFJ);
  EXPECT_EQ(callback.origins, (std::vector<int>{1, 0, 1, 0}));
  dm.population.add_external_solutions_to_population();
  EXPECT_EQ(callback.origins, (std::vector<int>{1, 0, 1, 0}));
}

TEST(Lns, OptionalClimberDoesNotAdvanceFeasibilityRng)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_test_problem(op, true, 1);
  opt::mip_solver_settings_t<int, double> settings;
  settings.seed = 42;
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_context_t<int, double> reference_context(&handle, &problem, settings);
  mip::mip_solver_context_t<int, double> lns_context(&handle, &problem, settings);
  mip::fj_t<int, double> reference(reference_context), with_lns(lns_context);
  mip::solution_t<int, double> solution(problem);
  solution.copy_new_assignment(std::vector<double>{1, 0});
  const std::vector<double> weights(problem.n_constraints, 1.0);
  std::atomic<bool> preemption{false};
  const auto ordinary = [&](auto& fj, bool randomize) {
    // Omit preserve_rng to exercise its existing default behavior.
    return fj.create_cpu_climber(
      solution, weights, weights, 0.0, preemption, nullptr, mip::fj_settings_t{}, randomize);
  };
  const auto configuration = [](const auto& climber) {
    return std::make_tuple(climber.settings.seed,
                           climber.mtm_viol_samples,
                           climber.mtm_sat_samples,
                           climber.nnz_samples,
                           climber.perturb_interval);
  };

  auto expected = ordinary(reference, false);
  auto actual   = ordinary(with_lns, false);
  ASSERT_EQ(configuration(*actual), configuration(*expected));
  auto previous_seed = expected->settings.seed;

  // An optional randomized LNS climber consumes a private copy of the current stream.
  auto lns = with_lns.create_cpu_climber(solution,
                                         weights,
                                         weights,
                                         0.0,
                                         preemption,
                                         nullptr,
                                         mip::fj_settings_t{},
                                         /*randomize_params=*/true,
                                         /*preserve_rng=*/true);
  expected = ordinary(reference, true);
  actual   = ordinary(with_lns, true);
  EXPECT_EQ(configuration(*lns), configuration(*expected));
  EXPECT_EQ(configuration(*actual), configuration(*expected));
  EXPECT_NE(expected->settings.seed, previous_seed);
  previous_seed = expected->settings.seed;

  // Both randomized and ordinary creation must retain the same subsequent stream.
  for (bool randomize : {false, true}) {
    expected = ordinary(reference, randomize);
    actual   = ordinary(with_lns, randomize);
    EXPECT_EQ(configuration(*actual), configuration(*expected));
    EXPECT_NE(expected->settings.seed, previous_seed);
    previous_seed = expected->settings.seed;
  }
}

TEST(Lns, PresolvePortfolioBudgetAndLifetime)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_test_problem(op, true);
  opt::mip_solver_settings_t<int, double> settings;
  for (int team_size : {2, 6, 7, 10}) {
#pragma omp parallel num_threads(team_size)
    {
#pragma omp masked
      {
        mip::early_cpufj_t<int, double> portfolio(op, settings.get_tolerances(), {}, 42);
        for (int restart = 0; restart < 2; ++restart) {
          // Even an oversized request must honor the presolve reservation.
          portfolio.start(team_size);
          EXPECT_EQ(portfolio.lane_count(), std::max(1, team_size - 4));
          EXPECT_NO_THROW(portfolio.stop());
          EXPECT_EQ(portfolio.lane_count(), 0);
          EXPECT_NO_THROW(portfolio.stop());
        }
      }
    }
  }
  // The short initialization probe precedes the OMP team and keeps its single lane.
  mip::early_cpufj_t<int, double> probe(op, settings.get_tolerances(), {}, 42);
  probe.start(20, true);
  EXPECT_EQ(probe.lane_count(), 1);
  probe.stop();
}

TEST(Lns, PresolveIncumbentIsHandedToMainPopulationOnce)
{
  raft::handle_t handle;
  opt::optimization_problem_t<int, double> op(&handle);
  init_test_problem(op, true, 1);
  opt::mip_solver_settings_t<int, double> settings;
  mip::problem_t<int, double> problem(op, settings.get_tolerances());
  problem.preprocess_problem();
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  context.initial_incumbent_assignment = {1, 0};
  mip::diversity_manager_t<int, double> dm(context);
  std::vector<mip::solution_t<int, double>> solutions;
  dm.add_user_given_solutions(solutions);
  EXPECT_TRUE(solutions.empty());
  context.initial_incumbent_from_papilo_model = true;
  dm.add_user_given_solutions(solutions);
  ASSERT_EQ(solutions.size(), 1);
  EXPECT_TRUE(solutions[0].get_feasible());
  EXPECT_DOUBLE_EQ(solutions[0].get_user_objective(), 1);
}

TEST(Lns, BenchmarkApiErrorsAreDistinctFromNoSolution)
{
  EXPECT_NO_THROW(cuopt_bench::check_c_api(CUOPT_SUCCESS, "cuOptSolve"));
  try {
    cuopt_bench::check_c_api(CUOPT_RUNTIME_ERROR, "cuOptSolve");
    FAIL() << "API error was ignored";
  } catch (const cuopt_bench::c_api_error_t& error) {
    EXPECT_EQ(error.code, CUOPT_RUNTIME_ERROR);
    EXPECT_STREQ(error.operation, "cuOptSolve");
  }
}
TEST(Lns, PresolveBudgetsIncludePapiloAndAuxiliaryWorkers)
{
  for (int team_size = 2; team_size <= 128; ++team_size) {
    const int gpu_workers = team_size >= CUOPT_MIP_EARLY_GPUFJ_REQUIRED_THREAD_COUNT ? 1 : 0;
    for (bool recognized_structure : {false, true}) {
      const int structural_cpu_budget =
        mip::presolve_early_worker_budget(team_size, CUOPT_MIP_PAPILO_THREAD_LIMIT, gpu_workers, 1);
      const int structural_workers =
        recognized_structure &&
        mip::early_structural_has_capacity(team_size, structural_cpu_budget, gpu_workers);
      const int cpu_workers = mip::presolve_early_worker_budget(
        team_size, CUOPT_MIP_PAPILO_THREAD_LIMIT, gpu_workers, structural_workers);
      EXPECT_GE(cpu_workers, 1);
      const int papilo_workers =
        mip::papilo_thread_budget(team_size, cpu_workers, gpu_workers, structural_workers);
      EXPECT_GE(papilo_workers, 1);
      EXPECT_LE(papilo_workers, CUOPT_MIP_PAPILO_THREAD_LIMIT);
      EXPECT_LE(cpu_workers + gpu_workers + structural_workers + papilo_workers, team_size);
      if (team_size >= 9) {
        EXPECT_EQ(papilo_workers, CUOPT_MIP_PAPILO_THREAD_LIMIT);
        EXPECT_EQ(cpu_workers + gpu_workers + structural_workers + papilo_workers, team_size);
      }

      // The replacement portfolio keeps the cuOpt caller/probing reservation.
      const int reduced_structural_budget =
        mip::presolve_early_worker_budget(team_size, CUOPT_MIP_EARLY_CPUFJ_RESERVED_THREADS, 0, 1);
      const int reduced_structural_workers =
        recognized_structure &&
        mip::early_structural_has_capacity(team_size, reduced_structural_budget, 0);
      const int reduced_workers = mip::presolve_early_worker_budget(
        team_size, CUOPT_MIP_EARLY_CPUFJ_RESERVED_THREADS, 0, reduced_structural_workers);
      const int probing_workers =
        mip::probing_thread_budget(team_size, reduced_workers, reduced_structural_workers);
      EXPECT_LE(reduced_workers + reduced_structural_workers + 1, team_size);
      // At tiny team sizes the caller can execute the single probing task itself.
      EXPECT_LE(reduced_workers + reduced_structural_workers + probing_workers, team_size);
      if (team_size >= 8) {
        EXPECT_EQ(
          reduced_workers + reduced_structural_workers + CUOPT_MIP_EARLY_CPUFJ_RESERVED_THREADS,
          team_size);
      }
    }
    EXPECT_EQ(mip::papilo_thread_budget(team_size, 0, 0, 0),
              std::min(team_size, CUOPT_MIP_PAPILO_THREAD_LIMIT));
  }
  EXPECT_EQ(mip::presolve_early_worker_budget(28, 4, 1, 0), 23);
  EXPECT_EQ(mip::presolve_early_worker_budget(28, 4, 1, 1), 22);
  EXPECT_EQ(mip::presolve_early_worker_budget(28, 4, 0, 0), 24);
  EXPECT_EQ(mip::presolve_early_worker_budget(28, 4, 0, 1), 23);
  EXPECT_FALSE(mip::early_structural_has_capacity(2, 1, 0));
  EXPECT_FALSE(mip::early_structural_has_capacity(3, 1, 1));
}

TEST(Lns, MainWorkersStopWithoutBranchAndBoundOnSmallTeams)
{
  raft::handle_t handle;
  for (int team_size : {6, 7, 8, 9, 10}) {
    SCOPED_TRACE(team_size);
    opt::optimization_problem_t<int, double> op(&handle);
    init_test_problem(op, true);
    opt::mip_solver_settings_t<int, double> settings;
    settings.heuristics_only = true;
    settings.seed            = 42;
    mip::problem_t<int, double> problem(op, settings.get_tolerances());
    problem.preprocess_problem();
    mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
    std::exception_ptr task_exception;
    context.task_exception = &task_exception;
    mip::diversity_manager_t<int, double> dm(context);
    dm.population.initialize_population();
    dm.population.allocate_solutions();
    dm.population.timer = cuopt::timer_t(10.0);

#pragma omp parallel num_threads(team_size)
    {
#pragma omp masked
      {
#pragma omp taskgroup
        {
          dm.ls.start_cpufj_scratch_threads(dm.population);
          dm.ls.start_cpufj_lns_improvement_thread(dm.population);
          const int lns_workers = mip::lns_worker_count(team_size, false);
          EXPECT_EQ(dm.ls.scratch_cpu_fj_lns != nullptr, lns_workers > 0);
          // Reserve a slot for the LNS worker only on teams large enough to run it.
          for (const auto& worker : dm.ls.scratch_cpu_fj)
            EXPECT_EQ(worker != nullptr,
                      team_size - lns_workers >= CUOPT_MIP_FJ_REQUIRED_THREAD_COUNT);

          dm.ls.stop_cpufj_scratch_threads();
          bool stopped = true;
          if (dm.ls.scratch_cpu_fj_lns) {
            EXPECT_TRUE(dm.ls.scratch_cpu_fj_lns->halted.load());
            stopped &= dm.ls.scratch_cpu_fj_lns->halted.load();
          }
          for (const auto& worker : dm.ls.scratch_cpu_fj) {
            if (worker) {
              EXPECT_TRUE(worker->halted.load());
              stopped &= worker->halted.load();
            }
          }
          EXPECT_FALSE(context.preempt_heuristic_solver_.load());
          // Keep a regression from hanging the test itself. Successful cleanup
          // must not need the preemption signal normally supplied by B&B.
          if (!stopped) context.preempt_heuristic_solver_ = true;
          dm.ls.stop_cpufj_scratch_threads();
        }
      }
    }
    EXPECT_FALSE(task_exception);
  }
}

}  // namespace cuopt::lns::test
