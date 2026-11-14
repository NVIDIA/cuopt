/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/mathematical_optimization/constants.h>
#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <cuopt/mathematical_optimization/utilities/internals.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/error.hpp>
#include <utilities/scope_guard.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace cuopt::mathematical_optimization::test {

class scoped_env_restore_t {
 public:
  scoped_env_restore_t(const char* env_name, const char* new_value) : name_(env_name)
  {
    if (const char* prev = std::getenv(env_name)) { prev_value_ = prev; }
    ::setenv(env_name, new_value, 1);
  }
  ~scoped_env_restore_t() { ::setenv(name_, prev_value_.c_str(), 1); }
  scoped_env_restore_t(const scoped_env_restore_t&)            = delete;
  scoped_env_restore_t& operator=(const scoped_env_restore_t&) = delete;

 private:
  const char* name_;
  std::string prev_value_;
};

class test_set_solution_callback_t : public cuopt::internals::set_solution_callback_t {
 public:
  test_set_solution_callback_t(std::vector<std::pair<std::vector<double>, double>>& solutions_,
                               void* expected_user_data_)
    : solutions(solutions_), expected_user_data(expected_user_data_), n_calls(0)
  {
  }
  // This will check that the we are able to recompute our own solution
  void set_solution(void* data, void* cost, void* solution_bound, void* user_data) override
  {
    EXPECT_EQ(user_data, expected_user_data);
    n_calls++;
    auto bound_ptr = static_cast<double*>(solution_bound);
    EXPECT_FALSE(std::isnan(bound_ptr[0]));
    auto assignment = static_cast<double*>(data);
    auto cost_ptr   = static_cast<double*>(cost);
    if (solutions.empty()) { return; }

    auto const& [last_assignment, last_cost] = solutions.back();
    std::copy(last_assignment.begin(), last_assignment.end(), assignment);
    *cost_ptr = last_cost;
  }
  std::vector<std::pair<std::vector<double>, double>>& solutions;
  void* expected_user_data;
  int n_calls;
};

class test_get_solution_callback_t : public cuopt::internals::get_solution_callback_t {
 public:
  test_get_solution_callback_t(std::vector<std::pair<std::vector<double>, double>>& solutions_in,
                               int n_variables_,
                               void* expected_user_data_)
    : solutions(solutions_in),
      expected_user_data(expected_user_data_),
      n_calls(0),
      n_variables(n_variables_)
  {
  }
  void get_solution(void* data, void* cost, void* solution_bound, void* user_data) override
  {
    EXPECT_EQ(user_data, expected_user_data);
    n_calls++;
    auto bound_ptr = static_cast<double*>(solution_bound);
    EXPECT_FALSE(std::isnan(bound_ptr[0]));
    auto assignment_ptr = static_cast<double*>(data);
    auto cost_ptr       = static_cast<double*>(cost);
    std::vector<double> assignment(assignment_ptr, assignment_ptr + n_variables);
    solutions.push_back(std::make_pair(std::move(assignment), *cost_ptr));
  }
  std::vector<std::pair<std::vector<double>, double>>& solutions;
  void* expected_user_data;
  int n_calls;
  int n_variables;
};

void check_solutions(
  const test_get_solution_callback_t& get_solution_callback,
  const cuopt::mathematical_optimization::io::mps_data_model_t<int, double>& op_problem,
  const cuopt::mathematical_optimization::mip_solver_settings_t<int, double>& settings)
{
  for (const auto& solution : get_solution_callback.solutions) {
    EXPECT_EQ(solution.first.size(), op_problem.get_variable_lower_bounds().size());
    test_variable_bounds(op_problem, solution.first, settings);
    const double unscaled_acceptable_tol = 0.1;
    test_constraint_sanity_per_row(
      op_problem,
      solution.first,
      // because of scaling the values are not as accurate, so add more relative tolerance
      unscaled_acceptable_tol,
      settings.tolerances.relative_tolerance);
    test_objective_sanity(op_problem, solution.first, solution.second, 1e-4);
  }
}

void test_incumbent_callback(std::string test_instance, bool include_set_callback)
{
  const raft::handle_t handle_{};
  std::cout << "Running: " << test_instance << std::endl;
  auto path = make_path_absolute(test_instance);
  cuopt::mathematical_optimization::io::mps_data_model_t<int, double> mps_problem =
    cuopt::mathematical_optimization::io::read_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);

  auto settings          = mip_solver_settings_t<int, double>{};
  settings.time_limit    = 30.;
  settings.presolver     = presolver_t::Papilo;
  settings.presolve_file = (std::filesystem::temp_directory_path() /
                            ("cuopt_callback_presolve_" + std::to_string(::getpid()) + ".mps"))
                             .string();
  cuopt::scope_guard remove_presolve_file([&]() {
    std::error_code error;
    std::filesystem::remove(settings.presolve_file, error);
  });
  int user_data = 42;
  std::vector<std::pair<std::vector<double>, double>> solutions;
  test_get_solution_callback_t get_solution_callback(
    solutions, op_problem.get_n_variables(), &user_data);
  settings.set_mip_callback(&get_solution_callback, &user_data);
  std::unique_ptr<test_set_solution_callback_t> set_solution_callback;
  if (include_set_callback) {
    set_solution_callback = std::make_unique<test_set_solution_callback_t>(solutions, &user_data);
    settings.set_mip_callback(set_solution_callback.get(), &user_data);
  }
  auto solution = solve_mip(op_problem, settings);
  ASSERT_TRUE(std::filesystem::exists(settings.presolve_file));
  EXPECT_GT(std::filesystem::file_size(settings.presolve_file), 0);
  EXPECT_GE(get_solution_callback.n_calls, 1);
  if (include_set_callback) { EXPECT_GE(set_solution_callback->n_calls, 1); }
  check_solutions(get_solution_callback, mps_problem, settings);
}

TEST(mip_solve, incumbent_get_callback_test)
{
  std::vector<std::string> test_instances = {
    "mip/50v-10.mps", "mip/neos5-free-bound.mps", "mip/swath1.mps"};
  for (const auto& test_instance : test_instances) {
    test_incumbent_callback(test_instance, false);
  }
}

TEST(mip_solve, incumbent_get_set_callback_test)
{
  std::vector<std::string> test_instances = {
    "mip/50v-10.mps", "mip/neos5-free-bound.mps", "mip/swath1.mps"};
  for (const auto& test_instance : test_instances) {
    test_incumbent_callback(test_instance, true);
  }
}

// Verify that when only early heuristics find a feasible incumbent but the solver-space
// pipeline (B&B + GPU heuristics) does not, the solver still returns that incumbent.
// B&B runs but exits immediately (node_limit=0); GPU heuristics are disabled so the
// population stays empty. The fallback in solver.cu must use the OG-space incumbent.
TEST(mip_solve, early_heuristic_incumbent_fallback)
{
  scoped_env_restore_t disable_gpu_heuristics_env("CUOPT_DISABLE_GPU_HEURISTICS", "1");

  const raft::handle_t handle_{};
  auto path = make_path_absolute("mip/pk1.mps");
  cuopt::mathematical_optimization::io::mps_data_model_t<int, double> mps_problem =
    cuopt::mathematical_optimization::io::read_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);

  auto settings       = mip_solver_settings_t<int, double>{};
  settings.time_limit = 10.;
  settings.presolver  = presolver_t::Papilo;
  settings.node_limit = 0;

  int user_data = 0;
  std::vector<std::pair<std::vector<double>, double>> callback_solutions;
  test_get_solution_callback_t get_cb(callback_solutions, op_problem.get_n_variables(), &user_data);
  settings.set_mip_callback(&get_cb, &user_data);

  auto solution = solve_mip(op_problem, settings);

  EXPECT_GE(get_cb.n_calls, 1) << "Early heuristics should have emitted at least one incumbent";
  auto status = solution.get_termination_status();
  EXPECT_TRUE(status == mip_termination_status_t::FeasibleFound ||
              status == mip_termination_status_t::Optimal)
    << "Expected feasible result, got "
    << mip_solution_t<int, double>::get_termination_status_string(status);
  EXPECT_TRUE(std::isfinite(solution.get_objective_value()));

  if (!callback_solutions.empty()) { check_solutions(get_cb, mps_problem, settings); }
}

// Verify that a user-provided MIP start in original space is correctly crushed
// through PaPILO presolve and accepted into the heuristic population.
TEST(mip_solve, initial_solution_survives_papilo_crush)
{
  scoped_env_restore_t disable_gpu_heuristics_env("CUOPT_DISABLE_GPU_HEURISTICS", "1");

  const raft::handle_t handle_{};
  auto path = make_path_absolute("mip/pk1.mps");
  cuopt::mathematical_optimization::io::mps_data_model_t<int, double> mps_problem =
    cuopt::mathematical_optimization::io::read_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);
  auto stream     = op_problem.get_handle_ptr()->get_stream();

  // Step 1: solve to get a reference feasible solution. Pkl is easily solved to optimality
  auto settings1       = mip_solver_settings_t<int, double>{};
  settings1.time_limit = 5.;
  settings1.presolver  = presolver_t::Papilo;
  auto result1         = solve_mip(op_problem, settings1);
  auto status1         = result1.get_termination_status();
  ASSERT_TRUE(status1 == mip_termination_status_t::FeasibleFound ||
              status1 == mip_termination_status_t::Optimal)
    << "Reference solve must find a feasible solution";
  auto reference_obj      = result1.get_objective_value();
  auto reference_solution = cuopt::host_copy(result1.get_solution(), stream);
  ASSERT_EQ((int)reference_solution.size(), op_problem.get_n_variables());

  // Step 2: feed the reference solution as a MIP start with presolve ON
  // and GPU heuristics disabled. B&B runs with node_limit=0 so it exits
  // immediately. The only way we get a good objective is if the MIP start
  // was crushed through PaPILO and accepted by add_user_given_solutions.
  // Early FJ is not strong enough to find the 11 optimal before the node limit stops the solve.
  auto settings2       = mip_solver_settings_t<int, double>{};
  settings2.presolver  = presolver_t::Papilo;
  settings2.node_limit = 0;
  settings2.add_initial_solution(reference_solution.data(), reference_solution.size(), stream);

  int user_data = 0;
  std::vector<std::pair<std::vector<double>, double>> callback_solutions;
  test_get_solution_callback_t get_cb(callback_solutions, op_problem.get_n_variables(), &user_data);
  settings2.set_mip_callback(&get_cb, &user_data);

  auto result2 = solve_mip(op_problem, settings2);

  auto status2 = result2.get_termination_status();
  EXPECT_TRUE(status2 == mip_termination_status_t::FeasibleFound ||
              status2 == mip_termination_status_t::Optimal)
    << "Crushed MIP start should yield a feasible result, got "
    << mip_solution_t<int, double>::get_termination_status_string(status2);
  EXPECT_TRUE(std::isfinite(result2.get_objective_value()));
  // Heuristics may improve on the MIP start when step 1 stops before optimality.
  EXPECT_LE(result2.get_objective_value(), reference_obj + 1e-4);

  if (!callback_solutions.empty()) { check_solutions(get_cb, mps_problem, settings2); }
}

// Solves with a time limit that expires before the main solve, so the MIP starts never reach the
// population there.
static mip_solution_t<int, double> solve_with_early_time_limit(
  const raft::handle_t& handle,
  const cuopt::mathematical_optimization::io::mps_data_model_t<int, double>& mps_problem,
  const std::vector<std::vector<double>>& initial_solutions,
  presolver_t presolver,
  int determinism_mode,
  test_get_solution_callback_t* get_solution_callback = nullptr)
{
  auto op_problem           = mps_data_model_to_optimization_problem(&handle, mps_problem);
  auto settings             = mip_solver_settings_t<int, double>{};
  settings.time_limit       = 1e-3;
  settings.presolver        = presolver;
  settings.determinism_mode = determinism_mode;
  if (get_solution_callback != nullptr) {
    settings.set_mip_callback(get_solution_callback, get_solution_callback->expected_user_data);
  }
  for (const auto& initial_solution : initial_solutions) {
    settings.add_initial_solution(
      initial_solution.data(), initial_solution.size(), handle.get_stream());
  }
  return solve_mip(op_problem, settings);
}

// Verify that the best feasible user-provided MIP start is returned when the time limit is
// reached before the main solve, and that an infeasible or non-finite one is not.
TEST(mip_solve, initial_solution_returned_on_early_time_limit)
{
  const raft::handle_t handle_{};
  auto path = make_path_absolute("mip/pk1.mps");
  cuopt::mathematical_optimization::io::mps_data_model_t<int, double> mps_problem =
    cuopt::mathematical_optimization::io::read_mps<int, double>(path, false);
  handle_.sync_stream();

  auto op_problem     = mps_data_model_to_optimization_problem(&handle_, mps_problem);
  auto settings       = mip_solver_settings_t<int, double>{};
  settings.time_limit = 5.;
  auto reference      = solve_mip(op_problem, settings);
  auto status         = reference.get_termination_status();
  ASSERT_TRUE(status == mip_termination_status_t::FeasibleFound ||
              status == mip_termination_status_t::Optimal)
    << "Reference solve must find a feasible solution";
  auto reference_obj      = reference.get_objective_value();
  auto reference_solution = cuopt::host_copy(reference.get_solution(), handle_.get_stream());
  ASSERT_EQ((int)reference_solution.size(), op_problem.get_n_variables());

  // pk1's objective is the continuous variable x1, which has no upper bound and appears only with
  // coefficient +1 in >= rows, so raising it gives a feasible start with a worse objective.
  const auto& variable_names = mps_problem.get_variable_names();
  auto objective_var         = std::find(variable_names.begin(), variable_names.end(), "x1");
  ASSERT_NE(objective_var, variable_names.end());
  constexpr double worse_by = 10.;
  auto worse_solution       = reference_solution;
  worse_solution[objective_var - variable_names.begin()] += worse_by;

  const auto& variable_types = mps_problem.get_variable_types();
  auto integer_var           = std::find(variable_types.begin(), variable_types.end(), 'I');
  ASSERT_NE(integer_var, variable_types.end());
  auto infeasible_solution                                  = reference_solution;
  infeasible_solution[integer_var - variable_types.begin()] = 0.5;
  // x1 has no upper bound, so an infinite value passes the bound validation.
  auto nonfinite_solution = reference_solution;
  nonfinite_solution[objective_var - variable_names.begin()] =
    std::numeric_limits<double>::infinity();
  auto nan_solution                                    = reference_solution;
  nan_solution[objective_var - variable_names.begin()] = std::numeric_limits<double>::quiet_NaN();

  // Deterministic mode disables the pre-solve probe and the early heuristics, so the MIP starts
  // are the only feasible candidates and the outcome is exact.
  for (auto presolver : {presolver_t::Papilo, presolver_t::None}) {
    SCOPED_TRACE(presolver == presolver_t::Papilo ? "Papilo presolve" : "no presolve");

    int user_data = 0;
    std::vector<std::pair<std::vector<double>, double>> callback_solutions;
    test_get_solution_callback_t get_cb(
      callback_solutions, op_problem.get_n_variables(), &user_data);
    auto result =
      solve_with_early_time_limit(handle_,
                                  mps_problem,
                                  {infeasible_solution, worse_solution, reference_solution},
                                  presolver,
                                  CUOPT_MODE_DETERMINISTIC,
                                  &get_cb);
    ASSERT_EQ(result.get_termination_status(), mip_termination_status_t::FeasibleFound)
      << mip_solution_t<int, double>::get_termination_status_string(
           result.get_termination_status());
    EXPECT_NEAR(result.get_objective_value(), reference_obj, 1e-6);
    auto returned_solution = cuopt::host_copy(result.get_solution(), handle_.get_stream());
    ASSERT_EQ(returned_solution.size(), reference_solution.size());
    for (size_t i = 0; i < returned_solution.size(); ++i) {
      EXPECT_NEAR(returned_solution[i], reference_solution[i], 1e-9) << "variable " << i;
    }
    // Only the returned MIP start is reported to the callback.
    ASSERT_EQ(callback_solutions.size(), 1u);
    EXPECT_NEAR(callback_solutions[0].second, reference_obj, 1e-6);
    check_solutions(get_cb, mps_problem, mip_solver_settings_t<int, double>{});

    // The better start wins regardless of its position.
    auto best_first = solve_with_early_time_limit(handle_,
                                                  mps_problem,
                                                  {reference_solution, worse_solution},
                                                  presolver,
                                                  CUOPT_MODE_DETERMINISTIC);
    ASSERT_EQ(best_first.get_termination_status(), mip_termination_status_t::FeasibleFound)
      << mip_solution_t<int, double>::get_termination_status_string(
           best_first.get_termination_status());
    EXPECT_NEAR(best_first.get_objective_value(), reference_obj, 1e-6);

    auto worse_only = solve_with_early_time_limit(
      handle_, mps_problem, {worse_solution}, presolver, CUOPT_MODE_DETERMINISTIC);
    ASSERT_EQ(worse_only.get_termination_status(), mip_termination_status_t::FeasibleFound)
      << mip_solution_t<int, double>::get_termination_status_string(
           worse_only.get_termination_status());
    EXPECT_NEAR(worse_only.get_objective_value(), reference_obj + worse_by, 1e-6);

    auto infeasible_only = solve_with_early_time_limit(
      handle_, mps_problem, {infeasible_solution}, presolver, CUOPT_MODE_DETERMINISTIC);
    EXPECT_EQ(infeasible_only.get_termination_status(), mip_termination_status_t::TimeLimit)
      << mip_solution_t<int, double>::get_termination_status_string(
           infeasible_only.get_termination_status());

    auto nonfinite_only = solve_with_early_time_limit(
      handle_, mps_problem, {nonfinite_solution}, presolver, CUOPT_MODE_DETERMINISTIC);
    EXPECT_EQ(nonfinite_only.get_termination_status(), mip_termination_status_t::TimeLimit)
      << mip_solution_t<int, double>::get_termination_status_string(
           nonfinite_only.get_termination_status());

    auto nan_only = solve_with_early_time_limit(
      handle_, mps_problem, {nan_solution}, presolver, CUOPT_MODE_DETERMINISTIC);
    EXPECT_EQ(nan_only.get_termination_status(), mip_termination_status_t::TimeLimit)
      << mip_solution_t<int, double>::get_termination_status_string(
           nan_only.get_termination_status());
  }

  // With the early heuristics running, the MIP start has to win over their incumbent.
  auto result = solve_with_early_time_limit(
    handle_, mps_problem, {reference_solution}, presolver_t::Papilo, CUOPT_MODE_OPPORTUNISTIC);
  ASSERT_EQ(result.get_termination_status(), mip_termination_status_t::FeasibleFound)
    << mip_solution_t<int, double>::get_termination_status_string(result.get_termination_status());
  EXPECT_LE(result.get_objective_value(), reference_obj + 1e-6);
}

}  // namespace cuopt::mathematical_optimization::test
