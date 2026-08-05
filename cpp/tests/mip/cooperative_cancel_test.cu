/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"

#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <cuopt/mathematical_optimization/mip/solver_solution.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_solution.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace cuopt::mathematical_optimization::test {

namespace {

using clock = std::chrono::steady_clock;

double elapsed_seconds(clock::time_point t0)
{
  return std::chrono::duration<double>(clock::now() - t0).count();
}

bool file_exists(const std::string& rel_path)
{
  return std::filesystem::exists(make_path_absolute(rel_path));
}

}  // namespace

// Mid-solve cancel on local MIP solves across several MIPLIB-style instances.
// Each instance gets a long time limit so cancel (not TL) is the expected exit.
TEST(cooperative_cancel, mip_mid_solve_cancel_loop)
{
  constexpr double long_time_limit_s = 300.0;
  constexpr double cancel_after_s    = 2.0;
  // Allow Papilo / early FJ / root LP some room after the flag flips.
  constexpr double max_total_s = 90.0;

  const std::vector<std::string> instances = {
    "mip/neos5.mps",
    "mip/gen-ip054.mps",
    "mip/swath1.mps",
    "mip/seymour1.mps",
    "mip/ns1208400.mps",
    "mip/rmatr200-p5.mps",
  };

  int ran = 0;
  for (const auto& rel : instances) {
    if (!file_exists(rel)) {
      GTEST_LOG_(WARNING) << "Skipping missing dataset " << rel;
      continue;
    }
    ++ran;

    const raft::handle_t handle{};
    auto path    = make_path_absolute(rel);
    auto problem = io::read_mps<int, double>(path, false);
    handle.sync_stream();

    mip_solver_settings_t<int, double> settings;
    settings.time_limit     = long_time_limit_s;
    settings.log_to_console = false;
    std::atomic<bool> cancel{false};
    settings.cancel_requested = &cancel;

    std::optional<mip_solution_t<int, double>> solution;
    const auto t0 = clock::now();
    std::thread worker([&] {
      solution = solve_mip(&handle, problem, settings);
      handle.sync_stream();
    });

    std::this_thread::sleep_for(std::chrono::duration<double>(cancel_after_s));
    cancel.store(true, std::memory_order_release);
    worker.join();
    const double secs = elapsed_seconds(t0);

    ASSERT_TRUE(solution.has_value()) << rel << " produced no solution object";
    EXPECT_GE(secs, cancel_after_s * 0.9) << rel << " finished before cancel was set";
    EXPECT_LT(secs, max_total_s) << rel << " did not unwind promptly after cancel (" << secs
                                 << "s)";
    EXPECT_EQ(solution->get_termination_status(), mip_termination_status_t::Cancelled)
      << rel << " status=" << static_cast<int>(solution->get_termination_status()) << " after "
      << secs << "s";
  }
  ASSERT_GE(ran, 3) << "Need at least a few MIP datasets under RAPIDS_DATASET_ROOT_DIR";
}

// Pre-armed cancel should not burn the wall-clock budget.
TEST(cooperative_cancel, mip_preset_cancel_exits_early)
{
  constexpr double long_time_limit_s = 300.0;

  const std::string rel = "mip/seymour1.mps";
  if (!file_exists(rel)) { GTEST_SKIP() << "Missing " << rel; }

  const raft::handle_t handle{};
  auto problem = io::read_mps<int, double>(make_path_absolute(rel), false);
  handle.sync_stream();

  mip_solver_settings_t<int, double> settings;
  settings.time_limit     = long_time_limit_s;
  settings.log_to_console = false;
  std::atomic<bool> cancel{true};
  settings.cancel_requested = &cancel;

  const auto t0 = clock::now();
  auto solution = solve_mip(&handle, problem, settings);
  handle.sync_stream();
  const double secs = elapsed_seconds(t0);

  EXPECT_LT(secs, 60.0) << "pre-set cancel should exit early, took " << secs << "s";
  EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Cancelled);
}

// Local LP cancel on RAPIDS datasets that stay busy under a long time limit.
// Easy Optinals (savsched1/ex10/...) finish before cancel and are omitted.
TEST(cooperative_cancel, lp_mid_solve_cancel_loop)
{
  constexpr double long_time_limit_s = 300.0;
  constexpr double cancel_after_s    = 2.0;
  constexpr double max_total_s       = 90.0;

  const std::vector<std::string> instances = {
    "linear_programming/scpm1/scpm1.mps",
  };

  int ran = 0;
  for (const auto& rel : instances) {
    if (!file_exists(rel)) {
      GTEST_LOG_(WARNING) << "Skipping missing LP " << rel;
      continue;
    }
    ++ran;

    const raft::handle_t handle{};
    auto problem = io::read_mps<int, double>(make_path_absolute(rel), false);
    handle.sync_stream();

    pdlp_solver_settings_t<int, double> settings;
    settings.time_limit     = long_time_limit_s;
    settings.log_to_console = false;
    std::atomic<bool> cancel{false};
    settings.cancel_requested = &cancel;

    std::optional<optimization_problem_solution_t<int, double>> solution;
    const auto t0 = clock::now();
    std::thread worker([&] {
      solution = solve_lp(&handle, problem, settings);
      handle.sync_stream();
    });

    std::this_thread::sleep_for(std::chrono::duration<double>(cancel_after_s));
    cancel.store(true, std::memory_order_release);
    worker.join();
    const double secs = elapsed_seconds(t0);

    ASSERT_TRUE(solution.has_value()) << rel << " produced no solution object";
    EXPECT_GE(secs, cancel_after_s * 0.9) << rel << " finished before cancel was set";
    EXPECT_LT(secs, max_total_s) << rel << " did not unwind promptly after cancel (" << secs
                                 << "s)";
    EXPECT_EQ(solution->get_termination_status(), pdlp_termination_status_t::Cancelled)
      << rel << " status=" << static_cast<int>(solution->get_termination_status()) << " after "
      << secs << "s";
  }
  if (ran == 0) { GTEST_SKIP() << "No hard LP datasets found under RAPIDS_DATASET_ROOT_DIR"; }
  ASSERT_GE(ran, 1) << "Need at least one hard LP for cancel coverage";
}

}  // namespace cuopt::mathematical_optimization::test
