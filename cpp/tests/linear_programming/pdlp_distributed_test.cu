/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Multi-GPU distributed PDLP parity tests.
// Binary name PDLP_MG_TEST matches the *_MG_TEST glob in ci/test_cpp_multi_gpu.sh.

#include "utilities/pdlp_test_utilities.cuh"

#include <cuopt/mathematical_optimization/constants.h>
#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_solution.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>

#include <utilities/copy_helpers.hpp>

#include <raft/core/device_setter.hpp>
#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <string>

namespace cuopt::mathematical_optimization::test {

// Solve `mps_rel_path` with the single-GPU PDLP ("base") and with distributed PDLP
// (num_gpus = -1 => auto-detect), then assert the distributed run matches the base run on
// everything meaningful: termination status, step count (within 15%), primal/dual objective,
// and the full primal/dual solution vectors. All value comparisons use a loose relative tolerance.
static void expect_distributed_matches_base(raft::handle_t const& handle,
                                            std::string const& mps_rel_path,
                                            bool fixed_mps_format = false)
{
  constexpr double loose_rel = 1e-3;
  auto near_rel              = [](double a, double b, double rel) {
    return std::fabs(a - b) <= rel * (1.0 + std::fabs(a));
  };

  auto path                                 = make_path_absolute(mps_rel_path);
  io::mps_data_model_t<int, double> problem = io::read_mps<int, double>(path, fixed_mps_format);

  pdlp_solver_settings_t<int, double> base_settings{};
  base_settings.method = method_t::PDLP;

  auto base_op = mps_data_model_to_optimization_problem<int, double>(&handle, problem);
  auto base    = solve_lp(base_op, base_settings);

  pdlp_solver_settings_t<int, double> dist_settings = base_settings;
  dist_settings.use_distributed_pdlp                = true;
  dist_settings.distributed_pdlp_num_gpus           = -1;
  auto dist                                         = solve_lp(&handle, problem, dist_settings);

  ASSERT_EQ(static_cast<int>(base.get_termination_status()), CUOPT_TERMINATION_STATUS_OPTIMAL)
    << mps_rel_path << ": base did not reach optimal";
  EXPECT_EQ(static_cast<int>(dist.get_termination_status()),
            static_cast<int>(base.get_termination_status()))
    << mps_rel_path << ": distributed termination status differs from base";

  const auto& base_info = base.get_additional_termination_information();
  const auto& dist_info = dist.get_additional_termination_information();

  EXPECT_TRUE(near_rel(base_info.primal_objective, dist_info.primal_objective, loose_rel))
    << mps_rel_path << ": primal objective base=" << base_info.primal_objective
    << " distributed=" << dist_info.primal_objective;
  EXPECT_TRUE(near_rel(base_info.dual_objective, dist_info.dual_objective, loose_rel))
    << mps_rel_path << ": dual objective base=" << base_info.dual_objective
    << " distributed=" << dist_info.dual_objective;

  const int base_steps = base_info.number_of_steps_taken;
  const int dist_steps = dist_info.number_of_steps_taken;
  const int max_steps  = std::max(base_steps, dist_steps);
  const int step_diff  = std::max(base_steps, dist_steps) - std::min(base_steps, dist_steps);
  EXPECT_LE(static_cast<double>(step_diff), 0.15 * max_steps)
    << mps_rel_path << ": step counts differ by >15% (base=" << base_steps
    << ", distributed=" << dist_steps << ")";

  auto base_primal = cuopt::host_copy(base.get_primal_solution(), handle.get_stream());
  auto dist_primal = cuopt::host_copy(dist.get_primal_solution(), handle.get_stream());
  ASSERT_EQ(base_primal.size(), dist_primal.size()) << mps_rel_path << ": primal size mismatch";
  for (std::size_t i = 0; i < base_primal.size(); ++i) {
    EXPECT_TRUE(near_rel(base_primal[i], dist_primal[i], loose_rel))
      << mps_rel_path << ": primal[" << i << "] base=" << base_primal[i]
      << " distributed=" << dist_primal[i];
  }

  auto base_dual = cuopt::host_copy(base.get_dual_solution(), handle.get_stream());
  auto dist_dual = cuopt::host_copy(dist.get_dual_solution(), handle.get_stream());
  ASSERT_EQ(base_dual.size(), dist_dual.size()) << mps_rel_path << ": dual size mismatch";
  for (std::size_t i = 0; i < base_dual.size(); ++i) {
    EXPECT_TRUE(near_rel(base_dual[i], dist_dual[i], loose_rel))
      << mps_rel_path << ": dual[" << i << "] base=" << base_dual[i]
      << " distributed=" << dist_dual[i];
  }
}

TEST(pdlp_class, distributed_parity_afiro)
{
  if (raft::device_setter::get_device_count() < 2) {
    GTEST_SKIP() << "Requires >=2 GPUs, found " << raft::device_setter::get_device_count();
  }
  const raft::handle_t handle{};
  expect_distributed_matches_base(handle, "linear_programming/afiro_original.mps", true);
}

TEST(pdlp_class, distributed_parity_neos3)
{
  if (raft::device_setter::get_device_count() < 2) {
    GTEST_SKIP() << "Requires >=2 GPUs, found " << raft::device_setter::get_device_count();
  }
  const raft::handle_t handle{};
  expect_distributed_matches_base(handle, "linear_programming/neos3/neos3.mps");
}

TEST(pdlp_class, distributed_parity_a2864)
{
  if (raft::device_setter::get_device_count() < 2) {
    GTEST_SKIP() << "Requires >=2 GPUs, found " << raft::device_setter::get_device_count();
  }
  const raft::handle_t handle{};
  expect_distributed_matches_base(handle, "linear_programming/a2864/a2864.mps");
}

}  // namespace cuopt::mathematical_optimization::test
