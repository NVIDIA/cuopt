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
// (num_gpus = -1 selects all visible devices), then assert the distributed run is:
//   - optimal (same status as base),
//   - within a loose relative tolerance of base on primal/dual objective and step count.
static void expect_distributed_matches_base(raft::handle_t const& handle,
                                            std::string const& mps_rel_path,
                                            bool fixed_mps_format = false)
{
  constexpr double loose_rel = 1e-3;
  auto approx_equal          = [](double a, double b, double rel) {
    const double scale = std::max(std::fabs(a), std::fabs(b));
    return std::fabs(a - b) <= rel * (1.0 + scale);
  };

  auto path                                 = make_path_absolute(mps_rel_path);
  io::mps_data_model_t<int, double> problem = io::read_mps<int, double>(path, fixed_mps_format);

  pdlp_solver_settings_t<int, double> base_settings{};
  base_settings.method = method_t::PDLP;

  auto base_op = mps_data_model_to_optimization_problem<int, double>(&handle, problem);
  auto base    = solve_lp(base_op, base_settings);

  pdlp_solver_settings_t<int, double> dist_settings = base_settings;
  dist_settings.use_distributed_pdlp                = true;
  dist_settings.num_gpus                            = -1;
  auto dist                                         = solve_lp(&handle, problem, dist_settings);

  ASSERT_EQ(static_cast<int>(base.get_termination_status()), CUOPT_TERMINATION_STATUS_OPTIMAL)
    << mps_rel_path << ": base did not reach optimal";
  ASSERT_EQ(static_cast<int>(dist.get_termination_status()), CUOPT_TERMINATION_STATUS_OPTIMAL)
    << mps_rel_path << ": distributed did not reach optimal";

  const auto& base_info = base.get_additional_termination_information();
  const auto& dist_info = dist.get_additional_termination_information();

  EXPECT_TRUE(approx_equal(base_info.primal_objective, dist_info.primal_objective, loose_rel))
    << mps_rel_path << ": primal objective base=" << base_info.primal_objective
    << " distributed=" << dist_info.primal_objective;
  EXPECT_TRUE(approx_equal(base_info.dual_objective, dist_info.dual_objective, loose_rel))
    << mps_rel_path << ": dual objective base=" << base_info.dual_objective
    << " distributed=" << dist_info.dual_objective;

  const int base_steps = base_info.number_of_steps_taken;
  const int dist_steps = dist_info.number_of_steps_taken;
  const int max_steps  = std::max(base_steps, dist_steps);
  const int step_diff  = max_steps - std::min(base_steps, dist_steps);
  EXPECT_LE(static_cast<double>(step_diff), 0.15 * max_steps)
    << mps_rel_path << ": step counts differ by >15% (base=" << base_steps
    << ", distributed=" << dist_steps << ")";
}

struct distributed_pdlp_test_param_t {
  std::string name;
  std::string mps_path;
  bool fixed_mps_format{false};
};

// Shared fixture: skip the whole class when fewer than 2 GPUs are visible and
// provide a single per-test raft::handle_t.
class DistributedPdlpParityTest : public ::testing::TestWithParam<distributed_pdlp_test_param_t> {
 protected:
  void SetUp() override
  {
    const int device_count = raft::device_setter::get_device_count();
    if (device_count < 2) { GTEST_SKIP() << "Requires >=2 GPUs, found " << device_count; }
  }
  raft::handle_t handle{};
};

TEST_P(DistributedPdlpParityTest, matches_base)
{
  const auto& param = GetParam();
  expect_distributed_matches_base(handle, param.mps_path, param.fixed_mps_format);
}

INSTANTIATE_TEST_SUITE_P(
  distributed_pdlp,
  DistributedPdlpParityTest,
  ::testing::Values(
    distributed_pdlp_test_param_t{"afiro", "linear_programming/afiro_original.mps", true},
    distributed_pdlp_test_param_t{"cod105_max_maximization_problem", "mip/cod105_max.mps"},
    distributed_pdlp_test_param_t{"graph40_40", "linear_programming/graph40-40/graph40-40.mps"},
    distributed_pdlp_test_param_t{"ex10", "linear_programming/ex10/ex10.mps"}),
  [](const ::testing::TestParamInfo<distributed_pdlp_test_param_t>& info) {
    return info.param.name;
  });

}  // namespace cuopt::mathematical_optimization::test
