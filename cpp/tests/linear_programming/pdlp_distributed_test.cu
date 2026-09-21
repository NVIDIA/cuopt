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
#include <cuopt/mathematical_optimization/cuopt_c.h>
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

// Same instances through the public C API: cuOptReadProblem materializes a GPU
// optimization_problem_t, then cuOptSolve must dispatch to distributed PDLP
// when method=PDLP and num_gpus=-1.
struct c_api_lp_result_t {
  cuopt_int_t solve_status{CUOPT_RUNTIME_ERROR};
  cuopt_int_t termination{0};
  cuopt_float_t primal_objective{0.0};
  cuopt_float_t dual_objective{0.0};
  std::string error;
};

static c_api_lp_result_t solve_via_c_api(std::string const& mps_path, cuopt_int_t num_gpus)
{
  c_api_lp_result_t out;
  cuOptOptimizationProblem problem = nullptr;
  cuOptSolverSettings settings     = nullptr;
  cuOptSolution solution           = nullptr;

  auto cleanup = [&]() {
    cuOptDestroySolution(&solution);
    cuOptDestroySolverSettings(&settings);
    cuOptDestroyProblem(&problem);
  };

  if (cuOptReadProblem(mps_path.c_str(), &problem) != CUOPT_SUCCESS) {
    cleanup();
    return out;
  }
  if (cuOptCreateSolverSettings(&settings) != CUOPT_SUCCESS) {
    cleanup();
    return out;
  }
  if (cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_PDLP) != CUOPT_SUCCESS ||
      cuOptSetIntegerParameter(settings, CUOPT_NUM_GPUS, num_gpus) != CUOPT_SUCCESS) {
    cleanup();
    return out;
  }

  out.solve_status = cuOptSolve(problem, settings, &solution);
  if (solution != nullptr) {
    char err[512] = {};
    cuOptGetErrorString(solution, err, sizeof(err));
    out.error = err;
    if (out.solve_status == CUOPT_SUCCESS) {
      cuOptGetTerminationStatus(solution, &out.termination);
      cuOptGetObjectiveValue(solution, &out.primal_objective);
      cuOptGetDualObjectiveValue(solution, &out.dual_objective);
    }
  }
  cleanup();
  return out;
}

static void expect_c_api_distributed_matches_single_gpu(std::string const& mps_rel_path)
{
  constexpr double loose_rel = 1e-3;
  auto approx_equal          = [](double a, double b, double rel) {
    const double scale = std::max(std::fabs(a), std::fabs(b));
    return std::fabs(a - b) <= rel * (1.0 + scale);
  };

  auto path = make_path_absolute(mps_rel_path);
  auto base = solve_via_c_api(path, /*num_gpus=*/1);
  auto dist = solve_via_c_api(path, /*num_gpus=*/-1);

  ASSERT_EQ(base.solve_status, CUOPT_SUCCESS)
    << mps_rel_path << ": C API single-GPU solve failed: " << base.error;
  ASSERT_EQ(dist.solve_status, CUOPT_SUCCESS)
    << mps_rel_path << ": C API distributed solve failed (num_gpus=-1): " << dist.error;
  ASSERT_EQ(base.termination, CUOPT_TERMINATION_STATUS_OPTIMAL)
    << mps_rel_path << ": C API single-GPU did not reach optimal";
  ASSERT_EQ(dist.termination, CUOPT_TERMINATION_STATUS_OPTIMAL)
    << mps_rel_path << ": C API distributed did not reach optimal";
  EXPECT_TRUE(approx_equal(base.primal_objective, dist.primal_objective, loose_rel))
    << mps_rel_path << ": primal objective base=" << base.primal_objective
    << " distributed=" << dist.primal_objective;
  EXPECT_TRUE(approx_equal(base.dual_objective, dist.dual_objective, loose_rel))
    << mps_rel_path << ": dual objective base=" << base.dual_objective
    << " distributed=" << dist.dual_objective;
}

class DistributedPdlpCApiTest : public ::testing::TestWithParam<distributed_pdlp_test_param_t> {
 protected:
  void SetUp() override
  {
    const int device_count = raft::device_setter::get_device_count();
    if (device_count < 2) { GTEST_SKIP() << "Requires >=2 GPUs, found " << device_count; }
  }
};

TEST_P(DistributedPdlpCApiTest, matches_single_gpu)
{
  expect_c_api_distributed_matches_single_gpu(GetParam().mps_path);
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

INSTANTIATE_TEST_SUITE_P(
  distributed_pdlp_c_api,
  DistributedPdlpCApiTest,
  ::testing::Values(
    distributed_pdlp_test_param_t{"afiro", "linear_programming/afiro_original.mps", true},
    distributed_pdlp_test_param_t{"graph40_40", "linear_programming/graph40-40/graph40-40.mps"},
    distributed_pdlp_test_param_t{"ex10", "linear_programming/ex10/ex10.mps"}),
  [](const ::testing::TestParamInfo<distributed_pdlp_test_param_t>& info) {
    return info.param.name;
  });

}  // namespace cuopt::mathematical_optimization::test
