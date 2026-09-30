/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Multi-GPU PDLP gRPC workflow test. Binary name GRPC_MG_TEST matches the *_MG_TEST
// glob in ci/test_cpp_multi_gpu.sh.

#include "grpc_integration_test_fixture.hpp"

#include <raft/core/device_setter.hpp>

#include <cmath>

using namespace cuopt::mathematical_optimization;

namespace {

class MultiGpuServerTests : public GrpcIntegrationTestBase {
 protected:
  static void SetUpTestSuite()
  {
    if (raft::device_setter::get_device_count() < 2) { return; }
    s_port_   = get_test_port();
    s_server_ = std::make_unique<ServerProcess>();
    ASSERT_TRUE(s_server_->start(s_port_, {}))
      << "Failed to start shared multi-GPU test server on port " << s_port_;
  }

  static void TearDownTestSuite()
  {
    if (s_server_) EXPECT_TRUE(s_server_->stop());
    s_server_.reset();
  }

  void SetUp() override
  {
    if (raft::device_setter::get_device_count() < 2) { GTEST_SKIP() << "Requires >=2 GPUs"; }
    ASSERT_NE(s_server_, nullptr) << "Shared server not running";
    port_ = s_port_;
  }

  static std::unique_ptr<ServerProcess> s_server_;
  static int s_port_;
};

std::unique_ptr<ServerProcess> MultiGpuServerTests::s_server_;
int MultiGpuServerTests::s_port_ = 0;

}  // namespace

// Exercises grpc_worker.cpp's is_multigpu_pdlp_requested routing end to end, against a live
// server: a method=PDLP, num_gpus=-1 job must dispatch through the mps_data_model_t solve_lp
// overload and match a single-GPU baseline submitted the same way, mirroring the parity check
// in pdlp_distributed_test.cu (PDLP_MG_TEST) but through the gRPC path instead of calling
// solve_lp directly.
TEST_F(MultiGpuServerTests, SolveLPMultiGpuPDLPMatchesBaseline)
{
  constexpr double loose_rel = 1e-3;
  auto approx_equal          = [](double a, double b, double rel) {
    const double scale = std::max(std::fabs(a), std::fabs(b));
    return std::fabs(a - b) <= rel * (1.0 + scale);
  };

  auto client = create_client();
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_lp_path("ex10/ex10.mps");
  auto problem         = load_problem_from_file(mps_path);

  auto submit_and_wait = [&](int num_gpus) {
    pdlp_solver_settings_t<int32_t, double> settings;
    settings.method   = method_t::PDLP;
    settings.num_gpus = num_gpus;

    auto submit_result = client->submit_lp(problem, settings);
    EXPECT_TRUE(submit_result.success) << submit_result.error_message;

    wait_for_job_done(client.get(), submit_result.job_id, 120);
    auto status = client->check_status(submit_result.job_id);
    EXPECT_EQ(status.status, job_status_t::COMPLETED)
      << "num_gpus=" << num_gpus << " status: " << job_status_to_string(status.status);

    return client->get_lp_result<int32_t, double>(submit_result.job_id);
  };

  auto base = submit_and_wait(1);
  ASSERT_TRUE(base.success) << base.error_message;
  ASSERT_NE(base.solution, nullptr);

  auto multi_gpu = submit_and_wait(-1);
  ASSERT_TRUE(multi_gpu.success) << multi_gpu.error_message;
  ASSERT_NE(multi_gpu.solution, nullptr);

  EXPECT_TRUE(approx_equal(
    base.solution->get_objective_value(), multi_gpu.solution->get_objective_value(), loose_rel))
    << "base=" << base.solution->get_objective_value()
    << " multi_gpu=" << multi_gpu.solution->get_objective_value();
}
