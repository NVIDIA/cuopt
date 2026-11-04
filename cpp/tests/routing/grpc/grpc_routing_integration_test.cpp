/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file grpc_routing_integration_test.cpp
 * @brief VRP over gRPC against a real cuopt_grpc_server
 *
 * Split out of GRPC_MATHOPT_INTEGRATION_TEST so the routing job owns the end-to-end routing
 * mappers while the LP/MIP integration suite stays with the mathopt job. Shares the
 * server fixture with it through grpc_integration_fixture.hpp.
 *
 * Environment variables:
 *   CUOPT_GRPC_SERVER_PATH - Path to cuopt_grpc_server binary
 *   CUOPT_TEST_PORT_BASE   - Base port for test servers (default: 19000)
 */

#include <cuopt/routing/cpu_routing_problem.hpp>
#include <cuopt/routing/solver_settings.hpp>

#include "utilities/grpc/grpc_integration_fixture.hpp"

#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>

using namespace grpc_integration_fixture;

namespace {

// Shared server with a small message limit and verbose logging, so the chunked-upload
// test can assert on the server's path-selection log lines.
class GrpcRoutingServerTests : public GrpcIntegrationTestBase {
 protected:
  static void SetUpTestSuite()
  {
    s_port_   = get_test_port();
    s_server_ = std::make_unique<ServerProcess>();
    ASSERT_TRUE(s_server_->start(s_port_, {"--max-message-bytes", "12288", "--verbose"}))
      << "Failed to start routing integration server";
  }

  static void TearDownTestSuite()
  {
    if (s_server_) EXPECT_TRUE(s_server_->stop());
    s_server_.reset();
  }

  void SetUp() override
  {
    ASSERT_NE(s_server_, nullptr);
    port_ = s_port_;
  }

  std::string server_log_path() const { return s_server_->log_path(); }

  static std::unique_ptr<ServerProcess> s_server_;
  static int s_port_;
};

std::unique_ptr<ServerProcess> GrpcRoutingServerTests::s_server_;
int GrpcRoutingServerTests::s_port_ = 0;

// VRP over gRPC, in the same shape as the LP and MIP cases above: submit a problem,
// poll to completion, fetch the solution and check it. This is the only routing case in
// this suite, and it is the end-to-end exercise of the routing mappers -- the problem and
// settings on the way in, the solution on the way back.
// Four locations on a line, depot at 0, one vehicle. Any correct solver serves all
// three orders, so the assertions do not depend on a particular route ordering.
cuopt::routing::cpu_routing_problem_t make_line_vrp_problem()
{
  constexpr int32_t kLocations = 4;
  cuopt::routing::cpu_routing_problem_t problem;
  problem.num_locations = kLocations;
  problem.fleet_size    = 1;
  problem.num_orders    = kLocations;

  cuopt::routing::cpu_cost_matrix_t cost;
  cost.vehicle_type = 0;
  cost.matrix.resize(static_cast<size_t>(kLocations) * kLocations);
  for (int32_t i = 0; i < kLocations; ++i) {
    for (int32_t j = 0; j < kLocations; ++j) {
      cost.matrix[static_cast<size_t>(i) * kLocations + j] = static_cast<float>(std::abs(i - j));
    }
  }
  problem.cost_matrices.push_back(std::move(cost));
  problem.vehicle_start_locations.assign(problem.fleet_size, 0);
  problem.vehicle_return_locations.assign(problem.fleet_size, 0);
  return problem;
}

void solve_vrp_and_expect_served(grpc_client_t* client,
                                 const cuopt::routing::cpu_routing_problem_t& problem)
{
  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(10.0f);

  auto submit_result = client->submit_vrp(problem, settings);
  ASSERT_TRUE(submit_result.success) << submit_result.error_message;
  EXPECT_FALSE(submit_result.job_id.empty());

  job_status_t final_status = job_status_t::QUEUED;
  for (int i = 0; i < 60; ++i) {
    auto status = client->check_status(submit_result.job_id);
    ASSERT_TRUE(status.success) << status.error_message;
    final_status = status.status;
    if (final_status == job_status_t::COMPLETED || final_status == job_status_t::FAILED ||
        final_status == job_status_t::CANCELLED) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  ASSERT_EQ(final_status, job_status_t::COMPLETED)
    << "Status: " << job_status_to_string(final_status);

  auto result = client->get_vrp_result(submit_result.job_id);
  ASSERT_TRUE(result.success) << result.error_message;

  const auto& solution = result.solution;
  EXPECT_EQ(solution.status, 0) << solution.status_message;
  EXPECT_GT(solution.vehicle_count, 0);
  EXPECT_GT(solution.total_objective_value, 0.0);
  EXPECT_FALSE(solution.route.empty());
  EXPECT_EQ(solution.route.size(), solution.truck_id.size());
  EXPECT_TRUE(solution.unserviced_nodes.empty())
    << solution.unserviced_nodes.size() << " orders left unserved";
}

TEST_F(GrpcRoutingServerTests, SolveVRP)
{
  auto client = create_client();
  ASSERT_NE(client, nullptr);
  solve_vrp_and_expect_served(client.get(), make_line_vrp_problem());
}

// Chunked upload for VRP: the cost matrix is split across container ArrayChunks, and the
// server logs must show the chunked path was used (a unary submit would still solve).
TEST_F(GrpcRoutingServerTests, ChunkedUploadVRPWithPathLogging)
{
  GrpcTestLogCapture log_capture;
  log_capture.set_server_log_path(server_log_path());
  log_capture.mark_test_start();

  grpc_client_config_t config;
  config.timeout_seconds               = 60;
  config.chunk_size_bytes              = 8;
  config.chunked_array_threshold_bytes = 0;

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);
  solve_vrp_and_expect_served(client.get(), make_line_vrp_problem());

  EXPECT_TRUE(log_capture.wait_for_server_log("[Worker] IPC path: CHUNKED", 5000))
    << "Expected CHUNKED path in server log.\nServer log:\n"
    << log_capture.get_server_logs();
  EXPECT_TRUE(log_capture.server_log_contains("FinishChunkedUpload: CHUNKED path"))
    << "Expected FinishChunkedUpload log.\nServer log:\n"
    << log_capture.get_server_logs();
}

}  // anonymous namespace

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  const int rc = RUN_ALL_TESTS();
  // Skip static destructors: gRPC / Abseil / protobuf can double-free at process exit
  // after a green run. The shared server is already stopped in TearDownTestSuite.
  std::_Exit(rc);
}
