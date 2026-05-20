/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file grpc_client_test.cpp
 * @brief Unit tests for grpc_client_t using mock stubs
 *
 * These tests verify client-side error handling without requiring a real server.
 * For integration tests with a real server, see grpc_integration_test.cpp.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "grpc_client_test_helper.hpp"

#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/optimization_problem_utils.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include "grpc_client.hpp"
#include "grpc_problem_mapper.hpp"
#include "grpc_service_mapper.hpp"
#include "grpc_settings_mapper.hpp"
#include "grpc_solution_mapper.hpp"

#include <cuopt_remote.pb.h>
#include <cuopt_remote_service.grpc.pb.h>
#include <cuopt_remote_service.pb.h>
#include <grpcpp/grpcpp.h>

#include <map>

using namespace cuopt::linear_programming;
using namespace ::testing;

/**
 * @brief Mock stub for CuOptRemoteService
 *
 * This mock allows us to control exactly what the "server" returns
 * without running an actual server.
 */
class MockCuOptStub : public cuopt::remote::CuOptRemoteService::StubInterface {
 public:
  // Unary RPCs
  MOCK_METHOD(grpc::Status,
              SubmitJob,
              (grpc::ClientContext*,
               const cuopt::remote::SubmitJobRequest&,
               cuopt::remote::SubmitJobResponse*),
              (override));

  MOCK_METHOD(grpc::Status,
              CheckStatus,
              (grpc::ClientContext*,
               const cuopt::remote::StatusRequest&,
               cuopt::remote::StatusResponse*),
              (override));

  MOCK_METHOD(grpc::Status,
              GetResult,
              (grpc::ClientContext*,
               const cuopt::remote::GetResultRequest&,
               cuopt::remote::ResultResponse*),
              (override));

  MOCK_METHOD(grpc::Status,
              DeleteResult,
              (grpc::ClientContext*,
               const cuopt::remote::DeleteRequest&,
               cuopt::remote::DeleteResponse*),
              (override));

  MOCK_METHOD(grpc::Status,
              CancelJob,
              (grpc::ClientContext*,
               const cuopt::remote::CancelRequest&,
               cuopt::remote::CancelResponse*),
              (override));

  MOCK_METHOD(grpc::Status,
              WaitForCompletion,
              (grpc::ClientContext*,
               const cuopt::remote::WaitRequest&,
               cuopt::remote::WaitResponse*),
              (override));

  MOCK_METHOD(grpc::Status,
              GetIncumbents,
              (grpc::ClientContext*,
               const cuopt::remote::IncumbentRequest&,
               cuopt::remote::IncumbentResponse*),
              (override));

  // Streaming RPCs - these need special handling
  // Chunked result download RPCs
  MOCK_METHOD(grpc::Status,
              StartChunkedDownload,
              (grpc::ClientContext*,
               const cuopt::remote::StartChunkedDownloadRequest&,
               cuopt::remote::StartChunkedDownloadResponse*),
              (override));

  MOCK_METHOD(grpc::Status,
              GetResultChunk,
              (grpc::ClientContext*,
               const cuopt::remote::GetResultChunkRequest&,
               cuopt::remote::GetResultChunkResponse*),
              (override));

  MOCK_METHOD(grpc::Status,
              FinishChunkedDownload,
              (grpc::ClientContext*,
               const cuopt::remote::FinishChunkedDownloadRequest&,
               cuopt::remote::FinishChunkedDownloadResponse*),
              (override));

  MOCK_METHOD(grpc::ClientReaderInterface<cuopt::remote::LogMessage>*,
              StreamLogsRaw,
              (grpc::ClientContext*, const cuopt::remote::StreamLogsRequest&),
              (override));

  // Chunked upload RPCs
  MOCK_METHOD(grpc::Status,
              StartChunkedUpload,
              (grpc::ClientContext*,
               const cuopt::remote::StartChunkedUploadRequest&,
               cuopt::remote::StartChunkedUploadResponse*),
              (override));

  MOCK_METHOD(grpc::Status,
              SendArrayChunk,
              (grpc::ClientContext*,
               const cuopt::remote::SendArrayChunkRequest&,
               cuopt::remote::SendArrayChunkResponse*),
              (override));

  MOCK_METHOD(grpc::Status,
              FinishChunkedUpload,
              (grpc::ClientContext*,
               const cuopt::remote::FinishChunkedUploadRequest&,
               cuopt::remote::SubmitJobResponse*),
              (override));

  // Required by interface - async versions (not used in our client but required for interface)
  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::SubmitJobResponse>*,
              AsyncSubmitJobRaw,
              (grpc::ClientContext*,
               const cuopt::remote::SubmitJobRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::SubmitJobResponse>*,
              PrepareAsyncSubmitJobRaw,
              (grpc::ClientContext*,
               const cuopt::remote::SubmitJobRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::StatusResponse>*,
              AsyncCheckStatusRaw,
              (grpc::ClientContext*, const cuopt::remote::StatusRequest&, grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::StatusResponse>*,
              PrepareAsyncCheckStatusRaw,
              (grpc::ClientContext*, const cuopt::remote::StatusRequest&, grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::ResultResponse>*,
              AsyncGetResultRaw,
              (grpc::ClientContext*,
               const cuopt::remote::GetResultRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::ResultResponse>*,
              PrepareAsyncGetResultRaw,
              (grpc::ClientContext*,
               const cuopt::remote::GetResultRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::DeleteResponse>*,
              AsyncDeleteResultRaw,
              (grpc::ClientContext*, const cuopt::remote::DeleteRequest&, grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::DeleteResponse>*,
              PrepareAsyncDeleteResultRaw,
              (grpc::ClientContext*, const cuopt::remote::DeleteRequest&, grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::CancelResponse>*,
              AsyncCancelJobRaw,
              (grpc::ClientContext*, const cuopt::remote::CancelRequest&, grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::CancelResponse>*,
              PrepareAsyncCancelJobRaw,
              (grpc::ClientContext*, const cuopt::remote::CancelRequest&, grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::WaitResponse>*,
              AsyncWaitForCompletionRaw,
              (grpc::ClientContext*, const cuopt::remote::WaitRequest&, grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::WaitResponse>*,
              PrepareAsyncWaitForCompletionRaw,
              (grpc::ClientContext*, const cuopt::remote::WaitRequest&, grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::IncumbentResponse>*,
              AsyncGetIncumbentsRaw,
              (grpc::ClientContext*,
               const cuopt::remote::IncumbentRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::IncumbentResponse>*,
              PrepareAsyncGetIncumbentsRaw,
              (grpc::ClientContext*,
               const cuopt::remote::IncumbentRequest&,
               grpc::CompletionQueue*),
              (override));

  // Async chunked result download RPCs
  MOCK_METHOD(
    grpc::ClientAsyncResponseReaderInterface<cuopt::remote::StartChunkedDownloadResponse>*,
    AsyncStartChunkedDownloadRaw,
    (grpc::ClientContext*,
     const cuopt::remote::StartChunkedDownloadRequest&,
     grpc::CompletionQueue*),
    (override));

  MOCK_METHOD(
    grpc::ClientAsyncResponseReaderInterface<cuopt::remote::StartChunkedDownloadResponse>*,
    PrepareAsyncStartChunkedDownloadRaw,
    (grpc::ClientContext*,
     const cuopt::remote::StartChunkedDownloadRequest&,
     grpc::CompletionQueue*),
    (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::GetResultChunkResponse>*,
              AsyncGetResultChunkRaw,
              (grpc::ClientContext*,
               const cuopt::remote::GetResultChunkRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::GetResultChunkResponse>*,
              PrepareAsyncGetResultChunkRaw,
              (grpc::ClientContext*,
               const cuopt::remote::GetResultChunkRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(
    grpc::ClientAsyncResponseReaderInterface<cuopt::remote::FinishChunkedDownloadResponse>*,
    AsyncFinishChunkedDownloadRaw,
    (grpc::ClientContext*,
     const cuopt::remote::FinishChunkedDownloadRequest&,
     grpc::CompletionQueue*),
    (override));

  MOCK_METHOD(
    grpc::ClientAsyncResponseReaderInterface<cuopt::remote::FinishChunkedDownloadResponse>*,
    PrepareAsyncFinishChunkedDownloadRaw,
    (grpc::ClientContext*,
     const cuopt::remote::FinishChunkedDownloadRequest&,
     grpc::CompletionQueue*),
    (override));

  MOCK_METHOD(
    grpc::ClientAsyncReaderInterface<cuopt::remote::LogMessage>*,
    AsyncStreamLogsRaw,
    (grpc::ClientContext*, const cuopt::remote::StreamLogsRequest&, grpc::CompletionQueue*, void*),
    (override));

  MOCK_METHOD(grpc::ClientAsyncReaderInterface<cuopt::remote::LogMessage>*,
              PrepareAsyncStreamLogsRaw,
              (grpc::ClientContext*,
               const cuopt::remote::StreamLogsRequest&,
               grpc::CompletionQueue*),
              (override));

  // Async chunked upload RPCs
  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::StartChunkedUploadResponse>*,
              AsyncStartChunkedUploadRaw,
              (grpc::ClientContext*,
               const cuopt::remote::StartChunkedUploadRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::StartChunkedUploadResponse>*,
              PrepareAsyncStartChunkedUploadRaw,
              (grpc::ClientContext*,
               const cuopt::remote::StartChunkedUploadRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::SendArrayChunkResponse>*,
              AsyncSendArrayChunkRaw,
              (grpc::ClientContext*,
               const cuopt::remote::SendArrayChunkRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::SendArrayChunkResponse>*,
              PrepareAsyncSendArrayChunkRaw,
              (grpc::ClientContext*,
               const cuopt::remote::SendArrayChunkRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::SubmitJobResponse>*,
              AsyncFinishChunkedUploadRaw,
              (grpc::ClientContext*,
               const cuopt::remote::FinishChunkedUploadRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::SubmitJobResponse>*,
              PrepareAsyncFinishChunkedUploadRaw,
              (grpc::ClientContext*,
               const cuopt::remote::FinishChunkedUploadRequest&,
               grpc::CompletionQueue*),
              (override));
};

/**
 * @brief Test fixture for grpc_client_t tests with mock stub injection
 */
class GrpcClientTest : public ::testing::Test {
 protected:
  std::shared_ptr<NiceMock<MockCuOptStub>> mock_stub_;
  std::unique_ptr<grpc_client_t> client_;

  void SetUp() override
  {
    mock_stub_ = std::make_shared<NiceMock<MockCuOptStub>>();

    // Create a client and inject the mock stub
    grpc_client_config_t config;
    config.server_address = "mock://test";
    client_               = std::make_unique<grpc_client_t>(config);

    // Inject the mock stub using typed helper
    grpc_test_inject_mock_stub_typed(*client_, mock_stub_);
  }

  void TearDown() override
  {
    client_.reset();
    mock_stub_.reset();
  }
};

// =============================================================================
// CheckStatus Tests
// =============================================================================

TEST_F(GrpcClientTest, CheckStatus_Success_Completed)
{
  // Setup mock to return COMPLETED status
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest& req,
                 cuopt::remote::StatusResponse* resp) {
      EXPECT_EQ(req.job_id(), "test-job-123");
      resp->set_job_status(cuopt::remote::COMPLETED);
      resp->set_message("Job completed successfully");
      resp->set_result_size_bytes(1024);
      return grpc::Status::OK;
    });

  auto result = client_->check_status("test-job-123");

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.status, job_status_t::COMPLETED);
  EXPECT_EQ(result.message, "Job completed successfully");
  EXPECT_EQ(result.result_size_bytes, 1024);
}

TEST_F(GrpcClientTest, CheckStatus_Success_Processing)
{
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse* resp) {
      resp->set_job_status(cuopt::remote::PROCESSING);
      resp->set_message("Solving...");
      return grpc::Status::OK;
    });

  auto result = client_->check_status("test-job-456");

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.status, job_status_t::PROCESSING);
}

TEST_F(GrpcClientTest, CheckStatus_JobNotFound)
{
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse* resp) {
      resp->set_job_status(cuopt::remote::NOT_FOUND);
      resp->set_message("Job not found");
      return grpc::Status::OK;
    });

  auto result = client_->check_status("nonexistent-job");

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.status, job_status_t::NOT_FOUND);
}

TEST_F(GrpcClientTest, CheckStatus_RpcFailure_Unavailable)
{
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse*) {
      return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Server unavailable");
    });

  auto result = client_->check_status("test-job");

  EXPECT_FALSE(result.success);
  EXPECT_TRUE(result.error_message.find("Server unavailable") != std::string::npos);
}

TEST_F(GrpcClientTest, CheckStatus_RpcFailure_Internal)
{
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse*) {
      return grpc::Status(grpc::StatusCode::INTERNAL, "Internal server error");
    });

  auto result = client_->check_status("test-job");

  EXPECT_FALSE(result.success);
  EXPECT_TRUE(result.error_message.find("Internal server error") != std::string::npos);
}

// =============================================================================
// CancelJob Tests
// =============================================================================

TEST_F(GrpcClientTest, CancelJob_Success)
{
  EXPECT_CALL(*mock_stub_, CancelJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::CancelRequest& req,
                 cuopt::remote::CancelResponse* resp) {
      EXPECT_EQ(req.job_id(), "job-to-cancel");
      resp->set_job_status(cuopt::remote::CANCELLED);
      resp->set_message("Job cancelled");
      return grpc::Status::OK;
    });

  auto result = client_->cancel_job("job-to-cancel");

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.job_status, job_status_t::CANCELLED);
}

TEST_F(GrpcClientTest, CancelJob_AlreadyCompleted)
{
  EXPECT_CALL(*mock_stub_, CancelJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::CancelRequest&,
                 cuopt::remote::CancelResponse* resp) {
      resp->set_job_status(cuopt::remote::COMPLETED);
      resp->set_message("Job already completed");
      return grpc::Status::OK;
    });

  auto result = client_->cancel_job("completed-job");

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.job_status, job_status_t::COMPLETED);
}

TEST_F(GrpcClientTest, CancelJob_RpcFailure)
{
  EXPECT_CALL(*mock_stub_, CancelJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::CancelRequest&,
                 cuopt::remote::CancelResponse*) {
      return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Server down");
    });

  auto result = client_->cancel_job("job-id");

  EXPECT_FALSE(result.success);
  EXPECT_TRUE(result.error_message.find("Server down") != std::string::npos);
}

// =============================================================================
// DeleteJob Tests
// =============================================================================

TEST_F(GrpcClientTest, DeleteJob_Success)
{
  EXPECT_CALL(*mock_stub_, DeleteResult(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::DeleteRequest& req,
                 cuopt::remote::DeleteResponse* resp) {
      EXPECT_EQ(req.job_id(), "job-to-delete");
      resp->set_status(cuopt::remote::SUCCESS);
      return grpc::Status::OK;
    });

  bool result = client_->delete_job("job-to-delete");

  EXPECT_TRUE(result);
}

TEST_F(GrpcClientTest, DeleteJob_NotFound)
{
  EXPECT_CALL(*mock_stub_, DeleteResult(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::DeleteRequest&,
                 cuopt::remote::DeleteResponse* resp) {
      resp->set_status(cuopt::remote::ERROR_NOT_FOUND);
      return grpc::Status::OK;
    });

  bool result = client_->delete_job("nonexistent-job");

  // Job not found should return false to prevent silent failures
  EXPECT_FALSE(result);
}

TEST_F(GrpcClientTest, DeleteJob_RpcFailure)
{
  EXPECT_CALL(*mock_stub_, DeleteResult(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::DeleteRequest&,
                 cuopt::remote::DeleteResponse*) {
      return grpc::Status(grpc::StatusCode::INTERNAL, "Delete failed");
    });

  bool result = client_->delete_job("job-id");

  EXPECT_FALSE(result);
}

// =============================================================================
// WaitForCompletion Tests
// =============================================================================

TEST_F(GrpcClientTest, WaitForCompletion_Success)
{
  EXPECT_CALL(*mock_stub_, WaitForCompletion(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::WaitRequest& req,
                 cuopt::remote::WaitResponse* resp) {
      EXPECT_EQ(req.job_id(), "wait-job");
      resp->set_job_status(cuopt::remote::COMPLETED);
      resp->set_message("Done");
      resp->set_result_size_bytes(2048);
      return grpc::Status::OK;
    });

  auto result = client_->wait_for_completion("wait-job");

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.status, job_status_t::COMPLETED);
  EXPECT_EQ(result.result_size_bytes, 2048);
}

TEST_F(GrpcClientTest, WaitForCompletion_Failed)
{
  EXPECT_CALL(*mock_stub_, WaitForCompletion(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::WaitRequest&,
                 cuopt::remote::WaitResponse* resp) {
      resp->set_job_status(cuopt::remote::FAILED);
      resp->set_message("Solve failed: out of memory");
      return grpc::Status::OK;
    });

  auto result = client_->wait_for_completion("failed-job");

  EXPECT_TRUE(result.success);  // RPC succeeded, job failed
  EXPECT_EQ(result.status, job_status_t::FAILED);
  EXPECT_TRUE(result.message.find("out of memory") != std::string::npos);
}

TEST_F(GrpcClientTest, WaitForCompletion_RpcTimeout)
{
  EXPECT_CALL(*mock_stub_, WaitForCompletion(_, _, _))
    .WillOnce(
      [](grpc::ClientContext*, const cuopt::remote::WaitRequest&, cuopt::remote::WaitResponse*) {
        return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "Deadline exceeded");
      });

  auto result = client_->wait_for_completion("timeout-job");

  EXPECT_FALSE(result.success);
  EXPECT_TRUE(result.error_message.find("Deadline exceeded") != std::string::npos);
}

// =============================================================================
// GetIncumbents Tests
// =============================================================================

TEST_F(GrpcClientTest, GetIncumbents_Success)
{
  EXPECT_CALL(*mock_stub_, GetIncumbents(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::IncumbentRequest& req,
                 cuopt::remote::IncumbentResponse* resp) {
      EXPECT_EQ(req.job_id(), "mip-job");
      EXPECT_EQ(req.from_index(), 0);

      auto* inc1 = resp->add_incumbents();
      inc1->set_index(0);
      inc1->set_objective(100.5);
      inc1->add_assignment(1.0);
      inc1->add_assignment(0.0);

      auto* inc2 = resp->add_incumbents();
      inc2->set_index(1);
      inc2->set_objective(95.3);
      inc2->add_assignment(1.0);
      inc2->add_assignment(1.0);

      resp->set_next_index(2);
      resp->set_job_complete(false);
      return grpc::Status::OK;
    });

  auto result = client_->get_incumbents("mip-job", 0, 10);

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.incumbents.size(), 2);
  EXPECT_EQ(result.incumbents[0].index, 0);
  EXPECT_DOUBLE_EQ(result.incumbents[0].objective, 100.5);
  EXPECT_EQ(result.incumbents[1].index, 1);
  EXPECT_DOUBLE_EQ(result.incumbents[1].objective, 95.3);
  EXPECT_EQ(result.next_index, 2);
  EXPECT_FALSE(result.job_complete);
}

TEST_F(GrpcClientTest, GetIncumbents_NoNewIncumbents)
{
  EXPECT_CALL(*mock_stub_, GetIncumbents(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::IncumbentRequest& req,
                 cuopt::remote::IncumbentResponse* resp) {
      resp->set_next_index(req.from_index());  // No new incumbents
      resp->set_job_complete(false);
      return grpc::Status::OK;
    });

  auto result = client_->get_incumbents("mip-job", 5, 10);

  EXPECT_TRUE(result.success);
  EXPECT_TRUE(result.incumbents.empty());
  EXPECT_EQ(result.next_index, 5);
}

// =============================================================================
// Connection Test (without mock - tests real connection failure)
// =============================================================================

TEST(GrpcClientConnectionTest, Connect_ServerUnavailable)
{
  grpc_client_config_t config;
  config.server_address  = "localhost:1";  // Invalid port
  config.timeout_seconds = 1;

  grpc_client_t client(config);
  EXPECT_FALSE(client.connect());
  EXPECT_FALSE(client.get_last_error().empty());
}

TEST(GrpcClientConnectionTest, IsConnected_BeforeConnect)
{
  grpc_client_config_t config;
  config.server_address = "localhost:9999";

  grpc_client_t client(config);
  EXPECT_FALSE(client.is_connected());
}

// =============================================================================
// Transient Failure / Retry Behavior Tests
// =============================================================================

TEST_F(GrpcClientTest, CheckStatus_TransientFailureThenSuccess)
{
  // First call fails with UNAVAILABLE (transient), second succeeds
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse*) {
      return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Temporary failure");
    })
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse* resp) {
      resp->set_job_status(cuopt::remote::COMPLETED);
      return grpc::Status::OK;
    });

  // First call should fail
  auto result1 = client_->check_status("retry-job");
  EXPECT_FALSE(result1.success);

  // Second call should succeed (simulates retry at higher level)
  auto result2 = client_->check_status("retry-job");
  EXPECT_TRUE(result2.success);
  EXPECT_EQ(result2.status, job_status_t::COMPLETED);
}

TEST_F(GrpcClientTest, GetResult_InternalError)
{
  // Server reports internal error
  EXPECT_CALL(*mock_stub_, GetResult(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::GetResultRequest&,
                 cuopt::remote::ResultResponse*) {
      return grpc::Status(grpc::StatusCode::INTERNAL, "Internal server error");
    });

  auto result = client_->get_lp_result<int32_t, double>("error-job");
  EXPECT_FALSE(result.success);
  EXPECT_FALSE(result.error_message.empty());
}

TEST_F(GrpcClientTest, CancelJob_DeadlineExceeded)
{
  EXPECT_CALL(*mock_stub_, CancelJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::CancelRequest&,
                 cuopt::remote::CancelResponse*) {
      return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "Request timeout");
    });

  auto result = client_->cancel_job("timeout-job");
  EXPECT_FALSE(result.success);
}

// =============================================================================
// Malformed Response Tests
// =============================================================================

TEST_F(GrpcClientTest, CheckStatus_MalformedResponse_InvalidStatus)
{
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse* resp) {
      // Set an invalid/unexpected status value
      resp->set_job_status(static_cast<cuopt::remote::JobStatus>(999));
      return grpc::Status::OK;
    });

  auto result = client_->check_status("malformed-job");

  // Should handle gracefully - either map to unknown or report error
  EXPECT_TRUE(result.success);  // RPC succeeded
}

TEST_F(GrpcClientTest, GetIncumbents_MalformedResponse_NegativeIndex)
{
  EXPECT_CALL(*mock_stub_, GetIncumbents(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::IncumbentRequest&,
                 cuopt::remote::IncumbentResponse* resp) {
      auto* inc = resp->add_incumbents();
      inc->set_index(-1);  // Invalid negative index
      inc->set_objective(100.0);
      resp->set_next_index(-5);  // Invalid
      return grpc::Status::OK;
    });

  auto result = client_->get_incumbents("malformed-job", 0, 10);

  // Should handle gracefully
  EXPECT_TRUE(result.success);
}

TEST_F(GrpcClientTest, WaitForCompletion_EmptyMessage)
{
  EXPECT_CALL(*mock_stub_, WaitForCompletion(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::WaitRequest&,
                 cuopt::remote::WaitResponse* resp) {
      // Don't set any fields - empty response
      return grpc::Status::OK;
    });

  auto result = client_->wait_for_completion("empty-response-job");

  // Should handle gracefully with default values
  EXPECT_TRUE(result.success);
}

// =============================================================================
// Chunked Download Tests (Mock)
// =============================================================================

TEST_F(GrpcClientTest, ChunkedDownload_FallbackOnResourceExhausted)
{
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse* resp) {
      resp->set_job_status(cuopt::remote::COMPLETED);
      resp->set_result_size_bytes(500);
      resp->set_max_message_bytes(256 * 1024 * 1024);
      return grpc::Status::OK;
    });

  EXPECT_CALL(*mock_stub_, GetResult(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::GetResultRequest&,
                 cuopt::remote::ResultResponse*) {
      return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, "Too large");
    });

  EXPECT_CALL(*mock_stub_, StartChunkedDownload(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StartChunkedDownloadRequest& req,
                 cuopt::remote::StartChunkedDownloadResponse* resp) {
      resp->set_download_id("dl-001");
      auto* h = resp->mutable_header();
      h->set_problem_category(cuopt::remote::LP);
      h->set_lp_termination_status(cuopt::remote::PDLP_OPTIMAL);
      h->set_primal_objective(-464.753);
      auto* arr = h->add_arrays();
      arr->set_field_id(cuopt::remote::RESULT_PRIMAL_SOLUTION);
      arr->set_total_elements(2);
      arr->set_element_size_bytes(8);
      resp->set_max_message_bytes(4 * 1024 * 1024);
      return grpc::Status::OK;
    });

  EXPECT_CALL(*mock_stub_, GetResultChunk(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::GetResultChunkRequest& req,
                 cuopt::remote::GetResultChunkResponse* resp) {
      EXPECT_EQ(req.download_id(), "dl-001");
      EXPECT_EQ(req.field_id(), cuopt::remote::RESULT_PRIMAL_SOLUTION);
      resp->set_download_id("dl-001");
      resp->set_field_id(req.field_id());
      resp->set_element_offset(0);
      resp->set_elements_in_chunk(2);
      double vals[2] = {1.5, 2.5};
      resp->set_data(reinterpret_cast<const char*>(vals), sizeof(vals));
      return grpc::Status::OK;
    });

  EXPECT_CALL(*mock_stub_, FinishChunkedDownload(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::FinishChunkedDownloadRequest& req,
                 cuopt::remote::FinishChunkedDownloadResponse* resp) {
      resp->set_download_id(req.download_id());
      return grpc::Status::OK;
    });

  auto lp_result = client_->get_lp_result<int32_t, double>("test-job");

  EXPECT_TRUE(lp_result.success) << lp_result.error_message;
  ASSERT_NE(lp_result.solution, nullptr);
  EXPECT_NEAR(lp_result.solution->get_objective_value(), -464.753, 0.01);
}

TEST_F(GrpcClientTest, ChunkedDownload_StartFails)
{
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse* resp) {
      resp->set_job_status(cuopt::remote::COMPLETED);
      resp->set_result_size_bytes(1000000);
      resp->set_max_message_bytes(100);
      return grpc::Status::OK;
    });

  EXPECT_CALL(*mock_stub_, StartChunkedDownload(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StartChunkedDownloadRequest&,
                 cuopt::remote::StartChunkedDownloadResponse*) {
      return grpc::Status(grpc::StatusCode::NOT_FOUND, "Job not found");
    });

  auto lp_result = client_->get_lp_result<int32_t, double>("test-job");

  EXPECT_FALSE(lp_result.success);
  EXPECT_TRUE(lp_result.error_message.find("StartChunkedDownload") != std::string::npos);
}

// =============================================================================
// Helper: Build minimal test problems
// =============================================================================

namespace {

cpu_optimization_problem_t<int32_t, double> create_test_lp_problem()
{
  cpu_optimization_problem_t<int32_t, double> problem;

  // minimize x  subject to x >= 1
  std::vector<double> obj    = {1.0};
  std::vector<double> var_lb = {0.0};
  std::vector<double> var_ub = {10.0};
  std::vector<double> con_lb = {1.0};
  std::vector<double> con_ub = {1e20};
  std::vector<double> A_vals = {1.0};
  std::vector<int32_t> A_idx = {0};
  std::vector<int32_t> A_off = {0, 1};

  problem.set_objective_coefficients(obj.data(), 1);
  problem.set_maximize(false);
  problem.set_variable_lower_bounds(var_lb.data(), 1);
  problem.set_variable_upper_bounds(var_ub.data(), 1);
  problem.set_csr_constraint_matrix(A_vals.data(), 1, A_idx.data(), 1, A_off.data(), 2);
  problem.set_constraint_lower_bounds(con_lb.data(), 1);
  problem.set_constraint_upper_bounds(con_ub.data(), 1);

  return problem;
}

cpu_optimization_problem_t<int32_t, double> create_test_mip_problem()
{
  cpu_optimization_problem_t<int32_t, double> problem;

  // minimize x  subject to x >= 1, x integer
  std::vector<double> obj    = {1.0};
  std::vector<double> var_lb = {0.0};
  std::vector<double> var_ub = {10.0};
  std::vector<var_t> var_ty  = {var_t::INTEGER};
  std::vector<double> con_lb = {1.0};
  std::vector<double> con_ub = {1e20};
  std::vector<double> A_vals = {1.0};
  std::vector<int32_t> A_idx = {0};
  std::vector<int32_t> A_off = {0, 1};

  problem.set_objective_coefficients(obj.data(), 1);
  problem.set_maximize(false);
  problem.set_variable_lower_bounds(var_lb.data(), 1);
  problem.set_variable_upper_bounds(var_ub.data(), 1);
  problem.set_variable_types(var_ty.data(), 1);
  problem.set_csr_constraint_matrix(A_vals.data(), 1, A_idx.data(), 1, A_off.data(), 2);
  problem.set_constraint_lower_bounds(con_lb.data(), 1);
  problem.set_constraint_upper_bounds(con_ub.data(), 1);

  return problem;
}

}  // namespace

// =============================================================================
// SubmitLP / SubmitMIP Tests
// =============================================================================

TEST_F(GrpcClientTest, SubmitLP_Success)
{
  EXPECT_CALL(*mock_stub_, SubmitJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::SubmitJobRequest& req,
                 cuopt::remote::SubmitJobResponse* resp) {
      EXPECT_TRUE(req.has_lp_request());
      resp->set_job_id("lp-job-001");
      resp->set_message("Job submitted");
      return grpc::Status::OK;
    });

  auto problem = create_test_lp_problem();
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto result = client_->submit_lp(problem, settings);

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.job_id, "lp-job-001");
  EXPECT_TRUE(result.error_message.empty());
}

TEST_F(GrpcClientTest, SubmitLP_NotConnected)
{
  // Create a fresh client that is NOT marked as connected
  grpc_client_config_t config;
  config.server_address = "mock://disconnected";
  grpc_client_t disconnected_client(config);

  auto problem = create_test_lp_problem();
  pdlp_solver_settings_t<int32_t, double> settings;

  auto result = disconnected_client.submit_lp(problem, settings);

  EXPECT_FALSE(result.success);
  EXPECT_TRUE(result.error_message.find("Not connected") != std::string::npos);
}

TEST_F(GrpcClientTest, SubmitLP_RpcFailure)
{
  EXPECT_CALL(*mock_stub_, SubmitJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::SubmitJobRequest&,
                 cuopt::remote::SubmitJobResponse*) {
      return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Server unreachable");
    });

  auto problem = create_test_lp_problem();
  pdlp_solver_settings_t<int32_t, double> settings;

  auto result = client_->submit_lp(problem, settings);

  EXPECT_FALSE(result.success);
  EXPECT_TRUE(result.error_message.find("Server unreachable") != std::string::npos);
}

TEST_F(GrpcClientTest, SubmitLP_EmptyJobId)
{
  EXPECT_CALL(*mock_stub_, SubmitJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::SubmitJobRequest&,
                 cuopt::remote::SubmitJobResponse* resp) {
      resp->set_job_id("");
      return grpc::Status::OK;
    });

  auto problem = create_test_lp_problem();
  pdlp_solver_settings_t<int32_t, double> settings;

  auto result = client_->submit_lp(problem, settings);

  EXPECT_FALSE(result.success);
}

TEST_F(GrpcClientTest, SubmitMIP_Success)
{
  EXPECT_CALL(*mock_stub_, SubmitJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::SubmitJobRequest& req,
                 cuopt::remote::SubmitJobResponse* resp) {
      EXPECT_TRUE(req.has_mip_request());
      resp->set_job_id("mip-job-001");
      resp->set_message("MIP job submitted");
      return grpc::Status::OK;
    });

  auto problem = create_test_mip_problem();
  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto result = client_->submit_mip(problem, settings);

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.job_id, "mip-job-001");
}

TEST_F(GrpcClientTest, SubmitMIP_RpcFailure)
{
  EXPECT_CALL(*mock_stub_, SubmitJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::SubmitJobRequest&,
                 cuopt::remote::SubmitJobResponse*) {
      return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Server unreachable");
    });

  auto problem = create_test_mip_problem();
  mip_solver_settings_t<int32_t, double> settings;

  auto result = client_->submit_mip(problem, settings);

  EXPECT_FALSE(result.success);
}

// =============================================================================
// SolveLP / SolveMIP Tests (end-to-end mock flow)
// =============================================================================

TEST_F(GrpcClientTest, SolveLP_SuccessWithPolling)
{
  // 1. SubmitJob succeeds
  auto problem = create_test_lp_problem();
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  grpc_client_config_t cfg;
  cfg.server_address   = "mock://test";
  cfg.poll_interval_ms = 10;
  cfg.timeout_seconds  = 5;

  auto client = std::make_unique<grpc_client_t>(cfg);
  auto mock   = std::make_shared<NiceMock<MockCuOptStub>>();
  grpc_test_inject_mock_stub_typed(*client, mock);

  EXPECT_CALL(*mock, SubmitJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::SubmitJobRequest& req,
                 cuopt::remote::SubmitJobResponse* resp) {
      EXPECT_TRUE(req.has_lp_request());
      resp->set_job_id("solve-lp-001");
      return grpc::Status::OK;
    });

  EXPECT_CALL(*mock, CheckStatus(_, _, _))
    .WillRepeatedly([](grpc::ClientContext*,
                       const cuopt::remote::StatusRequest&,
                       cuopt::remote::StatusResponse* resp) {
      resp->set_job_status(cuopt::remote::COMPLETED);
      resp->set_result_size_bytes(64);
      return grpc::Status::OK;
    });

  EXPECT_CALL(*mock, GetResult(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::GetResultRequest& req,
                 cuopt::remote::ResultResponse* resp) {
      EXPECT_EQ(req.job_id(), "solve-lp-001");
      cuopt::remote::LPSolution solution;
      solution.add_primal_solution(1.0);
      solution.set_primal_objective(1.0);
      solution.set_lp_termination_status(cuopt::remote::PDLP_OPTIMAL);
      resp->mutable_lp_solution()->CopyFrom(solution);
      resp->set_status(cuopt::remote::SUCCESS);
      return grpc::Status::OK;
    });

  auto result = client->solve_lp(problem, settings);

  EXPECT_TRUE(result.success) << "Error: " << result.error_message;
  EXPECT_NE(result.solution, nullptr);
  if (result.solution) { EXPECT_DOUBLE_EQ(result.solution->get_objective_value(), 1.0); }
}

TEST_F(GrpcClientTest, SolveLP_SuccessWithWait)
{
  grpc_client_config_t cfg;
  cfg.server_address   = "mock://test";
  cfg.poll_interval_ms = 10;
  cfg.timeout_seconds  = 5;

  auto client = std::make_unique<grpc_client_t>(cfg);
  auto mock   = std::make_shared<NiceMock<MockCuOptStub>>();
  grpc_test_inject_mock_stub_typed(*client, mock);

  EXPECT_CALL(*mock, SubmitJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::SubmitJobRequest&,
                 cuopt::remote::SubmitJobResponse* resp) {
      resp->set_job_id("wait-lp-001");
      return grpc::Status::OK;
    });

  EXPECT_CALL(*mock, CheckStatus(_, _, _))
    .WillRepeatedly([](grpc::ClientContext*,
                       const cuopt::remote::StatusRequest&,
                       cuopt::remote::StatusResponse* resp) {
      resp->set_job_status(cuopt::remote::COMPLETED);
      resp->set_result_size_bytes(64);
      return grpc::Status::OK;
    });

  EXPECT_CALL(*mock, GetResult(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::GetResultRequest&,
                 cuopt::remote::ResultResponse* resp) {
      cuopt::remote::LPSolution solution;
      solution.add_primal_solution(1.0);
      solution.set_primal_objective(1.0);
      solution.set_lp_termination_status(cuopt::remote::PDLP_OPTIMAL);
      resp->mutable_lp_solution()->CopyFrom(solution);
      resp->set_status(cuopt::remote::SUCCESS);
      return grpc::Status::OK;
    });

  auto problem = create_test_lp_problem();
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto result = client->solve_lp(problem, settings);

  EXPECT_TRUE(result.success) << "Error: " << result.error_message;
  EXPECT_NE(result.solution, nullptr);
}

TEST_F(GrpcClientTest, SolveLP_JobFails)
{
  grpc_client_config_t cfg;
  cfg.server_address   = "mock://test";
  cfg.poll_interval_ms = 10;
  cfg.timeout_seconds  = 5;

  auto client = std::make_unique<grpc_client_t>(cfg);
  auto mock   = std::make_shared<NiceMock<MockCuOptStub>>();
  grpc_test_inject_mock_stub_typed(*client, mock);

  EXPECT_CALL(*mock, SubmitJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::SubmitJobRequest&,
                 cuopt::remote::SubmitJobResponse* resp) {
      resp->set_job_id("fail-lp-001");
      return grpc::Status::OK;
    });

  EXPECT_CALL(*mock, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse* resp) {
      resp->set_job_status(cuopt::remote::FAILED);
      resp->set_message("Out of GPU memory");
      return grpc::Status::OK;
    });

  auto problem = create_test_lp_problem();
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto result = client->solve_lp(problem, settings);

  EXPECT_FALSE(result.success);
  EXPECT_TRUE(result.error_message.find("Out of GPU memory") != std::string::npos)
    << "Error: " << result.error_message;
}

TEST_F(GrpcClientTest, SolveLP_SubmitFails)
{
  grpc_client_config_t cfg;
  cfg.server_address  = "mock://test";
  cfg.timeout_seconds = 5;

  auto client = std::make_unique<grpc_client_t>(cfg);
  auto mock   = std::make_shared<NiceMock<MockCuOptStub>>();
  grpc_test_inject_mock_stub_typed(*client, mock);

  EXPECT_CALL(*mock, SubmitJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::SubmitJobRequest&,
                 cuopt::remote::SubmitJobResponse*) {
      return grpc::Status(grpc::StatusCode::INTERNAL, "Server crashed");
    });

  auto problem = create_test_lp_problem();
  pdlp_solver_settings_t<int32_t, double> settings;

  auto result = client->solve_lp(problem, settings);

  EXPECT_FALSE(result.success);
  EXPECT_TRUE(result.error_message.find("Server crashed") != std::string::npos)
    << "Error: " << result.error_message;
}

TEST_F(GrpcClientTest, SolveLP_NotConnected)
{
  grpc_client_config_t cfg;
  cfg.server_address = "mock://disconnected";

  grpc_client_t client(cfg);
  // Don't inject mock or mark as connected

  auto problem = create_test_lp_problem();
  pdlp_solver_settings_t<int32_t, double> settings;

  auto result = client.solve_lp(problem, settings);

  EXPECT_FALSE(result.success);
  EXPECT_TRUE(result.error_message.find("Not connected") != std::string::npos);
}

TEST_F(GrpcClientTest, SolveMIP_Success)
{
  grpc_client_config_t cfg;
  cfg.server_address   = "mock://test";
  cfg.poll_interval_ms = 10;
  cfg.timeout_seconds  = 5;

  auto client = std::make_unique<grpc_client_t>(cfg);
  auto mock   = std::make_shared<NiceMock<MockCuOptStub>>();
  grpc_test_inject_mock_stub_typed(*client, mock);

  EXPECT_CALL(*mock, SubmitJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::SubmitJobRequest& req,
                 cuopt::remote::SubmitJobResponse* resp) {
      EXPECT_TRUE(req.has_mip_request());
      resp->set_job_id("mip-solve-001");
      return grpc::Status::OK;
    });

  EXPECT_CALL(*mock, CheckStatus(_, _, _))
    .WillRepeatedly([](grpc::ClientContext*,
                       const cuopt::remote::StatusRequest&,
                       cuopt::remote::StatusResponse* resp) {
      resp->set_job_status(cuopt::remote::COMPLETED);
      resp->set_result_size_bytes(64);
      return grpc::Status::OK;
    });

  EXPECT_CALL(*mock, GetResult(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::GetResultRequest&,
                 cuopt::remote::ResultResponse* resp) {
      cuopt::remote::MIPSolution solution;
      solution.add_mip_solution(1.0);
      solution.set_mip_objective(1.0);
      solution.set_mip_termination_status(cuopt::remote::MIP_OPTIMAL);
      resp->mutable_mip_solution()->CopyFrom(solution);
      resp->set_status(cuopt::remote::SUCCESS);
      return grpc::Status::OK;
    });

  auto problem = create_test_mip_problem();
  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto result = client->solve_mip(problem, settings);

  EXPECT_TRUE(result.success) << "Error: " << result.error_message;
  EXPECT_NE(result.solution, nullptr);
  if (result.solution) { EXPECT_DOUBLE_EQ(result.solution->get_objective_value(), 1.0); }
}

// =============================================================================
// GetResult on PROCESSING job
// =============================================================================

TEST_F(GrpcClientTest, GetResult_ProcessingJobReturnsError)
{
  // When a job is still PROCESSING, GetResult returns UNAVAILABLE.
  // The client's get_result_or_stream first calls CheckStatus; if the job
  // is not complete, it should not attempt GetResult at all.
  // Here we test the lower-level get_lp_result path with a CheckStatus
  // returning PROCESSING (small result size so no streaming fallback).
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse* resp) {
      resp->set_job_status(cuopt::remote::PROCESSING);
      resp->set_result_size_bytes(0);
      return grpc::Status::OK;
    });

  // GetResult should be called because CheckStatus doesn't show large result
  EXPECT_CALL(*mock_stub_, GetResult(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::GetResultRequest&,
                 cuopt::remote::ResultResponse*) {
      return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Result not ready");
    });

  auto result = client_->get_lp_result<int32_t, double>("processing-job");
  EXPECT_FALSE(result.success);
}

// =============================================================================
// DeleteJob then verify subsequent operations fail
// =============================================================================

TEST_F(GrpcClientTest, DeleteJob_ThenCheckStatusNotFound)
{
  // Delete succeeds
  EXPECT_CALL(*mock_stub_, DeleteResult(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::DeleteRequest& req,
                 cuopt::remote::DeleteResponse* resp) {
      EXPECT_EQ(req.job_id(), "delete-then-check");
      resp->set_status(cuopt::remote::SUCCESS);
      return grpc::Status::OK;
    });

  bool deleted = client_->delete_job("delete-then-check");
  EXPECT_TRUE(deleted);

  // Subsequent CheckStatus returns NOT_FOUND
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest& req,
                 cuopt::remote::StatusResponse* resp) {
      EXPECT_EQ(req.job_id(), "delete-then-check");
      resp->set_job_status(cuopt::remote::NOT_FOUND);
      resp->set_message("Job not found");
      return grpc::Status::OK;
    });

  auto status = client_->check_status("delete-then-check");
  EXPECT_TRUE(status.success);
  EXPECT_EQ(status.status, job_status_t::NOT_FOUND);
}

TEST_F(GrpcClientTest, DeleteJob_ThenGetResultFails)
{
  EXPECT_CALL(*mock_stub_, DeleteResult(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::DeleteRequest&,
                 cuopt::remote::DeleteResponse* resp) {
      resp->set_status(cuopt::remote::SUCCESS);
      return grpc::Status::OK;
    });

  client_->delete_job("deleted-job");

  // GetResult after deletion
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse* resp) {
      resp->set_job_status(cuopt::remote::NOT_FOUND);
      return grpc::Status::OK;
    });

  EXPECT_CALL(*mock_stub_, GetResult(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::GetResultRequest&,
                 cuopt::remote::ResultResponse*) {
      return grpc::Status(grpc::StatusCode::NOT_FOUND, "Job not found");
    });

  auto result = client_->get_lp_result<int32_t, double>("deleted-job");
  EXPECT_FALSE(result.success);
}

// =============================================================================
// WaitForCompletion with cancelled job
// =============================================================================

TEST_F(GrpcClientTest, WaitForCompletion_Cancelled)
{
  EXPECT_CALL(*mock_stub_, WaitForCompletion(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::WaitRequest& req,
                 cuopt::remote::WaitResponse* resp) {
      EXPECT_EQ(req.job_id(), "cancelled-job");
      resp->set_job_status(cuopt::remote::CANCELLED);
      resp->set_message("Job was cancelled");
      return grpc::Status::OK;
    });

  auto result = client_->wait_for_completion("cancelled-job");

  EXPECT_TRUE(result.success);  // RPC succeeded
  EXPECT_EQ(result.status, job_status_t::CANCELLED);
  EXPECT_TRUE(result.message.find("cancelled") != std::string::npos);
}

// =============================================================================
// StreamLogs Tests (Mock)
// =============================================================================

class MockLogStream : public grpc::ClientReaderInterface<cuopt::remote::LogMessage> {
 public:
  explicit MockLogStream(std::vector<cuopt::remote::LogMessage> msgs)
    : messages_(std::move(msgs)), idx_(0)
  {
  }

  bool Read(cuopt::remote::LogMessage* msg) override
  {
    if (idx_ >= messages_.size()) return false;
    *msg = messages_[idx_++];
    return true;
  }

  grpc::Status Finish() override { return grpc::Status::OK; }
  bool NextMessageSize(uint32_t* sz) override
  {
    if (idx_ >= messages_.size()) return false;
    *sz = messages_[idx_].ByteSizeLong();
    return true;
  }
  void WaitForInitialMetadata() override {}

 private:
  std::vector<cuopt::remote::LogMessage> messages_;
  size_t idx_;
};

TEST_F(GrpcClientTest, StreamLogs_ReceivesLogLines)
{
  std::vector<cuopt::remote::LogMessage> msgs;

  cuopt::remote::LogMessage msg1;
  msg1.set_line("Iteration 1: obj=100.0");
  msg1.set_job_complete(false);
  msgs.push_back(msg1);

  cuopt::remote::LogMessage msg2;
  msg2.set_line("Iteration 2: obj=50.0");
  msg2.set_job_complete(false);
  msgs.push_back(msg2);

  cuopt::remote::LogMessage msg3;
  msg3.set_line("Solve complete");
  msg3.set_job_complete(true);
  msgs.push_back(msg3);

  auto* mock_reader = new MockLogStream(msgs);
  EXPECT_CALL(*mock_stub_, StreamLogsRaw(_, _)).WillOnce(Return(mock_reader));

  std::vector<std::string> received_lines;
  bool result = client_->stream_logs("log-job", 0, [&](const std::string& line, bool complete) {
    received_lines.push_back(line);
    return true;  // keep streaming
  });

  EXPECT_TRUE(result);
  EXPECT_EQ(received_lines.size(), 3);
  EXPECT_EQ(received_lines[0], "Iteration 1: obj=100.0");
  EXPECT_EQ(received_lines[2], "Solve complete");
}

TEST_F(GrpcClientTest, StreamLogs_CallbackStopsEarly)
{
  std::vector<cuopt::remote::LogMessage> msgs;

  cuopt::remote::LogMessage msg1;
  msg1.set_line("Line 1");
  msg1.set_job_complete(false);
  msgs.push_back(msg1);

  cuopt::remote::LogMessage msg2;
  msg2.set_line("Line 2");
  msg2.set_job_complete(false);
  msgs.push_back(msg2);

  auto* mock_reader = new MockLogStream(msgs);
  EXPECT_CALL(*mock_stub_, StreamLogsRaw(_, _)).WillOnce(Return(mock_reader));

  int count = 0;
  client_->stream_logs("log-job", 0, [&](const std::string&, bool) {
    count++;
    return false;  // stop after first line
  });

  EXPECT_EQ(count, 1);
}

TEST_F(GrpcClientTest, StreamLogs_EmptyStream)
{
  std::vector<cuopt::remote::LogMessage> msgs;  // empty

  auto* mock_reader = new MockLogStream(msgs);
  EXPECT_CALL(*mock_stub_, StreamLogsRaw(_, _)).WillOnce(Return(mock_reader));

  int count   = 0;
  bool result = client_->stream_logs("log-job", 0, [&](const std::string&, bool) {
    count++;
    return true;
  });

  EXPECT_TRUE(result);
  EXPECT_EQ(count, 0);
}

// =============================================================================
// Chunked Upload Tests (Mock)
// =============================================================================

TEST_F(GrpcClientTest, SubmitLP_ChunkedUploadForLargePayload)
{
  grpc_client_config_t cfg;
  cfg.server_address                = "mock://test";
  cfg.chunked_array_threshold_bytes = 0;  // Force chunked upload for all sizes
  cfg.chunk_size_bytes              = 4 * 1024;

  auto client = std::make_unique<grpc_client_t>(cfg);
  auto mock   = std::make_shared<NiceMock<MockCuOptStub>>();
  grpc_test_inject_mock_stub_typed(*client, mock);

  EXPECT_CALL(*mock, StartChunkedUpload(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StartChunkedUploadRequest& req,
                 cuopt::remote::StartChunkedUploadResponse* resp) {
      EXPECT_TRUE(req.has_problem_header());
      resp->set_upload_id("chunked-upload-001");
      resp->set_max_message_bytes(4 * 1024 * 1024);
      return grpc::Status::OK;
    });

  int chunk_count = 0;
  EXPECT_CALL(*mock, SendArrayChunk(_, _, _))
    .WillRepeatedly([&chunk_count](grpc::ClientContext*,
                                   const cuopt::remote::SendArrayChunkRequest& req,
                                   cuopt::remote::SendArrayChunkResponse* resp) {
      EXPECT_EQ(req.upload_id(), "chunked-upload-001");
      EXPECT_TRUE(req.has_chunk());
      chunk_count++;
      resp->set_upload_id("chunked-upload-001");
      resp->set_chunks_received(chunk_count);
      return grpc::Status::OK;
    });

  EXPECT_CALL(*mock, FinishChunkedUpload(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::FinishChunkedUploadRequest& req,
                 cuopt::remote::SubmitJobResponse* resp) {
      EXPECT_EQ(req.upload_id(), "chunked-upload-001");
      resp->set_job_id("chunked-job-001");
      return grpc::Status::OK;
    });

  auto problem = create_test_lp_problem();
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto result = client->submit_lp(problem, settings);

  EXPECT_TRUE(result.success) << "Error: " << result.error_message;
  EXPECT_EQ(result.job_id, "chunked-job-001");
  EXPECT_GT(chunk_count, 0) << "Should have sent at least one array chunk";
}

TEST_F(GrpcClientTest, SubmitLP_UnaryForSmallPayload)
{
  EXPECT_CALL(*mock_stub_, SubmitJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::SubmitJobRequest&,
                 cuopt::remote::SubmitJobResponse* resp) {
      resp->set_job_id("unary-lp-001");
      return grpc::Status::OK;
    });

  auto problem = create_test_lp_problem();
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto result = client_->submit_lp(problem, settings);

  EXPECT_TRUE(result.success) << "Error: " << result.error_message;
  EXPECT_EQ(result.job_id, "unary-lp-001");
}

// =============================================================================
// Mapper Roundtrip Tests
// =============================================================================

TEST(MapperRoundtrip, MIPSettingsAllFields)
{
  mip_solver_settings_t<int32_t, double> orig;

  // Limits
  orig.time_limit = 42.5;
  orig.work_limit = 1000.0;
  orig.node_limit = 5000;

  // Tolerances
  orig.tolerances.relative_mip_gap            = 1e-3;
  orig.tolerances.absolute_mip_gap            = 1e-8;
  orig.tolerances.integrality_tolerance       = 1e-4;
  orig.tolerances.absolute_tolerance          = 2e-6;
  orig.tolerances.relative_tolerance          = 3e-12;
  orig.tolerances.presolve_absolute_tolerance = 5e-7;

  // Solver configuration
  orig.log_to_console  = false;
  orig.heuristics_only = true;
  orig.num_cpu_threads = 8;
  orig.num_gpus        = 2;
  orig.presolver       = presolver_t::Default;
  orig.mip_scaling     = true;

  // Branching
  orig.reliability_branching           = 32;
  orig.mip_batch_pdlp_strong_branching = 16;

  // Cut configuration
  orig.max_cut_passes             = 20;
  orig.mir_cuts                   = 1;
  orig.mixed_integer_gomory_cuts  = 2;
  orig.knapsack_cuts              = 0;
  orig.clique_cuts                = 3;
  orig.strong_chvatal_gomory_cuts = -1;
  orig.reduced_cost_strengthening = 1;
  orig.cut_change_threshold       = 0.05;
  orig.cut_min_orthogonality      = 0.8;

  // Determinism and reproducibility
  orig.determinism_mode = CUOPT_MODE_DETERMINISTIC;
  orig.seed             = 12345;

  // Roundtrip: C++ -> proto -> C++
  cuopt::remote::MIPSolverSettings pb;
  map_mip_settings_to_proto(orig, &pb);

  mip_solver_settings_t<int32_t, double> restored;
  map_proto_to_mip_settings(pb, restored);

  // Limits
  EXPECT_DOUBLE_EQ(restored.time_limit, 42.5);
  EXPECT_DOUBLE_EQ(restored.work_limit, 1000.0);
  EXPECT_EQ(restored.node_limit, 5000);

  // Tolerances
  EXPECT_DOUBLE_EQ(restored.tolerances.relative_mip_gap, 1e-3);
  EXPECT_DOUBLE_EQ(restored.tolerances.absolute_mip_gap, 1e-8);
  EXPECT_DOUBLE_EQ(restored.tolerances.integrality_tolerance, 1e-4);
  EXPECT_DOUBLE_EQ(restored.tolerances.absolute_tolerance, 2e-6);
  EXPECT_DOUBLE_EQ(restored.tolerances.relative_tolerance, 3e-12);
  EXPECT_DOUBLE_EQ(restored.tolerances.presolve_absolute_tolerance, 5e-7);

  // Solver configuration
  EXPECT_EQ(restored.log_to_console, false);
  EXPECT_EQ(restored.heuristics_only, true);
  EXPECT_EQ(restored.num_cpu_threads, 8);
  EXPECT_EQ(restored.num_gpus, 2);
  EXPECT_EQ(restored.presolver, presolver_t::Default);
  EXPECT_EQ(restored.mip_scaling, true);

  // Branching
  EXPECT_EQ(restored.reliability_branching, 32);
  EXPECT_EQ(restored.mip_batch_pdlp_strong_branching, 16);

  // Cut configuration
  EXPECT_EQ(restored.max_cut_passes, 20);
  EXPECT_EQ(restored.mir_cuts, 1);
  EXPECT_EQ(restored.mixed_integer_gomory_cuts, 2);
  EXPECT_EQ(restored.knapsack_cuts, 0);
  EXPECT_EQ(restored.clique_cuts, 3);
  EXPECT_EQ(restored.strong_chvatal_gomory_cuts, -1);
  EXPECT_EQ(restored.reduced_cost_strengthening, 1);
  EXPECT_DOUBLE_EQ(restored.cut_change_threshold, 0.05);
  EXPECT_DOUBLE_EQ(restored.cut_min_orthogonality, 0.8);

  // Determinism and reproducibility
  EXPECT_EQ(restored.determinism_mode, CUOPT_MODE_DETERMINISTIC);
  EXPECT_EQ(restored.seed, 12345);
}

TEST(MapperRoundtrip, MIPSettingsNodeLimitSentinel)
{
  mip_solver_settings_t<int32_t, double> orig;
  orig.node_limit = std::numeric_limits<int32_t>::max();

  cuopt::remote::MIPSolverSettings pb;
  map_mip_settings_to_proto(orig, &pb);
  EXPECT_EQ(pb.node_limit(), -1) << "max() should map to -1 sentinel in proto";

  mip_solver_settings_t<int32_t, double> restored;
  restored.node_limit = 0;
  map_proto_to_mip_settings(pb, restored);
  EXPECT_EQ(restored.node_limit, 0) << "Negative sentinel should leave node_limit unchanged";
}

TEST(MapperRoundtrip, ProblemWithVariableTypes)
{
  cpu_optimization_problem_t<int32_t, double> orig;

  std::vector<double> obj    = {1.0, 2.0, 3.0};
  std::vector<double> var_lb = {0.0, 0.0, 0.0};
  std::vector<double> var_ub = {10.0, 10.0, 10.0};
  std::vector<var_t> var_ty  = {var_t::CONTINUOUS, var_t::INTEGER, var_t::CONTINUOUS};
  std::vector<double> con_lb = {1.0};
  std::vector<double> con_ub = {1e20};
  std::vector<double> A_vals = {1.0, 1.0, 1.0};
  std::vector<int32_t> A_idx = {0, 1, 2};
  std::vector<int32_t> A_off = {0, 3};

  orig.set_objective_coefficients(obj.data(), 3);
  orig.set_maximize(true);
  orig.set_variable_lower_bounds(var_lb.data(), 3);
  orig.set_variable_upper_bounds(var_ub.data(), 3);
  orig.set_variable_types(var_ty.data(), 3);
  orig.set_csr_constraint_matrix(A_vals.data(), 3, A_idx.data(), 3, A_off.data(), 2);
  orig.set_constraint_lower_bounds(con_lb.data(), 1);
  orig.set_constraint_upper_bounds(con_ub.data(), 1);

  cuopt::remote::OptimizationProblem pb;
  map_problem_to_proto(orig, &pb);

  ASSERT_EQ(pb.variable_types_size(), 3);
  EXPECT_EQ(pb.variable_types(0), cuopt::remote::CONTINUOUS);
  EXPECT_EQ(pb.variable_types(1), cuopt::remote::INTEGER);
  EXPECT_EQ(pb.variable_types(2), cuopt::remote::CONTINUOUS);

  cpu_optimization_problem_t<int32_t, double> restored;
  map_proto_to_problem(pb, restored);

  auto restored_types = restored.get_variable_types_host();
  ASSERT_EQ(restored_types.size(), 3u);
  EXPECT_EQ(restored_types[0], var_t::CONTINUOUS);
  EXPECT_EQ(restored_types[1], var_t::INTEGER);
  EXPECT_EQ(restored_types[2], var_t::CONTINUOUS);

  EXPECT_EQ(restored.get_sense(), true);
  auto restored_obj = restored.get_objective_coefficients_host();
  ASSERT_EQ(restored_obj.size(), 3u);
  EXPECT_DOUBLE_EQ(restored_obj[0], 1.0);
  EXPECT_DOUBLE_EQ(restored_obj[1], 2.0);
  EXPECT_DOUBLE_EQ(restored_obj[2], 3.0);
}

TEST(MapperRoundtrip, MIPSolutionAllFields)
{
  std::vector<double> sol_vec = {1.0, 0.0, 1.0, 0.0, 1.0};

  cpu_mip_solution_t<int32_t, double> orig(std::move(sol_vec),
                                           mip_termination_status_t::FeasibleFound,
                                           42.5,    // objective
                                           0.015,   // mip_gap
                                           40.0,    // solution_bound
                                           12.34,   // total_solve_time
                                           0.56,    // presolve_time
                                           1e-8,    // max_constraint_violation
                                           1e-9,    // max_int_violation
                                           1e-10,   // max_variable_bound_violation
                                           1234,    // num_nodes
                                           56789);  // num_simplex_iterations

  cuopt::remote::MIPSolution pb;
  map_mip_solution_to_proto(orig, &pb);

  EXPECT_EQ(pb.mip_termination_status(), cuopt::remote::MIP_FEASIBLE_FOUND);
  EXPECT_EQ(pb.mip_solution_size(), 5);
  EXPECT_DOUBLE_EQ(pb.mip_objective(), 42.5);
  EXPECT_DOUBLE_EQ(pb.mip_gap(), 0.015);

  auto restored = map_proto_to_mip_solution<int32_t, double>(pb);

  EXPECT_EQ(restored.get_termination_status(), mip_termination_status_t::FeasibleFound);
  EXPECT_DOUBLE_EQ(restored.get_objective_value(), 42.5);
  EXPECT_DOUBLE_EQ(restored.get_mip_gap(), 0.015);
  EXPECT_DOUBLE_EQ(restored.get_solution_bound(), 40.0);
  EXPECT_DOUBLE_EQ(restored.get_solve_time(), 12.34);
  EXPECT_DOUBLE_EQ(restored.get_presolve_time(), 0.56);
  EXPECT_DOUBLE_EQ(restored.get_max_constraint_violation(), 1e-8);
  EXPECT_DOUBLE_EQ(restored.get_max_int_violation(), 1e-9);
  EXPECT_DOUBLE_EQ(restored.get_max_variable_bound_violation(), 1e-10);
  EXPECT_EQ(restored.get_num_nodes(), 1234);
  EXPECT_EQ(restored.get_num_simplex_iterations(), 56789);

  auto restored_sol = restored.get_solution_host();
  ASSERT_EQ(restored_sol.size(), 5u);
  EXPECT_DOUBLE_EQ(restored_sol[0], 1.0);
  EXPECT_DOUBLE_EQ(restored_sol[1], 0.0);
  EXPECT_DOUBLE_EQ(restored_sol[4], 1.0);
}

TEST(MapperRoundtrip, LPSolutionAllFields)
{
  std::vector<double> primal       = {1.5, 2.5, 3.5};
  std::vector<double> dual         = {0.1, 0.2};
  std::vector<double> reduced_cost = {0.0, 0.0, 0.5};

  cpu_lp_solution_t<int32_t, double> orig(std::move(primal),
                                          std::move(dual),
                                          std::move(reduced_cost),
                                          pdlp_termination_status_t::Optimal,
                                          -464.753,         // primal_objective
                                          -464.0,           // dual_objective
                                          1.23,             // solve_time
                                          1e-8,             // l2_primal_residual
                                          2e-8,             // l2_dual_residual
                                          3e-8,             // gap
                                          500,              // num_iterations
                                          method_t::PDLP);  // solved_by

  cuopt::remote::LPSolution pb;
  map_lp_solution_to_proto(orig, &pb);

  EXPECT_EQ(pb.lp_termination_status(), cuopt::remote::PDLP_OPTIMAL);
  EXPECT_EQ(pb.primal_solution_size(), 3);
  EXPECT_EQ(pb.dual_solution_size(), 2);
  EXPECT_EQ(pb.reduced_cost_size(), 3);

  auto restored = map_proto_to_lp_solution<int32_t, double>(pb);

  EXPECT_EQ(restored.get_termination_status(), pdlp_termination_status_t::Optimal);
  EXPECT_NEAR(restored.get_objective_value(), -464.753, 1e-6);
  EXPECT_NEAR(restored.get_dual_objective_value(), -464.0, 1e-6);
  EXPECT_DOUBLE_EQ(restored.get_solve_time(), 1.23);
  EXPECT_DOUBLE_EQ(restored.get_l2_primal_residual(), 1e-8);
  EXPECT_DOUBLE_EQ(restored.get_l2_dual_residual(), 2e-8);
  EXPECT_DOUBLE_EQ(restored.get_gap(), 3e-8);
  EXPECT_EQ(restored.get_num_iterations(), 500);
  EXPECT_EQ(restored.solved_by(), method_t::PDLP);

  auto restored_primal = restored.get_primal_solution_host();
  ASSERT_EQ(restored_primal.size(), 3u);
  EXPECT_DOUBLE_EQ(restored_primal[0], 1.5);
  EXPECT_DOUBLE_EQ(restored_primal[2], 3.5);

  auto restored_dual = restored.get_dual_solution_host();
  ASSERT_EQ(restored_dual.size(), 2u);
  EXPECT_DOUBLE_EQ(restored_dual[0], 0.1);
}

TEST(MapperRoundtrip, PDLPSettingsAllFields)
{
  pdlp_solver_settings_t<int32_t, double> orig;

  orig.tolerances.absolute_gap_tolerance      = 1e-7;
  orig.tolerances.relative_gap_tolerance      = 1e-6;
  orig.tolerances.primal_infeasible_tolerance = 1e-5;
  orig.tolerances.dual_infeasible_tolerance   = 2e-5;
  orig.tolerances.absolute_dual_tolerance     = 3e-7;
  orig.tolerances.relative_dual_tolerance     = 4e-7;
  orig.tolerances.absolute_primal_tolerance   = 5e-7;
  orig.tolerances.relative_primal_tolerance   = 6e-7;

  orig.time_limit                 = 99.5;
  orig.iteration_limit            = 10000;
  orig.log_to_console             = false;
  orig.detect_infeasibility       = true;
  orig.strict_infeasibility       = true;
  orig.pdlp_solver_mode           = pdlp_solver_mode_t::Fast1;
  orig.method                     = method_t::Barrier;
  orig.presolver                  = presolver_t::Default;
  orig.dual_postsolve             = true;
  orig.crossover                  = true;
  orig.num_gpus                   = 4;
  orig.per_constraint_residual    = true;
  orig.cudss_deterministic        = true;
  orig.folding                    = 1;
  orig.augmented                  = 1;
  orig.dualize                    = 1;
  orig.ordering                   = 2;
  orig.barrier_dual_initial_point = 1;
  orig.eliminate_dense_columns    = true;
  orig.pdlp_precision             = pdlp_precision_t::MixedPrecision;
  orig.save_best_primal_so_far    = true;
  orig.first_primal_feasible      = true;

  cuopt::remote::PDLPSolverSettings pb;
  map_pdlp_settings_to_proto(orig, &pb);

  pdlp_solver_settings_t<int32_t, double> restored;
  map_proto_to_pdlp_settings(pb, restored);

  EXPECT_DOUBLE_EQ(restored.tolerances.absolute_gap_tolerance, 1e-7);
  EXPECT_DOUBLE_EQ(restored.tolerances.relative_gap_tolerance, 1e-6);
  EXPECT_DOUBLE_EQ(restored.tolerances.primal_infeasible_tolerance, 1e-5);
  EXPECT_DOUBLE_EQ(restored.tolerances.dual_infeasible_tolerance, 2e-5);
  EXPECT_DOUBLE_EQ(restored.tolerances.absolute_dual_tolerance, 3e-7);
  EXPECT_DOUBLE_EQ(restored.tolerances.relative_dual_tolerance, 4e-7);
  EXPECT_DOUBLE_EQ(restored.tolerances.absolute_primal_tolerance, 5e-7);
  EXPECT_DOUBLE_EQ(restored.tolerances.relative_primal_tolerance, 6e-7);

  EXPECT_DOUBLE_EQ(restored.time_limit, 99.5);
  EXPECT_EQ(restored.iteration_limit, 10000);
  EXPECT_EQ(restored.log_to_console, false);
  EXPECT_EQ(restored.detect_infeasibility, true);
  EXPECT_EQ(restored.strict_infeasibility, true);
  EXPECT_EQ(restored.pdlp_solver_mode, pdlp_solver_mode_t::Fast1);
  EXPECT_EQ(restored.method, method_t::Barrier);
  EXPECT_EQ(restored.presolver, presolver_t::Default);
  EXPECT_EQ(restored.dual_postsolve, true);
  EXPECT_EQ(restored.crossover, true);
  EXPECT_EQ(restored.num_gpus, 4);
  EXPECT_EQ(restored.per_constraint_residual, true);
  EXPECT_EQ(restored.cudss_deterministic, true);
  EXPECT_EQ(restored.folding, 1);
  EXPECT_EQ(restored.augmented, 1);
  EXPECT_EQ(restored.dualize, 1);
  EXPECT_EQ(restored.ordering, 2);
  EXPECT_EQ(restored.barrier_dual_initial_point, 1);
  EXPECT_EQ(restored.eliminate_dense_columns, true);
  EXPECT_EQ(restored.pdlp_precision, pdlp_precision_t::MixedPrecision);
  EXPECT_EQ(restored.save_best_primal_so_far, true);
  EXPECT_EQ(restored.first_primal_feasible, true);
}

TEST(MapperRoundtrip, PDLPSettingsIterationLimitSentinel)
{
  pdlp_solver_settings_t<int32_t, double> orig;
  orig.iteration_limit = std::numeric_limits<int32_t>::max();

  cuopt::remote::PDLPSolverSettings pb;
  map_pdlp_settings_to_proto(orig, &pb);
  EXPECT_EQ(pb.iteration_limit(), -1) << "max() should map to -1 sentinel";

  pdlp_solver_settings_t<int32_t, double> restored;
  auto default_limit = restored.iteration_limit;
  map_proto_to_pdlp_settings(pb, restored);
  EXPECT_EQ(restored.iteration_limit, default_limit) << "Negative sentinel should keep default";
}
