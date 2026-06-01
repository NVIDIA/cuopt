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
#include "server/grpc_field_element_size.hpp"

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
  orig.symmetry        = 2;  // orbital fixing + lexical reduction

  // Semi-continuous variables
  orig.semi_continuous_big_m = 7.5e9;  // not the default 1e10, to detect overwrite-on-decode

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

  // Heuristic hyper-parameters (mip_heuristics_hyper_params_t).
  // Set every field to a value distinct from its C++ default so a missed
  // mapping line would produce a default-valued mismatch on decode.
  orig.heuristic_params.population_size                    = 64;     // default 32
  orig.heuristic_params.num_cpufj_threads                  = 4;      // default 8
  orig.heuristic_params.presolve_time_ratio                = 0.2;    // default 0.1
  orig.heuristic_params.presolve_max_time                  = 45.0;   // default 60.0
  orig.heuristic_params.root_lp_time_ratio                 = 0.25;   // default 0.1
  orig.heuristic_params.root_lp_max_time                   = 7.5;    // default 15.0
  orig.heuristic_params.rins_time_limit                    = 4.0;    // default 3.0
  orig.heuristic_params.rins_max_time_limit                = 25.0;   // default 20.0
  orig.heuristic_params.rins_fix_rate                      = 0.75;   // default 0.5
  orig.heuristic_params.stagnation_trigger                 = 5;      // default 3
  orig.heuristic_params.max_iterations_without_improvement = 12;     // default 8
  orig.heuristic_params.initial_infeasibility_weight       = 500.0;  // default 1000.0
  orig.heuristic_params.n_of_minimums_for_exit             = 9000;   // default 7000
  orig.heuristic_params.enabled_recombiners                = 7;      // default 15 (bitmask)
  orig.heuristic_params.cycle_detection_length             = 40;     // default 30
  orig.heuristic_params.relaxed_lp_time_limit              = 2.5;    // default 1.0
  orig.heuristic_params.related_vars_time_limit            = 45.0;   // default 30.0

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
  EXPECT_EQ(restored.symmetry, 2);

  // Semi-continuous variables
  EXPECT_DOUBLE_EQ(restored.semi_continuous_big_m, 7.5e9);

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

  // Heuristic hyper-parameters
  EXPECT_EQ(restored.heuristic_params.population_size, 64);
  EXPECT_EQ(restored.heuristic_params.num_cpufj_threads, 4);
  EXPECT_DOUBLE_EQ(restored.heuristic_params.presolve_time_ratio, 0.2);
  EXPECT_DOUBLE_EQ(restored.heuristic_params.presolve_max_time, 45.0);
  EXPECT_DOUBLE_EQ(restored.heuristic_params.root_lp_time_ratio, 0.25);
  EXPECT_DOUBLE_EQ(restored.heuristic_params.root_lp_max_time, 7.5);
  EXPECT_DOUBLE_EQ(restored.heuristic_params.rins_time_limit, 4.0);
  EXPECT_DOUBLE_EQ(restored.heuristic_params.rins_max_time_limit, 25.0);
  EXPECT_DOUBLE_EQ(restored.heuristic_params.rins_fix_rate, 0.75);
  EXPECT_EQ(restored.heuristic_params.stagnation_trigger, 5);
  EXPECT_EQ(restored.heuristic_params.max_iterations_without_improvement, 12);
  EXPECT_DOUBLE_EQ(restored.heuristic_params.initial_infeasibility_weight, 500.0);
  EXPECT_EQ(restored.heuristic_params.n_of_minimums_for_exit, 9000);
  EXPECT_EQ(restored.heuristic_params.enabled_recombiners, 7);
  EXPECT_EQ(restored.heuristic_params.cycle_detection_length, 40);
  EXPECT_DOUBLE_EQ(restored.heuristic_params.relaxed_lp_time_limit, 2.5);
  EXPECT_DOUBLE_EQ(restored.heuristic_params.related_vars_time_limit, 45.0);
}

TEST(MapperRoundtrip, MIPSettingsSymmetryClampsOutOfRange)
{
  // The local-solve binding (solver_settings.cu) restricts symmetry to [-1, 2].
  // The mapper applies the same range to defend against buggy/untrusted clients.
  for (int bad_value : {-2, 3, 99, std::numeric_limits<int32_t>::min()}) {
    cuopt::remote::MIPSolverSettings pb;
    pb.set_symmetry(bad_value);

    mip_solver_settings_t<int32_t, double> restored;
    restored.symmetry = 0;  // confirm clamp actively overwrites
    map_proto_to_mip_settings(pb, restored);

    EXPECT_EQ(restored.symmetry, -1) << "symmetry=" << bad_value << " should clamp to -1 (default)";
  }

  // In-range values pass through unchanged.
  for (int good_value : {-1, 0, 1, 2}) {
    cuopt::remote::MIPSolverSettings pb;
    pb.set_symmetry(good_value);

    mip_solver_settings_t<int32_t, double> restored;
    map_proto_to_mip_settings(pb, restored);

    EXPECT_EQ(restored.symmetry, good_value)
      << "symmetry=" << good_value << " should round-trip unchanged";
  }
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

  orig.time_limit                   = 99.5;
  orig.iteration_limit              = 10000;
  orig.log_to_console               = false;
  orig.detect_infeasibility         = true;
  orig.strict_infeasibility         = true;
  orig.pdlp_solver_mode             = pdlp_solver_mode_t::Fast1;
  orig.method                       = method_t::Barrier;
  orig.presolver                    = presolver_t::Default;
  orig.dual_postsolve               = true;
  orig.crossover                    = true;
  orig.num_gpus                     = 4;
  orig.per_constraint_residual      = true;
  orig.cudss_deterministic          = true;
  orig.folding                      = 1;
  orig.augmented                    = 1;
  orig.dualize                      = 1;
  orig.ordering                     = 2;
  orig.barrier_dual_initial_point   = 1;
  orig.eliminate_dense_columns      = true;
  orig.barrier_iterative_refinement = false;  // not the default true, to detect overwrite-on-decode
  orig.barrier_step_scale           = 0.75;   // not the default 0.9
  orig.pdlp_precision               = pdlp_precision_t::MixedPrecision;
  orig.save_best_primal_so_far      = true;
  orig.first_primal_feasible        = true;

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
  EXPECT_EQ(restored.barrier_iterative_refinement, false);
  EXPECT_DOUBLE_EQ(restored.barrier_step_scale, 0.75);
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

// ───────────────────────────────────────────────────────────────────────────
// Proto3 `optional` presence handling
// ───────────────────────────────────────────────────────────────────────────
//
// A handful of bool settings have a C++ default of `true` but live on the wire
// in a proto3 message. Without `optional`, an omitted field decodes as the
// proto3 zero (`false`) and the mapper would silently overwrite the C++
// default. The codegen emits `optional <type>` for these fields and
// guards the assignment with `has_<X>()`, so an omitted field preserves the
// solver default. The tests below pin that behavior for the three currently
// converted fields.
TEST(MapperRoundtrip, MIPSettingsProbingOmittedPreservesDefault)
{
  cuopt::remote::MIPSolverSettings pb;  // default-constructed: probing absent

  mip_solver_settings_t<int32_t, double> restored;
  ASSERT_TRUE(restored.probing) << "C++ default is expected to be true";
  restored.probing = false;  // confirm the guard actively skips the assignment
  map_proto_to_mip_settings(pb, restored);
  EXPECT_FALSE(restored.probing)
    << "Omitted optional bool must not overwrite the existing struct value; "
       "the in-class default would be restored only if the struct was fresh";

  mip_solver_settings_t<int32_t, double> fresh;
  map_proto_to_mip_settings(pb, fresh);
  EXPECT_TRUE(fresh.probing) << "Omitted optional bool must preserve the C++ default `true`";
}

TEST(MapperRoundtrip, MIPSettingsProbingExplicitFalseRoundtrips)
{
  cuopt::remote::MIPSolverSettings pb;
  pb.set_probing(false);
  ASSERT_TRUE(pb.has_probing()) << "set_probing must mark presence on optional field";

  mip_solver_settings_t<int32_t, double> restored;
  map_proto_to_mip_settings(pb, restored);
  EXPECT_FALSE(restored.probing) << "Explicit false must apply";
}

TEST(MapperRoundtrip, PDLPSettingsDualPostsolveOmittedPreservesDefault)
{
  cuopt::remote::PDLPSolverSettings pb;  // default-constructed

  pdlp_solver_settings_t<int32_t, double> fresh;
  ASSERT_TRUE(fresh.dual_postsolve) << "C++ default is expected to be true";
  map_proto_to_pdlp_settings(pb, fresh);
  EXPECT_TRUE(fresh.dual_postsolve) << "Omitted optional bool must preserve the C++ default `true`";
}

TEST(MapperRoundtrip, PDLPSettingsDualPostsolveExplicitFalseRoundtrips)
{
  cuopt::remote::PDLPSolverSettings pb;
  pb.set_dual_postsolve(false);
  ASSERT_TRUE(pb.has_dual_postsolve());

  pdlp_solver_settings_t<int32_t, double> restored;
  map_proto_to_pdlp_settings(pb, restored);
  EXPECT_FALSE(restored.dual_postsolve);
}

TEST(MapperRoundtrip, PDLPSettingsBarrierIterativeRefinementOmittedPreservesDefault)
{
  cuopt::remote::PDLPSolverSettings pb;

  pdlp_solver_settings_t<int32_t, double> fresh;
  ASSERT_TRUE(fresh.barrier_iterative_refinement);
  map_proto_to_pdlp_settings(pb, fresh);
  EXPECT_TRUE(fresh.barrier_iterative_refinement)
    << "Omitted optional bool must preserve the C++ default `true`";
}

TEST(MapperRoundtrip, PDLPSettingsBarrierIterativeRefinementExplicitFalseRoundtrips)
{
  cuopt::remote::PDLPSolverSettings pb;
  pb.set_barrier_iterative_refinement(false);
  ASSERT_TRUE(pb.has_barrier_iterative_refinement());

  pdlp_solver_settings_t<int32_t, double> restored;
  map_proto_to_pdlp_settings(pb, restored);
  EXPECT_FALSE(restored.barrier_iterative_refinement);
}

// Wide-coverage sanity: a default-constructed proto (no fields touched on the
// wire) must, after the mapper, leave every C++ scalar settings field at its
// in-class default. Spot-checks a representative cross-section of the fields
// converted to `optional` in field_registry.yaml — booleans with default true,
// numeric tolerances with non-zero defaults, time/work limits at infinity,
// int32 knobs whose default is -1 or +N, and a handful of heuristic_params
// values. If a field is ever added with a non-zero C++ default but without
// `optional: true` in the registry, this test will fail and point at the gap.
TEST(MapperRoundtrip, PDLPSettingsDefaultProtoPreservesAllCppDefaults)
{
  cuopt::remote::PDLPSolverSettings pb;
  pdlp_solver_settings_t<int32_t, double> fresh;
  pdlp_solver_settings_t<int32_t, double> after = fresh;
  map_proto_to_pdlp_settings(pb, after);

  // Tolerances (all 1e-4 / 1e-10 by C++ default).
  EXPECT_DOUBLE_EQ(after.tolerances.absolute_gap_tolerance,
                   fresh.tolerances.absolute_gap_tolerance);
  EXPECT_DOUBLE_EQ(after.tolerances.relative_gap_tolerance,
                   fresh.tolerances.relative_gap_tolerance);
  EXPECT_DOUBLE_EQ(after.tolerances.absolute_primal_tolerance,
                   fresh.tolerances.absolute_primal_tolerance);
  EXPECT_DOUBLE_EQ(after.tolerances.relative_primal_tolerance,
                   fresh.tolerances.relative_primal_tolerance);
  EXPECT_DOUBLE_EQ(after.tolerances.absolute_dual_tolerance,
                   fresh.tolerances.absolute_dual_tolerance);
  EXPECT_DOUBLE_EQ(after.tolerances.relative_dual_tolerance,
                   fresh.tolerances.relative_dual_tolerance);
  EXPECT_DOUBLE_EQ(after.tolerances.primal_infeasible_tolerance,
                   fresh.tolerances.primal_infeasible_tolerance);
  EXPECT_DOUBLE_EQ(after.tolerances.dual_infeasible_tolerance,
                   fresh.tolerances.dual_infeasible_tolerance);
  // Limits.
  EXPECT_EQ(after.time_limit, fresh.time_limit) << "time_limit default (infinity) preserved";
  // Bools with non-zero defaults.
  EXPECT_EQ(after.log_to_console, fresh.log_to_console);
  EXPECT_EQ(after.dual_postsolve, fresh.dual_postsolve);
  EXPECT_EQ(after.eliminate_dense_columns, fresh.eliminate_dense_columns);
  EXPECT_EQ(after.barrier_iterative_refinement, fresh.barrier_iterative_refinement);
  // Numeric defaults != 0.
  EXPECT_EQ(after.num_gpus, fresh.num_gpus);
  EXPECT_EQ(after.folding, fresh.folding);
  EXPECT_EQ(after.augmented, fresh.augmented);
  EXPECT_EQ(after.dualize, fresh.dualize);
  EXPECT_EQ(after.ordering, fresh.ordering);
  EXPECT_EQ(after.barrier_dual_initial_point, fresh.barrier_dual_initial_point);
  EXPECT_DOUBLE_EQ(after.barrier_step_scale, fresh.barrier_step_scale);
  // Enum-int32 fields (post-decode clamping defends out-of-range; default `0`
  // on the wire is in-range so the clamp does not fire, but the `optional`
  // guard prevents the assignment entirely and the C++ default survives).
  EXPECT_EQ(static_cast<int>(after.presolver), static_cast<int>(fresh.presolver));
  EXPECT_EQ(static_cast<int>(after.pdlp_precision), static_cast<int>(fresh.pdlp_precision));
  // True-enum field: the proto3 enum zero is `Stable1` (first listed value)
  // and the C++ default is `Stable3`. Without `optional` on this field the
  // mapper would silently apply `Stable1`; the `has_pdlp_solver_mode()`
  // guard preserves the C++ default.
  EXPECT_EQ(static_cast<int>(after.pdlp_solver_mode), static_cast<int>(fresh.pdlp_solver_mode));
  EXPECT_EQ(static_cast<int>(fresh.pdlp_solver_mode), static_cast<int>(pdlp_solver_mode_t::Stable3))
    << "pre-condition: C++ default is expected to be Stable3";
}

TEST(MapperRoundtrip, MIPSettingsDefaultProtoPreservesAllCppDefaults)
{
  cuopt::remote::MIPSolverSettings pb;
  mip_solver_settings_t<int32_t, double> fresh;
  mip_solver_settings_t<int32_t, double> after = fresh;
  map_proto_to_mip_settings(pb, after);

  // Tolerances.
  EXPECT_DOUBLE_EQ(after.tolerances.absolute_mip_gap, fresh.tolerances.absolute_mip_gap);
  EXPECT_DOUBLE_EQ(after.tolerances.relative_mip_gap, fresh.tolerances.relative_mip_gap);
  EXPECT_DOUBLE_EQ(after.tolerances.integrality_tolerance, fresh.tolerances.integrality_tolerance);
  EXPECT_DOUBLE_EQ(after.tolerances.absolute_tolerance, fresh.tolerances.absolute_tolerance);
  EXPECT_DOUBLE_EQ(after.tolerances.relative_tolerance, fresh.tolerances.relative_tolerance);
  EXPECT_DOUBLE_EQ(after.tolerances.presolve_absolute_tolerance,
                   fresh.tolerances.presolve_absolute_tolerance);
  // Limits.
  EXPECT_EQ(after.time_limit, fresh.time_limit);
  EXPECT_EQ(after.work_limit, fresh.work_limit);
  EXPECT_EQ(after.node_limit, fresh.node_limit);  // sentinel path
  // Bools with non-zero defaults.
  EXPECT_EQ(after.log_to_console, fresh.log_to_console);
  EXPECT_EQ(after.probing, fresh.probing);
  // Numeric knobs.
  EXPECT_EQ(after.num_cpu_threads, fresh.num_cpu_threads);
  EXPECT_EQ(after.num_gpus, fresh.num_gpus);
  EXPECT_EQ(after.reliability_branching, fresh.reliability_branching);
  EXPECT_EQ(after.symmetry, fresh.symmetry);  // clamp-defended; default -1
  EXPECT_EQ(after.max_cut_passes, fresh.max_cut_passes);
  EXPECT_EQ(after.mir_cuts, fresh.mir_cuts);
  EXPECT_EQ(after.mixed_integer_gomory_cuts, fresh.mixed_integer_gomory_cuts);
  EXPECT_EQ(after.knapsack_cuts, fresh.knapsack_cuts);
  EXPECT_EQ(after.clique_cuts, fresh.clique_cuts);
  EXPECT_EQ(after.implied_bound_cuts, fresh.implied_bound_cuts);
  EXPECT_EQ(after.strong_chvatal_gomory_cuts, fresh.strong_chvatal_gomory_cuts);
  EXPECT_EQ(after.reduced_cost_strengthening, fresh.reduced_cost_strengthening);
  EXPECT_DOUBLE_EQ(after.cut_change_threshold, fresh.cut_change_threshold);
  EXPECT_DOUBLE_EQ(after.cut_min_orthogonality, fresh.cut_min_orthogonality);
  EXPECT_EQ(after.strong_branching_simplex_iteration_limit,
            fresh.strong_branching_simplex_iteration_limit);
  EXPECT_EQ(after.seed, fresh.seed);
  EXPECT_DOUBLE_EQ(after.semi_continuous_big_m, fresh.semi_continuous_big_m);
  EXPECT_EQ(static_cast<int>(after.presolver), static_cast<int>(fresh.presolver));
  EXPECT_EQ(after.mip_scaling, fresh.mip_scaling);
  // heuristic_params: spot-check one of each kind (int, double).
  EXPECT_EQ(after.heuristic_params.population_size, fresh.heuristic_params.population_size);
  EXPECT_EQ(after.heuristic_params.num_cpufj_threads, fresh.heuristic_params.num_cpufj_threads);
  EXPECT_DOUBLE_EQ(after.heuristic_params.presolve_time_ratio,
                   fresh.heuristic_params.presolve_time_ratio);
  EXPECT_DOUBLE_EQ(after.heuristic_params.presolve_max_time,
                   fresh.heuristic_params.presolve_max_time);
  EXPECT_DOUBLE_EQ(after.heuristic_params.rins_fix_rate, fresh.heuristic_params.rins_fix_rate);
  EXPECT_EQ(after.heuristic_params.enabled_recombiners, fresh.heuristic_params.enabled_recombiners);
  EXPECT_DOUBLE_EQ(after.heuristic_params.initial_infeasibility_weight,
                   fresh.heuristic_params.initial_infeasibility_weight);
}

// ============================================================================
// Quadratic constraints round-trip tests
//
// These exercise the QCQP transport end-to-end at the mapper level:
//  * Unary path:  map_problem_to_proto → map_proto_to_problem
//  * Chunked path: populate_chunked_header_* + build_array_chunk_requests
//                  → reassemble chunks → map_chunked_arrays_to_problem
// ============================================================================

namespace {

using QC = optimization_problem_interface_t<int32_t, double>::quadratic_constraint_t;

// Q is stored COO-style on quadratic_constraint_t: three parallel arrays
// (rows, cols, vals) of the same length, one entry per non-zero in the
// Q matrix block for this row.  Older CSR storage was replaced by the
// SOCP barrier work upstream; the wire format renames track the struct.
QC make_qc(int32_t row_index,
           std::string name,
           char row_type,
           double rhs,
           std::vector<double> lin_vals,
           std::vector<int32_t> lin_idx,
           std::vector<int32_t> q_rows,
           std::vector<int32_t> q_cols,
           std::vector<double> q_vals)
{
  QC qc;
  qc.constraint_row_index = row_index;
  qc.constraint_row_name  = std::move(name);
  qc.constraint_row_type  = row_type;
  qc.rhs_value            = rhs;
  qc.linear_values        = std::move(lin_vals);
  qc.linear_indices       = std::move(lin_idx);
  qc.rows                 = std::move(q_rows);
  qc.cols                 = std::move(q_cols);
  qc.vals                 = std::move(q_vals);
  return qc;
}

void expect_qc_equal(const QC& a, const QC& b)
{
  EXPECT_EQ(a.constraint_row_index, b.constraint_row_index);
  EXPECT_EQ(a.constraint_row_name, b.constraint_row_name);
  // Compare via int so failure messages render unprintable bytes legibly.
  EXPECT_EQ(static_cast<int>(static_cast<unsigned char>(a.constraint_row_type)),
            static_cast<int>(static_cast<unsigned char>(b.constraint_row_type)));
  EXPECT_DOUBLE_EQ(a.rhs_value, b.rhs_value);
  EXPECT_EQ(a.linear_values, b.linear_values);
  EXPECT_EQ(a.linear_indices, b.linear_indices);
  EXPECT_EQ(a.rows, b.rows);
  EXPECT_EQ(a.cols, b.cols);
  EXPECT_EQ(a.vals, b.vals);
}

// Reassemble the chunk requests produced by build_array_chunk_requests into
// the same (arrays, container_arrays) shape that map_chunked_arrays_to_problem
// consumes — i.e. what the worker reconstructs from the pipe in production.
// This is the only place in the test where we mirror server-side wire logic.
void assemble_chunk_requests(
  const std::vector<cuopt::remote::SendArrayChunkRequest>& reqs,
  std::map<int32_t, std::vector<uint8_t>>& arrays,
  std::map<container_array_key_t, std::vector<uint8_t>>& container_arrays)
{
  for (const auto& req : reqs) {
    const auto& ac             = req.chunk();
    int32_t fid                = ac.field_id();
    int64_t total              = ac.total_elements();
    int64_t elem_size          = 0;
    std::vector<uint8_t>* dest = nullptr;
    if (ac.has_container_field_num()) {
      container_array_key_t key{ac.container_field_num(), ac.container_index(), fid};
      dest      = &container_arrays[key];
      elem_size = array_field_element_size(key.container_field_num, key.field_id);
    } else {
      dest      = &arrays[fid];
      elem_size = array_field_element_size(-1, fid);
    }
    ASSERT_GT(elem_size, 0) << "Unknown element size for chunk";
    ASSERT_GE(total, 0) << "Negative total_elements in chunk";
    ASSERT_GE(ac.element_offset(), 0) << "Negative element_offset in chunk";
    auto needed = static_cast<size_t>(total) * static_cast<size_t>(elem_size);
    if (dest->size() < needed) dest->resize(needed);
    auto byte_offset = static_cast<size_t>(ac.element_offset()) * static_cast<size_t>(elem_size);
    ASSERT_LE(byte_offset, dest->size()) << "Chunk byte_offset exceeds destination size";
    ASSERT_LE(ac.data().size(), dest->size() - byte_offset)
      << "Chunk payload exceeds destination bounds";
    std::memcpy(dest->data() + byte_offset, ac.data().data(), ac.data().size());
  }
}

// A minimal LP scaffold used to satisfy the optimization-problem-level
// invariants that populate_chunked_header_lp inspects on its way to populating
// QC fields.  The actual coefficients are not exercised by these tests; we
// only care about QC round-trip.
void seed_minimal_problem(cpu_optimization_problem_t<int32_t, double>& problem)
{
  std::vector<double> obj    = {1.0, 2.0, 3.0};
  std::vector<double> var_lb = {0.0, 0.0, 0.0};
  std::vector<double> var_ub = {10.0, 10.0, 10.0};
  std::vector<double> A_vals = {1.0, 1.0, 1.0};
  std::vector<int32_t> A_idx = {0, 1, 2};
  std::vector<int32_t> A_off = {0, 3};
  std::vector<double> b_lb   = {1.0};
  std::vector<double> b_ub   = {1e20};
  problem.set_objective_coefficients(obj.data(), 3);
  problem.set_variable_lower_bounds(var_lb.data(), 3);
  problem.set_variable_upper_bounds(var_ub.data(), 3);
  problem.set_csr_constraint_matrix(A_vals.data(), 3, A_idx.data(), 3, A_off.data(), 2);
  problem.set_constraint_lower_bounds(b_lb.data(), 1);
  problem.set_constraint_upper_bounds(b_ub.data(), 1);
}

}  // namespace

TEST(MapperRoundtrip, QuadraticConstraintsUnaryPath)
{
  cpu_optimization_problem_t<int32_t, double> orig;
  seed_minimal_problem(orig);

  std::vector<QC> qcs;
  qcs.push_back(make_qc(/*row_index=*/0,
                        "qc_row_0",
                        'L',
                        4.5,
                        /*lin_vals=*/{1.5, -2.5},
                        /*lin_idx=*/{0, 2},
                        /*q_rows=*/{0, 1, 1},
                        /*q_cols=*/{0, 1, 2},
                        /*q_vals=*/{2.0, 0.5, 3.0}));
  qcs.push_back(make_qc(/*row_index=*/1,
                        "qc_row_1",
                        'G',
                        -7.0,
                        /*lin_vals=*/{0.25, 0.75, 1.0},
                        /*lin_idx=*/{0, 1, 2},
                        /*q_rows=*/{2},
                        /*q_cols=*/{2},
                        /*q_vals=*/{4.0}));
  orig.set_quadratic_constraints(qcs);
  ASSERT_TRUE(orig.has_quadratic_constraints());

  cuopt::remote::OptimizationProblem pb;
  map_problem_to_proto(orig, &pb);

  ASSERT_EQ(pb.quadratic_constraints_size(), 2);
  EXPECT_EQ(pb.quadratic_constraints(0).constraint_row_name(), "qc_row_0");
  EXPECT_EQ(pb.quadratic_constraints(0).linear_values_size(), 2);
  EXPECT_EQ(pb.quadratic_constraints(0).vals_size(), 3);
  EXPECT_EQ(pb.quadratic_constraints(1).vals_size(), 1);

  cpu_optimization_problem_t<int32_t, double> restored;
  map_proto_to_problem(pb, restored);

  ASSERT_TRUE(restored.has_quadratic_constraints());
  const auto& got = restored.get_quadratic_constraints();
  ASSERT_EQ(got.size(), qcs.size());
  for (size_t i = 0; i < qcs.size(); ++i) {
    SCOPED_TRACE("QC entry " + std::to_string(i));
    expect_qc_equal(qcs[i], got[i]);
  }
}

TEST(MapperRoundtrip, QuadraticConstraintsChunkedPath)
{
  cpu_optimization_problem_t<int32_t, double> orig;
  seed_minimal_problem(orig);

  // Build QC entries with arrays large enough that build_array_chunk_requests
  // with a small chunk_size_bytes is forced to split them across multiple
  // chunks, exercising the slow-path stitching inside the container code.
  // Q is COO so rows/cols/vals are three parallel arrays of equal length.
  constexpr int n0_linear = 64;   // 64 doubles = 512 bytes
  constexpr int n0_q      = 100;  // 100 COO entries (rows/cols int, vals double)
  constexpr int n1_linear = 32;
  std::vector<double> lv0(n0_linear);
  std::vector<int32_t> li0(n0_linear);
  std::vector<int32_t> qr0(n0_q);
  std::vector<int32_t> qc0(n0_q);
  std::vector<double> qv0(n0_q);
  for (int i = 0; i < n0_linear; ++i) {
    lv0[i] = 0.5 * i + 1.0;
    li0[i] = i;
  }
  for (int i = 0; i < n0_q; ++i) {
    qr0[i] = i % n0_linear;
    qc0[i] = (i + 7) % n0_linear;
    qv0[i] = -0.25 * i + 7.0;
  }

  std::vector<double> lv1(n1_linear);
  std::vector<int32_t> li1(n1_linear);
  for (int i = 0; i < n1_linear; ++i) {
    lv1[i] = 100.0 + i;
    li1[i] = n1_linear - 1 - i;
  }

  std::vector<QC> qcs;
  qcs.push_back(make_qc(0, "big_qc", 'L', 12.5, lv0, li0, qr0, qc0, qv0));
  qcs.push_back(make_qc(2, "small_qc", 'E', 0.0, lv1, li1, {}, {}, {}));
  orig.set_quadratic_constraints(qcs);

  // 1) Client side: populate header (scalars only) + build chunk requests.
  pdlp_solver_settings_t<int32_t, double> settings;
  cuopt::remote::ChunkedProblemHeader header;
  populate_chunked_header_lp(orig, settings, &header);

  ASSERT_EQ(header.quadratic_constraints_size(), 2);
  // Per-entry arrays must NOT have ridden the header — they belong on chunks.
  EXPECT_EQ(header.quadratic_constraints(0).linear_values_size(), 0);
  EXPECT_EQ(header.quadratic_constraints(0).vals_size(), 0);

  // Small chunk budget forces at least one container array to split.
  constexpr int64_t kChunkBytes = 96;
  auto requests                 = build_array_chunk_requests(orig, "upload-test", kChunkBytes);

  // Sanity-check that at least one container chunk was produced and that the
  // big_qc linear_values (512 B) was split into multiple chunks.
  size_t container_chunks = 0;
  size_t big_lv_chunks    = 0;
  for (const auto& r : requests) {
    const auto& ac = r.chunk();
    if (!ac.has_container_field_num()) continue;
    ++container_chunks;
    if (ac.container_index() == 0 && ac.field_id() == 0) ++big_lv_chunks;
  }
  EXPECT_GT(container_chunks, 0u);
  EXPECT_GT(big_lv_chunks, 1u) << "Expected multi-chunk split for big_qc.linear_values";

  // 2) Server side: reassemble chunks into raw byte maps and reconstruct.
  std::map<int32_t, std::vector<uint8_t>> arrays;
  std::map<container_array_key_t, std::vector<uint8_t>> container_arrays;
  assemble_chunk_requests(requests, arrays, container_arrays);

  cpu_optimization_problem_t<int32_t, double> restored;
  map_chunked_arrays_to_problem(header, arrays, container_arrays, restored);

  ASSERT_TRUE(restored.has_quadratic_constraints());
  const auto& got = restored.get_quadratic_constraints();
  ASSERT_EQ(got.size(), qcs.size());
  for (size_t i = 0; i < qcs.size(); ++i) {
    SCOPED_TRACE("QC entry " + std::to_string(i));
    expect_qc_equal(qcs[i], got[i]);
  }
}

TEST(MapperRoundtrip, QuadraticConstraintsEmpty)
{
  cpu_optimization_problem_t<int32_t, double> orig;
  seed_minimal_problem(orig);
  ASSERT_FALSE(orig.has_quadratic_constraints());

  // Unary path: proto carries zero entries and decode leaves QC unset.
  cuopt::remote::OptimizationProblem pb;
  map_problem_to_proto(orig, &pb);
  EXPECT_EQ(pb.quadratic_constraints_size(), 0);

  cpu_optimization_problem_t<int32_t, double> restored_unary;
  map_proto_to_problem(pb, restored_unary);
  EXPECT_FALSE(restored_unary.has_quadratic_constraints());

  // Chunked path: header carries zero entries, build_array_chunk_requests
  // produces no container chunks, and the worker-side mapper leaves QC unset.
  pdlp_solver_settings_t<int32_t, double> settings;
  cuopt::remote::ChunkedProblemHeader header;
  populate_chunked_header_lp(orig, settings, &header);
  EXPECT_EQ(header.quadratic_constraints_size(), 0);

  auto requests = build_array_chunk_requests(orig, "upload-empty", /*chunk_size_bytes=*/1024);
  for (const auto& r : requests) {
    EXPECT_FALSE(r.chunk().has_container_field_num())
      << "No container chunks should be emitted when there are no QC entries";
  }

  std::map<int32_t, std::vector<uint8_t>> arrays;
  std::map<container_array_key_t, std::vector<uint8_t>> container_arrays;
  assemble_chunk_requests(requests, arrays, container_arrays);
  EXPECT_TRUE(container_arrays.empty());

  cpu_optimization_problem_t<int32_t, double> restored_chunked;
  map_chunked_arrays_to_problem(header, arrays, container_arrays, restored_chunked);
  EXPECT_FALSE(restored_chunked.has_quadratic_constraints());
}

TEST(MapperRoundtrip, QuadraticConstraintsRowTypeLenient)
{
  // Verify that constraint_row_type survives any byte value through the
  // int32 wire encoding without rejection.  cpu_optimization_problem stores
  // it as a `char` and the gRPC transport is intentionally lenient so the
  // remote path matches the local-solve binding, which accepts whatever the
  // C++ caller supplies (e.g. mixed-case 'L'/'l', 0, or any extended byte).
  cpu_optimization_problem_t<int32_t, double> orig;
  seed_minimal_problem(orig);

  std::vector<char> row_types = {
    'L', 'G', 'E', 'l', 'g', 'e', '\0', '\x7F', static_cast<char>(0xFF), static_cast<char>(0x80)};
  std::vector<QC> qcs;
  qcs.reserve(row_types.size());
  for (size_t i = 0; i < row_types.size(); ++i) {
    qcs.push_back(make_qc(static_cast<int32_t>(i),
                          "row_" + std::to_string(i),
                          row_types[i],
                          /*rhs=*/static_cast<double>(i),
                          /*lin_vals=*/{1.0},
                          /*lin_idx=*/{0},
                          /*q_rows=*/{},
                          /*q_cols=*/{},
                          /*q_vals=*/{}));
  }
  orig.set_quadratic_constraints(qcs);

  cuopt::remote::OptimizationProblem pb;
  map_problem_to_proto(orig, &pb);
  ASSERT_EQ(pb.quadratic_constraints_size(), static_cast<int>(row_types.size()));

  cpu_optimization_problem_t<int32_t, double> restored;
  map_proto_to_problem(pb, restored);

  ASSERT_TRUE(restored.has_quadratic_constraints());
  const auto& got = restored.get_quadratic_constraints();
  ASSERT_EQ(got.size(), qcs.size());
  for (size_t i = 0; i < qcs.size(); ++i) {
    EXPECT_EQ(static_cast<int>(static_cast<unsigned char>(got[i].constraint_row_type)),
              static_cast<int>(static_cast<unsigned char>(row_types[i])))
      << "Mismatch at index " << i;
  }
}
