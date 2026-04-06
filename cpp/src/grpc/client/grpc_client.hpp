/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>

#include "../cuopt_default_grpc_port.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Forward declarations for gRPC types (to avoid exposing gRPC headers in public API)
namespace grpc {
class Channel;
}

namespace cuopt::remote {
class CuOptRemoteService;
class ChunkedProblemHeader;
class ChunkedResultHeader;
class ResultResponse;
class SubmitJobRequest;
}  // namespace cuopt::remote

namespace cuopt::linear_programming {

// Forward declarations for test helper functions (implemented in grpc_client.cpp)
void grpc_test_inject_mock_stub(class grpc_client_t& client, std::shared_ptr<void> stub);
void grpc_test_mark_as_connected(class grpc_client_t& client);

/**
 * @brief Configuration options for the gRPC client
 *
 * Large Problem Handling:
 * - Small problems use unary SubmitJob (single message).  If the estimated
 *   serialized size exceeds 75% of max_message_bytes, the client automatically
 *   switches to the chunked array protocol: StartChunkedUpload + N × SendArrayChunk
 *   + FinishChunkedUpload (all unary RPCs). This bypasses the protobuf 2GB limit and
 *   reduces peak memory usage.
 * - chunk_size_bytes controls the payload size of individual SendArrayChunk calls.
 * - Result retrieval uses chunked download for results exceeding max_message_bytes.
 */
struct grpc_client_config_t {
  std::string server_address = std::string("localhost:") + std::to_string(cuopt_default_grpc_port);
  int poll_interval_ms       = 1000;   // How often to poll for job status
  int timeout_seconds        = 0;      // Max time to wait for job completion (0 = no limit)
  bool stream_logs           = false;  // Whether to stream logs from server
  std::function<void(const std::string&)> log_callback = nullptr;  // Called for each log line

  // Incumbent callback for MIP solves — invoked each time the server finds a
  // new best-feasible (incumbent) solution.  Parameters: index, objective value,
  // solution vector.  Return true to continue solving, or false to request
  // early termination (e.g. the objective is good enough for the caller's
  // purposes).
  std::function<bool(int64_t index, double objective, const std::vector<double>& solution)>
    incumbent_callback           = nullptr;
  int incumbent_poll_interval_ms = 1000;  // How often to poll for new incumbents

  // TLS configuration
  bool enable_tls = false;
  std::string tls_root_certs;   // PEM-encoded root CA certificates (for verifying server)
  std::string tls_client_cert;  // PEM-encoded client certificate (for mTLS)
  std::string tls_client_key;   // PEM-encoded client private key (for mTLS)

  // gRPC max message size (used for unary SubmitJob and result download decisions).
  // Clamped at construction to [4 MiB, 2 GiB - 1 MiB] (protobuf serialization limit).
  int64_t max_message_bytes = 256LL * 1024 * 1024;  // 256 MiB

  // Chunk size for chunked array upload and chunked result download.
  int64_t chunk_size_bytes = 16LL * 1024 * 1024;  // 16 MiB

  // gRPC keepalive — periodic HTTP/2 PINGs to detect dead connections.
  int keepalive_time_ms    = 30000;  // send PING every 30s of inactivity
  int keepalive_timeout_ms = 10000;  // wait 10s for PONG before declaring dead

  // --- Test / debug options (not intended for production use) -----------------

  // Receives internal client debug messages (for test verification).
  std::function<void(const std::string&)> debug_log_callback = nullptr;

  // Enable debug / throughput logging to stderr.
  // Controlled by CUOPT_GRPC_DEBUG env var (0|1). Default: off.
  bool enable_debug_log = false;

  // Override for the chunked upload threshold (bytes). Normally computed
  // automatically as 75% of max_message_bytes.  Set to 0 to force chunked
  // upload for all problems, or a positive value to override.  -1 = auto.
  int64_t chunked_array_threshold_bytes = -1;
};

/**
 * @brief Job status enum (transport-agnostic)
 */
enum class job_status_t { QUEUED, PROCESSING, COMPLETED, FAILED, CANCELLED, NOT_FOUND };

/**
 * @brief Convert job status to string
 */
inline const char* job_status_to_string(job_status_t status)
{
  switch (status) {
    case job_status_t::QUEUED: return "QUEUED";
    case job_status_t::PROCESSING: return "PROCESSING";
    case job_status_t::COMPLETED: return "COMPLETED";
    case job_status_t::FAILED: return "FAILED";
    case job_status_t::CANCELLED: return "CANCELLED";
    case job_status_t::NOT_FOUND: return "NOT_FOUND";
    default: return "UNKNOWN";
  }
}

/**
 * @brief Result of a job status check
 */
struct job_status_result_t {
  bool success = false;
  std::string error_message;
  job_status_t status = job_status_t::NOT_FOUND;
  std::string message;
  int64_t result_size_bytes = 0;
};

/**
 * @brief Result of a submit operation (job ID)
 */
struct submit_result_t {
  bool success = false;
  std::string error_message;
  std::string job_id;
};

/**
 * @brief Result of a cancel operation
 */
struct cancel_result_t {
  bool success = false;
  std::string error_message;
  job_status_t job_status = job_status_t::NOT_FOUND;
  std::string message;
};

/**
 * @brief Incumbent solution entry
 */
struct incumbent_t {
  int64_t index    = 0;
  double objective = 0.0;
  std::vector<double> assignment;
};

/**
 * @brief Result of get incumbents operation
 */
struct incumbents_result_t {
  bool success = false;
  std::string error_message;
  std::vector<incumbent_t> incumbents;
  int64_t next_index = 0;
  bool job_complete  = false;
};

/**
 * @brief Result of a remote solve operation
 */
template <typename i_t, typename f_t>
struct remote_lp_result_t {
  bool success = false;
  std::string error_message;
  std::unique_ptr<cpu_lp_solution_t<i_t, f_t>> solution;
};

template <typename i_t, typename f_t>
struct remote_mip_result_t {
  bool success = false;
  std::string error_message;
  std::unique_ptr<cpu_mip_solution_t<i_t, f_t>> solution;
};

/**
 * @brief gRPC client for remote cuOpt solving
 *
 * This class provides a high-level interface for submitting optimization problems
 * to a remote cuopt_grpc_server and retrieving results. It handles:
 * - Connection management
 * - Job submission
 * - Status polling
 * - Optional log streaming
 * - Result retrieval and parsing
 *
 * Usage:
 * @code
 * grpc_client_t client;  // default server: localhost:<cuopt_default_grpc_port>
 * if (!client.connect()) { ... handle error ... }
 *
 * auto result = client.solve_lp(problem, settings);
 * if (result.success) {
 *   // Use result.solution
 * }
 * @endcode
 *
 * This class is designed to be used by:
 * - Test clients for validation
 * - solve_lp_remote() and solve_mip_remote() for production use
 */
class grpc_client_t {
  // Allow test helpers to access internal implementation for mock injection
  friend void grpc_test_inject_mock_stub(grpc_client_t&, std::shared_ptr<void>);
  friend void grpc_test_mark_as_connected(grpc_client_t&);

 public:
  /**
   * @brief Construct a gRPC client with configuration
   * @param config Client configuration options
   */
  explicit grpc_client_t(const grpc_client_config_t& config = grpc_client_config_t{});

  /**
   * @brief Construct a gRPC client with just server address
   * @param server_address Server address in "host:port" format
   */
  explicit grpc_client_t(const std::string& server_address);

  ~grpc_client_t();

  // Non-copyable, non-movable (due to atomic member and thread)
  grpc_client_t(const grpc_client_t&)            = delete;
  grpc_client_t& operator=(const grpc_client_t&) = delete;
  grpc_client_t(grpc_client_t&&)                 = delete;
  grpc_client_t& operator=(grpc_client_t&&)      = delete;

  /**
   * @brief Connect to the gRPC server
   * @return true if connection successful
   */
  bool connect();

  /**
   * @brief Check if connected to server
   */
  bool is_connected() const;

  /**
   * @brief Solve an LP problem remotely
   *
   * This is a blocking call that:
   * 1. Submits the problem to the server
   * 2. Polls for completion (with optional log streaming)
   * 3. Retrieves and parses the result
   *
   * @param problem The CPU optimization problem to solve
   * @param settings Solver settings
   * @return Result containing success status and solution (if successful)
   */
  template <typename i_t, typename f_t>
  remote_lp_result_t<i_t, f_t> solve_lp(const cpu_optimization_problem_t<i_t, f_t>& problem,
                                        const pdlp_solver_settings_t<i_t, f_t>& settings);

  /**
   * @brief Solve a MIP problem remotely
   *
   * This is a blocking call that:
   * 1. Submits the problem to the server
   * 2. Polls for completion (with optional log streaming)
   * 3. Retrieves and parses the result
   *
   * @param problem The CPU optimization problem to solve
   * @param settings Solver settings
   * @param enable_incumbents Whether to enable incumbent solution streaming
   * @return Result containing success status and solution (if successful)
   */
  template <typename i_t, typename f_t>
  remote_mip_result_t<i_t, f_t> solve_mip(const cpu_optimization_problem_t<i_t, f_t>& problem,
                                          const mip_solver_settings_t<i_t, f_t>& settings,
                                          bool enable_incumbents = false);

  // =========================================================================
  // Async Operations (for manual job management)
  // =========================================================================

  /**
   * @brief Submit an LP problem without waiting for result
   * @return Result containing job_id if successful
   */
  template <typename i_t, typename f_t>
  submit_result_t submit_lp(const cpu_optimization_problem_t<i_t, f_t>& problem,
                            const pdlp_solver_settings_t<i_t, f_t>& settings);

  /**
   * @brief Submit a MIP problem without waiting for result
   * @return Result containing job_id if successful
   */
  template <typename i_t, typename f_t>
  submit_result_t submit_mip(const cpu_optimization_problem_t<i_t, f_t>& problem,
                             const mip_solver_settings_t<i_t, f_t>& settings,
                             bool enable_incumbents = false);

  /**
   * @brief Check status of a submitted job
   * @param job_id The job ID to check
   * @return Status result including job state and optional result size
   */
  job_status_result_t check_status(const std::string& job_id);

  /**
   * @brief Wait for a job to complete (blocking)
   *
   * This is more efficient than polling check_status() but does not
   * return the result - call get_lp_result/get_mip_result afterward.
   *
   * @param job_id The job ID to wait for
   * @return Status result when job completes (COMPLETED, FAILED, or CANCELLED)
   */
  job_status_result_t wait_for_completion(const std::string& job_id);

  /**
   * @brief Get LP result for a completed job
   * @param job_id The job ID
   * @return Result containing solution if successful
   */
  template <typename i_t, typename f_t>
  remote_lp_result_t<i_t, f_t> get_lp_result(const std::string& job_id);

  /**
   * @brief Get MIP result for a completed job
   * @param job_id The job ID
   * @return Result containing solution if successful
   */
  template <typename i_t, typename f_t>
  remote_mip_result_t<i_t, f_t> get_mip_result(const std::string& job_id);

  /**
   * @brief Cancel a running job
   * @param job_id The job ID to cancel
   * @return Cancel result with status
   */
  cancel_result_t cancel_job(const std::string& job_id);

  /**
   * @brief Delete a job and its results from server
   * @param job_id The job ID to delete
   * @return true if deletion successful
   */
  bool delete_job(const std::string& job_id);

  /**
   * @brief Get incumbent solutions for a MIP job
   * @param job_id The job ID
   * @param from_index Start from this incumbent index
   * @param max_count Maximum number to return (0 = no limit)
   * @return Incumbents result
   */
  incumbents_result_t get_incumbents(const std::string& job_id,
                                     int64_t from_index = 0,
                                     int32_t max_count  = 0);

  /**
   * @brief Get the last error message
   */
  const std::string& get_last_error() const { return last_error_; }

  // --- Test / debug public API -----------------------------------------------

  /**
   * @brief Stream logs for a job (blocking until job completes or callback returns false).
   *
   * This is a low-level, synchronous API for test tools and CLI utilities.
   * Production callers should use config_.stream_logs + config_.log_callback
   * instead, which streams logs automatically on a background thread during
   * solve_lp / solve_mip calls.
   *
   * @param job_id The job ID
   * @param from_byte Starting byte offset in log
   * @param callback Called for each log line; return false to stop streaming
   * @return true if streaming completed normally
   */
  bool stream_logs(const std::string& job_id,
                   int64_t from_byte,
                   std::function<bool(const std::string& line, bool job_complete)> callback);

 private:
  struct impl_t;
  std::unique_ptr<impl_t> impl_;

  grpc_client_config_t config_;
  std::string last_error_;

  // Track server-reported max message size (may differ from our config).
  // Accessed from multiple RPC methods; atomic to avoid data races.
  std::atomic<int64_t> server_max_message_bytes_{0};

  // 75% of max_message_bytes — computed at construction time.
  int64_t chunked_array_threshold_bytes_ = 0;

  // Background log streaming for solve_lp / solve_mip (production path).
  // Activated when config_.stream_logs is true and config_.log_callback is set.
  void start_log_streaming(const std::string& job_id);
  void stop_log_streaming();

  // Shared polling loop used by solve_lp and solve_mip.
  struct poll_result_t {
    bool completed             = false;
    bool cancelled_by_callback = false;
    std::string error_message;
  };
  poll_result_t poll_for_completion(const std::string& job_id);

  std::unique_ptr<std::thread> log_thread_;
  std::atomic<bool> stop_logs_{false};
  mutable std::mutex log_context_mutex_;
  // Points to the grpc::ClientContext* of the in-flight StreamLogs RPC (if
  // any).  Typed as void* to avoid exposing grpc headers in the public API.
  // Protected by log_context_mutex_; stop_log_streaming() calls TryCancel()
  // through this pointer to unblock a stuck reader->Read().
  void* active_log_context_ = nullptr;

  // =========================================================================
  // Result Retrieval Support
  // =========================================================================

  /**
   * @brief Result from get_result_or_download: either a unary ResultResponse or
   *        a chunked header + raw arrays map. Exactly one variant is populated.
   */
  struct downloaded_result_t {
    bool was_chunked = false;

    // Populated when was_chunked == false (unary path).
    std::unique_ptr<cuopt::remote::ResultResponse> response;

    // Populated when was_chunked == true (chunked path).
    std::unique_ptr<cuopt::remote::ChunkedResultHeader> chunked_header;
    std::map<int32_t, std::vector<uint8_t>> chunked_arrays;
  };

  /**
   * @brief Get result, choosing unary GetResult or chunked download based on size.
   *
   * Returns a downloaded_result_t with either the unary ResultResponse or the
   * chunked header + arrays map populated.
   */
  bool get_result_or_download(const std::string& job_id, downloaded_result_t& result_out);

  /**
   * @brief Download result via chunked unary RPCs (StartChunkedDownload +
   *        N × GetResultChunk + FinishChunkedDownload).
   */
  bool download_chunked_result(const std::string& job_id, downloaded_result_t& result_out);

  // =========================================================================
  // Chunked Array Upload (for large problems)
  // =========================================================================

  /**
   * @brief Submit a SubmitJobRequest via unary SubmitJob RPC.
   */
  bool submit_unary(const cuopt::remote::SubmitJobRequest& request, std::string& job_id_out);

  /**
   * @brief Upload a problem using chunked array RPCs (StartChunkedUpload +
   *        N × SendArrayChunk + FinishChunkedUpload).
   */
  template <typename i_t, typename f_t>
  bool upload_chunked_arrays(const cpu_optimization_problem_t<i_t, f_t>& problem,
                             const cuopt::remote::ChunkedProblemHeader& header,
                             std::string& job_id_out);
};

}  // namespace cuopt::linear_programming
