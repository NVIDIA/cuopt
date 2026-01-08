/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/utilities/remote_serialization.hpp>
#include <cuopt/linear_programming/utilities/remote_solve.hpp>
#include <utilities/logger.hpp>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace cuopt::linear_programming {

namespace {

// Message types for streaming protocol (must match server)
enum class MessageType : uint8_t {
  LOG_MESSAGE = 0,  // Log output from server
  SOLUTION    = 1,  // Final solution data
};

// Check if sync mode is enabled (default is async)
static bool use_sync_mode()
{
  const char* sync_env = std::getenv("CUOPT_REMOTE_USE_SYNC");
  return (sync_env != nullptr && std::string(sync_env) == "1");
}

/**
 * @brief Simple socket client for remote solve with streaming support
 */
class remote_client_t {
 public:
  remote_client_t(const std::string& host, int port) : host_(host), port_(port), sockfd_(-1) {}

  ~remote_client_t() { disconnect(); }

  bool connect()
  {
    if (sockfd_ >= 0) return true;

    sockfd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd_ < 0) {
      CUOPT_LOG_ERROR("[remote_solve] Failed to create socket");
      return false;
    }

    struct hostent* server = gethostbyname(host_.c_str());
    if (server == nullptr) {
      CUOPT_LOG_ERROR("[remote_solve] Unknown host: {}", host_);
      close(sockfd_);
      sockfd_ = -1;
      return false;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    addr.sin_port = htons(port_);

    if (::connect(sockfd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
      CUOPT_LOG_ERROR("[remote_solve] Failed to connect to {}:{}", host_, port_);
      close(sockfd_);
      sockfd_ = -1;
      return false;
    }

    return true;
  }

  void disconnect()
  {
    if (sockfd_ >= 0) {
      close(sockfd_);
      sockfd_ = -1;
    }
  }

  bool send_request(const std::vector<uint8_t>& data)
  {
    if (sockfd_ < 0) return false;

    // Send size first (4 bytes, network byte order)
    uint32_t size = static_cast<uint32_t>(data.size());
    if (!write_all(&size, sizeof(size))) return false;
    if (!write_all(data.data(), data.size())) return false;
    return true;
  }

  /**
   * @brief Receive response with streaming log support.
   *
   * This method reads messages from the server. If the server sends LOG_MESSAGE
   * types, they are printed to the console. When a SOLUTION message is received,
   * the solution data is returned.
   *
   * @param data Output buffer for solution data
   * @param log_to_console If true, print received log messages
   * @return true if solution was received, false on error
   */
  bool receive_streaming_response(std::vector<uint8_t>& data, bool log_to_console)
  {
    if (sockfd_ < 0) return false;

    while (true) {
      // Read message type (1 byte)
      uint8_t msg_type;
      if (!read_all(&msg_type, 1)) {
        // If we can't read the message type, try legacy format
        // (server might not support streaming)
        return receive_response_legacy(data);
      }

      // Read payload size (4 bytes)
      uint32_t payload_size;
      if (!read_all(&payload_size, sizeof(payload_size))) return false;

      // Sanity check
      if (payload_size > 100 * 1024 * 1024) {
        CUOPT_LOG_ERROR("[remote_solve] Message too large: {} bytes", payload_size);
        return false;
      }

      if (static_cast<MessageType>(msg_type) == MessageType::LOG_MESSAGE) {
        // Read and display log message
        if (payload_size > 0) {
          std::vector<char> log_msg(payload_size + 1);
          if (!read_all(log_msg.data(), payload_size)) return false;
          log_msg[payload_size] = '\0';

          if (log_to_console) {
            // Print log message from server (already formatted)
            std::cout << log_msg.data() << std::flush;
          }
        }
      } else if (static_cast<MessageType>(msg_type) == MessageType::SOLUTION) {
        // Read solution data
        data.resize(payload_size);
        if (payload_size > 0) {
          if (!read_all(data.data(), payload_size)) return false;
        }
        return true;
      } else {
        CUOPT_LOG_WARN("[remote_solve] Unknown message type: {}", static_cast<int>(msg_type));
        // Skip unknown message
        if (payload_size > 0) {
          std::vector<uint8_t> skip_buf(payload_size);
          if (!read_all(skip_buf.data(), payload_size)) return false;
        }
      }
    }
  }

  // Legacy response format (non-streaming)
  bool receive_response(std::vector<uint8_t>& data) { return receive_response_legacy(data); }

 private:
  bool receive_response_legacy(std::vector<uint8_t>& data)
  {
    if (sockfd_ < 0) return false;

    // Read size first
    uint32_t size;
    if (!read_all(&size, sizeof(size))) return false;

    // Sanity check
    if (size > 100 * 1024 * 1024) {
      CUOPT_LOG_ERROR("[remote_solve] Response too large: {} bytes", size);
      return false;
    }

    data.resize(size);
    if (!read_all(data.data(), size)) return false;
    return true;
  }

  bool write_all(const void* buf, size_t len)
  {
    const uint8_t* ptr = static_cast<const uint8_t*>(buf);
    size_t remaining   = len;
    while (remaining > 0) {
      ssize_t n = ::write(sockfd_, ptr, remaining);
      if (n <= 0) {
        CUOPT_LOG_ERROR("[remote_solve] Write failed");
        return false;
      }
      ptr += n;
      remaining -= n;
    }
    return true;
  }

  bool read_all(void* buf, size_t len)
  {
    uint8_t* ptr     = static_cast<uint8_t*>(buf);
    size_t remaining = len;
    while (remaining > 0) {
      ssize_t n = ::read(sockfd_, ptr, remaining);
      if (n <= 0) {
        CUOPT_LOG_ERROR("[remote_solve] Read failed");
        return false;
      }
      ptr += n;
      remaining -= n;
    }
    return true;
  }

  std::string host_;
  int port_;
  int sockfd_;
};

//============================================================================
// Async Mode Helpers
//============================================================================

template <typename i_t, typename f_t>
static std::pair<bool, std::string> submit_job(const std::string& host,
                                               int port,
                                               const std::vector<uint8_t>& request_data)
{
  remote_client_t client(host, port);
  if (!client.connect()) { return {false, "Failed to connect to server"}; }

  if (!client.send_request(request_data)) { return {false, "Failed to send request"}; }

  std::vector<uint8_t> response_data;
  if (!client.receive_response(response_data)) { return {false, "Failed to receive response"}; }

  auto serializer = get_serializer<i_t, f_t>();
  std::string job_id;
  std::string error_message;
  if (!serializer->deserialize_submit_response(response_data, job_id, error_message)) {
    return {false, error_message};
  }

  return {true, job_id};
}

/**
 * @brief Retrieve and display buffered logs from the server.
 *
 * @param host Server host
 * @param port Server port
 * @param job_id Job ID
 * @param frombyte Byte offset to start reading from
 * @return std::pair<bool, int64_t> - (job_exists, new_frombyte)
 */
template <typename i_t, typename f_t>
static std::pair<bool, int64_t> get_logs(const std::string& host,
                                         int port,
                                         const std::string& job_id,
                                         int64_t frombyte)
{
  remote_client_t client(host, port);
  if (!client.connect()) { return {false, frombyte}; }

  auto serializer   = get_serializer<i_t, f_t>();
  auto logs_request = serializer->serialize_get_logs_request(job_id, frombyte);

  if (!client.send_request(logs_request)) { return {false, frombyte}; }

  std::vector<uint8_t> response_data;
  if (!client.receive_response(response_data)) { return {false, frombyte}; }

  auto result = serializer->deserialize_logs_response(response_data);

  // Print any new log lines
  for (const auto& line : result.log_lines) {
    std::cout << line << "\n";
  }
  if (!result.log_lines.empty()) { std::cout.flush(); }

  return {result.job_exists, result.nbytes};
}

template <typename i_t, typename f_t>
static bool poll_until_complete(const std::string& host,
                                int port,
                                const std::string& job_id,
                                bool verbose)
{
  auto serializer    = get_serializer<i_t, f_t>();
  using job_status_t = typename remote_serializer_t<i_t, f_t>::job_status_t;

  int64_t log_frombyte = 0;  // Track position in log file

  while (true) {
    // Fetch and display any new log entries
    if (verbose) {
      auto [job_exists, new_frombyte] = get_logs<i_t, f_t>(host, port, job_id, log_frombyte);
      if (job_exists) { log_frombyte = new_frombyte; }
    }

    remote_client_t client(host, port);
    if (!client.connect()) {
      CUOPT_LOG_ERROR("[remote_solve] Failed to connect for status check");
      return false;
    }

    auto status_request = serializer->serialize_status_request(job_id);
    if (!client.send_request(status_request)) {
      CUOPT_LOG_ERROR("[remote_solve] Failed to send status request");
      return false;
    }

    std::vector<uint8_t> response_data;
    if (!client.receive_response(response_data)) {
      CUOPT_LOG_ERROR("[remote_solve] Failed to receive status response");
      return false;
    }

    auto status = serializer->deserialize_status_response(response_data);

    if (status == job_status_t::COMPLETED) {
      // Fetch any remaining log entries
      if (verbose) {
        get_logs<i_t, f_t>(host, port, job_id, log_frombyte);
        CUOPT_LOG_INFO("[remote_solve] Job {} completed", job_id);
      }
      return true;
    } else if (status == job_status_t::FAILED) {
      // Fetch any remaining log entries (may contain error info)
      if (verbose) { get_logs<i_t, f_t>(host, port, job_id, log_frombyte); }
      CUOPT_LOG_ERROR("[remote_solve] Job {} failed", job_id);
      return false;
    } else if (status == job_status_t::NOT_FOUND) {
      CUOPT_LOG_ERROR("[remote_solve] Job {} not found", job_id);
      return false;
    }

    // Job still queued or processing, wait and try again
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

template <typename i_t, typename f_t>
static std::pair<bool, std::vector<uint8_t>> get_result(const std::string& host,
                                                        int port,
                                                        const std::string& job_id)
{
  remote_client_t client(host, port);
  if (!client.connect()) { return {false, {}}; }

  auto serializer     = get_serializer<i_t, f_t>();
  auto result_request = serializer->serialize_get_result_request(job_id);

  if (!client.send_request(result_request)) { return {false, {}}; }

  std::vector<uint8_t> response_data;
  if (!client.receive_response(response_data)) { return {false, {}}; }

  return {true, response_data};
}

template <typename i_t, typename f_t>
static void delete_job(const std::string& host, int port, const std::string& job_id)
{
  remote_client_t client(host, port);
  if (!client.connect()) { return; }

  auto serializer     = get_serializer<i_t, f_t>();
  auto delete_request = serializer->serialize_delete_request(job_id);

  if (!client.send_request(delete_request)) { return; }

  std::vector<uint8_t> response_data;
  client.receive_response(response_data);  // Ignore result
}

template <typename i_t, typename f_t>
static cancel_job_result_t cancel_job_impl(const std::string& host,
                                           int port,
                                           const std::string& job_id)
{
  cancel_job_result_t result;
  result.success    = false;
  result.message    = "Unknown error";
  result.job_status = remote_job_status_t::NOT_FOUND;

  remote_client_t client(host, port);
  if (!client.connect()) {
    result.message = "Failed to connect to server";
    return result;
  }

  auto serializer     = get_serializer<i_t, f_t>();
  auto cancel_request = serializer->serialize_cancel_request(job_id);

  if (!client.send_request(cancel_request)) {
    result.message = "Failed to send cancel request";
    return result;
  }

  std::vector<uint8_t> response_data;
  if (!client.receive_response(response_data)) {
    result.message = "Failed to receive response";
    return result;
  }

  // Deserialize the cancel response
  auto cancel_result = serializer->deserialize_cancel_response(response_data);

  result.success = cancel_result.success;
  result.message = cancel_result.message;

  // Map serializer job_status_t to remote_job_status_t
  using serializer_status = typename remote_serializer_t<i_t, f_t>::job_status_t;
  switch (cancel_result.job_status) {
    case serializer_status::QUEUED: result.job_status = remote_job_status_t::QUEUED; break;
    case serializer_status::PROCESSING: result.job_status = remote_job_status_t::PROCESSING; break;
    case serializer_status::COMPLETED: result.job_status = remote_job_status_t::COMPLETED; break;
    case serializer_status::FAILED: result.job_status = remote_job_status_t::FAILED; break;
    case serializer_status::NOT_FOUND: result.job_status = remote_job_status_t::NOT_FOUND; break;
    case serializer_status::CANCELLED: result.job_status = remote_job_status_t::CANCELLED; break;
  }

  return result;
}

}  // namespace

//============================================================================
// LP Remote Solve
//============================================================================

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<i_t, f_t>& view,
  const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  const bool sync_mode = use_sync_mode();

  CUOPT_LOG_INFO("[remote_solve] Solving LP remotely on {}:{} ({} mode)",
                 config.host,
                 config.port,
                 sync_mode ? "sync" : "async");

  // Log problem info (similar to local solve)
  if (settings.log_to_console) {
    auto n_rows = view.get_constraint_matrix_offsets().size() > 0
                    ? static_cast<i_t>(view.get_constraint_matrix_offsets().size()) - 1
                    : 0;
    auto n_cols = static_cast<i_t>(view.get_objective_coefficients().size());
    auto nnz    = static_cast<i_t>(view.get_constraint_matrix_values().size());
    CUOPT_LOG_INFO("Solving a problem with %d constraints, %d variables, and %d nonzeros (remote)",
                   n_rows,
                   n_cols,
                   nnz);
  }

  auto serializer = get_serializer<i_t, f_t>();

  if (sync_mode) {
    //=========================================================================
    // SYNC/BLOCKING MODE: Unified architecture
    //
    // Server-side: Job goes through queue, handled by worker process.
    // Client blocks until completion (server uses condition variable).
    // This enables cancellation for "sync" jobs and concurrent solves.
    //=========================================================================

    // Serialize as async request with blocking=true
    std::vector<uint8_t> request_data =
      serializer->serialize_async_lp_request(view, settings, true /* blocking */);
    CUOPT_LOG_DEBUG("[remote_solve] Serialized LP request (blocking): {} bytes",
                    request_data.size());

    // Connect and send
    remote_client_t client(config.host, config.port);
    if (!client.connect()) {
      return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
        "Failed to connect to remote server", cuopt::error_type_t::RuntimeError));
    }

    if (!client.send_request(request_data)) {
      return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
        "Failed to send request to remote server", cuopt::error_type_t::RuntimeError));
    }

    // Receive response (server blocks until job completes, then returns result)
    std::vector<uint8_t> response_data;
    if (!client.receive_response(response_data)) {
      return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
        "Failed to receive response from remote server", cuopt::error_type_t::RuntimeError));
    }

    CUOPT_LOG_DEBUG("[remote_solve] Received LP result (blocking): {} bytes", response_data.size());

    // Deserialize solution from result response (same format as async GET_RESULT)
    return serializer->deserialize_lp_result_response(response_data);

  } else {
    //=========================================================================
    // ASYNC MODE: Submit → Poll → Get Result → Delete
    //=========================================================================

    // Serialize as async request with blocking=false
    std::vector<uint8_t> request_data =
      serializer->serialize_async_lp_request(view, settings, false /* blocking */);
    CUOPT_LOG_DEBUG("[remote_solve] Serialized LP request (async): {} bytes", request_data.size());

    // Submit job
    auto [submit_ok, job_id_or_error] =
      submit_job<i_t, f_t>(config.host, config.port, request_data);
    if (!submit_ok) {
      return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
        "Job submission failed: " + job_id_or_error, cuopt::error_type_t::RuntimeError));
    }
    std::string job_id = job_id_or_error;
    CUOPT_LOG_INFO("[remote_solve] Job submitted, ID: {}", job_id);

    // Poll until complete
    if (!poll_until_complete<i_t, f_t>(config.host, config.port, job_id, settings.log_to_console)) {
      delete_job<i_t, f_t>(config.host, config.port, job_id);
      return optimization_problem_solution_t<i_t, f_t>(
        cuopt::logic_error("Job failed or not found", cuopt::error_type_t::RuntimeError));
    }

    // Get result
    auto [result_ok, result_data] = get_result<i_t, f_t>(config.host, config.port, job_id);
    if (!result_ok) {
      delete_job<i_t, f_t>(config.host, config.port, job_id);
      return optimization_problem_solution_t<i_t, f_t>(
        cuopt::logic_error("Failed to retrieve result", cuopt::error_type_t::RuntimeError));
    }

    // Delete job from server
    delete_job<i_t, f_t>(config.host, config.port, job_id);
    CUOPT_LOG_DEBUG("[remote_solve] Job {} deleted from server", job_id);

    // Deserialize solution from async result response
    return serializer->deserialize_lp_result_response(result_data);
  }
}

//============================================================================
// MIP Remote Solve
//============================================================================

template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<i_t, f_t>& view,
  const mip_solver_settings_t<i_t, f_t>& settings)
{
  const bool sync_mode = use_sync_mode();

  CUOPT_LOG_INFO("[remote_solve] Solving MIP remotely on {}:{} ({} mode)",
                 config.host,
                 config.port,
                 sync_mode ? "sync" : "async");

  // Log problem info
  {
    auto n_rows = view.get_constraint_matrix_offsets().size() > 0
                    ? static_cast<i_t>(view.get_constraint_matrix_offsets().size()) - 1
                    : 0;
    auto n_cols = static_cast<i_t>(view.get_objective_coefficients().size());
    auto nnz    = static_cast<i_t>(view.get_constraint_matrix_values().size());
    CUOPT_LOG_INFO(
      "Solving a problem with %d constraints, %d variables, and %d nonzeros (remote MIP)",
      n_rows,
      n_cols,
      nnz);
  }

  auto serializer = get_serializer<i_t, f_t>();

  if (sync_mode) {
    //=========================================================================
    // SYNC/BLOCKING MODE: Unified architecture
    //
    // Server-side: Job goes through queue, handled by worker process.
    // Client blocks until completion (server uses condition variable).
    // This enables cancellation for "sync" jobs and concurrent solves.
    //=========================================================================

    std::vector<uint8_t> request_data =
      serializer->serialize_async_mip_request(view, settings, true /* blocking */);
    CUOPT_LOG_DEBUG("[remote_solve] Serialized MIP request (blocking): {} bytes",
                    request_data.size());

    remote_client_t client(config.host, config.port);
    if (!client.connect()) {
      return mip_solution_t<i_t, f_t>(cuopt::logic_error("Failed to connect to remote server",
                                                         cuopt::error_type_t::RuntimeError));
    }

    if (!client.send_request(request_data)) {
      return mip_solution_t<i_t, f_t>(cuopt::logic_error("Failed to send request to remote server",
                                                         cuopt::error_type_t::RuntimeError));
    }

    // Receive response (server blocks until job completes, then returns result)
    std::vector<uint8_t> response_data;
    if (!client.receive_response(response_data)) {
      return mip_solution_t<i_t, f_t>(cuopt::logic_error(
        "Failed to receive response from remote server", cuopt::error_type_t::RuntimeError));
    }

    CUOPT_LOG_DEBUG("[remote_solve] Received MIP result (blocking): {} bytes",
                    response_data.size());

    // Deserialize solution from result response (same format as async GET_RESULT)
    return serializer->deserialize_mip_result_response(response_data);

  } else {
    //=========================================================================
    // ASYNC MODE: Submit → Poll → Get Result → Delete
    //=========================================================================

    std::vector<uint8_t> request_data =
      serializer->serialize_async_mip_request(view, settings, false /* blocking */);
    CUOPT_LOG_DEBUG("[remote_solve] Serialized MIP request (async): {} bytes", request_data.size());

    // Submit job
    auto [submit_ok, job_id_or_error] =
      submit_job<i_t, f_t>(config.host, config.port, request_data);
    if (!submit_ok) {
      return mip_solution_t<i_t, f_t>(cuopt::logic_error(
        "Job submission failed: " + job_id_or_error, cuopt::error_type_t::RuntimeError));
    }
    std::string job_id = job_id_or_error;
    CUOPT_LOG_INFO("[remote_solve] Job submitted, ID: {}", job_id);

    // Poll until complete
    if (!poll_until_complete<i_t, f_t>(config.host, config.port, job_id, true /* verbose */)) {
      delete_job<i_t, f_t>(config.host, config.port, job_id);
      return mip_solution_t<i_t, f_t>(
        cuopt::logic_error("Job failed or not found", cuopt::error_type_t::RuntimeError));
    }

    // Get result
    auto [result_ok, result_data] = get_result<i_t, f_t>(config.host, config.port, job_id);
    if (!result_ok) {
      delete_job<i_t, f_t>(config.host, config.port, job_id);
      return mip_solution_t<i_t, f_t>(
        cuopt::logic_error("Failed to retrieve result", cuopt::error_type_t::RuntimeError));
    }

    // Delete job from server
    delete_job<i_t, f_t>(config.host, config.port, job_id);
    CUOPT_LOG_DEBUG("[remote_solve] Job {} deleted from server", job_id);

    // Deserialize solution from async result response
    return serializer->deserialize_mip_result_response(result_data);
  }
}

//============================================================================
// Cancel Job Remote
//============================================================================

cancel_job_result_t cancel_job_remote(const remote_solve_config_t& config,
                                      const std::string& job_id)
{
  CUOPT_LOG_INFO("[remote_solve] Cancelling job {} on {}:{}", job_id, config.host, config.port);

  // Use int32_t, double as the type parameters (doesn't affect cancel logic)
  auto result = cancel_job_impl<int32_t, double>(config.host, config.port, job_id);

  if (result.success) {
    CUOPT_LOG_INFO("[remote_solve] Job {} cancelled successfully", job_id);
  } else {
    CUOPT_LOG_WARN("[remote_solve] Failed to cancel job {}: {}", job_id, result.message);
  }

  return result;
}

// Explicit instantiations
#if CUOPT_INSTANTIATE_FLOAT
template optimization_problem_solution_t<int32_t, float> solve_lp_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<int32_t, float>& view,
  const pdlp_solver_settings_t<int32_t, float>& settings);

template mip_solution_t<int32_t, float> solve_mip_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<int32_t, float>& view,
  const mip_solver_settings_t<int32_t, float>& settings);
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template optimization_problem_solution_t<int32_t, double> solve_lp_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<int32_t, double>& view,
  const pdlp_solver_settings_t<int32_t, double>& settings);

template mip_solution_t<int32_t, double> solve_mip_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<int32_t, double>& view,
  const mip_solver_settings_t<int32_t, double>& settings);
#endif

}  // namespace cuopt::linear_programming
