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
#include <vector>

namespace cuopt::linear_programming {

namespace {

// Message types for streaming protocol (must match server)
enum class MessageType : uint8_t {
  LOG_MESSAGE = 0,  // Log output from server
  SOLUTION    = 1,  // Final solution data
};

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

    CUOPT_LOG_INFO("[remote_solve] Connected to {}:{}", host_, port_);
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

}  // namespace

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<i_t, f_t>& view,
  const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  CUOPT_LOG_INFO("[remote_solve] Solving LP remotely on {}:{}", config.host, config.port);

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

  // Serialize the request
  std::vector<uint8_t> request_data = serializer->serialize_lp_request(view, settings);
  CUOPT_LOG_DEBUG("[remote_solve] Serialized LP request: {} bytes", request_data.size());

  // Connect and send
  remote_client_t client(config.host, config.port);
  if (!client.connect()) {
    return optimization_problem_solution_t<i_t, f_t>(
      cuopt::logic_error("Failed to connect to remote server", cuopt::error_type_t::RuntimeError));
  }

  if (!client.send_request(request_data)) {
    return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
      "Failed to send request to remote server", cuopt::error_type_t::RuntimeError));
  }

  // Receive response with streaming log support
  // Server sends LOG_MESSAGE types during solve, then SOLUTION at end
  std::vector<uint8_t> response_data;
  if (!client.receive_streaming_response(response_data, settings.log_to_console)) {
    return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
      "Failed to receive response from remote server", cuopt::error_type_t::RuntimeError));
  }

  CUOPT_LOG_DEBUG("[remote_solve] Received LP solution: {} bytes", response_data.size());

  // Deserialize solution
  auto solution = serializer->deserialize_lp_solution(response_data);

  // Note: Detailed logs were already streamed from server, no need to duplicate summary here

  return solution;
}

template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<i_t, f_t>& view,
  const mip_solver_settings_t<i_t, f_t>& settings)
{
  CUOPT_LOG_INFO("[remote_solve] Solving MIP remotely on {}:{}", config.host, config.port);

  // Log problem info (similar to local solve)
  // Note: MIP settings don't have log_to_console, so we always log
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

  // Serialize the request
  std::vector<uint8_t> request_data = serializer->serialize_mip_request(view, settings);
  CUOPT_LOG_DEBUG("[remote_solve] Serialized MIP request: {} bytes", request_data.size());

  // Connect and send
  remote_client_t client(config.host, config.port);
  if (!client.connect()) {
    return mip_solution_t<i_t, f_t>(
      cuopt::logic_error("Failed to connect to remote server", cuopt::error_type_t::RuntimeError));
  }

  if (!client.send_request(request_data)) {
    return mip_solution_t<i_t, f_t>(cuopt::logic_error("Failed to send request to remote server",
                                                       cuopt::error_type_t::RuntimeError));
  }

  // Receive response with streaming log support
  std::vector<uint8_t> response_data;
  if (!client.receive_streaming_response(response_data, true /* log_to_console */)) {
    return mip_solution_t<i_t, f_t>(cuopt::logic_error(
      "Failed to receive response from remote server", cuopt::error_type_t::RuntimeError));
  }

  CUOPT_LOG_DEBUG("[remote_solve] Received MIP solution: {} bytes", response_data.size());

  // Deserialize solution
  auto solution = serializer->deserialize_mip_solution(response_data);

  // Note: Detailed logs were already streamed from server, no need to duplicate summary here

  return solution;
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
