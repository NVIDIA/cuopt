/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/cpu_pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <utilities/logger.hpp>
#include "grpc_client.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace cuopt::linear_programming {

// Buffer added to the solver's time_limit to account for worker startup,
// GPU init, and result pipe transfer.
constexpr int kTimeoutBufferSeconds = 120;

// ============================================================================
// Helper function to get gRPC server address from environment variables
// ============================================================================

static std::string get_grpc_server_address()
{
  const char* host = std::getenv("CUOPT_REMOTE_HOST");
  const char* port = std::getenv("CUOPT_REMOTE_PORT");

  if (host == nullptr || port == nullptr) {
    throw std::runtime_error(
      "Remote execution enabled but CUOPT_REMOTE_HOST and/or CUOPT_REMOTE_PORT not set");
  }

  return std::string(host) + ":" + std::string(port);
}

static int64_t parse_env_int64(const char* name, int64_t default_value)
{
  const char* val = std::getenv(name);
  if (val == nullptr) return default_value;
  try {
    return std::stoll(val);
  } catch (...) {
    return default_value;
  }
}

// Derive client-side polling timeout from the solver's time_limit.
// Returns 0 (no limit) when the solver has no finite time_limit.
template <typename f_t>
static int solver_timeout_seconds(f_t time_limit)
{
  if (!std::isfinite(static_cast<double>(time_limit)) || time_limit <= 0) { return 0; }
  double secs = static_cast<double>(time_limit) + kTimeoutBufferSeconds;
  if (secs > static_cast<double>(std::numeric_limits<int>::max())) { return 0; }
  return static_cast<int>(std::ceil(secs));
}

static std::string read_pem_file(const char* path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) { throw std::runtime_error(std::string("Cannot open TLS file: ") + path); }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static const char* get_env(const char* name)
{
  const char* v = std::getenv(name);
  return (v && v[0] != '\0') ? v : nullptr;
}

// Apply env-var overrides for transfer, debug, and TLS configuration.
static void apply_env_overrides(grpc_client_config_t& config)
{
  constexpr int64_t kMinChunkSize   = 4096;
  constexpr int64_t kMaxChunkSize   = 2LL * 1024 * 1024 * 1024;  // 2 GiB
  constexpr int64_t kMinMessageSize = 4096;
  constexpr int64_t kMaxMessageSize = 2LL * 1024 * 1024 * 1024;

  auto chunk = parse_env_int64("CUOPT_CHUNK_SIZE", config.chunk_size_bytes);
  if (chunk >= kMinChunkSize && chunk <= kMaxChunkSize) { config.chunk_size_bytes = chunk; }

  auto msg = parse_env_int64("CUOPT_MAX_MESSAGE_BYTES", config.max_message_bytes);
  if (msg >= kMinMessageSize && msg <= kMaxMessageSize) { config.max_message_bytes = msg; }

  config.enable_debug_log = (parse_env_int64("CUOPT_GRPC_DEBUG", 0) != 0);

  // TLS configuration from environment variables
  if (parse_env_int64("CUOPT_TLS_ENABLED", 0) != 0) {
    config.enable_tls = true;

    const char* root_cert = get_env("CUOPT_TLS_ROOT_CERT");
    if (root_cert) { config.tls_root_certs = read_pem_file(root_cert); }

    const char* client_cert = get_env("CUOPT_TLS_CLIENT_CERT");
    const char* client_key  = get_env("CUOPT_TLS_CLIENT_KEY");
    if (client_cert && client_key) {
      config.tls_client_cert = read_pem_file(client_cert);
      config.tls_client_key  = read_pem_file(client_key);
    }
  }

  CUOPT_LOG_DEBUG("gRPC client config: chunk_size=%lld max_message=%lld tls=%s",
                  static_cast<long long>(config.chunk_size_bytes),
                  static_cast<long long>(config.max_message_bytes),
                  config.enable_tls ? "on" : "off");
}

// ============================================================================
// Remote execution via gRPC
// ============================================================================

template <typename i_t, typename f_t>
std::unique_ptr<lp_solution_interface_t<i_t, f_t>> solve_lp_remote(
  cpu_optimization_problem_t<i_t, f_t> const& cpu_problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings)
{
  init_logger_t log(settings.log_file, settings.log_to_console);

  CUOPT_LOG_INFO("Using remote GPU backend");

  // Build gRPC client configuration
  grpc_client_config_t config;
  config.server_address  = get_grpc_server_address();
  config.timeout_seconds = solver_timeout_seconds(settings.time_limit);
  apply_env_overrides(config);

  // Stream the server's solver log to the client.  The server already
  // filters by the requested log level, so we just pass lines through to
  // stdout and/or the log file as-is.
  std::unique_ptr<std::ofstream> log_file_stream;
  if (!settings.log_file.empty()) {
    log_file_stream = std::make_unique<std::ofstream>(settings.log_file, std::ios::app);
  }
  bool want_console = settings.log_to_console;
  bool want_file    = log_file_stream && log_file_stream->is_open();

  if (want_console || want_file) {
    config.stream_logs  = true;
    config.log_callback = [want_console, want_file, &log_file_stream](const std::string& line) {
      if (want_console) { std::cout << line << std::endl; }
      if (want_file) { *log_file_stream << line << std::endl; }
    };
  }

  // Create client and connect
  grpc_client_t client(config);
  if (!client.connect()) {
    throw std::runtime_error("Failed to connect to gRPC server: " + client.get_last_error());
  }

  CUOPT_LOG_DEBUG("solve_lp_remote - connected to %s, submitting problem (timeout=%ds)",
                  config.server_address.c_str(),
                  config.timeout_seconds);

  // Call the remote solver
  auto result = client.solve_lp(cpu_problem, settings);

  if (!result.success) {
    throw std::runtime_error("Remote LP solve failed: " + result.error_message);
  }

  CUOPT_LOG_DEBUG("solve_lp_remote - solve completed successfully");

  return std::move(result.solution);
}

template <typename i_t, typename f_t>
std::unique_ptr<mip_solution_interface_t<i_t, f_t>> solve_mip_remote(
  cpu_optimization_problem_t<i_t, f_t> const& cpu_problem,
  mip_solver_settings_t<i_t, f_t> const& settings)
{
  init_logger_t log(settings.log_file, settings.log_to_console);

  CUOPT_LOG_INFO("Using remote GPU backend");

  // Build gRPC client configuration
  grpc_client_config_t config;
  config.server_address  = get_grpc_server_address();
  config.timeout_seconds = solver_timeout_seconds(settings.time_limit);
  apply_env_overrides(config);

  // Stream server log — same passthrough logic as the LP callback above.
  std::unique_ptr<std::ofstream> log_file_stream;
  if (!settings.log_file.empty()) {
    log_file_stream = std::make_unique<std::ofstream>(settings.log_file, std::ios::app);
  }
  bool want_console = settings.log_to_console;
  bool want_file    = log_file_stream && log_file_stream->is_open();

  if (want_console || want_file) {
    config.stream_logs  = true;
    config.log_callback = [want_console, want_file, &log_file_stream](const std::string& line) {
      if (want_console) { std::cout << line << std::endl; }
      if (want_file) { *log_file_stream << line << std::endl; }
    };
  }

  // Check if user has set incumbent callbacks
  auto mip_callbacks   = settings.get_mip_callbacks();
  bool has_incumbents  = !mip_callbacks.empty();
  bool enable_tracking = has_incumbents;

  // Initialize callbacks with problem size (needed for Python callbacks to work correctly)
  // The local MIP solver does this in solve.cu, but for remote solves we need to do it here
  if (has_incumbents) {
    size_t n_vars = cpu_problem.get_n_variables();
    for (auto* callback : mip_callbacks) {
      if (callback != nullptr) { callback->template setup<f_t>(n_vars); }
    }
  }

  // Set up incumbent callback forwarding
  if (has_incumbents) {
    CUOPT_LOG_INFO("solve_mip_remote - setting up inline incumbent callback forwarding");
    config.incumbent_callback = [&mip_callbacks](int64_t index,
                                                 double objective,
                                                 const std::vector<double>& solution) -> bool {
      // Forward incumbent to all user callbacks (invoked from main thread with GIL)
      for (auto* callback : mip_callbacks) {
        if (callback != nullptr &&
            callback->get_type() == internals::base_solution_callback_type::GET_SOLUTION) {
          auto* get_callback = static_cast<internals::get_solution_callback_t*>(callback);
          // Copy solution to non-const buffer for callback interface
          std::vector<double> solution_copy = solution;
          double obj_copy                   = objective;
          double bound_copy                 = std::numeric_limits<double>::quiet_NaN();
          get_callback->get_solution(
            solution_copy.data(), &obj_copy, &bound_copy, callback->get_user_data());
        }
      }
      return true;  // Continue solving
    };
  }

  // Create client and connect
  grpc_client_t client(config);
  if (!client.connect()) {
    throw std::runtime_error("Failed to connect to gRPC server: " + client.get_last_error());
  }

  CUOPT_LOG_DEBUG(
    "solve_mip_remote - connected to %s, submitting problem (incumbents=%s, timeout=%ds)",
    config.server_address.c_str(),
    enable_tracking ? "enabled" : "disabled",
    config.timeout_seconds);

  // Call the remote solver
  auto result = client.solve_mip(cpu_problem, settings, enable_tracking);

  if (!result.success) {
    throw std::runtime_error("Remote MIP solve failed: " + result.error_message);
  }

  CUOPT_LOG_DEBUG("solve_mip_remote - solve completed successfully");

  return std::move(result.solution);
}

// Explicit template instantiations for remote execution stubs
template std::unique_ptr<lp_solution_interface_t<int, double>> solve_lp_remote(
  cpu_optimization_problem_t<int, double> const&, pdlp_solver_settings_t<int, double> const&);

template std::unique_ptr<mip_solution_interface_t<int, double>> solve_mip_remote(
  cpu_optimization_problem_t<int, double> const&, mip_solver_settings_t<int, double> const&);

}  // namespace cuopt::linear_programming
