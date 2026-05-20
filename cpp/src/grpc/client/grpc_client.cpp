/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "grpc_client.hpp"

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <utilities/logger.hpp>
#include "grpc_problem_mapper.hpp"
#include "grpc_service_mapper.hpp"
#include "grpc_settings_mapper.hpp"
#include "grpc_solution_mapper.hpp"

#include <cuopt_remote_service.grpc.pb.h>
#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <thread>

namespace cuopt::linear_programming {

// =============================================================================
// Constants
// =============================================================================

constexpr int kDefaultRpcTimeoutSeconds = 60;  // per-RPC deadline for short operations

constexpr int64_t kMinMessageBytes = 4LL * 1024 * 1024;  // 4 MiB floor for max_message_bytes

// Protobuf's hard serialization limit is 2 GiB (int32 sizes internally).
// Reserve 1 MiB headroom for gRPC framing and internal bookkeeping.
constexpr int64_t kMaxMessageBytes = 2LL * 1024 * 1024 * 1024 - 1LL * 1024 * 1024;

// =============================================================================
// Debug Logging Helper
// =============================================================================

// Helper macro to log to debug callback if configured, otherwise to std::cerr.
// Only emits output when enable_debug_log is true or a debug_log_callback is set.
#define GRPC_CLIENT_DEBUG_LOG(config, msg)                                      \
  do {                                                                          \
    if (!(config).enable_debug_log && !(config).debug_log_callback) break;      \
    std::ostringstream _oss;                                                    \
    _oss << msg;                                                                \
    std::string _msg_str = _oss.str();                                          \
    if ((config).debug_log_callback) { (config).debug_log_callback(_msg_str); } \
    if ((config).enable_debug_log) { std::cerr << _msg_str << "\n"; }           \
  } while (0)

// Structured throughput log for benchmarking. Parseable format:
//   [THROUGHPUT] phase=<name> bytes=<N> elapsed_ms=<N> throughput_mb_s=<N.N>
#define GRPC_CLIENT_THROUGHPUT_LOG(config, phase_name, byte_count, start_time)                      \
  do {                                                                                              \
    auto _end = std::chrono::steady_clock::now();                                                   \
    auto _ms  = std::chrono::duration_cast<std::chrono::microseconds>(_end - (start_time)).count(); \
    double _sec = _ms / 1e6;                                                                        \
    double _mb  = static_cast<double>(byte_count) / (1024.0 * 1024.0);                              \
    double _mbs = (_sec > 0.0) ? (_mb / _sec) : 0.0;                                                \
    GRPC_CLIENT_DEBUG_LOG(                                                                          \
      config,                                                                                       \
      "[THROUGHPUT] phase=" << (phase_name) << " bytes=" << (byte_count) << " elapsed_ms="          \
                            << std::fixed << std::setprecision(1) << (_ms / 1000.0)                 \
                            << " throughput_mb_s=" << std::setprecision(1) << _mbs);                \
  } while (0)

// Private implementation (PIMPL pattern to hide gRPC types)
struct grpc_client_t::impl_t {
  std::shared_ptr<grpc::Channel> channel;
  // Use StubInterface to support both real stubs and mock stubs for testing
  std::shared_ptr<cuopt::remote::CuOptRemoteService::StubInterface> stub;
  bool mock_mode = false;  // Set to true when using injected mock stub
};

// All finite-duration RPCs (CheckStatus, SendArrayChunk, GetResult, etc.)
// use kDefaultRpcTimeoutSeconds (60s).  Indefinite RPCs (StreamLogs,
// WaitForCompletion) omit the deadline entirely and rely on TryCancel or
// client-side polling for cancellation — a fixed deadline would kill
// legitimate long-running solves.
static void set_rpc_deadline(grpc::ClientContext& ctx, int timeout_seconds)
{
  if (timeout_seconds > 0) {
    ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(timeout_seconds));
  }
}

// =============================================================================
// Test Helper Functions (for mock stub injection)
// =============================================================================

void grpc_test_inject_mock_stub(grpc_client_t& client, std::shared_ptr<void> stub)
{
  // Cast from void* to StubInterface* - caller must ensure correct type
  client.impl_->stub =
    std::static_pointer_cast<cuopt::remote::CuOptRemoteService::StubInterface>(stub);
  client.impl_->mock_mode = true;
}

void grpc_test_mark_as_connected(grpc_client_t& client) { client.impl_->mock_mode = true; }

grpc_client_t::grpc_client_t(const grpc_client_config_t& config)
  : impl_(std::make_unique<impl_t>()), config_(config)
{
  config_.max_message_bytes =
    std::clamp(config_.max_message_bytes, kMinMessageBytes, kMaxMessageBytes);
  if (config_.chunked_array_threshold_bytes >= 0) {
    chunked_array_threshold_bytes_ = config_.chunked_array_threshold_bytes;
  } else {
    chunked_array_threshold_bytes_ = config_.max_message_bytes * 3 / 4;
  }
}

grpc_client_t::grpc_client_t(const std::string& server_address) : impl_(std::make_unique<impl_t>())
{
  config_.server_address = server_address;
  config_.max_message_bytes =
    std::clamp(config_.max_message_bytes, kMinMessageBytes, kMaxMessageBytes);
  chunked_array_threshold_bytes_ = config_.max_message_bytes * 3 / 4;
}

grpc_client_t::~grpc_client_t() { stop_log_streaming(); }

bool grpc_client_t::connect()
{
  std::shared_ptr<grpc::ChannelCredentials> creds;

  if (config_.enable_tls) {
    grpc::SslCredentialsOptions ssl_opts;

    // Root CA certificates for verifying the server
    if (!config_.tls_root_certs.empty()) { ssl_opts.pem_root_certs = config_.tls_root_certs; }

    // Client certificate and key for mTLS
    if (!config_.tls_client_cert.empty() && !config_.tls_client_key.empty()) {
      ssl_opts.pem_cert_chain  = config_.tls_client_cert;
      ssl_opts.pem_private_key = config_.tls_client_key;
    }

    creds = grpc::SslCredentials(ssl_opts);
  } else {
    creds = grpc::InsecureChannelCredentials();
  }

  grpc::ChannelArguments channel_args;
  const int channel_limit = static_cast<int>(config_.max_message_bytes);
  channel_args.SetMaxReceiveMessageSize(channel_limit);
  channel_args.SetMaxSendMessageSize(channel_limit);
  channel_args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, config_.keepalive_time_ms);
  channel_args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, config_.keepalive_timeout_ms);
  channel_args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);

  impl_->channel = grpc::CreateCustomChannel(config_.server_address, creds, channel_args);
  impl_->stub    = cuopt::remote::CuOptRemoteService::NewStub(impl_->channel);

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] Connecting to " << config_.server_address
                                                       << (config_.enable_tls ? " (TLS)" : ""));

  // Verify connectivity with a lightweight RPC probe. Channel-level checks like
  // WaitForConnected are unreliable (gRPC lazy connection on localhost can
  // report READY even without a server). A real RPC with a deadline is the
  // only reliable way to confirm the server is reachable.
  {
    grpc::ClientContext probe_ctx;
    probe_ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    cuopt::remote::StatusRequest probe_req;
    probe_req.set_job_id("__connection_probe__");
    cuopt::remote::StatusResponse probe_resp;
    auto probe_status = impl_->stub->CheckStatus(&probe_ctx, probe_req, &probe_resp);

    auto code = probe_status.error_code();
    if (code != grpc::StatusCode::OK && code != grpc::StatusCode::NOT_FOUND) {
      last_error_ = "Failed to connect to server at " + config_.server_address + " (" +
                    probe_status.error_message() + ")";
      GRPC_CLIENT_DEBUG_LOG(config_, "[grpc_client] Connection failed: " << last_error_);
      return false;
    }
  }

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] Connected successfully to " << config_.server_address);
  return true;
}

bool grpc_client_t::is_connected() const
{
  // In mock mode, we're always "connected" if a stub is present
  if (impl_->mock_mode) { return impl_->stub != nullptr; }

  if (!impl_->channel) return false;
  auto state = impl_->channel->GetState(false);
  return state == GRPC_CHANNEL_READY || state == GRPC_CHANNEL_IDLE;
}

void grpc_client_t::start_log_streaming(const std::string& job_id)
{
  if (!config_.stream_logs || !config_.log_callback) return;

  if (log_thread_ && log_thread_->joinable()) {
    stop_logs_.store(true);
    {
      std::lock_guard<std::mutex> lk(log_context_mutex_);
      if (active_log_context_) {
        static_cast<grpc::ClientContext*>(active_log_context_)->TryCancel();
      }
    }
    log_thread_->join();
    log_thread_.reset();
  }

  stop_logs_.store(false);
  log_thread_ = std::make_unique<std::thread>([this, job_id]() {
    stream_logs(job_id, 0, [this](const std::string& line, bool /*job_complete*/) {
      if (stop_logs_.load()) return false;
      if (config_.log_callback) { config_.log_callback(line); }
      return true;
    });
  });
}

void grpc_client_t::stop_log_streaming()
{
  stop_logs_.store(true);
  // Cancel the in-flight streaming RPC so reader->Read() returns false
  // immediately instead of blocking until the server sends a message.
  {
    std::lock_guard<std::mutex> lk(log_context_mutex_);
    if (active_log_context_) {
      static_cast<grpc::ClientContext*>(active_log_context_)->TryCancel();
    }
  }
  // Move to local so we can join without racing against other callers.
  // TryCancel above guarantees the thread will unblock promptly.
  std::unique_ptr<std::thread> t;
  std::swap(t, log_thread_);
  if (t && t->joinable()) { t->join(); }
}

// =============================================================================
// Proto → Client Enum Conversion
// =============================================================================

job_status_t map_proto_job_status(cuopt::remote::JobStatus proto_status)
{
  switch (proto_status) {
    case cuopt::remote::QUEUED: return job_status_t::QUEUED;
    case cuopt::remote::PROCESSING: return job_status_t::PROCESSING;
    case cuopt::remote::COMPLETED: return job_status_t::COMPLETED;
    case cuopt::remote::FAILED: return job_status_t::FAILED;
    case cuopt::remote::CANCELLED: return job_status_t::CANCELLED;
    default: return job_status_t::NOT_FOUND;
  }
}

// =============================================================================
// Async Job Management Operations
// =============================================================================

job_status_result_t grpc_client_t::check_status(const std::string& job_id)
{
  job_status_result_t result;

  if (!impl_->stub) {
    result.error_message = "Not connected to server";
    return result;
  }

  grpc::ClientContext context;
  set_rpc_deadline(context, kDefaultRpcTimeoutSeconds);
  auto request = build_status_request(job_id);
  cuopt::remote::StatusResponse response;
  auto status = impl_->stub->CheckStatus(&context, request, &response);

  if (!status.ok()) {
    result.error_message = "CheckStatus failed: " + status.error_message();
    return result;
  }

  result.success           = true;
  result.message           = response.message();
  result.result_size_bytes = response.result_size_bytes();

  // Track server max message size
  if (response.max_message_bytes() > 0) {
    server_max_message_bytes_.store(response.max_message_bytes(), std::memory_order_relaxed);
  }

  result.status = map_proto_job_status(response.job_status());

  return result;
}

job_status_result_t grpc_client_t::wait_for_completion(const std::string& job_id)
{
  job_status_result_t result;

  if (!impl_->stub) {
    result.error_message = "Not connected to server";
    return result;
  }

  grpc::ClientContext context;
  // No RPC deadline: WaitForCompletion blocks until the solver finishes,
  // which may exceed any fixed timeout.  The server detects client
  // disconnect (context->IsCancelled), and the production path uses
  // poll_for_completion which has its own config_.timeout_seconds loop.
  cuopt::remote::WaitRequest request;
  request.set_job_id(job_id);
  cuopt::remote::WaitResponse response;

  auto status = impl_->stub->WaitForCompletion(&context, request, &response);

  if (!status.ok()) {
    result.error_message = "WaitForCompletion failed: " + status.error_message();
    return result;
  }

  result.success           = true;
  result.message           = response.message();
  result.result_size_bytes = response.result_size_bytes();

  result.status = map_proto_job_status(response.job_status());

  return result;
}

cancel_result_t grpc_client_t::cancel_job(const std::string& job_id)
{
  cancel_result_t result;

  if (!impl_->stub) {
    result.error_message = "Not connected to server";
    return result;
  }

  grpc::ClientContext context;
  set_rpc_deadline(context, kDefaultRpcTimeoutSeconds);
  auto request = build_cancel_request(job_id);
  cuopt::remote::CancelResponse response;
  auto status = impl_->stub->CancelJob(&context, request, &response);

  if (!status.ok()) {
    result.error_message = "CancelJob failed: " + status.error_message();
    return result;
  }

  result.success = (response.status() == cuopt::remote::SUCCESS);
  result.message = response.message();

  result.job_status = map_proto_job_status(response.job_status());

  return result;
}

bool grpc_client_t::delete_job(const std::string& job_id)
{
  if (!impl_->stub) {
    last_error_ = "Not connected to server";
    return false;
  }

  grpc::ClientContext context;
  set_rpc_deadline(context, kDefaultRpcTimeoutSeconds);
  cuopt::remote::DeleteRequest request;
  request.set_job_id(job_id);
  cuopt::remote::DeleteResponse response;
  auto status = impl_->stub->DeleteResult(&context, request, &response);

  if (!status.ok()) {
    last_error_ = "DeleteResult RPC failed: " + status.error_message();
    return false;
  }

  // Check response status - job must exist to be deleted
  if (response.status() == cuopt::remote::ERROR_NOT_FOUND) {
    last_error_ = "Job not found: " + job_id;
    return false;
  }

  if (response.status() != cuopt::remote::SUCCESS) {
    last_error_ = "DeleteResult failed: " + response.message();
    return false;
  }

  return true;
}

incumbents_result_t grpc_client_t::get_incumbents(const std::string& job_id,
                                                  int64_t from_index,
                                                  int32_t max_count)
{
  incumbents_result_t result;

  if (!impl_->stub) {
    result.error_message = "Not connected to server";
    return result;
  }

  grpc::ClientContext context;
  set_rpc_deadline(context, kDefaultRpcTimeoutSeconds);
  cuopt::remote::IncumbentRequest request;
  request.set_job_id(job_id);
  request.set_from_index(from_index);
  request.set_max_count(max_count);

  cuopt::remote::IncumbentResponse response;
  auto status = impl_->stub->GetIncumbents(&context, request, &response);

  if (!status.ok()) {
    result.error_message = "GetIncumbents failed: " + status.error_message();
    return result;
  }

  result.success      = true;
  result.next_index   = response.next_index();
  result.job_complete = response.job_complete();

  for (const auto& inc : response.incumbents()) {
    incumbent_t entry;
    entry.index     = inc.index();
    entry.objective = inc.objective();
    entry.assignment.reserve(inc.assignment_size());
    for (int i = 0; i < inc.assignment_size(); ++i) {
      entry.assignment.push_back(inc.assignment(i));
    }
    result.incumbents.push_back(std::move(entry));
  }

  return result;
}

bool grpc_client_t::stream_logs(
  const std::string& job_id,
  int64_t from_byte,
  std::function<bool(const std::string& line, bool job_complete)> callback)
{
  if (!impl_->stub) {
    last_error_ = "Not connected to server";
    return false;
  }

  grpc::ClientContext context;
  // No RPC deadline here: this stream stays open for the entire solve, which
  // can exceed any fixed timeout.  Shutdown is via TryCancel from
  // stop_log_streaming(), not a deadline.
  cuopt::remote::StreamLogsRequest request;
  request.set_job_id(job_id);
  request.set_from_byte(from_byte);

  // Publish this context so stop_log_streaming() can TryCancel it from
  // another thread.  The mutex ensures the pointer is never dangling:
  // we clear it under the same lock before `context` goes out of scope.
  {
    std::lock_guard<std::mutex> lk(log_context_mutex_);
    active_log_context_ = &context;
  }

  auto reader = impl_->stub->StreamLogs(&context, request);

  cuopt::remote::LogMessage log_msg;
  while (reader->Read(&log_msg)) {
    bool done = log_msg.job_complete();
    if (!log_msg.line().empty()) {
      bool should_continue = callback(log_msg.line(), done);
      if (!should_continue) {
        context.TryCancel();
        break;
      }
    }
    if (done) { break; }
  }

  auto status = reader->Finish();

  {
    std::lock_guard<std::mutex> lk(log_context_mutex_);
    active_log_context_ = nullptr;
  }

  return status.ok() || status.error_code() == grpc::StatusCode::CANCELLED;
}

bool grpc_client_t::submit_unary(const cuopt::remote::SubmitJobRequest& request,
                                 std::string& job_id_out)
{
  job_id_out.clear();

  if (!impl_->stub) {
    last_error_ = "Not connected to server";
    return false;
  }

  auto t0 = std::chrono::steady_clock::now();

  grpc::ClientContext context;
  set_rpc_deadline(context, kDefaultRpcTimeoutSeconds);
  cuopt::remote::SubmitJobResponse response;
  auto status = impl_->stub->SubmitJob(&context, request, &response);

  GRPC_CLIENT_THROUGHPUT_LOG(config_, "upload_unary", request.ByteSizeLong(), t0);

  if (!status.ok()) {
    last_error_ = "SubmitJob failed: " + status.error_message();
    return false;
  }

  job_id_out = response.job_id();
  if (job_id_out.empty()) {
    last_error_ = "SubmitJob succeeded but no job_id returned";
    return false;
  }

  GRPC_CLIENT_DEBUG_LOG(config_, "[grpc_client] Unary submit succeeded, job_id=" << job_id_out);
  return true;
}

// =============================================================================
// Async Submit and Get Result
// =============================================================================

template <typename i_t, typename f_t>
submit_result_t grpc_client_t::submit_lp(const cpu_optimization_problem_t<i_t, f_t>& problem,
                                         const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  submit_result_t result;

  GRPC_CLIENT_DEBUG_LOG(config_, "[grpc_client] submit_lp: starting submission");

  if (!is_connected()) {
    result.error_message = "Not connected to server";
    GRPC_CLIENT_DEBUG_LOG(config_, "[grpc_client] submit_lp: not connected to server");
    return result;
  }

  // Check if chunked array upload should be used
  bool use_chunked = false;
  if (chunked_array_threshold_bytes_ >= 0) {
    size_t est  = estimate_problem_proto_size(problem);
    use_chunked = (static_cast<int64_t>(est) > chunked_array_threshold_bytes_);
    GRPC_CLIENT_DEBUG_LOG(config_,
                          "[grpc_client] submit_lp: estimated_size="
                            << est << " threshold=" << chunked_array_threshold_bytes_
                            << " use_chunked=" << use_chunked);
  }

  if (use_chunked) {
    cuopt::remote::ChunkedProblemHeader header;
    populate_chunked_header_lp(problem, settings, &header);
    if (!upload_chunked_arrays(problem, header, result.job_id)) {
      result.error_message = last_error_;
      return result;
    }
  } else {
    auto submit_request = build_lp_submit_request(problem, settings);
    if (!submit_unary(submit_request, result.job_id)) {
      result.error_message = last_error_;
      return result;
    }
  }

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] submit_lp: job submitted, job_id=" << result.job_id);
  result.success = true;
  return result;
}

template <typename i_t, typename f_t>
submit_result_t grpc_client_t::submit_mip(const cpu_optimization_problem_t<i_t, f_t>& problem,
                                          const mip_solver_settings_t<i_t, f_t>& settings,
                                          bool enable_incumbents)
{
  submit_result_t result;

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] submit_mip: starting submission"
                          << (enable_incumbents ? " (incumbents enabled)" : ""));

  if (!is_connected()) {
    result.error_message = "Not connected to server";
    return result;
  }

  bool use_chunked = false;
  if (chunked_array_threshold_bytes_ >= 0) {
    size_t est  = estimate_problem_proto_size(problem);
    use_chunked = (static_cast<int64_t>(est) > chunked_array_threshold_bytes_);
    GRPC_CLIENT_DEBUG_LOG(config_,
                          "[grpc_client] submit_mip: estimated_size="
                            << est << " threshold=" << chunked_array_threshold_bytes_
                            << " use_chunked=" << use_chunked);
  }

  if (use_chunked) {
    cuopt::remote::ChunkedProblemHeader header;
    populate_chunked_header_mip(problem, settings, enable_incumbents, &header);
    if (!upload_chunked_arrays(problem, header, result.job_id)) {
      result.error_message = last_error_;
      return result;
    }
  } else {
    auto submit_request = build_mip_submit_request(problem, settings, enable_incumbents);
    if (!submit_unary(submit_request, result.job_id)) {
      result.error_message = last_error_;
      return result;
    }
  }

  GRPC_CLIENT_DEBUG_LOG(
    config_, "[grpc_client] submit_mip: job submitted successfully, job_id=" << result.job_id);
  result.success = true;
  return result;
}

template <typename i_t, typename f_t>
remote_lp_result_t<i_t, f_t> grpc_client_t::get_lp_result(const std::string& job_id)
{
  remote_lp_result_t<i_t, f_t> result;

  if (!is_connected()) {
    result.error_message = "Not connected to server";
    return result;
  }

  downloaded_result_t dl;
  if (!get_result_or_download(job_id, dl)) {
    result.error_message = last_error_;
    return result;
  }

  if (dl.was_chunked) {
    result.solution = std::make_unique<cpu_lp_solution_t<i_t, f_t>>(
      chunked_result_to_lp_solution<i_t, f_t>(*dl.chunked_header, dl.chunked_arrays));
  } else {
    result.solution = std::make_unique<cpu_lp_solution_t<i_t, f_t>>(
      map_proto_to_lp_solution<i_t, f_t>(dl.response->lp_solution()));
  }
  result.success = true;
  return result;
}

template <typename i_t, typename f_t>
remote_mip_result_t<i_t, f_t> grpc_client_t::get_mip_result(const std::string& job_id)
{
  remote_mip_result_t<i_t, f_t> result;

  if (!is_connected()) {
    result.error_message = "Not connected to server";
    return result;
  }

  downloaded_result_t dl;
  if (!get_result_or_download(job_id, dl)) {
    result.error_message = last_error_;
    return result;
  }

  if (dl.was_chunked) {
    result.solution = std::make_unique<cpu_mip_solution_t<i_t, f_t>>(
      chunked_result_to_mip_solution<i_t, f_t>(*dl.chunked_header, dl.chunked_arrays));
  } else {
    result.solution = std::make_unique<cpu_mip_solution_t<i_t, f_t>>(
      map_proto_to_mip_solution<i_t, f_t>(dl.response->mip_solution()));
  }
  result.success = true;
  return result;
}

// =============================================================================
// Polling helper
// =============================================================================

grpc_client_t::poll_result_t grpc_client_t::poll_for_completion(const std::string& job_id)
{
  poll_result_t poll_result;

  int poll_count = 0;
  int poll_ms    = std::max(config_.poll_interval_ms, 1);
  // timeout_seconds <= 0 means "wait indefinitely" — the solver's own
  // time_limit (passed via settings) is the authoritative bound.
  int max_polls;
  if (config_.timeout_seconds > 0) {
    int64_t total_ms = static_cast<int64_t>(config_.timeout_seconds) * 1000;
    int64_t computed = total_ms / poll_ms;
    max_polls =
      static_cast<int>(std::min(computed, static_cast<int64_t>(std::numeric_limits<int>::max())));
  } else {
    max_polls = std::numeric_limits<int>::max();
  }

  int64_t incumbent_next_index = 0;
  auto last_incumbent_poll     = std::chrono::steady_clock::now();
  bool cancel_requested        = false;

  while (poll_count < max_polls) {
    std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));

    if (cancel_requested) {
      cancel_job(job_id);
      poll_result.cancelled_by_callback = true;
      poll_result.error_message         = "Cancelled by incumbent callback";
      return poll_result;
    }

    if (config_.incumbent_callback) {
      auto now = std::chrono::steady_clock::now();
      auto ms_since_last =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_incumbent_poll).count();
      if (ms_since_last >= config_.incumbent_poll_interval_ms) {
        auto inc_result = get_incumbents(job_id, incumbent_next_index, 0);
        if (inc_result.success) {
          for (const auto& inc : inc_result.incumbents) {
            bool should_continue =
              config_.incumbent_callback(inc.index, inc.objective, inc.assignment);
            if (!should_continue) {
              cancel_requested = true;
              break;
            }
          }
          incumbent_next_index = inc_result.next_index;
        }
        last_incumbent_poll = now;
      }
    }

    auto status_result = check_status(job_id);
    if (!status_result.success) {
      poll_result.error_message = status_result.error_message;
      return poll_result;
    }

    switch (status_result.status) {
      case job_status_t::COMPLETED: poll_result.completed = true; break;
      case job_status_t::FAILED:
        poll_result.error_message = "Job failed: " + status_result.message;
        return poll_result;
      case job_status_t::CANCELLED:
        poll_result.error_message = "Job was cancelled";
        return poll_result;
      default: break;
    }

    if (poll_result.completed) break;
    poll_count++;
  }

  // Drain any incumbents that arrived between the last poll and job completion.
  if (config_.incumbent_callback && poll_result.completed) {
    auto inc_result = get_incumbents(job_id, incumbent_next_index, 0);
    if (inc_result.success) {
      for (const auto& inc : inc_result.incumbents) {
        config_.incumbent_callback(inc.index, inc.objective, inc.assignment);
      }
    }
  }

  if (!poll_result.completed && poll_result.error_message.empty()) {
    poll_result.error_message = "Timeout waiting for job completion";
  }

  return poll_result;
}

// =============================================================================
// End-to-end solve helpers
// =============================================================================

template <typename i_t, typename f_t>
remote_lp_result_t<i_t, f_t> grpc_client_t::solve_lp(
  const cpu_optimization_problem_t<i_t, f_t>& problem,
  const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  auto solve_t0 = std::chrono::steady_clock::now();

  auto sub = submit_lp(problem, settings);
  if (!sub.success) { return {.error_message = sub.error_message}; }

  start_log_streaming(sub.job_id);
  auto poll = poll_for_completion(sub.job_id);
  stop_log_streaming();

  if (!poll.completed) { return {.error_message = poll.error_message}; }

  auto result = get_lp_result<i_t, f_t>(sub.job_id);
  if (result.success) { delete_job(sub.job_id); }

  GRPC_CLIENT_THROUGHPUT_LOG(config_, "end_to_end_lp", 0, solve_t0);

  return result;
}

template <typename i_t, typename f_t>
remote_mip_result_t<i_t, f_t> grpc_client_t::solve_mip(
  const cpu_optimization_problem_t<i_t, f_t>& problem,
  const mip_solver_settings_t<i_t, f_t>& settings,
  bool enable_incumbents)
{
  auto solve_t0 = std::chrono::steady_clock::now();

  bool track_incumbents = enable_incumbents || (config_.incumbent_callback != nullptr);

  auto sub = submit_mip(problem, settings, track_incumbents);
  if (!sub.success) { return {.error_message = sub.error_message}; }

  start_log_streaming(sub.job_id);
  auto poll = poll_for_completion(sub.job_id);
  stop_log_streaming();

  if (!poll.completed) { return {.error_message = poll.error_message}; }

  auto result = get_mip_result<i_t, f_t>(sub.job_id);
  if (result.success) { delete_job(sub.job_id); }

  GRPC_CLIENT_THROUGHPUT_LOG(config_, "end_to_end_mip", 0, solve_t0);

  return result;
}

// =============================================================================
// Chunked Transfer utils (upload and download)
// =============================================================================

template <typename i_t, typename f_t>
bool grpc_client_t::upload_chunked_arrays(const cpu_optimization_problem_t<i_t, f_t>& problem,
                                          const cuopt::remote::ChunkedProblemHeader& header,
                                          std::string& job_id_out)
{
  job_id_out.clear();
  auto upload_t0 = std::chrono::steady_clock::now();

  // --- 1. StartChunkedUpload ---
  std::string upload_id;
  {
    grpc::ClientContext context;
    set_rpc_deadline(context, kDefaultRpcTimeoutSeconds);
    cuopt::remote::StartChunkedUploadRequest request;
    *request.mutable_problem_header() = header;

    cuopt::remote::StartChunkedUploadResponse response;
    auto status = impl_->stub->StartChunkedUpload(&context, request, &response);

    if (!status.ok()) {
      last_error_ = "StartChunkedUpload failed: " + status.error_message();
      return false;
    }

    upload_id = response.upload_id();
    if (response.max_message_bytes() > 0) {
      server_max_message_bytes_.store(response.max_message_bytes(), std::memory_order_relaxed);
    }
  }

  GRPC_CLIENT_DEBUG_LOG(config_, "[grpc_client] ChunkedUpload started, upload_id=" << upload_id);

  // --- 2. Build chunk requests directly from problem arrays ---
  int64_t chunk_data_budget = config_.chunk_size_bytes;
  if (chunk_data_budget <= 0) { chunk_data_budget = 1LL * 1024 * 1024; }
  int64_t srv_max = server_max_message_bytes_.load(std::memory_order_relaxed);
  if (srv_max > 0 && chunk_data_budget > srv_max * 9 / 10) { chunk_data_budget = srv_max * 9 / 10; }

  auto chunk_requests = build_array_chunk_requests(problem, upload_id, chunk_data_budget);

  // --- 3. Send each chunk request ---
  int total_chunks         = 0;
  int64_t total_bytes_sent = 0;

  for (auto& chunk_request : chunk_requests) {
    grpc::ClientContext chunk_context;
    set_rpc_deadline(chunk_context, kDefaultRpcTimeoutSeconds);
    cuopt::remote::SendArrayChunkResponse chunk_response;
    auto status = impl_->stub->SendArrayChunk(&chunk_context, chunk_request, &chunk_response);

    if (!status.ok()) {
      last_error_ = "SendArrayChunk failed: " + status.error_message();
      return false;
    }

    total_bytes_sent += chunk_request.chunk().data().size();
    ++total_chunks;
  }

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] ChunkedUpload sent " << total_chunks << " chunk requests");

  // --- 4. FinishChunkedUpload ---
  {
    grpc::ClientContext context;
    set_rpc_deadline(context, kDefaultRpcTimeoutSeconds);
    cuopt::remote::FinishChunkedUploadRequest request;
    request.set_upload_id(upload_id);

    cuopt::remote::SubmitJobResponse response;
    auto status = impl_->stub->FinishChunkedUpload(&context, request, &response);

    if (!status.ok()) {
      last_error_ = "FinishChunkedUpload failed: " + status.error_message();
      return false;
    }

    job_id_out = response.job_id();
  }

  GRPC_CLIENT_THROUGHPUT_LOG(config_, "upload_chunked", total_bytes_sent, upload_t0);
  GRPC_CLIENT_DEBUG_LOG(
    config_,
    "[grpc_client] ChunkedUpload complete: " << total_chunks << " chunks, job_id=" << job_id_out);
  return true;
}

bool grpc_client_t::get_result_or_download(const std::string& job_id,
                                           downloaded_result_t& result_out)
{
  result_out = downloaded_result_t{};

  if (!impl_->stub) {
    last_error_ = "Not connected to server";
    return false;
  }

  int64_t result_size_hint = 0;
  {
    grpc::ClientContext context;
    set_rpc_deadline(context, kDefaultRpcTimeoutSeconds);
    auto request = build_status_request(job_id);
    cuopt::remote::StatusResponse response;
    auto status = impl_->stub->CheckStatus(&context, request, &response);

    if (status.ok()) {
      result_size_hint = response.result_size_bytes();
      if (response.max_message_bytes() > 0) {
        server_max_message_bytes_.store(response.max_message_bytes(), std::memory_order_relaxed);
      }
    }
  }

  int64_t srv_max_msg   = server_max_message_bytes_.load(std::memory_order_relaxed);
  int64_t effective_max = config_.max_message_bytes;
  if (srv_max_msg > 0 && srv_max_msg < effective_max) { effective_max = srv_max_msg; }

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] get_result_or_download: result_size_hint="
                          << result_size_hint << " bytes, client_max=" << config_.max_message_bytes
                          << ", server_max=" << srv_max_msg << ", effective_max=" << effective_max);

  if (result_size_hint > 0 && effective_max > 0 && result_size_hint > effective_max) {
    GRPC_CLIENT_DEBUG_LOG(config_,
                          "[grpc_client] Using chunked download directly (result_size_hint="
                            << result_size_hint << " > effective_max=" << effective_max << ")");
    return download_chunked_result(job_id, result_out);
  }

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] Attempting unary GetResult (result_size_hint="
                          << result_size_hint << " <= effective_max=" << effective_max << ")");

  auto download_t0 = std::chrono::steady_clock::now();

  grpc::ClientContext context;
  set_rpc_deadline(context, kDefaultRpcTimeoutSeconds);
  auto request  = build_get_result_request(job_id);
  auto response = std::make_unique<cuopt::remote::ResultResponse>();
  auto status   = impl_->stub->GetResult(&context, request, response.get());

  if (status.ok() && response->status() == cuopt::remote::SUCCESS) {
    if (response->has_lp_solution() || response->has_mip_solution()) {
      GRPC_CLIENT_THROUGHPUT_LOG(config_, "download_unary", response->ByteSizeLong(), download_t0);
      GRPC_CLIENT_DEBUG_LOG(config_,
                            "[grpc_client] Unary GetResult succeeded, result_size="
                              << response->ByteSizeLong() << " bytes");
      result_out.was_chunked = false;
      result_out.response    = std::move(response);
      return true;
    }
    last_error_ = "GetResult succeeded but no solution in response";
    return false;
  }

  if (status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED) {
    GRPC_CLIENT_DEBUG_LOG(config_,
                          "[grpc_client] GetResult rejected (RESOURCE_EXHAUSTED), "
                          "falling back to chunked download");
    return download_chunked_result(job_id, result_out);
  }

  if (!status.ok()) {
    last_error_ = "GetResult failed: " + status.error_message();
  } else if (response->status() != cuopt::remote::SUCCESS) {
    last_error_ = "GetResult indicates failure: " + response->error_message();
  }
  return false;
}

bool grpc_client_t::download_chunked_result(const std::string& job_id,
                                            downloaded_result_t& result_out)
{
  result_out.was_chunked = true;
  result_out.chunked_arrays.clear();
  auto download_t0 = std::chrono::steady_clock::now();

  GRPC_CLIENT_DEBUG_LOG(config_, "[grpc_client] Starting chunked download for job " << job_id);

  // --- 1. StartChunkedDownload ---
  std::string download_id;
  auto header = std::make_unique<cuopt::remote::ChunkedResultHeader>();
  {
    grpc::ClientContext context;
    set_rpc_deadline(context, kDefaultRpcTimeoutSeconds);
    cuopt::remote::StartChunkedDownloadRequest request;
    request.set_job_id(job_id);

    cuopt::remote::StartChunkedDownloadResponse response;
    auto status = impl_->stub->StartChunkedDownload(&context, request, &response);

    if (!status.ok()) {
      last_error_ = "StartChunkedDownload failed: " + status.error_message();
      return false;
    }

    download_id = response.download_id();
    *header     = response.header();
    if (response.max_message_bytes() > 0) {
      server_max_message_bytes_.store(response.max_message_bytes(), std::memory_order_relaxed);
    }
  }

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] ChunkedDownload started, download_id="
                          << download_id << " arrays=" << header->arrays_size()
                          << " problem_category=" << header->problem_category());

  // --- 2. Fetch each array via GetResultChunk RPCs ---
  int64_t chunk_data_budget = config_.chunk_size_bytes;
  if (chunk_data_budget <= 0) { chunk_data_budget = 1LL * 1024 * 1024; }
  int64_t dl_srv_max = server_max_message_bytes_.load(std::memory_order_relaxed);
  if (dl_srv_max > 0 && chunk_data_budget > dl_srv_max * 9 / 10) {
    chunk_data_budget = dl_srv_max * 9 / 10;
  }

  int total_chunks             = 0;
  int64_t total_bytes_received = 0;

  for (const auto& arr_desc : header->arrays()) {
    auto field_id       = arr_desc.field_id();
    int64_t total_elems = arr_desc.total_elements();
    int64_t elem_size   = arr_desc.element_size_bytes();
    if (total_elems <= 0) continue;

    if (elem_size <= 0) {
      last_error_ = "Invalid chunk metadata: non-positive element_size_bytes for field " +
                    std::to_string(field_id);
      return false;
    }
    // Guard against total_elems * elem_size overflowing int64_t (both are
    // positive at this point, so dividing INT64_MAX is safe and avoids the
    // signed/unsigned pitfall of casting SIZE_MAX to int64_t).
    if (total_elems > std::numeric_limits<int64_t>::max() / elem_size) {
      last_error_ =
        "Invalid chunk metadata: total byte size overflow for field " + std::to_string(field_id);
      return false;
    }

    int64_t elems_per_chunk = chunk_data_budget / elem_size;
    if (elems_per_chunk <= 0) elems_per_chunk = 1;

    std::vector<uint8_t> array_bytes(static_cast<size_t>(total_elems * elem_size));

    for (int64_t elem_offset = 0; elem_offset < total_elems; elem_offset += elems_per_chunk) {
      int64_t elems_wanted = std::min(elems_per_chunk, total_elems - elem_offset);

      grpc::ClientContext chunk_ctx;
      set_rpc_deadline(chunk_ctx, kDefaultRpcTimeoutSeconds);
      cuopt::remote::GetResultChunkRequest chunk_req;
      chunk_req.set_download_id(download_id);
      chunk_req.set_field_id(field_id);
      chunk_req.set_element_offset(elem_offset);
      chunk_req.set_max_elements(elems_wanted);

      cuopt::remote::GetResultChunkResponse chunk_resp;
      auto status = impl_->stub->GetResultChunk(&chunk_ctx, chunk_req, &chunk_resp);

      if (!status.ok()) {
        last_error_ = "GetResultChunk failed: " + status.error_message();
        return false;
      }

      int64_t elems_received = chunk_resp.elements_in_chunk();
      const auto& data       = chunk_resp.data();

      if (elems_received < 0 || elems_received > elems_wanted ||
          elems_received > total_elems - elem_offset) {
        last_error_ = "GetResultChunk: invalid element count";
        return false;
      }
      if (static_cast<int64_t>(data.size()) != elems_received * elem_size) {
        last_error_ = "GetResultChunk: data size mismatch";
        return false;
      }

      std::memcpy(array_bytes.data() + elem_offset * elem_size, data.data(), data.size());
      total_bytes_received += static_cast<int64_t>(data.size());
      ++total_chunks;
    }

    result_out.chunked_arrays[static_cast<int32_t>(field_id)] = std::move(array_bytes);
  }

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] ChunkedDownload fetched "
                          << total_chunks << " chunks for " << header->arrays_size() << " arrays");

  // --- 3. FinishChunkedDownload ---
  {
    grpc::ClientContext context;
    set_rpc_deadline(context, kDefaultRpcTimeoutSeconds);
    cuopt::remote::FinishChunkedDownloadRequest request;
    request.set_download_id(download_id);

    cuopt::remote::FinishChunkedDownloadResponse response;
    auto status = impl_->stub->FinishChunkedDownload(&context, request, &response);

    if (!status.ok()) {
      GRPC_CLIENT_DEBUG_LOG(
        config_, "[grpc_client] FinishChunkedDownload warning: " << status.error_message());
    }
  }

  result_out.chunked_header = std::move(header);

  GRPC_CLIENT_THROUGHPUT_LOG(config_, "download_chunked", total_bytes_received, download_t0);
  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] ChunkedDownload complete: "
                          << total_chunks << " chunks, " << total_bytes_received << " bytes");

  return true;
}

// Explicit template instantiations
#if CUOPT_INSTANTIATE_FLOAT
template remote_lp_result_t<int32_t, float> grpc_client_t::solve_lp(
  const cpu_optimization_problem_t<int32_t, float>& problem,
  const pdlp_solver_settings_t<int32_t, float>& settings);
template remote_mip_result_t<int32_t, float> grpc_client_t::solve_mip(
  const cpu_optimization_problem_t<int32_t, float>& problem,
  const mip_solver_settings_t<int32_t, float>& settings,
  bool enable_incumbents);
template submit_result_t grpc_client_t::submit_lp(
  const cpu_optimization_problem_t<int32_t, float>& problem,
  const pdlp_solver_settings_t<int32_t, float>& settings);
template submit_result_t grpc_client_t::submit_mip(
  const cpu_optimization_problem_t<int32_t, float>& problem,
  const mip_solver_settings_t<int32_t, float>& settings,
  bool enable_incumbents);
template remote_lp_result_t<int32_t, float> grpc_client_t::get_lp_result(const std::string& job_id);
template remote_mip_result_t<int32_t, float> grpc_client_t::get_mip_result(
  const std::string& job_id);
template bool grpc_client_t::upload_chunked_arrays(
  const cpu_optimization_problem_t<int32_t, float>& problem,
  const cuopt::remote::ChunkedProblemHeader& header,
  std::string& job_id_out);
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template remote_lp_result_t<int32_t, double> grpc_client_t::solve_lp(
  const cpu_optimization_problem_t<int32_t, double>& problem,
  const pdlp_solver_settings_t<int32_t, double>& settings);
template remote_mip_result_t<int32_t, double> grpc_client_t::solve_mip(
  const cpu_optimization_problem_t<int32_t, double>& problem,
  const mip_solver_settings_t<int32_t, double>& settings,
  bool enable_incumbents);
template submit_result_t grpc_client_t::submit_lp(
  const cpu_optimization_problem_t<int32_t, double>& problem,
  const pdlp_solver_settings_t<int32_t, double>& settings);
template submit_result_t grpc_client_t::submit_mip(
  const cpu_optimization_problem_t<int32_t, double>& problem,
  const mip_solver_settings_t<int32_t, double>& settings,
  bool enable_incumbents);
template remote_lp_result_t<int32_t, double> grpc_client_t::get_lp_result(
  const std::string& job_id);
template remote_mip_result_t<int32_t, double> grpc_client_t::get_mip_result(
  const std::string& job_id);
template bool grpc_client_t::upload_chunked_arrays(
  const cpu_optimization_problem_t<int32_t, double>& problem,
  const cuopt::remote::ChunkedProblemHeader& header,
  std::string& job_id_out);
#endif

}  // namespace cuopt::linear_programming
