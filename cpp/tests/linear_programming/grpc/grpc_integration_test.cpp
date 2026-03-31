/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file grpc_integration_test.cpp
 * @brief Integration tests for gRPC client-server communication
 *
 * Tests are organized into shared-server fixtures to minimize server startup overhead.
 * Total target runtime: ~3 minutes.
 *
 * Fixture layout:
 *   NoServerTests          - Tests that don't need a server
 *   DefaultServerTests     - Shared server with default config (~21 tests)
 *   ChunkedUploadTests     - Shared server with --max-message-mb 256 (4 tests)
 *   PathSelectionTests     - Shared server with --max-message-bytes 4096 --verbose (4 tests)
 *   ErrorRecoveryTests     - Per-test server lifecycle (4 tests)
 *   TlsServerTests         - Shared TLS server (2 tests)
 *   MtlsServerTests        - Shared mTLS server (2 tests)
 *
 * Environment variables:
 *   CUOPT_GRPC_SERVER_PATH - Path to cuopt_grpc_server binary
 *   CUOPT_TEST_PORT_BASE   - Base port for test servers (default: 19000)
 *   RAPIDS_DATASET_ROOT_DIR - Path to test datasets
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/optimization_problem_utils.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <mps_parser/parser.hpp>
#include "grpc_client.hpp"

#include "grpc_test_log_capture.hpp"

#include <cuopt_remote_service.grpc.pb.h>
#include <grpcpp/grpcpp.h>

#include "grpc_service_mapper.hpp"

#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <random>
#include <string>
#include <thread>

using namespace cuopt::linear_programming;
using cuopt::linear_programming::testing::GrpcTestLogCapture;

namespace {

// =============================================================================
// Server Process Manager
// =============================================================================

class ServerProcess {
 public:
  ServerProcess() : pid_(-1), port_(0) {}
  ~ServerProcess() { stop(); }

  void set_tls_config(const std::string& root_certs,
                      const std::string& client_cert = "",
                      const std::string& client_key  = "")
  {
    tls_root_certs_  = root_certs;
    tls_client_cert_ = client_cert;
    tls_client_key_  = client_key;
  }

  bool start(int port, const std::vector<std::string>& extra_args = {})
  {
    port_ = port;

    std::string server_path = find_server_binary();
    if (server_path.empty()) {
      std::cerr << "Could not find cuopt_grpc_server binary\n";
      return false;
    }

    pid_ = fork();
    if (pid_ < 0) {
      std::cerr << "fork() failed\n";
      return false;
    }

    if (pid_ == 0) {
      std::vector<const char*> args;
      args.push_back(server_path.c_str());
      args.push_back("--port");
      std::string port_str = std::to_string(port);
      args.push_back(port_str.c_str());
      args.push_back("--workers");
      args.push_back("1");

      for (const auto& arg : extra_args) {
        args.push_back(arg.c_str());
      }
      args.push_back(nullptr);

      std::string log_file = "/tmp/cuopt_test_server_" + std::to_string(port) + ".log";
      int fd               = open(log_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (fd >= 0) {
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
        close(fd);
      }

      execv(server_path.c_str(), const_cast<char**>(args.data()));
      _exit(127);
    }

    return wait_for_ready(15000);
  }

  void stop()
  {
    if (pid_ > 0) {
      kill(pid_, SIGTERM);

      int status;
      int wait_ms = 0;
      while (wait_ms < 5000) {
        int ret = waitpid(pid_, &status, WNOHANG);
        if (ret != 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_ms += 100;
      }

      if (waitpid(pid_, &status, WNOHANG) == 0) {
        kill(pid_, SIGKILL);
        waitpid(pid_, &status, 0);
      }

      pid_ = -1;
    }
  }

  int port() const { return port_; }

  bool is_running() const
  {
    if (pid_ <= 0) return false;
    return kill(pid_, 0) == 0;
  }

  std::string log_path() const
  {
    if (port_ <= 0) return "";
    return "/tmp/cuopt_test_server_" + std::to_string(port_) + ".log";
  }

 private:
  std::string find_in_path(const std::string& name)
  {
    const char* path_env = std::getenv("PATH");
    if (!path_env) return "";

    std::string path_str(path_env);
    std::string::size_type start = 0;
    std::string::size_type end;

    while ((end = path_str.find(':', start)) != std::string::npos || start < path_str.size()) {
      std::string dir;
      if (end != std::string::npos) {
        dir   = path_str.substr(start, end - start);
        start = end + 1;
      } else {
        dir   = path_str.substr(start);
        start = path_str.size();
      }

      if (dir.empty()) continue;

      std::string full_path = dir + "/" + name;
      if (access(full_path.c_str(), X_OK) == 0) { return full_path; }
    }

    return "";
  }

  std::string find_server_binary()
  {
    const char* env_path = std::getenv("CUOPT_GRPC_SERVER_PATH");
    if (env_path && access(env_path, X_OK) == 0) { return env_path; }

    std::string path_result = find_in_path("cuopt_grpc_server");
    if (!path_result.empty()) { return path_result; }

    std::vector<std::string> paths = {
      "./cuopt_grpc_server",
      "../cuopt_grpc_server",
      "../../cuopt_grpc_server",
      "./build/cuopt_grpc_server",
      "../build/cuopt_grpc_server",
    };

    for (const auto& path : paths) {
      if (access(path.c_str(), X_OK) == 0) { return path; }
    }

    return "";
  }

  bool wait_for_ready(int timeout_ms)
  {
    auto start = std::chrono::steady_clock::now();

    while (true) {
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);

      if (elapsed.count() >= timeout_ms) { return false; }

      grpc_client_config_t config;
      config.server_address = "localhost:" + std::to_string(port_);

      if (!tls_root_certs_.empty()) {
        config.enable_tls      = true;
        config.tls_root_certs  = tls_root_certs_;
        config.tls_client_cert = tls_client_cert_;
        config.tls_client_key  = tls_client_key_;
      }

      grpc_client_t client(config);
      if (client.connect()) { return true; }

      int status;
      if (waitpid(pid_, &status, WNOHANG) != 0) {
        std::cerr << "Server process died during startup\n";
        return false;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  }

  pid_t pid_;
  int port_;
  std::string tls_root_certs_;
  std::string tls_client_cert_;
  std::string tls_client_key_;
};

int get_test_port()
{
  static std::atomic<int> port_counter{0};

  int base_port        = 19000;
  const char* env_base = std::getenv("CUOPT_TEST_PORT_BASE");
  if (env_base) { base_port = std::atoi(env_base); }

  return base_port + port_counter.fetch_add(1);
}

// =============================================================================
// TLS Certificate Generation (shared across TLS fixtures)
// =============================================================================

std::string g_tls_certs_dir;
bool g_tls_certs_ready = false;

bool ensure_test_certs()
{
  if (g_tls_certs_ready) return true;

  // Check for CI-provided certs
  const char* cert_folder = std::getenv("CERT_FOLDER");
  if (cert_folder) {
    g_tls_certs_dir   = cert_folder;
    g_tls_certs_ready = true;
    return true;
  }

  const char* ssl_certfile = std::getenv("CUOPT_SSL_CERTFILE");
  if (ssl_certfile) {
    g_tls_certs_dir   = std::filesystem::path(ssl_certfile).parent_path().string();
    g_tls_certs_ready = true;
    return true;
  }

  g_tls_certs_dir = "/tmp/cuopt_test_certs_" + std::to_string(getpid());
  std::filesystem::create_directories(g_tls_certs_dir);

  auto run = [](const std::string& cmd) { return std::system(cmd.c_str()) == 0; };

  std::string ca_key = g_tls_certs_dir + "/ca.key";
  std::string ca_crt = g_tls_certs_dir + "/ca.crt";
  if (!run("openssl req -x509 -newkey rsa:2048 -keyout " + ca_key + " -out " + ca_crt +
           " -days 1 -nodes -subj '/CN=TestCA' 2>/dev/null"))
    return false;

  std::string server_key = g_tls_certs_dir + "/server.key";
  std::string server_csr = g_tls_certs_dir + "/server.csr";
  std::string server_crt = g_tls_certs_dir + "/server.crt";
  if (!run("openssl req -newkey rsa:2048 -keyout " + server_key + " -out " + server_csr +
           " -nodes -subj '/CN=localhost' 2>/dev/null"))
    return false;
  if (!run("openssl x509 -req -in " + server_csr + " -CA " + ca_crt + " -CAkey " + ca_key +
           " -CAcreateserial -out " + server_crt + " -days 1 2>/dev/null"))
    return false;

  std::string client_key = g_tls_certs_dir + "/client.key";
  std::string client_csr = g_tls_certs_dir + "/client.csr";
  std::string client_crt = g_tls_certs_dir + "/client.crt";
  if (!run("openssl req -newkey rsa:2048 -keyout " + client_key + " -out " + client_csr +
           " -nodes -subj '/CN=TestClient' 2>/dev/null"))
    return false;
  if (!run("openssl x509 -req -in " + client_csr + " -CA " + ca_crt + " -CAkey " + ca_key +
           " -CAcreateserial -out " + client_crt + " -days 1 2>/dev/null"))
    return false;

  g_tls_certs_ready = true;
  return true;
}

std::string read_file_contents(const std::string& path)
{
  std::ifstream file(path);
  if (!file) return "";
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

// =============================================================================
// Base Test Class
// =============================================================================

class GrpcIntegrationTestBase : public ::testing::Test {
 protected:
  std::unique_ptr<grpc_client_t> create_client(grpc_client_config_t config = {})
  {
    config.server_address   = "localhost:" + std::to_string(port_);
    config.poll_interval_ms = 100;

    if (config.timeout_seconds == 3600) { config.timeout_seconds = 60; }

    config.enable_transfer_hash = true;

    auto client = std::make_unique<grpc_client_t>(config);
    if (!client->connect()) { return nullptr; }
    return client;
  }

  std::string get_test_data_path(const std::string& subdir, const std::string& filename)
  {
    const char* env_var      = std::getenv("RAPIDS_DATASET_ROOT_DIR");
    std::string dataset_root = env_var ? env_var : "./datasets";
    return dataset_root + "/" + subdir + "/" + filename;
  }

  std::string get_test_lp_path(const std::string& filename)
  {
    return get_test_data_path("linear_programming", filename);
  }

  std::string get_test_mip_path(const std::string& filename)
  {
    return get_test_data_path("mip", filename);
  }

  cpu_optimization_problem_t<int32_t, double> load_problem_from_mps(const std::string& mps_path)
  {
    auto mps_data = cuopt::mps_parser::parse_mps<int32_t, double>(mps_path);
    cpu_optimization_problem_t<int32_t, double> problem;
    populate_from_mps_data_model(&problem, mps_data);
    return problem;
  }

  cpu_optimization_problem_t<int32_t, double> create_simple_mip()
  {
    cpu_optimization_problem_t<int32_t, double> problem;

    std::vector<double> c = {1.0, 2.0};
    problem.set_objective_coefficients(c.data(), 2);
    problem.set_maximize(false);

    std::vector<double> A_values   = {1.0, 1.0};
    std::vector<int32_t> A_indices = {0, 1};
    std::vector<int32_t> A_offsets = {0, 2};
    problem.set_csr_constraint_matrix(A_values.data(), 2, A_indices.data(), 2, A_offsets.data(), 2);

    std::vector<double> var_lb = {0.0, 0.0};
    std::vector<double> var_ub = {1.0, 1.0};
    problem.set_variable_lower_bounds(var_lb.data(), 2);
    problem.set_variable_upper_bounds(var_ub.data(), 2);

    std::vector<var_t> var_types = {var_t::INTEGER, var_t::INTEGER};
    problem.set_variable_types(var_types.data(), 2);

    std::vector<double> con_lb = {1.0};
    std::vector<double> con_ub = {1e20};
    problem.set_constraint_lower_bounds(con_lb.data(), 1);
    problem.set_constraint_upper_bounds(con_ub.data(), 1);

    return problem;
  }

  void wait_for_job_done(grpc_client_t* client, const std::string& job_id, int max_seconds = 30)
  {
    for (int i = 0; i < max_seconds * 2; ++i) {
      auto status = client->check_status(job_id);
      if (status.status == job_status_t::COMPLETED || status.status == job_status_t::FAILED ||
          status.status == job_status_t::CANCELLED) {
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }

  int port_ = 0;
};

// =============================================================================
// No-Server Tests
// =============================================================================

class NoServerTests : public GrpcIntegrationTestBase {
 protected:
  void SetUp() override { port_ = get_test_port(); }
};

TEST_F(NoServerTests, ConnectToNonexistentServer)
{
  grpc_client_config_t config;
  config.server_address = "localhost:" + std::to_string(port_);

  GrpcTestLogCapture log_capture;
  config.debug_log_callback = log_capture.client_callback();

  grpc_client_t client(config);
  EXPECT_FALSE(client.connect());
  EXPECT_FALSE(client.get_last_error().empty());

  EXPECT_TRUE(log_capture.client_log_contains("Connection failed"))
    << "Expected failure log. Captured logs:\n"
    << log_capture.get_client_logs();
}

TEST_F(NoServerTests, LogCaptureInfrastructure)
{
  GrpcTestLogCapture log_capture;

  // -- client_log_count --
  log_capture.add_client_log("Test message 1");
  log_capture.add_client_log("Test message 2");
  log_capture.add_client_log("Another test");
  log_capture.add_client_log("Test message 3");

  EXPECT_EQ(log_capture.client_log_count("Test message"), 3);
  EXPECT_EQ(log_capture.client_log_count("Another"), 1);
  EXPECT_EQ(log_capture.client_log_count("Not found"), 0);

  // -- mark_test_start isolation with server log file --
  std::string tmp_log = "/tmp/cuopt_test_log_infra_" + std::to_string(getpid()) + ".log";
  {
    std::ofstream f(tmp_log);
    f << "[Phase1] Before mark\n";
    f.flush();
  }

  log_capture.set_server_log_path(tmp_log);
  log_capture.mark_test_start();

  // Logs written before the mark should be invisible
  EXPECT_FALSE(log_capture.server_log_contains("[Phase1]"))
    << "Should NOT see logs from before mark_test_start()";

  // Append new content after the mark
  {
    std::ofstream f(tmp_log, std::ios::app);
    f << "[Phase2] After mark\n";
    f.flush();
  }

  EXPECT_TRUE(log_capture.server_log_contains("[Phase2]"))
    << "Should see logs from after mark_test_start()";

  // get_all_server_logs bypasses the mark
  std::string all = log_capture.get_all_server_logs();
  EXPECT_TRUE(all.find("[Phase1]") != std::string::npos);
  EXPECT_TRUE(all.find("[Phase2]") != std::string::npos);

  // -- wait_for_server_log (content already present -> immediate return) --
  EXPECT_TRUE(log_capture.wait_for_server_log("[Phase2]", 500));
  EXPECT_FALSE(log_capture.wait_for_server_log("never_appears", 200));

  // Clean up
  std::filesystem::remove(tmp_log);
}

// =============================================================================
// Default Server Tests (shared server, default config)
// =============================================================================

class DefaultServerTests : public GrpcIntegrationTestBase {
 protected:
  static void SetUpTestSuite()
  {
    s_port_   = get_test_port();
    s_server_ = std::make_unique<ServerProcess>();
    ASSERT_TRUE(s_server_->start(s_port_, {"--enable-transfer-hash"}))
      << "Failed to start shared default server on port " << s_port_;
  }

  static void TearDownTestSuite()
  {
    if (s_server_) s_server_->stop();
    s_server_.reset();
  }

  void SetUp() override
  {
    ASSERT_NE(s_server_, nullptr) << "Shared server not running";
    port_ = s_port_;
  }

  std::string server_log_path() const { return s_server_->log_path(); }

  static std::unique_ptr<ServerProcess> s_server_;
  static int s_port_;
};

std::unique_ptr<ServerProcess> DefaultServerTests::s_server_;
int DefaultServerTests::s_port_ = 0;

// -- Connectivity --

TEST_F(DefaultServerTests, ServerAcceptsConnections)
{
  ASSERT_TRUE(s_server_->is_running());

  GrpcTestLogCapture log_capture;
  grpc_client_config_t config;
  config.debug_log_callback = log_capture.client_callback();

  auto client = create_client(config);
  ASSERT_NE(client, nullptr) << "Failed to connect to server";
  EXPECT_TRUE(client->is_connected());

  EXPECT_TRUE(log_capture.client_log_contains("Connecting to"))
    << "Expected connection log. Logs:\n"
    << log_capture.get_client_logs();
  EXPECT_TRUE(log_capture.client_log_contains("Connected successfully"))
    << "Expected success log. Logs:\n"
    << log_capture.get_client_logs();
}

// -- Status / Cancel / Delete on nonexistent jobs --

TEST_F(DefaultServerTests, CheckStatusNotFound)
{
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  auto status = client->check_status("nonexistent-job-id");
  EXPECT_TRUE(status.success);
  EXPECT_EQ(status.status, job_status_t::NOT_FOUND);
}

TEST_F(DefaultServerTests, CancelNonexistentJob)
{
  auto client = create_client();
  ASSERT_NE(client, nullptr);
  auto result = client->cancel_job("nonexistent-job-id");
  EXPECT_EQ(result.job_status, job_status_t::NOT_FOUND);
}

TEST_F(DefaultServerTests, DeleteNonexistentJob)
{
  auto client = create_client();
  ASSERT_NE(client, nullptr);
  bool deleted = client->delete_job("nonexistent-job-id");
  EXPECT_FALSE(deleted);
  EXPECT_FALSE(client->get_last_error().empty());
}

TEST_F(DefaultServerTests, StreamLogsNotFound)
{
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  bool callback_called = false;
  bool result =
    client->stream_logs("nonexistent-job-id", 0, [&callback_called](const std::string&, bool) {
      callback_called = true;
      return true;
    });

  EXPECT_FALSE(callback_called);
  EXPECT_FALSE(result);
}

TEST_F(DefaultServerTests, GetResultNonexistentJob)
{
  auto client = create_client();
  ASSERT_NE(client, nullptr);
  auto result = client->get_lp_result<int32_t, double>("nonexistent-job-12345");
  EXPECT_FALSE(result.success);
}

// -- LP Solves --

TEST_F(DefaultServerTests, SolveLPPolling)
{
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto submit_result = client->submit_lp(problem, settings);
  ASSERT_TRUE(submit_result.success) << submit_result.error_message;
  EXPECT_FALSE(submit_result.job_id.empty());

  job_status_t final_status = job_status_t::QUEUED;
  for (int i = 0; i < 60; ++i) {
    auto status = client->check_status(submit_result.job_id);
    ASSERT_TRUE(status.success) << status.error_message;
    final_status = status.status;
    if (final_status == job_status_t::COMPLETED || final_status == job_status_t::FAILED) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  EXPECT_EQ(final_status, job_status_t::COMPLETED)
    << "Status: " << job_status_to_string(final_status);

  auto result = client->get_lp_result<int32_t, double>(submit_result.job_id);
  EXPECT_TRUE(result.success) << result.error_message;
  ASSERT_NE(result.solution, nullptr);
  EXPECT_NEAR(result.solution->get_objective_value(), -464.753, 1.0);
}

TEST_F(DefaultServerTests, SolveLPWaitRPC)
{
  grpc_client_config_t config;
  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto result = client->solve_lp(problem, settings);
  EXPECT_TRUE(result.success) << result.error_message;
  ASSERT_NE(result.solution, nullptr);
  EXPECT_NEAR(result.solution->get_objective_value(), -464.753, 1.0);
}

TEST_F(DefaultServerTests, SolveInfeasibleLP)
{
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  cpu_optimization_problem_t<int32_t, double> problem;
  std::vector<double> var_lb   = {1.0};
  std::vector<double> var_ub   = {0.0};
  std::vector<double> obj      = {1.0};
  std::vector<int32_t> offsets = {0};

  problem.set_variable_lower_bounds(var_lb.data(), 1);
  problem.set_variable_upper_bounds(var_ub.data(), 1);
  problem.set_objective_coefficients(obj.data(), 1);
  problem.set_maximize(false);
  problem.set_csr_constraint_matrix(nullptr, 0, nullptr, 0, offsets.data(), 1);
  problem.set_constraint_lower_bounds(nullptr, 0);
  problem.set_constraint_upper_bounds(nullptr, 0);

  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto result = client->solve_lp(problem, settings);
  ASSERT_TRUE(result.success) << result.error_message;
  ASSERT_NE(result.solution, nullptr);
  auto status = result.solution->get_termination_status();
  EXPECT_NE(status, pdlp_termination_status_t::Optimal)
    << "Expected non-optimal termination for infeasible problem";
}

// -- MIP Solve --

TEST_F(DefaultServerTests, SolveMIPBlocking)
{
  auto client = create_client();
  ASSERT_NE(client, nullptr);
  auto problem = create_simple_mip();

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto result = client->solve_mip(problem, settings, false);
  EXPECT_TRUE(result.success) << result.error_message;
  ASSERT_NE(result.solution, nullptr);
  EXPECT_EQ(result.solution->get_termination_status(), mip_termination_status_t::Optimal);
  EXPECT_NEAR(result.solution->get_objective_value(), 1.0, 0.01);
}

// -- Explicit Async LP Flow (submit/poll/get/delete) --

TEST_F(DefaultServerTests, ExplicitAsyncLPFlow)
{
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto submit_result = client->submit_lp(problem, settings);
  ASSERT_TRUE(submit_result.success) << submit_result.error_message;
  ASSERT_FALSE(submit_result.job_id.empty());
  std::string job_id = submit_result.job_id;

  wait_for_job_done(client.get(), job_id, 30);

  auto result = client->get_lp_result<int32_t, double>(job_id);
  EXPECT_TRUE(result.success) << result.error_message;
  ASSERT_NE(result.solution, nullptr);
  EXPECT_NEAR(result.solution->get_objective_value(), -464.753, 1.0);

  bool deleted = client->delete_job(job_id);
  EXPECT_TRUE(deleted);
}

// -- Log Verification --

TEST_F(DefaultServerTests, ServerLogsJobProcessing)
{
  GrpcTestLogCapture log_capture;
  log_capture.set_server_log_path(server_log_path());
  log_capture.mark_test_start();

  auto client = create_client();
  ASSERT_NE(client, nullptr);

  auto problem = create_simple_mip();
  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;
  auto result         = client->solve_mip(problem, settings, false);
  EXPECT_TRUE(result.success) << result.error_message;

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_TRUE(log_capture.server_log_contains("[Worker"))
    << "Expected worker logs. Server log: " << server_log_path();
}

TEST_F(DefaultServerTests, ClientDebugLogsSubmission)
{
  GrpcTestLogCapture log_capture;
  grpc_client_config_t config;
  config.debug_log_callback = log_capture.client_callback();
  auto client               = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto result = client->solve_lp(problem, settings);
  EXPECT_TRUE(result.success) << result.error_message;

  EXPECT_TRUE(log_capture.client_log_contains("submit")) << "Expected submit log. Logs:\n"
                                                         << log_capture.get_client_logs();
  EXPECT_TRUE(log_capture.client_log_contains_pattern("job_id=[a-f0-9-]+"))
    << "Expected job_id pattern. Logs:\n"
    << log_capture.get_client_logs();
}

// -- Multiple & Concurrent Solves --

TEST_F(DefaultServerTests, MultipleSequentialSolves)
{
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  for (int i = 0; i < 3; ++i) {
    std::string mps_path = get_test_lp_path("afiro_original.mps");
    auto problem         = load_problem_from_mps(mps_path);
    pdlp_solver_settings_t<int32_t, double> settings;
    settings.time_limit = 10.0;

    auto result = client->solve_lp(problem, settings);
    EXPECT_TRUE(result.success) << "Solve #" << i << " failed: " << result.error_message;
    ASSERT_NE(result.solution, nullptr);
    EXPECT_NEAR(result.solution->get_objective_value(), -464.753, 1.0);
  }
}

TEST_F(DefaultServerTests, ConcurrentJobSubmission)
{
  auto client1 = create_client();
  auto client2 = create_client();
  ASSERT_NE(client1, nullptr);
  ASSERT_NE(client2, nullptr);

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  std::vector<std::pair<grpc_client_t*, std::string>> jobs;

  auto s1 = client1->submit_lp(problem, settings);
  ASSERT_TRUE(s1.success);
  jobs.push_back({client1.get(), s1.job_id});

  auto s2 = client2->submit_lp(problem, settings);
  ASSERT_TRUE(s2.success);
  jobs.push_back({client2.get(), s2.job_id});

  auto s3 = client1->submit_lp(problem, settings);
  ASSERT_TRUE(s3.success);
  jobs.push_back({client1.get(), s3.job_id});

  std::vector<bool> completed(3, false);
  int completed_count = 0;

  for (int poll = 0; poll < 120 && completed_count < 3; ++poll) {
    for (size_t i = 0; i < jobs.size(); ++i) {
      if (completed[i]) continue;
      auto status = jobs[i].first->check_status(jobs[i].second);
      ASSERT_TRUE(status.success);
      if (status.status == job_status_t::COMPLETED) {
        completed[i] = true;
        completed_count++;
      } else if (status.status == job_status_t::FAILED) {
        FAIL() << "Job " << i << " failed: " << status.message;
      }
    }
    if (completed_count < 3) { std::this_thread::sleep_for(std::chrono::milliseconds(500)); }
  }

  ASSERT_EQ(completed_count, 3) << "Not all jobs completed in time";

  for (size_t i = 0; i < jobs.size(); ++i) {
    auto result = jobs[i].first->get_lp_result<int32_t, double>(jobs[i].second);
    EXPECT_TRUE(result.success);
    ASSERT_NE(result.solution, nullptr);
    EXPECT_NEAR(result.solution->get_objective_value(), -464.753, 1.0);
    jobs[i].first->delete_job(jobs[i].second);
  }
}

// -- Unary Path Verification --

TEST_F(DefaultServerTests, VerifyUnaryUploadSmallProblem)
{
  GrpcTestLogCapture log_capture;
  grpc_client_config_t config;
  config.debug_log_callback = log_capture.client_callback();
  auto client               = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto result = client->solve_lp(problem, settings);
  EXPECT_TRUE(result.success) << result.error_message;

  EXPECT_TRUE(log_capture.client_log_contains("Unary submit succeeded"))
    << "Logs:\n"
    << log_capture.get_client_logs();
  EXPECT_FALSE(log_capture.client_log_contains("Starting streaming upload"))
    << "Logs:\n"
    << log_capture.get_client_logs();
}

TEST_F(DefaultServerTests, VerifyUnaryDownloadSmallResult)
{
  GrpcTestLogCapture log_capture;
  grpc_client_config_t config;
  config.debug_log_callback = log_capture.client_callback();
  auto client               = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto result = client->solve_lp(problem, settings);
  EXPECT_TRUE(result.success) << result.error_message;

  EXPECT_TRUE(log_capture.client_log_contains("Attempting unary GetResult"))
    << "Logs:\n"
    << log_capture.get_client_logs();
  EXPECT_TRUE(log_capture.client_log_contains("Unary GetResult succeeded"))
    << "Logs:\n"
    << log_capture.get_client_logs();
}

TEST_F(DefaultServerTests, SolveLPReturnsWarmStartData)
{
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto result = client->solve_lp(problem, settings);
  EXPECT_TRUE(result.success) << result.error_message;
  ASSERT_NE(result.solution, nullptr);

  EXPECT_TRUE(result.solution->has_warm_start_data())
    << "LP solution should contain PDLP warm start data";

  const auto& ws = result.solution->get_cpu_pdlp_warm_start_data();

  EXPECT_FALSE(ws.current_primal_solution_.empty())
    << "current_primal_solution should be populated";
  EXPECT_FALSE(ws.current_dual_solution_.empty()) << "current_dual_solution should be populated";
  EXPECT_FALSE(ws.initial_primal_average_.empty()) << "initial_primal_average should be populated";
  EXPECT_FALSE(ws.initial_dual_average_.empty()) << "initial_dual_average should be populated";
  EXPECT_FALSE(ws.current_ATY_.empty()) << "current_ATY should be populated";
  EXPECT_FALSE(ws.sum_primal_solutions_.empty()) << "sum_primal_solutions should be populated";
  EXPECT_FALSE(ws.sum_dual_solutions_.empty()) << "sum_dual_solutions should be populated";
  EXPECT_FALSE(ws.last_restart_duality_gap_primal_solution_.empty())
    << "last_restart_duality_gap_primal_solution should be populated";
  EXPECT_FALSE(ws.last_restart_duality_gap_dual_solution_.empty())
    << "last_restart_duality_gap_dual_solution should be populated";

  EXPECT_GT(ws.initial_primal_weight_, 0.0) << "initial_primal_weight should be positive";
  EXPECT_GT(ws.initial_step_size_, 0.0) << "initial_step_size should be positive";
  EXPECT_GE(ws.total_pdlp_iterations_, 0) << "total_pdlp_iterations should be non-negative";
  EXPECT_GE(ws.total_pdhg_iterations_, 0) << "total_pdhg_iterations should be non-negative";
}

// -- MIP Log Callback --

TEST_F(DefaultServerTests, SolveMIPWithLogCallback)
{
  std::vector<std::string> received_logs;
  std::mutex log_mutex;

  grpc_client_config_t config;
  config.timeout_seconds = 30;
  config.stream_logs     = true;
  config.log_callback    = [&](const std::string& line) {
    std::lock_guard<std::mutex> lock(log_mutex);
    received_logs.push_back(line);
  };

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_mip_path("bb_optimality.mps");
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit     = 10.0;
  settings.log_to_console = true;

  auto result = client->solve_mip(problem, settings, false);
  EXPECT_TRUE(result.success) << result.error_message;
}

// -- Incumbent Callbacks --

TEST_F(DefaultServerTests, IncumbentCallbacksMIP)
{
  std::vector<double> incumbent_objectives;
  std::mutex incumbent_mutex;

  grpc_client_config_t config;
  config.timeout_seconds    = 30;
  config.incumbent_callback = [&](int64_t, double objective, const std::vector<double>&) {
    std::lock_guard<std::mutex> lock(incumbent_mutex);
    incumbent_objectives.push_back(objective);
    return true;
  };

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_mip_path("neos5-free-bound.mps");
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto result = client->solve_mip(problem, settings, true);
  EXPECT_TRUE(result.success) << result.error_message;

  if (incumbent_objectives.size() > 1) {
    for (size_t i = 1; i < incumbent_objectives.size(); ++i) {
      EXPECT_LE(incumbent_objectives[i], incumbent_objectives[i - 1] + 1e-6);
    }
  }
}

TEST_F(DefaultServerTests, IncumbentCallbackCancelsSolve)
{
  int callback_count = 0;

  grpc_client_config_t config;
  config.timeout_seconds    = 30;
  config.incumbent_callback = [&](int64_t, double, const std::vector<double>&) {
    return ++callback_count < 2;
  };

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_mip_path("neos5-free-bound.mps");
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto start  = std::chrono::steady_clock::now();
  auto result = client->solve_mip(problem, settings, true);
  auto elapsed =
    std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);

  EXPECT_LT(elapsed.count(), 25) << "Solve should have cancelled early";
}

// -- Cancel Running Job --

TEST_F(DefaultServerTests, CancelRunningJob)
{
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_mip_path("neos5-free-bound.mps");
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 120.0;

  auto submit_result = client->submit_mip(problem, settings);
  ASSERT_TRUE(submit_result.success);
  std::string job_id = submit_result.job_id;

  std::this_thread::sleep_for(std::chrono::seconds(2));

  auto cancel_result = client->cancel_job(job_id);
  EXPECT_TRUE(cancel_result.job_status == job_status_t::CANCELLED ||
              cancel_result.job_status == job_status_t::COMPLETED ||
              cancel_result.job_status == job_status_t::PROCESSING ||
              cancel_result.job_status == job_status_t::FAILED)
    << "Unexpected job_status=" << static_cast<int>(cancel_result.job_status)
    << " message=" << cancel_result.message;

  // Wait for worker to free up before next test
  wait_for_job_done(client.get(), job_id, 15);
  client->delete_job(job_id);
}

// =============================================================================
// Chunked Upload Tests (--max-message-mb 256)
// =============================================================================

class ChunkedUploadTests : public GrpcIntegrationTestBase {
 protected:
  static void SetUpTestSuite()
  {
    s_port_   = get_test_port();
    s_server_ = std::make_unique<ServerProcess>();
    ASSERT_TRUE(s_server_->start(s_port_, {"--max-message-mb", "256"}))
      << "Failed to start chunked upload server";
  }

  static void TearDownTestSuite()
  {
    if (s_server_) s_server_->stop();
    s_server_.reset();
  }

  void SetUp() override
  {
    ASSERT_NE(s_server_, nullptr);
    port_ = s_port_;
  }

  void TearDown() override
  {
    if (HasFailure() && s_server_) {
      std::string log_file = s_server_->log_path();
      if (!log_file.empty()) {
        std::ifstream f(log_file);
        if (f) {
          std::cerr << "\n=== Server log (" << log_file << ") ===\n"
                    << f.rdbuf() << "\n=== End server log ===\n";
        }
      }
    }
  }

  static std::unique_ptr<ServerProcess> s_server_;
  static int s_port_;
};

std::unique_ptr<ServerProcess> ChunkedUploadTests::s_server_;
int ChunkedUploadTests::s_port_ = 0;

TEST_F(ChunkedUploadTests, ChunkedUploadLP)
{
  grpc_client_config_t config;
  config.timeout_seconds               = 60;
  config.chunk_size_bytes              = 8 * 1024;
  config.chunked_array_threshold_bytes = 0;  // Force chunked upload

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto result = client->solve_lp(problem, settings);
  EXPECT_TRUE(result.success) << result.error_message;
  ASSERT_NE(result.solution, nullptr);
  EXPECT_NEAR(result.solution->get_objective_value(), -464.753, 1.0);
}

TEST_F(ChunkedUploadTests, ChunkedUploadMIP)
{
  grpc_client_config_t config;
  config.timeout_seconds               = 60;
  config.chunk_size_bytes              = 4 * 1024;
  config.chunked_array_threshold_bytes = 0;  // Force chunked upload

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_mip_path("sudoku.mps");
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto result = client->solve_mip(problem, settings, false);
  EXPECT_TRUE(result.success) << result.error_message;
}

TEST_F(ChunkedUploadTests, ConcurrentChunkedUploads)
{
  const int num_clients = 3;
  std::vector<std::unique_ptr<grpc_client_t>> clients;

  for (int i = 0; i < num_clients; ++i) {
    grpc_client_config_t config;
    config.timeout_seconds               = 60;
    config.chunk_size_bytes              = 4 * 1024;
    config.chunked_array_threshold_bytes = 0;
    auto client                          = create_client(config);
    ASSERT_NE(client, nullptr);
    clients.push_back(std::move(client));
  }

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  std::atomic<int> success_count{0};

  auto solve_task = [&](int idx) -> bool {
    auto result = clients[idx]->solve_lp(problem, settings);
    if (result.success && result.solution &&
        std::abs(result.solution->get_objective_value() - (-464.753)) < 1.0) {
      success_count++;
      return true;
    }
    return false;
  };

  std::vector<std::future<bool>> futures;
  for (int i = 0; i < num_clients; ++i) {
    futures.push_back(std::async(std::launch::async, solve_task, i));
  }

  for (int i = 0; i < num_clients; ++i) {
    EXPECT_TRUE(futures[i].get()) << "Client " << i << " failed";
  }

  EXPECT_EQ(success_count.load(), num_clients);
}

TEST_F(ChunkedUploadTests, UnaryFallbackSmallProblem)
{
  grpc_client_config_t config;
  config.timeout_seconds               = 60;
  config.chunked_array_threshold_bytes = 100 * 1024 * 1024;  // 100 MiB, well above afiro size

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto result = client->solve_lp(problem, settings);
  EXPECT_TRUE(result.success) << result.error_message;
  ASSERT_NE(result.solution, nullptr);
  EXPECT_NEAR(result.solution->get_objective_value(), -464.753, 1.0);
}

// =============================================================================
// Path Selection Tests (unary vs chunked IPC and result retrieval)
//
// Uses --max-message-bytes to set a very low logical threshold so that even
// small test problems exercise both unary and chunked code paths.
// Uses --verbose so the server emits IPC path tags we can verify in logs.
// =============================================================================

class PathSelectionTests : public GrpcIntegrationTestBase {
 protected:
  static void SetUpTestSuite()
  {
    s_port_   = get_test_port();
    s_server_ = std::make_unique<ServerProcess>();
    // Small threshold (clamped to 4 KiB) forces chunked result downloads for
    // anything larger than ~4 KB, exercising the chunked download path.
    ASSERT_TRUE(s_server_->start(s_port_, {"--max-message-bytes", "4096", "--verbose"}))
      << "Failed to start path-selection server";
  }

  static void TearDownTestSuite()
  {
    if (s_server_) s_server_->stop();
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

std::unique_ptr<ServerProcess> PathSelectionTests::s_server_;
int PathSelectionTests::s_port_ = 0;

// Unary upload for a small LP (afiro). The result is small enough that
// the server returns it via unary GetResult. We verify the upload and
// result paths in the server logs but don't assert the download method
// since the result size may or may not exceed the 4 KiB threshold.
TEST_F(PathSelectionTests, UnaryUploadLPWithPathLogging)
{
  GrpcTestLogCapture log_capture;
  log_capture.set_server_log_path(server_log_path());
  log_capture.mark_test_start();

  grpc_client_config_t config;
  config.timeout_seconds               = 60;
  config.chunked_array_threshold_bytes = 100 * 1024 * 1024;  // high threshold => unary upload
  config.debug_log_callback            = log_capture.client_callback();

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto result = client->solve_lp(problem, settings);
  EXPECT_TRUE(result.success) << result.error_message;
  ASSERT_NE(result.solution, nullptr);
  EXPECT_NEAR(result.solution->get_objective_value(), -464.753, 1.0);

  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Worker should have received via the UNARY path
  EXPECT_TRUE(log_capture.wait_for_server_log("[Worker] IPC path: UNARY LP", 5000))
    << "Expected UNARY LP path in server log.\nServer log:\n"
    << log_capture.get_server_logs();

  // Worker should have serialized the result
  EXPECT_TRUE(log_capture.server_log_contains("[Worker] Result path: LP solution"))
    << "Expected LP result path in server log.\nServer log:\n"
    << log_capture.get_server_logs();
}

// Chunked upload, verify server receives via CHUNKED path
TEST_F(PathSelectionTests, ChunkedUploadLPWithPathLogging)
{
  GrpcTestLogCapture log_capture;
  log_capture.set_server_log_path(server_log_path());
  log_capture.mark_test_start();

  grpc_client_config_t config;
  config.timeout_seconds               = 60;
  config.chunk_size_bytes              = 4 * 1024;
  config.chunked_array_threshold_bytes = 0;  // force chunked upload
  config.debug_log_callback            = log_capture.client_callback();

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto result = client->solve_lp(problem, settings);
  EXPECT_TRUE(result.success) << result.error_message;
  ASSERT_NE(result.solution, nullptr);
  EXPECT_NEAR(result.solution->get_objective_value(), -464.753, 1.0);

  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Worker should have received via the CHUNKED path
  EXPECT_TRUE(log_capture.wait_for_server_log("[Worker] IPC path: CHUNKED", 5000))
    << "Expected CHUNKED path in server log.\nServer log:\n"
    << log_capture.get_server_logs();

  // Server main process should have logged FinishChunkedUpload
  EXPECT_TRUE(log_capture.server_log_contains("FinishChunkedUpload: CHUNKED path"))
    << "Expected FinishChunkedUpload log.\nServer log:\n"
    << log_capture.get_server_logs();
}

// Chunked upload + chunked result download for MIP.
// sudoku.mps produces ~5.8 KB result which exceeds the 4 KB threshold,
// so the client should use chunked download.
TEST_F(PathSelectionTests, ChunkedUploadAndChunkedDownloadMIP)
{
  GrpcTestLogCapture log_capture;
  log_capture.set_server_log_path(server_log_path());
  log_capture.mark_test_start();

  grpc_client_config_t config;
  config.timeout_seconds               = 60;
  config.chunk_size_bytes              = 4 * 1024;
  config.chunked_array_threshold_bytes = 0;  // force chunked upload
  config.debug_log_callback            = log_capture.client_callback();

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_mip_path("sudoku.mps");
  auto problem         = load_problem_from_mps(mps_path);
  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto result = client->solve_mip(problem, settings, false);
  EXPECT_TRUE(result.success) << result.error_message;

  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Upload should have gone through the CHUNKED path
  EXPECT_TRUE(log_capture.wait_for_server_log("[Worker] IPC path: CHUNKED", 5000))
    << "Expected CHUNKED upload path in server log.\nServer log:\n"
    << log_capture.get_server_logs();

  // Client should have used chunked download (result > 4096 bytes)
  EXPECT_TRUE(log_capture.client_log_contains("chunked download") ||
              log_capture.client_log_contains("ChunkedDownload"))
    << "Expected chunked download in client log.\nClient log:\n"
    << log_capture.get_client_logs();

  // Server should log CHUNKED response
  EXPECT_TRUE(log_capture.wait_for_server_log("StartChunkedDownload: CHUNKED response", 5000))
    << "Expected chunked download path in server log.\nServer log:\n"
    << log_capture.get_server_logs();
}

// MIP path: unary upload, verify UNARY MIP tag
TEST_F(PathSelectionTests, UnaryUploadMIPWithPathLogging)
{
  GrpcTestLogCapture log_capture;
  log_capture.set_server_log_path(server_log_path());
  log_capture.mark_test_start();

  grpc_client_config_t config;
  config.timeout_seconds    = 60;
  config.debug_log_callback = log_capture.client_callback();

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_mip_path("bb_optimality.mps");
  auto problem         = load_problem_from_mps(mps_path);
  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto result = client->solve_mip(problem, settings, false);
  EXPECT_TRUE(result.success) << result.error_message;

  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  EXPECT_TRUE(log_capture.wait_for_server_log("[Worker] IPC path: UNARY MIP", 5000))
    << "Expected UNARY MIP path in server log.\nServer log:\n"
    << log_capture.get_server_logs();

  EXPECT_TRUE(log_capture.server_log_contains("[Worker] Result path: MIP solution"))
    << "Expected MIP result path in server log.\nServer log:\n"
    << log_capture.get_server_logs();
}

// =============================================================================
// Error Recovery Tests (per-test server lifecycle)
// =============================================================================

class ErrorRecoveryTests : public GrpcIntegrationTestBase {
 protected:
  void SetUp() override { port_ = get_test_port(); }
  void TearDown() override { server_.stop(); }

  bool start_server(const std::vector<std::string>& extra_args = {})
  {
    return server_.start(port_, extra_args);
  }

  ServerProcess server_;
};

TEST_F(ErrorRecoveryTests, ClientReconnectsAfterServerRestart)
{
  ASSERT_TRUE(start_server());
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  auto status_before = client->check_status("test-job");
  EXPECT_TRUE(status_before.success);

  server_.stop();
  EXPECT_FALSE(server_.is_running());

  auto status_down = client->check_status("test-job");
  EXPECT_FALSE(status_down.success);

  ASSERT_TRUE(start_server());

  auto status_after = client->check_status("test-job");
  EXPECT_TRUE(status_after.success) << "Should auto-reconnect: " << status_after.error_message;
}

TEST_F(ErrorRecoveryTests, ClientHandlesServerCrashDuringSolve)
{
  ASSERT_TRUE(start_server());
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_mip_path("neos5-free-bound.mps");
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 120.0;

  auto submit_result = client->submit_mip(problem, settings);
  ASSERT_TRUE(submit_result.success);

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  server_.stop();

  auto status_result = client->check_status(submit_result.job_id);
  EXPECT_FALSE(status_result.success);
  EXPECT_FALSE(status_result.error_message.empty());
}

TEST_F(ErrorRecoveryTests, ClientTimeoutConfiguration)
{
  ASSERT_TRUE(start_server());

  grpc_client_config_t config;
  config.timeout_seconds  = 1;
  config.poll_interval_ms = 100;

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_mip_path("neos5-free-bound.mps");
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 60.0;

  auto submit_result = client->submit_mip(problem, settings);
  ASSERT_TRUE(submit_result.success);

  auto start     = std::chrono::steady_clock::now();
  bool completed = false;
  while (!completed) {
    auto elapsed =
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);
    if (elapsed.count() >= config.timeout_seconds) break;
    auto status = client->check_status(submit_result.job_id);
    if (status.status == job_status_t::COMPLETED || status.status == job_status_t::FAILED) {
      completed = true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  EXPECT_FALSE(completed) << "Complex MIP should not complete in 1 second";
  client->cancel_job(submit_result.job_id);
}

TEST_F(ErrorRecoveryTests, ChunkedUploadAfterServerRestart)
{
  ASSERT_TRUE(start_server({"--max-message-mb", "256"}));

  grpc_client_config_t config;
  config.timeout_seconds               = 30;
  config.chunk_size_bytes              = 4 * 1024;
  config.chunked_array_threshold_bytes = 0;

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_mip_path("sudoku.mps");
  auto problem         = load_problem_from_mps(mps_path);
  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto result1 = client->solve_mip(problem, settings, false);
  EXPECT_TRUE(result1.success) << result1.error_message;

  server_.stop();
  ASSERT_TRUE(start_server({"--max-message-mb", "256"}));

  auto client2 = create_client(config);
  ASSERT_NE(client2, nullptr);

  auto result2 = client2->solve_mip(problem, settings, false);
  EXPECT_TRUE(result2.success) << result2.error_message;
}

// =============================================================================
// TLS Tests
// =============================================================================

class TlsServerTests : public GrpcIntegrationTestBase {
 protected:
  static void SetUpTestSuite()
  {
    if (!ensure_test_certs()) {
      s_certs_available_ = false;
      return;
    }

    s_certs_available_ = std::filesystem::exists(g_tls_certs_dir + "/server.crt") &&
                         std::filesystem::exists(g_tls_certs_dir + "/server.key") &&
                         std::filesystem::exists(g_tls_certs_dir + "/ca.crt");

    if (!s_certs_available_) return;

    s_port_   = get_test_port();
    s_server_ = std::make_unique<ServerProcess>();

    std::string root_certs = read_file_contents(g_tls_certs_dir + "/ca.crt");
    s_server_->set_tls_config(root_certs);

    std::vector<std::string> args = {"--tls",
                                     "--tls-cert",
                                     g_tls_certs_dir + "/server.crt",
                                     "--tls-key",
                                     g_tls_certs_dir + "/server.key",
                                     "--tls-root",
                                     g_tls_certs_dir + "/ca.crt",
                                     "--enable-transfer-hash"};

    if (!s_server_->start(s_port_, args)) {
      s_server_.reset();
      s_certs_available_ = false;
    }
  }

  static void TearDownTestSuite()
  {
    if (s_server_) s_server_->stop();
    s_server_.reset();
  }

  void SetUp() override
  {
    if (!s_certs_available_) { GTEST_SKIP() << "TLS certificates not available"; }
    ASSERT_NE(s_server_, nullptr) << "TLS server not running";
    port_ = s_port_;
  }

  std::unique_ptr<grpc_client_t> create_tls_client()
  {
    grpc_client_config_t config;
    config.server_address  = "localhost:" + std::to_string(port_);
    config.timeout_seconds = 30;
    config.enable_tls      = true;
    config.tls_root_certs  = read_file_contents(g_tls_certs_dir + "/ca.crt");

    auto client = std::make_unique<grpc_client_t>(config);
    if (!client->connect()) return nullptr;
    return client;
  }

  static std::unique_ptr<ServerProcess> s_server_;
  static int s_port_;
  static bool s_certs_available_;
};

std::unique_ptr<ServerProcess> TlsServerTests::s_server_;
int TlsServerTests::s_port_             = 0;
bool TlsServerTests::s_certs_available_ = false;

TEST_F(TlsServerTests, BasicConnection)
{
  auto client = create_tls_client();
  ASSERT_NE(client, nullptr) << "Failed to connect with TLS";
  EXPECT_TRUE(client->is_connected());
}

TEST_F(TlsServerTests, SolveLP)
{
  auto client = create_tls_client();
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto result = client->solve_lp(problem, settings);
  EXPECT_TRUE(result.success) << result.error_message;
  ASSERT_NE(result.solution, nullptr);
  EXPECT_NEAR(result.solution->get_objective_value(), -464.753, 1.0);
}

// =============================================================================
// mTLS Tests
// =============================================================================

class MtlsServerTests : public GrpcIntegrationTestBase {
 protected:
  static void SetUpTestSuite()
  {
    if (!ensure_test_certs()) {
      s_certs_available_ = false;
      return;
    }

    s_certs_available_ = std::filesystem::exists(g_tls_certs_dir + "/client.crt") &&
                         std::filesystem::exists(g_tls_certs_dir + "/client.key") &&
                         std::filesystem::exists(g_tls_certs_dir + "/server.crt") &&
                         std::filesystem::exists(g_tls_certs_dir + "/ca.crt");

    if (!s_certs_available_) return;

    s_port_   = get_test_port();
    s_server_ = std::make_unique<ServerProcess>();

    std::string root_certs  = read_file_contents(g_tls_certs_dir + "/ca.crt");
    std::string client_cert = read_file_contents(g_tls_certs_dir + "/client.crt");
    std::string client_key  = read_file_contents(g_tls_certs_dir + "/client.key");
    s_server_->set_tls_config(root_certs, client_cert, client_key);

    std::vector<std::string> args = {"--tls",
                                     "--tls-cert",
                                     g_tls_certs_dir + "/server.crt",
                                     "--tls-key",
                                     g_tls_certs_dir + "/server.key",
                                     "--tls-root",
                                     g_tls_certs_dir + "/ca.crt",
                                     "--require-client-cert",
                                     "--enable-transfer-hash"};

    if (!s_server_->start(s_port_, args)) {
      s_server_.reset();
      s_certs_available_ = false;
    }
  }

  static void TearDownTestSuite()
  {
    if (s_server_) s_server_->stop();
    s_server_.reset();
  }

  void SetUp() override
  {
    if (!s_certs_available_) { GTEST_SKIP() << "mTLS certificates not available"; }
    ASSERT_NE(s_server_, nullptr) << "mTLS server not running";
    port_ = s_port_;
  }

  std::unique_ptr<grpc_client_t> create_mtls_client(bool with_client_cert = true)
  {
    grpc_client_config_t config;
    config.server_address  = "localhost:" + std::to_string(port_);
    config.timeout_seconds = 30;
    config.enable_tls      = true;
    config.tls_root_certs  = read_file_contents(g_tls_certs_dir + "/ca.crt");

    if (with_client_cert) {
      config.tls_client_cert = read_file_contents(g_tls_certs_dir + "/client.crt");
      config.tls_client_key  = read_file_contents(g_tls_certs_dir + "/client.key");
    }

    auto client = std::make_unique<grpc_client_t>(config);
    if (!client->connect()) return nullptr;
    return client;
  }

  static std::unique_ptr<ServerProcess> s_server_;
  static int s_port_;
  static bool s_certs_available_;
};

std::unique_ptr<ServerProcess> MtlsServerTests::s_server_;
int MtlsServerTests::s_port_             = 0;
bool MtlsServerTests::s_certs_available_ = false;

TEST_F(MtlsServerTests, ConnectionWithClientCert)
{
  auto client = create_mtls_client(true);
  ASSERT_NE(client, nullptr) << "Failed to connect with mTLS";
  EXPECT_TRUE(client->is_connected());
}

TEST_F(MtlsServerTests, RejectsClientWithoutCert)
{
  auto client = create_mtls_client(false);
  EXPECT_EQ(client, nullptr) << "Server should reject client without certificate";
}

// =============================================================================
// Chunk Validation Tests
//
// Uses a raw gRPC stub to send malformed chunk requests and verify the server
// rejects them with appropriate error codes. Exercises items 1-8 from the
// chunked transfer hardening work.
// =============================================================================

class ChunkValidationTests : public GrpcIntegrationTestBase {
 protected:
  static void SetUpTestSuite()
  {
    s_port_   = get_test_port();
    s_server_ = std::make_unique<ServerProcess>();
    ASSERT_TRUE(s_server_->start(s_port_, {"--verbose"}))
      << "Failed to start chunk validation server";
  }

  static void TearDownTestSuite()
  {
    if (s_server_) s_server_->stop();
    s_server_.reset();
  }

  void SetUp() override
  {
    ASSERT_NE(s_server_, nullptr);
    port_ = s_port_;

    auto channel =
      grpc::CreateChannel("localhost:" + std::to_string(port_), grpc::InsecureChannelCredentials());
    stub_ = cuopt::remote::CuOptRemoteService::NewStub(channel);
  }

  std::string start_upload()
  {
    grpc::ClientContext ctx;
    cuopt::remote::StartChunkedUploadRequest req;
    auto* hdr = req.mutable_problem_header()->mutable_header();
    hdr->set_version(1);
    hdr->set_problem_category(cuopt::remote::LP);
    cuopt::remote::StartChunkedUploadResponse resp;
    auto status = stub_->StartChunkedUpload(&ctx, req, &resp);
    EXPECT_TRUE(status.ok()) << status.error_message();
    return resp.upload_id();
  }

  grpc::Status send_chunk(const std::string& upload_id,
                          cuopt::remote::ArrayFieldId field_id,
                          int64_t element_offset,
                          int64_t total_elements,
                          const std::string& data)
  {
    grpc::ClientContext ctx;
    cuopt::remote::SendArrayChunkRequest req;
    req.set_upload_id(upload_id);
    auto* ac = req.mutable_chunk();
    ac->set_field_id(field_id);
    ac->set_element_offset(element_offset);
    ac->set_total_elements(total_elements);
    ac->set_data(data);
    cuopt::remote::SendArrayChunkResponse resp;
    return stub_->SendArrayChunk(&ctx, req, &resp);
  }

  std::unique_ptr<cuopt::remote::CuOptRemoteService::Stub> stub_;
  static std::unique_ptr<ServerProcess> s_server_;
  static int s_port_;
};

std::unique_ptr<ServerProcess> ChunkValidationTests::s_server_;
int ChunkValidationTests::s_port_ = 0;

TEST_F(ChunkValidationTests, RejectsNegativeElementOffset)
{
  auto uid = start_upload();
  std::string data(8, '\0');  // 1 double
  auto status = send_chunk(uid, cuopt::remote::FIELD_C, -1, 10, data);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("non-negative"));
}

TEST_F(ChunkValidationTests, RejectsNegativeTotalElements)
{
  auto uid = start_upload();
  std::string data(8, '\0');
  auto status = send_chunk(uid, cuopt::remote::FIELD_C, 0, -5, data);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("non-negative"));
}

TEST_F(ChunkValidationTests, RejectsHugeTotalElements)
{
  auto uid = start_upload();
  std::string data(8, '\0');
  auto status = send_chunk(uid, cuopt::remote::FIELD_C, 0, int64_t(1) << 60, data);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("too large"));
}

TEST_F(ChunkValidationTests, RejectsInvalidFieldId)
{
  auto uid = start_upload();
  std::string data(8, '\0');
  auto status = send_chunk(uid, static_cast<cuopt::remote::ArrayFieldId>(999), 0, 10, data);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("field_id"));
}

TEST_F(ChunkValidationTests, RejectsUnalignedChunkData)
{
  auto uid = start_upload();
  // First chunk to allocate the array (doubles, elem_size=8)
  std::string good_data(80, '\0');  // 10 doubles
  auto s1 = send_chunk(uid, cuopt::remote::FIELD_C, 0, 10, good_data);
  EXPECT_TRUE(s1.ok()) << s1.error_message();

  // Send a misaligned chunk (7 bytes, not a multiple of 8)
  std::string bad_data(7, '\0');
  auto s2 = send_chunk(uid, cuopt::remote::FIELD_C, 0, 10, bad_data);
  EXPECT_FALSE(s2.ok());
  EXPECT_EQ(s2.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(s2.error_message(), ::testing::HasSubstr("aligned"));
}

TEST_F(ChunkValidationTests, RejectsOffsetBeyondArraySize)
{
  auto uid = start_upload();
  // Allocate array of 10 doubles
  std::string data(80, '\0');
  auto s1 = send_chunk(uid, cuopt::remote::FIELD_C, 0, 10, data);
  EXPECT_TRUE(s1.ok()) << s1.error_message();

  // Offset 100 is way past the 10-element array
  std::string small_data(8, '\0');
  auto s2 = send_chunk(uid, cuopt::remote::FIELD_C, 100, 10, small_data);
  EXPECT_FALSE(s2.ok());
  EXPECT_EQ(s2.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(ChunkValidationTests, RejectsChunkOverflow)
{
  auto uid = start_upload();
  // Allocate array of 4 doubles (32 bytes)
  std::string init_data(32, '\0');
  auto s1 = send_chunk(uid, cuopt::remote::FIELD_C, 0, 4, init_data);
  EXPECT_TRUE(s1.ok()) << s1.error_message();

  // Offset 3 + 2 doubles = writes past end
  std::string over_data(16, '\0');  // 2 doubles
  auto s2 = send_chunk(uid, cuopt::remote::FIELD_C, 3, 4, over_data);
  EXPECT_FALSE(s2.ok());
  EXPECT_EQ(s2.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(ChunkValidationTests, RejectsUnknownUploadId)
{
  std::string data(8, '\0');
  auto status = send_chunk("nonexistent-upload-id", cuopt::remote::FIELD_C, 0, 10, data);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::NOT_FOUND);
}

TEST_F(ChunkValidationTests, AcceptsValidChunk)
{
  auto uid = start_upload();
  // 10 doubles = 80 bytes
  std::string data(80, '\x42');
  auto status = send_chunk(uid, cuopt::remote::FIELD_C, 0, 10, data);
  EXPECT_TRUE(status.ok()) << status.error_message();
}

}  // anonymous namespace

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
