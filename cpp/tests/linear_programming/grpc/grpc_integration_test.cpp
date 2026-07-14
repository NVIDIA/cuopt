/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file grpc_integration_test.cpp
 * @brief Integration tests for gRPC server end-to-end behaviour.
 *
 * All tests use the public blocking API: solve_lp_remote() / solve_mip_remote().
 * Server address is passed via CUOPT_REMOTE_HOST / CUOPT_REMOTE_PORT env vars
 * that each fixture configures in SetUp() and clears in TearDown().
 *
 * Fixture layout:
 *   NoServerTests          - Tests that don't need a server
 *   DefaultServerTests     - Shared server with default config
 *   ChunkedUploadTests     - Shared server with small CUOPT_MAX_MESSAGE_BYTES
 *   ErrorRecoveryTests     - Per-test server lifecycle
 *   TlsServerTests         - Shared TLS server
 *   MtlsServerTests        - Shared mTLS server
 *   ChunkValidationTests   - Raw gRPC stub; malformed chunk rejection
 *
 * Environment variables:
 *   CUOPT_GRPC_SERVER_PATH  - Path to cuopt_grpc_server binary
 *   CUOPT_TEST_PORT_BASE    - Base port for test servers (default: 19000)
 *   RAPIDS_DATASET_ROOT_DIR - Path to test datasets
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cerrno>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

#include <cuopt/mathematical_optimization/cpu_optimization_problem.hpp>
#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <cuopt/mathematical_optimization/optimization_problem.hpp>
#include <cuopt/mathematical_optimization/optimization_problem_interface.hpp>
#include <cuopt/mathematical_optimization/optimization_problem_utils.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <cuopt/mathematical_optimization/solve_remote.hpp>
#include <utilities/inline_lp_test_utils.hpp>

#include <cuopt_remote_service.grpc.pb.h>
#include <grpcpp/grpcpp.h>

#include <fcntl.h>
#include <signal.h>
#include <sys/prctl.h>
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

using namespace cuopt::mathematical_optimization;

namespace {

// GRPC_INTEGRATION_TEST
//   `-- cuopt_grpc_server parent (pid_, leads process group pgid_ == pid_)
//         `-- GPU worker (inherits pgid_)
//
// The server is forked into its own process group so stop() can signal the whole
// group. Killing the server parent orphans its GPU worker, so this test process
// also marks itself a subreaper (PR_SET_CHILD_SUBREAPER): the orphan reparents
// here and stop() reaps the entire group, even inside a container whose PID 1
// does not reap.

class ServerProcess {
 public:
  ServerProcess() : pid_(-1), pgid_(-1), port_(0) {}
  ServerProcess(const ServerProcess&)            = delete;
  ServerProcess& operator=(const ServerProcess&) = delete;
  ~ServerProcess()
  {
    if (!stop()) { std::cerr << "Failed to clean up test server process group\n"; }
  }

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
    if (pid_ > 0) {
      std::cerr << "Cannot reuse a ServerProcess while it still owns a process lifecycle\n";
      return false;
    }

    std::string server_path = find_server_binary();
    if (server_path.empty()) {
      std::cerr << "Could not find cuopt_grpc_server binary\n";
      return false;
    }

    prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0);

    port_ = port;
    pid_  = fork();
    if (pid_ < 0) {
      std::cerr << "fork() failed\n";
      return false;
    }

    if (pid_ == 0) {
      setpgid(0, 0);

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

    setpgid(pid_, pid_);
    pgid_ = pid_;

    if (!wait_for_ready(15000)) {
      if (!stop()) { std::cerr << "Failed to clean up server after readiness failure\n"; }
      return false;
    }

    return true;
  }

  bool stop()
  {
    if (pid_ <= 0) return true;

    kill(-pgid_, SIGTERM);
    if (!reap_group(std::chrono::seconds(15))) {
      kill(-pgid_, SIGKILL);
      if (!reap_group(std::chrono::seconds(15))) {
        std::cerr << "Server process group " << pgid_ << " did not exit\n";
        return false;
      }
    }

    clear_lifecycle_state();
    return true;
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
  void clear_lifecycle_state()
  {
    pid_  = -1;
    pgid_ = -1;
    port_ = 0;
  }

  bool reap_group(std::chrono::milliseconds timeout)
  {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (true) {
      int status = 0;
      pid_t ret  = waitpid(-pgid_, &status, WNOHANG);
      if (ret > 0) continue;
      if (ret < 0 && errno == EINTR) continue;
      if (ret < 0 && errno == ECHILD) return true;
      if (std::chrono::steady_clock::now() >= deadline) return false;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

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

      std::shared_ptr<grpc::ChannelCredentials> creds;
      if (!tls_root_certs_.empty()) {
        grpc::SslCredentialsOptions ssl_opts;
        ssl_opts.pem_root_certs  = tls_root_certs_;
        ssl_opts.pem_cert_chain  = tls_client_cert_;
        ssl_opts.pem_private_key = tls_client_key_;
        creds                    = grpc::SslCredentials(ssl_opts);
      } else {
        creds = grpc::InsecureChannelCredentials();
      }

      auto channel =
        grpc::CreateChannel("localhost:" + std::to_string(port_), std::move(creds));
      auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(200);
      if (channel->WaitForConnected(deadline)) { return true; }

      int status = 0;
      if (waitpid(pid_, &status, WNOHANG) == pid_) {
        std::cerr << "Server process died during startup\n";
        return false;
      }
    }
  }

  pid_t pid_;
  pid_t pgid_;
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

// =============================================================================
// Base Test Class
// =============================================================================

class GrpcIntegrationTestBase : public ::testing::Test {
 protected:
  void set_remote_host(int port)
  {
    setenv("CUOPT_REMOTE_HOST", "localhost", 1);
    setenv("CUOPT_REMOTE_PORT", std::to_string(port).c_str(), 1);
  }

  void clear_remote_env()
  {
    unsetenv("CUOPT_REMOTE_HOST");
    unsetenv("CUOPT_REMOTE_PORT");
    unsetenv("CUOPT_TLS_ENABLED");
    unsetenv("CUOPT_TLS_ROOT_CERT");
    unsetenv("CUOPT_TLS_CLIENT_CERT");
    unsetenv("CUOPT_TLS_CLIENT_KEY");
    unsetenv("CUOPT_MAX_MESSAGE_BYTES");
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

  cpu_optimization_problem_t<int32_t, double> load_problem_from_file(const std::string& path)
  {
    auto mps_data = cuopt::mathematical_optimization::io::read<int32_t, double>(path);
    cpu_optimization_problem_t<int32_t, double> problem;
    populate_from_mps_data_model(&problem, mps_data);
    return problem;
  }

  cpu_optimization_problem_t<int32_t, double> create_simple_mip()
  {
    auto data = cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: x0 + 2 x1
Subject To
  c1: x0 + x1 >= 1
Binaries
  x0
  x1
End
)LP");
    cpu_optimization_problem_t<int32_t, double> problem;
    populate_from_mps_data_model(&problem, data);
    return problem;
  }

  int port_ = 0;
};

// =============================================================================
// No-Server Tests
// =============================================================================

class NoServerTests : public GrpcIntegrationTestBase {
 protected:
  void SetUp() override
  {
    port_ = get_test_port();
    set_remote_host(port_);
  }
  void TearDown() override { clear_remote_env(); }
};

TEST_F(NoServerTests, ConnectToNonexistentServer)
{
  auto problem = create_simple_mip();
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 5.0;

  EXPECT_THROW(solve_lp_remote(problem, settings), std::runtime_error);
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
    ASSERT_TRUE(s_server_->start(s_port_, {}))
      << "Failed to start shared default server on port " << s_port_;
  }

  static void TearDownTestSuite()
  {
    if (s_server_) EXPECT_TRUE(s_server_->stop());
    s_server_.reset();
  }

  void SetUp() override
  {
    ASSERT_NE(s_server_, nullptr) << "Shared server not running";
    port_ = s_port_;
    set_remote_host(port_);
  }

  void TearDown() override { clear_remote_env(); }

  static std::unique_ptr<ServerProcess> s_server_;
  static int s_port_;
};

std::unique_ptr<ServerProcess> DefaultServerTests::s_server_;
int DefaultServerTests::s_port_ = 0;

TEST_F(DefaultServerTests, ServerAcceptsConnections)
{
  ASSERT_TRUE(s_server_->is_running());

  // A successful solve proves the server accepts and processes connections.
  auto problem = create_simple_mip();
  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto solution = solve_mip_remote(problem, settings);
  ASSERT_NE(solution, nullptr);
  EXPECT_EQ(solution->get_termination_status(), mip_termination_status_t::Optimal);
}

TEST_F(DefaultServerTests, SolveLPBlocking)
{
  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_file(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto solution = solve_lp_remote(problem, settings);
  ASSERT_NE(solution, nullptr);
  EXPECT_NEAR(solution->get_objective_value(), -464.753, 1.0);
}

TEST_F(DefaultServerTests, SolveInfeasibleLP)
{
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

  auto solution = solve_lp_remote(problem, settings);
  ASSERT_NE(solution, nullptr);
  EXPECT_NE(solution->get_termination_status(), pdlp_termination_status_t::Optimal)
    << "Expected non-optimal termination for infeasible problem";
}

TEST_F(DefaultServerTests, SolveMIPBlocking)
{
  auto problem = create_simple_mip();
  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto solution = solve_mip_remote(problem, settings);
  ASSERT_NE(solution, nullptr);
  EXPECT_EQ(solution->get_termination_status(), mip_termination_status_t::Optimal);
  EXPECT_NEAR(solution->get_objective_value(), 1.0, 0.01);
}

TEST_F(DefaultServerTests, MultipleSequentialSolves)
{
  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_file(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  for (int i = 0; i < 3; ++i) {
    auto solution = solve_lp_remote(problem, settings);
    ASSERT_NE(solution, nullptr) << "Solve #" << i << " returned null";
    EXPECT_NEAR(solution->get_objective_value(), -464.753, 1.0) << "Solve #" << i;
  }
}

TEST_F(DefaultServerTests, ConcurrentSolves)
{
  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_file(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  std::atomic<int> success_count{0};
  auto solve_task = [&]() {
    try {
      auto solution = solve_lp_remote(problem, settings);
      if (solution && std::abs(solution->get_objective_value() - (-464.753)) < 1.0) {
        success_count++;
      }
    } catch (const std::exception& e) {
      std::cerr << "Concurrent solve exception: " << e.what() << "\n";
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < 3; ++i) {
    threads.emplace_back(solve_task);
  }
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(success_count.load(), 3);
}

TEST_F(DefaultServerTests, SolveLPReturnsWarmStartData)
{
  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_file(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto solution = solve_lp_remote(problem, settings);
  ASSERT_NE(solution, nullptr);

  EXPECT_TRUE(solution->has_warm_start_data())
    << "LP solution should contain PDLP warm start data";

  const auto& ws = solution->get_cpu_pdlp_warm_start_data();

  EXPECT_FALSE(ws.current_primal_solution_.empty());
  EXPECT_FALSE(ws.current_dual_solution_.empty());
  EXPECT_FALSE(ws.initial_primal_average_.empty());
  EXPECT_FALSE(ws.initial_dual_average_.empty());
  EXPECT_FALSE(ws.current_ATY_.empty());
  EXPECT_FALSE(ws.sum_primal_solutions_.empty());
  EXPECT_FALSE(ws.sum_dual_solutions_.empty());
  EXPECT_FALSE(ws.last_restart_duality_gap_primal_solution_.empty());
  EXPECT_FALSE(ws.last_restart_duality_gap_dual_solution_.empty());

  EXPECT_GT(ws.initial_primal_weight_, 0.0);
  EXPECT_GT(ws.initial_step_size_, 0.0);
  EXPECT_GE(ws.total_pdlp_iterations_, 0);
  EXPECT_GE(ws.total_pdhg_iterations_, 0);
}

// =============================================================================
// Chunked Upload Tests
//
// Uses CUOPT_MAX_MESSAGE_BYTES=4096 so any problem larger than ~3 KiB is
// transported via the chunked array protocol. Verifies that the chunked
// path produces correct results and that QCQP problems round-trip cleanly.
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
    if (s_server_) EXPECT_TRUE(s_server_->stop());
    s_server_.reset();
  }

  void SetUp() override
  {
    ASSERT_NE(s_server_, nullptr);
    port_ = s_port_;
    set_remote_host(port_);
    // Small message limit forces chunked upload for problems larger than ~3 KiB.
    setenv("CUOPT_MAX_MESSAGE_BYTES", "4096", 1);
  }

  void TearDown() override { clear_remote_env(); }

  static std::unique_ptr<ServerProcess> s_server_;
  static int s_port_;
};

std::unique_ptr<ServerProcess> ChunkedUploadTests::s_server_;
int ChunkedUploadTests::s_port_ = 0;

TEST_F(ChunkedUploadTests, ChunkedUploadLP)
{
  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_file(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto solution = solve_lp_remote(problem, settings);
  ASSERT_NE(solution, nullptr);
  EXPECT_NEAR(solution->get_objective_value(), -464.753, 1.0);
}

TEST_F(ChunkedUploadTests, ChunkedUploadMIP)
{
  std::string mps_path = get_test_mip_path("sudoku.mps");
  auto problem         = load_problem_from_file(mps_path);
  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto solution = solve_mip_remote(problem, settings);
  ASSERT_NE(solution, nullptr);
}

TEST_F(ChunkedUploadTests, ConcurrentSolves)
{
  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_file(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  std::atomic<int> success_count{0};
  auto solve_task = [&]() {
    try {
      auto solution = solve_lp_remote(problem, settings);
      if (solution && std::abs(solution->get_objective_value() - (-464.753)) < 1.0) {
        success_count++;
      }
    } catch (const std::exception& e) {
      std::cerr << "ConcurrentSolves exception: " << e.what() << "\n";
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < 3; ++i) {
    threads.emplace_back(solve_task);
  }
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(success_count.load(), 3);
}

// Verify that the QCQP wire format survives the chunked transport path and
// that the general convex quadratic solver handles nonzero-RHS QC rows.
TEST_F(ChunkedUploadTests, QuadraticConstraintsNonZeroRhs)
{
  std::string mps_path = get_test_data_path("qcqp", "QC_Test_1.mps");
  auto problem         = load_problem_from_file(mps_path);
  ASSERT_TRUE(problem.has_quadratic_constraints());
  EXPECT_EQ(problem.get_quadratic_constraints().size(), 2u);

  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  // QC_Test_1 has rhs != 0; handled by the general convex quadratic path.
  auto solution = solve_lp_remote(problem, settings);
  EXPECT_TRUE(solution != nullptr);
}

// End-to-end SOCP correctness via gRPC.
// QC_Test_2 is SOC-compatible (rhs = 0 on every QC).
// Closed-form optimum: x = y = 1/sqrt(2), z = 1, obj = -(1 + sqrt(2)).
TEST_F(ChunkedUploadTests, QuadraticConstraintsEndToEndSocp)
{
  std::string lp_path = get_test_data_path("qcqp", "QC_Test_2.lp");
  auto problem        = load_problem_from_file(lp_path);
  ASSERT_TRUE(problem.has_quadratic_constraints());
  EXPECT_EQ(problem.get_quadratic_constraints().size(), 2u);

  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto solution = solve_lp_remote(problem, settings);
  ASSERT_NE(solution, nullptr);

  EXPECT_EQ(solution->get_termination_status(), pdlp_termination_status_t::Optimal);

  constexpr double kTol   = 1e-3;
  const double sqrt2      = std::sqrt(2.0);
  const double opt_obj    = -(1.0 + sqrt2);
  const double opt_x_y    = 1.0 / sqrt2;
  const double opt_z      = 1.0;
  EXPECT_NEAR(solution->get_objective_value(), opt_obj, kTol);

  const auto primal = solution->get_primal_solution_host();
  ASSERT_GE(primal.size(), 3u);
  EXPECT_NEAR(primal[0], opt_x_y, kTol);
  EXPECT_NEAR(primal[1], opt_x_y, kTol);
  EXPECT_NEAR(primal[2], opt_z, kTol);
}

// =============================================================================
// Error Recovery Tests (per-test server lifecycle)
// =============================================================================

class ErrorRecoveryTests : public GrpcIntegrationTestBase {
 protected:
  void SetUp() override { port_ = get_test_port(); }
  void TearDown() override
  {
    clear_remote_env();
    EXPECT_TRUE(server_.stop());
  }

  bool start_server(const std::vector<std::string>& extra_args = {})
  {
    return server_.start(port_, extra_args);
  }

  ServerProcess server_;
};

TEST_F(ErrorRecoveryTests, SolveMIPAfterServerRestart)
{
  ASSERT_TRUE(start_server({"--max-message-mb", "256"}));
  set_remote_host(port_);

  std::string mps_path = get_test_mip_path("sudoku.mps");
  auto problem         = load_problem_from_file(mps_path);
  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto solution1 = solve_mip_remote(problem, settings);
  EXPECT_NE(solution1, nullptr) << "First solve failed";

  ASSERT_TRUE(server_.stop());
  ASSERT_TRUE(start_server({"--max-message-mb", "256"}));

  auto solution2 = solve_mip_remote(problem, settings);
  EXPECT_NE(solution2, nullptr) << "Second solve after restart failed";
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
                                     g_tls_certs_dir + "/ca.crt"};

    if (!s_server_->start(s_port_, args)) {
      s_server_.reset();
      s_certs_available_ = false;
    }
  }

  static void TearDownTestSuite()
  {
    if (s_server_) EXPECT_TRUE(s_server_->stop());
    s_server_.reset();
  }

  void SetUp() override
  {
    if (!s_certs_available_) { GTEST_SKIP() << "TLS certificates not available"; }
    ASSERT_NE(s_server_, nullptr) << "TLS server not running";
    port_ = s_port_;
    set_remote_host(port_);
    setenv("CUOPT_TLS_ENABLED", "1", 1);
    setenv("CUOPT_TLS_ROOT_CERT", (g_tls_certs_dir + "/ca.crt").c_str(), 1);
  }

  void TearDown() override { clear_remote_env(); }

  static std::string read_file_contents(const std::string& path)
  {
    std::ifstream file(path);
    if (!file) return "";
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
  }

  static std::unique_ptr<ServerProcess> s_server_;
  static int s_port_;
  static bool s_certs_available_;
};

std::unique_ptr<ServerProcess> TlsServerTests::s_server_;
int TlsServerTests::s_port_             = 0;
bool TlsServerTests::s_certs_available_ = false;

TEST_F(TlsServerTests, SolveLP)
{
  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_file(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto solution = solve_lp_remote(problem, settings);
  ASSERT_NE(solution, nullptr);
  EXPECT_NEAR(solution->get_objective_value(), -464.753, 1.0);
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
                                     "--require-client-cert"};

    if (!s_server_->start(s_port_, args)) {
      s_server_.reset();
      s_certs_available_ = false;
    }
  }

  static void TearDownTestSuite()
  {
    if (s_server_) EXPECT_TRUE(s_server_->stop());
    s_server_.reset();
  }

  void SetUp() override
  {
    if (!s_certs_available_) { GTEST_SKIP() << "mTLS certificates not available"; }
    ASSERT_NE(s_server_, nullptr) << "mTLS server not running";
    port_ = s_port_;
    set_remote_host(port_);
    setenv("CUOPT_TLS_ENABLED", "1", 1);
    setenv("CUOPT_TLS_ROOT_CERT", (g_tls_certs_dir + "/ca.crt").c_str(), 1);
    setenv("CUOPT_TLS_CLIENT_CERT", (g_tls_certs_dir + "/client.crt").c_str(), 1);
    setenv("CUOPT_TLS_CLIENT_KEY", (g_tls_certs_dir + "/client.key").c_str(), 1);
  }

  void TearDown() override { clear_remote_env(); }

  static std::string read_file_contents(const std::string& path)
  {
    std::ifstream file(path);
    if (!file) return "";
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
  }

  static std::unique_ptr<ServerProcess> s_server_;
  static int s_port_;
  static bool s_certs_available_;
};

std::unique_ptr<ServerProcess> MtlsServerTests::s_server_;
int MtlsServerTests::s_port_             = 0;
bool MtlsServerTests::s_certs_available_ = false;

TEST_F(MtlsServerTests, SolveLP)
{
  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_file(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto solution = solve_lp_remote(problem, settings);
  ASSERT_NE(solution, nullptr);
  EXPECT_NEAR(solution->get_objective_value(), -464.753, 1.0);
}

TEST_F(MtlsServerTests, RejectsClientWithoutCert)
{
  // Unset client cert/key — server requires them.
  unsetenv("CUOPT_TLS_CLIENT_CERT");
  unsetenv("CUOPT_TLS_CLIENT_KEY");

  auto problem = create_simple_mip();
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 5.0;

  EXPECT_THROW(solve_lp_remote(problem, settings), std::runtime_error);
}

// =============================================================================
// Chunk Validation Tests
//
// Uses a raw gRPC stub to send malformed chunk requests and verify the server
// rejects them with appropriate error codes.
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
    if (s_server_) EXPECT_TRUE(s_server_->stop());
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
  std::string data(8, '\0');
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
  std::string good_data(80, '\0');
  auto s1 = send_chunk(uid, cuopt::remote::FIELD_C, 0, 10, good_data);
  EXPECT_TRUE(s1.ok()) << s1.error_message();

  std::string bad_data(7, '\0');
  auto s2 = send_chunk(uid, cuopt::remote::FIELD_C, 0, 10, bad_data);
  EXPECT_FALSE(s2.ok());
  EXPECT_EQ(s2.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(s2.error_message(), ::testing::HasSubstr("aligned"));
}

TEST_F(ChunkValidationTests, RejectsOffsetBeyondArraySize)
{
  auto uid = start_upload();
  std::string data(80, '\0');
  auto s1 = send_chunk(uid, cuopt::remote::FIELD_C, 0, 10, data);
  EXPECT_TRUE(s1.ok()) << s1.error_message();

  std::string small_data(8, '\0');
  auto s2 = send_chunk(uid, cuopt::remote::FIELD_C, 100, 10, small_data);
  EXPECT_FALSE(s2.ok());
  EXPECT_EQ(s2.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(ChunkValidationTests, RejectsChunkOverflow)
{
  auto uid = start_upload();
  std::string init_data(32, '\0');
  auto s1 = send_chunk(uid, cuopt::remote::FIELD_C, 0, 4, init_data);
  EXPECT_TRUE(s1.ok()) << s1.error_message();

  std::string over_data(16, '\0');
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

TEST_F(ChunkValidationTests, RejectsContainerFieldNumWithoutContainerIndex)
{
  auto uid = start_upload();
  grpc::ClientContext ctx;
  cuopt::remote::SendArrayChunkRequest req;
  req.set_upload_id(uid);
  auto* ac = req.mutable_chunk();
  ac->set_field_id(0);
  ac->set_element_offset(0);
  ac->set_total_elements(1);
  ac->set_data(std::string(8, '\0'));
  ac->set_container_field_num(25);
  cuopt::remote::SendArrayChunkResponse resp;
  auto status = stub_->SendArrayChunk(&ctx, req, &resp);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("container_field_num"));
}

TEST_F(ChunkValidationTests, RejectsContainerIndexWithoutContainerFieldNum)
{
  auto uid = start_upload();
  grpc::ClientContext ctx;
  cuopt::remote::SendArrayChunkRequest req;
  req.set_upload_id(uid);
  auto* ac = req.mutable_chunk();
  ac->set_field_id(0);
  ac->set_element_offset(0);
  ac->set_total_elements(1);
  ac->set_data(std::string(8, '\0'));
  ac->set_container_index(0);
  cuopt::remote::SendArrayChunkResponse resp;
  auto status = stub_->SendArrayChunk(&ctx, req, &resp);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("container_index"));
}

TEST_F(ChunkValidationTests, AcceptsValidChunk)
{
  auto uid = start_upload();
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
