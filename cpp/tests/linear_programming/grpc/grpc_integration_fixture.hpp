/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file grpc_integration_fixture.hpp
 * @brief Server-process and base fixture shared by the gRPC integration test binaries
 *
 * Used by GRPC_MATHOPT_INTEGRATION_TEST (LP/MIP) and GRPC_ROUTING_INTEGRATION_TEST (VRP).
 */

#pragma once

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
#include <cuopt/mathematical_optimization/solver_settings.hpp>
#include <utilities/inline_lp_test_utils.hpp>
#include "grpc_client.hpp"

#include "grpc_test_log_capture.hpp"

#include <cuopt_remote_service.grpc.pb.h>
#include <grpcpp/generic/generic_stub.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/support/byte_buffer.h>

#include "grpc_service_mapper.hpp"

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
using cuopt::mathematical_optimization::testing::GrpcTestLogCapture;

namespace grpc_integration_fixture {

// GRPC_MATHOPT_INTEGRATION_TEST
//   `-- cuopt_grpc_server parent (pid_, leads process group pgid_ == pid_)
//         `-- GPU worker (inherits pgid_)
//
// The server is forked into its own process group so stop() can signal the whole
// group. Killing the server parent orphans its GPU worker, so this test process
// also marks itself a subreaper (PR_SET_CHILD_SUBREAPER): the orphan reparents
// here and stop() reaps the entire group, even inside a container whose PID 1
// does not reap. Polling kill(-pgid_, 0) instead is not enough -- it counts an
// unreaped zombie as a live member and can never observe a grandchild's death.

class ServerProcess {
 public:
  ServerProcess() : pid_(-1), pgid_(-1), port_(0), exit_code_(-1) {}
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

  bool start(int port,
             const std::vector<std::string>& extra_args                                = {},
             const std::vector<std::pair<std::string, std::string>>& extra_environment = {})
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

    // Reparent any process orphaned by killing the server (i.e. the GPU worker)
    // onto this test process so stop() can reap it without relying on init.
    prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0);

    port_      = port;
    exit_code_ = -1;
    pid_       = fork();
    if (pid_ < 0) {
      std::cerr << "fork() failed\n";
      return false;
    }

    if (pid_ == 0) {
      setpgid(0, 0);  // child leads its own group; parent mirrors this below

      for (const auto& [name, value] : extra_environment) {
        setenv(name.c_str(), value.c_str(), 1);
      }

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

    // Mirror the child's setpgid() so the group is established regardless of which
    // process runs first; the child is a fresh fork, so its group id is its pid.
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

    // Ask the whole group (server + GPU worker) to exit gracefully, then reap
    // every member. Killing the server parent can orphan a mid-solve worker;
    // because we are a subreaper it reparents here and reap_group() collects it.
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

  pid_t pid() const { return pid_; }

  int exit_code() const { return exit_code_; }

  bool is_running() const
  {
    if (pid_ <= 0) return false;
    return kill(pid_, 0) == 0;
  }

  // Wait until the server parent has exited and been reaped. Unlike is_running(),
  // this treats a zombie as exited (kill(pid,0) is true for zombies).
  // Returns true once the parent has exited. If leftover workers are still
  // alive (e.g. GPU D-state), lifecycle state is preserved so TearDown's
  // stop() can finish draining the process group.
  bool wait_exited(std::chrono::milliseconds timeout)
  {
    if (pid_ <= 0) return true;

    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (true) {
      int status = 0;
      pid_t ret  = waitpid(pid_, &status, WNOHANG);
      if (ret == pid_ || (ret < 0 && errno == ECHILD)) {
        if (ret == pid_) { exit_code_ = WIFEXITED(status) ? WEXITSTATUS(status) : -1; }
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
          deadline - std::chrono::steady_clock::now());
        if (remaining > std::chrono::milliseconds(2000)) {
          remaining = std::chrono::milliseconds(2000);
        }
        if (remaining.count() < 0) { remaining = std::chrono::milliseconds(0); }
        if (reap_group(remaining)) { clear_lifecycle_state(); }
        // Parent is gone either way; leave pid_/pgid_ intact if the group is
        // not fully drained so TearDown can still signal it.
        return true;
      }
      if (std::chrono::steady_clock::now() >= deadline) { return false; }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
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

  // Reap every member of the server's process group, returning true once the
  // group is fully drained. start() makes this process a subreaper, so a worker
  // orphaned when the server parent dies reparents here and is reaped too --
  // first the parent, then (once its death reparents the worker) the worker.
  bool reap_group(std::chrono::milliseconds timeout)
  {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (true) {
      int status = 0;
      pid_t ret  = waitpid(-pgid_, &status, WNOHANG);
      if (ret > 0) continue;  // reaped a member; keep draining
      if (ret < 0 && errno == EINTR) continue;
      if (ret < 0 && errno == ECHILD) return true;  // no members left
      // ret == 0: members remain but none have exited yet.
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

      int status = 0;
      if (waitpid(pid_, &status, WNOHANG) == pid_) {
        exit_code_ = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        std::cerr << "Server process died during startup\n";
        return false;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  }

  pid_t pid_;
  pid_t pgid_;
  int port_;
  int exit_code_;
  std::string tls_root_certs_;
  std::string tls_client_cert_;
  std::string tls_client_key_;
};

inline int get_test_port()
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

inline std::string g_tls_certs_dir;
inline bool g_tls_certs_ready = false;

inline bool ensure_test_certs()
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

inline std::string read_file_contents(const std::string& path)
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

  // Load a problem from disk, dispatching by file extension to read_mps()
  // for .mps/.qps (with optional .gz/.bz2) and read_lp() for .lp (with
  // optional .gz/.bz2).  See io::read() in parser.hpp.
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

  // Result of warm_up_worker. Callers ASSERT on ok so a failure stops the test.
  // detail is set only on failure and includes the last job status plus a tail
  // of the server log.
  struct worker_warmup_result_t {
    bool ok = false;
    std::string detail;
  };

  // Prove the shared worker is idle and warm before a timed test begins,
  // instead of trusting whatever the previous test left behind. DefaultServerTests
  // shares one worker across the whole suite, and a prior test's SIGKILL can leave it
  // mid-respawn (paying for a fresh CUDA context and RMM pool); that debt is absorbed
  // here so it cannot leak into the caller's timing assertions -- see #1814.
  //
  // The job stays QUEUED until the replacement worker finishes init, which can
  // exceed 60s on a busy runner. Allow 3 minutes for that. Once the worker claims
  // the job, allow 30s more for this 5s MIP. delete_job of a PROCESSING job
  // SIGKILLs the worker, so a probe that is still solving is left in place.
  // A QUEUED probe was never claimed and can be deleted.
  worker_warmup_result_t warm_up_worker(grpc_client_t* client, const std::string& server_log_path)
  {
    worker_warmup_result_t result;

    auto fail = [&](const std::string& why, const job_status_result_t& status) {
      std::ostringstream out;
      out << "Worker warm-up failed: " << why << ", status=" << job_status_to_string(status.status);
      if (!status.message.empty()) { out << ", message=" << status.message; }
      if (!status.error_message.empty()) { out << ", error=" << status.error_message; }
      out << tail_server_log(server_log_path);
      result.ok     = false;
      result.detail = out.str();
      return result;
    };

    solver_settings_t<int32_t, double> warmup_settings;
    warmup_settings.set_parameter(CUOPT_TIME_LIMIT, 5.0);
    auto warmup = client->submit_mip(create_simple_mip(), warmup_settings);
    if (!warmup.success) {
      job_status_result_t submit_status;
      submit_status.error_message = warmup.error_message;
      return fail("submit failed", submit_status);
    }

    constexpr auto queued_budget     = std::chrono::seconds(180);
    constexpr auto processing_budget = std::chrono::seconds(30);
    const auto start                 = std::chrono::steady_clock::now();
    bool saw_processing              = false;
    auto processing_since            = start;
    job_status_result_t last_status;

    for (;;) {
      last_status         = client->check_status(warmup.job_id);
      const bool terminal = last_status.success && (last_status.status == job_status_t::COMPLETED ||
                                                    last_status.status == job_status_t::FAILED ||
                                                    last_status.status == job_status_t::CANCELLED);
      if (terminal || (last_status.success && last_status.status == job_status_t::NOT_FOUND)) {
        break;
      }

      const auto now = std::chrono::steady_clock::now();
      if (last_status.success && last_status.status == job_status_t::PROCESSING) {
        if (!saw_processing) {
          saw_processing   = true;
          processing_since = now;
        }
        if (now - processing_since >= processing_budget) { break; }
      } else if (now - start >= queued_budget) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    const bool terminal = last_status.success && (last_status.status == job_status_t::COMPLETED ||
                                                  last_status.status == job_status_t::FAILED ||
                                                  last_status.status == job_status_t::CANCELLED);
    const bool queued_unclaimed = last_status.success && last_status.status == job_status_t::QUEUED;
    if (terminal || queued_unclaimed) {
      if (!client->delete_job(warmup.job_id)) { return fail("delete_job failed", last_status); }
    }

    if (last_status.success && last_status.status == job_status_t::COMPLETED) {
      result.ok = true;
      return result;
    }

    std::string why = "job ended without COMPLETED";
    if (last_status.success && last_status.status == job_status_t::PROCESSING) {
      why = "still PROCESSING after 30s (left running; delete would SIGKILL the worker)";
    } else if (last_status.success && last_status.status == job_status_t::QUEUED) {
      why = "still QUEUED after 180s";
    } else if (!last_status.success) {
      why = "status check failed";
    }
    return fail(why, last_status);
  }

  static std::string tail_server_log(const std::string& path)
  {
    constexpr std::streamoff kTailBytes = 4096;
    std::ifstream in(path, std::ios::binary);
    if (!in) { return "\n(server log unavailable: " + path + ")\n"; }
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    if (size > kTailBytes) {
      in.seekg(-kTailBytes, std::ios::end);
    } else {
      in.seekg(0);
    }
    std::ostringstream out;
    out << "\n--- server log tail (" << path << ") ---\n"
        << in.rdbuf() << "\n--- end server log tail ---\n";
    return out.str();
  }

  // Shared by the unary and chunked incumbent-set tests. EXPECT_* here still
  // fail the calling TEST_F; early returns only skip later checks after a
  // hard setup failure (ASSERT in a helper would not stop the test).
  void expect_incumbent_set_callback_registered(grpc_client_config_t config,
                                                const std::string& log_path)
  {
    GrpcTestLogCapture log_capture;
    log_capture.set_server_log_path(log_path);
    log_capture.mark_test_start();

    auto client = create_client(config);
    EXPECT_NE(client, nullptr);
    if (client == nullptr) { return; }

    solver_settings_t<int32_t, double> settings;
    settings.set_parameter(CUOPT_TIME_LIMIT, 10.0);
    auto submission = client->submit_mip(create_simple_mip(), settings, false, true);
    EXPECT_TRUE(submission.success) << submission.error_message;
    if (!submission.success) { return; }

    wait_for_job_done(client.get(), submission.job_id);
    EXPECT_EQ(client->check_status(submission.job_id).status, job_status_t::COMPLETED);
    EXPECT_TRUE(log_capture.wait_for_server_log("Registered incumbent set callback", 5000))
      << log_capture.get_server_logs();
  }

  int port_ = 0;
};

}  // namespace grpc_integration_fixture
