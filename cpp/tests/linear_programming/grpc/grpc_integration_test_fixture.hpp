/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file grpc_integration_test_fixture.hpp
 * @brief Shared server-spawn fixture for gRPC integration tests.
 *
 * Included (not linked) separately by grpc_integration_test.cpp and
 * grpc_multi_gpu_test.cpp, so each test binary gets its own private copy of
 * these classes -- fine, since the two never link together.
 *
 * Environment variables:
 *   CUOPT_GRPC_SERVER_PATH - Path to cuopt_grpc_server binary
 *   CUOPT_TEST_PORT_BASE   - Base port for test servers (default: 19000)
 *   RAPIDS_DATASET_ROOT_DIR - Path to test datasets
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cerrno>

#include <cuopt/mathematical_optimization/cpu_optimization_problem.hpp>
#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <cuopt/mathematical_optimization/optimization_problem_utils.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <utilities/inline_lp_test_utils.hpp>
#include "grpc_client.hpp"

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
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

// ServerProcess spawn tree:
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

    // Reparent any process orphaned by killing the server (i.e. the GPU worker)
    // onto this test process so stop() can reap it without relying on init.
    prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0);

    port_ = port;
    pid_  = fork();
    if (pid_ < 0) {
      std::cerr << "fork() failed\n";
      return false;
    }

    if (pid_ == 0) {
      setpgid(0, 0);  // child leads its own group; parent mirrors this below

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

      cuopt::mathematical_optimization::grpc_client_config_t config;
      config.server_address = "localhost:" + std::to_string(port_);

      if (!tls_root_certs_.empty()) {
        config.enable_tls      = true;
        config.tls_root_certs  = tls_root_certs_;
        config.tls_client_cert = tls_client_cert_;
        config.tls_client_key  = tls_client_key_;
      }

      cuopt::mathematical_optimization::grpc_client_t client(config);
      if (client.connect()) { return true; }

      int status = 0;
      if (waitpid(pid_, &status, WNOHANG) == pid_) {
        std::cerr << "Server process died during startup\n";
        return false;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(200));
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
// Base Test Class
// =============================================================================

class GrpcIntegrationTestBase : public ::testing::Test {
 protected:
  std::unique_ptr<cuopt::mathematical_optimization::grpc_client_t> create_client(
    cuopt::mathematical_optimization::grpc_client_config_t config = {})
  {
    config.server_address   = "localhost:" + std::to_string(port_);
    config.poll_interval_ms = 100;

    if (config.timeout_seconds == 3600) { config.timeout_seconds = 60; }

    auto client = std::make_unique<cuopt::mathematical_optimization::grpc_client_t>(config);
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
  cuopt::mathematical_optimization::cpu_optimization_problem_t<int32_t, double>
  load_problem_from_file(const std::string& path)
  {
    auto mps_data = cuopt::mathematical_optimization::io::read<int32_t, double>(path);
    cuopt::mathematical_optimization::cpu_optimization_problem_t<int32_t, double> problem;
    cuopt::mathematical_optimization::populate_from_mps_data_model(&problem, mps_data);
    return problem;
  }

  void wait_for_job_done(cuopt::mathematical_optimization::grpc_client_t* client,
                         const std::string& job_id,
                         int max_seconds = 30)
  {
    for (int i = 0; i < max_seconds * 2; ++i) {
      auto status = client->check_status(job_id);
      if (status.status == cuopt::mathematical_optimization::job_status_t::COMPLETED ||
          status.status == cuopt::mathematical_optimization::job_status_t::FAILED ||
          status.status == cuopt::mathematical_optimization::job_status_t::CANCELLED) {
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }

  int port_ = 0;
};

}  // namespace
