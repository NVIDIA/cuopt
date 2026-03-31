/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "c_api_tests.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include <cuopt/linear_programming/cuopt_c.h>
#include <pdlp/cuopt_c_internal.hpp>

#include <utilities/common_utils.hpp>
#include <utilities/error.hpp>

namespace cuopt::linear_programming::detail {
bool is_cusparse_runtime_mixed_precision_supported();
}

#include <gtest/gtest.h>

TEST(c_api, int_size) { EXPECT_EQ(test_int_size(), sizeof(int32_t)); }

TEST(c_api, float_size) { EXPECT_EQ(test_float_size(), sizeof(double)); }

TEST(c_api, afiro)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  std::string filename = rapidsDatasetRootDir + "/linear_programming/" + "afiro_original.mps";
  int termination_status;
  EXPECT_EQ(solve_mps_file(filename.c_str(), 60, CUOPT_INFINITY, &termination_status),
            CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, CUOPT_TERMINATION_STATUS_OPTIMAL);
}

// Test both LP and MIP codepaths
class TimeLimitTestFixture : public ::testing::TestWithParam<std::tuple<std::string, double, int>> {
};
TEST_P(TimeLimitTestFixture, time_limit)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  std::string filename                    = rapidsDatasetRootDir + std::get<0>(GetParam());
  double target_solve_time                = std::get<1>(GetParam());
  int method                              = std::get<2>(GetParam());
  int termination_status;
  double solve_time = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(solve_mps_file(filename.c_str(),
                           target_solve_time,
                           CUOPT_INFINITY,
                           &termination_status,
                           &solve_time,
                           method),
            CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, CUOPT_TERMINATION_STATUS_TIME_LIMIT);

  // Dual simplex is spending some time for factorizing the basis, and this computation does not
  // check for time limit
  double excess_allowed_time = 3.0;
  EXPECT_NEAR(solve_time, target_solve_time, excess_allowed_time);
}
INSTANTIATE_TEST_SUITE_P(
  c_api,
  TimeLimitTestFixture,
  ::testing::Values(
    std::make_tuple("/linear_programming/square41/square41.mps",
                    5,
                    CUOPT_METHOD_DUAL_SIMPLEX),  // LP, Dual Simplex
    std::make_tuple("/linear_programming/square41/square41.mps", 5, CUOPT_METHOD_PDLP),  // LP, PDLP
    std::make_tuple("/mip/supportcase22.mps", 15, CUOPT_METHOD_DUAL_SIMPLEX)             // MIP
    ));

TEST(c_api, iteration_limit)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  std::string filename = rapidsDatasetRootDir + "/linear_programming/" + "afiro_original.mps";
  int termination_status;
  EXPECT_EQ(solve_mps_file(filename.c_str(), 60, 1, &termination_status), CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, CUOPT_TERMINATION_STATUS_ITERATION_LIMIT);
}

TEST(c_api, solve_time_bb_preemption)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  std::string filename                    = rapidsDatasetRootDir + "/mip/" + "bb_optimality.mps";
  int termination_status;
  double solve_time = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(solve_mps_file(filename.c_str(), 5, CUOPT_INFINITY, &termination_status, &solve_time),
            CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_GT(solve_time, 0);  // solve time should not be equal to 0, even on very simple instances
  // solved by B&B before the diversity solver has time to run
}

TEST(c_api, bad_parameter_name) { EXPECT_EQ(test_bad_parameter_name(), CUOPT_INVALID_ARGUMENT); }

TEST(c_api, mip_get_callbacks_only) { EXPECT_EQ(test_mip_get_callbacks_only(), CUOPT_SUCCESS); }

TEST(c_api, mip_get_set_callbacks) { EXPECT_EQ(test_mip_get_set_callbacks(), CUOPT_SUCCESS); }

TEST(c_api, burglar) { EXPECT_EQ(burglar_problem(), CUOPT_SUCCESS); }

TEST(c_api, test_missing_file) { EXPECT_EQ(test_missing_file(), CUOPT_MPS_FILE_ERROR); }

TEST(c_api, test_infeasible_problem) { EXPECT_EQ(test_infeasible_problem(), CUOPT_SUCCESS); }

TEST(c_api, test_ranged_problem)
{
  cuopt_int_t termination_status;
  cuopt_float_t objective;
  EXPECT_EQ(test_ranged_problem(&termination_status, &objective), CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_NEAR(objective, 32.0, 1e-3);
}

TEST(c_api, test_invalid_bounds)
{
  // Test LP codepath
  EXPECT_EQ(test_invalid_bounds(false), CUOPT_SUCCESS);
  // Test MIP codepath
  EXPECT_EQ(test_invalid_bounds(true), CUOPT_SUCCESS);
}

TEST(c_api, test_quadratic_problem)
{
  cuopt_int_t termination_status;
  cuopt_float_t objective;
  EXPECT_EQ(test_quadratic_problem(&termination_status, &objective), CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_NEAR(objective, -32.0, 1e-3);
}

TEST(c_api, test_quadratic_ranged_problem)
{
  cuopt_int_t termination_status;
  cuopt_float_t objective;
  EXPECT_EQ(test_quadratic_ranged_problem(&termination_status, &objective), CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, (int)CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_NEAR(objective, -32.0, 1e-3);
}

TEST(c_api, test_write_problem)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  std::string input_file = rapidsDatasetRootDir + "/linear_programming/afiro_original.mps";
  std::string temp_file = std::filesystem::temp_directory_path().string() + "/c_api_test_write.mps";
  EXPECT_EQ(test_write_problem(input_file.c_str(), temp_file.c_str()), CUOPT_SUCCESS);
  std::filesystem::remove(temp_file);
}

TEST(c_api, test_maximize_problem_dual_variables)
{
  cuopt_int_t termination_status;
  cuopt_float_t objective, dual_objective;
  cuopt_float_t dual_variables[3];
  cuopt_float_t reduced_costs[4];
  for (cuopt_int_t method = CUOPT_METHOD_CONCURRENT; method <= CUOPT_METHOD_BARRIER; method++) {
    EXPECT_EQ(
      test_maximize_problem_dual_variables(
        method, &termination_status, &objective, dual_variables, reduced_costs, &dual_objective),
      CUOPT_SUCCESS);
    EXPECT_EQ(termination_status, CUOPT_TERMINATION_STATUS_OPTIMAL);
    EXPECT_NEAR(objective,
                dual_objective,
                method == CUOPT_METHOD_CONCURRENT || method == CUOPT_METHOD_PDLP ? 1e-2 : 1e-5);
  }
}

static bool test_mps_roundtrip(const std::string& mps_file_path)
{
  using cuopt::linear_programming::problem_and_stream_view_t;

  cuOptOptimizationProblem original_handle = nullptr;
  cuOptOptimizationProblem reread_handle   = nullptr;
  bool result                              = false;

  std::string model_basename = std::filesystem::path(mps_file_path).filename().string();
  std::string temp_file =
    std::filesystem::temp_directory_path().string() + "/roundtrip_temp_" + model_basename;

  if (cuOptReadProblem(mps_file_path.c_str(), &original_handle) != CUOPT_SUCCESS) {
    std::cerr << "Failed to read original MPS file: " << mps_file_path << std::endl;
    goto cleanup;
  }

  if (cuOptWriteProblem(original_handle, temp_file.c_str(), CUOPT_FILE_FORMAT_MPS) !=
      CUOPT_SUCCESS) {
    std::cerr << "Failed to write MPS file: " << temp_file << std::endl;
    goto cleanup;
  }

  if (cuOptReadProblem(temp_file.c_str(), &reread_handle) != CUOPT_SUCCESS) {
    std::cerr << "Failed to re-read MPS file: " << temp_file << std::endl;
    goto cleanup;
  }

  {
    auto* original_problem_wrapper = static_cast<problem_and_stream_view_t*>(original_handle);
    auto* reread_problem_wrapper   = static_cast<problem_and_stream_view_t*>(reread_handle);

    // Use the interface method to compare (works for both CPU and GPU backends)
    result = original_problem_wrapper->get_problem()->is_equivalent(
      *reread_problem_wrapper->get_problem());
  }

cleanup:
  std::filesystem::remove(temp_file);
  cuOptDestroyProblem(&original_handle);
  cuOptDestroyProblem(&reread_handle);

  return result;
}

class WriteRoundtripTestFixture : public ::testing::TestWithParam<std::string> {};
TEST_P(WriteRoundtripTestFixture, roundtrip)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  EXPECT_TRUE(test_mps_roundtrip(rapidsDatasetRootDir + GetParam()));
}
INSTANTIATE_TEST_SUITE_P(c_api,
                         WriteRoundtripTestFixture,
                         ::testing::Values("/linear_programming/afiro_original.mps",
                                           "/mip/50v-10.mps",
                                           "/mip/fiball.mps",
                                           "/mip/gen-ip054.mps",
                                           "/mip/sct2.mps",
                                           "/mip/uccase9.mps",
                                           "/mip/drayage-25-23.mps",
                                           "/mip/tr12-30.mps",
                                           "/mip/neos-3004026-krka.mps",
                                           "/mip/ns1208400.mps",
                                           "/mip/gmu-35-50.mps",
                                           "/mip/n2seq36q.mps",
                                           "/mip/seymour1.mps",
                                           "/mip/rmatr200-p5.mps",
                                           "/mip/cvs16r128-89.mps",
                                           "/mip/thor50dday.mps",
                                           "/mip/stein9inf.mps",
                                           "/mip/neos5.mps",
                                           "/mip/neos5-free-bound.mps",
                                           "/mip/crossing_var_bounds.mps",
                                           "/mip/cod105_max.mps",
                                           "/mip/sudoku.mps",
                                           "/mip/presolve-infeasible.mps",
                                           "/mip/swath1.mps",
                                           "/mip/enlight_hard.mps",
                                           "/mip/enlight11.mps",
                                           "/mip/supportcase22.mps"));

class DeterministicBBTestFixture
  : public ::testing::TestWithParam<std::tuple<std::string, int, double, double>> {};
TEST_P(DeterministicBBTestFixture, deterministic_reproducibility)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  std::string filename                    = rapidsDatasetRootDir + std::get<0>(GetParam());
  int num_threads                         = std::get<1>(GetParam());
  double time_limit                       = std::get<2>(GetParam());
  double work_limit                       = std::get<3>(GetParam());

  // Run 3 times and verify identical results
  EXPECT_EQ(test_deterministic_bb(filename.c_str(), 3, num_threads, time_limit, work_limit),
            CUOPT_SUCCESS);
}
INSTANTIATE_TEST_SUITE_P(c_api,
                         DeterministicBBTestFixture,
                         ::testing::Values(
                           // Low thread count
                           std::make_tuple("/mip/gen-ip054.mps", 4, 60.0, 2),
                           // High thread count (high contention)
                           std::make_tuple("/mip/gen-ip054.mps", 128, 60.0, 2),
                           // Different instance
                           std::make_tuple("/mip/bb_optimality.mps", 8, 60.0, 2)));

// =============================================================================
// PDLP Precision Tests
// =============================================================================

TEST(c_api, pdlp_precision_single)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  std::string filename = rapidsDatasetRootDir + "/linear_programming/afiro_original.mps";
  cuopt_int_t termination_status;
  cuopt_float_t objective;
  EXPECT_EQ(test_pdlp_precision_single(filename.c_str(), &termination_status, &objective),
            CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_NEAR(objective, -464.7531, 1e-1);
}

TEST(c_api, pdlp_precision_mixed)
{
  using namespace cuopt::linear_programming::detail;
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  std::string filename           = rapidsDatasetRootDir + "/linear_programming/afiro_original.mps";
  cuopt_int_t termination_status = -1;
  cuopt_float_t objective;
  if (!is_cusparse_runtime_mixed_precision_supported()) {
    auto status = test_pdlp_precision_mixed(filename.c_str(), &termination_status, &objective);
    bool solve_returned_error = (status != CUOPT_SUCCESS);
    bool solve_returned_non_optimal =
      (status == CUOPT_SUCCESS && termination_status != CUOPT_TERMINATION_STATUS_OPTIMAL);
    EXPECT_TRUE(solve_returned_error || solve_returned_non_optimal);
    return;
  }
  EXPECT_EQ(test_pdlp_precision_mixed(filename.c_str(), &termination_status, &objective),
            CUOPT_SUCCESS);
  EXPECT_EQ(termination_status, CUOPT_TERMINATION_STATUS_OPTIMAL);
  EXPECT_NEAR(objective, -464.7531, 1e-1);
}

// =============================================================================
// Solution Interface Polymorphism Tests
// =============================================================================

TEST(c_api, lp_solution_mip_methods) { EXPECT_EQ(test_lp_solution_mip_methods(), CUOPT_SUCCESS); }

TEST(c_api, mip_solution_lp_methods) { EXPECT_EQ(test_mip_solution_lp_methods(), CUOPT_SUCCESS); }

// =============================================================================
// CPU-Only Execution Tests
// These tests verify that cuOpt can run on a CPU-only host with remote execution
// enabled, forwarding solves to a real cuopt_grpc_server over gRPC.
//
// A single shared server is started once for all tests in this fixture
// (SetUpTestSuite / TearDownTestSuite) to avoid per-test startup overhead.
// =============================================================================

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <thread>

namespace {

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

bool tcp_connect_check(int port, int timeout_ms)
{
  auto start = std::chrono::steady_clock::now();
  while (true) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return false;

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == 0) {
      close(sock);
      return true;
    }
    close(sock);

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
    if (elapsed.count() >= timeout_ms) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}

}  // namespace

class CpuOnlyWithServerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite()
  {
    server_path_ = find_server_binary();
    if (server_path_.empty()) {
      skip_reason_ = "cuopt_grpc_server binary not found";
      return;
    }

    port_                = 18500;
    const char* env_base = std::getenv("CUOPT_TEST_PORT_BASE");
    if (env_base) { port_ = std::atoi(env_base) + 500; }

    server_pid_ = fork();
    if (server_pid_ < 0) {
      skip_reason_ = "fork() failed";
      return;
    }

    if (server_pid_ == 0) {
      std::string port_str = std::to_string(port_);
      std::string log_file = "/tmp/cuopt_c_api_test_server_" + port_str + ".log";
      int fd               = open(log_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (fd >= 0) {
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
        close(fd);
      }
      execl(server_path_.c_str(),
            server_path_.c_str(),
            "--port",
            port_str.c_str(),
            "--workers",
            "1",
            nullptr);
      _exit(127);
    }

    if (!tcp_connect_check(port_, 15000)) {
      skip_reason_ = "cuopt_grpc_server failed to start within 15 seconds";
      kill(server_pid_, SIGKILL);
      waitpid(server_pid_, nullptr, 0);
      server_pid_ = -1;
      return;
    }

    const char* cv     = getenv("CUDA_VISIBLE_DEVICES");
    const char* rh     = getenv("CUOPT_REMOTE_HOST");
    const char* rp     = getenv("CUOPT_REMOTE_PORT");
    orig_cuda_visible_ = cv ? cv : "";
    orig_remote_host_  = rh ? rh : "";
    orig_remote_port_  = rp ? rp : "";
    cuda_was_set_      = (cv != nullptr);
    host_was_set_      = (rh != nullptr);
    port_was_set_      = (rp != nullptr);

    setenv("CUDA_VISIBLE_DEVICES", "", 1);
    setenv("CUOPT_REMOTE_HOST", "localhost", 1);
    setenv("CUOPT_REMOTE_PORT", std::to_string(port_).c_str(), 1);
  }

  static void TearDownTestSuite()
  {
    if (cuda_was_set_) {
      setenv("CUDA_VISIBLE_DEVICES", orig_cuda_visible_.c_str(), 1);
    } else {
      unsetenv("CUDA_VISIBLE_DEVICES");
    }
    if (host_was_set_) {
      setenv("CUOPT_REMOTE_HOST", orig_remote_host_.c_str(), 1);
    } else {
      unsetenv("CUOPT_REMOTE_HOST");
    }
    if (port_was_set_) {
      setenv("CUOPT_REMOTE_PORT", orig_remote_port_.c_str(), 1);
    } else {
      unsetenv("CUOPT_REMOTE_PORT");
    }

    if (server_pid_ > 0) {
      kill(server_pid_, SIGTERM);
      int status;
      int wait_ms = 0;
      while (wait_ms < 5000) {
        if (waitpid(server_pid_, &status, WNOHANG) != 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_ms += 100;
      }
      if (waitpid(server_pid_, &status, WNOHANG) == 0) {
        kill(server_pid_, SIGKILL);
        waitpid(server_pid_, &status, 0);
      }
      server_pid_ = -1;
    }
  }

  void SetUp() override
  {
    if (!skip_reason_.empty()) { GTEST_SKIP() << skip_reason_; }
  }

  static std::string server_path_;
  static std::string skip_reason_;
  static pid_t server_pid_;
  static int port_;

  static std::string orig_cuda_visible_;
  static std::string orig_remote_host_;
  static std::string orig_remote_port_;
  static bool cuda_was_set_;
  static bool host_was_set_;
  static bool port_was_set_;
};

std::string CpuOnlyWithServerTest::server_path_;
std::string CpuOnlyWithServerTest::skip_reason_;
pid_t CpuOnlyWithServerTest::server_pid_ = -1;
int CpuOnlyWithServerTest::port_         = 0;
std::string CpuOnlyWithServerTest::orig_cuda_visible_;
std::string CpuOnlyWithServerTest::orig_remote_host_;
std::string CpuOnlyWithServerTest::orig_remote_port_;
bool CpuOnlyWithServerTest::cuda_was_set_ = false;
bool CpuOnlyWithServerTest::host_was_set_ = false;
bool CpuOnlyWithServerTest::port_was_set_ = false;

TEST_F(CpuOnlyWithServerTest, lp_solve)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  std::string lp_file = rapidsDatasetRootDir + "/linear_programming/afiro_original.mps";
  EXPECT_EQ(test_cpu_only_execution(lp_file.c_str()), CUOPT_SUCCESS);
}

TEST_F(CpuOnlyWithServerTest, mip_solve)
{
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  std::string mip_file                    = rapidsDatasetRootDir + "/mip/bb_optimality.mps";
  EXPECT_EQ(test_cpu_only_mip_execution(mip_file.c_str()), CUOPT_SUCCESS);
}

// Note: cuopt_cli subprocess tests are in Python (test_cpu_only_execution.py)
// which provides better cross-platform subprocess handling
