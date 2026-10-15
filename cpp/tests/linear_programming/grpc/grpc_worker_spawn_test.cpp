/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file grpc_worker_spawn_test.cpp
 * @brief Checks the internal --worker entry point without starting a server.
 *
 * These cases exit during argument parsing, before CUDA or a listen socket.
 */

#include <gtest/gtest.h>

#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

// Must match kWorkerAttachFailedExitCode in grpc_server_types.hpp. The monitor
// treats this code as fatal. Exit code 1 would be respawned forever.
constexpr int kWorkerAttachFailedExitCode = 87;

namespace {

struct ServerExit {
  int code       = -1;
  bool timed_out = false;
  std::string stderr_text;
};

std::string server_binary()
{
  const char* path = std::getenv("CUOPT_GRPC_SERVER_PATH");
  return path == nullptr ? std::string() : std::string(path);
}

ServerExit run_server(const std::vector<std::string>& args)
{
  ServerExit result;
  const std::string binary = server_binary();
  int pipe_fds[2]          = {-1, -1};
  if (binary.empty() || pipe(pipe_fds) != 0) {
    result.timed_out = true;
    return result;
  }

  const pid_t pid = fork();
  if (pid < 0) {
    close(pipe_fds[0]);
    close(pipe_fds[1]);
    result.timed_out = true;
    return result;
  }
  if (pid == 0) {
    dup2(pipe_fds[1], STDERR_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
      dup2(devnull, STDOUT_FILENO);
      close(devnull);
    }
    close(pipe_fds[0]);
    close(pipe_fds[1]);

    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(binary.c_str()));
    for (const std::string& arg : args) {
      argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);
    execv(binary.c_str(), argv.data());
    _exit(127);
  }

  close(pipe_fds[1]);
  // Passing runs return as soon as the child exits. The deadline only bounds a
  // hang, and loading the server's shared libraries can exceed a few seconds
  // on a cold CI runner.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  std::string err;
  bool child_done = false;
  int status      = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    pollfd pfd{pipe_fds[0], POLLIN, 0};
    const int poll_rc = poll(&pfd, 1, 50);
    if (poll_rc > 0 && (pfd.revents & (POLLIN | POLLHUP)) != 0) {
      char buf[512];
      const ssize_t n = read(pipe_fds[0], buf, sizeof(buf));
      if (n > 0) err.append(buf, static_cast<size_t>(n));
    }
    const pid_t waited = waitpid(pid, &status, WNOHANG);
    if (waited == pid) {
      child_done = true;
      break;
    }
  }
  if (child_done) {
    while (true) {
      pollfd pfd{pipe_fds[0], POLLIN, 0};
      if (poll(&pfd, 1, 0) <= 0) break;
      char buf[512];
      const ssize_t n = read(pipe_fds[0], buf, sizeof(buf));
      if (n <= 0) break;
      err.append(buf, static_cast<size_t>(n));
    }
  }
  close(pipe_fds[0]);
  if (!child_done) {
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    result.timed_out   = true;
    result.stderr_text = std::move(err);
    return result;
  }
  result.stderr_text = std::move(err);
  if (WIFEXITED(status)) result.code = WEXITSTATUS(status);
  return result;
}

}  // namespace

TEST(GrpcWorkerSpawn, IncompleteWorkerArgsExitAttachFailure)
{
  const std::string binary = server_binary();
  if (binary.empty()) { GTEST_SKIP() << "CUOPT_GRPC_SERVER_PATH is not set"; }

  const ServerExit exit = run_server({"--worker"});
  EXPECT_FALSE(exit.timed_out);
  EXPECT_EQ(exit.code, kWorkerAttachFailedExitCode);
  EXPECT_NE(exit.stderr_text.find("incomplete arguments"), std::string::npos);
}

TEST(GrpcWorkerSpawn, WorkerFlagAfterUserOptionIsNotWorkerMode)
{
  const std::string binary = server_binary();
  if (binary.empty()) { GTEST_SKIP() << "CUOPT_GRPC_SERVER_PATH is not set"; }

  // --worker is not argv[1], so this must stay on the server argument parser.
  const ServerExit exit = run_server({"--server-log", "/tmp/cuopt_spawn_review.log", "--worker"});
  EXPECT_FALSE(exit.timed_out);
  EXPECT_EQ(exit.code, 1);
  EXPECT_EQ(exit.stderr_text.find("incomplete arguments"), std::string::npos);
  EXPECT_EQ(exit.stderr_text.find("cuOpt gRPC Remote Solve Server"), std::string::npos);
}
