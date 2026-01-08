/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file cuopt_cancel_job.cpp
 * @brief Standalone utility to cancel jobs on a cuopt_remote_server
 *
 * Usage:
 *   cuopt_cancel_job <job_id> [-h host] [-p port]
 *
 * Examples:
 *   cuopt_cancel_job job_1234567890abcdef
 *   cuopt_cancel_job job_1234567890abcdef -h 192.168.1.100 -p 9090
 */

#include <cuopt/linear_programming/utilities/remote_solve.hpp>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

using namespace cuopt::linear_programming;

void print_usage(const char* prog)
{
  std::cout << "Usage: " << prog << " <job_id> [options]\n"
            << "\n"
            << "Cancel a job on a cuopt_remote_server.\n"
            << "\n"
            << "Arguments:\n"
            << "  job_id           The job ID to cancel\n"
            << "\n"
            << "Options:\n"
            << "  -h HOST          Server hostname (default: localhost)\n"
            << "  -p PORT          Server port (default: 9090)\n"
            << "  --help           Show this help message\n"
            << "\n"
            << "Environment Variables:\n"
            << "  CUOPT_REMOTE_HOST   Default server host\n"
            << "  CUOPT_REMOTE_PORT   Default server port\n"
            << "\n"
            << "Examples:\n"
            << "  " << prog << " job_1234567890abcdef\n"
            << "  " << prog << " job_1234567890abcdef -h 192.168.1.100 -p 9090\n";
}

const char* status_to_string(remote_job_status_t status)
{
  switch (status) {
    case remote_job_status_t::QUEUED: return "QUEUED";
    case remote_job_status_t::PROCESSING: return "PROCESSING";
    case remote_job_status_t::COMPLETED: return "COMPLETED";
    case remote_job_status_t::FAILED: return "FAILED";
    case remote_job_status_t::NOT_FOUND: return "NOT_FOUND";
    case remote_job_status_t::CANCELLED: return "CANCELLED";
    default: return "UNKNOWN";
  }
}

int main(int argc, char** argv)
{
  // Parse arguments
  std::string job_id;
  std::string host = "localhost";
  int port         = 9090;

  // Check environment variables first
  const char* env_host = std::getenv("CUOPT_REMOTE_HOST");
  const char* env_port = std::getenv("CUOPT_REMOTE_PORT");
  if (env_host && env_host[0]) { host = env_host; }
  if (env_port && env_port[0]) { port = std::atoi(env_port); }

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    } else if (strcmp(argv[i], "-h") == 0 && i + 1 < argc) {
      host = argv[++i];
    } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
      port = std::stoi(argv[++i]);
    } else if (argv[i][0] != '-') {
      job_id = argv[i];
    }
  }

  if (job_id.empty()) {
    std::cerr << "Error: No job_id specified\n\n";
    print_usage(argv[0]);
    return 1;
  }

  // Cancel the job using the remote solve API
  remote_solve_config_t config{host, port};
  auto result = cancel_job_remote(config, job_id);

  // Print result
  std::cout << "Job ID: " << job_id << "\n";
  std::cout << "Result: " << (result.success ? "SUCCESS" : "FAILED") << "\n";
  std::cout << "Message: " << result.message << "\n";
  std::cout << "Job Status: " << status_to_string(result.job_status) << "\n";

  return result.success ? 0 : 1;
}
