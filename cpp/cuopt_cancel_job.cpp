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

#include <cuopt/linear_programming/utilities/remote_serialization.hpp>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace cuopt::linear_programming;

static bool write_all(int sockfd, const void* data, size_t size)
{
  const uint8_t* ptr = static_cast<const uint8_t*>(data);
  size_t remaining   = size;
  while (remaining > 0) {
    ssize_t written = ::write(sockfd, ptr, remaining);
    if (written <= 0) return false;
    ptr += written;
    remaining -= written;
  }
  return true;
}

static bool read_all(int sockfd, void* data, size_t size)
{
  uint8_t* ptr     = static_cast<uint8_t*>(data);
  size_t remaining = size;
  while (remaining > 0) {
    ssize_t nread = ::read(sockfd, ptr, remaining);
    if (nread <= 0) return false;
    ptr += nread;
    remaining -= nread;
  }
  return true;
}

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

  // Connect to server
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    std::cerr << "Error: Failed to create socket\n";
    return 1;
  }

  struct sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port   = htons(port);

  if (inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr) <= 0) {
    // Try hostname resolution
    struct hostent* he = gethostbyname(host.c_str());
    if (he == nullptr) {
      std::cerr << "Error: Invalid host: " << host << "\n";
      close(sockfd);
      return 1;
    }
    memcpy(&server_addr.sin_addr, he->h_addr_list[0], he->h_length);
  }

  if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    std::cerr << "Error: Failed to connect to " << host << ":" << port << "\n";
    close(sockfd);
    return 1;
  }

  // Send cancel request
  auto serializer    = get_serializer<int, double>();
  auto request_bytes = serializer->serialize_cancel_request(job_id);

  uint32_t size = request_bytes.size();
  if (!write_all(sockfd, &size, sizeof(size)) ||
      !write_all(sockfd, request_bytes.data(), request_bytes.size())) {
    std::cerr << "Error: Failed to send cancel request\n";
    close(sockfd);
    return 1;
  }

  // Receive response
  if (!read_all(sockfd, &size, sizeof(size))) {
    std::cerr << "Error: Failed to receive response size\n";
    close(sockfd);
    return 1;
  }

  std::vector<uint8_t> response_bytes(size);
  if (!read_all(sockfd, response_bytes.data(), size)) {
    std::cerr << "Error: Failed to receive response\n";
    close(sockfd);
    return 1;
  }

  close(sockfd);

  // Parse response
  auto result = serializer->deserialize_cancel_response(response_bytes);

  // Print result
  std::cout << "Job ID: " << job_id << "\n";
  std::cout << "Result: " << (result.success ? "SUCCESS" : "FAILED") << "\n";
  std::cout << "Message: " << result.message << "\n";

  const char* status_str = "UNKNOWN";
  using job_status_t     = remote_serializer_t<int, double>::job_status_t;
  switch (result.job_status) {
    case job_status_t::QUEUED: status_str = "QUEUED"; break;
    case job_status_t::PROCESSING: status_str = "PROCESSING"; break;
    case job_status_t::COMPLETED: status_str = "COMPLETED"; break;
    case job_status_t::FAILED: status_str = "FAILED"; break;
    case job_status_t::NOT_FOUND: status_str = "NOT_FOUND"; break;
    case job_status_t::CANCELLED: status_str = "CANCELLED"; break;
  }
  std::cout << "Job Status: " << status_str << "\n";

  return result.success ? 0 : 1;
}
