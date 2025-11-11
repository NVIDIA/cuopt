/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file cuopt_remote_server.cpp
 * @brief Remote solve server for cuOpt
 *
 * This server listens for TCP connections and solves optimization problems
 * sent from remote clients using binary serialization.
 */

#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/remote_solve.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <csignal>
#include <cstring>
#include <iostream>
#include <stdexcept>

// Global flag for graceful shutdown
volatile sig_atomic_t keep_running = 1;

void signal_handler(int signal)
{
  if (signal == SIGINT || signal == SIGTERM) {
    std::cout << "\n[Server] Received shutdown signal, cleaning up...\n";
    keep_running = 0;
  }
}

// Helper to write data to socket
static void write_all(int sockfd, const void* data, size_t size)
{
  const uint8_t* ptr = static_cast<const uint8_t*>(data);
  size_t remaining   = size;

  while (remaining > 0) {
    ssize_t written = ::write(sockfd, ptr, remaining);
    if (written <= 0) { throw std::runtime_error("Socket write failed"); }
    ptr += written;
    remaining -= written;
  }
}

// Helper to read data from socket
static void read_all(int sockfd, void* data, size_t size)
{
  uint8_t* ptr     = static_cast<uint8_t*>(data);
  size_t remaining = size;

  while (remaining > 0) {
    ssize_t nread = ::read(sockfd, ptr, remaining);
    if (nread <= 0) { throw std::runtime_error("Socket read failed"); }
    ptr += nread;
    remaining -= nread;
  }
}

// Handle a single client connection
void handle_client(int client_socket)
{
  try {
    std::cout << "[Server] Client connected\n";

    // Read request header
    cuopt::linear_programming::remote_solve_header_t header;
    read_all(client_socket, &header, sizeof(header));

    std::cout << "[Server] Received request:\n";
    std::cout << "  Version: " << header.version << "\n";
    std::cout << "  Problem type: " << (header.problem_type == 0 ? "LP" : "MIP") << "\n";
    std::cout << "  Problem size: " << header.problem_size << " bytes\n";
    std::cout << "  Index type size: " << header.i_type_size << "\n";
    std::cout << "  Float type size: " << header.f_type_size << "\n";

    // Validate header
    if (header.version != 1) {
      throw std::runtime_error("Unsupported protocol version: " + std::to_string(header.version));
    }

    if (header.i_type_size != sizeof(int)) {
      throw std::runtime_error("Unsupported index type size: " +
                               std::to_string(header.i_type_size));
    }

    if (header.f_type_size != sizeof(double)) {
      throw std::runtime_error("Unsupported float type size: " +
                               std::to_string(header.f_type_size));
    }

    // Read problem data
    std::vector<uint8_t> problem_data(header.problem_size);
    read_all(client_socket, problem_data.data(), header.problem_size);

    std::cout << "[Server] Received problem data (" << problem_data.size() << " bytes)\n";

    // Deserialize problem
    std::cout << "[Server] Deserializing problem...\n";
    auto problem = cuopt::linear_programming::deserialize_problem<int, double>(problem_data);

    std::cout << "[Server] Problem deserialized:\n";
    std::cout << "  Variables: " << problem.get_n_variables() << "\n";
    std::cout << "  Constraints: " << problem.get_n_constraints() << "\n";

    // Solve problem
    std::vector<uint8_t> solution_data;

    if (header.problem_type == 0) {
      // LP solve
      std::cout << "[Server] Solving LP problem...\n";

      cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings;
      auto solution = cuopt::linear_programming::solve_lp(problem, settings);

      std::cout << "[Server] LP solve completed\n";
      std::cout << "  Status: " << static_cast<int>(solution.get_termination_status()) << "\n";
      std::cout << "  Objective: " << solution.get_objective_value() << "\n";

      // Serialize solution
      solution_data = cuopt::linear_programming::serialize_solution(solution);

    } else {
      // MIP solve
      std::cout << "[Server] Solving MIP problem...\n";

      cuopt::linear_programming::mip_solver_settings_t<int, double> settings;
      auto solution = cuopt::linear_programming::solve_mip(problem, settings);

      std::cout << "[Server] MIP solve completed\n";
      std::cout << "  Status: " << static_cast<int>(solution.get_termination_status()) << "\n";
      std::cout << "  Objective: " << solution.get_objective_value() << "\n";

      // Serialize solution
      solution_data = cuopt::linear_programming::serialize_mip_solution(solution);
    }

    std::cout << "[Server] Solution serialized (" << solution_data.size() << " bytes)\n";

    // Send response header
    cuopt::linear_programming::remote_solve_response_header_t response_header;
    response_header.status        = 0;  // Success
    response_header.solution_size = solution_data.size();

    write_all(client_socket, &response_header, sizeof(response_header));

    // Send solution data
    write_all(client_socket, solution_data.data(), solution_data.size());

    std::cout << "[Server] Response sent successfully\n";

  } catch (const std::exception& e) {
    std::cerr << "[Server] Error handling client: " << e.what() << "\n";

    // Try to send error response
    try {
      cuopt::linear_programming::remote_solve_response_header_t response_header;
      response_header.status        = 1;  // Error
      response_header.solution_size = 0;
      write_all(client_socket, &response_header, sizeof(response_header));
    } catch (...) {
      // Ignore write errors during error handling
    }
  }

  close(client_socket);
  std::cout << "[Server] Client disconnected\n\n";
}

int main(int argc, char* argv[])
{
  // Parse command line arguments
  int port = 9999;  // Default port

  if (argc > 1) {
    port = std::atoi(argv[1]);
    if (port <= 0 || port > 65535) {
      std::cerr << "Error: Invalid port number: " << argv[1] << "\n";
      std::cerr << "Usage: " << argv[0] << " [port]\n";
      std::cerr << "  Default port: 9999\n";
      return 1;
    }
  }

  std::cout << "==========================================================\n";
  std::cout << "cuOpt Remote Solve Server\n";
  std::cout << "==========================================================\n";
  std::cout << "Port: " << port << "\n";
  std::cout << "Press Ctrl+C to stop\n";
  std::cout << "==========================================================\n\n";

  // Setup signal handlers for graceful shutdown
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  // Create socket
  int server_socket = socket(AF_INET, SOCK_STREAM, 0);
  if (server_socket < 0) {
    std::cerr << "Error: Failed to create socket\n";
    return 1;
  }

  // Set socket options
  int opt = 1;
  if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    std::cerr << "Warning: Failed to set SO_REUSEADDR\n";
  }

  // Bind to port
  struct sockaddr_in server_addr;
  std::memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family      = AF_INET;
  server_addr.sin_addr.s_addr = INADDR_ANY;
  server_addr.sin_port        = htons(port);

  if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    std::cerr << "Error: Failed to bind to port " << port << "\n";
    std::cerr << "Port may already be in use\n";
    close(server_socket);
    return 1;
  }

  // Listen for connections
  if (listen(server_socket, 5) < 0) {
    std::cerr << "Error: Failed to listen on socket\n";
    close(server_socket);
    return 1;
  }

  std::cout << "[Server] Listening on port " << port << "...\n\n";

  // Main server loop
  while (keep_running) {
    // Set timeout for accept to allow checking keep_running
    struct timeval tv;
    tv.tv_sec  = 1;
    tv.tv_usec = 0;
    setsockopt(server_socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    // Accept client connection
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);

    int client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);

    if (client_socket < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        // Timeout, check keep_running and continue
        continue;
      }
      if (keep_running) { std::cerr << "[Server] Warning: Failed to accept connection\n"; }
      continue;
    }

    // Get client IP
    char client_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
    std::cout << "[Server] Connection from " << client_ip << ":" << ntohs(client_addr.sin_port)
              << "\n";

    // Handle client (blocking - single-threaded for simplicity)
    handle_client(client_socket);
  }

  // Cleanup
  close(server_socket);
  std::cout << "[Server] Server stopped\n";

  return 0;
}
