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
 * @brief Remote solve server for cuOpt using Protocol Buffers
 *
 * This server listens for TCP connections and solves optimization problems
 * sent from remote clients using Protocol Buffers serialization.
 */

#include <cuopt/linear_programming/solve.hpp>

#include <cuopt_remote.pb.h>

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

// Convert protobuf OptimizationProblem to optimization_problem_t
template <typename i_t, typename f_t>
static cuopt::linear_programming::optimization_problem_t<i_t, f_t> protobuf_to_problem(
  const cuopt::remote::OptimizationProblem& pb_problem)
{
  cuopt::linear_programming::optimization_problem_t<i_t, f_t> problem;

  // Set problem sense
  problem.set_maximize(pb_problem.maximize());
  problem.set_objective_scaling_factor(static_cast<f_t>(pb_problem.objective_scaling_factor()));
  problem.set_objective_offset(static_cast<f_t>(pb_problem.objective_offset()));

  // Convert constraint matrix
  std::vector<f_t> matrix_values;
  std::vector<i_t> matrix_indices;
  std::vector<i_t> matrix_offsets;

  matrix_values.reserve(pb_problem.constraint_matrix_values_size());
  for (int i = 0; i < pb_problem.constraint_matrix_values_size(); ++i) {
    matrix_values.push_back(static_cast<f_t>(pb_problem.constraint_matrix_values(i)));
  }

  matrix_indices.reserve(pb_problem.constraint_matrix_indices_size());
  for (int i = 0; i < pb_problem.constraint_matrix_indices_size(); ++i) {
    matrix_indices.push_back(static_cast<i_t>(pb_problem.constraint_matrix_indices(i)));
  }

  matrix_offsets.reserve(pb_problem.constraint_matrix_offsets_size());
  for (int i = 0; i < pb_problem.constraint_matrix_offsets_size(); ++i) {
    matrix_offsets.push_back(static_cast<i_t>(pb_problem.constraint_matrix_offsets(i)));
  }

  problem.set_csr_constraint_matrix(matrix_values.data(),
                                    matrix_values.size(),
                                    matrix_indices.data(),
                                    matrix_indices.size(),
                                    matrix_offsets.data(),
                                    matrix_offsets.size());

  // Convert problem vectors
  std::vector<f_t> obj_coeffs;
  std::vector<f_t> constraint_bounds;
  std::vector<f_t> var_lower;
  std::vector<f_t> var_upper;

  obj_coeffs.reserve(pb_problem.objective_coefficients_size());
  for (int i = 0; i < pb_problem.objective_coefficients_size(); ++i) {
    obj_coeffs.push_back(static_cast<f_t>(pb_problem.objective_coefficients(i)));
  }

  constraint_bounds.reserve(pb_problem.constraint_bounds_size());
  for (int i = 0; i < pb_problem.constraint_bounds_size(); ++i) {
    constraint_bounds.push_back(static_cast<f_t>(pb_problem.constraint_bounds(i)));
  }

  var_lower.reserve(pb_problem.variable_lower_bounds_size());
  for (int i = 0; i < pb_problem.variable_lower_bounds_size(); ++i) {
    var_lower.push_back(static_cast<f_t>(pb_problem.variable_lower_bounds(i)));
  }

  var_upper.reserve(pb_problem.variable_upper_bounds_size());
  for (int i = 0; i < pb_problem.variable_upper_bounds_size(); ++i) {
    var_upper.push_back(static_cast<f_t>(pb_problem.variable_upper_bounds(i)));
  }

  problem.set_objective_coefficients(obj_coeffs.data(), obj_coeffs.size());
  problem.set_constraint_bounds(constraint_bounds.data(), constraint_bounds.size());
  problem.set_variable_lower_bounds(var_lower.data(), var_lower.size());
  problem.set_variable_upper_bounds(var_upper.data(), var_upper.size());

  // Constraint lower/upper bounds (if provided)
  if (pb_problem.constraint_lower_bounds_size() > 0) {
    std::vector<f_t> constraint_lower;
    constraint_lower.reserve(pb_problem.constraint_lower_bounds_size());
    for (int i = 0; i < pb_problem.constraint_lower_bounds_size(); ++i) {
      constraint_lower.push_back(static_cast<f_t>(pb_problem.constraint_lower_bounds(i)));
    }
    problem.set_constraint_lower_bounds(constraint_lower.data(), constraint_lower.size());
  }

  if (pb_problem.constraint_upper_bounds_size() > 0) {
    std::vector<f_t> constraint_upper;
    constraint_upper.reserve(pb_problem.constraint_upper_bounds_size());
    for (int i = 0; i < pb_problem.constraint_upper_bounds_size(); ++i) {
      constraint_upper.push_back(static_cast<f_t>(pb_problem.constraint_upper_bounds(i)));
    }
    problem.set_constraint_upper_bounds(constraint_upper.data(), constraint_upper.size());
  }

  // Row types (if provided)
  if (!pb_problem.row_types().empty()) {
    const std::string& rt = pb_problem.row_types();
    problem.set_row_types(rt.data(), rt.size());
  }

  return problem;
}

// Convert LP solution to protobuf
template <typename i_t, typename f_t>
static void lp_solution_to_protobuf(
  cuopt::linear_programming::optimization_problem_solution_t<i_t, f_t>& solution,
  cuopt::remote::LPSolution* pb_solution)
{
  // Solution vectors
  for (const auto& val : solution.get_primal_solution()) {
    pb_solution->add_primal_solution(static_cast<double>(val));
  }
  for (const auto& val : solution.get_dual_solution()) {
    pb_solution->add_dual_solution(static_cast<double>(val));
  }
  for (const auto& val : solution.get_reduced_cost()) {
    pb_solution->add_reduced_cost(static_cast<double>(val));
  }

  // Termination status
  pb_solution->set_termination_status(
    static_cast<cuopt::remote::PDLPTerminationStatus>(solution.get_termination_status()));

  // Solution statistics
  const auto& stats = solution.get_additional_termination_information();
  pb_solution->set_l2_primal_residual(stats.l2_primal_residual);
  pb_solution->set_l2_dual_residual(stats.l2_dual_residual);
  pb_solution->set_primal_objective(stats.primal_objective);
  pb_solution->set_dual_objective(stats.dual_objective);
  pb_solution->set_gap(stats.gap);
  pb_solution->set_nb_iterations(stats.number_of_steps_taken);
  pb_solution->set_solve_time(stats.solve_time);
  pb_solution->set_solved_by_pdlp(stats.solved_by_pdlp);

  // Warm start data
  const auto& ws = solution.get_pdlp_warm_start_data();
  auto* pb_ws    = pb_solution->mutable_warm_start_data();

  for (const auto& val : ws.current_primal_solution_) {
    pb_ws->add_current_primal_solution(static_cast<double>(val));
  }
  for (const auto& val : ws.current_dual_solution_) {
    pb_ws->add_current_dual_solution(static_cast<double>(val));
  }
  for (const auto& val : ws.initial_primal_average_) {
    pb_ws->add_initial_primal_average(static_cast<double>(val));
  }
  for (const auto& val : ws.initial_dual_average_) {
    pb_ws->add_initial_dual_average(static_cast<double>(val));
  }
  for (const auto& val : ws.current_ATY_) {
    pb_ws->add_current_aty(static_cast<double>(val));
  }
  for (const auto& val : ws.sum_primal_solutions_) {
    pb_ws->add_sum_primal_solutions(static_cast<double>(val));
  }
  for (const auto& val : ws.sum_dual_solutions_) {
    pb_ws->add_sum_dual_solutions(static_cast<double>(val));
  }
  for (const auto& val : ws.last_restart_duality_gap_primal_solution_) {
    pb_ws->add_last_restart_duality_gap_primal_solution(static_cast<double>(val));
  }
  for (const auto& val : ws.last_restart_duality_gap_dual_solution_) {
    pb_ws->add_last_restart_duality_gap_dual_solution(static_cast<double>(val));
  }

  pb_ws->set_initial_primal_weight(static_cast<double>(ws.initial_primal_weight_));
  pb_ws->set_initial_step_size(static_cast<double>(ws.initial_step_size_));
  pb_ws->set_total_pdlp_iterations(ws.total_pdlp_iterations_);
  pb_ws->set_total_pdhg_iterations(ws.total_pdhg_iterations_);
  pb_ws->set_last_candidate_kkt_score(static_cast<double>(ws.last_candidate_kkt_score_));
  pb_ws->set_last_restart_kkt_score(static_cast<double>(ws.last_restart_kkt_score_));
  pb_ws->set_sum_solution_weight(static_cast<double>(ws.sum_solution_weight_));
  pb_ws->set_iterations_since_last_restart(ws.iterations_since_last_restart_);
}

// Handle a single client connection
void handle_client(int client_socket)
{
  try {
    std::cout << "[Server] Client connected\n";

    // Read request size
    uint32_t request_size;
    read_all(client_socket, &request_size, sizeof(request_size));

    std::cout << "[Server] Receiving request (" << request_size << " bytes)...\n";

    // Read request data
    std::vector<uint8_t> request_data(request_size);
    read_all(client_socket, request_data.data(), request_size);

    // Parse request
    cuopt::remote::SolveLPRequest lp_request;
    cuopt::remote::SolveMIPRequest mip_request;

    // Try LP first
    bool is_lp  = lp_request.ParseFromArray(request_data.data(), request_size);
    bool is_mip = false;

    if (!is_lp) {
      // Try MIP
      is_mip = mip_request.ParseFromArray(request_data.data(), request_size);
    }

    if (!is_lp && !is_mip) { throw std::runtime_error("Failed to parse request as LP or MIP"); }

    // Create response
    cuopt::remote::SolveResponse response;
    response.set_status(cuopt::remote::SUCCESS);

    if (is_lp) {
      std::cout << "[Server] Processing LP request\n";
      std::cout << "  Version: " << lp_request.header().version() << "\n";
      std::cout << "  Variables: " << lp_request.problem().objective_coefficients_size() << "\n";
      std::cout << "  Constraints: " << (lp_request.problem().constraint_matrix_offsets_size() - 1)
                << "\n";

      // Convert problem
      auto problem = protobuf_to_problem<int, double>(lp_request.problem());

      // Solve LP
      std::cout << "[Server] Solving LP problem...\n";
      cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings;
      auto solution = cuopt::linear_programming::solve_lp(problem, settings);

      std::cout << "[Server] LP solve completed\n";
      std::cout << "  Status: " << static_cast<int>(solution.get_termination_status()) << "\n";
      std::cout << "  Objective: " << solution.get_objective_value() << "\n";

      // Convert solution to protobuf
      lp_solution_to_protobuf(solution, response.mutable_lp_solution());

    } else if (is_mip) {
      std::cout << "[Server] Processing MIP request\n";

      // MIP not yet implemented with protobuf
      response.set_status(cuopt::remote::ERROR_INTERNAL);
      response.set_error_message("MIP solving not yet implemented with protobuf");
    }

    // Serialize response
    std::string response_data = response.SerializeAsString();
    uint32_t response_size    = static_cast<uint32_t>(response_data.size());

    std::cout << "[Server] Sending response (" << response_size << " bytes)...\n";

    // Send response size and data
    write_all(client_socket, &response_size, sizeof(response_size));
    write_all(client_socket, response_data.data(), response_data.size());

    std::cout << "[Server] Response sent successfully\n";

  } catch (const std::exception& e) {
    std::cerr << "[Server] Error handling client: " << e.what() << "\n";

    // Try to send error response
    try {
      cuopt::remote::SolveResponse response;
      response.set_status(cuopt::remote::ERROR_INTERNAL);
      response.set_error_message(e.what());

      std::string response_data = response.SerializeAsString();
      uint32_t response_size    = static_cast<uint32_t>(response_data.size());

      write_all(client_socket, &response_size, sizeof(response_size));
      write_all(client_socket, response_data.data(), response_data.size());
    } catch (...) {
      // Ignore write errors during error handling
    }
  }

  close(client_socket);
  std::cout << "[Server] Client disconnected\n\n";
}

int main(int argc, char* argv[])
{
  // Verify protobuf version
  GOOGLE_PROTOBUF_VERIFY_VERSION;

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
  std::cout << "cuOpt Remote Solve Server (Protocol Buffers)\n";
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

  // Cleanup protobuf
  google::protobuf::ShutdownProtobufLibrary();

  return 0;
}
