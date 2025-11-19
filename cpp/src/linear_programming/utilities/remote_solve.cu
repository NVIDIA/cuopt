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

#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/utilities/remote_solve.hpp>
#include <mip/mip_constants.hpp>

#include <cuopt_remote.pb.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

namespace cuopt::linear_programming {

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

// Convert optimization_problem_t to protobuf message
template <typename i_t, typename f_t>
static void problem_to_protobuf(const optimization_problem_t<i_t, f_t>& problem,
                                cuopt::remote::OptimizationProblem* pb_problem)
{
  // Problem metadata
  pb_problem->set_maximize(problem.get_sense());
  pb_problem->set_objective_scaling_factor(problem.get_objective_scaling_factor());
  pb_problem->set_objective_offset(problem.get_objective_offset());

  // Constraint matrix (CSR format)
  const auto& matrix_values  = problem.get_constraint_matrix_values();
  const auto& matrix_indices = problem.get_constraint_matrix_indices();
  const auto& matrix_offsets = problem.get_constraint_matrix_offsets();

  for (const auto& val : matrix_values) {
    pb_problem->add_constraint_matrix_values(static_cast<double>(val));
  }
  for (const auto& idx : matrix_indices) {
    pb_problem->add_constraint_matrix_indices(static_cast<int32_t>(idx));
  }
  for (const auto& offset : matrix_offsets) {
    pb_problem->add_constraint_matrix_offsets(static_cast<int32_t>(offset));
  }

  // Problem vectors
  const auto& obj_coeffs        = problem.get_objective_coefficients();
  const auto& constraint_bounds = problem.get_constraint_bounds();
  const auto& var_lower         = problem.get_variable_lower_bounds();
  const auto& var_upper         = problem.get_variable_upper_bounds();

  for (const auto& val : obj_coeffs) {
    pb_problem->add_objective_coefficients(static_cast<double>(val));
  }
  for (const auto& val : constraint_bounds) {
    pb_problem->add_constraint_bounds(static_cast<double>(val));
  }
  for (const auto& val : var_lower) {
    pb_problem->add_variable_lower_bounds(static_cast<double>(val));
  }
  for (const auto& val : var_upper) {
    pb_problem->add_variable_upper_bounds(static_cast<double>(val));
  }

  // Constraint lower/upper bounds (additional representation)
  const auto& constraint_lower = problem.get_constraint_lower_bounds();
  const auto& constraint_upper = problem.get_constraint_upper_bounds();
  for (const auto& val : constraint_lower) {
    pb_problem->add_constraint_lower_bounds(static_cast<double>(val));
  }
  for (const auto& val : constraint_upper) {
    pb_problem->add_constraint_upper_bounds(static_cast<double>(val));
  }

  // Row types (constraint types: '<', '>', '=')
  const auto& row_types = problem.get_row_types();
  if (!row_types.empty()) { pb_problem->set_row_types(row_types.data(), row_types.size()); }

  // Variable types (for MIP: CONTINUOUS or INTEGER)
  const auto& var_types = problem.get_variable_types();
  if (!var_types.empty()) {
    for (const auto& vt : var_types) {
      bool is_int = (vt == cuopt::linear_programming::var_t::INTEGER);
      pb_problem->add_is_integer(is_int);
      pb_problem->add_is_binary(false);  // cuOpt uses INTEGER for both
    }
  }
}

// Convert protobuf message to optimization_problem_t
template <typename i_t, typename f_t>
static optimization_problem_t<i_t, f_t> protobuf_to_problem(
  const cuopt::remote::OptimizationProblem& pb_problem)
{
  optimization_problem_t<i_t, f_t> problem;

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

// Convert PDLP warm start data to protobuf
template <typename i_t, typename f_t>
static void warm_start_to_protobuf(const pdlp_warm_start_data_t<i_t, f_t>& ws,
                                   cuopt::remote::PDLPWarmStartData* pb_ws)
{
  // Convert vectors
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

  // Convert scalars
  pb_ws->set_initial_primal_weight(static_cast<double>(ws.initial_primal_weight_));
  pb_ws->set_initial_step_size(static_cast<double>(ws.initial_step_size_));
  pb_ws->set_total_pdlp_iterations(ws.total_pdlp_iterations_);
  pb_ws->set_total_pdhg_iterations(ws.total_pdhg_iterations_);
  pb_ws->set_last_candidate_kkt_score(static_cast<double>(ws.last_candidate_kkt_score_));
  pb_ws->set_last_restart_kkt_score(static_cast<double>(ws.last_restart_kkt_score_));
  pb_ws->set_sum_solution_weight(static_cast<double>(ws.sum_solution_weight_));
  pb_ws->set_iterations_since_last_restart(ws.iterations_since_last_restart_);
}

// Convert protobuf to PDLP warm start data
template <typename i_t, typename f_t>
static pdlp_warm_start_data_t<i_t, f_t> protobuf_to_warm_start(
  const cuopt::remote::PDLPWarmStartData& pb_ws)
{
  pdlp_warm_start_data_t<i_t, f_t> ws;

  // Convert vectors
  ws.current_primal_solution_.reserve(pb_ws.current_primal_solution_size());
  for (int i = 0; i < pb_ws.current_primal_solution_size(); ++i) {
    ws.current_primal_solution_.push_back(static_cast<f_t>(pb_ws.current_primal_solution(i)));
  }

  ws.current_dual_solution_.reserve(pb_ws.current_dual_solution_size());
  for (int i = 0; i < pb_ws.current_dual_solution_size(); ++i) {
    ws.current_dual_solution_.push_back(static_cast<f_t>(pb_ws.current_dual_solution(i)));
  }

  ws.initial_primal_average_.reserve(pb_ws.initial_primal_average_size());
  for (int i = 0; i < pb_ws.initial_primal_average_size(); ++i) {
    ws.initial_primal_average_.push_back(static_cast<f_t>(pb_ws.initial_primal_average(i)));
  }

  ws.initial_dual_average_.reserve(pb_ws.initial_dual_average_size());
  for (int i = 0; i < pb_ws.initial_dual_average_size(); ++i) {
    ws.initial_dual_average_.push_back(static_cast<f_t>(pb_ws.initial_dual_average(i)));
  }

  ws.current_ATY_.reserve(pb_ws.current_aty_size());
  for (int i = 0; i < pb_ws.current_aty_size(); ++i) {
    ws.current_ATY_.push_back(static_cast<f_t>(pb_ws.current_aty(i)));
  }

  ws.sum_primal_solutions_.reserve(pb_ws.sum_primal_solutions_size());
  for (int i = 0; i < pb_ws.sum_primal_solutions_size(); ++i) {
    ws.sum_primal_solutions_.push_back(static_cast<f_t>(pb_ws.sum_primal_solutions(i)));
  }

  ws.sum_dual_solutions_.reserve(pb_ws.sum_dual_solutions_size());
  for (int i = 0; i < pb_ws.sum_dual_solutions_size(); ++i) {
    ws.sum_dual_solutions_.push_back(static_cast<f_t>(pb_ws.sum_dual_solutions(i)));
  }

  ws.last_restart_duality_gap_primal_solution_.reserve(
    pb_ws.last_restart_duality_gap_primal_solution_size());
  for (int i = 0; i < pb_ws.last_restart_duality_gap_primal_solution_size(); ++i) {
    ws.last_restart_duality_gap_primal_solution_.push_back(
      static_cast<f_t>(pb_ws.last_restart_duality_gap_primal_solution(i)));
  }

  ws.last_restart_duality_gap_dual_solution_.reserve(
    pb_ws.last_restart_duality_gap_dual_solution_size());
  for (int i = 0; i < pb_ws.last_restart_duality_gap_dual_solution_size(); ++i) {
    ws.last_restart_duality_gap_dual_solution_.push_back(
      static_cast<f_t>(pb_ws.last_restart_duality_gap_dual_solution(i)));
  }

  // Convert scalars
  ws.initial_primal_weight_         = static_cast<f_t>(pb_ws.initial_primal_weight());
  ws.initial_step_size_             = static_cast<f_t>(pb_ws.initial_step_size());
  ws.total_pdlp_iterations_         = pb_ws.total_pdlp_iterations();
  ws.total_pdhg_iterations_         = pb_ws.total_pdhg_iterations();
  ws.last_candidate_kkt_score_      = static_cast<f_t>(pb_ws.last_candidate_kkt_score());
  ws.last_restart_kkt_score_        = static_cast<f_t>(pb_ws.last_restart_kkt_score());
  ws.sum_solution_weight_           = static_cast<f_t>(pb_ws.sum_solution_weight());
  ws.iterations_since_last_restart_ = pb_ws.iterations_since_last_restart();

  return ws;
}

// Convert LP solution to protobuf
template <typename i_t, typename f_t>
static void lp_solution_to_protobuf(optimization_problem_solution_t<i_t, f_t>& solution,
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

  // Warm start data
  const auto& ws = solution.get_pdlp_warm_start_data();
  warm_start_to_protobuf<i_t, f_t>(ws, pb_solution->mutable_warm_start_data());

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
}

// Convert protobuf to LP solution
template <typename i_t, typename f_t>
static optimization_problem_solution_t<i_t, f_t> protobuf_to_lp_solution(
  const cuopt::remote::LPSolution& pb_solution)
{
  // Convert solution vectors
  std::vector<f_t> primal_solution;
  std::vector<f_t> dual_solution;
  std::vector<f_t> reduced_cost;

  primal_solution.reserve(pb_solution.primal_solution_size());
  for (int i = 0; i < pb_solution.primal_solution_size(); ++i) {
    primal_solution.push_back(static_cast<f_t>(pb_solution.primal_solution(i)));
  }

  dual_solution.reserve(pb_solution.dual_solution_size());
  for (int i = 0; i < pb_solution.dual_solution_size(); ++i) {
    dual_solution.push_back(static_cast<f_t>(pb_solution.dual_solution(i)));
  }

  reduced_cost.reserve(pb_solution.reduced_cost_size());
  for (int i = 0; i < pb_solution.reduced_cost_size(); ++i) {
    reduced_cost.push_back(static_cast<f_t>(pb_solution.reduced_cost(i)));
  }

  // Convert warm start data
  pdlp_warm_start_data_t<i_t, f_t> warm_start_data;
  if (pb_solution.has_warm_start_data()) {
    warm_start_data = protobuf_to_warm_start<i_t, f_t>(pb_solution.warm_start_data());
  }

  // Convert solution statistics
  typename optimization_problem_solution_t<i_t, f_t>::additional_termination_information_t stats{};
  stats.l2_primal_residual    = pb_solution.l2_primal_residual();
  stats.l2_dual_residual      = pb_solution.l2_dual_residual();
  stats.primal_objective      = pb_solution.primal_objective();
  stats.dual_objective        = pb_solution.dual_objective();
  stats.gap                   = pb_solution.gap();
  stats.number_of_steps_taken = pb_solution.nb_iterations();
  stats.solve_time            = pb_solution.solve_time();
  stats.solved_by_pdlp        = pb_solution.solved_by_pdlp();

  // Create solution
  return optimization_problem_solution_t<i_t, f_t>(
    std::move(primal_solution),
    std::move(dual_solution),
    std::move(reduced_cost),
    std::move(warm_start_data),
    "",                          // objective_name
    std::vector<std::string>(),  // var_names
    std::vector<std::string>(),  // row_names
    stats,
    static_cast<pdlp_termination_status_t>(pb_solution.termination_status()));
}

// Convert protobuf to MIP solution
template <typename i_t, typename f_t>
static mip_solution_t<i_t, f_t> protobuf_to_mip_solution(
  const cuopt::remote::MIPSolution& pb_solution)
{
  // Convert solution vector
  std::vector<f_t> solution;
  solution.reserve(pb_solution.solution_size());
  for (int i = 0; i < pb_solution.solution_size(); ++i) {
    solution.push_back(static_cast<f_t>(pb_solution.solution(i)));
  }

  // Convert solver stats
  solver_stats_t<i_t, f_t> stats;
  stats.total_solve_time       = static_cast<f_t>(pb_solution.total_solve_time());
  stats.presolve_time          = static_cast<f_t>(pb_solution.presolve_time());
  stats.solution_bound         = static_cast<f_t>(pb_solution.solution_bound());
  stats.num_nodes              = static_cast<i_t>(pb_solution.nodes());
  stats.num_simplex_iterations = static_cast<i_t>(pb_solution.simplex_iterations());

  // Create solution
  auto mip_sol = mip_solution_t<i_t, f_t>(
    std::move(solution),
    std::vector<std::string>(),  // var_names
    static_cast<f_t>(pb_solution.objective()),
    static_cast<f_t>(pb_solution.mip_gap()),
    static_cast<mip_termination_status_t>(pb_solution.termination_status()),
    static_cast<f_t>(pb_solution.max_constraint_violation()),
    static_cast<f_t>(pb_solution.max_int_violation()),
    static_cast<f_t>(pb_solution.max_variable_bound_violation()),
    stats);

  // Print solution stats using shared method
  mip_sol.print_solution_stats();

  return mip_sol;
}

// Check if remote solve is enabled via environment variables
bool is_remote_solve_enabled(const char** host, const char** port)
{
  *host = std::getenv("CUOPT_REMOTE_HOST");
  *port = std::getenv("CUOPT_REMOTE_PORT");
  return (*host != nullptr && *port != nullptr);
}

// Check if sync mode is enabled (default is async)
static bool use_sync_mode()
{
  const char* sync_env = std::getenv("CUOPT_REMOTE_USE_SYNC");
  return (sync_env != nullptr && std::string(sync_env) == "1");
}

// Helper: Create and connect socket
static int connect_to_server(const char* host, const char* port)
{
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) { throw std::runtime_error("Failed to create socket"); }

  struct hostent* server = gethostbyname(host);
  if (server == nullptr) {
    close(sockfd);
    throw std::runtime_error("Failed to resolve hostname");
  }

  struct sockaddr_in serv_addr;
  std::memset(&serv_addr, 0, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  std::memcpy(&serv_addr.sin_addr.s_addr, server->h_addr, server->h_length);
  serv_addr.sin_port = htons(std::atoi(port));

  if (connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
    close(sockfd);
    throw std::runtime_error("Failed to connect to remote server");
  }

  return sockfd;
}

// Helper: Submit a job to the async server
static std::string submit_job(const char* host,
                              const char* port,
                              const cuopt::remote::AsyncRequest& request)
{
  int sockfd = connect_to_server(host, port);

  try {
    // Serialize and send request
    std::string request_data = request.SerializeAsString();
    uint32_t request_size    = static_cast<uint32_t>(request_data.size());

    write_all(sockfd, &request_size, sizeof(request_size));
    write_all(sockfd, request_data.data(), request_data.size());

    // Read response
    uint32_t response_size;
    read_all(sockfd, &response_size, sizeof(response_size));

    std::vector<uint8_t> response_data(response_size);
    read_all(sockfd, response_data.data(), response_size);

    close(sockfd);

    // Parse response
    cuopt::remote::AsyncResponse async_response;
    if (!async_response.ParseFromArray(response_data.data(), response_size)) {
      throw std::runtime_error("Failed to parse AsyncResponse");
    }

    if (!async_response.has_submit_response()) {
      throw std::runtime_error("AsyncResponse does not contain submit_response");
    }

    const auto& submit_resp = async_response.submit_response();
    if (submit_resp.status() != cuopt::remote::SUCCESS) {
      throw std::runtime_error("Job submission failed: " + submit_resp.message());
    }

    return submit_resp.job_id();
  } catch (...) {
    close(sockfd);
    throw;
  }
}

// Helper: Poll job status until complete
static void poll_until_complete(const char* host, const char* port, const std::string& job_id)
{
  while (true) {
    int sockfd = connect_to_server(host, port);

    try {
      // Create status request
      cuopt::remote::AsyncRequest status_request;
      status_request.set_request_type(cuopt::remote::CHECK_STATUS);
      status_request.set_job_id(job_id);

      // Send request
      std::string request_data = status_request.SerializeAsString();
      uint32_t request_size    = static_cast<uint32_t>(request_data.size());

      write_all(sockfd, &request_size, sizeof(request_size));
      write_all(sockfd, request_data.data(), request_data.size());

      // Read response
      uint32_t response_size;
      read_all(sockfd, &response_size, sizeof(response_size));

      std::vector<uint8_t> response_data(response_size);
      read_all(sockfd, response_data.data(), response_size);

      close(sockfd);

      // Parse response
      cuopt::remote::AsyncResponse async_response;
      if (!async_response.ParseFromArray(response_data.data(), response_size)) {
        throw std::runtime_error("Failed to parse status response");
      }

      if (!async_response.has_status_response()) {
        throw std::runtime_error("AsyncResponse does not contain status_response");
      }

      const auto& status_resp = async_response.status_response();

      if (status_resp.job_status() == cuopt::remote::COMPLETED) {
        return;  // Job is done
      } else if (status_resp.job_status() == cuopt::remote::FAILED) {
        throw std::runtime_error("Job failed: " + status_resp.message());
      }

      // Job still pending/running, wait a bit before polling again
      usleep(100000);  // 100ms
    } catch (...) {
      close(sockfd);
      throw;
    }
  }
}

// Helper: Get result from completed job
static cuopt::remote::ResultResponse get_result(const char* host,
                                                const char* port,
                                                const std::string& job_id)
{
  int sockfd = connect_to_server(host, port);

  try {
    // Create get result request
    cuopt::remote::AsyncRequest result_request;
    result_request.set_request_type(cuopt::remote::GET_RESULT);
    result_request.set_job_id(job_id);

    // Send request
    std::string request_data = result_request.SerializeAsString();
    uint32_t request_size    = static_cast<uint32_t>(request_data.size());

    write_all(sockfd, &request_size, sizeof(request_size));
    write_all(sockfd, request_data.data(), request_data.size());

    // Read response
    uint32_t response_size;
    read_all(sockfd, &response_size, sizeof(response_size));

    std::vector<uint8_t> response_data(response_size);
    read_all(sockfd, response_data.data(), response_size);

    close(sockfd);

    // Parse response
    cuopt::remote::AsyncResponse async_response;
    if (!async_response.ParseFromArray(response_data.data(), response_size)) {
      throw std::runtime_error("Failed to parse result response");
    }

    if (!async_response.has_result_response()) {
      throw std::runtime_error("AsyncResponse does not contain result_response");
    }

    return async_response.result_response();
  } catch (...) {
    close(sockfd);
    throw;
  }
}

// Helper: Delete job after retrieving result
static void delete_job(const char* host, const char* port, const std::string& job_id)
{
  int sockfd = connect_to_server(host, port);

  try {
    // Create delete request
    cuopt::remote::AsyncRequest delete_request;
    delete_request.set_request_type(cuopt::remote::DELETE_RESULT);
    delete_request.set_job_id(job_id);

    // Send request
    std::string request_data = delete_request.SerializeAsString();
    uint32_t request_size    = static_cast<uint32_t>(request_data.size());

    write_all(sockfd, &request_size, sizeof(request_size));
    write_all(sockfd, request_data.data(), request_data.size());

    // Read response (but don't need to check it)
    uint32_t response_size;
    read_all(sockfd, &response_size, sizeof(response_size));

    std::vector<uint8_t> response_data(response_size);
    read_all(sockfd, response_data.data(), response_size);

    close(sockfd);
  } catch (...) {
    close(sockfd);
    throw;
  }
}

// Helper: Submit job in sync mode (blocking) and get result directly
static cuopt::remote::ResultResponse submit_job_sync(const char* host,
                                                     const char* port,
                                                     const cuopt::remote::AsyncRequest& request)
{
  int sockfd = connect_to_server(host, port);

  try {
    // Serialize and send request
    std::string request_data = request.SerializeAsString();
    uint32_t request_size    = static_cast<uint32_t>(request_data.size());

    write_all(sockfd, &request_size, sizeof(request_size));
    write_all(sockfd, request_data.data(), request_data.size());

    // Read response
    uint32_t response_size;
    read_all(sockfd, &response_size, sizeof(response_size));

    std::vector<uint8_t> response_data(response_size);
    read_all(sockfd, response_data.data(), response_size);

    close(sockfd);

    // Parse response
    cuopt::remote::AsyncResponse async_response;
    if (!async_response.ParseFromArray(response_data.data(), response_size)) {
      throw std::runtime_error("Failed to parse AsyncResponse");
    }

    if (!async_response.has_result_response()) {
      throw std::runtime_error("AsyncResponse does not contain result_response (sync mode)");
    }

    return async_response.result_response();
  } catch (...) {
    close(sockfd);
    throw;
  }
}

// Solve LP problem remotely using async server (sync or async mode)
template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp_remote(
  const optimization_problem_t<i_t, f_t>& problem, const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  // Get remote host and port
  const char* host;
  const char* port;
  if (!is_remote_solve_enabled(&host, &port)) {
    throw std::runtime_error("Remote solve not enabled (CUOPT_REMOTE_HOST/PORT not set)");
  }

  // Check if sync or async mode
  const bool sync_mode = use_sync_mode();
  fprintf(stderr,
          "[solve_lp_remote] Connecting to %s:%s (async server, %s mode)\n",
          host,
          port,
          sync_mode ? "sync" : "async");

  try {
    // Create AsyncRequest
    cuopt::remote::AsyncRequest request;
    request.set_request_type(cuopt::remote::SUBMIT_JOB);
    request.set_blocking(sync_mode);

    // Add LP problem data
    auto* lp_request = request.mutable_lp_request();

    // Set header
    auto* header = lp_request->mutable_header();
    header->set_version(1);
    header->set_problem_type(cuopt::remote::LP);
    header->set_index_type(sizeof(i_t) == 4 ? cuopt::remote::INT32 : cuopt::remote::INT64);
    header->set_float_type(sizeof(f_t) == 4 ? cuopt::remote::FLOAT32 : cuopt::remote::DOUBLE);

    // Convert problem to protobuf
    problem_to_protobuf(problem, lp_request->mutable_problem());

    cuopt::remote::ResultResponse result_response;

    if (sync_mode) {
      // ===== SYNC MODE: Submit with blocking=true, get result directly =====
      fprintf(stderr, "[solve_lp_remote] Sending blocking request, waiting for solution...\n");
      result_response = submit_job_sync(host, port, request);
      fprintf(stderr, "[solve_lp_remote] Solution received from sync request\n");

    } else {
      // ===== ASYNC MODE: Submit → Poll → Get Result → Delete =====
      std::string job_id = submit_job(host, port, request);
      fprintf(stderr, "[solve_lp_remote] Job submitted, ID: %s\n", job_id.c_str());

      // Poll until job completes
      fprintf(stderr, "[solve_lp_remote] Polling for completion...\n");
      poll_until_complete(host, port, job_id);

      // Get result
      fprintf(stderr, "[solve_lp_remote] Job complete, retrieving result...\n");
      result_response = get_result(host, port, job_id);

      // Delete job
      delete_job(host, port, job_id);
      fprintf(stderr, "[solve_lp_remote] Job deleted from server\n");
    }

    // Check result status
    if (result_response.status() != cuopt::remote::SUCCESS) {
      throw std::runtime_error("Remote solve failed: " + result_response.error_message());
    }

    if (!result_response.has_lp_solution()) {
      throw std::runtime_error("ResultResponse does not contain LP solution");
    }

    fprintf(stderr, "[solve_lp_remote] Solution received successfully\n");

    // Convert protobuf solution to C++ solution
    auto lp_sol = protobuf_to_lp_solution<i_t, f_t>(result_response.lp_solution());

    // Print solution stats using shared method
    lp_sol.print_solution_stats();

    return lp_sol;

  } catch (...) {
    throw;
  }
}

// Solve MIP problem remotely using async server (sync or async mode)
template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip_remote(const optimization_problem_t<i_t, f_t>& problem,
                                          const mip_solver_settings_t<i_t, f_t>& settings)
{
  // Get remote host and port
  const char* host;
  const char* port;
  if (!is_remote_solve_enabled(&host, &port)) {
    throw std::runtime_error("Remote solve not enabled (CUOPT_REMOTE_HOST/PORT not set)");
  }

  // Check if sync or async mode
  const bool sync_mode = use_sync_mode();
  fprintf(stderr,
          "[solve_mip_remote] Connecting to %s:%s (async server, %s mode)\n",
          host,
          port,
          sync_mode ? "sync" : "async");

  try {
    // Create AsyncRequest
    cuopt::remote::AsyncRequest request;
    request.set_request_type(cuopt::remote::SUBMIT_JOB);
    request.set_blocking(sync_mode);

    // Add MIP problem data
    auto* mip_request = request.mutable_mip_request();

    // Set header
    auto* header = mip_request->mutable_header();
    header->set_version(1);
    header->set_problem_type(cuopt::remote::MIP);
    header->set_index_type(sizeof(i_t) == 4 ? cuopt::remote::INT32 : cuopt::remote::INT64);
    header->set_float_type(sizeof(f_t) == 4 ? cuopt::remote::FLOAT32 : cuopt::remote::DOUBLE);

    // Convert problem to protobuf
    problem_to_protobuf(problem, mip_request->mutable_problem());

    cuopt::remote::ResultResponse result_response;

    if (sync_mode) {
      // ===== SYNC MODE: Submit with blocking=true, get result directly =====
      fprintf(stderr, "[solve_mip_remote] Sending blocking request, waiting for solution...\n");
      result_response = submit_job_sync(host, port, request);
      fprintf(stderr, "[solve_mip_remote] Solution received from sync request\n");

    } else {
      // ===== ASYNC MODE: Submit → Poll → Get Result → Delete =====
      std::string job_id = submit_job(host, port, request);
      fprintf(stderr, "[solve_mip_remote] Job submitted, ID: %s\n", job_id.c_str());

      // Poll until job completes
      fprintf(stderr, "[solve_mip_remote] Polling for completion...\n");
      poll_until_complete(host, port, job_id);

      // Get result
      fprintf(stderr, "[solve_mip_remote] Job complete, retrieving result...\n");
      result_response = get_result(host, port, job_id);

      // Delete job
      delete_job(host, port, job_id);
      fprintf(stderr, "[solve_mip_remote] Job deleted from server\n");
    }

    // Check result status
    if (result_response.status() != cuopt::remote::SUCCESS) {
      throw std::runtime_error("Remote solve failed: " + result_response.error_message());
    }

    if (!result_response.has_mip_solution()) {
      throw std::runtime_error("ResultResponse does not contain MIP solution");
    }

    fprintf(stderr, "[solve_mip_remote] Solution received successfully\n");

    // Convert protobuf solution to C++ solution (printing happens inside)
    return protobuf_to_mip_solution<i_t, f_t>(result_response.mip_solution());

  } catch (...) {
    throw;
  }
}

// Explicit template instantiations for double precision
#if MIP_INSTANTIATE_DOUBLE
template optimization_problem_solution_t<int, double> solve_lp_remote(
  const optimization_problem_t<int, double>&, const pdlp_solver_settings_t<int, double>&);

template mip_solution_t<int, double> solve_mip_remote(const optimization_problem_t<int, double>&,
                                                      const mip_solver_settings_t<int, double>&);
#endif

// Explicit template instantiations for float precision (if enabled)
#if MIP_INSTANTIATE_FLOAT
template optimization_problem_solution_t<int, float> solve_lp_remote(
  const optimization_problem_t<int, float>&, const pdlp_solver_settings_t<int, float>&);

template mip_solution_t<int, float> solve_mip_remote(const optimization_problem_t<int, float>&,
                                                     const mip_solver_settings_t<int, float>&);
#endif

}  // namespace cuopt::linear_programming
