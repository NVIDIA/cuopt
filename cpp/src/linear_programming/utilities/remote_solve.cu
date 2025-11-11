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

// Helper to write vector to buffer
template <typename T>
static void write_vector(std::vector<uint8_t>& buffer, const std::vector<T>& vec)
{
  uint64_t size = vec.size();
  size_t start  = buffer.size();
  buffer.resize(start + sizeof(uint64_t) + size * sizeof(T));

  std::memcpy(buffer.data() + start, &size, sizeof(uint64_t));
  if (size > 0) {
    std::memcpy(buffer.data() + start + sizeof(uint64_t), vec.data(), size * sizeof(T));
  }
}

// Helper to read vector from buffer
template <typename T>
static std::vector<T> read_vector(const uint8_t*& ptr)
{
  uint64_t size;
  std::memcpy(&size, ptr, sizeof(uint64_t));
  ptr += sizeof(uint64_t);

  std::vector<T> vec(size);
  if (size > 0) {
    std::memcpy(vec.data(), ptr, size * sizeof(T));
    ptr += size * sizeof(T);
  }
  return vec;
}

// Helper to write scalar to buffer
template <typename T>
static void write_to_buffer(std::vector<uint8_t>& buffer, const T& value)
{
  size_t start = buffer.size();
  buffer.resize(start + sizeof(T));
  std::memcpy(buffer.data() + start, &value, sizeof(T));
}

// Helper to read scalar from buffer with offset tracking
template <typename T>
static void read_from_buffer(const std::vector<uint8_t>& buffer, size_t& offset, T& value)
{
  if (offset + sizeof(T) > buffer.size()) {
    throw std::runtime_error("Buffer underrun during scalar deserialization");
  }
  std::memcpy(&value, buffer.data() + offset, sizeof(T));
  offset += sizeof(T);
}

// Helper to write vector with offset-based reading
template <typename T>
static void write_vector_to_buffer(std::vector<uint8_t>& buffer, const std::vector<T>& vec)
{
  uint64_t size = vec.size();
  write_to_buffer(buffer, size);
  if (size > 0) {
    size_t start = buffer.size();
    buffer.resize(start + size * sizeof(T));
    std::memcpy(buffer.data() + start, vec.data(), size * sizeof(T));
  }
}

// Helper to read vector with offset tracking
template <typename T>
static void read_vector_from_buffer(const std::vector<uint8_t>& buffer,
                                    size_t& offset,
                                    std::vector<T>& vec)
{
  uint64_t size;
  read_from_buffer(buffer, offset, size);
  vec.resize(size);
  if (size > 0) {
    if (offset + size * sizeof(T) > buffer.size()) {
      throw std::runtime_error("Buffer underrun during vector deserialization");
    }
    std::memcpy(vec.data(), buffer.data() + offset, size * sizeof(T));
    offset += size * sizeof(T);
  }
}

// Helper to write string to buffer
static void write_string(std::vector<uint8_t>& buffer, const std::string& str)
{
  uint64_t size = str.size();
  size_t start  = buffer.size();
  buffer.resize(start + sizeof(uint64_t) + size);

  std::memcpy(buffer.data() + start, &size, sizeof(uint64_t));
  if (size > 0) { std::memcpy(buffer.data() + start + sizeof(uint64_t), str.data(), size); }
}

// Helper to read string from buffer
static std::string read_string(const uint8_t*& ptr)
{
  uint64_t size;
  std::memcpy(&size, ptr, sizeof(uint64_t));
  ptr += sizeof(uint64_t);

  std::string str(size, '\0');
  if (size > 0) {
    std::memcpy(&str[0], ptr, size);
    ptr += size;
  }
  return str;
}

template <typename i_t, typename f_t>
std::vector<uint8_t> serialize_problem(const optimization_problem_t<i_t, f_t>& problem)
{
  std::vector<uint8_t> buffer;

  // Write all the vectors
  write_vector(buffer, problem.get_constraint_matrix_values());
  write_vector(buffer, problem.get_constraint_matrix_indices());
  write_vector(buffer, problem.get_constraint_matrix_offsets());
  write_vector(buffer, problem.get_objective_coefficients());
  write_vector(buffer, problem.get_variable_lower_bounds());
  write_vector(buffer, problem.get_variable_upper_bounds());
  write_vector(buffer, problem.get_constraint_lower_bounds());
  write_vector(buffer, problem.get_constraint_upper_bounds());
  write_vector(buffer, problem.get_constraint_bounds());
  write_vector(buffer, problem.get_row_types());
  write_vector(buffer, problem.get_variable_types());

  // Write scalars
  size_t start = buffer.size();
  buffer.resize(start + sizeof(bool) + sizeof(f_t) * 2);
  bool maximize = problem.get_sense();
  std::memcpy(buffer.data() + start, &maximize, sizeof(bool));
  start += sizeof(bool);

  f_t obj_scale = problem.get_objective_scaling_factor();
  std::memcpy(buffer.data() + start, &obj_scale, sizeof(f_t));
  start += sizeof(f_t);

  f_t obj_offset = problem.get_objective_offset();
  std::memcpy(buffer.data() + start, &obj_offset, sizeof(f_t));

  // Write strings
  write_string(buffer, problem.get_objective_name());
  write_string(buffer, problem.get_problem_name());

  // Write variable names
  const auto& var_names = problem.get_variable_names();
  uint64_t n_var_names  = var_names.size();
  size_t string_start   = buffer.size();
  buffer.resize(string_start + sizeof(uint64_t));
  std::memcpy(buffer.data() + string_start, &n_var_names, sizeof(uint64_t));
  for (const auto& name : var_names) {
    write_string(buffer, name);
  }

  // Write row names
  const auto& row_names = problem.get_row_names();
  uint64_t n_row_names  = row_names.size();
  string_start          = buffer.size();
  buffer.resize(string_start + sizeof(uint64_t));
  std::memcpy(buffer.data() + string_start, &n_row_names, sizeof(uint64_t));
  for (const auto& name : row_names) {
    write_string(buffer, name);
  }

  return buffer;
}

template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t> deserialize_problem(const std::vector<uint8_t>& buffer)
{
  const uint8_t* ptr = buffer.data();

  optimization_problem_t<i_t, f_t> problem;

  // Read all the vectors
  auto constraint_matrix_values  = read_vector<f_t>(ptr);
  auto constraint_matrix_indices = read_vector<i_t>(ptr);
  auto constraint_matrix_offsets = read_vector<i_t>(ptr);
  auto objective_coefficients    = read_vector<f_t>(ptr);
  auto variable_lower_bounds     = read_vector<f_t>(ptr);
  auto variable_upper_bounds     = read_vector<f_t>(ptr);
  auto constraint_lower_bounds   = read_vector<f_t>(ptr);
  auto constraint_upper_bounds   = read_vector<f_t>(ptr);
  auto constraint_bounds         = read_vector<f_t>(ptr);
  auto row_types                 = read_vector<char>(ptr);
  auto variable_types            = read_vector<var_t>(ptr);

  // Read scalars
  bool maximize;
  std::memcpy(&maximize, ptr, sizeof(bool));
  ptr += sizeof(bool);

  f_t obj_scale;
  std::memcpy(&obj_scale, ptr, sizeof(f_t));
  ptr += sizeof(f_t);

  f_t obj_offset;
  std::memcpy(&obj_offset, ptr, sizeof(f_t));
  ptr += sizeof(f_t);

  // Read strings
  std::string obj_name     = read_string(ptr);
  std::string problem_name = read_string(ptr);

  // Read variable names
  uint64_t n_var_names;
  std::memcpy(&n_var_names, ptr, sizeof(uint64_t));
  ptr += sizeof(uint64_t);
  std::vector<std::string> var_names(n_var_names);
  for (uint64_t i = 0; i < n_var_names; ++i) {
    var_names[i] = read_string(ptr);
  }

  // Read row names
  uint64_t n_row_names;
  std::memcpy(&n_row_names, ptr, sizeof(uint64_t));
  ptr += sizeof(uint64_t);
  std::vector<std::string> row_names(n_row_names);
  for (uint64_t i = 0; i < n_row_names; ++i) {
    row_names[i] = read_string(ptr);
  }

  // Set all the data
  if (!constraint_matrix_values.empty()) {
    problem.set_csr_constraint_matrix(constraint_matrix_values.data(),
                                      constraint_matrix_values.size(),
                                      constraint_matrix_indices.data(),
                                      constraint_matrix_indices.size(),
                                      constraint_matrix_offsets.data(),
                                      constraint_matrix_offsets.size());
  }

  if (!objective_coefficients.empty()) {
    problem.set_objective_coefficients(objective_coefficients.data(),
                                       objective_coefficients.size());
  }

  if (!variable_lower_bounds.empty()) {
    problem.set_variable_lower_bounds(variable_lower_bounds.data(), variable_lower_bounds.size());
  }

  if (!variable_upper_bounds.empty()) {
    problem.set_variable_upper_bounds(variable_upper_bounds.data(), variable_upper_bounds.size());
  }

  if (!constraint_lower_bounds.empty()) {
    problem.set_constraint_lower_bounds(constraint_lower_bounds.data(),
                                        constraint_lower_bounds.size());
  }

  if (!constraint_upper_bounds.empty()) {
    problem.set_constraint_upper_bounds(constraint_upper_bounds.data(),
                                        constraint_upper_bounds.size());
  }

  if (!constraint_bounds.empty()) {
    problem.set_constraint_bounds(constraint_bounds.data(), constraint_bounds.size());
  }

  if (!row_types.empty()) { problem.set_row_types(row_types.data(), row_types.size()); }

  if (!variable_types.empty()) {
    problem.set_variable_types(variable_types.data(), variable_types.size());
  }

  problem.set_maximize(maximize);
  problem.set_objective_scaling_factor(obj_scale);
  problem.set_objective_offset(obj_offset);

  if (!obj_name.empty()) { problem.set_objective_name(obj_name); }

  if (!problem_name.empty()) { problem.set_problem_name(problem_name.c_str()); }

  if (!var_names.empty()) { problem.set_variable_names(var_names); }

  if (!row_names.empty()) { problem.set_row_names(row_names); }

  return problem;
}

template <typename i_t, typename f_t>
std::vector<uint8_t> serialize_solution(optimization_problem_solution_t<i_t, f_t>& solution)
{
  std::vector<uint8_t> buffer;

  // Write solution vectors (may be empty)
  write_vector_to_buffer(buffer, solution.get_primal_solution());
  write_vector_to_buffer(buffer, solution.get_dual_solution());
  write_vector_to_buffer(buffer, solution.get_reduced_cost());

  // Write termination status
  uint32_t status = static_cast<uint32_t>(solution.get_termination_status());
  write_to_buffer(buffer, status);

  // Write warm start data (vectors may be empty if no warm start was used)
  const auto& ws = solution.get_pdlp_warm_start_data();
  write_vector_to_buffer(buffer, ws.current_primal_solution_);
  write_vector_to_buffer(buffer, ws.current_dual_solution_);
  write_vector_to_buffer(buffer, ws.initial_primal_average_);
  write_vector_to_buffer(buffer, ws.initial_dual_average_);
  write_vector_to_buffer(buffer, ws.current_ATY_);
  write_vector_to_buffer(buffer, ws.sum_primal_solutions_);
  write_vector_to_buffer(buffer, ws.sum_dual_solutions_);
  write_vector_to_buffer(buffer, ws.last_restart_duality_gap_primal_solution_);
  write_vector_to_buffer(buffer, ws.last_restart_duality_gap_dual_solution_);
  write_to_buffer(buffer, ws.initial_primal_weight_);
  write_to_buffer(buffer, ws.initial_step_size_);
  write_to_buffer(buffer, ws.total_pdlp_iterations_);
  write_to_buffer(buffer, ws.total_pdhg_iterations_);
  write_to_buffer(buffer, ws.last_candidate_kkt_score_);
  write_to_buffer(buffer, ws.last_restart_kkt_score_);
  write_to_buffer(buffer, ws.sum_solution_weight_);
  write_to_buffer(buffer, ws.iterations_since_last_restart_);

  // Write additional termination information
  const auto& info = solution.get_additional_termination_information();
  write_to_buffer(buffer, info.primal_objective);
  write_to_buffer(buffer, info.dual_objective);
  write_to_buffer(buffer, info.l2_primal_residual);
  write_to_buffer(buffer, info.l2_dual_residual);
  write_to_buffer(buffer, info.gap);
  write_to_buffer(buffer, info.number_of_steps_taken);
  write_to_buffer(buffer, info.solve_time);
  write_to_buffer(buffer, info.solved_by_pdlp);

  return buffer;
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> deserialize_lp_solution(
  const std::vector<uint8_t>& buffer)
{
  size_t offset = 0;

  // Read solution vectors
  std::vector<f_t> primal_solution;
  read_vector_from_buffer(buffer, offset, primal_solution);
  std::vector<f_t> dual_solution;
  read_vector_from_buffer(buffer, offset, dual_solution);
  std::vector<f_t> reduced_cost;
  read_vector_from_buffer(buffer, offset, reduced_cost);

  // Read termination status
  uint32_t status_val;
  read_from_buffer(buffer, offset, status_val);
  auto status = static_cast<pdlp_termination_status_t>(status_val);

  // Read warm start data (all fields, vectors may be empty)
  pdlp_warm_start_data_t<i_t, f_t> warm_start_data;
  read_vector_from_buffer(buffer, offset, warm_start_data.current_primal_solution_);
  read_vector_from_buffer(buffer, offset, warm_start_data.current_dual_solution_);
  read_vector_from_buffer(buffer, offset, warm_start_data.initial_primal_average_);
  read_vector_from_buffer(buffer, offset, warm_start_data.initial_dual_average_);
  read_vector_from_buffer(buffer, offset, warm_start_data.current_ATY_);
  read_vector_from_buffer(buffer, offset, warm_start_data.sum_primal_solutions_);
  read_vector_from_buffer(buffer, offset, warm_start_data.sum_dual_solutions_);
  read_vector_from_buffer(
    buffer, offset, warm_start_data.last_restart_duality_gap_primal_solution_);
  read_vector_from_buffer(buffer, offset, warm_start_data.last_restart_duality_gap_dual_solution_);
  read_from_buffer(buffer, offset, warm_start_data.initial_primal_weight_);
  read_from_buffer(buffer, offset, warm_start_data.initial_step_size_);
  read_from_buffer(buffer, offset, warm_start_data.total_pdlp_iterations_);
  read_from_buffer(buffer, offset, warm_start_data.total_pdhg_iterations_);
  read_from_buffer(buffer, offset, warm_start_data.last_candidate_kkt_score_);
  read_from_buffer(buffer, offset, warm_start_data.last_restart_kkt_score_);
  read_from_buffer(buffer, offset, warm_start_data.sum_solution_weight_);
  read_from_buffer(buffer, offset, warm_start_data.iterations_since_last_restart_);

  // Read additional termination information
  typename optimization_problem_solution_t<i_t, f_t>::additional_termination_information_t stats{};
  read_from_buffer(buffer, offset, stats.primal_objective);
  read_from_buffer(buffer, offset, stats.dual_objective);
  read_from_buffer(buffer, offset, stats.l2_primal_residual);
  read_from_buffer(buffer, offset, stats.l2_dual_residual);
  read_from_buffer(buffer, offset, stats.gap);
  read_from_buffer(buffer, offset, stats.number_of_steps_taken);
  read_from_buffer(buffer, offset, stats.solve_time);
  read_from_buffer(buffer, offset, stats.solved_by_pdlp);

  // Create solution with all data
  return optimization_problem_solution_t<i_t, f_t>(std::move(primal_solution),
                                                   std::move(dual_solution),
                                                   std::move(reduced_cost),
                                                   std::move(warm_start_data),
                                                   "",                          // objective_name
                                                   std::vector<std::string>(),  // var_names
                                                   std::vector<std::string>(),  // row_names
                                                   stats,
                                                   status);
}

template <typename i_t, typename f_t>
std::vector<uint8_t> serialize_mip_solution(mip_solution_t<i_t, f_t>& solution)
{
  std::vector<uint8_t> buffer;

  // Write solution vector (may be empty if solve failed)
  write_vector_to_buffer(buffer, solution.get_solution());

  // Write termination status
  uint32_t status = static_cast<uint32_t>(solution.get_termination_status());
  write_to_buffer(buffer, status);

  // Write objective value
  f_t obj_val = solution.get_objective_value();
  write_to_buffer(buffer, obj_val);

  return buffer;
}

template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> deserialize_mip_solution(const std::vector<uint8_t>& buffer)
{
  size_t offset = 0;

  // Read solution vector (may be empty if solve failed)
  std::vector<f_t> solution_vec;
  read_vector_from_buffer(buffer, offset, solution_vec);

  // Read termination status
  uint32_t status_val;
  read_from_buffer(buffer, offset, status_val);
  auto status = static_cast<mip_termination_status_t>(status_val);

  // Read objective value
  f_t obj_val;
  read_from_buffer(buffer, offset, obj_val);

  // Create minimal solver_stats
  solver_stats_t<i_t, f_t> stats{};
  stats.solution_bound = obj_val;

  // Create solution using the full constructor with minimal values
  return mip_solution_t<i_t, f_t>(std::move(solution_vec),           // solution
                                  std::vector<std::string>(),        // var_names
                                  obj_val,                           // objective
                                  f_t{0.0},                          // mip_gap
                                  status,                            // termination_status
                                  f_t{0.0},                          // max_constraint_violation
                                  f_t{0.0},                          // max_int_violation
                                  f_t{0.0},                          // max_variable_bound_violation
                                  stats,                             // stats
                                  std::vector<std::vector<f_t>>());  // solution_pool
}

// Connect to remote server
static int connect_to_server(const std::string& host, int port)
{
  struct addrinfo hints, *servinfo, *p;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family   = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  std::string port_str = std::to_string(port);
  int rv               = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &servinfo);
  if (rv != 0) { throw std::runtime_error(std::string("getaddrinfo failed: ") + gai_strerror(rv)); }

  int sockfd = -1;
  for (p = servinfo; p != nullptr; p = p->ai_next) {
    sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (sockfd == -1) { continue; }

    if (connect(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
      close(sockfd);
      sockfd = -1;
      continue;
    }

    break;
  }

  freeaddrinfo(servinfo);

  if (sockfd == -1) { throw std::runtime_error("Failed to connect to server"); }

  return sockfd;
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp_remote(
  const std::string& host,
  int port,
  const optimization_problem_t<i_t, f_t>& problem,
  const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  // Connect to server
  int sockfd = connect_to_server(host, port);

  try {
    // Serialize problem
    auto problem_data = serialize_problem(problem);

    // Create and send header
    remote_solve_header_t header;
    header.version      = 1;
    header.problem_type = 0;  // LP
    header.problem_size = problem_data.size();
    header.i_type_size  = sizeof(i_t);
    header.f_type_size  = sizeof(f_t);

    write_all(sockfd, &header, sizeof(header));

    // Send problem data
    write_all(sockfd, problem_data.data(), problem_data.size());

    // TODO: Send settings if needed

    // Read response header
    remote_solve_response_header_t response_header;
    read_all(sockfd, &response_header, sizeof(response_header));

    if (response_header.status != 0) {
      close(sockfd);
      throw std::runtime_error("Remote solve failed with status: " +
                               std::to_string(response_header.status));
    }

    // Read solution data
    std::vector<uint8_t> solution_data(response_header.solution_size);
    read_all(sockfd, solution_data.data(), solution_data.size());
    std::fprintf(
      stderr, "[solve_lp_remote] Received %zu bytes of solution data\n", solution_data.size());
    std::fflush(stderr);

    close(sockfd);

    // Deserialize solution
    std::fprintf(stderr, "[solve_lp_remote] Deserializing solution...\n");
    std::fflush(stderr);
    auto solution = deserialize_lp_solution<i_t, f_t>(solution_data);
    std::fprintf(stderr, "[solve_lp_remote] Deserialization complete, returning solution\n");
    std::fflush(stderr);
    return solution;

  } catch (...) {
    close(sockfd);
    throw;
  }
}

template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip_remote(const std::string& host,
                                          int port,
                                          const optimization_problem_t<i_t, f_t>& problem,
                                          const mip_solver_settings_t<i_t, f_t>& settings)
{
  // Connect to server
  int sockfd = connect_to_server(host, port);

  try {
    // Serialize problem
    auto problem_data = serialize_problem(problem);

    // Create and send header
    remote_solve_header_t header;
    header.version      = 1;
    header.problem_type = 1;  // MIP
    header.problem_size = problem_data.size();
    header.i_type_size  = sizeof(i_t);
    header.f_type_size  = sizeof(f_t);

    write_all(sockfd, &header, sizeof(header));

    // Send problem data
    write_all(sockfd, problem_data.data(), problem_data.size());

    // TODO: Send settings if needed

    // Read response header
    remote_solve_response_header_t response_header;
    read_all(sockfd, &response_header, sizeof(response_header));

    if (response_header.status != 0) {
      close(sockfd);
      throw std::runtime_error("Remote solve failed with status: " +
                               std::to_string(response_header.status));
    }

    // Read solution data
    std::vector<uint8_t> solution_data(response_header.solution_size);
    read_all(sockfd, solution_data.data(), solution_data.size());

    close(sockfd);

    // Deserialize solution
    return deserialize_mip_solution<i_t, f_t>(solution_data);

  } catch (...) {
    close(sockfd);
    throw;
  }
}

// Explicit template instantiations
#if MIP_INSTANTIATE_FLOAT
template std::vector<uint8_t> serialize_problem(const optimization_problem_t<int, float>& problem);

template optimization_problem_t<int, float> deserialize_problem(const std::vector<uint8_t>& buffer);

template std::vector<uint8_t> serialize_solution(
  optimization_problem_solution_t<int, float>& solution);

template optimization_problem_solution_t<int, float> deserialize_lp_solution(
  const std::vector<uint8_t>& buffer);

template std::vector<uint8_t> serialize_mip_solution(mip_solution_t<int, float>& solution);

template mip_solution_t<int, float> deserialize_mip_solution(const std::vector<uint8_t>& buffer);

template optimization_problem_solution_t<int, float> solve_lp_remote(
  const std::string& host,
  int port,
  const optimization_problem_t<int, float>& problem,
  const pdlp_solver_settings_t<int, float>& settings);

template mip_solution_t<int, float> solve_mip_remote(
  const std::string& host,
  int port,
  const optimization_problem_t<int, float>& problem,
  const mip_solver_settings_t<int, float>& settings);
#endif

#if MIP_INSTANTIATE_DOUBLE
template std::vector<uint8_t> serialize_problem(const optimization_problem_t<int, double>& problem);

template optimization_problem_t<int, double> deserialize_problem(
  const std::vector<uint8_t>& buffer);

template std::vector<uint8_t> serialize_solution(
  optimization_problem_solution_t<int, double>& solution);

template optimization_problem_solution_t<int, double> deserialize_lp_solution(
  const std::vector<uint8_t>& buffer);

template std::vector<uint8_t> serialize_mip_solution(mip_solution_t<int, double>& solution);

template mip_solution_t<int, double> deserialize_mip_solution(const std::vector<uint8_t>& buffer);

template optimization_problem_solution_t<int, double> solve_lp_remote(
  const std::string& host,
  int port,
  const optimization_problem_t<int, double>& problem,
  const pdlp_solver_settings_t<int, double>& settings);

template mip_solution_t<int, double> solve_mip_remote(
  const std::string& host,
  int port,
  const optimization_problem_t<int, double>& problem,
  const mip_solver_settings_t<int, double>& settings);
#endif

}  // namespace cuopt::linear_programming
