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

#pragma once

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>

#include <string>
#include <vector>

namespace cuopt::linear_programming {

/**
 * @brief Protocol header for remote solve requests
 *
 * Binary protocol format:
 * 1. remote_solve_header_t (20 bytes)
 * 2. Serialized optimization_problem_t
 * 3. Serialized settings
 *
 * Response format:
 * 1. remote_solve_response_header_t (8 bytes)
 * 2. Serialized solution
 */
struct remote_solve_header_t {
  uint32_t version;       // Protocol version (currently 1)
  uint32_t problem_type;  // 0 = LP, 1 = MIP
  uint64_t problem_size;  // Size of serialized problem in bytes
  uint32_t i_type_size;   // sizeof(i_t) - 4 or 8
  uint32_t f_type_size;   // sizeof(f_t) - 4 or 8
};

struct remote_solve_response_header_t {
  uint32_t status;         // 0 = success, non-zero = error
  uint32_t solution_size;  // Size of serialized solution in bytes
};

/**
 * @brief Serialize optimization_problem_t to binary buffer
 */
template <typename i_t, typename f_t>
std::vector<uint8_t> serialize_problem(const optimization_problem_t<i_t, f_t>& problem);

/**
 * @brief Deserialize optimization_problem_t from binary buffer
 */
template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t> deserialize_problem(const std::vector<uint8_t>& buffer);

/**
 * @brief Serialize optimization_problem_solution_t to binary buffer
 */
template <typename i_t, typename f_t>
std::vector<uint8_t> serialize_solution(optimization_problem_solution_t<i_t, f_t>& solution);

/**
 * @brief Deserialize optimization_problem_solution_t from binary buffer
 */
template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> deserialize_lp_solution(
  const std::vector<uint8_t>& buffer);

/**
 * @brief Serialize mip_solution_t to binary buffer
 */
template <typename i_t, typename f_t>
std::vector<uint8_t> serialize_mip_solution(mip_solution_t<i_t, f_t>& solution);

/**
 * @brief Deserialize mip_solution_t from binary buffer
 */
template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> deserialize_mip_solution(const std::vector<uint8_t>& buffer);

/**
 * @brief Solve LP problem on remote server
 *
 * @tparam i_t Integer type for indices
 * @tparam f_t Float type for values
 * @param host Hostname or IP address of remote server
 * @param port Port number of remote server
 * @param problem The optimization problem (host memory)
 * @param settings Solver settings
 * @return Solution from remote server
 */
template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp_remote(
  const std::string& host,
  int port,
  const optimization_problem_t<i_t, f_t>& problem,
  const pdlp_solver_settings_t<i_t, f_t>& settings);

/**
 * @brief Solve MIP problem on remote server
 *
 * @tparam i_t Integer type for indices
 * @tparam f_t Float type for values
 * @param host Hostname or IP address of remote server
 * @param port Port number of remote server
 * @param problem The optimization problem (host memory)
 * @param settings Solver settings
 * @return Solution from remote server
 */
template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip_remote(const std::string& host,
                                          int port,
                                          const optimization_problem_t<i_t, f_t>& problem,
                                          const mip_solver_settings_t<i_t, f_t>& settings);

}  // namespace cuopt::linear_programming
