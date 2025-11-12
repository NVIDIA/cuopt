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

namespace cuopt::linear_programming {

/**
 * @brief Check if remote solve is enabled via environment variables
 *
 * Checks for CUOPT_REMOTE_HOST and CUOPT_REMOTE_PORT environment variables.
 *
 * @param[out] host Pointer to store the host string (or nullptr if not set)
 * @param[out] port Pointer to store the port string (or nullptr if not set)
 * @return true if both environment variables are set
 */
bool is_remote_solve_enabled(const char** host, const char** port);

/**
 * @brief Solve LP problem on remote server using Protocol Buffers
 *
 * Reads CUOPT_REMOTE_HOST and CUOPT_REMOTE_PORT from environment,
 * serializes the problem and settings using Protocol Buffers,
 * sends to remote server via TCP, and deserializes the solution.
 *
 * @tparam i_t Integer type for indices
 * @tparam f_t Float type for values
 * @param problem The optimization problem (host memory)
 * @param settings Solver settings
 * @return Solution from remote server
 * @throws std::runtime_error if remote solve is not enabled or connection fails
 */
template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp_remote(
  const optimization_problem_t<i_t, f_t>& problem,
  const pdlp_solver_settings_t<i_t, f_t>& settings);

/**
 * @brief Solve MIP problem on remote server using Protocol Buffers
 *
 * Reads CUOPT_REMOTE_HOST and CUOPT_REMOTE_PORT from environment,
 * serializes the problem and settings using Protocol Buffers,
 * sends to remote server via TCP, and deserializes the solution.
 *
 * @tparam i_t Integer type for indices
 * @tparam f_t Float type for values
 * @param problem The optimization problem (host memory)
 * @param settings Solver settings
 * @return Solution from remote server
 * @throws std::runtime_error if remote solve is not enabled or connection fails
 */
template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip_remote(const optimization_problem_t<i_t, f_t>& problem,
                                          const mip_solver_settings_t<i_t, f_t>& settings);

}  // namespace cuopt::linear_programming
