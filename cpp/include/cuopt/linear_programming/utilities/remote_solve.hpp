/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <mps_parser/data_model_view.hpp>

#include <cstdlib>
#include <optional>
#include <string>

namespace cuopt::linear_programming {

/**
 * @brief Configuration for remote solve connection
 */
struct remote_solve_config_t {
  std::string host;
  int port;
};

/**
 * @brief Check if remote solve is enabled via environment variables.
 *
 * Remote solve is enabled when both CUOPT_REMOTE_HOST and CUOPT_REMOTE_PORT
 * environment variables are set.
 *
 * @return std::optional<remote_solve_config_t> containing the remote config if
 *         remote solve is enabled, std::nullopt otherwise
 */
inline std::optional<remote_solve_config_t> get_remote_solve_config()
{
  const char* host = std::getenv("CUOPT_REMOTE_HOST");
  const char* port = std::getenv("CUOPT_REMOTE_PORT");

  if (host != nullptr && port != nullptr && host[0] != '\0' && port[0] != '\0') {
    try {
      int port_num = std::stoi(port);
      return remote_solve_config_t{std::string(host), port_num};
    } catch (...) {
      // Invalid port number, fall back to local solve
      return std::nullopt;
    }
  }
  return std::nullopt;
}

/**
 * @brief Check if remote solve is enabled.
 *
 * @return true if CUOPT_REMOTE_HOST and CUOPT_REMOTE_PORT are both set
 */
inline bool is_remote_solve_enabled() { return get_remote_solve_config().has_value(); }

/**
 * @brief Solve an LP problem on a remote server.
 *
 * Stub implementation for the memory-model-only branch.
 */
template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp_remote(
  const remote_solve_config_t&,
  const cuopt::mps_parser::data_model_view_t<i_t, f_t>&,
  const pdlp_solver_settings_t<i_t, f_t>&)
{
  return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
    "Remote solve is not enabled in this build", cuopt::error_type_t::RuntimeError));
}

/**
 * @brief Solve a MIP problem on a remote server.
 *
 * Stub implementation for the memory-model-only branch.
 */
template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip_remote(const remote_solve_config_t&,
                                          const cuopt::mps_parser::data_model_view_t<i_t, f_t>&,
                                          const mip_solver_settings_t<i_t, f_t>&)
{
  return mip_solution_t<i_t, f_t>(cuopt::logic_error("Remote solve is not enabled in this build",
                                                     cuopt::error_type_t::RuntimeError));
}

}  // namespace cuopt::linear_programming
