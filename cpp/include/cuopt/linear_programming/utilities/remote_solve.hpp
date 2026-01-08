/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

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
 * @tparam i_t Index type (int32_t)
 * @tparam f_t Float type (float or double)
 * @param config Remote server configuration
 * @param view Problem data view
 * @param settings Solver settings
 * @return Solution from the remote server
 */
template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<i_t, f_t>& view,
  const pdlp_solver_settings_t<i_t, f_t>& settings);

/**
 * @brief Solve a MIP problem on a remote server.
 *
 * @tparam i_t Index type (int32_t)
 * @tparam f_t Float type (float or double)
 * @param config Remote server configuration
 * @param view Problem data view
 * @param settings Solver settings
 * @return Solution from the remote server
 */
template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<i_t, f_t>& view,
  const mip_solver_settings_t<i_t, f_t>& settings);

/**
 * @brief Job status enumeration for remote jobs.
 */
enum class remote_job_status_t {
  QUEUED,      ///< Job is waiting in queue
  PROCESSING,  ///< Job is being processed by a worker
  COMPLETED,   ///< Job completed successfully
  FAILED,      ///< Job failed with an error
  NOT_FOUND,   ///< Job ID not found on server
  CANCELLED    ///< Job was cancelled
};

/**
 * @brief Result of a cancel job request.
 */
struct cancel_job_result_t {
  bool success;                    ///< True if cancellation was successful
  std::string message;             ///< Success/error message
  remote_job_status_t job_status;  ///< Status of job after cancel attempt
};

/**
 * @brief Cancel a job on a remote server.
 *
 * This function can cancel jobs that are queued (waiting for a worker) or
 * currently running. For running jobs, the worker process is killed and
 * automatically restarted by the server.
 *
 * @param config Remote server configuration
 * @param job_id The job ID to cancel
 * @return Result containing success status, message, and job status
 */
cancel_job_result_t cancel_job_remote(const remote_solve_config_t& config,
                                      const std::string& job_id);

}  // namespace cuopt::linear_programming
