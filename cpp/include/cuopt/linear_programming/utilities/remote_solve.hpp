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
  const cuopt::mps_parser::data_model_view_t<i_t, f_t>& view,
  const pdlp_solver_settings_t<i_t, f_t>&)
{
  auto n_rows = view.get_constraint_matrix_offsets().size() > 0
                  ? static_cast<i_t>(view.get_constraint_matrix_offsets().size()) - 1
                  : 0;
  auto n_cols = static_cast<i_t>(view.get_objective_coefficients().size());

  std::vector<f_t> primal_solution(static_cast<size_t>(n_cols), f_t{0});
  std::vector<f_t> dual_solution(static_cast<size_t>(n_rows), f_t{0});
  std::vector<f_t> reduced_cost(static_cast<size_t>(n_cols), f_t{0});

  typename optimization_problem_solution_t<i_t, f_t>::additional_termination_information_t stats;
  stats.number_of_steps_taken           = 0;
  stats.total_number_of_attempted_steps = 0;
  stats.l2_primal_residual              = f_t{0};
  stats.l2_relative_primal_residual     = f_t{0};
  stats.l2_dual_residual                = f_t{0};
  stats.l2_relative_dual_residual       = f_t{0};
  stats.primal_objective                = f_t{0};
  stats.dual_objective                  = f_t{0};
  stats.gap                             = f_t{0};
  stats.relative_gap                    = f_t{0};
  stats.max_primal_ray_infeasibility    = f_t{0};
  stats.primal_ray_linear_objective     = f_t{0};
  stats.max_dual_ray_infeasibility      = f_t{0};
  stats.dual_ray_linear_objective       = f_t{0};
  stats.solve_time                      = 0.0;
  stats.solved_by_pdlp                  = false;
  return optimization_problem_solution_t<i_t, f_t>(std::move(primal_solution),
                                                   std::move(dual_solution),
                                                   std::move(reduced_cost),
                                                   "",
                                                   {},
                                                   {},
                                                   stats,
                                                   pdlp_termination_status_t::Optimal);
}

/**
 * @brief Solve a MIP problem on a remote server.
 *
 * Stub implementation for the memory-model-only branch.
 */
template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip_remote(
  const remote_solve_config_t&,
  const cuopt::mps_parser::data_model_view_t<i_t, f_t>& view,
  const mip_solver_settings_t<i_t, f_t>&)
{
  auto n_cols = static_cast<i_t>(view.get_objective_coefficients().size());
  std::vector<f_t> solution(static_cast<size_t>(n_cols), f_t{0});
  solver_stats_t<i_t, f_t> stats{};
  stats.total_solve_time       = f_t{0};
  stats.presolve_time          = f_t{0};
  stats.solution_bound         = f_t{0};
  stats.num_nodes              = 0;
  stats.num_simplex_iterations = 0;
  return mip_solution_t<i_t, f_t>(std::move(solution),
                                  {},
                                  f_t{0},
                                  f_t{0},
                                  mip_termination_status_t::Optimal,
                                  f_t{0},
                                  f_t{0},
                                  f_t{0},
                                  stats);
}

}  // namespace cuopt::linear_programming
