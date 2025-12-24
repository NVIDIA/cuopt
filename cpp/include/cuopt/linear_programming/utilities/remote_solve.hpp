/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

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

}  // namespace cuopt::linear_programming
