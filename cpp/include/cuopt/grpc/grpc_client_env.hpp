/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "grpc_client.hpp"

namespace cuopt::mathematical_optimization {

/** How TLS is chosen when building a grpc_client_config_t. */
enum class grpc_tls_mode_t {
  /** Read CUOPT_TLS_* from the environment (default). */
  ENV = 0,
  /** Plain TCP; ignore CUOPT_TLS_* even if set. */
  DISABLED,
  /** Use explicit PEM strings supplied by the caller. */
  EXPLICIT,
};

/** PEM material for EXPLICIT TLS / mTLS (contents, not file paths). */
struct grpc_explicit_tls_t {
  std::string root_certs;
  std::string client_cert;
  std::string client_key;
};

/**
 * @brief Apply CUOPT_GRPC_* / CUOPT_TLS_* / CUOPT_CHUNK_SIZE env overrides.
 *
 * @param apply_tls When false, CUOPT_TLS_* is ignored (chunk/debug env still apply).
 */
void apply_grpc_client_env_overrides(grpc_client_config_t& config, bool apply_tls = true);

/**
 * @brief Build grpc_client_config_t from host, port, and environment overrides.
 */
grpc_client_config_t make_grpc_client_config(const std::string& host, int port);

/**
 * @brief Build grpc_client_config_t with explicit TLS mode.
 *
 * For EXPLICIT mode, @p explicit_tls must be non-null and root_certs non-empty.
 */
grpc_client_config_t make_grpc_client_config(const std::string& host,
                                             int port,
                                             grpc_tls_mode_t tls_mode,
                                             const grpc_explicit_tls_t* explicit_tls = nullptr);

}  // namespace cuopt::mathematical_optimization
