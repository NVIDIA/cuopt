/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/grpc/grpc_client_env.hpp>

#include "grpc_client.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cuopt::mathematical_optimization {

namespace {

int64_t parse_env_int64(const char* name, int64_t default_value)
{
  const char* val = std::getenv(name);
  if (val == nullptr) return default_value;
  try {
    return std::stoll(val);
  } catch (...) {
    return default_value;
  }
}

std::string read_pem_file(const char* path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) { throw std::runtime_error(std::string("Cannot open TLS file: ") + path); }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

const char* get_env(const char* name)
{
  const char* v = std::getenv(name);
  return (v && v[0] != '\0') ? v : nullptr;
}

}  // namespace

void apply_grpc_client_env_overrides(grpc_client_config_t& config, bool apply_tls)
{
  constexpr int64_t kMinChunkSize   = 4096;
  constexpr int64_t kMaxChunkSize   = 2LL * 1024 * 1024 * 1024;
  constexpr int64_t kMinMessageSize = 4096;
  constexpr int64_t kMaxMessageSize = 2LL * 1024 * 1024 * 1024;

  auto chunk = parse_env_int64("CUOPT_CHUNK_SIZE", config.chunk_size_bytes);
  if (chunk >= kMinChunkSize && chunk <= kMaxChunkSize) { config.chunk_size_bytes = chunk; }

  auto msg = parse_env_int64("CUOPT_MAX_MESSAGE_BYTES", config.max_message_bytes);
  if (msg >= kMinMessageSize && msg <= kMaxMessageSize) { config.max_message_bytes = msg; }

  config.enable_debug_log = (parse_env_int64("CUOPT_GRPC_DEBUG", 0) != 0);

  if (!apply_tls) { return; }

  if (parse_env_int64("CUOPT_TLS_ENABLED", 0) != 0) {
    config.enable_tls = true;

    const char* root_cert = get_env("CUOPT_TLS_ROOT_CERT");
    if (root_cert) { config.tls_root_certs = read_pem_file(root_cert); }

    const char* client_cert = get_env("CUOPT_TLS_CLIENT_CERT");
    const char* client_key  = get_env("CUOPT_TLS_CLIENT_KEY");
    if (client_cert != nullptr || client_key != nullptr) {
      if (client_cert == nullptr || client_key == nullptr) {
        throw std::invalid_argument(
          "CUOPT_TLS_CLIENT_CERT and CUOPT_TLS_CLIENT_KEY must both be set for mTLS");
      }
      config.tls_client_cert = read_pem_file(client_cert);
      config.tls_client_key  = read_pem_file(client_key);
    }
  }
}

namespace {

void validate_host_port(const std::string& host, int port)
{
  if (host.empty()) { throw std::invalid_argument("gRPC host must not be empty"); }
  if (port <= 0 || port > 65535) {
    throw std::invalid_argument("gRPC port must be between 1 and 65535");
  }
}

void apply_explicit_tls(grpc_client_config_t& config, const grpc_explicit_tls_t& tls)
{
  const bool has_client_cert = !tls.client_cert.empty();
  const bool has_client_key  = !tls.client_key.empty();
  if (has_client_cert != has_client_key) {
    throw std::invalid_argument(
      "client_cert and client_key must both be set for mTLS, or neither for server TLS only");
  }

  config.enable_tls      = true;
  config.tls_root_certs  = tls.root_certs;
  config.tls_client_cert = tls.client_cert;
  config.tls_client_key  = tls.client_key;
}

}  // namespace

grpc_client_config_t make_grpc_client_config(const std::string& host, int port)
{
  return make_grpc_client_config(host, port, grpc_tls_mode_t::ENV, nullptr);
}

grpc_client_config_t make_grpc_client_config(const std::string& host,
                                             int port,
                                             grpc_tls_mode_t tls_mode,
                                             const grpc_explicit_tls_t* explicit_tls)
{
  validate_host_port(host, port);

  grpc_client_config_t config;
  config.server_address = host + ":" + std::to_string(port);

  switch (tls_mode) {
    case grpc_tls_mode_t::ENV: apply_grpc_client_env_overrides(config, true); break;
    case grpc_tls_mode_t::DISABLED: apply_grpc_client_env_overrides(config, false); break;
    case grpc_tls_mode_t::EXPLICIT:
      if (explicit_tls == nullptr) {
        throw std::invalid_argument("explicit_tls is required for EXPLICIT TLS mode");
      }
      apply_explicit_tls(config, *explicit_tls);
      apply_grpc_client_env_overrides(config, false);
      break;
  }

  return config;
}

}  // namespace cuopt::mathematical_optimization
