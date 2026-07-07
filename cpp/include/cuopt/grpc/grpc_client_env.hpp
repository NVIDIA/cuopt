/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "grpc_client.hpp"

namespace cuopt::mathematical_optimization {

/**
 * @brief Apply CUOPT_GRPC_* / CUOPT_TLS_* / CUOPT_CHUNK_SIZE env overrides.
 */
void apply_grpc_client_env_overrides(grpc_client_config_t& config);

/**
 * @brief Build grpc_client_config_t from host, port, and environment overrides.
 */
grpc_client_config_t make_grpc_client_config(const std::string& host, int port);

}  // namespace cuopt::mathematical_optimization
