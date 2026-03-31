/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file grpc_client_test_helper.hpp
 * @brief Test helper for injecting mock stubs into grpc_client_t
 *
 * This header is for unit testing only - it exposes internal gRPC types
 * that are normally hidden by the PIMPL pattern. Include this only in test code.
 */

#include <memory>

#include <cuopt_remote_service.grpc.pb.h>
#include <grpcpp/grpcpp.h>

#include "grpc_client.hpp"

namespace cuopt::linear_programming {

/**
 * @brief Inject a mock stub into a grpc_client_t instance for testing
 *
 * This allows unit tests to provide mock stubs that simulate various
 * server responses and error conditions without needing a real server.
 *
 * Usage:
 * @code
 * grpc_client_config_t config;
 * config.server_address = "mock://test";
 * grpc_client_t client(config);
 *
 * auto mock_stub = std::make_shared<MockCuOptStub>();
 * grpc_test_inject_mock_stub(client, mock_stub);
 *
 * // Now client.check_status() etc. will use the mock stub
 * @endcode
 *
 * @param client The client to inject the stub into
 * @param stub The mock stub to inject (takes ownership via shared_ptr)
 */
void grpc_test_inject_mock_stub(grpc_client_t& client, std::shared_ptr<void> stub);

/**
 * @brief Mark a client as "connected" without actually connecting
 *
 * For mock testing, we don't have a real channel but need the client
 * to think it's connected so it will use the stub.
 */
void grpc_test_mark_as_connected(grpc_client_t& client);

/**
 * @brief Helper template to cast mock stub to void pointer for injection
 */
template <typename T>
inline void grpc_test_inject_mock_stub_typed(grpc_client_t& client, std::shared_ptr<T> stub)
{
  grpc_test_inject_mock_stub(client, std::static_pointer_cast<void>(stub));
}

}  // namespace cuopt::linear_programming
