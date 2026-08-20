/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Internal detail shared by the translation units that implement
// grpc_python_client_t: the LP/MIP arm (cython_grpc_client.cpp) and the routing
// arm (grpc/routing/cython_grpc_client_vrp.cpp). Not installed.

#include <cuopt/grpc/cython_grpc_client.hpp>

#include "grpc_client.hpp"

#include <utility>

namespace cuopt {
namespace cython {

struct grpc_python_client_t::impl_t {
  cuopt::mathematical_optimization::grpc_client_t client;
  explicit impl_t(cuopt::mathematical_optimization::grpc_client_config_t config)
    : client(std::move(config))
  {
  }
};

/** A job the server has not finished yet; callers should keep polling. */
inline bool is_in_flight(grpc_job_status_t status)
{
  return status == grpc_job_status_t::QUEUED || status == grpc_job_status_t::PROCESSING;
}

}  // namespace cython
}  // namespace cuopt
