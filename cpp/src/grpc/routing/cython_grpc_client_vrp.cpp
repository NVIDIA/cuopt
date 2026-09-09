/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Routing arm of the Cython-facing client shim. Split out of
// cython_grpc_client.cpp so the routing gRPC path is a separate translation
// unit from the LP/MIP one (see grpc_client_vrp.cpp).

#include "../client/cython_grpc_client_impl.hpp"

#include <cuopt/routing/cpu_routing_problem.hpp>
#include <cuopt/routing/solver_settings.hpp>

namespace cuopt {
namespace cython {

grpc_submit_result_t grpc_python_client_t::submit_vrp(
  cuopt::routing::cpu_routing_problem_t* problem,
  cuopt::routing::solver_settings_t<int, float>* settings)
{
  grpc_submit_result_t out;
  if (problem == nullptr || settings == nullptr) {
    out.error_message = "problem and settings must not be null";
    return out;
  }
  auto sub          = impl_->client.submit_vrp(*problem, *settings);
  out.success       = sub.success;
  out.error_message = sub.error_message;
  out.job_id        = sub.job_id;
  out.is_mip        = false;
  return out;
}

grpc_vrp_result_outcome_t grpc_python_client_t::result_vrp(const std::string& job_id)
{
  grpc_vrp_result_outcome_t out;

  // Mirror result(): surface a structured "still running" signal instead of a
  // generic GetResult failure when a caller polls result_vrp on an in-flight job.
  auto st = status(job_id);
  if (!st.success) {
    out.error_message = st.error_message;
    return out;
  }
  if (is_in_flight(st.status)) {
    out.not_ready = true;
    return out;
  }

  auto remote = impl_->client.get_vrp_result(job_id);
  if (!remote.success) {
    out.error_message = remote.error_message;
    return out;
  }
  out.success  = true;
  out.solution = std::move(remote.solution);
  return out;
}

}  // namespace cython
}  // namespace cuopt
