/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// VRP (routing) arm of grpc_client_t. Kept in its own translation unit so the
// routing gRPC path can be compiled (and later packaged) independently of the
// LP/MIP client: everything here depends on the routing engine, while
// grpc_client.cpp depends only on mathematical_optimization.

#include "../client/grpc_client.hpp"

#include "../grpc_container_chunk.hpp"
#include "grpc_routing_chunk_ids.hpp"
#include "grpc_routing_problem_mapper.hpp"
#include "grpc_routing_settings_mapper.hpp"
#include "grpc_routing_solution_mapper.hpp"

#include <cuopt_remote_service.grpc.pb.h>

namespace cuopt::mathematical_optimization {

using cuopt::routing::map_proto_to_routing_solution;
using cuopt::routing::map_routing_problem_to_proto;
using cuopt::routing::map_routing_settings_to_proto;

submit_result_t grpc_client_t::submit_vrp(
  const cuopt::routing::cpu_routing_problem_t& problem,
  const cuopt::routing::solver_settings_t<int, float>& settings)
{
  submit_result_t result;

  if (!is_connected()) {
    result.error_message = "Not connected to server";
    return result;
  }

  // The header carries everything except the matrix values, which dominate the size.
  cuopt::remote::ChunkedProblemHeader header;
  header.mutable_header()->set_version(1);
  header.mutable_header()->set_problem_category(cuopt::remote::VRP);
  map_routing_problem_to_proto(problem, header.mutable_routing_problem(), false);
  map_routing_settings_to_proto(settings, header.mutable_vrp_settings());

  int64_t matrix_bytes = 0;
  for (const auto& m : problem.cost_matrices) {
    matrix_bytes += static_cast<int64_t>(m.matrix.size() * sizeof(float));
  }
  for (const auto& m : problem.transit_time_matrices) {
    matrix_bytes += static_cast<int64_t>(m.matrix.size() * sizeof(float));
  }

  const bool use_chunked =
    chunked_array_threshold_bytes_ >= 0 &&
    static_cast<int64_t>(header.ByteSizeLong()) + matrix_bytes > chunked_array_threshold_bytes_;

  if (use_chunked) {
    auto build_chunks = [&](const std::string& upload_id, int64_t chunk_data_budget) {
      std::vector<cuopt::remote::SendArrayChunkRequest> chunks;
      for (size_t i = 0; i < problem.cost_matrices.size(); ++i) {
        chunk_container_typed_array(chunks,
                                    cuopt::routing::kCostMatricesContainer,
                                    static_cast<int32_t>(i),
                                    cuopt::routing::kMatrixValuesFieldId,
                                    problem.cost_matrices[i].matrix,
                                    upload_id,
                                    chunk_data_budget);
      }
      for (size_t i = 0; i < problem.transit_time_matrices.size(); ++i) {
        chunk_container_typed_array(chunks,
                                    cuopt::routing::kTransitMatricesContainer,
                                    static_cast<int32_t>(i),
                                    cuopt::routing::kMatrixValuesFieldId,
                                    problem.transit_time_matrices[i].matrix,
                                    upload_id,
                                    chunk_data_budget);
      }
      return chunks;
    };
    if (!upload_chunked(header, build_chunks, result.job_id)) {
      result.error_message = last_error_;
      return result;
    }
  } else {
    cuopt::remote::SubmitJobRequest submit_request;
    auto* vrp = submit_request.mutable_vrp_request();
    vrp->mutable_header()->set_version(1);
    vrp->mutable_header()->set_problem_category(cuopt::remote::VRP);
    map_routing_problem_to_proto(problem, vrp->mutable_problem());
    map_routing_settings_to_proto(settings, vrp->mutable_settings());

    if (!submit_unary(submit_request, result.job_id)) {
      result.error_message = last_error_;
      return result;
    }
  }
  result.success = true;
  return result;
}

remote_vrp_result_t grpc_client_t::get_vrp_result(const std::string& job_id)
{
  remote_vrp_result_t result;

  if (!is_connected()) {
    result.error_message = "Not connected to server";
    return result;
  }

  downloaded_result_t dl;
  if (!get_result_or_download(job_id, dl)) {
    result.error_message = last_error_;
    return result;
  }
  // The solution is embedded in the result header on the chunked path too, with no arrays.
  const auto& routing_solution =
    dl.was_chunked ? dl.chunked_header->routing_solution() : dl.response->routing_solution();
  map_proto_to_routing_solution(routing_solution, result.solution);
  result.success = true;
  return result;
}

}  // namespace cuopt::mathematical_optimization
