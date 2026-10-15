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

static_assert(static_cast<int32_t>(cuopt::remote::CostMatrix::VALUES) ==
              cuopt::routing::kMatrixValuesFieldId);

using cuopt::routing::map_proto_to_routing_solution;
using cuopt::routing::map_routing_problem_to_proto;
using cuopt::routing::map_routing_settings_to_proto;

namespace {

void append_routing_matrix_chunks(std::vector<cuopt::remote::SendArrayChunkRequest>& chunks,
                                  const cuopt::routing::cpu_routing_problem_t& problem,
                                  const std::string& upload_id,
                                  int64_t chunk_data_budget)
{
#include "generated_routing_matrix_chunks.inc"
}

void append_routing_nonmatrix_chunks(std::vector<cuopt::remote::SendArrayChunkRequest>& chunks,
                                     const cuopt::routing::cpu_routing_problem_t& problem,
                                     const std::string& upload_id,
                                     int64_t chunk_data_budget)
{
#include "generated_routing_nonmatrix_chunks.inc"
}

}  // namespace

submit_result_t grpc_client_t::submit_vrp(
  const cuopt::routing::cpu_routing_problem_t& problem,
  const cuopt::routing::solver_settings_t<int, float>& settings)
{
  submit_result_t result;

  if (!is_connected()) {
    result.error_message = "Not connected to server";
    return result;
  }

  // Start from a header with matrix values omitted. If that header does not fit
  // the hard send limit, the other arrays are omitted too and sent as chunks.
  // Unified upload uses the same SolveVRPRequest as the unary RPC.
  cuopt::remote::SubmitJobRequest header;
  auto* vrp_job = header.mutable_vrp_request();
  vrp_job->mutable_header()->set_version(1);
  vrp_job->mutable_header()->set_problem_category(cuopt::remote::VRP);
  map_routing_problem_to_proto(problem, vrp_job->mutable_problem(), false, true);
  map_routing_settings_to_proto(settings, vrp_job->mutable_settings());

  int64_t matrix_bytes = 0;
  for (const auto& m : problem.cost_matrices) {
    matrix_bytes += static_cast<int64_t>(m.matrix.size() * sizeof(float));
  }
  for (const auto& m : problem.transit_time_matrices) {
    matrix_bytes += static_cast<int64_t>(m.matrix.size() * sizeof(float));
  }

  // The unary request has to fit the configured threshold, the client channel
  // limit, and the server limit stored by connect()'s CheckStatus probe. An
  // unknown server limit is treated as too small for a unary upload.
  const int64_t request_bytes = static_cast<int64_t>(header.ByteSizeLong()) + matrix_bytes;
  int64_t message_limit       = config_.max_message_bytes;
  if (chunked_array_threshold_bytes_ >= 0 && chunked_array_threshold_bytes_ < message_limit) {
    message_limit = chunked_array_threshold_bytes_;
  }
  const int64_t srv_max           = server_max_message_bytes_.load(std::memory_order_relaxed);
  const bool server_limit_unknown = srv_max <= 0;
  if (!server_limit_unknown && srv_max < message_limit) { message_limit = srv_max; }
  const bool use_chunked = server_limit_unknown || request_bytes > message_limit;

  // Once we are chunking, the header only has to fit the hard send limit.
  // The 75% threshold is not part of that check.
  int64_t hard_limit = config_.max_message_bytes;
  if (!server_limit_unknown && srv_max < hard_limit) { hard_limit = srv_max; }

  if (use_chunked) {
    cuopt::remote::StartChunkedUploadV2Request wrapped;
    *wrapped.mutable_job()  = header;
    bool chunk_other_arrays = false;
    if (static_cast<int64_t>(wrapped.ByteSizeLong()) > hard_limit) {
      map_routing_problem_to_proto(problem, vrp_job->mutable_problem(), false, false);
      *wrapped.mutable_job() = header;
      if (static_cast<int64_t>(wrapped.ByteSizeLong()) > hard_limit) {
        result.error_message = "VRP request header exceeds the maximum message size";
        return result;
      }
      chunk_other_arrays = true;
    }
    auto build_chunks = [&](const std::string& upload_id, int64_t chunk_data_budget) {
      std::vector<cuopt::remote::SendArrayChunkRequest> chunks;
      append_routing_matrix_chunks(chunks, problem, upload_id, chunk_data_budget);
      if (chunk_other_arrays) {
        append_routing_nonmatrix_chunks(chunks, problem, upload_id, chunk_data_budget);
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
