/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "grpc_problem_mapper.hpp"

#include <cuopt/linear_programming/constants.h>
#include <cuopt_remote.pb.h>
#include <cuopt_remote_service.pb.h>
#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include "grpc_settings_mapper.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cuopt::linear_programming {

namespace {
#include "generated_enum_converters_problem.inc"

template <typename T>
void chunk_typed_array(std::vector<cuopt::remote::SendArrayChunkRequest>& out,
                       cuopt::remote::ArrayFieldId field_id,
                       const std::vector<T>& data,
                       const std::string& upload_id,
                       int64_t chunk_data_budget)
{
  if (data.empty()) return;

  const int64_t elem_size      = static_cast<int64_t>(sizeof(T));
  const int64_t total_elements = static_cast<int64_t>(data.size());

  int64_t elems_per_chunk = chunk_data_budget / elem_size;
  if (elems_per_chunk <= 0) elems_per_chunk = 1;

  const auto* raw = reinterpret_cast<const uint8_t*>(data.data());

  for (int64_t offset = 0; offset < total_elements; offset += elems_per_chunk) {
    int64_t count       = std::min(elems_per_chunk, total_elements - offset);
    int64_t byte_offset = offset * elem_size;
    int64_t byte_count  = count * elem_size;

    cuopt::remote::SendArrayChunkRequest req;
    req.set_upload_id(upload_id);
    auto* ac = req.mutable_chunk();
    ac->set_field_id(field_id);
    ac->set_element_offset(offset);
    ac->set_total_elements(total_elements);
    ac->set_data(raw + byte_offset, byte_count);
    out.push_back(std::move(req));
  }
}

void chunk_byte_blob(std::vector<cuopt::remote::SendArrayChunkRequest>& out,
                     cuopt::remote::ArrayFieldId field_id,
                     const std::vector<uint8_t>& data,
                     const std::string& upload_id,
                     int64_t chunk_data_budget)
{
  chunk_typed_array(out, field_id, data, upload_id, chunk_data_budget);
}

// Container variant of chunk_typed_array: slices `data` and stamps each chunk
// with (container_field_num, container_index, container-relative field_id) so
// the server can reassemble it into a specific entry of a repeated_messages
// field (e.g. a single QuadraticConstraint's linear_values).  `field_id` here
// is container-relative (0..N-1), not an ArrayFieldId.
template <typename T>
void chunk_container_typed_array(std::vector<cuopt::remote::SendArrayChunkRequest>& out,
                                 int32_t container_field_num,
                                 int32_t container_index,
                                 int32_t field_id,
                                 const std::vector<T>& data,
                                 const std::string& upload_id,
                                 int64_t chunk_data_budget)
{
  if (data.empty()) return;

  const int64_t elem_size      = static_cast<int64_t>(sizeof(T));
  const int64_t total_elements = static_cast<int64_t>(data.size());

  int64_t elems_per_chunk = chunk_data_budget / elem_size;
  if (elems_per_chunk <= 0) elems_per_chunk = 1;

  const auto* raw = reinterpret_cast<const uint8_t*>(data.data());

  for (int64_t offset = 0; offset < total_elements; offset += elems_per_chunk) {
    int64_t count       = std::min(elems_per_chunk, total_elements - offset);
    int64_t byte_offset = offset * elem_size;
    int64_t byte_count  = count * elem_size;

    cuopt::remote::SendArrayChunkRequest req;
    req.set_upload_id(upload_id);
    auto* ac = req.mutable_chunk();
    ac->set_field_id(field_id);
    ac->set_element_offset(offset);
    ac->set_total_elements(total_elements);
    ac->set_data(raw + byte_offset, byte_count);
    ac->set_container_field_num(container_field_num);
    ac->set_container_index(container_index);
    out.push_back(std::move(req));
  }
}

std::vector<uint8_t> names_to_blob(const std::vector<std::string>& names)
{
  if (names.empty()) return {};
  size_t total = 0;
  for (const auto& n : names)
    total += n.size() + 1;
  std::vector<uint8_t> blob(total);
  size_t pos = 0;
  for (const auto& n : names) {
    std::memcpy(blob.data() + pos, n.data(), n.size());
    pos += n.size();
    blob[pos++] = '\0';
  }
  return blob;
}

}  // namespace

template <typename i_t, typename f_t>
void map_problem_to_proto(const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
                          cuopt::remote::OptimizationProblem* pb_problem)
{
#include "generated_problem_to_proto.inc"
}

template <typename i_t, typename f_t>
void map_proto_to_problem(const cuopt::remote::OptimizationProblem& pb_problem,
                          cpu_optimization_problem_t<i_t, f_t>& cpu_problem)
{
#include "generated_proto_to_problem.inc"
}

// ============================================================================
// Size estimation
// ============================================================================

template <typename i_t, typename f_t>
size_t estimate_problem_proto_size(const cpu_optimization_problem_t<i_t, f_t>& cpu_problem)
{
#include "generated_estimate_problem_size.inc"
}

// ============================================================================
// Chunked header population (client-side, for CHUNKED_ARRAYS upload)
// ============================================================================

template <typename i_t, typename f_t>
void populate_chunked_header_lp(const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
                                const pdlp_solver_settings_t<i_t, f_t>& settings,
                                cuopt::remote::ChunkedProblemHeader* header)
{
#include "generated_populate_chunked_header_lp.inc"
}

template <typename i_t, typename f_t>
void populate_chunked_header_mip(const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
                                 const mip_solver_settings_t<i_t, f_t>& settings,
                                 bool enable_incumbents,
                                 cuopt::remote::ChunkedProblemHeader* header)
{
#include "generated_populate_chunked_header_mip.inc"
}

// ============================================================================
// Chunked header reconstruction (server-side)
// ============================================================================

template <typename i_t, typename f_t>
void map_chunked_header_to_problem(const cuopt::remote::ChunkedProblemHeader& header,
                                   cpu_optimization_problem_t<i_t, f_t>& cpu_problem)
{
#include "generated_chunked_header_to_problem.inc"
}

// ============================================================================
// Chunked array reconstruction (server-side, consolidates all array mapping)
// ============================================================================

template <typename i_t, typename f_t>
void map_chunked_arrays_to_problem(
  const cuopt::remote::ChunkedProblemHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  const std::map<container_array_key_t, std::vector<uint8_t>>& container_arrays,
  cpu_optimization_problem_t<i_t, f_t>& cpu_problem)
{
#include "generated_chunked_arrays_to_problem.inc"
}

// =============================================================================
// Chunked array request building (client-side)
// =============================================================================

template <typename i_t, typename f_t>
std::vector<cuopt::remote::SendArrayChunkRequest> build_array_chunk_requests(
  const cpu_optimization_problem_t<i_t, f_t>& problem,
  const std::string& upload_id,
  int64_t chunk_size_bytes)
{
#include "generated_build_array_chunks.inc"
}

// Explicit template instantiations
#if CUOPT_INSTANTIATE_FLOAT
template void map_problem_to_proto(const cpu_optimization_problem_t<int32_t, float>& cpu_problem,
                                   cuopt::remote::OptimizationProblem* pb_problem);
template void map_proto_to_problem(const cuopt::remote::OptimizationProblem& pb_problem,
                                   cpu_optimization_problem_t<int32_t, float>& cpu_problem);
template size_t estimate_problem_proto_size(
  const cpu_optimization_problem_t<int32_t, float>& cpu_problem);
template void populate_chunked_header_lp(
  const cpu_optimization_problem_t<int32_t, float>& cpu_problem,
  const pdlp_solver_settings_t<int32_t, float>& settings,
  cuopt::remote::ChunkedProblemHeader* header);
template void populate_chunked_header_mip(
  const cpu_optimization_problem_t<int32_t, float>& cpu_problem,
  const mip_solver_settings_t<int32_t, float>& settings,
  bool enable_incumbents,
  cuopt::remote::ChunkedProblemHeader* header);
template void map_chunked_header_to_problem(
  const cuopt::remote::ChunkedProblemHeader& header,
  cpu_optimization_problem_t<int32_t, float>& cpu_problem);
template void map_chunked_arrays_to_problem(
  const cuopt::remote::ChunkedProblemHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  const std::map<container_array_key_t, std::vector<uint8_t>>& container_arrays,
  cpu_optimization_problem_t<int32_t, float>& cpu_problem);
template std::vector<cuopt::remote::SendArrayChunkRequest> build_array_chunk_requests(
  const cpu_optimization_problem_t<int32_t, float>& problem,
  const std::string& upload_id,
  int64_t chunk_size_bytes);
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template void map_problem_to_proto(const cpu_optimization_problem_t<int32_t, double>& cpu_problem,
                                   cuopt::remote::OptimizationProblem* pb_problem);
template void map_proto_to_problem(const cuopt::remote::OptimizationProblem& pb_problem,
                                   cpu_optimization_problem_t<int32_t, double>& cpu_problem);
template size_t estimate_problem_proto_size(
  const cpu_optimization_problem_t<int32_t, double>& cpu_problem);
template void populate_chunked_header_lp(
  const cpu_optimization_problem_t<int32_t, double>& cpu_problem,
  const pdlp_solver_settings_t<int32_t, double>& settings,
  cuopt::remote::ChunkedProblemHeader* header);
template void populate_chunked_header_mip(
  const cpu_optimization_problem_t<int32_t, double>& cpu_problem,
  const mip_solver_settings_t<int32_t, double>& settings,
  bool enable_incumbents,
  cuopt::remote::ChunkedProblemHeader* header);
template void map_chunked_header_to_problem(
  const cuopt::remote::ChunkedProblemHeader& header,
  cpu_optimization_problem_t<int32_t, double>& cpu_problem);
template void map_chunked_arrays_to_problem(
  const cuopt::remote::ChunkedProblemHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  const std::map<container_array_key_t, std::vector<uint8_t>>& container_arrays,
  cpu_optimization_problem_t<int32_t, double>& cpu_problem);
template std::vector<cuopt::remote::SendArrayChunkRequest> build_array_chunk_requests(
  const cpu_optimization_problem_t<int32_t, double>& problem,
  const std::string& upload_id,
  int64_t chunk_size_bytes);
#endif

}  // namespace cuopt::linear_programming
