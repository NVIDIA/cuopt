/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "grpc_problem_mapper.hpp"

#include <cuopt/export.hpp>

#include <cuopt/mathematical_optimization/constants.h>
#include <cuopt_remote.pb.h>

#include <cuopt_remote_service.pb.h>
#include <google/protobuf/descriptor.h>
#include <cuopt/mathematical_optimization/cpu_optimization_problem.hpp>
#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <mps_parser_internal.hpp>
#include "grpc_container_chunk.hpp"
#include "grpc_settings_mapper.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cuopt::mathematical_optimization {

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
#include "generated_problem_scalars_from_proto.inc"
#include "generated_proto_arrays_to_problem.inc"
}

template <typename i_t, typename f_t>
void map_proto_to_problem(
  const cuopt::remote::OptimizationProblem& problem,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  const std::map<container_array_key_t, std::vector<uint8_t>>& container_arrays,
  cpu_optimization_problem_t<i_t, f_t>& cpu_problem)
{
  const auto& pb_problem = problem;
#include "generated_problem_scalars_from_proto.inc"
  const auto& chunked_scalars = problem;
#include "generated_chunked_arrays_to_problem.inc"
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
                                 bool enable_set_incumbent,
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
  const auto& pb_problem = header;
#include "generated_chunked_header_to_problem.inc"
#include "generated_problem_scalars_from_proto.inc"
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
  map_chunked_header_to_problem(header, cpu_problem);
  const auto& chunked_scalars = header;
#include "generated_chunked_arrays_to_problem.inc"
}

namespace {

bool stripped_child_arrays_present(const google::protobuf::Message& child)
{
  const auto* child_desc = child.GetDescriptor();
  const auto* child_refl = child.GetReflection();
  for (int k = 0; k < child_desc->field_count(); ++k) {
    const auto* cf = child_desc->field(k);
    if (!cf->is_repeated()) {
      if (cf->type() == google::protobuf::FieldDescriptor::TYPE_BYTES &&
          !child_refl->GetString(child, cf).empty()) {
        return true;
      }
      continue;
    }
    if (cf->cpp_type() != google::protobuf::FieldDescriptor::CPPTYPE_STRING &&
        cf->cpp_type() != google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE &&
        child_refl->FieldSize(child, cf) > 0) {
      return true;
    }
  }
  return false;
}

}  // namespace

bool CUOPT_EXPORT inline_chunked_arrays_present(const cuopt::remote::OptimizationProblem& problem)
{
  const auto* desc = problem.GetDescriptor();
  const auto* refl = problem.GetReflection();
  for (int i = 0; i < desc->field_count(); ++i) {
    const auto* field = desc->field(i);
    if (field->is_repeated() &&
        field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE) {
      const int n = refl->FieldSize(problem, field);
      for (int j = 0; j < n; ++j) {
        if (stripped_child_arrays_present(refl->GetRepeatedMessage(problem, field, j))) {
          return true;
        }
      }
    } else if (field->is_repeated()) {
      if (refl->FieldSize(problem, field) > 0) return true;
    } else if (field->type() == google::protobuf::FieldDescriptor::TYPE_BYTES &&
               !refl->GetString(problem, field).empty()) {
      return true;
    }
  }
  return false;
}

void strip_chunked_numeric_arrays(cuopt::remote::OptimizationProblem* problem)
{
  // One level into repeated messages (quadratic constraints): clear their
  // numeric arrays, keep the per-entry scalars. Do not walk singular
  // submessages; OptimizationProblem has none that carry chunked arrays.
  const auto* desc = problem->GetDescriptor();
  const auto* refl = problem->GetReflection();
  for (int i = 0; i < desc->field_count(); ++i) {
    const auto* field = desc->field(i);
    if (field->is_repeated() &&
        field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE) {
      const int n = refl->FieldSize(*problem, field);
      for (int j = 0; j < n; ++j) {
        auto* child            = refl->MutableRepeatedMessage(problem, field, j);
        const auto* child_desc = child->GetDescriptor();
        const auto* child_refl = child->GetReflection();
        for (int k = 0; k < child_desc->field_count(); ++k) {
          const auto* cf = child_desc->field(k);
          if (!cf->is_repeated()) {
            if (cf->type() == google::protobuf::FieldDescriptor::TYPE_BYTES) {
              child_refl->ClearField(child, cf);
            }
            continue;
          }
          if (cf->cpp_type() != google::protobuf::FieldDescriptor::CPPTYPE_STRING &&
              cf->cpp_type() != google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE) {
            child_refl->ClearField(child, cf);
          }
        }
      }
    } else if (field->is_repeated()) {
      refl->ClearField(problem, field);
    } else if (field->type() == google::protobuf::FieldDescriptor::TYPE_BYTES) {
      refl->ClearField(problem, field);
    }
  }
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
template CUOPT_EXPORT void map_problem_to_proto(
  const cpu_optimization_problem_t<int32_t, float>& cpu_problem,
  cuopt::remote::OptimizationProblem* pb_problem);
template CUOPT_EXPORT void map_proto_to_problem(
  const cuopt::remote::OptimizationProblem& pb_problem,
  cpu_optimization_problem_t<int32_t, float>& cpu_problem);
template CUOPT_EXPORT size_t
estimate_problem_proto_size(const cpu_optimization_problem_t<int32_t, float>& cpu_problem);
template CUOPT_EXPORT void populate_chunked_header_lp(
  const cpu_optimization_problem_t<int32_t, float>& cpu_problem,
  const pdlp_solver_settings_t<int32_t, float>& settings,
  cuopt::remote::ChunkedProblemHeader* header);
template CUOPT_EXPORT void populate_chunked_header_mip(
  const cpu_optimization_problem_t<int32_t, float>& cpu_problem,
  const mip_solver_settings_t<int32_t, float>& settings,
  bool enable_incumbents,
  bool enable_set_incumbent,
  cuopt::remote::ChunkedProblemHeader* header);
template CUOPT_EXPORT void map_chunked_header_to_problem(
  const cuopt::remote::ChunkedProblemHeader& header,
  cpu_optimization_problem_t<int32_t, float>& cpu_problem);
template CUOPT_EXPORT void map_chunked_arrays_to_problem(
  const cuopt::remote::ChunkedProblemHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  const std::map<container_array_key_t, std::vector<uint8_t>>& container_arrays,
  cpu_optimization_problem_t<int32_t, float>& cpu_problem);
template CUOPT_EXPORT void map_proto_to_problem(
  const cuopt::remote::OptimizationProblem& problem,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  const std::map<container_array_key_t, std::vector<uint8_t>>& container_arrays,
  cpu_optimization_problem_t<int32_t, float>& cpu_problem);
template CUOPT_EXPORT std::vector<cuopt::remote::SendArrayChunkRequest> build_array_chunk_requests(
  const cpu_optimization_problem_t<int32_t, float>& problem,
  const std::string& upload_id,
  int64_t chunk_size_bytes);
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template CUOPT_EXPORT void map_problem_to_proto(
  const cpu_optimization_problem_t<int32_t, double>& cpu_problem,
  cuopt::remote::OptimizationProblem* pb_problem);
template CUOPT_EXPORT void map_proto_to_problem(
  const cuopt::remote::OptimizationProblem& pb_problem,
  cpu_optimization_problem_t<int32_t, double>& cpu_problem);
template CUOPT_EXPORT size_t
estimate_problem_proto_size(const cpu_optimization_problem_t<int32_t, double>& cpu_problem);
template CUOPT_EXPORT void populate_chunked_header_lp(
  const cpu_optimization_problem_t<int32_t, double>& cpu_problem,
  const pdlp_solver_settings_t<int32_t, double>& settings,
  cuopt::remote::ChunkedProblemHeader* header);
template CUOPT_EXPORT void populate_chunked_header_mip(
  const cpu_optimization_problem_t<int32_t, double>& cpu_problem,
  const mip_solver_settings_t<int32_t, double>& settings,
  bool enable_incumbents,
  bool enable_set_incumbent,
  cuopt::remote::ChunkedProblemHeader* header);
template CUOPT_EXPORT void map_chunked_header_to_problem(
  const cuopt::remote::ChunkedProblemHeader& header,
  cpu_optimization_problem_t<int32_t, double>& cpu_problem);
template CUOPT_EXPORT void map_chunked_arrays_to_problem(
  const cuopt::remote::ChunkedProblemHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  const std::map<container_array_key_t, std::vector<uint8_t>>& container_arrays,
  cpu_optimization_problem_t<int32_t, double>& cpu_problem);
template CUOPT_EXPORT void map_proto_to_problem(
  const cuopt::remote::OptimizationProblem& problem,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  const std::map<container_array_key_t, std::vector<uint8_t>>& container_arrays,
  cpu_optimization_problem_t<int32_t, double>& cpu_problem);
template CUOPT_EXPORT std::vector<cuopt::remote::SendArrayChunkRequest> build_array_chunk_requests(
  const cpu_optimization_problem_t<int32_t, double>& problem,
  const std::string& upload_id,
  int64_t chunk_size_bytes);
#endif

}  // namespace cuopt::mathematical_optimization
