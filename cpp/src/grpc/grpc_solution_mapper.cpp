/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "grpc_solution_mapper.hpp"

#include <cuopt/linear_programming/constants.h>
#include <cuopt_remote.pb.h>
#include <cuopt_remote_service.pb.h>
#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>

#include <cstring>
#include <map>
#include <stdexcept>
#include <string>

namespace cuopt::linear_programming {

namespace {
#include "generated_enum_converters_solution.inc"

void add_result_array_descriptor(cuopt::remote::ChunkedResultHeader* header,
                                 cuopt::remote::ResultFieldId fid,
                                 int64_t count,
                                 int64_t elem_size)
{
  if (count <= 0) return;
  auto* desc = header->add_arrays();
  desc->set_field_id(fid);
  desc->set_total_elements(count);
  desc->set_element_size_bytes(elem_size);
}

template <typename f_t>
std::vector<uint8_t> doubles_to_bytes(const std::vector<f_t>& vec)
{
  std::vector<double> tmp(vec.begin(), vec.end());
  std::vector<uint8_t> bytes(tmp.size() * sizeof(double));
  std::memcpy(bytes.data(), tmp.data(), bytes.size());
  return bytes;
}

template <typename T>
std::vector<T> bytes_to_typed(const std::map<int32_t, std::vector<uint8_t>>& arrays,
                              int32_t field_id)
{
  // An absent entry or empty payload means "field not transmitted" — callers
  // use the empty return to distinguish absence. A present-but-misaligned
  // payload is corrupt data and must fail loudly instead of silently masking.
  auto it = arrays.find(field_id);
  if (it == arrays.end() || it->second.empty()) return {};

  const auto& raw    = it->second;
  auto check_aligned = [&](size_t elem_size) {
    if (raw.size() % elem_size != 0) {
      throw std::invalid_argument("bytes_to_typed: payload size " + std::to_string(raw.size()) +
                                  " for field_id " + std::to_string(field_id) +
                                  " is not a multiple of element size " +
                                  std::to_string(elem_size));
    }
  };

  if constexpr (std::is_same_v<T, float>) {
    check_aligned(sizeof(double));
    size_t n = raw.size() / sizeof(double);
    std::vector<double> tmp(n);
    std::memcpy(tmp.data(), raw.data(), n * sizeof(double));
    return std::vector<T>(tmp.begin(), tmp.end());
  } else if constexpr (std::is_same_v<T, double>) {
    check_aligned(sizeof(double));
    size_t n = raw.size() / sizeof(double);
    std::vector<double> v(n);
    std::memcpy(v.data(), raw.data(), n * sizeof(double));
    return v;
  } else {
    check_aligned(sizeof(T));
    size_t n = raw.size() / sizeof(T);
    std::vector<T> v(n);
    std::memcpy(v.data(), raw.data(), n * sizeof(T));
    return v;
  }
}

}  // namespace

template <typename i_t, typename f_t>
void map_lp_solution_to_proto(const cpu_lp_solution_t<i_t, f_t>& solution,
                              cuopt::remote::LPSolution* pb_solution)
{
#include "generated_lp_solution_to_proto.inc"
}

template <typename i_t, typename f_t>
cpu_lp_solution_t<i_t, f_t> map_proto_to_lp_solution(const cuopt::remote::LPSolution& pb_solution)
{
#include "generated_proto_to_lp_solution.inc"
}

template <typename i_t, typename f_t>
void map_mip_solution_to_proto(const cpu_mip_solution_t<i_t, f_t>& solution,
                               cuopt::remote::MIPSolution* pb_solution)
{
#include "generated_mip_solution_to_proto.inc"
}

template <typename i_t, typename f_t>
cpu_mip_solution_t<i_t, f_t> map_proto_to_mip_solution(
  const cuopt::remote::MIPSolution& pb_solution)
{
#include "generated_proto_to_mip_solution.inc"
}

// ============================================================================
// Chunked result header population
// ============================================================================

template <typename i_t, typename f_t>
void populate_chunked_result_header_lp(const cpu_lp_solution_t<i_t, f_t>& solution,
                                       cuopt::remote::ChunkedResultHeader* header)
{
#include "generated_lp_chunked_header.inc"
}

template <typename i_t, typename f_t>
void populate_chunked_result_header_mip(const cpu_mip_solution_t<i_t, f_t>& solution,
                                        cuopt::remote::ChunkedResultHeader* header)
{
#include "generated_mip_chunked_header.inc"
}

// ============================================================================
// Collect solution arrays as raw bytes
// ============================================================================

template <typename i_t, typename f_t>
std::map<int32_t, std::vector<uint8_t>> collect_lp_solution_arrays(
  const cpu_lp_solution_t<i_t, f_t>& solution)
{
#include "generated_collect_lp_arrays.inc"
}

template <typename i_t, typename f_t>
std::map<int32_t, std::vector<uint8_t>> collect_mip_solution_arrays(
  const cpu_mip_solution_t<i_t, f_t>& solution)
{
#include "generated_collect_mip_arrays.inc"
}

// ============================================================================
// Chunked result -> solution (client-side)
// ============================================================================

template <typename i_t, typename f_t>
cpu_lp_solution_t<i_t, f_t> chunked_result_to_lp_solution(
  const cuopt::remote::ChunkedResultHeader& h,
  const std::map<int32_t, std::vector<uint8_t>>& arrays)
{
#include "generated_chunked_to_lp_solution.inc"
}

template <typename i_t, typename f_t>
cpu_mip_solution_t<i_t, f_t> chunked_result_to_mip_solution(
  const cuopt::remote::ChunkedResultHeader& h,
  const std::map<int32_t, std::vector<uint8_t>>& arrays)
{
#include "generated_chunked_to_mip_solution.inc"
}

// ============================================================================
// Build full protobuf from stored header + arrays (server-side GetResult RPC)
// ============================================================================

template <typename i_t, typename f_t>
void build_lp_solution_proto(const cuopt::remote::ChunkedResultHeader& header,
                             const std::map<int32_t, std::vector<uint8_t>>& arrays,
                             cuopt::remote::LPSolution* proto)
{
  auto cpu_sol = chunked_result_to_lp_solution<i_t, f_t>(header, arrays);
  map_lp_solution_to_proto(cpu_sol, proto);
}

template <typename i_t, typename f_t>
void build_mip_solution_proto(const cuopt::remote::ChunkedResultHeader& header,
                              const std::map<int32_t, std::vector<uint8_t>>& arrays,
                              cuopt::remote::MIPSolution* proto)
{
  auto cpu_sol = chunked_result_to_mip_solution<i_t, f_t>(header, arrays);
  map_mip_solution_to_proto(cpu_sol, proto);
}

// Explicit template instantiations
#if CUOPT_INSTANTIATE_FLOAT
template void map_lp_solution_to_proto(const cpu_lp_solution_t<int32_t, float>& solution,
                                       cuopt::remote::LPSolution* pb_solution);
template cpu_lp_solution_t<int32_t, float> map_proto_to_lp_solution(
  const cuopt::remote::LPSolution& pb_solution);
template void map_mip_solution_to_proto(const cpu_mip_solution_t<int32_t, float>& solution,
                                        cuopt::remote::MIPSolution* pb_solution);
template cpu_mip_solution_t<int32_t, float> map_proto_to_mip_solution(
  const cuopt::remote::MIPSolution& pb_solution);
template void populate_chunked_result_header_lp(const cpu_lp_solution_t<int32_t, float>& solution,
                                                cuopt::remote::ChunkedResultHeader* header);
template void populate_chunked_result_header_mip(const cpu_mip_solution_t<int32_t, float>& solution,
                                                 cuopt::remote::ChunkedResultHeader* header);
template std::map<int32_t, std::vector<uint8_t>> collect_lp_solution_arrays(
  const cpu_lp_solution_t<int32_t, float>& solution);
template std::map<int32_t, std::vector<uint8_t>> collect_mip_solution_arrays(
  const cpu_mip_solution_t<int32_t, float>& solution);
template cpu_lp_solution_t<int32_t, float> chunked_result_to_lp_solution(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays);
template cpu_mip_solution_t<int32_t, float> chunked_result_to_mip_solution(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays);
template void build_lp_solution_proto<int32_t, float>(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  cuopt::remote::LPSolution* proto);
template void build_mip_solution_proto<int32_t, float>(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  cuopt::remote::MIPSolution* proto);
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template void map_lp_solution_to_proto(const cpu_lp_solution_t<int32_t, double>& solution,
                                       cuopt::remote::LPSolution* pb_solution);
template cpu_lp_solution_t<int32_t, double> map_proto_to_lp_solution(
  const cuopt::remote::LPSolution& pb_solution);
template void map_mip_solution_to_proto(const cpu_mip_solution_t<int32_t, double>& solution,
                                        cuopt::remote::MIPSolution* pb_solution);
template cpu_mip_solution_t<int32_t, double> map_proto_to_mip_solution(
  const cuopt::remote::MIPSolution& pb_solution);
template void populate_chunked_result_header_lp(const cpu_lp_solution_t<int32_t, double>& solution,
                                                cuopt::remote::ChunkedResultHeader* header);
template void populate_chunked_result_header_mip(
  const cpu_mip_solution_t<int32_t, double>& solution, cuopt::remote::ChunkedResultHeader* header);
template std::map<int32_t, std::vector<uint8_t>> collect_lp_solution_arrays(
  const cpu_lp_solution_t<int32_t, double>& solution);
template std::map<int32_t, std::vector<uint8_t>> collect_mip_solution_arrays(
  const cpu_mip_solution_t<int32_t, double>& solution);
template cpu_lp_solution_t<int32_t, double> chunked_result_to_lp_solution(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays);
template cpu_mip_solution_t<int32_t, double> chunked_result_to_mip_solution(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays);
template void build_lp_solution_proto<int32_t, double>(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  cuopt::remote::LPSolution* proto);
template void build_mip_solution_proto<int32_t, double>(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  cuopt::remote::MIPSolution* proto);
#endif

}  // namespace cuopt::linear_programming
