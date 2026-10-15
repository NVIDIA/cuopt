/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "grpc_solution_mapper.hpp"

#include <cuopt/export.hpp>

#include <cuopt/mathematical_optimization/constants.h>
#include <cuopt_remote.pb.h>
#include <cuopt_remote_service.pb.h>
#include <cuopt/mathematical_optimization/cpu_optimization_problem_solution.hpp>
#include <cuopt/mathematical_optimization/mip/solver_solution.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_solution.hpp>

#include <cstring>
#include <map>
#include <stdexcept>
#include <string>

namespace cuopt::mathematical_optimization {

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

namespace {

void assign_doubles(google::protobuf::RepeatedField<double>* dst, const std::vector<uint8_t>& raw)
{
  if (raw.size() % sizeof(double) != 0) {
    throw std::invalid_argument("result chunk payload is not a multiple of sizeof(double)");
  }
  const auto* p  = reinterpret_cast<const double*>(raw.data());
  const size_t n = raw.size() / sizeof(double);
  dst->Clear();
  dst->Reserve(static_cast<int>(n));
  for (size_t i = 0; i < n; ++i) {
    dst->Add(p[i]);
  }
}

const std::vector<uint8_t>* find_array(const std::map<int32_t, std::vector<uint8_t>>& arrays,
                                       cuopt::remote::ResultFieldId id)
{
  auto it = arrays.find(static_cast<int32_t>(id));
  if (it == arrays.end() || it->second.empty()) return nullptr;
  return &it->second;
}

}  // namespace

void CUOPT_EXPORT clear_lp_solution_arrays(cuopt::remote::LPSolution* solution)
{
  solution->clear_primal_solution();
  solution->clear_dual_solution();
  solution->clear_reduced_cost();
  if (solution->has_warm_start_data()) {
    auto* ws = solution->mutable_warm_start_data();
    ws->clear_current_primal_solution();
    ws->clear_current_dual_solution();
    ws->clear_initial_primal_average();
    ws->clear_initial_dual_average();
    ws->clear_current_aty();
    ws->clear_sum_primal_solutions();
    ws->clear_sum_dual_solutions();
    ws->clear_last_restart_duality_gap_primal_solution();
    ws->clear_last_restart_duality_gap_dual_solution();
  }
}

void CUOPT_EXPORT clear_mip_solution_arrays(cuopt::remote::MIPSolution* solution)
{
  solution->clear_mip_solution();
}

void CUOPT_EXPORT apply_result_chunk_arrays(cuopt::remote::ResultResponse* result,
                                            const std::map<int32_t, std::vector<uint8_t>>& arrays)
{
  if (result->has_lp_solution()) {
    auto* sol = result->mutable_lp_solution();
    if (const auto* raw = find_array(arrays, cuopt::remote::RESULT_PRIMAL_SOLUTION)) {
      assign_doubles(sol->mutable_primal_solution(), *raw);
    }
    if (const auto* raw = find_array(arrays, cuopt::remote::RESULT_DUAL_SOLUTION)) {
      assign_doubles(sol->mutable_dual_solution(), *raw);
    }
    if (const auto* raw = find_array(arrays, cuopt::remote::RESULT_REDUCED_COST)) {
      assign_doubles(sol->mutable_reduced_cost(), *raw);
    }
    if (sol->has_warm_start_data()) {
      auto* ws = sol->mutable_warm_start_data();
      if (const auto* raw = find_array(arrays, cuopt::remote::RESULT_WS_CURRENT_PRIMAL_SOLUTION)) {
        assign_doubles(ws->mutable_current_primal_solution(), *raw);
      }
      if (const auto* raw = find_array(arrays, cuopt::remote::RESULT_WS_CURRENT_DUAL_SOLUTION)) {
        assign_doubles(ws->mutable_current_dual_solution(), *raw);
      }
      if (const auto* raw = find_array(arrays, cuopt::remote::RESULT_WS_INITIAL_PRIMAL_AVERAGE)) {
        assign_doubles(ws->mutable_initial_primal_average(), *raw);
      }
      if (const auto* raw = find_array(arrays, cuopt::remote::RESULT_WS_INITIAL_DUAL_AVERAGE)) {
        assign_doubles(ws->mutable_initial_dual_average(), *raw);
      }
      if (const auto* raw = find_array(arrays, cuopt::remote::RESULT_WS_CURRENT_ATY)) {
        assign_doubles(ws->mutable_current_aty(), *raw);
      }
      if (const auto* raw = find_array(arrays, cuopt::remote::RESULT_WS_SUM_PRIMAL_SOLUTIONS)) {
        assign_doubles(ws->mutable_sum_primal_solutions(), *raw);
      }
      if (const auto* raw = find_array(arrays, cuopt::remote::RESULT_WS_SUM_DUAL_SOLUTIONS)) {
        assign_doubles(ws->mutable_sum_dual_solutions(), *raw);
      }
      if (const auto* raw =
            find_array(arrays, cuopt::remote::RESULT_WS_LAST_RESTART_DUALITY_GAP_PRIMAL_SOLUTION)) {
        assign_doubles(ws->mutable_last_restart_duality_gap_primal_solution(), *raw);
      }
      if (const auto* raw =
            find_array(arrays, cuopt::remote::RESULT_WS_LAST_RESTART_DUALITY_GAP_DUAL_SOLUTION)) {
        assign_doubles(ws->mutable_last_restart_duality_gap_dual_solution(), *raw);
      }
    }
  } else if (result->has_mip_solution()) {
    if (const auto* raw = find_array(arrays, cuopt::remote::RESULT_MIP_SOLUTION)) {
      assign_doubles(result->mutable_mip_solution()->mutable_mip_solution(), *raw);
    }
  }
}

// Explicit template instantiations
#if CUOPT_INSTANTIATE_FLOAT
template CUOPT_EXPORT void map_lp_solution_to_proto(
  const cpu_lp_solution_t<int32_t, float>& solution, cuopt::remote::LPSolution* pb_solution);
template CUOPT_EXPORT cpu_lp_solution_t<int32_t, float> map_proto_to_lp_solution(
  const cuopt::remote::LPSolution& pb_solution);
template CUOPT_EXPORT void map_mip_solution_to_proto(
  const cpu_mip_solution_t<int32_t, float>& solution, cuopt::remote::MIPSolution* pb_solution);
template CUOPT_EXPORT cpu_mip_solution_t<int32_t, float> map_proto_to_mip_solution(
  const cuopt::remote::MIPSolution& pb_solution);
template CUOPT_EXPORT void populate_chunked_result_header_lp(
  const cpu_lp_solution_t<int32_t, float>& solution, cuopt::remote::ChunkedResultHeader* header);
template CUOPT_EXPORT void populate_chunked_result_header_mip(
  const cpu_mip_solution_t<int32_t, float>& solution, cuopt::remote::ChunkedResultHeader* header);
template CUOPT_EXPORT std::map<int32_t, std::vector<uint8_t>> collect_lp_solution_arrays(
  const cpu_lp_solution_t<int32_t, float>& solution);
template CUOPT_EXPORT std::map<int32_t, std::vector<uint8_t>> collect_mip_solution_arrays(
  const cpu_mip_solution_t<int32_t, float>& solution);
template CUOPT_EXPORT cpu_lp_solution_t<int32_t, float> chunked_result_to_lp_solution(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays);
template CUOPT_EXPORT cpu_mip_solution_t<int32_t, float> chunked_result_to_mip_solution(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays);
template CUOPT_EXPORT void build_lp_solution_proto<int32_t, float>(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  cuopt::remote::LPSolution* proto);
template CUOPT_EXPORT void build_mip_solution_proto<int32_t, float>(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  cuopt::remote::MIPSolution* proto);
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template CUOPT_EXPORT void map_lp_solution_to_proto(
  const cpu_lp_solution_t<int32_t, double>& solution, cuopt::remote::LPSolution* pb_solution);
template CUOPT_EXPORT cpu_lp_solution_t<int32_t, double> map_proto_to_lp_solution(
  const cuopt::remote::LPSolution& pb_solution);
template CUOPT_EXPORT void map_mip_solution_to_proto(
  const cpu_mip_solution_t<int32_t, double>& solution, cuopt::remote::MIPSolution* pb_solution);
template CUOPT_EXPORT cpu_mip_solution_t<int32_t, double> map_proto_to_mip_solution(
  const cuopt::remote::MIPSolution& pb_solution);
template CUOPT_EXPORT void populate_chunked_result_header_lp(
  const cpu_lp_solution_t<int32_t, double>& solution, cuopt::remote::ChunkedResultHeader* header);
template CUOPT_EXPORT void populate_chunked_result_header_mip(
  const cpu_mip_solution_t<int32_t, double>& solution, cuopt::remote::ChunkedResultHeader* header);
template CUOPT_EXPORT std::map<int32_t, std::vector<uint8_t>> collect_lp_solution_arrays(
  const cpu_lp_solution_t<int32_t, double>& solution);
template CUOPT_EXPORT std::map<int32_t, std::vector<uint8_t>> collect_mip_solution_arrays(
  const cpu_mip_solution_t<int32_t, double>& solution);
template CUOPT_EXPORT cpu_lp_solution_t<int32_t, double> chunked_result_to_lp_solution(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays);
template CUOPT_EXPORT cpu_mip_solution_t<int32_t, double> chunked_result_to_mip_solution(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays);
template CUOPT_EXPORT void build_lp_solution_proto<int32_t, double>(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  cuopt::remote::LPSolution* proto);
template CUOPT_EXPORT void build_mip_solution_proto<int32_t, double>(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  cuopt::remote::MIPSolution* proto);
#endif

}  // namespace cuopt::mathematical_optimization
