/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt_remote.pb.h>
#include <cuopt_remote_service.pb.h>

#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace cuopt::linear_programming {

/**
 * @brief Map cpu_lp_solution_t to protobuf LPSolution message.
 *
 * Populates a protobuf message using the generated protobuf C++ API.
 * Does not perform serialization — that is handled by the protobuf library.
 */
template <typename i_t, typename f_t>
void map_lp_solution_to_proto(const cpu_lp_solution_t<i_t, f_t>& solution,
                              cuopt::remote::LPSolution* pb_solution);

/**
 * @brief Map protobuf LPSolution message to cpu_lp_solution_t.
 *
 * Reads from a protobuf message using the generated protobuf C++ API.
 * Does not perform deserialization — that is handled by the protobuf library.
 */
template <typename i_t, typename f_t>
cpu_lp_solution_t<i_t, f_t> map_proto_to_lp_solution(const cuopt::remote::LPSolution& pb_solution);

/**
 * @brief Map cpu_mip_solution_t to protobuf MIPSolution message.
 *
 * Populates a protobuf message using the generated protobuf C++ API.
 * Does not perform serialization — that is handled by the protobuf library.
 */
template <typename i_t, typename f_t>
void map_mip_solution_to_proto(const cpu_mip_solution_t<i_t, f_t>& solution,
                               cuopt::remote::MIPSolution* pb_solution);

/**
 * @brief Map protobuf MIPSolution message to cpu_mip_solution_t.
 *
 * Reads from a protobuf message using the generated protobuf C++ API.
 * Does not perform deserialization — that is handled by the protobuf library.
 */
template <typename i_t, typename f_t>
cpu_mip_solution_t<i_t, f_t> map_proto_to_mip_solution(
  const cuopt::remote::MIPSolution& pb_solution);

/**
 * @brief Convert cuOpt termination status to protobuf enum.
 * @param status cuOpt PDLP termination status
 * @return Protobuf PDLPTerminationStatus enum
 */
cuopt::remote::PDLPTerminationStatus to_proto_pdlp_status(pdlp_termination_status_t status);

/**
 * @brief Convert protobuf enum to cuOpt termination status.
 * @param status Protobuf PDLPTerminationStatus enum
 * @return cuOpt PDLP termination status
 */
pdlp_termination_status_t from_proto_pdlp_status(cuopt::remote::PDLPTerminationStatus status);

/**
 * @brief Convert cuOpt MIP termination status to protobuf enum.
 * @param status cuOpt MIP termination status
 * @return Protobuf MIPTerminationStatus enum
 */
cuopt::remote::MIPTerminationStatus to_proto_mip_status(mip_termination_status_t status);

/**
 * @brief Convert protobuf enum to cuOpt MIP termination status.
 * @param status Protobuf MIPTerminationStatus enum
 * @return cuOpt MIP termination status
 */
mip_termination_status_t from_proto_mip_status(cuopt::remote::MIPTerminationStatus status);

// ============================================================================
// Chunked result support (for results exceeding gRPC max message size)
// ============================================================================

/**
 * @brief Estimate serialized protobuf size of an LP solution.
 */
template <typename i_t, typename f_t>
size_t estimate_lp_solution_proto_size(const cpu_lp_solution_t<i_t, f_t>& solution);

/**
 * @brief Estimate serialized protobuf size of a MIP solution.
 */
template <typename i_t, typename f_t>
size_t estimate_mip_solution_proto_size(const cpu_mip_solution_t<i_t, f_t>& solution);

/**
 * @brief Populate a ChunkedResultHeader from an LP solution (scalar fields + array descriptors).
 */
template <typename i_t, typename f_t>
void populate_chunked_result_header_lp(const cpu_lp_solution_t<i_t, f_t>& solution,
                                       cuopt::remote::ChunkedResultHeader* header);

/**
 * @brief Populate a ChunkedResultHeader from a MIP solution (scalar fields + array descriptors).
 */
template <typename i_t, typename f_t>
void populate_chunked_result_header_mip(const cpu_mip_solution_t<i_t, f_t>& solution,
                                        cuopt::remote::ChunkedResultHeader* header);

/**
 * @brief Collect LP solution arrays as raw bytes keyed by ResultFieldId.
 *
 * Returns a map of ResultFieldId -> raw byte data (doubles packed as bytes).
 * Used by the worker to send chunked result data.
 */
template <typename i_t, typename f_t>
std::map<int32_t, std::vector<uint8_t>> collect_lp_solution_arrays(
  const cpu_lp_solution_t<i_t, f_t>& solution);

/**
 * @brief Collect MIP solution arrays as raw bytes keyed by ResultFieldId.
 */
template <typename i_t, typename f_t>
std::map<int32_t, std::vector<uint8_t>> collect_mip_solution_arrays(
  const cpu_mip_solution_t<i_t, f_t>& solution);

// ============================================================================
// Chunked result -> solution (for gRPC client)
// ============================================================================

/**
 * @brief Reconstruct a cpu_lp_solution_t from chunked result header and raw array data.
 *
 * This is the client-side counterpart to collect_lp_solution_arrays +
 * populate_chunked_result_header_lp. It reads scalars from the header and typed arrays from the
 * byte map.
 */
template <typename i_t, typename f_t>
cpu_lp_solution_t<i_t, f_t> chunked_result_to_lp_solution(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays);

/**
 * @brief Reconstruct a cpu_mip_solution_t from chunked result header and raw array data.
 */
template <typename i_t, typename f_t>
cpu_mip_solution_t<i_t, f_t> chunked_result_to_mip_solution(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays);

// ============================================================================
// Build full protobuf solution from stored header + arrays (server-side GetResult RPC)
// ============================================================================

/**
 * @brief Build a full LPSolution protobuf from a ChunkedResultHeader and raw arrays.
 *
 * Used by the server's GetResult RPC to serve unary responses.
 * Composes chunked_result_to_lp_solution + map_lp_solution_to_proto.
 */
template <typename i_t, typename f_t>
void build_lp_solution_proto(const cuopt::remote::ChunkedResultHeader& header,
                             const std::map<int32_t, std::vector<uint8_t>>& arrays,
                             cuopt::remote::LPSolution* proto);

/**
 * @brief Build a full MIPSolution protobuf from a ChunkedResultHeader and raw arrays.
 */
template <typename i_t, typename f_t>
void build_mip_solution_proto(const cuopt::remote::ChunkedResultHeader& header,
                              const std::map<int32_t, std::vector<uint8_t>>& arrays,
                              cuopt::remote::MIPSolution* proto);

}  // namespace cuopt::linear_programming
