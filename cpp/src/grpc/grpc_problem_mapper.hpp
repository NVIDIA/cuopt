/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt_remote.pb.h>
#include <cuopt_remote_service.pb.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace cuopt::remote {
class ChunkedProblemHeader;
}

namespace cuopt::linear_programming {

// Forward declarations
template <typename i_t, typename f_t>
class cpu_optimization_problem_t;

template <typename i_t, typename f_t>
struct pdlp_solver_settings_t;

template <typename i_t, typename f_t>
struct mip_solver_settings_t;

/**
 * @brief Map cpu_optimization_problem_t to protobuf OptimizationProblem message.
 *
 * Populates a protobuf message using the generated protobuf C++ API.
 * Does not perform serialization — that is handled by the protobuf library.
 */
template <typename i_t, typename f_t>
void map_problem_to_proto(const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
                          cuopt::remote::OptimizationProblem* pb_problem);

/**
 * @brief Map protobuf OptimizationProblem message to cpu_optimization_problem_t.
 *
 * Reads from a protobuf message using the generated protobuf C++ API.
 * Does not perform deserialization — that is handled by the protobuf library.
 */
template <typename i_t, typename f_t>
void map_proto_to_problem(const cuopt::remote::OptimizationProblem& pb_problem,
                          cpu_optimization_problem_t<i_t, f_t>& cpu_problem);

/**
 * @brief Estimate the serialized protobuf size of a SolveLPRequest/SolveMIPRequest.
 *
 * Computes an approximate upper bound on the serialized size without actually building
 * the protobuf message. Used to decide whether to use chunked array transfer.
 *
 * @return Estimated size in bytes
 */
template <typename i_t, typename f_t>
size_t estimate_problem_proto_size(const cpu_optimization_problem_t<i_t, f_t>& cpu_problem);

/**
 * @brief Populate a ChunkedProblemHeader from a cpu_optimization_problem_t and LP settings.
 *
 * Fills the header with problem scalars, string arrays, and LP settings.
 * Numeric arrays are NOT included (they are sent as ArrayChunk messages).
 */
template <typename i_t, typename f_t>
void populate_chunked_header_lp(const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
                                const pdlp_solver_settings_t<i_t, f_t>& settings,
                                cuopt::remote::ChunkedProblemHeader* header);

/**
 * @brief Populate a ChunkedProblemHeader from a cpu_optimization_problem_t and MIP settings.
 *
 * Fills the header with problem scalars, string arrays, and MIP settings.
 * Numeric arrays are NOT included (they are sent as ArrayChunk messages).
 */
template <typename i_t, typename f_t>
void populate_chunked_header_mip(const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
                                 const mip_solver_settings_t<i_t, f_t>& settings,
                                 bool enable_incumbents,
                                 cuopt::remote::ChunkedProblemHeader* header);

/**
 * @brief Reconstruct a cpu_optimization_problem_t from a ChunkedProblemHeader.
 *
 * Populates problem scalars and string arrays from the header. Numeric arrays
 * must be populated separately from ArrayChunk data.
 */
template <typename i_t, typename f_t>
void map_chunked_header_to_problem(const cuopt::remote::ChunkedProblemHeader& header,
                                   cpu_optimization_problem_t<i_t, f_t>& cpu_problem);

/**
 * @brief Reconstruct a cpu_optimization_problem_t from a ChunkedProblemHeader and raw array data.
 *
 * This is the single entry point for reconstructing a problem from chunked transfer data.
 * It calls map_chunked_header_to_problem() for scalars/strings, then populates all numeric
 * arrays from the raw byte data keyed by ArrayFieldId.
 *
 * @param header The chunked problem header (scalars, settings metadata, string arrays)
 * @param arrays Map of ArrayFieldId (as int32_t) to raw byte data for each array field
 * @param cpu_problem The cpu_optimization_problem_t to populate (output parameter)
 */
template <typename i_t, typename f_t>
void map_chunked_arrays_to_problem(const cuopt::remote::ChunkedProblemHeader& header,
                                   const std::map<int32_t, std::vector<uint8_t>>& arrays,
                                   cpu_optimization_problem_t<i_t, f_t>& cpu_problem);

/**
 * @brief Build SendArrayChunkRequest messages for chunked upload of problem arrays.
 *
 * Iterates the problem's host arrays directly and slices each array into
 * chunk-sized SendArrayChunkRequest protobuf messages. The caller simply
 * iterates the returned vector and sends each message via SendArrayChunk RPC.
 *
 * @param problem The problem whose arrays to chunk
 * @param upload_id The upload session ID from StartChunkedUpload
 * @param chunk_size_bytes Maximum raw data bytes per chunk message
 * @return Vector of ready-to-send SendArrayChunkRequest protobuf messages
 */
template <typename i_t, typename f_t>
std::vector<cuopt::remote::SendArrayChunkRequest> build_array_chunk_requests(
  const cpu_optimization_problem_t<i_t, f_t>& problem,
  const std::string& upload_id,
  int64_t chunk_size_bytes);

}  // namespace cuopt::linear_programming
