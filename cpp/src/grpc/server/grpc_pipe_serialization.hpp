/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef CUOPT_ENABLE_GRPC

#include "../grpc_chunk_key.hpp"
#include "cuopt_remote.pb.h"
#include "cuopt_remote_service.pb.h"
#include "grpc_field_element_size.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <vector>

// Requested pipe buffer size (1 MiB). The kernel default is 64 KiB, which
// forces excessive context-switching on large transfers. fcntl(F_SETPIPE_SZ)
// may silently cap this to /proc/sys/fs/pipe-max-size.
static constexpr int kPipeBufferSize = 1024 * 1024;

static constexpr uint64_t kMaxPipeArrayBytes       = 4ULL * 1024 * 1024 * 1024;
static constexpr uint32_t kMaxPipeArrayFields      = 10000;
static constexpr uint32_t kMaxProtobufMessageBytes = 64 * 1024 * 1024;  // 64 MiB

// Pipe I/O primitives defined in grpc_job_management.cpp.
bool write_to_pipe(int fd, const void* data, size_t size);
bool read_from_pipe(int fd, void* data, size_t size, int timeout_ms = 120000);

// =============================================================================
// Low-level: write/read a single protobuf message with a uint32 length prefix.
// Uses standard protobuf SerializeToArray / ParseFromArray for the payload.
// =============================================================================

inline bool write_protobuf_to_pipe(int fd, const google::protobuf::MessageLite& msg)
{
  size_t byte_size = msg.ByteSizeLong();
  if (byte_size > kMaxProtobufMessageBytes) return false;
  uint32_t size = static_cast<uint32_t>(byte_size);
  if (!write_to_pipe(fd, &size, sizeof(size))) return false;
  if (size == 0) return true;
  std::vector<uint8_t> buf(size);
  if (!msg.SerializeToArray(buf.data(), static_cast<int>(size))) return false;
  return write_to_pipe(fd, buf.data(), size);
}

inline bool read_protobuf_from_pipe(int fd, google::protobuf::MessageLite& msg)
{
  uint32_t size;
  if (!read_from_pipe(fd, &size, sizeof(size))) return false;
  if (size > kMaxProtobufMessageBytes) return false;
  if (size == 0) return msg.ParseFromArray(nullptr, 0);
  std::vector<uint8_t> buf(size);
  if (!read_from_pipe(fd, buf.data(), size)) return false;
  return msg.ParseFromArray(buf.data(), static_cast<int>(size));
}

// =============================================================================
// Chunked request: server → worker pipe (ChunkedProblemHeader + raw arrays)
//
// Wire format (protobuf header + raw byte arrays):
//   [uint32 hdr_size][protobuf header bytes]
//   [uint32 num_arrays]                              // top-level arrays
//   per array: [int32 field_id][uint64 total_bytes][raw bytes...]
//   [uint32 num_container_arrays]                    // container arrays
//   per container array:
//     [int32 container_field_num][int32 container_index]
//     [int32 field_id][uint64 total_bytes][raw bytes...]
//
// The protobuf ChunkedProblemHeader carries all metadata (settings,
// scalars, string arrays, and per-entry scalars for repeated_messages).
// Numeric arrays — both top-level and per-container — bypass protobuf
// serialization and flow directly through the pipe as raw bytes.
//
// The pipe is intra-process (forked worker), so this format is not a
// public protocol and can evolve in lockstep with the server binary.
// =============================================================================

namespace detail {

// Per-field bookkeeping used by write_chunked_request_to_pipe to assemble a
// single contiguous payload from possibly many client-side chunks.
struct FieldChunks {
  std::vector<const cuopt::remote::ArrayChunk*> chunks;
  int64_t total_bytes = 0;
};

// Assemble a single field's chunks into a contiguous byte payload and write
// it to the pipe with a length prefix already supplied by the caller.
// Returns false on any chunk overlap, gap, or out-of-range byte position.
inline bool assemble_and_write_field_payload(int fd, const FieldChunks& fi)
{
  if (fi.total_bytes == 0) return true;

  // Fast path: a single chunk that covers the whole array.  Write directly
  // from the protobuf bytes string, avoiding an assembly copy.
  if (fi.chunks.size() == 1 && fi.chunks[0]->element_offset() == 0 &&
      static_cast<int64_t>(fi.chunks[0]->data().size()) == fi.total_bytes) {
    return write_to_pipe(fd, fi.chunks[0]->data().data(), fi.chunks[0]->data().size());
  }

  // Slow path: stitch multiple chunks into a contiguous buffer, placing each
  // chunk at its element_offset * elem_size byte position.
  int64_t total_elements = fi.chunks[0]->total_elements();
  if (total_elements <= 0 || fi.total_bytes % total_elements != 0) return false;
  int64_t elem_size = fi.total_bytes / total_elements;
  if (elem_size <= 0) return false;

  std::vector<uint8_t> assembled(static_cast<size_t>(fi.total_bytes), 0);
  // Per-element bitmap detects both overlaps (element written twice) and
  // gaps (element never written).
  std::vector<bool> covered(static_cast<size_t>(total_elements), false);

  for (const auto* ac : fi.chunks) {
    int64_t element_offset = ac->element_offset();
    const auto& chunk_data = ac->data();
    if (chunk_data.size() % static_cast<size_t>(elem_size) != 0) return false;
    int64_t chunk_elements = static_cast<int64_t>(chunk_data.size()) / elem_size;
    if (element_offset < 0 || chunk_elements < 0) return false;
    if (element_offset > total_elements - chunk_elements) return false;

    int64_t byte_offset = element_offset * elem_size;
    if (byte_offset + static_cast<int64_t>(chunk_data.size()) > fi.total_bytes) return false;

    for (int64_t e = 0; e < chunk_elements; ++e) {
      size_t idx = static_cast<size_t>(element_offset + e);
      if (covered[idx]) return false;  // overlap
      covered[idx] = true;
    }
    std::memcpy(assembled.data() + byte_offset, chunk_data.data(), chunk_data.size());
  }
  // Every element must be covered exactly once (no gaps).
  for (size_t e = 0; e < static_cast<size_t>(total_elements); ++e) {
    if (!covered[e]) return false;
  }
  return write_to_pipe(fd, assembled.data(), assembled.size());
}

}  // namespace detail

inline bool write_chunked_request_to_pipe(int fd,
                                          const cuopt::remote::ChunkedProblemHeader& header,
                                          const std::vector<cuopt::remote::ArrayChunk>& chunks)
{
  // Step 1: write the protobuf header (settings, scalars, string arrays,
  // and per-entry scalars for any repeated_messages entries).
  if (!write_protobuf_to_pipe(fd, header)) return false;

  // Step 2: partition incoming gRPC chunks into top-level vs container,
  // grouping by their respective composite keys.  A single field may arrive
  // as multiple chunks (the client splits large arrays at chunk_size_bytes).
  std::map<int32_t, detail::FieldChunks> top_fields;
  std::map<cuopt::linear_programming::ContainerArrayKey, detail::FieldChunks> container_fields;
  for (const auto& ac : chunks) {
    int64_t elem_size       = 0;
    detail::FieldChunks* fi = nullptr;
    if (ac.has_container_field_num()) {
      cuopt::linear_programming::ContainerArrayKey key{
        ac.container_field_num(), ac.container_index(), ac.field_id()};
      fi        = &container_fields[key];
      elem_size = array_field_element_size(key.container_field_num, key.field_id);
    } else {
      fi        = &top_fields[ac.field_id()];
      elem_size = array_field_element_size(-1, ac.field_id());
    }
    fi->chunks.push_back(&ac);
    if (fi->total_bytes == 0 && ac.total_elements() > 0) {
      if (elem_size > 0 && ac.total_elements() <= std::numeric_limits<int64_t>::max() / elem_size) {
        fi->total_bytes = ac.total_elements() * elem_size;
      }
    }
  }

  // Step 3a: top-level arrays — [int32 field_id][uint64 total_bytes][bytes...]
  uint32_t num_arrays = static_cast<uint32_t>(top_fields.size());
  if (!write_to_pipe(fd, &num_arrays, sizeof(num_arrays))) return false;

  for (const auto& [fid, fi] : top_fields) {
    int32_t field_id     = fid;
    uint64_t total_bytes = static_cast<uint64_t>(fi.total_bytes);
    if (!write_to_pipe(fd, &field_id, sizeof(field_id))) return false;
    if (!write_to_pipe(fd, &total_bytes, sizeof(total_bytes))) return false;
    if (!detail::assemble_and_write_field_payload(fd, fi)) return false;
  }

  // Step 3b: container arrays — [int32 cfn][int32 ci][int32 fid][uint64 sz][bytes...]
  uint32_t num_container_arrays = static_cast<uint32_t>(container_fields.size());
  if (!write_to_pipe(fd, &num_container_arrays, sizeof(num_container_arrays))) return false;

  for (const auto& [key, fi] : container_fields) {
    int32_t cfn          = key.container_field_num;
    int32_t ci           = key.container_index;
    int32_t field_id     = key.field_id;
    uint64_t total_bytes = static_cast<uint64_t>(fi.total_bytes);
    if (!write_to_pipe(fd, &cfn, sizeof(cfn))) return false;
    if (!write_to_pipe(fd, &ci, sizeof(ci))) return false;
    if (!write_to_pipe(fd, &field_id, sizeof(field_id))) return false;
    if (!write_to_pipe(fd, &total_bytes, sizeof(total_bytes))) return false;
    if (!detail::assemble_and_write_field_payload(fd, fi)) return false;
  }

  return true;
}

inline bool read_chunked_request_from_pipe(int fd,
                                           cuopt::remote::ChunkedProblemHeader& header_out,
                                           std::map<int32_t, std::vector<uint8_t>>& arrays_out,
                                           std::map<cuopt::linear_programming::ContainerArrayKey,
                                                    std::vector<uint8_t>>& container_arrays_out)
{
  if (!read_protobuf_from_pipe(fd, header_out)) return false;

  // Top-level arrays
  uint32_t num_arrays;
  if (!read_from_pipe(fd, &num_arrays, sizeof(num_arrays))) return false;
  if (num_arrays > kMaxPipeArrayFields) return false;

  for (uint32_t i = 0; i < num_arrays; ++i) {
    int32_t field_id;
    uint64_t total_bytes;
    if (!read_from_pipe(fd, &field_id, sizeof(field_id))) return false;
    if (!read_from_pipe(fd, &total_bytes, sizeof(total_bytes))) return false;
    if (total_bytes > kMaxPipeArrayBytes) return false;
    auto& dest = arrays_out[field_id];
    dest.resize(static_cast<size_t>(total_bytes));
    if (total_bytes > 0 && !read_from_pipe(fd, dest.data(), static_cast<size_t>(total_bytes)))
      return false;
  }

  // Container arrays
  uint32_t num_container_arrays;
  if (!read_from_pipe(fd, &num_container_arrays, sizeof(num_container_arrays))) return false;
  if (num_container_arrays > kMaxPipeArrayFields) return false;

  for (uint32_t i = 0; i < num_container_arrays; ++i) {
    int32_t cfn;
    int32_t ci;
    int32_t field_id;
    uint64_t total_bytes;
    if (!read_from_pipe(fd, &cfn, sizeof(cfn))) return false;
    if (!read_from_pipe(fd, &ci, sizeof(ci))) return false;
    if (!read_from_pipe(fd, &field_id, sizeof(field_id))) return false;
    if (!read_from_pipe(fd, &total_bytes, sizeof(total_bytes))) return false;
    if (total_bytes > kMaxPipeArrayBytes) return false;
    auto& dest =
      container_arrays_out[cuopt::linear_programming::ContainerArrayKey{cfn, ci, field_id}];
    dest.resize(static_cast<size_t>(total_bytes));
    if (total_bytes > 0 && !read_from_pipe(fd, dest.data(), static_cast<size_t>(total_bytes)))
      return false;
  }

  return true;
}

// =============================================================================
// Result: worker → server pipe (ChunkedResultHeader + raw arrays)
//
// Same wire format as the chunked request above. Unlike the request path,
// result arrays are already assembled into contiguous vectors by the worker,
// so no chunk grouping or assembly is needed.
// =============================================================================

inline bool write_result_to_pipe(int fd,
                                 const cuopt::remote::ChunkedResultHeader& header,
                                 const std::map<int32_t, std::vector<uint8_t>>& arrays)
{
  if (!write_protobuf_to_pipe(fd, header)) return false;

  uint32_t num_arrays = static_cast<uint32_t>(arrays.size());
  if (!write_to_pipe(fd, &num_arrays, sizeof(num_arrays))) return false;

  // Each array is already contiguous — write field_id, size, and raw bytes.
  for (const auto& [fid, data] : arrays) {
    int32_t field_id     = fid;
    uint64_t total_bytes = data.size();
    if (!write_to_pipe(fd, &field_id, sizeof(field_id))) return false;
    if (!write_to_pipe(fd, &total_bytes, sizeof(total_bytes))) return false;
    if (total_bytes > 0 && !write_to_pipe(fd, data.data(), data.size())) return false;
  }

  return true;
}

inline bool read_result_from_pipe(int fd,
                                  cuopt::remote::ChunkedResultHeader& header_out,
                                  std::map<int32_t, std::vector<uint8_t>>& arrays_out)
{
  if (!read_protobuf_from_pipe(fd, header_out)) return false;

  uint32_t num_arrays;
  if (!read_from_pipe(fd, &num_arrays, sizeof(num_arrays))) return false;
  if (num_arrays > kMaxPipeArrayFields) return false;

  for (uint32_t i = 0; i < num_arrays; ++i) {
    int32_t field_id;
    uint64_t total_bytes;
    if (!read_from_pipe(fd, &field_id, sizeof(field_id))) return false;
    if (!read_from_pipe(fd, &total_bytes, sizeof(total_bytes))) return false;
    if (total_bytes > kMaxPipeArrayBytes) return false;
    auto& dest = arrays_out[field_id];
    dest.resize(static_cast<size_t>(total_bytes));
    if (total_bytes > 0 && !read_from_pipe(fd, dest.data(), static_cast<size_t>(total_bytes)))
      return false;
  }

  return true;
}

// Serialize a SubmitJobRequest directly to a pipe blob using standard protobuf.
// Used for unary submits only (always well under 2 GiB).
inline std::vector<uint8_t> serialize_submit_request_to_pipe(
  const cuopt::remote::SubmitJobRequest& request)
{
  size_t byte_size = request.ByteSizeLong();
  if (byte_size == 0 || byte_size > static_cast<size_t>(std::numeric_limits<int>::max())) return {};
  std::vector<uint8_t> blob(byte_size);
  request.SerializeToArray(blob.data(), static_cast<int>(byte_size));
  return blob;
}

#endif  // CUOPT_ENABLE_GRPC
