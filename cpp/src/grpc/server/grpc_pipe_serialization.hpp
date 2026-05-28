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

// Container arrays count every per-entry array inside every repeated nested
// message (e.g., 5 arrays per QuadraticConstraint row), so the cap must scale
// with problem size rather than reuse the small top-level field cap.  10M
// container arrays = 2M QC rows at 5 arrays per row, which is well above any
// realistic QCQP workload while still bounding worst-case map overhead at
// ~500–800 MB on a malicious client sending many empty entries.
static constexpr uint32_t kMaxPipeContainerArrays = 10'000'000;

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

// Tri-state result for write_chunked_request_to_pipe.  The distinction lets
// callers respond differently to malformed-input (bad client data — reject the
// job, keep the worker) vs. real pipe failures (worker may be hung on a
// half-fed pipe — kill it).
enum class PipeWriteStatus {
  Success,
  // Phase 1 (pre-pipe-write) validation rejected the request.  No bytes were
  // written to the pipe and the worker is untouched.  Safe to fail the job
  // without disturbing the worker process.
  ValidationFailed,
  // Phase 2 pipe write failed (or a phase-2 invariant tripped).  The pipe
  // may contain a partial transfer and the worker may be blocked reading
  // bytes that will never arrive.  Caller should tear the worker down.
  PipeFailed,
};

namespace detail {

// Per-field bookkeeping used by write_chunked_request_to_pipe to assemble a
// single contiguous payload from possibly many client-side chunks.
struct FieldChunks {
  std::vector<const cuopt::remote::ArrayChunk*> chunks;
  int64_t total_bytes = 0;
};

// Pre-flight check on the chunks for a single field: alignment, in-range
// offsets, exact disjoint coverage of the logical array.  Allocates only a
// per-element bitmap (one bit per element), no byte buffer, no pipe I/O.
// Mirrors the validation arm of assemble_and_write_field_payload so we can
// reject malformed input before any bytes are written to the worker pipe.
inline bool validate_field_chunks(const FieldChunks& fi)
{
  if (fi.total_bytes == 0) return true;

  if (fi.chunks.size() == 1 && fi.chunks[0]->element_offset() == 0 &&
      static_cast<int64_t>(fi.chunks[0]->data().size()) == fi.total_bytes) {
    return true;
  }

  int64_t total_elements = fi.chunks[0]->total_elements();
  if (total_elements <= 0 || fi.total_bytes % total_elements != 0) return false;
  int64_t elem_size = fi.total_bytes / total_elements;
  if (elem_size <= 0) return false;

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
      if (covered[idx]) return false;
      covered[idx] = true;
    }
  }
  for (size_t e = 0; e < static_cast<size_t>(total_elements); ++e) {
    if (!covered[e]) return false;
  }
  return true;
}

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

// Returns one of three PipeWriteStatus values:
//   Success           — header and all arrays written
//   ValidationFailed  — malformed client input rejected BEFORE any pipe write;
//                       the pipe is untouched and the worker is unaffected
//   PipeFailed        — the protobuf header or an array body failed mid-write;
//                       the pipe may contain a partial transfer and the worker
//                       may be blocked reading bytes that will never arrive
//
// All input validation happens in phase 1, before any pipe I/O.  This means a
// caller flooded with malformed requests can safely fail-fast without
// disturbing the (otherwise healthy) worker process — only PipeFailed should
// trigger worker teardown.
inline PipeWriteStatus write_chunked_request_to_pipe(
  int fd,
  const cuopt::remote::ChunkedProblemHeader& header,
  const std::vector<cuopt::remote::ArrayChunk>& chunks)
{
  // -------- Phase 1: bin chunks per field and validate (no pipe I/O) --------
  std::map<int32_t, detail::FieldChunks> top_fields;
  std::map<cuopt::linear_programming::container_array_key_t, detail::FieldChunks> container_fields;
  for (const auto& ac : chunks) {
    int64_t elem_size       = 0;
    detail::FieldChunks* fi = nullptr;
    if (ac.has_container_field_num()) {
      cuopt::linear_programming::container_array_key_t key{
        ac.container_field_num(), ac.container_index(), ac.field_id()};
      fi        = &container_fields[key];
      elem_size = array_field_element_size(key.container_field_num, key.field_id);
    } else {
      fi        = &top_fields[ac.field_id()];
      elem_size = array_field_element_size(-1, ac.field_id());
    }
    fi->chunks.push_back(&ac);

    // Unknown field_id (or container_field_num): codegen out of sync with
    // the client's enum, or a malicious/corrupted chunk.  Reject hard even
    // for zero-element chunks — an unknown field_id is a protocol error
    // regardless of payload size.
    if (elem_size <= 0) return PipeWriteStatus::ValidationFailed;

    // Negative total_elements is always malformed.
    if (ac.total_elements() < 0) return PipeWriteStatus::ValidationFailed;

    if (ac.total_elements() == 0) {
      // Zero-element chunks must carry an empty payload.  Phase 2 writes
      // total_bytes=0 and no payload for such fields, so any data here
      // would otherwise be silently discarded.
      if (!ac.data().empty()) return PipeWriteStatus::ValidationFailed;
      continue;
    }

    // Overflow guard for total_elements * elem_size.
    if (ac.total_elements() > std::numeric_limits<int64_t>::max() / elem_size) {
      return PipeWriteStatus::ValidationFailed;
    }
    int64_t this_bytes = ac.total_elements() * elem_size;

    // Enforce the reader's per-array byte cap on the writer side, so a
    // too-large request fails as ValidationFailed (no pipe I/O, worker
    // untouched) instead of surfacing as a mid-stream PipeFailed at the
    // reader (which would force a worker SIGKILL).
    if (static_cast<uint64_t>(this_bytes) > kMaxPipeArrayBytes) {
      return PipeWriteStatus::ValidationFailed;
    }

    if (fi->total_bytes == 0) {
      fi->total_bytes = this_bytes;
    } else if (fi->total_bytes != this_bytes) {
      // Chunks for the same logical array disagree on the array's total
      // size — they should all carry the same total_elements value.
      return PipeWriteStatus::ValidationFailed;
    }
  }

  // Mirror the reader's count caps so an oversized field set fails cleanly
  // here instead of mid-pipe-write on the receiving end.
  if (top_fields.size() > kMaxPipeArrayFields) return PipeWriteStatus::ValidationFailed;
  if (container_fields.size() > kMaxPipeContainerArrays) {
    return PipeWriteStatus::ValidationFailed;
  }

  // Per-field deep validation (alignment, range, exact disjoint coverage).
  // assemble_and_write_field_payload still re-checks these as defense in
  // depth, but pre-validating here keeps phase 2 purely I/O so any failure
  // there is unambiguously a pipe failure.
  for (const auto& [fid, fi] : top_fields) {
    (void)fid;
    if (!detail::validate_field_chunks(fi)) return PipeWriteStatus::ValidationFailed;
  }
  for (const auto& [key, fi] : container_fields) {
    (void)key;
    if (!detail::validate_field_chunks(fi)) return PipeWriteStatus::ValidationFailed;
  }

  // -------- Phase 2: commit to the pipe (any failure = PipeFailed) --------
  if (!write_protobuf_to_pipe(fd, header)) return PipeWriteStatus::PipeFailed;

  uint32_t num_arrays = static_cast<uint32_t>(top_fields.size());
  if (!write_to_pipe(fd, &num_arrays, sizeof(num_arrays))) return PipeWriteStatus::PipeFailed;
  for (const auto& [fid, fi] : top_fields) {
    int32_t field_id     = fid;
    uint64_t total_bytes = static_cast<uint64_t>(fi.total_bytes);
    if (!write_to_pipe(fd, &field_id, sizeof(field_id))) return PipeWriteStatus::PipeFailed;
    if (!write_to_pipe(fd, &total_bytes, sizeof(total_bytes))) return PipeWriteStatus::PipeFailed;
    if (!detail::assemble_and_write_field_payload(fd, fi)) return PipeWriteStatus::PipeFailed;
  }

  uint32_t num_container_arrays = static_cast<uint32_t>(container_fields.size());
  if (!write_to_pipe(fd, &num_container_arrays, sizeof(num_container_arrays))) {
    return PipeWriteStatus::PipeFailed;
  }
  for (const auto& [key, fi] : container_fields) {
    int32_t cfn          = key.container_field_num;
    int32_t ci           = key.container_index;
    int32_t field_id     = key.field_id;
    uint64_t total_bytes = static_cast<uint64_t>(fi.total_bytes);
    if (!write_to_pipe(fd, &cfn, sizeof(cfn))) return PipeWriteStatus::PipeFailed;
    if (!write_to_pipe(fd, &ci, sizeof(ci))) return PipeWriteStatus::PipeFailed;
    if (!write_to_pipe(fd, &field_id, sizeof(field_id))) return PipeWriteStatus::PipeFailed;
    if (!write_to_pipe(fd, &total_bytes, sizeof(total_bytes))) return PipeWriteStatus::PipeFailed;
    if (!detail::assemble_and_write_field_payload(fd, fi)) return PipeWriteStatus::PipeFailed;
  }

  return PipeWriteStatus::Success;
}

inline bool read_chunked_request_from_pipe(
  int fd,
  cuopt::remote::ChunkedProblemHeader& header_out,
  std::map<int32_t, std::vector<uint8_t>>& arrays_out,
  std::map<cuopt::linear_programming::container_array_key_t, std::vector<uint8_t>>&
    container_arrays_out)
{
  // dest.resize(total_bytes) can throw std::bad_alloc if the wire claims an
  // allocation larger than the worker can satisfy.  Catch it here so the
  // caller sees a clean false and the worker can report job failure instead
  // of dying via an uncaught exception (which the worker monitor would just
  // respawn anyway).
  try {
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

    // Container arrays.  Cap is separate from kMaxPipeArrayFields because the
    // count scales with problem size (every per-entry array inside every
    // repeated nested message), not with the small fixed set of top-level
    // ArrayFieldId enum values.
    uint32_t num_container_arrays;
    if (!read_from_pipe(fd, &num_container_arrays, sizeof(num_container_arrays))) return false;
    if (num_container_arrays > kMaxPipeContainerArrays) return false;

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
        container_arrays_out[cuopt::linear_programming::container_array_key_t{cfn, ci, field_id}];
      dest.resize(static_cast<size_t>(total_bytes));
      if (total_bytes > 0 && !read_from_pipe(fd, dest.data(), static_cast<size_t>(total_bytes)))
        return false;
    }

    return true;
  } catch (const std::bad_alloc&) {
    return false;
  }
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
