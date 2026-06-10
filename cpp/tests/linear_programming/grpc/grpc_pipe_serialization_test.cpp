/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file grpc_pipe_serialization_test.cpp
 * @brief Round-trip unit tests for the hybrid pipe serialization format.
 *
 * Tests write data through a real pipe(2) and read it back, verifying that
 * protobuf headers and raw array bytes survive the round trip intact.
 * A writer thread is used because pipe buffers are finite; blocking writes
 * would deadlock if the reader isn't draining concurrently.
 */

#include <gtest/gtest.h>

#include <poll.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <thread>
#include <vector>

// write_to_pipe / read_from_pipe are the real implementations from
// grpc_pipe_io.cpp, compiled directly into this test target.
#include "grpc_pipe_serialization.hpp"

using namespace cuopt::remote;
using cuopt::linear_programming::container_array_key_t;

// ---------------------------------------------------------------------------
// RAII wrapper for a pipe(2) pair.
// ---------------------------------------------------------------------------
class PipePair {
 public:
  PipePair()
  {
    if (::pipe(fds_) != 0) { throw std::runtime_error("pipe() failed"); }
  }
  ~PipePair()
  {
    if (fds_[0] >= 0) ::close(fds_[0]);
    if (fds_[1] >= 0) ::close(fds_[1]);
  }
  int read_fd() const { return fds_[0]; }
  int write_fd() const { return fds_[1]; }

 private:
  int fds_[2]{-1, -1};
};

// ---------------------------------------------------------------------------
// Helpers to build test data.
// ---------------------------------------------------------------------------
namespace {

std::vector<uint8_t> make_pattern(size_t num_bytes, uint8_t seed = 0)
{
  std::vector<uint8_t> v(num_bytes);
  for (size_t i = 0; i < num_bytes; ++i) {
    v[i] = static_cast<uint8_t>((i + seed) & 0xFF);
  }
  return v;
}

ArrayChunk make_whole_chunk(ArrayFieldId field_id,
                            int64_t total_elements,
                            const std::vector<uint8_t>& data)
{
  ArrayChunk ac;
  ac.set_field_id(field_id);
  ac.set_element_offset(0);
  ac.set_total_elements(total_elements);
  ac.set_data(std::string(reinterpret_cast<const char*>(data.data()), data.size()));
  return ac;
}

ArrayChunk make_partial_chunk(ArrayFieldId field_id,
                              int64_t element_offset,
                              int64_t total_elements,
                              const uint8_t* data,
                              size_t data_size)
{
  ArrayChunk ac;
  ac.set_field_id(field_id);
  ac.set_element_offset(element_offset);
  ac.set_total_elements(total_elements);
  ac.set_data(std::string(reinterpret_cast<const char*>(data), data_size));
  return ac;
}

// Container variants: stamp the (cfn, ci) optional fields so the chunk is
// routed to the container-keyed map on the read side.  `field_id` here is a
// container-relative id (e.g. 0..4 inside a QuadraticConstraint), not an
// ArrayFieldId enum value.
ArrayChunk make_container_whole_chunk(int32_t container_field_num,
                                      int32_t container_index,
                                      int32_t field_id,
                                      int64_t total_elements,
                                      const std::vector<uint8_t>& data)
{
  ArrayChunk ac;
  ac.set_field_id(field_id);
  ac.set_element_offset(0);
  ac.set_total_elements(total_elements);
  ac.set_data(std::string(reinterpret_cast<const char*>(data.data()), data.size()));
  ac.set_container_field_num(container_field_num);
  ac.set_container_index(container_index);
  return ac;
}

ArrayChunk make_container_partial_chunk(int32_t container_field_num,
                                        int32_t container_index,
                                        int32_t field_id,
                                        int64_t element_offset,
                                        int64_t total_elements,
                                        const uint8_t* data,
                                        size_t data_size)
{
  ArrayChunk ac;
  ac.set_field_id(field_id);
  ac.set_element_offset(element_offset);
  ac.set_total_elements(total_elements);
  ac.set_data(std::string(reinterpret_cast<const char*>(data), data_size));
  ac.set_container_field_num(container_field_num);
  ac.set_container_index(container_index);
  return ac;
}

// Non-blocking poll on the read end of a pipe.  Returns true iff there is
// nothing readable right now (i.e. the writer left the pipe untouched).
// Used by the ValidationFailed tests to confirm the writer aborted before
// emitting a single byte.
//
// Retries on EINTR: even with a 0ms timeout, poll() can return -1/EINTR if a
// signal is delivered between syscall entry and the timer expiring.  Without
// the retry, an unlucky signal during a test run would make this report
// "pipe not empty" and falsely fail the assertion.
bool pipe_is_empty(int read_fd)
{
  pollfd pfd{};
  pfd.fd     = read_fd;
  pfd.events = POLLIN;
  int rc;
  do {
    rc = ::poll(&pfd, 1, 0);
  } while (rc == -1 && errno == EINTR);
  // rc == 0 means timeout fired with no events; rc < 0 is an error other
  // than EINTR.  Anything else (rc > 0) means data or hangup is pending.
  return rc == 0;
}

}  // namespace

// =============================================================================
// Chunked request round-trip tests
// =============================================================================

TEST(PipeSerialization, ChunkedRequest_SingleChunkPerField)
{
  PipePair pp;

  ChunkedProblemHeader header;
  header.set_maximize(true);
  header.set_objective_scaling_factor(2.5);
  header.set_problem_name("test_lp");

  // Two fields: FIELD_C (8-byte doubles, 100 elements) and FIELD_A_INDICES (4-byte ints, 50
  // elements)
  auto c_data = make_pattern(100 * 8, 0xAA);
  auto i_data = make_pattern(50 * 4, 0xBB);

  std::vector<ArrayChunk> chunks;
  chunks.push_back(make_whole_chunk(FIELD_C, 100, c_data));
  chunks.push_back(make_whole_chunk(FIELD_A_INDICES, 50, i_data));

  // Write in a thread (pipe buffer is finite).
  PipeWriteStatus write_status = PipeWriteStatus::ValidationFailed;
  std::thread writer(
    [&] { write_status = write_chunked_request_to_pipe(pp.write_fd(), header, chunks); });

  ChunkedProblemHeader header_out;
  std::map<int32_t, std::vector<uint8_t>> arrays_out;
  std::map<container_array_key_t, std::vector<uint8_t>> container_arrays_out;
  bool read_ok =
    read_chunked_request_from_pipe(pp.read_fd(), header_out, arrays_out, container_arrays_out);

  writer.join();

  ASSERT_EQ(write_status, PipeWriteStatus::Success);
  ASSERT_TRUE(read_ok);

  EXPECT_TRUE(header_out.maximize());
  EXPECT_DOUBLE_EQ(header_out.objective_scaling_factor(), 2.5);
  EXPECT_EQ(header_out.problem_name(), "test_lp");

  ASSERT_EQ(arrays_out.size(), 2u);
  EXPECT_EQ(arrays_out[FIELD_C], c_data);
  EXPECT_EQ(arrays_out[FIELD_A_INDICES], i_data);
  EXPECT_TRUE(container_arrays_out.empty());
}

TEST(PipeSerialization, ChunkedRequest_MultiChunkAssembly)
{
  PipePair pp;

  ChunkedProblemHeader header;
  header.set_maximize(false);

  // Split a 200-element double array (FIELD_C, 8 bytes each = 1600 bytes) into two chunks.
  constexpr int64_t total_elements = 200;
  constexpr int64_t elem_size      = 8;
  auto full_data                   = make_pattern(total_elements * elem_size, 0x42);

  int64_t split = 120;
  std::vector<ArrayChunk> chunks;
  chunks.push_back(make_partial_chunk(
    FIELD_C, 0, total_elements, full_data.data(), static_cast<size_t>(split * elem_size)));
  chunks.push_back(make_partial_chunk(FIELD_C,
                                      split,
                                      total_elements,
                                      full_data.data() + split * elem_size,
                                      static_cast<size_t>((total_elements - split) * elem_size)));

  PipeWriteStatus write_status = PipeWriteStatus::ValidationFailed;
  std::thread writer(
    [&] { write_status = write_chunked_request_to_pipe(pp.write_fd(), header, chunks); });

  ChunkedProblemHeader header_out;
  std::map<int32_t, std::vector<uint8_t>> arrays_out;
  std::map<container_array_key_t, std::vector<uint8_t>> container_arrays_out;
  bool read_ok =
    read_chunked_request_from_pipe(pp.read_fd(), header_out, arrays_out, container_arrays_out);

  writer.join();

  ASSERT_EQ(write_status, PipeWriteStatus::Success);
  ASSERT_TRUE(read_ok);
  ASSERT_EQ(arrays_out.size(), 1u);
  EXPECT_EQ(arrays_out[FIELD_C], full_data);
  EXPECT_TRUE(container_arrays_out.empty());
}

TEST(PipeSerialization, ChunkedRequest_EmptyArrays)
{
  PipePair pp;

  ChunkedProblemHeader header;
  header.set_problem_name("empty");

  // A field with total_elements=0 should produce a zero-length array entry.
  ArrayChunk empty_chunk;
  empty_chunk.set_field_id(FIELD_C);
  empty_chunk.set_element_offset(0);
  empty_chunk.set_total_elements(0);
  empty_chunk.set_data("");

  std::vector<ArrayChunk> chunks = {empty_chunk};

  PipeWriteStatus write_status = PipeWriteStatus::ValidationFailed;
  std::thread writer(
    [&] { write_status = write_chunked_request_to_pipe(pp.write_fd(), header, chunks); });

  ChunkedProblemHeader header_out;
  std::map<int32_t, std::vector<uint8_t>> arrays_out;
  std::map<container_array_key_t, std::vector<uint8_t>> container_arrays_out;
  bool read_ok =
    read_chunked_request_from_pipe(pp.read_fd(), header_out, arrays_out, container_arrays_out);

  writer.join();

  ASSERT_EQ(write_status, PipeWriteStatus::Success);
  ASSERT_TRUE(read_ok);
  EXPECT_EQ(header_out.problem_name(), "empty");
  ASSERT_EQ(arrays_out.size(), 1u);
  EXPECT_TRUE(arrays_out[FIELD_C].empty());
  EXPECT_TRUE(container_arrays_out.empty());
}

TEST(PipeSerialization, ChunkedRequest_NoChunks)
{
  PipePair pp;

  ChunkedProblemHeader header;
  header.set_problem_name("header_only");

  std::vector<ArrayChunk> chunks;  // no chunks at all

  PipeWriteStatus write_status = PipeWriteStatus::ValidationFailed;
  std::thread writer(
    [&] { write_status = write_chunked_request_to_pipe(pp.write_fd(), header, chunks); });

  ChunkedProblemHeader header_out;
  std::map<int32_t, std::vector<uint8_t>> arrays_out;
  std::map<container_array_key_t, std::vector<uint8_t>> container_arrays_out;
  bool read_ok =
    read_chunked_request_from_pipe(pp.read_fd(), header_out, arrays_out, container_arrays_out);

  writer.join();

  ASSERT_EQ(write_status, PipeWriteStatus::Success);
  ASSERT_TRUE(read_ok);
  EXPECT_EQ(header_out.problem_name(), "header_only");
  EXPECT_TRUE(arrays_out.empty());
  EXPECT_TRUE(container_arrays_out.empty());
}

TEST(PipeSerialization, ChunkedRequest_ManyFields)
{
  PipePair pp;

  ChunkedProblemHeader header;
  header.set_maximize(true);

  // Build one whole chunk per field for several different field types.
  struct TestField {
    ArrayFieldId id;
    int64_t elements;
  };
  std::vector<TestField> test_fields = {
    {FIELD_A_VALUES, 500},
    {FIELD_A_INDICES, 500},
    {FIELD_A_OFFSETS, 101},
    {FIELD_C, 100},
    {FIELD_VARIABLE_LOWER_BOUNDS, 100},
    {FIELD_VARIABLE_UPPER_BOUNDS, 100},
    {FIELD_CONSTRAINT_LOWER_BOUNDS, 100},
    {FIELD_CONSTRAINT_UPPER_BOUNDS, 100},
  };

  std::map<int32_t, std::vector<uint8_t>> expected;
  std::vector<ArrayChunk> chunks;
  for (size_t i = 0; i < test_fields.size(); ++i) {
    auto& tf   = test_fields[i];
    int64_t es = array_field_element_size(-1, tf.id);
    auto data  = make_pattern(static_cast<size_t>(tf.elements * es), static_cast<uint8_t>(i));
    expected[static_cast<int32_t>(tf.id)] = data;
    chunks.push_back(make_whole_chunk(tf.id, tf.elements, data));
  }

  PipeWriteStatus write_status = PipeWriteStatus::ValidationFailed;
  std::thread writer(
    [&] { write_status = write_chunked_request_to_pipe(pp.write_fd(), header, chunks); });

  ChunkedProblemHeader header_out;
  std::map<int32_t, std::vector<uint8_t>> arrays_out;
  std::map<container_array_key_t, std::vector<uint8_t>> container_arrays_out;
  bool read_ok =
    read_chunked_request_from_pipe(pp.read_fd(), header_out, arrays_out, container_arrays_out);

  writer.join();

  ASSERT_EQ(write_status, PipeWriteStatus::Success);
  ASSERT_TRUE(read_ok);
  ASSERT_EQ(arrays_out.size(), expected.size());
  for (const auto& [fid, data] : expected) {
    ASSERT_TRUE(arrays_out.count(fid)) << "Missing field_id " << fid;
    EXPECT_EQ(arrays_out[fid], data) << "Mismatch for field_id " << fid;
  }
}

// Round-trip a mix of top-level and container chunks through the pipe.
// Exercises:
//   * Top-level fast path (single whole chunk per field).
//   * Container fast path (single whole chunk per (cfn, ci, fid)).
//   * Container slow path (one container array delivered as two partial
//     chunks that must be stitched at the writer side).
//   * Multiple container indices under the same parent field_num to ensure
//     they don't collide on the int32_t key space.
// Uses cfn=25 / fids 0..4 which match the registered quadratic_constraints
// entry, so array_field_element_size() returns the correct widths.
TEST(PipeSerialization, ChunkedRequest_ContainerArrays)
{
  PipePair pp;

  ChunkedProblemHeader header;
  header.set_problem_name("qc_pipe_smoke");

  // One top-level FIELD_C (8-byte doubles, 50 elements) — confirms top-level
  // path still works when container chunks are present.
  auto c_data = make_pattern(50 * 8, 0xC0);

  // Container index 0: linear_values (cfn=25, fid=0, 8 bytes), small fast-path
  // single-chunk.  Also linear_indices (cfn=25, fid=1, 4 bytes) and
  // vals (cfn=25, fid=4, 8 bytes — doubles, COO storage of Q).
  auto lv0_data   = make_pattern(10 * 8, 0xA0);
  auto li0_data   = make_pattern(10 * 4, 0xA1);
  auto vals0_data = make_pattern(3 * 8, 0xA4);

  // Container index 1: linear_values delivered in two partial chunks (slow
  // path inside a container).  20 elements * 8 bytes = 160 bytes, split at 12.
  constexpr int64_t lv1_total = 20;
  constexpr int64_t lv1_split = 12;
  auto lv1_data               = make_pattern(lv1_total * 8, 0xB0);

  std::vector<ArrayChunk> chunks;
  chunks.push_back(make_whole_chunk(FIELD_C, 50, c_data));
  chunks.push_back(make_container_whole_chunk(25, 0, 0, 10, lv0_data));
  chunks.push_back(make_container_whole_chunk(25, 0, 1, 10, li0_data));
  chunks.push_back(make_container_whole_chunk(25, 0, 4, 3, vals0_data));
  chunks.push_back(make_container_partial_chunk(
    25, 1, 0, 0, lv1_total, lv1_data.data(), static_cast<size_t>(lv1_split * 8)));
  chunks.push_back(make_container_partial_chunk(25,
                                                1,
                                                0,
                                                lv1_split,
                                                lv1_total,
                                                lv1_data.data() + lv1_split * 8,
                                                static_cast<size_t>((lv1_total - lv1_split) * 8)));

  PipeWriteStatus write_status = PipeWriteStatus::ValidationFailed;
  std::thread writer(
    [&] { write_status = write_chunked_request_to_pipe(pp.write_fd(), header, chunks); });

  ChunkedProblemHeader header_out;
  std::map<int32_t, std::vector<uint8_t>> arrays_out;
  std::map<container_array_key_t, std::vector<uint8_t>> container_arrays_out;
  bool read_ok =
    read_chunked_request_from_pipe(pp.read_fd(), header_out, arrays_out, container_arrays_out);

  writer.join();

  ASSERT_EQ(write_status, PipeWriteStatus::Success);
  ASSERT_TRUE(read_ok);
  EXPECT_EQ(header_out.problem_name(), "qc_pipe_smoke");

  // Top-level array round-trips unchanged.
  ASSERT_EQ(arrays_out.size(), 1u);
  EXPECT_EQ(arrays_out[FIELD_C], c_data);

  // Four container arrays survived (3 in entry 0, 1 in entry 1).  Brace-init
  // commas confuse the EXPECT_EQ macro's argument parsing; extract the key.
  ASSERT_EQ(container_arrays_out.size(), 4u);
  container_array_key_t k0_0{25, 0, 0};
  container_array_key_t k0_1{25, 0, 1};
  container_array_key_t k0_4{25, 0, 4};
  container_array_key_t k1_0{25, 1, 0};
  EXPECT_EQ(container_arrays_out[k0_0], lv0_data);
  EXPECT_EQ(container_arrays_out[k0_1], li0_data);
  EXPECT_EQ(container_arrays_out[k0_4], vals0_data);
  EXPECT_EQ(container_arrays_out[k1_0], lv1_data);
}

// =============================================================================
// Malformed-input rejection: write_chunked_request_to_pipe must report
// PipeWriteStatus::ValidationFailed and leave the pipe completely untouched
// when given bad chunks.  This protects the worker from being killed (or
// hung on a half-fed pipe) by adversarial / buggy clients.
// =============================================================================

TEST(PipeSerialization, ValidationFailed_UnknownFieldId)
{
  PipePair pp;

  ChunkedProblemHeader header;
  header.set_problem_name("bad");

  // 99999 is not a valid ArrayFieldId — array_field_element_size() returns
  // -1, which the writer must reject before any pipe I/O.
  ArrayChunk ac;
  ac.set_field_id(static_cast<ArrayFieldId>(99999));
  ac.set_element_offset(0);
  ac.set_total_elements(8);
  ac.set_data(std::string(64, '\0'));

  std::vector<ArrayChunk> chunks{ac};

  PipeWriteStatus write_status = write_chunked_request_to_pipe(pp.write_fd(), header, chunks);

  EXPECT_EQ(write_status, PipeWriteStatus::ValidationFailed);
  EXPECT_TRUE(pipe_is_empty(pp.read_fd()))
    << "Writer must not write any bytes to the pipe when validation fails";
}

TEST(PipeSerialization, ValidationFailed_OverflowTotalBytes)
{
  PipePair pp;

  ChunkedProblemHeader header;

  // FIELD_C is an 8-byte-per-element field.  total_elements above
  // INT64_MAX / 8 makes total_elements * elem_size overflow int64_t, which
  // the writer's overflow guard must catch.
  ArrayChunk ac;
  ac.set_field_id(FIELD_C);
  ac.set_element_offset(0);
  ac.set_total_elements(std::numeric_limits<int64_t>::max() / 4);  // > INT64_MAX/8
  ac.set_data("");

  std::vector<ArrayChunk> chunks{ac};

  PipeWriteStatus write_status = write_chunked_request_to_pipe(pp.write_fd(), header, chunks);

  EXPECT_EQ(write_status, PipeWriteStatus::ValidationFailed);
  EXPECT_TRUE(pipe_is_empty(pp.read_fd()));
}

TEST(PipeSerialization, ValidationFailed_ChunksDisagreeOnTotalElements)
{
  PipePair pp;

  ChunkedProblemHeader header;

  // Two chunks for the same FIELD_C field disagreeing about the logical
  // array length.  Every chunk for a given field must carry the same
  // total_elements value (it describes the WHOLE array, not the chunk).
  auto data = make_pattern(80, 0x55);  // 10 doubles
  ArrayChunk a;
  a.set_field_id(FIELD_C);
  a.set_element_offset(0);
  a.set_total_elements(10);
  a.set_data(std::string(reinterpret_cast<const char*>(data.data()), data.size()));

  ArrayChunk b;
  b.set_field_id(FIELD_C);
  b.set_element_offset(0);
  b.set_total_elements(20);  // disagrees with chunk a
  b.set_data(std::string(reinterpret_cast<const char*>(data.data()), data.size()));

  std::vector<ArrayChunk> chunks{a, b};

  PipeWriteStatus write_status = write_chunked_request_to_pipe(pp.write_fd(), header, chunks);

  EXPECT_EQ(write_status, PipeWriteStatus::ValidationFailed);
  EXPECT_TRUE(pipe_is_empty(pp.read_fd()));
}

TEST(PipeSerialization, ValidationFailed_ChunkOverlap)
{
  PipePair pp;

  ChunkedProblemHeader header;

  // Two chunks for FIELD_C claiming overlapping element ranges of the same
  // 20-element logical array.  validate_field_chunks() must reject this.
  auto chunk_a = make_pattern(15 * 8, 0xAA);  // covers elements [0, 15)
  auto chunk_b = make_pattern(10 * 8, 0xBB);  // covers elements [10, 20) — overlaps a
  std::vector<ArrayChunk> chunks;
  chunks.push_back(make_partial_chunk(
    FIELD_C, /*element_offset=*/0, /*total_elements=*/20, chunk_a.data(), chunk_a.size()));
  chunks.push_back(make_partial_chunk(
    FIELD_C, /*element_offset=*/10, /*total_elements=*/20, chunk_b.data(), chunk_b.size()));

  PipeWriteStatus write_status = write_chunked_request_to_pipe(pp.write_fd(), header, chunks);

  EXPECT_EQ(write_status, PipeWriteStatus::ValidationFailed);
  EXPECT_TRUE(pipe_is_empty(pp.read_fd()));
}

TEST(PipeSerialization, ValidationFailed_ChunkGap)
{
  PipePair pp;

  ChunkedProblemHeader header;

  // Two chunks for FIELD_C covering [0, 8) and [12, 20) — elements 8..11
  // are never written.  validate_field_chunks() must reject this.
  auto chunk_a = make_pattern(8 * 8, 0xAA);
  auto chunk_b = make_pattern(8 * 8, 0xBB);
  std::vector<ArrayChunk> chunks;
  chunks.push_back(make_partial_chunk(
    FIELD_C, /*element_offset=*/0, /*total_elements=*/20, chunk_a.data(), chunk_a.size()));
  chunks.push_back(make_partial_chunk(
    FIELD_C, /*element_offset=*/12, /*total_elements=*/20, chunk_b.data(), chunk_b.size()));

  PipeWriteStatus write_status = write_chunked_request_to_pipe(pp.write_fd(), header, chunks);

  EXPECT_EQ(write_status, PipeWriteStatus::ValidationFailed);
  EXPECT_TRUE(pipe_is_empty(pp.read_fd()));
}

TEST(PipeSerialization, ValidationFailed_MisalignedChunkSize)
{
  PipePair pp;

  ChunkedProblemHeader header;

  // FIELD_C is 8 bytes per element, but this chunk's data is 7 bytes —
  // not a clean multiple of elem_size.  Caught by validate_field_chunks().
  ArrayChunk ac;
  ac.set_field_id(FIELD_C);
  ac.set_element_offset(0);
  ac.set_total_elements(2);
  ac.set_data(std::string(7, '\0'));  // 7 bytes, not 16

  std::vector<ArrayChunk> chunks{ac};

  PipeWriteStatus write_status = write_chunked_request_to_pipe(pp.write_fd(), header, chunks);

  EXPECT_EQ(write_status, PipeWriteStatus::ValidationFailed);
  EXPECT_TRUE(pipe_is_empty(pp.read_fd()));
}

// Sanity check that the spam scenario is bounded: 1000 malformed requests
// must produce 1000 ValidationFailed results without ever touching the pipe.
// In production this is what protects the worker pool from a hostile client.
TEST(PipeSerialization, ValidationFailed_RepeatedSpamLeavesPipeEmpty)
{
  PipePair pp;

  ChunkedProblemHeader header;
  ArrayChunk ac;
  ac.set_field_id(static_cast<ArrayFieldId>(99999));  // unknown
  ac.set_element_offset(0);
  ac.set_total_elements(1);
  ac.set_data(std::string(8, '\0'));
  std::vector<ArrayChunk> chunks{ac};

  for (int i = 0; i < 1000; ++i) {
    PipeWriteStatus s = write_chunked_request_to_pipe(pp.write_fd(), header, chunks);
    ASSERT_EQ(s, PipeWriteStatus::ValidationFailed) << "iteration " << i;
  }
  EXPECT_TRUE(pipe_is_empty(pp.read_fd()));
}

// =============================================================================
// Result round-trip tests
// =============================================================================

TEST(PipeSerialization, Result_RoundTrip)
{
  PipePair pp;

  ChunkedResultHeader header;
  header.set_problem_category(LP);
  header.set_lp_termination_status(PDLP_OPTIMAL);
  header.set_primal_objective(42.5);
  header.set_solve_time(1.23);

  // Two result arrays: primal solution and dual solution.
  auto primal = make_pattern(1000 * 8, 0x11);
  auto dual   = make_pattern(500 * 8, 0x22);

  std::map<int32_t, std::vector<uint8_t>> arrays;
  arrays[RESULT_PRIMAL_SOLUTION] = primal;
  arrays[RESULT_DUAL_SOLUTION]   = dual;

  bool write_ok = false;
  std::thread writer([&] { write_ok = write_result_to_pipe(pp.write_fd(), header, arrays); });

  ChunkedResultHeader header_out;
  std::map<int32_t, std::vector<uint8_t>> arrays_out;
  bool read_ok = read_result_from_pipe(pp.read_fd(), header_out, arrays_out);

  writer.join();

  ASSERT_TRUE(write_ok);
  ASSERT_TRUE(read_ok);

  EXPECT_EQ(header_out.problem_category(), LP);
  EXPECT_EQ(header_out.lp_termination_status(), PDLP_OPTIMAL);
  EXPECT_DOUBLE_EQ(header_out.primal_objective(), 42.5);
  EXPECT_DOUBLE_EQ(header_out.solve_time(), 1.23);

  ASSERT_EQ(arrays_out.size(), 2u);
  EXPECT_EQ(arrays_out[RESULT_PRIMAL_SOLUTION], primal);
  EXPECT_EQ(arrays_out[RESULT_DUAL_SOLUTION], dual);
}

TEST(PipeSerialization, Result_MIPFields)
{
  PipePair pp;

  ChunkedResultHeader header;
  header.set_problem_category(MIP);
  header.set_mip_termination_status(MIP_OPTIMAL);
  header.set_mip_objective(99.0);
  header.set_mip_gap(0.001);
  header.set_error_message("");

  auto solution = make_pattern(2000 * 8, 0x33);
  std::map<int32_t, std::vector<uint8_t>> arrays;
  arrays[RESULT_MIP_SOLUTION] = solution;

  bool write_ok = false;
  std::thread writer([&] { write_ok = write_result_to_pipe(pp.write_fd(), header, arrays); });

  ChunkedResultHeader header_out;
  std::map<int32_t, std::vector<uint8_t>> arrays_out;
  bool read_ok = read_result_from_pipe(pp.read_fd(), header_out, arrays_out);

  writer.join();

  ASSERT_TRUE(write_ok);
  ASSERT_TRUE(read_ok);

  EXPECT_EQ(header_out.problem_category(), MIP);
  EXPECT_EQ(header_out.mip_termination_status(), MIP_OPTIMAL);
  EXPECT_DOUBLE_EQ(header_out.mip_objective(), 99.0);

  ASSERT_EQ(arrays_out.size(), 1u);
  EXPECT_EQ(arrays_out[RESULT_MIP_SOLUTION], solution);
}

TEST(PipeSerialization, Result_EmptyArrays)
{
  PipePair pp;

  ChunkedResultHeader header;
  header.set_problem_category(LP);
  header.set_error_message("solver failed");

  std::map<int32_t, std::vector<uint8_t>> arrays;  // no arrays (error case)

  bool write_ok = false;
  std::thread writer([&] { write_ok = write_result_to_pipe(pp.write_fd(), header, arrays); });

  ChunkedResultHeader header_out;
  std::map<int32_t, std::vector<uint8_t>> arrays_out;
  bool read_ok = read_result_from_pipe(pp.read_fd(), header_out, arrays_out);

  writer.join();

  ASSERT_TRUE(write_ok);
  ASSERT_TRUE(read_ok);
  EXPECT_EQ(header_out.error_message(), "solver failed");
  EXPECT_TRUE(arrays_out.empty());
}

// =============================================================================
// Protobuf-only round-trip (write_protobuf_to_pipe / read_protobuf_from_pipe)
// =============================================================================

TEST(PipeSerialization, ProtobufRoundTrip)
{
  PipePair pp;

  ChunkedResultHeader msg;
  msg.set_problem_category(MIP);
  msg.set_primal_objective(3.14);
  msg.set_error_message("hello");

  bool write_ok = false;
  std::thread writer([&] { write_ok = write_protobuf_to_pipe(pp.write_fd(), msg); });

  ChunkedResultHeader msg_out;
  bool read_ok = read_protobuf_from_pipe(pp.read_fd(), msg_out);

  writer.join();

  ASSERT_TRUE(write_ok);
  ASSERT_TRUE(read_ok);
  EXPECT_EQ(msg_out.problem_category(), MIP);
  EXPECT_DOUBLE_EQ(msg_out.primal_objective(), 3.14);
  EXPECT_EQ(msg_out.error_message(), "hello");
}

// =============================================================================
// Larger transfer to exercise multi-iteration pipe I/O
// =============================================================================

TEST(PipeSerialization, Result_LargeArray)
{
  PipePair pp;

  ChunkedResultHeader header;
  header.set_problem_category(LP);
  header.set_primal_objective(0.0);

  // ~4 MiB array — large enough to require many kernel-level pipe iterations.
  constexpr size_t large_size = 4 * 1024 * 1024;
  auto large_data             = make_pattern(large_size, 0x77);

  std::map<int32_t, std::vector<uint8_t>> arrays;
  arrays[RESULT_PRIMAL_SOLUTION] = large_data;

  bool write_ok = false;
  std::thread writer([&] { write_ok = write_result_to_pipe(pp.write_fd(), header, arrays); });

  ChunkedResultHeader header_out;
  std::map<int32_t, std::vector<uint8_t>> arrays_out;
  bool read_ok = read_result_from_pipe(pp.read_fd(), header_out, arrays_out);

  writer.join();

  ASSERT_TRUE(write_ok);
  ASSERT_TRUE(read_ok);
  ASSERT_EQ(arrays_out.size(), 1u);
  EXPECT_EQ(arrays_out[RESULT_PRIMAL_SOLUTION], large_data);
}

// =============================================================================
// serialize_submit_request_to_pipe (pure function, no pipe needed)
// =============================================================================

TEST(PipeSerialization, SerializeSubmitRequest)
{
  SubmitJobRequest request;
  auto* lp = request.mutable_lp_request();
  lp->mutable_header()->set_problem_category(LP);

  auto blob = serialize_submit_request_to_pipe(request);
  ASSERT_FALSE(blob.empty());

  SubmitJobRequest parsed;
  ASSERT_TRUE(parsed.ParseFromArray(blob.data(), static_cast<int>(blob.size())));
  EXPECT_TRUE(parsed.has_lp_request());
  EXPECT_EQ(parsed.lp_request().header().problem_category(), LP);
}
