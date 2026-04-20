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

#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <thread>
#include <vector>

// write_to_pipe / read_from_pipe are the real implementations from
// grpc_pipe_io.cpp, compiled directly into this test target.
#include "grpc_pipe_serialization.hpp"

using namespace cuopt::remote;

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
  bool write_ok = false;
  std::thread writer(
    [&] { write_ok = write_chunked_request_to_pipe(pp.write_fd(), header, chunks); });

  ChunkedProblemHeader header_out;
  std::map<int32_t, std::vector<uint8_t>> arrays_out;
  bool read_ok = read_chunked_request_from_pipe(pp.read_fd(), header_out, arrays_out);

  writer.join();

  ASSERT_TRUE(write_ok);
  ASSERT_TRUE(read_ok);

  EXPECT_TRUE(header_out.maximize());
  EXPECT_DOUBLE_EQ(header_out.objective_scaling_factor(), 2.5);
  EXPECT_EQ(header_out.problem_name(), "test_lp");

  ASSERT_EQ(arrays_out.size(), 2u);
  EXPECT_EQ(arrays_out[FIELD_C], c_data);
  EXPECT_EQ(arrays_out[FIELD_A_INDICES], i_data);
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

  bool write_ok = false;
  std::thread writer(
    [&] { write_ok = write_chunked_request_to_pipe(pp.write_fd(), header, chunks); });

  ChunkedProblemHeader header_out;
  std::map<int32_t, std::vector<uint8_t>> arrays_out;
  bool read_ok = read_chunked_request_from_pipe(pp.read_fd(), header_out, arrays_out);

  writer.join();

  ASSERT_TRUE(write_ok);
  ASSERT_TRUE(read_ok);
  ASSERT_EQ(arrays_out.size(), 1u);
  EXPECT_EQ(arrays_out[FIELD_C], full_data);
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

  bool write_ok = false;
  std::thread writer(
    [&] { write_ok = write_chunked_request_to_pipe(pp.write_fd(), header, chunks); });

  ChunkedProblemHeader header_out;
  std::map<int32_t, std::vector<uint8_t>> arrays_out;
  bool read_ok = read_chunked_request_from_pipe(pp.read_fd(), header_out, arrays_out);

  writer.join();

  ASSERT_TRUE(write_ok);
  ASSERT_TRUE(read_ok);
  EXPECT_EQ(header_out.problem_name(), "empty");
  ASSERT_EQ(arrays_out.size(), 1u);
  EXPECT_TRUE(arrays_out[FIELD_C].empty());
}

TEST(PipeSerialization, ChunkedRequest_NoChunks)
{
  PipePair pp;

  ChunkedProblemHeader header;
  header.set_problem_name("header_only");

  std::vector<ArrayChunk> chunks;  // no chunks at all

  bool write_ok = false;
  std::thread writer(
    [&] { write_ok = write_chunked_request_to_pipe(pp.write_fd(), header, chunks); });

  ChunkedProblemHeader header_out;
  std::map<int32_t, std::vector<uint8_t>> arrays_out;
  bool read_ok = read_chunked_request_from_pipe(pp.read_fd(), header_out, arrays_out);

  writer.join();

  ASSERT_TRUE(write_ok);
  ASSERT_TRUE(read_ok);
  EXPECT_EQ(header_out.problem_name(), "header_only");
  EXPECT_TRUE(arrays_out.empty());
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
    int64_t es = array_field_element_size(tf.id);
    auto data  = make_pattern(static_cast<size_t>(tf.elements * es), static_cast<uint8_t>(i));
    expected[static_cast<int32_t>(tf.id)] = data;
    chunks.push_back(make_whole_chunk(tf.id, tf.elements, data));
  }

  bool write_ok = false;
  std::thread writer(
    [&] { write_ok = write_chunked_request_to_pipe(pp.write_fd(), header, chunks); });

  ChunkedProblemHeader header_out;
  std::map<int32_t, std::vector<uint8_t>> arrays_out;
  bool read_ok = read_chunked_request_from_pipe(pp.read_fd(), header_out, arrays_out);

  writer.join();

  ASSERT_TRUE(write_ok);
  ASSERT_TRUE(read_ok);
  ASSERT_EQ(arrays_out.size(), expected.size());
  for (const auto& [fid, data] : expected) {
    ASSERT_TRUE(arrays_out.count(fid)) << "Missing field_id " << fid;
    EXPECT_EQ(arrays_out[fid], data) << "Mismatch for field_id " << fid;
  }
}

// =============================================================================
// Result round-trip tests
// =============================================================================

TEST(PipeSerialization, Result_RoundTrip)
{
  PipePair pp;

  ChunkedResultHeader header;
  header.set_problem_category(cuopt::remote::LP);
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

  EXPECT_EQ(header_out.problem_category(), cuopt::remote::LP);
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
  header.set_problem_category(cuopt::remote::MIP);
  header.set_mip_termination_status(MIP_OPTIMAL);
  header.set_mip_objective(99.0);
  header.set_mip_gap(0.001);
  header.set_mip_error_message("");

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

  EXPECT_EQ(header_out.problem_category(), cuopt::remote::MIP);
  EXPECT_EQ(header_out.mip_termination_status(), MIP_OPTIMAL);
  EXPECT_DOUBLE_EQ(header_out.mip_objective(), 99.0);

  ASSERT_EQ(arrays_out.size(), 1u);
  EXPECT_EQ(arrays_out[RESULT_MIP_SOLUTION], solution);
}

TEST(PipeSerialization, Result_EmptyArrays)
{
  PipePair pp;

  ChunkedResultHeader header;
  header.set_problem_category(cuopt::remote::LP);
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
  msg.set_problem_category(cuopt::remote::MIP);
  msg.set_primal_objective(3.14);
  msg.set_error_message("hello");

  bool write_ok = false;
  std::thread writer([&] { write_ok = write_protobuf_to_pipe(pp.write_fd(), msg); });

  ChunkedResultHeader msg_out;
  bool read_ok = read_protobuf_from_pipe(pp.read_fd(), msg_out);

  writer.join();

  ASSERT_TRUE(write_ok);
  ASSERT_TRUE(read_ok);
  EXPECT_EQ(msg_out.problem_category(), cuopt::remote::MIP);
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
  header.set_problem_category(cuopt::remote::LP);
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
