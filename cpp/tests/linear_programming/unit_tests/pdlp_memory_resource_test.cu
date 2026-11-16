/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <pdlp/pdlp_memory_resource.hpp>

#include <raft/core/handle.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <cuda.h>
#include <gtest/gtest.h>
#include <omp.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <vector>

namespace cuopt::mathematical_optimization::pdlp::test {
namespace {

bool supports_compression()
{
  RAFT_CUDA_TRY(cudaFree(nullptr));
  CUdevice device{};
  int compression{}, vmm{};
  if (cuCtxGetDevice(&device) != CUDA_SUCCESS ||
      cuDeviceGetAttribute(
        &vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device) != CUDA_SUCCESS ||
      cuDeviceGetAttribute(
        &compression, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, device) != CUDA_SUCCESS) {
    return false;
  }
  return vmm && compression;
}

void check_bytes(rmm::device_buffer& buffer, cuda::stream_ref stream)
{
  std::vector<unsigned char> expected(buffer.size(), 0x5a), actual(buffer.size());
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    buffer.data(), expected.data(), expected.size(), cudaMemcpyHostToDevice, stream.get()));
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    actual.data(), buffer.data(), actual.size(), cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();
  EXPECT_EQ(actual, expected);
}

void allocate_on_independent_stream(int device, device_memory_resource_t resource)
{
  RAFT_CUDA_TRY(cudaSetDevice(device));
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  for (int iteration = 0; iteration < 8; ++iteration) {
    rmm::device_buffer buffer(4096 + iteration * 256, cuda::stream_ref{stream}, resource);
    check_bytes(buffer, cuda::stream_ref{stream});
  }
}

TEST(PdlpMemoryResource, BufferRetainsResourceAfterFactoryOwnerIsDestroyed)
{
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  std::optional<rmm::device_buffer> buffer;
  auto before = rmm::mr::get_current_device_resource_ref();
  {
    auto resource = make_pdlp_memory_resource(rmm::mr::cuda_memory_resource{});
    auto copy     = resource;
    buffer.emplace(1024 * 1024, cuda::stream_ref{stream}, copy);
  }
  check_bytes(*buffer, cuda::stream_ref{stream});
  buffer->resize(2 * 1024 * 1024, cuda::stream_ref{stream});
  check_bytes(*buffer, cuda::stream_ref{stream});
  EXPECT_EQ(before, rmm::mr::get_current_device_resource_ref());
}

TEST(PdlpMemoryResource, PreservesCallerResourceAndItsLifetime)
{
  struct state_t {
    rmm::mr::cuda_memory_resource cuda;
    std::atomic<int> allocations{0};
    std::atomic<int> deallocations{0};
  };
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  auto state                  = std::make_shared<state_t>();
  std::weak_ptr<state_t> weak = state;
  std::optional<rmm::device_buffer> buffer;
  {
    device_memory_resource_t original{rmm::mr::callback_memory_resource{
      [state](cuda::stream_ref s, std::size_t bytes, std::size_t alignment, void*) {
        ++state->allocations;
        return state->cuda.allocate(s, bytes, alignment);
      },
      [state](cuda::stream_ref s, void* pointer, std::size_t bytes, std::size_t alignment, void*) {
        ++state->deallocations;
        state->cuda.deallocate(s, pointer, bytes, alignment);
      }}};
    auto resource = make_pdlp_memory_resource(original);
    EXPECT_EQ(resource, original);
    EXPECT_FALSE(is_pdlp_memory_resource(resource));
    buffer.emplace(1024, cuda::stream_ref{stream}, resource);
  }
  EXPECT_EQ(state->allocations.load(), 1);
  EXPECT_EQ(state->deallocations.load(), 0);
  state.reset();
  EXPECT_FALSE(weak.expired());
  check_bytes(*buffer, cuda::stream_ref{stream});
  buffer.reset();
  EXPECT_TRUE(weak.expired());
}

TEST(PdlpMemoryResource, CompressionPropertiesAreVisibleOnSuballocations)
{
  if (!supports_compression()) { GTEST_SKIP() << "Device has no generic memory compression"; }
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  auto resource = make_pdlp_memory_resource(rmm::mr::cuda_memory_resource{});
  {
    rmm::device_buffer reserve(4 * 1024 * 1024, cuda::stream_ref{stream}, resource);
  }
  rmm::device_buffer first(256, cuda::stream_ref{stream}, resource);
  rmm::device_buffer second(256, cuda::stream_ref{stream}, resource);
  CUmemGenericAllocationHandle allocation{};
  ASSERT_EQ(cuMemRetainAllocationHandle(&allocation, second.data()), CUDA_SUCCESS);
  CUmemAllocationProp properties{};
  auto const status = cuMemGetAllocationPropertiesFromHandle(&properties, allocation);
  EXPECT_EQ(cuMemRelease(allocation), CUDA_SUCCESS);
  EXPECT_TRUE(is_pdlp_memory_resource(resource));
  ASSERT_EQ(status, CUDA_SUCCESS);
  // The driver may decline the compression hint when hardware resources are exhausted.
  if (properties.allocFlags.compressionType != CU_MEM_ALLOCATION_COMP_GENERIC) {
    GTEST_SKIP() << "Driver did not grant the optional compression attribute";
  }
  check_bytes(first, cuda::stream_ref{stream});
  check_bytes(second, cuda::stream_ref{stream});
}

TEST(PdlpMemoryResource, TypeErasedDefaultResourceRetainsConcreteType)
{
  rmm::mr::cuda_memory_resource original;
  rmm::device_async_resource_ref reference{original};
  device_memory_resource_t owned = reference;
  ASSERT_NE(cuda::mr::resource_cast<rmm::mr::cuda_memory_resource>(&owned), nullptr);
  auto resource = make_pdlp_memory_resource(reference);
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  rmm::device_buffer buffer(1024, cuda::stream_ref{stream}, resource);
  check_bytes(buffer, cuda::stream_ref{stream});
  if (supports_compression()) {
    EXPECT_EQ(cuda::mr::resource_cast<rmm::mr::cuda_memory_resource>(&resource), nullptr);
  }
}

TEST(PdlpMemoryResource, HonorsAlignmentAbovePoolAlignment)
{
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  auto resource = make_pdlp_memory_resource(rmm::mr::cuda_memory_resource{});
  rmm::device_buffer buffer(4097, 4096, cuda::stream_ref{stream}, resource);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buffer.data()) % 4096, 0);
  check_bytes(buffer, cuda::stream_ref{stream});
}

TEST(PdlpMemoryResource, IndependentStreamsShareOwningResourceSafely)
{
  int device{};
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  auto resource = make_pdlp_memory_resource(rmm::mr::cuda_memory_resource{});
  std::array<std::exception_ptr, 2> failures;
#pragma omp parallel num_threads(2)
  {
    auto const thread = omp_get_thread_num();
    try {
      allocate_on_independent_stream(device, resource);
    } catch (...) {
      failures[thread] = std::current_exception();
    }
  }
  for (auto const& failure : failures) {
    if (failure) { std::rethrow_exception(failure); }
  }
}

TEST(PdlpMemoryResource, LastOwnerWaitsForNonblockingStreamUses)
{
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  cudaEvent_t completed{};
  RAFT_CUDA_TRY(cudaEventCreateWithFlags(&completed, cudaEventDisableTiming));
  {
    auto resource = make_pdlp_memory_resource(rmm::mr::cuda_memory_resource{});
    rmm::device_buffer buffer(8 * 1024 * 1024, cuda::stream_ref{stream}, resource);
    RAFT_CUDA_TRY(cudaMemsetAsync(buffer.data(), 0x7f, buffer.size(), stream.value()));
    RAFT_CUDA_TRY(cudaEventRecord(completed, stream.value()));
    // Both the client allocation and the last resource owner die without a caller synchronize.
  }
  EXPECT_EQ(cudaEventQuery(completed), cudaSuccess);
  RAFT_CUDA_TRY(cudaEventDestroy(completed));
}

TEST(PdlpMemoryResource, DetachedOutputDoesNotRetainWorkingMappings)
{
  if (!supports_compression()) { GTEST_SKIP() << "Device has no generic memory compression"; }
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  device_memory_resource_t output_resource{rmm::mr::cuda_memory_resource{}};
  std::optional<rmm::device_uvector<int>> output;
  void* working_address{};
  const std::vector<int> expected(256, 42);
  {
    auto working_resource = make_pdlp_memory_resource(output_resource);
    {
      rmm::device_buffer workspace(8 * 1024 * 1024, cuda::stream_ref{stream}, working_resource);
      working_address = workspace.data();
      output.emplace(expected.size(), cuda::stream_ref{stream}, working_resource);
      RAFT_CUDA_TRY(cudaMemcpyAsync(output->data(),
                                    expected.data(),
                                    expected.size() * sizeof(int),
                                    cudaMemcpyHostToDevice,
                                    stream.value()));
    }
    CUmemGenericAllocationHandle mapping{};
    ASSERT_EQ(cuMemRetainAllocationHandle(&mapping, working_address), CUDA_SUCCESS);
    ASSERT_EQ(cuMemRelease(mapping), CUDA_SUCCESS);
    detach_pdlp_buffer(*output, working_resource, output_resource, cuda::stream_ref{stream});
    stream.synchronize();
  }
  CUmemGenericAllocationHandle mapping{};
  auto const status = cuMemRetainAllocationHandle(&mapping, working_address);
  if (status == CUDA_SUCCESS) { EXPECT_EQ(cuMemRelease(mapping), CUDA_SUCCESS); }
  EXPECT_NE(status, CUDA_SUCCESS);
  EXPECT_EQ(output->memory_resource(), output_resource);
  std::vector<int> actual(expected.size());
  RAFT_CUDA_TRY(cudaMemcpyAsync(actual.data(),
                                output->data(),
                                actual.size() * sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();
  EXPECT_EQ(actual, expected);
}

TEST(PdlpMemoryResource, DetachLeavesCallerOwnedBufferInPlace)
{
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  device_memory_resource_t output_resource{rmm::mr::cuda_memory_resource{}};
  auto working_resource = make_pdlp_memory_resource(output_resource);
  rmm::device_uvector<int> output(256, cuda::stream_ref{stream}, output_resource);
  auto* original = output.data();
  detach_pdlp_buffer(output, working_resource, output_resource, cuda::stream_ref{stream});
  EXPECT_EQ(output.data(), original);
  detach_pdlp_buffer(output, output_resource, output_resource, cuda::stream_ref{stream});
  EXPECT_EQ(output.data(), original);
}

TEST(PdlpMemoryResource, UnsupportedHardwareKeepsOriginalResource)
{
  if (supports_compression()) { GTEST_SKIP() << "Device supports generic memory compression"; }
  device_memory_resource_t original{rmm::mr::cuda_memory_resource{}};
  EXPECT_EQ(make_pdlp_memory_resource(original), original);
}

}  // namespace
}  // namespace cuopt::mathematical_optimization::pdlp::test
