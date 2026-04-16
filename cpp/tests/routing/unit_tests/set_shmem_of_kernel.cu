/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/cuda_helpers.cuh>

#include <utilities/base_fixture.hpp>

#include <gtest/gtest.h>

namespace cuopt {
namespace test {

/// @brief Dummy kernel used to test a zero-byte shared-memory request.
__global__ void kernel_zero() {}
/// @brief Dummy kernel used to test a normal (within-limit) shared-memory request.
__global__ void kernel_normal() {}
/// @brief Dummy kernel used to test a too-large shared-memory request (first call).
__global__ void kernel_too_large_a() {}
/// @brief Dummy kernel used to test a too-large shared-memory request (repeated call).
__global__ void kernel_too_large_b() {}
/// @brief Dummy kernel used to verify that a failed request leaves no sticky CUDA error.
__global__ void kernel_sticky_error() {}

/// @brief Zero request is a no-op and must return true.
TEST(set_shmem_of_kernel, zero_request)
{
  EXPECT_TRUE(set_shmem_of_kernel(kernel_zero, 0));
  EXPECT_EQ(cudaSuccess, cudaGetLastError());
}

/// @brief A modest request well within device limits must succeed.
TEST(set_shmem_of_kernel, normal_request)
{
  EXPECT_TRUE(set_shmem_of_kernel(kernel_normal, 4096));
  EXPECT_EQ(cudaSuccess, cudaGetLastError());
}

/// @brief Requesting more shared memory than the device supports must return false.
TEST(set_shmem_of_kernel, too_large_returns_false)
{
  int shmem_max{};
  ASSERT_EQ(cudaSuccess,
            cudaDeviceGetAttribute(&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0))
    << "cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin) failed";
  size_t too_large = static_cast<size_t>(shmem_max) + 1024;

  EXPECT_FALSE(set_shmem_of_kernel(kernel_too_large_a, too_large));
  EXPECT_EQ(cudaSuccess, cudaGetLastError());
}

/// @brief A second call with the same too-large size must still return false.
TEST(set_shmem_of_kernel, cache_not_poisoned_on_failure)
{
  int shmem_max{};
  ASSERT_EQ(cudaSuccess,
            cudaDeviceGetAttribute(&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0))
    << "cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin) failed";
  size_t too_large = static_cast<size_t>(shmem_max) + 1024;

  EXPECT_FALSE(set_shmem_of_kernel(kernel_too_large_b, too_large));
  EXPECT_FALSE(set_shmem_of_kernel(kernel_too_large_b, too_large));  // must not return true
  EXPECT_EQ(cudaSuccess, cudaGetLastError());
}

/// @brief A failed call must not leave a sticky CUDA error that would be caught
/// later by an unrelated RAFT_CHECK_CUDA.
TEST(set_shmem_of_kernel, no_sticky_error_after_failure)
{
  int shmem_max{};
  ASSERT_EQ(cudaSuccess,
            cudaDeviceGetAttribute(&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0))
    << "cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin) failed";
  size_t too_large = static_cast<size_t>(shmem_max) + 1024;

  EXPECT_FALSE(
    set_shmem_of_kernel(kernel_sticky_error, too_large));  // confirm failure branch taken
  EXPECT_EQ(cudaSuccess, cudaGetLastError());
}

}  // namespace test
}  // namespace cuopt
