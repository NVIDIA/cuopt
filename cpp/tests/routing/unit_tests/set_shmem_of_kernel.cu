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

__global__ void kernel_zero() {}
__global__ void kernel_normal() {}
__global__ void kernel_too_large_a() {}
__global__ void kernel_too_large_b() {}
__global__ void kernel_sticky_error() {}

// Zero request is a no-op and must return true.
TEST(set_shmem_of_kernel, zero_request)
{
  EXPECT_TRUE(set_shmem_of_kernel(kernel_zero, 0));
  EXPECT_EQ(cudaSuccess, cudaGetLastError());
}

// A modest request well within device limits must succeed.
TEST(set_shmem_of_kernel, normal_request)
{
  EXPECT_TRUE(set_shmem_of_kernel(kernel_normal, 4096));
  EXPECT_EQ(cudaSuccess, cudaGetLastError());
}

// Requesting more shared memory than the device supports must return false.
TEST(set_shmem_of_kernel, too_large_returns_false)
{
  int shmem_max{};
  cudaDeviceGetAttribute(&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  size_t too_large = static_cast<size_t>(shmem_max) + 1024;

  EXPECT_FALSE(set_shmem_of_kernel(kernel_too_large_a, too_large));
  EXPECT_EQ(cudaSuccess, cudaGetLastError());
}

// A second call with the same too-large size must still return false
TEST(set_shmem_of_kernel, cache_not_poisoned_on_failure)
{
  int shmem_max{};
  cudaDeviceGetAttribute(&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  size_t too_large = static_cast<size_t>(shmem_max) + 1024;

  EXPECT_FALSE(set_shmem_of_kernel(kernel_too_large_b, too_large));
  EXPECT_FALSE(set_shmem_of_kernel(kernel_too_large_b, too_large));  // must not return true
  EXPECT_EQ(cudaSuccess, cudaGetLastError());
}

// A failed call must not leave a sticky CUDA error that would be caught
// later by an unrelated RAFT_CHECK_CUDA.
TEST(set_shmem_of_kernel, no_sticky_error_after_failure)
{
  int shmem_max{};
  cudaDeviceGetAttribute(&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  size_t too_large = static_cast<size_t>(shmem_max) + 1024;

  set_shmem_of_kernel(kernel_sticky_error, too_large);
  EXPECT_EQ(cudaSuccess, cudaGetLastError());
}

}  // namespace test
}  // namespace cuopt
