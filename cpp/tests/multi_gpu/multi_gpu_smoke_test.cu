/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Minimal multi-GPU smoke test for the 2-GPU CI runner.
//
// It verifies that (a) at least two GPUs are visible and (b) data can be moved
// directly from one device to another (peer-to-peer copy). This exercises the
// 2-GPU runner end to end using only the CUDA runtime — no NCCL/MPI — so it
// builds and runs independently of the distributed-PDLP work. NCCL-level
// communication coverage arrives with that work (cuD-PDLP).
//
// Named with the *_MG_TEST suffix so ci/test_cpp_multi_gpu.sh discovers it and
// ci/run_ctests.sh (the single-GPU suite) skips it. As a further safeguard the
// tests GTEST_SKIP when fewer than two GPUs are present, so the binary is
// harmless if ever run on a single-GPU host.

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstring>
#include <vector>

namespace cuopt {

namespace {

void check(cudaError_t status, const char* what)
{
  ASSERT_EQ(status, cudaSuccess) << what << ": " << cudaGetErrorString(status);
}

int visible_gpu_count()
{
  int count = 0;
  EXPECT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  return count;
}

}  // namespace

// The whole point of this job: confirm the runner actually exposes >=2 GPUs.
TEST(MultiGpuSmoke, DeviceCountAtLeastTwo)
{
  const int count = visible_gpu_count();
  if (count < 2) { GTEST_SKIP() << "requires >=2 GPUs, found " << count; }
  EXPECT_GE(count, 2);
}

// Move a buffer from device 0 to device 1 and read it back, validating that
// cross-GPU data movement works on this runner.
TEST(MultiGpuSmoke, PeerToPeerCopy)
{
  const int count = visible_gpu_count();
  if (count < 2) { GTEST_SKIP() << "requires >=2 GPUs, found " << count; }

  constexpr int n_elements = 1024;
  constexpr size_t n_bytes = n_elements * sizeof(int);
  constexpr int fill_byte  = 0x2A;  // arbitrary sentinel

  int* d0 = nullptr;
  int* d1 = nullptr;

  check(cudaSetDevice(0), "cudaSetDevice(0)");
  check(cudaMalloc(&d0, n_bytes), "cudaMalloc(device 0)");
  check(cudaMemset(d0, fill_byte, n_bytes), "cudaMemset(device 0)");

  check(cudaSetDevice(1), "cudaSetDevice(1)");
  check(cudaMalloc(&d1, n_bytes), "cudaMalloc(device 1)");

  // Direct device-0 -> device-1 transfer (uses P2P when available, staging via
  // host otherwise). Either way it must produce the correct bytes on device 1.
  check(cudaMemcpyPeer(d1, 1, d0, 0, n_bytes), "cudaMemcpyPeer(0 -> 1)");

  std::vector<int> host(n_elements, 0);
  check(cudaMemcpy(host.data(), d1, n_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H)");

  int expected = 0;
  std::memset(&expected, fill_byte, sizeof(int));
  for (int value : host) {
    ASSERT_EQ(value, expected);
  }

  check(cudaSetDevice(0), "cudaSetDevice(0)");
  check(cudaFree(d0), "cudaFree(device 0)");
  check(cudaSetDevice(1), "cudaSetDevice(1)");
  check(cudaFree(d1), "cudaFree(device 1)");
}

}  // namespace cuopt
