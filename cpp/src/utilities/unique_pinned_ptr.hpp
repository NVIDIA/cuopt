/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <memory>

#include <cuda_runtime.h>

// This is a temporary solution to replace thrust::host_pinned_vector while this bug is not fixed:
// https://github.com/NVIDIA/cccl/issues/5027

namespace cuopt {

// Custom deleter using cudaFreeHost
template <typename T>
struct cuda_host_deleter {
  void operator()(T* ptr) const
  {
    if (ptr != nullptr) RAFT_CUDA_TRY(cudaFreeHost(ptr));
  }
};

// Creates a unique_ptr using cudaMallocHost
template <typename T>
std::unique_ptr<T, cuda_host_deleter<T>> make_unique_cuda_host_pinned()
{
  T* ptr = nullptr;
  RAFT_CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&ptr), sizeof(T)));
  return std::unique_ptr<T, cuda_host_deleter<T>>(ptr);
}

}  // namespace cuopt
