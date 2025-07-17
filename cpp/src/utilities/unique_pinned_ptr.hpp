/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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