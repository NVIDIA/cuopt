/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <utilities/macros.cuh>

#include <thrust/host_vector.h>
#include <raft/core/device_span.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/limiting_resource_adaptor.hpp>

namespace cuopt {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#error "cuOpt is only supported on Volta and newer architectures"
#endif

/** helper macro for device inlined functions */
#define DI  inline __device__
#define HDI inline __host__ __device__
#define HD  __host__ __device__

/**
 * For Pascal independent thread scheduling is not supported so we are using a seperate
 * add version. This version will return when there are duplicates instead of
 * udapting the key with the min value. Another approach would be to use a 64 bit
 * representation for values and predecessors and use atomicMin. This comes with
 * accuracy trade-offs. Hence the seperate add function for Pascal.
 **/
template <typename i_t>
DI bool acquire_lock(i_t* lock)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
  auto res = atomicCAS(lock, 0, 1);
  __threadfence();
  return res == 0;
#else
  while (atomicCAS(lock, 0, 1)) {
    __nanosleep(100);
  }
  __threadfence();
  return true;
#endif
}

template <typename i_t>
DI void release_lock(i_t* lock)
{
  __threadfence();
  atomicExch(lock, 0);
}

template <typename i_t>
DI bool try_acquire_lock_block(i_t* lock)
{
  auto res = atomicCAS_block(lock, 0, 1);
  __threadfence_block();
  return res == 0;
}

template <typename i_t>
DI bool acquire_lock_block(i_t* lock)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
  return try_acquire_lock_block(lock);
#else
  while (atomicCAS_block(lock, 0, 1)) {
    __nanosleep(100);
  }
  __threadfence_block();
  return true;
#endif
}

template <typename i_t>
DI void release_lock_block(i_t* lock)
{
  __threadfence_block();
  atomicExch_block(lock, 0);
}

template <typename T>
DI void init_shmem(T& shmem, T val)
{
  if (threadIdx.x == 0) { shmem = val; }
}

template <typename T>
DI void init_block_shmem(T* shmem, T val, size_t size)
{
  for (auto i = threadIdx.x; i < size; i += blockDim.x) {
    shmem[i] = val;
  }
}

template <typename T>
DI void init_block_shmem(raft::device_span<T> sh_span, T val)
{
  init_block_shmem(sh_span.data(), val, sh_span.size());
}

template <typename T>
DI void block_sequence(T* arr, const size_t size)
{
  for (auto i = threadIdx.x; i < size; i += blockDim.x) {
    arr[i] = i;
  }
}

template <typename T>
DI void block_copy(T* dst, const T* src, const size_t size)
{
  for (auto i = threadIdx.x; i < size; i += blockDim.x) {
    dst[i] = src[i];
  }
}

template <typename T>
DI void block_copy(raft::device_span<T> dst,
                   const raft::device_span<const T> src,
                   const size_t size)
{
  cuopt_assert(src.size() >= size, "block_copy::src does not have the sufficient size");
  cuopt_assert(dst.size() >= size, "block_copy::dst does not have the sufficient size");
  block_copy(dst.data(), src.data(), size);
}

template <typename T>
DI void block_copy(raft::device_span<T> dst, const raft::device_span<T> src, const size_t size)
{
  cuopt_assert(src.size() >= size, "block_copy::src does not have the sufficient size");
  cuopt_assert(dst.size() >= size, "block_copy::dst does not have the sufficient size");
  block_copy(dst.data(), src.data(), size);
}

template <typename T>
DI void block_copy(raft::device_span<T> dst, const raft::device_span<T> src)
{
  cuopt_assert(dst.size() >= src.size(), "");
  block_copy(dst, src, src.size());
}

template <typename i_t>
i_t next_pow2(i_t val)
{
  return 1 << (raft::log2(val) + 1);
}

// FIXME:: handle alignment when dealing with different sized precisions
template <typename T, typename i_t>
static DI thrust::tuple<raft::device_span<T>, i_t*> wrap_ptr_as_span(i_t* shmem, size_t sz)
{
  T* sh_ptr = (T*)shmem;
  auto s    = raft::device_span<T>{sh_ptr, sz};

  sh_ptr = sh_ptr + sz;
  return thrust::make_tuple(s, (i_t*)sh_ptr);
}

template <class To, class From>
HDI To bit_cast(const From& src)
{
  static_assert(sizeof(To) == sizeof(From));
  return *(To*)(&src);
}

template <typename Function>
inline bool set_shmem_of_kernel(Function* function, size_t dynamic_request_size)
{
  if (dynamic_request_size != 0) {
    dynamic_request_size = raft::alignTo(dynamic_request_size, size_t(1024));
    cudaFuncSetAttribute(
      function, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_request_size);
    return (cudaSuccess == cudaGetLastError());
  } else {
    return true;
  }
}

template <typename T>
DI void sorted_insert(T* array, T item, int curr_size, int max_size)
{
  for (int i = curr_size - 1; i >= 0; --i) {
    if (i == max_size - 1) continue;
    if (array[i] < item) {
      array[i + 1] = item;
      return;
    } else {
      array[i + 1] = array[i];
    }
  }
  array[0] = item;
}

inline size_t get_device_memory_size()
{
  // Otherwise, we need to get the free memory from the device
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);

  auto res = rmm::mr::get_current_device_resource();
  auto limiting_adaptor =
    dynamic_cast<rmm::mr::limiting_resource_adaptor<rmm::mr::cuda_async_memory_resource>*>(res);
  // Did we specifiy an explicit memory limit?
  if (limiting_adaptor) {
    printf("limiting_adaptor->get_allocation_limit(): %fMiB\n",
           limiting_adaptor->get_allocation_limit() / (double)1e6);
    printf("used_mem: %fMiB\n", limiting_adaptor->get_allocated_bytes() / (double)1e6);
    printf("free_mem: %fMiB\n",
           (limiting_adaptor->get_allocation_limit() - limiting_adaptor->get_allocated_bytes()) /
             (double)1e6);
    return std::min(total_mem, limiting_adaptor->get_allocation_limit());
  } else {
    return total_mem;
  }
}

}  // namespace cuopt
