/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/macros.cuh>

#include <cstdint>

#include <thrust/host_vector.h>
#include <thrust/tuple.h>
#include <mutex>
#include <raft/core/device_span.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <shared_mutex>
#include <unordered_map>

namespace cuopt {

#if !defined(__CUDA_ARCH__)
/** Host-only: runtime kernel symbol name for logging (CUDA 12+); otherwise nullptr. */
inline const char* kernel_host_pointer_symbol_name(const void* kernel_host_ptr)
{
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12000)
  const char* name = nullptr;
  if (cudaFuncGetName(&name, kernel_host_ptr) == cudaSuccess && name != nullptr) { return name; }
#else
  (void)kernel_host_ptr;
#endif
  return nullptr;
}
#endif  // !defined(__CUDA_ARCH__)

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

static DI std::uintptr_t shmem_align_address_up(std::uintptr_t addr, std::uintptr_t alignment)
{
  return (addr + alignment - 1u) & ~(alignment - 1u);
}

// Bump-pointer sub-allocator for dynamic shared memory: each segment starts and ends on an 8-byte
// boundary so mixed float/int/double layouts stay valid. Callers sizing shmem should reserve slack
// (e.g. align totals to 8 or add ~7 bytes per segment) if buffers are tightly packed.
template <typename T, typename i_t>
static DI thrust::tuple<raft::device_span<T>, i_t*> wrap_ptr_as_span(i_t* shmem, size_t sz)
{
  constexpr std::uintptr_t k_align = 8;

  std::uintptr_t base = shmem_align_address_up(
    reinterpret_cast<std::uintptr_t>(reinterpret_cast<void*>(shmem)), k_align);
  T* sh_ptr = reinterpret_cast<T*>(base);
  auto s    = raft::device_span<T>{sh_ptr, sz};

  std::uintptr_t end = reinterpret_cast<std::uintptr_t>(reinterpret_cast<void*>(sh_ptr + sz));
  end                = shmem_align_address_up(end, k_align);
  return thrust::make_tuple(s, reinterpret_cast<i_t*>(end));
}

template <class To, class From>
HDI To bit_cast(const From& src)
{
  static_assert(sizeof(To) == sizeof(From));
  return *(To*)(&src);
}

/**
 * @brief Raises the dynamic shared-memory limit for a CUDA kernel, with caching.
 *
 * Calls cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize) only when
 * @p dynamic_request_size exceeds the previously set limit for @p function.  The
 * per-kernel high-water mark is stored in a process-wide cache so that repeated
 * calls with the same or smaller sizes are cheap shared-lock reads.
 *
 * Thread safety: safe to call concurrently from multiple host threads.
 *
 * @param function             Host pointer to the __global__ kernel function.
 * @param dynamic_request_size Requested dynamic shared memory in bytes.
 *                             A value of 0 is a no-op and always returns true.
 * @return true  if the attribute was successfully set (or was already sufficient).
 * @return false if cudaFuncSetAttribute failed (e.g. size exceeds device limit);
 *               the sticky CUDA error is consumed so it cannot surface later.
 */
template <typename Function>
inline bool set_shmem_of_kernel(Function* function, size_t dynamic_request_size)
{
  static std::shared_mutex mtx;
  static std::unordered_map<Function*, size_t> shmem_sizes;

  if (dynamic_request_size != 0) {
    dynamic_request_size = raft::alignTo(dynamic_request_size, size_t(1024));

    {
      std::shared_lock<std::shared_mutex> rlock(mtx);
      auto it = shmem_sizes.find(function);
      if (it != shmem_sizes.end() && dynamic_request_size <= it->second) { return true; }
    }

    std::unique_lock<std::shared_mutex> wlock(mtx);
    size_t current_size = shmem_sizes.count(function) ? shmem_sizes[function] : 0;
    if (dynamic_request_size > current_size) {
      auto err = cudaFuncSetAttribute(
        function, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_request_size);
      if (err == cudaSuccess) {
        shmem_sizes[function] = dynamic_request_size;
        return true;
      } else {
#if 0
#if !defined(__CUDA_ARCH__)
        const void* const entry = reinterpret_cast<const void*>(function);
        const char* const sym  = kernel_host_pointer_symbol_name(entry);
        fprintf(stderr,
                "Could not set shared memory size for kernel %s (%p) to %zu: %s\n",
                sym != nullptr ? sym : "(unknown)",
                entry,
                dynamic_request_size,
                cudaGetErrorString(err));
#else
        fprintf(stderr,
                "Could not set shared memory size for kernel (%p) to %zu: %s\n",
                reinterpret_cast<const void*>(function),
                dynamic_request_size,
                cudaGetErrorString(err));
#endif
        exit(1);
#endif
        cudaGetLastError();  // clear sticky error so later RAFT_CHECK_CUDA doesn't catch it
        return false;
      }
    }
  }
  return true;
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
  size_t free_mem, total_mem;
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_mem, &total_mem));
  // TODO (bdice): Restore limiting adaptor check after updating CCCL to support resource_cast
  return total_mem;
}

}  // namespace cuopt
