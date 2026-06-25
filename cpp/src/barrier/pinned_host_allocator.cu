/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <raft/util/cuda_utils.cuh>

#include <dual_simplex/types.hpp>
#include <dual_simplex/vector_math.hpp>

namespace cuopt::mathematical_optimization::barrier {

template <typename T>
struct PinnedHostAllocator {
  using value_type = T;

  PinnedHostAllocator() noexcept {}
  template <class U>
  PinnedHostAllocator(const PinnedHostAllocator<U>&) noexcept
  {
  }

  T* allocate(std::size_t n)
  {
    T* ptr = nullptr;
    RAFT_CUDA_TRY(cudaMallocHost((void**)&ptr, n * sizeof(T)));
    return ptr;
  }

  void deallocate(T* p, std::size_t) { RAFT_CUDA_TRY(cudaFreeHost(p)); }
};

template <typename T, typename U>
bool operator==(const PinnedHostAllocator<T>&, const PinnedHostAllocator<U>&) noexcept
{
  return true;
}
template <typename T, typename U>
bool operator!=(const PinnedHostAllocator<T>&, const PinnedHostAllocator<U>&) noexcept
{
  return false;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class PinnedHostAllocator<double>;

template bool operator==(const PinnedHostAllocator<double>&,
                         const PinnedHostAllocator<double>&) noexcept;
template bool operator!=(const PinnedHostAllocator<double>&,
                         const PinnedHostAllocator<double>&) noexcept;
#endif
template class PinnedHostAllocator<int>;

}  // namespace cuopt::mathematical_optimization::barrier

namespace cuopt::mathematical_optimization::simplex {

// Explicit instantiation of the shared simplex vector_math template with
// barrier's PinnedHostAllocator must live in simplex (the template's namespace).
#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template double
vector_norm_inf<int,
                double,
                cuopt::mathematical_optimization::barrier::PinnedHostAllocator<double>>(
  const std::vector<double, cuopt::mathematical_optimization::barrier::PinnedHostAllocator<double>>&
    x);
#endif

}  // namespace cuopt::mathematical_optimization::simplex
