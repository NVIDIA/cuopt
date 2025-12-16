/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/dense_vector.hpp>

namespace cuopt::linear_programming::dual_simplex {

// Custom allocator to build pinned memory vector
template <typename T>
struct PinnedHostAllocator {
  using value_type = T;

  PinnedHostAllocator() noexcept;
  template <class U>
  PinnedHostAllocator(const PinnedHostAllocator<U>&) noexcept;

  T* allocate(std::size_t n);
  void deallocate(T* p, std::size_t);
};

template <typename T, typename U>
bool operator==(const PinnedHostAllocator<T>&, const PinnedHostAllocator<U>&) noexcept;
template <typename T, typename U>
bool operator!=(const PinnedHostAllocator<T>&, const PinnedHostAllocator<U>&) noexcept;

template <typename i_t, typename f_t>
using pinned_dense_vector_t = dense_vector_t<i_t, f_t, PinnedHostAllocator<f_t>>;

}  // namespace cuopt::linear_programming::dual_simplex
