/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
namespace cuopt {

/**
 * @brief strided_span container provides a view that can only be accessed in a strided manner
 * For example, if we want to just access one row of a given matrix that is stored in a column
 * major format, we would have to always calculate the offset to access each entry. This container
 * abstracts out that by storing the stride and allows contiguous easy to use indexing
 */
template <typename T>
class strided_span {
 public:
  strided_span() = default;
  HDI strided_span(T* ptr, size_t stride, size_t size) : ptr_(ptr), stride_(stride), size_(size) {}

  bool operator==(strided_span<T> const& rhs) const
  {
    for (size_t i = 0; i < size_; ++i) {
      if ((*this)[i] != rhs[i]) { return false; }
    }
    return true;
  }

  HDI const T& operator[](size_t i) const
  {
    assert(i < size_);
    return ptr_[i * stride_];
  }

  HDI T& operator[](size_t i)
  {
    assert(i < size_);
    return ptr_[i * stride_];
  }

  HDI size_t size() const { return size_; }
  HDI size_t stride() const { return stride_; }
  HDI size_t empty() const { return size_ == 0; }

 private:
  T* ptr_        = nullptr;
  size_t stride_ = 1;
  size_t size_   = 0;
};
}  // namespace cuopt
