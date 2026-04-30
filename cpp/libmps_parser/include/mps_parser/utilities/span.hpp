/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstddef>

namespace cuopt::mps_parser {

template <typename T>
class span {
 public:
  span() = default;
  span(T* ptr, std::size_t size) : ptr_(ptr), size_(size) {}
  std::size_t size() const noexcept { return size_; }
  const T* data() const noexcept { return ptr_; }
  T& operator[](std::size_t i) noexcept { return ptr_[i]; }
  T const& operator[](std::size_t i) const noexcept { return ptr_[i]; }

 private:
  T* ptr_{nullptr};
  std::size_t size_{0};
};

}  // namespace cuopt::mps_parser
