/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

 private:
  T* ptr_{nullptr};
  std::size_t size_{0};
};

}  // namespace cuopt::mps_parser
