/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utility>

namespace cuopt {

template <typename Func>
class scope_guard {
 public:
  explicit scope_guard(Func cleanup) : cleanup_(std::move(cleanup)) {}

  ~scope_guard() { cleanup_(); }

  scope_guard(const scope_guard&)            = delete;
  scope_guard& operator=(const scope_guard&) = delete;
  scope_guard(scope_guard&&)                 = delete;
  scope_guard& operator=(scope_guard&&)      = delete;

 private:
  Func cleanup_;
};

}  // namespace cuopt
