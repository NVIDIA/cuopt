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
