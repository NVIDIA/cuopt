/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include <cuopt/linear_programming/constants.h>

namespace cuopt::linear_programming {

enum class presolve_method_t {
  NONE            = CUOPT_PRESOLVE_METHOD_NONE,
  FULL            = CUOPT_PRESOLVE_METHOD_FULL,
  DUAL_PRESERVING = CUOPT_PRESOLVE_METHOD_DUAL_PRESERVING
};

}  // namespace cuopt::linear_programming
