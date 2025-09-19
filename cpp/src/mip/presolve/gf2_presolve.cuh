/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <mip/problem/problem.cuh>
#include <mip/solution/solution.cuh>
#include <mip/solver.cuh>
#include <mip/utils.cuh>

#include "utils.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
void gf2_presolve(problem_t<i_t, f_t>& problem);

}  // namespace cuopt::linear_programming::detail
