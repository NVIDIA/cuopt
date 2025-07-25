/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include <mip/feasibility_jump/feasibility_jump.cuh>
#include <mip/problem/problem.cuh>
#include <mip/solution/solution.cuh>

#include "Solver.h"

namespace cuopt::linear_programming::detail {

void LocalMipRead(Solver& solver,
                  const problem_t<int32_t, double>& problem,
                  solution_t<int32_t, double>& solution);
void CopyWeights(Solver& solver, fj_t<int32_t, double>& fj);
void GetSolution(Solver& solver, solution_t<int32_t, double>& solution);

}  // namespace cuopt::linear_programming::detail
