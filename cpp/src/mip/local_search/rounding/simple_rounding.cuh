/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <mip/solution/solution.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
void invoke_round_nearest(solution_t<i_t, f_t>& solution);

template <typename i_t, typename f_t>
bool invoke_simple_rounding(solution_t<i_t, f_t>& solution);

template <typename i_t, typename f_t>
void invoke_random_round_nearest(solution_t<i_t, f_t>& solution, i_t n_target_random_rounds);

template <typename i_t, typename f_t>
void invoke_correct_integers(solution_t<i_t, f_t>& solution, f_t tol);

template <typename i_t, typename f_t>
bool check_brute_force_rounding(solution_t<i_t, f_t>& solution);

}  // namespace cuopt::linear_programming::detail
