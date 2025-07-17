/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <raft/util/cuda_utils.cuh>

namespace cuopt::linear_programming::detail {
inline constexpr int block_size = 128;

// When using APIs that handle variable stride sizes these are used to express that we assume that
// the data accessed has a contigous layout in memory for both solutions
// {
inline constexpr int primal_stride = 1;
inline constexpr int dual_stride   = 1;
// }

// #define PDLP_DEBUG_MODE

// Value used to determine what we see as too small (the value) or too large (1/value) values when
// computing the new primal weight during the restart.
template <typename f_t>
inline constexpr f_t safe_guard_for_extreme_values_in_primal_weight_computation = 1.0e-10;
// }

// used to detect divergence in the movement as should trigger a numerical_error
template <typename f_t>
inline constexpr f_t divergent_movement = f_t{};

template <>
inline constexpr float divergent_movement<float> = 1.0e20f;

template <>
inline constexpr double divergent_movement<double> = 1.0e100;

// }

/**
 * as floats
 */
template <>
inline constexpr float safe_guard_for_extreme_values_in_primal_weight_computation<float> = 1.0e-10f;

/**
 * as doubles
 */
template <>
inline constexpr double safe_guard_for_extreme_values_in_primal_weight_computation<double> =
  1.0e-10;

}  // namespace cuopt::linear_programming::detail
