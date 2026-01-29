/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <raft/util/cuda_utils.cuh>

#include <cuda/cmath>

namespace cuopt::linear_programming::detail {
inline constexpr int block_size = 128;

static std::pair<size_t, size_t> inline kernel_config_from_batch_size(const size_t batch_size)
{
  assert(batch_size > 0 && "Batch size must be greater than 0");
  const size_t block_size = std::min(static_cast<size_t>(256), batch_size);
  const size_t grid_size  = cuda::ceil_div(batch_size, block_size);
  return std::make_pair(grid_size, block_size);
}

// When using APIs that handle variable stride sizes these are used to express that we assume that
// the data accessed has a contigous layout in memory for both solutions
// {
inline constexpr int primal_stride = 1;
inline constexpr int dual_stride   = 1;
// }

// #define PDLP_DEBUG_MODE

// #define CUPDLP_DEBUG_MODE

// #define BATCH_VERBOSE_MODE

inline constexpr bool deterministic_batch_pdlp = true;

inline constexpr bool enable_batch_resizing = true;

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
