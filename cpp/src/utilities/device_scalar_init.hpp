/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <limits>

namespace cuopt {

// inline constants to use as constructor arguments for rmm::device_scalar
// since the rvalue constructor is deleted

template <typename T>
inline constexpr T zero_v{};
template <typename T>
inline constexpr T one_v = T(1);
template <typename T>
inline constexpr T neg_one_v = T(-1);
template <typename T>
inline constexpr T inf_v = std::numeric_limits<T>::infinity();
template <typename T>
inline constexpr T neg_inf_v = -std::numeric_limits<T>::infinity();
template <typename T>
inline constexpr T max_v = std::numeric_limits<T>::max();
template <typename T>
inline constexpr T min_v = std::numeric_limits<T>::min();
template <typename T>
inline constexpr T lowest_v = std::numeric_limits<T>::lowest();

inline constexpr bool true_v  = true;
inline constexpr bool false_v = false;

}  // namespace cuopt
