/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

/**
 * @file data_model_view.hpp
 * @brief Provides data_model_view_t in the cuopt::linear_programming namespace.
 *
 * This header provides access to the data_model_view_t class, a non-owning view
 * over LP/MIP problem data. The view uses span<T> to hold pointers that can
 * reference either host or device memory, making it suitable for both local
 * GPU-based solves and remote CPU-based solves.
 *
 * The canonical implementation lives in cuopt::mps_parser for historical reasons
 * and to maintain mps_parser as a standalone library. This header provides
 * convenient aliases in the cuopt::linear_programming namespace.
 */

#include <mps_parser/data_model_view.hpp>
#include <mps_parser/utilities/span.hpp>

namespace cuopt::linear_programming {

/**
 * @brief Non-owning span type that can point to either host or device memory.
 *
 * This is an alias to the span type defined in mps_parser. The span holds
 * a pointer and size, but does not own the underlying memory.
 *
 * @tparam T Element type
 */
template <typename T>
using span = cuopt::mps_parser::span<T>;

/**
 * @brief Non-owning view of LP/MIP problem data.
 *
 * This is an alias to the data_model_view_t defined in mps_parser.
 * The view stores problem data (constraint matrix, bounds, objective, etc.)
 * as span<T> members, which can point to either host or device memory.
 *
 * Key features for remote solve support:
 * - Non-owning: does not allocate or free memory
 * - Memory-agnostic: spans can point to host OR device memory
 * - Serializable: host data can be directly serialized for remote solve
 *
 * @tparam i_t Integer type for indices (typically int)
 * @tparam f_t Floating point type for values (typically float or double)
 */
template <typename i_t, typename f_t>
using data_model_view_t = cuopt::mps_parser::data_model_view_t<i_t, f_t>;

}  // namespace cuopt::linear_programming
