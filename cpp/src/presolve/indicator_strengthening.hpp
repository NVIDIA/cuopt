/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#if !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"  // ignore boost error for pip wheel build
#pragma GCC diagnostic ignored "-Wnarrowing"
#endif
#include <papilo/Config.hpp>
#include <papilo/core/Problem.hpp>
#if !defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace cuopt::mathematical_optimization::mip {

// Adds an implied indicator row y <= sum_{g in D} z_g for every implication row
// y <= sum_{j in S} x_j whose members are bounded by indicators x_j <= z_g, and lifts every
// capacity row sum_{i in S} x_i - s <= K whose members share an indicator z into sum_{i in S} x_i -
// s <= K z. Both need an integral indicator, so this is for MIPs only.
template <typename i_t, typename f_t>
void strengthen_indicators(papilo::Problem<f_t>& problem);

}  // namespace cuopt::mathematical_optimization::mip
