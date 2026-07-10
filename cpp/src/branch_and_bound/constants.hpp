/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

namespace cuopt::mathematical_optimization::mip {

constexpr int num_search_strategies = 7;

// Indicate the search and variable selection algorithms used by each thread
// in B&B (See [1]).
//
// [1] T. Achterberg, “Constraint Integer Programming,” PhD, Technischen Universität Berlin,
// Berlin, 2007. doi: 10.14279/depositonce-1634.
// [2] J. Witzig and A. Gleixner, “Conflict-Driven Heuristics for Mixed Integer Programming,”
// Feb. 07, 2019, _arXiv_: arXiv:1902.02615. doi:
// [10.48550/arXiv.1902.02615](https://doi.org/10.48550/arXiv.1902.02615).
enum search_strategy_t : int {
  BEST_FIRST           = 0,  // Best-First + Plunging.
  PSEUDOCOST_DIVING    = 1,  // Pseudocost diving (9.2.5)
  LINE_SEARCH_DIVING   = 2,  // Line search diving (9.2.4)
  GUIDED_DIVING        = 3,  // Guided diving (9.2.3).
  COEFFICIENT_DIVING   = 4,  // Coefficient diving (9.2.1)
  FARKAS_DIVING        = 5,  // Farkas Diving (see [2])
  VECTOR_LENGTH_DIVING = 6   // Vector Length Diving (9.2.6)
};

constexpr search_strategy_t search_strategies[] = {BEST_FIRST,
                                                   PSEUDOCOST_DIVING,
                                                   LINE_SEARCH_DIVING,
                                                   GUIDED_DIVING,
                                                   COEFFICIENT_DIVING,
                                                   FARKAS_DIVING,
                                                   VECTOR_LENGTH_DIVING};

enum class branch_direction_t { NONE = -1, DOWN = 0, UP = 1 };

enum class branch_and_bound_mode_t { PARALLEL = 0, DETERMINISTIC = 1 };

}  // namespace cuopt::mathematical_optimization::mip
