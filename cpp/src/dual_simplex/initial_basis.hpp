/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <linear_algebra/sparse_matrix.hpp>
#include <math_optimization/types.hpp>

namespace cuopt::mathematical_optimization::simplex {

enum class variable_status_t : int8_t {
  BASIC          = 0,
  NONBASIC_LOWER = -1,
  NONBASIC_UPPER = 1,
  NONBASIC_FREE  = 2,
  NONBASIC_FIXED = 3,
  SUPERBASIC     = 4
};

std::vector<uint8_t> compress_vstatus(const std::vector<variable_status_t>& vstatus);
void decompress_vstatus(const std::vector<uint8_t>& packed_vstatus,
                        size_t vstatus_size,
                        std::vector<variable_status_t>& vstatus);

template <typename i_t, typename f_t>
i_t initial_basis_selection(const lp_problem_t<i_t, f_t>& problem,
                            const simplex_solver_settings_t<i_t, f_t>& settings,
                            const std::vector<i_t>& candidate_columns,
                            f_t start_time,
                            std::vector<variable_status_t>& vstatus,
                            std::vector<i_t>& dependent_rows);

}  // namespace cuopt::mathematical_optimization::simplex
