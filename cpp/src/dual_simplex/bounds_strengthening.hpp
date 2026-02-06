/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
class bounds_strengthening_t {
 public:
  // For pure LP bounds strengthening, var_types should be defaulted (i.e. left empty)
  bounds_strengthening_t(const lp_problem_t<i_t, f_t>& problem,
                         const csr_matrix_t<i_t, f_t>& Arow,
                         const std::vector<char>& row_sense,
                         const std::vector<variable_type_t>& var_types);

  // If bounds_changed is empty, all constraints are scanned for changes.
  // Otherwise, bounds_changed must be a vector of length n, where n is the number of variables.
  bool bounds_strengthening(const simplex_solver_settings_t<i_t, f_t>& settings,
                            const std::vector<bool>& bounds_changed,
                            std::vector<f_t>& lower_bounds,
                            std::vector<f_t>& upper_bounds);

 private:
  const csc_matrix_t<i_t, f_t>& A;
  const csr_matrix_t<i_t, f_t>& Arow;
  const std::vector<variable_type_t>& var_types;

  std::vector<f_t> lower;
  std::vector<f_t> upper;

  std::vector<f_t> delta_min_activity;
  std::vector<f_t> delta_max_activity;
  std::vector<f_t> constraint_lb;
  std::vector<f_t> constraint_ub;
};
}  // namespace cuopt::linear_programming::dual_simplex
