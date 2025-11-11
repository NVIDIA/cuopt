/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

  bool bounds_strengthening(std::vector<f_t>& lower_bounds,
                            std::vector<f_t>& upper_bounds,
                            const simplex_solver_settings_t<i_t, f_t>& settings);

  std::vector<bool> bounds_changed;

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
