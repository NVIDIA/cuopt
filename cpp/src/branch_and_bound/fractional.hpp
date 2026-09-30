/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/user_problem.hpp>

#include <cassert>
#include <cmath>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename f_t>
inline bool is_fractional(f_t x, simplex::variable_type_t var_type, f_t integer_tol)
{
  if (var_type == simplex::variable_type_t::CONTINUOUS) {
    return false;
  } else {
    f_t x_integer = std::round(x);
    return (std::abs(x_integer - x) > integer_tol);
  }
}

template <typename i_t, typename f_t>
inline i_t fractional_variables(const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                                const std::vector<f_t>& x,
                                const std::vector<simplex::variable_type_t>& var_types,
                                std::vector<i_t>& fractional)
{
  const i_t n = x.size();
  assert(x.size() == var_types.size());
  fractional.clear();
  for (i_t j = 0; j < n; ++j) {
    if (is_fractional(x[j], var_types[j], settings.integer_tol)) { fractional.push_back(j); }
  }
  return fractional.size();
}

}  // namespace cuopt::mathematical_optimization::mip
