/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "internal.hpp"
#include "state.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
inline std::pair<i_t, i_t> model_range_for_var(fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t var_idx)
{
  cuopt_assert(var_idx >= 0 && var_idx < fj_cpu.problem->n_variables,
               "Variable should be within the range");
  return std::make_pair(fj_cpu.problem->reverse_offsets[var_idx],
                        fj_cpu.problem->reverse_offsets[var_idx + 1]);
}

template <typename i_t, typename f_t>
inline std::pair<i_t, i_t> model_range_for_row(fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t cstr_idx)
{
  cuopt_assert(cstr_idx >= 0 && cstr_idx < fj_cpu.problem->n_constraints, "row out of range");
  return std::make_pair(fj_cpu.problem->offsets[cstr_idx], fj_cpu.problem->offsets[cstr_idx + 1]);
}

template <typename i_t, typename f_t>
inline bool check_variable_within_bounds(fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t var_idx, f_t val)
{
  const f_t int_tol  = fj_cpu.problem->tolerances.integrality_tolerance;
  auto bounds        = fj_cpu.h_var_bounds[var_idx].get();
  bool within_bounds = val <= (get_upper(bounds) + int_tol) && val >= (get_lower(bounds) - int_tol);
  return within_bounds;
}

template <typename i_t, typename f_t>
inline bool is_integer_var(fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t var_idx)
{
  return var_t::INTEGER == fj_cpu.problem->h_var_types[var_idx];
}

}  // namespace cuopt::mathematical_optimization::mip
