/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuts/cuts.hpp>
#include <math_optimization/tic_toc.hpp>

#include <algorithm>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

using simplex::lp_problem_t;
using simplex::simplex_solver_settings_t;
using simplex::variable_type_t;

template <typename i_t, typename f_t>
void cut_generation_t<i_t, f_t>::generate_implied_bound_cuts(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  f_t start_time)
{
  if (probing_implied_bound_.zero_offsets.empty()) { return; }

  const f_t tol                    = 1e-4;
  i_t num_cuts                     = 0;
  const i_t pib_cols               = probing_implied_bound_.zero_offsets.size() - 1;
  const i_t n_cols                 = std::min(lp.num_cols, pib_cols);
  f_t work_estimate                = 0.0;
  const f_t max_work_estimate      = 1e8;
  constexpr f_t implication_work   = 16.0;
  constexpr f_t generated_cut_work = 16.0;

  for (i_t j = 0; j < n_cols; j++) {
    if (work_estimate > max_work_estimate || toc(start_time) >= settings.time_limit) { return; }
    if (var_types[j] == variable_type_t::CONTINUOUS) { continue; }
    const f_t xstar_j = xstar[j];

    // x_j = 0 implications
    const i_t zero_begin = probing_implied_bound_.zero_offsets[j];
    const i_t zero_end   = probing_implied_bound_.zero_offsets[j + 1];
    const i_t one_begin  = probing_implied_bound_.one_offsets[j];
    const i_t one_end    = probing_implied_bound_.one_offsets[j + 1];
    for (i_t p = zero_begin; p < zero_end; p++) {
      work_estimate += implication_work;
      const i_t i = probing_implied_bound_.zero_variables[p];
      if (i == j) { continue; }
      const f_t l_i = lp.lower[i];
      const f_t u_i = lp.upper[i];

      // Tightened upper bound: x_j = 0 implies y_i <= b, where b < u_i
      // Valid inequality: y_i <= b + (u_i - b)*x_j  or  -y_i + (u_i - b)*x_j >= -b
      const f_t b_ub = probing_implied_bound_.zero_upper_bound[p];
      if (b_ub < u_i - tol) {
        const f_t coeff_j = u_i - b_ub;
        const f_t y_i     = xstar[i];
        const f_t lhs_val = -y_i + coeff_j * xstar_j;
        const f_t rhs_val = -b_ub;
        if (lhs_val < rhs_val - tol) {
          inequality_t<i_t, f_t> cut;
          cut.push_back(i, -1.0);
          cut.push_back(j, coeff_j);
          cut.rhs = -b_ub;
          cut_pool_.add_cut(cut_type_t::IMPLIED_BOUND, cut);
          work_estimate += generated_cut_work;
          num_cuts++;
        }
      }

      // Tightened lower bound: x_j = 0 implies y_i >= b, where b > l_i
      // Valid inequality: y_i >= b - (b - l_i)*x_j  or  y_i + (b - l_i)*x_j >= b
      const f_t b_lb = probing_implied_bound_.zero_lower_bound[p];
      if (b_lb > l_i + tol) {
        const f_t coeff_j = b_lb - l_i;
        const f_t y_i     = xstar[i];
        const f_t lhs_val = y_i + coeff_j * xstar_j;
        const f_t rhs_val = b_lb;
        if (lhs_val < rhs_val - tol) {
          inequality_t<i_t, f_t> cut;
          cut.push_back(i, 1.0);
          cut.push_back(j, coeff_j);
          cut.rhs = b_lb;
          cut_pool_.add_cut(cut_type_t::IMPLIED_BOUND, cut);
          work_estimate += generated_cut_work;
          num_cuts++;
        }
      }
    }
    if (work_estimate > max_work_estimate || toc(start_time) >= settings.time_limit) { return; }

    // x_j = 1 implications
    for (i_t p = one_begin; p < one_end; p++) {
      work_estimate += implication_work;
      const i_t i = probing_implied_bound_.one_variables[p];
      if (i == j) { continue; }
      const f_t l_i = lp.lower[i];
      const f_t u_i = lp.upper[i];

      // Tightened upper bound: x_j = 1 implies y_i <= b, where b < u_i
      // Valid inequality: y_i <= u_i - (u_i - b)*x_j  or  -y_i - (u_i - b)*x_j >= -u_i
      const f_t b_ub = probing_implied_bound_.one_upper_bound[p];
      if (b_ub < u_i - tol) {
        const f_t coeff_j = -(u_i - b_ub);
        const f_t y_i     = xstar[i];
        const f_t lhs_val = -y_i + coeff_j * xstar_j;
        const f_t rhs_val = -u_i;
        if (lhs_val < rhs_val - tol) {
          inequality_t<i_t, f_t> cut;
          cut.push_back(i, -1.0);
          cut.push_back(j, coeff_j);
          cut.rhs = -u_i;
          cut_pool_.add_cut(cut_type_t::IMPLIED_BOUND, cut);
          work_estimate += generated_cut_work;
          num_cuts++;
        }
      }

      // Tightened lower bound: x_j = 1 implies y_i >= b, where b > l_i
      // Valid inequality: y_i >= l_i + (b - l_i)*x_j  or  y_i - (b - l_i)*x_j >= l_i
      const f_t b_lb = probing_implied_bound_.one_lower_bound[p];
      if (b_lb > l_i + tol) {
        const f_t coeff_j = -(b_lb - l_i);
        const f_t lhs_val = xstar[i] + coeff_j * xstar_j;
        const f_t rhs_val = l_i;
        if (lhs_val < rhs_val - tol) {
          inequality_t<i_t, f_t> cut;
          cut.push_back(i, 1.0);
          cut.push_back(j, coeff_j);
          cut.rhs = rhs_val;
          cut_pool_.add_cut(cut_type_t::IMPLIED_BOUND, cut);
          work_estimate += generated_cut_work;
          num_cuts++;
        }
      }
    }
    if (work_estimate > max_work_estimate || toc(start_time) >= settings.time_limit) { return; }
  }

  if (num_cuts > 0) {
    settings.log.debug("Generated %d implied bounds cuts from probing\n", num_cuts);
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template void cut_generation_t<int, double>::generate_implied_bound_cuts(
  const lp_problem_t<int, double>& lp,
  const simplex_solver_settings_t<int, double>& settings,
  const std::vector<variable_type_t>& var_types,
  const std::vector<double>& xstar,
  double start_time);
#endif

}  // namespace cuopt::mathematical_optimization::mip
