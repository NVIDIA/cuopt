/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuts/cuts.hpp>

#include <math_optimization/tic_toc.hpp>
#include <utilities/macros.cuh>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

using simplex::lp_problem_t;
using simplex::simplex_solver_settings_t;
using simplex::variable_type_t;

namespace {

template <typename i_t>
void symmetric_difference_sorted(const std::vector<i_t>& a,
                                 const std::vector<i_t>& b,
                                 std::vector<i_t>& result)
{
  result.clear();
  result.reserve(a.size() + b.size());
  std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
}

template <typename i_t>
struct mod2_parity_row_t {
  std::vector<i_t> parity;
  bool rhs_parity{false};
};

template <typename i_t, typename f_t>
struct mod2_candidate_t : mod2_parity_row_t<i_t> {
  inequality_t<i_t, f_t> transformed_inequality;
  bool reversible{false};
};

template <typename i_t>
struct mod2_basis_row_t {
  std::vector<i_t> parity;
  std::vector<i_t> combination;
  bool rhs{false};
};

template <typename i_t, typename row_t>
struct mod2_row_order_t {
  const std::vector<row_t>& rows;

  bool operator()(i_t a, i_t b) const
  {
    if (rows[a].parity.size() != rows[b].parity.size()) {
      return rows[a].parity.size() < rows[b].parity.size();
    }
    return rows[a].rhs_parity < rows[b].rhs_parity;
  }
};

template <typename i_t, typename f_t, typename row_t>
std::vector<std::vector<i_t>> find_mod2_row_combinations(const std::vector<row_t>& rows,
                                                         i_t max_combination_size,
                                                         i_t max_combinations,
                                                         f_t* work_estimate,
                                                         f_t max_work_estimate,
                                                         f_t start_time = 0.0,
                                                         f_t time_limit = inf)
{
  cuopt_assert(max_combination_size > 0, "Maximum GF(2) combination size must be positive");
  cuopt_assert(max_combinations > 0, "Maximum number of GF(2) combinations must be positive");

  i_t max_index       = -1;
  f_t input_scan_work = 0.0;
  for (const auto& row : rows) {
    input_scan_work += row.parity.size() + 1;
    cuopt_assert(std::is_sorted(row.parity.begin(), row.parity.end()),
                 "GF(2) parity rows must be sorted");
    cuopt_assert(std::adjacent_find(row.parity.begin(), row.parity.end()) == row.parity.end(),
                 "GF(2) parity rows must not contain duplicates");
    if (!row.parity.empty()) {
      cuopt_assert(row.parity.front() >= 0, "GF(2) parity index must be nonnegative");
      max_index = std::max(max_index, row.parity.back());
    }
  }
  if (add_work_estimate(input_scan_work, work_estimate, max_work_estimate)) { return {}; }

  std::vector<i_t> permutation(rows.size());
  std::iota(permutation.begin(), permutation.end(), 0);
  const f_t sort_work = permutation.size() * std::log2(permutation.size() + 1.0);
  if (add_work_estimate(sort_work, work_estimate, max_work_estimate)) { return {}; }
  // this is to process small/sparse rows first, for faster perf and smaller combinations
  std::stable_sort(permutation.begin(), permutation.end(), mod2_row_order_t<i_t, row_t>{rows});

  if (add_work_estimate((f_t)(max_index + 1), work_estimate, max_work_estimate)) { return {}; }
  std::vector<i_t> pivot_to_basis((size_t)(max_index + 1), -1);
  std::vector<mod2_basis_row_t<i_t>> basis;
  basis.reserve(std::min(rows.size(), (size_t)(max_index + 1)));
  std::vector<std::vector<i_t>> combinations;
  combinations.reserve(std::min((size_t)max_combinations, rows.size()));

  std::vector<i_t> parity_tmp;
  std::vector<i_t> combination_tmp;
  for (const i_t candidate : permutation) {
    if (toc(start_time) >= time_limit) { break; }
    f_t candidate_work = rows[candidate].parity.size() + 2;
    mod2_basis_row_t<i_t> current;
    current.parity      = rows[candidate].parity;
    current.combination = {candidate};
    current.rhs         = rows[candidate].rhs_parity;

    bool abandoned = false;
    while (!current.parity.empty()) {
      const i_t pivot       = current.parity.front();
      const i_t basis_index = pivot_to_basis[pivot];
      // pivot has not been seen before
      if (basis_index < 0) { break; }

      const auto& pivot_row = basis[basis_index];
      candidate_work += current.parity.size() + pivot_row.parity.size() +
                        current.combination.size() + pivot_row.combination.size();
      symmetric_difference_sorted(current.parity, pivot_row.parity, parity_tmp);
      symmetric_difference_sorted(current.combination, pivot_row.combination, combination_tmp);
      if (combination_tmp.size() > (size_t)max_combination_size) {
        abandoned = true;
        break;
      }
      current.parity.swap(parity_tmp);
      current.combination.swap(combination_tmp);
      current.rhs = current.rhs != pivot_row.rhs;
    }
    if (add_work_estimate(candidate_work, work_estimate, max_work_estimate)) { break; }
    if (abandoned) { continue; }

    // when reduced, add to combinations and continue, don't add to basis
    if (current.parity.empty()) {
      if (current.rhs && !current.combination.empty()) {
        combinations.push_back(std::move(current.combination));
        if (combinations.size() >= (size_t)max_combinations) { break; }
      }
      continue;
    }

    const i_t pivot       = current.parity.front();
    pivot_to_basis[pivot] = basis.size();
    basis.push_back(std::move(current));
  }
  return combinations;
}

template <typename f_t>
bool value_is_integral(f_t value, f_t tolerance)
{
  return std::abs(value - std::round(value)) <= tolerance;
}

template <typename i_t, typename f_t>
i_t mod2_integral_scale(const inequality_t<i_t, f_t>& inequality,
                        const std::vector<variable_type_t>& var_types,
                        const std::vector<f_t>& transformed_xstar,
                        i_t max_integral_scale,
                        f_t row_tight_tol,
                        f_t coefficient_integral_tol,
                        f_t start_time,
                        f_t time_limit,
                        f_t& work_estimate,
                        f_t max_work_estimate,
                        bool& work_limit_reached)
{
  for (i_t scale = 1; scale <= max_integral_scale; ++scale) {
    if (toc(start_time) >= time_limit || work_limit_reached) { return i_t{0}; }
    f_t scale_work       = 1.0;
    bool integral        = true;
    const f_t scaled_rhs = scale * inequality.rhs;
    if (!value_is_integral(scaled_rhs, coefficient_integral_tol)) { integral = false; }
    for (i_t k = 0; integral && k < (i_t)inequality.size(); ++k) {
      scale_work += 1.0;
      const i_t j = inequality.index(k);
      if (var_types[j] == variable_type_t::CONTINUOUS || transformed_xstar[j] <= row_tight_tol) {
        continue;
      }
      const f_t scaled_coefficient = scale * inequality.coeff(k);
      if (!value_is_integral(scaled_coefficient, coefficient_integral_tol)) { integral = false; }
    }
    if (add_work_estimate(scale_work, &work_estimate, max_work_estimate, &work_limit_reached)) {
      return i_t{0};
    }
    if (integral) { return scale; }
  }
  return i_t{0};
}

template <typename i_t, typename f_t>
std::vector<mod2_candidate_t<i_t, f_t>> mod2_collect_candidates(
  complemented_mixed_integer_rounding_cut_t<i_t, f_t>& complemented_mir,
  const lp_problem_t<i_t, f_t>& lp,
  csr_matrix_t<i_t, f_t>& Arow,
  const variable_bounds_t<i_t, f_t>& variable_bounds,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& transformed_xstar,
  f_t start_time,
  f_t time_limit,
  f_t& work_estimate,
  f_t max_work_estimate,
  bool& work_limit_reached)
{
  constexpr i_t max_integral_scale       = 1000;
  const i_t max_integer_row_length       = 1000 + lp.num_cols / 10;
  constexpr f_t row_tight_tol            = 1e-6;
  constexpr f_t coefficient_integral_tol = 1e-6;

  std::vector<mod2_candidate_t<i_t, f_t>> candidates;
  candidates.reserve(lp.num_rows);
  for (i_t row = 0; row < lp.num_rows; ++row) {
    if (toc(start_time) >= time_limit || work_limit_reached) { break; }
    const i_t slack = complemented_mir.slack_cols(row);
    if (slack < 0 || transformed_xstar[slack] > row_tight_tol) { continue; }

    const i_t row_length = Arow.row_start[row + 1] - Arow.row_start[row];
    if (row_length > max_integer_row_length) { continue; }
    const f_t row_work = (8 * row_length + 5) + row_length * std::log2(row_length + 1.0);
    if (add_work_estimate(row_work, &work_estimate, max_work_estimate, &work_limit_reached)) {
      break;
    }
    inequality_t<i_t, f_t> inequality(Arow, row, lp.rhs[row]);
    complemented_mir.transform_inequality(variable_bounds, var_types, inequality);
    inequality.sort();

    // Every LP row is an equality after slack insertion. Remove a zero-valued transformed slack
    // in the direction that preserves a valid >= inequality.
    i_t slack_position = -1;
    for (i_t k = 0; k < (i_t)inequality.size(); ++k) {
      if (inequality.index(k) == slack) {
        slack_position = k;
        break;
      }
    }
    if (slack_position < 0 || inequality.coeff(slack_position) == 0.0) { continue; }
    // we want a row that is a.x >= b
    if (inequality.coeff(slack_position) > 0.0) { inequality.negate(); }
    inequality.vector.x[slack_position] = 0.0;
    inequality_t<i_t, f_t> squeezed_inequality(lp.num_cols);
    inequality.squeeze(squeezed_inequality);
    inequality = std::move(squeezed_inequality);

    // Continuous variables must be at their selected bounds to participate in the parity system.
    bool continuous_at_bounds = true;
    for (i_t k = 0; k < (i_t)inequality.size(); ++k) {
      const i_t j = inequality.index(k);
      if (var_types[j] == variable_type_t::CONTINUOUS &&
          std::abs(inequality.coeff(k)) > coefficient_integral_tol &&
          transformed_xstar[j] > row_tight_tol) {
        continuous_at_bounds = false;
        break;
      }
    }
    if (!continuous_at_bounds) { continue; }

    const i_t scale = mod2_integral_scale(inequality,
                                          var_types,
                                          transformed_xstar,
                                          max_integral_scale,
                                          row_tight_tol,
                                          coefficient_integral_tol,
                                          start_time,
                                          time_limit,
                                          work_estimate,
                                          max_work_estimate,
                                          work_limit_reached);
    // no integral scale found or time limit reached
    if (scale == 0) { continue; }
    if (scale != 1) { inequality.scale(scale); }

    mod2_candidate_t<i_t, f_t> candidate;
    candidate.transformed_inequality = std::move(inequality);
    candidate.rhs_parity = (std::abs(std::llround(candidate.transformed_inequality.rhs)) % 2) != 0;
    // checks if this could be safely reversed
    candidate.reversible = std::abs(lp.upper[slack] - lp.lower[slack]) <= row_tight_tol;
    for (i_t k = 0; k < (i_t)candidate.transformed_inequality.size(); ++k) {
      const i_t j = candidate.transformed_inequality.index(k);
      if (var_types[j] == variable_type_t::CONTINUOUS || transformed_xstar[j] <= row_tight_tol) {
        continue;
      }
      const auto coefficient = std::llround(candidate.transformed_inequality.coeff(k));
      if ((std::abs(coefficient) % 2) != 0) { candidate.parity.push_back(j); }
    }
    if (candidate.parity.size() > (size_t)max_integer_row_length) { continue; }
    candidates.push_back(std::move(candidate));
  }
  return candidates;
}

template <typename i_t, typename f_t>
void mod2_add_transformed_zero_half_cut(
  complemented_mixed_integer_rounding_cut_t<i_t, f_t>& complemented_mir,
  cut_pool_t<i_t, f_t>& cut_pool,
  const lp_problem_t<i_t, f_t>& lp,
  csr_matrix_t<i_t, f_t>& Arow,
  const variable_bounds_t<i_t, f_t>& variable_bounds,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  inequality_t<i_t, f_t> transformed_cut,
  f_t min_violation,
  f_t& work_estimate,
  i_t& cuts_added)
{
  work_estimate += 4 * transformed_cut.size() + 1;
  complemented_mir.untransform_inequality(variable_bounds, var_types, transformed_cut);
  complemented_mir.remove_small_coefficients(lp.lower, lp.upper, transformed_cut);
  complemented_mir.substitute_slacks(lp, Arow, transformed_cut, &work_estimate);
  complemented_mir.remove_small_coefficients(lp.lower, lp.upper, transformed_cut);
  const f_t violation = complemented_mir.compute_violation(transformed_cut, xstar);
  if (violation > min_violation) {
    const i_t pool_size = cut_pool.pool_size();
    cut_pool.add_cut(cut_type_t::ZERO_HALF, transformed_cut);
    if (cut_pool.pool_size() > pool_size) { ++cuts_added; }
  }
}

template <typename i_t, typename f_t>
void mod2_generate_cuts_from_aggregate(
  complemented_mixed_integer_rounding_cut_t<i_t, f_t>& complemented_mir,
  cut_pool_t<i_t, f_t>& cut_pool,
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const variable_bounds_t<i_t, f_t>& variable_bounds,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  const std::vector<f_t>& transformed_xstar,
  const inequality_t<i_t, f_t>& oriented_aggregate,
  f_t min_violation,
  f_t start_time,
  f_t& work_estimate,
  f_t max_work_estimate,
  bool& work_limit_reached,
  i_t& cuts_added)
{
  if (add_work_estimate((f_t)(3 * oriented_aggregate.size() + 1),
                        &work_estimate,
                        max_work_estimate,
                        &work_limit_reached)) {
    return;
  }
  inequality_t<i_t, f_t> mir_cut(lp.num_cols);
  const bool mir_cut_generated = complemented_mir.generate_cut_nonnegative_maintain_indicies(
    oriented_aggregate, var_types, mir_cut);
  if (mir_cut_generated) {
    mod2_add_transformed_zero_half_cut(complemented_mir,
                                       cut_pool,
                                       lp,
                                       Arow,
                                       variable_bounds,
                                       var_types,
                                       xstar,
                                       std::move(mir_cut),
                                       min_violation,
                                       work_estimate,
                                       cuts_added);
  }

  if (work_estimate > max_work_estimate) {
    work_limit_reached = true;
    return;
  }
  inequality_t<i_t, f_t> lifted_cover_cut(lp.num_cols);
  bool lifted_cover_cut_generated = false;
  if (toc(start_time) < settings.time_limit) {
    lifted_cover_cut_generated =
      complemented_mir.generate_lifted_mixed_binary_cover(oriented_aggregate,
                                                          var_types,
                                                          transformed_xstar,
                                                          lifted_cover_cut,
                                                          work_estimate,
                                                          max_work_estimate);
  }
  if (lifted_cover_cut_generated) {
    mod2_add_transformed_zero_half_cut(complemented_mir,
                                       cut_pool,
                                       lp,
                                       Arow,
                                       variable_bounds,
                                       var_types,
                                       xstar,
                                       std::move(lifted_cover_cut),
                                       min_violation,
                                       work_estimate,
                                       cuts_added);
  }
  if (work_estimate > max_work_estimate) { work_limit_reached = true; }
}

template <typename i_t, typename f_t>
struct lifted_cover_order_t {
  const std::vector<f_t>& solution_value;
  const inequality_t<i_t, f_t>& base;
  f_t tolerance;

  bool operator()(int a, int b) const
  {
    const bool a_at_upper = solution_value[a] >= 1.0 - tolerance;
    const bool b_at_upper = solution_value[b] >= 1.0 - tolerance;
    if (a_at_upper != b_at_upper) { return a_at_upper; }
    const f_t contribution_a = solution_value[a] * base.coeff(a);
    const f_t contribution_b = solution_value[b] * base.coeff(b);
    if (contribution_a != contribution_b) { return contribution_a > contribution_b; }
    return base.coeff(a) > base.coeff(b);
  }
};

template <typename f_t>
f_t lifted_cover_coefficient(
  f_t coefficient, const std::vector<f_t>& prefix, size_t p, f_t lambda, f_t tolerance)
{
  for (size_t h = 0; h < p; ++h) {
    if (coefficient <= prefix[h] - lambda + tolerance) { return h * lambda; }
    if (coefficient <= prefix[h] + tolerance) { return (h + 1) * lambda + coefficient - prefix[h]; }
  }
  return p * lambda + coefficient - prefix[p - 1];
}

}  // namespace

std::vector<std::vector<int>> find_mod2_row_combinations_for_test(
  const std::vector<std::vector<int>>& parity_rows,
  const std::vector<char>& rhs_parity,
  int max_combination_size,
  int max_combinations)
{
  return find_mod2_row_combinations_for_test(parity_rows,
                                             rhs_parity,
                                             max_combination_size,
                                             max_combinations,
                                             std::numeric_limits<double>::infinity(),
                                             nullptr);
}

std::vector<std::vector<int>> find_mod2_row_combinations_for_test(
  const std::vector<std::vector<int>>& parity_rows,
  const std::vector<char>& rhs_parity,
  int max_combination_size,
  int max_combinations,
  double max_work_estimate,
  double* work_estimate_out)
{
  cuopt_assert(parity_rows.size() == rhs_parity.size(),
               "GF(2) parity row and rhs sizes must match");
  std::vector<mod2_parity_row_t<int>> rows;
  rows.reserve(parity_rows.size());
  for (size_t i = 0; i < parity_rows.size(); ++i) {
    rows.push_back({parity_rows[i], rhs_parity[i] != 0});
  }

  double work_estimate = 0.0;
  auto combinations    = find_mod2_row_combinations<int, double>(
    rows, max_combination_size, max_combinations, &work_estimate, max_work_estimate);
  if (work_estimate_out != nullptr) { *work_estimate_out = work_estimate; }
  return combinations;
}

template <typename i_t, typename f_t>
bool generate_mod2_zero_half_cuts(cut_pool_t<i_t, f_t>& cut_pool,
                                  const lp_problem_t<i_t, f_t>& lp,
                                  const simplex_solver_settings_t<i_t, f_t>& settings,
                                  csr_matrix_t<i_t, f_t>& Arow,
                                  const std::vector<i_t>& new_slacks,
                                  const std::vector<variable_type_t>& var_types,
                                  const std::vector<f_t>& xstar,
                                  variable_bounds_t<i_t, f_t>& variable_bounds,
                                  f_t start_time,
                                  f_t& work_estimate)
{
  constexpr i_t max_combination_size   = 64;
  constexpr i_t max_row_combinations   = 1000;
  constexpr f_t min_violation          = 1e-6;
  constexpr f_t candidate_work_limit   = 3e7;
  constexpr f_t combination_work_limit = 3e7;
  constexpr f_t generation_work_limit  = 4e7;
  f_t candidate_work                   = 0.0;
  f_t combination_work                 = 0.0;
  f_t generation_work                  = 0.0;
  bool candidate_limit_reached         = false;
  bool generation_limit_reached        = false;

  if (add_work_estimate((f_t)(3 * lp.num_cols) + (f_t)(variable_bounds.upper_variables.size() +
                                                       variable_bounds.lower_variables.size()),
                        &candidate_work,
                        candidate_work_limit,
                        &candidate_limit_reached)) {
    work_estimate = candidate_work;
    return false;
  }
  complemented_mixed_integer_rounding_cut_t<i_t, f_t> complemented_mir(lp, settings, new_slacks);
  std::vector<f_t> transformed_xstar;
  complemented_mir.bound_substitution(
    lp, variable_bounds, var_types, xstar, transformed_xstar, true);

  auto candidates = mod2_collect_candidates(complemented_mir,
                                            lp,
                                            Arow,
                                            variable_bounds,
                                            var_types,
                                            transformed_xstar,
                                            start_time,
                                            settings.time_limit,
                                            candidate_work,
                                            candidate_work_limit,
                                            candidate_limit_reached);

  if (toc(start_time) >= settings.time_limit) {
    work_estimate = candidate_work;
    return true;
  }
  auto row_combinations = find_mod2_row_combinations<i_t, f_t>(candidates,
                                                               max_combination_size,
                                                               max_row_combinations,
                                                               &combination_work,
                                                               combination_work_limit,
                                                               start_time,
                                                               settings.time_limit);
  if (add_work_estimate((f_t)(2 * lp.num_cols),
                        &generation_work,
                        generation_work_limit,
                        &generation_limit_reached)) {
    work_estimate = candidate_work + combination_work + generation_work;
    return true;
  }
  scratch_pad_t<i_t, f_t> aggregate_pad(lp.num_cols);

  for (const auto& combination : row_combinations) {
    if (toc(start_time) >= settings.time_limit || generation_limit_reached) { break; }

    size_t aggregate_input_nz = 0;
    for (const i_t candidate_index : combination) {
      aggregate_input_nz += candidates[candidate_index].transformed_inequality.size();
    }
    if (add_work_estimate((f_t)(2 * aggregate_input_nz + 1),
                          &generation_work,
                          generation_work_limit,
                          &generation_limit_reached)) {
      break;
    }

    inequality_t<i_t, f_t> aggregate(lp.num_cols);
    bool reversible = true;
    for (const i_t candidate_index : combination) {
      const auto& candidate = candidates[candidate_index];
      aggregate.rhs += candidate.transformed_inequality.rhs;
      reversible = reversible && candidate.reversible;
      for (i_t k = 0; k < (i_t)candidate.transformed_inequality.size(); ++k) {
        aggregate_pad.add_to_pad(candidate.transformed_inequality.index(k),
                                 candidate.transformed_inequality.coeff(k));
      }
    }
    aggregate_pad.get_pad(aggregate.vector.i, aggregate.vector.x);
    aggregate_pad.clear_pad();
    const f_t aggregate_output_work =
      3 * aggregate.size() + aggregate.size() * std::log2(aggregate.size() + 1.0);
    if (add_work_estimate(aggregate_output_work,
                          &generation_work,
                          generation_work_limit,
                          &generation_limit_reached)) {
      break;
    }
    aggregate.sort();
    aggregate.scale(0.5);

    i_t cuts_added = 0;
    mod2_generate_cuts_from_aggregate(complemented_mir,
                                      cut_pool,
                                      lp,
                                      settings,
                                      Arow,
                                      variable_bounds,
                                      var_types,
                                      xstar,
                                      transformed_xstar,
                                      aggregate,
                                      min_violation,
                                      start_time,
                                      generation_work,
                                      generation_work_limit,
                                      generation_limit_reached,
                                      cuts_added);
    // if the final inequality is reversable, try the reversed version as well
    if (reversible && toc(start_time) < settings.time_limit && !generation_limit_reached) {
      aggregate.negate();
      mod2_generate_cuts_from_aggregate(complemented_mir,
                                        cut_pool,
                                        lp,
                                        settings,
                                        Arow,
                                        variable_bounds,
                                        var_types,
                                        xstar,
                                        transformed_xstar,
                                        aggregate,
                                        min_violation,
                                        start_time,
                                        generation_work,
                                        generation_work_limit,
                                        generation_limit_reached,
                                        cuts_added);
    }
  }
  work_estimate = candidate_work + combination_work + generation_work;
  return true;
}

template <typename i_t, typename f_t>
bool complemented_mixed_integer_rounding_cut_t<i_t, f_t>::generate_lifted_mixed_binary_cover(
  const inequality_t<i_t, f_t>& transformed_inequality,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& transformed_xstar,
  inequality_t<i_t, f_t>& transformed_cut,
  f_t& work_estimate,
  f_t max_work_estimate)
{
  constexpr f_t tolerance = 1e-6;

  const f_t estimated_work =
    12 * transformed_inequality.size() +
    transformed_inequality.size() * std::log2(transformed_inequality.size() + 1.0);
  if (add_work_estimate(estimated_work, &work_estimate, max_work_estimate)) { return false; }

  inequality_t<i_t, f_t> base = transformed_inequality;
  base.negate();

  std::vector<char> locally_complemented(base.size(), 0);
  std::vector<f_t> solution_value(base.size(), 0.0);
  std::vector<char> is_integral(base.size(), 0);
  for (i_t k = 0; k < (i_t)base.size(); ++k) {
    const i_t j = base.index(k);
    f_t aj      = base.coeff(k);
    if (var_types[j] == variable_type_t::CONTINUOUS) {
      solution_value[k] = transformed_xstar[j];
      if (aj > 0.0) { base.vector.x[k] = 0.0; }
      continue;
    }

    const f_t upper = new_upper(j);
    if (upper == inf || std::abs(upper - 1.0) > tolerance) { return false; }
    is_integral[k] = 1;
    if (aj < 0.0) {
      base.rhs -= aj * upper;
      base.vector.x[k]        = -aj;
      solution_value[k]       = upper - transformed_xstar[j];
      locally_complemented[k] = 1;
    } else {
      solution_value[k] = transformed_xstar[j];
    }
  }

  std::vector<i_t> cover;
  cover.reserve(base.size());
  for (i_t k = 0; k < (i_t)base.size(); ++k) {
    if (is_integral[k] && base.coeff(k) > tolerance && solution_value[k] > tolerance) {
      cover.push_back(k);
    }
  }
  if (cover.empty()) { return false; }

  std::stable_sort(
    cover.begin(), cover.end(), lifted_cover_order_t<i_t, f_t>{solution_value, base, tolerance});

  f_t cover_weight  = 0.0;
  size_t cover_size = 0;
  for (; cover_size < cover.size(); ++cover_size) {
    cover_weight += base.coeff(cover[cover_size]);
    if (cover_weight - base.rhs > tolerance * std::max((f_t)1.0, std::abs(base.rhs))) {
      ++cover_size;
      break;
    }
  }
  if (cover_size == 0 || cover_size > cover.size()) { return false; }
  cover.resize(cover_size);

  const f_t lambda = cover_weight - base.rhs;
  if (lambda <= tolerance) { return false; }
  std::sort(
    cover.begin(), cover.end(), [&](i_t a, i_t b) { return base.coeff(a) > base.coeff(b); });

  std::vector<f_t> prefix(cover.size(), 0.0);
  std::vector<char> in_cover(base.size(), 0);
  f_t prefix_sum = 0.0;
  size_t p       = cover.size();
  for (size_t h = 0; h < cover.size(); ++h) {
    const i_t k = cover[h];
    in_cover[k] = 1;
    if (base.coeff(k) - lambda <= tolerance && p == cover.size()) { p = h; }
    if (h < p) {
      prefix_sum += base.coeff(k);
      prefix[h] = prefix_sum;
    }
  }
  if (p == 0) { return false; }

  size_t non_cover_count = 0;
  for (i_t k = 0; k < (i_t)base.size(); ++k) {
    if (is_integral[k] && !in_cover[k]) { non_cover_count++; }
  }
  if (add_work_estimate((f_t)(non_cover_count * p), &work_estimate, max_work_estimate)) {
    return false;
  }

  transformed_cut     = base;
  transformed_cut.rhs = -lambda;
  for (i_t k = 0; k < (i_t)base.size(); ++k) {
    if (!is_integral[k]) {
      if (base.coeff(k) >= 0.0) { transformed_cut.vector.x[k] = 0.0; }
      continue;
    }
    if (in_cover[k]) {
      transformed_cut.vector.x[k] = std::min(base.coeff(k), lambda);
      transformed_cut.rhs += transformed_cut.coeff(k);
    } else {
      transformed_cut.vector.x[k] =
        lifted_cover_coefficient(base.coeff(k), prefix, p, lambda, tolerance);
    }
  }

  for (i_t k = 0; k < (i_t)transformed_cut.size(); ++k) {
    if (!locally_complemented[k]) { continue; }
    const i_t j           = transformed_cut.index(k);
    const f_t coefficient = transformed_cut.coeff(k);
    transformed_cut.rhs -= coefficient * new_upper(j);
    transformed_cut.vector.x[k] = -coefficient;
  }
  inequality_t<i_t, f_t> squeezed_cut(transformed_cut.vector.n);
  transformed_cut.squeeze(squeezed_cut);
  transformed_cut = std::move(squeezed_cut);
  transformed_cut.negate();
  return true;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template bool generate_mod2_zero_half_cuts<int, double>(
  cut_pool_t<int, double>& cut_pool,
  const lp_problem_t<int, double>& lp,
  const simplex_solver_settings_t<int, double>& settings,
  csr_matrix_t<int, double>& Arow,
  const std::vector<int>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  const std::vector<double>& xstar,
  variable_bounds_t<int, double>& variable_bounds,
  double start_time,
  double& work_estimate);

template bool
complemented_mixed_integer_rounding_cut_t<int, double>::generate_lifted_mixed_binary_cover(
  const inequality_t<int, double>& transformed_inequality,
  const std::vector<variable_type_t>& var_types,
  const std::vector<double>& transformed_xstar,
  inequality_t<int, double>& transformed_cut,
  double& work_estimate,
  double max_work_estimate);
#endif

}  // namespace cuopt::mathematical_optimization::mip
