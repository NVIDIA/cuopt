/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuts/cuts.hpp>
#include <cuts/greedy_knapsack.hpp>
#include <math_optimization/tic_toc.hpp>
#include <utilities/macros.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

using simplex::lp_problem_t;
using simplex::simplex_solver_settings_t;
using simplex::variable_type_t;

enum class flow_cover_bound_side_t : int8_t { UPPER, LOWER };

template <typename i_t, typename f_t>
f_t flow_cover_scaled_primal_tol(const flow_cover_context_t<i_t, f_t>& context, f_t scale)
{
  return context.settings.primal_tol * std::max<f_t>(1.0, std::abs(scale));
}

template <typename i_t, typename f_t>
bool flow_cover_is_zero_one_integer_variable(const flow_cover_context_t<i_t, f_t>& context, i_t j)
{
  const f_t bound_tol = context.settings.primal_tol;
  return context.var_types[j] == variable_type_t::INTEGER &&
         std::abs(context.lp.lower[j]) <= bound_tol &&
         std::abs(context.lp.upper[j] - 1.0) <= bound_tol;
}

template <typename f_t>
static bool flow_cover_valid_endpoint_terms(f_t endpoint_term,
                                            f_t binary_coefficient,
                                            bool in_n2,
                                            f_t bound_tol)
{
  return in_n2 ? endpoint_term <= bound_tol && endpoint_term + binary_coefficient <= bound_tol
               : endpoint_term >= -bound_tol && endpoint_term + binary_coefficient >= -bound_tol;
}

// Per-arc feasibility tolerances shared by the arc-acceptance gate and the assertion in
// build_single_node_flow_relaxation, so the two sites cannot drift apart.
template <typename i_t, typename f_t>
f_t flow_cover_arc_lower_tol(const flow_cover_context_t<i_t, f_t>& context)
{
  return std::max<f_t>(10 * context.settings.primal_tol, static_cast<f_t>(1e-5));
}

template <typename i_t, typename f_t>
f_t flow_cover_arc_upper_tol(const flow_cover_context_t<i_t, f_t>& context, f_t u)
{
  return std::max<f_t>(100 * context.settings.primal_tol, static_cast<f_t>(1e-4)) *
         std::max<f_t>(1.0, u);
}

template <typename i_t, typename f_t>
single_node_flow_arc_t<i_t, f_t> flow_cover_build_arc(const flow_cover_context_t<i_t, f_t>& context,
                                                      f_t u,
                                                      bool in_n2,
                                                      i_t x_col,
                                                      f_t fixed_x,
                                                      f_t y_const,
                                                      i_t y_col,
                                                      f_t y_coeff,
                                                      f_t y_x_coeff)
{
  auto clamp01      = [](f_t x) { return std::min<f_t>(1.0, std::max<f_t>(0.0, x)); };
  const f_t x_value = x_col >= 0 ? clamp01(context.xstar[x_col]) : fixed_x;
  f_t y_value       = y_const;
  if (y_col >= 0) { y_value += y_coeff * context.xstar[y_col]; }
  if (x_col >= 0) {
    y_value += y_x_coeff * context.xstar[x_col];
  } else {
    y_value += y_x_coeff * fixed_x;
  }
  return single_node_flow_arc_t<i_t, f_t>{
    u, in_n2, x_col, x_value, y_const, y_col, y_coeff, y_x_coeff, y_value};
}

template <typename i_t, typename f_t>
void flow_cover_try_add_candidate(const flow_cover_context_t<i_t, f_t>& context,
                                  const flow_cover_arc_spec_t<i_t, f_t>& spec,
                                  f_t y_star,
                                  std::vector<single_node_flow_candidate_t<i_t, f_t>>& candidates)
{
  const f_t bound_tol       = context.settings.primal_tol;
  const f_t feasibility_tol = context.settings.primal_tol;
  if (spec.u < -bound_tol) { return; }
  const f_t capacity = std::max<f_t>(0.0, spec.u);
  if (capacity <= feasibility_tol) { return; }

  single_node_flow_candidate_t<i_t, f_t> candidate;
  candidate.arc                  = flow_cover_build_arc(context,
                                       capacity,
                                       spec.in_n2,
                                       spec.x_col,
                                       spec.fixed_x,
                                       spec.y_const,
                                       spec.y_col,
                                       spec.y_coeff,
                                       spec.y_x_coeff);
  candidate.b_shift              = spec.b_shift;
  candidate.distance             = std::abs(spec.active_bound - y_star);
  candidate.absorbs_binary_coeff = spec.absorbs_binary_coeff;
  candidates.push_back(candidate);
}

template <typename f_t>
f_t flow_cover_mir_f(f_t alpha, f_t value)
{
  const f_t floor_value = std::floor(value);
  const f_t frac_value  = value - floor_value;
  return floor_value + std::max<f_t>(0.0, frac_value - alpha) / (1.0 - alpha);
}

template <typename i_t, typename f_t>
flow_cover_generation_t<i_t, f_t>::flow_cover_generation_t(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks)
  : is_slack_(lp.num_cols, 0)
{
  for (i_t j : new_slacks) {
    is_slack_[j] = 1;
  }

  // Wolter's separator is applied to every finite row side after attempting to construct a 0-1
  // single-node-flow relaxation. Rows that do not admit the relaxation are rejected inside
  // generate_cut.
  flow_cover_constraints_.reserve(2 * lp.num_rows);
  for (i_t i = 0; i < lp.num_rows; i++) {
    if (Arow.row_start[i + 1] <= Arow.row_start[i]) { continue; }
    i_t slack_col   = -1;
    f_t slack_coeff = 0.0;
    i_t slack_count = 0;
    for (i_t p = Arow.row_start[i]; p < Arow.row_start[i + 1]; p++) {
      if (is_slack_[Arow.j[p]]) {
        slack_col   = Arow.j[p];
        slack_coeff = Arow.x[p];
        slack_count++;
      }
    }
    if (slack_count > 1 || (slack_count == 1 && std::abs(slack_coeff) <= settings.zero_tol)) {
      continue;
    }
    if (slack_count == 0) {
      flow_cover_constraints_.push_back({i, false});
      flow_cover_constraints_.push_back({i, true});
      continue;
    }

    const f_t sigma_slack_lower =
      slack_coeff > 0.0 ? slack_coeff * lp.lower[slack_col] : slack_coeff * lp.upper[slack_col];
    const f_t sigma_slack_upper =
      slack_coeff > 0.0 ? slack_coeff * lp.upper[slack_col] : slack_coeff * lp.lower[slack_col];
    if (sigma_slack_lower > -inf) { flow_cover_constraints_.push_back({i, false}); }
    if (sigma_slack_upper < inf) { flow_cover_constraints_.push_back({i, true}); }
  }
}

// Repeated (variable, side, coefficient) queries share the same a=0 candidate because xstar and
// implied bounds are fixed for the pass. Cache that candidate and index nonzero-a candidates by
// controller so each row examines only bounds affected by its binary coefficients, to avoid an
// O(N^2) occurrence-by-bound scan when both grow with the model.
template <typename i_t, typename f_t>
void flow_cover_generation_t<i_t, f_t>::initialize_implied_bound_index(
  implied_bound_index_t& index,
  const std::vector<i_t>& offsets,
  const std::vector<i_t>& variables,
  const std::vector<f_t>& weights,
  const std::vector<f_t>& biases,
  const std::vector<variable_type_t>& var_types)
{
  cuopt_assert(!index.topology_initialized, "");
  cuopt_assert(index.groups.empty(), "");
  cuopt_assert(index.bound_indices.empty(), "");
  cuopt_assert(index.active_bounds.empty(), "");
  cuopt_assert(!offsets.empty(), "");
  cuopt_assert(variables.size() == weights.size(), "");
  cuopt_assert(variables.size() == biases.size(), "");
  cuopt_assert(offsets.back() == (i_t)variables.size(), "");

  index.y_group_offsets.resize(offsets.size());
  std::vector<std::pair<i_t, i_t>> source_bounds;
  const i_t number_of_variables = offsets.size() - 1;
  for (i_t j = 0; j < number_of_variables; j++) {
    cuopt_assert(offsets[j] <= offsets[j + 1], "");
    cuopt_assert(offsets[j] >= 0, "");
    cuopt_assert(offsets[j + 1] <= (i_t)variables.size(), "");
    index.y_group_offsets[j] = index.groups.size();
    source_bounds.clear();
    source_bounds.reserve(offsets[j + 1] - offsets[j]);
    for (i_t p = offsets[j]; p < offsets[j + 1]; p++) {
      const i_t controller = variables[p];
      cuopt_assert(controller >= 0 && controller < (i_t)var_types.size(), "");
      if (var_types[controller] != variable_type_t::INTEGER || !std::isfinite(weights[p]) ||
          !std::isfinite(biases[p])) {
        continue;
      }
      source_bounds.emplace_back(controller, p);
    }
    std::sort(source_bounds.begin(), source_bounds.end());

    size_t first = 0;
    while (first < source_bounds.size()) {
      implied_bound_group_t group;
      group.controller    = source_bounds[first].first;
      group.first_bound   = index.bound_indices.size();
      group.minimum_alpha = biases[source_bounds[first].second];
      group.maximum_alpha = biases[source_bounds[first].second];
      size_t last         = first;
      while (last < source_bounds.size() && source_bounds[last].first == group.controller) {
        const i_t source_bound = source_bounds[last].second;
        index.bound_indices.push_back(source_bound);
        index.active_bounds.push_back(0.0);
        group.minimum_alpha = std::min(group.minimum_alpha, biases[source_bound]);
        group.maximum_alpha = std::max(group.maximum_alpha, biases[source_bound]);
        last++;
      }
      group.number_of_bounds = index.bound_indices.size() - group.first_bound;
      cuopt_assert(group.number_of_bounds > 0, "");
      index.groups.push_back(group);
      first = last;
    }
  }
  index.y_group_offsets[number_of_variables] = index.groups.size();
  cuopt_assert(index.bound_indices.size() == index.active_bounds.size(), "");
  index.source_column_count  = number_of_variables;
  index.source_bound_count   = variables.size();
  index.topology_initialized = true;
}

template <typename i_t, typename f_t>
void flow_cover_generation_t<i_t, f_t>::preprocess_cut_pass(
  const lp_problem_t<i_t, f_t>& lp,
  const variable_bounds_t<i_t, f_t>& variable_bounds,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar)
{
  cut_pass_preprocessed = false;
  cuopt_assert(var_types.size() >= (size_t)lp.num_cols, "");
  cuopt_assert(xstar.size() >= (size_t)lp.num_cols, "");
  for (flow_cover_bound_side_t side :
       {flow_cover_bound_side_t::UPPER, flow_cover_bound_side_t::LOWER}) {
    const bool use_upper_bound = side == flow_cover_bound_side_t::UPPER;
    const auto& offsets =
      use_upper_bound ? variable_bounds.upper_offsets : variable_bounds.lower_offsets;
    const auto& variables =
      use_upper_bound ? variable_bounds.upper_variables : variable_bounds.lower_variables;
    const auto& weights =
      use_upper_bound ? variable_bounds.upper_weights : variable_bounds.lower_weights;
    const auto& biases =
      use_upper_bound ? variable_bounds.upper_biases : variable_bounds.lower_biases;
    auto& preprocessed = use_upper_bound ? upper_implied_bounds : lower_implied_bounds;
    cuopt_assert(offsets.size() >= (size_t)(lp.num_cols + 1), "");
    cuopt_assert(variables.size() == weights.size(), "");
    cuopt_assert(variables.size() == biases.size(), "");

    if (!preprocessed.topology_initialized) {
      initialize_implied_bound_index(preprocessed, offsets, variables, weights, biases, var_types);
    } else {
      cuopt_assert((size_t)preprocessed.source_bound_count == variables.size(), "");
      cuopt_assert(offsets.size() >= (size_t)(preprocessed.source_column_count + 1), "");
      for (size_t j = preprocessed.source_column_count; j < offsets.size(); j++) {
        cuopt_assert(offsets[j] == preprocessed.source_bound_count, "");
      }
      preprocessed.y_group_offsets.resize(offsets.size(), preprocessed.groups.size());
    }
    preprocessed.zero_candidate_cache.clear();

    for (const auto& group : preprocessed.groups) {
      cuopt_assert(group.controller >= 0 && group.controller < lp.num_cols, "");
      cuopt_assert(group.first_bound >= 0, "");
      cuopt_assert(group.number_of_bounds > 0, "");
      cuopt_assert(
        group.first_bound + group.number_of_bounds <= (i_t)preprocessed.bound_indices.size(), "");
      for (i_t bound = group.first_bound; bound < group.first_bound + group.number_of_bounds;
           bound++) {
        const i_t source_bound = preprocessed.bound_indices[bound];
        cuopt_assert(source_bound >= 0 && source_bound < preprocessed.source_bound_count, "");
        preprocessed.active_bounds[bound] =
          weights[source_bound] * xstar[group.controller] + biases[source_bound];
      }
    }
  }
  cut_pass_preprocessed = true;
}

template <typename i_t, typename f_t>
bool flow_cover_generation_t<i_t, f_t>::normalize_row_side(
  const flow_cover_context_t<i_t, f_t>& context,
  const flow_cover_row_t<i_t>& flow_cover_row,
  inequality_t<i_t, f_t>& row,
  f_t& b,
  bool& negate_row)
{
  const f_t coefficient_tol = static_cast<f_t>(1e-6);
  auto& scratch             = *this;
  row =
    inequality_t<i_t, f_t>(context.Arow, flow_cover_row.row, context.lp.rhs[flow_cover_row.row]);

  i_t slack_count   = 0;
  i_t slack_col     = -1;
  f_t slack_coeff   = 0.0;
  const i_t row_len = row.size();
  for (i_t k = 0; k < row_len; k++) {
    const i_t j = row.index(k);
    if (!is_slack_[j]) { continue; }
    slack_count++;
    slack_col   = j;
    slack_coeff = row.coeff(k);
  }
  if (slack_count > 1) { return false; }

  negate_row = flow_cover_row.reverse;
  b          = negate_row ? -row.rhs : row.rhs;
  if (slack_count == 1) {
    if (std::abs(slack_coeff) <= coefficient_tol) { return false; }
    const f_t sigma_slack_lower = slack_coeff > 0.0 ? slack_coeff * context.lp.lower[slack_col]
                                                    : slack_coeff * context.lp.upper[slack_col];
    const f_t sigma_slack_upper = slack_coeff > 0.0 ? slack_coeff * context.lp.upper[slack_col]
                                                    : slack_coeff * context.lp.lower[slack_col];
    if (!flow_cover_row.reverse) {
      if (sigma_slack_lower <= -inf) { return false; }
      negate_row = false;
      b          = row.rhs - sigma_slack_lower;
    } else {
      if (sigma_slack_upper >= inf) { return false; }
      negate_row = true;
      b          = -row.rhs + sigma_slack_upper;
    }
  }
  if (!std::isfinite(b)) { return false; }

  cuopt_assert(
    [&]() {
      f_t row_side_activity = 0.0;
      f_t row_side_scale    = std::max<f_t>(1.0, std::abs(b));
      for (i_t k = 0; k < row_len; k++) {
        const i_t j = row.index(k);
        if (is_slack_[j]) { continue; }
        const f_t coeff = negate_row ? -row.coeff(k) : row.coeff(k);
        row_side_activity += coeff * context.xstar[j];
        row_side_scale += std::abs(coeff * context.xstar[j]);
      }
      return row_side_activity <= b + flow_cover_scaled_primal_tol(context, row_side_scale);
    }(),
    "Flow cover normalized row side excludes LP solution");

  scratch.continuous_terms.reserve(row_len);
  scratch.binary_columns.reserve(row_len);
  auto add_binary_coeff = [&](i_t j, f_t coeff) {
    if (!scratch.binary_coefficients_touched[j]) {
      scratch.binary_coefficients_touched[j] = 1;
      scratch.touched_binary_coefficients.push_back(j);
      scratch.binary_columns.push_back(j);
    }
    scratch.binary_coefficients[j] += coeff;
  };

  for (i_t k = 0; k < row_len; k++) {
    const i_t j = row.index(k);
    if (is_slack_[j]) { continue; }
    const f_t coeff = negate_row ? -row.coeff(k) : row.coeff(k);
    if (std::abs(coeff) <= coefficient_tol) { continue; }
    if (flow_cover_is_zero_one_integer_variable(context, j)) {
      add_binary_coeff(j, coeff);
    } else {
      scratch.continuous_terms.push_back({j, coeff});
    }
  }

  return true;
}

template <typename i_t, typename f_t>
bool flow_cover_generation_t<i_t, f_t>::try_add_implied_bound_candidate(
  const flow_cover_context_t<i_t, f_t>& context,
  i_t variable,
  f_t coefficient,
  bool use_upper_bound,
  i_t bound,
  f_t binary_coefficient)
{
  const auto& preprocessed = use_upper_bound ? upper_implied_bounds : lower_implied_bounds;
  cuopt_assert(bound >= 0 && bound < (i_t)preprocessed.bound_indices.size(), "");
  const i_t source_bound = preprocessed.bound_indices[bound];
  cuopt_assert(source_bound >= 0 && source_bound < preprocessed.source_bound_count, "");
  const auto& bound_variables = use_upper_bound ? context.variable_bounds.upper_variables
                                                : context.variable_bounds.lower_variables;
  const auto& bound_weights =
    use_upper_bound ? context.variable_bounds.upper_weights : context.variable_bounds.lower_weights;
  const auto& bound_biases =
    use_upper_bound ? context.variable_bounds.upper_biases : context.variable_bounds.lower_biases;

  const f_t endpoint = use_upper_bound ? context.lp.lower[variable] : context.lp.upper[variable];
  const bool in_n2   = use_upper_bound ? coefficient < 0.0 : coefficient > 0.0;
  const i_t x_col    = bound_variables[source_bound];
  const f_t gamma    = bound_weights[source_bound];
  const f_t alpha    = bound_biases[source_bound];
  cuopt_assert(flow_cover_is_zero_one_integer_variable(context, x_col), "");
  const f_t signed_capacity = coefficient * gamma + binary_coefficient;
  const f_t endpoint_term   = coefficient * (endpoint - alpha);
  const f_t bound_tol       = context.settings.primal_tol;
  const bool valid_endpoint =
    flow_cover_valid_endpoint_terms(endpoint_term, binary_coefficient, in_n2, bound_tol) &&
    (in_n2 ? signed_capacity <= bound_tol : signed_capacity >= -bound_tol);
  if (!valid_endpoint) { return false; }

  flow_cover_arc_spec_t<i_t, f_t> spec;
  spec.u                    = in_n2 ? -signed_capacity : signed_capacity;
  spec.in_n2                = in_n2;
  spec.x_col                = x_col;
  spec.fixed_x              = 0.0;
  spec.y_const              = in_n2 ? coefficient * alpha : -coefficient * alpha;
  spec.y_col                = variable;
  spec.y_coeff              = in_n2 ? -coefficient : coefficient;
  spec.y_x_coeff            = in_n2 ? -binary_coefficient : binary_coefficient;
  spec.b_shift              = coefficient * alpha;
  spec.active_bound         = preprocessed.active_bounds[bound];
  spec.absorbs_binary_coeff = std::abs(binary_coefficient) > static_cast<f_t>(1e-6);
  const size_t size         = candidates.size();
  flow_cover_try_add_candidate(context, spec, context.xstar[variable], candidates);
  return candidates.size() > size;
}

template <typename i_t, typename f_t>
bool flow_cover_generation_t<i_t, f_t>::build_single_node_flow_relaxation(
  const flow_cover_context_t<i_t, f_t>& context, f_t b, f_t& single_node_flow_b)
{
  auto& scratch             = *this;
  const f_t coefficient_tol = static_cast<f_t>(1e-6);
  const f_t feasibility_tol = context.settings.primal_tol;
  f_t b_shift               = 0.0;

  scratch.arcs.reserve(scratch.continuous_terms.size() + scratch.binary_columns.size());

  auto add_variable_bound_candidates = [&](i_t j, f_t c, flow_cover_bound_side_t side) {
    const bool use_upper_bound = side == flow_cover_bound_side_t::UPPER;
    const f_t lower_j          = context.lp.lower[j];
    const f_t upper_j          = context.lp.upper[j];
    if (use_upper_bound && lower_j <= -inf) { return; }
    if (!use_upper_bound && upper_j >= inf) { return; }

    auto& preprocessed = use_upper_bound ? upper_implied_bounds : lower_implied_bounds;
    cuopt_assert(j >= 0 && j + 1 < (i_t)preprocessed.y_group_offsets.size(), "");
    const i_t first_group = preprocessed.y_group_offsets[j];
    const i_t last_group  = preprocessed.y_group_offsets[j + 1];
    cuopt_assert(first_group >= 0 && first_group <= last_group, "");
    cuopt_assert(last_group <= (i_t)preprocessed.groups.size(), "");

    // Small nonzero coefficients exclude a=0 for their controller, making the cache row-dependent.
    bool has_small_direct_coeff = false;
    for (i_t x_col : scratch.binary_columns) {
      const f_t direct_coeff = scratch.binary_coefficients[x_col];
      if (direct_coeff != 0.0 && std::abs(direct_coeff) <= coefficient_tol) {
        has_small_direct_coeff = true;
        break;
      }
    }

    auto& zero_candidate_cache = preprocessed.zero_candidate_cache;
    const zero_candidate_key_t zero_candidate_key{j, c};
    auto cached_zero = zero_candidate_cache.end();
    if (!has_small_direct_coeff) { cached_zero = zero_candidate_cache.find(zero_candidate_key); }
    if (has_small_direct_coeff || cached_zero == zero_candidate_cache.end()) {
      // Compute the best a=0 candidate when it is uncached or row-specific.
      i_t best_zero         = -1;
      i_t best_source_bound = -1;
      f_t best_distance     = inf;
      for (i_t group_index = first_group; group_index < last_group; group_index++) {
        const auto& group = preprocessed.groups[group_index];
        if (!flow_cover_is_zero_one_integer_variable(context, group.controller)) { continue; }
        const f_t direct_coeff = scratch.binary_coefficients_touched[group.controller]
                                   ? scratch.binary_coefficients[group.controller]
                                   : 0.0;
        if (direct_coeff != 0.0 && std::abs(direct_coeff) <= coefficient_tol) { continue; }
        for (i_t bound = group.first_bound; bound < group.first_bound + group.number_of_bounds;
             bound++) {
          // The normal acceptance gate keeps cached and uncached candidate validity identical.
          if (!try_add_implied_bound_candidate(context, j, c, use_upper_bound, bound, 0.0)) {
            continue;
          }
          scratch.candidates.pop_back();
          const i_t source_bound = preprocessed.bound_indices[bound];
          const f_t distance     = std::abs(preprocessed.active_bounds[bound] - context.xstar[j]);
          if (best_zero < 0 || distance < best_distance ||
              (distance == best_distance && source_bound < best_source_bound)) {
            best_zero         = bound;
            best_source_bound = source_bound;
            best_distance     = distance;
          }
        }
      }
      if (!has_small_direct_coeff) { zero_candidate_cache.emplace(zero_candidate_key, best_zero); }
      if (best_zero >= 0) {
        const bool added =
          try_add_implied_bound_candidate(context, j, c, use_upper_bound, best_zero, 0.0);
        cuopt_assert(added, "");
      }
    } else {
      // Reuse the pass-wide a=0 candidate for this variable, side, and coefficient.
      if (cached_zero->second >= 0) {
        const i_t bound = cached_zero->second;
        const bool added =
          try_add_implied_bound_candidate(context, j, c, use_upper_bound, bound, 0.0);
        cuopt_assert(added, "");
      }
    }

    for (i_t x_col : scratch.binary_columns) {
      const f_t direct_coeff = scratch.binary_coefficients[x_col];
      if (direct_coeff == 0.0) { continue; }
      const auto first = preprocessed.groups.begin() + first_group;
      const auto last  = preprocessed.groups.begin() + last_group;
      const auto group = std::lower_bound(
        first, last, x_col, [](const implied_bound_group_t& candidate, i_t controller) {
          return candidate.controller < controller;
        });
      if (group == last || group->controller != x_col) { continue; }
      cuopt_assert(group->number_of_bounds > 0, "");
      cuopt_assert(group->minimum_alpha <= group->maximum_alpha, "");
      // Endpoint feasibility is monotone in alpha, so the side's extreme bounds the whole group.
      const f_t endpoint      = use_upper_bound ? lower_j : upper_j;
      const f_t alpha         = use_upper_bound ? group->minimum_alpha : group->maximum_alpha;
      const f_t endpoint_term = c * (endpoint - alpha);
      const bool in_n2        = use_upper_bound ? c < 0.0 : c > 0.0;
      if (!flow_cover_valid_endpoint_terms(
            endpoint_term, direct_coeff, in_n2, context.settings.primal_tol)) {
        continue;
      }
      for (i_t bound = group->first_bound; bound < group->first_bound + group->number_of_bounds;
           bound++) {
        try_add_implied_bound_candidate(context, j, c, use_upper_bound, bound, direct_coeff);
      }
    }
  };

  auto add_simple_bound_candidates = [&](i_t j, f_t c) {
    const f_t lower_j = context.lp.lower[j];
    const f_t upper_j = context.lp.upper[j];
    if (lower_j <= -inf || upper_j >= inf) { return; }
    const f_t capacity = std::abs(c) * (upper_j - lower_j);
    if (capacity <= feasibility_tol) { return; }

    auto add_simple_candidate =
      [&](bool in_n2, f_t y_const, f_t y_coeff, f_t shift, f_t active_bound) {
        flow_cover_arc_spec_t<i_t, f_t> spec;
        spec.u                    = capacity;
        spec.in_n2                = in_n2;
        spec.x_col                = -1;
        spec.fixed_x              = 1.0;
        spec.y_const              = y_const;
        spec.y_col                = j;
        spec.y_coeff              = y_coeff;
        spec.y_x_coeff            = 0.0;
        spec.b_shift              = shift;
        spec.active_bound         = active_bound;
        spec.absorbs_binary_coeff = false;
        flow_cover_try_add_candidate(context, spec, context.xstar[j], scratch.candidates);
      };

    if (c > 0.0) {
      add_simple_candidate(false, -c * lower_j, c, c * lower_j, upper_j);
      add_simple_candidate(true, c * upper_j, -c, c * upper_j, lower_j);
    } else {
      add_simple_candidate(true, c * lower_j, -c, c * lower_j, upper_j);
      add_simple_candidate(false, -c * upper_j, c, c * upper_j, lower_j);
    }
  };

  for (const auto& [j, c] : scratch.continuous_terms) {
    scratch.candidates.clear();
    add_variable_bound_candidates(j, c, flow_cover_bound_side_t::UPPER);
    add_variable_bound_candidates(j, c, flow_cover_bound_side_t::LOWER);
    add_simple_bound_candidates(j, c);

    if (scratch.candidates.empty()) { return false; }
    auto best = std::min_element(
      scratch.candidates.begin(),
      scratch.candidates.end(),
      [](const single_node_flow_candidate_t<i_t, f_t>& a,
         const single_node_flow_candidate_t<i_t, f_t>& b) { return a.distance < b.distance; });
    const f_t arc_lower_tol = flow_cover_arc_lower_tol(context);
    const f_t arc_upper_tol = flow_cover_arc_upper_tol(context, best->arc.u);
    if (best->arc.y_value < -arc_lower_tol ||
        best->arc.y_value > best->arc.u * best->arc.x_value + arc_upper_tol) {
      return false;
    }
    if (best->arc.y_value < 0.0) { best->arc.y_value = 0.0; }
    scratch.arcs.push_back(best->arc);
    b_shift += best->b_shift;
    if (best->absorbs_binary_coeff && best->arc.x_col >= 0) {
      scratch.binary_coefficients[best->arc.x_col] = 0.0;
    }
  }

  for (i_t j : scratch.binary_columns) {
    const f_t coeff = scratch.binary_coefficients[j];
    if (std::abs(coeff) <= coefficient_tol) { continue; }
    const f_t u = std::abs(coeff);
    single_node_flow_arc_t<i_t, f_t> arc =
      flow_cover_build_arc(context, u, coeff < 0.0, j, 0.0, 0.0, -1, 0.0, u);
    // y_value is built from the raw LP value (u * xstar[j]) while x_value is clamped to [0, 1].
    // When xstar[j] sits marginally below its bound (e.g. -5e-7, normal LP feasibility slop),
    // y_value picks up a tiny negative (u * xstar) that trips the arc's lower gate even though the
    // flow is really 0. Continuous-term arcs are already clamped this way above (see the
    // `best->arc.y_value < 0.0` clamp); the binary arcs bypassed both the gate and the clamp.
    if (arc.y_value < 0.0) { arc.y_value = 0.0; }
    scratch.arcs.push_back(arc);
  }

  if (scratch.arcs.empty()) { return false; }
  single_node_flow_b = b - b_shift;

  cuopt_assert(
    [&]() {
      f_t single_node_flow_activity = 0.0;
      f_t single_node_flow_scale    = std::max<f_t>(1.0, std::abs(single_node_flow_b));
      // Each accepted arc may sit up to arc_{lower,upper}_tol outside its capacity — the same
      // slack the acceptance gate above tolerates — and the y_value<0 -> 0 clamp can shift it by
      // another arc_lower_tol. Those per-arc slacks accumulate into the aggregate activity, so the
      // relaxation tolerance must sum them; primal_tol * scale alone (which only covers FP rounding
      // in the summation) is far too tight and rejects LP points that are genuinely inside.
      f_t single_node_flow_slack = 0.0;
      for (const auto& arc : scratch.arcs) {
        const f_t arc_lower_tol = flow_cover_arc_lower_tol(context);
        const f_t arc_upper_tol = flow_cover_arc_upper_tol(context, arc.u);
        if (arc.y_value < -arc_lower_tol) { return false; }
        if (arc.y_value > arc.u * arc.x_value + arc_upper_tol) { return false; }
        const f_t signed_y = arc.in_n2 ? -arc.y_value : arc.y_value;
        single_node_flow_activity += signed_y;
        single_node_flow_scale += std::abs(signed_y);
        single_node_flow_slack += arc_lower_tol + arc_upper_tol;
      }
      return single_node_flow_activity <= single_node_flow_b +
                                            context.settings.primal_tol * single_node_flow_scale +
                                            single_node_flow_slack;
    }(),
    "Flow cover single-node-flow relaxation excludes LP solution");

  return true;
}

template <typename i_t, typename f_t>
bool flow_cover_generation_t<i_t, f_t>::separate_single_node_flow_cover(
  const flow_cover_context_t<i_t, f_t>& context, f_t single_node_flow_b, f_t& lambda)
{
  auto& scratch             = *this;
  const f_t feasibility_tol = context.settings.primal_tol;

  bool has_binary_controlled_arc = false;
  bool has_n1                    = false;
  f_t sum_n1_u                   = 0.0;
  for (const auto& arc : scratch.arcs) {
    if (arc.x_col >= 0) { has_binary_controlled_arc = true; }
    if (!arc.in_n2) {
      has_n1 = true;
      sum_n1_u += arc.u;
    }
  }
  const f_t cover_capacity = -single_node_flow_b + sum_n1_u;
  if (!has_binary_controlled_arc || !has_n1 || cover_capacity <= feasibility_tol) { return false; }

  scratch.values.reserve(scratch.arcs.size());
  scratch.weights.reserve(scratch.arcs.size());
  scratch.item_to_arc.reserve(scratch.arcs.size());
  for (i_t k = 0; k < static_cast<i_t>(scratch.arcs.size()); k++) {
    const auto& arc = scratch.arcs[k];
    if (arc.u <= feasibility_tol) { continue; }
    const f_t value = arc.in_n2 ? arc.x_value : 1.0 - arc.x_value;
    scratch.values.push_back(std::max<f_t>(0.0, value));
    scratch.weights.push_back(arc.u);
    scratch.item_to_arc.push_back(k);
  }
  if (scratch.values.empty()) { return false; }

  const f_t objective =
    greedy_knapsack_problem<i_t, f_t>(scratch.values,
                                      scratch.weights,
                                      cover_capacity,
                                      scratch.solution,
                                      greedy_knapsack_mode_t::STRICT_RATIO_PREFIX);
  if (std::isnan(objective)) { return false; }

  scratch.in_c1.assign(scratch.arcs.size(), 0);
  scratch.in_c2.assign(scratch.arcs.size(), 0);
  for (i_t item = 0; item < static_cast<i_t>(scratch.solution.size()); item++) {
    const i_t arc_index = scratch.item_to_arc[item];
    if (scratch.arcs[arc_index].in_n2) {
      if (scratch.solution[item] > 0.5) { scratch.in_c2[arc_index] = 1; }
    } else {
      if (scratch.solution[item] <= 0.5) { scratch.in_c1[arc_index] = 1; }
    }
  }

  f_t sum_c1_u     = 0.0;
  f_t sum_c2_u     = 0.0;
  bool c1_nonempty = false;
  for (i_t k = 0; k < static_cast<i_t>(scratch.arcs.size()); k++) {
    if (scratch.in_c1[k]) {
      sum_c1_u += scratch.arcs[k].u;
      c1_nonempty = true;
    }
    if (scratch.in_c2[k]) { sum_c2_u += scratch.arcs[k].u; }
  }

  lambda = -single_node_flow_b + sum_c1_u - sum_c2_u;
  return c1_nonempty && lambda > feasibility_tol;
}

template <typename i_t, typename f_t>
flow_cover_evaluation_t<f_t>
flow_cover_generation_t<i_t, f_t>::evaluate_c_mir_flow_cover_inequality(
  const flow_cover_context_t<i_t, f_t>& context, f_t single_node_flow_b, f_t lambda)
{
  auto& scratch                       = *this;
  constexpr f_t min_mir_beta_fraction = 0.01;
  constexpr f_t max_mir_beta_fraction = 0.95;
  // Wolter (2006), Section 5.4: testing at most MAXTESTDELTA = 10 values of ubar keeps nearly all
  // of the gap closed by the full candidate set N*_3, whose size grows with the number of distinct
  // arc capacities in the row.
  constexpr size_t max_test_delta = 10;
  const f_t min_violation         = 1e-6;
  const f_t lambda_tol            = min_violation;
  f_t max_c1_ltilde2              = -inf;
  f_t max_c1                      = -inf;
  f_t max_u_ge_lambda             = -inf;
  const i_t number_of_arcs        = scratch.arcs.size();

  for (i_t k = 0; k < number_of_arcs; k++) {
    const auto& arc = scratch.arcs[k];
    if (arc.u >= lambda - lambda_tol) { max_u_ge_lambda = std::max(max_u_ge_lambda, arc.u); }
    if (!arc.in_n2 && scratch.in_c1[k] && arc.u > lambda + lambda_tol) {
      max_c1         = std::max(max_c1, arc.u);
      max_c1_ltilde2 = std::max(max_c1_ltilde2, arc.u);
    }
    if (arc.in_n2 && !scratch.in_c2[k] && arc.u > lambda + lambda_tol &&
        std::min(arc.u, lambda) * arc.x_value <= arc.y_value + min_violation) {
      max_c1_ltilde2 = std::max(max_c1_ltilde2, arc.u);
    }
  }

  // Only candidates that pass the f_beta filter generate an inequality, so only they count
  // towards max_test_delta.
  auto add_ubar_candidate = [&](f_t candidate) {
    if (candidate <= lambda + lambda_tol || !std::isfinite(candidate)) { return; }
    const f_t f_beta = fractional_part(-lambda / candidate);
    if (f_beta < min_mir_beta_fraction || f_beta > max_mir_beta_fraction) { return; }
    for (f_t existing : scratch.ubar_candidates) {
      const f_t duplicate_tol = 1e-8 * std::max<f_t>(1.0, std::abs(candidate));
      if (std::abs(existing - candidate) <= duplicate_tol) { return; }
    }
    scratch.ubar_candidates.push_back(candidate);
  };

  // Test N*_2 and max{u_j : u_j >= lambda} + 1 first, then fill up with the arc capacities of N*_3.
  add_ubar_candidate(max_c1_ltilde2);
  add_ubar_candidate(max_c1);
  add_ubar_candidate(lambda + 1.0);
  add_ubar_candidate(max_u_ge_lambda + 1.0);
  for (i_t k = 0; k < number_of_arcs && scratch.ubar_candidates.size() < max_test_delta; k++) {
    add_ubar_candidate(scratch.arcs[k].u);
  }

  flow_cover_evaluation_t<f_t> best{0.0, 0.0};
  if (scratch.ubar_candidates.empty()) { return best; }

  // L1 and L2 are chosen by comparison (Wolter (5.11) and (5.12)), so each arc contributes the
  // better of its two options directly and the sets are only materialized for the best ubar.
  for (f_t ubar : scratch.ubar_candidates) {
    const f_t f_beta = fractional_part(-lambda / ubar);
    f_t lhs_value    = 0.0;
    for (i_t k = 0; k < number_of_arcs; k++) {
      const auto& arc = scratch.arcs[k];
      if (scratch.in_c1[k]) {
        const f_t f_neg = flow_cover_mir_f(f_beta, -arc.u / ubar);
        lhs_value += arc.y_value + (arc.u + lambda * f_neg) * (1.0 - arc.x_value);
      } else if (!arc.in_n2) {
        const f_t f_pos   = flow_cover_mir_f(f_beta, arc.u / ubar);
        const f_t l1_term = arc.y_value - (arc.u - lambda * f_pos) * arc.x_value;
        if (l1_term >= -min_violation) { lhs_value += l1_term; }
      } else if (scratch.in_c2[k]) {
        const f_t f_pos = flow_cover_mir_f(f_beta, arc.u / ubar);
        lhs_value += -arc.u + lambda * f_pos * (1.0 - arc.x_value);
      } else {
        const f_t f_neg   = flow_cover_mir_f(f_beta, -arc.u / ubar);
        const f_t l2_term = lambda * f_neg * arc.x_value;
        lhs_value += l2_term >= -arc.y_value - min_violation ? l2_term : -arc.y_value;
      }
    }
    const f_t violation = lhs_value - single_node_flow_b;
    if (violation > best.violation + min_violation) {
      best.violation = violation;
      best.ubar      = ubar;
    }
  }

  scratch.best_in_l1.assign(number_of_arcs, 0);
  scratch.best_in_l2.assign(number_of_arcs, 0);
  if (best.ubar > 0.0) {
    const f_t f_beta = fractional_part(-lambda / best.ubar);
    for (i_t k = 0; k < number_of_arcs; k++) {
      const auto& arc = scratch.arcs[k];
      if (!arc.in_n2 && !scratch.in_c1[k]) {
        const f_t f_pos = flow_cover_mir_f(f_beta, arc.u / best.ubar);
        if (arc.y_value - (arc.u - lambda * f_pos) * arc.x_value >= -min_violation) {
          scratch.best_in_l1[k] = 1;
        }
      } else if (arc.in_n2 && !scratch.in_c2[k]) {
        const f_t f_neg = flow_cover_mir_f(f_beta, -arc.u / best.ubar);
        if (lambda * f_neg * arc.x_value >= -arc.y_value - min_violation) {
          scratch.best_in_l2[k] = 1;
        }
      }
    }
  }

  return best;
}

template <typename i_t, typename f_t>
flow_cover_evaluation_t<f_t>
flow_cover_generation_t<i_t, f_t>::evaluate_simple_generalized_flow_cover_inequality(
  const flow_cover_context_t<i_t, f_t>& context, f_t single_node_flow_b, f_t lambda)
{
  auto& scratch           = *this;
  const f_t min_violation = static_cast<f_t>(1e-6);
  scratch.simple_generalized_flow_cover_in_l2.assign(scratch.arcs.size(), 0);

  f_t lhs_value = 0.0;
  for (i_t k = 0; k < static_cast<i_t>(scratch.arcs.size()); k++) {
    const auto& arc = scratch.arcs[k];
    if (scratch.in_c1[k]) {
      lhs_value += arc.y_value + std::max<f_t>(0.0, arc.u - lambda) * (1.0 - arc.x_value);
    } else if (!arc.in_n2) {
      continue;
    } else if (scratch.in_c2[k]) {
      lhs_value -= arc.u;
    } else if (std::min(arc.u, lambda) * arc.x_value <= arc.y_value + min_violation) {
      scratch.simple_generalized_flow_cover_in_l2[k] = 1;
      lhs_value -= std::min(arc.u, lambda) * arc.x_value;
    } else {
      lhs_value -= arc.y_value;
    }
  }

  return {lhs_value - single_node_flow_b, 0.0};
}

template <typename i_t, typename f_t>
bool flow_cover_generation_t<i_t, f_t>::emit_flow_cover_cut(
  const flow_cover_context_t<i_t, f_t>& context,
  f_t single_node_flow_b,
  f_t lambda,
  const flow_cover_evaluation_t<f_t>& c_mir_inequality,
  const flow_cover_evaluation_t<f_t>& simple_generalized_inequality,
  inequality_t<i_t, f_t>& cut)
{
  auto& scratch             = *this;
  const f_t accumulator_tol = static_cast<f_t>(1e-12);
  const f_t output_drop_tol = static_cast<f_t>(1e-9);
  const bool use_c_mir_inequality =
    c_mir_inequality.violation >= simple_generalized_inequality.violation;
  f_t lhs_constant = 0.0;

  scratch.lhs_indices.reserve(scratch.arcs.size() * 2);
  auto add_lhs_coeff = [&](i_t j, f_t value) {
    if (j < 0 || std::abs(value) <= accumulator_tol) { return; }
    if (!scratch.lhs_coefficients_touched[j]) {
      scratch.lhs_coefficients_touched[j] = 1;
      scratch.touched_lhs_coefficients.push_back(j);
      scratch.lhs_indices.push_back(j);
    }
    scratch.lhs_coefficients[j] += value;
  };

  auto add_y = [&](const single_node_flow_arc_t<i_t, f_t>& arc, f_t multiplier) {
    lhs_constant += multiplier * arc.y_const;
    if (arc.y_col >= 0) { add_lhs_coeff(arc.y_col, multiplier * arc.y_coeff); }
    if (arc.x_col >= 0) {
      add_lhs_coeff(arc.x_col, multiplier * arc.y_x_coeff);
    } else {
      lhs_constant += multiplier * arc.y_x_coeff * arc.x_value;
    }
  };

  auto add_x = [&](const single_node_flow_arc_t<i_t, f_t>& arc, f_t multiplier) {
    if (arc.x_col >= 0) {
      add_lhs_coeff(arc.x_col, multiplier);
    } else {
      lhs_constant += multiplier * arc.x_value;
    }
  };

  auto add_one_minus_x = [&](const single_node_flow_arc_t<i_t, f_t>& arc, f_t multiplier) {
    lhs_constant += multiplier;
    add_x(arc, -multiplier);
  };

  if (use_c_mir_inequality) {
    const f_t f_beta_best = fractional_part(-lambda / c_mir_inequality.ubar);
    for (i_t k = 0; k < static_cast<i_t>(scratch.arcs.size()); k++) {
      const auto& arc = scratch.arcs[k];
      const f_t f_pos = flow_cover_mir_f(f_beta_best, arc.u / c_mir_inequality.ubar);
      const f_t f_neg = flow_cover_mir_f(f_beta_best, -arc.u / c_mir_inequality.ubar);
      if (scratch.in_c1[k]) {
        add_y(arc, 1.0);
        add_one_minus_x(arc, arc.u + lambda * f_neg);
      } else if (!arc.in_n2) {
        if (scratch.best_in_l1[k]) {
          add_y(arc, 1.0);
          add_x(arc, -(arc.u - lambda * f_pos));
        }
      } else if (scratch.in_c2[k]) {
        lhs_constant -= arc.u;
        add_one_minus_x(arc, lambda * f_pos);
      } else if (scratch.best_in_l2[k]) {
        add_x(arc, lambda * f_neg);
      } else {
        add_y(arc, -1.0);
      }
    }
  } else {
    for (i_t k = 0; k < static_cast<i_t>(scratch.arcs.size()); k++) {
      const auto& arc = scratch.arcs[k];
      if (scratch.in_c1[k]) {
        add_y(arc, 1.0);
        add_one_minus_x(arc, std::max<f_t>(0.0, arc.u - lambda));
      } else if (!arc.in_n2) {
        continue;
      } else if (scratch.in_c2[k]) {
        lhs_constant -= arc.u;
      } else if (scratch.simple_generalized_flow_cover_in_l2[k]) {
        add_x(arc, -std::min(arc.u, lambda));
      } else {
        add_y(arc, -1.0);
      }
    }
  }

  cut.clear();
  cut.reserve(scratch.lhs_indices.size());
  for (i_t j : scratch.lhs_indices) {
    const f_t coeff = scratch.lhs_coefficients[j];
    if (std::abs(coeff) > output_drop_tol) { cut.push_back(j, -coeff); }
  }
  if (cut.size() == 0) { return false; }
  cut.rhs = lhs_constant - single_node_flow_b;
  cut.sort();

  // Small residual-RHS cuts are common and useful at normal scales. They become
  // numerically dangerous when the RHS is the difference of two large internal
  // SNF constants, because the added row can destabilize warm dual-simplex
  // reoptimization even though the cut is algebraically valid.
  const f_t cut_rhs_scale = std::max<f_t>(
    static_cast<f_t>(1.0), std::max(std::abs(lhs_constant), std::abs(single_node_flow_b)));
  const f_t cut_rhs_cancel_ratio          = std::abs(cut.rhs) / cut_rhs_scale;
  constexpr f_t large_cut_rhs_scale       = static_cast<f_t>(1e5);
  constexpr f_t unstable_cut_rhs_residual = static_cast<f_t>(1e-5);
  if (cut_rhs_scale >= large_cut_rhs_scale && cut_rhs_cancel_ratio <= unstable_cut_rhs_residual) {
    return false;
  }

  const f_t dot           = cut.vector.dot(context.xstar);
  const f_t violation_tol = std::max(context.settings.primal_tol, static_cast<f_t>(1e-6));
  return dot < cut.rhs - violation_tol;
}

template <typename i_t, typename f_t>
i_t flow_cover_generation_t<i_t, f_t>::generate_cut(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const variable_bounds_t<i_t, f_t>& variable_bounds,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  const flow_cover_row_t<i_t>& flow_cover_row,
  inequality_t<i_t, f_t>& cut)
{
  cuopt_assert(cut_pass_preprocessed, "");
  flow_cover_context_t<i_t, f_t> context{lp, settings, Arow, variable_bounds, var_types, xstar};
  clear_cut_state(lp.num_cols);

  inequality_t<i_t, f_t> row;
  f_t b           = 0.0;
  bool negate_row = false;
  if (!normalize_row_side(context, flow_cover_row, row, b, negate_row)) { return -1; }

  f_t single_node_flow_b = 0.0;
  if (!build_single_node_flow_relaxation(context, b, single_node_flow_b)) { return -1; }

  f_t lambda = 0.0;
  if (!separate_single_node_flow_cover(context, single_node_flow_b, lambda)) { return -1; }

  const auto c_mir_inequality =
    evaluate_c_mir_flow_cover_inequality(context, single_node_flow_b, lambda);
  const auto simple_generalized_inequality =
    evaluate_simple_generalized_flow_cover_inequality(context, single_node_flow_b, lambda);
  const f_t violation_tol = std::max(settings.primal_tol, static_cast<f_t>(1e-6));
  if (c_mir_inequality.violation <= violation_tol &&
      simple_generalized_inequality.violation <= violation_tol) {
    return -1;
  }

  if (!emit_flow_cover_cut(context,
                           single_node_flow_b,
                           lambda,
                           c_mir_inequality,
                           simple_generalized_inequality,
                           cut)) {
    return -1;
  }
  return 0;
}

template <typename i_t, typename f_t>
void cut_generation_t<i_t, f_t>::generate_flow_cover_cuts(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  variable_bounds_t<i_t, f_t>& variable_bounds,
  f_t start_time)
{
  if (flow_cover_generation_.num_constraints() > 0) {
    if (toc(start_time) >= settings.time_limit) { return; }
    flow_cover_generation_.preprocess_cut_pass(lp, variable_bounds, var_types, xstar);
    if (toc(start_time) >= settings.time_limit) { return; }
    for (const auto& flow_cover_row : flow_cover_generation_.get_constraints()) {
      if (toc(start_time) >= settings.time_limit) { return; }
      inequality_t<i_t, f_t> cut(lp.num_cols);
      i_t status = flow_cover_generation_.generate_cut(
        lp, settings, Arow, variable_bounds, var_types, xstar, flow_cover_row, cut);
      if (status == 0) { cut_pool_.add_cut(cut_type_t::FLOW_COVER, cut); }
    }
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class flow_cover_generation_t<int, double>;

template void cut_generation_t<int, double>::generate_flow_cover_cuts(
  const lp_problem_t<int, double>& lp,
  const simplex_solver_settings_t<int, double>& settings,
  csr_matrix_t<int, double>& Arow,
  const std::vector<variable_type_t>& var_types,
  const std::vector<double>& xstar,
  variable_bounds_t<int, double>& variable_bounds,
  double start_time);
#endif

}  // namespace cuopt::mathematical_optimization::mip
