/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuts/cuts.hpp>

#include <dual_simplex/basis_solves.hpp>
#include <dual_simplex/tic_toc.hpp>

#include <barrier/dense_matrix.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
void cut_pool_t<i_t, f_t>::add_cut(cut_type_t cut_type,
                                   const sparse_vector_t<i_t, f_t>& cut,
                                   f_t rhs)
{
  // TODO: Need to deduplicate cuts and only add if the cut is not already in the pool

  for (i_t p = 0; p < cut.i.size(); p++) {
    const i_t j = cut.i[p];
    if (j >= original_vars_) {
      settings_.log.printf(
        "Cut has variable %d that is greater than original_vars_ %d\n", j, original_vars_);
      return;
    }
  }

  sparse_vector_t<i_t, f_t> cut_squeezed;
  cut.squeeze(cut_squeezed);
  if (cut_squeezed.i.size() == 0) {
    settings_.log.printf("Cut has no coefficients\n");
    return;
  }
  cut_storage_.append_row(cut_squeezed);
  rhs_storage_.push_back(rhs);
  cut_type_.push_back(cut_type);
  cut_age_.push_back(0);
}

template <typename i_t, typename f_t>
f_t cut_pool_t<i_t, f_t>::cut_distance(i_t row,
                                       const std::vector<f_t>& x,
                                       f_t& cut_violation,
                                       f_t& cut_norm)
{
  const i_t row_start = cut_storage_.row_start[row];
  const i_t row_end   = cut_storage_.row_start[row + 1];
  f_t cut_x           = 0.0;
  f_t dot             = 0.0;
  for (i_t p = row_start; p < row_end; p++) {
    const i_t j         = cut_storage_.j[p];
    const f_t cut_coeff = cut_storage_.x[p];
    cut_x += cut_coeff * x[j];
    dot += cut_coeff * cut_coeff;
  }
  cut_violation      = rhs_storage_[row] - cut_x;
  cut_norm           = std::sqrt(dot);
  const f_t distance = cut_violation / cut_norm;
  return distance;
}

template <typename i_t, typename f_t>
f_t cut_pool_t<i_t, f_t>::cut_density(i_t row)
{
  const i_t row_start     = cut_storage_.row_start[row];
  const i_t row_end       = cut_storage_.row_start[row + 1];
  const i_t cut_nz        = row_end - row_start;
  const i_t original_vars = original_vars_;
  return static_cast<f_t>(cut_nz) / original_vars;
}

template <typename i_t, typename f_t>
f_t cut_pool_t<i_t, f_t>::cut_orthogonality(i_t i, i_t j)
{
  const i_t i_start = cut_storage_.row_start[i];
  const i_t i_end   = cut_storage_.row_start[i + 1];
  const i_t i_nz    = i_end - i_start;
  const i_t j_start = cut_storage_.row_start[j];
  const i_t j_end   = cut_storage_.row_start[j + 1];
  const i_t j_nz    = j_end - j_start;

  f_t dot = sparse_dot(cut_storage_.j.data() + i_start,
                       cut_storage_.x.data() + i_start,
                       i_nz,
                       cut_storage_.j.data() + j_start,
                       cut_storage_.x.data() + j_start,
                       j_nz);

  f_t norm_i = cut_norms_[i];
  f_t norm_j = cut_norms_[j];
  return 1.0 - std::abs(dot) / (norm_i * norm_j);
}

template <typename i_t, typename f_t>
void cut_pool_t<i_t, f_t>::score_cuts(std::vector<f_t>& x_relax)
{
  cut_distances_.resize(cut_storage_.m, 0.0);
  cut_norms_.resize(cut_storage_.m, 0.0);

  const bool verbose = false;
  for (i_t i = 0; i < cut_storage_.m; i++) {
    f_t violation;
    f_t cut_dist      = cut_distance(i, x_relax, violation, cut_norms_[i]);
    cut_distances_[i] = cut_dist <= min_cut_distance_ ? 0.0 : cut_dist;
    if (verbose) {
      settings_.log.printf("Cut %d type %d distance %+e violation %+e cut_norm %e\n",
                           i,
                           static_cast<int>(cut_type_[i]),
                           cut_distances_[i],
                           violation,
                           cut_norms_[i]);
    }
  }

  std::vector<i_t> sorted_indices;
  best_score_last_permutation(cut_distances_, sorted_indices);

  const i_t max_cuts          = 2000;
  const f_t min_orthogonality = settings_.cut_min_orthogonality;
  best_cuts_.reserve(std::min(max_cuts, cut_storage_.m));
  best_cuts_.clear();
  scored_cuts_ = 0;

  if (!sorted_indices.empty()) {
    const i_t i = sorted_indices.back();
    sorted_indices.pop_back();
    best_cuts_.push_back(i);
    scored_cuts_++;
  }

  while (scored_cuts_ < max_cuts && !sorted_indices.empty()) {
    const i_t i = sorted_indices.back();
    sorted_indices.pop_back();

    if (cut_distances_[i] <= min_cut_distance_) { break; }

    f_t cut_ortho            = 1.0;
    const i_t best_cuts_size = best_cuts_.size();
    for (i_t k = 0; k < best_cuts_size; k++) {
      const i_t j = best_cuts_[k];
      cut_ortho   = std::min(cut_ortho, cut_orthogonality(i, j));
    }
    if (cut_ortho >= min_orthogonality) {
      best_cuts_.push_back(i);
      scored_cuts_++;
    }
  }
}

template <typename i_t, typename f_t>
i_t cut_pool_t<i_t, f_t>::get_best_cuts(csr_matrix_t<i_t, f_t>& best_cuts,
                                        std::vector<f_t>& best_rhs,
                                        std::vector<cut_type_t>& best_cut_types)
{
  best_cuts.m = 0;
  best_cuts.n = original_vars_;
  best_cuts.row_start.clear();
  best_cuts.j.clear();
  best_cuts.x.clear();
  best_cuts.row_start.reserve(scored_cuts_ + 1);
  best_cuts.row_start.push_back(0);
  best_rhs.clear();
  best_rhs.reserve(scored_cuts_);
  best_cut_types.clear();
  best_cut_types.reserve(scored_cuts_);

  for (i_t i : best_cuts_) {
    if (cut_distances_[i] <= min_cut_distance_) { continue; }
    sparse_vector_t<i_t, f_t> cut(cut_storage_, i);
    cut.negate();
    best_cuts.append_row(cut);
    best_rhs.push_back(-rhs_storage_[i]);
    best_cut_types.push_back(cut_type_[i]);
  }

  age_cuts();

  return static_cast<i_t>(best_rhs.size());
}

template <typename i_t, typename f_t>
void cut_pool_t<i_t, f_t>::age_cuts()
{
  for (i_t i = 0; i < cut_age_.size(); i++) {
    cut_age_[i]++;
  }
}

template <typename i_t, typename f_t>
void cut_pool_t<i_t, f_t>::drop_cuts()
{
  // TODO: Implement this
}

template <typename i_t, typename f_t>
knapsack_generation_t<i_t, f_t>::knapsack_generation_t(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types)
  : settings_(settings)
{
  const bool verbose = false;
  knapsack_constraints_.reserve(lp.num_rows);

  is_slack_.resize(lp.num_cols, 0);
  for (i_t j : new_slacks) {
    is_slack_[j] = 1;
  }

  for (i_t i = 0; i < lp.num_rows; i++) {
    const i_t row_start = Arow.row_start[i];
    const i_t row_end   = Arow.row_start[i + 1];
    const i_t row_len   = row_end - row_start;
    if (row_len < 3) { continue; }
    bool is_knapsack = true;
    f_t sum_pos      = 0.0;
    for (i_t p = row_start; p < row_end; p++) {
      const i_t j = Arow.j[p];
      if (is_slack_[j]) { continue; }
      const f_t aj = Arow.x[p];
      if (std::abs(aj - std::round(aj)) > settings.integer_tol) {
        is_knapsack = false;
        break;
      }
      if (var_types[j] != variable_type_t::INTEGER || lp.lower[j] != 0.0 || lp.upper[j] != 1.0) {
        is_knapsack = false;
        break;
      }
      if (aj < 0.0) {
        is_knapsack = false;
        break;
      }
      sum_pos += aj;
    }

    if (is_knapsack) {
      const f_t beta = lp.rhs[i];
      if (std::abs(beta - std::round(beta)) <= settings.integer_tol) {
        if (beta > 0.0 && beta <= sum_pos && std::abs(sum_pos / (row_len - 1) - beta) > 1e-3) {
          if (verbose) {
            settings.log.printf(
              "Knapsack constraint %d row len %d beta %e sum_pos %e sum_pos / (row_len - 1) %e\n",
              i,
              row_len,
              beta,
              sum_pos,
              sum_pos / (row_len - 1));
          }
          knapsack_constraints_.push_back(i);
        }
      }
    }
  }

#ifdef PRINT_KNAPSACK_INFO
  i_t num_knapsack_constraints = knapsack_constraints_.size();
  settings.log.printf("Number of knapsack constraints %d\n", num_knapsack_constraints);
#endif
}

template <typename i_t, typename f_t>
i_t knapsack_generation_t<i_t, f_t>::generate_knapsack_cuts(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  i_t knapsack_row,
  sparse_vector_t<i_t, f_t>& cut,
  f_t& cut_rhs)
{
  const bool verbose = false;
  // Get the row associated with the knapsack constraint
  sparse_vector_t<i_t, f_t> knapsack_inequality(Arow, knapsack_row);
  f_t knapsack_rhs = lp.rhs[knapsack_row];

  // Remove the slacks from the inequality
  f_t seperation_rhs = 0.0;
  if (verbose) { settings.log.printf(" Knapsack : "); }
  for (i_t k = 0; k < knapsack_inequality.i.size(); k++) {
    const i_t j = knapsack_inequality.i[k];
    if (is_slack_[j]) {
      knapsack_inequality.x[k] = 0.0;
    } else {
      if (verbose) { settings.log.printf(" %g x%d +", knapsack_inequality.x[k], j); }
      seperation_rhs += knapsack_inequality.x[k];
    }
  }
  if (verbose) { settings.log.printf(" <= %g\n", knapsack_rhs); }
  seperation_rhs -= (knapsack_rhs + 1);

  if (verbose) {
    settings.log.printf("\t");
    for (i_t k = 0; k < knapsack_inequality.i.size(); k++) {
      const i_t j = knapsack_inequality.i[k];
      if (!is_slack_[j]) {
        if (std::abs(xstar[j]) > 1e-3) { settings.log.printf("x_relax[%d]= %g ", j, xstar[j]); }
      }
    }
    settings.log.printf("\n");

    settings.log.printf("seperation_rhs %g\n", seperation_rhs);
  }
  if (seperation_rhs <= 0.0) { return -1; }

  std::vector<f_t> values;
  values.resize(knapsack_inequality.i.size() - 1);
  std::vector<f_t> weights;
  weights.resize(knapsack_inequality.i.size() - 1);
  i_t h                  = 0;
  f_t objective_constant = 0.0;
  for (i_t k = 0; k < knapsack_inequality.i.size(); k++) {
    const i_t j = knapsack_inequality.i[k];
    if (!is_slack_[j]) {
      const f_t vj = std::min(1.0, std::max(0.0, 1.0 - xstar[j]));
      objective_constant += vj;
      values[h]  = vj;
      weights[h] = knapsack_inequality.x[k];
      h++;
    }
  }
  std::vector<f_t> solution;
  solution.resize(knapsack_inequality.i.size() - 1);

  if (verbose) { settings.log.printf("Calling solve_knapsack_problem\n"); }
  f_t objective = solve_knapsack_problem(values, weights, seperation_rhs, solution);
  if (std::isnan(objective)) { return -1; }
  if (verbose) {
    settings.log.printf("objective %e objective_constant %e\n", objective, objective_constant);
  }
  f_t seperation_value = -objective + objective_constant;
  if (verbose) { settings.log.printf("seperation_value %e\n", seperation_value); }
  const f_t tol = 1e-6;
  if (seperation_value >= 1.0 - tol) { return -1; }

  i_t cover_size = 0;
  for (i_t k = 0; k < solution.size(); k++) {
    if (solution[k] == 0.0) { cover_size++; }
  }

  cut.i.clear();
  cut.x.clear();
  cut.i.reserve(cover_size);
  cut.x.reserve(cover_size);

  h = 0;
  for (i_t k = 0; k < knapsack_inequality.i.size(); k++) {
    const i_t j = knapsack_inequality.i[k];
    if (!is_slack_[j]) {
      if (solution[h] == 0.0) {
        cut.i.push_back(j);
        cut.x.push_back(-1.0);
      }
      h++;
    }
  }
  cut_rhs = -cover_size + 1;
  cut.sort();

  // The cut is in the form: - sum_{j in cover} x_j >= -cover_size + 1
  // Which is equivalent to: sum_{j in cover} x_j <= cover_size - 1

  // Verify the cut is violated
  f_t dot       = cut.dot(xstar);
  f_t violation = dot - cut_rhs;
  if (verbose) {
    settings.log.printf("Knapsack cut %d violation %e < 0\n", knapsack_row, violation);
  }

  if (violation >= -tol) { return -1; }

#ifdef PRINT_KNAPSACK_CUT
  settings.log.printf("knapsack cut (cover %d): \n", cover_size);
  for (i_t k = 0; k < cut.i.size(); k++) {
    settings.log.printf("x%d coeff %g value %g\n", cut.i[k], -cut.x[k], xstar[cut.i[k]]);
  }
  settings.log.printf("cut_rhs %g\n", -cut_rhs);
#endif
  return 0;
}

template <typename i_t, typename f_t>
f_t knapsack_generation_t<i_t, f_t>::greedy_knapsack_problem(const std::vector<f_t>& values,
                                                             const std::vector<f_t>& weights,
                                                             f_t rhs,
                                                             std::vector<f_t>& solution)
{
  i_t n = weights.size();
  solution.assign(n, 0.0);

  // Build permutation
  std::vector<i_t> perm(n);
  std::iota(perm.begin(), perm.end(), 0);

  std::vector<f_t> ratios;
  ratios.resize(n);
  for (i_t i = 0; i < n; i++) {
    ratios[i] = values[i] / weights[i];
  }

  // Sort by value / weight ratio
  std::sort(perm.begin(), perm.end(), [&](i_t i, i_t j) { return ratios[i] > ratios[j]; });

  // Greedy select items with the best value / weight ratio until the remaining capacity is
  // exhausted
  f_t remaining   = rhs;
  f_t total_value = 0.0;

  for (i_t j : perm) {
    if (weights[j] <= remaining) {
      solution[j] = 1.0;
      remaining -= weights[j];
      total_value += values[j];
    }
  }

  // Best single-item fallback
  f_t best_single_value = 0.0;
  i_t best_single_idx   = -1;

  for (i_t j = 0; j < n; ++j) {
    if (weights[j] <= rhs && values[j] > best_single_value) {
      best_single_value = values[j];
      best_single_idx   = j;
    }
  }

  if (best_single_value > total_value) {
    solution.assign(n, 0.0);
    solution[best_single_idx] = 1.0;
    return best_single_value;
  }

  return total_value;
}

template <typename i_t, typename f_t>
f_t knapsack_generation_t<i_t, f_t>::solve_knapsack_problem(const std::vector<f_t>& values,
                                                            const std::vector<f_t>& weights,
                                                            f_t rhs,
                                                            std::vector<f_t>& solution)
{
  // Solve the knapsack problem
  // maximize sum_{j=0}^n values[j] * solution[j]
  // subject to sum_{j=0}^n weights[j] * solution[j] <= rhs
  // values: values of the items
  // weights: weights of the items
  // return the value of the solution

  // Using approximate dynamic programming

  i_t n         = weights.size();
  f_t objective = std::numeric_limits<f_t>::quiet_NaN();

  // Compute the maximum value
  f_t vmax = *std::max_element(values.begin(), values.end());

  // Check if all the values are integers
  bool all_integers     = true;
  const f_t integer_tol = 1e-5;
  for (i_t j = 0; j < n; j++) {
    if (std::abs(values[j] - std::round(values[j])) > integer_tol) {
      all_integers = false;
      break;
    }
  }

  const bool verbose = false;

  if (verbose) { settings_.log.printf("all_integers %d\n", all_integers); }

  // Compute the scaling factor and comptue the scaled integer values
  f_t scale = 1.0;
  std::vector<i_t> scaled_values(n);
  if (all_integers) {
    for (i_t j = 0; j < n; j++) {
      scaled_values[j] = static_cast<i_t>(std::floor(values[j]));
    }
  } else {
    const f_t epsilon = 0.1;
    scale             = epsilon * vmax / static_cast<f_t>(n);
    if (scale <= 0.0) { return std::numeric_limits<f_t>::quiet_NaN(); }
    if (verbose) {
      settings_.log.printf("scale %g epsilon %g vmax %g n %d\n", scale, epsilon, vmax, n);
    }
    for (i_t i = 0; i < n; ++i) {
      scaled_values[i] = static_cast<i_t>(std::floor(values[i] / scale));
    }
  }

  i_t sum_value     = std::accumulate(scaled_values.begin(), scaled_values.end(), 0);
  const i_t INT_INF = std::numeric_limits<i_t>::max() / 2;
  if (verbose) { settings_.log.printf("sum value %d\n", sum_value); }
  const i_t max_size = 10000;
  if (sum_value <= 0.0 || sum_value >= max_size) {
    if (verbose) {
      settings_.log.printf("sum value %d is negative or too large using greedy solution\n",
                           sum_value);
    }
    return greedy_knapsack_problem(values, weights, rhs, solution);
  }

  // dp(j, v) = minimum weight using first j items to get value v
  dense_matrix_t<i_t, i_t> dp(n + 1, sum_value + 1, INT_INF);
  dense_matrix_t<i_t, uint8_t> take(n + 1, sum_value + 1, 0);
  dp(0, 0) = 0;

  // 4. Dynamic programming
  for (i_t j = 1; j <= n; ++j) {
    for (i_t v = 0; v <= sum_value; ++v) {
      // Do not take item i-1
      dp(j, v) = dp(j - 1, v);

      // Take item j-1 if possible
      if (v >= scaled_values[j - 1]) {
        i_t candidate =
          dp(j - 1, v - scaled_values[j - 1]) + static_cast<i_t>(std::floor(weights[j - 1]));
        if (candidate < dp(j, v)) {
          dp(j, v)   = candidate;
          take(j, v) = 1;
        }
      }
    }
  }

  // 5. Find best achievable value within capacity
  i_t best_value = 0;
  for (i_t v = 0; v <= sum_value; ++v) {
    if (dp(n, v) <= rhs) { best_value = v; }
  }

  // 6. Backtrack to recover solution
  i_t v = best_value;
  for (i_t j = n; j >= 1; --j) {
    if (take(j, v)) {
      solution[j - 1] = 1.0;
      v -= scaled_values[j - 1];
    } else {
      solution[j - 1] = 0.0;
    }
  }

  objective = best_value * scale;
  return objective;
}

template <typename i_t, typename f_t>
void cut_generation_t<i_t, f_t>::generate_cuts(const lp_problem_t<i_t, f_t>& lp,
                                               const simplex_solver_settings_t<i_t, f_t>& settings,
                                               csr_matrix_t<i_t, f_t>& Arow,
                                               const std::vector<i_t>& new_slacks,
                                               const std::vector<variable_type_t>& var_types,
                                               basis_update_mpf_t<i_t, f_t>& basis_update,
                                               const std::vector<f_t>& xstar,
                                               const std::vector<f_t>& ystar,
                                               const std::vector<i_t>& basic_list,
                                               const std::vector<i_t>& nonbasic_list,
                                               variable_bounds_t<i_t, f_t>& variable_bounds)
{
  // Generate Gomory and CG Cuts
  if (settings.mixed_integer_gomory_cuts != 0 || settings.strong_chvatal_gomory_cuts != 0) {
    f_t cut_start_time = tic();
    generate_gomory_cuts(
      lp, settings, Arow, new_slacks, var_types, basis_update, xstar, basic_list, nonbasic_list);
    f_t cut_generation_time = toc(cut_start_time);
    if (cut_generation_time > 1.0) {
      settings.log.debug("Gomory and CG cut generation time %.2f seconds\n", cut_generation_time);
    }
  }

  // Generate Knapsack cuts
  if (settings.knapsack_cuts != 0) {
    f_t cut_start_time = tic();
    generate_knapsack_cuts(lp, settings, Arow, new_slacks, var_types, xstar);
    f_t cut_generation_time = toc(cut_start_time);
    if (cut_generation_time > 1.0) {
      settings.log.debug("Knapsack cut generation time %.2f seconds\n", cut_generation_time);
    }
  }

  // Generate MIR and CG cuts
  if (settings.mir_cuts != 0 || settings.strong_chvatal_gomory_cuts != 0) {
    f_t cut_start_time = tic();
    generate_mir_cuts(lp, settings, Arow, new_slacks, var_types, xstar, ystar, variable_bounds);
    f_t cut_generation_time = toc(cut_start_time);
    if (cut_generation_time > 1.0) {
      settings.log.debug("MIR and CG cut generation time %.2f seconds\n", cut_generation_time);
    }
  }
}

template <typename i_t, typename f_t>
void cut_generation_t<i_t, f_t>::generate_knapsack_cuts(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar)
{
  if (knapsack_generation_.num_knapsack_constraints() > 0) {
    for (i_t knapsack_row : knapsack_generation_.get_knapsack_constraints()) {
      sparse_vector_t<i_t, f_t> cut(lp.num_cols, 0);
      f_t cut_rhs;
      i_t knapsack_status = knapsack_generation_.generate_knapsack_cuts(
        lp, settings, Arow, new_slacks, var_types, xstar, knapsack_row, cut, cut_rhs);
      if (knapsack_status == 0) { cut_pool_.add_cut(cut_type_t::KNAPSACK, cut, cut_rhs); }
    }
  }
}

template <typename i_t, typename f_t>
void cut_generation_t<i_t, f_t>::generate_mir_cuts(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  const std::vector<f_t>& ystar,
  variable_bounds_t<i_t, f_t>& variable_bounds)
{
  f_t mir_start_time     = tic();
  constexpr bool verbose = false;
  complemented_mixed_integer_rounding_cut_t<i_t, f_t> complemented_mir(lp, settings, new_slacks);
  strong_cg_cut_t<i_t, f_t> cg(lp, var_types, xstar);

  std::vector<i_t> slack_map(lp.num_rows, -1);
  for (i_t slack : new_slacks) {
    const i_t col_start = lp.A.col_start[slack];
    const i_t col_end   = lp.A.col_start[slack + 1];
    const i_t col_len   = col_end - col_start;
    assert(col_len == 1);
    const i_t i  = lp.A.i[col_start];
    slack_map[i] = slack;
  }
  const i_t n         = lp.num_cols;
  const f_t obj_norm  = vector_norm2<i_t, f_t>(lp.objective);
  const f_t obj_denom = std::max(1.0, obj_norm);

  // Compute initial scores for all rows
  std::vector<f_t> score(lp.num_rows, 0.0);
  for (i_t i = 0; i < lp.num_rows; i++) {
    const i_t row_start = Arow.row_start[i];
    const i_t row_end   = Arow.row_start[i + 1];

    const i_t row_nz = row_end - row_start;
    f_t row_norm     = 0.0;
    for (i_t p = row_start; p < row_end; p++) {
      const f_t a_j = Arow.x[p];
      row_norm += a_j * a_j;
    }
    row_norm = std::sqrt(row_norm);

    const f_t density = static_cast<f_t>(row_nz) / static_cast<f_t>(n);
    const f_t dual    = std::abs(ystar[i]);

    const i_t slack = slack_map[i];
    assert(slack >= 0);
    const f_t slack_value = xstar[slack];
    const f_t slack_denom = std::max(0.1, std::sqrt(row_norm));

    const f_t nz_weight    = 0.0001;
    const f_t dual_weight  = 1.0;
    const f_t slack_weight = 0.001;

    score[i] = nz_weight * (1.0 - density) + dual_weight * std::max(dual / obj_denom, 0.0001) +
               slack_weight * (1.0 - slack_value / slack_denom);

    if (verbose) {
      settings.log.printf(
        "Score[%d] = %e density %.2f dual %e slack %e\n", i, score[i], density, dual, slack_value);
    }
  }

  // Sort the rows by score
  std::vector<i_t> sorted_indices;
  best_score_last_permutation(score, sorted_indices);

  // These data structures are used to track the rows that have been aggregated
  // The invariant is that aggregated_rows is empty and aggregated_mark is all zeros
  // at the beginning of each iteration of the for loop below
  std::vector<i_t> aggregated_rows;
  std::vector<i_t> aggregated_mark(lp.num_rows, 0);

  // Transform the relaxation solution
  std::vector<f_t> transformed_xstar;
  complemented_mir.bound_substitution(lp, variable_bounds, var_types, xstar, transformed_xstar);

  const i_t max_cuts = std::min(lp.num_rows, 100000);
  f_t work_estimate  = 0.0;
  for (i_t h = 0; h < max_cuts; h++) {
    // Get the row with the highest score
    const i_t i = sorted_indices.back();
    sorted_indices.pop_back();
    const f_t max_score = score[i];

    const i_t row_nz      = Arow.row_start[i + 1] - Arow.row_start[i];
    const i_t slack       = slack_map[i];
    const f_t slack_value = xstar[slack];

    if (max_score <= 0.0) { break; }
    if (work_estimate > 2e9) { break; }

    sparse_vector_t<i_t, f_t> inequality(Arow, i);
    work_estimate += inequality.i.size();

    f_t inequality_rhs         = lp.rhs[i];
    const bool generate_cg_cut = settings.strong_chvatal_gomory_cuts != 0;
    f_t fractional_part_rhs    = fractional_part(inequality_rhs);
    if (generate_cg_cut && fractional_part_rhs > 1e-6 && fractional_part_rhs < (1 - 1e-6)) {
      // Try to generate a CG cut
      sparse_vector_t<i_t, f_t> cg_inequality = inequality;
      f_t cg_inequality_rhs                   = inequality_rhs;
      if (fractional_part(inequality_rhs) < 0.5) {
        // Multiply by -1 to force the fractional part to be greater than 0.5
        cg_inequality_rhs *= -1;
        cg_inequality.negate();
      }
      sparse_vector_t<i_t, f_t> cg_cut(lp.num_cols, 0);
      f_t cg_cut_rhs;
      i_t cg_status = cg.generate_strong_cg_cut(
        lp, settings, var_types, cg_inequality, cg_inequality_rhs, xstar, cg_cut, cg_cut_rhs);
      if (cg_status == 0) { cut_pool_.add_cut(cut_type_t::CHVATAL_GOMORY, cg_cut, cg_cut_rhs); }
    }

    // Remove the slack from the equality to get an inequality
    work_estimate += inequality.i.size();
    i_t negate_inequality = 1;
    for (i_t k = 0; k < inequality.i.size(); k++) {
      const i_t j = inequality.i[k];
      if (j == slack) {
        if (inequality.x[k] != 1.0) {
          if (inequality.x[k] == -1.0 && lp.lower[j] >= 0.0) {
            negate_inequality = 0;
          } else {
            settings.log.debug("Bad slack %d in inequality: aj %e lo %e up %e\n",
                               j,
                               inequality.x[k],
                               lp.lower[j],
                               lp.upper[j]);
            negate_inequality = -1;
            break;
          }
        }
        inequality.x[k] = 0.0;
      }
    }

    if (negate_inequality == -1) { continue; }

    if (negate_inequality) {
      // inequaility'*x <= inequality_rhs
      // But for MIR we need: inequality'*x >= inequality_rhs
      inequality_rhs *= -1;
      inequality.negate();
      work_estimate += inequality.i.size();
    }
    // We should now have: inequality'*x >= inequality_rhs

    for (i_t k = 0; k < inequality.i.size(); k++) {
      const i_t j = inequality.i[k];
      if (var_types[j] == variable_type_t::INTEGER) {
        if (transformed_xstar[j] > complemented_mir.new_upper(j) / 2.0) {
          settings.log.printf("!!!!!! j %d transformed x_j %e new_upper_j/2.0 %e\n",
                              j,
                              transformed_xstar[j],
                              complemented_mir.new_upper(j) / 2.0);
        }
      }
    }

    sparse_vector_t<i_t, f_t> cut(lp.num_cols, 0);
    f_t cut_rhs;
    bool add_cut             = false;
    i_t num_aggregated       = 0;
    const i_t max_aggregated = 6;
    work_estimate += lp.num_cols;

    while (!add_cut && num_aggregated < max_aggregated) {
      sparse_vector_t<i_t, f_t> transformed_inequality;
      inequality.squeeze(transformed_inequality);
      f_t transformed_rhs = inequality_rhs;
      work_estimate += transformed_inequality.i.size();

      complemented_mir.transform_inequality(
        variable_bounds, var_types, transformed_inequality, transformed_rhs);
      work_estimate += transformed_inequality.i.size();
      std::vector<sparse_vector_t<i_t, f_t>> transformed_cuts;
      std::vector<f_t> transformed_cut_rhs;
      std::vector<f_t> transformed_violations;

      //  Generate cut for delta = 1
      {
        sparse_vector_t<i_t, f_t> cut_1(lp.num_cols, 0);
        f_t cut_1_rhs;
        complemented_mir.generate_cut_nonnegative_maintain_indicies(
          transformed_inequality, transformed_rhs, var_types, cut_1, cut_1_rhs);
        f_t cut_1_violation =
          complemented_mir.compute_violation(cut_1, cut_1_rhs, transformed_xstar);
        if (cut_1_violation > 1e-6) {
          transformed_cuts.push_back(cut_1);
          transformed_cut_rhs.push_back(cut_1_rhs);
          transformed_violations.push_back(cut_1_violation);
        }
        work_estimate += transformed_inequality.i.size();
      }

      // Generate a cut for delta = max { |a_j|, j in I}
      {
        f_t max_coeff = 0.0;
        for (i_t k = 0; k < transformed_inequality.i.size(); k++) {
          const i_t j = transformed_inequality.i[k];
          if (var_types[j] == variable_type_t::INTEGER) {
            const f_t abs_aj = std::abs(transformed_inequality.x[k]);
            if (abs_aj > max_coeff) { max_coeff = abs_aj; }
          }
        }
        work_estimate += transformed_inequality.i.size();

        if (max_coeff > 1e-6 && max_coeff != 1.0) {
          sparse_vector_t<i_t, f_t> scaled_inequality = transformed_inequality;
          const i_t nz                                = transformed_inequality.i.size();
          for (i_t k = 0; k < nz; k++) {
            scaled_inequality.x[k] /= max_coeff;
          }
          const f_t scaled_rhs = transformed_rhs / max_coeff;
          sparse_vector_t<i_t, f_t> cut_2(lp.num_cols, 0);
          f_t cut_2_rhs;
          complemented_mir.generate_cut_nonnegative_maintain_indicies(
            scaled_inequality, scaled_rhs, var_types, cut_2, cut_2_rhs);
          f_t cut_2_violation =
            complemented_mir.compute_violation(cut_2, cut_2_rhs, transformed_xstar);
          if (cut_2_violation > 1e-6) {
            transformed_cuts.push_back(cut_2);
            transformed_cut_rhs.push_back(cut_2_rhs);
            transformed_violations.push_back(cut_2_violation);
          }
          work_estimate += 5 * transformed_inequality.i.size();
        }
      }

      // Generate cuts for delta in { |a_j| : j in I, 0 < x_j < new_upper_j }
      // and by successively complementing bounded integer variables
      {
        std::vector<f_t> deltas_to_try;
        deltas_to_try.reserve(transformed_inequality.i.size());
        work_estimate += transformed_inequality.i.size();
        i_t num_integers = 0;
        for (i_t k = 0; k < transformed_inequality.i.size(); k++) {
          const i_t j = transformed_inequality.i[k];
          if (var_types[j] == variable_type_t::INTEGER) {
            num_integers++;
            const f_t x_j             = transformed_xstar[j];
            const f_t new_upper_j     = complemented_mir.new_upper(j);
            const f_t dist_upper      = new_upper_j - x_j;
            const f_t dist_lower      = x_j;
            const bool between_bounds = x_j > 0.0 && (new_upper_j == inf || dist_upper > 0.0);
            if (between_bounds) {
              const f_t delta = std::abs(transformed_inequality.x[k]);
              if (delta == 0.0) {
                printf("delta is 0.0 for j %d k %d\n", j, k);
                exit(1);
              }
              deltas_to_try.push_back(delta);
            }
          }
        }
        work_estimate +=
          2 * transformed_inequality.i.size() + 2 * num_integers + deltas_to_try.size();

        bool found_cut = false;
        for (const f_t delta : deltas_to_try) {
          sparse_vector_t<i_t, f_t> scaled_inequality = transformed_inequality;
          f_t scaled_rhs                              = transformed_rhs / delta;
          scaled_inequality.scale(1.0 / delta);
          sparse_vector_t<i_t, f_t> cut_delta(lp.num_cols, 0);
          f_t cut_delta_rhs;
          complemented_mir.generate_cut_nonnegative_maintain_indicies(
            scaled_inequality, scaled_rhs, var_types, cut_delta, cut_delta_rhs);
          f_t cut_delta_violation =
            complemented_mir.compute_violation(cut_delta, cut_delta_rhs, transformed_xstar);
          if (cut_delta_violation > 1e-6) {
            transformed_cuts.push_back(cut_delta);
            transformed_cut_rhs.push_back(cut_delta_rhs);
            transformed_violations.push_back(cut_delta_violation);
            found_cut = true;
          }
          work_estimate += 5 * transformed_inequality.i.size();

          if (found_cut) { break; }
        }

        if (!found_cut) {
          std::vector<f_t> distance_from_midpoint;
          std::vector<i_t> integer_indices;
          distance_from_midpoint.reserve(num_integers);
          integer_indices.reserve(num_integers);

          for (i_t k = 0; k < transformed_inequality.i.size(); k++) {
            const i_t j = transformed_inequality.i[k];
            if (var_types[j] == variable_type_t::INTEGER && complemented_mir.new_upper(j) < inf) {
              const f_t x_j         = transformed_xstar[j];
              const f_t new_upper_j = complemented_mir.new_upper(j);
              if (x_j > 1e-6 && new_upper_j < inf) {
                const f_t midpoint_j = new_upper_j / 2.0;
                distance_from_midpoint.push_back(midpoint_j - x_j);
                integer_indices.push_back(k);
              }
            }
          }
          work_estimate += 3 * transformed_inequality.i.size() + 4 * distance_from_midpoint.size();

          std::vector<i_t> perm(integer_indices.size());
          best_score_first_permutation(distance_from_midpoint, perm);
          work_estimate += integer_indices.size() > 0
                             ? integer_indices.size() * std::log2(integer_indices.size())
                             : 0;

          std::vector<i_t> complemented_indices;
          complemented_indices.reserve(integer_indices.size());

          sparse_vector_t<i_t, f_t> complemented_inequality(lp.num_cols, 0);
          f_t complemented_inequality_rhs = transformed_rhs;
          complemented_inequality         = transformed_inequality;
          work_estimate += 4 * transformed_inequality.i.size();

          for (const i_t idx : perm) {
            const i_t k = integer_indices[idx];
            const i_t j = complemented_inequality.i[k];
            // We have an integer variable x_j <= b_j
            // We create a new varaible xbar_j such that
            // x_j + xbar_j = b_j
            // x_j = b_j - xbar_j, xbar_j = b_j - x_j
            //
            // The inequality
            // sum_{k != j} a_k x_k + a_j x_j >= beta
            // becomes
            // sum_{k != j} a_k x_k + a_j (b_j - xbar_j) >= beta
            // sum_{k != j} a_k x_k - a_j xbar_j >= beta - a_j b_j
            const f_t b_j                = complemented_mir.new_upper(j);
            const f_t a_j                = complemented_inequality.x[k];
            complemented_inequality.x[k] = -a_j;
            complemented_inequality_rhs -= a_j * b_j;
            complemented_indices.push_back(k);

            for (const f_t delta : deltas_to_try) {
              sparse_vector_t<i_t, f_t> scaled_inequality = complemented_inequality;
              f_t scaled_rhs                              = complemented_inequality_rhs / delta;
              scaled_inequality.scale(1.0 / delta);
              sparse_vector_t<i_t, f_t> cut_delta(lp.num_cols, 0);
              f_t cut_delta_rhs;
              complemented_mir.generate_cut_nonnegative_maintain_indicies(
                scaled_inequality, scaled_rhs, var_types, cut_delta, cut_delta_rhs);

              // Now we need to transform the complemented variables back
              for (i_t h = 0; h < complemented_indices.size(); h++) {
                const i_t l = complemented_indices[h];
                const i_t j = complemented_inequality.i[l];
                // Our cut is of the form
                // sum_{k != j} d_k x_k  + d_j xbar_j >= alpha
                // we have that xbar_j = b_j - x_j
                // So
                // sum_{k != j} d_k x_k  + d_j (b_j - x_j) >= alpha
                // Or
                // sum_{k != j} d_k x_k  - d_j x_j >= alpha - d_j b_j

                const f_t b_j  = complemented_mir.new_upper(j);
                const f_t d_j  = cut_delta.x[l];
                cut_delta.x[l] = -d_j;
                cut_delta_rhs -= d_j * b_j;
              }
              work_estimate += 5 * complemented_indices.size();

              f_t cut_delta_violation =
                complemented_mir.compute_violation(cut_delta, cut_delta_rhs, transformed_xstar);
              if (cut_delta_violation > 1e-6) {
                // settings.log.printf("Cut for delta %e has violation %e\n", delta,
                // cut_delta_violation);
                transformed_cuts.push_back(cut_delta);
                transformed_cut_rhs.push_back(cut_delta_rhs);
                transformed_violations.push_back(cut_delta_violation);
                found_cut = true;
              }
              work_estimate += 5 * complemented_inequality.i.size();

              if (found_cut) { break; }
            }

            if (found_cut) { break; }
          }
        }
      }

      if (!transformed_violations.empty()) {
        std::vector<i_t> permuted(transformed_violations.size());
        std::iota(permuted.begin(), permuted.end(), 0);
        std::sort(permuted.begin(), permuted.end(), [&](i_t i, i_t j) {
          return transformed_violations[i] > transformed_violations[j];
        });
        work_estimate += transformed_violations.size() * std::log2(transformed_violations.size());
        // Get the biggest violation
        const i_t best_index = permuted[0];
        f_t max_viol         = transformed_violations[best_index];
        cut                  = transformed_cuts[best_index];
        cut_rhs              = transformed_cut_rhs[best_index];

        if (max_viol > 1e-6) {
          // TODO: Divide by 1/2*violation, 1/4*violation, 1/8*violation
          // Transform back to the original variables
          complemented_mir.untransform_inequality(variable_bounds, var_types, cut, cut_rhs);
          complemented_mir.remove_small_coefficients(lp.lower, lp.upper, cut, cut_rhs);
          complemented_mir.substitute_slacks(lp, Arow, cut, cut_rhs);
          f_t viol = complemented_mir.compute_violation(cut, cut_rhs, xstar);
          work_estimate += 10 * cut.i.size();
          if (viol > 1e-6) { add_cut = true; }
        }
      }

      if (add_cut) {
        if (settings.mir_cuts != 0) {
          cut_pool_.add_cut(cut_type_t::MIXED_INTEGER_ROUNDING, cut, cut_rhs);
        }
        break;
      } else {
        // Perform aggregation to try and find a cut

        // Find all the continuous variables in the inequality
        i_t num_continuous    = 0;
        f_t max_off_bound     = 0.0;
        i_t max_off_bound_var = -1;
        for (i_t p = 0; p < inequality.i.size(); p++) {
          const i_t j = inequality.i[p];
          if (var_types[j] == variable_type_t::CONTINUOUS) {
            num_continuous++;

            const f_t off_lower = lp.lower[j] > -inf ? xstar[j] - lp.lower[j] : std::abs(xstar[j]);
            const f_t off_upper = lp.upper[j] < inf ? lp.upper[j] - xstar[j] : std::abs(xstar[j]);
            const f_t off_bound = std::max(off_lower, off_upper);
            const i_t col_start = lp.A.col_start[j];
            const i_t col_end   = lp.A.col_start[j + 1];
            const i_t col_len   = col_end - col_start;
            if (off_bound > max_off_bound && col_len > 1) {
              max_off_bound     = off_bound;
              max_off_bound_var = j;
            }
          }
        }
        work_estimate += 10 * inequality.i.size();

        if (num_continuous == 0 || max_off_bound < 1e-6) { break; }

        // The variable that is farthest from its bound is used as a pivot
        if (max_off_bound_var >= 0) {
          const i_t col_start          = lp.A.col_start[max_off_bound_var];
          const i_t col_end            = lp.A.col_start[max_off_bound_var + 1];
          const i_t col_len            = col_end - col_start;
          const i_t max_potential_rows = 10;
          if (col_len > 1) {
            std::vector<i_t> potential_rows;
            potential_rows.reserve(col_len);

            const f_t threshold = 1e-4;
            for (i_t q = col_start; q < col_end; q++) {
              const i_t i   = lp.A.i[q];
              const f_t val = lp.A.x[q];
              // Can't use rows that have already been aggregated
              if (std::abs(val) > threshold && aggregated_mark[i] == 0) {
                potential_rows.push_back(i);
              }
              if (potential_rows.size() >= max_potential_rows) { break; }
            }
            work_estimate += 5 * (col_end - col_start);

            if (!potential_rows.empty()) {
              std::sort(potential_rows.begin(), potential_rows.end(), [&](i_t a, i_t b) {
                return score[a] > score[b];
              });
              work_estimate += 10 * std::log2(10);

              const i_t pivot_row = potential_rows[0];

              sparse_vector_t<i_t, f_t> pivot_row_inequality(Arow, pivot_row);
              f_t pivot_row_rhs = lp.rhs[pivot_row];
              work_estimate += pivot_row_inequality.i.size();
              complemented_mir.combine_rows(lp,
                                            Arow,
                                            max_off_bound_var,
                                            pivot_row_inequality,
                                            pivot_row_rhs,
                                            inequality,
                                            inequality_rhs);
              aggregated_rows.push_back(pivot_row);
              aggregated_mark[pivot_row] = 1;
              work_estimate += inequality.i.size() + pivot_row_inequality.i.size();
            } else {
              // No potential rows to aggregate
              break;
            }
          }
        }
        num_aggregated++;  // Always increase so the loop terminates
      }
    }

    if (add_cut) {
      // We were successful in generating a cut.

      // Set the score of the aggregated rows to zero
      for (i_t row : aggregated_rows) {
        score[row] = 0.0;
      }
    }

    // Clear the aggregated mark
    for (i_t row : aggregated_rows) {
      aggregated_mark[row] = 0;
    }
    work_estimate += 2 * aggregated_rows.size();
    // Clear the aggregated rows
    aggregated_rows.clear();

    // Set the score of the current row to zero
    score[i] = 0.0;

    // Re-sort the rows by score
    // It's possible this could be made more efficient by storing the rows in a data structure
    // that allows us to:
    // 1. Get the row with the best score
    // 2. Get the row with a nonzero in column j that has the best score
    // 3. Remove the rows that have been aggregated
    // 4. Remove the current row
    best_score_last_permutation(score, sorted_indices);
    work_estimate += score.size() * std::log2(score.size());
  }
}

template <typename i_t, typename f_t>
void cut_generation_t<i_t, f_t>::generate_gomory_cuts(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  basis_update_mpf_t<i_t, f_t>& basis_update,
  const std::vector<f_t>& xstar,
  const std::vector<i_t>& basic_list,
  const std::vector<i_t>& nonbasic_list)
{
  tableau_equality_t<i_t, f_t> tableau(lp, basis_update, nonbasic_list);
  complemented_mixed_integer_rounding_cut_t<i_t, f_t> complemented_mir(lp, settings, new_slacks);
  simplex_solver_settings_t<i_t, f_t> variable_settings = settings;
  variable_settings.sub_mip                             = 1;
  variable_bounds_t<i_t, f_t> variable_bounds(lp, variable_settings, var_types, Arow, new_slacks);
  strong_cg_cut_t<i_t, f_t> cg(lp, var_types, xstar);
  std::vector<f_t> transformed_xstar;
  complemented_mir.bound_substitution(lp, variable_bounds, var_types, xstar, transformed_xstar);

  for (i_t i = 0; i < lp.num_rows; i++) {
    sparse_vector_t<i_t, f_t> inequality(lp.num_cols, 0);
    f_t inequality_rhs;
    const i_t j = basic_list[i];
    if (var_types[j] != variable_type_t::INTEGER) { continue; }
    const f_t x_j = xstar[j];
    if (std::abs(x_j - std::round(x_j)) < settings.integer_tol) { continue; }
    i_t tableau_status = tableau.generate_base_equality(lp,
                                                        settings,
                                                        Arow,
                                                        var_types,
                                                        basis_update,
                                                        xstar,
                                                        basic_list,
                                                        nonbasic_list,
                                                        i,
                                                        inequality,
                                                        inequality_rhs);
    if (tableau_status == 0) {
      // Generate a CG cut
      const bool generate_cg_cut = settings.strong_chvatal_gomory_cuts != 0;
      if (generate_cg_cut) {
        // Try to generate a CG cut
        sparse_vector_t<i_t, f_t> cg_inequality = inequality;
        f_t cg_inequality_rhs                   = inequality_rhs;
        if (fractional_part(inequality_rhs) < 0.5) {
          // Multiply by -1 to force the fractional part to be greater than 0.5
          cg_inequality_rhs *= -1;
          cg_inequality.negate();
        }
        sparse_vector_t<i_t, f_t> cg_cut(lp.num_cols, 0);
        f_t cg_cut_rhs;
        i_t cg_status = cg.generate_strong_cg_cut(
          lp, settings, var_types, cg_inequality, cg_inequality_rhs, xstar, cg_cut, cg_cut_rhs);
        if (cg_status == 0) { cut_pool_.add_cut(cut_type_t::CHVATAL_GOMORY, cg_cut, cg_cut_rhs); }
      }

      if (settings.mixed_integer_gomory_cuts == 0) { continue; }

      // Transform the inequality
      sparse_vector_t<i_t, f_t> transformed_inequality = inequality;
      f_t transformed_rhs                              = inequality_rhs;
      complemented_mir.transform_inequality(
        variable_bounds, var_types, transformed_inequality, transformed_rhs);

      // Generate a MIR cut from the transformed inequality
      sparse_vector_t<i_t, f_t> cut_A(lp.num_cols, 0);
      f_t cut_A_rhs;
      complemented_mir.generate_cut_nonnegative_maintain_indicies(
        transformed_inequality, transformed_rhs, var_types, cut_A, cut_A_rhs);

      // Transform the cut back to the original variables
      complemented_mir.untransform_inequality(variable_bounds, var_types, cut_A, cut_A_rhs);

      // See if the inequality is violated by the original relaxation solution
      f_t cut_A_violation = complemented_mir.compute_violation(cut_A, cut_A_rhs, xstar);
      bool A_valid        = false;
      f_t cut_A_distance  = 0.0;
      if (cut_A_violation > 1e-6) {
        if (cut_A.i.size() == 0) { continue; }
        complemented_mir.substitute_slacks(lp, Arow, cut_A, cut_A_rhs);
        if (cut_A.i.size() == 0) {
          A_valid = false;
        } else {
          // Check that the cut is violated
          f_t dot      = cut_A.dot(xstar);
          f_t cut_norm = cut_A.norm2_squared();
          if (dot >= cut_A_rhs) { continue; }
          cut_A_distance = (cut_A_rhs - dot) / std::sqrt(cut_norm);
          A_valid        = true;
        }
      }

      // Negate the base inequality
      inequality.negate();
      inequality_rhs *= -1;

      sparse_vector_t<i_t, f_t> cut_B(lp.num_cols, 0);
      f_t cut_B_rhs;

      transformed_inequality = inequality;
      transformed_rhs        = inequality_rhs;
      complemented_mir.transform_inequality(
        variable_bounds, var_types, transformed_inequality, transformed_rhs);

      complemented_mir.generate_cut_nonnegative_maintain_indicies(
        transformed_inequality, transformed_rhs, var_types, cut_B, cut_B_rhs);
      // Transform the cut back to the original variables
      complemented_mir.untransform_inequality(variable_bounds, var_types, cut_B, cut_B_rhs);

      bool B_valid        = false;
      f_t cut_B_distance  = 0.0;
      f_t cut_B_violation = complemented_mir.compute_violation(cut_B, cut_B_rhs, xstar);
      if (cut_B_violation > 1e-6) {
        if (cut_B.i.size() == 0) { continue; }
        complemented_mir.substitute_slacks(lp, Arow, cut_B, cut_B_rhs);
        if (cut_B.i.size() == 0) {
          B_valid = false;
        } else {
          // Check that the cut is violated
          f_t dot      = cut_B.dot(xstar);
          f_t cut_norm = cut_B.norm2_squared();
          if (dot >= cut_B_rhs) { continue; }
          cut_B_distance = (cut_B_rhs - dot) / std::sqrt(cut_norm);
          B_valid        = true;
        }
      }

      if ((cut_A_distance > cut_B_distance) && A_valid) {
        cut_pool_.add_cut(cut_type_t::MIXED_INTEGER_GOMORY, cut_A, cut_A_rhs);
      } else if (B_valid) {
        cut_pool_.add_cut(cut_type_t::MIXED_INTEGER_GOMORY, cut_B, cut_B_rhs);
      }
    }
  }
}

template <typename i_t, typename f_t>
i_t tableau_equality_t<i_t, f_t>::generate_base_equality(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<variable_type_t>& var_types,
  basis_update_mpf_t<i_t, f_t>& basis_update,
  const std::vector<f_t>& xstar,
  const std::vector<i_t>& basic_list,
  const std::vector<i_t>& nonbasic_list,
  i_t i,
  sparse_vector_t<i_t, f_t>& inequality,
  f_t& inequality_rhs)
{
  // Let's look for Gomory cuts
  const i_t j = basic_list[i];
  if (var_types[j] != variable_type_t::INTEGER) { return -1; }
  const f_t x_j = xstar[j];
  if (std::abs(x_j - std::round(x_j)) < settings.integer_tol) { return -1; }
#ifdef PRINT_CUT_INFO
  settings_.log.printf("Generating cut for variable %d relaxed value %e row %d\n", j, x_j, i);
#endif

  // Solve B^T u_bar = e_i
  sparse_vector_t<i_t, f_t> e_i(lp.num_rows, 1);
  e_i.i[0] = i;
  e_i.x[0] = 1.0;
  sparse_vector_t<i_t, f_t> u_bar(lp.num_rows, 0);
  basis_update.b_transpose_solve(e_i, u_bar);

#ifdef CHECK_B_TRANSPOSE_SOLVE
  std::vector<f_t> u_bar_dense(lp.num_rows);
  u_bar.to_dense(u_bar_dense);

  std::vector<f_t> BTu_bar(lp.num_rows);
  b_transpose_multiply(lp, basic_list, u_bar_dense, BTu_bar);
  for (i_t k = 0; k < lp.num_rows; k++) {
    if (k == i) {
      settings.log.printf("BTu_bar %d error %e\n", k, std::abs(BTu_bar[k] - 1.0));
      if (std::abs(BTu_bar[k] - 1.0) > 1e-10) {
        settings.log.printf("BTu_bar[%d] = %e i %d\n", k, BTu_bar[k], i);
        assert(false);
      }
    } else {
      settings.log.printf("BTu_bar %d error %e\n", k, std::abs(BTu_bar[k]));
      if (std::abs(BTu_bar[k]) > 1e-10) {
        settings.log.printf("BTu_bar[%d] = %e i %d\n", k, BTu_bar[k], i);
        assert(false);
      }
    }
  }
#endif

  // Compute a_bar = N^T u_bar
  // TODO: This is similar to a function in phase2 of dual simplex. See if it can be reused.
  const i_t nz_ubar = u_bar.i.size();
  std::vector<i_t> abar_indices;
  abar_indices.reserve(nz_ubar);
  for (i_t k = 0; k < nz_ubar; k++) {
    const i_t ii        = u_bar.i[k];
    const f_t u_bar_i   = u_bar.x[k];
    const i_t row_start = Arow.row_start[ii];
    const i_t row_end   = Arow.row_start[ii + 1];
    for (i_t p = row_start; p < row_end; p++) {
      const i_t jj = Arow.j[p];
      if (nonbasic_mark_[jj] == 1) {
        const f_t val    = u_bar_i * Arow.x[p];
        const f_t y      = val - c_workspace_[jj];
        const f_t t      = x_workspace_[jj] + y;
        c_workspace_[jj] = (t - x_workspace_[jj]) - y;
        x_workspace_[jj] = t;
        if (!x_mark_[jj]) {
          x_mark_[jj] = 1;
          abar_indices.push_back(jj);
        }
      }
    }
  }
  // TODO: abar has lots of small coefficients. Double check that
  // we do not accidently create a base (in)equality
  // that cuts off an integer solution, when we drop the small coefficients.

  i_t small_coeff              = 0;
  const f_t drop_tol           = 1e-12;
  const bool drop_coefficients = true;
  sparse_vector_t<i_t, f_t> a_bar(lp.num_cols, 0);
  a_bar.i.reserve(abar_indices.size() + 1);
  a_bar.x.reserve(abar_indices.size() + 1);
  for (i_t k = 0; k < abar_indices.size(); k++) {
    const i_t jj = abar_indices[k];
    if (drop_coefficients && std::abs(x_workspace_[jj]) < drop_tol) {
      small_coeff++;
    } else {
      a_bar.i.push_back(jj);
      a_bar.x.push_back(x_workspace_[jj]);
    }
  }
  const bool verbose = false;
  if (verbose && small_coeff > 0) { settings.log.printf("Small coeff dropped %d\n", small_coeff); }

  // Clear the workspace
  for (i_t jj : abar_indices) {
    x_workspace_[jj] = 0.0;
    x_mark_[jj]      = 0;
    c_workspace_[jj] = 0.0;
  }
  abar_indices.clear();

  // We should now have the base inequality
  // x_j + a_bar^T x_N >= b_bar_i
  // We add x_j into a_bar so that everything is in a single sparse_vector_t
  a_bar.i.push_back(j);
  a_bar.x.push_back(1.0);

  // Check that the tableau equality is satisfied
  const f_t tableau_tol = 1e-6;
  f_t a_bar_dot_xstar   = a_bar.dot(xstar);
  if (std::abs(a_bar_dot_xstar - b_bar_[i]) > tableau_tol) {
    settings.log.debug("bad tableau equality. error %e\n", std::abs(a_bar_dot_xstar - b_bar_[i]));
    return -1;
  }

  // We have that x_j + a_bar^T x_N == b_bar_i
  // So x_j + a_bar^T x_N >= b_bar_i
  // And x_j + a_bar^T x_N <= b_bar_i
  // Or -x_j - a_bar^T x_N >= -b_bar_i

  // Skip cuts that are shallow
  const f_t shallow_tol = 1e-2;
  if (std::abs(x_j - std::round(x_j)) < shallow_tol) {
    // Skip cuts where integer variable has small fractional part
    return -1;
  }

  const f_t f_val = b_bar_[i] - std::floor(b_bar_[i]);
  if (f_val < 0.01 || f_val > 0.99) {
    // Skip cuts with rhs has small fractional part
    return -1;
  }

#ifdef PRINT_BASE_INEQUALITY
  // Print out the base inequality
  for (i_t k = 0; k < a_bar.i.size(); k++) {
    const i_t jj = a_bar.i[k];
    const f_t aj = a_bar.x[k];
    settings_.log.printf("a_bar[%d] = %e\n", k, aj);
  }
  settings_.log.printf("b_bar[%d] = %e\n", i, b_bar[i]);
#endif

  inequality     = a_bar;
  inequality_rhs = b_bar_[i];

  return 0;
}

template <typename i_t, typename f_t>
variable_bounds_t<i_t, f_t>::variable_bounds_t(const lp_problem_t<i_t, f_t>& lp,
                                               const simplex_solver_settings_t<i_t, f_t>& settings,
                                               const std::vector<variable_type_t>& var_types,
                                               const csr_matrix_t<i_t, f_t>& Arow,
                                               const std::vector<i_t>& new_slacks)
  : upper_offsets(lp.num_cols + 1, 0),
    lower_offsets(lp.num_cols + 1, 0),
    upper_activities_(lp.num_rows, 0.0),
    lower_activities_(lp.num_rows, 0.0),
    num_pos_inf_(lp.num_rows, 0),
    num_neg_inf_(lp.num_rows, 0)
{
  if (settings.sub_mip) {
    return;  // Don't compute the variable upper/lower bounds inside sub-MIP
  }
  f_t start_time = tic();

  // Construct the slack map
  slack_map_.resize(lp.num_rows, -1);
  std::vector<f_t> slack_coeff(lp.num_rows, 0.0);
  for (i_t j : new_slacks) {
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    const i_t col_len   = col_end - col_start;
    assert(col_len == 1);
    const i_t i    = lp.A.i[col_start];
    slack_map_[i]  = j;
    slack_coeff[i] = lp.A.x[col_start];
  }

  // The constraints are in the form:
  // sum_j a_j x_j + sigma * slack = beta

  std::vector<i_t> num_integer_in_row(lp.num_rows, 0);
  // Compute the upper activities of the constraints
  for (i_t i = 0; i < lp.num_rows; i++) {
    const i_t row_start   = Arow.row_start[i];
    const i_t row_end     = Arow.row_start[i + 1];
    const i_t slack_index = slack_map_[i];
    f_t activity          = 0.0;
    for (i_t p = row_start; p < row_end; p++) {
      const i_t j = Arow.j[p];
      if (j == slack_index) { continue; }
      const f_t aj = Arow.x[p];
      const f_t uj = lp.upper[j];
      const f_t lj = lp.lower[j];

      if (aj > 0.0) {
        if (uj < inf) {
          activity += aj * uj;
        } else {
          num_pos_inf_[i]++;
        }
      } else {  // a_j < 0.0
        if (lj > -inf) {
          activity += aj * lj;
        } else {
          num_pos_inf_[i]++;
        }
      }

      if (var_types[j] == variable_type_t::INTEGER) { num_integer_in_row[i]++; }
    }
    upper_activities_[i] = activity;
  }

  // Compute the lower activities of the constraints
  for (i_t i = 0; i < lp.num_rows; i++) {
    const i_t row_start   = Arow.row_start[i];
    const i_t row_end     = Arow.row_start[i + 1];
    const i_t slack_index = slack_map_[i];
    f_t activity          = 0.0;
    f_t sigma             = 0.0;
    for (i_t p = row_start; p < row_end; p++) {
      const i_t j = Arow.j[p];
      if (j == slack_index) { continue; }
      const f_t aj = Arow.x[p];
      const f_t uj = lp.upper[j];
      const f_t lj = lp.lower[j];
      if (aj > 0.0) {
        if (lj > -inf) {
          activity += aj * lj;
        } else {
          num_neg_inf_[i]++;
        }
      } else {  // a_j < 0.0
        if (uj < inf) {
          activity += aj * uj;
        } else {
          num_neg_inf_[i]++;
        }
      }
    }
    lower_activities_[i] = activity;
  }

  // Now go through all continuous variables and use the activiites to get upper variable bounds
  i_t upper_edges = 0;
  for (i_t j = 0; j < lp.num_cols; j++) {
    upper_offsets[j] = upper_edges;
    if (var_types[j] != variable_type_t::CONTINUOUS) { continue; }
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    for (i_t p = col_start; p < col_end; p++) {
      const i_t i = lp.A.i[p];
      if (num_integer_in_row[i] < 1) { continue; }
      if (num_neg_inf_[i] > 2 && num_pos_inf_[i] > 2) { continue; }
      const i_t row_start = Arow.row_start[i];
      const i_t row_end   = Arow.row_start[i + 1];
      const i_t row_len   = row_end - row_start;
      if (row_len < 2) { continue; }
      const f_t a_ij              = lp.A.x[p];
      const f_t slack_lower       = lp.lower[slack_map_[i]];
      const f_t slack_upper       = lp.upper[slack_map_[i]];
      const f_t slack_coeff_i     = slack_coeff[i];
      const f_t sigma_slack_lower = slack_coeff_i == 1.0 ? slack_lower : -slack_upper;
      const f_t sigma_slack_upper = slack_coeff_i == 1.0 ? slack_upper : -slack_lower;

      if (sigma_slack_lower > -inf) {
        const f_t beta = lp.rhs[i] - sigma_slack_lower;
        // sum_k a_ik x_k <= beta

        // If we have too many variables in the row that would cause the activity to be infinite,
        // we cannot derive an variable bound
        if (a_ij > 0.0 && num_neg_inf_[i] <= 2) {
          const f_t lower_activity_j = lower_activity(lp.lower[j], lp.upper[j], a_ij);

          // This is inefficient if num_neg_inf_[i] > 0
          // If num_neg_inf_[i] == 1 and var_types[s] != INTEGER, we can't derive a bound
          // If num_neg_inf_[i] == 2 and var_types[s ^ j] != INTEGER, we can't derive a bound
          // If num_neg_inf_[i] == 2 and var_types[s ^ j] == INTEGER, and lower_activity_j != -inf,
          // we can't derive a bound
          for (i_t q = row_start; q < row_end; q++) {
            const i_t l = Arow.j[q];
            if (var_types[l] == variable_type_t::CONTINUOUS) { continue; }
            // sum_{k != l, k != j} a_ik x_k + a_ij x_j + a_il x_l <= beta
            // a_ij x_j <= -a_il x_l + beta - sum_{k != l, k != j} a_ik x_k
            const f_t a_il             = Arow.x[q];
            const f_t lower_activity_l = lower_activity(lp.lower[l], lp.upper[l], a_il);
            const f_t sum              = adjusted_lower_activity(
              lower_activities_[i], num_neg_inf_[i], lower_activity_j, lower_activity_l);
            if (sum > -inf) {
              // We have a valid variable upper bound
              // x_j <= -a_il/a_ij * x_l + beta/a_ij - 1/a_ij * sum_{k != l, k != j} a_ik *
              // bound(x_k)
              upper_variables.push_back(l);
              upper_weights.push_back(-a_il / a_ij);
              upper_biases.push_back(beta / a_ij - (1.0 / a_ij) * sum);
              upper_edges++;
            }
          }
        }
      }

      if (sigma_slack_upper < inf) {
        const f_t beta = lp.rhs[i] - sigma_slack_upper;
        // sum_k a_ik x_k >= beta

        // If we have too many variables in the row that would cause the activity to be infinite,
        // we cannot derive an variable bound
        if (a_ij < 0.0 && num_pos_inf_[i] <= 2) {
          const f_t upper_activity_j = upper_activity(lp.lower[j], lp.upper[j], a_ij);

          for (i_t q = row_start; q < row_end; q++) {
            const i_t l = Arow.j[q];
            if (var_types[l] == variable_type_t::CONTINUOUS) { continue; }
            // sum_{k != l, k != j} a_ik x_k + a_ij x_j + a_il x_l >= beta
            // a_ij x_j >= -a_il x_l + beta - sum_{k != l, k != j} a_ik x_k
            const f_t a_il             = Arow.x[q];
            const f_t upper_activity_l = upper_activity(lp.lower[l], lp.upper[l], a_il);
            const f_t sum              = adjusted_upper_activity(
              upper_activities_[i], num_pos_inf_[i], upper_activity_j, upper_activity_l);
            if (sum < inf) {
              // We have a valid variable upper bound
              // x_j <= -a_il/a_ij * x_l + beta/a_ij - 1/a_ij * sum_{k != l, k != j} a_ik *
              // bound(x_k)
              upper_variables.push_back(l);
              upper_weights.push_back(-a_il / a_ij);
              upper_biases.push_back(beta / a_ij - (1.0 / a_ij) * sum);
              upper_edges++;
            }
          }
        }
      }
    }
  }
  upper_offsets[lp.num_cols] = upper_edges;
  settings.log.printf("%d variable upper bounds in %.2f seconds\n", upper_edges, toc(start_time));

  // Now go through all continuous variables and use the activiites to get lower variable bounds
  i_t lower_edges = 0;
  for (i_t j = 0; j < lp.num_cols; j++) {
    lower_offsets[j] = lower_edges;
    if (var_types[j] != variable_type_t::CONTINUOUS) { continue; }
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    for (i_t p = col_start; p < col_end; p++) {
      const i_t i = lp.A.i[p];
      if (num_integer_in_row[i] < 1) { continue; }
      const i_t row_start = Arow.row_start[i];
      const i_t row_end   = Arow.row_start[i + 1];
      const i_t row_len   = row_end - row_start;
      if (row_len < 2) { continue; }
      const f_t a_ij              = lp.A.x[p];
      const f_t slack_lower       = lp.lower[slack_map_[i]];
      const f_t slack_upper       = lp.upper[slack_map_[i]];
      const f_t slack_coeff_i     = slack_coeff[i];
      const f_t sigma_slack_lower = slack_coeff_i == 1.0 ? slack_lower : -slack_upper;
      const f_t sigma_slack_upper = slack_coeff_i == 1.0 ? slack_upper : -slack_lower;

      if (sigma_slack_lower > -inf) {
        const f_t beta = lp.rhs[i] - sigma_slack_lower;
        // sum_k a_ik x_k <= beta

        // If we have too many variables in the row that would cause the activity to be infinite,
        // we cannot derive a variable bound
        if (a_ij < 0.0 && num_neg_inf_[i] <= 2) {
          const f_t lower_activity_j = lower_activity(lp.lower[j], lp.upper[j], a_ij);

          for (i_t q = row_start; q < row_end; q++) {
            const i_t l = Arow.j[q];
            if (var_types[l] == variable_type_t::CONTINUOUS) { continue; }
            // sum_{k != l, k != j} a_ik x_k + a_ij x_j + a_il x_l <= beta
            // a_ij x_j <= -a_il x_l + beta - sum_{k != l, k != j} a_ik x_k
            // x_j >= -a_il/a_ij * x_l + beta/a_ij - 1/a_ij * sum_{k != l, k != j} a_ik * bound(x_k)
            const f_t a_il             = Arow.x[q];
            const f_t lower_activity_l = lower_activity(lp.lower[l], lp.upper[l], a_il);
            const f_t sum              = adjusted_lower_activity(
              lower_activities_[i], num_neg_inf_[i], lower_activity_j, lower_activity_l);
            if (sum > -inf) {
              // We have a valid variable lower bound
              // x_j >= -a_il/a_ij * x_l + beta/a_ij - 1/a_ij * sum_{k != l, k != j} a_ik *
              // bound(x_k)
              lower_variables.push_back(l);
              lower_weights.push_back(-a_il / a_ij);
              lower_biases.push_back(beta / a_ij - (1.0 / a_ij) * sum);
              lower_edges++;
            }
          }
        }
      }

      if (sigma_slack_upper < inf) {
        const f_t beta = lp.rhs[i] - sigma_slack_upper;
        // sum_k a_ik x_k >= beta

        // If we have too many variables in the row that would cause the activity to be infinite,
        // we cannot derive a variable bound
        if (a_ij > 0.0 && num_pos_inf_[i] <= 2) {
          const f_t upper_activity_j = upper_activity(lp.lower[j], lp.upper[j], a_ij);

          for (i_t q = row_start; q < row_end; q++) {
            const i_t l = Arow.j[q];
            if (var_types[l] == variable_type_t::CONTINUOUS) { continue; }
            // sum_{k != l, k != j} a_ik x_k + a_ij x_j + a_il x_l >= beta
            // a_ij x_j >= -a_il x_l + beta - sum_{k != l, k != j} a_ik x_k
            const f_t a_il             = Arow.x[q];
            const f_t upper_activity_l = upper_activity(lp.lower[l], lp.upper[l], a_il);
            const f_t sum              = adjusted_upper_activity(
              upper_activities_[i], num_pos_inf_[i], upper_activity_j, upper_activity_l);
            if (sum < inf) {
              // We have a valid variable lower bound
              // x_j >= -a_il/a_ij * x_l + beta/a_ij - 1/a_ij * sum_{k != l, k != j} a_ik *
              // bound(x_k)
              lower_variables.push_back(l);
              lower_weights.push_back(-a_il / a_ij);
              lower_biases.push_back(beta / a_ij - (1.0 / a_ij) * sum);
              lower_edges++;
            }
          }
        }
      }
    }
  }
  lower_offsets[lp.num_cols] = lower_edges;
  settings.log.printf("%d variable lower bounds in %.2f seconds\n", lower_edges, toc(start_time));
}

template <typename i_t, typename f_t>
complemented_mixed_integer_rounding_cut_t<i_t, f_t>::complemented_mixed_integer_rounding_cut_t(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const std::vector<i_t>& new_slacks)
  : is_slack_(lp.num_cols, 0),
    slack_rows_(lp.num_cols, -1),
    lb_variable_(lp.num_cols, -1),
    lb_star_(lp.num_cols, 0.0),
    ub_variable_(lp.num_cols, -1),
    ub_star_(lp.num_cols, 0.0),
    transformed_upper_(lp.num_cols, inf),
    bound_changed_(lp.num_cols, 0),
    scratch_pad_(lp.num_cols)
{
  for (i_t j : new_slacks) {
    is_slack_[j]        = 1;
    const i_t col_start = lp.A.col_start[j];
    const i_t i         = lp.A.i[col_start];
    slack_rows_[j]      = i;
    assert(std::abs(lp.A.x[col_start]) == 1.0);
  }
}

template <typename i_t, typename f_t>
void complemented_mixed_integer_rounding_cut_t<i_t, f_t>::remove_small_coefficients(
  const std::vector<f_t>& lower_bounds,
  const std::vector<f_t>& upper_bounds,
  sparse_vector_t<i_t, f_t>& cut,
  f_t& cut_rhs)
{
  const i_t nz = cut.i.size();
  i_t removed  = 0;
  for (i_t k = 0; k < cut.i.size(); k++) {
    const i_t j = cut.i[k];

    // Check for small coefficients
    const f_t aj = cut.x[k];
    if (std::abs(aj) < 1e-6) {
      if (aj >= 0.0 && upper_bounds[j] < inf) {
        // Move this to the right-hand side
        cut_rhs -= aj * upper_bounds[j];
        cut.x[k] = 0.0;
        removed++;
      } else if (aj <= 0.0 && lower_bounds[j] > -inf) {
        cut_rhs += aj * lower_bounds[j];
        cut.x[k] = 0.0;
        removed++;
        continue;
      } else {
      }
    }
  }

  if (removed > 0) {
    sparse_vector_t<i_t, f_t> new_cut(cut.n, 0);
    cut.squeeze(new_cut);
    cut = new_cut;
  }
}

template <typename i_t, typename f_t>
void complemented_mixed_integer_rounding_cut_t<i_t, f_t>::bound_substitution(
  const lp_problem_t<i_t, f_t>& lp,
  const variable_bounds_t<i_t, f_t>& variable_bounds,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  std::vector<f_t>& transformed_xstar)
{
  transformed_xstar.resize(lp.num_cols);
  // Perform bound substitution for continuous variables
  for (i_t j = 0; j < lp.num_cols; j++) {
    if (var_types[j] != variable_type_t::CONTINUOUS) { continue; }
    // Step 1: Decide whether to use variable or simple bounds
    const f_t uj      = lp.upper[j];
    const f_t lj      = lp.lower[j];
    const f_t xstar_j = xstar[j];

    // Set lb and lb_star to the simple lower bound
    lb_variable_[j] = -1;
    lb_star_[j]     = lj;

    // Set ub and ub_star to the simple upper bound
    ub_variable_[j] = -1;
    ub_star_[j]     = uj;

    // Check the variable lower bound and update lb and lb_star
    // if these yield a tighter bound
    const i_t lower_variable_start = variable_bounds.lower_offsets[j];
    const i_t lower_variable_end   = variable_bounds.lower_offsets[j + 1];
    for (i_t p = lower_variable_start; p < lower_variable_end; p++) {
      const i_t i     = variable_bounds.lower_variables[p];
      const f_t gamma = variable_bounds.lower_weights[p];
      const f_t alpha = variable_bounds.lower_biases[p];
      // x_j >= gamma * x_i + alpha

      const f_t xstar_i = xstar[i];
      const f_t val     = gamma * xstar_i + alpha;
      if (val > lb_star_[j]) {
        lb_variable_[j] = p;
        lb_star_[j]     = val;
      }
    }

    // Check the variable upper bound and update ub and ub_star
    // if these yield a tighter bound
    const i_t upper_variable_start = variable_bounds.upper_offsets[j];
    const i_t upper_variable_end   = variable_bounds.upper_offsets[j + 1];
    for (i_t p = upper_variable_start; p < upper_variable_end; p++) {
      const i_t i     = variable_bounds.upper_variables[p];
      const f_t gamma = variable_bounds.upper_weights[p];
      const f_t alpha = variable_bounds.upper_biases[p];
      // x_j <= gamma * x_i + alpha

      const f_t xstar_i = xstar[i];
      const f_t val     = gamma * xstar_i + alpha;
      if (val < ub_star_[j]) {
        ub_variable_[j] = p;
        ub_star_[j]     = val;
      }
    }

    // Step 2: Decide to use the lower or upper bound
    const bool has_finite_lower_bound = lb_star_[j] > -inf;
    const bool has_finite_upper_bound = ub_star_[j] < inf;
    if (!has_finite_lower_bound && !has_finite_upper_bound) {
      transformed_xstar[j]  = xstar_j;
      transformed_upper_[j] = inf;
      bound_changed_[j]     = 0;
      continue;
    }
    if (has_finite_lower_bound &&
        (!has_finite_upper_bound || (xstar_j - lb_star_[j] <= ub_star_[j] - xstar_j))) {
      // Use the lower bound
      // lb_star_j <= x_j <= ub_star_j
      // v_j = x_j - lb_star_j,
      // 0 <= v_j <= ub_star_j - lb_star_j
      transformed_upper_[j] = ub_star_[j] - lb_star_[j];
      transformed_xstar[j]  = xstar_j - lb_star_[j];
      bound_changed_[j]     = (lb_star_[j] == 0.0) ? 0 : -1;
    } else if (has_finite_upper_bound) {
      // Use the upper bound
      // lb_star_j <= x_j <= ub_star_j
      // x_j + w_j = ub_star_j,
      // w_j = ub_star_j - x_j,
      // x_j = ub_star_j - w_j
      // 0 <= w_j <= ub_star_j - lb_star_j
      transformed_upper_[j] = ub_star_[j] - lb_star_[j];
      transformed_xstar[j]  = ub_star_[j] - xstar_j;
      bound_changed_[j]     = 1;
    }
  }

  // Perform bound substitution for the integer variables
  for (i_t j = 0; j < lp.num_cols; j++) {
    if (var_types[j] != variable_type_t::INTEGER) { continue; }
    const f_t uj      = lp.upper[j];
    const f_t lj      = lp.lower[j];
    const f_t xstar_j = xstar[j];

    lb_star_[j]           = lj;
    ub_star_[j]           = uj;
    transformed_xstar[j]  = xstar_j;
    transformed_upper_[j] = uj;
    bound_changed_[j]     = 0;

    if (uj < inf) {
      if (uj - xstar_j <= xstar_j - lj) {
        // Use the upper bound
        // lj <= x_j <= uj
        // x_j + w_j = uj,
        // w_j = uj - x_j,
        // x_j = uj - w_j
        // 0 <= w_j <= uj - lj
        transformed_upper_[j] = uj - lj;
        transformed_xstar[j]  = uj - xstar_j;
        bound_changed_[j]     = 1;
      } else if (lj != 0.0) {
        // Use the lower bound
        // lj <= x_j <= uj
        // v_j = x_j - lj,
        // 0 <= v_j <= uj - lj
        transformed_upper_[j] = uj - lj;
        transformed_xstar[j]  = xstar_j - lj;
        bound_changed_[j]     = -1;
      }
      continue;
    }

    if (lj > -inf && lj != 0.0) {
      // Use the lower bound
      // lj <= x_j <= uj
      // v_j = x_j - lj,
      // 0 <= v_j <= uj - lj
      transformed_upper_[j] = uj - lj;
      transformed_xstar[j]  = xstar_j - lj;
      bound_changed_[j]     = -1;
    }
  }
}

template <typename i_t, typename f_t>
void complemented_mixed_integer_rounding_cut_t<i_t, f_t>::transform_inequality(
  const variable_bounds_t<i_t, f_t>& variable_bounds,
  const std::vector<variable_type_t>& var_type,
  sparse_vector_t<i_t, f_t>& inequality,
  f_t& inequality_rhs)
{
  const i_t nz = inequality.i.size();
  for (i_t k = 0; k < nz; k++) {
    const i_t j  = inequality.i[k];
    const f_t aj = inequality.x[k];
    if (var_type[j] != variable_type_t::CONTINUOUS) {
      scratch_pad_.add_to_pad(j, aj);
      continue;
    }
    if (bound_changed_[j] == -1) {
      if (lb_variable_[j] == -1) {
        // v_j = x_j - l_j, v_j >= 0
        // x_j = v_j + l_j
        // sum_{k != j} a_k x_k + a_j x_j >= beta
        // sum_{k != j} a_k x_k + a_j (v_j + l_j) >= beta
        // sum_{k != j} a_k x_k + a_j v_j >= beta - a_j l_j
        const f_t lj = lb_star_[j];
        inequality_rhs -= aj * lj;
        scratch_pad_.add_to_pad(j, aj);
      } else {
        // v_j = x_j - lb*_j, v_j >= 0
        // x_j = v_j + lb*_j
        // lb*_j = gamma * x_i + alpha
        // x_j = v_j + gamma * x_i + alpha
        // sum_{k != j} a_k x_k + a_j x_j >= beta
        // sum_{k != j} a_k x_k + a_j (v_j + gamma * x_i + alpha) >= beta
        // sum_{k != j} a_k x_k + a_j v_j + a_j * gamma * x_i >= beta - a_j alpha
        const i_t p     = lb_variable_[j];
        const f_t alpha = variable_bounds.lower_biases[p];
        const f_t gamma = variable_bounds.lower_weights[p];
        const i_t i     = variable_bounds.lower_variables[p];
        inequality_rhs -= aj * alpha;
        scratch_pad_.add_to_pad(j, aj);
        scratch_pad_.add_to_pad(i, aj * gamma);
      }
    } else if (bound_changed_[j] == 1) {
      if (ub_variable_[j] == -1) {
        // w_j = u_j - x_j, w_j >= 0
        // x_j = u_j - w_j
        // sum_{k != j} a_k x_k + a_j x_j >= beta
        // sum_{k != j} a_k x_k + a_j (u_j - w_j) >= beta
        // sum_{k != j} a_k x_k - a_j w_j >= beta - a_j u_j
        const f_t uj = ub_star_[j];
        inequality_rhs -= aj * uj;
        scratch_pad_.add_to_pad(j, -aj);
      } else {
        // w_j = ub*_j - x_j, w_j >= 0
        // x_j = ub*_j - w_j
        // ub*_j = gamma * x_i + alpha
        // x_j = gamma * x_i + alpha - w_j
        // sum_{k != j} a_k x_k + a_j x_j >= beta
        // sum_{k != j} a_k x_k + a_j (ub*_j - w_j) >= beta
        // sum_{k != j} a_k x_k + a_j (gamma * x_i + alpha) - a_j w_j >= beta
        // sum_{k != j} a_k x_k + a_j gamma * x_i - a_j w_j >= beta - a_j alpha
        const i_t p     = ub_variable_[j];
        const f_t alpha = variable_bounds.upper_biases[p];
        const f_t gamma = variable_bounds.upper_weights[p];
        const i_t i     = variable_bounds.upper_variables[p];
        inequality_rhs -= aj * alpha;
        scratch_pad_.add_to_pad(j, -aj);
        scratch_pad_.add_to_pad(i, aj * gamma);
      }
    } else if (bound_changed_[j] == 0) {
      scratch_pad_.add_to_pad(j, aj);
    }
  }
  scratch_pad_.get_pad(inequality.i, inequality.x);
  // At this point we have converted all the continuous variables to be nonnegative
  // Note that since continuous variables had VUB or VLB, they modified
  // the integer variables.

  // We clear the scratch pad. As it is no longer needed.
  scratch_pad_.clear_pad();

  // We now convert all the integer variables to be nonnegative
  const i_t nz_after = inequality.i.size();
  for (i_t k = 0; k < nz_after; k++) {
    const i_t j = inequality.i[k];
    if (var_type[j] != variable_type_t::INTEGER) { continue; }
    const f_t aj = inequality.x[k];
    if (bound_changed_[j] == -1) {
      // v_j = x_j - l_j, v_j >= 0
      // x_j = v_j + l_j
      // sum_{k != j} a_k x_k + a_j x_j >= beta
      // sum_{k != j} a_k x_k + a_j (v_j + l_j) >= beta
      // sum_{k != j} a_k x_k + a_j v_j >= beta - a_j l_j
      const f_t lj = lb_star_[j];
      inequality_rhs -= aj * lj;
    } else if (bound_changed_[j] == 1) {
      // w_j = u_j - x_j, w_j >= 0
      // x_j = u_j - w_j
      // sum_{k != j} a_k x_j + a_j x_j >= beta
      // sum_{k != j} a_k x_j + a_j (u_j - w_j) >= beta
      // sum_{k != j} a_k x_j - a_j w_j >= beta - a_j u_j
      const f_t uj = ub_star_[j];
      inequality_rhs -= aj * uj;
      inequality.x[k] *= -1.0;
    }
  }
}

template <typename i_t, typename f_t>
void complemented_mixed_integer_rounding_cut_t<i_t, f_t>::untransform_inequality(
  const variable_bounds_t<i_t, f_t>& variable_bounds,
  const std::vector<variable_type_t>& var_type,
  sparse_vector_t<i_t, f_t>& inequality,
  f_t& rhs)
{
  // First convert all the integers variables back to their original form: l_j <= x_j <= u_j
  const i_t nz = inequality.i.size();
  for (i_t k = 0; k < nz; k++) {
    const i_t j = inequality.i[k];
    if (var_type[j] != variable_type_t::INTEGER) { continue; }
    const f_t dj = inequality.x[k];
    if (bound_changed_[j] == -1) {
      // v_j = x_j - l_j, v_j >= 0
      // sum_{k != j} d_k x_k + d_j v_j >= beta
      // sum_{k != j} d_k x_k + d_j (x_j - l_j) >= beta
      // sum_{k != j} d_k x_k + d_j x_j >= beta + d_j l_j
      const f_t lj = lb_star_[j];
      rhs += dj * lj;
    } else if (bound_changed_[j] == 1) {
      // w_j = u_j - x_j, w_j >= 0
      // sum_{k != j} d_k x_k + d_j w_j >= beta
      // sum_{k != j} d_k x_k + d_j (u_j - x_j) >= beta
      // sum_{k != j} d_k x_k - d_j x_j  >= beta - d_j u_j
      const f_t uj = ub_star_[j];
      rhs -= dj * uj;
      inequality.x[k] *= -1.0;
    }
  }
  // Then undo the VUB/VLB substitions and bring continuous variables back to their original form
  for (i_t k = 0; k < nz; k++) {
    const i_t j  = inequality.i[k];
    const f_t dj = inequality.x[k];
    if (var_type[j] != variable_type_t::CONTINUOUS) {
      scratch_pad_.add_to_pad(j, dj);
      continue;
    }
    if (bound_changed_[j] == -1) {
      if (lb_variable_[j] == -1) {
        // v_j = x_j - l_j, v_j >= 0
        // sum_{k != j} d_k x_k + d_j v_j >= beta
        // sum_{k != j} d_k x_k + d_j (x_j - l_j) >= beta
        // sum_{k != j} d_k x_k + d_j x_j >= beta + d_j l_j
        const f_t lj = lb_star_[j];
        rhs += dj * lj;
        scratch_pad_.add_to_pad(j, dj);
      } else {
        // v_j = x_j - lb*_j, v_j >= 0
        // lb*_j = gamma * x_i + alpha
        // v_j = x_j - gamma * x_i -  alpha
        // sum_{k != j} d_k x_k + d_j v_j >= beta
        // sum_{k != j} d_k x_k + d_j (x_j - gamma * x_i - alpha) >= beta
        // sum_{k != j} d_k x_k + d_j x_j - d_j * gamma * x_i >= beta + d_j alpha
        const i_t p     = lb_variable_[j];
        const f_t alpha = variable_bounds.lower_biases[p];
        const f_t gamma = variable_bounds.lower_weights[p];
        const i_t i     = variable_bounds.lower_variables[p];
        rhs += dj * alpha;
        scratch_pad_.add_to_pad(j, dj);
        scratch_pad_.add_to_pad(i, -dj * gamma);
      }
    } else if (bound_changed_[j] == 1) {
      if (ub_variable_[j] == -1) {
        // w_j = u_j - x_j, w_j >= 0
        // sum_{k != j} d_k x_k + d_j w_j >= beta
        // sum_{k != j} d_k x_k + d_j (u_j - x_j) >= beta
        // sum_{k != j} d_k x_k - d_j x_j  >= beta - d_j u_j
        const f_t uj = ub_star_[j];
        rhs -= dj * uj;
        scratch_pad_.add_to_pad(j, -dj);
      } else {
        // w_j = ub*_j - x_j, w_j >= 0
        // ub*_j = gamma * x_i + alpha
        // w_j = gamma * x_i + alpha - x_j
        // sum_{k != j} d_k x_k + d_j w_j >= beta
        // sum_{k != j} d_k x_k + d_j (gamma * x_i + alpha - x_j) >= beta
        // sum_{k != j} d_k x_k + d_j gamma * x_i - d_j x_j >= beta - d_j alpha
        const i_t p     = ub_variable_[j];
        const f_t alpha = variable_bounds.upper_biases[p];
        const f_t gamma = variable_bounds.upper_weights[p];
        const i_t i     = variable_bounds.upper_variables[p];
        rhs -= dj * alpha;
        scratch_pad_.add_to_pad(j, -dj);
        scratch_pad_.add_to_pad(i, dj * gamma);
      }
    } else {
      scratch_pad_.add_to_pad(j, dj);
    }
  }

  scratch_pad_.get_pad(inequality.i, inequality.x);
  scratch_pad_.clear_pad();
}

template <typename i_t, typename f_t>
void complemented_mixed_integer_rounding_cut_t<i_t, f_t>::
  generate_cut_nonnegative_maintain_indicies(const sparse_vector_t<i_t, f_t>& a,
                                             f_t beta,
                                             const std::vector<variable_type_t>& var_types,
                                             sparse_vector_t<i_t, f_t>& cut,
                                             f_t& cut_rhs)
{
  auto f = [](f_t q_1, f_t q_2) -> f_t {
    f_t q_1_hat = q_1 - std::floor(q_1);
    f_t q_2_hat = q_2 - std::floor(q_2);
    return std::min(q_1_hat, q_2_hat) + q_2_hat * std::floor(q_1);
  };

  auto h = [](f_t q) -> f_t { return std::max(q, 0.0); };

  cut     = a;
  cut_rhs = (beta - std::floor(beta)) * std::ceil(beta);

  for (i_t k = 0; k < a.i.size(); k++) {
    const i_t j = a.i[k];
    f_t aj      = a.x[k];
    if (var_types[j] == variable_type_t::INTEGER) {
      cut.x[k] = f(aj, beta);
    } else {
      cut.x[k] = h(aj);
    }
    if (cut.x[k] != cut.x[k]) {
      printf("cut.x[%d] %e != cut.x[%d] %e. aj %e beta %e var type %d\n",
             k,
             cut.x[k],
             k,
             cut.x[k],
             aj,
             beta,
             static_cast<int>(var_types[j]));
      exit(1);
    }
  }
}

template <typename i_t, typename f_t>
f_t complemented_mixed_integer_rounding_cut_t<i_t, f_t>::compute_violation(
  const sparse_vector_t<i_t, f_t>& cut, f_t cut_rhs, const std::vector<f_t>& xstar)
{
  f_t dot           = cut.dot(xstar);
  f_t cut_violation = cut_rhs - dot;
  return cut_violation;
}

template <typename i_t, typename f_t>
void complemented_mixed_integer_rounding_cut_t<i_t, f_t>::substitute_slacks(
  const lp_problem_t<i_t, f_t>& lp,
  csr_matrix_t<i_t, f_t>& Arow,
  sparse_vector_t<i_t, f_t>& cut,
  f_t& cut_rhs)
{
  // Remove slacks from the cut
  // So that the cut is only over the original variables
  bool found_slack = false;
  i_t cut_nz       = 0;
  std::vector<i_t> cut_indices;
  cut_indices.reserve(cut.i.size());

#ifdef CHECK_WORKSPACE
  for (i_t j = 0; j < x_workspace_.size(); j++) {
    if (x_workspace_[j] != 0.0) {
      printf("Begin Dirty x_workspace_[%d] = %e\n", j, x_workspace_[j]);
      assert(x_workspace_[j] == 0.0);
    }
    if (x_mark_[j] != 0) {
      printf("Begin Dirty x_mark_[%d] = %d\n", j, x_mark_[j]);
      assert(x_mark_[j] == 0);
    }
  }
#endif

  for (i_t k = 0; k < cut.i.size(); k++) {
    const i_t j  = cut.i[k];
    const f_t cj = cut.x[k];
    if (is_slack_[j]) {
      found_slack           = true;
      const i_t slack_start = lp.A.col_start[j];
#ifdef CHECK_SLACKS
      const i_t slack_end = lp.A.col_start[j + 1];
      const i_t slack_len = slack_end - slack_start;
      if (slack_len != 1) {
        printf("Slack %d has %d nzs in colum\n", j, slack_len);
        assert(slack_len == 1);
      }
#endif
      const f_t alpha = lp.A.x[slack_start];
#ifdef CHECK_SLACKS
      if (std::abs(alpha) != 1.0) {
        printf("Slack %d has non-unit coefficient %e\n", j, alpha);
        assert(std::abs(alpha) == 1.0);
      }
#endif

      // Do the substitution
      // Slack variable s_j participates in row i of the constraint matrix
      // Row i is of the form:
      // sum_{k != j} A(i, k) * x_k + alpha * s_j = rhs_i
      // where alpha = +1/-1
      /// So we have that
      // s_j = (rhs_i - sum_{k != j} A(i, k) * x_k)/alpha

      // Our cut is of the form:
      // sum_{k != j} C(k) * x_k + C(j) * s_j >= cut_rhs
      // So the cut becomes
      // sum_{k != j} C(k) * x_k + C(j)/alpha * (rhs_i - sum_{h != j} A(i, h) * x_h) >= cut_rhs
      // This is equivalent to:
      // sum_{k != j} C(k) * x_k + sum_{h != j} -C(j)/alpha * A(i, h) * x_h >= cut_rhs - C(j)/alpha
      // * rhs_i
      const i_t i = slack_rows_[j];
      cut_rhs -= cj * lp.rhs[i] / alpha;
      const i_t row_start = Arow.row_start[i];
      const i_t row_end   = Arow.row_start[i + 1];
      for (i_t q = row_start; q < row_end; q++) {
        const i_t h = Arow.j[q];
        if (h != j) {
          const f_t aih = Arow.x[q];
          scratch_pad_.add_to_pad(h, -cj * aih / alpha);
        } else {
          const f_t aij = Arow.x[q];
          if (std::abs(aij) != 1.0) {
            printf("Slack row %d has non-unit coefficient %e for variable %d\n", i, aij, j);
            assert(std::abs(aij) == 1.0);
          }
        }
      }

    } else {
      scratch_pad_.add_to_pad(j, cj);
    }
  }

  if (found_slack) {
    scratch_pad_.get_pad(cut.i, cut.x);
    // Sort the cut
    cut.sort();
  }

  // Clear the workspace
  scratch_pad_.clear_pad();
}

template <typename i_t, typename f_t>
void complemented_mixed_integer_rounding_cut_t<i_t, f_t>::combine_rows(
  const lp_problem_t<i_t, f_t>& lp,
  csr_matrix_t<i_t, f_t>& Arow,
  i_t xj,
  const sparse_vector_t<i_t, f_t>& pivot_row,
  f_t pivot_row_rhs,
  sparse_vector_t<i_t, f_t>& inequality,
  f_t& inequality_rhs)
{
  // Find the coefficient associated with variable xj in the pivot row
  f_t a_l_j = 0.0;
  for (i_t k = 0; k < pivot_row.i.size(); k++) {
    const i_t j = pivot_row.i[k];
    if (j == xj) {
      a_l_j = pivot_row.x[k];
      break;
    }
  }

  if (a_l_j == 0) { return; }

  f_t a_i_j = 0.0;

  i_t nz = 0;
  // Store the inequality in the workspace
  // and save the coefficient associated with variable xj
  for (i_t k = 0; k < inequality.i.size(); k++) {
    const i_t j = inequality.i[k];
    if (j != xj) {
      scratch_pad_.add_to_pad(j, inequality.x[k]);
    } else {
      a_i_j = inequality.x[k];
    }
  }

  f_t pivot_value = a_i_j / a_l_j;
  // Adjust the rhs of the inequality
  inequality_rhs -= pivot_value * pivot_row_rhs;

  // Adjust the coefficients of the inequality
  // based on the nonzeros in the pivot row
  for (i_t k = 0; k < pivot_row.i.size(); k++) {
    const i_t j = pivot_row.i[k];
    if (j != xj) { scratch_pad_.add_to_pad(j, -pivot_value * pivot_row.x[k]); }
  }

  // Store the new inequality
  scratch_pad_.get_pad(inequality.i, inequality.x);

  // Clear the workspace
  scratch_pad_.clear_pad();
}

template <typename i_t, typename f_t>
strong_cg_cut_t<i_t, f_t>::strong_cg_cut_t(const lp_problem_t<i_t, f_t>& lp,
                                           const std::vector<variable_type_t>& var_types,
                                           const std::vector<f_t>& xstar)
  : transformed_variables_(lp.num_cols, 0)
{
  // Determine the substition for the integer variables
  for (i_t j = 0; j < lp.num_cols; j++) {
    if (var_types[j] == variable_type_t::INTEGER) {
      const f_t l_j = lp.lower[j];
      const f_t u_j = lp.upper[j];
      if (l_j != 0.0) {
        // We need to transform the variable
        // Check the distance to each bound
        const f_t dist_to_lower = std::max(0.0, xstar[j] - l_j);
        const f_t dist_to_upper = std::max(0.0, u_j - xstar[j]);
        if (dist_to_upper >= dist_to_lower || u_j >= inf) {
          // We are closer to the lower bound.
          transformed_variables_[j] = -1;
        } else if (u_j < inf) {
          // We are closer to the finite upper bound
          transformed_variables_[j] = 1;
        }
      }
    }
  }
}

template <typename i_t, typename f_t>
i_t strong_cg_cut_t<i_t, f_t>::remove_continuous_variables_integers_nonnegative(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const std::vector<variable_type_t>& var_types,
  sparse_vector_t<i_t, f_t>& inequality,
  f_t& inequality_rhs)
{
  const bool verbose = false;
  // Count the number of continuous variables in the inequality
  i_t num_continuous = 0;
  const i_t nz       = inequality.i.size();
  for (i_t k = 0; k < nz; k++) {
    const i_t j = inequality.i[k];
    if (var_types[j] == variable_type_t::CONTINUOUS) { num_continuous++; }
  }

  if (verbose) { settings.log.printf("num_continuous %d\n", num_continuous); }
  // We assume the inequality is of the form sum_j a_j x_j <= rhs

  for (i_t k = 0; k < nz; k++) {
    const i_t j   = inequality.i[k];
    const f_t l_j = lp.lower[j];
    const f_t u_j = lp.upper[j];
    const f_t a_j = inequality.x[k];
    if (var_types[j] == variable_type_t::CONTINUOUS) {
      if (a_j == 0.0) { continue; }

      if (a_j > 0.0 && l_j > -inf) {
        // v_j = x_j - l_j >= 0
        // x_j = v_j + l_j
        // sum_{k != j} a_k x_k + a_j x_j <= rhs
        // sum_{k != j} a_k x_k + a_j (v_j + l_j) <= rhs
        // sum_{k != j} a_k x_k + a_j v_j <= rhs - a_j l_j
        inequality_rhs -= a_j * l_j;
        transformed_variables_[j] = -1;

        // We now have a_j * v_j with a_j, v_j >= 0
        // So we have sum_{k != j} a_k x_k <= sum_{k != j} a_k x_k + a_j v_j <= rhs - a_j l_j
        // So we can now drop the continuous variable v_j
        inequality.x[k] = 0.0;

      } else if (a_j < 0.0 && u_j < inf) {
        // w_j = u_j - x_j >= 0
        // x_j = u_j - w_j
        // sum_{k != j} a_k x_k + a_j x_j <= rhs
        // sum_{k != j} a_k x_k + a_j (u_j - w_j) <= rhs
        // sum_{k != j} a_k x_k - a_j w_j <= rhs - a_j u_j
        inequality_rhs -= a_j * u_j;
        transformed_variables_[j] = 1;

        // We now have a_j * w_j with a_j, w_j >= 0
        // So we have sum_{k != j} a_k x_k <= sum_{k != j} a_k x_k + a_j w_j <= rhs - a_j u_j
        // So we can now drop the continuous variable w_j
        inequality.x[k] = 0.0;
      } else {
        // We can't keep the coefficient of the continuous variable positive
        // This means we can't eliminate the continuous variable
        if (verbose) { settings.log.printf("x%d ak: %e lo: %e up: %e\n", j, a_j, l_j, u_j); }
        return -1;
      }
    } else {
      // The variable is integer. We just need to ensure it is nonnegative
      if (transformed_variables_[j] == -1) {
        // We are closer to the lower bound.
        // v_j = x_j - l_j >= 0
        // x_j = v_j + l_j
        // sum_{k != j} a_k x_k + a_j x_j <= rhs
        // sum_{k != j} a_k x_k + a_j (v_j + l_j) <= rhs
        // sum_{k != j} a_k x_k + a_j v_j <= rhs - a_j l_j
        inequality_rhs -= a_j * l_j;
      } else if (transformed_variables_[j] == 1) {
        // We are closer to the finite upper bound
        // w_j = u_j - x_j >= 0
        // x_j = u_j - w_j
        // sum_{k != j} a_k x_k + a_j x_j <= rhs
        // sum_{k != j} a_k x_k + a_j (u_j - w_j) <= rhs
        // sum_{k != j} a_k x_k - a_j w_j <= rhs - a_j u_j
        inequality_rhs -= a_j * u_j;
        inequality.x[k] *= -1.0;
      }
    }
  }

  // Squeeze out the zero coefficents
  sparse_vector_t<i_t, f_t> new_inequality(inequality.n, 0);
  inequality.squeeze(new_inequality);
  inequality = new_inequality;
  return 0;
}

template <typename i_t, typename f_t>
void strong_cg_cut_t<i_t, f_t>::to_original_integer_variables(const lp_problem_t<i_t, f_t>& lp,
                                                              sparse_vector_t<i_t, f_t>& cut,
                                                              f_t& cut_rhs)
{
  // We expect a cut of the form sum_j a_j y_j <= rhs
  // where y_j >= 0 is a transformed variable
  // We need to convert it back into a cut on the original variables

  for (i_t k = 0; k < cut.i.size(); k++) {
    const i_t j   = cut.i[k];
    const f_t a_j = cut.x[k];
    if (transformed_variables_[j] == -1) {
      // sum_{k != j} a_k x_k + a_j v_j <= rhs
      // v_j = x_j - l_j >= 0,
      // sum_{k != j} a_k x_k + a_j (x_j - l_j) <= rhs
      // sum_{k != j} a_k x_k + a_j x_j <= rhs + a_j l_j
      cut_rhs += a_j * lp.lower[j];
    } else if (transformed_variables_[j] == 1) {
      // sum_{k != j} a_k x_k + a_j w_j <= rhs
      // w_j = u_j - x_j >= 0
      // sum_{k != j} a_k x_k + a_j (u_j - x_j) <= rhs
      // sum_{k != j} a_k x_k - a_j x_j <= rhs - a_j u_j
      cut_rhs -= a_j * lp.upper[j];
      cut.x[k] *= -1.0;
    }
  }
}

template <typename i_t, typename f_t>
i_t strong_cg_cut_t<i_t, f_t>::generate_strong_cg_cut_integer_only(
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const std::vector<variable_type_t>& var_types,
  const sparse_vector_t<i_t, f_t>& inequality,
  f_t inequality_rhs,
  sparse_vector_t<i_t, f_t>& cut,
  f_t& cut_rhs)
{
  // We expect an inequality of the form sum_j a_j x_j <= rhs
  // where all the variables x_j are integer and nonnegative

  // We then apply the CG cut:
  // sum_j floor(a_j) x_j <= floor(rhs)
  cut.i.reserve(inequality.i.size());
  cut.x.reserve(inequality.i.size());
  cut.i.clear();
  cut.x.clear();

  f_t a_0   = inequality_rhs;
  f_t f_a_0 = fractional_part(a_0);

  if (f_a_0 == 0.0) {
    // f(a_0) == 0.0 so we do a weak CG cut
    cut.i.reserve(inequality.i.size());
    cut.x.reserve(inequality.i.size());
    cut.i.clear();
    cut.x.clear();
    for (i_t k = 0; k < inequality.i.size(); k++) {
      const i_t j   = inequality.i[k];
      const f_t a_j = inequality.x[k];
      if (var_types[j] == variable_type_t::INTEGER) {
        cut.i.push_back(j);
        cut.x.push_back(std::floor(a_j));
      } else {
        return -1;
      }
    }
    cut_rhs = std::floor(inequality_rhs);
  } else {
    return generate_strong_cg_cut_helper(
      inequality.i, inequality.x, inequality_rhs, var_types, cut, cut_rhs);
  }
  return 0;
}

template <typename i_t, typename f_t>
i_t strong_cg_cut_t<i_t, f_t>::generate_strong_cg_cut_helper(
  const std::vector<i_t>& indicies,
  const std::vector<f_t>& coefficients,
  f_t rhs,
  const std::vector<variable_type_t>& var_types,
  sparse_vector_t<i_t, f_t>& cut,
  f_t& cut_rhs)
{
  const bool verbose = false;
  const i_t nz       = indicies.size();
  const f_t f_a_0    = fractional_part(rhs);

  const f_t min_fractional_part = 1e-2;
  if (f_a_0 < min_fractional_part) { return -1; }
  if (f_a_0 > 1 - min_fractional_part) { return -1; }

  // We will try to generat a strong CG cut.
  // Find the unique integer k such that
  // 1/(k+1) <= f(a_0) < 1/k
  const f_t k_upper = 1.0 / f_a_0;
  i_t k             = static_cast<i_t>(std::ceil(k_upper)) - 1;

  const f_t alpha = 1.0 - f_a_0;
  f_t lower       = 1.0 / static_cast<f_t>(k + 1);
  f_t upper       = 1.0 / static_cast<f_t>(k);
  if (verbose) { printf("f_a_0 %e lower %e upper %e alpha %e\n", f_a_0, lower, upper, alpha); }
  if (f_a_0 >= lower && f_a_0 < upper) {
    cut.i.reserve(nz);
    cut.x.reserve(nz);
    cut.i.clear();
    cut.x.clear();
    for (i_t q = 0; q < nz; q++) {
      const i_t j   = indicies[q];
      const f_t a_j = coefficients[q];
      if (var_types[j] == variable_type_t::INTEGER) {
        const f_t f_a_j = fractional_part(a_j);
        const f_t tol   = 1e-4;
        if (f_a_j <= f_a_0 + tol) {
          cut.i.push_back(j);
          cut.x.push_back((k + 1.0) * std::floor(a_j));
          if (verbose) { printf("j %d a_j %e f_a_j %e k %d\n", j, a_j, f_a_j, k); }
        } else {
          // Find p such that p <= k * f(a_j) < p + 1
          i_t p = static_cast<i_t>(std::floor(k * f_a_j));
          // If f(a_j) > f(a_0) + p /k (1 - f(a_0)) then we can increase the cofficient by 1
          const f_t rhs_j = f_a_0 + static_cast<f_t>(p) / static_cast<f_t>(k) * alpha;
          const i_t coeff = (k + 1) * static_cast<i_t>(std::floor(a_j)) + p;
          if (f_a_j > rhs_j + tol) {
            cut.i.push_back(j);
            cut.x.push_back(static_cast<f_t>(coeff + 1));
          } else {
            cut.i.push_back(j);
            cut.x.push_back(static_cast<f_t>(coeff));
          }
        }
      } else {
        return -1;
      }
    }
  } else {
    if (verbose) { printf("Error: k %d lower %e f(a_0) %e upper %e\n", k, lower, f_a_0, upper); }
    return -1;
  }
  cut_rhs = (k + 1.0) * std::floor(rhs);
  if (verbose) {
    printf("Generated strong CG cut: k %d f_a_0 %e cut_rhs %e\n", k, f_a_0, cut_rhs);
    for (i_t q = 0; q < cut.i.size(); q++) {
      if (cut.x[q] != 0.0) { printf("%.16e x%d ", cut.x[q], cut.i[q]); }
    }
    printf("\n");
    printf("Original inequality rhs %e nz %ld\n", rhs, coefficients.size());
    for (i_t q = 0; q < nz; q++) {
      printf("%e x%d ", coefficients[q], indicies[q]);
    }
    printf("\n");
  }
  return 0;
}

template <typename i_t, typename f_t>
i_t strong_cg_cut_t<i_t, f_t>::generate_strong_cg_cut(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const std::vector<variable_type_t>& var_types,
  const sparse_vector_t<i_t, f_t>& inequality,
  const f_t inequality_rhs,
  const std::vector<f_t>& xstar,
  sparse_vector_t<i_t, f_t>& cut,
  f_t& cut_rhs)
{
#ifdef PRINT_INEQUALITY_INFO
  for (i_t k = 0; k < inequality.i.size(); k++) {
    printf("%e %c%d ",
           inequality.x[k],
           var_types[inequality.i[k]] == variable_type_t::CONTINUOUS ? 'x' : 'y',
           inequality.i[k]);
  }
  printf("CG inequality rhs %e\n", inequality_rhs);
#endif
  // Try to remove continuous variables from the inequality
  // and transform integer variables to be nonnegative

  // Copy the inequality since remove continuous variables will modify it
  sparse_vector_t<i_t, f_t> cg_inequality = inequality;
  f_t cg_inequality_rhs                   = inequality_rhs;
  i_t status                              = remove_continuous_variables_integers_nonnegative(
    lp, settings, var_types, cg_inequality, cg_inequality_rhs);

  if (status != 0) {
    // Try negating the equality and see if that helps
    cg_inequality = inequality;
    cg_inequality.negate();
    cg_inequality_rhs = -inequality_rhs;

    status = remove_continuous_variables_integers_nonnegative(
      lp, settings, var_types, cg_inequality, cg_inequality_rhs);
  }

  if (status == 0) {
    // We have an inequality with no continuous variables

    // Generate a CG cut
    status = generate_strong_cg_cut_integer_only(
      settings, var_types, cg_inequality, cg_inequality_rhs, cut, cut_rhs);
    if (status != 0) { return -1; }

    // Convert the CG cut back to the original variables
    to_original_integer_variables(lp, cut, cut_rhs);

    // Check for violation
    f_t dot = cut.dot(xstar);
    // If the cut is violated we will have: sum_j a_j xstar_j > rhs
    f_t violation                     = dot - cut_rhs;
    const f_t min_violation_threshold = 1e-6;
    if (violation > min_violation_threshold) {
      //  Note that no slacks are currently present. Since slacks are currently treated as
      //  continuous. However, this may change. We may need to substitute out the slacks here

      // The CG cut is in the form: sum_j a_j x_j <= rhs
      // The cut pool wants the cut in the form: sum_j a_j x_j >= rhs
      cut.negate();
      cut_rhs *= -1.0;
      return 0;
    }
  }
  return -1;
}

template <typename i_t, typename f_t>
i_t add_cuts(const simplex_solver_settings_t<i_t, f_t>& settings,
             const csr_matrix_t<i_t, f_t>& cuts,
             const std::vector<f_t>& cut_rhs,
             lp_problem_t<i_t, f_t>& lp,
             std::vector<i_t>& new_slacks,
             lp_solution_t<i_t, f_t>& solution,
             basis_update_mpf_t<i_t, f_t>& basis_update,
             std::vector<i_t>& basic_list,
             std::vector<i_t>& nonbasic_list,
             std::vector<variable_status_t>& vstatus,
             std::vector<f_t>& edge_norms)

{
  // Given a set of cuts: C*x <= d that are currently violated
  // by the current solution x* (i.e. C*x* > d), this function
  // adds the cuts into the LP and solves again.

#ifdef CHECK_BASIS
  {
    csc_matrix_t<i_t, f_t> Btest(lp.num_rows, lp.num_rows, 1);
    basis_update.multiply_lu(Btest);
    csc_matrix_t<i_t, f_t> B(lp.num_rows, lp.num_rows, 1);
    form_b(lp.A, basic_list, B);
    csc_matrix_t<i_t, f_t> Diff(lp.num_rows, lp.num_rows, 1);
    add(Btest, B, 1.0, -1.0, Diff);
    const f_t err = Diff.norm1();
    settings.log.printf("Before || B - L*U || %e\n", err);
    assert(err <= 1e-6);
  }
#endif

  const i_t p = cuts.m;
  if (cut_rhs.size() != static_cast<size_t>(p)) {
    settings.log.printf("cut_rhs must have the same number of rows as cuts\n");
    assert(cut_rhs.size() == static_cast<size_t>(p));
  }
  settings.log.debug("Number of cuts %d\n", p);
  settings.log.debug("Original lp rows %d\n", lp.num_rows);
  settings.log.debug("Original lp cols %d\n", lp.num_cols);

  csr_matrix_t<i_t, f_t> new_A_row(lp.num_rows, lp.num_cols, 1);
  lp.A.to_compressed_row(new_A_row);

  i_t append_status = new_A_row.append_rows(cuts);
  if (append_status != 0) {
    settings.log.printf("append_rows error: %d\n", append_status);
    assert(append_status == 0);
  }

  csc_matrix_t<i_t, f_t> new_A_col(lp.num_rows + p, lp.num_cols, 1);
  new_A_row.to_compressed_col(new_A_col);

  // Add in slacks variables for the new rows
  lp.lower.resize(lp.num_cols + p);
  lp.upper.resize(lp.num_cols + p);
  lp.objective.resize(lp.num_cols + p);
  edge_norms.resize(lp.num_cols + p);
  i_t nz = new_A_col.col_start[lp.num_cols];
  new_A_col.col_start.resize(lp.num_cols + p + 1);
  new_A_col.i.resize(nz + p);
  new_A_col.x.resize(nz + p);
  i_t k = lp.num_rows;
  for (i_t j = lp.num_cols; j < lp.num_cols + p; j++) {
    new_A_col.col_start[j] = nz;
    new_A_col.i[nz]        = k++;
    new_A_col.x[nz]        = 1.0;
    nz++;
    lp.lower[j]     = 0.0;
    lp.upper[j]     = inf;
    lp.objective[j] = 0.0;
    edge_norms[j]   = 1.0;
    new_slacks.push_back(j);
  }
  settings.log.debug("Done adding slacks\n");
  new_A_col.col_start[lp.num_cols + p] = nz;
  new_A_col.n                          = lp.num_cols + p;

  lp.A = new_A_col;

  // Check that all slack columns have length 1
  for (i_t slack : new_slacks) {
    const i_t col_start = lp.A.col_start[slack];
    const i_t col_end   = lp.A.col_start[slack + 1];
    const i_t col_len   = col_end - col_start;
    if (col_len != 1) {
      settings.log.printf("Add cuts: Slack %d has %d nzs in column\n", slack, col_len);
      assert(col_len == 1);
    }
  }

  i_t old_rows = lp.num_rows;
  lp.num_rows += p;
  i_t old_cols = lp.num_cols;
  lp.num_cols += p;

  lp.rhs.resize(lp.num_rows);
  for (i_t k = old_rows; k < old_rows + p; k++) {
    const i_t h = k - old_rows;
    lp.rhs[k]   = cut_rhs[h];
  }
  settings.log.debug("Done adding rhs\n");

  // Construct C_B = C(:, basic_list)
  std::vector<i_t> C_col_degree(lp.num_cols, 0);
  i_t cuts_nz = cuts.row_start[p];
  for (i_t q = 0; q < cuts_nz; q++) {
    const i_t j = cuts.j[q];
    if (j >= lp.num_cols) {
      settings.log.printf("Cut column index j=%d exceeds num_cols=%d\n", j, lp.num_cols);
      return -1;
    }
    C_col_degree[j]++;
  }
  settings.log.debug("Done computing C_col_degree\n");

  std::vector<i_t> in_basis(old_cols, -1);
  const i_t num_basic = static_cast<i_t>(basic_list.size());
  i_t C_B_nz          = 0;
  for (i_t k = 0; k < num_basic; k++) {
    const i_t j = basic_list[k];
    if (j < 0 || j >= old_cols) {
      settings.log.printf(
        "basic_list[%d] = %d is out of bounds %d old_cols %d\n", k, j, j, old_cols);
      assert(j >= 0 && j < old_cols);
    }
    in_basis[j] = k;
    // The cuts are on the original variables. So it is possible that
    // a slack will be basic and thus not part of the cuts matrix
    if (j < cuts.n) { C_B_nz += C_col_degree[j]; }
  }
  settings.log.debug("Done estimating C_B_nz\n");

  csr_matrix_t<i_t, f_t> C_B(p, num_basic, C_B_nz);
  nz = 0;
  for (i_t i = 0; i < p; i++) {
    C_B.row_start[i]    = nz;
    const i_t row_start = cuts.row_start[i];
    const i_t row_end   = cuts.row_start[i + 1];
    for (i_t q = row_start; q < row_end; q++) {
      const i_t j       = cuts.j[q];
      const i_t j_basis = in_basis[j];
      if (j_basis == -1) { continue; }
      C_B.j[nz] = j_basis;
      C_B.x[nz] = cuts.x[q];
      nz++;
    }
  }
  C_B.row_start[p] = nz;

  if (nz != C_B_nz) {
    settings.log.printf("Add cuts: predicted nz %d actual nz %d\n", C_B_nz, nz);
    assert(nz == C_B_nz);
  }
  settings.log.debug("C_B rows %d cols %d nz %d\n", C_B.m, C_B.n, nz);

  // Adjust the basis update to include the new cuts
  basis_update.append_cuts(C_B);

  basic_list.resize(lp.num_rows, 0);
  i_t h = old_cols;
  for (i_t j = old_rows; j < lp.num_rows; j++) {
    basic_list[j] = h++;
  }

#ifdef CHECK_BASIS
  // Check the basis update
  csc_matrix_t<i_t, f_t> Btest(lp.num_rows, lp.num_rows, 1);
  basis_update.multiply_lu(Btest);

  csc_matrix_t<i_t, f_t> B(lp.num_rows, lp.num_rows, 1);
  form_b(lp.A, basic_list, B);

  csc_matrix_t<i_t, f_t> Diff(lp.num_rows, lp.num_rows, 1);
  add(Btest, B, 1.0, -1.0, Diff);
  const f_t err = Diff.norm1();
  settings.log.printf("After || B - L*U || %e\n", err);
  if (err > 1e-6) {
    settings.log.printf("Diff matrix\n");
    // Diff.print_matrix();
    assert(err <= 1e-6);
  }
#endif
  // Adjust the vstatus
  vstatus.resize(lp.num_cols);
  for (i_t j = old_cols; j < lp.num_cols; j++) {
    vstatus[j] = variable_status_t::BASIC;
  }

  // Adjust the solution
  solution.x.resize(lp.num_cols, 0.0);
  solution.y.resize(lp.num_rows, 0.0);
  solution.z.resize(lp.num_cols, 0.0);

  return 0;
}

template <typename i_t, typename f_t>
i_t remove_cuts(lp_problem_t<i_t, f_t>& lp,
                const simplex_solver_settings_t<i_t, f_t>& settings,
                f_t start_time,
                csr_matrix_t<i_t, f_t>& Arow,
                std::vector<i_t>& new_slacks,
                i_t original_rows,
                std::vector<variable_type_t>& var_types,
                std::vector<variable_status_t>& vstatus,
                std::vector<f_t>& edge_norms,
                std::vector<f_t>& x,
                std::vector<f_t>& y,
                std::vector<f_t>& z,
                std::vector<i_t>& basic_list,
                std::vector<i_t>& nonbasic_list,
                basis_update_mpf_t<i_t, f_t>& basis_update)
{
  std::vector<i_t> cuts_to_remove;
  cuts_to_remove.reserve(lp.num_rows - original_rows);
  std::vector<i_t> slacks_to_remove;
  slacks_to_remove.reserve(lp.num_rows - original_rows);
  const f_t dual_tol = 1e-10;

  std::vector<i_t> is_slack(lp.num_cols, 0);
  for (i_t j : new_slacks) {
    is_slack[j] = 1;
#ifdef CHECK_SLACKS
    // Check that slack column length is 1
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    const i_t col_len   = col_end - col_start;
    if (col_len != 1) {
      printf("Remove cuts: Slack %d has %d nzs in column\n", j, col_len);
      assert(col_len == 1);
    }
#endif
  }

  for (i_t k = original_rows; k < lp.num_rows; k++) {
    if (std::abs(y[k]) < dual_tol) {
      const i_t row_start = Arow.row_start[k];
      const i_t row_end   = Arow.row_start[k + 1];
      i_t last_slack      = -1;
      const f_t slack_tol = 1e-3;
      for (i_t p = row_start; p < row_end; p++) {
        const i_t j = Arow.j[p];
        if (is_slack[j]) {
          if (vstatus[j] == variable_status_t::BASIC && x[j] > slack_tol) { last_slack = j; }
        }
      }
      if (last_slack != -1) {
        cuts_to_remove.push_back(k);
        slacks_to_remove.push_back(last_slack);
      }
    }
  }

  if (cuts_to_remove.size() > 0) {
    std::vector<i_t> marked_rows(lp.num_rows, 0);
    for (i_t i : cuts_to_remove) {
      marked_rows[i] = 1;
    }
    std::vector<i_t> marked_cols(lp.num_cols, 0);
    for (i_t j : slacks_to_remove) {
      marked_cols[j] = 1;
    }

    std::vector<f_t> new_rhs(lp.num_rows - cuts_to_remove.size());
    std::vector<f_t> new_solution_y(lp.num_rows - cuts_to_remove.size());
    i_t h = 0;
    for (i_t i = 0; i < lp.num_rows; i++) {
      if (!marked_rows[i]) {
        new_rhs[h]        = lp.rhs[i];
        new_solution_y[h] = y[i];
        h++;
      }
    }
    csr_matrix_t<i_t, f_t> new_Arow(1, 1, 0);
    Arow.remove_rows(marked_rows, new_Arow);
    Arow = new_Arow;
    Arow.to_compressed_col(lp.A);

    std::vector<f_t> new_objective(lp.num_cols - slacks_to_remove.size());
    std::vector<f_t> new_lower(lp.num_cols - slacks_to_remove.size());
    std::vector<f_t> new_upper(lp.num_cols - slacks_to_remove.size());
    std::vector<variable_type_t> new_var_types(lp.num_cols - slacks_to_remove.size());
    std::vector<variable_status_t> new_vstatus(lp.num_cols - slacks_to_remove.size());
    std::vector<f_t> new_edge_norms(lp.num_cols - slacks_to_remove.size());
    std::vector<i_t> new_basic_list;
    new_basic_list.reserve(lp.num_rows - slacks_to_remove.size());
    std::vector<i_t> new_nonbasic_list;
    new_nonbasic_list.reserve(nonbasic_list.size());
    std::vector<f_t> new_solution_x(lp.num_cols - slacks_to_remove.size());
    std::vector<f_t> new_solution_z(lp.num_cols - slacks_to_remove.size());
    std::vector<i_t> new_is_slacks(lp.num_cols - slacks_to_remove.size(), 0);
    h = 0;
    for (i_t k = 0; k < lp.num_cols; k++) {
      if (!marked_cols[k]) {
        new_objective[h]  = lp.objective[k];
        new_lower[h]      = lp.lower[k];
        new_upper[h]      = lp.upper[k];
        new_var_types[h]  = var_types[k];
        new_vstatus[h]    = vstatus[k];
        new_edge_norms[h] = edge_norms[k];
        new_solution_x[h] = x[k];
        new_solution_z[h] = z[k];
        new_is_slacks[h]  = is_slack[k];
        if (new_vstatus[h] != variable_status_t::BASIC) {
          new_nonbasic_list.push_back(h);
        } else {
          new_basic_list.push_back(h);
        }
        h++;
      }
    }
    lp.A.remove_columns(marked_cols);
    lp.A.to_compressed_row(Arow);
    lp.objective = new_objective;
    lp.lower     = new_lower;
    lp.upper     = new_upper;
    lp.rhs       = new_rhs;
    var_types    = new_var_types;
    lp.num_cols  = lp.A.n;
    lp.num_rows  = lp.A.m;

    new_slacks.clear();
    new_slacks.reserve(lp.num_cols);
    for (i_t j = 0; j < lp.num_cols; j++) {
      if (new_is_slacks[j]) { new_slacks.push_back(j); }
    }
    basic_list    = new_basic_list;
    nonbasic_list = new_nonbasic_list;
    vstatus       = new_vstatus;
    edge_norms    = new_edge_norms;
    x             = new_solution_x;
    y             = new_solution_y;
    z             = new_solution_z;

    settings.log.debug("Removed %d cuts. After removal %d rows %d columns %d nonzeros\n",
                       cuts_to_remove.size(),
                       lp.num_rows,
                       lp.num_cols,
                       lp.A.col_start[lp.A.n]);

    basis_update.resize(lp.num_rows);
    i_t refactor_status = basis_update.refactor_basis(
      lp.A, settings, lp.lower, lp.upper, start_time, basic_list, nonbasic_list, vstatus);
    if (refactor_status == CONCURRENT_HALT_RETURN) { return CONCURRENT_HALT_RETURN; }
    if (refactor_status == TIME_LIMIT_RETURN) { return TIME_LIMIT_RETURN; }
  }

  return 0;
}

template <typename i_t, typename f_t>
void read_saved_solution_for_cut_verification(const lp_problem_t<i_t, f_t>& lp,
                                              const simplex_solver_settings_t<i_t, f_t>& settings,
                                              std::vector<f_t>& saved_solution)
{
  settings.log.printf("Trying to open solution.dat\n");
  FILE* fid = NULL;
  fid       = fopen("solution.dat", "r");
  if (fid != NULL) {
    i_t n_solution_dat;
    i_t count = fscanf(fid, "%d\n", &n_solution_dat);
    settings.log.printf(
      "Solution.dat variables %d =? %d =? count %d\n", n_solution_dat, lp.num_cols, count);
    bool good = true;
    if (count == 1 && n_solution_dat == lp.num_cols) {
      settings.log.printf("Opened solution.dat with %d number of variables\n", n_solution_dat);
      saved_solution.resize(n_solution_dat);
      for (i_t j = 0; j < n_solution_dat; j++) {
        count = fscanf(fid, "%lf", &saved_solution[j]);
        if (count != 1) {
          settings.log.printf("bad read solution.dat: j %d count %d\n", j, count);
          good = false;
          break;
        }
      }
    } else {
      good = false;
    }
    fclose(fid);

    if (!good) {
      saved_solution.resize(0);
      settings.log.printf("Solution.dat is bad\n");
    } else {
      settings.log.printf("Read solution file\n");

      auto hash_combine_f = [](size_t seed, f_t x) {
        seed ^= std::hash<f_t>{}(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
      };
      size_t seed = lp.num_cols;
      for (i_t j = 0; j < lp.num_cols; ++j) {
        seed = hash_combine_f(seed, saved_solution[j]);
      }
      settings.log.printf("Saved solution hash: %20x\n", seed);

      // Compute || A * x - b ||_inf
      std::vector<f_t> residual = lp.rhs;
      matrix_vector_multiply(lp.A, 1.0, saved_solution, -1.0, residual);
      settings.log.printf("Saved solution: || A*x - b ||_inf %e\n",
                          vector_norm_inf<i_t, f_t>(residual));
      f_t infeas = 0;
      for (i_t j = 0; j < lp.num_cols; j++) {
        if (saved_solution[j] < lp.lower[j] - 1e-6) {
          f_t curr_infeas = (lp.lower[j] - saved_solution[j]);
          infeas += curr_infeas;
          settings.log.printf(
            "j: %d saved solution %e lower %e\n", j, saved_solution[j], lp.lower[j]);
        }
        if (saved_solution[j] > lp.upper[j] + 1e-6) {
          f_t curr_infeas = (saved_solution[j] - lp.upper[j]);
          infeas += curr_infeas;
          settings.log.printf(
            "j %d saved solution %e upper %e\n", j, saved_solution[j], lp.upper[j]);
        }
      }
      settings.log.printf("Bound infeasibility %e\n", infeas);
    }
  } else {
    settings.log.printf("Could not open solution.dat\n");
  }
}

template <typename i_t, typename f_t>
void write_solution_for_cut_verification(const lp_problem_t<i_t, f_t>& lp,
                                         const std::vector<f_t>& solution)
{
  FILE* fid = NULL;
  fid       = fopen("solution.dat", "w");
  if (fid != NULL) {
    printf("Writing solution.dat\n");

    std::vector<f_t> residual = lp.rhs;
    matrix_vector_multiply(lp.A, 1.0, solution, -1.0, residual);
    printf("|| A*x - b ||_inf %e\n", vector_norm_inf<i_t, f_t>(residual));
    auto hash_combine_f = [](size_t seed, f_t x) {
      seed ^= std::hash<f_t>{}(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      return seed;
    };
    printf("incumbent size %ld original lp cols %d\n", solution.size(), lp.num_cols);
    i_t n       = lp.num_cols;
    size_t seed = n;
    fprintf(fid, "%d\n", n);
    for (i_t j = 0; j < n; ++j) {
      fprintf(fid, "%.17g\n", solution[j]);
      seed = hash_combine_f(seed, solution[j]);
    }
    printf("Solution hash: %20x\n", seed);
    fclose(fid);
  }
}

template <typename i_t, typename f_t>
void verify_cuts_against_saved_solution(const csr_matrix_t<i_t, f_t>& cuts,
                                        const std::vector<f_t>& cut_rhs,
                                        const std::vector<f_t>& saved_solution)
{
  if (saved_solution.size() > 0) {
    csc_matrix_t<i_t, f_t> cuts_to_add_col(cuts.m, cuts.n, cuts.row_start[cuts.m]);
    cuts.to_compressed_col(cuts_to_add_col);
    std::vector<f_t> Cx(cuts.m);
    matrix_vector_multiply(cuts_to_add_col, 1.0, saved_solution, 0.0, Cx);
    const i_t num_cuts = cuts.m;
    for (i_t k = 0; k < num_cuts; k++) {
      if (Cx[k] > cut_rhs[k] + 1e-6) {
        printf("Cut %d is violated by saved solution. Cx %e cut_rhs %e Diff: %e\n",
               k,
               Cx[k],
               cut_rhs[k],
               Cx[k] - cut_rhs[k]);
      }
    }
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class cut_pool_t<int, double>;
template class cut_generation_t<int, double>;
template class knapsack_generation_t<int, double>;
template class tableau_equality_t<int, double>;
template class complemented_mixed_integer_rounding_cut_t<int, double>;
template class variable_bounds_t<int, double>;

template int add_cuts(const simplex_solver_settings_t<int, double>& settings,
                      const csr_matrix_t<int, double>& cuts,
                      const std::vector<double>& cut_rhs,
                      lp_problem_t<int, double>& lp,
                      std::vector<int>& new_slacks,
                      lp_solution_t<int, double>& solution,
                      basis_update_mpf_t<int, double>& basis_update,
                      std::vector<int>& basic_list,
                      std::vector<int>& nonbasic_list,
                      std::vector<variable_status_t>& vstatus,
                      std::vector<double>& edge_norms);

template int remove_cuts<int, double>(lp_problem_t<int, double>& lp,
                                      const simplex_solver_settings_t<int, double>& settings,
                                      double start_time,
                                      csr_matrix_t<int, double>& Arow,
                                      std::vector<int>& new_slacks,
                                      int original_rows,
                                      std::vector<variable_type_t>& var_types,
                                      std::vector<variable_status_t>& vstatus,
                                      std::vector<double>& edge_norms,
                                      std::vector<double>& x,
                                      std::vector<double>& y,
                                      std::vector<double>& z,
                                      std::vector<int>& basic_list,
                                      std::vector<int>& nonbasic_list,
                                      basis_update_mpf_t<int, double>& basis_update);

template void read_saved_solution_for_cut_verification<int, double>(
  const lp_problem_t<int, double>& lp,
  const simplex_solver_settings_t<int, double>& settings,
  std::vector<double>& saved_solution);

template void write_solution_for_cut_verification<int, double>(const lp_problem_t<int, double>& lp,
                                                               const std::vector<double>& solution);

template void verify_cuts_against_saved_solution<int, double>(
  const csr_matrix_t<int, double>& cuts,
  const std::vector<double>& cut_rhs,
  const std::vector<double>& saved_solution);
#endif

}  // namespace cuopt::linear_programming::dual_simplex
