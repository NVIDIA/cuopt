/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuts/cuts.hpp>
#include <cuts/greedy_knapsack.hpp>
#include <linear_algebra/dense_matrix.hpp>
#include <math_optimization/tic_toc.hpp>
#include <utilities/macros.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

using simplex::lp_problem_t;
using simplex::simplex_solver_settings_t;
using simplex::variable_type_t;

template <typename i_t, typename f_t>
knapsack_generation_t<i_t, f_t>::knapsack_generation_t(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types)
  : is_slack_(lp.num_cols, 0),
    is_complemented_(lp.num_cols, 0),
    is_marked_(lp.num_cols, 0),
    workspace_(lp.num_cols, 0.0),
    complemented_xstar_(lp.num_cols, 0.0),
    settings_(settings)
{
  const bool verbose = false;
  knapsack_constraints_.reserve(lp.num_rows);

  for (i_t j : new_slacks) {
    is_slack_[j] = 1;
  }

  for (i_t i = 0; i < lp.num_rows; i++) {
    inequality_t<i_t, f_t> inequality(Arow, i, lp.rhs[i]);
    inequality_t<i_t, f_t> rational_inequality = inequality;
    if (!rational_coefficients(var_types,
                               lp.lower,
                               lp.upper,
                               inequality_sense_t::LESS_EQUAL,
                               inequality,
                               rational_inequality)) {
      continue;
    }
    inequality = rational_inequality;

    const i_t row_len = rational_inequality.size();
    if (row_len < 3) { continue; }
    bool is_knapsack = true;
    f_t sum_pos      = 0.0;
    f_t sum_neg      = 0.0;
    for (i_t p = 0; p < row_len; p++) {
      const i_t j = inequality.index(p);
      if (is_slack_[j]) {
        if (inequality.coeff(p) < 0.0) {
          is_knapsack = false;
          break;
        }
        continue;
      }
      const f_t aj = inequality.coeff(p);
      if (var_types[j] != variable_type_t::INTEGER || lp.lower[j] != 0.0 || lp.upper[j] != 1.0) {
        is_knapsack = false;
        break;
      }
      if (aj < 0.0) {
        sum_pos += -aj;
        sum_neg += -aj;
      } else {
        sum_pos += aj;
      }
    }

    if (is_knapsack) {
      const f_t beta = inequality.rhs + sum_neg;
      if (beta > 0.0 && beta <= sum_pos && std::abs(sum_pos / (row_len - 1) - beta) > 1e-3) {
        if (verbose) {
          settings.log.printf(
            "Knapsack constraint %d row len %d beta %e sum_neg %e sum_pos %e sum_pos / (row_len - "
            "1) %e\n",
            i,
            row_len,
            beta,
            sum_neg,
            sum_pos,
            sum_pos / (row_len - 1));
        }
        knapsack_constraints_.push_back(i);
      }
    }
  }

#ifdef PRINT_KNAPSACK_INFO
  i_t num_knapsack_constraints = knapsack_constraints_.size();
  settings.log.printf("Number of knapsack constraints %d\n", num_knapsack_constraints);
#endif
}

template <typename i_t, typename f_t>
i_t knapsack_generation_t<i_t, f_t>::generate_knapsack_cut(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  i_t knapsack_row,
  inequality_t<i_t, f_t>& cut,
  f_t start_time)
{
  const bool verbose = false;
  // Get the row associated with the knapsack constraint
  inequality_t<i_t, f_t> knapsack_inequality(Arow, knapsack_row, lp.rhs[knapsack_row]);
  inequality_t<i_t, f_t> rational_knapsack_inequality = knapsack_inequality;
  if (!rational_coefficients(var_types,
                             lp.lower,
                             lp.upper,
                             inequality_sense_t::LESS_EQUAL,
                             knapsack_inequality,
                             rational_knapsack_inequality)) {
    return -1;
  }
  knapsack_inequality = rational_knapsack_inequality;

  // Given the following knapsack constraint with integer weights a_j and a real
  // capacity beta:
  // sum_j a_j * x_j <= beta
  //
  // The weight of a set C is given by C_w = sum_{j in C} a_j.
  // C is a cover if C_w > beta.
  //
  // We solve the following separation problem:
  // minimize   sum_j (1 - xstar_j) z_j
  // subject to sum_j a_j z_j > beta
  //            z_j in {0, 1}
  // When z_j = 1, then j is in the cover.
  // Let phi_star be the optimal objective of this problem.
  // We have a violated cover when phi_star < 1.0.
  //
  // Let W = sum_j a_j be the total weight. We have
  //
  //   W = C_w + sum_{j not in C} a_j,
  //
  // so the cover condition is
  //
  //   C_w = W - sum_{j not in C} a_j > beta.
  //
  // In order for this separation problem to be a 0-1 knapsack problem, we must
  // replace the strict greater-than (>) inequality with a greater-than-or-equal
  // (>=) inequality.
  // Since the weights are integers, C_w is also an integer.
  // The smallest integer strictly greater than beta is floor(beta) + 1.
  // For example, beta = 3 and beta = 3.2 both require a cover weight of at least 4.
  // Thus the cover condition is equivalent to
  //
  //   C_w = W - sum_{j not in C} a_j >= floor(beta) + 1.
  //
  // Rearranging gives
  //
  //   sum_{j not in C} a_j <= W - (floor(beta) + 1).
  //
  // Let zbar_j = 1 - z_j. Then sum_j a_j * zbar_j = sum_{j not in C} a_j,
  // so we obtain the following 0-1 knapsack problem:
  // maximize   sum_j (1 - xstar_j) zbar_j
  // subject to sum_j a_j zbar_j <= W - (floor(beta) + 1)
  //            zbar_j in {0, 1}
  //
  // This problem is in the form of a 0-1 knapsack problem
  // which we can solve with dynamic programming or generate
  // a heuristic solution with a greedy algorithm.

  // Remove the slacks from the inequality
  f_t seperation_rhs = 0.0;
  if (verbose) { settings.log.printf(" Knapsack : "); }
  std::vector<i_t> complemented_variables;
  complemented_variables.reserve(knapsack_inequality.size());
  for (i_t k = 0; k < knapsack_inequality.size(); k++) {
    const i_t j = knapsack_inequality.index(k);
    if (is_slack_[j]) {
      knapsack_inequality.vector.x[k] = 0.0;
    } else {
      const f_t aj = knapsack_inequality.vector.x[k];
      if (aj < 0.0) {
        knapsack_inequality.rhs -= aj;
        knapsack_inequality.vector.x[k] *= -1.0;
        complemented_variables.push_back(j);
        is_complemented_[j] = 1;
      }
      if (verbose) { settings.log.printf(" %g x%d +", knapsack_inequality.vector.x[k], j); }
      seperation_rhs += knapsack_inequality.vector.x[k];
    }
  }
  if (verbose) { settings.log.printf(" <= %g\n", knapsack_inequality.rhs); }
  // seperation_rhs currently contains W.
  seperation_rhs -= (std::floor(knapsack_inequality.rhs) + 1.0);

  if (verbose) {
    settings.log.printf("\t");
    for (i_t k = 0; k < knapsack_inequality.size(); k++) {
      const i_t j = knapsack_inequality.index(k);
      if (!is_slack_[j]) {
        if (std::abs(xstar[j]) > 1e-3) { settings.log.printf("x_relax[%d]= %g ", j, xstar[j]); }
      }
    }
    settings.log.printf("\n");

    settings.log.printf("seperation_rhs %g\n", seperation_rhs);
  }

  if (seperation_rhs < 0.0) {
    restore_complemented(complemented_variables);
    return -1;
  }

  std::vector<f_t> values;
  values.reserve(knapsack_inequality.size() - 1);
  std::vector<f_t> weights;
  weights.reserve(knapsack_inequality.size() - 1);
  std::vector<i_t> indices;
  indices.reserve(knapsack_inequality.size() - 1);
  f_t objective_constant = 0.0;
  std::vector<i_t> fixed_variables;
  std::vector<f_t> fixed_values;
  std::vector<f_t> fixed_weights;
  const f_t x_tol = 1e-5;
  for (i_t k = 0; k < knapsack_inequality.size(); k++) {
    const i_t j = knapsack_inequality.index(k);
    if (!is_slack_[j]) {
      const f_t xstar_j      = is_complemented_[j] ? 1.0 - xstar[j] : xstar[j];
      complemented_xstar_[j] = xstar_j;
      const f_t vj           = std::min(1.0, std::max(0.0, 1.0 - xstar_j));
      if (xstar_j < x_tol) {
        // if xstar_j is close to 0, then we can fix z to zero
        fixed_variables.push_back(j);
        fixed_values.push_back(0.0);
        fixed_weights.push_back(knapsack_inequality.vector.x[k]);
        seperation_rhs -= knapsack_inequality.vector.x[k];
        // No need to adjust the objective constant
        continue;
      }
      if (xstar_j > 1.0 - x_tol) {
        // if xstar_j is close to 1, then we can fix z to 1
        fixed_variables.push_back(j);
        fixed_values.push_back(1.0);
        fixed_weights.push_back(knapsack_inequality.vector.x[k]);
        // Note seperation rhs is unchanged
        objective_constant += vj;
        continue;
      }
      objective_constant += vj;
      values.push_back(vj);
      weights.push_back(knapsack_inequality.vector.x[k]);
      indices.push_back(j);
    }
  }
  std::vector<f_t> solution;
  solution.resize(values.size());

  if (seperation_rhs < 0.0) {
    restore_complemented(complemented_variables);
    return -1;
  }

  f_t objective = 0.0;
  if (!values.empty()) {
    if (verbose) { settings.log.printf("Calling solve_knapsack_problem\n"); }

    objective = solve_knapsack_problem(values, weights, seperation_rhs, solution);
  } else {
    solution.clear();
  }
  if (std::isnan(objective)) {
    restore_complemented(complemented_variables);
    return -1;
  }
  if (verbose) {
    settings.log.printf("objective %e objective_constant %e\n", objective, objective_constant);
  }
  f_t seperation_value = -objective + objective_constant;
  if (verbose) { settings.log.printf("seperation_value %e\n", seperation_value); }
  const f_t tol = 1e-6;
  if (seperation_value >= 1.0 - tol) {
    restore_complemented(complemented_variables);
    return -1;
  }

  i_t cover_size   = 0;
  f_t cover_weight = 0.0;
  for (i_t k = 0; k < solution.size(); k++) {
    if (solution[k] == 0.0) {
      cover_size++;
      cover_weight += weights[k];
    }
  }
  for (i_t k = 0; k < fixed_values.size(); k++) {
    if (fixed_values[k] == 1.0) {
      cover_size++;
      cover_weight += fixed_weights[k];
    }
  }

  // sum_{j in C} a_j > beta is what makes sum_{j in C} x_j <= |C| - 1 valid.
  const bool is_cover = cover_weight > knapsack_inequality.rhs;
  cuopt_assert(is_cover, "knapsack separation produced a set that is not a cover");
  if (!is_cover) {
    restore_complemented(complemented_variables);
    return -1;
  }

  cut.reserve(cover_size);
  cut.clear();

  for (i_t k = 0; k < solution.size(); k++) {
    const i_t j = indices[k];
    if (solution[k] == 0.0) { cut.push_back(j, -1.0); }
  }
  for (i_t k = 0; k < fixed_variables.size(); k++) {
    const i_t j = fixed_variables[k];
    if (fixed_values[k] == 1.0) { cut.push_back(j, -1.0); }
  }
  cut.rhs = -cover_size + 1;

  // The cut is in the form: - sum_{j in cover} x_j >= -cover_size + 1
  // Which is equivalent to: sum_{j in cover} x_j <= cover_size - 1

  // Compute the minimal cover and partition the variables into C1 and C2
  inequality_t<i_t, f_t> minimal_cover_cut(lp.num_cols);
  std::vector<i_t> c1_partition;
  std::vector<i_t> c2_partition;
  minimal_cover_and_partition(
    knapsack_inequality, cut, complemented_xstar_, minimal_cover_cut, c1_partition, c2_partition);

  // Lift the cut
  inequality_t<i_t, f_t> lifted_cut(lp.num_cols);
  lift_knapsack_cut(
    knapsack_inequality, minimal_cover_cut, c1_partition, c2_partition, lifted_cut, start_time);
  lifted_cut.negate();

  // The cut is now in the form:
  // -\sum_{j in C} x_j - \sum_{j in F} alpha_j x_j >= -cover_size + 1
  for (i_t k = 0; k < lifted_cut.size(); k++) {
    const i_t j = lifted_cut.index(k);
    // \sum_{k!=j} d_k x_k + d_j xbar_j >= gamma
    // xbar_j = 1 - x_j
    // \sum_{k!=j} d_k x_k + d_j (1 - x_j) >= gamma
    // \sum_{k!=j} d_k x_k + d_j - d_j x_j >= gamma
    // \sum_{k!=j} d_k x_k  + d_j x_j >= gamma - d_j
    if (is_complemented_[j]) {
      lifted_cut.rhs -= lifted_cut.vector.x[k];
      lifted_cut.vector.x[k] *= -1.0;
    }
  }
  lifted_cut.sort();

  // Verify the cut is violated
  f_t lifted_dot       = lifted_cut.vector.dot(xstar);
  f_t lifted_violation = lifted_dot - lifted_cut.rhs;
  if (verbose) {
    settings.log.printf(
      "Knapsack cut %d lifted violation %e < 0\n", knapsack_row, lifted_violation);
  }

  if (lifted_violation >= -tol) {
    restore_complemented(complemented_variables);
    return -1;
  }

  cut = lifted_cut;
  restore_complemented(complemented_variables);
  return 0;
}

template <typename i_t, typename f_t>
bool knapsack_generation_t<i_t, f_t>::is_minimal_cover(f_t cover_sum,
                                                       f_t beta,
                                                       const std::vector<f_t>& cover_coefficients)
{
  // Check if the cover is minimial
  // A set C is a cover if
  // sum_{j in C} a_j > beta
  // A set C is a minimal cover if
  // sum_{k in C \ {j}} a_k <= beta for all j in C
  bool minimal = true;

  // cover_sum = sum_{j in C} a_j

  // A cover is minimal if cover_sum - a_j <= beta for all j in C

  for (i_t k = 0; k < cover_coefficients.size(); k++) {
    const f_t a_j = cover_coefficients[k];
    if (a_j == 0.0) { continue; }
    if (cover_sum - a_j > beta) {
      minimal = false;
      break;
    }
  }
  return minimal;
}

template <typename i_t, typename f_t>
void knapsack_generation_t<i_t, f_t>::minimal_cover_and_partition(
  const inequality_t<i_t, f_t>& knapsack_inequality,
  const inequality_t<i_t, f_t>& negated_base_cut,
  const std::vector<f_t>& xstar,
  inequality_t<i_t, f_t>& minimal_cover_cut,
  std::vector<i_t>& c1_partition,
  std::vector<i_t>& c2_partition)
{
  // Compute the minimal cover cut
  inequality_t<i_t, f_t> base_cut = negated_base_cut;
  base_cut.negate();

  std::vector<i_t> cover_indicies;
  cover_indicies.reserve(base_cut.size());

  std::vector<f_t> cover_coefficients;
  cover_coefficients.reserve(base_cut.size());

  std::vector<f_t> score;
  score.reserve(base_cut.size());

  for (i_t k = 0; k < knapsack_inequality.size(); k++) {
    const i_t j   = knapsack_inequality.index(k);
    workspace_[j] = knapsack_inequality.coeff(k);
  }

  for (i_t k = 0; k < base_cut.size(); k++) {
    const i_t j       = base_cut.index(k);
    const f_t xstar_j = xstar[j];
    score.push_back((1.0 - xstar_j) / workspace_[j]);
    cover_indicies.push_back(j);
    cover_coefficients.push_back(workspace_[j]);
  }

  f_t cover_sum = std::accumulate(cover_coefficients.begin(), cover_coefficients.end(), 0.0);

  bool is_minimal = is_minimal_cover(cover_sum, knapsack_inequality.rhs, cover_coefficients);

  if (is_minimal) {
    minimal_cover_cut = base_cut;
    return;
  }

  // We don't have a minimal cover. So sort the score from smallest to largest breaking ties by
  // largest to smallest a_j
  std::vector<i_t> permuation(cover_indicies.size());
  std::iota(permuation.begin(), permuation.end(), 0);
  std::sort(permuation.begin(), permuation.end(), [&](i_t a, i_t b) {
    if (score[a] < score[b]) {
      return true;
    } else if (score[a] == score[b]) {
      return cover_coefficients[a] > cover_coefficients[b];
    } else {
      return false;
    }
  });

  const f_t beta = knapsack_inequality.rhs;
  for (i_t k = 0; k < permuation.size(); k++) {
    const i_t h   = permuation[k];
    const f_t a_j = cover_coefficients[h];
    if (cover_sum - a_j > beta) {
      // sum_{k in C} a_k - a_j > beta
      // so sum_{k in C \ {k}} a_k > beta and C \ {k} remains a cover

      cover_sum -= a_j;
      // Set the coefficient to 0 to remove it from the cover
      cover_coefficients[h] = 0.0;

      is_minimal = is_minimal_cover(cover_sum, beta, cover_coefficients);
      if (is_minimal) { break; }
    } else {
      // C \ {j} is no longer a cover.
      continue;
    }
  }

  cuopt_assert(cover_sum > beta, "minimal cover reduction dropped an item the cover needed");

  // Go through and correct cover_indicies and cover_coefficients
  for (i_t k = 0; k < cover_coefficients.size();) {
    if (cover_coefficients[k] == 0.0) {
      cover_indicies[k] = cover_indicies.back();
      cover_indicies.pop_back();
      cover_coefficients[k] = cover_coefficients.back();
      cover_coefficients.pop_back();
    } else {
      k++;
    }
  }

  // We now have a minimal cover cut
  // sum_{j in C} x_j <= |C| - 1
  minimal_cover_cut.vector.i = cover_indicies;
  minimal_cover_cut.vector.x.resize(cover_indicies.size(), 1.0);
  minimal_cover_cut.rhs = cover_coefficients.size() - 1;

  // Now we need to partition the variables into C1 and C2
  // C2 = {j in C | x_j = 1}
  // C1 = C / C2

  const f_t x_tol = 1e-5;
  for (i_t j : cover_indicies) {
    if (xstar[j] > 1.0 - x_tol) {
      c2_partition.push_back(j);
    } else {
      c1_partition.push_back(j);
    }
  }
}

template <typename i_t, typename f_t>
void knapsack_generation_t<i_t, f_t>::lift_knapsack_cut(
  const inequality_t<i_t, f_t>& knapsack_inequality,
  const inequality_t<i_t, f_t>& base_cut,
  const std::vector<i_t>& c1_partition,
  const std::vector<i_t>& c2_partition,
  inequality_t<i_t, f_t>& lifted_cut,
  f_t start_time)
{
  // The base cut is in the form: sum_{j in cover} x_j <= |cover| - 1

  // We will attempt to lift the cut by adding a new variable x_k with k not in C to the base cut
  // so that the cut becomes
  // sum_{j in cover} x_j + alpha_k * x_k <= |cover| - 1
  //
  // We can do this for multiple variables so that in the end the cut becomes
  //
  // sum_{j in cover} x_j + sum_{k in F} alpha_k * x_k <= |cover| - 1

  // Determine which variables are in the knapsack constraint and not in the cover
  std::vector<i_t> marked_variables;
  marked_variables.reserve(knapsack_inequality.size());
  for (i_t k = 0; k < knapsack_inequality.size(); k++) {
    const i_t j = knapsack_inequality.index(k);
    if (!is_marked_[j]) {
      is_marked_[j] = 1;  // is_marked_[j] = 1 for all j in N
      marked_variables.push_back(j);
    }
  }
  for (i_t k = 0; k < base_cut.size(); k++) {
    const i_t j = base_cut.index(k);
    if (is_marked_[j]) {
      is_marked_[j] = 0;  // is_marked_[j] = 1 for all j in N \ C
      // OK to leave marked_variables unchanged as marked_variables will be a superset of all dirty
      // is_marked
    }
  }
  std::vector<i_t> remaining_variables;
  std::vector<i_t> remaining_indices;
  std::vector<f_t> remaining_coefficients;
  remaining_variables.reserve(knapsack_inequality.size());
  remaining_indices.reserve(knapsack_inequality.size());
  remaining_coefficients.reserve(knapsack_inequality.size());

  for (i_t k = 0; k < knapsack_inequality.size(); k++) {
    const i_t j = knapsack_inequality.index(k);
    if (is_marked_[j]) {
      if (is_slack_[j]) { continue; }
      remaining_variables.push_back(j);
      remaining_indices.push_back(k);
      remaining_coefficients.push_back(knapsack_inequality.coeff(k));
    }
  }

  // We start with F = {} and lift remaining variables one by one
  // For a variable k not in C union F, the inequality
  //
  // alpha_k * x_k + sum_{j in C} x_j <= |C| - 1
  // is trivially satisfied when x_k = 0.
  // If x_k = 1, then the inequality will be valid for all alpha_k
  // such that
  // alpha_k +  maximize sum_{j in C} x_j                          <= |C| - 1
  //            subject to a_k + sum_{j in C} a_j x_j <= beta
  //
  // where here we require a_k + sum_{j in C} a_j x_j <= beta so that the inequality
  // is valid for the set { x_j in {0, 1}^(|C| + 1) | sum_{j in C union k} a_j x_j <= beta}
  //
  // Let phi^star_k denote the optimal objective value of the problem
  //
  // maximize sum_{j in C} x_j
  // subject to a_k + sum_{j in C} a_j x_j <= beta
  //             x_j in {0, 1} for all j in C
  // Then alpha_k <= |C| - 1 - phi^star_k
  // and we can set alpha_k = |C| - 1 - phi^star_k
  //
  // We can continue this process for each variable k not in C union F
  //
  // Assume the valid inequality
  // sum_{j in C} x_j + sum_{j in F} alpha_j * x_j <= |C| - 1
  // has been obtained so far. We now add the variable x_k with k not in C union F to the
  // inequality. So we have alpha_k * x_k + sum_{j in C} x_j + sum_{j in F} alpha_j * x_j <= |C| - 1
  //
  // Again, this is trivially satisfied when x_k = 0. And we can determine the max value of alpha_k
  // by solving the 0-1 knapsack problem:
  //
  // maximize sum_{j in C} x_j + sum_{j in F} alpha_j * x_j
  // subject to sum_{j in C} a_j x_j + sum_{j in F} a_j * x_j <= beta - a_k
  //            x_j in {0, 1} for all j in C union F
  //
  // Let phi^star_k denote the optimal objective value of the knapsack problem.
  // The lifted coefficient alpha_k = |C| - 1 - phi^star_k

  // Construct weight and values for C
  std::vector<i_t> values;
  values.reserve(knapsack_inequality.size());

  std::vector<f_t> weights;
  weights.reserve(knapsack_inequality.size());

  for (i_t k = 0; k < knapsack_inequality.size(); k++) {
    const i_t j = knapsack_inequality.index(k);
    if (!is_marked_[j]) {
      // j is in C
      weights.push_back(knapsack_inequality.coeff(k));
      values.push_back(1);
    }
  }

  std::vector<i_t> F;
  std::vector<f_t> alpha;

  std::vector<f_t> solution;

  f_t cover_size = base_cut.size();

  lifted_cut = base_cut;

  // Sort the coefficients such that the largest coefficients are lifted first
  // We will pop the largest coefficients from the back of the permutation
  std::vector<i_t> permutation;
  best_score_last_permutation(remaining_coefficients, permutation);

  while (permutation.size() > 0) {
    if (toc(start_time) >= settings_.time_limit) { break; }
    const i_t h   = permutation.back();
    const i_t k   = remaining_variables[h];
    const f_t a_k = remaining_coefficients[h];

    f_t capacity = knapsack_inequality.rhs - a_k;

    f_t objective = exact_knapsack_problem_integer_values_fraction_values(
      values, weights, capacity, solution, start_time);
    if (std::isnan(objective)) {
      settings_.log.debug("lifting knapsack problem failed\n");
      break;
    }

    f_t alpha_k = std::max(0.0, cover_size - 1.0 - objective);

    if (alpha_k > 0.0) {
      settings_.log.debug("Lifted variable %d with alpha %g\n", k, alpha_k);
      F.push_back(k);
      alpha.push_back(alpha_k);
      values.push_back(static_cast<i_t>(std::round(alpha_k)));
      weights.push_back(a_k);

      lifted_cut.vector.i.push_back(k);
      lifted_cut.vector.x.push_back(alpha_k);
    }

    // Remove the variable from the permutation
    permutation.pop_back();
  }
  // Restore is_marked_
  for (i_t j : marked_variables) {
    is_marked_[j] = 0;
  }
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
    return greedy_knapsack_problem<i_t, f_t>(values, weights, rhs, solution);
  }

  solution.assign(n, 0.0);

  // dp(j, v) = minimum weight using first j items to get value v.
  // The weights are rounded up: rounding one down would let the dynamic programming algorithm
  // return a set that violates the capacity, and the caller reads the complement of that set
  // as a cover.
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
        i_t candidate = dp(j - 1, v - scaled_values[j - 1]) + std::ceil(weights[j - 1]);
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

#ifdef ASSERT_MODE
  f_t selected_weight = 0.0;
  for (i_t j = 0; j < n; ++j) {
    selected_weight += solution[j] * weights[j];
  }
  cuopt_assert(selected_weight <= rhs + settings_.primal_tol,
               "knapsack dynamic program returned a solution over capacity");
#endif

  objective = best_value * scale;
  return objective;
}

template <typename i_t, typename f_t>
f_t knapsack_generation_t<i_t, f_t>::exact_knapsack_problem_integer_values_fraction_values(
  const std::vector<i_t>& values,
  const std::vector<f_t>& weights,
  f_t rhs,
  std::vector<f_t>& solution,
  f_t start_time)
{
  if (toc(start_time) >= settings_.time_limit) { return std::numeric_limits<f_t>::quiet_NaN(); }
  // Solve the knapsack problem
  // maximize sum_{j=0}^n values[j] * solution[j]
  // subject to sum_{j=0}^n weights[j] * solution[j] <= rhs
  // values: values of the items
  // weights: weights of the items
  // return the value of the solution

  const i_t n = weights.size();

  const bool verbose = false;
  i_t sum_value      = std::accumulate(values.begin(), values.end(), 0);
  if (verbose) { settings_.log.printf("sum value %d\n", sum_value); }
  const i_t max_size = 10000;
  if (sum_value <= 0.0 || sum_value >= max_size) {
    if (verbose) { settings_.log.printf("sum value %d is negative or too large\n", sum_value); }
    return std::numeric_limits<f_t>::quiet_NaN();
  }

  solution.assign(n, 0.0);

  // dp(j, v) = minimum weight using first j items to get value v
  dense_matrix_t<i_t, f_t> dp(n + 1, sum_value + 1, inf);
  dense_matrix_t<i_t, uint8_t> take(n + 1, sum_value + 1, 0);
  dp(0, 0) = 0;

  // 4. Dynamic programming
  for (i_t j = 1; j <= n; ++j) {
    if (toc(start_time) >= settings_.time_limit) { return std::numeric_limits<f_t>::quiet_NaN(); }
    for (i_t v = 0; v <= sum_value; ++v) {
      // Do not take item i-1
      dp(j, v) = dp(j - 1, v);

      // Take item j-1 if possible
      if (v >= values[j - 1]) {
        f_t candidate = dp(j - 1, v - values[j - 1]) + weights[j - 1];
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
      v -= values[j - 1];
    } else {
      solution[j - 1] = 0.0;
    }
  }

  return f_t(best_value);
}

template <typename i_t, typename f_t>
void cut_generation_t<i_t, f_t>::generate_knapsack_cuts(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  f_t start_time)
{
  if (knapsack_generation_.num_knapsack_constraints() > 0) {
    for (i_t knapsack_row : knapsack_generation_.get_knapsack_constraints()) {
      if (toc(start_time) >= settings.time_limit) { return; }
      inequality_t<i_t, f_t> cut(lp.num_cols);
      i_t knapsack_status = knapsack_generation_.generate_knapsack_cut(
        lp, settings, Arow, new_slacks, var_types, xstar, knapsack_row, cut, start_time);
      if (knapsack_status == 0) { cut_pool_.add_cut(cut_type_t::KNAPSACK, cut); }
    }
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class knapsack_generation_t<int, double>;

template void cut_generation_t<int, double>::generate_knapsack_cuts(
  const lp_problem_t<int, double>& lp,
  const simplex_solver_settings_t<int, double>& settings,
  csr_matrix_t<int, double>& Arow,
  const std::vector<int>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  const std::vector<double>& xstar,
  double start_time);
#endif

}  // namespace cuopt::mathematical_optimization::mip
