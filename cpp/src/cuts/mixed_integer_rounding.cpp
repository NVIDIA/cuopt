/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuts/cuts.hpp>
#include <math_optimization/tic_toc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <queue>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

using simplex::lp_problem_t;
using simplex::simplex_solver_settings_t;
using simplex::variable_type_t;

template <typename i_t, typename f_t>
void cut_generation_t<i_t, f_t>::generate_mir_cuts(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& xstar,
  const std::vector<f_t>& ystar,
  variable_bounds_t<i_t, f_t>& variable_bounds,
  f_t start_time)
{
  f_t mir_start_time     = tic();
  constexpr bool verbose = false;
  complemented_mixed_integer_rounding_cut_t<i_t, f_t> complemented_mir(lp, settings, new_slacks);
  strong_cg_cut_t<i_t, f_t> cg(lp, var_types, xstar);

  std::vector<f_t> scores;
  complemented_mir.compute_initial_scores_for_rows(lp, settings, Arow, xstar, ystar, scores);

  // Push all the scores onto the priority queue
  std::priority_queue<std::pair<f_t, i_t>> score_queue;
  for (i_t i = 0; i < lp.num_rows; i++) {
    score_queue.push(std::make_pair(scores[i], i));
  }

  // These data structures are used to track the rows that have been aggregated
  // The invariant is that aggregated_rows is empty and aggregated_mark is all zeros
  // at the beginning of each iteration of the for loop below
  std::vector<i_t> aggregated_rows;
  std::vector<i_t> aggregated_mark(lp.num_rows, 0);
  const i_t max_cuts = std::min(lp.num_rows, 100000);

  // Transform the relaxation solution
  std::vector<f_t> transformed_xstar;
  complemented_mir.bound_substitution(lp, variable_bounds, var_types, xstar, transformed_xstar);

  f_t work_estimate  = 0.0;
  i_t cuts_processed = 0;
  while (cuts_processed < max_cuts && !score_queue.empty()) {
    if (toc(start_time) >= settings.time_limit) { break; }
    // Get the row with the highest score from the queue
    auto [max_score, i] = score_queue.top();
    score_queue.pop();
    // skip stale score entries
    if (max_score != scores[i]) { continue; }
    ++cuts_processed;

    // Add the current row to the aggregated set
    aggregated_mark[i] = 1;
    aggregated_rows.push_back(i);

    const i_t row_nz      = Arow.row_length(i);
    const i_t slack       = complemented_mir.slack_cols(i);
    const f_t slack_value = xstar[slack];

    if (max_score <= 0.0) { break; }
    if (work_estimate > 2e9) { break; }

    inequality_t<i_t, f_t> inequality(Arow, i, lp.rhs[i]);
    work_estimate += inequality.size();

    const bool generate_cg_cut = settings.strong_chvatal_gomory_cuts != 0;
    f_t fractional_part_rhs    = fractional_part(inequality.rhs);
    if (generate_cg_cut && fractional_part_rhs > 1e-6 && fractional_part_rhs < (1 - 1e-6)) {
      // Try to generate a CG cut

      inequality_t<i_t, f_t> cg_inequality = inequality;
      if (fractional_part(inequality.rhs) < 0.5) {
        // Multiply by -1 to force the fractional part to be greater than 0.5
        cg_inequality.negate();
      }
      inequality_t<i_t, f_t> cg_cut;
      i_t cg_status =
        cg.generate_strong_cg_cut(lp, settings, var_types, cg_inequality, xstar, cg_cut);
      if (cg_status == 0) { cut_pool_.add_cut(cut_type_t::CHVATAL_GOMORY, cg_cut); }
    }

    if (settings.mir_cuts == 0) { continue; }

    // Remove the slack from the equality to get an inequality
    work_estimate += inequality.size();
    i_t negate_inequality = 1;
    for (i_t k = 0; k < inequality.size(); k++) {
      const i_t j = inequality.index(k);
      if (j == slack) {
        if (inequality.coeff(k) != 1.0) {
          if (inequality.coeff(k) == -1.0 && lp.lower[j] >= 0.0) {
            negate_inequality = 0;
          } else {
            settings.log.debug("Bad slack %d in inequality: aj %e lo %e up %e\n",
                               j,
                               inequality.coeff(k),
                               lp.lower[j],
                               lp.upper[j]);
            negate_inequality = -1;
            break;
          }
        }
        inequality.vector.x[k] = 0.0;
      }
    }

    if (negate_inequality == -1) { continue; }

    if (negate_inequality) {
      // inequaility'*x <= inequality_rhs
      // But for MIR we need: inequality'*x >= inequality_rhs
      inequality.negate();
      work_estimate += inequality.size();
    }
    // We should now have: inequality'*x >= inequality_rhs

    for (i_t k = 0; k < inequality.size(); k++) {
      const i_t j = inequality.index(k);
      if (var_types[j] == variable_type_t::INTEGER) {
        if (transformed_xstar[j] > complemented_mir.new_upper(j) / 2.0) {
          settings.log.printf("!!!!!! j %d transformed x_j %e new_upper_j/2.0 %e\n",
                              j,
                              transformed_xstar[j],
                              complemented_mir.new_upper(j) / 2.0);
        }
      }
    }

    bool add_cut             = false;
    i_t num_aggregated       = 0;
    const i_t max_aggregated = 6;
    f_t min_abs_multiplier   = 1.0;
    f_t max_abs_multiplier   = 1.0;
    work_estimate += lp.num_cols;

    while (!add_cut && num_aggregated < max_aggregated) {
      inequality_t<i_t, f_t> transformed_inequality;
      inequality.squeeze(transformed_inequality);
      work_estimate += transformed_inequality.size();

      complemented_mir.transform_inequality(variable_bounds, var_types, transformed_inequality);
      work_estimate += transformed_inequality.size();

      inequality_t<i_t, f_t> cut;
      bool cut_found = complemented_mir.cut_generation_heuristic(transformed_inequality,
                                                                 var_types,
                                                                 transformed_xstar,
                                                                 cut,
                                                                 work_estimate,
                                                                 start_time,
                                                                 settings.time_limit);
      if (toc(start_time) >= settings.time_limit) { return; }
      // Note cut is in the transformed variables

      if (cut_found) {
        // Transform back to the original variables
        complemented_mir.untransform_inequality(variable_bounds, var_types, cut);
        complemented_mir.remove_small_coefficients(lp.lower, lp.upper, cut);
        complemented_mir.substitute_slacks(lp, Arow, cut);
        complemented_mir.remove_small_coefficients(lp.lower, lp.upper, cut);
        f_t viol = complemented_mir.compute_violation(cut, xstar);
        work_estimate += 10 * cut.size();
        if (viol > 1e-6) { add_cut = true; }
      }

      if (add_cut) {
        if (settings.mir_cuts != 0) { cut_pool_.add_cut(cut_type_t::MIXED_INTEGER_ROUNDING, cut); }
        break;
      } else {
        // Perform aggregation to try and find a cut

        // Find all the continuous variables in the inequality
        i_t num_continuous    = 0;
        f_t max_off_bound     = 0.0;
        i_t max_off_bound_var = -1;
        for (i_t p = 0; p < inequality.size(); p++) {
          const i_t j  = inequality.index(p);
          const f_t aj = inequality.coeff(p);
          if (aj == 0.0) { continue; }
          if (var_types[j] == variable_type_t::CONTINUOUS) {
            num_continuous++;

            const f_t lb_star_j = complemented_mir.get_lb_star(j);
            const f_t ub_star_j = complemented_mir.get_ub_star(j);
            const f_t off_lower = lb_star_j > -inf ? xstar[j] - lb_star_j : std::abs(xstar[j]);
            const f_t off_upper = ub_star_j < inf ? ub_star_j - xstar[j] : std::abs(xstar[j]);
            const f_t off_bound = std::min(off_lower, off_upper);
            const i_t col_len   = lp.A.col_length(j);
            if (off_bound > max_off_bound && col_len > 1) {
              max_off_bound     = off_bound;
              max_off_bound_var = j;
            }
          }
        }
        work_estimate += 10 * inequality.size();

        if (num_continuous == 0 || max_off_bound < 1e-6) { break; }

        // The variable that is farthest from its bound is used as a pivot
        if (max_off_bound_var >= 0) {
          const i_t col_start          = lp.A.col_start[max_off_bound_var];
          const i_t col_end            = lp.A.col_start[max_off_bound_var + 1];
          const i_t col_len            = lp.A.col_length(max_off_bound_var);
          const i_t max_potential_rows = col_len;
          if (col_len > 1) {
            std::vector<i_t> potential_rows;
            potential_rows.reserve(col_len);

            const f_t threshold = 1e-4;
            for (i_t q = col_start; q < col_end; q++) {
              const i_t i   = lp.A.i[q];
              const f_t val = lp.A.x[q];
              // Can't use rows that have already been aggregated
              if (std::abs(val) > threshold && !aggregated_mark[i]) { potential_rows.push_back(i); }
              if (potential_rows.size() >= max_potential_rows) { break; }
            }
            work_estimate += 5 * (col_end - col_start);

            bool did_aggregate = false;
            while (!potential_rows.empty()) {
              const i_t pivot_row =
                *std::max_element(potential_rows.begin(), potential_rows.end(), [&](i_t a, i_t b) {
                  return scores[a] < scores[b];
                });
              work_estimate += potential_rows.size();

              inequality_t<i_t, f_t> pivot_row_inequality(Arow, pivot_row, lp.rhs[pivot_row]);
              work_estimate += pivot_row_inequality.size();
              // Save inequality before combine_rows mutates it, so we can restore on rejection
              inequality_t<i_t, f_t> saved_inequality = inequality;
              f_t multiplier                          = complemented_mir.combine_rows(
                lp, Arow, max_off_bound_var, pivot_row_inequality, inequality);
              if (max_abs_multiplier / std::abs(multiplier) > 10000 ||
                  std::abs(multiplier) / min_abs_multiplier > 10000) {
                inequality = saved_inequality;
                // Erase the pivot row from the potential rows
                potential_rows.erase(
                  std::remove(potential_rows.begin(), potential_rows.end(), pivot_row),
                  potential_rows.end());
                continue;
              }
              max_abs_multiplier = std::max(max_abs_multiplier, std::abs(multiplier));
              min_abs_multiplier = std::min(min_abs_multiplier, std::abs(multiplier));
              aggregated_rows.push_back(pivot_row);
              aggregated_mark[pivot_row] = 1;
              work_estimate += inequality.size() + pivot_row_inequality.size();
              did_aggregate = true;
              break;
            }

            if (!did_aggregate) {
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

      // Set the score of the aggregated rows to a lower value
      for (i_t row : aggregated_rows) {
        scores[row] = 0.99 * scores[row];
        score_queue.push(std::make_pair(scores[row], row));
      }
      work_estimate += aggregated_rows.size() * std::log2(score_queue.size());
    }

    // Clear the aggregated mark
    work_estimate += 2 * aggregated_rows.size();
    for (i_t row : aggregated_rows) {
      aggregated_mark[row] = 0;
    }
    // Clear the aggregated rows
    aggregated_rows.clear();

    scores[i] = 0.0;
    score_queue.push(std::make_pair(scores[i], i));
    work_estimate += std::log2(std::max(1, static_cast<i_t>(score_queue.size())));
  }
}

template <typename i_t, typename f_t>
complemented_mixed_integer_rounding_cut_t<i_t, f_t>::complemented_mixed_integer_rounding_cut_t(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const std::vector<i_t>& new_slacks)
  : is_slack_(lp.num_cols, 0),
    slack_rows_(lp.num_cols, -1),
    slack_cols_(lp.num_rows, -1),
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
    slack_cols_[i]      = j;
    assert(std::abs(lp.A.x[col_start]) == 1.0);
  }
}

template <typename i_t, typename f_t>
void complemented_mixed_integer_rounding_cut_t<i_t, f_t>::compute_initial_scores_for_rows(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<f_t>& xstar,
  const std::vector<f_t>& ystar,
  std::vector<f_t>& scores)
{
  const bool verbose  = false;
  const i_t n         = lp.num_cols;
  const f_t obj_norm  = vector_norm2<i_t, f_t>(lp.objective);
  const f_t obj_denom = std::max(1.0, obj_norm);

  // Compute initial scores for all rows
  scores.resize(lp.num_rows, 0.0);
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

    const i_t slack = slack_cols_[i];
    assert(slack >= 0);
    const f_t slack_value = std::max(xstar[slack], 0.0);
    const f_t slack_denom = std::max(0.1, std::sqrt(row_norm));

    const f_t nz_weight    = 0.0001;
    const f_t dual_weight  = 1.0;
    const f_t slack_weight = 0.001;

    scores[i] = nz_weight * (1.0 - density) + dual_weight * std::max(dual / obj_denom, 0.0001) +
                slack_weight * (1.0 - slack_value / slack_denom);

    if (verbose) {
      settings.log.printf("Scores[%d] = %e density %.2f dual %e slack %e\n",
                          i,
                          scores[i],
                          density,
                          dual,
                          slack_value);
    }
  }
}

template <typename i_t, typename f_t>
bool complemented_mixed_integer_rounding_cut_t<i_t, f_t>::cut_generation_heuristic(
  const inequality_t<i_t, f_t>& transformed_inequality,
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& transformed_xstar,
  inequality_t<i_t, f_t>& transformed_cut,
  f_t& work_estimate,
  f_t start_time,
  f_t time_limit)
{
  std::vector<f_t> deltas_to_try;
  deltas_to_try.reserve(transformed_inequality.size());
  deltas_to_try.push_back(1.0);
  work_estimate += transformed_inequality.size();
  i_t num_integers = 0;
  f_t max_coeff    = 0.0;
  for (i_t k = 0; k < transformed_inequality.size(); k++) {
    const i_t j      = transformed_inequality.index(k);
    const f_t abs_aj = std::abs(transformed_inequality.coeff(k));
    if (var_types[j] == variable_type_t::INTEGER) {
      num_integers++;
      max_coeff                 = std::max(max_coeff, abs_aj);
      const f_t x_j             = transformed_xstar[j];
      const f_t new_upper_j     = new_upper(j);
      const f_t dist_upper      = new_upper_j - x_j;
      const f_t dist_lower      = x_j;
      const bool between_bounds = x_j > 1e-6 && (new_upper_j == inf || dist_upper > 0.0);
      if (between_bounds && abs_aj > 1e-6) { deltas_to_try.push_back(abs_aj); }
    }
  }
  if (max_coeff > 1e-6 && max_coeff != 1.0) {
    deltas_to_try.push_back(max_coeff);
    deltas_to_try.push_back(max_coeff + 1.0);
  }

  std::vector<i_t> complemented_indices;
  complemented_indices.reserve(num_integers);
  std::vector<f_t> distance_from_midpoint;
  distance_from_midpoint.reserve(num_integers);
  std::vector<i_t> integer_indices;
  integer_indices.reserve(num_integers);
  for (i_t k = 0; k < transformed_inequality.size(); k++) {
    const i_t j = transformed_inequality.index(k);
    if (var_types[j] == variable_type_t::INTEGER && new_upper(j) < inf) {
      const f_t x_j         = transformed_xstar[j];
      const f_t new_upper_j = new_upper(j);
      if (x_j > 1e-6 && new_upper_j < inf) {
        const f_t midpoint_j = new_upper_j / 2.0;
        distance_from_midpoint.push_back(x_j - midpoint_j);
        integer_indices.push_back(k);
      }
    }
  }

  std::vector<i_t> perm(integer_indices.size());
  best_score_first_permutation(distance_from_midpoint, perm);
  work_estimate +=
    integer_indices.size() > 0 ? integer_indices.size() * std::log2(integer_indices.size()) : 0;

  bool cut_found = false;

  inequality_t<i_t, f_t> complemented_inequality = transformed_inequality;
  work_estimate += 4 * transformed_inequality.size();

  f_t delta          = 0.0;
  f_t best_violation = 0.0;

  // Dense rows can spend minutes in these candidate searches. Poll in batches to
  // avoid a clock read for every candidate.
  constexpr size_t time_check_interval = 64;
  size_t candidates                    = 0;
  const auto time_limit_reached        = [&]() {
    return candidates++ % time_check_interval == 0 && toc(start_time) >= time_limit;
  };

  // First try without any complementation
  for (const f_t tmp_delta : deltas_to_try) {
    if (time_limit_reached()) { return false; }
    bool cut_ok = scale_uncomplement_and_generate_cut(var_types,
                                                      transformed_xstar,
                                                      complemented_indices,
                                                      complemented_inequality,
                                                      tmp_delta,
                                                      transformed_cut,
                                                      work_estimate);
    if (!cut_ok) { continue; }
    // Check if the cut is violated
    best_violation = compute_violation(transformed_cut, transformed_xstar);
    work_estimate += 4 * transformed_cut.size();
    if (best_violation > 1e-6) {
      cut_found = true;
      delta     = tmp_delta;
      break;
    }
  }

  if (!cut_found) {
    // Complement an integer variable
    for (const i_t idx : perm) {
      const i_t l = integer_indices[idx];
      const i_t j = complemented_inequality.index(l);
      // We have an integer variable x_j <= b_j
      // We create a new variable xbar_j such that
      // x_j + xbar_j = b_j
      // x_j = b_j - xbar_j, xbar_j = b_j - x_j
      //
      // The inequality
      // sum_{k != j} a_k x_k + a_j x_j >= beta
      // becomes
      // sum_{k != j} a_k x_k + a_j (b_j - xbar_j) >= beta
      // sum_{k != j} a_k x_k - a_j xbar_j >= beta - a_j b_j
      const f_t b_j = new_upper(j);
      const f_t a_j = complemented_inequality.coeff(l);

      complemented_inequality.vector.x[l] = -a_j;
      complemented_inequality.rhs -= a_j * b_j;
      complemented_indices.push_back(l);

      for (const f_t tmp_delta : deltas_to_try) {
        if (time_limit_reached()) { return false; }
        bool cut_ok = scale_uncomplement_and_generate_cut(var_types,
                                                          transformed_xstar,
                                                          complemented_indices,
                                                          complemented_inequality,
                                                          tmp_delta,
                                                          transformed_cut,
                                                          work_estimate);
        if (!cut_ok) { continue; }
        // Check if the cut is violated
        best_violation = compute_violation(transformed_cut, transformed_xstar);
        work_estimate += 4 * transformed_cut.size();
        if (best_violation > 1e-6) {
          cut_found = true;
          delta     = tmp_delta;
          break;
        }
      }
      if (cut_found) { break; }
    }
  }

  if (!cut_found) { return false; }

  // We have found a cut. Now try to improve the violation by scaling the cut by 1/2, 1/4, 1/8, etc.
  std::vector<f_t> scaled_deltas_to_try = {delta / 2.0, delta / 4.0, delta / 8.0};
  for (const f_t tmp_delta : scaled_deltas_to_try) {
    inequality_t<i_t, f_t> tmp_cut_delta;
    bool cut_ok = scale_uncomplement_and_generate_cut(var_types,
                                                      transformed_xstar,
                                                      complemented_indices,
                                                      complemented_inequality,
                                                      tmp_delta,
                                                      tmp_cut_delta,
                                                      work_estimate);
    if (!cut_ok) { continue; }

    // Check if the cut is violated
    f_t violation = compute_violation(tmp_cut_delta, transformed_xstar);
    work_estimate += 4 * tmp_cut_delta.size();
    if (violation > best_violation) {
      best_violation  = violation;
      transformed_cut = tmp_cut_delta;
      delta           = tmp_delta;
    }
  }

  std::vector<i_t> best_complemented_indices = complemented_indices;
  work_estimate += 2 * best_complemented_indices.size();

  // Try to improve the violation by complementing integer variables
  complemented_inequality = transformed_inequality;
  work_estimate += 4 * transformed_inequality.size();
  complemented_indices.clear();
  for (const i_t idx : perm) {
    if (time_limit_reached()) { return false; }
    const i_t l = integer_indices[idx];
    const i_t j = complemented_inequality.index(l);
    // We have an integer variable x_j <= b_j
    // We create a new variable xbar_j such that
    // x_j + xbar_j = b_j
    // x_j = b_j - xbar_j, xbar_j = b_j - x_j
    //
    // The inequality
    // sum_{k != j} a_k x_k + a_j x_j >= beta
    // becomes
    // sum_{k != j} a_k x_k + a_j (b_j - xbar_j) >= beta
    // sum_{k != j} a_k x_k - a_j xbar_j >= beta - a_j b_j
    const f_t b_j = new_upper(j);
    const f_t a_j = complemented_inequality.coeff(l);

    complemented_inequality.vector.x[l] = -a_j;
    complemented_inequality.rhs -= a_j * b_j;
    complemented_indices.push_back(l);

    inequality_t<i_t, f_t> tmp_cut_delta;

    bool cut_ok = scale_uncomplement_and_generate_cut(var_types,
                                                      transformed_xstar,
                                                      complemented_indices,
                                                      complemented_inequality,
                                                      delta,
                                                      tmp_cut_delta,
                                                      work_estimate);
    if (!cut_ok) { continue; }
    // Check if the cut is violated
    f_t violation = compute_violation(tmp_cut_delta, transformed_xstar);
    work_estimate += 4 * tmp_cut_delta.size();
    if (violation > best_violation) {
      best_violation            = violation;
      best_complemented_indices = complemented_indices;
      transformed_cut           = tmp_cut_delta;
    }
  }

  return best_violation > 1e-6;
}

template <typename i_t, typename f_t>
bool complemented_mixed_integer_rounding_cut_t<i_t, f_t>::scale_uncomplement_and_generate_cut(
  const std::vector<variable_type_t>& var_types,
  const std::vector<f_t>& transformed_xstar,
  const std::vector<i_t>& complemented_indices,
  const inequality_t<i_t, f_t>& complemented_inequality,
  f_t delta,
  inequality_t<i_t, f_t>& cut_delta,
  f_t& work_estimate)
{
  inequality_t scaled_inequality = complemented_inequality;
  if (delta != 1.0) { scaled_inequality.scale(1.0 / delta); }
  bool cut_ok = generate_cut_nonnegative_maintain_indicies(scaled_inequality, var_types, cut_delta);
  if (!cut_ok) { return false; }
  work_estimate += 4 * scaled_inequality.size();

  // Now we need to transform the complemented variables back
  for (i_t h = 0; h < complemented_indices.size(); h++) {
    const i_t l = complemented_indices[h];
    const i_t j = complemented_inequality.index(l);
    // Our cut is of the form
    // sum_{k != j} d_k x_k  + d_j xbar_j >= alpha
    // we have that xbar_j = b_j - x_j
    // So
    // sum_{k != j} d_k x_k  + d_j (b_j - x_j) >= alpha
    // Or
    // sum_{k != j} d_k x_k  - d_j x_j >= alpha - d_j b_j

    const f_t b_j         = new_upper(j);
    const f_t d_j         = cut_delta.coeff(l);
    cut_delta.vector.x[l] = -d_j;
    cut_delta.rhs -= d_j * b_j;
  }
  work_estimate += 5 * complemented_indices.size();
  return true;
}

template <typename i_t, typename f_t>
void complemented_mixed_integer_rounding_cut_t<i_t, f_t>::remove_small_coefficients(
  const std::vector<f_t>& lower_bounds,
  const std::vector<f_t>& upper_bounds,
  inequality_t<i_t, f_t>& cut)
{
  const i_t nz = cut.size();
  i_t removed  = 0;
  for (i_t k = 0; k < cut.size(); k++) {
    const i_t j = cut.index(k);

    // Check for small coefficients
    const f_t aj = cut.coeff(k);
    if (std::abs(aj) < 1e-6) {
      if (aj >= 0.0 && upper_bounds[j] < inf) {
        // Move this to the right-hand side
        cut.rhs -= aj * upper_bounds[j];
        cut.vector.x[k] = 0.0;
        removed++;
      } else if (aj <= 0.0 && lower_bounds[j] > -inf) {
        cut.rhs -= aj * lower_bounds[j];
        cut.vector.x[k] = 0.0;
        removed++;
        continue;
      } else {
        // We need to keep the coefficient
      }
    }
  }

  if (removed > 0) {
    inequality_t<i_t, f_t> new_cut(cut.vector.n);
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
  std::vector<f_t>& transformed_xstar,
  bool prefer_variable_bound_on_tie)
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
    const f_t lower_distance = xstar_j - lb_star_[j];
    const f_t upper_distance = ub_star_[j] - xstar_j;
    const bool prefer_upper_variable_bound =
      prefer_variable_bound_on_tie && ub_variable_[j] >= 0 && lower_distance == upper_distance;
    if (has_finite_lower_bound && (!has_finite_upper_bound || (lower_distance <= upper_distance &&
                                                               !prefer_upper_variable_bound))) {
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
  inequality_t<i_t, f_t>& inequality)
{
  const i_t nz = inequality.size();
  for (i_t k = 0; k < nz; k++) {
    const i_t j  = inequality.index(k);
    const f_t aj = inequality.coeff(k);
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
        inequality.rhs -= aj * lj;
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
        inequality.rhs -= aj * alpha;
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
        inequality.rhs -= aj * uj;
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
        inequality.rhs -= aj * alpha;
        scratch_pad_.add_to_pad(j, -aj);
        scratch_pad_.add_to_pad(i, aj * gamma);
      }
    } else if (bound_changed_[j] == 0) {
      scratch_pad_.add_to_pad(j, aj);
    }
  }
  scratch_pad_.get_pad(inequality.vector.i, inequality.vector.x);
  // At this point we have converted all the continuous variables to be nonnegative
  // Note that since continuous variables had VUB or VLB, they modified
  // the integer variables.

  // We clear the scratch pad. As it is no longer needed.
  scratch_pad_.clear_pad();

  // We now convert all the integer variables to be nonnegative
  const i_t nz_after = inequality.size();
  for (i_t k = 0; k < nz_after; k++) {
    const i_t j = inequality.index(k);
    if (var_type[j] != variable_type_t::INTEGER) { continue; }
    const f_t aj = inequality.coeff(k);
    if (bound_changed_[j] == -1) {
      // v_j = x_j - l_j, v_j >= 0
      // x_j = v_j + l_j
      // sum_{k != j} a_k x_k + a_j x_j >= beta
      // sum_{k != j} a_k x_k + a_j (v_j + l_j) >= beta
      // sum_{k != j} a_k x_k + a_j v_j >= beta - a_j l_j
      const f_t lj = lb_star_[j];
      inequality.rhs -= aj * lj;
    } else if (bound_changed_[j] == 1) {
      // w_j = u_j - x_j, w_j >= 0
      // x_j = u_j - w_j
      // sum_{k != j} a_k x_j + a_j x_j >= beta
      // sum_{k != j} a_k x_j + a_j (u_j - w_j) >= beta
      // sum_{k != j} a_k x_j - a_j w_j >= beta - a_j u_j
      const f_t uj = ub_star_[j];
      inequality.rhs -= aj * uj;
      inequality.vector.x[k] *= -1.0;
    }
  }
}

template <typename i_t, typename f_t>
void complemented_mixed_integer_rounding_cut_t<i_t, f_t>::untransform_inequality(
  const variable_bounds_t<i_t, f_t>& variable_bounds,
  const std::vector<variable_type_t>& var_type,
  inequality_t<i_t, f_t>& inequality)
{
  // First convert all the integers variables back to their original form: l_j <= x_j <= u_j
  const i_t nz = inequality.size();
  for (i_t k = 0; k < nz; k++) {
    const i_t j = inequality.index(k);
    if (var_type[j] != variable_type_t::INTEGER) { continue; }
    const f_t dj = inequality.coeff(k);
    if (bound_changed_[j] == -1) {
      // v_j = x_j - l_j, v_j >= 0
      // sum_{k != j} d_k x_k + d_j v_j >= beta
      // sum_{k != j} d_k x_k + d_j (x_j - l_j) >= beta
      // sum_{k != j} d_k x_k + d_j x_j >= beta + d_j l_j
      const f_t lj = lb_star_[j];
      inequality.rhs += dj * lj;
    } else if (bound_changed_[j] == 1) {
      // w_j = u_j - x_j, w_j >= 0
      // sum_{k != j} d_k x_k + d_j w_j >= beta
      // sum_{k != j} d_k x_k + d_j (u_j - x_j) >= beta
      // sum_{k != j} d_k x_k - d_j x_j  >= beta - d_j u_j
      const f_t uj = ub_star_[j];
      inequality.rhs -= dj * uj;
      inequality.vector.x[k] *= -1.0;
    }
  }
  // Then undo the VUB/VLB substitions and bring continuous variables back to their original form
  for (i_t k = 0; k < nz; k++) {
    const i_t j  = inequality.index(k);
    const f_t dj = inequality.coeff(k);
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
        inequality.rhs += dj * lj;
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
        inequality.rhs += dj * alpha;
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
        inequality.rhs -= dj * uj;
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
        inequality.rhs -= dj * alpha;
        scratch_pad_.add_to_pad(j, -dj);
        scratch_pad_.add_to_pad(i, dj * gamma);
      }
    } else {
      scratch_pad_.add_to_pad(j, dj);
    }
  }

  scratch_pad_.get_pad(inequality.vector.i, inequality.vector.x);
  scratch_pad_.clear_pad();
}

template <typename i_t, typename f_t>
bool complemented_mixed_integer_rounding_cut_t<i_t, f_t>::
  generate_cut_nonnegative_maintain_indicies(const inequality_t<i_t, f_t>& inequality,
                                             const std::vector<variable_type_t>& var_types,
                                             inequality_t<i_t, f_t>& cut)
{
  auto f = [](f_t q_1, f_t q_2) -> f_t {
    f_t q_1_hat = q_1 - std::floor(q_1);
    f_t q_2_hat = q_2 - std::floor(q_2);
    return std::min(q_1_hat, q_2_hat) + q_2_hat * std::floor(q_1);
  };

  auto h = [](f_t q) -> f_t { return std::max(q, 0.0); };

  cut.vector       = inequality.vector;
  const f_t beta   = inequality.rhs;
  const f_t f_beta = fractional_part(beta);
  cut.rhs          = f_beta * std::ceil(beta);
  if (f_beta < 0.05 || f_beta > 0.95) { return false; }

  for (i_t k = 0; k < inequality.size(); k++) {
    const i_t j = inequality.index(k);
    f_t aj      = inequality.coeff(k);
    if (var_types[j] == variable_type_t::INTEGER) {
      cut.vector.x[k] = f(aj, beta);
    } else {
      cut.vector.x[k] = h(aj);
    }
    if (cut.vector.x[k] != cut.vector.x[k]) {
      printf("cut.x[%d] %e != cut.x[%d] %e. aj %e beta %e var type %d\n",
             k,
             cut.vector.x[k],
             k,
             cut.vector.x[k],
             aj,
             beta,
             static_cast<int>(var_types[j]));
      exit(1);
    }
  }

  return true;
}

template <typename i_t, typename f_t>
f_t complemented_mixed_integer_rounding_cut_t<i_t, f_t>::compute_violation(
  const inequality_t<i_t, f_t>& cut, const std::vector<f_t>& xstar)
{
  f_t dot           = cut.vector.dot(xstar);
  f_t cut_violation = cut.rhs - dot;
  return cut_violation;
}

template <typename i_t, typename f_t>
void complemented_mixed_integer_rounding_cut_t<i_t, f_t>::substitute_slacks(
  const lp_problem_t<i_t, f_t>& lp,
  csr_matrix_t<i_t, f_t>& Arow,
  inequality_t<i_t, f_t>& cut,
  f_t* work_estimate)
{
  // Remove slacks from the cut
  // So that the cut is only over the original variables
  bool found_slack = false;
  i_t cut_nz       = 0;
  std::vector<i_t> cut_indices;
  cut_indices.reserve(cut.size());
  if (work_estimate != nullptr) { *work_estimate += cut.size(); }

  for (i_t k = 0; k < cut.size(); k++) {
    const i_t j  = cut.index(k);
    const f_t cj = cut.coeff(k);
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
      cut.rhs -= cj * lp.rhs[i] / alpha;
      const i_t row_start = Arow.row_start[i];
      const i_t row_end   = Arow.row_start[i + 1];
      if (work_estimate != nullptr) { *work_estimate += row_end - row_start; }
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
    scratch_pad_.get_pad(cut.vector.i, cut.vector.x);
    if (work_estimate != nullptr) {
      *work_estimate += 2 * cut.size() + cut.size() * std::log2((f_t)cut.size() + (f_t)1.0);
    }
    // Sort the cut
    cut.sort();
  }

  // Clear the workspace
  scratch_pad_.clear_pad();
}

template <typename i_t, typename f_t>
f_t complemented_mixed_integer_rounding_cut_t<i_t, f_t>::combine_rows(
  const lp_problem_t<i_t, f_t>& lp,
  csr_matrix_t<i_t, f_t>& Arow,
  i_t xj,
  const inequality_t<i_t, f_t>& pivot_row,
  inequality_t<i_t, f_t>& inequality)
{
  // Find the coefficient associated with variable xj in the pivot row
  f_t a_l_j = 0.0;
  for (i_t k = 0; k < pivot_row.size(); k++) {
    const i_t j = pivot_row.index(k);
    if (j == xj) {
      a_l_j = pivot_row.coeff(k);
      break;
    }
  }

  if (a_l_j == 0) {
    printf("Pivot row has no coefficient for variable %d\n", xj);
    return 0.0;
  }

  f_t a_i_j = 0.0;

  // Store the inequality in the workspace
  // and save the coefficient associated with variable xj
  for (i_t k = 0; k < inequality.size(); k++) {
    const i_t j = inequality.index(k);
    if (j != xj) {
      scratch_pad_.add_to_pad(j, inequality.coeff(k));
    } else {
      a_i_j = inequality.coeff(k);
    }
  }
  if (a_i_j == 0.0) {
    printf("Inequality has zero coefficient for variable %d\n", xj);
    scratch_pad_.clear_pad();
    return 0.0;
  }

  f_t pivot_value = a_i_j / a_l_j;
  // Adjust the rhs of the inequality
  inequality.rhs -= pivot_value * pivot_row.rhs;

  // Adjust the coefficients of the inequality
  // based on the nonzeros in the pivot row
  for (i_t k = 0; k < pivot_row.size(); k++) {
    const i_t j = pivot_row.index(k);
    if (j != xj) { scratch_pad_.add_to_pad(j, -pivot_value * pivot_row.coeff(k)); }
  }

  // Store the new inequality
  scratch_pad_.get_pad(inequality.vector.i, inequality.vector.x);

  // Clear the workspace
  scratch_pad_.clear_pad();

  return -pivot_value;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class complemented_mixed_integer_rounding_cut_t<int, double>;

template void cut_generation_t<int, double>::generate_mir_cuts(
  const lp_problem_t<int, double>& lp,
  const simplex_solver_settings_t<int, double>& settings,
  csr_matrix_t<int, double>& Arow,
  const std::vector<int>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  const std::vector<double>& xstar,
  const std::vector<double>& ystar,
  variable_bounds_t<int, double>& variable_bounds,
  double start_time);
#endif

}  // namespace cuopt::mathematical_optimization::mip
