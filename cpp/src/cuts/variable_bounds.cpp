/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuts/cuts.hpp>
#include <math_optimization/tic_toc.hpp>

#include <cstdio>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

using simplex::lp_problem_t;
using simplex::simplex_solver_settings_t;
using simplex::variable_type_t;

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
  if (settings.inside_submip) {
    return;  // Don't compute the variable upper/lower bounds inside sub-MIP
  }
  f_t start_time = tic();

  settings.log.printf("Computing variable bounds...");

  std::vector<i_t> num_integer_in_row(lp.num_rows, 0);

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

  // Compute the upper activities of the constraints
  for (i_t i = 0; i < lp.num_rows; i++) {
    const i_t row_start   = Arow.row_start[i];
    const i_t row_end     = Arow.row_start[i + 1];
    const i_t slack_index = slack_map_[i];
    f_t activity          = 0.0;
    for (i_t p = row_start; p < row_end; p++) {
      const i_t j = Arow.j[p];
      if (j == slack_index) { continue; }
      if (var_types[j] == variable_type_t::INTEGER) { num_integer_in_row[i]++; }
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
    }
    upper_activities_[i] = activity;
  }

  // Compute the lower activities of the constraints
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
      const f_t a_ij          = lp.A.x[p];
      const i_t slack_index   = slack_map_[i];
      const f_t slack_lower   = slack_index >= 0 ? lp.lower[slack_index] : 0.0;
      const f_t slack_upper   = slack_index >= 0 ? lp.upper[slack_index] : 0.0;
      const f_t slack_coeff_i = slack_coeff[i];
      const f_t sigma_slack_lower =
        slack_coeff_i > 0.0 ? slack_coeff_i * slack_lower : slack_coeff_i * slack_upper;
      const f_t sigma_slack_upper =
        slack_coeff_i > 0.0 ? slack_coeff_i * slack_upper : slack_coeff_i * slack_lower;

      if (sigma_slack_lower > -inf) {
        const f_t beta = lp.rhs[i] - sigma_slack_lower;
        // sum_k a_ik x_k <= beta

        // If we have too many variables in the row that would cause the activity to be infinite,
        // we cannot derive a variable bound
        if (a_ij > 0.0 && num_neg_inf_[i] <= 2) {
          const f_t lower_activity_j = lower_activity(lp.lower[j], lp.upper[j], a_ij);

          // This loop may still scan integer variables when num_neg_inf_[i] > 0. A bound is finite
          // only if every lower-infinite contribution is removed by excluding the target j and the
          // bound variable l. For num_neg_inf_[i] == 1, the lower-infinite term must be either j or
          // l; for num_neg_inf_[i] == 2, both j and l must be lower-infinite.
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
  settings.log.print_format("{} variable upper bounds in {:.2f}s\n", upper_edges, toc(start_time));

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
      const f_t a_ij          = lp.A.x[p];
      const i_t slack_index   = slack_map_[i];
      const f_t slack_lower   = slack_index >= 0 ? lp.lower[slack_index] : 0.0;
      const f_t slack_upper   = slack_index >= 0 ? lp.upper[slack_index] : 0.0;
      const f_t slack_coeff_i = slack_coeff[i];
      const f_t sigma_slack_lower =
        slack_coeff_i > 0.0 ? slack_coeff_i * slack_lower : slack_coeff_i * slack_upper;
      const f_t sigma_slack_upper =
        slack_coeff_i > 0.0 ? slack_coeff_i * slack_upper : slack_coeff_i * slack_lower;

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
  settings.log.print_format("{} variable lower bounds in {:.2f}s\n", lower_edges, toc(start_time));
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class variable_bounds_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
