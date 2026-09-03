/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/bound_flipping_ratio_test.hpp>

#include <math_optimization/tic_toc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace cuopt::mathematical_optimization::simplex {

template <typename i_t, typename f_t>
i_t bound_flipping_ratio_test_t<i_t, f_t>::compute_breakpoints(std::vector<i_t>& indicies,
                                                               std::vector<f_t>& ratios,
                                                               std::vector<f_t>& harris_ratios)
{
  i_t n                  = n_;
  i_t m                  = m_;
  constexpr bool verbose = false;
  f_t pivot_tol          = settings_.pivot_tol;
  const f_t dual_tol     = settings_.dual_tol / 10;

  i_t idx = 0;
  while (idx == 0 && pivot_tol >= 1e-12) {
    // Loop over the nonbasic variables j with non-zero delta_z
    const i_t nz = delta_z_indices_.size();
    for (i_t h = 0; h < nz; ++h) {
      const i_t j = delta_z_indices_[h];
      const i_t k = nonbasic_mark_[j];
      if (vstatus_[j] == variable_status_t::NONBASIC_FIXED) { continue; }
      if (vstatus_[j] == variable_status_t::NONBASIC_LOWER && delta_z_[j] < -pivot_tol) {
        indicies[idx]      = k;
        ratios[idx]        = std::max((-z_[j]) / delta_z_[j], 0.0);
        harris_ratios[idx] = std::max((-dual_tol - z_[j]) / delta_z_[j], 0.0);
        if constexpr (verbose) { settings_.log.printf("ratios[%d] = %e\n", idx, ratios[idx]); }
        idx++;
      }
      if (vstatus_[j] == variable_status_t::NONBASIC_UPPER && delta_z_[j] > pivot_tol) {
        indicies[idx]      = k;
        ratios[idx]        = std::max((-z_[j]) / delta_z_[j], 0.0);
        harris_ratios[idx] = std::max((dual_tol - z_[j]) / delta_z_[j], 0.0);
        if constexpr (verbose) { settings_.log.printf("ratios[%d] = %e\n", idx, ratios[idx]); }
        idx++;
      }
    }
    work_estimate_ += 5 * nz + 5 * idx;
    pivot_tol /= 10;
  }
  return idx;
}

template <typename i_t, typename f_t>
i_t bound_flipping_ratio_test_t<i_t, f_t>::single_pass(i_t start,
                                                       i_t end,
                                                       const std::vector<i_t>& indicies,
                                                       const std::vector<f_t>& ratios,
                                                       f_t& step_length,
                                                       i_t& nonbasic_entering,
                                                       i_t& entering_index,
                                                       f_t& max_val)
{
  // Find the minimum ratio
  f_t min_val    = inf;
  entering_index = -1;
  i_t candidate  = -1;
  f_t zero_tol   = settings_.zero_tol;
  i_t k_idx      = -1;
  max_val        = 0.0;

  i_t min_found = 0;
  for (i_t k = start; k < end; ++k) {
    if (ratios[k] < min_val) {
      min_val   = ratios[k];
      candidate = indicies[k];
      k_idx     = k;
      min_found++;
    }
    if (ratios[k] > max_val) { max_val = ratios[k]; }
  }
  work_estimate_ += (end - start) + 2 * min_found;

  step_length       = min_val;
  nonbasic_entering = candidate;
  // this should be temporary, find root causes where the candidate is not filled
  if (nonbasic_entering == -1) { return RATIO_TEST_NUMERICAL_ISSUES; }
  const i_t j = entering_index = nonbasic_list_[nonbasic_entering];

  if (bounded_variables_[j]) { return k_idx; }
  return -1;  // we are done. do not increase the step-length further
}

template <typename i_t, typename f_t>
void bound_flipping_ratio_test_t<i_t, f_t>::determine_flips(f_t step_length,
                                                            i_t entering_index,
                                                            std::vector<i_t>& flip_indices)
{
  // The piecewise-linear model below assumes that a variable flips bounds as soon as
  // its reduced cost crosses zero. In practice, small changes between iterations can
  // make a reduced cost oscillate around zero, causing excessive bound flips and
  // cycling. We therefore flip only after the violation exceeds dual_tol / 10.
  // A bounded variable l_j <= x_j <= u_j contributes l_j*z_j to the dual objective
  // when z_j >= 0 and u_j*z_j when z_j < 0. If x_j = l_j and
  // -dual_tol/10 <= z_j < 0, the model uses u_j*z_j while the unflipped state uses
  // l_j*z_j. Their difference is (u_j - l_j)*|z_j|, bounded by
  // (u_j - l_j)*dual_tol/10. For multiple unflipped variables, the discrepancy is
  // bounded by sum_j (u_j - l_j)*dual_tol/10.
  const f_t flip_tol = settings_.dual_tol / 10;
  for (const i_t j : delta_z_indices_) {
    if (j == entering_index || !bounded_variables_[j]) { continue; }
    const f_t new_z = z_[j] + step_length * delta_z_[j];
    if ((vstatus_[j] == variable_status_t::NONBASIC_LOWER && new_z < -flip_tol) ||
        (vstatus_[j] == variable_status_t::NONBASIC_UPPER && new_z > flip_tol)) {
      flip_indices.push_back(j);
    }
  }
  work_estimate_ += 5 * delta_z_indices_.size() + flip_indices.size();
}

template <typename i_t, typename f_t>
i_t bound_flipping_ratio_test_t<i_t, f_t>::compute_step_length(f_t& step_length,
                                                               i_t& nonbasic_entering,
                                                               std::vector<i_t>& flip_indices)
{
  const i_t m            = m_;
  const i_t n            = n_;
  const i_t nz           = delta_z_indices_.size();
  constexpr bool verbose = false;
  flip_indices.clear();

  // Compute the initial set of breakpoints
  std::vector<i_t> indicies(nz);
  std::vector<f_t> ratios(nz);
  std::vector<f_t> harris_ratios(nz);
  work_estimate_ += 3 * nz;
  double t0           = tic();
  i_t num_breakpoints = compute_breakpoints(indicies, ratios, harris_ratios);
  time_compute_breakpoints_ += toc(t0);
  num_breakpoints_ = num_breakpoints;
  // Count zero ratios
  num_harris_zero_ = 0;
  num_exact_zero_  = 0;
  for (i_t k = 0; k < num_breakpoints; k++) {
    if (harris_ratios[k] == 0.0) num_harris_zero_++;
    if (ratios[k] == 0.0) num_exact_zero_++;
  }
  work_estimate_ += 2 * num_breakpoints;
  if constexpr (verbose) { settings_.log.printf("Initial breakpoints %d\n", num_breakpoints); }
  if (num_breakpoints == 0) {
    nonbasic_entering = -1;
    return RATIO_TEST_NO_ENTERING_VARIABLE;
  }

  f_t slope          = slope_;
  nonbasic_entering  = -1;
  i_t entering_index = RATIO_TEST_NO_ENTERING_VARIABLE;
  f_t max_step_length;

  t0        = tic();
  i_t k_idx = single_pass(0,
                          num_breakpoints,
                          indicies,
                          harris_ratios,
                          step_length,
                          nonbasic_entering,
                          entering_index,
                          max_step_length);
  time_single_pass_ += toc(t0);
  if (k_idx == RATIO_TEST_NUMERICAL_ISSUES) { return RATIO_TEST_NUMERICAL_ISSUES; }
  // The variable selected by single_pass is guaranteed to be in the first bucket: it
  // defines the minimum Harris ratio, and its exact ratio is no greater than its Harris
  // ratio. Its slope contribution is therefore applied by the bucket pass below.
  bool continue_search = k_idx >= 0 && num_breakpoints > 1;
  if (!continue_search) {
    if constexpr (verbose) {
      settings_.log.printf(
        "BFRT stopping. No bound flips. Step length %e Nonbasic entering %d Entering %d pivot %e\n",
        step_length,
        nonbasic_entering,
        entering_index,
        std::abs(delta_z_[entering_index]));
    }
    num_buckets_used_   = 0;
    step_length_result_ = step_length;
    determine_flips(step_length, entering_index, flip_indices);
    return entering_index;
  }

  if constexpr (verbose) {
    settings_.log.printf(
      "Continuing past initial step length %e entering index %d nonbasic entering %d slope %e\n",
      step_length,
      entering_index,
      nonbasic_entering,
      slope);
  }

  // This code is complicated. There are several important concepts that are needed to understand
  // it.
  //
  // We are trying to compute the maximum step length we can take while:
  //  1) Staying mostly dual feasible (we allow ourselves to be infeasible by dual_tol amount)
  //  2) Increasing the dual objective
  //  3) Selecting a variable with a large pivot (| delta_z[j] |)
  //
  // Let alpha be the step length. For each nonbasic variable j, we have
  // z_j(alpha) = z_j + alpha * delta_z_j
  //
  // To stay dual feasible, we either need to keep
  //  z_j(alpha) >= 0, if j is on it's lower bound, or
  //  z_j(alpha) <= 0, if j is on it's upper bound.
  //
  // Consider the equation z_j(alpha) = z_j + alpha * delta_z_j = 0. Each variable j puts a bound on
  // alpha:
  //
  // alpha_j <= -z_j / delta_z_j, if x_j = l_j (z_j >= 0) and delta_z_j < 0
  // alpha_j <= -z_j / delta_z_j, if x_j = u_j (z_j <= 0) and delta_z_j > 0
  //
  // The code refers to these alpha_j as ratios, since they are the ratio of z_j to delta_z_j.
  //
  // Now we could take alpha = min_j alpha_j, and remain dual feasible. However, we are allowed to
  // increase the step-length if j is a variable such that l_j <= x_j <= u_j. To see why imagine
  // that our variable was currenlty on it's lower bound, with z_j > 0 and delta_z_j < 0, if we push
  // alpha past alpha_j, than z_j(alpha) < 0. This is fine as long as we flip the variable to be on
  // it's upper bound. Thus, we can push alpha past alpha_j for *bounded* variables.
  //
  // Note that this does not work if we try to increase alpha past alpha_j for a variable with a
  // single bound. We would just be making ourselves dual infeasible. So we need to check whether a
  // variable is bounded.
  //
  // The dual objective as a function of the step-length alpha, is piecewise linear and concave. The
  // breakpoints of this piecewise linear function occur at each of the alpha_j values. We can keep
  // increasing the step-length as long as the slope remains nonnegative. After that we must stop,
  // because we could decrease the dual objective. So the code tracks the cumulative slope of the
  // dual objective.
  //
  // Now we don't need to exactly feasible: z_j >= 0 if x_j = l_j and z_j <= 0 if x_j = u_j. We can
  // violate these bounds by the dual feasibility tolerance eps. We allow ourselves to be infeasible
  // if it would help us get a larger pivot (delta_z_j). Small pivots can cause numerical issues, so
  // we would like to avoid them.
  //
  // With this tolerance we get the equations:
  //  z_j(alpha) = z_j + alpha * delta_z_j >= -eps if x_j = l_j
  //  z_j(alpha) = z_j + alpha * delta_z_j <= eps  if x_j = u_j
  //
  // This gives bounds on alpha. We call these alpha_harris_j, for Paula Harris, who proposed this
  // method.
  //
  // alpha_harris_j <= (-eps - z_j) / delta_z_j, if x_j = l_j and delta_z_j < 0
  // alpha_harris_j <= (eps - z_j) / delta_z_j, if x_j = u_j and delta_z_j > 0
  //
  // Let alpha_harris = min_j alpha_harris_j. We can select the variable with the largest |
  // delta_z_j | from those candidates { j | alpha_j <= alpha_harris }.
  //
  // We combine these two ideas (increasing the step length for bounded variables) and allowing
  // ourselves to be slightly dual infeasible to choose a larger pivot.
  //
  // We partition the variables into buckets. Let B_k be the set of variables in bucket k. B_0 is
  // defined as { j | alpha_j <= alpha_harris }. We then compute alpha_harris_1 = min_{j not in B_0}
  // alpha_j. And B_1 is defined as { j not in B_0 | alpha_j <= alpha_harris_1 }. And so on.
  //
  // We want to balance two different things:
  //  1) Taking a larger step length to increase the dual objective as much as possible,
  //  2) Choosing a large pivot for numerical stability.
  //
  // Let max_pivot = max_j | delta_z_j |. We start working our way backward from the largest bucket
  // to the smallest bucket, we choose a variable j that satisfies | delta_z_j | >= 0.1 * max_pivot.
  // Since we can always choose a smaller step length for the sake of numerical stability.
  //
  // Now the final thing to understand is that the ratios alpha_j are not sorted in any particular
  // order. And we don't want to pay the O(num_breakpoints * log(num_breakpoints)) cost of sorting
  // them.
  //
  // So we set a threshold on the step-length and check if all variables j with alpha_j <= threshold
  // have already caused the slope to go negative. If so, we just need to consider those candidate
  // variables with alpha_j <= threshold. If not, we multiply the threshold by 10. This cost us
  // O(log10(max_step_length/min_step_length) * num_breakpoints) time. So we aren't totally linear.
  // But the hope is we are better than a sort.

  // Use a coarse filter to find candidates
  f_t minimum_harris_ratio = step_length;
  f_t coarse_threshold     = (minimum_harris_ratio > 0.0)
                               ? std::min(10.0 * minimum_harris_ratio, max_step_length)
                               : max_step_length;
  f_t total_slope          = slope;
  bool found_unbounded     = false;
  std::vector<i_t> candidates(num_breakpoints);
  std::iota(candidates.begin(), candidates.end(), 0);
  work_estimate_ += 2 * num_breakpoints;
  i_t scan_start     = 0;
  i_t num_candidates = 0;

  // This is O( log10(max_step_length/min_step_length) * num_breakpoints)
  t0 = tic();
  while (total_slope >= 0.0 && coarse_threshold <= max_step_length &&
         scan_start < num_breakpoints && !found_unbounded) {
    for (i_t h = scan_start; h < num_breakpoints; ++h) {
      const i_t k = candidates[h];
      if (ratios[k] <= coarse_threshold) {
        // Candidate is less than coarse threshold, move it to the front of the candidate list
        std::swap(candidates[h], candidates[num_candidates]);
        num_candidates++;
        const i_t j = nonbasic_list_[indicies[k]];
        if (!bounded_variables_[j]) {
          found_unbounded = true;
        } else {
          total_slope -= std::abs(delta_z_[j]) * (upper_[j] - lower_[j]);
        }
      }
    }
    work_estimate_ += 2 * (num_breakpoints - scan_start) + 10 * (num_candidates - scan_start);
    scan_start = num_candidates;
    coarse_threshold *= 10.0;
  }
  time_coarse_filter_ += toc(t0);

  candidates.resize(num_candidates);

  // Check for variables with one sided bounds. These define the maximum step length.
  if (found_unbounded) {
    for (i_t h = 0; h < num_candidates; h++) {
      const i_t k = candidates[h];
      const i_t j = nonbasic_list_[indicies[k]];
      if (!bounded_variables_[j]) { max_step_length = std::min(max_step_length, harris_ratios[k]); }
    }
    work_estimate_ += 5 * num_candidates;

    // Remove candidates that are greater than the maximum step length
    const i_t candidates_before_removal = candidates.size();
    for (i_t h = candidates_before_removal - 1; h >= 0; h--) {
      const i_t k     = candidates[h];
      const f_t ratio = ratios[k];
      if (ratio > max_step_length) {
        // Swap with the last candidate and remove
        candidates[h] = candidates.back();
        candidates.pop_back();
      }
    }
    work_estimate_ +=
      2 * candidates_before_removal + 2 * (candidates_before_removal - candidates.size());
    num_candidates = candidates.size();
  }

  // Use a bucket sort to partition candidates into buckets by successive Harris breakpoints
  // bucket_start[k] = index in candidates[] where bucket k starts
  // Bucket k contains candidates[bucket_start[k]] .. candidates[bucket_start[k+1] - 1]
  f_t threshold   = minimum_harris_ratio;
  i_t num_buckets = 0;
  std::vector<i_t> bucket_start(num_candidates + 1, 0);
  f_t cumulative_slope = slope;
  scan_start           = 0;
  work_estimate_ += num_candidates + 1;

  // This is O(num_buckets * num_candidates)
  i_t slope_breaker_k = -1;  // the candidate k that made slope go negative
  t0                  = tic();
  while (cumulative_slope >= 0.0 && scan_start < num_candidates && threshold <= max_step_length) {
    f_t next_threshold = inf;
    i_t write          = scan_start;

    for (i_t h = scan_start; h < num_candidates; h++) {
      const i_t k     = candidates[h];
      const f_t ratio = ratios[k];

      if (ratio <= threshold) {
        const i_t j = nonbasic_list_[indicies[k]];
        if (bounded_variables_[j]) {
          cumulative_slope -= std::abs(delta_z_[j]) * (upper_[j] - lower_[j]);
          if (cumulative_slope < 0.0 && slope_breaker_k < 0) { slope_breaker_k = k; }
        }
        std::swap(candidates[h], candidates[write]);
        write++;
      } else {
        const i_t j            = nonbasic_list_[indicies[k]];
        const f_t harris_ratio = harris_ratios[k];
        next_threshold         = std::min(next_threshold, harris_ratio);
      }
    }
    work_estimate_ += 3 * (num_candidates - scan_start) + 9 * (write - scan_start);

    bucket_start[++num_buckets] = write;
    if (write == scan_start) break;  // No progress — prevent infinite loop
    scan_start = write;
    threshold  = next_threshold;

    if (cumulative_slope < 0.0) break;
  }
  time_bucket_sort_ += toc(t0);
  bucket0_size_ = (num_buckets > 0) ? bucket_start[1] : 0;

  // Compute the maximum pivot
  // This is O(num_candidates)
  f_t max_pivot = 0.0;
  for (i_t h = 0; h < bucket_start[num_buckets]; h++) {
    const i_t k     = candidates[h];
    const i_t j     = nonbasic_list_[indicies[k]];
    const f_t pivot = std::abs(delta_z_[j]);
    if (pivot > max_pivot) { max_pivot = pivot; }
  }
  work_estimate_ += 4 * bucket_start[num_buckets];

  // Select the entering variable
  // Scan from last bucket to first. Within each bucket, pick the variable with
  // the largest ratio (step length) that has |delta_z| > pivot_threshold
  f_t pivot_threshold = std::max(settings_.pivot_tol, std::min(0.1 * max_pivot, 1.0));
  i_t entering_k      = -1;

  // This is O(num_candidates)
  for (i_t b = num_buckets - 1; b >= 0; b--) {
    const i_t b_start = bucket_start[b];
    const i_t b_end   = bucket_start[b + 1];
    f_t best_ratio    = -1.0;
    for (i_t h = b_start; h < b_end; h++) {
      const i_t k     = candidates[h];
      const i_t j     = nonbasic_list_[indicies[k]];
      const f_t pivot = std::abs(delta_z_[j]);
      if (pivot > pivot_threshold && ratios[k] > best_ratio) {
        best_ratio = ratios[k];
        entering_k = k;
      }
    }
    work_estimate_ += 2 + 5 * (b_end - b_start);
    if (entering_k >= 0) break;
  }

  // Step = entering variable's breakpoint ratio
  num_buckets_used_ = num_buckets;
  if (entering_k < 0) {
    // Fallback to single_pass result
    used_fallback_             = true;
    bucket_selected_           = -1;
    step_length_result_        = step_length;
    selected_is_slope_breaker_ = false;
    determine_flips(step_length, entering_index, flip_indices);
    return entering_index;
  }
  step_length       = ratios[entering_k];
  nonbasic_entering = indicies[entering_k];
  entering_index    = nonbasic_list_[nonbasic_entering];

  // Record whether we selected the slope breaker
  selected_is_slope_breaker_ = (entering_k == slope_breaker_k);

  // Record which bucket was selected
  used_fallback_ = false;
  i_t pos        = -1;
  for (i_t b = 0; b < num_buckets; b++) {
    if (entering_k >= 0) {
      // Find which bucket entering_k is in based on its position in candidates
      pos = -1;
      for (i_t h = 0; h < num_candidates; h++) {
        if (candidates[h] == entering_k) {
          pos = h;
          break;
        }
      }
      if (pos >= bucket_start[b] && pos < bucket_start[b + 1]) {
        bucket_selected_ = b;
        break;
      }
    }
  }
  work_estimate_ += (bucket_selected_ + 1) * (pos + 3);
  step_length_result_ = step_length;
  determine_flips(step_length, entering_index, flip_indices);

  return entering_index;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template class bound_flipping_ratio_test_t<int, double>;

#endif

}  // namespace cuopt::mathematical_optimization::simplex
