/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <dual_simplex/bound_flipping_ratio_test.hpp>

#include <dual_simplex/tic_toc.hpp>

#include <algorithm>
#include <cmath>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
i_t bound_flipping_ratio_test_t<i_t, f_t>::compute_breakpoints(std::vector<i_t>& indicies,
                                                               std::vector<f_t>& ratios)
{
  i_t n                  = n_;
  i_t m                  = m_;
  constexpr bool verbose = false;
  f_t pivot_tol          = settings_.pivot_tol;
  const f_t dual_tol     = settings_.dual_tol / 10;

  i_t idx = 0;
  while (idx == 0 && pivot_tol >= 1e-12) {
    // for (i_t k = 0; k < n - m; ++k) {
    //   const i_t j = nonbasic_list_[k];
    for (i_t h = 0; h < delta_z_indices_.size(); ++h) {
      const i_t j = delta_z_indices_[h];
      const i_t k = nonbasic_mark_[j];
      if (vstatus_[j] == variable_status_t::NONBASIC_FIXED) { continue; }
      if (vstatus_[j] == variable_status_t::NONBASIC_LOWER && delta_z_[j] < -pivot_tol) {
        indicies[idx] = k;
        ratios[idx]   = std::max((-dual_tol - z_[j]) / delta_z_[j], 0.0);
        if constexpr (verbose) { settings_.log.printf("ratios[%d] = %e\n", idx, ratios[idx]); }
        idx++;
      }
      if (vstatus_[j] == variable_status_t::NONBASIC_UPPER && delta_z_[j] > pivot_tol) {
        indicies[idx] = k;
        ratios[idx]   = std::max((dual_tol - z_[j]) / delta_z_[j], 0.0);
        if constexpr (verbose) { settings_.log.printf("ratios[%d] = %e\n", idx, ratios[idx]); }
        idx++;
      }
    }
    pivot_tol /= 10;
  }
  return idx;
}

template <typename i_t, typename f_t>
i_t bound_flipping_ratio_test_t<i_t, f_t>::single_pass(i_t start,
                                                       i_t end,
                                                       const std::vector<i_t>& indicies,
                                                       const std::vector<f_t>& ratios,
                                                       f_t& slope,
                                                       f_t& step_length,
                                                       i_t& nonbasic_entering,
                                                       i_t& entering_index)
{
  // Find the minimum ratio
  f_t min_val    = inf;
  entering_index = -1;
  i_t candidate  = -1;
  f_t zero_tol   = settings_.zero_tol;
  i_t k_idx      = -1;
  for (i_t k = start; k < end; ++k) {
    if (ratios[k] < min_val) {
      min_val   = ratios[k];
      candidate = indicies[k];
      k_idx     = k;
    } else if (ratios[k] < min_val + zero_tol) {
      // Use Harris to select variables with larger pivots
      const i_t j = nonbasic_list_[indicies[k]];
      if (std::abs(delta_z_[j]) > std::abs(delta_z_[candidate])) {
        min_val   = ratios[k];
        candidate = indicies[k];
        k_idx     = k;
      }
    }
  }
  step_length       = min_val;
  nonbasic_entering = candidate;
  // this should be temporary, find root causes where the candidate is not filled
  if (nonbasic_entering == -1) {
    // -1,-2 and -3 are reserved for other things
    return -4;
  }
  const i_t j = entering_index = nonbasic_list_[nonbasic_entering];

  constexpr bool verbose = false;
  if (bounded_variables_[j]) {
    const f_t interval    = upper_[j] - lower_[j];
    const f_t delta_slope = std::abs(delta_z_[j]) * interval;
    if constexpr (verbose) {
      settings_.log.printf("single pass delta slope %e slope %e after slope %e step length %e\n",
                           delta_slope,
                           slope,
                           slope - delta_slope,
                           step_length);
    }
    slope -= delta_slope;
    return k_idx;  // we should see if we can continue to increase the step-length
  }
  return -1;  // we are done. do not increase the step-length further
}

template <typename i_t, typename f_t>
i_t bound_flipping_ratio_test_t<i_t, f_t>::compute_step_length(f_t& step_length,
                                                               i_t& nonbasic_entering)
{
  const i_t m            = m_;
  const i_t n            = n_;
  const i_t nz           = delta_z_indices_.size();
  constexpr bool verbose = false;

  // Compute the initial set of breakpoints
  std::vector<i_t> indicies(nz);
  std::vector<f_t> ratios(nz);
  i_t num_breakpoints = compute_breakpoints(indicies, ratios);
  if constexpr (verbose) { settings_.log.printf("Initial breakpoints %d\n", num_breakpoints); }
  if (num_breakpoints == 0) {
    nonbasic_entering = -1;
    return -1;
  }

  f_t slope          = slope_;
  nonbasic_entering  = -1;
  i_t entering_index = -1;

  i_t k_idx = single_pass(
    0, num_breakpoints, indicies, ratios, slope, step_length, nonbasic_entering, entering_index);
  if (k_idx == -4) { return -4; }
  bool continue_search = k_idx >= 0 && num_breakpoints > 1 && slope > 0.0;
  if (!continue_search) {
    if constexpr (0) {
      settings_.log.printf(
        "BFRT stopping. No bound flips. Step length %e Nonbasic entering %d Entering %d pivot %e\n",
        step_length,
        nonbasic_entering,
        entering_index,
        std::abs(delta_z_[entering_index]));
    }
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

  // Continue the search using a heap to order the breakpoints
  ratios[k_idx]   = ratios[num_breakpoints - 1];
  indicies[k_idx] = indicies[num_breakpoints - 1];

  constexpr bool use_bucket_pass = false;

  if (use_bucket_pass) {
    f_t max_ratio = 0.0;
    for (i_t k = 0; k < num_breakpoints - 1; ++k) {
      if (ratios[k] > max_ratio) { max_ratio = ratios[k]; }
    }
    settings_.log.printf(
      "Starting heap passes. %d breakpoints max ratio %e\n", num_breakpoints - 1, max_ratio);
    bucket_pass(
      indicies, ratios, num_breakpoints - 1, slope, step_length, nonbasic_entering, entering_index);
  }

  heap_passes(
    indicies, ratios, num_breakpoints - 1, slope, step_length, nonbasic_entering, entering_index);

  if constexpr (verbose) {
    settings_.log.printf("BFRT step length %e entering index %d non basic entering %d pivot %e\n",
                         step_length,
                         entering_index,
                         nonbasic_entering,
                         std::abs(delta_z_[entering_index]));
  }
  return entering_index;
}

template <typename i_t, typename f_t>
void bound_flipping_ratio_test_t<i_t, f_t>::heap_passes(const std::vector<i_t>& current_indicies,
                                                        const std::vector<f_t>& current_ratios,
                                                        i_t num_breakpoints,
                                                        f_t& slope,
                                                        f_t& step_length,
                                                        i_t& nonbasic_entering,
                                                        i_t& entering_index)
{
  std::vector<i_t> bare_idx(num_breakpoints);
  constexpr bool verbose                = false;
  const f_t dual_tol                    = settings_.dual_tol;
  const f_t zero_tol                    = settings_.zero_tol;
  const std::vector<f_t>& delta_z       = delta_z_;
  const std::vector<i_t>& nonbasic_list = nonbasic_list_;
  const i_t N                           = num_breakpoints;
  for (i_t k = 0; k < N; ++k) {
    bare_idx[k] = k;
    if constexpr (verbose) {
      settings_.log.printf("Adding index %d ratio %e pivot %e to heap\n",
                           current_indicies[k],
                           current_ratios[k],
                           std::abs(delta_z[nonbasic_list[current_indicies[k]]]));
    }
  }

  auto compare = [zero_tol, &current_ratios, &current_indicies, &delta_z, &nonbasic_list](
                   const i_t& a, const i_t& b) {
    return (current_ratios[a] > current_ratios[b]) ||
           (current_ratios[b] - current_ratios[a] < zero_tol &&
            std::abs(delta_z[nonbasic_list[current_indicies[a]]]) >
              std::abs(delta_z[nonbasic_list[current_indicies[b]]]));
  };

  std::make_heap(bare_idx.begin(), bare_idx.end(), compare);

  while (bare_idx.size() > 0 && slope > 0) {
    // Remove minimum ratio from the heap and rebalance
    i_t heap_index = bare_idx.front();
    std::pop_heap(bare_idx.begin(), bare_idx.end(), compare);
    bare_idx.pop_back();

    nonbasic_entering = current_indicies[heap_index];
    const i_t j = entering_index = nonbasic_list_[nonbasic_entering];
    step_length                  = current_ratios[heap_index];

    if (bounded_variables_[j]) {
      // We have a bounded variable
      const f_t interval    = upper_[j] - lower_[j];
      const f_t delta_slope = std::abs(delta_z_[j]) * interval;
      const f_t pivot       = std::abs(delta_z[j]);
      if constexpr (verbose) {
        settings_.log.printf(
          "heap %d step-length %.12e pivot %e nonbasic entering %d slope %e delta_slope %e new "
          "slope %e\n",
          bare_idx.size(),
          current_ratios[heap_index],
          pivot,
          nonbasic_entering,
          slope,
          delta_slope,
          slope - delta_slope);
      }
      slope -= delta_slope;
    } else {
      // The variable is not bounded. Stop the search.
      break;
    }

    if (toc(start_time_) > settings_.time_limit) {
      entering_index = -2;
      return;
    }
    if (settings_.concurrent_halt != nullptr &&
        settings_.concurrent_halt->load(std::memory_order_acquire) == 1) {
      entering_index = -3;
      return;
    }
  }
}

template <typename i_t, typename f_t>
void bound_flipping_ratio_test_t<i_t, f_t>::bucket_pass(const std::vector<i_t>& current_indicies,
                                                        const std::vector<f_t>& current_ratios,
                                                        i_t num_breakpoints,
                                                        f_t& slope,
                                                        f_t& step_length,
                                                        i_t& nonbasic_entering,
                                                        i_t& entering_index)
{
  const f_t dual_tol                    = settings_.dual_tol;
  const f_t zero_tol                    = settings_.zero_tol;
  const std::vector<f_t>& delta_z       = delta_z_;
  const std::vector<i_t>& nonbasic_list = nonbasic_list_;
  const i_t N                           = num_breakpoints;

  const i_t K = 400;  // 0, -16, -15, ...., 0, 1, ...., 400 - 18 = 382
  std::vector<f_t> buckets(K, 0.0);
  std::vector<i_t> bucket_count(K, 0);
  for (i_t k = 0; k < N; ++k) {
    const i_t idx          = current_indicies[k];
    const f_t ratio        = current_ratios[k];
    const f_t min_exponent = -16.0;
    const f_t max_exponent = 382.0;
    const f_t exponent     = std::max(min_exponent, std::min(max_exponent, std::log10(ratio)));
    const i_t bucket_idx   = ratio == 0.0 ? 0 : static_cast<i_t>(exponent - min_exponent + 1);
    // settings_.log.printf("Ratio %e exponent %e bucket_idx %d\n", ratio, exponent, bucket_idx);
    const i_t j           = nonbasic_list[idx];
    const f_t interval    = upper_[j] - lower_[j];
    const f_t delta_slope = std::abs(delta_z_[j]) * interval;
    buckets[bucket_idx] += delta_slope;
    bucket_count[bucket_idx]++;
  }

  std::vector<f_t> cumulative_sum(K, 0.0);
  cumulative_sum[0] = buckets[0];
  if (cumulative_sum[0] > slope) {
    settings_.log.printf(
      "Bucket 0. Count in bucket %d. Slope %e. Cumulative sum %e. Bucket value %e\n",
      bucket_count[0],
      slope,
      cumulative_sum[0],
      buckets[0]);
    return;
  }
  i_t k;
  bool exceeded = false;
  for (k = 1; k < K; ++k) {
    cumulative_sum[k] = cumulative_sum[k - 1] + buckets[k];
    if (cumulative_sum[k] > slope) {
      exceeded = true;
      break;
    }
  }

  if (exceeded) {
    settings_.log.printf(
      "Value in bucket %d. Count in buckets %d. Slope %e. Cumulative sum %e. Next sum %e Bucket "
      "value %e\n",
      k,
      bucket_count[k],
      slope,
      cumulative_sum[k - 1],
      cumulative_sum[k],
      buckets[k - 1]);
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template class bound_flipping_ratio_test_t<int, double>;

#endif

}  // namespace cuopt::linear_programming::dual_simplex
