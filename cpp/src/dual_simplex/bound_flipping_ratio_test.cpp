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
    for (i_t k = 0; k < n - m; ++k) {
      const i_t j = nonbasic_list_[k];
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
  const i_t j = entering_index = nonbasic_list_[nonbasic_entering];

  constexpr bool verbose = false;
  if (lower_[j] > -inf && upper_[j] < inf && lower_[j] != upper_[j]) {
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
  i_t m                  = m_;
  i_t n                  = n_;
  constexpr bool verbose = false;

  // Compute the initial set of breakpoints
  std::vector<i_t> indicies(n - m);
  std::vector<f_t> ratios(n - m);
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
  bool continue_search = k_idx >= 0 && num_breakpoints > 1 && slope > 0.0;
  if (!continue_search) {
    if constexpr (verbose) {
      settings_.log.printf(
        "BFRT stopping. No bound flips. Step length %e Nonbasic entering %d Entering %d.\n",
        step_length,
        nonbasic_entering,
        entering_index);
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

  heap_passes(
    indicies, ratios, num_breakpoints - 1, slope, step_length, nonbasic_entering, entering_index);

  if constexpr (verbose) {
    settings_.log.printf("BFRT step length %e entering index %d non basic entering %d\n",
                         step_length,
                         entering_index,
                         nonbasic_entering);
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

    if (lower_[j] > -inf && upper_[j] < inf && lower_[j] != upper_[j]) {
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

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template class bound_flipping_ratio_test_t<int, double>;

#endif

}  // namespace cuopt::linear_programming::dual_simplex
