/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "fj_cpu_binary.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Tabu for binary variables, expressed as a ring buffer
// There can be at most max_tenure tabu'd variables at any given time.
// since max_tenure << n_vars, it's cheaper to maintain a ring buffer than a full array
// and it allows smaller instances to become L1 resident
struct fj_bin_tabu_t {
  static constexpr int32_t ring_size  = 32;
  static constexpr int32_t max_tenure = ring_size;
  // Headroom so iter + tenure - iter_bias still fits uint16 when iter - iter_bias is at the rebase
  // threshold.
  static constexpr int32_t window = (int32_t)std::numeric_limits<uint16_t>::max() - max_tenure;

  std::vector<uint16_t> flip_until;
  std::vector<int32_t> last_flip;
  int32_t iter_bias{0};

  int32_t ring_var[ring_size];
  int32_t ring_expiry[ring_size];

  void resize(int32_t n)
  {
    flip_until.assign(n, 0);
    last_flip.assign(n, 0);
    clear_ring();
    iter_bias = 0;
  }

  void clear(int32_t iter)
  {
    std::fill(flip_until.begin(), flip_until.end(), (uint16_t)0);
    std::fill(last_flip.begin(), last_flip.end(), 0);
    clear_ring();
    iter_bias = iter;
  }

  void clear_ring()
  {
    for (int32_t i = 0; i < ring_size; ++i) {
      ring_var[i]    = -1;
      ring_expiry[i] = 0;
    }
  }

  void on_flip(int32_t v, int32_t iter, int32_t tenure)
  {
    flip_until[v] = (uint16_t)(iter + tenure - iter_bias);
    last_flip[v]  = iter;

    // keep only one tabu entry per var
    for (int32_t i = 0; i < ring_size; ++i) {
      if (ring_var[i] == v) ring_var[i] = -1;
    }

    const int32_t slot = iter & (ring_size - 1);
    ring_var[slot]     = v;
    ring_expiry[slot]  = iter + tenure;
  }

  // replace the scores of tabu'd variable with sentinel values
  int32_t block_tabu(int32_t iter,
                     int64_t* var_score,
                     int32_t (&saved_var)[ring_size],
                     int64_t (&saved_score)[ring_size]) const
  {
    int32_t k = 0;
    for (int32_t i = 0; i < ring_size; ++i) {
      const int32_t v = ring_var[i];
      if (v >= 0 && ring_expiry[i] > iter) {
        saved_var[k]   = v;
        saved_score[k] = var_score[v];
        var_score[v]   = fj_bin_score_invalid;
        ++k;
      }
    }
    return k;
  }

  // reverse the above operation.
  static void unblock_tabu(int32_t k,
                           int64_t* var_score,
                           const int32_t (&saved_var)[ring_size],
                           const int64_t (&saved_score)[ring_size])
  {
    for (int32_t i = k - 1; i >= 0; --i)
      var_score[saved_var[i]] = saved_score[i];
  }

  bool blocked(int32_t v, int32_t iter, bool localmin) const
  {
    return localmin ? (iter == last_flip[v] + 1) : ((uint16_t)(iter - iter_bias) < flip_until[v]);
  }

  // rebase the iteration bias value every 64k iter
  void maybe_rebase(int32_t iter)
  {
    if ((int64_t)iter - iter_bias <= window) return;
    const uint16_t shift = (uint16_t)(iter - iter_bias);
    for (uint16_t& fu : flip_until)
      fu = (fu > shift) ? (uint16_t)(fu - shift) : (uint16_t)0;
    iter_bias = iter;
  }
};

}  // namespace cuopt::mathematical_optimization::mip
