/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <math_optimization/types.hpp>

#include <vector>

namespace cuopt::mathematical_optimization::simplex {

#define RATIO_TEST_NO_ENTERING_VARIABLE -1
#define RATIO_TEST_CONCURRENT_LIMIT     CONCURRENT_HALT_RETURN  // -2
#define RATIO_TEST_TIME_LIMIT           -3
#define RATIO_TEST_NUMERICAL_ISSUES     -4

template <typename i_t, typename f_t>
class bound_flipping_ratio_test_t {
 public:
  bound_flipping_ratio_test_t(const simplex_solver_settings_t<i_t, f_t>& settings,
                              f_t start_time,
                              i_t m,
                              i_t n,
                              f_t initial_slope,
                              const std::vector<f_t>& lower,
                              const std::vector<f_t>& upper,
                              const std::vector<uint8_t>& bounded_variables,
                              const std::vector<variable_status_t>& vstatus,
                              const std::vector<i_t>& nonbasic_list,
                              const std::vector<f_t>& z,
                              const std::vector<f_t>& delta_z,
                              const std::vector<i_t>& delta_z_indices,
                              const std::vector<i_t>& nonbasic_mark)
    : settings_(settings),
      start_time_(start_time),
      m_(m),
      n_(n),
      slope_(initial_slope),
      lower_(lower),
      upper_(upper),
      bounded_variables_(bounded_variables),
      vstatus_(vstatus),
      nonbasic_list_(nonbasic_list),
      z_(z),
      delta_z_(delta_z),
      delta_z_indices_(delta_z_indices),
      nonbasic_mark_(nonbasic_mark)
  {
  }

  i_t compute_step_length(f_t& step_length, i_t& nonbasic_entering);
  f_t work_estimate() const { return work_estimate_; }

  // Timing fields (filled by compute_step_length)
  f_t time_compute_breakpoints_{0.0};
  f_t time_single_pass_{0.0};
  f_t time_coarse_filter_{0.0};
  f_t time_bucket_sort_{0.0};
  f_t time_pivot_selection_{0.0};

  // Diagnostic fields
  i_t num_buckets_used_{0};       // number of buckets in bucket sort
  i_t bucket_selected_{-1};       // which bucket the entering variable came from (-1 = single_pass/fallback)
  f_t step_length_result_{0.0};   // the step length chosen
  bool used_fallback_{false};     // true if we fell back to single_pass result
  i_t bucket0_size_{0};           // size of first bucket (candidates with ratio <= min_harris)
  i_t num_breakpoints_{0};        // total breakpoints computed
  bool selected_is_slope_breaker_{false}; // true if we selected the variable that made slope go negative
  i_t num_harris_zero_{0};        // number of harris_ratios that are exactly 0
  i_t num_exact_zero_{0};         // number of exact ratios that are exactly 0

 private:
  i_t compute_breakpoints(std::vector<i_t>& indices, std::vector<f_t>& ratios, std::vector<f_t>& harris_ratios);
  i_t single_pass(i_t start,
                  i_t end,
                  const std::vector<i_t>& indices,
                  const std::vector<f_t>& ratios,
                  f_t& slope,
                  f_t& step_length,
                  i_t& nonbasic_entering,
                  i_t& entering_index,
                  f_t& max_val);
  const std::vector<f_t>& lower_;
  const std::vector<f_t>& upper_;
  const std::vector<uint8_t>& bounded_variables_;
  const std::vector<i_t>& nonbasic_list_;
  const std::vector<variable_status_t>& vstatus_;
  const std::vector<f_t>& z_;
  const std::vector<f_t>& delta_z_;
  const std::vector<i_t>& delta_z_indices_;
  const std::vector<i_t>& nonbasic_mark_;

  const simplex_solver_settings_t<i_t, f_t>& settings_;

  f_t start_time_;
  f_t slope_;

  i_t n_;
  i_t m_;

  f_t work_estimate_{0.0};
};

}  // namespace cuopt::mathematical_optimization::simplex
