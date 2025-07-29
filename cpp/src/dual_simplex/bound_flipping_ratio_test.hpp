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
#pragma once

#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>

#include <vector>

namespace cuopt::linear_programming::dual_simplex {

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

 private:
  i_t compute_breakpoints(std::vector<i_t>& indices, std::vector<f_t>& ratios);
  i_t single_pass(i_t start,
                  i_t end,
                  const std::vector<i_t>& indices,
                  const std::vector<f_t>& ratios,
                  f_t& slope,
                  f_t& step_length,
                  i_t& nonbasic_entering,
                  i_t& enetering_index);
  void heap_passes(const std::vector<i_t>& current_indicies,
                   const std::vector<f_t>& current_ratios,
                   i_t num_breakpoints,
                   f_t& slope,
                   f_t& step_lenght,
                   i_t& nonbasic_entering,
                   i_t& entering_index);

  void bucket_pass(const std::vector<i_t>& current_indicies,
                   const std::vector<f_t>& current_ratios,
                   i_t num_breakpoints,
                   f_t& slope,
                   f_t& step_length,
                   i_t& nonbasic_entering,
                   i_t& entering_index);

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
};

}  // namespace cuopt::linear_programming::dual_simplex
