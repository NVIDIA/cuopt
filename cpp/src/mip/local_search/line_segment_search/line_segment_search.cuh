/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <mip/feasibility_jump/feasibility_jump.cuh>
#include <mip/local_search/rounding/constraint_prop.cuh>
#include <utilities/timer.hpp>

namespace cuopt::linear_programming::detail {

struct line_segment_settings_t {
  bool recombiner_mode        = false;
  double best_of_parents_cost = std::numeric_limits<double>::max();
  bool parents_infeasible     = false;
};

template <typename i_t, typename f_t>
class line_segment_search_t {
 public:
  line_segment_search_t() = delete;
  line_segment_search_t(fj_t<i_t, f_t>& fj, constraint_prop_t<i_t, f_t>& constraint_prop);
  bool search_line_segment(solution_t<i_t, f_t>& solution,
                           const rmm::device_uvector<f_t>& point_1,
                           const rmm::device_uvector<f_t>& point_2,
                           i_t n_points_to_search,
                           bool is_feasibility_run,
                           cuopt::timer_t& timer);

  bool search_line_segment(solution_t<i_t, f_t>& solution,
                           const rmm::device_uvector<f_t>& point_1,
                           const rmm::device_uvector<f_t>& point_2,
                           i_t n_points_to_search,
                           const rmm::device_uvector<f_t>& delta_vector,
                           bool is_feasibility_run,
                           cuopt::timer_t& timer);

  void save_solution_if_better(solution_t<i_t, f_t>& solution,
                               const rmm::device_uvector<f_t>& point_1,
                               const rmm::device_uvector<f_t>& point_2,
                               rmm::device_uvector<f_t>& best_assignment,
                               rmm::device_uvector<f_t>& best_feasible_assignment,
                               f_t& best_cost,
                               f_t& best_feasible_cost,
                               f_t curr_cost);

  fj_t<i_t, f_t>& fj;
  constraint_prop_t<i_t, f_t>& constraint_prop;
  line_segment_settings_t settings;
};

}  // namespace cuopt::linear_programming::detail
