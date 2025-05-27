/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <mip/presolve/load_balanced_bounds_presolve.cuh>
#include <mip/problem/load_balanced_problem.cuh>
#include "bounds_presolve.cuh"

#include <mip/utils.cuh>

#include <utilities/timer.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class bound_presolve_t;

template <typename i_t, typename f_t>
class load_balanced_bounds_presolve_t;

template <typename i_t, typename f_t>
class load_balanced_problem_t;

/*
  Probing cache is a set of implied bounds when we set a variable to some value.
  We keep two sets of changed bounds for each interval:
  For binary: 0 and 1
  For integer: finite_bound and finite_bound<> if it is unbounded on one side.
  Notice that we keep an interval here.
  Else if both sides are bounded, we do interval/2 > and interval/2 <.
  We can use this cache, for infeasibility detection, implied bounds, fast bounds setting and bulk
  rounding. To save from memory, we will keep the the results in host map.
*/

enum interval_type_t { EQUALS = 0, LEQ, GEQ };

template <typename i_t, typename f_t>
struct val_interval_t {
  void fill_cache_hits(i_t interval,
                       f_t first_probe,
                       f_t second_probe,
                       i_t& hit_interval_for_first_probe,
                       i_t& hit_interval_for_second_probe)
  {
    if (interval_type == interval_type_t::EQUALS) {
      if (val == first_probe) { hit_interval_for_first_probe = interval; }
      if (val == second_probe) { hit_interval_for_second_probe = interval; }
    } else if (interval_type == interval_type_t::LEQ) {
      if (val >= first_probe) { hit_interval_for_first_probe = interval; }
      if (val >= second_probe) { hit_interval_for_second_probe = interval; }
    } else if (interval_type == interval_type_t::GEQ) {
      if (val <= first_probe) { hit_interval_for_first_probe = interval; }
      if (val <= second_probe) { hit_interval_for_second_probe = interval; }
    }
  }
  f_t val;
  interval_type_t interval_type;
};

template <typename f_t>
struct cached_bound_t {
  f_t lb;
  f_t ub;
};

template <typename i_t, typename f_t>
struct cache_entry_t {
  val_interval_t<i_t, f_t> val_interval;
  std::unordered_map<i_t, cached_bound_t<f_t>> var_to_cached_bound_map;
};

template <typename i_t, typename f_t>
class probing_cache_t {
 public:
  bool contains(problem_t<i_t, f_t>& problem, i_t var_id);
  void update_bounds_with_selected(std::vector<f_t>& host_lb,
                                   std::vector<f_t>& host_ub,
                                   const cache_entry_t<i_t, f_t>& cache_entry,
                                   const std::vector<i_t>& reverse_original_ids);
  i_t check_number_of_conflicting_vars(const std::vector<f_t>& host_lb,
                                       const std::vector<f_t>& host_ub,
                                       const cache_entry_t<i_t, f_t>& cache_entry,
                                       f_t integrality_tolerance,
                                       const std::vector<i_t>& reverse_original_ids);
  // check if there are any conflicting bounds
  f_t get_least_conflicting_rounding(problem_t<i_t, f_t>& problem,
                                     std::vector<f_t>& host_lb,
                                     std::vector<f_t>& host_ub,
                                     i_t var_id_on_problem,
                                     f_t first_probe,
                                     f_t second_probe,
                                     f_t integrality_tolerance);

  std::unordered_map<i_t, std::array<cache_entry_t<i_t, f_t>, 2>> probing_cache;
  std::mutex probing_cache_mutex;
};

template <typename i_t, typename f_t>
class lb_probing_cache_t {
 public:
  bool contains(problem_t<i_t, f_t>& problem, i_t var_id);
  void update_bounds_with_selected(std::vector<f_t>& host_bounds,
                                   const cache_entry_t<i_t, f_t>& cache_entry,
                                   const std::vector<i_t>& reverse_original_ids);
  i_t check_number_of_conflicting_vars(const std::vector<f_t>& host_bounds,
                                       const cache_entry_t<i_t, f_t>& cache_entry,
                                       f_t integrality_tolerance,
                                       const std::vector<i_t>& reverse_original_ids);
  // check if there are any conflicting bounds
  f_t get_least_conflicting_rounding(problem_t<i_t, f_t>& problem,
                                     std::vector<f_t>& host_bounds,
                                     i_t var_id_on_problem,
                                     f_t first_probe,
                                     f_t second_probe,
                                     f_t integrality_tolerance);

  std::unordered_map<i_t, std::array<cache_entry_t<i_t, f_t>, 2>> probing_cache;
};

template <typename i_t, typename f_t>
void compute_probing_cache(bound_presolve_t<i_t, f_t>& bound_presolve,
                           problem_t<i_t, f_t>& problem,
                           timer_t timer);

template <typename i_t, typename f_t>
void compute_probing_cache(load_balanced_bounds_presolve_t<i_t, f_t>& bound_presolve,
                           load_balanced_problem_t<i_t, f_t>& problem,
                           timer_t timer);

}  // namespace cuopt::linear_programming::detail
