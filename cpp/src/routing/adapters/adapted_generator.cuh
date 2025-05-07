/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "../solution/pool_allocator.cuh"
#include "adapted_sol.cuh"

#include <utilities/timer.hpp>

namespace cuopt::routing::detail {

template <typename i_t, typename f_t, request_t REQUEST>
struct adapted_generator_t {
  using allocator = detail::
    pool_allocator_t<i_t, f_t, detail::solution_t<i_t, f_t, REQUEST>, detail::problem_t<i_t, f_t>>;
  adapted_generator_t(const problem_t<i_t, f_t>& problem_, allocator& pool_allocator_);
  // make feasible function greedily ejects
  bool make_feasible(adapted_sol_t<i_t, f_t, REQUEST>& adapted_solution,
                     f_t time_limit,
                     costs const& weight,
                     bool clear_scores);
  // creates an initial solution until the given time budget or n_routes are reached
  void generate_solution(adapted_sol_t<i_t, f_t, REQUEST>& sol,
                         const std::vector<i_t>& desired_vehicle_ids,
                         f_t time_limit,
                         costs const& weight,
                         const timer_t& timer);
  const problem_t<i_t, f_t>& problem;
  allocator& pool_allocator;
};

}  // namespace cuopt::routing::detail
