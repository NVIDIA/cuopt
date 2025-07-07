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

#include "adapted_sol.cuh"

#include <routing/crossovers/optimal_eax_cycles.cuh>
#include <routing/diversity/macros.hpp>

#include <array>

namespace cuopt::routing::detail {

template <typename i_t, typename f_t, request_t REQUEST>
struct adapted_modifier_t {
  using allocator = detail::
    pool_allocator_t<i_t, f_t, detail::solution_t<i_t, f_t, REQUEST>, detail::problem_t<i_t, f_t>>;
  adapted_modifier_t(allocator& pool_allocator_);
  // improves the solution with feasible local search
  void improve(adapted_sol_t<i_t, f_t, REQUEST>& adapted_solution,
               costs final_weight,
               f_t time_limit,
               bool run_cycle_finder = true);
  void perturbate(adapted_sol_t<i_t, f_t, REQUEST>& adapted_solution,
                  costs final_weight,
                  i_t perturbation_count = 20);
  void add_unserviced_request(adapted_sol_t<i_t, f_t, REQUEST>& adapted_solution,
                              costs final_weight);

  void add_selected_unserviced_requests(adapted_sol_t<i_t, f_t, REQUEST>& adapted_solution,
                                        const std::vector<i_t>& unserviced_nodes,
                                        costs final_weight);

  void equalize_routes_and_nodes(adapted_sol_t<i_t, f_t, REQUEST>& sol_a,
                                 adapted_sol_t<i_t, f_t, REQUEST>& sol_b,
                                 costs final_weight,
                                 bool skip_adding_nodes_to_a = false);

  void insert_infeasible_nodes(adapted_sol_t<i_t, f_t, REQUEST>& sol, costs& weights);

  bool make_cluster_order_feasible_request(adapted_sol_t<i_t, f_t, REQUEST>& adapted_solution,
                                           costs final_weight);
  bool add_cycles_request(adapted_sol_t<i_t, f_t, REQUEST>& a,
                          std::vector<std::vector<NodeInfo<>>>& cycles,
                          costs final_weight);
  bool eject_request_infeasible_nodes(adapted_sol_t<i_t, f_t, REQUEST>& sol);

  void squeeze_breaks(adapted_sol_t<i_t, f_t, REQUEST>& a, costs& weights);

  allocator& pool_allocator;
  std::vector<NodeInfo<>> helper_nodes;
  std::unordered_set<NodeInfo<>, NodeInfoHash> helper_set;

  // variables for EAX optimal cycle adding
  optimal_cycles_t<i_t, f_t, REQUEST> optimal_cycles;
};

}  // namespace cuopt::routing::detail
