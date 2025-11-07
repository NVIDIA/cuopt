/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <routing/adapters/adapted_sol.cuh>
#include <routing/diversity/macros.hpp>

#include <array>

namespace cuopt::routing::detail {

template <typename i_t, typename f_t, request_t REQUEST>
struct optimal_cycles_t {
  using allocator = detail::
    pool_allocator_t<i_t, f_t, detail::solution_t<i_t, f_t, REQUEST>, detail::problem_t<i_t, f_t>>;
  optimal_cycles_t(allocator& pool_allocator_);
  template <request_t r_t = REQUEST, std::enable_if_t<r_t == request_t::PDP, bool> = true>
  bool add_cycles_request(adapted_sol_t<i_t, f_t, REQUEST>& a,
                          std::vector<std::vector<NodeInfo<>>>& cycles,
                          costs final_weight);
  template <request_t r_t = REQUEST, std::enable_if_t<r_t == request_t::VRP, bool> = true>
  bool add_cycles_request(adapted_sol_t<i_t, f_t, REQUEST>& a,
                          std::vector<std::vector<NodeInfo<>>>& cycles,
                          costs final_weight);

  void get_min_delta_and_index(adapted_sol_t<i_t, f_t, REQUEST>& sol, i_t num_items);
  bool insert_cycle_to_found_position(adapted_sol_t<i_t, f_t, REQUEST>& sol, i_t cycle_size);

  void find_best_rotate_cycle(std::vector<NodeInfo<>>& cycle, adapted_sol_t<i_t, f_t, REQUEST>& s);

  void insert_cycle_to_route_request(std::vector<NodeInfo<>>& cycle,
                                     size_t route_id,
                                     adapted_sol_t<i_t, f_t, REQUEST>& s);

  allocator& pool_allocator;
  std::deque<NodeInfo<>> cycle_helper;
  std::deque<NodeInfo<>> best_so_far;
  std::unordered_map<i_t, size_t> in_cycle;

  rmm::device_uvector<NodeInfo<>> d_cycle;
  rmm::device_uvector<double> eax_cycle_delta;
  rmm::device_uvector<std::byte> d_cub_storage_bytes;
  rmm::device_scalar<cub::KeyValuePair<i_t, double>> index_delta_pair;
  dimensions_route_t<i_t, f_t, REQUEST> eax_fragment;
};

}  // namespace cuopt::routing::detail
