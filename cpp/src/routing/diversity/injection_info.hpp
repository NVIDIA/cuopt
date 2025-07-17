/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include <utilities/copy_helpers.hpp>
#include "../solution/solution.cuh"

#include <set>

namespace cuopt {
namespace routing {

template <class allocator, class solution, class problem>
struct injection_info_t {
  injection_info_t(const problem* p_, allocator& pool_allocator_)
    : p(p_), pool_allocator(pool_allocator_), accepted(0)
  {
  }

  auto has_info() const
  {
    auto [vehicle_ids, routes, node_types, sol_offsets] = p->data_view_ptr->get_initial_solutions();
    return !vehicle_ids.empty();
  }

  std::vector<solution> load_solutions()
  {
    auto stream = pool_allocator.sol_handles[0]->get_stream();
    auto [d_vehicle_ids, d_routes, d_types, d_sol_offsets] =
      p->data_view_ptr->get_initial_solutions();
    auto sol_offsets = cuopt::host_copy(d_sol_offsets, stream);
    auto tmp_routes  = cuopt::host_copy(d_routes, stream);
    auto vehicle_ids = cuopt::host_copy(d_vehicle_ids, stream);
    auto node_types  = cuopt::host_copy(d_types, stream);

    n_sol = sol_offsets.size() - 1;

    accepted.resize(n_sol, -1);

    std::vector<std::pair<int, std::vector<detail::NodeInfo<>>>> sol_routes;
    std::vector<detail::NodeInfo<>> new_route;
    std::vector<int> desired_vehicle_ids;
    std::set<int> added_node_ids;

    int sol_n_routes = 0;

    for (size_t i = 0; i < sol_offsets.size() - 1; ++i) {
      auto begin = sol_offsets[i];
      auto end   = sol_offsets[i + 1];
      sol_routes.clear();
      desired_vehicle_ids.clear();
      added_node_ids.clear();
      new_route.clear();
      sol_n_routes         = 0;
      auto curr_vehicle_id = vehicle_ids[begin];

      for (int j = begin; j < end; ++j) {
        if (node_types[j] == node_type_t::DEPOT || node_types[j] == node_type_t::BREAK) {
          continue;
        }

        if (vehicle_ids[j] != curr_vehicle_id) {
          desired_vehicle_ids.push_back(curr_vehicle_id);
          sol_routes.push_back({sol_n_routes++, new_route});
          curr_vehicle_id = vehicle_ids[j];
          new_route.clear();
        }

        new_route.push_back(p->get_node_info_of_node(tmp_routes[j]));
        cuopt_expects(added_node_ids.count(tmp_routes[j]) == 0,
                      error_type_t::ValidationError,
                      "Duplicate order id");
        added_node_ids.insert(tmp_routes[j]);
      }

      if (p->order_info.is_pdp()) {
        for (int node_id : added_node_ids) {
          int brother_id = p->get_brother_node_info(p->get_node_info_of_node(node_id)).node();
          auto cond      = added_node_ids.count(brother_id) > 0;
          if (!cond) {
            std::cout << "node_id: " << node_id << " brother_id: " << brother_id << std::endl;
          }
          cuopt_expects(cond, error_type_t::ValidationError, "Brother node is not served");
        }
      }

      desired_vehicle_ids.push_back(curr_vehicle_id);
      sol_routes.push_back({sol_n_routes++, new_route});
      cuopt_expects(sol_n_routes <= p->get_fleet_size(),
                    error_type_t::ValidationError,
                    "One solution has more vehicles than the fleet size");

      if (!p->has_prize_collection()) {
        cuopt_expects(
          p->get_num_orders() - (int)p->order_info.depot_included_ == (int)added_node_ids.size(),
          error_type_t::ValidationError,
          "Inconsistent order ids");
        cuopt_expects((p->get_num_orders() - 1) == *std::prev(added_node_ids.end()),
                      error_type_t::ValidationError,
                      "Inconsistent order ids");
        cuopt_expects((int)p->order_info.depot_included_ == *added_node_ids.begin(),
                      error_type_t::ValidationError,
                      "Inconsistent order ids");
      }

      cuopt_assert(desired_vehicle_ids.size() == sol_n_routes, "Inconsitent desired vehicle ids");

      solution S(p, pool_allocator.sol_handles[0].get(), desired_vehicle_ids);
      std::vector<int> sequence(sol_n_routes);
      std::iota(sequence.begin(), sequence.end(), 0);
      S.remove_routes(sequence);
      S.add_new_routes(sol_routes);
      S.sol.global_runtime_checks(false, false, "Check solution after injection");
      solutions.emplace_back(std::move(S));
    }
    return solutions;
  }

  const problem* p;
  allocator& pool_allocator;
  std::vector<solution> solutions;
  std::vector<int> accepted;
  int n_sol{};
};

}  // namespace routing
}  // namespace cuopt
