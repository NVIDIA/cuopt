/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "../../solution/solution_handle.cuh"

#include <thrust/count.h>
#include <raft/core/host_span.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {

struct __align__(16) prize_cand_t {
  HDI prize_cand_t(int pickup_insertion_,
                   int delivery_insertion_,
                   int ejected_node_id_,
                   int inserted_node_id_,
                   double cost_)
    : pickup_insertion(pickup_insertion_),
      delivery_insertion(delivery_insertion_),
      ejected_node_id(ejected_node_id_),
      inserted_node_id(inserted_node_id_),
      cost(cost_)
  {
  }

  uint16_t pickup_insertion   = 0;
  uint16_t delivery_insertion = 0;
  uint16_t ejected_node_id    = 0;
  uint16_t inserted_node_id   = 0;
  double cost                 = std::numeric_limits<double>::max();
};

struct is_candidate_improving {
  __device__ bool operator()(const prize_cand_t& x) { return x.cost < -EPSILON; }
};

template <typename i_t, typename f_t>
class prize_move_candidates_t {
 public:
  prize_move_candidates_t(i_t fleet_size, solution_handle_t<i_t, f_t> const* sol_handle)
    : best_cand_per_route(fleet_size, sol_handle->get_stream()),
      locks_per_route(fleet_size, sol_handle->get_stream())
  {
    async_fill(best_cand_per_route,
               prize_cand_t{0, 0, 0, 0, std::numeric_limits<double>::max()},
               sol_handle->get_stream());
    async_fill(locks_per_route, 0, sol_handle->get_stream());
  }

  void reset(solution_handle_t<i_t, f_t> const* sol_handle)
  {
    async_fill(best_cand_per_route,
               prize_cand_t{0, 0, 0, 0, std::numeric_limits<double>::max()},
               sol_handle->get_stream());

    // no need to reset locks because they will be released by threads eventually
  }

  bool has_improving_routes(solution_handle_t<i_t, f_t> const* sol_handle) const
  {
    return thrust::count_if(sol_handle->get_thrust_policy(),
                            best_cand_per_route.begin(),
                            best_cand_per_route.end(),
                            is_candidate_improving()) > 0;
  }

  struct view_t {
    raft::device_span<prize_cand_t> best_cand_per_route;
    raft::device_span<i_t> locks_per_route;
  };

  view_t view()
  {
    view_t v;
    v.best_cand_per_route =
      raft::device_span<prize_cand_t>{best_cand_per_route.data(), best_cand_per_route.size()};
    v.locks_per_route = raft::device_span<i_t>(locks_per_route.data(), locks_per_route.size());
    return v;
  }

  rmm::device_uvector<prize_cand_t> best_cand_per_route;
  rmm::device_uvector<i_t> locks_per_route;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
