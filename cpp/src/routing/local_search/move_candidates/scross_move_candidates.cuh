/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <rmm/device_uvector.hpp>
#include "../../solution/solution_handle.cuh"
#include "cand.cuh"

namespace cuopt {
namespace routing {
namespace detail {

constexpr int max_cross_cand = 8;
constexpr int min_cross_cand = 1;

template <typename i_t, typename f_t>
class scross_move_candidates_t {
 public:
  scross_move_candidates_t(solution_handle_t<i_t, f_t> const* sol_handle_)
    : scross_best_cand_list(0, sol_handle_->get_stream()),
      route_pair_locks(0, sol_handle_->get_stream())
  {
  }

  void reset(solution_handle_t<i_t, f_t> const* sol_handle)
  {
    async_fill(scross_best_cand_list,
               cross_cand_t{0, 0, std::numeric_limits<double>::max(), 0, 0},
               sol_handle->get_stream());
    async_fill(route_pair_locks, 0, sol_handle->get_stream());
  }

  struct view_t {
    DI void insert_best_scross_candidate(i_t route_pair_idx, cross_cand_t cand)
    {
      if (cand < scross_best_cand_list[route_pair_idx]) {
        scross_best_cand_list[route_pair_idx] = cand;
      }
    }

    raft::device_span<cross_cand_t> scross_best_cand_list;
    raft::device_span<i_t> route_pair_locks;
  };

  view_t view()
  {
    view_t v;
    v.scross_best_cand_list =
      raft::device_span<cross_cand_t>{scross_best_cand_list.data(), scross_best_cand_list.size()};
    v.route_pair_locks = raft::device_span<i_t>{route_pair_locks.data(), route_pair_locks.size()};
    return v;
  }

  rmm::device_uvector<cross_cand_t> scross_best_cand_list;
  rmm::device_uvector<i_t> route_pair_locks;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
