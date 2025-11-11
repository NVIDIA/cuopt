/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../../solution/solution_handle.cuh"

#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class random_move_candidates_t {
 public:
  random_move_candidates_t(i_t fleet_size,
                           i_t n_orders,
                           solution_handle_t<i_t, f_t> const* sol_handle)
    : moves_per_route_pair(n_orders * n_orders, sol_handle->get_stream()),
      move_begin_offset(fleet_size * fleet_size, sol_handle->get_stream()),
      move_end_offset(fleet_size * fleet_size, sol_handle->get_stream()),
      selected_move_indices(fleet_size * fleet_size, sol_handle->get_stream()),
      d_cub_storage_bytes(0, sol_handle->get_stream()),
      n_moves(sol_handle->get_stream()),
      n_selected_moves(sol_handle->get_stream())
  {
  }

  void reset(solution_handle_t<i_t, f_t> const* sol_handle)
  {
    constexpr i_t zero_val = 0;
    async_fill(moves_per_route_pair,
               int2{std::numeric_limits<int>::max(), std::numeric_limits<int>::max()},
               sol_handle->get_stream());
    async_fill(move_begin_offset, 0, sol_handle->get_stream());
    async_fill(move_end_offset, 0, sol_handle->get_stream());
    n_moves.set_value_async(zero_val, sol_handle->get_stream());
    n_selected_moves.set_value_async(zero_val, sol_handle->get_stream());
  }

  struct view_t {
    i_t* n_moves;
    i_t* n_selected_moves;
    raft::device_span<int2> moves_per_route_pair;
    raft::device_span<i_t> move_begin_offset;
    raft::device_span<i_t> move_end_offset;
    raft::device_span<i_t> selected_move_indices;
  };

  view_t view()
  {
    view_t v;
    v.n_moves          = n_moves.data();
    v.n_selected_moves = n_selected_moves.data();
    v.moves_per_route_pair =
      raft::device_span<int2>(moves_per_route_pair.data(), moves_per_route_pair.size());
    v.move_begin_offset =
      raft::device_span<i_t>(move_begin_offset.data(), move_begin_offset.size());
    v.move_end_offset = raft::device_span<i_t>(move_end_offset.data(), move_end_offset.size());
    v.selected_move_indices =
      raft::device_span<i_t>(selected_move_indices.data(), selected_move_indices.size());
    return v;
  }

  rmm::device_scalar<i_t> n_moves;
  rmm::device_scalar<i_t> n_selected_moves;
  rmm::device_uvector<int2> moves_per_route_pair;
  rmm::device_uvector<i_t> move_begin_offset;
  rmm::device_uvector<i_t> move_end_offset;
  rmm::device_uvector<i_t> selected_move_indices;
  rmm::device_uvector<std::byte> d_cub_storage_bytes;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
