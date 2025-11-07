/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <routing/cuda_graph.cuh>
#include <routing/solution/solution_handle.cuh>

#include <raft/core/host_span.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {

constexpr int max_relocate_size = 6;
constexpr int max_cross_size    = 3;
constexpr int max_fragment_size = std::max(max_relocate_size, max_cross_size);

template <typename i_t, typename f_t>
class vrp_move_candidates_t {
 public:
  vrp_move_candidates_t(i_t n_orders, i_t n_routes, solution_handle_t<i_t, f_t> const* sol_handle_)
    : node_id_1(n_routes * n_routes, sol_handle_->get_stream()),
      node_id_2(n_routes * n_routes, sol_handle_->get_stream()),
      frag_size_1(n_routes * n_routes, sol_handle_->get_stream()),
      frag_size_2(n_routes * n_routes, sol_handle_->get_stream()),
      move_type(n_routes * n_routes, sol_handle_->get_stream()),
      insert_offset(n_routes * n_routes, sol_handle_->get_stream()),
      cost_delta(n_routes * n_routes, sol_handle_->get_stream()),
      locks_per_route_pair(n_routes * n_routes, sol_handle_->get_stream()),
      compacted_move_indices(n_routes * n_routes, sol_handle_->get_stream()),
      n_best_route_pair_moves(sol_handle_->get_stream()),
      selected_move_indices((n_orders + n_routes), sol_handle_->get_stream()),
      best_cost_delta_per_node(n_orders + n_routes, sol_handle_->get_stream()),
      best_id_per_node(n_orders + n_routes, sol_handle_->get_stream()),
      locks_per_node(n_orders + n_routes, sol_handle_->get_stream()),
      n_of_selected_moves(sol_handle_->get_stream()),
      max_added_size(sol_handle_->get_stream())
  {
    i_t max_n_route_pairs = n_routes * n_routes;
    async_fill(
      locks_per_route_pair.data(), 0, locks_per_route_pair.size(), sol_handle_->get_stream());
    async_fill(locks_per_node.data(), 0, locks_per_node.size(), sol_handle_->get_stream());
  }

  void reset(solution_handle_t<i_t, f_t> const* sol_handle)
  {
    async_fill(cost_delta, std::numeric_limits<double>::max(), sol_handle->get_stream());
    async_fill(
      best_cost_delta_per_node, std::numeric_limits<double>::max(), sol_handle->get_stream());
    async_fill(best_id_per_node, -1, sol_handle->get_stream());
    n_best_route_pair_moves.set_value_to_zero_async(sol_handle->get_stream());
    max_added_size.set_value_async(max_fragment_size, sol_handle->get_stream());
  }

  struct view_t {
    DI void record_candidate(i_t route_pair_idx,
                             i_t node_id_1_,
                             i_t node_id_2_,
                             i_t frag_size_1_,
                             i_t frag_size_2_,
                             i_t move_type_,
                             i_t insert_offset_,
                             double cost_delta_,
                             raft::device_span<i_t>& active_nodes_impacted)
    {
      // we want to store the best per node, to retry the moves on changed routes
      if (cost_delta_ < best_cost_delta_per_node[node_id_1_]) {
        acquire_lock(&locks_per_node[node_id_1_]);
        if (cost_delta_ < best_cost_delta_per_node[node_id_1_]) {
          best_id_per_node[node_id_1_]         = node_id_2_;
          best_cost_delta_per_node[node_id_1_] = cost_delta_;
        }
        release_lock(&locks_per_node[node_id_1_]);
      }
      // atomicExch(&active_nodes_impacted[node_id_1_], 1);

      if (cost_delta_ > cost_delta[route_pair_idx]) return;
      acquire_lock(&locks_per_route_pair[route_pair_idx]);
      if (cost_delta_ < cost_delta[route_pair_idx]) {
        node_id_1[route_pair_idx]     = node_id_1_;
        node_id_2[route_pair_idx]     = node_id_2_;
        frag_size_1[route_pair_idx]   = frag_size_1_;
        frag_size_2[route_pair_idx]   = frag_size_2_;
        move_type[route_pair_idx]     = move_type_;
        insert_offset[route_pair_idx] = insert_offset_;
        cost_delta[route_pair_idx]    = cost_delta_;
      }
      release_lock(&locks_per_route_pair[route_pair_idx]);
    }

    DI i_t get_route_pair_idx(i_t r_1, i_t r_2, i_t n_routes)
    {
      i_t route_pair_idx;
      // get the route pair index with respect to the upper triangular matrix
      if (r_2 > r_1) {
        route_pair_idx = r_1 * n_routes + r_2;
      } else {
        route_pair_idx = r_2 * n_routes + r_1;
      }
      return route_pair_idx;
    }

    DI void record_move(i_t move_idx, i_t n_moves_found)
    {
      selected_move_indices[n_moves_found] = move_idx;
    }

    DI void set_n_changed_routes(i_t n_moves_found) { *n_of_selected_moves = n_moves_found; }

    raft::device_span<i_t> node_id_1;
    raft::device_span<i_t> node_id_2;
    raft::device_span<i_t> frag_size_1;
    raft::device_span<i_t> frag_size_2;
    raft::device_span<i_t> move_type;
    raft::device_span<i_t> insert_offset;
    raft::device_span<double> cost_delta;
    raft::device_span<i_t> locks_per_route_pair;
    raft::device_span<i_t> compacted_move_indices;
    raft::device_span<i_t> selected_move_indices;
    raft::device_span<double> best_cost_delta_per_node;
    raft::device_span<i_t> locks_per_node;
    raft::device_span<i_t> best_id_per_node;
    i_t* n_of_selected_moves;
    i_t* n_best_route_pair_moves;
    i_t* max_added_size;
  };

  view_t view()
  {
    view_t v;
    v.node_id_1     = raft::device_span<i_t>{node_id_1.data(), node_id_1.size()};
    v.node_id_2     = raft::device_span<i_t>{node_id_2.data(), node_id_2.size()};
    v.frag_size_1   = raft::device_span<i_t>{frag_size_1.data(), frag_size_1.size()};
    v.frag_size_2   = raft::device_span<i_t>{frag_size_2.data(), frag_size_2.size()};
    v.move_type     = raft::device_span<i_t>{move_type.data(), move_type.size()};
    v.insert_offset = raft::device_span<i_t>{insert_offset.data(), insert_offset.size()};
    v.cost_delta    = raft::device_span<double>{cost_delta.data(), cost_delta.size()};
    v.locks_per_route_pair =
      raft::device_span<i_t>{locks_per_route_pair.data(), locks_per_route_pair.size()};
    v.compacted_move_indices =
      raft::device_span<i_t>{compacted_move_indices.data(), compacted_move_indices.size()};
    v.selected_move_indices =
      raft::device_span<i_t>{selected_move_indices.data(), selected_move_indices.size()};
    v.best_id_per_node = raft::device_span<i_t>{best_id_per_node.data(), best_id_per_node.size()};
    v.locks_per_node   = raft::device_span<i_t>{locks_per_node.data(), locks_per_node.size()};
    v.best_cost_delta_per_node =
      raft::device_span<double>{best_cost_delta_per_node.data(), best_cost_delta_per_node.size()};
    v.n_of_selected_moves     = n_of_selected_moves.data();
    v.n_best_route_pair_moves = n_best_route_pair_moves.data();
    v.max_added_size          = max_added_size.data();
    return v;
  }

  rmm::device_uvector<i_t> node_id_1;
  rmm::device_uvector<i_t> node_id_2;
  rmm::device_uvector<i_t> frag_size_1;
  rmm::device_uvector<i_t> frag_size_2;
  rmm::device_uvector<i_t> move_type;
  rmm::device_uvector<i_t> insert_offset;
  rmm::device_uvector<double> cost_delta;
  rmm::device_uvector<i_t> locks_per_route_pair;
  rmm::device_uvector<i_t> compacted_move_indices;
  rmm::device_uvector<i_t> selected_move_indices;
  rmm::device_uvector<double> best_cost_delta_per_node;
  rmm::device_uvector<i_t> best_id_per_node;
  rmm::device_uvector<i_t> locks_per_node;
  rmm::device_scalar<i_t> n_of_selected_moves;
  rmm::device_scalar<i_t> n_best_route_pair_moves;
  rmm::device_scalar<i_t> max_added_size;

  cuda_graph_t find_kernel_graph;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
