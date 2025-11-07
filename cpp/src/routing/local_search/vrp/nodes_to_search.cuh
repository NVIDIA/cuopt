/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/vector_helpers.cuh>
#include "../../cuda_graph.cuh"
#include "../../solution/solution.cuh"
#include "../../solution/solution_handle.cuh"

#include <rmm/device_uvector.hpp>
#include <utilities/vector_helpers.cuh>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class move_candidates_t;

// we try inserting more after depot as longer edges tend to appear there
constexpr int after_depot_insertion_multiplier = 4;

// this enum works both as a classifier of different LS moves and also cumulative number of move
// types(including fragments) to distinguish blockids
enum class vrp_move_t {
  // we have in total 4 versions of cross (normal, reversed-normal, normal-reversed,
  // reversed-reversed), and in total 9 combinations of each, also positions +-1 brings 3 different
  // insertions TOTAL=108
  CROSS = 107,
  // we have in total 6 fragment sizes to relocate, reversed/non-reversed 2. TOTAL=12
  // we do both directions, relocating the changed fragments to unchanged routes too (there might be
  // some overlap here)
  RELOCATE = 131,
  // there is only one type of 2-opt* and we can form the routes in two different ways in
  // heterogeneous setting.
  TWO_OPT_STAR = 133,
  SIZE         = 134
};

template <typename i_t, typename f_t>
class nodes_to_search_t {
 public:
  nodes_to_search_t(i_t n_orders, i_t n_routes, solution_handle_t<i_t, f_t> const* sol_handle_)
    : n_nodes_to_search(sol_handle_->get_stream()),
      nodes_to_search(n_orders + after_depot_insertion_multiplier * n_routes,
                      sol_handle_->get_stream()),
      sampled_nodes_to_search(n_orders + after_depot_insertion_multiplier * n_routes,
                              sol_handle_->get_stream()),
      active_nodes_impacted(n_orders + n_routes, sol_handle_->get_stream()),
      recycled_node_pairs(n_orders + after_depot_insertion_multiplier * n_routes,
                          sol_handle_->get_stream()),
      h_active_nodes_impacted(n_orders + n_routes),
      h_recycled_node_pairs(n_orders + after_depot_insertion_multiplier * n_routes),
      h_best_id_per_node(n_orders + n_routes)
  {
  }

  // void restore_nodes_from_modified_routes(solution_t<i_t, f_t, request_t::VRP>& sol);
  void restore_found_nodes(solution_t<i_t, f_t, request_t::VRP>& sol);
  bool sample_nodes_to_search(const solution_t<i_t, f_t, request_t::VRP>& sol,
                              std::mt19937& rng,
                              bool full_set = false);
  bool sample_nodes_for_recycle(const solution_t<i_t, f_t, request_t::VRP>& sol,
                                move_candidates_t<i_t, f_t>& move_candidates);

  void reset(solution_handle_t<i_t, f_t> const* sol_handle)
  {
    n_nodes_to_search.set_value_to_zero_async(sol_handle->get_stream());
  }

  void reset_active_nodes(solution_handle_t<i_t, f_t> const* sol_handle)
  {
    async_fill(active_nodes_impacted, 0, sol_handle->get_stream());
  }

  struct view_t {
    i_t* n_nodes_to_search;
    i_t n_sampled_nodes;
    raft::device_span<NodeInfo<i_t>> nodes_to_search;
    raft::device_span<NodeInfo<i_t>> sampled_nodes_to_search;
    raft::device_span<i_t> active_nodes_impacted;
    // raft::device_span<i_t> routes_modified;
    raft::device_span<int2> recycled_node_pairs;
  };

  view_t view()
  {
    view_t v;
    v.n_nodes_to_search = n_nodes_to_search.data();
    v.n_sampled_nodes   = n_sampled_nodes;
    v.nodes_to_search =
      raft::device_span<NodeInfo<i_t>>{nodes_to_search.data(), nodes_to_search.size()};
    v.sampled_nodes_to_search = raft::device_span<NodeInfo<i_t>>{sampled_nodes_to_search.data(),
                                                                 sampled_nodes_to_search.size()};
    v.active_nodes_impacted =
      raft::device_span<i_t>{active_nodes_impacted.data(), active_nodes_impacted.size()};
    // v.routes_modified =
    //   raft::device_span<i_t>{routes_modified.data(), routes_modified.size()};
    v.recycled_node_pairs =
      raft::device_span<int2>{recycled_node_pairs.data(), recycled_node_pairs.size()};
    return v;
  }

  rmm::device_scalar<i_t> n_nodes_to_search;
  rmm::device_uvector<NodeInfo<i_t>> nodes_to_search;
  rmm::device_uvector<NodeInfo<i_t>> sampled_nodes_to_search;
  rmm::device_uvector<i_t> active_nodes_impacted;
  // rmm::device_uvector<i_t> routes_modified;
  rmm::device_uvector<int2> recycled_node_pairs;

  std::vector<NodeInfo<i_t>> h_sampled_nodes;
  std::vector<NodeInfo<i_t>> h_nodes_to_search;
  std::vector<i_t> h_active_nodes_impacted;
  std::vector<int2> h_recycled_node_pairs;
  std::vector<i_t> h_best_id_per_node;
  i_t n_sampled_nodes;

  cuda_graph_t sample_nodes_graph;
  cuda_graph_t extract_nodes_graph;
};

template <typename i_t, typename f_t, request_t REQUEST>
void extract_nodes_to_search(solution_t<i_t, f_t, REQUEST>& sol,
                             move_candidates_t<i_t, f_t>& move_candidates);

template <typename i_t, typename f_t, request_t REQUEST>
void sample_nodes(solution_t<i_t, f_t, REQUEST>& sol, move_candidates_t<i_t, f_t>& move_candidates);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
