/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "cycle.hpp"
#include "cycle_graph.hpp"
#include "device_bitset.hpp"
#include "device_map.cuh"

#include "../../solution/solution_handle.cuh"

#include <rmm/cuda_stream_view.hpp>

namespace cuopt {
namespace routing {
namespace detail {

//! The key structure of a valid negative path as described in the paper of Ahuja, Orlin and Sharma.
//! We store start (tail) and end (head) of the path. Also set of visited routes is stored (label).
template <size_t max_routes = 256>
struct key_t {
  uint32_t tail;
  uint32_t head;
  device_bitset_t<max_routes> label;
  HDI key_t(uint32_t tail_ = UINT32_MAX, uint32_t head_ = UINT32_MAX) : tail(tail_), head(head_) {}

  constexpr bool empty() noexcept { return tail == UINT32_MAX; }
  constexpr void set_empty() noexcept
  {
    tail = UINT32_MAX;
    label.reset();
  }
};

template <size_t max_routes = 256>
constexpr bool operator==(const key_t<max_routes>& k, const key_t<max_routes>& k1) noexcept
{
  // The hash funcion relies on the fact that the bitset size is the multiple of 64
  static_assert((64 * (max_routes / 64) == max_routes), "max_routes has to be a multiple of 64");

  return (k.head == k1.head) && (k.tail == k1.tail) && (k.label == k1.label);
}

template <typename i_t, typename f_t, size_t max_routes>
struct path_t {
  path_t(solution_handle_t<i_t, f_t> const* handle_ptr_)
    : key_ptr(max_routes, handle_ptr_->get_stream()),
      cost_ptr(max_routes, handle_ptr_->get_stream()),
      level_ptr(max_routes, handle_ptr_->get_stream()),
      n_cycles(0, handle_ptr_->get_stream()),
      all_mask(device_bitset_t<max_routes>{}, handle_ptr_->get_stream()),
      all_found(false, handle_ptr_->get_stream())
  {
  }

  void reset(rmm::cuda_stream_view stream)
  {
    n_cycles.set_value_to_zero_async(stream);
    all_found.set_value_to_zero_async(stream);
    const auto zero_bitset = device_bitset_t<max_routes>{};
    all_mask.set_value_async(zero_bitset, stream);
  }

  struct view_t {
    key_t<max_routes>* key_ptr{nullptr};
    double* cost_ptr{nullptr};
    i_t* level_ptr{nullptr};
    i_t* n_cycles{nullptr};
    bool* all_found{nullptr};
    device_bitset_t<max_routes>* all_mask{nullptr};
  };

  view_t subspan(i_t cycle_id)
  {
    view_t v;
    v.key_ptr   = key_ptr.data() + cycle_id;
    v.cost_ptr  = cost_ptr.data() + cycle_id;
    v.level_ptr = level_ptr.data() + cycle_id;
    v.n_cycles  = n_cycles.data();
    v.all_found = all_found.data();
    v.all_mask  = all_mask.data();
    return v;
  }

  view_t view()
  {
    view_t v;
    v.key_ptr   = key_ptr.data();
    v.cost_ptr  = cost_ptr.data();
    v.level_ptr = level_ptr.data();
    v.n_cycles  = n_cycles.data();
    v.all_found = all_found.data();
    v.all_mask  = all_mask.data();
    return v;
  }
  rmm::device_uvector<key_t<max_routes>> key_ptr;
  rmm::device_uvector<double> cost_ptr;
  rmm::device_uvector<i_t> level_ptr;
  rmm::device_scalar<i_t> n_cycles;
  rmm::device_scalar<bool> all_found;
  rmm::device_scalar<detail::device_bitset_t<max_routes>> all_mask;
};

template <typename i_t, typename f_t, size_t max_routes>
struct cycle_candidates_t {
  cycle_candidates_t(size_t size_, int n_paths_, rmm::cuda_stream_view stream)
    : keys(size_ * n_paths_, stream),
      costs(size_ * n_paths_, stream),
      level_vec(size_ * n_paths_, stream),
      size(size_),
      n_paths(n_paths_)
  {
  }

  void resize(size_t size_, solution_handle_t<i_t, f_t> const* handle_ptr)
  {
    keys.resize(size_, handle_ptr->get_stream());
    costs.resize(size_, handle_ptr->get_stream());
    level_vec.resize(size_, handle_ptr->get_stream());
  }

  void reset(size_t n_vertices, solution_handle_t<i_t, f_t> const* handle_ptr)
  {
    size = n_vertices;
    if (keys.size() < size * n_paths) { resize(size * n_paths, handle_ptr); }
    async_fill(costs, std::numeric_limits<double>::max(), handle_ptr->get_stream());
  }

  struct view_t {
    raft::device_span<key_t<max_routes>> keys;
    raft::device_span<double> costs;
    raft::device_span<int> level_vec;
  };

  view_t view()
  {
    view_t v;
    v.keys      = raft::device_span<key_t<max_routes>>{keys.data(), keys.size()};
    v.costs     = raft::device_span<double>(costs.data(), costs.size());
    v.level_vec = raft::device_span<int>(level_vec.data(), level_vec.size());
    return v;
  }

  view_t level_view(i_t level)
  {
    view_t v;
    v.keys      = raft::device_span<key_t<max_routes>>{keys.data() + level * size, size};
    v.costs     = raft::device_span<double>(costs.data() + level * size, size);
    v.level_vec = raft::device_span<int>(level_vec.data() + level * size, size);
    return v;
  }

  rmm::device_uvector<key_t<max_routes>> keys;
  rmm::device_uvector<double> costs;
  rmm::device_uvector<int> level_vec;
  size_t size;
  int n_paths;
};

/*! \brief { A set disjoint cycle finder based on : Operations Research Letters 31 (2003) 185 – 194
             A composite very large-scale neighborhoodstructure for the capacitated minimum spanning
   tree problem Ravindra K. Ahuja, James B. Orlin, Dushyant Sharma. The maximal number of routes is
   a static parameter denoting the possible set ids. The parameter is default set to 256: for this
             value hashset speed is not significantly compromised compared to int_64. For max_routes
   = 4096 the hashet speed detoriates 6x (because of hash calculation time). There are two limiting
   factors (speeding up the algorithm but making it heuristic): 1) max_level: the maximum length of
   the cycle 2) max_paths: maximum number of negative weight paths stored for each path length.
   (default ~100k) The total amount of memory used: O(max_paths*max_level*max_routes/register_size)}
             */
template <typename i_t, typename f_t, size_t max_routes = 256>
struct ExactCycleFinder {
 public:
  /*! \param[in] G_ Input list graph containing edges and weights}
      \param[in] ids_ Set ids of graph nodes. Every node has assigned set label
      \param[in] max_level_ Set ids of graph nodes. Every node has assigned set label}*/
  ExactCycleFinder(solution_handle_t<i_t, f_t> const* handle_ptr_,
                   const bool depot_included_,
                   i_t max_level_    = 10,
                   size_t max_paths_ = 50000)
    : max_level(max_level_),
      max_paths(max_paths_),
      d_valid_paths(handle_ptr_, max_level_, max_paths_),
      best_cycles(handle_ptr_),
      sorted_key_indices(0, handle_ptr_->get_stream()),
      copy_indices(0, handle_ptr_->get_stream()),
      copy_cost(0, handle_ptr_->get_stream()),
      d_cub_storage_bytes(0, handle_ptr_->get_stream()),
      max_threads(handle_ptr_->get_device_properties().maxThreadsPerBlock),
      max_blocks(handle_ptr_->get_device_properties().maxGridSize[0]),
      depot_included(depot_included_),
      cycle_candidates(0, max_level_, handle_ptr_->get_stream()),
      handle_ptr(handle_ptr_)
  {
  }

  void get_cycle(graph_t<i_t, f_t>& graph, ret_cycles_t<i_t, f_t>& ret);

  bool call_init(graph_t<i_t, f_t>& graph);

  bool call_find(graph_t<i_t, f_t>& graph, i_t level);

  bool find_cycle(graph_t<i_t, f_t>& graph);

  void find_best_cycles(graph_t<i_t, f_t>& graph,
                        ret_cycles_t<i_t, f_t>& ret,
                        solution_handle_t<i_t, f_t> const* sol_handle);

  bool check_cycle(graph_t<i_t, f_t>& graph, ret_cycles_t<i_t, f_t>& ret);

  void sort_occupied(int level, graph_t<i_t, f_t>& graph, int current_level_occupied);
  void sort_cycle_costs_by_key(int n_items);
  bool check_occupied_head(int level, graph_t<i_t, f_t>& graph);

 private:
  // use handle for thrust policy
  const solution_handle_t<i_t, f_t>* handle_ptr;
  path_t<i_t, f_t, max_routes> best_cycles;
  device_map_t<key_t<max_routes>, double> d_valid_paths;
  rmm::device_uvector<int> sorted_key_indices;
  rmm::device_uvector<std::byte> d_cub_storage_bytes;
  rmm::device_uvector<double> copy_cost;
  rmm::device_uvector<int> copy_indices;
  i_t max_level{};
  i_t n_occupied_heads;
  size_t max_paths{};
  size_t max_threads{};
  size_t max_blocks{};
  bool depot_included{};
  cycle_candidates_t<i_t, f_t, max_routes> cycle_candidates;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
