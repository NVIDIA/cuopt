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

#include <utilities/cuda_helpers.cuh>
#include "../../node/node.cuh"
#include "../../routing_helpers.cuh"
#include "../../solution/solution.cuh"

#include <routing/utilities/cuopt_utils.cuh>
#include <utilities/seed_generator.cuh>

#include "raft/core/span.hpp"

namespace cuopt {
namespace routing {
namespace detail {
template <typename dst_t, typename src_t>
DI static void copy_forward_data(dst_t& dst, const src_t& src)
{
  constexpr bool is_src_a_node = std::is_same<src_t, node_t<int, float, request_t::PDP>>::value ||
                                 std::is_same<src_t, node_t<int, float, request_t::VRP>>::value;
  constexpr bool is_dst_a_node = std::is_same<dst_t, node_t<int, float, request_t::PDP>>::value ||
                                 std::is_same<dst_t, node_t<int, float, request_t::VRP>>::value;

  if constexpr (is_src_a_node && !is_dst_a_node) {
    dst.distance_forward         = src.distance_dim.distance_forward;
    dst.transit_time_forward     = src.time_dim.transit_time_forward;
    dst.latest_arrival_forward   = src.time_dim.latest_arrival_forward;
    dst.unavoidable_wait_forward = src.time_dim.unavoidable_wait_forward;
    dst.departure_forward        = src.time_dim.departure_forward;
    dst.excess_forward           = src.time_dim.excess_forward;
    constexpr_for<src_t::max_capacity_dim>([&](auto i) {
      if (i < src.capacity_dim.n_capacity_dimensions) {
        dst.gathered[i]    = src.capacity_dim.gathered[i];
        dst.max_to_node[i] = src.capacity_dim.max_to_node[i];
      }
    });
  } else if constexpr (is_dst_a_node && !is_src_a_node) {
    dst.distance_dim.distance_forward     = src.distance_forward;
    dst.time_dim.transit_time_forward     = src.transit_time_forward;
    dst.time_dim.latest_arrival_forward   = src.latest_arrival_forward;
    dst.time_dim.unavoidable_wait_forward = src.unavoidable_wait_forward;
    dst.time_dim.departure_forward        = src.departure_forward;
    dst.time_dim.excess_forward           = src.excess_forward;
    constexpr_for<dst_t::max_capacity_dim>([&](auto i) {
      if (i < dst.capacity_dim.n_capacity_dimensions) {
        dst.capacity_dim.gathered[i]    = src.gathered[i];
        dst.capacity_dim.max_to_node[i] = src.max_to_node[i];
      }
    });
  } else if constexpr (!is_src_a_node && !is_dst_a_node) {
    dst.distance_forward         = src.distance_forward;
    dst.transit_time_forward     = src.transit_time_forward;
    dst.latest_arrival_forward   = src.latest_arrival_forward;
    dst.unavoidable_wait_forward = src.unavoidable_wait_forward;
    dst.departure_forward        = src.departure_forward;
    dst.excess_forward           = src.excess_forward;
    for (auto i = 0; i < dst.gathered.size(); ++i) {
      dst.gathered[i]    = src.gathered[i];
      dst.max_to_node[i] = src.max_to_node[i];
    }
  } else {
    dst.distance_dim.distance_forward     = src.distance_dim.distance_forward;
    dst.time_dim.transit_time_forward     = src.time_dim.transit_time_forward;
    dst.time_dim.latest_arrival_forward   = src.time_dim.latest_arrival_forward;
    dst.time_dim.unavoidable_wait_forward = src.time_dim.unavoidable_wait_forward;
    dst.time_dim.departure_forward        = src.time_dim.departure_forward;
    dst.time_dim.excess_forward           = src.time_dim.excess_forward;
    constexpr_for<src_t::max_capacity_dim>([&](auto i) {
      if (i < dst.capacity_dim.n_capacity_dimensions) {
        dst.capacity_dim.gathered[i]    = src.capacity_dim.gathered[i];
        dst.capacity_dim.max_to_node[i] = src.capacity_dim.max_to_node[i];
      }
    });
  }
}

template <typename i_t, request_t REQUEST>
constexpr i_t max_neighbors(i_t k_max)
{
  if constexpr (REQUEST == request_t::PDP) {
    return 2 * k_max + 1;
  } else {
    return k_max;
  }
  return {};
}

template <typename i_t, typename f_t, request_t REQUEST>
struct node_stack_t {
  DI node_stack_t(i_t* sh_ptr,
                  i_t k_max_,
                  i_t delivery_insertion_idx_in_permutation_,
                  i_t p_scores_of_pickup_,
                  node_t<i_t, f_t, REQUEST> delivery_node_,
                  i_t route_length_,
                  typename route_t<i_t, f_t, REQUEST>::view_t& route_)
    : k_max(k_max_),
      delivery_insertion_idx_in_permutation(delivery_insertion_idx_in_permutation_),
      min_p_score(p_scores_of_pickup_ - 1),
      best_sequence_size(std::numeric_limits<i_t>::max()),
      delivery_node(delivery_node_),
      dummy_node(delivery_node_.dimensions_info),
      route_length(route_length_),
      thread_rng(2727, uint64_t((threadIdx.x + blockIdx.x * blockDim.x)), 0)
  {
    sh_ptr  = set_spans(sh_ptr);
    s_route = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(sh_ptr, route_, route_length);
    __syncthreads();
    s_route.copy_from(route_);
    __syncthreads();
  }

  // this will be in shared memory for each thread
  struct __align__(32ul) item_t {
    double distance_forward;
    double transit_time_forward;
    double latest_arrival_forward;
    double unavoidable_wait_forward;
    f_t departure_forward;
    f_t excess_forward;
    i_t intra_idx;
    i_t from_idx;
    // TODO later we might use multiple node inheritence, but for now this will be in shared memory
    raft::device_span<i_t> gathered;
    raft::device_span<i_t> max_to_node;
    NodeInfo<i_t> node_info;
    // instead of dummy var, the align keyword forces struct to have it

    DI i_t id() const { return node_info.node(); }

    DI item_t& operator=(const node_t<i_t, f_t, REQUEST>& src)
    {
      node_info = src.node_info();
      copy_forward_data(*this, src);
      return *this;
    }

    // We are using device_spans for gathered and max_to_node. The default assignment operator
    // will override the pointers. We however, need a deep copy here so implementing explicitly here
    DI item_t& operator=(const item_t& src)
    {
      node_info = src.node_info;
      intra_idx = src.intra_idx;
      from_idx  = src.from_idx;
      copy_forward_data(*this, src);
      return *this;
    }
  };

  i_t stack_top;
  const i_t k_max;
  i_t current_p_score;
  i_t n_ejected_pickups;
  const i_t delivery_insertion_idx_in_permutation;

  // per thread min p_score until now
  i_t min_p_score;

  // the size of the best sequence
  i_t best_sequence_size;
  node_t<i_t, f_t, REQUEST> delivery_node;
  node_t<i_t, f_t, REQUEST> dummy_node;

  raft::random::PCGenerator thread_rng;
  i_t random_counter = 1;
  const i_t route_length;
  // common items per block are static
  raft::device_span<i_t> p_scores;
  typename route_t<i_t, f_t, REQUEST>::view_t s_route;

  // the sequence of insertion/ejections
  raft::device_span<i_t> best_sequence;
  raft::device_span<item_t> stack;
  raft::device_span<i_t> gathered;
  raft::device_span<i_t> max_to_node;

  raft::device_span<f_t> dim_buffer_route[size_t(dim_t::SIZE)];
  raft::device_span<f_t> dim_delivery_to_all[size_t(dim_t::SIZE)];

  static size_t get_shared_size(solution_t<i_t, f_t, REQUEST>* solution_ptr,
                                int added_size,
                                int k_max,
                                int threads_per_block_lexico)
  {
    const auto n_capacity_dimensions =
      solution_ptr->problem_ptr->dimensions_info.capacity_dim.n_capacity_dimensions;
    const size_t size_of_stack =
      sizeof(item_t) * max_neighbors<i_t, REQUEST>(k_max) * threads_per_block_lexico;
    const size_t size_of_stack_buffers_for_capacity = 2 * max_neighbors<i_t, REQUEST>(k_max) *
                                                      threads_per_block_lexico * sizeof(i_t) *
                                                      n_capacity_dimensions;
    const size_t size_of_best_sequence =
      max_neighbors<i_t, REQUEST>(k_max) * threads_per_block_lexico * sizeof(i_t);
    const size_t size_of_p_score =
      solution_ptr->get_max_active_nodes_for_all_routes() * sizeof(i_t);

    size_t sh_size =
      size_of_stack + size_of_stack_buffers_for_capacity + size_of_best_sequence + size_of_p_score;

    if (solution_ptr->problem_ptr->dimensions_info.time_dim.has_constraints()) {
      const size_t shared_for_time_buffers =
        (2 + (max_neighbors<i_t, REQUEST>(k_max) + 1)) *
        (solution_ptr->get_max_active_nodes_for_all_routes() + 1) * sizeof(f_t);
      sh_size += shared_for_time_buffers;
    }
    if (solution_ptr->problem_ptr->dimensions_info.distance_dim.has_constraints()) {
      const size_t shared_for_dist_buffers =
        (2 + (max_neighbors<i_t, REQUEST>(k_max) + 1)) *
        (solution_ptr->get_max_active_nodes_for_all_routes() + 1) * sizeof(f_t);
      sh_size += shared_for_dist_buffers;
    }

    sh_size = raft::alignTo(sh_size, sizeof(infeasible_cost_t));
    sh_size += solution_ptr->get_temp_route_shared_size(added_size);

    return sh_size;
  }

  DI i_t* set_spans(i_t* sh_ptr)
  {
    auto* shmem                = sh_ptr;
    auto n_capacity_dimensions = delivery_node.capacity_dim.n_capacity_dimensions;
    thrust::tie(stack, sh_ptr) = wrap_ptr_as_span<typename node_stack_t<i_t, f_t, REQUEST>::item_t>(
      sh_ptr, max_neighbors<i_t, REQUEST>(k_max) * blockDim.x);
    thrust::tie(best_sequence, sh_ptr) =
      wrap_ptr_as_span<i_t>(sh_ptr, max_neighbors<i_t, REQUEST>(k_max) * blockDim.x);

    loop_over_constrained_dimensions(dim_info(), [&](auto I) {
      if constexpr ((I == size_t(dim_t::DIST)) || (I == size_t(dim_t::TIME))) {
        thrust::tie(dim_delivery_to_all[I], sh_ptr) =
          wrap_ptr_as_span<f_t>(sh_ptr, 2 * (route_length + 1));
        thrust::tie(dim_buffer_route[I], sh_ptr) =
          wrap_ptr_as_span<f_t>(sh_ptr, route_length * (max_neighbors<i_t, REQUEST>(k_max) + 1));
      }
    });

    thrust::tie(p_scores, sh_ptr) = wrap_ptr_as_span<i_t>(sh_ptr, route_length);
    thrust::tie(gathered, sh_ptr) =
      wrap_ptr_as_span<i_t>(sh_ptr, stack.size() * n_capacity_dimensions);
    thrust::tie(max_to_node, sh_ptr) =
      wrap_ptr_as_span<i_t>(sh_ptr, stack.size() * n_capacity_dimensions);

    set_item_stack_ptrs();

    auto total_bytes =
      (stack.size() * sizeof(node_stack_t<i_t, f_t, REQUEST>::item_t)) +
      (best_sequence.size() + p_scores.size() + gathered.size() + max_to_node.size()) * sizeof(i_t);

    loop_over_constrained_dimensions(dim_info(), [&](auto I) {
      if constexpr ((I == size_t(dim_t::DIST)) || (I == size_t(dim_t::TIME))) {
        total_bytes += (dim_delivery_to_all[I].size() + dim_buffer_route[I].size()) * sizeof(f_t);
      }
    });
    auto aligned_bytes = raft::alignTo(total_bytes, sizeof(infeasible_cost_t));
    return (i_t*)(((uint8_t*)shmem) + aligned_bytes);
  }

  DI void set_item_stack_ptrs()
  {
    auto n_capacity_dimensions = gathered.size() / stack.size();
    for (i_t i = threadIdx.x; i < stack.size(); i += blockDim.x) {
      size_t offset     = i * n_capacity_dimensions;
      stack[i].gathered = raft::device_span<i_t>{gathered.data() + offset, n_capacity_dimensions};
      stack[i].max_to_node =
        raft::device_span<i_t>{max_to_node.data() + offset, n_capacity_dimensions};
    }
    __syncthreads();
  }

  DI item_t& get_stack_item(i_t idx) { return stack[threadIdx.x + blockDim.x * idx]; }

  DI i_t& get_best_sequnce(i_t idx) { return best_sequence[threadIdx.x + blockDim.x * idx]; }

  // as we are accessing top more often, keep top in a register
  DI item_t& top() { return get_stack_item(stack_top - 1); }

  // resets current values and keeps the best
  DI void reset()
  {
    current_p_score   = 0;
    n_ejected_pickups = 0;
    stack_top         = 0;  // empty stack
  }

  template <size_t d>
  DI void compute_dim_buffers()
  {
    if constexpr ((d == size_t(dim_t::DIST)) || (d == size_t(dim_t::TIME))) {
      constexpr dim_t dim = dim_t(d);
      i_t n_nodes_route   = s_route.get_num_nodes();
      auto& node_infos    = s_route.requests().node_info;
      if constexpr (REQUEST == request_t::PDP) {
        // compute delivery to all
        for (i_t i = threadIdx.x; i < n_nodes_route + 1; i += blockDim.x) {
          dim_delivery_to_all[d][i] = get_arc_of_dimension<i_t, f_t, dim>(
            delivery_node.node_info(), node_infos[i], s_route.vehicle_info());
        }

        // compute all to delivery
        for (i_t i = threadIdx.x; i < n_nodes_route + 1; i += blockDim.x) {
          dim_delivery_to_all[d][i + (n_nodes_route + 1)] = get_arc_of_dimension<i_t, f_t, dim>(
            node_infos[i], delivery_node.node_info(), s_route.vehicle_info());
        }
      }

      for (i_t k = 0; k <= max_neighbors<i_t, REQUEST>(k_max); ++k) {
        for (i_t i = threadIdx.x; i + k < n_nodes_route; i += blockDim.x) {
          dim_buffer_route[d][i + k * n_nodes_route] = get_arc_of_dimension<i_t, f_t, dim>(
            node_infos[i], node_infos[i + k + 1], s_route.vehicle_info());
        }
      }
    }
  }

  DI void insert_node_and_update_data(const i_t* __restrict__ p_scores_,
                                      i_t pickup_insert_idx,
                                      const node_t<i_t, f_t, REQUEST>& pickup_node,
                                      typename route_node_map_t<i_t>::view_t& route_node_map)
  {
    // insert the pickup at the respective position to compute the forward backward data
    if (threadIdx.x == 0) {
      s_route.insert_node(pickup_insert_idx, pickup_node, route_node_map, false);
      route_t<i_t, f_t, REQUEST>::view_t::compute_forward(s_route);
      // TODO compute backward data up to the pickup node
      route_t<i_t, f_t, REQUEST>::view_t::compute_backward(s_route);
    }
    __syncthreads();
    loop_over_constrained_dimensions(dim_info(), [&](auto I) { compute_dim_buffers<I>(); });
    copy_p_scores(p_scores_);
    __syncthreads();
  }

  DI void copy_p_scores(const i_t* p_scores_)
  {
    for (i_t i = threadIdx.x + 1; i < route_length; i += blockDim.x) {
      p_scores[i] = p_scores_[s_route.node_id(i)];
    }
  }

  template <dim_t dim>
  DI f_t get_dim_from_delivery(i_t intra_idx) const
  {
    if (!dim_info().has_dimension(dim)) { return f_t{}; }

    /*
     * Note that we are only storing dim_delivery_to_all for only the dimensions that have matrices
     * defined. For other dimensions, forward calculations depend only on the node values some
     * values like service time and vehicle order mismatch depend on the (vehicle, order) service
     * time dimension currently does not have any constraint and vehicle order match is handled
     * upfront in the lexicographic search kernel so we can safely skip here. Ideal way to handle
     * this smoothly is to pass vehicle_info to calculate_forward and calculate_backward methods
     */
    if constexpr ((dim == dim_t::DIST) || (dim == dim_t::TIME)) {
      return dim_delivery_to_all[size_t(dim)][intra_idx];
    } else {
      return f_t{};
    }
  }

  template <size_t dim>
  DI f_t get_dim_from_delivery(i_t intra_idx) const
  {
    return get_dim_from_delivery<dim_t(dim)>(intra_idx);
  }

  template <dim_t dim>
  DI f_t get_dim_to_delivery(i_t intra_idx) const
  {
    if (!dim_info().has_dimension(dim)) { return f_t{}; }
    if constexpr ((dim == dim_t::DIST) || (dim == dim_t::TIME)) {
      return dim_delivery_to_all[size_t(dim)][intra_idx + route_length + 1];
    } else {
      return f_t{};
    }
  }

  template <size_t dim>
  DI f_t get_dim_to_delivery(i_t intra_idx) const
  {
    return get_dim_to_delivery<dim_t(dim)>(intra_idx);
  }

  template <dim_t dim>
  DI f_t get_dim_between(i_t intra_idx_1, i_t intra_idx_2) const
  {
    if (!dim_info().has_dimension(dim)) { return f_t{}; }
    if constexpr ((dim == dim_t::DIST) || (dim == dim_t::TIME)) {
      i_t gap_between = intra_idx_2 - intra_idx_1 - 1;
      return dim_buffer_route[size_t(dim)][intra_idx_1 + gap_between * route_length];
    } else {
      return f_t{};
    }
  }

  template <size_t dim>
  DI f_t get_dim_between(i_t intra_idx_1, i_t intra_idx_2) const
  {
    return get_dim_between<dim_t(dim)>(intra_idx_1, intra_idx_2);
  }

  DI const enabled_dimensions_t& dim_info() const { return delivery_node.dimensions_info; }

  DI void calculate_forward_between(const i_t from_idx,
                                    const i_t to_idx,
                                    node_t<i_t, f_t, REQUEST>& node)
  {
    auto d_node = s_route.get_node(top().intra_idx);
    copy_forward_data(d_node, top());
    loop_over_dimensions(dim_info(), [&] __device__(auto I) {
      if (get_dimension_of<I>(dim_info()).has_constraints()) {
        auto dim_between = get_dim_between<I>(from_idx, to_idx);
        get_dimension_of<I>(d_node).calculate_forward(get_dimension_of<I>(node), dim_between);
      }
    });
  }

  DI bool check_dim_between(const i_t from_idx,
                            const i_t to_idx,
                            const NodeInfo<i_t>& compare_node_info_1,
                            const NodeInfo<i_t>& compare_node_info_2) const
  {
    bool valid = true;
    loop_over_dimensions(dim_info(), [&] __device__(auto I) {
      if (get_dimension_of<I>(dim_info()).has_constraints()) {
        auto dim_between = get_dim_between<I>(from_idx, to_idx);
        valid &= check_dim_between<I>(dim_between, compare_node_info_1, compare_node_info_2);
      }
    });
    return valid;
  }

  DI bool check_dim_between(const i_t from_idx,
                            const i_t to_idx,
                            const node_t<i_t, f_t, REQUEST>& node,
                            const NodeInfo<i_t>& compare_node_info) const
  {
    return check_dim_between(from_idx, to_idx, node.node_info(), compare_node_info);
  }

  DI bool check_dim_between(const i_t from_idx,
                            const i_t to_idx,
                            const node_t<i_t, f_t, REQUEST>& compare_node_1,
                            const node_t<i_t, f_t, REQUEST>& compare_node_2) const
  {
    return check_dim_between(
      from_idx, to_idx, compare_node_1.node_info(), compare_node_2.node_info());
  }

  DI void calculate_forward_to_delivery(const i_t idx, node_t<i_t, f_t, REQUEST>& node)
  {
    loop_over_dimensions(dim_info(), [&] __device__(auto I) {
      if (get_dimension_of<I>(dim_info()).has_constraints()) {
        auto dim_to_delivery = get_dim_to_delivery<I>(idx);
        get_dimension_of<I>(node).calculate_forward(get_dimension_of<I>(delivery_node),
                                                    dim_to_delivery);
      }
    });
  }

  DI bool check_dim_to_delivery(const i_t idx, const node_t<i_t, f_t, REQUEST>& node) const
  {
    bool valid = true;
    loop_over_dimensions(dim_info(), [&] __device__(auto I) {
      if (get_dimension_of<I>(dim_info()).has_constraints()) {
        auto dim_to_delivery = get_dim_to_delivery<I>(idx);
        valid &= check_dim_to_delivery<I>(dim_to_delivery, node.node_info());
      }
    });
    return valid;
  }

  DI void calculate_forward_from_delivery(const i_t idx, node_t<i_t, f_t, REQUEST>& node)
  {
    loop_over_dimensions(dim_info(), [&] __device__(auto I) {
      if (get_dimension_of<I>(dim_info()).has_constraints()) {
        auto dim_from_delivery = get_dim_from_delivery<I>(idx);
        get_dimension_of<I>(delivery_node)
          .calculate_forward(get_dimension_of<I>(node), dim_from_delivery);
      }
    });
  }

  DI bool check_dim_from_delivery(const i_t idx, const node_t<i_t, f_t, REQUEST>& node) const
  {
    bool valid = true;
    loop_over_dimensions(dim_info(), [&] __device__(auto I) {
      if (get_dimension_of<I>(dim_info()).has_constraints()) {
        auto dim_from_delivery = get_dim_from_delivery<I>(idx);
        valid &= check_dim_from_delivery<I>(dim_from_delivery, node.node_info());
      }
    });
    return valid;
  }

  template <request_t r_t, std::enable_if_t<r_t == request_t::VRP, bool> = true>
  DI void update_best_sequence()
  {
    cuopt_assert(stack_top <= k_max,
                 "At best move record stack_top must be smaller or equal to k_max");
    // Record move locally (thread level):
    if (current_p_score <= min_p_score) {
      // FIXME:: don't include sequences with break nodes for now
      for (i_t i = 0; i < stack_top; ++i) {
        if (get_stack_item(i).node_info.is_break()) { return; }
      }

      bool record = false;
      if (current_p_score < min_p_score || stack_top < best_sequence_size) {
        min_p_score    = current_p_score;
        random_counter = 1;
        record         = true;
      }
      // case where p scores and sequence size are equal
      else if (stack_top == best_sequence_size) {
        if (thread_rng.next_u32() % random_counter == 0) { record = true; }
        ++random_counter;
      }
      if (record) {
        best_sequence_size = stack_top;
        for (i_t i = 0; i < best_sequence_size; ++i) {
          get_best_sequnce(i) = get_stack_item(i).intra_idx;
        }
      }
    }
  }

  template <request_t r_t, std::enable_if_t<r_t == request_t::PDP, bool> = true>
  DI void update_best_sequence()
  {
    // if we a whole pd request is ejected and also the delivery insertion is already inserted
    if (stack_top % 2 == 1 && stack_top > delivery_insertion_idx_in_permutation) {
      cuopt_assert(stack_top <= 2 * k_max + 1,
                   "At best move record stack_top must be smaller or equal to 2 * k_max + 1 ");

      // if half of the stack is not the pickups then return
      if (((stack_top - 1) / 2) != n_ejected_pickups) { return; }
      // Record move locally (thread level):
      // if the current score is smaller and we have pairs of pickup delivery ejected
      if (current_p_score <= min_p_score) {
        // FIXME:: don't include sequences with break nodes for now
        for (i_t i = 0; i < stack_top; ++i) {
          if (get_stack_item(i).node_info.is_break()) { return; }
        }
        bool record = false;
        if (current_p_score < min_p_score || stack_top < best_sequence_size) {
          min_p_score    = current_p_score;
          random_counter = 1;
          record         = true;
        }
        // case where p scores and sequence size are equal
        else if (stack_top == best_sequence_size) {
          if (thread_rng.next_u32() % random_counter == 0) { record = true; }
          ++random_counter;
        }
        if (record) {
          best_sequence_size = stack_top;
          for (i_t i = 0; i < best_sequence_size; ++i) {
            get_best_sequnce(i) = get_stack_item(i).intra_idx;
          }
        }
      }
    }
  }

  DI bool check_paired_pickup_is_ejected(i_t brother_id)
  {
    cuopt_assert(stack_top >= 1, "stack_top should be greater than 0");
    // since we don't have efficient hashmap, do a linear search
    for (i_t i = 0; i < stack_top; ++i) {
      if (delivery_insertion_idx_in_permutation == i) continue;
      if (brother_id == get_stack_item(i).id()) return true;
    }
    return false;
  }

  template <request_t r_t, std::enable_if_t<r_t == request_t::VRP, bool> = true>
  DI bool is_stack_top_insertion()
  {
    return false;
  }

  template <request_t r_t, std::enable_if_t<r_t == request_t::PDP, bool> = true>
  DI bool is_stack_top_insertion()
  {
    return stack_top - 1 == delivery_insertion_idx_in_permutation;
  }

  template <request_t r_t, std::enable_if_t<r_t == request_t::VRP, bool> = true>
  DI bool advance_ejection(node_t<i_t, f_t, REQUEST>& temp_node)
  {
    i_t from_idx = top().from_idx;
    i_t to_idx   = top().intra_idx;
    calculate_forward_between(from_idx, to_idx, temp_node);
    cuopt_assert(
      check_dim_between(
        from_idx, to_idx, s_route.requests().node_info[top().from_idx], temp_node.node_info()),
      "Mismatch");
    return temp_node.forward_feasible(s_route.vehicle_info());
  }

  template <request_t r_t, std::enable_if_t<r_t == request_t::PDP, bool> = true>
  DI bool advance_ejection(node_t<i_t, f_t, REQUEST>& temp_node)
  {
    // if immediate previous item is delivery, the from node is the delivery node
    if (top().from_idx == route_length) {
      i_t idx = top().intra_idx;
      calculate_forward_from_delivery(idx, temp_node);
      cuopt_assert(check_dim_from_delivery(idx, temp_node), "Mismatch");
    } else {
      i_t from_idx = top().from_idx;
      i_t to_idx   = top().intra_idx;
      calculate_forward_between(from_idx, to_idx, temp_node);
      cuopt_assert(
        check_dim_between(
          from_idx, to_idx, s_route.requests().node_info[top().from_idx], temp_node.node_info()),
        "Mismatch");
    }
    return temp_node.forward_feasible(s_route.vehicle_info());
  }

  // temp_node is the next node after which we will insert the delivery
  DI bool advance_insertion(node_t<i_t, f_t, REQUEST>& temp_node)
  {
    // the index before the delivery node
    i_t prev_idx = top().intra_idx;
    // compute the forward data of the temp node
    calculate_forward_between(prev_idx, prev_idx + 1, temp_node);
    cuopt_assert(check_dim_between(prev_idx, prev_idx + 1, top().node_info, temp_node.node_info()),
                 "Mismatch");

    // compute the delivery forward data
    calculate_forward_to_delivery(prev_idx + 1, temp_node);
    cuopt_assert(check_dim_to_delivery(prev_idx + 1, temp_node), "Mismatch");

    return temp_node.forward_feasible(s_route.vehicle_info());
  }

  DI bool expand_insertion(node_t<i_t, f_t, REQUEST>& temp_node)
  {
    // compute the forward data of the node before the delivery
    i_t from_idx = top().from_idx;
    i_t to_idx   = top().intra_idx + 1;
    calculate_forward_between(from_idx, to_idx, temp_node);
    cuopt_assert(check_dim_between(
                   from_idx, to_idx, s_route.requests().node_info[from_idx], temp_node.node_info()),
                 "Mismatch");

    calculate_forward_to_delivery(to_idx, temp_node);
    cuopt_assert(check_dim_to_delivery(to_idx, temp_node), "Mismatch");
    return temp_node.forward_feasible(s_route.vehicle_info());
  }

  // RUNTIME CHECK FUNCTIONS
  DI bool node_id_coherence_check()
  {
    for (i_t i = 0; i < stack_top; ++i) {
      cuopt_assert(s_route.node_id(get_stack_item(i).intra_idx) == get_stack_item(i).id(),
                   "Node ids on stack and the route don't match!");
    }
    return true;
  }

  DI bool p_score_check()
  {
    i_t sum_p_score = 0;
    for (i_t i = 0; i < stack_top; ++i) {
      if (REQUEST == request_t::PDP && i == delivery_insertion_idx_in_permutation) continue;
      sum_p_score += p_scores[get_stack_item(i).intra_idx];
    }
    cuopt_assert(sum_p_score == current_p_score, "P score sums don't match!");
    return true;
  }

  template <request_t r_t, std::enable_if_t<r_t == request_t::VRP, bool> = true>
  DI bool forward_check(bool advance = false)
  {
    node_t<i_t, f_t, REQUEST> iter_node         = s_route.get_node(0);
    node_t<i_t, f_t, REQUEST> beginning_of_hole = s_route.get_node(0);
    i_t stack_item                              = 0;
    i_t size_of_hole                            = 0;
    for (i_t i = 0; i < top().intra_idx; ++i) {
      if (get_stack_item(stack_item).intra_idx == i) {
        size_of_hole++;
        auto next_node = s_route.get_node(i + 1);
        cuopt_assert(i - size_of_hole >= 0, "");
        cuopt_assert(check_dim_between(i - size_of_hole, i + 1, beginning_of_hole, next_node),
                     "dim buffer mismatch");
        loop_over_dimensions(beginning_of_hole.dimensions_info, [&] __device__(auto I) {
          if (get_dimension_of<I>(beginning_of_hole.dimensions_info).has_constraints()) {
            auto dim_between = get_dim_between<I>(i - size_of_hole, i + 1);
            get_dimension_of<I>(beginning_of_hole)
              .calculate_forward(get_dimension_of<I>(next_node), dim_between);
          }
        });

        cuopt_assert(abs(beginning_of_hole.time_dim.departure_forward -
                         get_stack_item(stack_item).departure_forward) < 0.01,
                     "Departure forward mismatch");
        cuopt_assert(abs(beginning_of_hole.time_dim.excess_forward -
                         get_stack_item(stack_item).excess_forward) < 0.01,
                     "excess_forward mismatch");
        constexpr_for<node_t<i_t, f_t, REQUEST>::max_capacity_dim>([&](auto d) {
          if (d < beginning_of_hole.capacity_dim.n_capacity_dimensions) {
            cuopt_assert(
              beginning_of_hole.capacity_dim.gathered[d] == get_stack_item(stack_item).gathered[d],
              "gathered mismatch");
            cuopt_assert(beginning_of_hole.capacity_dim.max_to_node[d] ==
                           get_stack_item(stack_item).max_to_node[d],
                         "max_to_node mismatch");
          }
        });
        iter_node = next_node;
        stack_item++;
        assert(stack_item < stack_top);
      } else {
        auto next_node = s_route.get_node(i + 1);
        cuopt_assert(check_dim_between(i, i + 1, iter_node, next_node), "dim buffer mismatch");
        loop_over_dimensions(iter_node.dimensions_info, [&] __device__(auto I) {
          if (get_dimension_of<I>(iter_node.dimensions_info).has_constraints()) {
            auto dim_between = get_dim_between<I>(i, i + 1);
            get_dimension_of<I>(iter_node).calculate_forward(get_dimension_of<I>(next_node),
                                                             dim_between);
          }
        });
        if (!advance) {
          cuopt_assert(iter_node.forward_feasible(s_route.vehicle_info()),
                       "Iter node is not forward feasible!");
        }
        beginning_of_hole = iter_node;
        iter_node         = next_node;
        size_of_hole      = 0;
      }
    }
    return true;
  }

  template <request_t r_t, std::enable_if_t<r_t == request_t::PDP, bool> = true>
  DI bool forward_check(bool advance = false)
  {
    node_t<i_t, f_t, REQUEST> iter_node         = s_route.get_node(0);
    node_t<i_t, f_t, REQUEST> delivery_copy     = delivery_node;
    node_t<i_t, f_t, REQUEST> beginning_of_hole = s_route.get_node(0);
    i_t stack_item                              = 0;
    i_t size_of_hole                            = 0;
    for (i_t i = 0; i < top().intra_idx; ++i) {
      if (get_stack_item(stack_item).intra_idx == i) {
        if (stack_item == delivery_insertion_idx_in_permutation) {
          cuopt_assert(check_dim_to_delivery(i, iter_node), "dim buffer mismatch");
          calculate_forward_to_delivery(i, iter_node);

          cuopt_assert(abs(iter_node.time_dim.departure_forward -
                           get_stack_item(stack_item).departure_forward) < 0.01,
                       "Departure forward mismatch");
          cuopt_assert(abs(iter_node.time_dim.excess_forward -
                           get_stack_item(stack_item).excess_forward) < 0.01,
                       "excess_forward mismatch");
          constexpr_for<node_t<i_t, f_t, REQUEST>::max_capacity_dim>([&](auto d) {
            if (d < iter_node.capacity_dim.n_capacity_dimensions) {
              cuopt_assert(
                iter_node.capacity_dim.gathered[d] == get_stack_item(stack_item).gathered[d],
                "gathered mismatch");
              cuopt_assert(
                iter_node.capacity_dim.max_to_node[d] == get_stack_item(stack_item).max_to_node[d],
                "max_to_node mismatch");
            }
          });
          if (!iter_node.forward_feasible(s_route.vehicle_info())) {
            printf("excess forward %f departure forward %f \n",
                   iter_node.time_dim.excess_forward,
                   iter_node.time_dim.departure_forward);
          }
          cuopt_assert(iter_node.forward_feasible(s_route.vehicle_info()),
                       "Iter node is not forward feasible!");
          cuopt_assert(delivery_copy.time_dim.forward_feasible(s_route.vehicle_info()),
                       "Delivery node is not forward time feasible!");
          auto next_node = s_route.get_node(i + 1);

          cuopt_assert(check_dim_from_delivery(i + 1, next_node), "dim buffer mismatch");
          calculate_forward_from_delivery(i + 1, next_node);

          iter_node         = next_node;
          beginning_of_hole = delivery_copy;
          size_of_hole      = 0;
        } else {
          size_of_hole++;
          auto next_node = s_route.get_node(i + 1);
          cuopt_assert(i - size_of_hole >= 0, "");

          if (beginning_of_hole.id() == delivery_copy.id()) {
            cuopt_assert(check_dim_from_delivery(i + 1, next_node), "dim buffer mismatch");
            loop_over_dimensions(beginning_of_hole.dimensions_info, [&] __device__(auto I) {
              if (get_dimension_of<I>(beginning_of_hole.dimensions_info).has_constraints()) {
                auto dim_between = get_dim_from_delivery<I>(i + 1);
                get_dimension_of<I>(beginning_of_hole)
                  .calculate_forward(get_dimension_of<I>(next_node), dim_between);
              }
            });
          } else {
            cuopt_assert(check_dim_between(i - size_of_hole, i + 1, beginning_of_hole, next_node),
                         "dim buffer mismatch");
            loop_over_dimensions(beginning_of_hole.dimensions_info, [&] __device__(auto I) {
              if (get_dimension_of<I>(beginning_of_hole.dimensions_info).has_constraints()) {
                auto dim_between = get_dim_between<I>(i - size_of_hole, i + 1);
                get_dimension_of<I>(beginning_of_hole)
                  .calculate_forward(get_dimension_of<I>(next_node), dim_between);
              }
            });
          }

          cuopt_assert(abs(beginning_of_hole.time_dim.departure_forward -
                           get_stack_item(stack_item).departure_forward) < 0.01,
                       "Departure forward mismatch");
          cuopt_assert(abs(beginning_of_hole.time_dim.excess_forward -
                           get_stack_item(stack_item).excess_forward) < 0.01,
                       "excess_forward mismatch");
          constexpr_for<node_t<i_t, f_t, REQUEST>::max_capacity_dim>([&](auto d) {
            if (d < beginning_of_hole.capacity_dim.n_capacity_dimensions) {
              cuopt_assert(beginning_of_hole.capacity_dim.gathered[d] ==
                             get_stack_item(stack_item).gathered[d],
                           "gathered mismatch");
              cuopt_assert(beginning_of_hole.capacity_dim.max_to_node[d] ==
                             get_stack_item(stack_item).max_to_node[d],
                           "max_to_node mismatch");
            }
          });
          iter_node = next_node;
        }
        stack_item++;
        assert(stack_item < stack_top);
      } else {
        auto next_node = s_route.get_node(i + 1);
        cuopt_assert(check_dim_between(i, i + 1, iter_node, next_node), "dim buffer mismatch");
        loop_over_dimensions(iter_node.dimensions_info, [&] __device__(auto I) {
          if (get_dimension_of<I>(iter_node.dimensions_info).has_constraints()) {
            auto dim_between = get_dim_between<I>(i, i + 1);
            get_dimension_of<I>(iter_node).calculate_forward(get_dimension_of<I>(next_node),
                                                             dim_between);
          }
        });
        if (!advance) {
          cuopt_assert(iter_node.forward_feasible(s_route.vehicle_info()),
                       "Iter node is not forward feasible!");
        }
        beginning_of_hole = iter_node;
        iter_node         = next_node;
        size_of_hole      = 0;
      }
    }
    return true;
  }

  template <int I>
  DI bool check_dim_from_delivery(f_t dim_from_buffer, const NodeInfo<i_t> node_info) const
  {
    if (!dim_info().has_dimension((dim_t)I)) { return true; }
    if constexpr (I == (size_t)dim_t::TIME) {
      f_t orig_time = get_arc_of_dimension<i_t, f_t, I>(
        delivery_node.node_info(), node_info, s_route.vehicle_info());
      return orig_time == dim_from_buffer;
    } else if constexpr (I == (size_t)dim_t::DIST) {
      f_t orig_dist = get_arc_of_dimension<i_t, f_t, I>(
        delivery_node.node_info(), node_info, s_route.vehicle_info());
      return orig_dist == dim_from_buffer;
    } else {
      return true;
    }
  }

  template <int I>
  DI bool check_dim_to_delivery(f_t dim_from_buffer, const NodeInfo<i_t> node_info) const
  {
    if (!dim_info().has_dimension((dim_t)I)) { return true; }
    if constexpr (I == (size_t)dim_t::TIME) {
      f_t orig_time = get_arc_of_dimension<i_t, f_t, I>(
        node_info, delivery_node.node_info(), s_route.vehicle_info());
      return orig_time == dim_from_buffer;
    } else if constexpr (I == (size_t)dim_t::DIST) {
      f_t orig_dist = get_arc_of_dimension<i_t, f_t, I>(
        node_info, delivery_node.node_info(), s_route.vehicle_info());
      return orig_dist == dim_from_buffer;
    } else {
      return true;
    }
  }

  template <int I>
  DI bool check_dim_between(f_t dim_from_buffer,
                            const NodeInfo<i_t> node_info_1,
                            const NodeInfo<i_t> node_info_2) const
  {
    if (!dim_info().has_dimension((dim_t)I)) { return true; }
    if constexpr (I == (size_t)dim_t::TIME) {
      f_t orig_time =
        get_arc_of_dimension<i_t, f_t, I>(node_info_1, node_info_2, s_route.vehicle_info());
      return orig_time == dim_from_buffer;
    } else if constexpr (I == (size_t)dim_t::DIST) {
      f_t orig_dist =
        get_arc_of_dimension<i_t, f_t, I>(node_info_1, node_info_2, s_route.vehicle_info());
      return orig_dist == dim_from_buffer;
    } else {
      return true;
    }
  }

  template <request_t r_t, std::enable_if_t<r_t == request_t::VRP, bool> = true>
  DI bool k_max_ejection_check()
  {
    i_t sum_ejections = 0;
    for (i_t i = 0; i < stack_top; ++i) {
      sum_ejections += 1;
    }
    cuopt_assert(sum_ejections == n_ejected_pickups, "Number of ejected nodes don't match!");
    return true;
  }

  template <request_t r_t, std::enable_if_t<r_t == request_t::PDP, bool> = true>
  DI bool k_max_ejection_check()
  {
    i_t sum_ejections = 0;
    for (i_t i = 0; i < stack_top; ++i) {
      if (i == delivery_insertion_idx_in_permutation) continue;
      sum_ejections += i_t(s_route.requests().is_pickup_node(get_stack_item(i).intra_idx));
    }
    cuopt_assert(sum_ejections == n_ejected_pickups, "Number of ejected nodes don't match!");
    return true;
  }

  DI void sanity_checks(bool advance)
  {
    cuopt_assert(node_id_coherence_check(), "");
    cuopt_assert(p_score_check(), "");
    cuopt_assert(k_max_ejection_check<REQUEST>(), "");
    cuopt_assert(forward_check<REQUEST>(advance), "");
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
