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

#include <utilities/vector_helpers.cuh>
#include "../../cuda_graph.cuh"
#include "../../node/node.cuh"
#include "../../optional.cuh"
#include "../../problem/problem.cuh"
#include "../../solution/solution_handle.cuh"
#include "../cycle_finder/cycle.hpp"
#include "../cycle_finder/cycle_finder.hpp"
#include "../vrp/nodes_to_search.cuh"
#include "breaks_move_candidates.cuh"
#include "prize_move_candidates.cuh"
#include "random_move_candidates.cuh"
#include "scross_move_candidates.cuh"
#include "vrp_move_candidates.cuh"

#include <raft/common/nvtx.hpp>
#include <raft/random/rng_device.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <utilities/vector_helpers.cuh>

#include <condition_variable>
#include <queue>
#include <tuple>

namespace cuopt {
namespace routing {
namespace detail {

constexpr float ls_excess_multiplier_route = 1.2f;

template <typename object_t, int n_instances = 1>
struct shared_pool_t {
  template <typename... Args>
  shared_pool_t(Args... args)
  {
    shared_resources.reserve(n_instances);
    for (int i = 0; i < n_instances; ++i) {
      shared_resources.emplace_back(args...);
      indices.push(i);
    }
  }

  std::tuple<object_t&, int> acquire()
  {
    std::unique_lock<std::mutex> lock(mutex);
    while (true) {
      if (indices.size()) {
        int index = indices.front();
        indices.pop();
        return std::tie(shared_resources[index], index);
      }
      cond.wait(lock);
    };
  }

  void release(int index)
  {
    std::unique_lock<std::mutex> lock(mutex);
    indices.push(index);
    cond.notify_one();
  }

  std::mutex mutex;
  std::condition_variable cond;
  std::queue<int> indices;
  std::vector<object_t> shared_resources;
};

template <typename i_t, typename f_t>
class move_path_t {
 public:
  move_path_t(i_t n_routes, solution_handle_t<i_t, f_t> const* sol_handle_)
    : path(n_routes * 2, sol_handle_->get_stream()),
      loop_closed(n_routes, sol_handle_->get_stream()),
      changed_routes(n_routes, sol_handle_->get_stream()),
      n_insertions(sol_handle_->get_stream())
  {
  }

  static cand_t HDI make_cycle_edge(i_t pickup_node,
                                    i_t route_id,
                                    i_t pickup_insertion,
                                    i_t delivery_insertion)
  {
    cuopt_assert(pickup_node < (1 << 15), "pickup_node cannot exceed 2^15");
    cuopt_assert(route_id < (1 << 15), "route_id cannot exceed 2^15");
    cuopt_assert(pickup_insertion < (1 << 15), "insertion_1 cannot exceed 2^15");
    cuopt_assert(delivery_insertion < (1 << 15), "insertion_2 cannot exceed 2^15");
    cuopt_assert(pickup_insertion <= delivery_insertion,
                 "pickup_insertion cannot be bigger than delivery_insertion");
    uint pair_1 = (pickup_node << 16) | (route_id);
    uint pair_2 = (pickup_insertion << 16) | (delivery_insertion);
    return cand_t{pair_1, pair_2, 0.};
  }

  template <request_t REQUEST>
  static void HDI get_cycle_edge(const cand_t cand,
                                 i_t& pickup_node,
                                 i_t& route_id,
                                 request_id_t<REQUEST>& request_location)
  {
    pickup_node           = (i_t)(cand.pair_1 >> 16);
    route_id              = (i_t)(cand.pair_1 & 0xFFFF);
    request_location.id() = (i_t)(cand.pair_2 >> 16);
    if constexpr (REQUEST == request_t::PDP) {
      request_location.delivery = (i_t)(cand.pair_2 & 0xFFFF);
      cuopt_assert(request_location.pickup <= request_location.delivery,
                   "corrupt cycle edge, pickup insertion is bigger than delivery insertion");
    }
  }

  void reset(solution_handle_t<i_t, f_t> const* sol_handle)
  {
    constexpr i_t zero_val = 0;
    n_insertions.set_value_async(zero_val, sol_handle->get_stream());
    async_fill(loop_closed, 1, sol_handle->get_stream());
    async_fill(changed_routes, 0, sol_handle->get_stream());
  }

  struct view_t {
    cand_t* path;
    i_t* loop_closed;
    i_t* changed_routes;
    i_t* n_insertions;
  };

  view_t view()
  {
    view_t v;
    v.path           = path.data();
    v.loop_closed    = loop_closed.data();
    v.changed_routes = changed_routes.data();
    v.n_insertions   = n_insertions.data();
    return v;
  }
  // cand_t is uint4 defined in move_candidates
  // we don't need to keep the cycle weight
  // the insertion_pairs should be ordered and carefully taken
  rmm::device_uvector<cand_t> path;
  rmm::device_uvector<i_t> loop_closed;
  rmm::device_uvector<i_t> changed_routes;
  rmm::device_scalar<i_t> n_insertions;
};

template <typename i_t, typename f_t>
class cand_matrix_t {
 public:
  cand_matrix_t(i_t n_orders, i_t n_routes, solution_handle_t<i_t, f_t> const* sol_handle_)
    : matrix_width(n_orders + n_routes + 1),
      matrix_height(n_orders + n_routes + 1),
      pair_1(matrix_height * matrix_width, sol_handle_->get_stream()),
      pair_2(matrix_height * matrix_width, sol_handle_->get_stream()),
      cost_counter(matrix_height * matrix_width, sol_handle_->get_stream()),
      intra_pair_1(n_routes, sol_handle_->get_stream()),
      intra_pair_2(n_routes, sol_handle_->get_stream()),
      intra_cost_counter(n_routes, sol_handle_->get_stream()),
      cand_locks(matrix_height * matrix_width, sol_handle_->get_stream()),
      d_cub_storage_bytes(0, sol_handle_->get_stream())
  {
    // the size of cand_locks never change. we acquire and release, so it will always reset itself
    // within the kernel
    thrust::fill(sol_handle_->get_thrust_policy(), cand_locks.begin(), cand_locks.end(), 0);
    async_fill(cand_locks, 0, sol_handle_->get_stream());
  }

  struct view_t {
    DI double get_cost(const i_t idx) { return cost_counter[idx].cost; }

    DI double get_cost(const i_t source, const i_t sink)
    {
      cuopt_assert(sink < matrix_width, "Sink should be smaller than matrix_width!");
      cuopt_assert(source < matrix_height, "Source should be smaller than matrix_height!");
      i_t idx = source * matrix_width + sink;
      return get_cost(idx);
    }

    // source and sink represent the indices in the candidate matrix
    DI cand_t get_candidate(const i_t idx)
    {
      cand_t cand;
      cand.pair_1       = pair_1[idx];
      cand.pair_2       = pair_2[idx];
      cand.cost_counter = cost_counter[idx];
      return cand;
    }

    // source and sink represent the indices in the candidate matrix
    DI cand_t get_candidate(const i_t source, const i_t sink)
    {
      cuopt_assert(sink < matrix_width, "Sink should be smaller than matrix_width!");
      cuopt_assert(source < matrix_height, "Source should be smaller than matrix_height!");
      cand_t cand;
      i_t idx = source * matrix_width + sink;
      return get_candidate(idx);
    }

    // source and sink represent the indices in the candidate matrix
    DI cand_t get_intra_candidate(const i_t idx)
    {
      cand_t cand;
      cand.pair_1       = intra_pair_1[idx];
      cand.pair_2       = intra_pair_2[idx];
      cand.cost_counter = intra_cost_counter[idx];
      return cand;
    }

    // source and sink represent the indices in the candidate matrix
    DI void set_intra_candidate(const cand_t cand, const i_t idx)
    {
      intra_pair_1[idx]       = cand.pair_1;
      intra_pair_2[idx]       = cand.pair_2;
      intra_cost_counter[idx] = cand.cost_counter;
    }
    // source and sink represent the indices in the candidate matrix
    DI void record_candidate(const cand_t cand, const i_t source, const i_t sink)
    {
      cuopt_assert(sink < matrix_width, "Sink should be smaller than matrix_width!");
      cuopt_assert(source < matrix_height, "Source should be smaller than matrix_height!");
      i_t idx           = source * matrix_width + sink;
      pair_1[idx]       = cand.pair_1;
      pair_2[idx]       = cand.pair_2;
      cost_counter[idx] = cand.cost_counter;
    }

    // source and sink represent the indices in the candidate matrix
    DI void record_if_better(const cand_t cand, const i_t source, const i_t sink)
    {
      cuopt_assert(sink < matrix_width, "Sink should be smaller than matrix_width!");
      cuopt_assert(source < matrix_height, "Source should be smaller than matrix_height!");
      i_t idx = source * matrix_width + sink;
      if (cand.cost_counter.cost < cost_counter[source * matrix_width + sink].cost)
        record_candidate(cand, source, sink);
    }

    // source and sink represent the indices in the candidate matrix
    DI void record_candidate_thread_safe(const cand_t cand, const i_t source, const i_t sink)
    {
      cuopt_assert(sink < matrix_width, "Sink should be smaller than matrix_width!");
      cuopt_assert(source < matrix_height, "Source should be smaller than matrix_height!");
      // an early check before acquiring the mutex
      if (cand.cost_counter.cost < cost_counter[source * matrix_width + sink].cost) {
        acquire_lock(&cand_locks[source * matrix_width + sink]);
        record_if_better(cand, source, sink);

        release_lock(&cand_locks[source * matrix_width + sink]);
      }
    }

    i_t matrix_width;
    i_t matrix_height;
    raft::device_span<i_t> pair_1;
    raft::device_span<i_t> pair_2;
    raft::device_span<cost_counter_t> cost_counter;
    raft::device_span<i_t> intra_pair_1;
    raft::device_span<i_t> intra_pair_2;
    raft::device_span<cost_counter_t> intra_cost_counter;
    raft::device_span<i_t> cand_locks;
  };

  void reset(solution_handle_t<i_t, f_t> const* sol_handle)
  {
    raft::common::nvtx::range fun_scope("cand_matrix_t reset");
    async_fill(cost_counter,
               cost_counter_t{.cost = std::numeric_limits<double>::max()},
               sol_handle->get_stream());
    async_fill(intra_cost_counter,
               cost_counter_t{.cost = std::numeric_limits<double>::max()},
               sol_handle->get_stream());
  }

  view_t view()
  {
    view_t v;
    v.matrix_width  = matrix_width;
    v.matrix_height = matrix_height;
    v.pair_1        = raft::device_span<i_t>{pair_1.data(), pair_1.size()};
    v.pair_2        = raft::device_span<i_t>{pair_2.data(), pair_2.size()};
    v.cost_counter  = raft::device_span<cost_counter_t>{cost_counter.data(), cost_counter.size()};
    v.intra_pair_1  = raft::device_span<i_t>{intra_pair_1.data(), intra_pair_1.size()};
    v.intra_pair_2  = raft::device_span<i_t>{intra_pair_2.data(), intra_pair_2.size()};
    v.intra_cost_counter =
      raft::device_span<cost_counter_t>{intra_cost_counter.data(), intra_cost_counter.size()};
    v.cand_locks = raft::device_span<i_t>{cand_locks.data(), cand_locks.size()};
    return v;
  }

  i_t matrix_width;
  i_t matrix_height;
  rmm::device_uvector<i_t> pair_1;
  rmm::device_uvector<i_t> pair_2;
  rmm::device_uvector<cost_counter_t> cost_counter;
  rmm::device_uvector<i_t> intra_pair_1;
  rmm::device_uvector<i_t> intra_pair_2;
  rmm::device_uvector<cost_counter_t> intra_cost_counter;
  rmm::device_uvector<i_t> cand_locks;
  rmm::device_uvector<std::byte> d_cub_storage_bytes;
};

template <typename i_t, typename f_t>
class move_candidates_t {
 public:
  move_candidates_t(i_t n_orders,
                    i_t n_routes,
                    solution_handle_t<i_t, f_t> const* sol_handle_,
                    const viables_t<i_t, f_t>& viables_)
    : cand_matrix(n_orders, n_routes, sol_handle_),
      debug_delta(sol_handle_->get_stream()),
      temp_storage(0, sol_handle_->get_stream()),
      graph(n_orders + n_routes + 1, sol_handle_->get_stream()),
      cycles(2 * n_routes, sol_handle_->get_stream()),
      move_path(n_routes, sol_handle_),
      viables(viables_),
      route_compatibility(n_routes * n_orders, sol_handle_->get_stream()),
      scross_move_candidates(sol_handle_),
      vrp_move_candidates(n_orders, n_routes, sol_handle_),
      prize_move_candidates(n_routes, sol_handle_),
      breaks_move_candidates(n_routes, sol_handle_),
      random_move_candidates(n_routes, n_orders, sol_handle_),
      nodes_to_search(n_orders, n_routes, sol_handle_)
  {
  }

  void find_best_negative_cycles(i_t pseudo_node_number,
                                 ExactCycleFinder<i_t, f_t, 128>& cycle_finder_small,
                                 ExactCycleFinder<i_t, f_t, 1024>& cycle_finder_big,
                                 solution_handle_t<i_t, f_t> const* sol_handle)
  {
    if (pseudo_node_number > 127) {
      cycle_finder_big.find_best_cycles(graph, cycles, sol_handle);
    } else {
      cycle_finder_small.find_best_cycles(graph, cycles, sol_handle);
    }
    sol_handle->sync_stream();
  }

  void set_random_selection_weights(std::mt19937& rng)
  {
    selection_weights = default_weights;
    // get a random uniform between 0.1 - 1.
    std::uniform_real_distribution<> uni_dis(0.1, 1.0);
    selection_weights[dim_t::TIME] = uni_dis(rng);
    selection_weights[dim_t::CAP]  = uni_dis(rng);
  }

  inline void reset(solution_handle_t<i_t, f_t> const* sol_handle)
  {
    raft::common::nvtx::range fun_scope("move_candidates reset");
    move_candidate_reset_graph.start_capture(sol_handle->get_stream());
    cycles.reset(sol_handle);
    graph.reset(sol_handle);
    move_path.reset(sol_handle);
    cand_matrix.reset(sol_handle);
    cuopt_func_call(debug_delta.set_value_to_zero_async(sol_handle->get_stream()));
    move_candidate_reset_graph.end_capture(sol_handle->get_stream());
    move_candidate_reset_graph.launch_graph(sol_handle->get_stream());
    sol_handle->sync_stream();
  }

  // currently candidate is 128-bits represented by int4
  DI static cand_t make_candidate(
    i_t insertion_1, i_t insertion_2, i_t insertion_3, i_t insertion_4, double cost_delta)
  {
    // assuming that request route is smaller than or equal to 15-bits
    // so that retriving back doesn't cause undefined behavior
    cuopt_assert(insertion_1 < (1 << 15), "insertion_1 cannot exceed 2^15");
    cuopt_assert(insertion_2 < (1 << 15), "insertion_2 cannot exceed 2^15");
    cuopt_assert(insertion_3 < (1 << 15), "insertion_3 cannot exceed 2^15");
    cuopt_assert(insertion_4 < (1 << 15), "insertion_4 cannot exceed 2^15");
    cuopt_assert(insertion_1 >= 0, "insertion_1 cannot be negative");
    cuopt_assert(insertion_2 >= 0, "insertion_2 cannot be negative");
    cuopt_assert(insertion_3 >= 0, "insertion_3 cannot be negative");
    cuopt_assert(insertion_4 >= 0, "insertion_4 cannot be negative");

    cuopt_assert(insertion_1 <= insertion_2, "the candidate should be in lexicographic order");
    cuopt_assert(insertion_3 <= insertion_4, "the candidate should be in lexicographic order");

    uint pair_1 = (insertion_1 << 16) | (insertion_2);
    uint pair_2 = (insertion_3 << 16) | (insertion_4);
    return cand_t{pair_1, pair_2, cost_delta};
  }

  // for the cross candidate, pickup_ids already imply the pickup
  DI static cross_cand_t make_cross_candidate(const cand_t& cand_1,
                                              const cand_t& cand_2,
                                              int id_1,
                                              int id_2)
  {
    i_t insertion_1, insertion_2, insertion_3, insertion_4, dump_1, dump_2;
    double cost_delta_1, cost_delta_2;
    get_candidate(cand_1, insertion_1, insertion_2, dump_1, dump_2, cost_delta_1);
    get_candidate(cand_2, insertion_3, insertion_4, dump_1, dump_2, cost_delta_2);
    cand_t cand = make_candidate(
      insertion_1, insertion_2, insertion_3, insertion_4, cost_delta_1 + cost_delta_2);
    cross_cand_t c_cand{cand, id_1, id_2};
    return c_cand;
  }

  // currently candidate is 128-bits represented by uint4
  HDI static void get_candidate(const cand_t& cand,
                                i_t& insertion_1,
                                i_t& insertion_2,
                                i_t& insertion_3,
                                i_t& insertion_4,
                                double& cost_delta)
  {
    insertion_1 = i_t(cand.pair_1 >> 16);
    insertion_2 = i_t(cand.pair_1 & 0XFFFFU);
    insertion_3 = i_t(cand.pair_2 >> 16);
    insertion_4 = i_t(cand.pair_2 & 0XFFFFU);
    cost_delta  = cand.cost_counter.cost;
  }

  struct view_t {
    typename cand_matrix_t<i_t, f_t>::view_t cand_matrix;
    i_t cross_moves_per_route;
    double* debug_delta;
    typename graph_t<i_t, f_t>::view_t graph;
    typename move_path_t<i_t, f_t>::view_t move_path;
    typename ret_cycles_t<i_t, f_t>::view_t cycles;
    infeasible_cost_t weights;
    infeasible_cost_t selection_weights;
    bool include_objective;
    typename viables_t<i_t, f_t>::view_t viables;
    raft::device_span<uint8_t> route_compatibility;
    typename scross_move_candidates_t<i_t, f_t>::view_t scross_move_candidates;
    typename vrp_move_candidates_t<i_t, f_t>::view_t vrp_move_candidates;
    typename prize_move_candidates_t<i_t, f_t>::view_t prize_move_candidates;
    typename breaks_move_candidates_t<i_t, f_t>::view_t breaks_move_candidates;
    typename random_move_candidates_t<i_t, f_t>::view_t random_move_candidates;
    typename nodes_to_search_t<i_t, f_t>::view_t nodes_to_search;
    i_t number_of_blocks_per_ls_route;

    // source and sink represent the indices in the candidate matrix
    DI void record_candidate(const cand_t cand, const i_t source, const i_t sink)
    {
      cand_matrix.record_candidate(cand, source, sink);
    }

    // source and sink represent the indices in the candidate matrix
    DI void record_if_better(const cand_t cand, const i_t source, const i_t sink)
    {
      cand_matrix.record_if_better(cand, source, sink);
    }

    // source and sink represent the indices in the candidate matrix
    DI void record_candidate_thread_safe(const cand_t cand, const i_t source, const i_t sink)
    {
      cand_matrix.record_candidate_thread_safe(cand, source, sink);
    }
  };

  view_t view()
  {
    view_t v;
    v.cand_matrix       = cand_matrix.view();
    v.debug_delta       = debug_delta.data();
    v.graph             = graph.view();
    v.move_path         = move_path.view();
    v.cycles            = cycles.view();
    v.weights           = weights;
    v.selection_weights = selection_weights;
    v.viables           = viables.view();
    v.route_compatibility =
      raft::device_span<uint8_t>{route_compatibility.data(), route_compatibility.size()};
    v.scross_move_candidates        = scross_move_candidates.view();
    v.vrp_move_candidates           = vrp_move_candidates.view();
    v.prize_move_candidates         = prize_move_candidates.view();
    v.breaks_move_candidates        = breaks_move_candidates.view();
    v.random_move_candidates        = random_move_candidates.view();
    v.nodes_to_search               = nodes_to_search.view();
    v.include_objective             = include_objective;
    v.number_of_blocks_per_ls_route = number_of_blocks_per_ls_route;
    return v;
  }

  cand_matrix_t<i_t, f_t> cand_matrix;
  // cost delta used in runtime tests
  rmm::device_scalar<double> debug_delta;
  rmm::device_uvector<std::byte> temp_storage;
  i_t special_index;
  graph_t<i_t, f_t> graph;
  move_path_t<i_t, f_t> move_path;
  ret_cycles_t<i_t, f_t> cycles;
  infeasible_cost_t weights;
  infeasible_cost_t selection_weights;
  bool include_objective;
  // viable structure
  const viables_t<i_t, f_t>& viables;
  // route_compatibility, this is dynamic and is filled in place
  rmm::device_uvector<uint8_t> route_compatibility;
  scross_move_candidates_t<i_t, f_t> scross_move_candidates;
  vrp_move_candidates_t<i_t, f_t> vrp_move_candidates;
  prize_move_candidates_t<i_t, f_t> prize_move_candidates;
  breaks_move_candidates_t<i_t, f_t> breaks_move_candidates;
  random_move_candidates_t<i_t, f_t> random_move_candidates;
  nodes_to_search_t<i_t, f_t> nodes_to_search;
  i_t number_of_blocks_per_ls_route;

  cuda_graph_t move_candidate_reset_graph;
  cuda_graph_t vrp_execute_graph;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
