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

#include "../cuda_graph.cuh"
#include "../routing_helpers.cuh"
#include "../solution/solution.cuh"
#include "cycle_finder/cycle_graph.hpp"
#include "hvrp/vehicle_assignment.cuh"
#include "sliding_tsp.cuh"

#include <rmm/device_uvector.hpp>

#include <deque>

namespace cuopt {
namespace routing {
namespace detail {

constexpr double SLIDING_ASSERT_EPSILON = EPSILON / 2;

enum class fast_operators_t : int { SLIDING, VRP, CROSS, REGRET, TWO_OPT };

// Used to store the sliding window candidate
// Don't need to store window start since it can be found based on the id where you find one
// solution
template <typename i_t>
struct found_sliding_solution_t {
  double delta;  // Holds the delta cost between old and new route (before/after window insertion)
  i_t window_size;
  i_t intra_insertion_index;  // Window should be inserted after this id (considering the window is
                              // present)
  i_t permutation_index;
  i_t window_start;  // Where the window starts inside the route
};

template <typename i_t>
struct is_sliding_uinitialized_t {
  static constexpr found_sliding_solution_t<i_t> init_data()
  {
    return {std::numeric_limits<double>::max(), -1, -1, -1, -1};
  }

  __device__ bool operator()(const found_sliding_solution_t<i_t>& x)
  {
    return x.delta == std::numeric_limits<double>::max();
  }
};

template <typename i_t>
struct is_sliding_initialized_t {
  __device__ bool operator()(const found_sliding_solution_t<i_t>& x)
  {
    return x.delta != std::numeric_limits<double>::max();
  }
};

template <typename i_t>
struct two_opt_cand_t {
  i_t first;
  i_t second;
  double selection_delta;
  static constexpr two_opt_cand_t<i_t> init_data()
  {
    return two_opt_cand_t<i_t>{-1, -1, std::numeric_limits<double>::max()};
  }
  constexpr bool operator!=(const two_opt_cand_t<i_t>& cand) const
  {
    return this->selection_delta != cand.selection_delta;
  }
};

template <typename i_t>
struct is_two_opt_uinitialized_t {
  static constexpr two_opt_cand_t<i_t> init_data()
  {
    return two_opt_cand_t<i_t>{-1, -1, std::numeric_limits<double>::max()};
  }

  __device__ bool operator()(const two_opt_cand_t<i_t>& x)
  {
    return x.selection_delta == std::numeric_limits<double>::max();
  }
};

template <typename i_t>
struct is_two_opt_initialized_t {
  __device__ bool operator()(const two_opt_cand_t<i_t>& x)
  {
    return x.selection_delta != std::numeric_limits<double>::max();
  }
};

template <typename i_t, typename f_t, request_t REQUEST>
class local_search_t {
 public:
  local_search_t(const solution_handle_t<i_t, f_t>* sol_handle_,
                 i_t n_orders,
                 i_t max_routes,
                 bool depot_included,
                 const viables_t<i_t, f_t>& viables_);
  // computes candidates of insertion and ejection on given solution

  void run_best_local_search(solution_t<i_t, f_t, REQUEST>& sol,
                             const bool consider_unserviced,
                             const bool time_limit_enabled,
                             const bool run_cycle_finder);
  void run_random_local_search(solution_t<i_t, f_t, REQUEST>& sol, bool time_limit_enabled = true);

  void perturb_solution(solution_t<i_t, f_t, REQUEST>& sol, i_t perturb_count = -1);
  void perform_moves(solution_t<i_t, f_t, REQUEST>& solution,
                     move_candidates_t<i_t, f_t>& move_candidates);
  bool perform_sliding_window(solution_t<i_t, f_t, REQUEST>& solution,
                              move_candidates_t<i_t, f_t>& move_candidates);
  bool perform_sliding_tsp(solution_t<i_t, f_t, REQUEST>& solution,
                           move_candidates_t<i_t, f_t>& move_candidates);
  bool perform_two_opt(solution_t<i_t, f_t, REQUEST>& solution,
                       move_candidates_t<i_t, f_t>& move_candidates);
  bool perform_pcross(solution_t<i_t, f_t, REQUEST>& solution,
                      move_candidates_t<i_t, f_t>& move_candidates);
  void populate_move_path(solution_t<i_t, f_t, REQUEST>& solution,
                          move_candidates_t<i_t, f_t>& move_candidates);
  void populate_random_moves(solution_t<i_t, f_t, REQUEST>& solution);
  bool populate_cross_moves(solution_t<i_t, f_t, REQUEST>& solution,
                            move_candidates_t<i_t, f_t>& move_candidates);

  bool perform_prize_collection(solution_t<i_t, f_t, REQUEST>& sol);
  bool perform_break_moves(solution_t<i_t, f_t, REQUEST>& sol);

  void set_active_weights(const infeasible_cost_t weights, bool include_objective = true);

  static inline void start_timer(f_t time_limit_)
  {
    time_limit         = time_limit_;
    start              = std::chrono::steady_clock::now();
    time_limit_reached = false;
  }

  static inline bool get_time_limit_reached()
  {
    bool v;
#pragma omp atomic read
    v = time_limit_reached;
    return v;
  }

  static inline void set_time_limit_reached()
  {
#pragma omp atomic write
    time_limit_reached = true;
  }

  static inline bool check_time_limit()
  {
    if (get_time_limit_reached()) return true;
    bool this_thread_finished =
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start)
        .count() > time_limit;
    if (this_thread_finished) { set_time_limit_reached(); }
    return this_thread_finished;
  }

  i_t max_iterations = std::numeric_limits<i_t>::max();
  // move candidates
  move_candidates_t<i_t, f_t> move_candidates;
  // hvrp
  vehicle_assignment_t<i_t, f_t, REQUEST> vehicle_assignment;
  void calculate_route_compatibility(solution_t<i_t, f_t, REQUEST>& sol);

 private:
  void fill_gpu_graph(solution_t<i_t, f_t, REQUEST>& sol);
  void sort_move_candidates_by_cost(solution_t<i_t, f_t, REQUEST>& sol);
  bool run_sliding_search(solution_t<i_t, f_t, REQUEST>& sol);
  bool run_two_opt_search(solution_t<i_t, f_t, REQUEST>& sol);
  bool run_cross_search(solution_t<i_t, f_t, REQUEST>& sol);
  bool run_inter_search(solution_t<i_t, f_t, REQUEST>& sol);
  template <request_t r_t = REQUEST, std::enable_if_t<r_t == request_t::PDP, bool> = true>
  bool run_fast_search(solution_t<i_t, f_t, r_t>& sol, bool full_set = false);
  template <request_t r_t = REQUEST, std::enable_if_t<r_t == request_t::VRP, bool> = true>
  bool run_fast_search(solution_t<i_t, f_t, r_t>& sol, bool full_set = false);

  void reset_cross_vectors(solution_t<i_t, f_t, REQUEST>& solution);

  bool run_collect_prizes(solution_t<i_t, f_t, REQUEST>& solution);
  void fill_pdp_considered_nodes(solution_t<i_t, f_t, REQUEST>& solution,
                                 move_candidates_t<i_t, f_t>& move_candidates);

  static inline f_t time_limit;
  static inline std::chrono::time_point<std::chrono::steady_clock> start;
  static inline bool time_limit_reached;
  ExactCycleFinder<i_t, f_t, 128> cycle_finder_small;
  ExactCycleFinder<i_t, f_t, 1024> cycle_finder_big;
  rmm::device_uvector<found_sliding_solution_t<i_t>> found_sliding_solution_data_;
  rmm::device_uvector<two_opt_cand_t<i_t>> two_opt_cand_data_;
  rmm::device_uvector<two_opt_cand_t<i_t>> sampled_nodes_data_;
  rmm::device_uvector<i_t> moved_regions_;
  rmm::device_uvector<sliding_tsp_cand_t<i_t>> sampled_tsp_data_;
  rmm::device_uvector<int> locks_;
  // random number generator
  std::mt19937 rng;

  // graphs
  cuda_graph_t sliding_cuda_graph;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
