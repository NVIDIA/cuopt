/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../local_search/local_search.cuh"
#include "../solution/solution.cuh"
#include "ejection_pool.cuh"
#include "found_solution.cuh"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/core/handle.hpp>

#include <map>
#include <optional>
#include <ostream>
#include <set>
#include <utility>

namespace cuopt {
namespace routing {
namespace detail {

// TODO move these constants to a better place
constexpr int allowed_max_k_max         = 6;
constexpr int lexico_result_buffer_size = 5;
constexpr int shuffle_interval          = 20;
constexpr int squeeze_try_interval      = 5;
constexpr int eject_new_route_threshold = shuffle_interval * 5;
constexpr auto const insertion_rate     = 0.003;

struct ges_config_t {
  int frag_eject_first = 1;
  int k_max            = 0;
  int pert             = 0;
  void print()
  {
    printf("frag_eject_first:%d , k_max: %d , pert:%d \n", frag_eject_first, k_max, pert);
  }
};

template <typename i_t, typename f_t, request_t REQUEST>
struct next_route_id_t {
  next_route_id_t(solution_t<i_t, f_t, REQUEST> const* solution_ptr_)
    : solution_ptr(solution_ptr_), dist(0, std::numeric_limits<i_t>::max()), gen(rd())
  {
  }

  std::tuple<i_t, i_t> operator()()
  {
    auto route_id     = dist(gen) % solution_ptr->get_n_routes();
    const auto& route = solution_ptr->get_route(route_id);
    return std::make_tuple(route.vehicle_id.value(route.vehicle_id.stream()), route_id);
  }

  solution_t<i_t, f_t, REQUEST> const* solution_ptr;
  // Uniform distribution to select a random route to delete
  std::uniform_int_distribution<i_t> dist;
  std::random_device rd;
  std::mt19937 gen;
};

template <typename i_t, typename f_t, request_t REQUEST>
class guided_ejection_search_t {
 public:
  explicit guided_ejection_search_t(solution_t<i_t, f_t, REQUEST>& dummy_sol,
                                    local_search_t<i_t, f_t, REQUEST>* local_search,
                                    std::ofstream* intermediate_file = nullptr);
  bool guided_ejection_search_loop(i_t& counter, bool minimize_routes, i_t desired_ep_size = 0);
  bool greedy_insert(bool insert_all = false);
  bool time_stop_condition_reached();
  void start_timer(std::chrono::time_point<std::chrono::steady_clock> start_time,
                   f_t intial_sol_time_limit);
  void set_solution_ptr(solution_t<i_t, f_t, REQUEST>* solution_ptr, bool clear_scores = true);
  // cannot be private because used in tests
  bool run_lexicographic_search(request_info_t<i_t, REQUEST>* __restrict__ request_id);
  void route_minimizer_loop();
  bool fixed_route_loop();
  bool construct_feasible_solution();
  // cannot be private due to device lambda
  void init_ejection_pool();
  bool try_squeeze_feasible(const request_info_t<i_t, REQUEST>* request, bool random_route = true);
  void squeeze(const request_info_t<i_t, REQUEST>* request, bool random_route = true);
  void squeeze_all_ep();
  bool repair_empty_routes();
  template <bool squeeze_mode>
  i_t try_multiple_insert(i_t n_insertions,
                          infeasible_cost_t weights,
                          double excess_limit,
                          bool include_objective);
  i_t try_multiple_feasible_insertions(i_t n_insertions, bool enable_perturbation);
  i_t get_num_non_empty_vehicles();
  // used in tests
  std::vector<i_t> brute_force_lexico(solution_t<i_t, f_t, REQUEST>& sol,
                                      request_info_t<i_t, REQUEST>* __restrict__ req);

  void squeeze_breaks();
  bool try_squeeze_breaks_feasible();

  rmm::device_uvector<i_t> p_scores_;
  rmm::device_scalar<uint32_t> global_min_p_;
  rmm::device_scalar<i_t> global_random_counter_;
  rmm::device_uvector<i_t> global_sequence_;
  solution_t<i_t, f_t, REQUEST>* solution_ptr;
  solution_t<i_t, f_t, REQUEST> ges_loop_save_state;
  local_search_t<i_t, f_t, REQUEST>* local_search_ptr_;
  ges_config_t config{};
  ejection_pool_t<request_info_t<i_t, REQUEST>> EP;
  f_t min_excess_overall = std::numeric_limits<f_t>::max();
  i_t min_ep_size        = std::numeric_limits<i_t>::max();

 private:
  bool execute_best_insertion_ejection_solution(request_info_t<i_t, REQUEST>* d_request,
                                                i_t& counter);

  template <request_t r_t = REQUEST, std::enable_if_t<r_t == request_t::VRP, bool> = true>
  void reset_p_scores();
  template <request_t r_t = REQUEST, std::enable_if_t<r_t == request_t::PDP, bool> = true>
  void reset_p_scores();
  bool perform_insertion(const request_info_t<i_t, REQUEST>* request);
  bool try_single_insert_with_perturbation(const request_info_t<i_t, REQUEST>* request);
  i_t find_single_insertion(const request_info_t<i_t, REQUEST>* request);
  void dump_to_file(std::string msg);
  void shuffle_pool();
  void squeeze_remaining_requests();
  bool squeeze_all_and_save();
  f_t remaining_time() const;

  // Data reuse for get_all_feasible_insertion
  // 0, Route_id, pickup_position, delivery_position
  rmm::device_uvector<found_sol_t> feasible_candidates_data_;
  rmm::device_scalar<i_t> feasible_candidates_size_;
  std::uniform_int_distribution<i_t> dist_candidate{0, std::numeric_limits<i_t>::max()};
  std::mt19937 gen_candidate;

  solution_t<i_t, f_t, REQUEST> squeeze_save_state;

  rmm::device_uvector<cand_t> best_squeeze_per_cand;
  rmm::device_uvector<cand_t> best_squeeze_per_route;
  // used in squeeze
  rmm::device_uvector<i_t> inserted_requests;
  rmm::device_scalar<i_t> number_of_inserted;

  // Data reuse for execute_best_insertion_ejection_solution
  // Size is n_orders to allow direct indexing from pickup (request) id

  f_t time_limit;
  std::chrono::time_point<std::chrono::steady_clock> start;
  std::ofstream* intermediate_file;
  bool dump_intermediate;
  i_t dump_iter = 0;
  // for now set here, later read from settings file
  f_t dump_interval = 60.f;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
