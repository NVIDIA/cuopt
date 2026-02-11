/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include <mip/feasibility_jump/feasibility_jump.cuh>
#include <mip/utilities/cpu_worker_thread.cuh>
#include <utilities/memory_instrumentation.hpp>
#include <utilities/producer_sync.hpp>

namespace cuopt::linear_programming::detail {

// NOTE: this seems an easy pick for reflection/xmacros once this is available (C++26?)
// Maintaining a single source of truth for all members would be nice
template <typename i_t, typename f_t>
struct fj_cpu_climber_t {
  fj_cpu_climber_t(std::atomic<bool>& preemption_flag) : preemption_flag(preemption_flag)
  {
#define ADD_INSTRUMENTED(var) \
  std::make_pair(#var, std::ref(static_cast<memory_instrumentation_base_t&>(var)))

    // Initialize memory aggregator with all ins_vector members
    memory_aggregator = instrumentation_aggregator_t{ADD_INSTRUMENTED(h_reverse_coefficients),
                                                     ADD_INSTRUMENTED(h_reverse_constraints),
                                                     ADD_INSTRUMENTED(h_reverse_offsets),
                                                     ADD_INSTRUMENTED(h_coefficients),
                                                     ADD_INSTRUMENTED(h_offsets),
                                                     ADD_INSTRUMENTED(h_variables),
                                                     ADD_INSTRUMENTED(h_obj_coeffs),
                                                     ADD_INSTRUMENTED(h_var_bounds),
                                                     ADD_INSTRUMENTED(h_cstr_lb),
                                                     ADD_INSTRUMENTED(h_cstr_ub),
                                                     ADD_INSTRUMENTED(h_var_types),
                                                     ADD_INSTRUMENTED(h_is_binary_variable),
                                                     ADD_INSTRUMENTED(h_objective_vars),
                                                     ADD_INSTRUMENTED(h_binary_indices),
                                                     ADD_INSTRUMENTED(h_tabu_nodec_until),
                                                     ADD_INSTRUMENTED(h_tabu_noinc_until),
                                                     ADD_INSTRUMENTED(h_tabu_lastdec),
                                                     ADD_INSTRUMENTED(h_tabu_lastinc),
                                                     ADD_INSTRUMENTED(h_lhs),
                                                     ADD_INSTRUMENTED(h_lhs_sumcomp),
                                                     ADD_INSTRUMENTED(h_cstr_left_weights),
                                                     ADD_INSTRUMENTED(h_cstr_right_weights),
                                                     ADD_INSTRUMENTED(h_assignment),
                                                     ADD_INSTRUMENTED(h_best_assignment),
                                                     ADD_INSTRUMENTED(cached_cstr_bounds),
                                                     ADD_INSTRUMENTED(iter_mtm_vars)};

#undef ADD_INSTRUMENTED
  }
  fj_cpu_climber_t(const fj_cpu_climber_t<i_t, f_t>& other)                      = delete;
  fj_cpu_climber_t<i_t, f_t>& operator=(const fj_cpu_climber_t<i_t, f_t>& other) = delete;

  fj_cpu_climber_t(fj_cpu_climber_t<i_t, f_t>&& other)                      = default;
  fj_cpu_climber_t<i_t, f_t>& operator=(fj_cpu_climber_t<i_t, f_t>&& other) = default;

  problem_t<i_t, f_t>* pb_ptr;
  fj_settings_t settings;
  typename fj_t<i_t, f_t>::climber_data_t::view_t view;
  // Host copies of device data as struct members
  ins_vector<f_t> h_reverse_coefficients;
  ins_vector<i_t> h_reverse_constraints;
  ins_vector<i_t> h_reverse_offsets;
  ins_vector<f_t> h_coefficients;
  ins_vector<i_t> h_offsets;
  ins_vector<i_t> h_variables;
  ins_vector<f_t> h_obj_coeffs;
  ins_vector<typename type_2<f_t>::type> h_var_bounds;
  ins_vector<f_t> h_cstr_lb;
  ins_vector<f_t> h_cstr_ub;
  ins_vector<var_t> h_var_types;
  ins_vector<i_t> h_is_binary_variable;
  ins_vector<i_t> h_objective_vars;
  ins_vector<i_t> h_binary_indices;

  ins_vector<i_t> h_tabu_nodec_until;
  ins_vector<i_t> h_tabu_noinc_until;
  ins_vector<i_t> h_tabu_lastdec;
  ins_vector<i_t> h_tabu_lastinc;

  ins_vector<f_t> h_lhs;
  ins_vector<f_t> h_lhs_sumcomp;
  ins_vector<f_t> h_cstr_left_weights;
  ins_vector<f_t> h_cstr_right_weights;
  f_t max_weight;
  ins_vector<f_t> h_assignment;
  ins_vector<f_t> h_best_assignment;
  f_t h_objective_weight;
  f_t h_incumbent_objective;
  f_t h_best_objective;
  i_t last_feasible_entrance_iter{0};
  i_t iterations;
  std::unordered_set<i_t> violated_constraints;
  std::unordered_set<i_t> satisfied_constraints;
  bool feasible_found{false};
  bool trigger_early_lhs_recomputation{false};
  f_t total_violations{0};

  // Timing data structures
  std::vector<double> find_lift_move_times;
  std::vector<double> find_mtm_move_viol_times;
  std::vector<double> find_mtm_move_sat_times;
  std::vector<double> apply_move_times;
  std::vector<double> update_weights_times;
  std::vector<double> compute_score_times;

  i_t hit_count{0};
  i_t miss_count{0};

  i_t candidate_move_hits[3]   = {0};
  i_t candidate_move_misses[3] = {0};

  // vector<bool> is actually likely beneficial here since we're memory bound
  std::vector<bool> flip_move_computed;
  ;
  // CSR nnz offset -> (delta, score)
  std::vector<std::pair<f_t, fj_staged_score_t>> cached_mtm_moves;

  // CSC (transposed!) nnz-offset-indexed constraint bounds (lb, ub)
  // std::pair<f_t, f_t> better compile down to 16 bytes!! GCC do your job!
  ins_vector<std::pair<f_t, f_t>> cached_cstr_bounds;

  std::vector<bool> var_bitmap;
  ins_vector<i_t> iter_mtm_vars;

  i_t mtm_viol_samples{25};
  i_t mtm_sat_samples{15};
  i_t nnz_samples{50000};
  i_t perturb_interval{100};

  i_t log_interval{1000};
  i_t diversity_callback_interval{3000};
  i_t timing_stats_interval{5000};

  // Callback with work unit timestamp for deterministic mode
  // Parameters: objective, solution, work_units
  std::function<void(f_t, const std::vector<f_t>&, double)> improvement_callback{nullptr};
  std::function<void(f_t, const std::vector<f_t>&)> diversity_callback{nullptr};
  std::string log_prefix{""};

  // Work unit tracking for deterministic synchronization
  std::atomic<double> work_units_elapsed{0.0};
  double work_unit_bias{1.5};               // Bias factor to keep CPUFJ ahead of B&B
  producer_sync_t* producer_sync{nullptr};  // Optional sync utility for notifying progress

  std::atomic<bool> halted{false};

  // Feature tracking for regression model (last 1000 iterations)
  i_t nnz_processed_window{0};
  i_t n_lift_moves_window{0};
  i_t n_mtm_viol_moves_window{0};
  i_t n_mtm_sat_moves_window{0};
  i_t n_variable_updates_window{0};
  i_t n_local_minima_window{0};
  std::chrono::high_resolution_clock::time_point last_feature_log_time;
  f_t prev_best_objective{std::numeric_limits<f_t>::infinity()};
  i_t iterations_since_best{0};

  // Cache and locality tracking
  i_t hit_count_window_start{0};
  i_t miss_count_window_start{0};
  std::unordered_set<i_t> unique_cstrs_accessed_window;
  std::unordered_set<i_t> unique_vars_accessed_window;

  // Precomputed static problem features
  i_t n_binary_vars{0};
  i_t n_integer_vars{0};
  i_t max_var_degree{0};
  i_t max_cstr_degree{0};
  double avg_var_degree{0.0};
  double avg_cstr_degree{0.0};
  double var_degree_cv{0.0};
  double cstr_degree_cv{0.0};
  double problem_density{0.0};

  // Memory instrumentation aggregator
  instrumentation_aggregator_t memory_aggregator;
  // TODO atomic ref? c++20
  std::atomic<bool>& preemption_flag;
};

template <typename i_t, typename f_t>
struct cpu_fj_thread_t : public cpu_worker_thread_base_t<cpu_fj_thread_t<i_t, f_t>> {
  ~cpu_fj_thread_t();

  void run_worker();
  void on_terminate();
  void on_start();
  bool get_result() { return cpu_fj_solution_found; }

  void stop_cpu_solver();

  std::atomic<bool> cpu_fj_solution_found{false};
  f_t time_limit{+std::numeric_limits<f_t>::infinity()};
  std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> fj_cpu;
  fj_t<i_t, f_t>* fj_ptr{nullptr};
};

}  // namespace cuopt::linear_programming::detail
