/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>

#include <mip_heuristics/feasibility_jump/feasibility_jump.cuh>
#include <mip_heuristics/feasibility_jump/fj_cpu_worker.cuh>
#include <utilities/memory_instrumentation.hpp>
#include <utilities/producer_sync.hpp>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
class probing_cache_t;

template <typename i_t>
struct host_contiguous_set_t {
  void resize(i_t max_size)
  {
    cuopt_assert(max_size >= 0, "invalid max size");
    contents.clear();
    contents.reserve(max_size);
    index_map.assign(max_size, -1);
    is_member.assign(max_size, 0);
  }

  void clear()
  {
    for (i_t val : contents) {
      index_map[val] = -1;
      is_member[val] = 0;
    }
    contents.clear();
  }

  void insert(i_t val)
  {
    cuopt_assert(val >= 0 && val < max_size(), "Value is out of bounds");
    cuopt_assert(!contains(val), "Value already exists");
    index_map[val] = contents.size();
    is_member[val] = 1;
    contents.push_back(val);
  }

  void remove(i_t val)
  {
    cuopt_assert(val >= 0 && val < max_size(), "Value is out of bounds");
    cuopt_assert(contains(val), "Value not found");
    const i_t idx       = index_map[val];
    const i_t last_val  = contents.back();
    contents[idx]       = last_val;
    index_map[last_val] = idx;
    contents.pop_back();
    index_map[val] = -1;
    is_member[val] = 0;
  }

  bool contains(i_t val) const
  {
    cuopt_assert(val >= 0 && val < max_size(), "Value is out of bounds");
    return is_member[val] != 0;
  }

  auto begin() const { return contents.begin(); }
  auto end() const { return contents.end(); }
  i_t size() const { return contents.size(); }
  i_t max_size() const { return index_map.size(); }
  bool empty() const { return contents.empty(); }

  std::vector<i_t> contents;
  std::vector<i_t> index_map;
  std::vector<uint8_t> is_member;
};

constexpr double fj_obj_mult_min = 0.25;
constexpr double fj_obj_mult_max = 4.0;

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
                                                     ADD_INSTRUMENTED(h_related_variables),
                                                     ADD_INSTRUMENTED(h_related_variables_offsets),
                                                     ADD_INSTRUMENTED(h_binrow_offsets),
                                                     ADD_INSTRUMENTED(h_binrow_vars),
                                                     ADD_INSTRUMENTED(h_original_ids),
                                                     ADD_INSTRUMENTED(h_reverse_original_ids),
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
                                                     ADD_INSTRUMENTED(h_best_infeasible_assignment),
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
  std::mt19937 rng;
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
  ins_vector<i_t> h_related_variables;
  ins_vector<i_t> h_related_variables_offsets;

  // precompute the binary variables per row for bin 2opt
  ins_vector<i_t> h_binrow_offsets;
  ins_vector<i_t> h_binrow_vars;
  const probing_cache_t<i_t, f_t>* probing_cache{nullptr};
  // Probing cache keys are pre-trivial-presolve variable ids; these translate to and from them
  ins_vector<i_t> h_original_ids;
  ins_vector<i_t> h_reverse_original_ids;

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
  // Lower bound h_objective_weight decays to, so a lane seeded with objective pressure keeps it.
  f_t seed_objective_weight{0};
  // Mean absolute nonzero objective coefficient; the unit of the objective score term.
  f_t obj_magnitude{1};
  f_t h_incumbent_objective;
  f_t h_best_objective;
  i_t last_feasible_entrance_iter{0};
  i_t iterations;
  host_contiguous_set_t<i_t> violated_constraints;
  host_contiguous_set_t<i_t> satisfied_constraints;
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

  // CSR nnz offset -> (delta, score)
  std::vector<std::pair<f_t, fj_staged_score_t>> cached_mtm_moves;

  // Entry i is live only while cached_mtm_moves_version[i] == h_cstr_version of i's row.
  std::vector<i_t> cached_mtm_moves_version;
  std::vector<i_t> h_cstr_version;

  // CSC (transposed!) nnz-offset-indexed constraint bounds (lb, ub)
  // std::pair<f_t, f_t> better compile down to 16 bytes!! GCC do your job!
  ins_vector<std::pair<f_t, f_t>> cached_cstr_bounds;

  std::vector<bool> var_bitmap;
  ins_vector<i_t> iter_mtm_vars;

  // Scratch reused by the binary 2-opt search, which runs at every local minimum
  std::vector<i_t> two_opt_target_cstrs;
  std::vector<i_t> two_opt_first_vars;
  std::vector<std::pair<i_t, f_t>> two_opt_partners;
  std::vector<std::pair<i_t, f_t>> two_opt_row_deltas;

  ins_vector<f_t> h_best_infeasible_assignment;
  f_t best_infeasible_severity{std::numeric_limits<f_t>::infinity()};
  f_t checkpoint_severity{std::numeric_limits<f_t>::infinity()};
  i_t iters_since_infeasible_improve{0};
  i_t restores_since_improvement{0};
  i_t max_restores_since_improvement{0};
  int64_t n_checkpoint_restores{0};
  int64_t n_checkpoint_snapshots{0};

  i_t mtm_viol_samples{25};
  i_t mtm_sat_samples{15};
  i_t nnz_samples{50000};
  i_t perturb_interval{100};
  i_t infeasible_restart_window{300};
  i_t infeasible_restart_max_streak{20};
  f_t infeasible_restart_degrade_ratio{1.15};
  f_t infeasible_checkpoint_refresh_ratio{0.99};

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
void cpufj_solve(fj_cpu_climber_t<i_t, f_t>* fj_cpu,
                 f_t in_time_limit      = std::numeric_limits<f_t>::infinity(),
                 double work_unit_limit = std::numeric_limits<double>::infinity());

// Standalone CPUFJ init for running without full fj_t infrastructure (avoids GPU allocations).
// Used for early CPUFJ during presolve.
template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> init_fj_cpu_standalone(
  problem_t<i_t, f_t>& problem,
  solution_t<i_t, f_t>& solution,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings = fj_settings_t{});

// Copies a climber that has already paid the O(nnz) problem construction. Everything the engine
// reads is host-owned, so this needs neither a problem handle nor any GPU work.
template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> init_fj_cpu_clone(
  const fj_cpu_climber_t<i_t, f_t>& tmpl,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings = fj_settings_t{});

// Per-lane behaviour for a CPUFJ portfolio, shared by every caller that races several climbers so
// the composition cannot drift between them.
template <typename i_t, typename f_t>
void apply_lane_diversification(fj_cpu_climber_t<i_t, f_t>& climber, int lane, int64_t base_seed);

// Builds the climber portfolio the standalone benchmark races: how many distinct
// behaviours, what parameters each gets, whether they are randomized or
// specialized. Defined in fj_cpu_portfolio.cpp -- host code, compiled by the host
// compiler, so editing it is markedly cheaper than editing this header. Runs
// inside the measured window.
template <typename i_t, typename f_t>
void build_climber_portfolio(problem_t<i_t, f_t>& problem,
                             solution_t<i_t, f_t>& solution,
                             std::vector<std::atomic<bool>>& preemption_flags,
                             std::vector<std::unique_ptr<fj_cpu_climber_t<i_t, f_t>>>& climbers,
                             int64_t base_seed);

template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> init_fj_cpu_from_optimization_problem(
  const optimization_problem_t<i_t, f_t>& problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings = fj_settings_t{});

}  // namespace cuopt::mathematical_optimization::mip
