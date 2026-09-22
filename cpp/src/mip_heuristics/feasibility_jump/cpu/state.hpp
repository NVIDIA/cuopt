/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <cuopt/error.hpp>
#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <cuopt/mathematical_optimization/optimization_problem_interface.hpp>
#include <math_optimization/tic_toc.hpp>
#include <mip_heuristics/feasibility_jump/cpu/tuning.hpp>
#include <mip_heuristics/feasibility_jump/fj_cpu_binary.cuh>
#include <mip_heuristics/feasibility_jump/fj_types.hpp>
#include <utilities/logger.hpp>
#include <utilities/macros.cuh>
#include <utilities/memory_instrumentation.hpp>
#include <utilities/pcgenerator.hpp>
#include <utilities/producer_sync.hpp>
#include <utilities/type_2.hpp>

namespace cuopt::mathematical_optimization::simplex {
template <typename i_t, typename f_t>
struct user_problem_t;
}

namespace cuopt::mathematical_optimization::mip {

// The default for one search knob, overridable through the environment so switching it costs a
// rerun rather than a rebuild. Read when the constant it initialises is constructed, so an override
// exported after the process has started has no effect. A value that does not parse in full leaves
// the default in place.
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

// Best feasible assignment found by any lane of one portfolio. A lane publishes its own
// improvements and adopts a better one when it perturbs, so a lane that has stalled resumes from
// the portfolio's progress instead of its own. Lanes run concurrently, so which lane observes
// which incumbent depends on scheduling: a portfolio that shares is not run-to-run reproducible.
template <typename i_t, typename f_t>
struct fj_cpu_shared_incumbent_t {
  // True when the candidate beat the shared best, in which case it was stored.
  bool publish(f_t candidate_objective,
               f_t candidate_user_objective,
               const std::vector<f_t>& candidate)
  {
    if (!(candidate_objective < objective.load(std::memory_order_relaxed))) return false;
    std::lock_guard<std::mutex> lock(guard);
    if (!(candidate_objective < objective.load(std::memory_order_relaxed))) return false;
    assignment = candidate;
    objective.store(candidate_objective, std::memory_order_relaxed);
    CUOPT_LOG_DEBUG("New portfolio best found: %.17g:", candidate_user_objective);
    return true;
  }

  // True when the shared best beat local_objective, in which case it was copied into destination.
  bool adopt(f_t local_objective, std::vector<f_t>& destination, f_t* adopted_objective = nullptr)
  {
    if (!(objective.load(std::memory_order_relaxed) < local_objective)) return false;
    std::lock_guard<std::mutex> lock(guard);
    const f_t shared_objective = objective.load(std::memory_order_relaxed);
    if (!(shared_objective < local_objective)) return false;
    cuopt_assert(assignment.size() == destination.size(), "shared incumbent size mismatch");
    destination = assignment;
    if (adopted_objective != nullptr) *adopted_objective = shared_objective;
    return true;
  }

  std::mutex guard;
  std::vector<f_t> assignment;
  std::atomic<f_t> objective{std::numeric_limits<f_t>::infinity()};
};

// The problem as given: two-sided rows, original column space. Written once during climber
// construction and read-only from then on, so every lane shares one copy rather than carrying its
// own.
template <typename i_t, typename f_t>
struct fj_cpu_problem_t {
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances;
  i_t n_variables{0};
  i_t n_constraints{0};
  i_t nnz{0};
  f_t objective_scaling_factor{1};
  f_t objective_offset{0};
  f_t obj_magnitude{1};
  i_t max_var_degree{0};
  double avg_var_degree{0.0};
  double equality_fraction{0.0};

  std::vector<f_t> h_obj_coeffs;
  std::vector<var_t> h_var_types;
  std::vector<i_t> h_objective_vars;
  std::vector<i_t> h_related_variables;
  std::vector<i_t> h_related_variables_offsets;
  const probing_cache_t<i_t, f_t>* probing_cache{nullptr};
  std::vector<i_t> h_original_ids;
  std::vector<i_t> h_reverse_original_ids;

  std::vector<f_t> coefficients;
  std::vector<i_t> offsets;
  std::vector<i_t> variables;
  std::vector<f_t> reverse_coefficients;
  std::vector<i_t> reverse_constraints;
  std::vector<i_t> reverse_offsets;
  std::vector<f_t> cstr_lb;
  std::vector<f_t> cstr_ub;

  // Host snapshot used by the optional LP-based setup phases. Keeping it with the immutable CPU
  // model prevents the host search engine from reaching back into the GPU-backed problem_t.
  std::shared_ptr<const simplex::user_problem_t<i_t, f_t>> host_lp;

  // Members of uniform-coefficient binary equality rows. Exchanging opposite-valued members
  // preserves the defining equality and provides a structural neighbourhood at local minima.
  std::vector<i_t> card_row_offsets;
  std::vector<i_t> card_variables;
  // Right-hand-side cardinality of each recognized group (sum(x) = k).
  std::vector<i_t> card_cardinalities;
  // Unique cardinality group containing each variable; -1 means none and -2 means ambiguous.
  std::vector<i_t> card_group_of_variable;

  bool is_integer(f_t value) const
  {
    return std::abs(std::round(value) - value) <= tolerances.integrality_tolerance;
  }

  bool integer_equal(f_t lhs, f_t rhs) const
  {
    return std::abs(lhs - rhs) <= tolerances.integrality_tolerance;
  }
};

// Variable domains and every index derived from them are lane-local. Bound propagation narrows
// these during setup, while the structural problem shared by the portfolio remains immutable.
template <typename i_t, typename f_t>
struct fj_domains_t {
  i_t n_integer_vars{0};
  i_t n_binary_vars{0};
  ins_vector<typename type_2<f_t>::type> h_var_bounds;
  ins_vector<i_t> h_is_binary_variable;
  ins_vector<i_t> h_binary_indices;
};

template <typename i_t>
struct fj_tabu_t {
  ins_vector<i_t> h_tabu_nodec_until;
  ins_vector<i_t> h_tabu_noinc_until;
  ins_vector<i_t> h_tabu_lastdec;
  ins_vector<i_t> h_tabu_lastinc;
};

template <typename i_t, typename f_t>
struct fj_weights_t {
  ins_vector<f_t> h_cstr_left_weights;
  ins_vector<f_t> h_cstr_right_weights;
  f_t max_weight;
  f_t h_objective_weight;
};

template <typename i_t, typename f_t>
struct fj_move_cache_t {
  std::vector<int64_t> flip_move_stamp;
  int64_t flip_move_epoch{1};
  std::vector<std::pair<f_t, fj_staged_score_t>> cached_mtm_moves;
  std::vector<i_t> cached_mtm_moves_version;
};

template <typename i_t, typename f_t>
struct fj_pair_scratch_t {
  std::vector<std::pair<i_t, f_t>> two_opt_row_deltas;
};

template <typename i_t>
struct fj_epigraph_t {
  std::vector<int8_t> epigraph_push;
  std::vector<i_t> epigraph_vars;
};

template <typename i_t, typename f_t>
struct fj_checkpoint_t {
  ins_vector<f_t> h_best_infeasible_assignment;
  f_t best_infeasible_severity{std::numeric_limits<f_t>::infinity()};
  f_t checkpoint_severity{std::numeric_limits<f_t>::infinity()};
  i_t iters_since_infeasible_improve{0};
  i_t restores_since_improvement{0};
};

template <typename i_t, typename f_t>
struct fj_search_rows_t {
  struct row_state_t {
    f_t slack;
    f_t weight;
  };
  static_assert(sizeof(row_state_t) == 2 * sizeof(f_t));
  f_t row_tolerance{0};
  i_t n_rows{0};
  ins_vector<row_state_t> h_row_state;
  ins_vector<uint8_t> h_row_is_integral;
  ins_vector<f_t> h_slack_sumcomp;
  ins_vector<f_t> h_bound;
  ins_vector<i_t> h_offsets;
  ins_vector<i_t> h_variables;
  ins_vector<f_t> h_coefficients;
  ins_vector<i_t> h_reverse_offsets;
  ins_vector<i_t> h_reverse_constraints;
  ins_vector<f_t> h_reverse_coefficients;
  std::vector<i_t> h_cstr_version;

  row_state_t* row_state() { return h_row_state.data(); }
  const row_state_t* row_state() const { return h_row_state.data(); }
  bool one_sided() const { return n_rows > 0; }

  std::pair<i_t, i_t> range_for_variable(i_t var_idx) const
  {
    cuopt_assert(var_idx >= 0 && var_idx < static_cast<i_t>(h_reverse_offsets.size()) - 1,
                 "Variable should be within the range");
    return std::make_pair(h_reverse_offsets[var_idx], h_reverse_offsets[var_idx + 1]);
  }

  std::pair<i_t, i_t> range_for_row(i_t row) const
  {
    cuopt_assert(row >= 0 && row < n_rows, "row out of range");
    return std::make_pair(h_offsets[row], h_offsets[row + 1]);
  }
};

template <typename i_t, typename f_t>
struct fj_search_state_t {
  cuopt::pcgenerator_t rng;
  ins_vector<f_t> h_lhs;
  ins_vector<f_t> h_lhs_sumcomp;
  ins_vector<f_t> h_assignment;
  ins_vector<f_t> h_best_assignment;
  f_t h_incumbent_objective;
  f_t h_objective_sumcomp{0};
  f_t h_best_objective;
  f_t h_last_reported_objective{std::numeric_limits<f_t>::max()};
  i_t iterations{0};
  host_contiguous_set_t<i_t> violated_constraints;
  host_contiguous_set_t<i_t> satisfied_constraints;
  bool feasible_found{false};
  bool trigger_early_lhs_recomputation{false};
  f_t total_violations{0};
  f_t total_violations_sumcomp{0};
  i_t perturb_streak{0};
  i_t iterations_since_best{0};
};

template <typename i_t, typename f_t>
struct fj_batching_t {
  i_t n_colors{0};
  std::vector<i_t> h_var_color;
  std::vector<fj_staged_score_t> h_var_best_score;
  std::vector<f_t> h_var_best_delta;
  std::vector<int64_t> h_var_best_stamp;
  std::vector<int64_t> h_var_best_rowsum;
  int64_t var_best_epoch{1};
  std::vector<std::vector<i_t>> h_color_candidates;
  std::vector<int64_t> h_color_epoch;
  std::vector<int64_t> h_var_bucket_stamp;
};

template <typename i_t, typename f_t>
struct fj_bin_bridge_t {
  struct bin_eliminated_row_t {
    i_t row;
    f_t rhs;
    std::vector<i_t> positive, negative, all;
    std::vector<f_t> positive_coeff, negative_coeff;
  };
  std::vector<bin_eliminated_row_t> bin_eliminated_rows;
  // (row, column) substitutions that retain the singleton's bounds as row bounds.
  std::vector<std::pair<i_t, i_t>> bin_singletons;
  std::vector<uint8_t> bin_ignore_row, bin_ignore_var;
  bool has_bin_elimination{false};
};

template <typename i_t, typename f_t>
struct fj_lane_policy_t {
  fj_settings_t settings;
  fj_cpu_hyper_parameters_t hp;
  f_t seed_objective_weight{0};
  bool use_move_batching{false};
  i_t mtm_viol_samples{25};
  i_t mtm_sat_samples{15};
  i_t nnz_samples{50000};
  i_t perturb_interval{100};
  i_t perturb_vars{2};
  bool use_lp_start{false};
  bool lp_start_feasibility_objective{false};
  bool use_deep_lp_pump{false};
  bool use_integer_bit_encoding{true};
  bool use_lp_polish{false};
  bool use_precedence_start{false};
  bool use_affine_equality_start{false};
  bool use_unit_commitment_start{false};
  bool use_fixed_charge_network_start{false};
  bool use_pmedian_start{false};
  bool use_bound_prop{false};
  bool low_latency{false};
  bool use_weight_donation{false};
  bool degree_balance_mtm{false};
  bool use_cardinality_exchange{false};
  bool use_directed_infeasible_kick{false};
  bool use_compound_repair{false};
  bool use_equality_substitution{false};
  bool suppress_incumbent_log{false};
  bool use_multiplicative_weights{false};
  f_t saps_multiplier{(f_t)1.3};
  i_t infeasible_kick_interval{0};
  i_t infeasible_kick_vars{4};
  i_t infeasible_restart_window{200};
  i_t infeasible_restart_max_streak{20};
  f_t infeasible_restart_degrade_ratio{1.15};
  f_t infeasible_checkpoint_refresh_ratio{0.99};
};

template <typename i_t>
struct fj_stats_t {
  int64_t n_batch_attempts{0};
  int64_t n_batched_moves{0};
  std::vector<int64_t> batch_size_hist;
  int64_t max_batch_size{0};
  double t_start{0};
  double t_bound_prop{0};
  double t_lp_start{0};
  double t_lp_relaxation{0};
  double t_coloring{0};
  double t_features{0};
  double t_init_lhs{0};
  fj_bin_setup_times_t bin_setup;
  int64_t hit_count{0};
  int64_t miss_count{0};
  int64_t n_moves_applied{0};
  int64_t apply_move_nnz{0};
  int64_t n_mtm_calls{0};
  int64_t mtm_row_entries{0};
  int64_t mtm_entries_capped{0};
  int64_t n_compute_score_calls{0};
  int64_t compute_score_nnz{0};
  int64_t n_version_bumps_apply{0};
  int64_t n_version_bumps_weights{0};
  int64_t n_mtm_cache_invalidations{0};
  int64_t n_lhs_recompute_total{0};
  int64_t n_lhs_recompute_periodic{0};
  int64_t n_lhs_recompute_bigval{0};
  int64_t n_lhs_recompute_perturb{0};
  int64_t n_lhs_recompute_restart{0};
  i_t lhs_refresh_period_used{0};
  int64_t n_epigraph_projections{0};
  i_t max_restores_since_improvement{0};
  int64_t n_checkpoint_restores{0};
  int64_t n_checkpoint_snapshots{0};
  i_t nnz_processed_window{0};
};

template <typename i_t, typename f_t>
struct fj_runtime_t {
  explicit fj_runtime_t(std::atomic<bool>& flag) : preemption_flag(flag) {}
  i_t log_interval{1000};
  i_t diversity_callback_interval{3000};
  std::function<void(f_t, const std::vector<f_t>&, double)> improvement_callback{nullptr};
  std::function<void(f_t, const std::vector<f_t>&)> diversity_callback{nullptr};
  std::string log_prefix;
  std::shared_ptr<fj_cpu_shared_incumbent_t<i_t, f_t>> shared_incumbent;
  std::atomic<double> work_units_elapsed{0.0};
  double work_unit_bias{1.5};
  producer_sync_t* producer_sync{nullptr};
  std::atomic<bool> halted{false};
  instrumentation_aggregator_t memory_aggregator;
  std::atomic<bool>& preemption_flag;
};

template <typename i_t, typename f_t>
struct fj_cpu_climber_t : fj_tabu_t<i_t>,
                          fj_weights_t<i_t, f_t>,
                          fj_domains_t<i_t, f_t>,
                          fj_move_cache_t<i_t, f_t>,
                          fj_pair_scratch_t<i_t, f_t>,
                          fj_epigraph_t<i_t>,
                          fj_checkpoint_t<i_t, f_t>,
                          fj_search_rows_t<i_t, f_t>,
                          fj_search_state_t<i_t, f_t>,
                          fj_batching_t<i_t, f_t>,
                          fj_bin_bridge_t<i_t, f_t>,
                          fj_lane_policy_t<i_t, f_t>,
                          fj_stats_t<i_t>,
                          fj_runtime_t<i_t, f_t> {
  fj_cpu_climber_t(std::atomic<bool>& preemption_flag) : fj_runtime_t<i_t, f_t>(preemption_flag)
  {
#define ADD_INSTRUMENTED(var) \
  std::make_pair(#var, std::ref(static_cast<memory_instrumentation_base_t&>(this->var)))

    // Initialize memory aggregator with all ins_vector members
    this->memory_aggregator =
      instrumentation_aggregator_t{ADD_INSTRUMENTED(h_tabu_nodec_until),
                                   ADD_INSTRUMENTED(h_tabu_noinc_until),
                                   ADD_INSTRUMENTED(h_tabu_lastdec),
                                   ADD_INSTRUMENTED(h_tabu_lastinc),
                                   ADD_INSTRUMENTED(h_lhs),
                                   ADD_INSTRUMENTED(h_lhs_sumcomp),
                                   ADD_INSTRUMENTED(h_cstr_left_weights),
                                   ADD_INSTRUMENTED(h_cstr_right_weights),
                                   ADD_INSTRUMENTED(h_var_bounds),
                                   ADD_INSTRUMENTED(h_is_binary_variable),
                                   ADD_INSTRUMENTED(h_binary_indices),
                                   ADD_INSTRUMENTED(h_assignment),
                                   ADD_INSTRUMENTED(h_best_assignment),
                                   ADD_INSTRUMENTED(h_best_infeasible_assignment),
                                   ADD_INSTRUMENTED(h_row_state),
                                   ADD_INSTRUMENTED(h_row_is_integral),
                                   ADD_INSTRUMENTED(h_slack_sumcomp),
                                   ADD_INSTRUMENTED(h_bound),
                                   ADD_INSTRUMENTED(h_offsets),
                                   ADD_INSTRUMENTED(h_variables),
                                   ADD_INSTRUMENTED(h_coefficients),
                                   ADD_INSTRUMENTED(h_reverse_offsets),
                                   ADD_INSTRUMENTED(h_reverse_constraints),
                                   ADD_INSTRUMENTED(h_reverse_coefficients)};

#undef ADD_INSTRUMENTED
  }
  fj_cpu_climber_t(const fj_cpu_climber_t<i_t, f_t>& other)                      = delete;
  fj_cpu_climber_t<i_t, f_t>& operator=(const fj_cpu_climber_t<i_t, f_t>& other) = delete;

  fj_cpu_climber_t(fj_cpu_climber_t<i_t, f_t>&& other)                      = default;
  fj_cpu_climber_t<i_t, f_t>& operator=(fj_cpu_climber_t<i_t, f_t>&& other) = default;

  f_t get_user_objective(f_t solver_objective) const
  {
    cuopt_assert(std::isfinite(problem->objective_scaling_factor) &&
                   problem->objective_scaling_factor != f_t{0},
                 "invalid objective scaling factor");
    return problem->objective_scaling_factor * (solver_objective + problem->objective_offset);
  }

  bool check_variable_within_bounds(i_t variable, f_t value) const
  {
    const auto bounds = this->h_var_bounds[variable];
    const f_t tol     = problem->tolerances.integrality_tolerance;
    return value <= get_upper(bounds) + tol && value >= get_lower(bounds) - tol;
  }

  bool move_numerically_stable(f_t old_value, f_t new_value, f_t infeasibility, f_t total) const
  {
    return std::abs(new_value - old_value) < 1e6 && std::abs(new_value) < 1e20 &&
           std::abs(total - infeasibility) < 1e20;
  }

  f_t excess_score(i_t row, f_t lhs, f_t lower, f_t upper) const
  {
    const f_t right = upper - lhs;
    if (right < 0) return right;
    const f_t left = lhs - lower;
    return left < 0 ? left : f_t{0};
  }

  f_t breakthrough_value(i_t variable) const
  {
    const f_t coefficient = problem->h_obj_coeffs[variable];
    const auto bounds     = this->h_var_bounds[variable];
    const f_t old_value   = this->h_assignment[variable];
    const f_t excess      = this->h_best_objective - this->h_incumbent_objective;
    cuopt_assert(std::isfinite(excess) && excess < 0, "invalid breakthrough state");
    f_t value = old_value + excess / coefficient;
    if (problem->h_var_types[variable] == var_t::INTEGER) {
      value = coefficient > 0 ? std::floor(value + problem->tolerances.integrality_tolerance)
                              : std::ceil(value - problem->tolerances.integrality_tolerance);
    }
    if (!check_variable_within_bounds(variable, value))
      value = coefficient > 0 ? get_lower(bounds) : get_upper(bounds);
    cuopt_assert(std::isfinite(value), "breakthrough move left the representable range");
    return value;
  }

  // Shared across every lane and frozen before the first clone is created; see fj_cpu_problem_t.
  std::shared_ptr<const fj_cpu_problem_t<i_t, f_t>> problem;
};

template <typename i_t, typename f_t>
void cpufj_solve(fj_cpu_climber_t<i_t, f_t>* fj_cpu,
                 f_t in_time_limit      = std::numeric_limits<f_t>::infinity(),
                 double work_unit_limit = std::numeric_limits<double>::infinity());

// Copies a climber that has already paid the O(nnz) problem construction. Everything the engine
// reads is host-owned, so this needs neither a problem handle nor any GPU work.
template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> init_fj_cpu_clone(
  const fj_cpu_climber_t<i_t, f_t>& tmpl,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings = fj_settings_t{});

}  // namespace cuopt::mathematical_optimization::mip
