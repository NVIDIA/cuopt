/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/mip_constants.hpp>

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/user_problem.hpp>
#include <math_optimization/tic_toc.hpp>

#include "feasibility_jump.cuh"
#include "feasibility_jump_impl_common.cuh"
#include "fj_cpu.cuh"
#include "fj_cpu_binary.cuh"
#include "fj_cpu_worker.cuh"

#include <mip_heuristics/presolve/probing_cache.cuh>

#include <utilities/pcgenerator.hpp>
#include <utilities/seed_generator.cuh>

#include <raft/core/nvtx.hpp>

#include <thrust/iterator/transform_iterator.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <vector>

#define CPUFJ_TIMING_TRACE 0

// Define CPUFJ_NVTX_RANGES to enable detailed NVTX profiling ranges
#ifdef CPUFJ_NVTX_RANGES
#define CPUFJ_NVTX_RANGE(name)        raft::common::nvtx::range CPUFJ_NVTX_UNIQUE_NAME(nvtx_scope_)(name)
#define CPUFJ_NVTX_UNIQUE_NAME(base)  CPUFJ_NVTX_CONCAT(base, __LINE__)
#define CPUFJ_NVTX_CONCAT(a, b)       CPUFJ_NVTX_CONCAT_INNER(a, b)
#define CPUFJ_NVTX_CONCAT_INNER(a, b) a##b
#else
#define CPUFJ_NVTX_RANGE(name) ((void)0)
#endif

namespace cuopt::mathematical_optimization::mip {

using simplex::lp_problem_t;
using simplex::simplex_solver_settings_t;
using simplex::variable_type_t;

template <typename i_t, typename f_t>
void finalize_fj_cpu_host_initialization(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu,
  i_t n_variables,
  i_t n_constraints,
  i_t n_integer_vars,
  i_t nnz,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances);

template <typename i_t, typename f_t>
static void finalize_fj_cpu_host_initialization_from_template(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu,
  const fj_cpu_climber_t<i_t, f_t>& tmpl,
  i_t n_variables,
  i_t n_constraints,
  i_t n_integer_vars,
  i_t nnz,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances);

template <typename i_t, typename f_t, typename ArrayType>
thrust::tuple<f_t, f_t> get_mtm_for_bound(const typename fj_t<i_t, f_t>::climber_data_t::view_t& fj,
                                          i_t var_idx,
                                          i_t cstr_idx,
                                          f_t cstr_coeff,
                                          f_t bound,
                                          f_t sign,
                                          const ArrayType& assignment,
                                          const ArrayType& lhs_vector)
{
  f_t delta_ij = 0;
  f_t slack    = 0;
  f_t old_val  = assignment[var_idx];

  f_t lhs = lhs_vector[cstr_idx] * sign;
  f_t rhs = bound * sign;
  slack   = rhs - lhs;  // bound might be infinite. let the caller handle this case

  delta_ij = slack / (cstr_coeff * sign);

  return {delta_ij, slack};
}

template <typename i_t, typename f_t, MTMMoveType move_type, typename ArrayType>
thrust::tuple<f_t, f_t, f_t, f_t> get_mtm_for_constraint(
  const typename fj_t<i_t, f_t>::climber_data_t::view_t& fj,
  i_t var_idx,
  i_t cstr_idx,
  f_t cstr_coeff,
  f_t c_lb,
  f_t c_ub,
  const ArrayType& assignment,
  const ArrayType& lhs_vector)
{
  f_t sign     = -1;
  f_t delta_ij = 0;
  f_t slack    = 0;

  f_t cstr_tolerance = fj.get_corrected_tolerance(cstr_idx, c_lb, c_ub);

  f_t old_val = assignment[var_idx];

  // process each bound as two separate constraints
  f_t bounds[2] = {c_lb, c_ub};
  cuopt_assert(isfinite(bounds[0]) || isfinite(bounds[1]), "bounds are not finite");

  for (i_t bound_idx = 0; bound_idx < 2; ++bound_idx) {
    if (!isfinite(bounds[bound_idx])) continue;

    // factor to correct the lhs/rhs to turn a lb <= lhs <= ub constraint into
    // two virtual constraints lhs <= ub and -lhs <= -lb
    sign    = bound_idx == 0 ? -1 : 1;
    f_t lhs = lhs_vector[cstr_idx] * sign;
    f_t rhs = bounds[bound_idx] * sign;
    slack   = rhs - lhs;

    // skip constraints that are violated/satisfied based on the MTM move type
    bool violated = slack < -cstr_tolerance;
    if (move_type == MTMMoveType::FJ_MTM_VIOLATED ? !violated : violated) continue;

    f_t new_val = old_val;

    delta_ij = slack / (cstr_coeff * sign);
    break;
  }

  return {delta_ij, sign, slack, cstr_tolerance};
}

template <typename i_t, typename f_t>
std::pair<f_t, f_t> feas_score_constraint(const typename fj_t<i_t, f_t>::climber_data_t::view_t& fj,
                                          f_t delta,
                                          i_t cstr_idx,
                                          f_t cstr_coeff,
                                          f_t c_lb,
                                          f_t c_ub,
                                          f_t current_lhs,
                                          f_t left_weight,
                                          f_t right_weight)
{
  cuopt_assert(isfinite(delta), "invalid delta");
  cuopt_assert(cstr_coeff != 0 && isfinite(cstr_coeff), "invalid coefficient");

  f_t base_feas    = 0;
  f_t bonus_robust = 0;

  f_t bounds[2] = {c_lb, c_ub};
  cuopt_assert(isfinite(c_lb) || isfinite(c_ub), "no range");

  // Independent of bound_idx.
  const f_t moved_lhs      = current_lhs + cstr_coeff * delta;
  const f_t cstr_tolerance = fj.get_corrected_tolerance(cstr_idx, c_lb, c_ub);
  const bool old_viol = fj.excess_score(cstr_idx, current_lhs, c_lb, c_ub) < -cstr_tolerance;
  const bool new_viol = fj.excess_score(cstr_idx, moved_lhs, c_lb, c_ub) < -cstr_tolerance;

  for (i_t bound_idx = 0; bound_idx < 2; ++bound_idx) {
    if (!isfinite(bounds[bound_idx])) continue;

    // factor to correct the lhs/rhs to turn a lb <= lhs <= ub constraint into
    // two virtual leq constraints "lhs <= ub" and "-lhs <= -lb" in order to match
    // the convention of the paper

    // TODO: broadcast left/right weights to a csr_offset-indexed table? local minimums
    // usually occur on a rarer basis (around 50 iteratiosn to 1 local minimum)
    // likely unreasonable and overkill however
    f_t cstr_weight = bound_idx == 0 ? left_weight : right_weight;
    f_t sign        = bound_idx == 0 ? -1 : 1;
    f_t rhs         = bounds[bound_idx] * sign;
    f_t old_lhs     = current_lhs * sign;
    f_t new_lhs     = moved_lhs * sign;
    f_t old_slack   = rhs - old_lhs;
    f_t new_slack   = rhs - new_lhs;

    cuopt_assert(isfinite(cstr_weight), "invalid weight");
    cuopt_assert(cstr_weight >= 0, "invalid weight");
    cuopt_assert(isfinite(old_lhs), "");
    cuopt_assert(isfinite(new_lhs), "");
    cuopt_assert(isfinite(old_slack) && isfinite(new_slack), "");

    bool old_sat = old_lhs < rhs + cstr_tolerance;
    bool new_sat = new_lhs < rhs + cstr_tolerance;

    // equality
    if (fj.pb.integer_equal(c_lb, c_ub)) {
      if (!old_viol) cuopt_assert(old_sat == !old_viol, "");
      if (!new_viol) cuopt_assert(new_sat == !new_viol, "");
    }

    // if it would feasibilize this constraint
    if (!old_sat && new_sat) {
      cuopt_assert(old_viol, "");
      base_feas += cstr_weight;
    }
    // would cause this constraint to be violated
    else if (old_sat && !new_sat) {
      cuopt_assert(new_viol, "");
      base_feas -= cstr_weight;
    }
    // simple improvement
    else if (!old_sat && !new_sat && old_lhs > new_lhs) {
      cuopt_assert(old_viol && new_viol, "");
      base_feas += (i_t)(cstr_weight * fj.settings->parameters.excess_improvement_weight);
    }
    // simple worsening
    else if (!old_sat && !new_sat && old_lhs < new_lhs) {
      cuopt_assert(old_viol && new_viol, "");
      base_feas -= (i_t)(cstr_weight * fj.settings->parameters.excess_improvement_weight);
    }

    // robustness score bonus if this would leave some strick slack
    bool old_stable = old_lhs < rhs - cstr_tolerance;
    bool new_stable = new_lhs < rhs - cstr_tolerance;
    if (!old_stable && new_stable) {
      bonus_robust += cstr_weight;
    } else if (old_stable && !new_stable) {
      bonus_robust -= cstr_weight;
    }
  }

  return {base_feas, bonus_robust};
}

static constexpr double BIGVAL_THRESHOLD = 1e20;

template <typename i_t, typename f_t>
class timing_raii_t {
 public:
  timing_raii_t(std::vector<double>& times_vec)
    : times_vec_(times_vec), start_time_(std::chrono::high_resolution_clock::now())
  {
  }

  ~timing_raii_t()
  {
    // vector::push_back can throw bad_alloc; the catch-all keeps the destructor
    // exception-free. Losing one timing sample under OOM is acceptable.
    // fprintf to stderr is allocation-free and cannot throw; using the project
    // logger here would risk a secondary bad_alloc that would escape the
    // destructor and re-introduce std::terminate.
    try {
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time_);
      times_vec_.push_back(duration.count());
    } catch (const std::exception& e) {
      std::fprintf(stderr, "timing_raii_t destructor: failed to record sample (%s).\n", e.what());
    } catch (...) {
      std::fprintf(stderr,
                   "timing_raii_t destructor: failed to record sample (unknown exception).\n");
    }
  }

 private:
  std::vector<double>& times_vec_;
  std::chrono::high_resolution_clock::time_point start_time_;
};

template <typename i_t, typename f_t>
static void print_timing_stats(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  auto compute_avg_and_total = [](const std::vector<double>& times) -> std::pair<double, double> {
    if (times.empty()) return {0.0, 0.0};
    double sum = 0.0;
    for (double time : times)
      sum += time;
    return {sum / times.size(), sum};
  };

  auto [lift_avg, lift_total]       = compute_avg_and_total(fj_cpu.find_lift_move_times);
  auto [viol_avg, viol_total]       = compute_avg_and_total(fj_cpu.find_mtm_move_viol_times);
  auto [sat_avg, sat_total]         = compute_avg_and_total(fj_cpu.find_mtm_move_sat_times);
  auto [apply_avg, apply_total]     = compute_avg_and_total(fj_cpu.apply_move_times);
  auto [weights_avg, weights_total] = compute_avg_and_total(fj_cpu.update_weights_times);
  auto [compute_score_avg, compute_score_total] = compute_avg_and_total(fj_cpu.compute_score_times);
  CUOPT_LOG_DEBUG("=== Timing Statistics (Iteration %d) ===", fj_cpu.iterations);
  CUOPT_LOG_DEBUG("find_lift_move:      avg=%.6f ms, total=%.6f ms, calls=%zu",
                  lift_avg * 1000.0,
                  lift_total * 1000.0,
                  fj_cpu.find_lift_move_times.size());
  CUOPT_LOG_DEBUG("find_mtm_move_viol:  avg=%.6f ms, total=%.6f ms, calls=%zu",
                  viol_avg * 1000.0,
                  viol_total * 1000.0,
                  fj_cpu.find_mtm_move_viol_times.size());
  CUOPT_LOG_DEBUG("find_mtm_move_sat:   avg=%.6f ms, total=%.6f ms, calls=%zu",
                  sat_avg * 1000.0,
                  sat_total * 1000.0,
                  fj_cpu.find_mtm_move_sat_times.size());
  CUOPT_LOG_DEBUG("apply_move:          avg=%.6f ms, total=%.6f ms, calls=%zu",
                  apply_avg * 1000.0,
                  apply_total * 1000.0,
                  fj_cpu.apply_move_times.size());
  CUOPT_LOG_DEBUG("update_weights:      avg=%.6f ms, total=%.6f ms, calls=%zu",
                  weights_avg * 1000.0,
                  weights_total * 1000.0,
                  fj_cpu.update_weights_times.size());
  CUOPT_LOG_DEBUG("compute_score:       avg=%.6f ms, total=%.6f ms, calls=%zu",
                  compute_score_avg * 1000.0,
                  compute_score_total * 1000.0,
                  fj_cpu.compute_score_times.size());
  CUOPT_LOG_DEBUG("cache hit percentage: %.2f%%",
                  (double)fj_cpu.hit_count / (fj_cpu.hit_count + fj_cpu.miss_count) * 100.0);
  CUOPT_LOG_DEBUG("bin  candidate move hit percentage: %.2f%%",
                  (double)fj_cpu.candidate_move_hits[0] /
                    (fj_cpu.candidate_move_hits[0] + fj_cpu.candidate_move_misses[0]) * 100.0);
  CUOPT_LOG_DEBUG("int  candidate move hit percentage: %.2f%%",
                  (double)fj_cpu.candidate_move_hits[1] /
                    (fj_cpu.candidate_move_hits[1] + fj_cpu.candidate_move_misses[1]) * 100.0);
  CUOPT_LOG_DEBUG("cont candidate move hit percentage: %.2f%%",
                  (double)fj_cpu.candidate_move_hits[2] /
                    (fj_cpu.candidate_move_hits[2] + fj_cpu.candidate_move_misses[2]) * 100.0);
  CUOPT_LOG_DEBUG("========================================");
}

template <typename i_t, typename f_t>
static void precompute_problem_features(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  fj_cpu.n_binary_vars  = 0;
  fj_cpu.n_integer_vars = 0;
  for (i_t i = 0; i < (i_t)fj_cpu.h_is_binary_variable.size(); i++) {
    if (fj_cpu.h_is_binary_variable[i]) {
      fj_cpu.n_binary_vars++;
    } else if (fj_cpu.h_var_types[i] == var_t::INTEGER) {
      fj_cpu.n_integer_vars++;
    }
  }

  i_t total_nnz = fj_cpu.h_reverse_offsets.back();
  i_t n_vars    = fj_cpu.h_reverse_offsets.size() - 1;
  i_t n_cstrs   = fj_cpu.h_offsets.size() - 1;

  fj_cpu.avg_var_degree = (double)total_nnz / n_vars;

  fj_cpu.max_var_degree = 0;
  std::vector<i_t> var_degrees(n_vars);
  for (i_t i = 0; i < n_vars; i++) {
    i_t degree            = fj_cpu.h_reverse_offsets[i + 1] - fj_cpu.h_reverse_offsets[i];
    var_degrees[i]        = degree;
    fj_cpu.max_var_degree = std::max(fj_cpu.max_var_degree, degree);
  }

  double var_deg_variance = 0.0;
  for (i_t i = 0; i < n_vars; i++) {
    double diff = var_degrees[i] - fj_cpu.avg_var_degree;
    var_deg_variance += diff * diff;
  }
  var_deg_variance /= n_vars;
  double var_degree_std = std::sqrt(var_deg_variance);
  fj_cpu.var_degree_cv  = fj_cpu.avg_var_degree > 0 ? var_degree_std / fj_cpu.avg_var_degree : 0.0;

  fj_cpu.avg_cstr_degree = (double)total_nnz / n_cstrs;

  fj_cpu.max_cstr_degree = 0;
  std::vector<i_t> cstr_degrees(n_cstrs);
  for (i_t i = 0; i < n_cstrs; i++) {
    i_t degree             = fj_cpu.h_offsets[i + 1] - fj_cpu.h_offsets[i];
    cstr_degrees[i]        = degree;
    fj_cpu.max_cstr_degree = std::max(fj_cpu.max_cstr_degree, degree);
  }

  double cstr_deg_variance = 0.0;
  for (i_t i = 0; i < n_cstrs; i++) {
    double diff = cstr_degrees[i] - fj_cpu.avg_cstr_degree;
    cstr_deg_variance += diff * diff;
  }
  cstr_deg_variance /= n_cstrs;
  double cstr_degree_std = std::sqrt(cstr_deg_variance);
  fj_cpu.cstr_degree_cv =
    fj_cpu.avg_cstr_degree > 0 ? cstr_degree_std / fj_cpu.avg_cstr_degree : 0.0;

  fj_cpu.problem_density = (double)total_nnz / ((double)n_vars * n_cstrs);
}

template <typename i_t, typename f_t>
static void log_regression_features(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                                    double time_window_ms,
                                    double total_time_ms,
                                    size_t mem_loads_bytes,
                                    size_t mem_stores_bytes)
{
  i_t total_nnz = fj_cpu.h_reverse_offsets.back();
  i_t n_vars    = fj_cpu.h_reverse_offsets.size() - 1;
  i_t n_cstrs   = fj_cpu.h_offsets.size() - 1;

  // Dynamic runtime features
  double violated_ratio = (double)fj_cpu.violated_constraints.size() / n_cstrs;

  // Compute per-iteration metrics
  [[maybe_unused]] double nnz_per_move = 0.0;
  i_t total_moves =
    fj_cpu.n_lift_moves_window + fj_cpu.n_mtm_viol_moves_window + fj_cpu.n_mtm_sat_moves_window;
  if (total_moves > 0) { nnz_per_move = (double)fj_cpu.nnz_processed_window / total_moves; }

  double eval_intensity = (double)fj_cpu.nnz_processed_window / 1000.0;

  // Cache and locality metrics
  i_t cache_hits_window    = fj_cpu.hit_count - fj_cpu.hit_count_window_start;
  i_t cache_misses_window  = fj_cpu.miss_count - fj_cpu.miss_count_window_start;
  i_t total_cache_accesses = cache_hits_window + cache_misses_window;
  double cache_hit_rate =
    total_cache_accesses > 0 ? (double)cache_hits_window / total_cache_accesses : 0.0;

  i_t unique_cstrs = fj_cpu.unique_cstrs_accessed_window.size();
  i_t unique_vars  = fj_cpu.unique_vars_accessed_window.size();

  // Reuse ratios: how many times each constraint/variable was accessed on average
  double cstr_reuse_ratio =
    unique_cstrs > 0 ? (double)fj_cpu.nnz_processed_window / unique_cstrs : 0.0;
  double var_reuse_ratio =
    unique_vars > 0 ? (double)fj_cpu.n_variable_updates_window / unique_vars : 0.0;

  // Working set size estimation (KB)
  // Each constraint: lhs (f_t) + 2 bounds (f_t) + sumcomp (f_t) = 4 * sizeof(f_t)
  // Each variable: assignment (f_t) = 1 * sizeof(f_t)
  i_t working_set_bytes = unique_cstrs * 4 * sizeof(f_t) + unique_vars * sizeof(f_t);
  double working_set_kb = working_set_bytes / 1024.0;

  // Coverage: what fraction of problem is actively touched
  double cstr_coverage = (double)unique_cstrs / n_cstrs;
  double var_coverage  = (double)unique_vars / n_vars;

  double loads_per_iter  = 0.0;
  double stores_per_iter = 0.0;
  double l1_miss         = -1.0;
  double l3_miss         = -1.0;

  // Compute memory statistics
  double mem_loads_mb             = mem_loads_bytes / 1e6;
  double mem_stores_mb            = mem_stores_bytes / 1e6;
  double mem_total_mb             = (mem_loads_bytes + mem_stores_bytes) / 1e6;
  double mem_bandwidth_gb_per_sec = (mem_total_mb / 1000.0) / (time_window_ms / 1000.0);

  // Build per-wrapper memory statistics string
  std::stringstream wrapper_stats;
  auto per_wrapper_stats = fj_cpu.memory_aggregator.collect_per_wrapper();
  for (const auto& [name, loads, stores] : per_wrapper_stats) {
    wrapper_stats << " " << name << "_loads=" << loads << " " << name << "_stores=" << stores;
  }

  fj_cpu.memory_aggregator.flush();

  // Print everything on a single line using precomputed features
  CUOPT_LOG_DEBUG(
    "%sCPUFJ_FEATURES iter=%d time_window=%.2f "
    "n_vars=%d n_cstrs=%d n_bin=%d n_int=%d total_nnz=%d "
    "avg_var_deg=%.2f max_var_deg=%d var_deg_cv=%.4f "
    "avg_cstr_deg=%.2f max_cstr_deg=%d cstr_deg_cv=%.4f "
    "density=%.6f "
    "total_viol=%.4f obj_weight=%.4f max_weight=%.4f "
    "n_locmin=%d iter_since_best=%d feas_found=%d "
    "nnz_proc=%d n_lift=%d n_mtm_viol=%d n_mtm_sat=%d n_var_updates=%d "
    "cache_hit_rate=%.4f unique_cstrs=%d unique_vars=%d "
    "cstr_reuse=%.2f var_reuse=%.2f working_set_kb=%.1f "
    "cstr_coverage=%.4f var_coverage=%.4f "
    "L1_miss=%.2f L3_miss=%.2f loads_per_iter=%.0f stores_per_iter=%.0f "
    "viol_ratio=%.4f nnz_per_move=%.2f eval_intensity=%.2f "
    "mem_loads_mb=%.3f mem_stores_mb=%.3f mem_total_mb=%.3f mem_bandwidth_gb_s=%.3f%s",
    fj_cpu.log_prefix.c_str(),
    fj_cpu.iterations,
    time_window_ms,
    n_vars,
    n_cstrs,
    fj_cpu.n_binary_vars,
    fj_cpu.n_integer_vars,
    total_nnz,
    fj_cpu.avg_var_degree,
    fj_cpu.max_var_degree,
    fj_cpu.var_degree_cv,
    fj_cpu.avg_cstr_degree,
    fj_cpu.max_cstr_degree,
    fj_cpu.cstr_degree_cv,
    fj_cpu.problem_density,
    fj_cpu.total_violations,
    fj_cpu.h_objective_weight,
    fj_cpu.max_weight,
    fj_cpu.n_local_minima_window,
    fj_cpu.iterations_since_best,
    fj_cpu.feasible_found ? 1 : 0,
    fj_cpu.nnz_processed_window,
    fj_cpu.n_lift_moves_window,
    fj_cpu.n_mtm_viol_moves_window,
    fj_cpu.n_mtm_sat_moves_window,
    fj_cpu.n_variable_updates_window,
    cache_hit_rate,
    unique_cstrs,
    unique_vars,
    cstr_reuse_ratio,
    var_reuse_ratio,
    working_set_kb,
    cstr_coverage,
    var_coverage,
    l1_miss,
    l3_miss,
    loads_per_iter,
    stores_per_iter,
    violated_ratio,
    nnz_per_move,
    eval_intensity,
    mem_loads_mb,
    mem_stores_mb,
    mem_total_mb,
    mem_bandwidth_gb_per_sec,
    wrapper_stats.str().c_str());

  // Reset window counters
  fj_cpu.nnz_processed_window      = 0;
  fj_cpu.n_lift_moves_window       = 0;
  fj_cpu.n_mtm_viol_moves_window   = 0;
  fj_cpu.n_mtm_sat_moves_window    = 0;
  fj_cpu.n_variable_updates_window = 0;
  fj_cpu.n_local_minima_window     = 0;
  fj_cpu.prev_best_objective       = fj_cpu.h_best_objective;

  // Reset cache and locality tracking
  fj_cpu.hit_count_window_start  = fj_cpu.hit_count;
  fj_cpu.miss_count_window_start = fj_cpu.miss_count;
  fj_cpu.unique_cstrs_accessed_window.clear();
  fj_cpu.unique_vars_accessed_window.clear();
}

template <typename i_t, typename f_t>
static inline std::pair<i_t, i_t> reverse_range_for_var(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                                                        i_t var_idx)
{
  cuopt_assert(var_idx >= 0 && var_idx < fj_cpu.view.pb.n_variables,
               "Variable should be within the range");
  return std::make_pair(fj_cpu.h_reverse_offsets[var_idx], fj_cpu.h_reverse_offsets[var_idx + 1]);
}

template <typename i_t, typename f_t>
static inline std::pair<i_t, i_t> range_for_constraint(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                                                       i_t cstr_idx)
{
  return std::make_pair(fj_cpu.h_offsets[cstr_idx], fj_cpu.h_offsets[cstr_idx + 1]);
}

template <typename i_t, typename f_t>
static inline bool check_variable_within_bounds(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                                                i_t var_idx,
                                                f_t val)
{
  const f_t int_tol  = fj_cpu.view.pb.tolerances.integrality_tolerance;
  auto bounds        = fj_cpu.h_var_bounds[var_idx].get();
  bool within_bounds = val <= (get_upper(bounds) + int_tol) && val >= (get_lower(bounds) - int_tol);
  return within_bounds;
}

template <typename i_t, typename f_t>
static inline bool is_integer_var(fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t var_idx)
{
  return var_t::INTEGER == fj_cpu.h_var_types[var_idx];
}

template <typename i_t, typename f_t>
static inline bool tabu_check(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                              i_t var_idx,
                              f_t delta,
                              bool localmin = false)
{
  if (localmin) {
    return (delta < 0 && fj_cpu.iterations == fj_cpu.h_tabu_lastinc[var_idx] + 1) ||
           (delta >= 0 && fj_cpu.iterations == fj_cpu.h_tabu_lastdec[var_idx] + 1);
  } else {
    return (delta < 0 && fj_cpu.iterations < fj_cpu.h_tabu_nodec_until[var_idx]) ||
           (delta >= 0 && fj_cpu.iterations < fj_cpu.h_tabu_noinc_until[var_idx]);
  }
}

template <typename i_t, typename f_t>
static bool check_variable_feasibility(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                                       bool check_integer = true)
{
  for (i_t var_idx = 0; var_idx < fj_cpu.view.pb.n_variables; var_idx += 1) {
    auto val      = fj_cpu.h_assignment[var_idx];
    bool feasible = check_variable_within_bounds<i_t, f_t>(fj_cpu, var_idx, val);

    if (!feasible) return false;
    if (check_integer && is_integer_var<i_t, f_t>(fj_cpu, var_idx) &&
        !fj_cpu.view.pb.is_integer(fj_cpu.h_assignment[var_idx]))
      return false;
  }
  return true;
}

template <typename i_t, typename f_t>
static inline std::pair<fj_staged_score_t, f_t> compute_score(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                                                              i_t var_idx,
                                                              f_t delta)
{
  // timing_raii_t<i_t, f_t> timer(fj_cpu.compute_score_times);

  f_t obj_diff = fj_cpu.h_obj_coeffs[var_idx] * delta;

  cuopt_assert(isfinite(delta), "");

  cuopt_assert(var_idx < fj_cpu.view.pb.n_variables, "variable index out of bounds");

  f_t base_feas_sum    = 0;
  f_t bonus_robust_sum = 0;

  auto [offset_begin, offset_end] = reverse_range_for_var<i_t, f_t>(fj_cpu, var_idx);
  fj_cpu.nnz_processed_window += (offset_end - offset_begin);

  const size_t nnz_read = (size_t)(offset_end - offset_begin);
  fj_cpu.h_reverse_constraints.byte_loads += nnz_read * sizeof(i_t);
  fj_cpu.h_reverse_coefficients.byte_loads += nnz_read * sizeof(f_t);
  fj_cpu.cached_cstr_bounds.byte_loads += nnz_read * sizeof(std::pair<f_t, f_t>);
  fj_cpu.h_lhs.byte_loads += nnz_read * sizeof(f_t);
  fj_cpu.h_cstr_left_weights.byte_loads += nnz_read * sizeof(f_t);
  fj_cpu.h_cstr_right_weights.byte_loads += nnz_read * sizeof(f_t);

  const i_t* const rev_cstr                    = fj_cpu.view.pb.reverse_constraints.data();
  const f_t* const rev_coeff                   = fj_cpu.view.pb.reverse_coefficients.data();
  const f_t* const row_lhs                     = fj_cpu.view.incumbent_lhs.data();
  const f_t* const weight_l                    = fj_cpu.view.cstr_left_weights.data();
  const f_t* const weight_r                    = fj_cpu.view.cstr_right_weights.data();
  const std::pair<f_t, f_t>* const cstr_bounds = fj_cpu.cached_cstr_bounds.data();

  for (i_t i = offset_begin; i < offset_end; i++) {
    const i_t cstr_idx      = rev_cstr[i];
    const f_t cstr_coeff    = rev_coeff[i];
    const auto [c_lb, c_ub] = cstr_bounds[i];

    cuopt_assert(c_lb <= c_ub, "invalid bounds");

    auto [cstr_base_feas, cstr_bonus_robust] = feas_score_constraint<i_t, f_t>(fj_cpu.view,
                                                                              delta,
                                                                              cstr_idx,
                                                                              cstr_coeff,
                                                                              c_lb,
                                                                              c_ub,
                                                                              row_lhs[cstr_idx],
                                                                              weight_l[cstr_idx],
                                                                              weight_r[cstr_idx]);

    base_feas_sum += cstr_base_feas;
    bonus_robust_sum += cstr_bonus_robust;
  }

  f_t base_obj = 0;
  if (fj_cpu.h_objective_weight > 0 && obj_diff != 0) {
    // Scaling base is only meaningful where there is feasibility impact to trade against.
    f_t weighted = fj_cpu.h_objective_weight;
    if (base_feas_sum != 0) {
      cuopt_assert(fj_cpu.obj_magnitude > 0, "objective magnitude unit must be positive");
      weighted *= min((f_t)fj_obj_mult_max,
                      max((f_t)fj_obj_mult_min, fabs(obj_diff) / fj_cpu.obj_magnitude));
    }
    base_obj = obj_diff < 0 ? weighted : -weighted;
  }

  f_t bonus_breakthrough = 0;

  bool old_obj_better = fj_cpu.h_incumbent_objective < fj_cpu.h_best_objective;
  bool new_obj_better = fj_cpu.h_incumbent_objective + obj_diff < fj_cpu.h_best_objective;
  if (!old_obj_better && new_obj_better)
    bonus_breakthrough += fj_cpu.h_objective_weight;
  else if (old_obj_better && !new_obj_better) {
    bonus_breakthrough -= fj_cpu.h_objective_weight;
  }

  fj_staged_score_t score;
  score.base  = round(base_obj + base_feas_sum);
  score.bonus = round(bonus_breakthrough + bonus_robust_sum);
  return std::make_pair(score, base_feas_sum);
}

struct two_opt_move_t {
  fj_move_t first{-1, 0};
  fj_move_t second{-1, 0};
  fj_staged_score_t score{fj_staged_score_t::invalid()};
  int age{std::numeric_limits<int>::max()};

  bool operator>(const two_opt_move_t& other) const
  {
    if (score != other.score) return score > other.score;
    if (age != other.age) return age < other.age;
    if (first.var_idx != other.first.var_idx) return first.var_idx < other.first.var_idx;
    return second.var_idx < other.second.var_idx;
  }
};

// returns the combined score of a joint 2opt move
template <typename i_t, typename f_t>
static fj_staged_score_t two_opt_compute_pair_score(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t first, f_t first_delta, i_t second, f_t second_delta)
{
  auto& row_deltas = fj_cpu.two_opt_row_deltas;
  row_deltas.clear();
  const fj_move_t endpoints[2] = {{first, first_delta}, {second, second_delta}};
  for (const auto& [var_idx, delta] : endpoints) {
    const auto [offset_begin, offset_end] = reverse_range_for_var<i_t, f_t>(fj_cpu, var_idx);
    fj_cpu.nnz_processed_window += offset_end - offset_begin;
    for (i_t i = offset_begin; i < offset_end; ++i) {
      const i_t cstr_idx = fj_cpu.h_reverse_constraints[i];
      const f_t coeff    = fj_cpu.h_reverse_coefficients[i];
      row_deltas.emplace_back(cstr_idx, coeff * delta);
    }
  }
  // Brings the entries of a shared row next to each other
  std::sort(row_deltas.begin(), row_deltas.end());

  f_t base_feas_sum    = 0;
  f_t bonus_robust_sum = 0;
  for (size_t pos = 0; pos < row_deltas.size();) {
    const i_t cstr_idx = row_deltas[pos].first;
    f_t lhs_delta      = 0;
    do {
      lhs_delta += row_deltas[pos++].second;
    } while (pos < row_deltas.size() && row_deltas[pos].first == cstr_idx);

    // The coefficients are already folded into lhs_delta, hence the unit coefficient
    auto [cstr_base_feas, cstr_bonus_robust] =
      feas_score_constraint<i_t, f_t>(fj_cpu.view,
                                      lhs_delta,
                                      cstr_idx,
                                      1,
                                      fj_cpu.h_cstr_lb[cstr_idx],
                                      fj_cpu.h_cstr_ub[cstr_idx],
                                      fj_cpu.h_lhs[cstr_idx],
                                      fj_cpu.h_cstr_left_weights[cstr_idx],
                                      fj_cpu.h_cstr_right_weights[cstr_idx]);
    base_feas_sum += cstr_base_feas;
    bonus_robust_sum += cstr_bonus_robust;
  }

  const f_t obj_diff =
    fj_cpu.h_obj_coeffs[first] * first_delta + fj_cpu.h_obj_coeffs[second] * second_delta;
  f_t base_obj = 0;
  if (obj_diff < 0)
    base_obj = fj_cpu.h_objective_weight;
  else if (obj_diff > 0)
    base_obj = -fj_cpu.h_objective_weight;

  f_t bonus_breakthrough = 0;
  bool old_obj_better    = fj_cpu.h_incumbent_objective < fj_cpu.h_best_objective;
  bool new_obj_better    = fj_cpu.h_incumbent_objective + obj_diff < fj_cpu.h_best_objective;
  if (!old_obj_better && new_obj_better)
    bonus_breakthrough += fj_cpu.h_objective_weight;
  else if (old_obj_better && !new_obj_better)
    bonus_breakthrough -= fj_cpu.h_objective_weight;

  fj_staged_score_t score;
  score.base  = round(base_obj + base_feas_sum);
  score.bonus = round(bonus_breakthrough + bonus_robust_sum);
  return score;
}

template <typename i_t, typename f_t>
static void two_opt_add_partner(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                                i_t first,
                                i_t var_idx,
                                f_t target)
{
  if (var_idx == first) return;
  const f_t val = fj_cpu.h_assignment[var_idx].get();
  // A partner between two integers has no opposite value to swap to
  if (!fj_cpu.view.pb.is_integer(val)) return;
  const f_t delta = target - val;
  // Already at the value we would move it to, so there is no compound move to make
  if (fabs(delta) < 0.5) return;
  if (!check_variable_within_bounds<i_t, f_t>(fj_cpu, var_idx, target)) return;
  if (tabu_check<i_t, f_t>(fj_cpu, var_idx, delta, true)) return;
  fj_cpu.two_opt_partners.emplace_back(var_idx, delta);
}

/**
 * @brief Fill fj_cpu.two_opt_partners with candidates to flip together with `first`.
 *
 * Preferred source is the probing cache: it recorded, for each probed variable and value, the
 * bounds propagation implies on every other variable. An implied bound pinning a binary to a value
 * names both the partner and the value it has to take once `first` moves, so a pair moving in the
 * same direction is reached as naturally as a swap. The
 * variables sharing a row with it are used as fallback.
 */
template <typename i_t, typename f_t>
static void two_opt_collect_partners(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                                     i_t first,
                                     f_t first_delta,
                                     size_t max_partners)
{
  auto& partners        = fj_cpu.two_opt_partners;
  const i_t n_variables = fj_cpu.view.pb.n_variables;
  partners.clear();
  cuopt_assert(fj_cpu.h_is_binary_variable[first], "2-opt is only defined for binaries");
  cuopt_assert(
    fj_cpu.probing_cache == nullptr || fj_cpu.h_original_ids.size() == (size_t)n_variables,
    "original id map does not cover every variable");
  cuopt_assert(fj_cpu.probing_cache == nullptr ||
                 fj_cpu.h_reverse_original_ids.size() >= fj_cpu.h_original_ids.size(),
               "reverse original id map smaller than the problem");

  if (fj_cpu.probing_cache != nullptr) {
    const auto& cache       = fj_cpu.probing_cache->probing_cache;
    const auto cached_probe = cache.find(fj_cpu.h_original_ids[first]);
    if (cached_probe != cache.end()) {
      const f_t new_val = fj_cpu.h_assignment[first].get() + first_delta;
      i_t hit_interval  = -1;
      i_t unused_hit    = -1;
      for (i_t interval = 0; interval < 2; ++interval) {
        const auto& entry = cached_probe->second[interval];
        if (entry.var_to_cached_bound_map.empty()) { continue; }
        entry.val_interval.fill_cache_hits(interval, new_val, new_val, hit_interval, unused_hit);
      }
      if (hit_interval != -1) {
        const auto& implications = cached_probe->second[hit_interval].var_to_cached_bound_map;
        for (const auto& [probed_id, implied] : implications) {
          if (partners.size() >= max_partners) break;
          const i_t var_idx = fj_cpu.h_reverse_original_ids[probed_id];
          // -1 means presolve removed the variable after the probe recorded it
          if (var_idx < 0) { continue; }
          cuopt_assert(var_idx < n_variables, "implied variable out of range");
          if (!fj_cpu.h_is_binary_variable[var_idx]) { continue; }
          if (!fj_cpu.view.pb.integer_equal(implied.lb, implied.ub)) { continue; }
          two_opt_add_partner<i_t, f_t>(fj_cpu, first, var_idx, round(implied.lb));
        }
      }
    }
  }

  const auto& related         = fj_cpu.h_related_variables;
  const auto& related_offsets = fj_cpu.h_related_variables_offsets;
  if (related_offsets.size() != (size_t)n_variables + 1) return;
  const f_t swap_target   = fj_cpu.h_assignment[first].get();
  const i_t related_begin = related_offsets[first];
  const i_t related_end   = related_offsets[first + 1];
  for (i_t i = related_begin; i < related_end && partners.size() < max_partners; ++i) {
    const i_t var_idx = related[i];
    if (fj_cpu.h_is_binary_variable[var_idx]) {
      two_opt_add_partner<i_t, f_t>(fj_cpu, first, var_idx, swap_target);
    }
  }
}

// Look for binary 2opt moves at a local minimum. by definition no 1opt move can improve, but
// combined moves may especially in the case of set partitioning constraints / cliques. Use
// information from the probing cache to find potential good 2opt moves.
template <typename i_t, typename f_t>
static two_opt_move_t find_two_opt_move(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  CPUFJ_NVTX_RANGE("CPUFJ::find_two_opt_move");
  constexpr size_t max_obj_starts       = 64;
  constexpr size_t max_partners_per_var = 16;

  const auto& params           = fj_cpu.settings.parameters;
  const size_t max_target_rows = params.two_opt_max_rows;
  const size_t max_first_vars  = params.two_opt_max_row_vars;
  const size_t max_pairs       = params.two_opt_max_pairs;

  two_opt_move_t best;

  const bool partner_source_exists =
    (fj_cpu.probing_cache != nullptr && !fj_cpu.probing_cache->probing_cache.empty()) ||
    (int64_t)fj_cpu.h_related_variables_offsets.size() == fj_cpu.view.pb.n_variables + 1;

  if (fj_cpu.n_binary_vars == 0 || !partner_source_exists) return best;

  auto& first_vars = fj_cpu.two_opt_first_vars;
  first_vars.clear();

  // target binvars in violated constraints for flips
  if (!fj_cpu.violated_constraints.empty()) {
    cuopt_assert(fj_cpu.h_binrow_offsets.size() == fj_cpu.view.pb.n_constraints + 1,
                 "binary row table missing");
    auto& target_cstrs = fj_cpu.two_opt_target_cstrs;
    target_cstrs.clear();
    std::sample(fj_cpu.violated_constraints.begin(),
                fj_cpu.violated_constraints.end(),
                std::back_inserter(target_cstrs),
                max_target_rows,
                fj_cpu.rng);
    for (i_t cstr_idx : target_cstrs) {
      const i_t bin_begin = fj_cpu.h_binrow_offsets[cstr_idx];
      const i_t bin_end   = fj_cpu.h_binrow_offsets[cstr_idx + 1];
      for (i_t i = bin_begin; i < bin_end && first_vars.size() < max_first_vars; ++i) {
        first_vars.push_back(fj_cpu.h_binrow_vars[i].get());
      }
    }
  } else {
    // target objective-bearing binary vars in satisfied constraints
    std::sample(fj_cpu.h_objective_vars.underlying().begin(),
                fj_cpu.h_objective_vars.underlying().end(),
                std::back_inserter(first_vars),
                max_obj_starts,
                fj_cpu.rng);
    first_vars.erase(std::remove_if(first_vars.begin(),
                                    first_vars.end(),
                                    [&](i_t var_idx) {
                                      if (!fj_cpu.h_is_binary_variable[var_idx]) return true;
                                      const f_t delta =
                                        round(1 - 2 * fj_cpu.h_assignment[var_idx].get());
                                      return fj_cpu.h_obj_coeffs[var_idx] * delta >= 0;
                                    }),
                     first_vars.end());
  }
  std::shuffle(first_vars.begin(), first_vars.end(), fj_cpu.rng);

  const i_t nnz_at_entry = fj_cpu.nnz_processed_window;
  size_t pairs_scored    = 0;
  // find a (first, second) pair for the 2opt
  for (i_t first : first_vars) {
    if (pairs_scored >= max_pairs) break;
    if (fj_cpu.nnz_processed_window - nnz_at_entry > fj_cpu.nnz_samples) break;
    const f_t first_val = fj_cpu.h_assignment[first].get();
    if (!fj_cpu.view.pb.is_integer(first_val)) continue;
    const f_t first_delta = round(1 - 2 * first_val);
    if (tabu_check<i_t, f_t>(fj_cpu, first, first_delta, true)) continue;
    if (!check_variable_within_bounds<i_t, f_t>(fj_cpu, first, first_val + first_delta)) continue;
    const i_t first_touch = std::max(fj_cpu.h_tabu_lastinc[first], fj_cpu.h_tabu_lastdec[first]);

    // look for potential other binary vars to flip alongside the first var
    two_opt_collect_partners(fj_cpu, first, first_delta, max_partners_per_var);
    for (const auto& [second, second_delta] : fj_cpu.two_opt_partners) {
      const i_t second_touch =
        std::max(fj_cpu.h_tabu_lastinc[second], fj_cpu.h_tabu_lastdec[second]);
      two_opt_move_t cand;
      cand.first  = {first, first_delta};
      cand.second = {second, second_delta};
      cand.score  = two_opt_compute_pair_score(fj_cpu, first, first_delta, second, second_delta);
      cand.age    = std::max(first_touch, second_touch);
      if (cand > best) { best = cand; }
      ++pairs_scored;

      if (pairs_scored >= max_pairs) return best;
      if (fj_cpu.nnz_processed_window - nnz_at_entry > fj_cpu.nnz_samples) return best;
    }
  }
  return best;
}

template <typename i_t, typename f_t>
static void smooth_weights(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  CPUFJ_NVTX_RANGE("CPUFJ::smooth_weights");
  for (i_t cstr_idx = 0; cstr_idx < fj_cpu.view.pb.n_constraints; cstr_idx++) {
    // consider only satisfied constraints
    if (fj_cpu.violated_constraints.contains(cstr_idx)) continue;

    f_t weight_l = max((f_t)0, fj_cpu.h_cstr_left_weights[cstr_idx] - 1);
    f_t weight_r = max((f_t)0, fj_cpu.h_cstr_right_weights[cstr_idx] - 1);

    fj_cpu.h_cstr_left_weights[cstr_idx]  = weight_l;
    fj_cpu.h_cstr_right_weights[cstr_idx] = weight_r;
  }

  if (fj_cpu.h_objective_weight > 0 && fj_cpu.h_incumbent_objective >= fj_cpu.h_best_objective) {
    fj_cpu.h_objective_weight =
      max(fj_cpu.seed_objective_weight, fj_cpu.h_objective_weight - 1);
  }
}

// Escalation threshold and step for the violated-row bump, in local minima without a severity gain.
constexpr int32_t fj_weight_escalate_after = 2000;
constexpr int32_t fj_weight_escalate_max   = 100;

template <typename i_t, typename f_t>
static i_t weight_escalation_delta(const fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  const i_t stall = fj_cpu.iters_since_infeasible_improve;
  if (stall <= fj_weight_escalate_after) return 1;
  const i_t steps = (stall - fj_weight_escalate_after) / fj_weight_escalate_after + 1;
  return steps < fj_weight_escalate_max ? steps : fj_weight_escalate_max;
}

template <typename i_t, typename f_t>
static void update_weights(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  timing_raii_t<i_t, f_t> timer(fj_cpu.update_weights_times);
  CPUFJ_NVTX_RANGE("CPUFJ::update_weights");

  raft::random::PCGenerator rng(fj_cpu.settings.seed + fj_cpu.iterations, 0, 0);
  bool smoothing = rng.next_float() <= fj_cpu.settings.parameters.weight_smoothing_probability;

  if (smoothing) {
    smooth_weights<i_t, f_t>(fj_cpu);
    return;
  }

  const i_t escalated_delta = weight_escalation_delta<i_t, f_t>(fj_cpu);

  for (auto cstr_idx : fj_cpu.violated_constraints) {
    f_t curr_incumbent_lhs = fj_cpu.h_lhs[cstr_idx];
    f_t curr_lower_excess =
      fj_cpu.view.lower_excess_score(cstr_idx, curr_incumbent_lhs, fj_cpu.h_cstr_lb[cstr_idx]);
    f_t curr_upper_excess =
      fj_cpu.view.upper_excess_score(cstr_idx, curr_incumbent_lhs, fj_cpu.h_cstr_ub[cstr_idx]);
    f_t curr_excess_score = curr_lower_excess + curr_upper_excess;

    f_t old_weight;
    if (curr_lower_excess < 0.) {
      old_weight = fj_cpu.h_cstr_left_weights[cstr_idx];
    } else {
      old_weight = fj_cpu.h_cstr_right_weights[cstr_idx];
    }

    cuopt_assert(curr_excess_score < 0, "constraint not violated");

    i_t int_delta = escalated_delta;
    f_t delta     = int_delta;

    f_t new_weight = old_weight + delta;
    new_weight     = round(new_weight);

    if (curr_lower_excess < 0.) {
      fj_cpu.h_cstr_left_weights[cstr_idx] = new_weight;
      fj_cpu.max_weight                    = max(fj_cpu.max_weight, new_weight);
    } else {
      fj_cpu.h_cstr_right_weights[cstr_idx] = new_weight;
      fj_cpu.max_weight                     = max(fj_cpu.max_weight, new_weight);
    }

    // Invalidate related cached move scores
    fj_cpu.h_cstr_version[cstr_idx]++;
  }

  if (fj_cpu.violated_constraints.empty()) { fj_cpu.h_objective_weight += 1; }
}

// Bump and ceiling applied to the objective weight when a new incumbent lands.
constexpr double fj_obj_weight_incumbent_bump = 4.0;
constexpr double fj_obj_weight_incumbent_cap  = 64.0;

template <typename i_t, typename f_t>
static void apply_move(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                       i_t var_idx,
                       f_t delta,
                       bool localmin = false)
{
  timing_raii_t<i_t, f_t> timer(fj_cpu.apply_move_times);
  CPUFJ_NVTX_RANGE("CPUFJ::apply_move");

  raft::random::PCGenerator rng(fj_cpu.settings.seed + fj_cpu.iterations, 0, 0);

  cuopt_assert(var_idx < fj_cpu.view.pb.n_variables, "variable index out of bounds");
  f_t old_val = fj_cpu.h_assignment[var_idx];
  f_t new_val = old_val + delta;
  if (is_integer_var<i_t, f_t>(fj_cpu, var_idx)) {
    cuopt_assert(fj_cpu.view.pb.integer_equal(new_val, round(new_val)), "new_val is not integer");
    new_val = round(new_val);
  }
  // clamp to var bounds
  new_val = std::min(std::max(new_val, get_lower(fj_cpu.h_var_bounds[var_idx].get())),
                     get_upper(fj_cpu.h_var_bounds[var_idx].get()));
  delta   = new_val - old_val;
  cuopt_assert(isfinite(new_val), "assignment is not finite");
  cuopt_assert(isfinite(delta), "applied delta is not finite");
  cuopt_assert((check_variable_within_bounds<i_t, f_t>(fj_cpu, var_idx, new_val)),
               "assignment not within bounds");

  // Update the LHSs of all involved constraints.
  auto [offset_begin, offset_end] = reverse_range_for_var<i_t, f_t>(fj_cpu, var_idx);

  fj_cpu.nnz_processed_window += (offset_end - offset_begin);
  fj_cpu.n_variable_updates_window++;
  fj_cpu.unique_vars_accessed_window.insert(var_idx);

  for (auto i = offset_begin; i < offset_end; i++) {
    cuopt_assert(i < (i_t)fj_cpu.h_reverse_constraints.size(), "");
    auto [c_lb, c_ub] = fj_cpu.cached_cstr_bounds[i].get();

    auto cstr_idx = fj_cpu.h_reverse_constraints[i];
    auto cstr_coeff = fj_cpu.h_reverse_coefficients[i];

    f_t old_lhs = fj_cpu.h_lhs[cstr_idx];
    // Kahan compensated summation
    f_t y                          = cstr_coeff * delta - fj_cpu.h_lhs_sumcomp[cstr_idx];
    f_t t                          = old_lhs + y;
    fj_cpu.h_lhs_sumcomp[cstr_idx] = (t - old_lhs) - y;
    fj_cpu.h_lhs[cstr_idx]         = t;
    f_t new_lhs                    = fj_cpu.h_lhs[cstr_idx];
    f_t old_cost                   = fj_cpu.view.excess_score(cstr_idx, old_lhs, c_lb, c_ub);
    f_t new_cost                   = fj_cpu.view.excess_score(cstr_idx, new_lhs, c_lb, c_ub);
    f_t cstr_tolerance             = fj_cpu.view.get_corrected_tolerance(cstr_idx, c_lb, c_ub);

    // trigger early lhs recomputation if the sumcomp term gets too large
    // to avoid large numerical errors
    if (fabs(fj_cpu.h_lhs_sumcomp[cstr_idx]) > BIGVAL_THRESHOLD)
      fj_cpu.trigger_early_lhs_recomputation = true;

    if (new_cost < -cstr_tolerance && !fj_cpu.violated_constraints.contains(cstr_idx)) {
      fj_cpu.violated_constraints.insert(cstr_idx);
      cuopt_assert(fj_cpu.satisfied_constraints.contains(cstr_idx), "");
      fj_cpu.satisfied_constraints.remove(cstr_idx);
    } else if (!(new_cost < -cstr_tolerance) && fj_cpu.violated_constraints.contains(cstr_idx)) {
      cuopt_assert(!fj_cpu.satisfied_constraints.contains(cstr_idx), "");
      fj_cpu.violated_constraints.remove(cstr_idx);
      fj_cpu.satisfied_constraints.insert(cstr_idx);
    }

    cuopt_assert(isfinite(delta), "delta should be finite");
    cuopt_assert(isfinite(fj_cpu.h_lhs[cstr_idx]), "assignment should be finite");

    // Invalidate related cached move scores
    fj_cpu.h_cstr_version[cstr_idx]++;
  }

  // update the assignment and objective proper
  fj_cpu.h_assignment[var_idx] = new_val;

  // Kahan compensated summation, as for h_lhs. The incumbent objective is reported as-is, so it
  // cannot carry the drift of a long uncompensated chain of deltas.
  const f_t obj_old = fj_cpu.h_incumbent_objective;
  const f_t obj_y   = fj_cpu.h_obj_coeffs[var_idx] * delta - fj_cpu.h_objective_sumcomp;
  const f_t obj_t   = obj_old + obj_y;
  fj_cpu.h_objective_sumcomp   = (obj_t - obj_old) - obj_y;
  fj_cpu.h_incumbent_objective = obj_t;

  if (fj_cpu.h_incumbent_objective < fj_cpu.h_best_objective &&
      fj_cpu.violated_constraints.empty() && check_variable_feasibility<i_t, f_t>(fj_cpu)) {
    cuopt_assert(fj_cpu.satisfied_constraints.size() == fj_cpu.view.pb.n_constraints, "");
    fj_cpu.h_best_objective =
      fj_cpu.h_incumbent_objective - fj_cpu.settings.parameters.breakthrough_move_epsilon;
    fj_cpu.h_best_assignment     = fj_cpu.h_assignment;
    fj_cpu.iterations_since_best = 0;
    // DEBUG, and reporting the stored best rather than the pre-epsilon incumbent,
    // so it matches the binary path and the end-of-solve incumbent audit.
    CUOPT_LOG_DEBUG("%sCPUFJ new incumbent: objective %.17g",
                    fj_cpu.log_prefix.c_str(),
                    fj_cpu.h_best_objective);
    if (fj_cpu.improvement_callback) {
      double current_work_units = fj_cpu.work_units_elapsed.load(std::memory_order_acquire);
      fj_cpu.improvement_callback(
        fj_cpu.h_incumbent_objective, fj_cpu.h_assignment, current_work_units);
    }
    fj_cpu.feasible_found = true;
    // The true objective of the assignment, not the epsilon-reduced threshold stored above, so
    // another lane comparing against it is not misled into adopting something no better.
    if (fj_cpu.shared_incumbent) {
      fj_cpu.shared_incumbent->publish(fj_cpu.h_incumbent_objective, fj_cpu.h_assignment);
    }
    // Counteract the smooth_weights decay for a lane that is actively improving, and hold the
    // weight at a scale where base_feas_sum still registers against it.
    if (fj_cpu.h_objective_weight > 0) {
      fj_cpu.h_objective_weight =
        min((f_t)fj_obj_weight_incumbent_cap,
            fj_cpu.h_objective_weight + (f_t)fj_obj_weight_incumbent_bump);
    }
  }

  i_t tabu_tenure = fj_cpu.settings.parameters.tabu_tenure_min +
                    rng.next_u32() % (fj_cpu.settings.parameters.tabu_tenure_max -
                                      fj_cpu.settings.parameters.tabu_tenure_min);
  if (delta > 0) {
    fj_cpu.h_tabu_lastinc[var_idx]     = fj_cpu.iterations;
    fj_cpu.h_tabu_nodec_until[var_idx] = fj_cpu.iterations + tabu_tenure;
    fj_cpu.h_tabu_noinc_until[var_idx] = fj_cpu.iterations + tabu_tenure / 2;
    // CUOPT_LOG_TRACE("CPU: tabu nodec_until: %d\n", fj_cpu.h_tabu_nodec_until[var_idx]);
  } else {
    fj_cpu.h_tabu_lastdec[var_idx]     = fj_cpu.iterations;
    fj_cpu.h_tabu_noinc_until[var_idx] = fj_cpu.iterations + tabu_tenure;
    fj_cpu.h_tabu_nodec_until[var_idx] = fj_cpu.iterations + tabu_tenure / 2;
    // CUOPT_LOG_TRACE("CPU: tabu noinc_until: %d\n", fj_cpu.h_tabu_noinc_until[var_idx]);
  }

  std::fill(fj_cpu.flip_move_computed.begin(), fj_cpu.flip_move_computed.end(), false);
  std::fill(fj_cpu.var_bitmap.begin(), fj_cpu.var_bitmap.end(), false);
  fj_cpu.iter_mtm_vars.clear();
}

template <typename i_t, typename f_t, MTMMoveType move_type>
static thrust::tuple<fj_move_t, fj_staged_score_t> find_mtm_move(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu, const std::vector<i_t>& target_cstrs, bool localmin = false)
{
  CPUFJ_NVTX_RANGE("CPUFJ::find_mtm_move");

  raft::random::PCGenerator rng(fj_cpu.settings.seed + fj_cpu.iterations, 0, 0);

  fj_move_t best_move          = fj_move_t{-1, 0};
  fj_staged_score_t best_score = fj_staged_score_t::invalid();

  // collect all the variables that are involved in the target constraints
  for (size_t cstr_idx : target_cstrs) {
    auto [offset_begin, offset_end] = range_for_constraint<i_t, f_t>(fj_cpu, cstr_idx);
    for (auto i = offset_begin; i < offset_end; i++) {
      i_t var_idx = fj_cpu.h_variables[i];
      if (fj_cpu.var_bitmap[var_idx]) continue;
      fj_cpu.iter_mtm_vars.push_back(var_idx);
      fj_cpu.var_bitmap[var_idx] = true;
    }
  }
  // estimate the amount of nnzs to consider
  i_t nnz_sum = 0;
  for (auto var_idx : fj_cpu.iter_mtm_vars) {
    auto [offset_begin, offset_end] = reverse_range_for_var<i_t, f_t>(fj_cpu, var_idx);
    nnz_sum += offset_end - offset_begin;
  }

  f_t nnz_pick_probability = 1;
  if (nnz_sum > fj_cpu.nnz_samples) nnz_pick_probability = (f_t)fj_cpu.nnz_samples / nnz_sum;

  for (size_t cstr_idx : target_cstrs) {
    auto c_lb    = fj_cpu.h_cstr_lb[cstr_idx];
    auto c_ub    = fj_cpu.h_cstr_ub[cstr_idx];
    f_t cstr_tol = fj_cpu.view.get_corrected_tolerance(cstr_idx, c_lb, c_ub);

    cuopt_assert(cstr_idx < fj_cpu.h_cstr_lb.size(), "cstr_idx is out of bounds");
    auto [offset_begin, offset_end] = range_for_constraint<i_t, f_t>(fj_cpu, cstr_idx);
    for (auto i = offset_begin; i < offset_end; i++) {
      // early cached check
      cuopt_assert(fj_cpu.cached_mtm_moves_version[i] <= fj_cpu.h_cstr_version[cstr_idx],
                   "cached move newer than its constraint");
      if (auto& cached_move = fj_cpu.cached_mtm_moves[i];
          cached_move.first != 0 &&
          fj_cpu.cached_mtm_moves_version[i] == fj_cpu.h_cstr_version[cstr_idx]) {
        if (best_score < cached_move.second) {
          auto var_idx = fj_cpu.h_variables[i];
          if (check_variable_within_bounds<i_t, f_t>(
                fj_cpu, var_idx, fj_cpu.h_assignment[var_idx] + cached_move.first)) {
            best_score = cached_move.second;
            best_move  = fj_move_t{var_idx, cached_move.first};
          }
          // cuopt_assert(fj_cpu.view.pb.check_variable_within_bounds(var_idx,
          // fj_cpu.h_assignment[var_idx] + cached_move.first), "best move is not within bounds");
        }
        fj_cpu.hit_count++;
        continue;
      }

      // random chance to skip this nnz if there are many to consider
      if (nnz_pick_probability < 1)
        if (rng.next_float() > nnz_pick_probability) continue;

      auto var_idx = fj_cpu.h_variables[i];

      f_t val     = fj_cpu.h_assignment[var_idx];
      f_t new_val = val;
      f_t delta   = 0;

      // Special case for binary variables
      if (fj_cpu.h_is_binary_variable[var_idx]) {
        if (fj_cpu.flip_move_computed[var_idx]) continue;
        fj_cpu.flip_move_computed[var_idx] = true;
        new_val                            = 1 - val;
      } else {
        auto cstr_coeff = fj_cpu.h_coefficients[i];

        f_t c_lb = fj_cpu.h_cstr_lb[cstr_idx];
        f_t c_ub = fj_cpu.h_cstr_ub[cstr_idx];
        auto [delta, sign, slack, cstr_tolerance] =
          get_mtm_for_constraint<i_t, f_t, move_type>(fj_cpu.view,
                                                      var_idx,
                                                      cstr_idx,
                                                      cstr_coeff,
                                                      c_lb,
                                                      c_ub,
                                                      fj_cpu.h_assignment,
                                                      fj_cpu.h_lhs);
        if (is_integer_var<i_t, f_t>(fj_cpu, var_idx)) {
          new_val = cstr_coeff * sign > 0
                      ? floor(val + delta + fj_cpu.view.pb.tolerances.integrality_tolerance)
                      : ceil(val + delta - fj_cpu.view.pb.tolerances.integrality_tolerance);
        } else {
          new_val = val + delta;
        }
        // fallback
        if (new_val < get_lower(fj_cpu.h_var_bounds[var_idx].get()) ||
            new_val > get_upper(fj_cpu.h_var_bounds[var_idx].get())) {
          new_val = cstr_coeff * sign > 0 ? get_lower(fj_cpu.h_var_bounds[var_idx].get())
                                          : get_upper(fj_cpu.h_var_bounds[var_idx].get());
        }
      }
      if (!isfinite(new_val)) continue;
      cuopt_assert((check_variable_within_bounds<i_t, f_t>(fj_cpu, var_idx, new_val)),
                   "new_val is not within bounds");
      delta = new_val - val;
      // more permissive tabu in the case of local minima
      if (tabu_check<i_t, f_t>(fj_cpu, var_idx, delta, localmin)) continue;
      if (fabs(delta) < cstr_tol) continue;

      auto move = fj_move_t{var_idx, delta};
      cuopt_assert(move.var_idx < fj_cpu.h_assignment.size(), "move.var_idx is out of bounds");
      cuopt_assert(move.var_idx >= 0, "move.var_idx is not positive");

      auto [score, infeasibility]        = compute_score<i_t, f_t>(fj_cpu, var_idx, delta);
      fj_cpu.cached_mtm_moves[i]         = std::make_pair(delta, score);
      fj_cpu.cached_mtm_moves_version[i] = fj_cpu.h_cstr_version[cstr_idx];
      fj_cpu.miss_count++;
      // reject this move if it would increase the target variable to a numerically unstable value
      if (fj_cpu.view.move_numerically_stable(
            val, new_val, infeasibility, fj_cpu.total_violations)) {
        if (best_score < score) {
          best_score = score;
          best_move  = move;
        }
      }
    }
  }

  // also consider BM moves if we have found a feasible solution at least once
  if (move_type == MTMMoveType::FJ_MTM_VIOLATED &&
      fj_cpu.h_best_objective < std::numeric_limits<f_t>::infinity() &&
      fj_cpu.h_incumbent_objective >=
        fj_cpu.h_best_objective + fj_cpu.settings.parameters.breakthrough_move_epsilon) {
    for (auto var_idx : fj_cpu.h_objective_vars) {
      f_t old_val = fj_cpu.h_assignment[var_idx];
      f_t new_val = get_breakthrough_move<i_t, f_t>(fj_cpu.view, var_idx);

      if (fj_cpu.view.pb.integer_equal(new_val, old_val) || !isfinite(new_val)) continue;

      f_t delta = new_val - old_val;

      // Check if we already have a move for this variable
      auto move = fj_move_t{var_idx, delta};
      cuopt_assert(move.var_idx < fj_cpu.h_assignment.size(), "move.var_idx is out of bounds");
      cuopt_assert(move.var_idx >= 0, "move.var_idx is not positive");

      if (tabu_check<i_t, f_t>(fj_cpu, var_idx, delta)) continue;

      auto [score, infeasibility] = compute_score<i_t, f_t>(fj_cpu, var_idx, delta);

      cuopt_assert((check_variable_within_bounds<i_t, f_t>(fj_cpu, var_idx, new_val)), "");
      cuopt_assert(isfinite(delta), "");

      if (fj_cpu.view.move_numerically_stable(
            old_val, new_val, infeasibility, fj_cpu.total_violations)) {
        if (best_score < score) {
          best_score = score;
          best_move  = move;
        }
      }
    }
  }

  return thrust::make_tuple(best_move, best_score);
}

template <typename i_t>
static void sample_with_replacement(const host_contiguous_set_t<i_t>& pool,
                                    i_t sample_size,
                                    uint64_t seed,
                                    std::vector<i_t>& out)
{
  cuopt_assert(sample_size > 0, "invalid sample size");
  out.clear();
  const i_t pool_size = pool.size();
  if (pool_size == 0) { return; }
  if (pool_size <= sample_size) {
    out.assign(pool.begin(), pool.end());
    return;
  }
  out.reserve(sample_size);
  cuopt::pcgenerator_t rng(seed);
  for (i_t i = 0; i < sample_size; ++i) {
    out.push_back(pool.contents[rng.next_u32() % (uint32_t)pool_size]);
  }
}

template <typename i_t, typename f_t>
static thrust::tuple<fj_move_t, fj_staged_score_t> find_mtm_move_viol(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t sample_size = 100, bool localmin = false)
{
  timing_raii_t<i_t, f_t> timer(fj_cpu.find_mtm_move_viol_times);
  CPUFJ_NVTX_RANGE("CPUFJ::find_mtm_move_viol");

  std::vector<i_t> sampled_cstrs;
  sample_with_replacement(fj_cpu.violated_constraints,
                          sample_size,
                          fj_cpu.settings.seed + fj_cpu.iterations,
                          sampled_cstrs);

  return find_mtm_move<i_t, f_t, MTMMoveType::FJ_MTM_VIOLATED>(fj_cpu, sampled_cstrs, localmin);
}

template <typename i_t, typename f_t>
static thrust::tuple<fj_move_t, fj_staged_score_t> find_mtm_move_sat(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t sample_size = 100)
{
  timing_raii_t<i_t, f_t> timer(fj_cpu.find_mtm_move_sat_times);
  CPUFJ_NVTX_RANGE("CPUFJ::find_mtm_move_sat");

  std::vector<i_t> sampled_cstrs;
  sample_with_replacement(fj_cpu.satisfied_constraints,
                          sample_size,
                          fj_cpu.settings.seed + fj_cpu.iterations,
                          sampled_cstrs);

  return find_mtm_move<i_t, f_t, MTMMoveType::FJ_MTM_SATISFIED>(fj_cpu, sampled_cstrs);
}

template <typename i_t, typename f_t>
static void recompute_lhs(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  CPUFJ_NVTX_RANGE("CPUFJ::recompute_lhs");
  cuopt_assert(fj_cpu.h_lhs.size() == fj_cpu.view.pb.n_constraints, "h_lhs size mismatch");

  // clamp to var bounds - defensive; apply_move should already have clamped appropriately
  for (i_t var_idx = 0; var_idx < fj_cpu.view.pb.n_variables; ++var_idx) {
    fj_cpu.h_assignment[var_idx] = std::min(
      std::max(fj_cpu.h_assignment[var_idx].get(), get_lower(fj_cpu.h_var_bounds[var_idx].get())),
      get_upper(fj_cpu.h_var_bounds[var_idx].get()));
  }

  fj_cpu.violated_constraints.clear();
  fj_cpu.satisfied_constraints.clear();
  fj_cpu.total_violations = 0;
  for (i_t cstr_idx = 0; cstr_idx < fj_cpu.view.pb.n_constraints; ++cstr_idx) {
    auto [offset_begin, offset_end] = range_for_constraint<i_t, f_t>(fj_cpu, cstr_idx);
    auto c_lb                       = fj_cpu.h_cstr_lb[cstr_idx];
    auto c_ub                       = fj_cpu.h_cstr_ub[cstr_idx];
    auto delta_it =
      thrust::make_transform_iterator(thrust::make_counting_iterator(0), [&fj_cpu](i_t j) {
        return fj_cpu.h_coefficients[j] * fj_cpu.h_assignment[fj_cpu.h_variables[j]];
      });
    fj_cpu.h_lhs[cstr_idx] =
      fj_kahan_babushka_neumaier_sum<i_t, f_t>(delta_it + offset_begin, delta_it + offset_end);
    fj_cpu.h_lhs_sumcomp[cstr_idx] = 0;

    f_t cstr_tolerance = fj_cpu.view.get_corrected_tolerance(cstr_idx, c_lb, c_ub);
    f_t new_cost       = fj_cpu.view.excess_score(cstr_idx, fj_cpu.h_lhs[cstr_idx]);
    if (new_cost < -cstr_tolerance) {
      fj_cpu.violated_constraints.insert(cstr_idx);
      fj_cpu.total_violations += new_cost;
    } else {
      fj_cpu.satisfied_constraints.insert(cstr_idx);
    }
  }

  // compute incumbent objective
  fj_cpu.h_incumbent_objective = thrust::inner_product(
    fj_cpu.h_assignment.begin(), fj_cpu.h_assignment.end(), fj_cpu.h_obj_coeffs.begin(), 0.);
  fj_cpu.h_objective_sumcomp = 0;
}


// Candidate draws per 2-opt lift search.
constexpr int32_t fj_2opt_candidates = 32;

// True when flipping both variables leaves every row they touch satisfied. Both reverse ranges are
// row-ascending, so a merge handles rows containing both variables with their joint delta.
template <typename i_t, typename f_t>
static bool paired_flip_keeps_feasible(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t var1, f_t delta1, i_t var2, f_t delta2)
{
  const auto range1 = reverse_range_for_var<i_t, f_t>(fj_cpu, var1);
  const auto range2 = reverse_range_for_var<i_t, f_t>(fj_cpu, var2);
  i_t i = range1.first, ie = range1.second;
  i_t j = range2.first, je = range2.second;

  while (i < ie || j < je) {
    const i_t r1 = i < ie ? (i_t)fj_cpu.h_reverse_constraints[i] : std::numeric_limits<i_t>::max();
    const i_t r2 = j < je ? (i_t)fj_cpu.h_reverse_constraints[j] : std::numeric_limits<i_t>::max();
    const i_t r  = r1 < r2 ? r1 : r2;

    f_t change = 0;
    f_t c_lb   = 0;
    f_t c_ub   = 0;
    if (r1 == r) {
      auto [lb, ub] = fj_cpu.cached_cstr_bounds[i].get();
      c_lb          = lb;
      c_ub          = ub;
      change += (f_t)fj_cpu.h_reverse_coefficients[i] * delta1;
      ++i;
    }
    if (r2 == r) {
      auto [lb, ub] = fj_cpu.cached_cstr_bounds[j].get();
      c_lb          = lb;
      c_ub          = ub;
      change += (f_t)fj_cpu.h_reverse_coefficients[j] * delta2;
      ++j;
    }

    const f_t new_lhs = fj_cpu.h_lhs[r] + (change - fj_cpu.h_lhs_sumcomp[r]);
    if (fj_cpu.view.excess_score(r, new_lhs, c_lb, c_ub) <
        -fj_cpu.view.get_corrected_tolerance(r, c_lb, c_ub))
      return false;
  }
  return true;
}

template <typename i_t, typename f_t>
static thrust::tuple<fj_move_t, fj_move_t, fj_staged_score_t> find_lift_2opt_move(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  timing_raii_t<i_t, f_t> timer(fj_cpu.find_lift_move_times);
  CPUFJ_NVTX_RANGE("CPUFJ::find_lift_2opt_move");
  cuopt_assert(fj_cpu.violated_constraints.empty(), "lift moves require a feasible incumbent");

  fj_move_t best_first         = fj_move_t{-1, 0};
  fj_move_t best_second        = fj_move_t{-1, 0};
  fj_staged_score_t best_score = fj_staged_score_t::zero();
  f_t best_improvement         = 0;

  const i_t n_obj = (i_t)fj_cpu.h_objective_vars.size();
  if (n_obj == 0) return thrust::make_tuple(best_first, best_second, best_score);

  raft::random::PCGenerator rng(fj_cpu.settings.seed + fj_cpu.iterations, 0, 0);
  const i_t n_draws = n_obj < fj_2opt_candidates ? n_obj : fj_2opt_candidates;

  for (i_t t = 0; t < n_draws; ++t) {
    const i_t var1 = fj_cpu.h_objective_vars[rng.next_u32() % (uint32_t)n_obj];
    if (!fj_cpu.h_is_binary_variable[var1]) continue;

    const f_t coeff1 = fj_cpu.h_obj_coeffs[var1];
    const f_t val1   = fj_cpu.h_assignment[var1];
    const f_t delta1 = round(1.0 - 2 * val1);
    if (delta1 * coeff1 >= 0) continue;
    if (tabu_check<i_t, f_t>(fj_cpu, var1, delta1)) continue;

    // Breaking nothing is the single-flip lift's job; breaking several rows cannot be repaired by
    // one companion.
    const auto range1 = reverse_range_for_var<i_t, f_t>(fj_cpu, var1);
    i_t broken        = -1;
    bool multiple     = false;
    for (i_t k = range1.first; k < range1.second && !multiple; ++k) {
      auto [c_lb, c_ub] = fj_cpu.cached_cstr_bounds[k].get();
      const i_t r       = fj_cpu.h_reverse_constraints[k];
      const f_t new_lhs = fj_cpu.h_lhs[r] + ((f_t)fj_cpu.h_reverse_coefficients[k] * delta1 -
                                             fj_cpu.h_lhs_sumcomp[r]);
      if (fj_cpu.view.excess_score(r, new_lhs, c_lb, c_ub) <
          -fj_cpu.view.get_corrected_tolerance(r, c_lb, c_ub)) {
        if (broken >= 0)
          multiple = true;
        else
          broken = r;
      }
    }
    if (multiple || broken < 0) continue;

    const auto row = range_for_constraint<i_t, f_t>(fj_cpu, broken);
    for (i_t k = row.first; k < row.second; ++k) {
      const i_t var2 = fj_cpu.h_variables[k];
      if (var2 == var1) continue;
      if (!fj_cpu.h_is_binary_variable[var2]) continue;

      const f_t coeff2   = fj_cpu.h_obj_coeffs[var2];
      const f_t val2     = fj_cpu.h_assignment[var2];
      const f_t delta2   = round(1.0 - 2 * val2);
      const f_t combined = delta1 * coeff1 + delta2 * coeff2;
      if (combined >= 0) continue;
      if (tabu_check<i_t, f_t>(fj_cpu, var2, delta2)) continue;
      if (!paired_flip_keeps_feasible<i_t, f_t>(fj_cpu, var1, delta1, var2, delta2)) continue;

      // Both lift operators rank on the objective gain in its own units: the score quantization
      // used elsewhere counts weights, so rounding a gain below 0.5 into it discards the move.
      const f_t improvement = -combined;
      if (improvement > best_improvement) {
        best_improvement = improvement;
        best_score.base  = 1;  // sign only, never compared against another operator's score
        best_first       = fj_move_t{var1, delta1};
        best_second      = fj_move_t{var2, delta2};
      }
    }
  }
  cuopt_assert((best_first.var_idx < 0) == (best_improvement <= 0),
               "pair and score must agree on whether a move was found");
  return thrust::make_tuple(best_first, best_second, best_score);
}

template <typename i_t, typename f_t>
static thrust::tuple<fj_move_t, fj_staged_score_t> find_lift_move(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  timing_raii_t<i_t, f_t> timer(fj_cpu.find_lift_move_times);
  CPUFJ_NVTX_RANGE("CPUFJ::find_lift_move");

  fj_move_t best_move          = fj_move_t{-1, 0};
  fj_staged_score_t best_score = fj_staged_score_t::zero();
  f_t best_improvement         = 0;

  for (auto var_idx : fj_cpu.h_objective_vars) {
    cuopt_assert(var_idx < fj_cpu.h_obj_coeffs.size(), "var_idx is out of bounds");
    cuopt_assert(var_idx >= 0, "var_idx is out of bounds");

    f_t obj_coeff = fj_cpu.h_obj_coeffs[var_idx];
    f_t delta     = -std::numeric_limits<f_t>::infinity();
    f_t val       = fj_cpu.h_assignment[var_idx];

    // special path for binary variables
    if (fj_cpu.h_is_binary_variable[var_idx]) {
      cuopt_assert(fj_cpu.view.pb.is_integer(val), "binary variable is not integer");
      cuopt_assert(fj_cpu.view.pb.integer_equal(val, 0) || fj_cpu.view.pb.integer_equal(val, 1),
                   "Current assignment is not binary!");
      delta = round(1.0 - 2 * val);
      // flip move wouldn't improve
      if (delta * obj_coeff >= 0) continue;

      auto [offset_begin, offset_end] = reverse_range_for_var<i_t, f_t>(fj_cpu, var_idx);
      bool breaks_a_row               = false;
      for (i_t j = offset_begin; j < offset_end; ++j) {
        auto [c_lb, c_ub]    = fj_cpu.cached_cstr_bounds[j].get();
        const i_t cstr_idx   = fj_cpu.h_reverse_constraints[j];
        const f_t cstr_coeff = fj_cpu.h_reverse_coefficients[j];
        const f_t lhs        = fj_cpu.h_lhs[cstr_idx];
        const f_t sumcomp    = fj_cpu.h_lhs_sumcomp[cstr_idx];
        const f_t new_lhs    = lhs + (cstr_coeff * delta - sumcomp);
        if (fj_cpu.view.excess_score(cstr_idx, new_lhs, c_lb, c_ub) <
            -fj_cpu.view.get_corrected_tolerance(cstr_idx, c_lb, c_ub)) {
          breaks_a_row = true;
          break;
        }
      }
      if (breaks_a_row) continue;
    } else {
      f_t lfd_lb                      = get_lower(fj_cpu.h_var_bounds[var_idx].get()) - val;
      f_t lfd_ub                      = get_upper(fj_cpu.h_var_bounds[var_idx].get()) - val;
      auto [offset_begin, offset_end] = reverse_range_for_var<i_t, f_t>(fj_cpu, var_idx);
      for (i_t j = offset_begin; j < offset_end; j += 1) {
        auto cstr_idx      = fj_cpu.h_reverse_constraints[j];
        auto cstr_coeff    = fj_cpu.h_reverse_coefficients[j];
        f_t c_lb           = fj_cpu.h_cstr_lb[cstr_idx];
        f_t c_ub           = fj_cpu.h_cstr_ub[cstr_idx];
        f_t cstr_tolerance = fj_cpu.view.get_corrected_tolerance(cstr_idx, c_lb, c_ub);
        cuopt_assert(c_lb <= c_ub, "invalid bounds");
        cuopt_assert(fj_cpu.view.cstr_satisfied(cstr_idx, fj_cpu.h_lhs[cstr_idx]),
                     "cstr should be satisfied");

        // Process each bound separately, as both are satified and may both be finite
        // otherwise range constraints aren't correctly handled
        for (auto [bound, sign] : {std::make_tuple(c_lb, -1), std::make_tuple(c_ub, 1)}) {
          auto [delta, slack] = get_mtm_for_bound<i_t, f_t>(fj_cpu.view,
                                                            var_idx,
                                                            cstr_idx,
                                                            cstr_coeff,
                                                            bound,
                                                            sign,
                                                            fj_cpu.h_assignment,
                                                            fj_cpu.h_lhs);

          if (cstr_coeff * sign < 0) {
            if (is_integer_var<i_t, f_t>(fj_cpu, var_idx)) delta = ceil(delta);
          } else {
            if (is_integer_var<i_t, f_t>(fj_cpu, var_idx)) delta = floor(delta);
          }

          // skip this variable if there is no slack
          if (fabs(slack) <= cstr_tolerance) {
            if (cstr_coeff * sign > 0) {
              lfd_ub = 0;
            } else {
              lfd_lb = 0;
            }
          } else if (!check_variable_within_bounds<i_t, f_t>(fj_cpu, var_idx, val + delta)) {
            continue;
          } else {
            if (cstr_coeff * sign < 0) {
              lfd_lb = max(lfd_lb, delta);
            } else {
              lfd_ub = min(lfd_ub, delta);
            }
          }
        }
        if (lfd_lb >= lfd_ub) break;
      }

      // invalid crossing bounds
      if (lfd_lb >= lfd_ub) { lfd_lb = lfd_ub = 0; }

      if (!check_variable_within_bounds<i_t, f_t>(fj_cpu, var_idx, val + lfd_lb)) { lfd_lb = 0; }
      if (!check_variable_within_bounds<i_t, f_t>(fj_cpu, var_idx, val + lfd_ub)) { lfd_ub = 0; }

      // Now that the lift move domain is computed, compute the correct lift move
      cuopt_assert(isfinite(val), "invalid assignment value");
      delta = obj_coeff < 0 ? lfd_ub : lfd_lb;
    }

    if (!isfinite(delta)) delta = 0;
    if (fj_cpu.view.pb.integer_equal(delta, (f_t)0)) continue;
    if (tabu_check<i_t, f_t>(fj_cpu, var_idx, delta)) continue;

    cuopt_assert(delta * obj_coeff < 0, "lift move doesn't improve the objective!");

    const f_t improvement = -obj_coeff * delta;
    if (improvement > best_improvement) {
      best_improvement = improvement;
      best_score.base  = 1;
      best_move        = fj_move_t{var_idx, delta};
    }
  }

  cuopt_assert((best_move.var_idx < 0) == (best_improvement <= 0),
               "move and score must agree on whether a move was found");
  return thrust::make_tuple(best_move, best_score);
}

// Draws a uniform in-bounds value, rounded and re-clamped for integer variables.
template <typename i_t, typename f_t>
static void randomize_variable(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                               i_t var_idx,
                               raft::random::PCGenerator& rng)
{
  f_t lb  = std::max(get_lower(fj_cpu.h_var_bounds[var_idx].get()), -1e7);
  f_t ub  = std::min(get_upper(fj_cpu.h_var_bounds[var_idx].get()), 1e7);
  f_t val = lb + (ub - lb) * rng.next_double();
  if (is_integer_var<i_t, f_t>(fj_cpu, var_idx)) {
    lb  = std::ceil(lb);
    ub  = std::floor(ub);
    val = std::round(val);
    val = std::min(std::max(val, lb), ub);
  }

  cuopt_assert((check_variable_within_bounds<i_t, f_t>(fj_cpu, var_idx, val)),
               "value is out of bounds");
  fj_cpu.h_assignment[var_idx] = val;
}

template <typename i_t, typename f_t>
static void perturb(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  CPUFJ_NVTX_RANGE("CPUFJ::perturb");
  if (fj_cpu.feasible_found) {
    cuopt_assert(fj_cpu.h_assignment.size() == fj_cpu.h_best_assignment.size(),
                 "incumbent_assignment span would be invalidated");
    fj_cpu.h_assignment = fj_cpu.h_best_assignment;
    if (fj_cpu.shared_incumbent) {
      fj_cpu.shared_incumbent->adopt(fj_cpu.h_best_objective, fj_cpu.h_assignment);
    }
  }

  // select N variables, assign them a random value between their bounds
  std::vector<i_t> sampled_vars;
  std::sample(fj_cpu.h_objective_vars.begin(),
              fj_cpu.h_objective_vars.end(),
              std::back_inserter(sampled_vars),
              2,
              fj_cpu.rng);
  raft::random::PCGenerator rng(fj_cpu.settings.seed + fj_cpu.iterations, 0, 0);

  for (auto var_idx : sampled_vars)
    randomize_variable<i_t, f_t>(fj_cpu, var_idx, rng);

  recompute_lhs(fj_cpu);
}

template <typename i_t, typename f_t>
static void reset_infeasible_checkpoint(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  fj_cpu.h_best_infeasible_assignment.clear();
  fj_cpu.best_infeasible_severity       = std::numeric_limits<f_t>::infinity();
  fj_cpu.checkpoint_severity            = std::numeric_limits<f_t>::infinity();
  fj_cpu.iters_since_infeasible_improve = 0;
}

template <typename i_t, typename f_t>
static void invalidate_mtm_cache(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  for (size_t c = 0; c < fj_cpu.h_cstr_version.size(); ++c)
    fj_cpu.h_cstr_version[c]++;
}

template <typename i_t, typename f_t>
static void restart_from_infeasible_checkpoint(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  cuopt_assert(fj_cpu.h_assignment.size() == fj_cpu.h_best_infeasible_assignment.size(),
               "incumbent_assignment span would be invalidated");
  fj_cpu.h_assignment = fj_cpu.h_best_infeasible_assignment;
  recompute_lhs(fj_cpu);
  invalidate_mtm_cache(fj_cpu);
}

// Nonzeros per extra restart window, the cap on that, and how many windows a lane waits.
constexpr int32_t fj_restart_window_nnz_scale = 100000;
constexpr int32_t fj_restart_window_scale_max = 4;
constexpr int32_t fj_restart_window_multiple  = 4;

template <typename i_t, typename f_t>
static void track_infeasible_checkpoint(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  CPUFJ_NVTX_RANGE("CPUFJ::track_infeasible_checkpoint");
  if (fj_cpu.violated_constraints.empty()) {
    reset_infeasible_checkpoint(fj_cpu);
    return;
  }

  const f_t severity = -fj_cpu.total_violations;
  cuopt_assert(severity >= 0, "violation severity should be positive or zero");

  if (severity < fj_cpu.best_infeasible_severity) {
    fj_cpu.best_infeasible_severity       = severity;
    fj_cpu.iters_since_infeasible_improve = 0;
    fj_cpu.restores_since_improvement     = 0;
    if (severity < fj_cpu.checkpoint_severity * fj_cpu.infeasible_checkpoint_refresh_ratio) {
      fj_cpu.h_best_infeasible_assignment = fj_cpu.h_assignment;
      fj_cpu.checkpoint_severity          = severity;
      ++fj_cpu.n_checkpoint_snapshots;
    }
    return;
  }

  // A lane that has never crossed and has exhausted its restores abandons the basin outright.
  if (!fj_cpu.feasible_found) {
    const i_t nnz_scale =
      1 + (i_t)fj_cpu.h_coefficients.size() / fj_restart_window_nnz_scale;
    const i_t capped = nnz_scale < fj_restart_window_scale_max ? nnz_scale
                                                              : fj_restart_window_scale_max;
    if (fj_cpu.iters_since_infeasible_improve >=
          fj_restart_window_multiple * fj_cpu.infeasible_restart_window * capped &&
        fj_cpu.restores_since_improvement >= fj_cpu.infeasible_restart_max_streak) {
      raft::random::PCGenerator rng(fj_cpu.settings.seed + fj_cpu.iterations, 0, 0);
      for (i_t var_idx = 0; var_idx < fj_cpu.view.pb.n_variables; ++var_idx)
        randomize_variable<i_t, f_t>(fj_cpu, var_idx, rng);

      recompute_lhs(fj_cpu);
      invalidate_mtm_cache(fj_cpu);
      reset_infeasible_checkpoint(fj_cpu);
      fj_cpu.restores_since_improvement = 0;

      CUOPT_LOG_DEBUG("%sCPUFJ randomized restart at iteration %d",
                      fj_cpu.log_prefix.c_str(),
                      fj_cpu.iterations);
      return;
    }
  }

  if (fj_cpu.restores_since_improvement >= fj_cpu.infeasible_restart_max_streak) return;
  if (++fj_cpu.iters_since_infeasible_improve < fj_cpu.infeasible_restart_window) return;
  if (severity <= fj_cpu.best_infeasible_severity * fj_cpu.infeasible_restart_degrade_ratio) return;
  if (fj_cpu.h_best_infeasible_assignment.empty()) return;

  cuopt_assert(fj_cpu.checkpoint_severity >= fj_cpu.best_infeasible_severity,
               "checkpoint cannot beat the best severity seen");

  restart_from_infeasible_checkpoint(fj_cpu);

  ++fj_cpu.n_checkpoint_restores;
  ++fj_cpu.restores_since_improvement;
  if (fj_cpu.restores_since_improvement > fj_cpu.max_restores_since_improvement)
    fj_cpu.max_restores_since_improvement = fj_cpu.restores_since_improvement;
  fj_cpu.iters_since_infeasible_improve = 0;
}

template <typename i_t, typename f_t>
static void init_fj_cpu(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                        solution_t<i_t, f_t>& solution,
                        const std::vector<f_t>& left_weights,
                        const std::vector<f_t>& right_weights,
                        f_t objective_weight,
                        const probing_cache_t<i_t, f_t>* probing_cache)
{
  auto& problem   = *solution.problem_ptr;
  auto handle_ptr = solution.handle_ptr;

  auto sol_copy = solution;
  clamp_within_var_bounds(sol_copy.assignment, &problem, handle_ptr);

  // build a cpu-based fj_view_t
  fj_cpu.view    = typename fj_t<i_t, f_t>::climber_data_t::view_t{};
  fj_cpu.view.pb = problem.view();
  fj_cpu.pb_ptr  = &problem;
  // Get host copies of device data
  fj_cpu.h_reverse_coefficients =
    cuopt::host_copy(problem.reverse_coefficients, handle_ptr->get_stream());
  fj_cpu.h_reverse_constraints =
    cuopt::host_copy(problem.reverse_constraints, handle_ptr->get_stream());
  fj_cpu.h_reverse_offsets = cuopt::host_copy(problem.reverse_offsets, handle_ptr->get_stream());
  fj_cpu.h_coefficients    = cuopt::host_copy(problem.coefficients, handle_ptr->get_stream());
  fj_cpu.h_offsets         = cuopt::host_copy(problem.offsets, handle_ptr->get_stream());
  fj_cpu.h_variables       = cuopt::host_copy(problem.variables, handle_ptr->get_stream());
  fj_cpu.h_obj_coeffs = cuopt::host_copy(problem.objective_coefficients, handle_ptr->get_stream());
  fj_cpu.h_var_bounds = cuopt::host_copy(problem.variable_bounds, handle_ptr->get_stream());
  fj_cpu.h_cstr_lb    = cuopt::host_copy(problem.constraint_lower_bounds, handle_ptr->get_stream());
  fj_cpu.h_cstr_ub    = cuopt::host_copy(problem.constraint_upper_bounds, handle_ptr->get_stream());
  fj_cpu.h_var_types  = cuopt::host_copy(problem.variable_types, handle_ptr->get_stream());
  fj_cpu.h_is_binary_variable =
    cuopt::host_copy(problem.is_binary_variable, handle_ptr->get_stream());
  fj_cpu.h_binary_indices = cuopt::host_copy(problem.binary_indices, handle_ptr->get_stream());
  fj_cpu.h_related_variables =
    cuopt::host_copy(problem.related_variables, handle_ptr->get_stream());
  fj_cpu.h_related_variables_offsets =
    cuopt::host_copy(problem.related_variables_offsets, handle_ptr->get_stream());
  fj_cpu.probing_cache          = probing_cache;
  fj_cpu.h_original_ids         = problem.original_ids;
  fj_cpu.h_reverse_original_ids = problem.reverse_original_ids;

  fj_cpu.h_cstr_left_weights  = left_weights;
  fj_cpu.h_cstr_right_weights = right_weights;
  fj_cpu.max_weight           = 1.0;
  fj_cpu.h_objective_weight   = objective_weight;
  auto h_assignment           = sol_copy.get_host_assignment();
  fj_cpu.h_assignment         = h_assignment;
  fj_cpu.h_best_assignment    = std::move(h_assignment);
  fj_cpu.h_lhs.resize(fj_cpu.pb_ptr->n_constraints);
  fj_cpu.h_lhs_sumcomp.resize(fj_cpu.pb_ptr->n_constraints, 0);
  fj_cpu.h_tabu_nodec_until.resize(fj_cpu.pb_ptr->n_variables, 0);
  fj_cpu.h_tabu_noinc_until.resize(fj_cpu.pb_ptr->n_variables, 0);
  fj_cpu.h_tabu_lastdec.resize(fj_cpu.pb_ptr->n_variables, 0);
  fj_cpu.h_tabu_lastinc.resize(fj_cpu.pb_ptr->n_variables, 0);
  fj_cpu.iterations = 0;

  finalize_fj_cpu_host_initialization(fj_cpu,
                                      problem.n_variables,
                                      problem.n_constraints,
                                      problem.n_integer_vars,
                                      problem.nnz,
                                      problem.tolerances);
}

template <typename i_t, typename f_t>
static void init_fj_cpu_from_template(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                                      const fj_cpu_climber_t<i_t, f_t>& tmpl,
                                      const std::vector<f_t>& left_weights,
                                      const std::vector<f_t>& right_weights,
                                      f_t objective_weight)
{
  const i_t n_variables   = (i_t)tmpl.h_reverse_offsets.size() - 1;
  const i_t n_constraints = (i_t)tmpl.h_offsets.size() - 1;
  const i_t nnz           = (i_t)tmpl.h_coefficients.size();

  cuopt_assert(n_variables == tmpl.view.pb.n_variables, "template variable count mismatch");
  cuopt_assert(n_constraints == tmpl.view.pb.n_constraints, "template constraint count mismatch");
  cuopt_assert(nnz == tmpl.view.pb.nnz, "template nnz mismatch");
  cuopt_assert(left_weights.size() == static_cast<size_t>(n_constraints),
               "left weight size mismatch");
  cuopt_assert(right_weights.size() == static_cast<size_t>(n_constraints),
               "right weight size mismatch");

  fj_cpu.view = typename fj_t<i_t, f_t>::climber_data_t::view_t{};
  // Every span the host views cover is re-pointed at this climber's own arrays below. The rest of
  // the problem view carries over from the template, which is also what makes this usable on
  // climbers built without a problem_t at all.
  fj_cpu.view.pb = tmpl.view.pb;
  fj_cpu.pb_ptr  = tmpl.pb_ptr;

  fj_cpu.h_reverse_coefficients = tmpl.h_reverse_coefficients;
  fj_cpu.h_reverse_constraints  = tmpl.h_reverse_constraints;
  fj_cpu.h_reverse_offsets      = tmpl.h_reverse_offsets;
  fj_cpu.h_coefficients         = tmpl.h_coefficients;
  fj_cpu.h_offsets              = tmpl.h_offsets;
  fj_cpu.h_variables            = tmpl.h_variables;
  fj_cpu.h_obj_coeffs           = tmpl.h_obj_coeffs;
  fj_cpu.h_var_bounds           = tmpl.h_var_bounds;
  fj_cpu.h_cstr_lb              = tmpl.h_cstr_lb;
  fj_cpu.h_cstr_ub              = tmpl.h_cstr_ub;
  fj_cpu.h_var_types            = tmpl.h_var_types;
  fj_cpu.h_is_binary_variable   = tmpl.h_is_binary_variable;
  fj_cpu.h_binary_indices       = tmpl.h_binary_indices;

  fj_cpu.h_cstr_left_weights  = left_weights;
  fj_cpu.h_cstr_right_weights = right_weights;
  fj_cpu.max_weight           = 1.0;
  fj_cpu.h_objective_weight   = objective_weight;
  fj_cpu.h_assignment         = tmpl.h_assignment;
  fj_cpu.h_best_assignment    = tmpl.h_assignment;
  fj_cpu.h_tabu_nodec_until.resize(n_variables, 0);
  fj_cpu.h_tabu_noinc_until.resize(n_variables, 0);
  fj_cpu.h_tabu_lastdec.resize(n_variables, 0);
  fj_cpu.h_tabu_lastinc.resize(n_variables, 0);
  fj_cpu.iterations = 0;

  finalize_fj_cpu_host_initialization_from_template(fj_cpu,
                                                    tmpl,
                                                    n_variables,
                                                    n_constraints,
                                                    tmpl.n_integer_vars,
                                                    nnz,
                                                    tmpl.view.pb.tolerances);
}

template <typename i_t, typename f_t>
static void set_host_data_view(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu,
  i_t n_variables,
  i_t n_constraints,
  i_t n_integer_vars,
  i_t nnz,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances)
{
  fj_cpu.view.pb.tolerances     = tolerances;
  fj_cpu.view.pb.n_variables    = n_variables;
  fj_cpu.view.pb.n_integer_vars = n_integer_vars;
  fj_cpu.view.pb.n_constraints  = n_constraints;
  fj_cpu.view.pb.nnz            = nnz;

  fj_cpu.view.pb.constraint_lower_bounds =
    raft::device_span<f_t>(fj_cpu.h_cstr_lb.data(), fj_cpu.h_cstr_lb.size());
  fj_cpu.view.pb.constraint_upper_bounds =
    raft::device_span<f_t>(fj_cpu.h_cstr_ub.data(), fj_cpu.h_cstr_ub.size());
  fj_cpu.view.pb.variable_bounds = raft::device_span<typename type_2<f_t>::type>(
    fj_cpu.h_var_bounds.data(), fj_cpu.h_var_bounds.size());
  fj_cpu.view.pb.variable_types =
    raft::device_span<var_t>(fj_cpu.h_var_types.data(), fj_cpu.h_var_types.size());
  fj_cpu.view.pb.is_binary_variable =
    raft::device_span<i_t>(fj_cpu.h_is_binary_variable.data(), fj_cpu.h_is_binary_variable.size());
  fj_cpu.view.pb.binary_indices =
    raft::device_span<i_t>(fj_cpu.h_binary_indices.data(), fj_cpu.h_binary_indices.size());
  fj_cpu.view.pb.coefficients =
    raft::device_span<f_t>(fj_cpu.h_coefficients.data(), fj_cpu.h_coefficients.size());
  fj_cpu.view.pb.offsets = raft::device_span<i_t>(fj_cpu.h_offsets.data(), fj_cpu.h_offsets.size());
  fj_cpu.view.pb.variables =
    raft::device_span<i_t>(fj_cpu.h_variables.data(), fj_cpu.h_variables.size());
  fj_cpu.view.pb.reverse_coefficients = raft::device_span<f_t>(
    fj_cpu.h_reverse_coefficients.data(), fj_cpu.h_reverse_coefficients.size());
  fj_cpu.view.pb.reverse_constraints = raft::device_span<i_t>(fj_cpu.h_reverse_constraints.data(),
                                                              fj_cpu.h_reverse_constraints.size());
  fj_cpu.view.pb.reverse_offsets =
    raft::device_span<i_t>(fj_cpu.h_reverse_offsets.data(), fj_cpu.h_reverse_offsets.size());
  fj_cpu.view.pb.objective_coefficients =
    raft::device_span<f_t>(fj_cpu.h_obj_coeffs.data(), fj_cpu.h_obj_coeffs.size());
}

template <typename i_t, typename f_t>
static void wire_fj_cpu_host_views(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu,
  i_t n_variables,
  i_t n_constraints,
  i_t n_integer_vars,
  i_t nnz,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances)
{
  cuopt_assert(n_variables >= 0, "invalid variable count");
  cuopt_assert(n_constraints >= 0, "invalid constraint count");
  cuopt_assert(fj_cpu.h_offsets.size() == static_cast<size_t>(n_constraints + 1),
               "invalid CSR offsets");
  cuopt_assert(fj_cpu.h_reverse_offsets.size() == static_cast<size_t>(n_variables + 1),
               "invalid reverse offsets");
  cuopt_assert(fj_cpu.h_assignment.size() == static_cast<size_t>(n_variables),
               "seed assignment size mismatch");

  set_host_data_view(fj_cpu, n_variables, n_constraints, n_integer_vars, nnz, tolerances);

  fj_cpu.view.cstr_left_weights =
    raft::device_span<f_t>(fj_cpu.h_cstr_left_weights.data(), fj_cpu.h_cstr_left_weights.size());
  fj_cpu.view.cstr_right_weights =
    raft::device_span<f_t>(fj_cpu.h_cstr_right_weights.data(), fj_cpu.h_cstr_right_weights.size());
  fj_cpu.view.objective_weight = &fj_cpu.h_objective_weight;
  fj_cpu.view.incumbent_assignment =
    raft::device_span<f_t>(fj_cpu.h_assignment.data(), fj_cpu.h_assignment.size());
  fj_cpu.view.incumbent_lhs = raft::device_span<f_t>(fj_cpu.h_lhs.data(), fj_cpu.h_lhs.size());
  fj_cpu.view.incumbent_lhs_sumcomp =
    raft::device_span<f_t>(fj_cpu.h_lhs_sumcomp.data(), fj_cpu.h_lhs_sumcomp.size());
  fj_cpu.view.tabu_nodec_until =
    raft::device_span<i_t>(fj_cpu.h_tabu_nodec_until.data(), fj_cpu.h_tabu_nodec_until.size());
  fj_cpu.view.tabu_noinc_until =
    raft::device_span<i_t>(fj_cpu.h_tabu_noinc_until.data(), fj_cpu.h_tabu_noinc_until.size());
  fj_cpu.view.tabu_lastdec =
    raft::device_span<i_t>(fj_cpu.h_tabu_lastdec.data(), fj_cpu.h_tabu_lastdec.size());
  fj_cpu.view.tabu_lastinc =
    raft::device_span<i_t>(fj_cpu.h_tabu_lastinc.data(), fj_cpu.h_tabu_lastinc.size());
  fj_cpu.view.incumbent_objective = &fj_cpu.h_incumbent_objective;
  fj_cpu.view.best_objective      = &fj_cpu.h_best_objective;
  fj_cpu.view.settings            = &fj_cpu.settings;

  fj_cpu.h_best_objective = +std::numeric_limits<f_t>::infinity();

  // nnz count
  fj_cpu.cached_mtm_moves.resize(fj_cpu.h_coefficients.size(),
                                 std::make_pair(0, fj_staged_score_t::zero()));
  fj_cpu.cached_mtm_moves_version.assign(fj_cpu.h_coefficients.size(), -1);
  fj_cpu.h_cstr_version.assign(n_constraints, 0);

  fj_cpu.flip_move_computed.resize(n_variables, false);
  fj_cpu.var_bitmap.resize(n_variables, false);
  fj_cpu.iter_mtm_vars.reserve(n_variables);
}

template <typename i_t, typename f_t>
void finalize_fj_cpu_host_initialization(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu,
  i_t n_variables,
  i_t n_constraints,
  i_t n_integer_vars,
  i_t nnz,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances)
{
  raft::common::nvtx::range scope("finalize_fj_cpu_host_initialization");

  wire_fj_cpu_host_views(fj_cpu, n_variables, n_constraints, n_integer_vars, nnz, tolerances);

  fj_cpu.h_objective_vars.resize(n_variables);
  auto end = std::copy_if(
    thrust::counting_iterator<i_t>(0),
    thrust::counting_iterator<i_t>(n_variables),
    fj_cpu.h_objective_vars.begin(),
    [&fj_cpu](i_t idx) { return !fj_cpu.view.pb.integer_equal(fj_cpu.h_obj_coeffs[idx], (f_t)0); });
  fj_cpu.h_objective_vars.resize(end - fj_cpu.h_objective_vars.begin());
  fj_cpu.view.objective_vars =
    raft::device_span<i_t>(fj_cpu.h_objective_vars.data(), fj_cpu.h_objective_vars.size());

  f_t abs_obj_sum = 0;
  for (auto var_idx : fj_cpu.h_objective_vars) {
    const f_t coeff = fj_cpu.h_obj_coeffs[var_idx];
    abs_obj_sum += coeff < 0 ? -coeff : coeff;
  }
  fj_cpu.obj_magnitude = abs_obj_sum > 0 ? abs_obj_sum / fj_cpu.h_objective_vars.size() : f_t{1};
  cuopt_assert(isfinite(fj_cpu.obj_magnitude) && fj_cpu.obj_magnitude > 0,
               "objective magnitude unit must be finite and positive");

  fj_cpu.cached_cstr_bounds.resize(fj_cpu.h_reverse_coefficients.size());
  for (i_t var_idx = 0; var_idx < n_variables; ++var_idx) {
    auto [offset_begin, offset_end] = reverse_range_for_var<i_t, f_t>(fj_cpu, var_idx);
    for (i_t i = offset_begin; i < offset_end; ++i) {
      fj_cpu.cached_cstr_bounds[i] =
        std::make_pair(fj_cpu.h_cstr_lb[fj_cpu.h_reverse_constraints[i]],
                       fj_cpu.h_cstr_ub[fj_cpu.h_reverse_constraints[i]]);
    }
  }

  // precompute the binvars-pre-row tables for 2opt
  fj_cpu.h_binrow_offsets.resize(n_constraints + 1);
  fj_cpu.h_binrow_vars.clear();
  for (i_t cstr_idx = 0; cstr_idx < n_constraints; ++cstr_idx) {
    fj_cpu.h_binrow_offsets[cstr_idx] = fj_cpu.h_binrow_vars.size();
    auto [offset_begin, offset_end]   = range_for_constraint<i_t, f_t>(fj_cpu, cstr_idx);
    for (i_t i = offset_begin; i < offset_end; ++i) {
      const i_t var_idx = fj_cpu.h_variables[i];
      if (fj_cpu.h_is_binary_variable[var_idx]) { fj_cpu.h_binrow_vars.push_back(var_idx); }
    }
  }
  fj_cpu.h_binrow_offsets[n_constraints] = fj_cpu.h_binrow_vars.size();

  fj_cpu.flip_move_computed.resize(n_variables, false);
  fj_cpu.var_bitmap.resize(n_variables, false);
  fj_cpu.iter_mtm_vars.reserve(n_variables);

  // Must precede recompute_lhs, which is what first populates them.
  fj_cpu.violated_constraints.resize(n_constraints);
  fj_cpu.satisfied_constraints.resize(n_constraints);

  recompute_lhs(fj_cpu);

  // Precompute static problem features for regression model
  precompute_problem_features(fj_cpu);
}

template <typename i_t, typename f_t>
static void finalize_fj_cpu_host_initialization_from_template(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu,
  const fj_cpu_climber_t<i_t, f_t>& tmpl,
  i_t n_variables,
  i_t n_constraints,
  i_t n_integer_vars,
  i_t nnz,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances)
{
  raft::common::nvtx::range scope("finalize_fj_cpu_host_initialization_from_template");

  cuopt_assert(tmpl.h_lhs.size() == static_cast<size_t>(n_constraints), "template lhs mismatch");
  cuopt_assert(tmpl.violated_constraints.max_size() == n_constraints,
               "template violated set mismatch");
  cuopt_assert(tmpl.satisfied_constraints.max_size() == n_constraints,
               "template satisfied set mismatch");
  cuopt_assert(tmpl.cached_cstr_bounds.size() == fj_cpu.h_reverse_coefficients.size(),
               "template cached bounds mismatch");

  fj_cpu.h_objective_vars   = tmpl.h_objective_vars;
  fj_cpu.cached_cstr_bounds = tmpl.cached_cstr_bounds;
  fj_cpu.obj_magnitude      = tmpl.obj_magnitude;

  fj_cpu.h_lhs                 = tmpl.h_lhs;
  fj_cpu.h_lhs_sumcomp         = tmpl.h_lhs_sumcomp;
  fj_cpu.violated_constraints  = tmpl.violated_constraints;
  fj_cpu.satisfied_constraints = tmpl.satisfied_constraints;
  fj_cpu.total_violations      = tmpl.total_violations;
  fj_cpu.h_incumbent_objective = tmpl.h_incumbent_objective;
  fj_cpu.h_objective_sumcomp   = tmpl.h_objective_sumcomp;

  fj_cpu.n_binary_vars   = tmpl.n_binary_vars;
  fj_cpu.n_integer_vars  = tmpl.n_integer_vars;
  fj_cpu.avg_var_degree  = tmpl.avg_var_degree;
  fj_cpu.max_var_degree  = tmpl.max_var_degree;
  fj_cpu.var_degree_cv   = tmpl.var_degree_cv;
  fj_cpu.avg_cstr_degree = tmpl.avg_cstr_degree;
  fj_cpu.max_cstr_degree = tmpl.max_cstr_degree;
  fj_cpu.cstr_degree_cv  = tmpl.cstr_degree_cv;
  fj_cpu.problem_density = tmpl.problem_density;

  wire_fj_cpu_host_views(fj_cpu, n_variables, n_constraints, n_integer_vars, nnz, tolerances);
  fj_cpu.view.objective_vars =
    raft::device_span<i_t>(fj_cpu.h_objective_vars.data(), fj_cpu.h_objective_vars.size());
}

template <typename i_t, typename f_t>
static std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> init_fj_cpu_from_host_lp(
  const lp_problem_t<i_t, f_t>& problem,
  const std::vector<variable_type_t>& variable_types,
  const std::vector<f_t>& seed_assignment,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  std::atomic<bool>& preemption_flag,
  int64_t seed)
{
  using f_t2 = typename type_2<f_t>::type;

  cuopt_assert(variable_types.size() >= static_cast<size_t>(problem.num_cols),
               "variable type size mismatch");

  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances{};
  tolerances.absolute_tolerance    = settings.primal_tol;
  tolerances.relative_tolerance    = settings.zero_tol;
  tolerances.integrality_tolerance = settings.integer_tol;
  tolerances.absolute_mip_gap      = settings.absolute_mip_gap_tol;
  tolerances.relative_mip_gap      = settings.relative_mip_gap_tol;

  const i_t n_variables   = problem.num_cols;
  const i_t n_constraints = problem.num_rows;

  csr_matrix_t<i_t, f_t> csr_A(problem.num_rows, problem.num_cols, problem.A.nnz());
  problem.A.to_compressed_row(csr_A);
  std::vector<f_t> coefficients            = csr_A.x;
  std::vector<i_t> variables               = csr_A.j;
  std::vector<i_t> offsets                 = csr_A.row_start;
  std::vector<f_t> constraint_lower_bounds = problem.rhs;
  std::vector<f_t> constraint_upper_bounds = problem.rhs;
  std::vector<f_t2> variable_bounds(n_variables);
  std::vector<var_t> cpufj_variable_types(n_variables);
  std::vector<i_t> is_binary_variable(n_variables, 0);
  i_t n_integer_vars = 0;

  for (i_t j = 0; j < n_variables; ++j) {
    variable_bounds[j]  = f_t2{problem.lower[j], problem.upper[j]};
    const auto var_type = variable_types[j];
    cpufj_variable_types[j] =
      var_type == variable_type_t::CONTINUOUS ? var_t::CONTINUOUS : var_t::INTEGER;

    const bool is_integer = cpufj_variable_types[j] == var_t::INTEGER;
    const bool is_binary  = is_integer &&
                           integer_equal<f_t>(problem.lower[j], f_t{0}, settings.integer_tol) &&
                           integer_equal<f_t>(problem.upper[j], f_t{1}, settings.integer_tol);
    if (is_integer) { ++n_integer_vars; }
    if (is_binary) { is_binary_variable[j] = 1; }
  }

  const i_t nnz = static_cast<i_t>(variables.size());
  csc_matrix_t<i_t, f_t> reverse_csc(n_constraints, n_variables, nnz);
  csr_A.to_compressed_col(reverse_csc);
  std::vector<f_t> reverse_coefficients = std::move(reverse_csc.x);
  std::vector<i_t> reverse_constraints  = std::move(reverse_csc.i);
  std::vector<i_t> reverse_offsets      = std::move(reverse_csc.col_start);

  std::vector<f_t> projected_seed(n_variables, f_t{0});
  for (i_t j = 0; j < n_variables; ++j) {
    f_t value = j < static_cast<i_t>(seed_assignment.size()) ? seed_assignment[j] : f_t{0};
    value     = std::clamp(value, problem.lower[j], problem.upper[j]);
    if (variable_types[j] != variable_type_t::CONTINUOUS) {
      value = std::clamp(std::round(value), problem.lower[j], problem.upper[j]);
    }
    projected_seed[j] = value;
  }

  fj_settings_t fj_settings;
  fj_settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj_settings.n_of_minimums_for_exit = std::numeric_limits<int>::max();
  fj_settings.time_limit             = std::numeric_limits<f_t>::infinity();
  fj_settings.iteration_limit        = std::numeric_limits<int>::max();
  fj_settings.update_weights         = true;
  fj_settings.feasibility_run        = false;
  fj_settings.seed                   = seed >= 0 ? seed : cuopt::seed_generator::get_seed();

  auto fj_cpu      = std::make_unique<fj_cpu_climber_t<i_t, f_t>>(preemption_flag);
  fj_cpu->view     = typename fj_t<i_t, f_t>::climber_data_t::view_t{};
  fj_cpu->pb_ptr   = nullptr;
  fj_cpu->settings = fj_settings;

  fj_cpu->h_reverse_coefficients = std::move(reverse_coefficients);
  fj_cpu->h_reverse_constraints  = std::move(reverse_constraints);
  fj_cpu->h_reverse_offsets      = std::move(reverse_offsets);
  fj_cpu->h_coefficients         = std::move(coefficients);
  fj_cpu->h_offsets              = std::move(offsets);
  fj_cpu->h_variables            = std::move(variables);
  fj_cpu->h_obj_coeffs           = problem.objective;
  fj_cpu->h_var_bounds           = std::move(variable_bounds);
  fj_cpu->h_cstr_lb              = std::move(constraint_lower_bounds);
  fj_cpu->h_cstr_ub              = std::move(constraint_upper_bounds);
  fj_cpu->h_var_types            = std::move(cpufj_variable_types);
  fj_cpu->h_is_binary_variable   = std::move(is_binary_variable);

  fj_cpu->h_cstr_left_weights.resize(n_constraints, 1.0);
  fj_cpu->h_cstr_right_weights.resize(n_constraints, 1.0);
  fj_cpu->max_weight         = 1.0;
  fj_cpu->h_objective_weight = 0.0;
  fj_cpu->h_assignment       = projected_seed;
  fj_cpu->h_best_assignment  = std::move(projected_seed);
  fj_cpu->h_lhs.resize(n_constraints);
  fj_cpu->h_lhs_sumcomp.resize(n_constraints, 0);
  fj_cpu->h_tabu_nodec_until.resize(n_variables, 0);
  fj_cpu->h_tabu_noinc_until.resize(n_variables, 0);
  fj_cpu->h_tabu_lastdec.resize(n_variables, 0);
  fj_cpu->h_tabu_lastinc.resize(n_variables, 0);
  fj_cpu->iterations = 0;

  finalize_fj_cpu_host_initialization(
    *fj_cpu, n_variables, n_constraints, n_integer_vars, nnz, tolerances);
  return fj_cpu;
}

template <typename i_t, typename f_t>
static void sanity_checks(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  // Check that each variable is within its bounds
  for (i_t var_idx = 0; var_idx < fj_cpu.view.pb.n_variables; ++var_idx) {
    f_t val = fj_cpu.h_assignment[var_idx];
    cuopt_assert(fj_cpu.view.pb.check_variable_within_bounds(var_idx, val),
                 "Variable is out of bounds");
  }

  // Check that each violated constraint is actually violated and not present in
  // satisfied_constraints
  for (const auto& cstr_idx : fj_cpu.violated_constraints) {
    cuopt_assert(!fj_cpu.satisfied_constraints.contains(cstr_idx),
                 "Violated constraint also in satisfied_constraints");
    f_t lhs    = fj_cpu.h_lhs[cstr_idx];
    f_t tol    = fj_cpu.view.get_corrected_tolerance(cstr_idx);
    f_t excess = fj_cpu.view.excess_score(cstr_idx, lhs);
    cuopt_assert(excess < -tol, "Constraint in violated_constraints is not actually violated");
  }

  // Check that each satisfied constraint is actually satisfied and not present in
  // violated_constraints
  for (const auto& cstr_idx : fj_cpu.satisfied_constraints) {
    cuopt_assert(!fj_cpu.violated_constraints.contains(cstr_idx),
                 "Satisfied constraint also in violated_constraints");
    f_t lhs    = fj_cpu.h_lhs[cstr_idx];
    f_t tol    = fj_cpu.view.get_corrected_tolerance(cstr_idx);
    f_t excess = fj_cpu.view.excess_score(cstr_idx, lhs);
    cuopt_assert(!(excess < -tol), "Constraint in satisfied_constraints is actually violated");
  }

  // Check that each constraint is in exactly one of violated_constraints or satisfied_constraints
  for (i_t cstr_idx = 0; cstr_idx < fj_cpu.view.pb.n_constraints; ++cstr_idx) {
    bool in_viol = fj_cpu.violated_constraints.contains(cstr_idx);
    bool in_sat  = fj_cpu.satisfied_constraints.contains(cstr_idx);
    cuopt_assert(
      in_viol != in_sat,
      "Constraint must be in exactly one of violated_constraints or satisfied_constraints");

    cuopt_assert(fj_cpu.h_cstr_left_weights[cstr_idx] >= 0, "Weights should be positive or zero");
    cuopt_assert(fj_cpu.h_cstr_right_weights[cstr_idx] >= 0, "Weights should be positive or zero");
  }
  cuopt_assert(fj_cpu.h_objective_weight >= 0, "Objective weight should be positive or zero");
  cuopt_assert(fj_cpu.seed_objective_weight >= 0,
               "Objective weight floor should be positive or zero");
}

template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> fj_t<i_t, f_t>::create_cpu_climber(
  solution_t<i_t, f_t>& solution,
  const std::vector<f_t>& left_weights,
  const std::vector<f_t>& right_weights,
  f_t objective_weight,
  std::atomic<bool>& preemption_flag,
  const probing_cache_t<i_t, f_t>* probing_cache,
  fj_settings_t settings,
  bool randomize_params)
{
  raft::common::nvtx::range scope("fj_cpu_init");

  auto fj_cpu = std::make_unique<fj_cpu_climber_t<i_t, f_t>>(preemption_flag);

  // Initialize fj_cpu with all the data
  init_fj_cpu(*fj_cpu, solution, left_weights, right_weights, objective_weight, probing_cache);
  fj_cpu->settings = settings;
  if (randomize_params) {
    auto rng                 = std::mt19937(cuopt::seed_generator::get_seed());
    fj_cpu->mtm_viol_samples = std::uniform_int_distribution<i_t>(15, 50)(rng);
    fj_cpu->mtm_sat_samples  = std::uniform_int_distribution<i_t>(10, 30)(rng);
    fj_cpu->nnz_samples      = std::uniform_int_distribution<i_t>(2000, 15000)(rng);
    fj_cpu->perturb_interval = std::uniform_int_distribution<i_t>(50, 500)(rng);
  }
  fj_cpu->settings.seed = cuopt::seed_generator::get_seed();
  return fj_cpu;  // move
}

constexpr int32_t fj_nnz_per_refresh_stretch = 100000;
constexpr int32_t fj_max_refresh_stretch     = 8;

// Above this a short LP spends more time moving the matrix than it can pay back as a seed, and the
// wall budget the LP is allowed out of the lane's own.
constexpr int64_t fj_lp_seed_nnz_limit = 8'000'000;
constexpr double fj_lp_seed_budget_s   = 2;

// Rounds the LP relaxation into this lane's start point. Solved by dual simplex on the lane's own
// thread rather than during portfolio construction, so the other seven start searching immediately.
template <typename i_t, typename f_t>
static void apply_lp_rounded_seed(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  if (!fj_cpu.use_lp_seed || fj_cpu.pb_ptr == nullptr) return;
  if (fj_cpu.view.pb.nnz > fj_lp_seed_nnz_limit) return;

  simplex::user_problem_t<i_t, f_t> relaxation(fj_cpu.pb_ptr->handle_ptr);
  fj_cpu.pb_ptr->get_host_user_problem(relaxation);

  simplex_solver_settings_t<i_t, f_t> lp_settings;
  lp_settings.relaxation = true;
  lp_settings.time_limit = fj_lp_seed_budget_s;
  lp_settings.log.log    = false;

  std::vector<f_t> relaxed;
  simplex::lp_status_t lp_status = simplex::lp_status_t::UNSET;
  double lp_seconds              = 0;

  // solve_linear_program_advanced rather than simplex::solve, whose collapsed int return cannot
  // separate a limit -- which leaves a usable vertex behind -- from infeasibility, which does not.
  // Guarded on f_t because dual simplex is only instantiated for double.
  if constexpr (std::is_same_v<f_t, double>) {
    const f_t lp_start = tic();
    lp_problem_t<i_t, f_t> converted(relaxation.handle_ptr,
                                     relaxation.num_rows,
                                     relaxation.num_cols,
                                     relaxation.A.col_start[relaxation.A.n]);
    std::vector<i_t> new_slacks;
    simplex::dualize_info_t<i_t, f_t> dualize_info;
    simplex::convert_user_problem(relaxation, lp_settings, converted, new_slacks, dualize_info);

    simplex::lp_solution_t<i_t, f_t> lp_solution(converted.num_rows, converted.num_cols);
    std::vector<simplex::variable_status_t> vstatus;
    std::vector<f_t> edge_norms;
    lp_status = simplex::solve_linear_program_advanced(
      converted, lp_start, lp_settings, lp_solution, vstatus, edge_norms);
    relaxed    = lp_solution.x;
    lp_seconds = toc(lp_start);
  }

  // A vertex reached at a limit is dual feasible and still worth rounding. The remaining
  // terminations leave nothing to round.
  const bool usable = lp_status == simplex::lp_status_t::OPTIMAL ||
                      lp_status == simplex::lp_status_t::TIME_LIMIT ||
                      lp_status == simplex::lp_status_t::ITERATION_LIMIT ||
                      lp_status == simplex::lp_status_t::CONCURRENT_LIMIT ||
                      lp_status == simplex::lp_status_t::WORK_LIMIT;
  CUOPT_LOG_DEBUG("%sCPUFJ LP seed: %s after %.3fs of %.3fs%s",
                  fj_cpu.log_prefix.c_str(),
                  simplex::lp_status_to_string(lp_status).c_str(),
                  lp_seconds,
                  fj_lp_seed_budget_s,
                  usable ? "" : ", discarded");
  if (!usable) return;

  const i_t n_variables = fj_cpu.view.pb.n_variables;
  // convert_user_problem appends slacks, so the model's own variables are the leading columns.
  cuopt_assert((i_t)relaxed.size() >= n_variables, "dual simplex returned too few columns");

  std::vector<f_t> candidate(n_variables);
  for (i_t var = 0; var < n_variables; ++var) {
    cuopt_assert(isfinite(relaxed[var]), "dual simplex returned a non-finite value");
    const auto bounds = fj_cpu.h_var_bounds[var].get();
    const f_t lower   = get_lower(bounds);
    const f_t upper   = get_upper(bounds);
    f_t value         = std::clamp(relaxed[var], lower, upper);
    if (is_integer_var<i_t, f_t>(fj_cpu, var)) {
      value = round(value);
      // Rounding can leave the bounds, and a variable with no integral value inside them cannot be
      // seeded at all without breaking the engine's integrality invariant.
      if (value < lower || value > upper) return;
    }
    candidate[var] = value;
  }

  fj_cpu.h_assignment      = candidate;
  fj_cpu.h_best_assignment = candidate;
  recompute_lhs(fj_cpu);

  // The rounded point can already be integral-feasible. It never passed through apply_move, so the
  // incumbent is recorded here through the same contract that path uses.
  if (fj_cpu.violated_constraints.empty() && check_variable_feasibility<i_t, f_t>(fj_cpu)) {
    fj_cpu.h_best_objective =
      fj_cpu.h_incumbent_objective - fj_cpu.settings.parameters.breakthrough_move_epsilon;
    fj_cpu.feasible_found = true;
    CUOPT_LOG_DEBUG("%sCPUFJ new incumbent: objective %.17g",
                    fj_cpu.log_prefix.c_str(),
                    fj_cpu.h_best_objective);
    if (fj_cpu.improvement_callback) {
      fj_cpu.improvement_callback(fj_cpu.h_incumbent_objective,
                                  fj_cpu.h_assignment,
                                  fj_cpu.work_units_elapsed.load(std::memory_order_acquire));
    }
    if (fj_cpu.shared_incumbent) {
      fj_cpu.shared_incumbent->publish(fj_cpu.h_incumbent_objective, fj_cpu.h_assignment);
    }
  }
}

template <typename i_t, typename f_t>
void cpufj_solve(fj_cpu_climber_t<i_t, f_t>* fj_cpu, f_t in_time_limit, double work_unit_limit)
{
  const auto solve_start = std::chrono::high_resolution_clock::now();
  // problem fits the binary fastpath shape? run it (engine is solve-local)
  if (try_cpufj_binary_solve(*fj_cpu, in_time_limit, work_unit_limit)) return;

  apply_lp_rounded_seed(*fj_cpu);

  i_t local_mins = 0;
  // The LP comes out of this lane's own budget; every other lane's clock starts where it did.
  auto loop_start = fj_cpu->use_lp_seed ? solve_start : std::chrono::high_resolution_clock::now();
  auto time_limit = std::chrono::milliseconds(static_cast<i_t>(std::floor(in_time_limit * 1000.0)));
  auto loop_time_start = loop_start;

  fj_cpu->rng.seed(fj_cpu->settings.seed);

  // Initialize feature tracking
  fj_cpu->last_feature_log_time = loop_start;
  fj_cpu->prev_best_objective   = fj_cpu->h_best_objective;
  fj_cpu->iterations_since_best = 0;
  reset_infeasible_checkpoint(*fj_cpu);
  fj_cpu->n_checkpoint_restores          = 0;
  fj_cpu->n_checkpoint_snapshots         = 0;
  fj_cpu->restores_since_improvement     = 0;
  fj_cpu->max_restores_since_improvement = 0;

  // The recompute is O(nnz), so a fixed period costs a growing share of the budget.
  cuopt_assert(fj_cpu->settings.parameters.lhs_refresh_period > 0,
               "lhs_refresh_period should be positive");
  const i_t nnz_stretch    = std::min<i_t>(
    (i_t)fj_cpu->h_coefficients.size() / fj_nnz_per_refresh_stretch, fj_max_refresh_stretch);
  const i_t refresh_period = fj_cpu->settings.parameters.lhs_refresh_period * (1 + nnz_stretch);
  cuopt_assert(refresh_period > 0, "refresh period overflowed");

  while (!fj_cpu->halted && !fj_cpu->preemption_flag.load()) {
    // Check if 5 seconds have passed
    auto now = std::chrono::high_resolution_clock::now();
    if (in_time_limit < std::numeric_limits<f_t>::infinity() &&
        now - loop_time_start > time_limit) {
      CUOPT_LOG_TRACE("%sTime limit of %.4f seconds reached, breaking loop at iteration %d",
                      fj_cpu->log_prefix.c_str(),
                      time_limit.count() / 1000.f,
                      fj_cpu->iterations);
      break;
    }
    if (fj_cpu->iterations >= fj_cpu->settings.iteration_limit) {
      CUOPT_LOG_TRACE("%sIteration limit of %d reached, breaking loop at iteration %d",
                      fj_cpu->log_prefix.c_str(),
                      fj_cpu->settings.iteration_limit,
                      fj_cpu->iterations);
      break;
    }

    // periodically recompute the LHS and violation scores
    // to correct any accumulated numerical errors
    if (fj_cpu->iterations % refresh_period == 0 || fj_cpu->trigger_early_lhs_recomputation) {
      recompute_lhs(*fj_cpu);
      fj_cpu->trigger_early_lhs_recomputation = false;
    }

    fj_move_t move          = fj_move_t{-1, 0};
    fj_staged_score_t score = fj_staged_score_t::invalid();
    bool is_lift            = false;
    bool is_mtm_viol        = false;
    bool is_mtm_sat         = false;

    // Perform lift moves
    fj_move_t lift_companion = fj_move_t{-1, 0};
    if (fj_cpu->violated_constraints.empty()) {
      thrust::tie(move, score) = find_lift_move(*fj_cpu);
      if (score > fj_staged_score_t::zero()) {
        is_lift = true;
      } else {
        // Pairs are only reachable once no single improving flip preserves feasibility.
        fj_move_t first, second;
        fj_staged_score_t pair_score;
        thrust::tie(first, second, pair_score) = find_lift_2opt_move(*fj_cpu);
        if (pair_score > fj_staged_score_t::zero()) {
          move           = first;
          lift_companion = second;
          score          = pair_score;
          is_lift        = true;
        }
      }
    }
    // Regular MTM
    if (!(score > fj_staged_score_t::zero())) {
      thrust::tie(move, score) = find_mtm_move_viol(*fj_cpu, fj_cpu->mtm_viol_samples);
      if (score > fj_staged_score_t::zero()) is_mtm_viol = true;
    }
    // try with MTM in satisfied constraints
    if (fj_cpu->feasible_found && !(score > fj_staged_score_t::zero())) {
      thrust::tie(move, score) = find_mtm_move_sat(*fj_cpu, fj_cpu->mtm_sat_samples);
      if (score > fj_staged_score_t::zero()) is_mtm_sat = true;
    }
    // if we're in the feasible region but haven't found improvements in the last n iterations,
    // perturb
    bool should_perturb = false;
    if (fj_cpu->violated_constraints.empty() &&
        fj_cpu->iterations_since_best > fj_cpu->perturb_interval) {
      should_perturb = true;
      // Without this the counter stays above the interval and every later iteration perturbs.
      fj_cpu->iterations_since_best = 0;
    }

    if (score > fj_staged_score_t::zero() && !should_perturb) {
      apply_move(*fj_cpu, move.var_idx, move.value, false);
      if (lift_companion.var_idx >= 0) {
        apply_move(*fj_cpu, lift_companion.var_idx, lift_companion.value, false);
        fj_cpu->n_lift_moves_window++;
      }
      // Track move types
      if (is_lift) fj_cpu->n_lift_moves_window++;
      if (is_mtm_viol) fj_cpu->n_mtm_viol_moves_window++;
      if (is_mtm_sat) fj_cpu->n_mtm_sat_moves_window++;
    } else {
      // Local Min
      update_weights(*fj_cpu);
      track_infeasible_checkpoint(*fj_cpu);
      if (should_perturb) {
        perturb(*fj_cpu);
        invalidate_mtm_cache(*fj_cpu);
      }

      two_opt_move_t two_opt_move;
      if (!should_perturb) two_opt_move = find_two_opt_move(*fj_cpu);
      if (two_opt_move.score > fj_staged_score_t::zero()) {
        apply_move(*fj_cpu, two_opt_move.first.var_idx, two_opt_move.first.value, true);
        apply_move(*fj_cpu, two_opt_move.second.var_idx, two_opt_move.second.value, true);
        fj_cpu->n_mtm_viol_moves_window += 2;
      } else {
        thrust::tie(move, score) =
          find_mtm_move_viol(*fj_cpu, 1, true);  // pick a single random violated constraint
        i_t var_idx = move.var_idx >= 0 ? move.var_idx : 0;
        f_t delta   = move.var_idx >= 0 ? move.value : 0;
        apply_move(*fj_cpu, var_idx, delta, true);
      }
      ++local_mins;
      ++fj_cpu->n_local_minima_window;
    }

    // number of violated constraints is usually small (<100). recomputing from all LHSs is cheap
    // and more numerically precise than just adding to the accumulator in apply_move
    fj_cpu->total_violations = 0;
    for (auto cstr_idx : fj_cpu->violated_constraints) {
      fj_cpu->total_violations += fj_cpu->view.excess_score(cstr_idx, fj_cpu->h_lhs[cstr_idx]);
    }
    if (fj_cpu->iterations % fj_cpu->log_interval == 0) {
      CUOPT_LOG_DEBUG(
        "%sCPUFJ iteration: %d/%d, local mins: %d, best_objective: %g, viol: %zu, obj weight %g, "
        "maxw %g",
        fj_cpu->log_prefix.c_str(),
        fj_cpu->iterations,
        fj_cpu->settings.iteration_limit != std::numeric_limits<i_t>::max()
          ? fj_cpu->settings.iteration_limit
          : -1,
        local_mins,
        fj_cpu->h_best_objective,
        fj_cpu->violated_constraints.size(),
        fj_cpu->h_objective_weight,
        fj_cpu->max_weight);
    }
    // send current solution to callback every 3000 steps for diversity
    if (fj_cpu->iterations % fj_cpu->diversity_callback_interval == 0) {
      if (fj_cpu->diversity_callback) {
        fj_cpu->diversity_callback(fj_cpu->h_incumbent_objective, fj_cpu->h_assignment);
      }
    }

    // Print timing statistics every N iterations
#if CPUFJ_TIMING_TRACE
    if (fj_cpu->iterations % fj_cpu->timing_stats_interval == 0 && fj_cpu->iterations > 0) {
      print_timing_stats(*fj_cpu);
    }
#endif

    if (fj_cpu->iterations % 100 == 0 && fj_cpu->iterations > 0) {
      // Use cumulative byte counts (collect() without flush). Each window's contribution to
      // work_units_elapsed therefore grows roughly with the running total of bytes touched,
      // i.e. quadratically in iterations rather than linearly. This is intentional: the
      // memory_aggregator is calibrated for medium/large MIPs, and a strictly-linear scheme
      // forces tiny instances (few KB per iteration) to run for tens of seconds before the
      // accumulated bytes cross a 0.5 horizon, causing the deterministic producer_sync to
      // stall and B&B to time out on instances that should solve in milliseconds. The
      // accumulation is still deterministic across runs of the same problem, which is what
      // the producer_sync contract actually requires.
      auto [loads, stores] = fj_cpu->memory_aggregator.collect();
      double biased_work   = (loads + stores) * fj_cpu->work_unit_bias / 1e10;
      fj_cpu->work_units_elapsed += biased_work;

      if (fj_cpu->producer_sync != nullptr) { fj_cpu->producer_sync->notify_progress(); }
      if (fj_cpu->work_units_elapsed >= work_unit_limit) { break; }
    }

    cuopt_func_call(sanity_checks(*fj_cpu));
    fj_cpu->iterations++;
    fj_cpu->iterations_since_best++;
  }
  auto loop_end = std::chrono::high_resolution_clock::now();
  double total_time =
    std::chrono::duration_cast<std::chrono::duration<double>>(loop_end - loop_start).count();
  double avg_time_per_iter = fj_cpu->iterations > 0 ? total_time / fj_cpu->iterations : 0;
  CUOPT_LOG_TRACE("%sCPUFJ Average time per iteration: %.8fms",
                  fj_cpu->log_prefix.c_str(),
                  avg_time_per_iter * 1000.0);
  CUOPT_LOG_DEBUG("%sCPUFJ checkpoint: %lld restores, %lld snapshots, max streak %d",
                  fj_cpu->log_prefix.c_str(),
                  (long long)fj_cpu->n_checkpoint_restores,
                  (long long)fj_cpu->n_checkpoint_snapshots,
                  fj_cpu->max_restores_since_improvement);

#if CPUFJ_TIMING_TRACE
  // Print final timing statistics
  CUOPT_LOG_DEBUG("=== Final Timing Statistics ===");
  print_timing_stats(*fj_cpu);
#endif
}

template <typename T>
static std::vector<T> copy_to_host_async(const rmm::device_uvector<T>& input,
                                         rmm::cuda_stream_view stream)
{
  std::vector<T> output(input.size());
  raft::copy(output.data(), input.data(), input.size(), stream);
  return output;
}

template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> init_fj_cpu_from_optimization_problem(
  const optimization_problem_t<i_t, f_t>& problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings)
{
  using f_t2 = typename type_2<f_t>::type;

  raft::common::nvtx::range scope("init_fj_cpu_from_optimization_problem");

  const i_t n_variables   = problem.get_n_variables();
  const i_t n_constraints = problem.get_n_constraints();
  const i_t nnz           = problem.get_nnz();
  auto stream             = problem.get_handle_ptr()->get_stream();

  auto coefficients           = copy_to_host_async(problem.get_constraint_matrix_values(), stream);
  auto variables              = copy_to_host_async(problem.get_constraint_matrix_indices(), stream);
  auto offsets                = copy_to_host_async(problem.get_constraint_matrix_offsets(), stream);
  auto objective_coefficients = copy_to_host_async(problem.get_objective_coefficients(), stream);
  auto variable_lower_bounds  = copy_to_host_async(problem.get_variable_lower_bounds(), stream);
  auto variable_upper_bounds  = copy_to_host_async(problem.get_variable_upper_bounds(), stream);
  auto constraint_lower_bounds = copy_to_host_async(problem.get_constraint_lower_bounds(), stream);
  auto constraint_upper_bounds = copy_to_host_async(problem.get_constraint_upper_bounds(), stream);
  auto constraint_bounds       = copy_to_host_async(problem.get_constraint_bounds(), stream);
  auto row_types               = copy_to_host_async(problem.get_row_types(), stream);
  auto variable_types          = copy_to_host_async(problem.get_variable_types(), stream);
  problem.get_handle_ptr()->sync_stream();

  cuopt_assert(coefficients.size() == (size_t)nnz, "coefficient size mismatch");
  cuopt_assert(variables.size() == (size_t)nnz, "variable index size mismatch");
  cuopt_assert(offsets.size() == (size_t)(n_constraints + 1),
               "constraint offset size mismatch");
  cuopt_assert(!offsets.empty() && offsets.front() == 0, "invalid first constraint offset");
  cuopt_assert(offsets.back() == nnz, "invalid final constraint offset");
  cuopt_assert(std::is_sorted(offsets.begin(), offsets.end()), "unsorted constraint offsets");
  cuopt_assert(
    std::all_of(variables.begin(),
                variables.end(),
                [n_variables](i_t variable) { return variable >= 0 && variable < n_variables; }),
    "variable index out of range");
  cuopt_assert(objective_coefficients.size() == (size_t)n_variables,
               "objective size mismatch");
  cuopt_assert(variable_lower_bounds.empty() ||
                 variable_lower_bounds.size() == (size_t)n_variables,
               "variable lower bound size mismatch");
  cuopt_assert(variable_upper_bounds.empty() ||
                 variable_upper_bounds.size() == (size_t)n_variables,
               "variable upper bound size mismatch");

  if (constraint_lower_bounds.empty() && constraint_upper_bounds.empty()) {
    cuopt_assert(row_types.size() == (size_t)n_constraints, "row type size mismatch");
    cuopt_assert(constraint_bounds.size() == (size_t)n_constraints,
                 "constraint bound size mismatch");
    constraint_lower_bounds.resize(n_constraints);
    constraint_upper_bounds.resize(n_constraints);
    for (i_t row = 0; row < n_constraints; ++row) {
      const f_t bound = constraint_bounds[row];
      if (row_types[row] == 'E') {
        constraint_lower_bounds[row] = bound;
        constraint_upper_bounds[row] = bound;
      } else if (row_types[row] == 'G') {
        constraint_lower_bounds[row] = bound;
        constraint_upper_bounds[row] = std::numeric_limits<f_t>::infinity();
      } else {
        cuopt_assert(row_types[row] == 'L', "invalid row type");
        constraint_lower_bounds[row] = -std::numeric_limits<f_t>::infinity();
        constraint_upper_bounds[row] = bound;
      }
    }
  } else {
    cuopt_assert(constraint_lower_bounds.size() == (size_t)n_constraints,
                 "constraint lower bound size mismatch");
    cuopt_assert(constraint_upper_bounds.size() == (size_t)n_constraints,
                 "constraint upper bound size mismatch");
  }

  if (variable_lower_bounds.empty()) { variable_lower_bounds.assign(n_variables, f_t{0}); }
  if (variable_upper_bounds.empty()) {
    variable_upper_bounds.assign(n_variables, std::numeric_limits<f_t>::infinity());
  }
  if (variable_types.empty()) { variable_types.assign(n_variables, var_t::CONTINUOUS); }
  cuopt_assert(variable_types.size() == (size_t)n_variables,
               "variable type size mismatch");

  if (problem.get_sense()) {
    std::transform(objective_coefficients.begin(),
                   objective_coefficients.end(),
                   objective_coefficients.begin(),
                   std::negate<f_t>{});
  }

  std::vector<f_t2> variable_bounds(n_variables);
  std::vector<i_t> is_binary_variable(n_variables, 0);
  std::vector<i_t> binary_indices;
  binary_indices.reserve(n_variables);
  i_t n_integer_vars = 0;
  for (i_t variable = 0; variable < n_variables; ++variable) {
    f_t lower             = variable_lower_bounds[variable];
    f_t upper             = variable_upper_bounds[variable];
    const bool is_integer = variable_types[variable] == var_t::INTEGER;
    if (is_integer) {
      lower = std::ceil(lower);
      upper = std::floor(upper);
      ++n_integer_vars;
    }
    cuopt_assert(lower <= upper, "crossing variable bounds");
    variable_bounds[variable] = f_t2{lower, upper};
    if (is_integer && lower == f_t{0} && upper == f_t{1}) {
      is_binary_variable[variable] = 1;
      binary_indices.push_back(variable);
    }
  }

  csr_matrix_t<i_t, f_t> csr(n_constraints, n_variables, nnz);
  csr.x         = coefficients;
  csr.j         = variables;
  csr.row_start = offsets;
  csc_matrix_t<i_t, f_t> csc(n_constraints, n_variables, nnz);
  csr.to_compressed_col(csc);

  std::vector<f_t> assignment(n_variables, f_t{0});
  for (i_t variable = 0; variable < n_variables; ++variable) {
    f_t value = std::clamp(
      f_t{0}, get_lower(variable_bounds[variable]), get_upper(variable_bounds[variable]));
    if (variable_types[variable] == var_t::INTEGER) { value = std::round(value); }
    assignment[variable] = value;
  }

  auto fj_cpu      = std::make_unique<fj_cpu_climber_t<i_t, f_t>>(preemption_flag);
  fj_cpu->view     = typename fj_t<i_t, f_t>::climber_data_t::view_t{};
  fj_cpu->pb_ptr   = nullptr;
  fj_cpu->settings = settings;

  fj_cpu->h_reverse_coefficients = std::move(csc.x);
  fj_cpu->h_reverse_constraints  = std::move(csc.i);
  fj_cpu->h_reverse_offsets      = std::move(csc.col_start);
  fj_cpu->h_coefficients         = std::move(coefficients);
  fj_cpu->h_offsets              = std::move(offsets);
  fj_cpu->h_variables            = std::move(variables);
  fj_cpu->h_obj_coeffs           = std::move(objective_coefficients);
  fj_cpu->h_var_bounds           = std::move(variable_bounds);
  fj_cpu->h_cstr_lb              = std::move(constraint_lower_bounds);
  fj_cpu->h_cstr_ub              = std::move(constraint_upper_bounds);
  fj_cpu->h_var_types            = std::move(variable_types);
  fj_cpu->h_is_binary_variable   = std::move(is_binary_variable);
  fj_cpu->h_binary_indices       = std::move(binary_indices);
  fj_cpu->h_cstr_left_weights.resize(n_constraints, f_t{1});
  fj_cpu->h_cstr_right_weights.resize(n_constraints, f_t{1});
  fj_cpu->max_weight         = f_t{1};
  fj_cpu->h_objective_weight = f_t{0};
  fj_cpu->h_assignment       = assignment;
  fj_cpu->h_best_assignment  = std::move(assignment);
  fj_cpu->h_lhs.resize(n_constraints);
  fj_cpu->h_lhs_sumcomp.resize(n_constraints, f_t{0});
  fj_cpu->h_tabu_nodec_until.resize(n_variables, 0);
  fj_cpu->h_tabu_noinc_until.resize(n_variables, 0);
  fj_cpu->h_tabu_lastdec.resize(n_variables, 0);
  fj_cpu->h_tabu_lastinc.resize(n_variables, 0);
  fj_cpu->iterations    = 0;
  fj_cpu->settings.seed = cuopt::seed_generator::get_seed();

  finalize_fj_cpu_host_initialization(
    *fj_cpu, n_variables, n_constraints, n_integer_vars, nnz, tolerances);
  return fj_cpu;
}

template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> init_fj_cpu_standalone(
  problem_t<i_t, f_t>& problem,
  solution_t<i_t, f_t>& solution,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings)
{
  raft::common::nvtx::range scope("init_fj_cpu_standalone");

  auto fj_cpu = std::make_unique<fj_cpu_climber_t<i_t, f_t>>(preemption_flag);

  std::vector<f_t> default_weights(problem.n_constraints, 1.0);
  // Early CPUFJ runs while presolve is still probing, so there are no implications to hand it
  const probing_cache_t<i_t, f_t>* no_implications = nullptr;
  init_fj_cpu(*fj_cpu, solution, default_weights, default_weights, 0.0, no_implications);
  // settings.seed is caller-drawn: seed_generator steps a non-atomic global and this may run
  // concurrently across lanes.
  fj_cpu->settings = settings;

  return fj_cpu;
}

template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> init_fj_cpu_clone(
  const fj_cpu_climber_t<i_t, f_t>& tmpl,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings)
{
  raft::common::nvtx::range scope("init_fj_cpu_clone");

  auto fj_cpu = std::make_unique<fj_cpu_climber_t<i_t, f_t>>(preemption_flag);

  std::vector<f_t> default_weights(tmpl.view.pb.n_constraints, 1.0);
  init_fj_cpu_from_template(*fj_cpu, tmpl, default_weights, default_weights, f_t{0});
  // See init_fj_cpu_standalone: the seed is caller-drawn, not taken from the global generator.
  fj_cpu->settings = settings;

  return fj_cpu;
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::fj_cpu_deleter_t::operator()(fj_cpu_climber_t<i_t, f_t>* ptr) const
{
  delete ptr;
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::create_worker(
  const lp_problem_t<i_t, f_t>& problem,
  const std::vector<simplex::variable_type_t>& variable_types,
  const std::vector<f_t>& seed_assignment,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  std::string log_prefix,
  int64_t seed)
{
  auto new_climber = init_fj_cpu_from_host_lp(
    problem, variable_types, seed_assignment, settings, preemption_flag, seed);
  fj_cpu.reset(new_climber.release());
  fj_cpu->log_prefix           = std::move(log_prefix);
  fj_cpu->improvement_callback = improvement_callback;
  fj_cpu->halted               = false;
  preemption_flag              = false;
  is_initialized               = true;
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::run_async(f_t time_limit, double work_unit_limit)
{
  if (!is_initialized) return;

  auto& fj_ptr = fj_cpu;
#pragma omp task shared(fj_cpu, is_initialized, fj_ptr) firstprivate(time_limit, work_unit_limit) \
  priority(CUOPT_DEFAULT_TASK_PRIORITY) default(none) depend(out : fj_ptr)
  {
    if (is_initialized) { cpufj_solve(fj_cpu.get(), time_limit, work_unit_limit); }
  }
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::run_sync(f_t time_limit, double work_unit_limit)
{
  if (!is_initialized) return;
  cpufj_solve(fj_cpu.get(), time_limit, work_unit_limit);
  is_initialized = false;
  fj_cpu.reset();
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::stop()
{
  if (!is_initialized) return;

  preemption_flag = true;

  auto& fj_ptr = fj_cpu;
#pragma omp taskwait depend(in : fj_ptr)
  is_initialized = false;
  fj_cpu.reset();
}

template <typename i_t, typename f_t>
void fj_cpu_worker_t<i_t, f_t>::send_stop_signal()
{
  preemption_flag = true;
}

#if MIP_INSTANTIATE_FLOAT
template class fj_t<int, float>;
template struct fj_cpu_worker_t<int, float>;
template void cpufj_solve(fj_cpu_climber_t<int, float>* fj_cpu,
                          float in_time_limit,
                          double work_unit_limit);
template std::unique_ptr<fj_cpu_climber_t<int, float>> init_fj_cpu_standalone(
  problem_t<int, float>& problem,
  solution_t<int, float>& solution,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings);
template std::unique_ptr<fj_cpu_climber_t<int, float>> init_fj_cpu_clone(
  const fj_cpu_climber_t<int, float>& tmpl,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings);
template std::unique_ptr<fj_cpu_climber_t<int, float>> init_fj_cpu_from_optimization_problem(
  const optimization_problem_t<int, float>& problem,
  const typename mip_solver_settings_t<int, float>::tolerances_t& tolerances,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings);
template void finalize_fj_cpu_host_initialization(
  fj_cpu_climber_t<int, float>& fj_cpu,
  int n_variables,
  int n_constraints,
  int n_integer_vars,
  int nnz,
  const typename mip_solver_settings_t<int, float>::tolerances_t& tolerances);
#endif

#if MIP_INSTANTIATE_DOUBLE
template class fj_t<int, double>;
template struct fj_cpu_worker_t<int, double>;
template void cpufj_solve(fj_cpu_climber_t<int, double>* fj_cpu,
                          double in_time_limit,
                          double work_unit_limit);
template std::unique_ptr<fj_cpu_climber_t<int, double>> init_fj_cpu_standalone(
  problem_t<int, double>& problem,
  solution_t<int, double>& solution,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings);
template std::unique_ptr<fj_cpu_climber_t<int, double>> init_fj_cpu_clone(
  const fj_cpu_climber_t<int, double>& tmpl,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings);
template std::unique_ptr<fj_cpu_climber_t<int, double>> init_fj_cpu_from_optimization_problem(
  const optimization_problem_t<int, double>& problem,
  const typename mip_solver_settings_t<int, double>::tolerances_t& tolerances,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings);
template void finalize_fj_cpu_host_initialization(
  fj_cpu_climber_t<int, double>& fj_cpu,
  int n_variables,
  int n_constraints,
  int n_integer_vars,
  int nnz,
  const typename mip_solver_settings_t<int, double>::tolerances_t& tolerances);
#endif

// Above this the O(nnz) seed passes eat a meaningful slice of a short budget, so they are skipped.
constexpr int64_t fj_seed_nnz_limit = 8'000'000;

// Budget for the matching seed and the widest exact-one row it will take into the graph.
constexpr double fj_matching_budget_s        = 0.45;
constexpr int32_t fj_matching_max_row_width  = 20000;

// The aggressive corner pushes harder than the covering seed: more passes, a longer budget, a
// tighter clock, and it gives up as soon as a pass changes nothing.
constexpr int32_t fj_aggressive_passes  = 6;
constexpr double fj_aggressive_budget_s = 0.9;

// Cardinality-row detection: coefficient agreement tolerance and the widest row worth peeling.
constexpr double fj_exact_k_tol       = 1e-6;
constexpr int32_t fj_exact_k_max_width = 20000;
constexpr double fj_exact_k_budget_s   = 0.5;
// The anchor repair only runs when this fraction of the rows is violated, and gets this long.
constexpr int32_t fj_anchor_repair_violated_share = 5;
constexpr double fj_anchor_repair_budget_s        = 0.1;

// Jumps each two-sided variable to whichever bound has fewer rows locking it in that direction.
template <typename i_t, typename f_t>
static void apply_lock_weighted_seed(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  if (fj_cpu.view.pb.nnz > fj_seed_nnz_limit) return;

  const i_t n_variables = fj_cpu.view.pb.n_variables;
  for (i_t var_idx = 0; var_idx < n_variables; ++var_idx) {
    const f_t lb = get_lower(fj_cpu.h_var_bounds[var_idx].get());
    const f_t ub = get_upper(fj_cpu.h_var_bounds[var_idx].get());
    if (!isfinite(lb) || !isfinite(ub) || lb >= ub) continue;

    i_t lock_up       = 0;
    i_t lock_down     = 0;
    const auto range  = reverse_range_for_var<i_t, f_t>(fj_cpu, var_idx);
    for (i_t i = range.first; i < range.second; ++i) {
      const f_t coeff    = fj_cpu.h_reverse_coefficients[i];
      const i_t cstr_idx = fj_cpu.h_reverse_constraints[i];
      const bool has_lb  = isfinite((f_t)fj_cpu.h_cstr_lb[cstr_idx]);
      const bool has_ub  = isfinite((f_t)fj_cpu.h_cstr_ub[cstr_idx]);
      if (coeff > 0) {
        lock_up += has_ub;
        lock_down += has_lb;
      } else if (coeff < 0) {
        lock_up += has_lb;
        lock_down += has_ub;
      }
    }

    f_t new_val = lock_up <= lock_down ? ub : lb;
    if (is_integer_var<i_t, f_t>(fj_cpu, var_idx)) new_val = std::round(new_val);
    fj_cpu.h_assignment[var_idx] = new_val;
  }

  recompute_lhs(fj_cpu);
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
}

// Jumps each bounded objective variable to the bound that minimises its own objective term.
template <typename i_t, typename f_t>
static void apply_objective_corner_seed(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  if (fj_cpu.view.pb.nnz > fj_seed_nnz_limit) return;

  const i_t n_variables = fj_cpu.view.pb.n_variables;
  for (i_t var_idx = 0; var_idx < n_variables; ++var_idx) {
    const f_t coeff = fj_cpu.h_obj_coeffs[var_idx];
    if (coeff == 0) continue;

    const f_t lb = get_lower(fj_cpu.h_var_bounds[var_idx].get());
    const f_t ub = get_upper(fj_cpu.h_var_bounds[var_idx].get());
    if (!isfinite(lb) || !isfinite(ub) || lb >= ub) continue;

    f_t new_val = coeff > 0 ? lb : ub;
    if (is_integer_var<i_t, f_t>(fj_cpu, var_idx)) new_val = std::round(new_val);
    fj_cpu.h_assignment[var_idx] = new_val;
  }

  recompute_lhs(fj_cpu);
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
}

// A single-variable integer step on a row, with the magnitude of its effect on the row sum.
template <typename i_t, typename f_t>
struct row_repair_move_t {
  f_t effect;
  i_t var;
  f_t coeff;
  f_t new_val;
};

// Collects the unit integer steps that push this row's sum in `direction`, largest effect first.
template <typename i_t, typename f_t>
static void collect_row_repair_moves(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                                     i_t row_begin,
                                     i_t row_end,
                                     f_t direction,
                                     f_t tol,
                                     std::vector<row_repair_move_t<i_t, f_t>>& out)
{
  out.clear();
  for (i_t i = row_begin; i < row_end; ++i) {
    const i_t var = fj_cpu.h_variables[i];
    if (!is_integer_var<i_t, f_t>(fj_cpu, var)) continue;

    const f_t coeff   = fj_cpu.h_coefficients[i];
    const f_t val     = fj_cpu.h_assignment[var];
    const f_t lb      = get_lower(fj_cpu.h_var_bounds[var].get());
    const f_t ub      = get_upper(fj_cpu.h_var_bounds[var].get());
    const bool is_bin = fj_cpu.h_is_binary_variable[var] != 0;

    // Raising the variable shifts the sum by `direction * coeff`; lowering it by the negation.
    const f_t raise = direction * coeff;
    if (raise > 0 && val < ub - tol) {
      const f_t new_val = is_bin ? (f_t)1 : std::floor(val) + 1;
      if (new_val > val && new_val <= ub + tol) out.push_back({raise, var, coeff, new_val});
    } else if (raise < 0 && val > lb + tol) {
      const f_t new_val = is_bin ? (f_t)0 : std::ceil(val) - 1;
      if (new_val < val && new_val >= lb - tol) out.push_back({-raise, var, coeff, new_val});
    }
  }
  std::sort(out.begin(), out.end(), [](const row_repair_move_t<i_t, f_t>& a,
                                       const row_repair_move_t<i_t, f_t>& b) {
    return a.effect > b.effect;
  });
}

// Time-boxed greedy row repair. Deliberately myopic, so it reverts unless it strictly reduces the
// violated-row count against the incoming anchor.
template <typename i_t, typename f_t>
static void apply_greedy_covering_seed(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  if (fj_cpu.view.pb.nnz > fj_seed_nnz_limit) return;

  recompute_lhs(fj_cpu);
  const i_t baseline_violated  = fj_cpu.violated_constraints.size();
  const auto anchor_assignment = fj_cpu.h_assignment;

  const i_t n_constraints = fj_cpu.view.pb.n_constraints;
  std::vector<i_t> row_order(n_constraints);
  for (i_t i = 0; i < n_constraints; ++i)
    row_order[i] = i;
  std::sort(row_order.begin(), row_order.end(), [&](i_t a, i_t b) {
    return (fj_cpu.h_offsets[a + 1] - fj_cpu.h_offsets[a]) <
           (fj_cpu.h_offsets[b + 1] - fj_cpu.h_offsets[b]);
  });

  const auto started         = std::chrono::steady_clock::now();
  const double time_budget_s = 0.4;
  const f_t tol              = 1e-6;
  const i_t max_passes       = 2;
  std::vector<row_repair_move_t<i_t, f_t>> candidates;
  bool out_of_time = false;

  for (i_t pass = 0; pass < max_passes && !out_of_time; ++pass) {
    for (i_t k = 0; k < n_constraints; ++k) {
      if ((k & 0xFFF) == 0 &&
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >
            time_budget_s) {
        out_of_time = true;
        break;
      }
      const i_t cstr_idx  = row_order[k];
      const i_t row_begin = fj_cpu.h_offsets[cstr_idx];
      const i_t row_end   = fj_cpu.h_offsets[cstr_idx + 1];
      if (row_begin == row_end) continue;

      const f_t lb      = fj_cpu.h_cstr_lb[cstr_idx];
      const f_t ub      = fj_cpu.h_cstr_ub[cstr_idx];
      const bool has_lb = isfinite(lb);
      const bool has_ub = isfinite(ub);
      if (!has_lb && !has_ub) continue;

      f_t sum = 0;
      for (i_t i = row_begin; i < row_end; ++i)
        sum += (f_t)fj_cpu.h_coefficients[i] * (f_t)fj_cpu.h_assignment[fj_cpu.h_variables[i]];

      // Equality rows are driven to their bound; one-sided rows only to the side they violate.
      const bool is_equality = has_lb && has_ub && std::abs(lb - ub) < tol;
      f_t direction          = 0;
      f_t target             = 0;
      if (is_equality && std::abs(sum - lb) > tol) {
        direction = sum < lb ? (f_t)1 : (f_t)-1;
        target    = lb;
      } else if (has_lb && sum < lb - tol) {
        direction = 1;
        target    = lb;
      } else if (has_ub && sum > ub + tol) {
        direction = -1;
        target    = ub;
      } else {
        continue;
      }

      collect_row_repair_moves<i_t, f_t>(fj_cpu, row_begin, row_end, direction, tol, candidates);
      for (const auto& m : candidates) {
        if (direction > 0 ? sum >= target - tol : sum <= target + tol) break;
        const f_t delta = m.new_val - (f_t)fj_cpu.h_assignment[m.var];
        sum += m.coeff * delta;
        fj_cpu.h_assignment[m.var] = m.new_val;
      }
    }
  }

  recompute_lhs(fj_cpu);
  if ((i_t)fj_cpu.violated_constraints.size() >= baseline_violated) {
    fj_cpu.h_assignment = anchor_assignment;
    recompute_lhs(fj_cpu);
  }
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
}

// Repeated one-sided row repair in CSR order. Unlike the covering seed it revisits rows until a
// pass changes nothing, so a repair that breaks a row already visited gets another chance.
template <typename i_t, typename f_t>
static void apply_aggressive_constraint_seed(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  if (fj_cpu.view.pb.nnz > fj_seed_nnz_limit) return;

  const auto started = std::chrono::steady_clock::now();
  auto timed_out     = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >
           fj_aggressive_budget_s;
  };

  recompute_lhs(fj_cpu);
  const i_t baseline = fj_cpu.violated_constraints.size();
  const auto anchor  = fj_cpu.h_assignment;
  const f_t tol      = 1e-6;
  std::vector<row_repair_move_t<i_t, f_t>> candidates;

  for (i_t pass = 0; pass < fj_aggressive_passes && !timed_out(); ++pass) {
    i_t moves = 0;
    for (i_t row = 0; row < fj_cpu.view.pb.n_constraints; ++row) {
      if ((row & 0xFF) == 0 && timed_out()) break;

      const i_t begin = fj_cpu.h_offsets[row];
      const i_t end   = fj_cpu.h_offsets[row + 1];
      if (begin == end) continue;

      const f_t lb      = fj_cpu.h_cstr_lb[row];
      const f_t ub      = fj_cpu.h_cstr_ub[row];
      const bool has_lb = isfinite(lb);
      const bool has_ub = isfinite(ub);
      if (!has_lb && !has_ub) continue;

      f_t sum = 0;
      for (i_t p = begin; p < end; ++p)
        sum += (f_t)fj_cpu.h_coefficients[p] * (f_t)fj_cpu.h_assignment[fj_cpu.h_variables[p]];

      f_t direction = 0;
      f_t target    = 0;
      if (has_lb && sum < lb - tol) {
        direction = 1;
        target    = lb;
      } else if (has_ub && sum > ub + tol) {
        direction = -1;
        target    = ub;
      } else {
        continue;
      }

      collect_row_repair_moves<i_t, f_t>(fj_cpu, begin, end, direction, tol, candidates);
      for (const auto& move : candidates) {
        if (direction > 0 ? sum >= target - tol : sum <= target + tol) break;
        const f_t delta = move.new_val - (f_t)fj_cpu.h_assignment[move.var];
        sum += move.coeff * delta;
        fj_cpu.h_assignment[move.var] = move.new_val;
        ++moves;
      }
    }
    if (moves == 0) break;
  }

  recompute_lhs(fj_cpu);
  if ((i_t)fj_cpu.violated_constraints.size() >= baseline) {
    fj_cpu.h_assignment = anchor;
    recompute_lhs(fj_cpu);
  }
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
}

// Treats the exact-one rows as a graph in which each variable is an edge between the two rows it
// appears in. A component that is bipartite and has equally many rows on each side admits a perfect
// matching, and the cheapest one is the assignment satisfying every row in the component at least
// cost. Solved per component as min-cost flow by successive shortest paths, which needs no
// potentials here because augmenting along shortest paths keeps the residual free of negative
// cycles. Components that are not of that shape are left to the search.
template <typename i_t, typename f_t>
static void apply_bipartite_matching_seed(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  if (fj_cpu.view.pb.nnz > fj_seed_nnz_limit) return;

  const auto started = std::chrono::steady_clock::now();
  auto timed_out     = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >
           fj_matching_budget_s;
  };
  const f_t tol = 1e-6;

  struct exact_one_row_t {
    i_t begin, end;
  };
  std::vector<exact_one_row_t> rows;
  for (i_t row = 0; row < fj_cpu.view.pb.n_constraints; ++row) {
    if ((row & 0xFFF) == 0 && timed_out()) return;

    const f_t lb = fj_cpu.h_cstr_lb[row];
    const f_t ub = fj_cpu.h_cstr_ub[row];
    if (!isfinite(lb) || !isfinite(ub) || std::abs(lb - ub) > tol) continue;

    const i_t begin = fj_cpu.h_offsets[row];
    const i_t end   = fj_cpu.h_offsets[row + 1];
    if (begin == end || end - begin > fj_matching_max_row_width) continue;

    const f_t scale = fj_cpu.h_coefficients[begin];
    if (!isfinite(scale) || std::abs(scale) <= tol || std::abs(lb / scale - 1) > 1e-5) continue;

    bool uniform_binary = true;
    for (i_t p = begin; p < end && uniform_binary; ++p) {
      const f_t coeff     = fj_cpu.h_coefficients[p];
      const f_t agreement = tol * std::max((f_t)1, std::abs(scale));
      uniform_binary = fj_cpu.h_is_binary_variable[fj_cpu.h_variables[p]] &&
                       std::abs(coeff - scale) <= agreement;
    }
    if (uniform_binary) rows.push_back({begin, end});
  }
  if (rows.size() < 2 || timed_out()) return;

  const i_t n_rows      = (i_t)rows.size();
  const i_t n_variables = fj_cpu.view.pb.n_variables;
  std::vector<i_t> degree(n_variables, 0);
  std::vector<i_t> endpoint_a(n_variables, -1);
  std::vector<i_t> endpoint_b(n_variables, -1);
  for (i_t row = 0; row < n_rows; ++row) {
    for (i_t p = rows[row].begin; p < rows[row].end; ++p) {
      const i_t var = fj_cpu.h_variables[p];
      if (degree[var] == 0) endpoint_a[var] = row;
      else if (degree[var] == 1) endpoint_b[var] = row;
      ++degree[var];
    }
  }

  struct edge_t {
    int to, reverse, capacity;
    f_t cost;
    i_t var;
  };
  auto add_edge = [](std::vector<std::vector<edge_t>>& graph, int from, int to, f_t cost, i_t var) {
    const int back = (int)graph[to].size();
    graph[from].push_back({to, back, 1, cost, var});
    graph[to].push_back({from, (int)graph[from].size() - 1, 0, -cost, -1});
  };

  std::vector<int8_t> color(n_rows, -1);
  std::vector<int8_t> state(n_variables, -1);
  std::vector<i_t> side_index(n_rows, -1);
  std::vector<i_t> component_rows, component_vars, left, right;
  std::queue<i_t> pending;
  bool installed = false;

  for (i_t root = 0; root < n_rows && !timed_out(); ++root) {
    if (color[root] >= 0) continue;

    component_rows.clear();
    component_vars.clear();
    color[root] = 0;
    pending.push(root);
    bool valid = true;
    while (!pending.empty()) {
      const i_t row = pending.front();
      pending.pop();
      component_rows.push_back(row);
      for (i_t p = rows[row].begin; p < rows[row].end; ++p) {
        const i_t var = fj_cpu.h_variables[p];
        // A variable outside exactly two rows is not an edge, and a self-loop cannot be 2-coloured.
        if (degree[var] != 2 || endpoint_a[var] == endpoint_b[var]) {
          valid = false;
          continue;
        }
        if (endpoint_a[var] == row) component_vars.push_back(var);
        const i_t other = endpoint_a[var] == row ? endpoint_b[var] : endpoint_a[var];
        if (color[other] < 0) {
          color[other] = 1 - color[row];
          pending.push(other);
        } else if (color[other] == color[row]) {
          valid = false;
        }
      }
      if ((component_rows.size() & 0x3FF) == 0 && timed_out()) return;
    }
    if (!valid || component_vars.empty()) continue;

    left.clear();
    right.clear();
    for (i_t row : component_rows)
      (color[row] == 0 ? left : right).push_back(row);
    if (left.size() != right.size()) continue;
    for (i_t k = 0; k < (i_t)left.size(); ++k)
      side_index[left[k]] = k;
    for (i_t k = 0; k < (i_t)right.size(); ++k)
      side_index[right[k]] = k;

    const int side   = (int)left.size();
    const int source = 2 * side;
    const int sink   = source + 1;
    std::vector<std::vector<edge_t>> graph(sink + 1);
    for (int k = 0; k < side; ++k) {
      add_edge(graph, source, k, 0, -1);
      add_edge(graph, side + k, sink, 0, -1);
    }
    // Every perfect matching uses exactly one variable edge per row, so shifting all of them by a
    // constant moves every matching's cost equally and leaves the cheapest one unchanged. Shifting
    // the negatives away is what lets the potentials below start at zero.
    f_t cheapest = 0;
    for (i_t var : component_vars) {
      const f_t cost = fj_cpu.h_obj_coeffs[var];
      if (!isfinite(cost)) {
        valid = false;
        break;
      }
      cheapest = std::min(cheapest, cost);
    }
    if (!valid) continue;
    const f_t shift = -cheapest;

    for (i_t var : component_vars) {
      i_t a = endpoint_a[var];
      i_t b = endpoint_b[var];
      if (color[a] == 1) std::swap(a, b);
      add_edge(graph, side_index[a], side + side_index[b], fj_cpu.h_obj_coeffs[var] + shift, var);
    }

    // Node potentials hold every reduced cost at or above zero, which is what makes Dijkstra
    // applicable. All shifted costs start non-negative, so the potentials start at zero. Rounding
    // can still leave a tree edge fractionally negative once the potentials move, so relaxation
    // below skips settled nodes: that keeps every predecessor older than its successor in
    // settlement order, which is what makes the retrace terminate.
    int flow = 0;
    std::vector<f_t> potential(graph.size(), 0);
    std::vector<f_t> distance(graph.size());
    std::vector<int> previous_node(graph.size());
    std::vector<int> previous_edge(graph.size());
    std::vector<uint8_t> settled(graph.size());
    using heap_entry_t = std::pair<f_t, int>;

    while (flow < side && !timed_out()) {
      std::fill(distance.begin(), distance.end(), std::numeric_limits<f_t>::infinity());
      std::fill(previous_node.begin(), previous_node.end(), -1);
      std::fill(settled.begin(), settled.end(), 0);
      distance[source] = 0;
      std::priority_queue<heap_entry_t, std::vector<heap_entry_t>, std::greater<heap_entry_t>> heap;
      heap.push({0, source});

      while (!heap.empty()) {
        const auto [reached_at, from] = heap.top();
        heap.pop();
        if (settled[from]) continue;
        settled[from] = 1;
        for (int e = 0; e < (int)graph[from].size(); ++e) {
          const auto& edge = graph[from][e];
          if (!edge.capacity || settled[edge.to]) continue;
          const f_t reduced = edge.cost + potential[from] - potential[edge.to];
          cuopt_assert(reduced >= -1e-9 * std::max((f_t)1, std::abs(edge.cost)),
                       "potentials failed to keep the reduced cost non-negative");
          if (reached_at + reduced >= distance[edge.to]) continue;
          distance[edge.to]      = reached_at + reduced;
          previous_node[edge.to] = from;
          previous_edge[edge.to] = e;
          heap.push({distance[edge.to], edge.to});
        }
      }
      if (previous_node[sink] < 0) break;

      for (int node = 0; node < (int)graph.size(); ++node)
        if (isfinite(distance[node])) potential[node] += distance[node];

      for (int node = sink; node != source; node = previous_node[node]) {
        cuopt_assert(previous_node[node] >= 0, "augmenting path is broken");
        auto& edge = graph[previous_node[node]][previous_edge[node]];
        --edge.capacity;
        ++graph[node][edge.reverse].capacity;
      }
      ++flow;
    }
    if (flow != side) continue;

    for (i_t var : component_vars)
      state[var] = 0;
    for (int node = 0; node < side; ++node)
      for (const auto& edge : graph[node])
        if (edge.var >= 0 && edge.capacity == 0) state[edge.var] = 1;
    installed = true;
  }
  if (!installed) return;

  recompute_lhs(fj_cpu);
  const i_t baseline = fj_cpu.violated_constraints.size();
  const auto anchor  = fj_cpu.h_assignment;
  for (i_t var = 0; var < n_variables; ++var)
    if (state[var] >= 0) fj_cpu.h_assignment[var] = state[var];

  recompute_lhs(fj_cpu);
  const i_t candidate = fj_cpu.violated_constraints.size();
  // Kept when it reaches feasibility outright, otherwise only on a strict gain.
  if (candidate != 0 && candidate >= baseline) {
    fj_cpu.h_assignment = anchor;
    recompute_lhs(fj_cpu);
  }
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
}

// Every variable to its lower bound, or its upper where the lower is infinite.
template <typename i_t, typename f_t>
static void apply_lower_bound_seed(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  if (fj_cpu.view.pb.nnz > fj_seed_nnz_limit) return;

  for (i_t var_idx = 0; var_idx < fj_cpu.view.pb.n_variables; ++var_idx) {
    auto bounds     = fj_cpu.h_var_bounds[var_idx].get();
    const f_t lower = get_lower(bounds);
    const f_t upper = get_upper(bounds);
    if (!isfinite(lower) && !isfinite(upper)) continue;

    f_t new_val = isfinite(lower) ? lower : upper;
    if (is_integer_var<i_t, f_t>(fj_cpu, var_idx)) new_val = std::round(new_val);
    fj_cpu.h_assignment[var_idx] = new_val;
  }

  recompute_lhs(fj_cpu);
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
}

// Constructively satisfies the equality rows that read as sum(x) = k over binaries sharing one
// coefficient: pick k members of each, narrowest rows first so the wide ones inherit the choices,
// and within a row the variables appearing in fewest other such rows.
template <typename i_t, typename f_t>
static void apply_exact_k_seed(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  if (fj_cpu.view.pb.nnz > fj_seed_nnz_limit) return;

  const auto started = std::chrono::steady_clock::now();
  auto timed_out     = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >
           fj_exact_k_budget_s;
  };

  struct exact_k_row_t {
    i_t k, begin, end;
  };
  std::vector<exact_k_row_t> rows;
  for (i_t row = 0; row < fj_cpu.view.pb.n_constraints; ++row) {
    if ((row & 0xFFF) == 0 && timed_out()) return;

    const f_t lb = fj_cpu.h_cstr_lb[row];
    const f_t ub = fj_cpu.h_cstr_ub[row];
    if (!isfinite(lb) || !isfinite(ub) || std::abs(lb - ub) > fj_exact_k_tol) continue;

    const i_t begin = fj_cpu.h_offsets[row];
    const i_t end   = fj_cpu.h_offsets[row + 1];
    if (end - begin < 2 || end - begin > fj_exact_k_max_width) continue;

    const f_t scale = fj_cpu.h_coefficients[begin];
    if (scale <= 0) continue;
    bool uniform_binary = true;
    for (i_t p = begin; p < end && uniform_binary; ++p) {
      const i_t var       = fj_cpu.h_variables[p];
      const f_t coeff     = fj_cpu.h_coefficients[p];
      const f_t agreement = fj_exact_k_tol * std::max((f_t)1, std::abs(scale));
      uniform_binary =
        fj_cpu.h_is_binary_variable[var] && coeff > 0 && std::abs(coeff - scale) <= agreement;
    }
    if (!uniform_binary) continue;

    const double cardinality = (double)lb / scale;
    const i_t k              = (i_t)std::lround(cardinality);
    if (std::abs(cardinality - k) <= 1e-4 && k >= 0 && k <= end - begin)
      rows.push_back({k, begin, end});
  }
  if (rows.empty()) return;

  std::sort(rows.begin(), rows.end(), [](const exact_k_row_t& a, const exact_k_row_t& b) {
    return a.end - a.begin < b.end - b.begin;
  });

  const i_t n_variables = fj_cpu.view.pb.n_variables;
  std::vector<i_t> degree(n_variables, 0);
  for (const auto& row : rows)
    for (i_t p = row.begin; p < row.end; ++p)
      ++degree[fj_cpu.h_variables[p]];

  std::vector<int8_t> state(n_variables, -1);
  std::vector<i_t> free_vars;
  for (size_t index = 0; index < rows.size(); ++index) {
    if ((index & 0xFFF) == 0 && timed_out()) break;
    const auto& row = rows[index];

    i_t selected = 0;
    free_vars.clear();
    for (i_t p = row.begin; p < row.end; ++p) {
      const i_t var = fj_cpu.h_variables[p];
      selected += state[var] == 1;
      if (state[var] < 0) free_vars.push_back(var);
    }
    const i_t needed = row.k - selected;
    if (needed < 0 || (i_t)free_vars.size() < needed) continue;

    std::sort(free_vars.begin(), free_vars.end(), [&degree](i_t a, i_t b) {
      return degree[a] < degree[b];
    });
    for (i_t p = 0; p < (i_t)free_vars.size(); ++p)
      state[free_vars[p]] = (int8_t)(p < needed);
  }

  recompute_lhs(fj_cpu);
  const i_t baseline = fj_cpu.violated_constraints.size();
  const auto anchor  = fj_cpu.h_assignment;
  for (i_t var = 0; var < n_variables; ++var)
    if (state[var] >= 0) fj_cpu.h_assignment[var] = state[var];

  recompute_lhs(fj_cpu);
  if ((i_t)fj_cpu.violated_constraints.size() >= baseline) {
    fj_cpu.h_assignment = anchor;
    recompute_lhs(fj_cpu);
  }
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
}

// One repair pass over the violated rows of a start that is mostly violated. Row sums are read from
// the lhs computed on entry, so a row does not see the repairs made for earlier rows; the revert
// below is what keeps that myopia from costing anything.
template <typename i_t, typename f_t>
static void repair_difficult_anchor(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  recompute_lhs(fj_cpu);
  const i_t baseline = fj_cpu.violated_constraints.size();
  if (baseline == 0 || baseline <= fj_cpu.view.pb.n_constraints / fj_anchor_repair_violated_share)
    return;

  const auto started = std::chrono::steady_clock::now();
  const auto anchor  = fj_cpu.h_assignment;
  const std::vector<i_t> violated(fj_cpu.violated_constraints.begin(),
                                  fj_cpu.violated_constraints.end());
  std::vector<row_repair_move_t<i_t, f_t>> candidates;

  for (i_t row : violated) {
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >
        fj_anchor_repair_budget_s)
      break;

    const f_t lb = fj_cpu.h_cstr_lb[row];
    const f_t ub = fj_cpu.h_cstr_ub[row];
    f_t sum      = fj_cpu.h_lhs[row];
    f_t target   = 0;
    f_t direction = 0;
    if (sum < lb) {
      direction = 1;
      target    = lb;
    } else if (sum > ub) {
      direction = -1;
      target    = ub;
    } else {
      continue;
    }

    collect_row_repair_moves<i_t, f_t>(fj_cpu,
                                       fj_cpu.h_offsets[row],
                                       fj_cpu.h_offsets[row + 1],
                                       direction,
                                       fj_exact_k_tol,
                                       candidates);
    for (const auto& move : candidates) {
      if (direction > 0 ? sum >= target : sum <= target) break;
      const f_t delta = move.new_val - (f_t)fj_cpu.h_assignment[move.var];
      sum += move.coeff * delta;
      fj_cpu.h_assignment[move.var] = move.new_val;
    }
  }

  recompute_lhs(fj_cpu);
  if ((i_t)fj_cpu.violated_constraints.size() >= baseline) {
    fj_cpu.h_assignment = anchor;
    recompute_lhs(fj_cpu);
  }
  fj_cpu.h_best_assignment = fj_cpu.h_assignment;
}

// What makes one lane of a CPUFJ portfolio behave differently from another: which corner it starts
// from, how it samples, and how hard it pulls on the objective. Lane 0 keeps the anchor assignment
// so it is the lane every clone is built from.
template <typename i_t, typename f_t>
void apply_lane_diversification(fj_cpu_climber_t<i_t, f_t>& climber, int lane, int64_t base_seed)
{
  // Objective pressure across the portfolio, indexed by lane. Lanes 0 and 3 stay pure feasibility
  // seekers until they cross, since the objective term only enters the score once the weight is
  // positive; their nonzero floor then keeps a pull on the objective afterwards rather than letting
  // smooth_weights decay it back to nothing.
  const f_t obj_weight_ladder[4] = {0, 4, 32, 0};
  const f_t obj_weight_floor[4]  = {1, 4, 32, 1};

  // One structural start per lane; lanes 0 and 4 keep the shared anchor. Lane 7 also keeps it here,
  // because its replacement is an LP solved inside that lane's own task rather than in setup.
  climber.use_lp_seed = lane % 8 == 7;
  switch (lane % 8) {
    case 1: apply_lock_weighted_seed<i_t, f_t>(climber); break;
    case 2: apply_aggressive_constraint_seed<i_t, f_t>(climber); break;
    case 3: apply_greedy_covering_seed<i_t, f_t>(climber); break;
    case 5: apply_bipartite_matching_seed<i_t, f_t>(climber); break;
    case 6: apply_objective_corner_seed<i_t, f_t>(climber); break;
    default: break;
  }

  // Default: every climber identical apart from its seed and a random draw of the
  // four sampling parameters. Diversification, decorrelated from the value RNG.
  std::mt19937 rng(base_seed + 7919u * lane);
  climber.mtm_viol_samples = std::uniform_int_distribution<i_t>(15, 50)(rng);
  climber.mtm_sat_samples  = std::uniform_int_distribution<i_t>(10, 30)(rng);
  climber.nnz_samples      = std::uniform_int_distribution<i_t>(2000, 15000)(rng);
  climber.perturb_interval = std::uniform_int_distribution<i_t>(50, 500)(rng);

  climber.enable_infeasible_repair = (lane % 8 == 1) || (lane % 8 == 5);

  climber.h_objective_weight    = obj_weight_ladder[lane % 4];
  //climber.seed_objective_weight = obj_weight_floor[lane % 4];
}

// Portfolio construction for the standalone benchmark. Host logic, but it lives
// in a .cu because fj_cpu.cuh pulls in raft/util/cuda_dev_essentials.cuh through
// solution.cuh, which does not compile under the host compiler. Kept out of the
// header regardless: editing this file rebuilds one translation unit rather than
// the fifteen that including headers pull in.
template <typename i_t, typename f_t>
void build_climber_portfolio(problem_t<i_t, f_t>& problem,
                             solution_t<i_t, f_t>& solution,
                             std::vector<std::atomic<bool>>& preemption_flags,
                             std::vector<std::unique_ptr<fj_cpu_climber_t<i_t, f_t>>>& climbers,
                             int64_t base_seed)
{
  const int n_climbers = static_cast<int>(climbers.size());

  for (int k = 0; k < n_climbers; ++k)
    preemption_flags[k].store(false);

  // cuopt::seed_generator::get_seed() steps a non-atomic global, so every lane's seed is drawn here
  // in lane order before any concurrent construction below.
  std::vector<int64_t> lane_seed(n_climbers);
  for (int k = 0; k < n_climbers; ++k)
    lane_seed[k] = cuopt::seed_generator::get_seed();

  // Lane 0 is a genuine dependency: it host-copies the problem and every other lane clones it.
  {
    fj_settings_t settings;
    settings.seed = (int)lane_seed[0];
    climbers[0]   = init_fj_cpu_standalone(problem, solution, preemption_flags[0], settings);
    // Runs before the clones are taken, so every lane starts from the repaired anchor.
    apply_exact_k_seed<i_t, f_t>(*climbers[0]);
    repair_difficult_anchor<i_t, f_t>(*climbers[0]);
    apply_lane_diversification<i_t, f_t>(*climbers[0], 0, base_seed);
  }

  // The remaining lanes depend only on lane 0's finished, read-only template, and the O(nnz) clone
  // and seed passes are otherwise paid serially on one thread while the other pinned CPUs idle.
#ifdef _OPENMP
#pragma omp parallel for num_threads(std::max(1, n_climbers - 1)) schedule(static)
#endif
  for (int k = 1; k < n_climbers; ++k) {
    fj_settings_t settings;
    settings.seed = (int)lane_seed[k];
    climbers[k]   = init_fj_cpu_clone(*climbers[0], preemption_flags[k], settings);
    apply_lane_diversification<i_t, f_t>(*climbers[k], k, base_seed);
  }

  auto shared = std::make_shared<fj_cpu_shared_incumbent_t<i_t, f_t>>();
  for (int k = 0; k < n_climbers; ++k)
    climbers[k]->shared_incumbent = shared;
}

#if MIP_INSTANTIATE_FLOAT
template void apply_lane_diversification<int, float>(fj_cpu_climber_t<int, float>&, int, int64_t);
template void build_climber_portfolio<int, float>(
  problem_t<int, float>&, solution_t<int, float>&, std::vector<std::atomic<bool>>&,
  std::vector<std::unique_ptr<fj_cpu_climber_t<int, float>>>&, int64_t);
#endif

#if MIP_INSTANTIATE_DOUBLE
template void apply_lane_diversification<int, double>(fj_cpu_climber_t<int, double>&, int, int64_t);
template void build_climber_portfolio<int, double>(
  problem_t<int, double>&, solution_t<int, double>&, std::vector<std::atomic<bool>>&,
  std::vector<std::unique_ptr<fj_cpu_climber_t<int, double>>>&, int64_t);
#endif

}  // namespace cuopt::mathematical_optimization::mip
