/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/mip_constants.hpp>

#include "feasibility_jump.cuh"
#include "feasibility_jump_impl_common.cuh"
#include "fj_cpu.cuh"

#include <utilities/seed_generator.cuh>

#include <raft/common/nvtx.hpp>

#include <chrono>
#include <iomanip>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
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

namespace cuopt::linear_programming::detail {

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
                                          i_t var_idx,
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
    f_t new_lhs     = (current_lhs + cstr_coeff * delta) * sign;
    f_t old_slack   = rhs - old_lhs;
    f_t new_slack   = rhs - new_lhs;

    cuopt_assert(isfinite(cstr_weight), "invalid weight");
    cuopt_assert(cstr_weight >= 0, "invalid weight");
    cuopt_assert(isfinite(old_lhs), "");
    cuopt_assert(isfinite(new_lhs), "");
    cuopt_assert(isfinite(old_slack) && isfinite(new_slack), "");

    f_t cstr_tolerance = fj.get_corrected_tolerance(cstr_idx, c_lb, c_ub);

    bool old_viol = fj.excess_score(cstr_idx, current_lhs, c_lb, c_ub) < -cstr_tolerance;
    bool new_viol =
      fj.excess_score(cstr_idx, current_lhs + cstr_coeff * delta, c_lb, c_ub) < -cstr_tolerance;

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
    else if (!old_sat && !new_sat && old_lhs <= new_lhs) {
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
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration =
      std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time_);
    times_vec_.push_back(duration.count());
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
  CUOPT_LOG_TRACE("=== Timing Statistics (Iteration %d) ===", fj_cpu.iterations);
  CUOPT_LOG_TRACE("find_lift_move:      avg=%.6f ms, total=%.6f ms, calls=%zu",
                  lift_avg * 1000.0,
                  lift_total * 1000.0,
                  fj_cpu.find_lift_move_times.size());
  CUOPT_LOG_TRACE("find_mtm_move_viol:  avg=%.6f ms, total=%.6f ms, calls=%zu",
                  viol_avg * 1000.0,
                  viol_total * 1000.0,
                  fj_cpu.find_mtm_move_viol_times.size());
  CUOPT_LOG_TRACE("find_mtm_move_sat:   avg=%.6f ms, total=%.6f ms, calls=%zu",
                  sat_avg * 1000.0,
                  sat_total * 1000.0,
                  fj_cpu.find_mtm_move_sat_times.size());
  CUOPT_LOG_TRACE("apply_move:          avg=%.6f ms, total=%.6f ms, calls=%zu",
                  apply_avg * 1000.0,
                  apply_total * 1000.0,
                  fj_cpu.apply_move_times.size());
  CUOPT_LOG_TRACE("update_weights:      avg=%.6f ms, total=%.6f ms, calls=%zu",
                  weights_avg * 1000.0,
                  weights_total * 1000.0,
                  fj_cpu.update_weights_times.size());
  CUOPT_LOG_TRACE("compute_score:       avg=%.6f ms, total=%.6f ms, calls=%zu",
                  compute_score_avg * 1000.0,
                  compute_score_total * 1000.0,
                  fj_cpu.compute_score_times.size());
  CUOPT_LOG_TRACE("cache hit percentage: %.2f%%",
                  (double)fj_cpu.hit_count / (fj_cpu.hit_count + fj_cpu.miss_count) * 100.0);
  CUOPT_LOG_TRACE("bin  candidate move hit percentage: %.2f%%",
                  (double)fj_cpu.candidate_move_hits[0] /
                    (fj_cpu.candidate_move_hits[0] + fj_cpu.candidate_move_misses[0]) * 100.0);
  CUOPT_LOG_TRACE("int  candidate move hit percentage: %.2f%%",
                  (double)fj_cpu.candidate_move_hits[1] /
                    (fj_cpu.candidate_move_hits[1] + fj_cpu.candidate_move_misses[1]) * 100.0);
  CUOPT_LOG_TRACE("cont candidate move hit percentage: %.2f%%",
                  (double)fj_cpu.candidate_move_hits[2] /
                    (fj_cpu.candidate_move_hits[2] + fj_cpu.candidate_move_misses[2]) * 100.0);
  CUOPT_LOG_TRACE("========================================");
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
  double nnz_per_move = 0.0;
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

  for (i_t i = offset_begin; i < offset_end; i++) {
    auto cstr_idx = fj_cpu.h_reverse_constraints[i];
    fj_cpu.unique_cstrs_accessed_window.insert(cstr_idx);
    auto cstr_coeff   = fj_cpu.h_reverse_coefficients[i];
    auto [c_lb, c_ub] = fj_cpu.cached_cstr_bounds[i].get();

    cuopt_assert(c_lb <= c_ub, "invalid bounds");

    auto [cstr_base_feas, cstr_bonus_robust] =
      feas_score_constraint<i_t, f_t>(fj_cpu.view,
                                      var_idx,
                                      delta,
                                      cstr_idx,
                                      cstr_coeff,
                                      c_lb,
                                      c_ub,
                                      fj_cpu.h_lhs[cstr_idx],
                                      fj_cpu.h_cstr_left_weights[cstr_idx],
                                      fj_cpu.h_cstr_right_weights[cstr_idx]);

    base_feas_sum += cstr_base_feas;
    bonus_robust_sum += cstr_bonus_robust;
  }

  f_t base_obj = 0;
  if (obj_diff < 0)  // improving move wrt objective
    base_obj = fj_cpu.h_objective_weight;
  else if (obj_diff > 0)
    base_obj = -fj_cpu.h_objective_weight;

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

template <typename i_t, typename f_t>
static void smooth_weights(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  CPUFJ_NVTX_RANGE("CPUFJ::smooth_weights");
  for (i_t cstr_idx = 0; cstr_idx < fj_cpu.view.pb.n_constraints; cstr_idx++) {
    // consider only satisfied constraints
    if (fj_cpu.violated_constraints.count(cstr_idx)) continue;

    f_t weight_l = max((f_t)0, fj_cpu.h_cstr_left_weights[cstr_idx] - 1);
    f_t weight_r = max((f_t)0, fj_cpu.h_cstr_right_weights[cstr_idx] - 1);

    fj_cpu.h_cstr_left_weights[cstr_idx]  = weight_l;
    fj_cpu.h_cstr_right_weights[cstr_idx] = weight_r;
  }

  if (fj_cpu.h_objective_weight > 0 && fj_cpu.h_incumbent_objective >= fj_cpu.h_best_objective) {
    fj_cpu.h_objective_weight = max((f_t)0, fj_cpu.h_objective_weight - 1);
  }
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

    i_t int_delta = 1.0;
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
    auto [relvar_offset_begin, relvar_offset_end] =
      range_for_constraint<i_t, f_t>(fj_cpu, cstr_idx);
    for (auto i = relvar_offset_begin; i < relvar_offset_end; i++) {
      fj_cpu.cached_mtm_moves[i].first = 0;
    }
  }

  if (fj_cpu.violated_constraints.empty()) { fj_cpu.h_objective_weight += 1; }
}

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
  // Update the LHSs of all involved constraints.
  auto [offset_begin, offset_end] = reverse_range_for_var<i_t, f_t>(fj_cpu, var_idx);

  fj_cpu.nnz_processed_window += (offset_end - offset_begin);
  fj_cpu.n_variable_updates_window++;
  fj_cpu.unique_vars_accessed_window.insert(var_idx);

  i_t previous_viol = fj_cpu.violated_constraints.size();

  for (auto i = offset_begin; i < offset_end; i++) {
    cuopt_assert(i < (i_t)fj_cpu.h_reverse_constraints.size(), "");
    auto [c_lb, c_ub] = fj_cpu.cached_cstr_bounds[i].get();

    auto cstr_idx = fj_cpu.h_reverse_constraints[i];
    fj_cpu.unique_cstrs_accessed_window.insert(cstr_idx);
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

    if (new_cost < -cstr_tolerance && !fj_cpu.violated_constraints.count(cstr_idx)) {
      fj_cpu.violated_constraints.insert(cstr_idx);
      cuopt_assert(fj_cpu.satisfied_constraints.count(cstr_idx) == 1, "");
      fj_cpu.satisfied_constraints.erase(cstr_idx);
    } else if (!(new_cost < -cstr_tolerance) && fj_cpu.violated_constraints.count(cstr_idx)) {
      cuopt_assert(fj_cpu.satisfied_constraints.count(cstr_idx) == 0, "");
      fj_cpu.violated_constraints.erase(cstr_idx);
      fj_cpu.satisfied_constraints.insert(cstr_idx);
    }

    cuopt_assert(isfinite(delta), "delta should be finite");
    cuopt_assert(isfinite(fj_cpu.h_lhs[cstr_idx]), "assignment should be finite");

    // Invalidate related cached move scores
    auto [relvar_offset_begin, relvar_offset_end] =
      range_for_constraint<i_t, f_t>(fj_cpu, cstr_idx);
    for (auto i = relvar_offset_begin; i < relvar_offset_end; i++) {
      fj_cpu.cached_mtm_moves[i].first = 0;
    }
  }

  if (previous_viol > 0 && fj_cpu.violated_constraints.empty()) {
    fj_cpu.last_feasible_entrance_iter = fj_cpu.iterations;
  }

  // update the assignment and objective proper
  f_t new_val = fj_cpu.h_assignment[var_idx] + delta;
  if (is_integer_var<i_t, f_t>(fj_cpu, var_idx)) {
    cuopt_assert(fj_cpu.view.pb.integer_equal(new_val, round(new_val)), "new_val is not integer");
    new_val = round(new_val);
  }
  fj_cpu.h_assignment[var_idx] = new_val;

  cuopt_assert((check_variable_within_bounds<i_t, f_t>(fj_cpu, var_idx, new_val)),
               "assignment not within bounds");
  cuopt_assert(isfinite(new_val), "assignment is not finite");

  fj_cpu.h_incumbent_objective += fj_cpu.h_obj_coeffs[var_idx] * delta;
  if (fj_cpu.h_incumbent_objective < fj_cpu.h_best_objective &&
      fj_cpu.violated_constraints.empty()) {
    // recompute the LHS values to cancel out accumulation errors, then check if feasibility remains
    recompute_lhs(fj_cpu);

    if (fj_cpu.violated_constraints.empty() && check_variable_feasibility<i_t, f_t>(fj_cpu)) {
      cuopt_assert(fj_cpu.satisfied_constraints.size() == fj_cpu.view.pb.n_constraints, "");
      fj_cpu.h_best_objective =
        fj_cpu.h_incumbent_objective - fj_cpu.settings.parameters.breakthrough_move_epsilon;
      fj_cpu.h_best_assignment     = fj_cpu.h_assignment;
      fj_cpu.iterations_since_best = 0;
      CUOPT_LOG_TRACE("%sCPUFJ: new best objective: %g",
                      fj_cpu.log_prefix.c_str(),
                      fj_cpu.pb_ptr->get_user_obj_from_solver_obj(fj_cpu.h_best_objective));
      if (fj_cpu.improvement_callback) {
        double current_work_units = fj_cpu.work_units_elapsed.load(std::memory_order_acquire);
        fj_cpu.improvement_callback(
          fj_cpu.h_best_objective, fj_cpu.h_assignment, current_work_units);
      }
      fj_cpu.feasible_found = true;
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
  auto& problem = *fj_cpu.pb_ptr;

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
      if (auto& cached_move = fj_cpu.cached_mtm_moves[i]; cached_move.first != 0) {
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

      auto [score, infeasibility] = compute_score<i_t, f_t>(fj_cpu, var_idx, delta);
      fj_cpu.cached_mtm_moves[i]  = std::make_pair(delta, score);
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

template <typename i_t, typename f_t>
static thrust::tuple<fj_move_t, fj_staged_score_t> find_mtm_move_viol(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t sample_size = 100, bool localmin = false)
{
  timing_raii_t<i_t, f_t> timer(fj_cpu.find_mtm_move_viol_times);
  CPUFJ_NVTX_RANGE("CPUFJ::find_mtm_move_viol");

  std::vector<i_t> sampled_cstrs;
  sampled_cstrs.reserve(sample_size);
  std::sample(fj_cpu.violated_constraints.begin(),
              fj_cpu.violated_constraints.end(),
              std::back_inserter(sampled_cstrs),
              sample_size,
              std::mt19937(fj_cpu.settings.seed + fj_cpu.iterations));

  return find_mtm_move<i_t, f_t, MTMMoveType::FJ_MTM_VIOLATED>(fj_cpu, sampled_cstrs, localmin);
}

template <typename i_t, typename f_t>
static thrust::tuple<fj_move_t, fj_staged_score_t> find_mtm_move_sat(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu, i_t sample_size = 100)
{
  timing_raii_t<i_t, f_t> timer(fj_cpu.find_mtm_move_sat_times);
  CPUFJ_NVTX_RANGE("CPUFJ::find_mtm_move_sat");

  std::vector<i_t> sampled_cstrs;
  sampled_cstrs.reserve(sample_size);
  std::sample(fj_cpu.satisfied_constraints.begin(),
              fj_cpu.satisfied_constraints.end(),
              std::back_inserter(sampled_cstrs),
              sample_size,
              std::mt19937(fj_cpu.settings.seed + fj_cpu.iterations));

  return find_mtm_move<i_t, f_t, MTMMoveType::FJ_MTM_SATISFIED>(fj_cpu, sampled_cstrs);
}

template <typename i_t, typename f_t>
static void recompute_lhs(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  CPUFJ_NVTX_RANGE("CPUFJ::recompute_lhs");
  cuopt_assert(fj_cpu.h_lhs.size() == fj_cpu.view.pb.n_constraints, "h_lhs size mismatch");

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
}

template <typename i_t, typename f_t>
static thrust::tuple<fj_move_t, fj_staged_score_t> find_lift_move(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  timing_raii_t<i_t, f_t> timer(fj_cpu.find_lift_move_times);
  CPUFJ_NVTX_RANGE("CPUFJ::find_lift_move");

  fj_move_t best_move          = fj_move_t{-1, 0};
  fj_staged_score_t best_score = fj_staged_score_t::zero();

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

    // get the score
    auto move               = fj_move_t{var_idx, delta};
    fj_staged_score_t score = fj_staged_score_t::zero();
    f_t obj_score           = -1 * obj_coeff * delta;  // negated to turn this into a positive score
    score.base              = round(obj_score);

    if (best_score < score) {
      best_score = score;
      best_move  = move;
    }
  }

  return thrust::make_tuple(best_move, best_score);
}

template <typename i_t, typename f_t>
static void perturb(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  CPUFJ_NVTX_RANGE("CPUFJ::perturb");
  // select N variables, assign them a random value between their bounds
  std::vector<i_t> sampled_vars;
  std::sample(fj_cpu.h_objective_vars.begin(),
              fj_cpu.h_objective_vars.end(),
              std::back_inserter(sampled_vars),
              2,
              std::mt19937(fj_cpu.settings.seed + fj_cpu.iterations));
  raft::random::PCGenerator rng(fj_cpu.settings.seed + fj_cpu.iterations, 0, 0);

  for (auto var_idx : sampled_vars) {
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

  recompute_lhs(fj_cpu);
}

template <typename i_t, typename f_t>
static void init_fj_cpu(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                        solution_t<i_t, f_t>& solution,
                        const std::vector<f_t>& left_weights,
                        const std::vector<f_t>& right_weights,
                        f_t objective_weight)
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

  // set pointers to host copies
  // technically not 'device_span's but raft doesn't have a universal span.
  // cuda::std::span?
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
  fj_cpu.view.objective_vars =
    raft::device_span<i_t>(fj_cpu.h_objective_vars.data(), fj_cpu.h_objective_vars.size());
  fj_cpu.view.incumbent_objective = &fj_cpu.h_incumbent_objective;
  fj_cpu.view.best_objective      = &fj_cpu.h_best_objective;

  fj_cpu.view.settings = &fj_cpu.settings;
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
  fj_cpu.h_objective_vars.resize(problem.n_variables);
  auto end = std::copy_if(
    thrust::counting_iterator<i_t>(0),
    thrust::counting_iterator<i_t>(problem.n_variables),
    fj_cpu.h_objective_vars.begin(),
    [&fj_cpu](i_t idx) { return !fj_cpu.view.pb.integer_equal(fj_cpu.h_obj_coeffs[idx], (f_t)0); });
  fj_cpu.h_objective_vars.resize(end - fj_cpu.h_objective_vars.begin());

  fj_cpu.h_best_objective = +std::numeric_limits<f_t>::infinity();

  // nnz count
  fj_cpu.cached_mtm_moves.resize(fj_cpu.h_coefficients.size(),
                                 std::make_pair(0, fj_staged_score_t::zero()));

  fj_cpu.cached_cstr_bounds.resize(fj_cpu.h_reverse_coefficients.size());
  for (i_t var_idx = 0; var_idx < (i_t)fj_cpu.view.pb.n_variables; ++var_idx) {
    auto [offset_begin, offset_end] = reverse_range_for_var<i_t, f_t>(fj_cpu, var_idx);
    for (i_t i = offset_begin; i < offset_end; ++i) {
      fj_cpu.cached_cstr_bounds[i] =
        std::make_pair(fj_cpu.h_cstr_lb[fj_cpu.h_reverse_constraints[i]],
                       fj_cpu.h_cstr_ub[fj_cpu.h_reverse_constraints[i]]);
    }
  }

  fj_cpu.flip_move_computed.resize(fj_cpu.view.pb.n_variables, false);
  fj_cpu.var_bitmap.resize(fj_cpu.view.pb.n_variables, false);
  fj_cpu.iter_mtm_vars.reserve(fj_cpu.view.pb.n_variables);

  recompute_lhs(fj_cpu);

  // Precompute static problem features for regression model
  precompute_problem_features(fj_cpu);
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
    cuopt_assert(fj_cpu.satisfied_constraints.count(cstr_idx) == 0,
                 "Violated constraint also in satisfied_constraints");
    f_t lhs    = fj_cpu.h_lhs[cstr_idx];
    f_t tol    = fj_cpu.view.get_corrected_tolerance(cstr_idx);
    f_t excess = fj_cpu.view.excess_score(cstr_idx, lhs);
    cuopt_assert(excess < -tol, "Constraint in violated_constraints is not actually violated");
  }

  // Check that each satisfied constraint is actually satisfied and not present in
  // violated_constraints
  for (const auto& cstr_idx : fj_cpu.satisfied_constraints) {
    cuopt_assert(fj_cpu.violated_constraints.count(cstr_idx) == 0,
                 "Satisfied constraint also in violated_constraints");
    f_t lhs    = fj_cpu.h_lhs[cstr_idx];
    f_t tol    = fj_cpu.view.get_corrected_tolerance(cstr_idx);
    f_t excess = fj_cpu.view.excess_score(cstr_idx, lhs);
    cuopt_assert(!(excess < -tol), "Constraint in satisfied_constraints is actually violated");
  }

  // Check that each constraint is in exactly one of violated_constraints or satisfied_constraints
  for (i_t cstr_idx = 0; cstr_idx < fj_cpu.view.pb.n_constraints; ++cstr_idx) {
    bool in_viol = fj_cpu.violated_constraints.count(cstr_idx) > 0;
    bool in_sat  = fj_cpu.satisfied_constraints.count(cstr_idx) > 0;
    cuopt_assert(
      in_viol != in_sat,
      "Constraint must be in exactly one of violated_constraints or satisfied_constraints");

    cuopt_assert(fj_cpu.h_cstr_left_weights[cstr_idx] >= 0, "Weights should be positive or zero");
    cuopt_assert(fj_cpu.h_cstr_right_weights[cstr_idx] >= 0, "Weights should be positive or zero");
  }
  cuopt_assert(fj_cpu.h_objective_weight >= 0, "Objective weight should be positive or zero");
}

template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> fj_t<i_t, f_t>::create_cpu_climber(
  solution_t<i_t, f_t>& solution,
  const std::vector<f_t>& left_weights,
  const std::vector<f_t>& right_weights,
  f_t objective_weight,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings,
  bool randomize_params)
{
  raft::common::nvtx::range scope("fj_cpu_init");

  auto fj_cpu = std::make_unique<fj_cpu_climber_t<i_t, f_t>>(preemption_flag);

  // Initialize fj_cpu with all the data
  init_fj_cpu(*fj_cpu, solution, left_weights, right_weights, objective_weight);
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

template <typename i_t, typename f_t>
bool fj_t<i_t, f_t>::cpu_solve(fj_cpu_climber_t<i_t, f_t>& fj_cpu, f_t in_time_limit)
{
  raft::common::nvtx::range scope("fj_cpu");

  i_t local_mins       = 0;
  auto loop_start      = std::chrono::high_resolution_clock::now();
  auto time_limit      = std::chrono::milliseconds((int)(in_time_limit * 1000));
  auto loop_time_start = std::chrono::high_resolution_clock::now();

  // Initialize feature tracking
  fj_cpu.last_feature_log_time = loop_start;
  fj_cpu.prev_best_objective   = fj_cpu.h_best_objective;
  fj_cpu.iterations_since_best = 0;

  while (!fj_cpu.halted && !fj_cpu.preemption_flag.load()) {
    // Check if 5 seconds have passed
    auto now = std::chrono::high_resolution_clock::now();
    if (in_time_limit < std::numeric_limits<f_t>::infinity() &&
        now - loop_time_start > time_limit) {
      CUOPT_LOG_TRACE("%sTime limit of %.4f seconds reached, breaking loop at iteration %d",
                      fj_cpu.log_prefix.c_str(),
                      time_limit.count() / 1000.f,
                      fj_cpu.iterations);
      break;
    }
    if (fj_cpu.iterations >= fj_cpu.settings.iteration_limit) {
      CUOPT_LOG_TRACE("%sIteration limit of %d reached, breaking loop at iteration %d",
                      fj_cpu.log_prefix.c_str(),
                      fj_cpu.settings.iteration_limit,
                      fj_cpu.iterations);
      break;
    }

    // periodically recompute the LHS and violation scores
    // to correct any accumulated numerical errors
    cuopt_assert(fj_cpu.settings.parameters.lhs_refresh_period > 0,
                 "lhs_refresh_period should be positive");
    if (fj_cpu.iterations % fj_cpu.settings.parameters.lhs_refresh_period == 0 ||
        fj_cpu.trigger_early_lhs_recomputation) {
      recompute_lhs(fj_cpu);
      fj_cpu.trigger_early_lhs_recomputation = false;
    }

    fj_move_t move          = fj_move_t{-1, 0};
    fj_staged_score_t score = fj_staged_score_t::invalid();
    bool is_lift            = false;
    bool is_mtm_viol        = false;
    bool is_mtm_sat         = false;

    // Perform lift moves
    if (fj_cpu.violated_constraints.empty()) {
      thrust::tie(move, score) = find_lift_move(fj_cpu);
      if (score > fj_staged_score_t::zero()) is_lift = true;
    }
    // Regular MTM
    if (!(score > fj_staged_score_t::zero())) {
      thrust::tie(move, score) = find_mtm_move_viol(fj_cpu, fj_cpu.mtm_viol_samples);
      if (score > fj_staged_score_t::zero()) is_mtm_viol = true;
    }
    // try with MTM in satisfied constraints
    if (fj_cpu.feasible_found && !(score > fj_staged_score_t::zero())) {
      thrust::tie(move, score) = find_mtm_move_sat(fj_cpu, fj_cpu.mtm_sat_samples);
      if (score > fj_staged_score_t::zero()) is_mtm_sat = true;
    }
    // if we're in the feasible region but haven't found improvements in the last n iterations,
    // perturb
    bool should_perturb = false;
    if (fj_cpu.violated_constraints.empty() &&
        fj_cpu.iterations - fj_cpu.last_feasible_entrance_iter > fj_cpu.perturb_interval) {
      should_perturb                     = true;
      fj_cpu.last_feasible_entrance_iter = fj_cpu.iterations;
    }

    if (score > fj_staged_score_t::zero() && !should_perturb) {
      apply_move(fj_cpu, move.var_idx, move.value, false);
      // Track move types
      if (is_lift) fj_cpu.n_lift_moves_window++;
      if (is_mtm_viol) fj_cpu.n_mtm_viol_moves_window++;
      if (is_mtm_sat) fj_cpu.n_mtm_sat_moves_window++;
    } else {
      // Local Min
      update_weights(fj_cpu);
      if (should_perturb) {
        perturb(fj_cpu);
        for (size_t i = 0; i < fj_cpu.cached_mtm_moves.size(); i++)
          fj_cpu.cached_mtm_moves[i].first = 0;
      }
      thrust::tie(move, score) =
        find_mtm_move_viol(fj_cpu, 1, true);  // pick a single random violated constraint
      i_t var_idx = move.var_idx >= 0 ? move.var_idx : 0;
      f_t delta   = move.var_idx >= 0 ? move.value : 0;
      apply_move(fj_cpu, var_idx, delta, true);
      ++local_mins;
      ++fj_cpu.n_local_minima_window;
    }

    // number of violated constraints is usually small (<100). recomputing from all LHSs is cheap
    // and more numerically precise than just adding to the accumulator in apply_move
    fj_cpu.total_violations = 0;
    for (auto cstr_idx : fj_cpu.violated_constraints) {
      fj_cpu.total_violations += fj_cpu.view.excess_score(cstr_idx, fj_cpu.h_lhs[cstr_idx]);
    }
    if (fj_cpu.iterations % fj_cpu.log_interval == 0) {
      CUOPT_LOG_TRACE(
        "%sCPUFJ iteration: %d/%d, local mins: %d, best_objective: %g, viol: %zu, obj weight %g, "
        "maxw "
        "%g",
        fj_cpu.log_prefix.c_str(),
        fj_cpu.iterations,
        fj_cpu.settings.iteration_limit != std::numeric_limits<i_t>::max()
          ? fj_cpu.settings.iteration_limit
          : -1,
        local_mins,
        fj_cpu.pb_ptr->get_user_obj_from_solver_obj(fj_cpu.h_best_objective),
        fj_cpu.violated_constraints.size(),
        fj_cpu.h_objective_weight,
        fj_cpu.max_weight);
    }
    // send current solution to callback every 3000 steps for diversity
    if (fj_cpu.iterations % fj_cpu.diversity_callback_interval == 0) {
      if (fj_cpu.diversity_callback) {
        fj_cpu.diversity_callback(fj_cpu.h_incumbent_objective, fj_cpu.h_assignment);
      }
    }

    // Print timing statistics every N iterations
#if CPUFJ_TIMING_TRACE
    if (fj_cpu.iterations % fj_cpu.timing_stats_interval == 0 && fj_cpu.iterations > 0) {
      print_timing_stats(fj_cpu);
    }
#endif

    if (fj_cpu.iterations % 100 == 0 && fj_cpu.iterations > 0) {
      // Collect memory statistics
      auto [loads, stores] = fj_cpu.memory_aggregator.collect();

      double biased_work = (loads + stores) * fj_cpu.work_unit_bias / 1e10;
      fj_cpu.work_units_elapsed += biased_work;

      if (fj_cpu.producer_sync != nullptr) { fj_cpu.producer_sync->notify_progress(); }

      CUOPT_LOG_TRACE("CPUFJ work units: %f incumbent %g",
                      fj_cpu.work_units_elapsed.load(std::memory_order_relaxed),
                      fj_cpu.pb_ptr->get_user_obj_from_solver_obj(fj_cpu.h_best_objective));
    }

    cuopt_func_call(sanity_checks(fj_cpu));
    fj_cpu.iterations++;
    fj_cpu.iterations_since_best++;
  }
  auto loop_end = std::chrono::high_resolution_clock::now();
  double total_time =
    std::chrono::duration_cast<std::chrono::duration<double>>(loop_end - loop_start).count();
  double avg_time_per_iter = total_time / fj_cpu.iterations;
  CUOPT_LOG_TRACE("%sCPUFJ Average time per iteration: %.8fms",
                  fj_cpu.log_prefix.c_str(),
                  avg_time_per_iter * 1000.0);

#if CPUFJ_TIMING_TRACE
  // Print final timing statistics
  CUOPT_LOG_TRACE("=== Final Timing Statistics ===");
  print_timing_stats(fj_cpu);
#endif

  return fj_cpu.feasible_found;
}

template <typename i_t, typename f_t>
cpu_fj_thread_t<i_t, f_t>::~cpu_fj_thread_t()
{
  this->request_termination();
}

template <typename i_t, typename f_t>
void cpu_fj_thread_t<i_t, f_t>::run_worker()
{
  bool solution_found   = fj_ptr->cpu_solve(*fj_cpu, time_limit);
  cpu_fj_solution_found = solution_found;
}

template <typename i_t, typename f_t>
void cpu_fj_thread_t<i_t, f_t>::on_terminate()
{
  if (fj_cpu) fj_cpu->halted = true;
}

template <typename i_t, typename f_t>
void cpu_fj_thread_t<i_t, f_t>::on_start()
{
  cuopt_assert(fj_cpu != nullptr, "fj_cpu must not be null");
  fj_cpu->halted = false;
}

template <typename i_t, typename f_t>
void cpu_fj_thread_t<i_t, f_t>::stop_cpu_solver()
{
  fj_cpu->halted = true;
}

#if MIP_INSTANTIATE_FLOAT
template class fj_t<int, float>;
template class cpu_fj_thread_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class fj_t<int, double>;
template class cpu_fj_thread_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
