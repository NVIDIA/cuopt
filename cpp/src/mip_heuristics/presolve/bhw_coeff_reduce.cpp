/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "bhw_coeff_reduce.hpp"

#include <mip_heuristics/mip_constants.hpp>
#include <utilities/integer_scaling.hpp>
#include <utilities/logger.hpp>
#include <utilities/macros.cuh>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

// Bradley, Hammer and Wolsey (1974), "Coefficient reduction for inequalities in 0-1 variables."
// Theorem 2.5 separates maximal feasible from minimal infeasible points. Positive normalized
// coefficients make activity monotone, so these supersets of BHW's ceilings and roofs suffice.
// Search non-negative, non-increasing weights by increasing max|w|; Lemma 3.6 bounds and prunes it.

namespace cuopt::mathematical_optimization::mip {

// N1 complements negative coefficients; N3 sorts them descending. N1 makes activity monotone and
// the weights of equivalent inequalities non-negative.
struct norm_row_t {
  int k = 0;
  std::array<int64_t, BHW_MAX_LEN> coef{};  // descending, all > 0
  std::array<int, BHW_MAX_LEN> slot{};      // position of this entry in the caller's row arrays
  std::array<bool, BHW_MAX_LEN> flipped{};  // whether this entry was complemented by N1
  int64_t rhs = 0;
};

struct partition_t {
  std::vector<uint32_t> maximal_feasible;
  std::vector<uint32_t> minimal_infeasible;
  // Lemma 3.6: where coef[i] > coef[i+1] and the two variables are not symmetric, every equivalent
  // inequality has w_i >= w_{i+1} + 1. Chaining those steps down to w_{k-1} >= 0 bounds max|w|.
  std::array<bool, BHW_MAX_LEN> strict{};
  std::array<int, BHW_MAX_LEN + 1> suffix_strict{};
  int lemma36_bound = 0;
};

static int64_t weight_activity(const int64_t* w, uint32_t mask)
{
  int64_t sum = 0;
  while (mask != 0u) {
    sum += w[std::countr_zero(mask)];
    mask &= mask - 1u;
  }
  return sum;
}

template <typename f_t>
static bool integerization_preserves_binary_feasible_set(
  const f_t* coefficients,
  int len,
  f_t side,
  int direction,
  const std::array<int64_t, BHW_MAX_LEN>& integral,
  int64_t integral_side)
{
  cuopt_assert(len >= 2 && len <= BHW_MAX_LEN, "row length outside the enumerable range");
  const uint32_t n_pat = 1u << len;
  std::vector<_Float128> original_activity(n_pat, 0.0L);
  std::vector<int64_t> integral_activity(n_pat, 0);
  for (uint32_t m = 1; m < n_pat; ++m) {
    const uint32_t previous = m & (m - 1u);
    const int j             = std::countr_zero(m);
    original_activity[m]    = original_activity[previous] + coefficients[j];
    integral_activity[m]    = integral_activity[previous] + integral[j];
  }

  for (uint32_t m = 0; m < n_pat; ++m) {
    const bool original_feasible = direction == 1 ? original_activity[m] <= side
                                                  : original_activity[m] >= side;
    const bool integral_feasible = integral_activity[m] <= integral_side;
    if (original_feasible != integral_feasible) return false;
  }
  return true;
}

// Collects maximal feasible and minimal infeasible points; rejects degenerate partitions.
static bool build_partition(const norm_row_t& row, partition_t& out)
{
  const int k          = row.k;
  const uint32_t n_pat = 1u << k;
  cuopt_assert(k >= 2 && k <= BHW_MAX_LEN, "row length outside the enumerable range");

  std::vector<int64_t> activity(n_pat, 0);
  std::vector<uint8_t> feasible(n_pat, 0);
  uint32_t n_feasible = 0;
  for (uint32_t m = 1; m < n_pat; ++m)
    activity[m] = activity[m & (m - 1u)] + row.coef[std::countr_zero(m)];
  for (uint32_t m = 0; m < n_pat; ++m) {
    feasible[m] = activity[m] <= row.rhs ? 1 : 0;
    n_feasible += feasible[m];
  }
  if (n_feasible == 0 || n_feasible == n_pat) return false;

  for (uint32_t m = 0; m < n_pat; ++m) {
    bool extremal = true;
    if (feasible[m] != 0) {
      for (int i = 0; i < k && extremal; ++i)
        if ((m >> i & 1u) == 0u && feasible[m | (1u << i)] != 0) extremal = false;
      if (extremal) out.maximal_feasible.push_back(m);
    } else {
      for (int i = 0; i < k && extremal; ++i)
        if ((m >> i & 1u) != 0u && feasible[m ^ (1u << i)] == 0) extremal = false;
      if (extremal) out.minimal_infeasible.push_back(m);
    }
  }
  cuopt_assert(!out.maximal_feasible.empty() && !out.minimal_infeasible.empty(),
               "a non-degenerate partition has at least one extremal point on each side");

  for (int i = 0; i + 1 < k; ++i) {
    if (row.coef[i] <= row.coef[i + 1]) continue;
    const uint32_t lo_bit = 1u << i;
    const uint32_t hi_bit = 1u << (i + 1);
    for (uint32_t m = 0; m < n_pat; ++m) {
      if ((m & lo_bit) != 0u || (m & hi_bit) == 0u) continue;
      // coef[i] > coef[i+1], so moving the set bit down raises the activity: a feasible point whose
      // swap is infeasible witnesses that the two variables are not interchangeable.
      if (feasible[m] != 0 && feasible[(m ^ hi_bit) | lo_bit] == 0) {
        out.strict[i] = true;
        break;
      }
    }
  }
  for (int i = k - 1; i >= 0; --i)
    out.suffix_strict[i] = out.suffix_strict[i + 1] + (out.strict[i] ? 1 : 0);
  out.lemma36_bound = out.suffix_strict[0];
  cuopt_assert(out.lemma36_bound < k, "at most k-1 strict steps exist in a row of length k");
  return true;
}

// BHW Theorem 2.5: integer w is equivalent iff its maximum over maximal feasible points is below
// its minimum over minimal infeasible points.
static bool accepts(const partition_t& part, const int64_t* w, int64_t& bound)
{
  int64_t hi = std::numeric_limits<int64_t>::min();
  for (uint32_t m : part.maximal_feasible)
    hi = std::max(hi, weight_activity(w, m));
  int64_t lo = std::numeric_limits<int64_t>::max();
  for (uint32_t m : part.minimal_infeasible)
    lo = std::min(lo, weight_activity(w, m));
  bound = hi;
  return hi < lo;
}

// Maximizing a.x over [0,1]^k with w.x <= t is a fractional knapsack. Greedy leaves at most one
// fractional entry, allowing the containment check to close exactly in integer arithmetic.
static bool lp_no_weakening(const norm_row_t& row, const int64_t* w, int64_t t)
{
  cuopt_assert(t >= 0, "acceptance implies the origin is feasible, so the bound is non-negative");

  std::array<int, BHW_MAX_LEN> order{};
  int n_items    = 0;
  __int128 value = 0;
  for (int i = 0; i < row.k; ++i) {
    if (w[i] == 0)
      value += row.coef[i];
    else
      order[n_items++] = i;
  }
  std::sort(order.begin(), order.begin() + n_items, [&](int x, int y) {
    const __int128 dx = (__int128)row.coef[x] * w[y];
    const __int128 dy = (__int128)row.coef[y] * w[x];
    return dx != dy ? dx > dy : x < y;
  });

  int64_t capacity = t;
  int fractional   = -1;
  for (int p = 0; p < n_items; ++p) {
    const int i = order[p];
    if (w[i] <= capacity) {
      value += row.coef[i];
      capacity -= w[i];
    } else {
      if (capacity > 0) fractional = i;
      break;
    }
  }

  if (fractional < 0) return value <= (__int128)row.rhs;
  return value * w[fractional] + (__int128)capacity * row.coef[fractional] <=
         (__int128)row.rhs * w[fractional];
}

[[maybe_unused]] static bool verify_equivalent(const norm_row_t& row,
                                               const int64_t* w,
                                               int64_t t)
{
  const uint32_t n_pat = 1u << row.k;
  for (uint32_t m = 0; m < n_pat; ++m) {
    int64_t a_activity = 0;
    int64_t w_activity = 0;
    for (int i = 0; i < row.k; ++i) {
      if ((m >> i & 1u) == 0u) continue;
      a_activity += row.coef[i];
      w_activity += w[i];
    }
    if ((a_activity <= row.rhs) != (w_activity <= t)) return false;
  }
  return true;
}

struct search_state_t {
  const norm_row_t* row   = nullptr;
  const partition_t* part = nullptr;
  std::array<int64_t, BHW_MAX_LEN> w{};
  std::array<int64_t, BHW_MAX_LEN> best_w{};
  int64_t best_bound = 0;
  int best_nonzeros  = 0;
  int64_t best_sum   = 0;
  bool found         = false;
};

// Enumerates non-negative, non-increasing weights; Lemma 3.6 bounds each suffix.
static void search_positions(search_state_t& st, int pos)
{
  const int k = st.row->k;
  if (pos == k) {
    int64_t bound = 0;
    if (!accepts(*st.part, st.w.data(), bound)) return;
    if (!lp_no_weakening(*st.row, st.w.data(), bound)) return;

    int nonzeros = 0;
    int64_t sum  = 0;
    for (int i = 0; i < k; ++i) {
      nonzeros += st.w[i] != 0 ? 1 : 0;
      sum += st.w[i];
    }
    // At fixed max|w|, prefer fewer nonzeros, then smaller sum.
    if (st.found &&
        (nonzeros > st.best_nonzeros || (nonzeros == st.best_nonzeros && sum >= st.best_sum)))
      return;
    st.found         = true;
    st.best_nonzeros = nonzeros;
    st.best_sum      = sum;
    st.best_bound    = bound;
    st.best_w        = st.w;
    return;
  }

  const int64_t upper = st.w[pos - 1] - (st.part->strict[pos - 1] ? 1 : 0);
  const int64_t lower = st.part->suffix_strict[pos];
  for (int64_t v = upper; v >= lower; --v) {
    st.w[pos] = v;
    search_positions(st, pos + 1);
  }
}

// Above the exact-search cap, try round(a/min(a)) and the all-ones row through the same gates.
static bool heuristic_reduce(const norm_row_t& row,
                             const partition_t& part,
                             std::vector<int64_t>& weights,
                             int64_t& bound)
{
  const int k         = row.k;
  const int64_t a_min = row.coef[k - 1];
  cuopt_assert(a_min > 0, "N1 leaves every coefficient positive");

  int64_t best_max  = row.coef[0];
  int best_nonzeros = k;
  bool found        = false;

  std::array<int64_t, BHW_MAX_LEN> candidate{};
  for (int variant = 0; variant < 2; ++variant) {
    for (int i = 0; i < k; ++i)
      candidate[i] = variant == 0 ? (row.coef[i] + a_min / 2) / a_min : 1;

    int64_t candidate_bound = 0;
    if (!accepts(part, candidate.data(), candidate_bound)) continue;
    if (!lp_no_weakening(row, candidate.data(), candidate_bound)) continue;

    int64_t candidate_max = 0;
    int nonzeros          = 0;
    for (int i = 0; i < k; ++i) {
      candidate_max = std::max(candidate_max, candidate[i]);
      nonzeros += candidate[i] != 0 ? 1 : 0;
    }
    if (candidate_max > best_max || (candidate_max == best_max && nonzeros >= best_nonzeros))
      continue;

    found         = true;
    best_max      = candidate_max;
    best_nonzeros = nonzeros;
    weights.assign(candidate.begin(), candidate.begin() + k);
    bound = candidate_bound;
  }
  return found;
}

static bool reduce_shape(const norm_row_t& row, std::vector<int64_t>& weights, int64_t& bound)
{
  partition_t part;
  if (!build_partition(row, part)) return false;

  const int64_t current = row.coef[0];
  cuopt_assert(current >= 2, "rows already at magnitude one are rejected before normalization");
  // Lemma 3.6 bounds max|w| from below over every equivalent inequality, so this row is provably
  // irreducible in magnitude and not worth searching.
  if (part.lemma36_bound >= current) return false;

  search_state_t st;
  st.row               = &row;
  st.part              = &part;
  const int64_t m_high = std::min<int64_t>(BHW_EXACT_MAX_WEIGHT, current - 1);
  for (int64_t m = std::max<int64_t>(part.lemma36_bound, 1); m <= m_high; ++m) {
    st.found = false;
    st.w[0]  = m;
    search_positions(st, 1);
    if (!st.found) continue;
    // First m with any acceptance, so this is the minimum achievable max|w|.
    weights.assign(st.best_w.begin(), st.best_w.begin() + row.k);
    bound = st.best_bound;
    return true;
  }
  return heuristic_reduce(row, part, weights, bound);
}

template <typename f_t>
bhw_row_rewrite_t bhw_reduce_row(
  const f_t* coefficients, int len, f_t side, int direction, bhw_shape_cache_t* cache)
{
  cuopt_assert(direction == 1 || direction == -1,
               "direction is the sign that orients the row to <=");
  bhw_row_rewrite_t rewrite;
  if (len < 2 || len > BHW_MAX_LEN) return rewrite;
  if (!scaling_bound_finite(side)) return rewrite;

  const double scale = row_int_scale<f_t>(
    coefficients, len, side, std::numeric_limits<f_t>::infinity(), BHW_MAX_LEN, BHW_INT_SCALE_MAX);
  if (scale == 0.0) return rewrite;

  std::array<int64_t, BHW_MAX_LEN> integral{};
  int64_t largest = 0;
  for (int j = 0; j < len; ++j) {
    integral[j] = std::llround((double)coefficients[j] * scale) * direction;
    if (integral[j] == 0) return rewrite;
    largest = std::max(largest, std::abs(integral[j]));
  }
  // can't coefficient-reduce a unit-magnitude row any further
  if (largest <= 1) return rewrite;

  norm_row_t norm_row;
  norm_row.k                 = len;
  const int64_t integral_side = std::llround((double)side * scale) * direction;
  norm_row.rhs               = integral_side;
  std::array<int, BHW_MAX_LEN> order{};
  for (int j = 0; j < len; ++j) {
    order[j] = j;
    // N1: complementing x_j = 1 - y_j moves the negative coefficient onto the right-hand side.
    if (integral[j] < 0) norm_row.rhs -= integral[j];
  }
  // N3: descending by magnitude, ties broken by position so the shape key is deterministic.
  std::sort(order.begin(), order.begin() + len, [&](int x, int y) {
    const int64_t ax = std::abs(integral[x]);
    const int64_t ay = std::abs(integral[y]);
    return ax != ay ? ax > ay : x < y;
  });
  for (int p = 0; p < len; ++p) {
    const int j         = order[p];
    norm_row.coef[p]    = std::abs(integral[j]);
    norm_row.slot[p]    = j;
    norm_row.flipped[p] = integral[j] < 0;
  }

  bhw_shape_result_t computed;
  const bhw_shape_result_t* result = nullptr;
  if (cache != nullptr) {
    std::vector<int64_t> key(norm_row.coef.begin(), norm_row.coef.begin() + len);
    key.push_back(norm_row.rhs);
    auto cached = cache->find(key);
    if (cached == cache->end()) {
      bhw_shape_result_t fresh;
      fresh.accepted = reduce_shape(norm_row, fresh.weights, fresh.bound);
      cached         = cache->emplace(std::move(key), std::move(fresh)).first;
    }
    result = &cached->second;
  } else {
    computed.accepted = reduce_shape(norm_row, computed.weights, computed.bound);
    result            = &computed;
  }
  if (!result->accepted) return rewrite;

  // check the feasible set remains unchanged under floating point math
  if (!integerization_preserves_binary_feasible_set(
        coefficients, len, side, direction, integral, integral_side))
    return rewrite;

  cuopt_assert((int)result->weights.size() == len, "cached shape has the wrong length");
  cuopt_assert(*std::min_element(result->weights.begin(), result->weights.end()) >= 0,
               "N1 leaves the reduced weights non-negative");
  cuopt_assert(
    *std::max_element(result->weights.begin(), result->weights.end()) == result->weights[0],
    "N3 leaves the reduced weights non-increasing");
  cuopt_assert(result->weights[0] < norm_row.coef[0] ||
                 std::count(result->weights.begin(), result->weights.end(), 0) > 0,
               "an accepted rewrite must shrink the magnitude or drop a variable");
  cuopt_assert(verify_equivalent(norm_row, result->weights.data(), result->bound),
               "BHW rewrite changed the 0/1 feasible set");

  // Undo N3 and N1, then undo the orientation. Complementing back turns w_i y_i into w_i - w_i x_i,
  // which flips the coefficient and moves w_i onto the bound.
  rewrite.coefficients.assign(len, 0);
  int64_t new_side = result->bound;
  for (int p = 0; p < len; ++p) {
    const int j = norm_row.slot[p];
    if (norm_row.flipped[p]) {
      rewrite.coefficients[j] = -result->weights[p];
      new_side -= result->weights[p];
    } else {
      rewrite.coefficients[j] = result->weights[p];
    }
  }
  for (int j = 0; j < len; ++j)
    rewrite.coefficients[j] *= direction;
  rewrite.side            = new_side * direction;
  rewrite.max_coef_before = norm_row.coef[0];
  rewrite.max_coef_after  = result->weights[0];
  rewrite.accepted        = true;
  return rewrite;
}

struct bhw_stats_t {
#if (CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_DEBUG)
  int64_t coefficients_reduced = 0;
  int64_t coefficients_dropped = 0;
  std::vector<double> row_shrinks;

  void changed_coefficient(int64_t new_coefficient)
  {
    ++coefficients_reduced;
    coefficients_dropped += new_coefficient == 0;
  }

  void rewrote_row(int64_t max_coef_before, int64_t max_coef_after)
  {
    row_shrinks.push_back((double)max_coef_before / max_coef_after);
  }

  void report()
  {
    if (coefficients_reduced == 0) return;
    const size_t n_rows_rewritten = row_shrinks.size();
    cuopt_assert(n_rows_rewritten > 0, "a changed coefficient implies an accepted row");
    const double mean =
      std::accumulate(row_shrinks.begin(), row_shrinks.end(), 0.0) / n_rows_rewritten;
    const auto middle = row_shrinks.begin() + n_rows_rewritten / 2;
    std::nth_element(row_shrinks.begin(), middle, row_shrinks.end());
    double median = *middle;
    if (n_rows_rewritten % 2 == 0)
      median = (median + *std::max_element(row_shrinks.begin(), middle)) / 2.0;

    CUOPT_LOG_DEBUG(
      "BHW reduced %ld coefficients (%ld dropped) in %zu rows, "
      "max|a| shrank %.1fx mean, %.1fx median",
      coefficients_reduced,
      coefficients_dropped,
      n_rows_rewritten,
      mean,
      median);
  }
#else
  void changed_coefficient(int64_t) {}
  void rewrote_row(int64_t, int64_t) {}
  void report() {}
#endif
};

template <typename f_t>
papilo::PresolveStatus BHWCoeffReduce<f_t>::execute(const papilo::Problem<f_t>& problem,
                                                    const papilo::ProblemUpdate<f_t>& problemUpdate,
                                                    const papilo::Num<f_t>& num,
                                                    papilo::Reductions<f_t>& reductions,
                                                    const papilo::Timer& timer,
                                                    int& reason_of_infeasibility)
{
  const auto& constraint_matrix = problem.getConstraintMatrix();
  const auto& lhs_values        = constraint_matrix.getLeftHandSides();
  const auto& rhs_values        = constraint_matrix.getRightHandSides();
  const auto& row_flags         = constraint_matrix.getRowFlags();
  const auto& domains           = problem.getVariableDomains();
  const auto& col_flags         = domains.flags;
  const auto& lower_bounds      = domains.lower_bounds;
  const auto& upper_bounds      = domains.upper_bounds;
  const auto& presolve_options  = problemUpdate.getPresolveOptions();

  const int num_rows            = constraint_matrix.getNRows();
  papilo::PresolveStatus status = papilo::PresolveStatus::kUnchanged;
  bhw_stats_t stats;

  // getChangedActivities() omits side-only changes, so screen every row. cache hits amortize
  for (int row = 0; row < num_rows; ++row) {
    if (reductions.size() >= presolve_options.max_reduction_seq) break;
    if (papilo::PresolveMethod<f_t>::is_interrupted(
          timer, presolve_options.tlim, presolve_options.early_exit_callback))
      break;

    auto row_coefficients = constraint_matrix.getRowCoefficients(row);
    const int len         = row_coefficients.getLength();
    if (len < 2 || len > BHW_MAX_LEN) continue;

    const auto& row_flag = row_flags[row];
    if (row_flag.test(papilo::RowFlag::kRedundant)) continue;
    const bool lhs_infinite = row_flag.test(papilo::RowFlag::kLhsInf);
    const bool rhs_infinite = row_flag.test(papilo::RowFlag::kRhsInf);
    // Equal flags mean either a ranged row / equation (both sides finite) or a free row.
    if (lhs_infinite == rhs_infinite) continue;

    const int* indices = row_coefficients.getIndices();
    const f_t* values  = row_coefficients.getValues();
    bool all_binary    = true;
    for (int j = 0; j < len && all_binary; ++j) {
      const int col = indices[j];
      all_binary    = col_flags[col].test(papilo::ColFlag::kIntegral) &&
                   !col_flags[col].test(papilo::ColFlag::kLbInf) &&
                   !col_flags[col].test(papilo::ColFlag::kUbInf) &&
                   !col_flags[col].test(papilo::ColFlag::kFixed) && num.isZero(lower_bounds[col]) &&
                   num.isEq(upper_bounds[col], f_t{1});
    }
    if (!all_binary) continue;

    const int direction = lhs_infinite ? 1 : -1;
    const f_t side      = lhs_infinite ? rhs_values[row] : lhs_values[row];
    const bhw_row_rewrite_t rewrite =
      bhw_reduce_row<f_t>(values, len, side, direction, &shape_cache_);
    if (!rewrite.accepted) continue;

    cuopt_assert(rewrite.max_coef_after >= 1,
                 "an accepted rewrite keeps at least one nonzero weight");
    stats.rewrote_row(rewrite.max_coef_before, rewrite.max_coef_after);

    papilo::TransactionGuard<f_t> guard{reductions};
    reductions.lockRow(row);
    [[maybe_unused]] int emitted = 0;
    for (int j = 0; j < len; ++j) {
      if ((f_t)rewrite.coefficients[j] == values[j]) continue;
      reductions.changeMatrixEntry(row, indices[j], (f_t)rewrite.coefficients[j]);
      ++emitted;
      stats.changed_coefficient(rewrite.coefficients[j]);
    }
    if (direction == 1) {
      if ((f_t)rewrite.side != rhs_values[row]) {
        reductions.changeRowRHS(row, (f_t)rewrite.side);
        ++emitted;
      }
    } else {
      if ((f_t)rewrite.side != lhs_values[row]) {
        reductions.changeRowLHS(row, (f_t)rewrite.side);
        ++emitted;
      }
    }
    cuopt_assert(emitted > 0, "accepted rewrite emitted no reduction");
    status = papilo::PresolveStatus::kReduced;
  }

  stats.report();

  return status;
}

#define INSTANTIATE(F_TYPE)                          \
  template class BHWCoeffReduce<F_TYPE>;             \
  template bhw_row_rewrite_t bhw_reduce_row<F_TYPE>( \
    const F_TYPE*, int, F_TYPE, int, bhw_shape_cache_t*);

#if MIP_INSTANTIATE_FLOAT || PDLP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::mathematical_optimization::mip
