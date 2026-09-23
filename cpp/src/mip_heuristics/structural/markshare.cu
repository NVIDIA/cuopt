/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "markshare.cuh"

#include <linear_algebra/sparse_matrix.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/utils.cuh>
#include <pdlp/translate.hpp>
#include <utilities/logger.hpp>
#include <utilities/macros.cuh>
#include <utilities/scope_guard.hpp>

#include <omp.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <format>
#include <numeric>
#include <string>
#include <type_traits>

// Unreachable residual. Also caps the core column count: a prefix length is stored in a byte.
#define MARKSHARE_UNREACHABLE 0xFF

namespace cuopt::mathematical_optimization::mip {

namespace {

// Shifts `src` left by `shift` bits into `dst`, truncated to `bits` significant bits.
// `dst` must not alias `src`.
void shift_left_into(const uint64_t* src, uint64_t* dst, size_t words, size_t shift, size_t bits)
{
  const size_t word_shift = shift / 64;
  const size_t bit_shift  = shift % 64;
  for (size_t w = words; w-- > 0;) {
    uint64_t value = 0;
    if (w >= word_shift) {
      value = src[w - word_shift] << bit_shift;
      // x >> 64 is undefined behaviour, so the carry-in is only valid for a nonzero shift
      if (bit_shift != 0 && w > word_shift) {
        value |= src[w - word_shift - 1] >> (64 - bit_shift);
      }
    }
    dst[w] = value;
  }
  const size_t tail = bits % 64;
  if (tail != 0) { dst[words - 1] &= (uint64_t(1) << tail) - 1; }
}

// Mirrors branch_and_bound.cpp so the two progress tables read the same.
template <typename f_t>
std::string to_percentage(f_t value)
{
  if (value == std::numeric_limits<f_t>::infinity()) return "-";
  if (value > 1e-3) { return std::format("{:5.1f}%", value * 100); }
  return std::format("{:5.2f}%", value * 100);
}

/**
 * @brief Subset sum reachability table for a single row.
 *
 * f[u] is the smallest prefix length p such that some subset of coefficients[0, p) sums to exactly
 * u, for every u in [0, capacity]. f[0] is 0 (the empty subset); unreachable residuals carry the
 * sentinel. The prefix is a byte because recognition caps the core column count well below it,
 * which halves the cache pressure of the hottest table lookup.
 *
 * Requires every coefficient to be non-negative and coefficients.size() below the sentinel.
 */
void build_row_table(const std::vector<int32_t>& coefficients,
                     int32_t capacity,
                     std::vector<uint8_t>& f)
{
  const size_t bits  = size_t(capacity) + 1;
  const size_t words = (bits + 63) / 64;

  f.assign(bits, MARKSHARE_UNREACHABLE);
  f[0] = 0;  // the empty subset reaches a residual of zero

  std::vector<uint64_t> reach(words, 0);
  std::vector<uint64_t> shifted(words, 0);
  reach[0] = 1;

  for (size_t j = 0; j < coefficients.size(); ++j) {
    const int32_t a = coefficients[j];
    if (a == 0 || a > capacity) { continue; }

    shift_left_into(reach.data(), shifted.data(), words, size_t(a), bits);

    for (size_t w = 0; w < words; ++w) {
      uint64_t fresh = shifted[w] & ~reach[w];
      reach[w] |= shifted[w];
      while (fresh != 0) {
        const size_t u = w * 64 + size_t(std::countr_zero(fresh));
        f[u]           = uint8_t(j + 1);
        fresh &= fresh - 1;
      }
    }
  }
}

/**
 * @brief Joint subset sum reachability table over two rows.
 *
 * f[u0 * (capacity1 + 1) + u1] is the smallest prefix length p such that a *single* subset of
 * [0, p) sums to u0 in the first row and simultaneously to u1 in the second. This is far stronger
 * than the conjunction of the two single row tables, and it is what makes the search tractable.
 *
 * Both coefficient vectors must be non-negative and of equal length.
 */
void build_joint_table(const std::vector<int32_t>& coefficients0,
                       const std::vector<int32_t>& coefficients1,
                       int32_t capacity0,
                       int32_t capacity1,
                       std::vector<uint8_t>& f)
{
  const size_t bits1 = size_t(capacity1) + 1;
  // Each u0 block is padded to a whole number of words, so a shift within a block physically
  // cannot alias into the u0 + 1 block. That makes the boundary mask implicit in the padding.
  const size_t stride_words = (bits1 + 63) / 64;
  const size_t blocks       = size_t(capacity0) + 1;

  f.assign(blocks * bits1, MARKSHARE_UNREACHABLE);
  f[0] = 0;

  std::vector<uint64_t> reach(blocks * stride_words, 0);
  std::vector<uint64_t> shifted(stride_words, 0);
  reach[0] = 1;

  for (size_t j = 0; j < coefficients0.size(); ++j) {
    const int32_t a0 = coefficients0[j];
    const int32_t a1 = coefficients1[j];
    if (a0 == 0 && a1 == 0) { continue; }
    if (a0 > capacity0 || a1 > capacity1) { continue; }

    // Descending u0 is what keeps this a 0/1 knapsack rather than an unbounded one.
    for (size_t u0 = blocks; u0-- > size_t(a0);) {
      const uint64_t* src = &reach[(u0 - size_t(a0)) * stride_words];
      uint64_t* dst       = &reach[u0 * stride_words];

      // `shifted` must be a real temporary: when a0 is zero, src and dst are the same block.
      shift_left_into(src, shifted.data(), stride_words, size_t(a1), bits1);

      for (size_t w = 0; w < stride_words; ++w) {
        uint64_t fresh = shifted[w] & ~dst[w];
        dst[w] |= shifted[w];
        while (fresh != 0) {
          const size_t u1    = w * 64 + size_t(std::countr_zero(fresh));
          f[u0 * bits1 + u1] = uint8_t(j + 1);
          fresh &= fresh - 1;
        }
      }
    }
  }
}

}  // namespace

template <typename i_t, typename f_t>
void markshare_t<i_t, f_t>::fingerprint_set_t::init(size_t capacity)
{
  size_t bits = 1;
  while ((size_t(1) << bits) < capacity * 2) {
    ++bits;
  }
  mask = (size_t(1) << bits) - 1;
  slot.assign(mask + 1, 0);
}

template <typename i_t, typename f_t>
void markshare_t<i_t, f_t>::fingerprint_set_t::insert(uint64_t fingerprint)
{
  // Zero marks an empty slot, so it is the one value that cannot be stored verbatim.
  if (fingerprint == 0) { fingerprint = 1; }
  size_t p = fingerprint & mask;
  for (;;) {
    std::atomic_ref<uint64_t> cell(slot[p]);
    const uint64_t current = cell.load(std::memory_order_relaxed);
    if (current == fingerprint) { return; }
    if (current == 0) {
      uint64_t expected = 0;
      if (cell.compare_exchange_strong(expected, fingerprint, std::memory_order_relaxed)) {
        return;
      }
      if (expected == fingerprint) { return; }
    }
    p = (p + 1) & mask;
  }
}

template <typename i_t, typename f_t>
bool markshare_t<i_t, f_t>::fingerprint_set_t::contains(uint64_t fingerprint) const
{
  if (fingerprint == 0) { fingerprint = 1; }
  size_t p = fingerprint & mask;
  while (slot[p] != 0) {
    if (slot[p] == fingerprint) { return true; }
    p = (p + 1) & mask;
  }
  return false;
}

template <typename i_t, typename f_t>
uint64_t markshare_t<i_t, f_t>::residual_fingerprint(const coefficient_type* residual) const
{
  uint64_t h = 1469598103934665603ull;
  for (i_t k = 0; k < model_.m; ++k) {
    h ^= uint64_t(uint32_t(residual[k])) + 0x9e3779b9ull;
    h *= 1099511628211ull;
  }
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdull;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ull;
  return h ^ (h >> 33);
}

// ---------------------------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------------------------

template <typename i_t, typename f_t>
void markshare_t<i_t, f_t>::print_table_header() const
{
  const std::string header = std::format("{:^1}|{:^12}|{:^19}|{:^15}|{:^11}|{:^8}|",
                                         "",
                                         "Explored",
                                         "Objective",
                                         "Bound",
                                         "Gap",
                                         "Time");
  CUOPT_LOG_INFO("%s", header.c_str());
}

template <typename i_t, typename f_t>
void markshare_t<i_t, f_t>::report(char symbol, bool have_incumbent)
{
  const i_t done         = levels_exhausted_.load(std::memory_order_relaxed);
  const f_t solver_bound = model_.weight * done + obj_offset_fixed_;
  const f_t user_bound   = user_objective(solver_bound);

  std::string objective_text;
  f_t gap = std::numeric_limits<f_t>::infinity();
  if (have_incumbent) {
    const f_t user_obj = user_objective(incumbent_);
    objective_text     = std::format("{:+.6e}", user_obj);
    // The same gap solution_t::get_solution derives the Optimal status from, so the table agrees
    // with the status the solve ends up reporting.
    gap = compute_rel_mip_gap(user_obj, user_bound);
  }

  const std::string line = std::format("{:^1} {:>12} {:^19} {:^+15.6e} {:^11} {:>8.2f}",
                                       symbol,
                                       live_nodes_.load(std::memory_order_relaxed),
                                       objective_text,
                                       user_bound,
                                       to_percentage(gap),
                                       timer_.elapsed_time());
  CUOPT_LOG_INFO("%s", line.c_str());
}

template <typename i_t, typename f_t>
void markshare_t<i_t, f_t>::maybe_report(double now)
{
  if (now < next_report_.load(std::memory_order_relaxed)) { return; }
  // Claim the slot. Whoever swaps in the next deadline and gets back one that was still due is
  // the single winner; every other task that raced here carries on searching.
  const double previous =
    next_report_.exchange(now + settings_.report_interval, std::memory_order_relaxed);
  if (now < previous) { return; }
  report(' ', false);
}

// ---------------------------------------------------------------------------------------------
// Recognition
// ---------------------------------------------------------------------------------------------

template <typename i_t, typename f_t>
bool markshare_t<i_t, f_t>::recognize(
  const optimization_problem_t<i_t, f_t>& op_problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances)
{
  detected_ = false;
  // Restricted to double on purpose: with float, an exact row scaling of the form
  // integer_multiple / gcd lands around 1e-7 relative error, which straddles the normalization
  // tolerance. The upside on float is zero and the downside is a wrong optimality claim.
  if constexpr (!std::is_same_v<f_t, double>) { return false; }

  const i_t n_variables   = op_problem.get_n_variables();
  const i_t n_constraints = op_problem.get_n_constraints();

  // Cheap rejects first, so nothing is copied off the device for a model that cannot match.
  if (n_constraints < 2 || n_constraints > settings_.max_rows) { return false; }
  // The raw form carries up to two slack columns per row on top of the binaries.
  if (n_variables < n_constraints + 1 ||
      n_variables > settings_.max_core_cols + 3 * settings_.max_rows) {
    return false;
  }
  if (op_problem.get_nnz() > n_constraints * n_variables) { return false; }
  if (op_problem.get_n_integers() <= 0) { return false; }
  if ((i_t)op_problem.get_variable_upper_bounds().size() != n_variables) { return false; }
  if (!op_problem.get_variable_lower_bounds().is_empty() &&
      (i_t)op_problem.get_variable_lower_bounds().size() != n_variables) {
    return false;
  }
  // The conversion maps every non-continuous type onto INTEGER, so a semi-continuous column would
  // reach the search as a plain binary.
  if (op_problem.has_semi_continuous_variables()) { return false; }
  // A quadratic constraint makes the conversion expand the model into second order cones.
  if (op_problem.has_quadratic_objective() || op_problem.has_quadratic_constraints()) {
    return false;
  }

  auto problem = cuopt_problem_to_user_problem<i_t, f_t>(op_problem.get_handle_ptr(), op_problem);
  // The variable lower bounds are optional on the model and pass through the conversion as they
  // are.
  if (problem.lower.empty()) { problem.lower.assign(n_variables, f_t{0}); }

  obj_scale_  = op_problem.get_objective_scaling_factor();
  obj_offset_ = op_problem.get_objective_offset();
  // The conversion records a maximization in obj_scale and leaves the coefficients alone, while
  // the classification below reads their signs.
  if (op_problem.get_sense()) {
    for (auto& coefficient : problem.objective) {
      coefficient = -coefficient;
    }
    obj_scale_  = -obj_scale_;
    obj_offset_ = -obj_offset_;
  }

  settings_.integrality_tolerance = tolerances.integrality_tolerance;

  auto discard_model = cuopt::scope_guard([&]() {
    if (!detected_) { model_ = normalized_model_t{}; }
  });

  const i_t m        = problem.num_rows;
  const i_t num_cols = problem.num_cols;

  const f_t int_tol      = settings_.integrality_tolerance;
  const double exact_tol = settings_.normalization_tolerance;
  auto to_integer        = [&](double value, coefficient_type& out) -> bool {
    const double rounded = std::round(value);
    if (std::abs(value - rounded) > exact_tol * std::max(1.0, std::abs(value))) { return false; }
    if (std::abs(rounded) > settings_.max_normalized_rhs) { return false; }
    out = coefficient_type(rounded);
    return true;
  };

  // A range row is carried as sense 'E' at its lower bound plus an entry in range_rows.
  if (problem.num_range_rows != 0) {
    CUOPT_LOG_DEBUG("markshare: the model has %d range rows", problem.num_range_rows);
    return false;
  }

  // Every row must be an equality with a small non-negative integer right hand side.
  std::vector<coefficient_type> b(m, 0);
  for (i_t k = 0; k < m; ++k) {
    if (problem.row_sense[k] != 'E' || !std::isfinite(problem.rhs[k])) {
      CUOPT_LOG_DEBUG("markshare: row %d is not a finite equality", k);
      return false;
    }
    if (!to_integer(problem.rhs[k], b[k]) || b[k] < 0) {
      CUOPT_LOG_DEBUG("markshare: rhs of row %d is not a small non-negative integer", k);
      return false;
    }
  }

  // Classification is per column, and the conversion already left the model in the compressed
  // column layout the search reconstructs from.
  const auto& column = problem.A;

  std::vector<i_t> core_col;
  std::vector<i_t> fixed_col;
  std::vector<f_t> fixed_val;
  std::vector<i_t> slack_col(m, -1);
  std::vector<coefficient_type> gathered;  // core columns, column major, m entries each
  std::vector<coefficient_type> col_sum;
  gathered.reserve(size_t(num_cols) * m);
  f_t weight     = 0;
  f_t fixed_cost = 0;
  std::vector<coefficient_type> entry(m, 0);

  for (i_t j = 0; j < num_cols; ++j) {
    // A column pinned at a single value carries no decision. This is what absorbs the second half
    // of the slack pairs that markshare1 and markshare2 spell out with an MPS FX bound.
    if (problem.lower[j] == problem.upper[j]) {
      const f_t value = problem.lower[j];
      if (!std::isfinite(value)) { return false; }
      if (value != f_t{0}) {
        for (i_t e = column.col_start[j]; e < column.col_start[j + 1]; ++e) {
          coefficient_type scaled;
          if (!to_integer(column.x[e] * value, scaled)) { return false; }
          b[column.i[e]] -= scaled;
        }
        fixed_cost += problem.objective[j] * value;
      }
      fixed_col.push_back(j);
      fixed_val.push_back(value);
      continue;
    }

    if (problem.var_types[j] == simplex::variable_type_t::CONTINUOUS) {
      // The only continuous column the form allows is a row's slack. An explicitly stored zero is
      // not a constraint role, so count the entries that actually carry one.
      i_t k          = -1;
      f_t slack_coef = 0;
      i_t nonzeros   = 0;
      for (i_t e = column.col_start[j]; e < column.col_start[j + 1]; ++e) {
        if (column.x[e] == f_t{0}) { continue; }
        k          = column.i[e];
        slack_coef = column.x[e];
        ++nonzeros;
      }
      if (nonzeros != 1) {
        CUOPT_LOG_DEBUG("markshare: continuous column %d is not a row singleton", j);
        return false;
      }
      if (slack_coef != f_t{1} || problem.lower[j] != f_t{0}) {
        CUOPT_LOG_DEBUG("markshare: column %d is not a unit slack at zero", j);
        return false;
      }
      if (slack_col[k] != -1) {
        CUOPT_LOG_DEBUG("markshare: row %d has more than one slack", k);
        return false;
      }
      const f_t cost = problem.objective[j];
      if (!(cost > 0)) {
        CUOPT_LOG_DEBUG("markshare: slack of row %d does not carry a positive cost", k);
        return false;
      }
      if (weight == f_t{0}) {
        weight = cost;
      } else if (std::abs(cost - weight) > exact_tol * std::max(1.0, double(weight))) {
        // Non-uniform weights break the correspondence between an objective level and the total
        // slack, which is what makes the ascending enumeration a proof. Rejecting is the safe
        // response.
        CUOPT_LOG_DEBUG("markshare: slack cost is not uniform across rows");
        return false;
      }
      slack_col[k] = j;
      continue;
    }

    // Everything else must be a binary with no direct objective cost.
    if (problem.lower[j] != f_t{0} || problem.upper[j] != f_t{1}) {
      CUOPT_LOG_DEBUG("markshare: integer column %d is not binary", j);
      return false;
    }
    if (std::abs(problem.objective[j]) > int_tol) {
      CUOPT_LOG_DEBUG("markshare: binary column %d carries objective cost", j);
      return false;
    }

    std::fill(entry.begin(), entry.end(), 0);
    coefficient_type sum = 0;
    for (i_t e = column.col_start[j]; e < column.col_start[j + 1]; ++e) {
      coefficient_type value;
      if (!to_integer(column.x[e], value)) {
        CUOPT_LOG_DEBUG("markshare: an entry of column %d is not integral", j);
        return false;
      }
      if (value < 0) {
        CUOPT_LOG_DEBUG("markshare: column %d has a negative coefficient", j);
        return false;
      }
      entry[column.i[e]] = value;
      sum += value;
    }
    if (sum == 0) {
      // No constraint role, so it can simply be written at its lower bound.
      fixed_col.push_back(j);
      fixed_val.push_back(problem.lower[j]);
      continue;
    }

    core_col.push_back(j);
    col_sum.push_back(sum);
    gathered.insert(gathered.end(), entry.begin(), entry.end());
  }

  for (i_t k = 0; k < m; ++k) {
    if (slack_col[k] == -1) {
      CUOPT_LOG_DEBUG("markshare: row %d has no slack", k);
      return false;
    }
    if (b[k] < 0) {
      CUOPT_LOG_DEBUG("markshare: rhs of row %d went negative after pinning", k);
      return false;
    }
    // The slack has to be able to absorb the whole right hand side, otherwise a level is not
    // reachable and the enumeration would prove the wrong thing.
    if (problem.upper[slack_col[k]] < b[k]) {
      CUOPT_LOG_DEBUG("markshare: slack of row %d cannot reach its rhs", k);
      return false;
    }
  }

  const i_t core_count = core_col.size();
  if (core_count < 1 || core_count >= MARKSHARE_UNREACHABLE) { return false; }
  if (core_count > settings_.max_search_cols) {
    CUOPT_LOG_DEBUG("markshare: %d columns is beyond the tractable range", core_count);
    return false;
  }

  // The enumeration runs backwards, so it decides the largest coefficients first while the tables'
  // prefixes hold the smallest. That is what makes both the remaining-capacity prune and the
  // reachability prune bite at shallow depth.
  std::vector<i_t> order(core_count);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(
    order.begin(), order.end(), [&](i_t x, i_t y) { return col_sum[x] < col_sum[y]; });

  const i_t n        = core_count;
  model_.m           = m;
  model_.n           = n;
  model_.b           = std::move(b);
  model_.slack_col   = std::move(slack_col);
  model_.fixed_col   = std::move(fixed_col);
  model_.fixed_val   = std::move(fixed_val);
  model_.weight      = weight;
  model_.n_variables = num_cols;
  model_.core_col.resize(n);
  model_.Arow.assign(size_t(m) * n, 0);
  model_.Acol.assign(size_t(n) * m, 0);
  for (i_t p = 0; p < n; ++p) {
    const i_t source   = order[p];
    model_.core_col[p] = core_col[source];
    for (i_t k = 0; k < m; ++k) {
      const coefficient_type value   = gathered[size_t(source) * m + k];
      model_.Arow[size_t(k) * n + p] = value;
      model_.Acol[size_t(p) * m + k] = value;
    }
  }

  model_.prefix_max.assign(size_t(m) * (n + 1), 0);
  model_.row_gcd.assign(m, 0);
  for (i_t k = 0; k < m; ++k) {
    coefficient_type running = 0;
    coefficient_type divisor = 0;
    for (i_t p = 0; p < n; ++p) {
      const coefficient_type value = model_.Arow[size_t(k) * n + p];
      running += value;
      model_.prefix_max[size_t(k) * (n + 1) + p + 1] = running;
      divisor                                        = std::gcd(divisor, value);
    }
    model_.row_gcd[k] = divisor;
  }

  int64_t best_cells = -1;
  for (i_t k0 = 0; k0 < m; ++k0) {
    for (i_t k1 = k0 + 1; k1 < m; ++k1) {
      const int64_t cells = (int64_t(model_.b[k0]) + 1) * (int64_t(model_.b[k1]) + 1);
      if (best_cells < 0 || cells < best_cells) {
        best_cells  = cells;
        joint_row0_ = k0;
        joint_row1_ = k1;
      }
    }
  }
  const size_t joint_bytes = size_t(best_cells) * sizeof(uint8_t);
  if (joint_bytes > settings_.max_table_bytes) {
    CUOPT_LOG_DEBUG("markshare: joint table would need %zu bytes", joint_bytes);
    return false;
  }

  obj_offset_fixed_ = fixed_cost;

  num_threads_ = omp_get_num_threads();
  CUOPT_LOG_INFO(
    "Markshare structure detected. Solving via dynamic programming with %d threads...\n",
    num_threads_);

  CUOPT_LOG_DEBUG("%s",
                  std::format("{} rows, {} binaries, rhs max {}, "
                              "slack cost {:g}, joint rows ({}, {}), joint table {:.1f} MB",
                              model_.m,
                              model_.n,
                              *std::max_element(model_.b.begin(), model_.b.end()),
                              double(model_.weight),
                              joint_row0_,
                              joint_row1_,
                              joint_bytes / (1024.0 * 1024.0))
                    .c_str());

  detected_ = true;
  problem_  = std::make_unique<simplex::user_problem_t<i_t, f_t>>(std::move(problem));
  return true;
}

// ---------------------------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------------------------

template <typename i_t, typename f_t>
void markshare_t<i_t, f_t>::build_tables()
{
  const i_t m = model_.m;
  const i_t n = model_.n;

  row_tables_.resize(m);
  extra_rows_.clear();
  std::vector<coefficient_type> coefficients(n);
  for (i_t k = 0; k < m; ++k) {
    for (i_t p = 0; p < n; ++p) {
      coefficients[p] = model_.Arow[size_t(k) * n + p];
    }
    build_row_table(coefficients, model_.b[k], row_tables_[k]);
    if (k != joint_row0_ && k != joint_row1_) { extra_rows_.push_back(k); }
  }

  std::vector<coefficient_type> c0(n), c1(n);
  for (i_t p = 0; p < n; ++p) {
    c0[p] = model_.Arow[size_t(joint_row0_) * n + p];
    c1[p] = model_.Arow[size_t(joint_row1_) * n + p];
  }
  build_joint_table(c0, c1, model_.b[joint_row0_], model_.b[joint_row1_], joint_);
  joint_stride_ = size_t(model_.b[joint_row1_]) + 1;

  context_.resize(n, m);
  value_.assign(n, 0);
  found_slack_.assign(m, 0);
  target_.assign(m, 0);
}

template <typename i_t, typename f_t>
i_t markshare_t<i_t, f_t>::choose_hash_depth() const
{
  const i_t n = model_.n;
  if (n <= settings_.hash_min_cols) { return 0; }
  i_t depth = std::min(n / 2 + settings_.hash_depth_offset, settings_.hash_max_depth);

  while (depth > 0) {
    // Open addressing at 50% load: 2^(depth + 1) slots of eight bytes.
    const size_t bytes = (size_t(1) << (depth + 1)) * sizeof(uint64_t);
    if (bytes <= settings_.hash_bytes) { break; }
    --depth;
  }
  return depth < 8 ? 0 : depth;
}

template <typename i_t, typename f_t>
void markshare_t<i_t, f_t>::build_hash()
{
  hash_depth_ = choose_hash_depth();
  if (hash_depth_ <= 0) { return; }

  const i_t m = model_.m;
  const i_t h = hash_depth_;
  hash_.init(size_t(1) << h);

  // Split the enumerated columns into a prefix that fans out across tasks and a suffix each task
  // walks in Gray code order, so every subset costs one add per row rather than h of them.
  i_t top = 0;
  while ((i_t(1) << top) < 4 * num_threads_ && top < h - 8) {
    ++top;
  }
  const i_t low       = h - top;
  const size_t blocks = size_t(1) << top;

#pragma omp taskloop grainsize(1) shared(m, h, top, low)
  for (size_t block = 0; block < blocks; ++block) {
    std::vector<coefficient_type> sum(m, 0);
    for (i_t t = 0; t < top; ++t) {
      if ((block >> t & 1) != 0) {
        const coefficient_type* col = &model_.Acol[size_t(low + t) * m];
        for (i_t k = 0; k < m; ++k) {
          sum[k] += col[k];
        }
      }
    }
    auto record = [&]() {
      for (i_t k = 0; k < m; ++k) {
        if (sum[k] > model_.b[k]) { return; }
      }
      hash_.insert(residual_fingerprint(sum.data()));
    };

    record();
    uint64_t previous = 0;
    for (uint64_t g = 1; g < (uint64_t(1) << low); ++g) {
      const uint64_t code         = g ^ (g >> 1);
      const uint64_t diff         = code ^ previous;
      const i_t j                 = std::countr_zero(diff);
      const coefficient_type* col = &model_.Acol[size_t(j) * m];
      if ((code & diff) != 0) {
        for (i_t k = 0; k < m; ++k) {
          sum[k] += col[k];
        }
      } else {
        for (i_t k = 0; k < m; ++k) {
          sum[k] -= col[k];
        }
      }
      previous = code;
      record();
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------------------------

template <typename i_t, typename f_t>
typename markshare_t<i_t, f_t>::dfs_result_t markshare_t<i_t, f_t>::run_dfs_from(
  dfs_context_t& ctx,
  i_t start_depth,
  const coefficient_type* start_residual,
  const std::atomic<bool>* stop,
  i_t terminal_depth)
{
  const i_t m = model_.m;
  const i_t n = model_.n;

  std::copy(start_residual, start_residual + m, ctx.residual.begin() + size_t(start_depth) * m);
  ctx.branch[start_depth] = 0;
  i_t j                   = start_depth;
  int64_t next_check      = ctx.nodes + settings_.node_report_interval;

  // Every exit path publishes the trailing nodes. Without this a search that finishes inside one
  // check interval -- which is every small instance -- reports zero nodes explored.
  auto flush_nodes = cuopt::scope_guard([&]() {
    live_nodes_.fetch_add(ctx.nodes - ctx.accounted, std::memory_order_relaxed);
    ctx.accounted = ctx.nodes;
  });

  for (;;) {
    if (j == terminal_depth) {
      // At depth zero the remaining-capacity prune has already forced every residual to exactly
      // zero, so reaching this point is a solution with no further check needed.
      if (terminal_depth == 0) { return dfs_result_t::FOUND; }
      // Otherwise ask the meet-in-the-middle table whether the columns below can supply the
      // residual exactly. A fingerprint collision surfaces as a sub-search that finds nothing,
      // which simply resumes the enumeration -- it can never hide a solution.
      const coefficient_type* residual = &ctx.residual[size_t(j) * m];
      if (hash_.contains(residual_fingerprint(residual))) {
        const std::vector<coefficient_type> below(residual, residual + m);
        if (run_dfs_from(ctx, terminal_depth, below.data(), stop, 0) == dfs_result_t::FOUND) {
          return dfs_result_t::FOUND;
        }
      }
      ++j;
      if (j > start_depth) { return dfs_result_t::EXHAUSTED; }
      continue;
    }
    if (ctx.branch[j] == 2) {
      ++j;
      if (j > start_depth) { return dfs_result_t::EXHAUSTED; }
      continue;
    }

    // Try one first: it shrinks the residual faster, so the capacity prune bites sooner.
    const uint8_t v = 1 - ctx.branch[j];
    ++ctx.branch[j];
    ++ctx.nodes;
    if (ctx.nodes >= next_check) {
      next_check = ctx.nodes + settings_.node_report_interval;
      live_nodes_.fetch_add(ctx.nodes - ctx.accounted, std::memory_order_relaxed);
      ctx.accounted    = ctx.nodes;
      const double now = timer_.elapsed_time();
      if (stop != nullptr && stop->load(std::memory_order_relaxed)) { return dfs_result_t::BUDGET; }
      if (preemption_ != nullptr && preemption_->load(std::memory_order_relaxed)) {
        return dfs_result_t::BUDGET;
      }
      if (timer_.check_time_limit()) { return dfs_result_t::BUDGET; }
      maybe_report(now);
    }

    const i_t p                      = j - 1;
    const coefficient_type* column   = &model_.Acol[size_t(p) * m];
    const coefficient_type* previous = &ctx.residual[size_t(j) * m];
    coefficient_type* current        = &ctx.residual[size_t(p) * m];

    // Prune 1 and 2 fused into one pass: negative residual, or a residual larger than the
    // remaining columns can possibly supply.
    bool pruned = false;
    for (i_t k = 0; k < m; ++k) {
      const coefficient_type left = previous[k] - (v != 0 ? column[k] : 0);
      if (left < 0 || left > model_.prefix_max[size_t(k) * (n + 1) + p]) {
        pruned = true;
        break;
      }
      current[k] = left;
    }
    // Prune 3: the rows outside the joint pair. Their tables are a couple of KB and stay in L1.
    if (!pruned) {
      for (i_t k : extra_rows_) {
        if (row_tables_[k][size_t(current[k])] > p) {
          pruned = true;
          break;
        }
      }
    }
    // Prune 4: the joint table. Strongest, but also the one lookup that misses cache, so it goes
    // last.
    if (!pruned && joint_at(current[joint_row0_], current[joint_row1_]) > p) { pruned = true; }

    if (!pruned) {
      ctx.value[p]  = v;
      ctx.branch[p] = 0;
      j             = p;
    }
  }
}

template <typename i_t, typename f_t>
void markshare_t<i_t, f_t>::collect_subtrees(const std::vector<coefficient_type>& target,
                                             i_t depth,
                                             std::vector<subtree_t>& seeds)
{
  const i_t m          = model_.m;
  const i_t n          = model_.n;
  const i_t stop_depth = n - depth;

  std::vector<coefficient_type> residual(size_t(n + 1) * m, 0);
  std::vector<uint8_t> branch(n + 1, 0);
  std::vector<uint8_t> value(n, 0);
  std::copy(target.begin(), target.end(), residual.begin() + size_t(n) * m);

  i_t j = n;
  for (;;) {
    if (j == stop_depth) {
      subtree_t seed;
      seed.value = value;
      seed.residual.assign(residual.begin() + size_t(j) * m, residual.begin() + size_t(j) * m + m);
      seeds.push_back(std::move(seed));
      ++j;
      if (j > n) { return; }
      continue;
    }
    if (branch[j] == 2) {
      ++j;
      if (j > n) { return; }
      continue;
    }

    const uint8_t v = 1 - branch[j];
    ++branch[j];

    const i_t p                      = j - 1;
    const coefficient_type* column   = &model_.Acol[size_t(p) * m];
    const coefficient_type* previous = &residual[size_t(j) * m];
    coefficient_type* current        = &residual[size_t(p) * m];

    bool pruned = false;
    for (i_t k = 0; k < m; ++k) {
      const coefficient_type left = previous[k] - (v != 0 ? column[k] : 0);
      if (left < 0 || left > model_.prefix_max[size_t(k) * (n + 1) + p]) {
        pruned = true;
        break;
      }
      current[k] = left;
    }
    if (!pruned) {
      for (i_t k : extra_rows_) {
        if (row_tables_[k][size_t(current[k])] > p) {
          pruned = true;
          break;
        }
      }
    }
    if (!pruned && joint_at(current[joint_row0_], current[joint_row1_]) > p) { pruned = true; }

    if (!pruned) {
      value[p]  = v;
      branch[p] = 0;
      j         = p;
    }
  }
}

template <typename i_t, typename f_t>
typename markshare_t<i_t, f_t>::dfs_result_t markshare_t<i_t, f_t>::run_dfs(
  const std::vector<coefficient_type>& target)
{
  const i_t n = model_.n;

  // Serial for small models: seeding and task overhead would dominate a search that finishes in
  // microseconds anyway.
  if (num_threads_ < 2 || n < 16 || n - 1 <= hash_depth_) {
    context_.branch.assign(n + 1, 0);
    const dfs_result_t rc = run_dfs_from(context_, n, target.data(), nullptr, hash_depth_);
    if (rc == dfs_result_t::FOUND) { value_ = context_.value; }
    return rc;
  }

  // Split the trailing columns into independent subtrees. Subtree sizes are wildly uneven, so aim
  // for several tasks per thread and let the scheduler balance them.
  std::vector<subtree_t> subtrees;
  i_t depth = 0;
  while (i_t(subtrees.size()) < 8 * num_threads_ && depth < n - 2) {
    ++depth;
    subtrees.clear();
    collect_subtrees(target, depth, subtrees);
    if (subtrees.empty()) { return dfs_result_t::EXHAUSTED; }
  }
  if (subtrees.empty()) { return dfs_result_t::EXHAUSTED; }

  const i_t start_depth = n - depth;
  std::atomic<bool> stop{false};
  std::atomic<bool> found{false};
  std::atomic<bool> budget{false};

#pragma omp taskloop grainsize(1) shared(subtrees, stop, found, budget)
  for (size_t s = 0; s < subtrees.size(); ++s) {
    if (!stop.load(std::memory_order_relaxed)) {
      dfs_context_t ctx;
      ctx.resize(model_.n, model_.m);
      std::copy(subtrees[s].value.begin(), subtrees[s].value.end(), ctx.value.begin());
      const dfs_result_t rc =
        run_dfs_from(ctx, start_depth, subtrees[s].residual.data(), &stop, hash_depth_);
      if (rc == dfs_result_t::FOUND) {
        bool expected = false;
        // First finder wins; the rest are told to stop.
        if (found.compare_exchange_strong(expected, true)) { value_ = ctx.value; }
        stop.store(true, std::memory_order_relaxed);
      } else if (rc == dfs_result_t::BUDGET) {
        budget.store(true, std::memory_order_relaxed);
        stop.store(true, std::memory_order_relaxed);
      }
    }
  }

  if (found.load(std::memory_order_relaxed)) { return dfs_result_t::FOUND; }
  if (budget.load(std::memory_order_relaxed)) { return dfs_result_t::BUDGET; }
  return dfs_result_t::EXHAUSTED;
}

template <typename i_t, typename f_t>
bool markshare_t<i_t, f_t>::enumerate_level(i_t level,
                                            std::vector<coefficient_type>& slack,
                                            i_t index,
                                            bool& found)
{
  const i_t m = model_.m;
  if (index == m - 1) {
    slack[index] = level;

    for (i_t k = 0; k < m; ++k) {
      if (slack[k] > model_.b[k]) { return false; }
      target_[k] = model_.b[k] - slack[k];
    }
    // Three O(m) rejects that kill most target vectors without entering the DFS at all.
    for (i_t k = 0; k < m; ++k) {
      if (model_.row_gcd[k] > 0 && (target_[k] % model_.row_gcd[k]) != 0) { return false; }
      if (row_tables_[k][size_t(target_[k])] > model_.n) { return false; }
    }
    if (joint_at(target_[joint_row0_], target_[joint_row1_]) > model_.n) { return false; }

    const dfs_result_t result = run_dfs(target_);
    if (result == dfs_result_t::FOUND) {
      found        = true;
      found_slack_ = slack;
      return true;
    }
    if (result == dfs_result_t::BUDGET) {
      budget_exhausted_ = true;
      return true;
    }
    return false;
  }

  for (i_t v = 0; v <= level; ++v) {
    slack[index] = v;
    if (enumerate_level(level - v, slack, index + 1, found)) { return true; }
  }
  return false;
}

template <typename i_t, typename f_t>
bool markshare_t<i_t, f_t>::reconstruct(std::vector<f_t>& assignment) const
{
  const i_t m        = model_.m;
  const i_t n        = model_.n;
  const i_t num_cols = model_.n_variables;

  assignment.assign(num_cols, f_t{0});
  for (i_t p = 0; p < n; ++p) {
    assignment[model_.core_col[p]] = value_[p];
  }
  for (size_t idx = 0; idx < model_.fixed_col.size(); ++idx) {
    assignment[model_.fixed_col[idx]] = model_.fixed_val[idx];
  }
  for (i_t k = 0; k < m; ++k) {
    assignment[model_.slack_col[k]] = found_slack_[k];
  }

  // Independent verification against the untouched problem. This is the last line of defence
  // against every assumption recognition made, and it is deliberately written in terms of the
  // original data rather than the normalized model.
  cuopt_assert(problem_ != nullptr, "reconstruct called without a successful recognize");
  const auto& column = problem_->A;
  std::vector<double> activity(m, 0.0);
  for (i_t j = 0; j < num_cols; ++j) {
    const double x = assignment[j];
    if (!std::isfinite(x)) {
      CUOPT_LOG_ERROR("markshare: reconstructed column %d is not finite", j);
      return false;
    }
    if (x < problem_->lower[j] - settings_.integrality_tolerance ||
        x > problem_->upper[j] + settings_.integrality_tolerance) {
      CUOPT_LOG_ERROR("markshare: reconstructed column %d violates its bounds", j);
      return false;
    }
    if (problem_->var_types[j] != simplex::variable_type_t::CONTINUOUS &&
        !is_integer<double>(x, settings_.integrality_tolerance)) {
      CUOPT_LOG_ERROR("markshare: reconstructed column %d is fractional", j);
      return false;
    }
    for (i_t e = column.col_start[j]; e < column.col_start[j + 1]; ++e) {
      activity[column.i[e]] += column.x[e] * x;
    }
  }
  for (i_t k = 0; k < m; ++k) {
    const double rhs       = problem_->rhs[k];
    const double tolerance = 1e-6 * std::max(1.0, std::abs(rhs));
    if (std::abs(activity[k] - rhs) > tolerance) {
      CUOPT_LOG_ERROR("markshare: reconstructed row %d is violated", k);
      return false;
    }
  }
  return true;
}

template <typename i_t, typename f_t>
bool markshare_t<i_t, f_t>::solve(
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  f_t time_limit,
  std::atomic<bool>& preemption,
  std::vector<f_t>& assignment)
{
  if (!detected_) { return false; }

  settings_.integrality_tolerance = tolerances.integrality_tolerance;
  preemption_                     = &preemption;
  timer_                          = timer_t(time_limit);
  budget_exhausted_               = false;
  live_nodes_.store(0, std::memory_order_relaxed);
  levels_exhausted_.store(0, std::memory_order_relaxed);
  next_report_.store(settings_.report_interval, std::memory_order_relaxed);

  build_tables();
  build_hash();
  if (hash_depth_ > 0) {
    CUOPT_LOG_DEBUG("%s",
                    std::format("Markshare meet-in-the-middle terminal at depth {} ({:.1f} MB)",
                                hash_depth_,
                                hash_.bytes() / (1024.0 * 1024.0))
                      .c_str());
  }
  print_table_header();

  std::vector<coefficient_type> slack(model_.m, 0);
  i_t solution_level = -1;

  for (i_t level = 0; level <= settings_.max_level; ++level) {
    bool found = false;
    enumerate_level(level, slack, 0, found);
    if (found) {
      solution_level = level;
      break;
    }
    if (budget_exhausted_ || preemption.load()) { break; }
    // Exhausting levels 0..level proves that no feasible point has a total slack below level + 1.
    // The slacks are implied integer, so the levels are spaced exactly one apart.
    levels_exhausted_.store(level + 1, std::memory_order_relaxed);
    report(' ', false);
  }

  if (solution_level < 0) {
    CUOPT_LOG_INFO("%s",
                   std::format("Heuristic stopped after exploring {} nodes in {:.2f}s",
                               live_nodes_.load(std::memory_order_relaxed),
                               timer_.elapsed_time())
                     .c_str());
    return false;
  }

  if (!reconstruct(assignment)) {
    assignment.clear();
    return false;
  }

  // A solution at level T is optimal because the loop only reaches T after exhausting every level
  // below it; budget exhaustion breaks out before advancing.
  incumbent_ = model_.weight * solution_level + obj_offset_fixed_;
  levels_exhausted_.store(solution_level, std::memory_order_relaxed);
  report('*', true);

  CUOPT_LOG_INFO("%s",
                 std::format("\nExplored {} nodes in {:.2f}s.",
                             live_nodes_.load(std::memory_order_relaxed),
                             timer_.elapsed_time())
                   .c_str());
  // Every level below the one that produced this point was exhausted, so it is optimal outright
  // rather than within a gap tolerance.
  CUOPT_LOG_INFO("Optimal solution found.");
  return true;
}

#if MIP_INSTANTIATE_FLOAT
template class markshare_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class markshare_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip

#undef MARKSHARE_UNREACHABLE
