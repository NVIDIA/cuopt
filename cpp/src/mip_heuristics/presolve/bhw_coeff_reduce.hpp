/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#if !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"  // ignore boost error for pip wheel build
#pragma GCC diagnostic ignored "-Wnarrowing"
#endif
#include <papilo/Config.hpp>
#include <papilo/core/PresolveMethod.hpp>
#include <papilo/core/Problem.hpp>
#include <papilo/core/ProblemUpdate.hpp>
#if !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <cstdint>
#include <map>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Widest row we enumerate: building the point partition walks 2^BHW_MAX_LEN patterns. A census of
// the PaPILO-reduced MIPLIB corpus (6.2M all-binary one-sided rows) puts 98.0% of the exactly
// integralizable ones at nnz <= 12; raising this to 40 would add 2.0%.
static constexpr int BHW_MAX_LEN = 12;
// Largest max|w| the exhaustive search considers before falling back to the heuristic candidates.
// Accepted weights were at most 3 throughout the corpus study.
static constexpr int64_t BHW_EXACT_MAX_WEIGHT = 6;
// Largest per-row rational multiplier / denominator used to integerize a row (passed to
// row_int_scale as its maxdnom/maxfinal caps).
static constexpr int64_t BHW_INT_SCALE_MAX = 1000000;  // 1e6

// Outcome for one canonical row shape, in BHW's normalized frame (coefficients complemented to be
// positive and sorted descending). Rejections are cached too, since re-deriving them is the bulk of
// the work on instances whose rows repeat.
struct bhw_shape_result_t {
  std::vector<int64_t> weights;
  int64_t bound = 0;
  bool accepted = false;
};

// Keyed by the normalized coefficients followed by the normalized right-hand side. The reduction is
// a pure function of that key, so entries stay valid across presolve rounds and problems.
using bhw_shape_cache_t = std::map<std::vector<int64_t>, bhw_shape_result_t>;

struct bhw_row_rewrite_t {
  std::vector<int64_t> coefficients;  // one per input entry, in input order; 0 drops that entry
  int64_t side = 0;                   // replaces the row's finite side
  // Largest |coefficient| before and after reduction, both in the integerized frame. Comparable
  // only there: the row is scaled on the way in, so the input coefficients sit in a different
  // frame.
  int64_t max_coef_before = 0;
  int64_t max_coef_after  = 0;
  bool accepted           = false;
};

// Rewrite one one-sided all-binary row with smaller integer coefficients spanning the same 0/1
// feasible set. direction is +1 for "coefficients . x <= side" and -1 for ">= side"; side is the
// finite side of the row. Rejects the row (accepted = false) unless it integerizes exactly, admits
// a strictly smaller equivalent form, and that form does not enlarge the row's LP relaxation. cache
// may be null to skip memoization.
//
// The caller checks that every entry is a binary integer variable and that exactly one side of the
// row is finite. Exposed for testing: BHWCoeffReduce::execute only screens rows and emits the
// result, so this covers the whole reduction without any papilo types.
template <typename f_t>
bhw_row_rewrite_t bhw_reduce_row(
  const f_t* coefficients, int len, f_t side, int direction, bhw_shape_cache_t* cache);

// Bradley-Hammer-Wolsey coefficient reduction: replace an all-binary row by an equivalent one with
// smaller integer coefficients. See bhw_coeff_reduce.cpp for the lineage.
template <typename f_t>
class BHWCoeffReduce : public papilo::PresolveMethod<f_t> {
 public:
  BHWCoeffReduce() : papilo::PresolveMethod<f_t>()
  {
    this->setName("bhwcoeffreduce");
    this->setType(papilo::PresolverType::kIntegralCols);
    this->setTiming(papilo::PresolverTiming::kMedium);
  }

  papilo::PresolveStatus execute(const papilo::Problem<f_t>& problem,
                                 const papilo::ProblemUpdate<f_t>& problemUpdate,
                                 const papilo::Num<f_t>& num,
                                 papilo::Reductions<f_t>& reductions,
                                 const papilo::Timer& timer,
                                 int& reason_of_infeasibility) override;

 private:
  // Only touched from execute, which papilo runs one task at a time per presolver object.
  bhw_shape_cache_t shape_cache_;
};

}  // namespace cuopt::mathematical_optimization::mip
