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

// Building the point partition visits at most 2^BHW_MAX_LEN patterns per row.
static constexpr int BHW_MAX_LEN = 12;
// Largest max|w| the exhaustive search considers before falling back to the heuristic candidates.
static constexpr int64_t BHW_EXACT_MAX_WEIGHT = 6;
// Passed to row_int_scale as its maxdnom and maxfinal caps.
static constexpr int64_t BHW_INT_SCALE_MAX = 1000000;

struct bhw_shape_result_t {
  std::vector<int64_t> weights;
  int64_t bound = 0;
  bool accepted = false;
};

using bhw_shape_cache_t = std::map<std::vector<int64_t>, bhw_shape_result_t>;

struct bhw_row_rewrite_t {
  std::vector<int64_t> coefficients;  // 0 drops the entry
  int64_t side            = 0;
  int64_t max_coef_before = 0;
  int64_t max_coef_after  = 0;
  bool accepted           = false;
};

// Rewrites a one-sided all-binary row with smaller integer coefficients and the same 0/1 feasible
// set. direction is +1 for <= and -1 for >=. The caller guarantees nonfixed
// binary variables and exactly one finite side.
template <typename f_t>
bhw_row_rewrite_t bhw_reduce_row(
  const f_t* coefficients, int len, f_t side, int direction, bhw_shape_cache_t* cache);

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
  bhw_shape_cache_t shape_cache_;
};

}  // namespace cuopt::mathematical_optimization::mip
