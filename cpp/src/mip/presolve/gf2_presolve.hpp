/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"  // ignore boost error for pip wheel build
#include <papilo/Config.hpp>
#include <papilo/core/PresolveMethod.hpp>
#include <papilo/core/Problem.hpp>
#include <papilo/core/ProblemUpdate.hpp>
#pragma GCC diagnostic pop

namespace cuopt::linear_programming::detail {

template <typename f_t>
class GF2Presolve : public papilo::PresolveMethod<f_t> {
 public:
  GF2Presolve() : papilo::PresolveMethod<f_t>()
  {
    this->setName("gf2presolve");
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
  struct gf2_constraint_t {
    size_t cstr_idx;
    std::vector<std::pair<size_t, f_t>> bin_vars;
    std::pair<size_t, f_t> key_var;
    size_t rhs;  // 0 or 1

    gf2_constraint_t() = default;
    gf2_constraint_t(size_t cstr_idx,
                     std::vector<std::pair<size_t, f_t>> bin_vars,
                     std::pair<size_t, f_t> key_var,
                     size_t rhs)
      : cstr_idx(cstr_idx), bin_vars(std::move(bin_vars)), key_var(key_var), rhs(rhs)
    {
    }
    gf2_constraint_t(const gf2_constraint_t& other)                = default;
    gf2_constraint_t(gf2_constraint_t&& other) noexcept            = default;
    gf2_constraint_t& operator=(const gf2_constraint_t& other)     = default;
    gf2_constraint_t& operator=(gf2_constraint_t&& other) noexcept = default;
  };

  inline bool is_integer(f_t value, f_t tolerance) const
  {
    return std::abs(value - std::round(value)) <= tolerance;
  }
};

}  // namespace cuopt::linear_programming::detail
