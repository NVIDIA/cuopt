/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/types.hpp>

#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
class lp_solution_t {
 public:
  lp_solution_t(i_t m, i_t n)
    : x(n),
      y(m),
      z(n),
      objective(std::numeric_limits<f_t>::quiet_NaN()),
      user_objective(std::numeric_limits<f_t>::quiet_NaN()),
      iterations(0),
      l2_primal_residual(std::numeric_limits<f_t>::quiet_NaN()),
      l2_dual_residual(std::numeric_limits<f_t>::quiet_NaN())
  {
  }

  void resize(i_t m, i_t n)
  {
    x.resize(n);
    y.resize(m);
    z.resize(n);
  }

  // Primal solution vector
  std::vector<f_t> x;
  // Dual solution vector. Lagrange multipliers for equality constraints.
  std::vector<f_t> y;
  // Reduced costs
  std::vector<f_t> z;
  f_t objective;
  f_t user_objective;
  i_t iterations;
  f_t l2_primal_residual;
  f_t l2_dual_residual;
};

template <typename i_t, typename f_t>
class mip_solution_t {
 public:
  mip_solution_t(i_t n)
    : x(n),
      objective(std::numeric_limits<f_t>::quiet_NaN()),
      lower_bound(-inf),
      has_incumbent(false)
  {
  }

  void resize(i_t n) { x.resize(n); }

  void set_incumbent_solution(f_t primal_objective, const std::vector<f_t>& primal_solution)
  {
    x             = primal_solution;
    objective     = primal_objective;
    has_incumbent = true;
  }

  // Primal solution vector
  std::vector<f_t> x;
  f_t objective;
  f_t lower_bound;
  i_t nodes_explored;
  i_t simplex_iterations;
  bool has_incumbent;
};

}  // namespace cuopt::linear_programming::dual_simplex
