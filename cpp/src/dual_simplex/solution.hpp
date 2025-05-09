/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <dual_simplex/types.hpp>

#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
class lp_solution_t {
 public:
  lp_solution_t(i_t m, i_t n) : x(n), y(m), z(n)
  {
    objective = std::numeric_limits<f_t>::quiet_NaN();
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
  // Dual solution vector. Lagrange multipliers for inequality constraints.
  std::vector<f_t> z;
  f_t objective;
  f_t user_objective;
  i_t iterations;
};

template <typename i_t, typename f_t>
class mip_solution_t {
 public:
  mip_solution_t(i_t n) : x(n), objective(std::numeric_limits<f_t>::quiet_NaN()), lower_bound(-inf)
  {
  }

  void resize(i_t n) { x.resize(n); }

  void set_incumbent_solution(f_t primal_objective, const std::vector<f_t>& primal_solution)
  {
    x         = primal_solution;
    objective = primal_objective;
  }

  // Primal solution vector
  std::vector<f_t> x;
  f_t objective;
  f_t lower_bound;
  i_t nodes_explored;
  i_t simplex_iterations;
};

}  // namespace cuopt::linear_programming::dual_simplex
