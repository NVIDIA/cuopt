/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/user_problem.hpp>

#include <cstdint>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Pure-host computation of the objective step for the case where lattice propagation is
// required (i.e. at least one variable with nonzero objective coefficient is continuous
// and not implied-integer). Returns a default-constructed (zero) objective_step_t when no
// nontrivial lattice is found.
//
// Callers should handle the fast path themselves: when every variable with nonzero
// objective coefficient is already lattice-known, step = gcd(|c_j|) can be computed
// without ever touching the constraint matrix.
template <typename i_t, typename f_t>
simplex::objective_step_t<f_t> compute_objective_step_info(
  const std::vector<f_t>& obj_coefs,
  const std::vector<bool>& is_lattice_known_initially,
  const std::vector<i_t>& offsets,
  const std::vector<i_t>& variables,
  const std::vector<f_t>& coefficients,
  const std::vector<f_t>& con_lb,
  const std::vector<f_t>& con_ub);

// Lattice propagation: for each variable, determine the step size of its lattice.
// Integer/implied-integer variables have step=1. Continuous variables may have their
// step discovered by propagating through equality constraints (where exactly one unknown
// remains), and for a single unknown objective variable, through inequality constraints
// using the objective-direction argument. The problem is assumed to be in minimization form.
//
// Returns true if at least one originally-unknown variable's step was discovered.
// On return, lattice_step[j] is the step for variable j (0 if still unknown).
template <typename i_t, typename f_t>
bool propagate_lattice(i_t n_vars,
                       i_t n_cons,
                       const std::vector<i_t>& offsets,
                       const std::vector<i_t>& variables,
                       const std::vector<f_t>& coefficients,
                       const std::vector<f_t>& con_lb,
                       const std::vector<f_t>& con_ub,
                       const std::vector<bool>& is_lattice_known_initially,
                       const std::vector<f_t>& obj_coefs,
                       std::vector<f_t>& lattice_step);

}  // namespace cuopt::mathematical_optimization::mip
