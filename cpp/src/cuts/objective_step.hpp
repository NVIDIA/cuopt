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

namespace cuopt::linear_programming::dual_simplex {

// Pure-host computation of the objective step for the case where lattice propagation is
// required (i.e. at least one variable with nonzero objective coefficient is continuous
// and not implied-integer). Returns a default-constructed (zero) objective_step_t when no
// nontrivial lattice is found.
//
// Callers should handle the fast path themselves: when every variable with nonzero
// objective coefficient is already lattice-known, step = gcd(|c_j|) and bias =
// sum(c_j * lb_j) mod step can be computed without ever touching the constraint matrix.
//
// is_lattice_known_initially[j] must be true exactly for variables whose lattice is
// known at entry (integer or implied-integer). See propagate_lattice for the full
// contract on that flag.
template <typename i_t, typename f_t>
objective_step_t<f_t> compute_objective_step_info(
  const std::vector<f_t>& obj_coefs,
  const std::vector<f_t>& var_lb,
  const std::vector<bool>& is_lattice_known_initially,
  const std::vector<i_t>& offsets,
  const std::vector<i_t>& variables,
  const std::vector<f_t>& coefficients,
  const std::vector<f_t>& con_lb,
  const std::vector<f_t>& con_ub);

// Lattice propagation: for each variable, determine if it must lie on a lattice
// x_j = k * step_j + bias_j for integer k. This is done by scanning equality rows of
// the constraint matrix for rows in which exactly one unknown remains, solving for it
// in terms of the other (already-lattice-known) variables, and updating that
// variable's lattice. Discoveries make further rows productive and the process is
// iterated to a fixed point via a worklist.
//
// is_lattice_known_initially[j] must be true for any variable whose lattice is known at
// entry (integer or implied-integer variables): such a variable's lattice is initialized
// to (step = 1, bias = lower bound) and is treated as "known" throughout. Every other
// variable starts unknown and may have its lattice discovered by the propagation.
//
// Returns true if at least one originally-unknown variable's lattice was discovered.
// On return, lattice_step[j] / lattice_bias[j] are populated for every variable whose
// lattice is known (initial or discovered); for still-unknown variables they are zero.
//
// Internally uses rational arithmetic (int64_t numerator/denominator pairs) to avoid
// floating-point GCD issues. The CSR row pointers are passed as offsets/variables/
// coefficients with offsets sized n_cons + 1.
template <typename i_t, typename f_t>
bool propagate_lattice(i_t n_vars,
                       i_t n_cons,
                       const std::vector<i_t>& offsets,
                       const std::vector<i_t>& variables,
                       const std::vector<f_t>& coefficients,
                       const std::vector<f_t>& con_lb,
                       const std::vector<f_t>& con_ub,
                       const std::vector<f_t>& var_lb,
                       const std::vector<bool>& is_lattice_known_initially,
                       std::vector<f_t>& lattice_step,
                       std::vector<f_t>& lattice_bias);

}  // namespace cuopt::linear_programming::dual_simplex
