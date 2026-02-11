/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/pseudo_costs.hpp>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
branch_variable_t<i_t> line_search_diving(const std::vector<i_t>& fractional,
                                          const std::vector<f_t>& solution,
                                          const std::vector<f_t>& root_solution,
                                          logger_t& log);

template <typename i_t, typename f_t>
branch_variable_t<i_t> pseudocost_diving(pseudo_costs_t<i_t, f_t>& pc,
                                         const std::vector<i_t>& fractional,
                                         const std::vector<f_t>& solution,
                                         const std::vector<f_t>& root_solution,
                                         logger_t& log);

template <typename i_t, typename f_t>
branch_variable_t<i_t> guided_diving(pseudo_costs_t<i_t, f_t>& pc,
                                     const std::vector<i_t>& fractional,
                                     const std::vector<f_t>& solution,
                                     const std::vector<f_t>& incumbent,
                                     logger_t& log);

// Calculate the variable locks assuming that the constraints
// has the following format: `Ax = b`.
template <typename i_t, typename f_t>
void calculate_variable_locks(const lp_problem_t<i_t, f_t>& lp_problem,
                              std::vector<i_t>& up_locks,
                              std::vector<i_t>& down_locks);

template <typename i_t, typename f_t>
branch_variable_t<i_t> coefficient_diving(const lp_problem_t<i_t, f_t>& lp_problem,
                                          const std::vector<i_t>& fractional,
                                          const std::vector<f_t>& solution,
                                          const std::vector<i_t>& up_locks,
                                          const std::vector<i_t>& down_locks,
                                          logger_t& log);

}  // namespace cuopt::linear_programming::dual_simplex
