/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/mip/diving_hyper_params.hpp>

#include <branch_and_bound/pseudo_costs.hpp>

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/bounds_strengthening.hpp>

#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void get_diving_heuristic_list(const mip_diving_hyper_params_t<i_t, f_t>& settings,
                               std::vector<search_strategy_t>& heuristic_list)
{
  heuristic_list.clear();
  if (settings.pseudocost_diving != 0)
    heuristic_list.push_back(search_strategy_t::PSEUDOCOST_DIVING);
  if (settings.line_search_diving != 0)
    heuristic_list.push_back(search_strategy_t::LINE_SEARCH_DIVING);
  if (settings.guided_diving != 0) heuristic_list.push_back(search_strategy_t::GUIDED_DIVING);
  if (settings.coefficient_diving != 0)
    heuristic_list.push_back(search_strategy_t::COEFFICIENT_DIVING);
  if (settings.farkas_diving != 0) heuristic_list.push_back(search_strategy_t::FARKAS_DIVING);
  if (settings.vector_length_diving != 0)
    heuristic_list.push_back(search_strategy_t::VECTOR_LENGTH_DIVING);
}

template <typename i_t, typename f_t>
branch_variable_t<i_t> line_search_diving(const std::vector<i_t>& fractional,
                                          const std::vector<f_t>& solution,
                                          const std::vector<f_t>& root_solution,
                                          simplex::logger_t& log);

template <typename i_t, typename f_t>
branch_variable_t<i_t> pseudocost_diving(pseudo_costs_t<i_t, f_t>& pc,
                                         const std::vector<i_t>& fractional,
                                         const std::vector<f_t>& solution,
                                         const std::vector<f_t>& root_solution,
                                         simplex::logger_t& log);

template <typename i_t, typename f_t>
branch_variable_t<i_t> guided_diving(pseudo_costs_t<i_t, f_t>& pc,
                                     const std::vector<i_t>& fractional,
                                     const std::vector<f_t>& solution,
                                     const std::vector<f_t>& incumbent,
                                     simplex::logger_t& log);

// Calculate the variable locks assuming that the constraints
// has the following format: `Ax = b`.
template <typename i_t, typename f_t>
void calculate_variable_locks(const simplex::lp_problem_t<i_t, f_t>& lp_problem,
                              std::vector<i_t>& up_locks,
                              std::vector<i_t>& down_locks);

template <typename i_t, typename f_t>
branch_variable_t<i_t> coefficient_diving(const simplex::lp_problem_t<i_t, f_t>& lp_problem,
                                          const std::vector<i_t>& fractional,
                                          const std::vector<f_t>& solution,
                                          const std::vector<i_t>& up_locks,
                                          const std::vector<i_t>& down_locks,
                                          simplex::logger_t& log);

template <typename i_t, typename f_t>
branch_variable_t<i_t> farkas_diving(const simplex::lp_problem_t<i_t, f_t>& lp,
                                     const std::vector<i_t>& fractional,
                                     const std::vector<f_t>& solution,
                                     f_t zero_tol,
                                     simplex::logger_t& log);

template <typename i_t, typename f_t>
branch_variable_t<i_t> vector_length_diving(const simplex::lp_problem_t<i_t, f_t>& lp,
                                            const std::vector<i_t>& fractional,
                                            const std::vector<f_t>& solution,
                                            simplex::logger_t& log);

}  // namespace cuopt::mathematical_optimization::mip
