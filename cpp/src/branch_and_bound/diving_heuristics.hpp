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

// When `log_diving_type` is true, each diving strategy gets its own letter;
// otherwise every dive collapses to 'D'.
inline char feasible_solution_symbol(search_strategy_t strategy, bool log_diving_type)
{
  if (strategy == BEST_FIRST) return 'B';
  if (!log_diving_type) { return 'D'; }
  switch (strategy) {
    case COEFFICIENT_DIVING: return 'C';
    case LINE_SEARCH_DIVING: return 'L';
    case PSEUDOCOST_DIVING: return 'P';
    case GUIDED_DIVING: return 'G';
    case FARKAS_DIVING: return 'F';
    case VECTOR_LENGTH_DIVING: return 'V';
    default: return 'U';
  }
}

template <typename i_t, typename f_t>
bool is_search_strategy_enabled(search_strategy_t strategy,
                                const mip_diving_hyper_params_t<i_t, f_t>& settings)
{
  switch (strategy) {
    case BEST_FIRST: return true;
    case PSEUDOCOST_DIVING: return settings.pseudocost_diving != 0;
    case LINE_SEARCH_DIVING: return settings.line_search_diving != 0;
    case GUIDED_DIVING: return settings.guided_diving != 0;
    case COEFFICIENT_DIVING: return settings.coefficient_diving != 0;
    case FARKAS_DIVING: return settings.farkas_diving != 0;
    case VECTOR_LENGTH_DIVING: return settings.vector_length_diving != 0;
  }

  return false;
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
