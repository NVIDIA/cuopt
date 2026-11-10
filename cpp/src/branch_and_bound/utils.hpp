/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/mip/mip_hyper_params.hpp>

#include <branch_and_bound/constants.hpp>
#include <branch_and_bound/mip_node.hpp>

#include <dual_simplex/phase2.hpp>
#include <dual_simplex/solve.hpp>

#include <utilities/logger.hpp>

#include <cmath>
#include <format>
#include <limits>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

inline simplex::dual_status_t convert_lp_status_to_dual_status(simplex::lp_status_t status)
{
  switch (status) {
    case simplex::lp_status_t::OPTIMAL: return simplex::dual_status_t::OPTIMAL;
    case simplex::lp_status_t::INFEASIBLE: return simplex::dual_status_t::DUAL_UNBOUNDED;
    case simplex::lp_status_t::ITERATION_LIMIT: return simplex::dual_status_t::ITERATION_LIMIT;
    case simplex::lp_status_t::TIME_LIMIT: return simplex::dual_status_t::TIME_LIMIT;
    case simplex::lp_status_t::WORK_LIMIT: return simplex::dual_status_t::WORK_LIMIT;
    case simplex::lp_status_t::NUMERICAL_ISSUES: return simplex::dual_status_t::NUMERICAL;
    case simplex::lp_status_t::CUTOFF: return simplex::dual_status_t::CUTOFF;
    case simplex::lp_status_t::CONCURRENT_LIMIT: return simplex::dual_status_t::CONCURRENT_LIMIT;
    case simplex::lp_status_t::UNSET: return simplex::dual_status_t::UNSET;
    default: return simplex::dual_status_t::NUMERICAL;
  }
}

inline char feasible_solution_symbol(heuristics_origin_t origin)
{
  switch (origin) {
    case heuristics_origin_t::SUBMIP: return 'S';
    case heuristics_origin_t::HEURISTICS: return 'H';
  }
  return 'U';
}

// When `log_diving_type` is true, each diving strategy gets its own letter;
// otherwise every dive collapses to 'D'.
inline char feasible_solution_symbol(search_strategy_t strategy, bool show_diving)
{
  if (strategy == search_strategy_t::BEST_FIRST) return 'B';
  if (strategy == search_strategy_t::RINS || strategy == search_strategy_t::RENS) return 'S';
  if (!show_diving) return 'D';

  switch (strategy) {
    case search_strategy_t::BEST_FIRST: return 'B';
    case search_strategy_t::COEFFICIENT_DIVING: return 'C';
    case search_strategy_t::LINE_SEARCH_DIVING: return 'L';
    case search_strategy_t::PSEUDOCOST_DIVING: return 'P';
    case search_strategy_t::GUIDED_DIVING: return 'G';
    case search_strategy_t::FARKAS_DIVING: return 'F';
    case search_strategy_t::VECTOR_LENGTH_DIVING: return 'V';
    case search_strategy_t::RINS: return 'S';
    case search_strategy_t::RENS: return 'S';
    case search_strategy_t::MUTATION: return 'S';
  }

  return 'U';
}

template <typename i_t, typename f_t>
f_t compute_user_abs_gap(const simplex::lp_problem_t<i_t, f_t>& lp, f_t obj_value, f_t lower_bound)
{
  // abs_gap = |user_obj - user_lower| = |obj_scale| * |obj_value - lower_bound|
  // obj_constant cancels out in the subtraction; obj_scale sign must be removed via abs
  f_t gap = std::abs(lp.obj_scale) * (obj_value - lower_bound);
  if (gap < -1e-4) { CUOPT_LOG_DEBUG("Gap is negative %e", gap); }
  return gap;
}

template <typename f_t>
f_t user_relative_gap(f_t user_obj, f_t user_lower_bound)
{
  f_t user_mip_gap = user_obj == 0.0
                       ? (user_lower_bound == 0.0 ? 0.0 : std::numeric_limits<f_t>::infinity())
                       : std::abs(user_obj - user_lower_bound) / std::abs(user_obj);
  if (std::isnan(user_mip_gap)) { return std::numeric_limits<f_t>::infinity(); }
  return user_mip_gap;
}

template <typename f_t>
std::string to_percentage(f_t value)
{
  if (value == std::numeric_limits<f_t>::infinity()) return "-";
  if (value > 1e-3) { return std::format("{:5.1f}%", value * 100); }
  return std::format("{:5.2f}%", value * 100);
}

// If the objective is integral or must move in steps than
// the lower bound will be different from the leaf objective.
// We use the leaf objective for RINS (on_optimal_callback)
// and if we are integer feasible (handle_integer_solution).
// We use the lower bound to decide if we should fathom the
// node or branch.
template <typename i_t, typename f_t>
void snap_to_lattice(const simplex::lp_problem_t<i_t, f_t>& lp,
                     const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                     mip_node_t<i_t, f_t>* node_ptr,
                     f_t leaf_obj)
{
  if (lp.objective_step.has_step()) {
    f_t step = lp.objective_step.step_size;
    f_t bias = lp.objective_step.bias;
    // Round up to next value on the lattice: k * step + bias >= leaf_obj
    f_t k                 = std::ceil((leaf_obj - bias) / step - settings.integer_tol);
    node_ptr->lower_bound = k * step + bias;
  } else if (lp.objective_is_integral) {
    node_ptr->lower_bound = std::ceil(leaf_obj - settings.integer_tol);
  }
}

// Largest objective value of a solution that improves on upper_bound. This is the value just
// below upper_bound on the objective lattice if the objective is integral or moves in steps.
template <typename i_t, typename f_t>
f_t compute_node_cutoff(const simplex::lp_problem_t<i_t, f_t>& lp,
                        const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                        f_t upper_bound)
{
  if (lp.objective_step.has_step()) {
    f_t step = lp.objective_step.step_size;
    f_t bias = lp.objective_step.bias;
    f_t k    = std::ceil((upper_bound - bias) / step - settings.integer_tol);
    return (k - 1) * step + bias + settings.dual_tol;
  } else if (lp.objective_is_integral) {
    return std::ceil(upper_bound - settings.integer_tol) - 1 + settings.dual_tol;
  }
  return upper_bound + settings.dual_tol;
}

template <typename i_t, typename f_t>
void get_diving_heuristic_list(const mip_hyper_params_t<i_t, f_t>& settings,
                               std::vector<search_strategy_t>& heuristic_list)
{
  heuristic_list.clear();
  if (settings.diving_pseudocost != 0)
    heuristic_list.push_back(search_strategy_t::PSEUDOCOST_DIVING);
  if (settings.diving_line_search != 0)
    heuristic_list.push_back(search_strategy_t::LINE_SEARCH_DIVING);
  if (settings.diving_guided != 0) heuristic_list.push_back(search_strategy_t::GUIDED_DIVING);
  if (settings.diving_coefficient != 0)
    heuristic_list.push_back(search_strategy_t::COEFFICIENT_DIVING);
  if (settings.diving_farkas != 0) heuristic_list.push_back(search_strategy_t::FARKAS_DIVING);
  if (settings.diving_vector_length != 0)
    heuristic_list.push_back(search_strategy_t::VECTOR_LENGTH_DIVING);
}

}  // namespace cuopt::mathematical_optimization::mip
