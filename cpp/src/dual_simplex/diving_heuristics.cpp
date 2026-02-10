/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/diving_heuristics.hpp>
#include <tuple>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
branch_variable_t<i_t> line_search_diving(const std::vector<i_t>& fractional,
                                          const std::vector<f_t>& solution,
                                          const std::vector<f_t>& root_solution,
                                          logger_t& log)
{
  constexpr f_t eps              = 1e-6;
  i_t branch_var                 = -1;
  f_t min_score                  = std::numeric_limits<f_t>::max();
  rounding_direction_t round_dir = rounding_direction_t::NONE;

  for (i_t j : fractional) {
    f_t score                = inf;
    rounding_direction_t dir = rounding_direction_t::NONE;

    if (solution[j] < root_solution[j] - eps) {
      f_t f = solution[j] - std::floor(solution[j]);
      f_t d = root_solution[j] - solution[j];
      score = f / d;
      dir   = rounding_direction_t::DOWN;

    } else if (solution[j] > root_solution[j] + eps) {
      f_t f = std::ceil(solution[j]) - solution[j];
      f_t d = solution[j] - root_solution[j];
      score = f / d;
      dir   = rounding_direction_t::UP;
    }

    if (min_score > score) {
      min_score  = score;
      branch_var = j;
      round_dir  = dir;
    }
  }

  // If the current solution is equal to the root solution, arbitrarily
  // set the branch variable to the first fractional variable and round it down
  if (round_dir == rounding_direction_t::NONE) {
    branch_var = fractional[0];
    round_dir  = rounding_direction_t::DOWN;
  }

  assert(round_dir != rounding_direction_t::NONE);
  assert(branch_var >= 0);

  log.debug("Line search diving: selected var %d with val = %e, round dir = %d and score = %e\n",
            branch_var,
            solution[branch_var],
            round_dir,
            min_score);

  return {branch_var, round_dir};
}

template <typename i_t, typename f_t>
branch_variable_t<i_t> pseudocost_diving(pseudo_costs_t<i_t, f_t>& pc,
                                         const std::vector<i_t>& fractional,
                                         const std::vector<f_t>& solution,
                                         const std::vector<f_t>& root_solution,
                                         logger_t& log)
{
  return pseudocost_diving_from_arrays(pc.pseudo_cost_sum_down.data(),
                                       pc.pseudo_cost_sum_up.data(),
                                       pc.pseudo_cost_num_down.data(),
                                       pc.pseudo_cost_num_up.data(),
                                       (i_t)pc.pseudo_cost_sum_down.size(),
                                       fractional,
                                       solution,
                                       root_solution);
}

template <typename i_t, typename f_t>
branch_variable_t<i_t> guided_diving(pseudo_costs_t<i_t, f_t>& pc,
                                     const std::vector<i_t>& fractional,
                                     const std::vector<f_t>& solution,
                                     const std::vector<f_t>& incumbent,
                                     logger_t& log)
{
  return guided_diving_from_arrays(pc.pseudo_cost_sum_down.data(),
                                   pc.pseudo_cost_sum_up.data(),
                                   pc.pseudo_cost_num_down.data(),
                                   pc.pseudo_cost_num_up.data(),
                                   (i_t)pc.pseudo_cost_sum_down.size(),
                                   fractional,
                                   solution,
                                   incumbent);
}

template <typename i_t, typename f_t>
void calculate_variable_locks(const lp_problem_t<i_t, f_t>& lp_problem,
                              std::vector<i_t>& up_locks,
                              std::vector<i_t>& down_locks)
{
  constexpr f_t eps = 1E-6;
  up_locks.assign(lp_problem.num_cols, 0);
  down_locks.assign(lp_problem.num_cols, 0);

  for (i_t j = 0; j < lp_problem.num_cols; ++j) {
    i_t start = lp_problem.A.col_start[j];
    i_t end   = lp_problem.A.col_start[j + 1];

    for (i_t p = start; p < end; ++p) {
      f_t val = lp_problem.A.x[p];
      if (std::abs(val) > eps) {
        up_locks[j]++;
        down_locks[j]++;
      }
    }
  }
}

template <typename i_t, typename f_t>
branch_variable_t<i_t> coefficient_diving(const lp_problem_t<i_t, f_t>& lp_problem,
                                          const std::vector<i_t>& fractional,
                                          const std::vector<f_t>& solution,
                                          const std::vector<i_t>& up_locks,
                                          const std::vector<i_t>& down_locks,
                                          logger_t& log)
{
  i_t branch_var                 = -1;
  i_t min_locks                  = std::numeric_limits<i_t>::max();
  rounding_direction_t round_dir = rounding_direction_t::NONE;
  constexpr f_t eps              = 1e-6;

  for (i_t j : fractional) {
    f_t f_down    = solution[j] - std::floor(solution[j]);
    f_t f_up      = std::ceil(solution[j]) - solution[j];
    i_t up_lock   = up_locks[j];
    i_t down_lock = down_locks[j];
    f_t upper     = lp_problem.upper[j];
    f_t lower     = lp_problem.lower[j];
    if (std::isfinite(upper)) { up_lock++; }
    if (std::isfinite(lower)) { down_lock++; }
    i_t alpha = std::min(up_lock, down_lock);

    if (min_locks > alpha) {
      min_locks  = alpha;
      branch_var = j;

      if (up_lock < down_lock) {
        round_dir = rounding_direction_t::UP;
      } else if (up_lock > down_lock) {
        round_dir = rounding_direction_t::DOWN;
      } else if (f_down < f_up + eps) {
        round_dir = rounding_direction_t::DOWN;
      } else {
        round_dir = rounding_direction_t::UP;
      }
    }
  }

  assert(round_dir != rounding_direction_t::NONE);
  assert(branch_var >= 0);

  log.debug(
    "Coefficient diving: selected var %d with val = %e, round dir = %d and min locks = %d\n",
    branch_var,
    solution[branch_var],
    round_dir,
    min_locks);

  return {branch_var, round_dir};
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template branch_variable_t<int> line_search_diving(const std::vector<int>& fractional,
                                                   const std::vector<double>& solution,
                                                   const std::vector<double>& root_solution,
                                                   logger_t& log);

template branch_variable_t<int> pseudocost_diving(pseudo_costs_t<int, double>& pc,
                                                  const std::vector<int>& fractional,
                                                  const std::vector<double>& solution,
                                                  const std::vector<double>& root_solution,
                                                  logger_t& log);

template branch_variable_t<int> guided_diving(pseudo_costs_t<int, double>& pc,
                                              const std::vector<int>& fractional,
                                              const std::vector<double>& solution,
                                              const std::vector<double>& incumbent,
                                              logger_t& log);

template void calculate_variable_locks(const lp_problem_t<int, double>& lp_problem,
                                       std::vector<int>& up_locks,
                                       std::vector<int>& down_locks);

template branch_variable_t<int> coefficient_diving(const lp_problem_t<int, double>& lp_problem,
                                                   const std::vector<int>& fractional,
                                                   const std::vector<double>& solution,
                                                   const std::vector<int>& up_locks,
                                                   const std::vector<int>& down_locks,
                                                   logger_t& log);
#endif

}  // namespace cuopt::linear_programming::dual_simplex
