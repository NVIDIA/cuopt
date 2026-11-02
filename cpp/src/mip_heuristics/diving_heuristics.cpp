/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "diving_heuristics.hpp"

#include <branch_and_bound/branch_and_bound.hpp>
#include <branch_and_bound/fractional.hpp>
#include <branch_and_bound/utils.hpp>
#include <dual_simplex/solve.hpp>

#include <raft/core/nvtx.hpp>

namespace cuopt::mathematical_optimization::mip {

using simplex::logger_t;
using simplex::lp_problem_t;

template <typename i_t, typename f_t>
node_status_t diving_worker_t<i_t, f_t>::classify_node(
  mip_node_t<i_t, f_t>* node_ptr,
  simplex::dual_status_t lp_status,
  f_t upper_bound,
  const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
  std::vector<i_t>& fractional,
  f_t& leaf_obj)
{
  if (lp_status != simplex::dual_status_t::OPTIMAL) { return node_status_t::FATHOMED; }

  i_t num_fractional =
    fractional_variables(settings, this->leaf_solution.x, this->var_types, fractional);
  leaf_obj = simplex::compute_objective(this->leaf_problem, this->leaf_solution.x);

  // The pseudocost update reads the lower bound inherited from the parent, so it must come first.
  this->pseudo_costs.update_pseudo_costs(node_ptr, leaf_obj);
  node_ptr->lower_bound = leaf_obj;
  snap_to_lattice(this->leaf_problem, settings, node_ptr, leaf_obj);

  if (num_fractional == 0) { return node_status_t::INTEGER_FEASIBLE; }
  if (node_ptr->lower_bound > upper_bound + settings.absolute_mip_gap_tol / 10) {
    return node_status_t::FATHOMED;
  }
  return node_status_t::HAS_CHILDREN;
}

template <typename i_t, typename f_t>
void diving_worker_t<i_t, f_t>::dive(const lp_problem_t<i_t, f_t>& original_lp,
                                     const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                                     const branch_and_bound_stats_t<i_t, f_t>& solver_stats)
{
  raft::common::nvtx::range scope("BB::diving_thread");
  if (this->orbital_fixing) { this->orbital_fixing->disable(); }
  logger_t log;
  log.log = false;

  const i_t diving_node_limit = settings.diving_settings.node_limit;
  const i_t backtrack_limit   = settings.diving_settings.backtrack_limit;
  this->recompute_basis       = true;

  bool is_start_feasible = this->rebuild_bounds_from_stack(original_lp, settings);
  if (!is_start_feasible) { return; }

  search_tree_t<i_t, f_t> dive_tree(std::move(this->start_node));

  // Since we are perform a DFS with a limit amount of backtracking, the
  // stack can hold at most `backtrack_limit` + 2 siblings nodes of the
  // current level
  circular_deque_t<mip_node_t<i_t, f_t>*> stack(backtrack_limit + 4);
  stack.push_front(&dive_tree.root);

  branch_and_bound_stats_t<i_t, f_t> dive_stats;
  f_t lower_bound = this->branch_and_bound_ptr->get_lower_bound();
  f_t upper_bound = this->branch_and_bound_ptr->get_upper_bound();
  f_t user_obj    = compute_user_objective(this->leaf_problem, upper_bound);
  f_t user_lower  = compute_user_objective(this->leaf_problem, lower_bound);
  f_t rel_gap     = user_relative_gap(user_obj, user_lower);
  f_t abs_gap     = compute_user_abs_gap(this->leaf_problem, upper_bound, lower_bound);

  while (stack.size() > 0 && !this->branch_and_bound_ptr->has_solver_stopped() &&
         rel_gap > settings.relative_mip_gap_tol && abs_gap > settings.absolute_mip_gap_tol &&
         !settings.received_halt_signal()) {
    mip_node_t<i_t, f_t>* node_ptr = stack.front();
    stack.pop_front();

    this->lower_bound = node_ptr->lower_bound;

    if (node_ptr->lower_bound > upper_bound) {
      this->recompute_basis = true;
      if (stack.size() >= 1) {
        for (i_t i = 0; i < node_ptr->depth - stack.front()->depth; ++i) {
          this->backtrack_bound_changes();
        }
      }
      continue;
    }

    if (toc(solver_stats.start_time) > settings.time_limit) { break; }
    if (dive_stats.nodes_explored >= diving_node_limit) { break; }

    int64_t bnb_lp_iters = solver_stats.total_simplex_iters;
    f_t factor           = settings.diving_settings.iteration_limit_factor;
    int64_t offset       = settings.diving_settings.iteration_limit_offset;
    int64_t max_iter     = offset + factor * bnb_lp_iters - dive_stats.total_simplex_iters;
    if (max_iter <= 0) { break; }

    decompress_vstatus(node_ptr->packed_vstatus, this->leaf_problem.num_cols, this->leaf_vstatus);
    assert(this->leaf_vstatus.size() == this->leaf_problem.num_cols);

    simplex::dual_status_t lp_status = this->solve_node_lp(original_lp,
                                                           settings,
                                                           node_ptr,
                                                           dive_stats,
                                                           log,
                                                           max_iter,
                                                           solver_stats.start_time,
                                                           upper_bound);
    ++dive_stats.nodes_explored;

    if (lp_status == simplex::dual_status_t::TIME_LIMIT) { break; }
    if (lp_status == simplex::dual_status_t::CONCURRENT_LIMIT) { break; }
    if (lp_status == simplex::dual_status_t::ITERATION_LIMIT) { break; }

    const i_t depth = node_ptr->depth;
    f_t leaf_obj;
    std::vector<i_t> fractional;
    node_status_t node_status =
      classify_node(node_ptr, lp_status, upper_bound, settings, fractional, leaf_obj);
    if (node_status == node_status_t::INTEGER_FEASIBLE) {
      this->branch_and_bound_ptr->set_solution_from_worker(this->leaf_problem,
                                                           leaf_obj,
                                                           this->leaf_solution.x,
                                                           node_ptr->depth,
                                                           this->search_strategy);
    }

    if (node_status != node_status_t::HAS_CHILDREN) {
      this->recompute_basis = true;

      if (stack.size() >= 1) {
        for (i_t i = 0; i < depth - stack.front()->depth + 1; ++i) {
          this->backtrack_bound_changes();
        }
      }

    } else {
      this->recompute_basis = false;

      auto [var, dir] = select_branch_variable(fractional, settings);
      dive_tree.branch(node_ptr,
                       var,
                       this->leaf_solution.x[var],
                       fractional.size(),
                       this->leaf_vstatus,
                       this->leaf_problem,
                       log);

      if (dir == branch_direction_t::UP) {
        stack.push_front(node_ptr->get_down_child());
        stack.push_front(node_ptr->get_up_child());
      } else {
        stack.push_front(node_ptr->get_up_child());
        stack.push_front(node_ptr->get_down_child());
      }
    }

    // Remove nodes that we can no longer backtrack to (i.e., from the current node, we can only
    // backtrack to a node that is has a depth of at most 5 levels lower than the current node).
    while (stack.size() > 1 && stack.front()->depth - stack.back()->depth > backtrack_limit) {
      stack.pop_back();
    }

    lower_bound = this->branch_and_bound_ptr->get_lower_bound();
    upper_bound = this->branch_and_bound_ptr->get_upper_bound();
    user_obj    = compute_user_objective(this->leaf_problem, upper_bound);
    user_lower  = compute_user_objective(this->leaf_problem, lower_bound);
    rel_gap     = user_relative_gap(user_obj, user_lower);
    abs_gap     = compute_user_abs_gap(this->leaf_problem, upper_bound, lower_bound);
  }
}

template <typename i_t, typename f_t>
branch_variable_t<i_t> diving_worker_t<i_t, f_t>::select_branch_variable(
  const std::vector<i_t>& fractional, const simplex::simplex_solver_settings_t<i_t, f_t>& settings)
{
  logger_t log;
  log.log                      = false;
  i_t branch_var               = -1;
  branch_direction_t round_dir = branch_direction_t::NONE;
  std::vector<f_t>& solution   = this->leaf_solution.x;

  switch (this->search_strategy) {
    case search_strategy_t::COEFFICIENT_DIVING:
      if (this->var_up_locks.empty()) {
        calculate_variable_locks(this->leaf_problem, this->var_up_locks, this->var_down_locks);
      }

      return coefficient_diving(
        this->leaf_problem, fractional, solution, this->var_up_locks, this->var_down_locks, log);

    case search_strategy_t::LINE_SEARCH_DIVING:
      return line_search_diving(fractional, solution, this->root_solution, log);

    case search_strategy_t::PSEUDOCOST_DIVING:
      return pseudocost_diving(this->pseudo_costs, fractional, solution, this->root_solution, log);

    case search_strategy_t::GUIDED_DIVING:
      assert(!this->current_incumbent.empty());
      return guided_diving(this->pseudo_costs, fractional, solution, this->current_incumbent, log);

    case search_strategy_t::FARKAS_DIVING:
      return farkas_diving(this->leaf_problem, fractional, solution, settings.zero_tol, log);

    case search_strategy_t::VECTOR_LENGTH_DIVING:
      return vector_length_diving(this->leaf_problem, fractional, solution, log);

    case search_strategy_t::BEST_FIRST:
    case search_strategy_t::RINS:  // This is used for solving the DFS of the sub-MIP.
    case search_strategy_t::RENS:
    case search_strategy_t::MUTATION:
      branch_var = this->pseudo_costs.variable_selection(fractional, solution);
      round_dir  = martin_criteria(solution[branch_var], this->root_solution[branch_var]);
      return {branch_var, round_dir};
  }

  log.debug("Unknown variable selection method: %d\n", this->search_strategy);
  return {-1, branch_direction_t::NONE};
}

template <typename i_t, typename f_t>
branch_variable_t<i_t> line_search_diving(const std::vector<i_t>& fractional,
                                          const std::vector<f_t>& solution,
                                          const std::vector<f_t>& root_solution,
                                          logger_t& log)
{
  constexpr f_t eps            = 1e-6;
  i_t branch_var               = -1;
  f_t min_score                = std::numeric_limits<f_t>::max();
  branch_direction_t round_dir = branch_direction_t::NONE;

  for (i_t j : fractional) {
    f_t score              = inf;
    branch_direction_t dir = branch_direction_t::NONE;

    if (solution[j] < root_solution[j] - eps) {
      f_t f = solution[j] - std::floor(solution[j]);
      f_t d = root_solution[j] - solution[j];
      score = f / d;
      dir   = branch_direction_t::DOWN;

    } else if (solution[j] > root_solution[j] + eps) {
      f_t f = std::ceil(solution[j]) - solution[j];
      f_t d = solution[j] - root_solution[j];
      score = f / d;
      dir   = branch_direction_t::UP;
    }

    if (min_score > score) {
      min_score  = score;
      branch_var = j;
      round_dir  = dir;
    }
  }

  // If the current solution is equal to the root solution, arbitrarily
  // set the branch variable to the first fractional variable and round it down
  if (round_dir == branch_direction_t::NONE) {
    branch_var = fractional[0];
    round_dir  = branch_direction_t::DOWN;
  }

  assert(round_dir != branch_direction_t::NONE);
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
  const i_t num_fractional = fractional.size();
  if (num_fractional == 0) return {-1, branch_direction_t::NONE};

  f_t avg_down = pc.compute_pseudocost_average_down();
  f_t avg_up   = pc.compute_pseudocost_average_up();

  i_t branch_var               = fractional[0];
  f_t max_score                = std::numeric_limits<f_t>::lowest();
  branch_direction_t round_dir = branch_direction_t::DOWN;
  constexpr f_t eps            = f_t(1e-6);

  for (i_t j : fractional) {
    f_t f_down     = solution[j] - std::floor(solution[j]);
    f_t f_up       = std::ceil(solution[j]) - solution[j];
    f_t pc_down    = pc.get_pseudocost_down(j, avg_down);
    f_t pc_up      = pc.get_pseudocost_up(j, avg_up);
    f_t score_down = std::sqrt(f_up) * (1 + pc_up) / (1 + pc_down);
    f_t score_up   = std::sqrt(f_down) * (1 + pc_down) / (1 + pc_up);

    f_t score              = 0;
    branch_direction_t dir = branch_direction_t::DOWN;

    f_t root_val = root_solution[j];

    if (solution[j] < root_val - f_t(0.4)) {
      score = score_down;
      dir   = branch_direction_t::DOWN;
    } else if (solution[j] > root_val + f_t(0.4)) {
      score = score_up;
      dir   = branch_direction_t::UP;
    } else if (f_down < f_t(0.3)) {
      score = score_down;
      dir   = branch_direction_t::DOWN;
    } else if (f_down > f_t(0.7)) {
      score = score_up;
      dir   = branch_direction_t::UP;
    } else if (pc_down < pc_up + eps) {
      score = score_down;
      dir   = branch_direction_t::DOWN;
    } else {
      score = score_up;
      dir   = branch_direction_t::UP;
    }

    if (score > max_score) {
      max_score  = score;
      branch_var = j;
      round_dir  = dir;
    }
  }

  if (round_dir == branch_direction_t::NONE) {
    branch_var = fractional[0];
    round_dir  = branch_direction_t::DOWN;
  }

  log.debug("Pseudocost diving: selected var %d with val = %e, round dir = %d and score = %e\n",
            branch_var,
            solution[branch_var],
            round_dir,
            max_score);

  return {branch_var, round_dir};
}

template <typename i_t, typename f_t>
branch_variable_t<i_t> guided_diving(pseudo_costs_t<i_t, f_t>& pc,
                                     const std::vector<i_t>& fractional,
                                     const std::vector<f_t>& solution,
                                     const std::vector<f_t>& incumbent,
                                     logger_t& log)
{
  const i_t num_fractional = fractional.size();
  if (num_fractional == 0) return {-1, branch_direction_t::NONE};

  f_t avg_down = pc.compute_pseudocost_average_down();
  f_t avg_up   = pc.compute_pseudocost_average_up();

  i_t branch_var               = fractional[0];
  f_t max_score                = std::numeric_limits<f_t>::lowest();
  branch_direction_t round_dir = branch_direction_t::DOWN;
  constexpr f_t eps            = f_t(1e-6);

  for (i_t j : fractional) {
    f_t f_down    = solution[j] - std::floor(solution[j]);
    f_t f_up      = std::ceil(solution[j]) - solution[j];
    f_t down_dist = std::abs(incumbent[j] - std::floor(solution[j]));
    f_t up_dist   = std::abs(std::ceil(solution[j]) - incumbent[j]);
    branch_direction_t dir =
      down_dist < up_dist + eps ? branch_direction_t::DOWN : branch_direction_t::UP;

    f_t pc_down = pc.get_pseudocost_down(j, avg_down);
    f_t pc_up   = pc.get_pseudocost_up(j, avg_up);
    f_t score1  = dir == branch_direction_t::DOWN ? 5 * pc_down * f_down : 5 * pc_up * f_up;
    f_t score2  = dir == branch_direction_t::DOWN ? pc_up * f_up : pc_down * f_down;
    f_t score   = (score1 + score2) / 6;

    if (score > max_score) {
      max_score  = score;
      branch_var = j;
      round_dir  = dir;
    }
  }

  log.debug("Guided diving: selected var %d with val = %e, round dir = %d and score = %e\n",
            branch_var,
            solution[branch_var],
            round_dir,
            max_score);

  return {branch_var, round_dir};
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
  i_t branch_var               = -1;
  i_t min_locks                = std::numeric_limits<i_t>::max();
  branch_direction_t round_dir = branch_direction_t::NONE;
  constexpr f_t eps            = 1e-6;

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
        round_dir = branch_direction_t::UP;
      } else if (up_lock > down_lock) {
        round_dir = branch_direction_t::DOWN;
      } else if (f_down < f_up + eps) {
        round_dir = branch_direction_t::DOWN;
      } else {
        round_dir = branch_direction_t::UP;
      }
    }
  }

  assert(round_dir != branch_direction_t::NONE);
  assert(branch_var >= 0);

  log.debug(
    "Coefficient diving: selected var %d with val = %e, round dir = %d and min locks = %d\n",
    branch_var,
    solution[branch_var],
    round_dir,
    min_locks);

  return {branch_var, round_dir};
}

template <typename i_t, typename f_t>
branch_variable_t<i_t> farkas_diving(const lp_problem_t<i_t, f_t>& lp,
                                     const std::vector<i_t>& fractional,
                                     const std::vector<f_t>& solution,
                                     f_t zero_tol,
                                     logger_t& log)
{
  if (fractional.size() == 0) return {-1, branch_direction_t::NONE};

  i_t branch_var               = -1;
  f_t max_score                = -1;
  branch_direction_t round_dir = branch_direction_t::NONE;

  for (i_t j : fractional) {
    f_t c                  = lp.objective[j];
    f_t f_down             = solution[j] - std::floor(solution[j]);
    f_t f_up               = std::ceil(solution[j]) - solution[j];
    f_t score              = 0;
    branch_direction_t dir = branch_direction_t::NONE;

    if (c > zero_tol) {
      dir = branch_direction_t::DOWN;
    } else if (c < -zero_tol) {
      dir = branch_direction_t::UP;
    } else if (f_down < 0.5) {
      dir = branch_direction_t::DOWN;
    } else {
      dir = branch_direction_t::UP;
    }

    if (dir == branch_direction_t::UP) {
      score = std::isfinite(lp.upper[j])
                ? (lp.upper[j] - std::floor(solution[j])) * f_up * std::abs(c)
                : std::numeric_limits<f_t>::infinity();
    } else {
      score = std::isfinite(lp.lower[j])
                ? (std::ceil(solution[j]) - lp.lower[j]) * f_down * std::abs(c)
                : std::numeric_limits<f_t>::infinity();
    }

    if (score > max_score) {
      max_score  = score;
      branch_var = j;
      round_dir  = dir;
    }
  }

  assert(round_dir != branch_direction_t::NONE);
  assert(branch_var >= 0);

  log.debug("Farkas diving: selected var %d with val = %e, round dir = %d\n",
            branch_var,
            solution[branch_var],
            round_dir);

  return {branch_var, round_dir};
}

template <typename i_t, typename f_t>
branch_variable_t<i_t> vector_length_diving(const lp_problem_t<i_t, f_t>& lp,
                                            const std::vector<i_t>& fractional,
                                            const std::vector<f_t>& solution,
                                            logger_t& log)
{
  if (fractional.size() == 0) return {-1, branch_direction_t::NONE};

  constexpr f_t eps            = 1E-6;
  i_t branch_var               = -1;
  f_t min_score                = std::numeric_limits<f_t>::infinity();
  branch_direction_t round_dir = branch_direction_t::NONE;

  for (i_t j : fractional) {
    f_t c                  = lp.objective[j];
    f_t f_down             = solution[j] - std::floor(solution[j]);
    f_t f_up               = std::ceil(solution[j]) - solution[j];
    i_t column_length      = lp.A.col_start[j + 1] - lp.A.col_start[j];
    f_t score              = 0;
    branch_direction_t dir = branch_direction_t::NONE;

    if (c < 0) {
      dir   = branch_direction_t::DOWN;
      score = (f_down * std::abs(c) + eps) / (column_length + 1);
    } else {
      dir   = branch_direction_t::UP;
      score = (f_up * std::abs(c) + eps) / (column_length + 1);
    }

    if (score < min_score) {
      branch_var = j;
      round_dir  = dir;
      min_score  = score;
    }
  }

  assert(round_dir != branch_direction_t::NONE);
  assert(branch_var >= 0);

  log.debug("Vector length diving: selected var %d with val = %e, round dir = %d and score = %e\n",
            branch_var,
            solution[branch_var],
            round_dir,
            min_score);

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

template branch_variable_t<int> farkas_diving(const lp_problem_t<int, double>& lp_problem,
                                              const std::vector<int>& fractional,
                                              const std::vector<double>& solution,
                                              double zero_tol,
                                              logger_t& log);

template branch_variable_t<int> vector_length_diving(const lp_problem_t<int, double>& lp_problem,
                                                     const std::vector<int>& fractional,
                                                     const std::vector<double>& solution,
                                                     logger_t& log);

template class diving_worker_t<int, double>;

#endif

}  // namespace cuopt::mathematical_optimization::mip
