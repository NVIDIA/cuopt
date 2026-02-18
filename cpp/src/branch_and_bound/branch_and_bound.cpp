/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <branch_and_bound/branch_and_bound.hpp>
#include <branch_and_bound/mip_node.hpp>
#include <branch_and_bound/pseudo_costs.hpp>

#include <cuts/cuts.hpp>

#include <dual_simplex/basis_solves.hpp>
#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/crossover.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/logger.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/random.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <dual_simplex/user_problem.hpp>

#include <raft/common/nvtx.hpp>
#include <utilities/hashing.hpp>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <future>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

namespace {

template <typename f_t>
bool is_fractional(f_t x, variable_type_t var_type, f_t integer_tol)
{
  if (var_type == variable_type_t::CONTINUOUS) {
    return false;
  } else {
    f_t x_integer = std::round(x);
    return (std::abs(x_integer - x) > integer_tol);
  }
}

template <typename i_t, typename f_t>
i_t fractional_variables(const simplex_solver_settings_t<i_t, f_t>& settings,
                         const std::vector<f_t>& x,
                         const std::vector<variable_type_t>& var_types,
                         std::vector<i_t>& fractional)
{
  const i_t n = x.size();
  assert(x.size() == var_types.size());
  for (i_t j = 0; j < n; ++j) {
    if (is_fractional(x[j], var_types[j], settings.integer_tol)) { fractional.push_back(j); }
  }
  return fractional.size();
}

template <typename i_t, typename f_t>
void full_variable_types(const user_problem_t<i_t, f_t>& original_problem,
                         const lp_problem_t<i_t, f_t>& original_lp,
                         std::vector<variable_type_t>& var_types)
{
  var_types = original_problem.var_types;
  if (original_lp.num_cols > original_problem.num_cols) {
    var_types.resize(original_lp.num_cols);
    for (i_t k = original_problem.num_cols; k < original_lp.num_cols; k++) {
      var_types[k] = variable_type_t::CONTINUOUS;
    }
  }
}

template <typename i_t, typename f_t>
bool check_guess(const lp_problem_t<i_t, f_t>& original_lp,
                 const simplex_solver_settings_t<i_t, f_t>& settings,
                 const std::vector<variable_type_t>& var_types,
                 const std::vector<f_t>& guess,
                 f_t& primal_error,
                 f_t& bound_error,
                 i_t& num_fractional)
{
  bool feasible = false;
  std::vector<f_t> residual(original_lp.num_rows);
  residual = original_lp.rhs;
  matrix_vector_multiply(original_lp.A, 1.0, guess, -1.0, residual);
  primal_error           = vector_norm_inf<i_t, f_t>(residual);
  bound_error            = 0.0;
  constexpr bool verbose = false;
  for (i_t j = 0; j < original_lp.num_cols; j++) {
    // l_j <= x_j  infeas means x_j < l_j or l_j - x_j > 0
    const f_t low_bound_err = std::max(0.0, original_lp.lower[j] - guess[j]);
    // x_j <= u_j infeas means u_j < x_j or x_j - u_j > 0
    const f_t up_bound_err = std::max(0.0, guess[j] - original_lp.upper[j]);

    if (verbose && (low_bound_err > settings.primal_tol || up_bound_err > settings.primal_tol)) {
      settings.log.printf(
        "Bound error %d variable value %e. Low %e Upper %e. Low Error %e Up Error %e\n",
        j,
        guess[j],
        original_lp.lower[j],
        original_lp.upper[j],
        low_bound_err,
        up_bound_err);
    }
    bound_error = std::max(bound_error, std::max(low_bound_err, up_bound_err));
  }
  if (verbose) { settings.log.printf("Bounds infeasibility %e\n", bound_error); }
  std::vector<i_t> fractional;
  num_fractional = fractional_variables(settings, guess, var_types, fractional);
  if (verbose) { settings.log.printf("Fractional in solution %d\n", num_fractional); }
  if (bound_error < settings.primal_tol && primal_error < 2 * settings.primal_tol &&
      num_fractional == 0) {
    if (verbose) { settings.log.printf("Solution is feasible\n"); }
    feasible = true;
  }
  return feasible;
}

template <typename i_t, typename f_t>
void set_uninitialized_steepest_edge_norms(const lp_problem_t<i_t, f_t>& lp,
                                           const std::vector<i_t>& basic_list,
                                           std::vector<f_t>& edge_norms)
{
  if (edge_norms.size() != lp.num_cols) { edge_norms.resize(lp.num_cols, -1.0); }
  for (i_t k = 0; k < lp.num_rows; k++) {
    const i_t j = basic_list[k];
    if (edge_norms[j] <= 0.0) { edge_norms[j] = 1e-4; }
  }
}

dual::status_t convert_lp_status_to_dual_status(lp_status_t status)
{
  if (status == lp_status_t::OPTIMAL) {
    return dual::status_t::OPTIMAL;
  } else if (status == lp_status_t::INFEASIBLE) {
    return dual::status_t::DUAL_UNBOUNDED;
  } else if (status == lp_status_t::ITERATION_LIMIT) {
    return dual::status_t::ITERATION_LIMIT;
  } else if (status == lp_status_t::TIME_LIMIT) {
    return dual::status_t::TIME_LIMIT;
  } else if (status == lp_status_t::WORK_LIMIT) {
    return dual::status_t::WORK_LIMIT;
  } else if (status == lp_status_t::NUMERICAL_ISSUES) {
    return dual::status_t::NUMERICAL;
  } else if (status == lp_status_t::CUTOFF) {
    return dual::status_t::CUTOFF;
  } else if (status == lp_status_t::CONCURRENT_LIMIT) {
    return dual::status_t::CONCURRENT_LIMIT;
  } else if (status == lp_status_t::UNSET) {
    return dual::status_t::UNSET;
  } else {
    return dual::status_t::NUMERICAL;
  }
}

template <typename f_t>
f_t sgn(f_t x)
{
  return x < 0 ? -1 : 1;
}

template <typename f_t>
f_t relative_gap(f_t obj_value, f_t lower_bound)
{
  f_t user_mip_gap = obj_value == 0.0
                       ? (lower_bound == 0.0 ? 0.0 : std::numeric_limits<f_t>::infinity())
                       : std::abs(obj_value - lower_bound) / std::abs(obj_value);
  if (std::isnan(user_mip_gap)) { return std::numeric_limits<f_t>::infinity(); }
  return user_mip_gap;
}

template <typename i_t, typename f_t>
f_t user_relative_gap(const lp_problem_t<i_t, f_t>& lp, f_t obj_value, f_t lower_bound)
{
  f_t user_obj         = compute_user_objective(lp, obj_value);
  f_t user_lower_bound = compute_user_objective(lp, lower_bound);
  f_t user_mip_gap     = user_obj == 0.0
                           ? (user_lower_bound == 0.0 ? 0.0 : std::numeric_limits<f_t>::infinity())
                           : std::abs(user_obj - user_lower_bound) / std::abs(user_obj);
  if (std::isnan(user_mip_gap)) { return std::numeric_limits<f_t>::infinity(); }
  return user_mip_gap;
}

template <typename f_t>
std::string user_mip_gap(f_t obj_value, f_t lower_bound)
{
  const f_t user_mip_gap = relative_gap(obj_value, lower_bound);
  if (user_mip_gap == std::numeric_limits<f_t>::infinity()) {
    return "   -  ";
  } else {
    constexpr int BUFFER_LEN = 32;
    char buffer[BUFFER_LEN];
    snprintf(buffer, BUFFER_LEN - 1, "%5.1f%%", user_mip_gap * 100);
    return std::string(buffer);
  }
}

#ifdef SHOW_DIVING_TYPE
inline char feasible_solution_symbol(search_strategy_t strategy)
{
  switch (strategy) {
    case search_strategy_t::BEST_FIRST: return 'B';
    case search_strategy_t::COEFFICIENT_DIVING: return 'C';
    case search_strategy_t::LINE_SEARCH_DIVING: return 'L';
    case search_strategy_t::PSEUDOCOST_DIVING: return 'P';
    case search_strategy_t::GUIDED_DIVING: return 'G';
    default: return 'U';
  }
}
#else
inline char feasible_solution_symbol(search_strategy_t strategy)
{
  switch (strategy) {
    case search_strategy_t::BEST_FIRST: return 'B';
    case search_strategy_t::COEFFICIENT_DIVING: return 'D';
    case search_strategy_t::LINE_SEARCH_DIVING: return 'D';
    case search_strategy_t::PSEUDOCOST_DIVING: return 'D';
    case search_strategy_t::GUIDED_DIVING: return 'D';
    default: return 'U';
  }
}
#endif

}  // namespace

template <typename i_t, typename f_t>
branch_and_bound_t<i_t, f_t>::branch_and_bound_t(
  const user_problem_t<i_t, f_t>& user_problem,
  const simplex_solver_settings_t<i_t, f_t>& solver_settings,
  f_t start_time)
  : original_problem_(user_problem),
    settings_(solver_settings),
    original_lp_(user_problem.handle_ptr, 1, 1, 1),
    Arow_(1, 1, 0),
    incumbent_(1),
    root_relax_soln_(1, 1),
    root_crossover_soln_(1, 1),
    pc_(1),
    solver_status_(mip_status_t::UNSET)
{
  exploration_stats_.start_time = start_time;
#ifdef PRINT_CONSTRAINT_MATRIX
  settings_.log.printf("A");
  original_problem_.A.print_matrix();
#endif

  dualize_info_t<i_t, f_t> dualize_info;
  convert_user_problem(original_problem_, settings_, original_lp_, new_slacks_, dualize_info);
  full_variable_types(original_problem_, original_lp_, var_types_);

  // Check slack
#ifdef CHECK_SLACKS
  assert(new_slacks_.size() == original_lp_.num_rows);
  for (i_t slack : new_slacks_) {
    const i_t col_start = original_lp_.A.col_start[slack];
    const i_t col_end   = original_lp_.A.col_start[slack + 1];
    const i_t col_len   = col_end - col_start;
    if (col_len != 1) {
      settings_.log.printf("Slack %d has %d nzs\n", slack, col_len);
      assert(col_len == 1);
    }
    const i_t i = original_lp_.A.i[col_start];
    const f_t x = original_lp_.A.x[col_start];
    if (std::abs(x) != 1.0) {
      settings_.log.printf("Slack %d row %d has non-unit coefficient %e\n", slack, i, x);
      assert(std::abs(x) == 1.0);
    }
  }
#endif

  upper_bound_    = inf;
  root_objective_ = std::numeric_limits<f_t>::quiet_NaN();
}

template <typename i_t, typename f_t>
f_t branch_and_bound_t<i_t, f_t>::get_lower_bound()
{
  f_t lower_bound      = lower_bound_ceiling_.load();
  f_t heap_lower_bound = node_queue_.get_lower_bound();
  lower_bound          = std::min(heap_lower_bound, lower_bound);
  lower_bound          = std::min(worker_pool_.get_lower_bound(), lower_bound);

  if (std::isfinite(lower_bound)) {
    return lower_bound;
  } else if (std::isfinite(root_objective_)) {
    return root_objective_;
  } else {
    return -inf;
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::report_heuristic(f_t obj)
{
  if (is_running_) {
    f_t user_obj         = compute_user_objective(original_lp_, obj);
    f_t user_lower       = compute_user_objective(original_lp_, get_lower_bound());
    std::string user_gap = user_mip_gap<f_t>(user_obj, user_lower);

    settings_.log.printf(
      "H                            %+13.6e    %+10.6e                               %s %9.2f\n",
      user_obj,
      user_lower,
      user_gap.c_str(),
      toc(exploration_stats_.start_time));
  } else {
    settings_.log.printf("New solution from primal heuristics. Objective %+.6e. Time %.2f\n",
                         compute_user_objective(original_lp_, obj),
                         toc(exploration_stats_.start_time));
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::report(
  char symbol, f_t obj, f_t lower_bound, i_t node_depth, i_t node_int_infeas, double work_time)
{
  update_user_bound(lower_bound);
  const i_t nodes_explored   = exploration_stats_.nodes_explored;
  const i_t nodes_unexplored = exploration_stats_.nodes_unexplored;
  const f_t user_obj         = compute_user_objective(original_lp_, obj);
  const f_t user_lower       = compute_user_objective(original_lp_, lower_bound);
  const f_t iters            = static_cast<f_t>(exploration_stats_.total_lp_iters);
  const f_t iter_node        = nodes_explored > 0 ? iters / nodes_explored : iters;
  const std::string user_gap = user_mip_gap<f_t>(user_obj, user_lower);
  if (work_time >= 0) {
    settings_.log.printf(
      "%c %10d   %10lu    %+13.6e    %+10.6e   %6d %6d   %7.1e     %s %9.2f %9.2f\n",
      symbol,
      nodes_explored,
      nodes_unexplored,
      user_obj,
      user_lower,
      node_int_infeas,
      node_depth,
      iter_node,
      user_gap.c_str(),
      work_time,
      toc(exploration_stats_.start_time));
  } else {
    settings_.log.printf("%c %10d   %10lu    %+13.6e    %+10.6e   %6d %6d   %7.1e     %s %9.2f\n",
                         symbol,
                         nodes_explored,
                         nodes_unexplored,
                         user_obj,
                         user_lower,
                         node_int_infeas,
                         node_depth,
                         iter_node,
                         user_gap.c_str(),
                         toc(exploration_stats_.start_time));
  }
}

template <typename i_t, typename f_t>
i_t branch_and_bound_t<i_t, f_t>::find_reduced_cost_fixings(f_t upper_bound,
                                                            std::vector<f_t>& lower_bounds,
                                                            std::vector<f_t>& upper_bounds)
{
  std::vector<f_t> reduced_costs = root_relax_soln_.z;
  lower_bounds                   = original_lp_.lower;
  upper_bounds                   = original_lp_.upper;
  std::vector<bool> bounds_changed(original_lp_.num_cols, false);
  const f_t root_obj    = compute_objective(original_lp_, root_relax_soln_.x);
  const f_t threshold   = 100.0 * settings_.integer_tol;
  const f_t weaken      = settings_.integer_tol;
  const f_t fixed_tol   = settings_.fixed_tol;
  i_t num_improved      = 0;
  i_t num_fixed         = 0;
  i_t num_cols_to_check = reduced_costs.size();  // Reduced costs will be smaller than the original
                                                 // problem because we have added slacks for cuts
  for (i_t j = 0; j < num_cols_to_check; j++) {
    if (std::isfinite(reduced_costs[j]) && std::abs(reduced_costs[j]) > threshold) {
      const f_t lower_j            = original_lp_.lower[j];
      const f_t upper_j            = original_lp_.upper[j];
      const f_t abs_gap            = upper_bound - root_obj;
      f_t reduced_cost_upper_bound = upper_j;
      f_t reduced_cost_lower_bound = lower_j;
      if (lower_j > -inf && reduced_costs[j] > 0) {
        const f_t new_upper_bound = lower_j + abs_gap / reduced_costs[j];
        reduced_cost_upper_bound  = var_types_[j] == variable_type_t::INTEGER
                                      ? std::floor(new_upper_bound + weaken)
                                      : new_upper_bound;
        if (reduced_cost_upper_bound < upper_j && var_types_[j] == variable_type_t::INTEGER) {
          num_improved++;
          upper_bounds[j]   = reduced_cost_upper_bound;
          bounds_changed[j] = true;
        }
      }
      if (upper_j < inf && reduced_costs[j] < 0) {
        const f_t new_lower_bound = upper_j + abs_gap / reduced_costs[j];
        reduced_cost_lower_bound  = var_types_[j] == variable_type_t::INTEGER
                                      ? std::ceil(new_lower_bound - weaken)
                                      : new_lower_bound;
        if (reduced_cost_lower_bound > lower_j && var_types_[j] == variable_type_t::INTEGER) {
          num_improved++;
          lower_bounds[j]   = reduced_cost_lower_bound;
          bounds_changed[j] = true;
        }
      }
      if (var_types_[j] == variable_type_t::INTEGER &&
          reduced_cost_upper_bound <= reduced_cost_lower_bound + fixed_tol) {
        num_fixed++;
      }
    }
  }

  if (num_fixed > 0 || num_improved > 0) {
    settings_.log.printf(
      "Reduced costs: Found %d improved bounds and %d fixed variables\n", num_improved, num_fixed);
  }
  return num_fixed;
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::update_user_bound(f_t lower_bound)
{
  if (user_bound_callback_ == nullptr) { return; }
  f_t user_lower = compute_user_objective(original_lp_, lower_bound);
  user_bound_callback_(user_lower);
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::set_new_solution(const std::vector<f_t>& solution)
{
  mutex_original_lp_.lock();
  if (solution.size() != original_problem_.num_cols) {
    settings_.log.printf(
      "Solution size mismatch %ld %d\n", solution.size(), original_problem_.num_cols);
  }
  std::vector<f_t> crushed_solution;
  crush_primal_solution<i_t, f_t>(
    original_problem_, original_lp_, solution, new_slacks_, crushed_solution);
  f_t obj = compute_objective(original_lp_, crushed_solution);
  mutex_original_lp_.unlock();
  bool is_feasible    = false;
  bool attempt_repair = false;
  mutex_upper_.lock();
  f_t current_upper_bound = upper_bound_;
  mutex_upper_.unlock();
  if (obj < current_upper_bound) {
    f_t primal_err;
    f_t bound_err;
    i_t num_fractional;
    mutex_original_lp_.lock();
    if (crushed_solution.size() != original_lp_.num_cols) {
      // original problem has been modified since the solution was crushed
      // we need to re-crush the solution
      crush_primal_solution<i_t, f_t>(
        original_problem_, original_lp_, solution, new_slacks_, crushed_solution);
    }
    is_feasible = check_guess(
      original_lp_, settings_, var_types_, crushed_solution, primal_err, bound_err, num_fractional);
    mutex_original_lp_.unlock();
    mutex_upper_.lock();
    if (is_feasible && obj < upper_bound_) {
      upper_bound_ = obj;
      incumbent_.set_incumbent_solution(obj, crushed_solution);
    } else {
      attempt_repair         = true;
      constexpr bool verbose = false;
      if (verbose) {
        settings_.log.printf(
          "Injected solution infeasible. Constraint error %e bound error %e integer infeasible "
          "%d\n",
          primal_err,
          bound_err,
          num_fractional);
      }
    }
    mutex_upper_.unlock();
  } else {
    settings_.log.debug("Solution objective not better than current upper_bound_. Not accepted.\n");
  }

  if (is_feasible) { report_heuristic(obj); }
  if (attempt_repair) {
    mutex_repair_.lock();
    repair_queue_.push_back(solution);
    mutex_repair_.unlock();
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::queue_external_solution_deterministic(
  const std::vector<f_t>& solution, double work_unit_ts)
{
  // In deterministic mode, queue the solution to be processed at the correct work unit timestamp
  // This ensures deterministic ordering of solution events

  if (solution.size() != original_problem_.num_cols) {
    settings_.log.printf(
      "Solution size mismatch %ld %d\n", solution.size(), original_problem_.num_cols);
    return;
  }

  mutex_original_lp_.lock();
  std::vector<f_t> crushed_solution;
  crush_primal_solution<i_t, f_t>(
    original_problem_, original_lp_, solution, new_slacks_, crushed_solution);
  f_t obj = compute_objective(original_lp_, crushed_solution);

  // Validate solution before queueing
  f_t primal_err;
  f_t bound_err;
  i_t num_fractional;
  bool is_feasible = check_guess(
    original_lp_, settings_, var_types_, crushed_solution, primal_err, bound_err, num_fractional);
  mutex_original_lp_.unlock();

  if (!is_feasible) {
    // Queue the uncrushed solution for repair; it will be crushed at
    // consumption time so that the crush reflects the current LP state
    // (which may have gained slack columns from cuts added after this point).
    mutex_repair_.lock();
    repair_queue_.push_back(solution);
    mutex_repair_.unlock();
    return;
  }

  // Queue the solution with its work unit timestamp
  mutex_heuristic_queue_.lock();
  heuristic_solution_queue_.push_back({obj, std::move(crushed_solution), 0, -1, 0, work_unit_ts});
  mutex_heuristic_queue_.unlock();
}

template <typename i_t, typename f_t>
bool branch_and_bound_t<i_t, f_t>::repair_solution(const std::vector<f_t>& edge_norms,
                                                   const std::vector<f_t>& potential_solution,
                                                   f_t& repaired_obj,
                                                   std::vector<f_t>& repaired_solution) const
{
  bool feasible = false;
  repaired_obj  = std::numeric_limits<f_t>::quiet_NaN();
  i_t n         = original_lp_.num_cols;
  assert(potential_solution.size() == n);

  lp_problem_t repair_lp = original_lp_;

  // Fix integer variables
  for (i_t j = 0; j < n; ++j) {
    if (var_types_[j] == variable_type_t::INTEGER) {
      const f_t fixed_val = std::round(potential_solution[j]);
      repair_lp.lower[j]  = fixed_val;
      repair_lp.upper[j]  = fixed_val;
    }
  }

  lp_solution_t<i_t, f_t> lp_solution(original_lp_.num_rows, original_lp_.num_cols);

  i_t iter                               = 0;
  f_t lp_start_time                      = tic();
  simplex_solver_settings_t lp_settings  = settings_;
  std::vector<variable_status_t> vstatus = root_vstatus_;
  lp_settings.set_log(false);
  lp_settings.inside_mip           = true;
  std::vector<f_t> leaf_edge_norms = edge_norms;
  // should probably set the cut off here lp_settings.cut_off
  dual::status_t lp_status = dual_phase2(
    2, 0, lp_start_time, repair_lp, lp_settings, vstatus, lp_solution, iter, leaf_edge_norms);
  repaired_solution = lp_solution.x;

  if (lp_status == dual::status_t::OPTIMAL) {
    f_t primal_error;
    f_t bound_error;
    i_t num_fractional;
    feasible               = check_guess(original_lp_,
                           settings_,
                           var_types_,
                           lp_solution.x,
                           primal_error,
                           bound_error,
                           num_fractional);
    repaired_obj           = compute_objective(original_lp_, repaired_solution);
    constexpr bool verbose = false;
    if (verbose) {
      settings_.log.printf(
        "After repair: feasible %d primal error %e bound error %e fractional %d. Objective %e\n",
        feasible,
        primal_error,
        bound_error,
        num_fractional,
        repaired_obj);
    }
  }

  return feasible;
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::repair_heuristic_solutions()
{
  raft::common::nvtx::range scope("BB::repair_heuristics");
  // Check if there are any solutions to repair
  std::vector<std::vector<f_t>> to_repair;
  mutex_repair_.lock();
  if (repair_queue_.size() > 0) {
    to_repair = repair_queue_;
    repair_queue_.clear();
  }
  mutex_repair_.unlock();

  if (to_repair.size() > 0) {
    settings_.log.debug("Attempting to repair %ld injected solutions\n", to_repair.size());
    for (const std::vector<f_t>& uncrushed_solution : to_repair) {
      std::vector<f_t> crushed_solution;
      crush_primal_solution<i_t, f_t>(
        original_problem_, original_lp_, uncrushed_solution, new_slacks_, crushed_solution);
      std::vector<f_t> repaired_solution;
      f_t repaired_obj;
      bool is_feasible =
        repair_solution(edge_norms_, crushed_solution, repaired_obj, repaired_solution);
      if (is_feasible) {
        mutex_upper_.lock();

        if (repaired_obj < upper_bound_) {
          upper_bound_ = repaired_obj;
          incumbent_.set_incumbent_solution(repaired_obj, repaired_solution);
          report_heuristic(repaired_obj);

          if (settings_.solution_callback != nullptr) {
            std::vector<f_t> original_x;
            uncrush_primal_solution(original_problem_, original_lp_, repaired_solution, original_x);
            settings_.solution_callback(original_x, repaired_obj);
          }
        }

        mutex_upper_.unlock();
      }
    }
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::set_solution_at_root(mip_solution_t<i_t, f_t>& solution,
                                                        const cut_info_t<i_t, f_t>& cut_info)
{
  mutex_upper_.lock();
  incumbent_.set_incumbent_solution(root_objective_, root_relax_soln_.x);
  upper_bound_ = root_objective_;
  mutex_upper_.unlock();

  print_cut_info(settings_, cut_info);

  // We should be done here
  uncrush_primal_solution(original_problem_, original_lp_, incumbent_.x, solution.x);
  solution.objective          = incumbent_.objective;
  solution.lower_bound        = root_objective_;
  solution.nodes_explored     = 0;
  solution.simplex_iterations = root_relax_soln_.iterations;
  settings_.log.printf("Optimal solution found at root node. Objective %.16e. Time %.2f.\n",
                       compute_user_objective(original_lp_, root_objective_),
                       toc(exploration_stats_.start_time));

  if (settings_.solution_callback != nullptr) {
    settings_.solution_callback(solution.x, solution.objective);
  }
  if (settings_.heuristic_preemption_callback != nullptr) {
    settings_.heuristic_preemption_callback();
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::set_final_solution(mip_solution_t<i_t, f_t>& solution,
                                                      f_t lower_bound)
{
  if (solver_status_ == mip_status_t::NUMERICAL) {
    settings_.log.printf("Numerical issue encountered. Stopping the solver...\n");
  }

  if (solver_status_ == mip_status_t::TIME_LIMIT) {
    settings_.log.printf("Time limit reached. Stopping the solver...\n");
  }
  if (solver_status_ == mip_status_t::WORK_LIMIT) {
    settings_.log.printf("Work limit reached. Stopping the solver...\n");
  }
  if (solver_status_ == mip_status_t::NODE_LIMIT) {
    settings_.log.printf("Node limit reached. Stopping the solver...\n");
  }

  if (settings_.heuristic_preemption_callback != nullptr) {
    settings_.heuristic_preemption_callback();
  }

  f_t gap              = upper_bound_ - lower_bound;
  f_t obj              = compute_user_objective(original_lp_, upper_bound_.load());
  f_t user_bound       = compute_user_objective(original_lp_, lower_bound);
  f_t gap_rel          = user_relative_gap(original_lp_, upper_bound_.load(), lower_bound);
  bool is_maximization = original_lp_.obj_scale < 0.0;

  settings_.log.printf("Explored %d nodes in %.2fs.\n",
                       exploration_stats_.nodes_explored,
                       toc(exploration_stats_.start_time));
  settings_.log.printf("Absolute Gap %e Objective %.16e %s Bound %.16e\n",
                       gap,
                       obj,
                       is_maximization ? "Upper" : "Lower",
                       user_bound);

  if (gap <= settings_.absolute_mip_gap_tol || gap_rel <= settings_.relative_mip_gap_tol) {
    solver_status_ = mip_status_t::OPTIMAL;
#ifdef CHECK_CUTS_AGAINST_SAVED_SOLUTION
    if (settings_.sub_mip == 0) { write_solution_for_cut_verification(original_lp_, incumbent_.x); }
#endif
    if (gap > 0 && gap <= settings_.absolute_mip_gap_tol) {
      settings_.log.printf("Optimal solution found within absolute MIP gap tolerance (%.1e)\n",
                           settings_.absolute_mip_gap_tol);
    } else if (gap > 0 && gap_rel <= settings_.relative_mip_gap_tol) {
      settings_.log.printf("Optimal solution found within relative MIP gap tolerance (%.1e)\n",
                           settings_.relative_mip_gap_tol);
    } else {
      settings_.log.printf("Optimal solution found.\n");
    }
    if (settings_.heuristic_preemption_callback != nullptr) {
      settings_.heuristic_preemption_callback();
    }
  }

  if (solver_status_ == mip_status_t::UNSET) {
    if (exploration_stats_.nodes_explored > 0 && exploration_stats_.nodes_unexplored == 0 &&
        upper_bound_ == inf) {
      settings_.log.printf("Integer infeasible.\n");
      solver_status_ = mip_status_t::INFEASIBLE;
      if (settings_.heuristic_preemption_callback != nullptr) {
        settings_.heuristic_preemption_callback();
      }
    }
  }

  if (upper_bound_ != inf) {
    assert(incumbent_.has_incumbent);
    uncrush_primal_solution(original_problem_, original_lp_, incumbent_.x, solution.x);
  }
  solution.objective          = incumbent_.objective;
  solution.lower_bound        = lower_bound;
  solution.nodes_explored     = exploration_stats_.nodes_explored;
  solution.simplex_iterations = exploration_stats_.total_lp_iters;
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::add_feasible_solution(f_t leaf_objective,
                                                         const std::vector<f_t>& leaf_solution,
                                                         i_t leaf_depth,
                                                         search_strategy_t thread_type)
{
  bool send_solution = false;

  settings_.log.debug("%c found a feasible solution with obj=%.10e.\n",
                      feasible_solution_symbol(thread_type),
                      compute_user_objective(original_lp_, leaf_objective));

  mutex_upper_.lock();
  if (leaf_objective < upper_bound_) {
    incumbent_.set_incumbent_solution(leaf_objective, leaf_solution);
    upper_bound_ = leaf_objective;
    report(feasible_solution_symbol(thread_type), leaf_objective, get_lower_bound(), leaf_depth, 0);
    send_solution = true;
  }

  if (send_solution && settings_.solution_callback != nullptr) {
    std::vector<f_t> original_x;
    uncrush_primal_solution(original_problem_, original_lp_, incumbent_.x, original_x);
    settings_.solution_callback(original_x, upper_bound_);
  }
  mutex_upper_.unlock();
}

// Martin's criteria for the preferred rounding direction (see [1])
// [1] A. Martin, “Integer Programs with Block Structure,”
// Technische Universit¨at Berlin, Berlin, 1999. Accessed: Aug. 08, 2025.
// [Online]. Available: https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/391
template <typename f_t>
rounding_direction_t martin_criteria(f_t val, f_t root_val)
{
  const f_t down_val  = std::floor(root_val);
  const f_t up_val    = std::ceil(root_val);
  const f_t down_dist = val - down_val;
  const f_t up_dist   = up_val - val;
  constexpr f_t eps   = 1e-6;

  if (down_dist < up_dist + eps) {
    return rounding_direction_t::DOWN;

  } else {
    return rounding_direction_t::UP;
  }
}

template <typename i_t, typename f_t>
branch_variable_t<i_t> branch_and_bound_t<i_t, f_t>::variable_selection(
  mip_node_t<i_t, f_t>* node_ptr,
  const std::vector<i_t>& fractional,
  branch_and_bound_worker_t<i_t, f_t>* worker)
{
  logger_t log;
  log.log                        = false;
  i_t branch_var                 = -1;
  rounding_direction_t round_dir = rounding_direction_t::NONE;
  std::vector<f_t> current_incumbent;
  std::vector<f_t>& solution = worker->leaf_solution.x;

  switch (worker->search_strategy) {
    case search_strategy_t::BEST_FIRST:

      if (settings_.reliability_branching != 0) {
        branch_var = pc_.reliable_variable_selection(node_ptr,
                                                     fractional,
                                                     solution,
                                                     settings_,
                                                     var_types_,
                                                     worker,
                                                     exploration_stats_,
                                                     upper_bound_,
                                                     worker_pool_.num_idle_workers(),
                                                     log);
      } else {
        branch_var = pc_.variable_selection(fractional, solution, log);
      }

      round_dir = martin_criteria(solution[branch_var], root_relax_soln_.x[branch_var]);

      return {branch_var, round_dir};

    case search_strategy_t::COEFFICIENT_DIVING:
      return coefficient_diving(
        original_lp_, fractional, solution, var_up_locks_, var_down_locks_, log);

    case search_strategy_t::LINE_SEARCH_DIVING:
      return line_search_diving(fractional, solution, root_relax_soln_.x, log);

    case search_strategy_t::PSEUDOCOST_DIVING:
      return pseudocost_diving(pc_, fractional, solution, root_relax_soln_.x, log);

    case search_strategy_t::GUIDED_DIVING:
      mutex_upper_.lock();
      current_incumbent = incumbent_.x;
      mutex_upper_.unlock();
      return guided_diving(pc_, fractional, solution, current_incumbent, log);

    default:
      log.debug("Unknown variable selection method: %d\n", worker->search_strategy);
      return {-1, rounding_direction_t::NONE};
  }
}

// ============================================================================
// Policies for update_tree
// These allow sharing the tree update logic between the default and deterministic codepaths
// ============================================================================

// Compiler is able to devirtualize the policy objects in update_tree_impl.
// This is for self-documenting purposes
template <typename i_t, typename f_t>
struct tree_update_policy_t {
  virtual ~tree_update_policy_t()                                                  = default;
  virtual f_t upper_bound() const                                                  = 0;
  virtual void update_pseudo_costs(mip_node_t<i_t, f_t>* node, f_t obj)            = 0;
  virtual void handle_integer_solution(mip_node_t<i_t, f_t>* node,
                                       f_t obj,
                                       const std::vector<f_t>& x)                  = 0;
  virtual branch_variable_t<i_t> select_branch_variable(mip_node_t<i_t, f_t>* node,
                                                        const std::vector<i_t>& fractional,
                                                        const std::vector<f_t>& x) = 0;
  virtual void update_objective_estimate(mip_node_t<i_t, f_t>* node,
                                         const std::vector<i_t>& fractional,
                                         const std::vector<f_t>& x)                = 0;
  virtual void on_node_completed(mip_node_t<i_t, f_t>* node,
                                 node_status_t status,
                                 rounding_direction_t dir)                         = 0;
  virtual void on_numerical_issue(mip_node_t<i_t, f_t>*)                           = 0;
  virtual void graphviz(search_tree_t<i_t, f_t>&, mip_node_t<i_t, f_t>*, const char*, f_t) = 0;
  virtual void on_optimal_callback(const std::vector<f_t>&, f_t)                           = 0;
};

template <typename i_t, typename f_t>
struct nondeterministic_policy_t : tree_update_policy_t<i_t, f_t> {
  branch_and_bound_t<i_t, f_t>& bnb;
  branch_and_bound_worker_t<i_t, f_t>* worker;
  logger_t& log;

  nondeterministic_policy_t(branch_and_bound_t<i_t, f_t>& bnb,
                            branch_and_bound_worker_t<i_t, f_t>* worker,
                            logger_t& log)
    : bnb(bnb), worker(worker), log(log)
  {
  }

  f_t upper_bound() const override { return bnb.upper_bound_.load(); }

  void update_pseudo_costs(mip_node_t<i_t, f_t>* node, f_t leaf_obj) override
  {
    bnb.pc_.update_pseudo_costs(node, leaf_obj);
  }

  void handle_integer_solution(mip_node_t<i_t, f_t>* node,
                               f_t obj,
                               const std::vector<f_t>& x) override
  {
    bnb.add_feasible_solution(obj, x, node->depth, worker->search_strategy);
  }

  branch_variable_t<i_t> select_branch_variable(mip_node_t<i_t, f_t>* node,
                                                const std::vector<i_t>& fractional,
                                                const std::vector<f_t>&) override
  {
    return bnb.variable_selection(node, fractional, worker);
  }

  void update_objective_estimate(mip_node_t<i_t, f_t>* node,
                                 const std::vector<i_t>& fractional,
                                 const std::vector<f_t>& x) override
  {
    if (worker->search_strategy == search_strategy_t::BEST_FIRST) {
      logger_t pc_log;
      pc_log.log               = false;
      node->objective_estimate = bnb.pc_.obj_estimate(fractional, x, node->lower_bound, pc_log);
    }
  }

  void on_numerical_issue(mip_node_t<i_t, f_t>* node) override
  {
    if (worker->search_strategy == search_strategy_t::BEST_FIRST) {
      fetch_min(bnb.lower_bound_ceiling_, node->lower_bound);
      log.printf("LP returned numerical issue on node %d. Best bound set to %+10.6e.\n",
                 node->node_id,
                 compute_user_objective(bnb.original_lp_, bnb.lower_bound_ceiling_.load()));
    }
  }

  void graphviz(search_tree_t<i_t, f_t>& tree,
                mip_node_t<i_t, f_t>* node,
                const char* label,
                f_t value) override
  {
    tree.graphviz_node(log, node, label, value);
  }

  void on_optimal_callback(const std::vector<f_t>& x, f_t objective) override
  {
    if (worker->search_strategy == search_strategy_t::BEST_FIRST &&
        bnb.settings_.node_processed_callback != nullptr) {
      std::vector<f_t> original_x;
      uncrush_primal_solution(bnb.original_problem_, bnb.original_lp_, x, original_x);
      bnb.settings_.node_processed_callback(original_x, objective);
    }
  }

  void on_node_completed(mip_node_t<i_t, f_t>*, node_status_t, rounding_direction_t) override {}
};

template <typename i_t, typename f_t, typename WorkerT>
struct deterministic_policy_base_t : tree_update_policy_t<i_t, f_t> {
  branch_and_bound_t<i_t, f_t>& bnb;
  WorkerT& worker;

  deterministic_policy_base_t(branch_and_bound_t<i_t, f_t>& bnb, WorkerT& worker)
    : bnb(bnb), worker(worker)
  {
  }

  f_t upper_bound() const override { return worker.local_upper_bound; }

  void update_pseudo_costs(mip_node_t<i_t, f_t>* node, f_t leaf_obj) override
  {
    if (node->branch_var < 0) return;
    f_t change = std::max(leaf_obj - node->lower_bound, f_t(0));
    f_t frac   = node->branch_dir == rounding_direction_t::DOWN
                   ? node->fractional_val - std::floor(node->fractional_val)
                   : std::ceil(node->fractional_val) - node->fractional_val;
    if (frac > 1e-10) {
      worker.pc_snapshot.queue_update(
        node->branch_var, node->branch_dir, change / frac, worker.clock, worker.worker_id);
    }
  }

  void on_numerical_issue(mip_node_t<i_t, f_t>*) override {}
  void graphviz(search_tree_t<i_t, f_t>&, mip_node_t<i_t, f_t>*, const char*, f_t) override {}
  void on_optimal_callback(const std::vector<f_t>&, f_t) override {}
};

template <typename i_t, typename f_t>
struct deterministic_bfs_policy_t
  : deterministic_policy_base_t<i_t, f_t, deterministic_bfs_worker_t<i_t, f_t>> {
  using base = deterministic_policy_base_t<i_t, f_t, deterministic_bfs_worker_t<i_t, f_t>>;
  using base::base;

  void handle_integer_solution(mip_node_t<i_t, f_t>* node,
                               f_t obj,
                               const std::vector<f_t>& x) override
  {
    if (obj < this->worker.local_upper_bound) {
      this->worker.local_upper_bound = obj;
      this->worker.integer_solutions.push_back(
        {obj, x, node->depth, this->worker.worker_id, this->worker.next_solution_seq++});
    }
  }

  branch_variable_t<i_t> select_branch_variable(mip_node_t<i_t, f_t>*,
                                                const std::vector<i_t>& fractional,
                                                const std::vector<f_t>& x) override
  {
    i_t var  = this->worker.pc_snapshot.variable_selection(fractional, x);
    auto dir = martin_criteria(x[var], this->bnb.root_relax_soln_.x[var]);
    return {var, dir};
  }

  void update_objective_estimate(mip_node_t<i_t, f_t>* node,
                                 const std::vector<i_t>& fractional,
                                 const std::vector<f_t>& x) override
  {
    node->objective_estimate =
      this->worker.pc_snapshot.obj_estimate(fractional, x, node->lower_bound);
  }

  void on_node_completed(mip_node_t<i_t, f_t>* node,
                         node_status_t status,
                         rounding_direction_t dir) override
  {
    switch (status) {
      case node_status_t::INFEASIBLE: this->worker.record_infeasible(node); break;
      case node_status_t::FATHOMED: this->worker.record_fathomed(node, node->lower_bound); break;
      case node_status_t::INTEGER_FEASIBLE:
        this->worker.record_integer_solution(node, node->lower_bound);
        break;
      case node_status_t::HAS_CHILDREN:
        this->worker.record_branched(node,
                                     node->get_down_child()->node_id,
                                     node->get_up_child()->node_id,
                                     node->branch_var,
                                     node->fractional_val);
        this->bnb.exploration_stats_.nodes_unexplored += 2;
        this->worker.enqueue_children_for_plunge(node->get_down_child(), node->get_up_child(), dir);
        break;
      case node_status_t::NUMERICAL: this->worker.record_numerical(node); break;
      default: break;
    }
    if (status != node_status_t::HAS_CHILDREN) { this->worker.recompute_bounds_and_basis = true; }
  }

  void on_numerical_issue(mip_node_t<i_t, f_t>* node) override
  {
    this->worker.local_lower_bound_ceiling =
      std::min<f_t>(node->lower_bound, this->worker.local_lower_bound_ceiling);
  }
};

template <typename i_t, typename f_t>
struct deterministic_diving_policy_t
  : deterministic_policy_base_t<i_t, f_t, deterministic_diving_worker_t<i_t, f_t>> {
  using base = deterministic_policy_base_t<i_t, f_t, deterministic_diving_worker_t<i_t, f_t>>;

  std::deque<mip_node_t<i_t, f_t>*>& stack;
  i_t max_backtrack_depth;

  deterministic_diving_policy_t(branch_and_bound_t<i_t, f_t>& bnb,
                                deterministic_diving_worker_t<i_t, f_t>& worker,
                                std::deque<mip_node_t<i_t, f_t>*>& stack,
                                i_t max_backtrack_depth)
    : base(bnb, worker), stack(stack), max_backtrack_depth(max_backtrack_depth)
  {
  }

  void handle_integer_solution(mip_node_t<i_t, f_t>* node,
                               f_t obj,
                               const std::vector<f_t>& x) override
  {
    if (obj < this->worker.local_upper_bound) {
      this->worker.local_upper_bound = obj;
      this->worker.queue_integer_solution(obj, x, node->depth);
    }
  }

  branch_variable_t<i_t> select_branch_variable(mip_node_t<i_t, f_t>*,
                                                const std::vector<i_t>& fractional,
                                                const std::vector<f_t>& x) override
  {
    switch (this->worker.diving_type) {
      case search_strategy_t::PSEUDOCOST_DIVING:
        return this->worker.variable_selection_from_snapshot(fractional, x);

      case search_strategy_t::LINE_SEARCH_DIVING:
        if (this->worker.root_solution) {
          logger_t log;
          log.log = false;
          return line_search_diving<i_t, f_t>(fractional, x, *this->worker.root_solution, log);
        }
        return this->worker.variable_selection_from_snapshot(fractional, x);

      case search_strategy_t::GUIDED_DIVING:
        return this->worker.guided_variable_selection(fractional, x);

      case search_strategy_t::COEFFICIENT_DIVING: {
        logger_t log;
        log.log = false;
        return coefficient_diving<i_t, f_t>(this->bnb.original_lp_,
                                            fractional,
                                            x,
                                            this->bnb.var_up_locks_,
                                            this->bnb.var_down_locks_,
                                            log);
      }

      default: return this->worker.variable_selection_from_snapshot(fractional, x);
    }
  }

  void update_objective_estimate(mip_node_t<i_t, f_t>* node,
                                 const std::vector<i_t>& fractional,
                                 const std::vector<f_t>& x) override
  {
  }

  void on_node_completed(mip_node_t<i_t, f_t>* node,
                         node_status_t status,
                         rounding_direction_t dir) override
  {
    if (status == node_status_t::HAS_CHILDREN) {
      if (dir == rounding_direction_t::UP) {
        stack.push_front(node->get_down_child());
        stack.push_front(node->get_up_child());
      } else {
        stack.push_front(node->get_up_child());
        stack.push_front(node->get_down_child());
      }
      if (stack.size() > 1 && stack.front()->depth - stack.back()->depth > max_backtrack_depth) {
        stack.pop_back();
      }
      this->worker.recompute_bounds_and_basis = false;
    } else {
      this->worker.recompute_bounds_and_basis = true;
    }
  }
};

template <typename i_t, typename f_t>
template <typename WorkerT, typename Policy>
std::pair<node_status_t, rounding_direction_t> branch_and_bound_t<i_t, f_t>::update_tree_impl(
  mip_node_t<i_t, f_t>* node_ptr,
  search_tree_t<i_t, f_t>& search_tree,
  WorkerT* worker,
  dual::status_t lp_status,
  Policy& policy)
{
  constexpr f_t inf                      = std::numeric_limits<f_t>::infinity();
  const f_t abs_fathom_tol               = settings_.absolute_mip_gap_tol / 10;
  lp_problem_t<i_t, f_t>& leaf_problem   = worker->leaf_problem;
  lp_solution_t<i_t, f_t>& leaf_solution = worker->leaf_solution;
  const f_t upper_bound                  = policy.upper_bound();
  node_status_t status                   = node_status_t::PENDING;
  rounding_direction_t round_dir         = rounding_direction_t::NONE;

  if (lp_status == dual::status_t::DUAL_UNBOUNDED) {
    node_ptr->lower_bound = inf;
    policy.graphviz(search_tree, node_ptr, "infeasible", 0.0);
    search_tree.update(node_ptr, node_status_t::INFEASIBLE);
    status = node_status_t::INFEASIBLE;

  } else if (lp_status == dual::status_t::CUTOFF) {
    f_t leaf_obj          = compute_objective(leaf_problem, leaf_solution.x);
    node_ptr->lower_bound = upper_bound;
    policy.graphviz(search_tree, node_ptr, "cut off", leaf_obj);
    search_tree.update(node_ptr, node_status_t::FATHOMED);
    status = node_status_t::FATHOMED;

  } else if (lp_status == dual::status_t::OPTIMAL) {
    std::vector<i_t> leaf_fractional;
    i_t num_frac = fractional_variables(settings_, leaf_solution.x, var_types_, leaf_fractional);

#ifdef DEBUG_FRACTIONAL_FIXED
    for (i_t j : leaf_fractional) {
      if (leaf_problem.lower[j] == leaf_problem.upper[j]) {
        printf(
          "Node %d: Fixed variable %d has a fractional value %e. Lower %e upper %e. Variable "
          "status %d\n",
          node_ptr->node_id,
          j,
          leaf_solution.x[j],
          leaf_problem.lower[j],
          leaf_problem.upper[j],
          node_ptr->vstatus[j]);
      }
    }
#endif

    f_t leaf_obj = compute_objective(leaf_problem, leaf_solution.x);

    policy.graphviz(search_tree, node_ptr, "lower bound", leaf_obj);
    policy.update_pseudo_costs(node_ptr, leaf_obj);
    node_ptr->lower_bound = leaf_obj;
    if (original_lp_.objective_is_integral) {
      node_ptr->lower_bound = std::ceil(leaf_obj - settings_.integer_tol);
    }
    policy.on_optimal_callback(leaf_solution.x, leaf_obj);

    if (num_frac == 0) {
      policy.handle_integer_solution(node_ptr, leaf_obj, leaf_solution.x);
      policy.graphviz(search_tree, node_ptr, "integer feasible", leaf_obj);
      search_tree.update(node_ptr, node_status_t::INTEGER_FEASIBLE);
      status = node_status_t::INTEGER_FEASIBLE;

    } else if (leaf_obj <= upper_bound + abs_fathom_tol) {
      auto [branch_var, dir] =
        policy.select_branch_variable(node_ptr, leaf_fractional, leaf_solution.x);
      round_dir = dir;

      assert(node_ptr->vstatus.size() == leaf_problem.num_cols);
      assert(branch_var >= 0);
      assert(dir != rounding_direction_t::NONE);

      policy.update_objective_estimate(node_ptr, leaf_fractional, leaf_solution.x);

      logger_t log;
      log.log = false;
      search_tree.branch(node_ptr,
                         branch_var,
                         leaf_solution.x[branch_var],
                         num_frac,
                         node_ptr->vstatus,
                         leaf_problem,
                         log);
      search_tree.update(node_ptr, node_status_t::HAS_CHILDREN);
      status = node_status_t::HAS_CHILDREN;

    } else {
      policy.graphviz(search_tree, node_ptr, "fathomed", leaf_obj);
      search_tree.update(node_ptr, node_status_t::FATHOMED);
      status = node_status_t::FATHOMED;
    }
  } else if (lp_status == dual::status_t::TIME_LIMIT) {
    policy.graphviz(search_tree, node_ptr, "timeout", 0.0);
    status = node_status_t::PENDING;
  } else if (lp_status == dual::status_t::WORK_LIMIT) {
    policy.graphviz(search_tree, node_ptr, "work limit", 0.0);
    status = node_status_t::PENDING;
  } else {
    policy.on_numerical_issue(node_ptr);
    policy.graphviz(search_tree, node_ptr, "numerical", 0.0);
    search_tree.update(node_ptr, node_status_t::NUMERICAL);
    status = node_status_t::NUMERICAL;
  }

  policy.on_node_completed(node_ptr, status, round_dir);
  return {status, round_dir};
}

template <typename i_t, typename f_t>
std::pair<node_status_t, rounding_direction_t> branch_and_bound_t<i_t, f_t>::update_tree(
  mip_node_t<i_t, f_t>* node_ptr,
  search_tree_t<i_t, f_t>& search_tree,
  branch_and_bound_worker_t<i_t, f_t>* worker,
  dual::status_t lp_status,
  logger_t& log)
{
  nondeterministic_policy_t<i_t, f_t> policy{*this, worker, log};
  return update_tree_impl(node_ptr, search_tree, worker, lp_status, policy);
}

template <typename i_t, typename f_t>
dual::status_t branch_and_bound_t<i_t, f_t>::solve_node_lp(
  mip_node_t<i_t, f_t>* node_ptr,
  branch_and_bound_worker_t<i_t, f_t>* worker,
  branch_and_bound_stats_t<i_t, f_t>& stats,
  logger_t& log)
{
  raft::common::nvtx::range scope("BB::solve_node");
#ifdef DEBUG_BRANCHING
  i_t num_integer_variables = 0;
  for (i_t j = 0; j < original_lp_.num_cols; j++) {
    if (var_types_[j] == variable_type_t::INTEGER) { num_integer_variables++; }
  }
  if (node_ptr->depth > num_integer_variables) {
    std::vector<i_t> branched_variables(original_lp_.num_cols, 0);
    std::vector<f_t> branched_lower(original_lp_.num_cols, std::numeric_limits<f_t>::quiet_NaN());
    std::vector<f_t> branched_upper(original_lp_.num_cols, std::numeric_limits<f_t>::quiet_NaN());
    mip_node_t<i_t, f_t>* parent = node_ptr->parent;
    while (parent != nullptr) {
      if (original_lp_.lower[parent->branch_var] != 0.0 ||
          original_lp_.upper[parent->branch_var] != 1.0) {
        break;
      }
      if (branched_variables[parent->branch_var] == 1) {
        printf(
          "Variable %d already branched. Previous lower %e upper %e. Current lower %e upper %e.\n",
          parent->branch_var,
          branched_lower[parent->branch_var],
          branched_upper[parent->branch_var],
          parent->branch_var_lower,
          parent->branch_var_upper);
      }
      branched_variables[parent->branch_var] = 1;
      branched_lower[parent->branch_var]     = parent->branch_var_lower;
      branched_upper[parent->branch_var]     = parent->branch_var_upper;
      parent                                 = parent->parent;
    }
    if (parent == nullptr) {
      printf("Depth %d > num_integer_variables %d\n", node_ptr->depth, num_integer_variables);
    }
  }
#endif

  std::vector<variable_status_t>& leaf_vstatus = node_ptr->vstatus;
  assert(leaf_vstatus.size() == worker->leaf_problem.num_cols);

  simplex_solver_settings_t lp_settings = settings_;
  lp_settings.set_log(false);
  if (original_lp_.objective_is_integral) {
    lp_settings.cut_off = std::ceil(upper_bound_ - settings_.integer_tol) - 1 + settings_.dual_tol;
  } else {
    lp_settings.cut_off = upper_bound_ + settings_.dual_tol;
  }
  lp_settings.inside_mip    = 2;
  lp_settings.time_limit    = settings_.time_limit - toc(exploration_stats_.start_time);
  lp_settings.scale_columns = false;

  if (worker->search_strategy != search_strategy_t::BEST_FIRST) {
    int64_t bnb_lp_iters        = exploration_stats_.total_lp_iters;
    f_t factor                  = settings_.diving_settings.iteration_limit_factor;
    int64_t max_iter            = factor * bnb_lp_iters;
    lp_settings.iteration_limit = max_iter - stats.total_lp_iters;
    if (lp_settings.iteration_limit <= 0) { return dual::status_t::ITERATION_LIMIT; }
  }

#ifdef LOG_NODE_SIMPLEX
  lp_settings.set_log(true);
  std::stringstream ss;
  ss << "simplex-" << std::this_thread::get_id() << ".log";
  std::string logname;
  ss >> logname;
  lp_settings.log.set_log_file(logname, "a");
  lp_settings.log.log_to_console = false;
  lp_settings.log.printf(
    "%scurrent node: id = %d, depth = %d, branch var = %d, branch dir = %s, fractional val = "
    "%f, variable lower bound = %f, variable upper bound = %f, branch vstatus = %d\n\n",
    settings_.log.log_prefix.c_str(),
    node_ptr->node_id,
    node_ptr->depth,
    node_ptr->branch_var,
    node_ptr->branch_dir == rounding_direction_t::DOWN ? "DOWN" : "UP",
    node_ptr->fractional_val,
    node_ptr->branch_var_lower,
    node_ptr->branch_var_upper,
    node_ptr->vstatus[node_ptr->branch_var]);
#endif

  bool feasible            = worker->set_lp_variable_bounds(node_ptr, settings_);
  dual::status_t lp_status = dual::status_t::DUAL_UNBOUNDED;
  worker->leaf_edge_norms  = edge_norms_;

  if (feasible) {
    i_t node_iter     = 0;
    f_t lp_start_time = tic();

    lp_status = dual_phase2_with_advanced_basis(2,
                                                0,
                                                worker->recompute_basis,
                                                lp_start_time,
                                                worker->leaf_problem,
                                                lp_settings,
                                                leaf_vstatus,
                                                worker->basis_factors,
                                                worker->basic_list,
                                                worker->nonbasic_list,
                                                worker->leaf_solution,
                                                node_iter,
                                                worker->leaf_edge_norms);

    if (lp_status == dual::status_t::NUMERICAL) {
      log.debug("Numerical issue node %d. Resolving from scratch.\n", node_ptr->node_id);
      lp_status_t second_status = solve_linear_program_with_advanced_basis(worker->leaf_problem,
                                                                           lp_start_time,
                                                                           lp_settings,
                                                                           worker->leaf_solution,
                                                                           worker->basis_factors,
                                                                           worker->basic_list,
                                                                           worker->nonbasic_list,
                                                                           leaf_vstatus,
                                                                           worker->leaf_edge_norms);

      lp_status = convert_lp_status_to_dual_status(second_status);
    }

    stats.total_lp_solve_time += toc(lp_start_time);
    stats.total_lp_iters += node_iter;
  }

#ifdef LOG_NODE_SIMPLEX
  lp_settings.log.printf("\nLP status: %d\n\n", lp_status);
#endif

  return lp_status;
}
template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::plunge_with(branch_and_bound_worker_t<i_t, f_t>* worker)
{
  std::deque<mip_node_t<i_t, f_t>*> stack;
  stack.push_front(worker->start_node);
  worker->recompute_basis  = true;
  worker->recompute_bounds = true;

  while (stack.size() > 0 && solver_status_ == mip_status_t::UNSET) {
    mip_node_t<i_t, f_t>* node_ptr = stack.front();
    stack.pop_front();

    f_t lower_bound = node_ptr->lower_bound;
    f_t upper_bound = upper_bound_;
    f_t rel_gap     = user_relative_gap(original_lp_, upper_bound, lower_bound);

    // This is based on three assumptions:
    // - The stack only contains sibling nodes, i.e., the current node and it sibling (if
    // applicable)
    // - The current node and its siblings uses the lower bound of the parent before solving the LP
    // relaxation
    // - The lower bound of the parent is lower or equal to its children
    worker->lower_bound = lower_bound;

    if (lower_bound > upper_bound || rel_gap < settings_.relative_mip_gap_tol) {
      search_tree_.graphviz_node(settings_.log, node_ptr, "cutoff", node_ptr->lower_bound);
      search_tree_.update(node_ptr, node_status_t::FATHOMED);
      worker->recompute_basis  = true;
      worker->recompute_bounds = true;
      --exploration_stats_.nodes_unexplored;
      continue;
    }

    if (toc(exploration_stats_.start_time) > settings_.time_limit) {
      solver_status_ = mip_status_t::TIME_LIMIT;
      break;
    }

    if (exploration_stats_.nodes_explored >= settings_.node_limit) {
      solver_status_ = mip_status_t::NODE_LIMIT;
      break;
    }

    dual::status_t lp_status = solve_node_lp(node_ptr, worker, exploration_stats_, settings_.log);

    if (lp_status == dual::status_t::TIME_LIMIT) {
      solver_status_ = mip_status_t::TIME_LIMIT;
      break;
    } else if (lp_status == dual::status_t::ITERATION_LIMIT) {
      break;
    }

    ++exploration_stats_.nodes_since_last_log;
    ++exploration_stats_.nodes_explored;
    --exploration_stats_.nodes_unexplored;

    auto [node_status, round_dir] =
      update_tree(node_ptr, search_tree_, worker, lp_status, settings_.log);

    worker->recompute_basis  = node_status != node_status_t::HAS_CHILDREN;
    worker->recompute_bounds = node_status != node_status_t::HAS_CHILDREN;

    if (node_status == node_status_t::HAS_CHILDREN) {
      // The stack should only contain the children of the current parent.
      // If the stack size is greater than 0,
      // we pop the current node from the stack and place it in the global heap,
      // since we are about to add the two children to the stack
      if (stack.size() > 0) {
        mip_node_t<i_t, f_t>* node = stack.back();
        stack.pop_back();
        node_queue_.push(node);
      }

      exploration_stats_.nodes_unexplored += 2;

      if (round_dir == rounding_direction_t::UP) {
        if (node_queue_.best_first_queue_size() < min_node_queue_size_) {
          node_queue_.push(node_ptr->get_down_child());
        } else {
          stack.push_front(node_ptr->get_down_child());
        }

        stack.push_front(node_ptr->get_up_child());
      } else {
        if (node_queue_.best_first_queue_size() < min_node_queue_size_) {
          node_queue_.push(node_ptr->get_up_child());
        } else {
          stack.push_front(node_ptr->get_up_child());
        }

        stack.push_front(node_ptr->get_down_child());
      }
    }
  }

  if (settings_.num_threads > 1) {
    worker_pool_.return_worker_to_pool(worker);
    active_workers_per_strategy_[BEST_FIRST]--;
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::dive_with(branch_and_bound_worker_t<i_t, f_t>* worker)
{
  raft::common::nvtx::range scope("BB::diving_thread");
  logger_t log;
  log.log = false;

  search_strategy_t search_strategy = worker->search_strategy;
  const i_t diving_node_limit       = settings_.diving_settings.node_limit;
  const i_t diving_backtrack_limit  = settings_.diving_settings.backtrack_limit;

  worker->recompute_basis  = true;
  worker->recompute_bounds = true;

  search_tree_t<i_t, f_t> dive_tree(std::move(*worker->start_node));
  std::deque<mip_node_t<i_t, f_t>*> stack;
  stack.push_front(&dive_tree.root);

  branch_and_bound_stats_t<i_t, f_t> dive_stats;
  dive_stats.total_lp_iters      = 0;
  dive_stats.total_lp_solve_time = 0;
  dive_stats.nodes_explored      = 0;
  dive_stats.nodes_unexplored    = 1;

  while (stack.size() > 0 && solver_status_ == mip_status_t::UNSET && is_running_) {
    mip_node_t<i_t, f_t>* node_ptr = stack.front();
    stack.pop_front();

    f_t lower_bound     = node_ptr->lower_bound;
    f_t upper_bound     = upper_bound_;
    f_t rel_gap         = user_relative_gap(original_lp_, upper_bound, lower_bound);
    worker->lower_bound = lower_bound;

    if (node_ptr->lower_bound > upper_bound || rel_gap < settings_.relative_mip_gap_tol) {
      worker->recompute_basis  = true;
      worker->recompute_bounds = true;
      continue;
    }

    if (toc(exploration_stats_.start_time) > settings_.time_limit) { break; }
    if (dive_stats.nodes_explored > diving_node_limit) { break; }

    dual::status_t lp_status = solve_node_lp(node_ptr, worker, dive_stats, log);

    if (lp_status == dual::status_t::TIME_LIMIT) {
      solver_status_ = mip_status_t::TIME_LIMIT;
      break;
    } else if (lp_status == dual::status_t::ITERATION_LIMIT) {
      break;
    }

    ++dive_stats.nodes_explored;

    auto [node_status, round_dir] = update_tree(node_ptr, dive_tree, worker, lp_status, log);

    worker->recompute_basis  = node_status != node_status_t::HAS_CHILDREN;
    worker->recompute_bounds = node_status != node_status_t::HAS_CHILDREN;

    if (node_status == node_status_t::HAS_CHILDREN) {
      if (round_dir == rounding_direction_t::UP) {
        stack.push_front(node_ptr->get_down_child());
        stack.push_front(node_ptr->get_up_child());
      } else {
        stack.push_front(node_ptr->get_up_child());
        stack.push_front(node_ptr->get_down_child());
      }
    }

    // Remove nodes that we no longer can backtrack to (i.e., from the current node, we can only
    // backtrack to a node that is has a depth of at most 5 levels lower than the current node).
    if (stack.size() > 1 && stack.front()->depth - stack.back()->depth > diving_backtrack_limit) {
      stack.pop_back();
    }
  }

  worker_pool_.return_worker_to_pool(worker);
  active_workers_per_strategy_[search_strategy]--;
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::run_scheduler()
{
  diving_heuristics_settings_t<i_t, f_t> diving_settings = settings_.diving_settings;
  const i_t num_workers                                  = 2 * settings_.num_threads;

  if (!std::isfinite(upper_bound_)) { diving_settings.guided_diving = false; }
  std::vector<search_strategy_t> strategies = get_search_strategies(diving_settings);
  std::array<i_t, num_search_strategies> max_num_workers_per_type =
    get_max_workers(num_workers, strategies);

  worker_pool_.init(num_workers, original_lp_, Arow_, var_types_, settings_);
  active_workers_per_strategy_.fill(0);

#ifdef CUOPT_LOG_DEBUG
  for (auto strategy : strategies) {
    settings_.log.debug("%c%d: max num of workers = %d",
                        feasible_solution_symbol(strategy),
                        strategy,
                        max_num_workers_per_type[strategy]);
  }
#endif

  f_t lower_bound     = get_lower_bound();
  f_t abs_gap         = upper_bound_ - lower_bound;
  f_t rel_gap         = user_relative_gap(original_lp_, upper_bound_.load(), lower_bound);
  i_t last_node_depth = 0;
  i_t last_int_infeas = 0;

  while (solver_status_ == mip_status_t::UNSET && abs_gap > settings_.absolute_mip_gap_tol &&
         rel_gap > settings_.relative_mip_gap_tol &&
         (active_workers_per_strategy_[0] > 0 || node_queue_.best_first_queue_size() > 0)) {
    bool launched_any_task = false;
    lower_bound            = get_lower_bound();
    abs_gap                = upper_bound_ - lower_bound;
    rel_gap                = user_relative_gap(original_lp_, upper_bound_.load(), lower_bound);

    repair_heuristic_solutions();

    // If the guided diving was disabled previously due to the lack of an incumbent solution,
    // re-enable as soon as a new incumbent is found.
    if (settings_.diving_settings.guided_diving != diving_settings.guided_diving) {
      if (std::isfinite(upper_bound_)) {
        diving_settings.guided_diving = settings_.diving_settings.guided_diving;
        strategies                    = get_search_strategies(diving_settings);
        max_num_workers_per_type      = get_max_workers(num_workers, strategies);

#ifdef CUOPT_LOG_DEBUG
        for (auto type : strategies) {
          settings_.log.debug("%c%d: max num of workers = %d",
                              feasible_solution_symbol(type),
                              type,
                              max_num_workers_per_type[type]);
        }
#endif
      }
    }

    f_t now = toc(exploration_stats_.start_time);
    f_t time_since_last_log =
      exploration_stats_.last_log == 0 ? 1.0 : toc(exploration_stats_.last_log);
    i_t nodes_since_last_log = exploration_stats_.nodes_since_last_log;

    if (((nodes_since_last_log >= 1000 || abs_gap < 10 * settings_.absolute_mip_gap_tol) &&
         time_since_last_log >= 1) ||
        (time_since_last_log > 30) || now > settings_.time_limit) {
      i_t queue_size = node_queue_.best_first_queue_size();
      i_t depth      = queue_size > 0 ? node_queue_.bfs_top()->depth : last_node_depth;
      i_t int_infeas = queue_size > 0 ? node_queue_.bfs_top()->integer_infeasible : last_int_infeas;
      report(' ', upper_bound_, lower_bound, depth, int_infeas);
      exploration_stats_.last_log             = tic();
      exploration_stats_.nodes_since_last_log = 0;
    }

    if (now > settings_.time_limit) {
      solver_status_ = mip_status_t::TIME_LIMIT;
      break;
    }

    for (auto strategy : strategies) {
      if (active_workers_per_strategy_[strategy] >= max_num_workers_per_type[strategy]) {
        continue;
      }

      // Get an idle worker.
      branch_and_bound_worker_t<i_t, f_t>* worker = worker_pool_.get_idle_worker();
      if (worker == nullptr) { break; }

      if (strategy == BEST_FIRST) {
        // If there any node left in the heap, we pop the top node and explore it.
        std::optional<mip_node_t<i_t, f_t>*> start_node = node_queue_.pop_best_first();

        if (!start_node.has_value()) { continue; }
        if (upper_bound_ < start_node.value()->lower_bound) {
          // This node was put on the heap earlier but its lower bound is now greater than the
          // current upper bound
          search_tree_.graphviz_node(
            settings_.log, start_node.value(), "cutoff", start_node.value()->lower_bound);
          search_tree_.update(start_node.value(), node_status_t::FATHOMED);
          continue;
        }

        // Remove the worker from the idle list.
        worker_pool_.pop_idle_worker();
        worker->init_best_first(start_node.value(), original_lp_);
        last_node_depth = start_node.value()->depth;
        last_int_infeas = start_node.value()->integer_infeasible;
        active_workers_per_strategy_[strategy]++;
        launched_any_task = true;

#pragma omp task affinity(worker)
        plunge_with(worker);

      } else {
        std::optional<mip_node_t<i_t, f_t>*> start_node = node_queue_.pop_diving();

        if (!start_node.has_value()) { continue; }
        if (upper_bound_ < start_node.value()->lower_bound ||
            start_node.value()->depth < diving_settings.min_node_depth) {
          continue;
        }

        bool is_feasible =
          worker->init_diving(start_node.value(), strategy, original_lp_, settings_);
        if (!is_feasible) { continue; }

        // Remove the worker from the idle list.
        worker_pool_.pop_idle_worker();
        active_workers_per_strategy_[strategy]++;
        launched_any_task = true;

#pragma omp task affinity(worker)
        dive_with(worker);
      }
    }

    // If no new task was launched in this iteration, suspend temporarily the
    // execution of the scheduler. As of 8/Jan/2026, GCC does not
    // implement taskyield, but LLVM does.
    if (!launched_any_task) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::single_threaded_solve()
{
  branch_and_bound_worker_t<i_t, f_t> worker(0, original_lp_, Arow_, var_types_, settings_);

  f_t lower_bound = get_lower_bound();
  f_t abs_gap     = upper_bound_ - lower_bound;
  f_t rel_gap     = user_relative_gap(original_lp_, upper_bound_.load(), lower_bound);

  while (solver_status_ == mip_status_t::UNSET && abs_gap > settings_.absolute_mip_gap_tol &&
         rel_gap > settings_.relative_mip_gap_tol && node_queue_.best_first_queue_size() > 0) {
    bool launched_any_task = false;
    lower_bound            = get_lower_bound();
    abs_gap                = upper_bound_ - lower_bound;
    rel_gap                = user_relative_gap(original_lp_, upper_bound_.load(), lower_bound);

    repair_heuristic_solutions();

    f_t now = toc(exploration_stats_.start_time);
    f_t time_since_last_log =
      exploration_stats_.last_log == 0 ? 1.0 : toc(exploration_stats_.last_log);
    i_t nodes_since_last_log = exploration_stats_.nodes_since_last_log;

    if (((nodes_since_last_log >= 1000 || abs_gap < 10 * settings_.absolute_mip_gap_tol) &&
         time_since_last_log >= 1) ||
        (time_since_last_log > 30) || now > settings_.time_limit) {
      i_t depth      = node_queue_.bfs_top()->depth;
      i_t int_infeas = node_queue_.bfs_top()->integer_infeasible;
      report(' ', upper_bound_, lower_bound, depth, int_infeas);
      exploration_stats_.last_log             = tic();
      exploration_stats_.nodes_since_last_log = 0;
    }

    if (now > settings_.time_limit) {
      solver_status_ = mip_status_t::TIME_LIMIT;
      break;
    }

    // If there any node left in the heap, we pop the top node and explore it.
    std::optional<mip_node_t<i_t, f_t>*> start_node = node_queue_.pop_best_first();

    if (!start_node.has_value()) { continue; }
    if (upper_bound_ < start_node.value()->lower_bound) {
      // This node was put on the heap earlier but its lower bound is now greater than the
      // current upper bound
      search_tree_.graphviz_node(
        settings_.log, start_node.value(), "cutoff", start_node.value()->lower_bound);
      search_tree_.update(start_node.value(), node_status_t::FATHOMED);
      continue;
    }

    worker.init_best_first(start_node.value(), original_lp_);
    plunge_with(&worker);
  }
}

template <typename i_t, typename f_t>
lp_status_t branch_and_bound_t<i_t, f_t>::solve_root_relaxation(
  simplex_solver_settings_t<i_t, f_t> const& lp_settings,
  lp_solution_t<i_t, f_t>& root_relax_soln,
  std::vector<variable_status_t>& root_vstatus,
  basis_update_mpf_t<i_t, f_t>& basis_update,
  std::vector<i_t>& basic_list,
  std::vector<i_t>& nonbasic_list,
  std::vector<f_t>& edge_norms)
{
  f_t start_time          = tic();
  f_t user_objective      = 0;
  i_t iter                = 0;
  std::string solver_name = "";

  // Root node path
  lp_status_t root_status;
  std::future<lp_status_t> root_status_future;
  root_status_future = std::async(std::launch::async,
                                  &solve_linear_program_with_advanced_basis<i_t, f_t>,
                                  std::ref(original_lp_),
                                  exploration_stats_.start_time,
                                  std::ref(lp_settings),
                                  std::ref(root_relax_soln),
                                  std::ref(basis_update),
                                  std::ref(basic_list),
                                  std::ref(nonbasic_list),
                                  std::ref(root_vstatus),
                                  std::ref(edge_norms),
                                  nullptr);
  // Wait for the root relaxation solution to be sent by the diversity manager or dual simplex
  // to finish
  while (!root_crossover_solution_set_.load(std::memory_order_acquire) &&
         *get_root_concurrent_halt() == 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    continue;
  }

  if (root_crossover_solution_set_.load(std::memory_order_acquire)) {
    // Crush the root relaxation solution on converted user problem
    std::vector<f_t> crushed_root_x;
    crush_primal_solution(
      original_problem_, original_lp_, root_crossover_soln_.x, new_slacks_, crushed_root_x);
    std::vector<f_t> crushed_root_y;
    std::vector<f_t> crushed_root_z;

    f_t dual_res_inf = crush_dual_solution(original_problem_,
                                           original_lp_,
                                           new_slacks_,
                                           root_crossover_soln_.y,
                                           root_crossover_soln_.z,
                                           crushed_root_y,
                                           crushed_root_z);

    root_crossover_soln_.x = crushed_root_x;
    root_crossover_soln_.y = crushed_root_y;
    root_crossover_soln_.z = crushed_root_z;

    // Call crossover on the crushed solution
    auto root_crossover_settings            = settings_;
    root_crossover_settings.log.log         = false;
    root_crossover_settings.concurrent_halt = get_root_concurrent_halt();
    crossover_status_t crossover_status     = crossover(original_lp_,
                                                    root_crossover_settings,
                                                    root_crossover_soln_,
                                                    exploration_stats_.start_time,
                                                    root_crossover_soln_,
                                                    crossover_vstatus_);

    // Check if crossover was stopped by dual simplex
    if (crossover_status == crossover_status_t::OPTIMAL) {
      set_root_concurrent_halt(1);             // Stop dual simplex
      root_status = root_status_future.get();  // Wait for dual simplex to finish
      set_root_concurrent_halt(0);             // Clear the concurrent halt flag
      // Override the root relaxation solution with the crossover solution
      root_relax_soln = root_crossover_soln_;
      root_vstatus    = crossover_vstatus_;
      root_status     = lp_status_t::OPTIMAL;
      basic_list.clear();
      nonbasic_list.reserve(original_lp_.num_cols - original_lp_.num_rows);
      nonbasic_list.clear();
      // Get the basic list and nonbasic list from the vstatus
      for (i_t j = 0; j < original_lp_.num_cols; j++) {
        if (crossover_vstatus_[j] == variable_status_t::BASIC) {
          basic_list.push_back(j);
        } else {
          nonbasic_list.push_back(j);
        }
      }
      if (basic_list.size() != original_lp_.num_rows) {
        settings_.log.printf(
          "basic_list size %d != m %d\n", basic_list.size(), original_lp_.num_rows);
        assert(basic_list.size() == original_lp_.num_rows);
      }
      if (nonbasic_list.size() != original_lp_.num_cols - original_lp_.num_rows) {
        settings_.log.printf("nonbasic_list size %d != n - m %d\n",
                             nonbasic_list.size(),
                             original_lp_.num_cols - original_lp_.num_rows);
        assert(nonbasic_list.size() == original_lp_.num_cols - original_lp_.num_rows);
      }
      // Populate the basis_update from the crossover vstatus
      i_t refactor_status = basis_update.refactor_basis(original_lp_.A,
                                                        root_crossover_settings,
                                                        original_lp_.lower,
                                                        original_lp_.upper,
                                                        basic_list,
                                                        nonbasic_list,
                                                        crossover_vstatus_);
      if (refactor_status != 0) {
        settings_.log.printf("Failed to refactor basis. %d deficient columns.\n", refactor_status);
        assert(refactor_status == 0);
        root_status = lp_status_t::NUMERICAL_ISSUES;
      }

      // Set the edge norms to a default value
      edge_norms.resize(original_lp_.num_cols, -1.0);
      set_uninitialized_steepest_edge_norms<i_t, f_t>(original_lp_, basic_list, edge_norms);
      user_objective = root_crossover_soln_.user_objective;
      iter           = root_crossover_soln_.iterations;
      solver_name    = "Barrier/PDLP and Crossover";
    } else {
      root_status    = root_status_future.get();
      user_objective = root_relax_soln_.user_objective;
      iter           = root_relax_soln_.iterations;
      solver_name    = "Dual Simplex";
    }
  } else {
    root_status    = root_status_future.get();
    user_objective = root_relax_soln_.user_objective;
    iter           = root_relax_soln_.iterations;
    solver_name    = "Dual Simplex";
  }

  settings_.log.printf("\n");
  if (root_status == lp_status_t::OPTIMAL) {
    settings_.log.printf("Root relaxation solution found in %d iterations and %.2fs by %s\n",
                         iter,
                         toc(start_time),
                         solver_name.c_str());
    settings_.log.printf("Root relaxation objective %+.8e\n", user_objective);
  } else {
    settings_.log.printf("Root relaxation returned: %s\n",
                         lp_status_to_string(root_status).c_str());
  }

  settings_.log.printf("\n");
  is_root_solution_set = true;

  return root_status;
}

template <typename i_t, typename f_t>
mip_status_t branch_and_bound_t<i_t, f_t>::solve(mip_solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range scope("BB::solve");

  logger_t log;
  log.log                             = false;
  log.log_prefix                      = settings_.log.log_prefix;
  solver_status_                      = mip_status_t::UNSET;
  is_running_                         = false;
  exploration_stats_.nodes_unexplored = 0;
  exploration_stats_.nodes_explored   = 0;
  original_lp_.A.to_compressed_row(Arow_);

  if (guess_.size() != 0) {
    raft::common::nvtx::range scope_guess("BB::check_initial_guess");
    std::vector<f_t> crushed_guess;
    crush_primal_solution(original_problem_, original_lp_, guess_, new_slacks_, crushed_guess);
    f_t primal_err;
    f_t bound_err;
    i_t num_fractional;
    const bool feasible = check_guess(
      original_lp_, settings_, var_types_, crushed_guess, primal_err, bound_err, num_fractional);
    if (feasible) {
      const f_t computed_obj = compute_objective(original_lp_, crushed_guess);
      mutex_upper_.lock();
      incumbent_.set_incumbent_solution(computed_obj, crushed_guess);
      upper_bound_ = computed_obj;
      mutex_upper_.unlock();
    }
  }

  root_relax_soln_.resize(original_lp_.num_rows, original_lp_.num_cols);

  i_t original_rows                     = original_lp_.num_rows;
  simplex_solver_settings_t lp_settings = settings_;
  lp_settings.inside_mip                = 1;
  lp_settings.scale_columns             = false;
  lp_settings.concurrent_halt           = get_root_concurrent_halt();
  std::vector<i_t> basic_list(original_lp_.num_rows);
  std::vector<i_t> nonbasic_list;
  basis_update_mpf_t<i_t, f_t> basis_update(original_lp_.num_rows, settings_.refactor_frequency);
  lp_status_t root_status;

  if (!enable_concurrent_lp_root_solve()) {
    // RINS/SUBMIP path
    settings_.log.printf("\nSolving LP root relaxation with dual simplex\n");
    root_status = solve_linear_program_with_advanced_basis(original_lp_,
                                                           exploration_stats_.start_time,
                                                           lp_settings,
                                                           root_relax_soln_,
                                                           basis_update,
                                                           basic_list,
                                                           nonbasic_list,
                                                           root_vstatus_,
                                                           edge_norms_);
  } else {
    settings_.log.printf("\nSolving LP root relaxation in concurrent mode\n");
    root_status = solve_root_relaxation(lp_settings,
                                        root_relax_soln_,
                                        root_vstatus_,
                                        basis_update,
                                        basic_list,
                                        nonbasic_list,
                                        edge_norms_);
  }
  exploration_stats_.total_lp_iters      = root_relax_soln_.iterations;
  exploration_stats_.total_lp_solve_time = toc(exploration_stats_.start_time);

  if (root_status == lp_status_t::INFEASIBLE) {
    settings_.log.printf("MIP Infeasible\n");
    // FIXME: rarely dual simplex detects infeasible whereas it is feasible.
    // to add a small safety net, check if there is a primal solution already.
    // Uncomment this if the issue with cost266-UUE is resolved
    // if (settings.heuristic_preemption_callback != nullptr) {
    //   settings.heuristic_preemption_callback();
    // }
    return mip_status_t::INFEASIBLE;
  }
  if (root_status == lp_status_t::UNBOUNDED) {
    settings_.log.printf("MIP Unbounded\n");
    if (settings_.heuristic_preemption_callback != nullptr) {
      settings_.heuristic_preemption_callback();
    }
    return mip_status_t::UNBOUNDED;
  }
  if (root_status == lp_status_t::TIME_LIMIT) {
    solver_status_ = mip_status_t::TIME_LIMIT;
    set_final_solution(solution, -inf);
    return solver_status_;
  }

  if (root_status == lp_status_t::WORK_LIMIT) {
    solver_status_ = mip_status_t::WORK_LIMIT;
    set_final_solution(solution, -inf);
    return solver_status_;
  }

  if (root_status == lp_status_t::NUMERICAL_ISSUES) {
    solver_status_ = mip_status_t::NUMERICAL;
    set_final_solution(solution, -inf);
    return solver_status_;
  }

  assert(root_vstatus_.size() == original_lp_.num_cols);
  set_uninitialized_steepest_edge_norms<i_t, f_t>(original_lp_, basic_list, edge_norms_);

  root_objective_ = compute_objective(original_lp_, root_relax_soln_.x);

  if (settings_.set_simplex_solution_callback != nullptr) {
    std::vector<f_t> original_x;
    uncrush_primal_solution(original_problem_, original_lp_, root_relax_soln_.x, original_x);
    std::vector<f_t> original_dual;
    std::vector<f_t> original_z;
    uncrush_dual_solution(original_problem_,
                          original_lp_,
                          root_relax_soln_.y,
                          root_relax_soln_.z,
                          original_dual,
                          original_z);
    settings_.set_simplex_solution_callback(
      original_x, original_dual, compute_user_objective(original_lp_, root_objective_));
  }

  std::vector<i_t> fractional;
  i_t num_fractional = fractional_variables(settings_, root_relax_soln_.x, var_types_, fractional);

  cut_info_t<i_t, f_t> cut_info;

  if (num_fractional == 0) {
    set_solution_at_root(solution, cut_info);
    return mip_status_t::OPTIMAL;
  }

  is_running_          = true;
  lower_bound_ceiling_ = inf;

  if (num_fractional != 0 && settings_.max_cut_passes > 0) {
    settings_.log.printf(
      " | Explored | Unexplored |    Objective    |     Bound     | IntInf | Depth | Iter/Node |   "
      "Gap    "
      "|  Time  |\n");
  }

  cut_pool_t<i_t, f_t> cut_pool(original_lp_.num_cols, settings_);
  cut_generation_t<i_t, f_t> cut_generation(
    cut_pool, original_lp_, settings_, Arow_, new_slacks_, var_types_);

  std::vector<f_t> saved_solution;
#ifdef CHECK_CUTS_AGAINST_SAVED_SOLUTION
  read_saved_solution_for_cut_verification(original_lp_, settings_, saved_solution);
#endif

  f_t last_upper_bound     = std::numeric_limits<f_t>::infinity();
  f_t last_objective       = root_objective_;
  f_t root_relax_objective = root_objective_;

  i_t cut_pool_size = 0;
  for (i_t cut_pass = 0; cut_pass < settings_.max_cut_passes; cut_pass++) {
    if (num_fractional == 0) {
      set_solution_at_root(solution, cut_info);
      return mip_status_t::OPTIMAL;
    } else {
#ifdef PRINT_FRACTIONAL_INFO
      settings_.log.printf(
        "Found %d fractional variables on cut pass %d\n", num_fractional, cut_pass);
      for (i_t j : fractional) {
        settings_.log.printf("Fractional variable %d lower %e value %e upper %e\n",
                             j,
                             original_lp_.lower[j],
                             root_relax_soln_.x[j],
                             original_lp_.upper[j]);
      }
#endif

      // Generate cuts and add them to the cut pool
      f_t cut_start_time = tic();
      cut_generation.generate_cuts(original_lp_,
                                   settings_,
                                   Arow_,
                                   new_slacks_,
                                   var_types_,
                                   basis_update,
                                   root_relax_soln_.x,
                                   basic_list,
                                   nonbasic_list);
      f_t cut_generation_time = toc(cut_start_time);
      if (cut_generation_time > 1.0) {
        settings_.log.debug("Cut generation time %.2f seconds\n", cut_generation_time);
      }
      // Score the cuts
      f_t score_start_time = tic();
      cut_pool.score_cuts(root_relax_soln_.x);
      f_t score_time = toc(score_start_time);
      if (score_time > 1.0) { settings_.log.debug("Cut scoring time %.2f seconds\n", score_time); }
      // Get the best cuts from the cut pool
      csr_matrix_t<i_t, f_t> cuts_to_add(0, original_lp_.num_cols, 0);
      std::vector<f_t> cut_rhs;
      std::vector<cut_type_t> cut_types;
      i_t num_cuts = cut_pool.get_best_cuts(cuts_to_add, cut_rhs, cut_types);
      if (num_cuts == 0) { break; }
      cut_info.record_cut_types(cut_types);
#ifdef PRINT_CUT_POOL_TYPES
      cut_pool.print_cutpool_types();
      print_cut_types("In LP      ", cut_types, settings_);
      printf("Cut pool size: %d\n", cut_pool.pool_size());
#endif

#ifdef CHECK_CUT_MATRIX
      if (cuts_to_add.check_matrix() != 0) {
        settings_.log.printf("Bad cuts matrix\n");
        for (i_t i = 0; i < static_cast<i_t>(cut_types.size()); ++i) {
          settings_.log.printf("row %d cut type %d\n", i, cut_types[i]);
        }
        return mip_status_t::NUMERICAL;
      }
#endif
      // Check against saved solution
#ifdef CHECK_CUTS_AGAINST_SAVED_SOLUTION
      verify_cuts_against_saved_solution(cuts_to_add, cut_rhs, saved_solution);
#endif
      cut_pool_size = cut_pool.pool_size();

      // Resolve the LP with the new cuts
      settings_.log.debug(
        "Solving LP with %d cuts (%d cut nonzeros). Cuts in pool %d. Total constraints %d\n",
        num_cuts,
        cuts_to_add.row_start[cuts_to_add.m],
        cut_pool.pool_size(),
        cuts_to_add.m + original_lp_.num_rows);
      lp_settings.log.log = false;

      f_t add_cuts_start_time = tic();
      mutex_original_lp_.lock();
      i_t add_cuts_status = add_cuts(settings_,
                                     cuts_to_add,
                                     cut_rhs,
                                     original_lp_,
                                     new_slacks_,
                                     root_relax_soln_,
                                     basis_update,
                                     basic_list,
                                     nonbasic_list,
                                     root_vstatus_,
                                     edge_norms_);
      var_types_.resize(original_lp_.num_cols, variable_type_t::CONTINUOUS);
      mutex_original_lp_.unlock();
      f_t add_cuts_time = toc(add_cuts_start_time);
      if (add_cuts_time > 1.0) {
        settings_.log.debug("Add cuts time %.2f seconds\n", add_cuts_time);
      }
      if (add_cuts_status != 0) {
        settings_.log.printf("Failed to add cuts\n");
        return mip_status_t::NUMERICAL;
      }

      if (settings_.reduced_cost_strengthening >= 1 && upper_bound_.load() < last_upper_bound) {
        mutex_upper_.lock();
        last_upper_bound = upper_bound_.load();
        std::vector<f_t> lower_bounds;
        std::vector<f_t> upper_bounds;
        find_reduced_cost_fixings(upper_bound_.load(), lower_bounds, upper_bounds);
        mutex_upper_.unlock();
        mutex_original_lp_.lock();
        original_lp_.lower = lower_bounds;
        original_lp_.upper = upper_bounds;
        mutex_original_lp_.unlock();
      }

      // Try to do bound strengthening
      std::vector<bool> bounds_changed(original_lp_.num_cols, true);
      std::vector<char> row_sense;
#ifdef CHECK_MATRICES
      settings_.log.printf("Before A check\n");
      original_lp_.A.check_matrix();
#endif
      original_lp_.A.to_compressed_row(Arow_);

      f_t node_presolve_start_time = tic();
      bounds_strengthening_t<i_t, f_t> node_presolve(original_lp_, Arow_, row_sense, var_types_);
      std::vector<f_t> new_lower = original_lp_.lower;
      std::vector<f_t> new_upper = original_lp_.upper;
      bool feasible =
        node_presolve.bounds_strengthening(settings_, bounds_changed, new_lower, new_upper);
      mutex_original_lp_.lock();
      original_lp_.lower = new_lower;
      original_lp_.upper = new_upper;
      mutex_original_lp_.unlock();
      f_t node_presolve_time = toc(node_presolve_start_time);
      if (node_presolve_time > 1.0) {
        settings_.log.debug("Node presolve time %.2f seconds\n", node_presolve_time);
      }
      if (!feasible) {
        settings_.log.printf("Bound strengthening detected infeasibility\n");
        return mip_status_t::INFEASIBLE;
      }

      i_t iter                    = 0;
      bool initialize_basis       = false;
      lp_settings.concurrent_halt = NULL;
      f_t dual_phase2_start_time  = tic();
      dual::status_t cut_status   = dual_phase2_with_advanced_basis(2,
                                                                  0,
                                                                  initialize_basis,
                                                                  exploration_stats_.start_time,
                                                                  original_lp_,
                                                                  lp_settings,
                                                                  root_vstatus_,
                                                                  basis_update,
                                                                  basic_list,
                                                                  nonbasic_list,
                                                                  root_relax_soln_,
                                                                  iter,
                                                                  edge_norms_);
      exploration_stats_.total_lp_iters += iter;
      root_objective_      = compute_objective(original_lp_, root_relax_soln_.x);
      f_t dual_phase2_time = toc(dual_phase2_start_time);
      if (dual_phase2_time > 1.0) {
        settings_.log.debug("Dual phase2 time %.2f seconds\n", dual_phase2_time);
      }
      if (cut_status == dual::status_t::TIME_LIMIT) {
        solver_status_ = mip_status_t::TIME_LIMIT;
        set_final_solution(solution, root_objective_);
        return solver_status_;
      }

      if (cut_status != dual::status_t::OPTIMAL) {
        settings_.log.printf("Numerical issue at root node. Resolving from scratch\n");
        lp_status_t scratch_status =
          solve_linear_program_with_advanced_basis(original_lp_,
                                                   exploration_stats_.start_time,
                                                   lp_settings,
                                                   root_relax_soln_,
                                                   basis_update,
                                                   basic_list,
                                                   nonbasic_list,
                                                   root_vstatus_,
                                                   edge_norms_);
        if (scratch_status == lp_status_t::OPTIMAL) {
          // We recovered
          cut_status = convert_lp_status_to_dual_status(scratch_status);
          exploration_stats_.total_lp_iters += root_relax_soln_.iterations;
          root_objective_ = compute_objective(original_lp_, root_relax_soln_.x);
        } else {
          settings_.log.printf("Cut status %s\n", dual::status_to_string(cut_status).c_str());
          return mip_status_t::NUMERICAL;
        }
      }

      f_t remove_cuts_start_time = tic();
      mutex_original_lp_.lock();
      remove_cuts(original_lp_,
                  settings_,
                  Arow_,
                  new_slacks_,
                  original_rows,
                  var_types_,
                  root_vstatus_,
                  edge_norms_,
                  root_relax_soln_.x,
                  root_relax_soln_.y,
                  root_relax_soln_.z,
                  basic_list,
                  nonbasic_list,
                  basis_update);
      mutex_original_lp_.unlock();
      f_t remove_cuts_time = toc(remove_cuts_start_time);
      if (remove_cuts_time > 1.0) {
        settings_.log.debug("Remove cuts time %.2f seconds\n", remove_cuts_time);
      }
      fractional.clear();
      num_fractional = fractional_variables(settings_, root_relax_soln_.x, var_types_, fractional);

      if (num_fractional == 0) {
        upper_bound_ = root_objective_;
        mutex_upper_.lock();
        incumbent_.set_incumbent_solution(root_objective_, root_relax_soln_.x);
        mutex_upper_.unlock();
      }
      f_t obj = upper_bound_.load();
      report(' ', obj, root_objective_, 0, num_fractional);

      f_t rel_gap = user_relative_gap(original_lp_, upper_bound_.load(), root_objective_);
      f_t abs_gap = upper_bound_.load() - root_objective_;
      if (rel_gap < settings_.relative_mip_gap_tol || abs_gap < settings_.absolute_mip_gap_tol) {
        set_solution_at_root(solution, cut_info);
        set_final_solution(solution, root_objective_);
        return mip_status_t::OPTIMAL;
      }

      f_t change_in_objective = root_objective_ - last_objective;
      const f_t factor        = settings_.cut_change_threshold;
      const f_t min_objective = 1e-3;
      if (change_in_objective <= factor * std::max(min_objective, std::abs(root_relax_objective))) {
        settings_.log.debug(
          "Change in objective %.16e is less than 1e-3 of root relax objective %.16e\n",
          change_in_objective,
          root_relax_objective);
        break;
      }
      last_objective = root_objective_;
    }
  }

  print_cut_info(settings_, cut_info);

  if (cut_info.has_cuts()) {
    settings_.log.printf("Cut pool size  : %d\n", cut_pool_size);
    settings_.log.printf("Size with cuts : %d constraints, %d variables, %d nonzeros\n",
                         original_lp_.num_rows,
                         original_lp_.num_cols,
                         original_lp_.A.col_start[original_lp_.A.n]);
  }

  set_uninitialized_steepest_edge_norms(original_lp_, basic_list, edge_norms_);

  pc_.resize(original_lp_.num_cols);
  {
    raft::common::nvtx::range scope_sb("BB::strong_branching");
    strong_branching<i_t, f_t>(original_problem_,
                               original_lp_,
                               settings_,
                               exploration_stats_.start_time,
                               var_types_,
                               root_relax_soln_.x,
                               fractional,
                               root_objective_,
                               root_vstatus_,
                               edge_norms_,
                               pc_);
  }

  if (toc(exploration_stats_.start_time) > settings_.time_limit) {
    solver_status_ = mip_status_t::TIME_LIMIT;
    set_final_solution(solution, root_objective_);
    return solver_status_;
  }

  if (settings_.reduced_cost_strengthening >= 2 && upper_bound_.load() < last_upper_bound) {
    std::vector<f_t> lower_bounds;
    std::vector<f_t> upper_bounds;
    i_t num_fixed = find_reduced_cost_fixings(upper_bound_.load(), lower_bounds, upper_bounds);
    if (num_fixed > 0) {
      std::vector<bool> bounds_changed(original_lp_.num_cols, true);
      std::vector<char> row_sense;

      bounds_strengthening_t<i_t, f_t> node_presolve(original_lp_, Arow_, row_sense, var_types_);

      mutex_original_lp_.lock();
      original_lp_.lower = lower_bounds;
      original_lp_.upper = upper_bounds;
      bool feasible      = node_presolve.bounds_strengthening(
        settings_, bounds_changed, original_lp_.lower, original_lp_.upper);
      mutex_original_lp_.unlock();
      if (!feasible) {
        settings_.log.printf("Bound strengthening failed\n");
        return mip_status_t::NUMERICAL;  // We had a feasible integer solution, but bound
                                         // strengthening thinks we are infeasible.
      }
      // Go through and check the fractional variables and remove any that are now fixed to their
      // bounds
      std::vector<i_t> to_remove(fractional.size(), 0);
      i_t num_to_remove = 0;
      for (i_t k = 0; k < fractional.size(); k++) {
        const i_t j = fractional[k];
        if (std::abs(original_lp_.upper[j] - original_lp_.lower[j]) < settings_.fixed_tol) {
          to_remove[k] = 1;
          num_to_remove++;
        }
      }
      if (num_to_remove > 0) {
        std::vector<i_t> new_fractional;
        new_fractional.reserve(fractional.size() - num_to_remove);
        for (i_t k = 0; k < fractional.size(); k++) {
          if (!to_remove[k]) { new_fractional.push_back(fractional[k]); }
        }
        fractional     = new_fractional;
        num_fractional = fractional.size();
      }
    }
  }

  // Choose variable to branch on
  i_t branch_var = pc_.variable_selection(fractional, root_relax_soln_.x, log);

  search_tree_.root      = std::move(mip_node_t<i_t, f_t>(root_objective_, root_vstatus_));
  search_tree_.num_nodes = 0;
  search_tree_.graphviz_node(settings_.log, &search_tree_.root, "lower bound", root_objective_);
  search_tree_.branch(&search_tree_.root,
                      branch_var,
                      root_relax_soln_.x[branch_var],
                      num_fractional,
                      root_vstatus_,
                      original_lp_,
                      log);
  node_queue_.push(search_tree_.root.get_down_child());
  node_queue_.push(search_tree_.root.get_up_child());

  settings_.log.printf("Exploring the B&B tree using %d threads\n\n", settings_.num_threads);

  exploration_stats_.nodes_explored       = 0;
  exploration_stats_.nodes_unexplored     = 2;
  exploration_stats_.nodes_since_last_log = 0;
  exploration_stats_.last_log             = tic();
  min_node_queue_size_                    = 2 * settings_.num_threads;

  if (settings_.diving_settings.coefficient_diving != 0) {
    calculate_variable_locks(original_lp_, var_up_locks_, var_down_locks_);
  }
  if (settings_.deterministic) {
    settings_.log.printf(
      " | Explored | Unexplored |    Objective    |     Bound     | IntInf | Depth | Iter/Node "
      "|   Gap    |  Work |  Time  |\n");
  } else {
    settings_.log.printf(
      " | Explored | Unexplored |    Objective    |     Bound     | IntInf | Depth | Iter/Node "
      "|   Gap    |  Time  |\n");
  }

  if (settings_.deterministic) {
    run_deterministic_coordinator(Arow_);
  } else if (settings_.num_threads > 1) {
#pragma omp parallel num_threads(settings_.num_threads)
    {
#pragma omp master
      run_scheduler();
    }
  } else {
    single_threaded_solve();
  }

  is_running_ = false;

  // Compute final lower bound
  f_t lower_bound;
  if (deterministic_mode_enabled_) {
    lower_bound    = deterministic_compute_lower_bound();
    solver_status_ = deterministic_global_termination_status_;
  } else {
    lower_bound = node_queue_.best_first_queue_size() > 0 ? node_queue_.get_lower_bound()
                                                          : search_tree_.root.lower_bound;
  }
  set_final_solution(solution, lower_bound);
  return solver_status_;
}

// ============================================================================
//  Deterministic implementation
// ============================================================================

// The deterministic BSP model is based on letting independent workers execute during virtual time
// intervals, and exchange data during serialized interval sync points.
/*

Work Units:   0                              0.5                              1.0
              │                               │                                │
              │◄──────── Horizon 0 ──────────►│◄───────── Horizon 1 ──────────►│
              │                               │                                │
══════════════╪═══════════════════════════════╪════════════════════════════════╪════
              │                               │                                │
              │                        ┌──────────────┐                  ┌──────────────┐
 BFS Worker 0 │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │              │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │              │
 ├ plunge     │  explore nodes         │              │  explore nodes   │              │
 │  stack     │  emit events (wut)     │              │  emit events     │              │
 ├ backlog    │                        │   SYNC S1    │                  │   SYNC S2    │
 │  heap      │                        │              │                  │              │
 ├ PC snap    │                        │ • Sort by    │                  │ • Sort by    │
 ├ events[]   │                        │   (wut, w,   │                  │   (wut, w,   │
 └ solutions[]│                        │    seq)      │                  │    seq)      │
──────────────┼────────────────────────│ • Replay     │──────────────────│ • Replay     │
              │                        │ • Merge PC   │                  │ • Merge PC   │
 BFS Worker 1 │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │ • Merge sols │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │ • Merge sols │
 ├ plunge     │  explore nodes         │ • Prune      │  explore nodes   │ • Prune      │
 │  stack     │  emit events (wut)     │ • Balance    │  emit events     │ • Balance    │
 ├ backlog    │                        │ • Assign     │                  │ • Assign     │
 │  heap      │                        │ • Snapshot   │                  │ • Snapshot   │
 ├ PC snap    │                        │              │                  │              │
 ├ events[]   │                        │ [38779ebd]   │                  │ [2ad65699]   │
 └ solutions[]│                        │              │                  │              │
──────────────┼────────────────────────│              │──────────────────│              │
              │                        │              │                  │              │
 Diving D0    │ ░░░░░░░░░░░░░░░░░░░░░░ │              │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │              │
 ├ dive_queue │  (waiting)             │              │  dive, find sols │              │
 ├ PC snap    │                        │              │                  │              │
 ├ incumbent  │                        │              │                  │              │
 │  snap      │                        │              │                  │              │
 ├ pc_updates │                        │              │                  │              │
 └ solutions[]│                        │              │                  │              │
──────────────┼────────────────────────│              │──────────────────│              │
              │                        │              │                  │              │
 Diving D1    │ ░░░░░░░░░░░░░░░░░░░░░░ │              │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │              │
 ├ dive_queue │  (waiting)             │              │  dive, find sols │              │
 ├ PC snap    │                        │              │                  │              │
 ├ incumbent  │                        └──────────────┘                  └──────────────┘
 │  snap      │
 ├ pc_updates │
 └ solutions[]│
══════════════╪═══════════════════════════════════════════════════════════════════════════
              │
              ▼
──────────────────────────────────────────────────────────────────────────────────────────►
                                                                        Work Unit Time

Legend:  ▓▓▓ = actively working    ░░░ = waiting at barrier    [hash] = state hash for verification
         wut = work unit timestamp    PC = pseudo-costs    snap = snapshot (local copy)

*/

/* Glossary for B&B Determinism:

Tree Update Policy:
  Class implementing the determinism_base_policy_t interface,
  specifying operations to be executed based on the outcomes of the current node
  in order to unify the deterministic and nondeterministic codepaths.
Worker Pool:
  Static structure containing worker types for deterministic B&B,
  with a 1thread:1worker mapping.
Work Unit Scheduler:
  Class orchestrating the deterministic workers, handling periodic synchronization
  after a set amount of work unit time is elapsed.
Snapshots:
  Local copy of the global state of the solver (incumbent, pseudocosts, upper bound)
  renewed after every sync step in the deterministic codepath
  in order to ensure deterministic playback
  Local snapshots are updated by their respective worker within a horizon,
  and then merged during the sync step, and broadcast to workers for the next horizon.
Producer:
  Independent thread which produces heuristic solutions without depending on the B&B state.
  Therefore, its synchronization requirements are more lax: it can run "ahead" of B&B safely.
Determinism Sync Callback:
  Function that is executed serially (by a single thread) at each synchronization point
  of the determinism codepath. Equivalent to the OpenMP 'single' directive.
Event / BB Event:
  Event susceptible of modifying the global state, recorded within each horizon to be
  sorted and replayed at the sync callback in order to update the global state serially.
Packed Id:
  Unique representation of a node from its <worker_id, seq_id> tuple, packed as a 64bit integer.
Producer Sync:
  Synchronization point ensuring the produced is never running "in the past" wrt B&B.
  Producing solutions in the past would break determinism, therefore this unidirectional sync
ensures no such thing can occur. Instrumentation Aggregator: Collects multiple instrument vectors
into a single aggregation point for estimating work from memory operations. Worker Context: Object
representing the "context" (e.g.: the worker) that should register the amount of work recorded There
is a 1context:1worker mapping. The Work Unit Scheduler registers such contexts and ensure they
remained synchronized together. Queued Integer Solutions: New integer solutions found within
horizons are queued with a work unit timestamp, in order to be sorted and played in order during the
sync callback. Creation Sequence: In nondeterministic mode, a single global atomic integer is used
to generate sequential IDs for the nodes. Since this is a global atomic, it is inherently
nondeterministic. To fix this, in deterministic mode, nodes are addressed by a tuple <worker_id,
seq_id>
  where "worker_id" is the ID of the worker that created this node, and "seq_id" is a sequential ID
local to the worker.\ This sequential ID is similar in principle to the global atomic ID sequence of
the nondeterminsitic mode but since it is local to each worker, it is updated serially and thus is
deterministic. worker IDs are unique, and sequence IDs are unique to their workers, therefor
  <worker_id, seq_id> is a globally unique node identifier.
Pseudocost Update:
  Each worker updates its local pseudocosts when branching. These updates are queued within
horizons. During the horizon sync, these updates are all played in order, and the newly updated
global pseudocosts are broadcast to the worker's pseudocost snapshots for the coming horizon.

*/

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::run_deterministic_coordinator(const csr_matrix_t<i_t, f_t>& Arow)
{
  raft::common::nvtx::range scope("BB::deterministic_coordinator");

  deterministic_horizon_step_ = 0.50;

  // Compute worker counts using the same formula as reliability-branching scheduler
  const i_t num_workers = 2 * settings_.num_threads;
  std::vector<search_strategy_t> search_strategies =
    get_search_strategies(settings_.diving_settings);
  std::array<i_t, num_search_strategies> max_num_workers =
    get_max_workers(num_workers, search_strategies);

  const int num_bfs_workers = max_num_workers[search_strategy_t::BEST_FIRST];
  int num_diving_workers    = 0;
  for (size_t i = 1; i < search_strategies.size(); ++i) {
    num_diving_workers += max_num_workers[search_strategies[i]];
  }

  deterministic_mode_enabled_              = true;
  deterministic_current_horizon_           = deterministic_horizon_step_;
  deterministic_horizon_number_            = 0;
  deterministic_global_termination_status_ = mip_status_t::UNSET;

  deterministic_workers_ = std::make_unique<deterministic_bfs_worker_pool_t<i_t, f_t>>(
    num_bfs_workers, original_lp_, Arow, var_types_, settings_);

  if (num_diving_workers > 0) {
    // Extract diving types from search_strategies (skip BEST_FIRST at index 0)
    std::vector<search_strategy_t> diving_types(search_strategies.begin() + 1,
                                                search_strategies.end());

    if (settings_.diving_settings.coefficient_diving != 0) {
      calculate_variable_locks(original_lp_, var_up_locks_, var_down_locks_);
    }

    if (!diving_types.empty()) {
      deterministic_diving_workers_ =
        std::make_unique<deterministic_diving_worker_pool_t<i_t, f_t>>(num_diving_workers,
                                                                       diving_types,
                                                                       original_lp_,
                                                                       Arow,
                                                                       var_types_,
                                                                       settings_,
                                                                       &root_relax_soln_.x);
    }
  }

  deterministic_scheduler_ = std::make_unique<work_unit_scheduler_t>(deterministic_horizon_step_);

  scoped_context_registrations_t context_registrations(*deterministic_scheduler_);
  for (auto& worker : *deterministic_workers_) {
    context_registrations.add(worker.work_context);
  }
  if (deterministic_diving_workers_) {
    for (auto& worker : *deterministic_diving_workers_) {
      context_registrations.add(worker.work_context);
    }
  }

  int actual_diving_workers =
    deterministic_diving_workers_ ? (int)deterministic_diving_workers_->size() : 0;
  settings_.log.printf(
    "Deterministic Mode: %d BFS workers + %d diving workers, horizon step = %.2f work "
    "units\n",
    num_bfs_workers,
    actual_diving_workers,
    deterministic_horizon_step_);

  search_tree_.root.get_down_child()->origin_worker_id = -1;
  search_tree_.root.get_down_child()->creation_seq     = 0;
  search_tree_.root.get_up_child()->origin_worker_id   = -1;
  search_tree_.root.get_up_child()->creation_seq       = 1;

  (*deterministic_workers_)[0].enqueue_node(search_tree_.root.get_down_child());
  (*deterministic_workers_)[1 % num_bfs_workers].enqueue_node(search_tree_.root.get_up_child());

  deterministic_scheduler_->set_sync_callback([this](double) { deterministic_sync_callback(); });

  std::vector<f_t> incumbent_snapshot;
  if (incumbent_.has_incumbent) { incumbent_snapshot = incumbent_.x; }

  deterministic_broadcast_snapshots(*deterministic_workers_, incumbent_snapshot);
  if (deterministic_diving_workers_) {
    deterministic_broadcast_snapshots(*deterministic_diving_workers_, incumbent_snapshot);
  }

  const int total_thread_count = num_bfs_workers + num_diving_workers;

#pragma omp parallel num_threads(total_thread_count)
  {
    int thread_id = omp_get_thread_num();
    if (thread_id < num_bfs_workers) {
      auto& worker          = (*deterministic_workers_)[thread_id];
      f_t worker_start_time = tic();
      run_deterministic_bfs_loop(worker, search_tree_);
      worker.total_runtime += toc(worker_start_time);
    } else {
      int diving_id         = thread_id - num_bfs_workers;
      auto& worker          = (*deterministic_diving_workers_)[diving_id];
      f_t worker_start_time = tic();
      run_deterministic_diving_loop(worker);
      worker.total_runtime += toc(worker_start_time);
    }
  }

  settings_.log.printf("\n");
  settings_.log.printf("BFS Worker Statistics:\n");
  settings_.log.printf(
    "  Worker |  Nodes  | Branched | Pruned | Infeas. | IntSol | Assigned |  Clock   | "
    "Sync%% | NoWork\n");
  settings_.log.printf(
    "  "
    "-------+---------+----------+--------+---------+--------+----------+----------+-------+-------"
    "\n");
  for (const auto& worker : *deterministic_workers_) {
    double sync_time    = worker.work_context.total_sync_time;
    double total_time   = worker.total_runtime;  // Already includes sync time
    double sync_percent = (total_time > 0) ? (100.0 * sync_time / total_time) : 0.0;
    settings_.log.printf("  %6d | %7d | %8d | %6d | %7d | %6d | %8d | %7.3fs | %4.1f%% | %5.2fs\n",
                         worker.worker_id,
                         worker.total_nodes_processed,
                         worker.total_nodes_branched,
                         worker.total_nodes_pruned,
                         worker.total_nodes_infeasible,
                         worker.total_integer_solutions,
                         worker.total_nodes_assigned,
                         total_time,
                         std::min(99.9, sync_percent),
                         worker.total_nowork_time);
  }

  // Print diving worker statistics
  if (deterministic_diving_workers_ && deterministic_diving_workers_->size() > 0) {
    settings_.log.printf("\n");
    settings_.log.printf("Diving Worker Statistics:\n");
    settings_.log.printf("  Worker |  Type  |  Dives  | Nodes  | IntSol |  Clock   | NoWork\n");
    settings_.log.printf("  -------+--------+---------+--------+--------+----------+-------\n");
    for (const auto& worker : *deterministic_diving_workers_) {
      const char* type_str = "???";
      switch (worker.diving_type) {
        case search_strategy_t::PSEUDOCOST_DIVING: type_str = "PC"; break;
        case search_strategy_t::LINE_SEARCH_DIVING: type_str = "LS"; break;
        case search_strategy_t::GUIDED_DIVING: type_str = "GD"; break;
        case search_strategy_t::COEFFICIENT_DIVING: type_str = "CD"; break;
        default: break;
      }
      settings_.log.printf("  %6d | %6s | %7d | %6d | %6d | %7.3fs | %5.2fs\n",
                           worker.worker_id,
                           type_str,
                           worker.total_dives,
                           worker.total_nodes_explored,
                           worker.total_integer_solutions,
                           worker.total_runtime,
                           worker.total_nowork_time);
    }
  }

  if (producer_sync_.num_producers() > 0 || producer_wait_count_ > 0) {
    double avg_wait =
      (producer_wait_count_ > 0) ? total_producer_wait_time_ / producer_wait_count_ : 0.0;
    settings_.log.printf("Producer Sync Statistics:\n");
    settings_.log.printf(
      "  Producers: %zu, Syncs: %d\n", producer_sync_.num_producers(), producer_wait_count_);
    settings_.log.printf("  Total wait: %.3fs, Avg: %.4fs, Max: %.4fs\n",
                         total_producer_wait_time_,
                         avg_wait,
                         max_producer_wait_time_);
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::run_deterministic_bfs_loop(
  deterministic_bfs_worker_t<i_t, f_t>& worker, search_tree_t<i_t, f_t>& search_tree)
{
  raft::common::nvtx::range scope("BB::worker_loop");

  while (deterministic_global_termination_status_ == mip_status_t::UNSET) {
    if (worker.has_work()) {
      mip_node_t<i_t, f_t>* node = worker.dequeue_node();
      if (node == nullptr) { continue; }

      worker.current_node = node;

      f_t upper_bound = worker.local_upper_bound;
      f_t rel_gap     = user_relative_gap(original_lp_, upper_bound, node->lower_bound);
      if (node->lower_bound > upper_bound || rel_gap < settings_.relative_mip_gap_tol) {
        worker.current_node = nullptr;
        worker.record_fathomed(node, node->lower_bound);
        search_tree.update(node, node_status_t::FATHOMED);
        --exploration_stats_.nodes_unexplored;
        continue;
      }

      bool is_child                     = (node->parent == worker.last_solved_node);
      worker.recompute_bounds_and_basis = !is_child;

      node_status_t status    = solve_node_deterministic(worker, node, search_tree);
      worker.last_solved_node = node;

      worker.current_node = nullptr;
      continue;
    }

    // No work - advance to sync point to participate in barrier
    f_t nowork_start = tic();
    deterministic_scheduler_->wait_for_next_sync(worker.work_context);
    worker.total_nowork_time += toc(nowork_start);
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::deterministic_sync_callback()
{
  raft::common::nvtx::range scope("BB::deterministic_sync_callback");

  ++deterministic_horizon_number_;
  double horizon_end = deterministic_current_horizon_;

  double wait_start = tic();
  producer_sync_.wait_for_producers(horizon_end);
  double wait_time = toc(wait_start);
  total_producer_wait_time_ += wait_time;
  max_producer_wait_time_ = std::max(max_producer_wait_time_, wait_time);
  ++producer_wait_count_;

  work_unit_context_.global_work_units_elapsed = horizon_end;

  bb_event_batch_t<i_t, f_t> all_events = deterministic_workers_->collect_and_sort_events();

  deterministic_sort_replay_events(all_events);

  // deterministic_prune_worker_nodes_vs_incumbent();

  deterministic_collect_diving_solutions_and_update_pseudocosts();

  for (auto& worker : *deterministic_workers_) {
    worker.integer_solutions.clear();
  }
  if (deterministic_diving_workers_) {
    for (auto& worker : *deterministic_diving_workers_) {
      worker.integer_solutions.clear();
    }
  }

  deterministic_populate_diving_heap();

  deterministic_assign_diving_nodes();

  deterministic_balance_worker_loads();

  uint32_t state_hash = 0;
  {
    std::vector<uint64_t> state_data;
    state_data.push_back(static_cast<uint64_t>(exploration_stats_.nodes_explored));
    state_data.push_back(static_cast<uint64_t>(exploration_stats_.nodes_unexplored));
    f_t ub = upper_bound_.load();
    f_t lb = deterministic_compute_lower_bound();
    state_data.push_back(std::bit_cast<uint64_t>(ub));
    state_data.push_back(std::bit_cast<uint64_t>(lb));

    for (auto& worker : *deterministic_workers_) {
      if (worker.current_node != nullptr) {
        state_data.push_back(worker.current_node->get_id_packed());
      }
      for (auto* node : worker.plunge_stack) {
        state_data.push_back(node->get_id_packed());
      }
      for (auto* node : worker.backlog.data()) {
        state_data.push_back(node->get_id_packed());
      }
    }

    if (deterministic_diving_workers_) {
      for (auto& diving_worker : *deterministic_diving_workers_) {
        for (const auto& dive_entry : diving_worker.dive_queue) {
          state_data.push_back(dive_entry.node.get_id_packed());
        }
      }
    }

    state_hash = detail::compute_hash(state_data);
    state_hash ^= pc_.compute_state_hash();
  }

  deterministic_current_horizon_ += deterministic_horizon_step_;

  std::vector<f_t> incumbent_snapshot;
  if (incumbent_.has_incumbent) { incumbent_snapshot = incumbent_.x; }

  deterministic_broadcast_snapshots(*deterministic_workers_, incumbent_snapshot);
  if (deterministic_diving_workers_) {
    deterministic_broadcast_snapshots(*deterministic_diving_workers_, incumbent_snapshot);
  }

  f_t lower_bound = deterministic_compute_lower_bound();
  f_t upper_bound = upper_bound_.load();
  f_t abs_gap     = upper_bound - lower_bound;
  f_t rel_gap     = user_relative_gap(original_lp_, upper_bound, lower_bound);

  if (abs_gap <= settings_.absolute_mip_gap_tol || rel_gap <= settings_.relative_mip_gap_tol) {
    deterministic_global_termination_status_ = mip_status_t::OPTIMAL;
  }

  if (!deterministic_workers_->any_has_work()) {
    // Tree exhausted - check if we found a solution
    if (upper_bound == std::numeric_limits<f_t>::infinity()) {
      deterministic_global_termination_status_ = mip_status_t::INFEASIBLE;
    } else {
      deterministic_global_termination_status_ = mip_status_t::OPTIMAL;
    }
  }

  if (toc(exploration_stats_.start_time) > settings_.time_limit) {
    deterministic_global_termination_status_ = mip_status_t::TIME_LIMIT;
  }

  // Stop early if next horizon exceeds work limit
  if (deterministic_current_horizon_ > settings_.work_limit) {
    deterministic_global_termination_status_ = mip_status_t::WORK_LIMIT;
  }

  // Signal shutdown to prevent threads from entering barriers after termination
  if (deterministic_global_termination_status_ != mip_status_t::UNSET) {
    deterministic_scheduler_->signal_shutdown();
  }

  f_t time_since_last_log =
    exploration_stats_.last_log == 0 ? 1.0 : toc(exploration_stats_.last_log);
  if (time_since_last_log >= 1) {
    report(' ', upper_bound, lower_bound, 0, 0, deterministic_current_horizon_);
    exploration_stats_.last_log = tic();
  }

  f_t obj              = compute_user_objective(original_lp_, upper_bound);
  f_t user_lower       = compute_user_objective(original_lp_, lower_bound);
  std::string gap_user = user_mip_gap<f_t>(obj, user_lower);

  std::string idle_workers;
  i_t idle_count = 0;
  for (const auto& w : *deterministic_workers_) {
    if (!w.has_work() && w.current_node == nullptr) { ++idle_count; }
  }
  idle_workers = idle_count > 0 ? std::to_string(idle_count) + " idle" : "";

#ifdef DETERMINISM_LOG_SYNCS
  settings_.log.printf("W%-5g %8d   %8lu    %+13.6e    %+10.6e    %s %8.2f  [%08x]%s%s\n",
                       deterministic_current_horizon_,
                       exploration_stats_.nodes_explored,
                       exploration_stats_.nodes_unexplored,
                       obj,
                       user_lower,
                       gap_user.c_str(),
                       toc(exploration_stats_.start_time),
                       state_hash,
                       idle_workers.empty() ? "" : " ",
                       idle_workers.c_str());
#endif
}

template <typename i_t, typename f_t>
node_status_t branch_and_bound_t<i_t, f_t>::solve_node_deterministic(
  deterministic_bfs_worker_t<i_t, f_t>& worker,
  mip_node_t<i_t, f_t>* node_ptr,
  search_tree_t<i_t, f_t>& search_tree)
{
  raft::common::nvtx::range scope("BB::solve_node_deterministic");

  double work_units_at_start = worker.work_context.global_work_units_elapsed;

  std::fill(worker.bounds_changed.begin(), worker.bounds_changed.end(), false);

  if (worker.recompute_bounds_and_basis) {
    worker.leaf_problem.lower = original_lp_.lower;
    worker.leaf_problem.upper = original_lp_.upper;
    node_ptr->get_variable_bounds(
      worker.leaf_problem.lower, worker.leaf_problem.upper, worker.bounds_changed);
  } else {
    node_ptr->update_branched_variable_bounds(
      worker.leaf_problem.lower, worker.leaf_problem.upper, worker.bounds_changed);
  }

  double remaining_time = settings_.time_limit - toc(exploration_stats_.start_time);

  // Bounds strengthening
  simplex_solver_settings_t<i_t, f_t> lp_settings = settings_;
  lp_settings.set_log(false);

  lp_settings.cut_off       = worker.local_upper_bound + settings_.dual_tol;
  lp_settings.inside_mip    = 2;
  lp_settings.time_limit    = remaining_time;
  lp_settings.scale_columns = false;

  bool feasible = true;
#ifndef DETERMINISM_DISABLE_BOUNDS_STRENGTHENING
  raft::common::nvtx::range scope_bs("BB::bound_strengthening");
  feasible = worker.node_presolver.bounds_strengthening(
    lp_settings, worker.bounds_changed, worker.leaf_problem.lower, worker.leaf_problem.upper);

  if (settings_.deterministic) {
    // TEMP APPROXIMATION;
    worker.work_context.record_work_sync_on_horizon(worker.node_presolver.last_nnz_processed / 1e8);
  }
#endif

  if (!feasible) {
    node_ptr->lower_bound = std::numeric_limits<f_t>::infinity();
    search_tree.update(node_ptr, node_status_t::INFEASIBLE);
    worker.record_infeasible(node_ptr);
    --exploration_stats_.nodes_unexplored;
    ++exploration_stats_.nodes_explored;
    worker.recompute_bounds_and_basis = true;
    return node_status_t::INFEASIBLE;
  }

  // Solve LP relaxation
  worker.leaf_solution.resize(worker.leaf_problem.num_rows, worker.leaf_problem.num_cols);
  std::vector<variable_status_t>& leaf_vstatus = node_ptr->vstatus;
  i_t node_iter                                = 0;
  f_t lp_start_time                            = tic();
  std::vector<f_t> leaf_edge_norms             = edge_norms_;

  dual::status_t lp_status = dual_phase2_with_advanced_basis(2,
                                                             0,
                                                             worker.recompute_bounds_and_basis,
                                                             lp_start_time,
                                                             worker.leaf_problem,
                                                             lp_settings,
                                                             leaf_vstatus,
                                                             worker.basis_factors,
                                                             worker.basic_list,
                                                             worker.nonbasic_list,
                                                             worker.leaf_solution,
                                                             node_iter,
                                                             leaf_edge_norms,
                                                             &worker.work_context);

  if (lp_status == dual::status_t::NUMERICAL) {
    settings_.log.printf("Numerical issue node %d. Resolving from scratch.\n", node_ptr->node_id);
    lp_status_t second_status = solve_linear_program_with_advanced_basis(worker.leaf_problem,
                                                                         lp_start_time,
                                                                         lp_settings,
                                                                         worker.leaf_solution,
                                                                         worker.basis_factors,
                                                                         worker.basic_list,
                                                                         worker.nonbasic_list,
                                                                         leaf_vstatus,
                                                                         leaf_edge_norms,
                                                                         &worker.work_context);
    lp_status                 = convert_lp_status_to_dual_status(second_status);
  }

  double work_performed = worker.work_context.global_work_units_elapsed - work_units_at_start;
  worker.clock += work_performed;

  exploration_stats_.total_lp_solve_time += toc(lp_start_time);
  exploration_stats_.total_lp_iters += node_iter;
  ++exploration_stats_.nodes_explored;
  --exploration_stats_.nodes_unexplored;

  deterministic_bfs_policy_t<i_t, f_t> policy{*this, worker};
  auto [status, round_dir] = update_tree_impl(node_ptr, search_tree, &worker, lp_status, policy);

  return status;
}

template <typename i_t, typename f_t>
template <typename PoolT, typename WorkerTypeGetter>
void branch_and_bound_t<i_t, f_t>::deterministic_process_worker_solutions(
  PoolT& pool, WorkerTypeGetter get_worker_type)
{
  std::vector<queued_integer_solution_t<i_t, f_t>*> all_solutions;
  for (auto& worker : pool) {
    for (auto& sol : worker.integer_solutions) {
      all_solutions.push_back(&sol);
    }
  }

  // relies on queued_integer_solution_t's operator<
  // sorts based on objective first, then the <worker_id, seq_id> tuple
  std::sort(all_solutions.begin(),
            all_solutions.end(),
            [](const queued_integer_solution_t<i_t, f_t>* a,
               const queued_integer_solution_t<i_t, f_t>* b) { return *a < *b; });

  f_t deterministic_lower = deterministic_compute_lower_bound();
  f_t current_upper       = upper_bound_.load();

  for (const auto* sol : all_solutions) {
    if (sol->objective < current_upper) {
      f_t user_obj         = compute_user_objective(original_lp_, sol->objective);
      f_t user_lower       = compute_user_objective(original_lp_, deterministic_lower);
      i_t nodes_explored   = exploration_stats_.nodes_explored.load();
      i_t nodes_unexplored = exploration_stats_.nodes_unexplored.load();

      search_strategy_t worker_type = get_worker_type(pool, sol->worker_id);
      report(feasible_solution_symbol(worker_type),
             sol->objective,
             deterministic_lower,
             sol->depth,
             0,
             deterministic_current_horizon_);

      bool improved = false;
      if (sol->objective < upper_bound_) {
        upper_bound_ = sol->objective;
        incumbent_.set_incumbent_solution(sol->objective, sol->solution);
        current_upper = sol->objective;
        improved      = true;
      }

      if (improved && settings_.solution_callback != nullptr) {
        std::vector<f_t> original_x;
        uncrush_primal_solution(original_problem_, original_lp_, sol->solution, original_x);
        settings_.solution_callback(original_x, sol->objective);
      }
    }
  }

  for (auto& worker : pool) {
    worker.integer_solutions.clear();
  }
}

template <typename i_t, typename f_t>
template <typename PoolT>
void branch_and_bound_t<i_t, f_t>::deterministic_merge_pseudo_cost_updates(PoolT& pool)
{
  std::vector<pseudo_cost_update_t<i_t, f_t>> all_pc_updates;
  for (auto& worker : pool) {
    auto updates = worker.pc_snapshot.take_updates();
    all_pc_updates.insert(all_pc_updates.end(), updates.begin(), updates.end());
  }
  std::sort(all_pc_updates.begin(), all_pc_updates.end());
  pc_.merge_updates(all_pc_updates);
}

template <typename i_t, typename f_t>
template <typename PoolT>
void branch_and_bound_t<i_t, f_t>::deterministic_broadcast_snapshots(
  PoolT& pool, const std::vector<f_t>& incumbent_snapshot)
{
  deterministic_snapshot_t<i_t, f_t> snap;
  snap.upper_bound    = upper_bound_.load();
  snap.total_lp_iters = exploration_stats_.total_lp_iters.load();
  snap.incumbent      = incumbent_snapshot;
  snap.pc_snapshot    = pc_.create_snapshot();

  for (auto& worker : pool) {
    worker.set_snapshots(snap);
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::deterministic_sort_replay_events(
  const bb_event_batch_t<i_t, f_t>& events)
{
  // Infeasible solutions from GPU heuristics are queued for repair; process them now
  {
    std::vector<std::vector<f_t>> to_repair;
    // TODO: support repair queue in deterministic mode
    // mutex_repair_.lock();
    // if (repair_queue_.size() > 0) {
    //   to_repair = repair_queue_;
    //   repair_queue_.clear();
    // }
    // mutex_repair_.unlock();

    std::sort(to_repair.begin(),
              to_repair.end(),
              [](const std::vector<f_t>& a, const std::vector<f_t>& b) { return a < b; });

    if (to_repair.size() > 0) {
      settings_.log.debug("Deterministic sync: Attempting to repair %ld injected solutions\n",
                          to_repair.size());
      for (const std::vector<f_t>& uncrushed_solution : to_repair) {
        std::vector<f_t> crushed_solution;
        crush_primal_solution<i_t, f_t>(
          original_problem_, original_lp_, uncrushed_solution, new_slacks_, crushed_solution);
        std::vector<f_t> repaired_solution;
        f_t repaired_obj;
        bool success =
          repair_solution(edge_norms_, crushed_solution, repaired_obj, repaired_solution);
        if (success) {
          // Queue repaired solution with work unit timestamp (...workstamp?)
          mutex_heuristic_queue_.lock();
          heuristic_solution_queue_.push_back(
            {repaired_obj, std::move(repaired_solution), 0, -1, 0, deterministic_current_horizon_});
          mutex_heuristic_queue_.unlock();
        }
      }
    }
  }

  // Extract heuristic solutions, keeping future solutions for next horizon
  // Use deterministic_current_horizon_ as the upper bound (horizon_end)
  std::vector<queued_integer_solution_t<i_t, f_t>> heuristic_solutions;
  mutex_heuristic_queue_.lock();
  {
    std::vector<queued_integer_solution_t<i_t, f_t>> future_solutions;
    for (auto& sol : heuristic_solution_queue_) {
      if (sol.work_timestamp < deterministic_current_horizon_) {
        heuristic_solutions.push_back(std::move(sol));
      } else {
        future_solutions.push_back(std::move(sol));
      }
    }
    heuristic_solution_queue_ = std::move(future_solutions);
  }
  mutex_heuristic_queue_.unlock();

  // sort by work unit timestamp, with objective and solution values as tie-breakers
  std::sort(
    heuristic_solutions.begin(),
    heuristic_solutions.end(),
    [](const queued_integer_solution_t<i_t, f_t>& a, const queued_integer_solution_t<i_t, f_t>& b) {
      if (a.work_timestamp != b.work_timestamp) { return a.work_timestamp < b.work_timestamp; }
      if (a.objective != b.objective) { return a.objective < b.objective; }
      return a.solution < b.solution;  // edge-case - lexicographical comparison
    });

  // Merge B&B events and heuristic solutions for unified timeline replay
  size_t event_idx     = 0;
  size_t heuristic_idx = 0;

  while (event_idx < events.events.size() || heuristic_idx < heuristic_solutions.size()) {
    bool process_event     = false;
    bool process_heuristic = false;

    if (event_idx >= events.events.size()) {
      process_heuristic = true;
    } else if (heuristic_idx >= heuristic_solutions.size()) {
      process_event = true;
    } else {
      // Both have items - pick the one with smaller WUT
      if (events.events[event_idx].work_timestamp <=
          heuristic_solutions[heuristic_idx].work_timestamp) {
        process_event = true;
      } else {
        process_heuristic = true;
      }
    }

    if (process_event) {
      const auto& event = events.events[event_idx++];
      switch (event.type) {
        case bb_event_type_t::NODE_INTEGER:
        case bb_event_type_t::NODE_BRANCHED:
        case bb_event_type_t::NODE_FATHOMED:
        case bb_event_type_t::NODE_INFEASIBLE:
        case bb_event_type_t::NODE_NUMERICAL: break;
      }
    }

    if (process_heuristic) {
      const auto& hsol = heuristic_solutions[heuristic_idx++];

      CUOPT_LOG_TRACE(
        "Deterministic sync: Heuristic solution received at WUT %f with objective %g, current "
        "horizon %f",
        hsol.work_timestamp,
        hsol.objective,
        deterministic_current_horizon_);

      // Process heuristic solution at its correct work unit timestamp position
      f_t new_upper = std::numeric_limits<f_t>::infinity();

      if (hsol.objective < upper_bound_) {
        upper_bound_ = hsol.objective;
        incumbent_.set_incumbent_solution(hsol.objective, hsol.solution);
        new_upper = hsol.objective;
      }

      if (new_upper < std::numeric_limits<f_t>::infinity()) {
        report_heuristic(new_upper);

        if (settings_.solution_callback != nullptr) {
          std::vector<f_t> original_x;
          uncrush_primal_solution(original_problem_, original_lp_, hsol.solution, original_x);
          settings_.solution_callback(original_x, hsol.objective);
        }
      }
    }
  }

  // Merge integer solutions from BFS workers and update global incumbent
  deterministic_process_worker_solutions(*deterministic_workers_,
                                         [](const deterministic_bfs_worker_pool_t<i_t, f_t>&, int) {
                                           return search_strategy_t::BEST_FIRST;
                                         });

  // Merge and apply pseudo-cost updates from BFS workers
  deterministic_merge_pseudo_cost_updates(*deterministic_workers_);

  for (const auto& worker : *deterministic_workers_) {
    fetch_min(lower_bound_ceiling_, worker.local_lower_bound_ceiling);
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::deterministic_prune_worker_nodes_vs_incumbent()
{
  f_t upper_bound = upper_bound_.load();

  for (auto& worker : *deterministic_workers_) {
    // Check nodes in plunge stack - filter in place
    {
      std::deque<mip_node_t<i_t, f_t>*> surviving;
      for (auto* node : worker.plunge_stack) {
        if (node->lower_bound >= upper_bound) {
          search_tree_.update(node, node_status_t::FATHOMED);
          --exploration_stats_.nodes_unexplored;
        } else {
          surviving.push_back(node);
        }
      }
      worker.plunge_stack = std::move(surviving);
    }

    // Check nodes in backlog heap - filter and rebuild
    {
      std::vector<mip_node_t<i_t, f_t>*> surviving;
      for (auto* node : worker.backlog.data()) {
        if (node->lower_bound >= upper_bound) {
          search_tree_.update(node, node_status_t::FATHOMED);
          --exploration_stats_.nodes_unexplored;
        } else {
          surviving.push_back(node);
        }
      }
      worker.backlog.clear();
      for (auto* node : surviving) {
        worker.backlog.push(node);
      }
    }
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::deterministic_balance_worker_loads()
{
  const size_t num_workers = deterministic_workers_->size();
  if (num_workers <= 1) return;

  constexpr bool force_rebalance_every_sync = false;

  // Count work for each worker: current_node (if any) + plunge_stack + backlog
  std::vector<size_t> work_counts(num_workers);
  size_t total_work = 0;
  size_t max_work   = 0;
  size_t min_work   = std::numeric_limits<size_t>::max();

  for (size_t w = 0; w < num_workers; ++w) {
    auto& worker   = (*deterministic_workers_)[w];
    work_counts[w] = worker.queue_size();
    total_work += work_counts[w];
    max_work = std::max(max_work, work_counts[w]);
    min_work = std::min(min_work, work_counts[w]);
  }
  if (total_work == 0) return;

  bool needs_balance;
  if (force_rebalance_every_sync) {
    needs_balance = (total_work > 1);
  } else {
    needs_balance = (min_work == 0 && max_work >= 2) || (min_work > 0 && max_work > 4 * min_work);
  }

  if (!needs_balance) return;

  std::vector<mip_node_t<i_t, f_t>*> all_nodes;
  for (auto& worker : *deterministic_workers_) {
    for (auto* node : worker.backlog.data()) {
      all_nodes.push_back(node);
    }
    worker.backlog.clear();
  }

  if (all_nodes.empty()) return;

  auto deterministic_less = [](const mip_node_t<i_t, f_t>* a, const mip_node_t<i_t, f_t>* b) {
    if (a->origin_worker_id != b->origin_worker_id) {
      return a->origin_worker_id < b->origin_worker_id;
    }
    return a->creation_seq < b->creation_seq;
  };
  std::sort(all_nodes.begin(), all_nodes.end(), deterministic_less);

  // Distribute nodes
  for (size_t i = 0; i < all_nodes.size(); ++i) {
    size_t worker_idx = i % num_workers;
    (*deterministic_workers_)[worker_idx].enqueue_node(all_nodes[i]);
  }
}

template <typename i_t, typename f_t>
f_t branch_and_bound_t<i_t, f_t>::deterministic_compute_lower_bound()
{
  // Compute lower bound from BFS worker local structures only
  f_t lower_bound = lower_bound_ceiling_.load();

  // Check all BFS worker queues
  for (const auto& worker : *deterministic_workers_) {
    // Check paused node (current_node)
    if (worker.current_node != nullptr) {
      lower_bound = std::min(worker.current_node->lower_bound, lower_bound);
    }

    // Check plunge stack nodes
    for (auto* node : worker.plunge_stack) {
      lower_bound = std::min(node->lower_bound, lower_bound);
    }

    // Check backlog heap nodes
    for (auto* node : worker.backlog.data()) {
      lower_bound = std::min(node->lower_bound, lower_bound);
    }
  }

  // Tree is exhausted
  if (lower_bound == std::numeric_limits<f_t>::infinity() && incumbent_.has_incumbent) {
    lower_bound = upper_bound_.load();
  }

  return lower_bound;
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::deterministic_populate_diving_heap()
{
  // Clear diving heap from previous horizon
  diving_heap_.clear();

  if (!deterministic_diving_workers_ || deterministic_diving_workers_->size() == 0) return;

  const int num_diving                  = deterministic_diving_workers_->size();
  constexpr int target_nodes_per_worker = 10;
  const int target_total                = num_diving * target_nodes_per_worker;
  f_t upper_bound                       = upper_bound_.load();

  // Collect candidate nodes from BFS worker backlog heaps
  std::vector<std::pair<mip_node_t<i_t, f_t>*, f_t>> candidates;

  for (auto& worker : *deterministic_workers_) {
    for (auto* node : worker.backlog.data()) {
      if (node->lower_bound < upper_bound) {
        f_t score = node->objective_estimate;
        if (score >= inf) { score = node->lower_bound; }
        candidates.push_back({node, score});
      }
    }
  }

  if (candidates.empty()) return;

  // Technically not necessary as it stands since the worker assignments and ordering are
  // deterministic
  std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
    if (a.second != b.second) return a.second < b.second;
    if (a.first->origin_worker_id != b.first->origin_worker_id) {
      return a.first->origin_worker_id < b.first->origin_worker_id;
    }
    return a.first->creation_seq < b.first->creation_seq;
  });

  int nodes_to_take = std::min(target_total, (int)candidates.size());

  for (int i = 0; i < nodes_to_take; ++i) {
    diving_heap_.push({candidates[i].first, candidates[i].second});
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::deterministic_assign_diving_nodes()
{
  if (!deterministic_diving_workers_ || deterministic_diving_workers_->size() == 0) {
    diving_heap_.clear();
    return;
  }

  constexpr int target_nodes_per_worker = 10;

  // Round-robin assignment
  int worker_idx        = 0;
  const int num_workers = deterministic_diving_workers_->size();

  while (!diving_heap_.empty()) {
    auto& worker = (*deterministic_diving_workers_)[worker_idx];
    worker_idx   = (worker_idx + 1) % num_workers;

    // Skip workers that already have enough nodes
    if ((int)worker.dive_queue_size() >= target_nodes_per_worker) {
      bool all_full = true;
      for (auto& w : *deterministic_diving_workers_) {
        if ((int)w.dive_queue_size() < target_nodes_per_worker) {
          all_full = false;
          break;
        }
      }
      if (all_full) break;  // all workers have enough nodes, stop assigning
      continue;             // this worker is full, try next one
    }

    auto entry = diving_heap_.pop();
    if (entry.has_value()) { worker.enqueue_dive_node(entry.value().node, original_lp_); }
  }

  diving_heap_.clear();
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::deterministic_collect_diving_solutions_and_update_pseudocosts()
{
  if (!deterministic_diving_workers_) return;

  // Collect integer solutions from diving workers and update global incumbent
  deterministic_process_worker_solutions(
    *deterministic_diving_workers_,
    [](const deterministic_diving_worker_pool_t<i_t, f_t>& pool, int worker_id) {
      return pool[worker_id].diving_type;
    });

  // Merge pseudo-cost updates from diving workers
  deterministic_merge_pseudo_cost_updates(*deterministic_diving_workers_);
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::run_deterministic_diving_loop(
  deterministic_diving_worker_t<i_t, f_t>& worker)
{
  raft::common::nvtx::range scope("BB::diving_worker_loop");

  while (deterministic_global_termination_status_ == mip_status_t::UNSET) {
    // Process dives from queue until empty or horizon exhausted
    auto entry_opt = worker.dequeue_dive_node();
    if (entry_opt.has_value()) {
      deterministic_dive(worker, std::move(entry_opt.value()));
      continue;
    }

    // Queue empty - wait for next sync point where we'll be assigned new nodes
    f_t nowork_start = tic();
    deterministic_scheduler_->wait_for_next_sync(worker.work_context);
    worker.total_nowork_time += toc(nowork_start);
    // Termination status is checked in loop condition
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::deterministic_dive(
  deterministic_diving_worker_t<i_t, f_t>& worker, dive_queue_entry_t<i_t, f_t> entry)
{
  raft::common::nvtx::range scope("BB::deterministic_dive");

  // Create local search tree for the dive
  search_tree_t<i_t, f_t> dive_tree(std::move(entry.node));
  std::deque<mip_node_t<i_t, f_t>*> stack;
  stack.push_front(&dive_tree.root);

  worker.dive_lower = std::move(entry.resolved_lower);
  worker.dive_upper = std::move(entry.resolved_upper);

  const i_t max_nodes_per_dive      = settings_.diving_settings.node_limit;
  const i_t max_backtrack_depth     = settings_.diving_settings.backtrack_limit;
  i_t nodes_this_dive               = 0;
  worker.lp_iters_this_dive         = 0;
  worker.recompute_bounds_and_basis = true;

  while (!stack.empty() && deterministic_global_termination_status_ == mip_status_t::UNSET &&
         nodes_this_dive < max_nodes_per_dive) {
    mip_node_t<i_t, f_t>* node_ptr = stack.front();
    stack.pop_front();

    // Prune check using snapshot upper bound
    f_t rel_gap = user_relative_gap(original_lp_, worker.local_upper_bound, node_ptr->lower_bound);
    if (node_ptr->lower_bound > worker.local_upper_bound ||
        rel_gap < settings_.relative_mip_gap_tol) {
      worker.recompute_bounds_and_basis = true;
      continue;
    }

    // Setup bounds for this node
    std::fill(worker.bounds_changed.begin(), worker.bounds_changed.end(), false);

    if (worker.recompute_bounds_and_basis) {
      worker.leaf_problem.lower = worker.dive_lower;
      worker.leaf_problem.upper = worker.dive_upper;
      node_ptr->get_variable_bounds(
        worker.leaf_problem.lower, worker.leaf_problem.upper, worker.bounds_changed);
    } else {
      node_ptr->update_branched_variable_bounds(
        worker.leaf_problem.lower, worker.leaf_problem.upper, worker.bounds_changed);
    }

    double remaining_time = settings_.time_limit - toc(exploration_stats_.start_time);
    if (remaining_time <= 0) { break; }

    // Setup LP settings
    simplex_solver_settings_t<i_t, f_t> lp_settings = settings_;
    lp_settings.set_log(false);
    lp_settings.cut_off       = worker.local_upper_bound + settings_.dual_tol;
    lp_settings.inside_mip    = 2;
    lp_settings.time_limit    = remaining_time;
    lp_settings.scale_columns = false;

#ifndef DETERMINISM_DISABLE_BOUNDS_STRENGTHENING
    bool feasible = worker.node_presolver.bounds_strengthening(
      lp_settings, worker.bounds_changed, worker.leaf_problem.lower, worker.leaf_problem.upper);

    if (settings_.deterministic) {
      // TEMP APPROXIMATION;
      worker.work_context.record_work_sync_on_horizon(worker.node_presolver.last_nnz_processed /
                                                      1e8);
    }

    if (!feasible) {
      worker.recompute_bounds_and_basis = true;
      continue;
    }
#endif

    {
      f_t factor                  = settings_.diving_settings.iteration_limit_factor;
      i_t max_iter                = (i_t)(factor * worker.total_lp_iters_snapshot);
      lp_settings.iteration_limit = max_iter - worker.lp_iters_this_dive;
      if (lp_settings.iteration_limit <= 0) { break; }
    }

    // Solve LP relaxation
    worker.leaf_solution.resize(worker.leaf_problem.num_rows, worker.leaf_problem.num_cols);
    std::vector<variable_status_t>& leaf_vstatus = node_ptr->vstatus;
    i_t node_iter                                = 0;
    f_t lp_start_time                            = tic();
    std::vector<f_t> leaf_edge_norms             = edge_norms_;

    dual::status_t lp_status = dual_phase2_with_advanced_basis(2,
                                                               0,
                                                               worker.recompute_bounds_and_basis,
                                                               lp_start_time,
                                                               worker.leaf_problem,
                                                               lp_settings,
                                                               leaf_vstatus,
                                                               worker.basis_factors,
                                                               worker.basic_list,
                                                               worker.nonbasic_list,
                                                               worker.leaf_solution,
                                                               node_iter,
                                                               leaf_edge_norms,
                                                               &worker.work_context);

    if (lp_status == dual::status_t::NUMERICAL) {
      lp_status_t second_status = solve_linear_program_with_advanced_basis(worker.leaf_problem,
                                                                           lp_start_time,
                                                                           lp_settings,
                                                                           worker.leaf_solution,
                                                                           worker.basis_factors,
                                                                           worker.basic_list,
                                                                           worker.nonbasic_list,
                                                                           leaf_vstatus,
                                                                           leaf_edge_norms,
                                                                           &worker.work_context);
      lp_status                 = convert_lp_status_to_dual_status(second_status);
    }

    ++nodes_this_dive;
    ++worker.total_nodes_explored;
    worker.lp_iters_this_dive += node_iter;

    worker.clock = worker.work_context.global_work_units_elapsed;

    if (lp_status == dual::status_t::TIME_LIMIT || lp_status == dual::status_t::WORK_LIMIT ||
        lp_status == dual::status_t::ITERATION_LIMIT) {
      break;
    }

    deterministic_diving_policy_t<i_t, f_t> policy{*this, worker, stack, max_backtrack_depth};
    update_tree_impl(node_ptr, dive_tree, &worker, lp_status, policy);
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template class branch_and_bound_t<int, double>;

#endif

}  // namespace cuopt::linear_programming::dual_simplex
