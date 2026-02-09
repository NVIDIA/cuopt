/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/branch_and_bound.hpp>

#include <dual_simplex/basis_solves.hpp>
#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/crossover.hpp>
#include <dual_simplex/cuts.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/logger.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/pseudo_costs.hpp>
#include <dual_simplex/random.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <dual_simplex/user_problem.hpp>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <future>
#include <limits>
#include <optional>
#include <string>
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
  char symbol, f_t obj, f_t lower_bound, i_t node_depth, i_t node_int_infeas)
{
  update_user_bound(lower_bound);
  const i_t nodes_explored   = exploration_stats_.nodes_explored;
  const i_t nodes_unexplored = exploration_stats_.nodes_unexplored;
  const f_t user_obj         = compute_user_objective(original_lp_, obj);
  const f_t user_lower       = compute_user_objective(original_lp_, lower_bound);
  const f_t iters            = static_cast<f_t>(exploration_stats_.total_lp_iters);
  const f_t iter_node        = nodes_explored > 0 ? iters / nodes_explored : iters;
  const std::string user_gap = user_mip_gap<f_t>(user_obj, user_lower);
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
    repair_queue_.push_back(crushed_solution);
    mutex_repair_.unlock();
  }
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
    for (const std::vector<f_t>& potential_solution : to_repair) {
      std::vector<f_t> repaired_solution;
      f_t repaired_obj;
      bool is_feasible =
        repair_solution(edge_norms_, potential_solution, repaired_obj, repaired_solution);
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
  if (solver_status_ == mip_status_t::NODE_LIMIT) {
    settings_.log.printf("Node limit reached. Stopping the solver...\n");
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

template <typename i_t, typename f_t>
dual::status_t branch_and_bound_t<i_t, f_t>::solve_node_lp(
  mip_node_t<i_t, f_t>* node_ptr,
  branch_and_bound_worker_t<i_t, f_t>* worker,
  branch_and_bound_stats_t<i_t, f_t>& stats,
  logger_t& log)
{
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
  lp_settings.cut_off       = upper_bound_ + settings_.dual_tol;
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
std::pair<node_status_t, rounding_direction_t> branch_and_bound_t<i_t, f_t>::update_tree(
  mip_node_t<i_t, f_t>* node_ptr,
  search_tree_t<i_t, f_t>& search_tree,
  branch_and_bound_worker_t<i_t, f_t>* worker,
  dual::status_t lp_status,
  logger_t& log)
{
  const f_t abs_fathom_tol                     = settings_.absolute_mip_gap_tol / 10;
  std::vector<variable_status_t>& leaf_vstatus = node_ptr->vstatus;
  lp_problem_t<i_t, f_t>& leaf_problem         = worker->leaf_problem;
  lp_solution_t<i_t, f_t>& leaf_solution       = worker->leaf_solution;

  if (lp_status == dual::status_t::DUAL_UNBOUNDED) {
    // Node was infeasible. Do not branch
    node_ptr->lower_bound = inf;
    search_tree.graphviz_node(log, node_ptr, "infeasible", 0.0);
    search_tree.update(node_ptr, node_status_t::INFEASIBLE);
    return {node_status_t::INFEASIBLE, rounding_direction_t::NONE};

  } else if (lp_status == dual::status_t::CUTOFF) {
    // Node was cut off. Do not branch
    node_ptr->lower_bound = upper_bound_;
    f_t leaf_objective    = compute_objective(leaf_problem, leaf_solution.x);
    search_tree.graphviz_node(log, node_ptr, "cut off", leaf_objective);
    search_tree.update(node_ptr, node_status_t::FATHOMED);
    return {node_status_t::FATHOMED, rounding_direction_t::NONE};

  } else if (lp_status == dual::status_t::OPTIMAL) {
    // LP was feasible
    std::vector<i_t> leaf_fractional;
    i_t leaf_num_fractional =
      fractional_variables(settings_, leaf_solution.x, var_types_, leaf_fractional);

#ifdef DEBUG_FRACTIONAL_FIXED
    // Check if any of the fractional variables were fixed to their bounds
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
          leaf_vstatus[j]);
      }
    }
#endif

    f_t leaf_objective = compute_objective(leaf_problem, leaf_solution.x);
    search_tree.graphviz_node(log, node_ptr, "lower bound", leaf_objective);
    pc_.update_pseudo_costs(node_ptr, leaf_objective);
    node_ptr->lower_bound = leaf_objective;

    if (worker->search_strategy == search_strategy_t::BEST_FIRST) {
      if (settings_.node_processed_callback != nullptr) {
        std::vector<f_t> original_x;
        uncrush_primal_solution(original_problem_, original_lp_, leaf_solution.x, original_x);
        settings_.node_processed_callback(original_x, leaf_objective);
      }
    }

    if (leaf_num_fractional == 0) {
      // Found a integer feasible solution
      add_feasible_solution(
        leaf_objective, leaf_solution.x, node_ptr->depth, worker->search_strategy);
      search_tree.graphviz_node(log, node_ptr, "integer feasible", leaf_objective);
      search_tree.update(node_ptr, node_status_t::INTEGER_FEASIBLE);
      return {node_status_t::INTEGER_FEASIBLE, rounding_direction_t::NONE};

    } else if (leaf_objective <= upper_bound_ + abs_fathom_tol) {
      // Choose fractional variable to branch on
      auto [branch_var, round_dir] = variable_selection(node_ptr, leaf_fractional, worker);

      assert(leaf_vstatus.size() == leaf_problem.num_cols);
      assert(branch_var >= 0);
      assert(round_dir != rounding_direction_t::NONE);

      // Note that the exploration thread is the only one that can insert new nodes into the heap,
      // and thus, we only need to calculate the objective estimate here (it is used for
      // sorting the nodes for diving).
      if (worker->search_strategy == search_strategy_t::BEST_FIRST) {
        logger_t pc_log;
        pc_log.log = false;
        node_ptr->objective_estimate =
          pc_.obj_estimate(leaf_fractional, leaf_solution.x, node_ptr->lower_bound, pc_log);
      }

      search_tree.branch(node_ptr,
                         branch_var,
                         leaf_solution.x[branch_var],
                         leaf_num_fractional,
                         leaf_vstatus,
                         leaf_problem,
                         log);
      search_tree.update(node_ptr, node_status_t::HAS_CHILDREN);
      return {node_status_t::HAS_CHILDREN, round_dir};

    } else {
      search_tree.graphviz_node(log, node_ptr, "fathomed", leaf_objective);
      search_tree.update(node_ptr, node_status_t::FATHOMED);
      return {node_status_t::FATHOMED, rounding_direction_t::NONE};
    }
  } else {
    if (worker->search_strategy == search_strategy_t::BEST_FIRST) {
      fetch_min(lower_bound_ceiling_, node_ptr->lower_bound);
      log.printf(
        "LP returned status %d on node %d. This indicates a numerical issue. The best bound is set "
        "to "
        "%+10.6e.\n",
        lp_status,
        node_ptr->node_id,
        compute_user_objective(original_lp_, lower_bound_ceiling_.load()));
    }

    search_tree.graphviz_node(log, node_ptr, "numerical", 0.0);
    search_tree.update(node_ptr, node_status_t::NUMERICAL);
    return {node_status_t::NUMERICAL, rounding_direction_t::NONE};
  }
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
                                  std::ref(edge_norms));
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
  logger_t log;
  log.log                             = false;
  log.log_prefix                      = settings_.log.log_prefix;
  solver_status_                      = mip_status_t::UNSET;
  is_running_                         = false;
  exploration_stats_.nodes_unexplored = 0;
  exploration_stats_.nodes_explored   = 0;
  original_lp_.A.to_compressed_row(Arow_);

  if (guess_.size() != 0) {
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

  settings_.log.printf(
    " | Explored | Unexplored |    Objective    |     Bound     | IntInf | Depth | Iter/Node |   "
    "Gap    "
    "|  Time  |\n");

  if (settings_.num_threads > 1) {
#pragma omp parallel num_threads(settings_.num_threads)
    {
#pragma omp master
      run_scheduler();
    }

  } else {
    single_threaded_solve();
  }

  is_running_ = false;

  f_t lower_bound = node_queue_.best_first_queue_size() > 0 ? node_queue_.get_lower_bound()
                                                            : search_tree_.root.lower_bound;
  set_final_solution(solution, lower_bound);
  return solver_status_;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template class branch_and_bound_t<int, double>;

#endif

}  // namespace cuopt::linear_programming::dual_simplex
