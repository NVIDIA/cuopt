/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/branch_and_bound.hpp>

#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/crossover.hpp>
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
void set_uninitialized_steepest_edge_norms(std::vector<f_t>& edge_norms)
{
  for (i_t j = 0; j < edge_norms.size(); ++j) {
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
inline char feasible_solution_symbol(bnb_worker_type_t type)
{
  switch (type) {
    case bnb_worker_type_t::BEST_FIRST: return 'B';
    case bnb_worker_type_t::COEFFICIENT_DIVING: return 'C';
    case bnb_worker_type_t::LINE_SEARCH_DIVING: return 'L';
    case bnb_worker_type_t::PSEUDOCOST_DIVING: return 'P';
    case bnb_worker_type_t::GUIDED_DIVING: return 'G';
    default: return 'U';
  }
}
#else
inline char feasible_solution_symbol(bnb_worker_type_t type)
{
  switch (type) {
    case bnb_worker_type_t::BEST_FIRST: return 'B';
    case bnb_worker_type_t::COEFFICIENT_DIVING: return 'D';
    case bnb_worker_type_t::LINE_SEARCH_DIVING: return 'D';
    case bnb_worker_type_t::PSEUDOCOST_DIVING: return 'D';
    case bnb_worker_type_t::GUIDED_DIVING: return 'D';
    default: return 'U';
  }
}
#endif

}  // namespace

template <typename i_t, typename f_t>
branch_and_bound_t<i_t, f_t>::branch_and_bound_t(
  const user_problem_t<i_t, f_t>& user_problem,
  const simplex_solver_settings_t<i_t, f_t>& solver_settings)
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
  exploration_stats_.start_time = tic();
  dualize_info_t<i_t, f_t> dualize_info;
  convert_user_problem(original_problem_, settings_, original_lp_, new_slacks_, dualize_info);
  full_variable_types(original_problem_, original_lp_, var_types_);

  upper_bound_ = inf;
}

template <typename i_t, typename f_t>
f_t branch_and_bound_t<i_t, f_t>::get_lower_bound()
{
  f_t lower_bound      = lower_bound_ceiling_.load();
  f_t heap_lower_bound = node_queue_.get_lower_bound();
  lower_bound          = std::min(heap_lower_bound, lower_bound);

  for (i_t i = 0; i < local_lower_bounds_.size(); ++i) {
    lower_bound = std::min(local_lower_bounds_[i].load(), lower_bound);
  }

  return std::isfinite(lower_bound) ? lower_bound : -inf;
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::report_heuristic(f_t obj)
{
  if (is_running) {
    f_t user_obj         = compute_user_objective(original_lp_, obj);
    f_t user_lower       = compute_user_objective(original_lp_, get_lower_bound());
    std::string user_gap = user_mip_gap<f_t>(user_obj, user_lower);

    settings_.log.printf(
      "H                            %+13.6e    %+10.6e                        %s %9.2f\n",
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
void branch_and_bound_t<i_t, f_t>::report(char symbol, f_t obj, f_t lower_bound, i_t node_depth)
{
  i_t nodes_explored   = exploration_stats_.nodes_explored;
  i_t nodes_unexplored = exploration_stats_.nodes_unexplored;
  f_t user_obj         = compute_user_objective(original_lp_, obj);
  f_t user_lower       = compute_user_objective(original_lp_, lower_bound);
  f_t iter_node        = exploration_stats_.total_lp_iters / nodes_explored;
  std::string user_gap = user_mip_gap<f_t>(user_obj, user_lower);
  settings_.log.printf("%c %10d   %10lu    %+13.6e    %+10.6e  %6d    %7.1e     %s %9.2f\n",
                       symbol,
                       nodes_explored,
                       nodes_unexplored,
                       user_obj,
                       user_lower,
                       node_depth,
                       iter_node,
                       user_gap.c_str(),
                       toc(exploration_stats_.start_time));
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::set_new_solution(const std::vector<f_t>& solution)
{
  if (solution.size() != original_problem_.num_cols) {
    settings_.log.printf(
      "Solution size mismatch %ld %d\n", solution.size(), original_problem_.num_cols);
  }
  std::vector<f_t> crushed_solution;
  crush_primal_solution<i_t, f_t>(
    original_problem_, original_lp_, solution, new_slacks_, crushed_solution);
  f_t obj             = compute_objective(original_lp_, crushed_solution);
  bool is_feasible    = false;
  bool attempt_repair = false;
  mutex_upper_.lock();
  if (obj < upper_bound_) {
    f_t primal_err;
    f_t bound_err;
    i_t num_fractional;
    is_feasible = check_guess(
      original_lp_, settings_, var_types_, crushed_solution, primal_err, bound_err, num_fractional);
    if (is_feasible) {
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
  }
  mutex_upper_.unlock();

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
                                                         bnb_worker_type_t thread_type)
{
  bool send_solution = false;

  settings_.log.debug("%c found a feasible solution with obj=%.10e.\n",
                      feasible_solution_symbol(thread_type),
                      compute_user_objective(original_lp_, leaf_objective));

  mutex_upper_.lock();
  if (leaf_objective < upper_bound_) {
    incumbent_.set_incumbent_solution(leaf_objective, leaf_solution);
    upper_bound_ = leaf_objective;
    report(feasible_solution_symbol(thread_type), leaf_objective, get_lower_bound(), leaf_depth);
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
  const std::vector<f_t>& solution,
  bnb_worker_type_t type)
{
  logger_t log;
  log.log                        = false;
  i_t branch_var                 = -1;
  rounding_direction_t round_dir = rounding_direction_t::NONE;
  std::vector<f_t> current_incumbent;

  // If there is no incumbent, use pseudocost diving instead of guided diving
  if (upper_bound_ == inf && type == bnb_worker_type_t::GUIDED_DIVING) {
    type = bnb_worker_type_t::PSEUDOCOST_DIVING;
  }

  switch (type) {
    case bnb_worker_type_t::BEST_FIRST:
      branch_var = pc_.variable_selection(fractional, solution, log);
      round_dir  = martin_criteria(solution[branch_var], root_relax_soln_.x[branch_var]);
      return {branch_var, round_dir};

    case bnb_worker_type_t::COEFFICIENT_DIVING:
      return coefficient_diving(
        original_lp_, fractional, solution, var_up_locks_, var_down_locks_, log);

    case bnb_worker_type_t::LINE_SEARCH_DIVING:
      return line_search_diving(fractional, solution, root_relax_soln_.x, log);

    case bnb_worker_type_t::PSEUDOCOST_DIVING:
      return pseudocost_diving(pc_, fractional, solution, root_relax_soln_.x, log);

    case bnb_worker_type_t::GUIDED_DIVING:
      mutex_upper_.lock();
      current_incumbent = incumbent_.x;
      mutex_upper_.unlock();
      return guided_diving(pc_, fractional, solution, current_incumbent, log);

    default:
      log.debug("Unknown variable selection method: %d\n", type);
      return {-1, rounding_direction_t::NONE};
  }
}

template <typename i_t, typename f_t>
dual::status_t branch_and_bound_t<i_t, f_t>::solve_node_lp(
  mip_node_t<i_t, f_t>* node_ptr,
  lp_problem_t<i_t, f_t>& leaf_problem,
  lp_solution_t<i_t, f_t>& leaf_solution,
  basis_update_mpf_t<i_t, f_t>& basis_factors,
  std::vector<i_t>& basic_list,
  std::vector<i_t>& nonbasic_list,
  bounds_strengthening_t<i_t, f_t>& node_presolver,
  bnb_worker_type_t thread_type,
  bool recompute_bounds_and_basis,
  const std::vector<f_t>& root_lower,
  const std::vector<f_t>& root_upper,
  bnb_stats_t<i_t, f_t>& stats,
  logger_t& log)
{
  std::vector<variable_status_t>& leaf_vstatus = node_ptr->vstatus;
  assert(leaf_vstatus.size() == leaf_problem.num_cols);

  simplex_solver_settings_t lp_settings = settings_;
  lp_settings.set_log(false);
  lp_settings.cut_off       = upper_bound_ + settings_.dual_tol;
  lp_settings.inside_mip    = 2;
  lp_settings.time_limit    = settings_.time_limit - toc(exploration_stats_.start_time);
  lp_settings.scale_columns = false;

  if (thread_type != bnb_worker_type_t::BEST_FIRST) {
    i_t bnb_lp_iters            = exploration_stats_.total_lp_iters;
    f_t factor                  = settings_.diving_settings.iteration_limit_factor;
    i_t max_iter                = factor * bnb_lp_iters;
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

  // Reset the bound_changed markers
  std::fill(node_presolver.bounds_changed.begin(), node_presolver.bounds_changed.end(), false);

  // Set the correct bounds for the leaf problem
  if (recompute_bounds_and_basis) {
    leaf_problem.lower = root_lower;
    leaf_problem.upper = root_upper;
    node_ptr->get_variable_bounds(
      leaf_problem.lower, leaf_problem.upper, node_presolver.bounds_changed);

  } else {
    node_ptr->update_branched_variable_bounds(
      leaf_problem.lower, leaf_problem.upper, node_presolver.bounds_changed);
  }

  bool feasible =
    node_presolver.bounds_strengthening(leaf_problem.lower, leaf_problem.upper, lp_settings);

  dual::status_t lp_status = dual::status_t::DUAL_UNBOUNDED;

  if (feasible) {
    i_t node_iter                    = 0;
    f_t lp_start_time                = tic();
    std::vector<f_t> leaf_edge_norms = edge_norms_;  // = node.steepest_edge_norms;

    lp_status = dual_phase2_with_advanced_basis(2,
                                                0,
                                                recompute_bounds_and_basis,
                                                lp_start_time,
                                                leaf_problem,
                                                lp_settings,
                                                leaf_vstatus,
                                                basis_factors,
                                                basic_list,
                                                nonbasic_list,
                                                leaf_solution,
                                                node_iter,
                                                leaf_edge_norms);

    if (lp_status == dual::status_t::NUMERICAL) {
      log.printf("Numerical issue node %d. Resolving from scratch.\n", node_ptr->node_id);
      lp_status_t second_status = solve_linear_program_with_advanced_basis(leaf_problem,
                                                                           lp_start_time,
                                                                           lp_settings,
                                                                           leaf_solution,
                                                                           basis_factors,
                                                                           basic_list,
                                                                           nonbasic_list,
                                                                           leaf_vstatus,
                                                                           leaf_edge_norms);

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
  lp_problem_t<i_t, f_t>& leaf_problem,
  lp_solution_t<i_t, f_t>& leaf_solution,
  bnb_worker_type_t thread_type,
  dual::status_t lp_status,
  logger_t& log)
{
  const f_t abs_fathom_tol                     = settings_.absolute_mip_gap_tol / 10;
  std::vector<variable_status_t>& leaf_vstatus = node_ptr->vstatus;

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

    f_t leaf_objective    = compute_objective(leaf_problem, leaf_solution.x);
    node_ptr->lower_bound = leaf_objective;
    search_tree.graphviz_node(log, node_ptr, "lower bound", leaf_objective);
    pc_.update_pseudo_costs(node_ptr, leaf_objective);

    if (thread_type == bnb_worker_type_t::BEST_FIRST) {
      if (settings_.node_processed_callback != nullptr) {
        std::vector<f_t> original_x;
        uncrush_primal_solution(original_problem_, original_lp_, leaf_solution.x, original_x);
        settings_.node_processed_callback(original_x, leaf_objective);
      }
    }

    if (leaf_num_fractional == 0) {
      // Found a integer feasible solution
      add_feasible_solution(leaf_objective, leaf_solution.x, node_ptr->depth, thread_type);
      search_tree.graphviz_node(log, node_ptr, "integer feasible", leaf_objective);
      search_tree.update(node_ptr, node_status_t::INTEGER_FEASIBLE);
      return {node_status_t::INTEGER_FEASIBLE, rounding_direction_t::NONE};

    } else if (leaf_objective <= upper_bound_ + abs_fathom_tol) {
      // Choose fractional variable to branch on
      auto [branch_var, round_dir] =
        variable_selection(node_ptr, leaf_fractional, leaf_solution.x, thread_type);

      assert(leaf_vstatus.size() == leaf_problem.num_cols);
      assert(branch_var >= 0);
      assert(round_dir != rounding_direction_t::NONE);

      // Note that the exploration thread is the only one that can insert new nodes into the heap,
      // and thus, we only need to calculate the objective estimate here (it is used for
      // sorting the nodes for diving).
      if (thread_type == bnb_worker_type_t::BEST_FIRST) {
        logger_t pc_log;
        pc_log.log = false;
        node_ptr->objective_estimate =
          pc_.obj_estimate(leaf_fractional, leaf_solution.x, node_ptr->lower_bound, pc_log);
      }

      search_tree.branch(
        node_ptr, branch_var, leaf_solution.x[branch_var], leaf_vstatus, leaf_problem, log);
      search_tree.update(node_ptr, node_status_t::HAS_CHILDREN);
      return {node_status_t::HAS_CHILDREN, round_dir};

    } else {
      search_tree.graphviz_node(log, node_ptr, "fathomed", leaf_objective);
      search_tree.update(node_ptr, node_status_t::FATHOMED);
      return {node_status_t::FATHOMED, rounding_direction_t::NONE};
    }
  } else {
    if (thread_type == bnb_worker_type_t::BEST_FIRST) {
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
void branch_and_bound_t<i_t, f_t>::exploration_ramp_up(mip_node_t<i_t, f_t>* node,
                                                       i_t initial_heap_size)
{
  if (solver_status_ != mip_status_t::UNSET) { return; }

  // Note that we do not know which thread will execute the
  // `exploration_ramp_up` task, so we allow to any thread
  // to repair the heuristic solution.
  repair_heuristic_solutions();

  f_t lower_bound = node->lower_bound;
  f_t upper_bound = upper_bound_;
  f_t rel_gap     = user_relative_gap(original_lp_, upper_bound, lower_bound);
  f_t abs_gap     = upper_bound - lower_bound;

  if (lower_bound > upper_bound || rel_gap < settings_.relative_mip_gap_tol) {
    search_tree_.graphviz_node(settings_.log, node, "cutoff", node->lower_bound);
    search_tree_.update(node, node_status_t::FATHOMED);
    --exploration_stats_.nodes_unexplored;
    return;
  }

  f_t now = toc(exploration_stats_.start_time);
  f_t time_since_last_log =
    exploration_stats_.last_log == 0 ? 1.0 : toc(exploration_stats_.last_log);

  if (((exploration_stats_.nodes_since_last_log >= 10 ||
        abs_gap < 10 * settings_.absolute_mip_gap_tol) &&
       (time_since_last_log >= 1)) ||
      (time_since_last_log > 30) || now > settings_.time_limit) {
    bool should_report = should_report_.exchange(false);

    if (should_report) {
      report(' ', upper_bound, root_objective_, node->depth);
      exploration_stats_.nodes_since_last_log = 0;
      exploration_stats_.last_log             = tic();
      should_report_                          = true;
    }
  }

  if (now > settings_.time_limit) {
    solver_status_ = mip_status_t::TIME_LIMIT;
    return;
  }

  // Make a copy of the original LP. We will modify its bounds at each leaf
  lp_problem_t<i_t, f_t> leaf_problem = original_lp_;
  std::vector<char> row_sense;
  bounds_strengthening_t<i_t, f_t> node_presolver(leaf_problem, Arow_, row_sense, var_types_);

  const i_t m = leaf_problem.num_rows;
  basis_update_mpf_t<i_t, f_t> basis_factors(m, settings_.refactor_frequency);
  std::vector<i_t> basic_list(m);
  std::vector<i_t> nonbasic_list;

  lp_solution_t<i_t, f_t> leaf_solution(leaf_problem.num_rows, leaf_problem.num_cols);
  dual::status_t lp_status = solve_node_lp(node,
                                           leaf_problem,
                                           leaf_solution,
                                           basis_factors,
                                           basic_list,
                                           nonbasic_list,
                                           node_presolver,
                                           bnb_worker_type_t::BEST_FIRST,
                                           true,
                                           original_lp_.lower,
                                           original_lp_.upper,
                                           exploration_stats_,
                                           settings_.log);
  if (lp_status == dual::status_t::TIME_LIMIT) {
    solver_status_ = mip_status_t::TIME_LIMIT;
    return;
  }

  ++exploration_stats_.nodes_since_last_log;
  ++exploration_stats_.nodes_explored;
  --exploration_stats_.nodes_unexplored;

  auto [node_status, round_dir] = update_tree(node,
                                              search_tree_,
                                              leaf_problem,
                                              leaf_solution,
                                              bnb_worker_type_t::BEST_FIRST,
                                              lp_status,
                                              settings_.log);

  if (node_status == node_status_t::HAS_CHILDREN) {
    exploration_stats_.nodes_unexplored += 2;

    // If we haven't generated enough nodes to keep the threads busy, continue the ramp up phase
    if (exploration_stats_.nodes_unexplored < initial_heap_size) {
#pragma omp task
      exploration_ramp_up(node->get_down_child(), initial_heap_size);

#pragma omp task
      exploration_ramp_up(node->get_up_child(), initial_heap_size);

    } else {
      // We've generated enough nodes, push further nodes onto the heap
      node_queue_.push(node->get_down_child());
      node_queue_.push(node->get_up_child());
    }
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::plunge_from(i_t task_id,
                                               mip_node_t<i_t, f_t>* start_node,
                                               lp_problem_t<i_t, f_t>& leaf_problem,
                                               bounds_strengthening_t<i_t, f_t>& node_presolver,
                                               basis_update_mpf_t<i_t, f_t>& basis_factors,
                                               std::vector<i_t>& basic_list,
                                               std::vector<i_t>& nonbasic_list)
{
  bool recompute_bounds_and_basis = true;
  std::deque<mip_node_t<i_t, f_t>*> stack;
  stack.push_front(start_node);

  while (stack.size() > 0 && solver_status_ == mip_status_t::UNSET && is_running) {
    if (task_id == 0) { repair_heuristic_solutions(); }

    mip_node_t<i_t, f_t>* node_ptr = stack.front();
    stack.pop_front();

    f_t lower_bound = node_ptr->lower_bound;
    f_t upper_bound = upper_bound_;
    f_t abs_gap     = upper_bound - lower_bound;
    f_t rel_gap     = user_relative_gap(original_lp_, upper_bound, lower_bound);

    // This is based on three assumptions:
    // - The stack only contains sibling nodes, i.e., the current node and it sibling (if
    // applicable)
    // - The current node and its siblings uses the lower bound of the parent before solving the LP
    // relaxation
    // - The lower bound of the parent is lower or equal to its children
    assert(task_id < local_lower_bounds_.size());
    local_lower_bounds_[task_id] = lower_bound;

    if (lower_bound > upper_bound || rel_gap < settings_.relative_mip_gap_tol) {
      search_tree_.graphviz_node(settings_.log, node_ptr, "cutoff", node_ptr->lower_bound);
      search_tree_.update(node_ptr, node_status_t::FATHOMED);
      recompute_bounds_and_basis = true;
      --exploration_stats_.nodes_unexplored;
      continue;
    }

    f_t now = toc(exploration_stats_.start_time);

    if (task_id == 0) {
      f_t time_since_last_log =
        exploration_stats_.last_log == 0 ? 1.0 : toc(exploration_stats_.last_log);

      if (((exploration_stats_.nodes_since_last_log >= 1000 ||
            abs_gap < 10 * settings_.absolute_mip_gap_tol) &&
           time_since_last_log >= 1) ||
          (time_since_last_log > 30) || now > settings_.time_limit) {
        report(' ', upper_bound, get_lower_bound(), node_ptr->depth);
        exploration_stats_.last_log             = tic();
        exploration_stats_.nodes_since_last_log = 0;
      }
    }

    if (now > settings_.time_limit) {
      solver_status_ = mip_status_t::TIME_LIMIT;
      break;
    }
    if (exploration_stats_.nodes_explored >= settings_.node_limit) {
      solver_status_ = mip_status_t::NODE_LIMIT;
      break;
    }

    lp_solution_t<i_t, f_t> leaf_solution(leaf_problem.num_rows, leaf_problem.num_cols);
    dual::status_t lp_status = solve_node_lp(node_ptr,
                                             leaf_problem,
                                             leaf_solution,
                                             basis_factors,
                                             basic_list,
                                             nonbasic_list,
                                             node_presolver,
                                             bnb_worker_type_t::BEST_FIRST,
                                             recompute_bounds_and_basis,
                                             original_lp_.lower,
                                             original_lp_.upper,
                                             exploration_stats_,
                                             settings_.log);

    if (lp_status == dual::status_t::TIME_LIMIT) {
      solver_status_ = mip_status_t::TIME_LIMIT;
      break;
    } else if (lp_status == dual::status_t::ITERATION_LIMIT) {
      break;
    }

    ++exploration_stats_.nodes_since_last_log;
    ++exploration_stats_.nodes_explored;
    --exploration_stats_.nodes_unexplored;

    auto [node_status, round_dir] = update_tree(node_ptr,
                                                search_tree_,
                                                leaf_problem,
                                                leaf_solution,
                                                bnb_worker_type_t::BEST_FIRST,
                                                lp_status,
                                                settings_.log);

    recompute_bounds_and_basis = node_status != node_status_t::HAS_CHILDREN;

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
        stack.push_front(node_ptr->get_down_child());
        stack.push_front(node_ptr->get_up_child());
      } else {
        stack.push_front(node_ptr->get_up_child());
        stack.push_front(node_ptr->get_down_child());
      }
    }
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::best_first_thread(i_t task_id)
{
  f_t lower_bound = -inf;
  f_t abs_gap     = inf;
  f_t rel_gap     = inf;

  // Make a copy of the original LP. We will modify its bounds at each leaf
  lp_problem_t<i_t, f_t> leaf_problem = original_lp_;
  std::vector<char> row_sense;
  bounds_strengthening_t<i_t, f_t> node_presolver(leaf_problem, Arow_, row_sense, var_types_);

  const i_t m = leaf_problem.num_rows;
  basis_update_mpf_t<i_t, f_t> basis_factors(m, settings_.refactor_frequency);
  std::vector<i_t> basic_list(m);
  std::vector<i_t> nonbasic_list;

  while (solver_status_ == mip_status_t::UNSET && abs_gap > settings_.absolute_mip_gap_tol &&
         rel_gap > settings_.relative_mip_gap_tol &&
         (active_subtrees_ > 0 || node_queue_.best_first_queue_size() > 0)) {
    // In the current implementation, we are use the active number of subtree to decide
    // when to stop the execution. We need to increment the counter at the same
    // time as we pop a node from the queue to avoid some threads exiting
    // the main loop thinking that the solver has already finished.
    // This will be not needed in the master-worker model.
    node_queue_.lock();
    // If there any node left in the heap, we pop the top node and explore it.
    std::optional<mip_node_t<i_t, f_t>*> start_node = node_queue_.pop_best_first();
    if (start_node.has_value()) { active_subtrees_++; };
    node_queue_.unlock();

    if (start_node.has_value()) {
      if (upper_bound_ < start_node.value()->lower_bound) {
        // This node was put on the heap earlier but its lower bound is now greater than the
        // current upper bound
        search_tree_.graphviz_node(
          settings_.log, start_node.value(), "cutoff", start_node.value()->lower_bound);
        search_tree_.update(start_node.value(), node_status_t::FATHOMED);
        active_subtrees_--;
        continue;
      }

      // Best-first search with plunging
      plunge_from(task_id,
                  start_node.value(),
                  leaf_problem,
                  node_presolver,
                  basis_factors,
                  basic_list,
                  nonbasic_list);

      active_subtrees_--;
    }

    lower_bound = get_lower_bound();
    abs_gap     = upper_bound_ - lower_bound;
    rel_gap     = user_relative_gap(original_lp_, upper_bound_.load(), lower_bound);
  }

  is_running = false;

  // Check if it is the last thread that exited the loop and no
  // timeout or numerical error has happen.
  if (solver_status_ == mip_status_t::UNSET) {
    if (active_subtrees_ > 0) { local_lower_bounds_[task_id] = inf; }
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::dive_from(mip_node_t<i_t, f_t>& start_node,
                                             const std::vector<f_t>& start_lower,
                                             const std::vector<f_t>& start_upper,
                                             lp_problem_t<i_t, f_t>& leaf_problem,
                                             bounds_strengthening_t<i_t, f_t>& node_presolver,
                                             basis_update_mpf_t<i_t, f_t>& basis_factors,
                                             std::vector<i_t>& basic_list,
                                             std::vector<i_t>& nonbasic_list,
                                             bnb_worker_type_t diving_type)
{
  logger_t log;
  log.log = false;

  const i_t diving_node_limit      = settings_.diving_settings.node_limit;
  const i_t diving_backtrack_limit = settings_.diving_settings.backtrack_limit;
  bool recompute_bounds_and_basis  = true;
  search_tree_t<i_t, f_t> dive_tree(std::move(start_node));
  std::deque<mip_node_t<i_t, f_t>*> stack;
  stack.push_front(&dive_tree.root);

  bnb_stats_t<i_t, f_t> dive_stats;
  dive_stats.total_lp_iters      = 0;
  dive_stats.total_lp_solve_time = 0;
  dive_stats.nodes_explored      = 0;
  dive_stats.nodes_unexplored    = 1;

  while (stack.size() > 0 && solver_status_ == mip_status_t::UNSET && is_running) {
    mip_node_t<i_t, f_t>* node_ptr = stack.front();
    stack.pop_front();

    f_t lower_bound = node_ptr->lower_bound;
    f_t upper_bound = upper_bound_;
    f_t rel_gap     = user_relative_gap(original_lp_, upper_bound, lower_bound);

    if (node_ptr->lower_bound > upper_bound || rel_gap < settings_.relative_mip_gap_tol) {
      recompute_bounds_and_basis = true;
      continue;
    }

    if (toc(exploration_stats_.start_time) > settings_.time_limit) { break; }
    if (dive_stats.nodes_explored > diving_node_limit) { break; }

    lp_solution_t<i_t, f_t> leaf_solution(leaf_problem.num_rows, leaf_problem.num_cols);
    dual::status_t lp_status = solve_node_lp(node_ptr,
                                             leaf_problem,
                                             leaf_solution,
                                             basis_factors,
                                             basic_list,
                                             nonbasic_list,
                                             node_presolver,
                                             diving_type,
                                             recompute_bounds_and_basis,
                                             start_lower,
                                             start_upper,
                                             dive_stats,
                                             log);

    if (lp_status == dual::status_t::TIME_LIMIT) {
      solver_status_ = mip_status_t::TIME_LIMIT;
      break;
    } else if (lp_status == dual::status_t::ITERATION_LIMIT) {
      break;
    }

    ++dive_stats.nodes_explored;

    auto [node_status, round_dir] =
      update_tree(node_ptr, dive_tree, leaf_problem, leaf_solution, diving_type, lp_status, log);
    recompute_bounds_and_basis = node_status != node_status_t::HAS_CHILDREN;

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
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::diving_thread(bnb_worker_type_t diving_type)
{
  // Make a copy of the original LP. We will modify its bounds at each leaf
  lp_problem_t<i_t, f_t> leaf_problem = original_lp_;
  std::vector<char> row_sense;
  bounds_strengthening_t<i_t, f_t> node_presolver(leaf_problem, Arow_, row_sense, var_types_);

  const i_t m = leaf_problem.num_rows;
  basis_update_mpf_t<i_t, f_t> basis_factors(m, settings_.refactor_frequency);
  std::vector<i_t> basic_list(m);
  std::vector<i_t> nonbasic_list;

  std::vector<f_t> start_lower;
  std::vector<f_t> start_upper;
  bool reset_starting_bounds = true;

  while (solver_status_ == mip_status_t::UNSET && is_running &&
         (active_subtrees_ > 0 || node_queue_.best_first_queue_size() > 0)) {
    if (reset_starting_bounds) {
      start_lower = original_lp_.lower;
      start_upper = original_lp_.upper;
      std::fill(node_presolver.bounds_changed.begin(), node_presolver.bounds_changed.end(), false);
      reset_starting_bounds = false;
    }

    // In the current implementation, multiple threads can pop the nodes
    // from the queue, so we need to initialize the lower and upper bound here
    // to avoid other thread fathoming the node (i.e., deleting) before we can read
    // the variable bounds from the tree.
    // This will be not needed in the master-worker model.
    node_queue_.lock();
    std::optional<mip_node_t<i_t, f_t>*> node_ptr  = node_queue_.pop_diving();
    std::optional<mip_node_t<i_t, f_t>> start_node = std::nullopt;

    if (node_ptr.has_value()) {
      node_ptr.value()->get_variable_bounds(
        start_lower, start_upper, node_presolver.bounds_changed);
      start_node = node_ptr.value()->detach_copy();
    }
    node_queue_.unlock();

    if (start_node.has_value()) {
      reset_starting_bounds = true;

      if (upper_bound_ < start_node->lower_bound) { continue; }
      bool is_feasible = node_presolver.bounds_strengthening(start_lower, start_upper, settings_);
      if (!is_feasible) { continue; }

      dive_from(start_node.value(),
                start_lower,
                start_upper,
                leaf_problem,
                node_presolver,
                basis_factors,
                basic_list,
                nonbasic_list,
                diving_type);
    }
  }
}

template <typename i_t, typename f_t>
lp_status_t branch_and_bound_t<i_t, f_t>::solve_root_relaxation(
  simplex_solver_settings_t<i_t, f_t> const& lp_settings)
{
  f_t start_time          = tic();
  f_t user_objective      = 0;
  i_t iter                = 0;
  std::string solver_name = "";

  // Root node path
  lp_status_t root_status;
  std::future<lp_status_t> root_status_future;
  root_status_future = std::async(std::launch::async,
                                  &solve_linear_program_advanced<i_t, f_t>,
                                  std::ref(original_lp_),
                                  exploration_stats_.start_time,
                                  std::ref(lp_settings),
                                  std::ref(root_relax_soln_),
                                  std::ref(root_vstatus_),
                                  std::ref(edge_norms_));
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
      set_root_concurrent_halt(1);  // Stop dual simplex
      root_status = root_status_future.get();

      // Override the root relaxation solution with the crossover solution
      root_relax_soln_ = root_crossover_soln_;
      root_vstatus_    = crossover_vstatus_;
      root_status      = lp_status_t::OPTIMAL;
      user_objective   = root_crossover_soln_.user_objective;
      iter             = root_crossover_soln_.iterations;
      solver_name      = "Barrier/PDLP and Crossover";

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
    settings_.log.printf("Root relaxation returned status: %s\n",
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
  is_running                          = false;
  exploration_stats_.nodes_unexplored = 0;
  exploration_stats_.nodes_explored   = 0;
  original_lp_.A.to_compressed_row(Arow_);

  std::vector<bnb_worker_type_t> diving_strategies;
  diving_strategies.reserve(4);

  if (settings_.diving_settings.pseudocost_diving != 0) {
    diving_strategies.push_back(bnb_worker_type_t::PSEUDOCOST_DIVING);
  }

  if (settings_.diving_settings.line_search_diving != 0) {
    diving_strategies.push_back(bnb_worker_type_t::LINE_SEARCH_DIVING);
  }

  if (settings_.diving_settings.guided_diving != 0) {
    diving_strategies.push_back(bnb_worker_type_t::GUIDED_DIVING);
  }

  if (settings_.diving_settings.coefficient_diving != 0) {
    diving_strategies.push_back(bnb_worker_type_t::COEFFICIENT_DIVING);
    calculate_variable_locks(original_lp_, var_up_locks_, var_down_locks_);
  }

  if (diving_strategies.empty()) {
    settings_.log.printf("Warning: All diving heuristics are disabled!\n");
  }

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

  lp_status_t root_status;
  simplex_solver_settings_t lp_settings = settings_;
  lp_settings.inside_mip                = 1;
  lp_settings.concurrent_halt           = get_root_concurrent_halt();
  // RINS/SUBMIP path
  if (!enable_concurrent_lp_root_solve()) {
    settings_.log.printf("\nSolving LP root relaxation with dual simplex\n");
    root_status = solve_linear_program_advanced(original_lp_,
                                                exploration_stats_.start_time,
                                                lp_settings,
                                                root_relax_soln_,
                                                root_vstatus_,
                                                edge_norms_);

  } else {
    settings_.log.printf("\nSolving LP root relaxation in concurrent mode\n");
    root_status = solve_root_relaxation(lp_settings);
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

  assert(root_vstatus_.size() == original_lp_.num_cols);
  set_uninitialized_steepest_edge_norms<i_t, f_t>(edge_norms_);

  root_objective_ = compute_objective(original_lp_, root_relax_soln_.x);
  local_lower_bounds_.assign(settings_.num_bfs_workers, root_objective_);

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
  const i_t num_fractional =
    fractional_variables(settings_, root_relax_soln_.x, var_types_, fractional);

  if (num_fractional == 0) {
    mutex_upper_.lock();
    incumbent_.set_incumbent_solution(root_objective_, root_relax_soln_.x);
    upper_bound_ = root_objective_;
    mutex_upper_.unlock();
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
    return mip_status_t::OPTIMAL;
  }

  pc_.resize(original_lp_.num_cols);
  strong_branching<i_t, f_t>(original_lp_,
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

  // Choose variable to branch on
  i_t branch_var = pc_.variable_selection(fractional, root_relax_soln_.x, log);

  search_tree_.root      = std::move(mip_node_t<i_t, f_t>(root_objective_, root_vstatus_));
  search_tree_.num_nodes = 0;
  search_tree_.graphviz_node(settings_.log, &search_tree_.root, "lower bound", root_objective_);
  search_tree_.branch(&search_tree_.root,
                      branch_var,
                      root_relax_soln_.x[branch_var],
                      root_vstatus_,
                      original_lp_,
                      log);

  settings_.log.printf("Exploring the B&B tree using %d threads (best-first = %d, diving = %d)\n\n",
                       settings_.num_threads,
                       settings_.num_bfs_workers,
                       settings_.num_threads - settings_.num_bfs_workers);

  exploration_stats_.nodes_explored       = 0;
  exploration_stats_.nodes_unexplored     = 2;
  exploration_stats_.nodes_since_last_log = 0;
  exploration_stats_.last_log             = tic();
  active_subtrees_                        = 0;
  is_running                              = true;
  lower_bound_ceiling_                    = inf;
  should_report_                          = true;

  settings_.log.printf(
    "  | Explored | Unexplored |    Objective    |     Bound     | Depth | Iter/Node |   Gap    "
    "|  Time  |\n");

#pragma omp parallel num_threads(settings_.num_threads)
  {
#pragma omp master
    {
      auto down_child          = search_tree_.root.get_down_child();
      auto up_child            = search_tree_.root.get_up_child();
      i_t initial_size         = 2 * settings_.num_threads;
      const i_t num_strategies = diving_strategies.size();

#pragma omp taskgroup
      {
#pragma omp task
        exploration_ramp_up(down_child, initial_size);

#pragma omp task
        exploration_ramp_up(up_child, initial_size);
      }

      for (i_t i = 0; i < settings_.num_bfs_workers; i++) {
#pragma omp task
        best_first_thread(i);
      }

      if (!diving_strategies.empty()) {
        for (i_t k = 0; k < settings_.diving_settings.num_diving_workers; k++) {
          const bnb_worker_type_t diving_type = diving_strategies[k % num_strategies];
#pragma omp task
          diving_thread(diving_type);
        }
      }
    }
  }

  is_running      = false;
  f_t lower_bound = node_queue_.best_first_queue_size() > 0 ? node_queue_.get_lower_bound()
                                                            : search_tree_.root.lower_bound;
  set_final_solution(solution, lower_bound);
  return solver_status_;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template class branch_and_bound_t<int, double>;

#endif

}  // namespace cuopt::linear_programming::dual_simplex
