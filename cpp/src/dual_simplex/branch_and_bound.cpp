/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <omp.h>
#include <algorithm>
#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/branch_and_bound.hpp>
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
    return "  -  ";
  } else {
    constexpr int BUFFER_LEN = 32;
    char buffer[BUFFER_LEN];
    snprintf(buffer, BUFFER_LEN - 1, "%4.1f%%", user_mip_gap * 100);
    return std::string(buffer);
  }
}

inline const char* feasible_solution_symbol(thread_type_t type)
{
  switch (type) {
    case thread_type_t::EXPLORATION: return "B";
    case thread_type_t::DIVING: return "D";
    default: return "U";
  }
}

inline bool has_children(node_solve_info_t status)
{
  return status == node_solve_info_t::UP_CHILD_FIRST ||
         status == node_solve_info_t::DOWN_CHILD_FIRST;
}

}  // namespace

template <typename i_t, typename f_t>
branch_and_bound_t<i_t, f_t>::branch_and_bound_t(
  const user_problem_t<i_t, f_t>& user_problem,
  const simplex_solver_settings_t<i_t, f_t>& solver_settings)
  : original_problem_(user_problem),
    settings_(solver_settings),
    original_lp_(user_problem.handle_ptr, 1, 1, 1),
    incumbent_(1),
    root_relax_soln_(1, 1),
    root_crossover_soln_(1, 1),
    pc_(1),
    solver_status_(mip_exploration_status_t::UNSET)
{
  exploration_stats_.start_time = tic();
  dualize_info_t<i_t, f_t> dualize_info;
  convert_user_problem(original_problem_, settings_, original_lp_, new_slacks_, dualize_info);
  full_variable_types(original_problem_, original_lp_, var_types_);

  mutex_upper_.lock();
  upper_bound_ = inf;
  mutex_upper_.unlock();
}

template <typename i_t, typename f_t>
f_t branch_and_bound_t<i_t, f_t>::get_upper_bound()
{
  mutex_upper_.lock();
  const f_t upper_bound = upper_bound_;
  mutex_upper_.unlock();
  return upper_bound;
}

template <typename i_t, typename f_t>
f_t branch_and_bound_t<i_t, f_t>::get_lower_bound()
{
  f_t lower_bound = lower_bound_ceiling_.load();
  mutex_heap_.lock();
  if (heap_.size() > 0) { lower_bound = std::min(heap_.top()->lower_bound, lower_bound); }
  mutex_heap_.unlock();

  for (i_t i = 0; i < local_lower_bounds_.size(); ++i) {
    lower_bound = std::min(local_lower_bounds_[i].load(), lower_bound);
  }

  return lower_bound;
}

template <typename i_t, typename f_t>
i_t branch_and_bound_t<i_t, f_t>::get_heap_size()
{
  mutex_heap_.lock();
  i_t size = heap_.size();
  mutex_heap_.unlock();
  return size;
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

  if (is_feasible) {
    if (solver_status_ == mip_exploration_status_t::RUNNING) {
      f_t user_obj    = compute_user_objective(original_lp_, obj);
      f_t user_lower  = compute_user_objective(original_lp_, get_lower_bound());
      std::string gap = user_mip_gap<f_t>(user_obj, user_lower);

      settings_.log.printf(
        "H                           %+13.6e    %+10.6e                        %s %9.2f\n",
        user_obj,
        user_lower,
        gap.c_str(),
        toc(exploration_stats_.start_time));
    } else {
      settings_.log.printf("New solution from primal heuristics. Objective %+.6e. Time %.2f\n",
                           compute_user_objective(original_lp_, obj),
                           toc(exploration_stats_.start_time));
    }
  }

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

          f_t obj              = compute_user_objective(original_lp_, repaired_obj);
          f_t lower            = compute_user_objective(original_lp_, get_lower_bound());
          std::string user_gap = user_mip_gap<f_t>(obj, lower);

          settings_.log.printf(
            "H                           %+13.6e    %+10.6e                        %s %9.2f\n",
            obj,
            lower,
            user_gap.c_str(),
            toc(exploration_stats_.start_time));

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
mip_status_t branch_and_bound_t<i_t, f_t>::set_final_solution(mip_solution_t<i_t, f_t>& solution,
                                                              f_t lower_bound)
{
  mip_status_t mip_status = mip_status_t::UNSET;

  if (solver_status_ == mip_exploration_status_t::NUMERICAL) {
    settings_.log.printf("Numerical issue encountered. Stopping the solver...\n");
    mip_status = mip_status_t::NUMERICAL;
  }

  if (solver_status_ == mip_exploration_status_t::TIME_LIMIT) {
    settings_.log.printf("Time limit reached. Stopping the solver...\n");
    mip_status = mip_status_t::TIME_LIMIT;
  }
  if (solver_status_ == mip_exploration_status_t::NODE_LIMIT) {
    settings_.log.printf("Node limit reached. Stopping the solver...\n");
    mip_status = mip_status_t::NODE_LIMIT;
  }

  f_t upper_bound      = get_upper_bound();
  f_t gap              = upper_bound - lower_bound;
  f_t obj              = compute_user_objective(original_lp_, upper_bound);
  f_t user_bound       = compute_user_objective(original_lp_, lower_bound);
  f_t gap_rel          = user_relative_gap(original_lp_, upper_bound, lower_bound);
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
    mip_status = mip_status_t::OPTIMAL;
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

  if (solver_status_ == mip_exploration_status_t::COMPLETED) {
    if (exploration_stats_.nodes_explored > 0 && exploration_stats_.nodes_unexplored == 0 &&
        upper_bound == inf) {
      settings_.log.printf("Integer infeasible.\n");
      mip_status = mip_status_t::INFEASIBLE;
      if (settings_.heuristic_preemption_callback != nullptr) {
        settings_.heuristic_preemption_callback();
      }
    }
  }

  if (upper_bound != inf) {
    assert(incumbent_.has_incumbent);
    uncrush_primal_solution(original_problem_, original_lp_, incumbent_.x, solution.x);
  }
  solution.objective          = incumbent_.objective;
  solution.lower_bound        = lower_bound;
  solution.nodes_explored     = exploration_stats_.nodes_explored;
  solution.simplex_iterations = exploration_stats_.total_lp_iters;

  return mip_status;
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::add_feasible_solution(f_t leaf_objective,
                                                         const std::vector<f_t>& leaf_solution,
                                                         i_t leaf_depth,
                                                         thread_type_t thread_type)
{
  bool send_solution   = false;
  i_t nodes_explored   = exploration_stats_.nodes_explored;
  i_t nodes_unexplored = exploration_stats_.nodes_unexplored;

  mutex_upper_.lock();
  if (leaf_objective < upper_bound_) {
    incumbent_.set_incumbent_solution(leaf_objective, leaf_solution);
    upper_bound_    = leaf_objective;
    f_t lower_bound = get_lower_bound();
    f_t obj         = compute_user_objective(original_lp_, upper_bound_);
    f_t lower       = compute_user_objective(original_lp_, lower_bound);
    settings_.log.printf(
      "%s%10d   %10lu    %+13.6e    %+10.6e   %6d   %7.1e     %s %9.2f\n",
      feasible_solution_symbol(thread_type),
      nodes_explored,
      nodes_unexplored,
      obj,
      lower,
      leaf_depth,
      nodes_explored > 0 ? exploration_stats_.total_lp_iters / nodes_explored : 0,
      user_mip_gap<f_t>(obj, lower).c_str(),
      toc(exploration_stats_.start_time));

    send_solution = true;
  }

  if (send_solution && settings_.solution_callback != nullptr) {
    std::vector<f_t> original_x;
    uncrush_primal_solution(original_problem_, original_lp_, incumbent_.x, original_x);
    settings_.solution_callback(original_x, upper_bound_);
  }
  mutex_upper_.unlock();
}

template <typename i_t, typename f_t>
rounding_direction_t branch_and_bound_t<i_t, f_t>::child_selection(mip_node_t<i_t, f_t>* node_ptr)
{
  const i_t branch_var     = node_ptr->get_down_child()->branch_var;
  const f_t branch_var_val = node_ptr->get_down_child()->fractional_val;
  const f_t down_val       = std::floor(root_relax_soln_.x[branch_var]);
  const f_t up_val         = std::ceil(root_relax_soln_.x[branch_var]);
  const f_t down_dist      = branch_var_val - down_val;
  const f_t up_dist        = up_val - branch_var_val;
  constexpr f_t eps        = 1e-6;

  if (down_dist < up_dist + eps) {
    return rounding_direction_t::DOWN;

  } else {
    return rounding_direction_t::UP;
  }
}

template <typename i_t, typename f_t>
node_solve_info_t branch_and_bound_t<i_t, f_t>::solve_node(
  mip_node_t<i_t, f_t>* node_ptr,
  search_tree_t<i_t, f_t>& search_tree,
  lp_problem_t<i_t, f_t>& leaf_problem,
  basis_update_mpf_t<i_t, f_t>& basis_factors,
  std::vector<i_t>& basic_list,
  std::vector<i_t>& nonbasic_list,
  bounds_strengthening_t<i_t, f_t>& node_presolver,
  thread_type_t thread_type,
  bool recompute_bounds_and_basis,
  const std::vector<f_t>& root_lower,
  const std::vector<f_t>& root_upper,
  logger_t& log)
{
  const f_t abs_fathom_tol = settings_.absolute_mip_gap_tol / 10;
  const f_t upper_bound    = get_upper_bound();

  lp_solution_t<i_t, f_t> leaf_solution(leaf_problem.num_rows, leaf_problem.num_cols);
  std::vector<variable_status_t>& leaf_vstatus = node_ptr->vstatus;
  assert(leaf_vstatus.size() == leaf_problem.num_cols);

  simplex_solver_settings_t lp_settings = settings_;
  lp_settings.set_log(false);
  lp_settings.cut_off       = upper_bound + settings_.dual_tol;
  lp_settings.inside_mip    = 2;
  lp_settings.time_limit    = settings_.time_limit - toc(exploration_stats_.start_time);
  lp_settings.scale_columns = false;

#ifdef LOG_NODE_SIMPLEX
  lp_settings.set_log(true);
  std::stringstream ss;
  ss << "simplex-" << std::this_thread::get_id() << ".log";
  std::string logname;
  ss >> logname;
  lp_settings.set_log_filename(logname);
  lp_settings.log.enable_log_to_file("a+");
  lp_settings.log.log_to_console = false;
  lp_settings.log.printf(
    "%s node id = %d, branch var = %d, fractional val = %f, variable lower bound = %f, variable "
    "upper bound = %f\n",
    settings_.log.log_prefix.c_str(),
    node_ptr->node_id,
    node_ptr->branch_var,
    node_ptr->fractional_val,
    node_ptr->branch_var_lower,
    node_ptr->branch_var_upper);
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

    if (thread_type == thread_type_t::EXPLORATION) {
      exploration_stats_.total_lp_solve_time += toc(lp_start_time);
      exploration_stats_.total_lp_iters += node_iter;
    }
  }

  if (lp_status == dual::status_t::DUAL_UNBOUNDED) {
    // Node was infeasible. Do not branch
    node_ptr->lower_bound = inf;
    search_tree.graphviz_node(log, node_ptr, "infeasible", 0.0);
    search_tree.update(node_ptr, node_status_t::INFEASIBLE);
    return node_solve_info_t::NO_CHILDREN;

  } else if (lp_status == dual::status_t::CUTOFF) {
    // Node was cut off. Do not branch
    node_ptr->lower_bound = upper_bound;
    f_t leaf_objective    = compute_objective(leaf_problem, leaf_solution.x);
    search_tree.graphviz_node(log, node_ptr, "cut off", leaf_objective);
    search_tree.update(node_ptr, node_status_t::FATHOMED);
    return node_solve_info_t::NO_CHILDREN;

  } else if (lp_status == dual::status_t::OPTIMAL) {
    // LP was feasible
    std::vector<i_t> leaf_fractional;
    i_t leaf_num_fractional =
      fractional_variables(settings_, leaf_solution.x, var_types_, leaf_fractional);

    f_t leaf_objective    = compute_objective(leaf_problem, leaf_solution.x);
    node_ptr->lower_bound = leaf_objective;
    search_tree.graphviz_node(log, node_ptr, "lower bound", leaf_objective);
    pc_.update_pseudo_costs(node_ptr, leaf_objective);

    if (settings_.node_processed_callback != nullptr) {
      std::vector<f_t> original_x;
      uncrush_primal_solution(original_problem_, original_lp_, leaf_solution.x, original_x);
      settings_.node_processed_callback(original_x, leaf_objective);
    }

    if (leaf_num_fractional == 0) {
      // Found a integer feasible solution
      add_feasible_solution(leaf_objective, leaf_solution.x, node_ptr->depth, thread_type);
      search_tree.graphviz_node(log, node_ptr, "integer feasible", leaf_objective);
      search_tree.update(node_ptr, node_status_t::INTEGER_FEASIBLE);
      return node_solve_info_t::NO_CHILDREN;

    } else if (leaf_objective <= upper_bound + abs_fathom_tol) {
      // Choose fractional variable to branch on
      const i_t branch_var =
        pc_.variable_selection(leaf_fractional, leaf_solution.x, lp_settings.log);

      assert(leaf_vstatus.size() == leaf_problem.num_cols);
      search_tree.branch(
        node_ptr, branch_var, leaf_solution.x[branch_var], leaf_vstatus, leaf_problem, log);
      search_tree.update(node_ptr, node_status_t::HAS_CHILDREN);

      rounding_direction_t round_dir = child_selection(node_ptr);

      if (round_dir == rounding_direction_t::UP) {
        return node_solve_info_t::UP_CHILD_FIRST;
      } else {
        return node_solve_info_t::DOWN_CHILD_FIRST;
      }

    } else {
      search_tree.graphviz_node(log, node_ptr, "fathomed", leaf_objective);
      search_tree.update(node_ptr, node_status_t::FATHOMED);
      return node_solve_info_t::NO_CHILDREN;
    }
  } else if (lp_status == dual::status_t::TIME_LIMIT) {
    search_tree.graphviz_node(log, node_ptr, "timeout", 0.0);
    return node_solve_info_t::TIME_LIMIT;

  } else {
    if (thread_type == thread_type_t::EXPLORATION) {
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
    return node_solve_info_t::NUMERICAL;
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::exploration_ramp_up(mip_node_t<i_t, f_t>* node,
                                                       search_tree_t<i_t, f_t>* search_tree,
                                                       const csr_matrix_t<i_t, f_t>& Arow,
                                                       i_t initial_heap_size)
{
  if (solver_status_ != mip_exploration_status_t::RUNNING) { return; }

  // Note that we do not know which thread will execute the
  // `exploration_ramp_up` task, so we allow to any thread
  // to repair the heuristic solution.
  repair_heuristic_solutions();

  f_t lower_bound = node->lower_bound;
  f_t upper_bound = get_upper_bound();
  f_t rel_gap     = user_relative_gap(original_lp_, upper_bound, lower_bound);
  f_t abs_gap     = upper_bound - lower_bound;

  if (lower_bound > upper_bound || rel_gap < settings_.relative_mip_gap_tol) {
    search_tree->graphviz_node(settings_.log, node, "cutoff", node->lower_bound);
    search_tree->update(node, node_status_t::FATHOMED);
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
      f_t obj              = compute_user_objective(original_lp_, upper_bound);
      f_t user_lower       = compute_user_objective(original_lp_, root_objective_);
      std::string gap_user = user_mip_gap<f_t>(obj, user_lower);
      i_t nodes_explored   = exploration_stats_.nodes_explored;
      i_t nodes_unexplored = exploration_stats_.nodes_unexplored;

      settings_.log.printf(
        " %10d   %10lu    %+13.6e    %+10.6e   %6d   %7.1e     %s %9.2f\n",
        nodes_explored,
        nodes_unexplored,
        obj,
        user_lower,
        node->depth,
        nodes_explored > 0 ? exploration_stats_.total_lp_iters / nodes_explored : 0,
        gap_user.c_str(),
        now);

      exploration_stats_.nodes_since_last_log = 0;
      exploration_stats_.last_log             = tic();
      should_report_                          = true;
    }
  }

  if (now > settings_.time_limit) {
    solver_status_ = mip_exploration_status_t::TIME_LIMIT;
    return;
  }

  // Make a copy of the original LP. We will modify its bounds at each leaf
  lp_problem_t<i_t, f_t> leaf_problem = original_lp_;
  std::vector<char> row_sense;
  bounds_strengthening_t<i_t, f_t> node_presolver(leaf_problem, Arow, row_sense, var_types_);

  const i_t m = leaf_problem.num_rows;
  basis_update_mpf_t<i_t, f_t> basis_factors(m, settings_.refactor_frequency);
  std::vector<i_t> basic_list(m);
  std::vector<i_t> nonbasic_list;

  node_solve_info_t status = solve_node(node,
                                        *search_tree,
                                        leaf_problem,
                                        basis_factors,
                                        basic_list,
                                        nonbasic_list,
                                        node_presolver,
                                        thread_type_t::EXPLORATION,
                                        true,
                                        original_lp_.lower,
                                        original_lp_.upper,
                                        settings_.log);

  ++exploration_stats_.nodes_since_last_log;
  ++exploration_stats_.nodes_explored;
  --exploration_stats_.nodes_unexplored;

  if (status == node_solve_info_t::TIME_LIMIT) {
    solver_status_ = mip_exploration_status_t::TIME_LIMIT;
    return;

  } else if (has_children(status)) {
    exploration_stats_.nodes_unexplored += 2;

    // If we haven't generated enough nodes to keep the threads busy, continue the ramp up phase
    if (exploration_stats_.nodes_unexplored < initial_heap_size) {
#pragma omp task
      exploration_ramp_up(node->get_down_child(), search_tree, Arow, initial_heap_size);

#pragma omp task
      exploration_ramp_up(node->get_up_child(), search_tree, Arow, initial_heap_size);

    } else {
      // We've generated enough nodes, push further nodes onto the heap
      mutex_heap_.lock();
      heap_.push(node->get_down_child());
      heap_.push(node->get_up_child());
      mutex_heap_.unlock();
    }
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::explore_subtree(i_t task_id,
                                                   mip_node_t<i_t, f_t>* start_node,
                                                   search_tree_t<i_t, f_t>& search_tree,
                                                   lp_problem_t<i_t, f_t>& leaf_problem,
                                                   bounds_strengthening_t<i_t, f_t>& node_presolver,
                                                   basis_update_mpf_t<i_t, f_t>& basis_factors,
                                                   std::vector<i_t>& basic_list,
                                                   std::vector<i_t>& nonbasic_list)
{
  bool recompute_bounds_and_basis = true;
  std::deque<mip_node_t<i_t, f_t>*> stack;
  stack.push_front(start_node);

  while (stack.size() > 0 && solver_status_ == mip_exploration_status_t::RUNNING) {
    if (task_id == 0) { repair_heuristic_solutions(); }

    mip_node_t<i_t, f_t>* node_ptr = stack.front();
    stack.pop_front();

    f_t lower_bound = node_ptr->lower_bound;
    f_t upper_bound = get_upper_bound();
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
      search_tree.graphviz_node(settings_.log, node_ptr, "cutoff", node_ptr->lower_bound);
      search_tree.update(node_ptr, node_status_t::FATHOMED);
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
        f_t obj              = compute_user_objective(original_lp_, upper_bound);
        f_t user_lower       = compute_user_objective(original_lp_, get_lower_bound());
        std::string gap_user = user_mip_gap<f_t>(obj, user_lower);
        i_t nodes_explored   = exploration_stats_.nodes_explored;
        i_t nodes_unexplored = exploration_stats_.nodes_unexplored;

        settings_.log.printf(
          " %10d   %10lu    %+13.6e    %+10.6e   %6d   %7.1e     %s %9.2f\n",
          nodes_explored,
          nodes_unexplored,
          obj,
          user_lower,
          node_ptr->depth,
          nodes_explored > 0 ? exploration_stats_.total_lp_iters / nodes_explored : 0,
          gap_user.c_str(),
          now);
        exploration_stats_.last_log             = tic();
        exploration_stats_.nodes_since_last_log = 0;
      }
    }

    if (now > settings_.time_limit) {
      solver_status_ = mip_exploration_status_t::TIME_LIMIT;
      return;
    }
    if (exploration_stats_.nodes_explored >= settings_.node_limit) {
      solver_status_ = mip_exploration_status_t::NODE_LIMIT;
      return;
    }

    node_solve_info_t status = solve_node(node_ptr,
                                          search_tree,
                                          leaf_problem,
                                          basis_factors,
                                          basic_list,
                                          nonbasic_list,
                                          node_presolver,
                                          thread_type_t::EXPLORATION,
                                          recompute_bounds_and_basis,
                                          original_lp_.lower,
                                          original_lp_.upper,
                                          settings_.log);

    recompute_bounds_and_basis = !has_children(status);

    ++exploration_stats_.nodes_since_last_log;
    ++exploration_stats_.nodes_explored;
    --exploration_stats_.nodes_unexplored;

    if (status == node_solve_info_t::TIME_LIMIT) {
      solver_status_ = mip_exploration_status_t::TIME_LIMIT;
      return;

    } else if (has_children(status)) {
      // The stack should only contain the children of the current parent.
      // If the stack size is greater than 0,
      // we pop the current node from the stack and place it in the global heap,
      // since we are about to add the two children to the stack
      if (stack.size() > 0) {
        mip_node_t<i_t, f_t>* node = stack.back();
        stack.pop_back();

        // The order here matters. We want to create a copy of the node
        // before adding to the global heap. Otherwise,
        // some thread may consume the node (possibly fathoming it)
        // before we had the chance to add to the diving queue.
        // This lead to a SIGSEGV. Although, in this case, it
        // would be better if we discard the node instead.
        if (get_heap_size() > settings_.num_bfs_threads) {
          std::vector<f_t> lower = original_lp_.lower;
          std::vector<f_t> upper = original_lp_.upper;
          std::fill(
            node_presolver.bounds_changed.begin(), node_presolver.bounds_changed.end(), false);
          node->get_variable_bounds(lower, upper, node_presolver.bounds_changed);

          mutex_dive_queue_.lock();
          diving_queue_.emplace(node->detach_copy(), std::move(lower), std::move(upper));
          mutex_dive_queue_.unlock();
        }

        mutex_heap_.lock();
        heap_.push(node);
        mutex_heap_.unlock();
      }

      exploration_stats_.nodes_unexplored += 2;

      if (status == node_solve_info_t::UP_CHILD_FIRST) {
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
void branch_and_bound_t<i_t, f_t>::best_first_thread(i_t task_id,
                                                     search_tree_t<i_t, f_t>& search_tree,
                                                     const csr_matrix_t<i_t, f_t>& Arow)
{
  f_t lower_bound = -inf;
  f_t upper_bound = inf;
  f_t abs_gap     = inf;
  f_t rel_gap     = inf;

  // Make a copy of the original LP. We will modify its bounds at each leaf
  lp_problem_t<i_t, f_t> leaf_problem = original_lp_;
  std::vector<char> row_sense;
  bounds_strengthening_t<i_t, f_t> node_presolver(leaf_problem, Arow, row_sense, var_types_);

  const i_t m = leaf_problem.num_rows;
  basis_update_mpf_t<i_t, f_t> basis_factors(m, settings_.refactor_frequency);
  std::vector<i_t> basic_list(m);
  std::vector<i_t> nonbasic_list;

  while (solver_status_ == mip_exploration_status_t::RUNNING &&
         abs_gap > settings_.absolute_mip_gap_tol && rel_gap > settings_.relative_mip_gap_tol &&
         (active_subtrees_ > 0 || get_heap_size() > 0)) {
    mip_node_t<i_t, f_t>* start_node = nullptr;

    // If there any node left in the heap, we pop the top node and explore it.
    mutex_heap_.lock();
    if (heap_.size() > 0) {
      start_node = heap_.top();
      heap_.pop();
      active_subtrees_++;
    }
    mutex_heap_.unlock();

    if (start_node != nullptr) {
      if (get_upper_bound() < start_node->lower_bound) {
        // This node was put on the heap earlier but its lower bound is now greater than the
        // current upper bound
        search_tree.graphviz_node(settings_.log, start_node, "cutoff", start_node->lower_bound);
        search_tree.update(start_node, node_status_t::FATHOMED);
        active_subtrees_--;
        continue;
      }

      // Best-first search with plunging
      explore_subtree(task_id,
                      start_node,
                      search_tree,
                      leaf_problem,
                      node_presolver,
                      basis_factors,
                      basic_list,
                      nonbasic_list);

      active_subtrees_--;
    }

    lower_bound = get_lower_bound();
    upper_bound = get_upper_bound();
    abs_gap     = upper_bound - lower_bound;
    rel_gap     = user_relative_gap(original_lp_, upper_bound, lower_bound);
  }

  // Check if it is the last thread that exited the loop and no
  // timeout or numerical error has happen.
  if (solver_status_ == mip_exploration_status_t::RUNNING) {
    if (active_subtrees_ == 0) {
      solver_status_ = mip_exploration_status_t::COMPLETED;
    } else {
      local_lower_bounds_[task_id] = inf;
    }
  }
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::diving_thread(const csr_matrix_t<i_t, f_t>& Arow)
{
  logger_t log;
  log.log = false;
  // Make a copy of the original LP. We will modify its bounds at each leaf
  lp_problem_t<i_t, f_t> leaf_problem = original_lp_;
  std::vector<char> row_sense;
  bounds_strengthening_t<i_t, f_t> node_presolver(leaf_problem, Arow, row_sense, var_types_);

  const i_t m = leaf_problem.num_rows;
  basis_update_mpf_t<i_t, f_t> basis_factors(m, settings_.refactor_frequency);
  std::vector<i_t> basic_list(m);
  std::vector<i_t> nonbasic_list;

  while (solver_status_ == mip_exploration_status_t::RUNNING &&
         (active_subtrees_ > 0 || get_heap_size() > 0)) {
    std::optional<diving_root_t<i_t, f_t>> start_node;

    mutex_dive_queue_.lock();
    if (diving_queue_.size() > 0) { start_node = diving_queue_.pop(); }
    mutex_dive_queue_.unlock();

    if (start_node.has_value()) {
      if (get_upper_bound() < start_node->node.lower_bound) { continue; }

      bool recompute_bounds_and_basis = true;
      search_tree_t<i_t, f_t> subtree(std::move(start_node->node));
      std::deque<mip_node_t<i_t, f_t>*> stack;
      stack.push_front(&subtree.root);

      while (stack.size() > 0 && solver_status_ == mip_exploration_status_t::RUNNING) {
        mip_node_t<i_t, f_t>* node_ptr = stack.front();
        stack.pop_front();
        f_t upper_bound = get_upper_bound();
        f_t rel_gap     = user_relative_gap(original_lp_, upper_bound, node_ptr->lower_bound);

        if (node_ptr->lower_bound > upper_bound || rel_gap < settings_.relative_mip_gap_tol) {
          recompute_bounds_and_basis = true;
          continue;
        }

        if (toc(exploration_stats_.start_time) > settings_.time_limit) { return; }

        node_solve_info_t status = solve_node(node_ptr,
                                              subtree,
                                              leaf_problem,
                                              basis_factors,
                                              basic_list,
                                              nonbasic_list,
                                              node_presolver,
                                              thread_type_t::DIVING,
                                              recompute_bounds_and_basis,
                                              start_node->lower,
                                              start_node->upper,
                                              log);

        recompute_bounds_and_basis = !has_children(status);

        if (status == node_solve_info_t::TIME_LIMIT) {
          solver_status_ = mip_exploration_status_t::TIME_LIMIT;
          return;

        } else if (has_children(status)) {
          if (status == node_solve_info_t::UP_CHILD_FIRST) {
            stack.push_front(node_ptr->get_down_child());
            stack.push_front(node_ptr->get_up_child());
          } else {
            stack.push_front(node_ptr->get_up_child());
            stack.push_front(node_ptr->get_down_child());
          }
        }

        if (stack.size() > 1) {
          // If the diving thread is consuming the nodes faster than the
          // best first search, then we split the current subtree at the
          // lowest possible point and move to the queue, so it can
          // be picked by another thread.
          if (std::lock_guard<omp_mutex_t> lock(mutex_dive_queue_);
              diving_queue_.size() < min_diving_queue_size_) {
            mip_node_t<i_t, f_t>* new_node = stack.back();
            stack.pop_back();

            std::vector<f_t> lower = start_node->lower;
            std::vector<f_t> upper = start_node->upper;
            std::fill(
              node_presolver.bounds_changed.begin(), node_presolver.bounds_changed.end(), false);
            new_node->get_variable_bounds(lower, upper, node_presolver.bounds_changed);

            diving_queue_.emplace(new_node->detach_copy(), std::move(lower), std::move(upper));
          }
        }
      }
    }
  }
}

template <typename i_t, typename f_t>
lp_status_t branch_and_bound_t<i_t, f_t>::solve_root_relaxation(
  simplex_solver_settings_t<i_t, f_t> const& lp_settings)
{
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

    if (crossover_status == crossover_status_t::OPTIMAL) {
      settings_.log.printf("Crossover status: %d\n", crossover_status);
    }

    // Check if crossover was stopped by dual simplex
    if (crossover_status == crossover_status_t::OPTIMAL) {
      set_root_concurrent_halt(1);  // Stop dual simplex
      root_status = root_status_future.get();
      // Override the root relaxation solution with the crossover solution
      root_relax_soln_ = root_crossover_soln_;
      root_vstatus_    = crossover_vstatus_;
      root_status      = lp_status_t::OPTIMAL;
    } else {
      root_status = root_status_future.get();
    }
  } else {
    root_status = root_status_future.get();
  }
  return root_status;
}

template <typename i_t, typename f_t>
mip_status_t branch_and_bound_t<i_t, f_t>::solve(mip_solution_t<i_t, f_t>& solution)
{
  logger_t log;
  log.log                             = false;
  log.log_prefix                      = settings_.log.log_prefix;
  solver_status_                      = mip_exploration_status_t::UNSET;
  exploration_stats_.nodes_unexplored = 0;
  exploration_stats_.nodes_explored   = 0;

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

  settings_.log.printf("Solving LP root relaxation\n");

  lp_status_t root_status;
  simplex_solver_settings_t lp_settings = settings_;
  lp_settings.inside_mip                = 1;
  lp_settings.concurrent_halt           = get_root_concurrent_halt();
  // RINS/SUBMIP path
  if (!enable_concurrent_lp_root_solve()) {
    root_status = solve_linear_program_advanced(original_lp_,
                                                exploration_stats_.start_time,
                                                lp_settings,
                                                root_relax_soln_,
                                                root_vstatus_,
                                                edge_norms_);

  } else {
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
    solver_status_ = mip_exploration_status_t::TIME_LIMIT;
    return set_final_solution(solution, -inf);
  }

  assert(root_vstatus_.size() == original_lp_.num_cols);
  set_uninitialized_steepest_edge_norms<i_t, f_t>(edge_norms_);

  root_objective_ = compute_objective(original_lp_, root_relax_soln_.x);
  local_lower_bounds_.assign(settings_.num_bfs_threads, root_objective_);

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
    solver_status_ = mip_exploration_status_t::TIME_LIMIT;
    return set_final_solution(solution, root_objective_);
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

  csr_matrix_t<i_t, f_t> Arow(1, 1, 0);
  original_lp_.A.to_compressed_row(Arow);

  settings_.log.printf("Exploring the B&B tree using %d threads (best-first = %d, diving = %d)\n",
                       settings_.num_threads,
                       settings_.num_bfs_threads,
                       settings_.num_diving_threads);

  settings_.log.printf(
    " | Explored | Unexplored |    Objective    |     Bound     | Depth | Iter/Node |   Gap    "
    "|  Time  |\n");

  exploration_stats_.nodes_explored       = 0;
  exploration_stats_.nodes_unexplored     = 2;
  exploration_stats_.nodes_since_last_log = 0;
  exploration_stats_.last_log             = tic();
  active_subtrees_                        = 0;
  min_diving_queue_size_                  = 4 * settings_.num_diving_threads;
  solver_status_                          = mip_exploration_status_t::RUNNING;
  lower_bound_ceiling_                    = inf;
  should_report_                          = true;

#pragma omp parallel num_threads(settings_.num_threads)
  {
#pragma omp master
    {
      auto down_child  = search_tree_.root.get_down_child();
      auto up_child    = search_tree_.root.get_up_child();
      i_t initial_size = 2 * settings_.num_threads;

#pragma omp task
      exploration_ramp_up(down_child, &search_tree_, Arow, initial_size);

#pragma omp task
      exploration_ramp_up(up_child, &search_tree_, Arow, initial_size);
    }

#pragma omp barrier

#pragma omp master
    {
      for (i_t i = 0; i < settings_.num_bfs_threads; i++) {
#pragma omp task
        best_first_thread(i, search_tree_, Arow);
      }

      for (i_t i = 0; i < settings_.num_diving_threads; i++) {
#pragma omp task
        diving_thread(Arow);
      }
    }
  }

  f_t lower_bound = heap_.size() > 0 ? heap_.top()->lower_bound : search_tree_.root.lower_bound;
  return set_final_solution(solution, lower_bound);
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template class branch_and_bound_t<int, double>;

#endif

}  // namespace cuopt::linear_programming::dual_simplex
