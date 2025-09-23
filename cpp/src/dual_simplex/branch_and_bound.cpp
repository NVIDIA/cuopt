/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include <dual_simplex/branch_and_bound.hpp>

#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/logger.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/pseudo_costs.hpp>
#include <dual_simplex/random.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <dual_simplex/user_problem.hpp>

#include <cstdio>
#include <cstdlib>
#include <future>
#include <limits>
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

constexpr int write_graphviz = false;

template <typename i_t, typename f_t>
void graphviz_node(const simplex_solver_settings_t<i_t, f_t>& settings,
                   mip_node_t<i_t, f_t>* node_ptr,
                   std::string label,
                   f_t val)
{
  if (write_graphviz) {
    settings.log.printf("Node%d [label=\"%s %.16e\"]\n", node_ptr->node_id, label.c_str(), val);
  }
}

template <typename i_t, typename f_t>
void graphviz_edge(const simplex_solver_settings_t<i_t, f_t>& settings,
                   mip_node_t<i_t, f_t>* origin_ptr,
                   mip_node_t<i_t, f_t>* dest_ptr,
                   i_t branch_var,
                   i_t branch_dir,
                   f_t bound)
{
  if (write_graphviz) {
    settings.log.printf("Node%d -> Node%d [label=\"x%d %s %e\"]\n",
                        origin_ptr->node_id,
                        dest_ptr->node_id,
                        branch_var,
                        branch_dir == 0 ? "<=" : ">=",
                        bound);
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
  // Handle NaNs (i.e., NaN != NaN)
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

}  // namespace

template <typename i_t, typename f_t>
branch_and_bound_t<i_t, f_t>::branch_and_bound_t(
  const user_problem_t<i_t, f_t>& user_problem,
  const simplex_solver_settings_t<i_t, f_t>& solver_settings)
  : original_problem_(user_problem),
    settings_(solver_settings),
    original_lp_(1, 1, 1),
    incumbent_(1),
    root_relax_soln_(1, 1),
    pc_(1)
{
  stats_.start_time = tic();
  convert_user_problem(original_problem_, settings_, original_lp_, new_slacks_);
  full_variable_types(original_problem_, original_lp_, var_types_);

  mutex_upper_.lock();
  upper_bound_ = inf;
  mutex_upper_.unlock();

  mutex_lower_.lock();
  lower_bound_ = -inf;
  mutex_lower_.unlock();

  mutex_branching_.lock();
  currently_branching_ = false;
  mutex_branching_.unlock();
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
  mutex_lower_.lock();
  const f_t lower_bound = lower_bound_;
  mutex_lower_.unlock();
  return lower_bound;
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
    mutex_branching_.lock();
    bool currently_branching = currently_branching_;
    mutex_branching_.unlock();
    if (currently_branching) {
      f_t user_obj    = compute_user_objective(original_lp_, obj);
      f_t user_lower  = compute_user_objective(original_lp_, get_lower_bound());
      std::string gap = user_mip_gap<f_t>(user_obj, user_lower);

      settings_.log.printf(
        "H                        %+13.6e  %+10.6e                      %s %9.2f\n",
        user_obj,
        user_lower,
        gap.c_str(),
        toc(stats_.start_time));
    } else {
      settings_.log.printf("New solution from primal heuristics. Objective %+.6e. Time %.2f\n",
                           compute_user_objective(original_lp_, obj),
                           toc(stats_.start_time));
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
            "H                        %+13.6e  %+10.6e                      %s %9.2f\n",
            obj,
            lower,
            user_gap.c_str(),
            toc(stats_.start_time));

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
void branch_and_bound_t<i_t, f_t>::branch(mip_node_t<i_t, f_t>* parent_node,
                                          i_t branch_var,
                                          f_t branch_var_val,
                                          const std::vector<variable_status_t>& parent_vstatus)
{
  // down child
  auto down_child = std::make_unique<mip_node_t<i_t, f_t>>(
    original_lp_, parent_node, ++stats_.num_nodes, branch_var, 0, branch_var_val, parent_vstatus);

  graphviz_edge(
    settings_, parent_node, down_child.get(), branch_var, 0, std::floor(branch_var_val));

  // up child
  auto up_child = std::make_unique<mip_node_t<i_t, f_t>>(
    original_lp_, parent_node, ++stats_.num_nodes, branch_var, 1, branch_var_val, parent_vstatus);

  graphviz_edge(settings_, parent_node, up_child.get(), branch_var, 1, std::ceil(branch_var_val));

  assert(parent_vstatus.size() == original_lp_.num_cols);
  parent_node->add_children(std::move(down_child),
                            std::move(up_child));  // child pointers moved into the tree
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::update_tree(mip_node_t<i_t, f_t>* node_ptr, node_status_t status)
{
  std::vector<mip_node_t<i_t, f_t>*> stack;
  node_ptr->set_status(status, stack);
  remove_fathomed_nodes(stack);
}

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::add_feasible_solution(f_t leaf_objective,
                                                         const std::vector<f_t>& leaf_solution,
                                                         i_t leaf_depth,
                                                         char symbol)
{
  bool send_solution   = false;
  i_t nodes_explored   = stats_.nodes_explored;
  i_t nodes_unexplored = stats_.nodes_unexplored;
  f_t gap;

  mutex_upper_.lock();
  if (leaf_objective < upper_bound_) {
    incumbent_.set_incumbent_solution(leaf_objective, leaf_solution);
    upper_bound_    = leaf_objective;
    f_t lower_bound = get_lower_bound();
    gap             = upper_bound_ - lower_bound;
    f_t obj         = compute_user_objective(original_lp_, upper_bound_);
    f_t lower       = compute_user_objective(original_lp_, lower_bound);
    settings_.log.printf("%c%8d %8lu       %+13.6e  %+10.6e   %4d   %7.1e     %s %9.2f\n",
                         symbol,
                         nodes_explored,
                         nodes_unexplored,
                         obj,
                         lower,
                         leaf_depth,
                         nodes_explored > 0 ? stats_.total_lp_iters / nodes_explored : 0,
                         user_mip_gap<f_t>(obj, lower).c_str(),
                         toc(stats_.start_time));

    send_solution = true;
  }

  if (send_solution && settings_.solution_callback != nullptr) {
    std::vector<f_t> original_x;
    uncrush_primal_solution(original_problem_, original_lp_, incumbent_.x, original_x);
    settings_.solution_callback(original_x, upper_bound_);
  }
  mutex_upper_.unlock();

  if (send_solution) {
    mutex_gap_.lock();
    gap_ = gap;
    mutex_gap_.unlock();
  }
}

template <typename i_t, typename f_t>
dual::status_t branch_and_bound_t<i_t, f_t>::node_dual_simplex(
  i_t leaf_id,
  lp_problem_t<i_t, f_t>& leaf_problem,
  std::vector<variable_status_t>& leaf_vstatus,
  lp_solution_t<i_t, f_t>& leaf_solution,
  std::vector<bool>& bounds_changed,
  csc_matrix_t<i_t, f_t>& Arow,
  f_t upper_bound,
  logger_t& log)
{
  i_t node_iter = 0;
  assert(leaf_vstatus.size() == leaf_problem.num_cols);
  f_t lp_start_time                     = tic();
  std::vector<f_t> leaf_edge_norms      = edge_norms_;  // = node.steepest_edge_norms;
  simplex_solver_settings_t lp_settings = settings_;
  lp_settings.set_log(false);
  lp_settings.cut_off    = upper_bound + settings_.dual_tol;
  lp_settings.inside_mip = 2;

  // in B&B we only have equality constraints, leave it empty for default
  std::vector<char> row_sense;
  bool feasible =
    bound_strengthening(row_sense, lp_settings, leaf_problem, Arow, var_types_, bounds_changed);

  dual::status_t lp_status = dual::status_t::DUAL_UNBOUNDED;

  if (feasible) {
    lp_status = dual_phase2(2,
                            0,
                            lp_start_time,
                            leaf_problem,
                            lp_settings,
                            leaf_vstatus,
                            leaf_solution,
                            node_iter,
                            leaf_edge_norms);

    if (lp_status == dual::status_t::NUMERICAL) {
      log.printf("Numerical issue node %d. Resolving from scratch.\n", leaf_id);
      lp_status_t second_status = solve_linear_program_advanced(
        leaf_problem, lp_start_time, lp_settings, leaf_solution, leaf_vstatus, leaf_edge_norms);
      lp_status = convert_lp_status_to_dual_status(second_status);
    }
  } else {
    log.printf("Infeasible after bounds strengthening. Fathoming node %d.\n", leaf_id);
  }

  mutex_stats_.lock();
  stats_.total_lp_solve_time += toc(lp_start_time);
  stats_.total_lp_iters += node_iter;
  mutex_stats_.unlock();

  return lp_status;
}

template <typename i_t, typename f_t>
mip_status_t branch_and_bound_t<i_t, f_t>::solve_node_lp(
  mip_node_t<i_t, f_t>* node_ptr,
  lp_problem_t<i_t, f_t>& leaf_problem,
  csc_matrix_t<i_t, f_t>& Arow,
  const std::vector<variable_type_t>& var_types,
  f_t upper_bound)
{
  logger_t log;
  log.log                                      = false;
  std::vector<variable_status_t>& leaf_vstatus = node_ptr->vstatus;
  lp_solution_t<i_t, f_t> leaf_solution(leaf_problem.num_rows, leaf_problem.num_cols);

  // Set the correct bounds for the leaf problem
  leaf_problem.lower = original_lp_.lower;
  leaf_problem.upper = original_lp_.upper;

  std::vector<bool> bounds_changed(original_lp_.num_cols, false);
  // Technically, we can get the already strengthened bounds from the node/parent instead of
  // getting it from the original problem and re-strengthening. But this requires storing
  // two vectors at each node and potentially cause memory issues
  node_ptr->get_variable_bounds(leaf_problem.lower, leaf_problem.upper, bounds_changed);

  dual::status_t lp_status = node_dual_simplex(node_ptr->node_id,
                                               leaf_problem,
                                               leaf_vstatus,
                                               leaf_solution,
                                               bounds_changed,
                                               Arow,
                                               upper_bound,
                                               settings_.log);

  if (lp_status == dual::status_t::DUAL_UNBOUNDED) {
    node_ptr->lower_bound = inf;
    graphviz_node(settings_, node_ptr, "infeasible", 0.0);
    update_tree(node_ptr, node_status_t::INFEASIBLE);
    // Node was infeasible. Do not branch

  } else if (lp_status == dual::status_t::CUTOFF) {
    node_ptr->lower_bound = upper_bound;
    f_t leaf_objective    = compute_objective(leaf_problem, leaf_solution.x);
    graphviz_node(settings_, node_ptr, "cut off", leaf_objective);
    update_tree(node_ptr, node_status_t::FATHOMED);
    // Node was cut off. Do not branch
  } else if (lp_status == dual::status_t::OPTIMAL) {
    // LP was feasible
    std::vector<i_t> fractional;
    const i_t leaf_fractional =
      fractional_variables(settings_, leaf_solution.x, var_types_, fractional);
    f_t leaf_objective = compute_objective(leaf_problem, leaf_solution.x);
    graphviz_node(settings_, node_ptr, "lower bound", leaf_objective);

    mutex_pc_.lock();
    pc_.update_pseudo_costs(node_ptr, leaf_objective);
    mutex_pc_.unlock();

    node_ptr->lower_bound = leaf_objective;

    constexpr f_t fathom_tol = 1e-5;
    if (leaf_fractional == 0) {
      add_feasible_solution(leaf_objective, leaf_solution.x, node_ptr->depth, 'B');
      graphviz_node(settings_, node_ptr, "integer feasible", leaf_objective);
      update_tree(node_ptr, node_status_t::INTEGER_FEASIBLE);

    } else if (leaf_objective <= upper_bound + fathom_tol) {
      // Choose fractional variable to branch on
      mutex_pc_.lock();
      const i_t branch_var = pc_.variable_selection(
        fractional, leaf_solution.x, leaf_problem.lower, leaf_problem.upper, log);
      mutex_pc_.unlock();

      assert(leaf_vstatus.size() == leaf_problem.num_cols);
      branch(node_ptr, branch_var, leaf_solution.x[branch_var], leaf_vstatus);
      node_ptr->status = node_status_t::HAS_CHILDREN;

    } else {
      graphviz_node(settings_, node_ptr, "fathomed", leaf_objective);
      update_tree(node_ptr, node_status_t::FATHOMED);
    }
  } else {
    graphviz_node(settings_, node_ptr, "numerical", 0.0);
    settings_.log.printf("Encountered LP status %d. This indicates a numerical issue.\n",
                         lp_status);
    return mip_status_t::NUMERICAL;
  }

  return mip_status_t::UNSET;
}

template <typename i_t, typename f_t>
mip_status_t branch_and_bound_t<i_t, f_t>::explore_tree(i_t branch_var,
                                                        mip_solution_t<i_t, f_t>& solution)
{
  mip_status_t status = mip_status_t::UNSET;
  logger_t log;
  log.log      = false;
  auto compare = [](mip_node_t<i_t, f_t>* a, mip_node_t<i_t, f_t>* b) {
    return a->lower_bound >
           b->lower_bound;  // True if a comes before b, elements that come before are output last
  };

  std::priority_queue<mip_node_t<i_t, f_t>*, std::vector<mip_node_t<i_t, f_t>*>, decltype(compare)>
    heap(compare);

  mip_node_t<i_t, f_t> root_node(root_objective_, root_vstatus_);
  graphviz_node(settings_, &root_node, "lower bound", root_objective_);

  branch(&root_node, branch_var, root_relax_soln_.x[branch_var], root_vstatus_);

  // the stack does not own the unique_ptr the tree does
  heap.push(root_node.get_down_child());
  heap.push(root_node.get_up_child());

  // Make a copy of the original LP. We will modify its bounds at each leaf
  lp_problem_t leaf_problem = original_lp_;
  csc_matrix_t<i_t, f_t> Arow(1, 1, 1);
  original_lp_.A.transpose(Arow);

  f_t lower_bound = get_lower_bound();
  f_t gap         = get_upper_bound() - lower_bound;
  f_t last_log    = 0;

  while (gap > settings_.absolute_mip_gap_tol &&
         relative_gap(get_upper_bound(), lower_bound) > settings_.relative_mip_gap_tol &&
         heap.size() > 0) {
    repair_heuristic_solutions();

    // Get a node off the heap
    mip_node_t<i_t, f_t>* node_ptr = heap.top();
    heap.pop();
    stats_.nodes_explored++;

    f_t upper_bound = get_upper_bound();
    if (upper_bound < node_ptr->lower_bound) {
      // This node was put on the heap earlier but its lower bound is now greater than the current
      // upper bound
      update_tree(node_ptr, node_status_t::FATHOMED);
      graphviz_node(settings_, node_ptr, "cutoff", node_ptr->lower_bound);
      continue;
    }
    mutex_lower_.lock();
    lower_bound = lower_bound_ = node_ptr->lower_bound;
    mutex_lower_.unlock();

    mutex_gap_.lock();
    gap_ = gap = upper_bound - lower_bound;
    mutex_gap_.unlock();

    i_t nodes_explored = stats_.nodes_explored;
    f_t now            = toc(stats_.start_time);
    f_t time_since_log = last_log == 0 ? 1.0 : toc(last_log);
    if ((nodes_explored % 1000 == 0 || gap < 10 * settings_.absolute_mip_gap_tol ||
         nodes_explored < 1000) &&
          (time_since_log >= 1) ||
        (time_since_log > 60) || now > settings_.time_limit) {
      f_t user_obj         = compute_user_objective(original_lp_, upper_bound);
      f_t user_lower       = compute_user_objective(original_lp_, lower_bound);
      std::string user_gap = user_mip_gap<f_t>(user_obj, user_lower);

      settings_.log.printf(" %8d %8lu       %+13.6e  %+10.6e   %4d   %7.1e     %s %9.2f\n",
                           nodes_explored,
                           heap.size(),
                           user_obj,
                           user_lower,
                           node_ptr->depth,
                           nodes_explored > 0 ? stats_.total_lp_iters / nodes_explored : 0,
                           user_gap.c_str(),
                           now);
      last_log = tic();
    }

    if (toc(stats_.start_time) > settings_.time_limit) {
      settings_.log.printf("Hit time limit. Stopping\n");
      status = mip_status_t::TIME_LIMIT;
      break;
    }

    status = solve_node_lp(node_ptr, leaf_problem, Arow, var_types_, upper_bound);

    if (status == mip_status_t::NUMERICAL) { break; }

    if (node_ptr->status == node_status_t::HAS_CHILDREN) {
      // the heap does not own the unique_ptr the tree does
      heap.push(node_ptr->get_down_child());
      heap.push(node_ptr->get_up_child());
    }
  }

  stats_.nodes_unexplored = heap.size();

  if (stats_.nodes_unexplored == 0) {
    mutex_lower_.lock();
    lower_bound = lower_bound_ = root_node.lower_bound;
    mutex_lower_.unlock();

    mutex_gap_.lock();
    gap_ = gap = get_upper_bound() - lower_bound;
    mutex_gap_.unlock();
  }

  return status;
}

template <typename i_t, typename f_t>
mip_status_t branch_and_bound_t<i_t, f_t>::dive(i_t branch_var, mip_solution_t<i_t, f_t>& solution)
{
  mip_status_t status = mip_status_t::UNSET;

  logger_t log;
  log.log = false;

  std::vector<mip_node_t<i_t, f_t>*> node_stack;

  mip_node_t<i_t, f_t> root_node(root_objective_, root_vstatus_);
  graphviz_node(settings_, &root_node, "lower bound", root_objective_);

  branch(&root_node, branch_var, root_relax_soln_.x[branch_var], root_vstatus_);

  // the stack does not own the unique_ptr the tree does
  node_stack.push_back(root_node.get_down_child());
  node_stack.push_back(root_node.get_up_child());

  // Make a copy of the original LP. We will modify its bounds at each leaf
  lp_problem_t leaf_problem = original_lp_;

  csc_matrix_t<i_t, f_t> Arow(1, 1, 1);
  original_lp_.A.transpose(Arow);

  f_t lower_bound    = get_lower_bound();
  f_t gap            = get_upper_bound() - lower_bound;
  i_t nodes_explored = 0;

  while (node_stack.size() > 0) {
    // Get a node off the stack
    mip_node_t<i_t, f_t>* node_ptr = node_stack.back();
    node_stack.pop_back();
    nodes_explored++;

    f_t upper_bound = get_upper_bound();
    lower_bound     = get_lower_bound();
    gap             = upper_bound - lower_bound;

    if (gap < settings_.absolute_mip_gap_tol &&
        relative_gap(get_upper_bound(), lower_bound) < settings_.relative_mip_gap_tol) {
      update_tree(node_ptr, node_status_t::FATHOMED);
      continue;
    }

    if (toc(stats_.start_time) > settings_.time_limit) {
      status = mip_status_t::TIME_LIMIT;
      break;
    }

    status = solve_node_lp(node_ptr, leaf_problem, Arow, var_types_, upper_bound);

    if (status == mip_status_t::NUMERICAL) { continue; }

    if (node_ptr->status == node_status_t::HAS_CHILDREN) {
      // Martin's child selection
      const i_t branch_var     = node_ptr->get_down_child()->branch_var;
      const f_t branch_var_val = node_ptr->get_down_child()->fractional_val;
      const f_t down_val       = std::floor(root_relax_soln_.x[branch_var]);
      const f_t up_val         = std::ceil(root_relax_soln_.x[branch_var]);
      const f_t down_dist      = branch_var_val - down_val;
      const f_t up_dist        = up_val - branch_var_val;

      if (down_dist < up_dist) {
        node_stack.push_back(node_ptr->get_up_child());
        node_stack.push_back(node_ptr->get_down_child());
      } else {
        node_stack.push_back(node_ptr->get_down_child());
        node_stack.push_back(node_ptr->get_up_child());
      }
    }
  }

  return status;
}

template <typename i_t, typename f_t>
mip_status_t branch_and_bound_t<i_t, f_t>::solve(mip_solution_t<i_t, f_t>& solution)
{
  mip_status_t status = mip_status_t::UNSET;

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
  simplex_solver_settings_t lp_settings = settings_;
  lp_settings.inside_mip                = 1;
  lp_status_t root_status               = solve_linear_program_advanced(
    original_lp_, stats_.start_time, lp_settings, root_relax_soln_, root_vstatus_, edge_norms_);
  stats_.total_lp_solve_time = toc(stats_.start_time);
  assert(root_vstatus_.size() == original_lp_.num_cols);
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
    settings_.log.printf("Hit time limit\n");
    return mip_status_t::TIME_LIMIT;
  }
  set_uninitialized_steepest_edge_norms<i_t, f_t>(edge_norms_);

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
  mutex_lower_.lock();
  lower_bound_ = root_objective_;
  mutex_lower_.unlock();

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
    solution.lower_bound        = lower_bound_;
    solution.nodes_explored     = 0;
    solution.simplex_iterations = root_relax_soln_.iterations;
    settings_.log.printf("Optimal solution found at root node. Objective %.16e. Time %.2f.\n",
                         compute_user_objective(original_lp_, root_objective_),
                         toc(stats_.start_time));
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
                             stats_.start_time,
                             var_types_,
                             root_relax_soln_.x,
                             fractional,
                             root_objective_,
                             root_vstatus_,
                             edge_norms_,
                             pc_);

  // Choose variable to branch on
  logger_t log;
  log.log        = false;
  i_t branch_var = pc_.variable_selection(
    fractional, root_relax_soln_.x, original_lp_.lower, original_lp_.upper, log);

  stats_.total_lp_iters   = 0;
  stats_.nodes_explored   = 0;
  stats_.nodes_unexplored = 0;
  stats_.num_nodes        = 1;

  settings_.log.printf(
    "| Explored | Unexplored | Objective   |    Bound    | Depth | Iter/Node |  Gap   | "
    "   Time \n");

  mutex_branching_.lock();
  currently_branching_ = true;
  mutex_branching_.unlock();

  std::future<mip_status_t> diving_thread;

  if (settings_.num_threads > 0) {
    diving_thread = std::async(std::launch::async, [&]() { return dive(branch_var, solution); });
  }

  status = explore_tree(branch_var, solution);

  if (settings_.num_threads > 0) { mip_status_t diving_status = diving_thread.get(); }

  mutex_branching_.lock();
  currently_branching_ = false;
  mutex_branching_.unlock();

  settings_.log.printf(
    "Explored %d nodes in %.2fs.\nAbsolute Gap %e Objective %.16e Lower Bound %.16e\n",
    stats_.nodes_explored.load(),
    toc(stats_.start_time),
    gap_,
    compute_user_objective(original_lp_, get_upper_bound()),
    compute_user_objective(original_lp_, lower_bound_));

  if (gap_ <= settings_.absolute_mip_gap_tol ||
      relative_gap(get_upper_bound(), lower_bound_) <= settings_.relative_mip_gap_tol) {
    status = mip_status_t::OPTIMAL;
    if (gap_ > 0 && gap_ <= settings_.absolute_mip_gap_tol) {
      settings_.log.printf("Optimal solution found within absolute MIP gap tolerance (%.1e)\n",
                           settings_.absolute_mip_gap_tol);
    } else if (gap_ > 0 &&
               relative_gap(get_upper_bound(), lower_bound_) <= settings_.relative_mip_gap_tol) {
      settings_.log.printf("Optimal solution found within relative MIP gap tolerance (%.1e)\n",
                           settings_.relative_mip_gap_tol);
    } else {
      settings_.log.printf("Optimal solution found.\n");
    }
    if (settings_.heuristic_preemption_callback != nullptr) {
      settings_.heuristic_preemption_callback();
    }
  }

  if (stats_.nodes_unexplored == 0 && get_upper_bound() == inf) {
    settings_.log.printf("Integer infeasible.\n");
    status = mip_status_t::INFEASIBLE;
    if (settings_.heuristic_preemption_callback != nullptr) {
      settings_.heuristic_preemption_callback();
    }
  }

  uncrush_primal_solution(original_problem_, original_lp_, incumbent_.x, solution.x);
  solution.objective          = incumbent_.objective;
  solution.lower_bound        = get_lower_bound();
  solution.nodes_explored     = stats_.nodes_explored;
  solution.simplex_iterations = stats_.total_lp_iters;
  return status;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template class branch_and_bound_t<int, double>;

#endif

}  // namespace cuopt::linear_programming::dual_simplex
