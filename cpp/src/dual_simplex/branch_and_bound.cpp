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
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace cuopt::linear_programming::dual_simplex {

namespace global_variables {

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

// Mutex for lower bound
std::mutex mutex_lower;
// Global variable for lower bound
double lower_bound;

// Mutex for upper bound
std::mutex mutex_upper;
// Global variable for upper bound
double upper_bound;
// Global variable for incumbent. The incumbent should be updated with the upper bound
mip_solution_t<int, double> incumbent(1);

// Mutex for gap
std::mutex mutex_gap;
// Global variable for gap
double gap;

// Mutex for branching
std::mutex mutex_branching;
bool currently_branching;

// Mutex for stats
std::mutex mutex_stats;
// Global variable for stats
struct stats_t {
  int nodes_explored;
  double total_lp_solve_time;
  double start_time;
} stats;

// Mutex for repair
std::mutex mutex_repair;
std::vector<std::vector<double>> repair_queue;

#endif

}  // namespace global_variables

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
void set_uninitialized_steepest_edge_norms(i_t n, std::vector<f_t>& edge_norms)
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

}  // namespace

template <typename f_t>
f_t get_upper_bound()
{
  global_variables::mutex_upper.lock();
  const f_t upper_bound = global_variables::upper_bound;
  global_variables::mutex_upper.unlock();
  return upper_bound;
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
  if (user_mip_gap != user_mip_gap) { return std::numeric_limits<f_t>::infinity(); }
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

template <typename i_t, typename f_t>
void branch_and_bound_t<i_t, f_t>::set_new_solution(const std::vector<f_t>& solution)
{
  if (solution.size() != original_problem.num_cols) {
    settings.log.printf(
      "Solution size mismatch %ld %d\n", solution.size(), original_problem.num_cols);
  }
  std::vector<f_t> crushed_solution;
  crush_primal_solution<i_t, f_t>(
    original_problem, original_lp, solution, new_slacks, crushed_solution);
  f_t obj             = compute_objective(original_lp, crushed_solution);
  bool is_feasible    = false;
  bool attempt_repair = false;
  global_variables::mutex_upper.lock();
  if (obj < global_variables::upper_bound) {
    f_t primal_err;
    f_t bound_err;
    i_t num_fractional;
    is_feasible = check_guess(
      original_lp, settings, var_types, crushed_solution, primal_err, bound_err, num_fractional);
    if (is_feasible) {
      global_variables::upper_bound = obj;
    } else {
      attempt_repair         = true;
      constexpr bool verbose = false;
      if (verbose) {
        settings.log.printf(
          "Injected solution infeasible. Constraint error %e bound error %e integer infeasible "
          "%d\n",
          primal_err,
          bound_err,
          num_fractional);
      }
    }
  }
  global_variables::mutex_upper.unlock();

  if (is_feasible) {
    global_variables::mutex_lower.lock();
    f_t lower_bound = global_variables::lower_bound;
    global_variables::mutex_lower.unlock();
    global_variables::mutex_branching.lock();
    bool currently_branching = global_variables::currently_branching;
    global_variables::mutex_branching.unlock();
    if (currently_branching) {
      settings.log.printf(
        "H                        %+13.6e  %+10.6e                      %s %9.2f\n",
        compute_user_objective(original_lp, obj),
        compute_user_objective(original_lp, lower_bound),
        user_mip_gap<f_t>(compute_user_objective(original_lp, obj),
                          compute_user_objective(original_lp, lower_bound))
          .c_str(),
        toc(start_time));
    } else {
      settings.log.printf("New solution from primal heuristics. Objective %+.6e. Time %.2f\n",
                          compute_user_objective(original_lp, obj),
                          toc(start_time));
    }
  }

  if (attempt_repair) {
    global_variables::mutex_repair.lock();
    global_variables::repair_queue.push_back(crushed_solution);
    global_variables::mutex_repair.unlock();
  }
}

template <typename i_t, typename f_t>
bool branch_and_bound_t<i_t, f_t>::repair_solution(
  const std::vector<variable_status_t>& root_vstatus,
  const std::vector<f_t>& edge_norms,
  const std::vector<f_t>& potential_solution,
  f_t& repaired_obj,
  std::vector<f_t>& repaired_solution) const
{
  bool feasible = false;
  repaired_obj  = std::numeric_limits<f_t>::quiet_NaN();
  i_t n         = original_lp.num_cols;
  assert(potential_solution.size() == n);

  lp_problem_t repair_lp = original_lp;

  // Fix integer variables
  for (i_t j = 0; j < n; ++j) {
    if (var_types[j] == variable_type_t::INTEGER) {
      const f_t fixed_val = std::round(potential_solution[j]);
      repair_lp.lower[j]  = fixed_val;
      repair_lp.upper[j]  = fixed_val;
    }
  }

  lp_solution_t<i_t, f_t> lp_solution(original_lp.num_rows, original_lp.num_cols);

  i_t iter                               = 0;
  f_t lp_start_time                      = tic();
  simplex_solver_settings_t lp_settings  = settings;
  std::vector<variable_status_t> vstatus = root_vstatus;
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
    feasible = check_guess(
      original_lp, settings, var_types, lp_solution.x, primal_error, bound_error, num_fractional);
    repaired_obj           = compute_objective(original_lp, repaired_solution);
    constexpr bool verbose = false;
    if (verbose) {
      settings.log.printf(
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
branch_and_bound_t<i_t, f_t>::branch_and_bound_t(
  const user_problem_t<i_t, f_t>& user_problem,
  const simplex_solver_settings_t<i_t, f_t>& solver_settings)
  : original_problem(user_problem), settings(solver_settings), original_lp(1, 1, 1)
{
  start_time = tic();
  convert_user_problem(original_problem, original_lp, new_slacks);
  full_variable_types(original_problem, original_lp, var_types);

  global_variables::mutex_upper.lock();
  global_variables::upper_bound = inf;
  global_variables::mutex_upper.unlock();

  global_variables::mutex_lower.lock();
  global_variables::lower_bound = -inf;
  global_variables::mutex_lower.unlock();

  global_variables::mutex_branching.lock();
  global_variables::currently_branching = false;
  global_variables::mutex_branching.unlock();
}

template <typename i_t, typename f_t>
mip_status_t branch_and_bound_t<i_t, f_t>::solve(mip_solution_t<i_t, f_t>& solution)
{
  mip_status_t status = mip_status_t::UNSET;
  mip_solution_t<i_t, f_t> incumbent(original_lp.num_cols);
  if (guess.size() != 0) {
    std::vector<f_t> crushed_guess;
    crush_primal_solution(original_problem, original_lp, guess, new_slacks, crushed_guess);
    f_t primal_err;
    f_t bound_err;
    i_t num_fractional;
    const bool feasible = check_guess(
      original_lp, settings, var_types, crushed_guess, primal_err, bound_err, num_fractional);
    if (feasible) {
      const f_t computed_obj = compute_objective(original_lp, crushed_guess);
      global_variables::mutex_upper.lock();
      incumbent.set_incumbent_solution(computed_obj, crushed_guess);
      global_variables::upper_bound = computed_obj;
      global_variables::mutex_upper.unlock();
    }
  }

  lp_solution_t<i_t, f_t> root_relax_soln(original_lp.num_rows, original_lp.num_cols);
  std::vector<variable_status_t> root_vstatus;
  std::vector<f_t> edge_norms;
  settings.log.printf("Solving LP root relaxation\n");
  simplex_solver_settings_t lp_settings = settings;
  lp_settings.inside_mip                = 1;
  lp_status_t root_status               = solve_linear_program_advanced(
    original_lp, start_time, lp_settings, root_relax_soln, root_vstatus, edge_norms);
  f_t total_lp_solve_time = toc(start_time);
  assert(root_vstatus.size() == original_lp.num_cols);
  if (root_status == lp_status_t::INFEASIBLE) {
    settings.log.printf("MIP Infeasible\n");
    if (settings.heuristic_preemption_callback != nullptr) {
      settings.heuristic_preemption_callback();
    }
    return mip_status_t::INFEASIBLE;
  }
  if (root_status == lp_status_t::UNBOUNDED) {
    settings.log.printf("MIP Unbounded\n");
    if (settings.heuristic_preemption_callback != nullptr) {
      settings.heuristic_preemption_callback();
    }
    return mip_status_t::UNBOUNDED;
  }
  if (root_status == lp_status_t::TIME_LIMIT) {
    settings.log.printf("Hit time limit\n");
    return mip_status_t::TIME_LIMIT;
  }
  set_uninitialized_steepest_edge_norms(original_lp.num_cols, edge_norms);

  std::vector<i_t> fractional;
  const i_t num_fractional =
    fractional_variables(settings, root_relax_soln.x, var_types, fractional);
  const f_t root_objective = compute_objective(original_lp, root_relax_soln.x);
  if (settings.solution_callback != nullptr) {
    std::vector<f_t> original_x;
    uncrush_primal_solution(original_problem, original_lp, root_relax_soln.x, original_x);
    settings.set_simplex_solution_callback(original_x,
                                           compute_user_objective(original_lp, root_objective));
  }
  global_variables::mutex_lower.lock();
  f_t lower_bound = global_variables::lower_bound = root_objective;
  global_variables::mutex_lower.unlock();

  if (num_fractional == 0) {
    global_variables::mutex_upper.lock();
    incumbent.set_incumbent_solution(root_objective, root_relax_soln.x);
    global_variables::upper_bound = root_objective;
    global_variables::mutex_upper.unlock();
    // We should be done here
    uncrush_primal_solution(original_problem, original_lp, incumbent.x, solution.x);
    solution.objective          = incumbent.objective;
    solution.lower_bound        = lower_bound;
    solution.nodes_explored     = 0;
    solution.simplex_iterations = root_relax_soln.iterations;
    settings.log.printf("Optimal solution found at root node. Objective %.16e. Time %.2f.\n",
                        compute_user_objective(original_lp, root_objective),
                        toc(start_time));
    if (settings.solution_callback != nullptr) {
      settings.solution_callback(solution.x, solution.objective);
    }
    if (settings.heuristic_preemption_callback != nullptr) {
      settings.heuristic_preemption_callback();
    }
    return mip_status_t::OPTIMAL;
  }

  pseudo_costs_t<i_t, f_t> pc(original_lp.num_cols);
  strong_branching<i_t, f_t>(original_lp,
                             settings,
                             start_time,
                             var_types,
                             root_relax_soln.x,
                             fractional,
                             root_objective,
                             root_vstatus,
                             edge_norms,
                             pc);

  auto compare = [](mip_node_t<i_t, f_t>* a, mip_node_t<i_t, f_t>* b) {
    return a->lower_bound >
           b->lower_bound;  // True if a comes before b, elements that come before are output last
  };
  std::priority_queue<mip_node_t<i_t, f_t>*, std::vector<mip_node_t<i_t, f_t>*>, decltype(compare)>
    heap(compare);
  i_t num_nodes = 0;
  mip_node_t<i_t, f_t> root_node(root_objective, root_vstatus);
  graphviz_node(settings, &root_node, "lower bound", lower_bound);

  // Choose variable to branch on
  logger_t log;
  log.log = false;
  const i_t branch_var =
    pc.variable_selection(fractional, root_relax_soln.x, original_lp.lower, original_lp.upper, log);

  // down child
  std::unique_ptr<mip_node_t<i_t, f_t>> down_child =
    std::make_unique<mip_node_t<i_t, f_t>>(original_lp,
                                           &root_node,
                                           ++num_nodes,
                                           branch_var,
                                           0,
                                           root_relax_soln.x[branch_var],
                                           root_vstatus);

  graphviz_edge(settings,
                &root_node,
                down_child.get(),
                branch_var,
                0,
                std::floor(root_relax_soln.x[branch_var]));

  // up child
  std::unique_ptr<mip_node_t<i_t, f_t>> up_child =
    std::make_unique<mip_node_t<i_t, f_t>>(original_lp,
                                           &root_node,
                                           ++num_nodes,
                                           branch_var,
                                           1,
                                           root_relax_soln.x[branch_var],
                                           root_vstatus);

  graphviz_edge(
    settings, &root_node, up_child.get(), branch_var, 0, std::ceil(root_relax_soln.x[branch_var]));

  assert(root_vstatus.size() == original_lp.num_cols);
  heap.push(down_child.get());  // the heap does not own the unique_ptr the tree does
  heap.push(up_child.get());    // the heap does not own the unqiue_ptr the tree does
  root_node.add_children(std::move(down_child),
                         std::move(up_child));  // child pointers moved into the tree
  lp_problem_t leaf_problem =
    original_lp;  // Make a copy of the original LP. We will modify its bounds at each leaf
  f_t gap            = get_upper_bound<f_t>() - lower_bound;
  i_t nodes_explored = 0;
  settings.log.printf(
    "| Explored | Unexplored | Objective   |    Bound    | Depth | Iter/Node |  Gap   | "
    "   Time \n");
  global_variables::mutex_branching.lock();
  global_variables::currently_branching = true;
  global_variables::mutex_branching.unlock();

  f_t total_lp_iters = 0.0;
  f_t last_log       = 0;
  while (gap > settings.absolute_mip_gap_tol &&
         relative_gap(get_upper_bound<f_t>(), lower_bound) > settings.relative_mip_gap_tol &&
         heap.size() > 0) {
    // Check if there are any solutions to repair
    std::vector<std::vector<f_t>> to_repair;
    global_variables::mutex_repair.lock();
    if (global_variables::repair_queue.size() > 0) {
      to_repair = global_variables::repair_queue;
      global_variables::repair_queue.clear();
    }
    global_variables::mutex_repair.unlock();

    if (to_repair.size() > 0) {
      settings.log.debug("Attempting to repair %ld injected solutions\n", to_repair.size());
      for (const std::vector<f_t>& potential_solution : to_repair) {
        std::vector<f_t> repaired_solution;
        f_t repaired_obj;
        bool is_feasible = repair_solution(
          root_vstatus, edge_norms, potential_solution, repaired_obj, repaired_solution);
        if (is_feasible) {
          global_variables::mutex_upper.lock();
          if (repaired_obj < global_variables::upper_bound) {
            global_variables::upper_bound = repaired_obj;
            incumbent.set_incumbent_solution(repaired_obj, repaired_solution);

            settings.log.printf(
              "H                        %+13.6e  %+10.6e                      %s %9.2f\n",
              compute_user_objective(original_lp, repaired_obj),
              compute_user_objective(original_lp, lower_bound),
              user_mip_gap<f_t>(compute_user_objective(original_lp, repaired_obj),
                                compute_user_objective(original_lp, lower_bound))
                .c_str(),
              toc(start_time));
            if (settings.solution_callback != nullptr) {
              std::vector<f_t> original_x;
              uncrush_primal_solution(original_problem, original_lp, repaired_solution, original_x);
              settings.solution_callback(original_x, repaired_obj);
            }
          }
          global_variables::mutex_upper.unlock();
        }
      }
    }

    // Get a node off the heap
    mip_node_t<i_t, f_t>* node_ptr = heap.top();
    heap.pop();  // Remove node from the heap
    f_t upper_bound = get_upper_bound<f_t>();
    if (upper_bound < node_ptr->lower_bound) {
      // This node was put on the heap earlier but its lower bound is now greater than the current
      // upper bound
      std::vector<mip_node_t<i_t, f_t>*> stack;
      node_ptr->set_status(node_status_t::FATHOMED, stack);
      graphviz_node(settings, node_ptr, "cutoff", node_ptr->lower_bound);
      remove_fathomed_nodes(stack);
      continue;
    }
    global_variables::mutex_lower.lock();
    global_variables::lower_bound = lower_bound = node_ptr->lower_bound;
    global_variables::mutex_lower.unlock();
    gap                  = upper_bound - lower_bound;
    const i_t leaf_depth = node_ptr->depth;
    f_t now              = toc(start_time);
    f_t time_since_log   = last_log == 0 ? 1.0 : toc(last_log);
    if ((nodes_explored % 1000 == 0 || gap < 10 * settings.absolute_mip_gap_tol ||
         nodes_explored < 1000) &&
          (time_since_log >= 1) ||
        (time_since_log > 60) || now > settings.time_limit) {
      settings.log.printf(" %8d %8lu       %+13.6e  %+10.6e   %4d   %7.1e     %s %9.2f\n",
                          nodes_explored,
                          heap.size(),
                          compute_user_objective(original_lp, upper_bound),
                          compute_user_objective(original_lp, lower_bound),
                          leaf_depth,
                          nodes_explored > 0 ? total_lp_iters / nodes_explored : 0,
                          user_mip_gap<f_t>(compute_user_objective(original_lp, upper_bound),
                                            compute_user_objective(original_lp, lower_bound))
                            .c_str(),
                          now);
      last_log = tic();
    }

    if (now > settings.time_limit) {
      settings.log.printf("Hit time limit. Stoppping\n");
      status = mip_status_t::TIME_LIMIT;
      break;
    }

    // Set the correct bounds for the leaf problem
    leaf_problem.lower = original_lp.lower;
    leaf_problem.upper = original_lp.upper;
    node_ptr->get_variable_bounds(leaf_problem.lower, leaf_problem.upper);

    std::vector<variable_status_t>& leaf_vstatus = node_ptr->vstatus;
    lp_solution_t<i_t, f_t> leaf_solution(leaf_problem.num_rows, leaf_problem.num_cols);

    i_t node_iter = 0;
    assert(leaf_vstatus.size() == leaf_problem.num_cols);
    f_t lp_start_time                     = tic();
    std::vector<f_t> leaf_edge_norms      = edge_norms;  // = node.steepest_edge_norms;
    simplex_solver_settings_t lp_settings = settings;
    lp_settings.set_log(false);
    lp_settings.cut_off      = upper_bound + settings.dual_tol;
    lp_settings.inside_mip   = 2;
    dual::status_t lp_status = dual_phase2(2,
                                           0,
                                           lp_start_time,
                                           leaf_problem,
                                           lp_settings,
                                           leaf_vstatus,
                                           leaf_solution,
                                           node_iter,
                                           leaf_edge_norms);
    total_lp_solve_time += toc(lp_start_time);
    total_lp_iters += node_iter;

    nodes_explored++;
    if (lp_status == dual::status_t::DUAL_UNBOUNDED) {
      node_ptr->lower_bound = inf;
      std::vector<mip_node_t<i_t, f_t>*> stack;
      node_ptr->set_status(node_status_t::INFEASIBLE, stack);
      graphviz_node(settings, node_ptr, "infeasible", 0.0);
      remove_fathomed_nodes(stack);
      // Node was infeasible. Do not branch
    } else if (lp_status == dual::status_t::CUTOFF) {
      node_ptr->lower_bound = upper_bound;
      std::vector<mip_node_t<i_t, f_t>*> stack;
      node_ptr->set_status(node_status_t::FATHOMED, stack);
      f_t leaf_objective = compute_objective(leaf_problem, leaf_solution.x);
      graphviz_node(settings, node_ptr, "cut off", leaf_objective);
      remove_fathomed_nodes(stack);
      // Node was cut off. Do not branch
    } else if (lp_status == dual::status_t::OPTIMAL) {
      // LP was feasible
      std::vector<i_t> fractional;
      const i_t leaf_fractional =
        fractional_variables(settings, leaf_solution.x, var_types, fractional);
      f_t leaf_objective = compute_objective(leaf_problem, leaf_solution.x);
      graphviz_node(settings, node_ptr, "lower bound", leaf_objective);

      pc.update_pseudo_costs(node_ptr, leaf_objective);
      node_ptr->lower_bound = leaf_objective;

      constexpr f_t fathom_tol = 1e-5;
      if (leaf_fractional == 0) {
        bool send_solution = false;
        global_variables::mutex_upper.lock();
        if (leaf_objective < global_variables::upper_bound) {
          incumbent.set_incumbent_solution(leaf_objective, leaf_solution.x);
          global_variables::upper_bound = upper_bound = leaf_objective;
          gap                                         = upper_bound - lower_bound;
          settings.log.printf("B%8d %8lu       %+13.6e  %+10.6e   %4d   %7.1e     %s %9.2f\n",
                              nodes_explored,
                              heap.size(),
                              compute_user_objective(original_lp, upper_bound),
                              compute_user_objective(original_lp, lower_bound),
                              leaf_depth,
                              nodes_explored > 0 ? total_lp_iters / nodes_explored : 0,
                              user_mip_gap<f_t>(compute_user_objective(original_lp, upper_bound),
                                                compute_user_objective(original_lp, lower_bound))
                                .c_str(),
                              toc(start_time));
          send_solution = true;
        }
        global_variables::mutex_upper.unlock();
        if (send_solution && settings.solution_callback != nullptr) {
          std::vector<f_t> original_x;
          uncrush_primal_solution(original_problem, original_lp, incumbent.x, original_x);
          settings.solution_callback(original_x, upper_bound);
        }
        graphviz_node(settings, node_ptr, "integer feasible", leaf_objective);
        std::vector<mip_node_t<i_t, f_t>*> stack;
        node_ptr->set_status(node_status_t::INTEGER_FEASIBLE, stack);
        remove_fathomed_nodes(stack);
      } else if (leaf_objective <= upper_bound + fathom_tol) {
        // Choose fractional variable to branch on
        const i_t branch_var = pc.variable_selection(
          fractional, leaf_solution.x, leaf_problem.lower, leaf_problem.upper, log);
        assert(leaf_vstatus.size() == leaf_problem.num_cols);

        // down child
        std::unique_ptr<mip_node_t<i_t, f_t>> down_child =
          std::make_unique<mip_node_t<i_t, f_t>>(original_lp,
                                                 node_ptr,
                                                 ++num_nodes,
                                                 branch_var,
                                                 0,
                                                 leaf_solution.x[branch_var],
                                                 leaf_vstatus);
        graphviz_edge(settings,
                      node_ptr,
                      down_child.get(),
                      branch_var,
                      0,
                      std::floor(leaf_solution.x[branch_var]));
        //  up child
        std::unique_ptr<mip_node_t<i_t, f_t>> up_child =
          std::make_unique<mip_node_t<i_t, f_t>>(original_lp,
                                                 node_ptr,
                                                 ++num_nodes,
                                                 branch_var,
                                                 1,
                                                 leaf_solution.x[branch_var],
                                                 leaf_vstatus);
        graphviz_edge(settings,
                      node_ptr,
                      up_child.get(),
                      branch_var,
                      0,
                      std::ceil(leaf_solution.x[branch_var]));
        heap.push(down_child.get());  // the heap does not own the unique_ptr the tree does
        heap.push(up_child.get());    // the heap does not own the unique_ptr the tree does
        node_ptr->add_children(std::move(down_child),
                               std::move(up_child));  // child pointers moved into the tree
      } else {
        graphviz_node(settings, node_ptr, "fathomed", leaf_objective);
        std::vector<mip_node_t<i_t, f_t>*> stack;
        node_ptr->set_status(node_status_t::FATHOMED, stack);
        remove_fathomed_nodes(stack);
      }
    } else {
      graphviz_node(settings, node_ptr, "numerical", 0.0);
      settings.log.printf("Encountered LP status %d. This indicates a numerical issue.\n",
                          lp_status);
      status = mip_status_t::NUMERICAL;
      break;
    }
  }
  global_variables::mutex_branching.lock();
  global_variables::currently_branching = false;
  global_variables::mutex_branching.unlock();

  if (heap.size() == 0) {
    global_variables::mutex_lower.lock();
    lower_bound = global_variables::lower_bound = root_node.lower_bound;
    global_variables::mutex_lower.unlock();
    gap = get_upper_bound<f_t>() - lower_bound;
  }

  settings.log.printf(
    "Explored %d nodes in %.2fs.\nAbsolute Gap %e Objective %.16e Lower Bound %.16e\n",
    nodes_explored,
    toc(start_time),
    gap,
    compute_user_objective(original_lp, get_upper_bound<f_t>()),
    compute_user_objective(original_lp, lower_bound));

  if (gap <= settings.absolute_mip_gap_tol ||
      relative_gap(get_upper_bound<f_t>(), lower_bound) <= settings.relative_mip_gap_tol) {
    status = mip_status_t::OPTIMAL;
    if (gap > 0 && gap <= settings.absolute_mip_gap_tol) {
      settings.log.printf("Optimal solution found within absolute MIP gap tolerance (%.1e)\n",
                          settings.absolute_mip_gap_tol);
    } else if (gap > 0 &&
               relative_gap(get_upper_bound<f_t>(), lower_bound) <= settings.relative_mip_gap_tol) {
      settings.log.printf("Optimal solution found within relative MIP gap tolerance (%.1e)\n",
                          settings.relative_mip_gap_tol);
    } else {
      settings.log.printf("Optimal solution found.\n");
    }
    if (settings.heuristic_preemption_callback != nullptr) {
      settings.heuristic_preemption_callback();
    }
  }

  if (heap.size() == 0 && get_upper_bound<f_t>() == inf) {
    settings.log.printf("Integer infeasible.\n");
    status = mip_status_t::INFEASIBLE;
    if (settings.heuristic_preemption_callback != nullptr) {
      settings.heuristic_preemption_callback();
    }
  }

  uncrush_primal_solution(original_problem, original_lp, incumbent.x, solution.x);
  solution.objective          = incumbent.objective;
  solution.lower_bound        = lower_bound;
  solution.nodes_explored     = nodes_explored;
  solution.simplex_iterations = total_lp_iters;
  return status;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template class branch_and_bound_t<int, double>;

#endif

}  // namespace cuopt::linear_programming::dual_simplex
