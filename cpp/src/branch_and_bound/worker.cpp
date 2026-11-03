/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "worker.hpp"
#include "branch_and_bound.hpp"
#include "degenerate_pivots.hpp"
#include "fractional.hpp"
#include "utils.hpp"

#include <raft/core/nvtx.hpp>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
bool branch_and_bound_worker_t<i_t, f_t>::apply_symmetry_reductions(
  const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
  mip_node_t<i_t, f_t>* node_ptr,
  branch_and_bound_stats_t<i_t, f_t>& stats)
{
  i_t start_stack_index = domain.size();

  // Perform orbital fixing
  if (orbital_fixing != nullptr && !orbital_fixing->disabled()) {
    i_t prev_fix  = node_ptr->orbital_fix_zero.size() + node_ptr->orbital_fix_one.size();
    i_t conflicts = orbital_fixing->orbital_fixing(
      symmetry_ptr, settings, node_ptr, leaf_problem, root_lp.lower, root_lp.upper, domain);
    i_t new_fix = node_ptr->orbital_fix_zero.size() + node_ptr->orbital_fix_one.size();
    if (new_fix > prev_fix) {
      ++stats.orbital_fixing_nodes;
      stats.orbital_fixings_applied += (new_fix - prev_fix);
    }
    if (conflicts > 0) { ++stats.orbital_conflict_nodes; }
  } else if (orbital_fixing != nullptr) {
    orbital_fixing->propagate_cumulative_fixings(node_ptr);
  }

  if (settings.symmetry == 2 && lexical_reduction != nullptr) {
    i_t lexical_reductions_info =
      lexical_reduction->lexical_reduce(symmetry_ptr, node_ptr, leaf_problem, domain);
    if (lexical_reductions_info > 0) {
      ++stats.lexical_reduction_nodes;
      stats.lexical_reduction_fixings_applied += lexical_reductions_info;
    }
    if (lexical_reductions_info == -1) {
      ++stats.lexical_reduction_pruned_nodes;
      return false;
    }
  }

  return domain.propagate_from_stack(Arow, var_types, settings, leaf_problem, start_stack_index);
}

template <typename i_t, typename f_t>
simplex::dual_status_t branch_and_bound_worker_t<i_t, f_t>::solve_node_lp(
  const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
  mip_node_t<i_t, f_t>* node_ptr,
  branch_and_bound_stats_t<i_t, f_t>& stats,
  simplex::logger_t& log,
  int64_t iter_limit,
  f_t start_time,
  f_t upper_bound)
{
  raft::common::nvtx::range scope("BB::solve_node");
#ifdef DEBUG_BRANCHING
  i_t num_integer_variables = 0;
  for (i_t j = 0; j < root_lp.num_cols; j++) {
    if (var_types[j] == simplex::variable_type_t::INTEGER) { num_integer_variables++; }
  }
  if (node_ptr->depth > num_integer_variables) {
    std::vector<i_t> branched_variables(root_lp.num_cols, 0);
    std::vector<f_t> branched_lower(root_lp.num_cols, std::numeric_limits<f_t>::quiet_NaN());
    std::vector<f_t> branched_upper(root_lp.num_cols, std::numeric_limits<f_t>::quiet_NaN());
    mip_node_t<i_t, f_t>* parent = node_ptr->parent;
    while (parent != nullptr) {
      if (root_lp.lower[parent->branch_var] != 0.0 || root_lp.upper[parent->branch_var] != 1.0) {
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

  simplex::simplex_solver_settings_t lp_settings = settings;
  lp_settings.set_log(false);
  const f_t cutoff       = compute_node_cutoff(leaf_problem, settings, upper_bound);
  lp_settings.cut_off    = cutoff;
  lp_settings.inside_mip = 2;
  lp_settings.time_limit = settings.time_limit - toc(start_time);
  if (lp_settings.time_limit <= 0.0) { return simplex::dual_status_t::TIME_LIMIT; }
  lp_settings.scale_columns   = false;
  lp_settings.iteration_limit = std::min<int64_t>(iter_limit, std::numeric_limits<i_t>::max());

  if (lp_settings.concurrent_halt == nullptr) {
    lp_settings.concurrent_halt = branch_and_bound_ptr->get_node_halt_flag();
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
    settings.log.log_prefix.c_str(),
    node_ptr->node_id,
    node_ptr->depth,
    node_ptr->branch_var,
    node_ptr->branch_dir == branch_direction_t::DOWN ? "DOWN" : "UP",
    node_ptr->fractional_val,
    node_ptr->branch_var_lower,
    node_ptr->branch_var_upper,
    node_ptr->vstatus[node_ptr->branch_var]);
#endif

  bool feasible                    = update_variable_bounds(node_ptr, lp_settings);
  simplex::dual_status_t lp_status = simplex::dual_status_t::DUAL_UNBOUNDED;
  leaf_edge_norms                  = root_edge_norm;
  if (recompute_basis && orbital_fixing && search_strategy == search_strategy_t::BEST_FIRST) {
    orbital_fixing->reset(symmetry_ptr, node_ptr);
  }

  if (feasible) {
    feasible = apply_symmetry_reductions(settings, node_ptr, stats);

    if (feasible) {
      lp_settings.time_limit = settings.time_limit - toc(start_time);
      if (lp_settings.time_limit <= 0.0) { return simplex::dual_status_t::TIME_LIMIT; }

      if (search_strategy == search_strategy_t::BEST_FIRST) {
        if (!domain.propagate_objective(Arow, var_types, settings, leaf_problem, cutoff)) {
          return simplex::dual_status_t::CUTOFF;
        }
      }

      i_t node_iter          = 0;
      f_t lp_start_time      = tic();
      f_t node_work_estimate = 0.0;

      lp_status = dual_phase2_with_advanced_basis(2,
                                                  0,
                                                  recompute_basis,
                                                  lp_start_time,
                                                  leaf_problem,
                                                  lp_settings,
                                                  leaf_vstatus,
                                                  basis_factors,
                                                  basic_list,
                                                  nonbasic_list,
                                                  leaf_solution,
                                                  node_iter,
                                                  leaf_edge_norms,
                                                  node_work_estimate);

      if (lp_status == simplex::dual_status_t::NUMERICAL) {
        log.debug_format("Numerical issue node {}. Resolving from scratch.\n", node_ptr->node_id);
        simplex::lp_status_t second_status =
          solve_linear_program_with_advanced_basis(leaf_problem,
                                                   lp_start_time,
                                                   lp_settings,
                                                   leaf_solution,
                                                   basis_factors,
                                                   basic_list,
                                                   nonbasic_list,
                                                   leaf_vstatus,
                                                   leaf_edge_norms,
                                                   node_work_estimate);

        lp_status = convert_lp_status_to_dual_status(second_status);
      }

      stats.total_lp_solve_time += toc(lp_start_time);
      stats.total_simplex_iters += node_iter;

      if (lp_status == simplex::dual_status_t::OPTIMAL) {
        if (settings.dual_degenerate_pivots == 2 ||
            (settings.dual_degenerate_pivots == 1 &&
             search_strategy != search_strategy_t::BEST_FIRST)) {
          std::vector<i_t> fractional;
          i_t num_fractional =
            fractional_variables(settings, leaf_solution.x, var_types, fractional);
          f_t integer_pivot_work = 0.0;  // Diagnostic only; do not charge node LP work.
          i_t num_integer_increased =
            pivot_out_integer_variables(leaf_problem,
                                        lp_settings,
                                        new_slacks,
                                        var_types,
                                        lp_start_time,
                                        work_estimate,
                                        std::numeric_limits<f_t>::infinity(),
                                        basic_list,
                                        nonbasic_list,
                                        leaf_vstatus,
                                        leaf_solution,
                                        basis_factors,
                                        num_fractional,
                                        fractional,
                                        integer_pivot_work);
          if (num_integer_increased > 0) {
            stats.integer_pivots.fetch_add(num_integer_increased, std::memory_order::relaxed);
          }
        }
      }
    }
  }

#ifdef LOG_NODE_SIMPLEX
  lp_settings.log.printf("\nLP status: %d\n\n", lp_status);
#endif

  return lp_status;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class branch_and_bound_worker_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
