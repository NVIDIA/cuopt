/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#pragma once

#include "cuopt/linear_programming/mip/solver_settings.hpp"
#include "recombiner.cuh"

#include <dual_simplex/branch_and_bound.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solve.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class sub_mip_recombiner_t : public recombiner_t<i_t, f_t> {
 public:
  sub_mip_recombiner_t(mip_solver_context_t<i_t, f_t>& context,
                       population_t<i_t, f_t>& population,
                       i_t n_vars,
                       const raft::handle_t* handle_ptr)
    : recombiner_t<i_t, f_t>(context, n_vars, handle_ptr),
      vars_to_fix(n_vars, handle_ptr->get_stream()),
      context(context),
      population(population)
  {
  }

  void solution_callback(std::vector<f_t>& solution, f_t objective)
  {
    CUOPT_LOG_DEBUG("SUBMIP added solution with objective %.16e", objective);
    solution_vector.push_back(solution);
  }

  std::pair<solution_t<i_t, f_t>, bool> recombine(solution_t<i_t, f_t>& a,
                                                  solution_t<i_t, f_t>& b,
                                                  const weight_t<i_t, f_t>& weights)
  {
    raft::common::nvtx::range fun_scope("Sub-MIP recombiner");
    solution_vector.clear();
    auto& guiding_solution = a.get_feasible() ? a : b;
    auto& other_solution   = a.get_feasible() ? b : a;
    // copy the solution from A
    solution_t<i_t, f_t> offspring(guiding_solution);
    // find same values and populate it to offspring
    i_t n_different_vars =
      this->assign_same_integer_values(guiding_solution, other_solution, offspring);
    CUOPT_LOG_DEBUG("SUB_MIP rec: Number of different variables %d MAX_VARS %d",
                    n_different_vars,
                    sub_mip_recombiner_config_t::max_n_of_vars_from_other);
    i_t n_vars_from_other = n_different_vars;
    if (n_vars_from_other > (i_t)sub_mip_recombiner_config_t::max_n_of_vars_from_other) {
      n_vars_from_other = sub_mip_recombiner_config_t::max_n_of_vars_from_other;
      thrust::default_random_engine g{(unsigned int)cuopt::seed_generator::get_seed()};
      thrust::shuffle(a.handle_ptr->get_thrust_policy(),
                      this->remaining_indices.data(),
                      this->remaining_indices.data() + n_different_vars,
                      g);
    }
    i_t n_vars_from_guiding = a.problem_ptr->n_integer_vars - n_vars_from_other;
    if (n_vars_from_other == 0 || n_vars_from_guiding == 0) {
      CUOPT_LOG_DEBUG("Returning false because all vars are common or different");
      return std::make_pair(offspring, false);
    }
    CUOPT_LOG_DEBUG(
      "n_vars_from_guiding %d n_vars_from_other %d", n_vars_from_guiding, n_vars_from_other);
    this->compute_vars_to_fix(offspring, vars_to_fix, n_vars_from_other, n_vars_from_guiding);
    auto [fixed_problem, fixed_assignment, variable_map] = offspring.fix_variables(vars_to_fix);
    pdlp_initial_scaling_strategy_t<i_t, f_t> scaling(
      fixed_problem.handle_ptr,
      fixed_problem,
      pdlp_hyper_params::default_l_inf_ruiz_iterations,
      (f_t)pdlp_hyper_params::default_alpha_pock_chambolle_rescaling,
      fixed_problem.reverse_coefficients,
      fixed_problem.reverse_offsets,
      fixed_problem.reverse_constraints,
      nullptr,
      true);
    scaling.scale_problem();
    fixed_problem.presolve_data.reset_additional_vars(fixed_problem, offspring.handle_ptr);
    fixed_problem.presolve_data.initialize_var_mapping(fixed_problem, offspring.handle_ptr);
    trivial_presolve(fixed_problem);
    fixed_problem.check_problem_representation(true);
    // brute force rounding threshold is 8
    const bool run_sub_mip                             = fixed_problem.n_integer_vars > 8;
    dual_simplex::mip_status_t branch_and_bound_status = dual_simplex::mip_status_t::UNSET;
    dual_simplex::mip_solution_t<i_t, f_t> branch_and_bound_solution(1);
    if (run_sub_mip) {
      // run sub-mip
      namespace dual_simplex = cuopt::linear_programming::dual_simplex;
      dual_simplex::user_problem_t<i_t, f_t> branch_and_bound_problem;
      dual_simplex::simplex_solver_settings_t<i_t, f_t> branch_and_bound_settings;
      fixed_problem.get_host_user_problem(branch_and_bound_problem);
      branch_and_bound_solution.resize(branch_and_bound_problem.num_cols);
      // Fill in the settings for branch and bound
      branch_and_bound_settings.time_limit = sub_mip_recombiner_config_t::sub_mip_time_limit;
      branch_and_bound_settings.print_presolve_stats = false;
      branch_and_bound_settings.absolute_mip_gap_tol = context.settings.tolerances.absolute_mip_gap;
      branch_and_bound_settings.relative_mip_gap_tol = context.settings.tolerances.relative_mip_gap;
      branch_and_bound_settings.integer_tol     = context.settings.tolerances.integrality_tolerance;
      branch_and_bound_settings.num_threads     = 2;
      branch_and_bound_settings.num_bfs_threads = 1;
      branch_and_bound_settings.num_diving_threads = 1;
      branch_and_bound_settings.solution_callback  = [this](std::vector<f_t>& solution,
                                                           f_t objective) {
        this->solution_callback(solution, objective);
      };

      // disable B&B logs, so that it is not interfering with the main B&B thread
      branch_and_bound_settings.log.log = false;
      dual_simplex::branch_and_bound_t<i_t, f_t> branch_and_bound(branch_and_bound_problem,
                                                                  branch_and_bound_settings);
      branch_and_bound_status = branch_and_bound.solve(branch_and_bound_solution);
      if (solution_vector.size() > 0) {
        cuopt_assert(fixed_assignment.size() == branch_and_bound_solution.x.size(),
                     "Assignment size mismatch");
        CUOPT_LOG_DEBUG("Sub-MIP solution found. Objective %.16e. Status %d",
                        branch_and_bound_solution.objective,
                        int(branch_and_bound_status));
        // first post process the trivial presolve on a device vector
        rmm::device_uvector<f_t> post_processed_solution(branch_and_bound_solution.x.size(),
                                                         offspring.handle_ptr->get_stream());
        raft::copy(post_processed_solution.data(),
                   branch_and_bound_solution.x.data(),
                   branch_and_bound_solution.x.size(),
                   offspring.handle_ptr->get_stream());
        fixed_problem.post_process_assignment(post_processed_solution, false);
        cuopt_assert(post_processed_solution.size() == fixed_assignment.size(),
                     "Assignment size mismatch");
        offspring.handle_ptr->sync_stream();
        std::swap(fixed_assignment, post_processed_solution);
      }
      offspring.handle_ptr->sync_stream();
    }
    if (solution_vector.size() > 0) {
      rmm::device_uvector<f_t> dummy(0, offspring.handle_ptr->get_stream());
      scaling.unscale_solutions(fixed_assignment, dummy);
      // unfix the assignment on given result no matter if it is feasible
      offspring.unfix_variables(fixed_assignment, variable_map);
    } else {
      offspring.round_nearest();
    }
    cuopt_func_call(offspring.test_variable_bounds());
    cuopt_assert(offspring.test_number_all_integer(), "All must be integers after offspring");
    offspring.compute_feasibility();
    // bool same_as_parents = this->check_if_offspring_is_same_as_parents(offspring, a, b);
    // adjust the max_n_of_vars_from_other
    if (n_different_vars > (i_t)sub_mip_recombiner_config_t::max_n_of_vars_from_other) {
      if (branch_and_bound_status == dual_simplex::mip_status_t::OPTIMAL) {
        sub_mip_recombiner_config_t::increase_max_n_of_vars_from_other();
      } else {
        sub_mip_recombiner_config_t::decrease_max_n_of_vars_from_other();
      }
    }
    // try adding all intermediate solutions to the population
    for (const auto& solution : solution_vector) {
      CUOPT_LOG_DEBUG("Adding intermediate submip solution to population");
      solution_t<i_t, f_t> sol(offspring);
      rmm::device_uvector<f_t> fixed_assignment(solution.size(),
                                                offspring.handle_ptr->get_stream());
      raft::copy(fixed_assignment.data(),
                 solution.data(),
                 solution.size(),
                 offspring.handle_ptr->get_stream());
      fixed_problem.post_process_assignment(fixed_assignment, false);
      rmm::device_uvector<f_t> dummy(0, offspring.handle_ptr->get_stream());
      scaling.unscale_solutions(fixed_assignment, dummy);
      sol.unfix_variables(fixed_assignment, variable_map);
      sol.compute_feasibility();
      cuopt_func_call(sol.test_variable_bounds());
      population.add_solution(std::move(sol));
    }
    bool better_cost_than_parents =
      offspring.get_quality(weights) <
      std::min(other_solution.get_quality(weights), guiding_solution.get_quality(weights));
    bool better_feasibility_than_parents = offspring.get_feasible() &&
                                           !other_solution.get_feasible() &&
                                           !guiding_solution.get_feasible();
    if (better_cost_than_parents || better_feasibility_than_parents) {
      CUOPT_LOG_DEBUG("Offspring is feasible or better than both parents");
      return std::make_pair(offspring, true);
    }
    return std::make_pair(offspring, !std::isnan(branch_and_bound_solution.objective));
  }
  rmm::device_uvector<i_t> vars_to_fix;
  mip_solver_context_t<i_t, f_t>& context;
  std::vector<std::vector<f_t>> solution_vector;
  population_t<i_t, f_t>& population;
};

}  // namespace cuopt::linear_programming::detail
