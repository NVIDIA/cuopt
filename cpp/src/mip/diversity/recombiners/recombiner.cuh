/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "recombiner_configs.hpp"
#include "recombiner_stats.hpp"

#include <mip/solution/solution.cuh>
#include <mip/solver.cuh>
#include <mip/utils.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/device_utils.cuh>
#include <utilities/seed_generator.cuh>

#include <thrust/random.h>
#include <thrust/set_operations.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

#include <unordered_set>

namespace cuopt::linear_programming::detail {

// checks whether the values of a variable are equal when we consider them in a diversity
// measurement context
template <typename f_t>
HDI bool diverse_equal(f_t val1, f_t val2, f_t lb, f_t ub, bool is_integer, f_t int_tol)
{
  f_t range;
  if (is_integer) {
    range = int_tol;
  } else {
    range = (ub - lb) / 10;
    range = min(range, 0.25);
  }
  return integer_equal<f_t>(val1, val2, range);
}

template <typename i_t, typename f_t>
__global__ void assign_same_variables_kernel(typename solution_t<i_t, f_t>::view_t a,
                                             typename solution_t<i_t, f_t>::view_t b,
                                             typename solution_t<i_t, f_t>::view_t offspring,
                                             raft::device_span<i_t> remaining_indices,
                                             i_t* n_remaining)
{
  if (TH_ID_X >= a.problem.n_integer_vars) return;
  const i_t var_idx = a.problem.integer_indices[TH_ID_X];

  if (integer_equal<f_t>(
        a.assignment[var_idx], b.assignment[var_idx], a.problem.tolerances.integrality_tolerance)) {
    offspring.assignment[var_idx] = a.assignment[var_idx];
  } else {
    i_t idx                = atomicAdd(n_remaining, 1);
    remaining_indices[idx] = var_idx;
  }
}

template <typename i_t, typename f_t>
class recombiner_t {
 public:
  recombiner_t(mip_solver_context_t<i_t, f_t>& context_,
               i_t n_integer_vars,
               const raft::handle_t* handle_ptr)
    : context(context_),
      remaining_indices(n_integer_vars, handle_ptr->get_stream()),
      n_remaining(handle_ptr->get_stream())
  {
  }

  void reset(i_t n_integer_vars, const raft::handle_t* handle_ptr)
  {
    n_remaining.set_value_to_zero_async(handle_ptr->get_stream());
    remaining_indices.resize(n_integer_vars, handle_ptr->get_stream());
  }

  i_t assign_same_integer_values(solution_t<i_t, f_t>& a,
                                 solution_t<i_t, f_t>& b,
                                 solution_t<i_t, f_t>& offspring)
  {
    reset(a.problem_ptr->n_integer_vars, a.handle_ptr);
    const i_t TPB = 128;
    i_t n_blocks  = (a.problem_ptr->n_integer_vars + TPB - 1) / TPB;
    assign_same_variables_kernel<i_t, f_t>
      <<<n_blocks, TPB, 0, a.handle_ptr->get_stream()>>>(a.view(),
                                                         b.view(),
                                                         offspring.view(),
                                                         cuopt::make_span(remaining_indices),
                                                         n_remaining.data());
    i_t remaining_variables = this->n_remaining.value(a.handle_ptr->get_stream());

    auto vec_remaining_indices =
      host_copy(this->remaining_indices.data(), remaining_variables, a.handle_ptr->get_stream());
    auto vec_objective_coeffs = host_copy(offspring.problem_ptr->objective_coefficients.data(),
                                          offspring.problem_ptr->n_variables,
                                          a.handle_ptr->get_stream());
    auto integer_indices      = host_copy(offspring.problem_ptr->integer_indices.data(),
                                     offspring.problem_ptr->n_integer_vars,
                                     a.handle_ptr->get_stream());
    std::vector<i_t> objective_indices;
    for (size_t i = 0; i < vec_objective_coeffs.size(); ++i) {
      if (vec_objective_coeffs[i] != 0 &&
          std::find(integer_indices.begin(), integer_indices.end(), i) != integer_indices.end()) {
        objective_indices.push_back(i);
      }
    }
    std::vector<i_t> objective_indices_in_subproblem;
    for (auto var : vec_remaining_indices) {
      if (vec_objective_coeffs[var] != 0) { objective_indices_in_subproblem.push_back(var); }
    }
    // needed for set_difference
    std::sort(objective_indices_in_subproblem.begin(), objective_indices_in_subproblem.end());
    CUOPT_LOG_DEBUG("n_objective_vars in different vars %d n_objective_vars %d",
                    objective_indices_in_subproblem.size(),
                    objective_indices.size());
    if (objective_indices.size() > 0 &&
        objective_indices_in_subproblem.size() < 0.4 * remaining_variables) {
      std::default_random_engine rng_host(cuopt::seed_generator::get_seed());
      std::vector<i_t> objective_indices_not_in_subproblem;
      std::set_difference(objective_indices.begin(),
                          objective_indices.end(),
                          objective_indices_in_subproblem.begin(),
                          objective_indices_in_subproblem.end(),
                          std::back_inserter(objective_indices_not_in_subproblem));
      std::shuffle(objective_indices_not_in_subproblem.begin(),
                   objective_indices_not_in_subproblem.end(),
                   rng_host);
      for (auto var : objective_indices_not_in_subproblem) {
        objective_indices_in_subproblem.push_back(var);
        vec_remaining_indices.push_back(var);
        if (objective_indices_in_subproblem.size() >= 0.4 * remaining_variables) { break; }
      }
    }
    raft::copy(this->remaining_indices.data(),
               vec_remaining_indices.data(),
               vec_remaining_indices.size(),
               a.handle_ptr->get_stream());
    return vec_remaining_indices.size();
  }

  bool check_if_offspring_is_same_as_parents(solution_t<i_t, f_t>& offspring,
                                             solution_t<i_t, f_t>& a,
                                             solution_t<i_t, f_t>& b)
  {
    bool equals_a = check_integer_equal_on_indices(offspring.problem_ptr->integer_indices,
                                                   offspring.assignment,
                                                   a.assignment,
                                                   a.problem_ptr->tolerances.integrality_tolerance,
                                                   a.handle_ptr);
    if (equals_a) {
      CUOPT_LOG_DEBUG("Offspring is same as parent guiding!");
      return true;
    }
    bool equals_b = check_integer_equal_on_indices(offspring.problem_ptr->integer_indices,
                                                   offspring.assignment,
                                                   b.assignment,
                                                   b.problem_ptr->tolerances.integrality_tolerance,
                                                   b.handle_ptr);
    if (equals_b) {
      CUOPT_LOG_DEBUG("Offspring is same as parent other!");
      return true;
    }
    return false;
  }

  void compute_vars_to_fix(solution_t<i_t, f_t>& offspring,
                           rmm::device_uvector<i_t>& vars_to_fix,
                           i_t n_vars_from_other,
                           i_t n_vars_from_guiding)
  {
    vars_to_fix.resize(n_vars_from_guiding, offspring.handle_ptr->get_stream());
    // set difference needs two sorted arrays
    thrust::sort(offspring.handle_ptr->get_thrust_policy(),
                 this->remaining_indices.data(),
                 this->remaining_indices.data() + n_vars_from_other);
    cuopt_assert((thrust::is_sorted(offspring.handle_ptr->get_thrust_policy(),
                                    offspring.problem_ptr->integer_indices.begin(),
                                    offspring.problem_ptr->integer_indices.end())),
                 "vars_to_fix should be sorted!");
    // get the variables to fix (common variables)
    auto iter = thrust::set_difference(offspring.handle_ptr->get_thrust_policy(),
                                       offspring.problem_ptr->integer_indices.begin(),
                                       offspring.problem_ptr->integer_indices.end(),
                                       this->remaining_indices.data(),
                                       this->remaining_indices.data() + n_vars_from_other,
                                       vars_to_fix.begin());
    cuopt_assert(iter - vars_to_fix.begin() == n_vars_from_guiding, "The size should match!");
    cuopt_assert((thrust::is_sorted(offspring.handle_ptr->get_thrust_policy(),
                                    vars_to_fix.data(),
                                    vars_to_fix.data() + n_vars_from_guiding)),
                 "vars_to_fix should be sorted!");
  }

  static void init_enabled_recombiners(const problem_t<i_t, f_t>& problem)
  {
    std::unordered_set<recombiner_enum_t> enabled_recombiners;
    for (auto recombiner : recombiner_types) {
      enabled_recombiners.insert(recombiner);
    }
    if (problem.expensive_to_fix_vars) {
      enabled_recombiners.erase(recombiner_enum_t::FP);
      enabled_recombiners.erase(recombiner_enum_t::SUB_MIP);
    }
    // check the size of the continous vars
    if (problem.n_variables - problem.n_integer_vars >
        (i_t)sub_mip_recombiner_config_t::max_continuous_vars) {
      enabled_recombiners.erase(recombiner_enum_t::SUB_MIP);
    }
    recombiner_t::enabled_recombiners =
      std::vector<recombiner_enum_t>(enabled_recombiners.begin(), enabled_recombiners.end());
  }

  mip_solver_context_t<i_t, f_t>& context;
  rmm::device_uvector<i_t> remaining_indices;
  rmm::device_scalar<i_t> n_remaining;
  static std::vector<recombiner_enum_t> enabled_recombiners;
};

}  // namespace cuopt::linear_programming::detail
