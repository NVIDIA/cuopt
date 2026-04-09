/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "recombiner_configs.hpp"
#include "recombiner_stats.hpp"

#include <mip_heuristics/solution/solution.cuh>
#include <mip_heuristics/solver.cuh>
#include <mip_heuristics/utils.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/determinism_log.hpp>
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
  static const char* recombiner_name(recombiner_enum_t recombiner)
  {
    switch (recombiner) {
      case recombiner_enum_t::BOUND_PROP: return "BOUND_PROP";
      case recombiner_enum_t::FP: return "FP";
      case recombiner_enum_t::LINE_SEGMENT: return "LINE_SEGMENT";
      case recombiner_enum_t::SUB_MIP: return "SUB_MIP";
      case recombiner_enum_t::SIZE: return "SIZE";
    }
    return "UNKNOWN";
  }

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
    // Sort the indices to resolve nondeterministic order due to atomicAdd
    thrust::sort(a.handle_ptr->get_thrust_policy(),
                 this->remaining_indices.data(),
                 this->remaining_indices.data() + remaining_variables);

    CUOPT_DETERMINISM_LOG(
      "remaining indices hash 0x%x, size %d",
      detail::compute_hash(make_span(this->remaining_indices), a.handle_ptr->get_stream()),
      remaining_variables);

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
    CUOPT_DETERMINISM_LOG(
      "remaining indices hash 0x%x",
      detail::compute_hash(make_span(this->remaining_indices), offspring.handle_ptr->get_stream()));
    CUOPT_DETERMINISM_LOG("integer_indices hash 0x%x",
                          detail::compute_hash(make_span(offspring.problem_ptr->integer_indices),
                                               offspring.handle_ptr->get_stream()));
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

  static void init_enabled_recombiners(mip_solver_context_t<i_t, f_t>& context,
                                       const problem_t<i_t, f_t>& problem,
                                       int user_enabled_mask = -1)
  {
    std::unordered_set<recombiner_enum_t> enabled_recombiners;
    const bool disable_fp_and_submip_for_expensive_fix = problem.expensive_to_fix_vars;
    const i_t n_continuous_vars = problem.n_variables - problem.n_integer_vars;
    const bool disable_submip_for_continuous_limit =
      n_continuous_vars > (i_t)sub_mip_recombiner_config_t::max_continuous_vars;
    const bool disable_submip_for_determinism =
      (context.settings.determinism_mode & CUOPT_DETERMINISM_GPU_HEURISTICS) != 0;
    for (auto recombiner : recombiner_types) {
      if (user_enabled_mask >= 0 && !(user_enabled_mask & (1 << (uint32_t)recombiner))) {
        continue;
      }
      enabled_recombiners.insert(recombiner);
    }
    if (disable_fp_and_submip_for_expensive_fix) {
      enabled_recombiners.erase(recombiner_enum_t::FP);
      enabled_recombiners.erase(recombiner_enum_t::SUB_MIP);
    }
    // check the size of the continous vars
    if (disable_submip_for_continuous_limit) {
      enabled_recombiners.erase(recombiner_enum_t::SUB_MIP);
    }
    // submip not supported in deterministic mode yet
    if (disable_submip_for_determinism) { enabled_recombiners.erase(recombiner_enum_t::SUB_MIP); }
    recombiner_t::enabled_recombiners =
      std::vector<recombiner_enum_t>(enabled_recombiners.begin(), enabled_recombiners.end());
    cuopt_assert(!recombiner_t::enabled_recombiners.empty(), "No recombiners enabled after init");
    std::string order_str;
    for (size_t i = 0; i < recombiner_t::enabled_recombiners.size(); ++i) {
      if (i > 0) { order_str += ','; }
      order_str += recombiner_name(recombiner_t::enabled_recombiners[i]);
    }
    CUOPT_DETERMINISM_LOG(
      "Deterministic recombiner init: expensive_to_fix=%d n_continuous=%d "
      "max_continuous=%zu disable_fp_submip_expensive=%d "
      "disable_submip_continuous=%d disable_submip_deterministic=%d size=%zu "
      "order=[%s]",
      (int)problem.expensive_to_fix_vars,
      (int)n_continuous_vars,
      sub_mip_recombiner_config_t::max_continuous_vars,
      (int)disable_fp_and_submip_for_expensive_fix,
      (int)disable_submip_for_continuous_limit,
      (int)disable_submip_for_determinism,
      recombiner_t::enabled_recombiners.size(),
      order_str.c_str());
  }

  mip_solver_context_t<i_t, f_t>& context;
  rmm::device_uvector<i_t> remaining_indices;
  rmm::device_scalar<i_t> n_remaining;
  static std::vector<recombiner_enum_t> enabled_recombiners;
};

}  // namespace cuopt::linear_programming::detail
