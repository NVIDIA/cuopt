/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "presolve_data.cuh"

#include "problem.cuh"

#include <mip/solution/solution.cuh>
#include <utilities/copy_helpers.hpp>

#include <mip/mip_constants.hpp>
#include <mip/presolve/third_party_presolve.hpp>

#include <mip/utils.cuh>

#include <raft/core/nvtx.hpp>

#include <thrust/gather.h>

namespace cuopt {
namespace linear_programming::detail {

template <typename i_t, typename f_t>
bool presolve_data_t<i_t, f_t>::pre_process_assignment(problem_t<i_t, f_t>& problem,
                                                       rmm::device_uvector<f_t>& assignment)
{
  raft::common::nvtx::range fun_scope("pre_process_assignment");
  auto has_nans = cuopt::linear_programming::detail::has_nans(problem.handle_ptr, assignment);
  if (has_nans) {
    CUOPT_LOG_DEBUG("Solution discarded due to nans");
    return false;
  }
  cuopt_assert(assignment.size() == problem.original_problem_ptr->get_n_variables(),
               "size mismatch");

  // NOTE: We can apply substitutions and fixed variables here.
  // However, variable fixing and substitutions are already included in the problem and objective
  // offsets. So the only advantage of applying them here would be to check the compatibility of the
  // assignment values. It is not so important as we would be correcting the infeasibility by
  // keeping the correct substitution.

  // create a temp assignment with the var size after bounds standardization (free vars added)
  rmm::device_uvector<f_t> temp_assignment(additional_var_used.size(),
                                           problem.handle_ptr->get_stream());
  // copy the assignment to the first part(the original variable count) of the temp_assignment
  raft::copy(
    temp_assignment.data(), assignment.data(), assignment.size(), problem.handle_ptr->get_stream());
  auto d_additional_var_used =
    cuopt::device_copy(additional_var_used, problem.handle_ptr->get_stream());
  auto d_additional_var_id_per_var =
    cuopt::device_copy(additional_var_id_per_var, problem.handle_ptr->get_stream());

  thrust::for_each(
    problem.handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator<i_t>(0),
    thrust::make_counting_iterator<i_t>(problem.original_problem_ptr->get_n_variables()),
    [additional_var_used       = d_additional_var_used.data(),
     additional_var_id_per_var = d_additional_var_id_per_var.data(),
     assgn                     = temp_assignment.data()] __device__(auto idx) {
      if (additional_var_used[idx]) {
        cuopt_assert(additional_var_id_per_var[idx] != -1, "additional_var_id_per_var is not set");
        // We have two non-negative variables y and z that simulate a free variable
        // x. If the value of x is negative, we can set z to be something higher than
        // y. If the value of  x is positive we can set y greater than z
        assgn[additional_var_id_per_var[idx]] = (assgn[idx] < 0 ? -assgn[idx] : 0.);
        assgn[idx] += assgn[additional_var_id_per_var[idx]];
      }
    });
  assignment.resize(problem.n_variables, problem.handle_ptr->get_stream());
  assignment.shrink_to_fit(problem.handle_ptr->get_stream());
  cuopt_assert(variable_mapping.size() == problem.n_variables, "size mismatch");
  thrust::gather(problem.handle_ptr->get_thrust_policy(),
                 variable_mapping.begin(),
                 variable_mapping.end(),
                 temp_assignment.begin(),
                 assignment.begin());
  problem.handle_ptr->sync_stream();

  auto has_integrality_discrepancy = cuopt::linear_programming::detail::has_integrality_discrepancy(
    problem.handle_ptr,
    problem.integer_indices,
    assignment,
    problem.tolerances.integrality_tolerance);
  if (has_integrality_discrepancy) {
    CUOPT_LOG_DEBUG("Solution discarded due to integrality discrepancy");
    return false;
  }

  auto has_variable_bounds_violation =
    cuopt::linear_programming::detail::has_variable_bounds_violation(
      problem.handle_ptr, assignment, &problem);
  if (has_variable_bounds_violation) {
    CUOPT_LOG_DEBUG("Solution discarded due to variable bounds violation");
    return false;
  }
  return true;
}

// this function is used to post process the assignment
// it removes the additional variable for free variables
// and expands the assignment to the original variable dimension
template <typename i_t, typename f_t>
void presolve_data_t<i_t, f_t>::post_process_assignment(
  problem_t<i_t, f_t>& problem,
  rmm::device_uvector<f_t>& current_assignment,
  bool resize_to_original_problem)
{
  raft::common::nvtx::range fun_scope("post_process_assignment");
  cuopt_assert(current_assignment.size() == variable_mapping.size(), "size mismatch");
  auto assgn       = make_span(current_assignment);
  auto fixed_assgn = make_span(fixed_var_assignment);
  auto var_map     = make_span(variable_mapping);
  if (current_assignment.size() > 0) {
    thrust::for_each(problem.handle_ptr->get_thrust_policy(),
                     thrust::make_counting_iterator<i_t>(0),
                     thrust::make_counting_iterator<i_t>(current_assignment.size()),
                     [fixed_assgn, var_map, assgn] __device__(auto idx) {
                       fixed_assgn[var_map[idx]] = assgn[idx];
                     });
  }
  expand_device_copy(current_assignment, fixed_var_assignment, problem.handle_ptr->get_stream());
  auto h_assignment = cuopt::host_copy(current_assignment, problem.handle_ptr->get_stream());
  cuopt_assert(additional_var_id_per_var.size() == h_assignment.size(), "Size mismatch");
  cuopt_assert(additional_var_used.size() == h_assignment.size(), "Size mismatch");
  for (i_t i = 0; i < (i_t)h_assignment.size(); ++i) {
    if (additional_var_used[i]) {
      cuopt_assert(additional_var_id_per_var[i] != -1, "additional_var_id_per_var is not set");
      h_assignment[i] -= h_assignment[additional_var_id_per_var[i]];
    }
  }

  // Apply variable substitutions from probing: x_substituted = offset + coefficient *
  // x_substituting
  for (const auto& sub : variable_substitutions) {
    cuopt_assert(sub.substituted_var < (i_t)h_assignment.size(), "substituted_var out of bounds");
    cuopt_assert(sub.substituting_var < (i_t)h_assignment.size(), "substituting_var out of bounds");
    h_assignment[sub.substituted_var] =
      sub.offset + sub.coefficient * h_assignment[sub.substituting_var];
    CUOPT_LOG_DEBUG("Post-process substitution: x[%d] = %f + %f * x[%d] = %f",
                    sub.substituted_var,
                    sub.offset,
                    sub.coefficient,
                    sub.substituting_var,
                    h_assignment[sub.substituted_var]);
  }

  raft::copy(current_assignment.data(),
             h_assignment.data(),
             h_assignment.size(),
             problem.handle_ptr->get_stream());
  // this separate resizing is needed because of the callback
  if (resize_to_original_problem) {
    current_assignment.resize(problem.original_problem_ptr->get_n_variables(),
                              problem.handle_ptr->get_stream());
  }
}

template <typename i_t, typename f_t>
void presolve_data_t<i_t, f_t>::post_process_solution(problem_t<i_t, f_t>& problem,
                                                      solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("post_process_solution");
  post_process_assignment(problem, solution.assignment);
  // this is for resizing other fields such as excess, slack so that we can compute the feasibility
  solution.resize_to_original_problem();
  problem.handle_ptr->sync_stream();
  solution.post_process_completed = true;
}

template <typename i_t, typename f_t>
void presolve_data_t<i_t, f_t>::set_papilo_presolve_data(
  const third_party_presolve_t<i_t, f_t>* presolver_ptr,
  std::vector<i_t> reduced_to_original,
  std::vector<i_t> original_to_reduced,
  i_t original_num_variables)
{
  if (original_num_variables <= 0) {
    CUOPT_LOG_DEBUG("Papilo presolve data invalid: original_num_variables=%d",
                    original_num_variables);
    return;
  }
  if (original_to_reduced.empty()) {
    CUOPT_LOG_DEBUG("Papilo presolve data invalid: original_to_reduced is empty");
    return;
  }
  if (original_to_reduced.size() != static_cast<size_t>(original_num_variables)) {
    CUOPT_LOG_DEBUG(
      "Papilo presolve data invalid: original_to_reduced.size()=%zu "
      "original_num_variables=%d",
      original_to_reduced.size(),
      original_num_variables);
    return;
  }
  for (size_t i = 0; i < reduced_to_original.size(); ++i) {
    const auto original_idx = reduced_to_original[i];
    if (original_idx < 0 || original_idx >= original_num_variables) {
      CUOPT_LOG_DEBUG(
        "Papilo presolve data invalid: reduced_to_original[%zu]=%d out of range [0,%d)",
        i,
        original_idx,
        original_num_variables);
      return;
    }
  }
  for (size_t i = 0; i < original_to_reduced.size(); ++i) {
    const auto reduced_idx = original_to_reduced[i];
    if (reduced_idx < -1 || reduced_idx >= static_cast<i_t>(reduced_to_original.size())) {
      CUOPT_LOG_DEBUG(
        "Papilo presolve data invalid: original_to_reduced[%zu]=%d out of range [-1,%zu)",
        i,
        reduced_idx,
        reduced_to_original.size());
      return;
    }
  }

  papilo_presolve_ptr            = presolver_ptr;
  papilo_reduced_to_original_map = std::move(reduced_to_original);
  papilo_original_to_reduced_map = std::move(original_to_reduced);
  papilo_original_num_variables  = original_num_variables;
}

template <typename i_t, typename f_t>
void presolve_data_t<i_t, f_t>::papilo_uncrush_assignment(
  problem_t<i_t, f_t>& problem, rmm::device_uvector<f_t>& assignment) const
{
  if (papilo_presolve_ptr == nullptr) {
    CUOPT_LOG_INFO("Papilo presolve data not set, skipping uncrushing assignment");
    return;
  }
  cuopt_assert(assignment.size() == papilo_reduced_to_original_map.size(),
               "Papilo uncrush assignment size mismatch");
  auto h_assignment = cuopt::host_copy(assignment, problem.handle_ptr->get_stream());
  std::vector<f_t> full_assignment;
  papilo_presolve_ptr->uncrush_primal_solution(h_assignment, full_assignment);
  assignment.resize(full_assignment.size(), problem.handle_ptr->get_stream());
  raft::copy(assignment.data(),
             full_assignment.data(),
             full_assignment.size(),
             problem.handle_ptr->get_stream());
  problem.handle_ptr->sync_stream();
}

#if MIP_INSTANTIATE_FLOAT
template class presolve_data_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class presolve_data_t<int, double>;
#endif

}  // namespace linear_programming::detail
}  // namespace cuopt
