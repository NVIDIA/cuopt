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

#include "simple_rounding.cuh"
#include "simple_rounding_kernels.cuh"

#include <mip/mip_constants.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/seed_generator.cuh>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
bool check_brute_force_rounding(solution_t<i_t, f_t>& solution)
{
  i_t TPB        = 128;
  i_t n_integers = solution.compute_number_of_integers();
  CUOPT_LOG_TRACE("before rounding n_integers %d total n_integers %d",
                  n_integers,
                  solution.problem_ptr->n_integer_vars);
  i_t n_integers_to_round = solution.problem_ptr->n_integer_vars - n_integers;
  if (n_integers_to_round == 0) { return solution.compute_feasibility(); }
  constexpr i_t brute_force_rounding_threshold = 8;
  if (n_integers_to_round <= brute_force_rounding_threshold) {
    solution.compute_constraints();
    i_t n_configs = pow(2, n_integers_to_round);
    i_t n_blocks  = (n_configs + TPB - 1) / TPB;
    // extract the variables to round
    rmm::device_uvector<i_t> var_map(n_integers_to_round, solution.handle_ptr->get_stream());
    rmm::device_uvector<f_t> constraint_buf(n_configs * solution.problem_ptr->n_constraints,
                                            solution.handle_ptr->get_stream());
    rmm::device_scalar<i_t> best_config(-1, solution.handle_ptr->get_stream());
    thrust::copy_if(
      solution.handle_ptr->get_thrust_policy(),
      solution.problem_ptr->integer_indices.begin(),
      solution.problem_ptr->integer_indices.end(),
      var_map.data(),
      [assignment = solution.assignment.data(), pb = solution.problem_ptr->view()] __device__(
        i_t var_id) { return !pb.is_integer(assignment[var_id]); });

    // // try all configs in parallel and compute feasibility
    brute_force_check_kernel<i_t, f_t>
      <<<n_blocks, TPB, 0, solution.handle_ptr->get_stream()>>>(solution.view(),
                                                                n_integers_to_round,
                                                                cuopt::make_span(var_map),
                                                                cuopt::make_span(constraint_buf),
                                                                best_config.data());
    if (best_config.value(solution.handle_ptr->get_stream()) != -1) {
      CUOPT_LOG_DEBUG("Feasible found during brute force rounding!");
      // apply the feasible rounding
      apply_feasible_rounding_kernel<i_t, f_t><<<1, TPB, 0, solution.handle_ptr->get_stream()>>>(
        solution.view(), n_integers_to_round, cuopt::make_span(var_map), best_config.data());
      solution.handle_ptr->sync_stream();
      bool feas = solution.compute_feasibility();
      cuopt_assert(feas, "Solution must be feasible!");
      return true;
    }
  }
  return false;
}

template <typename i_t, typename f_t>
bool invoke_simple_rounding(solution_t<i_t, f_t>& solution)
{
  solution.compute_feasibility();

  solution_t<i_t, f_t> sol_copy(*solution.problem_ptr);
  sol_copy.copy_from(solution);

  rmm::device_scalar<bool> successful(true, solution.handle_ptr->get_stream());
  i_t TPB = 128;
  simple_rounding_kernel<i_t, f_t>
    <<<2048, TPB, 0, solution.handle_ptr->get_stream()>>>(solution.view(), successful.data());
  if (!successful.value(solution.handle_ptr->get_stream())) {
    CUOPT_LOG_DEBUG("Simple rounding failed");
    solution.copy_from(sol_copy);
    return false;
  } else {
    solution.compute_feasibility();
    CUOPT_LOG_DEBUG("Simple rounding successful");
    CUOPT_LOG_DEBUG("objective %g, feas %d, excess %g, int violation %g",
                    solution.get_user_objective(),
                    solution.get_feasible(),
                    solution.get_total_excess(),
                    solution.compute_max_int_violation());
    return true;
  }
}

template <typename i_t, typename f_t>
void invoke_round_nearest(solution_t<i_t, f_t>& solution)
{
  i_t TPB                     = 128;
  bool brute_force_found_feas = check_brute_force_rounding(solution);
  if (brute_force_found_feas) { return; }

  bool simple_round = invoke_simple_rounding(solution);
  if (simple_round) { return; }

  i_t n_blocks = (solution.problem_ptr->n_integer_vars + TPB - 1) / TPB;
  nearest_rounding_kernel<i_t, f_t><<<n_blocks, TPB, 0, solution.handle_ptr->get_stream()>>>(
    solution.view(), cuopt::seed_generator::get_seed());
  RAFT_CHECK_CUDA(solution.handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void invoke_random_round_nearest(solution_t<i_t, f_t>& solution, i_t n_target_random_rounds)
{
  i_t TPB        = 128;
  i_t n_blocks   = (solution.problem_ptr->n_variables + TPB - 1) / TPB;
  i_t n_integers = solution.compute_number_of_integers();
  CUOPT_LOG_TRACE("before random roundin n_integers %d total n_integers %d",
                  n_integers,
                  solution.problem_ptr->n_integer_vars);
  rmm::device_scalar<i_t> n_randomly_rounded(0, solution.handle_ptr->get_stream());
  random_nearest_rounding_kernel<i_t, f_t><<<n_blocks, TPB, 0, solution.handle_ptr->get_stream()>>>(
    solution.view(), cuopt::seed_generator::get_seed(), n_randomly_rounded.data());
  i_t h_n_random_rounds = n_randomly_rounded.value(solution.handle_ptr->get_stream());
  CUOPT_LOG_TRACE("Randomly rounded integers %d", h_n_random_rounds);
  i_t additional_roundings_needed = n_target_random_rounds - h_n_random_rounds;
  if (additional_roundings_needed > 0) {
    // TODO sort the remaining integers with fractionality and round them randomly
    rmm::device_uvector<i_t> shuffled_indices(solution.problem_ptr->integer_indices,
                                              solution.handle_ptr->get_stream());
    thrust::default_random_engine rng(cuopt::seed_generator::get_seed());
    // from the remaining integers, populate randomly.
    thrust::shuffle(solution.handle_ptr->get_thrust_policy(),
                    shuffled_indices.begin(),
                    shuffled_indices.end(),
                    rng);
    random_rounding_kernel<i_t, f_t>
      <<<1, 1, 0, solution.handle_ptr->get_stream()>>>(solution.view(),
                                                       cuopt::seed_generator::get_seed(),
                                                       shuffled_indices.data(),
                                                       n_randomly_rounded.data(),
                                                       additional_roundings_needed);
    h_n_random_rounds = n_randomly_rounded.value(solution.handle_ptr->get_stream());
    CUOPT_LOG_TRACE("Randomly rounded integers, after adding close integers too %d",
                    h_n_random_rounds);
  }
  solution.round_nearest();
  RAFT_CHECK_CUDA(solution.handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void invoke_correct_integers(solution_t<i_t, f_t>& solution, f_t tol)
{
  const auto problem_view = solution.problem_ptr->view();
  f_t* assignment_ptr     = solution.assignment.data();
  thrust::for_each(solution.handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(solution.problem_ptr->n_variables),
                   [problem_view, assignment_ptr, tol] __device__(i_t var) {
                     if (problem_view.is_integer_var(var) &&
                         raft::abs(round(assignment_ptr[var]) - (assignment_ptr[var])) <= tol)
                       assignment_ptr[var] = floor(assignment_ptr[var] + tol);
                   });
}

#define INSTANTIATE(F_TYPE)                                                                  \
  template bool check_brute_force_rounding<int, F_TYPE>(solution_t<int, F_TYPE> & solution); \
  template void invoke_random_round_nearest<int, F_TYPE>(solution_t<int, F_TYPE> & solution, \
                                                         int n_target_random_rounds);        \
  template void invoke_round_nearest<int, F_TYPE>(solution_t<int, F_TYPE> & solution);       \
  template bool invoke_simple_rounding<int, F_TYPE>(solution_t<int, F_TYPE> & solution);     \
  template void invoke_correct_integers<int, F_TYPE>(solution_t<int, F_TYPE> & solution,     \
                                                     F_TYPE tol);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
