/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <linear_programming/pdlp_constants.hpp>
#include <linear_programming/termination_strategy/termination_strategy.hpp>
#include <mip/mip_constants.hpp>

#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>

#include <raft/common/nvtx.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
pdlp_termination_strategy_t<i_t, f_t>::pdlp_termination_strategy_t(
  raft::handle_t const* handle_ptr,
  problem_t<i_t, f_t>& op_problem,
  cusparse_view_t<i_t, f_t>& cusparse_view,
  const i_t primal_size,
  const i_t dual_size,
  const pdlp_solver_settings_t<i_t, f_t>& settings)
  : handle_ptr_(handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    problem_ptr(&op_problem),
    convergence_information_{handle_ptr_, op_problem, cusparse_view, primal_size, dual_size},
    infeasibility_information_{handle_ptr_,
                               op_problem,
                               cusparse_view,
                               primal_size,
                               dual_size,
                               settings.detect_infeasibility},
    termination_status_{0, stream_view_},
    settings_(settings)
{
}

template <typename i_t, typename f_t>
void pdlp_termination_strategy_t<i_t, f_t>::set_relative_dual_tolerance_factor(
  f_t dual_tolerance_factor)
{
  convergence_information_.set_relative_dual_tolerance_factor(dual_tolerance_factor);
}

template <typename i_t, typename f_t>
void pdlp_termination_strategy_t<i_t, f_t>::set_relative_primal_tolerance_factor(
  f_t primal_tolerance_factor)
{
  convergence_information_.set_relative_primal_tolerance_factor(primal_tolerance_factor);
}

template <typename i_t, typename f_t>
f_t pdlp_termination_strategy_t<i_t, f_t>::get_relative_dual_tolerance_factor() const
{
  return convergence_information_.get_relative_dual_tolerance_factor();
}

template <typename i_t, typename f_t>
f_t pdlp_termination_strategy_t<i_t, f_t>::get_relative_primal_tolerance_factor() const
{
  return convergence_information_.get_relative_primal_tolerance_factor();
}

template <typename i_t, typename f_t>
pdlp_termination_status_t pdlp_termination_strategy_t<i_t, f_t>::evaluate_termination_criteria(
  pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
  rmm::device_uvector<f_t>& primal_iterate,
  rmm::device_uvector<f_t>& dual_iterate,
  const rmm::device_uvector<f_t>& combined_bounds,
  const rmm::device_uvector<f_t>& objective_coefficients)
{
  raft::common::nvtx::range fun_scope("Evaluate termination criteria");

  convergence_information_.compute_convergence_information(current_pdhg_solver,
                                                           primal_iterate,
                                                           dual_iterate,
                                                           combined_bounds,
                                                           objective_coefficients,
                                                           settings_);
  if (settings_.detect_infeasibility) {
    infeasibility_information_.compute_infeasibility_information(
      current_pdhg_solver, primal_iterate, dual_iterate);
  }

  check_termination_criteria();

  i_t tmp;
  raft::copy(&tmp, termination_status_.data(), 1, stream_view_);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));

  return static_cast<pdlp_termination_status_t>(tmp);
}

template <typename i_t, typename f_t>
const convergence_information_t<i_t, f_t>&
pdlp_termination_strategy_t<i_t, f_t>::get_convergence_information() const
{
  return convergence_information_;
}

template <typename i_t, typename f_t>
__global__ void check_termination_criteria_kernel(
  const typename convergence_information_t<i_t, f_t>::view_t convergence_information,
  const typename infeasibility_information_t<i_t, f_t>::view_t infeasibility_information,
  i_t* termination_status,
  typename pdlp_solver_settings_t<i_t, f_t>::tolerances_t tolerance,
  bool infeasibility_detection,
  bool per_constraint_residual)
{
  if (threadIdx.x + blockIdx.x * blockDim.x > 0) { return; }

#ifdef PDLP_VERBOSE_MODE
  printf(
    "Gap : %lf <= %lf [%d] (tolerance.absolute_gap_tolerance %lf + "
    "tolerance.relative_gap_tolerance %lf * convergence_information.abs_objective %lf)\n",
    *convergence_information.gap,
    tolerance.absolute_gap_tolerance +
      tolerance.relative_gap_tolerance * *convergence_information.abs_objective,
    *convergence_information.gap <=
      tolerance.absolute_gap_tolerance +
        tolerance.relative_gap_tolerance * *convergence_information.abs_objective,
    tolerance.absolute_gap_tolerance,
    tolerance.relative_gap_tolerance,
    *convergence_information.abs_objective);

  if (per_constraint_residual) {
    printf(
      "Primal residual : convergence_information.linf_relative_primal_resiprimal %lf < "
      "tolerance.absolute_primal_tolerance %lf\n",
      *convergence_information.relative_l_inf_primal_residual,
      tolerance.absolute_primal_tolerance);
    printf(
      "Dual residual : convergence_information.linf_relative_dual_residual %lf < "
      "tolerance.absolute_dual_tolerance %lf\n",
      *convergence_information.relative_l_inf_dual_residual,
      tolerance.absolute_dual_tolerance);
  } else {
    printf(
      "Primal residual  %lf <= %lf [%d] (tolerance.absolute_primal_tolerance %lf + "
      "tolerance.relative_primal_tolerance %lf * "
      "convergence_information.l2_norm_primal_right_hand_side %lf)\n",
      *convergence_information.l2_primal_residual,
      tolerance.absolute_primal_tolerance +
        tolerance.relative_primal_tolerance *
          *convergence_information.l2_norm_primal_right_hand_side,
      *convergence_information.l2_primal_residual <=
        tolerance.absolute_primal_tolerance +
          tolerance.relative_primal_tolerance *
            *convergence_information.l2_norm_primal_right_hand_side,
      tolerance.absolute_primal_tolerance,
      tolerance.relative_primal_tolerance,
      *convergence_information.l2_norm_primal_right_hand_side);
  }
  printf(
    "Dual residual  %lf <= %lf [%d] (tolerance.absolute_dual_tolerance %lf + "
    "tolerance.relative_dual_tolerance %lf * "
    "convergence_information.l2_norm_primal_linear_objective %lf)\n",
    *convergence_information.l2_dual_residual,
    tolerance.absolute_dual_tolerance +
      tolerance.relative_dual_tolerance * *convergence_information.l2_norm_primal_linear_objective,
    *convergence_information.l2_dual_residual <=
      tolerance.absolute_dual_tolerance +
        tolerance.relative_dual_tolerance *
          *convergence_information.l2_norm_primal_linear_objective,
    tolerance.absolute_dual_tolerance,
    tolerance.relative_dual_tolerance,
    *convergence_information.l2_norm_primal_linear_objective);
#endif

  // By default set to No Termination
  *termination_status = (i_t)pdlp_termination_status_t::NumericalError;

  // test if gap optimal
  const bool optimal_gap =
    *convergence_information.gap <=
    tolerance.absolute_gap_tolerance +
      tolerance.relative_gap_tolerance * *convergence_information.abs_objective;

  // test if respect constraints
  if (per_constraint_residual) {
    // In residual we store l_inf(residual_i - rel * b/c_i)
    const bool primal_feasible = *convergence_information.relative_l_inf_primal_residual <=
                                 tolerance.absolute_primal_tolerance;
    // First check for optimality
    if (*convergence_information.relative_l_inf_dual_residual <=
          tolerance.absolute_dual_tolerance &&
        primal_feasible && optimal_gap) {
      *termination_status = (i_t)pdlp_termination_status_t::Optimal;
      return;
    } else if (primal_feasible)  // If not optimal maybe be at least primal feasible
    {
      *termination_status = (i_t)pdlp_termination_status_t::PrimalFeasible;
      return;
    }
  } else {
    const bool primal_feasible = *convergence_information.l2_primal_residual <=
                                 tolerance.absolute_primal_tolerance +
                                   tolerance.relative_primal_tolerance *
                                     *convergence_information.l2_norm_primal_right_hand_side;
    if (*convergence_information.l2_dual_residual <=
          tolerance.absolute_dual_tolerance +
            tolerance.relative_dual_tolerance *
              *convergence_information.l2_norm_primal_linear_objective &&
        primal_feasible && optimal_gap) {
      *termination_status = (i_t)pdlp_termination_status_t::Optimal;
      return;
    } else if (primal_feasible)  // If not optimal maybe be at least primal feasible
    {
      *termination_status = (i_t)pdlp_termination_status_t::PrimalFeasible;
      return;
    }
  }

  if (infeasibility_detection) {
    // test for primal infeasibility
    if (*infeasibility_information.dual_ray_linear_objective > 0.0 &&
        *infeasibility_information.max_dual_ray_infeasibility /
            *infeasibility_information.dual_ray_linear_objective <=
          tolerance.primal_infeasible_tolerance) {
      *termination_status = (i_t)pdlp_termination_status_t::PrimalInfeasible;
      return;
    }

    // test for dual infeasibility
    //  for QP add && primal_ray_quadratic_norm / (-primal_ray_linear_objective)
    //  <=eps_dual_infeasible
    if (*infeasibility_information.primal_ray_linear_objective < f_t(0.0) &&
        *infeasibility_information.max_primal_ray_infeasibility /
            -(*infeasibility_information.primal_ray_linear_objective) <=
          tolerance.dual_infeasible_tolerance) {
      *termination_status = (i_t)pdlp_termination_status_t::DualInfeasible;
      return;
    }
  }
}

template <typename i_t, typename f_t>
void pdlp_termination_strategy_t<i_t, f_t>::check_termination_criteria()
{
#ifdef PDLP_VERBOSE_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
#endif
  check_termination_criteria_kernel<i_t, f_t>
    <<<1, 1, 0, stream_view_>>>(convergence_information_.view(),
                                infeasibility_information_.view(),
                                termination_status_.data(),
                                settings_.tolerances,
                                settings_.detect_infeasibility,
                                settings_.per_constraint_residual);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t>
pdlp_termination_strategy_t<i_t, f_t>::fill_return_problem_solution(
  i_t number_of_iterations,
  pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
  rmm::device_uvector<f_t>& primal_iterate,
  rmm::device_uvector<f_t>& dual_iterate,
  pdlp_warm_start_data_t<i_t, f_t> warm_start_data,
  pdlp_termination_status_t termination_status,
  bool deep_copy)
{
  typename convergence_information_t<i_t, f_t>::view_t convergence_information_view =
    convergence_information_.view();
  typename infeasibility_information_t<i_t, f_t>::view_t infeasibility_information_view =
    infeasibility_information_.view();

  typename optimization_problem_solution_t<i_t, f_t>::additional_termination_information_t
    term_stats;
  term_stats.number_of_steps_taken           = number_of_iterations;
  term_stats.total_number_of_attempted_steps = current_pdhg_solver.get_total_pdhg_iterations();

  raft::copy(&term_stats.l2_primal_residual,
             (settings_.per_constraint_residual)
               ? convergence_information_view.relative_l_inf_primal_residual
               : convergence_information_view.l2_primal_residual,
             1,
             stream_view_);
  term_stats.l2_relative_primal_residual =
    convergence_information_.get_relative_l2_primal_residual_value();
  raft::copy(&term_stats.l2_dual_residual,
             (settings_.per_constraint_residual)
               ? convergence_information_view.relative_l_inf_dual_residual
               : convergence_information_view.l2_dual_residual,
             1,
             stream_view_);
  term_stats.l2_relative_dual_residual =
    convergence_information_.get_relative_l2_dual_residual_value();
  raft::copy(
    &term_stats.primal_objective, convergence_information_view.primal_objective, 1, stream_view_);
  raft::copy(
    &term_stats.dual_objective, convergence_information_view.dual_objective, 1, stream_view_);
  raft::copy(&term_stats.gap, convergence_information_view.gap, 1, stream_view_);
  term_stats.relative_gap = convergence_information_.get_relative_gap_value();
  raft::copy(&term_stats.max_primal_ray_infeasibility,
             infeasibility_information_view.max_primal_ray_infeasibility,
             1,
             stream_view_);
  raft::copy(&term_stats.primal_ray_linear_objective,
             infeasibility_information_view.primal_ray_linear_objective,
             1,
             stream_view_);
  raft::copy(&term_stats.max_dual_ray_infeasibility,
             infeasibility_information_view.max_dual_ray_infeasibility,
             1,
             stream_view_);
  raft::copy(&term_stats.dual_ray_linear_objective,
             infeasibility_information_view.dual_ray_linear_objective,
             1,
             stream_view_);
  term_stats.solved_by_pdlp = (termination_status != pdlp_termination_status_t::ConcurrentLimit);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));

  if (deep_copy) {
    optimization_problem_solution_t<i_t, f_t> op_solution{
      primal_iterate,
      dual_iterate,
      convergence_information_.get_reduced_cost(),
      problem_ptr->objective_name,
      problem_ptr->var_names,
      problem_ptr->row_names,
      term_stats,
      termination_status,
      handle_ptr_,
      deep_copy};
    return op_solution;
  } else {
    optimization_problem_solution_t<i_t, f_t> op_solution{
      primal_iterate,
      dual_iterate,
      convergence_information_.get_reduced_cost(),
      warm_start_data,
      problem_ptr->objective_name,
      problem_ptr->var_names,
      problem_ptr->row_names,
      term_stats,
      termination_status};
    return op_solution;
  }
}

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t>
pdlp_termination_strategy_t<i_t, f_t>::fill_return_problem_solution(
  i_t number_of_iterations,
  pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
  rmm::device_uvector<f_t>& primal_iterate,
  rmm::device_uvector<f_t>& dual_iterate,
  pdlp_termination_status_t termination_status,
  bool deep_copy)
{
  // Empty warm start data
  return fill_return_problem_solution(number_of_iterations,
                                      current_pdhg_solver,
                                      primal_iterate,
                                      dual_iterate,
                                      pdlp_warm_start_data_t<i_t, f_t>(),
                                      termination_status,
                                      deep_copy);
}

template <typename i_t, typename f_t>
void pdlp_termination_strategy_t<i_t, f_t>::print_termination_criteria(i_t iteration, f_t elapsed)
{
  CUOPT_LOG_INFO("%7d %+.8e %+.8e  %8.2e   %8.2e     %8.2e   %.3fs",
                 iteration,
                 convergence_information_.get_primal_objective().value(stream_view_),
                 convergence_information_.get_dual_objective().value(stream_view_),
                 convergence_information_.get_gap().value(stream_view_),
                 convergence_information_.get_l2_primal_residual().value(stream_view_),
                 convergence_information_.get_l2_dual_residual().value(stream_view_),
                 elapsed);
}

#define INSTANTIATE(F_TYPE)                                                                    \
  template class pdlp_termination_strategy_t<int, F_TYPE>;                                     \
                                                                                               \
  template __global__ void check_termination_criteria_kernel<int, F_TYPE>(                     \
    const typename convergence_information_t<int, F_TYPE>::view_t convergence_information,     \
    const typename infeasibility_information_t<int, F_TYPE>::view_t infeasibility_information, \
    int* termination_status,                                                                   \
    typename pdlp_solver_settings_t<int, F_TYPE>::tolerances_t tolerances,                     \
    bool infeasibility_detection,                                                              \
    bool per_constraint_residual);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
