/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <linear_programming/pdlp_climber_strategy.hpp>
#include <linear_programming/pdlp_constants.hpp>
#include <linear_programming/swap_and_resize_helper.cuh>
#include <linear_programming/termination_strategy/termination_strategy.hpp>

#include <mip/mip_constants.hpp>

#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
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
  const problem_t<i_t, f_t>& scaled_op_problem,
  cusparse_view_t<i_t, f_t>& cusparse_view,
  const cusparse_view_t<i_t, f_t>& scaled_cusparse_view,
  const i_t primal_size,
  const i_t dual_size,
  const pdlp_initial_scaling_strategy_t<i_t, f_t>& scaling_strategy,
  const pdlp_solver_settings_t<i_t, f_t>& settings,
  const std::vector<pdlp_climber_strategy_t>& climber_strategies)
  : handle_ptr_(handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    problem_ptr(&op_problem),
    convergence_information_{handle_ptr_,
                             op_problem,
                             cusparse_view,
                             primal_size,
                             dual_size,
                             climber_strategies,
                             settings.hyper_params},
    infeasibility_information_{handle_ptr_,
                               op_problem,
                               scaled_op_problem,
                               cusparse_view,
                               scaled_cusparse_view,
                               primal_size,
                               dual_size,
                               scaling_strategy,
                               settings.detect_infeasibility,
                               climber_strategies,
                               settings.hyper_params},
    termination_status_(climber_strategies.size()),
    settings_(settings),
    gpu_batch_additional_termination_information_{climber_strategies.size()},
    original_index_(climber_strategies.size()),
    climber_strategies_(climber_strategies)
{
  std::fill(termination_status_.begin(),
            termination_status_.end(),
            (i_t)pdlp_termination_status_t::NoTermination);
}

template <typename i_t, typename f_t>
void pdlp_termination_strategy_t<i_t, f_t>::swap_context(
  const thrust::universal_host_pinned_vector<swap_pair_t<i_t>>& swap_pairs)
{
  if (swap_pairs.empty()) { return; }

  convergence_information_.swap_context(swap_pairs);
  cuopt_assert(!settings_.detect_infeasibility,
               "Infeasibility detection must be disabled to swap the termination status");
  // infeasibility_information_.swap_context(swap_pairs);
  cuopt_assert(!termination_status_.empty(), "Termination status must not be empty");
  for (const auto& pair : swap_pairs) {
    host_vector_swap(termination_status_, pair.left, pair.right);
  }
}

template <typename i_t, typename f_t>
void pdlp_termination_strategy_t<i_t, f_t>::resize_context(i_t new_size)
{
  convergence_information_.resize_context(new_size);
  cuopt_assert(!settings_.detect_infeasibility,
               "Infeasibility detection must be disabled to resize the termination status");
  cuopt_assert(!termination_status_.empty(), "Termination status must not be empty");
  termination_status_.resize(new_size);
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
pdlp_termination_status_t pdlp_termination_strategy_t<i_t, f_t>::get_termination_status(
  i_t id) const
{
  return (pdlp_termination_status_t)termination_status_[id];
}

template <typename i_t, typename f_t>
std::vector<pdlp_termination_status_t>
pdlp_termination_strategy_t<i_t, f_t>::get_terminations_status()
{
  std::vector<pdlp_termination_status_t> out(climber_strategies_.size());
  cuopt_assert(out.size() == termination_status_.size(), "Both should have equal size");
  std::transform(termination_status_.begin(), termination_status_.end(), out.begin(), [](i_t in) {
    return (pdlp_termination_status_t)in;
  });
  return out;
}

// TODO later batch mode: will be useful once I bring back MCPDLP
template <typename i_t, typename f_t>
bool pdlp_termination_strategy_t<i_t, f_t>::has_optimal_status() const
{
  return std::any_of(termination_status_.begin(), termination_status_.end(), [](i_t status) {
    return status == (i_t)pdlp_termination_status_t::Optimal;
  });
}

template <typename i_t, typename f_t>
i_t pdlp_termination_strategy_t<i_t, f_t>::nb_optimal_solutions() const
{
  return std::count(termination_status_.begin(),
                    termination_status_.end(),
                    (i_t)pdlp_termination_status_t::Optimal);
}

template <typename i_t, typename f_t>
i_t pdlp_termination_strategy_t<i_t, f_t>::get_optimal_solution_id() const
{
  cuopt_assert(nb_optimal_solutions() == 1, "nb_optimal_solutions() must be 1");
  return std::distance(termination_status_.begin(),
                       std::find(termination_status_.begin(),
                                 termination_status_.end(),
                                 (i_t)pdlp_termination_status_t::Optimal));
}

template <typename i_t, typename f_t>
void pdlp_termination_strategy_t<i_t, f_t>::evaluate_termination_criteria(
  pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
  rmm::device_uvector<f_t>& primal_iterate,
  rmm::device_uvector<f_t>& dual_iterate,
  const rmm::device_uvector<f_t>& dual_slack,
  rmm::device_uvector<f_t>& delta_primal_iterate,
  rmm::device_uvector<f_t>& delta_dual_iterate,
  i_t total_pdlp_iterations,
  const rmm::device_uvector<f_t>& combined_bounds,
  const rmm::device_uvector<f_t>& objective_coefficients)
{
  raft::common::nvtx::range fun_scope("Evaluate termination criteria");

  convergence_information_.compute_convergence_information(current_pdhg_solver,
                                                           primal_iterate,
                                                           dual_iterate,
                                                           dual_slack,
                                                           combined_bounds,
                                                           objective_coefficients,
                                                           settings_);
  if (settings_.detect_infeasibility) {
    // TODO PDLP infeasible: looks like he is not checking as often as we do
    if (settings_.hyper_params.use_reflected_primal_dual) {
      if (total_pdlp_iterations != 0 &&
          total_pdlp_iterations % settings_.hyper_params.major_iteration == 0 &&
          total_pdlp_iterations < 3 * settings_.hyper_params.major_iteration) {
        infeasibility_information_.compute_infeasibility_information(
          current_pdhg_solver, delta_primal_iterate, delta_dual_iterate);
      }
    } else {
      infeasibility_information_.compute_infeasibility_information(
        current_pdhg_solver, primal_iterate, dual_iterate);
    }
  }
  check_termination_criteria();

  // Sync to make sure the termination status is updated
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));
}

template <typename i_t, typename f_t>
const convergence_information_t<i_t, f_t>&
pdlp_termination_strategy_t<i_t, f_t>::get_convergence_information() const
{
  return convergence_information_;
}

template <typename i_t, typename f_t>
const infeasibility_information_t<i_t, f_t>&
pdlp_termination_strategy_t<i_t, f_t>::get_infeasibility_information() const
{
  return infeasibility_information_;
}

template <typename i_t, typename f_t>
__global__ void check_termination_criteria_kernel(
  const typename convergence_information_t<i_t, f_t>::view_t convergence_information,
  const typename infeasibility_information_t<i_t, f_t>::view_t infeasibility_information,
  raft::device_span<i_t> termination_status,
  typename pdlp_solver_settings_t<i_t, f_t>::tolerances_t tolerance,
  bool infeasibility_detection,
  bool per_constraint_residual,
  i_t batch_size)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= batch_size) { return; }

#ifdef PDLP_VERBOSE_MODE
  if (idx == 0) {
    printf(
      "Gap : %lf <= %lf [%d] (tolerance.absolute_gap_tolerance %lf + "
      "tolerance.relative_gap_tolerance %lf * convergence_information.abs_objective %lf)\n",
      convergence_information.gap[idx],
      tolerance.absolute_gap_tolerance +
        tolerance.relative_gap_tolerance * convergence_information.abs_objective[idx],
      convergence_information.gap[idx] <=
        tolerance.absolute_gap_tolerance +
          tolerance.relative_gap_tolerance * convergence_information.abs_objective[idx],
      tolerance.absolute_gap_tolerance,
      tolerance.relative_gap_tolerance,
      convergence_information.abs_objective[idx]);
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
      // TODO later batch mode: per problem rhs
      printf(
        "Primal residual  %lf <= %lf [%d] (tolerance.absolute_primal_tolerance %lf + "
        "tolerance.relative_primal_tolerance %lf * "
        "convergence_information.l2_norm_primal_right_hand_side %lf)\n",
        convergence_information.l2_primal_residual[idx],
        tolerance.absolute_primal_tolerance +
          tolerance.relative_primal_tolerance *
            *convergence_information.l2_norm_primal_right_hand_side,
        convergence_information.l2_primal_residual[idx] <=
          tolerance.absolute_primal_tolerance +
            tolerance.relative_primal_tolerance *
              *convergence_information.l2_norm_primal_right_hand_side,
        tolerance.absolute_primal_tolerance,
        tolerance.relative_primal_tolerance,
        *convergence_information.l2_norm_primal_right_hand_side);
      printf(
        "Dual residual  %lf <= %lf [%d] (tolerance.absolute_dual_tolerance %lf + "
        "tolerance.relative_dual_tolerance %lf * "
        "convergence_information.l2_norm_primal_linear_objective %lf)\n",
        convergence_information.l2_dual_residual[idx],
        tolerance.absolute_dual_tolerance +
          tolerance.relative_dual_tolerance *
            *convergence_information.l2_norm_primal_linear_objective,
        convergence_information.l2_dual_residual[idx] <=
          tolerance.absolute_dual_tolerance +
            tolerance.relative_dual_tolerance *
              *convergence_information.l2_norm_primal_linear_objective,
        tolerance.absolute_dual_tolerance,
        tolerance.relative_dual_tolerance,
        *convergence_information.l2_norm_primal_linear_objective);
    }
    if (infeasibility_detection) {
      printf(
        "Primal infeasible ? [%d] : infeasibility_information.dual_ray_linear_objective (should "
        "positive) %lf / "
        "infeasibility_information.max_dual_ray_infeasibility %lf = %lf <= "
        "tolerance.primal_infeasible_tolerance %lf\n",
        infeasibility_information.dual_ray_linear_objective[idx] > f_t(0.0) &&
          infeasibility_information.max_dual_ray_infeasibility[idx] /
              infeasibility_information.dual_ray_linear_objective[idx] <=
            tolerance.primal_infeasible_tolerance,
        infeasibility_information.dual_ray_linear_objective[idx],
        infeasibility_information.max_dual_ray_infeasibility[idx],
        infeasibility_information.max_dual_ray_infeasibility[idx] /
          infeasibility_information.dual_ray_linear_objective[idx],
        tolerance.primal_infeasible_tolerance);
    }
  }
#endif

  // test if gap optimal
  const bool optimal_gap =
    convergence_information.gap[idx] <=
    tolerance.absolute_gap_tolerance +
      tolerance.relative_gap_tolerance * convergence_information.abs_objective[idx];

  // test if respect constraints
  if (per_constraint_residual) {
    // In residual we store l_inf(residual_i - rel * b/c_i)
    const bool primal_feasible = *convergence_information.relative_l_inf_primal_residual <=
                                 tolerance.absolute_primal_tolerance;
    // First check for optimality
    if (*convergence_information.relative_l_inf_dual_residual <=
          tolerance.absolute_dual_tolerance &&
        primal_feasible && optimal_gap) {
      termination_status[idx] = (i_t)pdlp_termination_status_t::Optimal;
      return;
    } else if (primal_feasible)  // If not optimal maybe be at least primal feasible
    {
      termination_status[idx] = (i_t)pdlp_termination_status_t::PrimalFeasible;
      return;
    }
  } else {
    const bool primal_feasible = convergence_information.l2_primal_residual[idx] <=
                                 tolerance.absolute_primal_tolerance +
                                   tolerance.relative_primal_tolerance *
                                     *convergence_information.l2_norm_primal_right_hand_side;
    if (convergence_information.l2_dual_residual[idx] <=
          tolerance.absolute_dual_tolerance +
            tolerance.relative_dual_tolerance *
              *convergence_information.l2_norm_primal_linear_objective &&
        primal_feasible && optimal_gap) {
      termination_status[idx] = (i_t)pdlp_termination_status_t::Optimal;
      return;
    } else if (primal_feasible)  // If not optimal maybe be at least primal feasible
    {
      termination_status[idx] = (i_t)pdlp_termination_status_t::PrimalFeasible;
      return;
    } else {
      termination_status[idx] = (i_t)pdlp_termination_status_t::NoTermination;
    }
  }

  if (infeasibility_detection) {
    // test for primal infeasibility
    if (infeasibility_information.dual_ray_linear_objective[idx] > f_t(0.0) &&
        infeasibility_information.max_dual_ray_infeasibility[idx] /
            infeasibility_information.dual_ray_linear_objective[idx] <=
          tolerance.primal_infeasible_tolerance) {
      termination_status[idx] = (i_t)pdlp_termination_status_t::PrimalInfeasible;
      return;
    }

    // test for dual infeasibility
    //  for QP add && primal_ray_quadratic_norm / (-primal_ray_linear_objective)
    //  <=eps_dual_infeasible
    if (infeasibility_information.primal_ray_linear_objective[idx] < f_t(0.0) &&
        infeasibility_information.max_primal_ray_infeasibility[idx] /
            -(infeasibility_information.primal_ray_linear_objective[idx]) <=
          tolerance.dual_infeasible_tolerance) {
      termination_status[idx] = (i_t)pdlp_termination_status_t::DualInfeasible;
      return;
    }
  }
}

template <typename i_t, typename f_t>
bool pdlp_termination_strategy_t<i_t, f_t>::all_optimal_status() const
{
  return std::all_of(
    termination_status_.cbegin(), termination_status_.cend(), [](i_t termination_status) {
      return termination_status == (i_t)pdlp_termination_status_t::Optimal;
    });
}

template <typename i_t, typename f_t>
__host__ __device__ bool pdlp_termination_strategy_t<i_t, f_t>::is_done(
  pdlp_termination_status_t termination_status)
{
  return termination_status == pdlp_termination_status_t::Optimal ||
         termination_status == pdlp_termination_status_t::PrimalInfeasible ||
         termination_status == pdlp_termination_status_t::DualInfeasible;
}

template <typename i_t, typename f_t>
bool pdlp_termination_strategy_t<i_t, f_t>::all_done() const
{
  return std::all_of(
    termination_status_.cbegin(), termination_status_.cend(), [](i_t termination_status) {
      return is_done((pdlp_termination_status_t)termination_status);
    });
}

template <typename i_t, typename f_t>
void pdlp_termination_strategy_t<i_t, f_t>::check_termination_criteria()
{
#ifdef PDLP_VERBOSE_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
#endif
  const auto [grid_size, block_size] = kernel_config_from_batch_size(climber_strategies_.size());
  check_termination_criteria_kernel<i_t, f_t>
    <<<grid_size, block_size, 0, stream_view_>>>(convergence_information_.view(),
                                                 infeasibility_information_.view(),
                                                 make_span(termination_status_),
                                                 settings_.tolerances,
                                                 settings_.detect_infeasibility,
                                                 settings_.per_constraint_residual,
                                                 climber_strategies_.size());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename i_t, typename f_t>
__global__ void fill_gpu_terms_stats_kernel(
  raft::device_span<i_t> termination_status,
  raft::device_span<i_t> original_indices,
  typename pdlp_termination_strategy_t<i_t,
                                       f_t>::gpu_batch_additional_termination_information_t::view_t
    additional_termination_information,
  typename convergence_information_t<i_t, f_t>::view_t convergence_information_view,
  i_t number_of_steps_taken)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= termination_status.size()) { return; }

  // TODO later batch mode: add infeasibility information here
  // TODO later batch mode: handle per climber rhs and objective

  // Will be removed store its data in the struct
  if (pdlp_termination_strategy_t<i_t, f_t>::is_done(
        (pdlp_termination_status_t)termination_status[idx])) {
    const i_t original_index = original_indices[idx];
    additional_termination_information.number_of_steps_taken[original_index] =
      number_of_steps_taken;
    additional_termination_information.total_number_of_attempted_steps[original_index] =
      number_of_steps_taken;
    additional_termination_information.l2_primal_residual[original_index] =
      convergence_information_view.l2_primal_residual[idx];
    additional_termination_information.l2_relative_primal_residual[original_index] =
      convergence_information_view.l2_primal_residual[idx] /
      (f_t(1.0) + *convergence_information_view.l2_norm_primal_right_hand_side);
    additional_termination_information.l2_dual_residual[original_index] =
      convergence_information_view.l2_dual_residual[idx];
    additional_termination_information.l2_relative_dual_residual[original_index] =
      convergence_information_view.l2_dual_residual[idx] /
      (f_t(1.0) + *convergence_information_view.l2_norm_primal_linear_objective);
    additional_termination_information.primal_objective[original_index] =
      convergence_information_view.primal_objective[idx];
    additional_termination_information.dual_objective[original_index] =
      convergence_information_view.dual_objective[idx];
    additional_termination_information.gap[original_index] = convergence_information_view.gap[idx];
    additional_termination_information.relative_gap[original_index] =
      convergence_information_view.gap[idx] /
      (f_t(1.0) + convergence_information_view.abs_objective[idx]);
  }
}

template <typename i_t, typename f_t>
void pdlp_termination_strategy_t<i_t, f_t>::fill_gpu_terms_stats(i_t number_of_iterations)
{
  typename convergence_information_t<i_t, f_t>::view_t convergence_information_view =
    convergence_information_.view();

  // Update original index pinned view so that we can read it safely from the kernel
  for (size_t i = 0; i < climber_strategies_.size(); ++i) {
    original_index_[i] = climber_strategies_[i].original_index;
  }

  const auto [grid_size, block_size] = kernel_config_from_batch_size(climber_strategies_.size());
  fill_gpu_terms_stats_kernel<i_t, f_t><<<grid_size, block_size, 0, stream_view_>>>(
    make_span(termination_status_),
    make_span(original_index_),
    gpu_batch_additional_termination_information_.view(),
    convergence_information_view,
    number_of_iterations);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));
}

template <typename i_t, typename f_t>
void pdlp_termination_strategy_t<i_t, f_t>::convert_gpu_terms_stats_to_host(
  std::vector<
    typename optimization_problem_solution_t<i_t, f_t>::additional_termination_information_t>&
    additional_termination_informations)
{
  for (size_t i = 0; i < additional_termination_informations.size(); ++i) {
    additional_termination_informations[i].number_of_steps_taken =
      gpu_batch_additional_termination_information_.number_of_steps_taken[i];
    additional_termination_informations[i].total_number_of_attempted_steps =
      gpu_batch_additional_termination_information_.total_number_of_attempted_steps[i];
    additional_termination_informations[i].l2_primal_residual =
      gpu_batch_additional_termination_information_.l2_primal_residual[i];
    additional_termination_informations[i].l2_relative_primal_residual =
      gpu_batch_additional_termination_information_.l2_relative_primal_residual[i];
    additional_termination_informations[i].l2_dual_residual =
      gpu_batch_additional_termination_information_.l2_dual_residual[i];
    additional_termination_informations[i].l2_relative_dual_residual =
      gpu_batch_additional_termination_information_.l2_relative_dual_residual[i];
    additional_termination_informations[i].primal_objective =
      gpu_batch_additional_termination_information_.primal_objective[i];
    additional_termination_informations[i].dual_objective =
      gpu_batch_additional_termination_information_.dual_objective[i];
    additional_termination_informations[i].gap =
      gpu_batch_additional_termination_information_.gap[i];
    additional_termination_informations[i].relative_gap =
      gpu_batch_additional_termination_information_.relative_gap[i];
  }
}
template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t>
pdlp_termination_strategy_t<i_t, f_t>::fill_return_problem_solution(
  i_t number_of_iterations,
  pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
  rmm::device_uvector<f_t>& primal_iterate,
  rmm::device_uvector<f_t>& dual_iterate,
  pdlp_warm_start_data_t<i_t, f_t>&& warm_start_data,
  std::vector<pdlp_termination_status_t>&& termination_status,
  bool deep_copy)
{
  cuopt_assert(
    primal_iterate.size() == current_pdhg_solver.get_primal_size() * termination_status.size(),
    "Primal iterate size mismatch");
  cuopt_assert(
    dual_iterate.size() == current_pdhg_solver.get_dual_size() * termination_status.size(),
    "Dual iterate size mismatch");

  typename convergence_information_t<i_t, f_t>::view_t convergence_information_view =
    convergence_information_.view();
  typename infeasibility_information_t<i_t, f_t>::view_t infeasibility_information_view =
    infeasibility_information_.view();

  std::vector<
    typename optimization_problem_solution_t<i_t, f_t>::additional_termination_information_t>
    term_stats_vector(climber_strategies_.size());
  for (size_t i = 0; i < climber_strategies_.size(); ++i) {
    // TODO later batch mode: handle per climber number_of_iterations
    term_stats_vector[i].number_of_steps_taken = number_of_iterations;
    term_stats_vector[i].total_number_of_attempted_steps =
      current_pdhg_solver.get_total_pdhg_iterations();

    raft::copy(&term_stats_vector[i].l2_primal_residual,
               (settings_.per_constraint_residual)
                 ? convergence_information_view
                     .relative_l_inf_primal_residual  // TODO later batch mode: handle per climber
                                                      // overall residual
                 : convergence_information_view.l2_primal_residual.data() + i,
               1,
               stream_view_);

    term_stats_vector[i].l2_relative_primal_residual =
      convergence_information_.get_relative_l2_primal_residual_value(i);

    raft::copy(&term_stats_vector[i].l2_dual_residual,
               (settings_.per_constraint_residual)
                 ? convergence_information_view.relative_l_inf_dual_residual
                 : convergence_information_view.l2_dual_residual.data() + i,
               1,
               stream_view_);

    term_stats_vector[i].l2_relative_dual_residual =
      convergence_information_.get_relative_l2_dual_residual_value(i);

    raft::copy(&term_stats_vector[i].primal_objective,
               convergence_information_view.primal_objective.data() + i,
               1,
               stream_view_);
    raft::copy(&term_stats_vector[i].dual_objective,
               convergence_information_view.dual_objective.data() + i,
               1,
               stream_view_);
    raft::copy(
      &term_stats_vector[i].gap, convergence_information_view.gap.data() + i, 1, stream_view_);
    term_stats_vector[i].relative_gap = convergence_information_.get_relative_gap_value(i);
    raft::copy(&term_stats_vector[i].max_primal_ray_infeasibility,
               &infeasibility_information_view.max_primal_ray_infeasibility[i],
               1,
               stream_view_);
    raft::copy(&term_stats_vector[i].primal_ray_linear_objective,
               &infeasibility_information_view.primal_ray_linear_objective[i],
               1,
               stream_view_);
    raft::copy(&term_stats_vector[i].max_dual_ray_infeasibility,
               &infeasibility_information_view.max_dual_ray_infeasibility[i],
               1,
               stream_view_);
    raft::copy(&term_stats_vector[i].dual_ray_linear_objective,
               &infeasibility_information_view.dual_ray_linear_objective[i],
               1,
               stream_view_);
    term_stats_vector[i].solved_by_pdlp =
      (termination_status[i] != pdlp_termination_status_t::ConcurrentLimit);
  }

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));

  if (deep_copy) {
    cuopt_assert(
      climber_strategies_.size() == 1,
      "Deep copy is linked to first primal feasible which is not supported in batch PDLP");
    optimization_problem_solution_t<i_t, f_t> op_solution{
      primal_iterate,
      dual_iterate,
      convergence_information_.get_reduced_cost(),
      problem_ptr->objective_name,
      problem_ptr->var_names,
      problem_ptr->row_names,
      term_stats_vector[0],
      termination_status[0],
      handle_ptr_,
      deep_copy};
    return op_solution;
  } else {
    optimization_problem_solution_t<i_t, f_t> op_solution{
      primal_iterate,
      dual_iterate,
      convergence_information_.get_reduced_cost(),
      std::move(warm_start_data),
      problem_ptr->objective_name,
      problem_ptr->var_names,
      problem_ptr->row_names,
      std::move(term_stats_vector),
      std::move(termination_status)};
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
  std::vector<pdlp_termination_status_t>&& termination_status,
  bool deep_copy)
{
  // Empty warm start data
  return fill_return_problem_solution(number_of_iterations,
                                      current_pdhg_solver,
                                      primal_iterate,
                                      dual_iterate,
                                      pdlp_warm_start_data_t<i_t, f_t>(),
                                      std::move(termination_status),
                                      deep_copy);
}

template <typename i_t, typename f_t>
void pdlp_termination_strategy_t<i_t, f_t>::print_termination_criteria(i_t iteration,
                                                                       f_t elapsed,
                                                                       i_t best_id) const
{
  // TODO less critical batch mode: handle this
  CUOPT_LOG_INFO("%7d %+.8e %+.8e  %8.2e   %8.2e     %8.2e   %.3fs",
                 iteration,
                 convergence_information_.get_primal_objective().element(best_id, stream_view_),
                 convergence_information_.get_dual_objective().element(best_id, stream_view_),
                 convergence_information_.get_gap().element(best_id, stream_view_),
                 convergence_information_.get_l2_primal_residual().element(best_id, stream_view_),
                 convergence_information_.get_l2_dual_residual().element(best_id, stream_view_),
                 elapsed);
}

#define INSTANTIATE(F_TYPE)                                                                    \
  template class pdlp_termination_strategy_t<int, F_TYPE>;                                     \
                                                                                               \
  template __global__ void check_termination_criteria_kernel<int, F_TYPE>(                     \
    const typename convergence_information_t<int, F_TYPE>::view_t convergence_information,     \
    const typename infeasibility_information_t<int, F_TYPE>::view_t infeasibility_information, \
    raft::device_span<int> termination_status,                                                 \
    typename pdlp_solver_settings_t<int, F_TYPE>::tolerances_t tolerances,                     \
    bool infeasibility_detection,                                                              \
    bool per_constraint_residual,                                                              \
    int batch_size);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
