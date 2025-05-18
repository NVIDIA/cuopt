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

#include <cuopt/error.hpp>

#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <linear_programming/pdlp_constants.hpp>
#include <linear_programming/restart_strategy/pdlp_restart_strategy.cuh>
#include <linear_programming/utils.cuh>
#include <mip/mip_constants.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/common/nvtx.hpp>
#include <raft/core/device_span.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/ternary_op.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include <cub/cub.cuh>

#include <cooperative_groups.h>

#include <cmath>

namespace cg = cooperative_groups;

namespace cuopt::linear_programming::detail {

void set_restart_hyper_parameters()
{
  RAFT_CUDA_TRY(cudaMemcpyToSymbol(pdlp_hyper_params::default_primal_weight_update_smoothing,
                                   &pdlp_hyper_params::host_default_primal_weight_update_smoothing,
                                   sizeof(double)));
  RAFT_CUDA_TRY(
    cudaMemcpyToSymbol(pdlp_hyper_params::default_sufficient_reduction_for_restart,
                       &pdlp_hyper_params::host_default_sufficient_reduction_for_restart,
                       sizeof(double)));
  RAFT_CUDA_TRY(cudaMemcpyToSymbol(pdlp_hyper_params::default_necessary_reduction_for_restart,
                                   &pdlp_hyper_params::host_default_necessary_reduction_for_restart,
                                   sizeof(double)));
  RAFT_CUDA_TRY(cudaMemcpyToSymbol(pdlp_hyper_params::primal_distance_smoothing,
                                   &pdlp_hyper_params::host_primal_distance_smoothing,
                                   sizeof(double)));
  RAFT_CUDA_TRY(cudaMemcpyToSymbol(pdlp_hyper_params::dual_distance_smoothing,
                                   &pdlp_hyper_params::host_dual_distance_smoothing,
                                   sizeof(double)));
}

template <typename i_t, typename f_t, int BLOCK_SIZE>
__global__ void solve_bound_constrained_trust_region_kernel(
  typename pdlp_restart_strategy_t<i_t, f_t>::view_t restart_strategy_view,
  typename problem_t<i_t, f_t>::view_t op_problem_view,
  i_t* testing_range_low,
  i_t* testing_range_high,
  f_t* test_radius_squared,
  f_t* low_radius_squared,
  f_t* high_radius_squared,
  const f_t* target_radius);

template <typename i_t, typename f_t>
pdlp_restart_strategy_t<i_t, f_t>::pdlp_restart_strategy_t(
  raft::handle_t const* handle_ptr,
  problem_t<i_t, f_t>& op_problem,
  const cusparse_view_t<i_t, f_t>& cusparse_view,
  const i_t primal_size,
  const i_t dual_size)
  : handle_ptr_(handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    weighted_average_solution_{handle_ptr_, primal_size, dual_size},
    primal_size_h_(primal_size),
    dual_size_h_(dual_size),
    problem_ptr(&op_problem),
    primal_size_{primal_size, stream_view_},
    dual_size_{dual_size, stream_view_},
    primal_norm_weight_{stream_view_},
    weights_{(is_KKT_restart<i_t, f_t>()) ? 0 : static_cast<size_t>(primal_size_h_ + dual_size_h_),
             stream_view_},
    dual_norm_weight_{stream_view_},
    restart_triggered_{0, stream_view_},
    candidate_is_avg_{0, stream_view_},
    avg_duality_gap_{handle_ptr_, primal_size, dual_size},
    current_duality_gap_{handle_ptr_, primal_size, dual_size},
    last_restart_duality_gap_{handle_ptr_, primal_size, dual_size},
    // If KKT restart, call the empty cusparse_view constructor
    avg_duality_gap_cusparse_view_{
      (is_KKT_restart<i_t, f_t>())
        ? cusparse_view_t<i_t, f_t>(handle_ptr_, cusparse_view.A_, cusparse_view.A_indices_)
        : cusparse_view_t<i_t, f_t>(handle_ptr_,
                                    op_problem,
                                    cusparse_view,
                                    avg_duality_gap_.primal_solution_.data(),
                                    avg_duality_gap_.dual_solution_.data(),
                                    avg_duality_gap_.primal_gradient_.data(),
                                    avg_duality_gap_.dual_gradient_.data())},
    current_duality_gap_cusparse_view_{
      (is_KKT_restart<i_t, f_t>())
        ? cusparse_view_t<i_t, f_t>(handle_ptr_, cusparse_view.A_, cusparse_view.A_indices_)
        : cusparse_view_t<i_t, f_t>(handle_ptr_,
                                    op_problem,
                                    cusparse_view,
                                    current_duality_gap_.primal_solution_.data(),
                                    current_duality_gap_.dual_solution_.data(),
                                    current_duality_gap_.primal_gradient_.data(),
                                    current_duality_gap_.dual_gradient_.data())},
    last_restart_duality_gap_cusparse_view_{
      (is_KKT_restart<i_t, f_t>())
        ? cusparse_view_t<i_t, f_t>(handle_ptr_, cusparse_view.A_, cusparse_view.A_indices_)
        : cusparse_view_t<i_t, f_t>(handle_ptr_,
                                    op_problem,
                                    cusparse_view,
                                    last_restart_duality_gap_.primal_solution_.data(),
                                    last_restart_duality_gap_.dual_solution_.data(),
                                    last_restart_duality_gap_.primal_gradient_.data(),
                                    last_restart_duality_gap_.dual_gradient_.data())},
    gap_reduction_ratio_last_trial_{stream_view_},
    last_restart_length_{0},
    // If KKT restart, don't need to init all of those
    center_point_{
      (is_KKT_restart<i_t, f_t>()) ? 0 : static_cast<size_t>(primal_size_h_ + dual_size_h_),
      stream_view_},
    objective_vector_{
      (is_KKT_restart<i_t, f_t>()) ? 0 : static_cast<size_t>(primal_size_h_ + dual_size_h_),
      stream_view_},
    unsorted_direction_full_{
      (is_KKT_restart<i_t, f_t>()) ? 0 : static_cast<size_t>(primal_size_h_ + dual_size_h_),
      stream_view_},
    direction_full_{
      (is_KKT_restart<i_t, f_t>()) ? 0 : static_cast<size_t>(primal_size_h_ + dual_size_h_),
      stream_view_},
    threshold_{
      (is_KKT_restart<i_t, f_t>()) ? 0 : static_cast<size_t>(primal_size_h_ + dual_size_h_),
      stream_view_},
    lower_bound_{
      (is_KKT_restart<i_t, f_t>()) ? 0 : static_cast<size_t>(primal_size_h_ + dual_size_h_),
      stream_view_},
    upper_bound_{
      (is_KKT_restart<i_t, f_t>()) ? 0 : static_cast<size_t>(primal_size_h_ + dual_size_h_),
      stream_view_},
    test_point_{
      (is_KKT_restart<i_t, f_t>()) ? 0 : static_cast<size_t>(primal_size_h_ + dual_size_h_),
      stream_view_},
    transformed_constraint_lower_bounds_{
      (is_KKT_restart<i_t, f_t>()) ? 0 : static_cast<size_t>(dual_size_h_), stream_view_},
    transformed_constraint_upper_bounds_{
      (is_KKT_restart<i_t, f_t>()) ? 0 : static_cast<size_t>(dual_size_h_), stream_view_},
    shared_live_kernel_accumulator_{0, stream_view_},
    target_threshold_{stream_view_},
    low_radius_squared_{stream_view_},
    high_radius_squared_{stream_view_},
    test_threshold_{stream_view_},
    test_radius_squared_{stream_view_},
    testing_range_low_{stream_view_},
    testing_range_high_{stream_view_},
    reusable_device_scalar_value_1_{1.0, stream_view_},
    reusable_device_scalar_value_0_{0.0, stream_view_},
    reusable_device_scalar_value_0_i_t_{0, stream_view_},
    reusable_device_scalar_value_neg_1_{f_t(-1.0), stream_view_},
    tmp_kkt_score_{stream_view_},
    reusable_device_scalar_1_{stream_view_},
    reusable_device_scalar_2_{stream_view_},
    reusable_device_scalar_3_{stream_view_}
{
  raft::common::nvtx::range fun_scope("Initializing restart strategy");

  // Init the vectors
  RAFT_CUDA_TRY(cudaMemsetAsync(last_restart_duality_gap_.primal_solution_.data(),
                                0.0,
                                sizeof(f_t) * primal_size_h_,
                                stream_view_));
  RAFT_CUDA_TRY(cudaMemsetAsync(last_restart_duality_gap_.dual_solution_.data(),
                                0.0,
                                sizeof(f_t) * dual_size_h_,
                                stream_view_));

  // Trigger the costly (costly for ms instances) GetDeviceProperty only if need trust region
  // restart
  if (pdlp_hyper_params::restart_strategy ==
      static_cast<int>(restart_strategy_t::TRUST_REGION_RESTART)) {
    raft::linalg::binaryOp(transformed_constraint_lower_bounds_.data(),
                           problem_ptr->constraint_lower_bounds.data(),
                           problem_ptr->constraint_upper_bounds.data(),
                           dual_size_h_,
                           transform_constraint_lower_bounds<f_t>(),
                           stream_view_);
    raft::linalg::binaryOp(transformed_constraint_upper_bounds_.data(),
                           problem_ptr->constraint_lower_bounds.data(),
                           problem_ptr->constraint_upper_bounds.data(),
                           dual_size_h_,
                           transform_constraint_upper_bounds<f_t>(),
                           stream_view_);

    // Check that device support CooperativeLaunch
    int dev                = 0;
    int supportsCoopLaunch = 0;
    RAFT_CUDA_TRY(cudaGetDevice(&dev));
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev));
    EXE_CUOPT_EXPECTS(supportsCoopLaunch == 1, "Current device does not support CooperativeLaunch");
    /// Compute max number of blocks for live kernel
    int numBlocksPerSm       = 0;
    constexpr int numThreads = 128;
    cudaDeviceProp deviceProp;
    RAFT_CUDA_TRY(cudaGetDeviceProperties(&deviceProp, dev));
    RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm,
      solve_bound_constrained_trust_region_kernel<i_t, f_t, numThreads>,
      numThreads,
      0));
    const int nb_block_to_launch =
      std::min(deviceProp.multiProcessorCount * numBlocksPerSm,
               (primal_size_h_ + dual_size_h_ + numThreads - 1) / numThreads);
    shared_live_kernel_accumulator_.resize(nb_block_to_launch, handle_ptr->get_stream());
  }
}

template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::add_current_solution_to_average_solution(
  const f_t* primal_solution,
  const f_t* dual_solution,
  const rmm::device_scalar<f_t>& weight,
  i_t total_pdlp_iterations)
{
  weighted_average_solution_.add_current_solution_to_weighted_average_solution(
    primal_solution, dual_solution, weight, total_pdlp_iterations);
}
template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::get_average_solutions(rmm::device_uvector<f_t>& avg_primal,
                                                              rmm::device_uvector<f_t>& avg_dual)
{
  weighted_average_solution_.compute_averages(avg_primal, avg_dual);
}

template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::run_trust_region_restart(
  pdhg_solver_t<i_t, f_t>& pdhg_solver,
  rmm::device_uvector<f_t>& primal_solution_avg,
  rmm::device_uvector<f_t>& dual_solution_avg,
  const i_t total_number_of_iterations,
  rmm::device_scalar<f_t>& primal_step_size,
  rmm::device_scalar<f_t>& dual_step_size,
  rmm::device_scalar<f_t>& primal_weight,
  const rmm::device_scalar<f_t>& step_size)
{
  raft::common::nvtx::range fun_scope("run trust region restart");
#ifdef PDLP_VERBOSE_MODE
  std::cout << "Trust region restart:" << std::endl;
#endif

  if (weighted_average_solution_.get_iterations_since_last_restart() == 0) {
#ifdef PDLP_VERBOSE_MODE
    std::cout << "    No internal iteration, can't restart yet, returning:" << std::endl;
#endif
    return;
  }

  // make the step sizes into the norm weights that will be used throughout the computations
  raft::linalg::eltwiseDivideCheckZero(primal_norm_weight_.data(),
                                       reusable_device_scalar_value_1_.data(),
                                       primal_step_size.data(),
                                       1,
                                       stream_view_);
  raft::linalg::eltwiseDivideCheckZero(dual_norm_weight_.data(),
                                       reusable_device_scalar_value_1_.data(),
                                       dual_step_size.data(),
                                       1,
                                       stream_view_);

  i_t restart = should_do_artificial_restart(total_number_of_iterations);

  compute_localized_duality_gaps(pdhg_solver.get_saddle_point_state(),
                                 primal_solution_avg,
                                 dual_solution_avg,
                                 primal_weight,
                                 pdhg_solver.get_primal_tmp_resource(),
                                 pdhg_solver.get_dual_tmp_resource());

  i_t restart_to_average_h = pick_restart_candidate();

  // Might retrigger restart from normalized duality gap
  if (!restart) {
    should_do_adaptive_restart_normalized_duality_gap(*candidate_duality_gap_,
                                                      pdhg_solver.get_primal_tmp_resource(),
                                                      pdhg_solver.get_dual_tmp_resource(),
                                                      primal_weight,
                                                      restart);
  }

  if (restart) {
#ifdef PDLP_VERBOSE_MODE
    std::cout << "    Doing a trust Region Restart" << std::endl;
#endif
    if (restart_to_average_h && !pdlp_hyper_params::never_restart_to_average) {
#ifdef PDLP_VERBOSE_MODE
      std::cout << "    Trust Region Restart To Average" << std::endl;
#endif
      raft::copy(pdhg_solver.get_primal_solution().data(),
                 candidate_duality_gap_->primal_solution_.data(),
                 primal_size_h_,
                 stream_view_);
      raft::copy(pdhg_solver.get_dual_solution().data(),
                 candidate_duality_gap_->dual_solution_.data(),
                 dual_size_h_,
                 stream_view_);
      set_last_restart_was_average(true);
    } else
      set_last_restart_was_average(false);

    if (pdlp_hyper_params::compute_last_restart_before_new_primal_weight) {
      update_last_restart_information(*candidate_duality_gap_, primal_weight);
      compute_new_primal_weight(
        *candidate_duality_gap_, primal_weight, step_size, primal_step_size, dual_step_size);
    } else {
      compute_new_primal_weight(
        *candidate_duality_gap_, primal_weight, step_size, primal_step_size, dual_step_size);
      update_last_restart_information(*candidate_duality_gap_, primal_weight);
    }
    reset_internal();
    weighted_average_solution_.reset_weighted_average_solution();
  }
}

template <typename f_t>
__global__ void kernel_compute_kkt_score(const f_t* l2_primal_residual,
                                         const f_t* l2_dual_residual,
                                         const f_t* gap,
                                         const f_t* primal_weight,
                                         f_t* kkt_score)
{
  const f_t weight_squared = *primal_weight * *primal_weight;
  *kkt_score               = raft::sqrt(weight_squared * *l2_primal_residual * *l2_primal_residual +
                          *l2_dual_residual * *l2_dual_residual / weight_squared + *gap * *gap);
#ifdef PDLP_DEBUG_MODE
  printf(
    "kernel_compute_kkt_score=%lf weight=%lf (^2 %lf), l2_primal_residual=%lf (^2 %lf), "
    "l2_dual_residual=%lf (^2 %lf), fap=%lf (^2 %lf)\n",
    *kkt_score,
    *primal_weight,
    weight_squared,
    *l2_primal_residual,
    (*l2_primal_residual * *l2_primal_residual),
    *l2_dual_residual,
    (*l2_dual_residual * *l2_dual_residual),
    *gap,
    (*gap * *gap));
#endif
}

template <typename i_t, typename f_t>
f_t pdlp_restart_strategy_t<i_t, f_t>::compute_kkt_score(
  const rmm::device_scalar<f_t>& l2_primal_residual,
  const rmm::device_scalar<f_t>& l2_dual_residual,
  const rmm::device_scalar<f_t>& gap,
  const rmm::device_scalar<f_t>& primal_weight)
{
  kernel_compute_kkt_score<f_t><<<1, 1, 0, stream_view_>>>(l2_primal_residual.data(),
                                                           l2_dual_residual.data(),
                                                           gap.data(),
                                                           primal_weight.data(),
                                                           tmp_kkt_score_.data());
  return tmp_kkt_score_.value(stream_view_);
}

template <typename i_t, typename f_t>
bool pdlp_restart_strategy_t<i_t, f_t>::kkt_decay(f_t candidate_kkt_score)
{
#ifdef PDLP_DEBUG_MODE
  std::cout << "last_candidate_kkt_score=" << last_candidate_kkt_score << std::endl;
  std::cout << "last_restart_kkt_score=" << last_restart_kkt_score << std::endl;
#endif
  if (candidate_kkt_score <
      pdlp_hyper_params::host_default_sufficient_reduction_for_restart * last_restart_kkt_score) {
#ifdef PDLP_DEBUG_MODE
    std::cout << "kkt_sufficient_decay restart" << std::endl;
#endif
    return true;
  } else if (candidate_kkt_score < pdlp_hyper_params::host_default_necessary_reduction_for_restart *
                                     last_restart_kkt_score &&
             candidate_kkt_score > last_candidate_kkt_score) {
#ifdef PDLP_DEBUG_MODE
    std::cout << "kkt_necessary_decay restart" << std::endl;
#endif
    return true;
  }
  return false;
}

template <typename i_t, typename f_t>
bool pdlp_restart_strategy_t<i_t, f_t>::kkt_restart_conditions(f_t candidate_kkt_score,
                                                               i_t total_number_of_iterations)
{
  return should_do_artificial_restart(total_number_of_iterations) == 1 ||
         kkt_decay(candidate_kkt_score);
}

template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::update_distance(pdhg_solver_t<i_t, f_t>& pdhg_solver,
                                                        rmm::device_scalar<f_t>& primal_weight,
                                                        rmm::device_scalar<f_t>& primal_step_size,
                                                        rmm::device_scalar<f_t>& dual_step_size,
                                                        const rmm::device_scalar<f_t>& step_size)
{
  raft::copy(current_duality_gap_.primal_solution_.data(),
             pdhg_solver.get_primal_solution().data(),
             primal_size_h_,
             stream_view_);
  raft::copy(current_duality_gap_.dual_solution_.data(),
             pdhg_solver.get_dual_solution().data(),
             dual_size_h_,
             stream_view_);
  candidate_duality_gap_ = &current_duality_gap_;

  // Comupute distance traveled
  compute_distance_traveled_from_last_restart(*candidate_duality_gap_,
                                              primal_weight,
                                              pdhg_solver.get_primal_tmp_resource(),
                                              pdhg_solver.get_dual_tmp_resource());

  update_last_restart_information(*candidate_duality_gap_, primal_weight);
  compute_new_primal_weight(
    *candidate_duality_gap_, primal_weight, step_size, primal_step_size, dual_step_size);
}

template <typename i_t, typename f_t>
bool pdlp_restart_strategy_t<i_t, f_t>::run_kkt_restart(
  pdhg_solver_t<i_t, f_t>& pdhg_solver,
  rmm::device_uvector<f_t>& primal_solution_avg,
  rmm::device_uvector<f_t>& dual_solution_avg,
  const convergence_information_t<i_t, f_t>& current_convergence_information,
  const convergence_information_t<i_t, f_t>& average_convergence_information,
  rmm::device_scalar<f_t>& primal_step_size,
  rmm::device_scalar<f_t>& dual_step_size,
  rmm::device_scalar<f_t>& primal_weight,
  const rmm::device_scalar<f_t>& step_size,
  i_t total_number_of_iterations)
{
#ifdef PDLP_DEBUG_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "Running KKT scheme" << std::endl;
#endif
  // For KKT restart we need current and average convergeance information:
  // Primal / Dual residual and duality gap
  // Both of them are computed before to know if optimality has been reached

#ifdef PDLP_DEBUG_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "  Current convergeance information:"
            << "    l2_primal_residual="
            << current_convergence_information.get_l2_primal_residual().value(stream_view_)
            << "    l2_dual_residual="
            << current_convergence_information.get_l2_dual_residual().value(stream_view_)
            << "    gap=" << current_convergence_information.get_gap().value(stream_view_)
            << std::endl;
#endif

  const f_t current_kkt_score =
    compute_kkt_score(current_convergence_information.get_l2_primal_residual(),
                      current_convergence_information.get_l2_dual_residual(),
                      current_convergence_information.get_gap(),
                      primal_weight);

  // Before computing average, check if it's a first iteration after a restart
  // Then there is no average since it's reset after each restart and no kkt candidate yet
  if (weighted_average_solution_.get_iterations_since_last_restart() == 0) {
#ifdef PDLP_DEBUG_MODE
    std::cout << "    First call too kkt restart, returning:" << std::endl;
#endif
    last_candidate_kkt_score = current_kkt_score;
    last_restart_kkt_score   = current_kkt_score;
    return false;
  }

  const f_t average_kkt_score =
    compute_kkt_score(average_convergence_information.get_l2_primal_residual(),
                      average_convergence_information.get_l2_dual_residual(),
                      average_convergence_information.get_gap(),
                      primal_weight);
  f_t candidate_kkt_score;

  bool restart_to_average;
  if (current_kkt_score < average_kkt_score) {
    restart_to_average  = false;
    candidate_kkt_score = current_kkt_score;
  } else {
    restart_to_average  = true;
    candidate_kkt_score = average_kkt_score;
  }

#ifdef PDLP_DEBUG_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "    current_kkt_score=" << current_kkt_score << "\n"
            << "    average_kkt_score=" << average_kkt_score << "\n"
            << "    candidate_kkt_score=" << candidate_kkt_score << "\n"
            << "    restart_to_average=" << restart_to_average << std::endl;
#endif

  bool has_restarted = false;

  if (kkt_restart_conditions(candidate_kkt_score, total_number_of_iterations)) {
    has_restarted = true;

    // If restart, need to compute distance travaled from last either from current or average
    // This is necessary to compute the new primal weight

#ifdef PDLP_DEBUG_MODE
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    std::cout << "  Doing KKT restart" << std::endl;
#endif

    // Set which localized_duality_gap_container will be used for candidate
    // (We could save the container copy but compute_distance_traveled_from_last_restart works with
    // containers)
    if (restart_to_average && !pdlp_hyper_params::never_restart_to_average) {
#ifdef PDLP_DEBUG_MODE
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      std::cout << "    KKT restart to average" << std::endl;
#endif

      raft::copy(avg_duality_gap_.primal_solution_.data(),
                 primal_solution_avg.data(),
                 primal_size_h_,
                 stream_view_);
      raft::copy(avg_duality_gap_.dual_solution_.data(),
                 dual_solution_avg.data(),
                 dual_size_h_,
                 stream_view_);
      candidate_duality_gap_ = &avg_duality_gap_;
    } else {
#ifdef PDLP_DEBUG_MODE
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      std::cout << "    KKT no restart to average" << std::endl;
#endif
      raft::copy(current_duality_gap_.primal_solution_.data(),
                 pdhg_solver.get_saddle_point_state().get_primal_solution().data(),
                 primal_size_h_,
                 stream_view_);
      raft::copy(current_duality_gap_.dual_solution_.data(),
                 pdhg_solver.get_saddle_point_state().get_dual_solution().data(),
                 dual_size_h_,
                 stream_view_);
      candidate_duality_gap_ = &current_duality_gap_;
    }

    // Comupute distance traveled
    compute_distance_traveled_from_last_restart(*candidate_duality_gap_,
                                                primal_weight,
                                                pdhg_solver.get_primal_tmp_resource(),
                                                pdhg_solver.get_dual_tmp_resource());

    if (restart_to_average && !pdlp_hyper_params::never_restart_to_average) {
      // Candidate is pointing to the average
      raft::copy(pdhg_solver.get_primal_solution().data(),
                 candidate_duality_gap_->primal_solution_.data(),
                 primal_size_h_,
                 stream_view_);
      raft::copy(pdhg_solver.get_dual_solution().data(),
                 candidate_duality_gap_->dual_solution_.data(),
                 dual_size_h_,
                 stream_view_);
      set_last_restart_was_average(true);
    } else
      set_last_restart_was_average(false);

    if (pdlp_hyper_params::compute_last_restart_before_new_primal_weight) {
      // Save last restart data (primal/dual solution and distance traveled)
      update_last_restart_information(*candidate_duality_gap_, primal_weight);
      compute_new_primal_weight(
        *candidate_duality_gap_, primal_weight, step_size, primal_step_size, dual_step_size);
    } else {
      // Save last restart data (primal/dual solution and distance traveled)
      compute_new_primal_weight(
        *candidate_duality_gap_, primal_weight, step_size, primal_step_size, dual_step_size);
      update_last_restart_information(*candidate_duality_gap_, primal_weight);
    }

    // Reset average
    weighted_average_solution_.reset_weighted_average_solution();

    // Set last restart candidate
    last_restart_kkt_score = candidate_kkt_score;
  } else {
#ifdef PDLP_DEBUG_MODE
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    std::cout << "KKT conditions not met for a restart" << std::endl;
#endif
  }

  // Record last kkt candidate
  last_candidate_kkt_score = candidate_kkt_score;

#ifdef PDLP_DEBUG_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "last_restart_kkt_score=" << last_restart_kkt_score
            << "last_candidate_kkt_score=" << last_candidate_kkt_score << std::endl;
#endif

  return has_restarted;
}

template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::compute_restart(
  pdhg_solver_t<i_t, f_t>& pdhg_solver,
  rmm::device_uvector<f_t>& primal_solution_avg,
  rmm::device_uvector<f_t>& dual_solution_avg,
  const i_t total_number_of_iterations,
  rmm::device_scalar<f_t>& primal_step_size,
  rmm::device_scalar<f_t>& dual_step_size,
  rmm::device_scalar<f_t>& primal_weight,
  const rmm::device_scalar<f_t>& step_size,
  const convergence_information_t<i_t, f_t>& current_convergence_information,
  const convergence_information_t<i_t, f_t>& average_convergence_information)
{
  raft::common::nvtx::range fun_scope("compute_restart");

  if (is_KKT_restart<i_t, f_t>()) {
    run_kkt_restart(pdhg_solver,
                    primal_solution_avg,
                    dual_solution_avg,
                    current_convergence_information,
                    average_convergence_information,
                    primal_step_size,
                    dual_step_size,
                    primal_weight,
                    step_size,
                    total_number_of_iterations);
  } else if (pdlp_hyper_params::restart_strategy ==
             static_cast<int>(restart_strategy_t::TRUST_REGION_RESTART)) {
    run_trust_region_restart(pdhg_solver,
                             primal_solution_avg,
                             dual_solution_avg,
                             total_number_of_iterations,
                             primal_step_size,
                             dual_step_size,
                             primal_weight,
                             step_size);
  } else {
    EXE_CUOPT_FAIL("Bad restart value");
  }
}

template <typename i_t, typename f_t>
__global__ void compute_new_primal_weight_kernel(
  const typename localized_duality_gap_container_t<i_t, f_t>::view_t duality_gap_view,
  f_t* primal_weight,
  const f_t* step_size,
  f_t* primal_step_size,
  f_t* dual_step_size)
{
  if (threadIdx.x + blockIdx.x * blockDim.x > 0) { return; }

  f_t primal_distance = raft::sqrt(*duality_gap_view.primal_distance_traveled);
  f_t dual_distance   = raft::sqrt(*duality_gap_view.dual_distance_traveled);

#ifdef PDLP_DEBUG_MODE
  printf("Compute new primal weight: primal_distance=%lf dual_distance=%lf\n",
         primal_distance,
         dual_distance);
#endif

  if (primal_distance < 0.0 + safe_guard_for_extreme_values_in_primal_weight_computation<f_t> ||
      primal_distance >= 1 / safe_guard_for_extreme_values_in_primal_weight_computation<f_t> ||
      dual_distance < 0.0 + safe_guard_for_extreme_values_in_primal_weight_computation<f_t> ||
      dual_distance >= 1 / safe_guard_for_extreme_values_in_primal_weight_computation<f_t>) {
#ifdef PDLP_DEBUG_MODE
    printf("Compute new primal weight: Invalid distance, returning without updates\n");
#endif
    return;
  }

  f_t new_primal_weight_estimate = dual_distance / primal_distance;

  f_t log_primal_weight =
    pdlp_hyper_params::default_primal_weight_update_smoothing *
      raft::myLog(new_primal_weight_estimate) +
    (1 - pdlp_hyper_params::default_primal_weight_update_smoothing) * raft::myLog(*primal_weight);

  *primal_weight = raft::myExp(log_primal_weight);
  cuopt_assert(!isnan(*primal_weight), "primal weight can't be nan");
  cuopt_assert(!isinf(*primal_weight), "primal weight can't be inf");
  *primal_step_size = *step_size / *primal_weight;
  *dual_step_size   = *step_size * *primal_weight;
#ifdef PDLP_DEBUG_MODE
  printf(
    "Compute new primal weight: primal_ratio=%lf, log_primal_weight=%lf new_primal_weight=%lf\n",
    new_primal_weight_estimate,
    log_primal_weight,
    *primal_weight);
#endif
}

template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::compute_new_primal_weight(
  localized_duality_gap_container_t<i_t, f_t>& duality_gap,
  rmm::device_scalar<f_t>& primal_weight,
  const rmm::device_scalar<f_t>& step_size,
  rmm::device_scalar<f_t>& primal_step_size,
  rmm::device_scalar<f_t>& dual_step_size)
{
  raft::common::nvtx::range fun_scope("compute_new_primal_weight");

  compute_new_primal_weight_kernel<i_t, f_t><<<1, 1, 0, stream_view_>>>(duality_gap.view(),
                                                                        primal_weight.data(),
                                                                        step_size.data(),
                                                                        primal_step_size.data(),
                                                                        dual_step_size.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::distance_squared_moved_from_last_restart_period(
  const rmm::device_uvector<f_t>& new_solution,
  const rmm::device_uvector<f_t>& old_solution,
  rmm::device_uvector<f_t>& tmp,
  i_t size_of_solutions_h,
  i_t stride,
  rmm::device_scalar<f_t>& distance_moved)
{
  raft::common::nvtx::range fun_scope("distance_squared_moved_from_last_restart_period");
#ifdef PDLP_DEBUG_MODE
  rmm::device_scalar<f_t> debuga{stream_view_};
  rmm::device_scalar<f_t> debugb{stream_view_};
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
                                                  size_of_solutions_h,
                                                  old_solution.data(),
                                                  stride,
                                                  old_solution.data(),
                                                  stride,
                                                  debuga.data(),
                                                  stream_view_));
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
                                                  size_of_solutions_h,
                                                  new_solution.data(),
                                                  stride,
                                                  new_solution.data(),
                                                  stride,
                                                  debugb.data(),
                                                  stream_view_));
  std::cout << "Distance squared moved:\n"
            << "  Old location=" << debuga.value(stream_view_) << "\n"
            << "  New location=" << debugb.value(stream_view_) << std::endl;
#endif

  raft::linalg::binaryOp(tmp.data(),
                         old_solution.data(),
                         new_solution.data(),
                         size_of_solutions_h,
                         a_sub_scalar_times_b<f_t>(reusable_device_scalar_value_1_.data()),
                         stream_view_);

  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
                                                  size_of_solutions_h,
                                                  tmp.data(),
                                                  stride,
                                                  tmp.data(),
                                                  stride,
                                                  distance_moved.data(),
                                                  stream_view_));
}

template <typename i_t, typename f_t>
__global__ void compute_distance_traveled_last_restart_kernel(
  const typename localized_duality_gap_container_t<i_t, f_t>::view_t duality_gap_view,
  const f_t* primal_weight,
  f_t* distance_traveled)
{
  if (threadIdx.x + blockIdx.x * blockDim.x > 0) { return; }

  f_t primal_weight_ = *primal_weight;

  *distance_traveled = raft::sqrt(*duality_gap_view.primal_distance_traveled *
                                    pdlp_hyper_params::primal_distance_smoothing * primal_weight_ +
                                  *duality_gap_view.dual_distance_traveled *
                                    (pdlp_hyper_params::dual_distance_smoothing / primal_weight_));
}

template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::update_last_restart_information(
  localized_duality_gap_container_t<i_t, f_t>& duality_gap, rmm::device_scalar<f_t>& primal_weight)
{
  raft::common::nvtx::range fun_scope("update_last_restart_information");

  compute_distance_traveled_last_restart_kernel<i_t, f_t><<<1, 1, 0, stream_view_>>>(
    duality_gap.view(), primal_weight.data(), last_restart_duality_gap_.distance_traveled_.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  raft::copy(last_restart_duality_gap_.primal_solution_.data(),
             duality_gap.primal_solution_.data(),
             primal_size_h_,
             stream_view_);
  raft::copy(last_restart_duality_gap_.dual_solution_.data(),
             duality_gap.dual_solution_.data(),
             dual_size_h_,
             stream_view_);

  last_restart_length_ = weighted_average_solution_.get_iterations_since_last_restart();
}

template <typename i_t, typename f_t>
__global__ void pick_restart_candidate_kernel(
  const typename localized_duality_gap_container_t<i_t, f_t>::view_t avg_duality_gap_view,
  const typename localized_duality_gap_container_t<i_t, f_t>::view_t current_duality_gap_view,
  typename pdlp_restart_strategy_t<i_t, f_t>::view_t restart_strategy_view)
{
  if (threadIdx.x + blockIdx.x * blockDim.x > 0) { return; }

  if (*current_duality_gap_view.normalized_gap / *current_duality_gap_view.distance_traveled >=
      *avg_duality_gap_view.normalized_gap / *avg_duality_gap_view.distance_traveled) {
    *restart_strategy_view.candidate_is_avg = 1;
  } else {
    *restart_strategy_view.candidate_is_avg = 0;
  }
}

template <typename i_t, typename f_t>
i_t pdlp_restart_strategy_t<i_t, f_t>::pick_restart_candidate()
{
  pick_restart_candidate_kernel<i_t, f_t>
    <<<1, 1, 0, stream_view_>>>(avg_duality_gap_.view(), current_duality_gap_.view(), this->view());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  i_t restart_to_average_h = candidate_is_avg_.value(stream_view_);
  if (restart_to_average_h) {
    candidate_duality_gap_ = &avg_duality_gap_;
  } else {
    candidate_duality_gap_ = &current_duality_gap_;
  }

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));
  return restart_to_average_h;
}

template <typename i_t, typename f_t>
__global__ void adaptive_restart_triggered(
  const typename localized_duality_gap_container_t<i_t, f_t>::view_t candidate_duality_gap_view,
  const typename localized_duality_gap_container_t<i_t, f_t>::view_t last_restart_duality_gap_view,
  typename pdlp_restart_strategy_t<i_t, f_t>::view_t restart_strategy_view)
{
  if (threadIdx.x + blockIdx.x * blockDim.x > 0) { return; }

  // First compute last restart normalized_gap
  // For current and average, they are computed in compute_localized_duality_gaps after the
  // bound_optimal_objective

  *last_restart_duality_gap_view.normalized_gap =
    (*last_restart_duality_gap_view.upper_bound_value -
     *last_restart_duality_gap_view.lower_bound_value) /
    *last_restart_duality_gap_view.distance_traveled;

  f_t gap_reduction_ratio =
    *candidate_duality_gap_view.normalized_gap / *last_restart_duality_gap_view.normalized_gap;
  if (gap_reduction_ratio < pdlp_hyper_params::default_necessary_reduction_for_restart &&
      (gap_reduction_ratio < pdlp_hyper_params::default_sufficient_reduction_for_restart ||
       gap_reduction_ratio > *restart_strategy_view.gap_reduction_ratio_last_trial)) {
    *restart_strategy_view.restart_triggered = 1;
  }
  *restart_strategy_view.gap_reduction_ratio_last_trial = gap_reduction_ratio;
}

template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::should_do_adaptive_restart_normalized_duality_gap(
  localized_duality_gap_container_t<i_t, f_t>& candidate_duality_gap,
  rmm::device_uvector<f_t>& tmp_primal,
  rmm::device_uvector<f_t>& tmp_dual,
  rmm::device_scalar<f_t>& primal_weight,
  i_t& restart)
{
  raft::common::nvtx::range fun_scope("should_do_adaptive_restart_normalized_duality_gap");
#ifdef PDLP_DEBUG_MODE
  std::cout << "Should do adaptive restart normalized duality gap" << std::endl;
#endif

  // potentially see if we can 'recompute' from cached version instead of the recomputation here
  // important to remember that it is with a potential new primal_weight -> new norms for both
  // primal and dual

  // they compute this first distance_traveled_last_restart = sqrt(
  // lri.primal_distance_moved_last_restart_period ^
  //   2 * primal_weight + lri.dual_distance_moved_last_restart_period ^ 2 / primal_weight,

  compute_distance_traveled_last_restart_kernel<i_t, f_t>
    <<<1, 1, 0, stream_view_>>>(candidate_duality_gap.view(),
                                primal_weight.data(),
                                last_restart_duality_gap_.distance_traveled_.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  bound_optimal_objective(
    last_restart_duality_gap_cusparse_view_, last_restart_duality_gap_, tmp_primal, tmp_dual);

  adaptive_restart_triggered<i_t, f_t><<<1, 1, 0, stream_view_>>>(
    candidate_duality_gap.view(), last_restart_duality_gap_.view(), this->view());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  restart = restart_triggered_.value(stream_view_);
}

template <typename i_t, typename f_t>
i_t pdlp_restart_strategy_t<i_t, f_t>::should_do_artificial_restart(
  i_t total_number_of_iterations) const
{
  // if long enough since last restart (artificial)
#ifdef PDLP_DEBUG_MODE
  std::cout << "Artifical restart:\n"
            << "    iterations_since_last_restart="
            << weighted_average_solution_.get_iterations_since_last_restart() << "\n"
            << "    total_number_of_iteration=" << total_number_of_iterations << "\n"
            << "    pdlp_hyper_params::default_artificial_restart_threshold="
            << pdlp_hyper_params::default_artificial_restart_threshold << std::endl;
#endif
  if (weighted_average_solution_.get_iterations_since_last_restart() >=
      pdlp_hyper_params::default_artificial_restart_threshold * total_number_of_iterations) {
#ifdef PDLP_VERBOSE_MODE
    std::cout << "    Doing artifical restart" << std::endl;
#endif
    return 1;
  }

  return 0;
}

template <typename i_t, typename f_t>
__global__ void compute_normalized_gaps_kernel(
  typename localized_duality_gap_container_t<i_t, f_t>::view_t avg_duality_gap_view,
  typename localized_duality_gap_container_t<i_t, f_t>::view_t current_duality_gap_view)
{
  if (threadIdx.x + blockIdx.x * blockDim.x > 0) { return; }
  cuopt_assert(
    *current_duality_gap_view.upper_bound_value >= *current_duality_gap_view.lower_bound_value,
    "The upper bound for the objective value of the current problem must be larger than "
    "the lower bound");

  *avg_duality_gap_view.normalized_gap =
    (*avg_duality_gap_view.upper_bound_value - *avg_duality_gap_view.lower_bound_value) /
    *avg_duality_gap_view.distance_traveled;
  *current_duality_gap_view.normalized_gap =
    (*current_duality_gap_view.upper_bound_value - *current_duality_gap_view.lower_bound_value) /
    *current_duality_gap_view.distance_traveled;
}

template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::compute_localized_duality_gaps(
  saddle_point_state_t<i_t, f_t>& current_saddle_point_state,
  rmm::device_uvector<f_t>& primal_solution_avg,
  rmm::device_uvector<f_t>& dual_solution_avg,
  rmm::device_scalar<f_t>& primal_weight,
  rmm::device_uvector<f_t>& tmp_primal,
  rmm::device_uvector<f_t>& tmp_dual)
{
  raft::common::nvtx::range fun_scope("compute_localized_duality_gaps");
#ifdef PDLP_DEBUG_MODE
  std::cout << "Compute localized duality gaps:" << std::endl;
#endif

  // copy avg solutions
  raft::copy(avg_duality_gap_.primal_solution_.data(),
             primal_solution_avg.data(),
             primal_size_h_,
             stream_view_);
  raft::copy(
    avg_duality_gap_.dual_solution_.data(), dual_solution_avg.data(), dual_size_h_, stream_view_);
  //  copy current solutions
  raft::copy(current_duality_gap_.primal_solution_.data(),
             current_saddle_point_state.get_primal_solution().data(),
             primal_size_h_,
             stream_view_);
  raft::copy(current_duality_gap_.dual_solution_.data(),
             current_saddle_point_state.get_dual_solution().data(),
             dual_size_h_,
             stream_view_);

  // Compute bar{r } _i
  compute_distance_traveled_from_last_restart(
    avg_duality_gap_, primal_weight, tmp_primal, tmp_dual);

  // Compute hat{r } _i
  compute_distance_traveled_from_last_restart(
    current_duality_gap_, primal_weight, tmp_primal, tmp_dual);

  // Compute Delta_{bar{r}_i}(bar{w}_{i+1}) localized_duality_gap_at_average
  bound_optimal_objective(avg_duality_gap_cusparse_view_, avg_duality_gap_, tmp_primal, tmp_dual);

  // Compute Delta_{hat{r}_i}(hat{w}_{i+1}) localized_duality_gap_at_current
  bound_optimal_objective(
    current_duality_gap_cusparse_view_, current_duality_gap_, tmp_primal, tmp_dual);

  compute_normalized_gaps_kernel<i_t, f_t>
    <<<1, 1, 0, stream_view_>>>(avg_duality_gap_.view(), current_duality_gap_.view());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::bound_optimal_objective(
  cusparse_view_t<i_t, f_t>& cusparse_view,
  localized_duality_gap_container_t<i_t, f_t>& duality_gap,
  rmm::device_uvector<f_t>& tmp_primal,
  rmm::device_uvector<f_t>& tmp_dual)
{
  raft::common::nvtx::range fun_scope("bound_optimal_objective");
#ifdef PDLP_DEBUG_MODE
  std::cout << "Bound optimal objective:" << std::endl;
#endif

  compute_primal_gradient(duality_gap, cusparse_view);
  compute_dual_gradient(duality_gap, cusparse_view, tmp_dual);
  compute_lagrangian_value(duality_gap, cusparse_view, tmp_primal, tmp_dual);

  solve_bound_constrained_trust_region(duality_gap, tmp_primal, tmp_dual);
}

template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::compute_bound(const rmm::device_uvector<f_t>& solution_tr,
                                                      const rmm::device_uvector<f_t>& solution,
                                                      const rmm::device_uvector<f_t>& gradient,
                                                      const rmm::device_scalar<f_t>& lagrangian,
                                                      const i_t size,
                                                      const i_t stride,
                                                      rmm::device_uvector<f_t>& tmp,
                                                      rmm::device_scalar<f_t>& bound)
{
  raft::common::nvtx::range fun_scope("compute_bound");
#ifdef PDLP_DEBUG_MODE
  std::cout << "Compute bound" << std::endl;
#endif
  raft::linalg::eltwiseSub(tmp.data(), solution_tr.data(), solution.data(), size, stream_view_);

  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
                                                  size,
                                                  tmp.data(),
                                                  stride,
                                                  gradient.data(),
                                                  stride,
                                                  bound.data(),
                                                  stream_view_));

  raft::linalg::eltwiseAdd(bound.data(), bound.data(), lagrangian.data(), 1, stream_view_);
}

template <typename i_t, typename f_t>
__global__ void target_threshold_determination_kernel(
  typename pdlp_restart_strategy_t<i_t, f_t>::view_t restart_strategy_view,
  const f_t* target_radius,
  const f_t* threshold_max_element,
  const f_t* threshold_max_element_dual)
{
  if (threadIdx.x + blockIdx.x * blockDim.x > 0) { return; }

  const f_t target_radius_ = *target_radius;

  // target_threshold is the solution of
  // low_radius_sq + target_threshold^2 * high_radius_sq = target_radius^2.
  if (*restart_strategy_view.high_radius_squared <= 0.0) {
    // Special case: high_radius_sq = 0.0, means all bounds hit before reaching target radius.
    *restart_strategy_view.target_threshold = *threshold_max_element;
  } else {
    *restart_strategy_view.target_threshold =
      raft::sqrt(((target_radius_ * target_radius_) - *restart_strategy_view.low_radius_squared) /
                 *restart_strategy_view.high_radius_squared);
  }
}

template <typename i_t, typename f_t>
DI f_t
compute_median(const typename pdlp_restart_strategy_t<i_t, f_t>::view_t& restart_strategy_view,
               i_t range_low,
               i_t range_high)
{
  cuopt_assert(range_low < range_high, "range_low should be stricly lower than range_high");
  const i_t size = range_high - range_low;
  if ((size & 1) == 0) {
    // Even, return average
    return f_t(0.5) * (restart_strategy_view.threshold[range_low + size / 2 - 1] +
                       restart_strategy_view.threshold[range_low + size / 2]);
  } else
    return restart_strategy_view.threshold[range_low + size / 2];
}

template <typename i_t, typename f_t>
DI void clamp_test_points(
  const typename pdlp_restart_strategy_t<i_t, f_t>::view_t& restart_strategy_view,
  const typename problem_t<i_t, f_t>::view_t& op_problem_view,
  f_t test_threshold,
  i_t range_low,
  i_t range_high)
{
  cuopt_assert(range_low < range_high, "range_low should be stricly lower than range_high");
  for (int i = blockIdx.x * blockDim.x + threadIdx.x + range_low; i < range_high;
       i += blockDim.x * gridDim.x) {
    const f_t lower_bound_value = restart_strategy_view.lower_bound[i];
    const f_t upper_bound_value = restart_strategy_view.upper_bound[i];
    restart_strategy_view.test_point[i] =
      raft::min<f_t>(raft::max<f_t>(restart_strategy_view.center_point[i] +
                                      test_threshold * restart_strategy_view.direction_full[i],
                                    lower_bound_value),
                     upper_bound_value);
  }

  cg::this_grid().sync();
}

// All live blocks accumulate then deterministic reduce to device the weighted_norm
// Results is "squared" by default (norm without sqrt)
template <typename i_t, typename f_t, int BLOCK_SIZE, typename functor_t>
DI void device_weighted_norm(
  const typename pdlp_restart_strategy_t<i_t, f_t>::view_t& restart_strategy_view,
  f_t* weighted_norm,
  i_t range_low,
  i_t range_high,
  raft::device_span<f_t> shared,
  functor_t functor,
  bool add = false)
{
  cuopt_assert(range_low < range_high, "Range high must be strictly greater than range low");

  // Compute per block weighted_norm

  f_t local_weighted_norm = f_t(0);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x + range_low; i < range_high;
       i += blockDim.x * gridDim.x) {
    const f_t val = functor(i);
    local_weighted_norm += (val * val) * restart_strategy_view.weights[i];
  }

  local_weighted_norm = deterministic_block_reduce<f_t, BLOCK_SIZE>(shared, local_weighted_norm);

  if (threadIdx.x == 0)
    restart_strategy_view.shared_live_kernel_accumulator[blockIdx.x] = local_weighted_norm;

  cg::this_grid().sync();

  // Reduce in the global shared_live_kernel_accumulator

  if (blockIdx.x == 0) {
    local_weighted_norm = f_t(0);

    for (int i = threadIdx.x; i < restart_strategy_view.shared_live_kernel_accumulator.size();
         i += BLOCK_SIZE)
      local_weighted_norm += restart_strategy_view.shared_live_kernel_accumulator[i];

    local_weighted_norm = deterministic_block_reduce<f_t, BLOCK_SIZE>(shared, local_weighted_norm);

    if (threadIdx.x == 0) {
      if (add)
        *weighted_norm += local_weighted_norm;
      else
        *weighted_norm = local_weighted_norm;
    }
  }

  cg::this_grid().sync();
}

// Find the new high_range : lowest index respecting threshold[i] < test_threshold (considering
// everything is sorted) Once a range of threads has reach the tilting point of condition, keep the
// one with the lowest index Example: [0, 1, 2, 3, 3.5, 3.5, 4, 5] With threshold = 3.5 If updating
// the high_range : we want to keep values [0, 1, 2, 3], lowest atomicMin triggered will be on
// first 3.5. Since high index is exlusive, it's correct
template <typename i_t, typename f_t>
DI void update_range_high(
  const typename pdlp_restart_strategy_t<i_t, f_t>::view_t& restart_strategy_view,
  f_t test_threshold,
  i_t range_low,
  i_t range_high,
  i_t* range)
{
  cuopt_assert(range_low < range_high, "Range high must be strictly greater than range low");

  __shared__ i_t shared_index;
  __shared__ bool has_found;

  if (threadIdx.x == 0) {
    shared_index = std::numeric_limits<i_t>::max();
    has_found    = false;
  }
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x + range_low; i < range_high;
       i += blockDim.x * gridDim.x) {
    // One thread of current has reach the limit
    // All threads which came across atomically write, only min is kept
    if (restart_strategy_view.threshold[i] >= test_threshold) {
      raft::myAtomicMin(&shared_index, i);
      has_found = true;  // Concurrent write is okay (as long as true is kept at the end)
      break;
    }
  }

  __syncthreads();

  cuopt_assert(
    (has_found) ? shared_index < range_high : shared_index == std::numeric_limits<i_t>::max(),
    "Invalid shared_index value");

  // Find the global min
  if (threadIdx.x == 0 && has_found) raft::myAtomicMin(range, shared_index);

  cg::this_grid().sync();
}

// Find the new low_range : highest index respecting threshold[i] <= test_threshold (considering
// everything is sorted) Once a range of threads has reach the tilting point of condition, keep the
// one with the highest index Example: [0, 1, 2, 3, 3.5, 3.5, 4, 5] With threshold = 3.5 If updating
// the low range :  we want to keep values [4, 5], highest atomicMax triggered will be on last 3.5,
// so pick index at + 1
template <typename i_t, typename f_t>
DI void update_range_low(
  const typename pdlp_restart_strategy_t<i_t, f_t>::view_t& restart_strategy_view,
  f_t test_threshold,
  i_t range_low,
  i_t range_high,
  i_t* range)
{
  cuopt_assert(range_low < range_high, "Range high must be strictly greater than range low");

  __shared__ i_t shared_index;
  __shared__ bool has_found;

  if (threadIdx.x == 0) {
    shared_index = std::numeric_limits<i_t>::min();
    has_found    = false;
  }
  __syncthreads();

  for (int i = range_high - 1 - (blockIdx.x * blockDim.x + threadIdx.x); i >= range_low;
       i -= blockDim.x * gridDim.x) {
    // One thread of current has reach the limit
    // All threads which came across atomically write, only max is kept
    if (restart_strategy_view.threshold[i] <= test_threshold) {
      raft::myAtomicMax(&shared_index, i);
      has_found = true;  // Concurrent write is okay (as long as true is kept at the end)
      break;
    }
  }

  __syncthreads();

  cuopt_assert(
    (has_found) ? (shared_index + 1) > range_low : shared_index == std::numeric_limits<i_t>::min(),
    "Invalid shared_index value");

  // Find the global max
  if (threadIdx.x == 0 && has_found)
    raft::myAtomicMax(range, shared_index + 1);  // + 1 because low should start after tilting point

  cg::this_grid().sync();
}

// Range low/high are passed as parameter so that function can be reused accross primal / dual
template <typename i_t, typename f_t, int BLOCK_SIZE>
__global__ void solve_bound_constrained_trust_region_kernel(
  typename pdlp_restart_strategy_t<i_t, f_t>::view_t restart_strategy_view,
  typename problem_t<i_t, f_t>::view_t op_problem_view,
  i_t* testing_range_low,
  i_t* testing_range_high,
  f_t* test_radius_squared,
  f_t* low_radius_squared,
  f_t* high_radius_squared,
  const f_t* target_radius)
{
  __shared__ f_t shared[BLOCK_SIZE / raft::WarpSize];
  auto shared_accumulator = raft::device_span<f_t>{shared, BLOCK_SIZE / raft::WarpSize};

  while (*testing_range_low != *testing_range_high) {
    const i_t range_low  = *testing_range_low;
    const i_t range_high = *testing_range_high;
    cuopt_assert(range_low < range_high, "range_low should be stricly lower than range_high");

    // Each of those calls perform an implicit grid sync
    const f_t test_threshold =
      compute_median<i_t, f_t>(restart_strategy_view, range_low, range_high);
    clamp_test_points(
      restart_strategy_view, op_problem_view, test_threshold, range_low, range_high);
    // Compute test radius
    device_weighted_norm<i_t, f_t, BLOCK_SIZE>(
      restart_strategy_view,
      test_radius_squared,
      range_low,
      range_high,
      shared_accumulator,
      minus<i_t, f_t>(restart_strategy_view.test_point, restart_strategy_view.center_point));

    const bool threshold_too_high = *low_radius_squared + *test_radius_squared +
                                      (test_threshold * test_threshold) * *high_radius_squared >=
                                    (*target_radius * *target_radius);
    if (threshold_too_high)  // Reduce high range until threshold)
    {
      // Range kept is [low, new_high) | Range to compute radius is [new_high, old_high)
      update_range_high<i_t, f_t>(
        restart_strategy_view, test_threshold, range_low, range_high, testing_range_high);
      const i_t new_high = *testing_range_high;
      cuopt_assert(new_high != range_high, "New range high can't be same as old");
      // Compute high_radius on discarded range
      if (range_high - new_high > 0)  // Range can be empty if they were all discarded
        device_weighted_norm<i_t, f_t, BLOCK_SIZE>(
          restart_strategy_view,
          high_radius_squared,
          new_high,
          range_high,
          shared_accumulator,
          identity<i_t, f_t>(restart_strategy_view.direction_full),
          true);
    } else  // Increase low range until threshold]
    {
      // Range kept is [new_low, high) | Range to compute radius is [old_low, new_low)
      update_range_low<i_t, f_t>(
        restart_strategy_view, test_threshold, range_low, range_high, testing_range_low);
      const i_t new_low = *testing_range_low;
      cuopt_assert(new_low != range_low, "New range low can't be same as old");
      // No need to reclamp center_points (done above)
      // Directly compute low_radius on discarded range
      if (new_low - range_low > 0)  // Range can be empty if they were all discarded
        device_weighted_norm<i_t, f_t, BLOCK_SIZE>(
          restart_strategy_view,
          low_radius_squared,
          range_low,
          new_low,
          shared_accumulator,
          minus<i_t, f_t>(restart_strategy_view.test_point, restart_strategy_view.center_point),
          true);
    }
  }
}

/** From Julia
Finds a solution to the problem:

argmin_x objective_vector' * x
s.t. variable_lower_bounds <= x <= variable_upper_bounds                    (1)
     || x - center_point || <= target_radius

where || . || is weighted by norm_weights.

for a positive value of target_radius, by solving the related problem

argmin_t objective_vector' * x
s.t. x := min(max(center_point - target_threshold * objective_vector, variable_lower_bounds),
              variable_upper_bounds)
     || x - center_point || <= target_radius

Note that the definition of x is just applying the lower/upper bound constraints
to center_point - target_threshold * objective_vector.

This problem is solved by computing the breakpoint at which each component of
x switches from varying with target_threshold to being fixed at its bounds.  The radius of the
median breakpoint is evaluated, eliminating half of the components.  The process
is iterated until the argmin is identified.
*/

template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::solve_bound_constrained_trust_region(
  localized_duality_gap_container_t<i_t, f_t>& duality_gap,
  rmm::device_uvector<f_t>& tmp_primal,
  rmm::device_uvector<f_t>& tmp_dual)
{
  raft::common::nvtx::range fun_scope("solve_bound_constrained_trust_region");
#ifdef PDLP_DEBUG_MODE
  std::cout << "    Solve bound constrained trust region:" << std::endl;
#endif

  // center point [primal_solution; dual_solution]
  // objective is [primal_gradient; -dual_gradient]
  raft::copy(
    center_point_.data(), duality_gap.primal_solution_.data(), primal_size_h_, stream_view_);
  raft::copy(center_point_.data() + primal_size_h_,
             duality_gap.dual_solution_.data(),
             dual_size_h_,
             stream_view_);
  raft::copy(
    objective_vector_.data(), duality_gap.primal_gradient_.data(), primal_size_h_, stream_view_);
  // need -dual_gradient
  raft::linalg::unaryOp(objective_vector_.data() + primal_size_h_,
                        duality_gap.dual_gradient_.data(),
                        dual_size_h_,
                        negate_t<f_t>(),
                        stream_view_);

  // Use high_radius_squared_ to store objective_vector l2_norm
  my_l2_norm<i_t, f_t>(objective_vector_, high_radius_squared_, handle_ptr_);
  if (duality_gap.distance_traveled_.value(stream_view_) == f_t(0.0) ||
      high_radius_squared_.value(stream_view_) == f_t(0.0)) {
    raft::copy(
      duality_gap.primal_solution_tr_.data(), center_point_.data(), primal_size_h_, stream_view_);
    raft::copy(duality_gap.dual_solution_tr_.data(),
               center_point_.data() + primal_size_h_,
               dual_size_h_,
               stream_view_);
    const f_t zero_float = 0.0;
    target_threshold_.set_value_async(zero_float, stream_view_);
  } else {
    thrust::fill(handle_ptr_->get_thrust_policy(),
                 weights_.data(),
                 weights_.data() + primal_size_h_,
                 primal_norm_weight_.value(stream_view_));
    thrust::fill(handle_ptr_->get_thrust_policy(),
                 weights_.data() + primal_size_h_,
                 weights_.data() + primal_size_h_ + dual_size_h_,
                 dual_norm_weight_.value(stream_view_));
    /* -- Init restart data -- */
    const f_t zero_float = f_t(0.0);
    high_radius_squared_.set_value_async(zero_float, stream_view_);
    low_radius_squared_.set_value_async(zero_float, stream_view_);
    RAFT_CUDA_TRY(cudaMemsetAsync(
      direction_full_.data(), 0, sizeof(f_t) * (primal_size_h_ + dual_size_h_), stream_view_));
    RAFT_CUDA_TRY(cudaMemsetAsync(
      threshold_.data(), 0, sizeof(f_t) * (primal_size_h_ + dual_size_h_), stream_view_));
    /* ----- */

    // Determine the direction which each component has moved and the threshold for when the
    // component becomes fixed by its bounds

    // Copying primal / dual bound before sorting them according to threshold
    raft::copy(
      lower_bound_.data(), problem_ptr->variable_lower_bounds.data(), primal_size_h_, stream_view_);
    raft::copy(
      upper_bound_.data(), problem_ptr->variable_upper_bounds.data(), primal_size_h_, stream_view_);
    raft::copy(lower_bound_.data() + primal_size_h_,
               transformed_constraint_lower_bounds_.data(),
               dual_size_h_,
               stream_view_);
    raft::copy(upper_bound_.data() + primal_size_h_,
               transformed_constraint_upper_bounds_.data(),
               dual_size_h_,
               stream_view_);

    thrust::for_each(handle_ptr_->get_thrust_policy(),
                     thrust::counting_iterator<i_t>(0),
                     thrust::counting_iterator<i_t>(primal_size_h_ + dual_size_h_),
                     compute_direction_and_threshold<i_t, f_t>(this->view()));

    // finding the target_threshold which defines the x (primal dual solutions concatenated) that
    // minimizes distance from the centerpoint (current primal dual solutions concatenated) to x
    // while restricting the distance to be the distance traveled since the last restart was
    // performed in practice this is done by finding the breakpoint at which each component
    // switching with target_threshold and becomes fixed at it's bounds. At each iteration we
    // eleminate half of the components since we use the median breakpoint for the radius we test.
    // (mathematical definitions are available above)

    // Compute the L2 norm on only infinite value before sorting to reduce effect of adding small /
    // big floating point values

    // Define the transformation functor
    weighted_l2_if_infinite<i_t, f_t> transform_func(this->view());

    // Create the transform iterator
    auto transformed_begin =
      thrust::make_transform_iterator(thrust::counting_iterator<i_t>(0), transform_func);
    auto transformed_end = thrust::make_transform_iterator(
      thrust::counting_iterator<i_t>(primal_size_h_ + dual_size_h_), transform_func);

    // Perform the reduction
    // Convert raw pointer to thrust::device_ptr to write directly device side through reduce
    thrust::device_ptr<f_t> thrust_hrsp(high_radius_squared_.data());
    *thrust_hrsp = thrust::reduce(handle_ptr_->get_thrust_policy(),
                                  transformed_begin,
                                  transformed_end,
                                  f_t(0.0),
                                  thrust::plus<f_t>());

    // Save direction_full_ before sorting it
    raft::copy(unsorted_direction_full_.data(),
               direction_full_.data(),
               primal_size_h_ + dual_size_h_,
               stream_view_);

    // Sort merged problem
    // This allows for easier threshold handling + bringing (-)infs values to the (beginning)end
    thrust::sort_by_key(handle_ptr_->get_thrust_policy(),
                        threshold_.data(),
                        threshold_.data() + primal_size_h_ + dual_size_h_,
                        thrust::make_zip_iterator(thrust::make_tuple(direction_full_.data(),
                                                                     lower_bound_.data(),
                                                                     upper_bound_.data(),
                                                                     center_point_.data(),
                                                                     weights_.data())));

    // Need to remove the inf part:
    // After sorting, find the highest index of -inf and lowest index of inf to adapt the low/high
    // range Example : [-inf, -inf, 4, 5, 6, inf, inf]
    auto lowest_inf = thrust::find(handle_ptr_->get_thrust_policy(),
                                   threshold_.begin(),
                                   threshold_.end(),
                                   std::numeric_limits<f_t>::infinity());
    // Easier / Cleaner than to do reverse iterator arithmetic
    f_t* start = threshold_.data();
    f_t* end   = threshold_.data() + primal_size_h_ + dual_size_h_;
    auto highest_negInf_primal =
      thrust::find(handle_ptr_->get_thrust_policy(),
                   thrust::make_reverse_iterator(thrust::device_ptr<f_t>(end)),
                   thrust::make_reverse_iterator(thrust::device_ptr<f_t>(start)),
                   -std::numeric_limits<f_t>::infinity());

    // Set ranges accordingly
    i_t index_start_primal = 0;
    i_t index_end_primal   = primal_size_h_ + dual_size_h_;
    if (highest_negInf_primal != thrust::make_reverse_iterator(thrust::device_ptr<f_t>(start))) {
      cuopt_assert(device_to_host_value(thrust::raw_pointer_cast(&*highest_negInf_primal)) ==
                     -std::numeric_limits<f_t>::infinity(),
                   "Incorrect primal reverse iterator");
      index_start_primal = thrust::raw_pointer_cast(&*highest_negInf_primal) - threshold_.data() +
                           1;  // + 1 to go after last negInf
      testing_range_low_.set_value_async(index_start_primal, stream_view_);
    } else  // No negInf found, start is 0
      testing_range_low_.set_value_async(index_start_primal, stream_view_);
    if (lowest_inf != end) {
      cuopt_assert(device_to_host_value(thrust::raw_pointer_cast(&*lowest_inf)) ==
                     std::numeric_limits<f_t>::infinity(),
                   "Incorrect primal iterator");
      index_end_primal =
        thrust::raw_pointer_cast(lowest_inf) -
        threshold_.data();  // no - 1 to go before the first inf because end is not included
      testing_range_high_.set_value_async(index_end_primal, stream_view_);
    } else  // No inf found, end is primal_size_h_
      testing_range_high_.set_value_async(index_end_primal, stream_view_);
    cuopt_assert(index_start_primal <= index_end_primal,
                 "Start should be stricly smalled than end");

    cuopt_assert(!thrust::any_of(handle_ptr_->get_thrust_policy(),
                                 threshold_.data() + index_start_primal,
                                 threshold_.data() + index_end_primal,
                                 is_nan_or_inf<f_t>()),
                 "Threshold vector should not contain inf or NaN values");

    // Init parameters for live kernel
    // Has to do this to pass lvalues (and not rvalue) to void* kernel_args
    auto restart_view        = this->view();
    auto op_view             = problem_ptr->view();
    i_t* testing_range_low   = testing_range_low_.data();
    i_t* testing_range_high  = testing_range_high_.data();
    f_t* test_radius_squared = test_radius_squared_.data();
    f_t* low_radius_squared  = low_radius_squared_.data();
    f_t* high_radius_squared = high_radius_squared_.data();
    f_t* distance_traveled   = duality_gap.distance_traveled_.data();

    void* kernel_args[] = {
      &restart_view,
      &op_view,
      &testing_range_low,
      &testing_range_high,
      &test_radius_squared,
      &low_radius_squared,
      &high_radius_squared,
      &distance_traveled,
    };
    constexpr int numThreads = 128;
    dim3 dimBlock(numThreads, 1, 1);
    // shared_live_kernel_accumulator_.size() contains deviceProp.multiProcessorCount *
    // numBlocksPerSm
    dim3 dimGrid(shared_live_kernel_accumulator_.size(), 1, 1);
    // Compute the median for the join problem, while loop is inside the live kernel
    RAFT_CUDA_TRY(cudaLaunchCooperativeKernel(
      (void*)solve_bound_constrained_trust_region_kernel<i_t, f_t, numThreads>,
      dimGrid,
      dimBlock,
      kernel_args,
      0,
      stream_view_));

    // Find max threshold for the join problem
    const f_t* max_threshold =
      thrust::max_element(handle_ptr_->get_thrust_policy(),
                          threshold_.data(),
                          threshold_.data() + primal_size_h_ + dual_size_h_);

    // we have now determined the test_threshold that should minimize the objective value of the
    // solution.

    //  if no component got fixed by their upper bound we can pick the maximum threshold to be the
    //  target_threshold which was computed before the loop in the direction_and_threshold_kernel
    // Otherwise use the test_threshold determined in the loop
    // {
    target_threshold_determination_kernel<i_t, f_t><<<1, 1, 0, stream_view_>>>(
      this->view(), duality_gap.distance_traveled_.data(), max_threshold, max_threshold);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    // }

    // Compute x (the solution which is defined by moving each component test_threshold *
    // direction[component]) clamp on upper and lower bounds.
    // Used unsorted_direction_full_ as the other one got sorted
    // {
    raft::linalg::binaryOp(duality_gap.primal_solution_tr_.data(),
                           duality_gap.primal_solution_.data(),
                           unsorted_direction_full_.data(),
                           primal_size_h_,
                           a_add_scalar_times_b<f_t>(target_threshold_.data()),
                           stream_view_);
    raft::linalg::binaryOp(duality_gap.dual_solution_tr_.data(),
                           duality_gap.dual_solution_.data(),
                           unsorted_direction_full_.data() + primal_size_h_,
                           dual_size_h_,
                           a_add_scalar_times_b<f_t>(target_threshold_.data()),
                           stream_view_);
    // project by max(min(x[i], upperbound[i]),lowerbound[i]) for primal part
    raft::linalg::ternaryOp(duality_gap.primal_solution_tr_.data(),
                            duality_gap.primal_solution_tr_.data(),
                            problem_ptr->variable_lower_bounds.data(),
                            problem_ptr->variable_upper_bounds.data(),
                            primal_size_h_,
                            clamp<f_t>(),
                            stream_view_);

    // project by max(min(y[i], upperbound[i]),lowerbound[i])
    raft::linalg::ternaryOp(duality_gap.dual_solution_tr_.data(),
                            duality_gap.dual_solution_tr_.data(),
                            transformed_constraint_lower_bounds_.data(),
                            transformed_constraint_upper_bounds_.data(),
                            dual_size_h_,
                            clamp<f_t>(),
                            stream_view_);
    // }
  }

  // Compute the current lower bound for the objective value using the primal solution_tr and
  // upper bound for the objective value using the dual solution_tr
  // {
  // -> compute 'lower bound' for saddle point (langrangian + dot(primal_tr - primal_solution,
  // primal_gradient))
  compute_bound(duality_gap.primal_solution_tr_,
                duality_gap.primal_solution_,
                duality_gap.primal_gradient_,
                duality_gap.lagrangian_value_,
                primal_size_h_,
                primal_stride,
                tmp_primal,
                duality_gap.lower_bound_value_);

  // compute 'upper bound' using dual
  compute_bound(duality_gap.dual_solution_tr_,
                duality_gap.dual_solution_,
                duality_gap.dual_gradient_,
                duality_gap.lagrangian_value_,
                dual_size_h_,
                dual_stride,
                tmp_dual,
                duality_gap.upper_bound_value_);

  // }
}

template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::compute_distance_traveled_from_last_restart(
  localized_duality_gap_container_t<i_t, f_t>& duality_gap,
  rmm::device_scalar<f_t>& primal_weight,
  rmm::device_uvector<f_t>& tmp_primal,
  rmm::device_uvector<f_t>& tmp_dual)
{
  raft::common::nvtx::range fun_scope("compute_distance_traveled_from_last_restart");
  // norm(
  //     new_primal_solution - last_restart.primal_solution,
  //   )^2

  // Julia / Paper use a weighted norm using primal weight for primal / dual distance
  // We simply use L2 norm of diff
  distance_squared_moved_from_last_restart_period(duality_gap.primal_solution_,
                                                  last_restart_duality_gap_.primal_solution_,
                                                  tmp_primal,
                                                  primal_size_h_,
                                                  primal_stride,
                                                  duality_gap.primal_distance_traveled_);

  // compute similarly for dual
  distance_squared_moved_from_last_restart_period(duality_gap.dual_solution_,
                                                  last_restart_duality_gap_.dual_solution_,
                                                  tmp_dual,
                                                  dual_size_h_,
                                                  dual_stride,
                                                  duality_gap.dual_distance_traveled_);

  // distance_traveled = primal_distance * 0.5 * primal_weight
  // + dual_distance * 0.5 / primal_weight
  compute_distance_traveled_last_restart_kernel<i_t, f_t><<<1, 1, 0, stream_view_>>>(
    duality_gap.view(), primal_weight.data(), duality_gap.distance_traveled_.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::compute_primal_gradient(
  localized_duality_gap_container_t<i_t, f_t>& duality_gap,
  cusparse_view_t<i_t, f_t>& cusparse_view)
{
  raft::common::nvtx::range fun_scope("compute_primal_gradient");
#ifdef PDLP_DEBUG_MODE
  std::cout << "    Compute primal gradient:" << std::endl;
#endif

  // for QP add problem.objective_matrix * primal_solution as well
  // c - A^T*y (copy c to primal_gradient for correct writing of result)
  raft::copy(duality_gap.primal_gradient_.data(),
             problem_ptr->objective_coefficients.data(),
             primal_size_h_,
             stream_view_);

  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       reusable_device_scalar_value_neg_1_.data(),
                                                       cusparse_view.A_T,
                                                       cusparse_view.dual_solution,
                                                       reusable_device_scalar_value_1_.data(),
                                                       cusparse_view.primal_gradient,
                                                       CUSPARSE_SPMV_CSR_ALG2,
                                                       (f_t*)cusparse_view.buffer_transpose.data(),
                                                       stream_view_));
}

template <typename i_t, typename f_t>
__global__ void compute_subgradient_kernel(
  const typename pdlp_restart_strategy_t<i_t, f_t>::view_t restart_strategy_view,
  const typename problem_t<i_t, f_t>::view_t op_problem_view,
  const typename localized_duality_gap_container_t<i_t, f_t>::view_t duality_gap_view,
  f_t* subgradient)
{
  i_t id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= duality_gap_view.dual_size) { return; }

  f_t lower          = op_problem_view.constraint_lower_bounds[id];
  f_t upper          = op_problem_view.constraint_upper_bounds[id];
  f_t primal_product = duality_gap_view.dual_gradient[id];
  f_t dual_solution  = duality_gap_view.dual_solution[id];

  f_t subgradient_coefficient;

  if (dual_solution < f_t(0)) {
    subgradient_coefficient = upper;
  } else if (dual_solution > f_t(0)) {
    subgradient_coefficient = lower;
  } else if (!isfinite(upper) && !isfinite(lower)) {
    subgradient_coefficient = f_t(0);
  } else if (!isfinite(upper) && isfinite(lower)) {
    subgradient_coefficient = lower;
  } else if (isfinite(upper) && !isfinite(lower)) {
    subgradient_coefficient = upper;
  } else {
    if (primal_product < lower) {
      subgradient_coefficient = lower;
    } else if (primal_product > upper) {
      subgradient_coefficient = upper;
    } else {
      subgradient_coefficient = primal_product;
    }
  }

  subgradient[id] = subgradient_coefficient;
}

template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::compute_dual_gradient(
  localized_duality_gap_container_t<i_t, f_t>& duality_gap,
  cusparse_view_t<i_t, f_t>& cusparse_view,
  rmm::device_uvector<f_t>& tmp_dual)
{
  raft::common::nvtx::range fun_scope("compute_dual_gradient");
#ifdef PDLP_DEBUG_MODE
  std::cout << "    Compute dual gradient:" << std::endl;
#endif

  // b - A*x
  // is changed with the introduction of constraint upper and lower bounds

  // gradient constains primal_product
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       reusable_device_scalar_value_1_.data(),
                                       cusparse_view.A,
                                       cusparse_view.primal_solution,
                                       reusable_device_scalar_value_0_.data(),
                                       cusparse_view.dual_gradient,
                                       CUSPARSE_SPMV_CSR_ALG2,
                                       (f_t*)cusparse_view.buffer_non_transpose.data(),
                                       stream_view_));

  // tmp_dual will contain the subgradient
  i_t number_of_blocks = dual_size_h_ / block_size;
  if (dual_size_h_ % block_size) number_of_blocks++;
  i_t number_of_threads = std::min(dual_size_h_, block_size);
  compute_subgradient_kernel<i_t, f_t><<<number_of_blocks, number_of_threads, 0, stream_view_>>>(
    this->view(), problem_ptr->view(), duality_gap.view(), tmp_dual.data());

  // dual gradient = subgradient - primal_product (tmp_dual-dual_gradient)
  raft::linalg::eltwiseSub(duality_gap.dual_gradient_.data(),
                           tmp_dual.data(),
                           duality_gap.dual_gradient_.data(),
                           dual_size_h_,
                           stream_view_);
}

template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::compute_lagrangian_value(
  localized_duality_gap_container_t<i_t, f_t>& duality_gap,
  cusparse_view_t<i_t, f_t>& cusparse_view,
  rmm::device_uvector<f_t>& tmp_primal,
  rmm::device_uvector<f_t>& tmp_dual)
{
  raft::common::nvtx::range fun_scope("compute_lagrangian_value");
#ifdef PDLP_DEBUG_MODE
  std::cout << "    Compute lagrangian value:" << std::endl;
#endif
  // if QP
  //  0.5 * dot(primal_solution, problem.objective_matrix * primal_solution) +
  //  dot(primal_solution, problem.objective_vector) -
  //  dot(primal_solution, problem.constraint_matrix' * dual_solution) +
  //  dot(dual_solution, dual_gradient+primal_product) +
  //  problem.objective_constant

  // when lp first term is irrelevant

  // second term
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
                                                  primal_size_h_,
                                                  duality_gap.primal_solution_.data(),
                                                  primal_stride,
                                                  problem_ptr->objective_coefficients.data(),
                                                  primal_stride,
                                                  reusable_device_scalar_1_.data(),
                                                  stream_view_));

  // third term, let beta be 0 to not add what is in tmp_primal, compute it and compute dot
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       reusable_device_scalar_value_1_.data(),
                                                       cusparse_view.A_T,
                                                       cusparse_view.dual_solution,
                                                       reusable_device_scalar_value_0_.data(),
                                                       cusparse_view.tmp_primal,
                                                       CUSPARSE_SPMV_CSR_ALG2,
                                                       (f_t*)cusparse_view.buffer_transpose.data(),
                                                       stream_view_));

  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
                                                  primal_size_h_,
                                                  duality_gap.primal_solution_.data(),
                                                  primal_stride,
                                                  tmp_primal.data(),
                                                  primal_stride,
                                                  reusable_device_scalar_2_.data(),
                                                  stream_view_));

  // fourth term //tmp_dual still contains subgradient from the dual_gradient computation
  reusable_device_scalar_3_.set_value_to_zero_async(stream_view_);
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
                                                  dual_size_h_,
                                                  duality_gap.dual_solution_.data(),
                                                  dual_stride,
                                                  tmp_dual.data(),
                                                  dual_stride,
                                                  reusable_device_scalar_3_.data(),
                                                  stream_view_));

  // subtract third term from second up
  raft::linalg::eltwiseSub(reusable_device_scalar_1_.data(),
                           reusable_device_scalar_1_.data(),
                           reusable_device_scalar_2_.data(),
                           1,
                           stream_view_);
  raft::linalg::eltwiseAdd(duality_gap.lagrangian_value_.data(),
                           reusable_device_scalar_1_.data(),
                           reusable_device_scalar_3_.data(),
                           1,
                           stream_view_);
}

template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::reset_internal()
{
  candidate_is_avg_.set_value_to_zero_async(stream_view_);
  restart_triggered_.set_value_to_zero_async(stream_view_);
}

template <typename i_t, typename f_t>
typename pdlp_restart_strategy_t<i_t, f_t>::view_t pdlp_restart_strategy_t<i_t, f_t>::view()
{
  pdlp_restart_strategy_t<i_t, f_t>::view_t v{};
  v.primal_size                         = primal_size_h_;
  v.dual_size                           = dual_size_h_;
  v.transformed_constraint_lower_bounds = raft::device_span<f_t>{
    transformed_constraint_lower_bounds_.data(), transformed_constraint_lower_bounds_.size()};
  v.transformed_constraint_upper_bounds = raft::device_span<f_t>{
    transformed_constraint_upper_bounds_.data(), transformed_constraint_upper_bounds_.size()};
  v.last_restart_length = last_restart_length_;

  v.weights = raft::device_span<f_t>{weights_.data(), weights_.size()};

  v.candidate_is_avg  = candidate_is_avg_.data();
  v.restart_triggered = restart_triggered_.data();

  v.gap_reduction_ratio_last_trial = gap_reduction_ratio_last_trial_.data();

  v.center_point     = raft::device_span<f_t>{center_point_.data(), center_point_.size()};
  v.objective_vector = raft::device_span<f_t>{objective_vector_.data(), objective_vector_.size()};
  v.direction_full   = raft::device_span<f_t>{direction_full_.data(), direction_full_.size()};
  v.threshold        = raft::device_span<f_t>{threshold_.data(), threshold_.size()};
  v.lower_bound      = raft::device_span<f_t>{lower_bound_.data(), lower_bound_.size()};
  v.upper_bound      = raft::device_span<f_t>{upper_bound_.data(), upper_bound_.size()};
  v.test_point       = raft::device_span<f_t>{test_point_.data(), test_point_.size()};

  v.target_threshold    = target_threshold_.data();
  v.low_radius_squared  = low_radius_squared_.data();
  v.high_radius_squared = high_radius_squared_.data();
  v.test_radius_squared = test_radius_squared_.data();

  v.testing_range_low  = testing_range_low_.data();
  v.testing_range_high = testing_range_high_.data();

  v.shared_live_kernel_accumulator = raft::device_span<f_t>{shared_live_kernel_accumulator_.data(),
                                                            shared_live_kernel_accumulator_.size()};

  return v;
}

template <typename i_t, typename f_t>
i_t pdlp_restart_strategy_t<i_t, f_t>::get_iterations_since_last_restart() const
{
  return weighted_average_solution_.get_iterations_since_last_restart();
}

template <typename i_t, typename f_t>
void pdlp_restart_strategy_t<i_t, f_t>::set_last_restart_was_average(bool value)
{
  last_restart_was_average_ = value;
}

template <typename i_t, typename f_t>
bool pdlp_restart_strategy_t<i_t, f_t>::get_last_restart_was_average() const
{
  return last_restart_was_average_;
}

#define INSTANTIATE(F_TYPE)                                                                     \
  template class pdlp_restart_strategy_t<int, F_TYPE>;                                          \
                                                                                                \
  template __global__ void compute_distance_traveled_last_restart_kernel<int, F_TYPE>(          \
    const typename localized_duality_gap_container_t<int, F_TYPE>::view_t duality_gap_view,     \
    const F_TYPE* primal_weight,                                                                \
    F_TYPE* distance_traveled);                                                                 \
                                                                                                \
  template __global__ void pick_restart_candidate_kernel<int, F_TYPE>(                          \
    const typename localized_duality_gap_container_t<int, F_TYPE>::view_t avg_duality_gap_view, \
    const typename localized_duality_gap_container_t<int, F_TYPE>::view_t                       \
      current_duality_gap_view,                                                                 \
    typename pdlp_restart_strategy_t<int, F_TYPE>::view_t restart_strategy_view);               \
                                                                                                \
  template __global__ void adaptive_restart_triggered<int, F_TYPE>(                             \
    const typename localized_duality_gap_container_t<int, F_TYPE>::view_t                       \
      candidate_duality_gap_view,                                                               \
    const typename localized_duality_gap_container_t<int, F_TYPE>::view_t                       \
      last_restart_duality_gap_view,                                                            \
    typename pdlp_restart_strategy_t<int, F_TYPE>::view_t restart_strategy_view);               \
                                                                                                \
  template __global__ void solve_bound_constrained_trust_region_kernel<int, F_TYPE, 128>(       \
    typename pdlp_restart_strategy_t<int, F_TYPE>::view_t restart_strategy_view,                \
    typename problem_t<int, F_TYPE>::view_t op_problem_view,                                    \
    int* testing_range_low,                                                                     \
    int* testing_range_high,                                                                    \
    F_TYPE* test_radius_squared,                                                                \
    F_TYPE* low_radius_squared,                                                                 \
    F_TYPE* high_radius_squared,                                                                \
    const F_TYPE* target_radius);                                                               \
                                                                                                \
  template __global__ void target_threshold_determination_kernel<int, F_TYPE>(                  \
    typename pdlp_restart_strategy_t<int, F_TYPE>::view_t restart_strategy_view,                \
    const F_TYPE* target_radius,                                                                \
    const F_TYPE* max_primal_threshold,                                                         \
    const F_TYPE* max_dual_threshold);                                                          \
                                                                                                \
  template __global__ void compute_normalized_gaps_kernel<int, F_TYPE>(                         \
    typename localized_duality_gap_container_t<int, F_TYPE>::view_t avg_duality_gap_view,       \
    typename localized_duality_gap_container_t<int, F_TYPE>::view_t current_duality_gap_view);  \
                                                                                                \
  template __global__ void compute_new_primal_weight_kernel<int, F_TYPE>(                       \
    const typename localized_duality_gap_container_t<int, F_TYPE>::view_t duality_gap_view,     \
    F_TYPE* primal_weight,                                                                      \
    const F_TYPE* step_size,                                                                    \
    F_TYPE* primal_step_size,                                                                   \
    F_TYPE* dual_step_size);                                                                    \
                                                                                                \
  template __global__ void compute_subgradient_kernel<int, F_TYPE>(                             \
    const typename pdlp_restart_strategy_t<int, F_TYPE>::view_t restart_strategy_view,          \
    const typename problem_t<int, F_TYPE>::view_t op_problem_view,                              \
    const typename localized_duality_gap_container_t<int, F_TYPE>::view_t duality_gap_view,     \
    F_TYPE* primal_product);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

}  // namespace cuopt::linear_programming::detail
