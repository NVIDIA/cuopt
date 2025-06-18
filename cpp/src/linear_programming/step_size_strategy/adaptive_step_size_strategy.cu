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

#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <linear_programming/pdlp_constants.hpp>
#include <linear_programming/step_size_strategy/adaptive_step_size_strategy.hpp>
#include <mip/mip_constants.hpp>
#include <utilities/unique_pinned_ptr.hpp>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/common/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <limits>

namespace cuopt::linear_programming::detail {

constexpr int parallel_stream_computation = 2;

template <typename i_t, typename f_t>
adaptive_step_size_strategy_t<i_t, f_t>::adaptive_step_size_strategy_t(
  raft::handle_t const* handle_ptr,
  rmm::device_scalar<f_t>* primal_weight,
  rmm::device_scalar<f_t>* step_size)
  : stream_pool_(parallel_stream_computation),
    dot_delta_X_(cudaEventDisableTiming),
    dot_delta_Y_(cudaEventDisableTiming),
    deltas_are_done_(cudaEventDisableTiming),
    handle_ptr_(handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    primal_weight_(primal_weight),
    step_size_(step_size),
    interaction_{stream_view_},
    movement_{stream_view_},
    norm_squared_delta_primal_{stream_view_},
    norm_squared_delta_dual_{stream_view_},
    reusable_device_scalar_value_1_{f_t(1.0), stream_view_},
    reusable_device_scalar_value_0_{f_t(0.0), stream_view_},
    graph(stream_view_)
{
  valid_step_size_ = make_unique_cuda_host_pinned<i_t>();
}

void set_adaptive_step_size_hyper_parameters(rmm::cuda_stream_view stream_view)
{
  RAFT_CUDA_TRY(cudaMemcpyToSymbolAsync(pdlp_hyper_params::default_reduction_exponent,
                                        &pdlp_hyper_params::host_default_reduction_exponent,
                                        sizeof(double),
                                        0,
                                        cudaMemcpyHostToDevice,
                                        stream_view));
  RAFT_CUDA_TRY(cudaMemcpyToSymbolAsync(pdlp_hyper_params::default_growth_exponent,
                                        &pdlp_hyper_params::host_default_growth_exponent,
                                        sizeof(double),
                                        0,
                                        cudaMemcpyHostToDevice,
                                        stream_view));
  RAFT_CUDA_TRY(cudaMemcpyToSymbolAsync(pdlp_hyper_params::primal_distance_smoothing,
                                        &pdlp_hyper_params::host_primal_distance_smoothing,
                                        sizeof(double),
                                        0,
                                        cudaMemcpyHostToDevice,
                                        stream_view));
  RAFT_CUDA_TRY(cudaMemcpyToSymbolAsync(pdlp_hyper_params::dual_distance_smoothing,
                                        &pdlp_hyper_params::host_dual_distance_smoothing,
                                        sizeof(double),
                                        0,
                                        cudaMemcpyHostToDevice,
                                        stream_view));
}

template <typename i_t, typename f_t>
__global__ void compute_step_sizes_from_movement_and_interaction(
  typename adaptive_step_size_strategy_t<i_t, f_t>::view_t step_size_strategy_view,
  f_t* primal_step_size,
  f_t* dual_step_size,
  i_t* pdhg_iteration)
{
  if (threadIdx.x + blockIdx.x * blockDim.x > 0) { return; }

  f_t primal_weight_ = *step_size_strategy_view.primal_weight;

  f_t movement = pdlp_hyper_params::primal_distance_smoothing * primal_weight_ *
                   *step_size_strategy_view.norm_squared_delta_primal +
                 (pdlp_hyper_params::dual_distance_smoothing / primal_weight_) *
                   *step_size_strategy_view.norm_squared_delta_dual;

#ifdef PDLP_DEBUG_MODE
  printf("-compute_step_sizes_from_movement_and_interaction:\n");
#endif
  if (movement <= 0 || movement >= divergent_movement<f_t>) {
    *step_size_strategy_view.valid_step_size = -1;
#ifdef PDLP_DEBUG_MODE
    printf("  Movement is %lf. Done or numerical error has happened\n", movement);
#endif
    return;
  }

  f_t interaction_ = raft::abs(*step_size_strategy_view.interaction);
  f_t step_size_   = *step_size_strategy_view.step_size;

  // Increase PDHG iteration
  *pdhg_iteration += 1;

  f_t iteration_coefficient_ = *pdhg_iteration;

  // proof of thm 1 requires movement / step_size >= interaction.
  f_t step_size_limit = interaction_ > 0.0 ? movement / interaction_ : raft::myInf<f_t>();

#ifdef PDLP_DEBUG_MODE
  printf("    interaction_=%lf movement=%lf\n", interaction_, movement);
  printf("    step_size_=%lf step_size_limit=%lf pdhg_iteration=%d iteration_coefficient_=%lf\n",
         step_size_,
         step_size_limit,
         *pdhg_iteration,
         iteration_coefficient_);
#endif

  if (step_size_ <= step_size_limit) {
    *step_size_strategy_view.valid_step_size = 1;

#ifdef PDLP_DEBUG_MODE
    printf("    Step size is smaller\n");
#endif
  }

  // The step size was too large and therefore we now compute the next stepsize to test out.
  // We have two candidates of which we take the smaller to retry taking a step
  const f_t potential_new_step_size_1 =
    (f_t(1.0) - raft::pow<f_t>(iteration_coefficient_ + f_t(1.0),
                               -pdlp_hyper_params::default_reduction_exponent)) *
    step_size_limit;
  const f_t potential_new_step_size_2 =
    (f_t(1.0) + raft::pow<f_t>(iteration_coefficient_ + f_t(1.0),
                               -pdlp_hyper_params::default_growth_exponent)) *
    step_size_;

#ifdef PDLP_DEBUG_MODE
  printf(
    "Compute adaptative step size: iteration_coefficient_=%lf "
    "-pdlp_hyper_params::default_reduction_exponent=%lf step_size_limit=%lf\n",
    iteration_coefficient_,
    -pdlp_hyper_params::default_reduction_exponent,
    step_size_limit);
  printf(
    "Compute adaptative step size: iteration_coefficient_=%lf "
    "-pdlp_hyper_params::default_growth_exponent=%lf step_size_=%lf\n",
    iteration_coefficient_,
    -pdlp_hyper_params::default_growth_exponent,
    step_size_);
  printf(
    "Compute adaptative step size: potential_new_step_size_1=%lf potential_new_step_size_2=%lf\n",
    potential_new_step_size_1,
    potential_new_step_size_2);
#endif

  step_size_ = raft::min<f_t>(potential_new_step_size_1, potential_new_step_size_2);

#ifdef PDLP_DEBUG_MODE
  printf("Compute adaptative step size: min_step_size_picked=%lf\n", step_size_);
#endif

  *primal_step_size = step_size_ / primal_weight_;
  *dual_step_size   = step_size_ * primal_weight_;

  *step_size_strategy_view.step_size = step_size_;
  cuopt_assert(!isnan(step_size_), "step size can't be nan");
  cuopt_assert(!isinf(step_size_), "step size can't be inf");
}

template <typename i_t, typename f_t>
i_t adaptive_step_size_strategy_t<i_t, f_t>::get_valid_step_size() const
{
  return *valid_step_size_;
}

template <typename i_t, typename f_t>
void adaptive_step_size_strategy_t<i_t, f_t>::set_valid_step_size(i_t valid)
{
  *valid_step_size_ = valid;
}

template <typename i_t, typename f_t>
void adaptive_step_size_strategy_t<i_t, f_t>::compute_step_sizes(
  pdhg_solver_t<i_t, f_t>& pdhg_solver,
  rmm::device_scalar<f_t>& primal_step_size,
  rmm::device_scalar<f_t>& dual_step_size,
  i_t total_pdlp_iterations)
{
  raft::common::nvtx::range fun_scope("compute_step_sizes");

  if (!graph.is_initialized(total_pdlp_iterations)) {
    graph.start_capture(total_pdlp_iterations);

    // compute numerator and deminator of n_lim
    compute_interaction_and_movement(pdhg_solver.get_primal_tmp_resource(),
                                     pdhg_solver.get_cusparse_view(),
                                     pdhg_solver.get_saddle_point_state());
    // Compute n_lim, n_next and decide if step size is valid
    compute_step_sizes_from_movement_and_interaction<i_t, f_t>
      <<<1, 1, 0, stream_view_>>>(this->view(),
                                  primal_step_size.data(),
                                  dual_step_size.data(),
                                  pdhg_solver.get_d_total_pdhg_iterations().data());
    graph.end_capture(total_pdlp_iterations);
  }
  graph.launch(total_pdlp_iterations);
  // Steam sync so that next call can see modification made to host var valid_step_size
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));
}

template <typename i_t, typename f_t>
void adaptive_step_size_strategy_t<i_t, f_t>::compute_interaction_and_movement(
  rmm::device_uvector<f_t>& tmp_primal,
  cusparse_view_t<i_t, f_t>& cusparse_view,
  saddle_point_state_t<i_t, f_t>& current_saddle_point_state)
{
  // QP would need this:
  // if iszero(problem.objective_matrix)
  //   primal_objective_interaction = 0.0
  // else
  //   primal_objective_interaction =
  //     0.5 * (delta_primal' * problem.objective_matrix * delta_primal)
  // end
  // would need to add abs(primal_objective_interaction) to interaction as well

  /*
    Here we compute : movement / interaction

    Movement: ||(x' - x), (y' - y)||²
    Interaction: (y' - y)_t . A @ (x' - x)

    Deltas x & y were computed during pdhg step

    We will compute in parallel (parallel cuda graph):
    ||(x' - x)||
    ||(y' - y)||
    (y' - y)_t . A @ (x' - x)

    And finally merge the results
  */

  // We need to make sure both dot products happens after previous operations (next_primal/dual)
  // Thus, we add another node in the main stream before starting the SpMVs

  deltas_are_done_.record(stream_view_);

  // primal_dual_interaction computation => we purposly diverge from the paper (delta_y . (A @ x' -
  // A@x)) to save one SpMV
  // Instead we do: delta_x . (A_t @ y' - A_t @ y)
  // A_t @ y has already been computed during compute next_primal
  // A_t @ y' is computed here each time but, if a valid step is found, A @ y'
  // becomes A @ y for next step (as what was y' becomes y if valid for next step). This saves the
  // first A @ y SpMV in the compute_next_primal of next PDHG step

  // Compute A_t @ (y' - y) = A_t @ y' - 1 * current_AtY

  // First compute Ay' to be reused as Ay in next PDHG iteration (if found step size if valid)
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       reusable_device_scalar_value_1_.data(),  // alpha
                                       cusparse_view.A_T,
                                       cusparse_view.potential_next_dual_solution,
                                       reusable_device_scalar_value_0_.data(),  // beta
                                       cusparse_view.next_AtY,
                                       CUSPARSE_SPMV_CSR_ALG2,
                                       (f_t*)cusparse_view.buffer_transpose.data(),
                                       stream_view_));

  // Compute Ay' - Ay = next_Aty - current_Aty
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(current_saddle_point_state.get_next_AtY().data(),
                          current_saddle_point_state.get_current_AtY().data()),
    tmp_primal.data(),
    current_saddle_point_state.get_primal_size(),
    raft::sub_op(),
    stream_view_);

  // compute interaction (x'-x) . (A(y'-y))
  RAFT_CUBLAS_TRY(
    raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
                                    current_saddle_point_state.get_primal_size(),
                                    tmp_primal.data(),
                                    primal_stride,
                                    current_saddle_point_state.get_delta_primal().data(),
                                    primal_stride,
                                    interaction_.data(),
                                    stream_view_));

  // Compute movement
  //  compute euclidean norm squared which is
  //  same as taking the dot product with itself
  //    movement = 0.5 * solver_state.primal_weight
  //    * norm(delta_primal) ^
  //               2 + (0.5 /
  //               solver_state.primal_weight) *
  //               norm(delta_dual) ^ 2;
  deltas_are_done_.stream_wait(stream_pool_.get_stream(0));
  RAFT_CUBLAS_TRY(
    raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
                                    current_saddle_point_state.get_primal_size(),
                                    current_saddle_point_state.get_delta_primal().data(),
                                    primal_stride,
                                    current_saddle_point_state.get_delta_primal().data(),
                                    primal_stride,
                                    norm_squared_delta_primal_.data(),
                                    stream_pool_.get_stream(0)));
  dot_delta_X_.record(stream_pool_.get_stream(0));

  deltas_are_done_.stream_wait(stream_pool_.get_stream(1));
  RAFT_CUBLAS_TRY(
    raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
                                    current_saddle_point_state.get_dual_size(),
                                    current_saddle_point_state.get_delta_dual().data(),
                                    dual_stride,
                                    current_saddle_point_state.get_delta_dual().data(),
                                    dual_stride,
                                    norm_squared_delta_dual_.data(),
                                    stream_pool_.get_stream(1)));
  dot_delta_Y_.record(stream_pool_.get_stream(1));

  // Wait on main stream for both dot to be done before launching the next kernel
  dot_delta_X_.stream_wait(stream_view_);
  dot_delta_Y_.stream_wait(stream_view_);
}

template <typename i_t, typename f_t>
__global__ void compute_actual_stepsizes(
  const typename adaptive_step_size_strategy_t<i_t, f_t>::view_t step_size_strategy_view,
  f_t* primal_step_size,
  f_t* dual_step_size)
{
  if (threadIdx.x + blockIdx.x * blockDim.x > 0) { return; }
  f_t step_size_     = *step_size_strategy_view.step_size;
  f_t primal_weight_ = *step_size_strategy_view.primal_weight;

  *primal_step_size = step_size_ / primal_weight_;
  *dual_step_size   = step_size_ * primal_weight_;
}

template <typename i_t, typename f_t>
void adaptive_step_size_strategy_t<i_t, f_t>::get_primal_and_dual_stepsizes(
  rmm::device_scalar<f_t>& primal_step_size, rmm::device_scalar<f_t>& dual_step_size)
{
  compute_actual_stepsizes<i_t, f_t>
    <<<1, 1, 0, stream_view_>>>(this->view(), primal_step_size.data(), dual_step_size.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename i_t, typename f_t>
typename adaptive_step_size_strategy_t<i_t, f_t>::view_t
adaptive_step_size_strategy_t<i_t, f_t>::view()
{
  adaptive_step_size_strategy_t<i_t, f_t>::view_t v{};

  v.primal_weight   = primal_weight_->data();
  v.step_size       = step_size_->data();
  v.valid_step_size = valid_step_size_.get();

  v.interaction = interaction_.data();
  v.movement    = movement_.data();

  v.norm_squared_delta_primal = norm_squared_delta_primal_.data();
  v.norm_squared_delta_dual   = norm_squared_delta_dual_.data();

  return v;
}

#define INSTANTIATE(F_TYPE)                                                                    \
  template class adaptive_step_size_strategy_t<int, F_TYPE>;                                   \
  template __global__ void compute_actual_stepsizes<int, F_TYPE>(                              \
    const typename adaptive_step_size_strategy_t<int, F_TYPE>::view_t step_size_strategy_view, \
    F_TYPE* primal_step_size,                                                                  \
    F_TYPE* dual_step_size);                                                                   \
                                                                                               \
  template __global__ void compute_step_sizes_from_movement_and_interaction<int, F_TYPE>(      \
    typename adaptive_step_size_strategy_t<int, F_TYPE>::view_t step_size_strategy_view,       \
    F_TYPE * primal_step_size,                                                                 \
    F_TYPE * dual_step_size,                                                                   \
    int* pdhg_iteration);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

}  // namespace cuopt::linear_programming::detail
