/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <linear_programming/termination_strategy/infeasibility_information.hpp>
#include <linear_programming/utils.cuh>
#include <mip/mip_constants.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/common/nvtx.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/ternary_op.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
infeasibility_information_t<i_t, f_t>::infeasibility_information_t(
  raft::handle_t const* handle_ptr,
  problem_t<i_t, f_t>& op_problem,
  cusparse_view_t<i_t, f_t>& cusparse_view,
  i_t primal_size,
  i_t dual_size,
  bool infeasibility_detection)
  : handle_ptr_(handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    primal_size_h_(primal_size),
    dual_size_h_(dual_size),
    problem_ptr(&op_problem),
    op_problem_cusparse_view_(cusparse_view),
    primal_ray_inf_norm_{0.0, stream_view_},
    primal_ray_inf_norm_inverse_{stream_view_},
    neg_primal_ray_inf_norm_inverse_{stream_view_},
    primal_ray_max_violation_{stream_view_},
    max_primal_ray_infeasibility_{0.0, stream_view_},
    primal_ray_linear_objective_{0.0, stream_view_},
    dual_ray_inf_norm_{0.0, stream_view_},
    max_dual_ray_infeasibility_{0.0, stream_view_},
    dual_ray_linear_objective_{0.0, stream_view_},
    reduced_cost_dual_objective_{0.0, stream_view_},
    reduced_cost_inf_norm_{0.0, stream_view_},
    // If infeasibility_detection is off, no need to allocate all those
    homogenous_primal_residual_{(!infeasibility_detection) ? 0 : static_cast<size_t>(dual_size_h_),
                                stream_view_},
    homogenous_dual_residual_{(!infeasibility_detection) ? 0 : static_cast<size_t>(primal_size_h_),
                              stream_view_},
    reduced_cost_{(!infeasibility_detection) ? 0 : static_cast<size_t>(primal_size_h_),
                  stream_view_},
    bound_value_{
      (!infeasibility_detection) ? 0 : static_cast<size_t>(std::max(primal_size_h_, dual_size_h_)),
      stream_view_},
    homogenous_dual_lower_bounds_{
      (!infeasibility_detection) ? 0 : static_cast<size_t>(dual_size_h_), stream_view_},
    homogenous_dual_upper_bounds_{
      (!infeasibility_detection) ? 0 : static_cast<size_t>(dual_size_h_), stream_view_},
    reusable_device_scalar_value_1_{1.0, stream_view_},
    reusable_device_scalar_value_0_{0.0, stream_view_},
    reusable_device_scalar_value_neg_1_{-1.0, stream_view_}
{
  if (infeasibility_detection) {
    RAFT_CUDA_TRY(cudaMemsetAsync(homogenous_primal_residual_.data(),
                                  0.0,
                                  sizeof(f_t) * homogenous_primal_residual_.size(),
                                  stream_view_));
    RAFT_CUDA_TRY(cudaMemsetAsync(homogenous_dual_residual_.data(),
                                  0.0,
                                  sizeof(f_t) * homogenous_dual_residual_.size(),
                                  stream_view_));

    // variable bounds in the homogenous primal are 0.0 if the original bound was finite, and
    // otherwise it is -inf for lower bounds and inf for upper bounds
    raft::linalg::unaryOp(homogenous_dual_lower_bounds_.data(),
                          problem_ptr->constraint_lower_bounds.data(),
                          dual_size_h_,
                          zero_if_is_finite<f_t>(),
                          stream_view_);
    raft::linalg::unaryOp(homogenous_dual_upper_bounds_.data(),
                          problem_ptr->constraint_upper_bounds.data(),
                          dual_size_h_,
                          zero_if_is_finite<f_t>(),
                          stream_view_);

    void* d_temp_storage        = NULL;
    size_t temp_storage_bytes_1 = 0;
    cub::DeviceReduce::Sum(d_temp_storage,
                           temp_storage_bytes_1,
                           bound_value_.begin(),
                           dual_ray_linear_objective_.data(),
                           dual_size_h_,
                           stream_view_);

    size_t temp_storage_bytes_2 = 0;
    cub::DeviceReduce::Sum(d_temp_storage,
                           temp_storage_bytes_2,
                           bound_value_.begin(),
                           reduced_cost_dual_objective_.data(),
                           primal_size_h_,
                           stream_view_);

    size_of_buffer_       = std::max({temp_storage_bytes_1, temp_storage_bytes_2});
    this->rmm_tmp_buffer_ = rmm::device_buffer{size_of_buffer_, stream_view_};
  }
}

template <typename i_t, typename f_t>
__global__ void compute_remaining_stats_kernel(
  typename infeasibility_information_t<i_t, f_t>::view_t infeasibility_information_view)
{
  if (threadIdx.x + blockIdx.x * blockDim.x > 0) { return; }
#ifdef PDLP_DEBUG_MODE
  printf("-compute_remaining_stats_kernel:\n");
#endif
  f_t scaling_factor = raft::max(*infeasibility_information_view.dual_ray_inf_norm,
                                 *infeasibility_information_view.reduced_cost_inf_norm);
#ifdef PDLP_DEBUG_MODE
  printf("    dual_ray_inf_norm=%lf reduced_cost_inf_norm=%lf scaling_factor=%lf\n",
         *infeasibility_information_view.dual_ray_inf_norm,
         *infeasibility_information_view.reduced_cost_inf_norm,
         scaling_factor);
#endif

#ifdef PDLP_DEBUG_MODE
  printf("    Before max_dual_ray_infeasibility=%lf dual_ray_linear_objective=%lf\n",
         *infeasibility_information_view.max_dual_ray_infeasibility,
         *infeasibility_information_view.dual_ray_linear_objective);
#endif
  if (scaling_factor < 0.0 || scaling_factor > 0.0) {
    *infeasibility_information_view.max_dual_ray_infeasibility =
      *infeasibility_information_view.max_dual_ray_infeasibility / scaling_factor;
    *infeasibility_information_view.dual_ray_linear_objective =
      *infeasibility_information_view.dual_ray_linear_objective / scaling_factor;
  } else {
    *infeasibility_information_view.max_dual_ray_infeasibility = 0.0;
    *infeasibility_information_view.dual_ray_linear_objective  = 0.0;
  }
#ifdef PDLP_DEBUG_MODE
  printf("    After max_dual_ray_infeasibility=%lf dual_ray_linear_objective=%lf\n",
         *infeasibility_information_view.max_dual_ray_infeasibility,
         *infeasibility_information_view.dual_ray_linear_objective);
  printf("    primal_ray_inf_norm=%lf\n", *infeasibility_information_view.primal_ray_inf_norm);
#endif
  // Update primal max ray infeasibility
  if (*infeasibility_information_view.primal_ray_inf_norm > f_t(0.0)) {
    *infeasibility_information_view.max_primal_ray_infeasibility =
      raft::max(*infeasibility_information_view.max_primal_ray_infeasibility,
                *infeasibility_information_view.primal_ray_max_violation) /
      *infeasibility_information_view.primal_ray_inf_norm;
  } else {
    *infeasibility_information_view.max_primal_ray_infeasibility = f_t(0.0);
    *infeasibility_information_view.primal_ray_linear_objective  = f_t(0.0);
  }
#ifdef PDLP_DEBUG_MODE
  printf("    max_primal_ray_infeasibility=%lf primal_ray_linear_objective=%lf\n",
         *infeasibility_information_view.max_primal_ray_infeasibility,
         *infeasibility_information_view.primal_ray_linear_objective);
#endif
}

template <typename i_t, typename f_t>
void infeasibility_information_t<i_t, f_t>::compute_infeasibility_information(
  pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
  rmm::device_uvector<f_t>& primal_ray,
  rmm::device_uvector<f_t>& dual_ray)
{
  raft::common::nvtx::range fun_scope("compute_infeasibility_information");

  my_inf_norm(primal_ray, primal_ray_inf_norm_, handle_ptr_);

  raft::linalg::eltwiseDivideCheckZero(primal_ray_inf_norm_inverse_.data(),
                                       reusable_device_scalar_value_1_.data(),
                                       primal_ray_inf_norm_.data(),
                                       1,
                                       stream_view_);
  raft::linalg::eltwiseMultiply(neg_primal_ray_inf_norm_inverse_.data(),
                                primal_ray_inf_norm_inverse_.data(),
                                reusable_device_scalar_value_neg_1_.data(),
                                1,
                                stream_view_);

  compute_homogenous_primal_residual(op_problem_cusparse_view_,
                                     current_pdhg_solver.get_dual_tmp_resource());
  compute_max_violation(primal_ray);
  compute_homogenous_primal_objective(primal_ray);
  my_inf_norm(homogenous_primal_residual_, max_primal_ray_infeasibility_, handle_ptr_);

  // QP would need this
  // primal_ray_quadratic_norm = norm(problem.objective_matrix * primal_ray_estimate, Inf)

  compute_homogenous_dual_residual(
    op_problem_cusparse_view_, current_pdhg_solver.get_primal_tmp_resource(), primal_ray);
  compute_homogenous_dual_objective(dual_ray);

  my_inf_norm(homogenous_dual_residual_, max_dual_ray_infeasibility_, handle_ptr_);
  my_inf_norm(dual_ray, dual_ray_inf_norm_, handle_ptr_);
  my_inf_norm(reduced_cost_, reduced_cost_inf_norm_, handle_ptr_);

  compute_remaining_stats_kernel<i_t, f_t><<<1, 1, 0, stream_view_>>>(this->view());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // reset for next round
  RAFT_CUDA_TRY(cudaMemsetAsync(homogenous_primal_residual_.data(),
                                0.0,
                                sizeof(f_t) * homogenous_primal_residual_.size(),
                                stream_view_));
  RAFT_CUDA_TRY(cudaMemsetAsync(
    homogenous_dual_residual_.data(), 0.0, sizeof(f_t) * homogenous_dual_residual_.size()));
}

template <typename i_t, typename f_t>
void infeasibility_information_t<i_t, f_t>::compute_homogenous_primal_residual(
  cusparse_view_t<i_t, f_t>& cusparse_view, rmm::device_uvector<f_t>& tmp_dual)
{
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       reusable_device_scalar_value_1_.data(),
                                       cusparse_view.A,
                                       cusparse_view.primal_solution,
                                       reusable_device_scalar_value_0_.data(),
                                       cusparse_view.tmp_dual,
                                       CUSPARSE_SPMV_CSR_ALG2,
                                       (f_t*)cusparse_view.buffer_non_transpose.data(),
                                       stream_view_));

  raft::linalg::ternaryOp(homogenous_primal_residual_.data(),
                          tmp_dual.data(),
                          homogenous_dual_lower_bounds_.data(),
                          homogenous_dual_upper_bounds_.data(),
                          dual_size_h_,
                          violation<f_t>(),
                          stream_view_);
}

template <typename i_t, typename f_t>
void infeasibility_information_t<i_t, f_t>::compute_max_violation(
  rmm::device_uvector<f_t>& primal_ray)
{
  // Convert raw pointer to thrust::device_ptr to write directly device side through reduce
  thrust::device_ptr<f_t> primal_ray_max_violation(primal_ray_max_violation_.data());

  using f_t2                = typename type_2<f_t>::type;
  *primal_ray_max_violation = thrust::transform_reduce(
    handle_ptr_->get_thrust_policy(),
    thrust::make_zip_iterator(
      thrust::make_tuple(primal_ray.data(), problem_ptr->variable_bounds.data())),
    thrust::make_zip_iterator(thrust::make_tuple(
      primal_ray.data() + primal_size_h_, problem_ptr->variable_bounds.data() + primal_size_h_)),
    max_violation<f_t, f_t2>(),
    f_t(0.0),
    thrust::maximum<f_t>());
}

template <typename i_t, typename f_t>
void infeasibility_information_t<i_t, f_t>::compute_homogenous_primal_objective(
  rmm::device_uvector<f_t>& primal_ray)
{
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
                                                  primal_size_h_,
                                                  primal_ray.data(),
                                                  primal_stride,
                                                  problem_ptr->objective_coefficients.data(),
                                                  primal_stride,
                                                  primal_ray_linear_objective_.data(),
                                                  stream_view_));

  // just to scale from the primal ray scaling
  raft::linalg::eltwiseMultiply(primal_ray_linear_objective_.data(),
                                primal_ray_linear_objective_.data(),
                                primal_ray_inf_norm_inverse_.data(),
                                1,
                                stream_view_);
}

template <typename i_t, typename f_t>
void infeasibility_information_t<i_t, f_t>::compute_homogenous_dual_residual(
  cusparse_view_t<i_t, f_t>& cusparse_view,
  rmm::device_uvector<f_t>& tmp_primal,
  rmm::device_uvector<f_t>& primal_ray)
{
  // compute objective product (Q*x) if QP

  // need to recompute the primal gradient since c is the all zero vector in the homogenous case
  // this means that the primal gradient is computed as -A^T*y
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       reusable_device_scalar_value_neg_1_.data(),
                                                       cusparse_view.A_T,
                                                       cusparse_view.dual_solution,
                                                       reusable_device_scalar_value_0_.data(),
                                                       cusparse_view.tmp_primal,
                                                       CUSPARSE_SPMV_CSR_ALG2,
                                                       (f_t*)cusparse_view.buffer_transpose.data(),
                                                       stream_view_));

  compute_reduced_cost_from_primal_gradient(tmp_primal,
                                            primal_ray);  // primal gradient is now in temp

  raft::linalg::eltwiseSub(homogenous_dual_residual_.data(),
                           tmp_primal.data(),  // primal_gradient
                           reduced_cost_.data(),
                           primal_size_h_,
                           stream_view_);
}

template <typename i_t, typename f_t>
void infeasibility_information_t<i_t, f_t>::compute_homogenous_dual_objective(
  rmm::device_uvector<f_t>& dual_ray)
{
  raft::linalg::ternaryOp(bound_value_.data(),
                          dual_ray.data(),
                          problem_ptr->constraint_lower_bounds.data(),
                          problem_ptr->constraint_upper_bounds.data(),
                          dual_size_h_,
                          constraint_bound_value_reduced_cost_product<f_t>(),
                          stream_view_);

  cub::DeviceReduce::Sum(rmm_tmp_buffer_.data(),
                         size_of_buffer_,
                         bound_value_.begin(),
                         dual_ray_linear_objective_.data(),
                         dual_size_h_,
                         stream_view_);

#ifdef PDLP_DEBUG_MODE
  std::cout << "-compute_homogenous_dual_objective:\n"
            << "  dual_ray_linear_objective_ before="
            << dual_ray_linear_objective_.value(stream_view_) << std::endl;
#endif

  compute_reduced_costs_dual_objective_contribution();

  raft::linalg::eltwiseAdd(dual_ray_linear_objective_.data(),
                           dual_ray_linear_objective_.data(),
                           reduced_cost_dual_objective_.data(),
                           1,
                           stream_view_);
#ifdef PDLP_DEBUG_MODE
  std::cout << "  reduced_cost_dual_objective_=" << reduced_cost_dual_objective_.value(stream_view_)
            << std::endl;
  std::cout << "  dual_ray_linear_objective_ after="
            << dual_ray_linear_objective_.value(stream_view_) << std::endl;
#endif
}

template <typename i_t, typename f_t>
void infeasibility_information_t<i_t, f_t>::compute_reduced_cost_from_primal_gradient(
  rmm::device_uvector<f_t>& primal_gradient, rmm::device_uvector<f_t>& primal_ray)
{
  using f_t2 = typename type_2<f_t>::type;
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(primal_gradient.data(), problem_ptr->variable_bounds.data()),
    bound_value_.data(),
    primal_size_h_,
    bound_value_gradient<f_t, f_t2>(),
    stream_view_);

  if (pdlp_hyper_params::handle_some_primal_gradients_on_finite_bounds_as_residuals) {
    raft::linalg::ternaryOp(reduced_cost_.data(),
                            primal_ray.data(),
                            bound_value_.data(),
                            primal_gradient.data(),
                            primal_size_h_,
                            copy_gradient_if_should_be_reduced_cost<f_t>(),
                            stream_view_);
  } else {
    raft::linalg::binaryOp(reduced_cost_.data(),
                           bound_value_.data(),
                           primal_gradient.data(),
                           primal_size_h_,
                           copy_gradient_if_finite_bounds<f_t>(),
                           stream_view_);
  }
}

template <typename i_t, typename f_t>
void infeasibility_information_t<i_t, f_t>::compute_reduced_costs_dual_objective_contribution()
{
  using f_t2 = typename type_2<f_t>::type;
  // Check if these bounds are the same as computed above
  // if reduced cost is positive -> lower bound, negative -> upper bounds, 0 -> 0
  // if bound_val is not finite let element be -inf, otherwise bound_value*reduced_cost
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(reduced_cost_.data(), problem_ptr->variable_bounds.data()),
    bound_value_.data(),
    primal_size_h_,
    bound_value_reduced_cost_product<f_t, f_t2>(),
    stream_view_);

  // sum over bound_value*reduced_cost
  cub::DeviceReduce::Sum(rmm_tmp_buffer_.data(),
                         size_of_buffer_,
                         bound_value_.begin(),
                         reduced_cost_dual_objective_.data(),
                         primal_size_h_,
                         stream_view_);
}

template <typename i_t, typename f_t>
typename infeasibility_information_t<i_t, f_t>::view_t infeasibility_information_t<i_t, f_t>::view()
{
  infeasibility_information_t<i_t, f_t>::view_t v;

  v.primal_ray_inf_norm          = primal_ray_inf_norm_.data();
  v.primal_ray_max_violation     = primal_ray_max_violation_.data();
  v.max_primal_ray_infeasibility = max_primal_ray_infeasibility_.data();
  v.primal_ray_linear_objective  = primal_ray_linear_objective_.data();

  v.dual_ray_inf_norm          = dual_ray_inf_norm_.data();
  v.max_dual_ray_infeasibility = max_dual_ray_infeasibility_.data();
  v.dual_ray_linear_objective  = dual_ray_linear_objective_.data();

  v.reduced_cost_inf_norm = reduced_cost_inf_norm_.data();

  v.homogenous_primal_residual = homogenous_primal_residual_.data();
  v.homogenous_dual_residual   = homogenous_dual_residual_.data();
  v.reduced_cost               = reduced_cost_.data();

  return v;
}

#if MIP_INSTANTIATE_FLOAT
template class infeasibility_information_t<int, float>;

template __global__ void compute_remaining_stats_kernel<int, float>(
  typename infeasibility_information_t<int, float>::view_t infeasibility_information_view);
#endif

#if MIP_INSTANTIATE_DOUBLE
template class infeasibility_information_t<int, double>;

template __global__ void compute_remaining_stats_kernel<int, double>(
  typename infeasibility_information_t<int, double>::view_t infeasibility_information_view);
#endif

}  // namespace cuopt::linear_programming::detail
