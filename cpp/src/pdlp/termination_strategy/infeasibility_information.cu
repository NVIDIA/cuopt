/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <pdlp/initial_scaling_strategy/initial_scaling.cuh>
#include <pdlp/pdlp_constants.hpp>
#include <pdlp/restart_strategy/pdlp_restart_strategy.cuh>
#include <pdlp/termination_strategy/infeasibility_information.hpp>
#include <pdlp/utils.cuh>

#include <cuopt/linear_programming/utilities/segmented_sum_handler.cuh>

#include <mip_heuristics/mip_constants.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/nvtx.hpp>
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
  const problem_t<i_t, f_t>& op_problem_scaled,
  cusparse_view_t<i_t, f_t>& cusparse_view,
  const cusparse_view_t<i_t, f_t>& scaled_cusparse_view,
  i_t primal_size,
  i_t dual_size,
  const pdlp_initial_scaling_strategy_t<i_t, f_t>& scaling_strategy,
  bool infeasibility_detection,
  const std::vector<pdlp_climber_strategy_t>& climber_strategies,
  const pdlp_hyper_params::pdlp_hyper_params_t& hyper_params)
  : handle_ptr_(handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    primal_size_h_(primal_size),
    dual_size_h_(dual_size),
    problem_ptr(&op_problem),
    op_problem_scaled_(op_problem_scaled),
    op_problem_cusparse_view_(cusparse_view),
    scaled_cusparse_view_(scaled_cusparse_view),
    primal_ray_inf_norm_(climber_strategies.size(), stream_view_),
    primal_ray_inf_norm_inverse_{stream_view_},
    neg_primal_ray_inf_norm_inverse_{stream_view_},
    primal_ray_max_violation_{stream_view_},
    max_primal_ray_infeasibility_{climber_strategies.size(), stream_view_},
    primal_ray_linear_objective_(climber_strategies.size(), stream_view_),
    dual_ray_inf_norm_(climber_strategies.size(), stream_view_),
    max_dual_ray_infeasibility_{climber_strategies.size(), stream_view_},
    dual_ray_linear_objective_{climber_strategies.size(), stream_view_},
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
    primal_slack_{(is_cupdlpx_restart<i_t, f_t>(hyper_params))
                    ? static_cast<size_t>(dual_size_h_ * climber_strategies.size())
                    : 0,
                  stream_view_},
    dual_slack_{(is_cupdlpx_restart<i_t, f_t>(hyper_params))
                  ? static_cast<size_t>(primal_size_h_ * climber_strategies.size())
                  : 0,
                stream_view_},
    sum_primal_slack_{climber_strategies.size(), stream_view_},
    sum_dual_slack_{climber_strategies.size(), stream_view_},
    reusable_device_scalar_value_1_{1.0, stream_view_},
    reusable_device_scalar_value_0_{0.0, stream_view_},
    reusable_device_scalar_value_neg_1_{-1.0, stream_view_},
    scaling_strategy_(scaling_strategy),
    segmented_sum_handler_(stream_view_),
    climber_strategies_(climber_strategies),
    hyper_params_(hyper_params)
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

    RAFT_CUDA_TRY(cudaMemsetAsync(dual_ray_linear_objective_.data(),
                                  0,
                                  sizeof(f_t) * dual_ray_linear_objective_.size(),
                                  stream_view_));
    RAFT_CUDA_TRY(cudaMemsetAsync(max_dual_ray_infeasibility_.data(),
                                  0,
                                  sizeof(f_t) * max_dual_ray_infeasibility_.size(),
                                  stream_view_));

    RAFT_CUDA_TRY(cudaMemsetAsync(primal_ray_linear_objective_.data(),
                                  0,
                                  sizeof(f_t) * primal_ray_linear_objective_.size(),
                                  stream_view_));
    RAFT_CUDA_TRY(cudaMemsetAsync(max_primal_ray_infeasibility_.data(),
                                  0,
                                  sizeof(f_t) * max_primal_ray_infeasibility_.size(),
                                  stream_view_));
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
         infeasibility_information_view.max_dual_ray_infeasibility[0],
         infeasibility_information_view.dual_ray_linear_objective[0]);
#endif
  if (scaling_factor < 0.0 || scaling_factor > 0.0) {
    infeasibility_information_view.max_dual_ray_infeasibility[0] =
      infeasibility_information_view.max_dual_ray_infeasibility[0] / scaling_factor;
    infeasibility_information_view.dual_ray_linear_objective[0] =
      infeasibility_information_view.dual_ray_linear_objective[0] / scaling_factor;
  } else {
    infeasibility_information_view.max_dual_ray_infeasibility[0] = f_t(0.0);
    infeasibility_information_view.dual_ray_linear_objective[0]  = f_t(0.0);
  }
#ifdef PDLP_DEBUG_MODE
  printf("    After max_dual_ray_infeasibility=%lf dual_ray_linear_objective=%lf\n",
         infeasibility_information_view.max_dual_ray_infeasibility[0],
         infeasibility_information_view.dual_ray_linear_objective[0]);
  printf("    primal_ray_inf_norm=%lf\n", *infeasibility_information_view.primal_ray_inf_norm);
#endif
  // Update primal max ray infeasibility
  if (*infeasibility_information_view.primal_ray_inf_norm > f_t(0.0)) {
    infeasibility_information_view.max_primal_ray_infeasibility[0] =
      raft::max(infeasibility_information_view.max_primal_ray_infeasibility[0],
                *infeasibility_information_view.primal_ray_max_violation) /
      infeasibility_information_view.primal_ray_inf_norm[0];
  } else {
    infeasibility_information_view.max_primal_ray_infeasibility[0] = f_t(0.0);
    infeasibility_information_view.primal_ray_linear_objective[0]  = f_t(0.0);
  }
#ifdef PDLP_DEBUG_MODE
  printf("    max_primal_ray_infeasibility=%lf primal_ray_linear_objective=%lf\n",
         infeasibility_information_view.max_primal_ray_infeasibility[0],
         infeasibility_information_view.primal_ray_linear_objective[0]);
#endif
}

template <typename f_t>
struct max_abs_t {
  HD f_t operator()(f_t a, f_t b) { return cuda::std::max(cuda::std::abs(a), cuda::std::abs(b)); }
};

template <typename f_t>
HDI f_t finite_or_zero(f_t in)
{
  return isfinite(in) ? in : f_t(0.0);
}

template <typename i_t, typename f_t>
void infeasibility_information_t<i_t, f_t>::compute_infeasibility_information(
  pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
  rmm::device_uvector<f_t>& primal_ray,
  rmm::device_uvector<f_t>& dual_ray)
{
  raft::common::nvtx::range fun_scope("compute_infeasibility_information");
  using f_t2 = typename type_2<f_t>::type;

  if (is_cupdlpx_restart<i_t, f_t>(hyper_params_)) {
    const f_t bound_rescaling     = (hyper_params_.bound_objective_rescaling)
                                      ? scaling_strategy_.get_h_bound_rescaling()
                                      : f_t(1.0);
    const f_t objective_rescaling = (hyper_params_.bound_objective_rescaling)
                                      ? scaling_strategy_.get_h_objective_rescaling()
                                      : f_t(1.0);

#ifdef CUPDLP_DEBUG_MODE
    print("delta_primal_solution after scale before mod", primal_ray);
    print("delta_dual_solution after scale before mod", dual_ray);
#endif

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(primal_ray.data(),
                            problem_wrap_container(op_problem_scaled_.variable_bounds)),
      primal_ray.data(),
      primal_ray.size(),
      [] HD(f_t primal, f_t2 bounds) {
        const f_t lower      = get_lower(bounds);
        const f_t upper      = get_upper(bounds);
        f_t primal_to_return = primal;
        if (isfinite(lower)) primal_to_return = cuda::std::max(primal_to_return, f_t(0.0));
        if (isfinite(upper)) primal_to_return = cuda::std::min(primal_to_return, f_t(0.0));
        return primal_to_return;
      },
      stream_view_);

    // Inf norm of primal ray
    segmented_sum_handler_.segmented_reduce_helper(primal_ray.data(),
                                                   primal_ray_inf_norm_.data(),
                                                   climber_strategies_.size(),
                                                   primal_size_h_,
                                                   max_abs_t<f_t>{},
                                                   f_t(0.0));

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(dual_ray.data(),
                            problem_wrap_container(op_problem_scaled_.constraint_lower_bounds),
                            problem_wrap_container(op_problem_scaled_.constraint_upper_bounds)),
      dual_ray.data(),
      dual_ray.size(),
      [] HD(f_t dual, f_t lower, f_t upper) {
        f_t dual_to_return = dual;
        if (!isfinite(lower)) dual_to_return = cuda::std::min(dual_to_return, f_t(0.0));
        if (!isfinite(upper)) dual_to_return = cuda::std::max(dual_to_return, f_t(0.0));
        return dual_to_return;
      },
      stream_view_);

#ifdef CUPDLP_DEBUG_MODE
    print("delta_primal_solution after", primal_ray);
    print("delta_dual_solution after", dual_ray);
#endif
    // Inf norm of dual ray
    segmented_sum_handler_.segmented_reduce_helper(dual_ray.data(),
                                                   dual_ray_inf_norm_.data(),
                                                   climber_strategies_.size(),
                                                   dual_size_h_,
                                                   max_abs_t<f_t>{},
                                                   f_t(0.0));

    cub::DeviceFor::Bulk(
      primal_ray.size(),
      [primal_ray_inf_norm = make_span(primal_ray_inf_norm_),
       primal_ray_data     = make_span(primal_ray),
       primal_size_h_      = primal_size_h_] __device__(i_t id) {
        const f_t primal_ray_inf_norm_value = primal_ray_inf_norm[id / primal_size_h_];
        if (primal_ray_inf_norm_value > f_t(0.0))
          primal_ray_data[id] = primal_ray_data[id] / primal_ray_inf_norm_value;
      },
      stream_view_);
#ifdef CUPDLP_DEBUG_MODE
    print("delta_primal_solution after scale", primal_ray);
    print("delta_dual_solution after scale", dual_ray);
    printf("primal_ray_inf_norm=%lf\n", primal_ray_inf_norm_.element(0, stream_view_));
    printf("dual_ray_inf_norm=%lf\n", dual_ray_inf_norm_.element(0, stream_view_));
#endif

    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm(
      handle_ptr_->get_cusparse_handle(),
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      reusable_device_scalar_value_1_.data(),
      scaled_cusparse_view_.A,
      scaled_cusparse_view_.batch_delta_primal_solutions,
      reusable_device_scalar_value_0_.data(),
      scaled_cusparse_view_.batch_tmp_duals,
      CUSPARSE_SPMM_CSR_ALG3,
      (f_t*)scaled_cusparse_view_.buffer_non_transpose_batch.data(),
      stream_view_));
    RAFT_CUSPARSE_TRY(
      raft::sparse::detail::cusparsespmm(handle_ptr_->get_cusparse_handle(),
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         reusable_device_scalar_value_1_.data(),
                                         scaled_cusparse_view_.A_T,
                                         scaled_cusparse_view_.batch_delta_dual_solutions,
                                         reusable_device_scalar_value_0_.data(),
                                         scaled_cusparse_view_.batch_tmp_primals,
                                         CUSPARSE_SPMM_CSR_ALG3,
                                         (f_t*)scaled_cusparse_view_.buffer_transpose_batch.data(),
                                         stream_view_));

#ifdef CUPDLP_DEBUG_MODE
    print("primal_product", current_pdhg_solver.get_dual_tmp_resource());
    print("dual_product", current_pdhg_solver.get_primal_tmp_resource());
#endif

    // Dot product on each objective . delta primal = primal_ray_linear_objective
    segmented_sum_handler_.segmented_sum_helper(
      thrust::make_transform_iterator(
        thrust::make_zip_iterator(thrust::make_tuple(
          problem_wrap_container(op_problem_scaled_.objective_coefficients), primal_ray.data())),
        tuple_multiplies<f_t>{}),
      thrust::make_transform_output_iterator(
        primal_ray_linear_objective_.data(),
        [bound_rescaling, objective_rescaling] __device__(f_t out) {
          return out / (bound_rescaling * objective_rescaling);
        }),
      climber_strategies_.size(),
      primal_size_h_);

#ifdef CUPDLP_DEBUG_MODE
    printf("primal_ray_linear_objective before=%lf\n",
           primal_ray_linear_objective_.element(0, stream_view_));
#endif

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(dual_ray.data(),
                            problem_wrap_container(op_problem_scaled_.constraint_lower_bounds),
                            problem_wrap_container(op_problem_scaled_.constraint_upper_bounds)),
      primal_slack_.data(),
      primal_slack_.size(),
      [] HD(f_t dual, f_t lower, f_t upper) {
        return cuda::std::max(dual, f_t(0.0)) * finite_or_zero(lower) +
               cuda::std::min(dual, f_t(0.0)) * finite_or_zero(upper);
      },
      stream_view_);

#ifdef CUPDLP_DEBUG_MODE
    print("primal_slack", primal_slack_);
#endif

    using f_t2 = typename type_2<f_t>::type;
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(current_pdhg_solver.get_primal_tmp_resource().data(),
                            problem_wrap_container(op_problem_scaled_.variable_bounds)),
      dual_slack_.data(),
      dual_slack_.size(),
      [] HD(f_t dual, f_t2 bounds) {
        const f_t lower = get_lower(bounds);
        const f_t upper = get_upper(bounds);
        return cuda::std::max(-dual, f_t(0.0)) * finite_or_zero(lower) +
               cuda::std::min(-dual, f_t(0.0)) * finite_or_zero(upper);
      },
      stream_view_);

#ifdef CUPDLP_DEBUG_MODE
    print("dual_slack", dual_slack_);
#endif

    segmented_sum_handler_.segmented_sum_helper(
      primal_slack_.data(), sum_primal_slack_.data(), climber_strategies_.size(), dual_size_h_);

    segmented_sum_handler_.segmented_sum_helper(
      dual_slack_.data(), sum_dual_slack_.data(), climber_strategies_.size(), primal_size_h_);

#ifdef CUPDLP_DEBUG_MODE
    printf("sum_primal_slack=%lf\n", sum_primal_slack_.element(0, stream_view_));
    printf("sum_dual_slack=%lf\n", sum_dual_slack_.element(0, stream_view_));
#endif

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(
        current_pdhg_solver.get_dual_tmp_resource().data(),
        problem_wrap_container(op_problem_scaled_.constraint_lower_bounds),
        problem_wrap_container(op_problem_scaled_.constraint_upper_bounds),
        problem_wrap_container(scaling_strategy_.get_constraint_matrix_scaling_vector())

          ),
      primal_slack_.data(),
      primal_slack_.size(),
      [] HD(f_t primal, f_t lower, f_t upper, f_t scale) {
        // TODO why is it max max here?
        return (cuda::std::max(-primal, f_t(0.0)) * isfinite(lower) +
                cuda::std::max(primal, f_t(0.0)) * isfinite(upper)) *
               scale;
      },
      stream_view_);

#ifdef CUPDLP_DEBUG_MODE
    print("primal_slack", primal_slack_);
#endif

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(
        current_pdhg_solver.get_primal_tmp_resource().data(),
        problem_wrap_container(op_problem_scaled_.variable_bounds),
        problem_wrap_container(scaling_strategy_.get_variable_scaling_vector())),
      dual_slack_.data(),
      dual_slack_.size(),
      [] HD(f_t dual, f_t2 bounds, f_t scale) {
        const f_t lower = get_lower(bounds);
        const f_t upper = get_upper(bounds);
        // TODO ask Chris: why is it max max above and max min here?
        return (cuda::std::max(-dual, f_t(0.0)) * !isfinite(lower) -
                cuda::std::min(-dual, f_t(0.0)) * !isfinite(upper)) *
               scale;
      },
      stream_view_);
#ifdef CUPDLP_DEBUG_MODE
    print("dual_slack", dual_slack_);
#endif
    // Inf norm to get max primal/dual infeasible
    segmented_sum_handler_.segmented_reduce_helper(primal_slack_.data(),
                                                   max_primal_ray_infeasibility_.data(),
                                                   climber_strategies_.size(),
                                                   dual_size_h_,
                                                   max_abs_t<f_t>{},
                                                   f_t(0.0));
    segmented_sum_handler_.segmented_reduce_helper(dual_slack_.data(),
                                                   max_dual_ray_infeasibility_.data(),
                                                   climber_strategies_.size(),
                                                   primal_size_h_,
                                                   max_abs_t<f_t>{},
                                                   f_t(0.0));

#ifdef CUPDLP_DEBUG_MODE
    printf("max_primal_ray_infeasibility=%lf\n",
           max_primal_ray_infeasibility_.element(0, stream_view_));
    printf("max_dual_ray_infeasibility=%lf\n",
           max_dual_ray_infeasibility_.element(0, stream_view_));
#endif

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(max_dual_ray_infeasibility_.data(),
                            dual_ray_inf_norm_.data(),
                            sum_primal_slack_.data(),
                            sum_dual_slack_.data()),
      thrust::make_zip_iterator(max_dual_ray_infeasibility_.data(),
                                dual_ray_linear_objective_.data()),
      dual_ray_linear_objective_.size(),
      [bound_rescaling, objective_rescaling] __device__(
        f_t max_dual_ray_infeasibility,
        f_t dual_ray_inf_norm,
        f_t sum_primal_slack,
        f_t sum_dual_slack) -> thrust::tuple<f_t, f_t> {
        const f_t scaling_factor = cuda::std::max(max_dual_ray_infeasibility, dual_ray_inf_norm);
        if (scaling_factor > f_t(0.0))
          return {max_dual_ray_infeasibility / scaling_factor,
                  ((sum_primal_slack + sum_dual_slack) / (bound_rescaling * objective_rescaling)) /
                    scaling_factor};
        else
          return {f_t(0.0), f_t(0.0)};
      },
      stream_view_);

#ifdef CUPDLP_DEBUG_MODE
    printf("max_dual_ray_infeasibility=%lf\n",
           max_dual_ray_infeasibility_.element(0, stream_view_));
    printf("dual_ray_objective=%lf\n", dual_ray_linear_objective_.element(0, stream_view_));
#endif
  } else {
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
            << dual_ray_linear_objective_.element(0, stream_view_) << std::endl;
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
            << dual_ray_linear_objective_.element(0, stream_view_) << std::endl;
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
    stream_view_.value());

  if (hyper_params_.handle_some_primal_gradients_on_finite_bounds_as_residuals) {
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
  v.max_primal_ray_infeasibility = make_span(max_primal_ray_infeasibility_);
  v.primal_ray_linear_objective  = make_span(primal_ray_linear_objective_);

  v.dual_ray_inf_norm          = dual_ray_inf_norm_.data();
  v.max_dual_ray_infeasibility = make_span(max_dual_ray_infeasibility_);
  v.dual_ray_linear_objective  = make_span(dual_ray_linear_objective_);

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
