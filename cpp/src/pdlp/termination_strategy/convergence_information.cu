/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <pdlp/pdlp_climber_strategy.hpp>
#include <pdlp/pdlp_constants.hpp>
#include <pdlp/swap_and_resize_helper.cuh>
#include <pdlp/termination_strategy/convergence_information.hpp>
#include <pdlp/utils.cuh>

#include <mip_heuristics/mip_constants.hpp>

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/nvtx.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/ternary_op.cuh>
#include <raft/util/cuda_utils.cuh>

#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>

#include <cub/cub.cuh>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
convergence_information_t<i_t, f_t>::convergence_information_t(
  raft::handle_t const* handle_ptr,
  problem_t<i_t, f_t>& op_problem,
  cusparse_view_t<i_t, f_t>& cusparse_view,
  i_t primal_size,
  i_t dual_size,
  const std::vector<pdlp_climber_strategy_t>& climber_strategies,
  const pdlp_hyper_params::pdlp_hyper_params_t& hyper_params)
  : batch_mode_(climber_strategies.size() > 1),
    handle_ptr_(handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    primal_size_h_(primal_size),
    dual_size_h_(dual_size),
    problem_ptr(&op_problem),
    op_problem_cusparse_view_(cusparse_view),
    l2_norm_primal_linear_objective_{0.0, stream_view_},
    l2_norm_primal_right_hand_side_{0.0, stream_view_},  // TODO later batch mode: per problem rhs
    primal_objective_{climber_strategies.size(), stream_view_},
    dual_objective_{climber_strategies.size(), stream_view_},
    reduced_cost_dual_objective_{f_t(0.0), stream_view_},
    l2_primal_residual_{climber_strategies.size(), stream_view_},
    l2_dual_residual_{climber_strategies.size(), stream_view_},
    linf_primal_residual_{0.0, stream_view_},
    linf_dual_residual_{0.0, stream_view_},
    nb_violated_constraints_{0, stream_view_},
    gap_{climber_strategies.size(), stream_view_},
    abs_objective_{climber_strategies.size(), stream_view_},
    primal_residual_{climber_strategies.size() * dual_size_h_, stream_view_},
    dual_residual_{climber_strategies.size() * primal_size_h_, stream_view_},
    reduced_cost_{climber_strategies.size() * primal_size_h_, stream_view_},
    bound_value_{static_cast<size_t>(std::max(primal_size_h_, dual_size_h_)), stream_view_},
    primal_slack_{(hyper_params.use_reflected_primal_dual)
                    ? static_cast<size_t>(dual_size_h_ * climber_strategies.size())
                    : 0,
                  stream_view_},
    reusable_device_scalar_value_1_{1.0, stream_view_},
    reusable_device_scalar_value_0_{0.0, stream_view_},
    reusable_device_scalar_value_neg_1_{-1.0, stream_view_},
    dual_dot_{climber_strategies.size(), stream_view_},
    sum_primal_slack_{climber_strategies.size(), stream_view_},
    climber_strategies_(climber_strategies),
    hyper_params_(hyper_params)
{
  RAFT_CUDA_TRY(cudaMemsetAsync(
    primal_objective_.data(), 0, sizeof(f_t) * primal_objective_.size(), stream_view_));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(dual_objective_.data(), 0, sizeof(f_t) * dual_objective_.size(), stream_view_));
  RAFT_CUDA_TRY(cudaMemsetAsync(gap_.data(), 0, sizeof(f_t) * gap_.size(), stream_view_));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(abs_objective_.data(), 0, sizeof(f_t) * abs_objective_.size(), stream_view_));

  RAFT_CUDA_TRY(cudaMemsetAsync(
    l2_dual_residual_.data(), 0, sizeof(f_t) * l2_dual_residual_.size(), stream_view_));
  RAFT_CUDA_TRY(cudaMemsetAsync(
    l2_primal_residual_.data(), 0, sizeof(f_t) * l2_primal_residual_.size(), stream_view_));

  combine_constraint_bounds(*problem_ptr,
                            primal_residual_,
                            batch_mode_);  // primal_residual_ will contain abs max of bounds when
                                           // finite, otherwise 0 //just reused allocated mem here

  // TODO later batch mode: different objective coefficients
  // constant throughout solving, so precompute
  my_l2_norm<i_t, f_t>(
    problem_ptr->objective_coefficients, l2_norm_primal_linear_objective_, handle_ptr_);

  if (hyper_params_.initial_primal_weight_combined_bounds) {
    cuopt_expects(!batch_mode_,
                  error_type_t::ValidationError,
                  "Batch mode not supported with initial_primal_weight_combined_bounds");
    my_l2_norm<i_t, f_t>(primal_residual_, l2_norm_primal_right_hand_side_, handle_ptr_);
  } else {
    // TODO later batch mode: different constraints bounds
    compute_sum_bounds(problem_ptr->constraint_lower_bounds,
                       problem_ptr->constraint_upper_bounds,
                       l2_norm_primal_right_hand_side_,
                       handle_ptr_->get_stream());
  }

  void* d_temp_storage        = NULL;
  size_t temp_storage_bytes_1 = 0;
  cub::DeviceReduce::Sum(d_temp_storage,
                         temp_storage_bytes_1,
                         bound_value_.begin(),
                         dual_objective_.data(),
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

  if (batch_mode_) {
    // Pass down any input pointer of the right type, actual pointer does not matter
    size_t byte_needed = 0;

    cub::DeviceSegmentedReduce::Sum(
      nullptr,
      byte_needed,
      thrust::make_transform_iterator(dual_dot_.data(), power_two_func_t<f_t>{}),
      dual_dot_.data(),
      climber_strategies_.size(),
      dual_size,
      stream_view_);
    dot_product_bytes_ = std::max(dot_product_bytes_, byte_needed);

    cub::DeviceSegmentedReduce::Sum(
      nullptr,
      byte_needed,
      thrust::make_transform_iterator(dual_dot_.data(), power_two_func_t<f_t>{}),
      dual_dot_.data(),
      climber_strategies_.size(),
      primal_size,
      stream_view_);
    dot_product_bytes_ = std::max(dot_product_bytes_, byte_needed);

    cub::DeviceSegmentedReduce::Sum(
      nullptr,
      byte_needed,
      thrust::make_transform_iterator(thrust::make_zip_iterator(dual_dot_.data(), dual_dot_.data()),
                                      tuple_multiplies<f_t>{}),
      dual_dot_.data(),
      climber_strategies_.size(),
      primal_size,
      stream_view_);
    dot_product_bytes_ = std::max(dot_product_bytes_, byte_needed);

    cub::DeviceSegmentedReduce::Sum(nullptr,
                                    byte_needed,
                                    dual_dot_.data(),
                                    dual_dot_.data(),
                                    climber_strategies_.size(),
                                    dual_size_h_,
                                    stream_view_);
    dot_product_bytes_ = std::max(dot_product_bytes_, byte_needed);

    cub::DeviceSegmentedReduce::Sum(
      nullptr,
      dot_product_bytes_,
      thrust::make_transform_iterator(
        thrust::make_zip_iterator(dual_dot_.data(),
                                  problem_wrap_container(problem_ptr->objective_coefficients)),
        tuple_multiplies<f_t>{}),
      primal_objective_.data(),
      climber_strategies_.size(),
      primal_size_h_,
      stream_view_);
    dot_product_bytes_ = std::max(dot_product_bytes_, byte_needed);

    dot_product_storage_.resize(dot_product_bytes_, stream_view_);
  }

  RAFT_CUDA_TRY(cudaMemsetAsync(
    primal_residual_.data(), 0.0, sizeof(f_t) * primal_residual_.size(), stream_view_));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(dual_residual_.data(), 0.0, sizeof(f_t) * dual_residual_.size(), stream_view_));
}

template <typename i_t, typename f_t>
__global__ void convergence_information_swap_device_vectors_kernel(
  const swap_pair_t<i_t>* swap_pairs,
  i_t swap_count,
  raft::device_span<f_t> primal_objective,
  raft::device_span<f_t> dual_objective,
  raft::device_span<f_t> l2_primal_residual,
  raft::device_span<f_t> l2_dual_residual,
  raft::device_span<f_t> gap,
  raft::device_span<f_t> abs_objective,
  raft::device_span<f_t> dual_dot,
  raft::device_span<f_t> sum_primal_slack)
{
  const i_t idx = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= swap_count) { return; }

  const i_t left  = swap_pairs[idx].left;
  const i_t right = swap_pairs[idx].right;
  cuda::std::swap(primal_objective[left], primal_objective[right]);
  cuda::std::swap(dual_objective[left], dual_objective[right]);
  cuda::std::swap(l2_primal_residual[left], l2_primal_residual[right]);
  cuda::std::swap(l2_dual_residual[left], l2_dual_residual[right]);
  cuda::std::swap(gap[left], gap[right]);
  cuda::std::swap(abs_objective[left], abs_objective[right]);
  cuda::std::swap(dual_dot[left], dual_dot[right]);
  cuda::std::swap(sum_primal_slack[left], sum_primal_slack[right]);
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::swap_context(
  const thrust::universal_host_pinned_vector<swap_pair_t<i_t>>& swap_pairs)
{
  if (swap_pairs.empty()) { return; }

  const auto batch_size = static_cast<i_t>(primal_objective_.size());
  cuopt_assert(batch_size > 0, "Batch size must be greater than 0");
  for (const auto& pair : swap_pairs) {
    cuopt_assert(pair.left < pair.right, "Left swap index must be less than right swap index");
    cuopt_assert(pair.left < batch_size, "Left swap index is out of bounds");
    cuopt_assert(pair.right < batch_size, "Right swap index is out of bounds");
  }

  matrix_swap(primal_residual_, dual_size_h_, swap_pairs);
  matrix_swap(dual_residual_, primal_size_h_, swap_pairs);
  matrix_swap(reduced_cost_, primal_size_h_, swap_pairs);
  matrix_swap(primal_slack_, dual_size_h_, swap_pairs);

  const auto [grid_size, block_size] =
    kernel_config_from_batch_size(static_cast<i_t>(swap_pairs.size()));
  convergence_information_swap_device_vectors_kernel<i_t, f_t>
    <<<grid_size, block_size, 0, stream_view_>>>(thrust::raw_pointer_cast(swap_pairs.data()),
                                                 static_cast<i_t>(swap_pairs.size()),
                                                 make_span(primal_objective_),
                                                 make_span(dual_objective_),
                                                 make_span(l2_primal_residual_),
                                                 make_span(l2_dual_residual_),
                                                 make_span(gap_),
                                                 make_span(abs_objective_),
                                                 make_span(dual_dot_),
                                                 make_span(sum_primal_slack_));
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::resize_context(i_t new_size)
{
  [[maybe_unused]] const auto batch_size = static_cast<i_t>(primal_objective_.size());
  cuopt_assert(batch_size > 0, "Batch size must be greater than 0");
  cuopt_assert(new_size > 0, "New size must be greater than 0");
  cuopt_assert(new_size < batch_size, "New size must be less than batch size");

  primal_residual_.resize(new_size * dual_size_h_, stream_view_);
  dual_residual_.resize(new_size * primal_size_h_, stream_view_);
  reduced_cost_.resize(new_size * primal_size_h_, stream_view_);
  primal_slack_.resize(new_size * dual_size_h_, stream_view_);

  primal_objective_.resize(new_size, stream_view_);
  dual_objective_.resize(new_size, stream_view_);
  l2_primal_residual_.resize(new_size, stream_view_);
  l2_dual_residual_.resize(new_size, stream_view_);
  gap_.resize(new_size, stream_view_);
  abs_objective_.resize(new_size, stream_view_);
  dual_dot_.resize(new_size, stream_view_);
  sum_primal_slack_.resize(new_size, stream_view_);
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::set_relative_dual_tolerance_factor(
  f_t dual_tolerance_factor)
{
  l2_norm_primal_linear_objective_.set_value_async(dual_tolerance_factor, stream_view_);
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::set_relative_primal_tolerance_factor(
  f_t primal_tolerance_factor)
{
  l2_norm_primal_right_hand_side_.set_value_async(primal_tolerance_factor, stream_view_);
}

template <typename i_t, typename f_t>
f_t convergence_information_t<i_t, f_t>::get_relative_dual_tolerance_factor() const
{
  return l2_norm_primal_linear_objective_.value(stream_view_);
}

template <typename i_t, typename f_t>
f_t convergence_information_t<i_t, f_t>::get_relative_primal_tolerance_factor() const
{
  return l2_norm_primal_right_hand_side_.value(stream_view_);
}

template <typename i_t, typename f_t>
__global__ void compute_remaining_stats_kernel(
  typename convergence_information_t<i_t, f_t>::view_t convergence_information_view, int batch_size)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= batch_size) { return; }

  convergence_information_view.gap[idx] =
    raft::abs(convergence_information_view.primal_objective[idx] -
              convergence_information_view.dual_objective[idx]);
  convergence_information_view.abs_objective[idx] =
    raft::abs(convergence_information_view.primal_objective[idx]) +
    raft::abs(convergence_information_view.dual_objective[idx]);
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::compute_convergence_information(
  pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
  rmm::device_uvector<f_t>& primal_iterate,
  rmm::device_uvector<f_t>& dual_iterate,
  [[maybe_unused]] const rmm::device_uvector<f_t>& dual_slack,
  const rmm::device_uvector<f_t>& combined_bounds,
  const rmm::device_uvector<f_t>& objective_coefficients,
  const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  cuopt_assert(
    primal_residual_.size() == dual_size_h_ * climber_strategies_.size(),
    "primal_residual_ size must be equal to primal_size_h_ * climber_strategies_.size()");
  cuopt_assert(primal_iterate.size() == primal_size_h_ * climber_strategies_.size(),
               "primal_iterate size must be equal to primal_size_h_ * climber_strategies_.size()");
  cuopt_assert(dual_residual_.size() == primal_size_h_ * climber_strategies_.size(),
               "dual_residual_ size must be equal to primal_size_h_ * climber_strategies_.size()");
  cuopt_assert(dual_iterate.size() == dual_size_h_ * climber_strategies_.size(),
               "dual_iterate size must be equal to dual_size_h_ * climber_strategies_.size()");
  cuopt_assert(l2_primal_residual_.size() == climber_strategies_.size(),
               "l2_primal_residual_ size must be equal to climber_strategies_.size()");
  cuopt_assert(l2_primal_residual_.size() == climber_strategies_.size(),
               "l2_primal_residual_ size must be equal to climber_strategies_.size()");
  cuopt_assert(l2_dual_residual_.size() == climber_strategies_.size(),
               "l2_dual_residual_ size must be equal to climber_strategies_.size()");

  raft::common::nvtx::range fun_scope("compute_convergence_information");

#ifdef CUPDLP_DEBUG_MODE
  print("primal_iterate", primal_iterate);
  print("dual_iterate", dual_iterate);
  print("dual_slack", dual_slack);
#endif

  compute_primal_residual(
    op_problem_cusparse_view_, current_pdhg_solver.get_dual_tmp_resource(), dual_iterate);
  compute_primal_objective(primal_iterate);

#ifdef CUPDLP_DEBUG_MODE
  print("Primal Residual", primal_residual_);
#endif

  if (!batch_mode_)
    my_l2_norm<i_t, f_t>(primal_residual_, l2_primal_residual_, handle_ptr_);
  else {
    cub::DeviceSegmentedReduce::Sum(
      dot_product_storage_.data(),
      dot_product_bytes_,
      thrust::make_transform_iterator(primal_residual_.data(), power_two_func_t<f_t>{}),
      l2_primal_residual_.data(),
      climber_strategies_.size(),
      dual_size_h_,
      stream_view_);
    cub::DeviceTransform::Transform(
      l2_primal_residual_.data(),
      l2_primal_residual_.data(),
      l2_primal_residual_.size(),
      [] HD(f_t x) { return raft::sqrt(x); },
      stream_view_);
  }

#ifdef CUPDLP_DEBUG_MODE
  print("Absolute Primal Residual", l2_primal_residual_);
#endif
  // If per_constraint_residual is false we still need to perform the l2 since it's used in kkt
  if (settings.per_constraint_residual) {
    // TODO later batch mode: handle per_constraint_residual here
    cuopt_expects(!batch_mode_,
                  error_type_t::ValidationError,
                  "Batch mode not supported for per_constraint_residual");

    // Compute the linf of (residual_i - rel * b_i)
    if (settings.save_best_primal_so_far) {
      const i_t zero_int = 0;
      nb_violated_constraints_.set_value_async(zero_int, handle_ptr_->get_stream());
    }
    auto transform_iter = thrust::make_transform_iterator(
      thrust::make_zip_iterator(primal_residual_.cbegin(), combined_bounds.cbegin()),
      relative_residual_t<i_t, f_t>{settings.tolerances.relative_primal_tolerance});
    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;
    RAFT_CUDA_TRY(cub::DeviceReduce::Max(d_temp_storage,
                                         temp_storage_bytes,
                                         transform_iter,
                                         linf_primal_residual_.data(),
                                         primal_residual_.size(),
                                         stream_view_));
    rmm::device_buffer temp_buf(temp_storage_bytes, stream_view_);
    RAFT_CUDA_TRY(cub::DeviceReduce::Max(temp_buf.data(),
                                         temp_storage_bytes,
                                         transform_iter,
                                         linf_primal_residual_.data(),
                                         primal_residual_.size(),
                                         stream_view_));
  }

  compute_dual_residual(op_problem_cusparse_view_,
                        current_pdhg_solver.get_primal_tmp_resource(),
                        primal_iterate,
                        dual_slack);
  compute_dual_objective(dual_iterate, primal_iterate, dual_slack);

#ifdef CUPDLP_DEBUG_MODE
  print("Dual Residual", dual_residual_);
#endif

  if (!batch_mode_)
    my_l2_norm<i_t, f_t>(dual_residual_, l2_dual_residual_, handle_ptr_);
  else {
    cub::DeviceSegmentedReduce::Sum(
      dot_product_storage_.data(),
      dot_product_bytes_,
      thrust::make_transform_iterator(dual_residual_.data(), power_two_func_t<f_t>{}),
      l2_dual_residual_.data(),
      climber_strategies_.size(),
      primal_size_h_,
      stream_view_);
    cub::DeviceTransform::Transform(
      l2_dual_residual_.data(),
      l2_dual_residual_.data(),
      l2_dual_residual_.size(),
      [] HD(f_t x) { return raft::sqrt(x); },
      stream_view_);
  }
#ifdef CUPDLP_DEBUG_MODE
  print("Absolute Dual Residual", l2_dual_residual_);
#endif
  // If per_constraint_residual is false we still need to perform the l2 since it's used in kkt
  if (settings.per_constraint_residual) {
    // TODO later batch mode: handle per_constraint_residual here
    cuopt_expects(!batch_mode_,
                  error_type_t::ValidationError,
                  "Batch mode not supported for per_constraint_residual");

    // Compute the linf of (residual_i - rel * c_i)
    {
      auto transform_iter = thrust::make_transform_iterator(
        thrust::make_zip_iterator(dual_residual_.cbegin(), objective_coefficients.cbegin()),
        relative_residual_t<i_t, f_t>{settings.tolerances.relative_dual_tolerance});
      void* d_temp_storage      = nullptr;
      size_t temp_storage_bytes = 0;
      cub::DeviceReduce::Max(d_temp_storage,
                             temp_storage_bytes,
                             transform_iter,
                             linf_dual_residual_.data(),
                             dual_residual_.size(),
                             stream_view_);
      rmm::device_buffer temp_buf(temp_storage_bytes, stream_view_);
      cub::DeviceReduce::Max(temp_buf.data(),
                             temp_storage_bytes,
                             transform_iter,
                             linf_dual_residual_.data(),
                             dual_residual_.size(),
                             stream_view_);
    }
  }

  const auto [grid_size, block_size] = kernel_config_from_batch_size(climber_strategies_.size());
  compute_remaining_stats_kernel<i_t, f_t>
    <<<grid_size, block_size, 0, stream_view_>>>(this->view(), climber_strategies_.size());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  //  cleanup for next termination evaluation
  RAFT_CUDA_TRY(cudaMemsetAsync(
    primal_residual_.data(), 0.0, sizeof(f_t) * primal_residual_.size(), stream_view_));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(dual_residual_.data(), 0.0, sizeof(f_t) * dual_residual_.size(), stream_view_));
}

template <typename f_t>
HDI f_t finite_or_zero(f_t in)
{
  return isfinite(in) ? in : f_t(0.0);
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::compute_primal_residual(
  cusparse_view_t<i_t, f_t>& cusparse_view,
  rmm::device_uvector<f_t>& tmp_dual,
  [[maybe_unused]] const rmm::device_uvector<f_t>& dual_iterate)
{
  raft::common::nvtx::range fun_scope("compute_primal_residual");

  // primal_product
  if (!batch_mode_) {
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
  } else {
    RAFT_CUSPARSE_TRY(
      raft::sparse::detail::cusparsespmm(handle_ptr_->get_cusparse_handle(),
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         reusable_device_scalar_value_1_.data(),
                                         cusparse_view.A,
                                         cusparse_view.batch_primal_solutions,
                                         reusable_device_scalar_value_0_.data(),
                                         cusparse_view.batch_tmp_duals,
                                         CUSPARSE_SPMM_CSR_ALG3,
                                         (f_t*)cusparse_view.buffer_non_transpose_batch.data(),
                                         stream_view_));
  }

  if (!hyper_params_.use_reflected_primal_dual) {
    // The constraint bound violations for the first part of the residual
    cuopt_expects(!batch_mode_,
                  error_type_t::ValidationError,
                  "Batch mode not supported for !use_reflected_primal_dual");

    raft::linalg::ternaryOp<f_t, violation<f_t>>(primal_residual_.data(),
                                                 tmp_dual.data(),
                                                 problem_ptr->constraint_lower_bounds.data(),
                                                 problem_ptr->constraint_upper_bounds.data(),
                                                 dual_size_h_,
                                                 violation<f_t>(),
                                                 stream_view_);
  } else {
    cuopt_assert(primal_residual_.size() == primal_slack_.size(),
                 "Both vectors should had the same size");
#ifdef CUPDLP_DEBUG_MODE
    print("tmp_dual", tmp_dual);
#endif
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(tmp_dual.data(),
                            problem_wrap_container(problem_ptr->constraint_lower_bounds),
                            problem_wrap_container(problem_ptr->constraint_upper_bounds),
                            dual_iterate.data()),
      thrust::make_zip_iterator(primal_residual_.data(), primal_slack_.data()),
      primal_residual_.size(),
      [] HD(f_t Ax, f_t lower, f_t upper, f_t dual) -> thrust::tuple<f_t, f_t> {
        const f_t clamped_Ax = raft::max(lower, raft::min(Ax, upper));
        return {Ax - clamped_Ax,
                raft::max(dual, f_t(0.0)) * finite_or_zero(lower) +
                  raft::min(dual, f_t(0.0)) * finite_or_zero(upper)};
      },
      stream_view_.value());
  }

#ifdef PDLP_DEBUG_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
#endif
}

template <typename i_t, typename f_t>
__global__ void apply_objective_scaling_and_offset(raft::device_span<f_t> objective,
                                                   f_t objective_scaling_factor,
                                                   f_t objective_offset,
                                                   int batch_size)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= batch_size) { return; }

  objective[idx] = objective_scaling_factor * (objective[idx] + objective_offset);
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::compute_primal_objective(
  rmm::device_uvector<f_t>& primal_solution)
{
  raft::common::nvtx::range fun_scope("compute_primal_objective");

  if (!batch_mode_) {
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
                                                    (int)primal_size_h_,
                                                    primal_solution.data(),
                                                    primal_stride,
                                                    problem_ptr->objective_coefficients.data(),
                                                    primal_stride,
                                                    primal_objective_.data(),
                                                    stream_view_));
  } else {
    cub::DeviceSegmentedReduce::Sum(
      dot_product_storage_.data(),
      dot_product_bytes_,
      thrust::make_transform_iterator(
        thrust::make_zip_iterator(primal_solution.data(),
                                  problem_wrap_container(problem_ptr->objective_coefficients)),
        tuple_multiplies<f_t>{}),
      primal_objective_.data(),
      climber_strategies_.size(),
      primal_size_h_,
      stream_view_);
  }

  // primal_objective = 1 * (primal_objective + 0) = primal_objective
  if (problem_ptr->presolve_data.objective_scaling_factor != 1 ||
      problem_ptr->presolve_data.objective_offset != 0) {
    const auto [grid_size, block_size] = kernel_config_from_batch_size(climber_strategies_.size());
    apply_objective_scaling_and_offset<i_t, f_t><<<grid_size, block_size, 0, stream_view_>>>(
      make_span(primal_objective_),
      problem_ptr->presolve_data.objective_scaling_factor,
      problem_ptr->presolve_data.objective_offset,
      climber_strategies_.size());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

#ifdef CUPDLP_DEBUG_MODE
  print("Primal objective", primal_objective_);
#endif
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::compute_dual_residual(
  cusparse_view_t<i_t, f_t>& cusparse_view,
  rmm::device_uvector<f_t>& tmp_primal,
  rmm::device_uvector<f_t>& primal_solution,
  [[maybe_unused]] const rmm::device_uvector<f_t>& dual_slack)
{
  cuopt_assert(tmp_primal.size() == primal_solution.size(),
               "tmp_primal size must be equal to primal_solution size");
  if (hyper_params_.use_reflected_primal_dual)
    cuopt_assert(tmp_primal.size() == dual_slack.size(),
                 "tmp_primal size must be equal to primal_solution size");
  cuopt_assert(dual_residual_.size() == primal_solution.size(),
               "dual_residual_ size must be equal to primal_solution size");

  raft::common::nvtx::range fun_scope("compute_dual_residual");
  // compute objective product (Q*x) if QP

  // gradient is recomputed with the dual solution that has been computed since the gradient was
  // last computed
  //  c-K^Ty -> copy c to gradient first
  thrust::fill(handle_ptr_->get_thrust_policy(), tmp_primal.begin(), tmp_primal.end(), f_t(0));

  if (!batch_mode_) {
    RAFT_CUSPARSE_TRY(
      raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         reusable_device_scalar_value_1_.data(),
                                         cusparse_view.A_T,
                                         cusparse_view.dual_solution,
                                         reusable_device_scalar_value_0_.data(),
                                         cusparse_view.tmp_primal,
                                         CUSPARSE_SPMV_CSR_ALG2,
                                         (f_t*)cusparse_view.buffer_transpose.data(),
                                         stream_view_));
  } else {
    RAFT_CUSPARSE_TRY(
      raft::sparse::detail::cusparsespmm(handle_ptr_->get_cusparse_handle(),
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         reusable_device_scalar_value_1_.data(),
                                         cusparse_view.A_T,
                                         cusparse_view.batch_dual_solutions,
                                         reusable_device_scalar_value_0_.data(),
                                         cusparse_view.batch_tmp_primals,
                                         CUSPARSE_SPMM_CSR_ALG3,
                                         (f_t*)cusparse_view.buffer_transpose_batch.data(),
                                         stream_view_));
  }

  // Substract with the objective vector manually to avoid possible cusparse bug w/ nonzero beta and
  // len(X)=1
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(problem_wrap_container(problem_ptr->objective_coefficients),
                          tmp_primal.data()),
    tmp_primal.data(),
    tmp_primal.size(),
    cuda::std::minus<>{},
    stream_view_);

  if (hyper_params_.use_reflected_primal_dual) {
    cub::DeviceTransform::Transform(cuda::std::make_tuple(tmp_primal.data(), dual_slack.data()),
                                    dual_residual_.data(),
                                    dual_residual_.size(),
                                    cuda::std::minus<>{},
                                    stream_view_.value());
  } else {
    cuopt_expects(!batch_mode_,
                  error_type_t::ValidationError,
                  "Batch mode not supported for !use_reflected_primal_dual");

    compute_reduced_cost_from_primal_gradient(tmp_primal, primal_solution);

    // primal_gradient - reduced_costs
    raft::linalg::eltwiseSub(dual_residual_.data(),
                             tmp_primal.data(),  // primal_gradient
                             reduced_cost_.data(),
                             primal_size_h_,
                             stream_view_);
  }
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::compute_dual_objective(
  rmm::device_uvector<f_t>& dual_solution,
  [[maybe_unused]] const rmm::device_uvector<f_t>& primal_solution,
  [[maybe_unused]] const rmm::device_uvector<f_t>& dual_slack)
{
  raft::common::nvtx::range fun_scope("compute_dual_objective");

  // for QP would need to add + problem.objective_constant - 0.5 * objective_product' *
  // primal_solution

  // the value of y term in the objective of the dual problem, see[]
  //  (l^c)^T[y]_+ − (u^c)^T[y]_− in the dual objective

  if (!hyper_params_.use_reflected_primal_dual) {
    cuopt_expects(!batch_mode_,
                  error_type_t::ValidationError,
                  "Batch mode not supported for !use_reflected_primal_dual");

    raft::linalg::ternaryOp(bound_value_.data(),
                            dual_solution.data(),
                            problem_ptr->constraint_lower_bounds.data(),
                            problem_ptr->constraint_upper_bounds.data(),
                            dual_size_h_,
                            constraint_bound_value_reduced_cost_product<f_t>(),
                            stream_view_);

    cub::DeviceReduce::Sum(rmm_tmp_buffer_.data(),
                           size_of_buffer_,
                           bound_value_.begin(),
                           dual_objective_.data(),
                           dual_size_h_,
                           stream_view_);

    compute_reduced_costs_dual_objective_contribution();

    raft::linalg::eltwiseAdd(dual_objective_.data(),
                             dual_objective_.data(),
                             reduced_cost_dual_objective_.data(),
                             1,
                             stream_view_);
  } else {
    // Could be the same but changed for backward compatiblity
    if (!batch_mode_) {
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
                                                      primal_size_h_,
                                                      dual_slack.data(),
                                                      primal_stride,
                                                      primal_solution.data(),
                                                      primal_stride,
                                                      dual_dot_.data(),
                                                      stream_view_));

      cub::DeviceReduce::Sum(rmm_tmp_buffer_.data(),
                             size_of_buffer_,
                             primal_slack_.data(),
                             sum_primal_slack_.data(),
                             dual_size_h_,
                             stream_view_);
    } else {
      cub::DeviceSegmentedReduce::Sum(
        dot_product_storage_.data(),
        dot_product_bytes_,
        thrust::make_transform_iterator(
          thrust::make_zip_iterator(dual_slack.data(), primal_solution.data()),
          tuple_multiplies<f_t>{}),
        dual_dot_.data(),
        climber_strategies_.size(),
        primal_size_h_,
        stream_view_);

      cub::DeviceSegmentedReduce::Sum(dot_product_storage_.data(),
                                      dot_product_bytes_,
                                      primal_slack_.data(),
                                      sum_primal_slack_.data(),
                                      climber_strategies_.size(),
                                      dual_size_h_,
                                      stream_view_);
    }

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(dual_dot_.data(), sum_primal_slack_.data()),
      dual_objective_.data(),
      dual_objective_.size(),
      cuda::std::plus<>{},
      stream_view_);
  }

  // dual_objective = 1 * (dual_objective + 0) = dual_objective
  if (problem_ptr->presolve_data.objective_scaling_factor != 1 ||
      problem_ptr->presolve_data.objective_offset != 0) {
    const auto [grid_size, block_size] = kernel_config_from_batch_size(climber_strategies_.size());
    apply_objective_scaling_and_offset<i_t, f_t><<<grid_size, block_size, 0, stream_view_>>>(
      make_span(dual_objective_),
      problem_ptr->presolve_data.objective_scaling_factor,
      problem_ptr->presolve_data.objective_offset,
      climber_strategies_.size());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

#ifdef CUPDLP_DEBUG_MODE
  print("Dual objective", dual_objective_);
#endif
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::compute_reduced_cost_from_primal_gradient(
  const rmm::device_uvector<f_t>& primal_gradient, const rmm::device_uvector<f_t>& primal_solution)
{
  raft::common::nvtx::range fun_scope("compute_reduced_cost_from_primal_gradient");

  using f_t2 = typename type_2<f_t>::type;
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(primal_gradient.data(), problem_ptr->variable_bounds.data()),
    bound_value_.data(),
    primal_size_h_,
    bound_value_gradient<f_t, f_t2>(),
    stream_view_.value());

  if (hyper_params_.handle_some_primal_gradients_on_finite_bounds_as_residuals) {
    raft::linalg::ternaryOp(reduced_cost_.data(),
                            primal_solution.data(),
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
void convergence_information_t<i_t, f_t>::compute_reduced_costs_dual_objective_contribution()
{
  raft::common::nvtx::range fun_scope("compute_reduced_costs_dual_objective_contribution");

  using f_t2 = typename type_2<f_t>::type;
  // if reduced cost is positive -> lower bound, negative -> upper bounds, 0 -> 0
  // if bound_val is not finite let element be -inf, otherwise bound_value*reduced_cost
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(reduced_cost_.data(), problem_ptr->variable_bounds.data()),
    bound_value_.data(),
    primal_size_h_,
    bound_value_reduced_cost_product<f_t, f_t2>(),
    stream_view_.value());

  // sum over bound_value*reduced_cost, but should be -inf if any element is -inf
  cub::DeviceReduce::Sum(rmm_tmp_buffer_.data(),
                         size_of_buffer_,
                         bound_value_.begin(),
                         reduced_cost_dual_objective_.data(),
                         primal_size_h_,
                         stream_view_);
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& convergence_information_t<i_t, f_t>::get_reduced_cost()
{
  return reduced_cost_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& convergence_information_t<i_t, f_t>::get_reduced_cost() const
{
  return reduced_cost_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& convergence_information_t<i_t, f_t>::get_l2_primal_residual() const
{
  return l2_primal_residual_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& convergence_information_t<i_t, f_t>::get_primal_objective() const
{
  return primal_objective_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& convergence_information_t<i_t, f_t>::get_dual_objective() const
{
  return dual_objective_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& convergence_information_t<i_t, f_t>::get_l2_dual_residual() const
{
  return l2_dual_residual_;
}

template <typename i_t, typename f_t>
const rmm::device_scalar<f_t>&
convergence_information_t<i_t, f_t>::get_relative_linf_primal_residual() const
{
  return linf_primal_residual_;
}

template <typename i_t, typename f_t>
const rmm::device_scalar<f_t>&
convergence_information_t<i_t, f_t>::get_relative_linf_dual_residual() const
{
  return linf_dual_residual_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& convergence_information_t<i_t, f_t>::get_gap() const
{
  return gap_;
}

template <typename i_t, typename f_t>
f_t convergence_information_t<i_t, f_t>::get_relative_gap_value(i_t climber_strategy_id) const
{
  return gap_.element(climber_strategy_id, stream_view_) /
         (f_t(1.0) + abs_objective_.element(climber_strategy_id, stream_view_));
}

template <typename i_t, typename f_t>
f_t convergence_information_t<i_t, f_t>::get_relative_l2_primal_residual_value(
  i_t climber_strategy_id) const
{
  // TODO later batch mode: handle per climber rhs
  return l2_primal_residual_.element(climber_strategy_id, stream_view_) /
         (f_t(1.0) + l2_norm_primal_right_hand_side_.value(stream_view_));
}

template <typename i_t, typename f_t>
f_t convergence_information_t<i_t, f_t>::get_relative_l2_dual_residual_value(
  i_t climber_strategy_id) const
{
  // TODO later batch mode: handle per climber objective
  return l2_dual_residual_.element(climber_strategy_id, stream_view_) /
         (f_t(1.0) + l2_norm_primal_linear_objective_.value(stream_view_));
}

template <typename i_t, typename f_t>
typename convergence_information_t<i_t, f_t>::view_t convergence_information_t<i_t, f_t>::view()
{
  convergence_information_t<i_t, f_t>::view_t v;
  v.primal_size = primal_size_h_;
  v.dual_size   = dual_size_h_;

  v.l2_norm_primal_linear_objective = l2_norm_primal_linear_objective_.data();
  v.l2_norm_primal_right_hand_side  = l2_norm_primal_right_hand_side_.data();

  v.primal_objective               = make_span(primal_objective_);
  v.dual_objective                 = make_span(dual_objective_);
  v.l2_primal_residual             = make_span(l2_primal_residual_);
  v.l2_dual_residual               = make_span(l2_dual_residual_);
  v.relative_l_inf_primal_residual = linf_primal_residual_.data();
  v.relative_l_inf_dual_residual   = linf_dual_residual_.data();

  v.gap           = make_span(gap_);
  v.abs_objective = make_span(abs_objective_);

  v.primal_residual = make_span(primal_residual_);
  v.dual_residual   = make_span(dual_residual_);
  v.reduced_cost    = make_span(reduced_cost_);
  v.bound_value     = make_span(bound_value_);

  return v;
}

template <typename i_t, typename f_t>
typename convergence_information_t<i_t, f_t>::primal_quality_adapter_t
convergence_information_t<i_t, f_t>::to_primal_quality_adapter(
  bool is_primal_feasible) const noexcept
{
  // TODO later batch mode: handle primal quality adapter here
  return {is_primal_feasible,
          nb_violated_constraints_.value(stream_view_),
          l2_primal_residual_.element(0, stream_view_),
          primal_objective_.element(0, stream_view_)};
}

#if MIP_INSTANTIATE_FLOAT
template class convergence_information_t<int, float>;

template __global__ void compute_remaining_stats_kernel<int, float>(
  typename convergence_information_t<int, float>::view_t convergence_information_view,
  int batch_size);
#endif

#if MIP_INSTANTIATE_DOUBLE
template class convergence_information_t<int, double>;

template __global__ void compute_remaining_stats_kernel<int, double>(
  typename convergence_information_t<int, double>::view_t convergence_information_view,
  int batch_size);
#endif

}  // namespace cuopt::linear_programming::detail
