/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>

#include <utilities/copy_helpers.hpp>

#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/utilities/segmented_sum_handler.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <pdlp/initial_scaling_strategy/initial_scaling.cuh>
#include <pdlp/pdlp_constants.hpp>
#include <pdlp/utils.cuh>

#include <raft/core/nvtx.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scatter.h>

namespace cuopt::linear_programming::detail {

template <typename f_t>
struct weighted_square_op {
  f_t weight;
  HDI f_t operator()(f_t v) { return v * v * weight; }
};

template <typename f_t>
struct rescaling_from_squared_norm_op {
  HDI f_t operator()(f_t sum) { return f_t(1.0) / (raft::sqrt(sum) + f_t(1.0)); }
};

template <typename f_t>
struct inverse_rescaling_op {
  HDI f_t operator()(f_t v)
  {
    cuopt_assert(v != f_t(0), "Numerical error: rescaling should never equal 0");
    return v != f_t(0) ? f_t(1.0) / v : v;
  }
};

template <typename i_t, typename f_t>
__global__ void scaling_swap_rescaling_kernel(const swap_pair_t<i_t>* swap_pairs,
                                              i_t swap_count,
                                              raft::device_span<f_t> bound_rescaling,
                                              raft::device_span<f_t> objective_rescaling)
{
  const i_t idx = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= swap_count) { return; }

  const i_t left  = swap_pairs[idx].left;
  const i_t right = swap_pairs[idx].right;

  cuda::std::swap(bound_rescaling[left], bound_rescaling[right]);
  cuda::std::swap(objective_rescaling[left], objective_rescaling[right]);
}

template <typename i_t, typename f_t>
pdlp_initial_scaling_strategy_t<i_t, f_t>::pdlp_initial_scaling_strategy_t(
  raft::handle_t const* handle_ptr,
  problem_t<i_t, f_t>& op_problem_scaled,
  i_t number_of_ruiz_iterations,
  f_t alpha,
  rmm::device_uvector<f_t>& A_T,
  rmm::device_uvector<i_t>& A_T_offsets,
  rmm::device_uvector<i_t>& A_T_indices,
  pdhg_solver_t<i_t, f_t>* pdhg_solver_ptr,
  const pdlp_hyper_params::pdlp_hyper_params_t& hyper_params,
  i_t original_batch_size,
  bool running_mip)
  : handle_ptr_(handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    primal_size_h_(op_problem_scaled.n_variables),
    dual_size_h_(op_problem_scaled.n_constraints),
    op_problem_scaled_(op_problem_scaled),
    pdhg_solver_ptr_(pdhg_solver_ptr),
    A_T_(A_T),
    A_T_offsets_(A_T_offsets),
    A_T_indices_(A_T_indices),
    hyper_params_(hyper_params),
    running_mip_(running_mip),
    iteration_constraint_matrix_scaling_{static_cast<size_t>(dual_size_h_), stream_view_},
    iteration_variable_scaling_{static_cast<size_t>(primal_size_h_), stream_view_},
    original_batch_size_(original_batch_size),
    bound_rescaling_(static_cast<size_t>(original_batch_size_), stream_view_),
    objective_rescaling_(static_cast<size_t>(original_batch_size_), stream_view_),
    h_bound_rescaling_(static_cast<size_t>(original_batch_size_), f_t(1)),
    h_objective_rescaling_(static_cast<size_t>(original_batch_size_), f_t(1)),
    cummulative_constraint_matrix_scaling_{static_cast<size_t>(dual_size_h_), stream_view_},
    cummulative_variable_scaling_{static_cast<size_t>(primal_size_h_), stream_view_}
{
  raft::common::nvtx::range fun_scope("Initializing initial_scaling_strategy");
#ifdef PDLP_DEBUG_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "Initializing initial_scaling_strategy" << std::endl;
#endif

  if (!running_mip_) cuopt_assert(pdhg_solver_ptr_ != nullptr, "PDHG solver pointer is null");
  cuopt_assert(original_batch_size_ > 0, "Original batch size must be positive");

  // start with all one for scaling vectors
  RAFT_CUDA_TRY(cudaMemsetAsync(
    iteration_constraint_matrix_scaling_.data(), 0.0, sizeof(f_t) * dual_size_h_, stream_view_));
  RAFT_CUDA_TRY(cudaMemsetAsync(
    iteration_variable_scaling_.data(), 0.0, sizeof(f_t) * primal_size_h_, stream_view_));
  thrust::fill(handle_ptr_->get_thrust_policy(),
               cummulative_constraint_matrix_scaling_.begin(),
               cummulative_constraint_matrix_scaling_.end(),
               f_t(1));
  thrust::fill(handle_ptr_->get_thrust_policy(),
               cummulative_variable_scaling_.begin(),
               cummulative_variable_scaling_.end(),
               f_t(1));
  thrust::fill(
    handle_ptr_->get_thrust_policy(), bound_rescaling_.begin(), bound_rescaling_.end(), f_t(1));
  thrust::fill(handle_ptr_->get_thrust_policy(),
               objective_rescaling_.begin(),
               objective_rescaling_.end(),
               f_t(1));

  compute_scaling_vectors(number_of_ruiz_iterations, alpha);

  iteration_constraint_matrix_scaling_.resize(0, stream_view_);
  iteration_variable_scaling_.resize(0, stream_view_);
}

template <typename i_t, typename f_t>
void pdlp_initial_scaling_strategy_t<i_t, f_t>::compute_scaling_vectors(
  i_t number_of_ruiz_iterations, f_t alpha)
{
  raft::common::nvtx::range fun_scope("compute_scaling_vectors");

  if (hyper_params_.do_ruiz_scaling) { ruiz_inf_scaling(number_of_ruiz_iterations); }
  if (hyper_params_.do_pock_chambolle_scaling) { pock_chambolle_scaling(alpha); }
}

template <typename i_t, typename f_t>
void pdlp_initial_scaling_strategy_t<i_t, f_t>::bound_objective_rescaling()
{
  // TODO: test bound obj scaling w/ MIP
  segmented_sum_handler_t<i_t, f_t> segmented_sum_handler(stream_view_);

  // ------- Constraints bounds scaling -------
  // This works whether we have different bounds per climber or not because of the
  // problem_wrap_container
  const i_t n_constrs  = op_problem_scaled_.n_constraints;
  const auto n_batches = original_batch_size_;
  auto bound_input     = thrust::make_transform_iterator(
    thrust::make_zip_iterator(problem_wrap_container(op_problem_scaled_.constraint_lower_bounds),
                              problem_wrap_container(op_problem_scaled_.constraint_upper_bounds)),
    rhs_sum_of_squares_t<f_t>{});
  auto bound_output = thrust::make_transform_output_iterator(bound_rescaling_.data(),
                                                             rescaling_from_squared_norm_op<f_t>{});

  segmented_sum_handler.segmented_sum_helper(bound_input, bound_output, n_batches, n_constrs);

  h_bound_rescaling_ = cuopt::host_copy(bound_rescaling_, stream_view_);

  // ------- Objective coefficients scaling -------

  const i_t n_variables = op_problem_scaled_.n_variables;
  auto objective_input  = thrust::make_transform_iterator(
    problem_wrap_container(op_problem_scaled_.objective_coefficients),
    weighted_square_op<f_t>{f_t(hyper_params_.initial_primal_weight_c_scaling)});
  auto objective_output = thrust::make_transform_output_iterator(
    objective_rescaling_.data(), rescaling_from_squared_norm_op<f_t>{});

  segmented_sum_handler.segmented_sum_helper(
    objective_input, objective_output, n_batches, n_variables);

  h_objective_rescaling_ = cuopt::host_copy(objective_rescaling_, stream_view_);
}

template <typename i_t, typename f_t>
__global__ void inf_norm_row_and_col_kernel(
  const typename problem_t<i_t, f_t>::view_t op_problem,
  typename pdlp_initial_scaling_strategy_t<i_t, f_t>::view_t initial_scaling_view)
{
  for (int row = blockIdx.x; row < op_problem.n_constraints; row += gridDim.x) {
    i_t row_offset              = op_problem.offsets[row];
    i_t nnz_in_row              = op_problem.offsets[row + 1] - row_offset;
    f_t constraint_scale_factor = initial_scaling_view.cummulative_constraint_matrix_scaling[row];
    for (int j = threadIdx.x; j < nnz_in_row; j += blockDim.x) {
      i_t col                   = op_problem.variables[row_offset + j];
      f_t variable_scale_factor = initial_scaling_view.cummulative_variable_scaling[col];
      f_t scaled_val =
        (op_problem.coefficients[row_offset + j] * constraint_scale_factor) * variable_scale_factor;
      f_t abs_val = raft::abs(scaled_val);

      // row part
      if (abs_val > initial_scaling_view.iteration_constraint_matrix_scaling[row]) {
        raft::myAtomicMax(&initial_scaling_view.iteration_constraint_matrix_scaling[row], abs_val);
      }

      // col part
      // Add max with abs val in objective_matrix here for QP for cols
      if (abs_val > initial_scaling_view.iteration_variable_scaling[col]) {
        raft::myAtomicMax(&initial_scaling_view.iteration_variable_scaling[col], abs_val);
      }
    }
  }
}

template <typename i_t, typename f_t>
void pdlp_initial_scaling_strategy_t<i_t, f_t>::ruiz_inf_scaling(i_t number_of_ruiz_iterations)
{
#ifdef PDLP_DEBUG_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  std::cout << "Doing ruiz_inf_scaling" << std::endl;
#endif
  for (int i = 0; i < number_of_ruiz_iterations; i++) {
    // find inf norm over rows and columns of the scaled matrix in given iteration (matrix is not
    // actually updated, but the scaled value is computed and evaluated)
    i_t number_of_blocks = op_problem_scaled_.n_constraints / block_size;
    if (op_problem_scaled_.n_constraints % block_size) number_of_blocks++;
    i_t number_of_threads = std::min(op_problem_scaled_.n_variables, (i_t)block_size);
    inf_norm_row_and_col_kernel<i_t, f_t><<<number_of_blocks, number_of_threads, 0, stream_view_>>>(
      op_problem_scaled_.view(), this->view());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    if (running_mip_) { reset_integer_variables(); }

    raft::linalg::binaryOp(cummulative_constraint_matrix_scaling_.data(),
                           cummulative_constraint_matrix_scaling_.data(),
                           iteration_constraint_matrix_scaling_.data(),
                           dual_size_h_,
                           a_divides_sqrt_b_bounded<f_t>(),
                           stream_view_);

    raft::linalg::binaryOp(cummulative_variable_scaling_.data(),
                           cummulative_variable_scaling_.data(),
                           iteration_variable_scaling_.data(),
                           primal_size_h_,
                           a_divides_sqrt_b_bounded<f_t>(),
                           stream_view_);

    // Reset the iteration_scaling vectors to all 0
    RAFT_CUDA_TRY(cudaMemsetAsync(
      iteration_constraint_matrix_scaling_.data(), 0.0, sizeof(f_t) * dual_size_h_, stream_view_));
    RAFT_CUDA_TRY(cudaMemsetAsync(
      iteration_variable_scaling_.data(), 0.0, sizeof(f_t) * primal_size_h_, stream_view_));
  }
}

template <typename i_t, typename f_t>
void pdlp_initial_scaling_strategy_t<i_t, f_t>::reset_integer_variables()
{
  thrust::scatter(
    handle_ptr_->get_thrust_policy(),
    thrust::make_constant_iterator<f_t>(1),
    thrust::make_constant_iterator<f_t>(1) + op_problem_scaled_.integer_indices.size(),
    op_problem_scaled_.integer_indices.begin(),
    iteration_variable_scaling_.begin());
}

template <typename i_t, typename f_t, int BLOCK_SIZE>
__global__ void pock_chambolle_scaling_kernel_row(
  const typename problem_t<i_t, f_t>::view_t op_problem,
  f_t alpha,
  typename pdlp_initial_scaling_strategy_t<i_t, f_t>::view_t initial_scaling_view)
{
  cuopt_assert(op_problem.n_constraints == gridDim.x,
               "Grid size should be equal to number of constraints");

  __shared__ f_t shared[BLOCK_SIZE / raft::WarpSize];
  auto accumlated_row_value = raft::device_span<f_t>{shared, BLOCK_SIZE / raft::WarpSize};
  f_t accumulated_value     = f_t(0);

  int row                     = blockIdx.x;
  i_t row_offset              = op_problem.offsets[row];
  i_t nnz_in_row              = op_problem.offsets[row + 1] - row_offset;
  f_t constraint_scale_factor = initial_scaling_view.cummulative_constraint_matrix_scaling[row];

  for (int j = threadIdx.x; j < nnz_in_row; j += blockDim.x) {
    i_t col                   = op_problem.variables[row_offset + j];
    f_t variable_scale_factor = initial_scaling_view.cummulative_variable_scaling[col];
    f_t scaled_val =
      (op_problem.coefficients[row_offset + j] * constraint_scale_factor) * variable_scale_factor;
    f_t abs_val = raft::abs(scaled_val);

    // row part
    f_t row_val = raft::pow(abs_val, alpha);
    accumulated_value += row_val;
  }

  accumulated_value =
    deterministic_block_reduce<f_t, BLOCK_SIZE>(accumlated_row_value, accumulated_value);

  if (threadIdx.x == 0)
    initial_scaling_view.iteration_constraint_matrix_scaling[row] = accumulated_value;
}

// All block browse through all the matrix but each block handle one column index
// This is to avoid multiple atomic between blocks and having indeterminism
template <typename i_t, typename f_t, int BLOCK_SIZE>
__global__ void pock_chambolle_scaling_kernel_col(
  const typename problem_t<i_t, f_t>::view_t op_problem,
  f_t alpha,
  typename pdlp_initial_scaling_strategy_t<i_t, f_t>::view_t initial_scaling_view,
  const f_t* A_T,
  const i_t* A_T_offsets,
  const i_t* A_T_indices)
{
  cuopt_assert(op_problem.n_variables == gridDim.x,
               "Grid size should be equal to number of variables");

  __shared__ f_t shared[BLOCK_SIZE / raft::WarpSize];
  auto accumlated_col_value = raft::device_span<f_t>{shared, BLOCK_SIZE / raft::WarpSize};
  f_t accumulated_value     = f_t(0);

  int col                   = blockIdx.x;
  i_t col_offset            = A_T_offsets[col];
  i_t nnz_in_col            = A_T_offsets[col + 1] - col_offset;
  f_t variable_scale_factor = initial_scaling_view.cummulative_variable_scaling[col];

  for (int j = threadIdx.x; j < nnz_in_col; j += blockDim.x) {
    i_t row                     = A_T_indices[col_offset + j];
    f_t constraint_scale_factor = initial_scaling_view.cummulative_constraint_matrix_scaling[row];
    f_t scaled_val = (A_T[col_offset + j] * constraint_scale_factor) * variable_scale_factor;
    f_t abs_val    = raft::abs(scaled_val);

    // col part
    // Add max with abs val in objective_matrix here for QP for cols
    f_t col_val = raft::pow(abs_val, f_t(2) - alpha);
    accumulated_value += col_val;
  }

  accumulated_value =
    deterministic_block_reduce<f_t, BLOCK_SIZE>(accumlated_col_value, accumulated_value);

  if (threadIdx.x == 0) initial_scaling_view.iteration_variable_scaling[col] = accumulated_value;
}

template <typename i_t, typename f_t>
void pdlp_initial_scaling_strategy_t<i_t, f_t>::pock_chambolle_scaling(f_t alpha)
{
  // Reset the iteration_scaling vectors to all 0
  RAFT_CUDA_TRY(cudaMemsetAsync(
    iteration_constraint_matrix_scaling_.data(), 0.0, sizeof(f_t) * dual_size_h_, stream_view_));
  RAFT_CUDA_TRY(cudaMemsetAsync(
    iteration_variable_scaling_.data(), 0.0, sizeof(f_t) * primal_size_h_, stream_view_));

  EXE_CUOPT_EXPECTS(
    alpha >= 0.0 && alpha <= 2.0,
    "Invalid alpha value for Pock Chambolle Scaling in initial scaling step. Must be "
    "be in interval [0,2] but was %f",
    alpha);

  // find sum over (weight^alpha) for rows and (weight^(2.0-alpha)) for columns of the scaled
  // matrix (scaled value is computed and evaluated within)

  // Row / Columns are treated seperately to be deterministic (floating point issues)

  constexpr i_t number_of_threads = 128;
  pock_chambolle_scaling_kernel_row<i_t, f_t, number_of_threads>
    <<<op_problem_scaled_.n_constraints, number_of_threads, 0, stream_view_>>>(
      op_problem_scaled_.view(), alpha, this->view());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Use transposed matrix instead to compute column-wise more easily
  pock_chambolle_scaling_kernel_col<i_t, f_t, number_of_threads>
    <<<op_problem_scaled_.n_variables, number_of_threads, 0, stream_view_>>>(
      op_problem_scaled_.view(),
      alpha,
      this->view(),
      A_T_.data(),
      A_T_offsets_.data(),
      A_T_indices_.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  if (running_mip_) { reset_integer_variables(); }

  // divide the sqrt of the vectors of the sums from above to the respective scaling vectors
  // (only if sqrt(sum)>0)
  raft::linalg::binaryOp(cummulative_constraint_matrix_scaling_.data(),
                         cummulative_constraint_matrix_scaling_.data(),
                         iteration_constraint_matrix_scaling_.data(),
                         dual_size_h_,
                         a_divides_sqrt_b_bounded<f_t>(),
                         stream_view_);
  raft::linalg::binaryOp(cummulative_variable_scaling_.data(),
                         cummulative_variable_scaling_.data(),
                         iteration_variable_scaling_.data(),
                         primal_size_h_,
                         a_divides_sqrt_b_bounded<f_t>(),
                         stream_view_);
}

template <typename i_t, typename f_t>
__global__ void scale_problem_kernel(
  const typename pdlp_initial_scaling_strategy_t<i_t, f_t>::view_t initial_scaling_view,
  const typename problem_t<i_t, f_t>::view_t op_problem)
{
  for (int row = blockIdx.x; row < op_problem.n_constraints; row += gridDim.x) {
    i_t row_offset              = op_problem.offsets[row];
    i_t nnz_in_row              = op_problem.offsets[row + 1] - row_offset;
    f_t constraint_scale_factor = initial_scaling_view.cummulative_constraint_matrix_scaling[row];

    for (int j = threadIdx.x; j < nnz_in_row; j += blockDim.x) {
      i_t col                   = op_problem.variables[row_offset + j];
      f_t variable_scale_factor = initial_scaling_view.cummulative_variable_scaling[col];
      op_problem.coefficients[row_offset + j] =
        op_problem.coefficients[row_offset + j] * constraint_scale_factor * variable_scale_factor;
    }
  }
}

template <typename i_t, typename f_t>
__global__ void scale_transposed_problem_kernel(
  const typename pdlp_initial_scaling_strategy_t<i_t, f_t>::view_t initial_scaling_view,
  f_t* A_T,
  i_t* A_T_offsets,
  i_t* A_T_indices)
{
  for (int row = blockIdx.x; row < initial_scaling_view.primal_size; row += gridDim.x) {
    i_t row_offset              = A_T_offsets[row];
    i_t nnz_in_row              = A_T_offsets[row + 1] - row_offset;
    f_t constraint_scale_factor = initial_scaling_view.cummulative_variable_scaling[row];
    for (int j = threadIdx.x; j < nnz_in_row; j += blockDim.x) {
      i_t col                   = A_T_indices[row_offset + j];
      f_t variable_scale_factor = initial_scaling_view.cummulative_constraint_matrix_scaling[col];
      A_T[row_offset + j] = A_T[row_offset + j] * constraint_scale_factor * variable_scale_factor;
    }
  }
}

template <typename i_t, typename f_t>
f_t pdlp_initial_scaling_strategy_t<i_t, f_t>::get_h_bound_rescaling() const
{
  cuopt_assert(!h_bound_rescaling_.empty(), "Bound rescaling vector should not be empty");
  return h_bound_rescaling_[0];
}

template <typename i_t, typename f_t>
f_t pdlp_initial_scaling_strategy_t<i_t, f_t>::get_h_objective_rescaling() const
{
  cuopt_assert(!h_objective_rescaling_.empty(), "Objective rescaling vector should not be empty");
  return h_objective_rescaling_[0];
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>&
pdlp_initial_scaling_strategy_t<i_t, f_t>::get_bound_rescaling_vector() const
{
  return bound_rescaling_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>&
pdlp_initial_scaling_strategy_t<i_t, f_t>::get_objective_rescaling_vector() const
{
  return objective_rescaling_;
}

template <typename i_t, typename f_t>
void pdlp_initial_scaling_strategy_t<i_t, f_t>::swap_context(
  const thrust::universal_host_pinned_vector<swap_pair_t<i_t>>& swap_pairs)
{
  if (swap_pairs.empty()) { return; }

  const auto batch_size = static_cast<i_t>(bound_rescaling_.size());
  cuopt_assert(batch_size == static_cast<i_t>(objective_rescaling_.size()),
               "Rescaling vectors must have the same size");
  cuopt_assert(h_bound_rescaling_.size() == static_cast<size_t>(batch_size),
               "Host/device bound rescaling sizes must match");
  cuopt_assert(h_objective_rescaling_.size() == static_cast<size_t>(batch_size),
               "Host/device objective rescaling sizes must match");
  for (const auto& pair : swap_pairs) {
    cuopt_assert(pair.left < pair.right, "Left swap index must be less than right swap index");
    cuopt_assert(pair.right < batch_size, "Right swap index is out of bounds");
  }

  const auto [grid_size, block_size] =
    kernel_config_from_batch_size(static_cast<i_t>(swap_pairs.size()));
  scaling_swap_rescaling_kernel<i_t, f_t>
    <<<grid_size, block_size, 0, stream_view_>>>(thrust::raw_pointer_cast(swap_pairs.data()),
                                                 static_cast<i_t>(swap_pairs.size()),
                                                 make_span(bound_rescaling_),
                                                 make_span(objective_rescaling_));
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  for (const auto& pair : swap_pairs) {
    std::swap(h_bound_rescaling_[pair.left], h_bound_rescaling_[pair.right]);
    std::swap(h_objective_rescaling_[pair.left], h_objective_rescaling_[pair.right]);
  }
}

template <typename i_t, typename f_t>
void pdlp_initial_scaling_strategy_t<i_t, f_t>::resize_context(i_t new_size)
{
  [[maybe_unused]] const auto batch_size = static_cast<i_t>(bound_rescaling_.size());
  cuopt_assert(batch_size == static_cast<i_t>(objective_rescaling_.size()),
               "Rescaling vectors must have the same size");
  cuopt_assert(new_size > 0, "New size must be greater than 0");
  cuopt_assert(new_size < batch_size, "New size must be less than batch size");

  bound_rescaling_.resize(new_size, stream_view_);
  objective_rescaling_.resize(new_size, stream_view_);
  h_bound_rescaling_.resize(new_size);
  h_objective_rescaling_.resize(new_size);
  original_batch_size_ = new_size;
}

template <typename i_t, typename f_t>
void pdlp_initial_scaling_strategy_t<i_t, f_t>::scale_problem()
{
  raft::common::nvtx::range fun_scope("scale_problem");

  // scale A
  i_t number_of_blocks = op_problem_scaled_.n_constraints / block_size;
  if (op_problem_scaled_.n_constraints % block_size) number_of_blocks++;
  i_t number_of_threads = std::min(op_problem_scaled_.n_variables, block_size);
  scale_problem_kernel<i_t, f_t><<<number_of_blocks, number_of_threads, 0, stream_view_>>>(
    this->view(), op_problem_scaled_.view());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // also scale A_T in cusparse view
  i_t number_of_blocks_transposed = op_problem_scaled_.n_variables / block_size;
  if (op_problem_scaled_.n_variables % block_size) number_of_blocks_transposed++;
  i_t number_of_threads_transposed = std::min(op_problem_scaled_.n_constraints, block_size);

  scale_transposed_problem_kernel<i_t, f_t>
    <<<number_of_blocks_transposed, number_of_threads_transposed, 0, stream_view_>>>(
      this->view(), A_T_.data(), A_T_offsets_.data(), A_T_indices_.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Scale c
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(op_problem_scaled_.objective_coefficients.data(),
                          problem_wrap_container(cummulative_variable_scaling_)),
    op_problem_scaled_.objective_coefficients.data(),
    op_problem_scaled_.objective_coefficients.size(),
    cuda::std::multiplies<f_t>{},
    stream_view_);

  using f_t2 = typename type_2<f_t>::type;
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(op_problem_scaled_.variable_bounds.data(),
                          problem_wrap_container(cummulative_variable_scaling_)),
    op_problem_scaled_.variable_bounds.data(),
    op_problem_scaled_.variable_bounds.size(),
    divide_check_zero<f_t, f_t2>(),
    stream_view_.value());

  if (pdhg_solver_ptr_ && pdhg_solver_ptr_->get_new_bounds_idx().size() != 0) {
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(
        pdhg_solver_ptr_->get_new_bounds_lower().data(),
        pdhg_solver_ptr_->get_new_bounds_upper().data(),
        thrust::make_permutation_iterator(cummulative_variable_scaling_.data(),
                                          pdhg_solver_ptr_->get_new_bounds_idx().data())),
      thrust::make_zip_iterator(pdhg_solver_ptr_->get_new_bounds_lower().data(),
                                pdhg_solver_ptr_->get_new_bounds_upper().data()),
      pdhg_solver_ptr_->get_new_bounds_idx().size(),
      [] __device__(f_t lower, f_t upper, f_t s) -> thrust::tuple<f_t, f_t> {
        if (s != f_t(0)) { return {lower / s, upper / s}; }
        return {lower, upper};
      },
      stream_view_);
  }

  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(op_problem_scaled_.constraint_lower_bounds.data(),
                          problem_wrap_container(cummulative_constraint_matrix_scaling_)),
    op_problem_scaled_.constraint_lower_bounds.data(),
    op_problem_scaled_.constraint_lower_bounds.size(),
    cuda::std::multiplies<f_t>{},
    stream_view_);
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(op_problem_scaled_.constraint_upper_bounds.data(),
                          problem_wrap_container(cummulative_constraint_matrix_scaling_)),
    op_problem_scaled_.constraint_upper_bounds.data(),
    op_problem_scaled_.constraint_upper_bounds.size(),
    cuda::std::multiplies<f_t>{},
    stream_view_);

  if (hyper_params_.bound_objective_rescaling && !running_mip_) {
    // Coefficients are computed on the already scaled values
    bound_objective_rescaling();

#ifdef CUPDLP_DEBUG_MODE
    print("bound_rescaling", bound_rescaling_);
    print("objective_rescaling", objective_rescaling_);
#endif

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(op_problem_scaled_.constraint_lower_bounds.data(),
                            op_problem_scaled_.constraint_upper_bounds.data(),
                            batch_wrapped_container(bound_rescaling_, dual_size_h_)),
      thrust::make_zip_iterator(op_problem_scaled_.constraint_lower_bounds.data(),
                                op_problem_scaled_.constraint_upper_bounds.data()),
      op_problem_scaled_.constraint_upper_bounds.size(),
      [] __device__(f_t constraint_lower_bound,
                    f_t constraint_upper_bound,
                    f_t bound_rescaling) -> thrust::tuple<f_t, f_t> {
        return {constraint_lower_bound * bound_rescaling, constraint_upper_bound * bound_rescaling};
      },
      stream_view_.value());

    // In batch mode we don't scale the variable bounds (here) because they are shared across
    // climbers. While the variable bounds are the same across climbers, there can be different
    // bound rescaling factors for each climber. One solution would be to have per climber variable
    // bounds but its costly from a memory perspective and from a memory bandwidth perspective.
    // Since the variable bounds are the same across climbers but only the scaling factor changes,
    // we pass the scaling factor to PDHG later. In PDHG we act the (almost fully) scaled variable
    // bounds and add this missing scaling factor.
    if (original_batch_size_ == 1) {
      cub::DeviceTransform::Transform(
        op_problem_scaled_.variable_bounds.data(),
        op_problem_scaled_.variable_bounds.data(),
        op_problem_scaled_.variable_bounds.size(),
        [bound_rescaling = bound_rescaling_.data()] __device__(f_t2 variable_bounds) -> f_t2 {
          return {variable_bounds.x * *bound_rescaling, variable_bounds.y * *bound_rescaling};
        },
        stream_view_);
    }

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(op_problem_scaled_.objective_coefficients.data(),
                            batch_wrapped_container(objective_rescaling_, primal_size_h_)),
      op_problem_scaled_.objective_coefficients.data(),
      op_problem_scaled_.objective_coefficients.size(),
      cuda::std::multiplies<f_t>{},
      stream_view_.value());
  }

#ifdef CUPDLP_DEBUG_MODE
  print("constraint_lower_bound", op_problem_scaled_.constraint_lower_bounds);
  print("constraint_upper_bound", op_problem_scaled_.constraint_upper_bounds);
  std::vector<f_t2> variable_bounds = host_copy(op_problem_scaled_.variable_bounds, stream_view_);
  std::vector<f_t> lower_bounds;
  std::vector<f_t> upper_bounds;
  for (const auto& variable_bound : variable_bounds) {
    lower_bounds.push_back(variable_bound.x);
    upper_bounds.push_back(variable_bound.y);
  }
  print("variable_lower_bound", lower_bounds);
  print("variable_upper_bound", upper_bounds);
  print("objective_vector", op_problem_scaled_.objective_coefficients);
  if (pdhg_solver_ptr_ && pdhg_solver_ptr_->get_new_bounds_idx().size() != 0) {
    print("New bounds idx", pdhg_solver_ptr_->get_new_bounds_idx());
    print("New bounds lower", pdhg_solver_ptr_->get_new_bounds_lower());
    print("New bounds upper", pdhg_solver_ptr_->get_new_bounds_upper());
  }
#endif
  op_problem_scaled_.is_scaled_ = true;
  if (!running_mip_) {
    scale_solutions(pdhg_solver_ptr_->get_primal_solution(), pdhg_solver_ptr_->get_dual_solution());
  }
}

template <typename i_t, typename f_t>
void pdlp_initial_scaling_strategy_t<i_t, f_t>::scale_solutions(
  rmm::device_uvector<f_t>& primal_solution,
  rmm::device_uvector<f_t>& dual_solution,
  rmm::device_uvector<f_t>& dual_slack) const
{
  if (primal_solution.size()) {
    cuopt_expects(primal_solution.size() % static_cast<size_t>(primal_size_h_) == 0,
                  error_type_t::RuntimeError,
                  "Scale primal didn't get a vector of size primal");

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(
        primal_solution.data(),
        thrust::make_transform_iterator(
          thrust::make_counting_iterator(0),
          problem_wrapped_iterator<f_t>(cummulative_variable_scaling_.data(), primal_size_h_))),
      primal_solution.data(),
      primal_solution.size(),
      batch_safe_div<f_t>(),
      stream_view_);

    if (hyper_params_.bound_objective_rescaling && !running_mip_) {
      cub::DeviceTransform::Transform(
        cuda::std::make_tuple(primal_solution.data(),
                              batch_wrapped_container(bound_rescaling_, primal_size_h_)),
        primal_solution.data(),
        primal_solution.size(),
        cuda::std::multiplies<f_t>{},
        stream_view_);
    }
  }

  if (dual_solution.size()) {
    cuopt_expects(dual_solution.size() % static_cast<size_t>(dual_size_h_) == 0,
                  error_type_t::RuntimeError,
                  "Unscale dual didn't get a vector of size dual");

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(dual_solution.data(),
                            thrust::make_transform_iterator(
                              thrust::make_counting_iterator(0),
                              problem_wrapped_iterator<f_t>(
                                cummulative_constraint_matrix_scaling_.data(), dual_size_h_))),
      dual_solution.data(),
      dual_solution.size(),
      batch_safe_div<f_t>(),
      stream_view_);

    if (hyper_params_.bound_objective_rescaling && !running_mip_) {
      cub::DeviceTransform::Transform(
        cuda::std::make_tuple(dual_solution.data(),
                              batch_wrapped_container(objective_rescaling_, dual_size_h_)),
        dual_solution.data(),
        dual_solution.size(),
        cuda::std::multiplies<f_t>{},
        stream_view_);
    }
  }

  if (dual_slack.size()) {
    cuopt_expects(dual_slack.size() % static_cast<size_t>(primal_size_h_) == 0,
                  error_type_t::RuntimeError,
                  "Unscale dual didn't get a vector of size dual");

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(
        dual_slack.data(),
        thrust::make_transform_iterator(
          thrust::make_counting_iterator(0),
          problem_wrapped_iterator<f_t>(cummulative_variable_scaling_.data(), primal_size_h_))),
      dual_slack.data(),
      dual_slack.size(),
      cuda::std::multiplies<>{},
      stream_view_);

    if (hyper_params_.bound_objective_rescaling && !running_mip_) {
      cub::DeviceTransform::Transform(
        cuda::std::make_tuple(dual_slack.data(),
                              batch_wrapped_container(objective_rescaling_, primal_size_h_)),
        dual_slack.data(),
        dual_slack.size(),
        cuda::std::multiplies<f_t>{},
        stream_view_);
    }
  }
}

template <typename i_t, typename f_t>
void pdlp_initial_scaling_strategy_t<i_t, f_t>::scale_solutions(
  rmm::device_uvector<f_t>& primal_solution, rmm::device_uvector<f_t>& dual_solution) const
{
  rmm::device_uvector<f_t> dummy(0, dual_solution.stream());
  scale_solutions(primal_solution, dual_solution, dummy);
}

template <typename i_t, typename f_t>
void pdlp_initial_scaling_strategy_t<i_t, f_t>::scale_solutions(
  rmm::device_uvector<f_t>& primal_solution) const
{
  rmm::device_uvector<f_t> dummy(0, primal_solution.stream());
  scale_solutions(primal_solution, dummy);
}

template <typename i_t, typename f_t>
void pdlp_initial_scaling_strategy_t<i_t, f_t>::scale_primal(
  rmm::device_uvector<f_t>& primal_solution) const
{
  scale_solutions(primal_solution);
}

template <typename i_t, typename f_t>
void pdlp_initial_scaling_strategy_t<i_t, f_t>::scale_dual(
  rmm::device_uvector<f_t>& dual_solution) const
{
  rmm::device_uvector<f_t> dummy(0, dual_solution.stream());
  scale_solutions(dummy, dual_solution);
}

template <typename i_t, typename f_t>
void pdlp_initial_scaling_strategy_t<i_t, f_t>::unscale_solutions(
  rmm::device_uvector<f_t>& primal_solution,
  rmm::device_uvector<f_t>& dual_solution,
  rmm::device_uvector<f_t>& dual_slack) const
{
  raft::common::nvtx::range fun_scope("unscale_solutions");

  if (primal_solution.size()) {
    cuopt_expects(primal_solution.size() % static_cast<size_t>(primal_size_h_) == 0,
                  error_type_t::RuntimeError,
                  "Unscale primal didn't get a vector of size primal");
    cuopt_assert(cummulative_variable_scaling_.size() == static_cast<size_t>(primal_size_h_), "");

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(
        primal_solution.data(),
        thrust::make_transform_iterator(
          thrust::make_counting_iterator(0),
          problem_wrapped_iterator<f_t>(cummulative_variable_scaling_.data(), primal_size_h_))),
      primal_solution.data(),
      primal_solution.size(),
      cuda::std::multiplies<>{},
      stream_view_);

    if (hyper_params_.bound_objective_rescaling && !running_mip_) {
      cub::DeviceTransform::Transform(
        cuda::std::make_tuple(
          primal_solution.data(),
          thrust::make_transform_iterator(batch_wrapped_container(bound_rescaling_, primal_size_h_),
                                          inverse_rescaling_op<f_t>{})),
        primal_solution.data(),
        primal_solution.size(),
        cuda::std::multiplies<f_t>{},
        stream_view_);
    }
  }

  if (dual_solution.size()) {
    cuopt_expects(dual_solution.size() % static_cast<size_t>(dual_size_h_) == 0,
                  error_type_t::RuntimeError,
                  "Unscale dual didn't get a vector of size dual");
    cuopt_assert(cummulative_constraint_matrix_scaling_.size() == static_cast<size_t>(dual_size_h_),
                 "");
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(dual_solution.data(),
                            thrust::make_transform_iterator(
                              thrust::make_counting_iterator(0),
                              problem_wrapped_iterator<f_t>(
                                cummulative_constraint_matrix_scaling_.data(), dual_size_h_))),
      dual_solution.data(),
      dual_solution.size(),
      cuda::std::multiplies<>{},
      stream_view_);
    if (hyper_params_.bound_objective_rescaling && !running_mip_) {
      cub::DeviceTransform::Transform(
        cuda::std::make_tuple(dual_solution.data(),
                              thrust::make_transform_iterator(
                                batch_wrapped_container(objective_rescaling_, dual_size_h_),
                                inverse_rescaling_op<f_t>{})),
        dual_solution.data(),
        dual_solution.size(),
        cuda::std::multiplies<f_t>{},
        stream_view_);
    }
  }

  if (dual_slack.size()) {
    cuopt_expects(dual_slack.size() % static_cast<size_t>(primal_size_h_) == 0,
                  error_type_t::RuntimeError,
                  "Unscale dual didn't get a vector of size dual");
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(
        dual_slack.data(),
        thrust::make_transform_iterator(
          thrust::make_counting_iterator(0),
          problem_wrapped_iterator<f_t>(cummulative_variable_scaling_.data(), primal_size_h_))),
      dual_slack.data(),
      dual_slack.size(),
      batch_safe_div<f_t>(),
      stream_view_);
    if (hyper_params_.bound_objective_rescaling && !running_mip_) {
      cub::DeviceTransform::Transform(
        cuda::std::make_tuple(dual_slack.data(),
                              thrust::make_transform_iterator(
                                batch_wrapped_container(objective_rescaling_, primal_size_h_),
                                inverse_rescaling_op<f_t>{})),
        dual_slack.data(),
        dual_slack.size(),
        cuda::std::multiplies<f_t>{},
        stream_view_);
    }
  }
}

template <typename i_t, typename f_t>
void pdlp_initial_scaling_strategy_t<i_t, f_t>::unscale_solutions(
  solution_t<i_t, f_t>& solution) const
{
  auto& primal_solution = solution.assignment;
  rmm::device_uvector<f_t> dummy(0, solution.handle_ptr->get_stream());
  solution.is_scaled_ = false;
  unscale_solutions(primal_solution, dummy);
}

template <typename i_t, typename f_t>
void pdlp_initial_scaling_strategy_t<i_t, f_t>::unscale_solutions(
  rmm::device_uvector<f_t>& solution, rmm::device_uvector<f_t>& s) const
{
  rmm::device_uvector<f_t> dummy(0, solution.stream());
  unscale_solutions(solution, s, dummy);
}

template <typename i_t, typename f_t>
const problem_t<i_t, f_t>& pdlp_initial_scaling_strategy_t<i_t, f_t>::get_scaled_op_problem()
{
  return op_problem_scaled_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>&
pdlp_initial_scaling_strategy_t<i_t, f_t>::get_constraint_matrix_scaling_vector() const
{
  return cummulative_constraint_matrix_scaling_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>&
pdlp_initial_scaling_strategy_t<i_t, f_t>::get_variable_scaling_vector() const
{
  return cummulative_variable_scaling_;
}

template <typename i_t, typename f_t>
typename pdlp_initial_scaling_strategy_t<i_t, f_t>::view_t
pdlp_initial_scaling_strategy_t<i_t, f_t>::view()
{
  pdlp_initial_scaling_strategy_t<i_t, f_t>::view_t v{};

  v.primal_size = primal_size_h_;
  v.dual_size   = dual_size_h_;

  v.iteration_constraint_matrix_scaling = raft::device_span<f_t>(
    iteration_constraint_matrix_scaling_.data(), iteration_constraint_matrix_scaling_.size());
  v.iteration_variable_scaling =
    raft::device_span<f_t>(iteration_variable_scaling_.data(), iteration_variable_scaling_.size());
  v.cummulative_constraint_matrix_scaling = raft::device_span<f_t>(
    cummulative_constraint_matrix_scaling_.data(), cummulative_constraint_matrix_scaling_.size());
  v.cummulative_variable_scaling = raft::device_span<f_t>(cummulative_variable_scaling_.data(),
                                                          cummulative_variable_scaling_.size());

  return v;
}

#define INSTANTIATE(F_TYPE)                                                                   \
  template class pdlp_initial_scaling_strategy_t<int, F_TYPE>;                                \
                                                                                              \
  template __global__ void inf_norm_row_and_col_kernel<int, F_TYPE>(                          \
    const typename problem_t<int, F_TYPE>::view_t op_problem,                                 \
    typename pdlp_initial_scaling_strategy_t<int, F_TYPE>::view_t initial_scaling_view);      \
                                                                                              \
  template __global__ void pock_chambolle_scaling_kernel_col<int, F_TYPE, 128>(               \
    const typename problem_t<int, F_TYPE>::view_t op_problem,                                 \
    F_TYPE alpha,                                                                             \
    typename pdlp_initial_scaling_strategy_t<int, F_TYPE>::view_t initial_scaling_view,       \
    const F_TYPE* A_T,                                                                        \
    const int* A_T_offsets,                                                                   \
    const int* A_T_indices);                                                                  \
                                                                                              \
  template __global__ void pock_chambolle_scaling_kernel_row<int, F_TYPE, 128>(               \
    const typename problem_t<int, F_TYPE>::view_t op_problem,                                 \
    F_TYPE alpha,                                                                             \
    typename pdlp_initial_scaling_strategy_t<int, F_TYPE>::view_t initial_scaling_view);      \
                                                                                              \
  template __global__ void scale_problem_kernel<int, F_TYPE>(                                 \
    const typename pdlp_initial_scaling_strategy_t<int, F_TYPE>::view_t initial_scaling_view, \
    const typename problem_t<int, F_TYPE>::view_t op_problem);                                \
                                                                                              \
  template __global__ void scale_transposed_problem_kernel<int, F_TYPE>(                      \
    const typename pdlp_initial_scaling_strategy_t<int, F_TYPE>::view_t initial_scaling_view, \
    F_TYPE* A_T,                                                                              \
    int* A_T_offsets,                                                                         \
    int* A_T_indices);

#if MIP_INSTANTIATE_FLOAT || PDLP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

}  // namespace cuopt::linear_programming::detail
