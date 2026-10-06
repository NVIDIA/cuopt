/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <pdlp/reduced_matrix.cuh>

#include <mip_heuristics/mip_constants.hpp>
#include <utilities/device_scalar_init.hpp>
#include <utilities/logger.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/sparse/linalg/transpose.cuh>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <cub/device/device_for.cuh>

#include <algorithm>
#include <cstdint>

namespace cuopt::mathematical_optimization::pdlp {
namespace {

constexpr uint8_t free_column = 0;
constexpr uint8_t at_lower    = 1;
constexpr uint8_t at_upper    = 2;
constexpr double enter_kkt    = 1e-2;
constexpr double enter_ratio  = 0.5;
constexpr double fast_ratio   = 0.25;
constexpr int stable_checks   = 3;

template <typename f_t>
struct initialize_mask_op {
  using f_t2 = typename type_2<f_t>::type;

  __device__ uint8_t operator()(f_t value, f_t2 bounds) const
  {
    return value == bounds.x ? at_lower : (value == bounds.y ? at_upper : free_column);
  }
};

template <typename f_t>
struct release_mask_op {
  using f_t2 = typename type_2<f_t>::type;

  __device__ uint8_t operator()(uint8_t old_mask, f_t value, f_t2 bounds) const
  {
    if (old_mask == at_lower && value != bounds.x) { return free_column; }
    if (old_mask == at_upper && value != bounds.y) { return free_column; }
    return old_mask;
  }
};

struct is_free_column {
  __device__ bool operator()(uint8_t mask) const { return mask == free_column; }
};

template <typename i_t>
__global__ void selected_row_lengths_kernel(const i_t* source_offsets,
                                            const i_t* selected_to_original,
                                            i_t* selected_offsets,
                                            i_t selected_count)
{
  const i_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < selected_count) {
    const i_t source_row      = selected_to_original[row];
    selected_offsets[row + 1] = source_offsets[source_row + 1] - source_offsets[source_row];
  }
}

template <typename i_t, typename f_t>
__global__ void copy_selected_rows_kernel(const i_t* source_offsets,
                                          const i_t* source_indices,
                                          const f_t* source_values,
                                          const i_t* selected_to_original,
                                          const i_t* selected_offsets,
                                          i_t* selected_indices,
                                          f_t* selected_values,
                                          i_t selected_count)
{
  const i_t selected_row = blockIdx.x;
  if (selected_row >= selected_count) { return; }
  const i_t source_row = selected_to_original[selected_row];
  const i_t source_beg = source_offsets[source_row];
  const i_t source_end = source_offsets[source_row + 1];
  const i_t target_beg = selected_offsets[selected_row];
  for (i_t source = source_beg + threadIdx.x; source < source_end; source += blockDim.x) {
    const i_t target         = target_beg + source - source_beg;
    selected_indices[target] = source_indices[source];
    selected_values[target]  = source_values[source];
  }
}

template <typename i_t, typename f_t>
__global__ void fixed_activities_kernel(const i_t* offsets,
                                        const i_t* columns,
                                        const f_t* values,
                                        const uint8_t* mask,
                                        const typename type_2<f_t>::type* bounds,
                                        const f_t* current,
                                        const f_t* initial,
                                        f_t* bound_activity,
                                        f_t* current_activity,
                                        f_t* initial_activity,
                                        i_t rows)
{
  const i_t row = blockIdx.x;
  if (row >= rows) { return; }

  extern __shared__ unsigned char shared_bytes[];
  auto* bound_sum   = reinterpret_cast<f_t*>(shared_bytes);
  auto* current_sum = bound_sum + blockDim.x;
  auto* initial_sum = current_sum + blockDim.x;
  f_t local_bound   = 0;
  f_t local_current = 0;
  f_t local_initial = 0;
  for (i_t entry = offsets[row] + threadIdx.x; entry < offsets[row + 1]; entry += blockDim.x) {
    const i_t column          = columns[entry];
    const uint8_t column_mask = mask[column];
    if (column_mask != free_column) {
      const f_t coefficient    = values[entry];
      const auto column_bounds = bounds[column];
      const f_t bound          = column_mask == at_lower ? column_bounds.x : column_bounds.y;
      local_bound              = fma(coefficient, bound, local_bound);
      local_current            = fma(coefficient, current[column], local_current);
      local_initial            = fma(coefficient, initial[column], local_initial);
    }
  }
  bound_sum[threadIdx.x]   = local_bound;
  current_sum[threadIdx.x] = local_current;
  initial_sum[threadIdx.x] = local_initial;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      bound_sum[threadIdx.x] += bound_sum[threadIdx.x + offset];
      current_sum[threadIdx.x] += current_sum[threadIdx.x + offset];
      initial_sum[threadIdx.x] += initial_sum[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    bound_activity[row]   = bound_sum[0];
    current_activity[row] = current_sum[0];
    initial_activity[row] = initial_sum[0];
  }
}

template <typename i_t, typename f_t>
struct materialize_fixed_primal_op {
  using f_t2 = typename type_2<f_t>::type;
  const uint8_t* mask;
  const f_t2* bounds;
  f_t* current;
  const f_t* initial;
  const f_t* alpha;
  const f_t* beta;

  __device__ void operator()(i_t column) const
  {
    const auto column_mask = mask[column];
    if (column_mask == free_column) { return; }
    const auto column_bounds = bounds[column];
    const f_t bound          = column_mask == at_lower ? column_bounds.x : column_bounds.y;
    current[column] =
      bound + *alpha * (current[column] - bound) + *beta * (initial[column] - bound);
  }
};

template <typename i_t, typename f_t>
struct compact_primal_op {
  f_t* current;
  const f_t* initial;
  const f_t* objective;
  const typename type_2<f_t>::type* bounds;
  const f_t* At_y;
  const f_t* step_size;
  const f_t* weight;
  f_t* reflected;

  __device__ void operator()(i_t column) const
  {
    const f_t next            = current[column] - *step_size * (objective[column] - At_y[column]);
    const auto bound          = bounds[column];
    const f_t projected       = raft::max<f_t>(raft::min<f_t>(next, bound.y), bound.x);
    const f_t reflected_value = f_t(2) * projected - current[column];
    reflected[column]         = reflected_value;
    current[column]           = *weight * reflected_value + (f_t(1) - *weight) * initial[column];
  }
};

template <typename f_t>
__global__ void advance_fixed_coefficients(f_t* alpha, f_t* beta, const f_t* weight)
{
  *alpha = -*weight * *alpha;
  *beta  = (f_t(1) - *weight) - *weight * *beta;
}

template <typename i_t, typename f_t>
struct gather_free_op {
  const i_t* free_to_original;
  const f_t* source;
  f_t* destination;
  __device__ void operator()(i_t column) const
  {
    destination[column] = source[free_to_original[column]];
  }
};

template <typename i_t, typename f_t>
struct scatter_free_op {
  const i_t* free_to_original;
  const f_t* source;
  f_t* destination;
  __device__ void operator()(i_t column) const
  {
    destination[free_to_original[column]] = source[column];
  }
};

template <typename i_t, typename f_t>
struct add_fixed_reflected_activity_op {
  f_t* output;
  const f_t* bound_activity;
  f_t* current_activity;
  const f_t* initial_activity;
  const f_t* halpern_weight;

  __device__ void operator()(i_t row) const
  {
    const f_t reflected = f_t(2) * bound_activity[row] - current_activity[row];
    output[row] += reflected;
    current_activity[row] =
      *halpern_weight * reflected + (f_t(1) - *halpern_weight) * initial_activity[row];
  }
};

}  // namespace

template <typename i_t, typename f_t>
reduced_matrix_t<i_t, f_t>::reduced_matrix_t(raft::handle_t const* handle_ptr,
                                             mip::problem_t<i_t, f_t>& problem,
                                             bool enabled)
  : handle_ptr_(handle_ptr),
    stream_view_(handle_ptr->get_stream()),
    problem_ptr_(&problem),
    structure_problem_ptr_(&problem),
    enabled_(enabled),
    mask_(problem.n_variables, stream_view_),
    free_to_original_(0, stream_view_),
    reduced_At_offsets_(0, stream_view_),
    reduced_At_indices_(0, stream_view_),
    reduced_At_values_(0, stream_view_),
    reduced_A_offsets_(0, stream_view_),
    reduced_A_indices_(0, stream_view_),
    reduced_A_values_(0, stream_view_),
    reduced_At_y_(0, stream_view_),
    reduced_x_(0, stream_view_),
    reduced_current_(0, stream_view_),
    reduced_initial_(0, stream_view_),
    reduced_objective_(0, stream_view_),
    reduced_bounds_(0, stream_view_),
    fixed_alpha_(one_v<f_t>, stream_view_),
    fixed_beta_(zero_v<f_t>, stream_view_),
    fixed_bound_activity_(problem.n_constraints, stream_view_),
    fixed_current_activity_(problem.n_constraints, stream_view_),
    fixed_initial_activity_(problem.n_constraints, stream_view_),
    A_buffer_(0, stream_view_),
    At_buffer_(0, stream_view_),
    one_(one_v<f_t>, stream_view_),
    zero_(zero_v<f_t>, stream_view_)
{
  thrust::fill(handle_ptr_->get_thrust_policy(), mask_.begin(), mask_.end(), free_column);
}

template <typename i_t, typename f_t>
void reduced_matrix_t<i_t, f_t>::initialize_mask(const rmm::device_uvector<f_t>& projected_primal)
{
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(projected_primal.data(), problem_ptr_->variable_bounds.data()),
    mask_.data(),
    problem_ptr_->n_variables,
    initialize_mask_op<f_t>{},
    stream_view_.get());
  mask_started_ = true;
}

template <typename i_t, typename f_t>
void reduced_matrix_t<i_t, f_t>::release_mask(const rmm::device_uvector<f_t>& projected_primal)
{
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(
      mask_.data(), projected_primal.data(), problem_ptr_->variable_bounds.data()),
    mask_.data(),
    problem_ptr_->n_variables,
    release_mask_op<f_t>{},
    stream_view_.get());
}

template <typename i_t, typename f_t>
bool reduced_matrix_t<i_t, f_t>::update_mode(f_t relative_kkt,
                                             bool restarted,
                                             const rmm::device_uvector<f_t>& projected_primal,
                                             const rmm::device_uvector<f_t>& current_primal,
                                             const rmm::device_uvector<f_t>& restart_primal)
{
  if (!enabled_) { return false; }
  if (!active_ && !refresh_pending_ && !(relative_kkt < static_cast<f_t>(enter_kkt))) {
    stable_checks_   = 0;
    mask_started_    = false;
    last_free_count_ = -1;
    clear_operator();
    return false;
  }

  if (active_) {
    if (restarted) { build_fixed_activities(current_primal, restart_primal); }
    return false;
  }
  if (refresh_pending_) { return false; }

  if (!mask_started_ || restarted) {
    initialize_mask(projected_primal);
  } else {
    release_mask(projected_primal);
  }
  free_count_ = static_cast<i_t>(
    thrust::count(handle_ptr_->get_thrust_policy(), mask_.begin(), mask_.end(), free_column));
  if (free_count_ == 0) { return false; }
  const f_t ratio  = problem_ptr_->n_variables > 0
                       ? static_cast<f_t>(free_count_) / problem_ptr_->n_variables
                       : f_t(1);
  stable_checks_   = free_count_ == last_free_count_ ? stable_checks_ + 1 : 0;
  last_free_count_ = free_count_;
  if (ratio >= static_cast<f_t>(enter_ratio) ||
      (stable_checks_ < stable_checks && ratio >= static_cast<f_t>(fast_ratio))) {
    return false;
  }
  return rebuild(projected_primal, current_primal, restart_primal, false);
}

template <typename i_t, typename f_t>
bool reduced_matrix_t<i_t, f_t>::request_refresh(bool update_mask,
                                                 rmm::device_uvector<f_t>& current_primal,
                                                 const rmm::device_uvector<f_t>& restart_primal)
{
  if (!active_) { return false; }
  materialize_primal(current_primal, restart_primal);
  active_               = false;
  refresh_pending_      = true;
  refresh_mask_pending_ = update_mask;
  return true;
}

template <typename i_t, typename f_t>
bool reduced_matrix_t<i_t, f_t>::finish_refresh(const rmm::device_uvector<f_t>& projected_primal,
                                                const rmm::device_uvector<f_t>& current_primal,
                                                const rmm::device_uvector<f_t>& restart_primal)
{
  if (!refresh_pending_) { return false; }
  refresh_pending_ = false;
  if (!refresh_mask_pending_) {
    build_fixed_activities(current_primal, restart_primal);
    active_ = true;
    return true;
  }
  refresh_mask_pending_         = false;
  const i_t previous_free_count = free_count_;
  release_mask(projected_primal);
  free_count_ = static_cast<i_t>(
    thrust::count(handle_ptr_->get_thrust_policy(), mask_.begin(), mask_.end(), free_column));
  const f_t ratio = problem_ptr_->n_variables > 0
                      ? static_cast<f_t>(free_count_) / problem_ptr_->n_variables
                      : f_t(1);
  if (ratio >= static_cast<f_t>(enter_ratio)) {
    clear_operator();
    active_ = false;
    return true;
  }
  if (free_count_ == previous_free_count) {
    build_fixed_activities(current_primal, restart_primal);
    active_ = true;
    return true;
  }
  rebuild(projected_primal, current_primal, restart_primal, false);
  return true;
}

template <typename i_t, typename f_t>
bool reduced_matrix_t<i_t, f_t>::rebuild(const rmm::device_uvector<f_t>& projected_primal,
                                         const rmm::device_uvector<f_t>& current_primal,
                                         const rmm::device_uvector<f_t>& restart_primal,
                                         bool initialize)
{
  if (initialize) { initialize_mask(projected_primal); }
  clear_operator();
  build_compact_operator();
  build_fixed_activities(current_primal, restart_primal);
  initialize_spmv();
  active_ = true;
  CUOPT_LOG_INFO("Reduced matrix activated: columns=%lld/%lld nnz=%lld/%lld",
                 static_cast<long long>(free_count_),
                 static_cast<long long>(problem_ptr_->n_variables),
                 static_cast<long long>(reduced_At_values_.size()),
                 static_cast<long long>(problem_ptr_->nnz));
  return true;
}

template <typename i_t, typename f_t>
void reduced_matrix_t<i_t, f_t>::clear_operator()
{
  reduced_A_descr_.reset();
  reduced_At_descr_.reset();
  dual_descr_.reset();
  reduced_At_y_descr_.reset();
  reduced_x_descr_.reset();
  dual_gradient_descr_.reset();
  free_to_original_.resize(0, stream_view_);
  reduced_At_offsets_.resize(0, stream_view_);
  reduced_At_indices_.resize(0, stream_view_);
  reduced_At_values_.resize(0, stream_view_);
  reduced_A_offsets_.resize(0, stream_view_);
  reduced_A_indices_.resize(0, stream_view_);
  reduced_A_values_.resize(0, stream_view_);
  reduced_At_y_.resize(0, stream_view_);
  reduced_x_.resize(0, stream_view_);
  reduced_current_.resize(0, stream_view_);
  reduced_initial_.resize(0, stream_view_);
  reduced_objective_.resize(0, stream_view_);
  reduced_bounds_.resize(0, stream_view_);
  A_buffer_.resize(0, stream_view_);
  At_buffer_.resize(0, stream_view_);
}

template <typename i_t, typename f_t>
void reduced_matrix_t<i_t, f_t>::build_compact_operator()
{
  free_to_original_.resize(free_count_, stream_view_);
  auto first = thrust::make_counting_iterator<i_t>(0);
  thrust::copy_if(handle_ptr_->get_thrust_policy(),
                  first,
                  first + problem_ptr_->n_variables,
                  mask_.begin(),
                  free_to_original_.begin(),
                  is_free_column{});

  reduced_At_offsets_.resize(static_cast<size_t>(free_count_) + 1, stream_view_);
  RAFT_CUDA_TRY(cudaMemsetAsync(reduced_At_offsets_.data(), 0, sizeof(i_t), stream_view_.get()));
  constexpr int block_size = 256;
  selected_row_lengths_kernel<<<(free_count_ + block_size - 1) / block_size,
                                block_size,
                                0,
                                stream_view_.get()>>>(
    structure_problem_ptr_->reverse_offsets.data(),
    free_to_original_.data(),
    reduced_At_offsets_.data(),
    free_count_);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  thrust::inclusive_scan(handle_ptr_->get_thrust_policy(),
                         reduced_At_offsets_.begin() + 1,
                         reduced_At_offsets_.end(),
                         reduced_At_offsets_.begin() + 1);
  const i_t reduced_nnz = reduced_At_offsets_.element(free_count_, stream_view_);
  reduced_At_indices_.resize(reduced_nnz, stream_view_);
  reduced_At_values_.resize(reduced_nnz, stream_view_);
  if (free_count_ > 0 && reduced_nnz > 0) {
    copy_selected_rows_kernel<<<free_count_, block_size, 0, stream_view_.get()>>>(
      structure_problem_ptr_->reverse_offsets.data(),
      structure_problem_ptr_->reverse_constraints.data(),
      problem_ptr_->reverse_coefficients.data(),
      free_to_original_.data(),
      reduced_At_offsets_.data(),
      reduced_At_indices_.data(),
      reduced_At_values_.data(),
      free_count_);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  reduced_A_offsets_.resize(static_cast<size_t>(problem_ptr_->n_constraints) + 1, stream_view_);
  reduced_A_indices_.resize(reduced_nnz, stream_view_);
  reduced_A_values_.resize(reduced_nnz, stream_view_);
  if (reduced_nnz == 0) {
    RAFT_CUDA_TRY(cudaMemsetAsync(
      reduced_A_offsets_.data(), 0, sizeof(i_t) * reduced_A_offsets_.size(), stream_view_.get()));
  } else {
    raft::sparse::linalg::csr_transpose(*handle_ptr_,
                                        reduced_At_offsets_.data(),
                                        reduced_At_indices_.data(),
                                        reduced_At_values_.data(),
                                        reduced_A_offsets_.data(),
                                        reduced_A_indices_.data(),
                                        reduced_A_values_.data(),
                                        free_count_,
                                        problem_ptr_->n_constraints,
                                        reduced_nnz,
                                        stream_view_.get());
  }
  reduced_At_y_.resize(free_count_, stream_view_);
  reduced_x_.resize(free_count_, stream_view_);
  reduced_current_.resize(free_count_, stream_view_);
  reduced_initial_.resize(free_count_, stream_view_);
  reduced_objective_.resize(free_count_, stream_view_);
  reduced_bounds_.resize(free_count_, stream_view_);
  cub::DeviceFor::Bulk(free_count_,
                       gather_free_op<i_t, f_t>{free_to_original_.data(),
                                                problem_ptr_->objective_coefficients.data(),
                                                reduced_objective_.data()},
                       stream_view_.get());
  cub::DeviceFor::Bulk(
    free_count_,
    gather_free_op<i_t, typename type_2<f_t>::type>{
      free_to_original_.data(), problem_ptr_->variable_bounds.data(), reduced_bounds_.data()},
    stream_view_.get());
}

template <typename i_t, typename f_t>
void reduced_matrix_t<i_t, f_t>::build_fixed_activities(
  const rmm::device_uvector<f_t>& current_primal, const rmm::device_uvector<f_t>& initial_primal)
{
  fixed_alpha_.set_value_async(one_v<f_t>, stream_view_);
  fixed_beta_.set_value_async(zero_v<f_t>, stream_view_);
  cub::DeviceFor::Bulk(free_count_,
                       gather_free_op<i_t, f_t>{
                         free_to_original_.data(), current_primal.data(), reduced_current_.data()},
                       stream_view_.get());
  cub::DeviceFor::Bulk(free_count_,
                       gather_free_op<i_t, f_t>{
                         free_to_original_.data(), initial_primal.data(), reduced_initial_.data()},
                       stream_view_.get());
  constexpr int block_size = 256;
  const size_t shared_size = 3 * block_size * sizeof(f_t);
  fixed_activities_kernel<<<problem_ptr_->n_constraints,
                            block_size,
                            shared_size,
                            stream_view_.get()>>>(structure_problem_ptr_->offsets.data(),
                                                  structure_problem_ptr_->variables.data(),
                                                  problem_ptr_->coefficients.data(),
                                                  mask_.data(),
                                                  problem_ptr_->variable_bounds.data(),
                                                  current_primal.data(),
                                                  initial_primal.data(),
                                                  fixed_bound_activity_.data(),
                                                  fixed_current_activity_.data(),
                                                  fixed_initial_activity_.data(),
                                                  problem_ptr_->n_constraints);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename i_t, typename f_t>
void reduced_matrix_t<i_t, f_t>::initialize_spmv()
{
  reduced_At_descr_    = make_csr<i_t, f_t>(free_count_,
                                         problem_ptr_->n_constraints,
                                         reduced_At_values_.size(),
                                         reduced_At_offsets_.data(),
                                         reduced_At_indices_.data(),
                                         reduced_At_values_.data());
  reduced_A_descr_     = make_csr<i_t, f_t>(problem_ptr_->n_constraints,
                                        free_count_,
                                        reduced_A_values_.size(),
                                        reduced_A_offsets_.data(),
                                        reduced_A_indices_.data(),
                                        reduced_A_values_.data());
  dual_descr_          = make_dnvec<f_t>(problem_ptr_->n_constraints, fixed_bound_activity_.data());
  reduced_At_y_descr_  = make_dnvec<f_t>(free_count_, reduced_At_y_.data());
  reduced_x_descr_     = make_dnvec<f_t>(free_count_, reduced_x_.data());
  dual_gradient_descr_ = make_dnvec<f_t>(problem_ptr_->n_constraints, fixed_bound_activity_.data());

  size_t At_buffer_size = 0;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv_buffersize(handle_ptr_->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  one_.data(),
                                                  reduced_At_descr_.get(),
                                                  dual_descr_.get(),
                                                  zero_.data(),
                                                  reduced_At_y_descr_.get(),
                                                  CUSPARSE_SPMV_CSR_ALG2,
                                                  &At_buffer_size,
                                                  stream_view_.get()));
  At_buffer_.resize(At_buffer_size, stream_view_);

  size_t A_buffer_size = 0;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv_buffersize(handle_ptr_->get_cusparse_handle(),
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  one_.data(),
                                                  reduced_A_descr_.get(),
                                                  reduced_x_descr_.get(),
                                                  zero_.data(),
                                                  dual_gradient_descr_.get(),
                                                  CUSPARSE_SPMV_CSR_ALG2,
                                                  &A_buffer_size,
                                                  stream_view_.get()));
  A_buffer_.resize(A_buffer_size, stream_view_);
}

template <typename i_t, typename f_t>
void reduced_matrix_t<i_t, f_t>::run_spmv(cusparse_sp_mat_descr_view matrix,
                                          cusparse_dn_vec_descr_view input,
                                          cusparse_dn_vec_descr_view output,
                                          rmm::device_uvector<uint8_t>& buffer)
{
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       one_.data(),
                                                       matrix,
                                                       input,
                                                       zero_.data(),
                                                       output,
                                                       CUSPARSE_SPMV_CSR_ALG2,
                                                       reinterpret_cast<f_t*>(buffer.data()),
                                                       stream_view_.get()));
}

template <typename i_t, typename f_t>
void reduced_matrix_t<i_t, f_t>::materialize_primal(rmm::device_uvector<f_t>& current_primal,
                                                    const rmm::device_uvector<f_t>& initial_primal)
{
  cub::DeviceFor::Bulk(problem_ptr_->n_variables,
                       materialize_fixed_primal_op<i_t, f_t>{mask_.data(),
                                                             problem_ptr_->variable_bounds.data(),
                                                             current_primal.data(),
                                                             initial_primal.data(),
                                                             fixed_alpha_.data(),
                                                             fixed_beta_.data()},
                       stream_view_.get());
  cub::DeviceFor::Bulk(free_count_,
                       scatter_free_op<i_t, f_t>{
                         free_to_original_.data(), reduced_current_.data(), current_primal.data()},
                       stream_view_.get());
}

template <typename i_t, typename f_t>
void reduced_matrix_t<i_t, f_t>::compute_At_y(const rmm::device_uvector<f_t>& dual)
{
  RAFT_CUSPARSE_TRY(cusparseDnVecSetValues(dual_descr_.get(), const_cast<f_t*>(dual.data())));
  run_spmv(reduced_At_descr_.get(), dual_descr_.get(), reduced_At_y_descr_.get(), At_buffer_);
}

template <typename i_t, typename f_t>
void reduced_matrix_t<i_t, f_t>::update_primal(const rmm::device_uvector<f_t>& primal_step_size,
                                               const rmm::device_scalar<f_t>& halpern_weight)
{
  cub::DeviceFor::Bulk(free_count_,
                       compact_primal_op<i_t, f_t>{reduced_current_.data(),
                                                   reduced_initial_.data(),
                                                   reduced_objective_.data(),
                                                   reduced_bounds_.data(),
                                                   reduced_At_y_.data(),
                                                   primal_step_size.data(),
                                                   halpern_weight.data(),
                                                   reduced_x_.data()},
                       stream_view_.get());
}

template <typename i_t, typename f_t>
void reduced_matrix_t<i_t, f_t>::compute_A_x(rmm::device_uvector<f_t>& dual_gradient,
                                             const rmm::device_scalar<f_t>& halpern_weight)
{
  RAFT_CUSPARSE_TRY(cusparseDnVecSetValues(dual_gradient_descr_.get(), dual_gradient.data()));
  run_spmv(reduced_A_descr_.get(), reduced_x_descr_.get(), dual_gradient_descr_.get(), A_buffer_);
  cub::DeviceFor::Bulk(problem_ptr_->n_constraints,
                       add_fixed_reflected_activity_op<i_t, f_t>{dual_gradient.data(),
                                                                 fixed_bound_activity_.data(),
                                                                 fixed_current_activity_.data(),
                                                                 fixed_initial_activity_.data(),
                                                                 halpern_weight.data()},
                       stream_view_.get());
  advance_fixed_coefficients<<<1, 1, 0, stream_view_.get()>>>(
    fixed_alpha_.data(), fixed_beta_.data(), halpern_weight.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

#if MIP_INSTANTIATE_FLOAT || PDLP_INSTANTIATE_FLOAT
template class reduced_matrix_t<int, float>;
#endif
#if MIP_INSTANTIATE_DOUBLE
template class reduced_matrix_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::pdlp
