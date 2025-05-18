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

#include <cuopt/error.hpp>

#include <linear_programming/pdlp_constants.hpp>
#include <linear_programming/restart_strategy/weighted_average_solution.hpp>
#include <linear_programming/utils.cuh>
#include <mip/mip_constants.hpp>

#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/divide.cuh>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
weighted_average_solution_t<i_t, f_t>::weighted_average_solution_t(raft::handle_t const* handle_ptr,
                                                                   i_t primal_size,
                                                                   i_t dual_size)
  : handle_ptr_(handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    primal_size_h_(primal_size),
    dual_size_h_(dual_size),
    sum_primal_solutions_{static_cast<size_t>(primal_size_h_), stream_view_},
    sum_dual_solutions_{static_cast<size_t>(dual_size_h_), stream_view_},
    sum_primal_solution_weights_{0.0, stream_view_},
    sum_dual_solution_weights_{0.0, stream_view_},
    iterations_since_last_restart_{0},
    graph(stream_view_)
{
  RAFT_CUDA_TRY(
    cudaMemsetAsync(sum_primal_solutions_.data(), 0.0, sizeof(f_t) * primal_size_h_, stream_view_));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(sum_dual_solutions_.data(), 0.0, sizeof(f_t) * dual_size_h_, stream_view_));
}

template <typename i_t, typename f_t>
void weighted_average_solution_t<i_t, f_t>::reset_weighted_average_solution()
{
  RAFT_CUDA_TRY(
    cudaMemsetAsync(sum_primal_solutions_.data(), 0.0, sizeof(f_t) * primal_size_h_, stream_view_));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(sum_dual_solutions_.data(), 0.0, sizeof(f_t) * dual_size_h_, stream_view_));
  sum_primal_solution_weights_.set_value_to_zero_async(stream_view_);
  sum_dual_solution_weights_.set_value_to_zero_async(stream_view_);
  iterations_since_last_restart_ = 0;
}

template <typename f_t>
__global__ void add_weight_sums(const f_t* primal_weight,
                                const f_t* dual_weight,
                                f_t* sum_primal_solution_weights,
                                f_t* sum_dual_solution_weights)
{
  *sum_primal_solution_weights += *primal_weight;
  *sum_dual_solution_weights += *dual_weight;
}

template <typename i_t, typename f_t>
void weighted_average_solution_t<i_t, f_t>::add_current_solution_to_weighted_average_solution(
  const f_t* primal_solution,
  const f_t* dual_solution,
  const rmm::device_scalar<f_t>& weight,
  i_t total_pdlp_iterations)
{
  // primalavg += primal_sol*weight     -- weight is just set to be step_size for the new solution
  // (same for primal and dual although julia repo makes it seem as though these should/could be
  // different)

  if (!graph.is_initialized(total_pdlp_iterations)) {
    graph.start_capture(total_pdlp_iterations);

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(sum_primal_solutions_.data(), primal_solution),
      sum_primal_solutions_.data(),
      primal_size_h_,
      a_add_scalar_times_b<f_t>(weight.data()),
      stream_view_);

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(sum_dual_solutions_.data(), dual_solution),
      sum_dual_solutions_.data(),
      dual_size_h_,
      a_add_scalar_times_b<f_t>(weight.data()),
      stream_view_);

    // update weight sums and count (add weight and +1 respectively)
    add_weight_sums<<<1, 1, 0, stream_view_>>>(weight.data(),
                                               weight.data(),
                                               sum_primal_solution_weights_.data(),
                                               sum_dual_solution_weights_.data());

    graph.end_capture(total_pdlp_iterations);
  }
  graph.launch(total_pdlp_iterations);

  iterations_since_last_restart_ += 1;
}

template <typename i_t, typename f_t>
void weighted_average_solution_t<i_t, f_t>::compute_averages(rmm::device_uvector<f_t>& avg_primal,
                                                             rmm::device_uvector<f_t>& avg_dual)
{
  // no iterations have added to the sum, so avg is all zero vector
  if (!iterations_since_last_restart_) {
    RAFT_CUDA_TRY(
      cudaMemsetAsync(avg_primal.data(), f_t(0.0), sizeof(f_t) * primal_size_h_, stream_view_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(avg_dual.data(), f_t(0.0), sizeof(f_t) * dual_size_h_, stream_view_));
    return;
  }

  // return weight sums to host to fit API call
  f_t sum_primal_solution_weights_h = sum_primal_solution_weights_.value(stream_view_);
  f_t sum_dual_solution_weights_h   = sum_dual_solution_weights_.value(stream_view_);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));

  // compute sum_primal_solutions/primal_size
  raft::linalg::divideScalar(avg_primal.data(),
                             sum_primal_solutions_.data(),
                             sum_primal_solution_weights_h,
                             primal_size_h_,
                             stream_view_);
  raft::linalg::divideScalar(avg_dual.data(),
                             sum_dual_solutions_.data(),
                             sum_dual_solution_weights_h,
                             dual_size_h_,
                             stream_view_);
}

template <typename i_t, typename f_t>
i_t weighted_average_solution_t<i_t, f_t>::get_iterations_since_last_restart() const
{
  return iterations_since_last_restart_;
}

#if MIP_INSTANTIATE_FLOAT
template __global__ void add_weight_sums<float>(const float* primal_weight,
                                                const float* dual_weight,
                                                float* sum_primal_solution_weights,
                                                float* sum_dual_solution_weights);

template class weighted_average_solution_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template __global__ void add_weight_sums<double>(const double* primal_weight,
                                                 const double* dual_weight,
                                                 double* sum_primal_solution_weights,
                                                 double* sum_dual_solution_weights);

template class weighted_average_solution_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
