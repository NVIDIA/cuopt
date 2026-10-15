/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/copy_helpers.hpp>

#include <cub/device/device_reduce.cuh>

#include <cuda/stream>
#include <raft/core/device_span.hpp>
#include <raft/core/error.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <cmath>
#include <vector>
namespace cuopt::mathematical_optimization {

// Reuse scratch storage for reductions submitted sequentially on the same stream.
template <typename f_t>
struct reduction_workspace_t {
  rmm::device_buffer buffer_data;
  rmm::device_scalar<f_t> out;
  size_t buffer_size = 0;

  reduction_workspace_t(cuda::stream_ref stream_view)
    : buffer_data(0, stream_view), out(stream_view)
  {
  }

  template <typename InputIteratorT, typename ReductionOpT, typename TransformOpT, typename i_t>
  void transform_reduce_async(InputIteratorT input,
                              ReductionOpT reduce_op,
                              TransformOpT transform_op,
                              f_t init,
                              i_t size,
                              raft::device_span<f_t> output,
                              cuda::stream_ref stream_view)
  {
    RAFT_EXPECTS(output.size() == 1, "Reduction output must contain exactly one element");
    RAFT_CUDA_TRY(cub::DeviceReduce::TransformReduce(nullptr,
                                                     buffer_size,
                                                     input,
                                                     output.data(),
                                                     size,
                                                     reduce_op,
                                                     transform_op,
                                                     init,
                                                     stream_view.get()));
    buffer_data.resize(buffer_size, stream_view);
    RAFT_CUDA_TRY(cub::DeviceReduce::TransformReduce(buffer_data.data(),
                                                     buffer_size,
                                                     input,
                                                     output.data(),
                                                     size,
                                                     reduce_op,
                                                     transform_op,
                                                     init,
                                                     stream_view.get()));
  }

  template <typename InputIteratorT, typename ReductionOpT, typename TransformOpT, typename i_t>
  f_t transform_reduce(InputIteratorT input,
                       ReductionOpT reduce_op,
                       TransformOpT transform_op,
                       f_t init,
                       i_t size,
                       cuda::stream_ref stream_view)
  {
    transform_reduce_async(input,
                           reduce_op,
                           transform_op,
                           init,
                           size,
                           raft::device_span<f_t>(out.data(), 1),
                           stream_view);
    return out.value(stream_view);
  }
};

// All GPU infinity norms share this reduction, including custom iterators.
template <typename f_t, typename InputIteratorT, typename i_t>
void vector_norm_inf_async(InputIteratorT input,
                           i_t size,
                           raft::device_span<f_t> output,
                           reduction_workspace_t<f_t>& reductions,
                           cuda::stream_ref stream_view)
{
  reductions.transform_reduce_async(
    input,
    thrust::maximum<f_t>{},
    [] __host__ __device__(f_t val) { return abs(val); },
    f_t(0),
    size,
    output,
    stream_view);
}

template <typename f_t>
void vector_norm_inf_async(raft::device_span<const f_t> input,
                           raft::device_span<f_t> output,
                           reduction_workspace_t<f_t>& reductions,
                           cuda::stream_ref stream_view)
{
  vector_norm_inf_async(input.data(), input.size(), output, reductions, stream_view);
}

template <typename i_t, typename f_t, typename InputIteratorT>
f_t device_custom_vector_norm_inf(InputIteratorT in, i_t size, cuda::stream_ref stream_view)
{
  if (size == 0) { return 0; }
  reduction_workspace_t<f_t> reductions(stream_view);
  vector_norm_inf_async(
    in, size, raft::device_span<f_t>(reductions.out.data(), 1), reductions, stream_view);
  return reductions.out.value(stream_view);
}

template <typename i_t, typename f_t>
f_t device_vector_norm_inf(const rmm::device_uvector<f_t>& in, cuda::stream_ref stream_view)
{
  return device_custom_vector_norm_inf<i_t, f_t>(in.data(), in.size(), stream_view);
}

// TMP we should just have a CPU and GPU version to do the comparison
// Should never have to norm inf a CPU vector if we are using the GPU
template <typename i_t, typename f_t, typename Allocator>
f_t vector_norm_inf(const std::vector<f_t, Allocator>& x, cuda::stream_ref stream_view)
{
  const auto d_x = device_copy(x, stream_view);
  return device_vector_norm_inf<i_t, f_t>(d_x, stream_view);
}

template <typename f_t>
f_t vector_norm_inf(const rmm::device_uvector<f_t>& x, reduction_workspace_t<f_t>& reductions)
{
  vector_norm_inf_async(raft::device_span<const f_t>(x.data(), x.size()),
                        raft::device_span<f_t>(reductions.out.data(), 1),
                        reductions,
                        x.stream());
  return reductions.out.value(x.stream());
}

template <typename f_t>
f_t vector_norm_inf(const rmm::device_uvector<f_t>& x)
{
  reduction_workspace_t<f_t> reductions(x.stream());
  return vector_norm_inf(x, reductions);
}

template <typename f_t>
f_t vector_norm2(const rmm::device_uvector<f_t>& x, reduction_workspace_t<f_t>& reductions)
{
  return std::sqrt(reductions.transform_reduce(
    x.data(),
    thrust::plus<f_t>{},
    [] __host__ __device__(f_t val) { return val * val; },
    f_t(0),
    x.size(),
    x.stream()));
}

template <typename f_t>
f_t vector_norm2(const rmm::device_uvector<f_t>& x)
{
  auto begin          = x.data();
  auto end            = x.data() + x.size();
  auto sum_of_squares = thrust::transform_reduce(
    rmm::exec_policy(x.stream()),
    begin,
    end,
    [] __host__ __device__(f_t val) { return val * val; },
    f_t(0),
    thrust::plus<f_t>{});
  RAFT_CHECK_CUDA(x.stream().get());
  return std::sqrt(sum_of_squares);
}

}  // namespace cuopt::mathematical_optimization
