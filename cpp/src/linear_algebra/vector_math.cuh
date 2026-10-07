/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/copy_helpers.hpp>

#include <cub/device/device_reduce.cuh>

#include <cuda/std/functional>
#include <cuda/stream>
#include <raft/core/copy.hpp>
#include <raft/core/device_span.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_span.hpp>
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
struct transform_reduce_helper_t {
  rmm::device_buffer buffer_data;
  rmm::device_scalar<f_t> out;
  size_t buffer_size = 0;

  transform_reduce_helper_t(cuda::stream_ref stream_view)
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

struct norm_inf_max {
  template <typename f_t>
  __device__ __forceinline__ f_t operator()(const f_t& a, const f_t& b) const
  {
    f_t x = cuda::std::abs(a);
    f_t y = cuda::std::abs(b);
    return x > y ? x : y;
  }
};

template <typename f_t>
void vector_norm_inf_async(raft::device_span<const f_t> input,
                           raft::device_span<f_t> output,
                           transform_reduce_helper_t<f_t>& reductions,
                           cuda::stream_ref stream_view)
{
  reductions.transform_reduce_async(
    input.data(), norm_inf_max{}, cuda::std::identity{}, f_t(0), input.size(), output, stream_view);
}

template <typename i_t, typename f_t, typename InputIteratorT>
f_t device_custom_vector_norm_inf(InputIteratorT in, i_t size, cuda::stream_ref stream_view)
{
  if (size == 0) { return 0; }
  // FIXME: Tmp storage stored in vector_math class.
  auto d_out = rmm::device_scalar<f_t>(stream_view);
  rmm::device_uvector<uint8_t> d_temp_storage(0, stream_view);
  size_t temp_storage_bytes = 0;
  f_t init                  = 0;
  auto custom_op            = norm_inf_max{};
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            in,
                            d_out.data(),
                            size,
                            custom_op,
                            init,
                            stream_view.get());

  d_temp_storage.resize(temp_storage_bytes, stream_view);

  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            in,
                            d_out.data(),
                            size,
                            custom_op,
                            init,
                            stream_view.get());
  return d_out.value(stream_view);
}

template <typename i_t, typename f_t>
f_t device_vector_norm_inf(const rmm::device_uvector<f_t>& in, cuda::stream_ref stream_view)
{
  return device_custom_vector_norm_inf<i_t, f_t>(in.data(), in.size(), stream_view);
}

template <typename i_t, typename f_t>
f_t device_vector_norm_inf(raft::device_span<const f_t> in, cuda::stream_ref stream_view)
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

template <typename i_t, typename f_t>
f_t vector_norm_inf(raft::host_span<const f_t> x, cuda::stream_ref stream_view)
{
  rmm::device_uvector<f_t> d_x(x.size(), stream_view);
  raft::copy(d_x.data(), x.data(), x.size(), stream_view);
  return device_vector_norm_inf<i_t, f_t>(d_x, stream_view);
}

template <typename f_t>
f_t vector_norm_inf(const rmm::device_uvector<f_t>& x, transform_reduce_helper_t<f_t>& reductions)
{
  return reductions.transform_reduce(
    x.data(),
    thrust::maximum<f_t>{},
    [] __host__ __device__(f_t val) { return abs(val); },
    f_t(0),
    x.size(),
    x.stream());
}

template <typename f_t>
f_t vector_norm_inf(const rmm::device_uvector<f_t>& x)
{
  auto begin   = x.data();
  auto end     = x.data() + x.size();
  auto max_abs = thrust::transform_reduce(
    rmm::exec_policy(x.stream()),
    begin,
    end,
    [] __host__ __device__(f_t val) { return abs(val); },
    static_cast<f_t>(0),
    thrust::maximum<f_t>{});
  RAFT_CHECK_CUDA(x.stream().get());
  return max_abs;
}

template <typename f_t>
f_t vector_norm2(const rmm::device_uvector<f_t>& x, transform_reduce_helper_t<f_t>& reductions)
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
