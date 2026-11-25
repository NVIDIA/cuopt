/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <cub/device/device_transform.cuh>
#include <cuda/stream>
#include <raft/util/cudart_utils.hpp>

namespace cuopt::mathematical_optimization::pdlp {

template <typename i_t, typename f_t>
struct halpern_weight_from_iteration {
  __device__ __forceinline__ f_t operator()(i_t k) const { return f_t(k + 1) / f_t(k + 2); }
};

namespace detail {

template <typename i_t, typename f_t>
__global__ void finish_batched_halpern_kernel(i_t* restart_age,
                                              f_t* final_weight,
                                              const f_t* weights,
                                              i_t batch_size)
{
  // The graph batch cannot cross a restart or a host checkpoint. Commit exactly
  // the scalar state left by the final ordinary iteration, once all steps finish.
  *restart_age += batch_size;
  *final_weight = weights[batch_size - 1];
}

template <typename i_t, typename f_t>
void prepare_batched_halpern_weights(const i_t* restart_age,
                                     f_t* weights,
                                     i_t batch_size,
                                     cuda::stream_ref stream)
{
  RAFT_CUDA_TRY(cub::DeviceTransform::Transform(
    thrust::make_counting_iterator<i_t>(0),
    weights,
    batch_size,
    [restart_age] __device__(i_t step) {
      return halpern_weight_from_iteration<i_t, f_t>{}(*restart_age + step);
    },
    stream.get()));
}

template <typename i_t, typename f_t>
void finish_batched_halpern(
  i_t* restart_age, f_t* final_weight, const f_t* weights, i_t batch_size, cuda::stream_ref stream)
{
  finish_batched_halpern_kernel<<<1, 1, 0, stream.get()>>>(
    restart_age, final_weight, weights, batch_size);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace detail
}  // namespace cuopt::mathematical_optimization::pdlp
