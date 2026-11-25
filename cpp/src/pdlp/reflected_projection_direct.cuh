/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <pdlp/primal_projection.cuh>

#include <cuda/cmath>
#include <cuda/stream>
#include <raft/util/cudart_utils.hpp>

#include <cstddef>
#include <type_traits>

namespace cuopt::mathematical_optimization::pdlp {

template <typename f_t>
struct dual_reflected_projection {
  dual_reflected_projection(const f_t* scalar,
                            const f_t* halpern_weight,
                            f_t reflection_coefficient)
    : scalar_{scalar},
      halpern_weight_{halpern_weight},
      reflection_coefficient_{reflection_coefficient}
  {
  }
  HDI thrust::tuple<f_t, f_t> operator()(
    f_t current_dual, f_t initial_dual, f_t Ax, f_t lower_bound, f_t upper_bounds)
  {
    cuopt_assert(*scalar_ != f_t(0.0), "Scalar can't be 0");
    const f_t tmp       = current_dual / *scalar_ - Ax;
    const f_t tmp_proj  = raft::max<f_t>(-upper_bounds, raft::min<f_t>(tmp, -lower_bound));
    const f_t next_dual = (tmp - tmp_proj) * *scalar_;
    const f_t reflected = f_t(2.0) * next_dual - current_dual;
    return {reflected,
            halpern_blend(
              reflected, current_dual, initial_dual, *halpern_weight_, reflection_coefficient_)};
  }

  const f_t* scalar_;
  const f_t* halpern_weight_;
  f_t reflection_coefficient_;
};

namespace detail {

template <typename f_t, typename bounds_t, typename projection_t>
__global__ void reflected_projection_direct_kernel(f_t* current_solution,
                                                   const f_t* initial_solution,
                                                   const f_t* objective_or_primal_product,
                                                   const f_t* dual_product_or_lower_bounds,
                                                   const bounds_t* variable_or_upper_bounds,
                                                   f_t* reflected,
                                                   size_t count,
                                                   projection_t projection)
{
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    // Read the current iterate before either output is overwritten.
    const auto result     = projection(current_solution[idx],
                                   initial_solution[idx],
                                   objective_or_primal_product[idx],
                                   dual_product_or_lower_bounds[idx],
                                   variable_or_upper_bounds[idx]);
    reflected[idx]        = thrust::get<0>(result);
    current_solution[idx] = thrust::get<1>(result);
  }
}

template <typename f_t, typename bounds_t, typename projection_t>
void launch_direct_reflected_projection(f_t* current_solution,
                                        const f_t* initial_solution,
                                        const f_t* objective_or_primal_product,
                                        const f_t* dual_product_or_lower_bounds,
                                        const bounds_t* variable_or_upper_bounds,
                                        f_t* reflected,
                                        size_t count,
                                        projection_t projection,
                                        cuda::stream_ref stream)
{
  if constexpr (std::is_same_v<f_t, double>) {
    if (count != 0) {
      constexpr size_t block_size = 256;
      const size_t grid_size      = cuda::ceil_div(count, block_size);
      reflected_projection_direct_kernel<<<grid_size, block_size, 0, stream.get()>>>(
        current_solution,
        initial_solution,
        objective_or_primal_product,
        dual_product_or_lower_bounds,
        variable_or_upper_bounds,
        reflected,
        count,
        projection);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
  }
}

}  // namespace detail
}  // namespace cuopt::mathematical_optimization::pdlp
