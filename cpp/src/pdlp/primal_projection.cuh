/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include <utilities/type_2.hpp>

#include <thrust/tuple.h>
#include <raft/util/cuda_utils.cuh>

namespace cuopt::mathematical_optimization::pdlp {

template <typename f_t>
HDI f_t
halpern_blend(f_t reflected, f_t current, f_t initial, f_t weight, f_t reflection_coefficient)
{
  const f_t relaxed =
    reflection_coefficient * reflected + (f_t(1.0) - reflection_coefficient) * current;
  return weight * relaxed + (f_t(1.0) - weight) * initial;
}

template <typename f_t>
struct primal_reflected_projection {
  using f_t2 = typename type_2<f_t>::type;
  primal_reflected_projection(const f_t* scalar,
                              const f_t* halpern_weight,
                              f_t reflection_coefficient)
    : scalar_{scalar},
      halpern_weight_{halpern_weight},
      reflection_coefficient_{reflection_coefficient}
  {
  }
  HDI thrust::tuple<f_t, f_t> operator()(
    f_t current_primal, f_t initial_primal, f_t objective, f_t Aty, f_t2 bounds)
  {
    const f_t next         = current_primal - *scalar_ * (objective - Aty);
    const f_t next_clamped = raft::max<f_t>(raft::min<f_t>(next, bounds.y), bounds.x);
    const f_t reflected    = f_t(2.0) * next_clamped - current_primal;
    return {
      reflected,
      halpern_blend(
        reflected, current_primal, initial_primal, *halpern_weight_, reflection_coefficient_)};
  }
  const f_t* scalar_;
  const f_t* halpern_weight_;
  f_t reflection_coefficient_;
};

}  // namespace cuopt::mathematical_optimization::pdlp
