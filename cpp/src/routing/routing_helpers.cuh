/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "dimensions.cuh"

namespace cuopt {
namespace routing {
namespace detail {

constexpr int DEPOT = 0;

// this is to be used in host code as device code cannot handle constexpr arrays
// the weights are always double as they are cauing issues with convergence
static constexpr const double default_weights[] = {1., 1., 1., 1., 1., 1., 1., 1., 1.};
static constexpr const double zero_cost[]       = {0., 0., 0., 0., 0., 0., 0., 0., 0.};
__device__ const double d_default_weights[]     = {1., 1., 1., 1., 1., 1., 1., 1., 1.};
__device__ const double d_zero_cost[]           = {0., 0., 0., 0., 0., 0., 0., 0., 0.};

static_assert(sizeof(default_weights) / sizeof(double) == (size_t)dim_t::SIZE);
static_assert(sizeof(d_default_weights) / sizeof(double) == (size_t)dim_t::SIZE);
static_assert(sizeof(d_zero_cost) / sizeof(double) == (size_t)dim_t::SIZE);

/**
 * @brief Generic implementation of constexpr_for
 *
 * @tparam Start
 * @tparam End
 * @tparam Inc
 * @tparam F    Function to be called for each iterate
 * @param f
 */
template <auto Start, auto End, auto Inc, class F>
static constexpr void constexpr_for(F&& f)
{
  if constexpr (Start < End) {
    f(std::integral_constant<decltype(Start), Start>());
    constexpr_for<Start + Inc, End, Inc>(f);
  }
}

template <auto End, class F>
static constexpr void constexpr_for(F&& f)
{
  constexpr_for<(size_t)0, (size_t)End, size_t(1)>(f);
}

/**
 * @brief  Loop over all dimensions
 * FIXME This is a placeholder code only. This needs to be deleted since we would never
 * loop over inactive dimensions
 *
 * @tparam F
 * @param f
 */
template <class F>
static constexpr void loop_over_all_dimensions(F&& f)
{
  constexpr_for<(size_t)0, (size_t)dim_t::SIZE, (size_t)1, F>(std::move(f));
}

/**
 * @brief Loop over only active dimensions. Note that except for checking if a dimension
 * exists, everything else is determined at compile time, including the lambda.
 *
 * @note Potentially everything can be done at compile time if enabled_dimensions is compile time
 *
 * @tparam Start
 * @tparam End
 * @tparam F
 * @param dimensions_info
 * @param f
 */
template <size_t Start = 0, size_t End = (size_t)dim_t::SIZE, class F>
static constexpr void loop_over_dimensions(const enabled_dimensions_t& dimensions_info, F&& f)
{
  if constexpr (Start < End) {
    if (dimensions_info.has_dimension((dim_t)Start)) {
      f(std::integral_constant<decltype(Start), Start>());
    }
    loop_over_dimensions<Start + 1, End>(dimensions_info, f);
  }
}

/**
 * @brief Loop over only active constrained dimensions. Note that except for
 * checking if a dimension exists, everything else is determined at compile
 * time, including the lambda.
 *
 * @note Potentially everything can be done at compile time if enabled_dimensions is compile time
 *
 * @tparam Start
 * @tparam End
 * @tparam F
 * @param dimensions_info
 * @param f
 */
template <size_t Start = 0, size_t End = (size_t)dim_t::SIZE, class F>
static constexpr void loop_over_constrained_dimensions(const enabled_dimensions_t& dimensions_info,
                                                       F&& f)
{
  if constexpr (Start < End) {
    if (dimensions_info.has_dimension((dim_t)Start) &&
        get_dimension_of<Start>(dimensions_info).has_constraints()) {
      f(std::integral_constant<decltype(Start), Start>());
    }
    loop_over_constrained_dimensions<Start + 1, End>(dimensions_info, f);
  }
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
