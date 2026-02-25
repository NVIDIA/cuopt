/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <utilities/device_utils.cuh>

#include <cub/cub.cuh>

#include "bounds_update_data.cuh"

namespace cuopt::linear_programming::detail {

template <typename f_t>
struct tuple_plus_t {
  __device__ thrust::tuple<f_t, f_t> operator()(thrust::tuple<f_t, f_t> t0,
                                                thrust::tuple<f_t, f_t> t1)
  {
    return thrust::make_tuple(thrust::get<0>(t0) + thrust::get<0>(t1),
                              thrust::get<1>(t0) + thrust::get<1>(t1));
  }
  __device__ thrust::tuple<f_t, f_t, f_t, f_t> operator()(thrust::tuple<f_t, f_t, f_t, f_t> t0,
                                                          thrust::tuple<f_t, f_t, f_t, f_t> t1)
  {
    return thrust::make_tuple(thrust::get<0>(t0) + thrust::get<0>(t1),
                              thrust::get<1>(t0) + thrust::get<1>(t1),
                              thrust::get<2>(t0) + thrust::get<2>(t1),
                              thrust::get<3>(t0) + thrust::get<3>(t1));
  }
};

}  // namespace cuopt::linear_programming::detail
