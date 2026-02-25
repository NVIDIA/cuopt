/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <utilities/macros.cuh>

#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/universal_vector.h>

#include <cuda/std/tuple>

#include <raft/util/cuda_utils.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t>
struct swap_pair_t {
  i_t left;
  i_t right;
};

template <typename i_t>
struct matrix_swap_index_functor {
  const swap_pair_t<i_t>* pairs;
  i_t vector_size;
  bool is_left;

  HDI size_t operator()(size_t idx) const
  {
    const i_t swap_idx = static_cast<i_t>(idx / static_cast<size_t>(vector_size));
    const i_t offset   = static_cast<i_t>(idx - static_cast<size_t>(swap_idx) * vector_size);
    const i_t base     = is_left ? pairs[swap_idx].left : pairs[swap_idx].right;
    return static_cast<size_t>(base) * vector_size + offset;
  }
};

template <typename i_t, typename f_t>
void matrix_swap(rmm::device_uvector<f_t>& matrix,
                 i_t vector_size,
                 const thrust::universal_host_pinned_vector<swap_pair_t<i_t>>& swap_pairs)
{
  if (swap_pairs.empty()) { return; }

  cuopt_assert(vector_size > 0, "Vector size must be greater than 0");
  cuopt_assert(matrix.size() % static_cast<size_t>(vector_size) == 0,
               "Matrix size must be divisible by vector size");
  const i_t batch_size = matrix.size() / vector_size;
  cuopt_assert(batch_size > 0, "Batch size must be greater than 0");

  const size_t swap_count  = swap_pairs.size();
  const size_t total_items = swap_count * static_cast<size_t>(vector_size);

  auto counting   = thrust::make_counting_iterator<size_t>(0);
  auto left_index = thrust::make_transform_iterator(
    counting,
    matrix_swap_index_functor<i_t>{thrust::raw_pointer_cast(swap_pairs.data()), vector_size, true});
  auto right_index = thrust::make_transform_iterator(
    counting,
    matrix_swap_index_functor<i_t>{
      thrust::raw_pointer_cast(swap_pairs.data()), vector_size, false});

  auto left_perm  = thrust::make_permutation_iterator(matrix.data(), left_index);
  auto right_perm = thrust::make_permutation_iterator(matrix.data(), right_index);
  auto in_zip     = thrust::make_zip_iterator(left_perm, right_perm);
  auto out_zip    = thrust::make_zip_iterator(left_perm, right_perm);

  cub::DeviceTransform::Transform(
    in_zip,
    out_zip,
    total_items,
    [] HD(thrust::tuple<f_t, f_t> values) -> thrust::tuple<f_t, f_t> {
      return thrust::make_tuple(thrust::get<1>(values), thrust::get<0>(values));
    },
    matrix.stream().value());
}

template <typename host_vector_t>
void host_vector_swap(host_vector_t& host_vector, int left_swap_index, int right_swap_index)
{
  cuopt_assert(left_swap_index < host_vector.size(), "Left swap index is out of bounds");
  cuopt_assert(right_swap_index < host_vector.size(), "Right swap index is out of bounds");
  cuopt_assert(left_swap_index < right_swap_index,
               "Left swap index must be less than right swap index");

  // Swap the id to swap to the end
  std::swap(host_vector[left_swap_index], host_vector[right_swap_index]);
}
}  // namespace cuopt::linear_programming::detail
