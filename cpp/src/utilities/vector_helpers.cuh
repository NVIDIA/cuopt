/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <thrust/extrema.h>
#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt {

template <typename T>
__global__ void fill_kernel(T* data_ptr, T item, size_t size)
{
  size_t index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= size) { return; }
  data_ptr[index] = item;
}

template <typename T>
__global__ void sequence_kernel(T* data_ptr, size_t size)
{
  size_t index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= size) { return; }
  data_ptr[index] = index;
}

template <typename T>
__global__ void sequence_with_multiplier_kernel(T* data_ptr, int mult, size_t size)
{
  size_t index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= size) { return; }
  data_ptr[index] = index * mult;
}

template <typename T>
void async_fill(rmm::device_uvector<T>& vec, T item, rmm::cuda_stream_view stream)
{
  constexpr size_t TPB = 256;
  size_t n_blocks      = (vec.size() + TPB - 1) / TPB;
  fill_kernel<<<n_blocks, TPB, 0, stream>>>(vec.data(), item, vec.size());
}

template <typename T>
void async_fill(T* vec, T item, size_t size, rmm::cuda_stream_view stream)
{
  constexpr size_t TPB = 256;
  size_t n_blocks      = (size + TPB - 1) / TPB;
  fill_kernel<<<n_blocks, TPB, 0, stream>>>(vec, item, size);
}

template <typename T>
void async_sequence(rmm::device_uvector<T>& vec, rmm::cuda_stream_view stream)
{
  constexpr size_t TPB = 256;
  size_t n_blocks      = (vec.size() + TPB - 1) / TPB;
  sequence_kernel<<<n_blocks, TPB, 0, stream>>>(vec.data(), vec.size());
}

template <typename T>
void async_sequence_with_multiplier(rmm::device_uvector<T>& vec,
                                    int mult,
                                    rmm::cuda_stream_view stream)
{
  constexpr size_t TPB = 256;
  size_t n_blocks      = (vec.size() + TPB - 1) / TPB;
  sequence_with_multiplier_kernel<<<n_blocks, TPB, 0, stream>>>(vec.data(), mult, vec.size());
}

template <typename T>
bool all_entries_are_equal(raft::handle_t const* handle_ptr, T const* ptr, const size_t sz)
{
  auto min_max = thrust::minmax_element(handle_ptr->get_thrust_policy(), ptr, ptr + sz);
  T min_val, max_val;
  raft::copy(&min_val, min_max.first, 1, handle_ptr->get_stream());
  raft::copy(&max_val, min_max.second, 1, handle_ptr->get_stream());
  return min_val == max_val;
}

template <typename T>
bool all_entries_are_equal(raft::handle_t const* handle_ptr, raft::device_span<T const> const& v)
{
  if (v.empty()) { return true; }
  return all_entries_are_equal(handle_ptr, v.data(), v.size());
}

template <typename T>
bool all_entries_are_zero(raft::handle_t const* handle_ptr, T const* ptr, const size_t sz)
{
  auto max = thrust::max_element(handle_ptr->get_thrust_policy(), ptr, ptr + sz);
  T max_val;
  raft::copy(&max_val, max, 1, handle_ptr->get_stream());
  return max_val == 0;
}

}  // namespace cuopt
