/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include "routing/utilities/constants.hpp"

#include <routing/structures.hpp>
#include <utilities/seed_generator.cuh>

#include <raft/random/rng_device.cuh>
#include <raft/util/cuda_dev_essentials.cuh>

#include <cuda_fp16.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>
#include <limits>

namespace cuopt {
namespace routing {
namespace detail {

template <typename cand_val_t>
__device__ inline float get_value(cand_val_t val)
{
  if constexpr (std::is_same<__half, cand_val_t>::value) { return __half2float(val); }
  return val;
}

template <typename i_t = int, typename f_t = float>
constexpr i_t get_intra_route_delivery_index(NodeInfo<i_t> const* path,
                                             i_t route_size,
                                             i_t start,
                                             i_t delivery_node)
{
  for (i_t i = start; i < start + route_size; ++i) {
    if (path[i].node() == delivery_node) { return i; }
  }
  return -1;
}
template <typename i_t = int, typename f_t = float>
struct objective_to_int {
  __device__ i_t operator()(objective_t objective) { return static_cast<i_t>(objective); }
};

template <typename T, typename i_t = int, typename f_t = float>
__device__ inline T shfl(T val, i_t srcLane, i_t width = warp_size, uint mask = 0xffffffffu)
{
  return __shfl_sync(mask, val, srcLane, width);
}

template <typename T, typename i_t = int, typename f_t = float>
__device__ inline T warp_reduce(T val)
{
#pragma unroll
  for (i_t i = warp_size / 2; i > 0; i >>= 1) {
    T tmp = shfl(val, raft::laneId() + i);
    val   = min(val, tmp);
  }
  return val;
}

// borrowed from raft(raft only has summation)
template <typename T, typename i_t = int>
__device__ inline void block_reduce(T val, T* shmem, const i_t size = blockDim.x)
{
  i_t nWarps = (size + warp_size - 1) / warp_size;
  i_t lid    = raft::laneId();
  i_t wid    = threadIdx.x / warp_size;
  T warp_min = warp_reduce<T>(val);
  if (lid == 0) shmem[wid] = warp_min;
  __syncthreads();
  warp_min    = lid < nWarps ? shmem[lid] : std::numeric_limits<T>::max();
  T final_min = warp_reduce<T>(warp_min);
  __syncthreads();
  if (threadIdx.x == 0) shmem[0] = final_min;
  __syncthreads();
}

/**
 * @brief Warp reduction for getting minimum for val1 and maximum for val2
 * @todo accept a custom comparator and a single merged type
 */
template <typename val1_t, typename val2_t, typename i_t>
__inline__ __device__ void warp_reduce_ranked(val1_t& val1, val2_t& val2, i_t& idx)
{
#pragma unroll
  for (i_t offset = warp_size / 2; offset > 0; offset /= 2) {
    val1_t tmp_val1 = shfl(val1, raft::laneId() + offset);
    val2_t tmp_val2 = shfl(val2, raft::laneId() + offset);
    i_t tmp_idx     = shfl(idx, raft::laneId() + offset);
    if (tmp_val1 < val1 || (tmp_val1 == val1 && tmp_val2 > val2)) {
      val1 = tmp_val1;
      val2 = tmp_val2;
      idx  = tmp_idx;
    }
  }
}

template <typename T, typename i_t = int>
__inline__ __device__ void warp_reduce_ranked(T& val, i_t& idx)
{
#pragma unroll
  for (i_t offset = warp_size / 2; offset > 0; offset /= 2) {
    T tmpVal   = shfl(val, raft::laneId() + offset);
    i_t tmpIdx = shfl(idx, raft::laneId() + offset);
    if (tmpVal < val) {
      val = tmpVal;
      idx = tmpIdx;
    }
  }
}

template <typename T, typename i_t = int>
__inline__ __device__ void block_reduce_ranked(T& val, i_t& idx, T* shbuf, i_t* min_index)
{
  T* values    = shbuf;
  i_t* indices = (i_t*)&shbuf[warp_size];
  i_t wid      = threadIdx.x / warp_size;
  i_t nWarps   = (blockDim.x + warp_size - 1) / warp_size;
  warp_reduce_ranked(val, idx);  // Each warp performs partial reduction
  i_t lane = raft::laneId();
  if (lane == 0) {
    values[wid]  = val;  // Write reduced value to shared memory
    indices[wid] = idx;  // Write reduced value to shared memory
  }

  __syncthreads();  // Wait for all partial reductions

  // read from shared memory only if that warp existed
  if (lane < nWarps) {
    val = values[lane];
    idx = indices[lane];
  } else {
    val = std::numeric_limits<T>::max();
    idx = -1;
  }
  __syncthreads();
  if (wid == 0) warp_reduce_ranked(val, idx);
  if (threadIdx.x == 0) {
    shbuf[0]   = val;
    *min_index = idx;
  }
  __syncthreads();
}

// there is an overload because there is no std::numeric_limits<__half>
template <typename i_t = int>
__inline__ __device__ void block_reduce_ranked(__half& val, i_t& idx, __half* shbuf, i_t* min_index)
{
  __half* values = shbuf;
  i_t* indices   = (i_t*)&shbuf[warp_size];
  i_t wid        = threadIdx.x / warp_size;
  i_t nWarps     = (blockDim.x + warp_size - 1) / warp_size;
  warp_reduce_ranked(val, idx);  // Each warp performs partial reduction
  i_t lane = raft::laneId();
  if (lane == 0) {
    values[wid]  = val;  // Write reduced value to shared memory
    indices[wid] = idx;  // Write reduced value to shared memory
  }

  __syncthreads();  // Wait for all partial reductions

  // read from shared memory only if that warp existed
  if (lane < nWarps) {
    val = values[lane];
    idx = indices[lane];
  } else {
    val = __int2half_rn(I_HALF_MAX);
    idx = -1;
  }
  __syncthreads();
  if (wid == 0) warp_reduce_ranked(val, idx);
  if (threadIdx.x == 0) {
    shbuf[0]   = val;
    *min_index = idx;
  }
  __syncthreads();
}

template <typename i_t = int, typename f_t = float>
__device__ i_t __forceinline__ get_shared_idx(const i_t idx)
{
  return idx * blockDim.x + threadIdx.x;
}

// utility functions for shared local array access
template <typename T, typename i_t = int, typename f_t = float>
__device__ void inline set_shared_item(T* shared_ptr, T val, const i_t idx)
{
  i_t sh_index         = get_shared_idx(idx);
  shared_ptr[sh_index] = val;
}

template <typename T, typename i_t = int, typename f_t = float>
__device__ T inline get_shared_item(T* shared_ptr, const i_t idx)
{
  i_t sh_index = get_shared_idx(idx);
  return shared_ptr[sh_index];
}

template <typename T, typename i_t = int, typename f_t = float>
__inline__ __device__ i_t block_reduce_on_shared(__half* shared_array,
                                                 T* shbuf,
                                                 const i_t items_per_thread)
{
  __half thread_min = __int2half_rn(I_HALF_MAX);
  i_t min_idx       = -1;
  for (i_t i = 0; i < items_per_thread; i++) {
    __half curr_val = get_shared_item(shared_array, i);
    if (curr_val < thread_min) {
      thread_min = curr_val;
      min_idx    = i * blockDim.x + threadIdx.x;  // get the shared index
    }
  }
  block_reduce_ranked(thread_min, min_idx, (__half*)shbuf, &(((i_t*)shbuf)[1]));
  return ((i_t*)shbuf)[1];
}
template <typename i_t = int, typename f_t = float>
i_t inline round_up(i_t number, i_t multiple)
{
  if (multiple == 0) return number;

  i_t remainder = number % multiple;
  if (remainder == 0) return number;

  return number + multiple - remainder;
}
template <typename i_t = int, typename f_t = float>
__device__ inline f_t get_noise(raft::random::PCGenerator& rng, f_t base, f_t max_noise_ratio)
{
  f_t noise     = max_noise_ratio * abs(base);
  f_t noise_mul = rng.next_float() - 0.5f;
  noise *= noise_mul;
  return noise;
}

template <typename i_t = int, typename f_t = float>
struct comparator {
  __device__ __host__ thrust::tuple<i_t, i_t> operator()(const thrust::tuple<i_t, i_t>& t1,
                                                         const thrust::tuple<i_t, i_t>& t2)
  {
    i_t first  = thrust::get<0>(t1);
    i_t second = thrust::get<0>(t2);
    if (first == 1)
      return t2;
    else if (second == 1)
      return t1;
    else
      return first < second ? t1 : t2;
  }
};

// use this to avoid additional register pressure and shared memory usage from cub::blocksort
// TODO fix this to use the whole block rather than just a warp
template <typename count_t, typename value_t, typename out_t>
void __device__ inline block_inclusive_scan(out_t* out, value_t const* in, count_t n)
{
  count_t i, j, mn;
  value_t v, last;
  value_t sum = 0.0;
  bool valid;

  if (threadIdx.x < warp_size) {
    // Parallel prefix sum (using __shfl)
    mn = (((n + warp_size - 1) / warp_size) * warp_size);
    for (i = threadIdx.x; i < mn; i += warp_size) {
      // All threads (especially the last one) must always participate
      // in the shfl instruction, otherwise their sum will be undefined.
      // So, the loop stopping condition is based on multiple of n in loop increments,
      // so that all threads enter into the loop and inside we make sure we do not
      // read out of bounds memory checking for the actual size n.

      // check if the thread is valid
      valid = i < n;

      // Notice that the last thread is used to propagate the prefix sum.
      // For all the threads, in the first iteration the last is 0, in the following
      // iterations it is the value at the last thread of the previous iterations.

      // get the value of the last thread
      last = __shfl_sync(raft::warp_full_mask(), sum, warp_size - 1);

      // if you are valid read the value from memory, otherwise set your value to 0
      sum = (valid) ? in[i] : 0.0;

      for (j = 1; j < warp_size; j *= 2) {
        v = __shfl_up_sync(raft::warp_full_mask(), sum, j);
        if (threadIdx.x >= j) sum += v;
      }
      // shift by last
      sum += last;
      // To avoid race condition if in / out point to the same location
      __syncwarp();
      if constexpr (std::is_same<out_t, __half>::value) {
        if (valid) out[i] = __float2half_rn(sum);
      } else {
        if (valid) out[i] = sum;
      }
    }
  }
}

/**
 * @brief Reduces the values by key and scatters them to the key index location
 *
 * @note thrust::reduce_by_key performs reduction only based on consecutive key matches and
 * writes output based on the number of unique keys but not the actual key. This function
 * scatters the values to the specific key
 *
 * @tparam DerivedPolicy	The name of the derived execution policy.
 * @tparam InputIterator1	is a model of Input Iterator,
 * @tparam InputIterator2	is a model of Input Iterator,
 * @tparam OutputIterator1	is a model of Output Iterator and and InputIterator1's value_type is
convertible to OutputIterator1's value_type.
 * @tparam OutputIterator2	is a model of Output Iterator and and InputIterator2's
value_type is convertible to OutputIterator2's value_type.
* @tparam BinaryPredicate	is a model of Binary Predicate.
* @tparam BinaryFunction	is a model of Binary Function and BinaryFunction's result_type is
convertible to OutputIterator2's value_type.
 * @param exec             The execution policy to use for parallelization.
 * @param keys_first       The beginning of the input key range.
 * @param keys_last        The end of the input key range.
 * @param values_first     The beginning of the input value range.
 * @param keys_output      The beginning of the output key range for unique keys.
 * @param values_output    The beginning of the output value range computed from reduction.
 * @param scattered_values_output The beginnign of the scattered output value range
 * @param binary_pred      The binary predicate used to determine equality.
 * @param binary_op        The binary function used to accumulate values.
 */
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
void reduce_by_key_and_scatter_to_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_output,
  OutputIterator2 values_output,
  OutputIterator2 scattered_values_output,
  BinaryPredicate binary_pred,
  BinaryFunction binary_op)
{
  auto [last_key, last_value] = thrust::reduce_by_key(
    exec, keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);

  int n_unique = last_key - keys_output;
  thrust::scatter(
    exec, values_output, values_output + n_unique, keys_output, scattered_values_output);
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
