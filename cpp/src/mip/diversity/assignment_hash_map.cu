/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "assignment_hash_map.cuh"

#include <mip/mip_constants.hpp>
#include <utilities/copy_helpers.hpp>

#include <thrust/gather.h>
#include <cub/cub.cuh>
#include <cuda/std/functional>

namespace cuopt {
namespace linear_programming {
namespace detail {

struct combine_hash {
  DI size_t operator()(size_t hash_1, size_t hash_2)
  {
    const std::size_t magic_constant = 0x9e3779b97f4a7c15;
    hash_1 ^= hash_2 + magic_constant + (hash_1 << 12) + (hash_1 >> 4);
    return hash_1;
  }
};

template <typename i_t, typename f_t, int TPB>
__global__ void hash_solution_kernel(raft::device_span<size_t> assignment,
                                     raft::device_span<size_t> reduction_buffer)
{
  typedef cub::BlockReduce<size_t, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  size_t th_hash = assignment.size();
#pragma unroll
  for (i_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < assignment.size();
       idx += blockDim.x * gridDim.x) {
    th_hash = combine_hash()(th_hash, assignment[idx]);
  }
  size_t hash_sum = BlockReduce(temp_storage).Reduce(th_hash, combine_hash(), TPB);
  if (threadIdx.x == 0) { reduction_buffer[blockIdx.x] = hash_sum; }
}

template <typename i_t, typename f_t, int TPB>
__global__ void reduce_hash_kernel(raft::device_span<size_t> reduction_buffer,
                                   size_t* global_hash_sum)
{
  typedef cub::BlockReduce<size_t, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  size_t th_hash  = reduction_buffer[threadIdx.x];
  size_t hash_sum = BlockReduce(temp_storage).Reduce(th_hash, combine_hash(), TPB);
  if (threadIdx.x == 0) { *global_hash_sum = hash_sum; }
}

template <typename i_t, typename f_t>
assignment_hash_map_t<i_t, f_t>::assignment_hash_map_t(const problem_t<i_t, f_t>& problem)
  : reduction_buffer(problem.n_variables, problem.handle_ptr->get_stream()),
    integer_assignment(problem.n_integer_vars, problem.handle_ptr->get_stream()),
    hash_sum(problem.handle_ptr->get_stream()),
    temp_storage(0, problem.handle_ptr->get_stream())
{
}

// we might move this to be a solution member if it is needed
// currently having an ordered integer array is only needed here
template <typename i_t, typename f_t>
void assignment_hash_map_t<i_t, f_t>::fill_integer_assignment(solution_t<i_t, f_t>& solution)
{
  static_assert(sizeof(f_t) == sizeof(size_t), "f_t must be double precision");
  thrust::gather(solution.handle_ptr->get_thrust_policy(),
                 solution.problem_ptr->integer_indices.begin(),
                 solution.problem_ptr->integer_indices.end(),
                 solution.assignment.begin(),
                 reinterpret_cast<double*>(integer_assignment.data()));
}

template <typename i_t, typename f_t>
size_t assignment_hash_map_t<i_t, f_t>::hash_solution(solution_t<i_t, f_t>& solution)
{
  const int TPB = 1024;

  fill_integer_assignment(solution);
  thrust::fill(
    solution.handle_ptr->get_thrust_policy(), reduction_buffer.begin(), reduction_buffer.end(), 0);
  hash_solution_kernel<i_t, f_t, TPB>
    <<<(integer_assignment.size() + TPB - 1) / TPB, TPB, 0, solution.handle_ptr->get_stream()>>>(
      cuopt::make_span(integer_assignment), cuopt::make_span(reduction_buffer));
  RAFT_CHECK_CUDA(solution.handle_ptr->get_stream());
  // Get the number of blocks used in the hash_solution_kernel
  int num_blocks = (integer_assignment.size() + TPB - 1) / TPB;

  // If we have more than one block, perform a device-wide reduction using CUB
  if (num_blocks > 1) {
    // Determine temporary device storage requirements
    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce(d_temp_storage,
                              temp_storage_bytes,
                              reduction_buffer.data(),
                              hash_sum.data(),
                              num_blocks,
                              combine_hash(),
                              0,
                              solution.handle_ptr->get_stream());

    // Allocate temporary storage
    temp_storage.resize(temp_storage_bytes, solution.handle_ptr->get_stream());
    d_temp_storage = temp_storage.data();

    // Run reduction
    cub::DeviceReduce::Reduce(d_temp_storage,
                              temp_storage_bytes,
                              reduction_buffer.data(),
                              hash_sum.data(),
                              num_blocks,
                              combine_hash(),
                              0,
                              solution.handle_ptr->get_stream());

    // Return early since we've already computed the hash sum
    return hash_sum.value(solution.handle_ptr->get_stream());
  } else {
    return reduction_buffer.element(0, solution.handle_ptr->get_stream());
  }
}

template <typename i_t, typename f_t>
void assignment_hash_map_t<i_t, f_t>::insert(solution_t<i_t, f_t>& solution)
{
  size_t sol_hash = hash_solution(solution);
  solution_hash_count[sol_hash]++;
}

template <typename i_t, typename f_t>
bool assignment_hash_map_t<i_t, f_t>::check_skip_solution(solution_t<i_t, f_t>& solution,
                                                          i_t max_occurance)
{
  size_t hash = hash_solution(solution);
  if (solution_hash_count[hash] > max_occurance) {
    CUOPT_LOG_DEBUG("Skipping solution which is encountered %d times", solution_hash_count[hash]);
    return true;
  }
  return false;
}

#if !MIP_INSTANTIATE_FLOAT && !MIP_INSTANTIATE_DOUBLE
static_assert(false, "MIP_INSTANTIATE_FLOAT or MIP_INSTANTIATE_DOUBLE must be defined");
#endif

#if MIP_INSTANTIATE_FLOAT
template class assignment_hash_map_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class assignment_hash_map_t<int, double>;
#endif

}  // namespace detail
}  // namespace linear_programming
}  // namespace cuopt
