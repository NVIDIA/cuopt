/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "../node/node.cuh"

#include <cuopt/error.hpp>
#include <utilities/cuda_helpers.cuh>
#include <utilities/seed_generator.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/core/error.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_device.cuh>

#include <random>

namespace cuopt {
namespace routing {
namespace detail {

template <typename elemt_t>
__global__ static void device_random_shuffle(elemt_t* data, int size, int64_t seed)
{
  raft::random::PCGenerator thread_rng(seed + (threadIdx.x + blockIdx.x * blockDim.x), size, 0);
  for (int i = 0; i < size * 2; ++i) {
    raft::swapVals(data[size - 1], data[thread_rng.next_u32() % (size - 1)]);
  }
}

/* Contains request id
 *  Nodes are kept device side to avoid going back and forth between host and device
 */
template <class elemt_t, int max_stack_size = std::numeric_limits<int>::max()>
struct ejection_pool_t {
  ejection_pool_t(int max_ejection_pool_size, rmm::cuda_stream_view stream)
    : stack_(max_ejection_pool_size, stream), index_(-1), stream_(stream)
  {
  }

  elemt_t* pop()
  {
    cuopt_assert(index_ >= 0, "Index needs to be superior or equal to 0");
    elemt_t* elem = stack_.element_ptr(index_);
    --index_;
    return elem;
  }

  void push_back_last() { ++index_; }

  void random_shuffle()
  {
    // replace with thrust shuffle
    // how to get sol_handle::get_thrust_policy?
    if (size() > 1)
      device_random_shuffle<elemt_t>
        <<<1, 1, 0, stream_>>>(stack_.data(), size(), seed_generator::get_seed());
  }

  bool empty() const
  {
    cuopt_assert(index_ >= -1, "Index of stack can only be greater or equal to -1");
    return (index_ == -1);
  }

  int size() const { return index_ + 1; }

  void print() const
  {
    raft::print_device_vector("EP :", (int*)stack_.data(), size() * 2, std::cout);
  }

  void clear() { index_ = -1; }

  struct view_t {
    elemt_t* stack_;
    int index_;
    int max_ejection_pool_size;

    DI bool empty() const
    {
      cuopt_assert(index_ >= -1, "Index of stack can only be greater or equal to -1");
      return index_ == -1;
    }
    DI void clear() { index_ = -1; }
    DI int size() const { return index_ + 1; }

    DI void pop(elemt_t to_pop)
    {
      cuopt_assert(index_ >= 0, "Index needs to be superior or equal to 0");
      cuopt_assert(__popc(__activemask() == 1),
                   "Ejection pool pop should be called by a single thread");
      int i = index_;
      while (i >= 0) {
        if (to_pop == this->stack_[i]) {
          raft::swapVals(this->stack_[i], this->stack_[index_]);
          --index_;
          break;
        }
        --i;
      }
    }

    DI void push(elemt_t to_insert)
    {
      cuopt_assert(index_ + 1 < max_ejection_pool_size,
                   "Trying to insert more than max size in ejection pool");
      cuopt_assert(__popc(__activemask() == 1),
                   "Ejection pool push should be called by a single thread");

      if (index_ < max_stack_size - 1) {
        // Incremeting in the view but should also be on host
        ++index_;
        this->stack_[index_] = to_insert;
      }
    }

    DI elemt_t const at(int pos) const
    {
      cuopt_assert(pos >= 0 && pos < size(), "Wrong position index");
      return this->stack_[pos];
    }

    DI elemt_t back() { return this->stack_[index_]; }
    DI elemt_t pop()
    {
      cuopt_assert(index_ >= 0, "Index needs to be superior or equal to 0");
      cuopt_assert(__popc(__activemask() == 1),
                   "Ejection pool pop should be called by a single thread");
      elemt_t elem = this->stack_[index_];
      --index_;
      return elem;
    }
  };

  view_t view()
  {
    view_t v;
    v.stack_                 = stack_.data();
    v.index_                 = index_;
    v.max_ejection_pool_size = stack_.capacity();

    return v;
  }

  rmm::device_uvector<elemt_t> stack_;
  int index_;
  rmm::cuda_stream_view stream_;
  std::uniform_int_distribution<int> dist{0, std::numeric_limits<int>::max()};
  std::mt19937 gen{66742};
};
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
