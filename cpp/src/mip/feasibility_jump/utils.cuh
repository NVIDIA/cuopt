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

#pragma once

#include "feasibility_jump.cuh"

#include <cuda/atomic>
#include <raft/core/device_span.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/cuda_helpers.cuh>
#include <utilities/device_utils.cuh>
#include <utilities/macros.cuh>

namespace cuopt::linear_programming::detail {

HDI uint64_t hash_64(uint64_t x)
{
  // to prevent the zero hash, add 1
  x++;
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  x = x ^ (x >> 31);
  return x;
}

template <typename word_t = uint32_t>
struct bitmap_t {
  static constexpr int bits_per_word = sizeof(word_t) * CHAR_BIT;

  bitmap_t(size_t size, const rmm::cuda_stream_view& stream)
    : validity_bitmap(size > 0 ? (size - 1) / bits_per_word + 1 : 0, stream)
  {
    clear(stream);
  }

  void clear(const rmm::cuda_stream_view& stream)
  {
    cudaMemsetAsync(
      validity_bitmap.data(), 0, sizeof(word_t) * validity_bitmap.size(), stream.value());
  }
  void clear(const raft::handle_t* handle_ptr)
  {
    thrust::uninitialized_fill(
      handle_ptr->get_thrust_policy(), validity_bitmap.begin(), validity_bitmap.end(), 0);
  }
  void resize(size_t size, const rmm::cuda_stream_view& stream)
  {
    validity_bitmap.resize(size > 0 ? (size - 1) / bits_per_word + 1 : 0, stream);
  }

  struct view_t {
    raft::device_span<word_t> validity_bitmap;

    DI thrust::pair<size_t, size_t> idx_to_bitmap(size_t idx) const
    {
      return {idx / bits_per_word, idx % bits_per_word};
    }

    DI void set(size_t idx) const
    {
      auto [word_idx, bit_idx] = idx_to_bitmap(idx);
      cuopt_assert(word_idx < validity_bitmap.size(), "invalid index");
      atomicOr(&validity_bitmap[word_idx], (1u << bit_idx));
    }

    DI void unset(size_t idx) const
    {
      auto [word_idx, bit_idx] = idx_to_bitmap(idx);
      cuopt_assert(word_idx < validity_bitmap.size(), "invalid index");
      atomicAnd(&validity_bitmap[word_idx], ~(1u << bit_idx));
    }

    DI void clear() const
    {
      for (size_t i = 0; i < validity_bitmap.size(); ++i)
        validity_bitmap[i] = 0;
    }

    DI bool contains(size_t idx) const
    {
      auto [word_idx, bit_idx] = idx_to_bitmap(idx);
      cuopt_assert(word_idx < validity_bitmap.size(), "invalid index");
      return !!(validity_bitmap[word_idx] & (1 << bit_idx));
    }
  };

  view_t view() { return {make_span(validity_bitmap)}; }

  rmm::device_uvector<word_t> validity_bitmap;
};

template <typename i_t, typename f_t>
struct contiguous_set_t {
  contiguous_set_t(i_t max_size, const rmm::cuda_stream_view& stream)
    : set_size(0, stream),
      lock(0, stream),
      contents(max_size, stream),
      index_map(max_size, stream),
      validity_bitmap(max_size, stream)
  {
    clear(stream);
  }

  void clear(const rmm::cuda_stream_view& stream)
  {
    set_size.set_value_to_zero_async(stream);
    // can't use thrust::fill, needs a memset node in order to be recorded in CUDA graphs
    // works bcs (uint8_t)-1 == 0xFF => (repeated 4 times) 0xFFFFFFFF == (uint32_t)-1
    cudaMemsetAsync(index_map.data(), -1, sizeof(i_t) * index_map.size(), stream.value());
    validity_bitmap.clear(stream);
  }

  void clear(const raft::handle_t* handle_ptr)
  {
    thrust::uninitialized_fill(
      handle_ptr->get_thrust_policy(), index_map.begin(), index_map.end(), -1);
    validity_bitmap.clear(handle_ptr);
    set_size.set_value_to_zero_async(handle_ptr->get_stream());
  }

  void resize(size_t size, const rmm::cuda_stream_view& stream)
  {
    contents.resize(size, stream);
    index_map.resize(size, stream);
    validity_bitmap.resize(size, stream);
  }

  struct view_t {
    i_t* set_size;
    i_t* lock;
    raft::device_span<i_t> contents;
    raft::device_span<i_t> index_map;
    // smaller bitmap to improve performance when we're only interested about
    // the existence of a certain index
    typename bitmap_t<uint32_t>::view_t validity_bitmap;

    DI void insert(i_t val) const
    {
      cuopt_assert(val >= 0 && val < index_map.size(), "Value is out of bounds");
      cuopt_assert(index_map[val] == -1, "Value already exists");

      auto idx       = atomicAdd(set_size, 1);
      index_map[val] = idx;
      contents[idx]  = val;
      validity_bitmap.set(val);

      cuopt_assert(*set_size <= contents.size(), "Set overrun");

      cuopt_assert(contains(val), "insert failed");
    }

    DI void remove(i_t val) const
    {
      cuopt_assert(val >= 0 && val < index_map.size(), "Value is out of bounds");
      cuopt_assert(index_map[val] != -1, "Value not found");
      cuopt_assert(*set_size > 0, "Set empty");

      auto size           = atomicSub(set_size, 1);
      i_t last_var        = contents[size - 1];
      i_t idx             = index_map[val];
      contents[idx]       = last_var;
      index_map[last_var] = idx;
      index_map[val]      = -1;

      validity_bitmap.unset(val);

      cuopt_assert(!contains(val), "remove failed");
    }

    DI void clear() const
    {
      for (i_t i = 0; i < index_map.size(); ++i)
        index_map[i] = -1;
      validity_bitmap.clear();
      *set_size = 0;
    }

    DI auto begin() const { return contents.begin(); }
    DI auto end() const { return contents.begin() + *set_size; }

    DI bool contains(i_t val) const
    {
      cuopt_assert(val >= 0 && val < index_map.size(), "Value is out of bounds");
      return validity_bitmap.contains(val);
    }
    DI i_t size() const { return *set_size; }
    DI i_t max_size() const { return contents.size(); }
    DI bool empty() const { return size() == 0; }
  };

  view_t view()
  {
    return {set_size.data(),
            lock.data(),
            make_span(contents),
            make_span(index_map),
            validity_bitmap.view()};
  }

  rmm::device_scalar<i_t> set_size;
  rmm::device_scalar<i_t> lock;
  rmm::device_uvector<i_t> contents;
  rmm::device_uvector<i_t> index_map;
  bitmap_t<uint32_t> validity_bitmap;
};

}  // namespace cuopt::linear_programming::detail
