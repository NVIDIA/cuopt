/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../../solution/solution_handle.cuh"
#include "hash_functions.cuh"

#include <utilities/macros.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/core/handle.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename map_key_t>
DI size_t hash(map_key_t const& key)
{
  static_assert(8 * (sizeof(map_key_t) / 8) == sizeof(map_key_t), "routes size error");
  uint64_t* p = (uint64_t*)&key;
  size_t seed = 2166136261;
  auto hasher = cuco::detail::MurmurHash3_32<uint64_t>(seed);
  for (size_t i = 0; i < sizeof(map_key_t) / 8; i++)
    seed ^= hasher(p[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}

template <typename map_key_t, typename value_t>
__global__ void clear_map();

template <typename map_key_t, typename value_t>
struct device_map_t {
  device_map_t(solution_handle_t<int, float> const* handle_ptr_,
               int max_level_,
               uint32_t max_size_ = 100000)
    : max_level(max_level_),
      max_size(max_size_),
      max_available(4 * max_size_),
      locks(max_level_ * max_available, handle_ptr_->get_stream()),
      keys(max_level_ * max_available, handle_ptr_->get_stream()),
      values(max_level_ * max_available, handle_ptr_->get_stream()),
      predecessors(max_level_ * max_available, handle_ptr_->get_stream()),
      occupied_indices(max_level_ * max_size, handle_ptr_->get_stream()),
      size_per_head(max_level_ * max_size, handle_ptr_->get_stream()),
      occupied(max_level_, handle_ptr_->get_stream()),
      stop_inserting(max_level_, handle_ptr_->get_stream()),
      max_threads(handle_ptr_->get_device_properties().maxThreadsPerBlock),
      max_blocks(handle_ptr_->get_device_properties().maxGridSize[0])
  {
  }

  void clear(rmm::cuda_stream_view stream);

  uint32_t get_max_size() const
  {
    uint32_t adj_max_size = max_size;
    return adj_max_size;
  }

  size_t get_size(int level, rmm::cuda_stream_view stream) const
  {
    return std::min(get_max_size(), occupied.element(level, stream));
  }

  struct view_t {
    DI void add(map_key_t const key, value_t const value, uint32_t const pred)
    {
      size_t index    = hash(key) % max_available;
      auto curr_value = values[index];
      // early exit logic
      while (true) {
        if (curr_value == std::numeric_limits<double>::max()) {
          break;
        } else if (keys[index] == key) {
          if (value >= values[index])
            return;
          else
            break;
        }
        index      = (index + 1) % max_available;
        curr_value = values[index];
      }

      while (true) {
        if (acquire_lock(&locks[index])) {
          if (keys[index] == key) {
            if (value < values[index]) {
              values[index]       = value;
              predecessors[index] = pred;
            }
            release_lock(&locks[index]);
            return;
          }

          if (keys[index].empty()) {
            auto offset = atomicAdd(occupied, 1);
            if (offset < max_size) {
              keys[index]              = key;
              values[index]            = value;
              predecessors[index]      = pred;
              occupied_indices[offset] = {(int)index, (int)key.head};
              atomicAdd(&size_per_head[key.head], 1);
            }
            release_lock(&locks[index]);
            return;
          } else {
            release_lock(&locks[index]);
            index = (index + 1) % max_available;
          }
        }
      }
    }

    uint32_t* occupied;
    int* stop_inserting;
    int max_level{};
    int level{};
    uint32_t max_size{};
    size_t max_available{};
    raft::device_span<map_key_t> keys;
    raft::device_span<value_t> values;
    raft::device_span<uint32_t> predecessors;
    raft::device_span<int2> occupied_indices;
    raft::device_span<int> size_per_head;
    raft::device_span<int> locks;
  };

  view_t subspan(int level)
  {
    view_t v;
    v.keys   = raft::device_span<map_key_t>{keys.data() + level * max_available, max_available};
    v.values = raft::device_span<value_t>{values.data() + level * max_available, max_available};
    v.predecessors =
      raft::device_span<uint32_t>{predecessors.data() + level * max_available, max_available};
    v.occupied_indices =
      raft::device_span<int2>{occupied_indices.data() + level * max_size, max_size};
    v.size_per_head  = raft::device_span<int>{size_per_head.data() + level * max_size, max_size};
    v.locks          = raft::device_span<int>{locks.data() + level * max_available, max_available};
    v.occupied       = occupied.data() + level;
    v.stop_inserting = stop_inserting.data() + level;
    v.max_available  = max_available;
    v.max_size       = get_max_size();
    v.level          = level;
    return v;
  }

  view_t view()
  {
    view_t v;
    v.keys             = raft::device_span<map_key_t>{keys.data(), keys.size()};
    v.values           = raft::device_span<value_t>{values.data(), values.size()};
    v.predecessors     = raft::device_span<uint32_t>{predecessors.data(), predecessors.size()};
    v.occupied_indices = raft::device_span<int2>{occupied_indices.data(), occupied_indices.size()};
    v.size_per_head    = raft::device_span<int>{size_per_head.data(), size_per_head.size()};
    v.locks            = raft::device_span<int>{locks.data(), locks.size()};
    v.occupied         = occupied.data();
    v.stop_inserting   = stop_inserting.data();
    v.max_available    = max_available;
    v.max_level        = max_level;
    v.max_size         = max_size;
    return v;
  }

  int max_level{};
  uint32_t max_size{};
  size_t max_threads{};
  size_t max_blocks{};
  size_t max_available{};
  rmm::device_uvector<uint32_t> occupied;
  rmm::device_uvector<int> stop_inserting;
  rmm::device_uvector<map_key_t> keys;
  rmm::device_uvector<value_t> values;
  rmm::device_uvector<uint32_t> predecessors;
  rmm::device_uvector<int2> occupied_indices;
  rmm::device_uvector<int> size_per_head;
  rmm::device_uvector<int> locks;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
