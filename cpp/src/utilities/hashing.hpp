/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace cuopt::linear_programming::detail {

template <typename i_t>
inline uint32_t compute_hash(const std::vector<i_t>& h_contents)
{
  // FNV-1a hash

  uint32_t hash = 2166136261u;  // FNV-1a 32-bit offset basis
  std::vector<uint8_t> byte_contents(h_contents.size() * sizeof(i_t));
  std::memcpy(byte_contents.data(), h_contents.data(), h_contents.size() * sizeof(i_t));
  for (size_t i = 0; i < byte_contents.size(); ++i) {
    hash ^= byte_contents[i];
    hash *= 16777619u;
  }
  return hash;
}

template <typename i_t>
#if defined(__CUDACC__)
__host__ __device__
#endif
  inline uint32_t
  compute_hash(const i_t val)
{
  uint32_t hash = 2166136261u;
  uint8_t byte_contents[sizeof(i_t)];
  std::memcpy(byte_contents, &val, sizeof(i_t));
  for (size_t i = 0; i < sizeof(i_t); ++i) {
    hash ^= byte_contents[i];
    hash *= 16777619u;
  }
  return hash;
}

}  // namespace cuopt::linear_programming::detail
