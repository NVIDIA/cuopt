/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <simde/x86/avx2.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mps_fast {

// FNV-1a over bytes in reverse order; row names commonly share long prefixes.
static inline uint32_t fnv1a_hash(const char* ptr, std::size_t len)
{
  constexpr uint32_t fnv_offset = 2166136261u;
  constexpr uint32_t fnv_prime  = 16777619u;

  uint32_t h    = fnv_offset;
  const char* p = ptr + len;
  while (p > ptr) {
    --p;
    h ^= (uint8_t)*p;
    h *= fnv_prime;
  }
  return h;
}

// 28-byte inline key + uint32 payload: two slots per 64-byte cache line.
// key_store writes a full 32-byte vector starting at key[0], so callers must
// publish the payload after storing the key. key_cmpeq masks those payload lanes
// away, leaving the trailing uint32 free for the row index + 1 sentinel.
struct alignas(32) hash_slot_28_t {
  char key[28];
  uint32_t count;
};

using hash_key_t                     = simde__m256i;
using hash_slot_var_t                = hash_slot_28_t;
constexpr std::size_t HASH_KEY_BYTES = 28;

static_assert(sizeof(hash_slot_28_t) == 32);
static_assert(alignof(hash_slot_28_t) == 32);
static_assert(offsetof(hash_slot_28_t, count) == HASH_KEY_BYTES);

static inline hash_key_t make_key(const char* ptr, std::size_t len)
{
  alignas(32) char buf[32] = {};
  std::memcpy(buf, ptr, len < HASH_KEY_BYTES ? len : HASH_KEY_BYTES);
  return simde_mm256_load_si256(reinterpret_cast<const simde__m256i*>(buf));
}

static inline bool key_cmpeq(const char* slot_key, hash_key_t key)
{
  simde__m256i slot_vec = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(slot_key));
  int mask              = simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi8(slot_vec, key));
  return (mask & 0x0fffffff) == 0x0fffffff;
}

static inline void key_store(char* slot_key, hash_key_t key)
{
  simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(slot_key), key);
}

}  // namespace mps_fast
