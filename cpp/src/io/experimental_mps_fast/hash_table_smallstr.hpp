/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <simde/x86/avx2.h>
#include <simde/x86/sse4.2.h>

#include <cstdint>
#include <cstring>

#define __assume(cond)                    \
  do {                                    \
    if (!(cond)) __builtin_unreachable(); \
  } while (0)

#define BUCKET_COUNT (4194304 * 2 * 2 * 4)  // 2^22

// Set to 1 for 32-byte keys, 0 for 16-byte keys
#ifndef USE_32B_HASH_KEYS
#define USE_32B_HASH_KEYS 1
#endif

namespace mps_fast {

static inline uint32_t crcHash(const uint8_t* key, int64_t len)
{
  __assume(len < 256);

  uint64_t crc = 0;
  while (len > 8) {
    uint64_t val = *(const uint64_t*)key;
    crc          = simde_mm_crc32_u64(crc, val);
    len -= 8;
    key += 8;
  }

  // CRC the final 1-7 bytes
  uint64_t val = *(const uint64_t*)key;
  val &= ~(~0ULL << len * 8);  // Compiles to a bzhi instruction (also UB)
  crc = simde_mm_crc32_u64(crc, val);

  return crc;
}

static inline uint32_t crcHash32B(uint64_t q0, uint64_t q1, uint64_t q2, uint64_t q3)
{
  uint64_t crc = 0;
  crc          = simde_mm_crc32_u64(crc, q0);
  crc          = simde_mm_crc32_u64(crc, q1);
  crc          = simde_mm_crc32_u64(crc, q2);
  crc          = simde_mm_crc32_u64(crc, q3);

  return crc;
}

// FNV-1a hash, processes bytes in reverse to better handle common-prefix strings
static inline uint32_t fnv1a_hash(const char* ptr, size_t len)
{
  constexpr uint32_t FNV_OFFSET = 2166136261u;
  constexpr uint32_t FNV_PRIME  = 16777619u;

  uint32_t h    = FNV_OFFSET;
  const char* p = ptr + len;
  while (p > ptr) {
    --p;
    h ^= (uint8_t)*p;
    h *= FNV_PRIME;
  }
  return h;
}

struct __attribute__((packed)) hash_slot_32_t {
  uint32_t count;
  simde__m256i node;
};

struct alignas(16) hash_slot_16_t {
  char key[16];
  uint32_t count;
};

static inline bool key_cmpeq_16(const char* slot_key, simde__m128i key)
{
  simde__m128i slot_vec = simde_mm_loadu_si128((const simde__m128i*)slot_key);
  int mask              = simde_mm_movemask_epi8(simde_mm_cmpeq_epi8(slot_vec, key));
  return mask == 0xFFFF;
}

// 32-byte aligned slot: 28-byte key + 4-byte count = 32 bytes total (one cache line half)
struct alignas(32) hash_slot_28_t {
  char key[28];
  uint32_t count;
};

static inline simde__m256i make_key_28(const char* ptr, size_t len)
{
  alignas(32) char buf[32] = {0};
  size_t copy_len          = len < 28 ? len : 28;
  std::memcpy(buf, ptr, copy_len);
  return simde_mm256_load_si256((const simde__m256i*)buf);
}

// Compare 28-byte keys stored in simde__m256i (ignore last 4 bytes)
static inline bool key_cmpeq_28(const char* slot_key, simde__m256i key)
{
  simde__m256i slot_vec = simde_mm256_loadu_si256((const simde__m256i*)slot_key);
  int mask              = simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi8(slot_vec, key));
  return (mask & 0x0FFFFFFF) == 0x0FFFFFFF;  // Only check first 28 bytes
}

#if USE_32B_HASH_KEYS
using hash_key_t                = simde__m256i;
using hash_slot_var_t           = hash_slot_28_t;
constexpr size_t HASH_KEY_BYTES = 28;
constexpr int HASH_KEY_CMP_MASK = 0x0FFFFFFF;
#define make_key                 make_key_28
#define key_cmpeq(slot_key, key) key_cmpeq_28(slot_key, key)
#define key_store(slot_key, key) simde_mm256_store_si256((simde__m256i*)(slot_key), key)
#else
using hash_key_t                = simde__m128i;
using hash_slot_var_t           = hash_slot_16_t;
constexpr size_t HASH_KEY_BYTES = 16;
constexpr int HASH_KEY_CMP_MASK = 0xFFFF;
#define make_key                 make_key_16
#define key_cmpeq(slot_key, key) key_cmpeq_16(slot_key, key)
#define key_store(slot_key, key) simde_mm_store_si128((simde__m128i*)(slot_key), key)
#endif

// Legacy alias
using hash_slot_t = hash_slot_32_t;

struct hash_table_t {
  hash_slot_t slots[BUCKET_COUNT];
};

static inline void hash_table_push(
  hash_table_t* table, uint32_t hash, simde__m256i val, int len, const uint8_t* ptr)
{
  hash %= BUCKET_COUNT;

  hash_slot_t* slot = &table->slots[hash];

  if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi8(slot->node, val)) == 0xFFFFFFFF) {
    ++slot->count;
    return;
  }

  bool relooped = false;

loop:
  for (; slot < &table->slots[BUCKET_COUNT]; ++slot) {
    if (slot->count == 0) {
      slot->count = 1;
      slot->node  = val;
      return;
    }

    if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi8(slot->node, val)) == 0xFFFFFFFF) {
      ++slot->count;
      return;
    }
  }

  if (!relooped) {
    relooped = true;
    slot     = &table->slots[0];
    goto loop;
  } else {
    __builtin_trap();
  }
}

extern char* string_buffer;
extern char* string_buffer_ptr;

// Lookup: returns the stored value (count-1) or SIZE_MAX if not found
// For small strings <= 32 bytes stored inline in node
static inline size_t hash_table_lookup(const hash_table_t* table, uint32_t hash, simde__m256i val)
{
  hash %= BUCKET_COUNT;
  const hash_slot_t* slot = &table->slots[hash];

  for (size_t i = 0; i < BUCKET_COUNT; ++i, ++slot) {
    if (slot >= &table->slots[BUCKET_COUNT]) { slot = &table->slots[0]; }

    if (slot->count == 0) {
      return SIZE_MAX;  // Not found
    }

    if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi8(slot->node, val)) == (int)0xFFFFFFFF) {
      return slot->count - 1;  // Found, return index
    }
  }

  return SIZE_MAX;  // Not found
}

// Insert with index: stores index+1 in count field (0 means empty)
static inline void hash_table_insert(hash_table_t* table,
                                     uint32_t hash,
                                     simde__m256i val,
                                     size_t index)
{
  hash %= BUCKET_COUNT;
  hash_slot_t* slot = &table->slots[hash];

  for (size_t i = 0; i < BUCKET_COUNT; ++i, ++slot) {
    if (slot >= &table->slots[BUCKET_COUNT]) { slot = &table->slots[0]; }

    if (slot->count == 0) {
      slot->count = (uint32_t)(index + 1);
      slot->node  = val;
      return;
    }

    if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi8(slot->node, val)) == (int)0xFFFFFFFF) {
      // Already exists, update index
      slot->count = (uint32_t)(index + 1);
      return;
    }
  }

  __builtin_trap();
}

// Create simde__m256i key from string_view (zero-padded)
static inline simde__m256i make_key_32(const char* ptr, size_t len)
{
  alignas(32) char buf[32] = {0};
  if (len > 32) len = 32;
  memcpy(buf, ptr, len);
  return simde_mm256_load_si256((const simde__m256i*)buf);
}

// Create simde__m128i key from string_view (zero-padded, for strings <= 16 bytes)
static inline simde__m128i make_key_16(const char* ptr, size_t len)
{
  alignas(16) char buf[16] = {0};
  if (len > 16) len = 16;
  memcpy(buf, ptr, len);
  return simde_mm_load_si128((const simde__m128i*)buf);
}

static inline uint64_t m256_u64_lane(simde__m256i value, size_t lane)
{
  simde__m256i_private private_value = simde__m256i_to_private(value);
  return private_value.u64[lane];
}

static inline void hash_table_push_ptr(hash_table_t* table,
                                       uint32_t hash,
                                       int len,
                                       const uint8_t* ptr)
{
  hash %= BUCKET_COUNT;

  hash_slot_t* slot = &table->slots[hash];
  bool relooped     = false;

  uint32_t len_in_qwords = (len / 8) + (len % 8 ? 1 : 0);

loop:
  do {
    uint64_t node_len = m256_u64_lane(slot->node, 3);
    uint64_t node_tag = m256_u64_lane(slot->node, 0);
    // nonzero, it's not a pointer of the same length, skip
    if (__builtin_expect(node_len != (uint64_t)len, 0)) {
      if (__builtin_expect(node_tag == 0, 1)) {
        slot->count = 1;
        slot->node  = simde_mm256_set_epi64x(len,
                                            ((uint64_t*)ptr)[0],
                                            (uint64_t)string_buffer_ptr,
                                            0u | ((uint64_t)len_in_qwords << 32u));

        memcpy(string_buffer_ptr, ptr, len);
        string_buffer_ptr += len;
        // Pad
        string_buffer_ptr += (8 - len % 8) + 8;

        return;
      } else
        continue;
    }
    if (m256_u64_lane(slot->node, 2) != ((uint64_t*)ptr)[0])  // First 8 bytes differ
      continue;

    uint8_t* other_ptr = reinterpret_cast<uint8_t*>(m256_u64_lane(slot->node, 1));
    if (__builtin_expect(memcmp(ptr + 16, other_ptr + 16, len - 16) == 0, 1)) {
      ++slot->count;

      return;
    }
  } while (++slot < &table->slots[BUCKET_COUNT]);

  if (!relooped) {
    relooped = true;
    slot     = &table->slots[0];
    goto loop;
  } else {
    __builtin_trap();
  }
}

}  // namespace mps_fast
