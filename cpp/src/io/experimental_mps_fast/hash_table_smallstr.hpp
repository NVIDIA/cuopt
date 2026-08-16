/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "mmap_region.hpp"

#include <cuda/cmath>

#include <simde/x86/avx2.h>

#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#ifdef MPS_FAST_PERF_COUNTERS
#include <cstdio>
#endif
#include <limits>
#include <string_view>
#include <unordered_map>

namespace cuopt::mathematical_optimization::io::detail {

// below this threshold, the serial row-hash build is usually cheaper than partition setup
inline constexpr size_t MPS_ROW_HASH_PARTITIONED_MIN_ROWS = 64 * 1024;
inline constexpr int MPS_ROW_HASH_PARTITION_BITS          = 5;
inline constexpr size_t MPS_ROW_HASH_PARTITIONS           = (1 << MPS_ROW_HASH_PARTITION_BITS);

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

struct hash_partition_t {
  hash_slot_var_t* slots = nullptr;
  size_t buckets         = 0;
  size_t mask            = 0;
};

static inline size_t hash_partition_for(uint32_t hash)
{
  return (size_t)(hash >> (32 - MPS_ROW_HASH_PARTITION_BITS));
}

static inline size_t hash_bucket_count_for(size_t n_rows, bool compact)
{
  if (compact) { return cuda::next_power_of_two(std::max(n_rows + n_rows / 2, (size_t)64)); }
  return cuda::next_power_of_two(std::max(n_rows * 2, (size_t)64));
}

static inline size_t hash_lookup_in(
  const hash_slot_var_t* slots, size_t buckets, size_t mask, hash_key_t key, uint32_t hash)
{
  const hash_slot_var_t* slot = &slots[hash & (uint32_t)mask];
  for (size_t i = 0; i < buckets; ++i, ++slot) {
    if (slot >= &slots[buckets]) { slot = &slots[0]; }
    if (slot->count == 0) { return std::numeric_limits<size_t>::max(); }
    if (key_cmpeq(slot->key, key)) { return slot->count - 1; }
  }
  return std::numeric_limits<size_t>::max();
}

static inline size_t hash_insert_into(hash_slot_var_t* slots,
                                      size_t buckets,
                                      size_t mask,
                                      std::string_view name,
                                      uint32_t hash,
                                      size_t index)
{
  hash_key_t key        = make_key(name.data(), name.size());
  hash_slot_var_t* slot = &slots[hash & (uint32_t)mask];
  for (size_t i = 0; i < buckets; ++i, ++slot) {
    if (slot >= &slots[buckets]) { slot = &slots[0]; }
    if (slot->count == 0) {
      key_store(slot->key, key);
      slot->count = (uint32_t)(index + 1);
      return i + 1;
    }
    if (key_cmpeq(slot->key, key)) {
      slot->count = (uint32_t)(index + 1);
      return i + 1;
    }
  }
  __builtin_unreachable();
}

#ifdef MPS_FAST_PERF_COUNTERS
struct hash_build_probe_stats_t {
  size_t total_probes = 0;
  size_t max_probes   = 0;
  size_t long_names   = 0;

  void seed_long_names(size_t n) { long_names = n; }

  void record_insert(size_t probes)
  {
    if (probes == 0) {
      ++long_names;
    } else {
      total_probes += probes;
      max_probes = std::max(max_probes, probes);
    }
  }

  void merge(const hash_build_probe_stats_t& other)
  {
    total_probes += other.total_probes;
    max_probes = std::max(max_probes, other.max_probes);
    long_names += other.long_names;
  }
};
#endif

class smallstr_hash_table_t {
 public:
  void note_long_name(std::string_view name, size_t index) { long_names_[name] = index; }

  size_t long_name_count() const { return long_names_.size(); }

  void reset_build_probe_stats()
  {
#ifdef MPS_FAST_PERF_COUNTERS
    build_probe_stats_ = {};
    build_probe_stats_.seed_long_names(long_names_.size());
    partition_probe_stats_ = {};
#endif
  }

  void print_build_probe_report(size_t n_rows) const
  {
#ifdef MPS_FAST_PERF_COUNTERS
    hash_build_probe_stats_t stats = build_probe_stats_;
    if (partition_count_ != 0) {
      for (size_t p = 0; p < partition_count_; ++p) {
        stats.merge(partition_probe_stats_[p]);
      }
    }
    size_t probed_rows = n_rows - stats.long_names;
    double mean_probes = probed_rows == 0 ? 0.0 : (double)stats.total_probes / (double)probed_rows;
    double load_factor = buckets_ == 0 ? 0.0 : (double)n_rows / (double)buckets_;
    std::fprintf(stderr,
                 "[ROW_HASH_PROBES] rows=%zu buckets=%zu load=%.3f long=%zu mean=%.3f max=%zu\n",
                 n_rows,
                 buckets_,
                 load_factor,
                 stats.long_names,
                 mean_probes,
                 stats.max_probes);
#endif
  }

  void configure_serial_buckets(size_t n_rows, bool compact)
  {
    partition_count_ = 0;
    buckets_         = hash_bucket_count_for(n_rows, compact);
    mask_            = buckets_ - 1;
  }

  void configure_partitioned_buckets(
    const std::array<size_t, MPS_ROW_HASH_PARTITIONS>& partition_counts, bool compact)
  {
    partition_count_ = MPS_ROW_HASH_PARTITIONS;
    buckets_         = 0;
    for (size_t p = 0; p < MPS_ROW_HASH_PARTITIONS; ++p) {
      partitions_[p].buckets = hash_bucket_count_for(partition_counts[p], compact);
      partitions_[p].mask    = partitions_[p].buckets - 1;
      buckets_ += partitions_[p].buckets;
    }
    mask_ = buckets_ - 1;
  }

  void allocate_mmap(const char* label)
  {
    size_t mmap_size = buckets_ * sizeof(hash_slot_var_t);
    region_ = mmap_region_t::anonymous(mmap_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, label);
    slots_  = (hash_slot_var_t*)region_.data();
    if (partition_count_ != 0) {
      hash_slot_var_t* next_slots = slots_;
      for (size_t p = 0; p < partition_count_; ++p) {
        partitions_[p].slots = next_slots;
        next_slots += partitions_[p].buckets;
      }
    }
    region_.advise(MADV_HUGEPAGE);
  }

  mmap_region_t& region() noexcept { return region_; }
  const mmap_region_t& region() const noexcept { return region_; }

  hash_slot_var_t* slots() noexcept { return slots_; }
  const hash_slot_var_t* slots() const noexcept { return slots_; }

  size_t buckets() const noexcept { return buckets_; }
  size_t mask() const noexcept { return mask_; }
  size_t partition_count() const noexcept { return partition_count_; }

  const hash_partition_t& partition(size_t p) const noexcept { return partitions_[p]; }

  size_t lookup(std::string_view name) const
  {
    if (name.size() > HASH_KEY_BYTES) {
      auto it = long_names_.find(name);
      return it != long_names_.end() ? it->second : std::numeric_limits<size_t>::max();
    }
    hash_key_t key = make_key(name.data(), name.size());
    uint32_t hash  = fnv1a_hash(name.data(), name.size());
    if (partition_count_ != 0) {
      const auto& part = partitions_[hash_partition_for(hash)];
      return hash_lookup_in(part.slots, part.buckets, part.mask, key, hash);
    }
    return hash_lookup_in(slots_, buckets_, mask_, key, hash);
  }

  size_t insert_serial(std::string_view name, size_t index)
  {
    size_t probes;
    if (name.size() > HASH_KEY_BYTES) {
      note_long_name(name, index);
      probes = 0;
    } else {
      probes = hash_insert_into(
        slots_, buckets_, mask_, name, fnv1a_hash(name.data(), name.size()), index);
    }
#ifdef MPS_FAST_PERF_COUNTERS
    build_probe_stats_.record_insert(probes);
#endif
    return probes;
  }

  size_t insert_partition(size_t partition, std::string_view name, uint32_t hash, size_t index)
  {
    const auto& part = partitions_[partition];
    size_t probes    = hash_insert_into(part.slots, part.buckets, part.mask, name, hash, index);
#ifdef MPS_FAST_PERF_COUNTERS
    partition_probe_stats_[partition].record_insert(probes);
#endif
    return probes;
  }

 private:
  mmap_region_t region_;
  hash_slot_var_t* slots_ = nullptr;
  size_t buckets_         = 0;
  size_t mask_            = 0;
  size_t partition_count_ = 0;
  std::array<hash_partition_t, MPS_ROW_HASH_PARTITIONS> partitions_{};
  std::unordered_map<std::string_view, size_t> long_names_{};
#ifdef MPS_FAST_PERF_COUNTERS
  hash_build_probe_stats_t build_probe_stats_{};
  std::array<hash_build_probe_stats_t, MPS_ROW_HASH_PARTITIONS> partition_probe_stats_{};
#endif
};

}  // namespace cuopt::mathematical_optimization::io::detail
