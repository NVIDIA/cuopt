// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: Apache-2.0

#include "mps_section_scanner.hpp"

#include <utilities/error.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

#include <simde/x86/avx2.h>
#include <simde/x86/sse4.2.h>

namespace cuopt::linear_programming::io::detail {

using cuopt::linear_programming::io::error_type_t;
using cuopt::linear_programming::io::mps_parser_expects;
using cuopt::linear_programming::io::mps_parser_fail;

namespace {

struct section_record_t {
  mps_section_kind kind;
  const char* name;
  std::size_t len;
};

constexpr section_record_t section_records[] = {
  {mps_section_kind::rows, "ROWS", 4},
  {mps_section_kind::columns, "COLUMNS", 7},
  {mps_section_kind::rhs, "RHS", 3},
  {mps_section_kind::bounds, "BOUNDS", 6},
  {mps_section_kind::ranges, "RANGES", 6},
  {mps_section_kind::quadobj, "QUADOBJ", 7},
  {mps_section_kind::qmatrix, "QMATRIX", 7},
  {mps_section_kind::qcmatrix, "QCMATRIX", 8},
  {mps_section_kind::endata, "ENDATA", 6},
};

constexpr const char* header_records[] = {"NAME", "OBJSENSE", "OBJNAME"};

constexpr std::size_t kSimdWidth = sizeof(simde__m256i);
static_assert(kSimdWidth == 32);
static_assert((std::size_t)mps_section_kind::rows == 0);
static_assert((std::size_t)mps_section_kind::endata + 1 == std::size(section_records));
static_assert((std::size_t)mps_phase_kind::header == 0);
static_assert((std::size_t)mps_phase_kind::quadratic + 1 == 7);

bool is_nonblank_column1(unsigned char c) noexcept { return c > ' '; }

simde__m256i nonblank_column1_mask(simde__m256i bytes)
{
  return simde_mm256_cmpgt_epi8(bytes, simde_mm256_set1_epi8(' '));
}

enum class section_record_match_t { invalid, header, section };

bool line_has_record_prefix(const char* line_start, const char* line_end, const char* name)
{
  std::size_t len = std::strlen(name);
  if ((std::size_t)(line_end - line_start) < len || std::memcmp(line_start, name, len) != 0) {
    return false;
  }
  const char* after = line_start + len;
  return after == line_end || *after <= ' ';
}

}  // namespace

std::size_t mps_phase_registry_t::phase_index(mps_phase_kind phase) { return (std::size_t)phase; }

void mps_phase_registry_t::publish(mps_phase_kind phase, mps_phase_range_t range)
{
  std::size_t idx = phase_index(phase);
  omp_event_handle_t event{};
  bool fulfill = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ready_[idx].load(std::memory_order_acquire)) { return; }
    ranges_[idx] = range;
    ready_[idx].store(true, std::memory_order_release);
    if (has_event_[idx] && !event_fulfilled_[idx]) {
      event                 = events_[idx];
      event_fulfilled_[idx] = true;
      fulfill               = true;
    }
  }
  if (fulfill) { omp_fulfill_event(event); }
}

void mps_phase_registry_t::attach_event(mps_phase_kind phase, omp_event_handle_t event)
{
  std::size_t idx = phase_index(phase);
  bool fulfill    = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    events_[idx]    = event;
    has_event_[idx] = true;
    if (ready_[idx].load(std::memory_order_acquire) && !event_fulfilled_[idx]) {
      event_fulfilled_[idx] = true;
      fulfill               = true;
    }
  }
  if (fulfill) { omp_fulfill_event(event); }
}

bool mps_phase_registry_t::ready(mps_phase_kind phase) const
{
  return ready_[phase_index(phase)].load(std::memory_order_acquire);
}

mps_phase_range_t mps_phase_registry_t::range(mps_phase_kind phase) const
{
  std::size_t idx = phase_index(phase);
  bool is_ready   = ready_[idx].load(std::memory_order_acquire);
  assert(is_ready);
  return ranges_[idx];
}

void mps_phase_registry_t::publish_endata(const char* begin, bool present)
{
  std::lock_guard<std::mutex> lock(mutex_);
  endata_begin_   = begin;
  endata_present_ = present;
  endata_ready_.store(true, std::memory_order_release);
}

bool mps_phase_registry_t::endata_ready() const
{
  return endata_ready_.load(std::memory_order_acquire);
}

const char* mps_phase_registry_t::endata_begin() const
{
  assert(endata_ready());
  return endata_begin_;
}

bool mps_phase_registry_t::endata_present() const
{
  assert(endata_ready());
  return endata_present_;
}

static section_record_match_t is_section_record(const char* line_start,
                                                const char* line_end,
                                                mps_section_kind* kind)
{
  if (line_start >= line_end) { return section_record_match_t::invalid; }

  for (const char* name : header_records) {
    if (line_has_record_prefix(line_start, line_end, name)) {
      return section_record_match_t::header;
    }
  }

  for (const section_record_t& record : section_records) {
    if ((std::size_t)(line_end - line_start) < record.len ||
        std::memcmp(line_start, record.name, record.len) != 0) {
      continue;
    }
    const char* after = line_start + record.len;
    while (after < line_end && (*after == ' ' || *after == '\t' || *after == '\r')) {
      ++after;
    }
    if (after != line_end) { return section_record_match_t::invalid; }
    *kind = record.kind;
    return section_record_match_t::section;
  }
  return section_record_match_t::invalid;
}

mps_section_block_scanner_t::mps_section_block_scanner_t(const char* data,
                                                         std::size_t block_count,
                                                         mps_phase_registry_t& registry)
  : data_(data),
    block_count_(block_count),
    registry_(registry),
    block_decoded_(std::make_unique<std::atomic<unsigned char>[]>(block_count)),
    block_begin_offsets_(std::make_unique<std::atomic_size_t[]>(block_count)),
    block_end_offsets_(std::make_unique<std::atomic_size_t[]>(block_count))
{
  for (std::size_t i = 0; i < block_count_; ++i) {
    block_decoded_[i].store(0, std::memory_order_relaxed);
    block_begin_offsets_[i].store(0, std::memory_order_relaxed);
    block_end_offsets_[i].store(0, std::memory_order_relaxed);
  }
}

std::size_t mps_section_block_scanner_t::section_hit_index(mps_section_kind kind)
{
  return (std::size_t)kind;
}

void mps_section_block_scanner_t::record_section_hit(mps_section_kind kind, const char* ptr)
{
  std::atomic<const char*>& slot = section_hits_[section_hit_index(kind)];
  const char* expected           = nullptr;
  if (slot.compare_exchange_strong(
        expected, ptr, std::memory_order_release, std::memory_order_acquire)) {
    notify_ready_phases();
  }
}

void mps_section_block_scanner_t::scan_section_range(const char* begin, const char* end)
{
  if (begin >= end) return;
  const char* p = begin;

  // Interior scans that start inside a decoded block skip the leading partial
  // line. A separate boundary scan covers section titles whose newline/title
  // bytes straddle adjacent LZ4 blocks.
  if (p != data_) {
    const void* nl = __builtin_memchr(p, '\n', (std::size_t)(end - p));
    if (nl == nullptr) { return; }
    p = (const char*)nl + 1;
  }

  auto try_candidate = [&](const char* line_start) {
    const void* nl       = __builtin_memchr(line_start, '\n', (std::size_t)(end - line_start));
    const char* line_end = nullptr;
    if (nl == nullptr) {
      const char* ready_ptr = data_ + ready_bytes_.load(std::memory_order_acquire);
      if (end != ready_ptr) { return; }
      line_end = end;
    } else {
      line_end = (const char*)nl;
    }
    if (*line_start == '*' || *line_start == '$') { return; }
    mps_section_kind kind;
    section_record_match_t match = is_section_record(line_start, line_end, &kind);
    if (match == section_record_match_t::section) {
      record_section_hit(kind, line_start);
      return;
    }
    if (match == section_record_match_t::invalid) {
      mps_parser_fail(error_type_t::ValidationError,
                      "unknown section record: %.*s",
                      (int)(line_end - line_start),
                      line_start);
    }
  };

  // Handle the very first line of a file (NAME indicator, usually)
  if (p == data_) {
    if (p < end && is_nonblank_column1((unsigned char)*p)) { try_candidate(p); }
    ++p;
  }

  // In compliant MPS, indicator records begin in column 1 while data records
  // begin in column 2+. use "\n[nonblank]" as a needle for the SIMD scan
  const simde__m256i newline = simde_mm256_set1_epi8('\n');
  while ((std::size_t)(end - p) >= kSimdWidth) {
    // The first-line path above increments p when p == data_, so p - 1 is
    // in-bounds here. Loading the previous vector lets us test "\nX" for all
    // 32 candidate column-1 bytes with one AVX2 mask.
    // loadu is comparable to aligned reads on modern SSE/AVX.
    // might warrant some checks on ARM though
    simde__m256i current  = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(p));
    simde__m256i previous = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(p - 1));
    std::uint32_t mask    = (std::uint32_t)simde_mm256_movemask_epi8(simde_mm256_and_si256(
      simde_mm256_cmpeq_epi8(previous, newline), nonblank_column1_mask(current)));
    while (mask != 0) {
      int bit = __builtin_ctz(mask);
      try_candidate(p + bit);
      mask &= mask - 1;
    }
    p += kSimdWidth;
  }

  // scalar tail
  while (p < end) {
    if (*(p - 1) == '\n' && is_nonblank_column1((unsigned char)*p)) { try_candidate(p); }
    ++p;
  }
}

void mps_section_block_scanner_t::scan_boundary(std::size_t left_index, std::size_t right_index)
{
  std::size_t left_begin = block_begin_offsets_[left_index].load(std::memory_order_acquire);
  std::size_t boundary   = block_begin_offsets_[right_index].load(std::memory_order_acquire);
  std::size_t right_end  = block_end_offsets_[right_index].load(std::memory_order_acquire);
  std::size_t begin =
    boundary - left_begin > boundary_overlap ? boundary - boundary_overlap : left_begin;
  std::size_t end =
    right_end - boundary > boundary_overlap ? boundary + boundary_overlap : right_end;
  scan_section_range(data_ + begin, data_ + end);
}

// scans a freshly decoded block for section titles, along with the start/end boundaries if a
// section title straddles blocks
void mps_section_block_scanner_t::observe_block(std::size_t block_index,
                                                const char* begin,
                                                const char* end)
{
  if (block_index >= block_count_) {
    mps_parser_fail(error_type_t::RuntimeError,
                    "MPS section scanner observed invalid LZ4 block index");
  }

  // --- Scan this block, then record its extent and mark it decoded. The release store on
  //     block_decoded_ publishes the two relaxed offset stores above it.
  scan_section_range(begin, end);
  block_begin_offsets_[block_index].store((std::size_t)(begin - data_), std::memory_order_relaxed);
  block_end_offsets_[block_index].store((std::size_t)(end - data_), std::memory_order_relaxed);
  block_decoded_[block_index].store(1, std::memory_order_release);

  // --- Rescan the seams with already-decoded neighbors, in case a title straddles the boundary.
  if (block_index > 0 && block_decoded_[block_index - 1].load(std::memory_order_acquire)) {
    scan_boundary(block_index - 1, block_index);
  }
  if (block_index + 1 < block_count_ &&
      block_decoded_[block_index + 1].load(std::memory_order_acquire)) {
    scan_boundary(block_index, block_index + 1);
  }

  // --- Extend the contiguous decoded-byte frontier and publish any newly bounded phases.
  advance_ready_frontier();
}

void mps_section_block_scanner_t::advance_ready_frontier()
{
  std::size_t new_ready = 0;
  bool grew             = false;
  {
    std::lock_guard<std::mutex> lock(frontier_mutex_);
    while (next_block_ < block_count_ &&
           block_decoded_[next_block_].load(std::memory_order_acquire)) {
      new_ready = block_end_offsets_[next_block_].load(std::memory_order_acquire);
      ++next_block_;
      grew = true;
    }
  }
  if (grew) { publish_ready(new_ready); }
}

void mps_section_block_scanner_t::publish_ready(std::size_t ready_bytes)
{
  ready_bytes_.store(ready_bytes, std::memory_order_release);
  std::size_t begin = ready_bytes > boundary_overlap ? ready_bytes - boundary_overlap : 0;
  scan_section_range(data_ + begin, data_ + ready_bytes);
  notify_ready_phases();
}

std::size_t mps_section_block_scanner_t::ready_bytes() const noexcept
{
  return ready_bytes_.load(std::memory_order_acquire);
}

void mps_section_block_scanner_t::notify_ready_phases()
{
  // Publication model: each present phase runs from its own section header to
  // the first later section header that has been discovered. Optional sections
  // publish present=false once a later boundary proves they cannot still appear.
  // ENDATA, or final ready bytes for truncated/non-newline files, is the final
  // boundary for the trailing optional/quadratic phases.
  std::lock_guard<std::mutex> lock(publish_mutex_);
  std::size_t ready     = ready_bytes_.load(std::memory_order_acquire);
  const char* ready_ptr = data_ + ready;
  const char* rows =
    section_hits_[section_hit_index(mps_section_kind::rows)].load(std::memory_order_acquire);
  const char* columns =
    section_hits_[section_hit_index(mps_section_kind::columns)].load(std::memory_order_acquire);
  const char* rhs =
    section_hits_[section_hit_index(mps_section_kind::rhs)].load(std::memory_order_acquire);
  const char* bounds =
    section_hits_[section_hit_index(mps_section_kind::bounds)].load(std::memory_order_acquire);
  const char* ranges =
    section_hits_[section_hit_index(mps_section_kind::ranges)].load(std::memory_order_acquire);
  const char* quadobj =
    section_hits_[section_hit_index(mps_section_kind::quadobj)].load(std::memory_order_acquire);
  const char* qmatrix =
    section_hits_[section_hit_index(mps_section_kind::qmatrix)].load(std::memory_order_acquire);
  const char* qcmatrix =
    section_hits_[section_hit_index(mps_section_kind::qcmatrix)].load(std::memory_order_acquire);
  const char* endata =
    section_hits_[section_hit_index(mps_section_kind::endata)].load(std::memory_order_acquire);
  auto available = [&](const char* p) { return p != nullptr && p <= ready_ptr; };
  bool final_ready =
    block_count_ == 0 ||
    (block_decoded_[block_count_ - 1].load(std::memory_order_acquire) &&
     ready == block_end_offsets_[block_count_ - 1].load(std::memory_order_acquire));
  const char* final_boundary    = available(endata) ? endata : (final_ready ? ready_ptr : nullptr);
  auto earliest_available_after = [&](const char* after,
                                      std::initializer_list<const char*> candidates) {
    const char* best = nullptr;
    for (const char* p : candidates) {
      if (!available(p) || (after != nullptr && p <= after)) { continue; }
      if (best == nullptr || p < best) { best = p; }
    }
    return best;
  };
  auto publish_optional = [&](mps_phase_kind phase,
                              const char* self,
                              const char* predecessor,
                              std::initializer_list<const char*> later_candidates) {
    if (registry_.ready(phase)) { return; }
    if (available(self)) {
      const char* end = earliest_available_after(self, later_candidates);
      if (end != nullptr) { registry_.publish(phase, {self, end, true}); }
      return;
    }
    if (predecessor != nullptr &&
        earliest_available_after(predecessor, later_candidates) != nullptr) {
      registry_.publish(phase, {nullptr, nullptr, false});
    }
  };

  // Three publication shapes follow:
  //   (1) mandatory header/rows/columns -- each spans from its start to the next mandatory
  //       section; published as soon as that bounding section is available.
  //   (2) optional rhs/ranges/bounds via publish_optional -- present=true once bounded, or
  //       present=false once a later section proves the optional one cannot still appear.
  //   (3) quadratic -- starts at the earliest of the three quad markers (quadobj/qmatrix/qcmatrix).
  // final_boundary (ENDATA, or the final ready frontier for truncated files) closes the tail.
  if (available(rows) && !registry_.ready(mps_phase_kind::header)) {
    registry_.publish(mps_phase_kind::header, {data_, rows, true});
  }
  if (available(rows) && available(columns) && !registry_.ready(mps_phase_kind::rows)) {
    registry_.publish(mps_phase_kind::rows, {rows, columns, true});
  }
  if (available(columns) && !registry_.ready(mps_phase_kind::columns)) {
    const char* columns_end = earliest_available_after(
      columns, {rhs, ranges, bounds, quadobj, qmatrix, qcmatrix, final_boundary});
    if (columns_end != nullptr) {
      registry_.publish(mps_phase_kind::columns, {columns, columns_end, true});
    }
  }

  publish_optional(mps_phase_kind::rhs,
                   rhs,
                   columns,
                   {ranges, bounds, quadobj, qmatrix, qcmatrix, final_boundary});
  publish_optional(mps_phase_kind::ranges,
                   ranges,
                   rhs ? rhs : columns,
                   {bounds, quadobj, qmatrix, qcmatrix, final_boundary});
  publish_optional(mps_phase_kind::bounds,
                   bounds,
                   ranges ? ranges : (rhs ? rhs : columns),
                   {quadobj, qmatrix, qcmatrix, final_boundary});

  if (!registry_.ready(mps_phase_kind::quadratic)) {
    const char* quadratic_begin = nullptr;
    if (available(quadobj)) { quadratic_begin = quadobj; }
    if (available(qmatrix) && (quadratic_begin == nullptr || qmatrix < quadratic_begin)) {
      quadratic_begin = qmatrix;
    }
    if (available(qcmatrix) && (quadratic_begin == nullptr || qcmatrix < quadratic_begin)) {
      quadratic_begin = qcmatrix;
    }
    if (quadratic_begin != nullptr && final_boundary != nullptr) {
      registry_.publish(mps_phase_kind::quadratic, {quadratic_begin, final_boundary, true});
    } else if (quadratic_begin == nullptr && final_boundary != nullptr) {
      registry_.publish(mps_phase_kind::quadratic, {nullptr, nullptr, false});
    }
  }

  if (available(endata)) {
    registry_.publish_endata(endata, true);
  } else if (final_ready && final_boundary != nullptr) {
    registry_.publish_endata(final_boundary, false);
  }
}

}  // namespace cuopt::linear_programming::io::detail
