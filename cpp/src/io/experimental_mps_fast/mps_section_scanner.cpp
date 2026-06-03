// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: Apache-2.0

#include "mps_section_scanner.hpp"
#include "simd_compat.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

namespace mps_fast {

namespace {

bool is_nonblank_column1(unsigned char c) noexcept { return c > ' '; }

simde__m256i nonblank_column1_mask(simde__m256i bytes)
{
  return simde_mm256_cmpgt_epi8(bytes, simde_mm256_set1_epi8(' '));
}

const char* section_name(mps_section_kind kind)
{
  switch (kind) {
    case mps_section_kind::rows: return "ROWS";
    case mps_section_kind::columns: return "COLUMNS";
    case mps_section_kind::rhs: return "RHS";
    case mps_section_kind::bounds: return "BOUNDS";
    case mps_section_kind::ranges: return "RANGES";
    case mps_section_kind::quadobj: return "QUADOBJ";
    case mps_section_kind::qmatrix: return "QMATRIX";
    case mps_section_kind::qcmatrix: return "QCMATRIX";
    case mps_section_kind::endata: return "ENDATA";
  }
  return "";
}

std::size_t section_name_len(mps_section_kind kind) { return std::strlen(section_name(kind)); }

}  // namespace

std::size_t mps_phase_registry_t::phase_index(mps_phase_kind phase)
{
  switch (phase) {
    case mps_phase_kind::header: return 0;
    case mps_phase_kind::rows: return 1;
    case mps_phase_kind::columns: return 2;
    case mps_phase_kind::rhs: return 3;
    case mps_phase_kind::bounds: return 4;
    case mps_phase_kind::ranges: return 5;
    case mps_phase_kind::quadratic: return 6;
  }
  throw std::runtime_error("invalid MPS phase kind");
}

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
  return ranges_[phase_index(phase)];
}

bool line_is_section(const char* line_start, const char* line_end, mps_section_kind* kind)
{
  if (line_start >= line_end) { return false; }

  mps_section_kind candidate;
  switch (*line_start) {
    case 'R':
      if (line_end - line_start >= 3 && std::memcmp(line_start, "RHS", 3) == 0) {
        candidate = mps_section_kind::rhs;
      } else if (line_end - line_start >= 4 && std::memcmp(line_start, "ROWS", 4) == 0) {
        candidate = mps_section_kind::rows;
      } else if (line_end - line_start >= 6 && std::memcmp(line_start, "RANGES", 6) == 0) {
        candidate = mps_section_kind::ranges;
      } else {
        return false;
      }
      break;
    case 'C':
      if (line_end - line_start >= 7 && std::memcmp(line_start, "COLUMNS", 7) == 0) {
        candidate = mps_section_kind::columns;
      } else {
        return false;
      }
      break;
    case 'B':
      if (line_end - line_start >= 6 && std::memcmp(line_start, "BOUNDS", 6) == 0) {
        candidate = mps_section_kind::bounds;
      } else {
        return false;
      }
      break;
    case 'E':
      if (line_end - line_start >= 6 && std::memcmp(line_start, "ENDATA", 6) == 0) {
        candidate = mps_section_kind::endata;
      } else {
        return false;
      }
      break;
    case 'Q':
      if (line_end - line_start >= 7 && std::memcmp(line_start, "QUADOBJ", 7) == 0) {
        candidate = mps_section_kind::quadobj;
      } else if (line_end - line_start >= 7 && std::memcmp(line_start, "QMATRIX", 7) == 0) {
        candidate = mps_section_kind::qmatrix;
      } else if (line_end - line_start >= 8 && std::memcmp(line_start, "QCMATRIX", 8) == 0) {
        candidate = mps_section_kind::qcmatrix;
      } else {
        return false;
      }
      break;
    default: return false;
  }

  const char* after = line_start + section_name_len(candidate);
  while (after < line_end && (*after == ' ' || *after == '\t' || *after == '\r')) {
    ++after;
  }
  if (after != line_end) { return false; }
  *kind = candidate;
  return true;
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
  switch (kind) {
    case mps_section_kind::rows: return 0;
    case mps_section_kind::columns: return 1;
    case mps_section_kind::rhs: return 2;
    case mps_section_kind::bounds: return 3;
    case mps_section_kind::ranges: return 4;
    case mps_section_kind::quadobj: return 5;
    case mps_section_kind::qmatrix: return 6;
    case mps_section_kind::qcmatrix: return 7;
    case mps_section_kind::endata: return 8;
  }
  return 0;
}

void mps_section_block_scanner_t::record_section_hit(mps_section_kind kind, const char* ptr)
{
  std::atomic<const char*>& slot = section_hits_[section_hit_index(kind)];
  const char* expected           = nullptr;
  if (slot.compare_exchange_strong(
        expected, ptr, std::memory_order_release, std::memory_order_acquire)) {
    publish_section_ranges();
  }
}

void mps_section_block_scanner_t::scan_section_range(const char* begin,
                                                     const char* end,
                                                     bool boundary_scan)
{
  (void)boundary_scan;
  if (begin >= end) return;
  const char* p = begin;

  // Interior scans that start inside a decoded block skip the leading partial
  // line. A separate boundary scan covers section titles whose newline/title
  // bytes straddle adjacent LZ4 blocks.
  if (p != data_) {
    const void* nl = __builtin_memchr(p, '\n', static_cast<std::size_t>(end - p));
    if (nl == nullptr) { return; }
    p = static_cast<const char*>(nl) + 1;
  }

  auto try_candidate = [&](const char* line_start) {
    const void* nl = __builtin_memchr(line_start, '\n', static_cast<std::size_t>(end - line_start));
    const char* line_end = nl == nullptr ? end : static_cast<const char*>(nl);
    mps_section_kind kind;
    if (line_is_section(line_start, line_end, &kind)) { record_section_hit(kind, line_start); }
  };

  // Handle the very first line of a file (NAME indicator, usually)
  if (p == data_) {
    if (p < end && is_nonblank_column1(static_cast<unsigned char>(*p))) { try_candidate(p); }
    ++p;
  }

  // In compliant MPS, indicator records begin in column 1 while data records
  // begin in column 2+. Treat start-of-file or "\n[nonblank]" as the cheap
  // candidate signal, then run the exact section matcher only for candidates.
  const simde__m256i newline = simde_mm256_set1_epi8('\n');
  while (static_cast<std::size_t>(end - p) >= 32) {
    simde__m256i current  = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(p));
    simde__m256i previous = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(p - 1));
    std::uint32_t mask = static_cast<std::uint32_t>(simde_mm256_movemask_epi8(simde_mm256_and_si256(
      simde_mm256_cmpeq_epi8(previous, newline), nonblank_column1_mask(current))));
    while (mask != 0) {
      int bit = __builtin_ctz(mask);
      try_candidate(p + bit);
      mask &= mask - 1;
    }
    p += 32;
  }

  // scalar tail
  while (p < end) {
    if (*(p - 1) == '\n' && is_nonblank_column1(static_cast<unsigned char>(*p))) {
      try_candidate(p);
    }
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
  scan_section_range(data_ + begin, data_ + end, true);
}

void mps_section_block_scanner_t::observe_block(std::size_t block_index,
                                                const char* begin,
                                                const char* end)
{
  if (block_index >= block_count_) {
    throw std::runtime_error("MPS section scanner observed invalid LZ4 block index");
  }

  scan_section_range(begin, end, false);
  block_begin_offsets_[block_index].store(static_cast<std::size_t>(begin - data_),
                                          std::memory_order_relaxed);
  block_end_offsets_[block_index].store(static_cast<std::size_t>(end - data_),
                                        std::memory_order_relaxed);
  block_decoded_[block_index].store(1, std::memory_order_release);

  if (block_index > 0 && block_decoded_[block_index - 1].load(std::memory_order_acquire)) {
    scan_boundary(block_index - 1, block_index);
  }
  if (block_index + 1 < block_count_ &&
      block_decoded_[block_index + 1].load(std::memory_order_acquire)) {
    scan_boundary(block_index, block_index + 1);
  }
}

void mps_section_block_scanner_t::publish_ready(std::size_t ready_bytes)
{
  ready_bytes_.store(ready_bytes, std::memory_order_release);
  publish_section_ranges();
}

void mps_section_block_scanner_t::publish_section_ranges()
{
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

  if (!registry_.ready(mps_phase_kind::rhs)) {
    if (available(rhs)) {
      const char* rhs_end =
        earliest_available_after(rhs, {ranges, bounds, quadobj, qmatrix, qcmatrix, final_boundary});
      if (rhs_end != nullptr) { registry_.publish(mps_phase_kind::rhs, {rhs, rhs_end, true}); }
    } else {
      const char* after_columns = earliest_available_after(
        columns, {ranges, bounds, quadobj, qmatrix, qcmatrix, final_boundary});
      if (after_columns != nullptr) {
        registry_.publish(mps_phase_kind::rhs, {nullptr, nullptr, false});
      }
    }
  }

  if (!registry_.ready(mps_phase_kind::ranges)) {
    const char* ranges_end =
      earliest_available_after(ranges, {bounds, quadobj, qmatrix, qcmatrix, final_boundary});
    const char* after_rhs = earliest_available_after(
      rhs ? rhs : columns, {bounds, quadobj, qmatrix, qcmatrix, final_boundary});
    if (available(ranges) && ranges_end != nullptr) {
      registry_.publish(mps_phase_kind::ranges, {ranges, ranges_end, true});
    } else if (!ranges && after_rhs != nullptr) {
      registry_.publish(mps_phase_kind::ranges, {nullptr, nullptr, false});
    }
  }

  if (!registry_.ready(mps_phase_kind::bounds)) {
    const char* bounds_end =
      earliest_available_after(bounds, {quadobj, qmatrix, qcmatrix, final_boundary});
    const char* after_ranges = earliest_available_after(
      ranges ? ranges : (rhs ? rhs : columns), {quadobj, qmatrix, qcmatrix, final_boundary});
    if (available(bounds) && bounds_end != nullptr) {
      registry_.publish(mps_phase_kind::bounds, {bounds, bounds_end, true});
    } else if (!bounds && after_ranges != nullptr) {
      registry_.publish(mps_phase_kind::bounds, {nullptr, nullptr, false});
    }
  }

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
}

}  // namespace mps_fast
