// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

#include <omp.h>

namespace mps_fast {

enum class mps_section_kind {
  rows,
  columns,
  rhs,
  bounds,
  ranges,
  quadobj,
  qmatrix,
  qcmatrix,
  endata,
};

enum class mps_phase_kind {
  header,
  rows,
  columns,
  rhs,
  bounds,
  ranges,
  quadratic,
};

struct mps_phase_range_t {
  const char* begin = nullptr;
  const char* end   = nullptr;
  bool present      = false;
};

class mps_phase_registry_t {
 public:
  void publish(mps_phase_kind phase, mps_phase_range_t range);
  void attach_event(mps_phase_kind phase, omp_event_handle_t event);

  bool ready(mps_phase_kind phase) const;
  // range() is lock-free: callers must observe ready(phase)==true first. The
  // acquire load in ready() pairs with publish()'s release store before ranges_.
  mps_phase_range_t range(mps_phase_kind phase) const;

 private:
  static constexpr std::size_t phase_count = 7;

  static std::size_t phase_index(mps_phase_kind phase);

  mps_phase_range_t ranges_[phase_count]{};
  std::atomic<bool> ready_[phase_count]{};
  omp_event_handle_t events_[phase_count]{};
  bool has_event_[phase_count]{};
  bool event_fulfilled_[phase_count]{};
  mutable std::mutex mutex_;
};

class mps_section_block_scanner_t {
 public:
  mps_section_block_scanner_t(const char* data,
                              std::size_t block_count,
                              mps_phase_registry_t& registry);

  void observe_block(std::size_t block_index, const char* begin, const char* end);
  void publish_ready(std::size_t ready_bytes);

 private:
  static constexpr std::size_t section_count = 9;
  // Section titles are short; 128 bytes is enough to rescan around a decoded
  // block boundary and catch a newline/title pair split across adjacent blocks.
  static constexpr std::size_t boundary_overlap = 128;

  static std::size_t section_hit_index(mps_section_kind kind);

  void scan_section_range(const char* begin, const char* end);
  void scan_boundary(std::size_t left_index, std::size_t right_index);
  void record_section_hit(mps_section_kind kind, const char* ptr);
  void publish_section_ranges();

  const char* data_        = nullptr;
  std::size_t block_count_ = 0;
  mps_phase_registry_t& registry_;
  std::mutex publish_mutex_;
  std::unique_ptr<std::atomic<unsigned char>[]> block_decoded_;
  std::unique_ptr<std::atomic_size_t[]> block_begin_offsets_;
  std::unique_ptr<std::atomic_size_t[]> block_end_offsets_;
  std::atomic_size_t ready_bytes_{0};
  std::atomic<const char*> section_hits_[section_count]{};
};

}  // namespace mps_fast
