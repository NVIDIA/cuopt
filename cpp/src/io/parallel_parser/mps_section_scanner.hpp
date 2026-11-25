// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// reserved. SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

#include <omp.h>

#include "source_location.hpp"

// The section scanner handles freshly read/decoded blocks and scans them for section titles while
// they're still warm in cache it then publishes read/decoded input ranges to the parser workers,
// which handle their respective sections in parallel.

namespace cuopt::mathematical_optimization::io::detail {

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
  name,
  objsense,
  objname,
  lazycons,
  COUNT,
};

enum class mps_phase_kind {
  header,
  rows,
  columns,
  rhs,
  bounds,
  ranges,
  quadratic,
  lazycons,
  COUNT,
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
  // range() acquire-loads ready_[phase] (pairs with publish()'s release store) before
  // reading ranges_[phase]. Callers must not invoke range() until the phase is published.
  mps_phase_range_t range(mps_phase_kind phase) const;

  void publish_endata(const char* begin, bool present);
  bool endata_ready() const;
  const char* endata_begin() const;
  bool endata_present() const;

 private:
  // mutex_ guards ranges_/events_/has_event_/event_fulfilled_ and the endata_* fields for writers.
  // Readers observe ready_[phase] / endata_ready_ (release-stored under the lock on publish,
  // acquire-loaded here) and may then read the matching range lock-free -- see range()'s contract.
  static constexpr std::size_t phase_count = (std::size_t)mps_phase_kind::COUNT;

  static std::size_t phase_index(mps_phase_kind phase);

  mps_phase_range_t ranges_[phase_count]{};
  std::atomic<bool> ready_[phase_count]{};
  omp_event_handle_t events_[phase_count]{};
  bool has_event_[phase_count]{};
  bool event_fulfilled_[phase_count]{};
  const char* endata_begin_ = nullptr;
  bool endata_present_      = false;
  std::atomic<bool> endata_ready_{false};
  mutable std::mutex mutex_;
};

// Turns out-of-order decoded blocks into ordered section-range publications for the parser:
//
//   producer --observe_block(i,...)--> [SIMD-scan block i for section titles] --> section_hits_
//                                       [advance contiguous decoded-byte frontier (ready_bytes_)]
//                                       --> notify_ready_phases --> registry --> parser tasks
//
// Producers (the LZ4 decoders / raw readers) call observe_block for each block in any order.
// Per block the scanner (1) SIMD-scans it for section titles starting in column 1 and records
// the first byte of each section via a first-writer-wins CAS; (2) advances a contiguous
// decoded-byte frontier across whatever leading blocks are now present; and (3) recomputes which
// phases are fully bounded and publishes their [begin,end) ranges to the registry, unblocking the
// matching parser task. A title can straddle two blocks, so each block boundary is rescanned as
// the frontier crosses it.
class mps_section_block_scanner_t {
 public:
  mps_section_block_scanner_t(const char* data,
                              std::size_t block_count,
                              mps_phase_registry_t& registry,
                              source_location_t& source);

  // Records a freshly decoded block, scans it for section titles, advances the
  // contiguous decoded-byte frontier across out-of-order completions, and
  // publishes any newly available section ranges. Producers only need to feed
  // blocks in any order; the frontier and publication live entirely here.
  void observe_block(std::size_t block_index, const char* begin, const char* end);
  // Marks the current frontier as end of input. Called once, after every observe_block returned.
  void publish_final();

  // Current contiguous decoded-byte frontier; producers use this as the final
  // view size once all blocks have been observed.
  std::size_t ready_bytes() const noexcept;

 private:
  static std::size_t section_hit_index(mps_section_kind kind);

  std::size_t scan_section_range(const char* begin, const char* end);
  void try_section_candidate(const char* line_start, const char* end, bool at_final_frontier);
  bool is_final_frontier(const char* p) const;
  void scan_boundary(std::size_t right_index);
  void record_section_hit(mps_section_kind kind, const char* ptr);
  void notify_ready_phases();
  void advance_ready_frontier();

  // Concurrency: observe_block runs concurrently on many producer threads.
  //   * frontier_mutex_ guards next_block_, boundary scans and the ready_bytes_ frontier advance.
  //   * publish_mutex_  serializes notify_ready_phases so each phase publishes once, in order.
  //   * block_decoded_[i] is release-stored after block_begin/end_offsets_[i] (relaxed), so an
  //     acquire-load of a set flag makes those offsets visible to the reader.
  //   * section_hits_[k] is a first-writer-wins CAS holding the earliest byte of section k.
  //   * registry_ carries its own internal lock.
  const char* data_        = nullptr;
  std::size_t block_count_ = 0;
  mps_phase_registry_t& registry_;
  source_location_t& source_;
  std::mutex publish_mutex_;
  std::unique_ptr<std::atomic<unsigned char>[]> block_decoded_;
  std::unique_ptr<std::atomic_size_t[]> block_begin_offsets_;
  std::unique_ptr<std::atomic_size_t[]> block_end_offsets_;
  std::unique_ptr<std::atomic_size_t[]> block_newlines_;
  std::atomic_size_t ready_bytes_{0};
  std::atomic<bool> final_{false};
  std::atomic<const char*> section_hits_[(std::size_t)mps_section_kind::COUNT]{};
  std::mutex frontier_mutex_;
  std::size_t next_block_ = 0;
};

}  // namespace cuopt::mathematical_optimization::io::detail
