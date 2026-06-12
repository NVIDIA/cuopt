// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: Apache-2.0

// Input layer for the fast MPS parser: turns on-disk bytes (plain or .lz4) into one
// contiguous parse buffer and publishes MPS section boundaries as data becomes available.
//
// Model:
//   - Output is an anonymous mmap'd buffer (THP-hinted, tail-padded for SIMD/cursor safety).
//     Raw inputs pread directly into fixed slots; LZ4 decodes into the same layout.
//   - Work is split into windows (fixed spans of compressed/raw file bytes). Workers use
//     parallel_for_indexed() — std::thread + shared-index dispatch, not OpenMP — because
//     blocking pread()/decode does not compose cleanly with OMP team barriers.
//   - Each completed window/block is handed to mps_section_block_scanner_t::observe_block().
//     Blocks may finish out of order; the scanner advances a contiguous ready_bytes_
//     frontier and publishes section ranges into mps_phase_registry_t only once the prefix
//     up to a section title is contiguous and scannable.
//   - The parser runs as OpenMP tasks on those published phases while run_decode_tasks()
//     (raw parallel pread, or the LZ4 reader → metadata scanner → decoder pipeline) fills
//     the buffer on separate threads. parallel_error_latch_t propagates the first worker
//     failure and stops the rest.
//
// LZ4 adds a resident-window pool (parallel pread of compressed spans), block metadata
// scanning with ptr_if_contiguous()/copy_to for window-boundary payloads, parallel decode
// workers, window ref-counting/release, and lazy commit_up_to() of decoded output pages.

#pragma once

#include "mmap_region.hpp"
#include "mps_section_scanner.hpp"
#include "nvtx_ranges.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cuopt::linear_programming::io::detail {

inline constexpr std::size_t input_buffer_padding_bytes = 64;

void ensure_input_buffer_padding(std::vector<char>& buffer, std::size_t input_size);

struct lz4_pipeline_t;

/**
 * @brief File reading method selection
 */
enum class FileReadMethod { Read, Lz4, Gzip, Bzip2 };

/**
 * @brief Return the effective method for a path.
 *
 * Compressed inputs are auto-detected by extension; all other inputs use raw input reads.
 */
FileReadMethod effective_file_read_method(const std::string& path, FileReadMethod method);

/**
 * @brief Human-readable method name.
 */
const char* file_read_method_name(FileReadMethod method) noexcept;

/**
 * @brief True when the file name has an lz4 extension.
 */
bool has_lz4_extension(const std::string& path) noexcept;
bool has_gzip_extension(const std::string& path) noexcept;
bool has_bzip2_extension(const std::string& path) noexcept;

/**
 * @brief Ask the OS to evict clean cached pages for this file.
 *
 * This is advisory and affects the local client page cache only.
 */
void drop_file_cache(const std::string& path);

/**
 * @brief OS memory page size, queried once and cached.
 */
std::size_t system_page_size();

/**
 * @brief File size in bytes; fails with a parser error if it cannot be determined.
 */
std::size_t get_file_size(int fd, const std::string& path);
std::size_t get_file_size(const std::string& path);

/**
 * @brief Read exactly @p bytes at @p offset into @p dst, retrying on EINTR.
 *
 * Returns false and leaves errno set on error or unexpected EOF.
 */
bool pread_full(int fd, char* dst, std::size_t bytes, std::size_t offset);

// First-error-wins latch shared by the parallel reader/decoder pipelines. The
// first captured exception is retained and a stop flag is raised so cooperating
// workers can unwind promptly. The retained exception is rethrown by the
// orchestrating thread once all workers have joined.
class parallel_error_latch_t {
 public:
  void capture(std::exception_ptr eptr)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!first_error_) {
      first_error_ = eptr;
      stopped_.store(true, std::memory_order_release);
    }
  }

  bool stopped() const noexcept { return stopped_.load(std::memory_order_acquire); }

  void rethrow_if_error() const
  {
    if (first_error_) { std::rethrow_exception(first_error_); }
  }

 private:
  std::mutex mutex_;
  std::exception_ptr first_error_ = nullptr;
  std::atomic_bool stopped_{false};
};

class scoped_thread_group {
 public:
  void reserve(std::size_t count) { threads_.reserve(count); }

  template <typename F>
  void emplace(F&& f)
  {
    threads_.emplace_back(std::forward<F>(f));
  }

  ~scoped_thread_group()
  {
    for (auto& thread : threads_) {
      if (thread.joinable()) { thread.join(); }
    }
  }

 private:
  std::vector<std::thread> threads_;
};

// Work-stealing parallel loop over [0, count). Each of thread_count workers pulls
// the next index from a shared counter and invokes body(index). An exception
// escaping body is captured into the latch and stops the loop; the caller is
// responsible for calling latch.rethrow_if_error() after this returns. Workers
// are named "<thread_name_prefix><worker-id>" when a prefix is supplied.
// OMP just doesn't really play well with blocking pread()
template <typename Body>
void parallel_for_indexed(std::size_t count,
                          std::size_t thread_count,
                          parallel_error_latch_t& latch,
                          const char* thread_name_prefix,
                          Body body)
{
  std::atomic_size_t next{0};
  scoped_thread_group workers;
  workers.reserve(thread_count);
  for (std::size_t t = 0; t < thread_count; ++t) {
    workers.emplace([&, t] {
      if (thread_name_prefix != nullptr) {
        std::string name = thread_name_prefix + std::to_string(t);
        nvtx::name_current_thread(name.c_str());
      }
      while (!latch.stopped()) {
        std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= count) { break; }
        try {
          body(index);
        } catch (...) {
          latch.capture(std::current_exception());
          return;
        }
      }
    });
  }
}

struct input_stream_view_t {
  const char* data               = nullptr;
  char* mutable_data             = nullptr;
  std::size_t size               = 0;
  std::size_t compressed_size    = 0;
  mps_phase_registry_t* registry = nullptr;
};

/**
 * @brief CRTP base supplying the registry and view() shared by every input
 * stream. Derived classes provide data()/mutable_data()/size()/compressed_size().
 */
template <typename Derived>
class input_stream_base_t {
 public:
  mps_phase_registry_t& registry() noexcept { return registry_; }

  input_stream_view_t view() noexcept
  {
    auto* self = static_cast<Derived*>(this);
    return {self->data(), self->mutable_data(), self->size(), self->compressed_size(), &registry_};
  }

 protected:
  mps_phase_registry_t registry_;
};

// Handles lz4 compressed files (useful since lz4 is very fast, works well for MPS, and makes
// parallel decompression trivial)
class lz4_input_stream_t : public input_stream_base_t<lz4_input_stream_t> {
 public:
  explicit lz4_input_stream_t(const std::string& path);
  ~lz4_input_stream_t();

  lz4_input_stream_t(const lz4_input_stream_t&)            = delete;
  lz4_input_stream_t& operator=(const lz4_input_stream_t&) = delete;

  const char* data() const noexcept;
  char* mutable_data() noexcept;
  std::size_t size() const noexcept;
  std::size_t compressed_size() const noexcept;
  std::size_t reserve_size_hint() const noexcept;

  void run_decode_tasks();

 private:
  friend struct lz4_pipeline_t;

  void commit_up_to(std::size_t bytes);

  std::string path_;
  int fd_ = -1;
  mmap_region_t output_region_;
  std::size_t compressed_size_       = 0;
  char* output_data_                 = nullptr;
  std::size_t output_mapped_size_    = 0;
  std::size_t output_view_size_      = 0;
  std::size_t output_committed_size_ = 0;
  std::size_t block_max_size_        = 0;
  std::size_t content_size_          = 0;
  std::size_t header_size_           = 0;
  bool content_size_present_         = false;
  bool block_checksum_               = false;
  bool content_checksum_             = false;
  bool dict_id_                      = false;
  std::mutex commit_mutex_;
  std::unique_ptr<mps_section_block_scanner_t> section_scanner_;
  std::size_t block_slot_count_ = 0;
};

// Takes a file path
class raw_input_stream_t : public input_stream_base_t<raw_input_stream_t> {
 public:
  explicit raw_input_stream_t(const std::string& path);
  ~raw_input_stream_t();

  raw_input_stream_t(const raw_input_stream_t&)            = delete;
  raw_input_stream_t& operator=(const raw_input_stream_t&) = delete;

  const char* data() const noexcept;
  char* mutable_data() noexcept;
  std::size_t size() const noexcept;
  std::size_t compressed_size() const noexcept;
  std::size_t reserve_size_hint() const noexcept;

  void run_decode_tasks();

 private:
  void read_window_payload(std::size_t offset, std::size_t size);

  std::string path_;
  int fd_          = -1;
  int buffered_fd_ = -1;
  bool direct_io_  = false;
  mmap_region_t output_region_;
  char* output_data_              = nullptr;
  std::size_t output_mapped_size_ = 0;
  std::size_t output_view_size_   = 0;
  std::size_t file_size_          = 0;
  std::size_t window_bytes_       = 0;
  std::size_t window_count_       = 0;
  std::unique_ptr<mps_section_block_scanner_t> section_scanner_;
};

// Takes an in-memory buffer
class memory_input_stream_t : public input_stream_base_t<memory_input_stream_t> {
 public:
  memory_input_stream_t(std::vector<char> buffer,
                        std::size_t input_size,
                        std::size_t compressed_size);

  memory_input_stream_t(const memory_input_stream_t&)            = delete;
  memory_input_stream_t& operator=(const memory_input_stream_t&) = delete;

  const char* data() const noexcept;
  char* mutable_data() noexcept;
  std::size_t size() const noexcept;
  std::size_t compressed_size() const noexcept;
  std::size_t reserve_size_hint() const noexcept;

  void run_decode_tasks();

 private:
  std::vector<char> buffer_;
  std::size_t input_size_      = 0;
  std::size_t compressed_size_ = 0;
  std::unique_ptr<mps_section_block_scanner_t> section_scanner_;
};

}  // namespace cuopt::linear_programming::io::detail
