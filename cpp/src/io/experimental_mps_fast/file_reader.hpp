// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mmap_region.hpp"
#include "mps_section_scanner.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mps_fast {

inline constexpr std::size_t input_buffer_padding_bytes = 64;

struct lz4_pipeline_t;

/**
 * @brief File reading method selection
 */
enum class FileReadMethod { Read, Lz4 };

/**
 * @brief Return the effective method for a path.
 *
 * .lz4 inputs are decompressed; all other inputs use raw input reads.
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

/**
 * @brief Ask the OS to evict clean cached pages for this file.
 *
 * This is advisory and affects the local client page cache only.
 */
void drop_file_cache(const std::string& path);

struct input_stream_view_t {
  const char* data               = nullptr;
  char* mutable_data             = nullptr;
  std::size_t size               = 0;
  std::size_t compressed_size    = 0;
  mps_phase_registry_t* registry = nullptr;
};

class Lz4InputStream {
 public:
  explicit Lz4InputStream(const std::string& path);
  ~Lz4InputStream();

  Lz4InputStream(const Lz4InputStream&)            = delete;
  Lz4InputStream& operator=(const Lz4InputStream&) = delete;

  const char* data() const noexcept;
  char* mutable_data() noexcept;
  std::size_t size() const noexcept;
  std::size_t compressed_size() const noexcept;
  std::size_t reserve_size_hint() const noexcept;
  mps_phase_registry_t& registry() noexcept;
  input_stream_view_t view() noexcept;

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
  mps_phase_registry_t registry_;
  std::mutex commit_mutex_;
  std::mutex frontier_mutex_;
  std::vector<unsigned char> block_done_;
  std::vector<std::size_t> block_end_;
  std::unique_ptr<mps_section_block_scanner_t> section_scanner_;
  std::size_t next_block_  = 0;
  std::size_t ready_bytes_ = 0;
};

class RawInputStream {
 public:
  explicit RawInputStream(const std::string& path);
  ~RawInputStream();

  RawInputStream(const RawInputStream&)            = delete;
  RawInputStream& operator=(const RawInputStream&) = delete;

  const char* data() const noexcept;
  char* mutable_data() noexcept;
  std::size_t size() const noexcept;
  std::size_t compressed_size() const noexcept;
  std::size_t reserve_size_hint() const noexcept;
  mps_phase_registry_t& registry() noexcept;
  input_stream_view_t view() noexcept;

  void run_decode_tasks();

 private:
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
  mps_phase_registry_t registry_;
  std::mutex frontier_mutex_;
  std::vector<unsigned char> block_done_;
  std::vector<std::size_t> block_end_;
  std::unique_ptr<mps_section_block_scanner_t> section_scanner_;
  std::size_t next_block_  = 0;
  std::size_t ready_bytes_ = 0;
};

}  // namespace mps_fast
