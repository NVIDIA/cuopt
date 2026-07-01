// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: Apache-2.0

#include "file_reader.hpp"
#include "nvtx_ranges.hpp"

#include <utilities/error.hpp>
#include <utilities/scope_guard.hpp>

#include <cuda/cmath>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::io::detail {

using cuopt::mathematical_optimization::io::error_type_t;
using cuopt::mathematical_optimization::io::mps_parser_fail;

namespace {

constexpr std::size_t raw_input_window_bytes              = 64ull * 1024ull * 1024ull;
constexpr std::size_t raw_input_max_read_threads          = 8;
constexpr std::size_t raw_input_direct_io_threshold_bytes = 1ull * 1024ull * 1024ull * 1024ull;
constexpr long nfs_super_magic                            = 0x6969;

bool path_has_suffix(const std::string& path, const char* suffix) noexcept
{
  std::size_t suffix_len = std::strlen(suffix);
  if (path.size() < suffix_len) { return false; }
  for (std::size_t i = 0; i < suffix_len; ++i) {
    unsigned char path_char = path[path.size() - suffix_len + i];
    if (std::tolower(path_char) != suffix[i]) { return false; }
  }
  return true;
}

std::size_t add_input_padding(std::size_t size)
{
  if (size > std::numeric_limits<std::size_t>::max() - input_buffer_padding_bytes) {
    mps_parser_fail(error_type_t::OutOfMemoryError, "input padding size overflow");
  }
  return size + input_buffer_padding_bytes;
}

bool is_nfs_backed_path(const std::string& path) noexcept
{
  struct statfs fs;
  return ::statfs(path.c_str(), &fs) == 0 && fs.f_type == nfs_super_magic;
}

}  // namespace

void ensure_input_buffer_padding(std::vector<char>& buffer, std::size_t input_size)
{
  if (input_size > buffer.size()) {
    mps_parser_fail(error_type_t::ValidationError,
                    "input_size %zu exceeds buffer size %zu",
                    input_size,
                    buffer.size());
  }
  std::size_t required = add_input_padding(input_size);
  if (buffer.size() < required) { buffer.resize(required, '\0'); }
}

std::size_t get_file_size(int fd, const std::string& path)
{
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    mps_parser_fail(error_type_t::RuntimeError,
                    "Failed to stat file '%s': %s",
                    path.c_str(),
                    std::strerror(errno));
  }
  if (st.st_size < 0) {
    mps_parser_fail(error_type_t::RuntimeError, "Negative file size for '%s'", path.c_str());
  }
  return (std::size_t)st.st_size;
}

std::size_t get_file_size(const std::string& path)
{
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    mps_parser_fail(error_type_t::RuntimeError,
                    "Failed to open file '%s': %s",
                    path.c_str(),
                    std::strerror(errno));
  }
  cuopt::scope_guard close_fd([&] {
    if (fd >= 0) { ::close(fd); }
  });

  std::size_t size = get_file_size(fd, path);
  ::close(fd);
  return size;
}

std::size_t system_page_size()
{
  static std::size_t page_size = [] {
    long value = ::sysconf(_SC_PAGESIZE);
    return value > 0 ? (std::size_t)value : (std::size_t)4096;
  }();
  return page_size;
}

bool pread_full(int fd, char* dst, std::size_t bytes, std::size_t offset)
{
  std::size_t done = 0;
  while (done < bytes) {
    std::size_t remaining = bytes - done;
    std::size_t chunk =
      std::min<std::size_t>(remaining, (std::size_t)std::numeric_limits<ssize_t>::max());
    ssize_t got = ::pread(fd, dst + done, chunk, (off_t)(offset + done));
    if (got < 0) {
      if (errno == EINTR) { continue; }
      return false;
    }
    if (got == 0) {
      errno = EIO;
      return false;
    }
    done += (std::size_t)got;
  }
  return true;
}

raw_input_stream_t::raw_input_stream_t(const std::string& path) : path_(path)
{
  MPS_NVTX_RANGE("raw_input_construct", nvtx::colors::io);
  int buffered_fd = ::open(path.c_str(), O_RDONLY);
  cuopt::scope_guard close_buffered([&] {
    if (buffered_fd >= 0) { ::close(buffered_fd); }
  });
  if (buffered_fd < 0) {
    mps_parser_fail(error_type_t::RuntimeError,
                    "Failed to open raw MPS file '%s': %s",
                    path.c_str(),
                    std::strerror(errno));
  }

  int direct_fd = -1;
  cuopt::scope_guard close_direct([&] {
    if (direct_fd >= 0) { ::close(direct_fd); }
  });

  file_size_                   = get_file_size(buffered_fd, path);
  int read_fd                  = buffered_fd;
  bool large_enough_for_direct = file_size_ > raw_input_direct_io_threshold_bytes;
  bool nfs_backed              = is_nfs_backed_path(path);
  // Buffered reads are consistently faster than O_DIRECT on our NFS mounts;
  // keep direct I/O for large local files where it wins.
  if (large_enough_for_direct && !nfs_backed) {
#ifdef O_DIRECT
    direct_fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (direct_fd >= 0) {
      read_fd    = direct_fd;
      direct_io_ = true;
    }
#endif
  }
  window_bytes_ = raw_input_window_bytes;
  window_count_ = std::max<std::size_t>(1, (file_size_ + window_bytes_ - 1) / window_bytes_);
#ifdef MPS_FAST_TIMERS
  read_window_ms_.assign(window_count_, 0);
#endif

  output_mapped_size_ =
    cuda::round_up(std::max<std::size_t>(add_input_padding(file_size_), 1), system_page_size());
  output_region_ = mmap_region_t::anonymous(
    output_mapped_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE, "raw input buffer");
  output_data_ = output_region_.char_data();
  output_region_.advise(MADV_HUGEPAGE);

  section_scanner_ =
    std::make_unique<mps_section_block_scanner_t>(output_data_, window_count_, registry_);

  buffered_fd_ = buffered_fd;
  buffered_fd  = -1;
  fd_          = read_fd;
  if (read_fd == direct_fd) { direct_fd = -1; }
}

raw_input_stream_t::~raw_input_stream_t()
{
  if (fd_ >= 0) { ::close(fd_); }
  if (buffered_fd_ >= 0 && buffered_fd_ != fd_) { ::close(buffered_fd_); }
}

const char* raw_input_stream_t::data() const noexcept { return output_data_; }
char* raw_input_stream_t::mutable_data() noexcept { return output_data_; }
std::size_t raw_input_stream_t::size() const noexcept { return output_view_size_; }
std::size_t raw_input_stream_t::compressed_size() const noexcept { return file_size_; }
std::size_t raw_input_stream_t::reserve_size_hint() const noexcept { return file_size_; }

void raw_input_stream_t::read_window_payload(std::size_t offset, std::size_t size)
{
  if (pread_full(fd_, output_data_ + offset, size, offset)) { return; }
  // O_DIRECT can reject an unaligned request with EINVAL; fall back to the
  // buffered descriptor for this window when that happens.
  if (direct_io_ && errno == EINVAL && buffered_fd_ >= 0 &&
      pread_full(buffered_fd_, output_data_ + offset, size, offset)) {
    return;
  }
  mps_parser_fail(error_type_t::RuntimeError,
                  "Failed to pread raw MPS file '%s': %s",
                  path_.c_str(),
                  std::strerror(errno));
}

void raw_input_stream_t::run_decode_tasks()
{
  MPS_NVTX_RANGE("raw_input_run_read_tasks", nvtx::colors::io);
  if (file_size_ == 0) {
    output_view_size_ = 0;
    section_scanner_->publish_ready(0);
    return;
  }

  std::size_t hw_threads =
    std::max<std::size_t>(1, (std::size_t)std::thread::hardware_concurrency());
  std::size_t thread_count = std::min(raw_input_max_read_threads, hw_threads);
  thread_count             = std::max<std::size_t>(1, std::min(thread_count, window_count_));

  // Each window is read independently and handed to the scanner, which owns the
  // contiguous decoded-byte frontier and the parallel section publication.
  parallel_error_latch_t latch;
#ifdef MPS_FAST_TIMERS
  auto read_wall_start = std::chrono::steady_clock::now();
#endif
  parallel_for_indexed(
    window_count_, thread_count, latch, "raw-input-read-", [&](std::size_t index) {
      MPS_NVTX_RANGE("raw_window_read", nvtx::colors::io);
      std::size_t offset = index * window_bytes_;
      std::size_t size   = std::min(window_bytes_, file_size_ - offset);
      {
        MPS_NVTX_RANGE("raw_window_pread", nvtx::colors::io);
#ifdef MPS_FAST_TIMERS
        auto start = std::chrono::steady_clock::now();
#endif
        read_window_payload(offset, size);
#ifdef MPS_FAST_TIMERS
        auto end     = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        read_window_ms_[index] =
          (uint32_t)std::min<long long>(elapsed.count(), std::numeric_limits<uint32_t>::max());
#endif
      }
      MPS_NVTX_RANGE("raw_window_scan_publish", nvtx::colors::io);
      section_scanner_->observe_block(index, output_data_ + offset, output_data_ + offset + size);
    });
#ifdef MPS_FAST_TIMERS
  auto read_wall_end = std::chrono::steady_clock::now();
#endif
  latch.rethrow_if_error();

#ifdef MPS_FAST_TIMERS
  if (!read_window_ms_.empty()) {
    std::vector<uint32_t> sorted = read_window_ms_;
    std::sort(sorted.begin(), sorted.end());
    auto percentile = [&](double pct) {
      std::size_t idx = (std::size_t)std::min<double>((double)(sorted.size() - 1),
                                                      pct * (double)(sorted.size() - 1));
      return sorted[idx];
    };
    uint64_t total_ms = 0;
    for (uint32_t value : read_window_ms_) {
      total_ms += value;
    }
    std::fprintf(
      stderr,
      "[RAW_READ_LATENCY] windows=%zu wall_ms=%lld total_window_ms=%llu avg_ms=%.3f min_ms=%u "
      "p50_ms=%u p90_ms=%u p99_ms=%u max_ms=%u\n",
      read_window_ms_.size(),
      (long long)std::chrono::duration_cast<std::chrono::milliseconds>(read_wall_end -
                                                                       read_wall_start)
        .count(),
      (unsigned long long)total_ms,
      (double)total_ms / (double)read_window_ms_.size(),
      sorted.front(),
      percentile(0.50),
      percentile(0.90),
      percentile(0.99),
      sorted.back());
  }
#endif

  output_view_size_ = section_scanner_->ready_bytes();
  section_scanner_->publish_ready(output_view_size_);
}

memory_input_stream_t::memory_input_stream_t(std::vector<char> buffer,
                                             std::size_t input_size,
                                             std::size_t compressed_size)
  : buffer_(std::move(buffer)), input_size_(input_size), compressed_size_(compressed_size)
{
  ensure_input_buffer_padding(buffer_, input_size_);
  section_scanner_ = std::make_unique<mps_section_block_scanner_t>(buffer_.data(), 1, registry_);
}

const char* memory_input_stream_t::data() const noexcept { return buffer_.data(); }
char* memory_input_stream_t::mutable_data() noexcept { return buffer_.data(); }
std::size_t memory_input_stream_t::size() const noexcept { return input_size_; }
std::size_t memory_input_stream_t::compressed_size() const noexcept { return compressed_size_; }
std::size_t memory_input_stream_t::reserve_size_hint() const noexcept { return input_size_; }

void memory_input_stream_t::run_decode_tasks()
{
  MPS_NVTX_RANGE("memory_input_scan", nvtx::colors::io);
  // Single block: observe_block advances the frontier and publishes.
  section_scanner_->observe_block(0, buffer_.data(), buffer_.data() + input_size_);
}

bool has_lz4_extension(const std::string& path) noexcept { return path_has_suffix(path, ".lz4"); }
bool has_gzip_extension(const std::string& path) noexcept { return path_has_suffix(path, ".gz"); }
bool has_bzip2_extension(const std::string& path) noexcept { return path_has_suffix(path, ".bz2"); }

void drop_file_cache(const std::string& path)
{
  MPS_NVTX_RANGE("drop_file_cache", nvtx::colors::io);
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) { return; }
  ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  ::close(fd);
}

FileReadMethod effective_file_read_method(const std::string& path, FileReadMethod method)
{
  if (has_lz4_extension(path)) { return FileReadMethod::Lz4; }
  if (has_gzip_extension(path)) { return FileReadMethod::Gzip; }
  if (has_bzip2_extension(path)) { return FileReadMethod::Bzip2; }
  if (method == FileReadMethod::Lz4) {
    mps_parser_fail(
      error_type_t::ValidationError, "lz4 read method requires a .lz4 input: %s", path.c_str());
  }
  return method;
}

const char* file_read_method_name(FileReadMethod method) noexcept
{
  switch (method) {
    case FileReadMethod::Read: return "read";
    case FileReadMethod::Lz4: return "lz4";
    case FileReadMethod::Gzip: return "gzip";
    case FileReadMethod::Bzip2: return "bzip2";
    default: return "unknown";
  }
}

}  // namespace cuopt::mathematical_optimization::io::detail
