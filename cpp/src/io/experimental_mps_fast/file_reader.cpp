// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: Apache-2.0

#include "file_reader.hpp"
#include "nvtx_ranges.hpp"

#include <utilities/error.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
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

namespace mps_fast {

using cuopt::linear_programming::io::error_type_t;
using cuopt::linear_programming::io::mps_parser_fail;

namespace {

constexpr std::size_t raw_input_window_bytes              = 64ull * 1024ull * 1024ull;
constexpr std::size_t raw_input_max_read_threads          = 8;
constexpr std::size_t raw_input_direct_io_threshold_bytes = 1ull * 1024ull * 1024ull * 1024ull;

bool path_has_suffix(const std::string& path, const char* suffix) noexcept
{
  std::size_t suffix_len = std::strlen(suffix);
  return path.size() >= suffix_len &&
         path.compare(path.size() - suffix_len, suffix_len, suffix) == 0;
}

std::size_t round_up_to_multiple(std::size_t value, std::size_t alignment)
{
  if (alignment == 0) { return value; }
  std::size_t remainder = value % alignment;
  if (remainder == 0) { return value; }
  std::size_t increment = alignment - remainder;
  if (value > std::numeric_limits<std::size_t>::max() - increment) {
    mps_parser_fail(error_type_t::OutOfMemoryError, "allocation size overflow");
  }
  return value + increment;
}

std::size_t add_input_padding(std::size_t size)
{
  if (size > std::numeric_limits<std::size_t>::max() - input_buffer_padding_bytes) {
    mps_parser_fail(error_type_t::OutOfMemoryError, "input padding size overflow");
  }
  return size + input_buffer_padding_bytes;
}

}  // namespace

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

raw_input_stream_t::raw_input_stream_t(const std::string& path) : path_(path)
{
  MPS_NVTX_RANGE("raw_input_construct", nvtx::colors::io);
  buffered_fd_ = ::open(path.c_str(), O_RDONLY);
  if (buffered_fd_ < 0) {
    mps_parser_fail(error_type_t::RuntimeError,
                    "Failed to open raw MPS file '%s': %s",
                    path.c_str(),
                    std::strerror(errno));
  }

  file_size_         = get_file_size(buffered_fd_, path);
  fd_                = buffered_fd_;
  bool use_direct_io = file_size_ > raw_input_direct_io_threshold_bytes;
  if (const char* raw_direct = std::getenv("MPS_FAST_RAW_DIRECT_IO")) {
    use_direct_io = raw_direct[0] != '0';
  }
  if (use_direct_io) {
#ifdef O_DIRECT
    int direct_fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (direct_fd >= 0) {
      fd_        = direct_fd;
      direct_io_ = true;
    }
#endif
  }
  window_bytes_ = raw_input_window_bytes;
  window_count_ = std::max<std::size_t>(1, (file_size_ + window_bytes_ - 1) / window_bytes_);

  output_mapped_size_ = round_up_to_multiple(
    std::max<std::size_t>(add_input_padding(file_size_), 1), system_page_size());
  output_region_ = mmap_region_t::anonymous(
    output_mapped_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE, "raw input buffer");
  output_data_ = output_region_.char_data();
  output_region_.advise(MADV_HUGEPAGE);

  block_done_.resize(window_count_, 0);
  block_end_.resize(window_count_, 0);
  section_scanner_ =
    std::make_unique<mps_section_block_scanner_t>(output_data_, window_count_, registry_);
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
mps_phase_registry_t& raw_input_stream_t::registry() noexcept { return registry_; }
input_stream_view_t raw_input_stream_t::view() noexcept
{
  return {output_data_, output_data_, output_view_size_, file_size_, &registry_};
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

  std::atomic_size_t next_window{0};
  std::exception_ptr first_error = nullptr;
  std::mutex error_mutex;
  std::atomic_bool stop{false};

  auto mark_error = [&](std::exception_ptr eptr) {
    std::lock_guard<std::mutex> lock(error_mutex);
    if (!first_error) {
      first_error = eptr;
      stop.store(true, std::memory_order_release);
    }
  };

  auto read_window = [&](std::size_t index) {
    MPS_NVTX_RANGE("raw_window_read", nvtx::colors::io);
    std::size_t offset = index * window_bytes_;
    std::size_t size   = std::min(window_bytes_, file_size_ - offset);
    std::size_t done   = 0;
    {
      MPS_NVTX_RANGE("raw_window_pread", nvtx::colors::io);
      while (done < size) {
        ssize_t got =
          ::pread(fd_, output_data_ + offset + done, size - done, (off_t)(offset + done));
        if (got < 0) {
          if (errno == EINTR) { continue; }
          if (direct_io_ && errno == EINVAL && buffered_fd_ >= 0) {
            got = ::pread(
              buffered_fd_, output_data_ + offset + done, size - done, (off_t)(offset + done));
            if (got >= 0) {
              done += (std::size_t)got;
              continue;
            }
            if (errno == EINTR) { continue; }
          }
          mps_parser_fail(error_type_t::RuntimeError,
                          "Failed to pread raw MPS file '%s': %s",
                          path_.c_str(),
                          std::strerror(errno));
        }
        if (got == 0) {
          mps_parser_fail(error_type_t::RuntimeError,
                          "Unexpected EOF while reading raw MPS file '%s'",
                          path_.c_str());
        }
        done += (std::size_t)got;
      }
    }

    {
      MPS_NVTX_RANGE("raw_window_scan_publish", nvtx::colors::io);
      section_scanner_->observe_block(index, output_data_ + offset, output_data_ + offset + size);
      frontier_mutex_.lock();
      block_done_[index] = 1;
      block_end_[index]  = offset + size;
      std::size_t before = ready_bytes_;
      while (next_block_ < block_done_.size() && block_done_[next_block_]) {
        ready_bytes_ = block_end_[next_block_];
        ++next_block_;
      }
      std::size_t after = ready_bytes_;
      frontier_mutex_.unlock();
      if (after > before) { section_scanner_->publish_ready(after); }
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(thread_count);
  for (std::size_t t = 0; t < thread_count; ++t) {
    workers.emplace_back([&, t] {
      std::string thread_name = "raw-input-read-" + std::to_string(t);
      nvtx::name_current_thread(thread_name.c_str());
      MPS_NVTX_RANGE("raw_worker_loop", nvtx::colors::io);
      while (!stop.load(std::memory_order_acquire)) {
        std::size_t index = next_window.fetch_add(1, std::memory_order_relaxed);
        if (index >= window_count_) { break; }
        try {
          read_window(index);
        } catch (...) {
          mark_error(std::current_exception());
          return;
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  if (first_error) { std::rethrow_exception(first_error); }

  output_view_size_ = ready_bytes_;
  section_scanner_->publish_ready(output_view_size_);
}

memory_input_stream_t::memory_input_stream_t(std::vector<char> buffer,
                                             std::size_t input_size,
                                             std::size_t compressed_size)
  : buffer_(std::move(buffer)), input_size_(input_size), compressed_size_(compressed_size)
{
  section_scanner_ = std::make_unique<mps_section_block_scanner_t>(buffer_.data(), 1, registry_);
}

const char* memory_input_stream_t::data() const noexcept { return buffer_.data(); }
char* memory_input_stream_t::mutable_data() noexcept { return buffer_.data(); }
std::size_t memory_input_stream_t::size() const noexcept { return input_size_; }
std::size_t memory_input_stream_t::compressed_size() const noexcept { return compressed_size_; }
std::size_t memory_input_stream_t::reserve_size_hint() const noexcept { return input_size_; }
mps_phase_registry_t& memory_input_stream_t::registry() noexcept { return registry_; }
input_stream_view_t memory_input_stream_t::view() noexcept
{
  return {buffer_.data(), buffer_.data(), input_size_, compressed_size_, &registry_};
}

void memory_input_stream_t::run_decode_tasks()
{
  MPS_NVTX_RANGE("memory_input_scan", nvtx::colors::io);
  section_scanner_->observe_block(0, buffer_.data(), buffer_.data() + input_size_);
  section_scanner_->publish_ready(input_size_);
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

}  // namespace mps_fast
