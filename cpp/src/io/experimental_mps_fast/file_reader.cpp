// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: Apache-2.0

#include "file_reader.hpp"
#include "nvtx_ranges.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mps_fast {

char* string_buffer;
char* string_buffer_ptr;

namespace {

constexpr std::size_t raw_input_window_bytes     = 64ull * 1024ull * 1024ull;
constexpr std::size_t raw_input_max_read_threads = 8;

bool path_has_suffix(const std::string& path, const char* suffix) noexcept
{
  std::size_t suffix_len = std::strlen(suffix);
  return path.size() >= suffix_len &&
         path.compare(path.size() - suffix_len, suffix_len, suffix) == 0;
}

}  // namespace

namespace {

class FileDescriptor {
 public:
  explicit FileDescriptor(int fd) : fd_(fd) {}
  ~FileDescriptor()
  {
    if (fd_ >= 0) { ::close(fd_); }
  }

  FileDescriptor(const FileDescriptor&)            = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

  int get() const noexcept { return fd_; }
  bool valid() const noexcept { return fd_ >= 0; }

 private:
  int fd_;
};

std::size_t get_file_size(int fd, const std::string& path)
{
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    throw std::runtime_error("Failed to stat file '" + path + "': " + std::strerror(errno));
  }
  return static_cast<std::size_t>(st.st_size);
}

std::size_t system_page_size()
{
  static std::size_t page_size = [] {
    long value = ::sysconf(_SC_PAGESIZE);
    return value > 0 ? static_cast<std::size_t>(value) : static_cast<std::size_t>(4096);
  }();
  return page_size;
}

std::size_t round_up_to_multiple(std::size_t value, std::size_t alignment)
{
  if (alignment == 0) { return value; }
  std::size_t remainder = value % alignment;
  if (remainder == 0) { return value; }
  std::size_t increment = alignment - remainder;
  if (value > std::numeric_limits<std::size_t>::max() - increment) {
    throw std::runtime_error("allocation size overflow");
  }
  return value + increment;
}

}  // namespace

RawInputStream::RawInputStream(const std::string& path) : path_(path)
{
  MPS_NVTX_RANGE("raw_input_construct", nvtx::colors::io);
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) {
    throw std::runtime_error("Failed to open raw MPS file '" + path + "': " + std::strerror(errno));
  }

  file_size_    = get_file_size(fd_, path);
  window_bytes_ = raw_input_window_bytes;
  window_count_ = std::max<std::size_t>(1, (file_size_ + window_bytes_ - 1) / window_bytes_);

  output_mapped_size_ =
    round_up_to_multiple(std::max<std::size_t>(file_size_, 1), system_page_size());
  output_region_ = mmap_region_t::anonymous(
    output_mapped_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE, "raw input buffer");
  output_data_ = output_region_.char_data();
  output_region_.advise(MADV_HUGEPAGE);

  block_done_.resize(window_count_, 0);
  block_end_.resize(window_count_, 0);
  section_scanner_ =
    std::make_unique<mps_section_block_scanner_t>(output_data_, window_count_, registry_);
}

RawInputStream::~RawInputStream()
{
  if (fd_ >= 0) { ::close(fd_); }
}

const char* RawInputStream::data() const noexcept { return output_data_; }
char* RawInputStream::mutable_data() noexcept { return output_data_; }
std::size_t RawInputStream::size() const noexcept { return output_view_size_; }
std::size_t RawInputStream::compressed_size() const noexcept { return file_size_; }
std::size_t RawInputStream::reserve_size_hint() const noexcept { return file_size_; }
mps_phase_registry_t& RawInputStream::registry() noexcept { return registry_; }
input_stream_view_t RawInputStream::view() noexcept
{
  return {output_data_, output_data_, output_view_size_, file_size_, &registry_};
}

void RawInputStream::run_decode_tasks()
{
  MPS_NVTX_RANGE("raw_input_run_read_tasks", nvtx::colors::io);
  if (file_size_ == 0) {
    output_view_size_ = 0;
    section_scanner_->publish_ready(0);
    return;
  }

  std::size_t hw_threads =
    std::max<std::size_t>(1, static_cast<std::size_t>(std::thread::hardware_concurrency()));
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
    std::size_t offset = index * window_bytes_;
    std::size_t size   = std::min(window_bytes_, file_size_ - offset);
    std::size_t done   = 0;
    while (done < size) {
      ssize_t got =
        ::pread(fd_, output_data_ + offset + done, size - done, static_cast<off_t>(offset + done));
      if (got < 0) {
        if (errno == EINTR) { continue; }
        throw std::runtime_error("Failed to pread raw MPS file '" + path_ +
                                 "': " + std::strerror(errno));
      }
      if (got == 0) {
        throw std::runtime_error("Unexpected EOF while reading raw MPS file '" + path_ + "'");
      }
      done += static_cast<std::size_t>(got);
    }

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
  };

  std::vector<std::thread> workers;
  workers.reserve(thread_count);
  for (std::size_t t = 0; t < thread_count; ++t) {
    workers.emplace_back([&, t] {
      std::string thread_name = "raw-input-read-" + std::to_string(t);
      nvtx::name_current_thread(thread_name.c_str());
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

bool has_lz4_extension(const std::string& path) noexcept { return path_has_suffix(path, ".lz4"); }

void drop_file_cache(const std::string& path)
{
  MPS_NVTX_RANGE("drop_file_cache", nvtx::colors::io);
  FileDescriptor fd(::open(path.c_str(), O_RDONLY));
  if (!fd.valid()) { return; }

  ::posix_fadvise(fd.get(), 0, 0, POSIX_FADV_DONTNEED);
}

FileReadMethod effective_file_read_method(const std::string& path, FileReadMethod method)
{
  if (has_lz4_extension(path)) { return FileReadMethod::Lz4; }
  if (method == FileReadMethod::Lz4) {
    throw std::runtime_error("lz4 read method requires a .lz4 input: " + path);
  }
  return method;
}

const char* file_read_method_name(FileReadMethod method) noexcept
{
  switch (method) {
    case FileReadMethod::Read: return "read";
    case FileReadMethod::Lz4: return "lz4";
    default: return "unknown";
  }
}

}  // namespace mps_fast
