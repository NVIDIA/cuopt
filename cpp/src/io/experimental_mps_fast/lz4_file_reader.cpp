// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: Apache-2.0

#include "file_reader.hpp"
#include "mps_section_scanner.hpp"
#include "nvtx_ranges.hpp"

#include <utilities/error.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mps_fast {

using cuopt::linear_programming::io::error_type_t;
using cuopt::linear_programming::io::mps_parser_expects;
using cuopt::linear_programming::io::mps_parser_fail;

namespace {

constexpr uint32_t lz4_frame_magic                      = 0x184D2204u;
constexpr uint32_t lz4_uncompressed_block               = 0x80000000u;
constexpr uint32_t lz4_block_size_mask                  = 0x7FFFFFFFu;
constexpr std::size_t lz4_pipeline_batch_bytes          = 64ull * 1024ull * 1024ull;
constexpr std::size_t lz4_input_max_io_threads          = 8;
constexpr std::size_t lz4_no_content_size_reserve_ratio = 16;

using LZ4_decompress_safe_t = int (*)(const char*, char*, int, int);

#if defined(MPS_PARSER_WITH_LZ4)
struct lz4_runtime_t {
  void* handle                          = nullptr;
  LZ4_decompress_safe_t decompress_safe = nullptr;

  lz4_runtime_t()
  {
    for (const char* soname : {"liblz4.so.1", "liblz4.so"}) {
      handle = ::dlopen(soname, RTLD_LAZY);
      if (handle != nullptr) { break; }
    }
    if (handle == nullptr) {
      mps_parser_fail(error_type_t::RuntimeError,
                      "Could not open .mps.lz4 file since liblz4 was not found "
                      "(tried liblz4.so.1, liblz4.so). Decompress the .lz4 file manually "
                      "or install liblz4.");
    }

    decompress_safe =
      reinterpret_cast<LZ4_decompress_safe_t>(::dlsym(handle, "LZ4_decompress_safe"));
    if (decompress_safe == nullptr) {
      mps_parser_fail(error_type_t::RuntimeError,
                      "Error loading LZ4_decompress_safe from liblz4. Decompress the .lz4 file "
                      "manually or install a compatible liblz4.");
    }
  }

  ~lz4_runtime_t()
  {
    if (handle != nullptr) { ::dlclose(handle); }
  }

  lz4_runtime_t(const lz4_runtime_t&)            = delete;
  lz4_runtime_t& operator=(const lz4_runtime_t&) = delete;
};

const lz4_runtime_t& lz4_runtime()
{
  static const lz4_runtime_t runtime;
  return runtime;
}
#endif

int lz4_decompress_safe_runtime(const char* src, char* dst, int compressed_size, int dst_capacity)
{
#if defined(MPS_PARSER_WITH_LZ4)
  return lz4_runtime().decompress_safe(src, dst, compressed_size, dst_capacity);
#else
  (void)src;
  (void)dst;
  (void)compressed_size;
  (void)dst_capacity;
  mps_parser_fail(
    error_type_t::RuntimeError,
    "Experimental fast MPS parser was built without LZ4 decompression support. "
    "Reconfigure with CUOPT_PARSER_WITH_LZ4=ON or decompress the .lz4 file manually.");
#endif
}

void ensure_lz4_runtime_available()
{
#if defined(MPS_PARSER_WITH_LZ4)
  (void)lz4_runtime();
#else
  mps_parser_fail(
    error_type_t::RuntimeError,
    "Experimental fast MPS parser was built without LZ4 decompression support. "
    "Reconfigure with CUOPT_PARSER_WITH_LZ4=ON or decompress the .lz4 file manually.");
#endif
}

int open_lz4_fd(const std::string& path)
{
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    mps_parser_fail(error_type_t::RuntimeError,
                    "Failed to open LZ4 file '%s': %s",
                    path.c_str(),
                    std::strerror(errno));
  }
  return fd;
}

std::size_t round_up_to_multiple(std::size_t value, std::size_t alignment);

uint32_t read_le32(const char* ptr)
{
  const auto* p = reinterpret_cast<const unsigned char*>(ptr);
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

uint64_t read_le64(const char* ptr)
{
  const auto* p  = reinterpret_cast<const unsigned char*>(ptr);
  uint64_t value = 0;
  for (int i = 7; i >= 0; --i) {
    value = (value << 8) | p[i];
  }
  return value;
}

std::size_t block_max_size_from_bd(unsigned char bd)
{
  unsigned block_size_id = (bd >> 4) & 0x7u;
  switch (block_size_id) {
    case 4: return 64ull * 1024ull;
    case 5: return 256ull * 1024ull;
    case 6: return 1024ull * 1024ull;
    case 7: return 4ull * 1024ull * 1024ull;
    default: mps_parser_fail(error_type_t::ValidationError, "unsupported LZ4 frame block size ID");
  }
}

std::size_t checked_size(uint64_t value, const char* label)
{
  if (value > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    mps_parser_fail(error_type_t::OutOfMemoryError, "LZ4 %s exceeds size_t", label);
  }
  return static_cast<std::size_t>(value);
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
    mps_parser_fail(
      error_type_t::RuntimeError, "Invalid negative file size for '%s'", path.c_str());
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
    mps_parser_fail(error_type_t::OutOfMemoryError, "allocation size overflow");
  }
  return value + increment;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label)
{
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    mps_parser_fail(error_type_t::OutOfMemoryError, "%s size overflow", label);
  }
  return a * b;
}

bool pread_full_plain(int fd, char* dst, std::size_t bytes, std::size_t offset)
{
  std::size_t done = 0;
  while (done < bytes) {
    std::size_t remaining = bytes - done;
    std::size_t chunk     = std::min<std::size_t>(
      remaining, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
    ssize_t got = ::pread(fd, dst + done, chunk, static_cast<off_t>(offset + done));
    if (got < 0) {
      if (errno == EINTR) { continue; }
      return false;
    }
    if (got == 0) {
      errno = EIO;
      return false;
    }
    done += static_cast<std::size_t>(got);
  }
  return true;
}

struct lz4_resident_window_t {
  std::size_t index       = 0;
  std::size_t file_offset = 0;
  std::size_t size        = 0;
  std::unique_ptr<char[]> data;
};

class lz4_resident_windows_t {
 public:
  explicit lz4_resident_windows_t(std::vector<lz4_resident_window_t>& windows) : windows_(windows)
  {
  }

  const char* ptr_if_contiguous(std::size_t offset, std::size_t size) const
  {
    if (size == 0) return nullptr;
    const auto& w     = window_for_offset(offset);
    std::size_t local = offset - w.file_offset;
    if (local <= w.size && size <= w.size - local) { return w.data.get() + local; }
    return nullptr;
  }

  void copy_to(std::size_t offset, char* dst, std::size_t size) const
  {
    std::size_t copied = 0;
    while (copied < size) {
      const auto& w     = window_for_offset(offset + copied);
      std::size_t local = offset + copied - w.file_offset;
      std::size_t take  = std::min(w.size - local, size - copied);
      std::memcpy(dst + copied, w.data.get() + local, take);
      copied += take;
    }
  }

  uint8_t read_u8(std::size_t offset) const
  {
    uint8_t value = 0;
    copy_to(offset, reinterpret_cast<char*>(&value), sizeof(value));
    return value;
  }

  uint32_t read_u32(std::size_t offset) const
  {
    char bytes[4];
    copy_to(offset, bytes, sizeof(bytes));
    return read_le32(bytes);
  }

  uint64_t read_u64(std::size_t offset) const
  {
    char bytes[8];
    copy_to(offset, bytes, sizeof(bytes));
    return read_le64(bytes);
  }

 private:
  const lz4_resident_window_t& window_for_offset(std::size_t offset) const
  {
    if (windows_.empty()) {
      mps_parser_fail(error_type_t::RuntimeError, "LZ4 resident window lookup with no windows");
    }
    std::size_t lo = 0;
    std::size_t hi = windows_.size();
    while (lo < hi) {
      std::size_t mid = lo + (hi - lo) / 2;
      const auto& w   = windows_[mid];
      if (offset < w.file_offset) {
        hi = mid;
      } else if (offset >= w.file_offset + w.size) {
        lo = mid + 1;
      } else {
        return w;
      }
    }
    mps_parser_fail(error_type_t::RuntimeError, "LZ4 offset outside resident windows");
  }

  std::vector<lz4_resident_window_t>& windows_;
};

}  // namespace

Lz4InputStream::Lz4InputStream(const std::string& path) : path_(path)
{
  MPS_NVTX_RANGE("lz4_input_construct", nvtx::colors::io);

  ensure_lz4_runtime_available();

  fd_ = open_lz4_fd(path);
  ::posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);

  compressed_size_ = get_file_size(fd_, path);

  char header[32];
  if (compressed_size_ < 7) {
    mps_parser_fail(error_type_t::ValidationError,
                    "LZ4 input is too small to contain a frame header");
  }
  std::size_t header_bytes = std::min<std::size_t>(sizeof(header), compressed_size_);
  if (!pread_full_plain(fd_, header, header_bytes, 0)) {
    mps_parser_fail(error_type_t::RuntimeError,
                    "Failed to read LZ4 frame header '%s': %s",
                    path.c_str(),
                    std::strerror(errno));
  }

  std::size_t offset = 0;
  uint32_t magic     = read_le32(header + offset);
  if (magic != lz4_frame_magic) {
    mps_parser_fail(error_type_t::ValidationError,
                    "unsupported LZ4 input: expected standard LZ4 frame magic");
  }
  offset += 4;
  unsigned char flg = static_cast<unsigned char>(header[offset++]);
  unsigned char bd  = static_cast<unsigned char>(header[offset++]);
  unsigned version  = (flg >> 6) & 0x3u;
  if (version != 1) {
    mps_parser_fail(error_type_t::ValidationError, "unsupported LZ4 frame version");
  }
  bool block_independent = (flg & 0x20u) != 0;
  block_checksum_        = (flg & 0x10u) != 0;
  content_size_present_  = (flg & 0x08u) != 0;
  content_checksum_      = (flg & 0x04u) != 0;
  dict_id_               = (flg & 0x01u) != 0;
  if (!block_independent) {
    mps_parser_fail(error_type_t::ValidationError,
                    "parallel LZ4 reader requires independent blocks; compress with -BI");
  }
  block_max_size_ = block_max_size_from_bd(bd);
  if (content_size_present_) {
    if (offset + 8 > header_bytes) {
      mps_parser_fail(error_type_t::ValidationError,
                      "truncated LZ4 frame while reading content size");
    }
    content_size_ = checked_size(read_le64(header + offset), "content size");
    offset += 8;
  }
  if (dict_id_) {
    if (offset + 4 > header_bytes) {
      mps_parser_fail(error_type_t::ValidationError,
                      "truncated LZ4 frame while reading dictionary id");
    }
    offset += 4;
  }
  if (offset + 1 > header_bytes) {
    mps_parser_fail(error_type_t::ValidationError,
                    "truncated LZ4 frame while reading header checksum");
  }
  offset += 1;
  header_size_ = offset;

  std::size_t reserve_size = content_size_;
  if (!content_size_present_) {
    reserve_size =
      checked_mul(compressed_size_, lz4_no_content_size_reserve_ratio, "LZ4 output reserve");
    reserve_size = std::max(reserve_size, block_max_size_);
  }

  constexpr std::size_t huge_alignment = 2 * 1024 * 1024;
  output_mapped_size_                  = round_up_to_multiple(reserve_size, system_page_size());
  output_region_                       = mmap_region_t::anonymous_aligned(output_mapped_size_,
                                                    huge_alignment,
                                                    PROT_NONE,
                                                    MAP_PRIVATE | MAP_NORESERVE,
                                                    "LZ4 output buffer");
  output_data_                         = output_region_.char_data();

  std::size_t block_slots =
    std::max<std::size_t>(1, (reserve_size + block_max_size_ - 1) / block_max_size_ + 1);
  block_done_.resize(block_slots, 0);
  block_end_.resize(block_slots, 0);

  section_scanner_ =
    std::make_unique<mps_section_block_scanner_t>(output_data_, block_slots, registry_);
}

Lz4InputStream::~Lz4InputStream()
{
  if (fd_ >= 0) { ::close(fd_); }
}

const char* Lz4InputStream::data() const noexcept { return output_data_; }
char* Lz4InputStream::mutable_data() noexcept { return output_data_; }
std::size_t Lz4InputStream::size() const noexcept { return output_view_size_; }
std::size_t Lz4InputStream::compressed_size() const noexcept { return compressed_size_; }
std::size_t Lz4InputStream::reserve_size_hint() const noexcept
{
  return content_size_present_ ? content_size_
                               : std::max<std::size_t>(compressed_size_ * 6, 1024 * 1024);
}
mps_phase_registry_t& Lz4InputStream::registry() noexcept { return registry_; }
input_stream_view_t Lz4InputStream::view() noexcept
{
  return {output_data_, output_data_, output_view_size_, compressed_size_, &registry_};
}

void Lz4InputStream::commit_up_to(std::size_t bytes)
{
  MPS_NVTX_RANGE("lz4_commit_output", nvtx::colors::alloc);
  std::lock_guard<std::mutex> lock(commit_mutex_);
  if (bytes <= output_committed_size_) return;
  if (bytes > output_mapped_size_) {
    mps_parser_fail(error_type_t::OutOfMemoryError, "LZ4 output exceeded reserved virtual mapping");
  }
  std::size_t new_committed = round_up_to_multiple(bytes, system_page_size());
  if (new_committed > output_mapped_size_) new_committed = output_mapped_size_;
  std::size_t add = new_committed - output_committed_size_;
  void* target    = output_data_ + output_committed_size_;
  mmap_region_t::map_fixed_or_throw(
    target, add, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0, "LZ4 output commit");
  ::madvise(target, add, MADV_HUGEPAGE);
  output_committed_size_ = new_committed;
}

void Lz4InputStream::run_decode_tasks()
{
  MPS_NVTX_RANGE("lz4_input_run_decode_tasks", nvtx::colors::io);
  std::exception_ptr first_error = nullptr;
  std::mutex error_mutex;
  std::atomic_bool stop_workers{false};
  auto mark_error = [&](std::exception_ptr eptr) {
    std::lock_guard<std::mutex> lock(error_mutex);
    if (!first_error) {
      first_error = eptr;
      stop_workers.store(true, std::memory_order_release);
    }
  };

  const std::size_t window_bytes = lz4_pipeline_batch_bytes;
  const std::size_t window_count = (compressed_size_ + window_bytes - 1) / window_bytes;
  std::vector<lz4_resident_window_t> windows(window_count);
  for (std::size_t i = 0; i < window_count; ++i) {
    std::size_t offset     = i * window_bytes;
    std::size_t size       = std::min(window_bytes, compressed_size_ - offset);
    windows[i].index       = i;
    windows[i].file_offset = offset;
    windows[i].size        = size;
    windows[i].data.reset(new char[size]);
  }

  const std::size_t io_threads = std::min(lz4_input_max_io_threads, window_count);

  struct resident_block_desc_t {
    const char* src                 = nullptr;
    std::size_t compressed_size     = 0;
    std::size_t decompressed_offset = 0;
    std::size_t decompressed_size   = 0;
    std::size_t index               = 0;
    bool uncompressed               = false;
  };

  std::atomic_size_t next_window{0};
  std::vector<unsigned char> window_done(window_count, 0);
  std::mutex window_mutex;
  std::condition_variable window_cv;

  std::deque<std::vector<resident_block_desc_t>> desc_queue;
  bool scanner_done = false;
  std::mutex desc_mutex;
  std::condition_variable desc_cv;

  auto fail_and_notify = [&](std::exception_ptr eptr) {
    mark_error(eptr);
    window_cv.notify_all();
    desc_cv.notify_all();
  };

  auto decode_worker = [&](std::size_t tid) {
    try {
      std::string thread_name = "lz4-window-decode-" + std::to_string(tid);
      nvtx::name_current_thread(thread_name.c_str());
      while (true) {
        std::vector<resident_block_desc_t> batch;
        {
          MPS_NVTX_RANGE("lz4_decode_wait_batch", nvtx::colors::io);
          std::unique_lock<std::mutex> lock(desc_mutex);
          desc_cv.wait(lock, [&] {
            return stop_workers.load(std::memory_order_acquire) || scanner_done ||
                   !desc_queue.empty();
          });
          if (stop_workers.load(std::memory_order_acquire)) { return; }
          if (desc_queue.empty()) {
            if (scanner_done) return;
            continue;
          }
          batch = std::move(desc_queue.front());
          desc_queue.pop_front();
        }

        MPS_NVTX_RANGE("lz4_decode_batch", nvtx::colors::decode);
        for (const auto& block : batch) {
          char* dst  = output_data_ + block.decompressed_offset;
          int actual = 0;
          {
            MPS_NVTX_RANGE("lz4_decode_block_payload", nvtx::colors::decode);
            if (block.uncompressed) {
              std::memcpy(dst, block.src, block.decompressed_size);
              actual = static_cast<int>(block.decompressed_size);
            } else if (block.compressed_size >
                         static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                       block.decompressed_size >
                         static_cast<std::size_t>(std::numeric_limits<int>::max())) {
              actual = -1;
            } else {
              actual = lz4_decompress_safe_runtime(block.src,
                                                   dst,
                                                   static_cast<int>(block.compressed_size),
                                                   static_cast<int>(block.decompressed_size));
            }
          }
          if (actual < 0 || static_cast<std::size_t>(actual) > block.decompressed_size) {
            mps_parser_fail(error_type_t::ValidationError,
                            "LZ4 input block decompressed to invalid size");
          }

          std::size_t actual_size = static_cast<std::size_t>(actual);
          {
            MPS_NVTX_RANGE("lz4_section_scan_block", nvtx::colors::generic);
            section_scanner_->observe_block(block.index, dst, dst + actual_size);
          }
          std::size_t before = 0;
          std::size_t after  = 0;
          {
            MPS_NVTX_RANGE("lz4_frontier_update", nvtx::colors::generic);
            frontier_mutex_.lock();
            block_done_[block.index] = 1;
            block_end_[block.index]  = block.decompressed_offset + actual_size;
            before                   = ready_bytes_;
            while (next_block_ < block_done_.size() && block_done_[next_block_]) {
              ready_bytes_ = block_end_[next_block_];
              ++next_block_;
            }
            after = ready_bytes_;
            frontier_mutex_.unlock();
          }
          if (after > before) {
            MPS_NVTX_RANGE("lz4_publish_ready", nvtx::colors::generic);
            section_scanner_->publish_ready(after);
          }
        }
      }
    } catch (...) {
      fail_and_notify(std::current_exception());
    }
  };

  std::vector<std::thread> readers;
  readers.reserve(io_threads);
  for (std::size_t t = 0; t < io_threads; ++t) {
    readers.emplace_back([&, t] {
      std::string thread_name = "lz4-window-read-" + std::to_string(t);
      nvtx::name_current_thread(thread_name.c_str());
      while (!stop_workers.load(std::memory_order_acquire)) {
        std::size_t index = next_window.fetch_add(1, std::memory_order_relaxed);
        if (index >= windows.size()) { break; }
        auto& w = windows[index];
        bool ok = false;
        {
          MPS_NVTX_RANGE("lz4_window_pread", nvtx::colors::io);
          ok = pread_full_plain(fd_, w.data.get(), w.size, w.file_offset);
        }
        if (!ok) {
          try {
            mps_parser_fail(error_type_t::RuntimeError,
                            "Failed to pread LZ4 resident window: %s",
                            std::strerror(errno));
          } catch (...) {
            fail_and_notify(std::current_exception());
          }
          return;
        }
        {
          MPS_NVTX_RANGE("lz4_window_publish", nvtx::colors::generic);
          std::lock_guard<std::mutex> lock(window_mutex);
          window_done[index] = 1;
        }
        window_cv.notify_all();
      }
    });
  }

  std::atomic_size_t blocks_scanned{0};
  std::vector<std::vector<char>> crossing_payloads;
  std::thread scanner([&] {
    try {
      nvtx::name_current_thread("lz4-metadata-scan");
      lz4_resident_windows_t resident(windows);
      auto wait_range_ready = [&](std::size_t begin, std::size_t size) {
        if (size == 0) return;
        std::size_t first = begin / window_bytes;
        std::size_t last  = (begin + size - 1) / window_bytes;
        for (std::size_t wi = first; wi <= last; ++wi) {
          MPS_NVTX_RANGE("lz4_metadata_wait_window", nvtx::colors::io);
          std::unique_lock<std::mutex> lock(window_mutex);
          window_cv.wait(lock, [&] {
            return stop_workers.load(std::memory_order_acquire) || window_done[wi] != 0;
          });
          if (stop_workers.load(std::memory_order_acquire) && window_done[wi] == 0) {
            mps_parser_fail(error_type_t::RuntimeError,
                            "LZ4 metadata scanner stopped before required window was ready");
          }
        }
      };
      auto push_batch = [&](std::vector<resident_block_desc_t>& batch) {
        if (batch.empty()) return;
        {
          MPS_NVTX_RANGE("lz4_metadata_commit_batch", nvtx::colors::alloc);
          commit_up_to(batch.back().decompressed_offset + batch.back().decompressed_size);
        }
        {
          MPS_NVTX_RANGE("lz4_metadata_enqueue_batch", nvtx::colors::generic);
          std::lock_guard<std::mutex> lock(desc_mutex);
          desc_queue.push_back(std::move(batch));
        }
        batch.clear();
        desc_cv.notify_one();
      };

      std::vector<resident_block_desc_t> batch;
      batch.reserve(1024);
      std::size_t offset              = header_size_;
      std::size_t decompressed_offset = 0;
      while (true) {
        MPS_NVTX_RANGE("lz4_metadata_scan_block", nvtx::colors::generic);
        wait_range_ready(offset, 4);
        if (offset + 4 > compressed_size_) {
          mps_parser_fail(error_type_t::ValidationError,
                          "truncated LZ4 frame while reading block header");
        }
        uint32_t raw_block_size = resident.read_u32(offset);
        offset += 4;
        if (raw_block_size == 0) { break; }

        bool uncompressed              = (raw_block_size & lz4_uncompressed_block) != 0;
        std::size_t block_payload_size = raw_block_size & lz4_block_size_mask;
        if (block_payload_size == 0) {
          mps_parser_fail(error_type_t::ValidationError, "invalid zero-sized LZ4 data block");
        }
        if (block_payload_size > block_max_size_ && uncompressed) {
          mps_parser_fail(error_type_t::ValidationError,
                          "LZ4 uncompressed block exceeds frame block maximum");
        }
        if (content_size_present_ && decompressed_offset >= content_size_) {
          mps_parser_fail(error_type_t::ValidationError,
                          "LZ4 frame contains more blocks than content size allows");
        }
        wait_range_ready(offset, block_payload_size);
        if (offset + block_payload_size > compressed_size_) {
          mps_parser_fail(error_type_t::ValidationError,
                          "truncated LZ4 frame while reading block payload");
        }

        std::size_t decompressed_size = block_payload_size;
        if (!uncompressed) {
          if (content_size_present_) {
            decompressed_size = std::min(block_max_size_, content_size_ - decompressed_offset);
          } else {
            decompressed_size = block_max_size_;
          }
        }
        if (content_size_present_ && decompressed_size > content_size_ - decompressed_offset) {
          mps_parser_fail(error_type_t::ValidationError, "LZ4 block exceeds declared content size");
        }

        const char* src = resident.ptr_if_contiguous(offset, block_payload_size);
        if (src == nullptr) {
          crossing_payloads.emplace_back(block_payload_size);
          resident.copy_to(offset, crossing_payloads.back().data(), block_payload_size);
          src = crossing_payloads.back().data();
        }
        batch.push_back({src,
                         block_payload_size,
                         decompressed_offset,
                         decompressed_size,
                         blocks_scanned.load(std::memory_order_relaxed),
                         uncompressed});
        blocks_scanned.fetch_add(1, std::memory_order_relaxed);
        decompressed_offset += decompressed_size;
        offset += block_payload_size;
        if (block_checksum_) {
          wait_range_ready(offset, 4);
          if (offset + 4 > compressed_size_) {
            mps_parser_fail(error_type_t::ValidationError,
                            "truncated LZ4 frame while reading block checksum");
          }
          offset += 4;
        }
        if (blocks_scanned.load(std::memory_order_relaxed) > block_done_.size()) {
          mps_parser_fail(error_type_t::OutOfMemoryError,
                          "LZ4 input block count exceeded reserved metadata slots");
        }
        if (batch.size() >= 1024) { push_batch(batch); }
      }
      if (content_checksum_) {
        wait_range_ready(offset, 4);
        if (offset + 4 > compressed_size_) {
          mps_parser_fail(error_type_t::ValidationError,
                          "truncated LZ4 frame while reading content checksum");
        }
        offset += 4;
      }
      if (content_size_present_ && decompressed_offset != content_size_) {
        mps_parser_fail(error_type_t::ValidationError,
                        "LZ4 frame ended before declared content size was reached");
      }
      if (offset != compressed_size_) {
        mps_parser_fail(error_type_t::ValidationError,
                        "LZ4 input contains trailing data after the first frame");
      }
      push_batch(batch);
      {
        std::lock_guard<std::mutex> lock(desc_mutex);
        scanner_done = true;
      }
      desc_cv.notify_all();
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(desc_mutex);
        scanner_done = true;
      }
      fail_and_notify(std::current_exception());
    }
  });

  std::vector<std::thread> io_workers;
  io_workers.reserve(io_threads);
  for (std::size_t t = 0; t < io_threads; ++t) {
    io_workers.emplace_back(decode_worker, t);
  }
  for (auto& reader : readers) {
    reader.join();
  }
  scanner.join();
  for (auto& worker : io_workers) {
    worker.join();
  }
  if (first_error) std::rethrow_exception(first_error);
  output_view_size_ = ready_bytes_;
  section_scanner_->publish_ready(output_view_size_);
}

}  // namespace mps_fast
