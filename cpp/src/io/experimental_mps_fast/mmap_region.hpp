// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sys/mman.h>
#include <sys/types.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <utilities/error.hpp>

#include <limits>
#include <stdexcept>
#include <string>

namespace mps_fast {

using cuopt::linear_programming::io::error_type_t;
using cuopt::linear_programming::io::mps_parser_expects;
using cuopt::linear_programming::io::mps_parser_fail;

// Move-only owner for a Linux mmap range. Fixed sub-maps inside a reserved range
// are still released by unmapping the owning outer range.
class mmap_region_t {
 public:
  mmap_region_t() = default;
  mmap_region_t(void* ptr, std::size_t size) noexcept : ptr_(ptr), size_(size) {}

  mmap_region_t(const mmap_region_t&)            = delete;
  mmap_region_t& operator=(const mmap_region_t&) = delete;

  mmap_region_t(mmap_region_t&& other) noexcept : ptr_(other.ptr_), size_(other.size_)
  {
    other.ptr_  = nullptr;
    other.size_ = 0;
  }

  mmap_region_t& operator=(mmap_region_t&& other) noexcept
  {
    if (this != &other) {
      reset();
      ptr_        = other.ptr_;
      size_       = other.size_;
      other.ptr_  = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  ~mmap_region_t() { reset(); }

  static mmap_region_t map(
    void* address, std::size_t size, int prot, int flags, int fd, off_t offset, const char* context)
  {
    void* ptr = ::mmap(address, size, prot, flags, fd, offset);
    if (ptr == MAP_FAILED) {
      mps_parser_fail(
        error_type_t::RuntimeError, "mmap failed for %s: %s", context, std::strerror(errno));
    }
    return mmap_region_t(ptr, size);
  }

  static mmap_region_t anonymous(std::size_t size, int prot, int flags, const char* context)
  {
    return map(nullptr, size, prot, flags | MAP_ANONYMOUS, -1, 0, context);
  }

  static mmap_region_t anonymous_aligned(
    std::size_t size, std::size_t alignment, int prot, int flags, const char* context)
  {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
      mps_parser_fail(error_type_t::RuntimeError,
                      "mmap aligned allocation requires power-of-two alignment");
    }
    if (size > std::numeric_limits<std::size_t>::max() - alignment) {
      mps_parser_fail(error_type_t::OutOfMemoryError, "mmap aligned allocation size overflow");
    }

    std::size_t raw_size = size + alignment;
    void* raw            = ::mmap(nullptr, raw_size, prot, flags | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) {
      mps_parser_fail(
        error_type_t::RuntimeError, "mmap failed for %s: %s", context, std::strerror(errno));
    }

    uintptr_t raw_addr     = reinterpret_cast<uintptr_t>(raw);
    uintptr_t aligned_addr = (raw_addr + alignment - 1) & ~(uintptr_t)(alignment - 1);
    std::size_t prefix     = static_cast<std::size_t>(aligned_addr - raw_addr);
    std::size_t suffix     = raw_size - prefix - size;
    if (prefix > 0) { ::munmap(raw, prefix); }
    if (suffix > 0) { ::munmap(reinterpret_cast<void*>(aligned_addr + size), suffix); }
    return mmap_region_t(reinterpret_cast<void*>(aligned_addr), size);
  }

  static void map_fixed_or_throw(
    void* address, std::size_t size, int prot, int flags, int fd, off_t offset, const char* context)
  {
    void* ptr = ::mmap(address, size, prot, flags | MAP_FIXED, fd, offset);
    if (ptr == MAP_FAILED) {
      mps_parser_fail(
        error_type_t::RuntimeError, "mmap failed for %s: %s", context, std::strerror(errno));
    }
  }

  void reset() noexcept
  {
    if (ptr_ != nullptr && size_ != 0) { ::munmap(ptr_, size_); }
    ptr_  = nullptr;
    size_ = 0;
  }

  void reset(void* ptr, std::size_t size) noexcept
  {
    reset();
    ptr_  = ptr;
    size_ = size;
  }

  void* release() noexcept
  {
    void* ptr = ptr_;
    ptr_      = nullptr;
    size_     = 0;
    return ptr;
  }

  void advise(int advice) const noexcept
  {
    if (ptr_ != nullptr && size_ != 0) { ::madvise(ptr_, size_, advice); }
  }

  void* data() noexcept { return ptr_; }
  const void* data() const noexcept { return ptr_; }
  char* char_data() noexcept { return static_cast<char*>(ptr_); }
  const char* char_data() const noexcept { return static_cast<const char*>(ptr_); }
  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return ptr_ == nullptr || size_ == 0; }
  explicit operator bool() const noexcept { return !empty(); }

 private:
  void* ptr_        = nullptr;
  std::size_t size_ = 0;
};

}  // namespace mps_fast
