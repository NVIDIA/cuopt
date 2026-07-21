// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sys/mman.h>
#include <sys/types.h>

#include <cuda/cmath>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <utilities/error.hpp>

#include <limits>
#include <stdexcept>
#include <string>

namespace cuopt::mathematical_optimization::io::detail {

using cuopt::mathematical_optimization::io::error_type_t;
using cuopt::mathematical_optimization::io::mps_parser_expects;
using cuopt::mathematical_optimization::io::mps_parser_fail;

// Move-only owner for a Linux mmap range. Fixed sub-maps inside a reserved range
// are still released by unmapping the owning outer range.
class mmap_region_t {
 public:
  mmap_region_t() = default;
  mmap_region_t(void* ptr, std::size_t size) noexcept : ptr_(ptr), size_(size) {}

  mmap_region_t(const mmap_region_t&)            = delete;
  mmap_region_t& operator=(const mmap_region_t&) = delete;

  mmap_region_t(mmap_region_t&& other) noexcept
    : ptr_(other.ptr_),
      size_(other.size_),
      unmap_ptr_(other.unmap_ptr_),
      unmap_size_(other.unmap_size_)
  {
    other.ptr_        = nullptr;
    other.size_       = 0;
    other.unmap_ptr_  = nullptr;
    other.unmap_size_ = 0;
  }

  mmap_region_t& operator=(mmap_region_t&& other) noexcept
  {
    if (this != &other) {
      reset();
      ptr_              = other.ptr_;
      size_             = other.size_;
      unmap_ptr_        = other.unmap_ptr_;
      unmap_size_       = other.unmap_size_;
      other.ptr_        = nullptr;
      other.size_       = 0;
      other.unmap_ptr_  = nullptr;
      other.unmap_size_ = 0;
    }
    return *this;
  }

  ~mmap_region_t() { reset(); }

 private:
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

 public:
  static mmap_region_t anonymous(std::size_t size, int prot, int flags, const char* context)
  {
    return map(nullptr, size, prot, flags | MAP_ANONYMOUS, -1, 0, context);
  }

  static mmap_region_t anonymous_aligned(
    std::size_t size, std::size_t alignment, int prot, int flags, const char* context)
  {
    if (!cuda::is_power_of_two(alignment)) {
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
    return mmap_region_t(reinterpret_cast<void*>(aligned_addr), size, raw, raw_size);
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
    void* base      = unmap_ptr_ != nullptr ? unmap_ptr_ : ptr_;
    std::size_t len = unmap_ptr_ != nullptr ? unmap_size_ : size_;
    if (base != nullptr && len != 0) { ::munmap(base, len); }
    ptr_        = nullptr;
    size_       = 0;
    unmap_ptr_  = nullptr;
    unmap_size_ = 0;
  }

  void advise(int advice) const noexcept
  {
    if (ptr_ != nullptr && size_ != 0) { ::madvise(ptr_, size_, advice); }
  }

  void* data() noexcept { return ptr_; }
  char* char_data() noexcept { return (char*)ptr_; }
  std::size_t size() const noexcept { return size_; }

 private:
  mmap_region_t(void* ptr, std::size_t size, void* unmap_ptr, std::size_t unmap_size) noexcept
    : ptr_(ptr), size_(size), unmap_ptr_(unmap_ptr), unmap_size_(unmap_size)
  {
  }

  void* ptr_              = nullptr;
  std::size_t size_       = 0;
  void* unmap_ptr_        = nullptr;
  std::size_t unmap_size_ = 0;
};

}  // namespace cuopt::mathematical_optimization::io::detail
