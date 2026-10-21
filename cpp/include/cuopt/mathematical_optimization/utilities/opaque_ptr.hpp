/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utility>

namespace cuopt {
namespace mathematical_optimization {

/**
 * @brief An owning pointer with value semantics whose copy and destroy are fixed at creation.
 *
 * Lets a header hold state that only the GPU library can build (for example device buffers)
 * without making every user of the header link the GPU library: creating one is the only
 * operation that needs T to be complete, and it is done by the library that knows T. Copying,
 * moving and destroying go through function pointers recorded at that point, so a translation
 * unit that never creates one (a CPU-only client) needs no GPU symbol for any of them. An empty
 * one copies and destroys trivially.
 *
 * Copying clones the pointee, like a by-value member would; it does not share it.
 */
template <typename T>
class opaque_ptr_t {
 public:
  opaque_ptr_t() = default;

  opaque_ptr_t(const opaque_ptr_t& other)
    : p_(other.p_ ? other.clone_(other.p_) : nullptr), clone_(other.clone_), del_(other.del_)
  {
  }

  opaque_ptr_t(opaque_ptr_t&& other) noexcept
    : p_(std::exchange(other.p_, nullptr)), clone_(other.clone_), del_(other.del_)
  {
  }

  opaque_ptr_t& operator=(const opaque_ptr_t& other)
  {
    if (this != &other) {
      opaque_ptr_t copy(other);
      swap(copy);
    }
    return *this;
  }

  opaque_ptr_t& operator=(opaque_ptr_t&& other) noexcept
  {
    if (this != &other) {
      reset();
      p_     = std::exchange(other.p_, nullptr);
      clone_ = other.clone_;
      del_   = other.del_;
    }
    return *this;
  }

  ~opaque_ptr_t() { reset(); }

  /** Creates a T. Instantiated only where T is complete, i.e. in the library that owns T. */
  template <typename... Args>
  static opaque_ptr_t make(Args&&... args)
  {
    opaque_ptr_t out;
    out.p_     = new T(std::forward<Args>(args)...);
    out.clone_ = [](const T* p) { return new T(*p); };
    out.del_   = [](T* p) { delete p; };
    return out;
  }

  T* get() const noexcept { return p_; }
  explicit operator bool() const noexcept { return p_ != nullptr; }

  void swap(opaque_ptr_t& other) noexcept
  {
    std::swap(p_, other.p_);
    std::swap(clone_, other.clone_);
    std::swap(del_, other.del_);
  }

 private:
  void reset() noexcept
  {
    if (p_) { del_(p_); }
    p_ = nullptr;
  }

  T* p_                  = nullptr;
  T* (*clone_)(const T*) = nullptr;
  void (*del_)(T*)       = nullptr;
};

}  // namespace mathematical_optimization
}  // namespace cuopt
