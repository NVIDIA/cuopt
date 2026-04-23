/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cassert>
#include <utility>
#include <vector>

namespace cuopt {

// A fixed-capacity double-ended queue backed by a circular buffer.
// All operations are O(1) with no dynamic allocation after construction.
//
// Preconditions (asserted in debug builds):
//   - push_front / push_back : size() < capacity()
//   - pop_front  / pop_back  : !empty()
//   - front / back           : !empty()
template <typename T>
class circular_deque_t {
 public:
  circular_deque_t() : buffer_(1), capacity_(1), head_(0), tail_(0) {}

  // Allocates storage for exactly `capacity` elements up front.
  explicit circular_deque_t(size_t capacity)
    : buffer_(capacity + 1),  // +1 to distinguish full (next(tail)==head) from empty (head==tail)
      capacity_(capacity + 1),
      head_(0),
      tail_(0)
  {
    assert(capacity > 0);
  }

  bool empty() const { return head_ == tail_; }
  bool full() const { return next(tail_) == head_; }

  size_t size() const { return (tail_ - head_ + capacity_) % capacity_; }
  size_t capacity() const { return capacity_ - 1; }

  void clear_resize(size_t new_capacity)
  {
    assert(new_capacity > 0);
    head_     = 0;
    tail_     = 0;
    capacity_ = new_capacity + 1;
    buffer_.resize(capacity_);
  }

  void push_back(T val)
  {
    assert(!full());
    buffer_[tail_] = std::move(val);
    tail_          = next(tail_);
  }

  void push_front(T val)
  {
    assert(!full());
    head_          = prev(head_);
    buffer_[head_] = std::move(val);
  }

  T pop_front()
  {
    assert(!empty());
    T val = std::move(buffer_[head_]);
    head_ = next(head_);
    return val;
  }

  T pop_back()
  {
    assert(!empty());
    tail_ = prev(tail_);
    return std::move(buffer_[tail_]);
  }

  T& front()
  {
    assert(!empty());
    return buffer_[head_];
  }
  const T& front() const
  {
    assert(!empty());
    return buffer_[head_];
  }

  T& back()
  {
    assert(!empty());
    return buffer_[prev(tail_)];
  }
  const T& back() const
  {
    assert(!empty());
    return buffer_[prev(tail_)];
  }

 private:
  size_t next(size_t idx) const { return (idx + 1) % capacity_; }
  size_t prev(size_t idx) const { return (idx + capacity_ - 1) % capacity_; }

  std::vector<T> buffer_;
  size_t capacity_;
  size_t head_;
  size_t tail_;
};

}  // namespace cuopt
