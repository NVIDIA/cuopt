/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <utility>
#include <vector>

namespace cuopt {

/**
 * @brief A queue that orders work units by timestamp (float). The earliest timestamp is at the
 * front.
 *
 * @tparam T The type of the work unit.
 */
template <typename T>
class work_unit_ordered_queue_t {
 public:
  // Work entry: (timestamp, work)
  using entry_t = std::pair<float, T>;

  work_unit_ordered_queue_t() = default;

  /**
   * @brief Push a work unit with a timestamp into the queue.
   *
   * @param timestamp The associated float timestamp.
   * @param work The work unit to enqueue.
   */
  void push(float timestamp, const T& work)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.emplace(timestamp, work);
  }

  /**
   * @brief Push a work unit with a timestamp into the queue (move version).
   *
   * @param timestamp The associated float timestamp.
   * @param work The work unit to enqueue (moved).
   */
  void push(float timestamp, T&& work)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.emplace(timestamp, std::move(work));
  }

  /**
   * @brief Pop the entry at the front of the queue.
   */
  void pop()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.pop();
  }

  /**
   * @brief Get the entry at the front of the queue.
   *
   * @return const entry_t& Earliest (timestamp, work).
   */
  entry_t top_with_timestamp() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.top();
  }

  /**
   * @brief Get only the content, not the timestamp.
   *
   */
  T top() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.top().second;
  }

  /**
   * @brief Check if the queue is empty.
   *
   * @return true if empty, false otherwise.
   */
  bool empty() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  /**
   * @brief Number of items in the queue.
   */
  std::size_t size() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  void clear()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_ = std::priority_queue<entry_t, std::vector<entry_t>, min_heap_cmp>();
  }

 private:
  // Custom comparator for min-heap based on timestamp
  struct min_heap_cmp {
    bool operator()(const entry_t& a, const entry_t& b) const
    {
      return a.first > b.first;  // earlier time has higher priority
    }
  };

  std::priority_queue<entry_t, std::vector<entry_t>, min_heap_cmp> queue_;
  mutable std::mutex mutex_;
};

}  // namespace cuopt
