/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace cuopt::linear_programming::detail {

template <typename Derived>
class cpu_worker_thread_base_t {
 public:
  cpu_worker_thread_base_t();
  ~cpu_worker_thread_base_t();

  void start_cpu_solver();
  bool wait_for_cpu_solver();

  // Derived classes MUST call this in their destructor before the base destructor runs.
  // This ensures on_terminate() is called while the derived object is still fully alive.
  void request_termination();

  // Internal method for thread management - safe to call during destruction
  void join_worker();
  void cpu_worker_thread();

  std::thread cpu_worker;
  std::mutex cpu_mutex;
  std::condition_variable cpu_cv;
  std::atomic<bool> should_stop{false};
  std::atomic<bool> cpu_thread_should_start{false};
  std::atomic<bool> cpu_thread_done{true};
  std::atomic<bool> cpu_thread_terminate{false};
};

template <typename Derived>
cpu_worker_thread_base_t<Derived>::cpu_worker_thread_base_t()
{
  cpu_worker = std::thread(&cpu_worker_thread_base_t<Derived>::cpu_worker_thread, this);
}

template <typename Derived>
cpu_worker_thread_base_t<Derived>::~cpu_worker_thread_base_t()
{
  // Note: We don't call on_terminate() here since the derived object is already destroyed.
  join_worker();
}

template <typename Derived>
void cpu_worker_thread_base_t<Derived>::cpu_worker_thread()
{
  while (!cpu_thread_terminate) {
    {
      std::unique_lock<std::mutex> lock(cpu_mutex);
      cpu_cv.wait(lock, [this] { return cpu_thread_should_start || cpu_thread_terminate; });

      if (cpu_thread_terminate) break;

      cpu_thread_done         = false;
      cpu_thread_should_start = false;
    }

    static_cast<Derived*>(this)->run_worker();

    {
      std::lock_guard<std::mutex> lock(cpu_mutex);
      cpu_thread_done = true;
    }
  }
}

template <typename Derived>
void cpu_worker_thread_base_t<Derived>::request_termination()
{
  bool should_terminate = false;
  {
    std::lock_guard<std::mutex> lock(cpu_mutex);
    if (cpu_thread_terminate) return;
    cpu_thread_terminate = true;
    should_terminate     = true;
    static_cast<Derived*>(this)->on_terminate();
  }

  if (should_terminate) {
    cpu_cv.notify_one();
    join_worker();
  }
}

template <typename Derived>
void cpu_worker_thread_base_t<Derived>::join_worker()
{
  {
    std::lock_guard<std::mutex> lock(cpu_mutex);
    if (!cpu_thread_terminate) { cpu_thread_terminate = true; }
  }
  cpu_cv.notify_one();

  if (cpu_worker.joinable()) { cpu_worker.join(); }
}

template <typename Derived>
void cpu_worker_thread_base_t<Derived>::start_cpu_solver()
{
  {
    std::lock_guard<std::mutex> lock(cpu_mutex);
    cpu_thread_done         = false;
    cpu_thread_should_start = true;
    static_cast<Derived*>(this)->on_start();
  }
  cpu_cv.notify_one();
}

template <typename Derived>
bool cpu_worker_thread_base_t<Derived>::wait_for_cpu_solver()
{
  while (!cpu_thread_done && !cpu_thread_terminate) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  return static_cast<Derived*>(this)->get_result();
}

}  // namespace cuopt::linear_programming::detail
