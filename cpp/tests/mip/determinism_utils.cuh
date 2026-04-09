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

#include <thread>

#include <cuda/atomic>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>

#include <utilities/copy_helpers.hpp>

namespace cuopt::linear_programming::test {

static __global__ void spin_kernel(int* flag, unsigned long long timeout_clocks = 10000000)
{
  cuda::atomic_ref<int> flag_ref(*flag);

  long long int start_clock, sample_clock;
  start_clock = clock64();

  while (flag_ref.load() == 0) {
    sample_clock = clock64();

    if (sample_clock - start_clock > timeout_clocks) { break; }
  }
}

static void launch_spin_kernel_stream_thread(rmm::cuda_stream_view stream_view, int* flag)
{
  while (true) {
    int blocks  = rand() % 64 + 1;
    int threads = rand() % 1024 + 1;
    spin_kernel<<<blocks, threads, 0, stream_view>>>(flag);
    cudaStreamSynchronize(stream_view);
    if (host_copy(flag, 1, stream_view)[0] != 0) { break; }
    std::this_thread::sleep_for(std::chrono::milliseconds(rand() % 1000 + 1));
  }
}

class spin_stream_raii_t {
 public:
  spin_stream_raii_t()
    : flag(0, stream), spin_thread(launch_spin_kernel_stream_thread, stream.view(), flag.data())
  {
  }

  ~spin_stream_raii_t()
  {
    int one = 1;
    flag.set_value_async(one, stream);
    spin_thread.join();
  }

 private:
  rmm::cuda_stream stream;
  rmm::device_scalar<int> flag;
  std::thread spin_thread;
};

}  // namespace cuopt::linear_programming::test
