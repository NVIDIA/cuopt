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

#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct segmented_sum_handler_t {
  segmented_sum_handler_t(rmm::cuda_stream_view stream_view) : stream_view_(stream_view) {}

  template <typename InputIteratorT, typename OutputIteratorT>
  void segmented_sum_helper(InputIteratorT input,
                            OutputIteratorT output,
                            i_t batch_size,
                            i_t problem_size)
  {
    cub::DeviceSegmentedReduce::Sum(
      nullptr, byte_needed_, input, output, batch_size, problem_size, stream_view_);

    segmented_sum_storage_.resize(byte_needed_, stream_view_);

    cub::DeviceSegmentedReduce::Sum(segmented_sum_storage_.data(),
                                    byte_needed_,
                                    input,
                                    output,
                                    batch_size,
                                    problem_size,
                                    stream_view_);
  }

  template <typename InputIteratorT, typename ReductionOpT>
  void segmented_reduce_helper(InputIteratorT input,
                               f_t* output,
                               i_t batch_size,
                               i_t problem_size,
                               ReductionOpT reduction_op,
                               f_t initial_value)
  {
    cub::DeviceSegmentedReduce::Reduce(nullptr,
                                       byte_needed_,
                                       input,
                                       output,
                                       batch_size,
                                       problem_size,
                                       reduction_op,
                                       initial_value,
                                       stream_view_.value());

    segmented_sum_storage_.resize(byte_needed_, stream_view_.value());

    cub::DeviceSegmentedReduce::Reduce(segmented_sum_storage_.data(),
                                       byte_needed_,
                                       input,
                                       output,
                                       batch_size,
                                       problem_size,
                                       reduction_op,
                                       initial_value,
                                       stream_view_.value());
  }

  size_t byte_needed_;
  rmm::device_buffer segmented_sum_storage_;
  rmm::cuda_stream_view stream_view_;
};

}  // namespace cuopt::linear_programming::detail
