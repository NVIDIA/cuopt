/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/stream>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>

namespace cuopt {

template <typename i_t, typename f_t>
struct segmented_sum_handler_t {
  segmented_sum_handler_t(cuda::stream_ref stream_view) : stream_view_(stream_view) {}

  template <typename InputIteratorT, typename OutputIteratorT>
  void segmented_sum_helper(InputIteratorT input,
                            OutputIteratorT output,
                            i_t batch_size,
                            i_t problem_size)
  {
    cub::DeviceSegmentedReduce::Sum(
      nullptr, byte_needed_, input, output, batch_size, problem_size, stream_view_.get());

    segmented_sum_storage_.resize(byte_needed_, stream_view_);

    cub::DeviceSegmentedReduce::Sum(segmented_sum_storage_.data(),
                                    byte_needed_,
                                    input,
                                    output,
                                    batch_size,
                                    problem_size,
                                    stream_view_.get());
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
                                       stream_view_.get());

    segmented_sum_storage_.resize(byte_needed_, stream_view_.get());

    cub::DeviceSegmentedReduce::Reduce(segmented_sum_storage_.data(),
                                       byte_needed_,
                                       input,
                                       output,
                                       batch_size,
                                       problem_size,
                                       reduction_op,
                                       initial_value,
                                       stream_view_.get());
  }

  size_t byte_needed_;
  rmm::device_buffer segmented_sum_storage_;
  cuda::stream_ref stream_view_;
};

}  // namespace cuopt
