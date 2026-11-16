/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <rmm/device_uvector.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda/memory_resource>

namespace cuopt::mathematical_optimization::pdlp {

using device_memory_resource_t = cuda::mr::any_resource<cuda::mr::device_accessible>;

/**
 * Build an owning resource for PDLP's private working storage. On supported devices, plain CUDA
 * resources use a pool backed by compressible allocations. Explicit caller resources, including
 * pools, tracking and limiting adaptors, retain their allocation policy. No global resource
 * changes. Buffers must receive the owning resource by value so they retain it when moved out of
 * the solver.
 */
device_memory_resource_t make_pdlp_memory_resource(
  device_memory_resource_t upstream = rmm::mr::get_current_device_resource_ref());

// Identifies only the private pool created above, never an unrelated caller callback or pool.
bool is_pdlp_memory_resource(device_memory_resource_t const& resource) noexcept;

// Returned solutions must not retain a solve's entire working pool. Copy only buffers belonging
// to that pool onto the caller's resource; untouched caller buffers do not need another copy.
// The caller synchronizes the stream before destroying the private working resource.
template <typename T>
void detach_pdlp_buffer(rmm::device_uvector<T>& buffer,
                        device_memory_resource_t const& working_resource,
                        device_memory_resource_t const& output_resource,
                        cuda::stream_ref stream)
{
  if (working_resource != output_resource && buffer.memory_resource() == working_resource) {
    buffer = rmm::device_uvector<T>(buffer, stream, output_resource);
  }
}

}  // namespace cuopt::mathematical_optimization::pdlp
