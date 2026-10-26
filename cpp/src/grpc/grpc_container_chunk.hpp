/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt_remote_service.pb.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization {

// FieldId is ArrayFieldId for LP/MIP top-level arrays and int32_t for routing
// arrays, whose ids are not all members of ArrayFieldId.
template <typename T, typename FieldId>
void chunk_typed_array(std::vector<cuopt::remote::SendArrayChunkRequest>& out,
                       FieldId field_id,
                       const std::vector<T>& data,
                       const std::string& upload_id,
                       int64_t chunk_data_budget)
{
  if (data.empty()) return;

  const int64_t elem_size      = static_cast<int64_t>(sizeof(T));
  const int64_t total_elements = static_cast<int64_t>(data.size());

  int64_t elems_per_chunk = chunk_data_budget / elem_size;
  if (elems_per_chunk <= 0) elems_per_chunk = 1;

  const auto* raw = reinterpret_cast<const uint8_t*>(data.data());

  for (int64_t offset = 0; offset < total_elements; offset += elems_per_chunk) {
    int64_t count       = std::min(elems_per_chunk, total_elements - offset);
    int64_t byte_offset = offset * elem_size;
    int64_t byte_count  = count * elem_size;

    cuopt::remote::SendArrayChunkRequest req;
    req.set_upload_id(upload_id);
    auto* ac = req.mutable_chunk();
    ac->set_field_id(field_id);
    ac->set_element_offset(offset);
    ac->set_total_elements(total_elements);
    ac->set_data(raw + byte_offset, byte_count);
    out.push_back(std::move(req));
  }
}

template <typename T>
void chunk_top_level_typed_array(std::vector<cuopt::remote::SendArrayChunkRequest>& out,
                                 int32_t field_id,
                                 const std::vector<T>& data,
                                 const std::string& upload_id,
                                 int64_t chunk_data_budget)
{
  chunk_typed_array(out, field_id, data, upload_id, chunk_data_budget);
}

// Stamps each chunk with (container_field_num, container_index, container-relative field_id)
// so the server can rebuild one entry of a repeated message.
template <typename T>
void chunk_container_typed_array(std::vector<cuopt::remote::SendArrayChunkRequest>& out,
                                 int32_t container_field_num,
                                 int32_t container_index,
                                 int32_t field_id,
                                 const std::vector<T>& data,
                                 const std::string& upload_id,
                                 int64_t chunk_data_budget)
{
  const size_t first = out.size();
  chunk_typed_array(out, field_id, data, upload_id, chunk_data_budget);
  for (size_t i = first; i < out.size(); ++i) {
    auto* ac = out[i].mutable_chunk();
    ac->set_container_field_num(container_field_num);
    ac->set_container_index(container_index);
  }
}

}  // namespace cuopt::mathematical_optimization
