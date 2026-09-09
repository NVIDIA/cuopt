/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <google/protobuf/repeated_field.h>

#include <cstdint>
#include <vector>

namespace cuopt {
namespace routing {
namespace grpc_map_detail {

template <typename T>
inline void copy_repeated_to_vector(const google::protobuf::RepeatedField<T>& src,
                                    std::vector<T>& dst)
{
  dst.assign(src.begin(), src.end());
}

template <typename T>
inline void copy_vector_to_repeated(const std::vector<T>& src,
                                    google::protobuf::RepeatedField<T>* dst)
{
  dst->Clear();
  dst->Reserve(static_cast<int>(src.size()));
  for (auto v : src) {
    dst->Add(v);
  }
}

inline void copy_u32_to_u8(const google::protobuf::RepeatedField<uint32_t>& src,
                           std::vector<uint8_t>& dst)
{
  dst.clear();
  dst.reserve(static_cast<size_t>(src.size()));
  for (auto v : src) {
    dst.push_back(static_cast<uint8_t>(v));
  }
}

inline void copy_bool_to_u8(const google::protobuf::RepeatedField<bool>& src,
                            std::vector<uint8_t>& dst)
{
  dst.clear();
  dst.reserve(static_cast<size_t>(src.size()));
  for (bool v : src) {
    dst.push_back(v ? 1 : 0);
  }
}

}  // namespace grpc_map_detail
}  // namespace routing
}  // namespace cuopt
