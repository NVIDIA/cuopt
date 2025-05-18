/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/error.hpp>
#include <cuopt/routing/routing_structures.hpp>
#include <utilities/macros.cuh>

namespace cuopt {
namespace routing {

enum class request_t { PDP, VRP };
namespace detail {

/**
 * @brief Use encoded 32 bit integer with 2 bits for node type, 15 bits for
 * node and 15 bits for location
 * @note If we bump it up to 64 bits, we need to make sure to aligned shared memory access
 *
 * @tparam i_t
 */
template <typename i_t = int>
class __attribute__((aligned(4))) NodeInfo {
 public:
  constexpr NodeInfo() = default;

  constexpr NodeInfo(i_t node, i_t location, node_type_t node_type)
  {
    cuopt_assert(node < (1 << 17), "node id should be less than 131072");
    cuopt_assert(location < (1 << 15), "location id should be less than 32678");
    number_ = (uint32_t)node << 17 | (uint32_t)location << 2 | (uint32_t)node_type;

    cuopt_assert(is_valid(), "Corner case in NodeInfo struct!");
  }

  constexpr i_t node() const
  {
    // assert(is_valid());
    if (!is_valid()) {
      // FIXME:: This should not happen, but putting these guards for the release
      return 0;
    }
    return number_ >> 17;
  }

  constexpr i_t location() const
  {
    // assert(is_valid());
    if (!is_valid()) {
      // FIXME:: This should not happen, but putting these guards for the release
      return 0;
    }
    return (number_ >> 2) & ((1 << 15) - 1);
  }

  static auto get_string(node_type_t node_type)
  {
    std::string s;
    switch (node_type) {
      case node_type_t::DEPOT: s = "DEPOT"; break;
      case node_type_t::BREAK: s = "B"; break;
      case node_type_t::PICKUP: s = "P"; break;
      case node_type_t::DELIVERY: s = "D"; break;
    }
    return s;
  }

  constexpr node_type_t node_type() const
  {
    if (!is_valid()) {
      // FIXME:: This should not happen, but putting these guards for the release
      return node_type_t::DEPOT;
    }
    return node_type_t(number_ & ((1 << 2) - 1));
  }

  constexpr bool is_break() const { return node_type() == node_type_t::BREAK; }

  constexpr bool is_pickup() const { return node_type() == node_type_t::PICKUP; }
  constexpr bool is_delivery() const { return node_type() == node_type_t::DELIVERY; }

  constexpr bool is_depot() const { return node_type() == node_type_t::DEPOT; }

  constexpr i_t break_dim() const
  {
    if (!is_break()) { return -1; }
    return node();
  }

  constexpr bool is_service_node() const
  {
    auto ntype = node_type();

    return ntype == node_type_t::PICKUP || ntype == node_type_t::DELIVERY;
  }

  constexpr bool operator==(const NodeInfo& other) const { return number_ == other.number_; }
  constexpr bool operator!=(const NodeInfo& other) const { return number_ != other.number_; }

  constexpr bool is_valid() const { return number_ != std::numeric_limits<uint32_t>::max(); }

  constexpr uint32_t get_hash() const { return number_; }

 private:
  uint32_t number_ = std::numeric_limits<uint32_t>::max();
};

struct NodeInfoHash {
  size_t operator()(const NodeInfo<>& info) const { return info.get_hash(); }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
