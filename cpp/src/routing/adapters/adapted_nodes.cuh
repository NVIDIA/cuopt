/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../structures.hpp"

namespace cuopt::routing::detail {

template <typename i_t, typename f_t>
struct adapted_node_t {
  //! Index in route
  size_t r_index{0};
  //! Route id
  size_t r_id{0};

  //! vehicle id
  size_t v_id{0};

  //! Global id of the node (as defined in adapted_problem_t<i_t, f_t>)
  NodeInfo<i_t> node_info;

  bool is_depot() const { return node_info.is_depot(); }

  int node_id() const { return node_info.node(); }
};

template <typename i_t, typename f_t>
bool operator==(const adapted_node_t<i_t, f_t>& lhs, const adapted_node_t<i_t, f_t>& rhs)
{
  bool equal = true;
  equal      = equal && (lhs.r_index == rhs.r_index);
  equal      = equal && (lhs.r_id == rhs.r_id);
  equal      = equal && (lhs.v_id == rhs.v_id);
  equal      = equal && (lhs.node_info == rhs.node_info);
  return equal;
}

template struct adapted_node_t<int, float>;
}  // namespace cuopt::routing::detail
