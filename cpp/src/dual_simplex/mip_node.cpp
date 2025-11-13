/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/mip_node.hpp>

namespace cuopt::linear_programming::dual_simplex {

bool inactive_status(node_status_t status)
{
  return (status == node_status_t::FATHOMED || status == node_status_t::INTEGER_FEASIBLE ||
          status == node_status_t::INFEASIBLE || status == node_status_t::NUMERICAL);
}

}  // namespace cuopt::linear_programming::dual_simplex
