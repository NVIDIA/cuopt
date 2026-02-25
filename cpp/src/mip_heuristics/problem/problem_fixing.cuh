/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace linear_programming::detail {

template <typename i_t, typename f_t>
struct problem_fixing_helpers_t {
  problem_fixing_helpers_t(i_t n_constraints, i_t n_variables, const raft::handle_t* handle_ptr)
    : reduction_in_rhs(n_constraints, handle_ptr->get_stream()),
      variable_fix_mask(n_variables, handle_ptr->get_stream())
  {
  }

  problem_fixing_helpers_t(const problem_fixing_helpers_t& other, const raft::handle_t* handle_ptr)
    : reduction_in_rhs(other.reduction_in_rhs, handle_ptr->get_stream()),
      variable_fix_mask(other.variable_fix_mask, handle_ptr->get_stream())
  {
  }

  rmm::device_uvector<f_t> reduction_in_rhs;
  rmm::device_uvector<i_t> variable_fix_mask;
};

}  // namespace linear_programming::detail
}  // namespace cuopt
