/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <mip/problem/problem.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct bounds_update_data_t {
  rmm::device_scalar<i_t> bounds_changed;
  rmm::device_uvector<f_t> min_activity;
  rmm::device_uvector<f_t> max_activity;
  rmm::device_uvector<f_t> lb;
  rmm::device_uvector<f_t> ub;
  rmm::device_uvector<i_t> changed_constraints;
  rmm::device_uvector<i_t> next_changed_constraints;
  rmm::device_uvector<i_t> changed_variables;

  struct view_t {
    i_t* bounds_changed;
    raft::device_span<f_t> min_activity;
    raft::device_span<f_t> max_activity;
    raft::device_span<f_t> lb;
    raft::device_span<f_t> ub;
    raft::device_span<i_t> changed_constraints;
    raft::device_span<i_t> next_changed_constraints;
    raft::device_span<i_t> changed_variables;
  };

  bounds_update_data_t(problem_t<i_t, f_t>& pb);
  void resize(problem_t<i_t, f_t>& problem);
  void init_changed_constraints(const raft::handle_t* handle_ptr);
  void prepare_for_next_iteration(const raft::handle_t* handle_ptr);
  view_t view();
};

}  // namespace cuopt::linear_programming::detail
