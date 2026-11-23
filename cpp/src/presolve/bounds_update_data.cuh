/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <mip_heuristics/problem/problem.cuh>
#include <utilities/type_2.hpp>

#include <limits>

namespace cuopt::mathematical_optimization::mip {

template <typename f_t2>
inline __host__ __device__ auto min_activity_value(const f_t2& activity) -> decltype(activity.x)
{
  using f_t = decltype(activity.x);
  return activity.y > 0 ? -std::numeric_limits<f_t>::infinity() : activity.x;
}

template <typename f_t2>
inline __host__ __device__ auto max_activity_value(const f_t2& activity) -> decltype(activity.x)
{
  using f_t = decltype(activity.x);
  return activity.y > 0 ? std::numeric_limits<f_t>::infinity() : activity.x;
}

template <typename i_t, typename f_t>
struct bounds_update_data_t {
  using f_t2 = typename type_2<f_t>::type;

  rmm::device_scalar<i_t> bounds_changed;
  rmm::device_uvector<f_t2> min_activity;
  rmm::device_uvector<f_t2> max_activity;
  rmm::device_uvector<f_t> lb;
  rmm::device_uvector<f_t> ub;
  rmm::device_uvector<i_t> changed_constraints;
  rmm::device_uvector<i_t> next_changed_constraints;
  rmm::device_uvector<i_t> changed_variables;
  f_t candidate_bound_scale;

  struct view_t {
    i_t* bounds_changed;
    raft::device_span<f_t2> min_activity;
    raft::device_span<f_t2> max_activity;
    raft::device_span<f_t> lb;
    raft::device_span<f_t> ub;
    raft::device_span<i_t> changed_constraints;
    raft::device_span<i_t> next_changed_constraints;
    raft::device_span<i_t> changed_variables;
    f_t candidate_bound_scale;
  };

  bounds_update_data_t(problem_t<i_t, f_t>& pb);
  void resize(problem_t<i_t, f_t>& problem);
  void init_changed_constraints(const raft::handle_t* handle_ptr);
  void prepare_for_next_iteration(const raft::handle_t* handle_ptr);
  view_t view();
};

}  // namespace cuopt::mathematical_optimization::mip
