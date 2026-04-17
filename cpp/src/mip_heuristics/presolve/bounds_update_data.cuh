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

  // Per-(constraint, clique-group) dynamic state for clique-aware activity
  // tightening. These arrays are sized to the number of groups in the static
  // clique group table (see clique_group_table_t). They are empty when
  // clique-aware tightening is not enabled for this problem. Recomputed every
  // bound-propagation iteration from (lb, ub) by compute_clique_corrections_kernel.
  rmm::device_uvector<f_t> group_max_correction;
  rmm::device_uvector<f_t> group_min_correction;
  rmm::device_uvector<f_t> group_max_pos;
  rmm::device_uvector<f_t> group_second_max_pos;
  rmm::device_uvector<f_t> group_min_neg;
  rmm::device_uvector<f_t> group_second_min_neg;

  struct view_t {
    i_t* bounds_changed;
    raft::device_span<f_t> min_activity;
    raft::device_span<f_t> max_activity;
    raft::device_span<f_t> lb;
    raft::device_span<f_t> ub;
    raft::device_span<i_t> changed_constraints;
    raft::device_span<i_t> next_changed_constraints;
    raft::device_span<i_t> changed_variables;

    raft::device_span<f_t> group_max_correction;
    raft::device_span<f_t> group_min_correction;
    raft::device_span<f_t> group_max_pos;
    raft::device_span<f_t> group_second_max_pos;
    raft::device_span<f_t> group_min_neg;
    raft::device_span<f_t> group_second_min_neg;
  };

  bounds_update_data_t(problem_t<i_t, f_t>& pb);
  void resize(problem_t<i_t, f_t>& problem);
  // Size (or shrink) the 6 clique-correction buffers to n_groups. Called from
  // the orchestrator after clique_group_table_t::build_from_host determines
  // how many groups exist. No-op if already sized correctly.
  void resize_clique_buffers(i_t n_groups, rmm::cuda_stream_view stream);
  void init_changed_constraints(const raft::handle_t* handle_ptr);
  void prepare_for_next_iteration(const raft::handle_t* handle_ptr);
  view_t view();
};

}  // namespace cuopt::linear_programming::detail
