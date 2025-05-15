/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <mip/presolve/load_balanced_bounds_presolve.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct lb_bounds_update_data_t {
  rmm::device_scalar<i_t> bounds_changed;
  rmm::device_uvector<f_t> cnst_slack;
  rmm::device_uvector<f_t> vars_bnd;
  rmm::device_uvector<f_t> tmp_cnst_slack;
  rmm::device_uvector<f_t> tmp_vars_bnd;
  rmm::device_uvector<i_t> var_bounds_changed;
  rmm::device_uvector<i_t> changed_constraints;
  rmm::device_uvector<i_t> next_changed_constraints;
  rmm::device_uvector<i_t> changed_variables;

  struct view_t {
    using f_t2 = typename type_2<f_t>::type;
    i_t* bounds_changed;
    raft::device_span<f_t2> cnst_slack;
    raft::device_span<f_t2> vars_bnd;
    raft::device_span<f_t2> tmp_cnst_slack;
    raft::device_span<f_t2> tmp_vars_bnd;
    raft::device_span<i_t> var_bounds_changed;
    raft::device_span<i_t> changed_constraints;
    raft::device_span<i_t> next_changed_constraints;
    raft::device_span<i_t> changed_variables;
  };

  lb_bounds_update_data_t(const raft::handle_t* handle);
  // void resize(load_balanced_bounds_presolve_t<i_t, f_t>& prs);
  void resize(const raft::handle_t* handle_ptr,
              i_t n_constraints,
              i_t n_variables,
              i_t num_blocks_heavy_cnst,
              i_t num_blocks_heavy_vars);
  void init_changed_constraints(const raft::handle_t* handle_ptr);
  void prepare_for_next_iteration(const raft::handle_t* handle_ptr);
  view_t view();
};

}  // namespace cuopt::linear_programming::detail
