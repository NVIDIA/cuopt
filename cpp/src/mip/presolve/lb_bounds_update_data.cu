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

#include <mip/mip_constants.hpp>
#include <mip/presolve/load_balanced_partition_helpers.cuh>

#include <utilities/copy_helpers.hpp>
#include "lb_bounds_update_data.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
lb_bounds_update_data_t<i_t, f_t>::lb_bounds_update_data_t(const raft::handle_t* handle_ptr)
  : bounds_changed(handle_ptr->get_stream()),
    cnst_slack(0, handle_ptr->get_stream()),
    vars_bnd(0, handle_ptr->get_stream()),
    tmp_cnst_slack(0, handle_ptr->get_stream()),
    tmp_vars_bnd(0, handle_ptr->get_stream()),
    var_bounds_changed(0, handle_ptr->get_stream()),
    changed_constraints(0, handle_ptr->get_stream()),
    next_changed_constraints(0, handle_ptr->get_stream()),
    changed_variables(0, handle_ptr->get_stream())
{
}

template <typename i_t, typename f_t>
void lb_bounds_update_data_t<i_t, f_t>::resize(const raft::handle_t* handle_ptr,
                                               i_t n_constraints,
                                               i_t n_variables,
                                               i_t num_blocks_heavy_cnst,
                                               i_t num_blocks_heavy_vars)
{
  cnst_slack.resize(2 * n_constraints, handle_ptr->get_stream());
  tmp_cnst_slack.resize(2 * num_blocks_heavy_cnst, handle_ptr->get_stream());
  vars_bnd.resize(2 * n_variables, handle_ptr->get_stream());
  tmp_vars_bnd.resize(2 * num_blocks_heavy_vars, handle_ptr->get_stream());

  var_bounds_changed.resize(n_variables, handle_ptr->get_stream());
  changed_constraints.resize(n_constraints, handle_ptr->get_stream());
  next_changed_constraints.resize(n_constraints, handle_ptr->get_stream());
  changed_variables.resize(n_variables, handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
typename lb_bounds_update_data_t<i_t, f_t>::view_t lb_bounds_update_data_t<i_t, f_t>::view()
{
  view_t v;
  v.bounds_changed           = bounds_changed.data();
  v.cnst_slack               = make_span_2(cnst_slack);
  v.vars_bnd                 = make_span_2(vars_bnd);
  v.tmp_cnst_slack           = make_span_2(cnst_slack);
  v.tmp_vars_bnd             = make_span_2(vars_bnd);
  v.var_bounds_changed       = make_span(var_bounds_changed);
  v.changed_constraints      = make_span(changed_constraints);
  v.next_changed_constraints = make_span(next_changed_constraints);
  v.changed_variables        = make_span(changed_variables);
  return v;
}

template <typename i_t, typename f_t>
void lb_bounds_update_data_t<i_t, f_t>::init_changed_constraints(const raft::handle_t* handle_ptr)
{
  thrust::fill(
    handle_ptr->get_thrust_policy(), var_bounds_changed.begin(), var_bounds_changed.end(), 0);
  thrust::fill(
    handle_ptr->get_thrust_policy(), changed_variables.begin(), changed_variables.end(), 1);
  thrust::fill(
    handle_ptr->get_thrust_policy(), changed_constraints.begin(), changed_constraints.end(), 1);
  thrust::fill(handle_ptr->get_thrust_policy(),
               next_changed_constraints.begin(),
               next_changed_constraints.end(),
               0);
}

template <typename i_t, typename f_t>
void lb_bounds_update_data_t<i_t, f_t>::prepare_for_next_iteration(const raft::handle_t* handle_ptr)
{
  std::swap(changed_constraints, next_changed_constraints);
  handle_ptr->sync_stream();
  thrust::fill(handle_ptr->get_thrust_policy(),
               next_changed_constraints.begin(),
               next_changed_constraints.end(),
               0);
  thrust::fill(
    handle_ptr->get_thrust_policy(), changed_variables.begin(), changed_variables.end(), 0);
  thrust::fill(
    handle_ptr->get_thrust_policy(), var_bounds_changed.begin(), var_bounds_changed.end(), 0);
}

#if MIP_INSTANTIATE_FLOAT
template class lb_bounds_update_data_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class lb_bounds_update_data_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
