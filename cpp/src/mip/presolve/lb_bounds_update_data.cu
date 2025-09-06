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
#include <mip/presolve/lb_problem.cuh>
#include <mip/presolve/load_balanced_partition_helpers.cuh>

#include <utilities/copy_helpers.hpp>
#include "lb_bounds_update_data.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
lb_bounds_update_data_t<i_t, f_t>::lb_bounds_update_data_t(lb_problem_t<i_t, f_t>& problem)
  : bounds_changed(problem.handle_ptr->get_stream()),
    cnst_slack(0, problem.handle_ptr->get_stream()),
    vars_bnd(0, problem.handle_ptr->get_stream()),
    tmp_act(0, problem.handle_ptr->get_stream()),
    changed_constraints(0, problem.handle_ptr->get_stream()),
    next_changed_constraints(0, problem.handle_ptr->get_stream()),
    changed_variables(0, problem.handle_ptr->get_stream()),
    heavy_bounds_changed_agg(0, problem.handle_ptr->get_stream()),
    heavy_bounds_changed(0, problem.handle_ptr->get_stream())
{
  resize(problem);
}

template <typename i_t, typename f_t>
void lb_bounds_update_data_t<i_t, f_t>::copy(lb_problem_t<i_t, f_t>& problem)
{
  // TODO : remove resize?
  resize(problem);
  raft::copy(
    vars_bnd.data(), problem.vars_bnd.data(), vars_bnd.size(), problem.handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void lb_bounds_update_data_t<i_t, f_t>::resize(lb_problem_t<i_t, f_t>& problem)
{
  auto num_heavy_vars = problem.n_variables - problem.vars_csr.heavy_beg_id;
  resize(problem.handle_ptr,
         problem.n_constraints,
         problem.n_variables,
         problem.cnst_csr.num_blocks_heavy,
         problem.vars_csr.num_blocks_heavy,
         num_heavy_vars);
}

template <typename i_t, typename f_t>
void lb_bounds_update_data_t<i_t, f_t>::resize(const raft::handle_t* handle_ptr,
                                               i_t n_constraints,
                                               i_t n_variables,
                                               i_t num_blocks_heavy_cnst,
                                               i_t num_blocks_heavy_vars,
                                               i_t num_heavy_vars)
{
  cnst_slack.resize(n_constraints, handle_ptr->get_stream());
  tmp_act.resize(num_blocks_heavy_cnst, handle_ptr->get_stream());
  vars_bnd.resize(n_variables, handle_ptr->get_stream());

  changed_constraints.resize(n_constraints, handle_ptr->get_stream());
  next_changed_constraints.resize(n_constraints, handle_ptr->get_stream());
  changed_variables.resize(n_variables, handle_ptr->get_stream());

  heavy_bounds_changed_agg.resize(num_heavy_vars, handle_ptr->get_stream());
  heavy_bounds_changed.resize(num_heavy_vars, handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
typename lb_bounds_update_data_t<i_t, f_t>::view_t lb_bounds_update_data_t<i_t, f_t>::view()
{
  view_t v;
  v.bounds_changed           = bounds_changed.data();
  v.cnst_slack               = make_span(cnst_slack);
  v.vars_bnd                 = make_span(vars_bnd);
  v.tmp_act                  = make_span(tmp_act);
  v.changed_constraints      = make_span(changed_constraints);
  v.next_changed_constraints = make_span(next_changed_constraints);
  v.changed_variables        = make_span(changed_variables);

  v.heavy_bounds_changed_agg = make_span(heavy_bounds_changed_agg);
  v.heavy_bounds_changed     = make_span(heavy_bounds_changed);
  return v;
}

template <typename i_t, typename f_t>
void lb_bounds_update_data_t<i_t, f_t>::init_changed_constraints(const raft::handle_t* handle_ptr)
{
  thrust::fill(
    handle_ptr->get_thrust_policy(), changed_variables.begin(), changed_variables.end(), 1);
  thrust::fill(
    handle_ptr->get_thrust_policy(), changed_constraints.begin(), changed_constraints.end(), 1);
  thrust::fill(handle_ptr->get_thrust_policy(),
               next_changed_constraints.begin(),
               next_changed_constraints.end(),
               0);
  thrust::fill(handle_ptr->get_thrust_policy(),
               heavy_bounds_changed_agg.begin(),
               heavy_bounds_changed_agg.end(),
               0);
  thrust::fill(
    handle_ptr->get_thrust_policy(), heavy_bounds_changed.begin(), heavy_bounds_changed.end(), 0);
}

template <typename i_t, typename f_t>
void lb_bounds_update_data_t<i_t, f_t>::disable_changed_constraints(
  const raft::handle_t* handle_ptr)
{
  thrust::fill(
    handle_ptr->get_thrust_policy(), changed_variables.begin(), changed_variables.end(), 1);
  thrust::fill(
    handle_ptr->get_thrust_policy(), changed_constraints.begin(), changed_constraints.end(), 0);
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
  thrust::fill(handle_ptr->get_thrust_policy(),
               heavy_bounds_changed_agg.begin(),
               heavy_bounds_changed_agg.end(),
               0);
  thrust::fill(
    handle_ptr->get_thrust_policy(), heavy_bounds_changed.begin(), heavy_bounds_changed.end(), 0);
}

#if MIP_INSTANTIATE_FLOAT
template class lb_bounds_update_data_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class lb_bounds_update_data_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
