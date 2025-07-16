/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <linear_programming/utils.cuh>
#include <mip/presolve/trivial_presolve_helpers.cuh>
#include <mip/problem/problem.cuh>
#include <utilities/copy_helpers.hpp>

#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/uninitialized_fill.h>
#include <cuda/std/functional>

#include <unordered_set>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
void test_renumbered_coo(raft::device_span<i_t> coo_major, const problem_t<i_t, f_t>& pb)
{
  auto handle_ptr = pb.handle_ptr;
  auto h_coo      = cuopt::host_copy(coo_major, handle_ptr->get_stream());

  for (i_t i = 0; i < (i_t)h_coo.size() - 1; ++i) {
    cuopt_assert((h_coo[i + 1] - h_coo[i]) <= 1, "renumbering error");
  }
}

template <typename i_t, typename f_t>
void cleanup_vectors(problem_t<i_t, f_t>& pb,
                     const rmm::device_uvector<i_t>& cnst_map,
                     const rmm::device_uvector<i_t>& var_map)
{
  auto handle_ptr   = pb.handle_ptr;
  auto cnst_lb_iter = thrust::remove_if(handle_ptr->get_thrust_policy(),
                                        pb.constraint_lower_bounds.begin(),
                                        pb.constraint_lower_bounds.end(),
                                        cnst_map.begin(),
                                        is_zero_t<i_t>{});
  auto cnst_ub_iter = thrust::remove_if(handle_ptr->get_thrust_policy(),
                                        pb.constraint_upper_bounds.begin(),
                                        pb.constraint_upper_bounds.end(),
                                        cnst_map.begin(),
                                        is_zero_t<i_t>{});
  handle_ptr->sync_stream();
  pb.constraint_lower_bounds.resize(cnst_lb_iter - pb.constraint_lower_bounds.begin(),
                                    handle_ptr->get_stream());
  pb.constraint_upper_bounds.resize(cnst_ub_iter - pb.constraint_upper_bounds.begin(),
                                    handle_ptr->get_stream());

  handle_ptr->sync_stream();
  auto lb_iter     = thrust::remove_if(handle_ptr->get_thrust_policy(),
                                   pb.variable_lower_bounds.begin(),
                                   pb.variable_lower_bounds.end(),
                                   var_map.begin(),
                                   is_zero_t<i_t>{});
  auto ub_iter     = thrust::remove_if(handle_ptr->get_thrust_policy(),
                                   pb.variable_upper_bounds.begin(),
                                   pb.variable_upper_bounds.end(),
                                   var_map.begin(),
                                   is_zero_t<i_t>{});
  auto type_iter   = thrust::remove_if(handle_ptr->get_thrust_policy(),
                                     pb.variable_types.begin(),
                                     pb.variable_types.end(),
                                     var_map.begin(),
                                     is_zero_t<i_t>{});
  auto binary_iter = thrust::remove_if(handle_ptr->get_thrust_policy(),
                                       pb.is_binary_variable.begin(),
                                       pb.is_binary_variable.end(),
                                       var_map.begin(),
                                       is_zero_t<i_t>{});
  auto obj_iter    = thrust::remove_if(handle_ptr->get_thrust_policy(),
                                    pb.objective_coefficients.begin(),
                                    pb.objective_coefficients.end(),
                                    var_map.begin(),
                                    is_zero_t<i_t>{});
  pb.variable_lower_bounds.resize(lb_iter - pb.variable_lower_bounds.begin(),
                                  handle_ptr->get_stream());
  pb.variable_upper_bounds.resize(ub_iter - pb.variable_upper_bounds.begin(),
                                  handle_ptr->get_stream());
  pb.variable_types.resize(type_iter - pb.variable_types.begin(), handle_ptr->get_stream());
  pb.is_binary_variable.resize(binary_iter - pb.is_binary_variable.begin(),
                               handle_ptr->get_stream());
  pb.objective_coefficients.resize(obj_iter - pb.objective_coefficients.begin(),
                                   handle_ptr->get_stream());
  handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
void update_from_csr(problem_t<i_t, f_t>& pb)
{
  auto handle_ptr = pb.handle_ptr;
  rmm::device_uvector<i_t> cnst(pb.coefficients.size(), handle_ptr->get_stream());
  thrust::uninitialized_fill(handle_ptr->get_thrust_policy(), cnst.begin(), cnst.end(), 0);

  //  csr to coo
  thrust::scatter_if(handle_ptr->get_thrust_policy(),
                     thrust::counting_iterator<i_t>(0),
                     thrust::counting_iterator<i_t>(pb.offsets.size() - 1),
                     pb.offsets.begin(),
                     thrust::counting_iterator<i_t>(0),
                     cnst.begin(),
                     non_zero_degree_t{make_span(pb.offsets)});
  thrust::inclusive_scan(handle_ptr->get_thrust_policy(),
                         cnst.begin(),
                         cnst.end(),
                         cnst.begin(),
                         thrust::maximum<i_t>{});
  RAFT_CHECK_CUDA(handle_ptr->get_stream());

  //  partition coo - fixed variables reside in second partition
  i_t nnz_edge_count = pb.coefficients.size();
  {
    auto coo_begin = thrust::make_zip_iterator(
      thrust::make_tuple(cnst.begin(), pb.coefficients.begin(), pb.variables.begin()));
    auto partition_iter =
      thrust::stable_partition(handle_ptr->get_thrust_policy(),
                               coo_begin,
                               coo_begin + cnst.size(),
                               is_variable_free_t<f_t>{pb.tolerances.integrality_tolerance,
                                                       make_span(pb.variable_lower_bounds),
                                                       make_span(pb.variable_upper_bounds)});
    RAFT_CHECK_CUDA(handle_ptr->get_stream());
    nnz_edge_count = partition_iter - coo_begin;
  }

  //  maps to denote active constraints and non-fixed variables
  rmm::device_uvector<i_t> cnst_map(pb.n_constraints, handle_ptr->get_stream());
  rmm::device_uvector<i_t> var_map(pb.n_variables, handle_ptr->get_stream());
  thrust::uninitialized_fill(handle_ptr->get_thrust_policy(), cnst_map.begin(), cnst_map.end(), 0);
  thrust::uninitialized_fill(handle_ptr->get_thrust_policy(), var_map.begin(), var_map.end(), 0);
  // maps to denote active constraints and non-fixed variables
  thrust::scatter(handle_ptr->get_thrust_policy(),
                  thrust::make_constant_iterator<i_t>(1),
                  thrust::make_constant_iterator<i_t>(1) + nnz_edge_count,
                  cnst.begin(),
                  cnst_map.begin());
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
  thrust::scatter(handle_ptr->get_thrust_policy(),
                  thrust::make_constant_iterator<i_t>(1),
                  thrust::make_constant_iterator<i_t>(1) + nnz_edge_count,
                  pb.variables.begin(),
                  var_map.begin());
  RAFT_CHECK_CUDA(handle_ptr->get_stream());

  auto unused_var_count =
    thrust::count(handle_ptr->get_thrust_policy(), var_map.begin(), var_map.end(), 0);
  if (unused_var_count > 0) {
    CUOPT_LOG_INFO("Unused variables detected, eliminating them! Unused var count %d",
                   unused_var_count);
    thrust::for_each(
      handle_ptr->get_thrust_policy(),
      thrust::make_counting_iterator<i_t>(0),
      thrust::make_counting_iterator<i_t>(pb.n_variables),
      assign_fixed_var_t<i_t, f_t>{make_span(var_map),
                                   make_span(pb.variable_lower_bounds),
                                   make_span(pb.variable_upper_bounds),
                                   make_span(pb.objective_coefficients),
                                   make_span(pb.presolve_data.variable_mapping),
                                   make_span(pb.presolve_data.fixed_var_assignment)});
    auto used_iter = thrust::stable_partition(handle_ptr->get_thrust_policy(),
                                              pb.presolve_data.variable_mapping.begin(),
                                              pb.presolve_data.variable_mapping.end(),
                                              var_map.begin(),
                                              cuda::std::identity{});
    pb.presolve_data.variable_mapping.resize(used_iter - pb.presolve_data.variable_mapping.begin(),
                                             handle_ptr->get_stream());

    RAFT_CHECK_CUDA(handle_ptr->get_stream());
  }

  if (nnz_edge_count != static_cast<i_t>(pb.coefficients.size())) {
    //   Calculate updates to constraint bounds affected by fixed variables
    rmm::device_uvector<i_t> unused_coo_cnst(cnst.size() - nnz_edge_count,
                                             handle_ptr->get_stream());
    rmm::device_uvector<f_t> unused_coo_cnst_bound_updates(cnst.size() - nnz_edge_count,
                                                           handle_ptr->get_stream());
    elem_multi_t<i_t, f_t> mul{make_span(pb.coefficients),
                               make_span(pb.variables),
                               make_span(pb.objective_coefficients),
                               make_span(pb.variable_lower_bounds),
                               make_span(pb.variable_upper_bounds)};

    auto iter = thrust::reduce_by_key(
      handle_ptr->get_thrust_policy(),
      cnst.begin() + nnz_edge_count,
      cnst.end(),
      thrust::make_transform_iterator(thrust::make_counting_iterator<i_t>(nnz_edge_count), mul),
      unused_coo_cnst.begin(),
      unused_coo_cnst_bound_updates.begin());
    RAFT_CHECK_CUDA(handle_ptr->get_stream());
    auto unused_coo_cnst_count = iter.first - unused_coo_cnst.begin();
    unused_coo_cnst.resize(unused_coo_cnst_count, handle_ptr->get_stream());
    unused_coo_cnst_bound_updates.resize(unused_coo_cnst_count, handle_ptr->get_stream());

    //  update constraint bounds using fixed variables
    thrust::for_each(handle_ptr->get_thrust_policy(),
                     thrust::make_counting_iterator<i_t>(0),
                     thrust::make_counting_iterator<i_t>(unused_coo_cnst.size()),
                     update_constraint_bounds_t<i_t, f_t>{make_span(unused_coo_cnst),
                                                          make_span(unused_coo_cnst_bound_updates),
                                                          make_span(pb.constraint_lower_bounds),
                                                          make_span(pb.constraint_upper_bounds)});
    RAFT_CHECK_CUDA(handle_ptr->get_stream());
  }

  //  update objective_offset
  pb.presolve_data.objective_offset +=
    thrust::transform_reduce(handle_ptr->get_thrust_policy(),
                             thrust::counting_iterator<i_t>(0),
                             thrust::counting_iterator<i_t>(pb.n_variables),
                             unused_var_obj_offset_t<i_t, f_t>{make_span(var_map),
                                                               make_span(pb.objective_coefficients),
                                                               make_span(pb.variable_lower_bounds),
                                                               make_span(pb.variable_upper_bounds)},
                             0.,
                             thrust::plus<f_t>{});
  RAFT_CHECK_CUDA(handle_ptr->get_stream());

  //  create renumbering maps
  rmm::device_uvector<i_t> cnst_renum_ids(pb.n_constraints, handle_ptr->get_stream());
  rmm::device_uvector<i_t> var_renum_ids(pb.n_variables, handle_ptr->get_stream());
  thrust::inclusive_scan(
    handle_ptr->get_thrust_policy(),
    cnst_map.begin(),
    cnst_map.end(),
    thrust::make_transform_output_iterator(cnst_renum_ids.begin(), sub_t<i_t>{}));
  thrust::inclusive_scan(
    handle_ptr->get_thrust_policy(),
    var_map.begin(),
    var_map.end(),
    thrust::make_transform_output_iterator(var_renum_ids.begin(), sub_t<i_t>{}));
  //  renumber coo
  thrust::transform(handle_ptr->get_thrust_policy(),
                    cnst.begin(),
                    cnst.begin() + nnz_edge_count,
                    cnst.begin(),
                    apply_renumbering_t{make_span(cnst_renum_ids)});
  thrust::transform(handle_ptr->get_thrust_policy(),
                    pb.variables.begin(),
                    pb.variables.begin() + nnz_edge_count,
                    pb.variables.begin(),
                    apply_renumbering_t{make_span(var_renum_ids)});

  cuopt_func_call(test_renumbered_coo(make_span(cnst, 0, nnz_edge_count), pb));

  auto updated_n_cnst = 1 + cnst_renum_ids.back_element(handle_ptr->get_stream());
  auto updated_n_vars = 1 + var_renum_ids.back_element(handle_ptr->get_stream());

  pb.n_constraints = updated_n_cnst;
  pb.n_variables   = updated_n_vars;
  CUOPT_LOG_INFO("After trivial presolve updated %d constraints %d variables. Objective offset %f",
                 updated_n_cnst,
                 updated_n_vars,
                 pb.presolve_data.objective_offset);
  // check successive cnst in coo increases by atmost 1
  // update csr offset
  pb.offsets.resize(pb.n_constraints + 1, handle_ptr->get_stream());
  thrust::fill(handle_ptr->get_thrust_policy(), pb.offsets.begin(), pb.offsets.end(), 0);
  thrust::for_each(handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator<i_t>(0),
                   thrust::make_counting_iterator<i_t>(nnz_edge_count),
                   coo_to_offset_t{make_span(cnst, 0, nnz_edge_count), make_span(pb.offsets)});

  // clean up vectors
  cleanup_vectors(pb, cnst_map, var_map);

  //  reorder coo by var
  rmm::device_uvector<i_t> coo_variables(nnz_edge_count, handle_ptr->get_stream());
  raft::copy(coo_variables.data(), pb.variables.data(), nnz_edge_count, handle_ptr->get_stream());

  pb.reverse_constraints.resize(nnz_edge_count, handle_ptr->get_stream());
  raft::copy(pb.reverse_constraints.data(), cnst.data(), nnz_edge_count, handle_ptr->get_stream());

  pb.reverse_coefficients.resize(nnz_edge_count, handle_ptr->get_stream());
  raft::copy(pb.reverse_coefficients.data(),
             pb.coefficients.data(),
             nnz_edge_count,
             handle_ptr->get_stream());

  pb.variables.resize(nnz_edge_count, handle_ptr->get_stream());
  pb.coefficients.resize(nnz_edge_count, handle_ptr->get_stream());

  auto coo_begin = thrust::make_zip_iterator(
    thrust::make_tuple(pb.reverse_constraints.begin(), pb.reverse_coefficients.begin()));
  thrust::sort_by_key(
    handle_ptr->get_thrust_policy(), coo_variables.begin(), coo_variables.end(), coo_begin);

  //  update csc offset
  pb.reverse_offsets.resize(pb.n_variables + 1, handle_ptr->get_stream());
  thrust::fill(
    handle_ptr->get_thrust_policy(), pb.reverse_offsets.begin(), pb.reverse_offsets.end(), 0);
  thrust::for_each(handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator<i_t>(0),
                   thrust::make_counting_iterator<i_t>(nnz_edge_count),
                   coo_to_offset_t{make_span(coo_variables), make_span(pb.reverse_offsets)});
  pb.nnz = nnz_edge_count;
}

template <typename i_t, typename f_t>
void test_reverse_matches(const problem_t<i_t, f_t>& pb)
{
  auto h_offsets              = cuopt::host_copy(pb.offsets);
  auto h_coefficients         = cuopt::host_copy(pb.coefficients);
  auto h_variables            = cuopt::host_copy(pb.variables);
  auto h_reverse_offsets      = cuopt::host_copy(pb.reverse_offsets);
  auto h_reverse_constraints  = cuopt::host_copy(pb.reverse_constraints);
  auto h_reverse_coefficients = cuopt::host_copy(pb.reverse_coefficients);

  std::vector<std::unordered_set<i_t>> vars_per_constr(pb.n_constraints);
  std::vector<std::unordered_set<f_t>> coeff_per_constr(pb.n_constraints);
  for (i_t i = 0; i < (i_t)h_offsets.size() - 1; ++i) {
    for (i_t c = h_offsets[i]; c < h_offsets[i + 1]; c++) {
      vars_per_constr[i].insert(h_variables[c]);
      coeff_per_constr[i].insert(h_coefficients[c]);
    }
  }

  for (i_t i = 0; i < (i_t)h_reverse_offsets.size() - 1; ++i) {
    for (i_t v = h_reverse_offsets[i]; v < h_reverse_offsets[i + 1]; v++) {
      cuopt_assert(vars_per_constr[h_reverse_constraints[v]].count(i) != 0,
                   "Constraint var mismatch");
      cuopt_assert(coeff_per_constr[h_reverse_constraints[v]].count(h_reverse_coefficients[v]) != 0,
                   "Constraint var mismatch");
    }
  }
}

template <typename i_t, typename f_t>
void trivial_presolve(problem_t<i_t, f_t>& problem)
{
  cuopt_expects(problem.preprocess_called,
                error_type_t::RuntimeError,
                "preprocess_problem should be called before running the solver");
  update_from_csr(problem);
  problem.recompute_auxilliary_data(
    false);  // check problem representation later once cstr bounds are computed
  cuopt_func_call(test_reverse_matches(problem));
  combine_constraint_bounds<i_t, f_t>(problem, problem.combined_bounds);
  // The problem has been solved by presolve. Mark its empty status as valid
  if (problem.n_variables == 0) { problem.empty = true; }
  problem.check_problem_representation(true);
}

}  // namespace cuopt::linear_programming::detail
