/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <mip/problem/problem.cuh>

#include <utilities/copy_helpers.hpp>
#include <utilities/macros.cuh>

#include <linear_programming/utils.cuh>

#include <raft/core/logger.hpp>

#include "load_balanced_problem.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t>
std::tuple<i_t, i_t, i_t> bin_meta(std::vector<i_t>& bins, i_t deg_beg, i_t deg_end)
{
  auto beg = ceil_log_2(deg_beg);
  auto end = ceil_log_2(deg_end) + 1;
  return std::make_tuple(beg, end, bins[end] - bins[beg]);
}

template <typename i_t>
std::tuple<i_t, i_t, i_t, i_t> v_bin_meta(std::vector<i_t>& bins, i_t deg_beg, i_t deg_end)
{
  i_t warp_total = 0;
  for (i_t deg = deg_beg; deg <= deg_end; deg = deg * 2) {
    auto beg = ceil_log_2(deg);
    auto end = ceil_log_2(deg) + 1;
    warp_total += (31 + (bins[end] - bins[beg]) * deg) / 32;
  }
  auto beg = ceil_log_2(deg_beg);
  auto end = ceil_log_2(deg_end) + 1;
  return std::make_tuple(beg, end, bins[end] - bins[beg], warp_total);
}

template <typename i_t, typename f_t, typename f_t2>
__global__ void constraint_data_copy(raft::device_span<i_t> reorg_ids,
                                     raft::device_span<i_t> offsets,
                                     raft::device_span<f_t> coeff,
                                     raft::device_span<i_t> edge,
                                     raft::device_span<f_t2> bounds,
                                     typename problem_t<i_t, f_t>::view_t pb)
{
  i_t new_idx = blockIdx.x;
  i_t idx     = reorg_ids[new_idx];

  auto read_beg = pb.offsets[idx];
  auto read_end = pb.offsets[idx + 1];

  auto write_beg = offsets[new_idx];

  if (threadIdx.x == 0) {
    // new indexing to improve sub warp property access
    bounds[new_idx] = f_t2{pb.constraint_lower_bounds[idx], pb.constraint_upper_bounds[idx]};
  }
  for (i_t i = threadIdx.x; i < (read_end - read_beg); i += blockDim.x) {
    coeff[i + write_beg] = pb.coefficients[i + read_beg];
    edge[i + write_beg]  = pb.variables[i + read_beg];
  }
}

template <typename i_t, typename f_t, typename f_t2>
__global__ void variable_data_copy(raft::device_span<i_t> reorg_ids,
                                   raft::device_span<i_t> offsets,
                                   raft::device_span<f_t> coeff,
                                   raft::device_span<i_t> edge,
                                   raft::device_span<f_t2> bounds,
                                   raft::device_span<var_t> var_types,
                                   typename problem_t<i_t, f_t>::view_t pb)
{
  i_t new_idx = blockIdx.x;
  i_t idx     = reorg_ids[new_idx];

  auto read_beg = pb.reverse_offsets[idx];
  auto read_end = pb.reverse_offsets[idx + 1];

  auto write_beg = offsets[new_idx];

  if (threadIdx.x == 0) {
    // old indexing to match original bounds presolve random access performance
    bounds[idx] = f_t2{pb.variable_lower_bounds[idx], pb.variable_upper_bounds[idx]};
    // new indexing to improve sub warp property access
    var_types[new_idx] = pb.variable_types[idx];
  }
  for (i_t i = threadIdx.x; i < (read_end - read_beg); i += blockDim.x) {
    coeff[i + write_beg] = pb.reverse_coefficients[i + read_beg];
    edge[i + write_beg]  = pb.reverse_constraints[i + read_beg];
  }
}

template <typename i_t, typename f_t, typename f_t2>
__global__ void check_constraint_data(raft::device_span<i_t> reorg_ids,
                                      raft::device_span<i_t> offsets,
                                      raft::device_span<f_t> coeff,
                                      raft::device_span<i_t> edge,
                                      raft::device_span<f_t2> bounds,
                                      typename problem_t<i_t, f_t>::view_t pb,
                                      i_t* errors)
{
  i_t new_idx = blockIdx.x;
  i_t idx     = reorg_ids[new_idx];

  auto src_read_beg = pb.offsets[idx];
  auto src_read_end = pb.offsets[idx + 1];

  auto dst_read_beg = offsets[new_idx];
  auto bnd          = bounds[new_idx];
  bool bnd_match =
    (bnd.x == pb.constraint_lower_bounds[idx]) && (bnd.y == pb.constraint_upper_bounds[idx]);

  if (!bnd_match) {
    if (threadIdx.x == 0) {
      printf("vertex id %d orig id %d bounds mismatch\n", new_idx, idx);
      atomicAdd(errors, 1);
    }
  }
  for (i_t i = threadIdx.x; i < (src_read_end - src_read_beg); i += blockDim.x) {
    if (coeff[i + dst_read_beg] != pb.coefficients[i + src_read_beg]) {
      printf("coeff mismatch vertex id %d orig id %d at edge index %d\n", new_idx, idx, i);
      atomicAdd(errors, 1);
    }
    if (edge[i + dst_read_beg] != pb.variables[i + src_read_beg]) {
      printf("edge mismatch vertex id %d orig id %d at edge index %d\n", new_idx, idx, i);
      atomicAdd(errors, 1);
    }
    if (edge[i + dst_read_beg] >= pb.variables.size()) { printf("oob\n"); }
  }
}

template <typename i_t, typename f_t, typename f_t2>
__global__ void check_variable_data(raft::device_span<i_t> reorg_ids,
                                    raft::device_span<i_t> offsets,
                                    raft::device_span<f_t> coeff,
                                    raft::device_span<i_t> edge,
                                    raft::device_span<f_t2> bounds,
                                    raft::device_span<var_t> var_types,
                                    typename problem_t<i_t, f_t>::view_t pb,
                                    i_t* errors)
{
  i_t new_idx = blockIdx.x;
  i_t idx     = reorg_ids[new_idx];

  auto src_read_beg = pb.reverse_offsets[idx];
  auto src_read_end = pb.reverse_offsets[idx + 1];

  auto dst_read_beg = offsets[new_idx];
  auto bnd          = bounds[idx];
  bool bnd_match =
    (bnd.x == pb.variable_lower_bounds[idx]) && (bnd.y == pb.variable_upper_bounds[idx]);
  auto var_type_match = (var_types[new_idx] == pb.variable_types[idx]);

  if (threadIdx.x == 0) {
    if (!bnd_match) {
      printf("vertex id %d orig id %d bounds mismatch\n", new_idx, idx);
      atomicAdd(errors, 1);
    }
    if (!var_type_match) {
      printf("vertex id %d orig id %d var type mismatch\n", new_idx, idx);
      atomicAdd(errors, 1);
    }
  }
  for (i_t i = threadIdx.x; i < (src_read_end - src_read_beg); i += blockDim.x) {
    if (coeff[i + dst_read_beg] != pb.reverse_coefficients[i + src_read_beg]) {
      printf("coeff mismatch vertex id %d orig id %d at edge index %d\n", new_idx, idx, i);
      atomicAdd(errors, 1);
    }
    if (edge[i + dst_read_beg] != pb.reverse_constraints[i + src_read_beg]) {
      printf("edge mismatch vertex id %d orig id %d at edge index %d\n", blockIdx.x, idx, i);
      atomicAdd(errors, 1);
    }
  }
}

template <typename i_t, typename f_t, typename f_t2>
void create_constraint_graph(const raft::handle_t* handle_ptr,
                             rmm::device_uvector<i_t>& reorg_ids,
                             rmm::device_uvector<i_t>& offsets,
                             rmm::device_uvector<f_t>& coeff,
                             rmm::device_uvector<i_t>& edge,
                             raft::device_span<f_t2> bounds,
                             problem_t<i_t, f_t>& pb,
                             bool debug)
{
  // calculate degree and store in offsets
  thrust::transform(
    handle_ptr->get_thrust_policy(),
    reorg_ids.begin(),
    reorg_ids.end(),
    offsets.begin(),
    [off = make_span(pb.offsets)] __device__(auto id) { return off[id + 1] - off[id]; });
  // create offsets
  thrust::exclusive_scan(
    handle_ptr->get_thrust_policy(), offsets.begin(), offsets.end(), offsets.begin());

  // copy adjacency lists and vertex properties
  constraint_data_copy<i_t, f_t><<<reorg_ids.size(), 256, 0, handle_ptr->get_stream()>>>(
    make_span(reorg_ids), make_span(offsets), make_span(coeff), make_span(edge), bounds, pb.view());

  if (debug) {
    rmm::device_scalar<i_t> errors(0, handle_ptr->get_stream());
    check_constraint_data<i_t, f_t>
      <<<reorg_ids.size(), 256, 0, handle_ptr->get_stream()>>>(make_span(reorg_ids),
                                                               make_span(offsets),
                                                               make_span(coeff),
                                                               make_span(edge),
                                                               bounds,
                                                               pb.view(),
                                                               errors.data());
    i_t error_count = errors.value(handle_ptr->get_stream());
    if (error_count != 0) { std::cerr << "adjacency list copy mismatch\n"; }
  }
}

template <typename i_t, typename f_t, typename f_t2>
void create_variable_graph(const raft::handle_t* handle_ptr,
                           rmm::device_uvector<i_t>& reorg_ids,
                           rmm::device_uvector<i_t>& offsets,
                           rmm::device_uvector<f_t>& coeff,
                           rmm::device_uvector<i_t>& edge,
                           raft::device_span<f_t2> bounds,
                           rmm::device_uvector<var_t>& types,
                           problem_t<i_t, f_t>& pb,
                           bool debug)
{
  // calculate degree and store in offsets
  thrust::transform(
    handle_ptr->get_thrust_policy(),
    reorg_ids.begin(),
    reorg_ids.end(),
    offsets.begin(),
    [off = make_span(pb.reverse_offsets)] __device__(auto id) { return off[id + 1] - off[id]; });
  // create offsets
  thrust::exclusive_scan(
    handle_ptr->get_thrust_policy(), offsets.begin(), offsets.end(), offsets.begin());

  // copy adjacency lists and vertex properties
  variable_data_copy<i_t, f_t>
    <<<reorg_ids.size(), 256, 0, handle_ptr->get_stream()>>>(make_span(reorg_ids),
                                                             make_span(offsets),
                                                             make_span(coeff),
                                                             make_span(edge),
                                                             bounds,
                                                             make_span(types),
                                                             pb.view());

  if (debug) {
    rmm::device_scalar<i_t> errors(0, handle_ptr->get_stream());
    check_variable_data<i_t, f_t>
      <<<reorg_ids.size(), 256, 0, handle_ptr->get_stream()>>>(make_span(reorg_ids),
                                                               make_span(offsets),
                                                               make_span(coeff),
                                                               make_span(edge),
                                                               bounds,
                                                               make_span(types),
                                                               pb.view(),
                                                               errors.data());
    i_t error_count = errors.value(handle_ptr->get_stream());
    if (error_count != 0) { std::cerr << "adjacency list copy mismatch\n"; }
  }
}

template <typename i_t>
void compact_bins(std::vector<i_t>& bins, i_t num_items)
{
  auto found_last_bin  = std::lower_bound(bins.begin(), bins.end(), num_items) - bins.begin();
  auto max_degree_cnst = 2 << (found_last_bin - 3);
  if (max_degree_cnst > 256) { found_last_bin = 10; }
  // bins[0:found_last_bin-1] = 0;
  for (int i = 2; i <= found_last_bin - 1; ++i) {
    bins[i] = bins[1];
  }
  for (size_t i = found_last_bin; i < bins.size(); ++i) {
    bins[i] = num_items;
  }
}

template <typename i_t, typename f_t>
load_balanced_problem_t<i_t, f_t>::load_balanced_problem_t(problem_t<i_t, f_t>& problem_,
                                                           bool debug)
  : pb(&problem_),
    handle_ptr(pb->handle_ptr),
    tolerances(problem_.tolerances),
    n_constraints(pb->n_constraints),
    n_variables(pb->n_variables),
    nnz(pb->nnz),
    cnst_reorg_ids(n_constraints, handle_ptr->get_stream()),
    coefficients(nnz, handle_ptr->get_stream()),
    variables(nnz, handle_ptr->get_stream()),
    offsets(n_constraints + 1, handle_ptr->get_stream()),
    vars_reorg_ids(n_variables, handle_ptr->get_stream()),
    reverse_coefficients(nnz, handle_ptr->get_stream()),
    reverse_constraints(nnz, handle_ptr->get_stream()),
    reverse_offsets(n_variables + 1, handle_ptr->get_stream()),
    vars_types(n_variables, handle_ptr->get_stream()),
    cnst_bounds_data(2 * n_constraints, handle_ptr->get_stream()),
    variable_bounds(2 * n_variables, handle_ptr->get_stream()),
    constraint_lower_bounds(make_span(problem_.constraint_lower_bounds)),
    constraint_upper_bounds(make_span(problem_.constraint_upper_bounds)),
    tmp_cnst_ids(n_constraints, handle_ptr->get_stream()),
    tmp_vars_ids(n_variables, handle_ptr->get_stream()),
    cnst_binner(handle_ptr),
    vars_binner(handle_ptr)
{
  setup(problem_, debug);
}

template <typename i_t, typename f_t>
void load_balanced_problem_t<i_t, f_t>::setup(problem_t<i_t, f_t>& problem_, bool debug)
{
  pb            = &problem_;
  handle_ptr    = pb->handle_ptr;
  n_constraints = pb->n_constraints;
  n_variables   = pb->n_variables;
  nnz           = pb->nnz;
  cnst_reorg_ids.resize(n_constraints, handle_ptr->get_stream());
  coefficients.resize(nnz, handle_ptr->get_stream());
  variables.resize(nnz, handle_ptr->get_stream());
  offsets.resize(n_constraints + 1, handle_ptr->get_stream());
  vars_reorg_ids.resize(n_variables, handle_ptr->get_stream());
  reverse_coefficients.resize(nnz, handle_ptr->get_stream());
  reverse_constraints.resize(nnz, handle_ptr->get_stream());
  reverse_offsets.resize(n_variables + 1, handle_ptr->get_stream());
  vars_types.resize(n_variables, handle_ptr->get_stream());
  cnst_bounds_data.resize(2 * n_constraints, handle_ptr->get_stream());
  variable_bounds.resize(2 * n_variables, handle_ptr->get_stream());
  constraint_lower_bounds = make_span(problem_.constraint_lower_bounds);
  constraint_upper_bounds = make_span(problem_.constraint_upper_bounds);
  tmp_cnst_ids.resize(n_constraints, handle_ptr->get_stream());
  tmp_vars_ids.resize(n_variables, handle_ptr->get_stream());

  cnst_binner.setup(pb->offsets.data(), nullptr, 0, n_constraints);
  auto dist_cnst = cnst_binner.run(tmp_cnst_ids, handle_ptr);
  vars_binner.setup(pb->reverse_offsets.data(), nullptr, 0, n_variables);
  auto dist_vars = vars_binner.run(tmp_vars_ids, handle_ptr);

  auto cnst_bucket = dist_cnst.degree_range();
  auto vars_bucket = dist_vars.degree_range();

  cnst_reorg_ids.resize(cnst_bucket.vertex_ids.size(), handle_ptr->get_stream());
  vars_reorg_ids.resize(vars_bucket.vertex_ids.size(), handle_ptr->get_stream());

  raft::copy(cnst_reorg_ids.data(),
             cnst_bucket.vertex_ids.data(),
             cnst_bucket.vertex_ids.size(),
             handle_ptr->get_stream());
  raft::copy(vars_reorg_ids.data(),
             vars_bucket.vertex_ids.data(),
             vars_bucket.vertex_ids.size(),
             handle_ptr->get_stream());

  auto cnst_bounds = make_span_2(cnst_bounds_data);
  auto vars_bounds = make_span_2(variable_bounds);

  create_constraint_graph<i_t, f_t, f_t2>(handle_ptr,
                                          cnst_reorg_ids,
                                          offsets,
                                          coefficients,
                                          variables,
                                          cnst_bounds,  // new indexing
                                          problem_,
                                          debug);

  create_variable_graph<i_t, f_t, f_t2>(handle_ptr,
                                        vars_reorg_ids,
                                        reverse_offsets,
                                        reverse_coefficients,
                                        reverse_constraints,
                                        vars_bounds,  // old indexing
                                        vars_types,   // new indexing
                                        problem_,
                                        debug);

  cnst_bin_offsets = dist_cnst.bin_offsets_;
  vars_bin_offsets = dist_vars.bin_offsets_;
  if (nnz < 10000) {
    compact_bins(cnst_bin_offsets, n_constraints);
    compact_bins(vars_bin_offsets, n_variables);
  }
  handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
void load_balanced_problem_t<i_t, f_t>::set_updated_bounds(
  const load_balanced_bounds_presolve_t<i_t, f_t>& prs)
{
  raft::copy(
    variable_bounds.data(), prs.vars_bnd.data(), prs.vars_bnd.size(), handle_ptr->get_stream());
}

#if MIP_INSTANTIATE_FLOAT
template class load_balanced_problem_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class load_balanced_problem_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
