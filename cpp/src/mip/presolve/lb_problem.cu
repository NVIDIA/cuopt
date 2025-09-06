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
#include "lb_problem.cuh"
#include "lb_problem_setup.cuh"
#include "load_balanced_partition_helpers.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
lb_problem_t<i_t, f_t>::csr_data_t::csr_data_t(lb_problem_t<i_t, f_t>& lb_problem_,
                                               problem_t<i_t, f_t>& problem_,
                                               csr_type_t type_)
  : type(type_),
    lb_problem(lb_problem_),
    reorg_ids(0, problem_.handle_ptr->get_stream()),
    coefficients(0, problem_.handle_ptr->get_stream()),
    col_elem(0, problem_.handle_ptr->get_stream()),
    offsets(0, problem_.handle_ptr->get_stream()),
    warp_offsets(0, problem_.handle_ptr->get_stream()),
    warp_id_offsets(0, problem_.handle_ptr->get_stream()),
    block_offsets(0, problem_.handle_ptr->get_stream()),
    block_id_offsets(0, problem_.handle_ptr->get_stream()),
    heavy_block_segments(0, problem_.handle_ptr->get_stream()),
    heavy_vertex_ids(0, problem_.handle_ptr->get_stream()),
    heavy_pseudo_block_ids(0, problem_.handle_ptr->get_stream()),
    binner(problem_.handle_ptr)
{
}

template <typename i_t, typename f_t>
csr_data_view_t<i_t, f_t> lb_problem_t<i_t, f_t>::csr_data_t::view()
{
  csr_data_view_t<i_t, f_t> view;
  view.reorg_ids    = make_span(reorg_ids);
  view.coefficients = make_span(coefficients);
  view.col_elem     = make_span(col_elem);
  view.offsets      = make_span(offsets);

  view.heavy_beg_id         = heavy_beg_id;
  view.sub_warp_count       = sub_warp_count;
  view.sub_warp_block_count = sub_warp_block_count;
  view.med_block_count      = med_block_count;

  view.warp_offsets     = make_span(warp_offsets);
  view.warp_id_offsets  = make_span(warp_id_offsets);
  view.block_offsets    = make_span(block_offsets);
  view.block_id_offsets = make_span(block_id_offsets);

  view.num_blocks_heavy = num_blocks_heavy;

  view.heavy_block_segments   = make_span(heavy_block_segments);
  view.heavy_vertex_ids       = make_span(heavy_vertex_ids);
  view.heavy_pseudo_block_ids = make_span(heavy_pseudo_block_ids);

  view.cnst_bnd   = make_span(lb_problem.cnst_bnd);
  view.vars_bnd   = make_span(lb_problem.vars_bnd);
  view.var_types  = make_span(lb_problem.var_types);
  view.tolerances = lb_problem.pb->tolerances;
  return view;
}

template <typename i_t, typename f_t>
void lb_problem_t<i_t, f_t>::csr_data_t::setup(problem_t<i_t, f_t>& problem,
                                               i_t heavy_deg_cutoff,
                                               bool debug)
{
  rows = (type == csr_type_t::CNST) ? problem.n_constraints : problem.n_variables;
  cols = (type == csr_type_t::VARS) ? problem.n_constraints : problem.n_variables;
  nnz  = problem.nnz;
  reorg_ids.resize(rows, problem.handle_ptr->get_stream());
  coefficients.resize(nnz, problem.handle_ptr->get_stream());
  col_elem.resize(nnz, problem.handle_ptr->get_stream());
  offsets.resize(rows + 1, problem.handle_ptr->get_stream());

  if (type == csr_type_t::CNST) {
    // std::cout << "cnst setup\n";
    binner.setup(problem.offsets.data(), rows);
  } else {
    // std::cout << "var setup\n";
    binner.setup(problem.reverse_offsets.data(), rows);
  }
  auto dist   = binner.run(reorg_ids, problem.handle_ptr);
  bin_offsets = dist.bin_offsets_;

  if (type == csr_type_t::CNST) {
    create_graph<i_t, f_t>(problem.handle_ptr,
                           reorg_ids,
                           offsets,
                           coefficients,
                           col_elem,
                           problem.offsets,
                           problem.coefficients,
                           problem.variables,
                           debug);
  } else {
    create_graph<i_t, f_t>(problem.handle_ptr,
                           reorg_ids,
                           offsets,
                           coefficients,
                           col_elem,
                           problem.reverse_offsets,
                           problem.reverse_coefficients,
                           problem.reverse_constraints,
                           debug);
  }

  std::tie(num_blocks_heavy, heavy_beg_id) =
    create_heavy_item_block_segments(problem.handle_ptr->get_stream(),
                                     heavy_vertex_ids,
                                     heavy_pseudo_block_ids,
                                     heavy_block_segments,
                                     heavy_deg_cutoff,
                                     bin_offsets,
                                     offsets);

  i_t w_t_r = 4;
  std::tie(sub_warp_count, sub_warp_block_count, med_block_count) =
    block_meta(problem.handle_ptr->get_stream(),
               heavy_beg_id,
               warp_offsets,
               warp_id_offsets,
               block_offsets,
               block_id_offsets,
               bin_offsets,
               w_t_r,
               heavy_deg_cutoff,
               true);
}

template <typename i_t, typename f_t>
lb_problem_t<i_t, f_t>::lb_problem_t(problem_t<i_t, f_t>& problem)
  : pb(&problem),
    handle_ptr(problem.handle_ptr),
    cnst_csr(*this, problem, csr_type_t::CNST),
    vars_csr(*this, problem, csr_type_t::VARS),
    cnst_bnd(0, handle_ptr->get_stream()),
    vars_bnd(0, handle_ptr->get_stream()),
    var_types(0, handle_ptr->get_stream()),
    n_constraints(problem.n_constraints),
    n_variables(problem.n_variables),
    nnz(problem.nnz)
{
  setup(problem);
}

template <typename i_t, typename f_t>
void lb_problem_t<i_t, f_t>::setup(problem_t<i_t, f_t>& problem)
{
  handle_ptr = problem.handle_ptr;
  cnst_csr.setup(problem, heavy_degree_cutoff);
  vars_csr.setup(problem, heavy_degree_cutoff);
  cnst_bnd.resize(problem.n_constraints, handle_ptr->get_stream());
  vars_bnd.resize(problem.n_variables, handle_ptr->get_stream());
  var_types.resize(problem.n_variables, handle_ptr->get_stream());
  thrust::for_each(handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(problem.n_constraints),
                   [pb                = problem.view(),
                    reorg_ids         = make_span(cnst_csr.reorg_ids),
                    constraint_bounds = make_span(cnst_bnd)] __device__(auto idx) {
                     auto r_id  = reorg_ids[idx];
                     using f_t2 = typename type_2<f_t>::type;
                     constraint_bounds[idx] =
                       f_t2{pb.constraint_lower_bounds[r_id], pb.constraint_upper_bounds[r_id]};
                   });
  thrust::for_each(handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(problem.n_variables),
                   [pb        = problem.view(),
                    reorg_ids = make_span(vars_csr.reorg_ids),
                    v_bnd     = make_span(vars_bnd),
                    v_types   = make_span(var_types)] __device__(auto idx) {
                     auto r_id    = reorg_ids[idx];
                     v_types[idx] = pb.variable_types[r_id];
                     v_bnd[idx]   = pb.variable_bounds[idx];
                   });

  pb            = &problem;
  n_constraints = problem.n_constraints;
  n_variables   = problem.n_variables;
  nnz           = problem.nnz;
}

#if MIP_INSTANTIATE_FLOAT
template class lb_problem_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class lb_problem_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
