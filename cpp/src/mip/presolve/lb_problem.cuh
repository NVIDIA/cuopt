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

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>

#include <mip/problem/problem.cuh>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <utilities/copy_helpers.hpp>
#include "load_balanced_partition_helpers.cuh"

#include <vector>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class problem_t;

enum class csr_type_t { CNST = 0, VARS = 1 };

template <typename i_t, typename f_t>
struct csr_data_view_t {
  using f_t2 = typename type_2<f_t>::type;
  raft::device_span<i_t> reorg_ids;
  raft::device_span<f_t> coefficients;
  raft::device_span<i_t> col_elem;
  raft::device_span<i_t> offsets;

  i_t heavy_beg_id;
  i_t sub_warp_count;
  i_t sub_warp_block_count;
  i_t med_block_count;
  raft::device_span<i_t> warp_offsets;
  raft::device_span<i_t> warp_id_offsets;
  raft::device_span<i_t> block_offsets;
  raft::device_span<i_t> block_id_offsets;

  i_t num_blocks_heavy;
  raft::device_span<i_t> heavy_block_segments;
  raft::device_span<i_t> heavy_vertex_ids;
  raft::device_span<i_t> heavy_pseudo_block_ids;

  raft::device_span<f_t2> cnst_bnd;
  raft::device_span<f_t2> vars_bnd;
  raft::device_span<var_t> var_types;

  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances;

  static constexpr i_t work_per_block = 16 * 1024;
};

template <typename i_t, typename f_t>
class lb_problem_t {
 public:
  using f_t2 = typename type_2<f_t>::type;
  lb_problem_t(problem_t<i_t, f_t>& problem);
  lb_problem_t(lb_problem_t<i_t, f_t>&& problem) = default;
  lb_problem_t& operator=(lb_problem_t&&)        = default;
  void setup(problem_t<i_t, f_t>& problem);

  struct csr_data_t {
    csr_type_t type;
    lb_problem_t<i_t, f_t>& lb_problem;
    i_t rows;
    i_t cols;
    i_t nnz;
    rmm::device_uvector<i_t> reorg_ids;
    rmm::device_uvector<f_t> coefficients;
    rmm::device_uvector<i_t> col_elem;
    rmm::device_uvector<i_t> offsets;

    i_t heavy_beg_id;
    i_t sub_warp_count;
    i_t sub_warp_block_count;
    i_t med_block_count;
    rmm::device_uvector<i_t> warp_offsets;
    rmm::device_uvector<i_t> warp_id_offsets;
    rmm::device_uvector<i_t> block_offsets;
    rmm::device_uvector<i_t> block_id_offsets;

    i_t num_blocks_heavy;
    rmm::device_uvector<i_t> heavy_block_segments;
    rmm::device_uvector<i_t> heavy_vertex_ids;
    rmm::device_uvector<i_t> heavy_pseudo_block_ids;

    vertex_bin_t<i_t> binner;
    std::vector<i_t> bin_offsets;

    csr_data_t(lb_problem_t<i_t, f_t>& lb_problem, problem_t<i_t, f_t>& problem, csr_type_t type);
    void setup(problem_t<i_t, f_t>& problem, i_t heavy_deg_cutoff, bool debug = false);
    csr_data_view_t<i_t, f_t> view();
  };

  problem_t<i_t, f_t>* pb;
  const raft::handle_t* handle_ptr;

  csr_data_t cnst_csr;
  csr_data_t vars_csr;
  rmm::device_uvector<f_t2> cnst_bnd;
  rmm::device_uvector<f_t2> vars_bnd;
  rmm::device_uvector<var_t> var_types;

  i_t n_constraints;
  i_t n_variables;
  i_t nnz;

  static constexpr i_t heavy_degree_cutoff = 16 * 1024;
};

}  // namespace cuopt::linear_programming::detail
