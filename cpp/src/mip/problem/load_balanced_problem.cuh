/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

// THIS IS LIKELY THE INNER-MOST INCLUDE
// FOR COMPILE TIME, WE SHOULD KEEP THE INCLUDES ON THIS HEADER MINIMAL

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include "host_helper.cuh"

#include <utilities/macros.cuh>

#include <mip/logger.hpp>
#include <mip/presolve/load_balanced_bounds_presolve.cuh>
#include <mip/problem/problem.cuh>

#include <mip/presolve/load_balanced_partition_helpers.cuh>
#include <raft/core/nvtx.hpp>
#include <raft/random/rng_device.cuh>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class load_balanced_bounds_presolve_t;

template <typename i_t, typename f_t>
class load_balanced_problem_t {
 public:
  using i_t2 = typename type_2<i_t>::type;
  using f_t2 = typename type_2<f_t>::type;
  load_balanced_problem_t(problem_t<i_t, f_t>& problem, bool debug = false);
  load_balanced_problem_t() = delete;
  void setup(problem_t<i_t, f_t>& problem, bool debug = false);
  void set_updated_bounds(const load_balanced_bounds_presolve_t<i_t, f_t>& prs);

  problem_t<i_t, f_t>* pb;
  const raft::handle_t* handle_ptr;
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances{};

  i_t n_variables;
  i_t n_constraints;
  i_t nnz;

  // csr - cnst
  rmm::device_uvector<i_t> cnst_reorg_ids;
  rmm::device_uvector<f_t> coefficients;
  rmm::device_uvector<i_t> variables;
  rmm::device_uvector<i_t> offsets;

  // csc - vars

  rmm::device_uvector<i_t> vars_reorg_ids;

  // same adjacency list contents but reorganized by new indexing
  rmm::device_uvector<f_t> reverse_coefficients;
  rmm::device_uvector<i_t> reverse_constraints;

  // new indexing
  rmm::device_uvector<i_t> reverse_offsets;
  rmm::device_uvector<var_t> vars_types;
  rmm::device_uvector<f_t> cnst_bounds_data;

  // old indexing
  rmm::device_uvector<f_t> variable_bounds;

  rmm::device_uvector<i_t> tmp_cnst_ids;
  rmm::device_uvector<i_t> tmp_vars_ids;

  raft::device_span<f_t> constraint_lower_bounds;
  raft::device_span<f_t> constraint_upper_bounds;

  std::vector<i_t> cnst_bin_offsets;
  std::vector<i_t> vars_bin_offsets;

  vertex_bin_t<i_t> cnst_binner;
  vertex_bin_t<i_t> vars_binner;
};

}  // namespace cuopt::linear_programming::detail
