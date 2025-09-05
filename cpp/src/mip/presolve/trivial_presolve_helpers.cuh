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

#include <raft/core/device_span.hpp>

#include <thrust/functional.h>

namespace cuopt::linear_programming::detail {

template <typename i_t>
struct non_zero_degree_t {
  raft::device_span<i_t> offsets;
  non_zero_degree_t(raft::device_span<i_t> offsets_) : offsets(offsets_) {}
  __device__ i_t operator()(i_t i) { return offsets[i] != offsets[i + 1]; }
};

template <typename f_t, typename f_t2>
struct is_variable_free_t {
  f_t tol;
  raft::device_span<f_t2> bnd;
  is_variable_free_t(f_t tol_, raft::device_span<f_t2> bnd_) : tol(tol_), bnd(bnd_) {}
  template <typename tuple_t>
  __device__ bool operator()(tuple_t edge)
  {
    auto var    = thrust::get<2>(edge);
    auto bounds = bnd[var];
    return abs(get_upper(bounds) - get_lower(bounds)) > tol;
  }
};

template <typename i_t, typename f_t, typename f_t2>
struct assign_fixed_var_t {
  raft::device_span<i_t> is_var_used;
  raft::device_span<f_t2> variable_bounds;
  raft::device_span<f_t> objective_coefficients;
  raft::device_span<i_t> variable_mapping;
  raft::device_span<f_t> fixed_assignment;
  assign_fixed_var_t(raft::device_span<i_t> is_var_used_,
                     raft::device_span<f_t2> variable_bounds_,
                     raft::device_span<f_t> objective_coefficients_,
                     raft::device_span<i_t> variable_mapping_,
                     raft::device_span<f_t> fixed_assignment_)
    : is_var_used(is_var_used_),
      variable_bounds(variable_bounds_),
      objective_coefficients(objective_coefficients_),
      variable_mapping(variable_mapping_),
      fixed_assignment(fixed_assignment_)
  {
  }

  __device__ void operator()(i_t i) const
  {
    if (!is_var_used[i]) {
      auto orig_v_idx = variable_mapping[i];
      auto bounds     = variable_bounds[i];
      fixed_assignment[orig_v_idx] =
        (objective_coefficients[i] > 0) ? get_lower(bounds) : get_upper(bounds);
    }
  }
};

template <typename i_t, typename f_t, typename f_t2>
struct elem_multi_t {
  raft::device_span<f_t> coefficients;
  raft::device_span<i_t> variables;
  raft::device_span<f_t> obj_coefficients;
  raft::device_span<f_t2> variable_bounds;
  elem_multi_t(raft::device_span<f_t> coefficients_,
               raft::device_span<i_t> variables_,
               raft::device_span<f_t> obj_coefficients_,
               raft::device_span<f_t2> variable_bounds_)
    : coefficients(coefficients_),
      variables(variables_),
      obj_coefficients(obj_coefficients_),
      variable_bounds(variable_bounds_)
  {
  }

  __device__ f_t operator()(i_t i) const
  {
    auto var    = variables[i];
    auto bounds = variable_bounds[var];
    if (obj_coefficients[var] > 0) {
      return get_lower(bounds) * coefficients[i];
    } else {
      return get_upper(bounds) * coefficients[i];
    }
  }
};

template <typename i_t, typename f_t>
struct update_constraint_bounds_t {
  raft::device_span<i_t> unused_csr_cnst;
  raft::device_span<f_t> unused_csr_cnst_bound_offsets;
  raft::device_span<f_t> constraint_lower_bounds;
  raft::device_span<f_t> constraint_upper_bounds;
  update_constraint_bounds_t(raft::device_span<i_t> unused_csr_cnst_,
                             raft::device_span<f_t> unused_csr_cnst_bound_offsets_,
                             raft::device_span<f_t> constraint_lower_bounds_,
                             raft::device_span<f_t> constraint_upper_bounds_)
    : unused_csr_cnst(unused_csr_cnst_),
      unused_csr_cnst_bound_offsets(unused_csr_cnst_bound_offsets_),
      constraint_lower_bounds(constraint_lower_bounds_),
      constraint_upper_bounds(constraint_upper_bounds_)
  {
  }
  __device__ void operator()(i_t i) const
  {
    auto cnst     = unused_csr_cnst[i];
    auto cnst_off = unused_csr_cnst_bound_offsets[i];
    constraint_lower_bounds[cnst] -= cnst_off;
    constraint_upper_bounds[cnst] -= cnst_off;
  }
};

template <typename i_t, typename f_t, typename f_t2>
struct unused_var_obj_offset_t {
  raft::device_span<i_t> var_map;
  raft::device_span<f_t> objective_coefficients;
  raft::device_span<f_t2> bnd;

  unused_var_obj_offset_t(raft::device_span<i_t> var_map_,
                          raft::device_span<f_t> objective_coefficients_,
                          raft::device_span<f_t2> bnd_)
    : var_map(var_map_), objective_coefficients(objective_coefficients_), bnd(bnd_)
  {
  }

  __host__ __device__ f_t operator()(const i_t i) const
  {
    auto obj_coeff = objective_coefficients[i];
    // in case both bounds are infinite
    if (obj_coeff == 0.) return 0.;
    auto bounds  = bnd[i];
    auto obj_off = (obj_coeff > 0) ? obj_coeff * get_lower(bounds) : obj_coeff * get_upper(bounds);
    return var_map[i] ? 0. : obj_off;
  }
};

template <typename i_t>
struct sub_t {
  __device__ i_t operator()(i_t i) const { return i - 1; }
};

template <typename i_t>
struct apply_renumbering_t {
  raft::device_span<i_t> gather_map;
  apply_renumbering_t(raft::device_span<i_t> gather_map_) : gather_map(gather_map_) {}
  __device__ i_t operator()(i_t i) { return gather_map[i]; }
};

template <typename i_t>
struct coo_to_offset_t {
  raft::device_span<i_t> coo_major;
  raft::device_span<i_t> offset;
  coo_to_offset_t(raft::device_span<i_t> coo_major_, raft::device_span<i_t> offset_)
    : coo_major(coo_major_), offset(offset_)
  {
  }
  __device__ void operator()(i_t i)
  {
    if (i == static_cast<i_t>(coo_major.size()) - 1) {
      offset[coo_major[i] + 1] = i + 1;
      offset[0]                = 0;
    } else if (coo_major[i] != coo_major[i + 1]) {
      offset[coo_major[i + 1]] = i + 1;
    }
  }
};

template <typename i_t>
struct is_zero_t {
  __device__ bool operator()(const i_t x) { return (x == 0); }
};

}  // namespace cuopt::linear_programming::detail
