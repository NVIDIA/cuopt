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

#include <cuopt/linear_programming/optimization_problem.hpp>

#include <thrust/sequence.h>
#include <thrust/uninitialized_fill.h>
#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace linear_programming::detail {

template <typename i_t, typename f_t>
class problem_t;

template <typename i_t, typename f_t>
class presolve_data_t {
 public:
  presolve_data_t(const optimization_problem_t<i_t, f_t>& problem, rmm::cuda_stream_view stream)
    : variable_offsets(problem.get_n_variables(), 0),
      additional_var_used(problem.get_n_variables(), false),
      additional_var_id_per_var(problem.get_n_variables(), -1),
      objective_offset(problem.get_objective_offset()),
      objective_scaling_factor(problem.get_objective_scaling_factor()),
      variable_mapping(0, stream),
      fixed_var_assignment(0, stream)
  {
  }

  presolve_data_t(const presolve_data_t& other, rmm::cuda_stream_view stream)
    : variable_offsets(other.variable_offsets),
      additional_var_used(other.additional_var_used),
      additional_var_id_per_var(other.additional_var_id_per_var),
      objective_offset(other.objective_offset),
      objective_scaling_factor(other.objective_scaling_factor),
      variable_mapping(other.variable_mapping, stream),
      fixed_var_assignment(other.fixed_var_assignment, stream)
  {
  }

  void initialize_var_mapping(const problem_t<i_t, f_t>& problem, const raft::handle_t* handle_ptr)
  {
    variable_mapping.resize(problem.n_variables, handle_ptr->get_stream());
    thrust::sequence(
      handle_ptr->get_thrust_policy(), variable_mapping.begin(), variable_mapping.end());
    fixed_var_assignment.resize(problem.n_variables, handle_ptr->get_stream());
    thrust::uninitialized_fill(handle_ptr->get_thrust_policy(),
                               fixed_var_assignment.begin(),
                               fixed_var_assignment.end(),
                               0.);
  }

  void reset_additional_vars(const problem_t<i_t, f_t>& problem, const raft::handle_t* handle_ptr)
  {
    variable_offsets.assign(problem.n_variables, 0);
    additional_var_used.assign(problem.n_variables, false);
    additional_var_id_per_var.assign(problem.n_variables, -1);
  }

  presolve_data_t(presolve_data_t&&)                 = default;
  presolve_data_t& operator=(presolve_data_t&&)      = default;
  presolve_data_t& operator=(const presolve_data_t&) = delete;

  // offsets of variables
  std::vector<f_t> variable_offsets;
  std::vector<bool> additional_var_used;
  std::vector<i_t> additional_var_id_per_var;
  f_t objective_offset;
  f_t objective_scaling_factor;

  rmm::device_uvector<i_t> variable_mapping;
  rmm::device_uvector<f_t> fixed_var_assignment;
};

}  // namespace linear_programming::detail
}  // namespace cuopt
