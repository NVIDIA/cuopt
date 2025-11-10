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

#include <cuopt/linear_programming/gpu_optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>

#include <raft/common/nvtx.hpp>
#include <raft/core/handle.hpp>

namespace cuopt::linear_programming {

/**
 * @brief Convert host-based optimization_problem_t to GPU-based gpu_optimization_problem_t
 *
 * This function performs the actual host-to-device memory copies for local solving.
 * For remote solving, this conversion is skipped entirely.
 *
 * @tparam i_t Integer type for indices
 * @tparam f_t Floating point type for values
 * @param handle_ptr RAFT handle for stream and device context
 * @param host_problem The problem with data in host memory (std::vector)
 * @return gpu_optimization_problem_t The problem with data in GPU memory (rmm::device_uvector)
 */
template <typename i_t, typename f_t>
gpu_optimization_problem_t<i_t, f_t> host_to_gpu_problem(
  raft::handle_t const* handle_ptr, const optimization_problem_t<i_t, f_t>& host_problem)
{
  raft::common::nvtx::range fun_scope("host_to_gpu_problem");

  // Create GPU problem
  gpu_optimization_problem_t<i_t, f_t> gpu_problem(handle_ptr);

  // Copy basic properties
  gpu_problem.set_maximize(host_problem.get_sense());
  gpu_problem.set_objective_scaling_factor(host_problem.get_objective_scaling_factor());
  gpu_problem.set_objective_offset(host_problem.get_objective_offset());
  gpu_problem.set_problem_category(host_problem.get_problem_category());
  gpu_problem.set_objective_name(host_problem.get_objective_name());
  gpu_problem.set_problem_name(host_problem.get_problem_name());
  gpu_problem.set_variable_names(host_problem.get_variable_names());
  gpu_problem.set_row_names(host_problem.get_row_names());

  // Copy constraint matrix (CSR format)
  const auto& A_values  = host_problem.get_constraint_matrix_values();
  const auto& A_indices = host_problem.get_constraint_matrix_indices();
  const auto& A_offsets = host_problem.get_constraint_matrix_offsets();
  if (!A_values.empty()) {
    gpu_problem.set_csr_constraint_matrix(A_values.data(),
                                          A_values.size(),
                                          A_indices.data(),
                                          A_indices.size(),
                                          A_offsets.data(),
                                          A_offsets.size());
  }

  // Copy objective coefficients
  const auto& c = host_problem.get_objective_coefficients();
  if (!c.empty()) { gpu_problem.set_objective_coefficients(c.data(), c.size()); }

  // Copy constraint bounds
  const auto& b = host_problem.get_constraint_bounds();
  if (!b.empty()) { gpu_problem.set_constraint_bounds(b.data(), b.size()); }

  // Copy variable bounds
  const auto& var_lb = host_problem.get_variable_lower_bounds();
  if (!var_lb.empty()) { gpu_problem.set_variable_lower_bounds(var_lb.data(), var_lb.size()); }

  const auto& var_ub = host_problem.get_variable_upper_bounds();
  if (!var_ub.empty()) { gpu_problem.set_variable_upper_bounds(var_ub.data(), var_ub.size()); }

  // Copy constraint bounds
  const auto& con_lb = host_problem.get_constraint_lower_bounds();
  if (!con_lb.empty()) { gpu_problem.set_constraint_lower_bounds(con_lb.data(), con_lb.size()); }

  const auto& con_ub = host_problem.get_constraint_upper_bounds();
  if (!con_ub.empty()) { gpu_problem.set_constraint_upper_bounds(con_ub.data(), con_ub.size()); }

  // Copy row types
  const auto& row_types = host_problem.get_row_types();
  if (!row_types.empty()) { gpu_problem.set_row_types(row_types.data(), row_types.size()); }

  // Copy variable types
  const auto& var_types = host_problem.get_variable_types();
  if (!var_types.empty()) { gpu_problem.set_variable_types(var_types.data(), var_types.size()); }

  return gpu_problem;
}

}  // namespace cuopt::linear_programming
