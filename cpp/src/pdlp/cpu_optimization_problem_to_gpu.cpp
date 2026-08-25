/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// CPU -> GPU conversion for cpu_optimization_problem_t.
//
// Split out of cpu_optimization_problem.cpp so that the rest of that class -- which is
// pure host code -- can be compiled into the CUDA-free cuopt_client library. This is the
// only member that constructs an optimization_problem_t, so it is the only one that needs
// <optimization_problem.hpp> and a raft handle. It stays in cuopt_objs (libcuopt).
//
// The explicit member instantiations at the bottom are required: the `template class`
// instantiation in cpu_optimization_problem.cpp no longer sees this definition, so it
// cannot emit this member.

#include <cuopt/export.hpp>
#include <cuopt/mathematical_optimization/cpu_optimization_problem.hpp>
#include <cuopt/mathematical_optimization/optimization_problem.hpp>

// Required: the explicit instantiations below are guarded on MIP_INSTANTIATE_*.
// Without this header those macros are undefined and this TU emits no symbols.
#include <mip_heuristics/mip_constants.hpp>

#include <memory>
#include <stdexcept>
#include <vector>

namespace cuopt::mathematical_optimization {

// Free function (was a virtual member; see optimization_problem_interface.hpp).
// Dispatches on the concrete type: a GPU problem is already what the caller wants, so it
// yields nullptr, matching the previous optimization_problem_t override.
template <typename i_t, typename f_t>
std::unique_ptr<optimization_problem_t<i_t, f_t>> to_optimization_problem(
  optimization_problem_interface_t<i_t, f_t>& problem, raft::handle_t const* handle_ptr)
{
  auto* cpu_problem = dynamic_cast<cpu_optimization_problem_t<i_t, f_t>*>(&problem);
  if (cpu_problem == nullptr) {
    // Already a GPU-backed problem.
    return nullptr;
  }
  auto& self = *cpu_problem;

  if (handle_ptr == nullptr) {
    throw std::runtime_error(
      "cpu_optimization_problem_t::to_optimization_problem(): "
      "handle_ptr is null. A RAFT handle with CUDA resources is required to convert "
      "a CPU-backed problem to a GPU-backed optimization_problem_t.");
  }

  auto gpu_problem = std::make_unique<optimization_problem_t<i_t, f_t>>(handle_ptr);

  // Set scalar values
  gpu_problem->set_maximize(self.maximize_);
  gpu_problem->set_objective_scaling_factor(self.objective_scaling_factor_);
  gpu_problem->set_objective_offset(self.objective_offset_);
  gpu_problem->set_problem_category(self.problem_category_);

  // Set string values
  if (!self.objective_name_.empty()) gpu_problem->set_objective_name(self.objective_name_);
  if (!self.problem_name_.empty()) gpu_problem->set_problem_name(self.problem_name_);
  if (!self.var_names_.empty()) gpu_problem->set_variable_names(self.var_names_);
  if (!self.row_names_.empty()) gpu_problem->set_row_names(self.row_names_);

  // Set CSR constraint matrix (data will be copied to GPU by optimization_problem_t setters)
  // Use self.A_offsets_ presence as the guard: a valid CSR can have zero non-zeros but still
  // needs row offsets to define the number of constraints.
  if (!self.A_offsets_.empty()) {
    gpu_problem->set_csr_constraint_matrix(self.A_.data(),
                                           self.A_.size(),
                                           self.A_indices_.data(),
                                           self.A_indices_.size(),
                                           self.A_offsets_.data(),
                                           self.A_offsets_.size());
  }

  // Set constraint bounds
  if (!self.b_.empty()) { gpu_problem->set_constraint_bounds(self.b_.data(), self.b_.size()); }

  // Set objective coefficients
  if (!self.c_.empty()) { gpu_problem->set_objective_coefficients(self.c_.data(), self.c_.size()); }

  // Set quadratic objective if present (GPU setter symmetrizes once: H = Q + Q^T)
  if (!self.Q_values_.empty()) {
    gpu_problem->set_quadratic_objective_matrix(self.Q_values_.data(),
                                                self.Q_values_.size(),
                                                self.Q_indices_.data(),
                                                self.Q_indices_.size(),
                                                self.Q_offsets_.data(),
                                                self.Q_offsets_.size());
  }

  if (!self.quadratic_constraints_.empty()) {
    gpu_problem->set_quadratic_constraints(
      std::vector<typename optimization_problem_interface_t<i_t, f_t>::quadratic_constraint_t>(
        self.quadratic_constraints_));
  }

  // Set variable bounds
  if (!self.variable_lower_bounds_.empty()) {
    gpu_problem->set_variable_lower_bounds(self.variable_lower_bounds_.data(),
                                           self.variable_lower_bounds_.size());
  }
  if (!self.variable_upper_bounds_.empty()) {
    gpu_problem->set_variable_upper_bounds(self.variable_upper_bounds_.data(),
                                           self.variable_upper_bounds_.size());
  }

  // Set variable types
  if (!self.variable_types_.empty()) {
    gpu_problem->set_variable_types(self.variable_types_.data(), self.variable_types_.size());
  }

  // Set constraint bounds
  if (!self.constraint_lower_bounds_.empty()) {
    gpu_problem->set_constraint_lower_bounds(self.constraint_lower_bounds_.data(),
                                             self.constraint_lower_bounds_.size());
  }
  if (!self.constraint_upper_bounds_.empty()) {
    gpu_problem->set_constraint_upper_bounds(self.constraint_upper_bounds_.data(),
                                             self.constraint_upper_bounds_.size());
  }

  // Set row types
  if (!self.row_types_.empty()) { gpu_problem->set_row_types(self.row_types_.data(), self.row_types_.size()); }

  return gpu_problem;
}


// ==============================================================================
// Template instantiations matching cpu_optimization_problem.cpp
// ==============================================================================

#if MIP_INSTANTIATE_FLOAT
template CUOPT_EXPORT std::unique_ptr<optimization_problem_t<int32_t, float>> to_optimization_problem(
  optimization_problem_interface_t<int32_t, float>&, raft::handle_t const*);
#endif
#if MIP_INSTANTIATE_DOUBLE
template CUOPT_EXPORT std::unique_ptr<optimization_problem_t<int32_t, double>> to_optimization_problem(
  optimization_problem_interface_t<int32_t, double>&, raft::handle_t const*);
#endif

}  // namespace cuopt::mathematical_optimization
