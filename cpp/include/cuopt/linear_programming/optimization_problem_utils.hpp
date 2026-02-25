/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/cpu_pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <mps_parser/data_model_view.hpp>
#include <mps_parser/mps_data_model.hpp>

namespace cuopt::linear_programming {

/**
 * @brief Helper function to populate optimization_problem_interface_t from mps_data_model_t
 *
 * This avoids creating a temporary optimization_problem_t which requires GPU memory allocation.
 * Instead, it directly populates the interface which can use either CPU or GPU memory.
 *
 * @tparam i_t Integer type for indices
 * @tparam f_t Floating point type for values
 * @param[out] problem The optimization problem interface to populate
 * @param[in] data_model The MPS data model containing the problem data
 */
template <typename i_t, typename f_t>
void populate_from_mps_data_model(optimization_problem_interface_t<i_t, f_t>* problem,
                                  const mps_parser::mps_data_model_t<i_t, f_t>& data_model)
{
  // Set scalar values
  problem->set_maximize(data_model.get_sense());
  problem->set_objective_scaling_factor(data_model.get_objective_scaling_factor());
  problem->set_objective_offset(data_model.get_objective_offset());

  // Set string values
  if (!data_model.get_objective_name().empty())
    problem->set_objective_name(data_model.get_objective_name());
  if (!data_model.get_problem_name().empty())
    problem->set_problem_name(data_model.get_problem_name());
  if (!data_model.get_variable_names().empty())
    problem->set_variable_names(data_model.get_variable_names());
  if (!data_model.get_row_names().empty()) problem->set_row_names(data_model.get_row_names());

  // Set array values
  i_t n_vars        = data_model.get_n_variables();
  i_t n_constraints = data_model.get_n_constraints();

  const auto& obj_coeffs = data_model.get_objective_coefficients();
  if (!obj_coeffs.empty()) { problem->set_objective_coefficients(obj_coeffs.data(), n_vars); }

  const auto& A_offsets = data_model.get_constraint_matrix_offsets();
  if (!A_offsets.empty() && A_offsets.size() > static_cast<size_t>(n_constraints)) {
    i_t n_nonzeros = A_offsets[n_constraints];
    if (n_nonzeros > 0) {
      problem->set_csr_constraint_matrix(data_model.get_constraint_matrix_values().data(),
                                         n_nonzeros,
                                         data_model.get_constraint_matrix_indices().data(),
                                         n_nonzeros,
                                         A_offsets.data(),
                                         n_constraints + 1);
    }
  }

  const auto& con_bounds = data_model.get_constraint_bounds();
  if (!con_bounds.empty()) { problem->set_constraint_bounds(con_bounds.data(), n_constraints); }
  const auto& con_lb = data_model.get_constraint_lower_bounds();
  if (!con_lb.empty()) { problem->set_constraint_lower_bounds(con_lb.data(), n_constraints); }

  const auto& con_ub = data_model.get_constraint_upper_bounds();
  if (!con_ub.empty()) { problem->set_constraint_upper_bounds(con_ub.data(), n_constraints); }

  const auto& row_types = data_model.get_row_types();
  if (!row_types.empty()) { problem->set_row_types(row_types.data(), n_constraints); }

  const auto& var_lb = data_model.get_variable_lower_bounds();
  if (!var_lb.empty()) { problem->set_variable_lower_bounds(var_lb.data(), n_vars); }

  const auto& var_ub = data_model.get_variable_upper_bounds();
  if (!var_ub.empty()) { problem->set_variable_upper_bounds(var_ub.data(), n_vars); }

  // Convert variable types from char to enum
  const auto& char_variable_types = data_model.get_variable_types();
  if (!char_variable_types.empty()) {
    std::vector<var_t> enum_variable_types(char_variable_types.size());
    for (size_t i = 0; i < char_variable_types.size(); ++i) {
      enum_variable_types[i] = (char_variable_types[i] == 'I' || char_variable_types[i] == 'B')
                                 ? var_t::INTEGER
                                 : var_t::CONTINUOUS;
    }
    problem->set_variable_types(enum_variable_types.data(), enum_variable_types.size());
    // Problem category (LP/MIP/IP) is auto-detected by set_variable_types
  }

  // Handle quadratic objective if present
  if (data_model.has_quadratic_objective()) {
    auto& q_offsets = data_model.get_quadratic_objective_offsets();
    cuopt_expects(q_offsets.size() >= static_cast<size_t>(n_vars + 1),
                  error_type_t::ValidationError,
                  "Quadratic objective offsets vector too small for number of variables");
    i_t q_nonzeros = q_offsets[n_vars];
    problem->set_quadratic_objective_matrix(data_model.get_quadratic_objective_values().data(),
                                            q_nonzeros,
                                            data_model.get_quadratic_objective_indices().data(),
                                            q_nonzeros,
                                            q_offsets.data(),
                                            n_vars + 1);
  }
}

/**
 * @brief Helper function to populate optimization_problem_interface_t from data_model_view_t
 *
 * This is used by the Python Cython interface which provides data_model_view_t.
 * Similar to populate_from_mps_data_model but works with data_model_view_t instead.
 *
 * @tparam i_t Integer type for indices
 * @tparam f_t Floating point type for values
 * @param[out] problem The optimization problem interface to populate
 * @param[in] data_model The data model view containing the problem data
 * @param[in] solver_settings Optional solver settings (for warmstart data, GPU only)
 * @param[in] handle Optional RAFT handle (for warmstart data, GPU only)
 */
template <typename i_t, typename f_t>
void populate_from_data_model_view(optimization_problem_interface_t<i_t, f_t>* problem,
                                   cuopt::mps_parser::data_model_view_t<i_t, f_t>* data_model,
                                   solver_settings_t<i_t, f_t>* solver_settings = nullptr,
                                   const raft::handle_t* handle                 = nullptr)
{
  problem->set_maximize(data_model->get_sense());

  if (data_model->get_constraint_matrix_values().size() != 0 &&
      data_model->get_constraint_matrix_indices().size() != 0 &&
      data_model->get_constraint_matrix_offsets().size() != 0) {
    problem->set_csr_constraint_matrix(data_model->get_constraint_matrix_values().data(),
                                       data_model->get_constraint_matrix_values().size(),
                                       data_model->get_constraint_matrix_indices().data(),
                                       data_model->get_constraint_matrix_indices().size(),
                                       data_model->get_constraint_matrix_offsets().data(),
                                       data_model->get_constraint_matrix_offsets().size());
  }

  if (data_model->get_constraint_bounds().size() != 0) {
    problem->set_constraint_bounds(data_model->get_constraint_bounds().data(),
                                   data_model->get_constraint_bounds().size());
  }

  if (data_model->get_objective_coefficients().size() != 0) {
    problem->set_objective_coefficients(data_model->get_objective_coefficients().data(),
                                        data_model->get_objective_coefficients().size());
  }

  problem->set_objective_scaling_factor(data_model->get_objective_scaling_factor());
  problem->set_objective_offset(data_model->get_objective_offset());

  // Handle warmstart data with GPU↔CPU conversion if needed
  if (solver_settings != nullptr) {
    bool target_is_gpu = (handle != nullptr);

    // Check which warmstart type is populated
    // Note: Python sets the VIEW (spans), so check both view and data for GPU warmstart
    // CPU warmstart is set directly in the data structure
    bool has_gpu_warmstart_view = (solver_settings->get_pdlp_warm_start_data_view()
                                     .last_restart_duality_gap_dual_solution_.size() > 0);
    bool has_gpu_warmstart_data =
      solver_settings->get_pdlp_settings().get_pdlp_warm_start_data().is_populated();
    bool has_cpu_warmstart =
      solver_settings->get_pdlp_settings().get_cpu_pdlp_warm_start_data().is_populated();

    bool has_gpu_warmstart = has_gpu_warmstart_view || has_gpu_warmstart_data;

    if (has_gpu_warmstart || has_cpu_warmstart) {
      if (target_is_gpu) {
        // Target is GPU backend
        if (has_gpu_warmstart_view) {
          // GPU warmstart from Python → GPU backend: copy view (spans) to data (device_uvectors)
          // Python sets the view (spans over cuDF), but solver needs device_uvectors
          pdlp_warm_start_data_t<i_t, f_t> pdlp_warm_start_data(
            solver_settings->get_pdlp_warm_start_data_view(), handle->get_stream());
          solver_settings->get_pdlp_settings().set_pdlp_warm_start_data(pdlp_warm_start_data);
        } else if (has_gpu_warmstart_data) {
          // GPU warmstart from C++ API → GPU backend: data already set, nothing to do
          // The device_uvectors are already populated in the settings
        } else {
          // CPU warmstart → GPU backend: convert H2D
          pdlp_warm_start_data_t<i_t, f_t> gpu_warmstart = convert_to_gpu_warmstart(
            solver_settings->get_pdlp_settings().get_cpu_pdlp_warm_start_data(),
            handle->get_stream());
          solver_settings->get_pdlp_settings().set_pdlp_warm_start_data(gpu_warmstart);
        }
      } else {
        // Target is CPU backend (remote execution)
        if (has_cpu_warmstart) {
          // CPU warmstart → CPU backend: data already in correct form, nothing to do
        } else if (has_gpu_warmstart_view) {
          // Warmstart view (host spans from Cython) → CPU backend: copy directly, no CUDA needed
          solver_settings->get_pdlp_settings().get_cpu_pdlp_warm_start_data() =
            cpu_pdlp_warm_start_data_t<i_t, f_t>(solver_settings->get_pdlp_warm_start_data_view());
        } else {
          // GPU warmstart data (device_uvectors) → CPU backend: convert D2H
          auto& gpu_ws = solver_settings->get_pdlp_settings().get_pdlp_warm_start_data();
          cpu_pdlp_warm_start_data_t<i_t, f_t> cpu_warmstart =
            convert_to_cpu_warmstart(gpu_ws, gpu_ws.current_primal_solution_.stream());
          solver_settings->get_pdlp_settings().get_cpu_pdlp_warm_start_data() =
            std::move(cpu_warmstart);
        }
      }
    }
  }

  if (data_model->get_quadratic_objective_values().size() != 0 &&
      data_model->get_quadratic_objective_indices().size() != 0 &&
      data_model->get_quadratic_objective_offsets().size() != 0) {
    problem->set_quadratic_objective_matrix(data_model->get_quadratic_objective_values().data(),
                                            data_model->get_quadratic_objective_values().size(),
                                            data_model->get_quadratic_objective_indices().data(),
                                            data_model->get_quadratic_objective_indices().size(),
                                            data_model->get_quadratic_objective_offsets().data(),
                                            data_model->get_quadratic_objective_offsets().size());
  }

  if (data_model->get_variable_lower_bounds().size() != 0) {
    problem->set_variable_lower_bounds(data_model->get_variable_lower_bounds().data(),
                                       data_model->get_variable_lower_bounds().size());
  }

  if (data_model->get_variable_upper_bounds().size() != 0) {
    problem->set_variable_upper_bounds(data_model->get_variable_upper_bounds().data(),
                                       data_model->get_variable_upper_bounds().size());
  }

  if (data_model->get_row_types().size() != 0) {
    problem->set_row_types(data_model->get_row_types().data(), data_model->get_row_types().size());
  }

  if (data_model->get_constraint_lower_bounds().size() != 0) {
    problem->set_constraint_lower_bounds(data_model->get_constraint_lower_bounds().data(),
                                         data_model->get_constraint_lower_bounds().size());
  }

  if (data_model->get_constraint_upper_bounds().size() != 0) {
    problem->set_constraint_upper_bounds(data_model->get_constraint_upper_bounds().data(),
                                         data_model->get_constraint_upper_bounds().size());
  }

  if (data_model->get_variable_types().size() != 0) {
    std::vector<var_t> enum_variable_types(data_model->get_variable_types().size());
    std::transform(
      data_model->get_variable_types().data(),
      data_model->get_variable_types().data() + data_model->get_variable_types().size(),
      enum_variable_types.begin(),
      [](const auto val) -> var_t {
        return (val == 'I' || val == 'B') ? var_t::INTEGER : var_t::CONTINUOUS;
      });
    problem->set_variable_types(enum_variable_types.data(), enum_variable_types.size());
    // Problem category (LP/MIP/IP) is auto-detected by set_variable_types
  }

  if (data_model->get_variable_names().size() != 0) {
    problem->set_variable_names(data_model->get_variable_names());
  }

  if (data_model->get_row_names().size() != 0) {
    problem->set_row_names(data_model->get_row_names());
  }
}

}  // namespace cuopt::linear_programming
