/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuopt_java_native_api.hpp"

#include <cuopt/mathematical_optimization/solver_settings.hpp>
#include <pdlp/cuopt_c_internal.hpp>

#include <cstdio>
#include <exception>

namespace {

using namespace cuopt::mathematical_optimization;
using java_solver_settings_t =
  cuopt::mathematical_optimization::solver_settings_t<cuopt_int_t, cuopt_float_t>;
using java_lp_solution_t  = lp_solution_interface_t<cuopt_int_t, cuopt_float_t>;
using java_mip_solution_t = mip_solution_interface_t<cuopt_int_t, cuopt_float_t>;

// cuOpt's C API intentionally keeps its settings handle opaque. The first member is the
// solver_settings_t pointer; this private view avoids adding Java-only fields to the public C API.
struct settings_handle_prefix_t {
  java_solver_settings_t* settings;
};

java_solver_settings_t* settings_from_handle(cuOptSolverSettings handle)
{
  return handle == nullptr ? nullptr : static_cast<settings_handle_prefix_t*>(handle)->settings;
}

java_lp_solution_t* get_lp_solution(cuOptSolution solution)
{
  auto* view = static_cast<solution_and_stream_view_t*>(solution);
  return view == nullptr || view->is_mip ? nullptr : view->lp_solution_interface_ptr;
}

java_mip_solution_t* get_mip_solution(cuOptSolution solution)
{
  auto* view = static_cast<solution_and_stream_view_t*>(solution);
  return view == nullptr || !view->is_mip ? nullptr : view->mip_solution_interface_ptr;
}

}  // namespace

extern "C" cuopt_int_t cuOptLoadParametersFromFile(cuOptSolverSettings settings, const char* path)
{
  if (settings == nullptr || path == nullptr || path[0] == '\0') { return CUOPT_INVALID_ARGUMENT; }
  try {
    settings_from_handle(settings)->load_parameters_from_file(path);
  } catch (const std::exception&) {
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

extern "C" cuopt_int_t cuOptDumpParametersToFile(cuOptSolverSettings settings,
                                                 const char* path,
                                                 cuopt_int_t hyperparameters_only,
                                                 cuopt_int_t* dumped_successfully)
{
  if (settings == nullptr || path == nullptr || path[0] == '\0' || dumped_successfully == nullptr) {
    return CUOPT_INVALID_ARGUMENT;
  }
  try {
    *dumped_successfully = static_cast<cuopt_int_t>(
      settings_from_handle(settings)->dump_parameters_to_file(path, hyperparameters_only != 0));
  } catch (const std::exception&) {
    return CUOPT_INVALID_ARGUMENT;
  }
  return CUOPT_SUCCESS;
}

extern "C" cuopt_int_t cuOptGetNumSolverParameters(cuopt_int_t* num_parameters_ptr)
{
  if (num_parameters_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  try {
    java_solver_settings_t settings;
    *num_parameters_ptr = static_cast<cuopt_int_t>(settings.get_parameter_names().size());
  } catch (const std::exception&) {
    return CUOPT_RUNTIME_ERROR;
  }
  return CUOPT_SUCCESS;
}

extern "C" cuopt_int_t cuOptGetSolverParameterName(cuopt_int_t index,
                                                   cuopt_int_t parameter_name_size,
                                                   char* parameter_name)
{
  if (index < 0 || parameter_name_size <= 0 || parameter_name == nullptr) {
    return CUOPT_INVALID_ARGUMENT;
  }
  try {
    java_solver_settings_t settings;
    const auto names = settings.get_parameter_names();
    if (index >= static_cast<cuopt_int_t>(names.size())) { return CUOPT_INVALID_ARGUMENT; }
    std::snprintf(parameter_name,
                  static_cast<size_t>(parameter_name_size),
                  "%s",
                  names[static_cast<size_t>(index)].c_str());
  } catch (const std::exception&) {
    return CUOPT_RUNTIME_ERROR;
  }
  return CUOPT_SUCCESS;
}

extern "C" cuopt_int_t cuOptSolutionIsMIP(cuOptSolution solution, cuopt_int_t* is_mip_ptr)
{
  if (solution == nullptr || is_mip_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  *is_mip_ptr =
    static_cast<cuopt_int_t>(static_cast<solution_and_stream_view_t*>(solution)->is_mip);
  return CUOPT_SUCCESS;
}

extern "C" cuopt_int_t cuOptGetLPSolverStats(cuOptSolution solution,
                                             cuopt_float_t* primal_residual_ptr,
                                             cuopt_float_t* dual_residual_ptr,
                                             cuopt_float_t* gap_ptr,
                                             cuopt_int_t* num_iterations_ptr,
                                             cuopt_int_t* solved_by_ptr)
{
  auto* lp = get_lp_solution(solution);
  if (lp == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  try {
    if (primal_residual_ptr != nullptr) { *primal_residual_ptr = lp->get_l2_primal_residual(); }
    if (dual_residual_ptr != nullptr) { *dual_residual_ptr = lp->get_l2_dual_residual(); }
    if (gap_ptr != nullptr) { *gap_ptr = lp->get_gap(); }
    if (num_iterations_ptr != nullptr) { *num_iterations_ptr = lp->get_num_iterations(); }
    if (solved_by_ptr != nullptr) { *solved_by_ptr = static_cast<cuopt_int_t>(lp->solved_by()); }
  } catch (const std::exception&) {
    return CUOPT_RUNTIME_ERROR;
  }
  return CUOPT_SUCCESS;
}

extern "C" cuopt_int_t cuOptGetMIPSolverStats(cuOptSolution solution,
                                              cuopt_float_t* presolve_time_ptr,
                                              cuopt_float_t* max_constraint_violation_ptr,
                                              cuopt_float_t* max_int_violation_ptr,
                                              cuopt_float_t* max_variable_bound_violation_ptr,
                                              cuopt_int_t* num_nodes_ptr,
                                              cuopt_int_t* num_simplex_iterations_ptr)
{
  auto* mip = get_mip_solution(solution);
  if (mip == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  try {
    if (presolve_time_ptr != nullptr) { *presolve_time_ptr = mip->get_presolve_time(); }
    if (max_constraint_violation_ptr != nullptr) {
      *max_constraint_violation_ptr = mip->get_max_constraint_violation();
    }
    if (max_int_violation_ptr != nullptr) { *max_int_violation_ptr = mip->get_max_int_violation(); }
    if (max_variable_bound_violation_ptr != nullptr) {
      *max_variable_bound_violation_ptr = mip->get_max_variable_bound_violation();
    }
    if (num_nodes_ptr != nullptr) { *num_nodes_ptr = mip->get_num_nodes(); }
    if (num_simplex_iterations_ptr != nullptr) {
      *num_simplex_iterations_ptr = mip->get_num_simplex_iterations();
    }
  } catch (const std::exception&) {
    return CUOPT_RUNTIME_ERROR;
  }
  return CUOPT_SUCCESS;
}
