/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuopt_java_native_api.hpp"

#include <cuopt/linear_programming/solver_settings.hpp>
#include <pdlp/cuopt_c_internal.hpp>

namespace {

using namespace cuopt::linear_programming;
using java_solver_settings_t =
  cuopt::linear_programming::solver_settings_t<cuopt_int_t, cuopt_float_t>;
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

struct warm_start_storage_t {
  std::vector<cuopt_float_t> current_primal_solution;
  std::vector<cuopt_float_t> current_dual_solution;
  std::vector<cuopt_float_t> initial_primal_average;
  std::vector<cuopt_float_t> initial_dual_average;
  std::vector<cuopt_float_t> current_aty;
  std::vector<cuopt_float_t> sum_primal_solutions;
  std::vector<cuopt_float_t> sum_dual_solutions;
  std::vector<cuopt_float_t> last_restart_primal_solution;
  std::vector<cuopt_float_t> last_restart_dual_solution;
};

std::mutex warm_start_mutex;
std::unordered_map<cuOptSolverSettings, std::unique_ptr<warm_start_storage_t>> warm_starts;

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

std::vector<cuopt_float_t> get_warm_start_vector(java_lp_solution_t* solution, cuopt_int_t field_id)
{
  switch (field_id) {
    case CUOPT_JAVA_PDLP_WARM_START_CURRENT_PRIMAL_SOLUTION:
      return solution->get_current_primal_solution_host();
    case CUOPT_JAVA_PDLP_WARM_START_CURRENT_DUAL_SOLUTION:
      return solution->get_current_dual_solution_host();
    case CUOPT_JAVA_PDLP_WARM_START_INITIAL_PRIMAL_AVERAGE:
      return solution->get_initial_primal_average_host();
    case CUOPT_JAVA_PDLP_WARM_START_INITIAL_DUAL_AVERAGE:
      return solution->get_initial_dual_average_host();
    case CUOPT_JAVA_PDLP_WARM_START_CURRENT_ATY: return solution->get_current_ATY_host();
    case CUOPT_JAVA_PDLP_WARM_START_SUM_PRIMAL_SOLUTIONS:
      return solution->get_sum_primal_solutions_host();
    case CUOPT_JAVA_PDLP_WARM_START_SUM_DUAL_SOLUTIONS:
      return solution->get_sum_dual_solutions_host();
    case CUOPT_JAVA_PDLP_WARM_START_LAST_RESTART_DUALITY_GAP_PRIMAL_SOLUTION:
      return solution->get_last_restart_duality_gap_primal_solution_host();
    case CUOPT_JAVA_PDLP_WARM_START_LAST_RESTART_DUALITY_GAP_DUAL_SOLUTION:
      return solution->get_last_restart_duality_gap_dual_solution_host();
    default: throw std::invalid_argument("Invalid PDLP warm-start vector field");
  }
}

}  // namespace

void cuopt_java_release_settings_state(cuOptSolverSettings settings)
{
  std::lock_guard<std::mutex> lock(warm_start_mutex);
  warm_starts.erase(settings);
}

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

extern "C" cuopt_int_t cuOptSetPDLPWarmStartData(
  cuOptSolverSettings settings,
  const cuopt_float_t* current_primal_solution,
  const cuopt_float_t* current_dual_solution,
  const cuopt_float_t* initial_primal_average,
  const cuopt_float_t* initial_dual_average,
  const cuopt_float_t* current_ATY,
  const cuopt_float_t* sum_primal_solutions,
  const cuopt_float_t* sum_dual_solutions,
  const cuopt_float_t* last_restart_duality_gap_primal_solution,
  const cuopt_float_t* last_restart_duality_gap_dual_solution,
  cuopt_int_t primal_size,
  cuopt_int_t dual_size,
  cuopt_float_t initial_primal_weight,
  cuopt_float_t initial_step_size,
  cuopt_int_t total_pdlp_iterations,
  cuopt_int_t total_pdhg_iterations,
  cuopt_float_t last_candidate_kkt_score,
  cuopt_float_t last_restart_kkt_score,
  cuopt_float_t sum_solution_weight,
  cuopt_int_t iterations_since_last_restart)
{
  if (settings == nullptr || primal_size <= 0 || dual_size <= 0) { return CUOPT_INVALID_ARGUMENT; }
  if (current_primal_solution == nullptr || current_dual_solution == nullptr ||
      initial_primal_average == nullptr || initial_dual_average == nullptr ||
      current_ATY == nullptr || sum_primal_solutions == nullptr || sum_dual_solutions == nullptr ||
      last_restart_duality_gap_primal_solution == nullptr ||
      last_restart_duality_gap_dual_solution == nullptr) {
    return CUOPT_INVALID_ARGUMENT;
  }

  auto storage = std::make_unique<warm_start_storage_t>();
  storage->current_primal_solution.assign(current_primal_solution,
                                          current_primal_solution + primal_size);
  storage->current_dual_solution.assign(current_dual_solution, current_dual_solution + dual_size);
  storage->initial_primal_average.assign(initial_primal_average,
                                         initial_primal_average + primal_size);
  storage->initial_dual_average.assign(initial_dual_average, initial_dual_average + dual_size);
  storage->current_aty.assign(current_ATY, current_ATY + primal_size);
  storage->sum_primal_solutions.assign(sum_primal_solutions, sum_primal_solutions + primal_size);
  storage->sum_dual_solutions.assign(sum_dual_solutions, sum_dual_solutions + dual_size);
  storage->last_restart_primal_solution.assign(
    last_restart_duality_gap_primal_solution,
    last_restart_duality_gap_primal_solution + primal_size);
  storage->last_restart_dual_solution.assign(last_restart_duality_gap_dual_solution,
                                             last_restart_duality_gap_dual_solution + dual_size);
  try {
    settings_from_handle(settings)->set_pdlp_warm_start_data(
      storage->current_primal_solution.data(),
      storage->current_dual_solution.data(),
      storage->initial_primal_average.data(),
      storage->initial_dual_average.data(),
      storage->current_aty.data(),
      storage->sum_primal_solutions.data(),
      storage->sum_dual_solutions.data(),
      storage->last_restart_primal_solution.data(),
      storage->last_restart_dual_solution.data(),
      primal_size,
      dual_size,
      initial_primal_weight,
      initial_step_size,
      total_pdlp_iterations,
      total_pdhg_iterations,
      last_candidate_kkt_score,
      last_restart_kkt_score,
      sum_solution_weight,
      iterations_since_last_restart);
  } catch (const std::exception&) {
    return CUOPT_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(warm_start_mutex);
  warm_starts[settings] = std::move(storage);
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

extern "C" cuopt_int_t cuOptHasPDLPWarmStartData(cuOptSolution solution,
                                                 cuopt_int_t* has_warm_start_ptr)
{
  auto* lp = get_lp_solution(solution);
  if (lp == nullptr || has_warm_start_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  try {
    *has_warm_start_ptr = static_cast<cuopt_int_t>(lp->has_warm_start_data());
  } catch (const std::exception&) {
    return CUOPT_RUNTIME_ERROR;
  }
  return CUOPT_SUCCESS;
}

extern "C" cuopt_int_t cuOptGetPDLPWarmStartVectorSize(cuOptSolution solution,
                                                       cuopt_int_t field_id,
                                                       cuopt_int_t* size_ptr)
{
  auto* lp = get_lp_solution(solution);
  if (lp == nullptr || size_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  try {
    *size_ptr = static_cast<cuopt_int_t>(get_warm_start_vector(lp, field_id).size());
  } catch (const std::invalid_argument&) {
    return CUOPT_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return CUOPT_RUNTIME_ERROR;
  }
  return CUOPT_SUCCESS;
}

extern "C" cuopt_int_t cuOptGetPDLPWarmStartVector(cuOptSolution solution,
                                                   cuopt_int_t field_id,
                                                   cuopt_float_t* values)
{
  auto* lp = get_lp_solution(solution);
  if (lp == nullptr || values == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  try {
    const auto vector = get_warm_start_vector(lp, field_id);
    if (!vector.empty()) {
      std::memcpy(values, vector.data(), vector.size() * sizeof(cuopt_float_t));
    }
  } catch (const std::invalid_argument&) {
    return CUOPT_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return CUOPT_RUNTIME_ERROR;
  }
  return CUOPT_SUCCESS;
}

extern "C" cuopt_int_t cuOptGetPDLPWarmStartScalar(cuOptSolution solution,
                                                   cuopt_int_t field_id,
                                                   cuopt_float_t* value_ptr)
{
  auto* lp = get_lp_solution(solution);
  if (lp == nullptr || value_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  try {
    switch (field_id) {
      case CUOPT_JAVA_PDLP_WARM_START_INITIAL_PRIMAL_WEIGHT:
        *value_ptr = lp->get_initial_primal_weight();
        break;
      case CUOPT_JAVA_PDLP_WARM_START_INITIAL_STEP_SIZE:
        *value_ptr = lp->get_initial_step_size();
        break;
      case CUOPT_JAVA_PDLP_WARM_START_LAST_CANDIDATE_KKT_SCORE:
        *value_ptr = lp->get_last_candidate_kkt_score();
        break;
      case CUOPT_JAVA_PDLP_WARM_START_LAST_RESTART_KKT_SCORE:
        *value_ptr = lp->get_last_restart_kkt_score();
        break;
      case CUOPT_JAVA_PDLP_WARM_START_SUM_SOLUTION_WEIGHT:
        *value_ptr = lp->get_sum_solution_weight();
        break;
      default: return CUOPT_INVALID_ARGUMENT;
    }
  } catch (const std::exception&) {
    return CUOPT_RUNTIME_ERROR;
  }
  return CUOPT_SUCCESS;
}

extern "C" cuopt_int_t cuOptGetPDLPWarmStartInteger(cuOptSolution solution,
                                                    cuopt_int_t field_id,
                                                    cuopt_int_t* value_ptr)
{
  auto* lp = get_lp_solution(solution);
  if (lp == nullptr || value_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  try {
    switch (field_id) {
      case CUOPT_JAVA_PDLP_WARM_START_TOTAL_PDLP_ITERATIONS:
        *value_ptr = lp->get_total_pdlp_iterations();
        break;
      case CUOPT_JAVA_PDLP_WARM_START_TOTAL_PDHG_ITERATIONS:
        *value_ptr = lp->get_total_pdhg_iterations();
        break;
      case CUOPT_JAVA_PDLP_WARM_START_ITERATIONS_SINCE_LAST_RESTART:
        *value_ptr = lp->get_iterations_since_last_restart();
        break;
      default: return CUOPT_INVALID_ARGUMENT;
    }
  } catch (const std::exception&) {
    return CUOPT_RUNTIME_ERROR;
  }
  return CUOPT_SUCCESS;
}
