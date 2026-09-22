/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "feasibility_jump.cuh"

#include "cpu/climber.hpp"

#include <dual_simplex/user_problem.hpp>
#include <math_optimization/tic_toc.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/pcgenerator.hpp>
#include <utilities/seed_generator.cuh>

#include <algorithm>
#include <cmath>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void init_fj_cpu_from_problem(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                              problem_t<i_t, f_t>& problem,
                              const raft::handle_t* handle_ptr,
                              std::vector<f_t> start_assignment,
                              const std::vector<f_t>& left_weights,
                              const std::vector<f_t>& right_weights,
                              f_t objective_weight,
                              const probing_cache_t<i_t, f_t>* probing_cache)
{
  auto problem_data                      = std::make_shared<fj_cpu_problem_t<i_t, f_t>>();
  fj_cpu.problem                         = problem_data;
  problem_data->tolerances               = problem.tolerances;
  problem_data->n_variables              = problem.n_variables;
  problem_data->n_constraints            = problem.n_constraints;
  problem_data->nnz                      = problem.nnz;
  const auto problem_view                = problem.view();
  problem_data->objective_scaling_factor = problem_view.objective_scaling_factor;
  problem_data->objective_offset         = problem_view.objective_offset;

  // Queue the device-to-host copies together and synchronize once before constructing host state.
  auto stream                        = handle_ptr->get_stream();
  const double download_start        = tic();
  problem_data->reverse_coefficients = cuopt::host_copy_async(problem.reverse_coefficients, stream);
  problem_data->reverse_constraints  = cuopt::host_copy_async(problem.reverse_constraints, stream);
  problem_data->reverse_offsets      = cuopt::host_copy_async(problem.reverse_offsets, stream);
  problem_data->coefficients         = cuopt::host_copy_async(problem.coefficients, stream);
  problem_data->offsets              = cuopt::host_copy_async(problem.offsets, stream);
  problem_data->variables            = cuopt::host_copy_async(problem.variables, stream);
  problem_data->h_obj_coeffs  = cuopt::host_copy_async(problem.objective_coefficients, stream);
  fj_cpu.h_var_bounds         = cuopt::host_copy_async(problem.variable_bounds, stream);
  problem_data->cstr_lb       = cuopt::host_copy_async(problem.constraint_lower_bounds, stream);
  problem_data->cstr_ub       = cuopt::host_copy_async(problem.constraint_upper_bounds, stream);
  problem_data->h_var_types   = cuopt::host_copy_async(problem.variable_types, stream);
  fj_cpu.h_is_binary_variable = cuopt::host_copy_async(problem.is_binary_variable, stream);
  fj_cpu.h_binary_indices     = cuopt::host_copy_async(problem.binary_indices, stream);
  problem_data->h_related_variables = cuopt::host_copy_async(problem.related_variables, stream);
  problem_data->h_related_variables_offsets =
    cuopt::host_copy_async(problem.related_variables_offsets, stream);
  handle_ptr->sync_stream();
  CUOPT_LOG_DEBUG(
    "CPUFJ model download from device: %.4fs for %d nnz", toc(download_start), problem.nnz);

  auto host_lp = std::make_shared<simplex::user_problem_t<i_t, f_t>>(handle_ptr);
  problem.get_host_user_problem(*host_lp);
  problem_data->host_lp                = std::move(host_lp);
  problem_data->probing_cache          = probing_cache;
  problem_data->h_original_ids         = problem.original_ids;
  problem_data->h_reverse_original_ids = problem.reverse_original_ids;

  fj_cpu.h_initial_left_weights  = left_weights;
  fj_cpu.h_initial_right_weights = right_weights;
  fj_cpu.max_weight              = f_t{1};
  fj_cpu.h_objective_weight      = objective_weight;
  if (start_assignment.empty()) {
    start_assignment.resize(problem.n_variables);
    for (i_t variable = 0; variable < problem.n_variables; ++variable) {
      const auto bounds = fj_cpu.h_var_bounds[variable].get();
      f_t value         = std::clamp(f_t{0}, get_lower(bounds), get_upper(bounds));
      if (fj_cpu.problem->h_var_types[variable] == var_t::INTEGER) { value = std::round(value); }
      start_assignment[variable] = value;
    }
  }
  cuopt_assert(start_assignment.size() == static_cast<size_t>(problem.n_variables),
               "start assignment must cover every variable");
  fj_cpu.h_assignment      = start_assignment;
  fj_cpu.h_best_assignment = std::move(start_assignment);
  fj_cpu.h_lhs.resize(problem.n_constraints);
  fj_cpu.h_lhs_sumcomp.resize(problem.n_constraints, f_t{0});
  fj_cpu.h_tabu_nodec_until.resize(problem.n_variables, 0);
  fj_cpu.h_tabu_noinc_until.resize(problem.n_variables, 0);
  fj_cpu.h_tabu_lastdec.resize(problem.n_variables, 0);
  fj_cpu.h_tabu_lastinc.resize(problem.n_variables, 0);
  fj_cpu.iterations = 0;

  finalize_fj_cpu_host_initialization(fj_cpu,
                                      *problem_data,
                                      problem.n_variables,
                                      problem.n_constraints,
                                      problem.n_integer_vars,
                                      problem.nnz,
                                      problem.tolerances);
}

template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> init_fj_cpu_from_optimization_problem(
  const optimization_problem_t<i_t, f_t>& problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings)
{
  raft::common::nvtx::range scope("init_fj_cpu_from_optimization_problem");
  auto stream = problem.get_handle_ptr()->get_stream();

  const double download_start = tic();
  auto coefficients = cuopt::host_copy_async(problem.get_constraint_matrix_values(), stream);
  auto variables    = cuopt::host_copy_async(problem.get_constraint_matrix_indices(), stream);
  auto offsets      = cuopt::host_copy_async(problem.get_constraint_matrix_offsets(), stream);
  auto objective_coefficients =
    cuopt::host_copy_async(problem.get_objective_coefficients(), stream);
  auto variable_lower_bounds = cuopt::host_copy_async(problem.get_variable_lower_bounds(), stream);
  auto variable_upper_bounds = cuopt::host_copy_async(problem.get_variable_upper_bounds(), stream);
  auto constraint_lower_bounds =
    cuopt::host_copy_async(problem.get_constraint_lower_bounds(), stream);
  auto constraint_upper_bounds =
    cuopt::host_copy_async(problem.get_constraint_upper_bounds(), stream);
  auto constraint_bounds = cuopt::host_copy_async(problem.get_constraint_bounds(), stream);
  auto row_types         = cuopt::host_copy_async(problem.get_row_types(), stream);
  auto variable_types    = cuopt::host_copy_async(problem.get_variable_types(), stream);
  problem.get_handle_ptr()->sync_stream();
  CUOPT_LOG_DEBUG(
    "CPUFJ model download from device: %.4fs for %d nnz", toc(download_start), problem.get_nnz());

  return init_fj_cpu_from_host_model<i_t, f_t>(problem.get_n_variables(),
                                               problem.get_n_constraints(),
                                               problem.get_nnz(),
                                               problem.get_sense(),
                                               problem.get_objective_scaling_factor(),
                                               problem.get_objective_offset(),
                                               std::move(coefficients),
                                               std::move(variables),
                                               std::move(offsets),
                                               std::move(objective_coefficients),
                                               std::move(variable_lower_bounds),
                                               std::move(variable_upper_bounds),
                                               std::move(constraint_lower_bounds),
                                               std::move(constraint_upper_bounds),
                                               std::move(constraint_bounds),
                                               std::move(row_types),
                                               std::move(variable_types),
                                               tolerances,
                                               preemption_flag,
                                               settings);
}

template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> fj_t<i_t, f_t>::create_cpu_climber(
  solution_t<i_t, f_t>& solution,
  const std::vector<f_t>& left_weights,
  const std::vector<f_t>& right_weights,
  f_t objective_weight,
  std::atomic<bool>& preemption_flag,
  const probing_cache_t<i_t, f_t>* probing_cache,
  fj_settings_t settings,
  bool randomize_params)
{
  raft::common::nvtx::range scope("fj_cpu_init");

  auto fj_cpu   = std::make_unique<fj_cpu_climber_t<i_t, f_t>>(preemption_flag);
  auto sol_copy = solution;
  clamp_within_var_bounds(sol_copy.assignment, solution.problem_ptr, solution.handle_ptr);

  init_fj_cpu_from_problem(*fj_cpu,
                           *solution.problem_ptr,
                           solution.handle_ptr,
                           sol_copy.get_host_assignment(),
                           left_weights,
                           right_weights,
                           objective_weight,
                           probing_cache);
  fj_cpu->settings = settings;
  if (randomize_params) {
    cuopt::pcgenerator_t rng(cuopt::seed_generator::get_seed());
    fj_cpu->mtm_viol_samples = rng.uniform<i_t>(15, 51);
    fj_cpu->mtm_sat_samples  = rng.uniform<i_t>(10, 31);
    fj_cpu->nnz_samples      = rng.uniform<i_t>(2000, 15001);
    fj_cpu->perturb_interval = rng.uniform<i_t>(50, 501);
  }
  return fj_cpu;
}

#if MIP_INSTANTIATE_FLOAT
template std::unique_ptr<fj_cpu_climber_t<int, float>> init_fj_cpu_from_optimization_problem(
  const optimization_problem_t<int, float>&,
  const typename mip_solver_settings_t<int, float>::tolerances_t&,
  std::atomic<bool>&,
  fj_settings_t);
template std::unique_ptr<fj_cpu_climber_t<int, float>> fj_t<int, float>::create_cpu_climber(
  solution_t<int, float>&,
  const std::vector<float>&,
  const std::vector<float>&,
  float,
  std::atomic<bool>&,
  const probing_cache_t<int, float>*,
  fj_settings_t,
  bool);
#endif

#if MIP_INSTANTIATE_DOUBLE
template std::unique_ptr<fj_cpu_climber_t<int, double>> init_fj_cpu_from_optimization_problem(
  const optimization_problem_t<int, double>&,
  const typename mip_solver_settings_t<int, double>::tolerances_t&,
  std::atomic<bool>&,
  fj_settings_t);
template std::unique_ptr<fj_cpu_climber_t<int, double>> fj_t<int, double>::create_cpu_climber(
  solution_t<int, double>&,
  const std::vector<double>&,
  const std::vector<double>&,
  double,
  std::atomic<bool>&,
  const probing_cache_t<int, double>*,
  fj_settings_t,
  bool);
#endif

}  // namespace cuopt::mathematical_optimization::mip
