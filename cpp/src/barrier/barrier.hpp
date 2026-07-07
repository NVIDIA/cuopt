/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <linear_algebra/dense_vector.hpp>

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/solve.hpp>
#include <linear_algebra/sparse_matrix.hpp>
#include <math_optimization/tic_toc.hpp>

#include <rmm/device_uvector.hpp>
namespace cuopt::mathematical_optimization::barrier {

/** Validates SOC layout on an simplex::lp_problem_t before barrier presolve/solve. */
template <typename i_t, typename f_t>
bool validate_barrier_cone_layout(const simplex::lp_problem_t<i_t, f_t>& problem,
                                  const simplex::simplex_solver_settings_t<i_t, f_t>& settings);

template <typename i_t, typename f_t>
class iteration_data_t;  // Forward declare

template <typename i_t, typename f_t>
class barrier_solver_t {
 public:
  barrier_solver_t(const simplex::lp_problem_t<i_t, f_t>& lp,
                   const simplex::presolve_info_t<i_t, f_t>& presolve,
                   const simplex::simplex_solver_settings_t<i_t, f_t>& settings);
  simplex::lp_status_t solve(f_t start_time, simplex::lp_solution_t<i_t, f_t>& solution);

 private:
  void my_pop_range(bool debug) const;
  void create_Q(const simplex::lp_problem_t<i_t, f_t>& lp, csc_matrix_t<i_t, f_t>& Q);
  int initial_point(iteration_data_t<i_t, f_t>& data);
  void compute_residual_norms(const dense_vector_t<i_t, f_t>& w,
                              const dense_vector_t<i_t, f_t>& x,
                              const dense_vector_t<i_t, f_t>& y,
                              const dense_vector_t<i_t, f_t>& v,
                              const dense_vector_t<i_t, f_t>& z,
                              iteration_data_t<i_t, f_t>& data,
                              f_t& primal_residual_norm,
                              f_t& dual_residual_norm,
                              f_t& complementarity_residual_norm);

  void compute_primal_dual_step_length(iteration_data_t<i_t, f_t>& data,
                                       f_t step_scale,
                                       f_t& step_primal,
                                       f_t& step_dual);

  void compute_residual_norms(iteration_data_t<i_t, f_t>& data,
                              f_t& primal_residual_norm,
                              f_t& dual_residual_norm,
                              f_t& complementarity_residual_norm);
  void compute_mu(iteration_data_t<i_t, f_t>& data, f_t& mu);
  void compute_primal_dual_objective(iteration_data_t<i_t, f_t>& data,
                                     f_t& primal_objective,
                                     f_t& dual_objective);

  // To be able to directly pass lambdas to transform functions
 public:
  void compute_next_iterate(iteration_data_t<i_t, f_t>& data,
                            f_t step_scale,
                            f_t step_primal,
                            f_t step_dual);
  void compute_final_direction(iteration_data_t<i_t, f_t>& data);
  void compute_cc_rhs(iteration_data_t<i_t, f_t>& data, f_t& new_mu);
  void compute_target_mu(
    iteration_data_t<i_t, f_t>& data, f_t mu, f_t& mu_aff, f_t& sigma, f_t& new_mu);
  void compute_affine_rhs(iteration_data_t<i_t, f_t>& data);
  void gpu_compute_residuals(rmm::device_uvector<f_t> const& d_w,
                             rmm::device_uvector<f_t> const& d_x,
                             rmm::device_uvector<f_t> const& d_y,
                             rmm::device_uvector<f_t> const& d_v,
                             rmm::device_uvector<f_t> const& d_z,
                             iteration_data_t<i_t, f_t>& data);
  void gpu_compute_residual_norms(const rmm::device_uvector<f_t>& d_w,
                                  const rmm::device_uvector<f_t>& d_x,
                                  const rmm::device_uvector<f_t>& d_y,
                                  const rmm::device_uvector<f_t>& d_v,
                                  const rmm::device_uvector<f_t>& d_z,
                                  iteration_data_t<i_t, f_t>& data,
                                  f_t& primal_residual_norm,
                                  f_t& dual_residual_norm,
                                  f_t& complementarity_residual_norm);

  f_t compute_nonnegative_step_length(iteration_data_t<i_t, f_t>& data,
                                      const rmm::device_uvector<f_t>& x,
                                      const rmm::device_uvector<f_t>& dx);
  i_t gpu_compute_search_direction(iteration_data_t<i_t, f_t>& data,
                                   f_t& dual_perturb,
                                   f_t& primal_perturb,
                                   f_t& max_residual);

 private:
  simplex::lp_status_t check_for_suboptimal_solution(iteration_data_t<i_t, f_t>& data,
                                                     f_t start_time,
                                                     i_t iter,
                                                     f_t& primal_objective,
                                                     f_t& primal_residual_norm,
                                                     f_t& dual_residual_norm,
                                                     f_t& complementarity_residual_norm,
                                                     f_t& relative_primal_residual,
                                                     f_t& relative_dual_residual,
                                                     f_t& relative_complementarity_residual,
                                                     simplex::lp_solution_t<i_t, f_t>& solution);

  const simplex::lp_problem_t<i_t, f_t>& lp;
  const simplex::simplex_solver_settings_t<i_t, f_t>& settings;
  const simplex::presolve_info_t<i_t, f_t>& presolve_info;
  rmm::cuda_stream_view stream_view_;
};

}  // namespace cuopt::mathematical_optimization::barrier
