/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include <rmm/device_uvector.hpp>
#include "dual_simplex/dense_vector.hpp"
#include "dual_simplex/pinned_host_allocator.hpp"
#include "dual_simplex/presolve.hpp"
#include "dual_simplex/simplex_solver_settings.hpp"
#include "dual_simplex/solution.hpp"
#include "dual_simplex/solve.hpp"
#include "dual_simplex/sparse_matrix.hpp"
#include "dual_simplex/tic_toc.hpp"

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
struct barrier_solver_settings_t {
  i_t iteration_limit = 1000;
  f_t step_scale      = 0.9;
};

template <typename i_t, typename f_t>
class iteration_data_t;  // Forward declare

template <typename i_t, typename f_t>
class barrier_solver_t {
 public:
  barrier_solver_t(const lp_problem_t<i_t, f_t>& lp,
                   const presolve_info_t<i_t, f_t>& presolve,
                   const simplex_solver_settings_t<i_t, f_t>& settings);
  lp_status_t solve(f_t start_time,
                    const barrier_solver_settings_t<i_t, f_t>& options,
                    lp_solution_t<i_t, f_t>& solution);

 private:
  void my_pop_range(bool debug) const;
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

  template <typename AllocatorA>
  void compute_residuals(const dense_vector_t<i_t, f_t, AllocatorA>& w,
                         const dense_vector_t<i_t, f_t, AllocatorA>& x,
                         const dense_vector_t<i_t, f_t, AllocatorA>& y,
                         const dense_vector_t<i_t, f_t, AllocatorA>& v,
                         const dense_vector_t<i_t, f_t, AllocatorA>& z,
                         iteration_data_t<i_t, f_t>& data);
  template <typename AllocatorA, typename AllocatorB>
  f_t max_step_to_boundary(const dense_vector_t<i_t, f_t, AllocatorA>& x,
                           const dense_vector_t<i_t, f_t, AllocatorB>& dx,
                           i_t& index) const;
  template <typename AllocatorA, typename AllocatorB>
  f_t max_step_to_boundary(const dense_vector_t<i_t, f_t, AllocatorA>& x,
                           const dense_vector_t<i_t, f_t, AllocatorB>& dx) const;
  // To be able to directly pass lambdas to transform functions
 public:
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

  f_t gpu_max_step_to_boundary(iteration_data_t<i_t, f_t>& data,
                               const rmm::device_uvector<f_t>& x,
                               const rmm::device_uvector<f_t>& dx);
  i_t gpu_compute_search_direction(iteration_data_t<i_t, f_t>& data,
                                   pinned_dense_vector_t<i_t, f_t>& dw,
                                   pinned_dense_vector_t<i_t, f_t>& dx,
                                   pinned_dense_vector_t<i_t, f_t>& dy,
                                   pinned_dense_vector_t<i_t, f_t>& dv,
                                   pinned_dense_vector_t<i_t, f_t>& dz,
                                   f_t& max_residual);
  i_t compute_search_direction(iteration_data_t<i_t, f_t>& data,
                               pinned_dense_vector_t<i_t, f_t>& dw,
                               pinned_dense_vector_t<i_t, f_t>& dx,
                               pinned_dense_vector_t<i_t, f_t>& dy,
                               pinned_dense_vector_t<i_t, f_t>& dv,
                               pinned_dense_vector_t<i_t, f_t>& dz,
                               f_t& max_residual);

 private:
  lp_status_t check_for_suboptimal_solution(const barrier_solver_settings_t<i_t, f_t>& options,
                                            iteration_data_t<i_t, f_t>& data,
                                            f_t start_time,
                                            i_t iter,
                                            f_t& primal_objective,
                                            f_t& primal_residual_norm,
                                            f_t& dual_residual_norm,
                                            f_t& complementarity_residual_norm,
                                            f_t& relative_primal_residual,
                                            f_t& relative_dual_residual,
                                            f_t& relative_complementarity_residual,
                                            lp_solution_t<i_t, f_t>& solution);

  const lp_problem_t<i_t, f_t>& lp;
  const simplex_solver_settings_t<i_t, f_t>& settings;
  const presolve_info_t<i_t, f_t>& presolve_info;

  rmm::device_uvector<f_t> d_diag_;
  rmm::device_uvector<f_t> d_bound_rhs_;
  rmm::device_uvector<f_t> d_x_;
  rmm::device_uvector<f_t> d_z_;
  rmm::device_uvector<f_t> d_w_;
  rmm::device_uvector<f_t> d_v_;
  rmm::device_uvector<f_t> d_h_;
  rmm::device_uvector<f_t> d_y_;
  rmm::device_uvector<i_t> d_upper_bounds_;
  rmm::device_uvector<f_t> d_tmp3_;
  rmm::device_uvector<f_t> d_tmp4_;
  rmm::device_uvector<f_t> d_r1_;
  rmm::device_uvector<f_t> d_r1_prime_;
  rmm::device_uvector<f_t> d_dx_;
  rmm::device_uvector<f_t> d_dy_;
  rmm::device_uvector<f_t> d_c_;

  rmm::device_uvector<f_t> d_dual_residual_;
  rmm::device_uvector<f_t> d_complementarity_xz_rhs_;
  rmm::device_uvector<f_t> d_complementarity_wv_rhs_;
  rmm::device_uvector<f_t> d_dual_rhs_;
  rmm::device_uvector<f_t> d_dz_;
  rmm::device_uvector<f_t> d_dv_;
  rmm::device_uvector<f_t> d_y_residual_;
  rmm::device_uvector<f_t> d_dx_residual_;
  rmm::device_uvector<f_t> d_xz_residual_;
  rmm::device_uvector<f_t> d_dw_;
  rmm::device_uvector<f_t> d_dw_residual_;
  rmm::device_uvector<f_t> d_wv_residual_;
  // After Compute search direction
  rmm::device_uvector<f_t> d_dw_aff_;
  rmm::device_uvector<f_t> d_dx_aff_;
  rmm::device_uvector<f_t> d_dv_aff_;
  rmm::device_uvector<f_t> d_dz_aff_;
  rmm::device_uvector<f_t> d_dy_aff_;
  // GPU ADAT multiply
  rmm::device_uvector<f_t> d_u_;
  // Compute residuals
  rmm::device_uvector<f_t> d_primal_residual_;
  rmm::device_uvector<f_t> d_bound_residual_;
  rmm::device_uvector<f_t> d_upper_;
  rmm::device_uvector<f_t> d_complementarity_xz_residual_;
  rmm::device_uvector<f_t> d_complementarity_wv_residual_;

  pinned_dense_vector_t<i_t, f_t> restrict_u_;

  rmm::cuda_stream_view stream_view_;
};

}  // namespace cuopt::linear_programming::dual_simplex
