/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <linear_programming/pdhg.hpp>

#include <mip/solution/solution.cuh>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class pdlp_initial_scaling_strategy_t {
 public:
  /**
   * @brief A device-side view of the `pdlp_initial_scaling_strategy_t` structure with the RAII
   * stuffs stripped out, to make it easy to work inside kernels
   *
   * @note It is assumed that the pointers are NOT owned by this class, but rather
   *       by the encompassing `pdlp_initial_scaling_strategy_t` class via RAII abstractions like
   *       `rmm::device_uvector`
   */
  struct view_t {
    /** size of the primal problem */
    i_t primal_size;
    /** size of the dual problem */
    i_t dual_size;

    raft::device_span<f_t> iteration_constraint_matrix_scaling;
    raft::device_span<f_t> iteration_variable_scaling;
    raft::device_span<f_t> cummulative_constraint_matrix_scaling;
    raft::device_span<f_t> cummulative_variable_scaling;
  };  // struct view_t

  pdlp_initial_scaling_strategy_t(raft::handle_t const* handle_ptr,
                                  problem_t<i_t, f_t>& op_problem_scaled,
                                  i_t number_of_ruiz_iterations,
                                  f_t alpha,
                                  rmm::device_uvector<f_t>& A_T,
                                  rmm::device_uvector<i_t>& A_T_offsets,
                                  rmm::device_uvector<i_t>& A_T_indices,
                                  pdhg_solver_t<i_t, f_t>* pdhg_solver_ptr,
                                  bool running_mip = false);

  void scale_problem();

  void scale_solutions(rmm::device_uvector<f_t>& primal_solution) const;
  void scale_solutions(rmm::device_uvector<f_t>& primal_solution,
                       rmm::device_uvector<f_t>& dual_solution) const;
  void scale_solutions(rmm::device_uvector<f_t>& primal_solution,
                       rmm::device_uvector<f_t>& dual_solution,
                       rmm::device_uvector<f_t>& dual_slack) const;
  void scale_primal(rmm::device_uvector<f_t>& primal_solution) const;
  void scale_dual(rmm::device_uvector<f_t>& dual_solution) const;
  void unscale_solutions(rmm::device_uvector<f_t>& primal_solution,
                         rmm::device_uvector<f_t>& dual_solution) const;
  void unscale_solutions(rmm::device_uvector<f_t>& primal_solution,
                         rmm::device_uvector<f_t>& dual_solution,
                         rmm::device_uvector<f_t>& dual_slack) const;
  void unscale_solutions(solution_t<i_t, f_t>& solution) const;
  rmm::device_uvector<f_t>& get_constraint_matrix_scaling_vector();
  rmm::device_uvector<f_t>& get_variable_scaling_vector();
  const problem_t<i_t, f_t>& get_scaled_op_problem();

  void bound_objective_rescaling();

  /**
   * @brief Gets the device-side view (with raw pointers), for ease of access
   *        inside cuda kernels
   */
  view_t view();

 private:
  void compute_scaling_vectors(i_t number_of_ruiz_iterations, f_t alpha);
  void ruiz_inf_scaling(i_t number_of_ruiz_iterations);
  void pock_chambolle_scaling(f_t alpha);
  void reset_integer_variables();

  raft::handle_t const* handle_ptr_{nullptr};
  rmm::cuda_stream_view stream_view_;

  i_t primal_size_h_;
  i_t dual_size_h_;
  problem_t<i_t, f_t>& op_problem_scaled_;

  rmm::device_uvector<f_t> iteration_constraint_matrix_scaling_;
  rmm::device_uvector<f_t> iteration_variable_scaling_;

  rmm::device_scalar<f_t> bound_rescaling_;
  rmm::device_scalar<f_t> objective_rescaling_;
  // Since we need it on the host
  f_t h_bound_rescaling     = std::numeric_limits<f_t>::signaling_NaN();
  f_t h_objective_rescaling = std::numeric_limits<f_t>::signaling_NaN();

  rmm::device_uvector<f_t> cummulative_constraint_matrix_scaling_;
  rmm::device_uvector<f_t> cummulative_variable_scaling_;
  pdhg_solver_t<i_t, f_t>* pdhg_solver_ptr_;
  rmm::device_uvector<f_t>& A_T_;
  rmm::device_uvector<i_t>& A_T_offsets_;
  rmm::device_uvector<i_t>& A_T_indices_;
  bool running_mip_;
};
}  // namespace cuopt::linear_programming::detail
