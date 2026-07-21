/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/pdlp/pdlp_hyper_params.cuh>
#include <pdlp/pdhg.hpp>
#include <pdlp/swap_and_resize_helper.cuh>

#include <mip_heuristics/solution/solution.cuh>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::pdlp {

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
                                  mip::problem_t<i_t, f_t>& op_problem_scaled,
                                  i_t number_of_ruiz_iterations,
                                  f_t alpha,
                                  rmm::device_uvector<f_t>& A_T,
                                  rmm::device_uvector<i_t>& A_T_offsets,
                                  rmm::device_uvector<i_t>& A_T_indices,
                                  pdhg_solver_t<i_t, f_t>* pdhg_solver_ptr,
                                  const pdlp::pdlp_hyper_params_t& hyper_params,
                                  i_t original_batch_size,
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
  void unscale_solutions(mip::solution_t<i_t, f_t>& solution) const;
  const rmm::device_uvector<f_t>& get_constraint_matrix_scaling_vector() const;
  const rmm::device_uvector<f_t>& get_variable_scaling_vector() const;
  const mip::problem_t<i_t, f_t>& get_scaled_op_problem();

  f_t get_h_bound_rescaling() const;
  f_t get_h_objective_rescaling() const;
  const rmm::device_uvector<f_t>& get_bound_rescaling_vector() const;
  const rmm::device_uvector<f_t>& get_objective_rescaling_vector() const;
  void swap_context(const thrust::universal_host_pinned_vector<swap_pair_t<i_t>>& swap_pairs);
  void resize_context(i_t new_size);

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
  mip::problem_t<i_t, f_t>& op_problem_scaled_;

  rmm::device_uvector<f_t> iteration_constraint_matrix_scaling_;
  rmm::device_uvector<f_t> iteration_variable_scaling_;

  i_t original_batch_size_;
  rmm::device_uvector<f_t> bound_rescaling_;
  rmm::device_uvector<f_t> objective_rescaling_;
  // Since we need it on the host
  std::vector<f_t> h_bound_rescaling_;
  std::vector<f_t> h_objective_rescaling_;

  rmm::device_uvector<f_t> cummulative_constraint_matrix_scaling_;
  rmm::device_uvector<f_t> cummulative_variable_scaling_;
  pdhg_solver_t<i_t, f_t>* pdhg_solver_ptr_;
  rmm::device_uvector<f_t>& A_T_;
  rmm::device_uvector<i_t>& A_T_offsets_;
  rmm::device_uvector<i_t>& A_T_indices_;
  const pdlp::pdlp_hyper_params_t& hyper_params_;
  bool running_mip_;
};
}  // namespace cuopt::mathematical_optimization::pdlp
