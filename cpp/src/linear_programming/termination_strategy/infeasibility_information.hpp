/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <linear_programming/cusparse_view.hpp>
#include <linear_programming/initial_scaling_strategy/initial_scaling.cuh>
#include <linear_programming/pdhg.hpp>
#include <linear_programming/saddle_point.hpp>

#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/utilities/segmented_sum_handler.cuh>

#include <mip/problem/problem.cuh>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
class infeasibility_information_t {
 public:
  infeasibility_information_t(
    raft::handle_t const* handle_ptr,
    problem_t<i_t, f_t>& op_problem,
    const problem_t<i_t, f_t>& op_problem_scaled,  // Only used for cuPDLPx infeasibility detection
    cusparse_view_t<i_t, f_t>& cusparse_view,
    const cusparse_view_t<i_t, f_t>& scaled_cusparse_view,
    i_t primal_size,
    i_t dual_size,
    const pdlp_initial_scaling_strategy_t<i_t, f_t>&
      scaling_strategy,  // Only used for cuPDLPx infeasibility detection
    bool infeasibility_detection,
    const std::vector<pdlp_climber_strategy_t>& climber_strategies,
    const pdlp_hyper_params::pdlp_hyper_params_t& hyper_params);

  void compute_infeasibility_information(pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
                                         rmm::device_uvector<f_t>& primal_ray,
                                         rmm::device_uvector<f_t>& dual_ray);

  struct view_t {
    f_t* primal_ray_inf_norm;
    f_t* primal_ray_max_violation;
    raft::device_span<f_t> max_primal_ray_infeasibility;
    raft::device_span<f_t> primal_ray_linear_objective;

    f_t* dual_ray_inf_norm;
    raft::device_span<f_t> max_dual_ray_infeasibility;
    raft::device_span<f_t> dual_ray_linear_objective;

    f_t* reduced_cost_inf_norm;

    f_t* homogenous_primal_residual;
    f_t* homogenous_dual_residual;
    f_t* reduced_cost;
  };  // struct view_t

  /**
   * @brief Gets the device-side view (with raw pointers), for ease of access
   *        inside cuda kernels
   */
  view_t view();

 private:
  void compute_homogenous_primal_residual(cusparse_view_t<i_t, f_t>& cusparse_view,
                                          rmm::device_uvector<f_t>& tmp_dual);

  void compute_max_violation(rmm::device_uvector<f_t>& primal_ray);

  void compute_homogenous_primal_objective(rmm::device_uvector<f_t>& primal_ray);

  void compute_homogenous_dual_residual(cusparse_view_t<i_t, f_t>& cusparse_view,
                                        rmm::device_uvector<f_t>& tmp_primal,
                                        rmm::device_uvector<f_t>& primal_ray);
  void compute_homogenous_dual_objective(rmm::device_uvector<f_t>& dual_ray);
  void compute_reduced_cost_from_primal_gradient(rmm::device_uvector<f_t>& primal_gradient,
                                                 rmm::device_uvector<f_t>& primal_ray);
  void compute_reduced_costs_dual_objective_contribution();

  raft::handle_t const* handle_ptr_{nullptr};
  rmm::cuda_stream_view stream_view_;

  i_t primal_size_h_;
  i_t dual_size_h_;

  problem_t<i_t, f_t>* problem_ptr;
  cusparse_view_t<i_t, f_t>& op_problem_cusparse_view_;
  const cusparse_view_t<i_t, f_t>& scaled_cusparse_view_;

  rmm::device_uvector<f_t> primal_ray_inf_norm_;
  rmm::device_scalar<f_t> primal_ray_inf_norm_inverse_;
  rmm::device_scalar<f_t> neg_primal_ray_inf_norm_inverse_;
  rmm::device_scalar<f_t> primal_ray_max_violation_;
  rmm::device_uvector<f_t> max_primal_ray_infeasibility_;
  rmm::device_uvector<f_t> primal_ray_linear_objective_;

  rmm::device_uvector<f_t> dual_ray_inf_norm_;
  rmm::device_uvector<f_t> max_dual_ray_infeasibility_;
  rmm::device_uvector<f_t> dual_ray_linear_objective_;
  rmm::device_scalar<f_t> reduced_cost_dual_objective_;

  rmm::device_scalar<f_t> reduced_cost_inf_norm_;

  // used for computations and can be reused
  rmm::device_uvector<f_t> homogenous_primal_residual_;
  rmm::device_uvector<f_t> homogenous_dual_residual_;
  rmm::device_uvector<f_t> reduced_cost_;
  rmm::device_uvector<f_t> bound_value_;
  rmm::device_uvector<f_t> homogenous_dual_lower_bounds_;
  rmm::device_uvector<f_t> homogenous_dual_upper_bounds_;

  // Used for cuPDLPx infeasibility detection
  rmm::device_uvector<f_t> primal_slack_;
  rmm::device_uvector<f_t> dual_slack_;
  rmm::device_uvector<f_t> sum_primal_slack_;
  rmm::device_uvector<f_t> sum_dual_slack_;

  rmm::device_buffer rmm_tmp_buffer_;
  size_t size_of_buffer_;

  const rmm::device_scalar<f_t> reusable_device_scalar_value_1_;
  const rmm::device_scalar<f_t> reusable_device_scalar_value_0_;
  const rmm::device_scalar<f_t> reusable_device_scalar_value_neg_1_;

  const pdlp_initial_scaling_strategy_t<i_t, f_t>& scaling_strategy_;
  const problem_t<i_t, f_t>& op_problem_scaled_;

  segmented_sum_handler_t<i_t, f_t> segmented_sum_handler_;
  const std::vector<pdlp_climber_strategy_t>& climber_strategies_;
  const pdlp_hyper_params::pdlp_hyper_params_t& hyper_params_;
};
}  // namespace cuopt::linear_programming::detail
