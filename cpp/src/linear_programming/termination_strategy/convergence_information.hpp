/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <linear_programming/cusparse_view.hpp>
#include <linear_programming/pdhg.hpp>
#include <linear_programming/saddle_point.hpp>

#include <cuopt/linear_programming/pdlp/solver_settings.hpp>

#include <mip/problem/problem.cuh>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
class convergence_information_t {
 public:
  convergence_information_t(raft::handle_t const* handle_ptr,
                            problem_t<i_t, f_t>& op_problem,
                            cusparse_view_t<i_t, f_t>& cusparse_view,
                            i_t primal_size,
                            i_t dual_size);

  void compute_convergence_information(
    pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
    rmm::device_uvector<f_t>& primal_iterate,
    rmm::device_uvector<f_t>& dual_iterate,
    [[maybe_unused]] const rmm::device_uvector<f_t>& dual_slack,
    const rmm::device_uvector<f_t>& combined_bounds,  // Only useful if per_constraint_residual
    const rmm::device_uvector<f_t>&
      objective_coefficients,  // Only useful if per_constraint_residual
    const pdlp_solver_settings_t<i_t, f_t>& settings);

  rmm::device_uvector<f_t>& get_reduced_cost();

  // Needed for kkt restart & debug prints
  const rmm::device_scalar<f_t>& get_primal_objective() const;
  const rmm::device_scalar<f_t>& get_dual_objective() const;
  const rmm::device_scalar<f_t>& get_l2_primal_residual() const;
  const rmm::device_scalar<f_t>& get_l2_dual_residual() const;
  const rmm::device_scalar<f_t>& get_relative_linf_primal_residual() const;
  const rmm::device_scalar<f_t>& get_relative_linf_dual_residual() const;
  const rmm::device_scalar<f_t>& get_gap() const;
  f_t get_relative_gap_value() const;
  f_t get_relative_l2_primal_residual_value() const;
  f_t get_relative_l2_dual_residual_value() const;

  void set_relative_dual_tolerance_factor(f_t dual_tolerance_factor);
  void set_relative_primal_tolerance_factor(f_t primal_tolerance_factor);
  f_t get_relative_dual_tolerance_factor() const;
  f_t get_relative_primal_tolerance_factor() const;

  struct view_t {
    i_t primal_size;
    i_t dual_size;

    f_t eps_ratio;

    f_t* l_inf_norm_primal_linear_objective;
    f_t* l_inf_norm_primal_right_hand_side;
    f_t* l2_norm_primal_linear_objective;
    f_t* l2_norm_primal_right_hand_side;

    f_t* primal_objective;
    f_t* dual_objective;
    f_t* l2_primal_residual;
    f_t* l2_dual_residual;

    f_t* relative_l_inf_primal_residual;
    f_t* relative_l_inf_dual_residual;

    f_t* gap;
    f_t* abs_objective;

    f_t* primal_residual;
    f_t* dual_residual;
    f_t* reduced_cost;
    f_t* bound_value;
  };  // struct view_t

  /**
   * @brief Gets the device-side view (with raw pointers), for ease of access
   *        inside cuda kernels
   */
  view_t view();

  // Light-weight adaptor for best primal so far
  struct primal_quality_adapter_t {
    bool is_primal_feasible{false};
    i_t nb_violated_constraints{std::numeric_limits<i_t>::max()};
    f_t primal_residual{std::numeric_limits<f_t>::infinity()};
    f_t primal_objective;  // Init in pdlp constructor since need to know sense of optimization

    bool operator==(const primal_quality_adapter_t& other) const
    {
      return is_primal_feasible == other.is_primal_feasible &&
             nb_violated_constraints == other.nb_violated_constraints &&
             primal_residual == other.primal_residual && primal_objective == other.primal_objective;
    }

    bool operator!=(const primal_quality_adapter_t& other) const { return !(*this == other); }
  };

  primal_quality_adapter_t to_primal_quality_adapter(bool is_primal_feasible) const noexcept;

  void compute_primal_residual(cusparse_view_t<i_t, f_t>& cusparse_view,
                               rmm::device_uvector<f_t>& tmp_dual,
                               [[maybe_unused]] const rmm::device_uvector<f_t>& dual_iterate);

 private:
  void compute_primal_objective(rmm::device_uvector<f_t>& primal_solution);

  void compute_dual_residual(cusparse_view_t<i_t, f_t>& cusparse_view,
                             rmm::device_uvector<f_t>& tmp_primal,
                             rmm::device_uvector<f_t>& primal_solution,
                             [[maybe_unused]] const rmm::device_uvector<f_t>& dual_slack);

  void compute_dual_objective(rmm::device_uvector<f_t>& dual_solution,
                              [[maybe_unused]] const rmm::device_uvector<f_t>& primal_solution,
                              [[maybe_unused]] const rmm::device_uvector<f_t>& dual_slack);

  void compute_reduced_cost_from_primal_gradient(const rmm::device_uvector<f_t>& primal_gradient,
                                                 const rmm::device_uvector<f_t>& primal_solution);

  void compute_reduced_costs_dual_objective_contribution();

  raft::handle_t const* handle_ptr_{nullptr};

  rmm::cuda_stream_view stream_view_;

  i_t primal_size_h_;
  i_t dual_size_h_;

  problem_t<i_t, f_t>* problem_ptr;
  cusparse_view_t<i_t, f_t>& op_problem_cusparse_view_;

  rmm::device_scalar<f_t> l2_norm_primal_linear_objective_;
  rmm::device_scalar<f_t> l2_norm_primal_right_hand_side_;

  rmm::device_scalar<f_t> primal_objective_;
  rmm::device_scalar<f_t> dual_objective_;
  rmm::device_scalar<f_t> reduced_cost_dual_objective_;
  rmm::device_scalar<f_t> l2_primal_residual_;
  rmm::device_scalar<f_t> l2_dual_residual_;
  // Useful in per constraint mode
  // To compute residual we check: residual[i] < absolute_tolerance + relative_tolerance * rhs[i]
  // Which can be rewritten as: residual[i] - relative_tolerance * rhs[i] < absolute_tolerance
  // We thus store l_inf(residual_i - rel * b/c_i) ran over all the constraints
  rmm::device_scalar<f_t> linf_primal_residual_;
  rmm::device_scalar<f_t> linf_dual_residual_;
  // Useful for best_primal_so_far
  rmm::device_scalar<i_t> nb_violated_constraints_;

  rmm::device_scalar<f_t> gap_;
  rmm::device_scalar<f_t> abs_objective_;

  // used for computations and can be reused
  rmm::device_uvector<f_t> primal_residual_;
  rmm::device_uvector<f_t> dual_residual_;
  rmm::device_uvector<f_t> reduced_cost_;
  rmm::device_uvector<f_t> bound_value_;

  // used for reflected
  rmm::device_uvector<f_t> primal_slack_;

  rmm::device_buffer rmm_tmp_buffer_;
  size_t size_of_buffer_;

  const rmm::device_scalar<f_t> reusable_device_scalar_value_1_;
  const rmm::device_scalar<f_t> reusable_device_scalar_value_0_;
  const rmm::device_scalar<f_t> reusable_device_scalar_value_neg_1_;

  rmm::device_scalar<f_t> dual_dot_;
  rmm::device_scalar<f_t> sum_primal_slack_;
};
}  // namespace cuopt::linear_programming::detail
