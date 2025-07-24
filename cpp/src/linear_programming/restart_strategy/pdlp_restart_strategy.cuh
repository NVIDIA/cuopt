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
#include <linear_programming/restart_strategy/localized_duality_gap_container.hpp>
#include <linear_programming/restart_strategy/weighted_average_solution.hpp>
#include <linear_programming/saddle_point.hpp>
#include <linear_programming/termination_strategy/convergence_information.hpp>

#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>

#include <mip/problem/problem.cuh>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/core/device_span.hpp>

namespace cuopt::linear_programming::detail {
void set_restart_hyper_parameters(rmm::cuda_stream_view stream_view);
template <typename i_t, typename f_t>
class pdlp_restart_strategy_t {
 public:
  /**
   * @brief A device-side view of the `pdlp_restart_strategy_t` structure with the RAII stuffs
   *        stripped out, to make it easy to work inside kernels
   *
   * @note It is assumed that the pointers are NOT owned by this class, but rather
   *       by the encompassing `pdlp_restart_strategy_t` class via RAII abstractions like
   *       `rmm::device_uvector`
   */
  struct view_t {
    i_t primal_size;
    i_t dual_size;
    i_t size_of_saddle_point_problem;
    i_t n_blocks_needed;
    i_t n_threads_needed_full_blocks;
    i_t n_potential_items_in_last_block;

    i_t primal_aligned_size;
    i_t dual_aligned_size;

    i_t last_restart_length;

    raft::device_span<f_t> weights;

    i_t* candidate_is_avg;
    i_t* restart_triggered;

    f_t* gap_reduction_ratio_last_trial;

    raft::device_span<f_t> center_point;
    raft::device_span<f_t> objective_vector;
    raft::device_span<f_t> direction_full;
    raft::device_span<f_t> threshold;
    raft::device_span<f_t> lower_bound;
    raft::device_span<f_t> upper_bound;
    raft::device_span<f_t> test_point;
    raft::device_span<f_t> transformed_constraint_lower_bounds;
    raft::device_span<f_t> transformed_constraint_upper_bounds;

    f_t* target_threshold;
    f_t* low_radius_squared;
    f_t* high_radius_squared;
    f_t* test_threshold;
    f_t* test_radius_squared;

    i_t* testing_range_low;
    i_t* testing_range_high;

    raft::device_span<f_t> shared_live_kernel_accumulator;
  };

  enum class restart_strategy_t {
    NO_RESTART           = 0,
    KKT_RESTART          = 1,
    TRUST_REGION_RESTART = 2,
  };

  pdlp_restart_strategy_t(raft::handle_t const* handle_ptr,
                          problem_t<i_t, f_t>& op_problem,
                          const cusparse_view_t<i_t, f_t>& cusparse_view,
                          const i_t primal_size,
                          const i_t dual_size,
                          bool is_batch_mode = false);

  // Compute kkt score on passed argument using the container tmp_kkt score and stream view
  f_t compute_kkt_score(const rmm::device_scalar<f_t>& l2_primal_residual,
                        const rmm::device_scalar<f_t>& l2_dual_residual,
                        const rmm::device_scalar<f_t>& gap,
                        const rmm::device_scalar<f_t>& primal_weight);

  void update_distance(pdhg_solver_t<i_t, f_t>& pdhg_solver,
                       rmm::device_scalar<f_t>& primal_weight,
                       rmm::device_scalar<f_t>& primal_step_size,
                       rmm::device_scalar<f_t>& dual_step_size,
                       const rmm::device_scalar<f_t>& step_size);

  void add_current_solution_to_average_solution(const f_t* primal_solution,
                                                const f_t* dual_solution,
                                                const rmm::device_scalar<f_t>& weight,
                                                i_t total_pdlp_iterations);

  void get_average_solutions(rmm::device_uvector<f_t>& avg_primal,
                             rmm::device_uvector<f_t>& avg_dual);

  void compute_restart(pdhg_solver_t<i_t, f_t>& pdhg_solver,
                       rmm::device_uvector<f_t>& primal_solution_avg,
                       rmm::device_uvector<f_t>& dual_solution_avg,
                       const i_t total_number_of_iterations,
                       rmm::device_scalar<f_t>& primal_step_size,  // Updated if new primal weight
                       rmm::device_scalar<f_t>& dual_step_size,    // Updated if new primal weight
                       rmm::device_scalar<f_t>& primal_weight,
                       const rmm::device_scalar<f_t>& step_size,  // To update primal/dual step size
                       const convergence_information_t<i_t, f_t>& current_convergence_information,
                       const convergence_information_t<i_t, f_t>& average_convergence_information);

  /**
   * @brief Gets the device-side view (with raw pointers), for ease of access
   *        inside cuda kernels
   */
  view_t view();

  i_t get_iterations_since_last_restart() const;

  void set_last_restart_was_average(bool value);
  bool get_last_restart_was_average() const;

  i_t should_do_artificial_restart(i_t total_number_of_iterations) const;

 private:
  void run_trust_region_restart(pdhg_solver_t<i_t, f_t>& pdhg_solver,
                                rmm::device_uvector<f_t>& primal_solution_avg,
                                rmm::device_uvector<f_t>& dual_solution_avg,
                                const i_t total_number_of_iterations,
                                rmm::device_scalar<f_t>& primal_step_size,
                                rmm::device_scalar<f_t>& dual_step_size,
                                rmm::device_scalar<f_t>& primal_weight,
                                const rmm::device_scalar<f_t>& step_size);
  bool run_kkt_restart(pdhg_solver_t<i_t, f_t>& pdhg_solver,
                       rmm::device_uvector<f_t>& primal_solution_avg,
                       rmm::device_uvector<f_t>& dual_solution_avg,
                       const convergence_information_t<i_t, f_t>& current_convergence_information,
                       const convergence_information_t<i_t, f_t>& average_convergence_information,
                       rmm::device_scalar<f_t>& primal_step_size,
                       rmm::device_scalar<f_t>& dual_step_size,
                       rmm::device_scalar<f_t>& primal_weight,
                       const rmm::device_scalar<f_t>& step_size,
                       i_t total_number_of_iterations);
  bool kkt_restart_conditions(f_t candidate_kkt_score, i_t total_number_of_iterations);
  bool kkt_decay(f_t candidate_kkt_score);
  void compute_localized_duality_gaps(saddle_point_state_t<i_t, f_t>& current_saddle_point_state,
                                      rmm::device_uvector<f_t>& primal_solution_avg,
                                      rmm::device_uvector<f_t>& dual_solution_avg,
                                      rmm::device_scalar<f_t>& primal_weight,
                                      rmm::device_uvector<f_t>& tmp_primal,
                                      rmm::device_uvector<f_t>& tmp_dual);

  void distance_squared_moved_from_last_restart_period(const rmm::device_uvector<f_t>& new_solution,
                                                       const rmm::device_uvector<f_t>& old_solution,
                                                       rmm::device_uvector<f_t>& tmp,
                                                       i_t size_of_solutions_h,
                                                       i_t stride,
                                                       rmm::device_scalar<f_t>& distance_moved);

  void compute_primal_gradient(localized_duality_gap_container_t<i_t, f_t>& duality_gap,
                               cusparse_view_t<i_t, f_t>& cusparse_view);

  void compute_dual_gradient(localized_duality_gap_container_t<i_t, f_t>& duality_gap,
                             cusparse_view_t<i_t, f_t>& cusparse_view,
                             rmm::device_uvector<f_t>& tmp_dual);

  void compute_lagrangian_value(localized_duality_gap_container_t<i_t, f_t>& duality_gap,
                                cusparse_view_t<i_t, f_t>& cusparse_view,
                                rmm::device_uvector<f_t>& tmp_primal,
                                rmm::device_uvector<f_t>& tmp_dual);

  i_t pick_restart_candidate();

  void should_do_adaptive_restart_normalized_duality_gap(
    localized_duality_gap_container_t<i_t, f_t>& candidate_duality_gap,
    rmm::device_uvector<f_t>& tmp_primal,
    rmm::device_uvector<f_t>& tmp_dual,
    rmm::device_scalar<f_t>& primal_weight,
    i_t& restart);

  void bound_optimal_objective(cusparse_view_t<i_t, f_t>& existing_cusparse_view,
                               localized_duality_gap_container_t<i_t, f_t>& duality_gap,
                               rmm::device_uvector<f_t>& tmp_primal,
                               rmm::device_uvector<f_t>& tmp_dual);

  void compute_bound(const rmm::device_uvector<f_t>& solution_tr,
                     const rmm::device_uvector<f_t>& solution,
                     const rmm::device_uvector<f_t>& gradient,
                     const rmm::device_scalar<f_t>& lagrangian,
                     const i_t size,
                     const i_t stride,
                     rmm::device_uvector<f_t>& tmp,
                     rmm::device_scalar<f_t>& bound);

  /*
   * Updates from last restart the three distances:
   * - duality_gap.primal_distance_traveled
   * - duality_gap.dual_distance_traveled
   * - duality_gap.distance_traveled
   */
  void compute_distance_traveled_from_last_restart(
    localized_duality_gap_container_t<i_t, f_t>& duality_gap,
    rmm::device_scalar<f_t>& primal_weight,
    rmm::device_uvector<f_t>& tmp_primal,
    rmm::device_uvector<f_t>& tmp_dual);

  void solve_bound_constrained_trust_region(
    localized_duality_gap_container_t<i_t, f_t>& duality_gap,
    rmm::device_uvector<f_t>& tmp_primal,
    rmm::device_uvector<f_t>& tmp_dual);

  void update_last_restart_information(localized_duality_gap_container_t<i_t, f_t>& duality_gap,
                                       rmm::device_scalar<f_t>& primal_weight);

  void reset_internal();

  void compute_new_primal_weight(localized_duality_gap_container_t<i_t, f_t>& duality_gap,
                                 rmm::device_scalar<f_t>& primal_weight,
                                 const rmm::device_scalar<f_t>& step_size,
                                 rmm::device_scalar<f_t>& primal_step_size,
                                 rmm::device_scalar<f_t>& dual_step_size);

  raft::handle_t const* handle_ptr_{nullptr};
  rmm::cuda_stream_view stream_view_;

 public:
  weighted_average_solution_t<i_t, f_t> weighted_average_solution_;

  i_t primal_size_h_;
  i_t dual_size_h_;

  problem_t<i_t, f_t>* problem_ptr;

  rmm::device_scalar<i_t> primal_size_;
  rmm::device_scalar<i_t> dual_size_;

  rmm::device_scalar<f_t> primal_norm_weight_;
  rmm::device_scalar<f_t> dual_norm_weight_;
  // 1D vector of size primal + dual size, combining primal_norm_weight on the first part and
  // dual_norm_weight on the second part It's mandatory as trust_bound_region is done on a sorted
  // merge problem in which primal and dual are mixed together It is then impossible to
  // differenciate primal from dual and thus choose the correct weight
  rmm::device_uvector<f_t> weights_;

  rmm::device_scalar<i_t> candidate_is_avg_;
  rmm::device_scalar<i_t> restart_triggered_;

  localized_duality_gap_container_t<i_t, f_t> avg_duality_gap_;
  localized_duality_gap_container_t<i_t, f_t> current_duality_gap_;
  localized_duality_gap_container_t<i_t, f_t> last_restart_duality_gap_;
  localized_duality_gap_container_t<i_t, f_t>* candidate_duality_gap_;

  cusparse_view_t<i_t, f_t> avg_duality_gap_cusparse_view_;
  cusparse_view_t<i_t, f_t> current_duality_gap_cusparse_view_;
  cusparse_view_t<i_t, f_t> last_restart_duality_gap_cusparse_view_;

  rmm::device_scalar<f_t> gap_reduction_ratio_last_trial_;
  i_t last_restart_length_;

  // All mainly used in bound_objective
  // {
  rmm::device_uvector<f_t> center_point_;
  rmm::device_uvector<f_t> objective_vector_;
  // direction_full_ is sorted following threshold to ease test_radius computation
  // But an unsorted direction is necessary to compute solution_tr_
  rmm::device_uvector<f_t> unsorted_direction_full_;
  rmm::device_uvector<f_t> direction_full_;
  rmm::device_uvector<f_t> threshold_;
  rmm::device_uvector<f_t> lower_bound_;
  rmm::device_uvector<f_t> upper_bound_;
  rmm::device_uvector<f_t> test_point_;
  rmm::device_uvector<f_t> transformed_constraint_lower_bounds_;
  rmm::device_uvector<f_t> transformed_constraint_upper_bounds_;
  rmm::device_uvector<f_t> shared_live_kernel_accumulator_;

  rmm::device_scalar<f_t> target_threshold_;
  rmm::device_scalar<f_t> low_radius_squared_;
  rmm::device_scalar<f_t> high_radius_squared_;
  rmm::device_scalar<f_t> test_threshold_;
  rmm::device_scalar<f_t> test_radius_squared_;

  rmm::device_scalar<i_t> testing_range_low_;
  rmm::device_scalar<i_t> testing_range_high_;
  //}

  const rmm::device_scalar<f_t> reusable_device_scalar_value_1_;
  const rmm::device_scalar<f_t> reusable_device_scalar_value_0_;
  const rmm::device_scalar<i_t> reusable_device_scalar_value_0_i_t_;
  const rmm::device_scalar<f_t> reusable_device_scalar_value_neg_1_;
  // Used to store temporarily on the device the kkt scores before host retrival
  rmm::device_scalar<f_t> tmp_kkt_score_;
  rmm::device_scalar<f_t> reusable_device_scalar_1_;
  rmm::device_scalar<f_t> reusable_device_scalar_2_;
  rmm::device_scalar<f_t> reusable_device_scalar_3_;

  f_t last_candidate_kkt_score = f_t(0.0);
  f_t last_restart_kkt_score   = f_t(0.0);

  bool last_restart_was_average_ = false;
};

template <typename i_t, typename f_t>
bool is_KKT_restart()
{
  return pdlp_hyper_params::restart_strategy ==
         static_cast<int>(pdlp_restart_strategy_t<i_t, f_t>::restart_strategy_t::KKT_RESTART);
}

}  // namespace cuopt::linear_programming::detail
