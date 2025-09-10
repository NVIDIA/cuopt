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
#include <linear_programming/cusparse_view.hpp>
#include <linear_programming/saddle_point.hpp>
#include <linear_programming/utilities/ping_pong_graph.cuh>
#include <mip/problem/problem.cuh>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
class pdhg_solver_t {
 public:
  pdhg_solver_t(raft::handle_t const* handle_ptr,
                problem_t<i_t, f_t>& op_problem,
                bool is_batch_mode = false);

  saddle_point_state_t<i_t, f_t>& get_saddle_point_state();
  cusparse_view_t<i_t, f_t>& get_cusparse_view();
  rmm::device_uvector<f_t>& get_primal_tmp_resource();
  rmm::device_uvector<f_t>& get_dual_tmp_resource();
  rmm::device_uvector<f_t>& get_potential_next_primal_solution();
  rmm::device_uvector<f_t>& get_dual_slack();
  const rmm::device_uvector<f_t>& get_potential_next_primal_solution() const;
  rmm::device_uvector<f_t>& get_potential_next_dual_solution();
  const rmm::device_uvector<f_t>& get_potential_next_dual_solution() const;
  const rmm::device_uvector<f_t>& get_reflected_dual() const;
  const rmm::device_uvector<f_t>& get_reflected_primal() const;
  i_t get_total_pdhg_iterations();
  rmm::device_scalar<i_t>& get_d_total_pdhg_iterations();
  rmm::device_uvector<f_t>& get_primal_solution();
  rmm::device_uvector<f_t>& get_dual_solution();

  void take_step(rmm::device_scalar<f_t>& primal_step_size,
                 rmm::device_scalar<f_t>& dual_step_size,
                 i_t iterations_since_last_restart,
                 bool last_restart_was_average,
                 i_t total_pdlp_iterations,
                 bool is_major_iteration);
  void update_solution(cusparse_view_t<i_t, f_t>& current_op_problem_evaluation_cusparse_view_);

  i_t total_pdhg_iterations_;

 private:
  void compute_next_primal_dual_solution(rmm::device_scalar<f_t>& primal_step_size,
                                         i_t iterations_since_last_restart,
                                         bool last_restart_was_average,
                                         rmm::device_scalar<f_t>& dual_step_size,
                                         i_t total_pdlp_iterations);
  void compute_next_dual_solution(rmm::device_scalar<f_t>& dual_step_size);
  void compute_next_primal_dual_solution_reflected(rmm::device_scalar<f_t>& primal_step_size,
                                                   rmm::device_scalar<f_t>& dual_step_size,
                                                   bool should_major);

  void compute_primal_projection_with_gradient(rmm::device_scalar<f_t>& primal_step_size);
  void compute_primal_projection(rmm::device_scalar<f_t>& primal_step_size);
  void compute_At_y();
  void compute_A_x();

  raft::handle_t const* handle_ptr_{nullptr};
  rmm::cuda_stream_view stream_view_;

  problem_t<i_t, f_t>* problem_ptr;

  i_t primal_size_h_;
  i_t dual_size_h_;

  rmm::device_uvector<f_t> tmp_primal_;
  rmm::device_uvector<f_t> tmp_dual_;

  saddle_point_state_t<i_t, f_t> current_saddle_point_state_;

  rmm::device_uvector<f_t> potential_next_primal_solution_;
  rmm::device_uvector<f_t> potential_next_dual_solution_;

  rmm::device_uvector<f_t> dual_slack_;
  rmm::device_uvector<f_t> reflected_primal_;
  rmm::device_uvector<f_t> reflected_dual_;

  // Important that vectors passed down to the cusparse_view are allocated before
  cusparse_view_t<i_t, f_t> cusparse_view_;

  const rmm::device_scalar<f_t> reusable_device_scalar_value_1_;
  const rmm::device_scalar<f_t> reusable_device_scalar_value_0_;
  const rmm::device_scalar<f_t> reusable_device_scalar_value_neg_1_;
  rmm::device_scalar<f_t> reusable_device_scalar_1_;

  // Different graphs for each case
  // Either compute the whole next primal step
  // Or skip the SpMV (most cases) if it was done at the previous iteration
  ping_pong_graph_t<i_t> graph_all;
  ping_pong_graph_t<i_t> graph_prim_proj_gradient_dual;

  // Needed for faster graph launch
  // Passing the host value each time would require updating the graph each time
  rmm::device_scalar<i_t> d_total_pdhg_iterations_;
};

}  // namespace cuopt::linear_programming::detail
