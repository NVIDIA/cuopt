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

#include <utilities/event_handler.cuh>
#include <utilities/unique_pinned_ptr.hpp>

#include <linear_programming/cusparse_view.hpp>
#include <linear_programming/pdhg.hpp>
#include <linear_programming/saddle_point.hpp>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {
void set_adaptive_step_size_hyper_parameters(rmm::cuda_stream_view stream_view);
template <typename i_t, typename f_t>
class adaptive_step_size_strategy_t {
 public:
  /**
   * @brief A device-side view of the `adaptive_step_size_strategy_t` structure with the RAII stuffs
   *        stripped out, to make it easy to work inside kernels
   *
   * @note It is assumed that the pointers are NOT owned by this class, but rather
   *       by the encompassing `adaptive_step_size_strategy_t` class via RAII abstractions like
   *       `rmm::device_uvector`
   */
  struct view_t {
    f_t* primal_weight;
    f_t* step_size;
    i_t* valid_step_size;

    f_t* interaction;
    f_t* movement;

    f_t* norm_squared_delta_primal;
    f_t* norm_squared_delta_dual;
  };

  adaptive_step_size_strategy_t(raft::handle_t const* handle_ptr,
                                rmm::device_scalar<f_t>* primal_weight,
                                rmm::device_scalar<f_t>* step_size);

  void compute_step_sizes(pdhg_solver_t<i_t, f_t>& pdhg_solver,
                          rmm::device_scalar<f_t>& primal_step_size,
                          rmm::device_scalar<f_t>& dual_step_size,
                          i_t total_pdlp_iterations);

  void get_primal_and_dual_stepsizes(rmm::device_scalar<f_t>& primal_step_size,
                                     rmm::device_scalar<f_t>& dual_step_size);
  /**
   * @brief Gets the device-side view (with raw pointers), for ease of access
   *        inside cuda kernels
   */
  view_t view();

  i_t get_valid_step_size() const;
  void set_valid_step_size(i_t);

 private:
  void compute_interaction_and_movement(rmm::device_uvector<f_t>& tmp_primal,
                                        cusparse_view_t<i_t, f_t>& cusparse_view,
                                        saddle_point_state_t<i_t, f_t>& current_saddle_point_state);

  // Stream pool to run different step size computation in parallel
  // Because we already have the main stream, we just need 2 extra streams from this
  rmm::cuda_stream_pool stream_pool_;

  // Events to record when dot product of both delta_x and y are done and when to start them
  event_handler_t deltas_are_done_;
  event_handler_t dot_delta_X_;
  event_handler_t dot_delta_Y_;

  raft::handle_t const* handle_ptr_{nullptr};
  rmm::cuda_stream_view stream_view_;

  rmm::device_scalar<f_t>* primal_weight_;
  rmm::device_scalar<f_t>* step_size_;
  // Host pinned memory scalar written in kernel
  // Combines both numerical_issue and valid_step size and save the device/host memcpy
  // -1: Error ; 0: Invalid step size ; 1: Valid step size
  // TODO: Replace with thrust::universal_host_pinned_vector once the bug is fixed:
  // https://github.com/NVIDIA/cccl/issues/5027
  std::unique_ptr<i_t, cuda_host_deleter<i_t>> valid_step_size_;

  rmm::device_scalar<f_t> interaction_;
  rmm::device_scalar<f_t> movement_;

  rmm::device_scalar<f_t> norm_squared_delta_primal_;
  rmm::device_scalar<f_t> norm_squared_delta_dual_;

  const rmm::device_scalar<f_t> reusable_device_scalar_value_1_;
  const rmm::device_scalar<f_t> reusable_device_scalar_value_0_;

  ping_pong_graph_t<i_t> graph;
};
}  // namespace cuopt::linear_programming::detail
