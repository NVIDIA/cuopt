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

#include <linear_programming/saddle_point.hpp>
#include <linear_programming/utilities/ping_pong_graph.cuh>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
class weighted_average_solution_t {
 public:
  weighted_average_solution_t(raft::handle_t const* handle_ptr,
                              i_t primal_size,
                              i_t dual_size,
                              bool is_batch_mode);

  void reset_weighted_average_solution();
  void add_current_solution_to_weighted_average_solution(const f_t* primal_solution,
                                                         const f_t* dual_solution,
                                                         const rmm::device_scalar<f_t>& weight,
                                                         i_t total_pdlp_iterations);

  void compute_averages(rmm::device_uvector<f_t>& avg_primal, rmm::device_uvector<f_t>& avg_dual);

  i_t get_iterations_since_last_restart() const;

 private:
  raft::handle_t const* handle_ptr_{nullptr};
  rmm::cuda_stream_view stream_view_;

  i_t primal_size_h_;
  i_t dual_size_h_;

 public:
  rmm::device_uvector<f_t> sum_primal_solutions_;
  rmm::device_uvector<f_t> sum_dual_solutions_;
  rmm::device_scalar<f_t> sum_primal_solution_weights_;
  rmm::device_scalar<f_t> sum_dual_solution_weights_;

  i_t iterations_since_last_restart_;

  // Graph to capture the average computation
  ping_pong_graph_t<i_t> graph;
};
}  // namespace cuopt::linear_programming::detail
