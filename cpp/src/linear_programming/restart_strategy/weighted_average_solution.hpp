/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
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
