/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <pdlp/pdlp_constants.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

namespace cuopt::linear_programming::detail {

// Helper class to capture and launch CUDA graph
// No additional checks for safe usage (calling launch() before initializing the graph) use with
// caution Binary part is because in pdlp we swap pointers instead of copying vectors to accept a
// valid pdhg step So every odd pdlp step it's one graph, every even step it's another graph
template <typename i_t>
class ping_pong_graph_t {
 public:
  ping_pong_graph_t(rmm::cuda_stream_view stream_view, bool is_legacy_batch_mode = false);
  ~ping_pong_graph_t();

  void start_capture(i_t total_pdlp_iterations);
  void end_capture(i_t total_pdlp_iterations);
  void launch(i_t total_pdlp_iterations);
  bool is_initialized(i_t total_pdlp_iterations);

 private:
  cudaGraph_t even_graph;
  cudaGraph_t odd_graph;
  cudaGraphExec_t even_instance;
  cudaGraphExec_t odd_instance;
  rmm::cuda_stream_view stream_view_;
  bool even_initialized{false};
  bool odd_initialized{false};
  bool capture_even_active_{false};
  bool capture_odd_active_{false};
  bool is_legacy_batch_mode_{false};
};

}  // namespace cuopt::linear_programming::detail
