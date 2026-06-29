/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <pdlp/pdlp_constants.hpp>
#include <utilities/manual_cuda_graph.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <utility>

namespace cuopt::mathematical_optimization::pdlp {

// Two-slot CUDA-graph cache for PDLP. PDLP swaps pointers (rather than
// copying vectors) at the end of adaptive pdhg step, so the captured graph
// topology alternates between two layouts depending on iteration parity.
// Each slot is a manual_cuda_graph_t, which (a) builds the parent graph
// explicitly via cudaGraphCreate + cudaStreamBeginCaptureToGraph and
// (b) recovers from cudaErrorStreamCaptureInvalidated by re-executing the
// supplied work eagerly. See manual_cuda_graph.cuh for details.
template <typename i_t>
class ping_pong_graph_t {
 public:
  ping_pong_graph_t(rmm::cuda_stream_view stream_view, bool is_legacy_batch_mode = false);
  ~ping_pong_graph_t() = default;

  // Non-copyable because the underlying manual_cuda_graph_t owns a
  // cudaGraphExec_t handle. Move-assignment is needed by pdlp.cu, which
  // re-binds the existing slot to a freshly-constructed legacy-batch-mode
  // instance after an SpMM run.
  ping_pong_graph_t(const ping_pong_graph_t&)                = delete;
  ping_pong_graph_t& operator=(const ping_pong_graph_t&)     = delete;
  ping_pong_graph_t(ping_pong_graph_t&&) noexcept            = default;
  ping_pong_graph_t& operator=(ping_pong_graph_t&&) noexcept = default;

  // Either launch the cached graph for this parity slot, or capture `work`
  // into a freshly-created parent graph, instantiate, and launch. Capture
  // invalidation is recovered by re-running `work` eagerly (see
  // manual_cuda_graph_t::run). In CUPDLP_DEBUG_MODE or legacy-batch mode
  // the work is always executed eagerly with no graph involvement.
  template <typename F>
  void run(i_t total_pdlp_iterations, F&& work)
  {
#ifdef CUPDLP_DEBUG_MODE
    work();
#else
    if (is_legacy_batch_mode_) {
      work();
      return;
    }
    if (total_pdlp_iterations % 2 == 0) {
      even_graph_.run(stream_view_, std::forward<F>(work));
    } else {
      odd_graph_.run(stream_view_, std::forward<F>(work));
    }
#endif
  }

 private:
  manual_cuda_graph_t even_graph_;
  manual_cuda_graph_t odd_graph_;
  rmm::cuda_stream_view stream_view_;
  bool is_legacy_batch_mode_{false};
};

}  // namespace cuopt::mathematical_optimization::pdlp
